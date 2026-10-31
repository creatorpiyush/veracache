//! Exact and semantic lookup, with fake components.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde_json::Value;
use veracache::{
    Cache, CacheConfig, CacheEntry, CacheRequest, Candidate, ChatMessage, ContextMode,
    DeleteFilter, Embedder, EnglishAnalyzer, EntryId, Error, Feedback, Namespace, ResponseMeta,
    Result, SetOptions, VectorStore, default_guards, memory_cache,
};

/// Maps known prompts to fixed vectors.
struct FakeEmbedder {
    vectors: HashMap<&'static str, Vec<f32>>,
    failing: AtomicBool,
}

impl FakeEmbedder {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            vectors: HashMap::from([
                ("what is the capital of France?", vec![1.0, 0.0]),
                ("tell me the capital of France", vec![3.0, 0.3]),
                ("how do I bake bread?", vec![0.0, 1.0]),
                ("zero vector", vec![0.0, 0.0]),
            ]),
            failing: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl Embedder for FakeEmbedder {
    fn id(&self) -> &str {
        "fake"
    }

    fn dim(&self) -> usize {
        2
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(Error::Embed("down".into()));
        }
        texts
            .iter()
            .map(|text| {
                self.vectors
                    .get(text.as_str())
                    .cloned()
                    .ok_or_else(|| Error::Embed("unknown text".into()))
            })
            .collect()
    }
}

/// Fails every call.
struct FailingStore;

#[async_trait]
impl VectorStore for FailingStore {
    async fn upsert(&self, _: CacheEntry) -> Result<()> {
        Err(Error::Store("down".into()))
    }

    async fn search(&self, _: &Namespace, _: &[f32], _: usize) -> Result<Vec<Candidate>> {
        Err(Error::Store("down".into()))
    }

    async fn get_exact(&self, _: &Namespace, _: &[u8; 32]) -> Result<Option<CacheEntry>> {
        Err(Error::Store("down".into()))
    }

    async fn delete(&self, _: DeleteFilter) -> Result<u64> {
        Err(Error::Store("down".into()))
    }

    async fn record_feedback(&self, _: EntryId, _: Feedback) -> Result<()> {
        Err(Error::Store("down".into()))
    }
}

fn user(content: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: content.into(),
    }]
}

fn request<'a>(tenant: &'a str, messages: &'a [ChatMessage]) -> CacheRequest<'a> {
    CacheRequest {
        tenant,
        model: "model",
        messages,
        tools: &Value::Null,
        response_format: &Value::Null,
        version_tag: "",
    }
}

async fn set(cache: &Cache, request: &CacheRequest<'_>, response: &[u8]) {
    cache
        .set(
            request,
            response.to_vec(),
            ResponseMeta::default(),
            SetOptions::default(),
        )
        .await;
}

const FRANCE: &str = "what is the capital of France?";
const PARAPHRASE: &str = "tell me the capital of France";
const BREAD: &str = "how do I bake bread?";

#[tokio::test]
async fn repeated_prompt_is_an_exact_hit() {
    let cache = memory_cache(FakeEmbedder::new(), CacheConfig::default());
    let messages = user(FRANCE);
    let req = request("a", &messages);

    assert_eq!(cache.get(&req).await, None);
    set(&cache, &req, b"Paris").await;

    let spaced = user("what is  the capital\nof France?");
    let hit = cache.get(&request("a", &spaced)).await.unwrap();
    assert!(hit.exact);
    assert_eq!(hit.entry.response, b"Paris");
}

#[tokio::test]
async fn paraphrase_is_a_semantic_hit() {
    let cache = memory_cache(FakeEmbedder::new(), CacheConfig::default());
    let messages = user(FRANCE);
    set(&cache, &request("a", &messages), b"Paris").await;

    let paraphrase = user(PARAPHRASE);
    let hit = cache.get(&request("a", &paraphrase)).await.unwrap();
    assert!(!hit.exact);
    assert!(hit.score > 0.99 && hit.score <= 1.0, "{}", hit.score);
    assert_eq!(hit.entry.response, b"Paris");
}

#[tokio::test]
async fn below_threshold_is_a_miss() {
    let config = CacheConfig {
        threshold: 0.999,
        ..CacheConfig::default()
    };
    let cache = memory_cache(FakeEmbedder::new(), config);
    let messages = user(FRANCE);
    set(&cache, &request("a", &messages), b"Paris").await;

    let paraphrase = user(PARAPHRASE);
    assert_eq!(cache.get(&request("a", &paraphrase)).await, None);

    let unrelated = user(BREAD);
    assert_eq!(cache.get(&request("a", &unrelated)).await, None);
}

#[tokio::test]
async fn best_candidate_wins() {
    let cache = memory_cache(FakeEmbedder::new(), CacheConfig::default());
    let (bread, france) = (user(BREAD), user(FRANCE));
    set(&cache, &request("a", &bread), b"knead").await;
    set(&cache, &request("a", &france), b"Paris").await;

    let paraphrase = user(PARAPHRASE);
    let hit = cache.get(&request("a", &paraphrase)).await.unwrap();
    assert_eq!(hit.entry.response, b"Paris");
}

#[tokio::test]
async fn no_hit_crosses_a_namespace() {
    let cache = memory_cache(FakeEmbedder::new(), CacheConfig::default());
    let messages = user(FRANCE);
    set(&cache, &request("a", &messages), b"Paris").await;

    assert_eq!(cache.get(&request("b", &messages)).await, None);

    let other_model = CacheRequest {
        model: "other",
        ..request("a", &messages)
    };
    assert_eq!(cache.get(&other_model).await, None);

    let mut with_system = user(FRANCE);
    with_system.insert(
        0,
        ChatMessage {
            role: "system".into(),
            content: "answer in French".into(),
        },
    );
    assert_eq!(cache.get(&request("a", &with_system)).await, None);
}

#[tokio::test]
async fn follow_up_is_not_cached_by_default() {
    let cache = memory_cache(FakeEmbedder::new(), CacheConfig::default());
    let standalone = user(FRANCE);
    set(&cache, &request("a", &standalone), b"Paris").await;

    let mut follow_up = user(BREAD);
    follow_up.push(ChatMessage {
        role: "assistant".into(),
        content: "Knead it.".into(),
    });
    follow_up.extend(user(FRANCE));
    assert_eq!(cache.get(&request("a", &follow_up)).await, None);

    set(&cache, &request("a", &follow_up), b"stored").await;
    assert_eq!(cache.get(&request("a", &follow_up)).await, None);
    assert_eq!(cache.error_count(), 0);
}

#[tokio::test]
async fn follow_up_matches_only_its_own_history() {
    let config = CacheConfig {
        context_mode: ContextMode::LastNTurns(4),
        ..CacheConfig::default()
    };
    let cache = memory_cache(FakeEmbedder::new(), config);
    let mut follow_up = user(BREAD);
    follow_up.push(ChatMessage {
        role: "assistant".into(),
        content: "Knead it.".into(),
    });
    follow_up.extend(user(FRANCE));
    set(&cache, &request("a", &follow_up), b"Paris").await;

    assert!(cache.get(&request("a", &follow_up)).await.is_some());
    let standalone = user(FRANCE);
    assert_eq!(cache.get(&request("a", &standalone)).await, None);
}

#[tokio::test]
async fn embedder_failure_is_a_counted_miss() {
    let embedder = FakeEmbedder::new();
    let cache = memory_cache(embedder.clone(), CacheConfig::default());
    let messages = user(FRANCE);
    set(&cache, &request("a", &messages), b"Paris").await;

    embedder.failing.store(true, Ordering::SeqCst);
    let paraphrase = user(PARAPHRASE);
    assert_eq!(cache.get(&request("a", &paraphrase)).await, None);
    assert_eq!(cache.error_count(), 1);

    assert!(cache.get(&request("a", &messages)).await.is_some());
    set(&cache, &request("a", &paraphrase), b"Paris").await;
    assert_eq!(cache.error_count(), 2);
}

#[tokio::test]
async fn store_failure_is_a_counted_miss() {
    let cache = Cache::new(
        FakeEmbedder::new(),
        Arc::new(FailingStore),
        Arc::new(EnglishAnalyzer),
        default_guards(),
        CacheConfig::default(),
    );
    let messages = user(FRANCE);
    let req = request("a", &messages);

    assert_eq!(cache.get(&req).await, None);
    set(&cache, &req, b"Paris").await;
    assert_eq!(cache.error_count(), 2);
}

#[tokio::test]
async fn zero_vector_is_a_counted_miss() {
    let cache = memory_cache(FakeEmbedder::new(), CacheConfig::default());
    let messages = user("zero vector");
    assert_eq!(cache.get(&request("a", &messages)).await, None);
    assert_eq!(cache.error_count(), 1);
}
