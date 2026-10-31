use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde_json::Value;

use crate::context::is_system;
use crate::{
    CacheEntry, ChatMessage, ContextMode, Embedder, EntryId, EntryStats, Error, Namespace,
    NamespaceParts, ResponseMeta, Result, VectorStore, context_fingerprint, key_hash, normalize,
};

/// Lookup settings.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Minimum similarity for a semantic hit.
    pub threshold: f32,
    /// Candidates fetched from vector search.
    pub k: usize,
    /// How prior conversation turns affect caching.
    pub context_mode: ContextMode,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            threshold: 0.9,
            k: 5,
            context_mode: ContextMode::default(),
        }
    }
}

/// What identifies a request to the cache.
#[derive(Debug, Clone, Copy)]
pub struct CacheRequest<'a> {
    /// Tenant or user scope.
    pub tenant: &'a str,
    /// Model name.
    pub model: &'a str,
    /// Conversation, ending in the user message.
    pub messages: &'a [ChatMessage],
    /// Tools or function schema. `Value::Null` when absent.
    pub tools: &'a Value,
    /// Response format. `Value::Null` when absent.
    pub response_format: &'a Value,
    /// User-supplied cache version tag.
    pub version_tag: &'a str,
}

/// Optional settings for a stored response.
#[derive(Debug, Clone, Default)]
pub struct SetOptions {
    /// Time to live. `None` means no expiry.
    pub ttl: Option<Duration>,
    /// Invalidation tags.
    pub tags: Vec<String>,
    /// Invalidation version.
    pub version: Option<String>,
}

/// A cached response accepted for a request.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// The stored entry.
    pub entry: CacheEntry,
    /// Similarity to the query. 1.0 for an exact hit.
    pub score: f32,
    /// Whether the hit came from the exact path.
    pub exact: bool,
}

struct Key {
    namespace: Namespace,
    key_hash: [u8; 32],
    normalized: String,
}

/// The lookup pipeline. A cache failure is a miss, never a failed request.
pub struct Cache {
    embedder: Arc<dyn Embedder>,
    store: Arc<dyn VectorStore>,
    config: CacheConfig,
    errors: AtomicU64,
}

impl Cache {
    /// Creates a cache from its parts.
    pub fn new(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        config: CacheConfig,
    ) -> Self {
        Self {
            embedder,
            store,
            config,
            errors: AtomicU64::new(0),
        }
    }

    /// Looks up a request. `None` means call upstream.
    pub async fn get(&self, request: &CacheRequest<'_>) -> Option<Hit> {
        let key = self.key(request)?;
        match self.lookup(&key).await {
            Ok(hit) => hit,
            Err(_) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Stores an upstream response. Uncacheable conversations are skipped.
    pub async fn set(
        &self,
        request: &CacheRequest<'_>,
        response: Vec<u8>,
        response_meta: ResponseMeta,
        options: SetOptions,
    ) {
        let Some(key) = self.key(request) else {
            return;
        };
        if self
            .store_entry(key, response, response_meta, options)
            .await
            .is_err()
        {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Counts failures that were turned into misses or skipped stores.
    pub fn error_count(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    fn key(&self, request: &CacheRequest<'_>) -> Option<Key> {
        let context = context_fingerprint(request.messages, self.config.context_mode)?;
        let prompt = request.messages.iter().rev().find(|m| !is_system(m))?;
        let system_prompt = request
            .messages
            .iter()
            .filter(|m| is_system(m))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let namespace = NamespaceParts {
            tenant: request.tenant,
            model: request.model,
            embedder_id: self.embedder.id(),
            system_prompt: &system_prompt,
            tools: request.tools,
            response_format: request.response_format,
            context_fingerprint: &context,
            version_tag: request.version_tag,
        }
        .build();
        let normalized = normalize(&prompt.content);
        Some(Key {
            namespace,
            key_hash: key_hash(&normalized),
            normalized,
        })
    }

    async fn lookup(&self, key: &Key) -> Result<Option<Hit>> {
        if let Some(entry) = self.store.get_exact(&key.namespace, &key.key_hash).await? {
            return Ok(Some(Hit {
                entry,
                score: 1.0,
                exact: true,
            }));
        }

        let vector = self.embed(&key.normalized).await?;
        let candidates = self
            .store
            .search(&key.namespace, &vector, self.config.k)
            .await?;
        Ok(candidates
            .into_iter()
            .find(|candidate| candidate.score >= self.config.threshold)
            .map(|candidate| Hit {
                entry: candidate.entry,
                score: candidate.score,
                exact: false,
            }))
    }

    async fn store_entry(
        &self,
        key: Key,
        response: Vec<u8>,
        response_meta: ResponseMeta,
        options: SetOptions,
    ) -> Result<()> {
        let embedding = self.embed(&key.normalized).await?;
        self.store
            .upsert(CacheEntry {
                id: entry_id(&key),
                namespace: key.namespace,
                key_hash: key.key_hash,
                prompt_text: Some(key.normalized),
                analyzed: None,
                embedding,
                response,
                response_meta,
                created_at: SystemTime::now(),
                ttl: options.ttl,
                tags: options.tags,
                version: options.version,
                stats: EntryStats::default(),
                threshold_override: None,
            })
            .await
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let vectors = self.embedder.embed(&[text.to_owned()]).await?;
        match <[Vec<f32>; 1]>::try_from(vectors) {
            Ok([vector]) if vector.len() == self.embedder.dim() => unit_vector(vector),
            _ => Err(Error::Embed("unexpected embedding shape".into())),
        }
    }
}

// Stored vectors are normalized so that dot product is cosine similarity.
fn unit_vector(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_normal() {
        return Err(Error::Embed("vector cannot be normalized".into()));
    }
    vector.iter_mut().for_each(|x| *x /= norm);
    Ok(vector)
}

fn entry_id(key: &Key) -> EntryId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&key.namespace.0);
    hasher.update(&key.key_hash);
    let mut id = [0; 16];
    id.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    EntryId(u128::from_le_bytes(id))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;
    use crate::{Candidate, DeleteFilter, Feedback};

    #[derive(Default)]
    struct FakeEmbedder {
        calls: AtomicU64,
        reply: Mutex<Option<Vec<Vec<f32>>>>,
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
            self.calls.fetch_add(1, Ordering::SeqCst);
            let reply = self.reply.lock().unwrap().clone();
            Ok(reply.unwrap_or_else(|| vec![vec![3.0, 4.0]; texts.len()]))
        }
    }

    /// Search returns every entry in the namespace with a scripted score.
    #[derive(Default)]
    struct FakeStore {
        entries: Mutex<Vec<CacheEntry>>,
        score: Mutex<f32>,
        last_k: Mutex<usize>,
    }

    impl FakeStore {
        fn only_entry(&self) -> CacheEntry {
            let entries = self.entries.lock().unwrap();
            assert_eq!(entries.len(), 1);
            entries[0].clone()
        }

        fn last_entry(&self) -> CacheEntry {
            self.entries.lock().unwrap().last().unwrap().clone()
        }
    }

    #[async_trait]
    impl VectorStore for FakeStore {
        async fn upsert(&self, entry: CacheEntry) -> Result<()> {
            self.entries.lock().unwrap().push(entry);
            Ok(())
        }

        async fn search(&self, ns: &Namespace, _: &[f32], k: usize) -> Result<Vec<Candidate>> {
            *self.last_k.lock().unwrap() = k;
            let score = *self.score.lock().unwrap();
            let entries = self.entries.lock().unwrap();
            Ok(entries
                .iter()
                .filter(|entry| entry.namespace == *ns)
                .map(|entry| Candidate {
                    entry: entry.clone(),
                    score,
                })
                .collect())
        }

        async fn get_exact(&self, ns: &Namespace, key: &[u8; 32]) -> Result<Option<CacheEntry>> {
            let entries = self.entries.lock().unwrap();
            Ok(entries
                .iter()
                .find(|entry| entry.namespace == *ns && entry.key_hash == *key)
                .cloned())
        }

        async fn delete(&self, _: DeleteFilter) -> Result<u64> {
            unimplemented!()
        }

        async fn record_feedback(&self, _: EntryId, _: Feedback) -> Result<()> {
            unimplemented!()
        }
    }

    struct Harness {
        cache: Cache,
        embedder: Arc<FakeEmbedder>,
        store: Arc<FakeStore>,
    }

    impl Harness {
        fn new(config: CacheConfig) -> Self {
            let embedder = Arc::new(FakeEmbedder::default());
            let store = Arc::new(FakeStore::default());
            let cache = Cache::new(embedder.clone(), store.clone(), config);
            Self {
                cache,
                embedder,
                store,
            }
        }

        fn embed_calls(&self) -> u64 {
            self.embedder.calls.load(Ordering::SeqCst)
        }

        fn reply_with(&self, vectors: Vec<Vec<f32>>) {
            *self.embedder.reply.lock().unwrap() = Some(vectors);
        }

        async fn set(&self, request: &CacheRequest<'_>) {
            self.cache
                .set(
                    request,
                    b"answer".to_vec(),
                    ResponseMeta::default(),
                    SetOptions::default(),
                )
                .await;
        }
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    fn request<'a>(messages: &'a [ChatMessage]) -> CacheRequest<'a> {
        CacheRequest {
            tenant: "tenant",
            model: "model",
            messages,
            tools: &Value::Null,
            response_format: &Value::Null,
            version_tag: "",
        }
    }

    #[test]
    fn default_config_matches_the_spec() {
        let config = CacheConfig::default();
        assert_eq!(config.k, 5);
        assert_eq!(config.context_mode, ContextMode::SingleTurn);
    }

    #[tokio::test]
    async fn exact_hit_makes_no_embed_call() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [msg("user", "hello")];
        harness.set(&request(&messages)).await;
        assert_eq!(harness.embed_calls(), 1);

        let hit = harness.cache.get(&request(&messages)).await.unwrap();

        assert!(hit.exact);
        assert_eq!(hit.score, 1.0);
        assert_eq!(harness.embed_calls(), 1);
    }

    #[tokio::test]
    async fn case_difference_is_not_an_exact_hit() {
        let harness = Harness::new(CacheConfig::default());
        let stored = [msg("user", "Turkey")];
        harness.set(&request(&stored)).await;
        *harness.store.score.lock().unwrap() = 0.95;

        let query = [msg("user", "turkey")];
        let hit = harness.cache.get(&request(&query)).await.unwrap();

        assert!(!hit.exact);
        assert_eq!(hit.score, 0.95);
        assert_eq!(harness.embed_calls(), 2);
    }

    #[tokio::test]
    async fn threshold_is_inclusive() {
        let harness = Harness::new(CacheConfig {
            threshold: 0.8,
            ..CacheConfig::default()
        });
        let stored = [msg("user", "stored")];
        let query = [msg("user", "query")];
        harness.set(&request(&stored)).await;

        *harness.store.score.lock().unwrap() = 0.8;
        assert!(harness.cache.get(&request(&query)).await.is_some());

        *harness.store.score.lock().unwrap() = 0.799_999;
        assert_eq!(harness.cache.get(&request(&query)).await, None);
    }

    #[tokio::test]
    async fn searches_with_the_configured_k() {
        let harness = Harness::new(CacheConfig {
            k: 7,
            ..CacheConfig::default()
        });
        let query = [msg("user", "query")];
        harness.cache.get(&request(&query)).await;
        assert_eq!(*harness.store.last_k.lock().unwrap(), 7);
    }

    #[tokio::test]
    async fn bad_embeddings_are_counted_misses() {
        let bad_replies = [
            vec![],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0]],
            vec![vec![f32::NAN, 1.0]],
            vec![vec![f32::INFINITY, 1.0]],
        ];
        for (i, reply) in bad_replies.into_iter().enumerate() {
            let harness = Harness::new(CacheConfig::default());
            harness.reply_with(reply);
            let messages = [msg("user", "query")];

            assert_eq!(harness.cache.get(&request(&messages)).await, None, "{i}");
            assert_eq!(harness.cache.error_count(), 1, "{i}");

            harness.set(&request(&messages)).await;
            assert_eq!(harness.cache.error_count(), 2, "{i}");
            assert!(harness.store.entries.lock().unwrap().is_empty(), "{i}");
        }
    }

    #[tokio::test]
    async fn stored_vectors_are_unit_length() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [msg("user", "hello")];
        harness.set(&request(&messages)).await;
        assert_eq!(harness.store.only_entry().embedding, [0.6, 0.8]);
    }

    #[tokio::test]
    async fn set_stores_what_it_was_given() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [msg("user", "  hello\n world ")];
        let meta = ResponseMeta {
            model: "model".into(),
            finish_reason: "stop".into(),
            prompt_tokens: Some(3),
            completion_tokens: Some(4),
        };
        let options = SetOptions {
            ttl: Some(Duration::from_secs(60)),
            tags: vec!["tag".into()],
            version: Some("v1".into()),
        };

        let before = SystemTime::now();
        harness
            .cache
            .set(
                &request(&messages),
                b"answer".to_vec(),
                meta.clone(),
                options,
            )
            .await;

        let entry = harness.store.only_entry();
        assert_eq!(entry.prompt_text.as_deref(), Some("hello world"));
        assert_eq!(entry.key_hash, key_hash("hello world"));
        assert_eq!(entry.response, b"answer");
        assert_eq!(entry.response_meta, meta);
        assert_eq!(entry.ttl, Some(Duration::from_secs(60)));
        assert_eq!(entry.tags, ["tag"]);
        assert_eq!(entry.version.as_deref(), Some("v1"));
        assert!(entry.created_at >= before);
        assert_eq!(entry.stats, EntryStats::default());
        assert_eq!(entry.threshold_override, None);
    }

    #[tokio::test]
    async fn entry_id_is_stable_for_the_same_key_only() {
        let harness = Harness::new(CacheConfig::default());
        let (one, two) = ([msg("user", "one")], [msg("user", "two")]);

        harness.set(&request(&one)).await;
        let first = harness.store.last_entry();
        harness.set(&request(&one)).await;
        assert_eq!(harness.store.last_entry().id, first.id);

        harness.set(&request(&two)).await;
        assert_ne!(harness.store.last_entry().id, first.id);

        let other_tenant = CacheRequest {
            tenant: "other",
            ..request(&one)
        };
        harness.set(&other_tenant).await;
        assert_ne!(harness.store.last_entry().id, first.id);
    }

    #[tokio::test]
    async fn every_request_field_isolates_the_namespace() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [msg("user", "hello")];
        let (tools, format) = (json!(["tool"]), json!({"type": "json_object"}));
        let base = request(&messages);
        let variants = [
            base,
            CacheRequest {
                tenant: "x",
                ..base
            },
            CacheRequest { model: "x", ..base },
            CacheRequest {
                tools: &tools,
                ..base
            },
            CacheRequest {
                response_format: &format,
                ..base
            },
            CacheRequest {
                version_tag: "v2",
                ..base
            },
        ];

        let mut namespaces = Vec::new();
        for variant in &variants {
            harness.set(variant).await;
            namespaces.push(harness.store.last_entry().namespace);
        }
        for (i, a) in namespaces.iter().enumerate() {
            for b in &namespaces[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    async fn namespace_of(messages: &[ChatMessage]) -> Namespace {
        let harness = Harness::new(CacheConfig::default());
        harness.set(&request(messages)).await;
        harness.store.only_entry().namespace
    }

    #[tokio::test]
    async fn system_prompt_is_part_of_the_namespace() {
        let plain = namespace_of(&[msg("user", "hi")]).await;
        let system = namespace_of(&[msg("system", "be brief"), msg("user", "hi")]).await;
        let other = namespace_of(&[msg("system", "be long"), msg("user", "hi")]).await;
        assert_ne!(plain, system);
        assert_ne!(system, other);
    }

    #[tokio::test]
    async fn developer_role_counts_as_system() {
        assert_eq!(
            namespace_of(&[msg("system", "be brief"), msg("user", "hi")]).await,
            namespace_of(&[msg("developer", "be brief"), msg("user", "hi")]).await,
        );
    }

    #[tokio::test]
    async fn all_system_messages_count_in_order() {
        let (a, b) = (msg("system", "a"), msg("system", "b"));
        let user = msg("user", "hi");
        assert_ne!(
            namespace_of(&[a.clone(), b.clone(), user.clone()]).await,
            namespace_of(&[b, a, user]).await,
        );
    }

    #[tokio::test]
    async fn prompt_is_the_last_user_message_not_a_trailing_system_message() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [msg("user", "the question"), msg("system", "trailing")];
        harness.set(&request(&messages)).await;

        let entry = harness.store.only_entry();
        assert_eq!(entry.prompt_text.as_deref(), Some("the question"));
    }

    #[tokio::test]
    async fn uncacheable_conversations_touch_nothing() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [
            msg("user", "first"),
            msg("assistant", "reply"),
            msg("user", "second"),
        ];

        assert_eq!(harness.cache.get(&request(&messages)).await, None);
        harness.set(&request(&messages)).await;

        assert_eq!(harness.embed_calls(), 0);
        assert!(harness.store.entries.lock().unwrap().is_empty());
        assert_eq!(harness.cache.error_count(), 0);
    }
}
