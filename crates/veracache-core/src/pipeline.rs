use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use serde_json::Value;

use crate::context::is_system;
use crate::{
    Analyzed, Analyzer, CacheEntry, ChatMessage, ContextMode, Embedder, EntryId, EntryStats, Error,
    Guard, Namespace, NamespaceParts, RejectReason, ResponseMeta, Result, VectorStore,
    context_fingerprint, key_hash, normalize,
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
            threshold: 0.88,
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
    /// Route, which selects its guards. Take it from server configuration,
    /// never from client input, or a client could turn guards off.
    pub route: &'a str,
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

/// What one candidate went through during a lookup.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateTrace {
    /// The candidate's entry.
    pub id: EntryId,
    /// Similarity to the query.
    pub score: f32,
    /// Every guard rejection. Empty if guards did not run or all passed.
    pub rejects: Vec<RejectReason>,
    /// Whether this candidate became the hit.
    pub accepted: bool,
}

/// How a lookup ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The conversation is not cacheable.
    Bypassed,
    /// The exact path hit, so no guard ran.
    ExactHit,
    /// A candidate passed the threshold and every guard.
    SemanticHit,
    /// Nothing survived.
    Miss,
    /// A cache failure, turned into a miss.
    Failed(String),
}

/// The record of one lookup. Holds no response bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Similarity threshold applied.
    pub threshold: f32,
    /// Candidates from vector search, best score first.
    pub candidates: Vec<CandidateTrace>,
    /// How the lookup ended.
    pub outcome: Outcome,
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
    analyzer: Arc<dyn Analyzer>,
    guards: Vec<Arc<dyn Guard>>,
    route_guards: HashMap<String, Vec<Arc<dyn Guard>>>,
    config: CacheConfig,
    errors: AtomicU64,
}

impl Cache {
    /// Creates a cache from its parts.
    pub fn new(
        embedder: Arc<dyn Embedder>,
        store: Arc<dyn VectorStore>,
        analyzer: Arc<dyn Analyzer>,
        guards: Vec<Arc<dyn Guard>>,
        config: CacheConfig,
    ) -> Self {
        Self {
            embedder,
            store,
            analyzer,
            guards,
            route_guards: HashMap::new(),
            config,
            errors: AtomicU64::new(0),
        }
    }

    /// Replaces the guards for one route. Other routes keep the default list,
    /// so an unknown route is never less guarded than the default.
    pub fn with_route_guards(
        mut self,
        route: impl Into<String>,
        guards: Vec<Arc<dyn Guard>>,
    ) -> Self {
        self.route_guards.insert(route.into(), guards);
        self
    }

    /// Looks up a request. `None` means call upstream.
    pub async fn get(&self, request: &CacheRequest<'_>) -> Option<Hit> {
        self.decide(request).await.1
    }

    /// Looks up a request and reports how the decision was reached.
    pub async fn decide(&self, request: &CacheRequest<'_>) -> (Decision, Option<Hit>) {
        let (decision, hit) = self.run(request).await;
        tracing::debug!(
            target: "veracache::decision",
            threshold = decision.threshold,
            outcome = ?decision.outcome,
            candidates = ?decision.candidates,
            "lookup",
        );
        (decision, hit)
    }

    async fn run(&self, request: &CacheRequest<'_>) -> (Decision, Option<Hit>) {
        let mut decision = Decision {
            threshold: self.config.threshold,
            candidates: Vec::new(),
            outcome: Outcome::Bypassed,
        };
        let Some(key) = self.key(request) else {
            return (decision, None);
        };
        let guards = self.route_guards.get(request.route);
        let guards = guards.unwrap_or(&self.guards);
        let hit = match self.lookup(&key, guards, &mut decision).await {
            Ok(hit) => hit,
            Err(error) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                decision.outcome = Outcome::Failed(error.to_string());
                None
            }
        };
        (decision, hit)
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

    async fn lookup(
        &self,
        key: &Key,
        guards: &[Arc<dyn Guard>],
        decision: &mut Decision,
    ) -> Result<Option<Hit>> {
        if let Some(entry) = self.store.get_exact(&key.namespace, &key.key_hash).await? {
            decision.outcome = Outcome::ExactHit;
            return Ok(Some(Hit {
                entry,
                score: 1.0,
                exact: true,
            }));
        }

        let query = self.analyzer.analyze(&key.normalized)?;
        let vector = self.embed(&key.normalized).await?;
        let candidates = self
            .store
            .search(&key.namespace, &vector, self.config.k)
            .await?;

        let mut hit = None;
        for candidate in candidates {
            let eligible = hit.is_none() && candidate.score >= self.config.threshold;
            let rejects = if eligible {
                self.rejects(&query, &candidate.entry, guards)
            } else {
                Vec::new()
            };
            let accepted = eligible && rejects.is_empty();
            decision.candidates.push(CandidateTrace {
                id: candidate.entry.id,
                score: candidate.score,
                rejects,
                accepted,
            });
            if accepted {
                hit = Some(Hit {
                    entry: candidate.entry,
                    score: candidate.score,
                    exact: false,
                });
            }
        }
        decision.outcome = if hit.is_some() {
            Outcome::SemanticHit
        } else {
            Outcome::Miss
        };
        Ok(hit)
    }

    // A candidate that cannot be checked is rejected, never assumed safe. All
    // guards run, so the record shows every reason and not only the first.
    fn rejects(
        &self,
        query: &Analyzed,
        entry: &CacheEntry,
        guards: &[Arc<dyn Guard>],
    ) -> Vec<RejectReason> {
        let Some(features) = &entry.analyzed else {
            return vec![RejectReason::new("features", "missing")];
        };
        if features.feature_version != self.analyzer.feature_version() {
            return vec![RejectReason::new("features", "version mismatch")];
        }
        guards
            .iter()
            .filter_map(|guard| guard.check(query, features).err())
            .collect()
    }

    async fn store_entry(
        &self,
        key: Key,
        response: Vec<u8>,
        response_meta: ResponseMeta,
        options: SetOptions,
    ) -> Result<()> {
        let analyzed = self.analyzer.analyze(&key.normalized)?;
        let embedding = self.embed(&key.normalized).await?;
        self.store
            .upsert(CacheEntry {
                id: entry_id(&key),
                namespace: key.namespace,
                key_hash: key.key_hash,
                prompt_text: Some(key.normalized),
                analyzed: Some(analyzed),
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
    use std::sync::atomic::{AtomicBool, AtomicU32};

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

    #[derive(Default)]
    struct FakeAnalyzer {
        calls: AtomicU64,
        failing: AtomicBool,
        version: AtomicU32,
    }

    impl Analyzer for FakeAnalyzer {
        fn feature_version(&self) -> u32 {
            self.version.load(Ordering::SeqCst)
        }

        fn analyze(&self, text: &str) -> Result<Analyzed> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.failing.load(Ordering::SeqCst) {
                return Err(Error::Analyze("broken".into()));
            }
            Ok(Analyzed {
                text: text.into(),
                feature_version: self.feature_version(),
                ..Analyzed::default()
            })
        }
    }

    /// Vetoes candidates whose text contains `word`, and records its calls.
    struct VetoWord {
        word: &'static str,
        seen: Mutex<Vec<(String, String)>>,
    }

    impl VetoWord {
        fn new(word: &'static str) -> Arc<Self> {
            Arc::new(Self {
                word,
                seen: Mutex::default(),
            })
        }
    }

    impl Guard for VetoWord {
        fn name(&self) -> &'static str {
            "veto_word"
        }

        fn check(
            &self,
            query: &Analyzed,
            candidate: &Analyzed,
        ) -> std::result::Result<(), RejectReason> {
            let pair = (query.text.clone(), candidate.text.clone());
            self.seen.lock().unwrap().push(pair);
            if candidate.text.contains(self.word) {
                return Err(RejectReason::new(self.name(), self.word));
            }
            Ok(())
        }
    }

    struct Harness {
        cache: Cache,
        embedder: Arc<FakeEmbedder>,
        store: Arc<FakeStore>,
        analyzer: Arc<FakeAnalyzer>,
    }

    impl Harness {
        fn new(config: CacheConfig) -> Self {
            Self::guarded(config, Vec::new())
        }

        fn guarded(config: CacheConfig, guards: Vec<Arc<dyn Guard>>) -> Self {
            let embedder = Arc::new(FakeEmbedder::default());
            let store = Arc::new(FakeStore::default());
            let analyzer = Arc::new(FakeAnalyzer::default());
            let cache = Cache::new(
                embedder.clone(),
                store.clone(),
                analyzer.clone(),
                guards,
                config,
            );
            Self {
                cache,
                embedder,
                store,
                analyzer,
            }
        }

        fn analyze_calls(&self) -> u64 {
            self.analyzer.calls.load(Ordering::SeqCst)
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
            route: "",
        }
    }

    #[test]
    fn default_config_matches_the_spec() {
        let config = CacheConfig::default();
        assert_eq!(config.threshold, 0.88);
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

    fn loose() -> CacheConfig {
        CacheConfig {
            threshold: 0.5,
            ..CacheConfig::default()
        }
    }

    async fn stored_with_score(harness: &Harness, prompts: &[&str]) {
        *harness.store.score.lock().unwrap() = 0.9;
        for prompt in prompts {
            let messages = [msg("user", prompt)];
            harness.set(&request(&messages)).await;
        }
    }

    #[tokio::test]
    async fn set_stores_analyzed_features() {
        let harness = Harness::new(CacheConfig::default());
        harness.analyzer.version.store(3, Ordering::SeqCst);
        let messages = [msg("user", "  hello\n world ")];
        harness.set(&request(&messages)).await;

        let analyzed = harness.store.only_entry().analyzed.unwrap();
        assert_eq!(analyzed.text, "hello world");
        assert_eq!(analyzed.feature_version, 3);
    }

    #[tokio::test]
    async fn a_candidate_passing_all_guards_is_a_hit() {
        let harness = Harness::guarded(loose(), vec![VetoWord::new("x"), VetoWord::new("y")]);
        stored_with_score(&harness, &["stored"]).await;

        let query = [msg("user", "query")];
        assert!(harness.cache.get(&request(&query)).await.is_some());
    }

    #[tokio::test]
    async fn one_veto_rejects_the_candidate() {
        let pass = VetoWord::new("nothing");
        let veto = VetoWord::new("stored");
        let harness = Harness::guarded(loose(), vec![pass, veto]);
        stored_with_score(&harness, &["stored"]).await;

        let query = [msg("user", "query")];
        assert_eq!(harness.cache.get(&request(&query)).await, None);
        assert_eq!(harness.cache.error_count(), 0);
    }

    #[tokio::test]
    async fn a_veto_moves_on_to_the_next_candidate() {
        let harness = Harness::guarded(loose(), vec![VetoWord::new("blocked")]);
        stored_with_score(&harness, &["blocked one", "safe two"]).await;

        let query = [msg("user", "query")];
        let hit = harness.cache.get(&request(&query)).await.unwrap();
        assert_eq!(hit.entry.prompt_text.as_deref(), Some("safe two"));
    }

    #[tokio::test]
    async fn a_guard_left_off_a_route_does_not_veto_there() {
        let mut harness = Harness::guarded(loose(), vec![VetoWord::new("stored")]);
        harness.cache = harness.cache.with_route_guards("/open", Vec::new());
        stored_with_score(&harness, &["stored"]).await;

        let query = [msg("user", "query")];
        let open = CacheRequest {
            route: "/open",
            ..request(&query)
        };
        assert!(harness.cache.get(&open).await.is_some());
        assert_eq!(harness.cache.get(&request(&query)).await, None);
    }

    #[tokio::test]
    async fn a_route_can_add_a_guard_the_default_lacks() {
        let mut harness = Harness::guarded(loose(), Vec::new());
        let strict: Vec<Arc<dyn Guard>> = vec![VetoWord::new("stored")];
        harness.cache = harness.cache.with_route_guards("/strict", strict);
        stored_with_score(&harness, &["stored"]).await;

        let query = [msg("user", "query")];
        let vetoed = CacheRequest {
            route: "/strict",
            ..request(&query)
        };
        assert_eq!(harness.cache.get(&vetoed).await, None);
        assert!(harness.cache.get(&request(&query)).await.is_some());
    }

    #[tokio::test]
    async fn an_unknown_route_keeps_the_default_guards() {
        let mut harness = Harness::guarded(loose(), vec![VetoWord::new("stored")]);
        harness.cache = harness.cache.with_route_guards("/open", Vec::new());
        stored_with_score(&harness, &["stored"]).await;

        let query = [msg("user", "query")];
        let unknown = CacheRequest {
            route: "/other",
            ..request(&query)
        };
        assert_eq!(harness.cache.get(&unknown).await, None);
    }

    #[tokio::test]
    async fn guards_see_the_query_and_candidate_features() {
        let guard = VetoWord::new("nothing");
        let harness = Harness::guarded(loose(), vec![guard.clone()]);
        stored_with_score(&harness, &["stored"]).await;

        let query = [msg("user", "query")];
        harness.cache.get(&request(&query)).await;

        let seen = guard.seen.lock().unwrap();
        assert_eq!(*seen, [("query".to_string(), "stored".to_string())]);
    }

    #[tokio::test]
    async fn candidates_below_threshold_are_not_guarded() {
        let guard = VetoWord::new("nothing");
        let harness = Harness::guarded(loose(), vec![guard.clone()]);
        stored_with_score(&harness, &["stored"]).await;
        *harness.store.score.lock().unwrap() = 0.1;

        let query = [msg("user", "query")];
        assert_eq!(harness.cache.get(&request(&query)).await, None);
        assert!(guard.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn exact_hits_skip_analysis_and_guards() {
        let guard = VetoWord::new("stored");
        let harness = Harness::guarded(loose(), vec![guard.clone()]);
        stored_with_score(&harness, &["stored"]).await;
        let analyzed = harness.analyze_calls();

        let same = [msg("user", "stored")];
        let hit = harness.cache.get(&request(&same)).await.unwrap();

        assert!(hit.exact);
        assert_eq!(harness.analyze_calls(), analyzed);
        assert!(guard.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn query_analyzer_failure_is_a_counted_miss() {
        let harness = Harness::guarded(loose(), Vec::new());
        stored_with_score(&harness, &["stored"]).await;
        let embeds = harness.embed_calls();

        harness.analyzer.failing.store(true, Ordering::SeqCst);
        let query = [msg("user", "query")];

        assert_eq!(harness.cache.get(&request(&query)).await, None);
        assert_eq!(harness.cache.error_count(), 1);
        assert_eq!(harness.embed_calls(), embeds);
    }

    #[tokio::test]
    async fn store_analyzer_failure_stores_nothing() {
        let harness = Harness::new(CacheConfig::default());
        harness.analyzer.failing.store(true, Ordering::SeqCst);
        let messages = [msg("user", "hello")];

        harness.set(&request(&messages)).await;

        assert!(harness.store.entries.lock().unwrap().is_empty());
        assert_eq!(harness.cache.error_count(), 1);
    }

    #[tokio::test]
    async fn candidate_without_features_is_rejected() {
        let harness = Harness::guarded(loose(), Vec::new());
        stored_with_score(&harness, &["stored"]).await;
        harness.store.entries.lock().unwrap()[0].analyzed = None;

        let query = [msg("user", "query")];
        assert_eq!(harness.cache.get(&request(&query)).await, None);
    }

    #[tokio::test]
    async fn candidate_with_old_feature_version_is_rejected() {
        let harness = Harness::guarded(loose(), Vec::new());
        stored_with_score(&harness, &["stored"]).await;

        harness.analyzer.version.store(1, Ordering::SeqCst);

        let query = [msg("user", "query")];
        assert_eq!(harness.cache.get(&request(&query)).await, None);
    }

    #[tokio::test]
    async fn decision_records_a_rejected_then_an_accepted_candidate() {
        let harness = Harness::guarded(loose(), vec![VetoWord::new("blocked")]);
        stored_with_score(&harness, &["blocked one", "safe two"]).await;
        let ids: Vec<_> = harness
            .store
            .entries
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.id)
            .collect();

        let query = [msg("user", "query")];
        let (decision, hit) = harness.cache.decide(&request(&query)).await;

        assert_eq!(hit.unwrap().entry.id, ids[1]);
        assert_eq!(
            decision,
            Decision {
                threshold: 0.5,
                candidates: vec![
                    CandidateTrace {
                        id: ids[0],
                        score: 0.9,
                        rejects: vec![RejectReason::new("veto_word", "blocked")],
                        accepted: false,
                    },
                    CandidateTrace {
                        id: ids[1],
                        score: 0.9,
                        rejects: Vec::new(),
                        accepted: true,
                    },
                ],
                outcome: Outcome::SemanticHit,
            }
        );
    }

    #[tokio::test]
    async fn decision_lists_every_guard_that_rejected() {
        let guards: Vec<Arc<dyn Guard>> = vec![VetoWord::new("one"), VetoWord::new("two")];
        let harness = Harness::guarded(loose(), guards);
        stored_with_score(&harness, &["one two"]).await;

        let query = [msg("user", "query")];
        let (decision, _) = harness.cache.decide(&request(&query)).await;

        assert_eq!(decision.outcome, Outcome::Miss);
        assert_eq!(
            decision.candidates[0].rejects,
            [
                RejectReason::new("veto_word", "one"),
                RejectReason::new("veto_word", "two"),
            ]
        );
    }

    #[tokio::test]
    async fn decision_records_a_candidate_below_threshold_unguarded() {
        let harness = Harness::guarded(loose(), vec![VetoWord::new("stored")]);
        stored_with_score(&harness, &["stored"]).await;
        *harness.store.score.lock().unwrap() = 0.1;

        let query = [msg("user", "query")];
        let (decision, _) = harness.cache.decide(&request(&query)).await;

        let trace = &decision.candidates[0];
        assert_eq!((trace.score, trace.accepted), (0.1, false));
        assert!(trace.rejects.is_empty());
        assert_eq!(decision.outcome, Outcome::Miss);
    }

    #[tokio::test]
    async fn decision_names_why_a_candidate_could_not_be_checked() {
        let harness = Harness::guarded(loose(), Vec::new());
        stored_with_score(&harness, &["stored"]).await;
        harness.store.entries.lock().unwrap()[0].analyzed = None;

        let query = [msg("user", "query")];
        let (decision, _) = harness.cache.decide(&request(&query)).await;
        assert_eq!(decision.candidates[0].rejects[0].guard, "features");

        stored_with_score(&harness, &["stored"]).await;
        harness.store.entries.lock().unwrap().remove(0);
        harness.analyzer.version.store(1, Ordering::SeqCst);
        let (decision, _) = harness.cache.decide(&request(&query)).await;
        assert_eq!(decision.candidates[0].rejects[0].detail, "version mismatch");
    }

    #[tokio::test]
    async fn decision_records_exact_bypassed_and_failed_lookups() {
        let harness = Harness::new(CacheConfig::default());
        let messages = [msg("user", "hello")];
        harness.set(&request(&messages)).await;
        let (decision, _) = harness.cache.decide(&request(&messages)).await;
        assert_eq!(decision.outcome, Outcome::ExactHit);

        let multi = [msg("user", "a"), msg("assistant", "b"), msg("user", "c")];
        let (decision, _) = harness.cache.decide(&request(&multi)).await;
        assert_eq!(decision.outcome, Outcome::Bypassed);

        harness.analyzer.failing.store(true, Ordering::SeqCst);
        let query = [msg("user", "query")];
        let (decision, hit) = harness.cache.decide(&request(&query)).await;
        assert!(matches!(decision.outcome, Outcome::Failed(_)));
        assert_eq!(hit, None);
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
