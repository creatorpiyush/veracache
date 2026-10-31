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
