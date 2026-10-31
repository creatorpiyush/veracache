//! Correctness-first semantic cache for LLMs.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::sync::Arc;

pub use veracache_core::*;
pub use veracache_embed::*;
pub use veracache_store::MemoryStore;

/// Creates a cache backed by an in-process store.
pub fn memory_cache(embedder: Arc<dyn Embedder>, config: CacheConfig) -> Cache {
    Cache::new(embedder, Arc::new(MemoryStore::new()), config)
}
