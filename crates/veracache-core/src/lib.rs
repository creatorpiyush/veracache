//! Types, traits, and lookup pipeline for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod context;
mod error;
mod namespace;
mod normalize;
mod pipeline;
mod traits;
mod types;

pub use context::{ChatMessage, ContextMode, context_fingerprint};
pub use error::{Error, Result};
pub use namespace::{NAMESPACE_FORMAT_VERSION, NamespaceParts};
pub use normalize::{key_hash, normalize};
pub use pipeline::{
    Cache, CacheConfig, CacheRequest, CandidateTrace, Decision, Hit, Outcome, SetOptions,
};
pub use traits::{Analyzer, Embedder, Guard, Reranker, VectorStore};
pub use types::{
    Analyzed, CacheEntry, Candidate, DeleteFilter, EntryId, EntryStats, Feedback, Namespace,
    RejectReason, ResponseMeta,
};
