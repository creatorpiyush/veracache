//! Types, traits, and lookup pipeline for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod traits;
mod types;

pub use error::{Error, Result};
pub use traits::{Analyzer, Embedder, Guard, Reranker, VectorStore};
pub use types::{
    Analyzed, CacheEntry, Candidate, DeleteFilter, EntryId, EntryStats, Feedback, Namespace,
    RejectReason, ResponseMeta,
};
