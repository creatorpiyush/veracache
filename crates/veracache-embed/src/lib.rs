//! Embedders for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cached;
#[cfg(feature = "local")]
mod local;

pub use cached::CachedEmbedder;
#[cfg(feature = "local")]
pub use local::LocalEmbedder;
