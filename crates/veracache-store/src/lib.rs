//! Vector stores for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(test)]
mod conformance;
mod memory;

pub use memory::{Clock, MemoryStore};
