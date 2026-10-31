//! Analyzer and guards for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod analyzer;

use std::sync::Arc;

pub use analyzer::EnglishAnalyzer;
use veracache_core::Guard;

/// The guards every cache runs unless configured otherwise.
pub fn default_guards() -> Vec<Arc<dyn Guard>> {
    Vec::new()
}
