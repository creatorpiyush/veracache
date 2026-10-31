//! Analyzer and guards for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod analyzer;
mod date;
mod direction;
mod negation;
mod number;

use std::sync::Arc;

pub use analyzer::EnglishAnalyzer;
pub use date::DateGuard;
pub use direction::DirectionGuard;
pub use negation::NegationGuard;
pub use number::NumberGuard;
use veracache_core::Guard;

/// The guards every cache runs unless configured otherwise.
pub fn default_guards() -> Vec<Arc<dyn Guard>> {
    vec![
        Arc::new(NumberGuard),
        Arc::new(DateGuard),
        Arc::new(NegationGuard),
        Arc::new(DirectionGuard),
    ]
}
