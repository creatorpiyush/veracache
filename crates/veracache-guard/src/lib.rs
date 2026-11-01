//! Analyzer and guards for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod analyzer;
mod date;
mod direction;
mod entity;
mod length;
mod names;
mod negation;
mod number;
mod quote;

use std::sync::Arc;

pub use analyzer::EnglishAnalyzer;
pub use date::DateGuard;
pub use direction::DirectionGuard;
pub use entity::EntityGuard;
pub use length::LengthRatioGuard;
pub use negation::NegationGuard;
pub use number::NumberGuard;
pub use quote::QuoteGuard;
use veracache_core::Guard;

/// The guards every cache runs unless configured otherwise.
pub fn default_guards() -> Vec<Arc<dyn Guard>> {
    vec![
        Arc::new(NumberGuard),
        Arc::new(DateGuard),
        Arc::new(NegationGuard),
        Arc::new(DirectionGuard),
        Arc::new(EntityGuard),
        Arc::new(QuoteGuard),
        Arc::new(LengthRatioGuard::default()),
    ]
}
