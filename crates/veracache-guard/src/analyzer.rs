use veracache_core::{Analyzed, Analyzer, Result};

use crate::date::extract_dates;
use crate::direction::extract_directions;
use crate::negation::extract_negations;
use crate::number::extract_numbers;

const FEATURE_VERSION: u32 = 1;

/// Extracts guard features from English prompts.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnglishAnalyzer;

impl Analyzer for EnglishAnalyzer {
    fn feature_version(&self) -> u32 {
        FEATURE_VERSION
    }

    fn analyze(&self, text: &str) -> Result<Analyzed> {
        let (dates, without_dates) = extract_dates(text);
        Ok(Analyzed {
            text: text.to_owned(),
            feature_version: FEATURE_VERSION,
            numbers: extract_numbers(&without_dates),
            dates,
            negations: extract_negations(text),
            directions: extract_directions(text),
            ..Analyzed::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_text_and_stamps_the_version() {
        let analyzed = EnglishAnalyzer.analyze("hello world").unwrap();
        assert_eq!(analyzed.text, "hello world");
        assert_eq!(analyzed.feature_version, EnglishAnalyzer.feature_version());
    }

    #[test]
    fn date_digits_are_not_counted_as_numbers() {
        let iso = EnglishAnalyzer.analyze("price on 2024-01-15").unwrap();
        let written = EnglishAnalyzer
            .analyze("price on January 15, 2024")
            .unwrap();
        assert_eq!(iso.dates, written.dates);
        assert!(iso.numbers.is_empty());
        assert!(written.numbers.is_empty());
    }

    #[test]
    fn numbers_outside_dates_are_kept() {
        let analyzed = EnglishAnalyzer.analyze("buy 300 on 2024-01-15").unwrap();
        assert_eq!(analyzed.numbers, ["300"]);
    }
}
