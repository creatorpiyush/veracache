use veracache_core::{Analyzed, Analyzer, Result};

use crate::date::extract_dates;
use crate::direction::extract_directions;
use crate::entity::extract_entities;
use crate::language::check_language;
use crate::negation::extract_negations;
use crate::number::extract_numbers;
use crate::quote::extract_quotes;

const FEATURE_VERSION: u32 = 2;

/// Extracts guard features from English prompts.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnglishAnalyzer;

impl Analyzer for EnglishAnalyzer {
    fn feature_version(&self) -> u32 {
        FEATURE_VERSION
    }

    fn analyze(&self, text: &str) -> Result<Analyzed> {
        let (dates, without_dates) = extract_dates(text);
        let (quotes, without_quotes) = extract_quotes(text);
        Ok(Analyzed {
            text: text.to_owned(),
            feature_version: FEATURE_VERSION,
            numbers: extract_numbers(&without_dates),
            dates,
            entities: extract_entities(&without_quotes),
            negations: extract_negations(text),
            directions: extract_directions(text),
            quotes,
            language: check_language(text),
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
    fn records_the_language_check() {
        let english = EnglishAnalyzer.analyze("What is the capital of France?");
        assert!(english.unwrap().language.is_supported());
        let hindi = EnglishAnalyzer.analyze("भारत की राजधानी क्या है?").unwrap();
        assert_eq!(
            hindi.language,
            veracache_core::LanguageCheck::NonLatinScript
        );
    }

    // Entries stored before the language check have no language result, so
    // they must be rejected instead of read as checked.
    #[test]
    fn feature_version_counts_the_language_check() {
        assert_eq!(EnglishAnalyzer.feature_version(), 2);
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
