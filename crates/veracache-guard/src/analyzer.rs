use veracache_core::{Analyzed, Analyzer, Result};

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
        Ok(Analyzed {
            text: text.to_owned(),
            feature_version: FEATURE_VERSION,
            numbers: extract_numbers(text),
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
}
