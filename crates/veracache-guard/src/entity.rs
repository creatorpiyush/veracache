use veracache_core::{Analyzed, Guard, RejectReason};

use crate::names::NAMES;

/// Rejects a candidate whose entities conflict with the query's.
///
/// One side having no entities is not a conflict, and neither is one side
/// naming a subset of the other, as in "Albert Einstein" and "Einstein".
#[derive(Debug, Clone, Copy, Default)]
pub struct EntityGuard;

impl Guard for EntityGuard {
    fn name(&self) -> &'static str {
        "entity"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        let (a, b) = (&query.entities, &candidate.entities);
        let subset = |x: &[String], y: &[String]| x.iter().all(|word| y.contains(word));
        if a.is_empty() || b.is_empty() || subset(a, b) || subset(b, a) {
            Ok(())
        } else {
            Err(RejectReason::new(self.name(), "entities differ"))
        }
    }
}

/// Capitalized words that name calendar dates, which `DateGuard` handles.
const CALENDAR: [&str; 19] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

/// Different spellings of one entity, as (phrase, canonical word).
const ALIASES: [(&str, &str); 3] = [
    ("united states", "us"),
    ("united kingdom", "uk"),
    ("great britain", "uk"),
];

const WORD_ALIASES: [(&str, &str); 3] = [("usa", "us"), ("america", "us"), ("britain", "uk")];

struct Word<'a> {
    text: &'a str,
    sentence_start: bool,
}

impl Word<'_> {
    fn capitalized(&self) -> bool {
        self.text.starts_with(char::is_uppercase)
    }

    fn all_caps(&self) -> bool {
        self.text.chars().count() > 1 && self.text.chars().all(char::is_uppercase)
    }

    /// Single letters and words with digits are never entities here. "I" is a
    /// pronoun, and "Q3" and "K2" belong to other features or are too rare.
    fn can_be_entity(&self) -> bool {
        self.text.chars().count() > 1 && !self.text.chars().any(|c| c.is_ascii_digit())
    }
}

fn words(text: &str) -> Vec<Word<'_>> {
    let mut words = Vec::new();
    let mut sentence_start = true;
    let mut rest = text;
    while !rest.is_empty() {
        let end = rest
            .find(|c: char| !c.is_alphanumeric())
            .unwrap_or(rest.len());
        if end > 0 {
            words.push(Word {
                text: &rest[..end],
                sentence_start,
            });
            sentence_start = false;
        }
        rest = &rest[end..];
        let separators = rest.find(char::is_alphanumeric).unwrap_or(rest.len());
        sentence_start |= rest[..separators].contains(['.', '?', '!']);
        rest = &rest[separators..];
    }
    words
}

/// Extracts entity words, lowercased and sorted.
pub(crate) fn extract_entities(text: &str) -> Vec<String> {
    let words = words(text);
    let capitalized_mid_sentence: Vec<String> = words
        .iter()
        .filter(|word| word.capitalized() && !word.sentence_start)
        .map(|word| word.text.to_lowercase())
        .collect();

    let mut entities = Vec::new();
    for word in words.iter().filter(|word| word.can_be_entity()) {
        let lower = word.text.to_lowercase();
        let named = word.capitalized()
            && !CALENDAR.contains(&lower.as_str())
            && (!word.sentence_start
                || word.all_caps()
                || capitalized_mid_sentence.contains(&lower));
        if named || NAMES.contains(&lower.as_str()) {
            entities.push(lower);
        }
    }

    for (spelling, canonical) in WORD_ALIASES {
        for entity in entities.iter_mut().filter(|entity| *entity == spelling) {
            *entity = canonical.to_owned();
        }
    }
    let lower = text.to_lowercase();
    for (phrase, canonical) in ALIASES {
        if lower.contains(phrase) {
            entities.retain(|entity| !phrase.split(' ').any(|word| word == entity));
            entities.push(canonical.to_owned());
        }
    }

    entities.sort();
    entities.dedup();
    entities
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn entities(text: &str) -> Vec<String> {
        extract_entities(text)
    }

    fn check(query: &str, candidate: &str) -> Result<(), RejectReason> {
        let analyzed = |text: &str| Analyzed {
            entities: entities(text),
            ..Analyzed::default()
        };
        EntityGuard.check(&analyzed(query), &analyzed(candidate))
    }

    #[test]
    fn capitalized_words_are_entities() {
        assert_eq!(entities("the capital of France"), ["france"]);
        assert_eq!(entities("tell me about Isaac Newton"), ["isaac", "newton"]);
    }

    #[test]
    fn sentence_initial_capitals_are_not_entities() {
        assert!(entities("What is the answer?").is_empty());
        assert!(entities("Tell me. Show me! Why? Because.").is_empty());
    }

    #[test]
    fn a_first_word_counts_if_it_is_capitalized_later_or_all_caps() {
        assert_eq!(entities("NASA is great"), ["nasa"]);
        assert_eq!(entities("Paris is nice, and I love Paris"), ["paris"]);
    }

    #[test]
    fn possessives_keep_the_name() {
        assert_eq!(entities("what is NASA's job"), ["nasa"]);
    }

    #[test]
    fn the_pronoun_i_and_calendar_words_are_not_entities() {
        assert!(entities("can I go on Monday in October").is_empty());
    }

    #[test]
    fn known_names_count_in_lowercase() {
        assert_eq!(entities("capital of france"), ["france"]);
        assert_eq!(entities("who founded microsoft"), ["microsoft"]);
        assert_eq!(entities("best food in chicago"), ["chicago"]);
    }

    #[test]
    fn common_nouns_are_not_names_in_lowercase() {
        assert!(entities("an apple from the amazon in turkey").is_empty());
        assert_eq!(entities("who founded Apple"), ["apple"]);
    }

    #[test]
    fn known_aliases_are_the_same_entity() {
        assert_eq!(entities("the United States"), entities("the US"));
        assert_eq!(entities("the USA"), entities("the US"));
        assert_eq!(entities("the united states"), entities("the US"));
        assert_eq!(entities("the United Kingdom"), entities("the UK"));
    }

    #[test]
    fn different_entities_conflict() {
        assert!(check("capital of France", "capital of Germany").is_err());
        assert!(check("capital of france", "capital of germany").is_err());
        assert!(check("capital of france", "capital of Germany").is_err());
    }

    #[test]
    fn same_entities_pass_whatever_the_case() {
        assert!(check("capital of France", "capital of france").is_ok());
        assert!(check("about Albert Einstein", "about Einstein").is_ok());
    }

    #[test]
    fn one_side_without_entities_passes() {
        assert!(check("the best pizza", "the best pizza in Rome").is_ok());
    }

    #[test]
    fn rejection_names_the_guard() {
        assert_eq!(check("France", "in Germany").unwrap_err().guard, "entity");
    }

    proptest! {
        #[test]
        fn extraction_never_panics(text in "\\PC*") {
            let _ = entities(&text);
        }
    }
}
