use veracache_core::{Analyzed, Guard, RejectReason};

/// Rejects a candidate whose direction conflicts with the query's: opposite
/// words such as before and after, or different endpoints of to, from, than,
/// and vs. Word order alone is not a conflict.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectionGuard;

impl Guard for DirectionGuard {
    fn name(&self) -> &'static str {
        "direction"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        let has = |side: &Analyzed, word: &str| side.directions.iter().any(|d| d == word);
        for (a, b) in OPPOSITES {
            let only = |side: &Analyzed, yes: &str, no: &str| has(side, yes) && !has(side, no);
            if (only(query, a, b) && only(candidate, b, a))
                || (only(query, b, a) && only(candidate, a, b))
            {
                return Err(RejectReason::new(self.name(), "opposite direction"));
            }
        }

        for kind in ["from:", "to:", "than:", "vs:"] {
            let endpoints = |side: &'_ Analyzed| -> Vec<String> {
                let mut found: Vec<String> = side
                    .directions
                    .iter()
                    .filter(|d| d.starts_with(kind))
                    .cloned()
                    .collect();
                found.sort();
                found
            };
            let (a, b) = (endpoints(query), endpoints(candidate));
            if !a.is_empty() && !b.is_empty() && a != b {
                return Err(RejectReason::new(self.name(), "endpoints differ"));
            }
        }
        Ok(())
    }
}

const OPPOSITES: [(&str, &str); 4] = [
    ("before", "after"),
    ("above", "below"),
    ("more", "less"),
    ("higher", "lower"),
];

const ENDPOINT_WORDS: [&str; 3] = ["from", "to", "than"];

const SKIPPED: [&str; 4] = ["the", "a", "an", "my"];

/// Extracts directional features, such as `from:delhi` and `before`.
pub(crate) fn extract_directions(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();

    let mut directions = Vec::new();
    for (i, word) in words.iter().enumerate() {
        if OPPOSITES.iter().any(|(a, b)| a == word || b == word) {
            directions.push((*word).to_owned());
        } else if ENDPOINT_WORDS.contains(word) {
            let target = words[i + 1..].iter().find(|next| !SKIPPED.contains(next));
            if let Some(target) = target {
                directions.push(format!("{word}:{target}"));
            }
        } else if matches!(*word, "vs" | "versus")
            && let (Some(left), Some(right)) = (
                i.checked_sub(1).and_then(|left| words.get(left)),
                words.get(i + 1),
            )
        {
            directions.push(format!("vs:{left}>{right}"));
        }
    }
    directions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(query: &str, candidate: &str) -> Result<(), RejectReason> {
        let analyzed = |text: &str| Analyzed {
            directions: extract_directions(text),
            ..Analyzed::default()
        };
        DirectionGuard.check(&analyzed(query), &analyzed(candidate))
    }

    #[test]
    fn swapped_endpoints_conflict() {
        assert!(
            check(
                "flights from Delhi to Mumbai",
                "flights from Mumbai to Delhi"
            )
            .is_err()
        );
        assert!(check("convert USD to EUR", "convert EUR to USD").is_err());
    }

    #[test]
    fn reordered_phrases_do_not_conflict() {
        assert!(
            check(
                "flights from Delhi to Mumbai",
                "flights to Mumbai from Delhi"
            )
            .is_ok()
        );
        assert!(check("from the US to India", "to India from the US").is_ok());
    }

    #[test]
    fn opposites_conflict_in_either_direction() {
        for (a, b) in [
            ("stretch before running", "stretch after running"),
            ("above freezing", "below freezing"),
            ("more fiber", "less fiber"),
            ("higher cost", "lower cost"),
        ] {
            assert!(check(a, b).is_err(), "{a}");
            assert!(check(b, a).is_err(), "{b}");
        }
    }

    #[test]
    fn matching_direction_words_pass() {
        assert!(check("after a car accident", "after an accident").is_ok());
        assert!(check("which has more protein", "which one has more protein").is_ok());
    }

    #[test]
    fn a_word_on_one_side_only_is_not_a_conflict() {
        assert!(check("convert USD to EUR", "change dollars into euros").is_ok());
        assert!(check("what to do after a crash", "steps after a crash").is_ok());
    }

    #[test]
    fn comparison_operands_conflict() {
        assert!(check("is Python faster than Java", "is Java faster than Python").is_err());
        assert!(check("Python vs Java", "Java vs Python").is_err());
        assert!(check("Python versus Java", "Java vs Python").is_err());
        assert!(check("Python vs Java", "Python vs Java").is_ok());
    }

    #[test]
    fn articles_are_skipped_after_prepositions() {
        assert_eq!(extract_directions("from the US"), ["from:us"]);
    }

    #[test]
    fn plain_text_has_no_directions() {
        assert!(extract_directions("what is recursion").is_empty());
    }

    #[test]
    fn rejection_names_the_guard() {
        let reason = check("before", "after").unwrap_err();
        assert_eq!(reason.guard, "direction");
    }
}
