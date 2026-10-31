use veracache_core::{Analyzed, Guard, RejectReason};

/// Rejects a candidate when only one side contains a negation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NegationGuard;

impl Guard for NegationGuard {
    fn name(&self) -> &'static str {
        "negation"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        if query.negations.is_empty() == candidate.negations.is_empty() {
            Ok(())
        } else {
            Err(RejectReason::new(self.name(), "negation differs"))
        }
    }
}

const NEGATIONS: [&str; 13] = [
    "not",
    "no",
    "never",
    "without",
    "except",
    "excluding",
    "cannot",
    "unable",
    "none",
    "nothing",
    "nobody",
    "neither",
    "nor",
];

fn is_negation(word: &str) -> bool {
    NEGATIONS.contains(&word)
        || word.ends_with("n't")
        || word.ends_with("n’t")
        || (word.len() > "-free".len() && word.ends_with("-free"))
}

/// Extracts negation words, sorted.
pub(crate) fn extract_negations(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut negations: Vec<String> = lower
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| is_negation(word))
        .map(str::to_owned)
        .collect();
    negations.sort();
    negations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn negated(text: &str) -> bool {
        !extract_negations(text).is_empty()
    }

    #[test]
    fn finds_negation_words() {
        for text in [
            "not safe",
            "no army",
            "never arrived",
            "without eggs",
            "all except France",
            "cannot log in",
            "unable to log in",
            "nothing works",
            "nobody came",
        ] {
            assert!(negated(text), "{text}");
        }
    }

    #[test]
    fn finds_contractions_with_either_apostrophe() {
        assert!(negated("can't log in"));
        assert!(negated("doesn’t work"));
        assert!(negated("WON'T START"));
    }

    #[test]
    fn free_compounds_are_negations() {
        assert!(negated("an egg-free cake"));
        assert!(negated("gluten-free bread"));
        assert!(!negated("feel free to ask"));
    }

    #[test]
    fn plain_text_has_no_negation() {
        for text in ["notable events", "know the answer", "recipe with eggs"] {
            assert!(!negated(text), "{text}");
        }
    }

    #[test]
    fn guard_compares_presence_not_wording() {
        let analyzed = |text: &str| Analyzed {
            negations: extract_negations(text),
            ..Analyzed::default()
        };
        let guard = NegationGuard;
        assert!(
            guard
                .check(&analyzed("no eggs"), &analyzed("without eggs"))
                .is_ok()
        );
        assert!(
            guard
                .check(&analyzed("eggs"), &analyzed("more eggs"))
                .is_ok()
        );
        let reason = guard.check(&analyzed("eggs"), &analyzed("no eggs"));
        assert_eq!(reason.unwrap_err().guard, "negation");
    }
}
