use veracache_core::{Analyzed, Guard, RejectReason};

/// Rejects a candidate whose quoted strings or code spans differ.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuoteGuard;

impl Guard for QuoteGuard {
    fn name(&self) -> &'static str {
        "quote"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        if query.quotes == candidate.quotes {
            Ok(())
        } else {
            Err(RejectReason::new(self.name(), "quotes differ"))
        }
    }
}

/// Extracts double-quoted strings and code spans, sorted, and returns the text
/// without them. Single quotes are skipped, because they are also apostrophes.
pub(crate) fn extract_quotes(text: &str) -> (Vec<String>, String) {
    let (mut quotes, mut without) = (Vec::new(), String::new());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        let after = &rest[c.len_utf8()..];
        let closing = match c {
            '"' => Some('"'),
            '“' => Some('”'),
            '`' => Some('`'),
            _ => None,
        };
        match closing.and_then(|close| after.find(close).map(|end| (close, end))) {
            Some((close, end)) => {
                if end > 0 {
                    quotes.push(after[..end].to_owned());
                }
                without.push(' ');
                rest = &after[end + close.len_utf8()..];
            }
            None => {
                without.push(c);
                rest = after;
            }
        }
    }
    quotes.sort();
    (quotes, without)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn quotes(text: &str) -> Vec<String> {
        extract_quotes(text).0
    }

    #[test]
    fn finds_double_quotes_and_code_spans() {
        assert_eq!(quotes(r#"what does "carpe diem" mean"#), ["carpe diem"]);
        assert_eq!(quotes("what does `git rebase` do"), ["git rebase"]);
    }

    #[test]
    fn finds_curly_quotes() {
        assert_eq!(quotes("what does “carpe diem” mean"), ["carpe diem"]);
    }

    #[test]
    fn finds_several_sorted() {
        assert_eq!(quotes(r#"compare "b" and "a""#), ["a", "b"]);
    }

    #[test]
    fn apostrophes_are_not_quotes() {
        assert!(quotes("don't say it's fine").is_empty());
    }

    #[test]
    fn an_unclosed_quote_is_ignored() {
        assert!(quotes(r#"the 6" screen"#).is_empty());
    }

    #[test]
    fn empty_quotes_are_skipped() {
        assert!(quotes(r#"an empty "" string"#).is_empty());
    }

    #[test]
    fn case_matters_in_quotes() {
        assert_ne!(quotes(r#""Word""#), quotes(r#""word""#));
    }

    #[test]
    fn quotes_are_removed_from_the_remaining_text() {
        let (_, rest) = extract_quotes(r#"who said "I Came" in Rome"#);
        assert!(!rest.contains("Came"), "{rest}");
        assert!(rest.contains("Rome"), "{rest}");
    }

    #[test]
    fn guard_rejects_only_differences() {
        let analyzed = |text: &str| Analyzed {
            quotes: quotes(text),
            ..Analyzed::default()
        };
        let guard = QuoteGuard;
        assert!(guard.check(&analyzed(r#""a""#), &analyzed("`a`")).is_ok());
        let reason = guard.check(&analyzed(r#""a""#), &analyzed(r#""b""#));
        assert_eq!(reason.unwrap_err().guard, "quote");
    }

    proptest! {
        #[test]
        fn extraction_never_panics(text in "\\PC*") {
            let _ = extract_quotes(&text);
        }
    }
}
