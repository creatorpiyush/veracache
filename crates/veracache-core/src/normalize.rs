use unicode_normalization::UnicodeNormalization;

/// Applies Unicode NFKC and collapses whitespace. Case is preserved, because an
/// exact hit skips guards and "Turkey" must not collide with "turkey".
///
/// ```
/// assert_eq!(veracache_core::normalize("  Hello\n\tworld "), "Hello world");
/// ```
pub fn normalize(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    nfkc.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Hashes a normalized prompt into its exact-match key.
pub fn key_hash(normalized: &str) -> [u8; 32] {
    *blake3::hash(normalized.as_bytes()).as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn preserves_case() {
        assert_eq!(normalize("Turkey"), "Turkey");
        assert_ne!(
            key_hash(&normalize("Turkey")),
            key_hash(&normalize("turkey"))
        );
    }

    #[test]
    fn applies_nfkc() {
        assert_eq!(normalize("ﬁle\u{00A0}Ｑ３"), "file Q3");
    }

    proptest! {
        #[test]
        fn is_idempotent(text in "\\PC*") {
            let once = normalize(&text);
            prop_assert_eq!(normalize(&once), once);
        }

        #[test]
        fn whitespace_runs_do_not_change_the_key(a in "\\S+", b in "\\S+", gap in "[ \t\n]{1,5}") {
            prop_assert_eq!(
                key_hash(&normalize(&format!("{a}{gap}{b}"))),
                key_hash(&normalize(&format!("{a} {b}"))),
            );
        }
    }
}
