use veracache_core::{Analyzed, Guard, RejectReason};

/// Rejects a candidate whose numbers differ from the query's.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumberGuard;

impl Guard for NumberGuard {
    fn name(&self) -> &'static str {
        "number"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        if query.numbers == candidate.numbers {
            Ok(())
        } else {
            Err(RejectReason::new(self.name(), "numbers differ"))
        }
    }
}

// "one" is left out: it is usually a pronoun, as in "which one is better".
const CARDINALS: [(&str, u64); 26] = [
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
];

const ORDINALS: [(&str, u64); 20] = [
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
    ("eighth", 8),
    ("ninth", 9),
    ("tenth", 10),
    ("eleventh", 11),
    ("twelfth", 12),
    ("thirteenth", 13),
    ("fourteenth", 14),
    ("fifteenth", 15),
    ("sixteenth", 16),
    ("seventeenth", 17),
    ("eighteenth", 18),
    ("nineteenth", 19),
    ("twentieth", 20),
];

const SCALES: [(&str, u64); 4] = [
    ("hundred", 100),
    ("thousand", 1_000),
    ("million", 1_000_000),
    ("billion", 1_000_000_000),
];

fn lookup(table: &[(&str, u64)], word: &str) -> Option<u64> {
    table
        .iter()
        .find(|(name, _)| *name == word)
        .map(|(_, value)| *value)
}

/// A number being spelled out across several words.
#[derive(Default)]
struct SpelledNumber {
    total: u64,
    current: u64,
    active: bool,
}

impl SpelledNumber {
    fn add(&mut self, value: u64) {
        self.current = self.current.saturating_add(value);
        self.active = true;
    }

    fn scale(&mut self, scale: u64) {
        let scaled = self.current.max(1).saturating_mul(scale);
        if scale == 100 {
            self.current = scaled;
        } else {
            self.total = self.total.saturating_add(scaled);
            self.current = 0;
        }
        self.active = true;
    }

    fn finish(&mut self, numbers: &mut Vec<String>) {
        if self.active {
            let value = self.total.saturating_add(self.current);
            numbers.push(value.to_string());
        }
        *self = Self::default();
    }
}

/// Extracts numbers in a canonical form, sorted so equal multisets compare equal.
pub(crate) fn extract_numbers(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '.' | ',' | '%')))
        .map(|token| token.trim_matches(['.', ',', '%']))
        .filter(|token| !token.is_empty())
        .collect();

    let mut numbers = Vec::new();
    let mut spelled = SpelledNumber::default();
    for (i, token) in tokens.iter().enumerate() {
        let next_is_scale = tokens
            .get(i + 1)
            .is_some_and(|next| lookup(&SCALES, next).is_some());
        if let Some(value) = lookup(&CARDINALS, token) {
            spelled.add(value);
        } else if *token == "one" && (spelled.active || next_is_scale) {
            spelled.add(1);
        } else if let Some(scale) = lookup(&SCALES, token) {
            spelled.scale(scale);
        } else if let Some(value) = lookup(&ORDINALS, token) {
            spelled.add(value);
            spelled.finish(&mut numbers);
        } else {
            spelled.finish(&mut numbers);
            push_numeric(token, &mut numbers);
        }
    }
    spelled.finish(&mut numbers);
    numbers.sort();
    numbers
}

fn push_numeric(token: &str, numbers: &mut Vec<String>) {
    let bytes = token.as_bytes();
    if bytes.len() == 2 && bytes[0] == b'q' && (b'1'..=b'4').contains(&bytes[1]) {
        numbers.push(token.to_owned());
        return;
    }

    let ordinal = ["st", "nd", "rd", "th"]
        .iter()
        .find_map(|suffix| token.strip_suffix(suffix));
    if let Some(digits) = ordinal.filter(|digits| is_numeric(digits)) {
        numbers.push(canonical(digits));
        return;
    }

    let scaled = [("k", 1e3), ("m", 1e6), ("b", 1e9)]
        .iter()
        .find_map(|(suffix, scale)| Some((token.strip_suffix(suffix)?, *scale)));
    if let Some((digits, scale)) = scaled.filter(|(digits, _)| is_numeric(digits))
        && let Ok(value) = digits.replace(',', "").parse::<f64>()
    {
        numbers.push(format!("{}", (value * scale).round() as u64));
        return;
    }

    if !is_numeric(token) {
        return;
    }
    let groups: Vec<&str> = token.split(',').collect();
    let thousands = groups[0].len() <= 3 && groups[1..].iter().all(|group| group.len() == 3);
    if thousands {
        numbers.push(canonical(&groups.concat()));
    } else {
        numbers.extend(
            groups
                .iter()
                .filter(|group| !group.is_empty())
                .map(|group| canonical(group)),
        );
    }
}

fn is_numeric(token: &str) -> bool {
    token.starts_with(|c: char| c.is_ascii_digit())
        && token
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | ','))
}

// Decimals stay as written, because version 3.10 is not version 3.1.
fn canonical(digits: &str) -> String {
    if digits.contains('.') {
        return digits.to_owned();
    }
    match digits.trim_start_matches('0') {
        "" => "0".to_owned(),
        trimmed => trimmed.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn numbers(text: &str) -> Vec<String> {
        extract_numbers(text)
    }

    #[test]
    fn plain_integers_and_years() {
        assert_eq!(numbers("the 2018 World Cup"), ["2018"]);
        assert_eq!(numbers("no numbers here"), Vec::<String>::new());
    }

    #[test]
    fn quarters_stay_tagged() {
        assert_eq!(numbers("revenue in Q3 2024"), ["2024", "q3"]);
        assert_ne!(numbers("Q2"), numbers("Q3"));
        assert_ne!(numbers("Q3"), numbers("3"));
    }

    #[test]
    fn thousands_separators_and_suffixes() {
        assert_eq!(numbers("10,000 meters"), ["10000"]);
        assert_eq!(numbers("walk 10k"), ["10000"]);
        assert_eq!(numbers("80K salary"), ["80000"]);
        assert_eq!(numbers("1.5m users"), ["1500000"]);
        assert_eq!(numbers("2b market"), ["2000000000"]);
    }

    #[test]
    fn comma_lists_are_separate_numbers() {
        assert_eq!(numbers("pick 1,2,3"), ["1", "2", "3"]);
    }

    #[test]
    fn percent_is_the_number() {
        assert_eq!(numbers("15% of 200"), numbers("15 percent of 200"));
    }

    #[test]
    fn decimals_keep_their_digits() {
        assert_eq!(numbers("Python 3.11."), ["3.11"]);
        assert_ne!(numbers("3.10"), numbers("3.1"));
        assert_ne!(numbers("3.11"), numbers("3.12"));
    }

    #[test]
    fn ordinals_match_words_and_digits() {
        assert_eq!(numbers("the 16th president"), ["16"]);
        assert_eq!(numbers("the sixteenth president"), ["16"]);
        assert_eq!(numbers("the 2nd largest"), numbers("the second largest"));
        assert_eq!(numbers("the 1st"), numbers("the first"));
        assert_eq!(numbers("the 3rd"), numbers("the third"));
    }

    #[test]
    fn number_words_match_digits() {
        assert_eq!(numbers("two plus two"), ["2", "2"]);
        assert_eq!(numbers("five tips"), numbers("5 tips"));
        assert_eq!(numbers("a seven-day forecast"), numbers("a 7 day forecast"));
        assert_eq!(numbers("twenty one days"), ["21"]);
        assert_eq!(numbers("two hundred"), ["200"]);
        assert_eq!(numbers("one hundred"), ["100"]);
        assert_eq!(numbers("three thousand five hundred"), ["3500"]);
    }

    #[test]
    fn bare_one_is_not_a_number() {
        assert!(numbers("which one is better").is_empty());
    }

    #[test]
    fn letters_with_digits_are_not_numbers() {
        assert!(numbers("climb K2 in 3D with 5G and 100kPa").is_empty());
    }

    #[test]
    fn leading_zeros_are_ignored() {
        assert_eq!(numbers("007"), ["7"]);
        assert_eq!(numbers("0"), ["0"]);
    }

    #[test]
    fn multisets_compare_regardless_of_order() {
        assert_eq!(numbers("2 then 10"), numbers("10 then 2"));
        assert_ne!(numbers("2 and 2"), numbers("2"));
    }

    #[test]
    fn guard_rejects_only_differences() {
        let analyzed = |text: &str| Analyzed {
            numbers: numbers(text),
            ..Analyzed::default()
        };
        let guard = NumberGuard;
        assert!(guard.check(&analyzed("Q3"), &analyzed("q3")).is_ok());
        let reason = guard.check(&analyzed("Q2"), &analyzed("Q3")).unwrap_err();
        assert_eq!(reason.guard, "number");
    }

    proptest! {
        #[test]
        fn integers_round_trip(n in 0u32..10_000_000) {
            prop_assert_eq!(numbers(&format!("about {n} items")), [n.to_string()]);
        }

        #[test]
        fn separators_do_not_change_the_value(n in 1_000u32..1_000_000) {
            let (high, low) = (n / 1_000, n % 1_000);
            let with_comma = format!("{high},{low:03}");
            prop_assert_eq!(numbers(&with_comma), [n.to_string()]);
        }

        #[test]
        fn order_does_not_matter(a in 0u32..1_000, b in 0u32..1_000) {
            prop_assert_eq!(numbers(&format!("{a} {b}")), numbers(&format!("{b} {a}")));
        }

        #[test]
        fn extraction_never_panics(text in "\\PC*") {
            let _ = numbers(&text);
        }
    }
}
