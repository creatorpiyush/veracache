use veracache_core::{Analyzed, Guard, RejectReason};

/// Rejects a candidate whose dates differ from the query's.
#[derive(Debug, Clone, Copy, Default)]
pub struct DateGuard;

impl Guard for DateGuard {
    fn name(&self) -> &'static str {
        "date"
    }

    fn check(&self, query: &Analyzed, candidate: &Analyzed) -> Result<(), RejectReason> {
        if query.dates == candidate.dates {
            Ok(())
        } else {
            Err(RejectReason::new(self.name(), "dates differ"))
        }
    }
}

/// Month names. The flag says a name counts alone, without an adjacent number,
/// which excludes words like "may" and "march".
const MONTHS: [(&str, u32, bool); 24] = [
    ("january", 1, true),
    ("february", 2, true),
    ("march", 3, false),
    ("april", 4, true),
    ("may", 5, false),
    ("june", 6, true),
    ("july", 7, true),
    ("august", 8, true),
    ("september", 9, true),
    ("october", 10, true),
    ("november", 11, true),
    ("december", 12, true),
    ("jan", 1, false),
    ("feb", 2, false),
    ("mar", 3, false),
    ("apr", 4, false),
    ("jun", 6, false),
    ("jul", 7, false),
    ("aug", 8, false),
    ("sep", 9, false),
    ("sept", 9, false),
    ("oct", 10, false),
    ("nov", 11, false),
    ("dec", 12, false),
];

const WEEKDAYS: [&str; 7] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

const RELATIVE_DAYS: [&str; 4] = ["today", "tonight", "tomorrow", "yesterday"];

const PERIODS: [&str; 9] = [
    "week",
    "weekend",
    "month",
    "quarter",
    "year",
    "morning",
    "afternoon",
    "evening",
    "night",
];

fn month(word: &str) -> Option<(u32, bool)> {
    MONTHS
        .iter()
        .find(|(name, _, _)| *name == word)
        .map(|(_, number, alone)| (*number, *alone))
}

fn day(word: &str) -> Option<u32> {
    let digits = ["st", "nd", "rd", "th"]
        .iter()
        .find_map(|suffix| word.strip_suffix(suffix))
        .unwrap_or(word);
    let is_digits = digits.len() <= 2 && digits.bytes().all(|b| b.is_ascii_digit());
    digits
        .parse()
        .ok()
        .filter(|day| is_digits && (1..=31).contains(day))
}

fn year(word: &str) -> Option<u32> {
    let is_digits = word.len() == 4 && word.bytes().all(|b| b.is_ascii_digit());
    word.parse()
        .ok()
        .filter(|year| is_digits && (1000..3000).contains(year))
}

fn two_digits(word: &str) -> Option<u32> {
    let is_digits = word.len() == 2 && word.bytes().all(|b| b.is_ascii_digit());
    word.parse().ok().filter(|_| is_digits)
}

fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_alphanumeric(), start) {
            (true, None) => start = Some(i),
            (false, Some(first)) => {
                spans.push((first, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(first) = start {
        spans.push((first, text.len()));
    }
    spans
}

/// The words of a text, with what separates them.
struct Words<'a> {
    text: &'a str,
    spans: Vec<(usize, usize)>,
}

impl<'a> Words<'a> {
    fn get(&self, i: usize) -> Option<&'a str> {
        self.spans
            .get(i)
            .map(|&(start, end)| &self.text[start..end])
    }

    fn joined_by_hyphen(&self, i: usize) -> bool {
        self.spans
            .get(i + 1)
            .is_some_and(|next| &self.text[self.spans[i].1..next.0] == "-")
    }

    fn iso_date(&self, i: usize) -> Option<(String, usize)> {
        let y = year(self.get(i)?)?;
        let m = two_digits(self.get(i + 1)?)?;
        let d = two_digits(self.get(i + 2)?)?;
        (self.joined_by_hyphen(i) && self.joined_by_hyphen(i + 1))
            .then(|| (format!("{y}-{m:02}-{d:02}"), 3))
    }

    fn month_first(&self, i: usize) -> Option<(String, usize)> {
        let (m, alone) = month(self.get(i)?)?;
        let next = self.get(i + 1);
        if let Some(d) = next.and_then(day) {
            return Some(match self.get(i + 2).and_then(year) {
                Some(y) => (format!("{y}-{m:02}-{d:02}"), 3),
                None => (format!("{m:02}-{d:02}"), 2),
            });
        }
        if let Some(y) = next.and_then(year) {
            return Some((format!("{y}-{m:02}"), 2));
        }
        alone.then(|| (format!("month:{m}"), 1))
    }

    fn day_first(&self, i: usize) -> Option<(String, usize)> {
        let d = day(self.get(i)?)?;
        let of = usize::from(self.get(i + 1) == Some("of"));
        let (m, _) = month(self.get(i + 1 + of)?)?;
        Some(match self.get(i + 2 + of).and_then(year) {
            Some(y) => (format!("{y}-{m:02}-{d:02}"), 3 + of),
            None => (format!("{m:02}-{d:02}"), 2 + of),
        })
    }

    fn relative(&self, i: usize) -> Option<(String, usize)> {
        let word = self.get(i)?;
        if matches!(word, "this" | "next" | "last") {
            let next = self.get(i + 1)?;
            let names_time = PERIODS.contains(&next) || WEEKDAYS.contains(&next);
            return names_time.then(|| (format!("{word} {next}"), 2));
        }
        (WEEKDAYS.contains(&word) || RELATIVE_DAYS.contains(&word)).then(|| (word.to_owned(), 1))
    }

    fn date_at(&self, i: usize) -> Option<(String, usize)> {
        self.iso_date(i)
            .or_else(|| self.month_first(i))
            .or_else(|| self.day_first(i))
            .or_else(|| self.relative(i))
    }
}

/// Extracts normalized dates, and returns the text with those dates removed so
/// their digits are not counted again as numbers.
pub(crate) fn extract_dates(text: &str) -> (Vec<String>, String) {
    let lower = text.to_lowercase();
    let words = Words {
        text: &lower,
        spans: word_spans(&lower),
    };

    let (mut dates, mut rest) = (Vec::new(), String::new());
    let (mut i, mut copied) = (0, 0);
    while i < words.spans.len() {
        let Some((date, length)) = words.date_at(i) else {
            i += 1;
            continue;
        };
        rest.push_str(&lower[copied..words.spans[i].0]);
        rest.push(' ');
        copied = words.spans[i + length - 1].1;
        dates.push(date);
        i += length;
    }
    rest.push_str(&lower[copied..]);
    dates.sort();
    (dates, rest)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn dates(text: &str) -> Vec<String> {
        extract_dates(text).0
    }

    #[test]
    fn written_and_iso_dates_normalize_the_same() {
        assert_eq!(dates("on October 3, 2025"), ["2025-10-03"]);
        assert_eq!(dates("on Oct 3rd, 2025"), ["2025-10-03"]);
        assert_eq!(dates("on 2025-10-03"), ["2025-10-03"]);
        assert_eq!(dates("on 3 October 2025"), ["2025-10-03"]);
        assert_eq!(dates("on the 3rd of October 2025"), ["2025-10-03"]);
    }

    #[test]
    fn month_and_year_forms() {
        assert_eq!(dates("in November 2020"), ["2020-11"]);
        assert_eq!(dates("on March 5"), ["03-05"]);
        assert_eq!(dates("holidays in December"), ["month:12"]);
    }

    #[test]
    fn ambiguous_month_words_need_a_number() {
        assert!(dates("you may march on").is_empty());
        assert_eq!(dates("in May 2024"), ["2024-05"]);
        assert_eq!(dates("on March 5, 2024"), ["2024-03-05"]);
    }

    #[test]
    fn relative_words() {
        assert_eq!(dates("the weather today"), ["today"]);
        assert_eq!(dates("today's news"), ["today"]);
        assert_eq!(dates("tomorrow"), ["tomorrow"]);
        assert_eq!(dates("yesterday's spend"), ["yesterday"]);
        assert_ne!(dates("today"), dates("tomorrow"));
    }

    #[test]
    fn this_next_and_last() {
        assert_eq!(dates("sales from last week"), ["last week"]);
        assert_eq!(dates("last week's sales"), ["last week"]);
        assert_ne!(dates("last week"), dates("this week"));
        assert_ne!(dates("last month"), dates("last year"));
        assert_ne!(dates("this weekend"), dates("next weekend"));
        assert_ne!(dates("this morning"), dates("this evening"));
    }

    #[test]
    fn weekdays_keep_their_qualifier() {
        assert_eq!(dates("open on Monday"), ["monday"]);
        assert_eq!(dates("next Tuesday"), ["next tuesday"]);
        assert_eq!(dates("this Friday's deadline"), ["this friday"]);
        assert_ne!(dates("next Tuesday"), dates("next Thursday"));
    }

    #[test]
    fn plain_text_has_no_dates() {
        assert!(dates("the last word on this topic").is_empty());
        assert!(dates("the 2018 World Cup").is_empty());
    }

    #[test]
    fn dates_are_removed_from_the_remaining_text() {
        let (_, rest) = extract_dates("price on January 15, 2024 was 300");
        assert!(!rest.contains("15"), "{rest}");
        assert!(!rest.contains("2024"), "{rest}");
        assert!(rest.contains("300"), "{rest}");

        let (_, rest) = extract_dates("price on 2024-01-15");
        assert!(!rest.contains("2024"), "{rest}");
    }

    #[test]
    fn invalid_days_are_not_dates() {
        assert_eq!(dates("October 45"), ["month:10"]);
    }

    #[test]
    fn guard_rejects_only_differences() {
        let analyzed = |text: &str| Analyzed {
            dates: dates(text),
            ..Analyzed::default()
        };
        let guard = DateGuard;
        assert!(
            guard
                .check(&analyzed("today"), &analyzed("today's"))
                .is_ok()
        );
        let reason = guard.check(&analyzed("today"), &analyzed("tomorrow"));
        assert_eq!(reason.unwrap_err().guard, "date");
    }

    proptest! {
        #[test]
        fn iso_dates_round_trip(y in 1900u32..2100, m in 1u32..=12, d in 1u32..=28) {
            let iso = format!("{y}-{m:02}-{d:02}");
            prop_assert_eq!(dates(&format!("on {iso} we met")), [iso]);
        }

        #[test]
        fn extraction_never_panics(text in "\\PC*") {
            let _ = extract_dates(&text);
        }
    }
}
