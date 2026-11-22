//! The language check: is this prompt English the guards can trust?

use veracache_core::LanguageCheck;

// Tuned on the golden tuning split only. The held-out split checks them.
/// Fewer words than this is too short to tell.
const MIN_WORDS: u32 = 2;
/// At least this share of words must be on the English list.
const MIN_ENGLISH_PERCENT: u32 = 25;

/// Common English words. A word before an apostrophe is matched alone, so
/// "doesn't" matches "doesn".
#[rustfmt::skip]
const ENGLISH: &[&str] = &[
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are", "aren",
    "as", "at", "be", "because", "been", "before", "being", "best", "between", "but", "by",
    "calculate", "can", "change", "compare", "convert", "could", "couldn", "create",
    "define", "describe", "did", "didn", "difference", "display", "do", "does", "doesn",
    "don", "down", "during", "each", "every", "explain", "find", "first", "for", "free",
    "from", "get", "give", "good", "had", "hadn", "has", "hasn", "have", "haven", "having",
    "he", "help", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "isn",
    "it", "its", "just", "know", "last", "list", "make", "many", "may", "me", "mean",
    "might", "more", "most", "much", "must", "my", "need", "never", "new", "next", "not",
    "of", "off", "on", "only", "or", "other", "our", "out", "over", "please", "same",
    "shall", "she", "should", "shouldn", "show", "so", "some", "such", "summarize", "tell",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "top", "translate", "under", "up", "us", "use", "very",
    "want", "was", "wasn", "way", "we", "were", "weren", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "will", "with", "within", "without", "won", "work",
    "would", "wouldn", "write", "you", "your",
];

/// Common words in other Latin-script languages that are not English words:
/// Spanish, French, German, Italian, Portuguese, and romanized Hindi.
const FOREIGN: &[&str] = &[
    // Spanish
    "el", "la", "los", "las", "del", "que", "y", "un", "una", "es", "por", "para", "con", "como",
    "cual", "cuantas", "cuantos", "esta", "son", "se", "lo", "le", "su", "sus", "pero", "muy",
    "mas", "dame", "debo", "hace", // French
    "les", "des", "du", "et", "est", "une", "pour", "dans", "sur", "pas", "qui", "ce", "cette",
    "il", "elle", "je", "tu", "nous", "vous", "avec", "sont", "au", "aux", "ne", "quel", "quelle",
    "quels", "comment", "pourquoi", "faire", "faut", "peut", "l", "d", "j", "qu", "c", "n",
    // German
    "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "einen", "einem", "mit", "von", "zu",
    "den", "dem", "des", "sich", "auf", "wie", "wann", "warum", "wo", "wer", "kann", "ich", "du",
    "er", "sie", "es", "wir", "ihr", "sind", "hat", "haben", "werden", "auch", "oder", "aber",
    "noch", "nur", "nach", "bei", "aus", "um", "nenne", "viele",
    // Italian and Portuguese
    "di", "che", "per", "non", "sono", "della", "uma", "voce", // Romanized Hindi
    "hai", "hain", "ka", "ki", "ke", "ko", "mein", "kya", "kaise", "kyun", "kab", "kahan", "kaun",
    "aur", "nahi", "nahin", "bhi", "hoga", "hogi", "tha", "thi", "karna", "karein", "karo",
    "batao", "chahiye", "liye", "aap", "tum", "hum", "mujhe", "kuch",
];

/// Checks whether `text` is English the guards can trust.
///
/// Every error refuses a semantic hit instead of allowing one: non-Latin
/// letters, too few words, or too little English evidence all fail.
pub(crate) fn check_language(text: &str) -> LanguageCheck {
    if text.chars().any(|c| c.is_alphabetic() && !is_latin(c)) {
        return LanguageCheck::NonLatinScript;
    }
    let (mut words, mut english, mut foreign) = (0, 0, 0);
    for word in text
        .split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '’'))
        .filter(|word| word.chars().any(char::is_alphabetic))
    {
        words += 1;
        let stem = word
            .split(['\'', '’'])
            .next()
            .unwrap_or_default()
            .to_lowercase();
        if !word.is_ascii() || FOREIGN.contains(&stem.as_str()) {
            foreign += 1;
        } else if ENGLISH.contains(&stem.as_str()) {
            english += 1;
        }
    }
    if words < MIN_WORDS {
        LanguageCheck::TooShort { words }
    } else if english > foreign && english * 100 >= words * MIN_ENGLISH_PERCENT {
        LanguageCheck::Supported
    } else {
        LanguageCheck::TooFewEnglishWords {
            english,
            foreign,
            words,
        }
    }
}

/// True for letters in the Latin script blocks.
fn is_latin(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z' | 'ª' | 'º'
        | '\u{00C0}'..='\u{024F}' // Latin-1 letters, Extended-A and -B
        | '\u{1E00}'..='\u{1EFF}' // Latin Extended Additional
        | '\u{2C60}'..='\u{2C7F}' // Latin Extended-C
        | '\u{A720}'..='\u{A7FF}' // Latin Extended-D
        | '\u{AB30}'..='\u{AB6F}' // Latin Extended-E
    ) && c != '×'
        && c != '÷'
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn check(text: &str) -> LanguageCheck {
        check_language(text)
    }

    #[test]
    fn plain_english_is_supported() {
        for text in [
            "What is the capital of France?",
            "Give me 5 tips for better sleep",
            "How long does it take to walk 10k?",
            "What's the difference between TCP and UDP?",
            "Why doesn't my Rust code compile?",
        ] {
            assert_eq!(check(text), LanguageCheck::Supported, "{text}");
        }
    }

    #[test]
    fn non_latin_letters_fail_the_script_gate() {
        for text in [
            "भारत की राजधानी क्या है?",
            "中国的首都是哪里？",
            "What is the capital of Россия?",
            "Explain the word καλημέρα",
        ] {
            assert_eq!(check(text), LanguageCheck::NonLatinScript, "{text}");
        }
    }

    #[test]
    fn digits_and_symbols_are_not_letters() {
        assert_eq!(
            check("What is 2 + 2 = 4 in base १०?"),
            LanguageCheck::Supported
        );
        assert_eq!(
            check("Is 50% off a good deal? 🙂"),
            LanguageCheck::Supported
        );
    }

    #[test]
    fn latin_script_non_english_is_unsupported() {
        for text in [
            "¿Cuál es la capital de Australia?",
            "Dame tres ideas para una cena rápida",
            "Quels aliments sont riches en fer ?",
            "Was ist die Hauptstadt von Kanada?",
            "Wann beginnt die Sommerzeit in Deutschland?",
            "Delhi se Mumbai ki train kab hai?",
            "Chai banane ka tarika batao",
            "¿Cómo convierto grados Celsius a Fahrenheit?",
        ] {
            assert!(
                matches!(check(text), LanguageCheck::TooFewEnglishWords { .. }),
                "{text}: {:?}",
                check(text)
            );
        }
    }

    #[test]
    fn foreign_words_must_not_outnumber_english_ones() {
        let result = check("Was ist die Hauptstadt von Kanada?");
        let LanguageCheck::TooFewEnglishWords {
            english,
            foreign,
            words,
        } = result
        else {
            panic!("{result:?}");
        };
        assert_eq!(words, 6);
        assert!(foreign >= english, "{result:?}");
    }

    #[test]
    fn short_prompts_are_undetermined() {
        assert_eq!(check("hello"), LanguageCheck::TooShort { words: 1 });
        assert_eq!(check("Why?"), LanguageCheck::TooShort { words: 1 });
        assert_eq!(check("?? 42"), LanguageCheck::TooShort { words: 0 });
        assert_eq!(check(""), LanguageCheck::TooShort { words: 0 });
    }

    #[test]
    fn two_word_prompts_need_an_english_word() {
        assert_eq!(check("Explain recursion"), LanguageCheck::Supported);
        assert_eq!(check("Define inflation"), LanguageCheck::Supported);
        assert!(!check("Hola amigo").is_supported());
        assert!(!check("Danke schön").is_supported());
    }

    #[test]
    fn jargon_without_english_words_is_unsupported() {
        assert!(matches!(
            check("k8s pod restart loop fix"),
            LanguageCheck::TooFewEnglishWords { english: 0, .. }
        ));
    }

    #[test]
    fn accented_words_count_against_english() {
        let result = check("Quel temps fait-il à Paris au printemps ?");
        assert!(
            matches!(result, LanguageCheck::TooFewEnglishWords { .. }),
            "{result:?}"
        );
        // One loanword does not sink an English prompt.
        assert_eq!(
            check("Where is the best café in the city?"),
            LanguageCheck::Supported
        );
    }

    fn non_latin_letter() -> impl Strategy<Value = char> {
        prop_oneof![
            proptest::char::range('\u{0905}', '\u{0939}'), // Devanagari
            proptest::char::range('\u{4E00}', '\u{9FFF}'), // CJK
            proptest::char::range('\u{0410}', '\u{044F}'), // Cyrillic
            proptest::char::range('\u{03B1}', '\u{03C9}'), // Greek
            proptest::char::range('\u{0627}', '\u{064A}'), // Arabic
        ]
    }

    proptest! {
        #[test]
        fn any_non_latin_letter_is_never_supported(
            english in "[A-Za-z ,.?']{0,60}",
            letter in non_latin_letter(),
            at in 0usize..=60,
        ) {
            let mut text = format!("What is the best way to do this {english}");
            let at = text.char_indices().map(|(i, _)| i).nth(at).unwrap_or(text.len());
            text.insert(at, letter);
            prop_assert_eq!(check(&text), LanguageCheck::NonLatinScript);
        }
    }
}
