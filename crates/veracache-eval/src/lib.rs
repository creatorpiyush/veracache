//! Replay, sweep, and reports for veracache.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

use serde_json::Value;

/// Error type for loading evaluation data.
pub type LoadError = Box<dyn Error + Send + Sync>;

/// Whether a pair should share one cached answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    /// One answer is correct for both prompts.
    Hit,
    /// The prompts need different answers.
    Miss,
}

/// Which part of the suite a pair belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// Used while tuning guards.
    Tuning,
    /// Reserved for the final measurement.
    HeldOut,
}

/// A labelled pair of prompts from the golden suite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    /// Stable identifier.
    pub id: String,
    /// Which false-hit class the pair exercises.
    pub category: String,
    /// Expected outcome.
    pub label: Label,
    /// Suite split.
    pub split: Split,
    /// The stored prompt.
    pub a: String,
    /// The query prompt.
    pub b: String,
    /// Conversation before `a`, for follow-ups.
    pub a_history: Option<Vec<String>>,
    /// Conversation before `b`, for follow-ups.
    pub b_history: Option<Vec<String>>,
}

impl Pair {
    /// Hashes the pair's texts, so stale precomputed scores can be detected.
    pub fn content_hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        for text in [&self.a, &self.b] {
            hasher.update(&(text.len() as u64).to_le_bytes());
            hasher.update(text.as_bytes());
        }
        hasher.finalize().to_hex().to_string()
    }
}

fn parse_pair(line: &str) -> Result<Pair, LoadError> {
    let row: Value = serde_json::from_str(line)?;
    let text = |field: &str| -> Result<String, LoadError> {
        Ok(row[field]
            .as_str()
            .ok_or_else(|| format!("missing {field}: {line}"))?
            .to_owned())
    };
    let history = |field: &str| {
        row[field].as_array().map(|turns| {
            turns
                .iter()
                .filter_map(|turn| turn.as_str().map(str::to_owned))
                .collect()
        })
    };
    Ok(Pair {
        id: text("id")?,
        category: text("category")?,
        label: match text("label")?.as_str() {
            "hit" => Label::Hit,
            "miss" => Label::Miss,
            other => return Err(format!("bad label: {other}").into()),
        },
        split: match text("split")?.as_str() {
            "tuning" => Split::Tuning,
            "held_out" => Split::HeldOut,
            other => return Err(format!("bad split: {other}").into()),
        },
        a: text("a")?,
        b: text("b")?,
        a_history: history("a_history"),
        b_history: history("b_history"),
    })
}

/// Loads a golden suite from a JSONL file.
pub fn load_pairs(path: impl AsRef<Path>) -> Result<Vec<Pair>, LoadError> {
    std::fs::read_to_string(path)?
        .lines()
        .map(parse_pair)
        .collect()
}

/// Loads precomputed similarity scores by pair id, refusing stale ones.
pub fn load_scores(
    path: impl AsRef<Path>,
    pairs: &[Pair],
) -> Result<HashMap<String, f32>, LoadError> {
    let mut hashes: HashMap<_, _> = pairs
        .iter()
        .map(|pair| (pair.id.as_str(), pair.content_hash()))
        .collect();
    let mut scores = HashMap::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let id = row["id"].as_str().ok_or("score row has no id")?;
        let expected = hashes
            .remove(id)
            .ok_or_else(|| format!("score for unknown pair: {id}"))?;
        if row["hash"].as_str() != Some(expected.as_str()) {
            return Err(format!("stale score: {id}").into());
        }
        let score = row["score"]
            .as_f64()
            .ok_or_else(|| format!("score row has no score: {id}"))?;
        scores.insert(id.to_owned(), score as f32);
    }
    match hashes.keys().next() {
        Some(missing) => Err(format!("no score for pair: {missing}").into()),
        None => Ok(scores),
    }
}

/// Outcome counts for a set of pairs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Pairs that should have hit and did.
    pub correct_hits: usize,
    /// Pairs that should have missed and hit.
    pub false_hits: usize,
    /// Pairs labelled as should-hit.
    pub should_hit: usize,
}

impl Counts {
    /// Records one pair's outcome.
    pub fn record(&mut self, label: Label, served: bool) {
        match (label, served) {
            (Label::Hit, true) => {
                self.correct_hits += 1;
                self.should_hit += 1;
            }
            (Label::Hit, false) => self.should_hit += 1,
            (Label::Miss, true) => self.false_hits += 1,
            (Label::Miss, false) => {}
        }
    }

    /// False hits divided by all hits served. `None` if nothing was served.
    pub fn false_hit_rate(&self) -> Option<f32> {
        let served = self.correct_hits + self.false_hits;
        (served > 0).then(|| self.false_hits as f32 / served as f32)
    }

    /// Correct hits divided by pairs that should have hit. `None` if none should.
    pub fn hit_recall(&self) -> Option<f32> {
        (self.should_hit > 0).then(|| self.correct_hits as f32 / self.should_hit as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: &str =
        r#"{"id":"x","category":"number","label":"hit","split":"tuning","a":"A","b":"B"}"#;

    #[test]
    fn counts_each_outcome() {
        let mut counts = Counts::default();
        counts.record(Label::Hit, true);
        counts.record(Label::Hit, false);
        counts.record(Label::Miss, true);
        counts.record(Label::Miss, false);
        assert_eq!(
            counts,
            Counts {
                correct_hits: 1,
                false_hits: 1,
                should_hit: 2
            }
        );
    }

    #[test]
    fn rates_follow_their_definitions() {
        let counts = Counts {
            correct_hits: 3,
            false_hits: 1,
            should_hit: 6,
        };
        assert_eq!(counts.false_hit_rate(), Some(0.25));
        assert_eq!(counts.hit_recall(), Some(0.5));
    }

    #[test]
    fn rates_are_none_without_data() {
        assert_eq!(Counts::default().false_hit_rate(), None);
        assert_eq!(Counts::default().hit_recall(), None);
    }

    #[test]
    fn parses_a_pair() {
        let pair = parse_pair(ROW).unwrap();
        assert_eq!(pair.id, "x");
        assert_eq!(pair.label, Label::Hit);
        assert_eq!(pair.split, Split::Tuning);
        assert_eq!(pair.a_history, None);
    }

    #[test]
    fn parses_held_out_miss_with_history() {
        let row = r#"{"id":"x","category":"follow_up","label":"miss","split":"held_out","a":"A","b":"B","a_history":[],"b_history":["q","r"]}"#;
        let pair = parse_pair(row).unwrap();
        assert_eq!(pair.label, Label::Miss);
        assert_eq!(pair.split, Split::HeldOut);
        assert_eq!(pair.a_history, Some(vec![]));
        assert_eq!(pair.b_history, Some(vec!["q".into(), "r".into()]));
    }

    #[test]
    fn rejects_bad_rows() {
        assert!(parse_pair("not json").is_err());
        assert!(parse_pair(r#"{"id":"x"}"#).is_err());
        assert!(parse_pair(&ROW.replace("\"hit\"", "\"maybe\"")).is_err());
        assert!(parse_pair(&ROW.replace("\"tuning\"", "\"other\"")).is_err());
    }

    #[test]
    fn hash_tracks_text_and_field_boundaries() {
        let pair = parse_pair(ROW).unwrap();
        let mut changed = pair.clone();
        changed.b = "C".into();
        assert_ne!(pair.content_hash(), changed.content_hash());

        let (mut one, mut two) = (pair.clone(), pair);
        (one.a, one.b) = ("ab".into(), "c".into());
        (two.a, two.b) = ("a".into(), "bc".into());
        assert_ne!(one.content_hash(), two.content_hash());
    }
}
