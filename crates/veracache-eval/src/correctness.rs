//! Decides whether a would-be hit was correct: labels, then an LLM judge, then
//! heuristics.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use veracache::{
    Analyzer, DateGuard, EnglishAnalyzer, EntityGuard, Guard, NegationGuard, NumberGuard,
};

use crate::{LlmJudge, LoadError, LogRow};

/// Where a verdict came from. Reports name it, because a heuristic verdict
/// is weaker evidence than a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    /// A human label for the pair.
    Label,
    /// The LLM judge.
    Judge,
    /// The LLM judge failed or gave no clear answer. Counted as a false hit,
    /// never as correct.
    JudgeError,
    /// The response heuristic: the two logged responses are compared.
    Heuristic,
}

/// Whether one hit was correct, and how that was decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verdict {
    /// True if serving the source's response to the query was right.
    pub correct: bool,
    /// What decided it.
    pub source: Source,
}

/// Human labels for pairs of log rows, in either order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Labels(HashMap<(String, String), bool>);

impl Labels {
    /// Parses JSONL lines of `{"a": id, "b": id, "label": "hit" | "miss"}`.
    ///
    /// Ids must name rows in `rows`, so a typo is an error, not a silent gap.
    /// A pair labelled twice must agree.
    pub fn parse(text: &str, rows: &[LogRow]) -> Result<Self, LoadError> {
        let ids: HashSet<&str> = rows.iter().map(|row| row.id.as_str()).collect();
        let mut labels = HashMap::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let number = index + 1;
            let row: Value =
                serde_json::from_str(line).map_err(|e| format!("line {number}: {e}"))?;
            let id = |field: &str| -> Result<String, LoadError> {
                let id = row[field]
                    .as_str()
                    .ok_or(format!("line {number}: missing {field}"))?;
                if !ids.contains(id) {
                    return Err(format!("line {number}: unknown row id {id}").into());
                }
                Ok(id.to_owned())
            };
            let correct = match row["label"].as_str() {
                Some("hit") => true,
                Some("miss") => false,
                _ => return Err(format!("line {number}: label must be hit or miss").into()),
            };
            let key = key(&id("a")?, &id("b")?);
            if labels
                .insert(key.clone(), correct)
                .is_some_and(|old| old != correct)
            {
                return Err(format!(
                    "line {number}: conflicting label for {} and {}",
                    key.0, key.1
                )
                .into());
            }
        }
        Ok(Self(labels))
    }

    /// Loads labels from a file. See [`Labels::parse`].
    pub fn load(path: impl AsRef<Path>, rows: &[LogRow]) -> Result<Self, LoadError> {
        Self::parse(&std::fs::read_to_string(path)?, rows)
    }

    /// The label for a pair, in either order.
    pub fn get(&self, a: &str, b: &str) -> Option<bool> {
        self.0.get(&key(a, b)).copied()
    }

    pub(crate) fn insert(&mut self, a: &str, b: &str, correct: bool) {
        self.0.insert(key(a, b), correct);
    }
}

fn key(a: &str, b: &str) -> (String, String) {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    (a.to_owned(), b.to_owned())
}

/// Judges hits from labels, then an optional LLM judge, then by comparing the
/// logged responses.
///
/// The heuristic runs the number, date, entity, and negation checks on the
/// two rows' upstream responses, not on their prompts: the prompt guards
/// already passed every guarded hit, so checking prompts again would always
/// say correct. Responses that differ in those features mean the cached
/// answer is probably wrong for the query. It misses wrong answers that share
/// those features, and it flags right answers worded with different facts,
/// so treat its FHR as an estimate.
pub struct Correctness {
    labels: Labels,
    judge: Option<LlmJudge>,
    guards: Vec<Arc<dyn Guard>>,
}

impl Correctness {
    /// Uses `labels` first, then the response heuristic.
    pub fn new(labels: Labels) -> Self {
        Self {
            labels,
            judge: None,
            guards: vec![
                Arc::new(NumberGuard),
                Arc::new(DateGuard),
                Arc::new(EntityGuard),
                Arc::new(NegationGuard),
            ],
        }
    }

    /// Asks `judge` about unlabelled pairs instead of using the heuristic.
    pub fn with_judge(mut self, judge: LlmJudge) -> Self {
        self.judge = Some(judge);
        self
    }

    /// Judges serving `source`'s response to `query`.
    pub async fn judge(&self, query: &LogRow, source: &LogRow) -> Verdict {
        if let Some(correct) = self.labels.get(&query.id, &source.id) {
            return Verdict {
                correct,
                source: Source::Label,
            };
        }
        if let Some(judge) = &self.judge {
            return match judge.judge(query, source).await {
                Ok(correct) => Verdict {
                    correct,
                    source: Source::Judge,
                },
                Err(_) => Verdict {
                    correct: false,
                    source: Source::JudgeError,
                },
            };
        }
        let analyze = |row: &LogRow| EnglishAnalyzer.analyze(&response_text(&row.response));
        let correct = match (analyze(query), analyze(source)) {
            (Ok(query), Ok(source)) => self
                .guards
                .iter()
                .all(|guard| guard.check(&query, &source).is_ok()),
            // A response that cannot be read cannot be shown correct.
            _ => false,
        };
        Verdict {
            correct,
            source: Source::Heuristic,
        }
    }
}

/// The text of a logged response: the first choice's message content for an
/// OpenAI-style body, otherwise the bytes as text.
pub(crate) fn response_text(response: &[u8]) -> String {
    let body = String::from_utf8_lossy(response);
    serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| {
            value["choices"][0]["message"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_log;

    fn rows(responses: &[(&str, &str)]) -> Vec<LogRow> {
        let lines: Vec<String> = responses
            .iter()
            .map(|(id, response)| {
                serde_json::json!({
                    "id": id,
                    "messages": [{"role": "user", "content": "question"}],
                    "response": response,
                })
                .to_string()
            })
            .collect();
        parse_log(&lines.join("\n")).unwrap()
    }

    fn row<'a>(rows: &'a [LogRow], id: &str) -> &'a LogRow {
        rows.iter().find(|row| row.id == id).unwrap()
    }

    #[tokio::test]
    async fn labels_win_over_the_heuristic() {
        let rows = rows(&[("a", "The answer is 3."), ("b", "The answer is 3.")]);
        let labels = Labels::parse(r#"{"a":"b","b":"a","label":"miss"}"#, &rows).unwrap();
        let verdict = Correctness::new(labels)
            .judge(row(&rows, "a"), row(&rows, "b"))
            .await;
        assert_eq!(
            verdict,
            Verdict {
                correct: false,
                source: Source::Label
            }
        );
    }

    #[tokio::test]
    async fn heuristic_judges_unlabelled_pairs_by_their_responses() {
        let rows = rows(&[
            ("paris-1", "The capital of France is Paris."),
            ("paris-2", "The capital is Paris, of course."),
            ("berlin", "The capital is Berlin, of course."),
            ("moons-2", "It has 2 moons."),
            ("moons-95", "It has 95 moons."),
            ("yes", "Yes, it is safe to eat."),
            ("no", "No, it is not safe to eat."),
        ]);
        let correctness = Correctness::new(Labels::default());
        let judge = async |a, b| correctness.judge(row(&rows, a), row(&rows, b)).await;

        assert_eq!(judge("paris-1", "paris-2").await.source, Source::Heuristic);
        assert!(judge("paris-1", "paris-2").await.correct);
        assert!(!judge("paris-2", "berlin").await.correct);
        assert!(!judge("moons-2", "moons-95").await.correct);
        assert!(!judge("yes", "no").await.correct);
    }

    #[tokio::test]
    async fn openai_bodies_are_compared_by_their_content() {
        let body = |text: &str| serde_json::json!({"choices": [{"message": {"role": "assistant", "content": text}}]});
        let text = [
            serde_json::json!({"id": "a", "messages": [], "response": body("It has 2 moons.")}),
            serde_json::json!({"id": "b", "messages": [], "response": body("It has 95 moons.")}),
        ]
        .map(|row| row.to_string())
        .join("\n");
        let rows = parse_log(&text).unwrap();
        let verdict = Correctness::new(Labels::default())
            .judge(&rows[0], &rows[1])
            .await;
        assert!(!verdict.correct);
    }

    #[test]
    fn labels_are_checked_against_the_log() {
        let rows = rows(&[("a", "x"), ("b", "y")]);
        let cases = [
            (
                r#"{"a":"a","b":"zz","label":"hit"}"#,
                "line 1: unknown row id zz",
            ),
            (r#"{"a":"a","label":"hit"}"#, "line 1: missing b"),
            (
                r#"{"a":"a","b":"b","label":"yes"}"#,
                "line 1: label must be hit or miss",
            ),
            (
                "{\"a\":\"a\",\"b\":\"b\",\"label\":\"hit\"}\n{\"a\":\"b\",\"b\":\"a\",\"label\":\"miss\"}",
                "line 2: conflicting label for a and b",
            ),
        ];
        for (text, error) in cases {
            assert_eq!(Labels::parse(text, &rows).unwrap_err().to_string(), error);
        }
        let same = "{\"a\":\"a\",\"b\":\"b\",\"label\":\"hit\"}\n\n{\"a\":\"b\",\"b\":\"a\",\"label\":\"hit\"}";
        let labels = Labels::parse(same, &rows).unwrap();
        assert_eq!(labels.get("b", "a"), Some(true));
        assert_eq!(labels.get("a", "a"), None);
    }
}
