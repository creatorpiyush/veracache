//! Replays a request log through the real cache, with no upstream calls.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use serde_json::Value;
use veracache::{
    Cache, CacheConfig, CacheRequest, ChatMessage, Decision, Embedder, EnglishAnalyzer, EntryId,
    MemoryStore, ResponseMeta, SetOptions, default_guards,
};

use crate::LoadError;

/// Token counts upstream reported for a logged response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Tokens in the request.
    pub prompt_tokens: u32,
    /// Tokens in the response.
    pub completion_tokens: u32,
}

/// One logged request and the response upstream gave it.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRow {
    /// Row identifier. Defaults to `line-<n>`.
    pub id: String,
    /// Tenant scope. Defaults to `default`.
    pub tenant: String,
    /// Model name. Defaults to empty.
    pub model: String,
    /// Conversation, ending in the user message.
    pub messages: Vec<ChatMessage>,
    /// Tools schema. `Value::Null` when absent.
    pub tools: Value,
    /// Response format. `Value::Null` when absent.
    pub response_format: Value,
    /// Upstream response: a JSON string's text, or other JSON as written.
    pub response: Vec<u8>,
    /// Token counts, when the log has them.
    pub usage: Option<Usage>,
}

/// Parses a JSONL request log. Blank lines are skipped; ids must be unique.
///
/// Each line is an object with `messages` (`role` and string `content`) and
/// `response`. `id`, `tenant`, `model`, `tools`, `response_format`, and
/// `usage` (`prompt_tokens`, `completion_tokens`) are optional. Other fields,
/// such as `ts`, are ignored.
pub fn parse_log(text: &str) -> Result<Vec<LogRow>, LoadError> {
    let mut rows = Vec::new();
    let mut ids = HashSet::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let number = index + 1;
        let row = parse_row(line, number).map_err(|e| format!("line {number}: {e}"))?;
        if !ids.insert(row.id.clone()) {
            return Err(format!("line {number}: duplicate id {}", row.id).into());
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Loads a JSONL request log from a file. See [`parse_log`].
pub fn load_log(path: impl AsRef<Path>) -> Result<Vec<LogRow>, LoadError> {
    parse_log(&std::fs::read_to_string(path)?)
}

fn parse_row(line: &str, number: usize) -> Result<LogRow, LoadError> {
    let row: Value = serde_json::from_str(line)?;
    let text = |field: &str, default: &str| -> Result<String, LoadError> {
        match &row[field] {
            Value::Null => Ok(default.to_owned()),
            Value::String(value) => Ok(value.clone()),
            _ => Err(format!("{field} must be a string").into()),
        }
    };
    let messages = row["messages"]
        .as_array()
        .ok_or("missing messages")?
        .iter()
        .map(|message| {
            let field = |name: &str| {
                message[name]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("message {name} must be a string"))
            };
            Ok(ChatMessage {
                role: field("role")?,
                content: field("content")?,
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    let response = match &row["response"] {
        Value::Null => return Err("missing response".into()),
        Value::String(text) => text.clone().into_bytes(),
        other => other.to_string().into_bytes(),
    };
    let usage = match &row["usage"] {
        Value::Null => None,
        usage => {
            let count = |name: &str| -> Result<u32, LoadError> {
                let value = usage[name].as_u64().ok_or(format!("usage needs {name}"))?;
                Ok(u32::try_from(value)?)
            };
            Some(Usage {
                prompt_tokens: count("prompt_tokens")?,
                completion_tokens: count("completion_tokens")?,
            })
        }
    };
    Ok(LogRow {
        id: text("id", &format!("line-{number}"))?,
        tenant: text("tenant", "default")?,
        model: text("model", "")?,
        messages,
        tools: row["tools"].clone(),
        response_format: row["response_format"].clone(),
        response,
        usage,
    })
}

/// What happened to one logged row during a replay.
#[derive(Debug, Clone, PartialEq)]
pub struct Replayed {
    /// The row's id.
    pub id: String,
    /// The cache's decision record for the row.
    pub decision: Decision,
    /// For a hit, the id of the earlier row whose response was served.
    pub served_from: Option<String>,
    /// For each candidate in `decision.candidates`, the row that stored it.
    pub candidate_rows: Vec<Option<String>>,
}

const ROW_TAG: &str = "row:";

/// Replays `rows` in order through a guarded cache over a fresh memory store.
///
/// Each row is looked up first. A hit is recorded and stores nothing, as a
/// proxy would serve it. A miss stores the logged response instead of calling
/// upstream. The store's clock is fixed, so no entry expires and results do
/// not depend on when the replay runs.
///
/// To embed each text once across many replays, pass a
/// [`veracache::CachedEmbedder`] with enough capacity for the whole log.
pub async fn replay(
    rows: &[LogRow],
    embedder: Arc<dyn Embedder>,
    config: CacheConfig,
) -> Vec<Replayed> {
    let store = MemoryStore::with_clock(Arc::new(|| SystemTime::UNIX_EPOCH));
    let cache = Cache::new(
        embedder,
        Arc::new(store),
        Arc::new(EnglishAnalyzer),
        default_guards(),
        config,
    );
    let mut stored_by: HashMap<EntryId, String> = HashMap::new();
    let mut replayed = Vec::with_capacity(rows.len());
    for row in rows {
        let request = CacheRequest {
            tenant: &row.tenant,
            model: &row.model,
            messages: &row.messages,
            tools: &row.tools,
            response_format: &row.response_format,
            version_tag: "",
            route: "",
        };
        let (decision, hit) = cache.decide(&request).await;
        let served_from = hit.as_ref().and_then(|hit| {
            let tags = &hit.entry.tags;
            tags.iter()
                .find_map(|tag| tag.strip_prefix(ROW_TAG).map(str::to_owned))
        });
        if hit.is_none() {
            let meta = ResponseMeta {
                model: row.model.clone(),
                finish_reason: String::new(),
                prompt_tokens: row.usage.map(|usage| usage.prompt_tokens),
                completion_tokens: row.usage.map(|usage| usage.completion_tokens),
            };
            let options = SetOptions {
                tags: vec![format!("{ROW_TAG}{}", row.id)],
                ..SetOptions::default()
            };
            cache
                .set(&request, row.response.clone(), meta, options)
                .await;
            if let Some(id) = cache.entry_id(&request) {
                stored_by.insert(id, row.id.clone());
            }
        }
        let candidate_rows = decision
            .candidates
            .iter()
            .map(|candidate| stored_by.get(&candidate.id).cloned())
            .collect();
        replayed.push(Replayed {
            id: row.id.clone(),
            decision,
            served_from,
            candidate_rows,
        });
    }
    replayed
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use veracache::{CachedEmbedder, Error, Outcome};

    use super::*;

    /// Maps known texts to vectors and counts every text it embeds.
    struct FakeEmbedder {
        vectors: HashMap<&'static str, Vec<f32>>,
        embedded: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embedder for FakeEmbedder {
        fn id(&self) -> &str {
            "fake"
        }

        fn dim(&self) -> usize {
            2
        }

        async fn embed(&self, texts: &[String]) -> veracache::Result<Vec<Vec<f32>>> {
            self.embedded.fetch_add(texts.len(), Ordering::SeqCst);
            texts
                .iter()
                .map(|text| {
                    self.vectors
                        .get(text.as_str())
                        .cloned()
                        .ok_or_else(|| Error::Embed(format!("unknown text: {text}")))
                })
                .collect()
        }
    }

    const FRANCE: &str = "What is the capital of France?";
    const PARAPHRASE: &str = "What's the capital city of France?";
    const GERMANY: &str = "What is the capital of Germany?";
    const SPANISH: &str = "¿Cuál es la capital de Francia?";

    fn embedder(embedded: Arc<AtomicUsize>) -> FakeEmbedder {
        FakeEmbedder {
            vectors: HashMap::from([
                (FRANCE, vec![1.0, 0.0]),
                (PARAPHRASE, vec![0.99, 0.141]),
                (GERMANY, vec![0.97, 0.243]),
                (SPANISH, vec![0.0, 1.0]),
            ]),
            embedded,
        }
    }

    fn line(id: &str, prompt: &str) -> String {
        serde_json::json!({
            "id": id,
            "messages": [{"role": "user", "content": prompt}],
            "response": format!("answer to {id}"),
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        })
        .to_string()
    }

    fn log() -> Vec<LogRow> {
        let follow_up = serde_json::json!({
            "id": "r6",
            "messages": [
                {"role": "user", "content": FRANCE},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "And its population?"},
            ],
            "response": "About two million.",
        });
        let text = [
            line("r1", FRANCE),
            line("r2", FRANCE),
            line("r3", PARAPHRASE),
            line("r4", GERMANY),
            line("r5", SPANISH),
            follow_up.to_string(),
        ]
        .join("\n");
        parse_log(&text).unwrap()
    }

    #[tokio::test]
    async fn replays_in_order_with_known_decisions() {
        let embedder = Arc::new(embedder(Arc::default()));
        let replayed = replay(&log(), embedder, CacheConfig::default()).await;

        let outcome = |i: usize| replayed[i].decision.outcome.clone();
        assert_eq!(outcome(0), Outcome::Miss);
        assert_eq!(outcome(1), Outcome::ExactHit);
        assert_eq!(outcome(2), Outcome::SemanticHit);
        assert_eq!(outcome(3), Outcome::Miss);
        assert!(
            matches!(outcome(4), Outcome::Unsupported(_)),
            "{:?}",
            outcome(4)
        );
        assert_eq!(outcome(5), Outcome::Bypassed);

        let served: Vec<_> = replayed.iter().map(|r| r.served_from.as_deref()).collect();
        assert_eq!(served, [None, Some("r1"), Some("r1"), None, None, None]);

        let germany = &replayed[3].decision.candidates[0];
        assert!(
            germany.rejects.iter().any(|r| r.guard == "entity"),
            "{germany:?}"
        );
        assert_eq!(replayed[3].candidate_rows, [Some("r1".to_owned())]);
    }

    #[tokio::test]
    async fn embeds_each_text_once_across_replays() {
        let embedded = Arc::new(AtomicUsize::new(0));
        let cached = CachedEmbedder::new(embedder(embedded.clone()), usize::MAX);
        let cached: Arc<dyn Embedder> = Arc::new(cached);
        let rows = log();

        let first = replay(&rows, cached.clone(), CacheConfig::default()).await;
        let strict = CacheConfig {
            threshold: 0.995,
            ..CacheConfig::default()
        };
        let second = replay(&rows, cached, strict).await;

        // France, the paraphrase, Germany, and Spanish. Exact hits and the
        // bypassed follow-up embed nothing.
        assert_eq!(embedded.load(Ordering::SeqCst), 4);
        assert_eq!(first[2].decision.outcome, Outcome::SemanticHit);
        assert_eq!(second[2].decision.outcome, Outcome::Miss);
    }

    #[test]
    fn rows_get_defaults_and_parse_usage() {
        let rows = parse_log(&format!("\n{}\n", line("a", FRANCE))).unwrap();
        let row = &rows[0];
        assert_eq!(row.tenant, "default");
        assert_eq!(row.model, "");
        assert_eq!(row.response, b"answer to a");
        assert_eq!(
            row.usage,
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 20
            })
        );

        let bare = r#"{"messages":[{"role":"user","content":"hi"}],"response":{"text":"hi"}}"#;
        let row = &parse_log(bare).unwrap()[0];
        assert_eq!(row.id, "line-1");
        assert_eq!(row.response, br#"{"text":"hi"}"#);
        assert_eq!(row.usage, None);
    }

    #[test]
    fn bad_rows_name_their_line() {
        let cases = [
            (r#"{"messages":[]}"#, "line 1: missing response"),
            (r#"{"response":"x"}"#, "line 1: missing messages"),
            (
                r#"{"messages":[{"role":"user","content":[]}],"response":"x"}"#,
                "line 1: message content must be a string",
            ),
            (
                r#"{"messages":[],"response":"x","usage":{"prompt_tokens":1}}"#,
                "line 1: usage needs completion_tokens",
            ),
        ];
        for (text, error) in cases {
            assert_eq!(parse_log(text).unwrap_err().to_string(), error);
        }
        let ok = line("a", FRANCE);
        let error = parse_log(&format!("{ok}\n{ok}")).unwrap_err().to_string();
        assert_eq!(error, "line 2: duplicate id a");
    }
}
