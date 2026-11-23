//! An LLM judge, through an OpenAI-compatible chat completions API.

use std::time::Duration;

use reqwest::Client;
use serde_json::{Value, json};

use crate::correctness::response_text;
use crate::{LoadError, LogRow};

const TIMEOUT: Duration = Duration::from_secs(60);

const INSTRUCTIONS: &str = "You check a semantic cache. A user asked a question. \
The reference answer is what the model answered for this exact question. The cached \
answer was given to a similar earlier question. Decide if the cached answer is a \
correct and complete answer to the user's question: same facts, same numbers, same \
entities, and the same yes or no. Wording may differ. Reply with one word: yes or no.";

/// Settings for an OpenAI-compatible chat completions API.
#[derive(Debug, Clone)]
pub struct JudgeConfig {
    /// API root, such as `https://api.openai.com/v1`.
    pub base_url: String,
    /// Bearer token. `None` for servers that need no key.
    pub api_key: Option<String>,
    /// Judge model name.
    pub model: String,
}

/// Asks a model whether a cached response answers a query.
pub struct LlmJudge {
    client: Client,
    url: String,
    api_key: Option<String>,
    model: String,
}

impl LlmJudge {
    /// Creates a judge. No request is sent until the first [`LlmJudge::judge`].
    pub fn new(config: JudgeConfig) -> Result<Self, LoadError> {
        let client = Client::builder().timeout(TIMEOUT).build()?;
        Ok(Self {
            client,
            url: format!("{}/chat/completions", config.base_url.trim_end_matches('/')),
            api_key: config.api_key,
            model: config.model,
        })
    }

    /// The judge model's name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// True if `source`'s response correctly answers `query`. An error means
    /// the judge could not decide; callers must not count it as correct.
    pub async fn judge(&self, query: &LogRow, source: &LogRow) -> Result<bool, String> {
        let question = query
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .map_or("", |message| message.content.as_str());
        let prompt = format!(
            "Question:\n{question}\n\nReference answer:\n{}\n\nCached answer:\n{}",
            response_text(&query.response),
            response_text(&source.response),
        );
        let mut request = self.client.post(&self.url).json(&json!({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        }));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let body: Value = request
            .send()
            .await
            .and_then(|response| response.error_for_status())
            .map_err(|error| error.without_url().to_string())?
            .json()
            .await
            .map_err(|error| error.without_url().to_string())?;
        let reply = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("judge reply has no content")?;
        let word = reply
            .trim()
            .trim_end_matches(['.', '!'])
            .to_ascii_lowercase();
        match word.as_str() {
            "yes" => Ok(true),
            "no" => Ok(false),
            _ => Err(format!("judge reply is not yes or no: {reply:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{bearer_token, body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::{Correctness, Labels, Source, Verdict, parse_log};

    fn rows() -> Vec<LogRow> {
        let text = [
            serde_json::json!({
                "id": "query",
                "messages": [{"role": "user", "content": "How many moons does Mars have?"}],
                "response": "Mars has 2 moons.",
            }),
            serde_json::json!({
                "id": "source",
                "messages": [{"role": "user", "content": "How many moons orbit Mars?"}],
                "response": "Mars has two moons.",
            }),
        ]
        .map(|row| row.to_string())
        .join("\n");
        parse_log(&text).unwrap()
    }

    fn reply(text: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": text}}],
        }))
    }

    async fn server(response: ResponseTemplate, calls: u64) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(bearer_token("secret"))
            .and(body_partial_json(
                serde_json::json!({"model": "judge-model", "temperature": 0}),
            ))
            .respond_with(response)
            .expect(calls)
            .mount(&server)
            .await;
        server
    }

    fn correctness(server: &MockServer, labels: Labels) -> Correctness {
        let judge = LlmJudge::new(JudgeConfig {
            base_url: format!("{}/", server.uri()),
            api_key: Some("secret".into()),
            model: "judge-model".into(),
        })
        .unwrap();
        Correctness::new(labels).with_judge(judge)
    }

    async fn verdict(response: ResponseTemplate) -> Verdict {
        let server = server(response, 1).await;
        let rows = rows();
        correctness(&server, Labels::default())
            .judge(&rows[0], &rows[1])
            .await
    }

    #[tokio::test]
    async fn yes_and_no_replies_decide_the_verdict() {
        let yes = verdict(reply("Yes")).await;
        assert_eq!((yes.correct, yes.source), (true, Source::Judge));
        let no = verdict(reply(" No.\n")).await;
        assert_eq!((no.correct, no.source), (false, Source::Judge));
    }

    #[tokio::test]
    async fn judge_failures_are_reported_and_never_correct() {
        for response in [
            ResponseTemplate::new(500),
            reply("Probably, mostly yes"),
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices": []})),
        ] {
            let failed = verdict(response).await;
            assert_eq!((failed.correct, failed.source), (false, Source::JudgeError));
        }
    }

    #[tokio::test]
    async fn the_judge_sees_the_question_and_both_answers() {
        let server = server(reply("yes"), 1).await;
        let rows = rows();
        correctness(&server, Labels::default())
            .judge(&rows[0], &rows[1])
            .await;
        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        let prompt = body["messages"][1]["content"].as_str().unwrap();
        assert!(
            prompt.contains("How many moons does Mars have?"),
            "{prompt}"
        );
        assert!(
            prompt.contains("Reference answer:\nMars has 2 moons."),
            "{prompt}"
        );
        assert!(
            prompt.contains("Cached answer:\nMars has two moons."),
            "{prompt}"
        );
    }

    #[tokio::test]
    async fn labels_are_used_before_the_judge() {
        let server = server(reply("yes"), 0).await;
        let rows = rows();
        let labels = Labels::parse(r#"{"a":"query","b":"source","label":"miss"}"#, &rows).unwrap();
        let verdict = correctness(&server, labels).judge(&rows[0], &rows[1]).await;
        assert_eq!((verdict.correct, verdict.source), (false, Source::Label));
    }
}
