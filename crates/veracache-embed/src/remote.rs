use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};
use veracache_core::{Embedder, Error, Result};

const MAX_BATCH: usize = 256;
const TIMEOUT: Duration = Duration::from_secs(30);

/// Settings for an OpenAI-compatible embeddings API.
#[derive(Clone)]
pub struct RemoteConfig {
    /// API root, such as `https://api.openai.com/v1`.
    pub base_url: String,
    /// Bearer token. `None` for servers that need no key.
    pub api_key: Option<String>,
    /// Model name. Also the embedder id, so it is part of the namespace.
    pub model: String,
    /// Expected vector length.
    pub dim: usize,
}

/// Embeds text through an OpenAI-compatible `/embeddings` endpoint.
pub struct RemoteEmbedder {
    client: Client,
    url: String,
    api_key: Option<String>,
    model: String,
    dim: usize,
}

impl RemoteEmbedder {
    /// Creates an embedder. No request is sent until the first `embed`.
    pub fn new(config: RemoteConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(TIMEOUT)
            .build()
            .map_err(http_error)?;
        Ok(Self {
            client,
            url: format!("{}/embeddings", config.base_url.trim_end_matches('/')),
            api_key: config.api_key,
            model: config.model,
            dim: config.dim,
        })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut request = self
            .client
            .post(&self.url)
            .json(&json!({"model": self.model, "input": texts}));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .and_then(|response| response.error_for_status())
            .map_err(http_error)?;
        let body: Value = response.json().await.map_err(http_error)?;
        parse_vectors(&body, texts.len(), self.dim)
    }
}

#[async_trait]
impl Embedder for RemoteEmbedder {
    fn id(&self) -> &str {
        &self.model
    }

    fn dim(&self) -> usize {
        self.dim
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH) {
            vectors.extend(self.embed_batch(batch).await?);
        }
        Ok(vectors)
    }
}

// Places vectors by `index`, because the API does not promise response order.
fn parse_vectors(body: &Value, expected: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    let bad = |detail: &str| Error::Embed(format!("bad embeddings response: {detail}"));
    let data = body["data"].as_array().ok_or_else(|| bad("no data"))?;
    if data.len() != expected {
        return Err(bad("wrong number of vectors"));
    }

    let mut slots = vec![None; expected];
    for item in data {
        let slot = item["index"]
            .as_u64()
            .and_then(|index| slots.get_mut(index as usize))
            .ok_or_else(|| bad("invalid index"))?;
        let vector = item["embedding"]
            .as_array()
            .ok_or_else(|| bad("no embedding"))?
            .iter()
            .map(|x| x.as_f64().map(|x| x as f32))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| bad("non-numeric embedding"))?;
        if vector.len() != dim {
            return Err(bad("dimension mismatch"));
        }
        if slot.replace(vector).is_some() {
            return Err(bad("duplicate index"));
        }
    }
    // data.len() == expected and no index repeats, so every slot is filled.
    Ok(slots.into_iter().flatten().collect())
}

fn http_error(error: reqwest::Error) -> Error {
    Error::Embed(error.without_url().to_string())
}
