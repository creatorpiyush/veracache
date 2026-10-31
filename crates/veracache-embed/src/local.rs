use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use veracache_core::{Embedder, Error, Result};

/// Embeds text in-process with ONNX Runtime, using bge-small-en-v1.5.
pub struct LocalEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
}

impl LocalEmbedder {
    /// Loads the model from `cache_dir`, downloading it first if absent. A
    /// directory that already holds the model needs no network.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let options = TextInitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_cache_dir(cache_dir.into())
            .with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).map_err(embed_error)?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    fn id(&self) -> &str {
        "bge-small-en-v1.5"
    }

    fn dim(&self) -> usize {
        384
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let texts = texts.to_vec();
        tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|_| Error::Embed("lock poisoned".into()))?;
            model.embed(texts, None).map_err(embed_error)
        })
        .await
        .map_err(embed_error)?
    }
}

fn embed_error(error: impl std::fmt::Display) -> Error {
    Error::Embed(error.to_string())
}
