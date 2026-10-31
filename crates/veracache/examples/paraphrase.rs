//! Caches an answer, then serves it for a paraphrase.
//!
//! Run with `cargo run -p veracache --example paraphrase`. The first run
//! downloads the embedding model to `.fastembed_cache`.

use std::sync::Arc;

use serde_json::Value;
use veracache::{
    CacheConfig, CacheRequest, CachedEmbedder, ChatMessage, LocalEmbedder, ResponseMeta,
    SetOptions, memory_cache,
};

fn request<'a>(messages: &'a [ChatMessage]) -> CacheRequest<'a> {
    CacheRequest {
        tenant: "example",
        model: "gpt-example",
        messages,
        tools: &Value::Null,
        response_format: &Value::Null,
        version_tag: "",
    }
}

fn user(content: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: content.into(),
    }]
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> veracache::Result<()> {
    let embedder = CachedEmbedder::new(LocalEmbedder::new(".fastembed_cache")?, 1_000);
    let cache = memory_cache(Arc::new(embedder), CacheConfig::default());

    let question = user("What is the capital of France?");
    match cache.get(&request(&question)).await {
        Some(_) => println!("HIT   (unexpected on an empty cache)"),
        None => println!("MISS  What is the capital of France?"),
    }

    let answer = b"Paris".to_vec();
    let meta = ResponseMeta::default();
    cache
        .set(&request(&question), answer, meta, SetOptions::default())
        .await;

    for prompt in [
        "What is the capital of France?",
        "What's the capital city of France?",
        "How do I bake sourdough bread?",
    ] {
        let messages = user(prompt);
        match cache.get(&request(&messages)).await {
            Some(hit) => {
                let kind = if hit.exact { "exact" } else { "semantic" };
                let answer = String::from_utf8_lossy(&hit.entry.response);
                println!("HIT   {prompt} -> {answer} ({kind}, {:.3})", hit.score);
            }
            None => println!("MISS  {prompt}"),
        }
    }
    Ok(())
}
