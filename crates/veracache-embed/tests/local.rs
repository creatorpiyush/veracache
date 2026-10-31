//! Runs the real model. Ignored by default because it downloads weights.
//! Run with `cargo test -p veracache-embed -- --ignored`.

#![cfg(feature = "local")]

use veracache_core::Embedder;
use veracache_embed::LocalEmbedder;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[tokio::test]
#[ignore = "downloads the model"]
async fn paraphrases_score_higher_than_unrelated_text() {
    let cache_dir = concat!(env!("CARGO_TARGET_TMPDIR"), "/fastembed");
    let embedder = LocalEmbedder::new(cache_dir).unwrap();

    let texts = [
        "what is the capital of France?",
        "tell me the capital city of France",
        "how do I bake sourdough bread?",
    ]
    .map(String::from);
    let vectors = embedder.embed(&texts).await.unwrap();

    assert_eq!(vectors.len(), 3);
    assert!(vectors.iter().all(|v| v.len() == embedder.dim()));
    let paraphrase = dot(&vectors[0], &vectors[1]);
    let unrelated = dot(&vectors[0], &vectors[2]);
    assert!(paraphrase > unrelated + 0.2, "{paraphrase} vs {unrelated}");
}

#[tokio::test]
#[ignore = "downloads the model"]
async fn reports_its_model_and_handles_empty_input() {
    let cache_dir = concat!(env!("CARGO_TARGET_TMPDIR"), "/fastembed");
    let embedder = LocalEmbedder::new(cache_dir).unwrap();

    assert_eq!(embedder.id(), "bge-small-en-v1.5");
    assert_eq!(embedder.dim(), 384);
    assert!(embedder.embed(&[]).await.unwrap().is_empty());
}

#[tokio::test]
#[ignore = "downloads the model"]
async fn identical_text_gives_identical_vectors() {
    let cache_dir = concat!(env!("CARGO_TARGET_TMPDIR"), "/fastembed");
    let embedder = LocalEmbedder::new(cache_dir).unwrap();
    let text = ["same text".to_string()];

    let first = embedder.embed(&text).await.unwrap();
    let second = embedder.embed(&text).await.unwrap();

    assert_eq!(first, second);
}
