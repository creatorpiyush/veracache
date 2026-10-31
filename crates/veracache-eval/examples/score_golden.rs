//! Precomputes the golden suite's similarity scores with the real model, so
//! measurements and CI need no model.
//!
//! Run with `cargo run -p veracache-eval --example score_golden` after any
//! change to `eval/datasets/golden.jsonl`.

use serde_json::json;
use veracache::{Embedder, LocalEmbedder, normalize};
use veracache_eval::{LoadError, load_pairs};

const DATASETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/datasets");

fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|x| *x /= norm);
    vector
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), LoadError> {
    let pairs = load_pairs(format!("{DATASETS}/golden.jsonl"))?;
    let embedder = LocalEmbedder::new(".fastembed_cache")?;

    let mut lines = String::new();
    for pair in &pairs {
        let texts = [normalize(&pair.a), normalize(&pair.b)];
        let vectors = embedder.embed(&texts).await?;
        let (a, b) = (unit(vectors[0].clone()), unit(vectors[1].clone()));
        let score: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let row = json!({
            "id": pair.id,
            "hash": pair.content_hash(),
            "score": (f64::from(score) * 1e6).round() / 1e6,
        });
        lines.push_str(&format!("{row}\n"));
    }

    let path = format!("{DATASETS}/golden.scores.{}.jsonl", embedder.id());
    std::fs::write(&path, lines)?;
    println!("wrote {} scores to {path}", pairs.len());
    Ok(())
}
