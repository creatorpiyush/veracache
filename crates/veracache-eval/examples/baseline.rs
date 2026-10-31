//! Reports threshold-only results on the golden suite: what a plain semantic
//! cache would do, with no guards and no namespaces.
//!
//! Run with `cargo run -p veracache-eval --example baseline`. It uses the
//! tuning split. Pass `held_out` only for the final measurement.

use std::collections::BTreeMap;

use veracache_eval::{Counts, LoadError, Split, load_pairs, load_scores};

const DATASETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/datasets");
const EMBEDDER: &str = "bge-small-en-v1.5";
const REPORT_THRESHOLD: f32 = 0.90;

fn percent(rate: Option<f32>) -> String {
    rate.map_or("n/a".into(), |rate| format!("{:.1}%", rate * 100.0))
}

fn main() -> Result<(), LoadError> {
    let split = match std::env::args().nth(1).as_deref() {
        None | Some("tuning") => Split::Tuning,
        Some("held_out") => Split::HeldOut,
        Some(other) => return Err(format!("unknown split: {other}").into()),
    };

    let pairs = load_pairs(format!("{DATASETS}/golden.jsonl"))?;
    let scores = load_scores(format!("{DATASETS}/golden.scores.{EMBEDDER}.jsonl"), &pairs)?;
    let selected: Vec<_> = pairs.iter().filter(|pair| pair.split == split).collect();
    println!("{split:?} split, {} pairs, {EMBEDDER}\n", selected.len());

    println!("threshold  false-hit rate  hit recall  false hits  correct hits");
    for step in 0..=9 {
        let threshold = 0.80 + step as f32 * 0.02;
        let mut counts = Counts::default();
        for pair in &selected {
            counts.record(pair.label, scores[&pair.id] >= threshold);
        }
        println!(
            "{threshold:>9.2}  {:>14}  {:>10}  {:>10}  {:>12}",
            percent(counts.false_hit_rate()),
            percent(counts.hit_recall()),
            counts.false_hits,
            counts.correct_hits,
        );
    }

    let mut categories: BTreeMap<&str, Counts> = BTreeMap::new();
    for pair in &selected {
        categories
            .entry(&pair.category)
            .or_default()
            .record(pair.label, scores[&pair.id] >= REPORT_THRESHOLD);
    }
    println!("\nby category at {REPORT_THRESHOLD:.2}");
    println!("category    false-hit rate  hit recall");
    for (category, counts) in categories {
        println!(
            "{category:<10}  {:>14}  {:>10}",
            percent(counts.false_hit_rate()),
            percent(counts.hit_recall()),
        );
    }
    Ok(())
}
