//! Compares threshold-only and guarded results on the golden suite.
//!
//! Run with `cargo run -p veracache-eval --example guarded`. It uses the
//! tuning split at 0.90. Pass `held_out` only for the final measurement, and
//! a threshold as the second argument to change it.

use std::collections::BTreeMap;

use veracache::{EnglishAnalyzer, default_guards};
use veracache_eval::{Counts, Label, LoadError, Split, load_pairs, load_scores, serves};

const DATASETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/datasets");
const EMBEDDER: &str = "bge-small-en-v1.5";

fn percent(rate: Option<f32>) -> String {
    rate.map_or("n/a".into(), |rate| format!("{:.1}%", rate * 100.0))
}

fn row(name: &str, plain: &Counts, guarded: &Counts) {
    println!(
        "{name:<10}  {:>7} -> {:>6}  {:>7} -> {:>6}  {:>3} -> {:>3}",
        percent(plain.false_hit_rate()),
        percent(guarded.false_hit_rate()),
        percent(plain.hit_recall()),
        percent(guarded.hit_recall()),
        plain.false_hits,
        guarded.false_hits,
    );
}

fn main() -> Result<(), LoadError> {
    let split = match std::env::args().nth(1).as_deref() {
        None | Some("tuning") => Split::Tuning,
        Some("held_out") => Split::HeldOut,
        Some(other) => return Err(format!("unknown split: {other}").into()),
    };

    let threshold: f32 = std::env::args().nth(2).map_or(Ok(0.90), |t| t.parse())?;

    let pairs = load_pairs(format!("{DATASETS}/golden.jsonl"))?;
    let scores = load_scores(format!("{DATASETS}/golden.scores.{EMBEDDER}.jsonl"), &pairs)?;
    let guards = default_guards();

    let mut categories: BTreeMap<&str, (Counts, Counts)> = BTreeMap::new();
    let (mut plain, mut guarded) = (Counts::default(), Counts::default());
    for pair in pairs.iter().filter(|pair| pair.split == split) {
        let score = scores[&pair.id];
        let served = serves(pair, score, threshold, &EnglishAnalyzer, &guards);
        let category = categories.entry(&pair.category).or_default();
        category.0.record(pair.label, score >= threshold);
        category.1.record(pair.label, served);
        plain.record(pair.label, score >= threshold);
        guarded.record(pair.label, served);
        if pair.label == Label::Hit && score >= threshold && !served {
            println!("lost hit: {} | {} | {}", pair.id, pair.a, pair.b);
        }
    }

    println!("\n{split:?} split at {threshold:.2}, {EMBEDDER}");
    println!("category    false-hit rate      hit recall          false hits");
    for (name, (plain, guarded)) in &categories {
        row(name, plain, guarded);
    }
    row("all", &plain, &guarded);
    Ok(())
}
