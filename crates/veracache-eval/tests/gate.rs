//! CI gate: guarded false hits on the golden suite must not rise above the
//! committed baseline in `eval/baseline.json`.
//!
//! After an intended improvement, rewrite the baseline with
//! `UPDATE_BASELINE=1 cargo test -p veracache-eval --test gate`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use veracache::{EnglishAnalyzer, Guard, NumberGuard, default_guards};
use veracache_eval::{Label, load_pairs, load_scores, serves};

const DATASETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/datasets");
const BASELINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/baseline.json");
const EMBEDDER: &str = "bge-small-en-v1.5";

/// 0.88 is the default. 0.80 exercises the guards that similarity alone
/// hides at 0.88: entity, quote, and length.
const THRESHOLDS: [&str; 2] = ["0.80", "0.88"];

/// False hits by threshold, then by category, with an `all` total.
type FalseHits = BTreeMap<String, BTreeMap<String, u64>>;

fn false_hits(guards: &[Arc<dyn Guard>]) -> FalseHits {
    let pairs = load_pairs(format!("{DATASETS}/golden.jsonl")).unwrap();
    let scores = load_scores(format!("{DATASETS}/golden.scores.{EMBEDDER}.jsonl"), &pairs).unwrap();
    THRESHOLDS
        .iter()
        .map(|&threshold| {
            let value: f32 = threshold.parse().unwrap();
            let mut counts = BTreeMap::from([("all".to_owned(), 0)]);
            for pair in &pairs {
                let served = serves(pair, scores[&pair.id], value, &EnglishAnalyzer, guards);
                let false_hit = u64::from(pair.label == Label::Miss && served);
                *counts.entry(pair.category.clone()).or_default() += false_hit;
                *counts.get_mut("all").unwrap() += false_hit;
            }
            (threshold.to_owned(), counts)
        })
        .collect()
}

fn load_baseline() -> FalseHits {
    let text = std::fs::read_to_string(BASELINE)
        .expect("missing eval/baseline.json; run with UPDATE_BASELINE=1 to create it");
    let baseline: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        baseline["embedder"], EMBEDDER,
        "baseline is for another embedder"
    );
    serde_json::from_value(baseline["false_hits"].clone()).unwrap()
}

/// Every count above its baseline. A category or threshold missing from the
/// baseline counts as 0, so a new category cannot add false hits unnoticed.
fn regressions(baseline: &FalseHits, current: &FalseHits) -> Vec<String> {
    let mut found = Vec::new();
    for (threshold, counts) in current {
        for (category, &count) in counts {
            let allowed = baseline
                .get(threshold)
                .and_then(|counts| counts.get(category))
                .copied()
                .unwrap_or(0);
            if count > allowed {
                found.push(format!(
                    "{category} at {threshold}: {count} false hits, baseline {allowed}"
                ));
            }
        }
    }
    found
}

#[test]
fn golden_false_hits_do_not_regress() {
    let current = false_hits(&default_guards());
    if std::env::var_os("UPDATE_BASELINE").is_some() {
        let file = json!({"embedder": EMBEDDER, "false_hits": current});
        let text = serde_json::to_string_pretty(&file).unwrap() + "\n";
        std::fs::write(BASELINE, text).unwrap();
    }
    let found = regressions(&load_baseline(), &current);
    assert!(
        found.is_empty(),
        "golden false hits regressed:\n  {}",
        found.join("\n  ")
    );
}

#[test]
fn a_weakened_guard_fails_the_gate() {
    let weakened: Vec<Arc<dyn Guard>> = default_guards()
        .into_iter()
        .filter(|guard| guard.name() != NumberGuard.name())
        .collect();
    let found = regressions(&load_baseline(), &false_hits(&weakened));
    assert!(
        found.iter().any(|line| line.starts_with("number at ")),
        "{found:?}"
    );
    assert!(
        found.iter().any(|line| line.starts_with("all at ")),
        "{found:?}"
    );
}

#[test]
fn missing_baseline_entries_count_as_zero() {
    let current = FalseHits::from([(
        "0.88".to_owned(),
        BTreeMap::from([("brand_new".to_owned(), 1)]),
    )]);
    let found = regressions(&FalseHits::new(), &current);
    assert_eq!(found, ["brand_new at 0.88: 1 false hits, baseline 0"]);
}
