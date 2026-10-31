//! Keeps `eval/datasets/golden.jsonl` well formed and balanced.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use veracache::{EnglishAnalyzer, default_guards};
use veracache_eval::{Counts, Pair, Split, load_pairs, load_scores, serves};

const DATASETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/datasets");

fn loaded() -> Vec<Pair> {
    load_pairs(format!("{DATASETS}/golden.jsonl")).unwrap()
}

fn scores_path() -> String {
    format!("{DATASETS}/golden.scores.bge-small-en-v1.5.jsonl")
}

#[test]
fn scores_cover_every_pair_and_are_fresh() {
    let pairs = loaded();
    let scores = load_scores(scores_path(), &pairs).unwrap();
    assert_eq!(scores.len(), pairs.len());
    assert!(scores.values().all(|s| (-1.0..=1.01).contains(s)));
}

#[test]
fn edited_pair_makes_scores_stale() {
    let mut pairs = loaded();
    pairs[0].b.push('!');
    let error = load_scores(scores_path(), &pairs).unwrap_err();
    assert!(error.to_string().contains("stale score"), "{error}");
}

#[test]
fn missing_and_unknown_pairs_are_errors() {
    let mut pairs = loaded();
    let removed = pairs.pop().unwrap();
    let error = load_scores(scores_path(), &pairs).unwrap_err();
    assert!(error.to_string().contains(&removed.id), "{error}");

    let mut pairs = loaded();
    let mut extra = pairs[0].clone();
    extra.id = "extra".into();
    pairs.push(extra);
    let error = load_scores(scores_path(), &pairs).unwrap_err();
    assert!(error.to_string().contains("extra"), "{error}");
}

// The number guards must beat. If this changes, the scores or pairs changed.
#[test]
fn threshold_only_baseline_is_pinned() {
    let pairs = loaded();
    let scores = load_scores(scores_path(), &pairs).unwrap();
    let mut counts = Counts::default();
    for pair in pairs.iter().filter(|pair| pair.split == Split::Tuning) {
        counts.record(pair.label, scores[&pair.id] >= 0.90);
    }
    assert_eq!(counts.false_hits, 36);
    assert_eq!(counts.correct_hits, 72);
    assert_eq!(counts.should_hit, 80);
}

/// Tuning-split outcomes for one category, with and without guards, at 0.90.
fn tuning_counts(category: &str) -> (Counts, Counts) {
    let pairs = loaded();
    let scores = load_scores(scores_path(), &pairs).unwrap();
    let guards = default_guards();
    let (mut plain, mut guarded) = (Counts::default(), Counts::default());
    for pair in &pairs {
        if pair.split == Split::Tuning && pair.category == category {
            let score = scores[&pair.id];
            plain.record(pair.label, score >= 0.90);
            let served = serves(pair, score, 0.90, &EnglishAnalyzer, &guards);
            guarded.record(pair.label, served);
        }
    }
    (plain, guarded)
}

#[test]
fn number_guard_stops_number_false_hits_and_keeps_true_hits() {
    let (plain, guarded) = tuning_counts("number");
    assert!(plain.false_hits > 0);
    assert_eq!(guarded.false_hits, 0);
    assert_eq!(guarded.correct_hits, plain.correct_hits);
}

#[test]
fn date_guard_stops_date_false_hits_and_keeps_true_hits() {
    let (plain, guarded) = tuning_counts("date");
    assert!(plain.false_hits > 0);
    assert_eq!(guarded.false_hits, 0);
    assert_eq!(guarded.correct_hits, plain.correct_hits);
}

#[test]
fn negation_guard_stops_negation_false_hits_and_keeps_true_hits() {
    let (plain, guarded) = tuning_counts("negation");
    assert!(plain.false_hits > 0);
    assert_eq!(guarded.false_hits, 0);
    assert_eq!(guarded.correct_hits, plain.correct_hits);
}

#[test]
fn direction_guard_stops_direction_false_hits_and_keeps_true_hits() {
    let (plain, guarded) = tuning_counts("direction");
    assert!(plain.false_hits > 0);
    assert_eq!(guarded.false_hits, 0);
    assert_eq!(guarded.correct_hits, plain.correct_hits);
}

const CATEGORIES: [&str; 8] = [
    "number",
    "date",
    "negation",
    "direction",
    "entity",
    "quote",
    "length",
    "follow_up",
];

fn pairs() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../eval/datasets/golden.jsonl"
    );
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn text<'a>(pair: &'a Value, field: &str) -> &'a str {
    pair[field]
        .as_str()
        .unwrap_or_else(|| panic!("{field}: {pair}"))
}

#[test]
fn every_pair_has_a_valid_schema() {
    for pair in pairs() {
        assert!(CATEGORIES.contains(&text(&pair, "category")), "{pair}");
        assert!(matches!(text(&pair, "label"), "hit" | "miss"), "{pair}");
        assert!(
            matches!(text(&pair, "split"), "tuning" | "held_out"),
            "{pair}"
        );
        assert!(!text(&pair, "a").is_empty(), "{pair}");
        assert!(!text(&pair, "b").is_empty(), "{pair}");
    }
}

#[test]
fn ids_are_unique() {
    let pairs = pairs();
    let ids: HashSet<_> = pairs.iter().map(|pair| text(pair, "id")).collect();
    assert_eq!(ids.len(), pairs.len());
}

#[test]
fn suite_has_at_least_200_pairs_half_of_each_label() {
    let pairs = pairs();
    let hits = pairs.iter().filter(|p| text(p, "label") == "hit").count();
    assert!(pairs.len() >= 200, "{}", pairs.len());
    assert_eq!(hits * 2, pairs.len());
}

#[test]
fn every_category_is_balanced() {
    let mut counts: HashMap<(&str, &str), usize> = HashMap::new();
    let pairs = pairs();
    for pair in &pairs {
        *counts
            .entry((text(pair, "category"), text(pair, "label")))
            .or_default() += 1;
    }
    for category in CATEGORIES {
        let hits = counts[&(category, "hit")];
        assert!(hits >= 10, "{category}");
        assert_eq!(hits, counts[&(category, "miss")], "{category}");
    }
}

#[test]
fn every_category_has_held_out_pairs_of_both_labels() {
    let pairs = pairs();
    for category in CATEGORIES {
        for label in ["hit", "miss"] {
            let held_out = pairs.iter().any(|p| {
                text(p, "category") == category
                    && text(p, "label") == label
                    && text(p, "split") == "held_out"
            });
            assert!(held_out, "{category} {label}");
        }
    }
}

#[test]
fn entity_category_includes_lowercase_misses() {
    let lowercase = pairs()
        .iter()
        .filter(|p| text(p, "category") == "entity" && text(p, "label") == "miss")
        .filter(|p| {
            let (a, b) = (text(p, "a"), text(p, "b"));
            a == a.to_lowercase() && b == b.to_lowercase()
        })
        .count();
    assert!(lowercase >= 5, "{lowercase}");
}

#[test]
fn only_follow_ups_have_history_and_miss_pairs_differ() {
    for pair in pairs() {
        let is_follow_up = text(&pair, "category") == "follow_up";
        assert_eq!(pair["a_history"].is_array(), is_follow_up, "{pair}");
        assert_eq!(pair["b_history"].is_array(), is_follow_up, "{pair}");

        let same_text = text(&pair, "a") == text(&pair, "b");
        if !is_follow_up {
            assert!(!same_text, "{pair}");
        } else if text(&pair, "label") == "miss" {
            assert_ne!(pair["a_history"], pair["b_history"], "{pair}");
        }
    }
}
