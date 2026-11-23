//! Prints decision records from a replay, for `replay` and `inspect`.

use std::fmt::Write;

use serde_json::{Value, json};
use veracache::Outcome;

use crate::Replayed;

/// A short name for how a lookup ended.
pub fn outcome_name(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Bypassed => "bypassed".into(),
        Outcome::ExactHit => "exact hit".into(),
        Outcome::SemanticHit => "semantic hit".into(),
        Outcome::Miss => "miss".into(),
        Outcome::Failed(error) => format!("failed: {error}"),
        Outcome::Unsupported(check) => format!("refused: {check}"),
    }
}

/// One line per row, then totals.
pub fn replay_text(replayed: &[Replayed], embedder: &str) -> String {
    let threshold = replayed.first().map_or(0.0, |r| r.decision.threshold);
    let mut out = format!(
        "veracache replay: {} rows, threshold {threshold:.2}, embedder {embedder}\n\n",
        replayed.len()
    );
    let width = replayed
        .iter()
        .map(|r| r.id.len())
        .max()
        .unwrap_or(2)
        .max(2);
    let _ = writeln!(
        out,
        "{:<width$}  {:<14}  {:<width$}  best score",
        "id", "outcome", "from"
    );
    let mut totals = [0usize; 6];
    for row in replayed {
        let best = row
            .decision
            .candidates
            .iter()
            .map(|c| c.score)
            .reduce(f32::max)
            .map_or("-".into(), |score| format!("{score:.4}"));
        let outcome = &row.decision.outcome;
        let slot = match outcome {
            Outcome::ExactHit => 0,
            Outcome::SemanticHit => 1,
            Outcome::Miss => 2,
            Outcome::Unsupported(_) => 3,
            Outcome::Bypassed => 4,
            Outcome::Failed(_) => 5,
        };
        totals[slot] += 1;
        let name = match outcome {
            Outcome::Unsupported(_) => "refused".to_owned(),
            Outcome::Failed(_) => "failed".to_owned(),
            other => outcome_name(other),
        };
        let _ = writeln!(
            out,
            "{:<width$}  {name:<14}  {:<width$}  {best}",
            row.id,
            row.served_from.as_deref().unwrap_or("-"),
        );
    }
    let [exact, semantic, miss, refused, bypassed, failed] = totals;
    let _ = writeln!(
        out,
        "\nexact hits {exact}, semantic hits {semantic}, misses {miss}, refused {refused}, \
         bypassed {bypassed}, failed {failed}"
    );
    out
}

/// The full decision record for one row.
pub fn decision_text(row: &Replayed) -> String {
    let decision = &row.decision;
    let mut out = format!("row {}: {}", row.id, outcome_name(&decision.outcome));
    if let Some(source) = &row.served_from {
        let _ = write!(out, ", served from {source}");
    }
    let _ = writeln!(out, "\nthreshold {:.2}", decision.threshold);
    if decision.candidates.is_empty() {
        out.push_str("no candidates\n");
        return out;
    }
    out.push_str("candidates, best first:\n");
    let name = |source: &Option<String>| source.as_deref().unwrap_or("unknown row").to_owned();
    let width = row
        .candidate_rows
        .iter()
        .map(|s| name(s).len())
        .max()
        .unwrap_or(0);
    for (candidate, source) in decision.candidates.iter().zip(&row.candidate_rows) {
        let status = if candidate.accepted {
            "accepted".to_owned()
        } else if candidate.score < decision.threshold {
            "below threshold".to_owned()
        } else if candidate.rejects.is_empty() {
            "not checked: an earlier candidate was accepted".to_owned()
        } else {
            let reasons: Vec<String> = candidate
                .rejects
                .iter()
                .map(|r| format!("{} ({})", r.guard, r.detail))
                .collect();
            format!("rejected: {}", reasons.join(", "))
        };
        let source = name(source);
        let _ = writeln!(out, "  {source:<width$}  {:.4}  {status}", candidate.score);
    }
    out
}

/// One row's decision record as JSON.
pub fn replayed_json(row: &Replayed) -> Value {
    let candidates: Vec<Value> = row
        .decision
        .candidates
        .iter()
        .zip(&row.candidate_rows)
        .map(|(candidate, source)| {
            let rejects: Vec<Value> = candidate
                .rejects
                .iter()
                .map(|r| json!({"guard": r.guard, "detail": r.detail}))
                .collect();
            json!({
                "row": source,
                "score": candidate.score,
                "accepted": candidate.accepted,
                "rejects": rejects,
            })
        })
        .collect();
    json!({
        "id": row.id,
        "outcome": outcome_name(&row.decision.outcome),
        "served_from": row.served_from,
        "threshold": row.decision.threshold,
        "candidates": candidates,
    })
}

#[cfg(test)]
mod tests {
    use veracache::{CandidateTrace, Decision, EntryId, LanguageCheck, RejectReason};

    use super::*;

    fn candidate(score: f32, accepted: bool, rejects: Vec<RejectReason>) -> CandidateTrace {
        CandidateTrace {
            id: EntryId(1),
            score,
            rejects,
            accepted,
        }
    }

    fn row(id: &str, outcome: Outcome, candidates: Vec<CandidateTrace>) -> Replayed {
        let candidate_rows = (0..candidates.len())
            .map(|i| Some(format!("src{i}")))
            .collect();
        Replayed {
            id: id.into(),
            served_from: matches!(outcome, Outcome::SemanticHit).then(|| "src1".to_owned()),
            decision: Decision {
                threshold: 0.88,
                candidates,
                outcome,
            },
            candidate_rows,
        }
    }

    fn semantic() -> Replayed {
        row(
            "query",
            Outcome::SemanticHit,
            vec![
                candidate(
                    0.97,
                    false,
                    vec![RejectReason::new("entity", "entities differ")],
                ),
                candidate(0.95, true, Vec::new()),
                candidate(0.90, false, Vec::new()),
                candidate(0.70, false, Vec::new()),
            ],
        )
    }

    #[test]
    fn decision_text_explains_every_candidate() {
        let text = decision_text(&semantic());
        assert!(
            text.starts_with("row query: semantic hit, served from src1\nthreshold 0.88\n"),
            "{text}"
        );
        assert!(
            text.contains("src0  0.9700  rejected: entity (entities differ)"),
            "{text}"
        );
        assert!(text.contains("src1  0.9500  accepted"), "{text}");
        assert!(text.contains("src2  0.9000  not checked"), "{text}");
        assert!(text.contains("src3  0.7000  below threshold"), "{text}");
    }

    #[test]
    fn refused_and_empty_records_say_so() {
        let refused = row(
            "es",
            Outcome::Unsupported(LanguageCheck::NonLatinScript),
            Vec::new(),
        );
        let text = decision_text(&refused);
        assert_eq!(
            text,
            "row es: refused: non-Latin script\nthreshold 0.88\nno candidates\n"
        );
    }

    #[test]
    fn replay_text_lists_rows_and_totals() {
        let rows = [
            row("first", Outcome::Miss, Vec::new()),
            semantic(),
            row(
                "es",
                Outcome::Unsupported(LanguageCheck::NonLatinScript),
                Vec::new(),
            ),
        ];
        let text = replay_text(&rows, "fake");
        assert!(
            text.starts_with("veracache replay: 3 rows, threshold 0.88, embedder fake"),
            "{text}"
        );
        assert!(
            text.contains("query  semantic hit    src1   0.9700"),
            "{text}"
        );
        assert!(text.contains("es     refused         -      -"), "{text}");
        assert!(
            text.contains(
                "exact hits 0, semantic hits 1, misses 1, refused 1, bypassed 0, failed 0"
            ),
            "{text}"
        );
    }

    #[test]
    fn json_carries_the_whole_record() {
        let json = replayed_json(&semantic());
        assert_eq!(json["outcome"], "semantic hit");
        assert_eq!(json["served_from"], "src1");
        assert_eq!(json["candidates"][0]["row"], "src0");
        assert_eq!(json["candidates"][0]["rejects"][0]["guard"], "entity");
        assert_eq!(json["candidates"][1]["accepted"], true);
    }
}
