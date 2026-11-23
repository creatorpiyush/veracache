//! Replays a log at many thresholds and recommends one for a target FHR.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use veracache::{CacheConfig, Embedder, Outcome};

use crate::{LogRow, Source, Verdict, replay};

/// Thresholds from `start` to `end` inclusive, `step` apart, without float
/// drift: each is rounded to 4 decimals.
pub fn thresholds(start: f32, end: f32, step: f32) -> Vec<f32> {
    if step <= 0.0 || end < start {
        return Vec::new();
    }
    let count = ((end - start) / step).round() as usize;
    (0..=count)
        .map(|i| ((start + i as f32 * step) * 1e4).round() / 1e4)
        .collect()
}

/// Price per million tokens, for estimating cost saved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pricing {
    /// Price per million prompt tokens.
    pub prompt: f64,
    /// Price per million completion tokens.
    pub completion: f64,
}

/// One threshold's replay, summarized.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SweepPoint {
    /// Similarity threshold replayed.
    pub threshold: f32,
    /// Rows replayed.
    pub rows: usize,
    /// Exact hits. Always correct: same prompt, same namespace.
    pub exact_hits: usize,
    /// Semantic hits.
    pub semantic_hits: usize,
    /// Semantic hits judged wrong.
    pub false_hits: usize,
    /// Prompt tokens saved by correct hits, from logged usage.
    pub prompt_tokens_saved: u64,
    /// Completion tokens saved by correct hits, from logged usage.
    pub completion_tokens_saved: u64,
    /// Candidate rejections by guard name, including `language`.
    pub rejects: BTreeMap<String, usize>,
    /// Lookups refused because the query's language is unsupported.
    pub language_refusals: usize,
    /// Semantic hits by what judged them, so a report can say how much rests
    /// on labels and how much on the heuristic.
    pub judged_by: BTreeMap<Source, usize>,
}

impl SweepPoint {
    /// All hits divided by rows replayed.
    pub fn hit_rate(&self) -> f32 {
        let hits = self.exact_hits + self.semantic_hits;
        if self.rows == 0 {
            0.0
        } else {
            hits as f32 / self.rows as f32
        }
    }

    /// False hits divided by semantic hits. `None` if no semantic hit was served.
    pub fn false_hit_rate(&self) -> Option<f32> {
        (self.semantic_hits > 0).then(|| self.false_hits as f32 / self.semantic_hits as f32)
    }

    /// Estimated cost saved by correct hits.
    pub fn cost_saved(&self, pricing: Pricing) -> f64 {
        (self.prompt_tokens_saved as f64 * pricing.prompt
            + self.completion_tokens_saved as f64 * pricing.completion)
            / 1e6
    }
}

/// Replays `rows` once per threshold, each over a fresh store, because a hit
/// changes what gets stored next.
///
/// `judge(query, source)` judges a semantic hit: `query` is the row that was
/// served, `source` the earlier row whose response it got. Usually
/// [`crate::Correctness::judge`]. Each pair is judged once and the verdict
/// reused at every threshold, so an LLM judge is called once per pair.
pub async fn sweep(
    rows: &[LogRow],
    embedder: Arc<dyn Embedder>,
    base: CacheConfig,
    thresholds: &[f32],
    judge: impl AsyncFn(&LogRow, &LogRow) -> Verdict,
) -> Vec<SweepPoint> {
    let by_id: HashMap<&str, &LogRow> = rows.iter().map(|row| (row.id.as_str(), row)).collect();
    let mut verdicts: HashMap<(&str, &str), Verdict> = HashMap::new();
    let mut points = Vec::with_capacity(thresholds.len());
    for &threshold in thresholds {
        let config = CacheConfig { threshold, ..base };
        let mut point = SweepPoint {
            threshold,
            rows: rows.len(),
            ..SweepPoint::default()
        };
        for (row, replayed) in rows
            .iter()
            .zip(replay(rows, embedder.clone(), config).await)
        {
            for reject in replayed.decision.candidates.iter().flat_map(|c| &c.rejects) {
                *point.rejects.entry(reject.guard.to_owned()).or_default() += 1;
            }
            let correct = match replayed.decision.outcome {
                Outcome::ExactHit => {
                    point.exact_hits += 1;
                    true
                }
                Outcome::SemanticHit => {
                    point.semantic_hits += 1;
                    let source = replayed.served_from.as_deref().and_then(|id| by_id.get(id));
                    // A hit whose source cannot be found cannot be shown correct.
                    let correct = match source {
                        Some(source) => {
                            let key = (row.id.as_str(), source.id.as_str());
                            let verdict = match verdicts.get(&key) {
                                Some(verdict) => *verdict,
                                None => *verdicts.entry(key).or_insert(judge(row, source).await),
                            };
                            *point.judged_by.entry(verdict.source).or_default() += 1;
                            verdict.correct
                        }
                        None => false,
                    };
                    point.false_hits += usize::from(!correct);
                    correct
                }
                Outcome::Unsupported(_) => {
                    point.language_refusals += 1;
                    false
                }
                _ => false,
            };
            if let (true, Some(usage)) = (correct, row.usage) {
                point.prompt_tokens_saved += u64::from(usage.prompt_tokens);
                point.completion_tokens_saved += u64::from(usage.completion_tokens);
            }
        }
        points.push(point);
    }
    points
}

/// The threshold with the highest hit rate whose FHR is at or below
/// `target_fhr`; ties go to the higher, stricter threshold.
///
/// A threshold that served no semantic hit has no FHR and never qualifies.
/// `None` means no threshold meets the target, and none should be used.
pub fn recommend(points: &[SweepPoint], target_fhr: f32) -> Option<f32> {
    points
        .iter()
        .filter(|point| point.false_hit_rate().is_some_and(|fhr| fhr <= target_fhr))
        .max_by(|a, b| {
            a.hit_rate()
                .total_cmp(&b.hit_rate())
                .then(a.threshold.total_cmp(&b.threshold))
        })
        .map(|point| point.threshold)
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use veracache::Error;

    use super::*;
    use crate::parse_log;

    struct FakeEmbedder(HashMap<&'static str, Vec<f32>>);

    #[async_trait]
    impl Embedder for FakeEmbedder {
        fn id(&self) -> &str {
            "fake"
        }

        fn dim(&self) -> usize {
            2
        }

        async fn embed(&self, texts: &[String]) -> veracache::Result<Vec<Vec<f32>>> {
            texts
                .iter()
                .map(|text| {
                    let vector = self.0.get(text.as_str()).cloned();
                    vector.ok_or_else(|| Error::Embed(format!("unknown text: {text}")))
                })
                .collect()
        }
    }

    const FRANCE: &str = "What is the capital of France?";
    const PARAPHRASE: &str = "What's the capital city of France?";
    const GERMANY: &str = "What is the capital of Germany?";
    const LAPTOP: &str = "What is the best laptop?";
    const PHONE: &str = "What is the best phone?";
    const SPANISH: &str = "¿Cuál es la capital de Francia?";

    fn embedder() -> Arc<dyn Embedder> {
        Arc::new(FakeEmbedder(HashMap::from([
            (FRANCE, vec![1.0, 0.0]),
            (PARAPHRASE, vec![0.99, 0.141]), // 0.99 to France
            (GERMANY, vec![0.97, 0.243]),    // 0.97 to France, stopped by EntityGuard
            (LAPTOP, vec![0.6, 0.8]),
            (PHONE, vec![0.1914, 0.9815]), // 0.90 to laptop, no guard sees it
            (SPANISH, vec![-1.0, 0.0]),
        ])))
    }

    fn rows() -> Vec<LogRow> {
        let lines: Vec<String> = [
            ("france", FRANCE),
            ("france-again", FRANCE),
            ("paraphrase", PARAPHRASE),
            ("germany", GERMANY),
            ("laptop", LAPTOP),
            ("phone", PHONE),
            ("spanish", SPANISH),
        ]
        .iter()
        .map(|(id, prompt)| {
            serde_json::json!({
                "id": id,
                "messages": [{"role": "user", "content": prompt}],
                "response": format!("answer to {id}"),
                "usage": {"prompt_tokens": 10, "completion_tokens": 100},
            })
            .to_string()
        })
        .collect();
        parse_log(&lines.join("\n")).unwrap()
    }

    // Laptop and phone need different answers; the paraphrase does not.
    async fn judge(query: &LogRow, source: &LogRow) -> Verdict {
        Verdict {
            correct: !(query.id == "phone" && source.id == "laptop"),
            source: Source::Label,
        }
    }

    async fn points() -> Vec<SweepPoint> {
        let thresholds = [0.85, 0.95, 0.995];
        sweep(
            &rows(),
            embedder(),
            CacheConfig::default(),
            &thresholds,
            judge,
        )
        .await
    }

    #[test]
    fn thresholds_cover_the_range_without_drift() {
        let range = thresholds(0.80, 0.98, 0.01);
        assert_eq!(range.len(), 19);
        assert_eq!(range[0], 0.80);
        assert_eq!(range[7], 0.87);
        assert_eq!(range[18], 0.98);
        assert!(thresholds(0.9, 0.8, 0.01).is_empty());
        assert!(thresholds(0.8, 0.9, 0.0).is_empty());
    }

    #[tokio::test]
    async fn each_threshold_is_replayed_and_summarized() {
        let points = points().await;
        let [low, high, strict] = &points[..] else {
            panic!("{points:?}");
        };

        assert_eq!(low.threshold, 0.85);
        assert_eq!(
            (low.exact_hits, low.semantic_hits, low.false_hits),
            (1, 2, 1)
        );
        assert_eq!(low.false_hit_rate(), Some(0.5));
        assert_eq!(low.rejects.get("entity"), Some(&1));
        assert_eq!(low.language_refusals, 1);
        assert_eq!(low.judged_by, BTreeMap::from([(Source::Label, 2)]));
        // The exact hit and the paraphrase; the false hit saves nothing.
        assert_eq!(low.prompt_tokens_saved, 20);
        assert_eq!(low.completion_tokens_saved, 200);

        assert_eq!(
            (high.exact_hits, high.semantic_hits, high.false_hits),
            (1, 1, 0)
        );
        assert_eq!(high.false_hit_rate(), Some(0.0));
        assert!((high.hit_rate() - 2.0 / 7.0).abs() < 1e-6);

        assert_eq!((strict.semantic_hits, strict.false_hit_rate()), (0, None));
        assert_eq!(strict.rejects.get("entity"), None);
    }

    #[tokio::test]
    async fn each_pair_is_judged_once_across_thresholds() {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let counting = async |query: &LogRow, source: &LogRow| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            judge(query, source).await
        };
        let thresholds = [0.85, 0.86, 0.87];
        sweep(
            &rows(),
            embedder(),
            CacheConfig::default(),
            &thresholds,
            counting,
        )
        .await;
        // Paraphrase-to-France and phone-to-laptop, judged at the first
        // threshold and reused at the other two.
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn recommends_the_best_threshold_that_meets_the_target() {
        let points = points().await;
        assert_eq!(recommend(&points, 0.0), Some(0.95));
        assert_eq!(recommend(&points, 0.5), Some(0.85));
    }

    #[tokio::test]
    async fn unmeetable_target_gives_no_recommendation() {
        let points = points().await;
        assert_eq!(recommend(&points[..1], 0.1), None);
        // A threshold that serves no semantic hit does not count as safe.
        assert_eq!(recommend(&points[2..], 0.0), None);
        assert_eq!(recommend(&[], 0.0), None);
    }

    #[test]
    fn ties_go_to_the_stricter_threshold() {
        let point = |threshold| SweepPoint {
            threshold,
            rows: 10,
            semantic_hits: 2,
            ..SweepPoint::default()
        };
        assert_eq!(recommend(&[point(0.90), point(0.92)], 0.0), Some(0.92));
    }

    #[test]
    fn cost_uses_prices_per_million() {
        let point = SweepPoint {
            prompt_tokens_saved: 2_000_000,
            completion_tokens_saved: 1_000_000,
            ..SweepPoint::default()
        };
        let pricing = Pricing {
            prompt: 0.15,
            completion: 0.60,
        };
        assert!((point.cost_saved(pricing) - 0.90).abs() < 1e-9);
    }
}
