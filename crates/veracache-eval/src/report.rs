//! Renders a sweep as a terminal table or JSON.

use std::fmt::Write;

use serde_json::{Value, json};

use crate::{Pricing, Source, SweepPoint, recommend};

/// A finished sweep, ready to print.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Embedder id used for the replay.
    pub embedder: String,
    /// Rows in the replayed log.
    pub rows: usize,
    /// Highest acceptable false-hit rate.
    pub target_fhr: f32,
    /// Prices for the cost column, if given.
    pub pricing: Option<Pricing>,
    /// One point per threshold, in sweep order.
    pub points: Vec<SweepPoint>,
}

fn percent(value: f32) -> String {
    format!("{:.1}%", value * 100.0)
}

fn source_name(source: Source) -> &'static str {
    match source {
        Source::Label => "labelled",
        Source::Judge => "LLM judge",
        Source::JudgeError => "judge errors",
        Source::Heuristic => "heuristic",
    }
}

const HEURISTIC_WARNING: &str = "Heuristic verdicts compare the logged responses and are \
    estimates. Add --labels or --judge for real numbers.";
const JUDGE_ERROR_WARNING: &str = "Judge errors are counted as false hits.";

fn judged_summary(point: &SweepPoint) -> String {
    let judged: Vec<String> = point
        .judged_by
        .iter()
        .map(|(source, count)| format!("{count} {}", source_name(*source)))
        .collect();
    if judged.is_empty() {
        "no semantic hits to judge".to_owned()
    } else {
        judged.join(", ")
    }
}

fn rejects_summary(point: &SweepPoint) -> String {
    let rejects: Vec<String> = point
        .rejects
        .iter()
        .map(|(guard, count)| format!("{guard} {count}"))
        .collect();
    if rejects.is_empty() {
        "none".to_owned()
    } else {
        rejects.join(", ")
    }
}

fn warnings(point: &SweepPoint) -> Vec<&'static str> {
    let mut warnings = Vec::new();
    if point.judged_by.contains_key(&Source::Heuristic) {
        warnings.push(HEURISTIC_WARNING);
    }
    if point.judged_by.contains_key(&Source::JudgeError) {
        warnings.push(JUDGE_ERROR_WARNING);
    }
    warnings
}

/// Escapes text for HTML element content and attribute values.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

impl Report {
    fn no_recommendation(&self) -> String {
        format!(
            "No threshold meets FHR <= {}. Do not enable semantic hits for this traffic; \
             exact hits are still safe.",
            percent(self.target_fhr)
        )
    }

    /// The recommended threshold, or `None` if no threshold meets the target.
    pub fn recommended(&self) -> Option<f32> {
        recommend(&self.points, self.target_fhr)
    }

    /// The point shown in detail: the recommended one, else the strictest.
    fn detail(&self) -> Option<&SweepPoint> {
        match self.recommended() {
            Some(threshold) => self.points.iter().find(|p| p.threshold == threshold),
            None => self.points.last(),
        }
    }

    /// A plain-text report for a terminal.
    pub fn text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "veracache eval: {} rows, embedder {}, target FHR {}\n",
            self.rows,
            self.embedder,
            percent(self.target_fhr)
        );
        let cost = if self.pricing.is_some() {
            "  cost saved"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "threshold  hit rate  exact  semantic  false  FHR     tokens saved{cost}"
        );
        for point in &self.points {
            let fhr = point.false_hit_rate().map_or("-".into(), percent);
            let tokens = point.prompt_tokens_saved + point.completion_tokens_saved;
            let cost = self
                .pricing
                .map_or(String::new(), |p| format!("  ${:.4}", point.cost_saved(p)));
            let _ = writeln!(
                out,
                "{:<9.2}  {:>8}  {:>5}  {:>8}  {:>5}  {:>6}  {:>12}{cost}",
                point.threshold,
                percent(point.hit_rate()),
                point.exact_hits,
                point.semantic_hits,
                point.false_hits,
                fhr,
                tokens,
            );
        }
        out.push('\n');
        match (self.recommended(), self.detail()) {
            (Some(threshold), Some(point)) => {
                let fhr = point.false_hit_rate().map_or("-".into(), percent);
                let _ = writeln!(
                    out,
                    "Recommended threshold: {threshold:.2} (hit rate {}, FHR {fhr})",
                    percent(point.hit_rate())
                );
            }
            _ => {
                let _ = writeln!(out, "{}", self.no_recommendation());
            }
        }
        let Some(point) = self.detail() else {
            return out;
        };
        let _ = writeln!(out, "\nAt {:.2}:", point.threshold);
        let _ = writeln!(out, "  correctness: {}", judged_summary(point));
        let _ = writeln!(out, "  guard rejections: {}", rejects_summary(point));
        let _ = writeln!(out, "  language refusals: {}", point.language_refusals);
        for warning in warnings(point) {
            let _ = write!(out, "\n{warning}\n");
        }
        out
    }

    /// The same report as one self-contained HTML page: inline styles and an
    /// inline SVG chart, with no scripts and no external resources.
    pub fn html(&self) -> String {
        let recommended = self.recommended();
        let cost_head = if self.pricing.is_some() {
            "<th>Cost saved</th>"
        } else {
            ""
        };
        let mut rows = String::new();
        for point in &self.points {
            let class = if Some(point.threshold) == recommended {
                " class=\"rec\""
            } else {
                ""
            };
            let fhr = point.false_hit_rate().map_or("-".into(), percent);
            let tokens = point.prompt_tokens_saved + point.completion_tokens_saved;
            let cost = self.pricing.map_or(String::new(), |p| {
                format!("<td>${:.4}</td>", point.cost_saved(p))
            });
            let _ = writeln!(
                rows,
                "<tr{class}><td>{:.2}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{fhr}</td><td>{tokens}</td>{cost}</tr>",
                point.threshold,
                percent(point.hit_rate()),
                point.exact_hits,
                point.semantic_hits,
                point.false_hits,
            );
        }
        let verdict = match (recommended, self.detail()) {
            (Some(threshold), Some(point)) => format!(
                "<p class=\"verdict ok\">Recommended threshold: <strong>{threshold:.2}</strong> \
                 (hit rate {}, FHR {})</p>",
                percent(point.hit_rate()),
                point.false_hit_rate().map_or("-".into(), percent),
            ),
            _ => format!(
                "<p class=\"verdict none\">{}</p>",
                escape(&self.no_recommendation())
            ),
        };
        let detail = self.detail().map_or(String::new(), |point| {
            let warnings: String = warnings(point)
                .iter()
                .map(|warning| format!("<p class=\"warn\">{}</p>\n", escape(warning)))
                .collect();
            format!(
                "<h2>At {:.2}</h2>\n<dl>\n<dt>Correctness</dt><dd>{}</dd>\n\
                 <dt>Guard rejections</dt><dd>{}</dd>\n<dt>Language refusals</dt><dd>{}</dd>\n</dl>\n{warnings}",
                point.threshold,
                escape(&judged_summary(point)),
                escape(&rejects_summary(point)),
                point.language_refusals,
            )
        });
        format!(
            "<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>veracache eval</title>
<style>
:root {{ --fg: #1d1d1f; --muted: #6e6e73; --bg: #ffffff; --line: #d2d2d7; --hit: #1f6feb; --fhr: #cf222e; --rec: #fff8c5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg: #f5f5f7; --muted: #a1a1a6; --bg: #161618; --line: #3a3a3c; --hit: #58a6ff; --fhr: #ff7b72; --rec: #3b3417; }} }}
body {{ font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); max-width: 960px; margin: 0 auto; padding: 24px 16px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 28px 0 8px; }}
.meta, .warn {{ color: var(--muted); }}
.verdict {{ font-size: 17px; padding: 10px 14px; border-left: 4px solid var(--hit); }}
.verdict.none {{ border-color: var(--fhr); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
tr.rec td {{ background: var(--rec); font-weight: 600; }}
dt {{ font-weight: 600; }}
dd {{ margin: 0 0 8px; }}
svg {{ width: 100%; height: auto; }}
.axis {{ stroke: var(--line); }}
.label {{ fill: var(--muted); font-size: 12px; }}
.hit {{ stroke: var(--hit); fill: none; stroke-width: 2; }}
.fhr {{ stroke: var(--fhr); fill: none; stroke-width: 2; }}
.mark {{ stroke: var(--muted); stroke-dasharray: 4 4; }}
</style>
</head>
<body>
<h1>veracache eval</h1>
<p class=\"meta\">{} rows, embedder {}, target FHR {}</p>
{verdict}
{chart}
<div class=\"scroll\"><table>
<thead><tr><th>Threshold</th><th>Hit rate</th><th>Exact</th><th>Semantic</th><th>False</th><th>FHR</th><th>Tokens saved</th>{cost_head}</tr></thead>
<tbody>
{rows}</tbody>
</table></div>
{detail}</body>
</html>
",
            self.rows,
            escape(&self.embedder),
            percent(self.target_fhr),
            chart = self.chart(recommended),
        )
    }

    /// Hit rate and FHR against threshold, as inline SVG. FHR has gaps where
    /// no semantic hit was served.
    fn chart(&self, recommended: Option<f32>) -> String {
        let (Some(first), Some(last)) = (self.points.first(), self.points.last()) else {
            return String::new();
        };
        let (width, height, pad) = (640.0_f32, 220.0_f32, 36.0_f32);
        let span = (last.threshold - first.threshold).max(f32::EPSILON);
        let x = |t: f32| pad + (t - first.threshold) / span * (width - 2.0 * pad);
        let y = |rate: f32| height - pad - rate * (height - 2.0 * pad);
        let line = |rates: Vec<Option<(f32, f32)>>, class: &str| -> String {
            rates
                .split(Option::is_none)
                .filter(|run| !run.is_empty())
                .map(|run| {
                    let points: Vec<String> = run
                        .iter()
                        .flatten()
                        .map(|(t, rate)| format!("{:.1},{:.1}", x(*t), y(*rate)))
                        .collect();
                    format!(
                        "<polyline class=\"{class}\" points=\"{}\"/>",
                        points.join(" ")
                    )
                })
                .collect()
        };
        let hits = self
            .points
            .iter()
            .map(|p| Some((p.threshold, p.hit_rate())))
            .collect();
        let fhr = self
            .points
            .iter()
            .map(|p| p.false_hit_rate().map(|rate| (p.threshold, rate)))
            .collect();
        let mark = recommended.map_or(String::new(), |t| {
            format!(
                "<line class=\"mark\" x1=\"{0:.1}\" x2=\"{0:.1}\" y1=\"{1:.1}\" y2=\"{2:.1}\"/>",
                x(t),
                y(1.0),
                y(0.0)
            )
        });
        format!(
            "<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Hit rate and false-hit rate by threshold\">
<line class=\"axis\" x1=\"{pad}\" x2=\"{right}\" y1=\"{bottom:.1}\" y2=\"{bottom:.1}\"/>
<line class=\"axis\" x1=\"{pad}\" x2=\"{pad}\" y1=\"{top:.1}\" y2=\"{bottom:.1}\"/>
<text class=\"label\" x=\"{pad}\" y=\"{below:.1}\">{:.2}</text>
<text class=\"label\" x=\"{right}\" y=\"{below:.1}\" text-anchor=\"end\">{:.2}</text>
<text class=\"label\" x=\"{left_label}\" y=\"{top:.1}\" text-anchor=\"end\">100%</text>
<text class=\"label\" x=\"{left_label}\" y=\"{bottom:.1}\" text-anchor=\"end\">0%</text>
{mark}{}{}
<line class=\"hit\" x1=\"{pad}\" x2=\"{key_end}\" y1=\"12\" y2=\"12\"/><text class=\"label\" x=\"{key_text}\" y=\"16\">hit rate</text>
<line class=\"fhr\" x1=\"{key2}\" x2=\"{key2_end}\" y1=\"12\" y2=\"12\"/><text class=\"label\" x=\"{key2_text}\" y=\"16\">false-hit rate</text>
</svg>",
            first.threshold,
            last.threshold,
            line(hits, "hit"),
            line(fhr, "fhr"),
            right = width - pad,
            top = y(1.0),
            bottom = y(0.0),
            below = y(0.0) + 16.0,
            left_label = pad - 4.0,
            key_end = pad + 20.0,
            key_text = pad + 26.0,
            key2 = pad + 96.0,
            key2_end = pad + 116.0,
            key2_text = pad + 122.0,
        )
    }

    /// The same report as JSON.
    pub fn json(&self) -> Value {
        let points: Vec<Value> = self
            .points
            .iter()
            .map(|point| {
                let judged_by: serde_json::Map<String, Value> = point
                    .judged_by
                    .iter()
                    .map(|(source, count)| (source_name(*source).replace(' ', "_"), json!(count)))
                    .collect();
                json!({
                    "threshold": point.threshold,
                    "hit_rate": point.hit_rate(),
                    "exact_hits": point.exact_hits,
                    "semantic_hits": point.semantic_hits,
                    "false_hits": point.false_hits,
                    "false_hit_rate": point.false_hit_rate(),
                    "prompt_tokens_saved": point.prompt_tokens_saved,
                    "completion_tokens_saved": point.completion_tokens_saved,
                    "cost_saved": self.pricing.map(|p| point.cost_saved(p)),
                    "rejects": point.rejects,
                    "language_refusals": point.language_refusals,
                    "judged_by": judged_by,
                })
            })
            .collect();
        json!({
            "embedder": self.embedder,
            "rows": self.rows,
            "target_fhr": self.target_fhr,
            "recommended_threshold": self.recommended(),
            "points": points,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn point(threshold: f32, semantic_hits: usize, false_hits: usize) -> SweepPoint {
        SweepPoint {
            threshold,
            rows: 10,
            exact_hits: 1,
            semantic_hits,
            false_hits,
            prompt_tokens_saved: 100,
            completion_tokens_saved: 400,
            rejects: BTreeMap::from([("entity".into(), 2), ("language".into(), 1)]),
            language_refusals: 3,
            judged_by: BTreeMap::from([(Source::Label, semantic_hits)]),
        }
    }

    fn report(target_fhr: f32) -> Report {
        Report {
            embedder: "bge-small-en-v1.5".into(),
            rows: 10,
            target_fhr,
            pricing: Some(Pricing {
                prompt: 1.0,
                completion: 2.0,
            }),
            points: vec![point(0.85, 4, 2), point(0.90, 2, 0)],
        }
    }

    #[test]
    fn text_recommends_and_explains_the_point() {
        let text = report(0.01).text();
        assert!(
            text.contains("10 rows, embedder bge-small-en-v1.5, target FHR 1.0%"),
            "{text}"
        );
        assert!(text.contains("0.85"), "{text}");
        assert!(text.contains("50.0%"), "{text}");
        assert!(text.contains("$0.0009"), "{text}");
        assert!(
            text.contains("Recommended threshold: 0.90 (hit rate 30.0%, FHR 0.0%)"),
            "{text}"
        );
        assert!(text.contains("correctness: 2 labelled"), "{text}");
        assert!(
            text.contains("guard rejections: entity 2, language 1"),
            "{text}"
        );
        assert!(text.contains("language refusals: 3"), "{text}");
        assert!(!text.contains("Heuristic verdicts"), "{text}");
    }

    #[test]
    fn text_says_when_no_threshold_meets_the_target() {
        let mut report = report(0.01);
        report.points[1].false_hits = 1;
        report.points[1].judged_by = BTreeMap::from([(Source::Heuristic, 2)]);
        let text = report.text();
        assert!(text.contains("No threshold meets FHR <= 1.0%"), "{text}");
        assert!(!text.contains("Recommended threshold"), "{text}");
        assert!(text.contains("At 0.90:"), "{text}");
        assert!(
            text.contains("Heuristic verdicts compare the logged responses"),
            "{text}"
        );
    }

    #[test]
    fn json_carries_every_point_and_the_recommendation() {
        let json = report(0.01).json();
        assert_eq!(
            json["recommended_threshold"].as_f64().map(|t| t as f32),
            Some(0.90)
        );
        assert_eq!(json["points"].as_array().unwrap().len(), 2);
        let first = &json["points"][0];
        assert_eq!(first["false_hit_rate"], 0.5);
        assert_eq!(first["rejects"]["entity"], 2);
        assert_eq!(first["judged_by"]["labelled"], 4);
        assert_eq!(first["cost_saved"], 0.0009);

        let none = report(0.0001);
        let mut none = none;
        none.points[1].false_hits = 1;
        assert_eq!(none.json()["recommended_threshold"], Value::Null);
    }
    const SNAPSHOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/report.html");

    // Run with UPDATE_SNAPSHOTS=1 to rewrite the snapshot after an intended change.
    #[test]
    fn html_matches_the_snapshot() {
        let html = report(0.01).html();
        if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
            let path = std::path::Path::new(SNAPSHOT);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, &html).unwrap();
        }
        let expected = std::fs::read_to_string(SNAPSHOT)
            .expect("missing snapshot; run with UPDATE_SNAPSHOTS=1 to create it");
        assert_eq!(html, expected);
    }

    #[test]
    fn html_makes_no_network_requests() {
        let mut unmet = report(0.01);
        unmet.points[1].false_hits = 1;
        for page in [report(0.01).html(), unmet.html()] {
            for banned in [
                "http://", "https://", "src=", "href=", "<link", "<script", "@import", "url(",
            ] {
                assert!(!page.contains(banned), "{banned}");
            }
        }
    }

    #[test]
    fn html_escapes_text_from_the_log() {
        let mut report = report(0.01);
        report.embedder = "<script>alert(1)</script>".into();
        report.points[1].rejects = BTreeMap::from([("a&b<i>".into(), 1)]);
        let html = report.html();
        assert!(
            html.contains("embedder &lt;script&gt;alert(1)&lt;/script&gt;"),
            "{html}"
        );
        assert!(html.contains("a&amp;b&lt;i&gt; 1"), "{html}");
        assert!(!html.contains("<script"), "{html}");
    }

    #[test]
    fn html_says_when_no_threshold_meets_the_target() {
        let mut report = report(0.01);
        report.points[1].false_hits = 1;
        report.points[1].judged_by = BTreeMap::from([(Source::Heuristic, 2)]);
        let html = report.html();
        assert!(
            html.contains("class=\"verdict none\">No threshold meets FHR &lt;= 1.0%"),
            "{html}"
        );
        assert!(!html.contains("Recommended threshold"), "{html}");
        assert!(!html.contains("class=\"rec\""), "{html}");
        assert!(!html.contains("class=\"mark\""), "{html}");
        assert!(html.contains("class=\"warn\">Heuristic verdicts"), "{html}");
    }
}
