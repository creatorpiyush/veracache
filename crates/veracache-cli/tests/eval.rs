//! Runs the `veracache eval` binary end to end.

use std::process::{Command, Output};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

fn eval(args: &[&str]) -> Output {
    let model_cache = format!("{ROOT}/.fastembed_cache");
    Command::new(env!("CARGO_BIN_EXE_veracache"))
        .arg("eval")
        .args(args)
        .args(["--model-cache", &model_cache])
        .current_dir(ROOT)
        .output()
        .unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn bad_arguments_exit_2_with_usage() {
    let output = eval(&["--sweep", "0.9:0.8:0.01"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stderr(&output).contains("usage: veracache eval"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn unreadable_input_exits_1() {
    let output = eval(&["--input", "no/such/log.jsonl"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).starts_with("error: "),
        "{}",
        stderr(&output)
    );
}

#[test]
fn labels_with_unknown_ids_exit_1() {
    let labels = std::env::temp_dir().join("veracache-bad-labels.jsonl");
    std::fs::write(&labels, r#"{"a":"france","b":"nowhere","label":"hit"}"#).unwrap();
    let output = eval(&[
        "--input",
        "eval/datasets/sample/log.jsonl",
        "--labels",
        labels.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("unknown row id nowhere"),
        "{}",
        stderr(&output)
    );
}

#[test]
#[ignore = "downloads the model"]
fn golden_suite_gets_a_recommended_threshold() {
    let output = eval(&["--golden", "eval/datasets/golden.jsonl", "--format", "json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let report: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(report["rows"], 504);
    assert_eq!(report["points"].as_array().unwrap().len(), 19);
    assert!(report["recommended_threshold"].is_f64(), "{report}");
}

#[test]
#[ignore = "downloads the model"]
fn sample_log_gets_a_recommended_threshold() {
    let output = eval(&[
        "--input",
        "eval/datasets/sample/log.jsonl",
        "--labels",
        "eval/datasets/sample/labels.jsonl",
        "--price",
        "0.15,0.60",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("Recommended threshold: "), "{text}");
    assert!(text.contains("cost saved"), "{text}");
    assert!(text.contains("correctness: 3 labelled"), "{text}");
}

#[test]
#[ignore = "downloads the model"]
fn unmeetable_target_gets_no_recommendation() {
    // Below 0.87 the sample log serves a labelled false hit at every threshold.
    let output = eval(&[
        "--input",
        "eval/datasets/sample/log.jsonl",
        "--labels",
        "eval/datasets/sample/labels.jsonl",
        "--sweep",
        "0.80:0.86:0.01",
        "--target-fhr",
        "0",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("No threshold meets FHR <= 0.0%"), "{text}");
    assert!(!text.contains("Recommended threshold"), "{text}");
}

#[test]
#[ignore = "downloads the model"]
fn html_report_is_one_self_contained_page() {
    let output = eval(&[
        "--input",
        "eval/datasets/sample/log.jsonl",
        "--labels",
        "eval/datasets/sample/labels.jsonl",
        "--format",
        "html",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let page = stdout(&output);
    assert!(page.starts_with("<!doctype html>"), "{page}");
    assert!(page.contains("Recommended threshold: <strong>"), "{page}");
    assert!(!page.contains("http"), "{page}");
}
