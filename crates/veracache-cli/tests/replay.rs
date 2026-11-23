//! Runs the `veracache replay` and `veracache inspect` binaries end to end.

use std::process::{Command, Output};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const LOG: &str = "eval/datasets/sample/log.jsonl";

fn run(args: &[&str]) -> Output {
    let model_cache = format!("{ROOT}/.fastembed_cache");
    Command::new(env!("CARGO_BIN_EXE_veracache"))
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
fn inspect_needs_an_id() {
    let output = run(&["inspect", "--input", LOG]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stderr(&output).contains("--id is required"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn inspect_reports_an_unknown_id() {
    let output = run(&["inspect", "--input", LOG, "--id", "nope"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("no row with id nope"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn unknown_commands_exit_2() {
    let output = run(&["serve-forever"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        stderr(&output).contains("unknown command serve-forever"),
        "{}",
        stderr(&output)
    );
}

#[test]
#[ignore = "downloads the model"]
fn replay_lists_every_row_and_totals() {
    let output = run(&["replay", "--input", LOG]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        text.contains("veracache replay: 14 rows, threshold 0.88"),
        "{text}"
    );
    assert!(text.contains("france-again"), "{text}");
    assert!(
        text.contains("exact hits 1, semantic hits 3, misses 9, refused 1"),
        "{text}"
    );
}

#[test]
#[ignore = "downloads the model"]
fn replay_json_has_one_record_per_row() {
    let output = run(&["replay", "--input", LOG, "--format", "json"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let records: serde_json::Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(records.as_array().unwrap().len(), 14);
    assert_eq!(records[6]["outcome"], "exact hit");
    assert_eq!(records[6]["served_from"], "france");
}

#[test]
#[ignore = "downloads the model"]
fn inspect_prints_the_decision_record() {
    let output = run(&[
        "inspect",
        "--input",
        LOG,
        "--id",
        "jupiter",
        "--threshold",
        "0.80",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        text.starts_with("row jupiter: miss\nthreshold 0.80\n"),
        "{text}"
    );
    assert!(text.contains("mars"), "{text}");
    assert!(text.contains("rejected: entity"), "{text}");
}
