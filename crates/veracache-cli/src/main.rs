//! The veracache command-line tool.

#![forbid(unsafe_code)]

use std::process::ExitCode;
use std::sync::Arc;

use veracache::{CacheConfig, CachedEmbedder, Embedder, LocalEmbedder};
use veracache_eval::{
    Correctness, JudgeConfig, Labels, LlmJudge, LoadError, LogRow, Pricing, Report, decision_text,
    golden_log, load_log, load_pairs, replay, replay_text, replayed_json, sweep, thresholds,
};

const USAGE: &str = "\
usage: veracache eval --input <log.jsonl> [--labels <pairs.jsonl>] [options]
       veracache eval --golden <golden.jsonl> [options]
       veracache replay --input <log.jsonl> [--threshold <t>] [--format table|json]
       veracache inspect --input <log.jsonl> --id <row> [--threshold <t>] [--format table|json]

options:
  --judge openai:<model>       ask a model about unlabelled hits
                               (reads OPENAI_BASE_URL and OPENAI_API_KEY)
  --sweep <start:end:step>     thresholds to replay [default: 0.80:0.98:0.01]
  --target-fhr <rate>          highest acceptable false-hit rate [default: 0.01]
  --price <prompt,completion>  price per million tokens, to estimate cost saved
  --format <table|json|html>   output format [default: table]
  --threshold <t>              similarity threshold for replay and inspect [default: 0.88]
  --model-cache <dir>          local embedding model directory [default: .fastembed_cache]";

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Where the replayed traffic comes from.
#[derive(Debug, PartialEq)]
enum Data {
    Log {
        input: String,
        labels: Option<String>,
    },
    Golden(String),
}

/// How the report is printed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Table,
    Json,
    Html,
}

#[derive(Debug, PartialEq)]
struct EvalArgs {
    data: Data,
    judge: Option<String>,
    sweep: (f32, f32, f32),
    target_fhr: f32,
    pricing: Option<Pricing>,
    format: Format,
    model_cache: String,
}

fn parse_eval(args: &[String]) -> Result<EvalArgs, String> {
    let (mut input, mut labels, mut golden, mut judge) = (None, None, None, None);
    let mut sweep = (0.80, 0.98, 0.01);
    let mut target_fhr = 0.01;
    let mut pricing = None;
    let mut format = Format::Table;
    let mut model_cache = ".fastembed_cache".to_owned();
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let mut value = || args.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--input" => input = Some(value()?),
            "--labels" => labels = Some(value()?),
            "--golden" => golden = Some(value()?),
            "--judge" => {
                let spec = value()?;
                let model = spec
                    .strip_prefix("openai:")
                    .filter(|model| !model.is_empty())
                    .ok_or("--judge must be openai:<model>")?;
                judge = Some(model.to_owned());
            }
            "--sweep" => sweep = parse_sweep(&value()?)?,
            "--target-fhr" => {
                let rate: f32 = value()?
                    .parse()
                    .map_err(|_| "--target-fhr must be a number")?;
                if !(0.0..=1.0).contains(&rate) {
                    return Err("--target-fhr must be between 0 and 1".into());
                }
                target_fhr = rate;
            }
            "--price" => pricing = Some(parse_price(&value()?)?),
            "--format" => {
                format = match value()?.as_str() {
                    "table" => Format::Table,
                    "json" => Format::Json,
                    "html" => Format::Html,
                    _ => return Err("--format must be table, json, or html".into()),
                }
            }
            "--model-cache" => model_cache = value()?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let data = match (input, labels, golden) {
        (Some(input), labels, None) => Data::Log { input, labels },
        (None, None, Some(golden)) => Data::Golden(golden),
        (None, None, None) => return Err("--input or --golden is required".into()),
        _ => return Err("--golden cannot be combined with --input or --labels".into()),
    };
    Ok(EvalArgs {
        data,
        judge,
        sweep,
        target_fhr,
        pricing,
        format,
        model_cache,
    })
}

/// Arguments for `replay` and `inspect`.
#[derive(Debug, PartialEq)]
struct ReplayArgs {
    input: String,
    /// The row to inspect. `None` for `replay`.
    id: Option<String>,
    threshold: f32,
    json: bool,
    model_cache: String,
}

fn parse_replay(args: &[String], inspect: bool) -> Result<ReplayArgs, String> {
    let (mut input, mut id) = (None, None);
    let mut threshold = CacheConfig::default().threshold;
    let mut json = false;
    let mut model_cache = ".fastembed_cache".to_owned();
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let mut value = || args.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--input" => input = Some(value()?),
            "--id" if inspect => id = Some(value()?),
            "--threshold" => {
                let value: f32 = value()?
                    .parse()
                    .map_err(|_| "--threshold must be a number")?;
                if !(value > 0.0 && value <= 1.0) {
                    return Err("--threshold must be above 0 and at most 1".into());
                }
                threshold = value;
            }
            "--format" => {
                json = match value()?.as_str() {
                    "table" => false,
                    "json" => true,
                    _ => return Err("--format must be table or json".into()),
                }
            }
            "--model-cache" => model_cache = value()?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if inspect && id.is_none() {
        return Err("--id is required".into());
    }
    Ok(ReplayArgs {
        input: input.ok_or("--input is required")?,
        id,
        threshold,
        json,
        model_cache,
    })
}

fn parse_sweep(spec: &str) -> Result<(f32, f32, f32), String> {
    let bad = || format!("--sweep must be start:end:step within 0..1, got {spec}");
    let parts: Vec<f32> = spec
        .split(':')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| bad())?;
    match parts[..] {
        [start, end, step] if 0.0 < start && start <= end && end <= 1.0 && step > 0.0 => {
            Ok((start, end, step))
        }
        _ => Err(bad()),
    }
}

fn parse_price(spec: &str) -> Result<Pricing, String> {
    let bad = || format!("--price must be <prompt>,<completion> per million tokens, got {spec}");
    let parts: Vec<f64> = spec
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| bad())?;
    match parts[..] {
        [prompt, completion] if prompt >= 0.0 && completion >= 0.0 => {
            Ok(Pricing { prompt, completion })
        }
        _ => Err(bad()),
    }
}

async fn run_eval(args: EvalArgs) -> Result<String, LoadError> {
    let (rows, labels) = match &args.data {
        Data::Log { input, labels } => {
            let rows = load_log(input)?;
            let labels = match labels {
                Some(path) => Labels::load(path, &rows)?,
                None => Labels::default(),
            };
            (rows, labels)
        }
        Data::Golden(path) => golden_log(&load_pairs(path)?),
    };
    let mut correctness = Correctness::new(labels);
    if let Some(model) = &args.judge {
        correctness = correctness.with_judge(LlmJudge::new(JudgeConfig {
            base_url: std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into()),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            model: model.clone(),
        })?);
    }

    let (embedder_id, embedder) = local_embedder(&args.model_cache)?;
    let (start, end, step) = args.sweep;
    let judge = async |query: &LogRow, source: &LogRow| correctness.judge(query, source).await;
    let points = sweep(
        &rows,
        embedder,
        CacheConfig::default(),
        &thresholds(start, end, step),
        judge,
    )
    .await;

    let report = Report {
        embedder: embedder_id,
        rows: rows.len(),
        target_fhr: args.target_fhr,
        pricing: args.pricing,
        points,
    };
    Ok(match args.format {
        Format::Table => report.text(),
        Format::Json => serde_json::to_string_pretty(&report.json())?,
        Format::Html => report.html(),
    })
}

/// The local embedding model, wrapped so each text is embedded once.
fn local_embedder(model_cache: &str) -> Result<(String, Arc<dyn Embedder>), LoadError> {
    let local = LocalEmbedder::new(model_cache)?;
    let id = local.id().to_owned();
    Ok((id, Arc::new(CachedEmbedder::new(local, usize::MAX))))
}

async fn run_replay(args: ReplayArgs) -> Result<String, LoadError> {
    let mut rows = load_log(&args.input)?;
    if let Some(id) = &args.id {
        // Rows after the inspected one cannot change its decision.
        let index = rows
            .iter()
            .position(|row| &row.id == id)
            .ok_or(format!("no row with id {id}"))?;
        rows.truncate(index + 1);
    }
    let (embedder_id, embedder) = local_embedder(&args.model_cache)?;
    let config = CacheConfig {
        threshold: args.threshold,
        ..CacheConfig::default()
    };
    let replayed = replay(&rows, embedder, config).await;
    Ok(match (&args.id, args.json) {
        (Some(_), false) => replayed.last().map(decision_text).unwrap_or_default(),
        (Some(_), true) => {
            let record = replayed.last().map(replayed_json).unwrap_or_default();
            serde_json::to_string_pretty(&record)?
        }
        (None, false) => replay_text(&replayed, &embedder_id),
        (None, true) => {
            let records: Vec<_> = replayed.iter().map(replayed_json).collect();
            serde_json::to_string_pretty(&records)?
        }
    })
}

/// Runs a command on a fresh runtime and prints its output or error.
fn finish(command: impl std::future::Future<Output = Result<String, LoadError>>) -> ExitCode {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    let result = match runtime {
        Ok(runtime) => runtime.block_on(command),
        Err(error) => Err(error.into()),
    };
    match result {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn usage_error(error: &str) -> ExitCode {
    eprintln!("error: {error}\n\n{USAGE}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = args.split_first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match command.as_str() {
        "eval" => match parse_eval(rest) {
            Ok(eval) => finish(run_eval(eval)),
            Err(error) => usage_error(&error),
        },
        "replay" | "inspect" => match parse_replay(rest, command == "inspect") {
            Ok(replay) => finish(run_replay(replay)),
            Err(error) => usage_error(&error),
        },
        "--help" | "-h" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        other => usage_error(&format!("unknown command {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &str) -> Result<EvalArgs, String> {
        let args: Vec<String> = args.split_whitespace().map(str::to_owned).collect();
        parse_eval(&args)
    }

    #[test]
    fn defaults_match_the_spec() {
        let args = parse("--input log.jsonl").unwrap();
        assert_eq!(
            args,
            EvalArgs {
                data: Data::Log {
                    input: "log.jsonl".into(),
                    labels: None
                },
                judge: None,
                sweep: (0.80, 0.98, 0.01),
                target_fhr: 0.01,
                pricing: None,
                format: Format::Table,
                model_cache: ".fastembed_cache".into(),
            }
        );
    }

    #[test]
    fn every_flag_is_read() {
        let args = parse(
            "--input log.jsonl --labels pairs.jsonl --judge openai:gpt-x --sweep 0.85:0.95:0.05 \
             --target-fhr 0.02 --price 0.15,0.6 --format json --model-cache models",
        )
        .unwrap();
        assert_eq!(
            args.data,
            Data::Log {
                input: "log.jsonl".into(),
                labels: Some("pairs.jsonl".into())
            }
        );
        assert_eq!(args.judge.as_deref(), Some("gpt-x"));
        assert_eq!(args.sweep, (0.85, 0.95, 0.05));
        assert_eq!(args.target_fhr, 0.02);
        assert_eq!(
            args.pricing,
            Some(Pricing {
                prompt: 0.15,
                completion: 0.6
            })
        );
        assert_eq!(args.format, Format::Json);
        assert_eq!(
            parse("--input a --format html").unwrap().format,
            Format::Html
        );
        assert_eq!(args.model_cache, "models");
        assert_eq!(
            parse("--golden golden.jsonl").unwrap().data,
            Data::Golden("golden.jsonl".into())
        );
    }

    #[test]
    fn replay_and_inspect_arguments() {
        let args: Vec<String> = "--input log.jsonl"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        assert_eq!(
            parse_replay(&args, false).unwrap(),
            ReplayArgs {
                input: "log.jsonl".into(),
                id: None,
                threshold: 0.88,
                json: false,
                model_cache: ".fastembed_cache".into(),
            }
        );
        let args: Vec<String> = "--input log.jsonl --id r3 --threshold 0.9 --format json"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        let inspect = parse_replay(&args, true).unwrap();
        assert_eq!(inspect.id.as_deref(), Some("r3"));
        assert_eq!(inspect.threshold, 0.9);
        assert!(inspect.json);

        let cases = [
            ("--input a", true, "--id is required"),
            ("--input a --id r1", false, "unknown argument --id"),
            ("--id r1", true, "--input is required"),
            (
                "--input a --threshold 0",
                false,
                "--threshold must be above 0",
            ),
            (
                "--input a --threshold x",
                false,
                "--threshold must be a number",
            ),
            (
                "--input a --format html",
                false,
                "--format must be table or json",
            ),
        ];
        for (args, inspect, error) in cases {
            let args: Vec<String> = args.split_whitespace().map(str::to_owned).collect();
            let message = parse_replay(&args, inspect).unwrap_err();
            assert!(message.contains(error), "{args:?}: {message}");
        }
    }

    #[test]
    fn bad_arguments_are_explained() {
        let cases = [
            ("", "--input or --golden is required"),
            ("--labels pairs.jsonl", "--golden cannot be combined"),
            ("--input a --golden b", "--golden cannot be combined"),
            ("--input", "--input needs a value"),
            ("--input a --judge gpt-x", "--judge must be openai:<model>"),
            (
                "--input a --judge openai:",
                "--judge must be openai:<model>",
            ),
            (
                "--input a --sweep 0.9:0.8:0.01",
                "--sweep must be start:end:step",
            ),
            (
                "--input a --sweep 0.8:0.9",
                "--sweep must be start:end:step",
            ),
            (
                "--input a --sweep 0.8:1.2:0.01",
                "--sweep must be start:end:step",
            ),
            (
                "--input a --target-fhr 2",
                "--target-fhr must be between 0 and 1",
            ),
            ("--input a --price 1", "--price must be"),
            (
                "--input a --format pdf",
                "--format must be table, json, or html",
            ),
            ("--input a --verbose", "unknown argument --verbose"),
        ];
        for (args, error) in cases {
            let message = parse(args).unwrap_err();
            assert!(message.contains(error), "{args}: {message}");
        }
    }
}
