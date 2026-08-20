use serde::{Deserialize, Serialize};
use std::{env, fs, path::Path, process, time::Instant};
use terb::{audio::read_wav_mono, beat::RealtimeBeatTracker, bpm::BpmAnalyzer};

const CHECKPOINTS: [f32; 4] = [4.0, 8.0, 16.0, 30.0];

#[derive(Clone, Deserialize)]
struct Item {
    path: String,
    bpm: f32,
    dataset: String,
    split: String,
    genre: Option<String>,
}

#[derive(Serialize)]
struct ResultRow {
    path: String,
    dataset: String,
    split: String,
    genre: String,
    algorithm: String,
    reference_bpm: f32,
    bpm_4s: Option<f32>,
    bpm_8s: Option<f32>,
    bpm_16s: Option<f32>,
    bpm_30s: Option<f32>,
    final_bpm: Option<f32>,
    confidence: f32,
    first_estimate_seconds: Option<f32>,
    stable_lock_seconds: Option<f32>,
    locked_jitter_bpm: Option<f32>,
    classification: String,
    elapsed_seconds: f64,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("terb-bpm-benchmark: {e}");
        process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return Err("usage: terb-bpm-benchmark <manifest.json> <output-dir> [--model model.onnx] [--chunk 512]".into());
    }
    let manifest: Vec<Item> = serde_json::from_slice(&fs::read(&args[1])?)?;
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    let model = option(&args, "--model");
    let chunk: usize = option(&args, "--chunk").unwrap_or("512").parse()?;
    let mut rows = Vec::new();
    for item in &manifest {
        if item.split != "test" {
            continue;
        }
        let wav = read_wav_mono(&item.path)?;
        rows.push(run_legacy(item, wav.sample_rate, &wav.samples, chunk));
        if let Some(path) = model {
            rows.push(run_neural(
                item,
                wav.sample_rate,
                &wav.samples,
                chunk,
                path,
            )?);
        }
    }
    fs::write(out.join("results.json"), serde_json::to_vec_pretty(&rows)?)?;
    fs::write(out.join("results.csv"), csv(&rows))?;
    fs::write(out.join("report.md"), report(&rows))?;
    println!("{}", serde_json::to_string_pretty(&summary(&rows))?);
    Ok(())
}

fn run_legacy(item: &Item, rate: u32, samples: &[f32], chunk: usize) -> ResultRow {
    let mut analyzer = BpmAnalyzer::new(rate as f32);
    collect(item, "legacy", rate, samples, chunk, |x| {
        analyzer.consume(x).map(|e| (e.bpm, e.confidence))
    })
}

fn run_neural(
    item: &Item,
    rate: u32,
    samples: &[f32],
    chunk: usize,
    model: &str,
) -> Result<ResultRow, Box<dyn std::error::Error>> {
    let mut analyzer = RealtimeBeatTracker::with_model(rate, model)?;
    Ok(collect(
        item,
        "causal-transformer",
        rate,
        samples,
        chunk,
        |x| analyzer.consume(x).map(|e| (e.bpm, e.confidence)),
    ))
}

fn collect<F: FnMut(&[f32]) -> Option<(f32, f32)>>(
    item: &Item,
    algorithm: &str,
    rate: u32,
    samples: &[f32],
    chunk: usize,
    mut consume: F,
) -> ResultRow {
    let started = Instant::now();
    let mut points = Vec::new();
    let mut cp = [None; 4];
    for (i, block) in samples.chunks(chunk).enumerate() {
        let time = ((i * chunk + block.len()) as f32 / rate as f32)
            .min(samples.len() as f32 / rate as f32);
        if let Some((bpm, confidence)) = consume(block) {
            points.push((time, bpm, confidence));
        }
        for (j, at) in CHECKPOINTS.iter().enumerate() {
            if time >= *at && cp[j].is_none() {
                cp[j] = points.last().map(|x| x.1);
            }
        }
    }
    let first = points.first().map(|x| x.0);
    let stable = stable_lock(&points);
    let jitter = stable.map(|lock| {
        let x: Vec<f32> = points.iter().filter(|x| x.0 >= lock).map(|x| x.1).collect();
        stddev(&x)
    });
    let last = points.last().copied();
    let final_bpm = last.map(|x| x.1);
    ResultRow {
        path: item.path.clone(),
        dataset: item.dataset.clone(),
        split: item.split.clone(),
        genre: item.genre.clone().unwrap_or_default(),
        algorithm: algorithm.into(),
        reference_bpm: item.bpm,
        bpm_4s: cp[0],
        bpm_8s: cp[1],
        bpm_16s: cp[2],
        bpm_30s: cp[3],
        final_bpm,
        confidence: last.map(|x| x.2).unwrap_or(0.0),
        first_estimate_seconds: first,
        stable_lock_seconds: stable,
        locked_jitter_bpm: jitter,
        classification: classify(final_bpm, item.bpm, stable).into(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    }
}

fn stable_lock(x: &[(f32, f32, f32)]) -> Option<f32> {
    for &(start, bpm, confidence) in x {
        let window: Vec<_> = x
            .iter()
            .filter(|p| p.0 >= start && p.0 <= start + 2.0)
            .collect();
        if confidence >= 0.25
            && window.len() >= 5
            && window.iter().all(|p| (p.1 - bpm).abs() / bpm < 0.015)
        {
            return Some(start);
        }
    }
    None
}

fn classify(value: Option<f32>, reference: f32, stable: Option<f32>) -> &'static str {
    let Some(value) = value else {
        return "no_result";
    };
    if stable.is_none() {
        return "unstable";
    }
    if relative(value, reference) <= 0.04 {
        "correct"
    } else if relative(value, reference * 0.5) <= 0.04 {
        "half_time"
    } else if relative(value, reference * 2.0) <= 0.04 {
        "double_time"
    } else {
        "wrong"
    }
}
fn relative(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.max(1e-6)
}
fn stddev(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let m = x.iter().sum::<f32>() / x.len() as f32;
    (x.iter().map(|v| (v - m).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
}

#[derive(Serialize)]
struct Summary {
    algorithm: String,
    count: usize,
    strict_accuracy: f32,
    metrical_accuracy: f32,
    half_time_rate: f32,
    double_time_rate: f32,
    wrong_rate: f32,
    unstable_rate: f32,
    no_result_rate: f32,
    accuracy_4s: f32,
    accuracy_8s: f32,
    accuracy_16s: f32,
    accuracy_30s: f32,
    mean_lock_seconds: Option<f32>,
    mean_jitter_bpm: Option<f32>,
}
fn summary(rows: &[ResultRow]) -> Vec<Summary> {
    let mut names: Vec<_> = rows.iter().map(|r| r.algorithm.as_str()).collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let x: Vec<_> = rows.iter().filter(|r| r.algorithm == name).collect();
            let n = x.len().max(1) as f32;
            let rate =
                |kind: &str| x.iter().filter(|r| r.classification == kind).count() as f32 / n;
            let locks: Vec<f32> = x.iter().filter_map(|r| r.stable_lock_seconds).collect();
            let jitters: Vec<f32> = x.iter().filter_map(|r| r.locked_jitter_bpm).collect();
            Summary {
                algorithm: name.into(),
                count: x.len(),
                strict_accuracy: rate("correct"),
                metrical_accuracy: rate("correct") + rate("half_time") + rate("double_time"),
                half_time_rate: rate("half_time"),
                double_time_rate: rate("double_time"),
                wrong_rate: rate("wrong"),
                unstable_rate: rate("unstable"),
                no_result_rate: rate("no_result"),
                accuracy_4s: x
                    .iter()
                    .filter(|r| {
                        r.bpm_4s
                            .map(|b| relative(b, r.reference_bpm) <= 0.04)
                            .unwrap_or(false)
                    })
                    .count() as f32
                    / n,
                accuracy_8s: x
                    .iter()
                    .filter(|r| {
                        r.bpm_8s
                            .map(|b| relative(b, r.reference_bpm) <= 0.04)
                            .unwrap_or(false)
                    })
                    .count() as f32
                    / n,
                accuracy_16s: x
                    .iter()
                    .filter(|r| {
                        r.bpm_16s
                            .map(|b| relative(b, r.reference_bpm) <= 0.04)
                            .unwrap_or(false)
                    })
                    .count() as f32
                    / n,
                accuracy_30s: x
                    .iter()
                    .filter(|r| {
                        r.bpm_30s
                            .map(|b| relative(b, r.reference_bpm) <= 0.04)
                            .unwrap_or(false)
                    })
                    .count() as f32
                    / n,
                mean_lock_seconds: mean(&locks),
                mean_jitter_bpm: mean(&jitters),
            }
        })
        .collect()
}
fn mean(x: &[f32]) -> Option<f32> {
    (!x.is_empty()).then(|| x.iter().sum::<f32>() / x.len() as f32)
}
fn report(rows: &[ResultRow]) -> String {
    let mut s=String::from("# Terb BPM benchmark\n\nTolerance: ±4%. Half/double-time are reported separately. Input is streamed in fixed chunks.\n\n|algorithm|n|strict|metrical|half|double|wrong|unstable|no result|4s|8s|16s|30s|lock s|jitter BPM|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for x in summary(rows) {
        s.push_str(&format!(
            "|{}|{}|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{:.1}%|{}|{}|\n",
            x.algorithm,
            x.count,
            100.0 * x.strict_accuracy,
            100.0 * x.metrical_accuracy,
            100.0 * x.half_time_rate,
            100.0 * x.double_time_rate,
            100.0 * x.wrong_rate,
            100.0 * x.unstable_rate,
            100.0 * x.no_result_rate,
            100.0 * x.accuracy_4s, 100.0 * x.accuracy_8s, 100.0 * x.accuracy_16s, 100.0 * x.accuracy_30s,
            x.mean_lock_seconds
                .map(|v| format!("{v:.2}"))
                .unwrap_or("-".into()),
            x.mean_jitter_bpm
                .map(|v| format!("{v:.2}"))
                .unwrap_or("-".into())
        ));
    }
    s
}
fn csv(rows: &[ResultRow]) -> String {
    let mut s=String::from("path,dataset,split,genre,algorithm,reference_bpm,bpm_4s,bpm_8s,bpm_16s,bpm_30s,final_bpm,confidence,first_estimate_seconds,stable_lock_seconds,locked_jitter_bpm,classification,elapsed_seconds\n");
    for r in rows {
        let q = |x: &str| format!("\"{}\"", x.replace('"', "\"\""));
        let f = |x: Option<f32>| x.map(|v| v.to_string()).unwrap_or_default();
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            q(&r.path),
            q(&r.dataset),
            q(&r.split),
            q(&r.genre),
            q(&r.algorithm),
            r.reference_bpm,
            f(r.bpm_4s),
            f(r.bpm_8s),
            f(r.bpm_16s),
            f(r.bpm_30s),
            f(r.final_bpm),
            r.confidence,
            f(r.first_estimate_seconds),
            f(r.stable_lock_seconds),
            f(r.locked_jitter_bpm),
            r.classification,
            r.elapsed_seconds
        ));
    }
    s
}
fn option<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|x| x == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}
