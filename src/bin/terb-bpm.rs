use std::{env, process, time::Instant};

use serde::Serialize;
use terb::{
    audio::read_wav_mono,
    beat::{embedded_model_bytes, BeatEstimate, RealtimeBeatTracker, StageTimings},
};

#[derive(Serialize)]
struct Report {
    bpm: Option<f32>,
    confidence: f32,
    first_estimate_seconds: Option<f32>,
    stable_lock_seconds: Option<f32>,
    realtime_factor: f64,
    audio_seconds: f64,
    model_bytes: u64,
    peak_memory_bytes: Option<u64>,
    timings: StageTimings,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("terb-bpm: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args[1] != "analyze" {
        return Err("usage: terb-bpm analyze <song.wav> [--model model.onnx] [--chunk 512]".into());
    }
    let wav_path = &args[2];
    let model = option(&args, "--model");
    let chunk: usize = option(&args, "--chunk").unwrap_or("512").parse()?;
    if chunk == 0 {
        return Err("--chunk must be greater than zero".into());
    }
    let wav = read_wav_mono(wav_path)?;
    let mut tracker = match model {
        Some(path) => RealtimeBeatTracker::with_model(wav.sample_rate, path)?,
        None => RealtimeBeatTracker::new(wav.sample_rate)?,
    };
    let started = Instant::now();
    let mut first = None;
    let mut stable = None;
    let mut estimates: Vec<(f32, BeatEstimate)> = Vec::new();
    for (index, samples) in wav.samples.chunks(chunk).enumerate() {
        if let Some(value) = tracker.consume(samples) {
            let seconds = ((index * chunk + samples.len()) as f32 / wav.sample_rate as f32)
                .min(wav.samples.len() as f32 / wav.sample_rate as f32);
            first.get_or_insert(seconds);
            estimates.push((seconds, value));
            if stable.is_none() && is_stable(&estimates) {
                stable = Some(seconds);
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let audio_seconds = wav.samples.len() as f64 / wav.sample_rate as f64;
    let last = estimates.last().map(|x| x.1);
    let report = Report {
        bpm: last.map(|x| x.bpm),
        confidence: last.map(|x| x.confidence).unwrap_or(0.0),
        first_estimate_seconds: first,
        stable_lock_seconds: stable,
        realtime_factor: elapsed / audio_seconds.max(1e-9),
        audio_seconds,
        model_bytes: match model {
            Some(path) => std::fs::metadata(path)?.len(),
            None => embedded_model_bytes(),
        },
        peak_memory_bytes: peak_memory_bytes(),
        timings: tracker.timings(),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn is_stable(values: &[(f32, BeatEstimate)]) -> bool {
    let Some(&(now, last)) = values.last() else {
        return false;
    };
    let recent: Vec<f32> = values
        .iter()
        .rev()
        .take_while(|(t, _)| now - t <= 2.0)
        .map(|x| x.1.bpm)
        .collect();
    recent.len() >= 5
        && recent
            .iter()
            .all(|b| (b - last.bpm).abs() / last.bpm < 0.015)
        && last.confidence >= 0.25
}

fn option<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|x| x == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

#[cfg(target_os = "linux")]
fn peak_memory_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|x| x.starts_with("VmHWM:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}
#[cfg(not(target_os = "linux"))]
fn peak_memory_bytes() -> Option<u64> {
    None
}
