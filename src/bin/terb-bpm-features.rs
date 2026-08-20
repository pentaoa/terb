use std::{env, process};
use terb::{audio::read_wav_mono, features::MelExtractor};

fn main() {
    if let Err(e) = run() {
        eprintln!("terb-bpm-features: {e}");
        process::exit(2);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: terb-bpm-features <input.wav> <output.json>".into());
    }
    let wav = read_wav_mono(&args[1])?;
    let mut extractor = MelExtractor::new(wav.sample_rate)?;
    let mut frames: Vec<Vec<f32>> = Vec::new();
    for block in wav.samples.chunks(512) {
        extractor.consume(block, |f| frames.push(f.values.to_vec()));
    }
    std::fs::write(&args[2], serde_json::to_vec(&frames)?)?;
    Ok(())
}
