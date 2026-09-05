use std::{
    collections::VecDeque,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use rten::{Model as RtenGraph, NodeId, Value as RtenValue};
use rten_tensor::AsView;
use serde::Serialize;
use thiserror::Error;

use crate::features::{MelExtractor, MelFrame, MEL_BANDS};

const FPS: f32 = 50.0;
const CONTEXT: usize = 112;
const INFERENCE_HOP: usize = 10;
const LOOKAHEAD: usize = 4;
const MAX_HISTORY: usize = 800;
const MIN_HISTORY: usize = 200;
const SILENCE_RMS: f32 = 0.00012;
const EMBEDDED_MODEL: &[u8] = include_bytes!("../assets/beat_tracker.onnx");

pub fn embedded_model_bytes() -> u64 {
    EMBEDDED_MODEL.len() as u64
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct BeatEstimate {
    pub bpm: f32,
    pub confidence: f32,
    pub beat_pulse: f32,
    pub phase: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downbeat_pulse: Option<f32>,
}

#[derive(Debug, Error)]
pub enum BeatTrackerError {
    #[error("beat model not found at {0}; check --model or TERB_BPM_MODEL")]
    ModelMissing(String),
    #[error("could not load beat model: {0}")]
    Model(String),
    #[error("invalid audio configuration: {0}")]
    Audio(&'static str),
}

pub struct RealtimeBeatTracker {
    features: MelExtractor,
    model: RtenGraph,
    model_input: NodeId,
    beat_output: NodeId,
    downbeat_output: NodeId,
    context: VecDeque<[f32; MEL_BANDS]>,
    input_buf: Vec<f32>,
    feature_frames: Vec<MelFrame>,
    pending_rms: VecDeque<f32>,
    decoder: ActivationDecoder,
    feature_seconds: f64,
    inference_seconds: f64,
    decode_seconds: f64,
    frames_since_inference: usize,
    emitted_any: bool,
}

struct SampleBlock {
    generation: u64,
    samples: Vec<f32>,
}

/// Non-blocking application wrapper. Model inference always runs on its own
/// worker thread; the producer only makes a bounded `try_send` call.
pub struct AsyncBeatTracker {
    sender: SyncSender<SampleBlock>,
    latest: Arc<Mutex<PublishedEstimate>>,
    generation: Arc<AtomicU64>,
    dropped_blocks: Arc<AtomicU64>,
}

#[derive(Default)]
struct PublishedEstimate {
    generation: u64,
    sequence: u64,
    estimate: Option<BeatEstimate>,
}

impl AsyncBeatTracker {
    pub fn new(sample_rate: u32) -> Result<Self, BeatTrackerError> {
        Self::from_tracker(RealtimeBeatTracker::new(sample_rate)?)
    }

    pub fn with_model(sample_rate: u32, path: impl AsRef<Path>) -> Result<Self, BeatTrackerError> {
        Self::from_tracker(RealtimeBeatTracker::with_model(sample_rate, path)?)
    }

    fn from_tracker(mut tracker: RealtimeBeatTracker) -> Result<Self, BeatTrackerError> {
        let (sender, receiver) = mpsc::sync_channel::<SampleBlock>(16);
        let latest = Arc::new(Mutex::new(PublishedEstimate::default()));
        let worker_latest = Arc::clone(&latest);
        let generation = Arc::new(AtomicU64::new(0));
        let worker_generation = Arc::clone(&generation);
        thread::Builder::new()
            .name("terb-beat-tracker".into())
            .spawn(move || {
                let mut active_generation = 0;
                while let Ok(block) = receiver.recv() {
                    if block.generation != worker_generation.load(Ordering::Acquire) {
                        continue;
                    }
                    if block.generation != active_generation {
                        tracker.reset();
                        active_generation = block.generation;
                    }
                    if let Some(estimate) = tracker.consume(&block.samples) {
                        if let Ok(mut value) = worker_latest.lock() {
                            value.generation = block.generation;
                            value.sequence = value.sequence.wrapping_add(1);
                            value.estimate = Some(estimate);
                        }
                    }
                }
            })
            .map_err(|e| BeatTrackerError::Model(format!("could not start beat worker: {e}")))?;
        Ok(Self {
            sender,
            latest,
            generation,
            dropped_blocks: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Returns immediately. A dropped block invalidates the previous audio
    /// history so the next accepted stream cannot splice across missing time.
    pub fn submit(&self, samples: &[f32]) -> bool {
        match self.sender.try_send(SampleBlock {
            generation: self.generation.load(Ordering::Acquire),
            samples: samples.to_vec(),
        }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                self.dropped_blocks.fetch_add(1, Ordering::Relaxed);
                self.reset();
                false
            }
            Err(TrySendError::Disconnected(_)) => {
                self.reset();
                false
            }
        }
    }

    pub fn latest(&self) -> Option<(u64, BeatEstimate)> {
        self.latest
            .try_lock()
            .ok()
            .filter(|value| value.generation == self.generation.load(Ordering::Acquire))
            .and_then(|value| value.estimate.map(|estimate| (value.sequence, estimate)))
    }

    pub fn reset(&self) {
        // Reset is independent of queue capacity. Old queued/in-flight results
        // are ignored; the worker resets before consuming the new generation.
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    pub fn dropped_blocks(&self) -> u64 {
        self.dropped_blocks.load(Ordering::Relaxed)
    }
}

impl RealtimeBeatTracker {
    pub fn new(sample_rate: u32) -> Result<Self, BeatTrackerError> {
        let model = match std::env::var("TERB_BPM_MODEL") {
            Ok(path) => {
                if !Path::new(&path).is_file() {
                    return Err(BeatTrackerError::ModelMissing(path));
                }
                RtenGraph::load_file(path)
            }
            Err(_) => RtenGraph::load_static_slice(EMBEDDED_MODEL),
        }
        .map_err(|e| BeatTrackerError::Model(e.to_string()))?;
        Self::from_graph(sample_rate, model)
    }

    pub fn with_model(sample_rate: u32, path: impl AsRef<Path>) -> Result<Self, BeatTrackerError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(BeatTrackerError::ModelMissing(path.display().to_string()));
        }
        let model =
            RtenGraph::load_file(path).map_err(|e| BeatTrackerError::Model(e.to_string()))?;
        Self::from_graph(sample_rate, model)
    }

    fn from_graph(sample_rate: u32, model: RtenGraph) -> Result<Self, BeatTrackerError> {
        let named = |name: &str| {
            model
                .input_ids()
                .iter()
                .chain(model.output_ids())
                .copied()
                .find(|&id| model.node_info(id).and_then(|x| x.name()) == Some(name))
                .ok_or_else(|| BeatTrackerError::Model(format!("model is missing tensor {name}")))
        };
        let model_input = named("spectrogram")?;
        let beat_output = named("beat")?;
        let downbeat_output = named("downbeat")?;
        Ok(Self {
            features: MelExtractor::new(sample_rate).map_err(BeatTrackerError::Audio)?,
            model,
            model_input,
            beat_output,
            downbeat_output,
            context: VecDeque::with_capacity(CONTEXT),
            input_buf: vec![0.0; MEL_BANDS * CONTEXT],
            feature_frames: Vec::with_capacity(8),
            pending_rms: VecDeque::with_capacity(INFERENCE_HOP + LOOKAHEAD),
            decoder: ActivationDecoder::default(),
            feature_seconds: 0.0,
            inference_seconds: 0.0,
            decode_seconds: 0.0,
            frames_since_inference: 0,
            emitted_any: false,
        })
    }

    pub fn consume(&mut self, samples: &[f32]) -> Option<BeatEstimate> {
        let started = Instant::now();
        self.feature_frames.clear();
        self.features
            .consume(samples, |frame| self.feature_frames.push(frame));
        self.feature_seconds += started.elapsed().as_secs_f64();
        let mut output = None;
        for index in 0..self.feature_frames.len() {
            let frame = self.feature_frames[index].clone();
            if let Some(activations) = self.infer(frame) {
                let started = Instant::now();
                let mut batch = None;
                for (beat, downbeat, rms) in activations {
                    self.decoder.observe_rms(rms);
                    if let Some(estimate) = self.decoder.push(beat, downbeat) {
                        batch = Some(merge_batch_estimate(batch, estimate));
                    }
                }
                output = batch.or(output);
                self.decode_seconds += started.elapsed().as_secs_f64();
            }
        }
        output
    }

    pub fn reset(&mut self) {
        self.features.reset();
        self.context.clear();
        self.decoder = ActivationDecoder::default();
        self.pending_rms.clear();
        self.frames_since_inference = 0;
        self.emitted_any = false;
    }

    pub fn timings(&self) -> StageTimings {
        StageTimings {
            feature_seconds: self.feature_seconds,
            inference_seconds: self.inference_seconds,
            decode_seconds: self.decode_seconds,
        }
    }

    fn infer(&mut self, frame: MelFrame) -> Option<Vec<(f32, f32, f32)>> {
        self.pending_rms.push_back(frame.rms);
        self.context.push_back(frame.values);
        self.frames_since_inference += 1;
        if self.context.len() > CONTEXT {
            self.context.pop_front();
        }
        let required = if self.emitted_any {
            INFERENCE_HOP
        } else {
            INFERENCE_HOP + LOOKAHEAD
        };
        if self.frames_since_inference < required {
            return None;
        }
        self.frames_since_inference = 0;
        self.emitted_any = true;
        self.input_buf.fill(0.0);
        let offset = CONTEXT - self.context.len();
        for t in 0..self.context.len() {
            for m in 0..MEL_BANDS {
                self.input_buf[(offset + t) * MEL_BANDS + m] = self.context[t][m];
            }
        }
        let input = RtenValue::from_shape(&[1, CONTEXT, MEL_BANDS], self.input_buf.clone()).ok()?;
        let started = Instant::now();
        let outputs = self
            .model
            .run(
                vec![(self.model_input, (&input).into())],
                &[self.beat_output, self.downbeat_output],
                None,
            )
            .ok()?;
        self.inference_seconds += started.elapsed().as_secs_f64();
        let mut outputs = outputs.into_iter();
        let beat = outputs.next()?.into_tensor::<f32>()?;
        let downbeat = outputs.next()?.into_tensor::<f32>()?;
        let beat = beat.to_vec();
        let downbeat = downbeat.to_vec();
        if beat.len() < CONTEXT || downbeat.len() < CONTEXT {
            return None;
        }
        let start = CONTEXT - LOOKAHEAD - INFERENCE_HOP;
        let mut result = Vec::with_capacity(INFERENCE_HOP);
        for i in start..start + INFERENCE_HOP {
            let rms = self.pending_rms.pop_front()?;
            result.push((sigmoid(beat[i]), sigmoid(downbeat[i]), rms));
        }
        Some(result)
    }
}

fn merge_batch_estimate(previous: Option<BeatEstimate>, current: BeatEstimate) -> BeatEstimate {
    let Some(previous) = previous else {
        return current;
    };
    BeatEstimate {
        bpm: current.bpm,
        confidence: current.confidence,
        beat_pulse: previous.beat_pulse.max(current.beat_pulse),
        phase: current.phase,
        downbeat_pulse: match (previous.downbeat_pulse, current.downbeat_pulse) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        },
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct StageTimings {
    pub feature_seconds: f64,
    pub inference_seconds: f64,
    pub decode_seconds: f64,
}

#[derive(Default)]
struct ActivationDecoder {
    values: VecDeque<f32>,
    rms: VecDeque<f32>,
    stable_bpm: Option<f32>,
    confidence: f32,
    frames: u64,
    last_peak: Option<u64>,
}

impl ActivationDecoder {
    fn observe_rms(&mut self, rms: f32) {
        self.rms.push_back(rms);
    }

    fn push(&mut self, activation: f32, downbeat: f32) -> Option<BeatEstimate> {
        self.frames += 1;
        self.values.push_back(activation.clamp(0.0, 1.0));
        if self.values.len() > MAX_HISTORY {
            self.values.pop_front();
        }
        while self.rms.len() > MAX_HISTORY {
            self.rms.pop_front();
        }
        let now = self.values.len() - 1;
        if is_peak(&self.values, now) {
            self.last_peak = Some(self.frames);
        }
        if self.values.len() < MIN_HISTORY {
            return None;
        }

        if self.frames % 10 == 0 {
            let audible = self
                .rms
                .iter()
                .rev()
                .take(100)
                .filter(|x| **x > SILENCE_RMS)
                .count();
            if audible < 25 {
                self.confidence *= 0.8;
            } else if let Some((raw_bpm, raw_confidence)) =
                decode_tempo(&self.values, self.stable_bpm)
            {
                if raw_confidence < 0.12 {
                    self.confidence *= 0.85;
                } else {
                    let bpm = match self.stable_bpm {
                        None => raw_bpm,
                        Some(old) => {
                            let octave_target = octave_nearest(raw_bpm, old);
                            let delta = (octave_target / old).ln().abs();
                            let follow = if delta < 0.025 {
                                0.28
                            } else if raw_confidence > 0.62 {
                                0.14
                            } else {
                                0.045
                            };
                            old + (octave_target - old) * follow
                        }
                    };
                    self.stable_bpm = Some(bpm);
                    self.confidence += (raw_confidence - self.confidence) * 0.25;
                }
            }
        }

        let bpm = self.stable_bpm?;
        let beat_pulse = self
            .last_peak
            .map(|peak| {
                let decay = (-((self.frames - peak) as f32) / 6.0).exp();
                if peak == self.frames {
                    self.values[now].max(decay)
                } else {
                    decay
                }
            })
            .unwrap_or(0.0);
        let period = FPS * 60.0 / bpm;
        let phase = self
            .last_peak
            .map(|p| ((self.frames - p) as f32 / period).fract())
            .unwrap_or(0.0);
        Some(BeatEstimate {
            bpm,
            confidence: self.confidence.clamp(0.0, 1.0),
            beat_pulse,
            phase,
            downbeat_pulse: Some(downbeat.clamp(0.0, 1.0)),
        })
    }
}

fn decode_tempo(values: &VecDeque<f32>, stable: Option<f32>) -> Option<(f32, f32)> {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let x: Vec<f32> = values.iter().map(|v| (v - mean).max(0.0)).collect();
    let energy: f32 = x.iter().map(|v| v * v).sum();
    if energy < 1e-4 {
        return None;
    }
    let peaks: Vec<usize> = (1..x.len() - 1)
        .filter(|&i| x[i] >= 0.18 && x[i] > x[i - 1] && x[i] >= x[i + 1])
        .collect();

    let mut scored = Vec::new();
    let mut bpm = 60.0;
    while bpm <= 210.0 {
        let lag = FPS * 60.0 / bpm;
        let ac = fractional_autocorrelation(&x, lag);
        let interval = interval_support(&peaks, lag);
        let continuity = stable
            .map(|s| (-2.0 * (bpm / s).ln().abs()).exp())
            .unwrap_or(0.6);
        let tempo_prior = (-0.10 * (bpm / 120.0).log2().abs()).exp();
        let score = (0.62 * ac + 0.28 * interval + 0.10 * continuity) * tempo_prior;
        scored.push((score, bpm));
        bpm += 0.25;
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (best_score, best_bpm) = scored[0];
    // Explicitly challenge the winning metrical level with 0.5x and 2x.
    let family = [best_bpm * 0.5, best_bpm, best_bpm * 2.0];
    let chosen = family
        .into_iter()
        .filter(|b| (60.0..=210.0).contains(b))
        .map(|b| {
            scored
                .iter()
                .min_by(|a, c| (a.1 - b).abs().total_cmp(&(c.1 - b).abs()))
                .copied()
                .unwrap()
        })
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap();
    let runner = scored
        .iter()
        .find(|(_, b)| (b - chosen.1).abs() > 2.0)
        .map(|x| x.0)
        .unwrap_or(0.0);
    let separation = ((chosen.0 - runner).max(0.0) / chosen.0.max(1e-6)).clamp(0.0, 1.0);
    Some((
        chosen.1,
        (best_score * 0.75 + separation * 0.25).clamp(0.0, 1.0),
    ))
}

fn fractional_autocorrelation(x: &[f32], lag: f32) -> f32 {
    let integer = lag.floor() as usize;
    let frac = lag - integer as f32;
    if integer + 1 >= x.len() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut a2 = 0.0;
    let mut b2 = 0.0;
    for i in integer + 1..x.len() {
        let delayed = x[i - integer] * (1.0 - frac) + x[i - integer - 1] * frac;
        dot += x[i] * delayed;
        a2 += x[i] * x[i];
        b2 += delayed * delayed;
    }
    dot / (a2 * b2 + 1e-8).sqrt()
}

fn interval_support(peaks: &[usize], lag: f32) -> f32 {
    if peaks.len() < 2 {
        return 0.0;
    }
    let sum: f32 = peaks
        .windows(2)
        .map(|p| {
            let d = (p[1] - p[0]) as f32;
            let direct = (-(d - lag).powi(2) / (2.0 * 1.5_f32.powi(2))).exp();
            let double = 0.65 * (-(d - 2.0 * lag).powi(2) / 8.0).exp();
            direct.max(double)
        })
        .sum();
    sum / (peaks.len() - 1) as f32
}

fn octave_nearest(mut bpm: f32, reference: f32) -> f32 {
    while bpm / reference > 1.5 {
        bpm *= 0.5;
    }
    while reference / bpm > 1.5 {
        bpm *= 2.0;
    }
    bpm.clamp(60.0, 210.0)
}

fn is_peak(values: &VecDeque<f32>, i: usize) -> bool {
    i > 1 && values[i] > 0.25 && values[i - 1] < values[i]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued_tracker() -> (AsyncBeatTracker, mpsc::Receiver<SampleBlock>) {
        let (sender, receiver) = mpsc::sync_channel(2);
        let tracker = AsyncBeatTracker {
            sender,
            latest: Arc::new(Mutex::new(PublishedEstimate {
                generation: 0,
                sequence: 1,
                estimate: Some(BeatEstimate {
                    bpm: 120.0,
                    confidence: 0.8,
                    beat_pulse: 1.0,
                    phase: 0.0,
                    downbeat_pulse: None,
                }),
            })),
            generation: Arc::new(AtomicU64::new(0)),
            dropped_blocks: Arc::new(AtomicU64::new(0)),
        };
        (tracker, receiver)
    }

    #[test]
    fn reset_invalidates_full_queue_and_inflight_estimates() {
        let (tracker, receiver) = queued_tracker();
        assert!(tracker.submit(&[0.1]));
        assert!(tracker.submit(&[0.2]));
        assert!(tracker.latest().is_some());

        tracker.reset();

        assert!(tracker.latest().is_none());
        let old = receiver.try_recv().unwrap();
        receiver.try_recv().unwrap();
        // A computation that finishes after reset must not revive its result.
        tracker.latest.lock().unwrap().generation = old.generation;
        assert!(tracker.latest().is_none());
        assert!(tracker.submit(&[0.3]));
        let new = receiver.try_recv().unwrap();
        assert_ne!(new.generation, old.generation);
        assert_eq!(new.generation, tracker.generation.load(Ordering::Acquire));
        assert_eq!(tracker.dropped_blocks(), 0);
    }

    #[test]
    fn dropped_audio_starts_a_new_history() {
        let (tracker, receiver) = queued_tracker();
        assert!(tracker.submit(&[0.1]));
        assert!(tracker.submit(&[0.2]));
        assert!(!tracker.submit(&[0.3]));

        assert_eq!(tracker.dropped_blocks(), 1);
        assert!(tracker.latest().is_none());
        let old = receiver.try_recv().unwrap();
        receiver.try_recv().unwrap();
        assert!(tracker.submit(&[0.4]));
        let new = receiver.try_recv().unwrap();
        assert_ne!(old.generation, new.generation);
        assert_eq!(new.samples, vec![0.4]);
    }

    #[test]
    fn activation_decoder_finds_120_bpm() {
        let mut d = ActivationDecoder::default();
        let mut out = None;
        for i in 0..600 {
            d.rms.push_back(0.1);
            out = d.push(if i % 25 == 0 { 0.95 } else { 0.02 }, 0.01).or(out);
        }
        assert!((out.unwrap().bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn activation_decoder_keeps_peaks_between_tempo_updates() {
        let mut decoder = ActivationDecoder::default();
        for frame in 0..600 {
            decoder.observe_rms(0.1);
            decoder.push(if frame % 25 == 0 { 0.95 } else { 0.02 }, 0.01);
        }

        decoder.observe_rms(0.1);
        decoder.push(0.02, 0.01);
        decoder.observe_rms(0.1);
        let estimate = decoder
            .push(0.95, 0.01)
            .expect("a known tempo should emit on every activation frame");

        assert!(estimate.beat_pulse >= 0.95);
        assert!((estimate.bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn inference_batch_preserves_strongest_detected_pulse() {
        let first = BeatEstimate {
            bpm: 120.0,
            confidence: 0.7,
            beat_pulse: 0.9,
            phase: 0.0,
            downbeat_pulse: Some(0.8),
        };
        let last = BeatEstimate {
            bpm: 121.0,
            confidence: 0.8,
            beat_pulse: 0.2,
            phase: 0.3,
            downbeat_pulse: Some(0.1),
        };

        let merged = merge_batch_estimate(Some(first), last);

        assert_eq!(merged.bpm, 121.0);
        assert_eq!(merged.confidence, 0.8);
        assert_eq!(merged.beat_pulse, 0.9);
        assert_eq!(merged.phase, 0.3);
        assert_eq!(merged.downbeat_pulse, Some(0.8));
    }
}
