use std::{collections::VecDeque, sync::Arc};

use rustfft::{num_complex::Complex, Fft, FftPlanner};

use crate::analysis::sample_frequency_band;

pub const BPM_MIN: f32 = 60.0;
pub const BPM_MAX: f32 = 210.0;
pub const BPM_PULSE_DECAY_SECONDS: f32 = 0.18;

const BPM_FFT_SIZE: usize = 2048;
const BPM_HOP_SIZE: usize = 512;
const BPM_BAND_COUNT: usize = 48;
const BPM_MIN_FREQUENCY: f32 = 40.0;
const BPM_MAX_FREQUENCY: f32 = 9_000.0;
const BPM_LOG_GAIN: f32 = 1_600.0;
const BPM_HISTORY_SECONDS: f32 = 16.0;
const BPM_MIN_HISTORY_SECONDS: f32 = 4.0;
const BPM_ESTIMATE_INTERVAL_FRAMES: usize = 16;
const BPM_SILENCE_GATE: f32 = 0.000_12;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpmEstimate {
    pub bpm: f32,
    pub confidence: f32,
}

#[derive(Clone)]
pub struct BpmAnalyzer {
    sample_rate: f32,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    pending: Vec<f32>,
    pending_start: usize,
    fft_buffer: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    current_bands: Vec<f32>,
    previous_bands: Vec<f32>,
    onset_average: f32,
    analysis_sample: u64,
    frames_since_estimate: usize,
    envelope: VecDeque<OnsetFrame>,
    stable_bpm: Option<f32>,
    stable_confidence: f32,
}

#[derive(Clone, Copy, Debug)]
struct OnsetFrame {
    time: f32,
    strength: f32,
}

impl BpmAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(BPM_FFT_SIZE);
        let window: Vec<f32> = (0..BPM_FFT_SIZE)
            .map(|index| {
                let position = index as f32 / (BPM_FFT_SIZE - 1) as f32;
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * position).cos()
            })
            .collect();

        Self {
            sample_rate,
            fft,
            window,
            pending: Vec::new(),
            pending_start: 0,
            fft_buffer: vec![Complex::new(0.0, 0.0); BPM_FFT_SIZE],
            magnitudes: vec![0.0; BPM_FFT_SIZE / 2],
            current_bands: vec![0.0; BPM_BAND_COUNT],
            previous_bands: vec![0.0; BPM_BAND_COUNT],
            onset_average: 0.0,
            analysis_sample: 0,
            frames_since_estimate: 0,
            envelope: VecDeque::new(),
            stable_bpm: None,
            stable_confidence: 0.0,
        }
    }

    pub fn reset(&mut self) {
        let sample_rate = self.sample_rate;
        *self = Self::new(sample_rate);
    }

    pub fn consume(&mut self, samples: &[f32]) -> Option<BpmEstimate> {
        if samples.is_empty() {
            return None;
        }

        self.pending.extend_from_slice(samples);
        let mut estimate = None;
        while self.pending.len().saturating_sub(self.pending_start) >= BPM_FFT_SIZE {
            estimate = self.process_frame(self.pending_start).or(estimate);
            self.pending_start += BPM_HOP_SIZE;
            self.analysis_sample = self.analysis_sample.saturating_add(BPM_HOP_SIZE as u64);
        }

        if self.pending_start >= BPM_FFT_SIZE * 2 {
            self.pending.drain(..self.pending_start);
            self.pending_start = 0;
        }

        estimate
    }

    fn process_frame(&mut self, start: usize) -> Option<BpmEstimate> {
        let end = start + BPM_FFT_SIZE;
        let source = &self.pending[start..end];
        let source_rms =
            (source.iter().map(|sample| sample * sample).sum::<f32>() / BPM_FFT_SIZE as f32).sqrt();
        if source_rms < BPM_SILENCE_GATE {
            self.push_onset_frame(0.0);
            return self.estimate_if_due();
        }

        let mean = source.iter().sum::<f32>() / BPM_FFT_SIZE as f32;
        for (index, output) in self.fft_buffer.iter_mut().enumerate() {
            output.re = (self.pending[start + index] - mean) * self.window[index];
            output.im = 0.0;
        }
        self.fft.process(&mut self.fft_buffer);

        let half = BPM_FFT_SIZE / 2;
        for index in 1..half {
            self.magnitudes[index] = self.fft_buffer[index].norm();
        }

        self.update_log_frequency_bands();
        let flux = spectral_flux(&self.current_bands, &self.previous_bands);
        std::mem::swap(&mut self.current_bands, &mut self.previous_bands);
        self.onset_average = if self.onset_average <= 0.0 {
            flux
        } else {
            lerp(self.onset_average, flux, 0.030)
        };

        let baseline = self.onset_average * 0.72;
        let strength =
            ((flux - baseline).max(0.0) / (self.onset_average + 0.000_001)).clamp(0.0, 5.0);
        self.push_onset_frame(strength);
        self.estimate_if_due()
    }

    fn push_onset_frame(&mut self, strength: f32) {
        let center_sample = self.analysis_sample + (BPM_FFT_SIZE / 2) as u64;
        let time = center_sample as f32 / self.sample_rate.max(1.0);
        self.envelope.push_back(OnsetFrame { time, strength });
        let history_start =
            self.analysis_sample as f32 / self.sample_rate.max(1.0) - BPM_HISTORY_SECONDS;
        while self
            .envelope
            .front()
            .map(|frame| frame.time < history_start)
            .unwrap_or(false)
        {
            self.envelope.pop_front();
        }
    }

    fn update_log_frequency_bands(&mut self) {
        let max_frequency = (self.sample_rate / 2.0).min(BPM_MAX_FREQUENCY);
        let ratio = max_frequency / BPM_MIN_FREQUENCY;
        for band_index in 0..BPM_BAND_COUNT {
            let lower_t = band_index as f32 / BPM_BAND_COUNT as f32;
            let upper_t = (band_index + 1) as f32 / BPM_BAND_COUNT as f32;
            let lower_frequency = BPM_MIN_FREQUENCY * ratio.powf(lower_t);
            let upper_frequency = BPM_MIN_FREQUENCY * ratio.powf(upper_t);
            let lower_bin = ((lower_frequency / self.sample_rate) * BPM_FFT_SIZE as f32).max(1.0);
            let upper_bin =
                ((upper_frequency / self.sample_rate) * BPM_FFT_SIZE as f32).max(lower_bin + 0.5);
            let band = sample_frequency_band(&self.magnitudes, lower_bin, upper_bin);
            self.current_bands[band_index] = (1.0 + band.rms * BPM_LOG_GAIN).ln();
        }
    }

    fn estimate_if_due(&mut self) -> Option<BpmEstimate> {
        self.frames_since_estimate += 1;
        if self.frames_since_estimate < BPM_ESTIMATE_INTERVAL_FRAMES {
            return None;
        }
        self.frames_since_estimate = 0;
        self.estimate()
    }

    fn estimate(&mut self) -> Option<BpmEstimate> {
        let estimate = estimate_bpm_from_envelope(
            &self.envelope,
            BPM_HOP_SIZE as f32 / self.sample_rate.max(1.0),
            self.stable_bpm,
        )?;

        let follow = if estimate.confidence >= self.stable_confidence {
            0.36
        } else {
            0.16
        };
        let bpm = self
            .stable_bpm
            .map(|stable| lerp_tempo(stable, estimate.bpm, follow))
            .unwrap_or(estimate.bpm);
        self.stable_bpm = Some(bpm);
        self.stable_confidence = lerp(self.stable_confidence, estimate.confidence, 0.28);
        Some(BpmEstimate {
            bpm,
            confidence: self.stable_confidence,
        })
    }
}

fn spectral_flux(current: &[f32], previous: &[f32]) -> f32 {
    if current.is_empty() || previous.is_empty() {
        return 0.0;
    }

    let count = current.len().min(previous.len());
    let mut changes = [0.0_f32; BPM_BAND_COUNT];
    let mut total = 0.0_f32;
    for (index, (current, previous)) in current.iter().zip(previous).take(count).enumerate() {
        let change = (current - previous).max(0.0);
        changes[index] = change;
        total += change;
    }
    changes[..count].sort_unstable_by(f32::total_cmp);
    let median = changes[count / 2];
    let mean = total / count as f32;
    median * 0.65 + mean * 0.35
}

fn estimate_bpm_from_envelope(
    envelope: &VecDeque<OnsetFrame>,
    hop_seconds: f32,
    stable_bpm: Option<f32>,
) -> Option<BpmEstimate> {
    let minimum_frames = (BPM_MIN_HISTORY_SECONDS / hop_seconds).ceil() as usize;
    if envelope.len() < minimum_frames.max(16) {
        return None;
    }

    let mut values: Vec<f32> = envelope.iter().map(|frame| frame.strength).collect();
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    for value in &mut values {
        *value = (*value - mean).max(0.0);
    }
    let energy = values.iter().map(|value| value * value).sum::<f32>();
    if energy < 0.000_01 {
        return None;
    }

    let mut best: Option<BpmEstimate> = None;
    let mut second_score = 0.0_f32;
    let min_lag = (60.0 / BPM_MAX / hop_seconds).round().max(1.0) as usize;
    let max_lag = (60.0 / BPM_MIN / hop_seconds)
        .round()
        .min(values.len().saturating_sub(1) as f32) as usize;
    if min_lag >= max_lag {
        return None;
    }
    let mut scores = vec![0.0_f32; max_lag + 1];

    for (lag, score_slot) in scores
        .iter_mut()
        .enumerate()
        .take(max_lag + 1)
        .skip(min_lag)
    {
        let bpm = 60.0 / (lag as f32 * hop_seconds);
        let raw_score = autocorrelation_score(&values, lag);
        if raw_score <= 0.0 {
            continue;
        }

        let continuity = stable_bpm
            .map(|stable| tempo_similarity(bpm, stable).powf(1.6))
            .unwrap_or(1.0);
        let tempo_prior = tempo_prior(bpm);
        let harmonic = harmonic_support(&values, lag);
        let score = raw_score * tempo_prior * (0.74 + harmonic * 0.26) * (0.72 + continuity * 0.28);
        *score_slot = score;

        match best {
            Some(current) if score <= current.confidence => {
                second_score = second_score.max(score);
            }
            Some(current) => {
                second_score = second_score.max(current.confidence);
                best = Some(BpmEstimate {
                    bpm,
                    confidence: score,
                });
            }
            None => {
                best = Some(BpmEstimate {
                    bpm,
                    confidence: score,
                });
            }
        }
    }

    let mut best = best?;
    let best_lag = (60.0 / (best.bpm * hop_seconds)).round() as usize;
    if best_lag > min_lag && best_lag < max_lag {
        let left = scores[best_lag - 1];
        let center = scores[best_lag];
        let right = scores[best_lag + 1];
        let denominator = left - 2.0 * center + right;
        if denominator.abs() > 0.000_001 {
            let offset = (0.5 * (left - right) / denominator).clamp(-0.5, 0.5);
            best.bpm = 60.0 / ((best_lag as f32 + offset) * hop_seconds);
        }
    }
    let separation = if best.confidence <= 0.0 {
        0.0
    } else {
        ((best.confidence - second_score).max(0.0) / best.confidence).clamp(0.0, 1.0)
    };
    let confidence = (best.confidence * (0.56 + separation * 0.44)).clamp(0.0, 1.0);
    Some(BpmEstimate {
        bpm: best.bpm,
        confidence,
    })
}

fn autocorrelation_score(values: &[f32], lag: usize) -> f32 {
    if lag == 0 || lag >= values.len() {
        return 0.0;
    }

    let mut numerator = 0.0_f32;
    let mut left_energy = 0.0_f32;
    let mut right_energy = 0.0_f32;
    for index in lag..values.len() {
        let left = values[index];
        let right = values[index - lag];
        let recency = 0.5 + 0.5 * index as f32 / values.len().max(1) as f32;
        numerator += left * right * recency;
        left_energy += left * left * recency;
        right_energy += right * right * recency;
    }

    numerator / (left_energy.sqrt() * right_energy.sqrt() + 0.000_001)
}

fn harmonic_support(values: &[f32], lag: usize) -> f32 {
    let double = autocorrelation_score(values, lag.saturating_mul(2));
    let half = if lag >= 2 {
        autocorrelation_score(values, lag / 2)
    } else {
        0.0
    };
    (double * 0.65 + half * 0.35).clamp(0.0, 1.0)
}

fn tempo_prior(bpm: f32) -> f32 {
    let center = 118.0_f32;
    let octaves = (bpm / center).max(0.001).log2().abs();
    (1.0 - octaves * 0.16).clamp(0.78, 1.0)
}

fn tempo_similarity(left: f32, right: f32) -> f32 {
    let ratio = (left / right.max(0.001)).max(0.001);
    let octave_distance = ratio.log2().abs();
    (1.0 - octave_distance).clamp(0.0, 1.0)
}

fn lerp_tempo(current: f32, target: f32, mix: f32) -> f32 {
    let mut target = target;
    while target / current.max(0.001) > 1.5 {
        target *= 0.5;
    }
    while current / target.max(0.001) > 1.5 {
        target *= 2.0;
    }
    lerp(current, target, mix)
}

fn lerp(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_waits_for_enough_rhythm_history() {
        let mut analyzer = BpmAnalyzer::new(48_000.0);
        let click_period = 24_000;
        let mut estimate = None;

        for chunk_start in (0..48_000 * 3).step_by(512) {
            let mut samples = vec![0.0; 512];
            for (offset, sample) in samples.iter_mut().enumerate() {
                if (chunk_start + offset) % click_period < 240 {
                    *sample = 0.8;
                }
            }
            estimate = analyzer.consume(&samples).or(estimate);
        }

        assert!(estimate.is_none());
    }

    #[test]
    fn estimator_locks_to_regular_low_frequency_onsets() {
        let mut analyzer = BpmAnalyzer::new(48_000.0);
        let mut estimate = None;
        let mut samples = Vec::new();
        let seconds = 10.0;
        let total = (48_000.0 * seconds) as usize;

        for index in 0..total {
            let beat_phase = index % 24_000;
            let click = if beat_phase < 480 {
                let decay = 1.0 - beat_phase as f32 / 480.0;
                (std::f32::consts::TAU * 70.0 * index as f32 / 48_000.0).sin() * decay * 0.9
            } else {
                0.0
            };
            samples.push(click);
            if samples.len() == 512 {
                estimate = analyzer.consume(&samples).or(estimate);
                samples.clear();
            }
        }

        let estimate = estimate.expect("regular clicks should produce bpm");
        assert!((estimate.bpm - 120.0).abs() <= 2.0);
        assert!(estimate.confidence > 0.25);
    }

    #[test]
    fn estimator_uses_wideband_onset_flux_not_only_low_end() {
        let mut analyzer = BpmAnalyzer::new(48_000.0);
        let mut estimate = None;
        let mut samples = Vec::new();
        let seconds = 12.0;
        let total = (48_000.0 * seconds) as usize;
        let beat_period = (48_000.0 * 60.0 / 128.0) as usize;

        for index in 0..total {
            let beat_phase = index % beat_period;
            let click = if beat_phase < 360 {
                let decay = 1.0 - beat_phase as f32 / 360.0;
                let high = (std::f32::consts::TAU * 4_800.0 * index as f32 / 48_000.0).sin();
                let mid = (std::f32::consts::TAU * 1_600.0 * index as f32 / 48_000.0).sin();
                (high * 0.55 + mid * 0.35) * decay
            } else {
                0.0
            };
            let tonal_bed = (std::f32::consts::TAU * 92.0 * index as f32 / 48_000.0).sin() * 0.08;
            samples.push(click + tonal_bed);
            if samples.len() == 512 {
                estimate = analyzer.consume(&samples).or(estimate);
                samples.clear();
            }
        }

        let estimate = estimate.expect("wideband attacks should produce bpm");
        assert!((estimate.bpm - 128.0).abs() <= 3.0);
        assert!(estimate.confidence > 0.20);
    }

    #[test]
    fn estimator_prefers_continuity_over_half_time_flip() {
        let mut analyzer = BpmAnalyzer::new(48_000.0);
        let mut estimate = None;
        let mut samples = Vec::new();
        let seconds = 18.0;
        let total = (48_000.0 * seconds) as usize;
        let beat_period = (48_000.0 * 60.0 / 120.0) as usize;

        for index in 0..total {
            let beat_phase = index % beat_period;
            let half_time_accent = index % (beat_period * 2) < 560;
            let beat = if beat_phase < 420 {
                let decay = 1.0 - beat_phase as f32 / 420.0;
                let carrier = if half_time_accent { 130.0 } else { 2_400.0 };
                (std::f32::consts::TAU * carrier * index as f32 / 48_000.0).sin()
                    * decay
                    * if half_time_accent { 1.0 } else { 0.55 }
            } else {
                0.0
            };
            samples.push(beat);
            if samples.len() == 512 {
                estimate = analyzer.consume(&samples).or(estimate);
                samples.clear();
            }
        }

        let estimate = estimate.expect("regular tempo should produce bpm");
        assert!((estimate.bpm - 120.0).abs() <= 4.0);
        assert!(estimate.confidence > 0.20);
    }
}
