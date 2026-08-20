use std::{collections::VecDeque, sync::Arc};

use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub const MODEL_SAMPLE_RATE: u32 = 22_050;
pub const FFT_SIZE: usize = 1024;
pub const HOP_SIZE: usize = 441;
pub const MEL_BANDS: usize = 128;

#[derive(Clone, Debug)]
pub struct MelFrame {
    pub values: [f32; MEL_BANDS],
    pub rms: f32,
}

/// Streaming, allocation-bounded preprocessing shared by CLI and application.
///
/// Resampling uses a stateful linear interpolator. It is intentionally simple and
/// deterministic so Python parity is straightforward; input chunks do not affect output.
pub struct MelExtractor {
    step: f64,
    next_input_pos: f64,
    input_base: u64,
    previous: Option<f32>,
    resampled: VecDeque<f32>,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    fft_buf: Vec<Complex<f32>>,
    mel: Vec<Vec<(usize, f32)>>,
}

impl MelExtractor {
    pub fn new(input_rate: u32) -> Result<Self, &'static str> {
        if input_rate == 0 {
            return Err("sample rate must be non-zero");
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window = (0..FFT_SIZE)
            .map(|i| {
                let x = i as f32 / FFT_SIZE as f32;
                0.5 - 0.5 * (std::f32::consts::TAU * x).cos()
            })
            .collect();
        let mut resampled = VecDeque::with_capacity(FFT_SIZE * 2);
        resampled.resize(FFT_SIZE / 2, 0.0);
        Ok(Self {
            step: input_rate as f64 / MODEL_SAMPLE_RATE as f64,
            next_input_pos: 0.0,
            input_base: 0,
            previous: None,
            resampled,
            fft,
            window,
            fft_buf: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            mel: mel_filterbank(),
        })
    }

    pub fn reset(&mut self) {
        self.next_input_pos = 0.0;
        self.input_base = 0;
        self.previous = None;
        self.resampled.clear();
        self.resampled.resize(FFT_SIZE / 2, 0.0);
        self.fft_buf.fill(Complex::new(0.0, 0.0));
    }

    pub fn consume<F: FnMut(MelFrame)>(&mut self, samples: &[f32], mut emit: F) {
        if samples.is_empty() {
            return;
        }
        let chunk_start = self.input_base;
        let chunk_end = chunk_start + samples.len() as u64;
        while self.next_input_pos < chunk_end as f64 {
            let left_abs = self.next_input_pos.floor() as i64;
            let frac = (self.next_input_pos - left_abs as f64) as f32;
            let left = sample_at(samples, chunk_start, left_abs, self.previous);
            let right = sample_at(samples, chunk_start, left_abs + 1, self.previous);
            if let (Some(a), Some(b)) = (left, right) {
                self.resampled.push_back(a + (b - a) * frac);
                self.next_input_pos += self.step;
                while self.resampled.len() >= FFT_SIZE {
                    emit(self.frame());
                    for _ in 0..HOP_SIZE {
                        self.resampled.pop_front();
                    }
                }
            } else {
                break;
            }
        }
        self.input_base = chunk_end;
        self.previous = samples.last().copied();
    }

    fn frame(&mut self) -> MelFrame {
        let mut square_sum = 0.0;
        for i in 0..FFT_SIZE {
            let sample = self.resampled[i];
            square_sum += sample * sample;
            self.fft_buf[i] = Complex::new(sample * self.window[i], 0.0);
        }
        self.fft.process(&mut self.fft_buf);
        let mut values = [0.0; MEL_BANDS];
        for (out, filter) in values.iter_mut().zip(&self.mel) {
            let magnitude = filter.iter().fold(0.0, |sum, &(bin, weight)| {
                sum + self.fft_buf[bin].norm() * weight
            });
            *out = (1.0 + 1_000.0 * magnitude / (FFT_SIZE as f32).sqrt()).ln();
        }
        MelFrame {
            values,
            rms: (square_sum / FFT_SIZE as f32).sqrt(),
        }
    }
}

fn sample_at(samples: &[f32], base: u64, absolute: i64, previous: Option<f32>) -> Option<f32> {
    if absolute == base as i64 - 1 {
        return previous;
    }
    let relative = absolute - base as i64;
    (relative >= 0)
        .then(|| samples.get(relative as usize).copied())
        .flatten()
}

fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    if hz < 1000.0 {
        hz / F_SP
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4_f32.ln() / 27.0)
    }
}
fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    if mel < 15.0 {
        mel * F_SP
    } else {
        1000.0 * (6.4_f32.ln() / 27.0 * (mel - 15.0)).exp()
    }
}

fn mel_filterbank() -> Vec<Vec<(usize, f32)>> {
    let lo = hz_to_mel(30.0);
    let hi = hz_to_mel(MODEL_SAMPLE_RATE as f32 / 2.0);
    let points: Vec<f32> = (0..MEL_BANDS + 2)
        .map(|i| {
            mel_to_hz(lo + (hi - lo) * i as f32 / (MEL_BANDS + 1) as f32) * FFT_SIZE as f32
                / MODEL_SAMPLE_RATE as f32
        })
        .collect();
    (0..MEL_BANDS)
        .map(|m| {
            let (left, center, right) = (points[m], points[m + 1], points[m + 2]);
            let mut filter = Vec::new();
            for bin in
                left.floor().max(0.0) as usize..=right.ceil().min((FFT_SIZE / 2) as f32) as usize
            {
                let x = bin as f32;
                let weight = if x <= center {
                    (x - left) / (center - left).max(1e-6)
                } else {
                    (right - x) / (right - center).max(1e-6)
                }
                .max(0.0);
                if weight > 0.0 {
                    filter.push((bin, weight));
                }
            }
            filter
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_does_not_change_features() {
        let audio: Vec<f32> = (0..48_000 * 3)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        let extract = |chunk: usize| {
            let mut result = Vec::new();
            let mut mel = MelExtractor::new(48_000).unwrap();
            for samples in audio.chunks(chunk) {
                mel.consume(samples, |x| result.push(x));
            }
            result
        };
        let a = extract(127);
        let b = extract(2048);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(x
                .values
                .iter()
                .zip(y.values)
                .all(|(a, b)| (a - b).abs() < 1e-5));
        }
    }
}
