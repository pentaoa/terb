#![allow(dead_code)]

use std::{io, sync::mpsc::Sender};

#[cfg(target_os = "windows")]
use std::sync::Arc;

pub(crate) const TARGET_SAMPLE_RATE: u32 = 48_000;

pub(crate) enum AudioEvent {
    Samples(u64, AudioSamples),
    Status(u64, String),
    Exit(u64, Option<i32>),
    Error(u64, String),
}

pub(crate) struct AudioSamples {
    pub mono: Vec<f32>,
    pub left_level: f32,
    pub right_level: f32,
}

pub(crate) struct AudioProcess {
    #[cfg(target_os = "macos")]
    child: std::process::Child,
    #[cfg(target_os = "windows")]
    stop: Arc<std::sync::atomic::AtomicBool>,
    #[cfg(target_os = "windows")]
    thread: Option<std::thread::JoinHandle<()>>,
}

impl AudioProcess {
    pub(crate) fn spawn(tx: Sender<AudioEvent>, capture_id: u64) -> io::Result<Self> {
        #[cfg(target_os = "macos")]
        {
            return macos::spawn(tx, capture_id);
        }
        #[cfg(target_os = "windows")]
        {
            return windows::spawn(tx, capture_id);
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let _ = (tx, capture_id);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "system audio capture is only available on macOS and Windows",
            ))
        }
    }

    pub(crate) fn stop(&mut self) {
        #[cfg(target_os = "macos")]
        {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        #[cfg(target_os = "windows")]
        {
            self.stop
                .store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

impl Drop for AudioProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

fn audio_level_from_square_sum(square_sum: f32, count: usize) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let rms = (square_sum / count as f32).sqrt();
    let db = 20.0 * rms.max(0.000_001).log10();
    ((db + 60.0) / 54.0).clamp(0.0, 1.0).powf(0.85)
}

fn samples_from_interleaved_stereo(frames: &[f32]) -> AudioSamples {
    let frame_count = frames.len() / 2;
    let mut mono = Vec::with_capacity(frame_count);
    let mut left_square_sum = 0.0_f32;
    let mut right_square_sum = 0.0_f32;
    for chunk in frames.chunks_exact(2) {
        let left = chunk[0];
        let right = chunk[1];
        left_square_sum += left * left;
        right_square_sum += right * right;
        mono.push((left + right) * 0.5);
    }
    AudioSamples {
        left_level: audio_level_from_square_sum(left_square_sum, frame_count),
        right_level: audio_level_from_square_sum(right_square_sum, frame_count),
        mono,
    }
}

fn take_interleaved_f32_frames(pending: &mut Vec<u8>) -> Option<AudioSamples> {
    let frame_count = pending.len() / 8;
    if frame_count == 0 {
        return None;
    }
    let bytes_to_read = frame_count * 8;
    let mut frames = Vec::with_capacity(frame_count * 2);
    for chunk in pending[..bytes_to_read].chunks_exact(8) {
        frames.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        frames.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
    }
    pending.drain(0..bytes_to_read);
    Some(samples_from_interleaved_stereo(&frames))
}

#[derive(Clone, Copy)]
enum PcmSampleKind {
    F32,
    I16,
    I32,
}

struct PcmFormat {
    channels: usize,
    #[allow(dead_code)]
    sample_rate: u32,
    kind: PcmSampleKind,
    bytes_per_sample: usize,
}

impl PcmFormat {
    fn frame_bytes(&self) -> usize {
        self.channels * self.bytes_per_sample
    }
}

fn decode_pcm_to_stereo(bytes: &[u8], format: &PcmFormat) -> (Vec<f32>, usize) {
    let frame_bytes = format.frame_bytes();
    if frame_bytes == 0 {
        return (Vec::new(), 0);
    }
    let frame_count = bytes.len() / frame_bytes;
    let mut stereo = Vec::with_capacity(frame_count * 2);
    for frame in bytes[..frame_count * frame_bytes].chunks_exact(frame_bytes) {
        let left = decode_pcm_sample(frame, 0, format.kind, format.bytes_per_sample);
        let right = if format.channels > 1 {
            decode_pcm_sample(frame, 1, format.kind, format.bytes_per_sample)
        } else {
            left
        };
        stereo.push(left);
        stereo.push(right);
    }
    (stereo, frame_count * frame_bytes)
}

fn decode_pcm_sample(frame: &[u8], channel: usize, kind: PcmSampleKind, bytes_per_sample: usize) -> f32 {
    let start = channel * bytes_per_sample;
    let sample = &frame[start..start + bytes_per_sample];
    match kind {
        PcmSampleKind::F32 => f32::from_le_bytes(sample.try_into().unwrap_or([0; 4])),
        PcmSampleKind::I16 => {
            i16::from_le_bytes(sample.try_into().unwrap_or([0; 2])) as f32 / i16::MAX as f32
        }
        PcmSampleKind::I32 => {
            i32::from_le_bytes(sample.try_into().unwrap_or([0; 4])) as f32 / i32::MAX as f32
        }
    }
}

struct StereoResampler {
    step: f64,
    pos: f64,
    frames: std::collections::VecDeque<(f32, f32)>,
}

impl StereoResampler {
    fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            step: input_rate as f64 / output_rate.max(1) as f64,
            pos: 0.0,
            frames: std::collections::VecDeque::new(),
        }
    }

    fn push(&mut self, interleaved: &[f32], output: &mut Vec<f32>) {
        for pair in interleaved.chunks_exact(2) {
            self.frames.push_back((pair[0], pair[1]));
        }
        if (self.step - 1.0).abs() < f64::EPSILON {
            while let Some((left, right)) = self.frames.pop_front() {
                output.push(left);
                output.push(right);
            }
            return;
        }
        while self.pos + 1.0 < self.frames.len() as f64 {
            let index = self.pos.floor() as usize;
            let frac = (self.pos - index as f64) as f32;
            let (left0, right0) = self.frames[index];
            let (left1, right1) = self.frames[index + 1];
            output.push(left0 + (left1 - left0) * frac);
            output.push(right0 + (right1 - right0) * frac);
            self.pos += self.step;
        }
        let drop = self.pos.floor() as usize;
        self.frames.drain(0..drop);
        self.pos -= drop as f64;
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{take_interleaved_f32_frames, AudioEvent, AudioProcess};
    use std::{
        env, io,
        io::{BufRead, BufReader, Read},
        path::PathBuf,
        process::{Command, Stdio},
        sync::mpsc::Sender,
        thread,
    };

    const AUDIO_READ_FRAMES: usize = 512;

    pub(super) fn spawn(tx: Sender<AudioEvent>, capture_id: u64) -> io::Result<AudioProcess> {
        let helper = helper_path()?;
        let mut child = Command::new(helper)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing helper stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("missing helper stderr"))?;

        let stdout_tx = tx.clone();
        thread::spawn(move || read_audio_stdout(stdout, stdout_tx, capture_id));

        let stderr_tx = tx.clone();
        thread::spawn(move || read_helper_stderr(stderr, stderr_tx, capture_id));

        Ok(AudioProcess { child })
    }

    fn helper_path() -> io::Result<PathBuf> {
        let adjacent = env::current_exe()?.with_file_name("terb-audio-helper");
        if adjacent.is_file() {
            return Ok(adjacent);
        }
        option_env!("TERB_AUDIO_HELPER")
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "macOS audio helper is unavailable; place terb-audio-helper beside terb or rebuild on macOS",
                )
            })
    }

    fn read_audio_stdout(mut stdout: impl Read, tx: Sender<AudioEvent>, capture_id: u64) {
        let mut buffer = vec![0_u8; AUDIO_READ_FRAMES * 8];
        let mut pending = Vec::new();
        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(size) => {
                    pending.extend_from_slice(&buffer[..size]);
                    if let Some(samples) = take_interleaved_f32_frames(&mut pending) {
                        if tx.send(AudioEvent::Samples(capture_id, samples)).is_err() {
                            break;
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(AudioEvent::Error(capture_id, error.to_string()));
                    break;
                }
            }
        }
        let _ = tx.send(AudioEvent::Exit(capture_id, None));
    }

    fn read_helper_stderr(stderr: impl Read, tx: Sender<AudioEvent>, capture_id: u64) {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            let _ = tx.send(AudioEvent::Status(capture_id, line));
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use super::{
        decode_pcm_to_stereo, samples_from_interleaved_stereo, AudioEvent, AudioProcess, PcmFormat,
        PcmSampleKind, StereoResampler, TARGET_SAMPLE_RATE,
    };
    use std::{
        io,
        sync::{
            atomic::{AtomicBool, Ordering},
            mpsc::Sender,
            Arc,
        },
        thread,
        time::Duration,
    };
    use wasapi::{
        initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat,
    };

    pub(super) fn spawn(tx: Sender<AudioEvent>, capture_id: u64) -> io::Result<AudioProcess> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("terb-wasapi".into())
            .spawn(move || {
                if let Err(error) = capture_loop(tx.clone(), capture_id, thread_stop) {
                    let _ = tx.send(AudioEvent::Error(capture_id, error.to_string()));
                }
                let _ = tx.send(AudioEvent::Exit(capture_id, None));
            })?;
        Ok(AudioProcess {
            stop,
            thread: Some(thread),
        })
    }

    fn capture_loop(
        tx: Sender<AudioEvent>,
        capture_id: u64,
        stop: Arc<AtomicBool>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if initialize_mta().is_err() {
            return Err(io::Error::other("failed to initialize COM for WASAPI").into());
        }

        let enumerator = DeviceEnumerator::new()?;
        let device = enumerator.get_default_device(&Direction::Render)?;
        let mut audio_client = device.get_iaudioclient()?;
        let mix_format = audio_client.get_mixformat()?;
        let pcm = pcm_format_from_wave(&mix_format)?;
        let (default_period, _) = audio_client.get_device_period()?;
        audio_client.initialize_client(
            &mix_format,
            &Direction::Capture,
            &StreamMode::PollingShared {
                autoconvert: false,
                buffer_duration_hns: default_period,
            },
        )?;

        let capture_client = audio_client.get_audiocaptureclient()?;
        let buffer_frames = audio_client.get_buffer_size()? as usize;
        let frame_bytes = pcm.frame_bytes().max(1);
        let mut packet = vec![0_u8; buffer_frames.max(1) * frame_bytes];
        let mut pending = Vec::new();
        let mut resampled = Vec::new();
        let mut resampler = StereoResampler::new(pcm.sample_rate, TARGET_SAMPLE_RATE);
        let poll_every = Duration::from_millis(((default_period / 10_000).clamp(5, 20)) as u64);

        audio_client.start_stream()?;

        while !stop.load(Ordering::Relaxed) {
            drain_packets(
                &tx,
                capture_id,
                &capture_client,
                &pcm,
                &mut packet,
                &mut pending,
                &mut resampler,
                &mut resampled,
            )?;
            if stop.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(poll_every);
        }

        let _ = audio_client.stop_stream();
        Ok(())
    }

    fn drain_packets(
        tx: &Sender<AudioEvent>,
        capture_id: u64,
        capture_client: &wasapi::AudioCaptureClient,
        pcm: &PcmFormat,
        packet: &mut Vec<u8>,
        pending: &mut Vec<u8>,
        resampler: &mut StereoResampler,
        resampled: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        loop {
            let Some(frames) = capture_client.get_next_packet_size()? else {
                break;
            };
            if frames == 0 {
                break;
            }
            let needed = frames as usize * pcm.frame_bytes();
            if packet.len() < needed {
                packet.resize(needed, 0);
            }
            let (read_frames, info) = capture_client.read_from_device(&mut packet[..needed.max(1)])?;
            if read_frames == 0 {
                break;
            }
            let read_bytes = (read_frames as usize * pcm.frame_bytes()).min(packet.len());
            if info.flags.silent {
                pending.resize(pending.len() + read_bytes, 0);
            } else {
                pending.extend_from_slice(&packet[..read_bytes]);
            }
            emit_pending(tx, capture_id, pcm, pending, resampler, resampled)?;
        }
        Ok(())
    }

    fn emit_pending(
        tx: &Sender<AudioEvent>,
        capture_id: u64,
        pcm: &PcmFormat,
        pending: &mut Vec<u8>,
        resampler: &mut StereoResampler,
        resampled: &mut Vec<f32>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (stereo, consumed) = decode_pcm_to_stereo(pending, pcm);
        if consumed == 0 {
            return Ok(());
        }
        pending.drain(0..consumed);
        resampled.clear();
        resampler.push(&stereo, resampled);
        if resampled.len() < 2 {
            return Ok(());
        }
        let samples = samples_from_interleaved_stereo(resampled);
        tx.send(AudioEvent::Samples(capture_id, samples))
            .map_err(|_| io::Error::other("audio listener closed"))?;
        Ok(())
    }

    fn pcm_format_from_wave(format: &WaveFormat) -> io::Result<PcmFormat> {
        let channels = format.get_nchannels() as usize;
        if channels == 0 {
            return Err(io::Error::other("WASAPI mix format has no channels"));
        }
        let sample_rate = format.get_samplespersec();
        if sample_rate == 0 {
            return Err(io::Error::other("WASAPI mix format has no sample rate"));
        }
        let bits = format.get_bitspersample();
        let sample_type = format
            .get_subformat()
            .map_err(|error| io::Error::other(error.to_string()))?;
        let kind = match (sample_type, bits) {
            (SampleType::Float, 32) => PcmSampleKind::F32,
            (SampleType::Int, 16) => PcmSampleKind::I16,
            (SampleType::Int, 32) => PcmSampleKind::I32,
            _ => {
                return Err(io::Error::other(format!(
                    "unsupported WASAPI mix format: {sample_type} {bits}-bit"
                )))
            }
        };
        Ok(PcmFormat {
            channels,
            sample_rate,
            kind,
            bytes_per_sample: (bits as usize / 8).max(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_pcm_to_stereo, samples_from_interleaved_stereo, take_interleaved_f32_frames,
        PcmFormat, PcmSampleKind, StereoResampler, TARGET_SAMPLE_RATE,
    };

    #[test]
    fn interleaved_f32_frames_average_to_mono() {
        let mut pending = Vec::new();
        pending.extend_from_slice(&0.20_f32.to_le_bytes());
        pending.extend_from_slice(&(-0.04_f32).to_le_bytes());
        pending.extend_from_slice(&0.10_f32.to_le_bytes());
        pending.extend_from_slice(&0.02_f32.to_le_bytes());
        pending.push(0xFF);
        let samples = take_interleaved_f32_frames(&mut pending).unwrap();
        assert!((samples.mono[0] - 0.08).abs() < 1e-6);
        assert!((samples.mono[1] - 0.06).abs() < 1e-6);
        assert_eq!(pending.len(), 1);
        assert!(samples.left_level > samples.right_level);
    }

    #[test]
    fn pcm_decoder_downmixes_mono_and_integer_frames() {
        let format = PcmFormat {
            channels: 1,
            sample_rate: 48_000,
            kind: PcmSampleKind::I16,
            bytes_per_sample: 2,
        };
        let max = i16::MAX.to_le_bytes();
        let (stereo, consumed) = decode_pcm_to_stereo(&max, &format);
        assert_eq!(consumed, 2);
        assert!((stereo[0] - 1.0).abs() < 0.000_1);
        assert!((stereo[1] - 1.0).abs() < 0.000_1);
    }

    #[test]
    fn identity_resample_passes_frames_through() {
        let mut resampler = StereoResampler::new(TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE);
        let mut output = Vec::new();
        resampler.push(&[0.25, -0.5, 0.75, 1.0], &mut output);
        assert_eq!(output, vec![0.25, -0.5, 0.75, 1.0]);
        let samples = samples_from_interleaved_stereo(&output);
        assert_eq!(samples.mono.len(), 2);
    }

    #[test]
    fn upsample_repeats_held_values_between_source_frames() {
        let mut resampler = StereoResampler::new(24_000, TARGET_SAMPLE_RATE);
        let mut output = Vec::new();
        resampler.push(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], &mut output);
        assert!(output.len() >= 4);
        assert!((output[0] - 1.0).abs() < f32::EPSILON);
        assert!((output[1] + 1.0).abs() < f32::EPSILON);
    }
}
