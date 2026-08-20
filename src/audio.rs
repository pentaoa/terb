use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WavError {
    #[error("WAV error: {0}")]
    Hound(#[from] hound::Error),
    #[error("unsupported WAV bit depth/encoding")]
    Unsupported,
    #[error("WAV contains no channels")]
    NoChannels,
}

pub struct WavAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

pub fn read_wav_mono(path: impl AsRef<Path>) -> Result<WavAudio, WavError> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err(WavError::NoChannels);
    }
    let channels = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float if spec.bits_per_sample == 32 => {
            reader.samples::<f32>().collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 16 => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i16>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Int if spec.bits_per_sample <= 32 => {
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
        _ => return Err(WavError::Unsupported),
    };
    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks_exact(channels) {
        mono.push(frame.iter().sum::<f32>() / channels as f32);
    }
    Ok(WavAudio {
        sample_rate: spec.sample_rate,
        samples: mono,
    })
}
