#[derive(Clone, Copy, Debug)]
pub(crate) struct BandStats {
    pub(crate) average: f32,
    pub(crate) rms: f32,
    pub(crate) peak: f32,
}

pub(crate) fn sample_frequency_band(
    magnitudes: &[f32],
    lower_bin: f32,
    upper_bin: f32,
) -> BandStats {
    if magnitudes.is_empty() {
        return BandStats {
            average: 0.0,
            rms: 0.0,
            peak: 0.0,
        };
    }

    let max_bin = (magnitudes.len() - 1) as f32;
    let lower_bin = lower_bin.clamp(1.0, max_bin);
    let upper_bin = upper_bin.clamp(lower_bin, max_bin);
    let width = (upper_bin - lower_bin).max(0.001);
    let sample_count = ((width.ceil() as usize) + 2).clamp(4, 64);
    let mut total = 0.0_f32;
    let mut squared_total = 0.0_f32;
    let mut weight_total = 0.0_f32;
    let mut peak = 0.0_f32;

    for sample in 0..sample_count {
        let position = (sample as f32 + 0.5) / sample_count as f32;
        let bin = lower_bin + width * position;
        let magnitude = sample_magnitude(magnitudes, bin);
        let weight = 1.0 + position;
        total += magnitude * weight;
        squared_total += magnitude * magnitude * weight;
        weight_total += weight;
        peak = peak.max(magnitude);
    }

    BandStats {
        average: total / weight_total.max(1.0),
        rms: (squared_total / weight_total.max(1.0)).sqrt(),
        peak,
    }
}

pub(crate) fn sample_magnitude(magnitudes: &[f32], bin: f32) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }

    let max_index = magnitudes.len() - 1;
    let bin = bin.clamp(0.0, max_index as f32);
    let left = bin.floor() as usize;
    let right = bin.ceil() as usize;
    let mix = bin - left as f32;
    let left_value = magnitudes[left];
    let right_value = magnitudes[right.min(max_index)];
    left_value * (1.0 - mix) + right_value * mix
}
