// Audio in and out: WAV files, and the log-mel spectrogram the speech encoder reads. The spectrogram follows
// transformers' feature_extraction_lfm3_audio.py and audio_utils.mel_filter_bank (Apache-2.0; Copyright 2026 The
// HuggingFace Team, and Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors), and is
// changed from them: rewritten in Rust.

use crate::tensor::{Matrix, Result, matmul};
use serde_json::Value;
use std::f64::consts::PI;

// The largest FFT a checkpoint may ask for, here and in the codec. Its DFT basis is 2 × (n/2 + 1) × n values,
// 67 MB of f32 at 4096; LFM3-Audio's are 400 (the features) and 1280 (the codec).
pub const MAX_FFT: usize = 4096;

// The most values the windowed frames and their spectrum may hold together, 256 MB of f32. LFM3-Audio's 300 s
// default budget is 30,000 frames of 400 samples and 402 spectrum values, 24M; this leaves a crafted hop and
// FFT under three times that.
const MAX_FEATURE_VALUES: usize = 64 << 20;

// Reads a RIFF WAV file of 16-bit or 32-bit integer PCM or 32-bit float samples and averages its channels to
// mono, as the model's preprocessor does ("channel_reduction": "mean"). Returns the samples in [-1, 1] and the
// sample rate. A data chunk whose length runs past the end (a WAV streamed through a pipe) is read to the end.
pub fn read_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (mut format, mut channels, mut rate, mut bits) = (0, 0, 0, 0);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let size = u32_at(at + 4) as usize;
        let body = &bytes[at + 8..bytes.len().min(at + 8 + size)];
        match &bytes[at..at + 4] {
            b"fmt " if body.len() >= 16 => {
                // 0xFFFE is WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of its sub-format GUID
                format = if u16_at(at + 8) == 0xFFFE && body.len() >= 26 { u16_at(at + 32) } else { u16_at(at + 8) };
                (channels, rate, bits) = (u16_at(at + 10) as usize, u32_at(at + 12), u16_at(at + 22));
            }
            b"data" => {
                let sample: fn(&[u8]) -> f32 = match (format, bits) {
                    (1, 16) => |b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                    (1, 32) => |b| i32::from_le_bytes(b.try_into().unwrap()) as f32 / 2147483648.0,
                    (3, 32) => |b| f32::from_le_bytes(b.try_into().unwrap()),
                    _ => {
                        return Err(format!("unsupported WAV format {format} with {bits}-bit samples"));
                    }
                };
                let width = bits as usize / 8 * channels.max(1);
                let mono = body
                    .chunks_exact(width)
                    .map(|frame| frame.chunks_exact(bits as usize / 8).map(sample).sum::<f32>() / channels as f32);
                return Ok((mono.collect(), rate));
            }
            _ => {}
        }
        // chunks are padded to an even length
        at += 8 + size + (size & 1);
    }
    Err("the WAV file has no data".into())
}

// 16-bit PCM mono WAV of `samples`, clipped to [-1, 1].
pub fn write_wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend(b"RIFF");
    out.extend((36 + data).to_le_bytes());
    out.extend(b"WAVEfmt ");
    // a 16-byte format chunk: PCM, one channel, the rate, bytes per second, bytes per frame, bits per sample
    for field in [16u32, 1 | (1 << 16), rate, rate * 2, 2 | (16 << 16)] {
        out.extend(field.to_le_bytes());
    }
    out.extend(b"data");
    out.extend(data.to_le_bytes());
    out.extend(samples.iter().flat_map(|v| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()));
    out
}

// The encoder's input: a power spectrogram of Hann-windowed frames (no centring), Slaney mel filters, log10
// clamped from below, then (x + offset) / divisor. Every setting comes from the checkpoint's "preprocessor".
pub struct MelSpectrogram {
    pub rate: u32,
    fft: usize,
    hop: usize,
    window: Vec<f32>,
    dft: Matrix,
    filters: Matrix,
    floor: f32,
    offset: f32,
    divisor: f32,
}

impl MelSpectrogram {
    pub fn new(config: &Value) -> Result<MelSpectrogram> {
        let number = |key: &str| config[key].as_f64().ok_or(format!("preprocessor has no {key}"));
        let (fft, hop, mels) = (number("n_fft")? as usize, number("hop_length")? as usize, number("n_mels")? as usize);
        let rate = number("sample_rate")?;
        if number("win_length")? as usize != fft || config["mel_scale"] != "slaney" || config["center"] != false {
            return Err("only the uncentred full-width window and Slaney mels of LFM3-Audio are supported".into());
        }
        // checked before anything is sized by them: no more filters than frequency bins
        if !(2..=MAX_FFT).contains(&fft) || !(1..=fft).contains(&hop) || !(1..=fft / 2 + 1).contains(&mels) {
            return Err(format!("preprocessor: {fft}-point FFT, hop {hop}, {mels} mels (FFT at most {MAX_FFT})"));
        }
        // torch.hann_window(periodic=True)
        let window = (0..fft).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / fft as f64).cos()) as f32).collect();
        // The one-sided DFT as a matrix: a row of cosines then a row of sines per frequency bin, so the frames'
        // spectrum is one matrix product.
        let bins = fft / 2 + 1;
        let basis = |trig: fn(f64) -> f64| {
            (0..bins)
                .flat_map(move |k| (0..fft).map(move |n| trig(2.0 * PI * (k * n % fft) as f64 / fft as f64) as f32))
        };
        let dft = Matrix::from_f32(basis(f64::cos).chain(basis(f64::sin)).collect(), 2 * bins)?;
        let filters = Matrix::from_f32(slaney_filters(bins, mels, number("f_min")?, number("f_max")?, rate), mels)?;
        let normalization = &config["normalization"];
        Ok(MelSpectrogram {
            rate: rate as u32,
            fft,
            hop,
            window,
            dft,
            filters,
            floor: number("logmel_floor")? as f32,
            offset: normalization["offset"].as_f64().unwrap_or(0.0) as f32,
            divisor: normalization["divisor"].as_f64().unwrap_or(1.0) as f32,
        })
    }

    pub fn mels(&self) -> usize {
        self.filters.rows()
    }

    // The features of `samples`, frames × mels, time-major. The clip is right-padded with zeros so that no
    // trailing samples are dropped: to one full window, or up to the next whole hop past the first window.
    // More than `max_frames` frames, or a working set over MAX_FEATURE_VALUES, is refused before anything is
    // allocated.
    pub fn compute(&self, samples: &[f32], max_frames: usize) -> Result<Vec<f32>> {
        let padded = if samples.len() < self.fft {
            self.fft
        } else {
            samples.len() + (self.hop - (samples.len() - self.fft) % self.hop) % self.hop
        };
        let sample = |i: usize| samples.get(i).copied().unwrap_or(0.0);
        let frames = 1 + (padded - self.fft) / self.hop;
        let values = frames.checked_mul(self.fft + self.dft.rows());
        if frames > max_frames || values.is_none_or(|values| values > MAX_FEATURE_VALUES) {
            let budgets = format!("{max_frames} frames and {MAX_FEATURE_VALUES} values");
            return Err(format!("the speech makes {frames} frames of {}, over the budget of {budgets}", self.fft));
        }
        let windowed: Vec<f32> = (0..frames)
            .flat_map(|t| self.window.iter().enumerate().map(move |(i, w)| sample(t * self.hop + i) * w))
            .collect();
        let spectrum = matmul(&windowed, self.dft);
        let bins = self.dft.rows() / 2;
        let power: Vec<f32> = spectrum
            .chunks(2 * bins)
            .flat_map(|row| (0..bins).map(move |k| row[k] * row[k] + row[bins + k] * row[bins + k]))
            .collect();
        let mut mel = matmul(&power, self.filters);
        mel.iter_mut().for_each(|v| *v = (v.log10().max(self.floor) + self.offset) / self.divisor);
        Ok(mel)
    }
}

// The Slaney-scale, Slaney-normalized triangular filters of transformers' mel_filter_bank, mels × bins: linear
// below 1 kHz and logarithmic above, each triangle scaled to unit area. Computed in f64 as numpy does.
fn slaney_filters(bins: usize, mels: usize, low: f64, high: f64, rate: f64) -> Vec<f32> {
    // the Slaney constants: 3 mels per 200 Hz below 1 kHz (15 mels), then 27 mels per factor of 6.4
    let step = 6.4f64.ln() / 27.0;
    let to_mel = |hz: f64| {
        if hz >= 1000.0 { 15.0 + (hz / 1000.0).ln() / step } else { 3.0 * hz / 200.0 }
    };
    let to_hz = |mel: f64| {
        if mel >= 15.0 { 1000.0 * ((mel - 15.0) * step).exp() } else { 200.0 * mel / 3.0 }
    };
    let (low, high) = (to_mel(low), to_mel(high));
    let edges: Vec<f64> = (0..mels + 2).map(|i| to_hz(low + (high - low) * i as f64 / (mels + 1) as f64)).collect();
    let nyquist = (rate as usize / 2) as f64;
    let mut filters = Vec::with_capacity(mels * bins);
    for m in 0..mels {
        for k in 0..bins {
            let hz = nyquist * k as f64 / (bins - 1) as f64;
            let rising = (hz - edges[m]) / (edges[m + 1] - edges[m]);
            let falling = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            filters.push((rising.min(falling).max(0.0) * 2.0 / (edges[m + 2] - edges[m])) as f32);
        }
    }
    filters
}
