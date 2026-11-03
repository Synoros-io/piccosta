// The Liquid codec's decoder: speech-frame codes back to 24 kHz audio. Ported from transformers'
// modeling_liquid_codec.py and configuration_liquid_codec.py (Apache-2.0, Copyright 2026 The HuggingFace Team),
// and changed from them: rewritten in Rust, the decoder only.

use crate::audio::MAX_FFT;
use crate::model::{AttentionKind, Stack};
use crate::tensor::{Linear, Matrix, Result, Weights, dump, matmul};
use serde_json::Value;
use std::{f64::consts::PI, path::Path};

// The rotary base and norm epsilon of the codec's LFM3 layers. configuration_liquid_codec.py writes them into
// the model it builds (lfm_config) instead of into config.json, so they are copied from there.
const ROPE_THETA: f32 = 1_000_000.0;
const NORM_EPS: f32 = 1e-5;

// The most STFT frames one codec frame may expand to. Every frame's row is repeated this many times before the
// decoder runs, so it multiplies the decoder's work and memory; LFM3-Audio's codec uses 6.
const MAX_FRAME_STACK: usize = 64;

pub struct Codec {
    pub rate: u32,
    // the codes in a frame, one per codebook
    pub codebooks: usize,
    levels: Vec<u32>,
    project_out: Linear,
    from_quantizer: Linear,
    stack: Stack,
    lin: Linear,
    inverse: Matrix,
    window: Vec<f32>,
    frame_stack: usize,
    hop: usize,
}

impl Codec {
    // Loads audio_tokenizer/ (config.json and model.safetensors) from the checkpoint directory.
    /// # Safety
    /// Its model.safetensors must not change while the process runs, as `Weights::open` says.
    pub unsafe fn load(directory: &Path) -> Result<Codec> {
        let directory = directory.join("audio_tokenizer");
        let path = directory.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let config: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let number = |key: &str| config[key].as_u64().map(|v| v as usize).ok_or(format!("codec config has no {key}"));
        // SAFETY: passed on to this function's caller.
        let weights = unsafe { Weights::open(&directory.join("model.safetensors"))? };
        // the decoder's attention layers look back over decoder_window frames
        let window = config["decoder_window"].as_u64().map(|w| w as usize);
        let kind = |_| Ok(AttentionKind { theta: ROPE_THETA, rotary_fraction: 1.0, window });
        let (fft, hop, frame_stack) = (number("stft_n_fft")?, number("stft_hop_length")?, number("stft_frame_stack")?);
        if number("stft_win_length")? != fft {
            return Err("the codec's window must span its FFT".into());
        }
        // checked before the inverse DFT below and the frames' repeats in `decode` are sized by them
        if !(2..=MAX_FFT).contains(&fft) || !(1..=fft).contains(&hop) || !(1..=MAX_FRAME_STACK).contains(&frame_stack) {
            let budgets = format!("FFT at most {MAX_FFT}, frame stack at most {MAX_FRAME_STACK}");
            return Err(format!("codec: {fft}-point FFT, hop {hop}, frame stack {frame_stack} ({budgets})"));
        }
        // The inverse real DFT as a matrix: sample n of a frame from the bins' real parts (the first half of the
        // columns) and imaginary parts (the second half). Like irfft it divides by N, doubles every bin but the
        // first and the Nyquist bin, and ignores their imaginary parts, whose sines are zero.
        let bins = fft / 2 + 1;
        let mut inverse = Vec::with_capacity(fft * 2 * bins);
        for n in 0..fft {
            let angle = |k: usize| 2.0 * PI * (k * n % fft) as f64 / fft as f64;
            let weight = |k: usize| if k == 0 || k == fft / 2 { 1.0 } else { 2.0 } / fft as f64;
            inverse.extend((0..bins).map(|k| (weight(k) * angle(k).cos()) as f32));
            inverse.extend((0..bins).map(|k| (-weight(k) * angle(k).sin()) as f32));
        }
        let levels: Vec<u32> = config["fsq_levels"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|l| l.as_u64().and_then(|l| u32::try_from(l).ok()))
            .collect();
        let codebooks = number("n_q")?;
        let (project_out, from_quantizer) =
            (weights.linear("quantizer.project_out")?, weights.linear("from_quantizer")?);
        let stack = Stack::load(&weights, "decoder.lfm", "codec", NORM_EPS, true, kind)?;
        let (lin, window) = (weights.linear("decoder.lin")?, weights.vector("decoder.istft.window")?);
        // a frame's latent is a digit per level per codebook, and each stage reads what the one before it writes
        let fits = [
            !levels.is_empty() && levels.iter().all(|&l| l >= 2),
            levels.iter().try_fold(1u32, |product, &l| product.checked_mul(l)).is_some(),
            codebooks > 0 && codebooks.checked_mul(levels.len()) == Some(project_out.weight.cols()),
            from_quantizer.weight.cols() == project_out.weight.rows() && from_quantizer.weight.rows() == stack.hidden,
            lin.weight.cols() == stack.hidden && lin.weight.rows() == 2 * bins,
            window.len() == fft,
        ];
        if fits.contains(&false) {
            return Err("the codec's parts don't fit together".into());
        }
        Ok(Codec {
            rate: number("sampling_rate")? as u32,
            codebooks,
            levels,
            project_out,
            from_quantizer,
            stack,
            lin,
            inverse: Matrix::from_f32(inverse, fft)?,
            window,
            frame_stack,
            hop,
        })
    }

    // The audio of `frames`, one code per codebook each: frame_stack·hop samples a frame (1920 at 24 kHz).
    // A frame with the wrong number of codes, or a code past the quantizer's range, is refused.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        let codes = self.levels.iter().product::<u32>();
        for (index, frame) in frames.iter().enumerate() {
            if frame.len() != self.codebooks || frame.iter().any(|&code| code >= codes) {
                return Err(format!("frame {index} is not {} codes each below {codes}", self.codebooks));
            }
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        // Finite scalar quantization: a code is a mixed-radix number over `levels`, and each digit, centred and
        // scaled to [-1, 1], is one dimension of the latent.
        let mut latents = Vec::new();
        for &code in frames.iter().flatten() {
            let mut basis = 1;
            for &level in &self.levels {
                let half = (level / 2) as f32;
                latents.push(((code / basis % level) as f32 - half) / half);
                basis *= level;
            }
        }
        let embeddings = self.from_quantizer.forward(&self.project_out.forward(&latents));
        // each codec frame becomes frame_stack STFT frames: upsampling by a whole factor repeats it
        let mut x: Vec<f32> =
            embeddings.chunks(self.stack.hidden).flat_map(|row| row.repeat(self.frame_stack)).collect();
        self.stack.forward(&mut x, &mut self.stack.cache());
        // each row's log magnitudes and phases become its complex spectrum, real parts then imaginary parts
        let out = self.lin.forward(&x);
        let bins = self.lin.weight.rows() / 2;
        let spectrum: Vec<f32> = out
            .chunks(2 * bins)
            .flat_map(|row| {
                let (magnitude, phase) = row.split_at(bins);
                let real = magnitude.iter().zip(phase).map(|(m, p)| m.exp() * p.cos());
                real.chain(magnitude.iter().zip(phase).map(|(m, p)| m.exp() * p.sin())).collect::<Vec<_>>()
            })
            .collect();
        let pieces = matmul(&spectrum, self.inverse);
        // windowed overlap-add, divided by the summed squared windows, without the first and last half overlap
        let fft = self.inverse.rows();
        let count = pieces.len() / fft;
        let (mut audio, mut envelope) =
            (vec![0f32; (count - 1) * self.hop + fft], vec![0f32; (count - 1) * self.hop + fft]);
        for (index, piece) in pieces.chunks(fft).enumerate() {
            for (i, (value, w)) in piece.iter().zip(&self.window).enumerate() {
                audio[index * self.hop + i] += value * w;
                envelope[index * self.hop + i] += w * w;
            }
        }
        let trim = (fft - self.hop) / 2;
        let audio: Vec<f32> =
            audio[trim..audio.len() - trim].iter().zip(&envelope[trim..]).map(|(a, e)| a / e).collect();
        dump("codec.audio", &audio);
        Ok(audio)
    }
}
