// The LFM3-Audio model: the speech encoder, the LFM3 language model, and the depth decoder that writes speech
// frames. Ported from transformers' modeling_lfm3_audio.py, modeling_lfm3.py and modeling_lfm2.py (Apache-2.0,
// Copyright 2026 and 2025 The HuggingFace Team), and changed from them: rewritten in Rust, for inference only.

use crate::audio::MelSpectrogram;
use crate::note;
use crate::tensor::*;
use crate::tokenizer::Tokenizer;
use rayon::prelude::*;
use serde_json::Value;
use std::{path::Path, time::Instant};

// torch's nn.LayerNorm default, which the audio projector is built with.
const LAYER_NORM_EPS: f32 = 1e-5;
// The downsampler's three stride-2 convolutions: a row of the encoder per eight spectrogram frames and mels
const DOWNSAMPLING: usize = 8;

// How one attention layer rotates positions and how far back it looks.
#[derive(Clone, Copy)]
pub struct AttentionKind {
    pub theta: f32,
    pub rotary_fraction: f32,
    pub window: Option<usize>,
}

// One per layer, built once and never moved while the model runs, so the variants' sizes don't matter.
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Conv {
        in_proj: Linear,
        out_proj: Linear,
        taps: Vec<f32>,
        width: usize,
    },
    Attention {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        gate: Option<Linear>,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        inv_freq: Vec<f32>,
        window: Option<usize>,
    },
}

struct Layer {
    mixer: Mixer,
    operator_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    post_operator_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

// An LFM2 or LFM3 model without its token embeddings: layers of gated short convolutions and attention, each
// followed by a SwiGLU feed-forward, and a final RMSNorm. The speech encoder, the language model, the depth
// decoder and the codec's decoder are all one of these.
pub struct Stack {
    name: &'static str,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    pub hidden: usize,
    eps: f32,
    causal: bool,
}

// What a stack remembers between calls, per layer: a conv layer's last inputs, an attention layer's keys and values.
#[derive(Default)]
struct LayerState {
    tail: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

pub struct Cache {
    position: usize,
    layers: Vec<LayerState>,
}

impl Stack {
    // Loads the stack whose tensors start with `prefix`. Each layer's operator, head layout, norms and widths
    // come from the tensors; `kind` gives what they cannot, the rotary base, rotated fraction and window of
    // attention layer i. `causal` is false only for the depth decoder, which sees all of a frame's codebooks.
    pub fn load(
        weights: &Weights,
        prefix: &str,
        name: &'static str,
        eps: f32,
        causal: bool,
        kind: impl Fn(usize) -> Result<AttentionKind>,
    ) -> Result<Stack> {
        let final_norm = weights.vector(&format!("{prefix}.embedding_norm.weight"))?;
        let hidden = final_norm.len();
        // every width the layers below rely on, checked once here so that no forward pass meets a mismatch
        let check = |fits: bool, what: &str| if fits { Ok(()) } else { Err(format!("{prefix}: {what}")) };
        let reads = |linear: &Linear, inputs: usize| linear.weight.cols() == inputs;
        let mut layers = Vec::new();
        loop {
            let p = format!("{prefix}.layers.{}", layers.len());
            if !weights.has(&format!("{p}.ffn_norm.weight")) {
                break;
            }
            let has = |n: &str| weights.has(&format!("{p}.{n}.weight"));
            let vector = |n: &str| weights.vector(&format!("{p}.{n}.weight"));
            let optional = |n: &str| has(n).then(|| vector(n)).transpose();
            let linear = |n: &str| weights.linear(&format!("{p}.{n}"));
            let mixer = if has("conv.in_proj") {
                // the depthwise kernel is [channels, 1, width]; keep it [width][channels] so a tap is one contiguous row
                let width = match weights.shape(&format!("{p}.conv.conv.weight"))? {
                    &[channels, 1, width] if channels == hidden => width,
                    _ => 0,
                };
                let kernel = &vector("conv.conv")?;
                let (in_proj, out_proj) = (linear("conv.in_proj")?, linear("conv.out_proj")?);
                let whole = width > 0 && width.checked_mul(hidden) == Some(kernel.len());
                check(whole, "the convolution is not [hidden, 1, width]")?;
                check(in_proj.weight.rows() == 3 * hidden && reads(&in_proj, hidden), "conv.in_proj's shape")?;
                check(out_proj.weight.rows() == hidden && reads(&out_proj, hidden), "conv.out_proj's shape")?;
                let taps = (0..width).flat_map(|t| (0..hidden).map(move |c| kernel[c * width + t])).collect();
                Mixer::Conv { in_proj, out_proj, taps, width }
            } else {
                // LFM2 names the head norms q_layernorm and k_layernorm and the output out_proj; LFM3 q_norm, k_norm, o_proj
                let (q_norm, k_norm, o) = if has("self_attn.q_layernorm") {
                    ("self_attn.q_layernorm", "self_attn.k_layernorm", "self_attn.out_proj")
                } else {
                    ("self_attn.q_norm", "self_attn.k_norm", "self_attn.o_proj")
                };
                let (q_norm, k_norm) = (vector(q_norm)?, vector(k_norm)?);
                let dim = q_norm.len();
                let (q, k, v, o) =
                    (linear("self_attn.q_proj")?, linear("self_attn.k_proj")?, linear("self_attn.v_proj")?, linear(o)?);
                let gate = has("self_attn.gate_proj").then(|| linear("self_attn.gate_proj")).transpose()?;
                // heads come in equal groups per key/value head, and the values have the keys' layout
                let (heads, kv_heads) = (q.weight.rows() / dim.max(1), k.weight.rows() / dim.max(1));
                check(dim > 0 && k_norm.len() == dim && dim.is_multiple_of(2), "the head norms' widths")?;
                check(q.weight.rows() == heads * dim && k.weight.rows() == kv_heads * dim, "q or k isn't whole heads")?;
                check(kv_heads > 0 && heads.is_multiple_of(kv_heads), "the heads don't group over the key heads")?;
                check(v.weight.rows() == k.weight.rows(), "v_proj is not as wide as k_proj")?;
                check(reads(&q, hidden) && reads(&k, hidden) && reads(&v, hidden), "q, k or v doesn't read the stack")?;
                check(o.weight.rows() == hidden && reads(&o, q.weight.rows()), "the output projection's shape")?;
                check(gate.as_ref().is_none_or(|g| g.weight.rows() == heads && reads(g, hidden)), "the gate's shape")?;
                let kind = kind(layers.len())?;
                check(kind.window != Some(0), "an attention window of zero")?;
                // The first fraction·dim/2 frequencies follow the full head's progression and the rest are zero,
                // leaving their dimensions unrotated (LFM3's partial rotation). In f32, the way torch builds them.
                let rotated = (kind.rotary_fraction * dim as f32) as usize / 2;
                let inv_freq = (0..dim / 2)
                    .map(|j| if j < rotated { 1.0 / kind.theta.powf((2 * j) as f32 / dim as f32) } else { 0.0 })
                    .collect();
                Mixer::Attention {
                    heads,
                    kv_heads,
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    gate,
                    dim,
                    inv_freq,
                    window: kind.window,
                }
            };
            let layer = Layer {
                mixer,
                operator_norm: vector("operator_norm")?,
                ffn_norm: vector("ffn_norm")?,
                post_operator_norm: optional("post_operator_norm")?,
                post_ffn_norm: optional("post_ffn_norm")?,
                w1: linear("feed_forward.w1")?,
                w2: linear("feed_forward.w2")?,
                w3: linear("feed_forward.w3")?,
            };
            let norms = [Some(&layer.operator_norm), Some(&layer.ffn_norm)];
            let norms = norms.into_iter().chain([layer.post_operator_norm.as_ref(), layer.post_ffn_norm.as_ref()]);
            check(norms.flatten().all(|norm| norm.len() == hidden), "a norm is not as wide as the stack")?;
            let (w1, w2, w3) = (&layer.w1, &layer.w2, &layer.w3);
            let inner = w1.weight.rows();
            check(reads(w1, hidden) && reads(w3, hidden) && w3.weight.rows() == inner, "the feed-forward's shape")?;
            check(w2.weight.rows() == hidden && reads(w2, inner), "the feed-forward's shape")?;
            layers.push(layer);
        }
        Ok(Stack { name, hidden, layers, final_norm, eps, causal })
    }

    pub fn cache(&self) -> Cache {
        Cache { position: 0, layers: self.layers.iter().map(|_| LayerState::default()).collect() }
    }

    // Runs x (rows × hidden: the rows that follow everything `cache` has seen) through every layer and the
    // final norm, in place.
    pub fn forward(&self, x: &mut [f32], cache: &mut Cache) {
        let position = cache.position;
        for (index, (layer, state)) in self.layers.iter().zip(&mut cache.layers).enumerate() {
            layer.forward(self, x, state, position);
            dump(&format!("{}.{index}", self.name), x);
        }
        rms_norm(x, &self.final_norm, self.eps);
        dump(&format!("{}.out", self.name), x);
        cache.position += x.len() / self.hidden;
    }
}

impl Layer {
    // The operator, then the feed-forward, each reading a normed copy of the residual stream and added back to
    // it. With sandwich norms (LFM3's language model and codec) their outputs are normed again before the add.
    fn forward(&self, stack: &Stack, x: &mut [f32], state: &mut LayerState, position: usize) {
        let normed = |weight: &[f32], x: &[f32]| {
            let mut copy = x.to_vec();
            rms_norm(&mut copy, weight, stack.eps);
            copy
        };
        let mut mixed = self.mixer.forward(stack, &normed(&self.operator_norm, x), state, position);
        if let Some(weight) = &self.post_operator_norm {
            rms_norm(&mut mixed, weight, stack.eps);
        }
        x.iter_mut().zip(&mixed).for_each(|(v, m)| *v += m);
        let h = normed(&self.ffn_norm, x);
        let mut gate = self.w1.forward(&h);
        gate.iter_mut().zip(self.w3.forward(&h)).for_each(|(g, up)| *g = silu(*g) * up);
        let mut out = self.w2.forward(&gate);
        if let Some(weight) = &self.post_ffn_norm {
            rms_norm(&mut out, weight, stack.eps);
        }
        x.iter_mut().zip(&out).for_each(|(v, o)| *v += o);
    }
}

impl Mixer {
    fn forward(&self, stack: &Stack, h: &[f32], state: &mut LayerState, position: usize) -> Vec<f32> {
        let hidden = stack.hidden;
        match self {
            Mixer::Conv { in_proj, out_proj, taps, width } => {
                // in_proj gives every row its gates B and C and its input x. B·x goes through a causal depthwise
                // convolution over this row and the width-1 before it (zeros before the first), and C gates the result.
                let bcx = in_proj.forward(h);
                let rows = bcx.len() / (3 * hidden);
                let mut inputs = std::mem::take(&mut state.tail);
                inputs.resize((width - 1) * hidden, 0.0);
                for row in bcx.chunks(3 * hidden) {
                    inputs.extend(row[..hidden].iter().zip(&row[2 * hidden..]).map(|(b, x)| b * x));
                }
                let mut gated = vec![0f32; rows * hidden];
                for (r, out) in gated.chunks_mut(hidden).enumerate() {
                    for (t, tap) in taps.chunks(hidden).enumerate() {
                        let input = &inputs[(r + t) * hidden..][..hidden];
                        out.iter_mut().zip(tap).zip(input).for_each(|((o, w), v)| *o += w * v);
                    }
                    let c = &bcx[r * 3 * hidden + hidden..][..hidden];
                    out.iter_mut().zip(c).for_each(|(o, c)| *o *= c);
                }
                state.tail = inputs[rows * hidden..].to_vec();
                out_proj.forward(&gated)
            }
            Mixer::Attention { q, k, v, o, q_norm, k_norm, gate, heads, kv_heads, dim, inv_freq, window } => {
                let (mut queries, mut keys) = (q.forward(h), k.forward(h));
                // every head's query and key is RMS-normed, then rotated by its row's position
                for (x, norm, per_row) in [(&mut queries, q_norm, *heads), (&mut keys, k_norm, *kv_heads)] {
                    rms_norm(x, norm, stack.eps);
                    for (index, head) in x.chunks_mut(*dim).enumerate() {
                        rotate(head, position + index / per_row, inv_freq);
                    }
                }
                state.keys.extend(&keys);
                state.values.extend(v.forward(h));
                let mut out =
                    attention(&queries, &state.keys, &state.values, *heads, *kv_heads, *dim, stack.causal, *window);
                // LFM3 scales each head's output by a sigmoid of its own projection of the layer input
                if let Some(gate) = gate {
                    for (head, g) in out.chunks_mut(*dim).zip(gate.forward(h)) {
                        let scale = sigmoid(g);
                        head.iter_mut().for_each(|v| *v *= scale);
                    }
                }
                o.forward(&out)
            }
        }
    }
}

// The rotary position embedding of one head, pairing dimension i with i + dim/2 as rotate_half does.
fn rotate(head: &mut [f32], position: usize, inv_freq: &[f32]) {
    let half = head.len() / 2;
    for (i, &frequency) in inv_freq.iter().enumerate().filter(|(_, f)| **f != 0.0) {
        let (sin, cos) = (position as f32 * frequency).sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cos - b * sin;
        head[i + half] = b * cos + a * sin;
    }
}

// One 3×3, stride-2, zero-padded convolution of the speech downsampler, over a (mel, time) plane stored
// [mel][time][channel]. With one input channel every filter reads the same plane (the first convolution); with
// as many as it has filters, filter c reads channel c (depthwise).
struct Conv2d {
    taps: Vec<f32>,
    bias: Vec<f32>,
}

impl Conv2d {
    fn load(weights: &Weights, name: &str) -> Result<Conv2d> {
        let kernel = &weights.vector(&format!("{name}.weight"))?;
        let bias = weights.vector(&format!("{name}.bias"))?;
        let channels = bias.len();
        let whole = Some(kernel.len()) == channels.checked_mul(9);
        if weights.shape(&format!("{name}.weight"))? != [channels, 1, 3, 3] || !whole {
            return Err(format!("{name}: the kernel is not [{channels}, 1, 3, 3], one 3×3 filter per bias"));
        }
        // [channels, 1, 3, 3] becomes [tap][channel]
        let taps = (0..9).flat_map(|tap| (0..channels).map(move |c| kernel[c * 9 + tap])).collect();
        Ok(Conv2d { taps, bias })
    }

    // Returns the output plane and its (mels, frames).
    fn forward(&self, input: &[f32], mels: usize, frames: usize) -> (Vec<f32>, usize, usize) {
        let channels = self.bias.len();
        let in_channels = input.len() / (mels * frames);
        let (out_mels, out_frames) = (mels.div_ceil(2), frames.div_ceil(2));
        let mut out = vec![0f32; out_mels * out_frames * channels];
        out.par_chunks_mut(out_frames * channels).enumerate().for_each(|(m, plane)| {
            for (f, cell) in plane.chunks_mut(channels).enumerate() {
                cell.copy_from_slice(&self.bias);
                for (tap, weights) in self.taps.chunks(channels).enumerate() {
                    // the input cell this tap reads; one before the start wraps to a huge index, which is padding too
                    let (row, column) = ((2 * m + tap / 3).wrapping_sub(1), (2 * f + tap % 3).wrapping_sub(1));
                    if row >= mels || column >= frames {
                        continue;
                    }
                    let at = (row * frames + column) * in_channels;
                    if in_channels == 1 {
                        cell.iter_mut().zip(weights).for_each(|(o, w)| *o += w * input[at]);
                    } else {
                        cell.iter_mut().zip(weights).zip(&input[at..at + channels]).for_each(|((o, w), v)| *o += w * v);
                    }
                }
            }
        });
        (out, out_mels, out_frames)
    }
}

// The speech encoder: three stride-2 convolutions turn 8 spectrogram frames of 128 mels into one vector, a
// causal LFM2 with sliding-window attention reads the sequence, and a projector maps it into the language
// model's width.
struct AudioEncoder {
    conv_in: Conv2d,
    conv_mid: Conv2d,
    lin_mid: Linear,
    conv_end: Conv2d,
    lin_end: Linear,
    lin_out: Linear,
    stack: Stack,
    norm: (Vec<f32>, Vec<f32>),
    linear_1: Linear,
    linear_2: Linear,
}

impl AudioEncoder {
    // mel is frames × mels, time-major; returns one row of the language model's width per eight frames.
    fn encode(&self, mel: &[f32], mels: usize) -> Vec<f32> {
        let frames = mel.len() / mels;
        let plane: Vec<f32> = (0..mels).flat_map(|m| (0..frames).map(move |t| mel[t * mels + m])).collect();
        let silu_all = |mut x: Vec<f32>| {
            x.iter_mut().for_each(|v| *v = silu(*v));
            x
        };
        let (h, mels, frames) = self.conv_in.forward(&plane, mels, frames);
        let (h, mels, frames) = self.conv_mid.forward(&silu_all(h), mels, frames);
        let (h, mels, frames) = self.conv_end.forward(&silu_all(self.lin_mid.forward(&h)), mels, frames);
        let h = silu_all(self.lin_end.forward(&h));
        // each time step's (channel, mel) grid flattened channel-major, as torch's flatten(1, 2) of
        // [batch, channel, mel, time] lays it out
        let channels = h.len() / (mels * frames);
        let mut flat = vec![0f32; h.len()];
        for (index, &value) in h.iter().enumerate() {
            let (m, t, c) = (index / (frames * channels), index / channels % frames, index % channels);
            flat[(t * channels + c) * mels + m] = value;
        }
        let mut x = self.lin_out.forward(&flat);
        dump("down", &x);
        self.stack.forward(&mut x, &mut self.stack.cache());
        layer_norm(&mut x, &self.norm.0, &self.norm.1, LAYER_NORM_EPS);
        let mut x = self.linear_1.forward(&x);
        x.iter_mut().for_each(|v| *v = gelu(*v));
        let x = self.linear_2.forward(&x);
        dump("proj", &x);
        x
    }
}

// Writes one speech frame (a code per codebook) from the language model's hidden state.
struct DepthDecoder {
    linear: Linear,
    mask: Vec<f32>,
    stack: Stack,
    embeddings: Vec<Matrix>,
    norms: Vec<Vec<f32>>,
    schedule: Vec<usize>,
}

impl DepthDecoder {
    // MaskGIT: the codebooks are revealed in a random order over the passes of `schedule`, each pass scoring the
    // codebooks still hidden given the ones already chosen. `history` is the frames so far, for the repetition
    // penalty.
    fn sample(&self, hidden: &[f32], history: &[Vec<u32>], sampler: &mut Sampler) -> Vec<u32> {
        let slots = self.linear.forward(hidden);
        let (width, count) = (self.stack.hidden, self.embeddings.len());
        let (mut codes, mut known) = (vec![0u32; count], vec![false; count]);
        let order = sampler.order(count);
        let mut start = 0;
        for &step in &self.schedule {
            // Each codebook's slot: its share of the hidden state, plus its chosen code's embedding or, while it
            // is still hidden, the learned mask embedding. All twelve slots attend to each other.
            let mut x = slots.clone();
            for (codebook, slot) in x.chunks_mut(width).enumerate() {
                let added = if known[codebook] {
                    self.embeddings[codebook].row(codes[codebook] as usize)
                } else {
                    self.mask.clone()
                };
                slot.iter_mut().zip(added).for_each(|(s, a)| *s += a);
            }
            self.stack.forward(&mut x, &mut self.stack.cache());
            let picks = &order[start..start + step];
            for &codebook in picks {
                // the codebook's logits come from its own norm and its embedding table, tied as the output head
                let mut h = x[codebook * width..(codebook + 1) * width].to_vec();
                rms_norm(&mut h, &self.norms[codebook], self.stack.eps);
                let column = history.iter().map(|frame| frame[codebook]);
                codes[codebook] = sampler.sample(matmul(&h, self.embeddings[codebook]), column);
            }
            picks.iter().for_each(|&codebook| known[codebook] = true);
            start += step;
        }
        codes
    }
}

// How speech codes are drawn: the repetition penalty over the last `window` frames, the temperature, top-k.
// Temperature 0 takes the most likely code and reveals the codebooks in order, which makes speech
// deterministic for checking against the reference.
pub struct Sampler {
    state: u64,
    pub temperature: f32,
    pub top_k: usize,
    pub penalty: f32,
    pub window: usize,
}

impl Sampler {
    // splitmix64 (Steele, Lea and Flood, "Fast splittable pseudorandom number generators", 2014): small, fast and
    // plenty for choosing speech codes.
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // The order codebooks are revealed in: a Fisher-Yates shuffle, or index order when sampling is greedy.
    fn order(&mut self, count: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..count).collect();
        if self.temperature > 0.0 {
            for i in (1..count).rev() {
                order.swap(i, (self.next() % (i as u64 + 1)) as usize);
            }
        }
        order
    }

    fn sample(&mut self, mut logits: Vec<f32>, history: impl DoubleEndedIterator<Item = u32>) -> u32 {
        // each code the window saw is penalized once: positive scores shrink, negative ones grow
        let mut seen = vec![false; logits.len()];
        for code in history.rev().take(self.window).map(|code| code as usize) {
            if !std::mem::replace(&mut seen[code], true) {
                let score = &mut logits[code];
                *score = if *score < 0.0 { *score * self.penalty } else { *score / self.penalty };
            }
        }
        if self.temperature == 0.0 {
            return argmax(&logits);
        }
        logits.iter_mut().for_each(|score| *score /= self.temperature);
        // keep the top_k scores; ties with the k-th are kept, as transformers' TopKLogitsWarper keeps them
        if self.top_k > 0 && self.top_k < logits.len() {
            let mut sorted = logits.clone();
            let threshold = *sorted.select_nth_unstable_by(self.top_k - 1, |a, b| b.total_cmp(a)).1;
            logits.iter_mut().filter(|score| **score < threshold).for_each(|score| *score = f32::NEG_INFINITY);
        }
        softmax(&mut logits);
        let mut target = ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32;
        for (code, &probability) in logits.iter().enumerate() {
            if target < probability {
                return code as u32;
            }
            target -= probability;
        }
        argmax(&logits)
    }
}

pub fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best }).0
        as u32
}

// The special tokens the prompts and the generation loop use, looked up by their text.
pub struct Tokens {
    bos: u32,
    im_start: u32,
    im_end: u32,
    audio: u32,
    audio_out: u32,
    audio_start: u32,
    mixed_start: u32,
}

// One part of a prompt: a token, plain text to encode, or a run of rows the speech encoder fills.
pub enum Part {
    Token(u32),
    Text(String),
    Audio(usize),
}

pub struct Model {
    pub tokenizer: Tokenizer,
    pub mel: MelSpectrogram,
    encoder: AudioEncoder,
    lm: Stack,
    embeddings: Matrix,
    codebook_embedding: Matrix,
    depth: DepthDecoder,
    tokens: Tokens,
    stop: Vec<u32>,
    codebook_size: usize,
    eos_code: u32,
    generation: Value,
    // the most rows a prompt may give the language model, text tokens and speech rows together; the caller's
    // resource budget, unlimited until it sets one
    pub max_prompt: usize,
}

impl Model {
    // Loads config.json, generation_config.json, tokenizer.json and model.safetensors from the checkpoint
    // directory. Weights stay mapped; only norms, biases and the like are copied.
    /// # Safety
    /// model.safetensors must not change while the process runs, as `Weights::open` says.
    pub unsafe fn load(directory: &Path) -> Result<Model> {
        let read = |name: &str| {
            let path = directory.join(name);
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let json =
            |name: &str| -> Result<Value> { serde_json::from_str(&read(name)?).map_err(|e| format!("{name}: {e}")) };
        let (config, generation) = (json("config.json")?, json("generation_config.json")?);
        // SAFETY: passed on to this function's caller.
        let weights = unsafe { Weights::open(&directory.join("model.safetensors"))? };
        let number = |value: &Value, what: &str| value.as_f64().ok_or(format!("config.json has no {what}"));
        // a layer type's rotary settings: LFM3 keys them by layer type, LFM2 has one flat set
        let rope = |config: &Value, layer_type: &str, window: Option<usize>| -> Result<AttentionKind> {
            let parameters = &config["rope_parameters"];
            let parameters = if parameters[layer_type].is_object() { &parameters[layer_type] } else { parameters };
            Ok(AttentionKind {
                theta: number(&parameters["rope_theta"], "rope_theta")? as f32,
                rotary_fraction: parameters["partial_rotary_factor"].as_f64().unwrap_or(1.0) as f32,
                window,
            })
        };
        let eps = |config: &Value| number(&config["norm_eps"], "norm_eps").map(|eps| eps as f32);

        // the speech encoder's attention layers all look back over the encoder's sliding window
        let encoder_config = &config["encoder"];
        let encoder_window = encoder_config["sliding_window"].as_u64().map(|w| w as usize);
        let lfm2 = &encoder_config["lfm"];
        let encoder_stack = Stack::load(&weights, "encoder.lfm.lfm", "enc", eps(lfm2)?, true, |_| {
            rope(lfm2, "full_attention", encoder_window)
        })?;
        let conv = |name: &str| Conv2d::load(&weights, &format!("encoder.downsample.{name}"));
        let linear = |name: &str| weights.linear(name);
        let encoder = AudioEncoder {
            conv_in: conv("conv_in")?,
            conv_mid: conv("conv_mid")?,
            lin_mid: linear("encoder.downsample.lin_mid")?,
            conv_end: conv("conv_end")?,
            lin_end: linear("encoder.downsample.lin_end")?,
            lin_out: linear("encoder.downsample.lin_out")?,
            stack: encoder_stack,
            norm: (weights.vector("mlp.model.0.weight")?, weights.vector("mlp.model.0.bias")?),
            linear_1: linear("mlp.model.1")?,
            linear_2: linear("mlp.model.3")?,
        };

        // the language model: full attention, or sliding attention over its window, by layer type
        let lm_config = &config["lfm"];
        let lm = Stack::load(&weights, "lfm.model", "lm", eps(lm_config)?, true, |i| {
            let layer_type = lm_config["layer_types"][i].as_str().unwrap_or("");
            let window = (layer_type == "sliding_attention").then(|| lm_config["sliding_window"].as_u64()).flatten();
            rope(lm_config, layer_type, window.map(|w| w as usize))
        })?;
        // Every token id must name a row of the embedding table, which is also the output head; its shape is
        // checked against the language model before the tokenizer is sized by it.
        let embeddings = weights.matrix("lfm.model.embed_tokens.weight")?;
        if embeddings.cols() != lm.hidden {
            return Err(format!("the token embeddings are {} wide, not the model's {}", embeddings.cols(), lm.hidden));
        }
        let tokenizer = Tokenizer::load(&read("tokenizer.json")?, embeddings.rows())?;

        // The depth decoder: its codebooks reveal in maskgit_steps passes of equal size, the last taking the
        // remainder. Each codebook's table is loaded before anything is sized by the count, so a count past the
        // checkpoint's tables stops at the first one missing, and the passes are then bounded by the codebooks.
        let depth_config = &config["depthformer"];
        let codebooks = number(&config["codebooks"], "codebooks")? as usize;
        let embeddings_of = |i| weights.matrix(&format!("depth_embeddings.{i}.embedding.weight"));
        let tables = (0..codebooks).map(embeddings_of).collect::<Result<Vec<_>>>()?;
        let steps = number(&config["maskgit_steps"], "maskgit_steps")? as usize;
        if steps == 0 || steps > codebooks {
            return Err(format!("config.json: {codebooks} codebooks can't reveal in {steps} passes"));
        }
        let mut schedule = vec![codebooks / steps; steps];
        schedule[steps - 1] += codebooks % steps;
        let depth = DepthDecoder {
            linear: linear("depth_linear")?,
            mask: weights.vector("mask_embedding")?,
            stack: Stack::load(&weights, "depthformer.model", "depth", eps(depth_config)?, false, |_| {
                rope(depth_config, "full_attention", None)
            })?,
            embeddings: tables,
            norms: (0..codebooks)
                .map(|i| weights.vector(&format!("depth_embeddings.{i}.embedding_norm.weight")))
                .collect::<Result<_>>()?,
            schedule,
        };

        let id = |token: &str| tokenizer.id(token);
        let bos = number(&generation["bos_token_id"], "bos_token_id")? as usize;
        let tokens = Tokens {
            bos: u32::try_from(bos).ok().filter(|&id| (id as usize) < embeddings.rows()).ok_or("bad bos_token_id")?,
            im_start: id("<|im_start|>")?,
            im_end: id("<|im_end|>")?,
            audio: id("<|audio|>")?,
            audio_out: id("<|audio_out|>")?,
            audio_start: id("<|audio_start|>")?,
            mixed_start: id("<|mixed_start|>")?,
        };
        // a turn ends at any of the generation config's end tokens, and at <|text_end|> outside interleaved
        // speech, which this port does not generate
        let mut stop: Vec<u32> = match &generation["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(Value::as_u64).map(|id| id as u32).collect(),
            single => single.as_u64().map(|id| id as u32).into_iter().collect(),
        };
        stop.push(id("<|text_end|>")?);
        let codebook_size = number(&config["audio_vocab_size"], "audio_vocab_size")? as usize;
        let codebook_embedding = weights.matrix("codebook_embedding.weight")?;
        let mel = MelSpectrogram::new(&config["preprocessor"])?;
        // the widths where one part hands its rows to the next; the downsampler's three stride-2 convolutions
        // leave an eighth of the mels, rounded up, in each of lin_end's channels for lin_out to read
        let hidden = lm.hidden;
        let times = |a: usize, b: usize| a.checked_mul(b);
        let (conv_in, conv_mid, conv_end) = (&encoder.conv_in.bias, &encoder.conv_mid.bias, &encoder.conv_end.bias);
        let (lin_mid, lin_end) = (&encoder.lin_mid.weight, &encoder.lin_end.weight);
        let fits = [
            // the depthwise convolutions read their input's channels one each, the 1×1 ones between remap them
            conv_mid.len() == conv_in.len() && lin_mid.cols() == conv_mid.len() && lin_mid.rows() == conv_end.len(),
            lin_end.cols() == conv_end.len(),
            codebook_size > 0 && times(codebooks, codebook_size) == Some(codebook_embedding.rows()),
            codebook_embedding.cols() == hidden,
            times(lin_end.rows(), mel.mels().div_ceil(DOWNSAMPLING)) == Some(encoder.lin_out.weight.cols()),
            encoder.linear_2.weight.rows() == hidden,
            times(codebooks, depth.stack.hidden) == Some(depth.linear.weight.rows()),
            depth.linear.weight.cols() == hidden,
            depth.mask.len() == depth.stack.hidden,
            depth.embeddings.iter().all(|table| table.rows() == codebook_size && table.cols() == depth.stack.hidden),
            depth.norms.iter().all(|norm| norm.len() == depth.stack.hidden),
        ];
        if fits.contains(&false) {
            return Err("the checkpoint's parts don't fit together".into());
        }
        Ok(Model {
            mel,
            encoder,
            lm,
            embeddings,
            codebook_embedding,
            depth,
            tokens,
            stop,
            // the end-of-audio code is the last of each codebook
            eos_code: codebook_size as u32 - 1,
            codebook_size,
            tokenizer,
            generation,
            max_prompt: usize::MAX,
        })
    }

    // The checkpoint's speech sampling settings (generation_config.json), seeded.
    pub fn sampler(&self, seed: u64) -> Sampler {
        let setting = |key: &str, default: f64| self.generation[key].as_f64().unwrap_or(default);
        Sampler {
            state: seed,
            temperature: setting("audio_temperature", 1.0) as f32,
            top_k: setting("audio_top_k", 0.0) as usize,
            penalty: setting("audio_repetition_penalty", 1.0) as f32,
            // transformers' default when nothing sets it
            window: setting("audio_repetition_window", 64.0) as usize,
        }
    }

    // The transcript of 16 kHz mono `samples`, following `system` (an instruction such as the card's
    // "Perform ASR in English."), in at most `max_tokens` tokens.
    pub fn transcribe(&self, samples: &[f32], system: &str, max_tokens: usize) -> Result<String> {
        let start = Instant::now();
        let mel = self.mel.compute(samples, self.max_prompt.saturating_mul(DOWNSAMPLING))?;
        dump("mel", &mel);
        let audio = self.encoder.encode(&mel, self.mel.mels());
        note(
            &format!(
                "{:.2} s of audio encoded to {} rows",
                samples.len() as f32 / self.mel.rate as f32,
                audio.len() / self.lm.hidden
            ),
            start,
        );
        let prompt = self.prompt(system, Part::Audio(audio.len() / self.lm.hidden), false);
        let suppress = [self.tokens.audio_start, self.tokens.mixed_start];
        let reply = self.generate(self.embed(&prompt, &audio)?, false, &suppress, max_tokens, &mut self.sampler(0));
        Ok(self.tokenizer.decode(&reply.0))
    }

    // The speech frames of `text`, in the voice `system` asks for (such as the card's "Perform TTS in American
    // English using a female voice."), at most `max_frames` of them.
    pub fn speak(&self, text: &str, system: &str, max_frames: usize, sampler: &mut Sampler) -> Result<Vec<Vec<u32>>> {
        let prompt = self.prompt(system, Part::Text(text.to_string()), true);
        Ok(self.generate(self.embed(&prompt, &[])?, true, &[self.tokens.mixed_start], max_frames, sampler).1)
    }

    // The ChatML turns of the checkpoint's chat template: <|startoftext|>, then <|im_start|>role\ncontent<|im_end|>\n
    // per turn, and the assistant's header. Speech output starts the reply with <|audio_start|>.
    fn prompt(&self, system: &str, user: Part, speak: bool) -> Vec<Part> {
        let t = &self.tokens;
        let mut parts = vec![Part::Token(t.bos), Part::Token(t.im_start), Part::Text(format!("system\n{system}"))];
        parts.extend([Part::Token(t.im_end), Part::Text("\n".into()), Part::Token(t.im_start)]);
        match user {
            Part::Text(text) => parts.push(Part::Text(format!("user\n{text}"))),
            other => parts.extend([Part::Text("user\n".into()), other]),
        }
        parts.extend([
            Part::Token(t.im_end),
            Part::Text("\n".into()),
            Part::Token(t.im_start),
            Part::Text("assistant\n".into()),
        ]);
        if speak {
            parts.push(Part::Token(t.audio_start));
        }
        parts
    }

    // The prompt's input rows: token embeddings, with the speech encoder's rows in place of the audio placeholders.
    // A prompt over the budget is refused before any row is made.
    fn embed(&self, parts: &[Part], audio: &[f32]) -> Result<Vec<f32>> {
        let mut ids = Vec::new();
        for part in parts {
            match part {
                Part::Token(id) => ids.push(*id),
                Part::Text(text) => ids.extend(self.tokenizer.encode(text, self.max_prompt.saturating_sub(ids.len()))?),
                Part::Audio(rows) => ids.extend(std::iter::repeat_n(self.tokens.audio, *rows)),
            }
        }
        if ids.len() > self.max_prompt {
            return Err(format!("the prompt is {} rows, over the budget of {}", ids.len(), self.max_prompt));
        }
        dump("prompt", &ids.iter().map(|&id| id as f32).collect::<Vec<_>>());
        let mut audio_rows = audio.chunks(self.lm.hidden);
        Ok(ids
            .iter()
            .flat_map(|&id| match id == self.tokens.audio {
                true => audio_rows.next().map(<[f32]>::to_vec).unwrap_or_default(),
                false => self.embeddings.row(id as usize),
            })
            .collect())
    }

    // Generates from the prompt's rows one slot at a time, a slot being a text token or a speech frame, until a
    // stop token or `max_slots`. In text the most likely token is taken (the checkpoint's generation config is
    // greedy), never a placeholder or one of `suppress`, and <|audio_start|> switches to speech. In speech the
    // depth decoder writes each frame until an end-of-audio frame switches back. Returns the text tokens and the
    // speech frames.
    fn generate(
        &self,
        prompt: Vec<f32>,
        mut speaking: bool,
        suppress: &[u32],
        max_slots: usize,
        sampler: &mut Sampler,
    ) -> (Vec<u32>, Vec<Vec<u32>>) {
        let start = Instant::now();
        let (mut text, mut frames) = (Vec::new(), Vec::new());
        let mut cache = self.lm.cache();
        let mut x = prompt;
        self.lm.forward(&mut x, &mut cache);
        note(&format!("prompt of {} rows", x.len() / self.lm.hidden), start);
        let start = Instant::now();
        for slot in 0..max_slots {
            let hidden = &x[x.len() - self.lm.hidden..];
            let next = if speaking {
                let mut codes = self.depth.sample(hidden, &frames, sampler);
                if codes.contains(&self.eos_code) {
                    // any end-of-audio code ends the speech, and the frame goes back in as all end-of-audio codes
                    speaking = false;
                    codes.fill(self.eos_code);
                } else {
                    frames.push(codes.clone());
                }
                // a frame enters the language model as the sum of its codes' embeddings, each codebook with its
                // own rows of one shared table
                let mut sum = vec![0f32; self.lm.hidden];
                for (codebook, &code) in codes.iter().enumerate() {
                    let row = self.codebook_embedding.row(codebook * self.codebook_size + code as usize);
                    sum.iter_mut().zip(row).for_each(|(s, e)| *s += e);
                }
                sum
            } else {
                // the output head is tied to the input embeddings
                let mut logits = matmul(hidden, self.embeddings);
                dump("logits.0", &logits);
                for &token in suppress.iter().chain([&self.tokens.audio, &self.tokens.audio_out]) {
                    logits[token as usize] = f32::NEG_INFINITY;
                }
                let token = argmax(&logits);
                if self.stop.contains(&token) {
                    break;
                }
                text.push(token);
                speaking = token == self.tokens.audio_start;
                self.embeddings.row(token as usize)
            };
            if slot + 1 == max_slots {
                break;
            }
            x = next;
            self.lm.forward(&mut x, &mut cache);
        }
        note(&format!("{} text tokens and {} speech frames", text.len(), frames.len()), start);
        (text, frames)
    }
}
