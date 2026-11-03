// Quantized weights: the language model's and the depth decoder's matrices, the ones every generated token or
// speech frame reads whole, in 8 or 4 bits instead of bf16. They are quantized once from the checkpoint into a
// safetensors file beside it, model.q8.safetensors or model.q4.safetensors, which is then mapped and read in
// place like the checkpoint itself. The rest (the speech encoder, norms, the codec) stays in bf16.
//
// q8 is a code per value and a scale per row: value = code · scale. q4 is a code per value, two to a byte, and
// a bf16 scale and minimum per group of GROUP values: value = code · scale + minimum, for the language model's
// layers, with the rest in q8 (`choose`). A matrix of either kind is stored under its own name as its codes (I8
// or U8), with its scales beside it as `{name}.scale`.

use crate::tensor::{Matrix, Result, Weights, bf16};
use rayon::prelude::*;
use serde_json::json;
use std::{fs::File, io::Write, os::unix::fs::MetadataExt, path::Path, time::Instant};

// The values in one 4-bit group. On the language model's matrices, groups of 32 with a searched scale and
// minimum came to 7.0–7.5% relative error, against 7.2–7.6% for the Q4_K tensors of Liquid's own Q4_K_M GGUF
// and 8.2–9.1% in groups of 64. 8 bits a row came to 0.8–1.3%.
pub const GROUP: usize = 32;

// How many spreads of a group's range the 4-bit search tries: llama.cpp's Q4_K quantizer (make_qkx2_quants,
// called with rmin -1, rdelta 0.1, nstep 20) spreads the codes over 14 to 16 steps of the range in tenths.
const SEARCH_STEPS: usize = 20;

// The quantizer's layout, part of each file's stamp: raised whenever what `write` writes changes (the groups,
// the search, the matrices chosen), so that a file an older lfm wrote is written again instead of read.
const LAYOUT: u32 = 1;

#[derive(Clone, Copy, PartialEq)]
pub enum Format {
    Q8,
    Q4,
}

impl Format {
    // The weights `--quant` names: bf16 (the checkpoint as it is, None), q8 or q4.
    pub fn parse(text: &str) -> Result<Option<Format>> {
        match text {
            "bf16" => Ok(None),
            "q8" => Ok(Some(Format::Q8)),
            "q4" => Ok(Some(Format::Q4)),
            _ => Err(format!("{text} is not bf16, q8 or q4")),
        }
    }

    fn name(self) -> &'static str {
        if self == Format::Q8 { "q8" } else { "q4" }
    }
}

// The format a checkpoint tensor is stored in under `format`, or None to leave it bf16: every matrix of the
// language model and the depth decoder. The depth embeddings' to_logits copies are never read (each table is
// tied as its own output head). Under q4 only the language model's layers take 4 bits. The token embeddings,
// its output head too, keep 8 as Q4_K_M keeps them at 6. So does the depth decoder: given bf16's codes so far,
// its greedy picks agreed with bf16's on 68% at 4 bits and 93% at 8 (the language model in q8), and it is bound
// by arithmetic rather than by the bytes it reads, so 4 bits made it no faster.
fn choose(name: &str, shape: &[usize], format: Format) -> Option<Format> {
    let read = ["lfm.model.", "depthformer.", "depth_linear."].iter().any(|part| name.starts_with(part))
        || name.starts_with("depth_embeddings.") && name.ends_with(".embedding.weight");
    let whole = matches!(shape, [rows, cols] if *rows > 0 && cols.is_multiple_of(GROUP) && *cols > 0);
    (read && whole).then_some(if name.starts_with("lfm.model.layers.") { format } else { Format::Q8 })
}

// The quantized weights for the checkpoint `source`, mapped from `path`, under `format`: the file beside it,
// opened when it was written from this same checkpoint (by its size and modification time) in this layout, and
// written first when it is missing or stale.
/// # Safety
/// As `Weights::open`: the quantized file must not change while the process runs. lfm never changes it in
/// place; it writes a new one under a temporary name and renames it over the old.
pub unsafe fn open(source: &Weights, path: &Path, format: Format) -> Result<Weights> {
    let target = path.with_extension(format!("{}.safetensors", format.name()));
    let file = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let stamp = format!("layout {LAYOUT}, {} bytes, modified {}.{:09}", file.len(), file.mtime(), file.mtime_nsec());
    if target.exists() {
        // SAFETY: passed on to this function's caller.
        let weights = unsafe { Weights::open(&target)? };
        if weights.metadata["source"] == stamp.as_str() {
            return Ok(weights);
        }
    }
    let start = Instant::now();
    eprintln!("lfm: quantizing the checkpoint to {} once, into {}", format.name(), target.display());
    // The partial file is made new, never opened through whatever already has its name (a link to the mapped
    // checkpoint would be truncated); a name that's taken moves on to the next, and only a file made here is
    // removed on failure.
    let (partial, file) = (0..)
        .map(|attempt| target.with_extension(format!("safetensors.{}.{attempt}", std::process::id())))
        .map(|path| (std::fs::OpenOptions::new().write(true).create_new(true).open(&path), path))
        .find(|(opened, _)| !matches!(opened, Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists))
        .map(|(opened, path)| opened.map(|file| (path.clone(), file)).map_err(|e| format!("{}: {e}", path.display())))
        .expect("an endless range")?;
    let written = write(source, format, &partial, file, &stamp)
        .and_then(|()| std::fs::rename(&partial, &target).map_err(|e| format!("{}: {e}", target.display())));
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written?;
    crate::note("quantized", start);
    // SAFETY: passed on to this function's caller.
    unsafe { Weights::open(&target) }
}

// Writes the chosen matrices of `source` to `path` as a safetensors file: a header laid out from the shapes
// alone, then each matrix's codes and scales as it is quantized, so only one matrix is in memory at a time.
// Every tensor's size is a multiple of 4 (rows of a multiple of GROUP), so the data stays aligned for the scales
// without gaps.
fn write(source: &Weights, format: Format, path: &Path, file: File, stamp: &str) -> Result<()> {
    let mut chosen = Vec::new();
    for name in source.names() {
        if let Some(format) = choose(name, source.shape(name)?, format) {
            chosen.push((name.clone(), format, source.matrix(name)?));
        }
    }
    chosen.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({ "source": stamp }));
    let mut offset = 0;
    let mut add = |name: String, dtype: &str, shape: &[usize], bytes: usize| {
        header.insert(name, json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes] }));
        offset += bytes;
    };
    for (name, format, matrix) in &chosen {
        let (rows, cols) = (matrix.rows(), matrix.cols());
        if *format == Format::Q8 {
            add(name.clone(), "I8", &[rows, cols], rows * cols);
            add(format!("{name}.scale"), "F32", &[rows], rows * 4);
        } else {
            add(name.clone(), "U8", &[rows, cols / 2], rows * cols / 2);
            add(format!("{name}.scale"), "BF16", &[rows, cols / GROUP, 2], rows * cols / GROUP * 4);
        }
    }
    // the header is padded with spaces, as the format allows, so that the data starts 8-aligned
    let mut text = serde_json::Value::Object(header).to_string();
    text.extend(std::iter::repeat_n(' ', (8 - text.len() % 8) % 8));
    let fail = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut out = std::io::BufWriter::new(file);
    out.write_all(&(text.len() as u64).to_le_bytes()).and_then(|()| out.write_all(text.as_bytes())).map_err(fail)?;
    for (_, format, matrix) in chosen {
        let (codes, scales) = quantize(matrix, format);
        out.write_all(&codes).and_then(|()| out.write_all(&scales)).map_err(fail)?;
    }
    out.into_inner().map_err(|e| fail(e.into_error()))?.sync_all().map_err(fail)
}

// A matrix's codes and its scales, as the bytes the file holds, quantizing the rows in parallel.
fn quantize(matrix: Matrix, format: Format) -> (Vec<u8>, Vec<u8>) {
    let (rows, cols) = (matrix.rows(), matrix.cols());
    if format == Format::Q8 {
        // each row's largest magnitude becomes ±127
        let (mut codes, mut scales) = (vec![0u8; rows * cols], vec![0f32; rows]);
        codes.par_chunks_mut(cols).zip(scales.par_iter_mut()).enumerate().for_each(|(index, (codes, scale))| {
            let row = matrix.row(index);
            let top = row.iter().fold(0f32, |top, v| top.max(v.abs()));
            let inverse = if top > 0.0 { 127.0 / top } else { 0.0 };
            codes.iter_mut().zip(&row).for_each(|(code, v)| *code = (v * inverse).round() as i8 as u8);
            *scale = top / 127.0;
        });
        return (codes, scales.iter().flat_map(|scale| scale.to_le_bytes()).collect());
    }
    let groups = cols / GROUP;
    let (mut codes, mut scales) = (vec![0u8; rows * cols / 2], vec![0u16; rows * groups * 2]);
    codes.par_chunks_mut(cols / 2).zip(scales.par_chunks_mut(groups * 2)).enumerate().for_each(
        |(index, (codes, scales))| {
            let row = matrix.row(index);
            let pieces = row.chunks(GROUP).zip(codes.chunks_mut(GROUP / 2)).zip(scales.chunks_mut(2));
            pieces.for_each(|((values, codes), scale)| scale.copy_from_slice(&quantize_group(values, codes)));
        },
    );
    (codes, scales.iter().flat_map(|scale| scale.to_le_bytes()).collect())
}

// One group's 4-bit codes, packed so that byte j holds value j's code low and value j + GROUP/2's high, and its
// bf16 scale and minimum. They are searched as llama.cpp's make_qkx2_quants does it, without importance weights:
// for each spread of the codes over the group's range, the least-squares scale and minimum for those codes,
// rounded to bf16; the codes are then taken again under the rounded pair, and the pair with the smallest squared
// error is kept.
fn quantize_group(values: &[f32], packed: &mut [u8]) -> [u16; 2] {
    let (low, high) = values.iter().fold((f32::MAX, f32::MIN), |(low, high), &v| (low.min(v), high.max(v)));
    let range = (high - low).max(f32::MIN_POSITIVE);
    let mut best = (f32::INFINITY, [0u16; 2], [0u8; GROUP]);
    for step in 0..=SEARCH_STEPS {
        let spread = (15.0 + 2.0 * step as f32 / SEARCH_STEPS as f32 - 1.0) / range;
        let trial = values.iter().map(|v| ((v - low) * spread).round().clamp(0.0, 15.0));
        // value ≈ code·scale + minimum, solved from the sums of codes, squared codes, values and their products
        let (mut sum_c, mut sum_cc, mut sum_v, mut sum_cv) = (0f32, 0f32, 0f32, 0f32);
        for (c, &v) in trial.zip(values) {
            (sum_c, sum_cc, sum_v, sum_cv) = (sum_c + c, sum_cc + c * c, sum_v + v, sum_cv + c * v);
        }
        let n = values.len() as f32;
        let determinant = n * sum_cc - sum_c * sum_c;
        let (scale, minimum) = match determinant > 0.0 {
            true => ((n * sum_cv - sum_v * sum_c) / determinant, (sum_cc * sum_v - sum_c * sum_cv) / determinant),
            false => (0.0, sum_v / n),
        };
        let pair = [to_bf16(scale), to_bf16(minimum)];
        let (scale, minimum) = (bf16(pair[0]), bf16(pair[1]));
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let (mut codes, mut error) = ([0u8; GROUP], 0f32);
        for (code, &v) in codes.iter_mut().zip(values) {
            let c = ((v - minimum) * inverse).round().clamp(0.0, 15.0);
            *code = c as u8;
            error += (c * scale + minimum - v).powi(2);
        }
        if error < best.0 {
            best = (error, pair, codes);
        }
    }
    let (codes, half) = (best.2, GROUP / 2);
    packed.iter_mut().enumerate().for_each(|(j, byte)| *byte = codes[j] | codes[j + half] << 4);
    best.1
}

// f32 to bf16, rounding to nearest with ties to even.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits() as u64;
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

// A 4-bit row's values: `codes` packed as quantize_group packs them, `scales` a (scale, minimum) pair per group.
pub fn widen_q4(codes: &[u8], scales: &[u16], out: &mut [f32]) {
    for ((out, codes), pair) in out.chunks_mut(GROUP).zip(codes.chunks(GROUP / 2)).zip(scales.chunks(2)) {
        let (scale, minimum) = (bf16(pair[0]), bf16(pair[1]));
        let (low, high) = out.split_at_mut(GROUP / 2);
        low.iter_mut().zip(codes).for_each(|(o, &c)| *o = (c & 15) as f32 * scale + minimum);
        high.iter_mut().zip(codes).for_each(|(o, &c)| *o = (c >> 4) as f32 * scale + minimum);
    }
}

// Σ x[i]·code[i] over a row of 8-bit codes, through AVX2 and FMA when the CPU has them; the row's scale is the
// caller's to apply.
pub fn dot_q8(x: &[f32], codes: &[i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: the features this needs were just detected.
        return unsafe { dot_q8_avx2(x, codes) };
    }
    x.iter().zip(codes).map(|(a, &c)| a * c as f32).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_avx2(x: &[f32], codes: &[i8]) -> f32 {
    use std::arch::x86_64::*;
    // four independent sums of eight lanes, 32 values a step, as dot_bf16 does it
    let mut sums = [_mm256_setzero_ps(); 4];
    let steps = x.len().min(codes.len()) / 32;
    for step in 0..steps {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let at = step * 32 + lane * 8;
            // SAFETY: at + 8 <= steps * 32 <= the length of both slices.
            let (codes, x) =
                unsafe { (_mm_loadl_epi64(codes.as_ptr().add(at).cast()), _mm256_loadu_ps(x.as_ptr().add(at))) };
            *sum = _mm256_fmadd_ps(x, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes)), *sum);
        }
    }
    let tail: f32 = (steps * 32..x.len().min(codes.len())).map(|i| x[i] * codes[i] as f32).sum();
    crate::tensor::total(sums) + tail
}

// Σ x[i]·value[i] over a 4-bit row (codes and scales as widen_q4 reads them), through AVX2 and FMA when the CPU
// has them. x must be whole groups, with their codes and scales; a mismatch panics before any is read.
pub fn dot_q4(x: &[f32], codes: &[u8], scales: &[u16]) -> f32 {
    let groups = x.len() / GROUP;
    assert!(x.len() == groups * GROUP && codes.len() == groups * GROUP / 2 && scales.len() == groups * 2);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: the features this needs were just detected, and the lengths were checked above.
        return unsafe { dot_q4_avx2(x, codes, scales) };
    }
    let mut values = vec![0f32; x.len()];
    widen_q4(codes, scales, &mut values);
    x.iter().zip(&values).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_avx2(x: &[f32], codes: &[u8], scales: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    let nibble = _mm_set1_epi8(0x0F);
    let mut sums = [_mm256_setzero_ps(); 4];
    for (group, x) in x.chunks_exact(GROUP).enumerate() {
        let scale = _mm256_set1_ps(bf16(scales[2 * group]));
        let minimum = _mm256_set1_ps(bf16(scales[2 * group + 1]));
        // SAFETY: the caller checked that there are GROUP/2 bytes of codes for every group of x.
        let bytes = unsafe { _mm_loadu_si128(codes.as_ptr().add(group * GROUP / 2).cast()) };
        // the low nibbles are the group's first 16 codes, the high nibbles its last 16
        let halves = [_mm_and_si128(bytes, nibble), _mm_and_si128(_mm_srli_epi16(bytes, 4), nibble)];
        for (lane, sum) in sums.iter_mut().enumerate() {
            let half = halves[lane / 2];
            let eight = if lane % 2 == 0 { half } else { _mm_srli_si128(half, 8) };
            let values = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(eight)), scale, minimum);
            // SAFETY: lane * 8 + 8 <= GROUP, the length of this chunk of x.
            *sum = _mm256_fmadd_ps(unsafe { _mm256_loadu_ps(x.as_ptr().add(lane * 8)) }, values, *sum);
        }
    }
    crate::tensor::total(sums)
}
