// Weights and the arithmetic every part of the model shares: a safetensors file read in place, linear layers
// over bf16 weights, attention, norms and activations. Everything computes in f32; the weights stay bf16 in the
// mapped file and are widened as they are used.

use rayon::prelude::*;
use serde_json::Value;
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::Path,
    sync::Mutex,
};

pub type Result<T> = std::result::Result<T, String>;

// Weight rows each thread takes at a time. 64 rows of the widest layer (7680 inputs) widen to 2 MB of f32,
// and the language model's narrowest layer (2048 rows) still splits into 32 tasks.
const BLOCK_ROWS: usize = 64;

// With fewer input rows than this, each weight row is streamed once per input row through the fused bf16
// dot product: the work is memory-bound and the row stays in cache. With more, a block of weights is widened
// once and handed to the sgemm kernel, which reuses it across all the rows. Timing both paths on this model's
// layers put them even at about eight rows.
const GEMM_MIN_ROWS: usize = 8;

// Query rows attention scores at a time. A block of queries sees at most its own rows plus the window before
// them, or every key in full attention, so a task's scores are 64 × keys rather than queries × keys: 2 MB at
// 8192 keys, where scoring the whole prompt at once would take 256 MB per head.
const QUERY_BLOCK: usize = 64;

// The rows of a weight matrix as they lie in memory: bf16 straight from the checkpoint, or f32 for the few
// matrices built at load time (the Fourier bases).
#[derive(Clone, Copy)]
pub enum Data {
    Bf16(&'static [u16]),
    F32(&'static [f32]),
}

// A rows × cols matrix in row-major order. A linear layer's weight is [outputs, inputs], as torch stores it.
// The fields are private so that every matrix comes through `new`, which the kernels' raw strides rely on.
#[derive(Clone, Copy)]
pub struct Matrix {
    data: Data,
    rows: usize,
    cols: usize,
}

pub struct Linear {
    pub weight: Matrix,
    pub bias: Option<Vec<f32>>,
}

struct Entry {
    dtype: String,
    shape: Vec<usize>,
    bytes: &'static [u8],
}

// A safetensors file mapped into memory for the life of the process. Weights are read where they lie, so
// loading copies nothing and the kernel can drop and re-read the pages under memory pressure.
pub struct Weights {
    tensors: HashMap<String, Entry>,
}

// The device and inode of every file mapped so far, so that output can refuse to overwrite one (`is_mapped`).
static MAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

// Whether the file with this metadata is one the process has mapped: the same file under any name or link.
pub fn is_mapped(file: &std::fs::Metadata) -> bool {
    MAPPED.lock().unwrap().contains(&(file.dev(), file.ino()))
}

pub fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl Weights {
    // Maps `path` and reads its header: an 8-byte little-endian length, then JSON naming each tensor's dtype,
    // shape and byte range in the data that follows. The mapping is never undone, which is what lets every
    // model part hold `'static` slices into it.
    /// # Safety
    /// The file must not be written to or truncated for the rest of the process's life. The slices handed out
    /// read the file's pages directly, so a write by anyone changes memory Rust treats as immutable, and a
    /// truncation turns the next read of a dropped page into SIGBUS. A private, read-only mapping prevents
    /// neither; nothing short of copying 4 GB into memory of our own does, so the caller has to promise it.
    pub unsafe fn open(path: &Path) -> Result<Weights> {
        let fail = |error: String| format!("{}: {error}", path.display());
        let file = File::open(path).map_err(|e| fail(e.to_string()))?;
        let metadata = file.metadata().map_err(|e| fail(e.to_string()))?;
        let length = metadata.len() as usize;
        // SAFETY: a read-only private mapping of a file we opened; nothing writes through it.
        let address = unsafe {
            libc::mmap(std::ptr::null_mut(), length, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(), 0)
        };
        if address == libc::MAP_FAILED || length < 8 {
            return Err(fail("cannot map the file".into()));
        }
        MAPPED.lock().unwrap().push((metadata.dev(), metadata.ino()));
        // SAFETY: the mapping is `length` bytes long and lives until the process exits.
        let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(address as *const u8, length) };
        let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let (header, data) = usize::try_from(header_length)
            .ok()
            .filter(|&length| length <= bytes.len() - 8)
            .map(|length| bytes[8..].split_at(length))
            .ok_or_else(|| fail("header past the end".into()))?;
        let header: Value = serde_json::from_slice(header).map_err(|e| fail(e.to_string()))?;
        let mut tensors = HashMap::new();
        for (name, info) in header.as_object().ok_or_else(|| fail("header is not an object".into()))? {
            if name == "__metadata__" {
                continue;
            }
            let offset = |index: usize| info["data_offsets"][index].as_u64().unwrap_or(0) as usize;
            let entry = Entry {
                dtype: info["dtype"].as_str().unwrap_or("").to_string(),
                shape: info["shape"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_u64)
                    .map(|d| d as usize)
                    .collect(),
                bytes: data.get(offset(0)..offset(1)).ok_or_else(|| fail(format!("{name} past the end")))?,
            };
            tensors.insert(name.clone(), entry);
        }
        Ok(Weights { tensors })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors.get(name).ok_or_else(|| format!("the checkpoint has no tensor {name}"))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.entry(name)?.shape)
    }

    // The tensor as a matrix of shape[0] rows; trailing dimensions fold into the columns, so a 1×1
    // convolution's [out, in, 1] weight reads as [out, in]. bf16 stays in place; f32 is borrowed in place too.
    pub fn matrix(&self, name: &str) -> Result<Matrix> {
        let entry = self.entry(name)?;
        let rows = entry.shape.first().copied().unwrap_or(1);
        // A hostile header can name a shape whose product wraps around to match a short byte range, so the
        // product and its size in bytes are checked. Equal to the length of a real slice, they fit in isize.
        let count = entry.shape.iter().try_fold(1usize, |count, &d| count.checked_mul(d));
        // the header promises aligned offsets and the mapping starts on a page, but check before reinterpreting
        let aligned = |size: usize| {
            (entry.bytes.as_ptr() as usize).is_multiple_of(size)
                && count.and_then(|count| count.checked_mul(size)) == Some(entry.bytes.len())
        };
        let count = count.unwrap_or(0);
        // SAFETY: the bytes are in the 'static mapping, aligned and exactly `count` elements long (checked above).
        // A shape with a zero in it is refused by `Matrix::new`, so rows can't claim more than the bytes hold.
        let data = match entry.dtype.as_str() {
            "BF16" if aligned(2) => {
                Data::Bf16(unsafe { std::slice::from_raw_parts(entry.bytes.as_ptr().cast(), count) })
            }
            "F32" if aligned(4) => Data::F32(unsafe { std::slice::from_raw_parts(entry.bytes.as_ptr().cast(), count) }),
            dtype => return Err(format!("{name}: unsupported or misaligned {dtype} tensor")),
        };
        Matrix::new(data, rows, count / rows.max(1)).map_err(|e| format!("{name}: {e}"))
    }

    // The whole tensor widened to an owned f32 vector: norms, biases and other small parameters.
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let matrix = self.matrix(name)?;
        Ok((0..matrix.rows).flat_map(|row| matrix.row(row)).collect())
    }

    // `{name}.weight` with its `{name}.bias` when the checkpoint has one.
    pub fn linear(&self, name: &str) -> Result<Linear> {
        let bias = format!("{name}.bias");
        Ok(Linear {
            weight: self.matrix(&format!("{name}.weight"))?,
            bias: if self.has(&bias) { Some(self.vector(&bias)?) } else { None },
        })
    }
}

impl Matrix {
    // `rows` × `cols` over `data`, refused unless both are positive and their product is exactly its length.
    pub fn new(data: Data, rows: usize, cols: usize) -> Result<Matrix> {
        let length = match data {
            Data::Bf16(values) => values.len(),
            Data::F32(values) => values.len(),
        };
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(length) {
            return Err(format!("{rows} × {cols} is not a matrix of {length} values"));
        }
        Ok(Matrix { data, rows, cols })
    }

    // A matrix computed at load time. It is leaked on purpose: like the mapped weights, it lives as long as the model.
    pub fn from_f32(values: Vec<f32>, rows: usize) -> Result<Matrix> {
        let cols = values.len() / rows.max(1);
        Matrix::new(Data::F32(values.leak()), rows, cols)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    // One row widened to f32: an embedding lookup.
    pub fn row(&self, index: usize) -> Vec<f32> {
        let range = index * self.cols..(index + 1) * self.cols;
        match self.data {
            Data::Bf16(data) => data[range].iter().map(|&bits| bf16(bits)).collect(),
            Data::F32(data) => data[range].to_vec(),
        }
    }

    // Rows `rows` as one f32 block, widened into `buffer` when they are bf16.
    fn block<'a>(&self, rows: std::ops::Range<usize>, buffer: &'a mut Vec<f32>) -> &'a [f32] {
        let range = rows.start * self.cols..rows.end * self.cols;
        match self.data {
            Data::F32(data) => &data[range],
            Data::Bf16(data) => {
                buffer.clear();
                buffer.extend(data[range].iter().map(|&bits| bf16(bits)));
                buffer
            }
        }
    }

    fn dot(&self, row: usize, x: &[f32]) -> f32 {
        let range = row * self.cols..(row + 1) * self.cols;
        match self.data {
            Data::Bf16(data) => dot_bf16(x, &data[range]),
            Data::F32(data) => x.iter().zip(&data[range]).map(|(a, b)| a * b).sum(),
        }
    }
}

impl Linear {
    // y = x·Wᵀ + b for every row of x.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = matmul(x, self.weight);
        if let Some(bias) = &self.bias {
            y.chunks_mut(bias.len()).for_each(|row| row.iter_mut().zip(bias).for_each(|(v, b)| *v += b));
        }
        y
    }
}

// A raw pointer the worker threads share to write disjoint parts of one output buffer. The callers guarantee
// that no two tasks touch the same element.
#[derive(Clone, Copy)]
struct Shared(*mut f32);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}
impl Shared {
    fn at(self, offset: usize) -> *mut f32 {
        // SAFETY: callers pass offsets inside the buffer the pointer came from.
        unsafe { self.0.add(offset) }
    }
}

thread_local! {
    // each worker's buffer for widening a block of bf16 weights, kept between calls
    static WIDENED: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

// x (m × k) times the transpose of w (n × k): m × n. The output columns are split into blocks of weight rows
// that the threads take in turn, so every thread streams its own part of the weights.
pub fn matmul(x: &[f32], w: Matrix) -> Vec<f32> {
    let (n, k) = (w.rows, w.cols);
    // k and n are positive and k × n is the weights' length (Matrix::new); m × n is checked before it sizes the
    // output the kernels write through
    let m = x.len() / k;
    let mut y = vec![0f32; m.checked_mul(n).expect("matmul: the output's size overflows")];
    let out = Shared(y.as_mut_ptr());
    (0..n.div_ceil(BLOCK_ROWS)).into_par_iter().for_each(|block| {
        let rows = block * BLOCK_ROWS..n.min((block + 1) * BLOCK_ROWS);
        if m < GEMM_MIN_ROWS {
            for row in rows {
                for i in 0..m {
                    // SAFETY: element (i, row) belongs to this block alone.
                    unsafe { *out.at(i * n + row) = w.dot(row, &x[i * k..(i + 1) * k]) };
                }
            }
            return;
        }
        WIDENED.with_borrow_mut(|buffer| {
            let weights = w.block(rows.clone(), buffer);
            // SAFETY: x is m × k with rows of k, the block is rows.len() weight rows of k read as its transpose,
            // and the output columns rows.start.. belong to this block alone.
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    rows.len(),
                    1.0,
                    x.as_ptr(),
                    k as isize,
                    1,
                    weights.as_ptr(),
                    1,
                    k as isize,
                    0.0,
                    out.at(rows.start),
                    n as isize,
                    1,
                )
            }
        });
    });
    y
}

// Σ x[i]·w[i] with w in bf16, through AVX2 and FMA when the CPU has them.
pub fn dot_bf16(x: &[f32], w: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: the features this needs were just detected.
        return unsafe { dot_bf16_avx2(x, w) };
    }
    x.iter().zip(w).map(|(a, &b)| a * bf16(b)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_avx2(x: &[f32], w: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    // four independent sums of eight lanes, 32 elements a step, so the multiply-adds don't wait on each other
    let mut sums = [_mm256_setzero_ps(); 4];
    let steps = x.len().min(w.len()) / 32;
    for step in 0..steps {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let at = step * 32 + lane * 8;
            // widen eight bf16 values: zero-extend each to 32 bits and shift it into the high half
            // SAFETY: at + 8 <= steps * 32 <= the length of both slices.
            let weights = unsafe {
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128(w.as_ptr().add(at).cast())),
                    16,
                ))
            };
            *sum = _mm256_fmadd_ps(unsafe { _mm256_loadu_ps(x.as_ptr().add(at)) }, weights, *sum);
        }
    }
    let total = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
    let mut lanes = [0f32; 8];
    // SAFETY: lanes holds exactly eight floats.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), total) };
    let tail: f32 = (steps * 32..x.len()).map(|i| x[i] * bf16(w[i])).sum();
    lanes.iter().sum::<f32>() + tail
}

// Scaled dot-product attention of `rows` new queries over the cached keys and values, the new ones last.
// Query row i sits at position len - rows + i; `causal` lets it see keys up to its own position only, and
// `window` only the last `window` of those (its own included). Heads share key/value heads in equal groups.
// q is rows × heads·dim, keys and values are len × kv_heads·dim; returns rows × heads·dim. Panics when the
// slices don't have those shapes or the window is zero, since the kernels below read through raw strides.
// Each task scores one head's block of QUERY_BLOCK queries against only the keys that block can see.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv_heads: usize,
    dim: usize,
    causal: bool,
    window: Option<usize>,
) -> Vec<f32> {
    let (width, kv_width) = (heads.checked_mul(dim).unwrap(), kv_heads.checked_mul(dim).unwrap());
    assert!(dim > 0 && heads > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads), "attention: {heads} heads");
    assert!(q.len().is_multiple_of(width) && keys.len().is_multiple_of(kv_width) && keys.len() == values.len());
    let (rows, len) = (q.len() / width, keys.len() / kv_width);
    assert!(rows <= len, "attention: {rows} queries over {len} keys");
    // a zero window would see no keys, and its first key would lie past the last: refused before any pointer
    assert!(window != Some(0), "attention: a zero window");
    // the keys query row i sees, start..end; both ends only move forward as i grows
    let seen = |i: usize| {
        let position = len - rows + i;
        let end = if causal { position + 1 } else { len };
        (window.map_or(0, |window| (position + 1).saturating_sub(window)), end)
    };
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0f32; q.len()];
    let shared = Shared(out.as_mut_ptr());
    let blocks = rows.div_ceil(QUERY_BLOCK);
    (0..heads * blocks).into_par_iter().for_each(|task| {
        let (head, first) = (task / blocks, task % blocks * QUERY_BLOCK);
        let (kv, count) = (head / (heads / kv_heads), QUERY_BLOCK.min(rows - first));
        let (from, to) = (seen(first).0, seen(first + count - 1).1);
        let mut scores = vec![0f32; count * (to - from)];
        // SAFETY: the asserts above make q rows × width and keys and values len × kv_width, so the strides walk
        // this head's columns of query rows first..first + count, of key and value rows from..to <= len, and
        // write this head's columns of those query rows in out, which no other task writes.
        unsafe {
            matrixmultiply::sgemm(
                count,
                dim,
                to - from,
                scale,
                q.as_ptr().add(first * width + head * dim),
                width as isize,
                1,
                keys.as_ptr().add(from * kv_width + kv * dim),
                1,
                kv_width as isize,
                0.0,
                scores.as_mut_ptr(),
                (to - from) as isize,
                1,
            );
        }
        for (i, row) in scores.chunks_mut(to - from).enumerate() {
            let (start, end) = seen(first + i);
            softmax(&mut row[start - from..end - from]);
            row[..start - from].fill(0.0);
            row[end - from..].fill(0.0);
        }
        unsafe {
            matrixmultiply::sgemm(
                count,
                to - from,
                dim,
                1.0,
                scores.as_ptr(),
                (to - from) as isize,
                1,
                values.as_ptr().add(from * kv_width + kv * dim),
                kv_width as isize,
                1,
                0.0,
                shared.at(first * width + head * dim),
                width as isize,
                1,
            );
        }
    });
    out
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut total = 0.0;
    x.iter_mut().for_each(|v| {
        *v = (*v - max).exp();
        total += *v;
    });
    x.iter_mut().for_each(|v| *v /= total);
}

// RMSNorm of each weight-sized chunk of x: every value over the chunk's root mean square, times its weight.
// The mean is summed in f64 so a 2048-wide row loses nothing to the order of the additions.
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    for row in x.chunks_mut(weight.len()) {
        let mean = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / row.len() as f64;
        let scale = 1.0 / (mean as f32 + eps).sqrt();
        row.iter_mut().zip(weight).for_each(|(v, w)| *v = *v * scale * w);
    }
}

// LayerNorm of each row: centred, divided by the standard deviation, then scaled and shifted.
pub fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
    for row in x.chunks_mut(weight.len()) {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / row.len() as f64;
        let variance = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / row.len() as f64;
        let scale = 1.0 / (variance + eps as f64).sqrt();
        for ((v, w), b) in row.iter_mut().zip(weight).zip(bias) {
            *v = ((*v as f64 - mean) * scale) as f32 * w + b;
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

// GELU as torch computes it by default, with the exact error function: x·Φ(x).
pub fn gelu(x: f32) -> f32 {
    (0.5 * x as f64 * (1.0 + erf(x as f64 / std::f64::consts::SQRT_2))) as f32
}

// The error function by formula 7.1.26 of Abramowitz and Stegun's Handbook of Mathematical Functions (1964),
// absolute error under 1.5e-7, about one f32 rounding step at the values GELU sees.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

// When LFM_DUMP names a directory, model parts write intermediate values there as raw little-endian f32, one
// file per name, for checking against the reference implementation. The first write of a name wins, so the
// prompt's forward pass is the one kept, not the decode steps that follow it.
pub fn dump(name: &str, values: &[f32]) {
    let Some(directory) = std::env::var_os("LFM_DUMP") else {
        return;
    };
    let path = Path::new(&directory).join(format!("{name}.f32"));
    if !path.exists() {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = std::fs::write(path, bytes);
    }
}
