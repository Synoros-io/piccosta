// The language model and the depth decoder on a GPU: their arithmetic as the WGSL compute kernels in gpu.wgsl,
// run through wgpu on Vulkan, Metal or DX12. Weights, key/value caches, convolution tails and activations all
// stay on the GPU. A generation step (the next token's or frame's input, every layer, then the text head or the
// depth decoder's passes and sampling) is recorded into one command buffer and submitted once, and only the
// chosen token or the frame's codes come back: profiling the PyTorch reference found its steps' time going to
// kernel launches and host round trips, not arithmetic. The speech encoder, the prompt's rows, the sampler's
// random numbers and the codec stay on the CPU.

use super::{Input, Mixer, Model, Sampler, Stack};
use crate::quant::GROUP;
use crate::tensor::{Data, Linear, Matrix, Result, dump};
use std::sync::{Arc, Mutex, mpsc};
use std::{borrow::Cow, cell::RefCell, collections::HashMap, rc::Rc, time::Instant};

const SOURCE: &str = include_str!("gpu.wgsl");
// gpu.wgsl's sizes: the threads of a workgroup (LANES); the weight rows a workgroup of the matrix-vector kernels
// takes (for matvec, sixteen lanes a weight row: with four lanes and sixteen weight rows, which reads x less often,
// a depth decoder layer was too few workgroups to keep the GPU busy and ran at less than half the speed); the most
// rows of x matvec reads at once (ROWS; more go to the tiled kernel, 64 rows by 64 weight rows a workgroup); and what
// its fixed arrays hold: codebooks in a frame, banned tokens, codes in a codebook, values in an attention head (a
// lane per rotated pair).
const LANES: u32 = 64;
const WEIGHT_ROWS: u32 = 4;
const MATVEC_ROWS: u32 = 16;
const TILE: u32 = 64;
const MAX_CODEBOOKS: usize = 16;
const MAX_BANNED: usize = 8;
const MAX_CODES: usize = 2048;
const MAX_HEAD: usize = 128;
// The Step struct of gpu.wgsl as words: count, frames, banned, then ids, order and draws a codebook each, then ban.
const STEP_WORDS: usize = 3 + 3 * MAX_CODEBOOKS + MAX_BANNED;
// WebGPU's default cap on workgroups along one dimension; a larger dispatch folds into a second dimension.
const MAX_GROUPS: u32 = 65535;
// Workgroups the text head's argmax splits the vocabulary over before one more finds the best of theirs: 128,000
// scores come to 500 a workgroup, 8 a lane.
const ARGMAX_GROUPS: u32 = 256;
// Prompt rows a submission takes, which sizes the activations: 512 rows of the 7680-wide feed-forward are 16 MB.
const PREFILL_ROWS: usize = 512;
// gpu.wgsl's numbers for the weight formats.
const BF16: u32 = 0;
const Q8: u32 = 1;
const Q4: u32 = 2;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: HashMap<&'static str, (wgpu::ComputePipeline, wgpu::BindGroupLayout)>,
    // bound in place of an input a kernel doesn't read this time
    empty: wgpu::Buffer,
    // every weight and vector uploaded, by the addresses, lengths and shapes of the data it came from, so that
    // each is uploaded once
    uploads: RefCell<HashMap<Vec<usize>, Weight>>,
    // the first failure the GPU reported (a validation error, running out of memory, the device lost), which
    // `check` turns into an error
    failure: Arc<Mutex<Option<String>>>,
}

// A matrix on the GPU, its codes and scales as gpu.wgsl reads them; a vector is its values in `codes`.
#[derive(Clone)]
struct Weight {
    codes: wgpu::Buffer,
    scales: wgpu::Buffer,
    format: u32,
    rows: u32,
    groups: u32,
}

// One dispatch: its bindings, and for the rows a step reads, the kernel and its workgroups. The heads' parts read
// rows of their own, not the language model's, and `rows` pins them. `dump` names the buffers LFM_DUMP saves
// after it, with their row width.
#[derive(Clone)]
struct Op {
    bind: wgpu::BindGroup,
    plan: Rc<dyn Fn(u32) -> (&'static str, [u32; 3])>,
    rows: Option<usize>,
    dump: Vec<(String, wgpu::Buffer, u32)>,
}

// One generation's buffers and recorded dispatches.
pub struct Session {
    lm: Vec<Op>,
    speech: Vec<Op>,
    text: Vec<Op>,
    frame: Op,
    token: Op,
    // the language model's input rows, its rows and position this step, the host's Step, the frames so far, and
    // the codes or token picked
    x: wgpu::Buffer,
    state: wgpu::Buffer,
    step: wgpu::Buffer,
    history: wgpu::Buffer,
    picked: wgpu::Buffer,
    position: usize,
    hidden: usize,
    codebooks: usize,
    codebook_size: usize,
}

// Polls a future to its end. wgpu's native backends finish adapter and device requests on the first poll.
fn block_on<T>(future: impl Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

fn pinned(op: Op, rows: usize) -> Op {
    Op { rows: Some(rows), ..op }
}

// Workgroups for `count` of something, folded into a second dimension past the cap.
fn grid(count: u32) -> [u32; 3] {
    [count.clamp(1, MAX_GROUPS), count.div_ceil(MAX_GROUPS).max(1), 1]
}

impl Gpu {
    // The GPU wgpu picks (WGPU_BACKEND and WGPU_POWER_PREF choose, high performance by default), with every kernel
    // compiled. Validation layers stay off unless WGPU_VALIDATION or WGPU_DEBUG turn them on.
    pub fn new() -> Result<Gpu> {
        let start = Instant::now();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.flags = wgpu::InstanceFlags::empty();
        let instance = wgpu::Instance::new(descriptor.with_env());
        let preference = wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
        let options = wgpu::RequestAdapterOptions { power_preference: preference, ..Default::default() };
        let adapter = block_on(instance.request_adapter(&options)).map_err(|e| format!("no GPU: {e}"))?;
        let info = adapter.get_info();
        // the adapter's own limits: the token embeddings alone are a 262 MB storage buffer
        let descriptor = wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() };
        let (device, queue) =
            block_on(adapter.request_device(&descriptor)).map_err(|e| format!("{}: {e}", info.name))?;
        // wgpu reports errors to these callbacks, not to the call that made them; without them it panics
        let failure: Arc<Mutex<Option<String>>> = Arc::default();
        let slot = failure.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            slot.lock().unwrap().get_or_insert(error.to_string());
        }));
        let slot = failure.clone();
        device.set_device_lost_callback(move |_, message| {
            slot.lock().unwrap().get_or_insert(format!("the device was lost: {message}"));
        });
        let limits = device.limits();
        // Each section of gpu.wgsl is a kernel. Its layout is the prelude's weights at bindings 0 and 1, then its
        // own bindings in order, written only when read_write; kernels with the same layout share one, so that a
        // linear layer's bindings serve both the matrix-vector and the tiled kernel.
        let (prelude, sections) = SOURCE.split_once("\n//!").ok_or("gpu.wgsl has no kernels")?;
        let (mut layouts, mut kernels) = (HashMap::new(), HashMap::new());
        for section in sections.split("\n//!") {
            let (name, body) = section.split_once('\n').unwrap_or((section, ""));
            let name = name.trim();
            let bindings =
                body.lines().filter(|line| line.starts_with("@group")).map(|line| line.contains("read_write"));
            let writes: Vec<bool> = [false, false].into_iter().chain(bindings).collect();
            if writes.len() as u32 > limits.max_storage_buffers_per_shader_stage {
                let most = limits.max_storage_buffers_per_shader_stage;
                return Err(format!(
                    "{}: the {name} kernel binds {} storage buffers, more than its max_storage_buffers_per_shader_stage of {most}",
                    info.name,
                    writes.len()
                ));
            }
            let layout: &wgpu::BindGroupLayout = layouts.entry(writes.clone()).or_insert_with(|| {
                let entries: Vec<_> = (writes.iter().enumerate())
                    .map(|(binding, &write)| wgpu::BindGroupLayoutEntry {
                        binding: binding as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: !write },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
            });
            let source = wgpu::ShaderSource::Wgsl(format!("{prelude}\n{body}").into());
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            kernels.insert(name, (pipeline, layout.clone()));
        }
        // wgpu zeroes new buffers
        let usage = wgpu::BufferUsages::STORAGE;
        let empty =
            device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage, mapped_at_creation: false });
        let gpu = Gpu { device, queue, kernels, empty, uploads: RefCell::default(), failure };
        gpu.check()?;
        crate::note(&format!("{} through {:?}", info.name, info.backend), start);
        Ok(gpu)
    }

    // The failure the GPU reported, if any, as an error; once there is one, every later check fails with it.
    fn check(&self) -> Result<()> {
        self.failure.lock().unwrap().as_ref().map_or(Ok(()), |failure| Err(format!("the GPU: {failure}")))
    }

    // `words` 4-byte values (None when counting them overflowed), if a buffer of them is one this GPU allocates and
    // binds, and gpu.wgsl's 32-bit indices reach all of it; else an error naming the buffer and the limit. Every
    // buffer not of a fixed size is checked here before it's built or its contents gathered.
    fn fits(&self, name: &str, words: Option<usize>) -> Result<usize> {
        let words = words.filter(|&words| words <= u32::MAX as usize);
        let words =
            words.ok_or_else(|| format!("the {name}: more values than the GPU kernels' 32-bit indices reach"))?;
        let (bytes, limits) = (words.max(4) as u64 * 4, self.device.limits());
        for (limit, most) in [
            ("max_buffer_size", limits.max_buffer_size),
            ("max_storage_buffer_binding_size", limits.max_storage_buffer_binding_size),
        ] {
            if bytes > most {
                return Err(format!("the {name}: {bytes} bytes, more than this GPU's {limit} of {most}"));
            }
        }
        Ok(words)
    }

    // A buffer holding `bytes`, which kernels read and write and the host writes and reads back.
    fn buffer(&self, bytes: &[u8]) -> wgpu::Buffer {
        let buffer = self.zeros(bytes.len().div_ceil(4));
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    // A buffer of `values` zeroed floats (or words), `fits` checked first.
    fn zeros(&self, values: usize) -> wgpu::Buffer {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let size = (values.max(4) * 4) as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false })
    }

    // The rows of `matrices`, one after another, as one matrix on the GPU: uploaded the first time, then reused.
    fn weight(&self, matrices: &[Matrix]) -> Result<Weight> {
        let parts = matrices.iter().map(|matrix| -> Result<(u32, &'static [u8], &'static [u8])> {
            Ok(match matrix.data() {
                Data::Bf16(values) => (BF16, bytemuck::cast_slice(values), &[]),
                Data::Q8(codes, scales) => (Q8, bytemuck::cast_slice(codes), bytemuck::cast_slice(scales)),
                Data::Q4(codes, scales) => (Q4, codes, bytemuck::cast_slice(scales)),
                Data::F32(_) => return Err("an f32 matrix, which the GPU kernels don't read".into()),
            })
        });
        let parts = parts.collect::<Result<Vec<_>>>()?;
        // two matrices can start at the same bytes and differ in shape or extent, so the key is all of each
        let key: Vec<usize> = (matrices.iter().zip(&parts))
            .flat_map(|(matrix, &(format, codes, scales))| {
                let (codes, scales) =
                    ((codes.as_ptr() as usize, codes.len()), (scales.as_ptr() as usize, scales.len()));
                [format as usize, matrix.rows(), matrix.cols(), codes.0, codes.1, scales.0, scales.1]
            })
            .collect();
        if let Some(weight) = self.uploads.borrow().get(&key) {
            return Ok(weight.clone());
        }
        let (format, cols) = (parts[0].0, matrices[0].cols());
        let alike = matrices.iter().zip(&parts).all(|(matrix, part)| matrix.cols() == cols && part.0 == format);
        if !alike || !cols.is_multiple_of(GROUP) {
            return Err(format!("{cols}-column weights the GPU kernels can't read in groups of {GROUP}"));
        }
        let rows = matrices.iter().map(Matrix::rows).sum::<usize>();
        let joined = |name: &str, pieces: Vec<&'static [u8]>| -> Result<Cow<'static, [u8]>> {
            let bytes = pieces.iter().try_fold(0usize, |sum, piece| sum.checked_add(piece.len()));
            self.fits(&format!("{rows}×{cols} matrix's {name}"), bytes.map(|bytes| bytes.div_ceil(4)))?;
            Ok(match pieces.len() {
                1 => Cow::Borrowed(pieces[0]),
                _ => Cow::Owned(pieces.concat()),
            })
        };
        let codes = joined("codes", parts.iter().map(|part| part.1).collect())?;
        let scales = joined("scales", parts.iter().map(|part| part.2).collect())?;
        let rows = rows as u32;
        let scales = if scales.is_empty() { self.empty.clone() } else { self.buffer(&scales) };
        let weight = Weight { codes: self.buffer(&codes), scales, format, rows, groups: (cols / GROUP) as u32 };
        self.uploads.borrow_mut().insert(key, weight.clone());
        Ok(weight)
    }

    // `parts`, one after another, as one f32 vector on the GPU, uploaded once. Only for vectors the model owns:
    // their addresses and lengths are the key, after a word no matrix's key starts with.
    fn vector(&self, parts: &[&Vec<f32>]) -> Result<wgpu::Buffer> {
        let key: Vec<usize> = [usize::MAX]
            .into_iter()
            .chain(parts.iter().flat_map(|part| [part.as_ptr() as usize, part.len()]))
            .collect();
        if let Some(weight) = self.uploads.borrow().get(&key) {
            return Ok(weight.codes.clone());
        }
        self.fits("vector", parts.iter().try_fold(0usize, |sum, part| sum.checked_add(part.len())))?;
        let values: Vec<f32> = parts.iter().flat_map(|part| part.iter()).copied().collect();
        let codes = self.buffer(bytemuck::cast_slice(&values));
        let weight = Weight { codes: codes.clone(), scales: self.empty.clone(), format: BF16, rows: 0, groups: 0 };
        self.uploads.borrow_mut().insert(key, weight);
        Ok(codes)
    }

    // A dispatch of `kernel` with `weight` at bindings 0 and 1, `args` (its Args struct as words) at 2 and
    // `buffers` after; `size` gives its workgroups for the rows of a step.
    fn op(
        &self,
        kernel: &'static str,
        weight: Option<&Weight>,
        args: &[u32],
        buffers: &[&wgpu::Buffer],
        size: impl Fn(u32) -> [u32; 3] + 'static,
    ) -> Op {
        self.bound(kernel, weight, args, buffers, Rc::new(move |rows| (kernel, size(rows))))
    }

    fn bound(
        &self,
        kernel: &'static str,
        weight: Option<&Weight>,
        args: &[u32],
        buffers: &[&wgpu::Buffer],
        plan: Rc<dyn Fn(u32) -> (&'static str, [u32; 3])>,
    ) -> Op {
        let args = self.buffer(bytemuck::cast_slice(args));
        let (codes, scales) = weight.map_or((&self.empty, &self.empty), |weight| (&weight.codes, &weight.scales));
        let all = [codes, scales, &args].into_iter().chain(buffers.iter().copied());
        let entries: Vec<_> = all
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let layout = &self.kernels[kernel].1;
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout,
            entries: &entries,
        });
        Op { bind, plan, rows: None, dump: Vec::new() }
    }

    // y = x·Wᵀ, W the rows of `linears` one after another, so that layers reading the same x go in one dispatch
    // (or written into a key or value cache at the stack's position, when `cached`): the matrix-vector kernels for
    // a few rows, the tiled one for many. The kernels add no bias; the one bias the GPU reads, the depth
    // decoder's, goes in with its slots.
    fn linear(
        &self,
        linears: &[&Linear],
        state: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        cached: bool,
    ) -> Result<Op> {
        if linears.iter().any(|linear| linear.bias.is_some()) {
            return Err("a linear layer with a bias, which the GPU kernels don't add".into());
        }
        let weight = self.weight(&linears.iter().map(|linear| linear.weight).collect::<Vec<_>>())?;
        let outputs = weight.rows;
        let args = [weight.format, weight.groups, outputs, cached as u32];
        Ok(self.bound(
            "matvec",
            Some(&weight),
            &args,
            &[state, x, y],
            Rc::new(move |rows| match rows {
                1 => ("matvec1", grid(outputs.div_ceil(WEIGHT_ROWS))),
                _ if rows <= MATVEC_ROWS => ("matvec", grid(outputs.div_ceil(WEIGHT_ROWS))),
                _ => ("gemm", [outputs.div_ceil(TILE), rows.div_ceil(TILE), 1]),
            }),
        ))
    }

    // The dispatches of one forward pass of `stack` over up to `rows` rows, with `capacity` rows of key/value
    // cache, reading its rows and position from `state`. Returns them with the residual stream x, which takes the
    // input rows, and the normed output h.
    fn stack(
        &self,
        stack: &Stack,
        rows: usize,
        capacity: usize,
        state: &wgpu::Buffer,
    ) -> Result<(Vec<Op>, wgpu::Buffer, wgpu::Buffer)> {
        let (hidden, eps) = (stack.hidden as u32, stack.eps.to_bits());
        // the scratch buffers as wide as the widest layer needs: B, C and x, the queries, keys, values and gates, or
        // the feed-forward's gate and up side by side; the feed-forward's inner width; and the heads
        let (mut wide, mut inner, mut heads) = (stack.hidden, stack.hidden, stack.hidden);
        for layer in &stack.layers {
            inner = inner.max(layer.w1.weight.rows());
            wide = wide.max(2 * layer.w1.weight.rows());
            match &layer.mixer {
                Mixer::Conv { .. } => wide = wide.max(3 * stack.hidden),
                Mixer::Attention { q, k, gate, .. } => {
                    heads = heads.max(q.weight.rows());
                    wide =
                        wide.max(q.weight.rows() + 2 * k.weight.rows() + gate.as_ref().map_or(0, |g| g.weight.rows()));
                }
            }
        }
        // the widest of them (wide starts at hidden) fits only if they all do
        self.fits(&format!("{} activations", stack.name), rows.checked_mul(wide.max(inner).max(heads)))?;
        let [x, h, m] = [(); 3].map(|_| self.zeros(rows * stack.hidden));
        let [both, gated, q, a] = [wide, inner, heads, heads].map(|width| self.zeros(rows * width));
        let empty = &self.empty;
        let residual = |add: bool, post: Option<&Vec<f32>>, next: &Vec<f32>| -> Result<Op> {
            let post = post.map(|weight| self.vector(&[weight])).transpose()?;
            let args = [hidden, eps, add as u32, post.is_some() as u32, 0];
            let m = if add { &m } else { empty };
            let buffers = [state, m, post.as_ref().unwrap_or(empty), &self.vector(&[next])?, &x, &h];
            Ok(self.op("residual", None, &args, &buffers, |rows| [rows, 1, 1]))
        };
        let first = stack.layers.first().map_or(&stack.final_norm, |layer| &layer.operator_norm);
        let mut ops = vec![residual(false, None, first)?];
        for (index, layer) in stack.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Conv { in_proj, out_proj, taps, width } => {
                    let tail = width.saturating_sub(1).checked_mul(stack.hidden);
                    let tail = self.zeros(self.fits(&format!("{} convolution tail", stack.name), tail)?);
                    ops.push(self.linear(&[in_proj], state, &h, &both, false)?);
                    let buffers = [state, &both, &self.vector(&[taps])?, &tail, &a];
                    ops.push(
                        self.op("conv", None, &[*width as u32, hidden], &buffers, move |_| {
                            [hidden.div_ceil(LANES), 1, 1]
                        }),
                    );
                    ops.push(self.linear(&[out_proj], state, &a, &m, false)?);
                }
                Mixer::Attention {
                    q: q_proj,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    gate: head_gate,
                    heads,
                    kv_heads,
                    dim,
                    inv_freq,
                    window,
                } => {
                    if *dim > MAX_HEAD || !dim.is_multiple_of(4) {
                        return Err(format!("{}: heads of {dim}, which the GPU kernels don't take", stack.name));
                    }
                    let cache = capacity.checked_mul(*kv_heads).and_then(|values| values.checked_mul(*dim));
                    let cache = self.fits(&format!("{} key/value cache", stack.name), cache)?;
                    let [keys, values] = [(); 2].map(|_| self.zeros(cache));
                    // the queries, keys, values and gates side by side in each row of `both`
                    let linears: Vec<&Linear> = [q_proj, k, v].into_iter().chain(head_gate).collect();
                    let width = linears.iter().map(|linear| linear.weight.rows()).sum::<usize>() as u32;
                    ops.push(self.linear(&linears, state, &h, &both, false)?);
                    // cos and sin of position·frequency at every position, as model.rs's rotate computes them
                    let turns = capacity.checked_mul(inv_freq.len()).and_then(|pairs| pairs.checked_mul(2));
                    self.fits(&format!("{} rotary table", stack.name), turns)?;
                    let turns = (0..capacity).flat_map(|position| {
                        inv_freq.iter().flat_map(move |frequency| {
                            let (sin, cos) = (position as f32 * frequency).sin_cos();
                            [cos, sin]
                        })
                    });
                    let rotary = self.buffer(bytemuck::cast_slice(&turns.collect::<Vec<f32>>()));
                    let (heads, kv_heads) = (*heads as u32, *kv_heads as u32);
                    let args = [heads, kv_heads, *dim as u32, eps, width];
                    let norms = [&self.vector(&[q_norm])?, &self.vector(&[k_norm])?];
                    let buffers = [state, norms[0], norms[1], &rotary, &both, &q, &keys, &values];
                    ops.push(self.op("rope", None, &args, &buffers, move |rows| [rows, heads + kv_heads, 1]));
                    // a window as long as the cache sees all of it, as the CPU's does
                    let window = window.map_or(Ok(0), |window| u32::try_from(window.min(capacity)));
                    let window =
                        window.map_err(|_| format!("{}: a window past the GPU kernels' 32 bits", stack.name))?;
                    let (is_gated, gates) = (head_gate.is_some() as u32, (heads + 2 * kv_heads) * *dim as u32);
                    let args = [heads, kv_heads, *dim as u32, stack.causal as u32, window, is_gated, gates, width];
                    let buffers = [state, &q, &keys, &values, &both, &a];
                    ops.push(self.op("attention", None, &args, &buffers, move |rows| [rows, heads, 1]));
                    ops.push(self.linear(&[o], state, &a, &m, false)?);
                }
            }
            ops.push(residual(true, layer.post_operator_norm.as_ref(), &layer.ffn_norm)?);
            ops.push(self.linear(&[&layer.w1, &layer.w3], state, &h, &both, false)?);
            let inner = layer.w1.weight.rows() as u32;
            ops.push(self.op("swiglu", None, &[inner], &[state, &both, &gated], move |rows| {
                grid((rows * inner).div_ceil(LANES))
            }));
            ops.push(self.linear(&[&layer.w2], state, &gated, &m, false)?);
            let next = stack.layers.get(index + 1).map_or(&stack.final_norm, |next| &next.operator_norm);
            let mut op = residual(true, layer.post_ffn_norm.as_ref(), next)?;
            op.dump.push((format!("{}.{index}", stack.name), x.clone(), hidden));
            ops.push(op);
        }
        ops.last_mut().expect("the first norm at least").dump.push((format!("{}.out", stack.name), h.clone(), hidden));
        Ok((ops, x, h))
    }

    // A generation's buffers and dispatches: prompts of `rows` rows (`capacity` rows of cache with the slots
    // that follow), `frames` speech frames at most, and `sampler`'s settings.
    pub fn session(
        &self,
        model: &Model,
        rows: usize,
        capacity: usize,
        frames: usize,
        sampler: &Sampler,
    ) -> Result<Session> {
        let (lm, depth) = (&model.lm, &model.depth);
        let (codebooks, size, width) = (depth.embeddings.len(), model.codebook_size, depth.stack.hidden);
        if codebooks > MAX_CODEBOOKS || size > MAX_CODES {
            return Err(format!("{codebooks} codebooks of {size} codes, more than the GPU kernels hold"));
        }
        let state = self.zeros(2);
        let (lm_ops, x, h) = self.stack(lm, rows.clamp(1, PREFILL_ROWS), capacity, &state)?;
        let (one, step, picked) =
            (self.buffer(bytemuck::cast_slice(&[1u32, 0])), self.zeros(STEP_WORDS), self.zeros(MAX_CODEBOOKS));
        let hidden = self.zeros(lm.hidden);
        let last = self.op("last", None, &[lm.hidden as u32], &[&state, &h, &hidden], {
            let groups = (lm.hidden as u32).div_ceil(LANES);
            move |_| [groups, 1, 1]
        });
        let input = |table: Matrix| -> Result<Op> {
            let table = self.weight(&[table])?;
            let groups = (table.groups * 4).div_ceil(LANES);
            Ok(self.op("embed", Some(&table), &[table.format, table.groups], &[&step, &x], move |_| [groups, 1, 1]))
        };
        // the text head: the token embeddings, tied as the output head, then the likeliest token not banned
        let embeddings = self.weight(&[model.embeddings])?;
        let (logits, candidates) = (self.zeros(embeddings.rows as usize), self.zeros(2 * ARGMAX_GROUPS as usize));
        let argmax = |args: [u32; 2], groups: u32| {
            self.op("argmax", None, &args, &[&step, &logits, &candidates, &picked], move |_| [groups, 1, 1])
        };
        let text = vec![
            last.clone(),
            pinned(self.linear(&[&Linear { weight: model.embeddings, bias: None }], &one, &hidden, &logits, false)?, 1),
            argmax([embeddings.rows, 0], ARGMAX_GROUPS),
            argmax([ARGMAX_GROUPS, 1], 1),
        ];
        // The speech head: depth_linear makes every codebook's slot, then each pass runs the depth decoder over
        // all of them, scores every codebook against its own table, and samples the pass's codebooks.
        let depth_state = self.buffer(bytemuck::cast_slice(&[codebooks as u32, 0]));
        let (depth_ops, depth_x, depth_h) = self.stack(&depth.stack, codebooks, codebooks, &depth_state)?;
        let tables = self.weight(&depth.embeddings)?;
        self.fits("depth decoder's slots", codebooks.checked_mul(width.max(size)))?;
        let [slots, normed, scores] = [width, width, size].map(|values| self.zeros(codebooks * values));
        let history = self.zeros(self.fits("frame history", frames.checked_mul(codebooks))?);
        let bias = match &depth.linear.bias {
            Some(bias) => self.vector(&[bias])?,
            None => self.zeros(codebooks * width),
        };
        let norms = self.vector(&depth.norms.iter().collect::<Vec<_>>())?;
        let depth_linear =
            self.linear(&[&Linear { weight: depth.linear.weight, bias: None }], &one, &hidden, &slots, false)?;
        let mut speech = vec![last, pinned(depth_linear, 1)];
        let (codebooks, size) = (codebooks as u32, size as u32);
        let settings =
            [sampler.window.min(u32::MAX as usize) as u32, sampler.temperature.to_bits(), sampler.penalty.to_bits()];
        let top_k = sampler.top_k.min(u32::MAX as usize) as u32;
        let mut known = 0;
        for &pass in &depth.schedule {
            let pass = pass as u32;
            let args = [tables.format, tables.groups, size, known];
            let buffers = [&step, &slots, &bias, &self.vector(&[&depth.mask])?, &picked, &depth_x];
            let chunks = (tables.groups * 4).div_ceil(LANES);
            speech.push(self.op("slots", Some(&tables), &args, &buffers, move |_| [chunks, codebooks, 1]));
            speech.extend(depth_ops.iter().map(|op| pinned(op.clone(), codebooks as usize)));
            let args = [width as u32, depth.stack.eps.to_bits(), 0, 0, 1];
            let buffers = [&depth_state, &self.empty, &self.empty, &norms, &depth_h, &normed];
            speech.push(self.op("residual", None, &args, &buffers, move |_| [codebooks, 1, 1]));
            let args = [tables.format, tables.groups, size];
            speech.push(self.op("heads", Some(&tables), &args, &[&normed, &scores], move |_| [size, codebooks, 1]));
            let args = [size, codebooks, known, settings[0], settings[1], settings[2], top_k];
            speech.push(self.op("sample", None, &args, &[&step, &history, &scores, &picked], move |_| [pass, 1, 1]));
            known += pass;
        }
        let (frame, token) = (input(model.codebook_embedding)?, input(model.embeddings)?);
        // the uploads' writes go out now, not with the first step, and any allocation that failed shows here
        self.queue.submit([]);
        self.check()?;
        Ok(Session {
            lm: lm_ops,
            speech,
            text,
            frame,
            token,
            x,
            state,
            step,
            history,
            picked,
            position: 0,
            hidden: lm.hidden,
            codebooks: codebooks as usize,
            codebook_size: size as usize,
        })
    }

    // Records `parts`, each a run of ops over a number of rows, into one command buffer and submits it, then reads
    // the first `bytes` of `read` back as words once the GPU is done (nothing when `bytes` is 0).
    fn run(&self, parts: &[(&[Op], usize)], read: &wgpu::Buffer, bytes: usize) -> Result<Vec<u32>> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let most = self.device.limits().max_compute_workgroups_per_dimension;
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (ops, rows) in parts {
                for op in *ops {
                    let (kernel, [x, y, z]) = (op.plan)(op.rows.unwrap_or(*rows) as u32);
                    if x.max(y).max(z) > most {
                        return Err(format!(
                            "{kernel}: {x}×{y}×{z} workgroups, more than this GPU's max_compute_workgroups_per_dimension of {most}"
                        ));
                    }
                    pass.set_pipeline(&self.kernels[kernel].0);
                    pass.set_bind_group(0, &op.bind, &[]);
                    pass.dispatch_workgroups(x, y, z);
                }
            }
        }
        if bytes == 0 {
            self.queue.submit([encoder.finish()]);
            return self.check().map(|_| Vec::new());
        }
        let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let size = bytes as u64;
        let staging =
            self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
        encoder.copy_buffer_to_buffer(read, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let (sender, mapped) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |result| drop(sender.send(result)));
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("the GPU: {e}"))?;
        self.check()?;
        let mapped = mapped.try_recv().map_err(|_| "the GPU's output was never mapped")?;
        mapped.map_err(|e| format!("the GPU's output: {e}"))?;
        let view = staging.get_mapped_range(..).map_err(|e| format!("the GPU's output: {e}"))?;
        Ok(view.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect())
    }

    // `run`, except that with LFM_DUMP set each op is its own submission and the layers' outputs are dumped as
    // they are written.
    fn submit(&self, parts: &[(&[Op], usize)], read: &wgpu::Buffer, bytes: usize) -> Result<Vec<u32>> {
        if std::env::var_os("LFM_DUMP").is_none() {
            return self.run(parts, read, bytes);
        }
        for (ops, rows) in parts {
            for op in *ops {
                self.run(&[(std::slice::from_ref(op), *rows)], read, 0)?;
                for (name, buffer, width) in &op.dump {
                    let values = self.run(&[], buffer, op.rows.unwrap_or(*rows) * *width as usize * 4)?;
                    dump(name, &values.into_iter().map(f32::from_bits).collect::<Vec<_>>());
                }
            }
        }
        self.run(&[], read, bytes)
    }

    // Reads `input` through the language model and returns the next slot, as model.rs's step does on the CPU: a
    // speech frame's codes while `speaking`, else the likeliest text token not `banned`.
    pub fn step(
        &self,
        session: &mut Session,
        input: &Input,
        speaking: bool,
        frames: &[Vec<u32>],
        sampler: &mut Sampler,
        banned: &[u32],
    ) -> Result<Vec<u32>> {
        let s = session;
        // gpu.wgsl's Step: count, frames, banned, then ids, order and draws a codebook each, then ban
        let mut words = [0u32; STEP_WORDS];
        let (ids, order, draws, ban) = (3, 3 + MAX_CODEBOOKS, 3 + 2 * MAX_CODEBOOKS, 3 + 3 * MAX_CODEBOOKS);
        let (first, rows): (&[Op], usize) = match input {
            Input::Rows(values) => {
                // every chunk of the prompt but the last is a submission of its own
                let chunks: Vec<&[f32]> = values.chunks(PREFILL_ROWS * s.hidden).collect();
                for chunk in &chunks[..chunks.len() - 1] {
                    let rows = chunk.len() / s.hidden;
                    self.queue.write_buffer(&s.x, 0, bytemuck::cast_slice(chunk));
                    self.queue.write_buffer(&s.state, 0, bytemuck::cast_slice(&[rows as u32, s.position as u32]));
                    self.submit(&[(&s.lm[..], rows)], &s.picked, 0)?;
                    s.position += rows;
                }
                let last = chunks[chunks.len() - 1];
                self.queue.write_buffer(&s.x, 0, bytemuck::cast_slice(last));
                (&[], last.len() / s.hidden)
            }
            // a frame's codes, each in its own codebook's rows of the shared table
            Input::Frame(codes) => {
                words[0] = codes.len() as u32;
                for (codebook, &code) in codes.iter().enumerate() {
                    words[ids + codebook] = (codebook * s.codebook_size) as u32 + code;
                }
                (std::slice::from_ref(&s.frame), 1)
            }
            Input::Token(token) => {
                (words[0], words[ids]) = (1, *token);
                (std::slice::from_ref(&s.token), 1)
            }
        };
        words[1] = frames.len() as u32;
        if speaking {
            // the order and the draws come from the sampler, in the order the CPU's depth decoder takes them
            for (i, codebook) in sampler.order(s.codebooks).into_iter().enumerate() {
                words[order + i] = codebook as u32;
            }
            if sampler.temperature > 0.0 {
                (0..s.codebooks).for_each(|i| words[draws + i] = sampler.draw().to_bits());
            }
            if let Some(frame) = frames.last() {
                let at = ((frames.len() - 1) * s.codebooks * 4) as u64;
                self.queue.write_buffer(&s.history, at, bytemuck::cast_slice(frame));
            }
        } else {
            words[2] = banned.len().min(MAX_BANNED) as u32;
            banned.iter().take(MAX_BANNED).enumerate().for_each(|(i, &token)| words[ban + i] = token);
        }
        self.queue.write_buffer(&s.step, 0, bytemuck::cast_slice(&words));
        self.queue.write_buffer(&s.state, 0, bytemuck::cast_slice(&[rows as u32, s.position as u32]));
        let head: &[Op] = if speaking { &s.speech } else { &s.text };
        let picked = self.submit(&[(first, 1), (&s.lm[..], rows), (head, rows)], &s.picked, MAX_CODEBOOKS * 4)?;
        s.position += rows;
        Ok(picked[..if speaking { s.codebooks } else { 1 }].to_vec())
    }
}
