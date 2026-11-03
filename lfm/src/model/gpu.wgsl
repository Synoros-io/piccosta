// lfm's GPU kernels, for gpu.rs: the language model's and the depth decoder's arithmetic as compute shaders. The
// file is cut at its "//!" lines. The part before the first is the prelude every kernel starts with; each part
// after one is a kernel, named on that line and compiled as its own module behind the prelude. Bindings 0 and 1
// are the prelude's weights, and a kernel's own bindings follow from 2, in the order gpu.rs passes its buffers.
// Every kernel but gemm runs LANES threads a workgroup.

// The formats a weight matrix comes in, as tensor.rs's Data has them.
const BF16: u32 = 0u;
const Q8: u32 = 1u;
const Q4: u32 = 2u;
const LANES: u32 = 64u;
// Stands in for minus infinity, which shader compilers may assume never occurs: exp of it less anything is 0.
const LOWEST: f32 = -3.0e38;

// A weight matrix: its codes as 16-byte words, and its scales. bf16 is two values a word and has no scales. q8 is
// four codes a word and an f32 scale a row. q4 is a group of 32 codes a 16-byte word, byte j holding value j's
// code low and value j + 16's high, with a bf16 scale and minimum a group, the scale in the low half.
@group(0) @binding(0) var<storage, read> codes: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;

// What a stack reads this step: how many rows, and the position of the first.
struct Stack { rows: u32, position: u32 }

// What the host writes before each step: `count` rows of a table to sum into the language model's next input
// (`ids`: a token, or a speech frame's code in each codebook's rows), the frames generated so far, the order the
// codebooks are revealed in with a random draw for each, and the text tokens never to pick.
struct Step {
    count: u32,
    frames: u32,
    banned: u32,
    ids: array<u32, 16>,
    order: array<u32, 16>,
    draws: array<f32, 16>,
    ban: array<u32, 8>,
}

// Values 8·chunk to 8·chunk + 7 of weight row `row`, of a matrix with `groups` groups of 32 values a row,
// widened to f32 as tensor.rs and quant.rs widen them.
fn weights8(format: u32, groups: u32, row: u32, chunk: u32) -> array<vec4<f32>, 2> {
    let group = row * groups + chunk / 4u;
    let part = chunk % 4u;
    if format == Q4 {
        // values 0-7 of a group are the low nibbles of its first 8 bytes, 8-15 of its last 8; 16-31 the high ones
        let word = codes[group];
        let pair = scales[group];
        let two = select(word.xy, word.zw, part % 2u == 1u);
        let shift = vec4<u32>(0u, 8u, 16u, 24u) + part / 2u * 4u;
        let scale = bitcast<f32>(pair << 16u);
        let minimum = bitcast<f32>(pair & 0xffff0000u);
        return array(
            vec4<f32>((vec4<u32>(two.x) >> shift) & vec4<u32>(15u)) * scale + minimum,
            vec4<f32>((vec4<u32>(two.y) >> shift) & vec4<u32>(15u)) * scale + minimum,
        );
    }
    if format == Q8 {
        let word = codes[group * 2u + part / 2u];
        let two = select(word.xy, word.zw, part % 2u == 1u);
        let scale = bitcast<f32>(scales[row]);
        let shift = vec4<u32>(24u, 16u, 8u, 0u);
        return array(
            vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(two.x) << shift) >> vec4<u32>(24u)) * scale,
            vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(two.y) << shift) >> vec4<u32>(24u)) * scale,
        );
    }
    let word = codes[group * 4u + part];
    let low = word << vec4<u32>(16u);
    let high = word & vec4<u32>(0xffff0000u);
    return array(
        bitcast<vec4<f32>>(vec4<u32>(low.x, high.x, low.y, high.y)),
        bitcast<vec4<f32>>(vec4<u32>(low.z, high.z, low.w, high.w)),
    );
}

var<workgroup> partial: array<f32, LANES>;

// The sum, or with `largest` the largest, of every lane's value, given back to all of them. Every lane calls it.
fn reduce(value: f32, lane: u32, largest: bool) -> f32 {
    partial[lane] = value;
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            let other = partial[lane + stride];
            partial[lane] = select(partial[lane] + other, max(partial[lane], other), largest);
        }
        workgroupBarrier();
    }
    let result = partial[0];
    workgroupBarrier();
    return result;
}

var<workgroup> best_value: array<f32, LANES>;
var<workgroup> best_index: array<u32, LANES>;

// The index of the largest of every lane's best value, the lowest index among equal ones, as tensor.rs's argmax
// takes the first. Every lane calls it.
fn best(value: f32, index: u32, lane: u32) -> u32 {
    best_value[lane] = value;
    best_index[lane] = index;
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            let other = best_value[lane + stride];
            let at = best_index[lane + stride];
            if other > best_value[lane] || (other == best_value[lane] && at < best_index[lane]) {
                best_value[lane] = other;
                best_index[lane] = at;
            }
        }
        workgroupBarrier();
    }
    let result = best_index[0];
    workgroupBarrier();
    return result;
}

//! matvec1
// y = x·Wᵀ for one row of x, four weight rows a workgroup: each lane widens its share of the four rows' values
// and multiplies them by the same values of x, read once. (A lane taking whole groups of 32 instead read the q8
// depth_linear at two thirds the speed.) With `cached`, the row is written at row `position`, into a key or value
// cache.
struct Args { format: u32, groups: u32, outputs: u32, cached: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

var<workgroup> sums: array<f32, 256>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) id: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let first = (id.x + id.y * count.x) * 4u;
    if first >= args.outputs {
        return;
    }
    var mine = vec4<f32>(0.0);
    for (var chunk = lane; chunk < args.groups * 4u; chunk += LANES) {
        let a = x[chunk * 2u];
        let b = x[chunk * 2u + 1u];
        for (var r = 0u; r < 4u; r++) {
            let w = weights8(args.format, args.groups, min(first + r, args.outputs - 1u), chunk);
            mine[r] += dot(w[0], a) + dot(w[1], b);
        }
    }
    for (var r = 0u; r < 4u; r++) {
        sums[r * LANES + lane] = mine[r];
    }
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            for (var r = 0u; r < 4u; r++) {
                sums[r * LANES + lane] += sums[r * LANES + lane + stride];
            }
        }
        workgroupBarrier();
    }
    if lane < 4u && first + lane < args.outputs {
        y[select(0u, stack.position, args.cached == 1u) * args.outputs + first + lane] = sums[lane * LANES];
    }
}

//! matvec
// y = x·Wᵀ for 2 to 16 rows of x: four weight rows a workgroup and sixteen lanes a weight row, each lane taking whole
// groups of 32 columns, widening the row's values there once and multiplying them into every row of x. `cached`
// as in matvec1, row r going to row position + r.
struct Args { format: u32, groups: u32, outputs: u32, cached: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

const ROWS: u32 = 16u;
var<workgroup> sums: array<f32, 1024>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) id: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let output = (id.x + id.y * count.x) * 4u + lane / 16u;
    let part = lane % 16u;
    let rows = min(stack.rows, ROWS);
    var mine: array<f32, ROWS>;
    if output < args.outputs {
        for (var group = part; group < args.groups; group += 16u) {
            for (var quarter = 0u; quarter < 4u; quarter++) {
                let chunk = group * 4u + quarter;
                let w = weights8(args.format, args.groups, output, chunk);
                for (var r = 0u; r < ROWS; r++) {
                    if r < rows {
                        let at = (r * args.groups * 4u + chunk) * 2u;
                        mine[r] += dot(w[0], x[at]) + dot(w[1], x[at + 1u]);
                    }
                }
            }
        }
    }
    for (var r = 0u; r < ROWS; r++) {
        sums[r * LANES + lane] = mine[r];
    }
    workgroupBarrier();
    for (var stride = 8u; stride > 0u; stride /= 2u) {
        if part < stride {
            for (var r = 0u; r < rows; r++) {
                sums[r * LANES + lane] += sums[r * LANES + lane + stride];
            }
        }
        workgroupBarrier();
    }
    // lane `part` of the weight row's sixteen writes row `part` of x's result
    if part < rows && output < args.outputs {
        let row = part + select(0u, stack.position, args.cached == 1u);
        y[row * args.outputs + output] = sums[part * LANES + lane - part];
    }
}
//! gemm
// y = x·Wᵀ for many rows of x: a workgroup takes 64 rows of x by 64 weight rows, staging a group of 32 columns of
// each in workgroup memory at a time, and each of its 256 threads sums a 4 × 4 block of the result. `cached` as
// in matvec.
struct Args { format: u32, groups: u32, outputs: u32, cached: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

// column k of the group for row (or weight row) i at k·64 + i
var<workgroup> xs: array<f32, 2048>;
var<workgroup> ws: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) thread: u32) {
    let first_output = id.x * 64u;
    let first_row = id.y * 64u;
    let tx = thread % 16u;
    let ty = thread / 16u;
    // each thread stages a fourth of one row's group: 8 values of x and 8 weights
    let staged = thread / 4u;
    let part = thread % 4u;
    let row = first_row + staged;
    let output = first_output + staged;
    var block: array<vec4<f32>, 4>;
    for (var group = 0u; group < args.groups; group++) {
        let chunk = group * 4u + part;
        var a = array(vec4<f32>(0.0), vec4<f32>(0.0));
        var w = array(vec4<f32>(0.0), vec4<f32>(0.0));
        if row < stack.rows {
            let at = (row * args.groups * 4u + chunk) * 2u;
            a = array(x[at], x[at + 1u]);
        }
        if output < args.outputs {
            w = weights8(args.format, args.groups, output, chunk);
        }
        for (var i = 0u; i < 8u; i++) {
            let k = part * 8u + i;
            xs[k * 64u + staged] = a[i / 4u][i % 4u];
            ws[k * 64u + staged] = w[i / 4u][i % 4u];
        }
        workgroupBarrier();
        for (var k = 0u; k < 32u; k++) {
            let at = k * 64u;
            let xv = vec4<f32>(xs[at + ty * 4u], xs[at + ty * 4u + 1u], xs[at + ty * 4u + 2u], xs[at + ty * 4u + 3u]);
            let wv = vec4<f32>(ws[at + tx * 4u], ws[at + tx * 4u + 1u], ws[at + tx * 4u + 2u], ws[at + tx * 4u + 3u]);
            block[0] += xv.x * wv;
            block[1] += xv.y * wv;
            block[2] += xv.z * wv;
            block[3] += xv.w * wv;
        }
        workgroupBarrier();
    }
    let offset = select(0u, stack.position, args.cached == 1u);
    for (var i = 0u; i < 4u; i++) {
        let row = first_row + ty * 4u + i;
        for (var j = 0u; j < 4u; j++) {
            let output = first_output + tx * 4u + j;
            if row < stack.rows && output < args.outputs {
                y[(row + offset) * args.outputs + output] = block[i][j];
            }
        }
    }
}

//! residual
// The residual stream after an operator or a feed-forward, and the norm the next part reads: x += m, RMS-normed
// by `post_weight` first under sandwich norms, then h = RMSNorm(x)·next_weight. Without `add` it only norms. With
// `each` the next norm has a weight row for every row (the depth decoder's per-codebook norms). A workgroup a row,
// four values at a time.
struct Args { width: u32, eps: f32, add: u32, post: u32, each: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> m: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> post_weight: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> next_weight: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> h: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let quads = args.width / 4u;
    let at = id.x * quads;
    var scale = 1.0;
    if args.add == 1u && args.post == 1u {
        var squares = 0.0;
        for (var i = lane; i < quads; i += LANES) {
            squares += dot(m[at + i], m[at + i]);
        }
        scale = 1.0 / sqrt(reduce(squares, lane, false) / f32(args.width) + args.eps);
    }
    var squares = 0.0;
    for (var i = lane; i < quads; i += LANES) {
        var value = x[at + i];
        if args.add == 1u {
            // an if, not a select, which would read post_weight past its end when there's no post-norm
            var change = m[at + i];
            if args.post == 1u {
                change = change * scale * post_weight[i];
            }
            value += change;
            x[at + i] = value;
        }
        squares += dot(value, value);
    }
    let norm = 1.0 / sqrt(reduce(squares, lane, false) / f32(args.width) + args.eps);
    let weights = select(0u, at, args.each == 1u);
    for (var i = lane; i < quads; i += LANES) {
        h[at + i] = x[at + i] * norm * next_weight[weights + i];
    }
}
//! conv
// LFM2's short convolution, a lane per channel: B·x through a causal depthwise filter over each row and the
// width - 1 before it (`tail` keeps the last of them between steps), gated by C. bcx is a row's B, C, x.
struct Args { width: u32, hidden: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> bcx: array<f32>;
@group(0) @binding(5) var<storage, read> taps: array<f32>;
@group(0) @binding(6) var<storage, read_write> tail: array<f32>;
@group(0) @binding(7) var<storage, read_write> gated: array<f32>;

// Input i of the filter in channel c: the tail first, then each row's B·x.
fn source(i: u32, c: u32) -> f32 {
    if i < args.width - 1u {
        return tail[i * args.hidden + c];
    }
    let at = (i - (args.width - 1u)) * 3u * args.hidden + c;
    return bcx[at] * bcx[at + 2u * args.hidden];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    let hidden = args.hidden;
    if c >= hidden {
        return;
    }
    for (var r = 0u; r < stack.rows; r++) {
        var sum = 0.0;
        for (var t = 0u; t < args.width; t++) {
            sum += taps[t * hidden + c] * source(r + t, c);
        }
        gated[r * hidden + c] = sum * bcx[r * 3u * hidden + hidden + c];
    }
    // the new tail is the last width - 1 inputs; entry j reads entry rows + j > j, so in place is safe
    for (var j = 0u; j + 1u < args.width; j++) {
        tail[j * hidden + c] = source(stack.rows + j, c);
    }
}

//! rope
// Every head's query and new key RMS-normed and rotated by its row's position, and its new values copied: the
// queries to q, the keys and values into the caches at the row's position. A row of `fused` holds a row's
// queries, keys, values and gates side by side, one dispatch's worth of q_proj, k_proj, v_proj and gate_proj. A
// workgroup per head of a row (queries first, then keys with their values), a lane per rotated pair (i, i + dim/2),
// with the cosines and sines from the host's table of positions.
struct Args { heads: u32, kv_heads: u32, dim: u32, eps: f32, width: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> q_norm: array<f32>;
@group(0) @binding(5) var<storage, read> k_norm: array<f32>;
@group(0) @binding(6) var<storage, read> rotary: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> fused: array<f32>;
@group(0) @binding(8) var<storage, read_write> q: array<f32>;
@group(0) @binding(9) var<storage, read_write> keys: array<f32>;
@group(0) @binding(10) var<storage, read_write> values: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let pairs = args.dim / 2u;
    let position = stack.position + id.x;
    let head = id.y;
    let query = head < args.heads;
    let read_at = id.x * args.width + head * args.dim;
    let at = select((position * args.kv_heads + head - args.heads) * args.dim, (id.x * args.heads + head) * args.dim, query);
    var a = 0.0;
    var b = 0.0;
    if lane < pairs {
        a = fused[read_at + lane];
        b = fused[read_at + lane + pairs];
    }
    if !query {
        // this key head's values follow all the keys
        for (var i = lane; i < args.dim; i += LANES) {
            values[at + i] = fused[read_at + args.kv_heads * args.dim + i];
        }
    }
    let scale = 1.0 / sqrt(reduce(a * a + b * b, lane, false) / f32(args.dim) + args.eps);
    if lane < pairs {
        a = a * scale * select(k_norm[lane], q_norm[lane], query);
        b = b * scale * select(k_norm[lane + pairs], q_norm[lane + pairs], query);
        let turn = rotary[position * pairs + lane];
        let first = a * turn.x - b * turn.y;
        let second = b * turn.x + a * turn.y;
        if query {
            q[at + lane] = first;
            q[at + lane + pairs] = second;
        } else {
            keys[at + lane] = first;
            keys[at + lane + pairs] = second;
        }
    }
}

//! attention
// Scaled dot-product attention, a workgroup per head of a query row, over the keys that row sees: up to its own
// position when causal, within the last `window` when there is one. The keys go 64 at a time, a lane each, with
// a running softmax; a lane keeps output values lane and lane + 64. With `gated` each head's output is scaled by
// the sigmoid of its gate, which is at `gates` in its row of `fused`, rows `width` apart.
struct Args { heads: u32, kv_heads: u32, dim: u32, causal: u32, window: u32, gated: u32, gates: u32, width: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> keys: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> values: array<f32>;
@group(0) @binding(7) var<storage, read> fused: array<f32>;
@group(0) @binding(8) var<storage, read_write> heads_out: array<f32>;

var<workgroup> query: array<vec4<f32>, 32>;
var<workgroup> weights: array<f32, LANES>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = id.x;
    let head = id.y;
    let dim = args.dim;
    let quads = dim / 4u;
    let kv = head / (args.heads / args.kv_heads);
    let position = stack.position + row;
    let end = select(stack.position + stack.rows, position + 1u, args.causal == 1u);
    let start = select(0u, max(position + 1u, args.window) - args.window, args.window > 0u);
    if lane < quads {
        query[lane] = q[(row * args.heads + head) * quads + lane];
    }
    workgroupBarrier();
    let scale = 1.0 / sqrt(f32(dim));
    var top = LOWEST;
    var total = 0.0;
    var first = 0.0;
    var second = 0.0;
    for (var block = start; block < end; block += LANES) {
        let key = block + lane;
        var score = LOWEST;
        if key < end {
            var sum = 0.0;
            let at = (key * args.kv_heads + kv) * quads;
            for (var i = 0u; i < quads; i++) {
                sum += dot(query[i], keys[at + i]);
            }
            score = sum * scale;
        }
        let new_top = max(top, reduce(score, lane, true));
        let weight = select(0.0, exp(score - new_top), key < end);
        let rescale = exp(top - new_top);
        total = total * rescale + reduce(weight, lane, false);
        weights[lane] = weight;
        workgroupBarrier();
        var one = 0.0;
        var two = 0.0;
        for (var j = 0u; j < min(LANES, end - block); j++) {
            let at = ((block + j) * args.kv_heads + kv) * dim + lane;
            if lane < dim {
                one += weights[j] * values[at];
            }
            if lane + LANES < dim {
                two += weights[j] * values[at + LANES];
            }
        }
        first = first * rescale + one;
        second = second * rescale + two;
        top = new_top;
        workgroupBarrier();
    }
    let at = (row * args.heads + head) * dim + lane;
    // an if, not a select, which would read the gate past the end of `fused` when there's none
    var gate = 1.0;
    if args.gated == 1u {
        gate = 1.0 / (1.0 + exp(-fused[row * args.width + args.gates + head]));
    }
    let scaled = gate / total;
    if lane < dim {
        heads_out[at] = first * scaled;
    }
    if lane + LANES < dim {
        heads_out[at + LANES] = second * scaled;
    }
}

//! swiglu
// The feed-forward's gate, silu(gate)·up, for every value of every row: w1's gate and w3's up come side by side in
// a row of `both`, and the gated row goes to g.
struct Args { width: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> both: array<f32>;
@group(0) @binding(5) var<storage, read_write> g: array<f32>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) id: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let i = (id.x + id.y * count.x) * LANES + lane;
    if i < stack.rows * args.width {
        let at = i / args.width * 2u * args.width + i % args.width;
        let gate = both[at];
        g[i] = gate * (1.0 / (1.0 + exp(-gate))) * both[at + args.width];
    }
}

//! last
// The last row of a stack's output: what the text head and the depth decoder read.
struct Args { width: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> stack: Stack;
@group(0) @binding(4) var<storage, read> h: array<f32>;
@group(0) @binding(5) var<storage, read_write> hidden: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < args.width {
        hidden[id.x] = h[(stack.rows - 1u) * args.width + id.x];
    }
}

//! embed
// The language model's next input: the sum of the step's rows of a table, a token's embedding or a speech
// frame's codes in the shared codebook table. A lane per 8 columns.
struct Args { format: u32, groups: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> current: Step;
@group(0) @binding(4) var<storage, read_write> x: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let chunk = id.x;
    if chunk >= args.groups * 4u {
        return;
    }
    var sum = array(vec4<f32>(0.0), vec4<f32>(0.0));
    for (var i = 0u; i < current.count; i++) {
        let w = weights8(args.format, args.groups, current.ids[i], chunk);
        sum[0] += w[0];
        sum[1] += w[1];
    }
    x[chunk * 2u] = sum[0];
    x[chunk * 2u + 1u] = sum[1];
}

//! slots
// The depth decoder's input for a pass: each codebook's slot of depth_linear's output and its bias, plus the
// embedding of the code chosen for it, or the mask embedding while it is hidden. The codebooks revealed so far
// are the first `known` of the step's order. A lane per 8 columns of a codebook's slot.
struct Args { format: u32, groups: u32, size: u32, known: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> current: Step;
@group(0) @binding(4) var<storage, read> slots: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> mask: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> picked: array<u32>;
@group(0) @binding(8) var<storage, read_write> x: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let chunk = id.x;
    let codebook = id.y;
    if chunk >= args.groups * 4u {
        return;
    }
    var known = false;
    for (var i = 0u; i < args.known; i++) {
        known = known || current.order[i] == codebook;
    }
    var added = array(mask[chunk * 2u], mask[chunk * 2u + 1u]);
    if known {
        added = weights8(args.format, args.groups, codebook * args.size + picked[codebook], chunk);
    }
    let at = codebook * args.groups * 8u + chunk * 2u;
    x[at] = slots[at] + bias[at] + added[0];
    x[at + 1u] = slots[at + 1u] + bias[at + 1u] + added[1];
}

//! heads
// Every codebook's scores for its next code: row c of x against codebook c's own table, rows c·size onward of
// the depth tables. A workgroup per code of a codebook.
struct Args { format: u32, groups: u32, size: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let code = id.x;
    let codebook = id.y;
    var sum = 0.0;
    for (var chunk = lane; chunk < args.groups * 4u; chunk += LANES) {
        let w = weights8(args.format, args.groups, codebook * args.size + code, chunk);
        let at = (codebook * args.groups * 4u + chunk) * 2u;
        sum += dot(w[0], x[at]) + dot(w[1], x[at + 1u]);
    }
    let total = reduce(sum, lane, false);
    if lane == 0u {
        logits[codebook * args.size + code] = total;
    }
}

//! sample
// Each code of this pass, as model.rs's Sampler draws it: the repetition penalty over the last `window` frames,
// then the likeliest code at temperature 0, or else the temperature, top-k and a draw from the softmax with the
// host's random number. A workgroup per codebook revealed in the pass, which is entry first + i of the order.
struct Args { size: u32, codebooks: u32, first: u32, window: u32, temperature: f32, penalty: f32, top_k: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> current: Step;
@group(0) @binding(4) var<storage, read> history: array<u32>;
@group(0) @binding(5) var<storage, read> logits: array<f32>;
@group(0) @binding(6) var<storage, read_write> picked: array<u32>;

// a bit per code the window saw, and the codebook's scores
var<workgroup> seen: array<atomic<u32>, 64>;
var<workgroup> scores: array<f32, 2048>;
var<workgroup> counted: atomic<u32>;

// Scores in the order of their u32 keys: total_cmp's order, negative values' bits reversed.
fn key(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(bits | 0x80000000u, ~bits, bits >> 31u == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let codebook = current.order[args.first + id.x];
    let size = args.size;
    if lane < 64u {
        atomicStore(&seen[lane], 0u);
    }
    workgroupBarrier();
    let frames = current.frames;
    for (var frame = select(0u, frames - args.window, frames > args.window) + lane; frame < frames; frame += LANES) {
        let code = history[frame * args.codebooks + codebook];
        atomicOr(&seen[code / 32u], 1u << (code % 32u));
    }
    workgroupBarrier();
    // each code the window saw is penalized once: positive scores shrink, negative ones grow
    for (var n = lane; n < size; n += LANES) {
        var score = logits[codebook * size + n];
        if ((atomicLoad(&seen[n / 32u]) >> (n % 32u)) & 1u) == 1u {
            score = select(score / args.penalty, score * args.penalty, score < 0.0);
        }
        scores[n] = select(score, score / args.temperature, args.temperature > 0.0);
    }
    workgroupBarrier();
    if args.temperature > 0.0 {
        // keep the top_k scores, and any equal to the k-th: the k-th largest key, found a bit at a time from the top
        if args.top_k > 0u && args.top_k < size {
            var threshold = 0u;
            for (var bit = 31i; bit >= 0i; bit--) {
                let candidate = threshold | (1u << u32(bit));
                if lane == 0u {
                    atomicStore(&counted, 0u);
                }
                workgroupBarrier();
                var above = 0u;
                for (var n = lane; n < size; n += LANES) {
                    above += u32(key(scores[n]) >= candidate);
                }
                atomicAdd(&counted, above);
                workgroupBarrier();
                if atomicLoad(&counted) >= args.top_k {
                    threshold = candidate;
                }
                workgroupBarrier();
            }
            for (var n = lane; n < size; n += LANES) {
                if key(scores[n]) < threshold {
                    scores[n] = LOWEST;
                }
            }
            workgroupBarrier();
        }
        var top = LOWEST;
        for (var n = lane; n < size; n += LANES) {
            top = max(top, scores[n]);
        }
        top = reduce(top, lane, true);
        var total = 0.0;
        for (var n = lane; n < size; n += LANES) {
            let weight = select(exp(scores[n] - top), 0.0, scores[n] == LOWEST);
            scores[n] = weight;
            total += weight;
        }
        total = reduce(total, lane, false);
        for (var n = lane; n < size; n += LANES) {
            scores[n] /= total;
        }
        workgroupBarrier();
    }
    var largest = LOWEST;
    var index = 0u;
    for (var n = lane; n < size; n += LANES) {
        if scores[n] > largest {
            largest = scores[n];
            index = n;
        }
    }
    let likeliest = best(largest, index, lane);
    if lane == 0u {
        var code = likeliest;
        if args.temperature > 0.0 {
            // walk the probabilities in code order until the draw falls inside one, as Sampler::sample does
            var draw = current.draws[args.first + id.x];
            for (var n = 0u; n < size; n++) {
                if draw < scores[n] {
                    code = n;
                    break;
                }
                draw -= scores[n];
            }
        }
        picked[codebook] = code;
    }
}

//! argmax
// The likeliest text token: the largest score, the first of equal ones, never one the step bans. It runs twice:
// first each workgroup finds the best of its share of the scores and leaves it, as a score and an index, in
// `candidates`; then one workgroup finds the best of those.
struct Args { count: u32, final_pass: u32 }
@group(0) @binding(2) var<storage, read> args: Args;
@group(0) @binding(3) var<storage, read> current: Step;
@group(0) @binding(4) var<storage, read> logits: array<f32>;
@group(0) @binding(5) var<storage, read_write> candidates: array<u32>;
@group(0) @binding(6) var<storage, read_write> picked: array<u32>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) id: vec3<u32>,
    @builtin(num_workgroups) count: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    var largest = LOWEST;
    var index = 0u;
    if args.final_pass == 1u {
        for (var n = lane; n < args.count; n += LANES) {
            let score = bitcast<f32>(candidates[2u * n]);
            if score > largest || (score == largest && candidates[2u * n + 1u] < index) {
                largest = score;
                index = candidates[2u * n + 1u];
            }
        }
        let token = best(largest, index, lane);
        if lane == 0u {
            picked[0] = token;
        }
        return;
    }
    let share = (args.count + count.x - 1u) / count.x;
    for (var n = id.x * share + lane; n < min(args.count, (id.x + 1u) * share); n += LANES) {
        var banned = false;
        for (var i = 0u; i < current.banned; i++) {
            banned = banned || current.ban[i] == n;
        }
        if !banned && logits[n] > largest {
            largest = logits[n];
            index = n;
        }
    }
    let found = best(largest, index, lane);
    let top = reduce(largest, lane, true);
    if lane == 0u {
        candidates[2u * id.x] = bitcast<u32>(top);
        candidates[2u * id.x + 1u] = found;
    }
}
