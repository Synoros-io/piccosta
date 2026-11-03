// Liquid AI's LFM3-Audio-2B in plain Rust, on the CPU: speech to text and text to speech from the official
// safetensors checkpoint, with no machine-learning framework. Ported from the model's transformers code
// (Apache-2.0, Copyright 2026 The HuggingFace Team, full text in LICENSE-APACHE) and changed from it, as each
// ported file says; the weights are Liquid AI's, under the LFM Open License.

pub mod audio;
pub mod codec;
pub mod model;
pub mod quant;
pub mod tensor;
pub mod tokenizer;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

// set by the command line's -v
pub static VERBOSE: AtomicBool = AtomicBool::new(false);

// With -v, how long `what` took since `since`, on stderr.
pub fn note(what: &str, since: Instant) {
    if VERBOSE.load(Ordering::Relaxed) {
        eprintln!("lfm: {what}: {:.3} s", since.elapsed().as_secs_f64());
    }
}

// The process's peak and current resident memory, from Linux's /proc/self/status. RssFile is mostly the mapped
// weights, which the kernel can drop and re-read; RssAnon is what the process itself holds.
pub fn memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let fields = ["VmHWM", "RssAnon", "RssFile"];
    let lines = status.lines().filter(|line| fields.iter().any(|field| line.starts_with(field)));
    lines.map(|line| line.split_whitespace().collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join(", ")
}
