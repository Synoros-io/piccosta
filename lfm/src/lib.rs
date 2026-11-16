// Liquid AI's LFM3-Audio-2B in plain Rust, on the CPU: speech to text and text to speech from the official
// safetensors checkpoint, with no machine-learning framework. Ported from Liquid AI's checkpoint code and
// changed from it, as each ported file says. This crate uses the LFM Open License v1.0 (LICENSE); NOTICE
// retains the checkpoint code's upstream attributions. The weights are Liquid AI's and licensed separately.

pub mod audio;
pub mod codec;
pub mod model;
mod normalize;
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
