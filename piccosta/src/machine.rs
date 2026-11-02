// How busy the machine is, so a model can be told when it's overused. An agent that knows the
// load is high, memory or swap is nearly gone, or /tmp or its disk is nearly full can hold off
// on starting more and look at what's running, instead of adding to it. Nothing is stopped or
// slowed: the note is all this does, and the run carries on whatever the load. Linux only (it
// reads /proc); elsewhere it finds nothing.

use std::path::Path;
use std::time::{Duration, Instant};

// The 1-minute load average per core past which work queues for CPU: above 1.0 something is
// always waiting, and 1.5 leaves room for a build's short bursts.
const LOAD_PER_CORE: f64 = 1.5;
// Memory available (MemAvailable) under a tenth of the total: the kernel is about to reclaim
// hard or swap.
const MEMORY_AVAILABLE_FRACTION: f64 = 0.10;
// More than half the swap in use: the machine has been short of memory for a while.
const SWAP_USED_FRACTION: f64 = 0.50;
// /tmp or the working directory's filesystem more than 90% full.
const DISK_USED_FRACTION: f64 = 0.90;
// While the machine stays overused, a session hears it again after this long.
const REPEAT: Duration = Duration::from_secs(600);
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

// Each overused resource, in words ("load 18.4 on 12 cores"); empty when all is well.
fn strain(cwd: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    let load: Option<f64> = std::fs::read_to_string("/proc/loadavg").ok().and_then(|text| text.split_whitespace().next()?.parse().ok());
    if let Some(load) = load.filter(|load| *load > LOAD_PER_CORE * cores as f64) {
        found.push(format!("load {load:.1} on {cores} cores"));
    }
    // /proc/meminfo's values are in kB.
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kilobytes = |key: &str| meminfo.lines().find_map(|line| line.strip_prefix(key)?.split_whitespace().next()?.parse::<f64>().ok());
    let gigabytes = |kb: f64| kb * 1024.0 / BYTES_PER_GB;
    if let (Some(total), Some(available)) = (kilobytes("MemTotal:"), kilobytes("MemAvailable:"))
        && available < MEMORY_AVAILABLE_FRACTION * total
    {
        found.push(format!("memory {:.1} GB free of {:.0}", gigabytes(available), gigabytes(total)));
    }
    if let (Some(total), Some(free)) = (kilobytes("SwapTotal:"), kilobytes("SwapFree:"))
        && total > 0.0
        && total - free > SWAP_USED_FRACTION * total
    {
        found.push(format!("swap {:.0}% used", 100.0 * (total - free) / total));
    }
    let mut seen_devices = Vec::new();
    for (name, path) in [("/tmp", Path::new("/tmp")), ("the working directory's disk", cwd)] {
        let Some((device, used, total)) = disk(path) else { continue };
        if !seen_devices.contains(&device) && used > DISK_USED_FRACTION * total {
            found.push(format!("{name} {:.0}% full ({:.1} GB left)", 100.0 * used / total, (total - used) / BYTES_PER_GB));
        }
        seen_devices.push(device);
    }
    found
}

// A filesystem's id, bytes used and bytes in all, from statvfs.
fn disk(path: &Path) -> Option<(u64, f64, f64)> {
    let path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: path is a valid C string and stats a zeroed statvfs for the call to fill.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stats) } != 0 || stats.f_blocks == 0 {
        return None;
    }
    let block = stats.f_frsize as f64;
    let total = stats.f_blocks as f64 * block;
    Some((stats.f_fsid as u64, total - stats.f_bavail as f64 * block, total))
}

// The note for a session, when it's due: the machine is overused and the session hasn't heard
// so in REPEAT. `last` is when it last heard, cleared once the machine is fine again, so the next
// time it's overused it hears at once.
pub fn note(cwd: &Path, last: &mut Option<Instant>) -> Option<String> {
    let found = strain(cwd);
    if found.is_empty() {
        *last = None;
        return None;
    }
    if last.is_some_and(|at| at.elapsed() < REPEAT) {
        return None;
    }
    *last = Some(Instant::now());
    Some(format!(
        "[piccosta: this machine is overused: {}. Hold off on starting more heavy work (builds, test suites, model or data jobs, more agents); nice and job-cap what you do run, and look at what's using it (ps -eo pid,pcpu,rss,args --sort=-pcpu | head) before adding more. Don't stop processes that aren't yours. Carry on with your task.]",
        found.join(", ")
    ))
}
