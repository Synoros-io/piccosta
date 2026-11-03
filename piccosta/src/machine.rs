// How busy the machine is, so a model can be told when it's overused. An agent that knows the
// load is high, memory is nearly gone or swap is filling, or $TMPDIR or its disk is nearly full
// can hold off on starting more and look at what's running, instead of adding to it. Nothing is stopped or
// slowed: the note is all this does, and the run carries on whatever the load. Linux only (it
// reads /proc); elsewhere it finds nothing. Each threshold below is the default for a `machine.*`
// setting of the same name (machine.loadPerCore, ...), for a machine whose normal looks different.

use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

// The 1-minute load average per core past which work queues for CPU: above 1.0 something is
// always waiting, and 1.5 leaves room for a build's short bursts.
const LOAD_PER_CORE: f64 = 1.5;
// Memory available (MemAvailable) under a tenth of the total: the kernel is about to reclaim
// hard or swap.
const MEMORY_AVAILABLE_FRACTION: f64 = 0.10;
// More than half the swap in use: the machine has been short of memory for a while. That alone
// isn't a note: zram setups sit at high swap use as their normal, so it counts only while memory
// is low or swap is still growing.
const SWAP_USED_FRACTION: f64 = 0.50;
// Swap counts as growing once it's up by a twentieth of its size within the last REPEAT: more than
// the drift of a steady machine, the start of one running short.
const SWAP_GROWTH_FRACTION: f64 = 0.05;
// $TMPDIR or the working directory's filesystem more than 90% full.
const DISK_USED_FRACTION: f64 = 0.90;
// While the machine stays overused, a session hears it again after this long.
const REPEAT: Duration = Duration::from_secs(600);
const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

// Swap use (kB) over the last REPEAT, each with when it was read, to tell growing swap from
// steady. One for the process: every session reads the same machine.
static SWAP_SEEN: Mutex<Vec<(Instant, f64)>> = Mutex::new(Vec::new());
// How often a swap reading is kept for the growth window: twenty in REPEAT's ten minutes.
const SWAP_SAMPLE_EVERY: Duration = Duration::from_secs(30);

// Each overused resource, in words ("load 18.4 on 12 cores"); empty when all is well.
fn strain(cwd: &Path) -> Vec<String> {
    let settings = crate::config::Settings::load(&crate::config::agent_dir(), cwd);
    let setting = |name: &str, default: f64| settings.get(&format!("machine.{name}")).and_then(serde_json::Value::as_f64).unwrap_or(default);
    let mut found = Vec::new();
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    let load: Option<f64> = std::fs::read_to_string("/proc/loadavg").ok().and_then(|text| text.split_whitespace().next()?.parse().ok());
    if let Some(load) = load.filter(|load| *load > setting("loadPerCore", LOAD_PER_CORE) * cores as f64) {
        found.push(format!("load {load:.1} on {cores} cores"));
    }
    // /proc/meminfo's values are in kB.
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kilobytes = |key: &str| meminfo.lines().find_map(|line| line.strip_prefix(key)?.split_whitespace().next()?.parse::<f64>().ok());
    let gigabytes = |kb: f64| kb * 1024.0 / BYTES_PER_GB;
    let mut memory_low = false;
    if let (Some(total), Some(available)) = (kilobytes("MemTotal:"), kilobytes("MemAvailable:"))
        && available < setting("memoryAvailableFraction", MEMORY_AVAILABLE_FRACTION) * total
    {
        memory_low = true;
        found.push(format!("memory {:.1} GB free of {:.0}", gigabytes(available), gigabytes(total)));
    }
    if let (Some(total), Some(free)) = (kilobytes("SwapTotal:"), kilobytes("SwapFree:"))
        && total > 0.0
    {
        // Growth is measured from the lowest use read in the last REPEAT, older readings dropped
        // first, so a rise has to be recent to count and one across the window's edge still does.
        let used = total - free;
        let growing = SWAP_SEEN.lock().is_ok_and(|mut seen| {
            seen.retain(|(at, _)| at.elapsed() <= REPEAT);
            let lowest = seen.iter().map(|(_, kb)| *kb).fold(used, f64::min);
            // At most one sample per SWAP_SAMPLE_EVERY, so the window holds a bounded few.
            if seen.last().is_none_or(|(at, _)| at.elapsed() >= SWAP_SAMPLE_EVERY) {
                seen.push((Instant::now(), used));
            }
            used - lowest > setting("swapGrowthFraction", SWAP_GROWTH_FRACTION) * total
        });
        if used > setting("swapUsedFraction", SWAP_USED_FRACTION) * total && (growing || memory_low) {
            found.push(format!("swap {:.0}% used{}", 100.0 * used / total, if growing { " and growing" } else { "" }));
        }
    }
    let mut seen_devices = Vec::new();
    let temp = std::env::temp_dir();
    for (name, path) in [(temp.display().to_string(), temp.as_path()), ("the working directory's disk".to_string(), cwd)] {
        let Some((device, used, total)) = disk(path) else { continue };
        if !seen_devices.contains(&device) && used > setting("diskUsedFraction", DISK_USED_FRACTION) * total {
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
