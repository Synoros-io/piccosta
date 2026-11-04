// Process identity and lifecycle. The rest of the harness uses the same operations; only these
// small pieces know how a kernel names, waits for and stops a process.
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::*;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::*;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;
#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

// A start time counts from boot. A record from another boot must never identify a current pid.
pub fn recorded_start(record: &str) -> Option<u64> {
    let (boot, start) = record.trim().split_once(' ')?;
    (!boot.is_empty() && boot == boot_id()).then(|| start.parse().ok())?
}

pub fn exe() -> std::io::Result<std::path::PathBuf> {
    Ok(std::path::PathBuf::from(std::env::current_exe()?.to_string_lossy().trim_end_matches(" (deleted)")))
}

// Console interrupts close runtimes before leaving. TUI input owns Ctrl-C itself, so it asks
// only for termination and hangup; Windows has no Unix hangup and uses the console close event.
pub async fn termination(interrupt: bool) -> u8 {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let (Ok(mut term), Ok(mut hangup)) = (signal(SignalKind::terminate()), signal(SignalKind::hangup())) else { return std::future::pending().await };
        // Catching SIGINT turns its default (ending the process) off, so it's caught only when asked.
        let mut int = if interrupt { signal(SignalKind::interrupt()).ok() } else { None };
        let interrupted = async {
            match int.as_mut() {
                Some(int) => drop(int.recv().await),
                None => std::future::pending().await,
            }
        };
        tokio::select! { _ = term.recv() => 143, _ = hangup.recv() => 129, () = interrupted => 143 }
    }
    #[cfg(windows)]
    {
        use tokio::signal::windows::{ctrl_break, ctrl_close, ctrl_shutdown};
        let (Ok(mut brk), Ok(mut close), Ok(mut shutdown)) = (ctrl_break(), ctrl_close(), ctrl_shutdown()) else { return std::future::pending().await };
        tokio::select! { _ = brk.recv() => 143, _ = close.recv() => 143, _ = shutdown.recv() => 143, _ = tokio::signal::ctrl_c(), if interrupt => 143 }
    }
}
