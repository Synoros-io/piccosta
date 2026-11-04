// A local byte stream with peer identity, synchronous deadlines and an async listener. Keeping
// these here lets runtime.rs own exactly one delivery/locking protocol on all three platforms.
#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;
