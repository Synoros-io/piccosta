use super::Resources;
use std::os::windows::ffi::OsStrExt;
use windows_sys::Win32::{Foundation::MAX_PATH, System::SystemInformation::*, Storage::FileSystem::*};

pub(super) fn resources() -> Resources {
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of_val(&status) as u32;
    let memory = (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some((status.ullTotalPhys as f64 / 1024.0, status.ullAvailPhys as f64 / 1024.0));
    // Windows has no Unix load average. ullTotalPageFile is the commit limit, not swap size;
    // calling it swap would issue false warnings, so only physical memory and disk are reported.
    Resources { load: None, memory, swap: None }
}
pub(super) fn disk(path: &std::path::Path) -> Option<(u64, f64, f64)> {
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let (mut available, mut total, mut free) = (0, 0, 0);
    if unsafe { GetDiskFreeSpaceExW(path.as_ptr(), &mut available, &mut total, &mut free) } == 0 || total == 0 { return None; }
    // Volume serial numbers let cwd and temp on the same disk share one warning.
    let mut root = vec![0u16; MAX_PATH as usize + 1];
    if unsafe { GetVolumePathNameW(path.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 { return None; }
    let mut serial = 0;
    if unsafe { GetVolumeInformationW(root.as_ptr(), std::ptr::null_mut(), 0, &mut serial, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0) } == 0 { return None; }
    Some((u64::from(serial), (total - available) as f64, total as f64))
}
