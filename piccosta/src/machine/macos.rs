use super::Resources;

fn sysctl<T: Default>(name: &std::ffi::CStr) -> Option<T> {
    let mut value = T::default();
    let mut size = size_of::<T>();
    (unsafe { libc::sysctlbyname(name.as_ptr(), (&raw mut value).cast(), &mut size, std::ptr::null_mut(), 0) } == 0 && size == size_of::<T>()).then_some(value)
}
pub(super) fn resources() -> Resources {
    let mut load = 0.0;
    let load = (unsafe { libc::getloadavg(&mut load, 1) } == 1).then_some(load);
    let memory = (|| {
        let total: u64 = sysctl(c"hw.memsize")?;
        let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        // One host right for the process, not a new send right on every warning check.
        static HOST: std::sync::OnceLock<libc::mach_port_t> = std::sync::OnceLock::new();
        #[allow(deprecated)]
        let host = *HOST.get_or_init(|| unsafe { libc::mach_host_self() });
        #[allow(deprecated)]
        let got = unsafe { libc::host_statistics64(host, libc::HOST_VM_INFO64, (&raw mut stats).cast(), &mut count) };
        if got != libc::KERN_SUCCESS || page <= 0 { return None; }
        let available = (u64::from(stats.free_count) + u64::from(stats.inactive_count)) * page as u64;
        Some((total as f64 / 1024.0, available as f64 / 1024.0))
    })();
    let swap = {
        let mut usage: libc::xsw_usage = unsafe { std::mem::zeroed() };
        let mut size = size_of_val(&usage);
        (unsafe { libc::sysctlbyname(c"vm.swapusage".as_ptr(), (&raw mut usage).cast(), &mut size, std::ptr::null_mut(), 0) } == 0).then_some((usage.xsu_total as f64 / 1024.0, usage.xsu_avail as f64 / 1024.0))
    };
    Resources { load, memory, swap }
}
