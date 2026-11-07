use std::collections::HashMap;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::{CommandExt, ExitStatusExt};
use std::sync::{Arc, Mutex, LazyLock};
use windows_sys::Win32::{Foundation::*, System::{Threading::*, JobObjects::*, Diagnostics::ToolHelp::*}};

// These are operation tags, not Windows signal numbers. Windows stops the whole Job Object.
pub const TERM: i32 = 15;
pub const KILL: i32 = 9;
const STOP_EXIT_CODE: u32 = 1;

fn owned(handle: HANDLE) -> Option<OwnedHandle> {
    (!handle.is_null() && handle != INVALID_HANDLE_VALUE).then(|| unsafe { OwnedHandle::from_raw_handle(handle) })
}

#[derive(Clone)]
pub struct Process { handle: Arc<OwnedHandle> }
impl Process {
    pub fn open(pid: i32) -> Option<Self> {
        if let Some(job) = JOBS.lock().unwrap().get(&(pid as u32)) { return Some(job.process.clone()); }
        Self::raw(pid)
    }
    fn raw(pid: i32) -> Option<Self> {
        owned(unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | PROCESS_SET_QUOTA | PROCESS_SYNCHRONIZE, 0, pid as u32) }).map(|handle| Self { handle: Arc::new(handle) })
    }
    fn start(&self) -> Option<u64> {
        let (mut created, mut exited, mut kernel, mut user) = unsafe { std::mem::zeroed::<(FILETIME, FILETIME, FILETIME, FILETIME)>() };
        (unsafe { GetProcessTimes(self.handle.as_raw_handle(), &mut created, &mut exited, &mut kernel, &mut user) } != 0).then_some((u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime))
    }
    pub fn started(pid: i32, start: u64) -> Option<Self> { Self::open(pid).filter(|process| process.start() == Some(start) && process.signal(0)) }
    pub fn signal(&self, signal: i32) -> bool {
        if signal == 0 { unsafe { WaitForSingleObject(self.handle.as_raw_handle(), 0) == WAIT_TIMEOUT } }
        else { unsafe { TerminateProcess(self.handle.as_raw_handle(), STOP_EXIT_CODE) != 0 } }
    }
}
pub fn start(pid: i32) -> Option<u64> { Process::open(pid).filter(|p| p.signal(0))?.start() }

// Creation times are absolute FILETIMEs on Windows, not ticks since boot. The fixed domain tag
// is intentional: a reboot cannot make an old creation time identify a newly reused pid.
pub fn boot_id() -> String { "windows-filetime".into() }
pub fn children() -> Vec<i32> { Vec::new() }

#[derive(Clone)]
pub struct JobHandle(Arc<OwnedHandle>);
impl JobHandle {
    pub fn kill(&self) { unsafe { TerminateJobObject(self.0.as_raw_handle(), STOP_EXIT_CODE); } }
}
pub fn job(pid: u32) -> Option<JobHandle> { JOBS.lock().unwrap().get(&pid).map(|job| job.handle.clone()) }
struct Job { handle: JobHandle, process: Process }
static JOBS: LazyLock<Mutex<HashMap<u32, Job>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn own_session(command: &mut std::process::Command) -> &mut std::process::Command {
    command.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
}
pub fn own_group(command: &mut std::process::Command) -> &mut std::process::Command { command.creation_flags(CREATE_NEW_PROCESS_GROUP) }
pub fn shell_setup(command: &mut std::process::Command, _read_only: bool) {
    command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP);
}
pub fn group(command: &mut tokio::process::Command) { command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP); }
pub fn parent_death(command: &mut std::process::Command) { command.creation_flags(CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP); }

// Start suspended, assign the job, then resume. Assigning after a normal spawn loses a fast
// child that starts helpers before assignment. A failed assignment must never run unsupervised.
pub fn attach(pid: u32) -> std::io::Result<()> {
    let process = Process::raw(pid as i32).ok_or_else(std::io::Error::last_os_error)?;
    let result = (|| {
        let handle = owned(unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) }).ok_or_else(std::io::Error::last_os_error)?;
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if unsafe { SetInformationJobObject(handle.as_raw_handle(), JobObjectExtendedLimitInformation, (&raw const limits).cast(), size_of_val(&limits) as u32) } == 0
            || unsafe { AssignProcessToJobObject(handle.as_raw_handle(), process.handle.as_raw_handle()) } == 0 { return Err(std::io::Error::last_os_error()); }
        let snapshot = owned(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) }).ok_or_else(std::io::Error::last_os_error)?;
        let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
        entry.dwSize = size_of_val(&entry) as u32;
        let mut more = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) } != 0;
        let mut resumed = false;
        while more {
            if entry.th32OwnerProcessID == pid {
                let thread = owned(unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) }).ok_or_else(std::io::Error::last_os_error)?;
                if unsafe { ResumeThread(thread.as_raw_handle()) } == u32::MAX { return Err(std::io::Error::last_os_error()); }
                resumed = true;
            }
            more = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) } != 0;
        }
        if !resumed { return Err(std::io::Error::other("command has no suspended thread")); }
        JOBS.lock().unwrap().insert(pid, Job { handle: JobHandle(Arc::new(handle)), process: process.clone() });
        Ok(())
    })();
    if result.is_err() { process.signal(KILL); }
    result
}
pub fn signal_group(pid: u32, _signal: i32) {
    if let Some(job) = JOBS.lock().unwrap().get(&pid) { job.handle.kill(); }
}
// The job alone; on Windows signal_group already touches nothing but the job.
pub fn kill_group(pid: u32, signal: i32) { signal_group(pid, signal) }
// A job is tied to its process by handle, so a number taken since can't reach it.
pub fn group_held(_pid: u32, _start: Option<u64>) -> bool { true }
pub fn release(pid: u32) { JOBS.lock().unwrap().remove(&pid); }
pub async fn wait(pid: u32) -> std::io::Result<std::process::ExitStatus> {
    let process = Process::open(pid as i32).ok_or_else(std::io::Error::last_os_error)?;
    tokio::task::spawn_blocking(move || {
        let mut code = 0;
        if unsafe { WaitForSingleObject(process.handle.as_raw_handle(), INFINITE) } != WAIT_OBJECT_0
            || unsafe { GetExitCodeProcess(process.handle.as_raw_handle(), &mut code) } == 0 { return Err(std::io::Error::last_os_error()); }
        Ok(std::process::ExitStatus::from_raw(code))
    }).await.map_err(std::io::Error::other)?
}
pub fn exit_code(status: std::process::ExitStatus) -> i32 { status.code().unwrap_or(1) }

pub struct Receiver(tokio::sync::Mutex<tokio::fs::File>);
pub fn pipe(reader: std::io::PipeReader) -> std::io::Result<Receiver> {
    Ok(Receiver(tokio::sync::Mutex::new(tokio::fs::File::from_std(std::fs::File::from(OwnedHandle::from(reader))))))
}
pub async fn read_pipe(pipe: &Receiver, buffer: &mut [u8]) -> std::io::Result<usize> {
    use tokio::io::AsyncReadExt;
    pipe.0.lock().await.read(buffer).await
}
pub fn resume(command: &mut std::process::Command) -> std::io::Error {
    match command.status() { Ok(status) => std::process::exit(status.code().unwrap_or(1)), Err(error) => error }
}
pub fn link(exe: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> { std::fs::copy(exe, link).map(|_| ()) }
