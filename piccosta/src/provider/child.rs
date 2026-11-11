// The newline-framed child transport shared by the Claude bridge and ACP. Protocol dispatch stays
// with each provider; ownership of the pipes, stderr and process belongs here.

use serde_json::Value;
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::time::timeout;

// The bridge's existing SDK close deadlines: input ends, TERM after 2 s, KILL after 5 more.
pub(super) const CLOSE_GRACE: Duration = Duration::from_secs(2);
pub(crate) const TERM_GRACE: Duration = Duration::from_secs(5);
// Keep the bridge's last 4 KiB of stderr, not an unbounded child log.
const STDERR_TAIL: usize = 4096;
pub(super) struct Child {
    pub(super) pid: u32,
    pub(super) status: Option<ExitStatus>,
    pub(super) stdin: Option<ChildStdin>,
    pub(super) stdout: BufReader<ChildStdout>,
    pub(super) line: Vec<u8>,
    pub(super) stderr: Arc<Mutex<Vec<u8>>>,
    pub(super) stderr_reader: Option<tokio::task::JoinHandle<()>>,
    #[cfg(unix)]
    pub(super) stderr_fd: std::os::fd::OwnedFd,
}

impl Child {
    // Start an owned child in the command's directory and environment. Parent death and Drop both
    // end it; a failed pipe setup kills and reaps the process before returning the error.
    pub(super) fn start(mut command: std::process::Command) -> Result<Self, String> {
        command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        crate::proc::parent_death(&mut command);
        let mut process = command.spawn().map_err(|error| error.to_string())?;
        let child = Self::new(process.id(), process.stdin.take(), process.stdout.take(), process.stderr.take(), Vec::new()).map_err(|error| {
            let _ = process.kill().and_then(|()| process.wait());
            error.to_string()
        })?;
        #[cfg(not(target_os = "linux"))]
        {
            crate::proc::attach(child.pid).map_err(|error| error.to_string())?;
            crate::tools::reaper::track(child.pid, true);
        }
        Ok(child)
    }

    // Adopt owned pipes, including the bridge's descriptors kept across exec.
    pub(super) fn new(
        pid: u32,
        stdin: Option<std::process::ChildStdin>,
        stdout: Option<std::process::ChildStdout>,
        stderr: Option<std::process::ChildStderr>,
        tail: Vec<u8>,
    ) -> std::io::Result<Self> {
        let missing = || std::io::Error::other("a pipe is missing");
        let (stdin, stdout, stderr) = (ChildStdin::from_std(stdin.ok_or_else(missing)?)?, ChildStdout::from_std(stdout.ok_or_else(missing)?)?, stderr.ok_or_else(missing)?);
        #[cfg(unix)]
        let stderr_fd = std::os::fd::AsFd::as_fd(&stderr).try_clone_to_owned()?;
        let mut pipe = tokio::process::ChildStderr::from_std(stderr)?;
        let stderr = Arc::new(Mutex::new(tail));
        let tail = stderr.clone();
        let stderr_reader = Some(tokio::spawn(async move {
            let mut chunk = [0u8; STDERR_TAIL];
            while let Ok(read) = pipe.read(&mut chunk).await
                && read > 0
            {
                let mut tail = tail.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                tail.extend_from_slice(&chunk[..read]);
                let excess = tail.len().saturating_sub(STDERR_TAIL);
                tail.drain(..excess);
            }
        }));
        Ok(Self {
            pid,
            status: None,
            stdin: Some(stdin),
            stdout: BufReader::new(stdout),
            line: Vec::new(),
            stderr,
            stderr_reader,
            #[cfg(unix)]
            stderr_fd,
        })
    }

    pub(super) async fn wait(&mut self) -> std::io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status);
        }
        let status = crate::proc::wait(self.pid).await?;
        self.status = Some(status);
        Ok(status)
    }

    pub(super) fn signal(&self, signal: i32) {
        if self.status.is_none()
            && let Some(process) = crate::proc::Process::open(self.pid as i32)
        {
            process.signal(signal);
        }
    }

    // Partial bytes survive a cancelled select. ACP supplies a strict frame budget; the bridge
    // keeps its old unlimited, lossy UTF-8 reader, including a final line without a newline.
    pub(super) async fn next_line(&mut self, max_bytes: Option<usize>) -> Option<String> {
        loop {
            let bytes = self.stdout.fill_buf().await.ok()?;
            let ended = bytes.is_empty();
            if ended && (max_bytes.is_some() || self.line.is_empty()) {
                return None;
            }
            let end = bytes.iter().position(|byte| *byte == b'\n').map(|at| at + 1);
            let count = end.unwrap_or(bytes.len());
            if max_bytes.is_some_and(|limit| self.line.len().saturating_add(count) > limit) {
                return None;
            }
            self.line.extend_from_slice(&bytes[..count]);
            self.stdout.consume(count);
            if end.is_some() || ended {
                let line = std::mem::take(&mut self.line);
                return if max_bytes.is_some() { String::from_utf8(line.trim_ascii_end().to_vec()).ok() } else { Some(String::from_utf8_lossy(&line).trim_end().to_string()) };
            }
        }
    }

    pub(super) async fn write(&mut self, message: &Value) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("child input is closed")?;
        stdin.write_all(format!("{message}\n").as_bytes()).await.map_err(|error| error.to_string())
    }

    // The bridge historically notices a failed write through EOF on stdout. ACP uses write's
    // error directly, since a prompt whose delivery is uncertain must not be retried.
    pub(super) async fn send(&mut self, message: &Value) {
        if self.write(message).await.is_err() {
            self.stdin = None;
        }
    }

    pub(super) async fn close(&mut self) -> Option<ExitStatus> {
        self.stdin = None;
        let mut status = timeout(CLOSE_GRACE, self.wait()).await.ok().and_then(Result::ok);
        if status.is_none() {
            self.signal(crate::proc::TERM);
            status = timeout(TERM_GRACE, self.wait()).await.ok().and_then(Result::ok);
        }
        if status.is_none() {
            self.signal(crate::proc::KILL);
            let _ = self.wait().await;
        }
        status
    }

    pub(super) async fn stderr_tail(&mut self) -> String {
        if let Some(reader) = &mut self.stderr_reader {
            let _ = timeout(CLOSE_GRACE, &mut *reader).await;
            reader.abort();
            self.stderr_reader = None;
        }
        self.stderr_so_far()
    }

    pub(super) fn stderr_so_far(&self) -> String {
        String::from_utf8_lossy(&self.stderr.lock().unwrap_or_else(std::sync::PoisonError::into_inner)).trim().to_string()
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        #[cfg(not(target_os = "linux"))]
        {
            crate::proc::signal_group(self.pid, crate::proc::KILL);
            crate::proc::release(self.pid);
            crate::tools::reaper::track(self.pid, false);
        }
        if self.status.is_none() {
            self.signal(crate::proc::KILL);
            #[cfg(unix)]
            {
                let pid = self.pid as libc::pid_t;
                // SAFETY: only this owner reaps its child.
                std::thread::spawn(move || unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) });
            }
        }
        if let Some(reader) = &self.stderr_reader {
            reader.abort();
        }
    }
}
