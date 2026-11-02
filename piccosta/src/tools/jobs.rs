// Background jobs: bash commands still running at the soft limit, and the message each one sends
// into the session when it exits.
//
// This is piccosta's longcmd (extensions/longcmd.ts, FSL-1.1-ALv2, Synoros), which gave pi's bash
// DeepSeek Harness's long-command behaviour (notes/research-2026-10-15.md, "Long-running commands"):
// - With an explicit `timeout`, a command is killed at that limit, as in pi.
// - Without one, a command that ends within the soft limit (60 s by default) returns exactly what
//   pi's bash returns.
// - One still running at the soft limit keeps running. Its whole output (the part so far and
//   everything after) goes to a log, and the call returns the output tail, job id, pid and log
//   path, and says a message will come when it exits.
// - When it exits, a message with the exit code, duration, output tail and log path goes into the
//   session: at the next turn boundary when the agent is busy, as a new turn when it's idle.
// - In print and json mode the run is held open after the model's last turn until a job exits,
//   so its message becomes a new turn before piccosta exits.
// - At session end and when piccosta is killed by a signal, running jobs die with their whole
//   process group, and their logs say so. /restart is not an end: its jobs are handed to the
//   image it execs (hand_over, adopt), which sends their messages when they exit. An abort during the model's turn doesn't kill a job
//   that is already in the background: the turn that started it is over.
//
// No new tools: the model stops a job with `kill -- -<pid>` through bash and reads the log with
// its read or bash tools.
//
// Where logs go: `<session cwd>/.pi-jobs/<session id prefix>-<job id>.log`, with a `.gitignore`
// of `*` in `.pi-jobs` so git never sees them. Not /tmp: some machines clear tmp, bench sandboxes
// give each run a private one, and a job log has to be readable by the same model with the same
// tools, which always see the working directory. PICCOSTA_LONGCMD_DIR moves it (a relative path
// is taken from the session cwd). Job logs stay after the session, so a human or a later session
// can read them; bash's spill files (full output of a truncated command) are removed when the
// session ends.
//
// Fixed from longcmd: the session id prefix only changes every 65.5 s (session ids are UUIDv7)
// and job ids restart per process, so two sessions in one cwd overwrote each other's `-1.log`.
// Logs are now created only where no file exists, moving on to the next job number if one does.

use crate::message::{CustomMessage, TextOrBlocks};
use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;

// The soft limit before a command moves to the background. 60 s is dsh-base's default bash
// timeout (dsh:packages/bundle/base/cordis.patch.yml:249-253).
const DEFAULT_SOFT_LIMIT_SECONDS: f64 = 60.0;
const SOFT_LIMIT_ENV: &str = "PICCOSTA_LONGCMD_SOFT_S";
const DEFAULT_JOB_DIR: &str = ".pi-jobs";
const JOB_DIR_ENV: &str = "PICCOSTA_LONGCMD_DIR";
// Optional cap, in seconds, on how long print and json mode wait for running jobs after the model
// ends its turn. Unset means wait as long as a job runs, which is what a blocking bash would have
// done. When the cap is hit the jobs are killed and the model is told.
pub const HOLD_ENV: &str = "PICCOSTA_LONGCMD_HOLD_S";
// "0" turns off the "did you mean" lines for missing paths (pathhint.rs). The piccosta launcher
// has always turned them on, so on is the default.
const PATH_HINTS_ENV: &str = "PICCOSTA_LONGCMD_PATHHINTS";
// The last two of pi's output windows are kept for the tails we show; truncate_tail then cuts
// that to pi's own limits.
pub const TAIL_KEEP_BYTES: usize = 2 * super::truncate::DEFAULT_MAX_BYTES;
// The finish message repeats the command so the model knows which job it is; a multi-line script
// shows its first line, cut at this many characters.
const COMMAND_PREVIEW_CHARS: usize = 200;
// customType of the finish message in the session.
pub const NOTICE_TYPE: &str = "longcmd";

// Input: an env var holding seconds. Output: that duration, or the fallback when it's unset or
// not a positive number.
fn seconds_from_env(name: &str, fallback: Option<f64>) -> Option<Duration> {
    let seconds = std::env::var(name).ok().and_then(|raw| raw.trim().parse::<f64>().ok()).filter(|seconds| seconds.is_finite() && *seconds > 0.0);
    seconds.or(fallback).map(Duration::from_secs_f64)
}

// "61.2 s": tenths, as longcmd printed durations.
pub fn seconds_text(duration: Duration) -> String {
    let tenths = (duration.as_millis() as f64 / 100.0).round() / 10.0;
    format!("{} s", crate::js::to_fixed(tenths, 1))
}

pub fn command_preview(command: &str) -> String {
    let trimmed = crate::js::trim(command);
    let first_line = trimmed.split('\n').next().unwrap_or_default();
    let more = crate::js::utf16_len(first_line) > COMMAND_PREVIEW_CHARS || trimmed.contains('\n');
    let cut: String = first_line.chars().take(COMMAND_PREVIEW_CHARS).collect();
    format!("{cut}{}", if more { " ..." } else { "" })
}

// A rolling window of a command's last raw bytes, and how many there were in all.
#[derive(Default)]
pub struct Tail {
    bytes: Vec<u8>,
    total: usize,
}

impl Tail {
    pub fn push(&mut self, data: &[u8]) {
        self.total += data.len();
        self.bytes.extend_from_slice(data);
        if self.bytes.len() > TAIL_KEEP_BYTES * 2 {
            let cut = self.bytes.len() - TAIL_KEEP_BYTES;
            self.bytes.drain(..cut);
        }
    }

    // The tail cut to pi's bash limits, and whether anything was left out. When the window
    // already dropped bytes its first line may be partial, so it goes.
    pub fn text(&self) -> (String, bool) {
        let mut window = &self.bytes[self.bytes.len().saturating_sub(TAIL_KEEP_BYTES)..];
        let dropped = self.total > window.len();
        if dropped && let Some(newline) = window.iter().position(|byte| *byte == b'\n') {
            window = &window[newline + 1..];
        }
        let text = String::from_utf8_lossy(window);
        let truncation = super::truncate::truncate_tail(&text, super::truncate::DEFAULT_MAX_LINES, super::truncate::DEFAULT_MAX_BYTES);
        (truncation.content, dropped || truncation.truncated)
    }

    fn ends_with_newline(&self) -> bool {
        self.bytes.last() == Some(&b'\n')
    }

    // A job taken back after /restart: its tail is the end of its log, which holds all it printed.
    pub fn from_log(path: &Path) -> Tail {
        let Ok(mut log) = std::fs::File::open(path) else { return Tail::default() };
        let total = log.metadata().map_or(0, |metadata| metadata.len() as usize);
        let mut bytes = Vec::new();
        let _ = log.seek(std::io::SeekFrom::Start(total.saturating_sub(TAIL_KEEP_BYTES) as u64)).and_then(|_| log.read_to_end(&mut bytes));
        Tail { bytes, total }
    }
}

// A running job as /restart hands it to the next image: enough to watch it again and send its
// message. `pipe` is the descriptor of its output pipe, kept open across the exec.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct HandedJob {
    pub id: u64,
    pub pid: u32,
    pub own: bool,
    pub command: String,
    pub elapsed_ms: u64,
    pub log_path: PathBuf,
    pub pipe: i32,
}

struct RunningJob {
    pid: u32,
    // What /ps shows of it.
    command: String,
    started: Instant,
    log_path: PathBuf,
    kill: CancellationToken,
    kill_reason: Arc<Mutex<Option<String>>>,
    silent: Arc<AtomicBool>,
}

// One session's background jobs, shared by every bash call.
pub struct Jobs {
    pub soft_limit: Duration,
    pub hold: Option<Duration>,
    pub path_hints: bool,
    job_dir_override: Option<String>,
    next_id: AtomicU64,
    running: Mutex<HashMap<u64, RunningJob>>,
    // Finish messages go here; the session decides when they reach the model.
    notices: mpsc::UnboundedSender<CustomMessage>,
    exited: Notify,
    shutting_down: AtomicBool,
    spill_files: Mutex<Vec<PathBuf>>,
    // Foreground commands that could move to the background, and the signal that moves them all
    // there at once, before the soft limit: the user asked for their message to be read now.
    detachable: AtomicUsize,
    pub detach: Notify,
}

// Counts a foreground command as one that can move to the background, until it's dropped.
pub struct Detachable<'a>(&'a Jobs);

impl Drop for Detachable<'_> {
    fn drop(&mut self) {
        self.0.detachable.fetch_sub(1, Ordering::SeqCst);
    }
}

// The reason /stop kills jobs with; their messages wait for the next prompt instead of waking the
// model (Host::spawn).
pub const STOPPED_BY_USER: &str = "stopped by /stop";

pub fn stopped_by_user(notice: &CustomMessage) -> bool {
    notice.details.as_ref().and_then(|details| details.get("killed")).and_then(serde_json::Value::as_str) == Some(STOPPED_BY_USER)
}

// What a background job's command came to.
pub enum JobEnd {
    Exited(i32),
    Failed(String),
}

impl Jobs {
    // Input: where finish messages go. Settings come from the environment (see the constants).
    pub fn new(notices: mpsc::UnboundedSender<CustomMessage>) -> Jobs {
        Jobs {
            soft_limit: seconds_from_env(SOFT_LIMIT_ENV, Some(DEFAULT_SOFT_LIMIT_SECONDS)).unwrap_or(Duration::from_secs(60)),
            hold: seconds_from_env(HOLD_ENV, None),
            path_hints: std::env::var(PATH_HINTS_ENV).map_or(true, |value| value != "0"),
            job_dir_override: std::env::var(JOB_DIR_ENV).ok().filter(|dir| !dir.is_empty()),
            next_id: AtomicU64::new(1),
            running: Mutex::new(HashMap::new()),
            notices,
            exited: Notify::new(),
            shutting_down: AtomicBool::new(false),
            spill_files: Mutex::new(Vec::new()),
            detachable: AtomicUsize::new(0),
            detach: Notify::new(),
        }
    }

    pub fn detachable(&self) -> Detachable<'_> {
        self.detachable.fetch_add(1, Ordering::SeqCst);
        Detachable(self)
    }

    // Move every foreground command that can go to the background there now. Output: whether
    // there was one, so the caller knows the turn will go on to what's queued without an abort.
    pub fn detach_all(&self) -> bool {
        let any = self.detachable.load(Ordering::SeqCst) > 0;
        self.detach.notify_waiters();
        any
    }

    // The directory a cwd's logs go in, made on first use with a `*` .gitignore.
    pub fn job_dir(&self, cwd: &Path) -> PathBuf {
        let dir = match &self.job_dir_override {
            Some(dir) if Path::new(dir).is_absolute() => PathBuf::from(dir),
            Some(dir) => cwd.join(dir),
            None => cwd.join(DEFAULT_JOB_DIR),
        };
        let _ = std::fs::create_dir_all(&dir);
        let ignore = dir.join(".gitignore");
        if !ignore.exists() {
            let _ = std::fs::write(&ignore, "*\n");
        }
        dir
    }

    // A fresh path for a bash spill file in the cwd's job directory, removed at session end.
    pub fn spill_path(&self, cwd: &Path) -> PathBuf {
        let name = format!("bash-{}.log", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        let path = self.job_dir(cwd).join(name);
        self.spill_files.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(path.clone());
        path
    }

    pub fn running_count(&self) -> usize {
        self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).len()
    }

    // Kill one job's process group, and say why in its log and message.
    fn kill_job(job: &RunningJob, reason: &str, silent: bool) {
        job.kill_reason.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get_or_insert_with(|| reason.to_string());
        if silent {
            job.silent.store(true, Ordering::SeqCst);
        }
        job.kill.cancel();
        super::bash::kill_group(job.pid);
    }

    // Kill every running job. Silent kills send no message: used when nothing is left to tell.
    pub fn kill_all(&self, reason: &str, silent: bool) {
        for job in self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).values() {
            Jobs::kill_job(job, reason, silent);
        }
    }

    // Session end: kill running jobs without messages and remove this session's spill files.
    pub fn shutdown(&self, reason: &str) {
        self.shutting_down.store(true, Ordering::SeqCst);
        self.kill_all(&format!("session shutdown ({reason})"), true);
        for path in self.spill_files.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).drain(..) {
            let _ = std::fs::remove_file(path);
        }
        self.exited.notify_waiters();
    }

    // The running jobs, one line each: id, pid, how long, the command's first line and the log.
    pub fn describe(&self) -> Vec<String> {
        let running = self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut ids: Vec<&u64> = running.keys().collect();
        ids.sort();
        ids.into_iter()
            .map(|id| {
                let job = &running[id];
                format!("job {id} · pid {} · {} · {} · {}", job.pid, seconds_text(job.started.elapsed()), command_preview(&job.command), job.log_path.display())
            })
            .collect()
    }

    // Input: the session tag, the cwd, the command, its pid and kill token, and its output so far.
    // Output: the new job and its open log, with the output so far already written.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &self,
        session_tag: &str,
        cwd: &Path,
        command: &str,
        pid: u32,
        kill: CancellationToken,
        started: Instant,
        earlier_output: super::output::RawOutput<'_>,
    ) -> std::io::Result<Job> {
        let dir = self.job_dir(cwd);
        // Next free job number for this tag: create_new fails on an existing file.
        let (id, path, mut log) = loop {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let path = dir.join(format!("{session_tag}-{id}.log"));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => break (id, path, file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        };
        match earlier_output {
            super::output::RawOutput::Memory(bytes) => log.write_all(bytes)?,
            super::output::RawOutput::File(spill) => {
                let mut source = std::fs::File::open(spill)?;
                std::io::copy(&mut source, &mut log)?;
            }
        }
        Ok(self.list(id, pid, command, started, path, log, kill))
    }

    // List a job as running. Output: its handle, holding its open log.
    #[allow(clippy::too_many_arguments)]
    fn list(&self, id: u64, pid: u32, command: &str, started: Instant, log_path: PathBuf, log: std::fs::File, kill: CancellationToken) -> Job {
        let kill_reason = Arc::new(Mutex::new(None));
        let silent = Arc::new(AtomicBool::new(false));
        let running = RunningJob { pid, command: command.to_string(), started, log_path: log_path.clone(), kill, kill_reason: kill_reason.clone(), silent: silent.clone() };
        self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(id, running);
        Job { id, pid, log_path, log: Some(log), kill_reason, silent, started }
    }

    // /restart: take the running jobs out of this session, so its end doesn't kill them, and hand
    // them to the next image. Output: one record each (a job whose shell already ended isn't one).
    pub fn hand_over(&self) -> Vec<HandedJob> {
        let mut handed = Vec::new();
        self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).retain(|&id, job| {
            let Some((own, pipe)) = super::bash::hand_over(job.pid) else { return true };
            let elapsed_ms = job.started.elapsed().as_millis() as u64;
            handed.push(HandedJob { id, pid: job.pid, own, command: job.command.clone(), elapsed_ms, log_path: job.log_path.clone(), pipe });
            false
        });
        handed
    }

    // The next image takes a handed job back: listed again under its id, its log opened to go on.
    pub fn adopt(&self, handed: &HandedJob, kill: CancellationToken) -> std::io::Result<Job> {
        let log = std::fs::OpenOptions::new().append(true).open(&handed.log_path)?;
        self.next_id.fetch_max(handed.id + 1, Ordering::SeqCst);
        let started = Instant::now().checked_sub(Duration::from_millis(handed.elapsed_ms)).unwrap_or_else(Instant::now);
        Ok(self.list(handed.id, handed.pid, &handed.command, started, handed.log_path.clone(), log, kill))
    }

    // Input: a job whose command ended, the command, its output tail and how it ended. Closes the
    // log and sends the finish message, unless the kill was silent or the session is over.
    pub fn finish(&self, mut job: Job, command: &str, tail: &Tail, end: JobEnd) {
        self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).remove(&job.id);
        let elapsed = job.started.elapsed();
        let duration = seconds_text(elapsed);
        let kill_reason = job.kill_reason.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone();
        let (status, exit_code) = match (&kill_reason, &end) {
            (Some(reason), _) => (format!("was killed ({reason}) after {duration}"), None),
            (None, JobEnd::Exited(code)) => (format!("exited with code {code} after {duration}"), Some(*code)),
            (None, JobEnd::Failed(error)) => (format!("failed after {duration}: {error}"), None),
        };
        // Lines we add start with "[longcmd]" so they can't be mistaken for the command's own.
        if let Some(mut log) = job.log.take() {
            let newline = if tail.total > 0 && !tail.ends_with_newline() { "\n" } else { "" };
            let trailer = kill_reason.as_ref().map(|reason| format!("killed: {reason}")).unwrap_or_else(|| status.clone());
            let _ = writeln!(log, "{newline}[longcmd] {trailer}");
        }
        self.exited.notify_waiters();
        if job.silent.load(Ordering::SeqCst) || self.shutting_down.load(Ordering::SeqCst) {
            return;
        }

        let (text, cut) = tail.text();
        let text = text.strip_suffix('\n').unwrap_or(&text);
        let lines = [
            format!("Background job {} (pid {}) {status}: {}", job.id, job.pid, command_preview(command)),
            if text.is_empty() { "(no output)".to_string() } else { text.to_string() },
            format!("[{}Full output: {}]", if cut { "Output above is the tail. " } else { "" }, job.log_path.display()),
        ];
        // Fields longcmd left undefined are absent, not null, as JSON.stringify writes them.
        let mut details = serde_json::Map::new();
        details.insert("jobId".into(), job.id.into());
        details.insert("pid".into(), job.pid.into());
        if let Some(code) = exit_code {
            details.insert("exitCode".into(), code.into());
        }
        if let Some(reason) = &kill_reason {
            details.insert("killed".into(), reason.clone().into());
        }
        details.insert("durationMs".into(), (elapsed.as_millis() as u64).into());
        details.insert("logPath".into(), job.log_path.display().to_string().into());
        let _ = self.notices.send(CustomMessage {
            custom_type: NOTICE_TYPE.to_string(),
            content: TextOrBlocks::Text(lines.join("\n")),
            display: true,
            details: Some(serde_json::Value::Object(details)),
            timestamp: crate::clock::now_ms(),
        });
    }
}

// A job's own handle, held by the task that watches its command.
pub struct Job {
    pub id: u64,
    pub pid: u32,
    pub log_path: PathBuf,
    log: Option<std::fs::File>,
    kill_reason: Arc<Mutex<Option<String>>>,
    silent: Arc<AtomicBool>,
    // When the command began, for the duration in its message.
    pub started: Instant,
}

impl Job {
    pub fn write(&mut self, data: &[u8]) {
        if let Some(log) = self.log.as_mut() {
            let _ = log.write_all(data);
        }
    }
}
