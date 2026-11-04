// Background jobs: bash commands still running at the soft limit, and the message each one sends
// into the session when it exits.
//
// This is piccosta's longcmd (extensions/longcmd.ts, FSL-1.1-ALv2, Synoros), which gave pi's bash
// DeepSeek Harness's long-command behaviour (notes/research-2026-10-15.md, "Long-running commands"):
// - With an explicit `timeout`, a command is killed at that limit, as in pi.
// - Without one, a command that ends within the soft limit (DEFAULT_SOFT_LIMIT_SECONDS) returns exactly what
//   pi's bash returns.
// - One still running at the soft limit keeps running. Its whole output (the part so far and
//   everything after) goes to a log, and the call returns the output tail, job id, pid and log
//   path, and says a message will come when it exits.
// - When it exits, a message with the exit code, duration, output tail and log path goes into the
//   session: at the next turn boundary when the agent is busy, as a new turn when it's idle.
// - In print and json mode the run is held open after the model's last turn until a job exits,
//   so its message becomes a new turn before piccosta exits.
// - Every command the session starts is listed here (shells) until it's reaped, foreground or
//   background: at session end its runtime stops them all with their whole process groups, and
//   their logs say so; when piccosta dies without stopping them, its watcher does (reaper.rs).
//   /restart is not an end: its jobs are handed to the image it execs (hand_over, adopt), which
//   sends their messages when they exit. An abort during the model's turn doesn't kill a job that
//   is already in the background: the turn that started it is over.
// - An interrupt never kills a command: Esc, an rpc abort, ACP's cancel and a steer message move
//   the turn's running commands to the background (runtime.rs). Closing or replacing a session
//   is an end, and kills them.
//
// No new tools: the model stops a job with `kill -- -<pid>` through bash and reads the log with
// its read or bash tools.
//
// Where logs go: `<session cwd>/.piccosta-jobs/<session id prefix>-<job id>.log`, with a `.gitignore`
// of `*` in `.piccosta-jobs` so git never sees them. Not /tmp: some machines clear tmp, bench sandboxes
// give each run a private one, and a job log has to be readable by the same model with the same
// tools, which always see the working directory. PICCOSTA_LONGCMD_DIR moves it (a relative path
// is taken from the session cwd). Job logs stay after the session, so a human or a later session
// can read them; bash's spill files (full output of a truncated command) are removed when the
// session ends.
//
// Fixed from longcmd: the session id prefix only changes every 65.5 s (session ids are UUIDv7)
// and job ids restart per process, so two sessions in one cwd overwrote each other's `-1.log`.
// Logs are now created only where no file exists, moving on to the next job number if one does.
//
// A job's log is the command's spill file, renamed: a command that moves keeps every byte it
// printed, with nothing to copy and no second file to make. One that never spilled spills now.

use crate::message::{CustomMessage, TextOrBlocks};
use std::collections::HashMap;
use std::os::fd::AsRawFd;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};
use tokio_util::sync::CancellationToken;

// The soft limit before a command moves to the background: long enough for the quick commands
// (ls, cat, rg, git status) to answer in place, short enough that nothing holds a session up.
// $PICCOSTA_LONGCMD_SOFT_S changes it; 0 sends every command to the background.
const DEFAULT_SOFT_LIMIT_SECONDS: f64 = 2.0;
pub const SOFT_LIMIT_ENV: &str = "PICCOSTA_LONGCMD_SOFT_S";
const DEFAULT_JOB_DIR: &str = ".piccosta-jobs";
const JOB_DIR_ENV: &str = "PICCOSTA_LONGCMD_DIR";
// Unix's null device: the log of a job whose output had nowhere to go (Jobs::start).
const NO_LOG: &str = "/dev/null";
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

// Signal a command's process group, or the process alone if the group is already gone
// (shell.ts:205-216). The shell leads its own group, so the group id is its pid.
pub fn signal_group(pid: u32, signal: i32) {
    let Ok(pid) = i32::try_from(pid) else { return };
    // SAFETY: kill() takes plain integers; a stale pid at worst fails with ESRCH.
    unsafe {
        if libc::kill(-pid, signal) != 0 {
            libc::kill(pid, signal);
        }
    }
}

// Input: an env var holding seconds. Output: that duration, zero included, or None when it's unset
// or not a number of seconds.
fn seconds_from_env(name: &str) -> Option<Duration> {
    let seconds = std::env::var(name).ok().and_then(|raw| raw.trim().parse::<f64>().ok()).filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
    seconds.map(Duration::from_secs_f64)
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

// A shell the session started, listed until it's reaped: its pid (its process group's), whether
// it's piccosta's own (a subagent: SIGTERM, then SIGKILL at `kill_at`), its output pipe (held, so
// the descriptor lives while it's listed and can go across /restart's exec), and whether it went
// to the next image, which reaps it from then on.
pub struct Shell {
    pub pid: u32,
    pub own: bool,
    pub kill_at: Option<Instant>,
    pub pipe: Arc<tokio::net::unix::pipe::Receiver>,
    pub handed: bool,
}

// One session's commands and background jobs, shared by every bash call.
pub struct Jobs {
    pub shells: Mutex<Vec<Shell>>,
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
// model (runtime.rs).
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
            shells: Mutex::new(Vec::new()),
            soft_limit: seconds_from_env(SOFT_LIMIT_ENV).unwrap_or(Duration::from_secs_f64(DEFAULT_SOFT_LIMIT_SECONDS)),
            // A hold of 0 would kill every job the moment the model is done, so it counts as unset.
            hold: seconds_from_env(HOLD_ENV).filter(|hold| !hold.is_zero()),
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

    pub fn shells(&self) -> std::sync::MutexGuard<'_, Vec<Shell>> {
        self.shells.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    // Stop a shell's process group, or the shell alone if the group is already gone: SIGTERM for
    // one of piccosta's own, which its watcher turns to SIGKILL at `kill_at` (bash.rs watch),
    // SIGKILL for any other. Repeated stops share the first deadline: a background job's stop and
    // its watcher can race.
    pub fn kill_group(&self, pid: u32) {
        let mut shells = self.shells();
        match shells.iter_mut().find(|shell| shell.pid == pid && shell.own) {
            Some(shell) if shell.kill_at.is_some() => {}
            Some(shell) => {
                shell.kill_at = Some(Instant::now() + super::bash::SUBAGENT_STOP_GRACE);
                signal_group(pid, libc::SIGTERM);
            }
            None => signal_group(pid, libc::SIGKILL),
        }
    }

    // Kill one job's process group, and say why in its log and message.
    fn kill_job(&self, job: &RunningJob, reason: &str, silent: bool) {
        job.kill_reason.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).get_or_insert_with(|| reason.to_string());
        if silent {
            job.silent.store(true, Ordering::SeqCst);
        }
        job.kill.cancel();
        self.kill_group(job.pid);
    }

    // Kill every running job. Silent kills send no message: used when nothing is left to tell.
    pub fn kill_all(&self, reason: &str, silent: bool) {
        for job in self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).values() {
            self.kill_job(job, reason, silent);
        }
    }

    // Session end: kill running jobs without messages, then every command still running in the
    // foreground, and remove this session's spill files.
    pub fn shutdown(&self, reason: &str) {
        self.shutting_down.store(true, Ordering::SeqCst);
        self.kill_all(&format!("session shutdown ({reason})"), true);
        let left: Vec<u32> = self.shells().iter().filter(|shell| !shell.handed).map(|shell| shell.pid).collect();
        left.into_iter().for_each(|pid| self.kill_group(pid));
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

    // Input: the session tag, the command, its pid and kill token, and its spill file (path and
    // open file) holding everything it printed so far. Output: the new job, its log that file under
    // the job's name. A hard link to the new name, then removing the old, is a rename that never
    // replaces another session's log: a taken name moves on to the next job number. If the link
    // can't be made, the log keeps the spill's name, and the session's end no longer removes it.
    // With no spill file at all (none could be made in the job directory) the log is the null
    // device: the job still runs, is listed, survives /restart and sends its message with the
    // output tail, and the log path it gives says that nothing more was kept.
    pub fn start(&self, session_tag: &str, command: &str, pid: u32, kill: CancellationToken, started: Instant, spill: Option<(PathBuf, std::fs::File)>) -> Job {
        let Some((spill_path, log)) = spill else {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            return self.list(id, pid, command, started, PathBuf::from(NO_LOG), std::fs::File::options().append(true).open(NO_LOG).ok(), kill);
        };
        // The spill name leaves the session's cleanup list only once it's gone or is the log itself;
        // a name the unlink couldn't remove is still cleaned up when the session ends.
        let untrack = |spill_path: &PathBuf| self.spill_files.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).retain(|path| path != spill_path);
        loop {
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let path = spill_path.with_file_name(format!("{session_tag}-{id}.log"));
            match std::fs::hard_link(&spill_path, &path) {
                Ok(()) => {
                    if std::fs::remove_file(&spill_path).is_ok() {
                        untrack(&spill_path);
                    }
                    return self.list(id, pid, command, started, path, Some(log), kill);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(_) => {
                    untrack(&spill_path);
                    return self.list(id, pid, command, started, spill_path, Some(log), kill);
                }
            }
        }
    }

    // List a job as running. Output: its handle, holding its open log.
    #[allow(clippy::too_many_arguments)]
    fn list(&self, id: u64, pid: u32, command: &str, started: Instant, log_path: PathBuf, log: Option<std::fs::File>, kill: CancellationToken) -> Job {
        let kill_reason = Arc::new(Mutex::new(None));
        let silent = Arc::new(AtomicBool::new(false));
        let running = RunningJob { pid, command: command.to_string(), started, log_path: log_path.clone(), kill, kill_reason: kill_reason.clone(), silent: silent.clone() };
        self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).insert(id, running);
        Job { id, pid, log_path, log, kill_reason, silent, started }
    }

    // /restart: take the running jobs out of this session, so its end doesn't kill them, and hand
    // them to the next image: from now on this image neither reaps nor kills their shells, whose
    // pipes go across the exec. Output: one record each (a job whose shell already ended isn't one).
    pub fn hand_over(&self) -> Vec<HandedJob> {
        let mut handed = Vec::new();
        let mut shells = self.shells();
        self.running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).retain(|&id, job| {
            let Some(shell) = shells.iter_mut().find(|shell| shell.pid == job.pid && !shell.handed) else { return true };
            shell.handed = true;
            super::reaper::keep_across_exec(shell.pipe.as_raw_fd());
            let elapsed_ms = job.started.elapsed().as_millis() as u64;
            handed.push(HandedJob { id, pid: job.pid, own: shell.own, command: job.command.clone(), elapsed_ms, log_path: job.log_path.clone(), pipe: shell.pipe.as_raw_fd() });
            false
        });
        handed
    }

    // The next image takes a handed job back: listed again under its id, its log opened to go on.
    pub fn adopt(&self, handed: &HandedJob, kill: CancellationToken) -> std::io::Result<Job> {
        let log = std::fs::OpenOptions::new().append(true).open(&handed.log_path)?;
        self.next_id.fetch_max(handed.id + 1, Ordering::SeqCst);
        let started = Instant::now().checked_sub(Duration::from_millis(handed.elapsed_ms)).unwrap_or_else(Instant::now);
        Ok(self.list(handed.id, handed.pid, &handed.command, started, handed.log_path.clone(), Some(log), kill))
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
