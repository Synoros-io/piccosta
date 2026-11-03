// The bash tool: run a command in the session's directory and return what it printed.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/tools/bash.ts and
// the shell helpers in utils/shell.ts and utils/child-process.ts, with piccosta's longcmd (jobs.rs)
// and path hints (pathhint.rs) built in. pi's own behaviour, kept:
// - The shell is settings' shellPath, else /bin/bash, else bash on PATH, else sh, run as
//   `<shell> -c "<shellCommandPrefix>\n<command>"`, in its own session (setsid), stdin closed,
//   stdout and stderr on one pipe, read as one stream.
// - PICCOSTA_SESSION_ID, PICCOSTA_SESSION_FILE, PICCOSTA_PROVIDER, PICCOSTA_MODEL and
//   PICCOSTA_REASONING_LEVEL describe the session; the agent dir's bin/ goes on the front of PATH.
// - A timeout or an abort SIGKILLs a bash command's whole process group. Subagents get SIGTERM
//   first so their own shell groups are cleaned up. After the shell exits, output is
//   still read until the pipe closes or nothing arrives for 100 ms, so a descendant that keeps
//   the pipe open can't hang the call (pi#5303).
// - A shell is waited on through a pidfd and reaped by hand rather than through tokio, so one
//   in the background can be handed to the image /restart execs (hand_over, adopt): it stays
//   this process's child, and its pipe goes across the exec.
// - The model sees the last 2000 lines or 50 KB, raw (ANSI codes and CRs included), with pi's
//   notice saying what was cut and where the full output is.
// - A non-zero exit is an error result whose text ends "Command exited with code N".

use super::jobs::{self, HandedJob, JobEnd, Tail};
use super::output::OutputAccumulator;
use super::truncate::{self, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};
use super::{pathhint, Tool, ToolContext, ToolFuture, ToolOutput, ToolUpdate};
use crate::message::{Content, ToolDefinition};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::unix::pipe::Receiver;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct BashTool;

#[derive(Deserialize)]
struct Arguments {
    command: String,
    timeout: Option<f64>,
    // A command piccosta wrote itself (run_own), which skips the shell prefix.
    #[serde(skip)]
    own: bool,
    // Straight to the background, without waiting the soft limit (run_in_background).
    #[serde(skip)]
    background: bool,
}

// setTimeout's limit, which pi checks timeouts against (bash.ts:20).
const MAX_TIMEOUT_MS: f64 = 2_147_483_647.0;
// How often the UI gets the running output (renderers/bash.ts:19).
const UPDATE_THROTTLE: Duration = Duration::from_millis(100);
// How long to keep reading after the shell exits while output still trickles in (child-process.ts:16).
const EXIT_STDIO_GRACE: Duration = Duration::from_millis(100);
// Print mode closes Claude before killing its shells. Claude's two CLOSE_GRACE waits can take
// four seconds (provider/claude/mod.rs); leave another second for shell cleanup and scheduling.
pub(super) const SUBAGENT_STOP_GRACE: Duration = Duration::from_secs(5);
const READ_CHUNK_BYTES: usize = 64 * 1024;
// The environment variables that describe the session to commands (bash.ts:186-212, under
// piccosta's own names).
const SESSION_VARIABLES: [&str; 5] = [super::subagent::SESSION_ENV, "PICCOSTA_SESSION_FILE", "PICCOSTA_PROVIDER", "PICCOSTA_MODEL", "PICCOSTA_REASONING_LEVEL"];

enum Event {
    Data(Vec<u8>),
    Done(Outcome),
}

enum Outcome {
    Exited(i32),
    Aborted,
    TimedOut,
    Failed(String),
}

impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".into(),
            description: format!(
                "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.",
                DEFAULT_MAX_BYTES / 1024
            ),
            parameters: json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"}
                }
            }),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        Some("run a shell command in the project directory")
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        &[
            "Commands see this session's id, file, provider, model and thinking level as PICCOSTA_SESSION_ID, PICCOSTA_SESSION_FILE, PICCOSTA_PROVIDER, PICCOSTA_MODEL and PICCOSTA_REASONING_LEVEL",
            "A command that runs long moves to the background on its own, and its result comes to you as a message when it ends: don't wait on it, sleep or poll. A timeout only stops a command that must not run longer; leave it out otherwise.",
        ]
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let arguments: Arguments = serde_json::from_value(Value::Object(arguments)).map_err(|error| error.to_string())?;
            let result = run(&arguments, context).await;
            let succeeded = matches!(&result, Ok(Run::Finished(output) | Run::Background(output)) if !output.is_error);
            if succeeded {
                context.shared.seen.record_shell_reads(&arguments.command, &context.cwd);
            }
            // A command that came back within its time gets "did you mean" lines when its output
            // says a path doesn't exist. One moved to the background doesn't: its message is about
            // the job, not a path the model can retry right away.
            let hints = |text: &str| if context.shared.jobs.path_hints { pathhint::hints_for(text, &arguments.command, &context.cwd) } else { Vec::new() };
            match result {
                Ok(Run::Finished(mut output)) => {
                    if let Some(Content::Text { text, .. }) = output.content.first_mut() {
                        let found = hints(text);
                        if !found.is_empty() {
                            text.push('\n');
                            text.push_str(&found.join("\n"));
                        }
                    }
                    Ok(output)
                }
                Ok(Run::Background(output)) => Ok(output),
                Err(message) if context.cancel.is_cancelled() => Err(message),
                Err(message) => {
                    let found = hints(&message);
                    if found.is_empty() { Err(message) } else { Err(format!("{message}\n{}", found.join("\n"))) }
                }
            }
        })
    }
}

enum Run {
    Finished(ToolOutput),
    Background(ToolOutput),
}

// A command piccosta builds itself (a subagent's child), run the way the model's commands run but
// without the settings' shellCommandPrefix: that prefix keeps piccosta's own environment from
// commands the model writes (a `piccosta` it starts by hand), and this one is meant to have it.
pub async fn run_own(command: String, context: &ToolContext) -> Result<ToolOutput, String> {
    match run(&Arguments { command, timeout: None, own: true, background: false }, context).await? {
        Run::Finished(output) | Run::Background(output) => Ok(output),
    }
}

// The same, straight to the background: the manager's sessions, whose end comes back as a message.
pub async fn run_in_background(command: String, context: &ToolContext) -> Result<ToolOutput, String> {
    match run(&Arguments { command, timeout: None, own: true, background: true }, context).await? {
        Run::Finished(output) | Run::Background(output) => Ok(output),
    }
}

// Seconds from the model, checked as pi checks them (bash.ts:23-35).
fn resolve_timeout(timeout: Option<f64>) -> Result<Option<Duration>, String> {
    let Some(seconds) = timeout else { return Ok(None) };
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("Invalid timeout: must be a finite number of seconds".into());
    }
    if seconds * 1000.0 > MAX_TIMEOUT_MS {
        return Err(format!("Invalid timeout: maximum is {} seconds", crate::js::number_string(MAX_TIMEOUT_MS / 1000.0)));
    }
    Ok(Some(Duration::from_secs_f64(seconds)))
}

// The shell to run commands in (shell.ts:67-120).
fn shell_path(custom: Option<&str>) -> Result<PathBuf, String> {
    if let Some(custom) = custom {
        return if Path::new(custom).exists() { Ok(PathBuf::from(custom)) } else { Err(format!("Custom shell path not found: {custom}")) };
    }
    if Path::new("/bin/bash").exists() {
        return Ok(PathBuf::from("/bin/bash"));
    }
    let on_path = std::env::var_os("PATH").and_then(|path| std::env::split_paths(&path).map(|dir| dir.join("bash")).find(|candidate| candidate.is_file()));
    Ok(on_path.unwrap_or_else(|| PathBuf::from("sh")))
}

// Signal a command's process group, or the process alone if the group is already gone
// (shell.ts:205-216). The shell leads its own group, so the group id is its pid.
fn signal_group(pid: u32, signal: i32) {
    let Ok(pid) = i32::try_from(pid) else { return };
    // SAFETY: kill() takes plain integers; a stale pid at worst fails with ESRCH.
    unsafe {
        if libc::kill(-pid, signal) != 0 {
            libc::kill(pid, signal);
        }
    }
}

// Every shell this process started that hasn't exited. On a signal or quit they're killed with
// their groups, so no command outlives piccosta, whether or not its tool call saw the abort.
struct Running {
    pid: u32,
    own: bool,
    kill_at: Option<Instant>,
    // Its output pipe, held here so the descriptor stays open as long as the shell is listed.
    pipe: Arc<Receiver>,
    // Handed to the next image (/restart): this one no longer reaps it or kills it.
    handed: bool,
}

static RUNNING: std::sync::Mutex<Vec<Running>> = std::sync::Mutex::new(Vec::new());

fn running() -> std::sync::MutexGuard<'static, Vec<Running>> {
    RUNNING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Repeated stops share the first deadline: a background job's stop and its watcher can race.
pub fn kill_group(pid: u32) {
    if let Some(child) = running().iter_mut().find(|child| child.pid == pid && child.own) {
        if child.kill_at.is_none() {
            child.kill_at = Some(Instant::now() + SUBAGENT_STOP_GRACE);
            signal_group(pid, libc::SIGTERM);
        }
    } else {
        signal_group(pid, libc::SIGKILL);
    }
}

async fn stop_group(pid: u32) {
    kill_group(pid);
    let deadline = running().iter().find(|child| child.pid == pid).and_then(|child| child.kill_at);
    if let Some(deadline) = deadline {
        tokio::time::sleep_until(deadline.into()).await;
        signal_group(pid, libc::SIGKILL);
    }
}

// Stop every shell this image still owns: all but the ones handed to the next image.
pub fn kill_all_running() {
    let pids: Vec<_> = running().iter().filter(|child| !child.handed).map(|child| child.pid).collect();
    for pid in pids {
        kill_group(pid);
    }
    // This also runs just before process::exit, when an async escalation would never finish.
    let deadline = running().iter().filter(|child| !child.handed).filter_map(|child| child.kill_at).max();
    if let Some(deadline) = deadline {
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
    }
    running().retain(|child| {
        if !child.handed {
            signal_group(child.pid, libc::SIGKILL);
        }
        child.handed
    });
}

// Hand a background shell to the image /restart is about to exec: from now on this image neither
// reaps it nor kills it (its pipe is still read into the log until the exec). Output: whether it's
// a command of piccosta's own, and its pipe's descriptor; None when it isn't running.
pub fn hand_over(pid: u32) -> Option<(bool, i32)> {
    let mut running = running();
    let child = running.iter_mut().find(|child| child.pid == pid && !child.handed)?;
    child.handed = true;
    super::reaper::keep_across_exec(child.pipe.as_raw_fd());
    Some((child.own, child.pipe.as_raw_fd()))
}

// Take back a background job the image before this one handed over: its shell is still this
// process's child and its pipe came across the exec. It's watched and finished the way
// to_background does it, so its end reaches the session as if nothing happened. The reaper
// already lists its group.
pub fn adopt(shared: Arc<super::ToolShared>, handed: HandedJob) -> std::io::Result<()> {
    let pipe = Arc::new(Receiver::from_owned_fd(super::reaper::inherited(handed.pipe))?);
    let kill = CancellationToken::new();
    let job = shared.jobs.adopt(&handed, kill.clone())?;
    running().push(Running { pid: handed.pid, own: handed.own, kill_at: None, pipe: pipe.clone(), handed: false });
    let (event_sender, events) = mpsc::unbounded_channel();
    tokio::spawn(watch(pipe, handed.pid, kill, None, event_sender));
    let tail = Tail::from_log(&job.log_path);
    tokio::spawn(log_until_done(shared, job, handed.command, tail, events));
    Ok(())
}

// Start the shell. Output: its pid and the read end of the one pipe its stdout and stderr share.
fn spawn(command: &str, context: &ToolContext) -> Result<(u32, Arc<Receiver>), String> {
    use std::os::unix::process::CommandExt;
    let shell = shell_path(context.shared.shell_path.as_deref())?;
    let (reader, writer) = std::io::pipe().map_err(|error| format!("pipe: {error}"))?;
    let error_writer = writer.try_clone().map_err(|error| format!("pipe: {error}"))?;
    let mut process = std::process::Command::new(&shell);
    process.arg("-c").arg(command).current_dir(&context.cwd).stdin(Stdio::null()).stdout(writer).stderr(error_writer);

    // The agent dir's bin/ first on PATH, and the session variables from this session only.
    let path = std::env::var_os("PATH").unwrap_or_default();
    if !std::env::split_paths(&path).any(|dir| dir == context.shared.bin_dir) {
        let joined = std::env::join_paths(std::iter::once(context.shared.bin_dir.clone()).chain(std::env::split_paths(&path)));
        if let Ok(joined) = joined {
            process.env("PATH", joined);
        }
    }
    for name in SESSION_VARIABLES {
        process.env_remove(name);
    }
    let [id, file, provider, model, level] = SESSION_VARIABLES;
    process.env(id, &context.session_id).env(provider, &context.provider).env(model, &context.model_id).env(level, &context.thinking_level);
    if let Some(path) = &context.session_file {
        process.env(file, path);
    }

    // Its own session, as Node's detached spawn does: a process group we can kill as one, and no
    // controlling terminal, so nothing it runs can grab the TUI's keyboard. A read-only job's
    // shell goes in the read-only sandbox too (readonly.rs), on a kernel that can hold it.
    let read_only = context.shared.read_only;
    if read_only {
        super::readonly::check()?;
    }
    // SAFETY: setsid() and the sandbox's system calls are async-signal-safe and run in the child
    // between fork and exec.
    unsafe {
        process.pre_exec(move || {
            libc::setsid();
            if read_only {
                super::readonly::enter()?;
            }
            Ok(())
        });
    }
    let sandbox = if read_only { " in the read-only sandbox (Landlock)" } else { "" };
    let child = process.spawn().map_err(|error| format!("spawn {}{sandbox} {}", shell.display(), super::paths::error_code(&error)))?;
    // The Command holds the pipe's write ends; they close here, so the pipe ends with the shell's
    // own copies. Dropping a std Child reaps nothing: exit_of does that.
    drop(process);
    let pipe = Receiver::from_owned_fd(OwnedFd::from(reader)).map_err(|error| format!("pipe: {error}"))?;
    Ok((child.id(), Arc::new(pipe)))
}

// Read the pipe into the channel until it closes.
async fn pump(pipe: Arc<Receiver>, chunks: mpsc::UnboundedSender<Vec<u8>>) {
    let mut buffer = vec![0u8; READ_CHUNK_BYTES];
    while pipe.readable().await.is_ok() {
        match pipe.try_read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                if chunks.send(buffer[..count].to_vec()).is_err() {
                    break;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(_) => break,
        }
    }
}

// Wait for a shell to exit, through a pidfd, then reap it. One handed to the next image (/restart)
// is left unreaped and this never returns: the next image reaps it and reports its end.
async fn exit_of(pid: u32) -> std::io::Result<std::process::ExitStatus> {
    use std::os::fd::FromRawFd;
    use std::os::unix::process::ExitStatusExt;
    // SAFETY: pidfd_open takes a pid and flags and returns a new descriptor (close-on-exec) or -1.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) } as i32;
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a descriptor just made and owned by nothing else, which the OwnedFd keeps open as
    // long as the AsyncFd holds it.
    let pidfd = unsafe { tokio::io::unix::AsyncFd::register(OwnedFd::from_raw_fd(fd)) }.map_err(std::io::Error::from)?;
    let _ready = pidfd.readable().await?;
    // The check and the reap happen under the list's lock, so hand_over can't come between them.
    let reaped = {
        let running = running();
        let handed = running.iter().any(|child| child.pid == pid && child.handed);
        let mut status = 0;
        // SAFETY: waitpid on our own child, which has exited, into a local status.
        (!handed).then(|| if unsafe { libc::waitpid(pid as libc::pid_t, &mut status, 0) } < 0 { Err(std::io::Error::last_os_error()) } else { Ok(std::process::ExitStatus::from_raw(status)) })
    };
    match reaped {
        Some(result) => result,
        None => std::future::pending().await,
    }
}

// Own the child until it's done: forward its output, kill it on `kill` or at the timeout, and
// report how it ended. Runs as its own task so a command can outlive the tool call that started it.
async fn watch(pipe: Arc<Receiver>, pid: u32, kill: CancellationToken, timeout: Option<Duration>, events: mpsc::UnboundedSender<Event>) {
    let (chunk_sender, mut chunks) = mpsc::unbounded_channel();
    let reader = tokio::spawn(pump(pipe, chunk_sender));
    let exit = exit_of(pid);
    tokio::pin!(exit);

    let deadline = timeout.map(|timeout| tokio::time::Instant::now() + timeout);
    let mut status = None;
    let mut pipes_open = true;
    let mut killed: Option<Outcome> = None;
    let mut grace_until: Option<tokio::time::Instant> = None;
    while status.is_none() || pipes_open {
        tokio::select! {
            chunk = chunks.recv(), if pipes_open => match chunk {
                Some(data) => {
                    let _ = events.send(Event::Data(data));
                    if status.is_some() {
                        grace_until = Some(tokio::time::Instant::now() + EXIT_STDIO_GRACE);
                    }
                }
                None => pipes_open = false,
            },
            result = &mut exit, if status.is_none() => {
                status = Some(result);
                grace_until = Some(tokio::time::Instant::now() + EXIT_STDIO_GRACE);
            }
            () = tokio::time::sleep_until(grace_until.unwrap_or_else(tokio::time::Instant::now)), if grace_until.is_some() && status.is_some() => break,
            () = kill.cancelled(), if killed.is_none() => {
                stop_group(pid).await;
                killed = Some(Outcome::Aborted);
            }
            () = tokio::time::sleep_until(deadline.unwrap_or_else(tokio::time::Instant::now)), if deadline.is_some() && killed.is_none() => {
                stop_group(pid).await;
                killed = Some(Outcome::TimedOut);
            }
        }
    }
    reader.abort();
    let outcome = match (killed, status) {
        (Some(outcome), _) => outcome,
        (None, Some(Ok(status))) => {
            use std::os::unix::process::ExitStatusExt;
            // A shell killed by a signal has no exit code: 128 + the signal, the shell convention.
            Outcome::Exited(status.code().unwrap_or_else(|| status.signal().map_or(1, |signal| 128 + signal)))
        }
        (None, Some(Err(error))) => Outcome::Failed(error.to_string()),
        (None, None) => Outcome::Failed("Command terminated without an exit code".into()),
    };
    // A stopped wrapper can close its pipes before this watcher sees the kill token. Keep its
    // pending escalation registered until it has run, whichever select branch finished first.
    let mut kill_at = None;
    running().retain(|child| {
        if child.pid != pid {
            return true;
        }
        kill_at = child.kill_at;
        kill_at.is_some()
    });
    if let Some(deadline) = kill_at {
        tokio::time::sleep_until(deadline.into()).await;
        let mut children = running();
        if children.iter().any(|child| child.pid == pid) {
            signal_group(pid, libc::SIGKILL);
            children.retain(|child| child.pid != pid);
        }
    }
    super::reaper::track(pid, false);
    let _ = events.send(Event::Done(outcome));
}

// pi's notice under truncated output (bash.ts:340-356).
fn format_output(snapshot: &super::output::Snapshot, last_line_bytes: usize, empty_text: &str) -> (String, Option<Value>) {
    let truncation = &snapshot.truncation;
    let mut text = if snapshot.content.is_empty() { empty_text.to_string() } else { snapshot.content.clone() };
    if !truncation.truncated {
        return (text, None);
    }
    let path = snapshot.full_output_path.as_ref().map(|path| path.display().to_string()).unwrap_or_default();
    let start_line = truncation.total_lines - truncation.output_lines + 1;
    let end_line = truncation.total_lines;
    if truncation.last_line_partial {
        text.push_str(&format!(
            "\n\n[Showing last {} of line {end_line} (line is {}). Full output: {path}]",
            truncate::format_size(truncation.output_bytes),
            truncate::format_size(last_line_bytes)
        ));
    } else if truncation.truncated_by == Some(truncate::TruncatedBy::Lines) {
        text.push_str(&format!("\n\n[Showing lines {start_line}-{end_line} of {}. Full output: {path}]", truncation.total_lines));
    } else {
        text.push_str(&format!(
            "\n\n[Showing lines {start_line}-{end_line} of {} ({} limit). Full output: {path}]",
            truncation.total_lines,
            truncate::format_size(DEFAULT_MAX_BYTES)
        ));
    }
    (text, Some(json!({"truncation": truncation, "fullOutputPath": snapshot.full_output_path})))
}

fn send_update(context: &ToolContext, output: &mut OutputAccumulator) {
    let Some(updates) = &context.updates else { return };
    let snapshot = output.snapshot();
    let details = json!({
        "truncation": snapshot.truncation.truncated.then_some(&snapshot.truncation),
        "fullOutputPath": snapshot.full_output_path,
    });
    let _ = updates.send(ToolUpdate { content: vec![Content::text(snapshot.content)], details: Some(details) });
}

fn append_status(text: &str, status: &str) -> String {
    if text.is_empty() { status.to_string() } else { format!("{text}\n\n{status}") }
}

async fn run(arguments: &Arguments, context: &ToolContext) -> Result<Run, String> {
    let shared = &context.shared;
    let command = match &shared.shell_command_prefix {
        Some(prefix) if !arguments.own => format!("{prefix}\n{}", arguments.command),
        _ => arguments.command.clone(),
    };
    if let Some(updates) = &context.updates {
        let _ = updates.send(ToolUpdate { content: Vec::new(), details: None });
    }
    let timeout = resolve_timeout(arguments.timeout)?;
    if context.cancel.is_cancelled() {
        return Err("Command aborted".into());
    }
    if !context.cwd.exists() {
        return Err(format!("Working directory does not exist: {}\nCannot execute bash commands.", context.cwd.display()));
    }
    let (pid, pipe) = spawn(&command, context)?;
    running().push(Running { pid, own: arguments.own, kill_at: None, pipe: pipe.clone(), handed: false });
    super::reaper::track(pid, true);
    let started = Instant::now();
    let kill = CancellationToken::new();
    let (event_sender, mut events) = mpsc::unbounded_channel();
    tokio::spawn(watch(pipe, pid, kill.clone(), timeout, event_sender));

    let spill_jobs = shared.clone();
    let spill_cwd = context.cwd.clone();
    let mut output = OutputAccumulator::new(DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES, Box::new(move || spill_jobs.jobs.spill_path(&spill_cwd)));
    // Any command still running at the soft limit moves to the background, and so does every one
    // when the user sends a message now. A timeout only kills a command (watch enforces it there
    // too); it never holds one in the foreground, where a long one would block the session.
    // longcmd kept commands with a timeout in the foreground.
    // With no wait at all (background: true, or a soft limit of 0) it goes now: a zero sleep fires
    // on the timer's next tick, by which time a quick command's output has won the loop below.
    if arguments.background || shared.jobs.soft_limit.is_zero() {
        return Ok(Run::Background(to_background(arguments, context, &mut output, pid, kill.clone(), started, Tail::default(), events)));
    }
    let mut tail = Tail::default();
    let soft_limit = tokio::time::sleep(shared.jobs.soft_limit);
    tokio::pin!(soft_limit);
    let _detachable = shared.jobs.detachable();
    let detach = shared.jobs.detach.notified();
    tokio::pin!(detach);
    let mut dirty = false;
    let mut last_update: Option<tokio::time::Instant> = None;
    let mut update_at: Option<tokio::time::Instant> = None;

    let outcome = loop {
        // In order: a move to the background ahead of an abort, so Esc (which asks for both) leaves
        // the command running as a job instead of killing it.
        tokio::select! {
            biased;
            event = events.recv() => match event {
                Some(Event::Data(data)) => {
                    output.append(&data);
                    tail.push(&data);
                    if context.updates.is_some() {
                        dirty = true;
                        if update_at.is_none() {
                            let now = tokio::time::Instant::now();
                            let due = last_update.map_or(now, |last| last + UPDATE_THROTTLE);
                            if due <= now {
                                dirty = false;
                                last_update = Some(now);
                                send_update(context, &mut output);
                            } else {
                                update_at = Some(due);
                            }
                        }
                    }
                }
                Some(Event::Done(outcome)) => break outcome,
                None => break Outcome::Failed("Command terminated without an exit code".into()),
            },
            () = tokio::time::sleep_until(update_at.unwrap_or_else(tokio::time::Instant::now)), if update_at.is_some() => {
                update_at = None;
                if dirty {
                    dirty = false;
                    last_update = Some(tokio::time::Instant::now());
                    send_update(context, &mut output);
                }
            }
            () = async {
                tokio::select! {
                    () = &mut soft_limit => {}
                    () = &mut detach => {}
                }
            }, if !kill.is_cancelled() => {
                return Ok(Run::Background(to_background(arguments, context, &mut output, pid, kill.clone(), started, std::mem::take(&mut tail), events)));
            }
            () = context.cancel.cancelled(), if !kill.is_cancelled() => kill.cancel(),
        }
    };

    output.finish();
    output.check_error()?;
    if dirty {
        send_update(context, &mut output);
    }
    let snapshot = output.snapshot();
    let last_line_bytes = output.last_line_bytes();
    match outcome {
        Outcome::Aborted => Err(append_status(&format_output(&snapshot, last_line_bytes, "").0, "Command aborted")),
        Outcome::TimedOut => {
            let seconds = crate::js::number_string(arguments.timeout.unwrap_or_default());
            Err(append_status(&format_output(&snapshot, last_line_bytes, "").0, &format!("Command timed out after {seconds} seconds")))
        }
        Outcome::Failed(message) => Err(message),
        Outcome::Exited(code) => {
            let (text, details) = format_output(&snapshot, last_line_bytes, "(no output)");
            let text = if code == 0 { text } else { append_status(&text, &format!("Command exited with code {code}")) };
            Ok(Run::Finished(ToolOutput { content: vec![Content::text(text)], details, is_error: code != 0 }))
        }
    }
}

// The soft limit passed, or the user sent a message now: hand the command to a background job and
// tell the model (longcmd.ts:493-529).
// Output: the call's result.
#[allow(clippy::too_many_arguments)]
fn to_background(
    arguments: &Arguments,
    context: &ToolContext,
    output: &mut OutputAccumulator,
    pid: u32,
    kill: CancellationToken,
    started: Instant,
    tail: Tail,
    events: mpsc::UnboundedReceiver<Event>,
) -> ToolOutput {
    let shared = context.shared.clone();
    let session_tag: String = if context.session_id.is_empty() { std::process::id().to_string() } else { context.session_id.chars().take(8).collect() };
    let job = shared.jobs.start(&session_tag, &arguments.command, pid, kill, started, output.take_spill());
    let (text, cut) = tail.text();
    let status = format!(
        "[Still running after {}; moved to background job {}, pid {}. {}Full output, including what it prints from now on: {}]",
        jobs::seconds_text(started.elapsed()),
        job.id,
        pid,
        if cut { "Output above is the tail. " } else { "" },
        job.log_path.display()
    );
    let instruction = format!(
        "It keeps running and you will get a message when it exits; do not poll or sleep. Continue with other work or end your turn. To stop it: kill -- -{pid}"
    );
    let reply = format!("{}{status}\n{instruction}", if text.is_empty() { String::new() } else { format!("{text}\n\n") });

    tokio::spawn(log_until_done(shared, job, arguments.command.clone(), tail, events));
    ToolOutput { content: vec![Content::text(reply)], details: None, is_error: false }
}

// A background job's output goes to its log and its tail until the command ends; then the job
// finishes, which sends its message.
async fn log_until_done(shared: Arc<super::ToolShared>, mut job: jobs::Job, command: String, mut tail: Tail, mut events: mpsc::UnboundedReceiver<Event>) {
    let end = loop {
        match events.recv().await {
            Some(Event::Data(data)) => {
                job.write(&data);
                tail.push(&data);
            }
            Some(Event::Done(Outcome::Exited(code))) => break JobEnd::Exited(code),
            Some(Event::Done(Outcome::Failed(message))) => break JobEnd::Failed(message),
            Some(Event::Done(Outcome::Aborted | Outcome::TimedOut)) => break JobEnd::Failed("Command aborted".into()),
            None => break JobEnd::Failed("Command terminated without an exit code".into()),
        }
    };
    shared.jobs.finish(job, &command, &tail, end);
}
