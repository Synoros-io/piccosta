// The script tool (the Python command batcher): the model writes one Python program that calls
// the session's own tools as async functions, and it runs in Monty (pydantic's Python interpreter
// written in Rust, MIT, github.com/pydantic/monty), so a batch of tool calls is one call. It does
// what Factory's Droid 1.0 Script tool does (factory.ai/news/frontier-harness), in Python:
// - Per model, by `script.mode` (mode): "only" sends the model script and read as its only direct
//   tools, every other tool reachable only as a function in a script, as Droid sends two tools up
//   front and Codex has CodeModeOnly; "offer" sends script beside the direct tools; "off" leaves
//   it out. The computer is never a function, always a direct call: computer use is look, act,
//   look again, one step at a time, and a script moved to the background loses its screenshots. The description declares each function as a typed Python signature with its tool's
//   description, as Codex renders its tools as TypeScript declarations.
// - The functions are the session's tool list as it stands (host.rs hands it over through bind):
//   a tool the session hands to a job (search, say) is missing here too and goes through
//   `subagent` the same way; arguments are validated as the agent loop validates them; every call
//   runs in the session's cwd, and an abort stops the script and its shells.
// - Each call a script makes passes the session's approval gate (ACP, rpc) as the model's own
//   calls do, shown on the script's card; the script itself isn't asked about (read_only), since
//   it reaches nothing except through those calls.
// - The model gets back only what the script prints, cut as any tool's text is. Each call's full
//   result goes to a log in the job directory (jobs.rs), and the output ends with its path, so the
//   model can read or rg it later instead of carrying every result in its context.
// - A script runs like a bash command under the jobs system: still running at the soft limit, or
//   when the user sends a message, it moves to the background with what it printed so far, and its
//   end comes as a job message. Its own commands stay in the foreground (stay_in_foreground).
// - An edit inside a script returns the same one-line summary a direct edit does; its diff goes to
//   the log.
//
// Each script runs in a worker: piccosta's own binary started with WORKER_ENV set, in a process
// group of its own, under kernel limits on its memory and CPU time. The worker runs the Monty
// program and asks piccosta for each tool call over its stdin and stdout, one JSON object a line;
// piccosta runs the call under the session's rules and answers. So a script is an ordinary process:
// a Monty panic or a script out of memory ends the worker alone, and a background script stops by
// its pid as a command does. /restart can't hand a worker to the next image (its calls are answered
// by this one), so it waits for running scripts to end (tui/mod.rs).
//
// Measured on 144 model-written scripts in the lab (2026-10-29, ledger lines 44 and 46): 13.7% hit
// a Monty gap, half of them next() on a generator expression, which the description warns about.

use super::jobs::{self, JobEnd};
use super::{truncate, validate, Tool, ToolContext, ToolFuture, ToolOutput, ToolShared};
use crate::message::{Content, ToolDefinition};
use monty::{MontyRun, RunProgress};
use monty_types::{CompileOptions, ExcType, ExtFunctionResult, MontyException, MontyObject, NameLookupResult, ObjectRef, OsFunctionCall, PrintWriter, PrintWriterCallback, ResourceLimits, ResourceTracker};
use serde_json::{json, Map, Value};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::process::Child;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

pub const NAME: &str = "script";
// Set in a worker's environment; main() checks it before anything else.
pub const WORKER_ENV: &str = "PICCOSTA_SCRIPT_WORKER";
// The worker is the image this process runs, even after a build has replaced the file at its path.
const WORKER_EXE: &str = "/proc/self/exe";
// Calls to tools one script may make. Monty leaves the count to the host; 1000 is its default too,
// and 17 times the most any of the lab's 144 scripts made (59). Past it a script is almost surely
// looping, and the error is uncatchable inside it.
const MAX_CALLS: usize = 1000;
// CPU time per script, which waiting on tools doesn't use. The lab's slowest script took 0.72 s in
// Monty (deepseek/loop_use_crate).
const MAX_CPU_SECONDS: u64 = 10;
// A worker's data (its heap and private mappings, RLIMIT_DATA), the runtime's idle threads
// included; and what print() may send (the Python package's own print cap, 10 MiB, from the lab's
// sandbox probes).
const MAX_MEMORY_BYTES: u64 = 512 * 1024 * 1024;
const MAX_PRINT_BYTES: usize = 10 * 1024 * 1024;
// One call's arguments, all its keywords together, counted over the value as JSON would spell it
// out (keys, quotes and commas too), so a list holding the same list or dict twice counts it twice
// and a small graph of shared references can't expand without end. Depth stays under serde_json's
// 128 levels, which piccosta reads the call with; the values and bytes are far past any tool's
// arguments (an edit's list of edits), and the bytes match the print cap.
const MAX_ARGUMENT_DEPTH: usize = 64;
const MAX_ARGUMENT_VALUES: usize = 100_000;
const MAX_ARGUMENT_BYTES: usize = MAX_PRINT_BYTES;
// The longest line piccosta reads from a worker: a call's arguments or a print at their caps, with
// room for the line's own keys, its call id and a function's name. A longer one means the worker
// is broken or not ours, and its script ends before piccosta holds the line.
const MAX_LINE_BYTES: usize = MAX_ARGUMENT_BYTES + 4096;
// How long a script's calls get to stop after it ends: a subagent's stop grace (bash.rs), the
// longest stop any tool takes, and as long again. A call still running then is left to its task
// and named in the script's error, so the script's end never waits on it forever.
const CLEANUP_BOUND: std::time::Duration = super::bash::SUBAGENT_STOP_GRACE.saturating_mul(2);
// How much of a dead worker's stderr (a panic's message, a failed allocation) its error shows.
const STDERR_BYTES: u64 = 4096;
// The interpreter thread's stack: Linux's main-thread default. Monty nests up to 12 native calls
// and found ~2 MiB too little at 19 (MAX_RUN_REENTRY_DEPTH, monty 1.4 bytecode/vm/recursion.rs).
const STACK_BYTES: usize = 8 * 1024 * 1024;
// Put in front of every script, so the line numbers Monty reports are one too high.
const PRELUDE: &str = "import asyncio\n";

// The models Factory's Droid 1.0 turns its Script tool on for by default, as provider/id patterns
// (factory.ai/news/frontier-harness, read 2026-11-02): `script.mode` is "only" for them unless
// the user's settings say otherwise, and "off" for every other model.
const DROID_SCRIPT_MODELS: [&str; 10] = [
    "*/gpt-6.1-sol", "*/gpt-6.1-terra", "*/gpt-6-astra", "*/gpt-6-sol", "*/gpt-5.6-sol",
    "*/claude-opus-5-5", "*/claude-opus-5", "*/claude-fable-5-1", "*/claude-fable-5", "*/kimi-k3",
];

#[derive(Default)]
pub struct ScriptTool {
    functions: Vec<Arc<dyn Tool>>,
    // "only": every tool but read and the computer is reached through scripts.
    only: bool,
}

// The session's tool list for `model` (provider/id), the script tool given the rest of it as
// functions: in "only" mode the direct tools are script, read and the computer, in "offer" all of
// them, and in "off" (or with no script tool in the list) the list as it was. Input: the list after every cut.
pub fn bind(list: Vec<Arc<dyn Tool>>, settings: &crate::config::Settings, model: &str) -> Vec<Arc<dyn Tool>> {
    let mode = mode(settings, model);
    let only = mode == "only" && list.iter().any(|tool| tool.name() == NAME);
    let functions: Vec<Arc<dyn Tool>> = list.iter().filter(|tool| !matches!(tool.name(), NAME | "computer")).cloned().collect();
    let script: Arc<dyn Tool> = Arc::new(ScriptTool { functions, only });
    let bound = |tool: Arc<dyn Tool>| match tool.name() {
        NAME => matches!(mode.as_str(), "only" | "offer").then(|| script.clone()),
        "read" | "computer" => Some(tool),
        _ => (!only).then_some(tool),
    };
    list.into_iter().filter_map(bound).collect()
}

// `script.mode` for a model: one mode for every model, or an object from provider/id patterns (*
// for any run of characters) to modes. The user's patterns come before DROID_SCRIPT_MODELS, and
// among either the longest pattern that matches wins.
fn mode(settings: &crate::config::Settings, model: &str) -> String {
    let setting = settings.get("script.mode");
    if let Some(mode) = setting.and_then(Value::as_str) {
        return mode.to_string();
    }
    let longest = |patterns: Vec<(&str, &str)>| patterns.into_iter().filter(|(pattern, _)| glob(pattern, model)).max_by_key(|(pattern, _)| pattern.len()).map(|(_, mode)| mode.to_string());
    let users = setting.and_then(Value::as_object).into_iter().flatten().filter_map(|(pattern, mode)| Some((pattern.as_str(), mode.as_str()?))).collect();
    longest(users).or_else(|| longest(DROID_SCRIPT_MODELS.iter().map(|pattern| (*pattern, "only")).collect())).unwrap_or_else(|| "off".into())
}

// Whether `text` matches `pattern`, where * stands for any run of characters.
fn glob(pattern: &str, text: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == text,
        Some((head, rest)) => text.strip_prefix(head).is_some_and(|tail| tail.char_indices().map(|(at, _)| at).chain([tail.len()]).any(|at| glob(rest, &tail[at..]))),
    }
}

impl Tool for ScriptTool {
    fn name(&self) -> &'static str {
        NAME
    }

    // The description is the same on every request of a session (its functions are the session's
    // tool list), so it stays in the cached prompt prefix.
    fn definition(&self) -> ToolDefinition {
        let lead = match self.only {
            true => "Run a Python program that calls this session's tools as async functions. Every tool but read and computer is reached this way: one call is a one-line script (`print(await bash(command=\"ls\"))`), and a step with several calls (several files to read or search, several commands, a loop over items, calls where one result decides the next) is one script.",
            false => "Run a Python program that calls this session's tools as async functions. Use it whenever a step takes more than one tool call: reading or searching several files, running several commands, a loop over items, or calls where one result decides the next. Call a tool directly only for a single call.",
        };
        let functions: Vec<String> = self.functions.iter().map(|tool| signature(&tool.definition())).collect();
        ToolDefinition {
            name: NAME.into(),
            description: format!(
                "{lead} \
                 Only what the program prints comes back, so print just what you need (counts, matches, a table); every call's full result is saved to a log, and the output ends with its path for read or rg later. \
                 A tool's error (a command's nonzero exit too) raises RuntimeError with its text. \
                 asyncio is already imported. Await every call: `text = await read(path=\"a.rs\")`; `await asyncio.gather(read(path=\"a\"), read(path=\"b\"))` runs calls at once. \
                 At most {MAX_CALLS} calls, {MAX_CPU_SECONDS} s of CPU and {} MiB of memory. A script still running after a few seconds moves to the background with what it printed so far, and the rest comes as a message when it ends. \
                 It runs in Monty, a sandboxed subset of Python: no files, environment, processes or sleep except through the tools; standard modules only, and not shlex, glob, textwrap or subprocess; \
                 generator expressions are lists, so next() on one fails (index a list instead); no del, yield, async with, create_task, wait_for, or keyword arguments to gather. \
                 Errors come back as a traceback; calls made before an error have already run.\n\n\
                 The functions. Each takes keyword arguments only (None counts as not given) and returns the tool's text, an image as \"[image]\" (the images themselves come to you with the script's result):\n\n{}",
                MAX_MEMORY_BYTES >> 20,
                functions.join("\n\n")
            ),
            parameters: json!({
                "type": "object",
                "required": ["code"],
                "properties": {"code": {"type": "string", "description": "The Python program"}}
            }),
            constrained_sampling: None,
        }
    }

    fn prompt_snippet(&self) -> Option<&'static str> {
        match self.only {
            true => Some("run a Python program that calls the tools: commands, edits, writes, searches and everything else but read and computer"),
            false => Some("Run a Python program that calls the other tools; the default for any step with more than one call"),
        }
    }

    fn prompt_guidelines(&self) -> &'static [&'static str] {
        match self.only {
            true => &["Every tool but read and computer is a function in a script: a single command is a one-line script, and a step with several calls (several files, several commands, a loop over items) is one script that prints only what you need"],
            false => &["When a step needs more than one tool call (several files to read or search, several commands, a loop over items), make them all in one script that prints only what you need; call a tool directly only for a single call."],
        }
    }

    // Each call a script makes is asked about by itself (drive), so the script isn't.
    fn read_only(&self) -> bool {
        true
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let code = arguments.get("code").and_then(Value::as_str).unwrap_or_default().to_string();
            let jobs = &context.shared.jobs;
            // The script's own stop token: the turn's abort fires it while the script is in the
            // foreground, its job's kill once it's in the background, and dropping this call does
            // until then.
            let stop = CancellationToken::new();
            let stop_on_drop = stop.clone().drop_guard();
            let (printed, images, log) = (Arc::new(Mutex::new(String::new())), Arc::new(Mutex::new(Vec::new())), CallLog::new(context));
            let log_path = log.path.clone();
            let worker = start_worker()?;
            let pid = worker.id().unwrap_or_default();
            let (sender, mut receiver) = tokio::sync::oneshot::channel();
            let driven = drive(worker, code.clone(), self.functions.clone(), context.clone(), stop.clone(), printed.clone(), images.clone(), log);
            tokio::spawn(async move { sender.send(driven.await) });

            // Wait as bash waits on a command: to the end, unless the soft limit passes or the user
            // sends a message first, which move the script to the background (ahead of an abort,
            // as Esc asks for both).
            let started = Instant::now();
            let soft_limit = tokio::time::sleep(jobs.soft_limit);
            let detach = jobs.detach.notified();
            let _detachable = jobs.detachable();
            tokio::pin!(soft_limit, detach);
            let finished = loop {
                tokio::select! {
                    biased;
                    finished = &mut receiver => break finished,
                    () = async {
                        tokio::select! {
                            () = &mut soft_limit => {}
                            () = &mut detach => {}
                        }
                    } => {
                        let session_tag: String = if context.session_id.is_empty() { std::process::id().to_string() } else { context.session_id.chars().take(8).collect() };
                        let kill = CancellationToken::new();
                        let so_far = locked(&printed).clone();
                        // Its job's log starts with what it printed so far, as a command's starts
                        // with its spill file.
                        let spill_path = jobs.spill_path(&context.cwd);
                        let spill = std::fs::File::create(&spill_path).and_then(|mut file| file.write_all(so_far.as_bytes()).map(|()| (spill_path, file))).ok();
                        let mut job = jobs.start(&session_tag, &code, pid, kill.clone(), started, spill);
                        stop_on_drop.disarm();
                        // The call log is named only once a call has finished and made it.
                        let reply = format!(
                            "{}[Still running after {}; moved to background job {}, pid {pid}. What it prints from now on and how it ends come in a message when it ends; do not poll or sleep. Continue with other work or end your turn. To stop it: kill -- -{pid}.{}]",
                            if so_far.is_empty() { String::new() } else { format!("{}\n\n", cut(so_far.trim_end())) },
                            jobs::seconds_text(started.elapsed()),
                            job.id,
                            if log_path.exists() { format!(" Full results of its tool calls so far: {}", log_path.display()) } else { String::new() }
                        );
                        // The rest of its printing, its error and the log line go to the job's log
                        // and its message when the script ends.
                        let shared = context.shared.clone();
                        tokio::spawn(async move {
                            let finished = tokio::select! {
                                finished = &mut receiver => finished,
                                () = kill.cancelled() => {
                                    stop.cancel();
                                    receiver.await
                                }
                            };
                            let printed = locked(&printed).clone();
                            let (text, end) = match finished {
                                Ok(Ok((calls, error))) => (report(&printed, error.as_deref(), calls, &log_path), JobEnd::Exited(i32::from(error.is_some()))),
                                Ok(Err(error)) => (printed, JobEnd::Failed(error)),
                                Err(_) => (printed, JobEnd::Failed(STOPPED.into())),
                            };
                            job.write(&text.as_bytes()[so_far.len().min(text.len())..]);
                            let mut tail = jobs::Tail::default();
                            tail.push(text.as_bytes());
                            shared.jobs.finish(job, &code, &tail, end);
                        });
                        return Ok(ToolOutput { content: vec![Content::text(reply)], details: None, is_error: false });
                    }
                    () = context.cancel.cancelled(), if !stop.is_cancelled() => stop.cancel(),
                }
            };
            let (calls, error) = finished.map_err(|_| STOPPED.to_string())??;
            let text = report(&locked(&printed), error.as_deref(), calls, &log_path);
            let mut content = vec![Content::text(cut(&text))];
            content.extend(std::mem::take(&mut *locked(&images)));
            Ok(ToolOutput { content, details: Some(json!({"calls": calls})), is_error: error.is_some() })
        })
    }
}

// What a script returns when it was stopped, or ended without an answer.
const STOPPED: &str = "Operation aborted";

fn locked<T>(shared: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

// Start a worker for one script: this binary in worker mode, in a process group of its own (so a
// background script stops by `kill -- -<pid>` as a command does), with no core files and its
// memory and CPU capped by the kernel. Its group goes on the reaper's list (reaper.rs), so a worker
// still computing when piccosta is killed dies with it; drive takes it off once it's reaped.
fn start_worker() -> Result<Child, String> {
    use std::process::Stdio;
    let mut command = tokio::process::Command::new(WORKER_EXE);
    command.env(WORKER_ENV, "1").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    // SAFETY: setsid() and setrlimit() are async-signal-safe and run in the child between fork and
    // exec. CPU's hard limit is a second past the soft one, so a worker that ignored SIGXCPU still
    // ends by SIGKILL.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            let limits = [(libc::RLIMIT_CORE, 0, 0), (libc::RLIMIT_DATA, MAX_MEMORY_BYTES, MAX_MEMORY_BYTES), (libc::RLIMIT_CPU, MAX_CPU_SECONDS, MAX_CPU_SECONDS + 1)];
            for (resource, soft, hard) in limits {
                if libc::setrlimit(resource, &libc::rlimit { rlim_cur: soft, rlim_max: hard }) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let worker = command.spawn().map_err(|error| format!("could not start the script's worker: {error}"))?;
    super::reaper::track(worker.id().unwrap_or_default(), true);
    Ok(worker)
}

// Counts a script as running in its session until it's dropped; /restart waits while any is.
struct Running(Arc<ToolShared>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.jobs.scripts.fetch_sub(1, Ordering::SeqCst);
    }
}

// Answer one worker's tool calls until its script ends. Input: the worker, its code, the functions
// it may call, the session's context, the script's stop token, where its printing and its calls'
// images go, and the call log. Output: how many tool calls it made and its error (a traceback, or why
// its worker died), if it ended in one; Err when it was stopped.
#[allow(clippy::too_many_arguments)]
async fn drive(mut worker: Child, code: String, functions: Vec<Arc<dyn Tool>>, context: ToolContext, stop: CancellationToken, printed: Arc<Mutex<String>>, images: Arc<Mutex<Vec<Content>>>, mut log: CallLog) -> Result<(usize, Option<String>), String> {
    context.shared.jobs.scripts.fetch_add(1, Ordering::SeqCst);
    let _running = Running(context.shared.clone());
    // Its calls get their own cancel token under the script's, fired whenever the script ends.
    let inner = ToolContext { updates: None, cancel: stop.child_token(), stay_in_foreground: true, ..context };
    let pid = worker.id().unwrap_or_default();
    let (Some(mut stdin), Some(stdout)) = (worker.stdin.take(), worker.stdout.take()) else {
        let _ = worker.kill().await;
        return Err("the script's worker has no pipes".into());
    };
    // What goes to the worker is written from a task of its own, so a worker busy computing never
    // holds up the reading of its calls. First the code and the functions it may call.
    let (answers, mut queued) = tokio::sync::mpsc::unbounded_channel::<Value>();
    let names: Vec<&str> = functions.iter().map(|tool| tool.name()).collect();
    let _ = answers.send(json!({"code": code, "functions": names}));
    tokio::spawn(async move {
        while let Some(line) = queued.recv().await {
            if stdin.write_all(format!("{line}\n").as_bytes()).await.is_err() {
                break;
            }
        }
    });
    // Its lines are read up to MAX_LINE_BYTES each; a read cut short by another branch leaves its
    // bytes in `line` and the next one carries on.
    let (mut stdout, mut line) = (tokio::io::BufReader::new(stdout), Vec::new());
    // Each call runs as a task answering its call id; its text as the script made it waits for the log.
    let mut running: JoinSet<(u64, Result<ToolOutput, String>)> = JoinSet::new();
    let mut made: HashMap<u64, String> = HashMap::new();
    let mut calls = 0;
    let ended = loop {
        tokio::select! {
            read = async { (&mut stdout).take((MAX_LINE_BYTES + 1 - line.len()) as u64).read_until(b'\n', &mut line).await } => {
                if line.len() > MAX_LINE_BYTES {
                    break Ok(Some(format!("Error: the script's process sent a line over {} MiB", MAX_LINE_BYTES >> 20)));
                }
                if read.is_err() || !line.ends_with(b"\n") {
                    break Ok(Some(died(&mut worker).await));
                }
                let message: Value = serde_json::from_slice(&std::mem::take(&mut line)).unwrap_or_default();
                if let Some(text) = message["print"].as_str() {
                    locked(&printed).push_str(text);
                } else if let Some(call_id) = message["call"].as_u64() {
                    // A tool call: checked as the agent loop checks one, then run while the script goes on.
                    let name = message["function"].as_str().unwrap_or_default();
                    let tool = functions.iter().find(|tool| tool.name() == name).cloned();
                    let given = message["arguments"].as_object().cloned().unwrap_or_default();
                    let checked = tool.ok_or_else(|| format!("no function {name}")).and_then(|tool| {
                        let schema = tool.definition().parameters;
                        validate::validate(tool.name(), &schema, &tool.prepare_arguments(given)).map(|arguments| (tool, arguments))
                    });
                    let shown = checked.as_ref().map_or(String::new(), |(_, arguments)| Value::Object(arguments.clone()).to_string());
                    made.insert(call_id, format!("{name}({shown})"));
                    calls += 1;
                    let context = inner.clone();
                    running.spawn(async move {
                        let (tool, arguments) = match checked {
                            Ok(ready) => ready,
                            Err(error) => return (call_id, Err(error)),
                        };
                        // The session's gate asks about it as about the model's own calls, on the
                        // script's card; the script ending stops the wait. A call that needs the
                        // user's yes, with no gate to ask, is refused (agent.rs).
                        let question = tool.approval(&arguments);
                        if let Some((script_call, gate)) = &context.gate {
                            let allowed = tokio::select! {
                                allowed = gate(script_call.clone(), tool.name().to_string(), arguments.clone(), question) => allowed,
                                () = context.cancel.cancelled() => Err(STOPPED.to_string()),
                            };
                            if let Err(refused) = allowed {
                                return (call_id, Err(refused));
                            }
                        } else if let Some(question) = question {
                            return (call_id, Err(format!("This call needs the user's yes (\"{question}\"), and nobody can answer here (print mode or a subagent).")));
                        }
                        (call_id, tool.execute(arguments, &context).await)
                    });
                } else if let Some(end) = message.get("end") {
                    break Ok(end.as_str().map(str::to_string));
                }
            }
            // A call finished: its full result to the log, and to the script its text, or its
            // error (a command's nonzero exit too) to raise as RuntimeError.
            Some(Ok((call_id, result))) = running.join_next() => {
                if let Some(call) = made.remove(&call_id) {
                    log.write(&call, &result);
                }
                // An image a call returned (a picture read) can't be printed, so the
                // latest MAX_IMAGES go to the model with the script's result.
                if let Ok(output) = &result {
                    let mut kept = locked(&images);
                    kept.extend(output.content.iter().filter(|block| matches!(block, Content::Image { .. })).cloned());
                    let over = kept.len().saturating_sub(MAX_IMAGES);
                    kept.drain(..over);
                }
                let _ = answers.send(match result {
                    Ok(output) if !output.is_error => json!({"call": call_id, "text": text_of(&output.content)}),
                    Ok(output) => json!({"call": call_id, "error": text_of(&output.content)}),
                    Err(error) => json!({"call": call_id, "error": error}),
                });
            }
            () = stop.cancelled() => break Err(STOPPED.to_string()),
        }
    };
    // However it ended, stop the worker and the calls still running, and wait for the calls to
    // clean up (a command dies with its group), so nothing the script started outlives it; up to
    // CLEANUP_BOUND, past which the script's error names the calls that didn't stop. The worker is
    // waited on too, so it's reaped before the running count comes down and /restart can exec.
    inner.cancel.cancel();
    let _ = worker.kill().await;
    super::reaper::track(pid, false);
    let cleaned = tokio::time::timeout(CLEANUP_BOUND, async {
        while let Some(finished) = running.join_next().await {
            if let Ok((call_id, _)) = finished {
                made.remove(&call_id);
            }
        }
    });
    if cleaned.await.is_err() {
        // Dropping the set would abort them mid-write and free their files' locks, so they're
        // detached instead: each finishes, holding its lock, after the script has ended.
        running.detach_all();
        let stuck: Vec<String> = made.into_values().collect();
        return Err(format!("the script ended, but these calls didn't stop within {} s and were left running: {}", CLEANUP_BOUND.as_secs(), stuck.join(", ")));
    }
    ended.map(|error| (calls, error))
}

// Why a worker's output closed before its script ended: the CPU limit, a kill (the model's own, or
// the kernel's out-of-memory killer), or what it wrote to stderr (a panic, a failed allocation).
async fn died(worker: &mut Child) -> String {
    use std::os::unix::process::ExitStatusExt;
    let mut stderr = String::new();
    if let Some(pipe) = worker.stderr.take() {
        let _ = pipe.take(STDERR_BYTES).read_to_string(&mut stderr).await;
    }
    let status = worker.wait().await.ok();
    let why = match status.and_then(|status| status.signal()) {
        Some(libc::SIGXCPU) => format!("it used its {MAX_CPU_SECONDS} s of CPU time"),
        Some(signal @ (libc::SIGKILL | libc::SIGTERM)) => format!("it was killed (signal {signal})"),
        _ if !stderr.trim().is_empty() => stderr.trim().to_string(),
        _ => status.map_or_else(|| "it could not be waited on".to_string(), |status| status.to_string()),
    };
    format!("Error: the script's process ended before the script did: {why}. A script has {} MiB of memory and {MAX_CPU_SECONDS} s of CPU.", MAX_MEMORY_BYTES >> 20)
}

// Each call's full result, appended as calls finish to a file in the session's job directory. The
// file is made at the first call, so a script that makes none leaves nothing behind.
struct CallLog {
    path: PathBuf,
    file: Option<std::fs::File>,
}

impl CallLog {
    fn new(context: &ToolContext) -> CallLog {
        let name = format!("script-{}.log", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        CallLog { path: context.shared.jobs.job_dir(&context.cwd).join(name), file: None }
    }

    // Input: the call as the script made it, and its result: the tool's text and, for an edit,
    // its diff; or an error's message.
    fn write(&mut self, call: &str, result: &Result<ToolOutput, String>) {
        let text = match result {
            Ok(output) => {
                let diff = output.details.as_ref().and_then(|details| details.get("diff")).and_then(Value::as_str);
                [Some(text_of(&output.content)), diff.map(str::to_string)].into_iter().flatten().collect::<Vec<_>>().join("\n")
            }
            Err(error) => format!("Error: {error}"),
        };
        if self.file.is_none() {
            self.file = std::fs::File::create(&self.path).ok();
        }
        if let Some(file) = &mut self.file {
            let _ = writeln!(file, "### {call}\n{text}\n");
        }
    }
}

// What the model reads: the printed text, then the error's traceback with a reminder that the calls
// before it have run, then where each call's full result is.
fn report(printed: &str, error: Option<&str>, calls: usize, log_path: &Path) -> String {
    let (mut text, silent) = (printed.to_string(), printed.is_empty() && error.is_none());
    let mut line = |line: &str| {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(line);
    };
    if let Some(error) = error {
        line(error);
    }
    match calls {
        0 if silent => line("(no output)"),
        0 => {}
        calls if error.is_some() => line(&format!("({calls} calls ran before the error; their effects stay. Full results: {})", log_path.display())),
        calls => line(&format!("[Full results of {calls} tool calls: {}]", log_path.display())),
    }
    text
}

// Text cut as any tool's is, from the end.
fn cut(text: &str) -> String {
    let cut = truncate::truncate_tail(text, truncate::DEFAULT_MAX_LINES, truncate::DEFAULT_MAX_BYTES);
    match cut.truncated {
        true => format!("{}\n\n[Showing the end only: {} of {}. Print less.]", cut.content, truncate::format_size(cut.output_bytes), truncate::format_size(cut.total_bytes)),
        false => text.to_string(),
    }
}

// A tool as the Python function a script calls: its parameters typed and keyword-only, the ones it
// requires first, and its description on one line as the docstring.
fn signature(definition: &ToolDefinition) -> String {
    let schema = &definition.parameters;
    let required: Vec<&str> = schema["required"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    let properties = schema["properties"].as_object().into_iter().flatten();
    let (needed, optional): (Vec<_>, Vec<_>) = properties.partition(|(name, _)| required.contains(&name.as_str()));
    let parameters: Vec<String> = needed.iter().map(|(name, kind)| format!("{name}: {}", python_type(kind)))
        .chain(optional.iter().map(|(name, kind)| format!("{name}: {} | None = None", python_type(kind))))
        .collect();
    let parameters = if parameters.is_empty() { String::new() } else { format!("*, {}", parameters.join(", ")) };
    let doc = definition.description.split_whitespace().collect::<Vec<_>>().join(" ").replace("\"\"\"", "'''");
    format!("async def {}({parameters}) -> str:\n    \"\"\"{doc}\"\"\"", definition.name)
}

// A JSON Schema type as Python spells it: str, int, float, bool, list[...], an object as its keys
// and their types, an enum as a Literal of its values, and a union for anyOf, oneOf or a list of
// types.
fn python_type(schema: &Value) -> String {
    let union = |options: &Vec<Value>| options.iter().map(python_type).collect::<Vec<_>>().join(" | ");
    if let Some(values) = schema["enum"].as_array() {
        return format!("Literal[{}]", values.iter().map(Value::to_string).collect::<Vec<_>>().join(", "));
    }
    if let Some(options) = schema["anyOf"].as_array().or(schema["oneOf"].as_array()) {
        return union(options);
    }
    if let Some(types) = schema["type"].as_array() {
        return union(&types.iter().map(|kind| json!({"type": kind})).collect());
    }
    match schema["type"].as_str() {
        Some("string") => "str".into(),
        Some("integer") => "int".into(),
        Some("number") => "float".into(),
        Some("boolean") => "bool".into(),
        Some("null") => "None".into(),
        Some("array") => format!("list[{}]", python_type(&schema["items"])),
        Some("object") => match schema["properties"].as_object().filter(|fields| !fields.is_empty()) {
            Some(fields) => format!("{{{}}}", fields.iter().map(|(name, kind)| format!("\"{name}\": {}", python_type(kind))).collect::<Vec<_>>().join(", ")),
            None => "dict".into(),
        },
        _ => "Any".into(),
    }
}

// How many images a script's result carries, its latest. A 1280x800 screenshot costs a model about
// 1,000-1,500 tokens, so four show what a script ended on without growing a turn by more than a few
// thousand; a script that needs to look at more is better as several calls.
const MAX_IMAGES: usize = 4;

// A tool result's text: its text blocks, an image as a marker.
fn text_of(content: &[Content]) -> String {
    let block = |block: &Content| match block {
        Content::Text { text, .. } => text.clone(),
        Content::Image { .. } => "[image]".to_string(),
        _ => String::new(),
    };
    content.iter().map(block).collect::<Vec<_>>().join("\n")
}

// The worker's side, on a thread with the stack Monty needs. A failure to talk to piccosta ends it:
// piccosta is gone or has stopped the script.
pub fn work() -> std::process::ExitCode {
    let thread = std::thread::Builder::new().name("script".into()).stack_size(STACK_BYTES).spawn(run);
    match thread.map(std::thread::JoinHandle::join) {
        Ok(Ok(Some(()))) => std::process::ExitCode::SUCCESS,
        _ => std::process::ExitCode::FAILURE,
    }
}

// One line to piccosta. None when piccosta is gone.
fn send(message: Value) -> Option<()> {
    writeln!(std::io::stdout().lock(), "{message}").ok()
}

// print() inside a script: each write goes to piccosta as it's made, up to MAX_PRINT_BYTES in all,
// counted as the lines that carry it (escapes too), so no one line passes MAX_LINE_BYTES.
#[derive(Default)]
struct Printer(usize);

impl PrintWriterCallback for Printer {
    fn stdout_write(&mut self, output: Cow<'_, str>) -> Result<(), MontyException> {
        let message = json!({"print": output});
        self.0 += message.to_string().len();
        if self.0 > MAX_PRINT_BYTES {
            return Err(MontyException::new(ExcType::MemoryError, Some(format!("the script printed more than {} MiB", MAX_PRINT_BYTES >> 20))));
        }
        send(message).ok_or_else(|| MontyException::new(ExcType::RuntimeError, Some(STOPPED.into())))
    }

    fn stdout_push(&mut self, end: char) -> Result<(), MontyException> {
        self.stdout_write(Cow::Owned(end.to_string()))
    }
}

// Run the script piccosta sends on the first line to the end, asking piccosta for each tool call and
// reading its answers, then say how it ended. Output: None when piccosta went away first.
fn run() -> Option<()> {
    let mut answers = std::io::stdin().lock().lines();
    let start: Value = serde_json::from_str(&answers.next()?.ok()?).ok()?;
    let code = start["code"].as_str()?;
    let names: Vec<String> = start["functions"].as_array()?.iter().filter_map(Value::as_str).map(str::to_string).collect();
    let mut printer = Printer::default();
    let mut suspensions = 0;
    // Each function goes in as an input of the same name; calling it suspends the run.
    let mut progress = MontyRun::new(format!("{PRELUDE}{code}"), "script.py", names.clone(), CompileOptions::default()).and_then(|runner| {
        let limits = ResourceLimits { max_suspensions: MAX_CALLS, ..ResourceLimits::default() };
        let inputs = names.iter().map(|name| MontyObject::function(name.clone(), None)).collect();
        runner.start(inputs, ResourceTracker::new(limits), PrintWriter::Callback(&mut printer))
    });
    loop {
        let step = match progress {
            Ok(step) => step,
            // Cut as the model would see it, so a huge exception message still fits on a line.
            Err(error) => return send(json!({"end": cut(&traceback(error))})),
        };
        let print = PrintWriter::Callback(&mut printer);
        // Every suspension but a wait on calls already made counts against the limit (refusals
        // and lookups too, so a loop of them can't run on).
        if !matches!(step, RunProgress::ResolveFutures(_) | RunProgress::Complete(_)) {
            suspensions += 1;
        }
        let over = MontyException::new(ExcType::RuntimeError, Some(format!("the script made more than {MAX_CALLS} calls")));
        progress = match step {
            RunProgress::Complete(_) => return send(json!({"end": null})),
            RunProgress::FunctionCall(call) if suspensions > MAX_CALLS => call.abort(over, print),
            RunProgress::OsCall(call) if suspensions > MAX_CALLS => call.abort(over, print),
            RunProgress::NameLookup(lookup) if suspensions > MAX_CALLS => lookup.abort(over, print),
            // A tool call: its keyword arguments as JSON, under one budget, go to piccosta, and the
            // script goes on.
            RunProgress::FunctionCall(call) => {
                let mut budget = Budget::default();
                let arguments: Result<Map<String, Value>, String> = match call.args.args().len() {
                    0 => call.args.kwargs().map(|(key, value)| Ok((budget.key(key)?, to_json(value, 0, &mut budget)?))).collect(),
                    _ => Err(format!("{}() takes keyword arguments only, as in {}(name=value)", call.function_name, call.function_name)),
                };
                match arguments {
                    Ok(mut arguments) => {
                        // None counts as not given.
                        arguments.retain(|_, value| !value.is_null());
                        send(json!({"call": call.call_id, "function": call.function_name, "arguments": arguments}))?;
                        call.resume_pending(print)
                    }
                    Err(error) => call.resume(ExtFunctionResult::Error(MontyException::new(ExcType::TypeError, Some(error))), print),
                }
            }
            // A missing module, a sleep: Monty's own error, as with no host.
            RunProgress::OsCall(call) if matches!(call.function_call, OsFunctionCall::Import(_) | OsFunctionCall::Sleep(_) | OsFunctionCall::SystemSleep(_) | OsFunctionCall::AsyncSleep(_) | OsFunctionCall::AsyncSystemSleep(_)) => {
                let refusal = call.function_call.on_no_handler();
                call.resume(ExtFunctionResult::Error(refusal), print)
            }
            // Files, environment and the rest: refused as Monty refuses them with no host, saying
            // what to use instead (its path alone, resolved from its virtual cwd /, misled a model).
            RunProgress::OsCall(call) => {
                let refusal = call.function_call.on_no_handler();
                let message = format!("{}; a script reaches files, the environment and processes only through the tool functions", refusal.message().unwrap_or_default());
                call.resume(ExtFunctionResult::Error(MontyException::new(refusal.exc_type(), Some(message))), print)
            }
            RunProgress::NameLookup(lookup) => lookup.resume(NameLookupResult::Undefined, print),
            // Every task in the script is waiting on calls: wait for piccosta's next answer.
            RunProgress::ResolveFutures(waiting) => {
                let answer: Value = serde_json::from_str(&answers.next()?.ok()?).ok()?;
                let call_id = u32::try_from(answer["call"].as_u64()?).ok()?;
                let result = match answer["error"].as_str() {
                    Some(error) => ExtFunctionResult::Error(MontyException::new(ExcType::RuntimeError, Some(error.to_string()))),
                    None => ExtFunctionResult::Return(MontyObject::string(answer["text"].as_str().unwrap_or_default().to_string())),
                };
                waiting.resume(vec![(call_id, result)], print)
            }
        };
    }
}

// An error's traceback, its line numbers moved back past the prelude.
fn traceback(error: MontyException) -> String {
    let mut frames = error.traceback().to_vec();
    for frame in &mut frames {
        frame.start.line = frame.start.line.saturating_sub(1);
        frame.end.line = frame.end.line.saturating_sub(1);
    }
    MontyException::with_traceback(error.exc_type(), error.message().map(str::to_string), frames).to_string()
}

// What one call's arguments have used of their budget so far: values visited and bytes of JSON.
#[derive(Default)]
struct Budget {
    values: usize,
    bytes: usize,
}

impl Budget {
    // Count a value's spelling as JSON, plus a byte for the comma or colon after it.
    fn charge(&mut self, spelled_bytes: usize) -> Result<(), String> {
        self.bytes += spelled_bytes + 1;
        match self.bytes > MAX_ARGUMENT_BYTES {
            true => Err(format!("arguments over {} MiB as JSON", MAX_ARGUMENT_BYTES >> 20)),
            false => Ok(()),
        }
    }

    // A keyword's or dict key's name, checked to be a str and charged as it's spelled.
    fn key(&mut self, key: ObjectRef) -> Result<String, String> {
        let key = key.as_str().ok_or("a keyword or dict key that isn't a str")?.to_string();
        self.charge(serde_json::to_string(&key).map_or(0, |spelled| spelled.len()))?;
        Ok(key)
    }
}

// A Python value as JSON: None, bools, ints, floats, strings, lists, tuples and dicts with str keys
// map across, each checked by its type before anything is read from it. Anything else, or a value
// past the budget (MAX_ARGUMENT_DEPTH, _VALUES, _BYTES), is refused with what went wrong.
fn to_json(value: ObjectRef, depth: usize, budget: &mut Budget) -> Result<Value, String> {
    budget.values += 1;
    if depth > MAX_ARGUMENT_DEPTH || budget.values > MAX_ARGUMENT_VALUES {
        return Err(format!("arguments nested more than {MAX_ARGUMENT_DEPTH} deep or holding more than {MAX_ARGUMENT_VALUES} values"));
    }
    let type_name = value.type_name();
    let wrong = || format!("a {type_name} can't go to a tool; pass str, int, float, bool, None, list or dict");
    let converted = match type_name {
        "NoneType" => Value::Null,
        "bool" => Value::Bool(value.as_bool().ok_or_else(wrong)?),
        "int" => json!(value.as_int().ok_or_else(wrong)?),
        "float" => json!(value.as_float().ok_or_else(wrong)?),
        "str" => json!(value.as_str().ok_or_else(wrong)?),
        "list" | "tuple" => Value::Array(value.items().ok_or_else(wrong)?.into_iter().map(|item| to_json(item, depth + 1, budget)).collect::<Result<_, _>>()?),
        "dict" => {
            let pairs = value.pairs().ok_or_else(wrong)?;
            let pair = |(key, value): (ObjectRef, ObjectRef)| Ok((budget.key(key)?, to_json(value, depth + 1, budget)?));
            Value::Object(pairs.into_iter().map(pair).collect::<Result<_, String>>()?)
        }
        _ => return Err(wrong()),
    };
    // A scalar is charged its whole spelling; a list or dict its brackets, its items having been
    // charged as they were converted.
    budget.charge(match &converted {
        Value::Array(_) | Value::Object(_) => 2,
        scalar => scalar.to_string().len(),
    })?;
    Ok(converted)
}
