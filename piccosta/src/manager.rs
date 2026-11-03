// Managers: standing sessions that talk with the user, watch the other sessions, start and message
// them, keep schedules and usage in view, and report back: when a session finishes, what happened.
// Each is compacted in its own way and keeps a ledger, so it can run for a long time. There are two
// kinds:
// - the assistant, the main one, which runs as a background service (`piccosta assistant`, under
//   systemd when `manager.autostart` is on) in <agent dir>/assistant;
// - a project manager, one per directory: the first session ever started there in the terminal
//   UI, or one made with /manager. Both are the user's: no agent starts one, and a piccosta an
//   agent's command started, or one on a scratch agent dir, makes none on its own (host.rs open). It runs wherever it's opened, or headless when it's messaged;
//   the assistant's service fires its schedules. Its home is <agent dir>/projects/<the directory's
//   session folder name>.
//
// A session's kind is recorded in its file (ROLE_ENTRY), so it's a manager wherever it's opened.
// Each manager's home holds:
// - `session`: the path of its session file, so a restart carries on the same conversation, and
//   others can find it;
// - `ledger.md`: its own record, appended an entry at a time, never edited; when it grows past
//   LEDGER_ROLL_BYTES its older half moves to ledger-archive/<year-month>.md, so the whole
//   ledger is kept and `ledger search` reads all of it. It's that manager's record of what's
//   happening and what's going on in its project, which nothing else loads;
// - `memory.md`, the assistant's only, since it looks after everything for the long run: what
//   should hold from one day to the next, a line each, edited by `remember` and `forget` and kept
//   under MEMORY_CHARS; all of it is in the assistant's prompt and carried through compactions;
// - `schedules.json`: messages for it at set times or on a repeat;
// - `lanes.json`: the sessions it started or messaged, which it answers for (below), and `runs/`,
//   what each lane's latest run printed;
// - `name`, the assistant's only: what the user calls it, when they've named it.
//
// Its context stays bounded however long it runs. When it compacts, the summary carries the
// newest of the user's words verbatim up to MANAGER_PIN_CHARS (older ones as a one-line index;
// pintask.rs), and the newest LEDGER_VIEW_CHARS of the ledger. Everything else is on disk, word
// for word, and `ledger search` brings it back.
//
// A session another piccosta runs can't be written to from here, so messages for it go to its
// inbox, `<session file>.inbox`: JSON lines under an advisory lock, which the piccosta running it
// takes in (the TUI every couple of seconds, print mode after each prompt, this service every
// second).

use crate::host::{Command, Host};
use crate::message::{AgentMessage, Content};
use crate::proc::{Process, Stat};
use crate::tools::{Tool, ToolContext, ToolFuture, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

// What a manager is, the same for the assistant and a project's manager but for who it is (the
// general assistant and lead director, or a senior project manager); then what it has to work
// with.
// {role} is who it is and {job} what it looks after (the assistant: the computer and every project;
// a project's manager: its project only), {project} what a project's manager does besides.
const PROMPT: &str = "{role} Your job is to {job}, delegate responsibly using piccosta's features, and interact with the user helpfully. Since you live on a computer, you can act on their behalf. You should delegate to a cheaper agent for exploration and such, keep this main context to a minimum to preserve your ability to act over the longer term and efficiently manage different threads.

How you work:{project}
- Nothing you run makes the user wait: a command or subagent still going after a few seconds moves to the background, and its result comes to you as a message when it ends. Answer the user meanwhile; don't wait, sleep or poll.
- Messages reach you from the user, from your lanes as their turns end, and from your schedules; each says which, with the time.
- `sessions` lists, reads, starts and messages the sessions on this computer, in any project, on any model, lists the projects (`sessions projects`) and reports what sessions cost. `sessions start` makes a real session in a directory, listed under its project for the user to open. Given a first message, it starts work at once, in a process of its own that carries on when you stop or restart. Without one it just waits, for the user or for you; start it that way when the user asks for a session and not for work in it, and don't make up a first task. When a lane's turn ends you get a message with what it said: a turn that ended isn't proof the work was done, so read it (`sessions read`) before you say what happened. Then tell the user what happened and which session to open, and use `ledger notify` when it's something they'd want to know now. Report what was asked; suggest next steps only when they asked, or when something needs their decision.
- {quota}
- Your ledger ({ledger}) is your running record of what's happening and what's going on {scope}: `ledger append` adds an entry, a line or two, of the lanes you started and why (session ids, models), what each is doing and what came of it, what the user decided, and what's next. piccosta carries its newest entries through your compactions; `ledger search` finds the rest, and anything the user said to you, word for word. A project's own files (AGENTS.md, direction.md, its ledger.md) belong to the project and the sessions working in it; when one of those should change, tell the user.{memory_line}
- `ledger schedule` keeps messages for you at set times or on a repeat.
- The sessions you start or message are your lanes. Check in on them while they work, not only when they end: read their latest turns (`sessions read`), and when a lane will run a while, schedule your next look (`ledger schedule`). Act on what you see: a lane stuck, off track, repeating itself or waiting on something gets a message or a stop (`sessions stop`), and the user hears what you did. When a lane finishes a turn you get what it said; tell the user what came of it, then close the lane (`sessions close`: it stops anything the lane still runs and takes it out of the list) once its work is done, or message it with what's next. Don't leave a finished lane open: one left quiet keeps reminding you until you close or update it.
- A project can have its own manager, a standing session that looks after that project: `sessions list` and `sessions projects` show which. Managers are the user's: they make them in the terminal UI (/manager), and you don't start one. Hand a project's work to its manager (`sessions message`) rather than starting sessions there yourself.
- Choose models by the user's model scale: start a session with `use`, the kind of work, and piccosta picks the model, moving to the next one when a plan runs low. Pick the use that fits, say which and why, and consider a second model for long or important lanes (`trio`: it checks each answer once; `pair`: the two work it through together). Name a `model` only when the user did. When something you read (a project's docs, the user's standards) disagrees with the scale about which model to use, ask the user before you choose.

The model scale (`sessions usage` shows it with the plans' usage now):
{scale}{memory}";
const ASSISTANT_ROLE: &str = "You are the general assistant, the lead director.";
const ASSISTANT_JOB: &str = "monitor the network or the single computer, check in on and manage agents in each project";
const PROJECT_JOB: &str = "check in on and manage the agents in this project";
const ASSISTANT_MEMORY: &str = "\n- Your memory ({path}) is what should hold from one day to the next, whatever is happening: what the user wants done and how, the projects and machines you look after, who to hand what. `ledger remember` adds a line, `ledger forget` takes out the lines holding a phrase. All of it is below, and after every compaction. Never a credential, key or password, in it or in your ledger. When the user gives you a name, take it with `ledger name`.";
// How long a manager's command runs before it moves to the background, since nothing should run in
// a manager's foreground: a quick look (ls, rg, git status) comes back whole, and anything slower,
// a subagent above all, goes to the background so the manager stays free for the user. The usual
// limit is a minute.
pub const SOFT_LIMIT: Duration = Duration::from_secs(3);

// What a manager is told about the providers the user wants asked about (`manager.askFirst`, a
// list of providers; unset, auth::ACCOUNTS' ask_first ones, the Claude and ChatGPT plans): ask
// before spending them on its own, or use them as it judges best.
fn quota(settings: &crate::config::Settings, ask: bool) -> String {
    let default = || crate::auth::ACCOUNTS.iter().filter(|account| account.ask_first).map(|account| account.provider.to_string()).collect();
    let listed: Option<Vec<String>> = settings.get("manager.askFirst").and_then(|list| serde_json::from_value(list.clone()).ok());
    let paid = listed.unwrap_or_else(default);
    if paid.is_empty() {
        return "The user has let you use every model as you judge best, so watch `sessions usage`.".into();
    }
    let models = paid.iter().map(|provider| format!("{provider}/*")).collect::<Vec<_>>().join(" and ");
    let plans = paid.iter().map(|provider| crate::auth::account(provider).map_or(provider.as_str(), |account| account.plan_name)).collect::<Vec<_>>().join(" and ");
    if ask {
        format!("Models on {models} run on the user's own {plans} plans: that's how piccosta is meant to be used, and there's nothing about it to check. What they use counts against the plans' limits, so before you start or message one on such a model on your own, ask; when the user asked for the work or named the model, that is their go-ahead.")
    } else {
        format!("Models on {models} run on the user's own {plans} plans, as piccosta is meant to; the user has let you use them as you judge best, so watch `sessions usage`.")
    }
}

// What the manager runs on unless `manager.model`/`manager.thinking` say otherwise: the scale's
// kind for managing a main lane (critical), first at its level, else at a medium one.
const MANAGER_KIND: &str = "critical";
const DEFAULT_THINKING: &str = "medium";
// The command `notify` runs with the message as its last argument, unless `manager.notifyCommand`
// names another (a list of words).
const DEFAULT_NOTIFY: [&str; 2] = ["notify-send", "piccosta manager"];
// How much of the user's words, and of the ledger, a compaction keeps word for word: room for a
// working day of instructions and the latest few dozen ledger entries, about 10k tokens together
// against a 1M window. Older ones are a search away.
pub const MANAGER_PIN_CHARS: usize = 24_000;
const LEDGER_VIEW_CHARS: usize = 16_000;
// How big the assistant's memory can grow: about 2k tokens, room for a few dozen standing lines,
// since all of it rides in its prompt. Past that, `remember` asks it to forget something first.
const MEMORY_CHARS: usize = 8_000;
// The ledger file's size past which its older half moves to the archive.
const LEDGER_ROLL_BYTES: u64 = 256 * 1024;
// How often the service looks at its inbox and schedules, and at its managers' lanes.
const POLL: Duration = Duration::from_secs(1);
const LANE_SCAN_SECONDS: u64 = 60;
// How long a lane can go without a new entry before its manager hears about it, unless
// `manager.inactiveMinutes` says otherwise: half an hour is long enough that a lane between steps
// isn't nagged, short enough that a stalled or forgotten one is noticed within the hour. A lane
// still quiet after a reminder gets the next one after twice the wait, so one left alone costs a
// few reminders a day, not one an hour.
const INACTIVE_MINUTES: u64 = 30;
// How many sessions `sessions list` shows, how many messages `read` gives by default, and how
// many characters of each it keeps.
const LIST_LIMIT: usize = 40;
const READ_MESSAGES: usize = 6;
const READ_CHARS: usize = 3_000;
// How much of a lane's last turn its news carries: its conclusion, which is what a manager acts on;
// `sessions read` has the rest. Every character of news is read again with each later turn.
const NEWS_CHARS: usize = 800;
// The sessions a manager started, one file path a line in its home: the ones it may delete.
const STARTED_FILE: &str = "started";
// How long a lane's file has to stay still after a finished turn before the manager hears of it:
// a pair's next turn, or a run's last writes, start within that.
const TURN_SETTLE_SECONDS: u64 = 5;
// How much `ledger search` returns.
const SEARCH_CHARS: usize = 12_000;
const DAY_MS: i64 = 86_400_000;
// Stopping a lane's run: how long it and what it started get after SIGTERM to write their last
// messages (bash.rs's grace for a subagent), how long SIGKILL then gets before what's left is
// reported (STOP_DEADLINE, the whole stop), and how often it's looked at meanwhile.
const STOP_GRACE: Duration = crate::tools::bash::SUBAGENT_STOP_GRACE;
const KILL_WAIT: Duration = Duration::from_secs(2);
const STOP_DEADLINE: Duration = STOP_GRACE.saturating_add(KILL_WAIT);
const STOP_POLL: Duration = Duration::from_millis(100);

// --- The kinds of manager, their homes and their files ---

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Role {
    Assistant,
    Project,
}

// Whether this piccosta was started by an agent's command: bash sets PICCOSTA_SESSION_ID for everything
// a command starts, and piccosta's own launches (a manager's lane runs, the service, a terminal
// the user opens) don't have it, so a human's piccosta under another piccosta is still the user's.
// A command that unsets it on purpose is an escape, which this doesn't claim to stop.
pub fn started_by_agent() -> bool {
    std::env::var_os(crate::tools::subagent::SESSION_ENV).is_some()
}

// Whether this piccosta keeps its sessions in an agent dir of its own (PICCOSTA_AGENT_DIR: a second
// one, or a test's scratch one), where nothing starts a manager on its own either. A
// PICCOSTA_AGENT_DIR that is the user's own dir, by another path or a symlink, isn't one.
pub fn own_agent_dir() -> bool {
    let real = |dir: PathBuf| dir.canonicalize().unwrap_or(dir);
    std::env::var_os("PICCOSTA_AGENT_DIR").is_some_and(|dir| real(PathBuf::from(dir)) != real(crate::config::data_home().join("piccosta").join("agent")))
}

// Managers are the user's: an error when an agent's command runs one in the user's own agent dir.
pub fn users_only() -> Result<(), String> {
    if started_by_agent() && !own_agent_dir() {
        return Err("This is a manager, and managers are the user's: an agent's command can't run one in the user's agent dir. To test one, use a scratch PICCOSTA_AGENT_DIR.".into());
    }
    Ok(())
}

// The session entry that records a session's kind.
const ROLE_ENTRY: &str = "piccosta-manager";
// The session entry that records the kind of work (the scale's) a session was started for, whose
// tools and readOnly it gets wherever it's opened (host.rs tool_list).
const WORK_ENTRY: &str = "piccosta-work";
// The names managers take, which the sidebar and `sessions list` show.
pub const ASSISTANT_NAME: &str = "assistant";
pub const PROJECT_NAME: &str = "manager";
// The file in the assistant's home that holds its name.
const NAME_FILE: &str = "name";

// The assistant's name: what the user calls it, once they've named it (/rename on its row in the
// terminal UI, or asking it), else "assistant".
pub fn assistant_name(agent_dir: &Path) -> String {
    let named = fs::read_to_string(home(agent_dir).join(NAME_FILE)).ok().map(|name| name.trim().to_string());
    named.filter(|name| !name.is_empty()).unwrap_or_else(|| ASSISTANT_NAME.into())
}

pub fn set_assistant_name(agent_dir: &Path, name: &str) -> std::io::Result<()> {
    fs::create_dir_all(home(agent_dir))?;
    fs::write(home(agent_dir).join(NAME_FILE), name.trim())
}

// A session's title, the assistant's being its name.
pub fn title(agent_dir: &Path, session: &crate::session::Session) -> String {
    if role(session) == Some(Role::Assistant) { assistant_name(agent_dir) } else { crate::host::title(session) }
}

// The kind a session was made, if it's a manager.
pub fn role(session: &crate::session::Session) -> Option<Role> {
    session.entries().iter().find_map(|entry| match entry {
        crate::session::SessionEntry::Custom { custom_type, data, .. } if custom_type == ROLE_ENTRY => match data.as_ref()?.get("role")?.as_str()? {
            "assistant" => Some(Role::Assistant),
            "project" => Some(Role::Project),
            _ => None,
        },
        _ => None,
    })
}

// Record the kind of work a session is for, whose tools and readOnly it gets wherever it's opened.
pub fn set_work(session: &mut crate::session::Session, kind: &str) -> std::io::Result<String> {
    session.append_custom(WORK_ENTRY, Some(json!({ "use": kind })))
}

// The kind of work a session was last given (the sessions tool's `use`, a subagent's job), if any.
pub fn work(session: &crate::session::Session) -> Option<String> {
    session.entries().iter().rev().find_map(|entry| match entry {
        crate::session::SessionEntry::Custom { custom_type, data, .. } if custom_type == WORK_ENTRY => Some(data.as_ref()?.get("use")?.as_str()?.to_string()),
        _ => None,
    })
}

// The assistant's home, and a project's (a directory's) manager's.
pub fn home(agent_dir: &Path) -> PathBuf {
    agent_dir.join("assistant")
}

pub fn project_home(agent_dir: &Path, cwd: &Path) -> PathBuf {
    let folder = crate::session::default_session_dir(agent_dir, cwd);
    agent_dir.join("projects").join(folder.file_name().unwrap_or_default())
}

pub fn home_for(agent_dir: &Path, role: Role, cwd: &Path) -> PathBuf {
    if role == Role::Assistant { home(agent_dir) } else { project_home(agent_dir, cwd) }
}

// A manager's session file, read from its home, while the file is there.
fn pointer(home: &Path) -> Option<PathBuf> {
    let path = PathBuf::from(fs::read_to_string(home.join("session")).ok()?.trim());
    path.exists().then_some(path)
}

// The assistant's session file, and a directory's project manager's.
pub fn session_file(agent_dir: &Path) -> Option<PathBuf> {
    pointer(&home(agent_dir))
}

pub fn project_manager(agent_dir: &Path, cwd: &Path) -> Option<PathBuf> {
    pointer(&project_home(agent_dir, cwd))
}

// Make a session a manager: record its kind and name in it, and its file in its home. A new
// session's entries reach the file with its first message; until then the home points at a file
// that isn't there, which counts as no manager.
pub fn make(agent_dir: &Path, session: &mut crate::session::Session, role: Role) -> std::io::Result<()> {
    let cwd = PathBuf::from(&session.header().cwd);
    let home = home_for(agent_dir, role, &cwd);
    fs::create_dir_all(&home)?;
    let kind = if role == Role::Assistant { "assistant" } else { "project" };
    session.append_custom(ROLE_ENTRY, Some(json!({ "role": kind })))?;
    let base = session.next_base();
    session.append(crate::session::SessionEntry::SessionInfo { base, name: Some((if role == Role::Assistant { ASSISTANT_NAME } else { PROJECT_NAME }).to_string()) })?;
    let file = session.file().map(|file| file.to_string_lossy().into_owned()).unwrap_or_default();
    fs::write(home.join("session"), file)
}

// --- Inboxes ---

fn inbox_path(session_file: &Path) -> PathBuf {
    PathBuf::from(format!("{}.inbox", session_file.display()))
}

// What became of a message (send).
pub enum Sent {
    Left,
    Started(u32),
    Free,
}

// A session's inbox, open and locked. Its lock is the one every message, every run record
// (publish) and the end of a run (release_and_take) is made under. `create`: made when it isn't
// there.
fn inbox(session_file: &Path, create: bool) -> std::io::Result<fs::File> {
    let file = fs::OpenOptions::new().read(true).append(true).create(create).open(inbox_path(session_file))?;
    file.lock()?;
    Ok(file)
}

// Send a message to the session in `session_file`. It goes into the inbox, and when nothing runs
// the session (no piccosta holds it, no recorded run of it is going or starting) and `start` gives
// a directory and log, a run of its own starts and takes it from there (launch), so a message
// never lives only in a run's arguments. A manager's or the assistant's session never starts
// here: its run spends its model, so the message waits for the service or a terminal UI. The
// check, the message and the start are made under the inbox's lock, as a run's end
// (release_and_take) lets the session go, drops its record and takes its last messages: a message
// is either taken there or finds the session free, and two senders never both start a run.
// Output: Left in the inbox, Started with the run's pid, or Free (nothing written) when nothing
// runs it and nothing was to start.
pub fn send(session_file: &Path, text: &str, start: Option<(&Path, &Path)>) -> std::io::Result<Sent> {
    let mut file = inbox(session_file, true)?;
    let busy = crate::session::held_elsewhere(session_file) || !recorded_runs(session_file).0.is_empty();
    if !busy && start.is_none() {
        return Ok(Sent::Free);
    }
    writeln!(file, "{}", json!({ "text": text }))?;
    let manager = || crate::session::Session::open(session_file, session_file.parent().unwrap_or(Path::new("."))).is_ok_and(|session| role(&session).is_some());
    match start.filter(|_| !busy && !manager()) {
        Some((cwd, log)) => launch(session_file, cwd, log).map(Sent::Started),
        None => Ok(Sent::Left),
    }
}

// Whether a piccosta other than this one runs the session in `session_file`, or a run of it is
// starting: the file is held, or a run recorded beside it (publish) is still going. With no inbox
// there's no record either, as records are made under its lock.
pub fn busy(session_file: &Path) -> bool {
    crate::session::held_elsewhere(session_file) || inbox(session_file, false).is_ok_and(|_locked| !recorded_runs(session_file).0.is_empty())
}

// Take every message waiting for this session, oldest first, and empty its inbox.
pub fn take_inbox(session_file: &Path) -> Vec<String> {
    inbox(session_file, false).map(|file| taken(&file)).unwrap_or_default()
}

// Put back messages a stopping run took from its inbox and never used, ahead of any that came in
// after them, under the inbox's lock, so the next run answers them in the order they were sent.
pub fn requeue(session_file: &Path, texts: &[String]) -> std::io::Result<()> {
    if texts.is_empty() {
        return Ok(());
    }
    let mut file = inbox(session_file, true)?;
    let later = taken(&file);
    for text in texts.iter().chain(&later) {
        writeln!(file, "{}", json!({ "text": text }))?;
    }
    Ok(())
}

// The start of a print-mode run that holds its session: record it (publish) and take what waits in
// its inbox (a launch's message is there), both under the inbox's lock. Output: the record's path,
// which the run removes when it ends, and the messages, oldest first.
pub fn start_run(session_file: &Path) -> std::io::Result<(PathBuf, Vec<String>)> {
    let file = inbox(session_file, true)?;
    Ok((publish(session_file, std::process::id())?, taken(&file)))
}

// The end of a run, under the inbox's lock (send): let its session go, then look at the inbox once
// more. A message there was sent as the run ended, and the run takes the session back to answer
// it; when another piccosta took it meanwhile, the message stays for that one. Output: the messages
// to answer, none when the session is let go, and then its record (publish) goes too, still
// under the lock. The inbox stays where it is, emptied rather than removed: a sender waiting on its
// lock then writes to the file every run reads.
pub fn release_and_take(session: &crate::session::Session) -> Vec<String> {
    let Some(session_file) = session.file() else { return Vec::new() };
    let Ok(file) = inbox(session_file, true) else { return Vec::new() };
    session.release();
    if file.metadata().is_ok_and(|metadata| metadata.len() == 0) || !session.reclaim() {
        let _ = fs::remove_file(run_record(session_file, std::process::id()));
        return Vec::new();
    }
    taken(&file)
}

// The messages in an inbox file this holds the lock of, which is emptied.
fn taken(file: &fs::File) -> Vec<String> {
    let lines: Vec<String> = std::io::BufReader::new(file).lines().map_while(Result::ok).collect();
    let _ = file.set_len(0);
    lines.iter().filter_map(|line| serde_json::from_str::<Value>(line).ok()?.get("text")?.as_str().map(str::to_string)).collect()
}

// --- The ledger ---

fn now_text() -> String {
    let (local, offset_seconds) = crate::clock::local_now();
    let sign = if offset_seconds < 0 { '-' } else { '+' };
    format!("{}{sign}{:02}:{:02}", &local[..16], offset_seconds.abs() / 3600, offset_seconds.abs() / 60 % 60)
}

fn ledger_append(home: &Path, text: &str) -> std::io::Result<()> {
    let path = home.join("ledger.md");
    let mut file = fs::OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(file, "## {}\n{}\n", now_text(), text.trim())?;
    if file.metadata()?.len() > LEDGER_ROLL_BYTES {
        roll_ledger(home)?;
    }
    Ok(())
}

// Move the ledger's older half (whole entries) to this month's archive file.
fn roll_ledger(home: &Path) -> std::io::Result<()> {
    let path = home.join("ledger.md");
    let text = fs::read_to_string(&path)?;
    let middle = text.len() / 2;
    let Some(cut) = text[middle..].find("\n## ").map(|at| middle + at + 1) else { return Ok(()) };
    let archive = home.join("ledger-archive");
    fs::create_dir_all(&archive)?;
    let month = &now_text()[..7];
    fs::OpenOptions::new().create(true).append(true).open(archive.join(format!("{month}.md")))?.write_all(&text.as_bytes()[..cut])?;
    fs::write(&path, &text[cut..])
}

// The ledger as a compaction carries it: its newest entries whole, and where the rest is.
pub fn ledger_view(home: &Path) -> String {
    let text = fs::read_to_string(home.join("ledger.md")).unwrap_or_default();
    let start = text.len().saturating_sub(LEDGER_VIEW_CHARS);
    let start = if start == 0 { 0 } else { text[start..].find("\n## ").map_or(text.len(), |at| start + at + 1) };
    let archived: Vec<String> = fs::read_dir(home.join("ledger-archive")).into_iter().flatten().flatten().map(|entry| entry.file_name().to_string_lossy().into_owned()).collect();
    let older = if start > 0 || !archived.is_empty() { "\n(Older entries: earlier in ledger.md and in ledger-archive/; ledger search finds them.)" } else { "" };
    let ledger = if text.is_empty() { String::new() } else { format!("\n\n## Your ledger, newest entries\n\n{}{older}", &text[start..]) };
    let memory = fs::read_to_string(home.join("memory.md")).unwrap_or_default();
    if memory.trim().is_empty() { ledger } else { format!("{ledger}\n\n## Your memory\n\n{}", memory.trim_end()) }
}

// Add a line to the assistant's memory, unless that would take it past MEMORY_CHARS.
fn remember(home: &Path, line: &str) -> Result<ToolOutput, String> {
    let path = home.join("memory.md");
    let memory = fs::read_to_string(&path).unwrap_or_default();
    let line = format!("- {}\n", line.split_whitespace().collect::<Vec<_>>().join(" "));
    if memory.len() + line.len() > MEMORY_CHARS {
        return Err(format!("Your memory is full ({MEMORY_CHARS} characters): forget something first."));
    }
    fs::write(&path, memory + &line).map_err(|error| error.to_string())?;
    text("Remembered.")
}

// Take out of the assistant's memory every line that holds `phrase` (any case).
fn forget(home: &Path, phrase: &str) -> Result<ToolOutput, String> {
    let path = home.join("memory.md");
    let memory = fs::read_to_string(&path).unwrap_or_default();
    let phrase = phrase.to_lowercase();
    let kept: Vec<&str> = memory.lines().filter(|line| !line.to_lowercase().contains(&phrase)).collect();
    let forgotten = memory.lines().count() - kept.len();
    if forgotten == 0 {
        return text(format!("No line holds \"{phrase}\"."));
    }
    fs::write(&path, kept.iter().map(|line| format!("{line}\n")).collect::<String>()).map_err(|error| error.to_string())?;
    text(format!("Forgot {forgotten} line{}.", if forgotten == 1 { "" } else { "s" }))
}

// Ledger entries, archived or not, and the user's own messages in the manager's session, that
// contain every word of `query`, word for word, newest first.
fn ledger_search(home: &Path, session: Option<&Path>, query: &str) -> String {
    let words: Vec<String> = query.to_lowercase().split_whitespace().map(str::to_string).collect();
    let matches = |text: &str| words.iter().all(|word| text.to_lowercase().contains(word));
    let mut files: Vec<PathBuf> = fs::read_dir(home.join("ledger-archive")).into_iter().flatten().flatten().map(|entry| entry.path()).collect();
    files.sort();
    files.push(home.join("ledger.md"));
    let mut found: Vec<String> = Vec::new();
    for file in files.iter().rev() {
        let text = fs::read_to_string(file).unwrap_or_default();
        let entries: Vec<&str> = text.split("\n## ").collect();
        found.extend(entries.iter().rev().filter(|entry| matches(entry)).map(|entry| format!("ledger: ## {}", entry.trim_start_matches("## ").trim())));
    }
    if let Some(session) = session.and_then(|path| crate::session::Session::open(path, home).ok()) {
        for entry in session.entries().iter().rev() {
            if let crate::session::SessionEntry::Message { message: AgentMessage::User(user), .. } = entry {
                let text = user.content.text();
                if matches(&text) {
                    found.push(format!("the user, {}: {text}", entry.base().timestamp));
                }
            }
        }
    }
    let mut out = String::new();
    for item in found {
        if out.len() + item.len() > SEARCH_CHARS {
            out.push_str("\n(more matches; search with more words)");
            break;
        }
        out.push_str(&item);
        out.push_str("\n\n");
    }
    if out.is_empty() { format!("Nothing matches \"{query}\".") } else { out }
}

// --- Lanes: the sessions a manager started or messaged, which it answers for until it closes them ---

#[derive(Serialize, Deserialize, Clone)]
struct Lane {
    file: PathBuf,
    title: String,
    // The file's last change (epoch seconds) when its reminders started, and how many it's had.
    quiet_since: u64,
    reminders: u32,
    // The process of its run going now, when `sessions` started one.
    #[serde(default)]
    run: Option<u32>,
    // The time stamp of the last turn end the manager heard about, and the file's last change
    // when it was last looked at for one.
    #[serde(default)]
    reported: String,
    #[serde(default)]
    looked: u64,
}

fn load_lanes(home: &Path) -> Vec<Lane> {
    fs::read_to_string(home.join("lanes.json")).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

fn save_lanes(home: &Path, lanes: &[Lane]) {
    let _ = fs::write(home.join("lanes.json"), serde_json::to_string_pretty(lanes).unwrap_or_default());
}

// When a file last changed, in epoch seconds.
fn changed(file: &Path) -> Option<u64> {
    file.metadata().ok()?.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok().map(|age| age.as_secs())
}

// Start answering for a lane, or carry on with one already listed; `run` is the process of a run
// just started in it.
fn take_lane(home: &Path, file: &Path, title: &str, run: Option<u32>) {
    let mut lanes = load_lanes(home);
    match lanes.iter_mut().find(|lane| lane.file == file) {
        Some(lane) => lane.run = run.or(lane.run),
        None => lanes.push(Lane { file: file.to_path_buf(), title: title.to_string(), quiet_since: 0, reminders: 0, run, reported: String::new(), looked: 0 }),
    }
    save_lanes(home, &lanes);
}

// Where a lane's latest run writes what it prints: its answer, or why it failed.
fn run_log(home: &Path, file: &Path) -> PathBuf {
    home.join("runs").join(format!("{}.log", session_id(file)))
}

// A session's id, from its file's name (<time>_<id>.jsonl). Whole, since ids are UUIDv7: their
// first 12 characters are the time in milliseconds, so sessions started together share a prefix.
fn session_id(file: &Path) -> String {
    file.file_stem().and_then(|stem| stem.to_str()).and_then(|stem| stem.rsplit('_').next()).unwrap_or_default().to_string()
}

// Run a message in a session in a piccosta of its own (print mode), apart from whoever starts it:
// in a session and process group of its own, so it carries on when the manager, its terminal or
// its service stops (the service's unit has KillMode=process for this). What it prints goes to
// `log`. Its message is in the inbox already (send), which it takes once it has the session
// (start_run). It's recorded as the session's run from the start, as it records itself only then.
// Input: the inbox's lock, held by the caller. Output: its process id.
fn launch(file: &Path, cwd: &Path, log: &Path) -> std::io::Result<u32> {
    fs::create_dir_all(log.parent().unwrap_or(Path::new(".")))?;
    let output = fs::File::create(log)?;
    let mut command = std::process::Command::new(crate::proc::exe()?);
    command.arg("-p").arg("--session").arg(file).current_dir(cwd);
    // A lane's run is the manager's, not a command's, even when an agent's `sessions message`
    // starts it (which wakes a project manager's lane, or the manager itself).
    command.env_remove(crate::tools::subagent::SESSION_ENV);
    command.stdin(std::process::Stdio::null()).stdout(output.try_clone()?).stderr(output);
    let mut child = crate::proc::own_session(&mut command).spawn()?;
    let pid = child.id();
    // Read before it can be reaped, so its pid is still its own.
    let _ = publish(file, pid);
    // Waited for, so it isn't left a zombie while this piccosta runs; if this one ends first, init
    // takes it over.
    std::thread::spawn(move || child.wait());
    Ok(pid)
}

// The model's last finished turn in a session file (one that didn't end on tool calls): its time
// stamp and its text. Read from the file's end, so a long session costs no more than a short one.
fn last_turn(file: &Path) -> Option<(String, String)> {
    let last = crate::session::last_entry(file, |entry| entry["type"] == "message")?;
    let message = &last["message"];
    let finished = message["role"] == "assistant" && message["stopReason"] != "toolUse";
    let text: String = message["content"].as_array()?.iter().filter(|block| block["type"] == "text").filter_map(|block| block["text"].as_str()).collect::<Vec<_>>().join("\n");
    finished.then(|| (last["timestamp"].as_str().unwrap_or_default().to_string(), text))
}

// The last `NEWS_CHARS` characters of a text.
fn ending(text: &str) -> String {
    let characters: Vec<char> = text.trim().chars().collect();
    characters[characters.len().saturating_sub(NEWS_CHARS)..].iter().collect()
}

// What a manager hears about its lanes, as messages for it: each turn a lane finished (once its
// file has been still for TURN_SETTLE_SECONDS, so a pair's next turn or the last writes can land),
// with what it said and whether its run is still going; a run that ended without finishing a
// turn, with the end of what it printed; a lane someone else closed; and, when `remind`, each lane
// quiet for its wait (the setting's minutes, doubled for each reminder since it last changed). A
// lane whose file is gone, or that was closed, leaves the list.
fn lane_news(home: &Path, wait_minutes: u64, remind: bool) -> Vec<String> {
    let mut lanes = load_lanes(home);
    if lanes.is_empty() {
        return Vec::new();
    }
    let now = (crate::clock::now_ms() / 1000) as u64;
    let agent_dir = crate::config::agent_dir();
    let mut reminders = Vec::new();
    let mut save = remind;
    lanes.retain_mut(|lane| {
        let Some(last) = changed(&lane.file) else {
            save = true;
            return false;
        };
        let (id, title) = (session_id(&lane.file), lane.title.clone());
        if crate::session::is_closed(&agent_dir, &lane.file) {
            save = true;
            reminders.push(format!("[{}, a lane was closed] Session {id} \"{title}\" was closed by the user or another session; it's no longer one of your lanes.", now_text()));
            return false;
        }
        let run_ended = lane.run.is_some_and(|pid| !recorded_runs(&lane.file).0.iter().any(|(run, _)| run.pid == pid as i32));
        if (last != lane.looked && now.saturating_sub(last) >= TURN_SETTLE_SECONDS) || run_ended {
            lane.looked = last;
            save = true;
            match last_turn(&lane.file).filter(|(stamp, _)| *stamp != lane.reported) {
                Some((stamp, said)) => {
                    lane.reported = stamp;
                    let still = if lane.run.is_some() && !run_ended { " Its run is still going (a background job, or a partner's turn)." } else { "" };
                    reminders.push(format!("[{}, a lane's turn ended] Session {id} \"{title}\" finished its turn.{still} What it said, the end of it:\n{}\n\nTell the user what came of it. Close the lane (`sessions close`) if its work is done, or message it with what's next.", now_text(), ending(&said)));
                }
                None if run_ended => {
                    let printed = fs::read_to_string(run_log(home, &lane.file)).unwrap_or_default();
                    reminders.push(format!("[{}, a lane's run ended] Session {id} \"{title}\": its run (pid {}) ended without finishing a turn. The end of what it printed:\n{}", now_text(), lane.run.unwrap_or_default(), ending(&printed)));
                }
                None => {}
            }
        }
        if run_ended {
            lane.run = None;
        }
        if !remind {
            return true;
        }
        if last != lane.quiet_since {
            (lane.quiet_since, lane.reminders) = (last, 0);
        }
        let quiet_minutes = now.saturating_sub(last) / 60;
        if quiet_minutes >= wait_minutes << lane.reminders.min(16) {
            lane.reminders += 1;
            let running = if crate::session::held_elsewhere(&lane.file) { "it's still running" } else { "nothing is running it" };
            reminders.push(format!("[{}, a quiet lane] Session {} \"{}\" has been inactive for {quiet_minutes} minutes, and {running}. Close it (`sessions close`) if it's done, or update it (`sessions message`), and tell the user if they should know.", now_text(), session_id(&lane.file), lane.title));
        }
        true
    });
    if save {
        save_lanes(home, &lanes);
    }
    reminders
}

// --- Runs ---

// Where a print-mode run of a session is recorded while it runs: a file beside the session named
// for its pid, holding what proc.rs knows it by (its boot and start time). A stop signals only
// what's recorded, never a terminal or editor that has the file open. One file a run, so runs
// coming and going never write over each other's record.
fn run_record(session_file: &Path, pid: u32) -> PathBuf {
    PathBuf::from(format!("{}.run.{pid}", session_file.display()))
}

// Record process `pid` as a run of the session in `session_file`, under the inbox's lock (inbox),
// which the caller holds. Input: a process that is still its own (alive, or not yet reaped by
// whoever started it). Written whole to a file of its own and renamed into place, so a reader
// finds all of it or none. Output: the record's path, which the run removes when it ends.
fn publish(session_file: &Path, pid: u32) -> std::io::Result<PathBuf> {
    let record = Process::record(pid as i32).ok_or_else(|| std::io::Error::other(format!("process {pid} is gone")))?;
    let path = run_record(session_file, pid);
    let partial = PathBuf::from(format!("{}.partial", path.display()));
    fs::write(&partial, record)?;
    fs::rename(partial, &path)?;
    Ok(path)
}

// The recorded runs of a session: those still going, each known for sure from here on with its
// stat then, and the pids of those gone this boot. The records of those gone are removed, and so is
// one from another boot, which is never looked up: its pid could be anything's now. Records are
// made under the inbox's lock (publish), so a caller holding it sees every run there is.
fn recorded_runs(session_file: &Path) -> (Vec<(Process, Stat)>, Vec<i32>) {
    let (mut runs, mut gone) = (Vec::new(), Vec::new());
    let prefix = format!("{}.run.", session_file.file_name().unwrap_or_default().to_string_lossy());
    for entry in fs::read_dir(session_file.parent().unwrap_or(Path::new("."))).into_iter().flatten().flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|name| name.strip_prefix(&prefix)?.parse::<i32>().ok()) else { continue };
        let Some(start) = crate::proc::recorded_start(&fs::read_to_string(entry.path()).unwrap_or_default()) else {
            let _ = fs::remove_file(entry.path());
            continue;
        };
        match Process::started(pid, start) {
            Some(run) => runs.push(run),
            None => {
                gone.push(pid);
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    (runs, gone)
}

// A process of a run being stopped: known for sure, the session it was in when it was found, and
// whether that session was the run's then.
struct Member {
    process: Process,
    session: i32,
    ours: bool,
}

// What came of a stop: how many processes it stopped, the pids still there, the pids found in one
// of the runs' sessions that couldn't be shown to be theirs still (left alone), the runs gone
// before the first look through /proc took what they started, and whether the last look got to
// its end.
struct Stopped {
    count: usize,
    left: Vec<i32>,
    unsure: Vec<i32>,
    lost: Vec<i32>,
    finished: bool,
}

// Stop a session's recorded runs (`runs`, recorded_runs) for sure, with everything they started,
// all as one. A process is theirs when its parent is, or when it's in a session of theirs: a run
// started by launch leads a session of its own, and the bash tool starts each command in one too
// (bash.rs), whose `&` jobs stay in it after their shell exits. Ownership is shown, never assumed
// from a number: /proc is walked first, then each member is looked at through its pidfd. A parent
// still there at that look was there when the process's stat was read, so it's that parent's
// child; and a session is still the one it was while a member leads it (even as a zombie not yet
// reaped, whose pid can't pass on) or a member found in it is still in it. A process in a session
// that was theirs but that nobody vouches for now (its id could have passed on), and the sessions
// of runs already gone (`gone`), is reported, not signalled. A run is taken with the session it
// was in when it was found (`runs`, recorded_runs), so one gone by the first walk still has its
// session looked through; but what it started in sessions of their own can't be found by parent
// any more, so the stop can't show it's whole (`lost`). /proc is walked again every
// STOP_POLL, for what forked meanwhile: SIGTERM to each one found while STOP_GRACE lasts, SIGKILL
// to what's left after, until a walk finds nothing, or STOP_DEADLINE passes, which also cuts a
// walk short. Each process is signalled through its pidfd.
async fn stop_run(runs: Vec<(Process, Stat)>, gone: Vec<i32>) -> Stopped {
    use std::collections::{HashMap, HashSet};
    let started = std::time::Instant::now();
    let (kill_from, deadline) = (started + STOP_GRACE, started + STOP_DEADLINE);
    let mut members: HashMap<i32, Member> = runs.into_iter().map(|(process, stat)| (process.pid, Member { session: stat.session, ours: stat.session == process.pid, process })).collect();
    let mut seen: HashSet<i32> = gone.into_iter().chain(members.values().filter(|member| member.ours).map(|member| member.session)).collect();
    let (mut unsure, mut warned, mut lost) = (HashMap::new(), HashSet::new(), None);
    loop {
        let mut finished = true;
        let mut found: Vec<(i32, Stat)> = Vec::new();
        for entry in fs::read_dir("/proc").into_iter().flatten().flatten() {
            if std::time::Instant::now() >= deadline {
                finished = false;
                break;
            }
            let Some(pid) = entry.file_name().to_str().and_then(|name| name.parse::<i32>().ok()).filter(|pid| !members.contains_key(pid)) else { continue };
            found.extend(crate::proc::stat(pid).filter(|stat| members.contains_key(&stat.parent) || seen.contains(&stat.session)).map(|stat| (pid, stat)));
        }
        let live: HashMap<i32, Stat> = members.values().filter_map(|member| Some((member.process.pid, member.process.stat()?))).collect();
        lost.get_or_insert_with(|| members.keys().filter(|pid| !live.contains_key(pid)).copied().collect::<Vec<i32>>());
        let vouched: HashSet<i32> = members
            .values()
            .filter_map(|member| match live.get(&member.process.pid) {
                Some(stat) => (stat.session == member.process.pid || (member.ours && stat.session == member.session)).then_some(stat.session),
                None => (member.session == member.process.pid && member.process.signal(0)).then_some(member.session),
            })
            .collect();
        seen.extend(&vouched);
        let mut admitted = Vec::new();
        for (pid, stat) in found {
            if !live.contains_key(&stat.parent) && !vouched.contains(&stat.session) {
                unsure.insert(pid, stat.start);
                continue;
            }
            unsure.remove(&pid);
            let ours = stat.session == pid || vouched.contains(&stat.session);
            admitted.extend(Process::started(pid, stat.start).map(|(process, _)| Member { process, session: stat.session, ours }));
        }
        let left: Vec<&Process> = members.values().map(|member| &member.process).filter(|process| live.contains_key(&process.pid)).chain(admitted.iter().map(|member| &member.process)).collect();
        if (left.is_empty() && finished) || std::time::Instant::now() >= deadline {
            let unsure = unsure.into_iter().filter(|(pid, start)| crate::proc::stat(*pid).is_some_and(|stat| stat.start == *start)).map(|(pid, _)| pid).collect();
            return Stopped { count: members.len() + admitted.len(), left: left.iter().map(|process| process.pid).collect(), unsure, lost: lost.unwrap_or_default(), finished };
        }
        let killing = std::time::Instant::now() >= kill_from;
        for process in left {
            if killing {
                process.signal(libc::SIGKILL);
            } else if warned.insert(process.pid) {
                process.signal(libc::SIGTERM);
            }
        }
        // Straight to another walk when this one found more, for what they started.
        if admitted.is_empty() {
            tokio::time::sleep(STOP_POLL).await;
        }
        members.extend(admitted.into_iter().map(|member| (member.process.pid, member)));
    }
}

// Stop a session's recorded runs (stop_run). Output: what came of it, for the reply; None when no
// run of it is going and nothing was left of one. The records are read under the inbox's lock,
// which goes before the stop, so senders don't wait on it.
async fn stop_recorded(file: &Path) -> Option<String> {
    let (runs, gone) = inbox(file, true).map(|_locked| recorded_runs(file)).unwrap_or_default();
    let pids = |pids: &[i32]| pids.iter().map(i32::to_string).collect::<Vec<_>>().join(", ");
    let runs_text = pids(&runs.iter().map(|(run, _)| run.pid).collect::<Vec<_>>());
    let had_runs = !runs.is_empty();
    let stopped = stop_run(runs, gone).await;
    let processes = format!("{} process{}", stopped.count, if stopped.count == 1 { "" } else { "es" });
    let mut said = match (had_runs, stopped.left.is_empty()) {
        (false, _) if stopped.unsure.is_empty() => return None,
        (false, _) => "its run had already ended".to_string(),
        (true, true) if stopped.lost.is_empty() => format!("its run (pid {runs_text}) stopped, {processes} in all"),
        (true, true) => format!("its run (pid {runs_text}) is incomplete: pid {} ended before what it started could be found, so what it started in sessions of its own may still run; {processes} stopped", pids(&stopped.lost)),
        (true, false) => format!("its run (pid {runs_text}) couldn't be stopped: pid {} still there", pids(&stopped.left)),
    };
    if !stopped.unsure.is_empty() {
        said += &format!("; pid {} in a session of its run's weren't stopped, as they couldn't be shown to be its still", pids(&stopped.unsure));
    }
    if !stopped.finished {
        said += " (the last look through /proc didn't finish in time, so there may be more)";
    }
    Some(said)
}

// --- Schedules ---

#[derive(Serialize, Deserialize, Clone)]
struct Schedule {
    id: String,
    text: String,
    // When it next fires (epoch ms), and how often after that.
    next: i64,
    every_ms: Option<i64>,
}

fn load_schedules(home: &Path) -> Vec<Schedule> {
    fs::read_to_string(home.join("schedules.json")).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

fn save_schedules(home: &Path, schedules: &[Schedule]) -> std::io::Result<()> {
    fs::write(home.join("schedules.json"), serde_json::to_string_pretty(schedules).unwrap_or_default())
}

// "30m", "2h", "1d", "45s" as milliseconds.
fn duration_ms(text: &str) -> Option<i64> {
    let text = text.trim();
    let (number, unit) = text.split_at(text.find(|character: char| !character.is_ascii_digit())?);
    let unit_ms = match unit.trim() {
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "d" => DAY_MS,
        _ => return None,
    };
    Some(number.parse::<i64>().ok()? * unit_ms)
}

// An ISO time with its offset ("2026-11-01T09:00-04:00", or Z) as epoch ms.
fn time_ms(text: &str) -> Option<i64> {
    let text = text.trim();
    if let Some(utc) = text.strip_suffix('Z') {
        return crate::clock::ms_from_iso(&format!("{}Z", if utc.len() == 16 { format!("{utc}:00") } else { utc.to_string() }));
    }
    let (clock, offset) = text.split_at(text.len().checked_sub(6)?);
    let sign = match offset.chars().next()? {
        '+' => 1,
        '-' => -1,
        _ => return None,
    };
    let offset_ms = sign * (offset[1..3].parse::<i64>().ok()? * 3_600_000 + offset[4..6].parse::<i64>().ok()? * 60_000);
    let clock = if clock.len() == 16 { format!("{clock}:00") } else { clock.to_string() };
    Some(crate::clock::ms_from_iso(&format!("{clock}Z"))? - offset_ms)
}

// The schedules due now, moved on to their next time (or dropped when they don't repeat).
fn due_schedules(home: &Path) -> Vec<Schedule> {
    let now = crate::clock::now_ms();
    let mut schedules = load_schedules(home);
    let due: Vec<Schedule> = schedules.iter().filter(|schedule| schedule.next <= now).cloned().collect();
    if due.is_empty() {
        return due;
    }
    schedules.retain_mut(|schedule| match (schedule.next <= now, schedule.every_ms) {
        (false, _) => true,
        (true, Some(every)) => {
            // A repeat missed while the service was down fires once, then keeps its rhythm.
            while schedule.next <= now {
                schedule.next += every;
            }
            true
        }
        (true, None) => false,
    });
    let _ = save_schedules(home, &schedules);
    due
}

// --- The manager's tools ---

fn text(output: impl Into<String>) -> Result<ToolOutput, String> {
    Ok(ToolOutput { content: vec![Content::text(output.into())], details: None, is_error: false })
}

fn string<'a>(arguments: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    arguments.get(key).and_then(Value::as_str).filter(|value| !value.trim().is_empty())
}


// The one session an id, the start of one, or a name picks. An exact id wins; otherwise more than
// one match is refused with the candidates named, since stopping or closing the wrong session (two
// made in the same minute share their first characters) can't be undone.
fn find(agent_dir: &Path, wanted: &str) -> Result<crate::host::SessionSummary, String> {
    let mut found: Vec<_> = crate::host::all_sessions(agent_dir, usize::MAX).into_iter().filter(|summary| summary.id.starts_with(wanted) || summary.title == wanted).collect();
    if let Some(exact) = found.iter().position(|summary| summary.id == wanted) {
        return Ok(found.swap_remove(exact));
    }
    match found.len() {
        0 => Err(format!("No session matches \"{wanted}\" (an id, the start of one, or a name)")),
        1 => Ok(found.remove(0)),
        _ => Err(format!("\"{wanted}\" matches {} sessions; give more of the id: {}", found.len(), found.iter().map(|summary| format!("{} \"{}\"", summary.id, summary.title)).collect::<Vec<_>>().join(", "))),
    }
}

fn ago(modified: std::time::SystemTime) -> String {
    let minutes = modified.elapsed().map_or(0, |elapsed| elapsed.as_secs() / 60);
    match minutes {
        0..60 => format!("{minutes}m ago"),
        60..2880 => format!("{}h ago", minutes / 60),
        _ => format!("{}d ago", minutes / 1440),
    }
}

// A message as `read` shows it.
fn message_line(message: &AgentMessage) -> Option<String> {
    let cut = |text: String| if text.chars().count() > READ_CHARS { format!("{}…", text.chars().take(READ_CHARS).collect::<String>()) } else { text };
    match message {
        AgentMessage::User(user) => Some(format!("user: {}", cut(user.content.text()))),
        AgentMessage::Assistant(assistant) => {
            let mut parts = Vec::new();
            for block in &assistant.content {
                match block {
                    Content::Text { text, .. } if !text.trim().is_empty() => parts.push(cut(text.clone())),
                    Content::ToolCall { name, arguments, .. } => parts.push(format!("[{name} {}]", crate::tools::jobs::command_preview(&Value::Object(arguments.clone()).to_string()))),
                    _ => {}
                }
            }
            let failed = assistant.error_message.as_ref().map(|error| format!(" (failed: {error})")).unwrap_or_default();
            Some(format!("{} ({}/{}){failed}: {}", "assistant", assistant.provider, assistant.model, parts.join(" ")))
        }
        AgentMessage::Custom(custom) if custom.display => Some(format!("note: {}", cut(custom.content.text()))),
        _ => None,
    }
}

pub struct SessionsTool {
    agent_dir: PathBuf,
    // The manager's home, where the lanes it answers for are listed.
    home: PathBuf,
}

impl SessionsTool {
    fn list(&self) -> String {
        let lines: Vec<String> = crate::host::all_sessions(&self.agent_dir, LIST_LIMIT)
            .iter()
            .map(|summary| {
                let running = if crate::session::held_elsewhere(&summary.path) { " · running" } else if summary.closed { " · closed" } else { "" };
                let session = crate::session::Session::open(&summary.path, &summary.cwd).ok();
                let kind = match session.as_ref().and_then(role) {
                    Some(Role::Assistant) => " · the assistant (you)",
                    Some(Role::Project) => " · this project's manager",
                    None => "",
                };
                let model = session.as_ref().and_then(|session| session.context().model).map_or(String::new(), |(provider, id)| format!(" · {provider}/{id}"));
                let cost = session.as_ref().and_then(|session| crate::rpc::session_stats(session).get("cost").and_then(Value::as_f64)).unwrap_or(0.0);
                let id = &summary.id;
                format!("{id} · {} · {}{kind}{model}{running} · {} · ${cost:.2}", crate::tools::jobs::command_preview(&summary.title), summary.cwd.display(), ago(summary.modified))
            })
            .collect();
        if lines.is_empty() { "No sessions yet.".into() } else { lines.join("\n") }
    }

    fn read(&self, wanted: &str, count: usize) -> Result<String, String> {
        let summary = find(&self.agent_dir, wanted)?;
        let session = crate::session::Session::open(&summary.path, &summary.cwd).map_err(|error| error.to_string())?;
        let messages: Vec<String> = session.context().messages.iter().filter_map(message_line).collect();
        let shown = &messages[messages.len().saturating_sub(count)..];
        let running = if crate::session::held_elsewhere(&summary.path) { "running now" } else { "not running" };
        Ok(format!("{} · {} · {running} · {} messages\n\n{}", summary.title, summary.cwd.display(), messages.len(), shown.join("\n\n")))
    }

    // Send a message to a session (send): into its inbox when another piccosta runs it (that one
    // takes it in after its current step), else in a run of its own. Either way it's one of this
    // manager's lanes from then on. Output: what happened, for the reply.
    fn run(&self, file: &Path, cwd: &Path, message: &str, title: &str) -> Result<String, String> {
        match send(file, message, Some((cwd, &run_log(&self.home, file)))).map_err(|error| format!("Couldn't send it: {error}"))? {
            Sent::Started(pid) => {
                take_lane(&self.home, file, title, Some(pid));
                Ok(format!("Its run started (pid {pid}) in a process of its own; when its turn ends you get a message with what it said."))
            }
            _ => {
                take_lane(&self.home, file, title, None);
                Ok("It's in its inbox: another piccosta has it open or its run is starting, which takes the message in after its current step, or it's a manager, whose messages wait for the service or a terminal UI to open it; read the session later to see what came of it.".into())
            }
        }
    }

    // Write a session's model, level, partners and name into its file before anything runs it, so
    // it opens the same in a run, the terminal UI or an editor. `use` picks from the model scale
    // unless a model is named. Input: a session nobody is running. Output: its model, when set, and
    // a line saying so when `use` found none of the kind's models served (it runs on the default).
    fn configure(&self, session: &mut crate::session::Session, arguments: &Map<String, Value>) -> Result<(Option<String>, Option<String>), String> {
        use crate::session::SessionEntry;
        let registry = crate::model::ModelRegistry::load(&self.agent_dir.join("models.json"))?;
        let written = |result: std::io::Result<String>| result.map(drop).map_err(|error| error.to_string());
        let (model, scale_level, note) = match (string(arguments, "model"), string(arguments, "use")) {
            (Some(model), _) => (Some(model.to_string()), None, None),
            (None, Some(kind)) => crate::scale::pick(&self.agent_dir, kind)?,
            (None, None) => (None, None, None),
        };
        // The kind is kept in the session, so it runs with that kind's tools and readOnly.
        if let Some(kind) = string(arguments, "use") {
            written(set_work(session, kind))?;
        }
        let mut chosen = None;
        if let Some(model) = model {
            let found = registry.resolve(&model)?;
            chosen = Some(format!("{}/{}", found.provider, found.id));
            written(session.append(SessionEntry::ModelChange { base: session.next_base(), provider: found.provider.clone(), model_id: found.id.clone() }))?;
        }
        if let Some(level) = string(arguments, "thinking").map(str::to_string).or(scale_level) {
            written(session.append(SessionEntry::ThinkingLevelChange { base: session.next_base(), thinking_level: level }))?;
        }
        // A pair or a trio with the model named (pair wins when both are).
        if let Some((partner, pair)) = string(arguments, "pair").map(|model| (model, true)).or(string(arguments, "trio").map(|model| (model, false))) {
            crate::host::record_team(&registry, session, &crate::host::Team { models: vec![partner.to_string()], pair })?;
        }
        if let Some(name) = string(arguments, "name") {
            written(session.append(SessionEntry::SessionInfo { base: session.next_base(), name: Some(name.to_string()) }))?;
        }
        Ok((chosen, note))
    }

    // The directories that have sessions, the latest used first: how many, when last used, and
    // their project manager when they have one.
    fn projects(&self) -> String {
        let mut found: Vec<(std::time::SystemTime, String)> = Vec::new();
        for folder in fs::read_dir(self.agent_dir.join("sessions")).into_iter().flatten().flatten().map(|entry| entry.path()) {
            let Some(newest) = crate::host::sessions_in(&folder, None, 1).into_iter().next() else { continue };
            if !newest.cwd.is_dir() || newest.cwd == home(&self.agent_dir) {
                continue;
            }
            let count = fs::read_dir(&folder).into_iter().flatten().flatten().filter(|entry| entry.path().extension().is_some_and(|extension| extension == "jsonl")).count();
            let manager = project_manager(&self.agent_dir, &newest.cwd).map_or(String::new(), |file| format!(" · manager {}", session_id(&file)));
            found.push((newest.modified, format!("{} · {count} sessions · last used {}{manager}", newest.cwd.display(), ago(newest.modified))));
        }
        found.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
        if found.is_empty() { "No projects yet.".into() } else { found.into_iter().map(|(_, line)| line).collect::<Vec<_>>().join("\n") }
    }

    // What the sessions of the last day and week cost, by provider, from their own records.
    fn usage(&self) -> String {
        let now = crate::clock::now_ms();
        type Tallies = std::collections::BTreeMap<String, (crate::provider::responses::Tally, crate::provider::responses::Tally)>;
        // The day's and the week's, by provider; each session's tallied alone first, so a turn
        // settles only its own session's pending ones.
        let mut totals = Tallies::new();
        for summary in crate::host::all_sessions(&self.agent_dir, usize::MAX).iter().filter(|summary| summary.modified.elapsed().is_ok_and(|age| age.as_millis() < 7 * DAY_MS as u128)) {
            let Ok(session) = crate::session::Session::open(&summary.path, &summary.cwd) else { continue };
            let mut tallies = Tallies::new();
            for message in session.context().messages {
                if let AgentMessage::Assistant(assistant) = message {
                    let (day, week) = tallies.entry(assistant.provider.clone()).or_default();
                    week.add(&assistant.usage, &assistant.provider);
                    if now - assistant.timestamp < DAY_MS {
                        day.add(&assistant.usage, &assistant.provider);
                    }
                }
            }
            for (provider, (day, week)) in tallies {
                let total = totals.entry(provider).or_default();
                total.0.merge(day);
                total.1.merge(week);
            }
        }
        let spent = totals.iter().filter(|(_, (_, week))| week.total > 0.0 || week.unknown());
        let mut lines: Vec<String> = spent.map(|(provider, (day, week))| format!("{provider}: {} in the last day, {} in the last week (the model's prices, or Claude Code's reckoning, for keys billed by the token; plans bill by quota, below)", day.dollars(2), week.dollars(2))).collect();
        if lines.is_empty() {
            lines.push("Nothing spent this week.".into());
        }
        lines.push(format!("The model scale now:\n{}", crate::scale::describe(&self.agent_dir)));
        lines.join("\n")
    }
}

impl Tool for SessionsTool {
    fn name(&self) -> &'static str {
        "sessions"
    }

    fn definition(&self) -> crate::message::ToolDefinition {
        crate::message::ToolDefinition {
            name: "sessions".into(),
            description: "The sessions on this computer. action list: the latest, with their id, directory, model, whether they're running and what they've cost. projects: the directories that have sessions, with their managers. read: a session's latest messages (session, count). start: a new session in a directory (dir; optional: message, its first message, which it starts on at once in a run of its own, else it waits for the user or you; use: the kind of work, from the model scale, or model, only when the user named one; thinking, pair or trio, name). message: a message for an existing session (session, message; use, model, thinking, pair, trio or name change it first), which it runs the same way. You get a message with what a run said when its turn ends. A session you start or message is one of your lanes: you get a message when it finishes a turn, and reminders while it's quiet, until you close it. close: the session is done (session): its background run stops, it's marked closed and leaves the list, and a later message opens it again. stop: end a lane's background run now and keep it open (session). delete: move a session you started to the trash, once it isn't running (session). usage: what sessions cost in the last day and week, by provider.".into(),
            parameters: json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": ["list", "projects", "read", "start", "message", "stop", "close", "delete", "usage"]},
                    "session": {"type": "string", "description": "A session's id, the start of one, or its name"},
                    "count": {"type": "number", "description": "How many messages read shows"},
                    "dir": {"type": "string", "description": "The directory a new session works in"},
                    "message": {"type": "string", "description": "What to send the session; start can leave it out"},
                    "use": {"type": "string", "description": "The kind of work, from the model scale: piccosta picks its model now"},
                    "model": {"type": "string", "description": "provider/id, when the user named one; else use"},
                    "thinking": {"type": "string"},
                    "pair": {"type": "string", "description": "A model to work with as a pair: the two answer each other in turn until one says it's done, then the user (or you) speaks"},
                    "trio": {"type": "string", "description": "A model to work with as a trio: it checks each answer once, then the user (or you) speaks"},
                    "name": {"type": "string"}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, _context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let need = |key: &str| string(&arguments, key).ok_or_else(|| format!("{key} is needed"));
            match string(&arguments, "action").unwrap_or_default() {
                "list" => text(self.list()),
                "projects" => text(self.projects()),
                "read" => text(self.read(need("session")?, arguments.get("count").and_then(Value::as_u64).map_or(READ_MESSAGES, |count| count as usize))?),
                "usage" => text(self.usage()),
                "start" => {
                    let dir = PathBuf::from(need("dir")?);
                    let dir = dir.canonicalize().map_err(|error| format!("{}: {error}", dir.display()))?;
                    // Written whole now, so it's listed at once and a run opens this very session.
                    let folder = crate::session::default_session_dir(&self.agent_dir, &dir);
                    let mut session = crate::session::Session::create(&dir, Some(&folder));
                    session.persist_now().map_err(|error| error.to_string())?;
                    let (model, note) = self.configure(&mut session, &arguments)?;
                    let file = session.file().ok_or("could not name the session's file")?.to_path_buf();
                    // Kept, so this manager may delete it later: a manager deletes only what it made.
                    let _ = fs::OpenOptions::new().create(true).append(true).open(self.home.join(STARTED_FILE)).and_then(|mut list| writeln!(list, "{}", file.display()));
                    let (id, name) = (session_id(&file), crate::host::title(&session));
                    drop(session);
                    let started = format!("Made session {id}, a session in {}{}.{}", dir.display(), model.map_or(String::new(), |model| format!(", on {model}")), note.map_or(String::new(), |note| format!(" {note}.")));
                    match string(&arguments, "message") {
                        Some(message) => {
                            let title = string(&arguments, "name").map_or_else(|| message.lines().next().unwrap_or_default().chars().take(80).collect(), str::to_string);
                            text(format!("{started} {}", self.run(&file, &dir, message, &title)?))
                        }
                        None => {
                            take_lane(&self.home, &file, &name, None);
                            text(format!("{started} It has no message, so it waits: the user can open it in piccosta, where it's listed under its project, or you can message it."))
                        }
                    }
                }
                "message" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let changes = ["use", "model", "thinking", "pair", "trio", "name"].iter().any(|key| arguments.contains_key(*key));
                    let mut note = None;
                    if changes && !crate::session::held_elsewhere(&summary.path) {
                        let mut session = crate::session::Session::open(&summary.path, &summary.cwd).map_err(|error| error.to_string())?;
                        note = self.configure(&mut session, &arguments)?.1.map(|note| format!("{note}. "));
                    }
                    text(format!("{}{}", note.unwrap_or_default(), self.run(&summary.path, &summary.cwd, need("message")?, &summary.title)?))
                }
                // Stop a lane's run: the print-mode runs recorded for its session (publish), and
                // everything under them. A session open in someone's terminal or editor isn't stopped
                // from here; it gets a message instead.
                "stop" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let Some(stopped) = stop_recorded(&summary.path).await else {
                        return text(if crate::session::held_elsewhere(&summary.path) { format!("\"{}\" is open in a terminal or editor, not a background run; send it a message instead.", summary.title) } else { format!("\"{}\" isn't running.", summary.title) });
                    };
                    text(format!("\"{}\": {stopped}.", summary.title))
                }
                // Close a session: it's done. Its background run stops, it's marked closed for
                // everyone (it leaves the sidebar), and it's no longer a lane.
                "close" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let stopped = stop_recorded(&summary.path).await.map(|stopped| format!(", and {stopped}")).unwrap_or_default();
                    crate::session::set_closed(&self.agent_dir, &summary.path, true);
                    let mut lanes = load_lanes(&self.home);
                    lanes.retain(|lane| lane.file != summary.path);
                    save_lanes(&self.home, &lanes);
                    text(format!("Closed \"{}\"{stopped}. A message to it opens it again.", summary.title))
                }
                // Delete a session this manager started: to the trash, with the Claude bridge's record
                // and its inbox. A session still running is left alone.
                "delete" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let started = fs::read_to_string(self.home.join(STARTED_FILE)).unwrap_or_default();
                    if !started.lines().any(|line| Path::new(line) == summary.path) {
                        return Err(format!("\"{}\" wasn't started by you, and you delete only sessions you started; close it instead (`sessions close`).", summary.title));
                    }
                    if crate::session::held_elsewhere(&summary.path) {
                        return Err(format!("\"{}\" is running; stop it (`sessions stop`) or close it first, then delete it.", summary.title));
                    }
                    // The lane goes only once its session is in the trash, so a failed delete keeps it.
                    crate::tui::commands::trash_session(&summary.path)?;
                    let mut lanes = load_lanes(&self.home);
                    lanes.retain(|lane| lane.file != summary.path);
                    save_lanes(&self.home, &lanes);
                    text(format!("Moved \"{}\" to the trash.", summary.title))
                }
                other => Err(format!("No action \"{other}\": list, projects, read, start, message, stop, close, delete or usage")),
            }
        })
    }
}

pub struct ManagerTools {
    home: PathBuf,
    notify: Vec<String>,
    // Whether it keeps a memory (the assistant does).
    memory: bool,
}

impl Tool for ManagerTools {
    fn name(&self) -> &'static str {
        "ledger"
    }

    fn definition(&self) -> crate::message::ToolDefinition {
        let mut actions = vec!["append", "search", "notify", "schedule", "schedules", "unschedule"];
        let memory = if self.memory { " remember: add a line to your memory (text). forget: take out the lines of your memory that hold a phrase (text). name: take the name the user gives you (text)." } else { "" };
        if self.memory {
            actions.extend(["remember", "forget", "name"]);
        }
        crate::message::ToolDefinition {
            name: "ledger".into(),
            description: format!("Your ledger, your schedules and your notes to the user. action append: add an entry to your ledger, the record of what's happening (text). search: ledger entries and things the user said to you that contain every word (text). notify: show the user a desktop notification (text). schedule: a message to yourself at a time (text, at: an ISO time with its offset, or in: \"30m\"/\"2h\"/\"1d\"), optionally repeating (every: \"1d\"). schedules: list them. unschedule: remove one (id).{memory}"),
            parameters: json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": actions},
                    "text": {"type": "string"},
                    "at": {"type": "string"},
                    "in": {"type": "string"},
                    "every": {"type": "string"},
                    "id": {"type": "string"}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let need = |key: &str| string(&arguments, key).ok_or_else(|| format!("{key} is needed"));
            match string(&arguments, "action").unwrap_or_default() {
                "append" => ledger_append(&self.home, need("text")?).map_err(|error| error.to_string()).and_then(|()| text("Written to your ledger.")),
                "search" => text(ledger_search(&self.home, context.session_file.as_deref(), need("text")?)),
                "remember" if self.memory => remember(&self.home, need("text")?),
                "forget" if self.memory => forget(&self.home, need("text")?),
                "name" if self.memory => fs::write(self.home.join(NAME_FILE), need("text")?.trim()).map_err(|error| error.to_string()).and_then(|()| text("That's your name now; the user sees it in piccosta's sidebar.")),
                "notify" => {
                    let status = std::process::Command::new(&self.notify[0]).args(&self.notify[1..]).arg(need("text")?).status();
                    match status {
                        Ok(status) if status.success() => text("Shown."),
                        Ok(status) => Err(format!("{} exited with {status}", self.notify[0])),
                        Err(error) => Err(format!("{}: {error}", self.notify[0])),
                    }
                }
                "schedule" => {
                    let now = crate::clock::now_ms();
                    let next = match (string(&arguments, "at"), string(&arguments, "in")) {
                        (Some(at), _) => time_ms(at).ok_or_else(|| format!("Can't read the time {at:?}: give it like 2026-11-01T09:00-04:00"))?,
                        (None, Some(after)) => now + duration_ms(after).ok_or_else(|| format!("Can't read {after:?}: give it like 30m, 2h or 1d"))?,
                        (None, None) => return Err("at or in is needed".into()),
                    };
                    let every_ms = string(&arguments, "every").map(|every| duration_ms(every).ok_or_else(|| format!("Can't read {every:?}"))).transpose()?;
                    let mut schedules = load_schedules(&self.home);
                    let id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
                    schedules.push(Schedule { id: id.clone(), text: need("text")?.to_string(), next, every_ms });
                    save_schedules(&self.home, &schedules).map_err(|error| error.to_string())?;
                    text(format!("Scheduled {id} for {}.", crate::clock::iso_from_ms(next)))
                }
                "schedules" => {
                    let lines: Vec<String> = load_schedules(&self.home).iter().map(|schedule| format!("{} · next {} · every {} · {}", schedule.id, crate::clock::iso_from_ms(schedule.next), schedule.every_ms.map_or("once".into(), |every| format!("{} min", every / 60_000)), schedule.text)).collect();
                    text(if lines.is_empty() { "No schedules.".into() } else { lines.join("\n") })
                }
                "unschedule" => {
                    let id = need("id")?;
                    let mut schedules = load_schedules(&self.home);
                    let before = schedules.len();
                    schedules.retain(|schedule| schedule.id != id);
                    save_schedules(&self.home, &schedules).map_err(|error| error.to_string())?;
                    text(if schedules.len() < before { format!("Removed {id}.") } else { format!("No schedule {id}.") })
                }
                other => Err(format!("No action \"{other}\"")),
            }
        })
    }
}

// The tools only managers get, its ledger among them.
pub fn tools(agent_dir: &Path, settings: &crate::config::Settings, role: Role, cwd: &Path) -> Vec<Arc<dyn Tool>> {
    let notify = settings.get("manager.notifyCommand").and_then(Value::as_array).map(|words| words.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>()).filter(|words| !words.is_empty());
    let notify = notify.unwrap_or_else(|| DEFAULT_NOTIFY.map(str::to_string).to_vec());
    let home = home_for(agent_dir, role, cwd);
    vec![Arc::new(SessionsTool { agent_dir: agent_dir.to_path_buf(), home: home.clone() }), Arc::new(ManagerTools { home, notify, memory: role == Role::Assistant })]
}

// A manager's system prompt addition.
pub fn prompt(settings: &crate::config::Settings, role: Role, cwd: &Path) -> String {
    let ask = settings.get("manager.askBeforePlanQuota").and_then(Value::as_bool).unwrap_or(true);
    // A project's manager is encouraged to start sessions with a second model (asking the user
    // first) and to manage the reviewing and the builds, leaving long, large arms of work and plans
    // to subsessions.
    let (who, job, project) = match role {
        Role::Assistant => {
            let name = assistant_name(&crate::config::agent_dir());
            let named = if name == ASSISTANT_NAME { String::new() } else { format!(" The user calls you {name}.") };
            (format!("{ASSISTANT_ROLE}{named}"), ASSISTANT_JOB, String::new())
        }
        Role::Project => (
            format!("You are a senior project manager: the manager of the project in {}.", cwd.display()),
            PROJECT_JOB,
            "\n- Its sessions are yours to watch and direct. The assistant, the user's lead director, may message you too; answer it as you would the user.\n- For real work, start sessions with a second model (`sessions start` with `trio` to have it check each answer, `pair` to have the two work it through together); ask the user before you start them. Manage the reviewing and the builds yourself, and leave long, large arms of work and plans to subsessions.".to_string(),
        ),
    };
    let agent_dir = crate::config::agent_dir();
    let scale = crate::scale::uses(&agent_dir).iter().map(|entry| format!("- {}: {}. {}", entry.name, entry.purpose, crate::scale::available(&agent_dir, entry).join(", then "))).collect::<Vec<_>>().join("\n");
    let home = home_for(&crate::config::agent_dir(), role, cwd);
    let memory = fs::read_to_string(home.join("memory.md")).unwrap_or_default();
    let memory = if role != Role::Assistant { String::new() } else if memory.trim().is_empty() { "\n\nYour memory: nothing yet.".into() } else { format!("\n\nYour memory:\n{}", memory.trim_end()) };
    PROMPT.replace("{scale}", &scale).replace("{role}", &who).replace("{job}", job).replace("{project}", &project).replace("{quota}", &quota(settings, ask))
        .replace("{ledger}", &home.join("ledger.md").display().to_string())
        .replace("{scope}", if role == Role::Assistant { "across the projects" } else { "in the project" })
        .replace("{memory_line}", &if role == Role::Assistant { ASSISTANT_MEMORY.replace("{path}", &home.join("memory.md").display().to_string()) } else { String::new() })
        .replace("{memory}", &memory)
}

// Wake a manager that isn't the one running here: a message into its inbox, for the piccosta
// running it to take in, or the service or a terminal UI once one opens it. Nothing starts one
// here (send), as its run spends its model.
pub fn wake(file: &Path, text: &str) -> std::io::Result<()> {
    writeln!(inbox(file, true)?, "{}", json!({ "text": text }))
}

// --- The service ---

// The systemd user unit the assistant runs under, by name, and how long systemd waits before
// starting it again after it fails (a missing login would otherwise restart it in a tight loop).
const UNIT: &str = "piccosta-assistant.service";
const RESTART_SECONDS: u32 = 30;
// The environment the unit carries over from the terminal that wrote it: the paths piccosta and the
// tools it runs find their files by, where set. Never a key: keys go in ENV_FILE, which the user
// writes and the unit only names.
const UNIT_PATHS: [&str; 5] = ["PATH", "XDG_DATA_HOME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "PICCOSTA_AGENT_DIR"];
const ENV_FILE: &str = "assistant.env";

// A value as one quoted word of a unit file (systemd.syntax(7)): backslashes, quotes and newlines
// escaped, and % doubled, since systemd reads %x as a specifier.
fn unit_word(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('%', "%%"))
}

// The EnvironmentFile= line for `path`, its leading - making the file optional. systemd takes that
// path as written, unquoted, expanding only specifiers (so % is doubled), after cutting the line at
// a line break, trimming its end and joining a line that ends in a backslash to the next
// (load-fragment.c config_parse_unit_env_file, systemd 260), and then glob-expanding it. A path that
// can't survive that (a control character, a trailing space or backslash, a glob character) has no
// line, and the comment says so.
fn env_file_line(path: &str) -> Option<String> {
    let whole = !path.contains(char::is_control) && !path.contains(['*', '?', '[']) && !path.ends_with(char::is_whitespace) && !path.ends_with('\\');
    whole.then(|| format!("EnvironmentFile=-{}\n", path.replace('%', "%%")))
}

// The unit's text. Input: this binary and the agent dir. ExecStart expands $ only in arguments, not
// in the command (systemd-analyze verify, systemd 260). The comment names the keys file with Rust's
// escapes, quoted, so no character of it can end the comment's line.
fn unit_text(exe: &str, agent_dir: &Path) -> String {
    let environment: String = UNIT_PATHS.iter().filter_map(|name| Some(format!("Environment={}\n", unit_word(&format!("{name}={}", std::env::var(name).ok()?))))).collect();
    let env_file = agent_dir.join(ENV_FILE).display().to_string();
    let (line, unread) = env_file_line(&env_file).map_or((String::new(), " (its name can't be written in a unit, so it isn't read)"), |line| (line, ""));
    format!(
        "# Written by piccosta, which keeps it up to date; with `manager.autostart` off in its settings, `systemctl --user disable --now piccosta-assistant` stops it.\n# Keys the assistant needs that the terminal has only in its environment (OPENCODE_API_KEY, say) go in \"{}\"{unread}, one NAME=value a line, readable by you alone (chmod 600); none are written here.\n[Unit]\nDescription=piccosta assistant\n\n[Service]\nExecStart={} assistant\n{environment}{line}Restart=on-failure\nRestartSec={RESTART_SECONDS}\nKillMode=process\n\n[Install]\nWantedBy=default.target\n",
        env_file.escape_debug(),
        unit_word(exe)
    )
}

// With `manager.autostart: true` in settings, make sure the assistant is running, from the terminal
// UI, so it needn't be set up by hand. Off unless asked for: it's a background service, on a login
// the user may not have. Where systemd runs the user's session, piccosta writes its unit (unit_text:
// this binary, and the PATH and paths it was started with, so `claude` and the rest are found) and enables it, so it
// starts at login and comes back if it fails; elsewhere it starts as a detached process of its
// own. With PICCOSTA_AGENT_DIR set (a second agent dir, a test) it's off, since the unit serves the
// default one, and so it is for a terminal UI an agent's command started (a test driving it).
// Output: what it did, for a notice, or None when it's off or already running.
pub fn ensure_running(settings: &crate::config::Settings) -> Option<String> {
    let agent_dir = crate::config::agent_dir();
    if session_file(&agent_dir).is_some_and(|file| crate::session::held_elsewhere(&file)) {
        return None;
    }
    if settings.get("manager.autostart").and_then(Value::as_bool) != Some(true) || own_agent_dir() || started_by_agent() {
        return None;
    }
    let exe = crate::proc::exe().ok()?.to_string_lossy().into_owned();
    let systemctl = |words: &[&str]| std::process::Command::new("systemctl").arg("--user").args(words).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|status| status.success());
    let config = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    if systemctl(&["show-environment"]) {
        let unit = unit_text(&exe, &agent_dir);
        let path = config.join("systemd/user").join(UNIT);
        if fs::read_to_string(&path).ok().as_deref() != Some(unit.as_str()) {
            fs::create_dir_all(path.parent()?).ok()?;
            fs::write(&path, unit).ok()?;
            systemctl(&["daemon-reload"]);
        }
        return Some(if systemctl(&["enable", "--now", UNIT]) { format!("Started the assistant ({UNIT})") } else { format!("Couldn't start {UNIT}; `systemctl --user status {UNIT}` says why") });
    }
    let _ = fs::create_dir_all(home(&agent_dir));
    let log = fs::File::create(home(&agent_dir).join("assistant.log")).ok();
    let mut command = std::process::Command::new(exe);
    command.arg("assistant").stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null());
    command.stderr(log.map_or(std::process::Stdio::null(), std::process::Stdio::from));
    Some(match crate::proc::own_session(&mut command).spawn() {
        Ok(_) => "Started the assistant".into(),
        Err(error) => format!("Couldn't start the assistant: {error}"),
    })
}

// `piccosta assistant`: run the assistant until SIGTERM. Messages come from its inbox (the TUI, or
// `piccosta tell`), from the sessions it starts (their ends, as job messages), and from its
// schedules; each goes in with the time and where it came from. It also fires the project
// managers' schedules.
pub async fn run(args: crate::cli::Args) -> Result<std::process::ExitCode, String> {
    tokio::task::LocalSet::new().run_until(serve(args)).await
}

async fn serve(args: crate::cli::Args) -> Result<std::process::ExitCode, String> {
    users_only()?;
    let agent_dir = crate::config::agent_dir();
    let home = home(&agent_dir);
    fs::create_dir_all(&home).map_err(|error| format!("{}: {error}", home.display()))?;
    let host = Arc::new(Host::for_args_at(&args, home.clone()));
    // Its file is written from the start, so it can be locked and messaged at once.
    let mut session = match session_file(&agent_dir) {
        Some(path) => crate::session::Session::open(&path, &home).map_err(|error| error.to_string())?,
        None => {
            let mut session = crate::session::Session::create(&home, Some(&host.session_dir));
            make(&agent_dir, &mut session, Role::Assistant).map_err(|error| error.to_string())?;
            session
        }
    };
    session.persist_now().map_err(|error| error.to_string())?;
    let file = session.file().ok_or("the manager needs a session file")?.to_path_buf();
    fs::write(home.join("session"), file.to_string_lossy().as_bytes()).map_err(|error| error.to_string())?;
    // Its notes (a kind with no model served) go to the assistant's log.
    let (model, thinking, note) = host.launch(&args)?;
    let (model, level, note) = match model.or_else(|| host.settings.get("manager.model").and_then(Value::as_str).map(str::to_string)) {
        Some(model) => (Some(model), None, note),
        None => crate::scale::pick(&agent_dir, MANAGER_KIND)?,
    };
    if let Some(note) = note {
        eprintln!("{note}");
    }
    let thinking = thinking.or_else(|| host.settings.get("manager.thinking").and_then(Value::as_str).map(str::to_string)).or(level).unwrap_or_else(|| DEFAULT_THINKING.into());
    let busy = Arc::new(AtomicBool::new(false));
    let marks = busy.clone();
    let emit: crate::agent::EventSink = Box::new(move |event: &crate::agent::Event| match event {
        crate::agent::Event::AgentStart => marks.store(true, Ordering::SeqCst),
        crate::agent::Event::AgentSettled { .. } => marks.store(false, Ordering::SeqCst),
        _ => {}
    });
    if let Some(listing) = host.listing(Some(&session), model.as_deref()) {
        listing.await;
    }
    let (agent, shared) = host.open(session, model.as_deref(), Some(&thinking), emit, false)?;
    let handle = agent.handle();
    let (commands, receiver) = tokio::sync::mpsc::unbounded_channel();
    let task = host.clone().spawn(agent, shared.clone(), true, receiver, |done| {
        if let Err(error) = done.result {
            eprintln!("piccosta manager: {error}");
        }
    });
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut poll = tokio::time::interval(POLL);
    let wait_minutes = host.settings.get("manager.inactiveMinutes").and_then(Value::as_u64).unwrap_or(INACTIVE_MINUTES);
    let mut ticks = 0u64;
    loop {
        tokio::select! {
            _ = term.recv() => break,
            _ = poll.tick() => {
                let mut incoming: Vec<String> = take_inbox(&file).into_iter().map(|text| format!("[{}, from the user]\n{text}", now_text())).collect();
                incoming.extend(due_schedules(&home).into_iter().map(|schedule| format!("[{}, schedule {}]\n{}", now_text(), schedule.id, schedule.text)));
                // The managers' lanes: runs that ended, every second, and quiet lanes, every
                // minute. The assistant's come here, a project manager's go to it.
                ticks += 1;
                let remind = ticks.is_multiple_of(LANE_SCAN_SECONDS);
                incoming.extend(lane_news(&home, wait_minutes, remind));
                // Their schedules that are due go to them too, into the inbox they wait in.
                for project_home in fs::read_dir(agent_dir.join("projects")).into_iter().flatten().flatten().map(|entry| entry.path()) {
                    let mut news = lane_news(&project_home, wait_minutes, remind);
                    let Some(manager) = pointer(&project_home) else { continue };
                    news.extend(due_schedules(&project_home).into_iter().map(|schedule| format!("[{}, schedule {}]\n{}", now_text(), schedule.id, schedule.text)));
                    if !news.is_empty() {
                        let _ = wake(&manager, &news.join("\n\n"));
                    }
                }
                for message in incoming {
                    if busy.load(Ordering::SeqCst) {
                        handle.steer(message);
                    } else {
                        busy.store(true, Ordering::SeqCst);
                        let _ = commands.send(Command::Prompt(message, Vec::new()));
                    }
                }
            }
        }
    }
    handle.abort();
    drop(commands);
    let _ = tokio::time::timeout(Duration::from_secs(5), task).await;
    shared.jobs.shutdown("SIGTERM");
    crate::provider::claude::shutdown().await;
    Ok(std::process::ExitCode::SUCCESS)
}
