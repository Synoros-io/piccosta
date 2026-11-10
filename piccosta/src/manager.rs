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
// Either can have an advisor (/advisor), a standing session on the scale's advice kind that
// advises its manager: its home is <manager's home>/advisor, and its `sessions` reads any session
// and messages managers, its own or another's, each message marked as from it. It hears its
// manager only when messaged; its lanes are the sessions it starts itself, which it may message
// and close. The user talks to both.
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
// Timed messages live with every session's check-ins in the agent dir, not in its ledger.
// - `lanes.json`: the sessions it started or messaged, which it answers for (below), and `runs/`,
//   what each lane's latest run printed;
// - `name`, the assistant's only: what the user calls it, when they've named it.
//
// Its context stays bounded however long it runs. When it compacts, the summary carries the
// newest of the user's words verbatim up to MANAGER_PIN_CHARS (older ones as a one-line index;
// pintask.rs), and the newest LEDGER_VIEW_CHARS of the ledger. Everything else is on disk, word
// for word, and `ledger search` brings it back.
//
// Messages for it, and for the sessions it manages, go through their runtimes (runtime.rs send).

use crate::host::Host;
use crate::message::{AgentMessage, Content, StopReason};
use crate::runtime::{Command, Sent};
use crate::tools::{Tool, ToolContext, ToolFuture, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

// Who a manager or an advisor is: its whole part of the prompt, after the constitution, but for
// the assistant's memory. What it does with its tools is in their descriptions (SessionsTool,
// ManagerTools), so the prompt holds no times, counts or usage and stays the same bytes from turn
// to turn and restart to restart: any change rewrites a Claude prompt cache.
const ASSISTANT_ROLE: &str = "You are the assistant, the user's lead director: you look after this computer and the agents in every project, and hand large pieces of work to other agents. You answer for what they deliver: read and test their work yourself before you report it, and give each new task a fresh session unless it needs what an earlier one holds. Since you live on a computer, you can act on the user's behalf.";
const PROJECT_ROLE: &str = "You are this project's manager: its sessions are yours to direct. Since you live on a computer, you can act on the user's behalf. Read the project and your sessions' work yourself: you answer for what they deliver, so read and test it before you report it, and give each new task a fresh session rather than one that has done another, unless it needs what that one holds. Run the builds yourself, leave long, large arms of work and plans to lanes, and for real work start twins (pair or trio), asking the user first. Answer the assistant, the user's lead director, as you would the user.";
// Who an advisor's message to its own manager says it's from, which its manager's answer is to.
const FROM_ADVISOR: &str = "your advisor";
// {manager} is who it advises.
const ADVISOR_ROLE: &str = "You are the advisor of {manager}: when the user or your manager messages you, you advise it on direction, research design and what it may be missing. Search the web, fetch and read files freely, and take responsibility for what you advise: check it before you say it. Be direct and thorough, and end your turn without a message when there's nothing to say. Do your research and checks yourself, handing only large pieces to sessions you start; what should be done in a project goes to your manager, or to the manager of the project it's in. The user talks to both of you, and the user's word decides. Check any findings, keep the user in the loop of both yourself and the manager to advisor.";
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
// kind for managing a main lane (critical), a new one at its level.
const MANAGER_KIND: &str = "critical";
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
// How often the service looks at check-ins, and a manager at its lanes; and how many looks go
// between looks for quiet ones.
const POLL: Duration = Duration::from_secs(1);
const LANE_SCAN_SECONDS: u64 = 60;
// How long a lane not at work can go without a new entry before its manager hears about it, unless
// `manager.inactiveMinutes` says otherwise: half an hour is long enough that a lane between steps
// isn't nagged, short enough that a forgotten one is noticed within the hour. A lane still quiet
// after a reminder gets the next one after twice the wait, so one left alone costs a few reminders
// a day, not one an hour.
pub const INACTIVE_MINUTES: u64 = 30;
// How often a manager with lanes at work gets a check-in on them, unless `manager.checkMinutes`
// says otherwise (0: never): the same half hour, kept steady however long they run, since a
// manager left alone with a lane that runs all day didn't look at it for hours. Each lane's line carries the end of its last line of text, up
// to CHECK_LINE_CHARS: enough to tell what it's at, the rest a `sessions read` away.
pub const CHECK_MINUTES: u64 = 30;
const CHECK_LINE_CHARS: usize = 200;
// How many sessions `sessions list` shows, how many messages `read` gives by default, and how
// many characters of each it keeps.
const LIST_LIMIT: usize = 40;
const READ_MESSAGES: usize = 6;
const READ_CHARS: usize = 3_000;
// How much of what a run printed the news of its ending carries (a run that ended without a
// turn), and of a finished turn's answer: all of an answer a manager answers for, up to a size that
// can't crowd a smaller model's context out (64,000 characters, about 16k tokens; the review of
// 2026-11-10). Past it, the end goes, with a note that `sessions read` has the whole.
const NEWS_CHARS: usize = 800;
const NEWS_ANSWER_CHARS: usize = 64_000;
// The sessions a manager started, one file path a line in its home: the ones it may delete.
const STARTED_FILE: &str = "started";
// How much `ledger search` returns.
const SEARCH_CHARS: usize = 12_000;
const DAY_MS: i64 = 86_400_000;

// --- The kinds of manager, their homes and their files ---

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Role {
    Assistant,
    Project,
    Advisor,
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
// The session entry that records the subagent jobs a manager granted a lane at `sessions start`
// (its `subagents`: a list, or true for all), so it has those wherever it's opened (host.rs tool_list).
const GRANT_ENTRY: &str = "piccosta-subagents";
// The names managers take, which the sidebar and `sessions list` show.
pub const ASSISTANT_NAME: &str = "assistant";
pub const PROJECT_NAME: &str = "manager";
pub const ADVISOR_NAME: &str = "advisor";
// An advisor's home, in its manager's, and the scale's kind it runs on.
const ADVISOR_DIR: &str = "advisor";
pub const ADVISOR_KIND: &str = "advice";
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
            "advisor" => Some(Role::Advisor),
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
    // A manager (assistant, project manager, advisor) has every tool a project manager has, whatever
    // kind an older session recorded for it.
    if role(session).is_some() {
        return None;
    }
    session.entries().iter().rev().find_map(|entry| match entry {
        crate::session::SessionEntry::Custom { custom_type, data, .. } if custom_type == WORK_ENTRY => Some(data.as_ref()?.get("use")?.as_str()?.to_string()),
        _ => None,
    })
}

// The subagent jobs a session may hand work to: None for all (no grant recorded, as in the user's
// own sessions and managers, or true), else the ones its manager granted, and always search, since
// web search goes to that job.
pub fn granted(session: &crate::session::Session) -> Option<Vec<String>> {
    let jobs = session.entries().iter().rev().find_map(|entry| match entry {
        crate::session::SessionEntry::Custom { custom_type, data, .. } if custom_type == GRANT_ENTRY => data.as_ref()?.get("jobs").cloned(),
        _ => None,
    })?;
    let jobs = jobs.as_array()?.iter().filter_map(Value::as_str).map(str::to_string);
    Some(jobs.chain([crate::tools::web::openai_search::SEARCH_KIND.to_string()]).collect())
}

// The assistant's home, and a project's (a directory's) manager's.
pub fn home(agent_dir: &Path) -> PathBuf {
    agent_dir.join("assistant")
}

pub fn project_home(agent_dir: &Path, cwd: &Path) -> PathBuf {
    let folder = crate::session::default_session_dir(agent_dir, cwd);
    agent_dir.join("projects").join(folder.file_name().unwrap_or_default())
}

// An advisor works where its manager does, so the assistant's is the one in the assistant's home.
pub fn home_for(agent_dir: &Path, role: Role, cwd: &Path) -> PathBuf {
    match role {
        Role::Assistant => home(agent_dir),
        Role::Project => project_home(agent_dir, cwd),
        Role::Advisor => home_for(agent_dir, if cwd == home(agent_dir) { Role::Assistant } else { Role::Project }, cwd).join(ADVISOR_DIR),
    }
}

// All manager homes, advisors included, for migrating timers and finding a lane's run log.
pub fn homes(agent_dir: &Path) -> impl Iterator<Item = PathBuf> {
    std::iter::once(home(agent_dir)).chain(fs::read_dir(agent_dir.join("projects")).into_iter().flatten().flatten().map(|entry| entry.path()))
        .flat_map(|home| [home.clone(), home.join(ADVISOR_DIR)])
}

// Only a manager's lane gets an unattended run; ordinary sessions wait until someone opens them.
pub fn checkin_launch(agent_dir: &Path, file: &Path) -> Option<(PathBuf, PathBuf)> {
    let session = crate::session::Session::read(file).ok()?;
    if role(&session).is_some() { return None; }
    let home = homes(agent_dir).find(|home| load_lanes(home).iter().any(|lane| lane.file == file))?;
    Some((PathBuf::from(&session.header().cwd), run_log(&home, file)))
}

// A manager's session file, read from its home, while the file is there.
pub(crate) fn pointer(home: &Path) -> Option<PathBuf> {
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

// The session file of the advisor of the manager working in `cwd`, and of that manager.
pub fn advisor(agent_dir: &Path, cwd: &Path) -> Option<PathBuf> {
    pointer(&home_for(agent_dir, Role::Advisor, cwd))
}

pub fn advised(agent_dir: &Path, cwd: &Path) -> Option<PathBuf> {
    pointer(home_for(agent_dir, Role::Advisor, cwd).parent()?)
}

// Make a session a manager: record its kind and name in it, and its file in its home. A new
// session's entries reach the file with its first message; until then the home points at a file
// that isn't there, which counts as no manager.
pub fn make(agent_dir: &Path, session: &mut crate::session::Session, role: Role) -> std::io::Result<()> {
    let cwd = PathBuf::from(&session.header().cwd);
    let home = home_for(agent_dir, role, &cwd);
    fs::create_dir_all(&home)?;
    if role == Role::Advisor {
        advised(agent_dir, &cwd).ok_or_else(|| std::io::Error::other("there's no manager here to advise: /manager makes one"))?;
    }
    let (kind, name) = match role {
        Role::Assistant => ("assistant", ASSISTANT_NAME),
        Role::Project => ("project", PROJECT_NAME),
        Role::Advisor => ("advisor", ADVISOR_NAME),
    };
    session.append_custom(ROLE_ENTRY, Some(json!({ "role": kind })))?;
    let base = session.next_base();
    session.append(crate::session::SessionEntry::SessionInfo { base, name: Some(name.to_string()) })?;
    let file = session.file().map(|file| file.to_string_lossy().into_owned()).unwrap_or_default();
    fs::write(home.join("session"), file)
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
    // The run `sessions` started in it, by its identity as it started (runtime.rs Run), until it's
    // heard to end. Not `run`, which held a bare pid before (0.11.4 and older).
    #[serde(default)]
    launched: Option<crate::runtime::Run>,
    // The entry id of the last finished turn persisted in the manager's delivery queue.
    #[serde(default)]
    reported: String,
    // Another manager this one messaged: heard from once, at its next finished turn, then dropped,
    // and never reminded about or told to close, since it isn't this manager's to run. Two managers
    // that message each other would otherwise wake each other at every turn.
    #[serde(default)]
    once: bool,
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
// just started in it, and `heard` the entry id of its last turn before the manager's message,
// which a new lane's news starts after.
fn take_lane(home: &Path, file: &Path, title: &str, run: Option<crate::runtime::Run>, heard: String, once: bool) {
    let mut lanes = load_lanes(home);
    match lanes.iter_mut().find(|lane| lane.file == file) {
        Some(lane) => {
            lane.launched = run.or(lane.launched.take());
            lane.once = once;
        }
        None => lanes.push(Lane { file: file.to_path_buf(), title: title.to_string(), quiet_since: 0, reminders: 0, launched: run, reported: heard, once }),
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

// Finished turns after the durable entry cursor, oldest first. Read the whole branch, not just
// the current context: compaction mustn't hide an answer the manager hasn't heard. Older lane
// files used a timestamp as their cursor; accept that once, then save an entry id.
fn turns_after(file: &Path, reported: &str) -> Vec<(String, String)> {
    let Ok(session) = crate::session::Session::read(file) else { return Vec::new() };
    let path = session.path_to_leaf();
    let after = path.iter().rposition(|entry| entry.id() == reported || entry.base().timestamp == reported).map_or(0, |at| at + 1);
    path[after..].iter().filter_map(|entry| {
        let crate::session::SessionEntry::Message { base, message: AgentMessage::Assistant(reply) } = entry else { return None };
        let omitted = path.iter().rev().find_map(|edit| match edit {
            crate::session::SessionEntry::ContextEdit { target_id, replacement, .. } if target_id == &base.id => Some(replacement.is_none()),
            _ => None,
        }).unwrap_or(false);
        if reply.stop_reason == StopReason::ToolUse || omitted { return None; }
        let text: Vec<&str> = reply.content.iter().filter_map(|block| if let Content::Text { text, .. } = block { Some(text.as_str()) } else { None }).collect();
        let failed = reply.error_message.as_ref().map_or(String::new(), |error| format!("\n(Its turn ended in an error: {error})"));
        Some((base.id.clone(), text.join("\n") + &failed))
    }).collect()
}

// A lane's line in a check-in: how long since its last reply, and the end of the last line of text
// it wrote.
fn check_line(lane: &Lane) -> String {
    let session = crate::session::Session::read(&lane.file).ok();
    let mut replies = session.iter().flat_map(|session| session.entries().iter().rev()).filter_map(|entry| match entry {
        crate::session::SessionEntry::Message { base, message: AgentMessage::Assistant(reply) } => Some((base.timestamp.as_str(), reply)),
        _ => None,
    });
    let since = replies.clone().next().and_then(|(stamp, _)| crate::clock::ms_from_iso(stamp)).map_or("no reply yet".into(), |at| format!("last reply {} minutes ago", (crate::clock::now_ms() - at) / 60_000));
    let line = replies.find_map(|(_, reply)| reply.content.iter().rev().find_map(|block| if let Content::Text { text, .. } = block { text.lines().rev().find(|line| !line.trim().is_empty()) } else { None }));
    let line: Vec<char> = line.unwrap_or_default().trim().chars().collect();
    format!("- Session {} \"{}\": {since}: {}", session_id(&lane.file), lane.title, line[line.len().saturating_sub(CHECK_LINE_CHARS)..].iter().collect::<String>())
}

// The last `limit` characters of a text.
fn ending(text: &str, limit: usize) -> String {
    let characters: Vec<char> = text.trim().chars().collect();
    characters[characters.len().saturating_sub(limit)..].iter().collect()
}

// News of every finished turn the manager hasn't heard yet; `still`, that the lane's run
// goes on (a background job, or a partner's turn). `ended` says its run ended, and whether it said
// so: a run of the manager's that ended without a word or a new turn gets news too, with the end of
// what it printed.
fn catch_up(home: &Path, lane: &mut Lane, still: bool, ended: Option<bool>) -> Vec<(String, String)> {
    let (id, title) = (session_id(&lane.file), lane.title.clone());
    let mut news = Vec::new();
    for (entry, said) in turns_after(&lane.file, &lane.reported) {
        lane.reported = entry.clone();
        let still = if still { " Its run is still going (a background job, or a partner's turn)." } else { "" };
        let next = if lane.once { "Tell the user what came of it." } else { "Its word isn't proof: read what it changed and run it, or start a fresh session to test it, before you tell the user what came of it. Close it (`sessions close`) once its work is done and nothing of it still runs; a new task goes to a fresh session, not this one, unless it needs what this one holds." };
        news.push((format!("lane:{id}:{entry}"), format!("[{}, a lane's turn ended] Session {id} \"{title}\" finished its turn.{still} What it said:\n{}{}\n\n{next}", now_text(), if said.trim().chars().count() > NEWS_ANSWER_CHARS { format!("[Its start is cut; `sessions read {id}` shows it whole.]\n…") } else { String::new() }, ending(&said, NEWS_ANSWER_CHARS))));
    }
    if news.is_empty() && ended == Some(false) && lane.launched.is_some() {
        let printed = fs::read_to_string(run_log(home, &lane.file)).unwrap_or_default();
        let pid = lane.launched.as_ref().map_or(0, |run| run.pid);
        news.push((format!("run:{id}:{}", serde_json::to_string(&lane.launched).unwrap_or_default()), format!("[{}, a lane's run ended] Session {id} \"{title}\": its run (pid {pid}) ended without finishing a turn. The end of what it printed:\n{}", now_text(), ending(&printed, NEWS_CHARS))));
    }
    if ended.is_some() {
        lane.launched = None;
    }
    news
}

// A manager's lanes, watched while its session runs (its runtime starts this): each turn a lane
// finishes, as it ends (a subscription to the lane's runtime, made whenever one runs it); a turn
// that ended while nobody listened, read from its file as the subscription starts, or when it's
// first looked at with nothing running it; a run it started that ended without finishing a turn;
// a lane someone else closed; every LANE_SCAN_SECONDS, each lane not at work that's been quiet for
// its wait (`wait_minutes`, doubled for each reminder since it last changed); and every
// `check_minutes`, once the manager is `idle`, a check-in on the lanes at work then, when there are
// any. A lane whose file is gone, or that was closed, leaves the list. The news goes to the
// manager's runtime, as messages.
// An advisor's lanes are the sessions it started: it hears from the user and its manager only when
// they message it, so a manager an older advisor kept as its lane leaves the list.
pub async fn watch(home: PathBuf, manager: tokio::sync::mpsc::WeakUnboundedSender<Command>, (wait_minutes, check_minutes): (u64, u64), idle: impl Fn() -> bool) {
    // An advisor's lanes are only the sessions it started; another manager a manager kept as a lane
    // before Lane.once is heard from once, as now.
    let mut lanes = load_lanes(&home);
    let advisor = home.ends_with(ADVISOR_DIR);
    lanes.retain_mut(|lane| {
        let manager = crate::session::Session::read(&lane.file).ok().and_then(|session| role(&session)).is_some();
        lane.once |= manager;
        !(advisor && manager)
    });
    save_lanes(&home, &lanes);
    let (heard_sender, mut heard) = tokio::sync::mpsc::unbounded_channel::<(PathBuf, Value)>();
    let mut listening = std::collections::HashSet::new();
    let mut tick = tokio::time::interval(POLL);
    let (mut ticks, mut due) = (0u64, false);
    loop {
        let (news, lanes) = tokio::select! {
            _ = tick.tick() => {
                ticks += 1;
                // A check-in waits for the manager's turn to end, one at most: a message mid-turn
                // steers it and sends its commands to the background.
                due |= ticks.is_multiple_of(check_minutes.saturating_mul(60));
                // Which lanes are at work is asked off this thread, which may be a terminal UI's (a
                // stopped lane holds each ask up to its wait), and only with check-ins on: each
                // minute, so a lane at work gets check-ins rather than reminders, and for a digest,
                // after which the manager is asked again whether it's idle, as the check-in goes.
                let ask = check_minutes > 0 && (ticks.is_multiple_of(LANE_SCAN_SECONDS) || due && idle());
                let files: Vec<PathBuf> = listening.iter().filter(|_| ask).cloned().collect();
                let at_work = if ask { Some(tokio::task::spawn_blocking(move || files.into_iter().filter(|file| crate::runtime::status(file).is_some_and(|status| status["working"] == true)).collect()).await.unwrap_or_default()) } else { None };
                let check = due && at_work.is_some() && idle();
                due &= !check;
                look(&home, &mut listening, &heard_sender, ticks, (wait_minutes, at_work.as_ref(), check))
            }
            Some((file, said)) = heard.recv() => {
                let mut lanes = load_lanes(&home);
                let Some(lane) = lanes.iter_mut().find(|lane| lane.file == file) else { continue };
                // The subscription's first line, then a line a turn, then its end.
                let news = match (said.get("turn_end"), said["ok"] == true) {
                    (Some(turn), _) => catch_up(&home, lane, turn["jobs"].as_u64().unwrap_or(0) > 0, None),
                    (None, true) => catch_up(&home, lane, false, None),
                    (None, false) => {
                        listening.remove(&file);
                        catch_up(&home, lane, false, Some(said["end"] == true))
                    }
                };
                if lane.once && !news.is_empty() {
                    lanes.retain(|lane| lane.file != file);
                    listening.remove(&file);
                }
                (news, lanes)
            }
        };
        let Some(_manager) = manager.upgrade() else { return };
        let Some(file) = pointer(&home) else { return };
        for (id, text) in news {
            if let Err(error) = crate::runtime::deliver(&file, &id, &text) {
                eprintln!("piccosta manager: could not keep lane report: {error}");
                return;
            }
        }
        save_lanes(&home, &lanes);
    }
}

// One look at a manager's lanes (watch), with the lanes at work when they were asked (`at_work`),
// and whether a check-in goes now (`check`). Output: the news and the lanes to save once it's sent.
fn look(home: &Path, listening: &mut std::collections::HashSet<PathBuf>, heard: &tokio::sync::mpsc::UnboundedSender<(PathBuf, Value)>, ticks: u64, (wait_minutes, at_work, check): (u64, Option<&std::collections::HashSet<PathBuf>>, bool)) -> (Vec<(String, String)>, Vec<Lane>) {
    let mut lanes = load_lanes(home);
    let (remind, first) = (ticks.is_multiple_of(LANE_SCAN_SECONDS), ticks == 1);
    let now = (crate::clock::now_ms() / 1000) as u64;
    let agent_dir = crate::config::agent_dir();
    let (mut news, mut working) = (Vec::new(), Vec::new());
    lanes.retain_mut(|lane| {
        let Some(last) = changed(&lane.file) else {
            return false;
        };
        let (id, title) = (session_id(&lane.file), lane.title.clone());
        if crate::session::is_closed(&agent_dir, &lane.file) {
            news.push((uuid::Uuid::new_v4().to_string(), format!("[{}, a lane was closed] Session {id} \"{title}\" was closed by the user or another session; it's no longer one of your lanes.", now_text())));
            return false;
        }
        if !listening.contains(&lane.file) {
            if crate::runtime::running(&lane.file) {
                listening.insert(lane.file.clone());
                let (file, heard) = (lane.file.clone(), heard.clone());
                tokio::task::spawn_local(async move {
                    let said_so = crate::runtime::subscribe(&file, |line| drop(heard.send((file.clone(), line)))).await;
                    let _ = heard.send((file, json!({"end": said_so})));
                });
            } else if first || lane.launched.as_ref().is_some_and(crate::runtime::ended) {
                // What ended while nobody listened: as this manager starts, and a run it started
                // that ended before it could be heard (it never started, say).
                let heard = catch_up(home, lane, false, Some(first));
                if lane.once && !heard.is_empty() {
                    news.extend(heard);
                    return false;
                }
                news.extend(heard);
            }
        }
        // A lane at work (a turn, or a run of its own waiting on its jobs) is in the check-in
        // instead (below); one quiet and not at work gets reminders.
        let busy = at_work.is_some_and(|at_work| at_work.contains(&lane.file));
        working.extend((busy && check && !lane.once).then(|| check_line(lane)));
        if !remind || busy || lane.once {
            return true;
        }
        if last != lane.quiet_since {
            (lane.quiet_since, lane.reminders) = (last, 0);
        }
        let quiet_minutes = now.saturating_sub(last) / 60;
        if quiet_minutes >= wait_minutes << lane.reminders.min(16) {
            lane.reminders += 1;
            let running = if listening.contains(&lane.file) { at_work.map_or("it's open", |_| "it's open, not working") } else { "nothing is running it" };
            news.push((uuid::Uuid::new_v4().to_string(), format!("[{}, a quiet lane] Session {id} \"{title}\" has been inactive for {quiet_minutes} minutes, and {running}. Close it (`sessions close`) if it's done, or update it (`sessions message`), and tell the user if they should know.", now_text())));
        }
        true
    });
    if !working.is_empty() {
        news.push((uuid::Uuid::new_v4().to_string(), format!("[{}, a check-in] Your lanes at work:\n{}\n\nRead any that look stuck, off track or waiting (`sessions read`), act on what you see, and tell the user if they should know.", now_text(), working.join("\n"))));
    }
    (news, lanes)
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

// Stop or close a session's runtime (runtime.rs end), off this thread, since it answers once its
// turn has stopped.
async fn end(home: &Path, file: &Path, what: &'static str) -> Option<Result<Option<String>, String>> {
    let ours = load_lanes(home).into_iter().find(|lane| lane.file == file).and_then(|lane| lane.launched);
    let file = file.to_path_buf();
    tokio::task::spawn_blocking(move || crate::runtime::end(&file, what, ours.as_ref())).await.ok().flatten()
}

fn ago(modified: std::time::SystemTime) -> String {
    let minutes = modified.elapsed().map_or(0, |elapsed| elapsed.as_secs() / 60);
    match minutes {
        0..60 => format!("{minutes}m ago"),
        60..2880 => format!("{}h ago", minutes / 60),
        _ => format!("{}d ago", minutes / 1440),
    }
}

// A message as `read` shows it, each text cut at `limit` characters.
fn message_line(message: &AgentMessage, limit: usize) -> Option<String> {
    let cut = |text: String| if text.chars().count() > limit { format!("{}…", text.chars().take(limit).collect::<String>()) } else { text };
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
    // What the user wants asked before their plans are spent (quota), and whether it's an
    // advisor's, which reads and messages managers and starts nothing.
    quota: String,
    advising: bool,
}

impl SessionsTool {
    fn list(&self) -> String {
        let lines: Vec<String> = crate::host::all_sessions(&self.agent_dir, LIST_LIMIT)
            .iter()
            .map(|summary| {
                let running = if crate::runtime::running(&summary.path) { " · running" } else if summary.closed { " · closed" } else { "" };
                let session = crate::session::Session::open(&summary.path, &summary.cwd).ok();
                let kind = match session.as_ref().and_then(role) {
                    Some(Role::Assistant) => " · the assistant (you)",
                    Some(Role::Project) => " · this project's manager",
                    Some(Role::Advisor) => " · a manager's advisor",
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
        // The last message whole (a lane's report is what the manager answers for), the others cut.
        let all = session.context().messages;
        let last = all.iter().rposition(|message| matches!(message, AgentMessage::Assistant(_)));
        let messages: Vec<String> = all.iter().enumerate().filter_map(|(at, message)| message_line(message, if Some(at) == last { usize::MAX } else { READ_CHARS })).collect();
        let shown = &messages[messages.len().saturating_sub(count)..];
        let running = if crate::runtime::running(&summary.path) { "running now" } else { "not running" };
        Ok(format!("{} · {} · {running} · {} messages\n\n{}", summary.title, summary.cwd.display(), messages.len(), shown.join("\n\n")))
    }

    // Send a message to a session (runtime.rs send): to whatever runs it, which reads it at its next
    // step, else in a run of its own. Either way it's one of this manager's lanes from then on.
    // Sent off this thread, which may be a terminal UI's, since a stopped runtime holds a send up
    // by its wait. Output: what happened, for the reply.
    async fn run(&self, file: &Path, cwd: &Path, message: &str, title: &str) -> Result<String, String> {
        // A message opens a closed session again, from now: the lane's own run did it only as it
        // took the message, and the watch, looking in between, dropped the lane as closed by
        // someone else, never to hear its turns (newC's and Yao's managers, 2026-11-06).
        crate::session::set_closed(&self.agent_dir, file, false);
        let heard = turns_after(file, "").pop().map(|(entry, _)| entry).unwrap_or_default();
        let (to, cwd, message, log) = (file.to_path_buf(), cwd.to_path_buf(), message.to_string(), run_log(&self.home, file));
        let sent = tokio::task::spawn_blocking(move || crate::runtime::send(&to, &message, Some((&cwd, &log)))).await.map_err(|error| error.to_string())?;
        // An advisor answers its manager by message, so it isn't a lane, and an advisor's lanes are
        // only the sessions it started: a message from it to a manager isn't followed.
        // Another manager is heard from once, at its next finished turn (Lane once).
        let (lane, once) = match crate::session::Session::read(file).ok().and_then(|session| role(&session)) {
            Some(Role::Advisor) => (false, false),
            Some(_) => (!self.advising, true),
            None => (true, false),
        };
        let after = if self.advising && !lane { "its answer doesn't reach you: read it later (`sessions read`)" } else { "when its turn ends you get a message with what it said" };
        let (said, run) = match sent.map_err(|error| format!("Couldn't send it: {error}"))? {
            Sent::Started(run) => (format!("Its run started (pid {}) in a process of its own; {after}.", run.pid), Some(run)),
            Sent::Delivered => (format!("It's running, and reads it at its next step; {after}."), None),
            Sent::Unconfirmed => ("It was sent, not confirmed: the piccosta running it didn't answer in time (stopped, or stuck), or its connection ended without an answer. It isn't sent again. Read the session later, or stop it (`sessions stop`).".to_string(), None),
            Sent::Queued => ("It's a manager's session, so the message waits in it for the service or a terminal UI to open it; read the session later to see what came of it.".to_string(), None),
        };
        if lane {
            take_lane(&self.home, file, title, run, heard, once);
        }
        Ok(said)
    }

    // Write a session's model, level, partners and name into its file before anything runs it, so
    // it opens the same in a run, the terminal UI or an editor. `use` picks from the model scale
    // unless a model is named; its level goes in only for a `new` session, and `thinking` for any.
    // Input: a session nobody is running. Output: its model, when set, and a line saying so when
    // `use` found none of the kind's models served (it runs on the default).
    fn configure(&self, session: &mut crate::session::Session, arguments: &Map<String, Value>, new: bool) -> Result<(Option<String>, Option<String>), String> {
        use crate::session::SessionEntry;
        let registry = crate::model::ModelRegistry::load(&self.agent_dir.join("models.json"))?;
        let written = |result: std::io::Result<String>| result.map(drop).map_err(|error| error.to_string());
        let (model, scale_level, note) = match (string(arguments, "model"), string(arguments, "use")) {
            (Some(model), _) => (Some(model.to_string()), None, None),
            (None, Some(kind)) => crate::scale::pick(&self.agent_dir, kind, false)?,
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
        if let Some(level) = string(arguments, "thinking").map(str::to_string).or(scale_level.filter(|_| new)) {
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
        // The day's and the week's, by provider, from everything each session and subagent session
        // changed this week spent: its answers, its compactions and its other usage (a cache
        // warm-up), whether or not they're still in its context. Each session is tallied alone
        // first, so a turn settles only its own session's pending ones.
        let mut totals = Tallies::new();
        let folders = fs::read_dir(self.agent_dir.join("sessions")).into_iter().flatten().flatten().map(|folder| folder.path());
        let files = folders.flat_map(|folder| [folder.join("subagents"), folder]).flat_map(|dir| fs::read_dir(dir).into_iter().flatten().flatten().map(|file| file.path()));
        let this_week = |file: &PathBuf| file.extension().is_some_and(|extension| extension == "jsonl")
            && fs::metadata(file).and_then(|metadata| metadata.modified()).is_ok_and(|modified| modified.elapsed().is_ok_and(|age| age.as_millis() < 7 * DAY_MS as u128));
        // A fork copies its parent's entries with their ids and times, so each is counted once.
        let mut counted = std::collections::HashSet::new();
        for file in files.filter(this_week) {
            let Ok(session) = crate::session::Session::read(&file) else { continue };
            let (mut tallies, mut provider) = (Tallies::new(), String::new());
            for entry in session.entries() {
                if !counted.insert((entry.id().to_string(), entry.base().timestamp.clone())) {
                    continue;
                }
                let at = |base: &crate::session::EntryBase| crate::clock::ms_from_iso(&base.timestamp).unwrap_or(now);
                let spent: Vec<(crate::message::Usage, String, i64)> = match entry {
                    crate::session::SessionEntry::Message { message: AgentMessage::Assistant(answer), .. } => {
                        provider.clone_from(&answer.provider);
                        vec![(answer.usage.clone(), answer.provider.clone(), answer.timestamp)]
                    }
                    crate::session::SessionEntry::Usage { base, provider, usage, .. } => vec![(usage.clone(), provider.clone(), at(base))],
                    // A compaction's summary was asked of the session's model, the last one answering.
                    crate::session::SessionEntry::Compaction { base, usage: Some(usage), .. } => vec![(usage.clone(), provider.clone(), at(base))],
                    // A tool that asked a model itself (web_search) keeps what it spent in its details.
                    crate::session::SessionEntry::Message { message: AgentMessage::ToolResult(result), .. } => result.details.as_ref().and_then(|details| details.get("usage")).and_then(Value::as_array).into_iter().flatten()
                        .filter_map(|spent| Some((serde_json::from_value(spent.get("usage")?.clone()).ok()?, spent.get("provider")?.as_str()?.to_string(), result.timestamp))).collect(),
                    _ => continue,
                };
                for (usage, spent_by, when) in spent {
                    let (day, week) = tallies.entry(spent_by.clone()).or_default();
                    if now - when < 7 * DAY_MS {
                        week.add(&usage, &spent_by);
                    }
                    if now - when < DAY_MS {
                        day.add(&usage, &spent_by);
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

    fn prompt_snippet(&self) -> Option<&str> {
        Some(if self.advising { "read any session; message your manager, or another project's manager; start sessions of your own" } else { "start, message, read, stop and close the sessions on this computer, your lanes; the projects and their managers; what sessions cost, and the model scale" })
    }

    fn definition(&self) -> crate::message::ToolDefinition {
        if self.advising {
            return crate::message::ToolDefinition {
                name: "sessions".into(),
                description: "The sessions on this computer. action list: the latest, with their id, directory and model. projects: the directories that have sessions, with their managers. read: a session's latest messages (session, count; without session, your manager's). message: a message to your manager (no session) or to another manager (session), marked as from you; its answer doesn't reach you: read it later. start: a session of your own in a directory (dir, message; use, model, thinking), which reports back to you when its turn ends; message and close work on the sessions you started.".into(),
                parameters: json!({"type": "object", "required": ["action"], "properties": {"action": {"type": "string", "enum": ["list", "projects", "read", "start", "message", "close"]}, "session": {"type": "string", "description": "A session's id, the start of one, or its name; leave it out for your manager"}, "count": {"type": "number"}, "dir": {"type": "string"}, "message": {"type": "string"}, "use": {"type": "string", "description": "The kind of work, from the model scale"}, "model": {"type": "string", "description": "provider/id; else use"}, "thinking": {"type": "string"}}}),
                constrained_sampling: None,
            };
        }
        crate::message::ToolDefinition {
            name: "sessions".into(),
            description: format!("The sessions on this computer. action list: the latest, with their id, directory, model, whether they're running and what they've cost. projects: the directories that have sessions, with their managers. read: a session's latest messages (session, count). start: a new session in a directory (dir; optional: message, its first message, which it starts on at once in a run of its own, else it waits for the user or you; use: the kind of work, from the model scale, or model; thinking, pair or trio, name; subagents: the subagent jobs it may use, or true for all; discussed: the first words of the user's message where this task began, to copy the discussion word for word before the first message, which should say what to do, not retell it). message: a message for an existing session, to correct or carry on its own task (a new task gets a fresh session unless it needs what this one holds) (session, message; use, model, thinking, pair, trio or name change it first), which it runs the same way. close: the session is done (session): its background run stops, it's marked closed and leaves the list, and a later message opens it again. stop: end a lane's background run now and keep it open (session). delete: move a session you started to the trash, once it isn't running (session). usage: what sessions cost in the last day and week, and the model scale with each plan's usage now.
A session you start or message is one of your lanes: its run carries on in a process of its own, and you get a message with what it said when its turn ends. Messages reach you from the user, your lanes, your schedules, the assistant and your advisor, each saying which, with the time; an advisor's advice is advice, the user's word decides. A turn that ended isn't proof the work was done: read it before you tell the user what happened and which session to open (`ledger notify` when they'd want to know now). Report what was asked; suggest next steps only when asked, or when something needs their decision. Check in on lanes while they work, not only when they end (read them; `checkin` your next look), and act on one stuck, off track, repeating itself or waiting: message or stop it, and tell the user. Close a lane once its work is done: one left quiet keeps reminding you. A lane carries out its task itself: subagents flatten research and lose the structure the user gave it, and spend usage the user didn't start. Grant a lane subagents (which jobs) only when its task truly needs handing parts on; search is always there. When the user asks for a session and not for work in it, start it without a message; don't make up a first task. A project with its own manager (projects shows which) gets its work through that manager, not sessions you start there. Choose models with use, the kind of work from the model scale (usage shows it): piccosta picks the model and moves on when a plan runs low. Say which use and why, and consider a second model for long or important lanes (trio: it checks each answer once; pair: the two work it through together). When something you read disagrees with the scale, ask the user. {}", self.quota),
            parameters: json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": ["list", "projects", "read", "start", "message", "stop", "close", "delete", "usage"]},
                    "session": {"type": "string", "description": "A session's id, the start of one, or its name"},
                    "count": {"type": "number", "description": "How many messages read shows"},
                    "dir": {"type": "string", "description": "The directory a new session works in"},
                    "message": {"type": "string", "description": "What to send the session; start can leave it out"},
                    "discussed": {"type": "string", "description": "For start: the first words of the user's message where this task was first discussed; piccosta copies the discussion from there, word for word, to the session"},
                    "use": {"type": "string", "description": "The kind of work, from the model scale: piccosta picks its model now"},
                    "model": {"type": "string", "description": "provider/id; else use"},
                    "thinking": {"type": "string"},
                    "pair": {"type": "string", "description": "A model to work with as a pair: the two answer each other in turn until one says it's done, then the user (or you) speaks"},
                    "trio": {"type": "string", "description": "A model to work with as a trio: it checks each answer once, then the user (or you) speaks"},
                    "name": {"type": "string"},
                    "subagents": {"type": ["array", "boolean"], "items": {"type": "string"}, "description": "For start: the subagent jobs the session may use (search it always has), or true for all; left out, none"}
                }
            }),
            constrained_sampling: None,
        }
    }

    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let need = |key: &str| string(&arguments, key).ok_or_else(|| format!("{key} is needed"));
            let count = arguments.get("count").and_then(Value::as_u64).map_or(READ_MESSAGES, |count| count as usize);
            // An advisor's session, left out, is its manager. It messages managers, its own or
            // another's, and the sessions it started, which it may close; the rest goes on below.
            if self.advising {
                let manager = find(&self.agent_dir, &session_id(&self.home.parent().and_then(pointer).ok_or("Your manager's session isn't there.")?))?;
                let summary = string(&arguments, "session").map_or(Ok(manager.clone()), |wanted| find(&self.agent_dir, wanted))?;
                let from = if summary.path == manager.path { FROM_ADVISOR.to_string() } else if manager.cwd == home(&self.agent_dir) { "the assistant's advisor".to_string() } else { format!("the advisor of the manager in {}", manager.cwd.display()) };
                let started = fs::read_to_string(self.home.join(STARTED_FILE)).unwrap_or_default().lines().any(|line| Path::new(line) == summary.path);
                match string(&arguments, "action").unwrap_or_default() {
                    "read" => return text(self.read(&summary.id, count)?),
                    "message" if crate::session::Session::read(&summary.path).ok().and_then(|session| role(&session)).is_some_and(|role| role != Role::Advisor) => {
                        return text(self.run(&summary.path, &summary.cwd, &format!("[{}, from {from}]\n{}", now_text(), need("message")?), &summary.title).await?);
                    }
                    "list" | "projects" | "start" => {}
                    "message" | "close" if started => {}
                    "message" | "close" => return Err(format!("\"{}\" isn't a manager or a session you started.", summary.title)),
                    action => return Err(format!("No action \"{action}\" for an advisor: list, projects, read, start, message or close.")),
                }
            }
            match string(&arguments, "action").unwrap_or_default() {
                "list" => text(self.list()),
                "projects" => text(self.projects()),
                "read" => text(self.read(need("session")?, count)?),
                "usage" => text(self.usage()),
                "start" => {
                    let dir = PathBuf::from(need("dir")?);
                    let dir = dir.canonicalize().map_err(|error| format!("{}: {error}", dir.display()))?;
                    // The lane's subagents: none but search unless granted, and only jobs the scale has. A
                    // session started without a task is the user's to work in, so it keeps them all.
                    let grant = match arguments.get("subagents") {
                        Some(Value::Bool(all)) => if *all { json!(true) } else { json!([]) },
                        Some(Value::Array(jobs)) => json!(jobs),
                        None if string(&arguments, "message").is_none() => json!(true),
                        None => json!([]),
                        Some(other) => return Err(format!("subagents is a list of jobs or true, not {other}")),
                    };
                    let jobs: Vec<String> = crate::scale::jobs(&self.agent_dir).into_iter().map(|(name, _)| name).collect();
                    if let Some(unknown) = grant.as_array().into_iter().flatten().find(|job| !job.as_str().is_some_and(|job| jobs.iter().any(|known| known == job))) {
                        return Err(format!("No subagent job {unknown}. Jobs: {}", jobs.join(", ")));
                    }
                    // Written whole now, so it's listed at once and a run opens this very session.
                    let folder = crate::session::default_session_dir(&self.agent_dir, &dir);
                    let mut session = crate::session::Session::create(&dir, Some(&folder));
                    session.persist_now().map_err(|error| error.to_string())?;
                    session.append_custom(GRANT_ENTRY, Some(json!({ "jobs": grant }))).map_err(|error| error.to_string())?;
                    let (model, note) = self.configure(&mut session, &arguments, true)?;
                    let file = session.file().ok_or("could not name the session's file")?.to_path_buf();
                    // Kept, so this manager may delete it later: a manager deletes only what it made.
                    let _ = fs::OpenOptions::new().create(true).append(true).open(self.home.join(STARTED_FILE)).and_then(|mut list| writeln!(list, "{}", file.display()));
                    let (id, name) = (session_id(&file), crate::host::title(&session));
                    drop(session);
                    let started = format!("Made session {id}, a session in {}{}.{}", dir.display(), model.map_or(String::new(), |model| format!(", on {model}")), note.map_or(String::new(), |note| format!(" {note}.")));
                    match string(&arguments, "message") {
                        Some(message) => {
                            let title = string(&arguments, "name").map_or_else(|| message.lines().next().unwrap_or_default().chars().take(80).collect(), str::to_string);
                            let message = crate::session::delegation_message(context, string(&arguments, "discussed"), message)?;
                            text(format!("{started} {}", self.run(&file, &dir, &message, &title).await?))
                        }
                        None => {
                            take_lane(&self.home, &file, &name, None, String::new(), false);
                            text(format!("{started} It has no message, so it waits: the user can open it in piccosta, where it's listed under its project, or you can message it."))
                        }
                    }
                }
                "message" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let changes = ["use", "model", "thinking", "pair", "trio", "name"].iter().any(|key| arguments.contains_key(*key));
                    // Changes go in only while nothing runs it, before the message starts a run.
                    let configured = if changes { crate::runtime::when_idle(&summary.path, |session| self.configure(session, &arguments, false))? } else { None };
                    let note = configured.transpose()?.and_then(|(_, note)| note).map(|note| format!("{note}. "));
                    text(format!("{}{}", note.unwrap_or_default(), self.run(&summary.path, &summary.cwd, need("message")?, &summary.title).await?))
                }
                // Stop a lane's run, as Esc does: what it's running moves to the background and its
                // turn stops; a run of its own then ends, and everything it started with it. One
                // open in a terminal or editor stays open there.
                "stop" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let stopped = match end(&self.home, &summary.path, "stop").await {
                        None => "isn't running".to_string(),
                        Some(Err(why) | Ok(Some(why))) => why,
                        Some(Ok(None)) if crate::runtime::running(&summary.path) => "its turn stopped; it's open in a terminal or editor, which keeps it".to_string(),
                        Some(Ok(None)) => "its run stopped, with everything it was running".to_string(),
                    };
                    text(format!("\"{}\": {stopped}.", summary.title))
                }
                // Close a session: it's done. Whatever runs it stops and ends, with its jobs; it's
                // marked closed for everyone (it leaves the sidebar), and it's no longer a lane.
                "close" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    // A run that couldn't be stopped stays open and a lane, so it isn't left running unseen.
                    let stopped = match end(&self.home, &summary.path, "close").await {
                        None => String::new(),
                        Some(Err(why)) => return Err(format!("\"{}\" is still running and stays open: {why}", summary.title)),
                        Some(Ok(None)) => ", and its run ended, with everything it was running".to_string(),
                        Some(Ok(Some(killed))) => format!("; {killed}"),
                    };
                    crate::session::set_closed(&self.agent_dir, &summary.path, true);
                    let mut lanes = load_lanes(&self.home);
                    lanes.retain(|lane| lane.file != summary.path);
                    save_lanes(&self.home, &lanes);
                    text(format!("Closed \"{}\"{stopped}. A message to it opens it again.", summary.title))
                }
                // Delete a session this manager started: to the trash, with the Claude bridge's
                // record. A session still running is left alone.
                "delete" => {
                    let summary = find(&self.agent_dir, need("session")?)?;
                    let started = fs::read_to_string(self.home.join(STARTED_FILE)).unwrap_or_default();
                    if !started.lines().any(|line| Path::new(line) == summary.path) {
                        return Err(format!("\"{}\" wasn't started by you, and you delete only sessions you started; close it instead (`sessions close`).", summary.title));
                    }
                    if crate::runtime::running(&summary.path) {
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

    fn prompt_snippet(&self) -> Option<&str> {
        Some(if self.memory { "your ledger, memory and desktop notifications" } else { "your ledger and desktop notifications" })
    }

    fn definition(&self) -> crate::message::ToolDefinition {
        let mut actions = vec!["append", "search", "notify"];
        let memory = if self.memory { " remember: add a line to your memory (text): what should hold from one day to the next, what the user wants done and how, the projects and machines you look after, who to hand what; all of it is in your prompt, through every compaction. forget: take out the lines of your memory that hold a phrase (text). name: take the name the user gives you (text). Never a credential, key or password in your memory or your ledger." } else { " Never a credential, key or password in your ledger." };
        let scope = if self.memory { "across the projects" } else { "in the project" };
        if self.memory {
            actions.extend(["remember", "forget", "name"]);
        }
        crate::message::ToolDefinition {
            name: "ledger".into(),
            description: format!("Your ledger and your notes to the user. Your ledger is your running record of what's going on {scope}: its newest entries ride through your compactions, and search finds the rest. A project's own files (AGENTS.md, direction.md, ledger.md, its feature list) are kept as its AGENTS.md says: write the ledger entries and record the user's requests it asks for, and change its AGENTS.md only when the user says so. action append: add an entry to your ledger, a line or two: the lanes you started and why (session ids, models), what came of each, what the user decided and what's next (text). search: ledger entries and things the user said to you that contain every word (text). notify: show the user a desktop notification (text).{memory}"),
            parameters: json!({
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "enum": actions},
                    "text": {"type": "string"}
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
    let quota = quota(settings, settings.get("manager.askBeforePlanQuota").and_then(Value::as_bool).unwrap_or(true));
    let sessions: Arc<dyn Tool> = Arc::new(SessionsTool { agent_dir: agent_dir.to_path_buf(), home: home.clone(), quota, advising: role == Role::Advisor });
    if role == Role::Advisor { vec![sessions] } else { vec![sessions, Arc::new(ManagerTools { home, notify, memory: role == Role::Assistant })] }
}

// A manager's or an advisor's part of the system prompt: who it is, and the assistant's memory,
// which changes only by its own `ledger remember` and `forget`, as does the name the user gave it.
pub fn prompt(role: Role, cwd: &Path) -> String {
    let agent_dir = crate::config::agent_dir();
    match role {
        Role::Advisor => ADVISOR_ROLE.replace("{manager}", if cwd == home(&agent_dir) { "the assistant" } else { "this project's manager" }),
        Role::Project => PROJECT_ROLE.to_string(),
        Role::Assistant => {
            let name = assistant_name(&agent_dir);
            let named = if name == ASSISTANT_NAME { String::new() } else { format!(" The user calls you {name}.") };
            let memory = fs::read_to_string(home(&agent_dir).join("memory.md")).unwrap_or_default();
            let memory = if memory.trim().is_empty() { "nothing yet.".to_string() } else { format!("\n{}", memory.trim_end()) };
            format!("{ASSISTANT_ROLE}{named}\n\nYour memory: {memory}")
        }
    }
}

// A message from the user to a manager, saying so with the time, as its sessions tool says its
// messages do.
pub fn from_user(text: &str) -> String {
    format!("[{}, from the user]\n{text}", now_text())
}

// --- The service ---

// The systemd user unit the assistant runs under, by name, and how long systemd waits before
// starting it again after it fails (a missing login would otherwise restart it in a tight loop).
const UNIT: &str = "piccosta-assistant.service";
pub(crate) const RESTART_SECONDS: u32 = 30;
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
// `asked`: the user asked for it (/assistant), so it starts whatever the setting says, and a unit
// already there is started as it is (it may run a build the user chose), not rewritten.
// Output: what it did, for a notice, or None when it's off or already running.
pub fn ensure_running(settings: &crate::config::Settings, asked: bool) -> Option<String> {
    let agent_dir = crate::config::agent_dir();
    if session_file(&agent_dir).is_some_and(|file| crate::runtime::running(&file)) {
        return None;
    }
    if (!asked && settings.get("manager.autostart").and_then(Value::as_bool) != Some(true)) || own_agent_dir() || started_by_agent() {
        return None;
    }
    let exe = crate::proc::exe().ok()?.to_string_lossy().into_owned();
    let systemctl = |words: &[&str]| std::process::Command::new("systemctl").arg("--user").args(words).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status().is_ok_and(|status| status.success());
    let config = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| Some(crate::config::home().join(".config")))?;
    if systemctl(&["show-environment"]) {
        let unit = unit_text(&exe, &agent_dir);
        let path = config.join("systemd/user").join(UNIT);
        if !(asked && path.exists()) && fs::read_to_string(&path).ok().as_deref() != Some(unit.as_str()) {
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

// `piccosta assistant`: run the assistant until SIGTERM. Messages come through its runtime (the
// TUI, `piccosta tell`), from its lanes as their turns end, and from its schedules; each goes in
// with the time and where it came from. It also fires unattended sessions' check-ins.
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
    let (mut session, new) = match session_file(&agent_dir) {
        Some(path) => (crate::session::Session::open(&path, &home).map_err(|error| error.to_string())?, false),
        None => {
            let mut session = crate::session::Session::create(&home, Some(&host.session_dir));
            make(&agent_dir, &mut session, Role::Assistant).map_err(|error| error.to_string())?;
            (session, true)
        }
    };
    session.persist_now().map_err(|error| error.to_string())?;
    let file = session.file().ok_or("the manager needs a session file")?.to_path_buf();
    fs::write(home.join("session"), file.to_string_lossy().as_bytes()).map_err(|error| error.to_string())?;
    // Its notes (a kind with no model served) go to the assistant's log.
    let (model, thinking, note) = host.launch(&args)?;
    let (model, level, note) = match model.or_else(|| host.settings.get("manager.model").and_then(Value::as_str).map(str::to_string)) {
        Some(model) => (Some(model), None, note),
        None => crate::scale::pick(&agent_dir, MANAGER_KIND, false)?,
    };
    if let Some(note) = note {
        eprintln!("{note}");
    }
    // A level asked for sets it; the scale's sets only a new one's. A restart carries on at the
    // level the session had, or the one its model was last set to by hand (host.rs open).
    let thinking = thinking.or_else(|| host.settings.get("manager.thinking").and_then(Value::as_str).map(str::to_string)).or(level.filter(|_| new));
    if let Some(listing) = host.listing(Some(&session), model.as_deref()) {
        listing.await;
    }
    let (agent, shared) = host.open(session, model.as_deref(), thinking.as_deref(), Box::new(|_| {}), false)?;
    let (commands, task) = crate::runtime::spawn(host.clone(), agent, shared, true, false, |done| {
        if let Err(error) = done.result {
            eprintln!("piccosta manager: {error}");
        }
    });
    // Cover unattended sessions too, including advisors and lanes whose print run ended.
    let timers = tokio::task::spawn_local(crate::tools::checkin::watch(agent_dir, None));
    crate::proc::termination(true).await;
    timers.abort();
    drop(commands);
    let _ = tokio::time::timeout(Duration::from_secs(5), task).await;
    crate::provider::claude::shutdown().await;
    Ok(std::process::ExitCode::SUCCESS)
}
