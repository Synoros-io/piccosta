// Check-ins are messages to this session, not prompt state. One dispatcher covers the process's
// runtimes; the assistant service also covers sessions nobody runs. The locked store joins them.
use super::{Tool, ToolContext, ToolFuture, ToolOutput};
use crate::message::{Content, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{collections::HashMap, fs, path::{Path, PathBuf}, sync::{Arc, Mutex, Weak}, time::Duration};

// The service and runtimes look once a second, like the manager's lane watcher. Failed deliveries
// start with the service's restart interval and back off to five minutes.
pub const POLL: Duration = Duration::from_secs(1);
pub(crate) const RETRY_MS: i64 = crate::manager::RESTART_SECONDS as i64 * 1_000;
pub(crate) const RETRY_MAX_MS: i64 = 5 * 60 * 1_000;
// Give delivery the same grace as stopping a runtime. Receipts make a late worker safe to retry;
// expiry also releases a claim whose final store write failed while its dispatcher stayed alive.
const CLAIM_MS: i64 = crate::runtime::STOP_WAIT.as_millis() as i64;
const DAY_MS: i64 = 86_400_000;
const STORE: &str = "checkins.json";

#[derive(Clone, PartialEq, Serialize, Deserialize)]
struct Checkin {
    id: String,
    text: String,
    next: i64,
    every_ms: Option<i64>,
    // Missing ownership is an old agent check-in, never one of the user's replay prompts.
    #[serde(default)]
    user: bool,
    #[serde(default)]
    created: i64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    review: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    remaining: Option<u64>,
    #[serde(default)]
    outcome: Option<(i64, crate::runtime::TimerOutcome)>,
    #[serde(default)]
    retry_at: i64,
    #[serde(default)]
    retry_ms: i64,
    #[serde(default)]
    file: PathBuf,
    // A dispatcher owns this occurrence until it finishes, dies or its lease expires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    claimed: Option<(crate::runtime::Run, i64)>,
    // Old schedule delivery ids include the manager's home. Keep that receipt across migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<PathBuf>,
}

impl Checkin {
    fn occurrence(&self) -> String {
        match &self.source {
            Some(home) => format!("schedule:{}:{}:{}", home.display(), self.id, self.next),
            None => format!("checkin:{}:{}", self.id, self.next),
        }
    }

    fn user_input(&self, agent_dir: &Path) -> crate::message::UserMessage {
        let text = format!("[user timer, set {}, fired {}]\n{}\n\n[Latest known usage — cached readings, not current]\n{}",
            crate::clock::iso_from_ms(self.created), crate::clock::iso_from_ms(crate::clock::now_ms()), self.text, crate::scale::latest_usage(agent_dir));
        let mut message = crate::message::UserMessage::new(&text, Vec::new(), crate::message::InputOrigin::UserTimer { set_at: self.created });
        message.timer = Some(crate::message::UserTimer { id: self.occurrence(), title: crate::host::title_text(&self.text), prompt: Some(self.text.clone()) });
        message
    }
}

// The dispatcher holds the store lock through admission. Read its atomically published snapshot
// without taking that lock, and verify the socket peer owns the claim. Timer words and dates
// come from the user's stored timer, never from a caller's payload or claimed origin.
pub(crate) fn timer_input(agent_dir: &Path, file: &Path, id: &str, peer: Option<i32>) -> Result<(crate::message::UserMessage, bool), String> {
    let bytes = fs::read(agent_dir.join(STORE)).map_err(|error| error.to_string())?;
    let timers: Vec<Checkin> = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    timers.iter().find(|timer| timer.user && timer.occurrence() == id && crate::tools::paths::canonical(&timer.file) == crate::tools::paths::canonical(file)
        && timer.claimed.as_ref().is_some_and(|(owner, _)| Some(owner.pid) == peer && owner.alive()))
        .map(|timer| (timer.user_input(agent_dir), timer.review)).ok_or_else(|| "timer dispatcher or user-owned occurrence could not be verified".into())
}

fn save(path: &Path, timers: &[Checkin]) -> Result<(), String> {
    let partial = path.with_extension("json.partial");
    let mut output = fs::File::create(&partial).map_err(|error| error.to_string())?;
    use std::io::Write;
    output.write_all(serde_json::to_string_pretty(timers).map_err(|error| error.to_string())?.as_bytes()).and_then(|()| output.sync_all()).map_err(|error| error.to_string())?;
    fs::rename(partial, path).map_err(|error| error.to_string())
}

// Read, change and replace the store under a separate lock: replacing the data mustn't replace
// the lock another process is waiting on. Never treat a broken store as an empty one.
fn store<T>(agent_dir: &Path, change: impl FnOnce(&mut Vec<Checkin>) -> Result<T, String>) -> Result<T, String> {
    fs::create_dir_all(agent_dir).map_err(|error| error.to_string())?;
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(agent_dir.join("checkins.lock")).map_err(|error| error.to_string())?;
    lock.lock().map_err(|error| error.to_string())?;
    let path = agent_dir.join(STORE);
    let mut timers: Vec<Checkin> = match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.to_string()),
    };
    for home in crate::manager::homes(agent_dir) {
        let legacy = home.join("schedules.json");
        if !legacy.exists() { continue; }
        let Some(file) = crate::manager::pointer(&home) else { continue };
        let old: Vec<Checkin> = serde_json::from_str(&fs::read_to_string(&legacy).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
        for mut timer in old {
            if timers.iter().any(|current| current.source.as_ref() == Some(&home) && current.id == timer.id) { continue; }
            timer.file = file.clone();
            timer.source = Some(home.clone());
            timers.push(timer);
        }
        // Save before removing the old file; deduplication above makes an interrupted move safe.
        // Don't fire until removal succeeds, or a consumed occurrence could be imported again.
        save(&path, &timers)?;
        fs::remove_file(legacy).map_err(|error| error.to_string())?;
    }
    let before = timers.clone();
    let result = change(&mut timers)?;
    if timers != before { save(&path, &timers)?; }
    Ok(result)
}

// "30m", "2h", "1d", "45s" as milliseconds. Overflow is an invalid duration, not a panic.
fn duration_ms(text: &str) -> Option<i64> {
    let text = text.trim();
    let (number, unit) = text.split_at(text.find(|character: char| !character.is_ascii_digit())?);
    let unit_ms = match unit.trim() { "s" => 1_000, "m" => 60_000, "h" => 3_600_000, "d" => DAY_MS, _ => return None };
    number.parse::<i64>().ok()?.checked_mul(unit_ms).filter(|ms| *ms > 0)
}

// An ISO time needs its offset. Work in ASCII before slicing; malformed model input is an error.
fn time_ms(text: &str) -> Option<i64> {
    let text = text.trim();
    if !text.is_ascii() { return None; }
    let (clock, offset) = if let Some(utc) = text.strip_suffix('Z') { (utc, 0) } else {
        let (clock, offset) = text.split_at(text.len().checked_sub(6)?);
        let sign = match offset.as_bytes()[0] { b'+' => 1, b'-' => -1, _ => return None };
        if &offset[3..4] != ":" { return None; }
        let (hours, minutes) = (offset[1..3].parse::<i64>().ok()?, offset[4..6].parse::<i64>().ok()?);
        if hours > 23 || minutes > 59 { return None; }
        (clock, sign * (hours * 3_600_000 + minutes * 60_000))
    };
    let clock = if clock.len() == 16 { format!("{clock}:00") } else { clock.to_string() };
    crate::clock::ms_from_iso(&format!("{clock}Z"))?.checked_sub(offset)
}

type Receivers = Mutex<HashMap<Option<PathBuf>, usize>>;
static DISPATCHERS: Mutex<Option<HashMap<PathBuf, Weak<Receivers>>>> = Mutex::new(None);

// Each watcher keeps its receiver registered until aborted. The service's None covers all files;
// ordinary runtimes mustn't launch unattended sessions just because they share the store.
struct Receiver(Arc<Receivers>, Option<PathBuf>);
impl Drop for Receiver {
    fn drop(&mut self) {
        let mut receivers = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(count) = receivers.get_mut(&self.1) {
            *count -= 1;
            if *count == 0 { receivers.remove(&self.1); }
        }
    }
}

// One awaited poll per process and agent dir, however many runtimes are open. A weak reference
// lets the dispatcher end when its last watcher goes away; missed ticks don't queue disk work.
pub async fn watch(agent_dir: PathBuf, only: Option<PathBuf>) {
    let receivers = {
        let mut dispatchers = DISPATCHERS.lock().unwrap_or_else(|error| error.into_inner());
        let dispatchers = dispatchers.get_or_insert_with(HashMap::new);
        dispatchers.retain(|_, dispatcher| dispatcher.strong_count() > 0);
        match dispatchers.get(&agent_dir).and_then(Weak::upgrade) {
            Some(receivers) => receivers,
            None => {
                let receivers = Arc::new(Receivers::default());
                let weak = Arc::downgrade(&receivers);
                dispatchers.insert(agent_dir.clone(), weak.clone());
                tokio::spawn(async move {
                    let mut poll = tokio::time::interval(POLL);
                    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                    loop {
                        poll.tick().await;
                        let Some(receivers) = weak.upgrade() else { break };
                        let only = receivers.lock().unwrap_or_else(|error| error.into_inner()).keys().cloned().collect::<Vec<_>>();
                        let agent_dir = agent_dir.clone();
                        match tokio::task::spawn_blocking(move || fire(&agent_dir, &only)).await {
                            Ok(Ok(())) => {},
                            Ok(Err(error)) => eprintln!("piccosta checkin: {error}"),
                            Err(error) => eprintln!("piccosta checkin: {error}"),
                        }
                        // A replacement watcher must join this dispatcher while its poll is
                        // blocked, even if the last old watcher was aborted in the meantime.
                        drop(receivers);
                    }
                });
                receivers
            }
        }
    };
    *receivers.lock().unwrap_or_else(|error| error.into_inner()).entry(only.clone()).or_default() += 1;
    let _receiver = Receiver(receivers, only);
    std::future::pending::<()>().await;
}

// Claim under the store lock, then give each delivery its own blocking worker. A stopped
// receiver holds neither the store nor another receiver; a dead dispatcher leaves a claim we
// can take over. The occurrence's delivery id makes that takeover safe even after a lost answer.
fn fire(agent_dir: &Path, only: &[Option<PathBuf>]) -> Result<(), String> {
    let owner = crate::runtime::Run::of(std::process::id() as i32).ok_or("could not identify check-in dispatcher")?;
    let due = store(agent_dir, |timers| {
        if timers.is_empty() { return Ok(Vec::new()); }
        let now = crate::clock::now_ms();
        let closed = crate::session::closed_sessions(agent_dir);
        timers.retain(|timer| timer.user || !closed.contains_key(&crate::tools::paths::canonical(&timer.file)) && timer.file.exists());
        let mut due = Vec::new();
        for timer in timers {
            // User timers never launch sessions. Any dispatcher can record an offline skip;
            // an agent-closed session keeps its user's timers paused until it opens again.
            if timer.user && closed.contains_key(&crate::tools::paths::canonical(&timer.file)) { continue; }
            if !(timer.user || only.contains(&None) || only.iter().any(|file| file.as_ref() == Some(&timer.file))) || timer.next > now || timer.retry_at > now || timer.claimed.as_ref().is_some_and(|(owner, until)| *until > now && owner.alive()) { continue; }
            timer.claimed = Some((owner.clone(), now.saturating_add(CLAIM_MS)));
            due.push(timer.clone());
        }
        Ok(due)
    })?;
    for timer in due {
        let agent_dir = agent_dir.to_path_buf();
        tokio::task::spawn_blocking(move || {
            if let Err(error) = deliver(&agent_dir, timer) { eprintln!("piccosta checkin: {error}"); }
        });
    }
    Ok(())
}

// Delivery is outside the store lock. Finalize only the occurrence we claimed: cancellation or
// a later occurrence mustn't be undone by a slow receiver's answer.
fn deliver(agent_dir: &Path, claimed: Checkin) -> Result<(), String> {
    let id = claimed.occurrence();
    let sent = if claimed.user {
        // Cancellation and admission have one order: off either removes this occurrence before
        // we ask, or waits for its admission answer. The owner never needs this store lock.
        store(agent_dir, |timers| {
            // A close after claiming mustn't advance past an admitted review's receipt. Keep
            // that occurrence paused; reopening reconciles it before any offline skip.
            if !timers.iter().any(|timer| timer == &claimed) || crate::session::is_closed(agent_dir, &claimed.file) { return Ok(if claimed.review { crate::runtime::TimerOutcome::Deferred } else { crate::runtime::TimerOutcome::SkippedBusy }); }
            crate::runtime::deliver_timer(&claimed.file, &claimed.user_input(agent_dir), claimed.review)
        })
    } else {
        let text = format!("[{}, checkin {}]\n{}", crate::clock::iso_from_ms(crate::clock::now_ms()), claimed.id, claimed.text);
        let launch = crate::manager::checkin_launch(agent_dir, &claimed.file);
        crate::runtime::deliver_checkin(&claimed.file, &id, &crate::message::UserMessage::new(&text, Vec::new(), crate::message::InputOrigin::Checkin), launch.as_ref().map(|(cwd, log)| (cwd.as_path(), log.as_path()))).map(|()| crate::runtime::TimerOutcome::Delivered)
    };
    store(agent_dir, |timers| {
        let now = crate::clock::now_ms();
        timers.retain_mut(|timer| {
            if timer.id != claimed.id || timer.file != claimed.file || timer.next != claimed.next || timer.claimed != claimed.claimed { return true; }
            timer.claimed = None;
            if let Err(error) = &sent {
                timer.retry_ms = timer.retry_ms.saturating_mul(2).clamp(RETRY_MS, RETRY_MAX_MS);
                timer.retry_at = now.saturating_add(timer.retry_ms);
                eprintln!("piccosta checkin: {error}; retry after {}", crate::clock::iso_from_ms(timer.retry_at));
                return true;
            }
            if timer.user { timer.outcome = sent.as_ref().ok().map(|outcome| (now, *outcome)); }
            (timer.retry_at, timer.retry_ms) = (0, 0);
            if sent == Ok(crate::runtime::TimerOutcome::Deferred) { return true; }
            if timer.review && sent == Ok(crate::runtime::TimerOutcome::Delivered) && let Some(remaining) = &mut timer.remaining {
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 { return false; }
            }
            if let Some(every) = timer.every_ms.filter(|every| *every > 0) {
                // A repeat missed while nothing ran fires once, then keeps its rhythm.
                timer.next = now.saturating_add(every - (now - timer.next) % every);
                true
            } else { false }
        });
        Ok(())
    })
}

pub fn cancel_all(agent_dir: &Path, file: &Path, by_user: bool) -> Result<(), String> {
    store(agent_dir, |timers| { timers.retain(|timer| timer.user && !by_user || crate::tools::paths::canonical(&timer.file) != file); Ok(()) })
}

// /timer and /align work on the shown session, even when another process owns it. The TUI's
// disk worker calls this; neither a store lock nor paused-session reads belong on input.
pub fn user_timer(agent_dir: &Path, file: &Path, argument: &str, review: bool) -> Result<String, String> {
    let file = crate::tools::paths::canonical(file);
    let kind = if review { "alignment" } else { "user" };
    store(agent_dir, |timers| {
        let ours = |timer: &Checkin| timer.user && timer.review == review && crate::tools::paths::canonical(&timer.file) == file;
        if argument.is_empty() {
            let paused = crate::session::is_closed(agent_dir, &file);
            let lines: Vec<_> = timers.iter().filter(|timer| ours(timer)).map(|timer| {
                let last = timer.outcome.map_or("not fired".into(), |(when, outcome)| format!("{} at {}", outcome.label(), crate::clock::iso_from_ms(when)));
                format!("{} · next {}{}{} · last {last}\n{}", timer.id, crate::clock::iso_from_ms(timer.next), if paused { " · paused while closed" } else { "" }, timer.remaining.map_or(String::new(), |count| format!(" · {count} remaining")), timer.text)
            }).collect();
            return Ok(if lines.is_empty() { format!("No {kind} timers.") } else { lines.join("\n") });
        }
        let (interval, prompt) = argument.split_once(char::is_whitespace).map_or((argument, ""), |(word, rest)| (word, rest));
        if interval == "off" {
            let prompt = prompt.trim();
            let before = timers.len();
            timers.retain(|timer| !ours(timer) || !prompt.is_empty() && timer.id != prompt);
            return Ok(format!("Cancelled {} {kind} timer(s). Admitted work is not revoked; Esc stops it.", before - timers.len()));
        }
        let every = duration_ms(interval).ok_or(if review { "/align <interval> [count], /align, or /align off [id]" } else { "/timer <interval> <prompt> (like 30m, 2h or 5s), /timer, or /timer off [id]" })?;
        let remaining = if review && !prompt.trim().is_empty() { Some(prompt.trim().parse::<u64>().ok().filter(|count| *count > 0).ok_or("/align count must be a positive integer")?) } else { None };
        if !review && prompt.trim().is_empty() { return Err("/timer needs the prompt to replay.".into()); }
        let created = crate::clock::now_ms();
        let next = created.checked_add(every).ok_or("duration is too large")?;
        let id = uuid::Uuid::new_v4().to_string();
        timers.push(Checkin { id: id.clone(), text: if review { "Intent review".into() } else { prompt.to_string() }, next, every_ms: Some(every), user: true, review, remaining, created, outcome: None, retry_at: 0, retry_ms: 0, file, claimed: None, source: None });
        Ok(format!("{kind} timer {id} · next {} · every {interval}. {}", crate::clock::iso_from_ms(next), if review { "Busy: deferred; unchanged or not running: skipped, count unchanged." } else { "Busy or not running: skipped, never queued." }))
    })
}

pub struct CheckinTool;
impl Tool for CheckinTool {
    fn name(&self) -> &'static str { "checkin" }
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "checkin".into(),
            description: "Set a message to yourself to check on a background job, lane, run or box instead of waiting or polling. action set: text and either in (\"20m\", \"2h\", \"5s\") or at (ISO time with offset); optional every repeats it. list: your timers. cancel: remove one by id.".into(),
            parameters: json!({"type": "object", "required": ["action"], "properties": {
                "action": {"type": "string", "enum": ["set", "list", "cancel"]},
                "text": {"type": "string"}, "in": {"type": "string"}, "at": {"type": "string"},
                "every": {"type": "string"}, "id": {"type": "string"}
            }}),
            constrained_sampling: None,
        }
    }
    fn execute<'a>(&'a self, arguments: Map<String, Value>, context: &'a ToolContext) -> ToolFuture<'a> {
        Box::pin(async move {
            let file = context.session_file.clone().ok_or("Check-ins need a saved session.")?;
            let agent_dir = crate::config::agent_dir();
            let output = tokio::task::spawn_blocking(move || store(&agent_dir, |timers| {
                let need = |key: &str| arguments.get(key).and_then(Value::as_str).ok_or_else(|| format!("{key} is needed"));
                match need("action")? {
                    "set" => {
                        let now = crate::clock::now_ms();
                        let next = match (arguments.get("at").and_then(Value::as_str), arguments.get("in").and_then(Value::as_str)) {
                            (Some(at), None) => time_ms(at).ok_or("at needs an ISO time with its offset")?,
                            (None, Some(after)) => now.checked_add(duration_ms(after).ok_or("in needs a duration like 20m, 2h or 5s")?).ok_or("duration is too large")?,
                            _ => return Err("give exactly one of at or in".into()),
                        };
                        let every_ms = arguments.get("every").and_then(Value::as_str).map(|every| duration_ms(every).ok_or("every needs a duration like 20m, 2h or 5s")).transpose()?;
                        let id = uuid::Uuid::new_v4().to_string();
                        timers.push(Checkin { id: id.clone(), text: need("text")?.to_string(), next, every_ms, user: false, review: false, remaining: None, created: now, outcome: None, retry_at: 0, retry_ms: 0, file, claimed: None, source: None });
                        Ok(format!("Check-in {id} set for {}.", crate::clock::iso_from_ms(next)))
                    }
                    "list" => {
                        let lines: Vec<_> = timers.iter().filter(|timer| !timer.user && timer.file == file).map(|timer| format!("{} · next {} · every {} · {}", timer.id, crate::clock::iso_from_ms(timer.next), timer.every_ms.map_or("once".into(), |every| format!("{}s", every / 1_000)), timer.text)).collect();
                        Ok(if lines.is_empty() { "No check-ins.".into() } else { lines.join("\n") })
                    }
                    "cancel" => {
                        let id = need("id")?;
                        let before = timers.len();
                        timers.retain(|timer| timer.user || timer.file != file || timer.id != id);
                        Ok(if timers.len() < before { format!("Cancelled {id}.") } else { format!("No check-in {id}.") })
                    }
                    other => Err(format!("No action {other:?}")),
                }
            })).await.map_err(|error| error.to_string())??;
            Ok(ToolOutput { content: vec![Content::text(output)], details: None, is_error: false })
        })
    }
}
