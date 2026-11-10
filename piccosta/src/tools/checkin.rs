// Check-ins are messages to this session, not prompt state. Every runtime fires its own; the
// assistant service covers sessions nobody runs. One locked store lets the two meet safely.
use super::{Tool, ToolContext, ToolFuture, ToolOutput};
use crate::message::{Content, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{fs, path::{Path, PathBuf}, time::Duration};

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

#[derive(Clone, Serialize, Deserialize)]
struct Checkin {
    id: String,
    text: String,
    next: i64,
    every_ms: Option<i64>,
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
    let result = change(&mut timers)?;
    save(&path, &timers)?;
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

// The same clock in a frontend and the service; neither waits on a receiver on its own thread.
pub async fn watch(agent_dir: PathBuf, only: Option<PathBuf>) {
    let mut poll = tokio::time::interval(POLL);
    loop {
        poll.tick().await;
        let (agent_dir, only) = (agent_dir.clone(), only.clone());
        tokio::task::spawn_blocking(move || {
            if let Err(error) = fire(&agent_dir, only.as_deref()) { eprintln!("piccosta checkin: {error}"); }
        });
    }
}

// Claim under the store lock, then give each delivery its own blocking worker. A stopped
// receiver holds neither the store nor another receiver; a dead dispatcher leaves a claim we
// can take over. The occurrence's delivery id makes that takeover safe even after a lost answer.
fn fire(agent_dir: &Path, only: Option<&Path>) -> Result<(), String> {
    let owner = crate::runtime::Run::of(std::process::id() as i32).ok_or("could not identify check-in dispatcher")?;
    let due = store(agent_dir, |timers| {
        let now = crate::clock::now_ms();
        timers.retain(|timer| !crate::session::is_closed(agent_dir, &timer.file) && timer.file.exists());
        let mut due = Vec::new();
        for timer in timers {
            if only.is_some_and(|file| file != timer.file) || timer.next > now || timer.retry_at > now || timer.claimed.as_ref().is_some_and(|(owner, until)| *until > now && owner.alive()) { continue; }
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
    let id = match &claimed.source {
        Some(home) => format!("schedule:{}:{}:{}", home.display(), claimed.id, claimed.next),
        None => format!("checkin:{}:{}", claimed.id, claimed.next),
    };
    let text = format!("[{}, checkin {}]\n{}", crate::clock::iso_from_ms(crate::clock::now_ms()), claimed.id, claimed.text);
    let launch = crate::manager::checkin_launch(agent_dir, &claimed.file);
    let sent = crate::runtime::deliver_checkin(&claimed.file, &id, &text, launch.as_ref().map(|(cwd, log)| (cwd.as_path(), log.as_path())));
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
            (timer.retry_at, timer.retry_ms) = (0, 0);
            if let Some(every) = timer.every_ms.filter(|every| *every > 0) {
                // A repeat missed while nothing ran fires once, then keeps its rhythm.
                timer.next = now.saturating_add(every - (now - timer.next) % every);
                true
            } else { false }
        });
        Ok(())
    })
}

pub fn cancel_all(agent_dir: &Path, file: &Path) -> Result<(), String> {
    store(agent_dir, |timers| { timers.retain(|timer| crate::tools::paths::canonical(&timer.file) != file); Ok(()) })
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
                        timers.push(Checkin { id: id.clone(), text: need("text")?.to_string(), next, every_ms, retry_at: 0, retry_ms: 0, file, claimed: None, source: None });
                        Ok(format!("Check-in {id} set for {}.", crate::clock::iso_from_ms(next)))
                    }
                    "list" => {
                        let lines: Vec<_> = timers.iter().filter(|timer| timer.file == file).map(|timer| format!("{} · next {} · every {} · {}", timer.id, crate::clock::iso_from_ms(timer.next), timer.every_ms.map_or("once".into(), |every| format!("{}s", every / 1_000)), timer.text)).collect();
                        Ok(if lines.is_empty() { "No check-ins.".into() } else { lines.join("\n") })
                    }
                    "cancel" => {
                        let id = need("id")?;
                        let before = timers.len();
                        timers.retain(|timer| timer.file != file || timer.id != id);
                        Ok(if timers.len() < before { format!("Cancelled {id}.") } else { format!("No check-in {id}.") })
                    }
                    other => Err(format!("No action {other:?}")),
                }
            })).await.map_err(|error| error.to_string())??;
            Ok(ToolOutput { content: vec![Content::text(output)], details: None, is_error: false })
        })
    }
}
