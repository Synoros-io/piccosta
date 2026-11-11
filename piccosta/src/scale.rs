// The model scale: the kinds of work, which model does each, and which takes over when a plan
// runs low. Each kind is also a subagent job (tools/subagent.rs): the subagent tool hands work to
// a kind by name. All of it is one file: piccosta/scale.json ships compiled in, and the user's
// <agent dir>/scale.json changes it by name, field by field (a kind it names gets those fields, new
// names join; "off": true drops one from the lists and the jobs, false brings back one that ships
// off; one that's off still runs what names it, as an advisor names advice). /scale shows it,
// /scale edit writes the whole of it there and opens it, and /scale <kind> <model[:level]> ...
// sets one kind's models.
//
// A kind (critical, general, review, ...) says what it's for and lists models as provider/id:level,
// first choice first; as a job, its tools (all when it lists none), whether it changes no files
// ("readOnly": its commands run in a sandbox that can't, and it gets no edit or write; the sandbox
// guards against a model's careless writes and deletes, not against one trying to escape, see
// tools/readonly.rs) and its instructions, which go on the end of the subagent's system prompt. A
// session the sessions tool starts for a kind gets its tools and readOnly too (host.rs). The
// shipped defaults give the reading kinds (review, reading, search, exploration, advice) read and
// a read-only bash, with the web tools for search and advice only; monitoring read and a bash that
// isn't sandboxed, since ssh makes sockets (ControlMaster); bugs read and bash, tests no web
// tools, and the rest everything. A kind can own tools ("owns"): a session that can
// hand work to it leaves those tools to its job (host.rs), so that work runs on its models.
// A model whose plan is running low (any of its usage windows at `lowUsagePercent` or more) goes
// to the back of the list, so the next one takes over until the window resets. A model no
// connected account serves doesn't count: the shipped kinds name models on every account
// piccosta knows as first choices, and each user gets those of the accounts they have. A kind none
// of whose models is served runs on the session's or the default model (pick), so an install with
// none of those accounts works as it is, and its own scale.json names the models it would rather.
//
// Plans report their usage as they're used: Claude Code in its rate_limit_event messages, Codex in
// its response headers and websocket events, opencode-go when it refuses a request past a limit.
// A piccosta that stays up (the TUI, rpc, ACP) also asks each connected plan, as its own tools do
// (ask): at start, every `usage.refreshSeconds`, and after a turn ends. piccosta keeps the latest
// in <agent dir>/usage.json, by plan, each window with usedPercent, resetsAt (epoch seconds),
// seenAt and, when known, windowMinutes; an "extra" entry notes usage bought past the plan.
// The pane header, /usage, rpc's stats and ACP's usage_update show it (plan_usage).

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const DEFAULTS: &str = include_str!("../scale.json");
// The share of a usage window past which a plan counts as running low, unless the file says.
const LOW_USAGE_PERCENT: f64 = 80.0;
// How long a reading that doesn't say when its window resets (opencode-go's refusal without a
// Retry-After) counts: long enough to
// steer the next few sessions, short enough that a plan read as low is tried again within the hour.
const READING_SECONDS: f64 = 3600.0;
// How often a piccosta that stays up asks the plans, unless settings say `usage.refreshSeconds`:
// the Codex CLI and Claude Code refresh their own displays about this often.
const REFRESH_SECONDS: u64 = 300;
// The least time between two asks a turn's end starts, so a run of short turns asks once.
const AFTER_TURN_GAP: std::time::Duration = std::time::Duration::from_secs(60);
// opencode's Go plan usage, unless settings say `usage.opencodeGoUrl`: GET with the API key, which
// answers {"usage": {"rolling" | "weekly" | "monthly": {"percent", "resetsAt" (ISO 8601)}}}
// (opencode issue #44189; read 2026-11-03).
const OPENCODE_GO_USAGE_URL: &str = "https://opencode.ai/zen/go/v1/usage";
// opencode's 5-hour and weekly windows, in minutes (opencode.ai/docs/go); its monthly one follows
// the subscription's day, so it has no fixed length.
const OPENCODE_WINDOW_MINUTES: [(&str, u64); 2] = [("rolling", 300), ("weekly", 10080)];

pub struct Use {
    pub name: String,
    pub purpose: String,
    // provider/id, each with its thinking level after a colon.
    pub models: Vec<String>,
    // Compact sessions of this use's models once their context passes this many tokens.
    pub compact_at: Option<u64>,
}

// The user's scale file.
pub fn path(agent_dir: &Path) -> PathBuf {
    agent_dir.join("scale.json")
}

// The user's changes: nothing when there's no file, an error naming it when it isn't valid JSON
// (// comments are allowed, as in settings).
pub fn changes(agent_dir: &Path) -> Result<Map<String, Value>, String> {
    let Ok(text) = std::fs::read_to_string(path(agent_dir)) else { return Ok(Map::new()) };
    serde_json::from_str(&crate::model::strip_json_comments(&text)).map_err(|error| format!("{}: {error}", path(agent_dir).display()))
}

// The scale in effect: the defaults with the user's changes on top. An invalid file changes
// nothing (/scale says why).
fn merged(agent_dir: &Path) -> Map<String, Value> {
    let mut scale: Map<String, Value> = serde_json::from_str(DEFAULTS).unwrap_or_default();
    for (key, change) in changes(agent_dir).unwrap_or_default() {
        let Some(section) = scale.get_mut(&key).and_then(Value::as_object_mut).filter(|_| change.is_object()) else {
            scale.insert(key, change);
            continue;
        };
        for (name, fields) in change.as_object().into_iter().flatten() {
            let entry = section.entry(name.clone()).or_insert_with(|| json!({}));
            if let (Some(entry), Some(fields)) = (entry.as_object_mut(), fields.as_object()) {
                entry.extend(fields.clone());
            }
        }
    }
    scale
}

// A section's entries that aren't off (with `off`, those that are too), by name.
fn entries(scale: &Map<String, Value>, section: &str, off: bool) -> Vec<(String, Map<String, Value>)> {
    let all = scale.get(section).and_then(Value::as_object).cloned().unwrap_or_default();
    all.into_iter().filter_map(|(name, entry)| Some((name, entry.as_object()?.clone()))).filter(|(_, entry)| off || entry.get("off").and_then(Value::as_bool) != Some(true)).collect()
}

// The kinds of work as subagent jobs, each as its name and fields.
pub fn jobs(agent_dir: &Path) -> Vec<(String, Map<String, Value>)> {
    entries(&merged(agent_dir), "uses", false)
}

// One kind's fields, off or not, for a session started for it.
pub fn job(agent_dir: &Path, name: &str) -> Option<Map<String, Value>> {
    entries(&merged(agent_dir), "uses", true).into_iter().find_map(|(kind, entry)| (kind == name).then_some(entry))
}

// The tools kinds of work own, each with the kind that owns it.
pub fn owned(agent_dir: &Path) -> Vec<(String, String)> {
    let owns = |entry: &Map<String, Value>| entry.get("owns").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>();
    jobs(agent_dir).into_iter().flat_map(|(kind, entry)| owns(&entry).into_iter().map(move |tool| (tool, kind.clone()))).collect()
}

// Set one use's models (first choice first) in the user's file, leaving the rest of it as it is.
pub fn set_use(agent_dir: &Path, name: &str, models: &[String]) -> Result<(), String> {
    find(agent_dir, name, true)?;
    let mut file = changes(agent_dir)?;
    let section = file.entry("uses").or_insert_with(|| json!({}));
    let entry = section.as_object_mut().ok_or("its \"uses\" isn't an object")?.entry(name).or_insert_with(|| json!({}));
    entry.as_object_mut().ok_or_else(|| format!("its use \"{name}\" isn't an object"))?.insert("models".into(), json!(models));
    std::fs::write(path(agent_dir), serde_json::to_string_pretty(&file).unwrap_or_default() + "\n").map_err(|error| error.to_string())
}

// Write the whole scale in effect to the user's file (their changes are in it), so all of it is
// there to change. A file that isn't valid is left as it is, to be fixed. Output: the file's path.
pub fn write_out(agent_dir: &Path) -> Result<PathBuf, String> {
    let file = path(agent_dir);
    if changes(agent_dir).is_ok() {
        std::fs::create_dir_all(agent_dir).map_err(|error| error.to_string())?;
        std::fs::write(&file, serde_json::to_string_pretty(&merged(agent_dir)).unwrap_or_default() + "\n").map_err(|error| format!("{}: {error}", file.display()))?;
    }
    Ok(file)
}

// Every use that isn't off; every_use with `off`, those that are too.
pub fn uses(agent_dir: &Path) -> Vec<Use> {
    every_use(agent_dir, false)
}

fn every_use(agent_dir: &Path, off: bool) -> Vec<Use> {
    entries(&merged(agent_dir), "uses", off)
        .into_iter()
        .map(|(name, entry)| Use {
            name,
            purpose: entry.get("for").and_then(Value::as_str).unwrap_or_default().to_string(),
            models: entry.get("models").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect(),
            compact_at: entry.get("compactAt").and_then(Value::as_u64),
        })
        .collect()
}

// A use by name; with `off`, one that's off too (an advisor's advice).
pub fn find(agent_dir: &Path, name: &str, off: bool) -> Result<Use, String> {
    let all = every_use(agent_dir, off);
    let names: Vec<String> = all.iter().map(|entry| entry.name.clone()).collect();
    all.into_iter().find(|entry| entry.name == name).ok_or_else(|| format!("No use \"{name}\" in the model scale: {}", names.join(", ")))
}

// A model's provider/id and its level, from provider/id:level.
pub fn split(model: &str) -> (&str, Option<&str>) {
    match model.rsplit_once(':') {
        Some((id, level)) if crate::provider::THINKING_LEVELS.contains(&level) => (id, Some(level)),
        _ => (model, None),
    }
}

// The plan a model spends: its account's (auth::ACCOUNTS), else its provider's own name.
fn plan(model: &str) -> &str {
    let provider = model.split('/').next().unwrap_or_default();
    crate::auth::account(provider).map_or(provider, |account| account.plan)
}

// Whether a scale model (provider/id:level) is one a connected account serves now. The model
// lists hold only those (model.rs), so it's whether the registry has it; with models.json
// invalid every model counts, and the run that uses one says what's wrong.
fn connected(agent_dir: &Path) -> impl Fn(&str) -> bool {
    let registry = crate::model::ModelRegistry::load(&agent_dir.join("models.json")).ok();
    move |model| registry.as_ref().is_none_or(|registry| registry.resolve(split(model).0).is_ok_and(|model| !model.interactive_only))
}

fn usage_path(agent_dir: &Path) -> std::path::PathBuf {
    agent_dir.join("usage.json")
}

// Note a plan's usage in one of its windows: the percent used and when it resets (epoch seconds).
// A failure to write only loses this reading.
pub fn record(plan: &str, window: &str, used_percent: f64, resets_at: Option<f64>) {
    store(plan, window, Some(json!({ "usedPercent": used_percent, "resetsAt": resets_at, "seenAt": crate::clock::now_ms() / 1000 })));
}

// Drop a window's reading: the plan says the window is off.
pub fn forget(plan: &str, window: &str) {
    store(plan, window, None);
}

// A reading's fields go over the window's (its windowMinutes, from an ask, stays).
fn store(plan: &str, window: &str, reading: Option<Value>) {
    update(|usage| {
        let windows = usage.entry(plan.to_string()).or_insert_with(|| json!({}));
        if let Some(windows) = windows.as_object_mut() {
            match reading.and_then(|reading| reading.as_object().cloned()) {
                Some(reading) => windows.entry(window.to_string()).or_insert_with(|| json!({})).as_object_mut().into_iter().for_each(|old| old.extend(reading.clone())),
                None => drop(windows.remove(window)),
            };
        }
    });
}

// Put what a plan said when asked in place of everything known about it.
pub fn replace(plan: &str, mut windows: Map<String, Value>) {
    let seen = crate::clock::now_ms() / 1000;
    windows.values_mut().filter_map(Value::as_object_mut).for_each(|window| drop(window.insert("seenAt".into(), seen.into())));
    update(|usage| drop(usage.insert(plan.to_string(), Value::Object(windows))));
}

fn update(change: impl FnOnce(&mut Map<String, Value>)) {
    let path = usage_path(&crate::config::agent_dir());
    let mut usage: Map<String, Value> = std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
    change(&mut usage);
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&usage).unwrap_or_default());
}

// usage.json as it is, read again only when it has changed: the pane header asks on every draw.
fn readings(agent_dir: &Path) -> Map<String, Value> {
    static READ: std::sync::Mutex<Option<(std::time::SystemTime, Map<String, Value>)>> = std::sync::Mutex::new(None);
    let path = usage_path(agent_dir);
    let Ok(changed) = std::fs::metadata(&path).and_then(|meta| meta.modified()) else { return Map::new() };
    let mut read = READ.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if read.as_ref().is_none_or(|(when, _)| *when != changed) {
        let usage = std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
        *read = Some((changed, usage));
    }
    read.as_ref().map(|(_, usage)| usage.clone()).unwrap_or_default()
}

// A plan's windows that haven't reset (or, with no reset time, gone stale) since they were read,
// by name. Codex sends a reading with every response, so one of its readings with no reset time
// stands until the next replaces it: a weekly window can't go stale in an hour.
fn windows(agent_dir: &Path, plan: &str) -> Vec<(String, Value)> {
    let now = (crate::clock::now_ms() / 1000) as f64;
    let current = |window: &Value| match window.get("resetsAt").and_then(Value::as_f64) {
        Some(reset) => reset > now,
        None => plan == "codex" || window.get("seenAt").and_then(Value::as_f64).is_some_and(|seen| seen + READING_SECONDS > now),
    };
    let usage = readings(agent_dir);
    usage.get(plan).and_then(Value::as_object).into_iter().flatten().filter(|(_, window)| current(window)).map(|(name, window)| (name.clone(), window.clone())).collect()
}

// The highest share used of any of a plan's current windows.
fn used(agent_dir: &Path, plan: &str) -> Option<f64> {
    windows(agent_dir, plan).iter().filter_map(|(_, window)| window.get("usedPercent")?.as_f64()).reduce(f64::max)
}

fn low_usage_percent(agent_dir: &Path) -> f64 {
    merged(agent_dir).get("lowUsagePercent").and_then(Value::as_f64).unwrap_or(LOW_USAGE_PERCENT)
}

// What a provider's plan has used now, for the pane header, /usage, rpc and ACP: null for a
// provider with no plan, else {plan, windows: [{name, label, usedPercent, resetsAt, low}], extra}
// (windows empty while the plan's usage is unknown).
pub fn plan_usage(agent_dir: &Path, provider: &str) -> Value {
    let Some(account) = crate::auth::account(provider) else { return Value::Null };
    let limit = low_usage_percent(agent_dir);
    let current = windows(agent_dir, account.plan);
    let extra = current.iter().find_map(|(_, window)| window.get("note").cloned());
    let windows: Vec<Value> = current.into_iter().filter_map(|(name, window)| {
        let used = window.get("usedPercent")?.as_f64()?;
        // A window's length names it (5h, 7d) when it's known, else its own name, shortened.
        let label = match window.get("windowMinutes").and_then(Value::as_u64) {
            Some(minutes) if minutes % 1440 == 0 => format!("{}d", minutes / 1440),
            Some(minutes) if minutes % 60 == 0 => format!("{}h", minutes / 60),
            Some(minutes) => format!("{minutes}m"),
            None => name.replace("five_hour", "5h").replace("seven_day", "7d").replace('_', " "),
        };
        Some(json!({"name": name, "label": label, "usedPercent": used, "resetsAt": window.get("resetsAt"), "low": used >= limit}))
    }).collect();
    json!({"plan": account.plan_name, "windows": windows, "extra": extra})
}

// /usage's lines for every connected plan: each window, its share used and when it resets (local
// time), and what's bought past the plan where the plan says.
pub fn report(agent_dir: &Path) -> String {
    let registry = crate::model::ModelRegistry::load(&agent_dir.join("models.json")).ok();
    let offset = crate::clock::local_now().1;
    let mut lines = Vec::new();
    for account in crate::auth::ACCOUNTS.iter().filter(|account| registry.as_ref().is_some_and(|registry| registry.models().iter().any(|model| model.provider == account.provider))) {
        let usage = plan_usage(agent_dir, account.provider);
        let windows = usage["windows"].as_array().cloned().unwrap_or_default();
        if windows.is_empty() {
            lines.push(format!("{:<12}unknown", account.plan_name));
        }
        for (index, window) in windows.iter().enumerate() {
            let resets = window["resetsAt"].as_i64().map_or(String::new(), |at| format!(", resets {}", crate::clock::iso_from_ms((at + offset) * 1000)[..16].replace('T', " ")));
            let name = if index == 0 { account.plan_name } else { "" };
            lines.push(format!("{name:<12}{:<8}{:>3.0}% used{resets}", window["label"].as_str().unwrap_or_default(), window["usedPercent"].as_f64().unwrap_or_default()));
        }
        if let Some(extra) = usage["extra"].as_str() {
            lines.push(format!("{:<12}{extra}", ""));
        }
    }
    lines.join("\n")
}

// A timer reports observations, not a fresh account fetch. Keep expired windows visible: a reset
// without a later reading says nothing about how much of the new window has been used.
pub fn latest_usage(agent_dir: &Path) -> String {
    let now = crate::clock::now_ms() as f64 / 1000.0;
    let at = |seconds: f64| crate::clock::iso_from_ms((seconds * 1000.0) as i64);
    let mut lines = Vec::new();
    for (plan, windows) in readings(agent_dir) {
        for (name, window) in windows.as_object().into_iter().flatten() {
            let seen = window["seenAt"].as_f64();
            let when = seen.map_or("unknown time".into(), at);
            let used = window["usedPercent"].as_f64().map_or("usage unknown".into(), |used| format!("{used}% used"));
            let reset = window["resetsAt"].as_f64().map_or("reset time unknown".into(), |reset| {
                let when = at(reset);
                if reset <= now && seen.is_none_or(|seen| seen < reset) { format!("reset at {when}, not read since") } else { format!("resets at {when}") }
            });
            let note = window["note"].as_str().map_or(String::new(), |note| format!("; {note}"));
            lines.push(format!("{plan} / {name}: {used}; seen at {when}; {reset}{note}"));
        }
    }
    if lines.is_empty() { "No usage readings known.".into() } else { lines.join("\n") }
}

// Whether this process asks the plans (watch), and when it last did.
static WATCHING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LAST_ASKED: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

// Ask the plans now and every `usage.refreshSeconds` after, in the background.
pub fn watch() {
    WATCHING.store(true, std::sync::atomic::Ordering::Relaxed);
    let settings = crate::config::Settings::load(&crate::config::agent_dir(), None);
    let period = settings.get("usage.refreshSeconds").and_then(Value::as_u64).unwrap_or(REFRESH_SECONDS).max(1);
    tokio::spawn(async move {
        loop {
            ask().await;
            tokio::time::sleep(std::time::Duration::from_secs(period)).await;
        }
    });
}

// A turn ended: ask the plans again, unless they were asked in the last minute or this process
// doesn't watch them (print mode).
pub fn after_turn() {
    let recent = LAST_ASKED.lock().ok().and_then(|last| *last).is_some_and(|last| last.elapsed() < AFTER_TURN_GAP);
    if WATCHING.load(std::sync::atomic::Ordering::Relaxed) && !recent {
        tokio::spawn(ask());
    }
}

// Ask each connected plan how much of it is used, noting what each says. One that can't be asked
// keeps what was known.
async fn ask() {
    if let Ok(mut last) = LAST_ASKED.lock() {
        *last = Some(std::time::Instant::now());
    }
    let agent_dir = crate::config::agent_dir();
    let Ok(registry) = crate::model::ModelRegistry::load(&agent_dir.join("models.json")) else { return };
    let paths = crate::auth::CredentialPaths::from_environment(&agent_dir);
    for account in &crate::auth::ACCOUNTS {
        let Some(model) = registry.models().iter().find(|model| model.provider == account.provider) else { continue };
        let _ = match account.owner {
            crate::auth::Owner::ClaudeCode => crate::provider::claude::ask_usage().await,
            _ => match crate::auth::credential_for(account.provider, registry.provider_config(account.provider), &paths).await {
                Ok(credential) if account.owner == crate::auth::Owner::CodexCli => crate::provider::codex::ask_usage(model, &credential).await,
                Ok(credential) => ask_opencode(&agent_dir, &credential).await,
                Err(error) => Err(error),
            },
        };
    }
}

async fn ask_opencode(agent_dir: &Path, credential: &crate::auth::Credential) -> Result<(), String> {
    let settings = crate::config::Settings::load(agent_dir, None);
    let url = settings.get("usage.opencodeGoUrl").and_then(Value::as_str).unwrap_or(OPENCODE_GO_USAGE_URL);
    let client = crate::provider::http_client().build().map_err(|error| error.to_string())?;
    let response = client.get(url).bearer_auth(&credential.key).timeout(crate::provider::HEADER_TIMEOUT).send().await.map_err(|error| error.to_string())?;
    let answer: Value = response.error_for_status().map_err(|error| error.to_string())?.json().await.map_err(|error| error.to_string())?;
    let mut windows = Map::new();
    for (name, window) in answer["usage"].as_object().ok_or("no usage in opencode's answer")? {
        let Some(used) = window["percent"].as_f64() else { continue };
        let resets_at = window["resetsAt"].as_str().and_then(crate::clock::ms_from_iso).map(|ms| ms / 1000);
        let minutes = OPENCODE_WINDOW_MINUTES.iter().find(|(known, _)| known == name).map(|(_, minutes)| minutes);
        windows.insert(name.clone(), json!({"usedPercent": used, "resetsAt": resets_at, "windowMinutes": minutes}));
    }
    replace("opencode-go", windows);
    Ok(())
}

pub fn low(agent_dir: &Path, model: &str) -> bool {
    used(agent_dir, plan(model)).is_some_and(|percent| percent >= low_usage_percent(agent_dir))
}

// A use's models in the order to try them now: those whose plan isn't running low, then the rest.
pub fn order(agent_dir: &Path, entry: &Use) -> Vec<String> {
    let (low, fine): (Vec<String>, Vec<String>) = entry.models.iter().cloned().partition(|model| low(agent_dir, model));
    fine.into_iter().chain(low).collect()
}

// A use's models in the order to try them now, without those no connected account serves.
pub fn available(agent_dir: &Path, entry: &Use) -> Vec<String> {
    let connected = connected(agent_dir);
    order(agent_dir, entry).into_iter().filter(|model| connected(model)).collect()
}

// A model, its level, and a line for the user about the choice, each when there is one.
pub type Pick = (Option<String>, Option<String>, Option<String>);

// The model and level a use runs on now: its first available model, or None for the model and a
// line saying so when no connected account serves any of them, so the caller runs on the session's
// or the default model. An error only for a use the scale doesn't have (or has off, without `off`).
pub fn pick(agent_dir: &Path, name: &str, off: bool) -> Result<Pick, String> {
    let entry = find(agent_dir, name, off)?;
    let Some(first) = available(agent_dir, &entry).into_iter().next() else {
        return Ok((None, None, Some(format!("None of the \"{name}\" kind's models is served by a connected account, so it runs on the session's model, or the default one; /scale {name} <model> sets one"))));
    };
    let (model, level) = split(&first);
    Ok((Some(model.to_string()), level.map(str::to_string), None))
}

// pick's line, with the model (provider/id) the session started on in the end.
pub fn fell_back(note: &str, model: &str) -> String {
    format!("{note}. This session runs on {model}.")
}

// The scale as text, for a prompt and /scale: each use with its models (low ones and those no
// connected account serves marked), and how much of each plan is used.
pub fn describe(agent_dir: &Path) -> String {
    let mut lines = Vec::new();
    let connected = connected(agent_dir);
    for entry in uses(agent_dir) {
        let mark = |model: &String| match (connected(model), low(agent_dir, model)) {
            (false, _) => format!("{model} (not connected)"),
            (true, true) => format!("{model} (plan running low)"),
            (true, false) => model.clone(),
        };
        let models: Vec<String> = order(agent_dir, &entry).iter().map(mark).collect();
        let compact = entry.compact_at.map_or(String::new(), |tokens| format!(" · compacts at {}k tokens", tokens / 1000));
        lines.push(format!("- {}: {}. {}{compact}", entry.name, entry.purpose, models.join(", then ")));
    }
    // Each plan the scale's models spend, and how much of it is used; one with no current reading
    // (opencode-go until it refuses a request) is unknown, which isn't the same as fine.
    let models: Vec<String> = uses(agent_dir).into_iter().flat_map(|entry| entry.models).collect();
    let plans: Vec<String> = crate::auth::ACCOUNTS
        .iter()
        .map(|account| account.plan)
        .filter(|name| models.iter().any(|model| plan(model) == *name))
        .map(|name| used(agent_dir, name).map_or(format!("{name} unknown"), |percent| format!("{name} {percent:.0}% used")))
        .collect();
    if !plans.is_empty() {
        lines.push(format!("Plan usage, latest reading: {}.", plans.join(", ")));
    }
    lines.join("\n")
}

// The compaction point a scale use sets for a model, if one does.
pub fn compact_at(agent_dir: &Path, model: &str) -> Option<u64> {
    uses(agent_dir).into_iter().filter(|entry| entry.models.iter().any(|listed| split(listed).0 == model)).find_map(|entry| entry.compact_at)
}
