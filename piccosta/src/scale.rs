// The model scale: the kinds of work, which model does each, and which takes over when a plan
// runs low. Each kind is also a subagent job (tools/subagent.rs): the subagent tool hands work to
// a kind by name. All of it is one file: piccosta/scale.json ships compiled in, and the user's
// <agent dir>/scale.json changes it by name, field by field (a kind it names gets those fields, new
// names join; "off": true drops one, false brings back one that ships off). /scale shows it,
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
// to the back of the list, so the next one takes over until the window resets.
//
// Plans report their usage as they're used: Claude Code in its rate_limit_event messages, Codex in
// its response headers. piccosta keeps the latest in <agent dir>/usage.json.

use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const DEFAULTS: &str = include_str!("../scale.json");
// The share of a usage window past which a plan counts as running low, unless the file says.
const LOW_USAGE_PERCENT: f64 = 80.0;
// How long a reading that doesn't say when its window resets (Codex's) counts: long enough to
// steer the next few sessions, short enough that a plan read as low is tried again within the hour.
const READING_SECONDS: f64 = 3600.0;

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

// A section's entries that aren't off, by name.
fn entries(scale: &Map<String, Value>, section: &str) -> Vec<(String, Map<String, Value>)> {
    let all = scale.get(section).and_then(Value::as_object).cloned().unwrap_or_default();
    all.into_iter().filter_map(|(name, entry)| Some((name, entry.as_object()?.clone()))).filter(|(_, entry)| entry.get("off").and_then(Value::as_bool) != Some(true)).collect()
}

// The kinds of work as subagent jobs, each as its name and fields.
pub fn jobs(agent_dir: &Path) -> Vec<(String, Map<String, Value>)> {
    entries(&merged(agent_dir), "uses")
}

// The tools kinds of work own, each with the kind that owns it.
pub fn owned(agent_dir: &Path) -> Vec<(String, String)> {
    let owns = |entry: &Map<String, Value>| entry.get("owns").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect::<Vec<_>>();
    jobs(agent_dir).into_iter().flat_map(|(kind, entry)| owns(&entry).into_iter().map(move |tool| (tool, kind.clone()))).collect()
}

// Set one use's models (first choice first) in the user's file, leaving the rest of it as it is.
pub fn set_use(agent_dir: &Path, name: &str, models: &[String]) -> Result<(), String> {
    find(agent_dir, name)?;
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

// Every use that isn't off.
pub fn uses(agent_dir: &Path) -> Vec<Use> {
    entries(&merged(agent_dir), "uses")
        .into_iter()
        .map(|(name, entry)| Use {
            name,
            purpose: entry.get("for").and_then(Value::as_str).unwrap_or_default().to_string(),
            models: entry.get("models").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect(),
            compact_at: entry.get("compactAt").and_then(Value::as_u64),
        })
        .collect()
}

pub fn find(agent_dir: &Path, name: &str) -> Result<Use, String> {
    let all = uses(agent_dir);
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

// The plan a model spends: the account behind its provider.
fn plan(model: &str) -> &str {
    let provider = model.split('/').next().unwrap_or_default();
    match provider {
        crate::provider::claude::PROVIDER => "claude",
        "openai-codex" => "codex",
        other => other,
    }
}

fn usage_path(agent_dir: &Path) -> std::path::PathBuf {
    agent_dir.join("usage.json")
}

// Note a plan's usage in one of its windows: the percent used and when it resets (epoch seconds).
// A failure to write only loses this reading.
pub fn record(plan: &str, window: &str, used_percent: f64, resets_at: Option<f64>) {
    let path = usage_path(&crate::config::agent_dir());
    let mut usage: Map<String, Value> = std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
    let windows = usage.entry(plan.to_string()).or_insert_with(|| json!({}));
    if let Some(windows) = windows.as_object_mut() {
        windows.insert(window.to_string(), json!({ "usedPercent": used_percent, "resetsAt": resets_at, "seenAt": crate::clock::now_ms() / 1000 }));
    }
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&usage).unwrap_or_default());
}

// The highest share used of any of a plan's windows that hasn't reset (or, with no reset time,
// gone stale) since it was read.
fn used(agent_dir: &Path, plan: &str) -> Option<f64> {
    let usage: Value = serde_json::from_str(&std::fs::read_to_string(usage_path(agent_dir)).ok()?).ok()?;
    let now = (crate::clock::now_ms() / 1000) as f64;
    let current = |window: &&Value| match window.get("resetsAt").and_then(Value::as_f64) {
        Some(reset) => reset > now,
        None => window.get("seenAt").and_then(Value::as_f64).is_some_and(|seen| seen + READING_SECONDS > now),
    };
    usage.get(plan)?.as_object()?.values().filter(current).filter_map(|window| window.get("usedPercent")?.as_f64()).reduce(f64::max)
}

fn low(agent_dir: &Path, model: &str) -> bool {
    let limit = merged(agent_dir).get("lowUsagePercent").and_then(Value::as_f64).unwrap_or(LOW_USAGE_PERCENT);
    used(agent_dir, plan(model)).is_some_and(|percent| percent >= limit)
}

// A use's models in the order to try them now: those whose plan isn't running low, then the rest.
pub fn order(agent_dir: &Path, entry: &Use) -> Vec<String> {
    let (low, fine): (Vec<String>, Vec<String>) = entry.models.iter().cloned().partition(|model| low(agent_dir, model));
    fine.into_iter().chain(low).collect()
}

// The model and level a use runs on now.
pub fn pick(agent_dir: &Path, name: &str) -> Result<(String, Option<String>), String> {
    let entry = find(agent_dir, name)?;
    let first = order(agent_dir, &entry).into_iter().next().ok_or_else(|| format!("The \"{name}\" use lists no models"))?;
    let (model, level) = split(&first);
    Ok((model.to_string(), level.map(str::to_string)))
}

// The scale as text, for a prompt and /scale: each use with its models (low ones marked), and
// how much of each plan is used.
pub fn describe(agent_dir: &Path) -> String {
    let mut lines = Vec::new();
    for entry in uses(agent_dir) {
        let models: Vec<String> = order(agent_dir, &entry).iter().map(|model| if low(agent_dir, model) { format!("{model} (plan running low)") } else { model.clone() }).collect();
        let compact = entry.compact_at.map_or(String::new(), |tokens| format!(" · compacts at {}k tokens", tokens / 1000));
        lines.push(format!("- {}: {}. {}{compact}", entry.name, entry.purpose, models.join(", then ")));
    }
    let plans: Vec<String> = ["claude", "codex"].iter().filter_map(|plan| used(agent_dir, plan).map(|percent| format!("{plan} {percent:.0}% used"))).collect();
    if !plans.is_empty() {
        lines.push(format!("Plan usage, latest reading: {}.", plans.join(", ")));
    }
    lines.join("\n")
}

// The compaction point a scale use sets for a model, if one does.
pub fn compact_at(agent_dir: &Path, model: &str) -> Option<u64> {
    uses(agent_dir).into_iter().filter(|entry| entry.models.iter().any(|listed| split(listed).0 == model)).find_map(|entry| entry.compact_at)
}
