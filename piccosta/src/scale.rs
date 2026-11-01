// The model scale: which model does which kind of work, and which takes over when a plan runs
// low. It ships with default assignments, each saying what it's for (piccosta/scale.json), and
// the user can change any of it.
//
// A use (critical, general, review, ...) says what it's for and lists models as provider/id:level,
// first choice first. The defaults are piccosta/scale.json; settings' `modelScale` changes a use
// field by field, adds new ones, and `"off": true` drops one; a project's .pi/settings.json
// does the same over the user's. A model whose plan is running low (any of its usage windows at
// `modelScaleLowUsage` percent or more, LOW_USAGE_PERCENT unless set) goes to the back of the
// list, so the next one takes over until the window resets.
//
// Plans report their usage as they're used: Claude Code in its rate_limit_event messages, Codex in
// its response headers. piccosta keeps the latest in <agent dir>/usage.json.

use serde_json::{json, Map, Value};
use std::path::Path;

const DEFAULTS: &str = include_str!("../scale.json");
// The share of a usage window past which a plan counts as running low.
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

// Every use, the defaults with the user's and the project's changes.
pub fn uses(settings: &crate::config::Settings) -> Vec<Use> {
    let mut scale: Map<String, Value> = serde_json::from_str(DEFAULTS).unwrap_or_default();
    for (name, change) in settings.get("modelScale").and_then(Value::as_object).into_iter().flatten() {
        if change.get("off").and_then(Value::as_bool) == Some(true) {
            scale.shift_remove(name);
            continue;
        }
        let entry = scale.entry(name.clone()).or_insert_with(|| json!({}));
        if let (Some(entry), Some(change)) = (entry.as_object_mut(), change.as_object()) {
            entry.extend(change.clone());
        }
    }
    scale
        .into_iter()
        .map(|(name, entry)| Use {
            name,
            purpose: entry.get("for").and_then(Value::as_str).unwrap_or_default().to_string(),
            models: entry.get("models").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect(),
            compact_at: entry.get("compactAt").and_then(Value::as_u64),
        })
        .collect()
}

pub fn find(settings: &crate::config::Settings, name: &str) -> Result<Use, String> {
    let all = uses(settings);
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

fn low(agent_dir: &Path, settings: &crate::config::Settings, model: &str) -> bool {
    let limit = settings.get("modelScaleLowUsage").and_then(Value::as_f64).unwrap_or(LOW_USAGE_PERCENT);
    used(agent_dir, plan(model)).is_some_and(|percent| percent >= limit)
}

// A use's models in the order to try them now: those whose plan isn't running low, then the rest.
pub fn order(agent_dir: &Path, settings: &crate::config::Settings, entry: &Use) -> Vec<String> {
    let (low, fine): (Vec<String>, Vec<String>) = entry.models.iter().cloned().partition(|model| low(agent_dir, settings, model));
    fine.into_iter().chain(low).collect()
}

// The model and level a use runs on now.
pub fn pick(agent_dir: &Path, settings: &crate::config::Settings, name: &str) -> Result<(String, Option<String>), String> {
    let entry = find(settings, name)?;
    let first = order(agent_dir, settings, &entry).into_iter().next().ok_or_else(|| format!("The \"{name}\" use lists no models"))?;
    let (model, level) = split(&first);
    Ok((model.to_string(), level.map(str::to_string)))
}

// The scale as text, for a prompt and /scale: each use with its models (low ones marked), and
// how much of each plan is used.
pub fn describe(agent_dir: &Path, settings: &crate::config::Settings) -> String {
    let mut lines = Vec::new();
    for entry in uses(settings) {
        let models: Vec<String> = order(agent_dir, settings, &entry).iter().map(|model| if low(agent_dir, settings, model) { format!("{model} (plan running low)") } else { model.clone() }).collect();
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
pub fn compact_at(settings: &crate::config::Settings, model: &str) -> Option<u64> {
    uses(settings).into_iter().filter(|entry| entry.models.iter().any(|listed| split(listed).0 == model)).find_map(|entry| entry.compact_at)
}
