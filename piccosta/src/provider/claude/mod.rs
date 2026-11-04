// Claude through Claude Code: the "claude-bridge" provider.
//
// On a Claude plan, Claude runs only through Claude Code, so each request goes to a `claude` child
// speaking its stream-json control protocol on stdin and stdout, with Claude Code reduced to a pipe:
// pi's context, system prompt and tools, no builtins, no settings files, no CLAUDE.md. pi's tools
// are served to it as one MCP server ("custom-tools") inside that same protocol, so a tool call
// reaches pi as a turn that ends on tool calls, and pi's results go back as the answers to Claude
// Code's tools/call requests.
//
// One live query per pi session. A turn that ends on tool calls parks the query: Claude Code waits
// inside tools/call. An answer holds it until the next turn or idle expiry; an interrupt closes
// it, keeping its records to resume with the same cached prefix (history.rs). A one-off summary
// (compaction's) is a query of its own that ends with its turn.
//
// Based on piccosta's TypeScript bridge, itself built on pi-claude-bridge (MIT, Eli Dickinson,
// see LICENSE.md), without the Agent SDK: the parts of its control protocol the
// bridge used (initialize with the MCP manifests, mcp_message, interrupt) are written here.
// Left out on purpose:
// - The SDK's stamps (CLAUDE_CODE_ENTRYPOINT=sdk-ts, CLAUDE_AGENT_SDK_VERSION): Claude Code puts
//   the entrypoint into its billing header, so it says what it is itself here.
// - The TypeScript bridge's removal of pi's docs block, which existed to get past Anthropic's
//   check for third-party apps on the plan. piccosta's prompt never has that block anyway.
// - onPayload and the rate-limit headers for after_provider_response (no extensions here), the
//   rate-limit notices (no UI yet), and the flattening of root anyOf/oneOf tool schemas (no
//   piccosta tool has one).
//
// Settings, each from the environment or else from the agent dir's settings.json, its "claudeBridge"
// object (so the assistant service, which gets no terminal's environment, sees the same ones). Never
// from a project's settings: they pick the account and the binary, which belong to the user, not to
// a project, and one piccosta process serves sessions in many projects at once.
//   CLAUDE_BRIDGE_CLAUDE_PATH, claudePath   the claude executable (default `claude` on PATH)
//   CLAUDE_BRIDGE_CONFIG_DIR, configDir     a Claude Code config dir for the children only (their
//                                  CLAUDE_CONFIG_DIR, and where their session files go); one with no
//                                  account record keeps the account's email out of the prompt, and
//                                  then needs CLAUDE_CODE_OAUTH_TOKEN (`claude setup-token`)
//   CLAUDE_BRIDGE_HOLD_SECONDS, holdSeconds   idle hold (300 s); 0 closes after each turn;
//                                  assistants and managers hold until their session closes
//   CLAUDE_BRIDGE_KEEP_SESSIONS, keepSessions   off: don't keep Claude Code's records between
//                                  queries (history.rs)
//   CLAUDE_BRIDGE_STRICT_MCP, strictMcp     off: let Claude Code load MCP servers from
//                                  ~/.claude.json and .mcp.json
//   CLAUDE_BRIDGE_ONE_M, oneM; CLAUDE_BRIDGE_LONG_CONTEXT_EXTRA_USAGE, longContextExtraUsage;
//   CLAUDE_BRIDGE_FORCE_200K, force200k     which models get 1M context (runtime_model); lists
//                                  are comma-separated in the environment, arrays in settings.json
//   CLAUDE_BRIDGE_NO_THINKING_OFF, noThinkingOff   models that can't turn thinking off
//   CLAUDE_BRIDGE_DEBUG_PATH, debugPath     append a debug log here
//   CLAUDE_BRIDGE_RECORD_STREAM, recordStream   append every line Claude Code writes here
// The plan (Max opens 1M on opus-4-6) and whether the login is an API key (the cache TTL) come
// from Claude Code itself: `claude auth status --json`, asked once per process (AUTH), given
// PROBE_DEADLINE like `claude --version`.

mod history;
pub use history::SIDECAR_SUFFIX;

use super::{new_output, transcript, Request, StreamEvent, BODY_IDLE_TIMEOUT};
use crate::clock;
use crate::message::{AssistantMessage, Content, Cost, CostState, Message, StopReason, ToolDefinition, Usage};
use crate::model::Model;
use history::{fingerprint, Shape};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{sleep_until, timeout, Instant};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub const PROVIDER: &str = "claude-bridge";
const MCP_SERVER: &str = "custom-tools";
pub const MCP_TOOL_PREFIX: &str = "mcp__custom-tools__";
// Claude Code keeps a tool result over 50,000 characters out of the conversation (a preview and a
// file path instead) unless the tool raises that ceiling, up to 500,000. pi cuts read and bash at
// 50 KB, so every tool declares the maximum and pi's results arrive whole (2.1.300's tool-result
// store, read in the binary; 2.1.316 against a stand-in API: a 51,341-character result arrives whole).
// Past the ceiling the model gets a 2,000-character preview and a path to the rest.
const MCP_RESULT_MAX_CHARS: u64 = 500_000;
// The ids Claude Code answers 1M context for when asked with [1m] (measured on every plan by
// 2026-11-02; a newer model serves 200K until it's measured and added to `claudeBridge.oneM`). An
// unentitled [1m] fails every turn, so the default is the bare id at 200K.
const MEASURED_ONE_M: [&str; 9] = ["claude-fable-5", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-opus-4-8",
    "claude-opus-4-7", "claude-sonnet-5", "claude-sonnet-5-5", "claude-haiku-5-5"];
const ONE_M: u64 = 1_000_000;
const TWO_HUNDRED_K: u64 = 200_000;
// Models Claude Code won't run with thinking off: given --thinking=disabled it still asks for
// adaptive thinking at its default effort (2.1.300 and 2.1.312 against a local mock API,
// 2026-10-29; the 4.5 models do send thinking disabled). The catalog fetched that day lets them
// turn it off, pi 1.2's didn't; marked off here, pi clamps "off" up to their lowest level instead.
// The default for `claudeBridge.noThinkingOff`.
const NO_THINKING_OFF: [&str; 1] = ["claude-haiku-5-5"];
// Claude Code refuses a prompt at window - min(max output, 20,000) - 3,000 tokens before sending it
// (2.1.300, read in the binary), so that is the window pi is given to compact against.
const OUTPUT_RESERVE_CAP: u64 = 20_000;
const BLOCKING_MARGIN: u64 = 3_000;
// pi's levels to Claude Code's efforts, for a level the model's own map leaves out.
const EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];
// What every child gets on top of pi's environment (each one checked against a captured request,
// 2026-10-25, Claude Code 2.1.300):
// - no claude.ai MCP servers, no auto-compaction (pi owns compaction), no update checks or telemetry;
// - no "<total_tokens>" reminders, no @file attachments (pi leaves "@path" text alone), no tool
//   search (every tool goes in full, as pi sends it);
// - MCP output cut only past 250,000 tokens: Claude Code estimates characters / 4 and counts the
//   tokens of anything over half the cap first, so twice 500,000 / 4 lets every whole result by;
// - tool descriptions never cut (its default is 4,096 characters);
// - no retries of its own: pi's provider retry count, 0 by default, so failures reach pi's auto-retry;
// - no remote feature flags. Two of them cut tool results to 2,000-character previews whatever the
//   tool declares: one sets a tool's ceiling by name, one caps a turn's results at 200,000
//   characters together. Turning off non-essential traffic already keeps the flags at their
//   defaults (off), but the flags' disk cache can be switched back on from the environment, and
//   this setting turns that off too (2.1.316, read in the binary and checked against a stand-in
//   API: five 51,000-character results in one turn arrive whole).
const CHILD_ENV: [(&str, &str); 10] = [("ENABLE_CLAUDEAI_MCP_SERVERS", "0"), ("DISABLE_AUTO_COMPACT", "1"),
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"), ("CLAUDE_CODE_TOTAL_TOKENS_REMINDER", "off"), ("CLAUDE_CODE_DISABLE_ATTACHMENTS", "1"),
    ("ENABLE_TOOL_SEARCH", "false"), ("MAX_MCP_OUTPUT_TOKENS", "250000"), ("CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH", "1000000"),
    ("CLAUDE_CODE_MAX_RETRIES", "0"), ("DISABLE_GROWTHBOOK", "1")];
// What a parent Claude Code session sets for the processes under it (what `env` showed inside a
// Claude Code Bash call, 2026-10-15, plus the two 2.1.312 sets on 2026-10-29). Left in, the child
// would act as part of that session. NODE_OPTIONS and DEBUG go too, as the SDK removed them.
const PARENT_ENV: [&str; 19] = ["CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_EFFORT", "CLAUDE_PID", "CLAUDE_PLUGIN_DATA", "CODEX_COMPANION_SESSION_ID",
    "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_EVEN_MORE_DANGEROUSLY_SKIP_PERMISSIONS", "CLAUDE_CODE_RESUME_INTERRUPTED_TURN", "NODE_OPTIONS", "DEBUG"];
// The settings file every provider child gets: no auto-memory, no CLAUDE.md or AGENTS.md (pi
// owns context files), no git status block (a git change would rewrite it and bust the cache), no
// hooks (they cost seconds and tokens per query, and an opened repo's would run untrusted).
const PROVIDER_SETTINGS: &str = r#"{"autoMemoryEnabled":false,"claudeMdExcludes":["**/CLAUDE.md","**/AGENTS.md","**/.claude/rules/**"],"includeGitInstructions":false,"disableAllHooks":true}"#;
// How long a parked tools/call waits before checking that pi was handed its call, and between
// checks after (pi-claude-bridge #226): far longer than the stream event announcing it can lag.
const LOST_CALL_CHECK: Duration = Duration::from_secs(120);
// How long a non-streamed answer's last tool call waits for another block before the turn ends:
// Claude Code writes all of a message's blocks at once (pi-claude-bridge #225).
const FALLBACK_SETTLE: Duration = Duration::from_millis(100);
// Closing a child: stdin ends, then SIGTERM after 2 s, then SIGKILL after 5 more (the SDK's close).
const CLOSE_GRACE: Duration = Duration::from_secs(2);
const TERM_GRACE: Duration = Duration::from_secs(5);
// Restarted requests are byte-identical; holding saves only startup (+449 ms to the first token,
// measured 2026-11-04). Keep it while the user replies, not after minutes away.
const DEFAULT_HOLD_SECONDS: u64 = 300;
// The longest shutdown waits for the queries' tasks to close their children that way.
const SHUTDOWN_WAIT: Duration = CLOSE_GRACE.saturating_add(TERM_GRACE).saturating_add(CLOSE_GRACE);
// The cache monitor's expiry window. Claude Code 2.1.316 asks for a
// 1 h cache on a subscription (ttl "1h", reason "subscriber", read in the binary) and 5 min with
// an API key; every cache write in 30 h of real claude-bridge sessions (2026-11-02) was a 1 h one.
const CACHE_TTL: Duration = Duration::from_secs(60 * 60);
const API_KEY_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
// The last Claude Code release the bridge was checked against: CHILD_ENV and
// MCP_RESULT_MAX_CHARS were read in this binary. A newer one gets a notice, since any of them
// could have been renamed.
const VERIFIED_VERSION: &str = "2.1.318";
// How much of a child's stderr is kept for a failure message: its last lines, where a refused
// option or a failed login is reported, without carrying a long log into the session.
const STDERR_TAIL: usize = 4096;
// How long `claude auth status` or `claude --version` gets before it's killed and taken as no
// answer: a normal one is done well inside it, and a hung one holds up registry loading or a turn
// for no longer. PROBE_OUTPUT_MAX is how much of its output is kept, far more than a version or a
// login status takes, so something it started that keeps writing can't grow it.
const PROBE_DEADLINE: Duration = Duration::from_secs(5);
const PROBE_OUTPUT_MAX: u64 = 64 * 1024;
// The custom entry the newer-than-verified notice is shown as (never part of the model's context).
pub const NOTICE_TYPE: &str = "piccosta-claude-notice";
// Claude Code's own wording for a failed connection, which pi's retry rules don't know; such a
// failure is led by pi's "Connection error" so it is retried.
const CONNECTION_FAILURES: [&str; 6] = ["Unable to connect to API", "Connection dropped (", "Connection refused \u{2014}", "Can't reach the API server",
    "No internet route", "Couldn't connect through your proxy"];

pub struct Settings {
    pub claude: String,
    pub config_dir: Option<PathBuf>,
    keep: bool,
    hold: Duration,
    strict_mcp: bool,
    one_m: Vec<String>,
    no_thinking_off: Vec<String>,
    extra_usage: bool,
    force_200k: Vec<String>,
    debug: Option<PathBuf>,
    record: Option<PathBuf>,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|value| value.trim().to_string()).filter(|value| !value.is_empty())
}

fn flag(name: &str) -> bool {
    env(name).is_some_and(|value| is_on(&value))
}

fn is_on(value: &str) -> bool {
    ["1", "true", "yes", "on"].contains(&value.to_lowercase().as_str())
}

pub static SETTINGS: LazyLock<Settings> = LazyLock::new(|| {
    // One bridge setting as text: CLAUDE_BRIDGE_<name> in the environment, else the agent dir's
    // claudeBridge.<key> (a list comes back comma-separated, as the environment writes it).
    let file = crate::config::Settings::load(&crate::config::agent_dir(), None);
    let setting = |name: &str, key: &str| env(&format!("CLAUDE_BRIDGE_{name}")).or_else(|| match file.get(&format!("claudeBridge.{key}"))? {
        Value::String(text) => Some(text.trim().to_string()).filter(|text| !text.is_empty()),
        Value::Array(items) => Some(items.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(",")),
        other => Some(other.to_string()),
    });
    let list = |name: &str, key: &str, default: &[&str]| setting(name, key)
        .map_or_else(|| default.iter().map(|id| id.to_string()).collect(), |ids| ids.split(',').map(|id| id.trim().to_string()).filter(|id| !id.is_empty()).collect());
    Settings {
        claude: setting("CLAUDE_PATH", "claudePath").unwrap_or_else(|| "claude".into()),
        config_dir: setting("CONFIG_DIR", "configDir").map(PathBuf::from),
        keep: setting("KEEP_SESSIONS", "keepSessions").is_none_or(|keep| is_on(&keep)),
        hold: Duration::from_secs(setting("HOLD_SECONDS", "holdSeconds").and_then(|seconds| seconds.parse().ok()).unwrap_or(DEFAULT_HOLD_SECONDS)),
        strict_mcp: setting("STRICT_MCP", "strictMcp").is_none_or(|strict| is_on(&strict)),
        one_m: list("ONE_M", "oneM", &MEASURED_ONE_M),
        no_thinking_off: list("NO_THINKING_OFF", "noThinkingOff", &NO_THINKING_OFF),
        extra_usage: setting("LONG_CONTEXT_EXTRA_USAGE", "longContextExtraUsage").is_some_and(|extra| is_on(&extra)),
        force_200k: list("FORCE_200K", "force200k", &[]),
        debug: setting("DEBUG_PATH", "debugPath").map(PathBuf::from),
        record: setting("RECORD_STREAM", "recordStream").map(PathBuf::from),
    }
});

// A claude command as the children run it: the bridge's executable and config dir.
fn claude_command() -> std::process::Command {
    let mut command = std::process::Command::new(&SETTINGS.claude);
    if let Some(dir) = &SETTINGS.config_dir {
        command.env("CLAUDE_CONFIG_DIR", dir);
    }
    command
}

// Input: the arguments of a quick claude command. Output: what it wrote to stdout (up to
// PROBE_OUTPUT_MAX), None when it couldn't be started or its output didn't close within
// PROBE_DEADLINE. Either way it's killed if still running and reaped before this returns. Its
// stdout is read on a thread of its own, so something it started that holds stdout open holds only
// that thread, never the caller.
fn probe(args: &[&str]) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut child = claude_command().args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let stdout = child.stdout.take()?;
    let (sender, output) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.take(PROBE_OUTPUT_MAX).read_to_end(&mut bytes);
        let _ = sender.send(bytes);
    });
    let output = output.recv_timeout(PROBE_DEADLINE).ok();
    // A kill after it exited does nothing; the wait reaps it.
    let _ = child.kill();
    let _ = child.wait();
    if output.is_none() {
        debug(&format!("`{} {}` gave no answer within {} s", SETTINGS.claude, args.join(" "), PROBE_DEADLINE.as_secs()));
    }
    output
}

// Claude Code's own account of its login (`claude auth status --json`: authMethod, subscriptionType,
// loggedIn), asked once per process. None when claude couldn't be run, didn't answer in time or said
// nothing readable; a logged-out one still answers, with loggedIn false.
pub static AUTH: LazyLock<Option<Value>> = LazyLock::new(|| {
    let status: Value = serde_json::from_slice(&probe(&["auth", "status", "--json"])?).ok()?;
    debug(&format!("auth: method {}, plan {}", status["authMethod"], status["subscriptionType"]));
    Some(status)
});

// Whether Claude Code bills by the token: logged in with no plan (an API key, or a cloud's), as
// its auth status says. When it couldn't be asked, the plan, as a Claude Code login most often is.
pub fn billed_by_token() -> bool {
    AUTH.as_ref().is_some_and(|auth| auth["loggedIn"] == true && auth["subscriptionType"].as_str().is_none())
}

// Price a turn's usage (responses::calculate_cost): on a plan nothing, at the user's own prices for
// the model when models.json gives some, else at what Claude Code reckoned it cost (`reported`),
// else unknown.
fn price(model: &Model, usage: &mut Usage, reported: Option<f64>) {
    match reported.filter(|_| billed_by_token() && !model.cost.supplied) {
        Some(total) => usage.cost = Cost { total, plan: Some(false), ..Default::default() },
        None => super::responses::calculate_cost(model, usage),
    }
}

// The running Claude Code's version (`claude --version`), asked once per process when the bridge
// first starts a child, and logged. None when it couldn't be read in time.
static VERSION: OnceLock<Option<String>> = OnceLock::new();
static NOTICED: AtomicBool = AtomicBool::new(false);

fn version() -> Option<&'static str> {
    VERSION.get_or_init(|| {
        let version = String::from_utf8_lossy(&probe(&["--version"])?).split_whitespace().next()?.to_string();
        debug(&format!("Claude Code {version} (last verified {VERIFIED_VERSION})"));
        Some(version)
    }).as_deref()
}

// Once per process, when the Claude Code the bridge ran is newer than VERIFIED_VERSION: a line for
// the user saying so. Never asks for the version itself, so a process that never ran the bridge
// gets nothing.
pub fn take_notice() -> Option<String> {
    let version = VERSION.get()?.as_deref()?;
    let numbers = |version: &str| version.split('.').map(|part| part.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    (numbers(version) > numbers(VERIFIED_VERSION) && !NOTICED.swap(true, Ordering::Relaxed)).then(|| format!(
        "Claude Code {version} is newer than {VERIFIED_VERSION}, the last version the bridge was checked against; if tool results arrive cut short, look there first"))
}

fn append(path: &PathBuf, line: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

pub fn debug(line: &str) {
    if let Some(path) = &SETTINGS.debug {
        append(path, &format!("{} {line}", clock::iso_from_ms(clock::now_ms())));
    }
}

// The life of the cache Claude Code asks for: 5 minutes on an API key, an hour on a subscription.
pub fn cache_ttl() -> Duration {
    if AUTH.as_ref().is_some_and(|auth| auth["authMethod"] == "api_key") { API_KEY_CACHE_TTL } else { CACHE_TTL }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// --- Models ---

// Input: a model id. Output: the id Claude Code is asked for and the window it serves. 1M needs the
// [1m] suffix, given only where measured; opus-4-6 serves it on Max or with extra usage, sonnet-4-6
// with extra usage only.
fn runtime_model(id: &str) -> (String, u64) {
    let settings = &*SETTINGS;
    let gated = match id {
        "claude-opus-4-6" => settings.extra_usage || AUTH.as_ref().is_some_and(|auth| auth["subscriptionType"] == "max"),
        "claude-sonnet-4-6" => settings.extra_usage,
        _ => false,
    };
    if !settings.force_200k.iter().any(|forced| forced == id) && (settings.one_m.iter().any(|one_m| one_m == id) || gated) {
        (format!("{id}[1m]"), ONE_M)
    } else {
        (id.to_string(), TWO_HUNDRED_K)
    }
}

// The window pi compacts against: where Claude Code starts refusing, or what its own overrides
// (CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE; CLAUDE_CODE_MAX_CONTEXT_TOKENS with DISABLE_COMPACT) make it.
fn effective_window(window: u64, max_tokens: u64) -> u64 {
    let positive = |name: &str| env(name).and_then(|value| value.parse::<u64>().ok()).filter(|value| *value > 0);
    if let Some(limit) = positive("CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE") {
        return limit;
    }
    let served = if flag("DISABLE_COMPACT") { positive("CLAUDE_CODE_MAX_CONTEXT_TOKENS").unwrap_or(window) } else { window };
    served.saturating_sub(max_tokens.min(OUTPUT_RESERVE_CAP) + BLOCKING_MARGIN)
}

// How long Claude Code may take to answer an initialize request (it took 0.5 s, 2026-11-03).
const OFFERED_WAIT: Duration = Duration::from_secs(30);

// The models Claude Code offers this account: its answer to an initialize request sent alone, which
// lists them (Claude Code 2.1.317, 2026-11-03; against a stand-in API it sent nothing but a
// connectivity check, so no prompt and no plan use). It starts as a plain `claude` does, without
// CHILD_ENV: with non-essential traffic or GrowthBook off it lists only its five built-in entries,
// not the account's (13 on a Max plan that day). Output: its entries as it gives them (value,
// resolvedModel, displayName, supportedEffortLevels, supportsThinkingOff, ...), or why there are none.
pub async fn offered_models() -> Result<Vec<Value>, String> {
    let answer = ask_alone("initialize").await?;
    answer["response"]["response"]["models"].as_array().cloned().ok_or_else(|| format!("Claude Code's initialize answer lists no models: {}", answer["response"]["error"]))
}

// A control request sent alone to a Claude Code started for it, which asks no model. Input: its
// subtype. Output: Claude Code's control_response.
async fn ask_alone(subtype: &str) -> Result<Value, String> {
    let mut command = tokio::process::Command::new(&SETTINGS.claude);
    command.args(base_args(false, None)).args(["--tools=", "--setting-sources=", "--strict-mcp-config", "--no-session-persistence"]);
    command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true);
    for name in PARENT_ENV {
        command.env_remove(name);
    }
    if let Some(dir) = &SETTINGS.config_dir {
        command.env("CLAUDE_CONFIG_DIR", dir);
    }
    // Killed and waited for here however the answer went.
    let mut child = start(command)?;
    let id = request_id();
    child.send(&json!({"type": "control_request", "request_id": id, "request": {"subtype": subtype}})).await;
    let answer = timeout(OFFERED_WAIT, async {
        while let Ok(Some(line)) = child.lines.next_line().await {
            let message: Value = serde_json::from_str(&line).unwrap_or_default();
            if message["type"] == "control_response" && message["response"]["request_id"] == id.as_str() {
                return Some(message);
            }
        }
        None
    }).await;
    let _ = child.process.kill().await;
    answer.map_err(|_| format!("Claude Code didn't answer {subtype}"))?.ok_or_else(|| format!("Claude Code ended before answering {subtype}"))
}

// Ask Claude Code how much of the plan is used (get_usage, no model request).
pub async fn ask_usage() -> Result<(), String> {
    note_usage(&ask_alone("get_usage").await?["response"]["response"]);
    Ok(())
}

// Note get_usage's answer for the model scale. Its rate_limits names each window (five_hour,
// seven_day, seven_day_opus, ...; null when the plan doesn't have it) with its utilization in
// percent and resets_at in ISO 8601; extra_usage says whether usage past the plan is bought. Only
// the five_hour and seven_day windows are kept: the rest are names Claude Code doesn't explain
// (one, priced in dollars, read 100% on a plan that worked, 2026-11-03).
fn note_usage(answer: &Value) {
    let Some(limits) = answer["rate_limits"].as_object() else { return };
    let mut windows = Map::new();
    for (name, window) in limits.iter().filter(|(name, _)| name.as_str() == "five_hour" || name.starts_with("seven_day")) {
        let Some(used) = window["utilization"].as_f64() else { continue };
        let resets_at = window["resets_at"].as_str().and_then(|time| crate::clock::ms_from_iso(&time.replace("+00:00", "Z"))).map(|ms| ms / 1000);
        windows.insert(name.clone(), json!({"usedPercent": used, "resetsAt": resets_at}));
    }
    let extra = &answer["rate_limits"]["extra_usage"];
    if extra["is_enabled"] == true {
        let cents = 10f64.powi(extra["decimal_places"].as_i64().unwrap_or(2) as i32);
        let money = |field: &str| extra[field].as_f64().map(|amount| format!("{:.2}", amount / cents));
        let limit = money("monthly_limit").map_or(String::new(), |limit| format!(" of {limit}"));
        let note = format!("extra usage {}{limit} {} this month", money("used_credits").unwrap_or("0".into()), extra["currency"].as_str().unwrap_or_default());
        windows.insert("extra".into(), json!({"note": note}));
    }
    crate::scale::replace("claude", windows);
}

// Input: the models Claude Code offers (model.rs). Output: the bridge's models,
// the fields that matter here and nothing that would make piccosta treat them as an Anthropic API
// (no compat, no prompt cache warming); no prices, so a turn billed by the token costs what Claude
// Code says, or what models.json's modelOverrides price it at (price).
pub fn bridge_models(anthropic: Vec<Map<String, Value>>) -> Vec<Map<String, Value>> {
    anthropic.into_iter().filter_map(|model| {
        let id = model.get("id")?.as_str()?.to_string();
        let (_, window) = runtime_model(&id);
        let max_tokens = model.get("maxTokens").and_then(Value::as_u64).unwrap_or(OUTPUT_RESERVE_CAP);
        let name = model.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
        let says_1m = name.split(|c: char| !c.is_alphanumeric()).any(|word| word.eq_ignore_ascii_case("1m"));
        let name = if window > TWO_HUNDRED_K && !says_1m { format!("{name} 1M") } else { name };
        let mut out = json!({"id": id, "name": name, "api": PROVIDER, "provider": PROVIDER, "baseUrl": PROVIDER,
            "reasoning": model.get("reasoning").cloned().unwrap_or(json!(false)), "input": model.get("input").cloned().unwrap_or(json!(["text"])),
            "contextWindow": effective_window(window, max_tokens), "maxTokens": max_tokens});
        if let Some(map) = model.get("thinkingLevelMap") {
            out["thinkingLevelMap"] = map.clone();
        }
        if SETTINGS.no_thinking_off.contains(&id) {
            out["thinkingLevelMap"]["off"] = Value::Null;
        }
        out.as_object().cloned()
    }).collect()
}

// pi's thinking level as Claude Code's effort: the model's own map first (a null entry means no
// effort is sent and Claude Code's default applies), else the generic table. None for "off".
fn effort(model: &Model, level: &str) -> Option<String> {
    match model.thinking_level_map.as_ref().and_then(|map| map.get(level)) {
        _ if level == "off" => None,
        None => Some(match level {
            "minimal" | "low" => "low",
            "medium" => "medium",
            "high" => "high",
            "xhigh" | "max" => "max",
            _ => return None,
        }.to_string()),
        Some(mapped) => mapped.clone().filter(|effort| EFFORTS.contains(&effort.as_str())),
    }
}

// --- The child process ---

struct Child {
    process: tokio::process::Child,
    pid: u32,
    // Its start time, so its group is signalled only while it's still its own (proc::group_held).
    start: Option<u64>,
    stdin: Option<ChildStdin>,
    lines: Lines<BufReader<ChildStdout>>,
    // The last STDERR_TAIL bytes of its stderr, and the task keeping them, stopped by stderr_tail or
    // when the child is dropped, so a descendant holding stderr open can't keep it alive.
    stderr: Arc<Mutex<Vec<u8>>>,
    stderr_reader: Option<tokio::task::JoinHandle<()>>,
}

fn spawn(args: &[String], cwd: &str) -> Result<Child, String> {
    version();
    let mut command = tokio::process::Command::from(claude_command());
    command.args(args).current_dir(cwd).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    for name in PARENT_ENV {
        command.env_remove(name);
    }
    command.envs(CHILD_ENV);
    debug(&format!("spawn: {} {}", SETTINGS.claude, args.join(" ")));
    start(command)
}

// Start a claude. Dropping its Child kills it. If piccosta itself dies, SIGTERM lets Claude
// Code flush its buffered native journal, which the pre-query checkpoint points at for recovery.
// Its own group is also watched by the reaper, which escalates if SIGTERM can't reach a wedged
// child. Normal shutdown gives every query its bounded close before the spawning thread ends.
fn start(mut command: tokio::process::Command) -> Result<Child, String> {
    crate::proc::group(&mut command);
    crate::proc::parent_death(&mut command);
    let mut process = command.spawn().map_err(|error| format!("Claude Code could not be started ({}): {error}", SETTINGS.claude))?;
    let pid = {
        let pid = process.id().ok_or("Claude Code exited before assignment")?;
        crate::proc::attach(pid).map_err(|error| format!("Claude Code job: {error}"))?;
        crate::tools::reaper::track(pid, true);
        pid
    };
    let stdout = process.stdout.take().ok_or("Claude Code started without a stdout")?;
    let stderr = Arc::new(Mutex::new(Vec::new()));
    let stderr_reader = process.stderr.take().map(|mut pipe| {
        let tail = stderr.clone();
        tokio::spawn(async move {
            let mut chunk = [0u8; STDERR_TAIL];
            while let Ok(read) = pipe.read(&mut chunk).await && read > 0 {
                let mut tail = lock(&tail);
                tail.extend_from_slice(&chunk[..read]);
                let excess = tail.len().saturating_sub(STDERR_TAIL);
                tail.drain(..excess);
            }
        })
    });
    Ok(Child { stdin: process.stdin.take(), process, lines: BufReader::new(stdout).lines(), stderr, stderr_reader, pid, start: crate::proc::start(pid as i32) })
}

// However a child ends, its stderr reader goes when it's dropped (kill_on_drop stops one still
// running then).
impl Drop for Child {
    fn drop(&mut self) {
        if crate::proc::group_held(self.pid, self.start) {
            crate::proc::kill_group(self.pid, crate::proc::KILL);
        }
        crate::proc::release(self.pid);
        crate::tools::reaper::track(self.pid, false);
        if let Some(reader) = &self.stderr_reader {
            reader.abort();
        }
    }
}

impl Child {
    // One protocol message, one line. A child that stopped reading just loses the rest.
    async fn send(&mut self, message: &Value) {
        if let Some(stdin) = &mut self.stdin
            && stdin.write_all(format!("{message}\n").as_bytes()).await.is_err()
        {
            self.stdin = None;
        }
    }

    // End its input, then SIGTERM and SIGKILL if it doesn't go. Output: how it exited.
    async fn close(&mut self) -> Option<ExitStatus> {
        self.stdin = None;
        let pid = self.process.id();
        let mut status = timeout(CLOSE_GRACE, self.process.wait()).await.ok().and_then(Result::ok);
        if let (None, Some(pid)) = (status, pid) {
            if let Some(process) = crate::proc::Process::open(pid as i32) { process.signal(crate::proc::TERM); }
            status = timeout(TERM_GRACE, self.process.wait()).await.ok().and_then(Result::ok);
        }
        if status.is_none() {
            let _ = self.process.kill().await;
        }
        status
    }

    // What it last wrote to stderr, once it has exited, for a failure message and the debug log:
    // empty when it wrote nothing. Something it started that keeps stderr open past CLOSE_GRACE gets
    // its reader stopped, and what was read by then is the tail.
    async fn stderr_tail(&mut self) -> String {
        if let Some(reader) = &mut self.stderr_reader {
            let _ = timeout(CLOSE_GRACE, &mut *reader).await;
            reader.abort();
            self.stderr_reader = None;
        }
        let tail = self.stderr_so_far();
        if !tail.is_empty() {
            debug(&format!("Claude Code's stderr: {tail}"));
        }
        tail
    }

    // What it has written to stderr so far, without waiting for more.
    fn stderr_so_far(&self) -> String {
        String::from_utf8_lossy(&lock(&self.stderr)).trim().to_string()
    }
}

// A failure with the child's stderr after it, when it wrote any.
fn with_stderr(failure: String, tail: &str) -> String {
    if tail.is_empty() { failure } else { format!("{failure}\nClaude Code's stderr: {tail}") }
}

// How many query tasks haven't finished, live or let go of and still closing their child, for
// shutdown to wait on, and how often it looks.
static QUERIES: AtomicUsize = AtomicUsize::new(0);
const SHUTDOWN_POLL: Duration = Duration::from_millis(10);

// piccosta is exiting (a quit, /restart's exec, a signal), where nothing would be dropped: stop
// every live query and wait for every query task, these and the ones let go of earlier, to close
// its child (end of input, SIGTERM, SIGKILL) and delete its session files. A child still there
// after that (Claude Code's model listing, or one that didn't close in time) dies with this
// piccosta (start), or at /restart's exec, where its input closes.
pub async fn shutdown() {
    for (_, live) in lock(&LIVE).drain() {
        let_go(live, "piccosta is exiting");
    }
    let deadline = Instant::now() + SHUTDOWN_WAIT;
    while QUERIES.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        tokio::time::sleep(SHUTDOWN_POLL).await;
    }
}

fn request_id() -> String {
    Uuid::new_v4().simple().to_string()[..13].to_string()
}

fn base_args(thinking_off: bool, effort: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = ["--output-format", "stream-json", "--verbose", "--input-format", "stream-json"].map(String::from).to_vec();
    if thinking_off {
        args.push("--thinking=disabled".into());
    }
    if let Some(effort) = effort {
        args.push(format!("--effort={effort}"));
    }
    args
}

// The tools as an MCP tools/list result lists them, pi's schemas served verbatim.
fn manifest(tools: &[ToolDefinition]) -> Value {
    tools.iter().map(|tool| json!({"name": tool.name, "description": tool.description, "inputSchema": tool.parameters,
        "_meta": {"anthropic/maxResultSizeChars": MCP_RESULT_MAX_CHARS}})).collect()
}

fn initialize_result() -> Value {
    json!({"protocolVersion": "2025-11-25", "capabilities": {"tools": {"listChanged": true}}, "serverInfo": {"name": MCP_SERVER, "version": "1.0.0"}})
}

// The control request a session opens with: pi's prompt in place of Claude Code's, and our MCP
// server with its handshake already answered, so Claude Code needs no round trip for it.
fn initialize(system_prompt: &str, tools: &[ToolDefinition]) -> Value {
    let mut request = json!({"subtype": "initialize"});
    if !tools.is_empty() {
        let mut manifests = Map::new();
        manifests.insert(MCP_SERVER.into(), json!({"initializeResult": initialize_result(), "toolsListResult": {"tools": manifest(tools)}}));
        request["sdkMcpServers"] = json!([MCP_SERVER]);
        request["sdkMcpServerManifests"] = Value::Object(manifests);
    }
    request["systemPrompt"] = json!([system_prompt]);
    json!({"request_id": request_id(), "type": "control_request", "request": request})
}

// --- Usage ---

// One API request's counters, as its frames (message_start, message_delta, a whole message) have
// built them up: each frame reports running totals, so a counter it carries replaces the last.
#[derive(Default, Clone)]
struct Counters {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    cache_write_1h: u64,
    reasoning: Option<u64>,
}

impl Counters {
    fn merge(&mut self, usage: &Value) {
        let set = |target: &mut u64, value: &Value| {
            if let Some(count) = value.as_u64() {
                *target = count;
            }
        };
        set(&mut self.input, &usage["input_tokens"]);
        set(&mut self.output, &usage["output_tokens"]);
        set(&mut self.cache_read, &usage["cache_read_input_tokens"]);
        set(&mut self.cache_write, &usage["cache_creation_input_tokens"]);
        set(&mut self.cache_write_1h, &usage["cache_creation"]["ephemeral_1h_input_tokens"]);
        if let Some(thinking) = usage["output_tokens_details"]["thinking_tokens"].as_u64() {
            self.reasoning = Some(thinking);
        }
    }
}

// A message's usage is the sum of the requests behind it: Claude Code can make more than one for
// one turn (a stalled stream and its retry, the re-ask after an unserved tool), all billed.
fn usage_of<'a>(requests: impl Iterator<Item = &'a Counters>) -> Usage {
    let mut sum = Counters::default();
    for request in requests {
        sum.input += request.input;
        sum.output += request.output;
        sum.cache_read += request.cache_read;
        sum.cache_write += request.cache_write;
        sum.cache_write_1h += request.cache_write_1h;
        sum.reasoning = request.reasoning.map(|reasoning| sum.reasoning.unwrap_or(0) + reasoning).or(sum.reasoning);
    }
    Usage { input: sum.input, output: sum.output, cache_read: sum.cache_read, cache_write: sum.cache_write, cache_write_1h: Some(sum.cache_write_1h),
        reasoning: sum.reasoning, total_tokens: sum.input + sum.output + sum.cache_read + sum.cache_write, cost: Default::default() }
}

// The response a message came from, as pi's own Anthropic provider names it: responseModel only
// when the served model isn't the one asked for.
fn record_response(output: &mut AssistantMessage, id: &Value, served: &Value, model: &Model) {
    if let Some(id) = id.as_str() {
        output.response_id = Some(id.to_string());
    }
    if let Some(served) = served.as_str() {
        output.response_model = (served != model.id).then(|| served.to_string());
    }
}

// --- Failures ---

// Input: a result message. Output: its failure text, None when it succeeded. API failures come as
// is_error on a success-shaped result, the error subtypes carry `errors`.
fn result_error(message: &Value) -> Option<String> {
    if message["subtype"] == "success" {
        let text = message["result"].as_str().filter(|text| !text.is_empty()).unwrap_or("Claude Code reported an error");
        let bare = text.strip_prefix("API Error: ").unwrap_or(text);
        let connection = CONNECTION_FAILURES.iter().any(|phrase| bare.starts_with(phrase));
        return (message["is_error"] == true).then(|| if connection { format!("Connection error: {text}") } else { text.to_string() });
    }
    if let Some(errors) = message["errors"].as_array().filter(|errors| !errors.is_empty()) {
        return Some(errors.iter().map(|error| error.as_str().map_or_else(|| error.to_string(), String::from)).collect::<Vec<_>>().join("\n"));
    }
    Some(message["error"].as_str().map_or_else(|| format!("Claude Code failed: {}", message["subtype"].as_str().unwrap_or("unknown result")), String::from))
}

// A failure that a rate-limit rejection preceded, named as one. pi has no typed rate-limit error,
// so fallbacks match the text; "quota exceeded" is on pi's non-retryable list and wins over its
// retryable "rate limit", so a limit that resets in hours goes to fallback instead of retrying.
fn describe_rate_limit(info: &Value, failure: &str) -> String {
    let kind = info["rateLimitType"].as_str().map(|kind| format!(" ({kind})")).unwrap_or_default();
    let resets = info["resetsAt"].as_f64().map(|seconds| format!(" \u{2014} resets {}", clock::iso_from_ms((seconds * 1000.0) as i64))).unwrap_or_default();
    format!("Claude quota exceeded \u{2014} rate limit{kind}{resets}: {failure}")
}

fn failed(model: &Model, reason: StopReason, message: &str) -> AssistantMessage {
    let mut output = new_output(model, PROVIDER);
    output.stop_reason = reason;
    output.error_message = Some(message.to_string());
    output
}

// Claude Code's argument names where they differ from pi's (pass-through otherwise; the first
// alias wins).
fn pi_arguments(tool: &str, arguments: Map<String, Value>) -> Map<String, Value> {
    let rename = |key: &str| match (tool.to_lowercase().as_str(), key) {
        ("read" | "write" | "edit", "file_path") => "path",
        ("edit", "old_string" | "old_text") => "oldText",
        ("edit", "new_string" | "new_text") => "newText",
        _ => key,
    }.to_string();
    let mut out = Map::new();
    for (key, value) in arguments {
        out.entry(rename(&key)).or_insert(value);
    }
    out
}

// --- Live queries ---

// What a call shares with its query's task: what Claude Code holds (its shape, one print per message
// of pi's it holds, and the tool calls it is parked on: None while a turn streams or after the last
// one), and the visible messages' prints and roles that a clean end keeps (with the final answer
// added).
struct State {
    shape: Shape,
    prints: Vec<String>,
    pending: Option<Vec<String>>,
    kept_base: (Vec<String>, Vec<String>),
}

struct Live {
    id: u64,
    state: Arc<Mutex<State>>,
    commands: mpsc::UnboundedSender<Command>,
}

enum Command {
    // The call's tool results (id, MCP result) and its stream.
    Deliver(Vec<(String, Value)>, Option<Vec<Value>>, Turn),
    Discard(String),
    EndRun(Option<Duration>, oneshot::Sender<()>),
}

// One provider call's stream: where its events go, the message it ends with, and its abort.
struct Turn {
    events: mpsc::UnboundedSender<StreamEvent>,
    reply: oneshot::Sender<AssistantMessage>,
    cancel: CancellationToken,
}

static LIVE: LazyLock<Mutex<HashMap<String, Live>>> = LazyLock::new(Default::default);
static NEXT_QUERY: AtomicU64 = AtomicU64::new(1);
// Why each session's latest call started a new Claude Code query, for the cache monitor.
static REWRITES: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Default::default);

// The reason the session's latest call started a new query, if it did.
pub fn take_rewrite(session_id: &str) -> Option<String> {
    lock(&REWRITES).remove(session_id)
}

fn kept_base(messages: &[Message]) -> (Vec<String>, Vec<String>) {
    messages.iter().filter(history::visible).map(|message| (fingerprint(message), match message {
        Message::User(_) => "user",
        Message::Assistant(_) => "assistant",
        _ => "toolResult",
    }.to_string())).unzip()
}

// Let go of a query taken out of LIVE: its task is told why and closes its child on its own time.
// Nothing waits for it here, so a session closing or switching lets go of it at once.
fn let_go(live: Live, reason: &str) {
    let _ = live.commands.send(Command::Discard(reason.to_string()));
}

fn discard(key: &str, reason: &str) {
    if let Some(live) = lock(&LIVE).remove(key) {
        let_go(live, reason);
    }
}

// A clean answer may wait for the next run. Errors and unfinished tool calls close; managers
// already carry their role in the session, and hold for the lifetime of their runtime.
pub async fn end_run(session: &crate::session::Session, keep: bool) {
    let commands = lock(&LIVE).get(session.id()).map(|live| live.commands.clone());
    if let Some(commands) = commands {
        let hold = if !keep { Some(Duration::ZERO) } else if crate::manager::role(session).is_some() { None } else { Some(SETTINGS.hold) };
        let (done, ended) = oneshot::channel();
        let _ = commands.send(Command::EndRun(hold, done));
        let _ = ended.await;
    }
}

pub fn close(session_id: &str) {
    discard(session_id, "the session was closed");
    history::detach(session_id);
}

// --- One provider call ---

// Input: the request and where its events go. Output: the turn's message. A cacheRetention "none"
// request (pi's compaction and branch summaries, under a fresh session id) starts a query of its
// own that ends with its turn and keeps nothing; every other one delivers to the session's query or
// rewrites.
pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let model = request.model;
    let system = transcript::current_system_message(request.messages);
    let system_prompt = system.as_ref().map(transcript::system_message_text).unwrap_or_default();
    let tools = system.and_then(|system| system.tools_added).unwrap_or_default();
    let messages = fold_into_results(request.messages.iter().filter(|message| !matches!(message, Message::System(_))).cloned());
    let (cli_model, _) = runtime_model(&model.id);
    let effort = effort(model, request.thinking_level);
    let thinking_off = request.thinking_level == "off";
    let cwd = request.cwd.to_string_lossy().into_owned();
    let (key, one_shot) = (request.session_id.unwrap_or("(none)").to_string(), !request.cache);
    if !one_shot {
        history::attach(&key, request.session_file, &messages);
        lock(&REWRITES).remove(&key);
    }
    let shape = history::shape(&system_prompt, &tools, &cli_model, if thinking_off { Some("off") } else { effort.as_deref() }, &cwd);
    let prints: Vec<String> = messages.iter().map(fingerprint).collect();
    if request.cancel.is_cancelled() {
        discard(&key, "the run was aborted");
        return failed(model, StopReason::Aborted, "Operation aborted");
    }
    let mut forced: Option<String> = one_shot.then(|| "a one-off summary".into());
    loop {
        let (reply, answer) = oneshot::channel();
        let turn = Turn { events: events.clone(), reply, cancel: request.cancel.clone() };
        let (reason, turn) = match deliver(&key, &messages, &prints, &shape, forced.take(), turn) {
            Ok(()) => match answer.await {
                Ok(message) => return message,
                // The query ended before it took them: rewrite like any other mismatch.
                Err(_) => {
                    forced = Some("the parked query ended before it took the results".into());
                    continue;
                }
            },
            Err(rewrite) => rewrite,
        };
        debug(&format!("sync: rewrite, {reason} (session {key})"));
        if !one_shot {
            lock(&REWRITES).insert(key.clone(), reason.clone());
        }
        let start = Start { key: &key, model, cli_model: &cli_model, effort: effort.as_deref(), thinking_off, system_prompt: &system_prompt,
            tools: &tools, messages: &messages, prints: &prints, shape: &shape, cwd: &cwd, one_shot };
        if let Err(error) = start_query(start, turn) {
            return failed(model, StopReason::Error, &error);
        }
        let mut answer = answer.await.unwrap_or_else(|_| failed(model, StopReason::Error, "Claude Code ended the query without answering"));
        // A summary with no text in it failed: a compaction would keep an empty checkpoint.
        if one_shot && answer.stop_reason == StopReason::Stop && !answer.content.iter().any(|block| matches!(block, Content::Text { text, .. } if !text.trim().is_empty())) {
            (answer.stop_reason, answer.error_message) = (StopReason::Error, Some("Claude Code summary returned empty text".into()));
        }
        return answer;
    }
}

// Messages that arrive while tools run (the user's, a background job's notice) come after the
// tool results in pi's context. Claude Code takes only tool results into a query that is waiting
// on them, so such a message would end the query and make Claude Code resume the turn from its
// session file instead: that rewrites the whole conversation into the prompt cache, and the
// resumed turn is often answered with nothing ("No response requested.", 2026-11-01: eleven
// such turns, 285K to 960K tokens written each). Here each such message goes on the end of the
// last result instead, under a line saying it arrived meanwhile, the way Claude Code itself hands
// the model a message typed while a tool runs. The same fold applies to every call, so a later
// rewrite sends the same bytes the live query did and the cache holds.
const ARRIVED_NOTE: &str = "[Sent while this tool ran:]";

fn fold_into_results(messages: impl Iterator<Item = Message>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    for message in messages {
        match (message, out.last_mut()) {
            (Message::User(user), Some(Message::ToolResult(result))) => {
                result.content.push(Content::text(ARRIVED_NOTE));
                result.content.extend(user.content.into_blocks().into_iter().filter(|block| matches!(block, Content::Text { .. } | Content::Image { .. })));
            }
            (message, _) => out.push(message),
        }
    }
    out
}

// Hand tool results or a new user turn to the held query when the sync check allows it. Output:
// Ok once sent (the next turn streams into `turn`), else the rewrite reason and the turn back; the
// live query, if any, is discarded then.
fn deliver(key: &str, messages: &[Message], prints: &[String], shape: &Shape, forced: Option<String>, turn: Turn) -> Result<(), (String, Turn)> {
    let mut live = lock(&LIVE);
    let decision = match forced {
        Some(reason) => Err(reason),
        None => history::decide(live.get(key).map(|live| lock(&live.state)).as_deref(), messages, prints, shape),
    };
    let start = match decision {
        Ok(start) => start,
        Err(reason) => {
            if let Some(old) = live.remove(key) {
                let_go(old, &format!("pi's context changed: {reason}"));
            }
            return Err((reason, turn));
        }
    };
    let Some(entry) = live.get(key) else { return Err(("no live query for this session".into(), turn)) };
    let mut state = lock(&entry.state);
    let results = messages[start..].iter().filter_map(|message| match message {
        Message::ToolResult(result) => {
            let mut content: Vec<Value> = result.content.iter().filter_map(|block| match block {
                Content::Text { text, .. } if !text.is_empty() => Some(json!({"type": "text", "text": text})),
                Content::Image { data, mime_type } if !data.is_empty() && !mime_type.is_empty() => Some(json!({"type": "image", "data": data, "mimeType": mime_type})),
                _ => None,
            }).collect();
            if content.is_empty() {
                content.push(json!({"type": "text", "text": if result.is_error { history::EMPTY_ERROR_RESULT } else { "" }}));
            }
            Some((result.tool_call_id.clone(), json!({"content": content, "isError": result.is_error})))
        }
        _ => None,
    }).collect();
    let prompt = state.pending.is_none().then(|| messages[start..].iter().flat_map(|message| match message {
        Message::User(user) => history::anthropic_blocks(&user.content.clone().into_blocks()),
        _ => Vec::new(),
    }).collect());
    debug(&format!("sync: deliver to the held query (session {key}, {} messages)", prints.len()));
    (state.prints, state.pending, state.kept_base) = (prints.to_vec(), None, kept_base(messages));
    drop(state);
    let Err(mpsc::error::SendError(Command::Deliver(_, _, turn))) = entry.commands.send(Command::Deliver(results, prompt, turn)) else { return Ok(()) };
    live.remove(key);
    Err(("the query is gone".into(), turn))
}

struct Start<'a> {
    key: &'a str,
    model: &'a Model,
    cli_model: &'a str,
    effort: Option<&'a str>,
    thinking_off: bool,
    system_prompt: &'a str,
    tools: &'a [ToolDefinition],
    messages: &'a [Message],
    prints: &'a [String],
    shape: &'a Shape,
    cwd: &'a str,
    // A summary keeps no records and takes no tools.
    one_shot: bool,
}

// Start a query holding exactly pi's context: the history in a fresh session file it resumes, the
// trailing user turn as its opening prompt. A history ending at tool results or an interrupt
// gets an invisible terminal boundary (history::write_session), so initialize always goes first.
fn start_query(start: Start, turn: Turn) -> Result<(), String> {
    let visible: Vec<Message> = start.messages.iter().filter(history::visible).cloned().collect();
    let (history_end, prompt, continued) = history::split(start.key, &visible);
    let project_path = history::real_or(start.cwd);
    let written = if history_end > 0 {
        let history = &visible[..history_end];
        let prints: Vec<String> = history.iter().map(fingerprint).collect();
        let id = history::write_session(start.key, history, &prints, start.cwd, &project_path, start.tools, start.cli_model, start.shape)
            .map_err(|error| format!("Could not write Claude Code's session file: {error}"))?;
        Some(id)
    } else {
        None
    };
    if continued {
        debug("start: context ends at an assistant message, opening with the continue note");
    }
    let mut args = base_args(start.thinking_off, start.effort);
    let cc_session = written.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
    args.push(if written.is_some() { format!("--resume={cc_session}") } else { format!("--session-id={cc_session}") });
    let base = kept_base(&visible);
    if !start.one_shot {
        history::checkpoint(start.key, &cc_session, &project_path, start.shape, &base.0, &base.1)
            .map_err(|error| format!("Could not checkpoint Claude Code's active session: {error}"))?;
    }
    args.extend(["--tools=", "--setting-sources=", "--permission-mode=bypassPermissions", "--include-partial-messages", "--model", start.cli_model,
        "--disable-slash-commands"].map(String::from));
    if SETTINGS.strict_mcp || start.one_shot {
        args.push("--strict-mcp-config".into());
    }
    // Some models default to omitted thinking text; summarized makes thinking_delta events arrive.
    if start.effort.is_some() {
        args.extend(["--thinking-display", "summarized"].map(String::from));
    }
    args.extend(["--settings", PROVIDER_SETTINGS].map(String::from));
    let child = spawn(&args, start.cwd).inspect_err(|_| {
        if let Some(id) = &written {
            history::delete_session(id, &project_path);
        }
    })?;
    let opening = vec![initialize(start.system_prompt, start.tools),
        json!({"type": "user", "message": {"role": "user", "content": prompt}, "parent_tool_use_id": null})];
    let state = Arc::new(Mutex::new(State { shape: start.shape.clone(), prints: start.prints.to_vec(), pending: None, kept_base: base }));
    let (commands, inbox) = mpsc::unbounded_channel();
    let id = NEXT_QUERY.fetch_add(1, Ordering::Relaxed);
    let mut query = Query::new(id, start.key, start.model.clone(), start.tools, child, inbox, state.clone(), project_path, start.cwd, start.shape.clone(), written);
    query.cc_session = Some(cc_session);
    query.one_shot = start.one_shot;
    query.turn = Some(turn);
    lock(&LIVE).insert(start.key.to_string(), Live { id, state, commands });
    QUERIES.fetch_add(1, Ordering::SeqCst);
    tokio::spawn(query.run(opening));
    Ok(())
}

// --- A running query ---

struct Parked {
    request_id: Value,
    rpc_id: Value,
    tool: String,
    check_at: Instant,
}

enum Wake {
    Line(Option<String>),
    Command(Option<Command>),
    Idle,
    Expire,
    Settle,
    Lost,
    Abort,
}

// Everything a query's task holds. Per turn: the message being built (`out`) with, per content
// block, the stream index it came under (None once it stopped) and its tool-call JSON so far.
struct Query {
    id: u64,
    key: String,
    model: Model,
    tools: Vec<String>,
    manifest: Value,
    child: Child,
    inbox: mpsc::UnboundedReceiver<Command>,
    state: Arc<Mutex<State>>,
    project_path: String,
    cwd: String,
    shape: Shape,
    written: Option<String>,
    cc_session: Option<String>,
    turn: Option<Turn>,
    out: AssistantMessage,
    open: Vec<Option<u64>>,
    partial: Vec<String>,
    started: bool,
    saw_stream_event: bool,
    saw_tool_call: bool,
    // The last message_start's id, whether its message_stop came, whether its message_delta did
    // (a stream Claude Code gave up on and closed itself has none), and where its blocks start.
    stream_id: Option<String>,
    stream_open: bool,
    stream_delta: bool,
    stream_start: usize,
    // The id of a non-streamed answer arriving one assistant message per block.
    fallback_id: Option<String>,
    // Every request of the query, and this turn's by response id.
    requests: Vec<Counters>,
    turn_requests: HashMap<String, usize>,
    // The tool calls of the current response (kept across a delivery), and every one pi was given.
    tool_ids: Vec<String>,
    handed: HashSet<String>,
    // tools/call requests waiting for pi's results, and results that came before their request.
    parked: HashMap<String, Parked>,
    queued: HashMap<String, Value>,
    settle_at: Option<Instant>,
    last_activity: Instant,
    hold_until: Option<Instant>,
    rejection: Option<Value>,
    failure: Option<String>,
    result_seen: bool,
    closed: bool,
    one_shot: bool,
    // Ignore the stopped stream while Claude Code flushes its records and ends.
    interrupting: bool,
    ended: Option<oneshot::Sender<()>>,
    // Claude Code's running total of what this query cost (total_cost_usd, in dollars) at its last
    // result, and at this turn's, if it had one.
    cost_seen: f64,
    result_cost: Option<f64>,
    // This query's tool turns handed to pi with their cost pending since its last priced turn.
    pending_turns: u32,
}

impl Query {
    #[allow(clippy::too_many_arguments)]
    fn new(id: u64, key: &str, model: Model, tools: &[ToolDefinition], child: Child, inbox: mpsc::UnboundedReceiver<Command>, state: Arc<Mutex<State>>,
        project_path: String, cwd: &str, shape: Shape, written: Option<String>) -> Query {
        let out = new_output(&model, PROVIDER);
        Query { id, key: key.to_string(), out, model, tools: tools.iter().map(|tool| tool.name.clone()).collect(), manifest: manifest(tools), child, inbox, state,
            project_path, cwd: cwd.to_string(), shape, written, cc_session: None, turn: None, open: Vec::new(), partial: Vec::new(), started: false,
            saw_stream_event: false, saw_tool_call: false, stream_id: None, stream_open: false, stream_delta: false, stream_start: 0, fallback_id: None,
            requests: Vec::new(), turn_requests: HashMap::new(), tool_ids: Vec::new(), handed: HashSet::new(), parked: HashMap::new(), queued: HashMap::new(),
            settle_at: None, last_activity: Instant::now(), hold_until: None, rejection: None, failure: None, result_seen: false, closed: false, one_shot: false, interrupting: false, ended: None, cost_seen: 0.0, result_cost: None, pending_turns: 0 }
    }

    async fn run(mut self, opening: Vec<Value>) {
        self.reset_turn();
        for message in &opening {
            self.child.send(message).await;
        }
        while !self.closed && self.failure.is_none() {
            let far = Instant::now() + LOST_CALL_CHECK;
            let streaming = self.turn.is_some();
            let cancel = self.turn.as_ref().map_or_else(CancellationToken::new, |turn| turn.cancel.clone());
            let lost = self.parked.values().map(|call| call.check_at).min();
            let wake = tokio::select! {
                line = self.child.lines.next_line() => Wake::Line(line.ok().flatten()),
                command = self.inbox.recv() => Wake::Command(command),
                () = sleep_until(self.last_activity + BODY_IDLE_TIMEOUT), if streaming => Wake::Idle,
                () = sleep_until(self.hold_until.unwrap_or(far)), if self.hold_until.is_some() => Wake::Expire,
                () = sleep_until(self.settle_at.unwrap_or(far)), if self.settle_at.is_some() => Wake::Settle,
                () = sleep_until(lost.unwrap_or(far)), if lost.is_some() => Wake::Lost,
                () = cancel.cancelled(), if streaming && !self.interrupting => Wake::Abort,
            };
            match wake {
                Wake::Line(Some(line)) => self.line(&line).await,
                Wake::Line(None) => break,
                Wake::Command(Some(Command::Deliver(results, prompt, turn))) => self.deliver(results, prompt, turn).await,
                Wake::Command(Some(Command::Discard(reason))) => self.discard(&reason).await,
                Wake::Command(Some(Command::EndRun(hold, done))) => {
                    if self.result_seen && self.turn.is_none() && hold != Some(Duration::ZERO) {
                        self.hold_until = hold.and_then(|hold| Instant::now().checked_add(hold));
                        let _ = done.send(());
                    } else {
                        self.ended = Some(done);
                        self.discard("pi's run ended with the query still live").await;
                    }
                }
                Wake::Command(None) => self.discard("the bridge let go of it").await,
                Wake::Idle => self.idle().await,
                Wake::Expire => {
                    debug(&format!("hold expired: session {}", self.key));
                    break;
                }
                Wake::Settle => self.end_turn_on_tool_use(),
                Wake::Lost => self.check_lost().await,
                Wake::Abort => self.interrupt().await,
            }
            if self.result_seen {
                if self.interrupting || self.one_shot || self.failure.is_some() {
                    break;
                }
                if self.turn.is_some() {
                    self.keep_records();
                    lock(&self.state).prints.push(fingerprint(&Message::Assistant(self.out.clone())));
                    self.reply();
                }
            }
        }
        self.forget();
        self.finish().await;
        QUERIES.fetch_sub(1, Ordering::SeqCst);
    }

    fn reset_turn(&mut self) {
        self.out = new_output(&self.model, PROVIDER);
        self.out.stop_reason = StopReason::Stop;
        (self.open, self.partial) = (Vec::new(), Vec::new());
        (self.started, self.saw_stream_event, self.saw_tool_call) = (false, false, false);
        (self.stream_id, self.stream_open, self.stream_delta, self.stream_start, self.fallback_id) = (None, false, false, 0, None);
        self.turn_requests.clear();
        self.result_seen = false;
        self.result_cost = None;
    }

    fn emit(&self, event: StreamEvent) {
        if let Some(turn) = &self.turn {
            let _ = turn.events.send(event);
        }
    }

    fn ensure_started(&mut self) {
        if !self.started && self.turn.is_some() {
            self.emit(StreamEvent::Start);
            self.started = true;
        }
    }

    fn push(&mut self, block: Content, stream_index: Option<u64>) -> usize {
        self.out.content.push(block);
        self.open.push(stream_index);
        self.partial.push(String::new());
        self.out.content.len() - 1
    }

    fn remove_block(&mut self, index: usize) {
        self.out.content.remove(index);
        self.open.remove(index);
        self.partial.remove(index);
    }

    fn tool_call_ids(&self) -> Vec<String> {
        self.out.content.iter().filter_map(|block| match block {
            Content::ToolCall { id, .. } => Some(id.clone()),
            _ => None,
        }).collect()
    }

    // End the open stream, if any, with a failure. A turn whose stream already ended stays as pi got it.
    fn end_open(&mut self, reason: StopReason, message: &str) {
        if let Some(turn) = self.turn.take() {
            self.out.stop_reason = reason;
            self.out.error_message.get_or_insert_with(|| message.to_string());
            price(&self.model, &mut self.out.usage, None);
            let _ = turn.reply.send(self.out.clone());
        }
    }

    // The one way a turn ends on tool calls: pi gets the message, and the query is parked on them.
    fn end_turn_on_tool_use(&mut self) {
        self.settle_at = None;
        let Some(turn) = self.turn.take() else { return };
        self.out.stop_reason = StopReason::ToolUse;
        self.hand_pending(turn);
        self.handed.extend(self.tool_call_ids());
        let mut state = lock(&self.state);
        state.prints.push(fingerprint(&Message::Assistant(self.out.clone())));
        state.pending = Some(self.tool_ids.clone());
        debug(&format!("sync: parked on [{}] (session {})", self.tool_ids.join(", "), self.key));
    }

    // Hand pi a turn Claude Code hasn't priced yet (it reckons cost only at a result, for everything
    // since the last): one it would price is pending until the turn that answers pays for it
    // (reply), and stays unknown if the query ends first.
    fn hand_pending(&mut self, turn: Turn) {
        price(&self.model, &mut self.out.usage, None);
        if self.out.usage.cost.state == CostState::Unknown {
            self.out.usage.cost.state = CostState::Pending;
            self.pending_turns += 1;
        }
        let _ = turn.reply.send(self.out.clone());
    }

    fn flush_settle(&mut self) {
        if self.settle_at.is_some() {
            self.end_turn_on_tool_use();
        }
    }

    // Out of LIVE, unless a later query of the session has taken its place.
    fn forget(&self) {
        let mut live = lock(&LIVE);
        if live.get(&self.key).is_some_and(|live| live.id == self.id) {
            live.remove(&self.key);
        }
    }

    // Stop the query: it holds a context pi no longer has, or something went wrong. The child is
    // interrupted and its input ended, so nothing it waits on gets answered and it starts no
    // further request; whatever waited on it is let go.
    async fn discard(&mut self, reason: &str) {
        debug(&format!("discard: session {}: {reason}", self.key));
        self.closed = true;
        self.forget();
        self.child.send(&json!({"type": "control_request", "request_id": request_id(), "request": {"subtype": "interrupt"}})).await;
        self.child.stdin = None;
        self.end_open(StopReason::Error, &format!("Claude Code query discarded: {reason}"));
        (self.parked, self.queued, self.settle_at) = (HashMap::new(), HashMap::new(), None);
    }

    // A stream gone silent mid-message ends as a retryable failure ("timed out" is on pi's list), the
    // way pi's own providers time out an idle response.
    async fn idle(&mut self) {
        let silent = self.last_activity.elapsed();
        if silent < BODY_IDLE_TIMEOUT {
            return;
        }
        // pi gets this failure now, before finish, so what Claude Code wrote to stderr by now goes with it.
        let message = with_stderr(format!("Claude Code stream timed out: no event for {} s (limit {} ms)", silent.as_secs_f64().round(), BODY_IDLE_TIMEOUT.as_millis()),
            &self.child.stderr_so_far());
        self.end_open(StopReason::Error, &message);
        self.discard(&message).await;
    }

    // A tools/call pi was never handed can never be answered: stop the query and fail the turn,
    // naming it, rather than wait forever. A call pi has is never failed, however long it runs.
    async fn check_lost(&mut self) {
        let now = Instant::now();
        let due: Vec<String> = self.parked.iter().filter(|(_, call)| call.check_at <= now).map(|(id, _)| id.clone()).collect();
        for id in due {
            let streaming = self.turn.is_some() && self.tool_call_ids().contains(&id);
            if let Some(call) = self.parked.get_mut(&id).filter(|_| streaming || self.handed.contains(&id)) {
                call.check_at = now + LOST_CALL_CHECK;
                continue;
            }
            let message = format!("Claude Code dispatched tool call {} [{id}] that pi never received; the query was stopped so the turn would not wait on it forever",
                self.parked[&id].tool);
            self.end_open(StopReason::Error, &message);
            return self.discard(&message).await;
        }
    }

    async fn answer(&mut self, request_id: Value, rpc_id: Value, result: Result<Value, String>) {
        let rpc = match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": rpc_id, "result": result}),
            Err(message) => json!({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32603, "message": message}}),
        };
        self.child.send(&json!({"type": "control_response", "response": {"subtype": "success", "request_id": request_id, "response": {"mcp_response": rpc}}})).await;
    }

    // An interrupt's records include completed blocks and Claude Code's interrupt note, which pi
    // doesn't show. Let Claude Code flush them before replying, so the next call can resume the
    // same bytes even when it arrives at once or piccosta restarts.
    async fn interrupt(&mut self) {
        debug(&format!("interrupt: saving the stopped turn for session {}", self.key));
        (self.settle_at, self.interrupting) = (None, true);
        self.out.stop_reason = StopReason::Aborted;
        self.out.error_message.get_or_insert_with(|| "Operation aborted".to_string());
        self.child.send(&json!({"type": "control_request", "request_id": request_id(), "request": {"subtype": "interrupt"}})).await;
    }

    async fn deliver(&mut self, results: Vec<(String, Value)>, prompt: Option<Vec<Value>>, turn: Turn) {
        self.hold_until = None;
        self.turn = Some(turn);
        self.reset_turn();
        self.last_activity = Instant::now();
        // A crash after a tool result must not replay the earlier part of this same query.
        if let Some(id) = &self.cc_session {
            let state = lock(&self.state);
            if let Err(error) = history::checkpoint(&self.key, id, &self.project_path, &self.shape, &state.kept_base.0, &state.kept_base.1) {
                self.failure = Some(format!("Could not checkpoint Claude Code's active session: {error}"));
                return;
            }
        }
        if let Some(prompt) = prompt {
            self.child.send(&json!({"type": "user", "message": {"role": "user", "content": prompt}, "parent_tool_use_id": null})).await;
        }
        for (id, result) in results {
            match self.parked.remove(&id) {
                Some(call) => self.answer(call.request_id, call.rpc_id, Ok(result)).await,
                // The result beat Claude Code's tools/call for it.
                None => {
                    self.queued.insert(id, result);
                }
            }
        }
    }

    async fn line(&mut self, line: &str) {
        if let Some(path) = &SETTINGS.record {
            append(path, line);
        }
        let Ok(message) = serde_json::from_str::<Value>(line) else { return };
        self.last_activity = Instant::now();
        match message["type"].as_str().unwrap_or_default() {
            "control_request" => self.control(&message).await,
            "control_response" if message["response"]["subtype"] == "error" => {
                self.failure = Some(format!("Claude Code refused the session: {}", message["response"]["error"].as_str().unwrap_or("no reason given")));
            }
            "control_response" | "control_cancel_request" | "keep_alive" | "transcript_mirror" => {}
            _ => self.message(&message),
        }
    }

    // Claude Code asking the host something. Only MCP traffic for our server is answered; the
    // handshake normally came with initialize, but it is answered here too.
    async fn control(&mut self, message: &Value) {
        let request_id = message["request_id"].clone();
        let request = &message["request"];
        if request["subtype"] != "mcp_message" {
            let error = format!("Unsupported control request subtype: {}", request["subtype"].as_str().unwrap_or("none"));
            return self.child.send(&json!({"type": "control_response", "response": {"subtype": "error", "request_id": request_id, "error": error}})).await;
        }
        let rpc = &request["message"];
        let rpc_id = rpc.get("id").cloned().unwrap_or(json!(0));
        let result = match rpc["method"].as_str().unwrap_or_default() {
            "initialize" => Ok(initialize_result()),
            "tools/list" => Ok(json!({"tools": self.manifest})),
            "tools/call" => {
                let name = rpc["params"]["name"].as_str().unwrap_or_default().to_string();
                let tool_use_id = rpc["params"]["_meta"]["claudecode/toolUseId"].as_str().map(String::from);
                match tool_use_id {
                    _ if !self.tools.contains(&name) => Err(format!("Unknown tool: {name}")),
                    // Claude Code stamps every call with its tool_use id, the only reliable pairing.
                    None => Err(format!("{name}: tools/call is missing _meta[\"claudecode/toolUseId\"], so its result can't be paired with its call")),
                    Some(id) => match self.queued.remove(&id) {
                        Some(result) => Ok(result),
                        None => {
                            self.parked.insert(id, Parked { request_id, rpc_id, tool: name, check_at: Instant::now() + LOST_CALL_CHECK });
                            return;
                        }
                    },
                }
            }
            _ => Ok(json!({})),
        };
        self.answer(request_id, rpc_id, result).await;
    }

    // pi's name for a tool Claude called, or None for a name we don't serve: a hallucinated builtin,
    // which Claude Code rejects and retries under a fresh id itself, so it must not reach pi.
    fn pi_tool_name(&self, name: &str) -> Option<String> {
        let lower = name.to_lowercase();
        self.tools.iter().find(|tool| format!("{MCP_TOOL_PREFIX}{tool}") == name)
            .or_else(|| self.tools.iter().find(|tool| format!("{MCP_TOOL_PREFIX}{tool}").to_lowercase() == lower)).cloned()
    }

    fn record_usage(&mut self, key: Option<&str>, usage: &Value) {
        let next = self.requests.len();
        let index = *self.turn_requests.entry(key.unwrap_or("(no message id)").to_string()).or_insert(next);
        if index == next {
            self.requests.push(Counters::default());
        }
        self.requests[index].merge(usage);
        self.out.usage = usage_of(self.turn_requests.values().map(|&index| &self.requests[index]));
    }

    // One message from Claude Code's stream. Lifecycle first (the end of input at the result, a
    // failure, rate limits, the session id), then content, which has nowhere to go once a turn has
    // ended on tool calls.
    fn message(&mut self, message: &Value) {
        let kind = message["type"].as_str().unwrap_or_default();
        let error = if kind == "result" { result_error(message) } else { None };
        // A non-streamed answer still settling ends when Claude Code moves on, unless it failed.
        if kind == "result" && error.is_some() {
            self.settle_at = None;
        } else if matches!(kind, "stream_event" | "user" | "result") {
            self.flush_settle();
        }
        if kind == "result" && self.interrupting {
            self.result_seen = true;
            self.add_unstreamed_usage(&message["usage"]);
            self.result_cost = message["total_cost_usd"].as_f64();
            self.child.stdin = None;
            return;
        }
        if kind == "result" {
            self.result_seen = true;
            if let Some(mut error) = error {
                if let Some(info) = self.rejection.take() {
                    error = describe_rate_limit(&info, &error);
                }
                self.failure = Some(error);
                return;
            }
        }
        if kind == "rate_limit_event" {
            let info = &message["rate_limit_info"];
            if info["status"] == "rejected" {
                self.rejection = Some(info.clone());
            }
            // How much of the plan is used, by window (utilization is a fraction), for the model scale.
            for (window, reading) in info["unifiedWindows"].as_object().into_iter().flatten() {
                if let Some(used) = reading["utilization"].as_f64() {
                    crate::scale::record("claude", window, used * 100.0, reading["resetsAt"].as_f64());
                }
            }
            if let (Some(used), Some(window)) = (info["utilization"].as_f64(), info["rateLimitType"].as_str()) {
                crate::scale::record("claude", window, used * 100.0, info["resetsAt"].as_f64());
            }
            return;
        }
        if kind == "system" && message["subtype"] == "init" {
            self.cc_session = message["session_id"].as_str().map(String::from).or(self.cc_session.take());
        }
        // What a stopped answer still sends belongs to no turn, the next message's included.
        if self.turn.is_none() || self.interrupting {
            return;
        }
        match kind {
            "stream_event" => self.stream_event(&message["event"]),
            "assistant" => self.assistant(&message["message"]),
            "result" => {
                self.add_unstreamed_usage(&message["usage"]);
                self.result_cost = message["total_cost_usd"].as_f64();
                if !self.saw_stream_event && message["subtype"] == "success" {
                    self.whole_text(message["result"].as_str().unwrap_or_default());
                }
            }
            // Claude Code re-sends a request whose response broke off, after closing the broken
            // stream itself: its blocks look finished, but it never got its message_delta.
            "system" if message["subtype"] == "api_retry" && self.saw_stream_event && !self.stream_delta && self.out.content.len() > self.stream_start => {
                self.drop_abandoned();
            }
            _ => {}
        }
    }

    // Keep the records before replying: a clean answer, a stopped stream's invisible tail, or a
    // parked tool call whose real results will be appended by the next run.
    fn keep_records(&self) {
        let (pending, (mut prints, mut roles)) = {
            let state = lock(&self.state);
            (state.pending.clone().unwrap_or_default(), state.kept_base.clone())
        };
        if let Some(cc_session) = self.cc_session.as_deref().filter(|_| !self.one_shot && (!pending.is_empty() || matches!(self.out.stop_reason, StopReason::Stop | StopReason::Length | StopReason::Aborted))) {
            if self.out.stop_reason != StopReason::Aborted {
                prints.push(fingerprint(&Message::Assistant(self.out.clone())));
                roles.push("assistant".into());
            }
            history::keep(&self.key, cc_session, &self.project_path, &self.cwd, &self.shape, prints, roles, &pending);
        }
    }

    // Hand pi the turn's message, if a turn is open.
    fn reply(&mut self) {
        if let Some(turn) = self.turn.take() {
            let reported = self.result_cost.map(|total| total - std::mem::replace(&mut self.cost_seen, total));
            price(&self.model, &mut self.out.usage, reported);
            if self.out.usage.cost.state == CostState::Known {
                self.out.usage.cost.settles = std::mem::take(&mut self.pending_turns);
            }
            if !self.started {
                let _ = turn.events.send(StreamEvent::Start);
            }
            let _ = turn.reply.send(self.out.clone());
        }
    }

    fn whole_text(&mut self, text: &str) {
        self.ensure_started();
        let index = self.push(Content::text(text), None);
        self.emit(StreamEvent::TextStart { index });
        self.emit(StreamEvent::TextDelta { index, delta: text.to_string() });
        self.emit(StreamEvent::TextEnd { index, content: text.to_string() });
    }

    // What the success result counts beyond what the stream showed (a request made out of sight) is
    // added to the turn it ends. Only a surplus: a stalled stream's first frame was billed too.
    fn add_unstreamed_usage(&mut self, total: &Value) {
        if !total.is_object() {
            return;
        }
        let seen = usage_of(self.requests.iter());
        let surplus = |name: &str, seen: u64| total[name].as_u64().unwrap_or(0).saturating_sub(seen);
        let extra = json!({"input_tokens": surplus("input_tokens", seen.input), "output_tokens": surplus("output_tokens", seen.output),
            "cache_read_input_tokens": surplus("cache_read_input_tokens", seen.cache_read), "cache_creation_input_tokens": surplus("cache_creation_input_tokens", seen.cache_write),
            "cache_creation": {"ephemeral_1h_input_tokens": total["cache_creation"]["ephemeral_1h_input_tokens"].as_u64().unwrap_or(0).saturating_sub(seen.cache_write_1h.unwrap_or(0))}});
        let any = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"].iter().any(|name| extra[*name] != 0);
        if any {
            self.record_usage(Some("(in the result, not streamed)"), &extra);
        }
    }

    // Blocks of the message that just stopped which never got their own content_block_stop (the
    // stream was cut inside them): a tool call Claude Code will never dispatch, a thinking block with
    // no signature. Per block, since Claude Code keeps the ones that did complete.
    fn drop_unfinished(&mut self) {
        let before = self.out.content.len();
        for index in (self.stream_start..before).rev() {
            if self.open[index].is_some() {
                self.remove_block(index);
            }
        }
        if self.out.content.len() != before {
            self.tool_ids = self.tool_call_ids();
            self.saw_tool_call = !self.tool_ids.is_empty();
        }
    }

    // The blocks of a stream Claude Code abandoned mid-message; its retry restarts those indices.
    fn drop_abandoned(&mut self) {
        while self.out.content.len() > self.stream_start {
            self.remove_block(self.out.content.len() - 1);
        }
        self.tool_ids.clear();
        self.saw_tool_call = !self.tool_call_ids().is_empty();
        self.stream_open = false;
    }

    fn stream_event(&mut self, event: &Value) {
        self.saw_stream_event = true;
        let stream_index = event["index"].as_u64();
        let position = self.open.iter().position(|open| open.is_some() && *open == stream_index);
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                // Still open from an earlier message_start: Claude Code gave up on that one.
                if self.stream_open {
                    self.drop_abandoned();
                }
                self.tool_ids.clear();
                self.stream_id = event["message"]["id"].as_str().map(String::from);
                (self.stream_open, self.stream_delta, self.stream_start) = (true, false, self.out.content.len());
                record_response(&mut self.out, &event["message"]["id"], &event["message"]["model"], &self.model);
                if event["message"]["usage"].is_object() {
                    self.record_usage(self.stream_id.clone().as_deref(), &event["message"]["usage"]);
                }
            }
            "content_block_start" => {
                self.ensure_started();
                let block = &event["content_block"];
                match block["type"].as_str() {
                    Some("text") => {
                        let index = self.push(Content::text(""), stream_index);
                        self.emit(StreamEvent::TextStart { index });
                    }
                    Some("thinking") => {
                        let index = self.push(Content::Thinking { thinking: String::new(), thinking_signature: Some(String::new()), redacted: None }, stream_index);
                        self.emit(StreamEvent::ThinkingStart { index });
                    }
                    Some("tool_use") => {
                        let Some(name) = self.pi_tool_name(block["name"].as_str().unwrap_or_default()) else { return };
                        let id = block["id"].as_str().unwrap_or_default().to_string();
                        self.tool_ids.push(id.clone());
                        let arguments = block["input"].as_object().cloned().unwrap_or_default();
                        let index = self.push(Content::ToolCall { id: id.clone(), name: name.clone(), arguments, namespace: None, arguments_error: None, thought_signature: None }, stream_index);
                        self.emit(StreamEvent::ToolCallStart { index, id, name });
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let Some(index) = position else { return };
                let delta = &event["delta"];
                let piece = |key: &str| delta[key].as_str().unwrap_or_default().to_string();
                match (delta["type"].as_str().unwrap_or_default(), &mut self.out.content[index]) {
                    ("text_delta", Content::Text { text, .. }) => {
                        text.push_str(&piece("text"));
                        self.emit(StreamEvent::TextDelta { index, delta: piece("text") });
                    }
                    ("thinking_delta", Content::Thinking { thinking, .. }) => {
                        thinking.push_str(&piece("thinking"));
                        self.emit(StreamEvent::ThinkingDelta { index, delta: piece("thinking") });
                    }
                    // The arguments update only once the JSON so far parses whole.
                    ("input_json_delta", Content::ToolCall { arguments, .. }) => {
                        self.partial[index].push_str(&piece("partial_json"));
                        if let Ok(Value::Object(parsed)) = serde_json::from_str(&self.partial[index]) {
                            *arguments = parsed;
                        }
                        self.emit(StreamEvent::ToolCallDelta { index, delta: piece("partial_json") });
                    }
                    ("signature_delta", Content::Thinking { thinking_signature, .. }) => {
                        thinking_signature.get_or_insert_default().push_str(&piece("signature"));
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let Some(index) = position else { return };
                self.open[index] = None;
                let event = match &mut self.out.content[index] {
                    Content::Text { text, .. } => StreamEvent::TextEnd { index, content: text.clone() },
                    Content::Thinking { thinking, .. } => StreamEvent::ThinkingEnd { index, content: thinking.clone() },
                    // The final arguments parse strictly: a call whose JSON doesn't parse runs with
                    // none and the error goes back to the model, not with the prefix that parsed.
                    // No deltas at all means the block's start held the whole input.
                    Content::ToolCall { name, arguments, arguments_error, .. } => {
                        self.saw_tool_call = true;
                        if !self.partial[index].trim().is_empty() {
                            (*arguments, *arguments_error) = super::responses::parse_tool_call_arguments(&self.partial[index]);
                        }
                        *arguments = pi_arguments(name, std::mem::take(arguments));
                        StreamEvent::ToolCallEnd { index, tool_call: self.out.content[index].clone() }
                    }
                    Content::Image { .. } => return,
                };
                self.emit(event);
            }
            "message_delta" => {
                self.stream_delta = true;
                self.out.stop_reason = match event["delta"]["stop_reason"].as_str() {
                    Some("tool_use") => StopReason::ToolUse,
                    Some("max_tokens") => StopReason::Length,
                    _ => StopReason::Stop,
                };
                // message_delta carries no id: it reports on the last message_start's message.
                if event["usage"].is_object() {
                    self.record_usage(self.stream_id.clone().as_deref(), &event["usage"]);
                }
            }
            "message_stop" => {
                self.drop_unfinished();
                self.stream_open = false;
                if self.saw_tool_call {
                    self.end_turn_on_tool_use();
                }
            }
            _ => {}
        }
    }

    // A whole assistant message. After a streamed one it repeats what the stream delivered; under a
    // new id it is Claude Code's non-streamed fallback for a stalled stream, one message per block,
    // and the content path. A "<synthetic>" one is Claude Code's own failure report, kept off the
    // stream so the turn still reads as one that produced nothing.
    fn assistant(&mut self, message: &Value) {
        let Some(blocks) = message["content"].as_array() else { return };
        let id = message["id"].as_str().map(String::from);
        if message["model"] == "<synthetic>" {
            if self.saw_stream_event && self.stream_open {
                self.drop_abandoned();
            }
            // Its no-reply notice isn't the model's answer, so the turn reads as one with none.
            for text in blocks.iter().filter(|block| block["type"] == "text").filter_map(|block| block["text"].as_str()).filter(|text| !text.is_empty() && *text != history::NO_RESPONSE) {
                self.push(Content::text(text), None);
            }
            if message["usage"].is_object() {
                self.record_usage(id.as_deref(), &message["usage"]);
            }
            return;
        }
        if self.saw_stream_event {
            if id.is_none() || self.stream_id.is_none() || id == self.stream_id {
                return;
            }
            if self.stream_open {
                self.drop_abandoned();
            }
        }
        if id.is_none() || id != self.fallback_id {
            self.tool_ids.clear();
            self.fallback_id = id.clone();
        }
        for block in blocks {
            match block["type"].as_str() {
                Some("text") if block["text"].as_str().is_some_and(|text| !text.is_empty()) => self.whole_text(block["text"].as_str().unwrap_or_default()),
                Some("thinking") => {
                    self.ensure_started();
                    let thinking = block["thinking"].as_str().unwrap_or_default().to_string();
                    let signature = block["signature"].as_str().unwrap_or_default().to_string();
                    let index = self.push(Content::Thinking { thinking: thinking.clone(), thinking_signature: Some(signature), redacted: None }, None);
                    self.emit(StreamEvent::ThinkingStart { index });
                    if !thinking.is_empty() {
                        self.emit(StreamEvent::ThinkingDelta { index, delta: thinking.clone() });
                    }
                    self.emit(StreamEvent::ThinkingEnd { index, content: thinking });
                }
                Some("tool_use") => {
                    let Some(name) = self.pi_tool_name(block["name"].as_str().unwrap_or_default()) else { continue };
                    self.ensure_started();
                    self.saw_tool_call = true;
                    let call_id = block["id"].as_str().unwrap_or_default().to_string();
                    self.tool_ids.push(call_id.clone());
                    let arguments = pi_arguments(&name, block["input"].as_object().cloned().unwrap_or_default());
                    let index = self.push(Content::ToolCall { id: call_id.clone(), name: name.clone(), arguments, namespace: None, arguments_error: None, thought_signature: None }, None);
                    self.emit(StreamEvent::ToolCallStart { index, id: call_id, name });
                    self.emit(StreamEvent::ToolCallEnd { index, tool_call: self.out.content[index].clone() });
                }
                _ => {}
            }
        }
        record_response(&mut self.out, &message["id"], &message["model"], &self.model);
        if message["usage"].is_object() {
            self.record_usage(id.as_deref(), &message["usage"]);
        }
        // Every block of a fallback answer says tool_use and no message_stop follows the last, so
        // the turn ends once its blocks have settled (or sooner, when Claude Code moves on).
        if self.saw_tool_call && self.turn.is_some() {
            self.settle_at = Some(Instant::now() + FALLBACK_SETTLE);
        }
    }

    // However the query ended: answer the open stream, keep a clean end's records for the next
    // rewrite (before answering, so pi's next call finds them), delete its session files.
    async fn finish(mut self) {
        if self.failure.is_some() {
            self.settle_at = None;
        } else {
            self.flush_settle();
        }
        let status = self.child.close().await;
        let stderr = self.child.stderr_tail().await;
        // A child that ended before the open turn's result, or failed, failed that turn: what it
        // streamed is a partial answer, not one (result_seen is per turn, so an earlier turn's
        // result doesn't vouch for this one).
        let failed = status.is_none_or(|status| !status.success());
        if !self.result_seen && !self.closed && self.failure.is_none() && (failed || self.turn.is_some()) {
            self.failure = Some(match status {
                Some(status) if status.success() => "Claude Code process exited before the turn's result".into(),
                Some(status) => match status.code() {
                    Some(code) => format!("Claude Code process exited with code {code}"),
                    None => format!("Claude Code process was stopped ({status})"),
                },
                None => "Claude Code process had to be killed".into(),
            });
        }
        // Whatever failed the query (its exit, a refused session, an error result), what Claude Code
        // wrote to stderr goes with it.
        self.failure = self.failure.take().map(|failure| with_stderr(failure, &stderr));
        if let Some(failure) = self.failure.clone() {
            if !self.closed {
                self.end_open(StopReason::Error, &failure);
            }
        } else if !self.closed {
            self.keep_records();
            self.reply();
        }
        if self.closed && self.ended.is_some() {
            self.keep_records();
        }
        let ids: HashSet<&String> = self.written.iter().chain(&self.cc_session).collect();
        for id in ids {
            history::delete_session(id, &self.project_path);
        }
        debug(&format!("query done: session {} closed={} failure={:?}", self.key, self.closed, self.failure));
        self.forget();
        if let Some(done) = self.ended.take() {
            let _ = done.send(());
        }
    }
}
