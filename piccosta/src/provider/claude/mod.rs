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
// inside tools/call. A turn that ends with an answer leaves it waiting for the next message, for
// as long as its prompt cache lives. A stop keeps it, as Esc keeps Claude Code's, whenever it comes.
// The next call either delivers exactly those results (or that message) to it, or discards it and
// starts a new one from a session file holding exactly pi's context (history.rs says how). A one-off summary (compaction's) is a query of its own that ends
// with its turn. A waiting query outlives the TUI's /restart: its Claude Code goes on in the new
// image (hand_over, adopt).
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
//   CLAUDE_BRIDGE_KEEP_SESSIONS, keepSessions   off: don't keep Claude Code's records between
//                                  queries (history.rs)
//   CLAUDE_BRIDGE_HOLD_SECONDS, holdSeconds     how long a query whose turn is over waits for the
//                                  next message (default the login's cache life, hold_time; 0 ends
//                                  it with each turn)
//   CLAUDE_BRIDGE_PERSISTENCE_SECONDS, persistenceSeconds   max wait for persistence, locks and
//                                  close fences (default the SDK shutdown grace, 9 s). Timeout
//                                  fails the turn; a started worker keeps ownership until it ends.
//   CLAUDE_BRIDGE_HOLD_MAX, holdMax         how many such queries may wait at once in this piccosta
//                                  process (default HOLD_MAX); past it the one with the smallest
//                                  context closes, never a manager's
//   CLAUDE_BRIDGE_STRICT_MCP, strictMcp     off: let Claude Code load MCP servers from
//                                  ~/.claude.json and .mcp.json
//   CLAUDE_BRIDGE_LONG_CONTEXT_EXTRA_USAGE, longContextExtraUsage;
//   CLAUDE_BRIDGE_FORCE_200K, force200k     which models get 1M context (runtime_model); lists
//                                  are comma-separated in the environment, arrays in settings.json
//   CLAUDE_BRIDGE_NO_THINKING_OFF, noThinkingOff   models that can't turn thinking off
//   CLAUDE_BRIDGE_DEBUG_PATH, debugPath     append a debug log here
//   CLAUDE_BRIDGE_RECORD_STREAM, recordStream   append every line Claude Code writes here
// The plan (Max opens 1M on opus-4-6) and whether the login is an API key (the hold time) come
// from Claude Code itself: `claude auth status --json`, asked once per process (AUTH), given
// PROBE_DEADLINE like `claude --version`.

mod history;
pub use history::SIDECAR_SUFFIX;

use super::{new_output, transcript, Request, StreamEvent, BODY_IDLE_TIMEOUT};
use crate::clock;
use crate::message::{AssistantMessage, Content, Cost, CostState, Message, StopReason, ToolDefinition, ToolResultMessage, Usage};
use crate::model::Model;
use history::{fingerprint, Shape};
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use super::child::{Child, CLOSE_GRACE};
pub(crate) use super::child::TERM_GRACE;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{sleep_until, timeout, Instant};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub const PROVIDER: &str = "claude-bridge";

pub(crate) fn persistence_timeout() -> Duration { SETTINGS.persistence }
pub(crate) fn sidecars(file: &Path) -> [PathBuf; 2] {
    let sidecar = PathBuf::from(format!("{}{SIDECAR_SUFFIX}", file.display()));
    let lock = history::lock_path(&sidecar);
    [sidecar, lock]
}
pub(crate) fn deletion_lock(file: &Path) -> std::io::Result<std::fs::File> {
    history::lock_sidecar(&sidecars(file)[0], &AtomicBool::new(false))
}
const MCP_SERVER: &str = "custom-tools";
pub const MCP_TOOL_PREFIX: &str = "mcp__custom-tools__";
// Claude Code keeps a tool result over 50,000 characters out of the conversation (a preview and a
// file path instead) unless the tool raises that ceiling, up to 500,000. pi cuts read and bash at
// 50 KB, so every tool declares the maximum and pi's results arrive whole (2.1.300's tool-result
// store, read in the binary; 2.1.316 against a stand-in API: a 51,341-character result arrives whole).
// Past the ceiling the model gets a 2,000-character preview and a path to the rest.
const MCP_RESULT_MAX_CHARS: u64 = 500_000;
// The file a resumed query reads pi's prompt from, in the session's own dir (deleted with it).
const SYSTEM_PROMPT_FILE: &str = "system-prompt.md";
// Every model is asked for with [1m], at 1M. An account not entitled to that gets every such turn
// refused with one of these (Claude Code 2.1.321, read in the binary: the API's 429 message, which
// its own check for the case matches, and the line it puts in its place on a subscription). The
// model is then asked for bare, at 200K, from that turn on, and kept in REFUSED_1M_FILE beside the
// model lists so a restart and the model list know it.
const LONG_CONTEXT_REFUSALS: [&str; 3] = ["Extra usage is required for long context", "Usage credits are required for long context", "Usage credits required for 1M context"];
const REFUSED_1M_FILE: &str = "claude-bridge-200k.json";
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
// Claude Code resuming a transcript that ends at a user record (tool results, maybe a message
// after them) takes that turn as owed and answers it at once, with the results and any message in
// one user turn. Without these it writes "No response requested." as the model's reply there,
// which the model then reads as its own and copies. Its own default age limit for that (an hour)
// is lifted, since a resumed history can be days old; the continue note is what it adds when the
// turn ends at tool results alone. Found in Claude Code 2.1.316's resume code, and checked against
// a stand-in model: one request, no notice.
const RESUME_ENV: [(&str, &str); 3] = [("CLAUDE_CODE_RESUME_INTERRUPTED_TURN", "1"), ("CLAUDE_CODE_RESUME_INTERRUPTED_TURN_MAX_AGE_MS", "315360000000"), ("CLAUDE_CODE_RESUME_PROMPT", history::CONTINUE_NOTE)];
// What a parent Claude Code session sets for the processes under it (what `env` showed inside a
// Claude Code Bash call, 2026-10-15, plus the two 2.1.312 sets on 2026-10-29). Left in, the child
// would act as part of that session. NODE_OPTIONS and DEBUG go too, as the SDK removed them.
const PARENT_ENV: [&str; 18] = ["CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_EXECPATH", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_EFFORT", "CLAUDE_PID", "CLAUDE_PLUGIN_DATA", "CODEX_COMPANION_SESSION_ID",
    "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_EVEN_MORE_DANGEROUSLY_SKIP_PERMISSIONS", "NODE_OPTIONS", "DEBUG"];
// The settings file every provider child gets: no auto-memory, no CLAUDE.md or AGENTS.md (pi
// owns context files), no git status block (a git change would rewrite it and bust the cache), no
// hooks (they cost seconds and tokens per query, and an opened repo's would run untrusted).
const PROVIDER_SETTINGS: &str = r#"{"autoMemoryEnabled":false,"claudeMdExcludes":["**/CLAUDE.md","**/AGENTS.md","**/.claude/rules/**"],"includeGitInstructions":false,"disableAllHooks":true}"#;
// How long a parked tools/call waits before checking that pi was handed its call, and between
// checks after (pi-claude-bridge #226): far longer than the stream event announcing it can lag.
const LOST_CALL_CHECK: Duration = Duration::from_secs(120);
// What a stopped turn's tool call is answered at once (interrupt): pi dropped that turn with its
// calls, so nothing else would ever answer it, and Claude Code would wait on it until the check above.
const STOPPED_CALL: &str = "Not run: the user stopped this turn. Wait for their next message.";
// How long a non-streamed answer's last tool call waits for another block before the turn ends:
// Claude Code writes all of a message's blocks at once (pi-claude-bridge #225).
const FALLBACK_SETTLE: Duration = Duration::from_millis(100);
// The longest shutdown waits for the queries' tasks to close their children that way.
const SHUTDOWN_WAIT: Duration = CLOSE_GRACE.saturating_add(TERM_GRACE).saturating_add(CLOSE_GRACE);
// How long a query that answered stays open for the next message: the life of the prompt cache it
// wrote, past which a new query costs no more than this one would. Claude Code 2.1.316 asks for a
// 1 h cache on a subscription (ttl "1h", reason "subscriber", read in the binary) and 5 min with
// an API key; every cache write in 30 h of real claude-bridge sessions (2026-11-02) was a 1 h one.
const CACHE_TTL: Duration = Duration::from_secs(60 * 60);
const API_KEY_CACHE_TTL: Duration = Duration::from_secs(5 * 60);
// How many answered queries may wait at once. Each is an idle Claude Code process of about 200 MB
// RSS (2.1.316, measured 2026-11-02), so four is under 1 GB: enough to keep the caches of the few
// sessions one person switches between warm, small enough not to matter on an ordinary laptop. A
// machine with more to spare raises it. Counted in this piccosta process only: another piccosta's
// waiting queries are its own.
const HOLD_MAX: usize = 4;
// The last Claude Code release the bridge was checked against: CHILD_ENV, RESUME_ENV and
// MCP_RESULT_MAX_CHARS were read in this binary. A newer one gets a notice, since any of them
// could have been renamed.
const VERIFIED_VERSION: &str = "2.1.322";
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
    // None: the login's cache life (hold_time).
    hold: Option<Duration>,
    hold_max: usize,
    persistence: Duration,
    strict_mcp: bool,
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
    // No room for a waiting query is the same as no wait: each query ends with its turn.
    let hold_max = setting("HOLD_MAX", "holdMax").and_then(|count| count.parse().ok()).unwrap_or(HOLD_MAX);
    let hold = setting("HOLD_SECONDS", "holdSeconds").and_then(|seconds| seconds.parse().ok()).map(Duration::from_secs);
    Settings {
        claude: setting("CLAUDE_PATH", "claudePath").unwrap_or_else(|| "claude".into()),
        config_dir: setting("CONFIG_DIR", "configDir").map(PathBuf::from),
        keep: setting("KEEP_SESSIONS", "keepSessions").is_none_or(|keep| is_on(&keep)),
        hold: if hold_max == 0 { Some(Duration::ZERO) } else { hold },
        hold_max,
        persistence: setting("PERSISTENCE_SECONDS", "persistenceSeconds").and_then(|seconds| seconds.parse().ok()).map(Duration::from_secs).unwrap_or(SHUTDOWN_WAIT),
        strict_mcp: setting("STRICT_MCP", "strictMcp").is_none_or(|strict| is_on(&strict)),
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
// The session's own line first, if Claude Code refused it something (fast mode).
pub fn take_notice(session_id: &str) -> Option<String> {
    let numbers = |version: &str| version.split('.').map(|part| part.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    lock(&NOTES).remove(session_id).or_else(|| VERSION.get().cloned().flatten().filter(|version| numbers(version) > numbers(VERIFIED_VERSION) && !NOTICED.swap(true, Ordering::Relaxed)).map(|version| format!(
        "Claude Code {version} is newer than {VERIFIED_VERSION}, the last version the bridge was checked against; if tool results arrive cut short, look there first")))
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

// How long an answered query waits for the next message, also for the cache monitor: a miss after
// a longer gap is one the cache had already lapsed for. Unset, it's the life of the cache Claude
// Code asks for: 5 minutes on an API key, an hour on a subscription.
pub fn hold_time() -> Duration {
    SETTINGS.hold.unwrap_or_else(cache_ttl)
}

// The life of the prompt cache Claude Code asks for on this login (CACHE_TTL).
pub fn cache_ttl() -> Duration {
    if AUTH.as_ref().is_some_and(|auth| auth["authMethod"] == "api_key") { API_KEY_CACHE_TTL } else { CACHE_TTL }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// --- Models ---

// The models Claude Code refused 1M context for, as REFUSED_1M_FILE kept them, read once per process.
static REFUSED_1M: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(read_refused_1m()));

fn refused_1m_path() -> PathBuf {
    crate::config::agent_dir().join(crate::model::LISTS_DIR).join(REFUSED_1M_FILE)
}

fn read_refused_1m() -> HashSet<String> {
    std::fs::read_to_string(refused_1m_path()).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

// Keep a model as refused 1M, with what another piccosta process kept meanwhile.
fn refuse_1m(id: &str) {
    let mut refused = lock(&REFUSED_1M);
    refused.insert(id.to_string());
    refused.extend(read_refused_1m());
    let _ = crate::platform::private_dir(&crate::config::agent_dir().join(crate::model::LISTS_DIR));
    if let Err(error) = crate::auth::write_private_json(&refused_1m_path(), &json!(refused.iter().collect::<std::collections::BTreeSet<_>>())) {
        debug(&format!("couldn't keep {id} as refused 1M: {error}"));
    }
}

// Input: a model id. Output: the id Claude Code is asked for and the window it serves. 1M needs the
// [1m] suffix, given unless Claude Code refused it (REFUSED_1M) or the user pinned the model to
// 200K; opus-4-6 and sonnet-4-6 bill 1M as extra usage, so they get it only where that's asked for
// (opus-4-6 also on Max, where it's in the plan).
fn runtime_model(id: &str) -> (String, u64) {
    let settings = &*SETTINGS;
    let billed = match id {
        "claude-opus-4-6" => !settings.extra_usage && AUTH.as_ref().is_none_or(|auth| auth["subscriptionType"] != "max"),
        "claude-sonnet-4-6" => !settings.extra_usage,
        _ => false,
    };
    if !billed && !settings.force_200k.iter().any(|forced| forced == id) && !lock(&REFUSED_1M).contains(id) {
        (format!("{id}[1m]"), ONE_M)
    } else {
        (id.to_string(), TWO_HUNDRED_K)
    }
}

// The window pi compacts against: where Claude Code starts refusing, or what its own overrides
// (CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE; CLAUDE_CODE_MAX_CONTEXT_TOKENS with DISABLE_COMPACT) make it.
// Input: the model's id and output limit.
pub fn context_window(id: &str, max_tokens: u64) -> u64 {
    let (_, window) = runtime_model(id);
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
    let mut command = claude_command();
    command.args(base_args(false, None)).args(["--tools=", "--setting-sources=", "--strict-mcp-config", "--no-session-persistence"]);
    for name in PARENT_ENV {
        command.env_remove(name);
    }
    // Killed and reaped as it's dropped here, however the answer went.
    let mut child = start(command)?;
    let id = request_id();
    child.send(&json!({"type": "control_request", "request_id": id, "request": {"subtype": subtype}})).await;
    let answer = timeout(OFFERED_WAIT, async {
        while let Some(line) = child.next_line(None).await {
            let message: Value = serde_json::from_str(&line).unwrap_or_default();
            if message["type"] == "control_response" && message["response"]["request_id"] == id.as_str() {
                return Some(message);
            }
        }
        None
    }).await;
    drop(child);
    answer.map_err(|_| format!("Claude Code didn't answer {subtype}"))?.ok_or_else(|| format!("Claude Code ended before answering {subtype}"))
}

// Ask Claude Code how much of the plan is used (get_usage, a query that asks no model): through a
// waiting query when there is one, whose answer is noted when it comes (Query::line), else through a
// Claude Code started for it (1.3 s, 2026-11-03).
pub async fn ask_usage() -> Result<(), String> {
    let waiting = lock(&LIVE).values().find(|live| lock(&live.state).waiting).map(|live| live.commands.clone());
    if let Some(commands) = waiting {
        return commands.send(Command::Usage).map_err(|_| "the waiting query ended".to_string());
    }
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
        let context_window = context_window(&id, max_tokens);
        let name = model.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
        let says_1m = name.split(|c: char| !c.is_alphanumeric()).any(|word| word.eq_ignore_ascii_case("1m"));
        let name = if window > TWO_HUNDRED_K && !says_1m { format!("{name} 1M") } else { name };
        let mut out = json!({"id": id, "name": name, "api": PROVIDER, "provider": PROVIDER, "baseUrl": PROVIDER,
            "reasoning": model.get("reasoning").cloned().unwrap_or(json!(false)), "input": model.get("input").cloned().unwrap_or(json!(["text"])),
            "contextWindow": context_window, "maxTokens": max_tokens});
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

fn spawn(args: &[String], cwd: &str, resumes_turn: bool) -> Result<Child, String> {
    version();
    let mut command = claude_command();
    command.args(args).current_dir(cwd);
    for name in PARENT_ENV {
        command.env_remove(name);
    }
    command.envs(CHILD_ENV);
    if resumes_turn {
        command.envs(RESUME_ENV);
    }
    debug(&format!("spawn: {} {}", SETTINGS.claude, args.join(" ")));
    start(command)
}

// The pipes and process ownership are shared with ACP; Claude's environment and handshake aren't.
fn start(command: std::process::Command) -> Result<Child, String> {
    Child::start(command).map_err(|error| format!("Claude Code could not be started ({}): {error}", SETTINGS.claude))
}

// A failure with the child's stderr after it, when it wrote any.
fn with_stderr(failure: String, tail: &str) -> String {
    if tail.is_empty() { failure } else { format!("{failure}\nClaude Code's stderr: {tail}") }
}

// How many query tasks haven't finished, live or let go of and still closing their child, for
// shutdown to wait on, and how often it looks.
static QUERIES: AtomicUsize = AtomicUsize::new(0);
const SHUTDOWN_POLL: Duration = Duration::from_millis(10);

// piccosta is exiting (a quit, a signal, /restart's exec once hand_over has taken the waiting
// queries out), where nothing would be dropped: stop every live query and wait for every query
// task, these and the ones let go of earlier, to close its child (end of input, SIGTERM, SIGKILL)
// and delete its session files. A child still there
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

// How a waiting query's get_usage request id begins, so its answer is told from the session's own.
const USAGE_REQUEST: &str = "usage-";
// And a fast-mode request's, followed by the value it asks for ("true-" or "false-").
const FAST_REQUEST: &str = "fast-";

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
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
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
    Usage { tokens_unknown: false, reported: None, input: sum.input, output: sum.output, cache_read: sum.cache_read, cache_write: sum.cache_write, cache_write_1h: Some(sum.cache_write_1h),
        reasoning: sum.reasoning, total_tokens: sum.input + sum.output + sum.cache_read + sum.cache_write, cost: Default::default() }
}

// The context a query holds, which a rewrite would write to the cache again: its largest request's
// (a query's context only grows, and a surplus the result counted is no request of its own).
fn context(requests: &[Counters]) -> u64 {
    requests.iter().map(|request| request.input + request.cache_read + request.cache_write).max().unwrap_or(0)
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
    // Its turn is over, stopped mid-answer (pi's abort) or answered, and it waits, as Claude Code
    // does at its prompt, for the next message, which goes into this same query (interrupt, hold).
    waiting: bool,
    // Parked on calls a stopped run left behind (hold_parked): it waits for their results as a
    // waiting query does for its message.
    stopped: bool,
    // For the cap on waiting queries: what closing it would write again (the context it holds, in
    // tokens), and whether it's a manager's, which is never closed.
    context: u64,
    manager: bool,
    kept_base: (Vec<String>, Vec<String>),
    // Bound to this query's attachment, never looked up again by a late worker after close.
    kept: Option<history::Handle>,
    ownership: Option<std::sync::Weak<crate::runtime::Ownership>>,
    // Whether Claude Code was told fast mode is on (/fast).
    fast: bool,
}

struct Live {
    id: u64,
    state: Arc<Mutex<State>>,
    commands: mpsc::UnboundedSender<Command>,
}

enum Command {
    // The call's tool results (id, MCP result) and its stream.
    Deliver(Vec<(String, Value)>, Turn),
    // The next message for a waiting query, and its stream.
    Prompt(Vec<Value>, Turn),
    Discard(String),
    // pi's run is over (end_run).
    Hold,
    // Ask Claude Code for the plan's usage (ask_usage).
    Usage,
    // Fast mode on or off from its next request, as the flag setting it started with would.
    Fast(bool),
    // /restart: acknowledge the safe handover boundary, then give it to the next image.
    Drain(oneshot::Sender<()>),
    HandOver(oneshot::Sender<Handed>),
}

// A query waiting for its session's next message, as /restart hands it to the image it execs (the
// TUI's handover file): its Claude Code, still this process's child, with copies of its stdin,
// stdout and stderr kept open across the exec, and what the bridge knows of it. The next message
// goes into that same process, so the request repeats what its prompt cache holds.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Handed {
    // Its pid and start time, which name it for sure even when the rest of the record can't be read,
    // and its pipes' copies, closed once it's reaped then (tui reap_inherited_children).
    pub pid: u32,
    start: Option<u64>,
    fds: [i32; 3],
    key: String,
    model: Model,
    tools: Vec<String>,
    manifest: Value,
    shape: Shape,
    prints: Vec<String>,
    project_path: String,
    cwd: String,
    written: Option<String>,
    cc_session: Option<String>,
    hold_ms: Option<u64>,
    cost_seen: f64,
    pending_turns: u32,
    // Every request of the query so far, which a result's usage is weighed against
    // (add_unstreamed_usage) as if there had been no restart; and what it had written to stderr.
    // A record without them (an earlier build's) has none: a result of Claude Code 2.1.319 counts
    // only its own turn's usage, so nothing is counted twice.
    #[serde(default)]
    requests: Vec<Counters>,
    #[serde(default)]
    stderr: Vec<u8>,
    // Whether its model came with prices (model.rs ModelCost, whose serde always reads true). Not
    // known: priced as unpriced, at Claude Code's own reckoning where it bills by the token (price).
    #[serde(default)]
    cost_supplied: bool,
    // Fast mode as the session last asked for it and as Claude Code last took it (/fast), a
    // refusal's line the session hasn't shown yet, and whether it's a manager's (State).
    #[serde(default)]
    fast: (bool, bool),
    #[serde(default)]
    notice: Option<String>,
    #[serde(default)]
    manager: bool,
    // Earlier queries were started without transcript-write acknowledgements.
    #[serde(default)]
    session_mirror: bool,
}

// Ask each query to acknowledge only once Claude Code has finished its turn, including the result
// of an interrupt. The UI keeps taking input and waiting, however long that result takes.
pub fn drain() -> Vec<oneshot::Receiver<()>> {
    lock(&LIVE).values().filter_map(|live| {
        let (reply, answer) = oneshot::channel();
        live.commands.send(Command::Drain(reply)).ok().map(|()| answer)
    }).collect()
}

// /restart: each query waiting for its next message gives its Claude Code to the image this process
// execs (adopt); any other closes in shutdown, as at every exit. Output: one record each, its
// pipes' copies now kept open across the exec.
pub async fn hand_over() -> Vec<Handed> {
    let asked: Vec<oneshot::Receiver<Handed>> = lock(&LIVE).values().filter_map(|live| {
        let (reply, answer) = oneshot::channel();
        live.commands.send(Command::HandOver(reply)).ok().map(|()| answer)
    }).collect();
    let handed: Vec<Handed> = futures_util::future::join_all(asked).await.into_iter().flatten().collect();
    #[cfg(unix)]
    handed.iter().flat_map(|query| query.fds).for_each(crate::tools::reaper::keep_across_exec);
    handed
}

// After /restart: take back the queries the image before handed over. Each Claude Code waits, as
// it did, for its session's next message, which then goes into it (decide) as if nothing happened.
#[cfg(unix)]
pub fn adopt(handed: Vec<Handed>) {
    for query in handed {
        let [stdin, stdout, stderr] = query.fds.map(crate::tools::reaper::inherited);
        let Ok(mut child) = Child::new(query.pid, Some(stdin.into()), Some(stdout.into()), Some(stderr.into()), query.stderr) else {
            // Its pipes closed with that; reaped here, since nothing else would, and ended only while
            // its pid still names the process the record started (its start time), never another.
            let named = query.start.is_some_and(|start| crate::proc::Process::started(query.pid as i32, start).is_some());
            crate::tools::reaper::reap_inherited(query.pid as i32, named.then_some(TERM_GRACE), Vec::new());
            continue;
        };
        // A Claude Code whose binary was updated since it started is let go too: the session's next
        // turn starts the new one on the records it kept, so every session runs the release it has.
        let updated = std::fs::read_link(format!("/proc/{}/exe", query.pid)).is_ok_and(|exe| exe.to_string_lossy().ends_with(" (deleted)"));
        if !query.session_mirror || updated {
            debug(&format!("adopt: session {} resumes its kept records; Claude Code {} {}", query.key, query.pid, if updated { "was updated since it started" } else { "has no transcript mirror" }));
            tokio::spawn(async move { child.close().await; });
            continue;
        }
        debug(&format!("adopt: session {} keeps Claude Code {} across the restart", query.key, query.pid));
        if let Some(notice) = query.notice {
            lock(&NOTES).insert(query.key.clone(), notice);
        }
        let state = Arc::new(Mutex::new(State { shape: query.shape.clone(), prints: query.prints, pending: None, waiting: true, stopped: false,
            context: context(&query.requests), manager: query.manager, kept_base: Default::default(), kept: None, ownership: None, fast: query.fast.0 }));
        let (commands, inbox) = mpsc::unbounded_channel();
        let id = NEXT_QUERY.fetch_add(1, Ordering::Relaxed);
        let mut adopted = Query::new(id, &query.key, query.model, &[], child, inbox, state.clone(), query.project_path, &query.cwd, query.shape, query.written);
        (adopted.tools, adopted.manifest, adopted.cc_session, adopted.fast) = (query.tools, query.manifest, query.cc_session, query.fast.1);
        (adopted.cost_seen, adopted.pending_turns, adopted.requests) = (query.cost_seen, query.pending_turns, query.requests);
        adopted.model.cost.supplied = query.cost_supplied;
        adopted.hold_until = query.hold_ms.map(|ms| Instant::now() + Duration::from_millis(ms));
        lock(&LIVE).insert(history::query_key(&query.key, &adopted.shape), Live { id, state, commands });
        QUERIES.fetch_add(1, Ordering::SeqCst);
        tokio::spawn(adopted.run(Vec::new()));
    }
}

#[cfg(windows)]
pub fn adopt(_handed: Vec<Handed>) {}

// One provider call's stream: where its events go, the message it ends with, and its abort.
struct Turn {
    events: mpsc::UnboundedSender<StreamEvent>,
    reply: oneshot::Sender<AssistantMessage>,
    cancel: CancellationToken,
}

type QueryKey = (String, String);
static LIVE: LazyLock<Mutex<HashMap<QueryKey, Live>>> = LazyLock::new(Default::default);
static NEXT_QUERY: AtomicU64 = AtomicU64::new(1);
// Why each session's latest Claude Code query was started, for the cache monitor (take_rewrite); and
// why its last query ended, for the reason the next one starts (deliver).
static REWRITES: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Default::default);
static ENDED: LazyLock<Mutex<HashMap<QueryKey, String>>> = LazyLock::new(Default::default);

// A line for each session's user, taken with take_notice.
static NOTES: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Default::default);

// The reason the session's latest query was started, for the first response that reached the API
// through it: a call stopped before that (no usage) leaves it for the next, which writes what the
// query holds to the cache.
pub fn take_rewrite(session_id: &str, message: &AssistantMessage) -> Option<String> {
    (message.usage.total_tokens > 0).then(|| lock(&REWRITES).remove(session_id)).flatten()
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

// The end of one of pi's runs. The session's query stays. A waiting one (it answered, or was stopped
// mid-answer) takes the next message; one parked on its calls (the run was stopped while they ran)
// holds them for the next call, which carries their results (hold_parked). Either ends when its
// cache would have lapsed, its session closes (close) or piccosta exits (shutdown).
pub fn end_run(session_id: &str) {
    for (_, live) in lock(&LIVE).iter().filter(|((session, _), _)| session == session_id) {
        let _ = live.commands.send(Command::Hold);
    }
}

// A pi session closed, deleted or replaced: its query, waiting or not, ends now rather than when
// its cache lapses.
pub async fn close(session_id: &str) -> Result<(), String> {
    lock(&LIVE).retain(|(session, _), live| {
        if session == session_id {
            let _ = live.commands.send(Command::Discard("the session was closed".into()));
        }
        session != session_id
    });
    history::close(session_id).await.map_err(|error| error.to_string())
}

// Whether the session's query waits for its next message, or for the results of the calls it's
// parked on: the only query an idle compaction may ask (agent.rs idle_compaction_due), whose summary
// request goes in as the next call would, on the end of those results. Parked is `pending`, set as
// the turn ends, not `stopped`, which waits on the query's task taking end_run's Hold, maybe after
// the session has already asked this.
pub fn waiting(session_id: &str) -> bool {
    lock(&LIVE).iter().any(|((session, _), live)| session == session_id && { let state = lock(&live.state); state.waiting || state.pending.is_some() })
}

// Keep the waiting queries within CLAUDE_BRIDGE_HOLD_MAX, as each starts waiting, once its context
// is known. The ones with the smallest context close first, the one just waiting too, since their
// rewrites write the least again. A manager's never closes: it's the standing session its lanes
// and schedules message.
fn make_room() {
    let mut all = lock(&LIVE);
    let mut waiting: Vec<(u64, QueryKey, bool)> = all.iter().filter_map(|(key, live)| {
        let state = lock(&live.state);
        (state.waiting || state.stopped).then(|| (state.context, key.clone(), state.manager))
    }).collect();
    // Managers don't count toward the limit, so other sessions keep their hold however many wait.
    waiting.retain(|(.., manager)| !manager);
    let over = waiting.len().saturating_sub(SETTINGS.hold_max);
    waiting.sort();
    for (_, key, _) in waiting.into_iter().take(over) {
        if let Some(live) = all.remove(&key) {
            let_go(live, &format!("more than {} queries waiting (claudeBridge.holdMax)", SETTINGS.hold_max));
        }
    }
}

// --- One provider call ---

// Input: the request and where its events go. Output: the turn's message, delivered to the
// session's query of this shape, or resumed from its kept records when the context changed.
pub async fn stream(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let id = &request.model.id;
    let (cli_model, _) = runtime_model(id);
    let mut answer = stream_as(request, events, &cli_model).await;
    // A refused 1M is that model's last: kept as 200K, and this same turn asked again bare. The
    // refusal streamed nothing to pi (assistant keeps Claude Code's report off the stream). A model
    // that quietly serves only 200K says so as an overflow well short of 1M, which counts the same.
    let refused = |answer: &AssistantMessage| answer.error_message.as_deref().is_some_and(|error| LONG_CONTEXT_REFUSALS.iter().any(|refusal| error.contains(refusal)));
    let short = |answer: &AssistantMessage| crate::retry::is_context_overflow(answer, 0)
        && !answer.error_message.as_deref().is_some_and(|error| ["1000000", "1,000,000", "1M"].iter().any(|million| error.contains(million)));
    if (refused(&answer) || short(&answer)) && cli_model != *id {
        debug(&format!("Claude Code refused 1M context for {id}; asking for it bare at 200K from now on"));
        // With extra usage on, a refusal is its spending limit, not the plan: this process stays at
        // 200K (no flip back each turn), and the next start asks for 1M again.
        let kept = if SETTINGS.extra_usage { lock(&REFUSED_1M).insert(id.to_string()); "until piccosta restarts".to_string() } else { refuse_1m(id); format!("from now on (kept in {})", refused_1m_path().display()) };
        if let Some(session) = request.session_id {
            lock(&NOTES).insert(session.to_string(), format!("Claude Code refused 1M context for {id} on this account, so it runs at 200K {kept}"));
        }
        answer = stream_as(request, events, id).await;
    }
    // A model with 1M of its own gets no long-context flag to refuse, only a context past 200K, and
    // that one is refused bare too: said as an overflow, so pi compacts to fit 200K and asks again.
    if refused(&answer) {
        answer.error_message = Some(format!("Prompt is too long for {id} at 200K context"));
    }
    answer
}

// One provider call as `stream` makes it, asking Claude Code for `cli_model`.
async fn stream_as(request: &Request<'_>, events: &mpsc::UnboundedSender<StreamEvent>, cli_model: &str) -> AssistantMessage {
    let model = request.model;
    let system = transcript::current_system_message(request.messages);
    let system_prompt = system.as_ref().map(transcript::system_message_text).unwrap_or_default();
    let tools = system.and_then(|system| system.tools_added).unwrap_or_default();
    // A turn stopped or failed is not part of what the model sees (history::visible), and leaving it in
    // would keep a message sent after a stopped run's results from going on the end of the last one.
    let messages = fold_into_results(request.messages.iter().filter(|message| !matches!(message, Message::System(_)) && history::visible(message)).cloned());
    let effort = effort(model, request.thinking_level);
    let thinking_off = request.thinking_level == "off";
    let fast = model.service_tier.as_deref() == Some(super::FAST_TIER);
    let cwd = request.cwd.to_string_lossy().into_owned();
    let key = request.session_id.unwrap_or("(none)").to_string();
    let shape = history::shape(&system_prompt, &tools, cli_model, if thinking_off { Some("off") } else { effort.as_deref() }, &cwd);
    let ownership = match request.ownership.as_ref().map(crate::runtime::Ownership::retain).transpose() {
        Ok(ownership) => ownership,
        Err(error) => return failed(model, StopReason::Error, &error.to_string()),
    };
    if let Err(error) = history::attach(&key, request.session_file, ownership).await {
        return failed(model, StopReason::Error, &error.to_string());
    }
    let prints: Vec<String> = messages.iter().map(fingerprint).collect();
    // A run stopped before its call got here leaves the session's query as Esc leaves Claude Code's,
    // and starts nothing. A waiting one waits on; one parked on the calls whose results this call
    // carries stays parked (hold_parked), and the next call, whose message goes on the end of the
    // last result (fold_into_results), hands them over once. Handing them over now and then stopping
    // could leave Claude Code holding them as interrupted while pi holds them as they ran.
    if request.cancel.is_cancelled() {
        return failed(model, StopReason::Aborted, "Operation aborted");
    }
    // An idle compaction that went in leaves the query's continuation as it found it: stopped, or its
    // summary refused, the session's next message goes into the query as if it hadn't run (Claude
    // Code keeps the turn, after the context the cache holds); kept, the context is the summary's, a
    // new query. What it holds then is this call's context without the summary prompt: on a parked
    // query, the results it was given, which the next call's message is folded onto (unfold).
    let before = request.in_place.then(|| fold_into_results(request.messages[..request.messages.len().saturating_sub(1)].iter()
        .filter(|message| !matches!(message, Message::System(_)) && history::visible(message)).cloned()).iter().map(fingerprint).collect::<Vec<_>>());
    let mut forced: Option<String> = None;
    loop {
        let (reply, answer) = oneshot::channel();
        let turn = Turn { events: events.clone(), reply, cancel: request.cancel.clone() };
        let (reason, turn) = match deliver(&key, &messages, &prints, &shape, fast, request.manager, forced.take(), turn, &request.ownership) {
            Ok(()) => match answer.await {
                Ok(message) => {
                    if let Some(prints) = before && let Some(live) = lock(&LIVE).get(&history::query_key(&key, &shape)) {
                        lock(&live.state).prints = prints;
                    }
                    return message;
                }
                // The query ended before it took them: rewrite like any other mismatch.
                Err(_) => {
                    forced = Some("the parked query ended before it took the results".into());
                    continue;
                }
            },
            Err(rewrite) => rewrite,
        };
        debug(&format!("sync: rewrite, {reason} (session {key})"));
        lock(&REWRITES).insert(key.clone(), reason.clone());
        let start = Start { key: &key, model, cli_model, effort: effort.as_deref(), thinking_off, fast, system_prompt: &system_prompt,
            tools: &tools, messages: &messages, prints: &prints, shape: &shape, cwd: &cwd, manager: request.manager, ownership: &request.ownership };
        if let Err(error) = start_query(start, turn).await {
            return failed(model, if request.cancel.is_cancelled() { StopReason::Aborted } else { StopReason::Error }, &error);
        }
        return answer.await.unwrap_or_else(|_| failed(model, StopReason::Error, "Claude Code ended the query without answering"));
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

// Hand this call's tool results to the parked query, or its next message to the waiting one, when
// the sync check allows it. Output: Ok once they're sent (the next turn streams into `turn`), else
// the rewrite reason and the turn back; the live query, if any, is discarded then.
#[allow(clippy::too_many_arguments)]
fn deliver(key: &str, messages: &[Message], prints: &[String], shape: &Shape, fast: bool, manager: bool, forced: Option<String>, turn: Turn, ownership: &Option<std::sync::Weak<crate::runtime::Ownership>>) -> Result<(), (String, Turn)> {
    let query_key = history::query_key(key, shape);
    let (mut live, mut unfolded) = (lock(&LIVE), None);
    let decision = match (forced, live.get(&query_key)) {
        (Some(reason), _) => Err(reason),
        (None, None) => Err(lock(&ENDED).get(&query_key).map_or("no live query for this session and shape".into(), |why| format!("no live query for this session and shape; the last one ended: {why}"))),
        (None, Some(entry)) => {
            let state = lock(&entry.state);
            history::decide(&state, messages, prints, shape).or_else(|reason| { unfolded = unfold(&state, messages, prints, shape); unfolded.as_ref().map(|_| 0).ok_or(reason) })
        }
    };
    let start = match decision {
        Ok(start) => start,
        Err(reason) => {
            if let Some(old) = live.remove(&query_key) {
                let_go(old, &format!("the session's context changed: {reason}"));
            }
            return Err((reason, turn));
        }
    };
    let Some(entry) = live.get(&query_key) else { return Err(("no live query for this session".into(), turn)) };
    // A waiting query takes the user messages that came after its turn as its next message; a
    // parked one, the results of the calls it's parked on.
    let mut state = lock(&entry.state);
    // Fast mode is a request field, not part of the prompt, so a running query takes it as it is.
    // Whether it's a manager's, from the session each turn (a query adopted from an older piccosta lacks it).
    state.manager = manager;
    if state.kept.is_none() { state.kept = history::handle(key); }
    if state.ownership.is_none() { state.ownership = ownership.clone(); }
    if state.fast != fast {
        state.fast = fast;
        let _ = entry.commands.send(Command::Fast(fast));
    }
    let command = if state.waiting {
        let content = unfolded.unwrap_or_else(|| messages[start..].iter().filter_map(|message| match message {
            Message::User(user) => Some(history::anthropic_blocks(&user.content.clone().into_blocks())),
            _ => None,
        }).flatten().collect());
        debug(&format!("sync: the waiting query takes the next message (session {key}, {} messages)", prints.len()));
        Command::Prompt(content, turn)
    } else {
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
        debug(&format!("sync: deliver to the parked query (session {key}, {} messages)", prints.len()));
        Command::Deliver(results, turn)
    };
    (state.prints, state.pending, state.waiting, state.stopped, state.kept_base) = (prints.to_vec(), None, false, false, kept_base(messages));
    drop(state);
    let Err(mpsc::error::SendError(Command::Prompt(_, turn) | Command::Deliver(_, turn))) = entry.commands.send(command) else { return Ok(()) };
    live.remove(&query_key);
    Err(("the query is gone".into(), turn))
}

// A waiting query whose context ends at tool results, as an idle compaction's summary stopped on a
// parked query leaves it (stream_as): pi's next call carries those results with its message folded
// on (fold_into_results). Output: that message, what comes after the ARRIVED_NOTE that leaves the
// results Claude Code holds, when everything before them is held too; else None.
fn unfold(state: &State, messages: &[Message], prints: &[String], shape: &Shape) -> Option<Vec<Value>> {
    let held = state.prints.len();
    let Some(Message::ToolResult(last)) = messages.last().filter(|_| state.waiting && held == messages.len() && *shape == state.shape && prints[..held - 1] == state.prints[..held - 1]) else { return None };
    let note = |block: &Content| matches!(block, Content::Text { text, .. } if text == ARRIVED_NOTE);
    let at = (0..last.content.len()).rev().find(|&at| note(&last.content[at])
        && fingerprint(&Message::ToolResult(ToolResultMessage { content: last.content[..at].to_vec(), ..last.clone() })) == state.prints[held - 1])?;
    debug("sync: the waiting query holds the results pi's message was folded onto");
    Some(history::anthropic_blocks(&last.content[at + 1..]))
}

struct Start<'a> {
    key: &'a str,
    model: &'a Model,
    cli_model: &'a str,
    effort: Option<&'a str>,
    thinking_off: bool,
    fast: bool,
    system_prompt: &'a str,
    tools: &'a [ToolDefinition],
    messages: &'a [Message],
    prints: &'a [String],
    shape: &'a Shape,
    cwd: &'a str,
    manager: bool,
    ownership: &'a Option<std::sync::Weak<crate::runtime::Ownership>>,
}

// Start a query holding exactly pi's context: the history in a fresh session file it resumes, the
// trailing user turn as its opening prompt; or, when the turn starts at tool results, all of it in
// the history, for Claude Code to answer on resume.
async fn start_query(start: Start<'_>, turn: Turn) -> Result<(), String> {
    let visible: Vec<Message> = start.messages.iter().filter(history::visible).cloned().collect();
    let (history_end, prompt, continued) = history::split(&visible);
    // A context whose last turn starts at tool results goes into the history whole, and Claude Code
    // answers it on resume (RESUME_ENV), rather than ending the history at the results.
    let resumes_turn = history_end > 0 && matches!(visible[history_end - 1], Message::ToolResult(_));
    let history_end = if resumes_turn { visible.len() } else { history_end };
    let project_path = history::real_or(start.cwd);
    let written = if history_end > 0 {
        let history = visible[..history_end].to_vec();
        let (key, cwd, project, tools, model, shape) = (start.key.to_string(), start.cwd.to_string(), project_path.clone(), start.tools.to_vec(), start.cli_model.to_string(), start.shape.clone());
        let ownership = start.ownership.as_ref().map(crate::runtime::Ownership::retain).transpose().map_err(|error| error.to_string())?;
        let file = timeout(SETTINGS.persistence, tokio::task::spawn_blocking(move || {
            let prints: Vec<String> = history.iter().map(fingerprint).collect();
            history::write_session(&key, &history, &prints, &cwd, &project, &tools, &model, &shape, ownership)
        })).await.map_err(|_| "Claude history resume timed out; pending work retains ownership until it exits".to_string())?
            .map_err(|error| error.to_string())?.map_err(|error| format!("Could not write Claude Code's session file: {error}"))?;
        Some(file)
    } else {
        None
    };
    // Writing a large resume file yields to the UI. A close or stop during that write must not
    // start a child after its session has already let go of the query.
    if turn.cancel.is_cancelled() { return Err("Operation aborted".into()); }
    if resumes_turn {
        debug("start: the turn starts at tool results; Claude Code answers it on resume");
    } else if continued {
        debug("start: context ends at an assistant message, opening with the continue note");
    }
    let mut args = base_args(start.thinking_off, start.effort);
    args.extend(written.iter().map(|file| format!("--resume={}", file.session_id())));
    args.extend(["--tools=", "--setting-sources=", "--permission-mode=bypassPermissions", "--include-partial-messages", "--session-mirror", "--model", start.cli_model,
        "--disable-slash-commands"].map(String::from));
    if SETTINGS.strict_mcp {
        args.push("--strict-mcp-config".into());
    }
    // Some models default to omitted thinking text; summarized makes thinking_delta events arrive.
    if start.effort.is_some() {
        args.extend(["--thinking-display", "summarized"].map(String::from));
    }
    // Fast mode (/fast) is one more flag setting, put in before the closing brace.
    let settings = if start.fast { PROVIDER_SETTINGS.replace('}', r#","fastMode":true}"#) } else { PROVIDER_SETTINGS.into() };
    args.extend(["--settings".into(), settings]);
    // A turn Claude Code answers on resume runs before the initialize request's prompt applies, so
    // without this it goes out with Claude Code's own system prompt: none of pi's, no AGENTS.md, a
    // prompt cache miss on everything, and the model told it may answer "No response requested."
    // (2.1.316 against a stand-in API). It goes in a file, since a prompt of 128 KiB or more can't
    // be one argument (Linux's MAX_ARG_STRLEN).
    if let Some(resume) = written.as_ref().filter(|_| resumes_turn) {
        let dir = history::session_path(resume.session_id(), &project_path).with_extension("");
        let file = dir.join(SYSTEM_PROMPT_FILE);
        if let Err(error) = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&file, start.system_prompt)) {
            return Err(format!("Could not write the prompt for Claude Code: {error}"));
        }
        args.extend(["--system-prompt-file".to_string(), file.to_string_lossy().into_owned()]);
    }
    let child = spawn(&args, start.cwd, resumes_turn)?;
    let mut opening = vec![initialize(start.system_prompt, start.tools)];
    if !resumes_turn {
        opening.push(json!({"type": "user", "message": {"role": "user", "content": prompt}, "parent_tool_use_id": null}));
    }
    let state = Arc::new(Mutex::new(State { shape: start.shape.clone(), prints: start.prints.to_vec(), pending: None, waiting: false, stopped: false,
        context: 0, manager: start.manager, kept_base: kept_base(&visible), kept: history::handle(start.key), ownership: start.ownership.clone(), fast: start.fast }));
    let (commands, inbox) = mpsc::unbounded_channel();
    let id = NEXT_QUERY.fetch_add(1, Ordering::Relaxed);
    let mut query = Query::new(id, start.key, start.model.clone(), start.tools, child, inbox, state.clone(), project_path, start.cwd, start.shape.clone(), written.map(history::ResumeFile::take));
    query.turn = Some(turn);
    lock(&LIVE).insert(history::query_key(start.key, start.shape), Live { id, state, commands });
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
    Settle,
    Lost,
    Abort,
    Lapsed,
}

// Everything a query's task holds. Per turn: the message being built (`out`) with, per content
// block, the stream index it came under (None once it stopped) and its tool-call JSON so far.
struct Query {
    id: u64,
    writer: Option<(u64, u32, u64)>,
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
    // The transcript mirror acknowledges these calls only after Claude Code has written them.
    saved_calls: HashSet<String>,
    // tools/call requests waiting for pi's results, and results that came before their request.
    parked: HashMap<String, Parked>,
    queued: HashMap<String, Value>,
    settle_at: Option<Instant>,
    last_activity: Instant,
    rejection: Option<Value>,
    failure: Option<String>,
    result_seen: bool,
    closed: bool,
    // An interrupt was sent and its turn hasn't ended yet; and the next message, held back until it
    // has, so nothing more of the stopped answer goes into the next one's turn.
    interrupting: bool,
    next_prompt: Option<Vec<Value>>,
    // A held-back message whose own turn was stopped too: it never reached Claude Code, so it goes
    // in with the next one.
    unsent: Vec<Value>,
    // Restart waits on these until the query can hand over, not merely until pi's turn ends.
    draining: Vec<oneshot::Sender<()>>,
    // When a waiting query's prompt cache lapses and it ends.
    hold_until: Option<Instant>,
    // Claude Code's running total of what this query cost (total_cost_usd, in dollars) at its last
    // result, and at this turn's, if it had one.
    cost_seen: f64,
    result_cost: Option<f64>,
    // This query's tool turns handed to pi with their cost pending since its last priced turn.
    pending_turns: u32,
    // The fast mode Claude Code last took: its flag setting at start, then each apply_flag_settings it answered.
    fast: bool,
}

impl Query {
    #[allow(clippy::too_many_arguments)]
    fn new(id: u64, key: &str, model: Model, tools: &[ToolDefinition], child: Child, inbox: mpsc::UnboundedReceiver<Command>, state: Arc<Mutex<State>>,
        project_path: String, cwd: &str, shape: Shape, written: Option<String>) -> Query {
        let (out, fast) = (new_output(&model, PROVIDER), lock(&state).fast);
        let writer = crate::proc::start(child.pid as i32).map(|start| (id, child.pid, start));
        Query { id, writer, key: key.to_string(), out, model, tools: tools.iter().map(|tool| tool.name.clone()).collect(), manifest: manifest(tools), child, inbox, state,
            project_path, cwd: cwd.to_string(), shape, written, cc_session: None, turn: None, open: Vec::new(), partial: Vec::new(), started: false,
            saw_stream_event: false, saw_tool_call: false, stream_id: None, stream_open: false, stream_delta: false, stream_start: 0, fallback_id: None,
            requests: Vec::new(), turn_requests: HashMap::new(), tool_ids: Vec::new(), handed: HashSet::new(), saved_calls: HashSet::new(), parked: HashMap::new(), queued: HashMap::new(),
            settle_at: None, last_activity: Instant::now(), rejection: None, failure: None, result_seen: false, closed: false, interrupting: false, next_prompt: None, unsent: Vec::new(), draining: Vec::new(), hold_until: None, cost_seen: 0.0, result_cost: None, pending_turns: 0, fast }
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
                line = self.child.next_line(None) => Wake::Line(line),
                command = self.inbox.recv() => Wake::Command(command),
                () = sleep_until(self.last_activity + BODY_IDLE_TIMEOUT), if streaming => Wake::Idle,
                () = sleep_until(self.settle_at.unwrap_or(far)), if self.settle_at.is_some() => Wake::Settle,
                () = sleep_until(lost.unwrap_or(far)), if lost.is_some() => Wake::Lost,
                () = cancel.cancelled(), if streaming => Wake::Abort,
                () = sleep_until(self.hold_until.unwrap_or(far)), if self.hold_until.is_some() => Wake::Lapsed,
            };
            match wake {
                Wake::Line(Some(line)) => self.line(&line).await,
                Wake::Line(None) => break,
                Wake::Command(Some(Command::Deliver(results, turn))) => self.deliver(results, turn).await,
                Wake::Command(Some(Command::Prompt(content, turn))) => self.prompt(content, turn).await,
                Wake::Command(Some(Command::Discard(reason))) => self.discard(&reason).await,
                Wake::Command(Some(Command::Hold)) => self.hold_parked(),
                Wake::Command(Some(Command::Fast(on))) => self.child.send(&json!({"type": "control_request", "request_id": format!("{FAST_REQUEST}{on}-{}", request_id()), "request": {"subtype": "apply_flag_settings", "settings": {"fastMode": on}}})).await,
                Wake::Command(Some(Command::Usage)) => self.child.send(&json!({"type": "control_request", "request_id": format!("{USAGE_REQUEST}{}", request_id()), "request": {"subtype": "get_usage"}})).await,
                Wake::Command(Some(Command::Drain(reply))) => self.draining.push(reply),
                // Taken: its Claude Code goes on in the next image, so nothing here may close it.
                Wake::Command(Some(Command::HandOver(reply))) => if let Some(handed) = self.hand_over().await && reply.send(handed).is_ok() {
                    self.forget("handed over to the next image");
                    QUERIES.fetch_sub(1, Ordering::SeqCst);
                    return std::mem::forget(self);
                },
                Wake::Command(None) => self.discard("the bridge let go of it").await,
                Wake::Idle => self.idle().await,
                Wake::Settle => self.end_turn_on_tool_use().await,
                Wake::Lost => self.check_lost().await,
                Wake::Abort => self.interrupt().await,
                Wake::Lapsed => self.discard("its prompt cache has lapsed").await,
            }
            if self.settled() {
                for reply in self.draining.drain(..) { let _ = reply.send(()); }
            }
        }
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
        self.saved_calls.clear();
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

    // Parking hands the calls to pi only after the transcript mirror acknowledged their write.
    // The stream can finish first; its mirror frame then completes this same boundary. Without
    // that acknowledgement, holdMax could close a brand-new query before its file even exists.
    async fn end_turn_on_tool_use(&mut self) {
        self.settle_at = None;
        if self.turn.is_none() || !self.tool_ids.iter().all(|id| self.saved_calls.contains(id)) {
            return;
        }
        self.out.stop_reason = StopReason::ToolUse;
        if !self.keep_records().await { return; }
        let Some(turn) = self.turn.take() else { return };
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

    async fn flush_settle(&mut self) {
        if self.settle_at.is_some() {
            self.end_turn_on_tool_use().await;
        }
    }

    // A runtime's stopped answer precedes Claude Code's interrupt result. Only the query knows
    // that result has arrived and that no partial output or mid-turn state would be lost: then it
    // has settled, and a restart may go. One waiting for a message is handed over; one parked on
    // tool calls closes, its records kept as it parked, so its next turn reads its cache.
    fn settled(&self) -> bool {
        self.turn.is_none() && !self.interrupting && self.unsent.is_empty() && self.child.stdout.buffer().is_empty() && self.child.line.is_empty()
    }

    #[cfg(unix)]
    fn ready_to_hand_over(&self) -> bool {
        lock(&self.state).waiting && self.settled()
    }


    // /restart: a ready query gives its Claude Code and copies of its pipes to the next image.
    #[cfg(unix)]
    async fn hand_over(&mut self) -> Option<Handed> {
        use std::os::fd::AsRawFd;
        if !self.ready_to_hand_over() {
            return None;
        }
        let fds = [self.child.stdin.as_ref()?.as_raw_fd(), self.child.stdout.get_ref().as_raw_fd(), self.child.stderr_fd.as_raw_fd()];
        // SAFETY: fcntl on descriptors this child holds open. The copies are the next image's.
        let fds = fds.map(|fd| unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) });
        // Its stderr reader stopped and gone, what it read goes in the record, and the rest is the
        // next image's to read.
        if let Some(reader) = self.child.stderr_reader.take() {
            reader.abort();
            let _ = reader.await;
        }
        // Still waiting: the session may have given it a message meanwhile (decide).
        let state = lock(&self.state);
        debug(&format!("hand over: session {}, Claude Code {}", self.key, self.child.pid));
        (!fds.contains(&-1) && state.waiting).then(|| Handed { pid: self.child.pid, start: crate::proc::start(self.child.pid as i32), fds, key: self.key.clone(), model: self.model.clone(), tools: self.tools.clone(),
            manifest: self.manifest.clone(), shape: state.shape.clone(), prints: state.prints.clone(), project_path: self.project_path.clone(),
            cwd: self.cwd.clone(), written: self.written.clone(), cc_session: self.cc_session.clone(),
            hold_ms: self.hold_until.map(|until| until.saturating_duration_since(Instant::now()).as_millis() as u64), cost_seen: self.cost_seen,
            pending_turns: self.pending_turns, requests: self.requests.clone(), stderr: lock(&self.child.stderr).clone(),
            cost_supplied: self.model.cost.supplied, fast: (state.fast, self.fast), notice: lock(&NOTES).remove(&self.key), manager: state.manager, session_mirror: true })
    }

    #[cfg(windows)]
    async fn hand_over(&mut self) -> Option<Handed> { None }

    // Out of LIVE, unless a later query of the session has taken its place; and if none has, `why` it
    // ended is kept for the next one's reason.
    fn forget(&self, why: &str) {
        let key = history::query_key(&self.key, &self.shape);
        let mut live = lock(&LIVE);
        if live.get(&key).is_some_and(|live| live.id == self.id) {
            live.remove(&key);
        }
        if !live.contains_key(&key) {
            lock(&ENDED).insert(key, why.to_string());
        }
    }

    // pi's run is over (end_run). A query parked on its calls was stopped while they ran: Claude Code
    // keeps such a turn, and pi keeps their results, which its next call carries (deliver). So it
    // waits for them, as an answered query waits for its next message, until its cache would lapse,
    // and counts among the waiting queries (make_room).
    fn hold_parked(&mut self) {
        {
            let mut state = lock(&self.state);
            if self.turn.is_some() || state.pending.is_none() {
                return;
            }
            (state.stopped, state.context) = (true, context(&self.requests));
        }
        self.hold_until = Some(Instant::now() + hold_time());
        debug(&format!("sync: the run is over, parked on its calls for {} s (session {})", hold_time().as_secs(), self.key));
        make_room();
    }

    // Stop the query: it holds a context pi no longer has, or something went wrong. The child is
    // interrupted and its input ended, so nothing it waits on gets answered and it starts no
    // further request; whatever waited on it is let go.
    async fn discard(&mut self, reason: &str) {
        debug(&format!("discard: session {}: {reason}", self.key));
        self.closed = true;
        self.forget(reason);
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
            let message = format!("Claude Code dispatched tool call {} [{id}] that piccosta never received; the query was stopped so the turn would not wait on it forever",
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

    // pi's abort: stop the answer and keep the query, as Claude Code does on Esc. pi gets the
    // stopped answer (and drops it from its context); Claude Code ends the turn and waits, and the
    // user's next message goes into this query (decide), so the next request is the last one's
    // context, unchanged, plus that message, and the prompt cache holds. Starting a new query here
    // instead rebuilt the whole conversation and wrote all of it to the cache again.
    async fn interrupt(&mut self) {
        debug(&format!("interrupt: session {}", self.key));
        // A message still held back behind an earlier interrupt never reached Claude Code. pi keeps
        // it, so it goes in with the next message (prompt).
        self.unsent.extend(self.next_prompt.take().into_iter().flatten());
        // It waits from now, not from the stopped turn's result, which can come after the next
        // message has already started a turn. And it's marked waiting before pi gets the stopped
        // answer, since pi's next call can come at once.
        (self.settle_at, self.interrupting, self.hold_until) = (None, true, Some(Instant::now() + hold_time()));
        {
            let mut state = lock(&self.state);
            state.pending = None;
            state.waiting = true;
            state.context = context(&self.requests);
        }
        make_room();
        if let Some(turn) = self.turn.take() {
            self.out.stop_reason = StopReason::Aborted;
            self.out.error_message.get_or_insert_with(|| "Operation aborted".to_string());
            self.hand_pending(turn);
        }
        self.child.send(&json!({"type": "control_request", "request_id": request_id(), "request": {"subtype": "interrupt"}})).await;
        // Calls Claude Code dispatched before the stop went with the stopped answer. Claude Code
        // cancels them as it stops; each is answered anyway, so none can be left waiting.
        for (id, call) in std::mem::take(&mut self.parked) {
            debug(&format!("interrupt: {} [{id}] of the stopped turn answered as not run (session {})", call.tool, self.key));
            self.answer(call.request_id, call.rpc_id, Err(STOPPED_CALL.into())).await;
        }
    }

    // The next message for a waiting query, as a new turn of it. Its stream opens now; the message
    // goes in once a stopped answer before it has ended (send_next).
    async fn prompt(&mut self, content: Vec<Value>, turn: Turn) {
        let content = std::mem::take(&mut self.unsent).into_iter().chain(content).collect();
        (self.turn, self.hold_until, self.next_prompt) = (Some(turn), None, Some(content));
        self.reset_turn();
        self.last_activity = Instant::now();
        self.send_next().await;
    }

    async fn send_next(&mut self) {
        if !self.interrupting
            && let Some(content) = self.next_prompt.take()
        {
            self.child.send(&json!({"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": null})).await;
        }
    }

    async fn deliver(&mut self, results: Vec<(String, Value)>, turn: Turn) {
        (self.turn, self.hold_until) = (Some(turn), None);
        self.reset_turn();
        self.last_activity = Instant::now();
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
        // A fast-mode answer: a refusal keeps the query and the mode Claude Code last took, and says why.
        let answer = &message["response"];
        if let Some(asked) = answer["request_id"].as_str().and_then(|id| id.strip_prefix(FAST_REQUEST)).filter(|_| message["type"] == "control_response") {
            if answer["subtype"] != "error" {
                self.fast = asked.starts_with("true");
            } else {
                let why = answer["error"].as_str().unwrap_or("no reason given");
                lock(&NOTES).insert(self.key.clone(), format!("Claude Code refused fast mode, which stays {}: {why}", if self.fast { "on" } else { "off" }));
            }
            return;
        }
        match message["type"].as_str().unwrap_or_default() {
            "control_request" => self.control(&message).await,
            // A usage answer, or its failure, which is no fault of the session's.
            "control_response" if message["response"]["request_id"].as_str().is_some_and(|id| id.starts_with(USAGE_REQUEST)) => note_usage(&message["response"]["response"]),
            "control_response" if message["response"]["subtype"] == "error" => {
                self.failure = Some(format!("Claude Code refused the session: {}", message["response"]["error"].as_str().unwrap_or("no reason given")));
            }
            "transcript_mirror" => {
                if self.cc_session.as_deref().is_some_and(|id| Path::new(message["filePath"].as_str().unwrap_or_default()) == history::session_path(id, &self.project_path)) {
                    self.saved_calls.extend(message["entries"].as_array().into_iter().flatten()
                        .flat_map(|entry| entry["message"]["content"].as_array().into_iter().flatten())
                        .filter(|block| block["type"] == "tool_use").filter_map(|block| block["id"].as_str().map(String::from)));
                    if self.saw_tool_call && !self.stream_open && self.settle_at.is_none() {
                        self.end_turn_on_tool_use().await;
                    }
                }
            }
            "control_response" | "control_cancel_request" | "keep_alive" => {}
            _ => {
                self.message(&message).await;
                self.send_next().await;
            }
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
                        // A call while a stopped turn ends is that turn's, which pi dropped: sent before
                        // the stop reached Claude Code (which then cancels it), or Claude Code went on
                        // past the stop (2.1.319 skips one read before its next turn starts, per its
                        // code). It's stopped again, now that it runs, and the call answered at once.
                        None if self.interrupting => {
                            debug(&format!("interrupt: {name} [{id}] came for the stopped turn, stopped again and answered as not run (session {})", self.key));
                            self.child.send(&json!({"type": "control_request", "request_id": self::request_id(), "request": {"subtype": "interrupt"}})).await;
                            Err(STOPPED_CALL.into())
                        }
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
    async fn message(&mut self, message: &Value) {
        let kind = message["type"].as_str().unwrap_or_default();
        let error = if kind == "result" { result_error(message) } else { None };
        // A non-streamed answer still settling ends when Claude Code moves on, unless it failed.
        if kind == "result" && error.is_some() {
            self.settle_at = None;
        } else if matches!(kind, "stream_event" | "user" | "result") {
            self.flush_settle().await;
            if self.failure.is_some() { return; }
        }
        // The stopped turn's end, after an interrupt: the query stays, waiting for the next message
        // (or already taking it).
        if kind == "result" && self.interrupting {
            self.interrupting = false;
            return;
        }
        if kind == "result" {
            self.result_seen = true;
            // A turn that answered leaves the query waiting (hold, below). Anything else ends the
            // input, and Claude Code exits once the last turn is done.
            if error.is_some() || !self.holds() {
                self.child.stdin = None;
            }
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
            "stream_event" => self.stream_event(&message["event"]).await,
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
        if kind == "result" && self.child.stdin.is_some() {
            self.hold().await;
        }
    }

    // Whether the turn ending now leaves the query waiting: holding is on, and pi's turn ended with
    // an answer, not on tool calls it was handed.
    fn holds(&self) -> bool {
        !hold_time().is_zero() && self.turn.is_some() && matches!(self.out.stop_reason, StopReason::Stop | StopReason::Length)
            && lock(&self.state).pending.is_none()
    }

    // A turn that answered: its records are kept for a later rewrite, the query waits for the next
    // message (decide) until its prompt cache lapses, and pi gets the answer. The next turn then
    // goes out as Claude Code's own continuation, the bytes the cache already holds.
    async fn hold(&mut self) {
        if !self.keep_records().await { return; }
        {
            let mut state = lock(&self.state);
            state.prints.push(fingerprint(&Message::Assistant(self.out.clone())));
            state.waiting = true;
            state.context = context(&self.requests);
        }
        make_room();
        self.hold_until = Some(Instant::now() + hold_time());
        debug(&format!("sync: waiting for the next message for {} s (session {})", hold_time().as_secs(), self.key));
        self.reply();
    }

    // Keep the prefix before handing back an answer or parking on calls. A holdMax eviction can
    // happen before the tools finish, so their results must not be needed to save this prefix.
    async fn keep_records(&mut self) -> bool {
        let (mut prints, mut roles) = lock(&self.state).kept_base.clone();
        let kept = lock(&self.state).kept.clone();
        let ownership = lock(&self.state).ownership.clone();
        if let Some(cc_session) = self.cc_session.clone().filter(|_| SETTINGS.keep && matches!(self.out.stop_reason, StopReason::Stop | StopReason::Length | StopReason::ToolUse)) {
            prints.push(fingerprint(&Message::Assistant(self.out.clone())));
            roles.push("assistant".into());
            let (key, project, cwd, shape) = (self.key.clone(), self.project_path.clone(), self.cwd.clone(), self.shape.clone());
            // Await the save, not merely its scheduling: holdMax or /restart may close the query as
            // soon as pi gets these calls, and that must never lose the prefix it just acknowledged.
            let ownership = match ownership.as_ref().map(crate::runtime::Ownership::retain).transpose() {
                Ok(ownership) => ownership,
                Err(error) => { self.failure = Some(error.to_string()); return false; }
            };
            let writer = self.writer;
            let fenced = kept.clone();
            let saved = timeout(SETTINGS.persistence, tokio::task::spawn_blocking(move || {
                // This strong lease belongs to the worker, not its cancelled JoinHandle or LocalSet.
                let _ownership = ownership;
                let kept = kept.ok_or_else(|| std::io::Error::other("the Claude history session was closed"))?;
                history::keep(kept, &key, &cc_session, &project, &cwd, &shape, prints, roles, writer)
            })).await;
            let error = match saved {
                Ok(Ok(Ok(()))) => None,
                Ok(Ok(Err(error))) => Some(error.to_string()),
                Ok(Err(error)) => Some(error.to_string()),
                Err(_) => {
                    if let Some(cache) = fenced { history::fence(&cache); }
                    Some("Claude history save timed out; pending save retains ownership until it exits".into())
                }
            };
            if let Some(error) = error {
                self.failure = Some(format!("Could not keep Claude Code's records: {error}"));
                return false;
            }
        }
        true
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

    async fn stream_event(&mut self, event: &Value) {
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
                        let index = self.push(Content::ToolCall { external: false, id: id.clone(), name: name.clone(), arguments, namespace: None, arguments_error: None, thought_signature: None }, stream_index);
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
                    self.end_turn_on_tool_use().await;
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
                    let index = self.push(Content::ToolCall { external: false, id: call_id.clone(), name: name.clone(), arguments, namespace: None, arguments_error: None, thought_signature: None }, None);
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
            self.flush_settle().await;
        }
        let status = self.child.close().await;
        let stderr = self.child.stderr_tail().await;
        if !stderr.is_empty() { debug(&format!("Claude Code's stderr: {stderr}")); }
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
            if self.keep_records().await {
                self.reply();
            } else {
                self.end_open(StopReason::Error, &self.failure.clone().unwrap_or_default());
            }
        }
        let ids: HashSet<&String> = self.written.iter().chain(&self.cc_session).collect();
        for id in ids {
            history::delete_session(id, &self.project_path);
        }
        debug(&format!("query done: session {} closed={} failure={:?}", self.key, self.closed, self.failure));
        // A discarded one was let go of with its reason already.
        if !self.closed {
            self.forget(self.failure.as_deref().unwrap_or("Claude Code ended after its turn"));
        }
    }
}
