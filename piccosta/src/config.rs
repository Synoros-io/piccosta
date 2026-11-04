// Where piccosta keeps its files, and the settings that steer it.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/coding-agent/src/core/settings-manager.ts
// and settings-defaults.ts (the defaults, and project settings merged over global ones).
//
// The agent dir holds settings.json, models.json, login/ and sessions/. It's the same dir the TS
// piccosta uses, so sessions and settings carry over. Settings stay JSON objects rather than a
// fixed struct: unknown keys (pi-subagents' `subagents`, anything newer) must survive, and each
// accessor below names the key it reads and the default pi uses when it's absent.

use serde_json::{Map, Value};
use std::fs;
use std::path::{Path, PathBuf};

// The folder in a project that holds its own settings. pi's own `.pi/` is never read: its
// settings are pi's, and a repo that also uses pi would hand them to piccosta.
pub const PROJECT_CONFIG_DIR: &str = ".piccosta";

// Defaults from settings-defaults.ts, used when neither settings file sets the key.
const DEFAULT_COMPACTION_RESERVE_TOKENS: u64 = 16_384;
const DEFAULT_COMPACTION_KEEP_RECENT_TOKENS: u64 = 20_000;
const DEFAULT_RETRY_MAX_RETRIES: u32 = 3;
const DEFAULT_RETRY_BASE_DELAY_MS: u64 = 2_000;
// The longest wait between retries (ai/src/utils/retry.ts:140).
const DEFAULT_RETRY_MAX_DELAY_MS: u64 = 60_000;
const DEFAULT_THINKING_LEVEL: &str = "medium";
const DEFAULT_TRANSPORT: &str = "auto";
// The settings key holding each model's last thinking level, by provider/id: pi's own key
// (settings-manager.ts setModelThinkingLevel), so pi and piccosta share one store.
const MODEL_THINKING_LEVELS: &str = "modelThinkingLevels";

// The agent dir: PICCOSTA_AGENT_DIR, else piccosta/agent under data_home (XDG on Unix,
// roaming AppData on Windows). The environment overrides keep existing installations movable.
pub fn agent_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("PICCOSTA_AGENT_DIR") {
        return PathBuf::from(dir);
    }
    data_home().join("piccosta").join("agent")
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_default()
}

// XDG remains an override everywhere. macOS uses XDG too; Windows defaults to roaming AppData.
pub fn data_home() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| {
        #[cfg(windows)]
        { std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(|| home().join("AppData/Roaming")) }
        #[cfg(unix)]
        { home().join(".local/share") }
    })
}

// Settings from the agent dir's settings.json, with a project's .piccosta/settings.json merged over it.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    merged: Map<String, Value>,
    // A file that exists but couldn't be read as a JSON object, with why. Reported, not fatal:
    // pi starts with that scope empty, and so does this.
    pub load_errors: Vec<String>,
}

// Merge `overlay` into `base`: objects merge key by key at every depth, anything else replaces.
fn deep_merge(base: &mut Map<String, Value>, overlay: &Map<String, Value>) {
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(Value::Object(existing)), Value::Object(update)) => deep_merge(existing, update),
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

// One JSON file (settings, a login). Output: its object (empty when the file is missing), or an
// error message when it exists but isn't a JSON object: a login file that's there but doesn't
// read is a failure, not a missing login.
pub(crate) fn read_json_object(path: &Path) -> Result<Map<String, Value>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Map::new()),
        Err(error) => return Err(format!("{}: {error}", path.display())),
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(object)) => Ok(object),
        Ok(_) => Err(format!("{}: not a JSON object", path.display())),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

// The thinking level `model` (provider/id) was last set to, from the agent dir's settings.json. Read
// fresh each time: another session, or another piccosta (Zed's, T3 Code's), may have set it since.
pub fn remembered_thinking_level(agent_dir: &Path, model: &str) -> Option<String> {
    Settings::load(agent_dir, None).get(MODEL_THINKING_LEVELS)?.get(model)?.as_str().map(str::to_string)
}

// Change the agent dir's settings.json: `edit` gets its object, and the file is written back only
// when that changed it. Everything else stays as it was, in its order; a file that isn't a JSON
// object is left alone. The file is written whole beside it and renamed over it, so nothing ever
// reads half a file. Another piccosta writing at the same moment can lose one of the two edits.
fn edit_settings(agent_dir: &Path, edit: impl FnOnce(&mut Map<String, Value>)) -> Result<(), String> {
    let path = agent_dir.join("settings.json");
    let failed = |error: &dyn std::fmt::Display| format!("could not save {}: {error}", path.display());
    let mut settings = read_json_object(&path)?;
    let before = settings.clone();
    edit(&mut settings);
    if settings == before {
        return Ok(());
    }
    let text = serde_json::to_string_pretty(&settings).map_err(|error| failed(&error))?;
    let partial = path.with_extension("json.partial");
    fs::create_dir_all(agent_dir).and_then(|()| fs::write(&partial, text)).and_then(|()| fs::rename(&partial, &path)).map_err(|error| failed(&error))
}

// Save the thinking level a model (provider/id) was set to by hand, so its sessions start there;
// None forgets it, and the default level applies again.
pub fn remember_thinking_level(agent_dir: &Path, model: &str, level: Option<&str>) -> Result<(), String> {
    edit_settings(agent_dir, |settings| {
        let levels = settings.entry(MODEL_THINKING_LEVELS).or_insert_with(|| Value::Object(Map::new()));
        if let Some(levels) = levels.as_object_mut() {
            match level {
                Some(level) => levels.insert(model.to_string(), level.into()),
                None => levels.remove(model),
            };
        }
    })
}

// Save the model picked by hand as the one new sessions start on (`defaultProvider` and
// `defaultModel`, as pi's "default model" choice and Claude Code's /model save it).
pub fn remember_model(agent_dir: &Path, provider: &str, model_id: &str) -> Result<(), String> {
    edit_settings(agent_dir, |settings| {
        settings.insert("defaultProvider".into(), provider.into());
        settings.insert("defaultModel".into(), model_id.into());
    })
}

impl Settings {
    // The agent dir's settings.json, and the project's in `cwd` merged over it. Without a cwd it's
    // the install's alone, for what belongs to the user rather than a project, and is the same for
    // every project one process serves: the model lists, the Claude bridge, the client name, the
    // models' thinking levels.
    pub fn load(agent_dir: &Path, cwd: Option<&Path>) -> Settings {
        let mut settings = Settings::default();
        let project = cwd.map(|cwd| cwd.join(PROJECT_CONFIG_DIR).join("settings.json"));
        for path in std::iter::once(agent_dir.join("settings.json")).chain(project) {
            match read_json_object(&path) {
                Ok(object) => deep_merge(&mut settings.merged, &object),
                Err(error) => settings.load_errors.push(error),
            }
        }
        settings
    }

    // A value by dotted path, e.g. "retry.maxRetries".
    pub fn get(&self, dotted: &str) -> Option<&Value> {
        let mut parts = dotted.split('.');
        let mut current = self.merged.get(parts.next()?)?;
        for part in parts {
            current = current.get(part)?;
        }
        Some(current)
    }

    fn get_str(&self, dotted: &str) -> Option<&str> {
        self.get(dotted).and_then(Value::as_str)
    }

    fn get_u64(&self, dotted: &str) -> Option<u64> {
        self.get(dotted).and_then(Value::as_u64)
    }

    fn get_bool(&self, dotted: &str) -> Option<bool> {
        self.get(dotted).and_then(Value::as_bool)
    }

    // `defaultProvider` and `defaultModel`, the model a session starts on.
    pub fn default_model(&self) -> Option<String> {
        match (self.get_str("defaultProvider"), self.get_str("defaultModel")) {
            (Some(provider), Some(model)) => Some(format!("{provider}/{model}")),
            (None, Some(model)) => Some(model.to_string()),
            _ => None,
        }
    }

    // `defaultThinkingLevel`, "medium" when unset.
    pub fn default_thinking_level(&self) -> String {
        self.get_str("defaultThinkingLevel").unwrap_or(DEFAULT_THINKING_LEVEL).to_string()
    }

    // `shellCommandPrefix`: a line run ahead of every bash command, in the same shell.
    pub fn shell_command_prefix(&self) -> Option<String> {
        self.get_str("shellCommandPrefix").map(str::to_string)
    }

    // `shellPath`: the shell bash commands run in, when it isn't /bin/bash.
    pub fn shell_path(&self) -> Option<String> {
        self.get_str("shellPath").map(str::to_string)
    }

    // `retry.*`: whether failed model requests are retried, how often, and the backoff.
    pub fn retry(&self) -> RetrySettings {
        RetrySettings {
            enabled: self.get_bool("retry.enabled").unwrap_or(true),
            max_retries: self.get_u64("retry.maxRetries").map(|n| n as u32).unwrap_or(DEFAULT_RETRY_MAX_RETRIES),
            base_delay_ms: self.get_u64("retry.baseDelayMs").unwrap_or(DEFAULT_RETRY_BASE_DELAY_MS),
            max_delay_ms: self.get_u64("retry.maxAgentDelayMs").unwrap_or(DEFAULT_RETRY_MAX_DELAY_MS),
        }
    }

    // `transport`: how Codex requests travel, "auto" by default (settings-defaults.ts:5). An older
    // settings file's `websockets` boolean means "websocket" or "sse" (settings-manager.ts:404-407).
    pub fn transport(&self) -> String {
        if let Some(transport) = self.get_str("transport") {
            return transport.to_string();
        }
        match self.get_bool("websockets") {
            Some(true) => "websocket".to_string(),
            Some(false) => "sse".to_string(),
            None => DEFAULT_TRANSPORT.to_string(),
        }
    }

    // `compaction.*`: when the context gets summarized, and how much recent history is kept whole,
    // for one model. `compaction.modelOverrides["provider/id"]` beats the ordinary keys
    // (settings-manager.ts:833-878). It's looked up by hand: model ids have dots in them.
    pub fn compaction(&self, provider: &str, model_id: &str) -> CompactionSettings {
        let model_key = format!("{provider}/{model_id}");
        let token_setting = |field: &str, default: u64| {
            self.get("compaction.modelOverrides")
                .and_then(|overrides| overrides.get(&model_key))
                .and_then(|entry| entry.get(field))
                .and_then(Value::as_u64)
                .or_else(|| self.get_u64(&format!("compaction.{field}")))
                .unwrap_or(default)
        };
        CompactionSettings {
            enabled: self.get_bool("compaction.enabled").unwrap_or(true),
            reserve_tokens: token_setting("reserveTokens", DEFAULT_COMPACTION_RESERVE_TOKENS),
            keep_recent_tokens: token_setting("keepRecentTokens", DEFAULT_COMPACTION_KEEP_RECENT_TOKENS),
            compact_at: Some(token_setting("compactAt", 0)).filter(|tokens| *tokens > 0).or_else(|| crate::scale::compact_at(&agent_dir(), &model_key)),
        }
    }
}

impl Settings {
    // `web.*`: the web tools' endpoints and limits. The keys are pi-web-access's web-search.json
    // keys, moved under "web" so piccosta has one settings file. Defaults are pi-web-access's
    // except the User-Agent (see web/net.rs).
    pub fn web(&self) -> WebSettings {
        let strings = |dotted: &str| -> Vec<String> {
            self.get(dotted)
                .and_then(Value::as_array)
                .map(|items| items.iter().filter_map(Value::as_str).map(|item| item.trim().to_string()).filter(|item| !item.is_empty()).collect())
                .unwrap_or_default()
        };
        // fetch.timeout is in seconds, as pi-web-access takes it (extract.ts:37-67).
        let fetch_timeout_ms = self
            .get("web.fetch.timeout")
            .and_then(Value::as_f64)
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .map(|seconds| ((seconds * 1000.0).ceil() as u64).max(1))
            .unwrap_or(DEFAULT_WEB_FETCH_TIMEOUT_MS);
        // maxInlineContentChars below the floor falls back to the default; above the ceiling it's
        // capped (web-tool-core.ts:295-301).
        let max_inline_content_chars = match self.get_u64("web.maxInlineContentChars") {
            Some(chars) if chars >= MIN_WEB_INLINE_CONTENT_CHARS => chars.min(MAX_WEB_INLINE_CONTENT_CHARS),
            _ => DEFAULT_WEB_INLINE_CONTENT_CHARS,
        } as usize;
        let searxng_url = std::env::var("SEARXNG_BASE_URL").ok().or_else(|| self.get_str("web.searxngBaseUrl").map(str::to_string));
        WebSettings {
            searxng_url: searxng_url.map(|url| url.trim().trim_end_matches('/').to_string()).filter(|url| !url.is_empty()),
            search_model: self.get_str("web.searchModel").map(str::trim).filter(|name| !name.is_empty()).map(str::to_string),
            max_inline_content_chars,
            fetch_timeout_ms,
            user_agent: self
                .get_str("web.fetchContent.userAgent")
                .map(str::trim)
                .filter(|agent| !agent.is_empty() && agent.chars().all(|c| c == '\t' || (' '..='~').contains(&c)))
                .map(str::to_string),
            allow_ranges: strings("web.ssrf.allowRanges"),
            domain_allow: strings("web.fetchContent.domainPolicy.allow"),
            domain_deny: strings("web.fetchContent.domainPolicy.deny"),
            max_response_size_mb: self.get_u64("web.fetch.maxSizeMB").filter(|mb| *mb > 0).unwrap_or(DEFAULT_WEB_MAX_RESPONSE_MB),
            pdf_max_size_mb: self.get_u64("web.pdf.maxSizeMB").filter(|mb| *mb > 0).unwrap_or(DEFAULT_WEB_PDF_MAX_SIZE_MB),
            pdf_max_pages: self.get_u64("web.pdf.maxPages").filter(|pages| *pages > 0).unwrap_or(DEFAULT_WEB_PDF_MAX_PAGES) as usize,
            github_clone_enabled: self.get_bool("web.githubClone.enabled").unwrap_or(true),
            github_max_repo_size_mb: self.get("web.githubClone.maxRepoSizeMB").and_then(Value::as_f64).filter(|mb| *mb > 0.0).unwrap_or(DEFAULT_GITHUB_MAX_REPO_SIZE_MB),
            github_clone_timeout_s: self.get("web.githubClone.cloneTimeoutSeconds").and_then(Value::as_f64).filter(|s| *s > 0.0).unwrap_or(DEFAULT_GITHUB_CLONE_TIMEOUT_S),
            github_clone_path: self.get_str("web.githubClone.clonePath").map(str::trim).filter(|path| !path.is_empty()).map(PathBuf::from),
        }
    }
}

// pi-web-access's defaults: extract.ts:32 (timeout), web-tool-core.ts:291-293 (inline content),
// extract.ts:1339 (any response but a PDF, `?? 5`), pdf-extract.ts:67-70 (PDF size and pages).
const DEFAULT_WEB_FETCH_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_WEB_MAX_RESPONSE_MB: u64 = 5;
const DEFAULT_WEB_INLINE_CONTENT_CHARS: u64 = 30_000;
const MIN_WEB_INLINE_CONTENT_CHARS: u64 = 1_000;
const MAX_WEB_INLINE_CONTENT_CHARS: u64 = 200_000;
const DEFAULT_WEB_PDF_MAX_SIZE_MB: u64 = 20;
const DEFAULT_WEB_PDF_MAX_PAGES: u64 = 100;
// github-extract.ts:112-116. The clone path's default isn't pi-web-access's /tmp/pi-github-repos:
// it's <agent dir>/github-clones (see web/github.rs).
const DEFAULT_GITHUB_MAX_REPO_SIZE_MB: f64 = 350.0;
const DEFAULT_GITHUB_CLONE_TIMEOUT_S: f64 = 30.0;

#[derive(Debug, Clone, Default)]
pub struct WebSettings {
    // The SearXNG instance web_search asks: $SEARXNG_BASE_URL, else `web.searxngBaseUrl`.
    pub searxng_url: Option<String>,
    // `web.searchModel`, the provider/model web_search asks when there's no SearXNG; None means
    // openai_search.rs's defaults.
    pub search_model: Option<String>,
    pub max_inline_content_chars: usize,
    pub fetch_timeout_ms: u64,
    // None means net.rs's own default.
    pub user_agent: Option<String>,
    // CIDR ranges the private-address check lets through (a fake-IP proxy's 198.18/15).
    pub allow_ranges: Vec<String>,
    pub domain_allow: Vec<String>,
    pub domain_deny: Vec<String>,
    // `web.fetch.maxSizeMB`: the largest response fetched that isn't a PDF.
    pub max_response_size_mb: u64,
    pub pdf_max_size_mb: u64,
    pub pdf_max_pages: usize,
    // `web.githubClone.*`: GitHub repo URLs as shallow clones.
    pub github_clone_enabled: bool,
    pub github_max_repo_size_mb: f64,
    pub github_clone_timeout_s: f64,
    pub github_clone_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
pub struct RetrySettings {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
    // Compact once the context passes this many tokens, before the window would make it (the
    // model scale's compactAt, or compaction.compactAt); None leaves it to the window.
    pub compact_at: Option<u64>,
}
