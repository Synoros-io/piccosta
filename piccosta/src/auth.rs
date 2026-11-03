// Credentials: where each provider's key or token comes from, and keeping the Codex login fresh.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): the auth.json format and lock of
// packages/coding-agent/src/core/auth-storage.ts, the refresh rules of packages/ai/src/auth/resolve.ts,
// and the token refresh of packages/ai/src/auth/oauth/openai-codex.ts.
//
// A request asks for a provider's credential, and the sources are tried in this order:
// 1. piccosta's own login store, <login dir>/auth.json, written by the TypeScript piccosta's
//    `piccosta auth login` (this one has no login command yet). A Codex
//    login there is refreshed here, in-process, when it nears expiry.
// 2. The tool that owns the account (ACCOUNTS): the Codex CLI's token for openai-codex
//    ($CODEX_HOME/auth.json, read only, never refreshed or written here, since that file is the
//    Codex CLI's), and opencode's key file for opencode-go.
// 3. models.json's `apiKey` for the provider: a literal key, `$VAR`, or `!command`.
// 4. The provider's environment variable, e.g. OPENCODE_API_KEY.
// Claude has none of these: Claude Code keeps its own login, which is checked by asking it.
// pi gets the first of these by starting a second pi through a models.json `!` command on every
// request (about 0.2 s each); here it's a file read, and a refresh only when one is due.

use crate::clock;
use crate::config::read_json_object;
use crate::model::ProviderConfig;
use base64::Engine;
use serde_json::Value;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

// OpenAI's OAuth client for Codex, and its token endpoint (openai-codex.ts:22-24). A login in
// piccosta's store that names its own `clientId` or `tokenUrl` is refreshed with those instead.
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
// The JWT claim holding the ChatGPT account id that Codex requests send (openai-codex.ts:35).
const CODEX_JWT_CLAIM: &str = "https://api.openai.com/auth";
// Refresh a login with less than this left, so a request never starts on a token about to lapse.
// pi refreshes at 5 minutes (resolve.ts) and piccosta's launcher asked for 15 (--min-expiry 900s).
const REFRESH_MARGIN_MS: i64 = 15 * 60 * 1000;
// How long a refresh may take before it's abandoned (resolve.ts: 15 s, regardless of the caller).
const REFRESH_TIMEOUT: Duration = Duration::from_secs(15);
// The auth.json lock, compatible with pi's proper-lockfile: a directory named <file>.lock,
// considered abandoned when its mtime is older than 30 s, waited for with backoff from 10 ms
// doubling to 1 s, up to 30 s in all (auth-storage.ts:118-198).
const LOCK_STALE: Duration = Duration::from_secs(30);
const LOCK_FIRST_WAIT: Duration = Duration::from_millis(10);
const LOCK_LONGEST_WAIT: Duration = Duration::from_secs(1);
const LOCK_DEADLINE: Duration = Duration::from_secs(30);
// How an error begins when a provider has no login here at all, as against one that couldn't be
// read or refreshed just now (offline, a timeout, a failing command): model.rs drops a provider's
// list for the first only.
pub const NO_CREDENTIAL: &str = "No credential for";

// The tool that owns an account's login.
#[derive(PartialEq)]
pub enum Owner {
    CodexCli,
    Opencode,
    ClaudeCode,
}

// The accounts piccosta logs into, one row each, which everything that names a provider reads:
// the provider id; the plan its use counts against, as usage.json, /scale and `auth status` call
// it; the plan as the user knows it; whether a manager asks before spending it on its own (the
// two paid plans); the tool that owns the login and how to log in with it; and the environment
// variable its key can come from (ai/src/env-api-keys.ts).
pub struct Account {
    pub provider: &'static str,
    pub plan: &'static str,
    pub plan_name: &'static str,
    pub ask_first: bool,
    pub owner: Owner,
    pub login: &'static str,
    pub env_key: Option<&'static str>,
}

pub const ACCOUNTS: [Account; 3] = [
    Account { provider: "openai-codex", plan: "codex", plan_name: "ChatGPT", ask_first: true, owner: Owner::CodexCli, login: "codex login", env_key: None },
    Account { provider: "opencode-go", plan: "opencode-go", plan_name: "opencode Go", ask_first: false, owner: Owner::Opencode, login: "opencode auth login", env_key: Some("OPENCODE_API_KEY") },
    Account { provider: crate::provider::claude::PROVIDER, plan: "claude", plan_name: "Claude", ask_first: true, owner: Owner::ClaudeCode, login: "claude auth login", env_key: None },
];

// A provider's row, when it's one of the accounts piccosta knows.
pub fn account(provider: &str) -> Option<&'static Account> {
    ACCOUNTS.iter().find(|account| account.provider == provider)
}

// Claude Code's login, from the bridge's own once-per-process `claude auth status` (bounded by its
// deadline, the child killed and reaped). Output: how it's logged in, or why Claude can't run: not
// installed or explicitly logged out (no login), or no answer (a failure, which keeps the list).
async fn claude_login() -> Result<Credential, String> {
    let settings = &*crate::provider::claude::SETTINGS;
    let provider = crate::provider::claude::PROVIDER;
    let status = tokio::task::spawn_blocking(|| crate::provider::claude::AUTH.clone()).await.ok().flatten();
    let Some(status) = status else {
        return Err(match installed(&settings.claude) {
            true => format!("Claude Code ({}) didn't say whether it's logged in", settings.claude),
            false => format!("{NO_CREDENTIAL} \"{provider}\": Claude Code ({}) isn't installed", settings.claude),
        });
    };
    // Only an explicit false is a logout; an answer without it ({}, an error) is a failure.
    if status["loggedIn"] == false {
        return Err(format!("{NO_CREDENTIAL} \"{provider}\": Claude Code ({}) isn't logged in; run: claude auth login", settings.claude));
    }
    if status["loggedIn"] != true {
        return Err(format!("Claude Code ({}) didn't say whether it's logged in: {status}", settings.claude));
    }
    let method = status["authMethod"].as_str().map_or(String::new(), |method| format!(" ({method})"));
    Ok(Credential { key: "not-used".into(), account_id: None, source: format!("Claude Code's own login{method}") })
}

// Whether a command name or path is a file that exists: a path as given, a bare name on PATH.
fn installed(command: &str) -> bool {
    match command.contains('/') {
        true => std::path::Path::new(command).is_file(),
        false => std::env::var_os("PATH").is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(command).is_file())),
    }
}

// A credential ready for a request: the bearer key, and for Codex the account it belongs to.
#[derive(Debug, Clone)]
pub struct Credential {
    pub key: String,
    pub account_id: Option<String>,
    // Where it came from, for `piccosta auth status` and error messages.
    pub source: String,
}

// Where the credential files live. Each path comes from the environment, with the default noted.
#[derive(Debug, Clone)]
pub struct CredentialPaths {
    // <agent dir>/login, or $PICCOSTA_LOGIN_DIR.
    pub login_dir: PathBuf,
    // $CODEX_HOME, default ~/.codex.
    pub codex_home: PathBuf,
    // $XDG_DATA_HOME/opencode, default ~/.local/share/opencode.
    pub opencode_dir: PathBuf,
}

impl CredentialPaths {
    // The paths for an agent dir, read from the environment.
    pub fn from_environment(agent_dir: &Path) -> CredentialPaths {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        CredentialPaths {
            login_dir: std::env::var_os("PICCOSTA_LOGIN_DIR").map(PathBuf::from).unwrap_or_else(|| agent_dir.join("login")),
            codex_home: std::env::var_os("CODEX_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".codex")),
            opencode_dir: crate::config::data_home().join("opencode"),
        }
    }
}

// A JWT's claims, or None. The payload is base64url; pi decodes it with atob, which rejects
// base64url's `-` and `_`, so both alphabets are accepted here.
fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?.trim_end_matches('=');
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload).or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload));
    serde_json::from_slice(&decoded.ok()?).ok()
}

// The account id inside a Codex access token: its chatgpt_account_id claim, or None.
fn codex_account_id(access_token: &str) -> Option<String> {
    jwt_claims(access_token)?.get(CODEX_JWT_CLAIM)?.get("chatgpt_account_id")?.as_str().filter(|id| !id.is_empty()).map(str::to_string)
}

// Run a models.json `!command` and take its output as the key. Input: the command after the `!`.
// Output: stdout, trimmed, or an error with the command's stderr. It runs through sh so the
// variable expansions in piccosta's default models.json work as they do in pi.
async fn run_key_command(command: &str) -> Result<String, String> {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .map_err(|error| format!("could not run apiKey command: {error}"))?;
    let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !output.status.success() || key.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("apiKey command failed: {}", stderr.trim()));
    }
    Ok(key)
}

// A models.json apiKey value made into a key. Input: the configured value. Output: the key, from
// the environment for `$VAR`, from a command for `!command`, otherwise the value itself.
async fn resolve_configured_key(value: &str) -> Result<String, String> {
    if let Some(command) = value.strip_prefix('!') {
        return run_key_command(command).await;
    }
    if let Some(name) = value.strip_prefix('$') {
        return std::env::var(name).map_err(|_| format!("environment variable {name} is not set"));
    }
    Ok(value.to_string())
}

// The credential for a provider. Input: the provider id, its models.json settings, and the paths.
// Output: the first credential the sources above give, or an error saying what was tried. A key
// that's there and empty (an apiKey of "", an empty variable or stored key) is an error: only a
// models.json server with no apiKey at all is asked with no key.
pub async fn credential_for(provider: &str, config: Option<&ProviderConfig>, paths: &CredentialPaths) -> Result<Credential, String> {
    let credential = first_credential(provider, config, paths).await?;
    if credential.key.is_empty() && credential.source != NO_KEY_NEEDED {
        return Err(format!("the key for \"{provider}\" from {} is empty", credential.source));
    }
    Ok(credential)
}

// What a keyless server's credential says it came from.
const NO_KEY_NEEDED: &str = "none needed (models.json has its baseUrl and no apiKey)";

async fn first_credential(provider: &str, config: Option<&ProviderConfig>, paths: &CredentialPaths) -> Result<Credential, String> {
    let account = account(provider);
    let owner = account.map(|account| &account.owner);
    // Claude Code keeps its own login; the bridge never sees a key, but the login has to be there.
    if owner == Some(&Owner::ClaudeCode) {
        return claude_login().await;
    }
    let login_file = paths.login_dir.join("auth.json");

    // 1. piccosta's own login.
    if let Some(stored) = read_json_object(&login_file)?.get(provider) {
        match stored.get("type").and_then(Value::as_str) {
            Some("oauth") if owner == Some(&Owner::CodexCli) => return fresh_codex_login(&login_file).await,
            Some("api_key") => {
                if let Some(key) = stored.get("key").and_then(Value::as_str) {
                    let key = resolve_configured_key(key).await?;
                    return Ok(Credential { key, account_id: None, source: format!("{}", login_file.display()) });
                }
            }
            _ => {}
        }
    }

    // 2. The account owner's own file.
    if owner == Some(&Owner::CodexCli) {
        let codex_file = paths.codex_home.join("auth.json");
        let codex = read_json_object(&codex_file)?;
        let tokens = codex.get("tokens");
        if let Some(access) = tokens.and_then(|tokens| tokens.get("access_token")).and_then(Value::as_str) {
            // The token's `exp` claim (epoch seconds): piccosta can't refresh this login, so an
            // expired one fails here, saying how to fix it, rather than as a bare 401 from Codex.
            let expires = jwt_claims(access).and_then(|claims| claims.get("exp")?.as_f64());
            if expires.is_some_and(|exp| exp * 1000.0 <= crate::clock::now_ms() as f64) {
                return Err(format!("the Codex CLI's login expired; run `codex` once ({})", codex_file.display()));
            }
            let account_id = tokens
                .and_then(|tokens| tokens.get("account_id"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| codex_account_id(access));
            return Ok(Credential {
                key: access.to_string(),
                account_id,
                source: format!("the Codex CLI's token in {} (refreshed only when the Codex CLI runs)", codex_file.display()),
            });
        }
    }
    if owner == Some(&Owner::Opencode) {
        let opencode_file = paths.opencode_dir.join("auth.json");
        if let Some(key) = read_json_object(&opencode_file)?.get(provider).and_then(|entry| entry.get("key")).and_then(Value::as_str) {
            return Ok(Credential { key: key.to_string(), account_id: None, source: format!("{}", opencode_file.display()) });
        }
    }

    // 3. models.json.
    if let Some(value) = config.and_then(|config| config.api_key.as_deref()) {
        let key = resolve_configured_key(value).await?;
        let account_id = if owner == Some(&Owner::CodexCli) { codex_account_id(&key) } else { None };
        return Ok(Credential { key, account_id, source: "models.json apiKey".to_string() });
    }

    // 4. The environment.
    if let Some(variable) = account.and_then(|account| account.env_key)
        && let Ok(key) = std::env::var(variable) {
            return Ok(Credential { key, account_id: None, source: format!("${variable}") });
        }

    // 5. None: a server of the user's own in models.json (a local one) with no apiKey wants no key,
    // and is asked with no Authorization header.
    if account.is_none() && config.is_some_and(|config| config.base_url.is_some()) {
        return Ok(Credential { key: String::new(), account_id: None, source: NO_KEY_NEEDED.to_string() });
    }

    let variable = account.and_then(|account| account.env_key).map_or(String::new(), |variable| format!(", or set ${variable}"));
    let how = account.map_or("set its apiKey in models.json".to_string(), |account| format!("log in with {}{variable}", account.login));
    Err(format!("{NO_CREDENTIAL} \"{provider}\": {how}"))
}

// The Codex login from piccosta's store, refreshed first when it has under REFRESH_MARGIN_MS left.
// The expiry is checked again after taking the lock, because another piccosta may have refreshed it
// while this one waited; refreshing twice would spend the refresh token and log one of them out.
async fn fresh_codex_login(login_file: &Path) -> Result<Credential, String> {
    let read_login = || -> Option<(String, String, i64, Option<String>)> {
        let stored = read_json_object(login_file).ok()?;
        let entry = stored.get("openai-codex")?;
        Some((
            entry.get("access")?.as_str()?.to_string(),
            entry.get("refresh")?.as_str()?.to_string(),
            entry.get("expires")?.as_i64()?,
            entry.get("accountId").and_then(Value::as_str).map(str::to_string),
        ))
    };
    let source = format!("{}", login_file.display());
    let (access, _, expires, account_id) = read_login().ok_or_else(|| format!("the Codex login in {source} is incomplete"))?;
    if expires - clock::now_ms() > REFRESH_MARGIN_MS {
        return Ok(Credential { account_id: account_id.or_else(|| codex_account_id(&access)), key: access, source });
    }

    let lock = FileLock::acquire(login_file).await.map_err(|error| format!("could not lock {source}: {error}"))?;
    let (access, refresh, expires, account_id) = read_login().ok_or_else(|| format!("the Codex login in {source} is incomplete"))?;
    if expires - clock::now_ms() > REFRESH_MARGIN_MS {
        drop(lock);
        return Ok(Credential { account_id: account_id.or_else(|| codex_account_id(&access)), key: access, source });
    }

    let stored = read_json_object(login_file)?;
    let named = |field: &str, default: &'static str| stored.get("openai-codex").and_then(|entry| entry.get(field)).and_then(Value::as_str).unwrap_or(default).to_string();
    let refreshed = tokio::time::timeout(REFRESH_TIMEOUT, refresh_codex_token(&refresh, &named("clientId", CODEX_CLIENT_ID), &named("tokenUrl", CODEX_TOKEN_URL)))
        .await
        .map_err(|_| "OAuth refresh failed for openai-codex: timed out".to_string())?
        .map_err(|error| format!("OAuth refresh failed for openai-codex: {error}"))?;
    let account_id = codex_account_id(&refreshed.access).ok_or("Failed to extract accountId from token")?;

    // Write back only our entry, keeping every other provider's login in the file as it was.
    let mut stored = read_json_object(login_file)?;
    let mut entry = stored.get("openai-codex").and_then(Value::as_object).cloned().unwrap_or_default();
    entry.insert("type".into(), "oauth".into());
    entry.insert("access".into(), refreshed.access.clone().into());
    entry.insert("refresh".into(), refreshed.refresh.into());
    entry.insert("expires".into(), refreshed.expires.into());
    entry.insert("accountId".into(), account_id.clone().into());
    stored.insert("openai-codex".into(), Value::Object(entry));
    write_private_json(login_file, &Value::Object(stored)).map_err(|error| format!("could not save {source}: {error}"))?;
    drop(lock);
    Ok(Credential { key: refreshed.access, account_id: Some(account_id), source })
}

struct RefreshedToken {
    access: String,
    refresh: String,
    expires: i64,
}

// Trade a refresh token for new tokens (openai-codex.ts:122-185). OpenAI rotates the refresh token,
// so the old one is dead once this succeeds and the new one must be saved.
async fn refresh_codex_token(refresh_token: &str, client_id: &str, token_url: &str) -> Result<RefreshedToken, String> {
    let form = [("grant_type", "refresh_token"), ("refresh_token", refresh_token), ("client_id", client_id)];
    let body: String = form
        .iter()
        .map(|(name, value)| format!("{name}={}", percent_encode(value)))
        .collect::<Vec<_>>()
        .join("&");
    let response = crate::provider::http_client()
        .build()
        .map_err(|error| format!("OpenAI Codex token refresh error: {error}"))?
        .post(token_url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|error| format!("OpenAI Codex token refresh error: {error}"))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("OpenAI Codex token refresh failed ({}): {text}", status.as_u16()));
    }
    let json: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    let access = json.get("access_token").and_then(Value::as_str);
    let refresh = json.get("refresh_token").and_then(Value::as_str);
    let expires_in = json.get("expires_in").and_then(Value::as_f64);
    match (access, refresh, expires_in) {
        (Some(access), Some(refresh), Some(seconds)) => Ok(RefreshedToken {
            access: access.to_string(),
            refresh: refresh.to_string(),
            expires: clock::now_ms() + (seconds * 1000.0) as i64,
        }),
        _ => Err(format!("OpenAI Codex token refresh response missing fields: {text}")),
    }
}

// Percent-encode a form value: everything but unreserved characters (RFC 3986 section 2.3).
fn percent_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

// Write JSON the way pi writes auth.json: two-space indent, no trailing newline, only the owner
// may read it, and its folder only the owner may open. Written to a temp file and renamed, so a
// crash never leaves half a file holding the only copy of a rotated refresh token.
pub fn write_private_json(path: &Path, value: &Value) -> io::Result<()> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::io::Write;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    let text = serde_json::to_string_pretty(value).map_err(io::Error::other)?;
    let temp = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let mut file = fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&temp)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    fs::rename(&temp, path)
}

// A lock on a file, held while the value exists. It's a directory next to the file, the way pi's
// proper-lockfile does it, so a TS pi and this binary sharing a login dir respect each other.
struct FileLock {
    lock_dir: PathBuf,
}

impl FileLock {
    // Take the lock. Waits with backoff while someone else holds it; takes over a lock whose
    // directory is older than LOCK_STALE, since its owner died without removing it.
    async fn acquire(file: &Path) -> io::Result<FileLock> {
        let lock_dir = PathBuf::from(format!("{}.lock", file.display()));
        if let Some(dir) = file.parent() {
            fs::create_dir_all(dir)?;
        }
        let started = std::time::Instant::now();
        let mut wait = LOCK_FIRST_WAIT;
        loop {
            match fs::create_dir(&lock_dir) {
                Ok(()) => return Ok(FileLock { lock_dir }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    let stale = fs::metadata(&lock_dir)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age > LOCK_STALE);
                    if stale {
                        let _ = fs::remove_dir(&lock_dir);
                        continue;
                    }
                    if started.elapsed() > LOCK_DEADLINE {
                        return Err(io::Error::new(io::ErrorKind::TimedOut, "lock held too long"));
                    }
                    tokio::time::sleep(wait).await;
                    wait = (wait * 2).min(LOCK_LONGEST_WAIT);
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.lock_dir);
    }
}
