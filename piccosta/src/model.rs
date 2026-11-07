// Models: what each connected account offers, the user's models.json on top of it, and turning a
// name like "openai-codex/gpt-6-luna" into one model.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/types.ts (the Model shape), the merge
// rules of packages/coding-agent/src/core/provider-composer.ts and model-config.ts, and the effort
// levels of packages/ai/scripts/models-dev-reasoning-options.ts.
//
// No model is listed by hand. Each provider offers what its account serves now: opencode-go's
// /models, the list Codex itself fetches with the ChatGPT login, the models Claude Code offers
// (its answer to initialize), and <baseUrl>/models for a models.json provider that names no
// models of its own. refresh_models fetches them as piccosta starts (and on /models refresh) into
// <agent dir>/models/<provider>.json, which is what load reads. A list that can't be fetched keeps
// the copy there is; a provider with no login has its copy removed, and so offers nothing.

use crate::json_number;
use crate::provider::{claude, THINKING_LEVELS};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

// Where the fetched lists are kept, under the agent dir.
pub const LISTS_DIR: &str = "models";
// opencode-go's list, the one opencode reads, or <baseUrl>/models when models.json gives opencode-go
// a baseUrl; it carries each model's API, limits, prices and efforts (it answers without a key or a
// session header too, 2026-11-03).
const OPENCODE_GO_MODELS: &str = "https://opencode.ai/zen/go/v1/models";
// Codex's list, under its backend. It leaves out a model newer than the client_version asked with,
// so piccosta asks as settings.json's `models.codexClientVersion` says, else as the Codex CLI last
// did ($CODEX_HOME/models_cache.json), else as the one installed says it is (`codex --version`),
// else as the CLI installed here on 2026-11-03 did.
const CODEX_MODELS_PATH: &str = "/codex/models";
const CODEX_CLIENT_VERSION: &str = "0.166.0";
// What Codex's list leaves out: the output limit, and the transcript features, both what pi's
// catalog gave every Codex model (fetched 2026-10-29).
const CODEX_MAX_TOKENS: u64 = 128_000;
const CODEX_COMPAT: &str = r#"{"supportsMidConvoSystemMessages": true, "supportsAdditionalTools": true}"#;
// A copy younger than this isn't fetched again at start, so the piccostas a session starts in a
// burst (subagents, tell) don't each ask; /models refresh always asks. And how long one list may
// take. settings.json's `models.freshSeconds` and `models.timeoutSeconds` replace them.
const LISTS_FRESH_S: u64 = 15 * 60;
const LIST_TIMEOUT_S: u64 = 10;

// Refresh and addition use the same deadline and origin-only redirect policy.
pub(crate) fn list_client(settings: &crate::config::Settings) -> reqwest::ClientBuilder {
    crate::provider::http_client().timeout(list_seconds(settings, "models.timeoutSeconds", LIST_TIMEOUT_S))
}

// OpenAI and Messages put their list in data; Google puts it in models and prefixes the names.
pub(crate) fn model_list(body: &Value) -> Option<&Vec<Value>> {
    body["data"].as_array().or_else(|| body["models"].as_array())
}
pub(crate) fn listed_id(raw: &Value) -> Option<&str> {
    raw["id"].as_str().or_else(|| raw["name"].as_str().map(|name| name.trim_start_matches("models/"))).filter(|id| !id.is_empty())
}

// A `models.*` setting in seconds, else its default.
fn list_seconds(settings: &crate::config::Settings, key: &str, default: u64) -> Duration {
    Duration::from_secs(settings.get(key).and_then(Value::as_u64).unwrap_or(default))
}

// The most of `codex --version` that's read. It says "codex-cli 0.166.0"; a wrapper that says more
// than this isn't saying a version.
const CODEX_VERSION_MAX_BYTES: u64 = 256;

// The version the installed Codex CLI says it is. Input: how long it may take. Output: its last
// word, or None when it isn't there, takes too long or says too much. The child is killed and
// reaped either way, and what it writes to stderr is thrown away.
async fn installed_codex_version(timeout: Duration) -> Option<String> {
    use std::process::Stdio;
    use tokio::io::AsyncReadExt;
    let mut child = tokio::process::Command::new("codex").arg("--version").stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).kill_on_drop(true).spawn().ok()?;
    let mut said = Vec::new();
    let read = tokio::time::timeout(timeout, child.stdout.take()?.take(CODEX_VERSION_MAX_BYTES + 1).read_to_end(&mut said)).await;
    let _ = child.start_kill();
    let _ = child.wait().await;
    read.ok()?.ok()?;
    (said.len() as u64 <= CODEX_VERSION_MAX_BYTES).then(|| String::from_utf8_lossy(&said).split_whitespace().last().map(str::to_string))?
}

// The refresh piccosta started with while it's still running, as a wait bounded by the list
// timeout, for a model asked for that isn't listed yet: a start with some list kept doesn't wait,
// and the model's own list may still be coming. It's awaited, never blocked on, since the
// frontends run on one thread (a LocalSet). None once the refresh is done, or once a wait for it
// has run out, so a request asked again after waiting isn't made to wait again.
pub fn listing(agent_dir: &Path) -> Option<futures_util::future::BoxFuture<'static, ()>> {
    use futures_util::FutureExt;
    use std::sync::atomic::Ordering;
    let refresh = STARTUP_REFRESH.get().filter(|refresh| refresh.peek().is_none() && !LISTING_WAITED_OUT.load(Ordering::Relaxed))?.clone();
    let timeout = list_seconds(&crate::config::Settings::load(agent_dir, None), "models.timeoutSeconds", LIST_TIMEOUT_S);
    Some(async move {
        if tokio::time::timeout(timeout, refresh).await.is_err() {
            LISTING_WAITED_OUT.store(true, Ordering::Relaxed);
        }
    }.boxed())
}
static LISTING_WAITED_OUT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// Whether a list is kept for `provider`.
pub fn kept(agent_dir: &Path, provider: &str) -> bool {
    list_copy(agent_dir, provider).is_file()
}

fn list_copy(agent_dir: &Path, provider: &str) -> PathBuf {
    agent_dir.join(LISTS_DIR).join(format!("{provider}.json"))
}

// The providers with a list: the accounts piccosta knows (auth::ACCOUNTS), then each models.json
// provider with a baseUrl and no models of its own.
fn listed_providers(file: &ModelsFile) -> Vec<String> {
    let known = crate::auth::ACCOUNTS.iter().map(|account| account.provider.to_string());
    let configured = file.providers.iter().filter(|(id, config)| crate::auth::account(id).is_none() && config.base_url.is_some() && config.models.is_empty());
    known.chain(configured.map(|(id, _)| id.clone())).collect()
}

// pi's thinking levels for the efforts a model names (getEffortThinkingLevelMap): each level it
// names, off as "none" when it names that, null for the rest; nothing when it names no level.
fn effort_map(efforts: &[&str]) -> Option<Value> {
    efforts.iter().any(|effort| *effort == "none" || THINKING_LEVELS.contains(effort)).then(|| {
        THINKING_LEVELS.iter().map(|level| {
            let name = if *level == "off" { "none" } else { level };
            (level.to_string(), if efforts.contains(&name) { name.into() } else { Value::Null })
        }).collect::<Map<_, _>>().into()
    })
}

// "text", and "image" when any of a model's input types is one.
fn input_kinds(types: &Value) -> Value {
    let image = types.as_array().into_iter().flatten().filter_map(Value::as_str).any(|kind| kind.starts_with("image"));
    if image { json!(["text", "image"]) } else { json!(["text"]) }
}

// One model of opencode-go's list as piccosta's, on the first of its APIs piccosta speaks, or
// None when it has none of them. A field it leaves out stays null, for checked to refuse the list.
fn opencode_model(raw: &Value) -> Option<Value> {
    let apis = [("chat-completions", "openai-completions"), ("responses", "openai-responses"), ("messages", "anthropic-messages"), ("generate-content", "google-generative-ai")];
    let (route, api) = apis.into_iter().find(|(route, _)| raw["api"].get(route).is_some())?;
    // The Messages route's base ends in /v1, which the Messages client adds itself.
    let base = &raw["api"][route]["base_url"];
    let base = base.as_str().filter(|_| api == "anthropic-messages").map_or(base.clone(), |base| base.trim_end_matches("/v1").into());
    let reasoning = &raw["capabilities"]["reasoning"];
    let efforts: Vec<&str> = reasoning["effort"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    let price = |pointer: &str| raw["cost"]["text"].pointer(pointer).cloned().unwrap_or(0.into());
    let mut model = json!({"id": raw["id"], "name": raw["name"].as_str().or(raw["id"].as_str()), "api": api, "baseUrl": base,
        "reasoning": reasoning["supported"] == true, "input": input_kinds(&raw["modalities"]["input"]),
        "cost": {"input": price("/input/price"), "output": price("/output/price"), "cacheRead": price("/input/cache/read"), "cacheWrite": price("/input/cache/write")},
        "contextWindow": raw["limit"]["context"], "maxTokens": raw["limit"]["output"]});
    if let Some(map) = effort_map(&efforts) {
        model["thinkingLevelMap"] = map;
    }
    // A Messages model that takes efforts and no token budget thinks adaptively (Claude Haiku 5.5,
    // which pi's catalog marked so).
    if api == "anthropic-messages" && !efforts.is_empty() && reasoning.get("budget").is_none() {
        model["compat"] = json!({"forceAdaptiveThinking": true});
    }
    Some(model)
}

// One model of Codex's list as piccosta's, or None for one Codex's own picker hides. A field it
// leaves out stays null, for checked to refuse the list.
fn codex_model(raw: &Value, base: &str) -> Option<Value> {
    if raw["visibility"] != "list" {
        return None;
    }
    let efforts: Vec<&str> = raw["supported_reasoning_levels"].as_array().into_iter().flatten().filter_map(|level| level["effort"].as_str()).collect();
    let mut model = json!({"id": raw["slug"], "name": raw["display_name"].as_str().or(raw["slug"].as_str()), "api": "openai-codex-responses",
        "baseUrl": base, "reasoning": !efforts.is_empty(), "input": input_kinds(&raw["input_modalities"]),
        "contextWindow": raw["context_window"], "maxTokens": CODEX_MAX_TOKENS, "compat": serde_json::from_str::<Value>(CODEX_COMPAT).ok()});
    if let Some(map) = effort_map(&efforts) {
        model["thinkingLevelMap"] = map;
    }
    Some(model)
}

// One model Claude Code offers, in the shape claude::bridge_models reads, by its full id. Its
// efforts are the ones Claude Code lists, and "none" when it can also run with thinking off. Every
// Claude model thinks and takes images (Anthropic's list, 2026-11-03), which Claude Code's leaves
// unsaid; it says nothing of limits either, so the bridge's own defaults apply.
fn claude_model(raw: &Value) -> Value {
    let id = &raw["resolvedModel"];
    let mut efforts: Vec<&str> = raw["supportedEffortLevels"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if !efforts.is_empty() && raw["supportsThinkingOff"] == true {
        efforts.push("none");
    }
    let mut model = json!({"id": id, "name": raw["displayName"].as_str().or(id.as_str()), "reasoning": true, "input": ["text", "image"]});
    if let Some(map) = effort_map(&efforts) {
        model["thinkingLevelMap"] = map;
    }
    model
}

// One model of a configured provider's OpenAI-style list, with pi's defaults for what it doesn't say.
fn listed_model(raw: &Value, config: &ProviderConfig) -> Option<Value> {
    let id = listed_id(raw)?;
    Some(json!({"id": id, "name": id, "api": config.api.as_deref().unwrap_or("openai-completions"), "baseUrl": config.base_url, "input": ["text"],
        "contextWindow": DEFAULT_CONTEXT_WINDOW, "maxTokens": DEFAULT_MAX_TOKENS}))
}

// The add flow already fetched these ids; keep them just as refresh_models keeps a fetched list.
pub(crate) fn keep_added_list(agent_dir: &Path, provider: &str, ids: &[String], config: &ProviderConfig) -> Result<(), String> {
    let (models, _) = checked(provider, ids.iter().filter_map(|id| listed_model(&json!({"id": id}), config)), home(provider, Some(config))?)?;
    keep_list(&list_copy(agent_dir, provider), &models).map_err(|error| error.to_string())
}

// One private partial per writer, renamed whole; neither add nor refresh can expose half a list.
fn keep_list(copy: &Path, models: &[Value]) -> std::io::Result<()> {
    crate::platform::private_dir(copy.parent().ok_or_else(|| std::io::Error::other("model list has no directory"))?)?;
    let partial = copy.with_extension(format!("json.{}.partial", uuid::Uuid::new_v4().simple()));
    let result = crate::auth::write_private_json(&partial, &json!(models)).and_then(|()| fs::rename(&partial, copy));
    if result.is_err() { let _ = fs::remove_file(&partial); }
    result
}

// One provider's list as its account serves it now, checked as load will read it. Output: its
// models and a note for each one left out, or why there's none (beginning auth::NO_CREDENTIAL when
// the provider has no login here).
async fn fetch_list(client: &reqwest::Client, provider: &str, config: Option<&ProviderConfig>, paths: &crate::auth::CredentialPaths, settings: &crate::config::Settings) -> Result<(Vec<Value>, Vec<String>), String> {
    type Convert = Box<dyn Fn(&Value) -> Option<Value> + Send>;
    use crate::auth::Owner;
    let credential = crate::auth::credential_for(provider, config, paths).await;
    let owner = crate::auth::account(provider).map(|account| &account.owner);
    // Claude counts as connected only with Claude Code found and logged in (auth.rs), and its
    // list is the one Claude Code itself offers.
    if owner == Some(&Owner::ClaudeCode) {
        credential?;
        return checked(provider, crate::provider::claude::offered_models().await?.iter().map(claude_model), None);
    }
    let (url, credential, pointer, convert): (String, _, &str, Convert) = match owner {
        Some(Owner::Opencode) => {
            let url = config.and_then(|config| config.base_url.as_deref()).map_or(OPENCODE_GO_MODELS.into(), |base| format!("{}/models", base.trim_end_matches('/')));
            (url, Some(credential?), "/data", Box::new(opencode_model))
        }
        Some(Owner::CodexCli) => {
            let base = config.and_then(|config| config.base_url.clone()).unwrap_or(crate::provider::codex::DEFAULT_BASE_URL.into());
            let cache: Value = fs::read_to_string(paths.codex_home.join("models_cache.json")).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
            let version = match settings.get("models.codexClientVersion").and_then(Value::as_str).or(cache["client_version"].as_str()) {
                Some(version) => version.to_string(),
                None => installed_codex_version(list_seconds(settings, "models.timeoutSeconds", LIST_TIMEOUT_S)).await.unwrap_or(CODEX_CLIENT_VERSION.into()),
            };
            (format!("{}{CODEX_MODELS_PATH}?client_version={version}", base.trim_end_matches('/')), Some(credential?), "/models", Box::new(move |raw: &Value| codex_model(raw, &base)))
        }
        // A configured provider, asked with no key when it has none (auth.rs).
        _ => {
            let config = config.ok_or("no settings in models.json")?.clone();
            let credential = Some(credential?);
            let base = config.base_url.as_deref().ok_or("no baseUrl")?.trim_end_matches('/');
            let route = if config.api.as_deref() == Some("anthropic-messages") { "/v1/models" } else { "/models" };
            (format!("{base}{route}"), credential, "/data", Box::new(move |raw: &Value| listed_model(raw, &config)))
        }
    };
    // Authorized as inference is: the key, then the provider's own headers from models.json.
    let mut request = client.get(&url);
    if let Some(credential) = credential.filter(|credential| !credential.key.is_empty()) {
        credential.check_origin(&url)?;
        match config.and_then(|config| config.api.as_deref()) {
            Some("anthropic-messages") => request = request.header("x-api-key", &credential.key).header("anthropic-version", crate::provider::anthropic_messages::ANTHROPIC_VERSION),
            Some("google-generative-ai") => request = request.header("x-goog-api-key", &credential.key),
            _ => {},
        }
        request = request.bearer_auth(credential.key);
        if let Some(account) = credential.account_id {
            request = request.header("chatgpt-account-id", account);
        }
    }
    for (name, value) in config.and_then(|config| config.headers.as_ref()).into_iter().flatten() {
        request = request.header(name, value);
    }
    let body = async { request.send().await?.error_for_status()?.json::<Value>().await }.await.map_err(crate::provider::http_error)?;
    let list = if owner.is_none() { model_list(&body) } else { body.pointer(pointer).and_then(Value::as_array) }.ok_or_else(|| format!("the answer has no list at {pointer}"))?;
    checked(provider, list.iter().filter_map(convert), home(provider, config)?)
}

// The origin a provider's models may send its key to: the one its list is asked of, its
// models.json baseUrl's, else opencode-go's or Codex's own (https). None for Claude, whose models
// run through the bridge.
fn home(provider: &str, config: Option<&ProviderConfig>) -> Result<Option<url::Origin>, String> {
    use crate::auth::Owner;
    let base = config.and_then(|config| config.base_url.as_deref());
    let base = match crate::auth::account(provider).map(|account| &account.owner) {
        Some(Owner::ClaudeCode) => return Ok(None),
        Some(Owner::Opencode) => base.unwrap_or(OPENCODE_GO_MODELS),
        Some(Owner::CodexCli) => base.unwrap_or(crate::provider::codex::DEFAULT_BASE_URL),
        None => base.ok_or("no baseUrl")?,
    };
    Ok(Some(reqwest::Url::parse(base).map_err(|error| format!("baseUrl {base}: {error}"))?.origin()))
}

// A fetched list made ready to keep. One model per id: a later entry for an id (Claude Code names a
// model again under an alias such as "opus") replaces the earlier one in its place. Every model is
// read as load reads it, so one bad entry keeps the last list instead of breaking them all. Then,
// as a list doesn't choose where the key goes, a model whose URL isn't on the provider's home is
// left out, with a note. Input: the provider, its converted models, and its home (home()). Output:
// the models and the notes, or why the list can't be kept.
fn checked(provider: &str, models: impl Iterator<Item = Value>, origin: Option<url::Origin>) -> Result<(Vec<Value>, Vec<String>), String> {
    let models: Vec<Value> = models.map(|model| (model["id"].to_string(), model)).collect::<IndexMap<_, _>>().into_values().collect();
    as_models(provider, &models)?;
    let mut notes = Vec::new();
    let models = models.into_iter().filter(|model| {
        let at_home = origin.as_ref().is_none_or(|origin| model["baseUrl"].as_str().and_then(|url| reqwest::Url::parse(url).ok()).is_some_and(|url| url.origin() == *origin));
        if !at_home {
            notes.push(format!("left out {}: its URL {} isn't on {}", model["id"], model["baseUrl"], origin.as_ref().map_or(String::new(), url::Origin::ascii_serialization)));
        }
        at_home
    }).collect();
    Ok((models, notes))
}

// A provider's list as load reads it (the bridge's shape for Claude). Output: the models, or the
// first one that isn't a model piccosta can read and why.
fn as_models(provider: &str, models: &[Value]) -> Result<Vec<Map<String, Value>>, String> {
    let list = models.iter().map(|model| model.as_object().cloned().ok_or_else(|| format!("{model} isn't a model"))).collect::<Result<Vec<_>, _>>()?;
    let read = list.len();
    let list = if provider == claude::PROVIDER { claude::bridge_models(list) } else { list };
    if list.len() < read {
        return Err("a model has no id".into());
    }
    for model in &list {
        let mut whole = model.clone();
        whole.entry("provider").or_insert(provider.into());
        serde_json::from_value::<Model>(Value::Object(whole)).map_err(|error| format!("model {}: {error}", model.get("id").unwrap_or(&Value::Null)))?;
    }
    Ok(list)
}

// models.json as the lists read it, empty when it's missing or doesn't read (load says why).
fn models_file(agent_dir: &Path) -> ModelsFile {
    fs::read_to_string(agent_dir.join("models.json")).ok().and_then(|text| serde_json::from_str(&strip_json_comments(text.trim_start_matches('\u{feff}'))).ok()).unwrap_or_default()
}

// Whether any provider has a kept list load can read with a model in it, so a start needn't wait
// for the lists.
pub fn has_lists(agent_dir: &Path) -> bool {
    let file = models_file(agent_dir);
    listed_providers(&file).iter().any(|provider| !listed(agent_dir, provider, file.providers.get(provider)).is_empty())
}

// The refresh piccosta started with, for the terminal UI to show what went wrong in it.
type Refresh = futures_util::future::Shared<futures_util::future::BoxFuture<'static, Vec<Result<String, String>>>>;
pub static STARTUP_REFRESH: std::sync::OnceLock<Refresh> = std::sync::OnceLock::new();

// Start the refresh piccosta starts with, in the background, kept in STARTUP_REFRESH. Output: it,
// for a start with no lists to wait on.
pub fn start_refresh(agent_dir: PathBuf) -> Refresh {
    use futures_util::FutureExt;
    let refresh = refresh_models(agent_dir, false).boxed().shared();
    tokio::spawn(refresh.clone());
    STARTUP_REFRESH.get_or_init(|| refresh).clone()
}

// Fetch every provider's list again, each kept whole under the agent dir for the next load to
// read. Input: whether to fetch a copy that's still fresh. Output: a line per provider saying what
// came of it (none for a copy left alone as fresh), Err for what went wrong.
pub async fn refresh_models(agent_dir: PathBuf, force: bool) -> Vec<Result<String, String>> {
    let file = models_file(&agent_dir);
    let paths = crate::auth::CredentialPaths::from_environment(&agent_dir);
    let settings = crate::config::Settings::load(&agent_dir, None);
    let lists_fresh = list_seconds(&settings, "models.freshSeconds", LISTS_FRESH_S);
    let client = match list_client(&settings).build() {
        Ok(client) => client,
        Err(error) => return vec![Err(format!("couldn't fetch the model lists: {error}"))],
    };
    let _ = fs::create_dir_all(agent_dir.join(LISTS_DIR));
    // Only an account with no login loses its list, fresh or not; one whose login couldn't be read
    // just now (offline, a timeout) keeps it. The rest are fetched when stale.
    let mut lines = Vec::new();
    let mut providers = Vec::new();
    for provider in listed_providers(&file) {
        let age = list_copy(&agent_dir, &provider).metadata().and_then(|metadata| metadata.modified()).ok().and_then(|modified| modified.elapsed().ok());
        if force || age.is_none_or(|age| age >= lists_fresh) {
            providers.push(provider);
        } else if crate::auth::account(&provider).is_some()
            && let Err(error) = crate::auth::credential_for(&provider, file.providers.get(&provider), &paths).await
        {
            if error.starts_with(crate::auth::NO_CREDENTIAL) {
                let _ = fs::remove_file(list_copy(&agent_dir, &provider));
                lines.push(Ok(format!("{provider}: not connected")));
            } else {
                lines.push(Err(format!("{provider}: {error}; kept the last list")));
            }
        }
    }
    let lists = futures_util::future::join_all(providers.iter().map(|provider| fetch_list(&client, provider, file.providers.get(provider), &paths, &settings))).await;
    for (provider, list) in providers.iter().zip(lists) {
        let copy = list_copy(&agent_dir, provider);
        match list {
            Err(error) if error.starts_with(crate::auth::NO_CREDENTIAL) => {
                let _ = fs::remove_file(&copy);
                lines.push(Ok(format!("{provider}: not connected")));
            }
            Err(error) if !copy.is_file() => lines.push(Err(format!("{provider}: not connected; couldn't fetch the list ({error})"))),
            Err(error) => lines.push(Err(format!("{provider}: couldn't fetch the list ({error}); kept the last one"))),
            // Written whole under a name of its own, then moved into place, so a load never reads
            // half of one and two refreshes at once don't write into the same file.
            Ok((models, notes)) => {
                lines.extend(notes.into_iter().map(|note| Err(format!("{provider}: {note}"))));
                lines.push(keep_list(&copy, &models).map(|()| format!("{provider}: {} models", models.len())).map_err(|error| format!("{provider}: couldn't keep the list ({error})")));
            }
        }
    }
    lines
}

// The models of a provider's kept list as load reads them, the ones on its home only (a copy kept
// before that was checked may hold others); none when there's no copy or it doesn't read as models.
pub(crate) fn listed(agent_dir: &Path, provider: &str, config: Option<&ProviderConfig>) -> Vec<Map<String, Value>> {
    let models: Vec<Value> = fs::read_to_string(list_copy(agent_dir, provider)).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
    home(provider, config).and_then(|home| checked(provider, models.into_iter(), home)).and_then(|(models, _)| as_models(provider, &models)).unwrap_or_default()
}

// What a model declared in models.json or listed by a configured provider gets when it says
// nothing else and no listed model shares its id (provider-composer.ts:232-247).
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
const DEFAULT_MAX_TOKENS: u64 = 16_384;

// Dollars per million tokens. `tiers` hold higher prices that apply past an input size.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    #[serde(serialize_with = "json_number::serialize", default)]
    pub input: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub output: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_read: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_write: f64,
    // Writes kept in the cache for an hour, when they cost other than responses.rs's fallback.
    #[serde(rename = "cacheWrite1h", serialize_with = "json_number::serialize_option", default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<CostTier>,
    // Whether the model's list or models.json gave prices at all, so a $0 given is $0 and none given
    // is unknown: true for any `cost` read, false for a model without one (Model's default).
    #[serde(skip, default = "supplied")]
    pub supplied: bool,
}

fn supplied() -> bool {
    true
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CostTier {
    pub input_tokens_above: u64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub input: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub output: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_read: f64,
    #[serde(serialize_with = "json_number::serialize", default)]
    pub cache_write: f64,
}

// One chat model. `compat` holds the provider quirks the request builders read (developer role,
// strict mode, thinking format, ...); it stays JSON because each wire protocol reads its own keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    #[serde(default)]
    pub reasoning: bool,
    // "text", and "image" when it takes images.
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
    // pi's thinking levels (off, minimal, low, medium, high, xhigh, max) to the provider's own
    // effort names. A null value marks a level the model doesn't support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<IndexMap<String, Option<String>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_limits: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<IndexMap<String, String>>,
    // The service tier to ask for, set per model in models.json's modelOverrides: OpenAI's
    // "priority" is Codex's fast mode (about twice as fast, counted higher against the plan), and
    // Claude Code's fast mode on the Claude bridge. A session's /fast replaces it (agent::service_tier).
    // It's a request field, outside the prompt. None: default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

impl Model {
    // A compat flag, false when the model doesn't set it.
    pub fn compat_flag(&self, key: &str) -> bool {
        self.compat.as_ref().and_then(|compat| compat.get(key)).and_then(Value::as_bool).unwrap_or(false)
    }

    // A compat string value, such as `thinkingFormat`.
    pub fn compat_str(&self, key: &str) -> Option<&str> {
        self.compat.as_ref().and_then(|compat| compat.get(key)).and_then(Value::as_str)
    }

    // The provider's effort name for a pi thinking level. Input: a level such as "medium".
    // Output: Some(name) when the model reasons and maps the level (a level absent from the map
    // passes through unchanged), None when the model doesn't reason or marks the level null.
    pub fn provider_thinking_level(&self, level: &str) -> Option<String> {
        if !self.reasoning {
            return None;
        }
        match self.thinking_level_map.as_ref().and_then(|map| map.get(level)) {
            Some(Some(name)) => Some(name.clone()),
            Some(None) => None,
            None => Some(level.to_string()),
        }
    }
}

// models.json: per-provider settings over the lists, and models of the user's own
// (model-config.ts:19-87). Only the fields piccosta acts on are read; others are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelsFile {
    #[serde(default)]
    providers: IndexMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    pub base_url: Option<String>,
    // The key, or `$VAR`, or `!command` whose output is the key. See auth.rs.
    pub api_key: Option<String>,
    pub api: Option<String>,
    pub headers: Option<IndexMap<String, String>>,
    pub compat: Option<Map<String, Value>>,
    #[serde(default)]
    pub models: Vec<Map<String, Value>>,
    // Per-model field overrides, applied last.
    #[serde(default)]
    pub model_overrides: IndexMap<String, Map<String, Value>>,
}

// Every model piccosta knows, and each provider's models.json settings.
pub struct ModelRegistry {
    models: Vec<Model>,
    providers: IndexMap<String, ProviderConfig>,
}

// Merge `overlay`'s keys into `base`, one level deep for objects (so a compat override changes
// one flag without dropping the others), replacing everything else.
fn merge_object(base: &mut Map<String, Value>, overlay: &Map<String, Value>) {
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(Value::Object(existing)), Value::Object(update)) => {
                for (inner_key, inner_value) in update {
                    existing.insert(inner_key.clone(), inner_value.clone());
                }
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

impl ModelRegistry {
    // The kept lists with models.json on top. Input: the path of models.json, which may not exist.
    // Output: the registry, or an error naming the file when it exists but isn't valid. The file may
    // have // comments and a BOM, as pi allows (model-config.ts:140).
    pub fn load(models_json: &Path) -> Result<ModelRegistry, String> {
        let file: ModelsFile = match fs::read_to_string(models_json) {
            Ok(text) => {
                let cleaned = strip_json_comments(text.trim_start_matches('\u{feff}'));
                serde_json::from_str(&cleaned)
                    .map_err(|error| format!("Invalid models.json: {error}\n\nFile: {}", models_json.display()))?
            }
            Err(_) => ModelsFile::default(),
        };

        // Each provider's kept list; Claude runs only through the bridge.
        let mut models_by_provider: IndexMap<String, Vec<Map<String, Value>>> = IndexMap::new();
        for provider in listed_providers(&file) {
            let list = models_json.parent().map(|dir| listed(dir, &provider, file.providers.get(&provider))).unwrap_or_default();
            if !list.is_empty() {
                models_by_provider.insert(provider, list);
            }
        }

        for (provider, config) in &file.providers {
            let list = models_by_provider.entry(provider.clone()).or_default();
            // Provider-wide settings reach every listed model of that provider.
            for model in list.iter_mut() {
                if let Some(base_url) = &config.base_url {
                    model.insert("baseUrl".into(), Value::String(base_url.clone()));
                }
                if let Some(compat) = &config.compat {
                    let mut overlay = Map::new();
                    overlay.insert("compat".into(), Value::Object(compat.clone()));
                    merge_object(model, &overlay);
                }
            }
            // The user's own models: replace a listed model with the same id, field by field, or
            // join the list. A new model starts from the provider's settings and pi's defaults.
            for declared in &config.models {
                let Some(id) = declared.get("id").and_then(Value::as_str) else { continue };
                if let Some(existing) = list.iter_mut().find(|model| model.get("id").and_then(Value::as_str) == Some(id)) {
                    if let Some(api) = &config.api {
                        existing.insert("api".into(), Value::String(api.clone()));
                    }
                    merge_object(existing, declared);
                    continue;
                }
                let mut model = Map::new();
                model.insert("id".into(), Value::String(id.to_string()));
                model.insert("name".into(), Value::String(id.to_string()));
                model.insert("provider".into(), Value::String(provider.clone()));
                if let Some(api) = &config.api {
                    model.insert("api".into(), Value::String(api.clone()));
                }
                if let Some(base_url) = &config.base_url {
                    model.insert("baseUrl".into(), Value::String(base_url.clone()));
                }
                model.insert("input".into(), serde_json::json!(["text"]));
                model.insert("contextWindow".into(), DEFAULT_CONTEXT_WINDOW.into());
                model.insert("maxTokens".into(), DEFAULT_MAX_TOKENS.into());
                if let Some(compat) = &config.compat {
                    model.insert("compat".into(), Value::Object(compat.clone()));
                }
                merge_object(&mut model, declared);
                list.push(model);
            }
            for (id, overrides) in &config.model_overrides {
                if let Some(model) = list.iter_mut().find(|model| model.get("id").and_then(Value::as_str) == Some(id)) {
                    merge_object(model, overrides);
                }
            }
            if let Some(headers) = &config.headers {
                for model in list.iter_mut() {
                    let mut merged = serde_json::to_value(headers).unwrap_or_default();
                    if let (Some(merged), Some(own)) = (merged.as_object_mut(), model.get("headers").and_then(Value::as_object)) {
                        merged.extend(own.clone());
                    }
                    model.insert("headers".into(), merged);
                }
            }
        }

        let mut models = Vec::new();
        for (provider, list) in models_by_provider {
            for mut raw in list {
                raw.entry("provider").or_insert(Value::String(provider.clone()));
                let id = raw.get("id").and_then(Value::as_str).unwrap_or("?").to_string();
                let model: Model = serde_json::from_value(Value::Object(raw))
                    .map_err(|error| format!("model {provider}/{id}: {error}"))?;
                models.push(model);
            }
        }
        Ok(ModelRegistry { models, providers: file.providers })
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    // A provider's models.json settings, if it has any.
    pub fn provider_config(&self, provider: &str) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }

    // The providers models.json names, in its order.
    pub fn configured(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    // Find one model by name. Input: "provider/id", or a bare id. Output: the model, or an error
    // listing what matched when a bare id is ambiguous or nothing matches. Matching is exact and
    // case-insensitive; pi's substring and fuzzy matching are left out, since a near-miss model name
    // should fail loudly rather than pick something else.
    pub fn resolve(&self, name: &str) -> Result<&Model, String> {
        if let Some((provider, id)) = name.split_once('/')
            && let Some(model) = self
                .models
                .iter()
                .find(|model| model.provider.eq_ignore_ascii_case(provider) && model.id.eq_ignore_ascii_case(id))
            {
                return Ok(model);
            }
        let matches: Vec<&Model> = self.models.iter().filter(|model| model.id.eq_ignore_ascii_case(name)).collect();
        match matches.as_slice() {
            [one] => Ok(one),
            [] => Err(match self.models.first() {
                Some(model) => format!("Model \"{name}\" not found. Use provider/id, e.g. {}/{}.", model.provider, model.id),
                None => format!("Model \"{name}\" not found: no account is connected. Log in to one, then /models refresh."),
            }),
            several => Err(format!(
                "Model \"{name}\" is offered by several providers; say which: {}",
                several.iter().map(|model| format!("{}/{}", model.provider, model.id)).collect::<Vec<_>>().join(", ")
            )),
        }
    }
}

// Remove // line comments and /* */ block comments from JSON text, leaving strings alone. Input:
// JSON with comments. Output: plain JSON.
pub fn strip_json_comments(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    let mut in_string = false;
    while let Some(character) = characters.next() {
        if in_string {
            output.push(character);
            if character == '\\' {
                if let Some(escaped) = characters.next() {
                    output.push(escaped);
                }
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match (character, characters.peek()) {
            ('"', _) => {
                in_string = true;
                output.push(character);
            }
            ('/', Some('/')) => {
                for skipped in characters.by_ref() {
                    if skipped == '\n' {
                        output.push('\n');
                        break;
                    }
                }
            }
            ('/', Some('*')) => {
                characters.next();
                let mut previous = '\0';
                for skipped in characters.by_ref() {
                    if previous == '*' && skipped == '/' {
                        break;
                    }
                    previous = skipped;
                }
            }
            _ => output.push(character),
        }
    }
    output
}
