// web_search through the model account the user already has: one Responses request to a search
// model with OpenAI's server-side web_search tool turned on. The provider runs the search and its
// model writes a cited answer, so nothing of ours is hosted and no search API key is needed.
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): openai-search.ts, its
// request body, instructions, stream parsing, result order and error messages. Left out: the
// alpha/search endpoint, api.openai.com keys, custom Responses URLs, and the picker that walks
// every registered model; the model is `web.searchModel`, else the first of the scale's search
// kind's models that speaks a Responses API and whose provider has a credential.
// - Measured 2026-10-29 (ledger.md, the 30 known-item queries SearXNG was measured on, through
//   the opencode-go plan): gpt-6-luna put the expected page first for 26 and in the top
//   three for all 30, median 6.4 s, max 44 s, $0.0035 of plan allowance a query. On 19 past
//   queries it answered all 19 (SearXNG 15). claude-haiku-5-5 with Anthropic's web_search tool
//   ranked it in the top ten for only 26 at about twice the cost; it speaks the Messages API, so
//   web_search passes over it in the search kind's list.
// - The search model can be any model the user has; the model asking needn't search itself, so a
//   DeepSeek session searches through gpt-6-luna on the same opencode-go plan.

use super::search::{domain_filters, SearchOptions, SearchResponse, SearchResult};
use crate::auth::{self, Credential, CredentialPaths};
use crate::model::{Model, ModelRegistry, ProviderConfig};
use crate::provider::{self, codex};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::sync::LazyLock;
use std::time::Duration;

// The scale kind whose models web_search asks when `web.searchModel` isn't set.
const SEARCH_KIND: &str = "search";
// The wire protocols this request speaks: Codex's backend and the Responses API as opencode-go
// serves it.
const CODEX_API: &str = "openai-codex-responses";
const RESPONSES_API: &str = "openai-responses";
// openai-search.ts:13.
const SEARCH_TIMEOUT: Duration = Duration::from_secs(60);
// normalizeSearchResultCount's ceiling, which "Prefer around N sources" and the result list share.
const MAX_RESULTS: f64 = 20.0;
// extractSnippetAround: 100 UTF-16 units each side of a citation, cut at 300.
const SNIPPET_CONTEXT: usize = 100;
const MAX_SNIPPET: usize = 300;

static MARKDOWN_LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]*)\]\([^)]*\)").expect("valid regex"));

// A model web_search can ask, with its provider's models.json settings for the credential lookup.
#[derive(Debug, Clone)]
pub struct Target {
    pub model: Model,
    pub config: Option<ProviderConfig>,
}

// The models web_search may ask: `web.searchModel`, else the search kind's models, in the scale's
// order. Input: the registry, the configured name and the agent dir. Output: the targets in order,
// or an error when the configured model isn't known or doesn't speak a Responses API, or none of
// the kind's does (it's reported when a search runs, not at startup), naming each one left out.
pub fn targets(registry: &ModelRegistry, configured: Option<&str>, agent_dir: &std::path::Path) -> Result<Vec<Target>, String> {
    let target = |model: &Model| Target { model: model.clone(), config: registry.provider_config(&model.provider).cloned() };
    let responses = |model: &Model| model.api == CODEX_API || model.api == RESPONSES_API;
    if let Some(name) = configured {
        let model = registry.resolve(name).map_err(|error| format!("web.searchModel: {error}"))?;
        if !responses(model) {
            return Err(format!("web.searchModel: {}/{} speaks {}; web search needs a model on an OpenAI Responses API ({CODEX_API} or {RESPONSES_API})", model.provider, model.id, model.api));
        }
        return Ok(vec![target(model)]);
    }
    let kind = crate::scale::find(agent_dir, SEARCH_KIND)?;
    let (mut targets, mut skipped) = (Vec::new(), Vec::new());
    for name in crate::scale::order(agent_dir, &kind) {
        match registry.resolve(crate::scale::split(&name).0) {
            Ok(model) if responses(model) => targets.push(target(model)),
            Ok(model) => skipped.push(format!("{name} (speaks {})", model.api)),
            Err(_) => skipped.push(format!("{name} (no connected account serves it)")),
        }
    }
    if targets.is_empty() {
        return Err(format!("None of the {SEARCH_KIND} kind's models can search: web search needs one on an OpenAI Responses API ({CODEX_API} or {RESPONSES_API}); left out {}. Add one with /scale {SEARCH_KIND}, set web.searchModel, or set web.searxngBaseUrl to your own SearXNG", skipped.join(", ")));
    }
    Ok(targets)
}

// The first target whose provider has a credential, with that credential.
async fn pick<'a>(targets: &'a [Target], paths: &CredentialPaths) -> Result<(&'a Target, Credential), String> {
    let mut tried = Vec::new();
    for target in targets {
        match auth::credential_for(&target.model.provider, target.config.as_ref(), paths).await {
            Ok(credential) => return Ok((target, credential)),
            Err(_) => tried.push(format!("{}/{}", target.model.provider, target.model.id)),
        }
    }
    let tried = if tried.is_empty() { String::new() } else { format!(" (no credential for {})", tried.join(", ")) };
    Err(format!("No search model available{tried}. Log in to one of the {SEARCH_KIND} kind's accounts, set web.searchModel to a provider/model on an OpenAI Responses API, or set web.searxngBaseUrl to your own SearXNG"))
}

// The number of sources asked for, when the call gave one (pi passes numResults through only when
// it's a positive number).
fn requested_count(options: &SearchOptions<'_>) -> Option<usize> {
    options.num_results.filter(|count| count.is_finite() && *count > 0.0).map(|count| count.floor().min(MAX_RESULTS) as usize)
}

// buildInstructions.
fn instructions(options: &SearchOptions<'_>) -> String {
    let mut lines = vec![
        "Search the web and return a concise answer grounded only in the web results.".to_string(),
        "Include clickable source citations in the response text when possible.".to_string(),
    ];
    if let Some(range) = options.recency {
        let label = match range {
            "day" => "past 24 hours",
            "week" => "past week",
            "month" => "past month",
            _ => "past year",
        };
        lines.push(format!("Prefer sources from the {label}."));
    }
    if let Some(count) = requested_count(options) {
        lines.push(format!("Prefer around {count} distinct sources."));
    }
    let filters = domain_filters(options.domain_filter);
    if !filters.allowed.is_empty() {
        lines.push(format!("Only use sources from: {}.", filters.allowed.join(", ")));
    }
    if !filters.blocked.is_empty() {
        lines.push(format!("Do not use sources from: {}.", filters.blocked.join(", ")));
    }
    lines.join(" ")
}

// buildWebSearchTool: the domain filters go to the tool as well as the instructions.
fn web_search_tool(options: &SearchOptions<'_>) -> Value {
    let mut tool = json!({"type": "web_search"});
    let filters = domain_filters(options.domain_filter);
    if !filters.allowed.is_empty() || !filters.blocked.is_empty() {
        let mut limits = Map::new();
        if !filters.allowed.is_empty() {
            limits.insert("allowed_domains".into(), json!(filters.allowed));
        }
        if !filters.blocked.is_empty() {
            limits.insert("blocked_domains".into(), json!(filters.blocked));
        }
        tool["filters"] = Value::Object(limits);
    }
    tool
}

// runOpenAISearch's body, keys in pi's order.
pub fn request_body(model_id: &str, query: &str, options: &SearchOptions<'_>) -> Value {
    json!({
        "model": model_id,
        "instructions": instructions(options),
        "input": [{"role": "user", "content": [{"type": "input_text", "text": query}]}],
        "tools": [web_search_tool(options)],
        "include": ["web_search_call.action.sources"],
        "store": false,
        "stream": true,
        "tool_choice": "required",
        "parallel_tool_calls": true,
    })
}

// Send the request with the headers the provider's own chat requests carry (codex.rs,
// openai_responses.rs). Output: the response, or the transport error.
async fn send(target: &Target, credential: &Credential, session_id: &str, body: &Value) -> Result<reqwest::Response, String> {
    let model = &target.model;
    credential.check_origin(&model.base_url)?;
    let client = provider::http_client().timeout(SEARCH_TIMEOUT).build().map_err(|error| error.to_string())?;
    let mut http = if model.api == CODEX_API {
        let account_id = credential.account_id.as_deref().ok_or("Failed to extract accountId from token")?;
        client.post(codex::endpoint(&model.base_url)).header("chatgpt-account-id", account_id).header("originator", provider::client_name())
    } else {
        let mut http = client.post(format!("{}/responses", model.base_url.trim_end_matches('/')));
        if provider::is_opencode(model) {
            http = http.header("x-opencode-client", provider::client_name()).header("x-opencode-session", session_id);
        }
        http
    };
    for (name, value) in model.headers.iter().flatten() {
        http = http.header(name, value);
    }
    http.header("Authorization", format!("Bearer {}", credential.key))
        .header("content-type", "application/json")
        .header("OpenAI-Beta", "responses=experimental")
        .body(body.to_string())
        .send()
        .await
        .map_err(|error| if error.is_timeout() { "The operation was aborted.".to_string() } else { error.to_string() })
}

fn is_web_search_call(item: &Value) -> bool {
    item.get("type").and_then(Value::as_str) == Some("web_search_call")
}

// parseOpenAIResponse. Input: the response body, a JSON object or server-sent events. Output: the
// response's output items and whether a web_search call was seen, or the stream's error.
fn parse_response(text: &str) -> Result<(Vec<Value>, bool), String> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        let parsed: Value = serde_json::from_str(trimmed).map_err(|error| format!("OpenAI API returned invalid JSON: {error}"))?;
        let output = match &parsed {
            Value::Array(items) => items.clone(),
            _ => parsed.get("output").and_then(Value::as_array).cloned().unwrap_or_default(),
        };
        let seen = output.iter().any(is_web_search_call);
        return Ok((output, seen));
    }

    let mut output_items = Vec::new();
    let mut completed: Option<Value> = None;
    let mut seen = false;
    let mut stream_error: Option<Value> = None;
    for line in text.split('\n') {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
        let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();
        // The API can fail mid-stream with HTTP 200: an `error` event or `response.failed` carries
        // the real error (quota, say) and no output ever arrives. The first error wins.
        if kind == "error" && stream_error.is_none() {
            stream_error = Some(match event.get("error") {
                Some(error) if error.is_object() => error.clone(),
                error => json!({"code": event.get("code"), "message": error.and_then(Value::as_str).or_else(|| event.get("message").and_then(Value::as_str))}),
            });
        }
        if kind.starts_with("response.web_search_call") {
            seen = true;
        }
        if kind == "response.output_item.done"
            && let Some(item) = event.get("item").filter(|item| !item.is_null())
        {
            seen |= is_web_search_call(item);
            output_items.push(item.clone());
        }
        if matches!(kind, "response.failed" | "response.incomplete" | "response.done" | "response.completed")
            && let Some(response) = event.get("response").filter(|response| response.is_object())
        {
            if let Some(error) = response.get("error").filter(|error| error.is_object()) {
                stream_error.get_or_insert_with(|| error.clone());
            }
            // A failed response can carry `error: null`; its partial output never passes for a search.
            if kind == "response.failed" {
                stream_error.get_or_insert_with(|| json!({"message": "response stream failed with no error payload (invalid response)"}));
            }
            completed = Some(response.clone());
        }
    }

    if let Some(error) = stream_error {
        let field = |key: &str| error.get(key).and_then(Value::as_str).filter(|value| !value.is_empty());
        let detail = [field("type"), field("code")].into_iter().flatten().collect::<Vec<_>>().join("/");
        let message = field("message").filter(|message| !message.trim().is_empty()).unwrap_or("OpenAI API stream failed");
        let detail = if detail.is_empty() { String::new() } else { format!(" ({detail})") };
        return Err(format!("OpenAI API stream error{detail}: {message}"));
    }
    if let Some(response) = completed {
        let output = response.get("output").and_then(Value::as_array).filter(|output| !output.is_empty()).cloned().unwrap_or(output_items);
        let seen = seen || output.iter().any(is_web_search_call);
        return Ok((output, seen));
    }
    if !output_items.is_empty() {
        let seen = seen || output_items.iter().any(is_web_search_call);
        return Ok((output_items, seen));
    }
    Err("OpenAI API returned no parseable response output".to_string())
}

// cleanSourceUrl: OpenAI tags cited links with utm_source=openai; the tag goes.
fn clean_source_url(raw: &str) -> String {
    match url::Url::parse(raw) {
        Ok(mut parsed) => {
            if parsed.query_pairs().any(|(key, value)| key == "utm_source" && value == "openai") {
                let kept: Vec<(String, String)> = parsed.query_pairs().filter(|(key, _)| key != "utm_source").map(|(key, value)| (key.into_owned(), value.into_owned())).collect();
                if kept.is_empty() {
                    parsed.set_query(None);
                } else {
                    parsed.query_pairs_mut().clear().extend_pairs(kept);
                }
            }
            parsed.to_string()
        }
        Err(_) => raw.strip_suffix("?utm_source=openai").or_else(|| raw.strip_suffix("&utm_source=openai")).unwrap_or(raw).to_string(),
    }
}

// Whether a cut before `index` would split a surrogate pair.
fn splits_pair(units: &[u16], index: usize) -> bool {
    index > 0 && index < units.len() && (0xd800..=0xdbff).contains(&units[index - 1]) && (0xdc00..=0xdfff).contains(&units[index])
}

// extractSnippetAround: the answer text around a citation, its markdown links reduced to their
// text. Offsets are the API's, counted in UTF-16 units as JavaScript counts them. Where pi's cuts
// would split an emoji's surrogate pair and leave half of it, the window widens to keep the whole
// character and the 300-unit cut moves back one, since a Rust string can't hold half a pair.
fn snippet_around(text: &str, start: Option<f64>, end: Option<f64>) -> String {
    let (Some(start), Some(end)) = (start, end) else { return String::new() };
    if text.is_empty() {
        return String::new();
    }
    let units: Vec<u16> = text.encode_utf16().collect();
    let clamp = |index: f64| index.max(0.0).min(units.len() as f64) as usize;
    let mut before = clamp(start - SNIPPET_CONTEXT as f64);
    let mut after = clamp(end + SNIPPET_CONTEXT as f64).max(before);
    if splits_pair(&units, before) {
        before -= 1;
    }
    if splits_pair(&units, after) {
        after += 1;
    }
    let window = String::from_utf16_lossy(&units[before..after]);
    let snippet = crate::js::trim(&MARKDOWN_LINK.replace_all(&window, "$1")).to_string();
    let snippet_units: Vec<u16> = snippet.encode_utf16().collect();
    if snippet_units.len() > MAX_SNIPPET {
        let cut = if splits_pair(&snippet_units, MAX_SNIPPET - 3) { MAX_SNIPPET - 4 } else { MAX_SNIPPET - 3 };
        format!("{}...", String::from_utf16_lossy(&snippet_units[..cut]))
    } else {
        snippet
    }
}

// addResult: a result per distinct cleaned URL, titled by the URL when the source has no title.
fn add_result(results: &mut Vec<SearchResult>, url: Option<&Value>, title: Option<&Value>, snippet: String) {
    let Some(url) = url.and_then(Value::as_str).filter(|url| !url.trim().is_empty()) else { return };
    let url = clean_source_url(url);
    if results.iter().any(|result| result.url == url) {
        return;
    }
    let title = title.and_then(Value::as_str).filter(|title| !title.trim().is_empty()).map(str::to_string).unwrap_or_else(|| url.clone());
    results.push(SearchResult { title, url, snippet });
}

// extractSearchResults: the answer's citations first, with the text around each, then every
// source the searches returned.
fn extract_results(output: &[Value], count: Option<usize>) -> Vec<SearchResult> {
    let mut results = Vec::new();
    for item in output.iter().filter(|item| item.get("type").and_then(Value::as_str) == Some("message")) {
        for part in item.get("content").and_then(Value::as_array).into_iter().flatten() {
            let text = part.get("text").and_then(Value::as_str).unwrap_or_default();
            for annotation in part.get("annotations").and_then(Value::as_array).into_iter().flatten() {
                if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
                    continue;
                }
                let snippet = snippet_around(text, annotation.get("start_index").and_then(Value::as_f64), annotation.get("end_index").and_then(Value::as_f64));
                add_result(&mut results, annotation.get("url"), annotation.get("title"), snippet);
            }
        }
    }
    for item in output.iter().filter(|item| is_web_search_call(item)) {
        let groups = [item.get("action").and_then(|action| action.get("sources")), item.get("sources"), item.get("results")];
        for source in groups.into_iter().flatten().filter_map(Value::as_array).flatten() {
            if source.is_object() {
                add_result(&mut results, source.get("url").or_else(|| source.get("source_website_url")), source.get("title").or_else(|| source.get("caption")), String::new());
            }
        }
    }
    if let Some(count) = count {
        results.truncate(count);
    }
    results
}

// extractAnswer: the message text, parts joined by line breaks.
fn extract_answer(output: &[Value]) -> String {
    let parts: Vec<&str> = output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .flat_map(|item| item.get("content").and_then(Value::as_array).into_iter().flatten())
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .collect();
    crate::js::trim(&parts.join("\n")).to_string()
}

// One search. Input: the targets, where credentials live, the session id (opencode routes by it),
// the query and its options. Output: the cited answer and its sources, or an error with the key
// redacted.
pub async fn search(targets: &[Target], paths: &CredentialPaths, session_id: &str, query: &str, options: &SearchOptions<'_>) -> Result<SearchResponse, String> {
    let (target, credential) = pick(targets, paths).await?;
    let redact = |text: String| if credential.key.is_empty() { text } else { text.replace(&credential.key, "[redacted]") };
    let run = async {
        let response = send(target, &credential, session_id, &request_body(&target.model.id, query, options)).await?;
        let status = response.status().as_u16();
        let text = response.text().await.map_err(|error| if error.is_timeout() { "The operation was aborted.".to_string() } else { error.to_string() })?;
        if !(200..300).contains(&status) {
            return Err(format!("OpenAI API error {status}: {}", text.chars().take(300).collect::<String>()));
        }
        read_response(&text, options)
    };
    run.await.map_err(redact)
}

// A successful response's body as web_search's answer and sources. Input: the body and the
// search's options (for the result count). Output: the answer and results, or why there are none.
pub fn read_response(text: &str, options: &SearchOptions<'_>) -> Result<SearchResponse, String> {
    let (output, seen) = parse_response(text)?;
    if !seen {
        return Err("OpenAI web_search returned no web_search_call".to_string());
    }
    let answer = extract_answer(&output);
    let results = extract_results(&output, requested_count(options));
    if answer.is_empty() && results.is_empty() {
        return Err("OpenAI web_search returned no answer or sources".to_string());
    }
    Ok(SearchResponse { answer, results, provider: "openai" })
}
