// The search engines web_search asks: the user's own SearXNG when one is configured, else a
// model on the user's own account through its provider's search tool (openai_search.rs).
//
// Based on pi-web-access 0.44.0 (MIT, see LICENSE.md): searxng.ts,
// domain-filter-normalization.ts and the shape of the "auto" route in gemini-search.ts. SearXNG is
// AGPL-3.0; it runs as its own service, which piccosta never ships, and we only speak HTTP to it.
// - Measured 2026-10-29 from this workstation (ledger.md):
//   SearXNG put the expected page first for 29 of 30 known-item queries, median 825 ms;
//   gpt-6-luna's web_search put it in the top three for all 30, median 6.4 s. DuckDuckGo's HTML
//   page, pi-web-access's keyless fallback, answered 0 of 49, bot-checked (HTTP 202) or held until
//   the 30 s timeout, so it's gone.
// - One engine per search, never a chain: a configured SearXNG is the user's choice, and a
//   failure is reported as it is.
// - The SearXNG request goes straight to the configured URL without the private-address check:
//   it's the user's own endpoint (often on the LAN), not a URL a model chose.
// - SearXNG answers given as objects ({"answer": ...}, newer SearXNG) are read as well as
//   strings.

use super::net;
use super::openai_search;
use crate::auth::CredentialPaths;
use reqwest::header::ACCEPT;
use serde_json::Value;
use std::time::Duration;
use url::Url;

// searxng.ts:9.
const SEARCH_TIMEOUT: Duration = Duration::from_secs(30);
// normalizeSearchResultCount: 5 when unset, 1 to 20.
const DEFAULT_RESULTS: usize = 5;
const MAX_RESULTS: usize = 20;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub struct SearchResponse {
    pub answer: String,
    pub results: Vec<SearchResult>,
    pub provider: &'static str,
}

pub struct SearchOptions<'a> {
    pub num_results: Option<f64>,
    pub recency: Option<&'a str>,
    pub domain_filter: &'a [String],
}

pub fn result_count(value: Option<f64>) -> usize {
    match value {
        Some(count) if count.is_finite() => (count.floor().clamp(1.0, MAX_RESULTS as f64)) as usize,
        _ => DEFAULT_RESULTS,
    }
}

// A domain filter entry as a bare hostname, or None when it isn't one
// (domain-filter-normalization.ts).
fn normalize_domain(value: &str) -> Option<String> {
    let mut input = crate::js::trim(value).to_lowercase();
    if let Some(rest) = input.strip_prefix('-') {
        input = crate::js::trim(rest).to_string();
    }
    if input.is_empty() {
        return None;
    }
    let with_scheme = if input.contains("://") { input.clone() } else { format!("https://{input}") };
    let host = match Url::parse(&with_scheme) {
        Ok(parsed) => parsed.host_str().unwrap_or("").to_string(),
        Err(_) => input.split('/').next().unwrap_or("").split(':').next().unwrap_or("").to_string(),
    };
    let host = host.trim_matches('.').to_string();
    let labels_ok = host.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && host.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        && host.rsplit('.').next().is_some_and(|tld| tld.len() >= 2 && tld.chars().all(|c| c.is_ascii_alphabetic()))
        && host.contains('.');
    labels_ok.then_some(host)
}

#[derive(Default)]
pub struct DomainFilters {
    pub allowed: Vec<String>,
    pub blocked: Vec<String>,
}

pub fn domain_filters(entries: &[String]) -> DomainFilters {
    let mut filters = DomainFilters::default();
    for raw in entries {
        let Some(domain) = normalize_domain(raw) else { continue };
        let target = if crate::js::trim(raw).starts_with('-') { &mut filters.blocked } else { &mut filters.allowed };
        if !target.contains(&domain) {
            target.push(domain);
        }
    }
    filters
}

fn host_matches(hostname: &str, domain: &str) -> bool {
    hostname == domain || hostname.ends_with(&format!(".{domain}"))
}

fn passes(url: &str, filters: &DomainFilters) -> bool {
    if filters.allowed.is_empty() && filters.blocked.is_empty() {
        return true;
    }
    let Some(hostname) = Url::parse(url).ok().and_then(|parsed| parsed.host_str().map(str::to_lowercase)) else { return false };
    if !filters.allowed.is_empty() && !filters.allowed.iter().any(|domain| host_matches(&hostname, domain)) {
        return false;
    }
    !filters.blocked.iter().any(|domain| host_matches(&hostname, domain))
}

// SearXNG's provider answer: each result's snippet with its source.
fn snippet_answer(results: &[SearchResult]) -> Vec<String> {
    results
        .iter()
        .map(|result| if result.snippet.is_empty() { format!("Source: {} ({})", result.title, result.url) } else { format!("{}\nSource: {} ({})", result.snippet, result.title, result.url) })
        .collect()
}

// searxng.ts searchWithSearXNG.
async fn search_searxng(base_url: &str, query: &str, options: &SearchOptions<'_>) -> Result<SearchResponse, String> {
    let count = result_count(options.num_results);
    let filters = domain_filters(options.domain_filter);
    let mut parts = vec![query.to_string()];
    match filters.allowed.len() {
        0 => {}
        1 => parts.push(format!("site:{}", filters.allowed[0])),
        _ => parts.push(filters.allowed.iter().map(|domain| format!("site:{domain}")).collect::<Vec<_>>().join(" OR ")),
    }
    parts.extend(filters.blocked.iter().map(|domain| format!("-site:{domain}")));
    let mut url = Url::parse(&format!("{base_url}/search")).map_err(|_| "SearXNG base URL is invalid or missing. Set web.searxngBaseUrl in settings.json or SEARXNG_BASE_URL to an HTTP(S) URL".to_string())?;
    url.query_pairs_mut().append_pair("q", &parts.join(" ")).append_pair("format", "json");
    if let Some(range) = options.recency {
        url.query_pairs_mut().append_pair("time_range", range);
    }
    let client = reqwest::Client::builder().timeout(SEARCH_TIMEOUT).build().map_err(|error| error.to_string())?;
    let response = client.get(url).header(ACCEPT, "application/json").send().await.map_err(|error| net::error_text(&error))?;
    let status = response.status().as_u16();
    if !response.status().is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("SearXNG search error {status}: {}", body.chars().take(300).collect::<String>()));
    }
    let data: Value = response.json().await.map_err(|error| format!("SearXNG returned invalid JSON: {error}"))?;
    let mut results = Vec::new();
    for item in data.get("results").and_then(Value::as_array).into_iter().flatten() {
        let Some(item_url) = item.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()) else { continue };
        if !passes(item_url, &filters) {
            continue;
        }
        let title = item.get("title").and_then(Value::as_str).filter(|t| !t.is_empty()).unwrap_or(item_url);
        let snippet = item.get("content").and_then(Value::as_str).unwrap_or("");
        results.push(SearchResult { title: title.to_string(), url: item_url.to_string(), snippet: snippet.to_string() });
        if results.len() >= count {
            break;
        }
    }
    let mut answer: Vec<String> = data
        .get("answers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|answer| answer.as_str().or_else(|| answer.get("answer").and_then(Value::as_str)))
        .map(crate::js::trim)
        .filter(|answer| !answer.is_empty())
        .map(str::to_string)
        .collect();
    answer.extend(snippet_answer(&results));
    Ok(SearchResponse { answer: answer.join("\n\n"), results, provider: "searxng" })
}

// The auto route: SearXNG when configured, else the search model. A response with neither results
// nor an answer is returned as it is, empty (gemini-search.ts:690-753 keeps the first empty answer
// when nothing better comes).
pub async fn search(searxng_url: Option<&str>, targets: &Result<Vec<openai_search::Target>, String>, paths: &CredentialPaths, session_id: &str, query: &str, options: &SearchOptions<'_>) -> Result<SearchResponse, String> {
    let (label, found) = match (searxng_url, targets) {
        (Some(base), _) => ("SearXNG", search_searxng(base, query, options).await),
        (None, Ok(targets)) => ("OpenAI", openai_search::search(targets, paths, session_id, query, options).await),
        (None, Err(error)) => ("OpenAI", Err(error.clone())),
    };
    found.map_err(|error| format!("Auto provider search failed:\n  - {label}: {error}"))
}
