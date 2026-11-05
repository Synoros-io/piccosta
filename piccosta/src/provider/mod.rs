// Talking to model providers: one request in, a stream of events and a finished assistant message
// out, whichever wire protocol the model uses.
//
// Each protocol lives in its own module and is picked by the model's `api`:
// - codex: OpenAI's Codex backend on a ChatGPT plan ("openai-codex-responses").
// - openai_responses: the Responses API from other providers, e.g. opencode-go ("openai-responses").
// - openai_completions: Chat Completions, most of opencode-go's models and local servers.
// - claude: Claude through Claude Code on a Claude plan ("claude-bridge").

pub mod anthropic_messages;
pub mod google;
pub mod claude;
pub mod codex;
mod codex_ws;
pub mod estimate;
pub mod openai_completions;
pub mod openai_responses;
pub mod responses;
pub mod sse;
pub mod transcript;
pub mod transform;

use crate::auth::Credential;
use serde_json::Value;
use std::time::Duration;
use crate::clock;
use crate::message::{AssistantMessage, Content, Message, StopReason, Usage};
use crate::model::Model;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// How long to wait for a response's headers, then for each piece of its body: pi leaves the first
// to its HTTP idle setting and the second to undici's 300 s body timeout.
pub const HEADER_TIMEOUT: Duration = Duration::from_secs(300);
pub const BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
// opencode's own host; requests there carry the session id opencode routes by
// (coding-agent/src/core/provider-attribution.ts:9, 67-77). Without it opencode answers 400 MissingSessionID.
const OPENCODE_HOST: &str = "opencode.ai";

// pi's thinking levels, lowest to highest (models.ts:1258).
pub const THINKING_LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

// The thinking levels a model accepts as they are (pi's getAvailableThinkingLevels): those its
// clamp leaves alone.
pub fn thinking_levels(model: &crate::model::Model) -> Vec<&'static str> {
    THINKING_LEVELS.iter().copied().filter(|level| clamp_thinking_level(model, level) == *level).collect()
}

// What happens while a response streams in. Indexes point into the assistant message's content.
// The finished message itself is the request's return value. Each end carries the finished block
// and a tool call's start its id and name, as pi's events do, so a listener needn't rebuild the
// message from deltas.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    Start,
    TextStart { index: usize },
    TextDelta { index: usize, delta: String },
    TextEnd { index: usize, content: String },
    ThinkingStart { index: usize },
    ThinkingDelta { index: usize, delta: String },
    ThinkingEnd { index: usize, content: String },
    ToolCallStart { index: usize, id: String, name: String },
    ToolCallDelta { index: usize, delta: String },
    ToolCallEnd { index: usize, tool_call: Content },
}

// One request to a model.
pub struct Request<'a> {
    pub model: &'a Model,
    // The conversation, leading with the system message that holds the prompt and tools.
    pub messages: &'a [Message],
    pub credential: &'a Credential,
    // The session id, which providers use to keep the prompt cache and route to the same backend.
    // None skips caching.
    pub session_id: Option<&'a str>,
    // False for pi's cacheRetention "none" (a one-off summary): no cache key and no cache-affinity
    // headers, though a provider that routes every request by session (opencode) still gets the id.
    pub cache: bool,
    // pi's `transport` setting, for an API that has a choice (Codex): "auto" (a websocket that
    // sends only new input when it can, SSE when it can't be used), "websocket" (always the full
    // input), "websocket-cached", or "sse".
    pub transport: &'a str,
    // pi's thinking level for this response, e.g. "medium".
    pub thinking_level: &'a str,
    // The session's file, when it is saved; the Claude bridge keeps its records next to it.
    pub session_file: Option<&'a std::path::Path>,
    // The directory the session works in, which the Claude bridge starts Claude Code in (it tells
    // the model that directory, and keys its records by it).
    pub cwd: &'a std::path::Path,
    pub cancel: CancellationToken,
}

// The levels a model supports (models.ts:1260-1269): "off" alone for a model that doesn't reason;
// otherwise every level its map doesn't mark null, with xhigh and max only when the map names them.
pub fn supported_thinking_levels(model: &Model) -> Vec<&'static str> {
    if !model.reasoning {
        return vec!["off"];
    }
    THINKING_LEVELS
        .into_iter()
        .filter(|level| {
            let mapped = model.thinking_level_map.as_ref().and_then(|map| map.get(*level));
            match mapped {
                Some(None) => false,
                None if *level == "xhigh" || *level == "max" => false,
                _ => true,
            }
        })
        .collect()
}

// The nearest level the model supports (models.ts:1271-1290): the requested one, else the next
// one up, else the next one down.
pub fn clamp_thinking_level(model: &Model, level: &str) -> &'static str {
    let available = supported_thinking_levels(model);
    if let Some(found) = available.iter().find(|candidate| **candidate == level) {
        return found;
    }
    let Some(requested) = THINKING_LEVELS.iter().position(|candidate| *candidate == level) else {
        return available.first().copied().unwrap_or("off");
    };
    THINKING_LEVELS[requested..]
        .iter()
        .chain(THINKING_LEVELS[..requested].iter().rev())
        .find(|candidate| available.contains(candidate))
        .copied()
        .unwrap_or("off")
}

pub fn user_agent() -> String {
    format!("piccosta/{} ({}; {})", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH)
}

// What piccosta calls itself to a service that asks which client sent a request: Codex's
// `originator` and opencode's `x-opencode-client`. The agent dir's `clientName` replaces it, for a
// service that stops taking this one; it's the install's name, not a project's, so it's read once.
const CLIENT_NAME: &str = "piccosta";

pub fn client_name() -> &'static str {
    static NAME: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::config::Settings::load(&crate::config::agent_dir(), None).get("clientName").and_then(serde_json::Value::as_str).unwrap_or(CLIENT_NAME).to_string()
    });
    &NAME
}

// How many redirects a request may follow: reqwest's own default (redirect::Policy::default).
const MAX_REDIRECTS: usize = 10;

// The client every request with a key starts from (tools/web/net.rs checks its own hops). A
// redirect is followed only on the origin first asked of, the same scheme, host and port: reqwest
// drops Authorization on a hop elsewhere but keeps x-api-key, x-goog-api-key and a provider's own
// headers, so a 307 to another origin would carry the key there.
pub fn http_client() -> reqwest::ClientBuilder {
    let policy = reqwest::redirect::Policy::custom(|attempt| {
        let (from, to) = (attempt.previous()[0].origin(), attempt.url().origin());
        if from != to {
            let refused = format!("refused a redirect from {} to {}", from.ascii_serialization(), to.ascii_serialization());
            attempt.error(refused)
        } else if attempt.previous().len() > MAX_REDIRECTS {
            attempt.error("too many redirects")
        } else {
            attempt.follow()
        }
    });
    reqwest::Client::builder().user_agent(user_agent()).redirect(policy)
}

// Whether a model is served by opencode, which wants its client and session headers.
pub fn is_opencode(model: &Model) -> bool {
    let host = model.base_url.split_once("://").map_or(model.base_url.as_str(), |(_, rest)| rest).split(['/', ':', '?']).next().unwrap_or_default();
    model.provider == "opencode" || model.provider == "opencode-go" || host == OPENCODE_HOST
}

// reqwest's display text leaves out the connection failure. Keep its causes so the retry rules
// can see a refused connection, DNS failure or reset, including one while reading the body.
pub fn http_error(error: reqwest::Error) -> String {
    use std::error::Error;
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    message
}

// POST a streaming request to an OpenAI-style API (openai-responses, openai-completions) with
// bearer auth, opencode's client header, the model's own headers, `extra`, then opencode's
// session, and wait for the response headers. Output: the response when its status is a
// success, else the status and the server's message (status first, as the openai SDK writes it,
// so retry rules can see a 5xx).
pub async fn post_openai(model: &Model, url: String, credential: &Credential, session_id: Option<&str>, extra: Vec<(&str, String)>, body: &Value) -> Result<reqwest::Response, String> {
    credential.check_origin(&url)?;
    let opencode = is_opencode(model);
    let mut http = http_client()
        .build()
        .map_err(http_error)?
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream");
    // A server that wants no key (auth.rs) gets no Authorization header at all.
    if !credential.key.is_empty() {
        http = http.header("Authorization", format!("Bearer {}", credential.key));
    }
    // opencode attributes requests to the client that sent them (identity.ts:96-109).
    if opencode {
        http = http.header("x-opencode-client", client_name());
    }
    for (name, value) in model.headers.iter().flatten() {
        http = http.header(name, value);
    }
    for (name, value) in extra {
        http = http.header(name, value);
    }
    // opencode routes every request by session, cached or not (opencode-headers.ts).
    if let Some(session) = session_id.filter(|_| opencode) {
        http = http.header("x-opencode-session", session);
    }
    let response = tokio::time::timeout(HEADER_TIMEOUT, http.body(body.to_string()).send())
        .await
        .map_err(|_| format!("{} API error: response headers timed out after {} s", model.provider, HEADER_TIMEOUT.as_secs()))?
        .map_err(|error| format!("{} API error: {}", model.provider, http_error(error)))?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let retry_after = response.headers().get("retry-after").and_then(|value| value.to_str().ok()?.trim().parse::<f64>().ok());
    let text = response.text().await.unwrap_or_default();
    // opencode refuses a Go plan past one of its limits (5 hour, weekly or monthly) with a
    // GoUsageLimitError naming it in metadata.limitName and when to come back in Retry-After, in
    // seconds (opencode's packages/core/src/session/to-session-error.ts). That's the only reading
    // Go gives: its successful responses carry no usage headers.
    if opencode && text.contains("GoUsageLimitError") {
        let limit = serde_json::from_str::<Value>(&text).ok().and_then(|json| json.pointer("/metadata/limitName").or(json.pointer("/error/metadata/limitName"))?.as_str().map(str::to_string));
        let resets_at = retry_after.map(|seconds| (clock::now_ms() / 1000) as f64 + seconds);
        crate::scale::record(&model.provider, limit.as_deref().unwrap_or("limit"), 100.0, resets_at);
    }
    let detail = serde_json::from_str::<Value>(&text).ok().and_then(|json| json.pointer("/error/message").and_then(Value::as_str).map(str::to_string)).unwrap_or(text);
    Err(format!("{} {detail}", status.as_u16()).trim().to_string())
}

// A provider's stream, run until it ends or the user stops it, which fails it.
pub async fn abortable(request: &Request<'_>, run: impl std::future::Future<Output = Result<(), String>>) -> Result<(), String> {
    tokio::select! {
        result = run => result,
        () = request.cancel.cancelled() => Err("Request was aborted".to_string()),
    }
}

// How a stream ended, on the message it built: a failure stops it with its reason, Aborted when
// the user stopped it, else Error.
pub fn settle(request: &Request<'_>, output: &mut AssistantMessage, result: Result<(), String>) {
    if let Err(message) = result {
        output.stop_reason = if request.cancel.is_cancelled() { StopReason::Aborted } else { StopReason::Error };
        output.error_message = Some(message);
    }
}

// An empty assistant message for a response that's about to start.
pub fn new_output(model: &Model, api: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: api.to_string(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        thinking_level: None,
        usage: Usage::default(),
        stop_reason: StopReason::Pending,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: clock::now_ms(),
        duration_ms: None,
    }
}

// Send one request and stream the response. Input: the request, and where events go. Output: the
// finished assistant message. A failure is not an Err: it's a message with stop reason Error (or
// Aborted when cancelled) and the reason in error_message, as pi records it.
pub async fn stream(request: Request<'_>, events: mpsc::UnboundedSender<StreamEvent>) -> AssistantMessage {
    let started = std::time::Instant::now();
    if let Err(error) = request.credential.check_origin(&request.model.base_url) {
        let mut output = new_output(request.model, &request.model.api);
        settle(&request, &mut output, Err(error));
        return output;
    }
    let mut message = match request.model.api.as_str() {
        "openai-codex-responses" => codex::stream(&request, &events).await,
        "openai-responses" => openai_responses::stream(&request, &events).await,
        "openai-completions" => openai_completions::stream(&request, &events).await,
        "anthropic-messages" => anthropic_messages::stream(&request, &events).await,
        "google-generative-ai" => google::stream(&request, &events).await,
        claude::PROVIDER => claude::stream(&request, &events).await,
        other => {
            let mut output = new_output(request.model, other);
            output.stop_reason = StopReason::Error;
            output.error_message = Some(format!("piccosta doesn't speak the \"{other}\" API yet"));
            output
        }
    };
    message.thinking_level = Some(request.thinking_level.to_string());
    message.duration_ms = Some(started.elapsed().as_millis() as f64);
    message
}
