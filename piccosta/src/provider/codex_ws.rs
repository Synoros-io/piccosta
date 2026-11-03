// Codex over a websocket: one connection per session, kept open between requests, and once a
// response has finished only the new input goes up, continuing from that response's id. The SSE
// path in codex.rs is the fallback.
//
// Based on pi 1.4.0 (MIT, see LICENSE.md): packages/ai/src/api/openai-codex-responses.ts
// (connectWebSocket, acquireWebSocket, parseWebSocket, processWebSocketStream, and the delta
// continuation helpers). The frames, headers, reuse rules and error texts are pi's. Different:
// - an idle socket isn't read between requests, so a close the server sent while it sat idle is
//   found when it's next taken (pi's readyState check), and the 5-minute idle limit is applied
//   then too, rather than by a timer;
// - each event may take at most the SSE body's idle limit (300 s); pi has no limit by default;
// - an HTTP proxy in the environment, as reqwest reads it on the SSE path, is gone through with a
//   CONNECT tunnel; any other kind of proxy fails the connection, which moves the session to SSE;
// - no debug counters.

use super::responses::{self, ConvertOptions, StreamState};
use super::StreamEvent;
use crate::message::{AssistantMessage, Message};
use crate::model::Model;
use futures_util::{FutureExt, SinkExt, StreamExt};
use hyper_util::client::proxy::matcher::{Intercept, Matcher};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::{Error as WsError, Message as WsMessage};
use tokio_tungstenite::WebSocketStream;

// The beta header the websocket endpoint wants (openai-codex-responses.ts:885).
pub(super) const OPENAI_BETA: &str = "responses_websockets=2026-02-06";
// A cached socket is closed after 5 minutes unused, or once it's 55 minutes old (lines 886-887).
const IDLE_LIMIT: Duration = Duration::from_secs(5 * 60);
const MAX_AGE: Duration = Duration::from_secs(55 * 60);
// How long opening a socket may take (DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS, line 61).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
// The longest wait for one event: the SSE body's idle limit (codex.rs).
const EVENT_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
// The close code a server sends for a frame over its size limit (line 66).
const MESSAGE_TOO_BIG: u16 = 1009;
// The most of a proxy's answer to CONNECT read before giving up on it: its status line and headers
// are a few hundred bytes.
const MAX_PROXY_REPLY: usize = 8192;
// The tool-call providers Codex keeps two-part ids for, as in codex.rs.
const TOOL_CALL_PROVIDERS: [&str; 3] = ["openai", "openai-codex", "opencode"];

// Anything a socket can run over: plain TCP for ws://, TLS for wss://.
trait Io: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Io for T {}
type Socket = WebSocketStream<Box<dyn Io>>;

// What the last response on a socket leaves for the next request to continue from
// (CachedWebSocketContinuationState): the full request minus its input, as JSON, and the input
// the server now holds, which is that request's input followed by the response's own items, each
// item as JSON.
struct Continuation {
    body_without_input: String,
    baseline: Vec<String>,
    response_id: String,
}

struct Cached {
    socket: Socket,
    created: Instant,
    idle_since: Instant,
    continuation: Option<Continuation>,
}

// A cache slot: in use by a request, or waiting for the next one.
enum Slot {
    Busy,
    Idle(Box<Cached>),
}

type Key = (String, String);

// Sockets by session id and account id, and the sessions that fell back to SSE for the rest of
// the process (websocketSessionCache, websocketSseFallbackSessions).
static SOCKETS: LazyLock<Mutex<HashMap<Key, Slot>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static SSE_FALLBACK: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

// Whether a session already fell back to SSE, which then holds for the rest of the process.
pub fn fallback_active(session: Option<&str>) -> bool {
    session.is_some_and(|session| SSE_FALLBACK.lock().is_ok_and(|sessions| sessions.contains(session)))
}

pub fn mark_fallback(session: Option<&str>) {
    if let (Some(session), Ok(mut sessions)) = (session, SSE_FALLBACK.lock()) {
        sessions.insert(session.to_string());
    }
}

// Why a websocket attempt failed, sorted the way the retry and fallback rules need: Codex's own
// error events (with their code), malformed events, and everything about the connection itself.
#[derive(Debug)]
pub enum Failure {
    Api { message: String, code: Option<String> },
    Protocol(String),
    Transport(String),
}

// The socket a request runs on. A cached one goes back in the cache when the request finishes
// cleanly; dropping the lease any other way (an error, an abort that drops the request) closes the
// socket and frees its slot.
struct Lease {
    key: Option<Key>,
    socket: Option<Socket>,
    created: Instant,
    continuation: Option<Continuation>,
}

impl Lease {
    fn socket(&mut self) -> &mut Socket {
        self.socket.as_mut().expect("a lease holds its socket until it's released")
    }

    // Done with the socket. `keep` puts a cached socket back for the next request.
    fn release(mut self, keep: bool) {
        let Some(key) = self.key.take() else { return };
        let Some(socket) = self.socket.take() else { return };
        if let Ok(mut sockets) = SOCKETS.lock() {
            if keep {
                let cached = Cached { socket, created: self.created, idle_since: Instant::now(), continuation: self.continuation.take() };
                sockets.insert(key, Slot::Idle(Box::new(cached)));
            } else {
                sockets.remove(&key);
            }
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if let (Some(key), Ok(mut sockets)) = (self.key.take(), SOCKETS.lock())
            && matches!(sockets.get(&key), Some(Slot::Busy))
        {
            sockets.remove(&key);
        }
    }
}

// Whether a socket that sat unread is still open: read whatever already arrived without waiting.
// Pings are answered as they're read; a close, an error, or data nobody asked for means it's done.
fn still_open(socket: &mut Socket) -> bool {
    loop {
        match socket.next().now_or_never() {
            None => return true,
            Some(Some(Ok(WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_)))) => continue,
            Some(_) => return false,
        }
    }
}

// The TLS settings for wss://: the same provider and the platform's verifier, as reqwest uses.
fn tls_config() -> Result<Arc<rustls::ClientConfig>, String> {
    use rustls_platform_verifier::BuilderVerifierExt;
    static CONFIG: LazyLock<Result<Arc<rustls::ClientConfig>, String>> = LazyLock::new(|| {
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .and_then(|builder| builder.with_platform_verifier())
            .map(|builder| Arc::new(builder.with_no_client_auth()))
            .map_err(|error| error.to_string())
    });
    CONFIG.clone()
}

// The text pi gives a websocket error (extractWebSocketError): Node's ws library words a refused
// upgrade as "Unexpected server response: <status>".
fn error_text(error: &WsError) -> String {
    match error {
        WsError::Http(response) => format!("Unexpected server response: {}", response.status().as_u16()),
        WsError::ConnectionClosed | WsError::AlreadyClosed => "WebSocket closed 1006".to_string(),
        WsError::Protocol(_) => "WebSocket closed 1006".to_string(),
        other => other.to_string(),
    }
}

// The text pi gives a close (extractWebSocketCloseError): "WebSocket closed <code> <reason>".
fn close_text(code: u16, reason: &str) -> String {
    let reason = if reason.is_empty() && code == MESSAGE_TOO_BIG { "message too big" } else { reason };
    format!("WebSocket closed {code} {reason}").trim().to_string()
}

// The proxy reqwest would send this host's requests through, read by reqwest's own matcher
// (hyper-util's): $HTTPS_PROXY (for wss://) or $HTTP_PROXY, else $ALL_PROXY, unless $NO_PROXY
// names the host, a domain it's in (any case, with or without a leading dot), its IP or a network
// holding it, or is *. The proxy's user and password come percent-decoded, as reqwest sends them.
fn proxy_for(host: &str, port: u16, secure: bool) -> Option<Intercept> {
    let uri = format!("{}://{host}:{port}", if secure { "https" } else { "http" }).parse().ok()?;
    Matcher::from_system().intercept(&uri)
}

// A TCP connection to host:port through an HTTP proxy's CONNECT tunnel, with the proxy url's user
// and password as basic auth. Output: the tunnel, or why the proxy wouldn't give one.
async fn tunnel(proxy: &Intercept, host: &str, port: u16) -> Result<tokio::net::TcpStream, String> {
    let url = proxy.uri();
    if url.scheme_str() != Some("http") {
        return Err(format!("the websocket can't go through a {} proxy, so this session uses SSE", url.scheme_str().unwrap_or_default()));
    }
    let address = (url.host().unwrap_or_default(), url.port_u16().unwrap_or(80));
    let mut tcp = tokio::net::TcpStream::connect(address).await.map_err(|error| format!("proxy {}: {error}", address.0))?;
    let mut connect = format!("CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n");
    if let Some(auth) = proxy.basic_auth().and_then(|auth| auth.to_str().ok()) {
        connect.push_str(&format!("Proxy-Authorization: {auth}\r\n"));
    }
    connect.push_str("\r\n");
    tcp.write_all(connect.as_bytes()).await.map_err(|error| error.to_string())?;
    // The answer up to its blank line, a byte at a time so nothing past it (the server's bytes)
    // is taken from the stream. Anything but a 200 means no tunnel.
    let mut reply = Vec::new();
    let mut byte = [0u8; 1];
    while !reply.ends_with(b"\r\n\r\n") && reply.len() < MAX_PROXY_REPLY && tcp.read(&mut byte).await.map_err(|error| error.to_string())? == 1 {
        reply.push(byte[0]);
    }
    let status = String::from_utf8_lossy(&reply).lines().next().unwrap_or_default().to_string();
    if status.split_whitespace().nth(1) != Some("200") {
        return Err(format!("proxy {} refused CONNECT: {status}", address.0));
    }
    Ok(tcp)
}

// Open a socket (connectWebSocket). Input: the ws:// or wss:// url and the headers in order.
async fn connect(url: &str, headers: &[(String, String)]) -> Result<Socket, Failure> {
    let transport = |message: String| Failure::Transport(message);
    let mut request = url.into_client_request().map_err(|error| transport(error.to_string()))?;
    for (name, value) in headers {
        let name = tokio_tungstenite::tungstenite::http::HeaderName::from_bytes(name.as_bytes()).map_err(|error| transport(error.to_string()))?;
        let value = HeaderValue::from_str(value).map_err(|error| transport(error.to_string()))?;
        request.headers_mut().insert(name, value);
    }
    let uri = request.uri().clone();
    let secure = uri.scheme_str() == Some("wss");
    let host = uri.host().ok_or_else(|| transport("WebSocket url has no host".into()))?.to_string();
    let port = uri.port_u16().unwrap_or(if secure { 443 } else { 80 });
    let opening = async {
        let tcp = match proxy_for(&host, port, secure) {
            Some(proxy) => tunnel(&proxy, &host, port).await.map_err(transport)?,
            None => tokio::net::TcpStream::connect((host.as_str(), port)).await.map_err(|error| transport(error.to_string()))?,
        };
        let _ = tcp.set_nodelay(true);
        let io: Box<dyn Io> = if secure {
            let config = tls_config().map_err(transport)?;
            let name = rustls::pki_types::ServerName::try_from(host.clone()).map_err(|error| transport(error.to_string()))?;
            Box::new(tokio_rustls::TlsConnector::from(config).connect(name, tcp).await.map_err(|error| transport(error.to_string()))?)
        } else {
            Box::new(tcp)
        };
        let (socket, response) = tokio_tungstenite::client_async(request, io).await.map_err(|error| transport(error_text(&error)))?;
        super::codex::record_usage(|name| response.headers().get(name).and_then(|value| value.to_str().ok()).map(str::to_string));
        Ok(socket)
    };
    tokio::time::timeout(CONNECT_TIMEOUT, opening)
        .await
        .map_err(|_| transport(format!("WebSocket connect timeout after {}ms", CONNECT_TIMEOUT.as_millis())))?
}

// A socket for one request (acquireWebSocket). Without a session it's a fresh one, closed after.
// With one, the session's idle socket is reused when it's still open and young enough; when the
// session's socket is busy (another request in flight), this one gets a socket of its own.
async fn acquire(url: &str, headers: &[(String, String)], key: Option<Key>) -> Result<Lease, Failure> {
    let Some(key) = key else {
        let socket = connect(url, headers).await?;
        return Ok(Lease { key: None, socket: Some(socket), created: Instant::now(), continuation: None });
    };
    let taken = {
        let mut sockets = SOCKETS.lock().map_err(|_| Failure::Transport("websocket cache poisoned".into()))?;
        match sockets.remove(&key) {
            Some(Slot::Busy) => {
                sockets.insert(key.clone(), Slot::Busy);
                None
            }
            Some(Slot::Idle(cached)) => {
                sockets.insert(key.clone(), Slot::Busy);
                Some(Some(cached))
            }
            None => {
                sockets.insert(key.clone(), Slot::Busy);
                Some(None)
            }
        }
    };
    match taken {
        // Busy: a throwaway socket, not cached.
        None => {
            let socket = connect(url, headers).await?;
            Ok(Lease { key: None, socket: Some(socket), created: Instant::now(), continuation: None })
        }
        Some(cached) => {
            // From here the slot is ours; the lease frees it if connecting fails.
            let mut lease = Lease { key: Some(key), socket: None, created: Instant::now(), continuation: None };
            if let Some(mut cached) = cached {
                let fresh = cached.created.elapsed() < MAX_AGE && cached.idle_since.elapsed() < IDLE_LIMIT;
                if fresh && still_open(&mut cached.socket) {
                    let cached = *cached;
                    lease.socket = Some(cached.socket);
                    lease.created = cached.created;
                    lease.continuation = cached.continuation;
                    return Ok(lease);
                }
                let _ = cached.socket.close(None).await;
            }
            lease.socket = Some(connect(url, headers).await?);
            Ok(lease)
        }
    }
}

// Each item of a JSON array, as JSON text, for comparing lists the way pi compares them
// (JSON.stringify of each side).
fn item_texts(items: &[Value]) -> Vec<String> {
    items.iter().map(Value::to_string).collect()
}

fn without_input(body: &Map<String, Value>) -> String {
    let mut rest = body.clone();
    rest.shift_remove("input");
    rest.shift_remove("previous_response_id");
    Value::Object(rest).to_string()
}

// The request to send on a socket that has a continuation (buildCachedWebSocketRequestBody):
// when only the input changed and the new input starts with what the server already holds, just
// the new items and the last response's id. Otherwise the full body, and the continuation is
// dropped.
fn continued_body(continuation: &mut Option<Continuation>, body: &Map<String, Value>) -> Map<String, Value> {
    let Some(state) = continuation.as_ref() else { return body.clone() };
    let input = body.get("input").and_then(Value::as_array).cloned().unwrap_or_default();
    let matches = without_input(body) == state.body_without_input
        && input.len() >= state.baseline.len()
        && item_texts(&input[..state.baseline.len()]) == state.baseline;
    if !matches || state.response_id.is_empty() {
        *continuation = None;
        return body.clone();
    }
    let mut continued = body.clone();
    continued.insert("previous_response_id".into(), Value::String(state.response_id.clone()));
    continued.insert("input".into(), Value::Array(input[state.baseline.len()..].to_vec()));
    continued
}

// What a finished response leaves for the next request: the server holds the request's input and
// the response's own items (its output converted back to input, less any tool outputs).
fn continuation_after(model: &Model, body: &Map<String, Value>, output: &AssistantMessage) -> Option<Continuation> {
    let response_id = output.response_id.clone()?;
    let options = ConvertOptions {
        include_system_prompt: false,
        tool_call_providers: &TOOL_CALL_PROVIDERS,
        default_strict: None,
        supports_strict: super::codex::supports_strict(model),
    };
    let items: Vec<Value> = responses::convert_messages(model, &[Message::Assistant(output.clone())], &options)
        .into_iter()
        .filter(|item| !matches!(item.get("type").and_then(Value::as_str), Some("function_call_output" | "custom_tool_call_output")))
        .collect();
    let mut baseline = item_texts(body.get("input").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default());
    baseline.extend(item_texts(&items));
    Some(Continuation { body_without_input: without_input(body), baseline, response_id })
}

// What one websocket request runs with.
pub struct Target<'a> {
    pub url: String,
    pub headers: Vec<(String, String)>,
    // The session the socket is cached under (with the account), when the request is cached.
    pub session: Option<&'a str>,
    pub account_id: &'a str,
    // pi's "auto" and "websocket-cached" transports send only the new input when they can.
    pub continue_responses: bool,
}

// One request over a websocket (processWebSocketStream). `started` turns true at the first event
// that isn't an error, when the response begins; the caller's retry and fallback rules depend on
// whether it had.
pub async fn stream(
    target: &Target<'_>,
    body: &Map<String, Value>,
    model: &Model,
    state: &mut StreamState<'_>,
    send: &mut dyn FnMut(StreamEvent),
    started: &mut bool,
) -> Result<(), Failure> {
    let key = target.session.map(|session| (session.to_string(), target.account_id.to_string()));
    let mut lease = acquire(&target.url, &target.headers, key).await?;
    let cached = lease.key.is_some();
    let request_body = if target.continue_responses && cached { continued_body(&mut lease.continuation, body) } else { body.clone() };
    let mut frame = Map::new();
    frame.insert("type".into(), Value::String("response.create".into()));
    frame.extend(request_body);
    lease
        .socket()
        .send(WsMessage::Text(Value::Object(frame).to_string().into()))
        .await
        .map_err(|error| Failure::Transport(error_text(&error)))?;

    loop {
        let next = tokio::time::timeout(EVENT_IDLE_TIMEOUT, lease.socket().next())
            .await
            .map_err(|_| Failure::Transport(format!("WebSocket idle timeout after {}ms", EVENT_IDLE_TIMEOUT.as_millis())))?;
        let text = match next {
            None => return Err(Failure::Transport("WebSocket closed 1006".into())),
            Some(Err(error)) => return Err(Failure::Transport(error_text(&error))),
            Some(Ok(WsMessage::Text(text))) => text.to_string(),
            Some(Ok(WsMessage::Binary(bytes))) => String::from_utf8_lossy(&bytes).into_owned(),
            Some(Ok(WsMessage::Close(frame))) => {
                let (code, reason) = frame.map(|frame| (u16::from(frame.code), frame.reason.to_string())).unwrap_or((u16::from(CloseCode::Status), String::new()));
                return Err(Failure::Transport(close_text(code, &reason)));
            }
            Some(Ok(_)) => continue,
        };
        let event: Value = serde_json::from_str(&text).map_err(|error| Failure::Protocol(format!("Invalid Codex WebSocket JSON: {error}")))?;
        // A usage reading isn't part of the response, so it doesn't start it.
        if event.get("type").and_then(Value::as_str).unwrap_or_default().is_empty() || super::codex::record_event_usage(&event) {
            continue;
        }
        if let Some((message, code)) = super::codex::api_error(&event) {
            return Err(Failure::Api { message, code });
        }
        if !*started {
            *started = true;
            send(StreamEvent::Start);
        }
        let terminal = super::codex::handle_codex_event(event, state, send).map_err(Failure::Protocol)?;
        if terminal {
            break;
        }
    }

    // The response is done and the socket stays open for the next one.
    if target.continue_responses && cached {
        lease.continuation = continuation_after(model, body, &state.output);
    }
    lease.release(true);
    Ok(())
}
