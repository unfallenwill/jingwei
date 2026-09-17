//! The streamable HTTP transport: one endpoint, one POST per message, and a
//! session the server hands out.
//!
//! The server may answer a request in one of two shapes — a single JSON object,
//! or an event stream that carries messages until it carries the answer — and a
//! client that supports one of them only is a client that works with half the
//! servers. Both are read here.
//!
//! There is no reader task and no map of waiting requests: every POST is a
//! request/answer pair of its own, so the answer is whatever comes back on the
//! same response. What the stream can carry besides the answer — notifications,
//! and requests from the server — is dispatched as it arrives, because a server
//! that asked something is waiting and the answer has to go out while the
//! stream is still open.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::sync::Mutex;

use super::config::ServerConfig;
use super::health::Health;
use super::wire;

/// How long a connection may take to open: the endpoint's own pace is its
/// business, and a TCP handshake that takes longer than this is a host that is
/// not answering.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a notification is given: the server answers one with an empty
/// `202 Accepted`, so nothing here is waiting on the server's work.
const NOTIFY_BUDGET: Duration = Duration::from_secs(30);

/// How long the `DELETE` that ends a session is given. It is a courtesy on the
/// way out: a server that does not answer it is not one to wait for.
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(5);

/// What this client calls itself to the endpoint.
const USER_AGENT: &str = concat!("jingwei/", env!("CARGO_PKG_VERSION"));

/// The header the server names the session with, and the one that says which
/// version of the protocol this request speaks.
const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

/// What a request tells the server it can read back.
const ACCEPT: &str = "application/json, text/event-stream";

/// One HTTP endpoint, as a connection to a server.
pub struct Http {
    http: reqwest::Client,
    url: reqwest::Url,
    /// The headers the entry asked for, by name: how an endpoint that wants a
    /// token is given one.
    headers: HeaderMap,
    /// The session the server handed out when it answered `initialize`. Sent
    /// back on everything after that, and `None` for a server that keeps none.
    session: Mutex<Option<String>>,
    /// The version to announce: the one asked for until the handshake settles
    /// it, and the one it settled on after that.
    version: Mutex<String>,
    health: Arc<Health>,
    next_id: AtomicU64,
}

/// What an endpoint is, in a test's own output: where it calls and what it
/// knows about the session, rather than the headers it was configured with.
impl std::fmt::Debug for Http {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Http")
            .field("url", &self.url.as_str())
            .field("headers", &self.headers.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Http {
    /// Read the entry into an endpoint: the URL it names, with the headers it
    /// asked for and a client to call it with.
    pub fn new(config: &ServerConfig) -> Result<Self, String> {
        let text = config.url.clone().unwrap_or_default();
        let url =
            reqwest::Url::parse(&text).map_err(|e| format!("url {text:?} does not parse: {e}"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!("url {text:?} is neither http nor https"));
        }
        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|e| format!("header name {name:?} cannot be sent: {e}"))?;
            let value = HeaderValue::from_str(value)
                .map_err(|e| format!("the value of header {name} cannot be sent: {e}"))?;
            headers.insert(name, value);
        }
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| format!("failed to build an HTTP client: {e}"))?;
        Ok(Self {
            http,
            url,
            headers,
            session: Mutex::new(None),
            version: Mutex::new(wire::PROTOCOL_VERSION.to_string()),
            health: Health::new(),
            next_id: AtomicU64::new(1),
        })
    }

    /// One request, and the answer to it.
    pub async fn ask(
        &self,
        method: &str,
        params: Value,
        budget: Duration,
    ) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let message = wire::request(id, method, params);
        self.exchange(&message, budget)
            .await?
            .ok_or_else(|| format!("the server accepted {method:?} and sent no answer to it"))
    }

    /// One notification: the server takes it and says nothing back.
    pub async fn tell(&self, method: &str, params: Value) -> Result<(), String> {
        let message = wire::notification(method, params);
        self.exchange(&message, NOTIFY_BUDGET).await.map(|_| ())
    }

    /// End the session, the way the specification asks a client that is
    /// leaving to: a `DELETE`, which a server that does not keep sessions
    /// answers with a `405` and no harm done.
    pub async fn shutdown(&self) {
        let Some(session) = self.session.lock().await.clone() else {
            return;
        };
        let request = self
            .http
            .delete(self.url.clone())
            .header(SESSION_HEADER, session)
            .header(VERSION_HEADER, self.version.lock().await.clone())
            .timeout(SHUTDOWN_BUDGET);
        let _ = request.send().await;
    }

    /// The id of the message this response answers, and no other. A message
    /// the stream carries on the way — a notification, a request from the
    /// server, somebody else's answer — is dispatched and read past.
    async fn dispatch(
        &self,
        message: &Value,
        wanted: Option<u64>,
    ) -> Option<Result<Value, String>> {
        match wire::classify(message) {
            wire::Incoming::Answer { id, outcome } if id == wanted => Some(match outcome {
                wire::Outcome::Ok(value) => Ok(value.clone()),
                wire::Outcome::Failed { .. } => Err(outcome.reason()),
            }),
            // The server is asking something of us, on a stream we are still
            // reading: it is waiting, so the answer goes out now. What it says
            // back (an empty `202`) is not read: a server that cannot take the
            // answer is a server whose own request fails, and that is its news
            // to give us rather than ours to guess at.
            wire::Incoming::Ask { id, method } => {
                let answer = wire::serve(id, method);
                let _ = self.post_notification(&answer).await;
                None
            }
            _ => None,
        }
    }

    /// One message out, and whatever comes back on the same response.
    ///
    /// `None` means the server took the message without an answer — which is
    /// what a notification gets, and what a request that went unanswered looks
    /// like from here.
    async fn exchange(&self, message: &Value, budget: Duration) -> Result<Option<Value>, String> {
        if !self.health.alive() {
            return Err(self.health.why());
        }
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let wanted = message.get("id").and_then(Value::as_u64);
        let body = serde_json::to_vec(message).map_err(|e| format!("cannot be written: {e}"))?;
        let mut request = self
            .http
            .post(self.url.clone())
            .header(reqwest::header::ACCEPT, ACCEPT)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(VERSION_HEADER, self.version.lock().await.clone());
        if let Some(session) = self.session.lock().await.clone() {
            request = request.header(SESSION_HEADER, session);
        }
        for (name, value) in &self.headers {
            request = request.header(name.clone(), value.clone());
        }
        let response = request
            .timeout(budget)
            .body(body)
            .send()
            .await
            .map_err(|e| format!("the request failed: {e}"))?;

        let status = response.status();
        // The session the server names in this answer is the one every later
        // request has to carry, and this is the only answer that ever names it.
        if method == "initialize" {
            if let Some(session) = response.headers().get(SESSION_HEADER) {
                let session = session
                    .to_str()
                    .map_err(|e| format!("the session id the server sent cannot be read: {e}"))?;
                *self.session.lock().await = Some(session.to_string());
            }
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 404 && self.session.lock().await.is_some() {
                // The one failure that is about the connection rather than
                // about one request: the server has forgotten this session.
                self.health.dies(
                    "the server no longer knows this session (HTTP 404); start a new session \
                     to connect again",
                );
            }
            return Err(format!(
                "the server answered {} {}",
                status.as_u16(),
                first_line(&body)
            ));
        }

        let stream = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains("text/event-stream"));
        if !stream {
            let text = response.text().await.unwrap_or_default();
            if text.trim().is_empty() {
                // `202 Accepted` with no body: what the server answers a
                // notification with, and what it says about a request it took
                // without intending to answer.
                return Ok(None);
            }
            let message: Value =
                serde_json::from_str(&text).map_err(|e| format!("the answer is not JSON: {e}"))?;
            // The version the handshake settled on is the one every later
            // request announces; it is read here because this is where the
            // answer passes, and the client above is what decides whether this
            // client can run it.
            if method == "initialize" {
                if let Some(version) = message
                    .get("result")
                    .and_then(|result| result.get("protocolVersion"))
                    .and_then(Value::as_str)
                {
                    *self.version.lock().await = version.to_string();
                }
            }
            return self.dispatch(&message, wanted).await.transpose();
        }

        let mut events = Sse::default();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("reading the server's stream failed: {e}"))?;
            events.push(&chunk);
            while let Some(payload) = events.next_payload() {
                let Ok(message) = serde_json::from_str::<Value>(&payload) else {
                    // A frame that is not a message: the stream is the
                    // server's, and this client reads past what it cannot use
                    // rather than failing a request that may still be answered.
                    continue;
                };
                if method == "initialize" {
                    if let Some(version) = message
                        .get("result")
                        .and_then(|result| result.get("protocolVersion"))
                        .and_then(Value::as_str)
                    {
                        *self.version.lock().await = version.to_string();
                    }
                }
                if let Some(answer) = self.dispatch(&message, wanted).await {
                    return answer.map(Some);
                }
            }
        }
        if wanted.is_none() {
            // Nothing was waiting on this message: a stream that ends carried
            // what it was going to carry, and the server took the message,
            // which was all it promised.
            return Ok(None);
        }
        Err(format!(
            "the server ended its stream before answering {method:?}"
        ))
    }

    /// One message out that expects no answer: a notification, or the reply to
    /// something the server asked. A server answers both with an empty `202`,
    /// and only the status is looked at.
    async fn post_notification(&self, message: &Value) -> Result<(), String> {
        let body = serde_json::to_vec(message).map_err(|e| format!("cannot be written: {e}"))?;
        let mut request = self
            .http
            .post(self.url.clone())
            .header(reqwest::header::ACCEPT, ACCEPT)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(VERSION_HEADER, self.version.lock().await.clone());
        if let Some(session) = self.session.lock().await.clone() {
            request = request.header(SESSION_HEADER, session);
        }
        for (name, value) in &self.headers {
            request = request.header(name.clone(), value.clone());
        }
        let response = request
            .timeout(NOTIFY_BUDGET)
            .body(body)
            .send()
            .await
            .map_err(|e| format!("the request failed: {e}"))?;
        if !response.status().is_success() {
            return Err(format!(
                "the server answered {} to a message it does not answer",
                response.status().as_u16()
            ));
        }
        Ok(())
    }
}

/// The first line of a body, so that a server's own words reach the report
/// without the whole page of them.
fn first_line(body: &str) -> String {
    let line = body.lines().next().unwrap_or_default().trim();
    if line.is_empty() {
        return "with no explanation".to_string();
    }
    if line.chars().count() > 200 {
        return line.chars().take(200).collect::<String>() + "…";
    }
    line.to_string()
}

/// The server-sent events of one response body, read a frame at a time.
///
/// The framing is the SSE specification's: a line ends at `\n`, a field is a
/// name and a value with one optional space after the colon, and **an event is
/// dispatched by the blank line that ends it**. Only `data:` is read: the
/// protocol rides entirely on it, and `event:`, `id:` and `retry:` say things
/// about redelivery this client does not do.
#[derive(Default)]
struct Sse {
    /// Bytes that have arrived without a complete line in them yet.
    buf: Vec<u8>,
    /// The `data:` lines of the event being read.
    data: Vec<String>,
}

impl Sse {
    fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// The next event's payload, when a blank line has ended one.
    fn next_payload(&mut self) -> Option<String> {
        loop {
            let end = self.buf.iter().position(|&b| b == b'\n')?;
            let line = String::from_utf8_lossy(&self.buf[..end])
                .trim_end_matches('\r')
                .to_string();
            self.buf.drain(..=end);
            if line.is_empty() {
                // The frame is over. An empty one is a heartbeat or a comment
                // followed by its blank line, and there is nothing in it.
                if self.data.is_empty() {
                    continue;
                }
                return Some(std::mem::take(&mut self.data).join("\n"));
            }
            if let Some(rest) = line.strip_prefix("data:") {
                self.data
                    .push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const BUDGET: Duration = Duration::from_secs(5);

    fn endpoint(server: &MockServer, extra: Value) -> Http {
        let mut entry = json!({"url": format!("{}/mcp", server.uri())});
        for (key, value) in extra.as_object().into_iter().flatten() {
            entry[key] = value.clone();
        }
        let config: ServerConfig = serde_json::from_value(entry).unwrap();
        Http::new(&config).unwrap()
    }

    /// An answer that carries one JSON message. The media type goes with the
    /// body rather than in a header of its own: the fixture lets a body set it,
    /// and a body set after a header would win.
    fn json_answer(message: Value) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(message)
    }

    /// An answer that carries a stream of them.
    fn sse_answer(frames: &[Value]) -> ResponseTemplate {
        let mut body = String::new();
        for frame in frames {
            body.push_str("event: message\n");
            body.push_str(&format!("data: {frame}\n\n"));
        }
        ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
    }

    #[tokio::test]
    async fn one_json_answer_is_the_answer() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/mcp"))
            .respond_with(json_answer(
                json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}),
            ))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let answer = http.ask("tools/list", json!({}), BUDGET).await.unwrap();
        assert_eq!(answer["ok"], json!(true));
    }

    #[tokio::test]
    async fn what_the_request_says_about_itself_is_the_specifications() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(json_answer(
                json!({"jsonrpc": "2.0", "id": 1, "result": {}}),
            ))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({"headers": {"Authorization": "Bearer t"}}));
        http.ask("tools/list", json!({}), BUDGET).await.unwrap();
        let sent = &server.received_requests().await.unwrap()[0];
        assert_eq!(
            sent.headers.get("accept").unwrap(),
            "application/json, text/event-stream"
        );
        assert_eq!(
            sent.headers.get("content-type").unwrap(),
            "application/json"
        );
        assert_eq!(
            sent.headers.get("mcp-protocol-version").unwrap(),
            wire::PROTOCOL_VERSION
        );
        assert_eq!(sent.headers.get("authorization").unwrap(), "Bearer t");
        assert!(sent.headers.get("mcp-session-id").is_none());
        let body: Value = serde_json::from_slice(&sent.body).unwrap();
        assert_eq!(body["method"], json!("tools/list"));
        assert_eq!(body["jsonrpc"], json!("2.0"));
    }

    #[tokio::test]
    async fn a_stream_is_read_past_what_it_carries_until_the_answer() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                ": a comment\n\
                 \n\
                 data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\"}\n\
                 \n\
                 data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}\n\
                 \n",
                "text/event-stream; charset=utf-8",
            ))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let answer = http.ask("tools/list", json!({}), BUDGET).await.unwrap();
        assert_eq!(answer["tools"], json!([]));
    }

    #[tokio::test]
    async fn a_stream_that_ends_without_the_answer_is_a_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(sse_answer(&[json!({
                "jsonrpc": "2.0",
                "method": "notifications/message"
            })]))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let failed = http.ask("tools/list", json!({}), BUDGET).await.unwrap_err();
        assert!(failed.contains("ended its stream"), "{failed}");
    }

    #[tokio::test]
    async fn a_refusal_is_the_servers_own_words() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(json_answer(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32601, "message": "no such method"}
            })))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let refused = http.ask("tools/nope", json!({}), BUDGET).await.unwrap_err();
        assert!(refused.contains("no such method"), "{refused}");
    }

    #[tokio::test]
    async fn a_server_that_asks_is_answered_while_the_stream_is_open() {
        let server = MockServer::start().await;
        // The first POST is the request; the second is the client's reply to
        // the ping the stream carried.
        Mock::given(method("POST"))
            .and(header("content-type", "application/json"))
            .respond_with(sse_answer(&[
                json!({"jsonrpc": "2.0", "id": "s1", "method": "ping"}),
                json!({"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}),
            ]))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let answer = http.ask("tools/list", json!({}), BUDGET).await.unwrap();
        assert_eq!(answer["ok"], json!(true));
        let sent = server.received_requests().await.unwrap();
        assert_eq!(sent.len(), 2, "{sent:?}");
        let reply: Value = serde_json::from_slice(&sent[1].body).unwrap();
        assert_eq!(reply["id"], json!("s1"));
        assert_eq!(reply["result"], json!({}));
    }

    #[tokio::test]
    async fn the_session_the_server_names_is_carried_afterwards() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("mcp-session-id", "sess-1"))
            .respond_with(json_answer(
                json!({"jsonrpc": "2.0", "id": 2, "result": {}}),
            ))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                json_answer(json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2025-03-26"}
                }))
                .insert_header("mcp-session-id", "sess-1"),
            )
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        http.ask("initialize", json!({}), BUDGET).await.unwrap();
        // The second request is matched by the session header alone: if it did
        // not carry one, the mock would answer with the initialize body and
        // this would read the wrong answer.
        http.ask("tools/list", json!({}), BUDGET).await.unwrap();
        let sent = server.received_requests().await.unwrap();
        assert_eq!(sent[1].headers.get("mcp-session-id").unwrap(), "sess-1");
        // And the version the handshake settled on is the one announced after.
        assert_eq!(
            sent[1].headers.get("mcp-protocol-version").unwrap(),
            "2025-03-26"
        );
    }

    #[tokio::test]
    async fn a_session_the_server_has_forgotten_ends_the_connection() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(header("mcp-session-id", "sess-1"))
            .respond_with(ResponseTemplate::new(404).set_body_string("no such session"))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                json_answer(json!({"jsonrpc": "2.0", "id": 1, "result": {}}))
                    .insert_header("mcp-session-id", "sess-1"),
            )
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        http.ask("initialize", json!({}), BUDGET).await.unwrap();
        let gone = http.ask("tools/list", json!({}), BUDGET).await.unwrap_err();
        assert!(gone.contains("404"), "{gone}");
        assert!(!http.health.alive());
        assert!(http.health.why().contains("no longer knows this session"));
        // Asking again says so rather than going back to the server.
        let again = http.ask("tools/list", json!({}), BUDGET).await.unwrap_err();
        assert!(again.contains("no longer knows this session"), "{again}");
    }

    #[tokio::test]
    async fn a_server_that_refuses_the_request_says_why_in_the_report() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom\nstack: x"))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let failed = http.ask("tools/list", json!({}), BUDGET).await.unwrap_err();
        assert!(failed.contains("500"), "{failed}");
        assert!(failed.contains("boom"), "{failed}");
        assert!(!failed.contains("stack"), "one line is enough: {failed}");
    }

    #[tokio::test]
    async fn a_notification_expects_nothing_back() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(202))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        http.tell("notifications/initialized", json!({}))
            .await
            .unwrap();
        let sent = &server.received_requests().await.unwrap()[0];
        let body: Value = serde_json::from_slice(&sent.body).unwrap();
        assert!(body.get("id").is_none(), "{body}");
        // And a request the server did not answer at all is a failure rather
        // than an answer of `null`.
        let unanswered = http.ask("tools/list", json!({}), BUDGET).await.unwrap_err();
        assert!(unanswered.contains("no answer"), "{unanswered}");
    }

    #[tokio::test]
    async fn a_notification_the_server_refuses_is_a_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad notification"))
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        let refused = http
            .tell("notifications/initialized", json!({}))
            .await
            .unwrap_err();
        assert!(refused.contains("400"), "{refused}");
    }

    #[tokio::test]
    async fn ending_a_session_is_a_delete_that_names_it() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(header("mcp-session-id", "sess-7"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(
                json_answer(json!({"jsonrpc": "2.0", "id": 1, "result": {}}))
                    .insert_header("mcp-session-id", "sess-7"),
            )
            .mount(&server)
            .await;
        let http = endpoint(&server, json!({}));
        // A connection with no session ends quietly: there is nothing to end.
        http.shutdown().await;
        http.ask("initialize", json!({}), BUDGET).await.unwrap();
        http.shutdown().await;
        let sent = server.received_requests().await.unwrap();
        let delete = sent
            .iter()
            .find(|r| r.method == wiremock::http::Method::DELETE)
            .expect("a DELETE was sent");
        assert_eq!(delete.headers.get("mcp-session-id").unwrap(), "sess-7");
    }

    #[test]
    fn an_endpoint_that_cannot_be_called_is_refused_before_it_is() {
        let parsed: ServerConfig = serde_json::from_value(json!({"url": "not a url"})).unwrap();
        assert!(Http::new(&parsed).unwrap_err().contains("does not parse"));
        let scheme: ServerConfig =
            serde_json::from_value(json!({"url": "ftp://example.test/mcp"})).unwrap();
        assert!(
            Http::new(&scheme)
                .unwrap_err()
                .contains("neither http nor https")
        );
        let name: ServerConfig = serde_json::from_value(
            json!({"url": "https://example.test/mcp", "headers": {"a b": "t"}}),
        )
        .unwrap();
        assert!(Http::new(&name).unwrap_err().contains("header name"));
        let value: ServerConfig = serde_json::from_value(
            json!({"url": "https://example.test/mcp", "headers": {"x-trailing": "t\n"}}),
        )
        .unwrap();
        assert!(Http::new(&value).unwrap_err().contains("cannot be sent"));
    }

    #[test]
    fn an_sse_frame_is_read_by_its_blank_line() {
        let mut events = Sse::default();
        events.push(b"data: one\ndata: two\n\n");
        assert_eq!(events.next_payload().unwrap(), "one\ntwo");
        assert_eq!(events.next_payload(), None);
        // A frame that arrives a byte at a time is the same frame.
        let mut split = Sse::default();
        for byte in b"data: {\"a\": 1}\r\n\r\n" {
            split.push(&[*byte]);
        }
        assert_eq!(split.next_payload().unwrap(), "{\"a\": 1}");
        // Heartbeats and fields this client does not read are stepped over.
        let mut mixed = Sse::default();
        mixed.push(b": keep-alive\n\nevent: message\nid: 4\nretry: 100\ndata:nospace\n\n");
        assert_eq!(mixed.next_payload().unwrap(), "nospace");
    }

    #[test]
    fn a_first_line_is_one_line() {
        assert_eq!(first_line("boom\nstack"), "boom");
        assert_eq!(first_line("   \n"), "with no explanation");
        assert_eq!(first_line(""), "with no explanation");
        let long = first_line(&"x".repeat(500));
        assert_eq!(long.chars().count(), 201, "cut with a mark: {long}");
    }
}
