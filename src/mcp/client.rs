//! One server, spoken to: the handshake, the tools it offers, and calling them.
//!
//! The protocol layer, above the transports and below the hub. What it knows is
//! the lifecycle — `initialize`, then `notifications/initialized`, then
//! `tools/list` — and what a tool result means, which is the part that is not a
//! straight copy: a result is a list of content blocks, and the model is handed
//! text, so every block has to become words that say what it was.
//!
//! This client declares no capabilities of its own: no roots to offer a server,
//! no sampling to answer with, no elicitation to ask about. What it does is
//! offer the server's tools to the model and run them when it asks, and a client
//! that claims more than that is a client that will be asked and have to refuse.

use std::time::Duration;

use serde_json::{Value, json};

use super::config::ServerConfig;
use super::http::Http;
use super::stdio::Stdio;
use super::wire;

/// The result-text ceiling the model is allowed to read. Same number as the
/// built-in tools use, so that a result the MCP server hands back is bounded
/// by the same limit a Bash command's stdout is.
///
/// Duplicated from `MAX_TOOL_OUTPUT` rather than pulled in: this module sits
/// beside the built-in tools, not below them, and a round-trip for one
/// constant would be a dependency the protocol layer does not need.
const MAX_OUTPUT: usize = 50_000;

/// How long a server is given to start and answer the handshake. Starting can
/// be slow — `npx` fetches a package the first time it is asked for one — and a
/// server that has not come up inside this is not coming up.
pub const START_SECS: u64 = 60;

/// How long one tool call may take. A call is the server's work rather than its
/// startup: a browser, a search, a query. Nothing here bounds the *session*,
/// and a turn can wait this long for one step of it.
pub const CALL_SECS: u64 = 300;

/// How many pages of a tool list this client will read. A server that keeps
/// handing back a cursor is a server nobody can finish talking to, and a turn
/// that waited for it would never start.
const MAX_PAGES: usize = 32;

/// How many tools one server may offer. The tool list is part of every request
/// the session sends, so this is a context-window bound rather than a sanity
/// check.
const MAX_TOOLS: usize = 512;

/// Truncate `s` to at most `max` bytes, backing off to a UTF-8 character
/// boundary so a multi-byte character is never cut in half.
///
/// Mirrors `tools::truncate` so the protocol layer has no dependency on the
/// built-in tools.
fn truncate(s: &str, max: usize) -> (String, bool) {
    if s.len() <= max {
        return (s.to_owned(), false);
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_owned(), true)
}

/// The transport a connection speaks through. An enum rather than a boxed trait
/// object: the set is closed (the specification defines exactly these two), and
/// a closed set that the compiler can see through is what makes adding a third
/// a decision rather than an accident.
enum Transport {
    Stdio(Stdio),
    Http(Http),
}

impl Transport {
    fn open(config: &ServerConfig) -> Result<Self, String> {
        match (&config.command, &config.url) {
            (Some(_), None) => Ok(Transport::Stdio(Stdio::spawn(config)?)),
            (None, Some(_)) => Ok(Transport::Http(Http::new(config)?)),
            // Unreachable: an entry that named both or neither was refused when
            // it was read, and nothing else builds one.
            _ => Err("the entry names both a command and a url, or neither".into()),
        }
    }

    async fn ask(&self, method: &str, params: Value, budget: Duration) -> Result<Value, String> {
        match self {
            Transport::Stdio(stdio) => stdio.ask(method, params, budget).await,
            Transport::Http(http) => http.ask(method, params, budget).await,
        }
    }

    async fn tell(&self, method: &str, params: Value) -> Result<(), String> {
        match self {
            Transport::Stdio(stdio) => stdio.tell(method, params).await,
            Transport::Http(http) => http.tell(method, params).await,
        }
    }

    async fn shutdown(&self) {
        match self {
            Transport::Stdio(stdio) => stdio.shutdown().await,
            Transport::Http(http) => http.shutdown().await,
        }
    }
}

/// One tool a server offers, as the server described it. The name is the
/// server's own and stays that way here: what the model sees is the hub's
/// business, and it is the hub that has to keep two servers' tools apart.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoteTool {
    pub name: String,
    pub description: Option<String>,
    /// The JSON Schema of the arguments. A server that offers none gets the one
    /// every backend accepts for a tool that takes none.
    pub schema: Value,
}

/// One server, live: the pipes to it, and what it said about itself.
pub struct Connection {
    transport: Transport,
    /// The server's own name and version, as it introduced itself. `/mcp`
    /// shows it beside the name the configuration gave the entry.
    pub identity: String,
    /// The protocol version both ends settled on.
    pub protocol: String,
    /// The tools it offers, in the order it listed them.
    pub tools: Vec<RemoteTool>,
    /// What had to be said about its answers on the way: entries in its tool
    /// list that are not tools, and a list that never ended.
    pub notes: Vec<String>,
    /// How long this server's entry allows one request to take, when it said.
    timeout: Option<u64>,
}

/// What a connection is, in a test's own output: the server it reached and
/// what came back from it, rather than the pipes it reached it through.
impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("identity", &self.identity)
            .field("protocol", &self.protocol)
            .field("tools", &self.tools.len())
            .finish_non_exhaustive()
    }
}

impl Connection {
    /// Open one connection: start the transport, shake hands, and read the tool
    /// list.
    ///
    /// Everything that can go wrong here is one sentence — the process would
    /// not start, the handshake refused, the version is not one this client
    /// knows — because the caller has one thing to do with it: tell the user
    /// which server did not come up and why, and carry on without it.
    pub async fn open(config: &ServerConfig) -> Result<Self, String> {
        let transport = Transport::open(config)?;
        let budget = Duration::from_secs(config.timeout.unwrap_or(START_SECS));
        let hello = transport.ask("initialize", initialize(), budget).await?;
        let protocol = wire::negotiate(&hello)?;
        let identity = identity_of(&hello);
        // The server is told the client is ready. A server that never gets this
        // is allowed to hold back — some refuse every request until it arrives.
        transport
            .tell("notifications/initialized", json!({}))
            .await?;
        let (tools, notes) = list_tools(&transport, budget).await?;
        Ok(Self {
            transport,
            identity,
            protocol,
            tools,
            notes,
            timeout: config.timeout,
        })
    }

    /// Call one of its tools, and render the result as the text the model reads.
    ///
    /// A tool that reported `isError: true` is the server's own way of saying
    /// the call did not succeed, with a body of text that explains why. The
    /// tool result text opening with `error: ` is what makes the agent core
    /// mark a call as a failure — so a server's own failure is reported in the
    /// same shape every other tool's failure is, and a server that already
    /// said `error:` in its text is not told twice. The hub's caller reads
    /// this string for both the failure flag and the model-readable text;
    /// nothing here decides what to do with it.
    pub async fn call(&self, tool: &str, arguments: Value) -> Result<String, String> {
        let budget = Duration::from_secs(self.timeout.unwrap_or(CALL_SECS));
        let result = self
            .transport
            .ask(
                "tools/call",
                json!({"name": tool, "arguments": arguments}),
                budget,
            )
            .await?;
        let text = render(&result);
        let failed = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(match (failed, text.starts_with("error:")) {
            (true, false) => format!("error: {text}"),
            // A server that already opened with `error:` said so itself — the
            // prefix stays once. A server that did not fail passes through.
            _ => text,
        })
    }

    /// End the connection: the server is told the session is over, and the
    /// transport closes what it opened.
    pub async fn shutdown(&self) {
        self.transport.shutdown().await;
    }
}

/// The handshake's parameters: the version asked for, the capabilities this
/// client offers, and who is asking.
///
/// Capabilities are an empty object: this client advertises no `roots`, no
/// `sampling`, no `elicitation`. The protocol asks a client to declare what it
/// can serve, and the tools-only client we are declares nothing — the
/// `tools/call` method is what the model reaches us through, not a capability
/// the server asks for. The base JSON-RPC shape still has to be there, so the
/// server can read it.
fn initialize() -> Value {
    json!({
        "protocolVersion": wire::PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "jingwei", "version": env!("CARGO_PKG_VERSION")},
    })
}

/// What a server calls itself, for the report: its name and version, either one
/// missing without fuss.
fn identity_of(hello: &Value) -> String {
    let name = hello
        .get("serverInfo")
        .and_then(|info| info.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("unnamed");
    match hello
        .get("serverInfo")
        .and_then(|info| info.get("version"))
        .and_then(Value::as_str)
    {
        Some(version) => format!("{name} {version}"),
        None => name.to_string(),
    }
}

/// Read the whole tool list, page by page.
///
/// The protocol's list may be paged, and a client that reads the first page and
/// stops is a client that quietly hides tools from the model. What is not a tool
/// is left out and said so: a server whose list carries an entry with no name
/// has a broken tool beside its working ones, and the working ones are worth
/// having.
async fn list_tools(
    transport: &Transport,
    budget: Duration,
) -> Result<(Vec<RemoteTool>, Vec<String>), String> {
    let mut tools = Vec::new();
    let mut notes = Vec::new();
    let mut cursor: Option<String> = None;
    for _ in 0..MAX_PAGES {
        let params = match &cursor {
            Some(cursor) => json!({"cursor": cursor}),
            None => json!({}),
        };
        let answer = transport.ask("tools/list", params, budget).await?;
        for entry in answer
            .get("tools")
            .and_then(Value::as_array)
            .unwrap_or(&Vec::new())
        {
            match read_tool(entry) {
                Some(tool) => tools.push(tool),
                None => notes.push(format!(
                    "an entry in its tool list is not a tool and was left out: {entry}"
                )),
            }
        }
        if tools.len() > MAX_TOOLS {
            tools.truncate(MAX_TOOLS);
            notes.push(format!(
                "it offers more than the {MAX_TOOLS} tools this client will take, so the rest \
                 are not offered"
            ));
            return Ok((tools, notes));
        }
        cursor = answer
            .get("nextCursor")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if cursor.is_none() {
            return Ok((tools, notes));
        }
    }
    notes.push(format!(
        "it was still handing back a cursor after {MAX_PAGES} pages, so the list was cut there"
    ));
    Ok((tools, notes))
}

/// One entry of a tool list, or `None` when it is not one.
fn read_tool(entry: &Value) -> Option<RemoteTool> {
    let name = entry.get("name").and_then(Value::as_str)?;
    if name.trim().is_empty() {
        return None;
    }
    // A tool with a title and no description is a tool whose title is what it
    // has to say for itself; the model reads one field, and this is it.
    let description = ["description", "title"]
        .iter()
        .find_map(|key| entry.get(*key).and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned);
    let schema = entry
        .get("inputSchema")
        .filter(|schema| schema.is_object())
        .cloned()
        .unwrap_or_else(|| json!({"type": "object"}));
    Some(RemoteTool {
        name: name.to_string(),
        description,
        schema,
    })
}

/// A tool result as text: every content block said in words, in the order the
/// server wrote them.
///
/// The blocks the model can use are the ones it can read, and a block it cannot
/// — an image, an audio clip, a link to a resource — is still something the
/// server answered with, so it is named rather than dropped: a result that
/// silently lost half of itself reads to the model as a tool that found nothing.
fn render(result: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(content) = result.get("content").and_then(Value::as_array) {
        parts.extend(content.iter().map(block_text));
    }
    // Structure with no prose around it: the shape is the answer, and the model
    // reads JSON as readily as it reads a sentence.
    if parts.is_empty() {
        if let Some(structured) = result.get("structuredContent") {
            parts.push(
                serde_json::to_string_pretty(structured).unwrap_or_else(|_| structured.to_string()),
            );
        }
    }
    if parts.is_empty() {
        return "(the tool answered with nothing)".to_string();
    }
    let text = parts.join("\n");
    let (text, cut) = truncate(&text, MAX_OUTPUT);
    if cut {
        return format!(
            "{text}\n[the result was cut at the {MAX_OUTPUT}-byte output limit; the server sent more]"
        );
    }
    text
}

/// One content block, in words.
fn block_text(block: &Value) -> String {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => block
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        Some("image") => format!(
            "[the server answered with an image: {}, {} bytes]",
            media_type(block),
            base64_bytes(block)
        ),
        Some("audio") => format!(
            "[the server answered with audio: {}, {} bytes]",
            media_type(block),
            base64_bytes(block)
        ),
        Some("resource_link") => match block.get("name").and_then(Value::as_str) {
            Some(name) => format!(
                "[a resource the server pointed at: {name} ({})]",
                uri(block)
            ),
            None => format!("[a resource the server pointed at: {}]", uri(block)),
        },
        Some("resource") => match block.get("resource") {
            Some(resource) => {
                if let Some(text) = resource.get("text").and_then(Value::as_str) {
                    format!(
                        "[a resource the server embedded: {}]\n{text}",
                        uri(resource)
                    )
                } else {
                    format!(
                        "[a resource the server embedded: {}, {} bytes of it]",
                        uri(resource),
                        resource
                            .get("blob")
                            .and_then(Value::as_str)
                            .map(base64_len)
                            .unwrap_or(0)
                    )
                }
            }
            None => format!("[an embedded resource with nothing in it: {block}]"),
        },
        Some(other) => format!("[{other}: {block}]"),
        None => format!("[a block with no type: {block}]"),
    }
}

fn media_type(block: &Value) -> String {
    block
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("no media type given")
        .to_string()
}

fn uri(value: &Value) -> String {
    value
        .get("uri")
        .and_then(Value::as_str)
        .unwrap_or("no uri given")
        .to_string()
}

/// What a base64 payload holds, in bytes: four characters carry three bytes,
/// and the padding cuts the last group short.
fn base64_encoded_len(bytes: usize) -> usize {
    bytes.div_ceil(4) * 3
}

fn base64_bytes(block: &Value) -> usize {
    block
        .get("data")
        .and_then(Value::as_str)
        .map(|data| base64_encoded_len(data.len()))
        .unwrap_or(0)
}

fn base64_len(data: &str) -> usize {
    base64_encoded_len(data.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::stub::Stub;

    /// A connection to a stub server, and the stub itself, so a test can ask
    /// what reached it.
    async fn connected(vars: &[(&str, &str)]) -> (Connection, Stub) {
        let stub = Stub::new();
        let connection = Connection::open(&stub.config(vars))
            .await
            .expect("the stub server comes up");
        (connection, stub)
    }

    #[tokio::test]
    async fn the_handshake_brings_back_what_it_offers() {
        let (connection, _stub) = connected(&[("STUB_TOOLS", "echo,fail")]).await;
        assert_eq!(connection.identity, "stub 0.0.0");
        assert_eq!(connection.protocol, wire::PROTOCOL_VERSION);
        let names: Vec<&str> = connection.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["echo", "fail"]);
        assert_eq!(
            connection.tools[0].description.as_deref(),
            Some("the echo tool")
        );
        assert_eq!(connection.tools[0].schema["type"], json!("object"));
        assert!(connection.notes.is_empty(), "{:?}", connection.notes);
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_call_reaches_the_server_and_its_answer_comes_back() {
        let (connection, stub) = connected(&[("STUB_TOOLS", "echo")]).await;
        let answer = connection
            .call("echo", json!({"text": "hi there"}))
            .await
            .unwrap();
        // The stub strips the quotes from what it was sent and says it back, so
        // this is the arguments as they arrived.
        assert_eq!(answer, "called with {text:hi there}");
        let received = stub.received();
        assert!(
            received
                .iter()
                .any(|line| line.contains(r#""method":"tools/call""#) && line.contains("hi there")),
            "{received:?}"
        );
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn the_client_says_what_it_is_and_asks_for_the_newest_version() {
        let (connection, stub) = connected(&[]).await;
        let first = stub.received().first().cloned().unwrap_or_default();
        assert!(
            first.contains(r#""protocolVersion":"2025-06-18""#),
            "{first}"
        );
        assert!(first.contains(r#""capabilities":{}"#), "{first}");
        assert!(first.contains("jingwei"), "{first}");
        // And it says it is ready, which is what a server waits for before it
        // answers anything else.
        assert!(
            stub.received()
                .iter()
                .any(|line| line.contains("notifications/initialized")),
            "the initialized notification was sent"
        );
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_ping_from_the_server_is_answered() {
        let (connection, stub) = connected(&[("STUB_TOOLS", "ping")]).await;
        let answer = connection.call("ping", json!({})).await.unwrap();
        assert_eq!(answer, "answered after asking");
        // Ended before the log is read: the reply is written while the server
        // is still asking, and the server is what writes it down — waiting for
        // the process to be gone is what makes this an assertion rather than a
        // race with it.
        connection.shutdown().await;
        let received = stub.received();
        assert!(
            received
                .iter()
                .any(|line| line.contains(r#""id":"s1""#) && line.contains(r#""result":{}"#)),
            "the server's own request was answered: {received:?}"
        );
    }

    #[tokio::test]
    async fn a_banner_where_a_message_belongs_is_read_past() {
        let (connection, _stub) = connected(&[("STUB_TOOLS", "noise,slow")]).await;
        assert_eq!(
            connection.call("noise", json!({})).await.unwrap(),
            "answered after the banner"
        );
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_tool_that_failed_says_so_in_its_own_words() {
        let (connection, _stub) = connected(&[("STUB_TOOLS", "fail")]).await;
        let answer = connection.call("fail", json!({})).await.unwrap();
        assert_eq!(answer, "error: the stub could not do it");
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn every_kind_of_block_is_said_in_words() {
        let (connection, _stub) = connected(&[("STUB_TOOLS", "rich,empty,structured")]).await;
        let answer = connection.call("rich", json!({})).await.unwrap();
        assert!(answer.contains("a picture"), "{answer}");
        assert!(answer.contains("image/png"), "{answer}");
        assert!(answer.contains("audio/wav"), "{answer}");
        assert!(
            answer.contains("notes.md (file:///tmp/notes.md)"),
            "{answer}"
        );
        assert!(
            answer
                .contains("[a resource the server embedded: file:///tmp/x.txt]\nthe embedded text"),
            "{answer}"
        );
        assert!(answer.contains("file:///tmp/blob.bin, 3 bytes"), "{answer}");
        assert!(answer.contains("something_new"), "{answer}");

        // A result with no content at all says so rather than saying nothing.
        assert_eq!(
            connection.call("empty", json!({})).await.unwrap(),
            "(the tool answered with nothing)"
        );
        // Structure and no prose: the shape is the answer.
        let structured = connection.call("structured", json!({})).await.unwrap();
        assert!(structured.contains("\"count\": 2"), "{structured}");
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_result_larger_than_one_may_be_is_cut() {
        // The MCP layer's cap (MAX_OUTPUT) is 50_000 bytes for jingwei —
        // matched to the built-in tool output ceiling. The stub sends 60_000
        // bytes of 'x', so the result is cut at 50_000 with the marker below.
        let (connection, _stub) = connected(&[("STUB_TOOLS", "big")]).await;
        let answer = connection.call("big", json!({})).await.unwrap();
        assert!(
            answer.contains("cut at the 50000-byte output limit"),
            "{answer}"
        );
        assert!(answer.len() < 60_000, "{} bytes", answer.len());
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_version_this_client_does_not_speak_is_refused() {
        let stub = Stub::new();
        let refused = Connection::open(&stub.config(&[("STUB_VERSION", "2099-01-01")]))
            .await
            .unwrap_err();
        assert!(refused.contains("2099-01-01"), "{refused}");
    }

    #[tokio::test]
    async fn a_server_that_never_answers_runs_out_of_time() {
        let stub = Stub::new();
        let refused = Connection::open(&stub.config_timed(&[("STUB_SILENT", "1")], 1))
            .await
            .unwrap_err();
        assert!(refused.contains("no answer to initialize"), "{refused}");
        assert!(refused.contains("within 1s"), "{refused}");
    }

    #[tokio::test]
    async fn a_call_that_runs_too_long_says_what_the_server_last_said() {
        let stub = Stub::new();
        let connection = Connection::open(&stub.config_timed(&[("STUB_TOOLS", "noise,slow")], 1))
            .await
            .unwrap();
        assert_eq!(
            connection.call("noise", json!({})).await.unwrap(),
            "answered after the banner"
        );
        let late = connection.call("slow", json!({})).await.unwrap_err();
        assert!(late.contains("no answer to tools/call within 1s"), "{late}");
        // The banner it wrote where a message belongs is what it has to say
        // for itself when nothing else arrives.
        assert!(late.contains("Stub MCP server ready."), "{late}");
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_command_that_cannot_start_is_one_sentence() {
        let stub = Stub::new();
        let mut entry = stub.entry(&[]);
        let config = entry.config.as_mut().unwrap();
        config.command = Some("jingwei-no-such-binary".into());
        let refused = Connection::open(config).await.unwrap_err();
        assert!(refused.contains("failed to start"), "{refused}");
    }

    #[tokio::test]
    async fn a_server_that_dies_mid_call_says_so_rather_than_waiting() {
        let (connection, _stub) = connected(&[("STUB_TOOLS", "die")]).await;
        let refused = connection.call("die", json!({})).await.unwrap_err();
        assert!(refused.contains("closed its output"), "{refused}");
        // And the connection is marked dead, so the next call does not go
        // looking for a process that is gone.
        let again = connection.call("echo", json!({})).await.unwrap_err();
        assert!(again.contains("closed its output"), "{again}");
    }

    #[tokio::test]
    async fn paging_is_read_to_the_end() {
        let (connection, _stub) = connected(&[("STUB_PAGES", "3")]).await;
        let names: Vec<&str> = connection.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["page1", "page2", "page3"]);
        assert!(connection.notes.is_empty(), "{:?}", connection.notes);
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn a_list_that_never_ends_is_left_alone() {
        let (connection, _stub) = connected(&[("STUB_FOREVER", "1")]).await;
        assert_eq!(connection.tools.len(), MAX_PAGES);
        assert_eq!(connection.notes.len(), 1, "{:?}", connection.notes);
        assert!(
            connection.notes[0].contains("cursor"),
            "{:?}",
            connection.notes
        );
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn an_entry_that_is_not_a_tool_is_left_out_and_said_so() {
        let (connection, _stub) = connected(&[("STUB_JUNK", "1")]).await;
        let names: Vec<&str> = connection.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["echo"]);
        assert_eq!(connection.notes.len(), 2, "{:?}", connection.notes);
        assert!(
            connection.notes[0].contains("not a tool"),
            "{:?}",
            connection.notes
        );
        connection.shutdown().await;
    }

    #[tokio::test]
    async fn ending_the_connection_ends_the_server() {
        let (connection, stub) = connected(&[]).await;
        connection.shutdown().await;
        // The server reads its input ending as the end of the session: it says
        // so in its log, which is where a test can see that it left.
        assert_eq!(stub.received().last().map(String::as_str), Some("eof"));
    }

    #[test]
    fn a_tool_without_a_name_is_not_a_tool() {
        assert!(read_tool(&json!({"name": "echo"})).is_some());
        assert!(read_tool(&json!({"description": "no name"})).is_none());
        assert!(read_tool(&json!({"name": "   "})).is_none());
        assert!(read_tool(&json!("a string")).is_none());
    }

    #[test]
    fn a_tool_with_a_title_and_no_description_is_described_by_its_title() {
        let tool = read_tool(&json!({
            "name": "search",
            "title": "Search the web",
            "inputSchema": "not an object"
        }))
        .unwrap();
        assert_eq!(tool.description.as_deref(), Some("Search the web"));
        // A schema that is not a schema is no schema: the model is told the
        // arguments are an object rather than told something untrue.
        assert_eq!(tool.schema, json!({"type": "object"}));
        let blank = read_tool(&json!({"name": "x", "description": "  "})).unwrap();
        assert_eq!(blank.description, None);
    }

    #[test]
    fn a_block_this_client_does_not_model_is_still_reported() {
        assert_eq!(block_text(&json!({"type": "text", "text": "hi"})), "hi");
        assert!(block_text(&json!({"type": "text"})).is_empty());
        assert!(block_text(&json!({"type": "image"})).contains("no media type given"));
        assert!(block_text(&json!({"type": "resource_link"})).contains("no uri given"));
        assert!(block_text(&json!({"type": "resource", "resource": {}})).contains("no uri given"));
        assert!(block_text(&json!({"type": "resource"})).contains("nothing in it"));
        assert!(block_text(&json!({"type": "mystery", "x": 1})).contains("mystery"));
        assert!(block_text(&json!({"x": 1})).contains("no type"));
    }

    #[test]
    fn what_a_server_calls_itself_is_read_leniently() {
        assert_eq!(
            identity_of(&json!({"serverInfo": {"name": "s", "version": "1"}})),
            "s 1"
        );
        assert_eq!(identity_of(&json!({"serverInfo": {"name": "s"}})), "s");
        assert_eq!(identity_of(&json!({})), "unnamed");
    }
}
