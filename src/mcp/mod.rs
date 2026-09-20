//! MCP: other programs' tools, offered to the model as though they were this
//! agent's own.
//!
//! The protocol is spoken in the modules below this one — the transports
//! (`stdio`, `http`), the JSON-RPC vocabulary (`wire`), one connection
//! (`client`) — and what is here is the part the rest of the agent sees: which
//! servers a session has, what their tools are called, how a call finds the
//! server that offers it, and what is said when none of them comes up.
//!
//! Three decisions are made here rather than in any of them:
//!
//! - **A tool keeps its own name, inside a name that says whose it is.** Two
//!   servers may offer a tool called `search`, and the model has to be able
//!   to say which one it means: [`tool_name`] is `mcp__<server>__<tool>`, the
//!   convention every other client of this protocol uses, so a tool the model
//!   learned from one is the same name in another.
//! - **A server that does not come up costs nothing else.** The model is
//!   offered the tools of the servers that did, the session runs, and `/mcp`
//!   says which server is missing and why — a workspace whose `.mcp.json`
//!   names a program this machine does not have is not a workspace nobody
//!   can work in.
//! - **A call that cannot be made is answered in words.** A name nobody
//!   offers, a server that died, a refusal from the far end: all of them come
//!   back as the result text, which is the same discipline the built-in tools
//!   follow and the only one the model can act on.
//!
//! ## Shape
//!
//! [`Hub`] is the public handle: a cheap clone of an `mpsc::Sender`. All
//! state lives in an actor task spawned by [`Hub::spawn`] (or [`Hub::empty`]
//! / [`Hub::of_entries`]); methods on `Hub` send a [`Command`] and await the
//! actor's reply. Nothing on the outside ever holds a `&mut HubInner`, and
//! the actor never shares `HubInner` with anyone — which is what keeps
//! enable / disable / reconnect / disconnect safe to call from any thread.

mod client;
mod config;
mod health;
mod http;
mod inner;
mod stdio;
/// Test fixture: a bash script that speaks the protocol. Always compiled
/// (not gated behind `#[cfg(test)]`) so a binary test that constructs a stub
/// can reach it through this module's public surface; the module is
/// `#[doc(hidden)]` so it does not appear in the rendered docs.
#[doc(hidden)]
pub mod stub;
mod wire;

use std::path::Path;

use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use config::Entry;
use inner::HubInner;
pub use inner::{ServerState, ServerStatus};

/// What a tool from a server is named, in front of the tool's own name.
pub const PREFIX: &str = "mcp__";

/// Whether a name is one this module answers for. The built-in tools are
/// dispatched by name too, and this is what keeps the two sets apart.
pub fn is_tool(name: &str) -> bool {
    name.starts_with(PREFIX)
}

/// A message to the hub's actor task: do this one thing, then send the result
/// back over `reply`. `None` of these are constructed by callers outside this
/// module — every method on [`Hub`] builds the one its name implies.
enum Command {
    Call {
        tool: String,
        args: String,
        reply: oneshot::Sender<String>,
    },
    Definitions {
        reply: oneshot::Sender<Vec<Value>>,
    },
    Notes {
        reply: oneshot::Sender<Vec<String>>,
    },
    Report {
        reply: oneshot::Sender<Vec<String>>,
    },
    List {
        reply: oneshot::Sender<Vec<ServerStatus>>,
    },
    ServerOp {
        op: ServerOp,
        name: String,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// The four ways to change one server's state — one variant because the
/// four are one fact with four words: each takes a name, each answers
/// `Result<(), String>`, and the actor spells the quartet exactly once.
#[derive(Clone, Copy)]
enum ServerOp {
    Enable,
    Disable,
    Reconnect,
    Disconnect,
}

/// A handle to the hub's actor task.
///
/// `Hub` is `Clone` — every clone is another sender into the same mailbox —
/// so callers can hand it out the way they would an `Arc<T>`, without an
/// `Arc` of their own. Methods are `async + &self` because the actor model
/// is "send a message, wait for the reply": there is no shared `&mut`, no
/// lock, and no surprise about who owns what.
#[derive(Clone)]
pub struct Hub {
    tx: mpsc::Sender<Command>,
}

impl Hub {
    /// One ask-reply round trip — the whole of every method below: build
    /// the command around a fresh reply channel, send it, await the
    /// actor's answer. `shut` is what the caller sees when the mailbox is
    /// closed (the actor is gone); `dropped` when the reply never arrives
    /// (the actor died mid-command). Every public method is one `ask`
    /// plus its two fallback answers.
    async fn ask<R>(
        &self,
        build: impl FnOnce(oneshot::Sender<R>) -> Command,
        shut: impl FnOnce() -> R,
        dropped: impl FnOnce() -> R,
    ) -> R {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(build(reply)).await.is_err() {
            return shut();
        }
        rx.await.unwrap_or_else(|_| dropped())
    }

    /// The four `ServerOp` verbs share everything but their name — one
    /// spelling here, four one-line methods below.
    async fn server_op(&self, op: ServerOp, name: &str) -> Result<(), String> {
        self.ask(
            |reply| Command::ServerOp { op, name: name.to_string(), reply },
            || Err("mcp hub is shut down".into()),
            || Err("mcp hub dropped the reply".into()),
        )
        .await
    }

    /// A hub with no servers behind it. The actor task is started up at once
    /// and answers every call as though no server were configured: the
    /// "unknown tool" message is the only thing the model ever sees.
    ///
    /// Synchronous because there is nothing to connect to: the actor's inner
    /// state is ready the moment the future starts polling.
    ///
    /// `#[allow(dead_code)]`: the production binary uses [`Hub::spawn`] to
    /// read configuration; this empty form is the test affordance — the
    /// wiring test in `tool_runtime` and the per-test setup in the TUI /
    /// plain frontends all start from a hub that has nothing configured.
    #[allow(dead_code)]
    pub fn empty() -> Self {
        spawn_with(async { HubInner::empty() })
    }

    /// Read the workspace and the user-supplied mcpServers, then start an
    /// actor task that connects to everything they name.
    ///
    /// Synchronous: the actor connects in the background, and calls made
    /// before the actor has finished connecting wait in the actor's mailbox
    /// until the connection step is done. A caller that wants the notes up
    /// front calls [`Hub::notes`] after — that is the moment the actor's
    /// inner state has settled into "every server either came up or said why
    /// it didn't".
    pub fn spawn(workspace: &Path, user_settings: Option<&Value>) -> Self {
        let workspace = workspace.to_path_buf();
        let user_settings = user_settings.cloned();
        spawn_with(
            async move { HubInner::from_workspace(&workspace, user_settings.as_ref()).await },
        )
    }

    /// Build a hub over a list of entries the caller already has — the test
    /// path that does not want a configuration file written. Async because
    /// the actor has to finish connecting before the caller can rely on the
    /// hub being ready, and a test that races its own fixture against the
    /// connection step would flake.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub async fn of_entries(entries: Vec<Entry>) -> Self {
        spawn_with(async move { HubInner::from_entries(entries, Vec::new()).await })
    }

    /// One tool call, answered with the text the model reads. Never fails —
    /// see the module-level third decision.
    pub async fn call(&self, tool: &str, args: &str) -> String {
        self.ask(
            |reply| Command::Call { tool: tool.to_string(), args: args.to_string(), reply },
            || "error: mcp hub is shut down".into(),
            || "error: mcp hub dropped the reply".into(),
        )
        .await
    }

    /// The tools currently offered to the model. Each entry is the
    /// wire-shaped JSON object (`{name, description, input_schema}`) the
    /// agent loop appends to its schemas list as-is.
    pub async fn definitions(&self) -> Vec<Value> {
        self.ask(|reply| Command::Definitions { reply }, Vec::new, Vec::new).await
    }

    /// What the session says about MCP when it starts: one line per server
    /// that came up, one per server that did not, and the warnings picked up
    /// along the way.
    pub async fn notes(&self) -> Vec<String> {
        self.ask(|reply| Command::Notes { reply }, Vec::new, Vec::new).await
    }

    /// What `/mcp` prints: every server, what it offers, and what is wrong.
    pub async fn report(&self) -> Vec<String> {
        self.ask(|reply| Command::Report { reply }, Vec::new, Vec::new).await
    }

    /// One row per server, with the state it stands in and how many of its
    /// tools are being offered. What `/mcp list` prints.
    pub async fn list(&self) -> Vec<ServerStatus> {
        self.ask(|reply| Command::List { reply }, Vec::new, Vec::new).await
    }

    /// Bring a `Disabled` server back online: open a fresh connection with
    /// its stored configuration, and route its tools to the model again.
    /// Errors out if the server is in any other state (`Ready`, `Failed`,
    /// `Disconnected`) or is not configured.
    pub async fn enable(&self, name: &str) -> Result<(), String> {
        self.server_op(ServerOp::Enable, name).await
    }

    /// Take a `Ready` server offline: close its connection and pull its
    /// tools out of the offered set. The configuration is kept so an
    /// `enable` can bring the server back.
    pub async fn disable(&self, name: &str) -> Result<(), String> {
        self.server_op(ServerOp::Disable, name).await
    }

    /// Reconnect a server in any state: close any live connection, then
    /// open a fresh one with the stored configuration.
    pub async fn reconnect(&self, name: &str) -> Result<(), String> {
        self.server_op(ServerOp::Reconnect, name).await
    }

    /// Close a `Ready` server's connection without touching its
    /// configuration. The server sits in `Disconnected` until `reconnect`
    /// brings it back.
    pub async fn disconnect(&self, name: &str) -> Result<(), String> {
        self.server_op(ServerOp::Disconnect, name).await
    }

    /// End every connection. Called by the binary at process exit so the
    /// programs the session started do not outlive it.
    pub async fn shutdown(&self) {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(Command::Shutdown { reply }).await.is_ok() {
            let _ = rx.await;
        }
    }
}

/// Start an actor task over `init`: a future that builds the initial
/// `HubInner`. Returns a [`Hub`] the caller can send to.
fn spawn_with<F>(init: F) -> Hub
where
    F: std::future::Future<Output = HubInner> + Send + 'static,
{
    let (tx, mut rx) = mpsc::channel::<Command>(64);
    tokio::spawn(async move {
        let mut inner = init.await;
        while let Some(cmd) = rx.recv().await {
            match cmd {
                Command::Call { tool, args, reply } => {
                    let result = inner.call(&tool, &args).await;
                    let _ = reply.send(result);
                }
                Command::Definitions { reply } => {
                    let _ = reply.send(inner.definitions());
                }
                Command::Notes { reply } => {
                    let _ = reply.send(inner.notes());
                }
                Command::Report { reply } => {
                    let _ = reply.send(inner.report());
                }
                Command::List { reply } => {
                    let _ = reply.send(inner.list());
                }
                Command::ServerOp { op, name, reply } => {
                    let out = match op {
                        ServerOp::Enable => inner.enable(&name).await,
                        ServerOp::Disable => inner.disable(&name).await,
                        ServerOp::Reconnect => inner.reconnect(&name).await,
                        ServerOp::Disconnect => inner.disconnect(&name).await,
                    };
                    let _ = reply.send(out);
                }
                Command::Shutdown { reply } => {
                    inner.shutdown_all().await;
                    let _ = reply.send(());
                    break;
                }
            }
        }
    });
    Hub { tx }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_empty_hub_answers_with_a_clear_unknown_tool_message() {
        let hub = Hub::empty();
        let out = hub.call("mcp__nope__nada", "{}").await;
        assert!(out.starts_with("error: no MCP tool named"), "{out}");
        assert!(out.contains("\"mcp__nope__nada\""), "{out}");
        hub.shutdown().await;
    }

    #[tokio::test]
    async fn definitions_is_empty_for_a_fresh_hub() {
        let hub = Hub::empty();
        let defs = hub.definitions().await;
        assert!(defs.is_empty(), "an empty hub offers nothing: {defs:?}");
        hub.shutdown().await;
    }

    #[tokio::test]
    async fn notes_and_report_are_empty_for_a_fresh_hub() {
        let hub = Hub::empty();
        assert!(hub.notes().await.is_empty());
        // report() is the only one that surfaces a default message when
        // nothing is configured, so an empty hub says so explicitly.
        let rep = hub.report().await;
        assert_eq!(rep.len(), 1);
        assert!(rep[0].contains("no MCP servers"), "{rep:?}");
        hub.shutdown().await;
    }

    #[tokio::test]
    async fn calls_to_an_unknown_name_include_what_was_offered() {
        let hub = Hub::empty();
        let out = hub.call("mcp__anyone__anything", "{}").await;
        assert!(out.contains("none came up"), "{out}");
        hub.shutdown().await;
    }

    #[tokio::test]
    async fn is_tool_recognises_only_mcp_prefixed_names() {
        assert!(is_tool("mcp__anything"));
        assert!(is_tool("mcp__"));
        assert!(!is_tool("bash"));
        assert!(!is_tool("read_file"));
        assert!(!is_tool(""));
    }

    // ---- Hub lifecycle: enable / disable / reconnect / disconnect --------

    use crate::mcp::stub::Stub;

    /// `disable` takes a `Ready` server offline, `enable` brings it back.
    /// The hub's report goes from listing the server's tools to listing the
    /// disabled note, then back to the live tools.
    #[tokio::test]
    async fn disable_then_enable_round_trips_a_ready_server() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo,fail")])]).await;
        let names = hub.definitions().await;
        assert!(!names.is_empty(), "stub offers echo + fail: {names:?}");
        hub.disable("stub").await.unwrap();
        // After disable, the tool list is empty — the model sees nothing
        // until the server comes back.
        assert!(hub.definitions().await.is_empty());
        // And the report names the server as disabled.
        let report = hub.report().await.join("\n");
        assert!(report.contains("stub"), "{report}");
        assert!(report.contains("disabled"), "{report}");
        hub.enable("stub").await.unwrap();
        // Back online, tools are offered again.
        let names = hub.definitions().await;
        assert_eq!(names.len(), 2, "got: {names:?}");
        hub.shutdown().await;
    }

    /// `enable` on a server that is not configured says so rather than
    /// silently succeeding. The error is a sentence the user can act on.
    #[tokio::test]
    async fn enable_a_server_that_does_not_exist_returns_an_error() {
        let hub = Hub::empty();
        let err = hub.enable("nope").await.unwrap_err();
        assert!(err.contains("no MCP server named"), "{err}");
        hub.shutdown().await;
    }

    /// `enable` on an already-enabled server refuses to do it again.
    #[tokio::test]
    async fn enable_an_already_enabled_server_is_an_error() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        let err = hub.enable("stub").await.unwrap_err();
        assert!(err.contains("already enabled"), "{err}");
        hub.shutdown().await;
    }

    /// `disable` on a server that is not configured says so.
    #[tokio::test]
    async fn disable_a_server_that_does_not_exist_returns_an_error() {
        let hub = Hub::empty();
        let err = hub.disable("nope").await.unwrap_err();
        assert!(err.contains("no MCP server named"), "{err}");
        hub.shutdown().await;
    }

    /// `disable` on a server that already failed to start is refused.
    #[tokio::test]
    async fn disable_a_failed_server_returns_an_error() {
        let mut entry = Stub::new().entry(&[("STUB_TOOLS", "echo")]);
        entry.config = Err("nope".into());
        let hub = Hub::of_entries(vec![entry]).await;
        let err = hub.disable("stub").await.unwrap_err();
        assert!(err.contains("failed to start"), "{err}");
        hub.shutdown().await;
    }

    /// `reconnect` brings a server back when its connection has died. We
    /// disconnect first (config preserved), then reconnect with the
    /// stored config — the new connection is healthy.
    #[tokio::test]
    async fn reconnect_brings_a_disconnected_server_back() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        hub.disconnect("stub").await.unwrap();
        // The hub reports the server as disconnected.
        assert!(hub.report().await.join("\n").contains("disconnected"));
        // Reconnect, and the tools come back.
        hub.reconnect("stub").await.unwrap();
        let names = hub.definitions().await;
        assert_eq!(names.len(), 1, "after reconnect: {names:?}");
        hub.shutdown().await;
    }

    /// `reconnect` on a server that was never configured says so.
    #[tokio::test]
    async fn reconnect_a_server_that_does_not_exist_returns_an_error() {
        let hub = Hub::empty();
        let err = hub.reconnect("nope").await.unwrap_err();
        assert!(err.contains("no MCP server named"), "{err}");
        hub.shutdown().await;
    }

    /// `disconnect` on a server that was never configured says so.
    #[tokio::test]
    async fn disconnect_a_server_that_does_not_exist_returns_an_error() {
        let hub = Hub::empty();
        let err = hub.disconnect("nope").await.unwrap_err();
        assert!(err.contains("no MCP server named"), "{err}");
        hub.shutdown().await;
    }

    /// `disconnect` closes a server's connection without dropping the
    /// configuration. The next call to one of its tools surfaces a
    /// "disconnected" error rather than routing to the dead server.
    #[tokio::test]
    async fn disconnect_then_call_surfaces_a_disconnected_error() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        // A ready server's report line carries the server's identity,
        // not the word "ready" — the state shows up only when something
        // is wrong. Before the disconnect, the report must therefore not
        // mention disconnected or disabled.
        let report_before = hub.report().await.join("\n");
        assert!(report_before.contains("stub 0.0.0"), "{report_before}");
        assert!(!report_before.contains("disconnected"), "{report_before}");
        assert!(!report_before.contains("disabled"), "{report_before}");
        hub.disconnect("stub").await.unwrap();
        let report_after = hub.report().await.join("\n");
        assert!(report_after.contains("disconnected"), "{report_after}");
        let out = hub.call("mcp__stub__echo", r#"{"text":"hi"}"#).await;
        assert!(out.starts_with("error: mcp server stub is disconnected"), "{out}");
        assert!(out.contains("/mcp reconnect stub"), "{out}");
        hub.shutdown().await;
    }

    /// A `Hub::call` for a name the hub knows but whose server is in
    /// `Disabled` state surfaces a disable-aware error rather than "no such
    /// tool".
    #[tokio::test]
    async fn call_to_a_disabled_server_says_so_in_its_own_words() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        hub.disable("stub").await.unwrap();
        let out = hub.call("mcp__stub__echo", r#"{"text":"hi"}"#).await;
        assert!(out.starts_with("error: mcp server stub is disabled"), "{out}");
        assert!(out.contains("/mcp enable stub"), "{out}");
        hub.shutdown().await;
    }

    /// A `Hub::call` whose arguments are not a JSON object is refused with
    /// a sentence the model can act on — not a panic, not a silent
    /// dispatch with empty arguments.
    #[tokio::test]
    async fn call_with_non_object_arguments_returns_a_model_readable_error() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        // Valid JSON, but not an object.
        let out = hub.call("mcp__stub__echo", r#""a string""#).await;
        assert!(out.starts_with("error: the arguments of"), "{out}");
        // Not even valid JSON.
        let out = hub.call("mcp__stub__echo", r#"not json"#).await;
        assert!(out.starts_with("error: the arguments of"), "{out}");
        hub.shutdown().await;
    }

    /// `Hub::list` returns one `ServerStatus` per server, with the state
    /// each one is in.
    #[tokio::test]
    async fn list_reports_one_row_per_server_with_state() {
        let stub1 = Stub::new();
        let mut entry1 = stub1.entry(&[("STUB_TOOLS", "echo,fail")]);
        entry1.name = "alpha".into();
        let stub2 = Stub::new();
        let mut entry2 = stub2.entry(&[("STUB_TOOLS", "noise")]);
        entry2.name = "beta".into();
        let hub = Hub::of_entries(vec![entry1, entry2]).await;
        let statuses = hub.list().await;
        assert_eq!(statuses.len(), 2, "got: {statuses:?}");
        for s in &statuses {
            assert_eq!(s.state, ServerState::Ready);
        }
        let names: Vec<&str> = statuses.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"alpha"), "{names:?}");
        assert!(names.contains(&"beta"), "{names:?}");
        let alpha = statuses.iter().find(|s| s.name == "alpha").unwrap();
        let beta = statuses.iter().find(|s| s.name == "beta").unwrap();
        // alpha was offered echo + fail; beta was offered noise.
        assert!(alpha.tools_count >= 1);
        assert!(beta.tools_count >= 1);
        hub.shutdown().await;
    }

    /// `Hub::call` against an `mcp__` name whose server is in `Failed`
    /// state at startup says why — a "did not start" error rather than a
    /// route lookup miss. We hand the hub an entry whose `config` is an
    /// error string, which `assemble` stores as `InternalState::Failed`.
    #[tokio::test]
    async fn call_to_a_failed_server_says_so_in_its_own_words() {
        let mut entry = Stub::new().entry(&[("STUB_TOOLS", "echo")]);
        entry.name = "broken".into();
        entry.config = Err("no such binary".into());
        let hub = Hub::of_entries(vec![entry]).await;
        let report = hub.report().await.join("\n");
        assert!(report.contains("broken"), "{report}");
        assert!(report.contains("did not start"), "{report}");
        assert!(report.contains("no such binary"), "{report}");
        hub.shutdown().await;
    }

    /// A hub whose actor task has been shut down answers every command
    /// with the same one-sentence error: "mcp hub is shut down". The
    /// path is taken from a held sender that the receiver has already
    /// dropped, and the model never sees a panic.
    #[tokio::test]
    async fn calls_after_shutdown_say_so_without_panicking() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        // First call works.
        let out = hub.call("mcp__stub__echo", r#"{"text":"hi"}"#).await;
        assert!(!out.starts_with("error: mcp hub"), "first call: {out}");
        hub.shutdown().await;
        // The actor task breaks out of the recv loop after Shutdown, so
        // every channel-side send now fails. The handler converts that
        // into the "hub is shut down" sentence.
        let out = hub.call("mcp__stub__echo", r#"{"text":"hi"}"#).await;
        assert_eq!(out, "error: mcp hub is shut down");
        assert!(hub.definitions().await.is_empty(), "no defs after shutdown");
        assert!(hub.notes().await.is_empty(), "no notes after shutdown");
        assert!(hub.report().await.is_empty(), "no report after shutdown");
        assert!(hub.list().await.is_empty(), "no list after shutdown");
        let err = hub.enable("stub").await.unwrap_err();
        assert!(err.contains("mcp hub is shut down"), "{err}");
        let err = hub.disable("stub").await.unwrap_err();
        assert!(err.contains("mcp hub is shut down"), "{err}");
        let err = hub.reconnect("stub").await.unwrap_err();
        assert!(err.contains("mcp hub is shut down"), "{err}");
        let err = hub.disconnect("stub").await.unwrap_err();
        assert!(err.contains("mcp hub is shut down"), "{err}");
    }

    /// `notes()` covers all four states. A ready server contributes its
    /// count to the headline; a failed one contributes a sentence; and
    /// the warnings the hub picked up on the way are appended.
    #[tokio::test]
    async fn notes_cover_ready_failed_disabled_and_warnings() {
        // Server A: ready (a real stub).
        let ready_entry = Stub::new().entry(&[("STUB_TOOLS", "echo")]);
        // Server B: failed at startup (an unusable config).
        let mut failed_entry = Stub::new().entry(&[("STUB_TOOLS", "echo")]);
        failed_entry.name = "broken".into();
        failed_entry.config = Err("no such binary".into());
        let hub = Hub::of_entries(vec![ready_entry, failed_entry]).await;
        let notes = hub.notes().await.join("\n");
        // The headline mentions the ready server and the count of tools.
        assert!(notes.contains("stub"), "notes: {notes}");
        // The failed server gets its own line.
        assert!(notes.contains("broken"), "notes: {notes}");
        assert!(notes.contains("did not start"), "notes: {notes}");
        assert!(notes.contains("no such binary"), "notes: {notes}");
        hub.shutdown().await;
    }

}
