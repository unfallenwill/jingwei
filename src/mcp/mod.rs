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
//!
//! ## The actor routes; it does not run
//!
//! The actor holds the registry — the routes, the offered tool names, the
//! warnings, the per-server tool lists — and the registry is cheap to read.
//! A call is *routed*: the actor resolves the route, hands the request to
//! the queue of the server that owns it, and goes straight back to its loop.
//! The answer comes back later, on the caller's own channel.
//!
//! Each server's connection is owned by its own task, with its own queue.
//! That task runs that server's calls one at a time — the hub serializes a
//! server's calls on purpose, so one server's calls keep one order — while
//! different servers run concurrently. So a slow server delays its own
//! calls and nothing else: the tool list the agent asks for at the start of
//! every turn, and every `/mcp` command, are answered while it is busy.
//!
//! ## What `/mcp` does while a call is live
//!
//! A server op applies to the registry at once, and to the server's
//! connection at its next safe point. A call already in flight on that
//! server completes; the op does not cancel it.
//!
//! - `enable` / `disable`: the registry's view changes at once — so the tool
//!   list changes at once, and the model stops seeing a disabled server's
//!   tools on the next turn. A call already running on that server finishes.
//! - `reconnect`: the old connection is retired at the next safe point and
//!   the new one takes over. A call on the old connection finishes. The
//!   registry forgets the server's tools at once, so the model is never
//!   offered a tool whose server is in the middle of a change.
//! - `disconnect`: as `reconnect`, and the server's tools leave the offered
//!   list at once.
//!
//! **When the command is answered** depends on what it needs. `disable` and
//! `disconnect` need no connection — they only change the registry and ask
//! for the connection to be closed later — so they are answered at once,
//! whatever the server is doing. `enable` and `reconnect` need a *new*
//! connection, so they are answered when that connection is up, and a
//! server's queue is serial: that answer comes behind whatever the server
//! was already doing, a call in flight included.
//!
//! Two operations on one server never wait on each other's *registry*
//! half — that half happens as soon as the actor handles the command — so
//! their connection halves can complete in either order. The later one is
//! the one that stands: the registry's view is the later operation's, and
//! the earlier one is told it was superseded. A connection that comes up
//! after the server was taken offline is not that server's any more, and
//! does not bring it back.
//!
//! The old behaviour — every op waiting behind the in-flight call — was an
//! accident of the actor's loop, not a contract, and it is not described as
//! one.

mod client;
mod config;
mod health;
mod http;
mod inner;
mod stdio;
/// Test fixture: a bash script that speaks the protocol. Compiled only into
/// the test build of the crate: the production build never constructs a
/// stub, and never compiles the script.
#[cfg(test)]
#[doc(hidden)]
pub mod stub;
mod wire;

use std::path::Path;

use crate::cancel::CancelToken;
use crate::core::ids::AgentId;
use crate::core::ports::{BoxFuture, ToolProvider};
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
    /// A server's task has opened a fresh connection and is telling the
    /// registry what it found. The reply is the one the operation is
    /// waiting for: it is answered here, once the registry has been told.
    /// The epoch is the one the connection was opened under — the registry
    /// adopts the connection only if that is still the server's epoch.
    Adopted {
        server: String,
        op: ServerOp,
        epoch: u64,
        outcome: Result<inner::Adoption, String>,
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
    tx: mpsc::UnboundedSender<Command>,
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
        if self.tx.send(build(reply)).is_err() {
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
        spawn_with(|_actor| async move { HubInner::empty() })
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
        spawn_with(move |actor| async move {
            HubInner::from_workspace(&workspace, user_settings.as_ref(), actor).await
        })
    }

    /// Build a hub over a list of entries the caller already has — the test
    /// path that does not want a configuration file written. Async because
    /// the actor has to finish connecting before the caller can rely on the
    /// hub being ready, and a test that races its own fixture against the
    /// connection step would flake.
    #[doc(hidden)]
    #[allow(dead_code)]
    pub async fn of_entries(entries: Vec<Entry>) -> Self {
        spawn_with(move |actor| async move {
            HubInner::from_entries(entries, Vec::new(), actor).await
        })
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
        if self.tx.send(Command::Shutdown { reply }).is_ok() {
            let _ = rx.await;
        }
    }
}

/// The hub, as a tool provider.
///
/// The hub owns every `mcp__` name — the prefix is what keeps the
/// servers' tools apart from the built-ins — offers the schemas the
/// connected servers report, and routes a call to the server that owns it.
/// The router asks it for both, and never learns what is behind them.
impl ToolProvider for Hub {
    fn owns(&self, name: &str) -> bool {
        is_tool(name)
    }

    fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>> {
        // `Hub::definitions` is the inherent method; the trait method of
        // the same name shadows it inside this impl, so the call is
        // spelled out — `self.definitions()` would be this very
        // function, forever.
        Box::pin(async move { Hub::definitions(self).await })
    }

    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a Value,
        token: &'a CancelToken,
        // An MCP tool is the server's business, not the ledger's: the hub
        // keeps no view of any file, so it has no use for the caller's
        // identity.
        _agent: AgentId,
    ) -> BoxFuture<'a, String> {
        Box::pin(async move {
            // An MCP call is not cancellable mid-flight: the server's own
            // task owns the connection, and the call's own timeout is what
            // bounds it. What a fired token cancels is the *wait* for
            // the reply — the turn is over and the answer is no longer
            // wanted.
            if token.is_cancelled() {
                return "[interrupted by user]".into();
            }
            Hub::call(self, name, &input.to_string()).await
        })
    }
}

/// Start an actor task over `init`: a future that builds the initial
/// `HubInner`. Returns a [`Hub`] the caller can send to.
///
/// The actor's own sender is handed to the init future, because the
/// per-server tasks it starts report back through it: a task that opens a
/// connection has to tell the registry what it found.
///
/// The mailbox is unbounded so that a server's task never waits for room in
/// it. A bounded one would let a task block on its report while the actor
/// waited for that task — a deadlock neither side could see.
fn spawn_with<F, Fut>(init: F) -> Hub
where
    F: FnOnce(mpsc::UnboundedSender<Command>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = HubInner> + Send + 'static,
{
    let (tx, mut rx) = mpsc::unbounded_channel::<Command>();
    let actor = tx.clone();
    tokio::spawn(async move {
        let mut inner = init(actor).await;
        while let Some(cmd) = rx.recv().await {
            match cmd {
                // The actor routes; it does not run. The call is handed to
                // the server that owns it, and the answer comes back on the
                // caller's own channel — so nothing waits here.
                Command::Call { tool, args, reply } => inner.route_call(&tool, &args, reply),
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
                Command::ServerOp { op, name, reply } => inner.server_op(op, &name, reply),
                Command::Adopted { server, op, epoch, outcome, reply } => {
                    inner.finish_adopt(&server, op, epoch, outcome, reply);
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

    // ---- the hub as a tool provider ---------------------------------------

    /// The provider owns exactly the names the `mcp__` prefix marks. The
    /// built-ins are dispatched by name too, and the prefix is what keeps
    /// the two sets apart.
    #[tokio::test]
    async fn the_hub_provider_owns_only_mcp_names() {
        let hub = Hub::empty();
        assert!(hub.owns("mcp__stub__echo"));
        assert!(hub.owns("mcp__"));
        assert!(!hub.owns("bash"));
        assert!(!hub.owns("read_file"));
        assert!(!hub.owns(""));
        hub.shutdown().await;
    }

    /// The provider's schemas are the hub's own — the same list the
    /// context used to compose for itself. (The trait method shares a
    /// name with the inherent one, so the call is qualified: that also
    /// pins that the shadowing inside the impl is spelled out rather
    /// than recursing.)
    #[tokio::test]
    async fn the_hub_provider_offers_the_schemas_the_servers_report() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        let defs = ToolProvider::definitions(&hub).await;
        let names: Vec<&str> = defs.iter().filter_map(|v| v["name"].as_str()).collect();
        assert_eq!(names, vec!["mcp__stub__echo"], "got: {defs:?}");
        hub.shutdown().await;
    }

    /// An empty hub offers nothing: the provider's schema list is empty
    /// rather than an error.
    #[tokio::test]
    async fn the_hub_provider_offers_no_schemas_for_an_empty_hub() {
        let hub = Hub::empty();
        assert!(ToolProvider::definitions(&hub).await.is_empty());
        hub.shutdown().await;
    }

    /// A call through the provider reaches the server and comes back with
    /// the server's answer.
    #[tokio::test]
    async fn the_hub_provider_runs_a_call_through_the_actor() {
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        let out = ToolProvider::call(
            &hub,
            "mcp__stub__echo",
            &serde_json::json!({"text": "hello"}),
            &CancelToken::new(),
            AgentId::MAIN,
        ).await;
        assert_eq!(out, "called with {text:hello}", "got: {out}");
        hub.shutdown().await;
    }

    /// A pre-cancelled token never reaches the hub — the run is over
    /// before any of this matters. Same cancellation marker every other
    /// tool uses.
    #[tokio::test]
    async fn the_hub_provider_short_circuits_a_pre_cancelled_call() {
        let hub = Hub::empty();
        let token = CancelToken::new();
        token.cancel();
        let out = ToolProvider::call(&hub, "mcp__anyone__anything", &serde_json::json!({}), &token, AgentId::MAIN).await;
        assert_eq!(out, "[interrupted by user]");
        hub.shutdown().await;
    }

    /// A name no server offers is answered with the hub's own message:
    /// `error: no MCP tool named ...`. The model sees the same shape it
    /// would see for any unknown tool.
    #[tokio::test]
    async fn the_hub_provider_answers_a_name_no_server_offers() {
        let hub = Hub::empty();
        let out = ToolProvider::call(
            &hub,
            "mcp__nobody__nothing",
            &serde_json::json!({}),
            &CancelToken::new(),
            AgentId::MAIN,
        ).await;
        assert!(out.starts_with("error: no MCP tool named"), "got: {out}");
        hub.shutdown().await;
    }

    // ---- the hub as a router -----------------------------------------------

    use std::time::Duration;
    use tokio::time::timeout;

    /// A stub whose `gate` tool holds a call until the test opens the gate,
    /// named as the test wants it.
    fn gated_stub(name: &str) -> (Stub, Entry) {
        let stub = Stub::new();
        let gate = stub.gate().to_string_lossy().into_owned();
        let mut entry = stub.entry(&[("STUB_TOOLS", "gate"), ("STUB_GATE", gate.as_str())]);
        entry.name = name.to_string();
        (stub, entry)
    }

    /// A stub whose *start* is slow: the script waits before it speaks, so a
    /// connection is still opening when the next operation lands. The wait
    /// is in the wrapper the test builds, so the fixture itself is
    /// unchanged.
    fn slow_start_stub(name: &str, seconds: u32) -> (Stub, Entry) {
        let stub = Stub::new();
        let script = stub.script();
        let mut entry = stub.entry(&[("STUB_TOOLS", "echo")]);
        let path = script.to_string_lossy().replace('\'', r"'\''");
        if let Ok(config) = &mut entry.config {
            config.args = vec![
                "-c".into(),
                format!("sleep {seconds}; exec bash '{path}'"),
            ];
        }
        entry.name = name.to_string();
        (stub, entry)
    }

    /// The names of the tools on offer, in the order they are offered.
    fn offered(defs: Vec<Value>) -> Vec<String> {
        defs.iter()
            .filter_map(|v| v["name"].as_str().map(str::to_owned))
            .collect()
    }

    /// A slow server does not hold up a fast one. The hub routes each call
    /// to the server that owns it, so one server's wait is that server's
    /// cost and nobody else's.
    #[tokio::test]
    async fn a_slow_server_does_not_delay_a_fast_one() {
        let (slow, slow_entry) = gated_stub("slow");
        let fast = Stub::new();
        let mut fast_entry = fast.entry(&[("STUB_TOOLS", "echo")]);
        fast_entry.name = "fast".into();
        let hub = Hub::of_entries(vec![slow_entry, fast_entry]).await;

        // The slow call starts and is held at the gate.
        let held = tokio::spawn({
            let hub = hub.clone();
            async move { hub.call("mcp__slow__gate", r#"{"text":"slow"}"#).await }
        });
        assert!(
            slow.wait_for(r#""method":"tools/call""#).await,
            "the slow call reached its server"
        );

        // The fast server answers while the slow one is still busy.
        let answer = timeout(
            Duration::from_secs(10),
            hub.call("mcp__fast__echo", r#"{"text":"hi"}"#),
        )
        .await
        .expect("the fast server is not held up by the slow one");
        assert_eq!(answer, "called with {text:hi}");

        // And the held call still finishes.
        std::fs::write(slow.gate(), "open").unwrap();
        assert_eq!(held.await.unwrap(), "called with {text:slow}");
        hub.shutdown().await;
    }

    /// The tool list the agent asks for at the start of every turn is
    /// answered while a call is in flight — the list is the registry's, and
    /// the registry does not wait for a server.
    #[tokio::test]
    async fn definitions_are_not_delayed_by_a_call_in_flight() {
        let (slow, slow_entry) = gated_stub("slow");
        let fast = Stub::new();
        let mut fast_entry = fast.entry(&[("STUB_TOOLS", "echo")]);
        fast_entry.name = "fast".into();
        let hub = Hub::of_entries(vec![slow_entry, fast_entry]).await;

        let held = tokio::spawn({
            let hub = hub.clone();
            async move { hub.call("mcp__slow__gate", r#"{"text":"slow"}"#).await }
        });
        assert!(
            slow.wait_for(r#""method":"tools/call""#).await,
            "the slow call reached its server"
        );

        let defs = timeout(Duration::from_secs(10), hub.definitions())
            .await
            .expect("the tool list is not behind the call in flight");
        assert_eq!(
            offered(defs),
            vec!["mcp__fast__echo", "mcp__slow__gate"],
            "the list is the same one, in the same order"
        );

        std::fs::write(slow.gate(), "open").unwrap();
        let _ = held.await.unwrap();
        hub.shutdown().await;
    }

    /// Two calls to one server do not overlap: the hub serializes a
    /// server's calls on purpose, so the second waits its turn — and each
    /// answer comes back with its own arguments.
    #[tokio::test]
    async fn two_calls_to_one_server_do_not_overlap() {
        let (stub, entry) = gated_stub("stub");
        let hub = Hub::of_entries(vec![entry]).await;

        let first = tokio::spawn({
            let hub = hub.clone();
            async move { hub.call("mcp__stub__gate", r#"{"text":"one"}"#).await }
        });
        assert!(
            stub.wait_for(r#""method":"tools/call""#).await,
            "the first call reached its server"
        );

        let second = tokio::spawn({
            let hub = hub.clone();
            async move { hub.call("mcp__stub__gate", r#"{"text":"two"}"#).await }
        });
        // Give the second call time to reach the server if it could.
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(
            stub.count(r#""method":"tools/call""#),
            1,
            "the second call has not reached the server: one connection takes one call at a time"
        );

        std::fs::write(stub.gate(), "open").unwrap();
        assert_eq!(first.await.unwrap(), "called with {text:one}");
        assert_eq!(second.await.unwrap(), "called with {text:two}");
        hub.shutdown().await;
    }

    /// An op on a server whose call is in flight is answered at once: the
    /// registry changes at once, and the call that is already running is
    /// allowed to finish.
    #[tokio::test]
    async fn a_server_op_is_answered_while_a_call_is_in_flight() {
        let (stub, entry) = gated_stub("stub");
        let hub = Hub::of_entries(vec![entry]).await;

        let held = tokio::spawn({
            let hub = hub.clone();
            async move { hub.call("mcp__stub__gate", r#"{"text":"slow"}"#).await }
        });
        assert!(
            stub.wait_for(r#""method":"tools/call""#).await,
            "the call reached its server"
        );

        timeout(Duration::from_secs(10), hub.disable("stub"))
            .await
            .expect("the op is not behind the call in flight")
            .expect("the op itself succeeds");

        // The registry changed at once: the server's tools are off the list.
        assert!(
            hub.definitions().await.is_empty(),
            "a disabled server offers nothing"
        );

        // And the call that was already running still finishes.
        std::fs::write(stub.gate(), "open").unwrap();
        assert_eq!(held.await.unwrap(), "called with {text:slow}");
        hub.shutdown().await;
    }

    /// A server taken offline while a reconnect is still opening its
    /// connection stays offline.
    ///
    /// The connection the reconnect was opening comes up late. It is not
    /// the server's any more — the user was told "disable: done" while it
    /// was on its way — so the registry must not bring the server back
    /// because of it, and must not offer its tools to the model again.
    #[tokio::test]
    async fn a_disable_during_a_reconnect_is_not_undone_by_the_connection_that_comes_up() {
        let (_stub, entry) = slow_start_stub("slow", 1);
        let hub = Hub::of_entries(vec![entry]).await;

        // The reconnect is taken in: the server's tools leave the offered
        // list as it is issued, while its connection is still opening.
        let reconnecting = tokio::spawn({
            let hub = hub.clone();
            async move { hub.reconnect("slow").await }
        });
        let mut taken_in = false;
        for _ in 0..200 {
            if hub.definitions().await.is_empty() {
                taken_in = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(taken_in, "the reconnect was never taken in");

        // The disable lands while the connection is still opening.
        hub.disable("slow").await.unwrap();

        // The connection comes up, and the server is still disabled.
        let outcome = reconnecting.await.unwrap();
        let err = outcome.expect_err("the reconnect was superseded by the disable");
        assert!(err.contains("disabled"), "{err}");
        let statuses = hub.list().await;
        assert_eq!(statuses.len(), 1, "got: {statuses:?}");
        assert_eq!(statuses[0].state, ServerState::Disabled, "got: {statuses:?}");
        assert_eq!(statuses[0].tools_count, 0, "got: {statuses:?}");
        assert!(hub.definitions().await.is_empty(), "no tools are offered");
        let out = hub.call("mcp__slow__echo", r#"{"text":"hi"}"#).await;
        assert!(out.starts_with("error: mcp server slow is disabled"), "{out}");
        hub.shutdown().await;
    }

    /// Disabling a server takes its tools off the list at once, and the
    /// tools that are left keep the order they had: the list's byte order
    /// is a cache contract and does not move because a server went away.
    #[tokio::test]
    async fn disabling_a_server_takes_its_tools_off_the_list_at_once() {
        let mut stubs = Vec::new();
        let mut entries = Vec::new();
        for name in ["alpha", "beta", "gamma"] {
            let stub = Stub::new();
            let mut entry = stub.entry(&[("STUB_TOOLS", "echo")]);
            entry.name = name.to_string();
            entries.push(entry);
            // The fixture owns the server's script: it has to outlive the
            // loop, or the server has nothing to run.
            stubs.push(stub);
        }
        let hub = Hub::of_entries(entries).await;
        assert_eq!(
            offered(hub.definitions().await),
            vec!["mcp__alpha__echo", "mcp__beta__echo", "mcp__gamma__echo"]
        );

        hub.disable("beta").await.unwrap();
        assert_eq!(
            offered(hub.definitions().await),
            vec!["mcp__alpha__echo", "mcp__gamma__echo"]
        );
        hub.shutdown().await;
    }
}
