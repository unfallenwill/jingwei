//! The hub's internal state: every server, the tools they offer, and where a
//! call lands.
//!
//! `Hub` is the channel that talks to this from the outside; the actor task
//! holds the only `HubInner`. The split is on purpose — the protocol and the
//! routing are the protocol layer's, and `HubInner` is the only thing that
//! has to know both at once.
//!
//! The registry holds no connection. A connection lives in the task that owns
//! it, and the registry holds the queue that task takes its work from. That
//! is what lets the actor answer `/mcp` and the tool list while a call is in
//! flight: the registry is cheap to read and is never held across a call.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

use super::client::{Connection, RemoteTool};
use super::config::{self, Entry};
use super::{Command, ServerOp};

/// What a tool the model named is: which server offers it, and what it is
/// called there.
struct Route {
    server: String,
    tool: String,
}

/// A call the registry has resolved: the server that owns it, the tool's own
/// name there, and the arguments it was given.
struct Prepared {
    server: String,
    tool: String,
    arguments: Value,
}

/// One call, handed to the server that owns it.
struct Call {
    /// The tool's own name on that server — not the public `mcp__server__tool`.
    tool: String,
    /// The arguments, already known to be a JSON object.
    arguments: Value,
    /// Where the answer goes. The task sends it; nobody waits for it here.
    reply: oneshot::Sender<String>,
}

/// What the actor says to one server's task.
enum ServerMsg {
    /// Run one call on this server's connection.
    Call(Call),
    /// Open a fresh connection with this configuration and take it over. The
    /// reply is answered by the actor, once the registry has been told what
    /// the new connection offers.
    Adopt {
        op: ServerOp,
        config: config::ServerConfig,
        /// The registry's epoch when this was asked for. The connection is
        /// the server's only if that is still the server's epoch when it
        /// comes up.
        epoch: u64,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Close the connection at the next safe point, and keep the task.
    Retire,
    /// Close the connection and end the task.
    Stop { done: oneshot::Sender<()> },
}

/// What a task reports when it has opened a connection: enough for the
/// registry to describe the server and to rebuild its routes.
pub(super) struct Adoption {
    pub(super) identity: String,
    pub(super) protocol: String,
    pub(super) tools: Vec<RemoteTool>,
    pub(super) notes: Vec<String>,
}

/// The state of one server, as the hub sees it. The public summary type:
/// external callers do not need to see what the internal `Ready` holds, so
/// this flat enum fits a `Vec` for `/mcp list`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerState {
    /// It came up and is offering its tools to the model.
    Ready,
    /// It did not, and this is what there is to say about it.
    Failed(String),
    /// It was deliberately taken out of service. The connection has been
    /// closed; the configuration is still on file and a reconnect is one
    /// command away.
    Disabled,
    /// Its connection was closed on purpose but its configuration has not
    /// been touched. A `reconnect` brings it back; an `enable` would also.
    Disconnected,
}

/// What `/mcp list` reports for one server: the name, the state it stands
/// in, and how many of its tools are currently being offered to the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerStatus {
    pub name: String,
    pub state: ServerState,
    pub tools_count: usize,
}

/// The state of one server, as the hub sees it.
enum InternalState {
    /// It came up, and it offered these tools (the server's own names, in the
    /// order the connection reported them). The connection itself lives in
    /// the server's task; what is kept here is what `/mcp` shows and what a
    /// call needs to find it.
    Ready {
        identity: String,
        protocol: String,
        tools: Vec<String>,
    },
    /// It did not, and this is what there is to say about it.
    Failed(String),
    /// It was deliberately taken out of service. The configuration is kept
    /// so an `enable` can bring it back without rereading the file.
    Disabled,
    /// Its connection was closed; the configuration is kept for the same
    /// reason as `Disabled`.
    Disconnected,
}

impl InternalState {
    /// The summary callers see, without the connection's details inside.
    fn summary(&self) -> ServerState {
        match self {
            InternalState::Ready { .. } => ServerState::Ready,
            InternalState::Failed(why) => ServerState::Failed(why.clone()),
            InternalState::Disabled => ServerState::Disabled,
            InternalState::Disconnected => ServerState::Disconnected,
        }
    }
}

/// Everything `Hub` knows about one configured server — both the parts the
/// configuration file said (name, what it is, where it came from) and the
/// parts the connection attempt found out (the state it stands in).
struct ServerEntry {
    /// What the entry says it is — the command line, or the url — for the
    /// report. Readable rather than parseable: nothing decides anything by it.
    how: String,
    /// The file that named it, so that a server in a project's `.mcp.json` is
    /// told apart from one in the user's own settings.
    from: String,
    /// The parsed configuration, kept around so a reconnect does not have
    /// to re-read the file. `Err` is the reason the connection itself
    /// failed, kept here so the report can name it without having to look
    /// up the file again.
    config: Result<config::ServerConfig, String>,
    state: InternalState,
    /// How many times this server has been taken offline.
    ///
    /// A connection is opened under the epoch that was current when it was
    /// asked for, and the number travels with it. If the server has been
    /// taken offline since, the connection that came up is not the server's
    /// any more: the registry has already said what it wants, and a
    /// connection that arrives late must not undo that.
    epoch: u64,
    /// The task that owns this server's connection, and the queue it takes
    /// calls from. Present for *every* configured server — including one
    /// that failed to start — because `enable` and `reconnect` have to be
    /// able to reach a server whose connection was closed.
    queue: mpsc::UnboundedSender<ServerMsg>,
}

/// Every server a session talks to, and the tools they offer between them.
///
/// The actor task holds one of these and never shares it. The order the
/// tools are offered in is the order the servers are in the configuration —
/// sorted by name — and, within a server, the order it listed them. It is
/// part of every request the session sends, so it has to be the same on
/// every run of the same configuration.
pub(super) struct HubInner {
    servers: BTreeMap<String, ServerEntry>,
    /// The public names of the tools each ready server offers, in the order
    /// the connection reported them. Used to rebuild the offered list at
    /// `definitions()` time — keeping the order as a per-server list lets
    /// enable / disable mutate the set of offered tools without re-shuffling
    /// the others.
    server_tools: BTreeMap<String, Vec<String>>,
    /// Tool definitions, keyed by the public name the model sees. Each value
    /// is the wire-shaped JSON object the agent will append to its schemas
    /// list: `{"name": ..., "description": ..., "input_schema": ...}`.
    /// Storing the wire shape here means `definitions()` is a clone, not a
    /// rebuild, and the conversion lives in exactly one place.
    offered: HashMap<String, Value>,
    /// Public name -> which server offers it and what it is called there.
    routes: HashMap<String, Route>,
    /// What had to be said on the way: an entry that cannot be used, a
    /// variable that is not set, two tools that would carry one name.
    warnings: Vec<String>,
}

impl HubInner {
    /// Nobody to talk to: the hub a session has when there is no
    /// configuration, and the one every test that is not about MCP gets.
    pub(super) fn empty() -> Self {
        Self {
            servers: BTreeMap::new(),
            server_tools: BTreeMap::new(),
            offered: HashMap::new(),
            routes: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    /// Read the configuration and connect to everything it names.
    ///
    /// All of them at once: a session's first request waits for the tools of
    /// every server, and doing this one server at a time would make that wait
    /// the sum of theirs. A server that is slow to start is slow either way,
    /// and this way it is no one else's cost.
    ///
    /// `actor` is the mailbox the server tasks report back through: a task
    /// that opens a connection has to tell the registry what it found.
    pub(super) async fn from_workspace(
        workspace: &Path,
        user_settings: Option<&Value>,
        actor: mpsc::UnboundedSender<Command>,
    ) -> Self {
        let (entries, warnings) = config::entries(workspace, user_settings);
        Self::from_entries(entries, warnings, actor).await
    }

    /// The same over entries that are already read: the file is where a
    /// session gets them from, and a test that has its own entries — a stub
    /// server, and no file to write — has no reason to go through one.
    pub(super) async fn from_entries(
        entries: Vec<Entry>,
        warnings: Vec<String>,
        actor: mpsc::UnboundedSender<Command>,
    ) -> Self {
        let opened = futures_util::future::join_all(entries.iter().map(|entry| async move {
            match &entry.config {
                Err(why) => Err(why.clone()),
                Ok(config) => Connection::open(config).await,
            }
        }))
        .await;
        Self::assemble(entries, opened, warnings, actor)
    }

    /// Build the hub from entries and whatever came of opening them: what
    /// the names are, which of them are offered, and what has to be said.
    ///
    /// Every entry gets a task, even one whose connection never came up: a
    /// `reconnect` has to be able to reach a server that failed to start.
    fn assemble(
        entries: Vec<Entry>,
        opened: Vec<Result<Connection, String>>,
        warnings: Vec<String>,
        actor: mpsc::UnboundedSender<Command>,
    ) -> Self {
        let mut inner = Self {
            servers: BTreeMap::new(),
            server_tools: BTreeMap::new(),
            offered: HashMap::new(),
            routes: HashMap::new(),
            warnings,
        };
        for (entry, opened) in entries.into_iter().zip(opened) {
            let Entry { name, from, config } = entry;
            let how = how_to_reach(&config);
            match opened {
                Err(why) => {
                    let queue = spawn_server(name.clone(), None, actor.clone());
                    inner.servers.insert(
                        name,
                        ServerEntry {
                            how,
                            from,
                            config,
                            state: InternalState::Failed(why),
                            epoch: 0,
                            queue,
                        },
                    );
                }
                Ok(mut connection) => {
                    // The connection hands its tool list to the registry and
                    // keeps the connection itself: the task owns that.
                    let identity = connection.identity.clone();
                    let protocol = connection.protocol.clone();
                    let tools = std::mem::take(&mut connection.tools);
                    let notes = std::mem::take(&mut connection.notes);
                    let offered_names = inner.add_tools(&name, &tools);
                    for note in &notes {
                        inner.warnings.push(format!("{name}: {note}"));
                    }
                    let queue = spawn_server(name.clone(), Some(connection), actor.clone());
                    inner.servers.insert(
                        name,
                        ServerEntry {
                            how,
                            from,
                            config,
                            state: InternalState::Ready {
                                identity,
                                protocol,
                                tools: offered_names,
                            },
                            epoch: 0,
                            queue,
                        },
                    );
                }
            }
        }
        inner
    }

    /// Route one call to the server that owns it. Never awaits.
    ///
    /// The registry answers what it can answer at once — a name nobody
    /// offers, a server that is not in service, arguments that are not an
    /// object — and hands everything else to the server's own task, which
    /// answers the caller when the call comes back.
    pub(super) fn route_call(&self, name: &str, args_json: &str, reply: oneshot::Sender<String>) {
        let prepared = match self.prepare(name, args_json) {
            Ok(prepared) => prepared,
            Err(message) => {
                let _ = reply.send(message);
                return;
            }
        };
        let Prepared { server, tool, arguments } = prepared;
        let Some(entry) = self.servers.get(&server) else {
            // Unreachable: a route exists only for a server that came up. Said
            // rather than assumed away, because a panic here would end a turn.
            let _ = reply.send(format!("error: mcp server {server} is not connected"));
            return;
        };
        let call = Call { tool, arguments, reply };
        if let Err(sent) = entry.queue.send(ServerMsg::Call(call)) {
            // The task is gone. The call still gets an answer: a call that is
            // dropped looks to the model like a tool that returned nothing.
            if let ServerMsg::Call(call) = sent.0 {
                let _ = call.reply.send(format!("error: mcp server {server} is not connected"));
            }
        }
    }

    /// Everything the registry can say about a call without touching a
    /// server: where it lands, and — when there is nowhere to send it — the
    /// sentence the model reads instead.
    fn prepare(&self, name: &str, args_json: &str) -> Result<Prepared, String> {
        let Some(route) = self.routes.get(name) else {
            return Err(format!(
                "error: no MCP tool named {name:?} is offered. The MCP tools this session has \
                 are: {}",
                self.offered_names()
            ));
        };
        let Some(server) = self.servers.get(&route.server) else {
            return Err(format!("error: mcp server {} is not connected", route.server));
        };
        match &server.state {
            InternalState::Ready { .. } => {}
            InternalState::Failed(why) => {
                return Err(format!(
                    "error: mcp server {} could not run {}: {why}",
                    route.server, route.tool
                ));
            }
            InternalState::Disabled => {
                return Err(format!(
                    "error: mcp server {} is disabled; enable it with /mcp enable {}",
                    route.server, route.server
                ));
            }
            InternalState::Disconnected => {
                return Err(format!(
                    "error: mcp server {} is disconnected; reconnect it with /mcp reconnect {}",
                    route.server, route.server
                ));
            }
        }
        let arguments = match serde_json::from_str::<Value>(args_json) {
            Ok(Value::Object(fields)) => Value::Object(fields),
            Ok(_) => {
                return Err(format!("error: the arguments of {name} have to be a JSON object"));
            }
            Err(e) => {
                return Err(format!("error: the arguments of {name} are not valid JSON: {e}"));
            }
        };
        Ok(Prepared {
            server: route.server.clone(),
            tool: route.tool.clone(),
            arguments,
        })
    }

    /// The tools currently offered, in the order a request should send them
    /// in. Returned as a `Vec<Value>` so the agent loop can append straight
    /// to its schemas list without rebuilding the wire shape.
    pub(super) fn definitions(&self) -> Vec<Value> {
        self.tool_order()
            .filter_map(|n| self.offered.get(n))
            .cloned()
            .collect()
    }

    /// What the session says about MCP when it starts: one line for the
    /// servers that came up, one for each that did not, and the notes from
    /// the way in.
    pub(super) fn notes(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let ready: Vec<&str> = self
            .servers
            .iter()
            .filter_map(|(name, entry)| match entry.state {
                InternalState::Ready { .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        if !ready.is_empty() {
            lines.push(format!(
                "mcp: {} · {} tools · /mcp names them",
                ready.join(", "),
                self.offered.len()
            ));
        }
        for (name, entry) in &self.servers {
            match &entry.state {
                InternalState::Failed(why) => {
                    lines.push(format!("mcp: {name} did not start: {why}"));
                }
                InternalState::Disabled => lines.push(format!("mcp: {name} is disabled")),
                InternalState::Disconnected => {
                    lines.push(format!("mcp: {name} is disconnected"));
                }
                InternalState::Ready { .. } => {}
            }
        }
        lines.extend(self.warnings.iter().map(|note| format!("mcp: {note}")));
        lines
    }

    /// What `/mcp` prints: every server, what it offers, and what is wrong.
    pub(super) fn report(&self) -> Vec<String> {
        const NAMES_SHOWN: usize = 12;
        if self.servers.is_empty() && self.warnings.is_empty() {
            return vec![
                "no MCP servers: add them to ~/.jingwei/mcp.json (user) or .mcp.json (this workspace)"
                    .to_string(),
            ];
        }
        let mut lines = Vec::new();
        for (name, entry) in &self.servers {
            match &entry.state {
                InternalState::Ready { identity, protocol, tools } => lines.push(format!(
                    "{name} · {} · {} tools · {identity} · protocol {protocol} · from {}",
                    entry.how,
                    tools.len(),
                    entry.from
                )),
                InternalState::Failed(why) => lines.push(format!(
                    "{name} · {} · did not start: {why} · from {}",
                    entry.how, entry.from
                )),
                InternalState::Disabled => lines.push(format!(
                    "{name} · {} · disabled · from {}",
                    entry.how, entry.from
                )),
                InternalState::Disconnected => lines.push(format!(
                    "{name} · {} · disconnected · from {}",
                    entry.how, entry.from
                )),
            }
            if let Some(names) = self.server_tools.get(name) {
                if !names.is_empty() {
                    let shown: Vec<&str> = names.iter().take(NAMES_SHOWN).map(String::as_str).collect();
                    let rest = names.len() - shown.len();
                    let mut line = format!("  {}", shown.join(", "));
                    if rest > 0 {
                        line.push_str(&format!(" (and {rest} more)"));
                    }
                    lines.push(line);
                }
            }
        }
        lines.extend(self.warnings.iter().map(|note| format!("warning: {note}")));
        lines
    }

    /// One line per server, with the state it stands in and how many of its
    /// tools are being offered. What `/mcp list` prints.
    pub(super) fn list(&self) -> Vec<ServerStatus> {
        self.servers
            .iter()
            .map(|(name, entry)| ServerStatus {
                name: name.clone(),
                state: entry.state.summary(),
                tools_count: match &entry.state {
                    InternalState::Ready { .. } => {
                        self.server_tools.get(name).map(Vec::len).unwrap_or(0)
                    }
                    _ => 0,
                },
            })
            .collect()
    }

    /// Apply one server operation. Never awaits.
    ///
    /// The registry's half of the work is done at once — the state changes,
    /// and the tool list changes with it. The connection's half is handed to
    /// the server's task, which does it at the server's next safe point: a
    /// call already running on that server finishes, and the operation does
    /// not cancel it and does not wait for it.
    pub(super) fn server_op(
        &mut self,
        op: ServerOp,
        name: &str,
        reply: oneshot::Sender<Result<(), String>>,
    ) {
        match op {
            ServerOp::Enable => self.enable(name, reply),
            ServerOp::Disable => self.take_offline(name, InternalState::Disabled, reply),
            ServerOp::Reconnect => self.reconnect(name, reply),
            ServerOp::Disconnect => self.take_offline(name, InternalState::Disconnected, reply),
        }
    }

    /// Bring a `Disabled` server back online: open a fresh connection with
    /// its stored configuration, and route its tools to the model again.
    ///
    /// The connection is opened by the server's own task, so the actor is
    /// free while it is being opened; the reply comes when the new
    /// connection is up, or with the reason it did not come up.
    fn enable(&mut self, name: &str, reply: oneshot::Sender<Result<(), String>>) {
        let Some(entry) = self.servers.get(name) else {
            let _ = reply.send(Err(format!("no MCP server named {name:?} is configured")));
            return;
        };
        match &entry.state {
            InternalState::Disabled => {}
            InternalState::Ready { .. } => {
                let _ = reply.send(Err(format!("mcp server {name} is already enabled")));
                return;
            }
            InternalState::Failed(why) => {
                let _ = reply.send(Err(format!("mcp server {name} failed to start: {why}")));
                return;
            }
            InternalState::Disconnected => {
                let _ = reply.send(Err(format!(
                    "mcp server {name} is disconnected; reconnect instead"
                )));
                return;
            }
        }
        let config = match entry.config.as_ref() {
            Ok(config) => config.clone(),
            Err(why) => {
                let _ = reply.send(Err(format!("mcp server {name} cannot be enabled: {why}")));
                return;
            }
        };
        let queue = entry.queue.clone();
        let epoch = entry.epoch;
        if let Err(sent) = queue.send(ServerMsg::Adopt { op: ServerOp::Enable, config, epoch, reply }) {
            if let ServerMsg::Adopt { reply, .. } = sent.0 {
                let _ = reply.send(Err(format!("mcp server {name} is not connected")));
            }
        }
    }

    /// Take a `Ready` server offline: the registry forgets its tools at
    /// once, and its connection is closed at the server's next safe point.
    /// The configuration is kept so an `enable` can bring the server back.
    fn take_offline(
        &mut self,
        name: &str,
        state: InternalState,
        reply: oneshot::Sender<Result<(), String>>,
    ) {
        let refused = match self.servers.get(name) {
            None => Some(format!("no MCP server named {name:?} is configured")),
            Some(entry) => match &entry.state {
                InternalState::Ready { .. } => None,
                InternalState::Failed(why) => Some(format!("mcp server {name} failed to start: {why}")),
                InternalState::Disabled => Some(format!("mcp server {name} is disabled")),
                InternalState::Disconnected => Some(format!("mcp server {name} is disconnected")),
            },
        };
        if let Some(refused) = refused {
            let _ = reply.send(Err(refused));
            return;
        }
        let queue = self
            .servers
            .get(name)
            .expect("the server was there a moment ago")
            .queue
            .clone();
        {
            let entry = self
                .servers
                .get_mut(name)
                .expect("the server was there a moment ago");
            // Taking the server offline is what supersedes a connection that
            // is still on its way: the epoch moves, and a connection opened
            // under the old one is not this server's when it arrives.
            entry.epoch += 1;
            entry.state = state;
        }
        self.remove_server_tools(name);
        let _ = queue.send(ServerMsg::Retire);
        let _ = reply.send(Ok(()));
    }

    /// Reconnect a server in any state: close any live connection, then
    /// open a fresh one with the stored configuration.
    fn reconnect(&mut self, name: &str, reply: oneshot::Sender<Result<(), String>>) {
        let Some(entry) = self.servers.get(name) else {
            let _ = reply.send(Err(format!("no MCP server named {name:?} is configured")));
            return;
        };
        let config = match entry.config.as_ref() {
            Ok(config) => config.clone(),
            Err(why) => {
                let _ = reply.send(Err(format!("mcp server {name} cannot be reconnected: {why}")));
                return;
            }
        };
        let queue = entry.queue.clone();
        let epoch = entry.epoch;
        // The registry forgets the server's tools at once: the connection
        // that is being replaced is on its way out, and the model should
        // not be offered a tool whose server is in the middle of a change.
        self.remove_server_tools(name);
        if let Err(sent) = queue.send(ServerMsg::Adopt { op: ServerOp::Reconnect, config, epoch, reply }) {
            if let ServerMsg::Adopt { reply, .. } = sent.0 {
                let _ = reply.send(Err(format!("mcp server {name} is not connected")));
            }
        }
    }

    /// The actor's half of an adopt: the new connection is up, so the
    /// registry can describe the server again and the operation can be
    /// answered.
    pub(super) fn finish_adopt(
        &mut self,
        server: &str,
        op: ServerOp,
        epoch: u64,
        outcome: Result<Adoption, String>,
        reply: oneshot::Sender<Result<(), String>>,
    ) {
        // The connection is the server's only if nothing has been done to
        // the server since it was asked for. A server that was taken
        // offline while the connection was opening keeps the state it was
        // taken to: the registry does not bring a server back because a
        // connection it had already given up on came up late.
        if self.servers.get(server).map(|entry| entry.epoch) != Some(epoch) {
            let _ = reply.send(Err(self.superseded(server, op)));
            return;
        }
        match outcome {
            Ok(adoption) => {
                let Adoption { identity, protocol, tools, notes } = adoption;
                let names = self.adopt_tools(server, &tools);
                for note in &notes {
                    self.warnings.push(format!("{server}: {note}"));
                }
                if let Some(entry) = self.servers.get_mut(server) {
                    entry.state = InternalState::Ready { identity, protocol, tools: names };
                }
                let _ = reply.send(Ok(()));
            }
            Err(why) => {
                match op {
                    // An `enable` that could not open its connection leaves
                    // the server where it was — disabled — and says why.
                    ServerOp::Enable => {
                        let _ = reply.send(Err(why));
                    }
                    _ => {
                        if let Some(entry) = self.servers.get_mut(server) {
                            entry.state = InternalState::Failed(why);
                        }
                        let _ = reply.send(Err(format!("mcp server {server} failed to reconnect")));
                    }
                }
            }
        }
    }

    /// What to say when a connection came up after the server it was opened
    /// for was taken offline: the operation did not happen, and what stopped
    /// it is named.
    fn superseded(&self, server: &str, op: ServerOp) -> String {
        let what = match self.servers.get(server).map(|entry| &entry.state) {
            Some(InternalState::Disabled) => "disabled",
            Some(InternalState::Disconnected) => "disconnected",
            _ => "changed",
        };
        let busy = match op {
            ServerOp::Enable => "enabling",
            ServerOp::Reconnect => "reconnecting",
            _ => "changing",
        };
        format!("mcp server {server} was {what} while it was {busy}")
    }

    /// End every connection: what a session does on its way out, so that the
    /// programs it started do not outlive it.
    ///
    /// Every server's task is told to stop and is waited for, so the child
    /// processes are gone before this returns.
    pub(super) async fn shutdown_all(&mut self) {
        let mut stopped = Vec::new();
        for entry in self.servers.values_mut() {
            let (done, waiting) = oneshot::channel();
            if entry.queue.send(ServerMsg::Stop { done }).is_ok() {
                stopped.push(waiting);
            }
        }
        for waiting in stopped {
            let _ = waiting.await;
        }
    }

    /// The names it offers, as the model sees them, cut to the first few so
    /// that a failure stays a sentence.
    fn offered_names(&self) -> String {
        const NAMES_SHOWN: usize = 12;
        if self.offered.is_empty() {
            return "none came up (see /mcp for which servers did not)".to_string();
        }
        let names: Vec<String> = self
            .tool_order()
            .take(NAMES_SHOWN)
            .map(str::to_owned)
            .collect();
        let rest = self.offered.len() - names.len();
        let mut listed = names.join(", ");
        if rest > 0 {
            listed.push_str(&format!(" (and {rest} more)"));
        }
        listed
    }

    /// The ordered list of public tool names, ready servers only. The order
    /// is "all ready servers in name order, each one's tools in the order
    /// the connection reported them".
    fn tool_order(&self) -> impl Iterator<Item = &str> {
        self.server_tools
            .iter()
            .filter(|(name, _)| {
                self.servers
                    .get(*name)
                    .is_some_and(|e| matches!(e.state, InternalState::Ready { .. }))
            })
            .flat_map(|(_, tools)| tools.iter().map(String::as_str))
    }

    /// Pull the tools a server offers into `routes`, `offered`, and
    /// `server_tools`. Returns the server's own tool names, in the order
    /// the connection reported them.
    fn add_tools(&mut self, server: &str, tools: &[RemoteTool]) -> Vec<String> {
        let mut names = Vec::with_capacity(tools.len());
        for tool in tools {
            let name = tool_name(server, &tool.name);
            if self.routes.contains_key(&name) {
                self.warnings.push(format!(
                    "{name}: two tools would carry this name, so the one from {server} is not \
                     offered"
                ));
                continue;
            }
            self.routes.insert(
                name.clone(),
                Route {
                    server: server.to_string(),
                    tool: tool.name.clone(),
                },
            );
            // jingwei's wire shape: {name, description, input_schema}. The
            // schema is what the server returned; a missing description is
            // a missing description (the model is told as much).
            let def = json!({
                "name": name,
                "description": tool.description.clone().unwrap_or_default(),
                "input_schema": tool.schema,
            });
            self.offered.insert(name.clone(), def);
            names.push(tool.name.clone());
            self.server_tools
                .entry(server.to_string())
                .or_default()
                .push(name);
        }
        names
    }

    /// Adopt a new connection's tools: the server's stale routes go, and the
    /// tools the new connection reports are the ones offered.
    fn adopt_tools(&mut self, server: &str, tools: &[RemoteTool]) -> Vec<String> {
        self.remove_server_tools(server);
        // Find every route that belongs to this server and drop it. Done
        // against `routes` rather than `server_tools` because the latter is
        // empty after a disable or disconnect, while the former is what
        // `add_tools` checks for collisions.
        let stale: Vec<String> = self
            .routes
            .iter()
            .filter(|(_, route)| route.server == server)
            .map(|(name, _)| name.clone())
            .collect();
        for name in stale {
            self.routes.remove(&name);
        }
        self.add_tools(server, tools)
    }

    /// Remove every tool of `server` from `offered` and `server_tools`. The
    /// `routes` table is left alone — a call to a tool whose server has
    /// been disabled or disconnected should still reach the `prepare` method
    /// and get a state-aware error ("is disabled", "is disconnected")
    /// rather than a "no such tool" message that confuses the model about
    /// what changed.
    fn remove_server_tools(&mut self, server: &str) {
        if let Some(names) = self.server_tools.remove(server) {
            for name in &names {
                self.offered.remove(name);
            }
        }
    }
}

/// Start the task that owns one server's connection, and hand back the queue
/// it takes its work from.
fn spawn_server(
    name: String,
    connection: Option<Connection>,
    actor: mpsc::UnboundedSender<Command>,
) -> mpsc::UnboundedSender<ServerMsg> {
    let (queue, rx) = mpsc::unbounded_channel();
    tokio::spawn(server_task(name, connection, rx, actor));
    queue
}

/// The task that owns one server's connection.
///
/// One message at a time, until `Stop`. A call is awaited where it is
/// found, so the messages behind it wait: one connection is a
/// request/response protocol and cannot be driven by two callers at once.
/// The server's next safe point — where a retire, a reconnect or a shutdown
/// lands — is therefore wherever this loop is between two messages.
async fn server_task(
    name: String,
    mut connection: Option<Connection>,
    mut rx: mpsc::UnboundedReceiver<ServerMsg>,
    actor: mpsc::UnboundedSender<Command>,
) {
    while let Some(message) = rx.recv().await {
        match message {
            ServerMsg::Call(call) => {
                let tool = call.tool.clone();
                let answer = match connection.as_ref() {
                    Some(connection) => match connection.call(&tool, call.arguments).await {
                        Ok(text) => text,
                        Err(why) => {
                            format!("error: mcp server {name} could not run {tool}: {why}")
                        }
                    },
                    None => format!("error: mcp server {name} is not connected"),
                };
                let _ = call.reply.send(answer);
            }
            ServerMsg::Adopt { op, config, epoch, reply } => {
                // The old connection is retired here — at the next safe
                // point, which is where this message is found.
                if let Some(old) = connection.take() {
                    old.shutdown().await;
                }
                let outcome = match Connection::open(&config).await {
                    Ok(mut fresh) => {
                        let adoption = Adoption {
                            identity: fresh.identity.clone(),
                            protocol: fresh.protocol.clone(),
                            tools: std::mem::take(&mut fresh.tools),
                            notes: std::mem::take(&mut fresh.notes),
                        };
                        connection = Some(fresh);
                        Ok(adoption)
                    }
                    Err(why) => Err(why),
                };
                let _ = actor.send(Command::Adopted {
                    server: name.clone(),
                    op,
                    epoch,
                    outcome,
                    reply,
                });
            }
            ServerMsg::Retire => {
                if let Some(old) = connection.take() {
                    old.shutdown().await;
                }
            }
            ServerMsg::Stop { done } => {
                if let Some(old) = connection.take() {
                    old.shutdown().await;
                }
                let _ = done.send(());
                return;
            }
        }
    }
}

/// What the entry says it is, for the report: the command line it runs, or
/// the url it calls. Quoted where the parts carry spaces, so that a reader
/// can see where one ends and the next begins.
fn how_to_reach(config: &Result<config::ServerConfig, String>) -> String {
    let Ok(config) = config else {
        return "nothing usable in the entry".to_string();
    };
    match (&config.command, &config.url) {
        (Some(command), _) => {
            let mut line = format!("stdio: {command}");
            for arg in &config.args {
                if arg.contains(' ') {
                    line.push_str(&format!(" {arg:?}"));
                } else {
                    line.push_str(&format!(" {arg}"));
                }
            }
            line
        }
        (None, Some(url)) => url.clone(),
        (None, None) => "nowhere".to_string(),
    }
}

/// What a tool from a server is named, in front of the tool's own name.
pub(super) const PREFIX: &str = "mcp__";

/// The longest name a tool may have. OpenAI's function names are capped at 64
/// characters, and a session may be talking to any of the providers in the
/// preset table, so every name this client sends is one all of them can read.
pub(super) const MAX_NAME: usize = 64;

/// What the model sees a server's tool called: `mcp__<server>__<tool>`.
///
/// A name may carry characters a function name may not, and a name may be
/// longer than a backend will take, so both are dealt with here rather than
/// at every use: what is not a letter, a digit, an underscore or a dash
/// becomes an underscore, and a name longer than [`MAX_NAME`] keeps its front
/// and ends with a short hash of the whole. The hash is what keeps two long
/// names from becoming one: cutting a pair of similar names at the same
/// length would otherwise be enough to make them the same name.
pub(super) fn tool_name(server: &str, tool: &str) -> String {
    let name = format!("{PREFIX}{}__{}", sanitize(server), sanitize(tool));
    if name.len() <= MAX_NAME {
        return name;
    }
    let hash = short_hash(&name);
    // The name is ASCII by construction ([`sanitize`] just made it so),
    // so every byte offset is a char boundary — the loop that used to
    // sit here was belt-and-braces for a caller that one day does not.
    let cut = MAX_NAME - hash.len() - 1;
    format!("{}~{}", &name[..cut], hash)
}

/// Keep what a name may carry and replace the rest. Two different names may
/// come out the same (a space and an underscore are one character to a
/// backend) — that is a collision the hub reports rather than a fix to guess
/// at.
fn sanitize(text: &str) -> String {
    text.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect()
}

/// A short, stable hash of a name.
///
/// FNV-1a, written out rather than taken from a crate: the name it is part of
/// is written into the session log, and a later process reading that log has
/// to work out the same name from the same server — so the hash has to be
/// the same on every machine, in every release, forever. A hash that could
/// change is a hash that would silently renumber the tools of a resumed
/// session.
fn short_hash(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:08x}", hash & 0xffff_ffff)
}
