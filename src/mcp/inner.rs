//! The hub's internal state: every server, the tools they offer, and where a
//! call lands.
//!
//! `Hub` is the channel that talks to this from the outside; the actor task
//! holds the only `HubInner`. The split is on purpose — the protocol and the
//! routing are the protocol layer's, and `HubInner` is the only thing that
//! has to know both at once.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::{Value, json};

use super::client::Connection;
use super::config::{self, Entry};

/// What a tool the model named is: which server offers it, and what it is
/// called there.
struct Route {
    server: String,
    tool: String,
}

/// The state of one server, as the hub sees it.
///
/// A summary view over the internal [`ServerState`]: the external callers
/// do not need to see the [`Connection`] held inside `Ready`, so the public
/// type is a flat enum that fits a `Vec` for `/mcp list`.
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
#[allow(clippy::large_enum_variant)] // The actor task owns this; boxing Connection would
// add a heap indirection the only consumer never shares.
enum InternalState {
    /// It came up, and it offered these tools (the server's own names, in the
    /// order the connection reported them).
    Ready {
        connection: Connection,
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
    /// The summary callers see, without the `Connection` inside.
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
    pub(super) async fn from_workspace(workspace: &Path, user_settings: Option<&Value>) -> Self {
        let (entries, warnings) = config::entries(workspace, user_settings);
        Self::from_entries(entries, warnings).await
    }

    /// The same over entries that are already read: the file is where a
    /// session gets them from, and a test that has its own entries — a stub
    /// server, and no file to write — has no reason to go through one.
    pub(super) async fn from_entries(entries: Vec<Entry>, warnings: Vec<String>) -> Self {
        let opened = futures_util::future::join_all(entries.iter().map(|entry| async move {
            match &entry.config {
                Err(why) => Err(why.clone()),
                Ok(config) => Connection::open(config).await,
            }
        }))
        .await;
        Self::assemble(entries, opened, warnings)
    }

    /// Build the hub from entries and whatever came of opening them: what
    /// the names are, which of them are offered, and what has to be said.
    fn assemble(
        entries: Vec<Entry>,
        opened: Vec<Result<Connection, String>>,
        warnings: Vec<String>,
    ) -> Self {
        let mut inner = Self {
            servers: BTreeMap::new(),
            server_tools: BTreeMap::new(),
            offered: HashMap::new(),
            routes: HashMap::new(),
            warnings,
        };
        for (entry, opened) in entries.into_iter().zip(opened) {
            let how = how_to_reach(&entry);
            let connection = match opened {
                Ok(connection) => connection,
                Err(why) => {
                    inner.servers.insert(
                        entry.name.clone(),
                        ServerEntry {
                            how,
                            from: entry.from,
                            config: entry.config,
                            state: InternalState::Failed(why),
                        },
                    );
                    continue;
                }
            };
            let offered_names = inner.add_connection_tools(&entry.name, &connection);
            for note in &connection.notes {
                inner.warnings.push(format!("{}: {note}", entry.name));
            }
            inner.servers.insert(
                entry.name.clone(),
                ServerEntry {
                    how,
                    from: entry.from,
                    config: entry.config,
                    state: InternalState::Ready {
                        connection,
                        tools: offered_names,
                    },
                },
            );
        }
        inner
    }

    /// One tool call, answered with the text the model reads. Never fails:
    /// a name nobody offers, a server that died, a refusal from the far end —
    /// all of them come back as the result text.
    pub(super) async fn call(&self, name: &str, args_json: &str) -> String {
        let Some(route) = self.routes.get(name) else {
            return format!(
                "error: no MCP tool named {name:?} is offered. The MCP tools this session has \
                 are: {}",
                self.offered_names()
            );
        };
        let Some(server) = self.servers.get(&route.server) else {
            // Unreachable: a route exists only for a server that came up. Said
            // rather than assumed away, because a panic here would end a turn.
            return format!("error: mcp server {} is not connected", route.server);
        };
        let connection = match &server.state {
            InternalState::Ready { connection, .. } => connection,
            InternalState::Failed(why) => {
                return format!(
                    "error: mcp server {} could not run {}: {why}",
                    route.server, route.tool
                );
            }
            InternalState::Disabled => {
                return format!(
                    "error: mcp server {} is disabled; enable it with /mcp enable {}",
                    route.server, route.server
                );
            }
            InternalState::Disconnected => {
                return format!(
                    "error: mcp server {} is disconnected; reconnect it with /mcp reconnect {}",
                    route.server, route.server
                );
            }
        };
        let arguments = match serde_json::from_str::<Value>(args_json) {
            Ok(Value::Object(fields)) => Value::Object(fields),
            Ok(_) => {
                return format!("error: the arguments of {name} have to be a JSON object");
            }
            Err(e) => {
                return format!("error: the arguments of {name} are not valid JSON: {e}");
            }
        };
        match connection.call(&route.tool, arguments).await {
            Ok(text) => text,
            Err(why) => format!(
                "error: mcp server {} could not run {}: {why}",
                route.server, route.tool
            ),
        }
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
                InternalState::Ready { connection, tools } => lines.push(format!(
                    "{name} · {} · {} tools · {} · protocol {} · from {}",
                    entry.how,
                    tools.len(),
                    connection.identity,
                    connection.protocol,
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

    /// Bring a `Disabled` server back online: open a fresh connection with
    /// its stored configuration, and route its tools to the model again.
    pub(super) async fn enable(&mut self, name: &str) -> Result<(), String> {
        let entry = self
            .servers
            .get(name)
            .ok_or_else(|| format!("no MCP server named {name:?} is configured"))?;
        match &entry.state {
            InternalState::Disabled => {}
            InternalState::Ready { .. } => {
                return Err(format!("mcp server {name} is already enabled"));
            }
            InternalState::Failed(why) => {
                return Err(format!("mcp server {name} failed to start: {why}"));
            }
            InternalState::Disconnected => {
                return Err(format!(
                    "mcp server {name} is disconnected; reconnect instead"
                ));
            }
        }
        let config = entry
            .config
            .as_ref()
            .map_err(|why| format!("mcp server {name} cannot be enabled: {why}"))?
            .clone();
        // Drop the entry borrow before mutating `self`.
        let connection = Connection::open(&config).await?;
        self.adopt_connection(name, connection);
        Ok(())
    }

    /// Take a `Ready` server offline: close its connection and pull its
    /// tools out of the offered set. The configuration is kept so an
    /// `enable` can bring the server back.
    pub(super) async fn disable(&mut self, name: &str) -> Result<(), String> {
        let connection = self.take_ready_connection(name)?;
        connection.shutdown().await;
        self.remove_server_tools(name);
        self.servers
            .get_mut(name)
            .expect("server still present")
            .state = InternalState::Disabled;
        Ok(())
    }

    /// Reconnect a server in any state: close any live connection, then
    /// open a fresh one with the stored configuration.
    pub(super) async fn reconnect(&mut self, name: &str) -> Result<(), String> {
        let entry = self
            .servers
            .get(name)
            .ok_or_else(|| format!("no MCP server named {name:?} is configured"))?;
        let config = entry
            .config
            .as_ref()
            .map_err(|why| format!("mcp server {name} cannot be reconnected: {why}"))?
            .clone();
        // Close the old connection (if any) before opening the new one.
        if let Ok(connection) = self.take_ready_connection(name) {
            connection.shutdown().await;
        }
        // Drop any stale tool entries left over from a previous Ready state.
        self.remove_server_tools(name);
        match Connection::open(&config).await {
            Ok(connection) => {
                self.adopt_connection(name, connection);
                Ok(())
            }
            Err(why) => {
                self.servers
                    .get_mut(name)
                    .expect("server still present")
                    .state = InternalState::Failed(why);
                Err(format!("mcp server {name} failed to reconnect"))
            }
        }
    }

    /// Close a `Ready` server's connection without touching its
    /// configuration. The server sits in `Disconnected` until `reconnect`
    /// brings it back.
    pub(super) async fn disconnect(&mut self, name: &str) -> Result<(), String> {
        let connection = self.take_ready_connection(name)?;
        connection.shutdown().await;
        self.remove_server_tools(name);
        self.servers
            .get_mut(name)
            .expect("server still present")
            .state = InternalState::Disconnected;
        Ok(())
    }

    /// End every connection: what a session does on its way out, so that the
    /// programs it started do not outlive it.
    pub(super) async fn shutdown_all(&mut self) {
        let mut connections: Vec<Connection> = Vec::new();
        for entry in self.servers.values_mut() {
            if let InternalState::Ready { connection, .. } =
                std::mem::replace(&mut entry.state, InternalState::Failed("shut down".into()))
            {
                connections.push(connection);
            }
        }
        futures_util::future::join_all(connections.iter().map(|c| c.shutdown())).await;
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

    /// Pull the tools a connection offers into `routes`, `offered`, and
    /// `server_tools`. Returns the server's own tool names, in the order
    /// the connection reported them.
    fn add_connection_tools(&mut self, server: &str, connection: &Connection) -> Vec<String> {
        let mut names = Vec::with_capacity(connection.tools.len());
        for tool in &connection.tools {
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

    /// Remove every tool of `server` from `offered` and `server_tools`. The
    /// `routes` table is left alone — a call to a tool whose server has
    /// been disabled or disconnected should still reach the `call()` method
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

    /// Take a `Ready` connection out of its server entry, leaving the server
    /// in `Failed("closing")` until the caller decides the next state. Pulls
    /// the server's tools out of the offered set so a `Ready -> Ready` cycle
    /// (e.g. through `reconnect`) does not temporarily advertise two copies
    /// of the same tool.
    fn take_ready_connection(&mut self, name: &str) -> Result<Connection, String> {
        let entry = self
            .servers
            .get_mut(name)
            .ok_or_else(|| format!("no MCP server named {name:?} is configured"))?;
        match std::mem::replace(&mut entry.state, InternalState::Failed("closing".into())) {
            InternalState::Ready { connection, .. } => Ok(connection),
            other => {
                let reason = match &other {
                    InternalState::Failed(why) => format!("failed to start: {why}"),
                    InternalState::Disabled => "is disabled".into(),
                    InternalState::Disconnected => "is disconnected".into(),
                    InternalState::Ready { .. } => unreachable!("we just matched it"),
                };
                entry.state = other;
                Err(format!("mcp server {name} {reason}"))
            }
        }
    }

    /// Adopt `connection` as the server's new link: clear any routes the
    /// server used to advertise (so a re-enabled server's tools do not
    /// collide with the stale routes left behind from a `disable` or
    /// `disconnect`), build fresh tool entries, then write the server's
    /// entry in `Ready { connection, tools }`.
    fn adopt_connection(&mut self, server: &str, connection: Connection) {
        // Find every route that belongs to this server and drop it. Done
        // against `routes` rather than `server_tools` because the latter is
        // empty after a disable or disconnect, while the former is what
        // `add_connection_tools` checks for collisions.
        let stale: Vec<String> = self
            .routes
            .iter()
            .filter(|(_, route)| route.server == server)
            .map(|(name, _)| name.clone())
            .collect();
        for name in stale {
            self.routes.remove(&name);
        }
        let names = self.add_connection_tools(server, &connection);
        for note in &connection.notes {
            self.warnings.push(format!("{server}: {note}"));
        }
        let entry = self
            .servers
            .get_mut(server)
            .expect("server was present when the connection opened");
        entry.state = InternalState::Ready {
            connection,
            tools: names,
        };
    }
}

/// What the entry says it is, for the report: the command line it runs, or
/// the url it calls. Quoted where the parts carry spaces, so that a reader
/// can see where one ends and the next begins.
fn how_to_reach(entry: &Entry) -> String {
    let Ok(config) = &entry.config else {
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
    let mut cut = MAX_NAME - hash.len() - 1;
    // A name is ASCII by construction — [`sanitize`] made it so — and this is
    // belt and braces for a caller that one day does not.
    while cut > 0 && !name.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}~{}", &name[..cut], hash)
}

/// Keep what a name may carry and replace the rest. Two different names may
/// come out the same (a space and an underscore are one character to a
/// backend) — that is a collision the hub reports rather than a fix to guess
/// at.
fn sanitize(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// A short, stable hash of a name.
///
/// FNV-1a, written out rather than taken from a crate: the name it is part of
/// is written into the session log, and a later process reading that log has
/// to work out the same name from the same server — so the hash has to be the
/// same on every machine, in every release, forever. A hash that could
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
