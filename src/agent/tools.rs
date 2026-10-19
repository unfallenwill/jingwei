//! The fleet's four tools: `spawn_agent`, `wait_agents`, `stop_agent`,
//! `list_agents`.
//!
//! The schemas live in their own table, not in `src/tools.rs`: the
//! built-in registry is pinned at four tools and the pin stays honest.
//! These are a **third provider** in the router, after the built-ins and
//! the MCP provider, so the tool list reads: the four built-ins, the
//! servers' tools, then the fleet's.
//!
//! The schemas are **byte-stable for the life of the process**. The tool
//! list sits inside the cached prefix, so a schema that changed every
//! turn would miss the cache every turn. The agent schemas never change
//! — not when a worker exists, not when the cap is reached, not when the
//! fleet is disabled.
//!
//! **The descriptions are the prompt.** Half of "the model can forget"
//! is solved here: `spawn_agent` says the result comes back through
//! `wait_agents`, and `wait_agents` says a report is delivered once.
//! They are written for a model that has not seen the fleet before.
//!
//! ## Depth one
//!
//! The caller's id is checked *before* the dispatch: a call from anyone
//! but the main agent is refused, so a worker that hallucinates
//! `spawn_agent` gets a clean error instead of a fleet.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::cancel::CancelToken;
use crate::core::ids::AgentId;
use crate::core::ports::{BoxFuture, ToolProvider};
use crate::core::reminder::{ReminderId, SystemReminder};
use crate::workflow::exec::Fleet;

/// The four tool names, in the order they are offered.
pub(crate) const AGENT_TOOL_NAMES: [&str; 4] =
    ["spawn_agent", "wait_agents", "stop_agent", "list_agents"];

/// The reminder's text. The same bytes every request of a session, so
/// it rides in the cached prefix like every other session-stable
/// reminder.
const AGENT_TOOLS_REMINDER: &str = "\n\n# Background agents\n\n\
    You can start background agents to work on parts of a task at the same \
    time. Use `spawn_agent` with a self-contained task, give each agent a \
    disjoint set of files, and collect the results with `wait_agents` before \
    you answer. A background agent's report is its final answer — read it \
    before you summarize.";

/// The agent tools, as a provider over the fleet.
pub(crate) struct AgentTools {
    fleet: Arc<Fleet>,
    schemas: Vec<Value>,
}

impl AgentTools {
    /// The provider, over this session's fleet.
    pub(crate) fn new(fleet: Arc<Fleet>) -> Self {
        Self { fleet, schemas: schemas() }
    }

    /// The fleet's instruction, as a reminder.
    ///
    /// The tool list is the *mechanism*; this is the *reason to use it*.
    /// A model that has never been told what a background agent is will
    /// not reach for one on its own. It rides in the context's reminders
    /// and never in `SYSTEM`: the system text is the stable identity the
    /// model reads once per session, and this is only there when the
    /// fleet is.
    pub(crate) fn reminder() -> SystemReminder {
        SystemReminder {
            id: ReminderId::AgentTools,
            text: AGENT_TOOLS_REMINDER.into(),
        }
    }
}

impl ToolProvider for AgentTools {
    fn owns(&self, name: &str) -> bool {
        AGENT_TOOL_NAMES.contains(&name)
    }

    /// The four schemas, in that order, always. The bytes are a cache
    /// contract, so the list is built once and handed out as it is.
    fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>> {
        Box::pin(async move { self.schemas.clone() })
    }

    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a Value,
        token: &'a CancelToken,
        agent: AgentId,
    ) -> BoxFuture<'a, String> {
        Box::pin(async move {
            // Depth one: the fleet is the main agent's, not a worker's.
            if agent != AgentId::MAIN {
                return "only the main agent can manage background agents".into();
            }
            match name {
                "spawn_agent" => self.spawn(input),
                "wait_agents" => self.wait(input, token).await,
                "stop_agent" => self.stop(input),
                "list_agents" => self.fleet.list(),
                other => format!("error: unknown agent tool '{other}'"),
            }
        })
    }
}

impl AgentTools {
    /// `spawn_agent`: start a worker on one self-contained task.
    fn spawn(&self, input: &Value) -> String {
        let Some(task) = input.get("task").and_then(|v| v.as_str()) else {
            return "error: spawn_agent needs a task".into();
        };
        if task.trim().is_empty() {
            return "error: spawn_agent needs a task".into();
        }
        let label = input.get("label").and_then(|v| v.as_str());
        match self.fleet.spawn(task, label) {
            Ok(id) => {
                // the same rule the fleet named the worker with: the
                // model hears the name the human will read
                let shown = crate::workflow::exec::label_for(task, label);
                format!("{id} started: {shown}")
            }
            Err(e) => format!("error: {e}"),
        }
    }

    /// `wait_agents`: wait for workers, and take their reports.
    async fn wait(&self, input: &Value, token: &CancelToken) -> String {
        let mut ids = Vec::new();
        match input.get("ids") {
            None | Some(Value::Null) => {}
            Some(Value::Array(items)) => {
                for item in items {
                    let Some(raw) = item.as_str() else {
                        return "error: ids are agent ids like \"a1\"".into();
                    };
                    match AgentId::parse(raw) {
                        Some(id) => ids.push(id),
                        None => return format!("error: '{raw}' is not an agent id"),
                    }
                }
            }
            Some(_) => return "error: ids is a list of agent ids like [\"a1\"]".into(),
        }
        self.fleet.wait(&ids, token).await
    }

    /// `stop_agent`: stop one worker.
    fn stop(&self, input: &Value) -> String {
        let Some(raw) = input.get("id").and_then(|v| v.as_str()) else {
            return "error: stop_agent needs an id".into();
        };
        match AgentId::parse(raw) {
            Some(id) => self.fleet.stop(id),
            None => format!("error: '{raw}' is not an agent id"),
        }
    }
}

/// The four schemas. Built once, byte for byte, and never rewritten.
fn schemas() -> Vec<Value> {
    vec![
        json!({
            "name": "spawn_agent",
            "description": "Start a background agent to work on one self-contained task, and return its id. The agent has its own conversation, its own context window and its own cancel token; it shares this session's workspace, shell and file ledger. Use it when a task splits into parts that can be done at the same time, and give each agent a disjoint set of files. Collect the agent's result with wait_agents when you are ready for it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task, verbatim. Self-contained: the agent sees only this and the workspace."},
                    "label": {"type": "string", "description": "A short name for the agent, shown in the status bar. Default: the first words of the task."}
                },
                "required": ["task"]
            }
        }),
        json!({
            "name": "wait_agents",
            "description": "Wait for background agents to finish, and take their reports. Returns the report of every agent that finished; an agent that is still running comes back as still running. Without ids, waits for every live agent. A report is delivered once: an agent whose report you already took comes back as already reported.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "description": "Agent ids to wait for, e.g. [\"a1\"]. Default: every live agent."}
                }
            }
        }),
        json!({
            "name": "stop_agent",
            "description": "Stop a background agent. The command it is running is killed; whatever it already produced is kept and reported back to you.",
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The agent id, e.g. \"a1\"."}},
                "required": ["id"]
            }
        }),
        json!({
            "name": "list_agents",
            "description": "List this session's background agents: id, label, state, how long it has run, and how many tools it has called. A finished agent's report is here too, if you have not taken it yet.",
            "input_schema": {"type": "object", "properties": {}}
        }),
    ]
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ports::ToolProvider;
    use crate::test_util::block_on;
    use crate::workflow::exec::AgentCfg;

    /// A provider that stands in for the MCP one: it owns one tool, and
    /// it sits between the built-ins and the fleet's schemas.
    struct Mcp {
        names: Vec<String>,
    }

    impl ToolProvider for Mcp {
        fn owns(&self, name: &str) -> bool {
            self.names.iter().any(|n| n == name)
        }
        fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>> {
            Box::pin(async move { self.names.iter().map(|n| json!({"name": n})).collect() })
        }
        fn call<'a>(
            &'a self,
            _name: &'a str,
            _input: &'a Value,
            _token: &'a CancelToken,
            _agent: AgentId,
        ) -> BoxFuture<'a, String> {
            Box::pin(async { String::new() })
        }
    }

    #[test]
    fn the_schemas_are_stable() {
        // The tool list is inside the cached prefix: a schema that moved
        // would miss the cache every turn.
        let a = schemas();
        let b = schemas();
        assert_eq!(a, b, "the schemas are the same bytes every time");
        let names: Vec<&str> = a.iter().filter_map(|s| s["name"].as_str()).collect();
        assert_eq!(names, vec!["spawn_agent", "wait_agents", "stop_agent", "list_agents"]);
    }

    #[test]
    fn the_agent_tools_are_not_built_ins() {
        // The built-in registry is pinned at four, and it stays four.
        let builtins = crate::tools::builtin_schemas();
        assert_eq!(builtins.len(), 4, "the built-ins are still four");
        let names: Vec<&str> = builtins.iter().filter_map(|s| s["name"].as_str()).collect();
        for n in AGENT_TOOL_NAMES {
            assert!(!names.contains(&n), "the agent tools live in their own table: {n}");
        }
    }

    #[test]
    fn the_tool_list_puts_the_agent_schemas_last() {
        // The order is a cache contract: the built-ins, the servers'
        // tools, then the fleet's. The fleet's schemas are appended, and
        // the pin on the built-ins stays honest.
        let fleet = Arc::new(Fleet::new(AgentCfg::default()));
        let router = crate::core::ports::ToolRouter::new(vec![
            Box::new(crate::tools::Builtins),
            Box::new(Mcp { names: vec!["server_tool".into()] }),
            Box::new(AgentTools::new(fleet)),
        ]);
        let defs = block_on(router.definitions());
        let names: Vec<&str> = defs.iter().filter_map(|s| s["name"].as_str()).collect();
        assert_eq!(names.len(), 9, "four built-ins, one server tool, four agent tools: {names:?}");
        assert_eq!(&names[..4], &["bash", "read_file", "write_file", "edit_file"]);
        assert_eq!(names[4], "server_tool", "the servers' tools sit in the middle: {names:?}");
        assert_eq!(&names[5..], &AGENT_TOOL_NAMES, "the fleet's schemas come last: {names:?}");
    }

    #[test]
    fn the_fleet_reminder_says_how_to_use_the_fleet() {
        // The tool list is the mechanism; the reminder is the reason to
        // use it. Without it the model has no reason to reach for a
        // background agent at all.
        let r = AgentTools::reminder();
        assert!(matches!(r.id, ReminderId::AgentTools), "got: {:?}", r.id);
        assert!(r.text.contains("spawn_agent"), "it names the tool: {}", r.text);
        assert!(r.text.contains("wait_agents"), "and the one that collects: {}", r.text);
        assert!(
            r.text.contains("collect the results with `wait_agents` before you answer"),
            "and says to collect before answering: {}",
            r.text
        );
        // the same bytes every request of a session, so it rides in the
        // cached prefix
        assert!(r.id.is_session_stable(), "the reminder never changes mid-session");
    }

    #[test]
    fn the_cap_is_not_a_schema_change() {
        // The cap is a runtime limit: the tool list does not move when
        // the fleet is full, and it does not move when the fleet is off.
        let on = block_on(AgentTools::new(Arc::new(Fleet::new(AgentCfg::default()))).definitions());
        let off = block_on(
            AgentTools::new(Arc::new(Fleet::new(AgentCfg { max_agents: 0, wait_ms: 0 }))).definitions(),
        );
        assert_eq!(on, off, "the schemas are the same bytes either way");
    }
}
