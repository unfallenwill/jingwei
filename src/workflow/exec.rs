//! The fleet: the session's background agents.
//!
//! One type owns the workers. It holds the record of them — who was
//! spawned, under which label, how long it ran, what it produced — and it
//! acts on that record: spawn one, stop one, wait for them, list them.
//!
//! The **roster** (`crate::agent::roster`) holds the *live* agents: an id
//! and the scope that stops it. The fleet holds the *run*: a worker's
//! scope is a child of the run's scope, so cancelling the run cancels
//! every worker under it, and stopping a worker leaves the run alone.
//!
//! ## The fleet does not survive a run
//!
//! A worker is a thing that happens inside a turn, not a thing that lives
//! in the session. [`Fleet::end_run`] stops every live worker and clears
//! the record; the next run starts from nothing. That is why the fleet
//! holds no session of its own: a worker's transcript is never persisted
//! and never enters the main's history.
//!
//! ## The report
//!
//! A worker's report is its final assistant text — the thing the model
//! would have said if it had done the work inline. A failure is a report
//! too: a failed worker is *information*, not a crash, and the main
//! agent's turn does not fail because a worker failed. The report is
//! capped, because a report that eats the main's context window is a
//! report the main cannot use.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::agent::cmd::Row;
use crate::agent::roster::Roster;
use crate::agent::worker::{run_worker, WorkerJob};
use crate::agent::Agent;
use crate::cancel::{CancelScope, CancelToken};
use crate::config::Config;
use crate::core::context::{Context, ContextSeed};
use crate::core::ids::AgentId;
use crate::core::ports::ToolRouter;
use crate::core::turn::TurnState;
use crate::ui::Show;

/// How long a report may be before it is cut down. The full text stays
/// in the fleet's record; this is what the main agent gets to read.
const REPORT_CAP: usize = 2_000;

/// How often a wait looks at the workers it is waiting for. Short
/// enough that a finished worker is noticed at once, long enough that a
/// long wait is not a spin.
const POLL: Duration = Duration::from_millis(25);

/// The fleet's two numbers: how many workers may run at once, and how
/// long a wait waits before it answers with what it has.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AgentCfg {
    /// The cap. Zero disables spawning — the tool schemas stay on the
    /// wire, so the cached prefix stays warm.
    pub(crate) max_agents: u32,
    /// How long `wait_agents` waits before it reports the workers that
    /// are still running.
    pub(crate) wait_ms: u64,
}

impl AgentCfg {
    /// The fleet's numbers, as the configuration resolved them.
    pub(crate) fn from_config(cfg: &Config) -> Self {
        Self { max_agents: cfg.max_agents, wait_ms: cfg.agent_wait_ms }
    }

    /// Is the fleet on? Zero agents disables *spawning* — the schemas
    /// stay on the wire, so the cache stays warm — but a model told
    /// about background agents it cannot start would only ask for them.
    pub(crate) fn enabled(&self) -> bool {
        self.max_agents > 0
    }
}

impl Default for AgentCfg {
    fn default() -> Self {
        Self {
            max_agents: crate::config::DEFAULT_MAX_AGENTS,
            wait_ms: crate::config::DEFAULT_AGENT_WAIT_MS,
        }
    }
}

/// The session's background agents.
///
/// `Fleet` is a handle: the frontends hold an `Arc<Fleet>`, the workers
/// it spawns hold what they need, and the state behind the mutex is the
/// one record of who is running.
pub(crate) struct Fleet {
    inner: Arc<Mutex<Inner>>,
    roster: Roster,
    cfg: AgentCfg,
}

/// The fleet's state, behind one lock.
struct Inner {
    /// The workers, in the order they were spawned.
    workers: BTreeMap<AgentId, Worker>,
    /// The run's parent scope, while a run is in flight. The workers'
    /// scopes are its children.
    run_scope: Option<CancelScope>,
    /// What a worker needs to run: the session's config, the router, the
    /// sink, and a seed of the main's context. `None` between runs.
    run: Option<RunBits>,
}

/// The session's own things, as a worker needs them.
#[derive(Clone)]
struct RunBits {
    cfg: Config,
    router: Arc<ToolRouter>,
    sink: Arc<dyn Show>,
    seed: ContextSeed,
}

/// One worker: what it was asked to do, and what it left behind.
struct Worker {
    /// The name the human gave it. The agent carries the same name, and
    /// the agent's is the one that is read once the worker is done — the
    /// value is the source of truth, the record is a projection of it.
    label: String,
    /// The worker's own scope: the handle a stop request reaches, and
    /// the bit that says the worker is on its way out.
    scope: CancelScope,
    started: Instant,
    /// What the worker left when it ended. `None` while it runs.
    done: Arc<Mutex<Option<WorkerDone>>>,
    /// The tool calls the worker made, counted as it made them.
    tools: Arc<AtomicUsize>,
    /// The worker was asked to stop.
    stopping: bool,
    /// The report was taken by a `wait_agents`.
    delivered: bool,
    /// The coroutine. Held so the task is not detached from the fleet.
    _handle: tokio::task::JoinHandle<()>,
}

/// What a finished worker leaves behind: the agent it became, and how
/// long it took.
#[derive(Clone)]
pub(crate) struct WorkerDone {
    /// The agent the worker was: its lifecycle, its report, its tools.
    pub(crate) agent: Agent,
    /// How long it ran, from its start to its end.
    pub(crate) elapsed: Duration,
}

impl Fleet {
    /// A fleet with no workers and no run in flight.
    pub(crate) fn new(cfg: AgentCfg) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                workers: BTreeMap::new(),
                run_scope: None,
                run: None,
            })),
            roster: Roster::new(),
            cfg,
        }
    }

    /// The registry of live agents: the fleet's workers live in it, and
    /// a stop request reaches them through it.
    pub(crate) fn roster(&self) -> &Roster {
        &self.roster
    }

    /// Open a run. The workers it spawns are children of `scope`, so
    /// cancelling the run cancels every one of them.
    pub(crate) fn begin_run(
        &self,
        cfg: &Config,
        ctx: &Context,
        router: &Arc<ToolRouter>,
        sink: &Arc<dyn Show>,
        scope: &CancelScope,
    ) {
        let mut inner = self.lock();
        inner.run_scope = Some(scope.clone());
        inner.run = Some(RunBits {
            cfg: cfg.clone(),
            router: router.clone(),
            sink: sink.clone(),
            seed: ctx.seed(),
        });
    }

    /// Close a run: every live worker is stopped and the fleet's record
    /// goes. The fleet does not survive a run — the next one starts from
    /// nothing.
    pub(crate) fn end_run(&self) {
        self.stop_all();
        let ids: Vec<AgentId> = {
            let inner = self.lock();
            inner.workers.keys().copied().collect()
        };
        for id in ids {
            self.roster.remove(id);
        }
        let mut inner = self.lock();
        inner.workers.clear();
        inner.run_scope = None;
        inner.run = None;
    }

    /// Start a background agent on one self-contained task.
    ///
    /// The cap is a *runtime* cap, never a schema change: the tool list
    /// is the same whether the fleet is full or empty.
    pub(crate) fn spawn(&self, task: &str, label: Option<&str>) -> Result<AgentId, String> {
        if self.cfg.max_agents == 0 {
            return Err("agent spawning is disabled".into());
        }
        let live = self.roster.list().len() as u32;
        if live >= self.cfg.max_agents {
            return Err(format!(
                "agent limit reached ({}) — wait for one to finish first",
                self.cfg.max_agents
            ));
        }
        let id = AgentId::next();
        let label = match label {
            Some(l) if !l.trim().is_empty() => l.trim().to_string(),
            _ => default_label(task),
        };
        // The run's bits and the run's scope: without a run there is
        // nothing for a worker to be a child of.
        let (scope, run) = {
            let inner = self.lock();
            let scope = inner.run_scope.clone().ok_or("no run in flight")?;
            let run = inner.run.clone().ok_or("no run in flight")?;
            (scope, run)
        };
        let scope = scope.child();
        // The worker is registered before it starts, so a stop request
        // that arrives in the first moments still finds it.
        self.roster.register(id, scope.clone());
        let done = Arc::new(Mutex::new(None));
        let tools = Arc::new(AtomicUsize::new(0));
        let job = WorkerJob {
            id,
            label: label.clone(),
            task: task.to_string(),
            scope: scope.clone(),
            cfg: run.cfg.clone(),
            router: run.router.clone(),
            sink: run.sink.clone(),
            seed: run.seed.clone(),
            roster: self.roster.clone(),
            done: done.clone(),
            tools: tools.clone(),
        };
        // The announcement goes out before the worker does: the human
        // sees the agent start, not start-and-finish.
        crate::ui::AgentSink::new(run.sink.clone(), id).started(label.clone());
        let handle = tokio::spawn(run_worker(job));
        let mut inner = self.lock();
        inner.workers.insert(
            id,
            Worker {
                label,
                scope: scope.clone(),
                started: Instant::now(),
                done,
                tools,
                stopping: false,
                delivered: false,
                _handle: handle,
            },
        );
        Ok(id)
    }

    /// Wait for workers, and take their reports.
    ///
    /// With no ids, every live worker is waited for; with ids, those.
    /// The wait ends when every requested worker has finished, when the
    /// wait timeout expires, or when the caller's token is cancelled —
    /// a Ctrl-C during a long wait must not hang the process.
    ///
    /// A report is delivered once: a worker whose report was already
    /// taken comes back as already reported.
    pub(crate) async fn wait(&self, ids: &[AgentId], token: &CancelToken) -> String {
        let deadline = Instant::now() + Duration::from_millis(self.cfg.wait_ms);
        loop {
            if self.pending(ids).is_empty() { break; }
            if token.is_cancelled() { return "the wait was cancelled".into(); }
            if Instant::now() >= deadline { break; }
            tokio::select! {
                _ = tokio::time::sleep(POLL) => {}
                _ = token.cancelled() => { return "the wait was cancelled".into(); }
            }
        }
        let blocks = self.blocks(ids);
        // The reports are handed over now: they are delivered once.
        self.deliver(ids);
        blocks.join("\n\n")
    }

    /// Stop one worker. What the worker already produced is kept, and
    /// comes back with the next wait.
    pub(crate) fn stop(&self, id: AgentId) -> String {
        let known = {
            let inner = self.lock();
            inner.workers.contains_key(&id)
        };
        if !known {
            return format!("{id} is not running");
        }
        if self.roster.stop(id) {
            let mut inner = self.lock();
            if let Some(w) = inner.workers.get_mut(&id) {
                w.stopping = true;
            }
            return format!("{id} stopped");
        }
        // It is not live: it finished, one way or another.
        let state = {
            let inner = self.lock();
            inner.workers.get(&id).and_then(|w| w.done.lock().unwrap().clone())
        };
        let word = state.map(|d| state_word(d.agent.state())).unwrap_or("done");
        format!("{id} is not running ({word})")
    }

    /// Stop every live worker, and say how many there were.
    pub(crate) fn stop_all(&self) -> usize {
        let ids = self.roster.list();
        if ids.is_empty() { return 0; }
        // The roster does the stopping: it holds the scopes, and it
        // leaves the main agent alone — the conversation is not the
        // fleet's to stop.
        self.roster.stop_all();
        let mut inner = self.lock();
        for id in &ids {
            if let Some(w) = inner.workers.get_mut(id) {
                w.stopping = true;
            }
        }
        ids.len()
    }

    /// The table of the workers — the same text `/agents` prints and the
    /// same text `list_agents` returns.
    pub(crate) fn list(&self) -> String {
        crate::agent::cmd::render_table(&self.rows())
    }

    /// The ids of the live workers, in the order they were spawned.
    ///
    /// The TUI completes them after `/agents stop ` — a stop request
    /// needs an id, and the ids the fleet knows are the live ones.
    pub(crate) fn live_ids(&self) -> Vec<AgentId> {
        self.roster.list()
    }

    /// The workers, one row each.
    pub(crate) fn rows(&self) -> Vec<Row> {
        let inner = self.lock();
        inner
            .workers
            .iter()
            .map(|(id, w)| {
                let done = w.done.lock().unwrap().clone();
                match done {
                    Some(d) => Row {
                        id: *id,
                        label: d.agent.name.clone(),
                        state: state_word(d.agent.state()),
                        elapsed: d.elapsed,
                        tools: d.agent.tools_used.len(),
                        report: if w.delivered { None } else { d.agent.report.clone() },
                        error: d.agent.lifecycle.error.clone(),
                        delivered: w.delivered,
                    },
                    None => Row {
                        id: *id,
                        label: w.label.clone(),
                        // a worker whose scope is cancelled is on its way
                        // out, whether the cancel came from a stop
                        // request or from the run's own cancel
                        state: if w.stopping || w.scope.is_cancelled() { "stopping" } else { "running" },
                        elapsed: w.started.elapsed(),
                        tools: w.tools.load(Ordering::Relaxed),
                        report: None,
                        error: None,
                        delivered: false,
                    },
                }
            })
            .collect()
    }

    /// Is a worker still running? The TUI's Ctrl-C ladder asks: the
    /// middle step only exists when there is something to stop.
    ///
    /// The roster is the answer: it holds a worker from the moment it
    /// is spawned until the moment it ends, and [`Fleet::stop_all`]
    /// reaches exactly the workers it holds. A worker on its way out is
    /// still in it — that is the point: a stop request that finds one
    /// is a stop request that has something to do.
    pub(crate) fn has_live_workers(&self) -> bool {
        !self.roster.list().is_empty()
    }

    /// The workers the wait is still waiting for: the requested ones, or
    /// every worker when none were named.
    fn pending(&self, ids: &[AgentId]) -> Vec<AgentId> {
        let inner = self.lock();
        let wanted: Vec<AgentId> = if ids.is_empty() {
            inner.workers.keys().copied().collect()
        } else {
            ids.to_vec()
        };
        wanted
            .into_iter()
            .filter(|id| {
                inner
                    .workers
                    .get(id)
                    .map(|w| w.done.lock().unwrap().is_none())
                    .unwrap_or(false)
            })
            .collect()
    }

    /// One block per worker: what it did, and what it produced.
    fn blocks(&self, ids: &[AgentId]) -> Vec<String> {
        let inner = self.lock();
        let wanted: Vec<AgentId> = if ids.is_empty() {
            inner.workers.keys().copied().collect()
        } else {
            ids.to_vec()
        };
        let mut out = Vec::new();
        for id in wanted {
            let Some(w) = inner.workers.get(&id) else { continue };
            let done = w.done.lock().unwrap().clone();
            match done {
                Some(d) => {
                    if w.delivered {
                        out.push(format!("{id} ({}) already reported", d.agent.name));
                    } else {
                        out.push(block_for(id, &d));
                    }
                }
                None => out.push(format!(
                    "{id} ({}) still running ({})",
                    w.label,
                    crate::ui::elapsed_str(w.started.elapsed())
                )),
            }
        }
        out
    }

    /// Mark the workers' reports as taken. A worker that is still
    /// running has no report to take: its block says so, and it keeps
    /// its report for the wait that comes after it.
    fn deliver(&self, ids: &[AgentId]) {
        let mut inner = self.lock();
        let wanted: Vec<AgentId> = if ids.is_empty() {
            inner.workers.keys().copied().collect()
        } else {
            ids.to_vec()
        };
        for id in wanted {
            if let Some(w) = inner.workers.get_mut(&id) {
                if w.done.lock().unwrap().is_some() {
                    w.delivered = true;
                }
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Clone for Fleet {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            roster: self.roster.clone(),
            cfg: self.cfg,
        }
    }
}

impl std::fmt::Debug for Fleet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fleet").field("workers", &self.lock().workers.len()).finish()
    }
}

/// The word a finished worker's state is shown as.
fn state_word(state: TurnState) -> &'static str {
    match state {
        TurnState::Completed => "done",
        TurnState::Failed => "failed",
        TurnState::Interrupted | TurnState::Canceled => "stopped",
        _ => "running",
    }
}

/// One worker's block: what it did, and what it produced.
fn block_for(id: AgentId, done: &WorkerDone) -> String {
    let elapsed = crate::ui::elapsed_str(done.elapsed);
    let agent = &done.agent;
    let label = &agent.name;
    let report = agent.report.clone().unwrap_or_default();
    match agent.state() {
        TurnState::Completed => format!(
            "{id} ({label}) done in {elapsed}, {} tool calls:\n{}",
            agent.tools_used.len(),
            cap_report(&report)
        ),
        TurnState::Failed => {
            let err = agent.lifecycle.error.clone().unwrap_or_default();
            format!("{id} ({label}) failed after {elapsed}: {err}")
        }
        TurnState::Interrupted | TurnState::Canceled => {
            if report.trim().is_empty() {
                format!("{id} ({label}) stopped after {elapsed}")
            } else {
                format!("{id} ({label}) stopped after {elapsed}:\n{}", cap_report(&report))
            }
        }
        _ => format!("{id} ({label}) still running ({elapsed})"),
    }
}

/// Cut a report down to what the main agent can afford to read. The
/// full text stays in the fleet's record.
fn cap_report(s: &str) -> String {
    let n = s.chars().count();
    if n <= REPORT_CAP {
        return s.to_string();
    }
    let cut: String = s.chars().take(REPORT_CAP).collect();
    format!("{cut}… (truncated, {} more chars)", n - REPORT_CAP)
}

/// The label a worker gets when the caller did not name one: the first
/// words of its task, which is usually what it is working on.
pub(crate) fn default_label(task: &str) -> String {
    let words: Vec<&str> = task.split_whitespace().collect();
    let mut label = words.iter().take(4).copied().collect::<Vec<_>>().join(" ");
    if label.chars().count() > 24 {
        label = label.chars().take(24).collect::<String>() + "…";
    }
    if label.is_empty() {
        "agent".into()
    } else {
        label
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::AgentTools;
    use crate::core::ports::ToolProvider;
    use crate::test_util::{block_on, cfg, mock_seq, request_body, Recorder};
    use crate::ui::Msg;
    use serde_json::json;
    use std::io::{Read, Write};

    /// A model that answers once and stops.
    const ANSWER: &str =
        r#"{"content":[{"type":"text","text":"the report"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
    /// A model that refuses.
    const REFUSED: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"no"}}"#;

    /// The world a fleet lives in: the fleet itself, the run's scope, the
    /// sink every event goes through, the main's own context, and the
    /// router the workers call.
    struct Harness {
        fleet: Arc<Fleet>,
        scope: CancelScope,
        rec: Recorder,
        ctx: Context,
    }

    impl Harness {
        /// A fleet over a mock model on `port`, with a run in flight.
        fn new(port: u16, agent_cfg: AgentCfg) -> Self {
            let fleet = Arc::new(Fleet::new(agent_cfg));
            // The router the workers call: the built-ins, then the
            // fleet's own four tools. The worker must not see the second
            // group, and the only way to know that is to offer it.
            let router = Arc::new(ToolRouter::new(vec![
                Box::new(crate::tools::Builtins),
                Box::new(AgentTools::new(fleet.clone())),
            ]));
            let rec = Recorder::default();
            let sink: Arc<dyn Show> = Arc::new(rec.clone());
            let scope = CancelScope::new();
            let ctx = crate::test_util::ctx();
            fleet.begin_run(&cfg(format!("http://127.0.0.1:{port}"), false), &ctx, &router, &sink, &scope);
            Self { fleet, scope, rec, ctx }
        }
    }

    /// A mock that answers, but late: the worker that asked is still
    /// running when the test looks at it.
    ///
    /// It answers rather than staying silent on purpose: a request that
    /// is never answered holds the transport's read timeout, and the
    /// runtime waits for it at shutdown — a test that hung on a silent
    /// mock would take three minutes to finish, not three seconds.
    fn mock_late(delay: Duration, body: &str) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = body.to_owned();
        std::thread::spawn(move || {
            while let Ok((mut s, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = s.read(&mut buf);
                std::thread::sleep(delay);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = s.write_all(resp.as_bytes());
                let _ = s.flush();
            }
        });
        port
    }

    /// How long the late mock takes to answer. Long enough that a
    /// worker is still running when a test looks at it, short enough
    /// that the suite does not wait for it.
    const LATE: Duration = Duration::from_millis(300);

    // ---- the cap ---------------------------------------------------------

    #[test]
    fn the_fleet_is_off_when_the_cap_is_zero() {
        // Zero agents disables spawning. The schemas stay on the wire —
        // the cache stays warm — but the model is not told about
        // background agents it could not start.
        assert!(AgentCfg { max_agents: 1, wait_ms: 0 }.enabled());
        assert!(!AgentCfg { max_agents: 0, wait_ms: 0 }.enabled());
    }

    #[test]
    fn the_cap_is_enforced() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 2, wait_ms: 1_000 });
            h.fleet.spawn("first", None).unwrap();
            h.fleet.spawn("second", None).unwrap();
            let err = h.fleet.spawn("third", None).unwrap_err();
            assert_eq!(err, "agent limit reached (2) — wait for one to finish first");
        });
    }

    #[test]
    fn spawning_is_disabled_at_zero() {
        let fleet = Fleet::new(AgentCfg { max_agents: 0, wait_ms: 0 });
        let err = fleet.spawn("count the files", None).unwrap_err();
        assert_eq!(err, "agent spawning is disabled");
    }

    // ---- the worker's own world -----------------------------------------

    #[test]
    fn a_worker_gets_its_own_context() {
        let (port, reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let h = block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            h.fleet.spawn("count the files", None).unwrap();
            h.fleet.wait(&[], &CancelToken::new()).await;
            h
        });
        // what the worker asked the model: the task, and nothing else
        let raw = reqs.lock().unwrap()[0].clone();
        let body = request_body(&raw);
        let msgs = body["messages"].as_array().expect("the request carries messages");
        assert_eq!(msgs.len(), 1, "the worker starts with its task and nothing else: {msgs:?}");
        assert_eq!(msgs[0]["content"], json!("count the files"), "got: {msgs:?}");
        // and the main's own transcript never saw any of it
        assert!(h.ctx.history().is_empty(), "the main's history is untouched: {:?}", h.ctx.history());
    }

    #[test]
    fn a_worker_does_not_see_the_agent_tools() {
        let (port, reqs) = mock_seq(vec![(200, ANSWER.into())]);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            h.fleet.spawn("count the files", None).unwrap();
            h.fleet.wait(&[], &CancelToken::new()).await;
        });
        let raw = reqs.lock().unwrap()[0].clone();
        let body = request_body(&raw);
        let tools = body["tools"].as_array().expect("the request carries tools");
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert_eq!(names.len(), 4, "the worker sees the four built-ins: {names:?}");
        for n in crate::agent::tools::AGENT_TOOL_NAMES {
            assert!(!names.contains(&n), "a worker does not see {n}: {names:?}");
        }
    }

    #[test]
    fn a_worker_does_not_end_the_conversation() {
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let h = block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            h.fleet.spawn("count the files", None).unwrap();
            h.fleet.wait(&[], &CancelToken::new()).await;
            h
        });
        let msgs = h.rec.msgs();
        assert!(!msgs.is_empty(), "the worker said something");
        for m in &msgs {
            assert!(matches!(m, Msg::FromAgent { .. }), "every event of the worker's arrives wrapped: {msgs:?}");
        }
        // a worker cannot end the human's run, and cannot spend the
        // session's tokens
        assert!(!msgs.iter().any(|m| matches!(m, Msg::TaskEnd | Msg::TaskBegin(_))), "got: {msgs:?}");
        assert!(!msgs.iter().any(|m| matches!(m, Msg::Usage(_) | Msg::OutTokens(_))), "got: {msgs:?}");
    }

    #[test]
    fn the_four_tools_drive_the_fleet() {
        // The acceptance in one test: the main agent starts a worker,
        // waits for it, reads its report, lists the workers and stops
        // one — through the tools, the way the model reaches them.
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            let tools = AgentTools::new(h.fleet.clone());
            let token = CancelToken::new();
            let started = tools
                .call("spawn_agent", &json!({"task": "count the files"}), &token, AgentId::MAIN)
                .await;
            assert!(started.ends_with("started: count the files"), "got: {started}");
            let id = AgentId::parse(started.split_whitespace().next().expect("an id"))
                .expect("the id the tool handed back");
            let waited = tools.call("wait_agents", &json!({}), &token, AgentId::MAIN).await;
            assert!(waited.contains("the report"), "the report came back: {waited}");
            let listed = tools.call("list_agents", &json!({}), &token, AgentId::MAIN).await;
            assert!(listed.contains(&id.to_string()), "the worker is on the list: {listed}");
            let stopped = tools
                .call("stop_agent", &json!({"id": id.to_string()}), &token, AgentId::MAIN)
                .await;
            assert!(stopped.contains("is not running"), "it had already finished: {stopped}");
        });
    }

    #[test]
    fn a_worker_may_not_spawn_a_worker() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 1_000 });
            let tools = AgentTools::new(h.fleet.clone());
            let out = tools
                .call("spawn_agent", &json!({"task": "count the files"}), &CancelToken::new(), AgentId::next())
                .await;
            assert_eq!(out, "only the main agent can manage background agents");
            assert!(h.fleet.rows().is_empty(), "and no worker appeared: {:?}", h.fleet.rows());
        });
    }

    // ---- the wait --------------------------------------------------------

    #[test]
    fn a_finished_worker_is_reported_once() {
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            let id = h.fleet.spawn("count the files", None).unwrap();
            let first = h.fleet.wait(&[id], &CancelToken::new()).await;
            assert!(first.contains("done in"), "the worker finished: {first}");
            assert!(first.contains("the report"), "and its report came back: {first}");
            let second = h.fleet.wait(&[id], &CancelToken::new()).await;
            assert!(second.contains("already reported"), "a report is delivered once: {second}");
        });
    }

    #[test]
    fn the_listing_shows_the_report_without_taking_it() {
        // The report is delivered once, by `wait_agents`: a listing is a
        // sighting, not a delivery, so the report still comes back as
        // the tool result the model asked for.
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            let id = h.fleet.spawn("count the files", None).unwrap();
            // wait for the worker to end, without taking its report
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if h.fleet.rows().iter().any(|r| r.id == id && r.state == "done") {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            let listing = h.fleet.list();
            assert!(listing.contains("the report"), "the listing shows the report: {listing}");
            let rows = h.fleet.rows();
            let row = rows.iter().find(|r| r.id == id).expect("the worker is on the list");
            assert!(!row.delivered, "but it did not take it: {rows:?}");
            // and the report still comes back, through the wait
            let out = h.fleet.wait(&[id], &CancelToken::new()).await;
            assert!(out.contains("the report"), "the report still comes back: {out}");
        });
    }

    #[test]
    fn a_running_worker_comes_back_as_still_running() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 50 });
            let id = h.fleet.spawn("count the files", None).unwrap();
            let out = h.fleet.wait(&[id], &CancelToken::new()).await;
            assert!(out.contains("still running"), "the wait gave up at its timeout: {out}");
        });
    }

    #[test]
    fn the_wait_is_cancellable() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 60_000 });
            let id = h.fleet.spawn("count the files", None).unwrap();
            let token = CancelToken::new();
            let t = token.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                t.cancel();
            });
            let out = h.fleet.wait(&[id], &token).await;
            assert_eq!(out, "the wait was cancelled");
        });
    }

    // ---- stopping --------------------------------------------------------

    #[test]
    fn a_stopped_worker_keeps_what_it_produced() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            let id = h.fleet.spawn("count the files", None).unwrap();
            let said = h.fleet.stop(id);
            assert_eq!(said, format!("{id} stopped"));
            let out = h.fleet.wait(&[id], &CancelToken::new()).await;
            assert!(out.contains("stopped"), "the worker says it was stopped: {out}");
            // and the fleet still knows about it: the work is not lost
            assert_eq!(h.fleet.rows().len(), 1, "the worker is still on the list: {:?}", h.fleet.rows());
        });
    }

    #[test]
    fn stopping_one_worker_leaves_the_others_running() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 1_000 });
            let a = h.fleet.spawn("first", None).unwrap();
            let b = h.fleet.spawn("second", None).unwrap();
            h.fleet.stop(a);
            let rows = h.fleet.rows();
            let state = |id: AgentId| rows.iter().find(|r| r.id == id).map(|r| r.state).unwrap_or("?");
            assert_eq!(state(a), "stopping", "the stopped one is on its way out: {rows:?}");
            assert_eq!(state(b), "running", "the other one is untouched: {rows:?}");
        });
    }

    #[test]
    fn stop_all_does_not_cancel_the_main() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 1_000 });
            // the main agent registers itself for the length of its run
            let main_scope = CancelScope::new();
            h.fleet.roster().register(AgentId::MAIN, main_scope.clone());
            h.fleet.spawn("first", None).unwrap();
            h.fleet.stop_all();
            assert!(!main_scope.is_cancelled(), "the conversation is not the fleet's to stop");
            assert!(!h.scope.is_cancelled(), "and neither is the run the workers hung under");
            // the worker is on its way out
            let rows = h.fleet.rows();
            let row = rows.first().expect("the worker is on the list");
            assert_eq!(row.state, "stopping", "the worker is stopping: {rows:?}");
        });
    }

    #[test]
    fn the_fleet_does_not_survive_a_run() {
        let port = mock_late(LATE, ANSWER);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 1_000 });
            h.fleet.spawn("first", None).unwrap();
            assert_eq!(h.fleet.rows().len(), 1);
            h.fleet.end_run();
            assert!(h.fleet.rows().is_empty(), "the records are gone: {:?}", h.fleet.rows());
            assert!(!h.fleet.has_live_workers(), "and no worker is live");
        });
    }

    // ---- the report ------------------------------------------------------

    #[test]
    fn the_report_is_capped() {
        let long = "x".repeat(REPORT_CAP + 50);
        let out = cap_report(&long);
        assert!(out.contains("… (truncated, 50 more chars)"), "got: {}", &out[out.len().saturating_sub(40)..]);
        assert_eq!(out.chars().count(), REPORT_CAP + "… (truncated, 50 more chars)".chars().count());
        // a short report is left alone
        assert_eq!(cap_report("short"), "short");
    }

    #[test]
    fn a_failed_worker_is_information_not_a_crash() {
        let (port, _reqs) = mock_seq(vec![(400, REFUSED.into())]);
        block_on(async {
            let h = Harness::new(port, AgentCfg { max_agents: 4, wait_ms: 2_000 });
            let id = h.fleet.spawn("count the files", None).unwrap();
            // the wait answers with text — a failure is a report, not an
            // error the main's turn would fail on
            let out = h.fleet.wait(&[id], &CancelToken::new()).await;
            assert!(out.contains("failed"), "the failure is said: {out}");
            assert!(out.contains("api 400"), "and says what went wrong: {out}");
        });
    }

    // ---- the label -------------------------------------------------------

    #[test]
    fn the_label_defaults_to_the_first_words_of_the_task() {
        assert_eq!(default_label("count the files in the src directory"), "count the files in");
        assert_eq!(default_label("  "), "agent");
        let long = "a very long task description that goes on and on and on";
        assert!(default_label(long).chars().count() <= 25, "the label is short: {}", default_label(long));
    }
}
