//! The worker: one background agent, and the coroutine that runs it.
//!
//! A worker is the same agent the main is — the same turn loop, the same
//! tools, the same ledger rules — with four differences:
//!
//! - its own **context**, seeded from the main's: the same system text
//!   and the same project conventions, but an empty transcript, and
//!   without the fleet's own tools (the fleet is one level deep, and a
//!   worker told it could spawn agents would try);
//! - its own **history**, which starts with the task and nothing else —
//!   the task must be self-contained, because the worker sees the task
//!   and the workspace and nothing more;
//! - its own **cancel scope**, a child of the run's, so cancelling the
//!   run cancels the worker and stopping the worker leaves the run
//!   alone;
//! - its own **sink**, wrapped so every event it produces arrives
//!   attributed to it — a worker cannot end the human's turn or spend
//!   the session's tokens.
//!
//! The worker's transcript never enters the main's history and is never
//! persisted. What it leaves behind is its report: the last thing it
//! said.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::agent::roster::Roster;
use crate::agent::runner::run_agent;
use crate::agent::tools::AGENT_TOOL_NAMES;
use crate::agent::{Agent, AgentRole};
use crate::cancel::CancelScope;
use crate::config::Config;
use crate::core::context::ContextSeed;
use crate::core::ids::AgentId;
use crate::core::ir::{Block, Message};
use crate::core::ports::ToolRouter;
use crate::core::turn::{Turn, TurnState};
use crate::ui::{AgentSink, Msg, Show};
use crate::workflow::exec::WorkerDone;

/// What a worker needs to run. The fleet builds it and hands it to the
/// coroutine; nothing else holds it.
pub(crate) struct WorkerJob {
    /// Who the worker is.
    pub(crate) id: AgentId,
    /// The name the human sees.
    pub(crate) label: String,
    /// The task, verbatim.
    pub(crate) task: String,
    /// The worker's own scope: a child of the run's.
    pub(crate) scope: CancelScope,
    /// The session's config — the worker uses the session's model.
    pub(crate) cfg: Config,
    /// The session's tool router. The dispatch is shared; the worker's
    /// *tool list* is not, and the depth-one check is what keeps the
    /// fleet's own tools out of the worker's reach.
    pub(crate) router: Arc<ToolRouter>,
    /// The session's sink. The worker wraps it, so its events arrive
    /// attributed to it.
    pub(crate) sink: Arc<dyn Show>,
    /// A seed of the main's context: the shared state, without the
    /// transcript.
    pub(crate) seed: ContextSeed,
    /// The registry of live agents, so the worker registers itself for
    /// the length of its run.
    pub(crate) roster: Roster,
    /// Where the finished worker leaves what it produced.
    pub(crate) done: Arc<Mutex<Option<WorkerDone>>>,
    /// The tool calls it made, counted as it made them.
    pub(crate) tools: Arc<AtomicUsize>,
}

/// Run one worker to its end, and leave what it produced behind.
///
/// The report is the last thing the worker said. If it said nothing —
/// it was stopped before it answered, or it failed — the failure or the
/// stopping is the report: a failed worker is information, not a crash.
pub(crate) async fn run_worker(job: WorkerJob) {
    let started = Instant::now();
    // The worker's own context: the main's prompt, without the main's
    // conversation, and without the fleet's tools.
    let mut ctx = job.seed.context(&AGENT_TOOL_NAMES);
    ctx.push(Message::User(job.task.clone()));
    let mut agent = Agent::new(
        job.id,
        job.label.clone(),
        AgentRole::Worker,
        job.task.clone(),
        Turn::new(0),
    );
    let sink = CountingSink {
        inner: AgentSink::new(job.sink.clone(), job.id),
        count: job.tools.clone(),
    };
    // The main agent's run, verbatim: one implementation, two callers.
    let _ = run_agent(
        &mut agent,
        &job.cfg,
        &mut ctx,
        &job.router,
        &job.scope,
        &job.roster,
        &sink,
    )
    .await;
    agent.report = report_of(&ctx, agent.state());
    let done = WorkerDone { agent, elapsed: started.elapsed() };
    if let Ok(mut slot) = job.done.lock() {
        *slot = Some(done);
    }
}

/// The worker's report: the last text it said, or a one-line account of
/// how it ended.
fn report_of(ctx: &crate::core::context::Context, state: TurnState) -> Option<String> {
    if let Some(text) = last_text(ctx) {
        if !text.trim().is_empty() {
            return Some(text);
        }
    }
    match state {
        // it finished and said nothing: that is the report.
        TurnState::Completed => Some("no answer".into()),
        // it failed or was stopped: the block that carries the report
        // already says which, so the text stays empty.
        _ => None,
    }
}

/// The last text the worker said, out of its own transcript.
fn last_text(ctx: &crate::core::context::Context) -> Option<String> {
    ctx.history().iter().rev().find_map(|m| match m {
        Message::Assistant(blocks) => blocks.iter().rev().find_map(|b| match b {
            Block::Text(t) => Some(t.clone()),
            _ => None,
        }),
        _ => None,
    })
}

/// The worker's sink: the session's sink, wrapped so every event is
/// attributed to the worker, and counting the tool calls it made.
struct CountingSink {
    inner: AgentSink,
    count: Arc<AtomicUsize>,
}

impl Show for CountingSink {
    fn show(&self, m: Msg) {
        if matches!(m, Msg::Tool { .. }) {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.show(m);
    }
}
