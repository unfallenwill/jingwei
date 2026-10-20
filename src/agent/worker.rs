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
        seen: Mutex::new(String::new()),
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
    // The worker announces its end. Without this the frontends never hear
    // that it is over: its prose stays buffered in the TUI, its summary
    // and its report never land, and its chip never leaves the status
    // bar. The announcement goes straight to the session's sink — the
    // `AgentSink`'s drop list would swallow a wrapped event.
    //
    // The report is announced only when the frontends have not seen it.
    // The report *is* the last thing the worker said, and the worker
    // streams what it says, so by the time it ends the human has read
    // the answer already; saying it again would print it twice. A
    // worker that was stopped before it spoke, or one that answered
    // outside the stream, still has something to announce.
    let report = agent.report.clone().unwrap_or_default();
    let announce = if sink.has_seen(&report) { String::new() } else { report };
    sink.inner.finished(agent.state(), announce);
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
    /// The prose the human has been shown so far.
    seen: Mutex<String>,
}

impl CountingSink {
    /// Has the human already read this text?
    ///
    /// The worker streams what it says, so the frontends have its last
    /// answer before the worker ends — and the report is that answer.
    fn has_seen(&self, report: &str) -> bool {
        let report = report.trim();
        if report.is_empty() {
            return false;
        }
        let seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        seen.trim_end().ends_with(report)
    }
}

impl Show for CountingSink {
    fn show(&self, m: Msg) {
        match &m {
            Msg::Tool { .. } => {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
            Msg::Text(t) => {
                self.seen.lock().unwrap_or_else(|e| e.into_inner()).push_str(t);
            }
            _ => {}
        }
        self.inner.show(m);
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{block_on, cfg, mock_seq, Recorder};

    /// A model that answers once and stops.
    const ANSWER: &str =
        r#"{"content":[{"type":"text","text":"the report"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;

    /// A worker job over a mock model, wired the way the fleet wires it.
    fn job(port: u16, done: Arc<Mutex<Option<WorkerDone>>>, tools: Arc<AtomicUsize>) -> WorkerJob {
        let router = Arc::new(ToolRouter::new(vec![Box::new(crate::tools::Builtins)]));
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec);
        let seed = crate::test_util::ctx().seed();
        WorkerJob {
            id: AgentId::next(),
            label: "w1".into(),
            task: "count the files".into(),
            scope: CancelScope::new(),
            cfg: cfg(format!("http://127.0.0.1:{port}"), false),
            router,
            sink,
            seed,
            roster: Roster::new(),
            done,
            tools,
        }
    }

    /// The worker's sink, built the way `run_worker` builds it: the
    /// session's sink, attributed to the worker, counting its tool calls.
    fn counting_sink() -> (CountingSink, Arc<AtomicUsize>) {
        let rec = Recorder::default();
        let count = Arc::new(AtomicUsize::new(0));
        let sink = CountingSink {
            inner: AgentSink::new(rec, AgentId::next()),
            count: count.clone(),
            seen: Mutex::new(String::new()),
        };
        (sink, count)
    }

    // ---- the worker's own report -----------------------------------------

    #[test]
    fn a_finished_worker_hands_its_result_back() {
        // The fleet waits on the worker's `done` slot: the worker has to
        // leave its result there, or the wait never ends.
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let done = Arc::new(Mutex::new(None));
        let tools = Arc::new(AtomicUsize::new(0));
        let j = job(port, done.clone(), tools.clone());
        let id = j.id;
        block_on(run_worker(j));
        let slot = done.lock().unwrap();
        let d = slot.as_ref().expect("the worker left its result behind");
        assert_eq!(d.agent.id, id, "the result is the worker's own");
        assert_eq!(d.agent.state(), TurnState::Completed, "and it says how it ended");
        assert_eq!(d.agent.report.as_deref(), Some("the report"), "with the report it produced");
    }

    #[test]
    fn a_worker_whose_result_slot_is_poisoned_leaves_nothing_behind() {
        // The slot the fleet reads is poisoned: a thread died holding
        // it. The worker cannot leave its result there, and it does not
        // panic over it — the wait is what notices.
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let done = Arc::new(Mutex::new(None));
        let doomed = done.clone();
        let _ = std::thread::spawn(move || {
            let _held = doomed.lock().unwrap();
            panic!("the writer died holding the lock");
        })
        .join();
        let tools = Arc::new(AtomicUsize::new(0));
        let j = job(port, done.clone(), tools);
        block_on(run_worker(j));
        let left = done.lock().unwrap_or_else(|e| e.into_inner());
        assert!(left.is_none(), "the worker left nothing behind");
    }

    #[test]
    fn the_report_of_a_worker_is_the_text_it_ended_with() {
        let mut ctx = crate::test_util::ctx();
        ctx.push(Message::User("count the files".into()));
        ctx.push(Message::Assistant(vec![Block::Text("the answer".into())]));
        assert_eq!(
            report_of(&ctx, TurnState::Completed).as_deref(),
            Some("the answer"),
            "the report is the last thing the worker said"
        );
        // a worker that finished without saying anything says so
        let blank = crate::test_util::ctx();
        assert_eq!(
            report_of(&blank, TurnState::Completed).as_deref(),
            Some("no answer"),
            "a silent worker says it had nothing to say"
        );
        // a worker that failed has no report: the failure is the report
        assert_eq!(report_of(&blank, TurnState::Failed), None);
    }

    #[test]
    fn a_worker_that_ended_with_blank_text_has_no_text_of_its_own() {
        // The last thing the worker said was whitespace: that is not a
        // report, so the way it ended is the report.
        let mut ctx = crate::test_util::ctx();
        ctx.push(Message::User("count the files".into()));
        ctx.push(Message::Assistant(vec![Block::Text("   ".into())]));
        assert_eq!(
            report_of(&ctx, TurnState::Completed).as_deref(),
            Some("no answer"),
            "a blank last text is no text: the state says how it ended"
        );
    }

    #[test]
    fn the_last_text_of_a_worker_steps_over_what_is_not_text() {
        // The last turn can end with something that is not text — a
        // thinking block, a tool call. The walk steps over it and finds
        // the text before it.
        let mut ctx = crate::test_util::ctx();
        ctx.push(Message::User("count the files".into()));
        ctx.push(Message::Assistant(vec![
            Block::Text("the answer".into()),
            Block::Thinking { text: "hmm".into(), signature: None },
        ]));
        assert_eq!(
            last_text(&ctx).as_deref(),
            Some("the answer"),
            "the text before the last non-text block is still the last text"
        );
    }

    #[test]
    fn the_last_text_of_a_worker_is_the_text_it_ended_with() {
        let mut ctx = crate::test_util::ctx();
        ctx.push(Message::User("count the files".into()));
        ctx.push(Message::Assistant(vec![Block::Text("the first answer".into())]));
        ctx.push(Message::Assistant(vec![Block::Text("the last answer".into())]));
        assert_eq!(
            last_text(&ctx).as_deref(),
            Some("the last answer"),
            "the last text, walking the transcript backwards"
        );
        // a transcript with no answer in it has no last text
        let mut quiet = crate::test_util::ctx();
        quiet.push(Message::User("count the files".into()));
        assert_eq!(last_text(&quiet), None);
    }

    // ---- the worker's sink -----------------------------------------------

    #[test]
    fn the_sink_counts_the_tool_calls_a_worker_made() {
        let (sink, count) = counting_sink();
        sink.show(Msg::Tool { name: "bash".into(), summary: "echo hi".into(), output: "hi".into() });
        sink.show(Msg::Tool { name: "read_file".into(), summary: "a.txt".into(), output: "x".into() });
        assert_eq!(count.load(Ordering::Relaxed), 2, "two tool calls, two counts");
        // and prose is not a tool call
        sink.show(Msg::Text("not a tool".into()));
        assert_eq!(count.load(Ordering::Relaxed), 2, "prose does not count");
    }

    #[test]
    fn the_sink_keeps_the_text_a_worker_streamed() {
        let (sink, _count) = counting_sink();
        sink.show(Msg::Text("the report".into()));
        assert!(sink.has_seen("the report"), "the sink kept what the worker said");
        assert!(!sink.has_seen("something else"), "and not what it did not say");
        assert!(!sink.has_seen(""), "an empty report is never seen");
    }

    #[test]
    fn the_worker_sink_recovers_from_a_poisoned_lock() {
        // A thread that dies holding the sink's own lock poisons it. The
        // next caller must still get the text: a poisoned lock is a
        // broken *writer*, not a broken sink.
        let seen = Arc::new(Mutex::new(String::new()));
        let doomed = seen.clone();
        let _ = std::thread::spawn(move || {
            let _held = doomed.lock().unwrap();
            panic!("the writer died holding the lock");
        })
        .join();
        let rec = Recorder::default();
        let count = Arc::new(AtomicUsize::new(0));
        let sink = CountingSink {
            inner: AgentSink::new(rec, AgentId::next()),
            count,
            seen: Arc::try_unwrap(seen).ok().expect("the only handle left"),
        };
        assert!(sink.seen.is_poisoned(), "the lock is poisoned");
        sink.show(Msg::Text("the report".into()));
        assert!(sink.has_seen("the report"), "the sink still keeps what the worker said");
    }
}
