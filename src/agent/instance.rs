//! The instance: one agent, one session, as many tasks as it is given.
//!
//! The **fleet** ([`crate::workflow::exec`]) owns the *background* agents —
//! the one-shot workers a model spawns inside a run. This module owns the
//! opposite: a **top-level** agent that outlives a single task. It holds a
//! session, a context, a config and a fleet of its own, and it runs task
//! after task in that one context, so the second task knows what the first
//! one was about.
//!
//! ## The life of an instance
//!
//! Four states, and the table between them is the whole story:
//!
//! ```text
//!            execute                  the task ends
//!   IDLE ───────────────► RUNNING ─────────────────► IDLE
//!    │                     │
//!    │ stop                │ stop
//!    ▼                     ▼
//!  STOPPING ◄──────────────┘
//!    │  the task let go, the runtime is released
//!    ▼
//!  STOPPED
//! ```
//!
//! `STOPPING` is the half-second between "a stop was asked for" and "the
//! task let go and the runtime went". It is not a state a caller waits in,
//! and it is not a state a task can undo: a task that ends *after* a stop
//! began does not put the instance back to `IDLE`, because the stop owns
//! the state from the moment it starts. That is the whole reason the states
//! are a table and not a flag.
//!
//! ## The scope
//!
//! The other half of the instance is its **cancel scope**: every task hangs
//! under it as a child, and a stop cancels it, so the task — and everything
//! the task spawned — dies with the instance. It is the cooperative cancel
//! the run boundary already uses; there is no second mechanism beside it.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use crate::agent::tools::AgentTools;
use crate::cancel::CancelScope;
use crate::config::Config;
use crate::core::agents_md::AgentsMdContext;
use crate::core::context::Context;
use crate::core::ids::AgentId;
use crate::core::ports::ToolRouter;
use crate::core::session::{Convo, Provenance, Session};
use crate::mcp::Hub;
use crate::tools::Builtins;
use crate::ui::{Msg, Show};
use crate::workflow::exec::{AgentCfg, Fleet};
use crate::workflow::run_task;
use crate::Error;

/// Where an instance is in its life.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum InstanceState {
    /// Started, and taking tasks.
    Idle,
    /// A task is running.
    Running,
    /// Stopping: no task is taken any more; the one that runs is being
    /// ended, and the runtime the instance owns is being released.
    Stopping,
    /// Stopped: the runtime is gone. The instance takes no more tasks — a
    /// new instance is a new session.
    Stopped,
}

impl InstanceState {
    /// What may follow this state. The one place the rules live: the
    /// transitions are a table, not a pile of flags, so "a task that ends
    /// while the instance is stopping" has exactly one answer.
    fn allows(self, next: InstanceState) -> bool {
        match self {
            InstanceState::Idle => matches!(next, InstanceState::Running | InstanceState::Stopping),
            InstanceState::Running => matches!(next, InstanceState::Idle | InstanceState::Stopping),
            InstanceState::Stopping => matches!(next, InstanceState::Stopped),
            InstanceState::Stopped => false,
        }
    }
}

/// Which session an instance starts on.
pub(crate) enum SessionSpec {
    /// A new session: its own file, its own context, nothing carried over.
    New,
    /// A saved session: the newest of this project, or the one whose id
    /// starts with the selector. Its transcript becomes the instance's
    /// context, so the instance picks the conversation up where it left
    /// off.
    Resume(Option<String>),
}

/// What an instance is made of.
pub(crate) struct InstanceParts {
    /// The config it runs with. Fixed for the instance's life: the
    /// model, the tools and the bounds do not change under it.
    pub(crate) config: Config,
    /// The project conventions — the AGENTS.md context. Kept so a new
    /// session can build a fresh context out of the same conventions.
    pub(crate) agents_md: AgentsMdContext,
    /// The MCP servers: the connections the instance *borrows* and does
    /// not own. The hub is shared with the rest of the process and
    /// outlives the instance.
    pub(crate) hub: Hub,
    /// Where the instance's events go.
    pub(crate) sink: Arc<dyn Show>,
    /// Which session it starts on.
    pub(crate) session: SessionSpec,
}

/// The instance's life: where it is, what it is running, and how the
/// stop in flight went.
///
/// It lives behind the one short lock of the instance — the lock that
/// decides. Nothing here is held across an await.
struct Life {
    /// Where the instance is.
    state: InstanceState,
    /// The session the instance is on. The manager claims sessions by
    /// this, so a session is advanced by one instance at a time.
    session: String,
    /// The task in flight and the scope it hangs under. `None` when the
    /// instance is idle.
    running: Option<CancelScope>,
    /// The answer of the stop in flight: `None` while a stop runs, `Some`
    /// once it has landed. Every waiter reads the same answer, so two
    /// stops never clean up twice.
    stop: Option<Result<(), String>>,
}

/// What a stop call is to do.
enum StopClaim {
    /// This call does the stopping.
    Mine,
    /// A stop is already in flight: wait for the answer it will leave.
    Wait,
    /// The instance is already stopped: there is nothing to do.
    Done,
}

/// What happened when a task was asked for.
enum Take {
    /// The task was accepted: this is the scope it runs under.
    Taken(CancelScope),
    /// The instance is running a task already.
    Busy,
    /// The instance is stopping or stopped: it takes no task.
    Refused,
}

impl Life {
    /// Move to the next state, if the table allows it. The only way the
    /// state changes.
    fn transition(&mut self, next: InstanceState) -> bool {
        if !self.state.allows(next) { return false; }
        self.state = next;
        true
    }

    /// Take a task. The check and the claim are one step, so a stop that
    /// lands at the same moment cannot slip between them: the task is
    /// either accepted before the stop, or refused by it — never both.
    fn take(&mut self, scope: CancelScope) -> Take {
        if !self.transition(InstanceState::Running) {
            return if self.state == InstanceState::Running {
                Take::Busy
            } else {
                Take::Refused
            };
        }
        self.running = Some(scope.clone());
        Take::Taken(scope)
    }

    /// The task ended. The instance is idle again — unless a stop took it
    /// in the meantime: the stop owns the state from the moment it
    /// starts, and a task that ends late must not reopen an instance that
    /// is on its way out.
    fn release(&mut self) {
        self.running = None;
        self.transition(InstanceState::Idle);
    }

    /// Begin a stop: no task is taken from here on.
    fn claim_stop(&mut self) -> StopClaim {
        match self.state {
            InstanceState::Stopped => StopClaim::Done,
            InstanceState::Idle | InstanceState::Running => {
                self.transition(InstanceState::Stopping);
                self.stop = None;
                StopClaim::Mine
            }
            InstanceState::Stopping => {
                if self.stop.is_some() {
                    // The last attempt failed and left its answer behind:
                    // this call is the retry, and it takes the slot over.
                    self.stop = None;
                    StopClaim::Mine
                } else {
                    // A stop is in flight: wait for the answer it will
                    // leave rather than stopping twice.
                    StopClaim::Wait
                }
            }
        }
    }

    /// The stop worked: the runtime is released and the instance is done.
    fn finish_stop(&mut self) {
        self.running = None;
        self.transition(InstanceState::Stopped);
        self.stop = Some(Ok(()));
    }

    /// The stop did not work. The instance stays in the stopping state —
    /// it takes no task — and the answer is left for the waiters.
    fn fail_stop(&mut self, why: String) {
        self.stop = Some(Err(why));
    }

    /// The instance is given back before it ever ran. It goes out the
    /// way every instance goes out — through the stopping state — even
    /// though there is nothing to stop.
    fn abandon(&mut self) {
        self.transition(InstanceState::Stopping);
        self.transition(InstanceState::Stopped);
    }
}

/// A live instance, and the handle the caller holds on it.
///
/// The handle is cheap to clone and cheap to keep: everything the instance
/// owns sits behind one `Arc`, and the manager holds the same handle the
/// caller does.
#[derive(Clone)]
pub(crate) struct Instance {
    /// Who this is: the manager's key, and the name a log line reads.
    id: AgentId,
    inner: Arc<Inner>,
}

/// Everything one instance owns.
struct Inner {
    /// The life: the state, the session, the task in flight, the stop's
    /// answer. The short lock — the one that decides, and the only one a
    /// caller may read without awaiting.
    life: Mutex<Life>,
    /// The instance's own cancel tree. Every task is a child of it, and a
    /// stop cancels it: the task, and everything the task spawned, dies
    /// with the instance.
    scope: CancelScope,
    /// The config — fixed for the instance's life. It lives here and not
    /// in the bits, because the stop reads it while a task holds the bits.
    config: Config,
    /// The tools the instance may call: the built-ins, the shared MCP
    /// servers, and the instance's own fleet.
    router: Arc<ToolRouter>,
    /// Where the instance's events go.
    sink: Arc<dyn Show>,
    /// The parts a task runs on. One lock, because one instance runs one
    /// task at a time — and the task holds it for the length of the run,
    /// so it is the async one.
    bits: tokio::sync::Mutex<Bits>,
    /// Woken when the life changes: a task ended, or a stop landed.
    changed: Notify,
}

/// The parts a task runs on.
struct Bits {
    /// The transcript the model sees.
    context: Context,
    /// The session the transcript is written to.
    convo: Convo,
    /// The instance's fleet: the background agents it may spawn.
    fleet: Fleet,
    /// The project conventions, kept so a new session can build a fresh
    /// context out of them.
    agents_md: AgentsMdContext,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("id", &self.id)
            .field("state", &self.state())
            .finish()
    }
}

impl Instance {
    /// Build an instance and start it.
    ///
    /// The scope is the caller's, and a failure **cancels** it: an
    /// instance that never came to be leaves nothing running under it.
    /// The instance comes back idle, with its session chosen and its
    /// context ready.
    pub(crate) fn start(scope: CancelScope, parts: InstanceParts) -> crate::Result<Self> {
        let built = Self::build(scope.clone(), parts);
        if built.is_err() {
            scope.cancel();
        }
        built
    }

    /// The instance's own pieces, in the order they have to come up:
    /// the session first (it is the one step that can fail on the disk),
    /// then the context that reads it, then the fleet, then the tools.
    fn build(scope: CancelScope, parts: InstanceParts) -> crate::Result<Self> {
        let InstanceParts { config, agents_md, hub, sink, session } = parts;
        let (session, history) = match session {
            SessionSpec::New => (
                Session::new(&provenance(&config), crate::config::HISTORY_FORMAT),
                Vec::new(),
            ),
            SessionSpec::Resume(selector) => Session::load(selector.as_deref())?,
        };
        let session_id = session.id().to_string();
        // The context: the transcript the model sees. A resumed session
        // starts with the transcript it saved.
        let mut context = Context::new(&agents_md, config.context_size);
        context.history_mut().extend(history);
        // The fleet: this instance's own background agents.
        let agent_cfg = AgentCfg::from_config(&config);
        let fleet = Fleet::new(agent_cfg);
        if agent_cfg.enabled() {
            // The fleet's tool schemas are on the wire either way; the
            // reminder is what tells the model they are there.
            context.reminders.push(AgentTools::reminder());
        }
        // The tools: the built-ins, the shared MCP servers, and the
        // instance's own fleet — the same three the composition root
        // builds, with the fleet bound to *this* instance.
        let router = Arc::new(ToolRouter::new(vec![
            Box::new(Builtins),
            Box::new(hub),
            Box::new(AgentTools::new(Arc::new(fleet.clone()))),
        ]));
        Ok(Self {
            id: AgentId::next(),
            inner: Arc::new(Inner {
                life: Mutex::new(Life {
                    state: InstanceState::Idle,
                    session: session_id,
                    running: None,
                    stop: None,
                }),
                scope,
                config,
                router,
                sink,
                bits: tokio::sync::Mutex::new(Bits {
                    context,
                    convo: Convo::persistent(session),
                    fleet,
                    agents_md,
                }),
                changed: Notify::new(),
            }),
        })
    }

    /// Who this is.
    pub(crate) fn id(&self) -> AgentId {
        self.id
    }

    /// Where it is in its life.
    pub(crate) fn state(&self) -> InstanceState {
        self.lock().state
    }

    /// The session it is on.
    pub(crate) fn session_id(&self) -> String {
        self.lock().session.clone()
    }

    /// Run one task on an idle instance.
    ///
    /// A busy instance says so rather than taking a second task: one
    /// instance runs one task at a time, and a task that runs on another
    /// instance's context is a task whose transcript belongs to somebody
    /// else.
    pub(crate) async fn execute(&self, task: &str) -> crate::Result<()> {
        let scope = {
            let mut life = self.lock();
            let state = life.state;
            // The task's own scope: a child of the instance's, so a stop
            // reaches it — and a task minted under a dead instance is
            // born dead.
            let scope = self.inner.scope.child();
            match life.take(scope) {
                Take::Taken(scope) => scope,
                Take::Busy => {
                    return Err(Error::Msg(format!(
                        "agent {} is busy — it is running a task",
                        self.id
                    )))
                }
                Take::Refused => {
                    return Err(Error::Msg(format!(
                        "agent {} is {} — it takes no task",
                        self.id,
                        if state == InstanceState::Stopped { "stopped" } else { "stopping" }
                    )))
                }
            }
        };
        let res = self.run(task, &scope).await;
        // The task is over. The instance is idle again — unless a stop
        // has taken it in the meantime.
        {
            let mut life = self.lock();
            life.release();
        }
        self.inner.changed.notify_waiters();
        res
    }

    /// Stop the instance: no task is taken any more, the one that runs
    /// is ended, and the runtime the instance owns is released.
    ///
    /// The stop is the cooperative cancel the run boundary already uses —
    /// the instance's scope dies, and the task under it lets go at its
    /// next suspension point. There is no second kind of stop and no
    /// force: the wait that follows the cancel is the wait the
    /// configuration already budgets, and a task that will not let go is
    /// reported as such rather than pretended away.
    pub(crate) async fn stop(&self) -> crate::Result<()> {
        let claim = { self.lock().claim_stop() };
        match claim {
            StopClaim::Done => return Ok(()),
            StopClaim::Wait => return self.wait_for_stop().await,
            StopClaim::Mine => {}
        }
        // The release: the instance's scope dies, and with it the task
        // that runs under it.
        self.inner.scope.cancel();
        // The wait: the task lets go and wraps up. What it leaves
        // behind — the transcript, the files it wrote, the session on
        // disk — is not the instance's to delete: a stopped instance gives
        // up its *runtime*, not its work.
        let ended = if self.running() { self.wait_for_task().await } else { true };
        let outcome = if ended {
            Ok(())
        } else {
            Err(format!(
                "the task did not stop within {}ms",
                self.inner.config.agent_wait_ms
            ))
        };
        {
            let mut life = self.lock();
            match &outcome {
                Ok(()) => life.finish_stop(),
                Err(why) => life.fail_stop(why.clone()),
            }
        }
        self.inner.changed.notify_waiters();
        outcome.map_err(Error::Msg)
    }

    /// Start a new session on this instance: a fresh file, a fresh
    /// context, and nothing carried over from the old one.
    ///
    /// Only an idle instance may switch: a session *is* the context a task
    /// runs in, and a task that is running already has that context.
    pub(crate) async fn new_session(&self) -> crate::Result<String> {
        let not_idle = || {
            Error::Msg(format!(
                "agent {} is {} — a new session needs an idle instance",
                self.id,
                match self.state() {
                    InstanceState::Running => "running",
                    InstanceState::Stopping => "stopping",
                    InstanceState::Stopped => "stopped",
                    InstanceState::Idle => "idle",
                }
            ))
        };
        // The first check: a running instance is refused without waiting
        // for anything.
        if self.state() != InstanceState::Idle {
            return Err(not_idle());
        }
        let mut bits = self.inner.bits.lock().await;
        // The check again, now that the parts are in hand: a task that
        // took the instance in between must not have its context swapped
        // out from under it.
        if self.state() != InstanceState::Idle {
            return Err(not_idle());
        }
        let session = Session::new(&provenance(&self.inner.config), crate::config::HISTORY_FORMAT);
        let session_id = session.id().to_string();
        // A fresh context: the same conventions, an empty transcript.
        let mut context = Context::new(&bits.agents_md, self.inner.config.context_size);
        if AgentCfg::from_config(&self.inner.config).enabled() {
            context.reminders.push(AgentTools::reminder());
        }
        bits.context = context;
        bits.convo = Convo::persistent(session);
        {
            let mut life = self.lock();
            life.session = session_id.clone();
        }
        Ok(session_id)
    }

    /// Give an instance back before it is used: the scope is cancelled,
    /// so nothing can be left under it, and the instance takes no task
    /// again.
    pub(crate) fn abandon(&self) {
        self.inner.scope.cancel();
        let mut life = self.lock();
        life.abandon();
    }

    /// The task itself: the run boundary, with this instance's parts.
    async fn run(&self, task: &str, scope: &CancelScope) -> crate::Result<()> {
        let mut bits = self.inner.bits.lock().await;
        let Bits { context, convo, fleet, .. } = &mut *bits;
        run_task(
            task,
            &self.inner.config,
            context,
            convo,
            &self.inner.router,
            scope,
            fleet,
            &self.inner.sink,
            // The failure is the caller's to read: `execute` hands it
            // back, so there is nothing to say here.
            &silent,
        )
        .await
    }

    /// Is a task in flight?
    fn running(&self) -> bool {
        self.lock().running.is_some()
    }

    /// Wait for the task in flight to end, on the configuration's wait
    /// budget. `true` when it ended, `false` when the budget ran out.
    async fn wait_for_task(&self) -> bool {
        let deadline = Instant::now() + self.wait_budget();
        loop {
            // Register for the wakeup *before* the check, so a task that
            // ends in between cannot leave the wait asleep.
            let notified = self.inner.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.running() {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            tokio::select! {
                _ = &mut notified => {}
                _ = tokio::time::sleep(left) => { return !self.running(); }
            }
        }
    }

    /// Wait for the stop in flight to land, and answer with it. The
    /// answer is the same one every waiter gets: the stop is done once.
    async fn wait_for_stop(&self) -> crate::Result<()> {
        let deadline = Instant::now() + self.wait_budget();
        loop {
            let notified = self.inner.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let answer = { self.lock().stop.clone() };
            if let Some(answer) = answer {
                return answer.map_err(Error::Msg);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(Error::Msg("the stop is still in flight".into()));
            }
            tokio::select! {
                _ = &mut notified => {}
                _ = tokio::time::sleep(left) => {}
            }
        }
    }

    /// How long a stop waits for the task to let go: the configuration's
    /// own wait budget — the same number the fleet waits on.
    fn wait_budget(&self) -> Duration {
        Duration::from_millis(self.inner.config.agent_wait_ms)
    }

    /// The short lock. Poisoning is ignored: a panic in one task must
    /// not take the instance's life with it.
    fn lock(&self) -> MutexGuard<'_, Life> {
        self.inner
            .life
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The session's header: who wrote this conversation, with what.
fn provenance(cfg: &Config) -> Provenance {
    Provenance {
        model: cfg.model.clone(),
        protocol: cfg.protocol_label().into(),
        base_url: cfg.base_url.clone(),
    }
}

/// A mapper that says nothing: the failure belongs to the caller of
/// `execute`, and it is handed back to them.
fn silent(_e: &Error) -> Option<Msg> {
    None
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{block_on, cfg, mock_seq};
    use std::path::PathBuf;

    /// A model that answers once and stops.
    const ANSWER: &str =
        r#"{"content":[{"type":"text","text":"three files"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;

    /// The parts of an instance, pointed at a mock model.
    fn parts(port: u16, hub: &Hub) -> InstanceParts {
        InstanceParts {
            config: cfg(format!("http://127.0.0.1:{port}"), false),
            agents_md: AgentsMdContext::empty(&PathBuf::from(".")),
            hub: hub.clone(),
            sink: Arc::new(crate::test_util::sink()),
            session: SessionSpec::New,
        }
    }

    #[test]
    fn the_states_only_go_where_the_table_says() {
        assert!(InstanceState::Idle.allows(InstanceState::Running));
        assert!(InstanceState::Idle.allows(InstanceState::Stopping));
        assert!(InstanceState::Running.allows(InstanceState::Idle));
        assert!(InstanceState::Running.allows(InstanceState::Stopping));
        assert!(InstanceState::Stopping.allows(InstanceState::Stopped));
        // A stopped instance takes nothing and goes nowhere.
        assert!(!InstanceState::Stopped.allows(InstanceState::Running));
        assert!(!InstanceState::Stopped.allows(InstanceState::Idle));
        assert!(!InstanceState::Stopped.allows(InstanceState::Stopping));
        // A task that ends while the instance is stopping does not put it
        // back to idle: the stop owns the state from the moment it starts.
        assert!(!InstanceState::Stopping.allows(InstanceState::Idle));
        assert!(!InstanceState::Stopping.allows(InstanceState::Running));
    }

    #[test]
    fn a_fresh_instance_is_idle_and_ready() {
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        block_on(async {
            let hub = Hub::empty();
            let instance = Instance::start(CancelScope::new(), parts(port, &hub)).unwrap();
            assert_eq!(instance.state(), InstanceState::Idle);
            assert!(!instance.session_id().is_empty());
            hub.shutdown().await;
        });
    }

    #[test]
    fn a_failed_start_cancels_the_scope_it_was_given() {
        let _lock = crate::test_util::env_lock();
        let _home = crate::test_util::HomeOverride::new("instance_failed_start");
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        block_on(async {
            let hub = Hub::empty();
            // A session that cannot be resumed: the start fails, and the
            // scope it was given is cancelled rather than left alive.
            let mut p = parts(port, &hub);
            p.session = SessionSpec::Resume(Some("no-such-session".into()));
            let scope = CancelScope::new();
            let res = Instance::start(scope.clone(), p);
            assert!(res.is_err(), "the start failed");
            assert!(scope.is_cancelled(), "the scope was let go");
            hub.shutdown().await;
        });
    }
}
