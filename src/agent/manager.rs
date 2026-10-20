//! The manager: the live instances, and the four operations on them.
//!
//! The manager is the one place that knows which instances exist. It
//! holds no work of its own: an instance runs its tasks, and the manager
//! only says *yes* or *no* —
//!
//! - **create** builds an instance and starts it, and claims the session
//!   it lands on;
//! - **execute** runs a task on an idle instance;
//! - **stop** ends the instance's runtime;
//! - **dispose** stops the instance and lets the manager forget it.
//!
//! The session claim is the manager's one piece of policy: a session is
//! advanced by one instance at a time, because two instances on one
//! session would write one transcript over each other. A second instance
//! on a claimed session is refused, and the refusal says who holds it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::agent::instance::{Instance, InstanceParts, InstanceState};
use crate::cancel::CancelScope;
use crate::core::ids::AgentId;
use crate::Error;

/// The manager: the live instances, and the four operations on them.
///
/// The handle is cheap to clone, so a caller may keep one and hand it
/// around; the registry behind it is the single record of which
/// instances exist.
#[derive(Clone)]
pub(crate) struct AgentManager {
    inner: Arc<Mutex<Inner>>,
}

/// What the manager knows.
struct Inner {
    /// The live instances, by id.
    instances: BTreeMap<AgentId, Instance>,
    /// The sessions the live instances hold. A session is advanced by
    /// one instance at a time.
    claims: BTreeMap<String, AgentId>,
}

impl AgentManager {
    /// A manager with no instances.
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                instances: BTreeMap::new(),
                claims: BTreeMap::new(),
            })),
        }
    }

    /// Create an instance and start it.
    ///
    /// The instance comes back idle, with its session chosen and its
    /// context ready. The session is claimed: no other instance may hold
    /// it while this one lives.
    pub(crate) fn create(&self, parts: InstanceParts) -> crate::Result<Instance> {
        let scope = CancelScope::new();
        // The instance: its own scope, its own session, its own context.
        // A failure cancels the scope, so nothing is left running under
        // an instance that never came to be.
        let instance = Instance::start(scope, parts)?;
        let session = instance.session_id();
        let mut inner = self.lock();
        if let Some(owner) = inner.claims.get(&session).copied() {
            drop(inner);
            // The instance is not handed out: it goes back with
            // everything it holds, and nothing ever ran under it.
            instance.abandon();
            return Err(Error::Msg(format!(
                "session {session} is already held by agent {owner}"
            )));
        }
        inner.claims.insert(session, instance.id());
        inner.instances.insert(instance.id(), instance.clone());
        Ok(instance)
    }

    /// Run a task on an idle instance.
    ///
    /// A busy instance says so rather than taking a second task, and an
    /// instance the manager does not know is not one it can run.
    pub(crate) async fn execute(&self, instance: &Instance, task: &str) -> crate::Result<()> {
        self.managed(instance)?;
        instance.execute(task).await
    }

    /// Stop an instance: no task is taken any more, the one that runs is
    /// ended, and the runtime is released.
    ///
    /// A stop that is already in flight is not a second stop: the caller
    /// waits for the one in flight and gets its answer.
    pub(crate) async fn stop(&self, instance: &Instance) -> crate::Result<()> {
        instance.stop().await
    }

    /// Stop the instance and let the manager forget it.
    ///
    /// The stop comes first: an instance that could not be stopped is
    /// kept, so the stop can be tried again. A disposed instance is
    /// disposed twice — the second call finds nothing to do and says so.
    pub(crate) async fn dispose(&self, instance: &Instance) -> crate::Result<()> {
        if !self.knows(instance) {
            return Ok(()); // already disposed
        }
        self.stop(instance).await?;
        let mut inner = self.lock();
        inner.instances.remove(&instance.id());
        inner.claims.remove(&instance.session_id());
        Ok(())
    }

    /// Start a new session on an idle instance: a fresh file, a fresh
    /// context, and nothing carried over from the old one. The claim
    /// moves with the session.
    pub(crate) async fn new_session(&self, instance: &Instance) -> crate::Result<()> {
        self.managed(instance)?;
        let old = instance.session_id();
        let new = instance.new_session().await?;
        let mut inner = self.lock();
        inner.claims.remove(&old);
        inner.claims.insert(new, instance.id());
        Ok(())
    }

    /// Where an instance is in its life, if the manager knows it.
    pub(crate) fn state(&self, instance: &Instance) -> Option<InstanceState> {
        if self.knows(instance) {
            Some(instance.state())
        } else {
            None
        }
    }

    /// The live instances, by id.
    pub(crate) fn list(&self) -> Vec<Instance> {
        self.lock().instances.values().cloned().collect()
    }

    /// Is this one of ours?
    fn knows(&self, instance: &Instance) -> bool {
        self.lock().instances.contains_key(&instance.id())
    }

    /// The instance must be one of ours: a handle the manager does not
    /// know is not one it can run.
    fn managed(&self, instance: &Instance) -> crate::Result<()> {
        if self.knows(instance) {
            return Ok(());
        }
        Err(Error::Msg(format!(
            "agent {} is not managed by this manager",
            instance.id()
        )))
    }

    /// The short lock. Poisoning is ignored: a panic in one instance
    /// must not take the manager's registry with it.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::instance::{InstanceParts, SessionSpec};
    use crate::core::agents_md::AgentsMdContext;
    use crate::mcp::Hub;
    use crate::test_util::{
        block_on, cfg, mock_seq, mock_stall, request_body, HomeOverride, Recorder,
    };
    use crate::ui::Show;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// A model that answers once and stops.
    const ANSWER: &str =
        r#"{"content":[{"type":"text","text":"three files"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
    /// A model that refuses.
    const REFUSED: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"no"}}"#;
    /// A model that calls a tool and then answers. The tool call is what
    /// makes the agent come back for a second request — and what keeps a
    /// task running while the test looks at it.
    const TOOL_SSE: &str = concat!(
        r#"data: {"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}"#, "\n\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#, "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"echo hi\"}"}}"#, "\n\n",
        r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
        r#"data: {"type":"message_delta","usage":{"output_tokens":1}}"#, "\n\n",
        r#"data: {"type":"message_stop"}"#, "\n\n");
    /// The second response: the head and one text delta, then the
    /// connection stays open and the content never arrives.
    const STALLED_SSE: &str = concat!(
        r#"data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}"#, "\n\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial answ"}}"#, "\n\n");

    /// The parts an instance is made of, pointed at a mock model.
    fn parts(port: u16, hub: &Hub, sink: &Arc<dyn Show>) -> InstanceParts {
        InstanceParts {
            config: cfg(format!("http://127.0.0.1:{port}"), false),
            agents_md: AgentsMdContext::empty(&PathBuf::from(".")),
            hub: hub.clone(),
            sink: sink.clone(),
            session: SessionSpec::New,
        }
    }

    /// A model that writes a file, then answers.
    fn write_call(path: &str) -> String {
        json!({
            "content": [{
                "type": "tool_use",
                "id": "t1",
                "name": "write_file",
                "input": { "path": path, "content": "written" },
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1 },
        })
        .to_string()
    }

    /// Wait for the mock to be holding the connection open: the task is
    /// running, and it will keep running until something stops it.
    async fn wait_for_stall(flag: &AtomicBool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !flag.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the mock never stalled");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    // ---- the four operations ------------------------------------------------

    #[test]
    fn a_created_instance_is_idle_and_takes_a_task() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_create");
        let (port, reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            assert_eq!(instance.state(), InstanceState::Idle);
            assert_eq!(manager.state(&instance), Some(InstanceState::Idle), "the manager knows it");
            manager.execute(&instance, "count the files").await.unwrap();
            assert_eq!(instance.state(), InstanceState::Idle, "the task ended");
            manager.dispose(&instance).await.unwrap();
            assert_eq!(manager.state(&instance), None, "the manager forgot it");
            hub.shutdown().await;
        });
        assert_eq!(reqs.lock().unwrap().len(), 1, "the model was called once");
    }

    #[test]
    fn the_second_task_in_a_session_carries_the_first_ones_context() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_context");
        let (port, reqs) = mock_seq(vec![(200, ANSWER.into()), (200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            manager.execute(&instance, "the first task").await.unwrap();
            manager.execute(&instance, "the second task").await.unwrap();
            hub.shutdown().await;
        });
        let seen = reqs.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "one request per task: {seen:?}");
        let second = request_body(&seen[1]).to_string();
        assert!(second.contains("the first task"), "the first task is in the second request: {second}");
        assert!(second.contains("three files"), "the first answer is in the second request: {second}");
        assert!(second.contains("the second task"), "the second task is in its own request: {second}");
    }

    #[test]
    fn a_new_session_starts_with_an_empty_context() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_new_session");
        let (port, reqs) = mock_seq(vec![(200, ANSWER.into()), (200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            let first = instance.session_id();
            manager.execute(&instance, "the first task").await.unwrap();
            manager.new_session(&instance).await.unwrap();
            assert_ne!(instance.session_id(), first, "the new session is its own file");
            manager.execute(&instance, "the second task").await.unwrap();
            hub.shutdown().await;
        });
        let seen = reqs.lock().unwrap().clone();
        let second = request_body(&seen[1]).to_string();
        assert!(second.contains("the second task"), "the new task is in the request: {second}");
        assert!(!second.contains("the first task"), "the new session starts empty: {second}");
    }

    #[test]
    fn a_failed_task_leaves_the_instance_usable() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_failed_task");
        let (port, _reqs) = mock_seq(vec![(400, REFUSED.into()), (200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            let err = manager.execute(&instance, "the doomed task").await.unwrap_err();
            assert!(matches!(err, crate::Error::Api(400, _)), "the failure came back: {err:?}");
            assert_eq!(instance.state(), InstanceState::Idle, "the instance is idle again");
            // a healthy instance is still good for the next task
            manager.execute(&instance, "the next task").await.unwrap();
            hub.shutdown().await;
        });
    }

    // ---- the stop ------------------------------------------------------------

    #[test]
    fn a_busy_instance_refuses_a_second_task() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_busy");
        let (port, stalled) = mock_stall(TOOL_SSE, STALLED_SSE);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let mut p = parts(port, &hub, &sink);
            p.config.streaming = true;
            let instance = manager.create(p).unwrap();
            let m = manager.clone();
            let i = instance.clone();
            let run = tokio::spawn(async move { m.execute(&i, "the long task").await });
            wait_for_stall(&stalled).await;
            let err = manager
                .execute(&instance, "the second task")
                .await
                .unwrap_err();
            assert!(err.to_string().contains("busy"), "the second task was refused: {err}");
            manager.stop(&instance).await.unwrap();
            let _ = run.await;
            hub.shutdown().await;
        });
    }

    #[test]
    fn a_stop_ends_the_running_task_and_stops_the_instance() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_stop");
        let (port, stalled) = mock_stall(TOOL_SSE, STALLED_SSE);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let mut p = parts(port, &hub, &sink);
            p.config.streaming = true;
            let instance = manager.create(p).unwrap();
            let m = manager.clone();
            let i = instance.clone();
            let run = tokio::spawn(async move { m.execute(&i, "the long task").await });
            wait_for_stall(&stalled).await;
            manager.stop(&instance).await.unwrap();
            assert_eq!(instance.state(), InstanceState::Stopped);
            let res = run.await.unwrap();
            assert!(res.is_err(), "the task was interrupted, not finished: {res:?}");
            hub.shutdown().await;
        });
    }

    #[test]
    fn a_task_that_ends_during_a_stop_does_not_reopen_the_instance() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_race");
        let (port, stalled) = mock_stall(TOOL_SSE, STALLED_SSE);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let mut p = parts(port, &hub, &sink);
            p.config.streaming = true;
            // No budget to wait for the task: the stop lands while the
            // task is still running, and the instance stays in the
            // stopping state — which is exactly the window this test
            // needs to look at. A stop with a budget would finish before
            // the test could see it.
            p.config.agent_wait_ms = 0;
            let instance = manager.create(p).unwrap();
            let m = manager.clone();
            let i = instance.clone();
            let run = tokio::spawn(async move { m.execute(&i, "the long task").await });
            wait_for_stall(&stalled).await;
            // the stop begins while the task runs
            let err = manager.stop(&instance).await.unwrap_err();
            assert!(err.to_string().contains("did not stop"), "the failure was reported: {err}");
            assert_eq!(instance.state(), InstanceState::Stopping, "the instance is stopping");
            // from here on the instance takes no task
            let err = manager
                .execute(&instance, "a late task")
                .await
                .unwrap_err();
            assert!(err.to_string().contains("takes no task"), "the late task was refused: {err}");
            // the manager keeps the instance, so the stop can be tried again
            assert_eq!(manager.list().len(), 1, "the instance was kept");
            assert!(manager.dispose(&instance).await.is_err(), "the dispose kept it too");
            assert_eq!(manager.list().len(), 1, "the instance was kept");
            // the task ends — and the instance does not go back to idle
            let _ = run.await;
            assert_eq!(
                instance.state(),
                InstanceState::Stopping,
                "the task that ended during the stop did not reopen the instance"
            );
            // now the stop can finish
            manager.stop(&instance).await.unwrap();
            assert_eq!(instance.state(), InstanceState::Stopped);
            hub.shutdown().await;
        });
    }

    #[test]
    fn a_repeated_stop_and_dispose_are_safe() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_repeat");
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            manager.stop(&instance).await.unwrap();
            manager.stop(&instance).await.unwrap();
            manager.dispose(&instance).await.unwrap();
            manager.dispose(&instance).await.unwrap();
            assert_eq!(manager.list().len(), 0, "the manager forgot the instance");
            hub.shutdown().await;
        });
    }

    // ---- the session ---------------------------------------------------------

    #[test]
    fn two_instances_cannot_hold_one_session() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_claim");
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            manager.execute(&instance, "a task").await.unwrap();
            let session = instance.session_id();
            // a second instance on the same session is refused
            let mut p = parts(port, &hub, &sink);
            p.session = SessionSpec::Resume(Some(session));
            let err = manager.create(p).unwrap_err();
            assert!(err.to_string().contains("already held"), "the claim was enforced: {err}");
            assert_eq!(manager.list().len(), 1, "the refused instance is not in the manager");
            hub.shutdown().await;
        });
    }

    #[test]
    fn the_session_and_the_work_survive_the_recycling() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_recycle");
        let dir = crate::test_util::temp_dir("manager_work");
        let written = dir.join("written.txt");
        let (port, _reqs) = mock_seq(vec![(200, write_call(&written.display().to_string())), (200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        let session = block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            manager.execute(&instance, "write the file").await.unwrap();
            let session = instance.session_id();
            manager.dispose(&instance).await.unwrap();
            hub.shutdown().await;
            session
        });
        // the file the agent wrote is still there
        assert!(written.exists(), "the work survived the recycling");
        assert_eq!(
            std::fs::read_to_string(&written).unwrap(),
            "written",
            "the file is the one the agent wrote"
        );
        // and so is the session
        let dir = crate::core::session::project_dir().expect("a project dir under the sandbox home");
        let path = dir.join(format!("{session}.jsonl"));
        assert!(path.exists(), "the session survived: {}", path.display());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("write the file"), "the transcript survived: {text}");
    }

    // ---- the minimal usage ---------------------------------------------------

    /// The minimal usage, end to end: create the instance, run two tasks
    /// in one session, stop it, and let the manager go.
    ///
    /// ```ignore
    /// let manager = AgentManager::new();
    /// let instance = manager.create(parts)?;
    /// manager.execute(&instance, "the first task").await?;
    /// manager.execute(&instance, "the second task").await?;  // same session, same context
    /// manager.stop(&instance).await?;
    /// manager.dispose(&instance).await?;
    /// ```
    #[test]
    fn the_minimal_usage_creates_runs_stops_and_recycles() {
        let _lock = crate::test_util::env_lock();
        let _home = HomeOverride::new("manager_usage");
        let (port, reqs) = mock_seq(vec![(200, ANSWER.into()), (200, ANSWER.into())]);
        let rec = Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        let manager = AgentManager::new();
        block_on(async {
            let hub = Hub::empty();
            let instance = manager.create(parts(port, &hub, &sink)).unwrap();
            manager.execute(&instance, "the first task").await.unwrap();
            manager.execute(&instance, "the second task").await.unwrap();
            manager.stop(&instance).await.unwrap();
            assert_eq!(instance.state(), InstanceState::Stopped);
            manager.dispose(&instance).await.unwrap();
            assert_eq!(manager.list().len(), 0);
            hub.shutdown().await;
        });
        assert_eq!(reqs.lock().unwrap().len(), 2, "both tasks ran");
    }
}
