//! The run boundary: one spelling of "run a task".
//!
//! Three frontends — the one-shot, the log REPL, the TUI — used to spell
//! the same sequence out for themselves: show the task, push it into the
//! transcript, save, run the turn, save again, end the run. Every
//! spelling was free to drift, and a frontend that forgot a step would
//! lose the transcript without saying so.
//!
//! This module is the one spelling. [`run_task`] is the run, and it
//! guarantees:
//!
//! - the task is **on disk before the first model call**, so a crash
//!   mid-run leaves a transcript rather than a silence;
//! - the run is bracketed by exactly one `TaskBegin` and one `TaskEnd` —
//!   *always*, including the error path, so a frontend never dangles;
//! - a failed save **warns and does not abort**: the conversation lives
//!   in memory either way, and the agent must not die because the disk
//!   did.
//!
//! What it does *not* do is decide what an error looks like. `run_task`
//! hands the run's `Result` back unchanged; the frontends map it (the
//! plain log and the TUI show a note, the one-shot propagates it so the
//! process exit code stays right). Presentation belongs to the
//! frontend, and the boundary does not own it.

use crate::agent::runner::run_agent;
use crate::agent::Agent;
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::core::context::Context;
use crate::core::ir::user_message;
use crate::core::ports::ToolRouter;
use crate::core::session::Convo;
use crate::ui::{Msg, Sev, Show};

/// Run one task, from the user's words to the end of the run.
///
/// The order is the contract: show the task, put it in the transcript,
/// put the transcript on disk, run the agent, put the transcript on disk
/// again, end the run. The saves are best-effort — a failure warns and
/// the run continues — and the run is bracketed by exactly one
/// `TaskBegin` and one `TaskEnd` whatever happens in between.
///
/// The `Result` is the run's own, unchanged: the caller decides what an
/// error means.
pub(crate) async fn run_task(
    task: &str,
    cfg: &Config,
    ctx: &mut Context,
    convo: &mut Convo,
    router: &ToolRouter,
    token: &CancelToken,
    sink: &dyn Show,
) -> crate::Result<()> {
    sink.show(Msg::TaskBegin(task.to_string()));
    ctx.push(user_message(task));
    // the task is on disk before the first stone moves
    if let Err(e) = convo.persist(ctx.history()) {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    let mut agent = Agent::main(task);
    let res = run_agent(&mut agent, cfg, ctx, router, token, sink).await;
    if let Err(e) = convo.persist(ctx.history()) {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    // run boundary: the file never ends mid-run
    sink.show(Msg::TaskEnd);
    res
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ir::Message;
    use crate::core::session::{Provenance, Session};
    use crate::mcp::Hub;
    use crate::test_util::{block_on, cfg, mock_seq, mock_seq_with};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// A model that answers once and stops.
    const ANSWER: &str =
        r#"{"content":[{"type":"text","text":"three files"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
    /// A model that refuses.
    const REFUSED: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"no"}}"#;

    /// A sink that keeps every message: the tests assert on the run's own
    /// sequence, not on what a frontend would draw.
    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<Msg>>>);

    impl Recorder {
        fn msgs(&self) -> Vec<Msg> { self.0.lock().unwrap().clone() }
    }

    impl Show for Recorder {
        fn show(&self, m: Msg) { self.0.lock().unwrap().push(m); }
    }

    /// The tool router a run uses: the four built-ins and an empty hub.
    /// The hub handle comes back so the test can shut its actor task down.
    fn router() -> (ToolRouter, Hub) {
        let hub = Hub::empty();
        let router = ToolRouter::new(vec![
            Box::new(crate::tools::Builtins),
            Box::new(hub.clone()),
        ]);
        (router, hub)
    }

    fn provenance() -> Provenance {
        Provenance {
            model: "test-model".into(),
            protocol: "minimax".into(),
            base_url: "http://127.0.0.1".into(),
        }
    }

    /// A home directory the run may write into, and the name of the
    /// variable that points at it.
    fn home_var() -> &'static str {
        if cfg!(windows) { "USERPROFILE" } else { "HOME" }
    }

    /// Point the home variable at a fresh temp dir and hand back the
    /// guard that puts it back. The override lasts only as long as the
    /// guard lives, and the guard lives only as long as the test needs
    /// the path to be *computed* — a session's path is fixed when the
    /// session is made, so the run itself never looks at home again.
    struct HomeOverride {
        prev: Option<OsString>,
        dir: PathBuf,
    }

    impl HomeOverride {
        fn new(tag: &str) -> Self {
            let dir = crate::test_util::temp_dir(&format!("workflow_home_{tag}"));
            let prev = std::env::var_os(home_var());
            std::env::set_var(home_var(), &dir);
            Self { prev, dir }
        }

        /// Put the home variable back. The session's path is already
        /// fixed, so the rest of the test does not need the sandbox to
        /// stay in place — and the shorter the window, the smaller the
        /// chance of colliding with another test that moves the same
        /// variable.
        fn release(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(home_var(), v),
                None => std::env::remove_var(home_var()),
            }
        }
    }

    impl Drop for HomeOverride {
        fn drop(&mut self) {
            if self.prev.is_some() {
                match self.prev.take() {
                    Some(v) => std::env::set_var(home_var(), v),
                    None => std::env::remove_var(home_var()),
                }
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// The session the run saves into, made under a sandboxed home and
    /// then released: the path is fixed, the home variable is not needed
    /// again.
    fn sandboxed_session(tag: &str) -> (Session, PathBuf) {
        let mut home = HomeOverride::new(tag);
        let session = Session::new(&provenance(), crate::config::HISTORY_FORMAT);
        home.release();
        let path = session.path().expect("a session path under the sandbox home").to_path_buf();
        (session, path)
    }

    // ---- the ordering ----------------------------------------------------

    /// Run one task against a mock model on a runtime of its own. The hub
    /// is spawned inside the runtime and shut down before it ends.
    fn run(
        port: u16,
        task: &str,
        ctx: &mut Context,
        convo: &mut Convo,
        sink: &dyn Show,
    ) -> crate::Result<()> {
        block_on(async {
            let (router, hub) = router();
            let res = run_task(task, &cfg(format!("http://127.0.0.1:{port}"), false),
                               ctx, convo, &router, &CancelToken::new(), sink).await;
            hub.shutdown().await;
            res
        })
    }

    #[test]
    fn the_task_is_on_disk_before_the_first_model_call() {
        let _lock = crate::test_util::env_lock();
        let (session, path) = sandboxed_session("ordering");
        let mut convo = Convo::persistent(session);
        // What the file held at the moment the request hit the wire.
        let on_disk = Arc::new(Mutex::new(String::new()));
        let seen = on_disk.clone();
        let (port, _reqs) = mock_seq_with(
            vec![(200, ANSWER.into())],
            move || {
                let body = std::fs::read_to_string(&path).unwrap_or_default();
                *seen.lock().unwrap() = body;
            },
        );
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        run(port, "count the files", &mut ctx, &mut convo, &sink).unwrap();
        let body = on_disk.lock().unwrap().clone();
        assert!(
            body.contains("count the files"),
            "the task was already on disk when the model was called — got: {body}"
        );
    }

    #[test]
    fn a_failed_save_warns_and_does_not_abort_the_run() {
        let _lock = crate::test_util::env_lock();
        // A *file* where the session directory wants to be: the save
        // cannot make its directory, so the save fails.
        let mut home = HomeOverride::new("save_fails");
        std::fs::write(home.dir.join(".jingwei"), b"not a directory").unwrap();
        let session = Session::new(&provenance(), crate::config::HISTORY_FORMAT);
        home.release();
        let mut convo = Convo::persistent(session);
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let res = run(port, "count the files", &mut ctx, &mut convo, &sink);
        assert!(res.is_ok(), "a failed save does not abort the run: {res:?}");
        let msgs = sink.msgs();
        assert!(
            msgs.iter().any(|m| matches!(m, Msg::Note { sev: Sev::Warn, text } if text.contains("session not saved"))),
            "the failed save was reported — got: {msgs:?}"
        );
        // the run still happened: the task and the answer are both in
        assert_eq!(ctx.history().len(), 2, "the run went on regardless: {:?}", ctx.history());
    }

    // ---- the bracket -----------------------------------------------------

    #[test]
    fn the_run_is_bracketed_by_a_task_begin_and_a_task_end() {
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let mut convo = Convo::ephemeral();
        run(port, "count the files", &mut ctx, &mut convo, &sink).unwrap();
        let msgs = sink.msgs();
        assert!(matches!(msgs.first(), Some(Msg::TaskBegin(t)) if t == "count the files"), "got: {msgs:?}");
        assert!(matches!(msgs.last(), Some(Msg::TaskEnd)), "got: {msgs:?}");
        assert_eq!(
            msgs.iter().filter(|m| matches!(m, Msg::TaskBegin(_))).count(),
            1,
            "exactly one run, one begin — got: {msgs:?}"
        );
    }

    #[test]
    fn a_failed_run_still_ends_the_run_and_hands_the_error_back() {
        let (port, _reqs) = mock_seq(vec![(400, REFUSED.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let mut convo = Convo::ephemeral();
        let res = run(port, "count the files", &mut ctx, &mut convo, &sink);
        // the error is handed back, not swallowed
        assert!(matches!(res, Err(crate::Error::Api(400, _))), "got: {res:?}");
        let msgs = sink.msgs();
        assert!(matches!(msgs.first(), Some(Msg::TaskBegin(_))), "got: {msgs:?}");
        assert!(matches!(msgs.last(), Some(Msg::TaskEnd)), "the run ends even when it fails — got: {msgs:?}");
    }

    // ---- the transcript --------------------------------------------------

    #[test]
    fn the_run_pushes_exactly_one_user_message() {
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let mut convo = Convo::ephemeral();
        run(port, "count the files", &mut ctx, &mut convo, &sink).unwrap();
        let history = ctx.history();
        assert_eq!(history.len(), 2, "the task and the answer — got: {history:?}");
        assert!(matches!(&history[0], Message::User(t) if t == "count the files"), "got: {history:?}");
        assert!(matches!(&history[1], Message::Assistant(_)), "got: {history:?}");
    }
}
