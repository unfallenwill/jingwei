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
//! - the agent the run works is **registered while it runs** and gone when
//!   it ends, so a stop request that reaches the roster finds it;
//! - a failed save **warns and does not abort**: the conversation lives
//!   in memory either way, and the agent must not die because the disk
//!   did.
//!
//! What it does *not* do is decide what an error looks like. The
//! frontend owns that: it hands the boundary a mapper — an error in,
//! the message to say about it (or nothing at all) out — and the
//! boundary calls it **inside** the run's bracket, so the note about a
//! failed run belongs to the run it is about. The mapper's result is
//! still the caller's to interpret: the plain log and the TUI show a
//! note, the one-shot propagates the error so the process exit code
//! stays right. Presentation belongs to the frontend, and the boundary
//! only decides *when* it is said.

use crate::agent::roster::Roster;
use crate::agent::runner::run_agent;
use crate::agent::Agent;
use crate::cancel::CancelScope;
use crate::config::Config;
use crate::core::context::Context;
use crate::core::ir::user_message;
use crate::core::ports::ToolRouter;
use crate::core::session::Convo;
use crate::ui::{Msg, Sev, Show};

/// Run one task, from the user's words to the end of the run.
///
/// The order is the contract: show the task, put it in the transcript,
/// put the transcript on disk, run the agent, say what went wrong (if
/// anything), put the transcript on disk again, end the run. The saves
/// are best-effort — a failure warns and the run continues — and the
/// run is bracketed by exactly one `TaskBegin` and one `TaskEnd`
/// whatever happens in between.
///
/// `on_error` is the frontend's own answer to "what does a failure
/// look like": it takes the run's error and returns the message to say
/// about it, or nothing to say. The boundary calls it before it closes
/// the run, so the note lands inside the bracket. The `Result` is the
/// run's own, unchanged: the caller still decides what an error means.
///
/// `scope` is the run's cancel scope: the run mints its leaf from it, so
/// a cancel of the scope stops the run and everything under it — and a run
/// that starts under an already-cancelled scope does no work.
///
/// `roster` is the registry of live agents: the run registers the agent
/// it runs and lets it go when the run ends, so a stop request that
/// reaches the roster finds the agent it is meant to stop.
pub(crate) async fn run_task(
    task: &str,
    cfg: &Config,
    ctx: &mut Context,
    convo: &mut Convo,
    router: &ToolRouter,
    scope: &CancelScope,
    roster: &Roster,
    sink: &dyn Show,
    on_error: &(dyn Fn(&crate::Error) -> Option<Msg> + Sync),
) -> crate::Result<()> {
    sink.show(Msg::TaskBegin(task.to_string()));
    ctx.push(user_message(task));
    // the task is on disk before the first stone moves
    if let Err(e) = convo.persist(ctx.history()) {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    let mut agent = Agent::main(task);
    let res = run_agent(&mut agent, cfg, ctx, router, scope, roster, sink).await;
    // the failure is said inside the run: a note after the run's end
    // would belong to nothing
    if let Err(e) = &res {
        if let Some(m) = on_error(e) {
            sink.show(m);
        }
    }
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
    use crate::test_util::{block_on, cfg, mock_seq, mock_seq_with, Recorder};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    /// A model that answers once and stops.
    const ANSWER: &str =
        r#"{"content":[{"type":"text","text":"three files"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
    /// A model that refuses.
    const REFUSED: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"no"}}"#;

    /// A mapper that says nothing: how a failure looks is the frontend's
    /// business, and these tests are about the run, not its face.
    fn silent(_e: &crate::Error) -> Option<Msg> { None }

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
    /// guard that puts it back. The guard owns the dir as well: the
    /// session's path points into it, so the dir has to outlive the
    /// writes that use that path and be removed once they are done. A
    /// test that takes the guard also takes the env lock, so a sibling
    /// test that moves the same variable at the same time cannot see
    /// the wrong home.
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
    }

    impl Drop for HomeOverride {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(home_var(), v),
                None => std::env::remove_var(home_var()),
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// The session the run saves into, made under a sandboxed home. The
    /// guard comes back with it: the session's path points into the
    /// sandbox, so the sandbox has to stay until the test is done
    /// writing through that path — and not a moment longer.
    fn sandboxed_session(tag: &str) -> (Session, PathBuf, HomeOverride) {
        let home = HomeOverride::new(tag);
        let session = Session::new(&provenance(), crate::config::HISTORY_FORMAT);
        let path = session.path().expect("a session path under the sandbox home").to_path_buf();
        (session, path, home)
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
        on_error: &(dyn Fn(&crate::Error) -> Option<Msg> + Sync),
    ) -> crate::Result<()> {
        block_on(async {
            let (router, hub) = router();
            let res = run_task(task, &cfg(format!("http://127.0.0.1:{port}"), false),
                               ctx, convo, &router, &CancelScope::new(), &Roster::new(), sink, on_error).await;
            hub.shutdown().await;
            res
        })
    }

    #[test]
    fn the_task_is_on_disk_before_the_first_model_call() {
        let _lock = crate::test_util::env_lock();
        let (session, path, _home) = sandboxed_session("ordering");
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
        run(port, "count the files", &mut ctx, &mut convo, &sink, &silent).unwrap();
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
        let home = HomeOverride::new("save_fails");
        std::fs::write(home.dir.join(".jingwei"), b"not a directory").unwrap();
        let session = Session::new(&provenance(), crate::config::HISTORY_FORMAT);
        let mut convo = Convo::persistent(session);
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let res = run(port, "count the files", &mut ctx, &mut convo, &sink, &silent);
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
        run(port, "count the files", &mut ctx, &mut convo, &sink, &silent).unwrap();
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
        let res = run(port, "count the files", &mut ctx, &mut convo, &sink, &silent);
        // the error is handed back, not swallowed
        assert!(matches!(res, Err(crate::Error::Api(400, _))), "got: {res:?}");
        let msgs = sink.msgs();
        assert!(matches!(msgs.first(), Some(Msg::TaskBegin(_))), "got: {msgs:?}");
        assert!(matches!(msgs.last(), Some(Msg::TaskEnd)), "the run ends even when it fails — got: {msgs:?}");
    }

    #[test]
    fn a_failed_run_says_so_inside_the_bracket() {
        // The note about a failed run belongs to the run it is about: it
        // has to land between the begin and the end, not after them.
        // Before the boundary owned the note, the frontend showed it
        // *after* the run had ended, and the error read as a footnote
        // to nothing.
        let (port, _reqs) = mock_seq(vec![(400, REFUSED.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let mut convo = Convo::ephemeral();
        // the frontend's own mapper: the two interactive frontends say a
        // failed run this way
        let res = run(port, "count the files", &mut ctx, &mut convo, &sink, &crate::ui::error_note);
        assert!(res.is_err(), "the run failed: {res:?}");
        let msgs = sink.msgs();
        let note = msgs.iter().position(|m| matches!(m, Msg::Note { sev: Sev::Err, .. }));
        let end = msgs.iter().position(|m| matches!(m, Msg::TaskEnd));
        assert!(note.is_some(), "the failure was said — got: {msgs:?}");
        assert!(end.is_some(), "the run ended — got: {msgs:?}");
        assert!(note < end, "the note lands inside the run's bracket — got: {msgs:?}");
    }

    // ---- the transcript --------------------------------------------------

    #[test]
    fn the_run_pushes_exactly_one_user_message() {
        let (port, _reqs) = mock_seq(vec![(200, ANSWER.into())]);
        let sink = Recorder::default();
        let mut ctx = crate::test_util::ctx();
        let mut convo = Convo::ephemeral();
        run(port, "count the files", &mut ctx, &mut convo, &sink, &silent).unwrap();
        let history = ctx.history();
        assert_eq!(history.len(), 2, "the task and the answer — got: {history:?}");
        assert!(matches!(&history[0], Message::User(t) if t == "count the files"), "got: {history:?}");
        assert!(matches!(&history[1], Message::Assistant(_)), "got: {history:?}");
    }
}
