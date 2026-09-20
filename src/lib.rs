// jingwei — 精卫填海. A minimal coding agent: speak a task, and a small agent
// carries stones — one tool call at a time — until the sea is land.
//
// Wire protocols: the Messages wire (minimax — whose compatible endpoint is
// the recommended one, thinking blocks, interleaved reasoning, cache_control),
// the Chat Completions dialect (zai, deepseek — each with a thinking story of
// its own). Vendors that live at one address name it themselves; everything
// else you bring the endpoint for.
//
// Env: NO_COLOR (the only environment variable consulted; all other
//      settings — key, base URL, model, protocol, cache, thinking,
//      effort, frontend choice — go through CLI flags)

mod api;
mod cancel;
mod config;
mod core;
mod file_io;
mod format;
mod mcp;
mod tools;
/// The ui port (this module's `mod.rs`) plus the two frontends that
/// implement it. `pub` because the thin binary renders its final error
/// banner through the same vocabulary the frontends share.
pub mod ui;

#[cfg(test)] mod test_util;
use crate::cancel::CancelToken;
use crate::config::{Args, Config, build_config};
use crate::core::agent::agent_turn;
use crate::core::agents_md::AgentsMdContext;
use crate::core::context::Context;
use crate::core::ir::user_message;
use crate::core::session;
use crate::core::turn::Turn;
use crate::ui::{Msg, Sev, Show};
use crate::mcp::Hub;
use crate::config::settings::Settings;
use serde_json::Value;
use std::env;
use std::io;
use std::path::PathBuf;
pub type Result<T> = std::result::Result<T, Error>;

// ---- error -----------------------------------------------------------------

#[derive(Debug)]
pub enum Error {
    Api(u16, String),
    Msg(String),
    Http(Box<ureq::Error>),
    Json(serde_json::Error),
    Io(io::Error),
    /// The coroutine was cancelled (Ctrl-C). Not a failure — a stop.
    Interrupted,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Api(c, b) => write!(f, "api {c}: {b}"),
            Self::Msg(m) => write!(f, "{m}"),
            Self::Http(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Interrupted => write!(f, "interrupted"),
        }
    }
}
impl std::error::Error for Error {}

impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self {
        match e {
            ureq::Error::Status(c, r) => Self::Api(c, r.into_string().unwrap_or_default()),
            other => Self::Http(Box::new(other)),
        }
    }
}
impl From<serde_json::Error> for Error { fn from(e: serde_json::Error) -> Self { Self::Json(e) } }
impl From<io::Error> for Error { fn from(e: io::Error) -> Self { Self::Io(e) } }

impl Error {
    /// Is this worth sending again? A 429 or a 5xx, or a socket that
    /// never answered, is not the request's fault — and the answer is
    /// the same on every wire, so it lives on the error type. Pure, so
    /// a test reaches it.
    fn is_transient(&self) -> bool {
        match self {
            Error::Http(_) => true,
            Error::Api(s, _) => *s == 408 || *s == 429 || *s >= 500,
            _ => false,
        }
    }
}

// ---- entry -----------------------------------------------------------------

/// The library's single entrypoint. Everything process-shaped — the tokio
/// runtime, exit codes, the red error banner — lives in the thin `main.rs`
/// binary that calls this; everything agent-shaped lives here, where
/// tests and future frontends can reach it without spawning a process.
pub async fn run() -> Result<()> {
    // The argv grammar is config's; `parse_env_args` owns clap end to end
    // (including --help, answered and reported as `None`) so nothing above
    // config ever learns what clap looks like.
    let Some(mut args) = crate::config::parse_env_args()? else { return Ok(()) };
    // Settings is the layer between persisted preferences and vendor defaults.
    // We fold its active profile into `args` here so `build_config` (which
    // only reads args) sees a fully-resolved request. The chain is
    // CLI flag > settings.json active profile > vendor default.
    if let Some(active) = Settings::load().ok().and_then(|s| s.active_args()) {
        if args.api_key.is_none() { args.api_key = Some(active.api_key); }
        if args.base_url.is_none() { args.base_url = active.base_url; }
        if args.protocol.is_none() { args.protocol = Some(active.protocol); }
    }
    // --list answers from the disk alone — no credentials needed
    if args.list {
        return session::print_list(args.all);
    }
    let cfg = build_config(&args)?;
    // AGENTS.md: discover the closest one walking up from the workspace.
    // `--no-agents-md` short-circuits to an empty context so the rest of
    // the run is identical to "no AGENTS.md exists at all". The result
    // hands off to `Context::new`, which holds `system_text = SYSTEM`
    // and enqueues an `AgentsMdClosest` reminder when an AGENTS.md was
    // loaded; vendors consume both fields and never reach into
    // AGENTS.md or the tool registry themselves.
    let workspace = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let agents_ctx = if args.no_agents_md {
        AgentsMdContext::empty(&workspace)
    } else {
        AgentsMdContext::load(&workspace)
    };
    let ctx = Context::new(&agents_ctx);
    // MCP hub: external servers' tools, offered to the model as though
    // they were the agent's own. Spawned once per process and shared
    // across every turn (and every front-end). The user's servers are
    // read from `~/.jingwei/mcp.json`; the project's live in `<cwd>/.mcp.json`
    // and are picked up by the hub itself. A missing file is a None — the
    // hub starts empty and the session runs as it always did.
    let user_mcp = read_user_mcp();
    let hub = Hub::spawn(&workspace, user_mcp.as_ref());
    run_with_hub(args, cfg, ctx, hub, agents_ctx).await
}

/// The rest of `run` after the hub is spawned. The TUI and plain REPL take
/// the hub by value (it is `Clone` — handing a clone is a clone of an
/// `mpsc::Sender`); the one-shot path also takes a clone, then calls
/// `shutdown` on its own copy at the end so the connections close in
/// every exit path.
async fn run_with_hub(args: Args, cfg: Config, mut ctx: Context, hub: Hub, agents_ctx: AgentsMdContext) -> Result<()> {
    let interactive = args.prompt.is_empty();
    let (mut convo, banners) = begin_session(&args, &cfg, interactive)?;
    // AGENTS.md gets a banner slot too, when one was loaded. Prepended
    // so the user sees it before the session id — the context comes
    // first in attention order, and the banner says the same thing.
    let mut banners = banners;
    if let Some(b) = agents_ctx.banner() {
        banners.insert(0, b);
    }
    if interactive {
        // A terminal gets the TUI; pipes, tests, and one-shots get the log.
        return if ui::tui::wanted() {
            ui::tui::run(cfg, ctx, convo, banners, hub).await
        } else {
            ui::plain::plain_repl(&cfg, ctx, convo, banners, hub).await
        };
    }
    // One-shot: always the plain frontend — its output must stay in the
    // terminal after the process exits, and an alt-screen would take it.
    let prompt = args.prompt.join(" ");
    let sink = ui::plain::PlainSink::new();
    for b in &banners {
        sink.show(Msg::Banner(b.clone()));
    }
    sink.show(Msg::TaskBegin(prompt.clone()));
    convo.history.push(user_message(&prompt));
    if let Err(e) = convo.persist() {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    let token = CancelToken::new();
    let mut turn = Turn::new(0);
    let res = agent_turn(&cfg, &mut ctx, &mut convo.history, &mut turn, &token, &hub, &sink).await;
    if let Err(e) = convo.persist() {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    sink.show(Msg::TaskEnd);
    hub.shutdown().await;
    res
}

/// Read `~/.jingwei/mcp.json` if it exists. A missing or unreadable
/// file is a `None`: the hub starts empty, and the rest of the agent
/// runs as it always did. The hub's loader reaches for the file again
/// to print warnings; here we just answer "did the user write one?".
fn read_user_mcp() -> Option<Value> {
    let path = Settings::home_dir()?.join(".jingwei").join("mcp.json");
    let text = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!(" jingwei: ~/.jingwei/mcp.json: cannot be parsed ({e}); running without MCP ");
            None
        }
    }
}

/// The conversation this process runs on, and the banners the frontend
/// shows beside its own. Policy lives here, at the composition root;
/// session.rs owns the mechanism. A one-shot stays ephemeral; a resumed
/// one persists to the file it came from.
fn begin_session(args: &Args, cfg: &Config, interactive: bool) -> Result<(session::Convo, Vec<String>)> {
    if !args.resuming() {
        if !interactive {
            return Ok((session::Convo::ephemeral(), vec![]));
        }
        let prov = session::Provenance {
            model: cfg.model.clone(),
            protocol: cfg.protocol_label().into(),
            base_url: cfg.base_url.clone(),
        };
        let s = session::Session::new(&prov, crate::config::HISTORY_FORMAT);
        let banner = match s.path() {
            Some(p) => format!("session {} · the conversation persists to {}", s.id(), p.display()),
            None => format!("session {} · no home directory: this conversation stays in memory", s.id()),
        };
        return Ok((session::Convo::persistent(s, vec![]), vec![banner]));
    }
    let here = session::current_dir_string();
    // Scoping comes from the layout: the project subdirectory is the
    // filter, so `-c` (and a bare --resume) means "this project's newest"
    // and an explicit id resolves within this project too.
    let explicit = args.resume.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let (s, history) = session::Session::load(explicit)?;
    let mut banners = vec![format!(
        "resumed session {} · {} {} · started {}",
        s.id(),
        history.len(),
        session::msg_word(history.len()),
        session::utc(s.created())
    )];
    if s.model() != cfg.model {
        // a different model is a different request — and a prefix cache
        // that starts cold. The header says which model wrote the session.
        banners.push(format!("session was written by {} — now on {}", s.model(), cfg.model));
    }
    if !session::same_dir(s.cwd(), &here) {
        // the history speaks of files relative to another tree
        banners.push(format!(
            "session started in {} — tools now run from {}",
            session::dir_tail(s.cwd()),
            session::dir_tail(&here)
        ));
    }
    Ok((session::Convo::persistent(s, history), banners))
}


// ---- tests -----------------------------------------------------------------

/// Cargo runs a crate's tests in parallel threads of one process, and
/// `std::env` is process-global — one test's `NO_COLOR` is another test's
/// surprise. Every test that reads or writes an environment variable holds
/// this lock, so the env-touching set runs one at a time. Test-only: the
/// binary never touches it.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        assert_eq!(Error::Api(429, "limited".into()).to_string(), "api 429: limited");
        assert_eq!(Error::Msg("bad".into()).to_string(), "bad");
        assert_eq!(Error::Interrupted.to_string(), "interrupted");
    }

    // ---- retry: the transport's own rule ----
    #[test]
    fn only_transient_errors_are_resent() {
        // the retryable kinds: a rate limit, a timeout, a server that is down
        for code in [408, 429, 500, 502, 503, 504] {
            assert!(Error::Api(code, "x".into()).is_transient(), "{code} should be resent");
        }
        // the request's own fault, or a stop: never resent
        for code in [400, 401, 403, 404, 422] {
            assert!(!Error::Api(code, "x".into()).is_transient(), "{code} must not be resent");
        }
        assert!(!Error::Msg("x".into()).is_transient());
        assert!(!Error::Interrupted.is_transient(), "a cancellation is not a failure to retry");
    }

    #[test]
    // ---- Error: the type's display & conversions -------------------------

    #[test]
    fn error_display_strings_are_actually_distinct() {
        // each variant has its own Display shape — the test pins them
        // so a refactor can't silently turn two variants into the same
        // string (which would defeat the user's ability to tell what
        // kind of failure they're looking at).
        assert_eq!(format!("{}", Error::Interrupted), "interrupted");
        // The passthrough variants delegate to their inner Display; the
        // outer prefix is the variant's own word — we just check the
        // inner value shows up verbatim, no "error: " or similar wrapper
        // added by Error's Display.
        let io = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        assert_eq!(format!("{}", Error::Io(io)), "denied");
        // Error::Json's body is the inner error verbatim
        let json_err = serde_json::from_str::<u32>("abc").unwrap_err();
        let json_str = format!("{}", json_err);
        assert_eq!(format!("{}", Error::Json(json_err)), json_str);
        // Error::Msg shows the wrapped message verbatim
        assert_eq!(format!("{}", Error::Msg("hi".into())), "hi");
        // Error::Api prefix error shape: "api {code}: {body}"
        assert_eq!(format!("{}", Error::Api(503, "unavailable".into())),
            "api 503: unavailable");
    }

    #[test]
    fn error_from_ureq_wraps_a_status_into_api() {
        // the Status arm of From<ureq::Error>: the status code and body
        // ride along as the Error::Api variant
        let body = r#"{"error":"model gone"}"#;
        let r = ureq::Response::new(404, "Not Found", body).unwrap();
        let err: Error = ureq::Error::Status(404, r).into();
        match err {
            Error::Api(code, ref b) => {
                assert_eq!(code, 404);
                assert!(b.contains("model gone"), "body carries: {b}");
            }
            other => panic!("expected Error::Api, got {other:?}"),
        }
    }

    #[test]
    fn error_from_ureq_wraps_other_kinds_into_http() {
        // transport / DNS / decode errors: they ride as Error::Http.
        // Transport cannot be constructed outside ureq, so we trigger a
        // real one — a request to a closed port.
        let err: Error = ureq::get("http://127.0.0.1:1/nope").call().unwrap_err().into();
        assert!(matches!(err, Error::Http(_)), "got {err:?}");
    }

    #[test]
    fn error_from_io_preserves_the_io_kind() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope");
        let err: Error = io.into();
        assert!(matches!(err, Error::Io(_)));
    }

    }
