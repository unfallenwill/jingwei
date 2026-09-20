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
mod login;
mod mcp;
mod settings;
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
use crate::settings::Settings;
use clap::{Parser, Subcommand};
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
    // parse() exits the process on error; we want our own error path so
    // the binary's stderr formatting (the red " jingwei: ... " banner)
    // matches what every other code path uses.
    let cli = Cli::try_parse_from(env::args()).map_err(|e| Error::Msg(short_clap_err(&e)))?;
    if cli.help {
        crate::config::print_help();
        return Ok(());
    }
    let mut args = args_from_cli(cli)?;
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

// ---- cli -------------------------------------------------------------------

/// The argv grammar: top-level subcommands for `login` / `list` / `resume`,
/// and a flat flag set for the default invocation (one-shot or REPL).
/// clap does the parsing; `args_from_cli` flattens it back into the
/// internal [`Args`] shape so the rest of the binary never learns
/// what clap looks like.
#[derive(Parser, Debug)]
#[command(name = "jingwei", disable_help_flag = true, disable_version_flag = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Cmd>,

    // connection
    #[arg(long)] api_key: Option<String>,
    #[arg(long)] base_url: Option<String>,
    #[arg(short = 'm', long)] model: Option<String>,
    #[arg(long)] protocol: Option<String>,

    // behavior
    #[arg(long)] cache: Option<String>,
    #[arg(long)] thinking: Option<String>,
    #[arg(long)] effort: Option<String>,
    #[arg(long)] max_tokens: Option<String>,
    #[arg(long)] context_size: Option<String>,
    #[arg(long)] max_turns: Option<String>,
    #[arg(short = 's', long)] stream: bool,
    #[arg(short = 'S', long)] no_stream: bool,

    // sessions
    #[arg(short = 'c', long)] r#continue: bool,
    /// Resume a session by id prefix. Bare `--resume` is like `-c`.
    /// clap's `Option<Option<String>>` idiom: outer `Some` means "flag
    /// present", inner `Some` means "value given".
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    resume: Option<Option<String>>,
    /// List this project's saved sessions.
    #[arg(long)] list: bool,
    /// With `--list`, list every project's sessions.
    #[arg(long)] all: bool,

    /// Skip AGENTS.md discovery entirely. The agent runs with no
    /// project conventions loaded — for paranoid sessions, or when
    /// the repo's AGENTS.md is known to be stale or misleading.
    #[arg(long)] no_agents_md: bool,

    // prompt — everything else, captured verbatim. clap's
    // `trailing_var_arg` lets the prompt begin after `--`, so users
    // who want to send `--help` as a literal task can still do so.
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    // -h / --help: clap's auto-generated help is disabled in favour
    // of the hand-written one in config::HELP (lists every vendor by
    // name and reads the same to a first-time user as the README).
    #[arg(short = 'h', long, action = clap::ArgAction::SetTrue)]
    help: bool,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Set up providers / api keys / active model.
    Login {
        #[arg(long)] show: bool,
        #[arg(long)] reset: bool,
        #[arg(long)] reveal: bool,
    },
    /// List sessions for this project.
    List {
        #[arg(long)] all: bool,
    },
    /// Resume a session by id prefix; bare `resume` is like `-c`.
    Resume {
        id: Option<String>,
    },
}

/// Translate [`Cli`] into the internal [`Args`] shape. Subcommands are
/// mapped onto `Args.resume` / `Args.cont` / `Args.list`; numeric flags
/// stay as strings here and parse later. Bare `--resume` and
/// `--resume <id>` both funnel through the same path.
fn args_from_cli(c: Cli) -> Result<Args> {
    let mut a = Args {
        streaming: true,
        ..Default::default()
    };
    if c.command.is_some() { a.list = false; } // subcommands carry their own intent
    match c.command {
        Some(Cmd::Login { show, reset, reveal }) => {
            crate::login::run(show, reset, reveal)?;
            std::process::exit(0);
        }
        Some(Cmd::List { all }) => {
            a.list = true;
            a.all = all;
            return Ok(a);
        }
        Some(Cmd::Resume { id }) => resume_into(&mut a, id),
        None => {
            // the default invocation — copy flags into Args
            a.api_key = c.api_key;
            a.base_url = c.base_url;
            a.model = c.model;
            a.protocol = c.protocol;
            a.cache = c.cache;
            a.thinking = c.thinking;
            a.effort = c.effort;
            // numeric flags stay as strings here so a parse error gets the
            // same message text build_config would produce later
            a.max_tokens = parse_num_flag("--max-tokens", c.max_tokens)?;
            a.context_size = parse_num_flag("--context-size", c.context_size)?;
            a.max_turns = parse_num_flag("--max-turns", c.max_turns)?;
            if c.no_stream { a.streaming = false; }
            if let Some(r) = c.resume {
                resume_into(&mut a, r);
            }
            a.cont |= c.r#continue;
            a.list = c.list;
            a.all = c.all;
            a.prompt = c.prompt;
            a.no_agents_md = c.no_agents_md;
        }
    }
    Ok(a)
}

/// Both the subcommand arm (`resume <id>`) and the flag arm (`--resume [id]`)
/// spell the same rule: a non-empty value is the id, anything else is
/// "the newest session" (`cont`).
fn resume_into(a: &mut Args, id: Option<String>) {
    match id {
        Some(s) if !s.is_empty() => a.resume = Some(s),
        _ => a.cont = true,
    }
}

/// Parse a numeric flag, naming the flag in the error so the user can
/// tell which one rejected its value.
fn parse_num_flag<T: std::str::FromStr>(flag: &str, raw: Option<String>) -> Result<Option<T>>
where T::Err: std::fmt::Display {
    match raw {
        None => Ok(None),
        Some(s) => s.parse::<T>().map(Some).map_err(|_| Error::Msg(format!("{flag} expects a number"))),
    }
}

/// Test-only entrypoint: parse an argv slice (no leading program name)
/// and yield the internal [`Args`] the same way `run()` does. Used by
/// `config::tests` to exercise the resolver without spinning up clap
/// from inside `config.rs`.
#[cfg(test)]
pub(crate) fn parse_argv(argv: &[&str]) -> Result<Args> {
    let mut full = vec!["jingwei".to_string()];
    full.extend(argv.iter().cloned().map(str::to_string));
    // `try_parse_from` returns `Err(clap::Error)`. We pull a short
    // summary from it so the message that ends up in `Error::Msg`
    // names the bad token when clap can.
    let cli = Cli::try_parse_from(full)
        .map_err(|e| Error::Msg(short_clap_err(&e)))?;
    args_from_cli(cli)
}

/// Reduce a clap error to a single line that mentions the offending
/// argument when clap can identify one. Long help blocks the user
/// cannot easily scroll past.
fn short_clap_err(e: &clap::Error) -> String {
    use clap::error::{ContextKind, ContextValue};
    let mut bad = None;
    for (kind, val) in e.context() {
        if matches!(kind, ContextKind::InvalidArg) { if let ContextValue::String(s) = val { bad = Some(s.clone()); } }
    }
    match (e.kind(), bad) {
        (clap::error::ErrorKind::UnknownArgument, Some(a)) => format!("unknown flag: {a}\ntry --help"),
        (_, Some(a)) => format!("invalid argument {a}: {e}"),
        (_, None) => e.to_string(),
    }
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
    #![allow(unused_imports)]
    use super::*;
    use crate::api::{Protocol, CacheMode, Thinking};
    use crate::config::{Args, DEFAULT_CONTEXT_SIZE, DEFAULT_MAX_TURNS};

    // ---- flags & config ----

    fn flags(list: &[&str]) -> Result<Args> { crate::parse_argv(list) }

    #[test]
    fn flags_parse_and_validate() {
        let a = flags(&["--base-url", "https://x/v1", "--api-key", "k", "-m", "m1",
            "--protocol", "zai", "--cache", "auto", "--thinking", "strip",
            "--max-tokens", "4096", "--context-size", "100000", "--max-turns", "5",
            "-S", "do", "stuff"]).unwrap();
        assert_eq!(a.prompt, vec!["do", "stuff"]);
        assert!(!a.streaming);
        let cfg = build_config(&a).unwrap();
        assert_eq!(cfg.protocol, Protocol::ZAI);
        assert_eq!(cfg.cache, CacheMode::Auto);
        assert_eq!(cfg.thinking, Thinking::Strip);
        assert_eq!(cfg.max_tokens, 4096);
        assert_eq!(cfg.context_size, 100000);
        assert_eq!(cfg.max_turns, 5);
        // --max-turns defaults and rejects 0 / garbage
        let default = build_config(&flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m"]).unwrap()).unwrap();
        assert_eq!(default.max_turns, crate::config::DEFAULT_MAX_TURNS);
        assert_eq!(default.max_tokens, crate::config::DEFAULT_MAX_TOKENS);
        assert_eq!(default.context_size, crate::config::DEFAULT_CONTEXT_SIZE);
        assert!(build_config(&flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--max-turns", "0"]).unwrap()).is_err());
        // active cache belongs to the Messages wire — the chat-completions
        // vendors reject it in their own words
        assert!(build_config(&flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--protocol", "zai", "--cache", "active"]).unwrap()).is_err());
        // required fields — and the one vendor that names its own endpoint
        // and flagship: a key alone is a whole config
        assert!(build_config(&flags(&[]).unwrap()).is_err());
        let bare = build_config(&flags(&["--api-key", "k"]).unwrap()).unwrap();
        assert_eq!((bare.protocol_label(), bare.base_url.as_str(), bare.model.as_str()),
            ("minimax", "https://api.minimax.cn/anthropic", "MiniMax-M3"));
        // every vendor names its endpoint now, but deepseek leaves the
        // model to the user — flash or pro is a choice, not a default
        assert!(build_config(&flags(&["--api-key", "k", "--protocol", "deepseek"]).unwrap()).is_err());
        // bad flags
        assert!(flags(&["--nope"]).is_err());
        assert!(flags(&["--max-tokens", "abc"]).is_err());
        assert!(flags(&["--max-turns", "abc"]).is_err());
        assert!(flags(&["--base-url"]).is_err());
        assert!(flags(&["--context-size"]).is_err());
        assert!(flags(&["-s"]).unwrap().streaming); // -s is the explicit default
    }

    #[test]
    fn session_flags_parse_with_optional_resume_value() {
        // -c and a bare --resume both mean "the newest session"
        assert!(flags(&["-c"]).unwrap().resuming());
        assert!(flags(&["--continue"]).unwrap().resuming());
        assert!(flags(&["--resume"]).unwrap().resuming());
        assert_eq!(flags(&["--resume"]).unwrap().resume, None);
        // a value is an id-prefix selector
        assert_eq!(flags(&["--resume", "20250916"]).unwrap().resume.as_deref(), Some("20250916"));
        // the optional value never swallows a flag that follows it
        let a = flags(&["--resume", "--model", "m"]).unwrap();
        assert_eq!(a.resume, None);
        assert_eq!(a.model.as_deref(), Some("m"));
        // --list is a mode of its own
        assert!(flags(&["--list"]).unwrap().list);
        assert!(!flags(&["--list"]).unwrap().resuming());
        assert!(!flags(&["task", "words"]).unwrap().resuming(), "a task alone resumes nothing");
    }

    #[test]
    fn enum_flags_case_insensitive_and_list_available_values() {
        let parse = |extra: &[&str]| {
            let mut argv: Vec<&str> = vec!["--api-key", "k", "--base-url", "https://x", "-m", "m"];
            argv.extend_from_slice(extra);
            crate::parse_argv(&argv).unwrap()
        };
        // spellings are case-insensitive in every position
        assert_eq!(build_config(&parse(&["--protocol", "ZAI"])).unwrap().protocol, Protocol::ZAI);
        let cfg = build_config(&parse(&["--cache", "Active", "--thinking", "STRIP"])).unwrap();
        assert_eq!(cfg.cache, CacheMode::Active);
        assert_eq!(cfg.thinking, Thinking::Strip);
        // an unknown value names the flag and lists what is accepted
        let err = match build_config(&parse(&["--cache", "turbo"])) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("--cache turbo should be rejected"),
        };
        assert!(err.contains("invalid cache 'turbo'"), "got: {err}");
        assert!(err.contains("auto, active"), "got: {err}");
    }

    // ---- zai vendor: dialect conversion ----

    // ---- api over local mocks ----

    // ---- deepseek vendor ----

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
