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
mod display;
mod edit;
mod file_io;
mod format;
mod ir;
mod ledger;
mod login;
mod mcp;
mod plain;
mod session;
mod settings;
mod tool_runtime;
mod tools;
mod tui;
mod turn;

#[cfg(test)] mod test_util;
use crate::api::call_api as api_call_api;
use crate::cancel::CancelToken;
use crate::config::{Args, Config, build_config};
use crate::display::{Msg, Sev, Show};
use crate::ir::{Block, Message, ToolResult};
use crate::mcp::Hub;
use crate::settings::Settings;
use crate::tool_runtime::run_tool;
use crate::tools::{print_tool_call, tools};
use crate::turn::{Turn, TurnOutcome, TurnState};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::env;
use std::io;
use std::path::PathBuf;
use std::time::Duration;
const MAX_TOOL_OUTPUT: usize = 50_000;
type Result<T> = std::result::Result<T, Error>;

// ---- error -----------------------------------------------------------------

#[derive(Debug)]
enum Error {
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
    /// Is this worth sending again? A 429 or a 5xx, or a socket that never
    /// answered, is not the request's fault — and the answer is the same on
    /// every wire, so it lives on the error type rather than at the call
    /// sites. Pure, so a test reaches it.
    fn is_transient(&self) -> bool {
        match self {
            Error::Http(_) => true,                                  // connect / timeout / DNS
            Error::Api(s, _) => *s == 408 || *s == 429 || *s >= 500, // the retryable statuses
            _ => false,                                              // Msg / Json / Io / Interrupted
        }
    }
}

// ---- entry -----------------------------------------------------------------

fn main() {
    // The agent runs as a coroutine on a single-threaded cooperative
    // scheduler: coroutines take turns at their suspension points, and
    // blocking work (HTTP, file tools) is farmed out to the blocking pool so
    // those suspension points stay real. Cancellation — Ctrl-C — rides the
    // same machinery: see `CancelToken` and `agent_turn`.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("runtime init: {e}"));
    let result = rt.block_on(run());
    // Abandoned blocking workers (an interrupted HTTP call still inside its
    // read timeout, say) must not stall shutdown — give them one second to
    // wind down, then drop the runtime regardless.
    rt.shutdown_timeout(Duration::from_secs(1));
    match result {
        Ok(()) => {}
        // interrupted by Ctrl-C — exit the way a shell command would
        Err(Error::Interrupted) => std::process::exit(130),
        Err(e) => {
            eprintln!("{}", display::paint(&format!(" jingwei: {e} "), display::ERR_BG, true));
            std::process::exit(1);
        }
    }
}

const SYSTEM: &str = "You are jingwei (精卫), a coding agent with these tools: \
    bash (run a shell command), read_file, write_file, edit_file (search/replace; exact match with line-aligned indentation fallback). \
    Pick the smallest tool that fits: read_file to inspect (not cat via bash); edit_file for a few lines; write_file only to create or wholesale-replace; bash for the rest — running tests, git, grep, ls. \
    Read a file before you edit it — never guess its contents. \
    Prefer small, targeted commands. \
    Never start a command that waits for input or never returns (editors, pagers, watch or server modes) — everything you run must terminate. \
    Never run destructive commands unless the user explicitly asks. \
    Never run git commit, push, reset, checkout, clean, or rebase unless the user explicitly asks. \
    Write for a terminal, not a Markdown renderer: plain text, no headings/tables/bold markers, short lines, with exact copy-pasteable paths and commands. And since your reasoning is folded away and unseen, put every conclusion in the visible answer. \
    Don't stop until the task is actually delivered: make the change, then verify it by running the build, the tests, or the exact command, fix what fails, and repeat until it passes — only then summarize. If you cannot finish, say exactly what is done, what is left, and what blocked you. \
    When asked your name, say you are jingwei (精卫). \
    When finished, reply with a concise 1-3 sentence summary of what you did.";

// ---- cli -------------------------------------------------------------------

/// The argv grammar: top-level subcommands for `login` / `list` / `resume`,
/// and a flat flag set for the default invocation (one-shot or REPL).
/// clap does the parsing; `args_from_cli` flattens it back into the
/// internal [`Args`] shape so the rest of the binary — `build_config`,
/// `begin_session` — never has to learn what clap looks like.
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

    // prompt — everything else, captured verbatim. clap's
    // `trailing_var_arg` lets the prompt begin after `--`, so users
    // who want to send `--help` as a literal task can still do so
    // (jingwei "show me -- --help" — the `--` ends flag parsing).
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    // -h / --help — clap's auto-generated help is disabled in favour
    // of the hand-written one in config::HELP (which lists every vendor
    // by name and reads the same to a first-time user as the README).
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
/// stay as strings here and parse later (the resolver knows what to do
/// with the error text). Bare `--resume` and `--resume <id>` both
/// funnel through the same path.
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
        Some(Cmd::Resume { id }) => {
            match id {
                Some(s) if !s.is_empty() => a.resume = Some(s),
                _ => a.cont = true,
            }
        }
        None => {
            // the default invocation — copy flags into Args
            a.api_key = c.api_key;
            a.base_url = c.base_url;
            a.model = c.model;
            a.protocol = c.protocol;
            a.cache = c.cache;
            a.thinking = c.thinking;
            a.effort = c.effort;
            // numeric strings — leave as String, build_config parses them
            // (we keep them as String here so a parse error gets the same
            //  message text the old parse_from produced).
            a.max_tokens = parse_num_flag("--max-tokens", c.max_tokens)?;
            a.context_size = parse_num_flag("--context-size", c.context_size)?;
            a.max_turns = parse_num_flag("--max-turns", c.max_turns)?;
            if c.no_stream { a.streaming = false; }
            // --resume value handling: bare (Some(None)) → cont; with value → resume
            if let Some(r) = c.resume {
                match r {
                    Some(s) if !s.is_empty() => a.resume = Some(s),
                    _ => a.cont = true,
                }
            }
            a.cont |= c.r#continue;
            a.list = c.list;
            a.all = c.all;
            a.prompt = c.prompt;
        }
    }
    Ok(a)
}

/// Parse a numeric flag, naming the flag in the error so the user can
/// tell which one rejected its value.
fn parse_num_flag<T>(flag: &str, raw: Option<String>) -> Result<Option<T>>
where T: std::str::FromStr, T::Err: std::fmt::Display {
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
    use clap::error::ContextKind;
    use clap::error::ContextValue;
    let mut bad: Option<String> = None;
    for (kind, val) in e.context() {
        if matches!(kind, ContextKind::InvalidArg) {
            if let ContextValue::String(s) = val {
                bad = Some(s.clone());
            }
        }
    }
    match (e.kind(), bad) {
        (clap::error::ErrorKind::UnknownArgument, Some(a)) => format!("unknown flag: {a}\ntry --help"),
        (_, Some(a)) => format!("invalid argument {a}: {e}"),
        (_, None) => e.to_string(),
    }
}

async fn run() -> Result<()> {
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
    // MCP hub: external servers' tools, offered to the model as though
    // they were the agent's own. Spawned once per process and shared
    // across every turn (and every front-end). The user's servers are
    // read from `~/.jingwei/mcp.json`; the project's live in `<cwd>/.mcp.json`
    // and are picked up by the hub itself. A missing file is a None — the
    // hub starts empty and the session runs as it always did.
    let user_mcp = read_user_mcp();
    let workspace = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let hub = Hub::spawn(&workspace, user_mcp.as_ref());
    run_with_hub(args, cfg, hub).await
}

/// The rest of `run` after the hub is spawned. The TUI and plain REPL take
/// the hub by value (it is `Clone` — handing a clone is a clone of an
/// `mpsc::Sender`); the one-shot path also takes a clone, then calls
/// `shutdown` on its own copy at the end so the connections close in
/// every exit path.
async fn run_with_hub(args: Args, cfg: Config, hub: Hub) -> Result<()> {
    let interactive = args.prompt.is_empty();
    let (mut convo, banners) = begin_session(&args, &cfg, interactive)?;
    if interactive {
        // A terminal gets the TUI; pipes, tests, and one-shots get the log.
        return if tui::wanted() {
            tui::run(cfg, convo, banners, hub).await
        } else {
            plain::plain_repl(&cfg, convo, banners, hub).await
        };
    }
    // One-shot: always the plain frontend — its output must stay in the
    // terminal after the process exits, and an alt-screen would take it.
    let prompt = args.prompt.join(" ");
    let sink = plain::PlainSink::new();
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
    let res = agent_turn(&cfg, &mut convo.history, &mut turn, &token, &hub, &sink).await;
    if let Err(e) = convo.persist() {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    sink.show(Msg::TaskEnd);
    hub.shutdown().await;
    res
}

/// Read `~/.jingwei/mcp.json` if it exists. A missing or unreadable file is
/// a `None`: the hub starts empty, and the rest of the agent runs as it
/// always did. The hub's own loader is what reaches for the file again to
/// print warnings about it; here we just answer "did the user write one?".
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

/// The conversation this process runs on, and the banner lines that say
/// which (the frontend shows them beside its own). Policy lives here, at
/// the composition root — mechanism is session.rs's, and the agent core
/// knows none of it. A one-shot stays ephemeral; a resumed one persists,
/// to the file it came from.
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
        // that starts cold; the header says which model wrote the session
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

/// The one constructor of a user task message. The internal history shape
/// belongs to the agent core — so its constructor does too, and no
/// frontend spells the shape by hand.
fn user_message(text: &str) -> Message {
    Message::User(text.to_string())
}

/// Drive one agent run as an interruptible coroutine.
///
/// The first Ctrl-C cancels the token: the agent coroutine notices at its
/// next suspension point (a streamed line, a tool poll, a turn boundary),
/// tidies the history so every tool call stays paired, and unwinds — the
/// REPL prompt comes back with everything the agent already carried still in
/// place. A second Ctrl-C while it is unwinding exits immediately, for when
/// even graceful is too slow.
///
/// `turn` is the per-run state machine: the caller hands in a fresh
/// `Turn` (in `Queued`) and the wrapper drives it through the table via
/// [`agent_loop`]. The caller can inspect `turn` after the wrapper
/// returns to read the terminal state — the wrapper exists to own the
/// cancel select (`tokio::select!` against Ctrl-C) and the double-signal
/// semantics, not to hide the turn from anyone.
async fn agent_turn(cfg: &crate::config::Config, history: &mut Vec<Message>, turn: &mut Turn, token: &crate::cancel::CancelToken, hub: &Hub, sink: &dyn Show) -> crate::Result<()> {
    let agent = agent_loop(cfg, history, turn, token, hub, sink);
    tokio::pin!(agent);
    tokio::select! {
        res = &mut agent => return res,
        // The plain frontend has no key loop of its own, so Ctrl-C arrives
        // as a signal here; the TUI cancels through its key handler instead.
        _ = tokio::signal::ctrl_c() => token.cancel(),
    }
    let res = tokio::select! {
        res = &mut agent => res,
        _ = tokio::signal::ctrl_c() => {
            sink.show(Msg::Note { sev: Sev::Err, text: " interrupted again — exiting ".into() });
            std::process::exit(130);
        }
    };
    if let Err(Error::Interrupted) = &res {
        sink.show(Msg::Note { sev: Sev::Warn, text: " interrupted — stopped at a safe point; history kept ".into() });
    }
    res
}



// ---- agent loop ------------------------------------------------------------

/// The agent itself, as a coroutine. Every await is a suspension point and a
/// cancellation point: between turns, on every streamed line, while each tool
/// runs. On cancellation it returns `Err(Interrupted)` after leaving the
/// history valid — interrupted tools get results, unfinished tool calls are
/// dropped from the turn, and the REPL prompt simply returns.
///
/// `turn` is the per-run state machine the wrapper owns; this function
/// drives it through the table — `Queued → InProgress` at the entry, and
/// the terminal transition on the way out. The mechanics of "do API calls
/// and run tools" live in [`run_turn`], kept separate so the mapping
/// between [`Result`] and [`TurnState`] has exactly one home (caller
/// translates to [`TurnOutcome`], [`Turn::finish`] maps that onto a
/// state).
///
/// `hub` is the MCP hub — its `definitions()` is the source of the schemas
/// the model sees, and its `call()` is what runs when the model asks for a
/// tool whose name starts with `mcp__`. Re-fetching the definitions every
/// turn is what lets `/mcp enable|disable|reconnect` take effect mid-session
/// without restarting.
async fn agent_loop(cfg: &crate::config::Config, history: &mut Vec<Message>, turn: &mut Turn, token: &crate::cancel::CancelToken, hub: &Hub, sink: &dyn Show) -> crate::Result<()> {
    // The only legal first move. An Illegal here would mean the caller
    // handed us a non-fresh turn — a programming bug — and we surface it
    // as a regular error rather than a panic so a misconfigured caller
    // doesn't kill the agent mid-sentence.
    turn.transition(TurnState::InProgress, None)
        .map_err(|e| Error::Msg(e.to_string()))?;

    let result = run_turn(cfg, history, token, hub, sink).await;
    // The Result → TurnOutcome translation lives here because `Error` is
    // a main-rs type and turn.rs deliberately does not depend on it.
    // From TurnOutcome onward the mapping is the turn's own concern
    // ([`Turn::finish`]).
    let outcome = match &result {
        Ok(()) => TurnOutcome::Completed,
        Err(Error::Interrupted) => TurnOutcome::Interrupted,
        Err(e) => TurnOutcome::Failed(e.to_string()),
    };
    turn.finish(outcome);
    result
}

/// Drive one turn's worth of API calls and tool runs. Pure mechanics — the
/// [`Turn`] state machine is owned by the caller. Returns whatever
/// happened; the caller maps it onto a terminal state via [`Turn::finish`].
async fn run_turn(cfg: &crate::config::Config, history: &mut Vec<Message>, token: &crate::cancel::CancelToken, hub: &Hub, sink: &dyn Show) -> crate::Result<()> {
    let mut schemas: Vec<Value> = tools().iter()
        .map(|t| json!({"name": t.name, "description": t.desc, "input_schema": t.schema}))
        .collect();
    schemas.extend(hub.definitions().await);
    for step in 0..cfg.max_turns {
        if token.is_cancelled() { return Err(Error::Interrupted); }
        if fit_context(history, cfg.context_size) {
            sink.show(Msg::Note { sev: Sev::Warn, text: " trimmed history to fit --context-size ".into() });
        }
        let (resp, interrupted) = api_call_api(cfg, history, &schemas, token, sink).await?;
        let content = resp.blocks;
        if interrupted {
            // Cut off mid-response: keep finished text/thinking, drop tool
            // calls — a tool_use whose tool_result never ran would poison
            // the next request.
            let kept: Vec<Block> = content.into_iter().filter(|b| b.tool_use().is_none()).collect();
            if !kept.is_empty() {
                history.push(Message::Assistant(kept));
            }
            return Err(Error::Interrupted);
        }
        let calls: Vec<Block> = content.iter().filter(|b| b.tool_use().is_some()).cloned().collect();
        // Echo the full assistant turn back so interleaved thinking stays continuous —
        // including the final text-only turn, so follow-up questions keep context.
        if !content.is_empty() {
            history.push(Message::Assistant(content));
        }
        if calls.is_empty() { return Ok(()); }

        // Run the tools. Cancellation stops the batch: the interrupted call
        // reports as much as it got, the rest are skipped, and every tool_use
        // still leaves with its tool_result.
        let mut results = Vec::with_capacity(calls.len());
        let mut stopped = false;
        for c in &calls {
            let (id, name, input) = c.tool_use().expect("filtered to tool calls");
            let out = if stopped {
                "skipped: this run was interrupted before the tool ran".into()
            } else {
                run_tool(name, input, token, hub).await
            };
            let out = truncate(&out);
            print_tool_call(name, input, &out, sink);
            results.push(ToolResult { id: id.to_string(), content: out });
            if token.is_cancelled() { stopped = true; }
        }
        history.push(Message::ToolResults(results));
        if stopped { return Err(Error::Interrupted); }

        if step + 1 == cfg.max_turns {
            sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: hit --max-turns={}, stopping ", cfg.max_turns) });
        }
    }
    Ok(())
}



fn truncate(s: &str) -> String {
    if s.len() <= MAX_TOOL_OUTPUT { return s.into(); }
    let mut end = MAX_TOOL_OUTPUT;
    while !s.is_char_boundary(end) { end -= 1; }
    format!("{}…[truncated, {} bytes total]", &s[..end], s.len())
}

/// Rough token estimate (bytes/3 — conservative for CJK-heavy content).
fn est_tokens(history: &[Message]) -> u64 {
    serde_json::to_string(&ir::history_value(history)).map_or(0, |s| (s.len() / 3) as u64)
}

/// Shrink history until the estimate fits `limit`, trimming the oldest
/// tool_result first. Both shrinks keep tool_use/result pairing valid: an
/// in-place cut touches only the result's text, and a minimal result leaves
/// together with its paired tool_use (an orphaned tool_use is a 400 on the
/// next request), along with any message left holding no blocks.
fn fit_context(history: &mut Vec<Message>, limit: u64) -> bool {
    let mut changed = false;
    while est_tokens(history) > limit {
        let Some((mi, bi)) = oldest_tool_result(history) else { break };
        let Message::ToolResults(rs) = &mut history[mi] else { break };
        if rs[bi].content.chars().count() > 200 {
            let cut: String = rs[bi].content.chars().take(200).collect();
            rs[bi].content = format!("{cut}…[trimmed to fit context]");
            changed = true;
            continue;
        }
        drop_result_and_pair(history, mi, bi);
        changed = true;
    }
    changed
}

/// The first (oldest) tool_result in the history, as (message, block) indices.
fn oldest_tool_result(history: &[Message]) -> Option<(usize, usize)> {
    history.iter().position(|m| matches!(m, Message::ToolResults(rs) if !rs.is_empty())).map(|mi| (mi, 0))
}

/// Delete the tool_result at (mi, bi) and its tool_use — which sits in the
/// assistant message just before — so neither survives unpaired. A thinking
/// block that only led up to that call goes too; messages emptied of blocks
/// are removed outright (an empty content array is its own API error).
fn drop_result_and_pair(history: &mut Vec<Message>, mi: usize, bi: usize) {
    let id = match &history[mi] {
        Message::ToolResults(rs) => rs[bi].id.clone(),
        _ => return,
    };
    if let Message::ToolResults(rs) = &mut history[mi] { rs.remove(bi); }
    if mi > 0 {
        if let Message::Assistant(blocks) = &mut history[mi - 1] {
            blocks.retain(|b| match b.tool_use() { Some((bid, _, _)) => bid != id, None => true });
            // thinking whose tool_use is gone: nothing left to reason towards
            if !blocks.is_empty() && blocks.iter().all(Block::is_thinking) { blocks.clear(); }
        }
    }
    let empty = |m: &Message| matches!(m, Message::Assistant(b) if b.is_empty())
        || matches!(m, Message::ToolResults(r) if r.is_empty());
    if empty(&history[mi]) {
        history.remove(mi);
        if mi > 0 && empty(&history[mi - 1]) {
            history.remove(mi - 1);
        }
    }
}

// ---- tests -----------------------------------------------------------------

/// Cargo runs a crate's tests in parallel threads of one process, and
/// `std::env` is process-global — one test's `NO_COLOR` is another test's
/// surprise. Every test that reads or writes an environment variable holds
/// this lock, so the env-touching set runs one at a time. Test-only: the
/// binary never touches it.
#[cfg(test)]
#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
    use crate::api::{backoff, call_api as api_call_api, CacheMode, Protocol, RETRY_MAX, Thinking};
    use crate::config::{Args, DEFAULT_CONTEXT_SIZE, DEFAULT_MAX_TURNS};
    use crate::tools::dispatch;
    use crate::tool_runtime::run_bash;
    use crate::test_util::{temp_dir, block_on, cfg, mock, mock_stall};
    use crate::cancel::CancelToken;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A typed history from JSON literals — the tests still speak the wire
    /// shape, and `ir` owns the only translation of it.
    fn hv(values: Vec<Value>) -> Vec<Message> {
        values.iter().map(Message::from_value).collect()
    }
    /// History back as JSON, for the assertions the tests have always made.
    fn hist(history: &[Message]) -> Value {
        ir::history_value(history)
    }
    /// A response's blocks as JSON, read like a `content` array.
    fn blocks(b: &[Block]) -> Value {
        Value::Array(b.iter().map(Block::to_value).collect())
    }

    /// A sink that drops everything — the tests assert on state and on the
    /// mock wire, never on what a frontend shows. The real frontends
    /// (`plain::PlainSink`, the TUI's `ChannelSink`) are exercised through
    /// the binary instead.
    struct NullSink;
    impl Show for NullSink {
        fn show(&self, _m: Msg) {}
    }
    fn sink() -> NullSink {
        NullSink
    }

    /// Sequential mock: serves one response per connection, in order, and
    /// records the raw bytes of every request it received.
    fn mock_seq(responses: Vec<(u16, String)>) -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for (status, body) in responses {
                let Ok((mut s, _)) = listener.accept() else { break };
                let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(200)));
                let mut req = Vec::new();
                let mut buf = [0u8; 16_384];
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 { break }
                    req.extend_from_slice(&buf[..n]);
                }
                log.lock().unwrap().push(String::from_utf8_lossy(&req).into_owned());
                let ctype = if body.starts_with("data:") { "text/event-stream" } else { "application/json" };
                let resp = format!("HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}\n\n", body.len() + 2);
                let _ = s.write_all(resp.as_bytes());
                let _ = s.flush();
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        });
        (port, seen)
    }

    /// SSE server with two connections: `first` is served complete (with a
    /// Content-Length, then closed); `stalled` is streamed and then the
    /// connection is *held open* — an answer cut off mid-stream, the exact
    /// moment Ctrl-C lands.
    #[test]
    fn truncate_caps_long_output() {
        assert_eq!(truncate("hello"), "hello");
        let big = "x".repeat(MAX_TOOL_OUTPUT + 5);
        assert!(truncate(&big).contains("truncated"));
    }

    #[test]
    fn truncate_cuts_on_char_boundaries() {
        // the cut lands inside a multi-byte char — must back off, not slice mid-char
        let mut big = "x".repeat(MAX_TOOL_OUTPUT - 2);
        big.push_str("精卫填海");
        let t = truncate(&big);
        assert!(t.starts_with('x'));
        assert!(t.contains(&format!("…[truncated, {} bytes total]", big.len())), "got: {t}");
        assert!(t.split('…').next().unwrap().len() <= MAX_TOOL_OUTPUT, "kept prefix must fit the cap");
    }

    #[test]
    fn bash_tool_returns_exit_code_and_output() {
        let ok = dispatch("bash", &json!({"command": "echo hi"}));
        assert!(ok.starts_with("exit=0") && ok.contains("hi"), "got: {ok}");
        let bad = dispatch("bash", &json!({"command": if cfg!(windows) { "exit 1" } else { "false" }}));
        assert!(!bad.starts_with("exit=0") && bad.contains("exit="), "got: {bad}");
    }

    #[test]
    fn write_read_edit_file_contract() {
        let dir = temp_dir("files");
        let p = dir.join("f.txt");
        // write (with missing parent dirs) + read roundtrip
        assert!(dispatch("write_file", &json!({"path": dir.join("a/b.txt").to_str().unwrap(), "content": "v1"})).starts_with("ok:"));
        assert_eq!(dispatch("read_file", &json!({"path": dir.join("a/b.txt").to_str().unwrap()})), "v1");
        // edit unique
        fs::write(&p, "hello world").unwrap();
        assert!(dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "hello", "new": "hi"})).starts_with("ok:"));
        assert_eq!(fs::read_to_string(&p).unwrap(), "hi world");
        // edit missing
        assert!(dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "zzz", "new": "x"})).contains("not found"));
        // edit ambiguous — re-read so the ledger knows the current bytes,
        // and the ambiguity message is the one we test (the ledger check
        // fires first and would otherwise steal the message).
        fs::write(&p, "ab ab").unwrap();
        dispatch("read_file", &json!({"path": p.to_str().unwrap()}));
        assert!(dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "ab", "new": "cd"})).contains("appears 2 times"));
        // read missing
        assert!(dispatch("read_file", &json!({"path": dir.join("nope").to_str().unwrap()})).starts_with("error:"));
        // unknown tool
        assert!(dispatch("nope", &json!({})).starts_with("error: unknown tool"));
    }

    #[test]
    fn tools_registry_is_wellformed() {
        let t = tools();
        assert_eq!(t.len(), 4);
        assert_eq!(t.iter().map(|x| x.name).collect::<std::collections::BTreeSet<_>>().len(), 4);
        for x in t {
            assert_eq!(x.schema["type"], "object");
            assert!(x.schema["properties"].is_object() && x.schema["required"].is_array());
        }
    }

    // ---- flags & config ----

    fn flags(list: &[&str]) -> Result<Args> {
        // clap lives in this module; tests here use the public test seam.
        crate::parse_argv(list)
    }

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
    fn user_message_is_the_single_spelling_of_a_task() {
        assert_eq!(user_message("count files").to_value(), json!({"role": "user", "content": "count files"}));
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

    // ---- thinking policy & context trim ----

    #[test]
    fn fit_context_trims_oldest_tool_result_and_keeps_pairing() {
        let big = "x".repeat(10_000);
        let mut history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]}),
        ]);
        assert!(fit_context(&mut history, 500));
        let j = hist(&history);
        assert!(j[2]["content"][0]["content"].as_str().unwrap().contains("trimmed"));
        assert_eq!(j[1]["content"][0]["id"], j[2]["content"][0]["tool_use_id"]);
        let mut tiny = hv(vec![json!({"role": "user", "content": "t"})]);
        assert!(!fit_context(&mut tiny, 10_000));
    }

    #[test]
    fn fit_context_drops_short_tool_result_messages_when_still_over() {
        let mut history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "one call, one thought"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}), // too short to trim in place
            json!({"role": "assistant", "content": [{"type": "text", "text": "padding to keep the estimate high"}]}),
        ]);
        let limit = est_tokens(&history) - 1; // guarantee the first pass is over
        assert!(fit_context(&mut history, limit));
        // the whole exchange is gone — the assistant message held only the call
        assert_eq!(history.len(), 2);
        assert!(!hist(&history).to_string().contains("tool_result"));
        assert_pairing(&history);
    }

    #[test]
    fn fit_context_keeps_unpaired_blocks_of_partially_dropped_batches() {
        // a batch of two calls where only the first result is minimal: the
        // second call/result pair must survive the first one's removal
        let mut history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan: run two"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "true"}},
                {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "x"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},        // minimal → dropped with t1
                {"type": "tool_result", "tool_use_id": "t2", "content": "y".repeat(500)}]}), // trimmed in place
            json!({"role": "assistant", "content": [{"type": "text", "text": "padding to keep the estimate high"}]}),
        ]);
        let limit = est_tokens(&history) - 1;
        assert!(fit_context(&mut history, limit));
        assert_pairing(&history);
        let body = hist(&history).to_string();
        assert!(!body.contains("\"id\": \"t1\""), "t1's tool_use must not survive its result: {body}");
        assert!(body.contains("t2"), "t2's pair must both survive: {body}");
        assert!(body.contains("thinking"), "thinking led up to t2 as well — it stays: {body}");
    }

    /// Every tool_use keeps exactly one tool_result and vice versa, and no
    /// message is left with an empty content array.
    fn assert_pairing(history: &[Message]) {
        let uses: Vec<String> = history.iter().flat_map(|m| match m {
            Message::Assistant(blocks) => blocks.iter()
                .filter_map(|b| b.tool_use().map(|(id, _, _)| id.to_string())).collect::<Vec<_>>(),
            _ => vec![],
        }).collect();
        let results: Vec<String> = history.iter().flat_map(|m| match m {
            Message::ToolResults(rs) => rs.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
            _ => vec![],
        }).collect();
        for u in &uses { assert!(results.contains(u), "tool_use {u} lost its tool_result"); }
        for r in &results { assert!(uses.contains(r), "tool_result {r} lost its tool_use"); }
        for m in history {
            match m {
                Message::Assistant(b) => assert!(!b.is_empty(), "empty message left behind"),
                Message::ToolResults(r) => assert!(!r.is_empty(), "empty message left behind"),
                Message::User(_) => {}
            }
        }
    }

    // ---- zai vendor: dialect conversion ----

    // ---- api over local mocks ----

    // ---- deepseek vendor ----

    // ---- agent loop ----

    #[test]
    fn agent_loop_runs_tool_then_stops_on_plain_text() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"echo hi"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let done = r#"{"content":[{"type":"text","text":"done"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, reqs) = mock_seq(vec![(200, tool.into()), (200, done.into())]);
        let mut history = hv(vec![json!({"role": "user", "content": "run echo"})]);
        let mut turn = Turn::new(0);
        block_on(async {
            // Hub::empty() spawns an actor task — the test needs a runtime
            // for the spawn to land, even though the hub itself has no work
            // to do in this test.
            let hub = crate::mcp::Hub::empty();
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await.unwrap();
            hub.shutdown().await;
        });
        assert_eq!(history.len(), 4); // user + assistant(tool_use) + user(tool_result) + assistant(text)
        let j = hist(&history);
        assert_eq!(j[1]["role"], json!("assistant"));
        assert_eq!(j[1]["content"][0]["id"], json!("t1")); // full turn echoed back, thinking intact
        assert_eq!(j[2]["content"][0]["type"], json!("tool_result"));
        assert_eq!(j[2]["content"][0]["tool_use_id"], json!("t1"));
        assert!(j[2]["content"][0]["content"].as_str().unwrap().contains("hi"));
        assert_eq!(j[3]["content"][0]["text"], json!("done")); // final answer stays in history for follow-ups
        let reqs = reqs.lock().unwrap();
        assert!(reqs[0].contains("input_schema"), "tool schemas must be sent");
        assert!(reqs[1].contains("\"type\":\"tool_result\""), "results must feed the next request");
        // the turn reached Completed — the natural exit on a final-text response
        assert_eq!(turn.state, TurnState::Completed);
        assert!(turn.started_at.is_some() && turn.ended_at.is_some());
    }

    #[test]
    fn agent_loop_stops_at_max_turns() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"true"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, _) = mock_seq(vec![(200, tool.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.max_turns = 1;
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        block_on(async {
            let hub = crate::mcp::Hub::empty();
            agent_loop(&c, &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await.unwrap(); // must return instead of spinning
            hub.shutdown().await;
        });
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
        // hitting max_turns is a normal exit, not a failure — Completed, not Failed
        assert_eq!(turn.state, TurnState::Completed);
    }

    // ---- coroutine cancellation ----

    // bash kill/cancel tests live in tool_runtime::tests (the natural
    // home for them) with tighter timing ceilings.

    #[test]
    fn agent_loop_interrupted_mid_stream_keeps_history_valid() {
        let tool = concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"echo hi\"}"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_delta","usage":{"output_tokens":1}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n");
        // Second response streams one text delta, then stalls with the
        // connection open — the exact moment Ctrl-C lands.
        let stalled = concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial answ"}}"#, "\n\n");
        let port = mock_stall(tool, stalled);
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        let err = block_on(async {
            let hub = crate::mcp::Hub::empty();
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            let res = agent_loop(&cfg(format!("http://127.0.0.1:{port}"), true), &mut history, &mut turn, &token, &hub, &sink()).await;
            hub.shutdown().await;
            res
        }).unwrap_err();
        assert!(matches!(err, Error::Interrupted), "got: {err}");
        // user + assistant(tool_use) + user(tool_result) + assistant(partial text; tool_use dropped)
        assert_eq!(history.len(), 4);
        let j = hist(&history);
        assert_eq!(j[3]["role"], json!("assistant"));
        assert_eq!(j[3]["content"][0]["type"], json!("text"));
        assert_eq!(j[3]["content"][0]["text"], json!("partial answ"));
        assert!(j[3]["content"].as_array().unwrap().iter().all(|b| b["type"] != "tool_use"));
        // every tool_use still has its tool_result — the history stays sendable
        let ids: Vec<&str> = j[1]["content"].as_array().unwrap().iter()
            .filter(|b| b["type"] == "tool_use").map(|b| b["id"].as_str().unwrap()).collect();
        let results: Vec<&str> = j[2]["content"].as_array().unwrap().iter()
            .map(|b| b["tool_use_id"].as_str().unwrap()).collect();
        assert!(ids.iter().all(|i| results.contains(i)));
        // Ctrl-C mid-stream lands the turn on `Interrupted`, not `Failed` —
        // interruption is a stop, not an error, and the table keeps the two
        // separate so the banner can read "interrupted" without "error".
        assert_eq!(turn.state, TurnState::Interrupted);
        assert!(turn.error.is_none(), "interrupted is not a failure");
    }

    #[test]
    fn agent_loop_interrupted_mid_tool_finishes_pairing() {
        let cmd = if cfg!(windows) { "ping -n 30 127.0.0.1 > nul" } else { "sleep 30" };
        let tool = format!(
            r#"{{"content":[{{"type":"tool_use","id":"t1","name":"bash","input":{{"command":"{cmd}"}}}}],"usage":{{"input_tokens":1,"output_tokens":1}}}}"#);
        let (port, _) = mock_seq(vec![(200, tool)]);
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        let err = block_on(async {
            let hub = crate::mcp::Hub::empty();
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            let res = agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &mut turn, &token, &hub, &sink()).await;
            hub.shutdown().await;
            res
        }).unwrap_err();
        assert!(matches!(err, Error::Interrupted), "got: {err}");
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
        let result = hist(&history)[2]["content"][0]["content"].as_str().unwrap().to_string();
        assert!(result.contains("[interrupted by user]"), "got: {result}");
        assert_eq!(hist(&history)[2]["content"][0]["tool_use_id"], json!("t1"));
        // mid-tool cancel also lands on `Interrupted` — same state, same reason.
        assert_eq!(turn.state, TurnState::Interrupted);
    }

    /// A 500 from the wire is *not* `Failed`: the transport retries it
    /// (`is_transient`), so the user only sees the eventual result. We
    /// exhaust `RETRY_MAX` and serve another 500 on every attempt — once
    /// retries are gone, the turn ends in `Failed` with the API status
    /// text in `error`.
    #[test]
    fn agent_loop_records_failed_when_api_rejects() {
        // 5xx is transient; serve enough 500s to exhaust retries, and the
        // final error surfaces as `Failed`. Each retry is a fresh
        // connection on the mock listener, so N retries need N responses.
        let five_hundred = r#"{"type":"error","error":{"type":"api_error","message":"overloaded"}}"#;
        let mut responses = Vec::new();
        for _ in 0..=crate::api::RETRY_ATTEMPTS {
            responses.push((500, five_hundred.into()));
        }
        let (port, _) = mock_seq(responses);
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        let err = block_on(async {
            let hub = crate::mcp::Hub::empty();
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await
        }).unwrap_err();
        // the error reaches the caller — the same one we record on the turn
        assert!(matches!(err, Error::Api(500, _)), "got: {err}");
        assert_eq!(turn.state, TurnState::Failed, "5xx after retries is Failed, not Interrupted");
        assert!(turn.error.as_deref().unwrap_or("").contains("500"), "error carries the api status: got {:?}", turn.error);
        assert!(turn.started_at.is_some() && turn.ended_at.is_some(), "timestamps stamp both ends");
    }

    // bash+grandchild test lives in tool_runtime::tests; see cmd/conn
    // there for a deeper timeout note.

    /// The wrapper owns the cancel select; the loop-level tests above
    /// exercise the mechanics without it. This one drives the wrapper
    /// end-to-end and asserts on the same Turn the caller owns — so a
    /// future change that forgets to plumb `&mut turn` through agent_turn,
    /// or hands a non-fresh turn to agent_loop, fails here.
    #[test]
    fn agent_turn_drives_a_fresh_turn_through_to_completed() {
        let done = r#"{"content":[{"type":"text","text":"all done"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, _) = mock_seq(vec![(200, done.into())]);
        let mut turn = Turn::new(7);
        // a wrapper-level run starts in Queued — that's the caller's job
        // to construct, and the wrapper must not silently rewind it.
        assert_eq!(turn.state, TurnState::Queued);
        let mut history = hv(vec![json!({"role": "user", "content": "say hi"})]);
        block_on(async {
            let hub = crate::mcp::Hub::empty();
            agent_turn(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await.unwrap();
            hub.shutdown().await;
        });
        // the wrapper drove Queued → InProgress → Completed on the turn
        // the caller handed in; the id survives, the timestamps stamp
        // both ends, and the loop's mechanics ran at least once.
        assert_eq!(turn.id, 7, "wrapper must not rewind turn id");
        assert_eq!(turn.state, TurnState::Completed);
        assert!(turn.started_at.is_some() && turn.ended_at.is_some());
        assert_eq!(hist(&history).as_array().unwrap().last().unwrap()["role"], json!("assistant"));
    }

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
    fn backoff_doubles_then_caps() {
        assert_eq!(backoff(1), Duration::from_millis(500));
        assert_eq!(backoff(2), Duration::from_millis(1000));
        assert_eq!(backoff(3), Duration::from_millis(2000));
        assert_eq!(backoff(4), Duration::from_millis(4000));
        assert_eq!(backoff(5), RETRY_MAX, "doubling is capped");
        assert_eq!(backoff(99), RETRY_MAX, "and stays capped");
    }

    #[test]
    fn a_transient_failure_is_resent_through_the_one_door() {
        // 429 first, then the real answer: the door resends and the turn
        // succeeds where it used to die on the first status
        let (port, reqs) = mock_seq(vec![
            (429, r#"{"error":{"message":"rate limited"}}"#.into()),
            (200, r#"{"content":[{"type":"text","text":"after retry"}],"usage":{}}"#.into()),
        ]);
        let (resp, _) = block_on(crate::api::call_api(
            &cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new(), &sink(),
        )).unwrap();
        assert_eq!(blocks(&resp.blocks)[0]["text"], "after retry");
        assert_eq!(reqs.lock().unwrap().len(), 2, "the request went out twice");
    }

    #[test]
    fn a_request_fault_is_not_resent() {
        // a 400 is the request's own; it must fail on the first answer
        let (port, reqs) = mock_seq(vec![
            (400, r#"{"error":{"message":"bad model"}}"#.into()),
        ]);
        let err = block_on(crate::api::call_api(
            &cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new(), &sink(),
        )).unwrap_err();
        assert!(err.to_string().contains("bad model"), "got: {err}");
        assert_eq!(reqs.lock().unwrap().len(), 1, "no second request");
    }

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

    // ---- truncate: the bounded-output safety net --------------------------

    #[test]
    fn truncate_passes_through_short_strings() {
        assert_eq!(truncate("hello"), "hello");
        assert_eq!(truncate(""), "");
        assert_eq!(truncate(&"x".repeat(MAX_TOOL_OUTPUT)), "x".repeat(MAX_TOOL_OUTPUT));
    }

    #[test]
    fn truncate_strips_after_cap_with_marker() {
        let big = "a".repeat(MAX_TOOL_OUTPUT + 100);
        let t = truncate(&big);
        assert!(t.contains(&format!("…[truncated, {} bytes total]", big.len())));
        assert!(t.len() <= MAX_TOOL_OUTPUT + 64, "the marker is bounded: {t}");
    }

    // ---- est_tokens / fit_context: the trim machinery --------------------

    #[test]
    fn est_tokens_is_zero_when_history_cannot_be_serialized() {
        // the or-default path: history that fails to serialize still
        // returns 0, not a panic
        let h: Vec<Message> = vec![];
        assert_eq!(est_tokens(&h), 0);
    }

    #[test]
    fn est_tokens_grows_with_content_size() {
        // a longer user message bulks a larger estimate
        let s = "x".repeat(3_000);
        let h = vec![Message::User(s)];
        let small = est_tokens(&[Message::User("hi".into())]);
        let big = est_tokens(&h);
        assert!(big > small, "more bytes → more tokens: small={small}, big={big}");
    }

    #[test]
    fn oldest_tool_result_returns_none_when_history_has_no_results() {
        let h = vec![Message::User("t".into())];
        assert!(oldest_tool_result(&h).is_none());
    }

    #[test]
    fn oldest_tool_result_skips_empty_tool_results() {
        // a ToolResults with no entries does not count
        let h = vec![
            Message::User("go".into()),
            Message::Assistant(vec![Block::Text("ok".into())]),
            Message::ToolResults(vec![]),
            Message::ToolResults(vec![crate::ir::ToolResult { id: "t".into(), content: "ok".into() }]),
        ];
        let (mi, _) = oldest_tool_result(&h).expect("the second ToolResults has entries");
        assert_eq!(mi, 3);
    }

    #[test]
    fn drop_result_and_pair_drops_both_use_and_result() {
        let mut h = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}),
        ]);
        drop_result_and_pair(&mut h, 2, 0);
        // the tool_use and its tool_result both gone; the user message survives
        assert_eq!(h.len(), 1, "the empty assistant + result messages collapse: {h:?}");
        assert!(matches!(&h[0], Message::User(_)));
    }

    #[test]
    fn drop_result_and_pair_keeps_thinking_blocks_for_other_calls() {
        // a batch where thinking was reasoning towards *two* calls:
        // dropping the first result must not lose the thinking block —
        // it's still the reasoning for the second call
        let mut h = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "two calls"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "x"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "id": "t1", "content": "ok"},
                {"type": "tool_result", "id": "t2", "content": "y"}]}),
        ]);
        // find t1's position in the result message
        let (mi, _) = oldest_tool_result(&h).unwrap();
        // bi to be 0 (first result), t1 at bi=0
        drop_result_and_pair(&mut h, mi, 0);
        let j = hist(&h).to_string();
        assert!(!j.contains("\"id\": \"t1\""), "t1's tool_use is gone");
        assert!(j.contains("t2"), "t2's pair survives");
        assert!(j.contains("two calls"), "the thinking block survives t2's sake");
    }

    #[test]
    fn drop_result_and_pair_returns_silently_on_wrong_index() {
        // mi points at a non-ToolResults: the function must not panic
        let mut h = hv(vec![json!({"role": "user", "content": "go"})]);
        let before = h.clone();
        drop_result_and_pair(&mut h, 0, 0);
        assert_eq!(hist(&h), hist(&before));
    }

    #[test]
    fn fit_context_returns_false_when_history_already_fits() {
        let h = hv(vec![json!({"role": "user", "content": "hi"})]);
        assert!(!fit_context(&mut h.clone(), u64::MAX));
        let mut h2 = h.clone();
        assert!(!fit_context(&mut h2, est_tokens(&h) + 1000));
    }

    #[test]
    fn fit_context_returns_false_when_there_is_nothing_to_trim() {
        // an over-limit history with no tool_results: nothing changes
        let mut h = hv(vec![json!({"role": "user", "content": "go"})]);
        assert!(!fit_context(&mut h, 0));
    }
}
