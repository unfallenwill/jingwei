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
use crate::core::ir::{user_message, Message};
use crate::core::ports::ToolRouter;
use crate::core::session;
use crate::core::turn::Turn;
use crate::ui::{Msg, Sev, Show};
use crate::mcp::Hub;
use crate::tools::Builtins;
use crate::config::settings::Settings;
use serde_json::Value;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

/// The crate's one error type, and the `Result` that rides with it. They
/// live in `core::error` (L0) — the root is the composition root, and a
/// type an L4 module can only name through the root looks like an upward
/// edge. The root re-exports them, so every call site keeps its spelling.
pub use crate::core::error::{Error, Result};

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
    let ctx = Context::new(&agents_ctx, cfg.context_size);
    // MCP hub: external servers' tools, offered to the model as though
    // they were the agent's own. Spawned once per process and shared
    // across every turn (and every front-end). The user's servers are
    // read from `~/.jingwei/mcp.json`; the project's live in `<cwd>/.mcp.json`
    // and are picked up by the hub itself. A missing file is a None — the
    // hub starts empty and the session runs as it always did.
    let user_mcp = read_user_mcp();
    let hub = Hub::spawn(&workspace, user_mcp.as_ref());
    // The tool router: the built-ins first, the servers' tools after
    // them. The order is a cache contract — the combined schema list is
    // part of the cached prefix — and the router is where it is fixed.
    let router = Arc::new(ToolRouter::new(vec![
        Box::new(Builtins),
        Box::new(hub.clone()),
    ]));
    run_with_hub(args, cfg, ctx, hub, agents_ctx, router).await
}

/// The rest of `run` after the hub is spawned. The TUI and plain REPL take
/// the hub by value (it is `Clone` — handing a clone is a clone of an
/// `mpsc::Sender`); the one-shot path also takes a clone, then calls
/// `shutdown` on its own copy at the end so the connections close in
/// every exit path.
async fn run_with_hub(args: Args, cfg: Config, mut ctx: Context, hub: Hub, agents_ctx: AgentsMdContext, router: Arc<ToolRouter>) -> Result<()> {
    let interactive = args.prompt.is_empty();
    // The session decides the file; the transcript it continues is the
    // context's — the context owns it from here on.
    let (mut convo, resumed, banners) = begin_session(&args, &cfg, interactive)?;
    ctx.history_mut().extend(resumed);
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
            ui::tui::run(cfg, ctx, convo, banners, hub, router).await
        } else {
            ui::plain::plain_repl(&cfg, ctx, convo, banners, hub, router).await
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
    ctx.push(user_message(&prompt));
    if let Err(e) = convo.persist(ctx.history()) {
        sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
    }
    let token = CancelToken::new();
    let mut turn = Turn::new(0);
    let res = agent_turn(&cfg, &mut ctx, &mut turn, &token, &router, &sink).await;
    if let Err(e) = convo.persist(ctx.history()) {
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

/// The conversation this process runs on, the transcript it continues, and
/// the banners the frontend shows beside its own. Policy lives here, at the
/// composition root; session.rs owns the mechanism. A one-shot stays
/// ephemeral; a resumed one persists to the file it came from.
///
/// The transcript comes back separately because the context owns it: the
/// caller seeds the context with it, and the context is what the agent
/// loop, the trim policy, and the file all read.
fn begin_session(args: &Args, cfg: &Config, interactive: bool) -> Result<(session::Convo, Vec<Message>, Vec<String>)> {
    if !args.resuming() {
        if !interactive {
            return Ok((session::Convo::ephemeral(), vec![], vec![]));
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
        return Ok((session::Convo::persistent(s), vec![], vec![banner]));
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
    Ok((session::Convo::persistent(s), history, banners))
}

