// The agent's configuration: how the user talks to the binary (the argv
// grammar and the help text), and the resolved shape that the runtime,
// vendors, and tests all read. This is the seam between "what the user
// asked for" (Args) and "what the agent runs with" (Config). The vendor
// types themselves live in `api` — this module references them by path
// but does not own them. `settings` (persisted preferences) and `login`
// (interactive setup) are this cluster's siblings.

pub(crate) mod login;
pub(crate) mod settings;

use crate::api::{CacheMode, Effort, Protocol, Thinking, VENDORS};
use crate::{Error, Result};
use clap::{Parser, Subcommand};
use std::env;

pub(crate) const DEFAULT_MAX_TOKENS: u32 = 131_072;
pub(crate) const DEFAULT_CONTEXT_SIZE: u64 = 1_000_000;
pub(crate) const DEFAULT_MAX_TURNS: u32 = 100;
/// How many background agents may run at once. Zero disables spawning —
/// the tool schemas stay on the wire, so the cache stays warm.
pub(crate) const DEFAULT_MAX_AGENTS: u32 = 4;
/// How long a `wait_agents` call waits before it answers with what it has
/// and says which workers are still running.
pub(crate) const DEFAULT_AGENT_WAIT_MS: u64 = 120_000;

/// Version of the internal history payload as it lands in session files —
/// owned here, beside the shape it versions; session.rs embeds it in the
/// header as an opaque integer and never interprets it. Bump it the day
/// the payload's *semantics* change; older files migrate at load.
pub(crate) const HISTORY_FORMAT: u32 = 1;

/// The resolved agent configuration: what the runtime and vendors read.
#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) protocol: Protocol,
    pub(crate) cache: CacheMode,
    pub(crate) thinking: Thinking,
    pub(crate) effort: Option<Effort>,
    pub(crate) max_tokens: u32,
    pub(crate) context_size: u64,
    pub(crate) max_turns: u32,
    /// How many background agents may run at once. Zero disables spawning.
    pub(crate) max_agents: u32,
    /// How long `wait_agents` waits before it reports what it has.
    pub(crate) agent_wait_ms: u64,
    pub(crate) streaming: bool,
}

/// What the user typed on the command line. Each field is `Option<String>`
/// so the resolver can reject a missing required one with a clear message.
#[derive(Default, Debug)]
pub(crate) struct Args {
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) protocol: Option<String>,
    pub(crate) cache: Option<String>,
    pub(crate) thinking: Option<String>,
    pub(crate) effort: Option<String>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) context_size: Option<u64>,
    pub(crate) max_turns: Option<u32>,
    pub(crate) max_agents: Option<u32>,
    pub(crate) agent_wait_ms: Option<u64>,
    pub(crate) streaming: bool,
    pub(crate) resume: Option<String>,
    pub(crate) cont: bool,
    pub(crate) list: bool,
    pub(crate) all: bool,
    pub(crate) prompt: Vec<String>,
    pub(crate) no_agents_md: bool,
}

impl Args {
    pub(crate) fn resuming(&self) -> bool {
        self.cont || self.resume.is_some()
    }
}

impl Config {
    pub(crate) fn protocol_label(&self) -> &'static str {
        self.protocol.label()
    }

    /// The banner's identity tail: protocol · model [· effort] · base url.
    pub(crate) fn identity(&self) -> String {
        let mut s = format!("{} · {}", self.protocol_label(), self.model);
        if let Some(e) = self.effort {
            s.push_str(&format!(" · effort {}", e.label()));
        }
        s.push_str(&format!(" · {}", self.base_url));
        s
    }

    pub(crate) fn effort_label(&self) -> Option<&'static str> {
        self.effort.map(Effort::label)
    }
}

/// Resolve an enum from a flag value, case-insensitive, defaulting to the
/// first variant. `variants` maps accepted spellings to values.
fn enum_of<T: Copy>(raw: &Option<String>, name: &str, variants: &[(&'static str, T)]) -> Result<T> {
    let raw = raw.as_deref().unwrap_or(variants[0].0);
    variants.iter().find(|(v, _)| v.eq_ignore_ascii_case(raw))
        .map(|(_, t)| *t)
        .ok_or_else(|| Error::Msg(format!(
            "invalid {name} '{raw}' (available: {})",
            variants.iter().map(|(v, _)| *v).collect::<Vec<_>>().join(", "),
        )))
}

/// [`enum_of`] without a default: absent flag means `None`, which the
/// caller turns into "say nothing on the wire".
fn opt_enum_of<T: Copy>(raw: &Option<String>, name: &str, variants: &[(&'static str, T)]) -> Result<Option<T>> {
    match raw.as_deref() {
        None => Ok(None),
        Some(s) => enum_of(&Some(s.to_string()), name, variants).map(Some),
    }
}

/// Resolve `Args` into a `Config`: parse numbers, ask the chosen vendor
/// what its defaults are, let it refuse anything it cannot serve. The
/// wire's own rules live with the wire (`protocol.accepts`), not in here.
pub(crate) fn build_config(args: &Args) -> Result<Config> {
    let req = |flag: &Option<String>, what: &str| -> Result<String> {
        flag.clone().ok_or_else(|| Error::Msg(format!("missing {what}")))
    };
    let protocol = enum_of(&args.protocol, "protocol", VENDORS)?;
    let cache = enum_of(&args.cache, "cache",
        &[("auto", CacheMode::Auto), ("active", CacheMode::Active)])?;
    let thinking = enum_of(&args.thinking, "thinking",
        &[("preserve", Thinking::Preserve), ("strip", Thinking::Strip)])?;
    let effort = opt_enum_of(&args.effort, "effort",
        &[("low", Effort::Low), ("medium", Effort::Medium), ("high", Effort::High), ("max", Effort::Max)])?;
    protocol.accepts(cache, thinking, effort)?;
    let max_turns = args.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    if max_turns == 0 {
        return Err(Error::Msg("--max-turns must be at least 1".into()));
    }
    let base_url = args.base_url.clone()
        .or_else(|| protocol.default_base().map(str::to_owned))
        .ok_or_else(|| Error::Msg("missing --base-url".into()))?;
    let model = args.model.clone()
        .or_else(|| protocol.default_model().map(str::to_owned))
        .ok_or_else(|| Error::Msg("missing -m/--model".into()))?;
    Ok(Config {
        api_key: req(&args.api_key, "--api-key")?,
        base_url, model,
        protocol, cache, thinking, effort,
        max_tokens: args.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        context_size: args.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE),
        max_turns,
        max_agents: args.max_agents.unwrap_or(DEFAULT_MAX_AGENTS),
        agent_wait_ms: args.agent_wait_ms.unwrap_or(DEFAULT_AGENT_WAIT_MS),
        streaming: args.streaming,
    })
}

pub(crate) const HELP: &str = "\
jingwei — 精卫填海 · a coding agent that fills the sea one stone at a time

No built-in providers. You bring an endpoint; jingwei speaks its wire protocol.

USAGE:
    jingwei [OPTIONS] [PROMPT]
    jingwei                       # interactive REPL

CONNECTION:
    --base-url <URL>    endpoint base
                        minimax: POST {base}/v1/messages — base defaults to
                        https://api.minimax.cn/anthropic, model to MiniMax-M3
                        zai: POST {base}/chat/completions — base defaults to
                        https://open.bigmodel.cn/api/paas/v4, model to
                        glm-5.3-flash
                        deepseek: POST {base}/chat/completions — base defaults
                        to https://api.deepseek.com
    --api-key <KEY>     API key
    -m, --model <NAME>  model name (minimax defaults MiniMax-M3,
                        zai defaults glm-5.3-flash)
    --protocol <P>      minimax (default) | zai | deepseek

BEHAVIOR:
    --max-tokens <N>    max output tokens per turn (default 131072)
    --max-turns <N>     stop the agent loop after N turns (default 100)
    --context-size <N>  trim history when estimated tokens exceed N (default 1000000)
    --max-agents <N>    how many background agents may run at once (default 4;
                        0 disables spawning)
    --agent-wait-ms <MS> how long wait_agents waits before reporting \"still running\"
                        (default 120000)
    --cache <MODE>      auto (default, passive server cache) | active (cache_control
                        breakpoints on the Messages wire — minimax's own)
    --thinking <MODE>   preserve (default) | strip reasoning from sent history
    --effort <TIER>     low | medium | high | max — reasoning effort, when the
                        endpoint offers the knob; unset (default) sends
                        nothing and the endpoint's default rules apply.
                        minimax wire: thinking adaptive + budget_tokens (low
                        1024 · medium 8k · high 32k · max = max-tokens minus
                        a floor for the reply)
                        zai wire: reasoning_effort (low/high/max its words;
                        medium folds into high — the wire rejects the word)
                        deepseek wire: reasoning_effort (the wire maps medium
                        to high; low/high/max are its own words)
    -s, --stream        stream output token by token (default)
    -S, --no-stream     wait for each turn to finish before printing
    -h, --help          this help

SESSIONS:
    interactive runs persist the conversation to
    ~/.jingwei/projects/<project>/<id>.jsonl — one subdirectory per
    project (the working directory's canonical path flattened to '-'),
    one JSON message per line, written at run boundaries (Ctrl-C keeps
    the runs before it); the header records cwd, model, protocol.
    One-shot runs stay ephemeral unless resumed.
    -c, --continue      resume the newest session from this project
    --resume [ID]       resume a session by id prefix; without an ID, like
                        -c. New turns append to the same file
    --list              list this project's saved sessions (newest first)
    --list --all        list every project's sessions

TUI (interactive, on a terminal):
    the transcript lives in the terminal's own scrollback — scroll with
    the mouse wheel or the terminal's keys; what was on screen before
    jingwei stays put, and the mouse is never captured
    Enter               submit the task — the input line clears; Up
                        recalls it from history
    Ctrl-J / Shift-Enter  break the line — compose multi-line tasks
                        (Shift-Enter needs a terminal with the kitty
                        keyboard protocol; Ctrl-J works everywhere)
    Up/Down             recall input history · Home/End line-wise
    Ctrl-O              unfold every folded block (thoughts, tool tails)
                        into a full-screen review; q or Ctrl-O returns to
                        the REPL — what opened stays open
    in the review view: ↑↓/j/k, PgUp/PgDn, g/G scroll
    Ctrl-C              interrupt the running task (twice: exit) · /exit
                        or Ctrl-D quit
    /model [<key>]       switch the active profile — <key> is <provider>/<model>
                        from ~/.jingwei/settings.json; bare /model lists. A
                        switch closes the current session file and opens a
                        new one under the same project subdirectory.
    /help                list slash commands

EXAMPLES:
    jingwei \"task\"                       # minimax · MiniMax-M3, endpoint known
    jingwei --effort high --cache active \"task\"
    jingwei --protocol zai --effort high \"task\"      # glm-5.3-flash, endpoint known
    jingwei --protocol deepseek --thinking strip \"task\"
";

pub(crate) fn print_help() {
    print!("{HELP}");
}

// ---- argv ------------------------------------------------------------------

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
    #[arg(long)] max_agents: Option<String>,
    #[arg(long)] agent_wait_ms: Option<String>,
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
    /// project conventions loaded — for paranoid sessions, or when the
    /// repo's AGENTS.md is known to be stale or misleading.
    #[arg(long)] no_agents_md: bool,

    // prompt — everything else, captured verbatim. clap's
    // `trailing_var_arg` lets the prompt begin after `--`, so users
    // who want to send `--help` as a literal task can still do so.
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    // -h / --help: clap's auto-generated help is disabled in favour
    // of the hand-written one in HELP (lists every vendor by
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

/// Parse the process argv into [`Args`]. `Ok(None)` means `--help` was
/// asked for and already answered on stdout. Subcommands with their own
/// exit paths (`login`) run here, exactly as they always did.
pub(crate) fn parse_env_args() -> Result<Option<Args>> {
    // parse() exits the process on error; the caller wants its own error
    // path so the binary's stderr formatting (the red " jingwei: ... "
    // banner) matches what every other code path uses.
    let cli = Cli::try_parse_from(env::args()).map_err(|e| Error::Msg(short_clap_err(&e)))?;
    if cli.help {
        print_help();
        return Ok(None);
    }
    Ok(Some(args_from_cli(cli)?))
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
            crate::config::login::run(show, reset, reveal)?;
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
            a.max_agents = parse_num_flag("--max-agents", c.max_agents)?;
            a.agent_wait_ms = parse_num_flag("--agent-wait-ms", c.agent_wait_ms)?;
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
/// and yield the internal [`Args`] the same way `parse_env_args` does.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{CacheMode, Effort, Protocol, Thinking};

    fn parse(argv: &[&str]) -> Result<Args> {
        // The argv grammar is this module's; tests go through the same
        // seam the binary does (prepends a fake argv[0]), so a test that
        // passes here passes at runtime.
        parse_argv(argv)
    }

    fn flags(list: &[&str]) -> Result<Args> { parse_argv(list) }

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
        assert_eq!(default.max_turns, DEFAULT_MAX_TURNS);
        assert_eq!(default.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(default.context_size, DEFAULT_CONTEXT_SIZE);
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
            parse_argv(&argv).unwrap()
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

    fn cfg_effort(effort: Option<Effort>) -> Config {
        Config {
            api_key: "k".into(), base_url: "https://x".into(), model: "m".into(),
            protocol: Protocol::MINIMAX, cache: CacheMode::Auto, thinking: Thinking::Preserve,
            effort, max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
            max_agents: DEFAULT_MAX_AGENTS, agent_wait_ms: DEFAULT_AGENT_WAIT_MS,
        }
    }

    /// `identity` always carries protocol + model + base, and only adds
    /// `· effort <tier>` when effort is actually set — the banner must not
    /// trail a stray `· effort` when the wire is silent.
    #[test]
    fn identity_includes_protocol_model_and_base_with_optional_effort() {
        let c = cfg_effort(None);
        assert_eq!(c.identity(), "minimax · m · https://x");
        assert_eq!(c.effort_label(), None);
        let c = cfg_effort(Some(Effort::High));
        assert_eq!(c.identity(), "minimax · m · effort high · https://x");
        assert_eq!(c.effort_label(), Some("high"));
    }

    /// `--resume` with no value is shorthand for `-c`: the resolver sets
    /// `cont = true` so the caller takes the newest-session path.
    #[test]
    fn parse_resume_without_value_falls_back_to_continue() {
        let a = parse(&["--resume"]).unwrap();
        assert!(a.resuming(), "bare --resume counts as a resume");
        assert!(a.cont && a.resume.is_none());
    }

    /// `--resume <id>` takes the next word; but a following flag is not
    /// swallowed as a value — it stays available to the loop.
    #[test]
    fn parse_resume_stops_at_the_next_flag() {
        let a = parse(&["--resume", "abc123", "--no-stream"]).unwrap();
        assert_eq!(a.resume.as_deref(), Some("abc123"));
        assert!(!a.streaming, "the second flag is still its own flag, not a value");
    }

    /// Every flag that needs a value errors when the value is missing.
    /// clap's wording is its own; the contract we test is just "parse
    /// must fail" — the user-visible message is generated by clap, not
    /// by us, so the test pins the failure mode without depending on
    /// the exact phrase.
    #[test]
    fn parse_flags_that_take_values_error_when_missing() {
        for flag in ["--api-key", "--base-url", "--model", "--protocol", "--cache",
                     "--thinking", "--effort", "--max-tokens", "--context-size", "--max-turns",
                     "--max-agents", "--agent-wait-ms"] {
            assert!(parse(&[flag]).is_err(), "{flag} without value should fail");
        }
        // numeric flag rejected for non-numeric input
        assert!(parse(&["--max-tokens", "abc"]).is_err(), "--max-tokens abc should fail");
        // and the message still names the flag, so the user can find it
        let e = parse(&["--max-tokens", "abc"]).unwrap_err().to_string();
        assert!(e.contains("--max-tokens"), "flag name leaks: {e}");
    }

    /// The fleet's two flags: they parse, and they default.
    #[test]
    fn the_fleet_flags_have_defaults() {
        let cfg = build_config(&flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m"]).unwrap()).unwrap();
        assert_eq!(cfg.max_agents, DEFAULT_MAX_AGENTS);
        assert_eq!(cfg.agent_wait_ms, DEFAULT_AGENT_WAIT_MS);
    }

    #[test]
    fn the_fleet_flags_parse() {
        let a = flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--max-agents", "2", "--agent-wait-ms", "5000"]).unwrap();
        assert_eq!(a.max_agents, Some(2));
        assert_eq!(a.agent_wait_ms, Some(5000));
        let cfg = build_config(&a).unwrap();
        assert_eq!(cfg.max_agents, 2);
        assert_eq!(cfg.agent_wait_ms, 5000);
        // zero is a value, not a mistake: it disables spawning
        let cfg = build_config(&flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--max-agents", "0"]).unwrap()).unwrap();
        assert_eq!(cfg.max_agents, 0);
    }

    #[test]
    fn a_bad_fleet_flag_is_a_clear_error() {
        let e = parse(&["--max-agents", "abc"]).unwrap_err().to_string();
        assert!(e.contains("--max-agents"), "the message names the flag: {e}");
        let e = parse(&["--agent-wait-ms", "soon"]).unwrap_err().to_string();
        assert!(e.contains("--agent-wait-ms"), "the message names the flag: {e}");
    }

    /// An unknown flag spells what it did not recognize. clap's error
    /// format names the flag and suggests `--help`; we check the flag
    /// appears in the message and `--help` is mentioned.
    #[test]
    fn parse_unknown_flag_says_what_it_got_and_points_at_help() {
        match parse(&["--frobnicate"]) {
            Ok(_) => panic!("--frobnicate should fail"),
            Err(e) => {
                let s = e.to_string();
                assert!(s.contains("--frobnicate"), "got: {s}");
                assert!(s.contains("--help") || s.contains("Usage"), "got: {s}");
            }
        }
    }

    // `is_flag` lived in the old hand-rolled parser; clap owns that
    // decision now, and the relevant behaviour is exercised through
    // `parse_flags_that_take_values_error_when_missing` and
    // `parse_unknown_flag_says_what_it_got_and_points_at_help` above.
    // Nothing to test in isolation here.

    /// `build_config` rejects `--max-turns 0`: a zero-iteration loop is
    /// not an agent, it is a silent no-op.
    #[test]
    fn build_config_rejects_zero_max_turns() {
        let args = parse(&["--max-turns", "0"]).unwrap();
        match build_config(&args) {
            Ok(_) => panic!("--max-turns 0 should fail"),
            Err(e) => assert!(e.to_string().contains("--max-turns must be at least 1")),
        }
    }

    /// `opt_enum_of`'s absent path: no flag means `None` (the wire
    /// stays silent and the endpoint's default rules apply).
    #[test]
    fn effort_absent_flag_stays_none() {
        let args = parse(&[]).unwrap();
        // no api_key: build_config itself fails before effort is consulted;
        // the resolver's absent-flag path is reached only when effort is
        // the only thing being decided — exercise it through the variant
        // pair directly so we don't need a full config.
        let effort = opt_enum_of(&args.effort, "effort",
            &[("low", Effort::Low), ("medium", Effort::Medium), ("high", Effort::High), ("max", Effort::Max)]).unwrap();
        assert!(effort.is_none());
    }

    /// `opt_enum_of`'s `Some(s)` arm: when the flag carries a value, the
    /// resolver honors it (parsed case-insensitively). Spelled without a
    /// full config so the case-insensitivity contract is the only thing
    /// under test.
    #[test]
    fn effort_flag_value_is_case_insensitive() {
        let args = parse(&["--effort", "HIGH"]).unwrap();
        let effort = opt_enum_of(&args.effort, "effort",
            &[("low", Effort::Low), ("medium", Effort::Medium), ("high", Effort::High), ("max", Effort::Max)]).unwrap();
        assert_eq!(effort, Some(Effort::High));
    }

    // ---- enum_of: the variant resolver -----------------------------------

    #[test]
    fn enum_of_uses_flag_value_when_provided() {
        // a flag for thinking resolves the value, no env involvement
        let args = parse(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--thinking", "strip"]).unwrap();
        let cfg = build_config(&args).unwrap();
        assert_eq!(cfg.thinking, Thinking::Strip);
    }

    #[test]
    fn enum_of_rejects_unknown_values_with_the_known_list() {
        // the error names the flag and lists what is accepted — it comes
        // from build_config, not parse (which has nothing to reject)
        let args = parse(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--cache", "turbo"]).unwrap();
        match build_config(&args) {
            Err(e) => {
                let s = e.to_string();
                assert!(s.contains("invalid cache 'turbo'"), "got: {s}");
                assert!(s.contains("auto, active"), "got: {s}");
            }
            Ok(_) => panic!("--cache turbo should fail"),
        }
    }

    #[test]
    fn protocol_label_and_identity_string_agree() {
        let c = cfg_effort(None);
        assert_eq!(c.protocol_label(), "minimax");
        assert!(c.identity().starts_with("minimax · m · "), "got: {}", c.identity());
    }

    #[test]
    fn parse_collects_bare_words_into_prompt() {
        // anything that does not look like a flag goes into the prompt
        // (and `is_flag` requires the dash to be the first character)
        let a = parse(&["count", "*.rs", "and", "more"]).unwrap();
        assert_eq!(a.prompt, vec!["count", "*.rs", "and", "more"]);
    }

    #[test]
    fn streaming_defaults_to_true_when_no_flag() {
        // the resolver defaults streaming to true; both -s and -S override it
        assert!(parse(&[]).unwrap().streaming);
        assert!(parse(&["-s"]).unwrap().streaming);
        assert!(!parse(&["-S"]).unwrap().streaming);
        assert!(!parse(&["--no-stream"]).unwrap().streaming);
    }
}
