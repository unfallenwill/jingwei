// The agent's configuration: how the user talks to the binary (CLI flags,
// env vars, the help text), and the resolved shape that the runtime,
// vendors, and tests all read. This is the seam between "what the user
// asked for" (Args) and "what the agent runs with" (Config). The vendor
// types themselves live in the api block of main.rs for now (phase 4 will
// move them to `api/`) — this module references them by path but does not
// own them.

use crate::api::{CacheMode, Effort, Protocol, Thinking, VENDORS};
use crate::{Error, Result};
use std::env;

pub(crate) const DEFAULT_MAX_TOKENS: u32 = 131_072;
pub(crate) const DEFAULT_CONTEXT_SIZE: u64 = 1_000_000;
pub(crate) const DEFAULT_MAX_TURNS: u32 = 60;

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
    pub(crate) streaming: bool,
}

/// What the user typed on the command line, before anything has been resolved
/// against the environment. Each field is `Option<String>` so the resolver
/// can fall back to an env var when the flag was absent.
#[derive(Default)]
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
    pub(crate) streaming: bool,
    pub(crate) resume: Option<String>,
    pub(crate) cont: bool,
    pub(crate) list: bool,
    pub(crate) all: bool,
    pub(crate) prompt: Vec<String>,
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

pub(crate) fn parse_from<I: Iterator<Item = String>>(it: I) -> Result<Args> {
    let mut it = it.peekable();
    let mut a = Args { streaming: true, ..Default::default() };
    while let Some(arg) = it.next() {
        // `--resume`'s value is optional: take the next word only when it
        // is not itself a flag; bare `--resume` means the newest, like -c
        if arg == "--resume" {
            a.resume = it.next_if(|v| !is_flag(v));
            if a.resume.is_none() { a.cont = true; }
            continue;
        }
        let mut val = || it.next().ok_or_else(|| Error::Msg(format!("{arg} needs a value")));
        match arg.as_str() {
            "-s" | "--stream" => a.streaming = true,
            "-S" | "--no-stream" => a.streaming = false,
            "--api-key" => a.api_key = Some(val()?),
            "--base-url" => a.base_url = Some(val()?),
            "-m" | "--model" => a.model = Some(val()?),
            "--protocol" => a.protocol = Some(val()?),
            "--cache" => a.cache = Some(val()?),
            "--thinking" => a.thinking = Some(val()?),
            "--effort" => a.effort = Some(val()?),
            "--max-tokens" => a.max_tokens = Some(val()?.parse().map_err(|_| Error::Msg("--max-tokens expects a number".into()))?),
            "--context-size" => a.context_size = Some(val()?.parse().map_err(|_| Error::Msg("--context-size expects a number (tokens)".into()))?),
            "--max-turns" => a.max_turns = Some(val()?.parse().map_err(|_| Error::Msg("--max-turns expects a number".into()))?),
            "-c" | "--continue" => a.cont = true,
            "--list" => a.list = true,
            "--all" => a.all = true,
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            x if is_flag(x) => return Err(Error::Msg(format!("unknown flag: {x}\ntry --help"))),
            _ => a.prompt.push(arg),
        }
    }
    Ok(a)
}

/// Does this word look like a flag? The one test both the unknown-flag arm
/// and `--resume`'s optional value ask of a word.
fn is_flag(s: &str) -> bool {
    s.starts_with('-') && s.len() > 1
}

/// Resolve an enum from flag or env, case-insensitive, defaulting to the
/// first variant. `variants` maps accepted spellings to values.
fn enum_of<T: Copy>(raw: &Option<String>, var: &str, name: &str, variants: &[(&'static str, T)]) -> Result<T> {
    let owned = raw.clone().or_else(|| env::var(var).ok());
    let raw = owned.as_deref().unwrap_or(variants[0].0);
    variants.iter().find(|(v, _)| v.eq_ignore_ascii_case(raw))
        .map(|(_, t)| *t)
        .ok_or_else(|| Error::Msg(format!(
            "invalid {name} '{raw}' (available: {})",
            variants.iter().map(|(v, _)| *v).collect::<Vec<_>>().join(", "),
        )))
}

/// [`enum_of`] without a default: absent flag and env mean `None`, which
/// the caller turns into "say nothing on the wire".
fn opt_enum_of<T: Copy>(raw: &Option<String>, var: &str, name: &str, variants: &[(&'static str, T)]) -> Result<Option<T>> {
    match raw.clone().or_else(|| env::var(var).ok()) {
        None => Ok(None),
        Some(s) => enum_of(&Some(s), var, name, variants).map(Some),
    }
}

/// Resolve `Args` against the environment into a `Config`: parse numbers,
/// fall back to env vars, ask the chosen vendor what its defaults are,
/// let it refuse anything it cannot serve. The wire's own rules live with
/// the wire (`protocol.accepts`), not in here.
pub(crate) fn build_config(args: &Args) -> Result<Config> {
    let req = |flag: &Option<String>, var: &str, what: &str| flag.clone()
        .or_else(|| env::var(var).ok())
        .ok_or_else(|| Error::Msg(format!("missing {what} (or env {var})")));
    let protocol = enum_of(&args.protocol, "JINGWEI_PROTOCOL", "protocol", VENDORS)?;
    let cache = enum_of(&args.cache, "JINGWEI_CACHE", "cache",
        &[("auto", CacheMode::Auto), ("active", CacheMode::Active)])?;
    let thinking = enum_of(&args.thinking, "JINGWEI_THINKING", "thinking",
        &[("preserve", Thinking::Preserve), ("strip", Thinking::Strip)])?;
    let effort = opt_enum_of(&args.effort, "JINGWEI_EFFORT", "effort",
        &[("low", Effort::Low), ("medium", Effort::Medium), ("high", Effort::High), ("max", Effort::Max)])?;
    protocol.accepts(cache, thinking, effort)?;
    let max_turns = args.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    if max_turns == 0 {
        return Err(Error::Msg("--max-turns must be at least 1".into()));
    }
    let base_url = args.base_url.clone()
        .or_else(|| env::var("JINGWEI_BASE_URL").ok())
        .or_else(|| protocol.default_base().map(str::to_owned))
        .ok_or_else(|| Error::Msg("missing --base-url (or env JINGWEI_BASE_URL)".into()))?;
    let model = args.model.clone()
        .or_else(|| env::var("JINGWEI_MODEL").ok())
        .or_else(|| protocol.default_model().map(str::to_owned))
        .ok_or_else(|| Error::Msg("missing -m/--model (or env JINGWEI_MODEL)".into()))?;
    Ok(Config {
        api_key: req(&args.api_key, "JINGWEI_API_KEY", "--api-key")?,
        base_url, model,
        protocol, cache, thinking, effort,
        max_tokens: args.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        context_size: args.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE),
        max_turns,
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
    --base-url <URL>    endpoint base (JINGWEI_BASE_URL)
                        minimax: POST {base}/v1/messages — base defaults to
                        https://api.minimax.cn/anthropic, model to MiniMax-M3
                        zai: POST {base}/chat/completions — base defaults to
                        https://open.bigmodel.cn/api/paas/v4, model to
                        glm-5.3-flash
                        deepseek: POST {base}/chat/completions — base defaults
                        to https://api.deepseek.com
    --api-key <KEY>     API key (JINGWEI_API_KEY)
    -m, --model <NAME>  model name (JINGWEI_MODEL; minimax defaults MiniMax-M3,
                        zai defaults glm-5.3-flash)
    --protocol <P>      minimax (default) | zai | deepseek

BEHAVIOR:
    --max-tokens <N>    max output tokens per turn (default 131072)
    --max-turns <N>     stop the agent loop after N turns (default 60)
    --context-size <N>  trim history when estimated tokens exceed N (default 1000000)
    --cache <MODE>      auto (default, passive server cache) | active (cache_control
                        breakpoints on the Messages wire — minimax's own)
    --thinking <MODE>   preserve (default) | strip reasoning from sent history
    --effort <TIER>     low | medium | high | max — reasoning effort, when the
                        endpoint offers the knob (JINGWEI_EFFORT); unset (default)
                        sends nothing and the endpoint's default rules.
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
    JINGWEI_NO_TUI=1    log-style REPL instead of the TUI

EXAMPLES:
    jingwei \"task\"                       # minimax · MiniMax-M3, endpoint known
    jingwei --effort high --cache active \"task\"
    jingwei --protocol zai --effort high \"task\"      # glm-5.3-flash, endpoint known
    jingwei --protocol deepseek --thinking strip \"task\"
";

pub(crate) fn print_help() {
    print!("{HELP}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::env_lock;
    use crate::api::{CacheMode, Effort, Protocol, Thinking};

    fn parse(argv: &[&str]) -> Result<Args> {
        parse_from(argv.iter().map(|s| s.to_string()))
    }

    fn cfg_effort(effort: Option<Effort>) -> Config {
        Config {
            api_key: "k".into(), base_url: "https://x".into(), model: "m".into(),
            protocol: Protocol::MINIMAX, cache: CacheMode::Auto, thinking: Thinking::Preserve,
            effort, max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
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

    /// Every flag that needs a value errors when the value is missing —
    /// the error names the flag so the user sees what to fix.
    #[test]
    fn parse_flags_that_take_values_error_when_missing() {
        for flag in ["--api-key", "--base-url", "--model", "--protocol", "--cache",
                     "--thinking", "--effort", "--max-tokens", "--context-size", "--max-turns"] {
            match parse(&[flag]) {
                Ok(_) => panic!("{flag} without value should fail"),
                Err(e) => assert!(e.to_string().contains(flag), "{flag}: got {e}"),
            }
        }
        match parse(&["--max-tokens", "abc"]) {
            Ok(_) => panic!("--max-tokens abc should fail"),
            Err(e) => assert!(e.to_string().contains("--max-tokens expects a number")),
        }
    }

    /// An unknown flag spells what it did not recognize — `try --help` so
    /// the user has a way out.
    #[test]
    fn parse_unknown_flag_says_what_it_got_and_points_at_help() {
        match parse(&["--frobnicate"]) {
            Ok(_) => panic!("--frobnicate should fail"),
            Err(e) => {
                let s = e.to_string();
                assert!(s.contains("unknown flag: --frobnicate"));
                assert!(s.contains("--help"));
            }
        }
    }

    /// `is_flag` is the seam two arms ask of a word — bare `-` is not a flag
    /// (it's a placeholder, not an option), and the empty string isn't either.
    #[test]
    fn is_flag_recognizes_dash_lead_only() {
        assert!(is_flag("--x") && is_flag("-x"));
        assert!(!is_flag("-"), "a lone dash is not a flag");
        assert!(!is_flag(""));
        assert!(!is_flag("x"), "no dash, no flag");
    }

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

    /// `opt_enum_of`'s `Some(s)` arm: when the env var is set, the resolver
    /// honors it even though no flag was passed. Parsed case-insensitively.
    #[test]
    fn effort_resolves_from_env_case_insensitively() {
        let _env = env_lock();
        std::env::set_var("JINGWEI_API_KEY", "k");
        std::env::set_var("JINGWEI_EFFORT", "HIGH");
        let args = parse(&[]).unwrap();
        let cfg = build_config(&args).unwrap();
        assert_eq!(cfg.effort, Some(Effort::High));
        std::env::remove_var("JINGWEI_EFFORT");
        std::env::remove_var("JINGWEI_API_KEY");
    }

    /// `opt_enum_of`'s absent path: no flag, no env → `None` (the wire
    /// stays silent and the endpoint's default rules apply).
    #[test]
    fn effort_absent_flag_and_env_stays_none() {
        let _env = env_lock();
        std::env::set_var("JINGWEI_API_KEY", "k");
        std::env::remove_var("JINGWEI_EFFORT");
        let args = parse(&[]).unwrap();
        let cfg = build_config(&args).unwrap();
        assert!(cfg.effort.is_none());
        std::env::remove_var("JINGWEI_API_KEY");
    }

    // ---- enum_of: the variant resolver -----------------------------------

    #[test]
    fn enum_of_falls_back_to_env_when_no_flag() {
        // no flag for thinking: the env var resolves the value
        let _env = env_lock();
        std::env::set_var("JINGWEI_API_KEY", "k");
        std::env::set_var("JINGWEI_THINKING", "strip");
        std::env::remove_var("JINGWEI_CACHE");
        let args = parse(&["--protocol", "minimax", "--base-url", "https://x", "-m", "m"]).unwrap();
        let cfg = build_config(&args).unwrap();
        assert_eq!(cfg.thinking, Thinking::Strip);
        std::env::remove_var("JINGWEI_API_KEY");
        std::env::remove_var("JINGWEI_THINKING");
    }

    #[test]
    fn enum_of_rejects_unknown_values_with_the_known_list() {
        // the error names the flag and lists what is accepted — it comes
        // from build_config, not parse_from (which has nothing to reject)
        let _env = env_lock();
        std::env::set_var("JINGWEI_API_KEY", "k");
        let args = parse(&["--cache", "turbo"]).unwrap();
        match build_config(&args) {
            Err(e) => {
                let s = e.to_string();
                assert!(s.contains("invalid cache 'turbo'"), "got: {s}");
                assert!(s.contains("auto, active"), "got: {s}");
            }
            Ok(_) => panic!("--cache turbo should fail"),
        }
        std::env::remove_var("JINGWEI_API_KEY");
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
