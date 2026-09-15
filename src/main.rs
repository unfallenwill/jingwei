// jingwei — 精卫填海. A minimal coding agent: speak a task, and a small agent
// carries stones — one tool call at a time — until the sea is land.
//
// Wire protocols: Anthropic Messages or OpenAI Chat Completions. No built-in
// providers; you bring the endpoint, key, and model.
//
// Env: JINGWEI_API_KEY, JINGWEI_BASE_URL, JINGWEI_MODEL, JINGWEI_PROTOCOL,
//      JINGWEI_CACHE, JINGWEI_THINKING, JINGWEI_SHOW_THINKING, NO_COLOR

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{mpsc, Notify};

const MAX_TOOL_OUTPUT: usize = 50_000;
const DEFAULT_MAX_TOKENS: u32 = 131_072;
const DEFAULT_CONTEXT_SIZE: u64 = 1_000_000;
const DEFAULT_MAX_TURNS: u32 = 60;
/// How many lines of tool output are echoed to the terminal per call.
const MAX_TOOL_DISPLAY_LINES: usize = 20;

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

// ---- colors: backgrounds only for short, high-attention signals ------------

const WARN_BG: (u8, u8, u8) = (255, 220, 100);
const ERR_BG: (u8, u8, u8) = (198, 40, 40);
const BANNER_BG: (u8, u8, u8) = (26, 115, 232);

fn color_on() -> bool {
    env::var_os("NO_COLOR").is_none()
        && env::var_os("JINGWEI_NO_COLOR").is_none()
        && io::stdout().is_terminal()
        && io::stderr().is_terminal()
}

fn paint(s: &str, bg: (u8, u8, u8), white: bool) -> String {
    if color_on() {
        format!("\x1b[{}m\x1b[48;2;{};{};{}m{s}\x1b[49m\x1b[39m", if white { 97 } else { 30 }, bg.0, bg.1, bg.2)
    } else {
        s.into()
    }
}

fn print_text(t: &str) { print!("{t}"); }

fn print_thinking(t: &str) {
    if color_on() { eprint!("\x1b[2m{t}\x1b[22m"); } else { eprint!("{t}"); }
}

/// One-line summary of a tool call's arguments: the command, the path, …
fn tool_summary(name: &str, input: &Value) -> String {
    match name {
        "bash" => format!("$ {}", input["command"].as_str().unwrap_or("")),
        "read_file" => input["path"].as_str().unwrap_or("").into(),
        "write_file" => format!("{} ({} bytes)",
            input["path"].as_str().unwrap_or(""),
            input["content"].as_str().map_or(0, str::len)),
        "edit_file" => input["path"].as_str().unwrap_or("").into(),
        _ => String::new(),
    }
}

/// Lines describing a tool call — header (name + argument summary), then up to
/// MAX_TOOL_DISPLAY_LINES lines of output, then a "+N more" marker.
fn tool_call_lines(name: &str, input: &Value, output: &str) -> Vec<String> {
    let mut lines = vec![format!("[{name}] {}", tool_summary(name, input))];
    let all: Vec<&str> = output.lines().collect();
    let shown = all.len().min(MAX_TOOL_DISPLAY_LINES);
    lines.extend(all[..shown].iter().map(|l| format!("│ {l}")));
    if all.len() > shown {
        lines.push(format!("… +{} more lines", all.len() - shown));
    }
    lines
}

/// Echo a tool call to the terminal so the human sees what the agent is doing.
fn print_tool_call(name: &str, input: &Value, output: &str) {
    for (i, line) in tool_call_lines(name, input, output).into_iter().enumerate() {
        if !color_on() {
            eprintln!("  {line}");
        } else if i == 0 {
            eprintln!("  \x1b[1;33m{line}\x1b[0m");
        } else {
            eprintln!("  \x1b[2m{line}\x1b[22m");
        }
    }
}

// ---- tools -----------------------------------------------------------------

struct Tool {
    name: &'static str,
    desc: &'static str,
    schema: Value,
    run: fn(&Value) -> String,
}

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "bash",
            desc: "Run a shell command; returns combined stdout/stderr and the exit code. \
                   Use `cd <dir> && <cmd>` to change directory within a call.",
            schema: json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
            run: |i| {
                let prog_flag = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
                match Command::new(prog_flag.0).args([prog_flag.1, i["command"].as_str().unwrap_or("")]).output() {
                    Ok(o) => bash_output(Some(&o.status),
                        &String::from_utf8_lossy(&o.stdout),
                        &String::from_utf8_lossy(&o.stderr)),
                    Err(e) => format!("error: {e}"),
                }
            },
        },
        Tool {
            name: "read_file",
            desc: "Read the full contents of a file.",
            schema: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            run: |i| match fs::read(i["path"].as_str().unwrap_or("")) {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(e) => format!("error: {e}"),
            },
        },
        Tool {
            name: "write_file",
            desc: "Write content to a file, creating parent dirs as needed. Overwrites existing files.",
            schema: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
            run: |i| {
                let p = i["path"].as_str().unwrap_or("");
                let c = i["content"].as_str().unwrap_or("");
                if let Some(dir) = Path::new(p).parent().filter(|d| !d.as_os_str().is_empty()) {
                    let _ = fs::create_dir_all(dir);
                }
                fs::write(p, c).map(|_| format!("ok: wrote {} bytes to {p}", c.len()))
                    .unwrap_or_else(|e| format!("error: {e}"))
            },
        },
        Tool {
            name: "edit_file",
            desc: "Replace the first exact occurrence of `old` with `new`. \
                   Fails if `old` is missing or appears more than once.",
            schema: json!({"type":"object","properties":{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},"required":["path","old","new"]}),
            run: |i| {
                let p = i["path"].as_str().unwrap_or("");
                let old = i["old"].as_str().unwrap_or("");
                let new = i["new"].as_str().unwrap_or("");
                match fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|s| {
                    match s.matches(old).count() {
                        0 => Err("old text not found".into()),
                        1 => {
                            let updated = s.replacen(old, new, 1);
                            fs::write(p, updated).map(|_| format!("ok: edited {p}")).map_err(|e| e.to_string())
                        }
                        n => Err(format!("old text appears {n} times; must be unique")),
                    }
                }) {
                    Ok(msg) => msg,
                    Err(e) => format!("error: {e} (in {p})"),
                }
            },
        },
    ]
}

fn dispatch(name: &str, input: &Value) -> String {
    tools().into_iter().find(|t| t.name == name)
        .map(|t| (t.run)(input))
        .unwrap_or_else(|| format!("error: unknown tool '{name}'"))
}

// ---- tools as coroutines ---------------------------------------------------

/// Combined-output shape shared by the blocking registry tool and the
/// cancellable coroutine version below.
fn bash_output(status: Option<&ExitStatus>, out: &str, err: &str) -> String {
    let mut s = format!("exit={}\n{out}", status.and_then(|st| st.code()).unwrap_or(-1));
    if !err.is_empty() {
        if !s.ends_with('\n') { s.push('\n'); }
        s.push_str(err);
    }
    s
}

/// A child's pipe, drained on its own thread. Chunks stream back over an
/// *async* channel — not one final buffer — so partial output survives even
/// when the child is killed while a grandchild still holds the pipe open,
/// and waiting for the rest suspends the coroutine instead of blocking the
/// scheduler thread. What arrived so far lives in `got`, on the pipe itself,
/// so it survives any cancelled collect.
struct Drained {
    rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    got: Vec<u8>,
}

fn drain_pipe<R: Read + Send + 'static>(pipe: Option<R>) -> Drained {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    std::thread::spawn(move || {
        let Some(mut r) = pipe else { return };
        let mut buf = [0u8; 8192];
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break, // EOF or broken pipe
                Ok(n) => { if tx.blocking_send(buf[..n].to_vec()).is_err() { break } } // collector gone
            }
        }
    });
    Drained { rx, got: Vec::new() }
}

/// How long an interrupted pipe waits for stragglers before abandoning them.
const PIPE_GRACE: Duration = Duration::from_millis(300);

impl Drained {
    /// Collect what the pipe produced, as a coroutine. With `eof` this
    /// awaits EOF (the reader thread finishes when every writer closes the
    /// pipe — the same contract as the blocking tool) — an await, not a
    /// block, so a grandchild holding the pipe keeps the coroutine
    /// suspensible. Every wait races the cancel token: Ctrl-C during the
    /// EOF wait salvages whatever arrives within a short grace and
    /// abandons the rest; without `eof` (child already killed) the grace
    /// window is all there is.
    async fn collect(&mut self, token: &CancelToken, eof: bool) -> String {
        loop {
            let chunk = if eof {
                tokio::select! {
                    c = self.rx.recv() => c,
                    _ = token.cancelled() => match tokio::time::timeout(PIPE_GRACE, self.rx.recv()).await {
                        Ok(c) => c,
                        Err(_) => None, // grace elapsed: abandon the rest
                    },
                }
            } else {
                match tokio::time::timeout(PIPE_GRACE, self.rx.recv()).await {
                    Ok(c) => c,
                    Err(_) => None,
                }
            };
            match chunk {
                Some(c) => self.got.extend_from_slice(&c),
                None => break, // EOF, grace elapsed, or abandoned after cancel
            }
        }
        String::from_utf8_lossy(&self.got).into_owned()
    }
}

/// bash as a coroutine: same tool as the registry's, but it watches the
/// cancel token while the command runs. Ctrl-C kills the child at once — no
/// waiting out a runaway `sleep 300` — and whatever output it already
/// produced (plus an `[interrupted]` note) still reaches the model. If the
/// child left grandchildren holding the pipe, they get a short grace period
/// and are then abandoned.
async fn run_bash(command: &str, token: &CancelToken) -> String {
    let (prog, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
    let mut child = match Command::new(prog).args([flag, command])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return format!("error: {e}"),
    };
    let mut out_pipe = drain_pipe(child.stdout.take());
    let mut err_pipe = drain_pipe(child.stderr.take());
    let mut killed = false;
    let status = loop {
        if !killed && token.is_cancelled() {
            killed = true;
            let _ = child.kill();
        }
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {}
            Err(e) => return format!("error: {e}"),
        }
        if killed { break child.wait().ok(); } // kill sent: this returns promptly
        // The wait itself is a suspension point — a sleep raced against
        // cancellation, so waiting for the child is interruptible too.
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            _ = token.cancelled() => {}
        }
    };
    // Gather the pipes as a coroutine: the EOF wait suspends (a grandchild
    // holding the pipe can delay it) and races cancellation — Ctrl-C
    // salvages what arrived within the grace window and moves on, so no
    // background process can hold the agent hostage.
    let (out, err) = tokio::join!(out_pipe.collect(token, !killed), err_pipe.collect(token, !killed));
    if !killed && token.is_cancelled() { killed = true; } // pipes abandoned mid-collect
    let mut s = bash_output(status.as_ref(), &out, &err);
    if killed { s.push_str("\n[interrupted by user]"); }
    s
}

/// Execute one tool call inside the agent coroutine. bash is cancellable (its
/// child process is killed); the file tools are quick, run on the blocking
/// pool, and are simply abandoned if cancellation wins the race.
async fn run_tool(name: &str, input: &Value, token: &CancelToken) -> String {
    if name == "bash" {
        return run_bash(input["command"].as_str().unwrap_or(""), token).await;
    }
    let name = name.to_string();
    let input = input.clone();
    let job = tokio::task::spawn_blocking(move || dispatch(&name, &input));
    tokio::select! {
        out = job => out.unwrap_or_else(|e| format!("error: {e}")),
        _ = token.cancelled() => "error: interrupted by user".into(),
    }
}

// ---- config ----------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Protocol { Anthropic, OpenAI }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CacheMode { Auto, Active }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Thinking { Preserve, Strip }

struct Config {
    api_key: String,
    base_url: String,
    model: String,
    protocol: Protocol,
    cache: CacheMode,
    thinking: Thinking,
    max_tokens: u32,
    context_size: u64,
    max_turns: u32,
    streaming: bool,
    show_thinking: bool,
}

#[derive(Default)]
struct Args {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    protocol: Option<String>,
    cache: Option<String>,
    thinking: Option<String>,
    max_tokens: Option<u32>,
    context_size: Option<u64>,
    max_turns: Option<u32>,
    streaming: bool,
    prompt: Vec<String>,
}

fn parse_from<I: Iterator<Item = String>>(mut it: I) -> Result<Args> {
    let mut a = Args { streaming: true, ..Default::default() };
    while let Some(arg) = it.next() {
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
            "--max-tokens" => a.max_tokens = Some(val()?.parse().map_err(|_| Error::Msg("--max-tokens expects a number".into()))?),
            "--context-size" => a.context_size = Some(val()?.parse().map_err(|_| Error::Msg("--context-size expects a number (tokens)".into()))?),
            "--max-turns" => a.max_turns = Some(val()?.parse().map_err(|_| Error::Msg("--max-turns expects a number".into()))?),
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            x if x.starts_with('-') && x.len() > 1 => {
                return Err(Error::Msg(format!("unknown flag: {x}\ntry --help")));
            }
            _ => a.prompt.push(arg),
        }
    }
    Ok(a)
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

fn build_config(args: &Args) -> Result<Config> {
    let req = |flag: &Option<String>, var: &str, what: &str| flag.clone()
        .or_else(|| env::var(var).ok())
        .ok_or_else(|| Error::Msg(format!("missing {what} (or env {var})")));
    let protocol = enum_of(&args.protocol, "JINGWEI_PROTOCOL", "protocol",
        &[("anthropic", Protocol::Anthropic), ("openai", Protocol::OpenAI)])?;
    let cache = enum_of(&args.cache, "JINGWEI_CACHE", "cache",
        &[("auto", CacheMode::Auto), ("active", CacheMode::Active)])?;
    let thinking = enum_of(&args.thinking, "JINGWEI_THINKING", "thinking",
        &[("preserve", Thinking::Preserve), ("strip", Thinking::Strip)])?;
    if cache == CacheMode::Active && protocol != Protocol::Anthropic {
        return Err(Error::Msg("--cache active needs --protocol anthropic (it is Anthropic's cache_control scheme)".into()));
    }
    let max_turns = args.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    if max_turns == 0 {
        return Err(Error::Msg("--max-turns must be at least 1".into()));
    }
    Ok(Config {
        api_key: req(&args.api_key, "JINGWEI_API_KEY", "--api-key")?,
        base_url: req(&args.base_url, "JINGWEI_BASE_URL", "--base-url")?,
        model: req(&args.model, "JINGWEI_MODEL", "-m/--model")?,
        protocol, cache, thinking,
        max_tokens: args.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        context_size: args.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE),
        max_turns,
        streaming: args.streaming,
        show_thinking: env::var("JINGWEI_SHOW_THINKING").is_ok(),
    })
}

const HELP: &str = "\
jingwei — 精卫填海 · a coding agent that fills the sea one stone at a time

No built-in providers. You bring an endpoint; jingwei speaks its wire protocol.

USAGE:
    jingwei [OPTIONS] [PROMPT]
    jingwei                       # interactive REPL

CONNECTION:
    --base-url <URL>    endpoint base (JINGWEI_BASE_URL)
                        anthropic: POST {base}/v1/messages · openai: POST {base}/chat/completions
    --api-key <KEY>     API key (JINGWEI_API_KEY)
    -m, --model <NAME>  model name (JINGWEI_MODEL)
    --protocol <P>      anthropic (default) | openai

BEHAVIOR:
    --max-tokens <N>    max output tokens per turn (default 131072)
    --max-turns <N>     stop the agent loop after N turns (default 60)
    --context-size <N>  trim history when estimated tokens exceed N (default 1000000)
    --cache <MODE>      auto (default, passive server cache) | active (Anthropic cache_control)
    --thinking <MODE>   preserve (default) | strip reasoning from sent history
    -s, --stream        stream output (default) · -S, --no-stream to block
    Ctrl-C              interrupt the running agent (press twice to exit at once)
    -h, --help          this help

EXAMPLES:
    jingwei --base-url https://api.minimax.cn/anthropic -m MiniMax-M3 \"task\"
    jingwei --protocol openai --base-url https://host/v1 -m glm-5.3 --max-tokens 16384 \"task\"
";

fn print_help() {
    print!("{HELP}");
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
            eprintln!("{}", paint(&format!(" jingwei: {e} "), ERR_BG, true));
            std::process::exit(1);
        }
    }
}

const SYSTEM: &str = "You are jingwei (精卫), a coding agent with these tools: \
    bash (run a shell command), read_file, write_file, edit_file (replace exact string). \
    Prefer small, targeted commands. Never run destructive commands unless the user explicitly asks. \
    When asked your name, say you are jingwei (精卫). \
    When finished, reply with a concise 1-3 sentence summary of what you did.";

async fn run() -> Result<()> {
    let args = parse_from(env::args().skip(1))?;
    let cfg = build_config(&args)?;
    if args.prompt.is_empty() { repl(&cfg).await } else {
        let mut history = vec![json!({"role": "user", "content": args.prompt.join(" ")})];
        agent_turn(&cfg, &mut history).await
    }
}

async fn repl(cfg: &Config) -> Result<()> {
    let mut history: Vec<Value> = vec![];
    let mut rl = rustyline::DefaultEditor::new().map_err(|e| Error::Msg(format!("readline init: {e}")))?;
    if let Some(dir) = home_dir() { let _ = rl.load_history(&dir.join(".jingwei_history")); }
    eprintln!("{}", paint(" 精卫 ", BANNER_BG, true));
    println!("\x1b[1mjingwei\x1b[0m — 精卫填海，一石一石 · type a task, Ctrl-C interrupts, Ctrl-D rests");
    println!("\x1b[2m{} · {} · {}{}\x1b[22m", cfg.protocol_label(), cfg.model, cfg.base_url,
        if cfg.show_thinking { " · thinking on" } else { "" });
    loop {
        let line = match rl.readline(prompt_str()) {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => continue,
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => { eprintln!("{}", paint(&format!("warning: readline ({e})"), WARN_BG, false)); continue; }
        };
        let line = line.trim();
        if line.is_empty() { continue; }
        let _ = rl.add_history_entry(line);
        history.push(json!({"role": "user", "content": line}));
        // The agent runs as an interruptible coroutine: Ctrl-C during the run
        // cancels it (reported by `agent_turn` itself) while the history —
        // and this prompt — survive for the next task.
        match agent_turn(cfg, &mut history).await {
            Err(Error::Interrupted) => {}
            Err(e) => eprintln!("{}", paint(&format!(" error: {e} "), ERR_BG, true)),
            Ok(()) => {}
        }
        println!(); // blank line between interactions
    }
    if let Some(dir) = home_dir() { let _ = rl.save_history(&dir.join(".jingwei_history")); }
    Ok(())
}

/// Drive one agent run as an interruptible coroutine.
///
/// The first Ctrl-C cancels the token: the agent coroutine notices at its
/// next suspension point (a streamed line, a tool poll, a turn boundary),
/// tidies the history so every tool call stays paired, and unwinds — the
/// REPL prompt comes back with everything the agent already carried still in
/// place. A second Ctrl-C while it is unwinding exits immediately, for when
/// even graceful is too slow.
async fn agent_turn(cfg: &Config, history: &mut Vec<Value>) -> Result<()> {
    let token = CancelToken::new();
    let agent = agent_loop(cfg, history, &token);
    tokio::pin!(agent);
    tokio::select! {
        res = &mut agent => return res,
        _ = tokio::signal::ctrl_c() => token.cancel(),
    }
    let res = tokio::select! {
        res = &mut agent => res,
        _ = tokio::signal::ctrl_c() => {
            eprintln!("{}", paint(" interrupted again — exiting ", ERR_BG, true));
            std::process::exit(130);
        }
    };
    if let Err(Error::Interrupted) = &res {
        eprintln!("{}", paint(" interrupted — stopped at a safe point; history kept ", WARN_BG, false));
    }
    res
}

impl Config {
    fn protocol_label(&self) -> &'static str {
        match self.protocol { Protocol::Anthropic => "anthropic", Protocol::OpenAI => "openai" }
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn prompt_str() -> &'static str {
    if color_on() { "\x1b[1;38;5;75mjingwei\x1b[0m\x1b[38;5;240m ❯\x1b[0m " } else { "jingwei> " }
}


// ---- coroutine & cancellation ----------------------------------------------
//
// The agent is an async coroutine: a computation that suspends at every
// `await` and can be abandoned at any of those suspension points. Suspension
// points double as cancellation points — `CancelToken` is the cooperative
// channel between the outside world (Ctrl-C) and the running coroutine, so
// the agent can be interrupted at any moment, at a safe point of its own
// choosing, with the conversation history left valid.

/// One-shot cooperative cancellation. `cancel()` fires when the user hits
/// Ctrl-C; code inside the coroutine races its real work against
/// `cancelled().await` (suspends until cancelled) or peeks with
/// `is_cancelled()` (no suspension).
struct CancelToken {
    flag: AtomicBool,
    notify: Notify,
}

impl Default for CancelToken {
    fn default() -> Self { Self::new() }
}

impl CancelToken {
    fn new() -> Self {
        Self { flag: AtomicBool::new(false), notify: Notify::new() }
    }

    /// Request cancellation. Idempotent; wakes every suspended `cancelled()`.
    fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Resolves once the token is cancelled. Never spins — it suspends.
    async fn cancelled(&self) {
        loop {
            if self.is_cancelled() { return; }
            // Register the waiter *before* re-checking the flag, so a cancel
            // landing in between can never be missed.
            let notified = self.notify.notified();
            if self.is_cancelled() { return; }
            notified.await;
        }
    }
}


// ---- agent loop ------------------------------------------------------------

/// The agent itself, as a coroutine. Every await is a suspension point and a
/// cancellation point: between turns, on every streamed line, while each tool
/// runs. On cancellation it returns `Err(Interrupted)` after leaving the
/// history valid — interrupted tools get results, unfinished tool calls are
/// dropped from the turn, and the REPL prompt simply returns.
async fn agent_loop(cfg: &Config, history: &mut Vec<Value>, token: &CancelToken) -> Result<()> {
    let schemas: Vec<Value> = tools().iter()
        .map(|t| json!({"name": t.name, "description": t.desc, "input_schema": t.schema}))
        .collect();
    for turn in 0..cfg.max_turns {
        if token.is_cancelled() { return Err(Error::Interrupted); }
        if fit_context(history, cfg.context_size) {
            eprintln!("{}", paint(" trimmed history to fit --context-size ", WARN_BG, false));
        }
        let (resp, interrupted) = call_api(cfg, history, &schemas, token).await?;
        let content = resp["content"].as_array().cloned().unwrap_or_default();
        if interrupted {
            // Cut off mid-response: keep finished text/thinking, drop tool
            // calls — a tool_use whose tool_result never ran would poison
            // the next request.
            let kept: Vec<Value> = content.into_iter()
                .filter(|b| b["type"] != "tool_use").collect();
            if !kept.is_empty() {
                history.push(json!({"role": "assistant", "content": kept}));
            }
            return Err(Error::Interrupted);
        }
        let calls: Vec<_> = content.iter().filter(|b| b["type"] == "tool_use").cloned().collect();
        // Echo the full assistant turn back so interleaved thinking stays continuous —
        // including the final text-only turn, so follow-up questions keep context.
        if !content.is_empty() {
            history.push(json!({"role": "assistant", "content": content}));
        }
        if calls.is_empty() { return Ok(()); }

        // Run the tools. Cancellation stops the batch: the interrupted call
        // reports as much as it got, the rest are skipped, and every tool_use
        // still leaves with its tool_result.
        let mut results = Vec::with_capacity(calls.len());
        let mut stopped = false;
        for c in &calls {
            let name = c["name"].as_str().unwrap_or("");
            let out = if stopped {
                "skipped: this run was interrupted before the tool ran".into()
            } else {
                run_tool(name, &c["input"], token).await
            };
            let out = truncate(&out);
            print_tool_call(name, &c["input"], &out);
            results.push(json!({"type": "tool_result", "tool_use_id": c["id"], "content": out}));
            if token.is_cancelled() { stopped = true; }
        }
        history.push(json!({"role": "user", "content": results}));
        if stopped { return Err(Error::Interrupted); }

        if turn + 1 == cfg.max_turns {
            eprintln!("{}", paint(&format!(" warning: hit --max-turns={}, stopping ", cfg.max_turns), WARN_BG, false));
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
fn est_tokens(history: &[Value]) -> u64 {
    serde_json::to_string(history).map_or(0, |s| (s.len() / 3) as u64)
}

/// Shrink history until the estimate fits `limit`, trimming the oldest
/// tool_result first so tool_use/result pairing stays valid.
fn fit_context(history: &mut Vec<Value>, limit: u64) -> bool {
    let mut changed = false;
    while est_tokens(history) > limit {
        let Some((mi, bi)) = history.iter().enumerate().find_map(|(mi, m)| {
            (m["role"] == "user").then_some(m["content"].as_array()).flatten()
                .and_then(|a| a.iter().enumerate().find(|(_, b)| b["type"] == "tool_result").map(|(bi, _)| (mi, bi)))
        }) else { break };
        let slot = &mut history[mi]["content"][bi];
        let s = slot["content"].as_str().unwrap_or("");
        if s.chars().count() > 200 {
            let cut: String = s.chars().take(200).collect();
            slot["content"] = json!(format!("{cut}…[trimmed to fit context]"));
            changed = true;
        } else if history.len() > 1 {
            history.remove(mi);
            changed = true;
        } else { break; }
    }
    changed
}

// ---- api: shared dispatch --------------------------------------------------

async fn call_api(cfg: &Config, history: &[Value], schemas: &[Value], token: &CancelToken) -> Result<(Value, bool)> {
    let messages = if cfg.thinking == Thinking::Strip { strip_thinking(history) } else { history.to_vec() };
    let (v, interrupted) = match (cfg.protocol, cfg.streaming) {
        (Protocol::Anthropic, false) => anthropic_blocking(cfg, &messages, schemas, token).await?,
        (Protocol::Anthropic, true) => anthropic_streaming(cfg, &messages, schemas, token).await?,
        (Protocol::OpenAI, false) => openai_blocking(cfg, &messages, schemas, token).await?,
        (Protocol::OpenAI, true) => openai_streaming(cfg, &messages, schemas, token).await?,
    };
    // Blocking paths don't print inline — emit text/thinking now.
    if !cfg.streaming {
        if let Some(blocks) = v["content"].as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => println!("\n{}", block["text"].as_str().unwrap_or("")),
                    Some("thinking") if cfg.show_thinking => {
                        eprintln!();
                        print_thinking(block["thinking"].as_str().unwrap_or(""));
                        eprintln!();
                    }
                    _ => {}
                }
            }
        }
    }
    eprintln!("[usage] in={} out={} cache_read={} cache_write={}",
        v["usage"]["input_tokens"], v["usage"]["output_tokens"],
        v["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0),
        v["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0));
    Ok((v, interrupted))
}

fn strip_thinking(messages: &[Value]) -> Vec<Value> {
    messages.iter().map(|m| match m["role"].as_str() {
        Some("assistant") => json!({"role": "assistant", "content": m["content"].as_array()
            .map(|a| a.iter().filter(|b| b["type"] != "thinking").cloned().collect::<Vec<_>>())
            .unwrap_or_default()}),
        _ => m.clone(),
    }).collect()
}

fn http() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(180))
        .timeout_write(Duration::from_secs(60))
        .build()
}

fn post(api_key: &str, url: String, body: Value, anthropic: bool) -> Result<ureq::Response> {
    let mut req = http().post(&url).set("Authorization", &format!("Bearer {api_key}"));
    if anthropic {
        req = req.set("x-api-key", api_key).set("anthropic-version", "2023-06-01");
    }
    Ok(req.send_json(body)?)
}

/// SSE lines from a response, newlines stripped — produced on a plain reader
/// thread and handed to the agent coroutine over a channel. A blocking
/// socket can't suspend, so the socket gets its own thread and the
/// *consumer* is the coroutine: cancellation just drops the receiver, and
/// the thread winds down at its next read. Interrupting never waits on the
/// network.
fn sse_channel(resp: ureq::Response) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(64);
    std::thread::spawn(move || {
        let mut reader = BufReader::new(resp.into_reader());
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,                       // EOF or broken stream
                Ok(_) => {
                    let l = line.trim_end_matches(['\r', '\n']).to_string();
                    if tx.blocking_send(l).is_err() { break; } // receiver gone: cancelled
                }
            }
        }
    });
    rx
}

/// Run a blocking computation on the runtime's blocking pool so the agent
/// coroutine stays suspensible: the work races against cancellation, and on
/// Ctrl-C nothing waits for the (uninterruptible) syscall — the pool thread
/// finishes on its own and its answer is simply discarded.
async fn blocking<T, F>(token: &CancelToken, f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::select! {
        res = tokio::task::spawn_blocking(f) => {
            res.unwrap_or_else(|e| Err(Error::Msg(format!("worker: {e}"))))
        }
        _ = token.cancelled() => Err(Error::Interrupted),
    }
}

fn empty_usage() -> Value {
    json!({"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
}

fn merge_usage(usage: &mut Value, src: Option<&serde_json::Map<String, Value>>) {
    if let (Some(dst), Some(src)) = (usage.as_object_mut(), src) {
        for (k, v) in src { dst.insert(k.clone(), v.clone()); }
    }
}

fn append_str_field(block: &mut Value, key: &str, tail: &str) {
    if let Some(o) = block.as_object_mut() {
        let cur = o.get(key).and_then(Value::as_str).unwrap_or("").to_string();
        o.insert(key.into(), json!(cur + tail));
    }
}

/// Blank line separating the prompt from the agent's reply — printed once.
fn lead_nl(lead: &mut bool, stderr: bool) {
    if !*lead {
        if stderr { eprintln!() } else { println!() }
        *lead = true;
    }
}

// ---- api: anthropic wire ---------------------------------------------------

fn anthropic_body(cfg: &Config, messages: &[Value], schemas: &[Value], stream: bool) -> Value {
    let active = cfg.cache == CacheMode::Active;
    let system = if active {
        json!([{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}])
    } else {
        json!(SYSTEM)
    };
    let mut tools = schemas.to_vec();
    if active {
        if let Some(last) = tools.last_mut() { last["cache_control"] = json!({"type": "ephemeral"}); }
    }
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "system": system, "tools": tools, "messages": messages});
    body["stream"] = json!(stream);
    body
}

async fn anthropic_blocking(cfg: &Config, messages: &[Value], schemas: &[Value], token: &CancelToken) -> Result<(Value, bool)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = anthropic_body(cfg, messages, schemas, false);
    let key = cfg.api_key.clone();
    let v = blocking(token, move || -> Result<Value> {
        let v: Value = post(&key, url, body, true)?.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    Ok((v, token.is_cancelled()))
}

async fn anthropic_streaming(cfg: &Config, messages: &[Value], schemas: &[Value], token: &CancelToken) -> Result<(Value, bool)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = anthropic_body(cfg, messages, schemas, true);
    let key = cfg.api_key.clone();
    let resp = blocking(token, move || post(&key, url, body, true)).await?;
    let (mut content, mut usage, mut lead, mut thinking_open, mut text_open) =
        (vec![], empty_usage(), false, false, false);
    let mut interrupted = false;
    let mut lines = sse_channel(resp);
    // Consume the stream as a coroutine: each line is one suspension point,
    // raced against cancellation — Ctrl-C lands between two deltas, and the
    // partial answer assembled so far is kept rather than lost.
    while let Some(line) = tokio::select! {
        biased;
        line = lines.recv() => line,
        _ = token.cancelled() => { interrupted = true; None }
    } {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" { break }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        match v["type"].as_str() {
            Some("message_start") => merge_usage(&mut usage, v["message"]["usage"].as_object()),
            Some("content_block_start") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                while content.len() <= i { content.push(Value::Null); }
                content[i] = v["content_block"].clone();
            }
            Some("content_block_delta") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                let (Some(block), d) = (content.get_mut(i), &v["delta"]) else { continue };
                match (block["type"].as_str(), d["type"].as_str()) {
                    (Some("text"), Some("text_delta")) => {
                        if let Some(t) = d["text"].as_str() {
                            if thinking_open { println!(); thinking_open = false; } // close the thinking line
                            lead_nl(&mut lead, false);
                            print_text(t); let _ = io::stdout().flush();
                            text_open = true;
                            append_str_field(block, "text", t);
                        }
                    }
                    (Some("thinking"), Some("thinking_delta")) => {
                        if let Some(t) = d["thinking"].as_str() {
                            if cfg.show_thinking {
                                lead_nl(&mut lead, true);
                                print_thinking(t); let _ = io::stderr().flush();
                                thinking_open = true;
                            }
                            append_str_field(block, "thinking", t);
                        }
                    }
                    (Some("tool_use"), Some("input_json_delta")) => {
                        if let Some(p) = d["partial_json"].as_str() {
                            let acc = block["input_json_str"].as_str().unwrap_or("").to_string() + p;
                            block["input_json_str"] = json!(acc);
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                if let Some(block) = content.get_mut(v["index"].as_u64().unwrap_or(0) as usize) {
                    if let Some(s) = block["input_json_str"].as_str() {
                        block["input"] = serde_json::from_str(s).unwrap_or(Value::Null);
                        if let Some(o) = block.as_object_mut() { o.remove("input_json_str"); }
                    }
                }
            }
            Some("message_delta") => merge_usage(&mut usage, v["usage"].as_object()),
            Some("message_stop") => break,
            Some("error") => return Err(Error::Msg(v["error"].to_string())),
            _ => {}
        }
    }
    if thinking_open || text_open { println!(); } // terminate the streamed line before [usage] lands
    content.retain(|v| !v.is_null());
    Ok((json!({"content": content, "usage": usage}), interrupted))
}

// ---- api: openai wire ------------------------------------------------------

/// Internal (Anthropic-style) history → OpenAI Chat Completions messages.
fn to_openai_messages(system: &str, history: &[Value], cfg: &Config) -> Vec<Value> {
    let mut out = vec![json!({"role": "system", "content": system})];
    for msg in history {
        match msg["role"].as_str() {
            Some("user") => match msg["content"].as_str() {
                Some(s) => out.push(json!({"role": "user", "content": s})),
                None => out.extend(msg["content"].as_array().into_iter().flatten()
                    .filter(|b| b["type"] == "tool_result")
                    .map(|b| json!({"role": "tool", "tool_call_id": b["tool_use_id"],
                                    "content": b["content"].as_str().unwrap_or("")}))),
            },
            Some("assistant") => {
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut tool_calls = vec![];
                for block in msg["content"].as_array().into_iter().flatten() {
                    match block["type"].as_str() {
                        Some("text") => text.push_str(block["text"].as_str().unwrap_or("")),
                        Some("thinking") if cfg.thinking == Thinking::Preserve => {
                            reasoning.push_str(block["thinking"].as_str().unwrap_or(""));
                        }
                        Some("tool_use") => tool_calls.push(json!({
                            "id": block["id"], "type": "function",
                            "function": {"name": block["name"], "arguments": block["input"].to_string()}})),
                        _ => {}
                    }
                }
                let mut m = json!({"role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { json!(text) }});
                if !reasoning.is_empty() { m["reasoning_content"] = json!(reasoning); }
                if !tool_calls.is_empty() { m["tool_calls"] = json!(tool_calls); }
                out.push(m);
            }
            _ => {}
        }
    }
    out
}

fn openai_tools(schemas: &[Value]) -> Value {
    json!(schemas.iter().map(|t| json!({"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}})).collect::<Vec<_>>())
}

/// OpenAI response → internal {content, usage} shape.
fn openai_to_internal(v: &Value) -> Value {
    let msg = &v["choices"][0]["message"];
    let mut content = vec![];
    if let Some(r) = msg["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
        content.push(json!({"type": "thinking", "thinking": r}));
    }
    if let Some(t) = msg["content"].as_str().filter(|t| !t.is_empty()) {
        content.push(json!({"type": "text", "text": t}));
    }
    for tc in msg["tool_calls"].as_array().into_iter().flatten() {
        content.push(json!({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
            "input": serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))}));
    }
    let u = &v["usage"];
    json!({"content": content, "usage": json!({
        "input_tokens": u["prompt_tokens"], "output_tokens": u["completion_tokens"],
        "cache_read_input_tokens": u["prompt_tokens_details"]["cached_tokens"],
        "cache_creation_input_tokens": 0})})
}

async fn openai_blocking(cfg: &Config, messages: &[Value], schemas: &[Value], token: &CancelToken) -> Result<(Value, bool)> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "messages": to_openai_messages(SYSTEM, messages, cfg), "tools": openai_tools(schemas)});
    let key = cfg.api_key.clone();
    let v = blocking(token, move || -> Result<Value> {
        let v: Value = post(&key, url, body, false)?.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    Ok((openai_to_internal(&v), token.is_cancelled()))
}

async fn openai_streaming(cfg: &Config, messages: &[Value], schemas: &[Value], token: &CancelToken) -> Result<(Value, bool)> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens, "stream": true,
        "messages": to_openai_messages(SYSTEM, messages, cfg), "tools": openai_tools(schemas)});
    let key = cfg.api_key.clone();
    let resp = blocking(token, move || post(&key, url, body, false)).await?;
    let (mut text, mut reasoning, mut tool_calls) = (String::new(), String::new(), vec![]);
    let (mut usage, mut lead, mut thinking_open, mut text_open) =
        (empty_usage(), false, false, false);
    let mut interrupted = false;
    let mut lines = sse_channel(resp);
    // Same contract as the anthropic consumer: one suspension point per line,
    // raced against cancellation, partial answer kept on interrupt.
    while let Some(line) = tokio::select! {
        biased;
        line = lines.recv() => line,
        _ = token.cancelled() => { interrupted = true; None }
    } {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" { break }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if v["usage"].is_object() {
            usage["input_tokens"] = v["usage"]["prompt_tokens"].clone();
            usage["output_tokens"] = v["usage"]["completion_tokens"].clone();
            usage["cache_read_input_tokens"] = v["usage"]["prompt_tokens_details"]["cached_tokens"].clone();
        }
        let Some(delta) = v["choices"][0]["delta"].as_object() else { continue };
        let delta = Value::Object(delta.clone());
        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            if thinking_open { println!(); thinking_open = false; }
            lead_nl(&mut lead, false);
            print_text(t); let _ = io::stdout().flush();
            text_open = true;
            text.push_str(t);
        }
        if let Some(r) = delta["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
            if cfg.show_thinking {
                lead_nl(&mut lead, true);
                print_thinking(r); let _ = io::stderr().flush();
                thinking_open = true;
            }
            reasoning.push_str(r);
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let i = tc["index"].as_u64().unwrap_or(0) as usize;
            while tool_calls.len() <= i { tool_calls.push(json!({"id": "", "name": "", "arguments": ""})); }
            let slot = &mut tool_calls[i];
            if let Some(id) = tc["id"].as_str().filter(|s| !s.is_empty()) { slot["id"] = json!(id); }
            if let Some(n) = tc["function"]["name"].as_str().filter(|s| !s.is_empty()) { slot["name"] = json!(n); }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                let acc = slot["arguments"].as_str().unwrap_or("").to_string() + a;
                slot["arguments"] = json!(acc);
            }
        }
    }
    if thinking_open || text_open { println!(); } // terminate the streamed line before [usage] lands
    let mut content = vec![];
    if !reasoning.is_empty() { content.push(json!({"type": "thinking", "thinking": reasoning})); }
    if !text.is_empty() { content.push(json!({"type": "text", "text": text})); }
    for tc in tool_calls {
        content.push(json!({"type": "tool_use", "id": tc["id"], "name": tc["name"],
            "input": serde_json::from_str::<Value>(tc["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))}));
    }
    if lead { println!(); }
    Ok((json!({"content": content, "usage": usage}), interrupted))
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn temp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("jingwei_test_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Drive a coroutine to completion on its own little scheduler.
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all().build().unwrap().block_on(f)
    }

    fn cfg(base: String, streaming: bool) -> Config {
        Config {
            api_key: "test-key".into(), base_url: base, model: "test-model".into(),
            protocol: Protocol::Anthropic, cache: CacheMode::Auto, thinking: Thinking::Preserve,
            max_tokens: 1024, context_size: DEFAULT_CONTEXT_SIZE, max_turns: DEFAULT_MAX_TURNS, streaming, show_thinking: false,
        }
    }

    fn mock(response: &'static str, status: u16, sse: bool) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let Ok((mut s, _)) = listener.accept() else { return };
            let mut buf = [0u8; 8192];
            let _ = s.read(&mut buf);
            let (ctype, body) = if sse {
                ("text/event-stream", format!("{}\n\n", response))
            } else {
                ("application/json", response.to_string())
            };
            let resp = format!("HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            let _ = s.write_all(resp.as_bytes());
            let _ = s.flush();
            std::thread::sleep(std::time::Duration::from_millis(20));
        });
        port
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
    fn mock_stall(first: &'static str, stalled: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for (i, body) in [first, stalled].into_iter().enumerate() {
                let Ok((mut s, _)) = listener.accept() else { return };
                let mut buf = [0u8; 8192];
                let _ = s.read(&mut buf);
                let head = if i == 0 {
                    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())
                } else {
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_string()
                };
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(body.as_bytes());
                let _ = s.flush();
                if i == 1 {
                    std::thread::sleep(std::time::Duration::from_secs(10)); // stall: keep the stream open
                }
            }
        });
        port
    }

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
    fn tool_call_display_shows_input_and_caps_output() {
        // header echoes the arguments, body echoes the output
        let lines = tool_call_lines("bash", &json!({"command": "echo hi"}), "exit=0\nhi");
        assert_eq!(lines[0], "[bash] $ echo hi");
        assert!(lines.contains(&"│ exit=0".to_string()));
        assert!(lines.contains(&"│ hi".to_string()));
        // per-tool summaries
        assert_eq!(tool_call_lines("write_file",
            &json!({"path": "a.txt", "content": "hello"}), "ok: wrote 5 bytes to a.txt")[0],
            "[write_file] a.txt (5 bytes)");
        assert_eq!(tool_call_lines("read_file", &json!({"path": "b.txt"}), "x")[0], "[read_file] b.txt");
        // long output is capped with a marker
        let big: String = (0..MAX_TOOL_DISPLAY_LINES + 5).map(|i| format!("line{i}\n")).collect();
        let lines = tool_call_lines("bash", &json!({"command": "cat big"}), &big);
        assert_eq!(lines.len(), MAX_TOOL_DISPLAY_LINES + 2);
        assert_eq!(lines.last().unwrap(), "… +5 more lines");
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
        // edit ambiguous
        fs::write(&p, "ab ab").unwrap();
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
        parse_from(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn flags_parse_and_validate() {
        let a = flags(&["--base-url", "https://x/v1", "--api-key", "k", "-m", "m1",
            "--protocol", "openai", "--cache", "auto", "--thinking", "strip",
            "--max-tokens", "4096", "--context-size", "100000", "--max-turns", "5",
            "-S", "do", "stuff"]).unwrap();
        assert_eq!(a.prompt, vec!["do", "stuff"]);
        assert!(!a.streaming);
        let cfg = build_config(&a).unwrap();
        assert_eq!(cfg.protocol, Protocol::OpenAI);
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
        // active cache is anthropic-only
        assert!(build_config(&flags(&["--api-key", "k", "--base-url", "https://x", "-m", "m",
            "--protocol", "openai", "--cache", "active"]).unwrap()).is_err());
        // required fields
        assert!(build_config(&flags(&[]).unwrap()).is_err());
        assert!(build_config(&flags(&["--api-key", "k"]).unwrap()).is_err());
        assert!(build_config(&flags(&["--api-key", "k", "--base-url", "https://x"]).unwrap()).is_err());
        // bad flags
        assert!(flags(&["--nope"]).is_err());
        assert!(flags(&["--max-tokens", "abc"]).is_err());
        assert!(flags(&["--max-turns", "abc"]).is_err());
        assert!(flags(&["--base-url"]).is_err());
        assert!(flags(&["--context-size"]).is_err());
        assert!(flags(&["-s"]).unwrap().streaming); // -s is the explicit default
    }

    #[test]
    fn enum_flags_case_insensitive_and_list_available_values() {
        let parse = |extra: &[&str]| parse_from(
            ["--api-key", "k", "--base-url", "https://x", "-m", "m"].iter()
                .chain(extra.iter()).map(|s| s.to_string())).unwrap();
        // spellings are case-insensitive in every position
        assert_eq!(build_config(&parse(&["--protocol", "OPENAI"])).unwrap().protocol, Protocol::OpenAI);
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
    fn strip_thinking_keeps_everything_else() {
        let history = vec![json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "hello"}]})];
        let stripped = strip_thinking(&history);
        assert_eq!(stripped[0]["content"].as_array().unwrap().len(), 1);
        assert_eq!(history[0]["content"].as_array().unwrap().len(), 2); // untouched
    }

    #[test]
    fn fit_context_trims_oldest_tool_result_and_keeps_pairing() {
        let big = "x".repeat(10_000);
        let mut history = vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]}),
        ];
        assert!(fit_context(&mut history, 500));
        assert!(history[2]["content"][0]["content"].as_str().unwrap().contains("trimmed"));
        assert_eq!(history[1]["content"][0]["id"], history[2]["content"][0]["tool_use_id"]);
        let mut tiny = vec![json!({"role": "user", "content": "t"})];
        assert!(!fit_context(&mut tiny, 10_000));
    }

    #[test]
    fn fit_context_drops_short_tool_result_messages_when_still_over() {
        let mut history = vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}), // too short to trim in place
            json!({"role": "assistant", "content": [{"type": "text", "text": "padding to keep the estimate high"}]}),
        ];
        let limit = est_tokens(&history) - 1; // guarantee the first pass is over
        assert!(fit_context(&mut history, limit));
        assert_eq!(history.len(), 3); // the whole tool_result message is gone
        assert!(!serde_json::to_string(&history).unwrap().contains("tool_result"));
    }

    // ---- openai wire conversion ----

    #[test]
    fn openai_conversion_maps_all_message_shapes() {
        let history = vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reason away"},
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "exit=0"}]}),
        ];
        let msgs = to_openai_messages("sys", &history, &cfg("https://x".into(), false));
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[2]["reasoning_content"], json!("reason away"));
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["arguments"], json!({"command": "ls"}).to_string());
        assert_eq!(msgs[3]["role"], json!("tool"));
        assert_eq!(msgs[3]["tool_call_id"], json!("t1"));
        // strip mode drops reasoning
        let mut cfg = cfg("https://x".into(), false);
        cfg.thinking = Thinking::Strip;
        let msgs = to_openai_messages("sys", &history, &cfg);
        assert!(msgs[2].get("reasoning_content").is_none());
    }

    #[test]
    fn openai_conversion_assistant_shapes_and_tool_schemas() {
        // text-only assistant keeps a plain string content
        let history = vec![json!({"role": "assistant", "content": [{"type": "text", "text": "hi"}]})];
        let msgs = to_openai_messages("sys", &history, &cfg("https://x".into(), false));
        assert_eq!(msgs[1]["content"], json!("hi"));
        assert!(msgs[1].get("tool_calls").is_none());
        // text-less assistant sends null content, not ""
        let history = vec![json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]})];
        let msgs = to_openai_messages("sys", &history, &cfg("https://x".into(), false));
        assert_eq!(msgs[1]["content"], Value::Null);
        assert!(msgs[1]["tool_calls"].is_array());
        // internal tool schema → OpenAI function spec
        let schemas = vec![json!({"name": "bash", "description": "run", "input_schema":
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}})];
        let t = openai_tools(&schemas);
        assert_eq!(t[0]["type"], json!("function"));
        assert_eq!(t[0]["function"]["name"], json!("bash"));
        assert_eq!(t[0]["function"]["description"], json!("run"));
        assert_eq!(t[0]["function"]["parameters"], schemas[0]["input_schema"]);
    }

    #[test]
    fn openai_response_normalizes_to_internal_shape() {
        let v = json!({"choices": [{"message": {
            "reasoning_content": "thinking", "content": "answer",
            "tool_calls": [{"id": "t9", "type": "function",
                "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 4}}});
        let n = openai_to_internal(&v);
        assert_eq!(n["content"][0]["type"], json!("thinking"));
        assert_eq!(n["content"][1]["text"], json!("answer"));
        assert_eq!(n["content"][2]["input"]["command"], json!("ls"));
        assert_eq!(n["usage"]["cache_read_input_tokens"], json!(4));
    }

    #[test]
    fn openai_streaming_assembles_deltas() {
        let events = concat!(
            r#"data: {"choices":[{"delta":{"content":"hel"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#, "\r\n\r\n", // CRLF is legal SSE framing
            r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"bash","arguments":"{\"comm"}}]}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"echo hi\"}"}}]}}]}"#, "\n\n",
            r#"data: {"usage":{"prompt_tokens":9,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}"#, "\n\n",
            r#"data: [DONE]"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.protocol = Protocol::OpenAI;
        c.streaming = true;
        let (resp, cut) = block_on(call_api(&c, &[json!({"role": "user", "content": "hey"})], &[], &CancelToken::new())).unwrap();
        assert!(!cut);
        assert_eq!(resp["content"][0], json!({"type": "thinking", "thinking": "think"}));
        assert_eq!(resp["content"][1], json!({"type": "text", "text": "hello"}));
        assert_eq!(resp["content"][2]["type"], json!("tool_use"));
        assert_eq!(resp["content"][2]["id"], json!("t1"));
        assert_eq!(resp["content"][2]["input"]["command"], json!("echo hi"));
        assert_eq!(resp["usage"]["input_tokens"], json!(9));
        assert_eq!(resp["usage"]["output_tokens"], json!(4));
        assert_eq!(resp["usage"]["cache_read_input_tokens"], json!(2));
    }

    #[test]
    fn anthropic_body_marks_cache_breakpoints_only_when_active() {
        let schemas = vec![json!({"name": "a"}), json!({"name": "b"}), json!({"name": "c"})];
        let mut c = cfg("https://x".into(), true);
        c.cache = CacheMode::Active;
        let body = anthropic_body(&c, &[], &schemas, true);
        assert_eq!(body["system"][0]["text"], json!(SYSTEM));
        assert_eq!(body["system"][0]["cache_control"]["type"], json!("ephemeral"));
        let tools = body["tools"].as_array().unwrap();
        assert!(tools[..2].iter().all(|t| t.get("cache_control").is_none()), "only the last tool is a breakpoint");
        assert_eq!(tools[2]["cache_control"]["type"], json!("ephemeral"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["model"], json!("test-model"));
        assert_eq!(body["max_tokens"], json!(1024));
        // auto mode: plain string system, no breakpoints anywhere
        let mut c = cfg("https://x".into(), false);
        c.cache = CacheMode::Auto;
        let body = anthropic_body(&c, &[], &schemas, false);
        assert_eq!(body["system"], json!(SYSTEM));
        assert!(body["tools"].as_array().unwrap().iter().all(|t| t.get("cache_control").is_none()));
        assert_eq!(body["stream"], json!(false));
    }

    #[test]
    fn wire_paths_and_auth_headers_per_protocol() {
        // anthropic: /v1/messages with x-api-key + anthropic-version + bearer
        let (port, reqs) = mock_seq(vec![(200, r#"{"content":[],"usage":{}}"#.into())]);
        block_on(call_api(&cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new())).unwrap();
        let req = reqs.lock().unwrap()[0].to_lowercase();
        assert!(req.contains("post /v1/messages http/1.1"), "{req}");
        assert!(req.contains("x-api-key: test-key"), "{req}");
        assert!(req.contains("anthropic-version: 2023-06-01"), "{req}");
        assert!(req.contains("authorization: bearer test-key"), "{req}");
        // trailing slash in base_url must not double the path
        let (port, reqs) = mock_seq(vec![(200, r#"{"content":[],"usage":{}}"#.into())]);
        block_on(call_api(&cfg(format!("http://127.0.0.1:{port}/"), false), &[], &[], &CancelToken::new())).unwrap();
        assert!(reqs.lock().unwrap()[0].contains("POST /v1/messages HTTP/1.1"));
        // openai: /chat/completions with bearer only
        let (port, reqs) = mock_seq(vec![(200, r#"{"choices":[{"message":{"content":"ok"}}]}"#.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::OpenAI;
        block_on(call_api(&c, &[json!({"role": "user", "content": "hi"})], &[], &CancelToken::new())).unwrap();
        let req = reqs.lock().unwrap()[0].to_lowercase();
        assert!(req.contains("post /chat/completions http/1.1"), "{req}");
        assert!(req.contains("authorization: bearer test-key"), "{req}");
        assert!(!req.contains("x-api-key:"), "{req}");
    }

    // ---- api over local mocks ----

    #[test]
    fn anthropic_blocking_roundtrip_and_error() {
        let port = mock(r#"{"content":[{"type":"text","text":"hi from mock"}],"usage":{"input_tokens":7,"output_tokens":3}}"#, 200, false);
        let (resp, cut) = block_on(call_api(&cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new())).unwrap();
        assert_eq!(resp["content"][0]["text"], "hi from mock");
        assert!(!cut);
        let port = mock(r#"{"error":{"message":"bad model"}}"#, 400, false);
        let err = block_on(call_api(&cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new())).unwrap_err();
        assert!(err.to_string().contains("bad model"), "got: {err}");
    }

    #[test]
    fn openai_blocking_roundtrip() {
        let port = mock(r#"{"choices":[{"message":{"content":"mock says hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":1}}}"#, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::OpenAI;
        let (resp, _) = block_on(call_api(&c, &[json!({"role": "user", "content": "hey"})], &[], &CancelToken::new())).unwrap();
        assert_eq!(resp["content"][0]["text"], "mock says hi");
        assert_eq!(resp["usage"]["cache_read_input_tokens"], json!(1));
    }

    #[test]
    fn anthropic_streaming_assembles_blocks() {
        let events = concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello world"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":1}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"comma"}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"nd\":\"ls\"}"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":2}"#, "\n\n",
            r#"data: {"type":"message_delta","usage":{"output_tokens":2}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.streaming = true;
        let (resp, cut) = block_on(call_api(&c, &[], &[], &CancelToken::new())).unwrap();
        assert!(!cut);
        assert_eq!(resp["content"][0]["thinking"], json!("step"));
        assert_eq!(resp["content"][1]["text"], json!("hello world"));
        assert_eq!(resp["content"][2]["input"]["command"], json!("ls"));
        assert_eq!(resp["usage"]["output_tokens"], json!(2));
    }

    #[test]
    fn anthropic_streaming_surfaces_error_events() {
        let events = concat!(
            r#"data: {"type":"error","error":{"type":"overloaded","message":"server overloaded"}}"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.streaming = true;
        assert!(block_on(call_api(&c, &[], &[], &CancelToken::new())).unwrap_err().to_string().contains("overloaded"));
    }

    // ---- agent loop ----

    #[test]
    fn agent_loop_runs_tool_then_stops_on_plain_text() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"echo hi"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let done = r#"{"content":[{"type":"text","text":"done"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, reqs) = mock_seq(vec![(200, tool.into()), (200, done.into())]);
        let mut history = vec![json!({"role": "user", "content": "run echo"})];
        block_on(agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &CancelToken::new())).unwrap();
        assert_eq!(history.len(), 4); // user + assistant(tool_use) + user(tool_result) + assistant(text)
        assert_eq!(history[1]["role"], json!("assistant"));
        assert_eq!(history[1]["content"][0]["id"], json!("t1")); // full turn echoed back, thinking intact
        assert_eq!(history[2]["content"][0]["type"], json!("tool_result"));
        assert_eq!(history[2]["content"][0]["tool_use_id"], json!("t1"));
        assert!(history[2]["content"][0]["content"].as_str().unwrap().contains("hi"));
        assert_eq!(history[3]["content"][0]["text"], json!("done")); // final answer stays in history for follow-ups
        let reqs = reqs.lock().unwrap();
        assert!(reqs[0].contains("input_schema"), "tool schemas must be sent");
        assert!(reqs[1].contains("\"type\":\"tool_result\""), "results must feed the next request");
    }

    #[test]
    fn agent_loop_stops_at_max_turns() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"true"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, _) = mock_seq(vec![(200, tool.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.max_turns = 1;
        let mut history = vec![json!({"role": "user", "content": "go"})];
        block_on(agent_loop(&c, &mut history, &CancelToken::new())).unwrap(); // must return instead of spinning
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
    }

    // ---- coroutine cancellation ----

    #[test]
    fn cancel_token_flips_and_wakes_a_suspended_coroutine() {
        block_on(async {
            let token = Arc::new(CancelToken::new());
            assert!(!token.is_cancelled());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                t.cancel();
            });
            token.cancelled().await; // suspends until the other thread cancels
            assert!(token.is_cancelled());
            token.cancelled().await; // already cancelled: resolves immediately
        });
    }

    #[test]
    fn bash_coroutine_is_killed_by_cancellation() {
        let cmd = if cfg!(windows) {
            "echo started & ping -n 30 127.0.0.1 > nul"
        } else {
            "echo started; sleep 30"
        };
        let start = std::time::Instant::now();
        let out = block_on(async {
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(300));
                t.cancel();
            });
            run_bash(cmd, &token).await
        });
        assert!(out.contains("[interrupted by user]"), "got: {out}");
        assert!(out.contains("started"), "partial output must survive: {out}");
        assert!(start.elapsed() < std::time::Duration::from_secs(10),
            "a 30s command was cancelled; took {:?}", start.elapsed());
    }

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
        let mut history = vec![json!({"role": "user", "content": "go"})];
        let err = block_on(async {
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), true), &mut history, &token).await
        }).unwrap_err();
        assert!(matches!(err, Error::Interrupted), "got: {err}");
        // user + assistant(tool_use) + user(tool_result) + assistant(partial text; tool_use dropped)
        assert_eq!(history.len(), 4);
        assert_eq!(history[3]["role"], json!("assistant"));
        assert_eq!(history[3]["content"][0]["type"], json!("text"));
        assert_eq!(history[3]["content"][0]["text"], json!("partial answ"));
        assert!(history[3]["content"].as_array().unwrap().iter().all(|b| b["type"] != "tool_use"));
        // every tool_use still has its tool_result — the history stays sendable
        let ids: Vec<&str> = history[1]["content"].as_array().unwrap().iter()
            .filter(|b| b["type"] == "tool_use").map(|b| b["id"].as_str().unwrap()).collect();
        let results: Vec<&str> = history[2]["content"].as_array().unwrap().iter()
            .map(|b| b["tool_use_id"].as_str().unwrap()).collect();
        assert!(ids.iter().all(|i| results.contains(i)));
    }

    #[test]
    fn agent_loop_interrupted_mid_tool_finishes_pairing() {
        let cmd = if cfg!(windows) { "ping -n 30 127.0.0.1 > nul" } else { "sleep 30" };
        let tool = format!(
            r#"{{"content":[{{"type":"tool_use","id":"t1","name":"bash","input":{{"command":"{cmd}"}}}}],"usage":{{"input_tokens":1,"output_tokens":1}}}}"#);
        let (port, _) = mock_seq(vec![(200, tool)]);
        let mut history = vec![json!({"role": "user", "content": "go"})];
        let err = block_on(async {
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &token).await
        }).unwrap_err();
        assert!(matches!(err, Error::Interrupted), "got: {err}");
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
        let result = history[2]["content"][0]["content"].as_str().unwrap();
        assert!(result.contains("[interrupted by user]"), "got: {result}");
        assert_eq!(history[2]["content"][0]["tool_use_id"], json!("t1"));
    }

    /// Regression: a grandchild holding the pipes (backgrounded process,
    /// daemon) must not be able to wedge the coroutine when the user
    /// interrupts. Runs the coroutine on its own thread and cancels from
    /// outside; a hang is reported by the timeout, not the test runner.
    #[test]
    fn bash_cancel_with_grandchild_holding_pipe() {
        // sh exits immediately; the backgrounded sleep keeps stdout open.
        let cmd = if cfg!(windows) {
            "echo started & start /b ping -n 30 127.0.0.1 > nul"
        } else {
            "sleep 30 & echo started"
        };
        let token = Arc::new(CancelToken::new());
        let t = token.clone();
        let (tx, done) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all().build().unwrap();
            let out = rt.block_on(run_bash(cmd, &t));
            let _ = tx.send(out);
        });
        std::thread::sleep(Duration::from_millis(500)); // let the child exit
        token.cancel();                                  // the user hits Ctrl-C
        match done.recv_timeout(Duration::from_secs(5)) {
            Ok(out) => {
                assert!(out.contains("started"), "got: {out}");
                assert!(out.contains("[interrupted by user]"), "got: {out}");
            }
            Err(_) => panic!("coroutine wedged: cancellation never reached run_bash — \
                collect() must suspend, not block the scheduler thread"),
        }
    }

    #[test]
    fn error_display() {
        assert_eq!(Error::Api(429, "limited".into()).to_string(), "api 429: limited");
        assert_eq!(Error::Msg("bad".into()).to_string(), "bad");
        assert_eq!(Error::Interrupted.to_string(), "interrupted");
    }
}
