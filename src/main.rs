// jingwei — 精卫填海. A minimal coding agent: speak a task, and a small agent
// carries stones — one tool call at a time — until the sea is land.
//
// Wire protocols: the Messages wire (minimax — whose compatible endpoint is
// the recommended one, thinking blocks, interleaved reasoning, cache_control),
// the Chat Completions dialect (zai, deepseek — each with a thinking story of
// its own). Vendors that live at one address name it themselves; everything
// else you bring the endpoint for.
//
// Env: JINGWEI_API_KEY, JINGWEI_BASE_URL, JINGWEI_MODEL, JINGWEI_PROTOCOL,
//      JINGWEI_CACHE, JINGWEI_THINKING, JINGWEI_NO_TUI, NO_COLOR

mod display;
mod ir;
mod plain;
mod session;
mod tui;

use display::{Msg, Sev, Show, Usage};
use ir::{Block, Message, Response, ToolResult};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::future::Future;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::{mpsc, Notify};

const MAX_TOOL_OUTPUT: usize = 50_000;
const DEFAULT_MAX_TOKENS: u32 = 131_072;
const DEFAULT_CONTEXT_SIZE: u64 = 1_000_000;
const DEFAULT_MAX_TURNS: u32 = 60;
/// Version of the internal history payload as it lands in session files —
/// owned here, beside the shape it versions; session.rs embeds it in the
/// header as an opaque integer and never interprets it. Bump it the day
/// the payload's *semantics* change; older files migrate at load.
const HISTORY_FORMAT: u32 = 1;
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

// ---- colors --------------------------------------------------------------
// (the gate, the palette, and the painter all live in display.rs — the
// agent core below never touches a terminal)

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

/// Echo a tool call through the display port: the frontends decide how
/// much of the output tail to show (and how to fold the rest).
fn print_tool_call(name: &str, input: &Value, output: &str, sink: &dyn Show) {
    sink.show(Msg::Tool { name: name.into(), summary: tool_summary(name, input), output: output.into() });
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
                    _ = token.cancelled() => {
                        // salvage what lands within the grace, abandon the rest
                        tokio::time::timeout(PIPE_GRACE, self.rx.recv()).await.unwrap_or_default()
                    }
                }
            } else {
                tokio::time::timeout(PIPE_GRACE, self.rx.recv()).await.unwrap_or_default()
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

/// The boxed future a vendor's `turn` returns. Boxed because the wire is
/// chosen at runtime — one allocation per request, and the vendor's own
/// streaming/blocking machinery stays its own.
type Turn<'a> = Pin<Box<dyn Future<Output = Result<(Response, bool)>> + Send + 'a>>;

/// The provider port. One impl per wire; the composition root holds a
/// [`Protocol`] handle and calls through it — no `match` anywhere. Adding a
/// vendor is one `impl` and one row in [`VENDORS`], both the compiler's
/// problem to keep complete.
trait Vendor: Sync {
    /// The name the user types (`--protocol`) and the session records.
    fn label(&self) -> &'static str;
    /// The endpoint the wire names for itself when the user names none — the
    /// one case where the address belongs to the vendor outright.
    fn default_base(&self) -> Option<&'static str>;
    /// The flagship the wire names for itself, when it has one.
    fn default_model(&self) -> Option<&'static str>;
    /// What this wire can serve, in its own words — the rules the
    /// composition root only *asks* about.
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()>;
    /// One turn: hand over the IR, get the IR back. `sink` is where the
    /// turn's own `Msg`s go — the vendor never learns which frontend is
    /// listening.
    fn turn<'a>(
        &'a self,
        cfg: &'a Config,
        history: &'a [Message],
        schemas: &'a [Value],
        token: &'a CancelToken,
        sink: &'a dyn Show,
    ) -> Turn<'a>;
}

/// The Messages wire (MiniMax): content blocks, tool_use/result pairs,
/// thinking with a signature, interleaved reasoning, cache_control
/// breakpoints — and its own dialect of policy words on top.
struct MiniMax;
/// Chat Completions with preserved thinking (Zhipu's zai): an always-on
/// flagship reasoner whose cache is implicit.
struct Zai;
/// Chat Completions plus a thinking toggle and a disk cache (DeepSeek).
struct DeepSeek;

static MINIMAX: MiniMax = MiniMax;
static ZAI: Zai = Zai;
static DEEPSEEK: DeepSeek = DeepSeek;

impl Vendor for MiniMax {
    fn label(&self) -> &'static str { "minimax" }
    fn default_base(&self) -> Option<&'static str> { Some("https://api.minimax.cn/anthropic") }
    fn default_model(&self) -> Option<&'static str> { Some("MiniMax-M3") }
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        minimax_accepts(cache, thinking, effort)
    }
    fn turn<'a>(&'a self, cfg: &'a Config, history: &'a [Message], schemas: &'a [Value], token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(minimax_turn(cfg, history, schemas, token, sink))
    }
}

impl Vendor for Zai {
    fn label(&self) -> &'static str { "zai" }
    fn default_base(&self) -> Option<&'static str> { Some("https://open.bigmodel.cn/api/paas/v4") }
    fn default_model(&self) -> Option<&'static str> { Some("glm-5.3-flash") }
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        zai_accepts(cache, thinking, effort)
    }
    fn turn<'a>(&'a self, cfg: &'a Config, history: &'a [Message], schemas: &'a [Value], token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(zai_turn(cfg, history, schemas, token, sink))
    }
}

impl Vendor for DeepSeek {
    fn label(&self) -> &'static str { "deepseek" }
    fn default_base(&self) -> Option<&'static str> { Some("https://api.deepseek.com") }
    fn default_model(&self) -> Option<&'static str> { None } // flash or pro is the user's call
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        deepseek_accepts(cache, thinking, effort)
    }
    fn turn<'a>(&'a self, cfg: &'a Config, history: &'a [Message], schemas: &'a [Value], token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(deepseek_turn(cfg, history, schemas, token, sink))
    }
}

/// Every wire, by the name the user types (`--protocol`, `JINGWEI_PROTOCOL`).
/// The one place a new vendor registers; the default is the first row.
const VENDORS: &[(&str, Protocol)] = &[
    ("minimax", Protocol::MINIMAX),
    ("zai", Protocol::ZAI),
    ("deepseek", Protocol::DEEPSEEK),
];

/// A handle to one wire — what used to be an enum spelling each protocol's
/// facts and dispatching by `match`, now a newtype over the trait object.
/// The three constants are the statics above; equality is by wire.
#[derive(Clone, Copy)]
struct Protocol(&'static dyn Vendor);

impl Protocol {
    const MINIMAX: Protocol = Protocol(&MINIMAX);
    const ZAI: Protocol = Protocol(&ZAI);
    const DEEPSEEK: Protocol = Protocol(&DEEPSEEK);

    fn label(self) -> &'static str { self.0.label() }
    fn default_base(self) -> Option<&'static str> { self.0.default_base() }
    fn default_model(self) -> Option<&'static str> { self.0.default_model() }
    fn accepts(self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        self.0.accepts(cache, thinking, effort)
    }
    async fn turn(self, cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
        self.0.turn(cfg, history, schemas, token, sink).await
    }
}

impl PartialEq for Protocol {
    fn eq(&self, other: &Self) -> bool { self.label() == other.label() }
}
impl Eq for Protocol {}
impl std::fmt::Debug for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.label()) }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CacheMode { Auto, Active }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Thinking { Preserve, Strip }

/// Reasoning effort, when the endpoint offers the knob. Absent (`None`)
/// means *say nothing on the wire* — the endpoint's own default rules, so
/// existing setups see byte-identical requests.
///
/// The wires spell it differently: zai/deepseek take a word
/// (`reasoning_effort`), the Messages wire a token budget
/// (`thinking.budget_tokens`). `max` is not every wire's word — each
/// vendor folds or passes the tiers its own way; on the minimax wire it maps
/// to "everything but a floor for the reply".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Effort { Low, Medium, High, Max }

impl Effort {
    fn label(self) -> &'static str {
        match self { Self::Low => "low", Self::Medium => "medium", Self::High => "high", Self::Max => "max" }
    }
    /// The Anthropic-style budget for this tier — the Messages wire's dial,
    /// clamped into its rules: at least 1024, and strictly under
    /// `max_tokens` so the reply has room (thinking enabled requires
    /// `max_tokens > budget_tokens`).
    fn budget(self, max_tokens: u32) -> u32 {
        let nominal = match self {
            Self::Low => 1024,
            Self::Medium => 8192,
            Self::High => 32768,
            Self::Max => max_tokens.saturating_sub(1024),
        };
        nominal.clamp(1024, max_tokens.saturating_sub(1024).max(1024))
    }
}

#[derive(Clone)]
struct Config {
    api_key: String,
    base_url: String,
    model: String,
    protocol: Protocol,
    cache: CacheMode,
    thinking: Thinking,
    effort: Option<Effort>,
    max_tokens: u32,
    context_size: u64,
    max_turns: u32,
    streaming: bool,
}

#[derive(Default)]
struct Args {
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    protocol: Option<String>,
    cache: Option<String>,
    thinking: Option<String>,
    effort: Option<String>,
    max_tokens: Option<u32>,
    context_size: Option<u64>,
    max_turns: Option<u32>,
    streaming: bool,
    resume: Option<String>,
    cont: bool,
    list: bool,
    all: bool,
    prompt: Vec<String>,
}

impl Args {
    /// Did the user ask for a saved conversation (`-c` or `--resume`)?
    fn resuming(&self) -> bool {
        self.cont || self.resume.is_some()
    }
}

fn parse_from<I: Iterator<Item = String>>(it: I) -> Result<Args> {
    let mut it = it.peekable();
    let mut a = Args { streaming: true, ..Default::default() };
    while let Some(arg) = it.next() {
        // `--resume`'s value is optional: take the next word only when it
        // is not itself a flag; bare `--resume` means the newest, like -c
        if arg == "--resume" {
            a.resume = it.next_if(|v| !is_flag(v));
            if a.resume.is_none() {
                a.cont = true;
            }
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
            x if is_flag(x) => {
                return Err(Error::Msg(format!("unknown flag: {x}\ntry --help")));
            }
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

fn build_config(args: &Args) -> Result<Config> {
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
    // Wire rules live with their wires: each adapter owns what it can
    // serve, and the composition root only asks — no match.
    protocol.accepts(cache, thinking, effort)?;
    let max_turns = args.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
    if max_turns == 0 {
        return Err(Error::Msg("--max-turns must be at least 1".into()));
    }
    // A protocol may name its own endpoint — the one case where the wire
    // belongs to the vendor outright. The user's word (flag, then env)
    // still wins: everything else keeps requiring --base-url, the user
    // pointing at someone's deployment of a wire.
    let base_url = args.base_url.clone()
        .or_else(|| env::var("JINGWEI_BASE_URL").ok())
        .or_else(|| protocol.default_base().map(str::to_owned))
        .ok_or_else(|| Error::Msg("missing --base-url (or env JINGWEI_BASE_URL)".into()))?;
    // A model, the same way: a vendor with one flagship names it; the rest
    // keep requiring -m/--model, the user picking the speaker.
    let model = args.model.clone()
        .or_else(|| env::var("JINGWEI_MODEL").ok())
        .or_else(|| protocol.default_model().map(str::to_owned))
        .ok_or_else(|| Error::Msg("missing -m/--model (or env JINGWEI_MODEL)".into()))?;
    Ok(Config {
        api_key: req(&args.api_key, "JINGWEI_API_KEY", "--api-key")?,
        base_url,
        model,
        protocol, cache, thinking, effort,
        max_tokens: args.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        context_size: args.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE),
        max_turns,
        streaming: args.streaming,
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
            eprintln!("{}", display::paint(&format!(" jingwei: {e} "), display::ERR_BG, true));
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
    // --list answers from the disk alone — no credentials needed
    if args.list {
        return session::print_list(args.all);
    }
    let cfg = build_config(&args)?;
    let interactive = args.prompt.is_empty();
    let (mut convo, banners) = begin_session(&args, &cfg, interactive)?;
    if interactive {
        // A terminal gets the TUI; pipes, tests, and one-shots get the log.
        return if tui::wanted() {
            tui::run(&cfg, convo, banners).await
        } else {
            plain::plain_repl(&cfg, convo, banners).await
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
    convo.persist(&sink);
    let token = CancelToken::new();
    let res = agent_turn(&cfg, &mut convo.history, &token, &sink).await;
    convo.persist(&sink);
    sink.show(Msg::TaskEnd);
    res
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
        let s = session::Session::new(&prov, HISTORY_FORMAT);
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
async fn agent_turn(cfg: &Config, history: &mut Vec<Message>, token: &CancelToken, sink: &dyn Show) -> Result<()> {
    let agent = agent_loop(cfg, history, token, sink);
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

impl Config {
    fn protocol_label(&self) -> &'static str {
        self.protocol.label()
    }

    /// The banner's identity tail: protocol · model [· effort] · base url.
    /// Effort appears only when set — the banner is where the bar's shed
    /// order sends identity when the pane narrows.
    fn identity(&self) -> String {
        let mut s = format!("{} · {}", self.protocol_label(), self.model);
        if let Some(e) = self.effort {
            s.push_str(&format!(" · effort {}", e.label()));
        }
        s.push_str(&format!(" · {}", self.base_url));
        s
    }

    /// The effort tier's label for the status bar, when one was chosen.
    fn effort_label(&self) -> Option<&'static str> {
        self.effort.map(Effort::label)
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
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
async fn agent_loop(cfg: &Config, history: &mut Vec<Message>, token: &CancelToken, sink: &dyn Show) -> Result<()> {
    let schemas: Vec<Value> = tools().iter()
        .map(|t| json!({"name": t.name, "description": t.desc, "input_schema": t.schema}))
        .collect();
    for turn in 0..cfg.max_turns {
        if token.is_cancelled() { return Err(Error::Interrupted); }
        if fit_context(history, cfg.context_size) {
            sink.show(Msg::Note { sev: Sev::Warn, text: " trimmed history to fit --context-size ".into() });
        }
        let (resp, interrupted) = call_api(cfg, history, &schemas, token, sink).await?;
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
                run_tool(name, input, token).await
            };
            let out = truncate(&out);
            print_tool_call(name, input, &out, sink);
            results.push(ToolResult { id: id.to_string(), content: out });
            if token.is_cancelled() { stopped = true; }
        }
        history.push(Message::ToolResults(results));
        if stopped { return Err(Error::Interrupted); }

        if turn + 1 == cfg.max_turns {
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

// ---- api: shared dispatch --------------------------------------------------

/// The provider port's one dispatch: one entry per wire, and nothing else.
/// Whether a wire streams or blocks, how it spells its fields, which rules
/// its history must obey — all of that is the adapter's own machinery,
/// invisible here. The core hands over the IR and gets the IR back.
async fn call_api(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    cfg.protocol.turn(cfg, history, schemas, token, sink).await
}

/// A blocking turn's events, shown once at the end — the same port the
/// streaming paths speak event-by-event as content arrives. Called by the
/// blocking paths only; a blocking socket cannot show anything sooner.
fn show_turn(resp: &Response, sink: &dyn Show) {
    for block in &resp.blocks {
        match block {
            Block::Text(t) => sink.show(Msg::Text(format!("\n{t}"))),
            Block::Thinking { text, .. } => {
                sink.show(Msg::Think(text.clone()));
                sink.show(Msg::ThinkEnd);
            }
            _ => {}
        }
    }
    sink.show(Msg::Usage(Usage::from_value(&resp.usage)));
    sink.show(Msg::Done);
}

fn strip_thinking(messages: &[Message]) -> Vec<Message> {
    messages.iter().map(|m| match m {
        Message::Assistant(blocks) => Message::Assistant(blocks.iter().filter(|b| !b.is_thinking()).cloned().collect()),
        other => other.clone(),
    }).collect()
}

fn http() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(180))
        .timeout_write(Duration::from_secs(60))
        .build()
}

/// One POST. The Messages wire authenticates with `x-api-key` and its
/// version header (Bearer rides along too); the chat-completions family
/// with Bearer alone — the one difference in plumbing between the two wire
/// families, and it lives here, below every vendor.
fn post(api_key: &str, url: String, body: Value, messages_wire: bool) -> Result<ureq::Response> {
    let mut req = http().post(&url).set("Authorization", &format!("Bearer {api_key}"));
    if messages_wire {
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

// (The old "blank line before the reply" lead is the pane's job now: the
// task echo already separates prompt from reply.)

// ---- api: minimax vendor ---------------------------------------------------
//
// MiniMax lives on the Messages wire: content blocks, tool_use/result
// pairs, thinking blocks that carry a signature, interleaved reasoning, and
// cache_control breakpoints — plus a vocabulary of its own on top, all of
// it staying here. (The Messages shape is also jingwei's internal history,
// so this adapter's response translation is the identity — the dialect
// *is* the IR. Probed live before this was written: `thinking: adaptive`
// and the Anthropic `enabled` spelling both accepted, `disabled` honored
// on M3, budget_tokens accepted beside either, and the docs' echo mandate
// — full content back every turn, thinking and signature included — is
// enforced leniently today; the adapter echoes anyway, the documented
// contract being the safe side.)

/// Policy the minimax wire cannot serve — its own rules, kept where its
/// translation lives. `--cache active` is served here (the breakpoints are
/// native: system + last tool, well under the wire's 4-breakpoint cap) and
/// rejected by the other vendors in their own words.
fn minimax_accepts(_cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
    if effort.is_some() && thinking == Thinking::Strip {
        return Err(Error::Msg("--effort needs --thinking preserve on the minimax protocol (interleaved thinking requires the history's thinking blocks, signature and all)".into()));
    }
    Ok(())
}

/// The minimax adapter's one entry into the provider port. Strip is a
/// wire rule wearing a policy flag: this wire carries thinking blocks in
/// its history verbatim, so stripping them is this adapter's job, never
/// the core's.
/// The Messages wire's response is the internal shape: its `content` blocks
/// decode straight, its `usage` is the internal ledger already.
fn response_from_value(v: &Value) -> Response {
    Response {
        blocks: v["content"].as_array().map(|a| a.iter().map(Block::from_value).collect()).unwrap_or_default(),
        usage: v["usage"].clone(),
    }
}

async fn minimax_turn(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let messages: Vec<Message> = if cfg.thinking == Thinking::Strip { strip_thinking(history) } else { history.to_vec() };
    if cfg.streaming {
        minimax_streaming(cfg, &messages, schemas, token, sink).await
    } else {
        minimax_blocking(cfg, &messages, schemas, token, sink).await
    }
}

fn minimax_body(cfg: &Config, messages: &[Message], schemas: &[Value], stream: bool) -> Value {
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
    let wire: Vec<Value> = messages.iter().map(Message::to_value).collect();
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "system": system, "tools": tools, "messages": wire});
    body["stream"] = json!(stream);
    // Thinking is this vendor's dial, spelled its way: `adaptive` turns it
    // on (M3 ships thinking *off* by default — an agent wants it on),
    // `disabled` is the honest strip (no reasoning generated at all, so
    // nothing needs echoing back), and an effort tier rides the Messages
    // budget beside the toggle — accepted on the live wire, probe-verified.
    body["thinking"] = match (cfg.thinking, cfg.effort) {
        (Thinking::Strip, _) => json!({"type": "disabled"}),
        (Thinking::Preserve, Some(e)) => json!({"type": "adaptive", "budget_tokens": e.budget(cfg.max_tokens)}),
        (Thinking::Preserve, None) => json!({"type": "adaptive"}),
    };
    body
}

async fn minimax_blocking(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = minimax_body(cfg, messages, schemas, false);
    let key = cfg.api_key.clone();
    let v = blocking(token, move || -> Result<Value> {
        let v: Value = post(&key, url, body, true)?.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    let resp = response_from_value(&v);
    show_turn(&resp, sink);
    Ok((resp, token.is_cancelled()))
}

async fn minimax_streaming(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = minimax_body(cfg, messages, schemas, true);
    let key = cfg.api_key.clone();
    let resp = blocking(token, move || post(&key, url, body, true)).await?;
    // Blocks arrive one at a time, indexed; a tool_use's arguments stream as
    // partial JSON, so they accumulate in `tool_json` beside the typed block
    // until the block closes and the assembled JSON parses into `input`.
    let mut blocks: Vec<Option<Block>> = vec![];
    let mut tool_json: Vec<String> = vec![];
    let mut usage = empty_usage();
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
            Some("message_start") => {
                merge_usage(&mut usage, v["message"]["usage"].as_object());
                sink.show(Msg::Usage(Usage::from_value(&v["message"]["usage"]))); // the bar jumps at stream start
            }
            Some("content_block_start") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                while blocks.len() <= i { blocks.push(None); tool_json.push(String::new()); }
                blocks[i] = Some(Block::from_value(&v["content_block"]));
            }
            Some("content_block_delta") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                let d = &v["delta"];
                let Some(Some(block)) = blocks.get_mut(i) else { continue };
                match (block, d["type"].as_str()) {
                    (Block::Text(t), Some("text_delta")) => {
                        if let Some(s) = d["text"].as_str() {
                            sink.show(Msg::Text(s.into()));
                            t.push_str(s);
                        }
                    }
                    (Block::Thinking { text, .. }, Some("thinking_delta")) => {
                        if let Some(s) = d["thinking"].as_str() {
                            sink.show(Msg::Think(s.into()));
                            text.push_str(s);
                        }
                    }
                    // the wire stamps a signature on the block as it closes;
                    // it rides the IR so the next request echoes the block
                    // whole — the vendor's continuity rule, signature and all
                    (Block::Thinking { signature, .. }, Some("signature_delta")) => {
                        if let Some(s) = d["signature"].as_str() {
                            *signature = Some(signature.take().unwrap_or_default() + s);
                        }
                    }
                    (Block::ToolUse { .. }, Some("input_json_delta")) => {
                        if let Some(p) = d["partial_json"].as_str() { tool_json[i].push_str(p); }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                match blocks.get_mut(i).and_then(Option::as_mut) {
                    // a thinking block closing is the fold point: its marker lands now
                    Some(Block::Thinking { .. }) => sink.show(Msg::ThinkEnd),
                    // the call's arguments are whole now: parse them into input
                    Some(Block::ToolUse { input, .. }) => {
                        *input = serde_json::from_str(&tool_json[i]).unwrap_or(Value::Null);
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                merge_usage(&mut usage, v["usage"].as_object());
                // This wire's real ledger lands here — message_start opens
                // all-zero, and input/cache arrive only in this final event
                // (probe-verified) — so the display port hears it too:
                // Usage replaces the live turn's counters wholesale, which
                // lands ctx and cache% on the bar before Done folds the
                // turn into the session totals.
                sink.show(Msg::Usage(Usage::from_value(&usage)));
                if let Some(n) = v["usage"]["output_tokens"].as_u64() {
                    sink.show(Msg::OutTokens(n)); // output count ticks while text flows
                }
            }
            Some("message_stop") => break,
            Some("error") => return Err(Error::Msg(v["error"].to_string())),
            _ => {}
        }
    }
    sink.show(Msg::Done);
    Ok((Response { blocks: blocks.into_iter().flatten().collect(), usage }, interrupted))
}

// ---- api: the chat-completions wire family --------------------------------
//
// The Chat Completions dialect is less a vendor than a *lingua franca*:
// several providers speak its request/response/SSE shapes and differ only
// in the field vocabularies layered on top (thinking objects, effort
// spellings, usage ledgers). This section is that shared dialect — and
// deliberately nameless: no vendor appears here, so no vendor's details
// can leak into another's. The vendors (zai, deepseek, …) are thin
// sections below, each owning its body extras, its usage translation, and
// its rules.

/// Where this family posts. Every speaker of the dialect agrees on the
/// path; only the host varies.
fn chat_url(cfg: &Config) -> String {
    format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'))
}

/// Internal history → the dialect's messages. Reasoning rides assistant
/// turns as `reasoning_content` when the policy preserves it — whether the
/// endpoint *requires* that echo (deepseek with tools), merely accepts it,
/// or ignores it — the vendor decides — is a vendor fact, and the dialect serves the
/// strictest reading: echo unless told to strip.
fn chat_messages(system: &str, history: &[Message], cfg: &Config) -> Vec<Value> {
    let mut out = vec![json!({"role": "system", "content": system})];
    for msg in history {
        match msg {
            Message::User(s) => out.push(json!({"role": "user", "content": s})),
            Message::ToolResults(results) => out.extend(results.iter()
                .map(|r| json!({"role": "tool", "tool_call_id": r.id, "content": r.content}))),
            Message::Assistant(blocks) => {
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut tool_calls = vec![];
                for block in blocks {
                    match block {
                        Block::Text(t) => text.push_str(t),
                        Block::Thinking { text: r, .. } if cfg.thinking == Thinking::Preserve => {
                            reasoning.push_str(r);
                        }
                        Block::ToolUse { id, name, input } => tool_calls.push(json!({
                            "id": id, "type": "function",
                            "function": {"name": name, "arguments": input.to_string()}})),
                        _ => {}
                    }
                }
                let mut m = json!({"role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { json!(text) }});
                if !reasoning.is_empty() { m["reasoning_content"] = json!(reasoning); }
                if !tool_calls.is_empty() { m["tool_calls"] = json!(tool_calls); }
                out.push(m);
            }
        }
    }
    out
}

/// Internal tool schemas → the dialect's tool list.
fn chat_tools(schemas: &[Value]) -> Value {
    json!(schemas.iter().map(|t| json!({"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}})).collect::<Vec<_>>())
}

/// One blocking call on the dialect: post, surface the wire's error,
/// translate through the vendor, show the turn once. The blocking/streaming
/// split is the family's machinery — a vendor hands over a body and its
/// response translator, nothing else.
async fn chat_blocking(cfg: &Config, body: Value, to_internal: fn(&Value) -> Response, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = chat_url(cfg);
    let key = cfg.api_key.clone();
    let v = blocking(token, move || -> Result<Value> {
        let v: Value = post(&key, url, body, false)?.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    let resp = to_internal(&v);
    show_turn(&resp, sink);
    Ok((resp, token.is_cancelled()))
}

/// A response in this dialect → internal {content, usage}: the message walk
/// is the dialect's own, the ledger is the one thing vendors spell
/// differently, so it arrives as a function.
fn chat_to_internal(v: &Value, usage_of: fn(&Value) -> (Value, Usage)) -> Response {
    let msg = &v["choices"][0]["message"];
    let mut blocks = vec![];
    if let Some(r) = msg["reasoning_content"].as_str().filter(|r| !r.is_empty()) {
        blocks.push(Block::Thinking { text: r.to_string(), signature: None });
    }
    if let Some(t) = msg["content"].as_str().filter(|t| !t.is_empty()) {
        blocks.push(Block::Text(t.to_string()));
    }
    for tc in msg["tool_calls"].as_array().into_iter().flatten() {
        blocks.push(Block::ToolUse {
            id: tc["id"].as_str().unwrap_or("").to_string(),
            name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
            input: serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({})),
        });
    }
    let (usage, _) = usage_of(&v["usage"]);
    Response { blocks, usage }
}

/// One streaming call on the dialect: SSE deltas assembled into internal
/// blocks, events shown as they arrive. `usage_of` is the vendor's ledger
/// translation — the field families cannot agree on.
async fn chat_streaming(cfg: &Config, body: Value, usage_of: fn(&Value) -> (Value, Usage), token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = chat_url(cfg);
    let key = cfg.api_key.clone();
    let resp = blocking(token, move || post(&key, url, body, false)).await?;
    let (mut text, mut reasoning, mut tool_calls) = (String::new(), String::new(), vec![]);
    let mut think_open = false;
    let mut usage = empty_usage();
    let mut interrupted = false;
    let mut lines = sse_channel(resp);
    // Same contract as the minimax consumer: one suspension point per line,
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
            let (internal, shown) = usage_of(&v["usage"]);
            usage = internal;
            sink.show(Msg::Usage(shown));
        }
        let Some(delta) = v["choices"][0]["delta"].as_object() else { continue };
        let delta = Value::Object(delta.clone());
        // The dialect has no block boundary around reasoning; the first
        // delta that isn't reasoning (or the stream's end) closes it — that
        // is the fold point.
        let has_text = delta["content"].as_str().is_some_and(|t| !t.is_empty());
        let has_reason = delta["reasoning_content"].as_str().is_some_and(|r| !r.is_empty());
        let has_tool = !delta["tool_calls"].is_null();
        if think_open && (has_text || has_tool) {
            think_open = false;
            sink.show(Msg::ThinkEnd);
        }
        if has_text {
            let t = delta["content"].as_str().unwrap();
            sink.show(Msg::Text(t.into()));
            text.push_str(t);
        }
        if has_reason {
            let r = delta["reasoning_content"].as_str().unwrap();
            sink.show(Msg::Think(r.into()));
            think_open = true;
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
    sink.show(Msg::Done);
    let mut blocks = vec![];
    if !reasoning.is_empty() { blocks.push(Block::Thinking { text: reasoning, signature: None }); }
    if !text.is_empty() { blocks.push(Block::Text(text)); }
    for tc in tool_calls {
        blocks.push(Block::ToolUse {
            id: tc["id"].as_str().unwrap_or("").to_string(),
            name: tc["name"].as_str().unwrap_or("").to_string(),
            input: serde_json::from_str::<Value>(tc["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({})),
        });
    }
    Ok((Response { blocks, usage }, interrupted))
}

// ---- api: zai vendor -------------------------------------------------------
//
// Zhipu's zai speaks the chat-completions dialect with a vocabulary of its
// own on top, all of it staying here. Probed live before this was written:
// the flagship (glm-5.3-flash) *always* thinks — `thinking: disabled` and
// any reasoning_effort outside low/high/max are hard 400s ("该模型始终思考，
// 不支持关闭思考") — reasoning streams as reasoning_content beside content,
// usage carries prompt_tokens_details.cached_tokens from an implicit cache
// that needs no breakpoints, and `clear_thinking: false` keeps prior
// assistant turns' reasoning in context: preserved thinking, recommended
// for coding/agents precisely because the echoed reasoning is part of the
// cached prefix.

/// Policy the zai wire cannot serve, in its own words. Its cache is
/// implicit — nothing to mark, hits reported in usage. Its reasoning
/// cannot be turned off (the flagship thinks whether asked or not), so
/// strip never means "off" here — and effort, the dial that turns
/// reasoning up, cannot pair with a strip that would drop what the wire's
/// interleaved-thinking rule says to carry back.
fn zai_accepts(cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
    if cache == CacheMode::Active {
        return Err(Error::Msg("--cache active needs --protocol minimax (zai's cache is implicit — hits are automatic, nothing to mark)".into()));
    }
    if effort.is_some() && thinking == Thinking::Strip {
        return Err(Error::Msg("--effort needs --thinking preserve on the zai protocol (interleaved thinking asks for the history's reasoning back with every tool result)".into()));
    }
    Ok(())
}

/// The zai vendor's one entry into the provider port.
async fn zai_turn(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    if cfg.streaming {
        zai_streaming(cfg, history, schemas, token, sink).await
    } else {
        zai_blocking(cfg, history, schemas, token, sink).await
    }
}

/// The effort word this wire understands — its vocabulary, not the knob's:
/// low/high/max are its own words, and `medium` (a word it rejects with a
/// 400 on the flagship) folds into `high`, the tier the vendor itself
/// maps it to on models that do accept it.
fn zai_effort_word(e: Effort) -> &'static str {
    match e { Effort::Low => "low", Effort::Medium | Effort::High => "high", Effort::Max => "max" }
}

/// The zai request body: the dialect's shape plus the thinking object.
/// Preserve is *preserved thinking* here — `clear_thinking: false`, the
/// vendor's own recommendation for coding/agents, keeping prior turns'
/// reasoning in the context (and in the cached prefix). Strip sends no
/// thinking object at all: the flagship cannot stop thinking (`disabled`
/// is a hard 400), so strip on this wire means "not kept", never "off" —
/// the history blocks drop in the dialect's message translation.
fn zai_body(cfg: &Config, messages: &[Message], schemas: &[Value], stream: bool) -> Value {
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "messages": chat_messages(SYSTEM, messages, cfg), "tools": chat_tools(schemas)});
    if cfg.thinking == Thinking::Preserve {
        body["thinking"] = json!({"type": "enabled", "clear_thinking": false});
    }
    if let Some(e) = cfg.effort {
        body["reasoning_effort"] = json!(zai_effort_word(e));
    }
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

/// Zai usage → internal shape + display usage, in one place so both
/// consumers stay identical. Normalizes a wire asymmetry: this wire's
/// `prompt_tokens` *includes* the implicit cache's `cached_tokens`, while
/// the internal ledger's `input_tokens` excludes cache traffic — so
/// `input` here becomes "non-cached input", and `context_in()` (input +
/// cache read + write) reads true on the wire.
fn zai_usage(u: &Value) -> (Value, Usage) {
    let prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    let fresh = prompt.saturating_sub(cached);
    (
        json!({"input_tokens": fresh, "output_tokens": u["completion_tokens"],
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}),
        Usage { input: fresh, output: u["completion_tokens"].as_u64().unwrap_or(0), cache_read: cached, cache_write: 0 },
    )
}

/// OpenAI response → internal {content, usage}: the dialect's walk, this
/// vendor's ledger.
fn zai_to_internal(v: &Value) -> Response {
    chat_to_internal(v, zai_usage)
}

async fn zai_blocking(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    chat_blocking(cfg, zai_body(cfg, messages, schemas, false), zai_to_internal, token, sink).await
}

async fn zai_streaming(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    chat_streaming(cfg, zai_body(cfg, messages, schemas, true), zai_usage, token, sink).await
}

// ---- api: deepseek vendor --------------------------------------------------
//
// The DeepSeek API speaks the chat-completions dialect with a vocabulary
// of its own on top: a `thinking` toggle, an effort spelling, and a disk
// cache that is always on and reports itself in `usage`. All of it stays
// here — the core asks nothing, the family knows nothing.

/// Policy the deepseek wire cannot serve, in its own words. Its disk cache
/// needs no breakpoints (there is no field for one, and none is needed);
/// and its reasoning carries an echo rule — with tools present, every past
/// turn's `reasoning_content` must ride the next request — so effort, the
/// knob that turns thinking up, cannot pair with strip, the flag that
/// would erase what must be echoed.
fn deepseek_accepts(cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
    if cache == CacheMode::Active {
        return Err(Error::Msg("--cache active needs --protocol minimax (deepseek's disk cache is always on — nothing to request)".into()));
    }
    if effort.is_some() && thinking == Thinking::Strip {
        return Err(Error::Msg("--effort needs --thinking preserve on the deepseek protocol (with tools present, the wire requires every past turn's reasoning_content back)".into()));
    }
    Ok(())
}

/// The deepseek vendor's one entry into the provider port.
async fn deepseek_turn(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    if cfg.streaming {
        deepseek_streaming(cfg, history, schemas, token, sink).await
    } else {
        deepseek_blocking(cfg, history, schemas, token, sink).await
    }
}

/// The deepseek request body: the dialect's shape plus the `thinking`
/// toggle. On this wire `--thinking strip` is not an erasure but the
/// toggle off — `disabled` makes the model generate no reasoning at all,
/// which is the only honest strip on a wire whose echo rule (tools ⇒
/// reasoning back) a stripped history would break on the second request.
/// Effort rides `reasoning_effort` verbatim: low/high/max are the wire's
/// own words, and it maps medium→high itself for compatibility.
fn deepseek_body(cfg: &Config, messages: &[Message], schemas: &[Value], stream: bool) -> Value {
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "thinking": {"type": if cfg.thinking == Thinking::Strip { "disabled" } else { "enabled" }},
        "messages": chat_messages(SYSTEM, messages, cfg), "tools": chat_tools(schemas)});
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
    }
    if let Some(e) = cfg.effort {
        body["reasoning_effort"] = json!(e.label());
    }
    body
}

/// DeepSeek usage → internal shape + display. The IR contract ("input is
/// non-cached input") is the core's; the spelling is the vendor's: this
/// ledger splits its disk cache natively, hit + miss = prompt, and both
/// halves are reported on every request.
fn deepseek_usage(u: &Value) -> (Value, Usage) {
    let cached = u["prompt_cache_hit_tokens"].as_u64().unwrap_or(0);
    let fresh = u["prompt_cache_miss_tokens"].as_u64()
        .unwrap_or_else(|| u["prompt_tokens"].as_u64().unwrap_or(0).saturating_sub(cached));
    (
        json!({"input_tokens": fresh, "output_tokens": u["completion_tokens"],
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}),
        Usage { input: fresh, output: u["completion_tokens"].as_u64().unwrap_or(0), cache_read: cached, cache_write: 0 },
    )
}

/// DeepSeek response → internal {content, usage}: the dialect's walk, this
/// vendor's ledger.
fn deepseek_to_internal(v: &Value) -> Response {
    chat_to_internal(v, deepseek_usage)
}

async fn deepseek_blocking(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    chat_blocking(cfg, deepseek_body(cfg, messages, schemas, false), deepseek_to_internal, token, sink).await
}

async fn deepseek_streaming(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    chat_streaming(cfg, deepseek_body(cfg, messages, schemas, true), deepseek_usage, token, sink).await
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
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
            protocol: Protocol::MINIMAX, cache: CacheMode::Auto, thinking: Thinking::Preserve,
            effort: None,
            max_tokens: 1024, context_size: DEFAULT_CONTEXT_SIZE, max_turns: DEFAULT_MAX_TURNS, streaming,
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
    fn user_message_is_the_single_spelling_of_a_task() {
        assert_eq!(user_message("count files").to_value(), json!({"role": "user", "content": "count files"}));
    }

    #[test]
    fn enum_flags_case_insensitive_and_list_available_values() {
        let parse = |extra: &[&str]| parse_from(
            ["--api-key", "k", "--base-url", "https://x", "-m", "m"].iter()
                .chain(extra.iter()).map(|s| s.to_string())).unwrap();
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
    fn strip_thinking_keeps_everything_else() {
        let history = hv(vec![json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "hello"}]})]);
        let stripped = strip_thinking(&history);
        assert_eq!(hist(&stripped)[0]["content"].as_array().unwrap().len(), 1);
        assert_eq!(hist(&history)[0]["content"].as_array().unwrap().len(), 2); // untouched
    }

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

    #[test]
    fn zai_conversion_maps_all_message_shapes() {
        let history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reason away"},
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "exit=0"}]}),
        ]);
        let msgs = chat_messages("sys", &history, &cfg("https://x".into(), false));
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[2]["reasoning_content"], json!("reason away"));
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["arguments"], json!({"command": "ls"}).to_string());
        assert_eq!(msgs[3]["role"], json!("tool"));
        assert_eq!(msgs[3]["tool_call_id"], json!("t1"));
        // strip mode drops reasoning
        let mut cfg = cfg("https://x".into(), false);
        cfg.thinking = Thinking::Strip;
        let msgs = chat_messages("sys", &history, &cfg);
        assert!(msgs[2].get("reasoning_content").is_none());
    }

    #[test]
    fn zai_conversion_assistant_shapes_and_tool_schemas() {
        // text-only assistant keeps a plain string content
        let history = hv(vec![json!({"role": "assistant", "content": [{"type": "text", "text": "hi"}]})]);
        let msgs = chat_messages("sys", &history, &cfg("https://x".into(), false));
        assert_eq!(msgs[1]["content"], json!("hi"));
        assert!(msgs[1].get("tool_calls").is_none());
        // text-less assistant sends null content, not ""
        let history = hv(vec![json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "bash", "input": {}}]})]);
        let msgs = chat_messages("sys", &history, &cfg("https://x".into(), false));
        assert_eq!(msgs[1]["content"], Value::Null);
        assert!(msgs[1]["tool_calls"].is_array());
        // internal tool schema → OpenAI function spec
        let schemas = vec![json!({"name": "bash", "description": "run", "input_schema":
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}})];
        let t = chat_tools(&schemas);
        assert_eq!(t[0]["type"], json!("function"));
        assert_eq!(t[0]["function"]["name"], json!("bash"));
        assert_eq!(t[0]["function"]["description"], json!("run"));
        assert_eq!(t[0]["function"]["parameters"], schemas[0]["input_schema"]);
    }

    #[test]
    fn zai_response_normalizes_to_internal_shape() {
        let v = json!({"choices": [{"message": {
            "reasoning_content": "thinking", "content": "answer",
            "tool_calls": [{"id": "t9", "type": "function",
                "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 4}}});
        let n = zai_to_internal(&v);
        let c = blocks(&n.blocks);
        assert_eq!(c[0]["type"], json!("thinking"));
        assert_eq!(c[1]["text"], json!("answer"));
        assert_eq!(c[2]["input"]["command"], json!("ls"));
        assert_eq!(n.usage["cache_read_input_tokens"], json!(4));
    }

    #[test]
    fn zai_streaming_assembles_deltas() {
        let events = concat!(
            r#"data: {"choices":[{"delta":{"content":"hel"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#, "\r\n\r\n", // CRLF is legal SSE framing
            r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","type":"function","function":{"name":"bash","arguments":"{\"comm"}}]}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"and\":\"echo hi\"}"}}]}}]}"#, "\n\n",
            r#"data: {"usage":{"prompt_tokens":9,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2}}}"#, "\n\n",
            r#"data: [DONE]"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.protocol = Protocol::ZAI;
        c.streaming = true;
        let (resp, cut) = block_on(call_api(&c, &hv(vec![json!({"role": "user", "content": "hey"})]), &[], &CancelToken::new(), &sink())).unwrap();
        assert!(!cut);
        let content = blocks(&resp.blocks);
        assert_eq!(content[0], json!({"type": "thinking", "thinking": "think"}));
        assert_eq!(content[1], json!({"type": "text", "text": "hello"}));
        assert_eq!(content[2]["type"], json!("tool_use"));
        assert_eq!(content[2]["id"], json!("t1"));
        assert_eq!(content[2]["input"]["command"], json!("echo hi"));
        // prompt_tokens includes cached_tokens on this wire; the internal
        // shape normalizes to non-cached input (9 - 2) so context_in()
        // reads true beside the Anthropic numbers
        assert_eq!(resp.usage["input_tokens"], json!(7));
        assert_eq!(resp.usage["output_tokens"], json!(4));
        assert_eq!(resp.usage["cache_read_input_tokens"], json!(2));
    }

    #[test]
    fn effort_maps_to_the_wires_and_respects_their_rules() {
        let mut c = cfg("https://x".into(), true);

        // minimax: adaptive thinking, the budget as its dial
        c.effort = Some(Effort::Low);
        let body = minimax_body(&c, &[], &[], true);
        assert_eq!(body["thinking"], json!({"type": "adaptive", "budget_tokens": 1024}));
        c.effort = Some(Effort::Max);
        c.max_tokens = 4096;
        assert_eq!(minimax_body(&c, &[], &[], true)["thinking"]["budget_tokens"], json!(3072),
            "max = everything but a floor for the reply");
        c.max_tokens = 1024; // degenerate: the floor itself
        assert_eq!(Effort::High.budget(1024), 1024, "clamped to the wire minimum");

        // zai: the word in its own vocabulary — medium folds into high
        c.protocol = Protocol::ZAI;
        c.effort = Some(Effort::High);
        assert_eq!(zai_body(&c, &[], &[], false)["reasoning_effort"], json!("high"));
        c.effort = Some(Effort::Medium);
        assert_eq!(zai_body(&c, &[], &[], false)["reasoning_effort"], json!("high"),
            "medium is not a word this wire accepts — the vendor's own fold");
        c.effort = Some(Effort::Max);
        assert_eq!(zai_body(&c, &[], &[], false)["reasoning_effort"], json!("max"));

        // effort absent: the word leaves the wire (byte-identical to before
        // the knob); preserve keeps the vendor's recommended default —
        // preserved thinking, the reasoning kept in context and in the
        // cached prefix
        c.effort = None;
        assert!(zai_body(&c, &[], &[], false).get("reasoning_effort").is_none());
        assert_eq!(zai_body(&c, &[], &[], false)["thinking"], json!({"type": "enabled", "clear_thinking": false}));
        c.thinking = Thinking::Strip;
        assert!(zai_body(&c, &[], &[], false).get("thinking").is_none(),
            "the flagship cannot stop thinking — strip sends no object, it just isn't kept");
        c.thinking = Thinking::Preserve;
        c.protocol = Protocol::MINIMAX;
        assert_eq!(minimax_body(&c, &[], &[], true)["thinking"], json!({"type": "adaptive"}));

        // minimax + strip: the wire forbids it (thinking continuity)
        let mut args = Args { effort: Some("high".into()), thinking: Some("strip".into()), ..Default::default() };
        std::env::remove_var("JINGWEI_EFFORT");
        std::env::remove_var("JINGWEI_THINKING");
        assert!(build_config(&args).is_err(), "effort + strip on minimax is rejected");
        args.protocol = Some("openai".into());
        // the openai wire has no thinking-continuity rule; but the rest of
        // the config is incomplete here, so only probe the guard itself
        let probe = |p: Protocol| {
            let word = p.label();
            let mut a = Args { effort: Some("high".into()), thinking: Some("strip".into()),
                protocol: Some(word.into()),
                ..Default::default() };
            a.api_key = Some("k".into()); a.base_url = Some("https://x".into()); a.model = Some("m".into());
            build_config(&a)
        };
        assert!(probe(Protocol::MINIMAX).is_err());
        assert!(probe(Protocol::ZAI).is_err(), "zai keeps its reasoning: strip + effort is refused");
        assert!(probe(Protocol::DEEPSEEK).is_err());
        // the env var spells it too
        std::env::set_var("JINGWEI_EFFORT", "max");
        let a = Args { api_key: Some("k".into()), base_url: Some("https://x".into()), model: Some("m".into()), ..Default::default() };
        assert_eq!(build_config(&a).unwrap().effort, Some(Effort::Max));
        std::env::remove_var("JINGWEI_EFFORT");
    }

    #[test]
    fn minimax_body_thinking_toggle_and_verbatim_echo() {
        let mut c = cfg("https://x".into(), true);
        // preserve: adaptive — this vendor ships M3 thinking *off* by
        // default; an agent's default is reasoning on
        assert_eq!(minimax_body(&c, &[], &[], false)["thinking"], json!({"type": "adaptive"}));
        // strip: the honest off — no reasoning generated, so none needs
        // echoing back (the strip of history blocks happens in the entry)
        c.thinking = Thinking::Strip;
        assert_eq!(minimax_body(&c, &[], &[], true)["thinking"], json!({"type": "disabled"}));
        // history rides the wire verbatim — signature and all, the vendor's
        // continuity rule for interleaved thinking
        c.thinking = Thinking::Preserve;
        let history = hv(vec![json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "step", "signature": "sig1"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]})]);
        let body = minimax_body(&c, &history, &[], false);
        assert_eq!(body["messages"][0]["content"][0]["signature"], json!("sig1"));
        assert_eq!(body["messages"][0]["content"][1]["id"], json!("t1"));
    }

    #[test]
    fn minimax_body_marks_cache_breakpoints_only_when_active() {
        let schemas = vec![json!({"name": "a"}), json!({"name": "b"}), json!({"name": "c"})];
        let mut c = cfg("https://x".into(), true);
        c.cache = CacheMode::Active;
        let body = minimax_body(&c, &[], &schemas, true);
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
        let body = minimax_body(&c, &[], &schemas, false);
        assert_eq!(body["system"], json!(SYSTEM));
        assert!(body["tools"].as_array().unwrap().iter().all(|t| t.get("cache_control").is_none()));
        assert_eq!(body["stream"], json!(false));
    }

    #[test]
    fn wire_paths_and_auth_headers_per_protocol() {
        // minimax: /v1/messages with x-api-key + the Messages-wire version
        // header + bearer (the endpoint honors both auth styles)
        let (port, reqs) = mock_seq(vec![(200, r#"{"content":[],"usage":{}}"#.into())]);
        block_on(call_api(&cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new(), &sink())).unwrap();
        let req = reqs.lock().unwrap()[0].to_lowercase();
        assert!(req.contains("post /v1/messages http/1.1"), "{req}");
        assert!(req.contains("x-api-key: test-key"), "{req}");
        assert!(req.contains("anthropic-version: 2023-06-01"), "{req}");
        assert!(req.contains("authorization: bearer test-key"), "{req}");
        // trailing slash in base_url must not double the path
        let (port, reqs) = mock_seq(vec![(200, r#"{"content":[],"usage":{}}"#.into())]);
        block_on(call_api(&cfg(format!("http://127.0.0.1:{port}/"), false), &[], &[], &CancelToken::new(), &sink())).unwrap();
        assert!(reqs.lock().unwrap()[0].contains("POST /v1/messages HTTP/1.1"));
        // openai: /chat/completions with bearer only
        let (port, reqs) = mock_seq(vec![(200, r#"{"choices":[{"message":{"content":"ok"}}]}"#.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::ZAI;
        block_on(call_api(&c, &hv(vec![json!({"role": "user", "content": "hi"})]), &[], &CancelToken::new(), &sink())).unwrap();
        let req = reqs.lock().unwrap()[0].to_lowercase();
        assert!(req.contains("post /chat/completions http/1.1"), "{req}");
        assert!(req.contains("authorization: bearer test-key"), "{req}");
        assert!(!req.contains("x-api-key:"), "{req}");
    }

    // ---- api over local mocks ----

    #[test]
    fn minimax_blocking_roundtrip_and_error() {
        let port = mock(r#"{"content":[{"type":"text","text":"hi from mock"}],"usage":{"input_tokens":7,"output_tokens":3}}"#, 200, false);
        let (resp, cut) = block_on(call_api(&cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new(), &sink())).unwrap();
        assert_eq!(blocks(&resp.blocks)[0]["text"], "hi from mock");
        assert!(!cut);
        let port = mock(r#"{"error":{"message":"bad model"}}"#, 400, false);
        let err = block_on(call_api(&cfg(format!("http://127.0.0.1:{port}"), false), &[], &[], &CancelToken::new(), &sink())).unwrap_err();
        assert!(err.to_string().contains("bad model"), "got: {err}");
    }

    #[test]
    fn zai_blocking_roundtrip() {
        let port = mock(r#"{"choices":[{"message":{"content":"mock says hi"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":1}}}"#, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::ZAI;
        let (resp, _) = block_on(call_api(&c, &hv(vec![json!({"role": "user", "content": "hey"})]), &[], &CancelToken::new(), &sink())).unwrap();
        assert_eq!(blocks(&resp.blocks)[0]["text"], "mock says hi");
        assert_eq!(resp.usage["cache_read_input_tokens"], json!(1));
    }

    #[test]
    fn zai_rules_and_defaults_live_with_the_vendor() {
        let ok = |cache, thinking, effort| zai_accepts(cache, thinking, effort).is_ok();
        assert!(ok(CacheMode::Auto, Thinking::Preserve, Some(Effort::High)));
        assert!(!ok(CacheMode::Active, Thinking::Preserve, None), "the cache is implicit — nothing to mark");
        assert!(!ok(CacheMode::Auto, Thinking::Strip, Some(Effort::Low)), "effort cannot pair with strip");
        assert!(ok(CacheMode::Auto, Thinking::Strip, None),
            "strip alone is 'not kept', never 'off' — the flagship cannot stop thinking");
        // the vendor names endpoint and flagship; a key alone reaches it
        std::env::remove_var("JINGWEI_BASE_URL");
        std::env::remove_var("JINGWEI_MODEL");
        let args = Args { protocol: Some("zai".into()), api_key: Some("k".into()), ..Default::default() };
        let c = match build_config(&args) {
            Ok(c) => c,
            Err(e) => panic!("zai names its endpoint and flagship: {e}"),
        };
        assert_eq!((c.protocol_label(), c.base_url.as_str(), c.model.as_str()),
            ("zai", "https://open.bigmodel.cn/api/paas/v4", "glm-5.3-flash"));
    }

    #[test]
    fn zai_usage_reads_the_implicit_cache_ledger() {
        // the live wire's shape: prompt includes the implicit cache's hits,
        // reported as cached_tokens (verified against open.bigmodel.cn)
        let (ir, shown) = zai_usage(&json!({"prompt_tokens": 210, "completion_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 128}, "total_tokens": 224}));
        assert_eq!(ir, json!({"input_tokens": 82, "output_tokens": 14,
            "cache_read_input_tokens": 128, "cache_creation_input_tokens": 0}));
        assert_eq!((shown.input, shown.cache_read), (82, 128));
        // reasoning rides completion_tokens_details, not the ledger — the
        // ctx gauge reads inputs, the reasoning is the output's business
        let (ir, _) = zai_usage(&json!({"prompt_tokens": 19, "completion_tokens": 66,
            "completion_tokens_details": {"reasoning_tokens": 61}}));
        assert_eq!(ir["input_tokens"], json!(19));
        assert_eq!(ir["output_tokens"], json!(66));
    }

    // ---- deepseek vendor ----

    #[test]
    fn deepseek_body_carries_the_toggle_and_the_effort_word() {
        let mut c = cfg("https://x".into(), true);
        c.protocol = Protocol::DEEPSEEK;
        // preserve + effort: thinking on, the wire's own word verbatim
        // (medium arrives as "medium" — the wire maps it to high itself)
        c.effort = Some(Effort::Max);
        let b = deepseek_body(&c, &[], &[], false);
        assert_eq!(b["thinking"], json!({"type": "enabled"}));
        assert_eq!(b["reasoning_effort"], json!("max"));
        assert!(b.get("stream_options").is_none(), "blocking asks for no stream options");
        // strip: the toggle off — this wire's only honest strip
        c.thinking = Thinking::Strip;
        c.effort = None;
        let b = deepseek_body(&c, &[], &[], true);
        assert_eq!(b["thinking"], json!({"type": "disabled"}));
        assert!(b.get("reasoning_effort").is_none());
        assert_eq!(b["stream_options"]["include_usage"], json!(true));
        // reasoning echo: the dialect's strictest reading serves this wire's
        // rule (tools ⇒ every past turn's reasoning_content back)
        c.thinking = Thinking::Preserve;
        let history = hv(vec![json!({"role": "assistant", "content": [
            {"type": "thinking", "thinking": "step one"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]})]);
        let msgs = chat_messages(SYSTEM, &history, &c);
        assert_eq!(msgs[1]["reasoning_content"], json!("step one"));
        assert_eq!(msgs[1]["tool_calls"][0]["id"], json!("t1"));
    }

    #[test]
    fn deepseek_rules_live_with_the_vendor() {
        let ok = |cache, thinking, effort| deepseek_accepts(cache, thinking, effort).is_ok();
        assert!(ok(CacheMode::Auto, Thinking::Preserve, Some(Effort::High)));
        assert!(!ok(CacheMode::Active, Thinking::Preserve, None), "the disk cache is always on — no breakpoints to request");
        assert!(!ok(CacheMode::Auto, Thinking::Strip, Some(Effort::Low)), "effort cannot pair with strip");
        assert!(ok(CacheMode::Auto, Thinking::Strip, None), "strip alone is the toggle off");
        // the vendor names its own endpoint (and flagship, when it has
        // one); the wires stay unnamed
        assert_eq!(Protocol::DEEPSEEK.default_base(), Some("https://api.deepseek.com"));
        assert_eq!(Protocol::MINIMAX.default_base(), Some("https://api.minimax.cn/anthropic"));
        assert_eq!(Protocol::MINIMAX.default_model(), Some("MiniMax-M3"));
        assert_eq!(Protocol::ZAI.default_base(), Some("https://open.bigmodel.cn/api/paas/v4"));
        assert_eq!(Protocol::ZAI.default_model(), Some("glm-5.3-flash"));
        assert_eq!(Protocol::DEEPSEEK.default_model(), None, "flash or pro is the user's call");
    }

    #[test]
    fn deepseek_default_base_and_policy_ride_build_config() {
        std::env::remove_var("JINGWEI_BASE_URL");
        std::env::remove_var("JINGWEI_EFFORT");
        std::env::remove_var("JINGWEI_THINKING");
        let args = Args { protocol: Some("deepseek".into()), api_key: Some("k".into()),
            model: Some("deepseek-flash".into()), ..Default::default() };
        let c = match build_config(&args) {
            Ok(c) => c,
            Err(e) => panic!("deepseek names its own endpoint: {e}"),
        };
        assert_eq!(c.base_url, "https://api.deepseek.com");
        assert_eq!(c.protocol_label(), "deepseek");
        let strip_effort = Args { thinking: Some("strip".into()), effort: Some("high".into()), ..args };
        let err = match build_config(&strip_effort) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("strip + effort must be rejected on this wire"),
        };
        assert!(err.contains("reasoning_content"), "vendor words for a vendor rule; got: {err}");
    }

    #[test]
    fn deepseek_usage_reads_the_disk_cache_ledger() {
        let (ir, shown) = deepseek_usage(&json!({"prompt_tokens": 210, "prompt_cache_hit_tokens": 128,
            "prompt_cache_miss_tokens": 82, "completion_tokens": 14}));
        assert_eq!(ir, json!({"input_tokens": 82, "output_tokens": 14,
            "cache_read_input_tokens": 128, "cache_creation_input_tokens": 0}));
        assert_eq!((shown.input, shown.cache_read), (82, 128));
        // no native split, no cache read claimed — the hit rate stays honest
        let (ir, _) = deepseek_usage(&json!({"prompt_tokens": 30, "completion_tokens": 5}));
        assert_eq!(ir["input_tokens"], json!(30));
        assert_eq!(ir["cache_read_input_tokens"], json!(0));
    }

    #[test]
    fn deepseek_blocking_roundtrip_and_error() {
        let port = mock(r#"{"choices":[{"message":{"content":"北京","reasoning_content":"thinking hard"}}],"usage":{"prompt_tokens":38,"prompt_cache_hit_tokens":12,"prompt_cache_miss_tokens":26,"completion_tokens":4}}"#, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::DEEPSEEK;
        let (resp, _) = block_on(call_api(&c, &hv(vec![json!({"role": "user", "content": "capital?"})]), &[], &CancelToken::new(), &sink())).unwrap();
        let content = blocks(&resp.blocks);
        assert_eq!(content[0]["thinking"], json!("thinking hard"));
        assert_eq!(content[1]["text"], json!("北京"));
        assert_eq!(resp.usage["input_tokens"], json!(26));
        assert_eq!(resp.usage["cache_read_input_tokens"], json!(12));
        let port = mock(r#"{"error":{"message":"The supported API model names are deepseek-flash, deepseek-v4-pro, but you passed deepseek-bogus."}}"#, 400, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::DEEPSEEK;
        let err = block_on(call_api(&c, &hv(vec![json!({"role": "user", "content": "hi"})]), &[], &CancelToken::new(), &sink())).unwrap_err();
        assert!(err.to_string().contains("deepseek-bogus"), "the wire's error body, verbatim; got: {err}");
    }

    #[test]
    fn deepseek_streaming_assembles_deltas() {
        // shapes taken from the live wire: reasoning deltas (no boundary —
        // the first non-reasoning delta folds), a tool_call split across
        // chunks (id+name first, arguments after), usage riding the last
        // content chunk, [DONE] to close
        let events = concat!(
            r#"data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#, "\n\n",
            r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":"The user wants"},"finish_reason":null}]}"#, "\n\n",
            r#"data: {"choices":[{"index":0,"delta":{"reasoning_content":" the date."},"finish_reason":null}]}"#, "\n\n",
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_00_x","type":"function","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#, "\n\n",
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":\"date\"}"}}]},"finish_reason":null}]}"#, "\n\n",
            r#"data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":275,"completion_tokens":37,"prompt_cache_hit_tokens":128,"prompt_cache_miss_tokens":147}}"#, "\n\n",
            r#"data: [DONE]"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.protocol = Protocol::DEEPSEEK;
        c.streaming = true;
        let (resp, cut) = block_on(call_api(&c, &[], &[], &CancelToken::new(), &sink())).unwrap();
        assert!(!cut);
        assert_eq!(blocks(&resp.blocks)[0]["thinking"], json!("The user wants the date."));
        assert_eq!(blocks(&resp.blocks)[1]["input"]["command"], json!("date"));
        assert_eq!(resp.usage["input_tokens"], json!(147));
        assert_eq!(resp.usage["cache_read_input_tokens"], json!(128));
    }

    #[test]
    fn minimax_streaming_assembles_blocks() {
        // shapes from the live wire: a thinking block stamped with its
        // signature as it closes (the vendor's echo rule wants the block
        // back whole), then text, then a tool_use arriving as partial JSON
        let events = concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":0}}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step"}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c8b7a9218aec"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello world"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":1}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"comma"}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"nd\":\"ls\"}"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":2}"#, "\n\n",
            r#"data: {"type":"message_delta","usage":{"output_tokens":2,"cache_read_input_tokens":203}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.streaming = true;
        let (resp, cut) = block_on(call_api(&c, &[], &[], &CancelToken::new(), &sink())).unwrap();
        assert!(!cut);
        assert_eq!(blocks(&resp.blocks)[0]["thinking"], json!("step"));
        assert_eq!(blocks(&resp.blocks)[0]["signature"], json!("c8b7a9218aec"));
        assert_eq!(blocks(&resp.blocks)[1]["text"], json!("hello world"));
        assert_eq!(blocks(&resp.blocks)[2]["input"]["command"], json!("ls"));
        assert_eq!(resp.usage["output_tokens"], json!(2));
        assert_eq!(resp.usage["cache_read_input_tokens"], json!(203));
    }

    #[test]
    fn minimax_streaming_surfaces_error_events() {
        let events = concat!(
            r#"data: {"type":"error","error":{"type":"overloaded","message":"server overloaded"}}"#, "\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{}", mock(events, 200, true)), true);
        c.streaming = true;
        assert!(block_on(call_api(&c, &[], &[], &CancelToken::new(), &sink())).unwrap_err().to_string().contains("overloaded"));
    }

    // ---- agent loop ----

    #[test]
    fn agent_loop_runs_tool_then_stops_on_plain_text() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"echo hi"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let done = r#"{"content":[{"type":"text","text":"done"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, reqs) = mock_seq(vec![(200, tool.into()), (200, done.into())]);
        let mut history = hv(vec![json!({"role": "user", "content": "run echo"})]);
        block_on(agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &CancelToken::new(), &sink())).unwrap();
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
    }

    #[test]
    fn agent_loop_stops_at_max_turns() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"true"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, _) = mock_seq(vec![(200, tool.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.max_turns = 1;
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        block_on(agent_loop(&c, &mut history, &CancelToken::new(), &sink())).unwrap(); // must return instead of spinning
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
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let err = block_on(async {
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), true), &mut history, &token, &sink()).await
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
    }

    #[test]
    fn agent_loop_interrupted_mid_tool_finishes_pairing() {
        let cmd = if cfg!(windows) { "ping -n 30 127.0.0.1 > nul" } else { "sleep 30" };
        let tool = format!(
            r#"{{"content":[{{"type":"tool_use","id":"t1","name":"bash","input":{{"command":"{cmd}"}}}}],"usage":{{"input_tokens":1,"output_tokens":1}}}}"#);
        let (port, _) = mock_seq(vec![(200, tool)]);
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let err = block_on(async {
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut history, &token, &sink()).await
        }).unwrap_err();
        assert!(matches!(err, Error::Interrupted), "got: {err}");
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
        let result = hist(&history)[2]["content"][0]["content"].as_str().unwrap().to_string();
        assert!(result.contains("[interrupted by user]"), "got: {result}");
        assert_eq!(hist(&history)[2]["content"][0]["tool_use_id"], json!("t1"));
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
