// The provider port and everything that has to sit behind it.
//
// One wire is one `Vendor` impl; the composition root holds a `Protocol`
// handle (a newtype over `&'static dyn Vendor`) and calls through it —
// no `match` anywhere. Adding a vendor is one `impl Vendor` and one row
// in `VENDORS`, both the compiler's problem to keep complete. The port
// itself, the shared transport (HTTP, retries, SSE), the chat-completions
// family helpers, and the per-vendor "rules" (what this wire can serve)
// all live here; the per-vendor files (`minimax.rs`, `zai.rs`, `deepseek.rs`)
// own only their bodies, their usage translations, and their tests.

use crate::config::Config;
use crate::cancel::CancelToken;
use crate::display::{Msg, Show};
use crate::ir::{Block, Message, Response};
use crate::display::Usage;
use crate::{Error, Result};
use serde_json::{json, Value};
use std::future::Future;
use std::io::{BufRead, BufReader};
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc;

/// The boxed future a vendor's `turn` returns. Boxed because the wire is
/// chosen at runtime — one allocation per request, and the vendor's own
/// streaming/blocking machinery stays its own.
pub(crate) type Turn<'a> = Pin<Box<dyn Future<Output = Result<(Response, bool)>> + Send + 'a>>;

/// The provider port. One impl per wire; the composition root holds a
/// [`Protocol`] handle and calls through it — no `match` anywhere. Adding a
/// vendor is one `impl` and one row in [`VENDORS`], both the compiler's
/// problem to keep complete.
pub(crate) trait Vendor: Sync {
    fn label(&self) -> &'static str;
    fn default_base(&self) -> Option<&'static str>;
    fn default_model(&self) -> Option<&'static str>;
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()>;
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
        Box::pin(minimax::turn(cfg, history, schemas, token, sink))
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
        Box::pin(zai::turn(cfg, history, schemas, token, sink))
    }
}

impl Vendor for DeepSeek {
    fn label(&self) -> &'static str { "deepseek" }
    fn default_base(&self) -> Option<&'static str> { Some("https://api.deepseek.com") }
    fn default_model(&self) -> Option<&'static str> { None }
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        deepseek_accepts(cache, thinking, effort)
    }
    fn turn<'a>(&'a self, cfg: &'a Config, history: &'a [Message], schemas: &'a [Value], token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(deepseek::turn(cfg, history, schemas, token, sink))
    }
}

pub(crate) const VENDORS: &[(&str, Protocol)] = &[
    ("minimax", Protocol::MINIMAX),
    ("zai", Protocol::ZAI),
    ("deepseek", Protocol::DEEPSEEK),
];

/// A handle to one wire — what used to be an enum spelling each protocol's
/// facts and dispatching by `match`, now a newtype over the trait object.
#[derive(Clone, Copy)]
pub(crate) struct Protocol(&'static dyn Vendor);

impl Protocol {
    pub(crate) const MINIMAX: Self = Self(&MINIMAX);
    pub(crate) const ZAI: Self = Self(&ZAI);
    pub(crate) const DEEPSEEK: Self = Self(&DEEPSEEK);
}

impl Protocol {
    pub(crate) fn label(self) -> &'static str { self.0.label() }
    pub(crate) fn default_base(self) -> Option<&'static str> { self.0.default_base() }
    pub(crate) fn default_model(self) -> Option<&'static str> { self.0.default_model() }
    pub(crate) fn accepts(self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        self.0.accepts(cache, thinking, effort)
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
pub(crate) enum CacheMode { Auto, Active }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Thinking { Preserve, Strip }

/// Reasoning effort, when the endpoint offers the knob. Absent (`None`)
/// means *say nothing on the wire* — the endpoint's own default rules.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Effort { Low, Medium, High, Max }

impl Effort {
    pub(crate) fn label(self) -> &'static str {
        match self { Self::Low => "low", Self::Medium => "medium", Self::High => "high", Self::Max => "max" }
    }
    /// The Anthropic-style budget for this tier — the Messages wire's dial,
    /// clamped into its rules: at least 1024, and strictly under
    /// `max_tokens` so the reply has room.
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

// ---- transport: shared by every wire --------------------------------------

pub(crate) async fn call_api(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    cfg.protocol.0.turn(cfg, history, schemas, token, sink).await
}

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

pub(crate) fn strip_thinking(messages: &[Message]) -> Vec<Message> {
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
/// with Bearer alone.
fn post(api_key: &str, url: String, body: Value, messages_wire: bool) -> Result<ureq::Response> {
    let mut req = http().post(&url).set("Authorization", &format!("Bearer {api_key}"));
    if messages_wire {
        req = req.set("x-api-key", api_key).set("anthropic-version", "2023-06-01");
    }
    Ok(req.send_json(body)?)
}

pub(crate) const RETRY_ATTEMPTS: u32 = 4;
pub(crate) const RETRY_BASE: Duration = Duration::from_millis(500);
pub(crate) const RETRY_MAX: Duration = Duration::from_secs(8);

pub(crate) fn backoff(attempt: u32) -> Duration {
    let shift = attempt.saturating_sub(1).min(5);
    (RETRY_BASE * 2u32.pow(shift)).min(RETRY_MAX)
}

async fn request<T>(
    token: &CancelToken,
    key: &str,
    url: &str,
    body: &Value,
    messages_wire: bool,
    decode: fn(ureq::Response) -> Result<T>,
) -> Result<T>
where
    T: Send + 'static,
{
    let mut attempt = 0;
    loop {
        attempt += 1;
        let (key, url, body) = (key.to_owned(), url.to_owned(), body.clone());
        let out = blocking(token, move || decode(post(&key, url, body, messages_wire)?)).await;
        match out {
            Ok(v) => return Ok(v),
            Err(e) if attempt < RETRY_ATTEMPTS && e.is_transient() => {
                tokio::select! {
                    _ = tokio::time::sleep(backoff(attempt)) => {}
                    _ = token.cancelled() => return Err(e),
                }
            }
            Err(e) => return Err(e),
        }
    }
}

pub(crate) fn sse_channel(resp: ureq::Response) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel(64);
    std::thread::spawn(move || {
        let mut reader = BufReader::new(resp.into_reader());
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let l = line.trim_end_matches(['\r', '\n']).to_string();
                    if tx.blocking_send(l).is_err() { break; }
                }
            }
        }
    });
    rx
}

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

pub(crate) fn empty_usage() -> Value {
    json!({"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
}

pub(crate) fn merge_usage(usage: &mut Value, src: Option<&serde_json::Map<String, Value>>) {
    if let (Some(dst), Some(src)) = (usage.as_object_mut(), src) {
        for (k, v) in src { dst.insert(k.clone(), v.clone()); }
    }
}

pub(crate) fn response_from_value(v: &Value) -> Response {
    Response {
        blocks: v["content"].as_array().map(|a| a.iter().map(Block::from_value).collect()).unwrap_or_default(),
        usage: v["usage"].clone(),
    }
}

// ---- chat-completions wire family ----------------------------------------

fn chat_url(cfg: &Config) -> String {
    format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'))
}

fn chat_messages(system: &str, history: &[Message], _cfg: &Config) -> Vec<Value> {
    let mut out = Vec::with_capacity(history.len() + 1);
    out.push(json!({"role": "system", "content": system}));
    for m in history {
        out.push(Message::to_value(m));
    }
    out
}

fn chat_tools(schemas: &[Value]) -> Value {
    Value::Array(schemas.to_vec())
}

async fn chat_blocking(cfg: &Config, body: Value, to_internal: fn(&Value) -> Response, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = chat_url(cfg);
    let key = cfg.api_key.clone();
    let resp = request(token, &key, &url, &body, false, |r| {
        let v: Value = r.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    let resp = to_internal(&resp);
    show_turn(&resp, sink);
    Ok((resp, token.is_cancelled()))
}

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

async fn chat_streaming(cfg: &Config, body: Value, usage_of: fn(&Value) -> (Value, Usage), token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = chat_url(cfg);
    let key = cfg.api_key.clone();
    let resp = request(token, &key, &url, &body, false, Ok).await?;
    let (mut text, mut reasoning, mut tool_calls) = (String::new(), String::new(), vec![]);
    let mut think_open = false;
    let mut usage = empty_usage();
    let mut interrupted = false;
    let mut lines = sse_channel(resp);
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

// ---- vendor rules ---------------------------------------------------------

fn minimax_accepts(_cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
    if effort.is_some() && thinking == Thinking::Strip {
        return Err(Error::Msg("--effort needs --thinking preserve on the minimax protocol (interleaved thinking requires the history's thinking blocks, signature and all)".into()));
    }
    Ok(())
}

fn zai_accepts(cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
    if cache == CacheMode::Active {
        return Err(Error::Msg("--cache active needs --protocol minimax (zai's cache is implicit — hits are automatic, nothing to mark)".into()));
    }
    if effort.is_some() && thinking == Thinking::Strip {
        return Err(Error::Msg("--effort needs --thinking preserve on the zai protocol (interleaved thinking asks for the history's reasoning back with every tool result)".into()));
    }
    Ok(())
}

fn deepseek_accepts(cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
    if cache == CacheMode::Active {
        return Err(Error::Msg("--cache active needs --protocol minimax (deepseek's disk cache is always on — nothing to request)".into()));
    }
    if effort.is_some() && thinking == Thinking::Strip {
        return Err(Error::Msg("--effort needs --thinking preserve on the deepseek protocol (with tools present, the wire requires every past turn's reasoning_content back)".into()));
    }
    Ok(())
}

// ---- vendor submodules ----------------------------------------------------

pub(crate) mod minimax;
pub(crate) mod zai;
pub(crate) mod deepseek;
