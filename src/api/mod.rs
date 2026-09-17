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

/// Run a synchronous closure on the blocking pool with the same
/// cancellation shape the streaming callers already use: the work
/// runs on a worker thread, the await is a real suspension point, and
/// the token races it so a Ctrl-C mid-reap returns `Interrupted`
/// instead of blocking the await. Shared between the api transport
/// (where every HTTP body lives) and the bash tool (where reaping
/// the killed child would otherwise freeze the single-thread runtime).
pub(crate) async fn blocking<T, F>(token: &CancelToken, f: F) -> Result<T>
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Block, Message};
    use serde_json::json;

    // ---- Protocol: the dispatcher's typed handle --------------------------

    #[test]
    fn protocol_label_names_each_vendor() {
        assert_eq!(Protocol::MINIMAX.label(), "minimax");
        assert_eq!(Protocol::ZAI.label(), "zai");
        assert_eq!(Protocol::DEEPSEEK.label(), "deepseek");
    }

    #[test]
    fn protocol_default_base_lists_each_endpoint() {
        assert_eq!(Protocol::MINIMAX.default_base(), Some("https://api.minimax.cn/anthropic"));
        assert_eq!(Protocol::ZAI.default_base(), Some("https://open.bigmodel.cn/api/paas/v4"));
        assert_eq!(Protocol::DEEPSEEK.default_base(), Some("https://api.deepseek.com"));
    }

    #[test]
    fn protocol_default_model_lists_known_flagships_but_lets_deepseek_pick() {
        assert_eq!(Protocol::MINIMAX.default_model(), Some("MiniMax-M3"));
        assert_eq!(Protocol::ZAI.default_model(), Some("glm-5.3-flash"));
        // deepseek: flagship is the caller's choice, no default
        assert!(Protocol::DEEPSEEK.default_model().is_none());
    }

    #[test]
    fn protocol_eq_and_debug_use_the_label() {
        assert_eq!(Protocol::MINIMAX, Protocol::MINIMAX);
        assert_ne!(Protocol::MINIMAX, Protocol::ZAI);
        assert_eq!(format!("{:?}", Protocol::DEEPSEEK), "deepseek");
    }

    #[test]
    fn vendors_table_matches_the_implementations() {
        let names: Vec<&str> = VENDORS.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["minimax", "zai", "deepseek"]);
        // every wire's handle is the matching constant
        assert_eq!(VENDORS[0].1, Protocol::MINIMAX);
        assert_eq!(VENDORS[1].1, Protocol::ZAI);
        assert_eq!(VENDORS[2].1, Protocol::DEEPSEEK);
    }

    // ---- Effort: the budget knob -----------------------------------------

    #[test]
    fn effort_label_is_lowercase() {
        assert_eq!(Effort::Low.label(), "low");
        assert_eq!(Effort::Medium.label(), "medium");
        assert_eq!(Effort::High.label(), "high");
        assert_eq!(Effort::Max.label(), "max");
    }

    #[test]
    fn effort_budget_clamps_into_the_messages_wire_rules() {
        // at least 1024, strictly under max_tokens, never zero
        let b = Effort::Low.budget(16_000);
        assert_eq!(b, 1024, "low is the floor");
        assert!(b < 16_000, "the reply always has room");
        // medium is 8192; high is 32768; max = max_tokens - 1024
        assert_eq!(Effort::Medium.budget(16_000), 8192);
        assert_eq!(Effort::High.budget(64_000), 32768);
        assert_eq!(Effort::Max.budget(64_000), 64_000 - 1024);
    }

    #[test]
    fn effort_budget_never_exceeds_max_tokens_minus_a_floor() {
        // when max_tokens is tiny, the clamp pins budget to the floor
        // (1024) — the messages wire requires at least that much, even
        // when the reply budget cannot fit it
        let b = Effort::Max.budget(2_000);
        assert!(b >= 1024, "the messages wire requires at least 1024: {b}");
        assert!(b <= 2_000, "the reply budget is never exceeded: {b}");
    }

    // ---- helpers the chat-completions wires share -------------------------

    #[test]
    fn chat_url_strips_a_trailing_slash() {
        let cfg = crate::config::Config {
            api_key: "k".into(),
            base_url: "https://api.example.com/".into(),
            model: "m".into(),
            protocol: Protocol::ZAI,
            cache: CacheMode::Auto, thinking: Thinking::Preserve, effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        assert_eq!(chat_url(&cfg), "https://api.example.com/chat/completions");
        // no trailing slash
        let cfg2 = crate::config::Config { base_url: "https://api.example.com".into(), ..cfg.clone() };
        assert_eq!(chat_url(&cfg2), "https://api.example.com/chat/completions");
    }

    #[test]
    fn chat_messages_prepends_the_system_prompt() {
        let cfg = crate::config::Config {
            api_key: "k".into(), base_url: "x".into(), model: "m".into(),
            protocol: Protocol::ZAI,
            cache: CacheMode::Auto, thinking: Thinking::Preserve, effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let msgs = vec![Message::User("hi".into())];
        let out = chat_messages("you are a bot", &msgs, &cfg);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[0]["content"], "you are a bot");
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn chat_tools_returns_an_array_of_schemas() {
        let schemas = vec![json!({"name": "a"}), json!({"name": "b"})];
        let v = chat_tools(&schemas);
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[test]
    fn chat_to_internal_parses_reasoning_text_and_tool_calls() {
        let v = json!({
            "choices": [{"message": {
                "reasoning_content": "I think",
                "content": "answer",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}}
                ]
            }}],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        // a minimal usage_of stub
        let usage_of = |u: &Value| (u.clone(), crate::display::Usage::default());
        let r = chat_to_internal(&v, usage_of);
        assert_eq!(r.blocks.len(), 3, "thinking, text, tool_use");
        match &r.blocks[0] {
            Block::Thinking { text, .. } => assert_eq!(text, "I think"),
            other => panic!("expected Thinking, got {other:?}"),
        }
        match &r.blocks[1] {
            Block::Text(t) => assert_eq!(t, "answer"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &r.blocks[2] {
            Block::ToolUse { id, name, input } => {
                assert_eq!(id, "tc1");
                assert_eq!(name, "bash");
                assert_eq!(input["command"], "ls");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn chat_to_internal_drops_empty_reasoning_and_content() {
        let v = json!({
            "choices": [{"message": {
                "reasoning_content": "",
                "content": "answer only"
            }}],
            "usage": {}
        });
        let usage_of = |_: &Value| (Value::Null, crate::display::Usage::default());
        let r = chat_to_internal(&v, usage_of);
        assert_eq!(r.blocks.len(), 1);
        assert!(matches!(&r.blocks[0], Block::Text(t) if t == "answer only"));
    }

    // ---- empty_usage / merge_usage / response_from_value -----------------

    #[test]
    fn empty_usage_is_four_zeros() {
        let v = empty_usage();
        assert_eq!(v["input_tokens"], 0);
        assert_eq!(v["output_tokens"], 0);
        assert_eq!(v["cache_read_input_tokens"], 0);
        assert_eq!(v["cache_creation_input_tokens"], 0);
    }

    #[test]
    fn merge_usage_merges_fields_when_both_are_objects() {
        let mut usage = empty_usage();
        let mut src = serde_json::Map::new();
        src.insert("input_tokens".into(), json!(42));
        src.insert("output_tokens".into(), json!(7));
        merge_usage(&mut usage, Some(&src));
        assert_eq!(usage["input_tokens"], 42);
        assert_eq!(usage["output_tokens"], 7);
    }

    #[test]
    fn merge_usage_is_a_no_op_when_src_is_none() {
        let mut usage = empty_usage();
        let before = usage.clone();
        merge_usage(&mut usage, None);
        assert_eq!(usage, before);
    }

    #[test]
    fn merge_usage_is_a_no_op_when_dst_is_not_an_object() {
        // empty_usage is an object; overwrite it with a non-object to
        // exercise the `if let (Some(dst), Some(src))` guard
        let mut usage = json!(42);
        let mut src = serde_json::Map::new();
        src.insert("input_tokens".into(), json!(1));
        merge_usage(&mut usage, Some(&src));
        assert_eq!(usage, json!(42), "not an object: nothing to merge into");
    }

    #[test]
    fn response_from_value_reads_content_and_usage() {
        let v = json!({
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "thinking", "thinking": "think", "signature": "sig"}
            ],
            "usage": {"input_tokens": 3}
        });
        let r = response_from_value(&v);
        assert_eq!(r.blocks.len(), 2);
        assert_eq!(r.usage["input_tokens"], 3);
    }

    #[test]
    fn response_from_value_missing_content_yields_empty_blocks() {
        let r = response_from_value(&json!({}));
        assert!(r.blocks.is_empty());
    }

    // ---- strip_thinking: the wire-side filter -----------------------------

    #[test]
    fn strip_thinking_drops_thinking_blocks_from_assistant_turns() {
        let blocks = vec![
            Block::Thinking { text: "thought".into(), signature: Some("s".into()) },
            Block::Text("answer".into()),
            Block::ToolUse { id: "t".into(), name: "bash".into(), input: json!({}) },
        ];
        let msgs = vec![Message::Assistant(blocks)];
        let stripped = strip_thinking(&msgs);
        match &stripped[0] {
            Message::Assistant(bs) => {
                assert_eq!(bs.len(), 2, "thinking dropped, text and tool_use kept");
                assert!(matches!(&bs[0], Block::Text(t) if t == "answer"));
                assert!(matches!(&bs[1], Block::ToolUse { .. }));
            }
            other => panic!("expected Assistant, got {other:?}"),
        }
    }

    #[test]
    fn strip_thinking_keeps_user_and_tool_results_intact() {
        // the `other => other.clone()` arm: user and tool-results pass
        // through without inspection
        let msgs = vec![
            Message::User("task".into()),
            Message::ToolResults(vec![crate::ir::ToolResult { id: "t".into(), content: "ok".into() }]),
        ];
        let stripped = strip_thinking(&msgs);
        assert_eq!(stripped.len(), 2);
        assert!(matches!(&stripped[0], Message::User(t) if t == "task"));
        assert!(matches!(&stripped[1], Message::ToolResults(rs) if rs.len() == 1));
    }

    // ---- the vendor rules: minimax/zai/deepseek accepts ------------------

    #[test]
    fn minimax_accepts_only_refuses_effort_with_strip_thinking() {
        // minimax's only rule: effort requires PreserveThinking. Cache
        // mode is irrelevant on the Messages wire. The truth table is
        // 2 cache × 2 thinking × 5 effort combinations; we cover the
        // cases that would catch a swap of the condition.
        // cache=Active, thinking=Preserve: every effort is fine
        for e in [None, Some(Effort::Low), Some(Effort::Medium), Some(Effort::High), Some(Effort::Max)] {
            assert!(minimax_accepts(CacheMode::Active, Thinking::Preserve, e).is_ok(),
                "active+preserve+{e:?} should pass");
        }
        // cache=Auto, thinking=Strip: only None effort passes
        assert!(minimax_accepts(CacheMode::Auto, Thinking::Strip, None).is_ok());
        assert!(minimax_accepts(CacheMode::Auto, Thinking::Strip, Some(Effort::Low)).is_err());
        assert!(minimax_accepts(CacheMode::Auto, Thinking::Strip, Some(Effort::Medium)).is_err());
        assert!(minimax_accepts(CacheMode::Auto, Thinking::Strip, Some(Effort::High)).is_err());
        assert!(minimax_accepts(CacheMode::Auto, Thinking::Strip, Some(Effort::Max)).is_err());
        // cache=Auto, thinking=Preserve: every effort is fine
        for e in [None, Some(Effort::Low), Some(Effort::Medium), Some(Effort::High), Some(Effort::Max)] {
            assert!(minimax_accepts(CacheMode::Auto, Thinking::Preserve, e).is_ok(),
                "auto+preserve+{e:?} should pass");
        }
        // cache=Active, thinking=Strip: same rule (cache is ignored)
        assert!(minimax_accepts(CacheMode::Active, Thinking::Strip, None).is_ok());
        assert!(minimax_accepts(CacheMode::Active, Thinking::Strip, Some(Effort::High)).is_err());
    }

    #[test]
    fn zai_rejects_active_cache_and_effort_with_strip() {
        // zai: cache=Active is always refused; otherwise the same
        // effort+strip rule as minimax.
        // cache=Active: every combo is refused
        for (t, e) in [(Thinking::Preserve, None), (Thinking::Strip, None),
                       (Thinking::Preserve, Some(Effort::High)),
                       (Thinking::Strip, Some(Effort::Low))] {
            assert!(zai_accepts(CacheMode::Active, t, e).is_err(),
                "active+{t:?}+{e:?} should fail");
        }
        // cache=Auto, thinking=Preserve: every effort is fine
        for e in [None, Some(Effort::Low), Some(Effort::Medium), Some(Effort::High), Some(Effort::Max)] {
            assert!(zai_accepts(CacheMode::Auto, Thinking::Preserve, e).is_ok(),
                "auto+preserve+{e:?} should pass");
        }
        // cache=Auto, thinking=Strip: only None effort passes
        assert!(zai_accepts(CacheMode::Auto, Thinking::Strip, None).is_ok());
        for e in [Some(Effort::Low), Some(Effort::Medium), Some(Effort::High), Some(Effort::Max)] {
            assert!(zai_accepts(CacheMode::Auto, Thinking::Strip, e).is_err(),
                "auto+strip+{e:?} should fail");
        }
    }

    #[test]
    fn deepseek_rejects_active_cache_and_effort_with_strip() {
        // deepseek: same two-rule policy as zai.
        for (t, e) in [(Thinking::Preserve, None), (Thinking::Strip, None),
                       (Thinking::Preserve, Some(Effort::High)),
                       (Thinking::Strip, Some(Effort::Low))] {
            assert!(deepseek_accepts(CacheMode::Active, t, e).is_err(),
                "active+{t:?}+{e:?} should fail");
        }
        for e in [None, Some(Effort::Low), Some(Effort::Medium), Some(Effort::High), Some(Effort::Max)] {
            assert!(deepseek_accepts(CacheMode::Auto, Thinking::Preserve, e).is_ok(),
                "auto+preserve+{e:?} should pass");
        }
        assert!(deepseek_accepts(CacheMode::Auto, Thinking::Strip, None).is_ok());
        for e in [Some(Effort::Low), Some(Effort::Medium), Some(Effort::High), Some(Effort::Max)] {
            assert!(deepseek_accepts(CacheMode::Auto, Thinking::Strip, e).is_err(),
                "auto+strip+{e:?} should fail");
        }
    }
}
