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
//
// The transcript reaches the wire through the context and nowhere else: a
// vendor's `turn` reads `ctx.history()`. One owner, one path — the context
// and the messages the model sees cannot drift apart.

use crate::config::Config;
use crate::cancel::{blocking, CancelToken};
use crate::core::context::Context;
use crate::ui::{Msg, Show};
use crate::core::ir::{Block, Message, Response};
use crate::ui::Usage;
use crate::{Error, Result};
use serde_json::{json, Value};
use std::future::Future;
use std::io::{BufRead, BufReader};
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::mpsc;

/// The boxed future a vendor's `turn` returns. Boxed because the wire
/// is chosen at runtime — one allocation per request, and the vendor's
/// streaming/blocking machinery stays its own. The third slot is the
/// rendered wire body the vendor actually sent: the session records it
/// verbatim.
pub(crate) type Turn<'a> = Pin<Box<dyn Future<Output = Result<(Response, bool, Value)>> + Send + 'a>>;

/// The provider port. One impl per wire; the composition root holds a
/// [`Protocol`] handle and calls through it — no `match` anywhere. Adding a
/// vendor is one `impl` and one row in [`VENDORS`], both the compiler's
/// problem to keep complete.
///
/// The transcript comes from the context: `turn` reads `ctx.history()`, so
/// there is one path to the wire and a context and its messages can never
/// drift apart.
pub(crate) trait Vendor: Sync {
    fn label(&self) -> &'static str;
    fn default_base(&self) -> Option<&'static str>;
    fn default_model(&self) -> Option<&'static str>;
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()>;
    fn turn<'a>(
        &'a self,
        cfg: &'a Config,
        ctx: &'a Context,
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
    fn turn<'a>(&'a self, cfg: &'a Config, ctx: &'a Context, token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(minimax::turn(cfg, ctx, token, sink))
    }
}

impl Vendor for Zai {
    fn label(&self) -> &'static str { "zai" }
    fn default_base(&self) -> Option<&'static str> { Some("https://open.bigmodel.cn/api/paas/v4") }
    fn default_model(&self) -> Option<&'static str> { Some("glm-5.3-flash") }
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        zai_accepts(cache, thinking, effort)
    }
    fn turn<'a>(&'a self, cfg: &'a Config, ctx: &'a Context, token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(zai::turn(cfg, ctx, token, sink))
    }
}

impl Vendor for DeepSeek {
    fn label(&self) -> &'static str { "deepseek" }
    fn default_base(&self) -> Option<&'static str> { Some("https://api.deepseek.com") }
    fn default_model(&self) -> Option<&'static str> { None }
    fn accepts(&self, cache: CacheMode, thinking: Thinking, effort: Option<Effort>) -> Result<()> {
        deepseek_accepts(cache, thinking, effort)
    }
    fn turn<'a>(&'a self, cfg: &'a Config, ctx: &'a Context, token: &'a CancelToken, sink: &'a dyn Show)
        -> Turn<'a>
    {
        Box::pin(deepseek::turn(cfg, ctx, token, sink))
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

pub(crate) async fn call_api(cfg: &Config, ctx: &Context, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    cfg.protocol.0.turn(cfg, ctx, token, sink).await
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
    ureq::AgentBuilder::new().timeout_read(Duration::from_secs(180)).timeout_write(Duration::from_secs(60)).build()
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

async fn request<T: Send + 'static>(
    token: &CancelToken, key: &str, url: &str, body: &Value, messages_wire: bool, decode: fn(ureq::Response) -> Result<T>,
) -> Result<T> {
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
                Ok(_) => if tx.blocking_send(line.trim_end_matches(['\r', '\n']).to_string()).is_err() { break; }
            }
        }
    });
    rx
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

/// Compose the full system-side text: `system_text` joined to each
/// reminder (reminders own their own separators). Used by every
/// chat-completions vendor and reused by the Messages wire's
/// inactive-cache branch. The shape mirrors the old `compose_system`
/// so cross-session cache keys stay stable.
pub(super) fn chat_system_text(ctx: &Context) -> String {
    let mut out = ctx.system_text.clone();
    for r in &ctx.reminders {
        out.push_str(&r.text);
    }
    out
}

fn chat_messages(ctx: &Context, messages: &[Message]) -> Vec<Value> {
    let system = chat_system_text(ctx);
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(json!({"role": "system", "content": system}));
    for m in messages {
        out.push(Message::to_value(m));
    }
    out
}

fn chat_tools(schemas: &[Value]) -> Value {
    Value::Array(schemas.to_vec())
}

/// The streaming options every chat-completions vendor appends:
/// `stream: true` and the usage-bearing `stream_options`. Kept here
/// so the two wires spell it once.
fn chat_stream_opts(stream: bool, body: &mut Value) {
    if stream { body["stream"] = json!(true); body["stream_options"] = json!({"include_usage": true}); }
}

async fn chat_blocking(cfg: &Config, body: Value, to_internal: fn(&Value) -> Response, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    let url = chat_url(cfg);
    let key = cfg.api_key.clone();
    let resp = request(token, &key, &url, &body, false, |r| {
        let v: Value = r.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    let resp = to_internal(&resp);
    show_turn(&resp, sink);
    Ok((resp, token.is_cancelled(), body))
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
    blocks.extend(msg["tool_calls"].as_array().into_iter().flatten().map(|tc| Block::ToolUse {
        id: tc["id"].as_str().unwrap_or("").to_string(),
        name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
        input: serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({})),
    }));
    let (usage, _) = usage_of(&v["usage"]);
    Response { blocks, usage }
}

async fn chat_streaming(cfg: &Config, body: Value, usage_of: fn(&Value) -> (Value, Usage), token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
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
    blocks.extend(tool_calls.into_iter().map(|tc| Block::ToolUse {
        id: tc["id"].as_str().unwrap_or("").to_string(),
        name: tc["name"].as_str().unwrap_or("").to_string(),
        input: serde_json::from_str::<Value>(tc["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({})),
    }));
    Ok((Response { blocks, usage }, interrupted, body))
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
    use crate::core::ir::{Block, Message};
    use crate::test_util::{block_on, cfg, mock, mock_seq, sink};
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
            max_agents: crate::config::DEFAULT_MAX_AGENTS,
            agent_wait_ms: crate::config::DEFAULT_AGENT_WAIT_MS,
        };
        assert_eq!(chat_url(&cfg), "https://api.example.com/chat/completions");
        // no trailing slash
        let cfg2 = crate::config::Config { base_url: "https://api.example.com".into(), ..cfg.clone() };
        assert_eq!(chat_url(&cfg2), "https://api.example.com/chat/completions");
    }

    #[test]
    fn chat_messages_prepends_the_system_prompt() {
        let msgs = vec![Message::User("hi".into())];
        // an empty reminders list ⇒ the system message is just `system_text`
        let cx = crate::test_util::ctx();
        let out = chat_messages(&cx, &msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[0]["content"], cx.system_text);
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn chat_system_text_concatenates_reminders() {
        // reminders own their own leading separator; the composer
        // concatenates verbatim with no extra insertion
        let dir = crate::test_util::temp_dir("chat_sys_compose");
        std::fs::write(dir.join("AGENTS.md"), "x").unwrap();
        let md = crate::core::agents_md::AgentsMdContext::load(&dir);
        let cx = crate::core::context::Context::new(&md, crate::config::DEFAULT_CONTEXT_SIZE);
        let s = chat_system_text(&cx);
        assert!(s.starts_with(crate::core::prompt::SYSTEM),
            "system_text comes first");
        assert!(s.contains("# Project conventions"),
            "AGENTS.md section rides in");
        // wire-shape parity with the old `compose_system`: between SYSTEM
        // and `# Project conventions` there should be exactly the
        // reminder's leading "\n\n" — two newlines, the same as today.
        let sep_idx = s.find("# Project conventions").unwrap();
        let between = &s[..sep_idx];
        assert!(between.ends_with("\n\n"),
            "exactly one blank line between SYSTEM and the AGENTS.md section");
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
        let usage_of = |u: &Value| (u.clone(), crate::ui::Usage::default());
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
        let usage_of = |_: &Value| (Value::Null, crate::ui::Usage::default());
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
            Message::ToolResults(vec![crate::core::ir::ToolResult { id: "t".into(), content: "ok".into() }]),
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

    // ---- retry: the transport's own rule ----------------------------------

    #[test]
    fn backoff_doubles_then_caps() {
        assert_eq!(backoff(1), std::time::Duration::from_millis(500));
        assert_eq!(backoff(2), std::time::Duration::from_millis(1000));
        assert_eq!(backoff(3), std::time::Duration::from_millis(2000));
        assert_eq!(backoff(4), std::time::Duration::from_millis(4000));
        assert_eq!(backoff(5), RETRY_MAX, "doubling is capped");
        assert_eq!(backoff(99), RETRY_MAX, "and stays capped");
    }

    /// A response's blocks as JSON, read like a `content` array.
    fn blocks(b: &[Block]) -> Value {
        Value::Array(b.iter().map(Block::to_value).collect())
    }

    #[test]
    fn a_transient_failure_is_resent_through_the_one_door() {
        // 429 first, then the real answer: the door resends and the turn
        // succeeds where it used to die on the first status
        let (port, reqs) = mock_seq(vec![
            (429, r#"{"error":{"message":"rate limited"}}"#.into()),
            (200, r#"{"content":[{"type":"text","text":"after retry"}],"usage":{}}"#.into()),
        ]);
        let cx = crate::test_util::ctx();
        let (resp, _, _) = block_on(call_api(
            &cfg(format!("http://127.0.0.1:{port}"), false), &cx, &CancelToken::new(), &sink(),
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
        let cx = crate::test_util::ctx();
        let err = block_on(call_api(
            &cfg(format!("http://127.0.0.1:{port}"), false), &cx, &CancelToken::new(), &sink(),
        )).unwrap_err();
        assert!(err.to_string().contains("bad model"), "got: {err}");
        assert_eq!(reqs.lock().unwrap().len(), 1, "no second request");
    }

    // ---- the transport: what the wire says, and how it is read ------------

    use crate::test_util::{BlockExt, Recorder};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    /// One turn's worth of blocks, as the transport shows them.
    fn turn(blocks: Vec<Block>) -> Response {
        Response { blocks, usage: json!({"input_tokens": 1, "output_tokens": 2}) }
    }

    /// A chat stream served by `events`, and what it produced.
    fn chat_stream(events: &str) -> (Response, bool, Vec<crate::ui::Msg>) {
        let port = mock(events, 200, true);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), true);
        c.protocol = Protocol::ZAI;
        let rec = Recorder::default();
        let (resp, interrupted, _) =
            block_on(chat_streaming(&c, json!({}), super::zai::usage, &CancelToken::new(), &rec)).unwrap();
        (resp, interrupted, rec.msgs())
    }

    #[test]
    fn a_thinking_block_is_shown_as_a_think_and_a_think_end() {
        let rec = Recorder::default();
        show_turn(&turn(vec![Block::Thinking { text: "thought".into(), signature: None }]), &rec);
        let msgs = rec.msgs();
        assert!(msgs.iter().any(|m| matches!(m, Msg::Think(t) if t == "thought")), "{msgs:?}");
        assert!(msgs.iter().any(|m| matches!(m, Msg::ThinkEnd)), "{msgs:?}");
    }

    #[test]
    fn a_block_the_turn_does_not_show_is_skipped() {
        let rec = Recorder::default();
        show_turn(&turn(vec![Block::ToolUse { id: "t1".into(), name: "bash".into(), input: json!({}) }]), &rec);
        let msgs = rec.msgs();
        assert!(!msgs.iter().any(|m| matches!(m, Msg::Text(_))), "a tool call is not text: {msgs:?}");
        assert!(!msgs.iter().any(|m| matches!(m, Msg::Think(_))), "{msgs:?}");
        assert!(msgs.iter().any(|m| matches!(m, Msg::Done)), "the turn still closes: {msgs:?}");
    }

    #[test]
    fn a_chat_wire_error_is_reported_in_the_wires_own_words() {
        let (port, reqs) = mock_seq(vec![
            (200, r#"{"error":{"message":"model not found"}}"#.into()),
        ]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::ZAI;
        let err = block_on(chat_blocking(&c, json!({}), |v| chat_to_internal(v, super::zai::usage),
                                        &CancelToken::new(), &sink())).unwrap_err();
        assert!(err.to_string().contains("model not found"), "the wire's own words: {err}");
        assert_eq!(reqs.lock().unwrap().len(), 1, "a wire error is the request's own: no resend");
    }

    #[test]
    fn a_chat_answer_that_is_not_json_says_so() {
        let (port, _) = mock_seq(vec![(200, "this is not json at all".into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::ZAI;
        let err = block_on(chat_blocking(&c, json!({}), |v| chat_to_internal(v, super::zai::usage),
                                        &CancelToken::new(), &sink())).unwrap_err();
        assert!(matches!(err, Error::Io(_)), "the body was not json: {err:?}");
        assert!(err.to_string().contains("JSON"), "the parser says why: {err}");
    }

    #[test]
    fn a_chat_answer_with_only_tool_calls_has_no_text() {
        let v = json!({"choices": [{"message": {"tool_calls": [
            {"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}
        ]}}], "usage": {}});
        let r = chat_to_internal(&v, |_: &Value| (Value::Null, Usage::default()));
        assert_eq!(r.blocks.len(), 1, "the tool call is the whole answer");
        assert!(matches!(&r.blocks[0], Block::ToolUse { name, .. } if name == "bash"));
    }

    #[test]
    fn a_chat_stream_the_server_refuses_says_so() {
        let (port, _) = mock_seq(vec![(400, r#"{"error":{"message":"bad model"}}"#.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), true);
        c.protocol = Protocol::ZAI;
        let err = block_on(chat_streaming(&c, json!({}), super::zai::usage,
                                          &CancelToken::new(), &sink())).unwrap_err();
        assert!(err.to_string().contains("bad model"), "got: {err}");
    }

    #[test]
    fn a_chat_stream_ignores_a_line_that_is_not_json() {
        let (resp, _, _) = chat_stream(concat!(
            "data: not json at all\n\n",
            r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#, "\n\n",
            "data: [DONE]\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        assert_eq!(resp.blocks[0].text().unwrap(), "hi");
    }

    #[test]
    fn a_reasoning_block_closes_when_the_text_arrives() {
        let (resp, _, msgs) = chat_stream(concat!(
            r#"data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"content":"answer"}}]}"#, "\n\n",
            "data: [DONE]\n\n"));
        let shape: Vec<&str> = msgs.iter().map(|m| match m {
            Msg::Think(_) => "think", Msg::ThinkEnd => "think_end", Msg::Text(_) => "text", Msg::Done => "done",
            _ => "other",
        }).collect();
        assert_eq!(shape, vec!["think", "think_end", "text", "done"],
            "the reasoning closes before the answer: {msgs:?}");
        assert_eq!(resp.blocks.len(), 2, "the thinking and the text");
    }

    #[test]
    fn a_reasoning_block_closes_when_a_tool_call_arrives() {
        let (resp, _, msgs) = chat_stream(concat!(
            r#"data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"bash","arguments":"{}"}}]}}]}"#, "\n\n",
            "data: [DONE]\n\n"));
        let shape: Vec<&str> = msgs.iter().map(|m| match m {
            Msg::Think(_) => "think", Msg::ThinkEnd => "think_end", Msg::Done => "done", _ => "other",
        }).collect();
        assert_eq!(shape, vec!["think", "think_end", "done"],
            "the tool call closed the reasoning: {msgs:?}");
        assert_eq!(resp.blocks.len(), 2, "the reasoning and the tool call");
        assert!(matches!(&resp.blocks[1], Block::ToolUse { name, .. } if name == "bash"));
    }

    #[test]
    fn a_tool_call_delta_with_an_empty_id_or_name_is_ignored() {
        let (resp, _, _) = chat_stream(concat!(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":""}}]}}]}"#, "\n\n",
            "data: [DONE]\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        match &resp.blocks[0] {
            Block::ToolUse { id, name, input } => {
                assert_eq!(id, "", "an empty id is not an id");
                assert_eq!(name, "", "an empty name is not a name");
                assert_eq!(input, &json!({}), "no arguments arrived");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn a_chat_stream_with_only_tool_calls_has_no_text() {
        let (resp, _, _) = chat_stream(concat!(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"bash","arguments":"{}"}}]}}]}"#, "\n\n",
            "data: [DONE]\n\n"));
        assert_eq!(resp.blocks.len(), 1, "the tool call is the whole answer");
        assert!(matches!(&resp.blocks[0], Block::ToolUse { name, .. } if name == "bash"));
    }

    /// An event stream that never ends: the head, the given events, and
    /// then the connection held open. The consumer waits for content that
    /// never arrives — which is what a cancellation interrupts.
    fn mock_open_stream(events: &str) -> (u16, Arc<AtomicBool>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let written = Arc::new(AtomicBool::new(false));
        let flag = written.clone();
        let events = events.to_owned();
        std::thread::spawn(move || {
            let Ok((mut s, _)) = listener.accept() else { return };
            let mut buf = [0u8; 8192];
            let _ = s.read(&mut buf);
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
            let _ = s.write_all(events.as_bytes());
            let _ = s.flush();
            flag.store(true, Ordering::SeqCst);
            // The connection is held open: the stream never ends.
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        (port, written)
    }

    #[test]
    fn a_cancelled_chat_stream_says_it_was_interrupted() {
        let (port, written) = mock_open_stream(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n");
        let mut c = cfg(format!("http://127.0.0.1:{port}"), true);
        c.protocol = Protocol::ZAI;
        let token = CancelToken::new();
        let rec = Recorder::default();
        let (resp, interrupted, _) = block_on(async {
            let t = token.clone();
            let seen = rec.clone();
            let flag = written.clone();
            // The cancel lands after the client *delivered* the text: the
            // sink has seen it, so the interruption cannot cut the answer
            // short of what the model already said. (Waiting for the
            // bytes on the wire alone is not enough — the client may not
            // have read them yet.)
            tokio::task::spawn_blocking(move || {
                let deadline = Instant::now() + Duration::from_secs(5);
                while !flag.load(Ordering::SeqCst) {
                    if Instant::now() > deadline { panic!("the mock never wrote the stream"); }
                    std::thread::sleep(Duration::from_millis(5));
                }
                while !seen.msgs().iter().any(|m| matches!(m, Msg::Text(t) if t.contains("partial"))) {
                    if Instant::now() > deadline { panic!("the client never delivered the text"); }
                    std::thread::sleep(Duration::from_millis(5));
                }
                t.cancel();
            });
            chat_streaming(&c, json!({}), super::zai::usage, &token, &rec).await
        }).unwrap();
        assert!(interrupted, "the stream was cut short by the cancellation");
        assert_eq!(resp.blocks[0].text().unwrap(), "partial", "what the model said before the cut");
    }

    /// A stream with no end: the server keeps writing until the client
    /// goes away. The write failing is how the server learns that the
    /// reader let go of the connection.
    fn mock_endless_stream() -> (u16, Arc<AtomicBool>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let gone = Arc::new(AtomicBool::new(false));
        let flag = gone.clone();
        std::thread::spawn(move || {
            let Ok((mut s, _)) = listener.accept() else { return };
            let mut buf = [0u8; 8192];
            let _ = s.read(&mut buf);
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
            let line = b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n";
            let mut written = 0usize;
            while written < 64 * 1024 * 1024 {
                match s.write_all(line) {
                    Ok(()) => written += line.len(),
                    Err(_) => { flag.store(true, Ordering::SeqCst); return; }
                }
            }
        });
        (port, gone)
    }

    #[test]
    fn the_sse_reader_stops_when_nobody_is_listening() {
        let (port, gone) = mock_endless_stream();
        let resp = post("k", format!("http://127.0.0.1:{port}/stream"), json!({}), false).unwrap();
        let rx = sse_channel(resp);
        drop(rx);   // nobody is listening any more
        // The reader lets go of the connection instead of reading the rest
        // of the stream — the server sees the client go away.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !gone.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the reader kept reading the stream");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn a_cancellation_during_the_backoff_stops_the_retry() {
        // A 500 is transient: the transport would send the request again.
        // The cancel lands inside the backoff window instead, so the
        // retry never goes out and the first failure is what comes back.
        let token = CancelToken::new();
        let (port, reqs) = mock_seq(vec![
            (500, r#"{"error":{"message":"the server is on fire"}}"#.into()),
            (200, r#"{"choices":[{"message":{"content":"never reached"}}],"usage":{}}"#.into()),
        ]);
        let t = token.clone();
        let seen = reqs.clone();
        let gave_up = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = gave_up.clone();
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::ZAI;
        let err = block_on(async move {
            tokio::spawn(async move {
                // The request went out (the mock has it), so the answer is
                // on its way and the client is about to sleep in the
                // backoff. The wait is for the fact, not the clock — and
                // it is bounded: a mock that never sees the request must
                // fail this test, not hang it.
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while seen.lock().unwrap().is_empty() {
                    if std::time::Instant::now() > deadline {
                        flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        t.cancel();
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                // The 500 is on the wire by the time the mock logs the
                // request, so the client is a moment away from reading it
                // and falling into the 500 ms backoff. The cancel has to
                // land in that window: before it the request is
                // interrupted instead, after it the retry goes out.
                tokio::time::sleep(Duration::from_millis(100)).await;
                t.cancel();
            });
            chat_blocking(&c, json!({}), |v| chat_to_internal(v, super::zai::usage), &token, &sink()).await
        }).unwrap_err();
        assert!(!gave_up.load(std::sync::atomic::Ordering::SeqCst),
            "the mock never saw the request — the test's premise is gone");
        assert_eq!(err.to_string(), r#"api 500: {"error":{"message":"the server is on fire"}}"#,
            "the first failure is what comes back, in the wire's own words");
        assert_eq!(reqs.lock().unwrap().len(), 1, "the retry never went out");
    }
}
