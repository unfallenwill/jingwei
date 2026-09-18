// Zhipu's zai speaks the chat-completions dialect with a vocabulary of
// its own on top, all of it staying here. Probed live before this was
// written: the flagship (glm-5.3-flash) *always* thinks — `thinking:
// disabled` and any reasoning_effort outside low/high/max are hard 400s
// ("该模型始终思考，不支持关闭思考") — reasoning streams as
// reasoning_content beside content, usage carries
// prompt_tokens_details.cached_tokens from an implicit cache that needs
// no breakpoints, and `clear_thinking: false` keeps prior assistant
// turns' reasoning in context: preserved thinking, recommended for
// coding/agents precisely because the echoed reasoning is part of the
// cached prefix.

use super::{chat_blocking, chat_messages, chat_streaming, chat_to_internal, chat_tools,
            Thinking};
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::display::Show;
use crate::ir::{Message, Response};
use crate::display::Usage;
use crate::Result;
use serde_json::{json, Value};

/// The zai vendor's one entry into the provider port.
pub(super) async fn turn(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    if cfg.streaming {
        streaming(cfg, history, schemas, token, sink).await
    } else {
        blocking(cfg, history, schemas, token, sink).await
    }
}

/// The effort word this wire understands — its vocabulary, not the knob's:
/// low/high/max are its own words, and `medium` (a word it rejects with a
/// 400 on the flagship) folds into `high`, the tier the vendor itself
/// maps it to on models that do accept it.
fn effort_word(e: super::Effort) -> &'static str {
    use super::Effort;
    match e { Effort::Low => "low", Effort::Medium | Effort::High => "high", Effort::Max => "max" }
}

/// The zai request body: the dialect's shape plus the thinking object.
/// Preserve is *preserved thinking* here — `clear_thinking: false`, the
/// vendor's own recommendation for coding/agents, keeping prior turns'
/// reasoning in the context (and in the cached prefix). Strip sends no
/// thinking object at all: the flagship cannot stop thinking (`disabled`
/// is a hard 400), so strip on this wire means "not kept", never "off" —
/// the history blocks drop in the dialect's message translation.
pub(super) fn body(cfg: &Config, messages: &[Message], schemas: &[Value], stream: bool) -> Value {
    let system_text = crate::agents_md::full_system_prompt(&cfg.agents_md_extra);
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "messages": chat_messages(&system_text, messages, cfg), "tools": chat_tools(schemas)});
    if cfg.thinking == Thinking::Preserve {
        body["thinking"] = json!({"type": "enabled", "clear_thinking": false});
    }
    if let Some(e) = cfg.effort {
        body["reasoning_effort"] = json!(effort_word(e));
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
pub(super) fn usage(u: &Value) -> (Value, Usage) {
    let prompt = u["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    let fresh = prompt.saturating_sub(cached);
    let out = u["completion_tokens"].as_u64().unwrap_or(0);
    (
        json!({"input_tokens": fresh, "output_tokens": out,
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}),
        Usage { input: fresh, output: out, cache_read: cached, cache_write: 0 },
    )
}

/// Chat Completions response → internal shape; the dialect's walk, this
/// vendor's ledger.
fn to_internal(v: &Value) -> Response {
    chat_to_internal(v, usage)
}

async fn blocking(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    chat_blocking(cfg, body(cfg, messages, schemas, false), to_internal, token, sink).await
}

async fn streaming(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    chat_streaming(cfg, body(cfg, messages, schemas, true), usage, token, sink).await
}

#[cfg(test)]
mod tests {
    use crate::api::{call_api, Protocol, Thinking as ApiThinking};
    use crate::api::{Effort, Thinking as CThinking};
    use crate::test_util::{block_on, cfg, mock, sink, BlockExt};
    use serde_json::json;

    #[test]
    fn body_carries_clear_thinking_and_folds_medium_into_high() {
        let mut c = cfg("https://x".into(), true);
        c.protocol = Protocol::ZAI;
        let b = super::body(&c, &[], &[], false);
        assert_eq!(b["thinking"], json!({"type": "enabled", "clear_thinking": false}));
        assert!(b.get("reasoning_effort").is_none());
        c.effort = Some(Effort::Medium);
        let b = super::body(&c, &[], &[], false);
        assert_eq!(b["reasoning_effort"], json!("high"), "medium folds into high");
        c.thinking = ApiThinking::Strip;
        let b = super::body(&c, &[], &[], false);
        assert!(b.get("thinking").is_none(), "strip on zai means no thinking object");
    }

    /// AGENTS.md content rides through `body()` as the system prompt on
    /// chat-completions wires. Pin it down: when `cfg.agents_md_extra`
    /// is set, the system message the wire carries contains both the
    /// base `SYSTEM` and the extras.
    #[test]
    fn body_includes_agents_md_extras_in_the_system_message() {
        use crate::test_util::temp_dir;
        let dir = temp_dir("agents_md_wire");
        std::fs::write(dir.join("AGENTS.md"), "always use pnpm").unwrap();
        let ctx = crate::agents_md::AgentsMdContext::load(&dir);
        assert!(ctx.found_path.is_some());

        let mut c = cfg("https://x".into(), true);
        c.protocol = Protocol::ZAI;
        c.agents_md_extra = ctx.system_prompt_extras();
        let msgs = vec![crate::ir::Message::User("task".into())];
        let b = super::body(&c, &msgs, &[], false);
        let system = b["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains(crate::SYSTEM), "base SYSTEM stays in the wire");
        assert!(system.contains("always use pnpm"), "AGENTS.md body rides in");
        assert!(system.contains("# Project conventions"), "extras are labelled");
        assert!(system.contains(&dir.display().to_string()),
            "banner path lands too, so the model knows where the content came from");
    }

    #[test]
    fn streaming_assembles_deltas() {
        let stream = concat!(
            r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"reasoning_content":"thinking…"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"content":"hello "}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"content":"world"}}]}"#, "\n\n",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"bash","arguments":"{\"command\":\"echo hi\"}"}}]}}]}"#, "\n\n",
            r#"data: {"usage":{"prompt_tokens":12,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":5}}}"#, "\n\n",
            r#"data: [DONE]"#, "\n\n");
        let port = mock(stream, 200, true);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), true);
        c.protocol = Protocol::ZAI;
        let token = crate::cancel::CancelToken::new();
        let (resp, _) = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap();
        assert_eq!(resp.blocks.len(), 3, "thinking + text + tool_use");
        let thinking_text = match &resp.blocks[0] {
            crate::ir::Block::Thinking { text, .. } => text.clone(),
            _ => panic!("expected thinking block, got {:?}", resp.blocks[0]),
        };
        assert_eq!(thinking_text, "thinking…");
        assert_eq!(resp.blocks[1].text().unwrap(), "hello world");
        assert_eq!(resp.blocks[2].name(), "bash");
        assert_eq!(resp.blocks[2].input(), &json!({"command": "echo hi"}));
        assert_eq!(resp.usage["input_tokens"], 7);
        assert_eq!(resp.usage["cache_read_input_tokens"], 5);
        assert_eq!(resp.usage["output_tokens"], 4);
    }

    #[test]
    fn blocking_roundtrip() {
        let body = json!({"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":3,"completion_tokens":1}}).to_string().into_boxed_str();
        let port = mock(&body, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::ZAI;
        let token = crate::cancel::CancelToken::new();
        let (resp, _) = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap();
        assert_eq!(resp.blocks[0].text().unwrap(), "ok");
    }

    #[test]
    fn rules_and_defaults_live_with_the_vendor() {
        assert_eq!(Protocol::ZAI.default_base(), Some("https://open.bigmodel.cn/api/paas/v4"));
        assert_eq!(Protocol::ZAI.default_model(), Some("glm-5.3-flash"));
        let err = Protocol::ZAI.accepts(crate::api::CacheMode::Active, CThinking::Preserve, None).unwrap_err();
        assert!(err.to_string().contains("zai"), "got: {err}");
        let err = Protocol::ZAI.accepts(crate::api::CacheMode::Auto, CThinking::Strip, Some(Effort::High)).unwrap_err();
        assert!(err.to_string().contains("interleaved"), "got: {err}");
        Protocol::ZAI.accepts(crate::api::CacheMode::Auto, CThinking::Preserve, None).unwrap();
        Protocol::ZAI.accepts(crate::api::CacheMode::Auto, CThinking::Preserve, Some(Effort::High)).unwrap();
        Protocol::ZAI.accepts(crate::api::CacheMode::Auto, CThinking::Strip, None).unwrap();
    }

    #[test]
    fn usage_reads_the_implicit_cache_ledger() {
        let u = json!({"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}});
        let (wire, shown) = super::usage(&u);
        assert_eq!(wire["input_tokens"], 6);
        assert_eq!(wire["cache_read_input_tokens"], 4);
        assert_eq!(wire["output_tokens"], 3);
        assert_eq!(shown.input, 6);
        assert_eq!(shown.cache_read, 4);
    }
}
