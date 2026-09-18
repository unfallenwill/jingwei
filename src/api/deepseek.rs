// The DeepSeek vendor: chat-completions dialect with a vocabulary of
// its own on top — a `thinking` toggle, an effort spelling, and a disk
// cache that is always on and reports itself in `usage`. All of it stays
// here; the core asks nothing, the family knows nothing.

use super::{chat_blocking, chat_messages, chat_streaming, chat_to_internal, chat_tools, Thinking};
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::context::Context;
use crate::display::Show;
use crate::display::Usage;
use crate::ir::{Message, Response};
use crate::Result;
use serde_json::{json, Value};

/// The deepseek vendor's one entry into the provider port.
pub(super) async fn turn(cfg: &Config, ctx: &Context, history: &[Message], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    if cfg.streaming {
        streaming(cfg, ctx, history, token, sink).await
    } else {
        blocking(cfg, ctx, history, token, sink).await
    }
}

/// The deepseek request body: the dialect's shape plus the `thinking`
/// toggle. On this wire `--thinking strip` is not an erasure but the
/// toggle off — `disabled` makes the model generate no reasoning at all,
/// which is the only honest strip on a wire whose echo rule (tools ⇒
/// reasoning back) a stripped history would break on the second request.
/// Effort rides `reasoning_effort` verbatim: low/high/max are the wire's
/// own words, and it maps medium→high itself for compatibility.
///
/// The system prompt is built by `chat_messages` from `Context.system_text`
/// (just `crate::SYSTEM`) plus every reminder's text — AGENTS.md content
/// rides in a `AgentsMdClosest` reminder today; runtime context will ride
/// here tomorrow. The chat-completions family has a single system-message
/// slot, so reminders concatenate inline; the wire-shape matches the old
/// `compose_system` output so cache keys stay stable.
pub(super) fn body(cfg: &Config, ctx: &Context, messages: &[Message], stream: bool) -> Value {
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "thinking": {"type": if cfg.thinking == Thinking::Strip { "disabled" } else { "enabled" }},
        "messages": chat_messages(ctx, messages, cfg), "tools": chat_tools(&ctx.tools)});
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
pub(super) fn usage(u: &Value) -> (Value, Usage) {
    let cached = u["prompt_cache_hit_tokens"].as_u64().unwrap_or(0);
    let fresh = u["prompt_cache_miss_tokens"].as_u64()
        .unwrap_or_else(|| u["prompt_tokens"].as_u64().unwrap_or(0).saturating_sub(cached));
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

async fn blocking(cfg: &Config, ctx: &Context, messages: &[Message], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    chat_blocking(cfg, body(cfg, ctx, messages, false), to_internal, token, sink).await
}

async fn streaming(cfg: &Config, ctx: &Context, messages: &[Message], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    chat_streaming(cfg, body(cfg, ctx, messages, true), usage, token, sink).await
}
#[cfg(test)]
mod tests {

    use crate::api::{call_api, Protocol};
    use crate::api::{Effort, Thinking as CThinking};
    use crate::test_util::{block_on, cfg, ctx as make_ctx, mock, sink, BlockExt};
    use serde_json::json;

    #[test]
    fn body_carries_the_toggle_and_the_effort_word() {
        let mut c = cfg("https://x".into(), true);
        c.protocol = Protocol::DEEPSEEK;
        let cx = make_ctx();
        let b = super::body(&c, &cx, &[], false);
        assert_eq!(b["thinking"], json!({"type": "enabled"}));
        c.effort = Some(Effort::Max);
        let b = super::body(&c, &cx, &[], false);
        assert_eq!(b["reasoning_effort"], "max");
        c.thinking = CThinking::Strip;
        c.effort = None;
        let b = super::body(&c, &cx, &[], false);
        assert_eq!(b["thinking"], json!({"type": "disabled"}));
        assert!(b.get("reasoning_effort").is_none());
    }

    #[test]
    fn streaming_assembles_deltas() {
        let stream = concat!(
            r#"data: {"choices":[{"delta":{"role":"assistant","content":"hello "}}]}"#, "

",
            r#"data: {"choices":[{"delta":{"content":"world"}}]}"#, "

",
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"bash","arguments":"{\"command\":\"echo hi\"}"}}]}}]}"#, "

",
            r#"data: {"usage":{"prompt_cache_hit_tokens":3,"prompt_cache_miss_tokens":4,"completion_tokens":2}}"#, "

",
            r#"data: [DONE]"#, "

");
        let port = mock(stream, 200, true);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), true);
        c.protocol = Protocol::DEEPSEEK;
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let (resp, _, _) = block_on(call_api(&c, &cx, &[], &token, &sink())).unwrap();
        assert_eq!(resp.blocks.len(), 2);
        assert_eq!(resp.blocks[0].text().unwrap(), "hello world");
        assert_eq!(resp.blocks[1].name(), "bash");
        assert_eq!(resp.blocks[1].input(), &json!({"command": "echo hi"}));
        assert_eq!(resp.usage["input_tokens"], 4);
        assert_eq!(resp.usage["cache_read_input_tokens"], 3);
        assert_eq!(resp.usage["output_tokens"], 2);
    }

    #[test]
    fn blocking_roundtrip_and_error() {
        let good = json!({"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_cache_hit_tokens":1,"prompt_cache_miss_tokens":2,"completion_tokens":1}}).to_string().into_boxed_str();
        let port = mock(&good, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::DEEPSEEK;
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let (resp, _, _) = block_on(call_api(&c, &cx, &[], &token, &sink())).unwrap();
        assert_eq!(resp.blocks[0].text().unwrap(), "ok");
        let bad = mock(&json!({"error":{"message":"boom"}}).to_string(), 400, false);
        let mut c = cfg(format!("http://127.0.0.1:{bad}"), false);
        c.protocol = Protocol::DEEPSEEK;
        let err = block_on(call_api(&c, &cx, &[], &token, &sink())).unwrap_err();
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    #[test]
    fn usage_reads_the_disk_cache_ledger() {
        let u = json!({"prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 3, "completion_tokens": 2});
        let (wire, _shown) = super::usage(&u);
        assert_eq!(wire["input_tokens"], 3);
        assert_eq!(wire["cache_read_input_tokens"], 7);
        assert_eq!(wire["output_tokens"], 2);
        let u = json!({"prompt_tokens": 10, "prompt_cache_hit_tokens": 4, "completion_tokens": 1});
        let (wire, _) = super::usage(&u);
        assert_eq!(wire["input_tokens"], 6);
        assert_eq!(wire["cache_read_input_tokens"], 4);
    }

    #[test]
    fn rules_live_with_the_vendor() {
        assert_eq!(Protocol::DEEPSEEK.default_base(), Some("https://api.deepseek.com"));
        assert_eq!(Protocol::DEEPSEEK.default_model(), None);
        let err = Protocol::DEEPSEEK.accepts(crate::api::CacheMode::Active, CThinking::Preserve, None).unwrap_err();
        assert!(err.to_string().contains("deepseek"), "got: {err}");
        let err = Protocol::DEEPSEEK.accepts(crate::api::CacheMode::Auto, CThinking::Strip, Some(Effort::High)).unwrap_err();
        assert!(err.to_string().contains("reasoning"), "got: {err}");
        Protocol::DEEPSEEK.accepts(crate::api::CacheMode::Auto, CThinking::Preserve, None).unwrap();
        Protocol::DEEPSEEK.accepts(crate::api::CacheMode::Auto, CThinking::Preserve, Some(Effort::Low)).unwrap();
        Protocol::DEEPSEEK.accepts(crate::api::CacheMode::Auto, CThinking::Strip, None).unwrap();
    }
}
