// The MiniMax vendor: the Messages wire (content blocks, tool_use/result
// pairs, thinking with a signature, interleaved reasoning, cache_control
// breakpoints — plus the dialect's own policy words). Probed live:
// `thinking: adaptive` and `enabled` both accepted, `disabled` honored
// on M3, budget_tokens accepted beside either. The wire mandates full
// content back every turn (thinking and signature included); we echo
// anyway, the documented contract being the safe side.

use super::{empty_usage, merge_usage, request, response_from_value, show_turn, sse_channel, strip_thinking,
            CacheMode, Thinking};
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::core::context::Context;
use crate::ui::{Msg, Show};
use crate::core::ir::{Block, Message, Response};
use crate::{Error, Result};
use serde_json::{json, Value};

/// The minimax adapter's one entry into the provider port. Strip is a
/// wire rule wearing a policy flag: this wire carries thinking blocks in
/// the context's transcript verbatim, so stripping them is this adapter's
/// job, never the core's. The third tuple slot is the rendered wire body
/// the vendor sent — the session records it verbatim.
pub(super) async fn turn(cfg: &Config, ctx: &Context, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    // The transcript is the context's: it is read from there, never
    // copied alongside it. The strip branch transforms it; the preserve
    // branch hands the very same slice over.
    let body = if cfg.thinking == Thinking::Strip {
        body(cfg, ctx, &strip_thinking(ctx.history()), cfg.streaming)
    } else {
        body(cfg, ctx, ctx.history(), cfg.streaming)
    };
    let key = cfg.api_key.clone();
    if cfg.streaming { streaming(url, body, key, token, sink).await }
    else { blocking(url, body, key, token, sink).await }
}

pub(super) fn body(cfg: &Config, ctx: &Context, messages: &[Message], stream: bool) -> Value {
    use super::chat_system_text;
    let active = cfg.cache == CacheMode::Active;
    // The system side is rendered as multiple blocks: `system_text` first
    // (just `crate::core::prompt::SYSTEM`), then one block per reminder. Every session-
    // stable reminder carries a `cache_control: ephemeral` marker so its
    // body sits in the cached prefix; unstable reminders skip the marker
    // and land as fresh blocks (no prefix pollution).
    let system = if active {
        let mut blocks = vec![json!({
            "type": "text", "text": ctx.system_text,
            "cache_control": {"type": "ephemeral"},
        })];
        for r in &ctx.reminders {
            let mut b = json!({"type": "text", "text": r.text});
            if r.id.is_session_stable() {
                b.as_object_mut().unwrap().insert("cache_control".into(), json!({"type": "ephemeral"}));
            }
            blocks.push(b);
        }
        Value::Array(blocks)
    } else {
        // No active cache: collapse to a single string, identical to the
        // chat-completions family.
        Value::String(chat_system_text(ctx))
    };
    let mut tools = ctx.tools.clone();
    if active { if let Some(last) = tools.last_mut() { last["cache_control"] = json!({"type": "ephemeral"}); } }
    let wire: Vec<Value> = messages.iter().map(Message::to_value).collect();
    let mut body = json!({"model": cfg.model, "max_tokens": cfg.max_tokens,
        "system": system, "tools": tools, "messages": wire});
    body["stream"] = json!(stream);
    // Thinking: `adaptive` turns it on (M3 ships thinking off by default
    // — an agent wants it on), `disabled` is the honest strip, and an
    // effort tier rides the Messages budget beside the toggle.
    body["thinking"] = match (cfg.thinking, cfg.effort) {
        (Thinking::Strip, _) => json!({"type": "disabled"}),
        (Thinking::Preserve, Some(e)) => json!({"type": "adaptive", "budget_tokens": e.budget(cfg.max_tokens)}),
        (Thinking::Preserve, None) => json!({"type": "adaptive"}),
    };
    body
}

async fn blocking(url: String, body: Value, key: String, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    let v: Value = request(token, &key, &url, &body, true, |r| {
        let v: Value = r.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    let resp = response_from_value(&v);
    show_turn(&resp, sink);
    Ok((resp, token.is_cancelled(), body))
}

async fn streaming(url: String, body: Value, key: String, token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool, Value)> {
    let resp = request(token, &key, &url, &body, true, Ok).await?;
    // Blocks arrive indexed; a tool_use's arguments stream as partial
    // JSON that accumulates until the block closes.
    let mut blocks: Vec<Option<Block>> = vec![];
    let mut tool_json: Vec<String> = vec![];
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
        let i = v["index"].as_u64().unwrap_or(0) as usize;
        match v["type"].as_str() {
            Some("message_start") => {
                // message_start carries the real input/cache counts;
                // message_delta later only carries output_tokens. Without
                // this merge, status.turn.context_in() is zero for the turn
                // and the bar's ctx gauge disappears.
                merge_usage(&mut usage, v["message"]["usage"].as_object());
            }
            Some("content_block_start") => {
                if i >= blocks.len() { blocks.resize_with(i + 1, || None); tool_json.resize_with(i + 1, String::new); }
                let cb = &v["content_block"];
                let kind = cb["type"].as_str();
                blocks[i] = match kind {
                    Some("text") => Some(Block::Text(String::new())),
                    Some("thinking") => Some(Block::Thinking { text: String::new(), signature: None }),
                    Some("tool_use") => Some(Block::ToolUse {
                        id: cb["id"].as_str().unwrap_or("").to_string(),
                        name: cb["name"].as_str().unwrap_or("").to_string(),
                        input: Value::Null,
                    }),
                    _ => None,
                };
                if kind == Some("thinking") { sink.show(Msg::Think(String::new())); }
            }
            Some("content_block_delta") => {
                let delta = &v["delta"];
                match blocks.get_mut(i).and_then(|b| b.as_mut()) {
                    Some(Block::Text(t)) => if let Some(s) = delta["text"].as_str() { t.push_str(s); sink.show(Msg::Text(s.into())); }
                    Some(Block::Thinking { text, .. }) => if let Some(s) = delta["thinking"].as_str() { text.push_str(s); sink.show(Msg::Think(s.into())); }
                    Some(Block::ToolUse { .. }) => if let Some(s) = delta["partial_json"].as_str() { tool_json[i].push_str(s); }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                if let Some(Block::ToolUse { input, .. }) = blocks.get_mut(i).and_then(|b| b.as_mut()) {
                    *input = serde_json::from_str(&tool_json[i]).unwrap_or(Value::Null);
                } else if matches!(blocks.get(i).and_then(|b| b.as_ref()), Some(Block::Thinking { .. })) {
                    sink.show(Msg::ThinkEnd);
                }
            }
            Some("message_delta") => {
                merge_usage(&mut usage, v["usage"].as_object());
                sink.show(Msg::Usage(crate::ui::Usage::from_value(&usage)));
                if let Some(n) = v["usage"]["output_tokens"].as_u64() { sink.show(Msg::OutTokens(n)); }
            }
            Some("message_stop") => break,
            Some("error") => return Err(Error::Msg(v["error"].to_string())),
            _ => {}
        }
    }
    sink.show(Msg::Done);
    Ok((Response { blocks: blocks.into_iter().flatten().collect(), usage }, interrupted, body))
}
#[cfg(test)]
mod tests {

    use crate::api::{call_api, Protocol};
    use crate::api::{Effort, Thinking as CThinking};
    use crate::core::ir::{Block, Message, Response, ToolResult};
    use crate::test_util::{block_on, cfg, ctx as make_ctx, mock, mock_seq, request_body, sink, BlockExt, Recorder};
    use crate::ui::Msg;
    use serde_json::json;

    #[test]
    fn body_thinking_toggle_and_verbatim_echo() {
        let mut c = cfg("https://x".into(), true);
        c.max_tokens = 65_536;  // enough headroom for Effort::Max to be visibly different from High
        let cx = make_ctx();
        assert_eq!(super::body(&c, &cx, &[], false)["thinking"], json!({"type": "adaptive"}));
        c.thinking = CThinking::Strip;
        assert_eq!(super::body(&c, &cx, &[], true)["thinking"], json!({"type": "disabled"}));
        c.thinking = CThinking::Preserve;
        c.effort = Some(Effort::High);
        assert_eq!(super::body(&c, &cx, &[], false)["thinking"],
            json!({"type": "adaptive", "budget_tokens": 32768}));
        c.effort = Some(Effort::Max);
        let budget = super::body(&c, &cx, &[], false)["thinking"]["budget_tokens"].as_u64().unwrap();
        assert!(budget > 0 && (budget as u32) < c.max_tokens,
            "max clamps below max_tokens: {budget}");
    }

    #[test]
    fn body_marks_cache_breakpoints_only_when_active() {
        let mut c = cfg("https://x".into(), false);
        c.cache = crate::api::CacheMode::Auto;
        let mut cx = make_ctx();
        cx.tools = vec![json!({"name":"t","description":"d","input_schema":{"type":"object"}})];
        let b = super::body(&c, &cx, &[], false);
        assert!(b["system"].is_string(), "no breakpoints on auto");
        assert!(b["tools"][0].get("cache_control").is_none());
        c.cache = crate::api::CacheMode::Active;
        let b = super::body(&c, &cx, &[], false);
        assert!(b["system"].as_array().unwrap()[0].get("cache_control").is_some());
        assert!(b["tools"][0].get("cache_control").is_some());
    }

    /// AGENTS.md content rides in its own system block, separate from
    /// `system_text` (which holds only `crate::core::prompt::SYSTEM`), and lands in
    /// the cached prefix via the stable reminder's `cache_control`
    /// marker. Two blocks = two cache segments: `SYSTEM` and the AGENTS.md
    /// body can change independently without invalidating each other's
    /// prefix.
    #[test]
    fn body_splits_agents_md_into_its_own_system_block() {
        use crate::test_util::temp_dir;
        let dir = temp_dir("agents_md_minimax");
        std::fs::write(dir.join("AGENTS.md"), "use rustfmt").unwrap();
        let md = crate::core::agents_md::AgentsMdContext::load(&dir);

        let mut c = cfg("https://x".into(), true);
        c.cache = crate::api::CacheMode::Active; // exercise the array-with-cache branch
        let cx = crate::core::context::Context::new(&md, crate::config::DEFAULT_CONTEXT_SIZE);
        let b = super::body(&c, &cx, &[], false);
        let arr = b["system"].as_array().expect("active cache ⇒ array of blocks");
        assert_eq!(arr.len(), 2,
            "two blocks: SYSTEM + AgentsMdClosest reminder — not the old single composed block");

        // block 0: just SYSTEM, with cache marker
        let sys_block = &arr[0];
        assert_eq!(sys_block["text"], crate::core::prompt::SYSTEM,
            "system_text holds the SYSTEM constant alone; AGENTS.md does not leak in here");
        assert!(sys_block.get("cache_control").is_some(),
            "SYSTEM rides in the cached prefix");

        // block 1: AgentsMdClosest reminder — labelled, content-bearing,
        // and also marked (it's session-stable)
        let agents_block = &arr[1];
        let agents_text = agents_block["text"].as_str().unwrap();
        assert!(agents_text.contains("use rustfmt"), "AGENTS.md body in the reminder");
        assert!(agents_text.contains("# Project conventions"),
            "reminder is labelled so the model recognizes where the content came from");
        assert!(agents_text.contains(&dir.display().to_string()),
            "the path lands in the reminder too — the model knows which AGENTS.md this is");
        assert!(agents_block.get("cache_control").is_some(),
            "AgentsMdClosest is session-stable → cache marker");
    }

    /// With no AGENTS.md loaded, `reminders` is empty and the wire is
    /// byte-equivalent to the old single-block behavior: one block,
    /// `system_text`, cache marker.
    #[test]
    fn body_without_agents_md_is_a_single_system_block() {
        use crate::test_util::temp_dir;
        let dir = temp_dir("agents_md_minimax_none");
        // no AGENTS.md written — `load` returns the empty context
        let md = crate::core::agents_md::AgentsMdContext::load(&dir);

        let mut c = cfg("https://x".into(), true);
        c.cache = crate::api::CacheMode::Active;
        let cx = crate::core::context::Context::new(&md, crate::config::DEFAULT_CONTEXT_SIZE);
        let b = super::body(&c, &cx, &[], false);
        let arr = b["system"].as_array().expect("active cache ⇒ array of blocks");
        assert_eq!(arr.len(), 1,
            "no AGENTS.md ⇒ no reminder ⇒ one SYSTEM block — same as the old wire shape");
        assert_eq!(arr[0]["text"], crate::core::prompt::SYSTEM);
        assert!(arr[0].get("cache_control").is_some());
    }

    #[test]
    fn blocking_roundtrip_and_error() {
        let good = json!({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 2}}).to_string().into_boxed_str();
        let port = mock(&good, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::MINIMAX;
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let (resp, _, _) = block_on(call_api(&c, &cx, &token, &sink())).unwrap();
        assert_eq!(resp.blocks.len(), 1);
        let bad = mock(&json!({"error": {"message": "boom"}}).to_string(), 400, false);
        let mut c = cfg(format!("http://127.0.0.1:{bad}"), false);
        c.protocol = Protocol::MINIMAX;
        let err = block_on(call_api(&c, &cx, &token, &sink())).unwrap_err();
        assert!(err.to_string().contains("boom"), "got: {err}");
    }

    #[test]
    fn streaming_assembles_blocks() {
        let stream = concat!(
            r#"data: {"type":"message_start","message":{"content":[],"usage":{"input_tokens":1,"output_tokens":0}}}"#, "

",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "

",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello "}}"#, "

",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}"#, "

",
            r#"data: {"type":"content_block_stop","index":0}"#, "

",
            r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#, "

",
            r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"echo hi\"}"}}"#, "

",
            r#"data: {"type":"content_block_stop","index":1}"#, "

",
            r#"data: {"type":"message_delta","usage":{"output_tokens":2}}"#, "

",
            r#"data: {"type":"message_stop"}"#, "

");
        let port = mock(stream, 200, true);
        let c = cfg(format!("http://127.0.0.1:{port}"), true);
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let (resp, _, _) = block_on(call_api(&c, &cx, &token, &sink())).unwrap();
        assert_eq!(resp.blocks.len(), 2);
        assert_eq!(resp.blocks[0].text().unwrap(), "hello world");
        assert_eq!(resp.blocks[1].name(), "bash");
        assert_eq!(resp.blocks[1].input(), &json!({"command": "echo hi"}));
    }

    #[test]
    fn streaming_merges_message_start_usage_so_ctx_is_not_zero() {
        // minimax opens the turn with input_tokens/cache counts in
        // message_start.usage; message_delta only carries output_tokens.
        // Drop message_start's usage and status.turn.context_in() stays
        // zero for the whole turn, the bar's ctx gauge disappears.
        let stream = concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":1234,"cache_read_input_tokens":800}}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_delta","usage":{"output_tokens":5}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n");
        let port = mock(stream, 200, true);
        let c = cfg(format!("http://127.0.0.1:{port}"), true);
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let (resp, _, _) = block_on(call_api(&c, &cx, &token, &sink())).unwrap();
        // message_start's usage must reach the response — both input and
        // cache_read survive even though message_delta carries neither.
        assert_eq!(resp.usage["input_tokens"], 1234, "message_start input must persist");
        assert_eq!(resp.usage["cache_read_input_tokens"], 800, "message_start cache_read must persist");
        assert_eq!(resp.usage["output_tokens"], 5);
        // And the ctx gauge can read it: context_in() = input + cache_read + cache_write.
        let u = crate::ui::Usage::from_value(&resp.usage);
        assert_eq!(u.context_in(), 2034, "ctx_in must include input + cache_read");
    }

    #[test]
    fn a_stream_with_no_usage_delta_still_ends_with_a_done() {
        // the wire can end a request without a usage delta: the port then
        // sees a `Done` and no `Usage`, and the frontend's accounting has
        // to leave the numbers alone rather than fold the last reading
        // a second time. The api layer must not invent a `Usage` it does
        // not have — the frontend is the one that copes.
        let stream = concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":1234}}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n");
        let port = mock(stream, 200, true);
        let c = cfg(format!("http://127.0.0.1:{port}"), true);
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let rec = Recorder::default();
        let (resp, _, _) = block_on(call_api(&c, &cx, &token, &rec)).unwrap();
        // The tokens the wire did report are still ours to keep — what is
        // missing is the delta, not the reading.
        assert_eq!(resp.usage["input_tokens"], 1234, "the reported tokens are kept: {}", resp.usage);
        let msgs = rec.msgs();
        assert!(msgs.iter().any(|m| matches!(m, Msg::Done)), "the request still ends: {msgs:?}");
        assert!(!msgs.iter().any(|m| matches!(m, Msg::Usage(_))), "no usage was reported: {msgs:?}");
    }

    #[test]
    fn streaming_surfaces_error_events() {
        let stream = concat!(
            r#"data: {"type":"message_start","message":{}}"#, "

",
            r#"data: {"type":"error","error":{"message":"rate limit"}}"#, "

");
        let port = mock(stream, 200, true);
        let c = cfg(format!("http://127.0.0.1:{port}"), true);
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let err = block_on(call_api(&c, &cx, &token, &sink())).unwrap_err();
        assert!(err.to_string().contains("rate limit"), "got: {err}");
    }

    #[test]
    fn wire_paths_and_auth_headers_per_protocol() {
        let token = crate::cancel::CancelToken::new();
        let body = json!({"content": [], "usage": {}}).to_string();
        let (port, seen) = mock_seq(vec![(200, body)]);
        let c = cfg(format!("http://127.0.0.1:{port}"), false);
        let cx = make_ctx();
        let _ = block_on(call_api(&c, &cx, &token, &sink())).unwrap();
        let raw = seen.lock().unwrap()[0].clone();
        assert!(raw.contains("POST /v1/messages"), "minimax hits /v1/messages: {raw}");
        assert!(raw.contains("x-api-key: test-key") && raw.contains("anthropic-version: 2023-06-01"),
            "minimax carries both auth headers: {raw}");
    }

    /// The transcript has one owner — the context — and the wire is
    /// rendered from it. What the model sees is what the context holds:
    /// the same turns, in the same order, with the tool_use and its
    /// tool_result still paired.
    #[test]
    fn the_contexts_transcript_reaches_the_messages_wire() {
        let mut cx = make_ctx();
        cx.push(Message::User("first".into()));
        cx.push(Message::Assistant(vec![Block::ToolUse {
            id: "t1".into(), name: "bash".into(), input: json!({}),
        }]));
        cx.push(Message::ToolResults(vec![ToolResult { id: "t1".into(), content: "ok".into() }]));
        let (port, seen) = mock_seq(vec![(200, r#"{"content":[],"usage":{}}"#.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::MINIMAX;
        let token = crate::cancel::CancelToken::new();
        block_on(call_api(&c, &cx, &token, &sink())).unwrap();
        let v = request_body(&seen.lock().unwrap()[0]);
        let msgs = v["messages"].as_array().expect("the Messages wire sends messages");
        assert_eq!(msgs.len(), 3, "the context's three turns reach the wire: {v}");
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "first");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][0]["type"], "tool_use");
        assert_eq!(msgs[1]["content"][0]["id"], "t1");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][0]["tool_use_id"], "t1",
            "the tool_use and its tool_result stay paired on the wire");
    }

    /// Strip is this wire's own rule, and it applies to the context's
    /// transcript: the thinking block goes, the rest of the turn stays.
    /// A vendor that "just read `ctx.history()`" without the strip would
    /// put the thinking block on the wire.
    #[test]
    fn strip_drops_thinking_from_the_contexts_transcript() {
        let mut cx = make_ctx();
        cx.push(Message::Assistant(vec![
            Block::Thinking { text: "secret".into(), signature: Some("sig".into()) },
            Block::Text("kept".into()),
        ]));
        let (port, seen) = mock_seq(vec![(200, r#"{"content":[],"usage":{}}"#.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::MINIMAX;
        c.thinking = CThinking::Strip;
        let token = crate::cancel::CancelToken::new();
        block_on(call_api(&c, &cx, &token, &sink())).unwrap();
        let v = request_body(&seen.lock().unwrap()[0]);
        let blocks = v["messages"][0]["content"].as_array().expect("the turn is on the wire");
        assert_eq!(blocks.len(), 1, "the thinking block is stripped: {v}");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "kept");
    }

    #[test]
    fn backoff_doubles_then_caps() {
        use crate::api::backoff;
        use crate::api::RETRY_BASE;
        use crate::api::RETRY_MAX;
        assert_eq!(backoff(1), RETRY_BASE);
        assert_eq!(backoff(2), RETRY_BASE * 2);
        assert!(backoff(20) <= RETRY_MAX, "caps at RETRY_MAX");
    }

    // ---- the Messages wire: the events it speaks -------------------------

    /// A Messages-wire stream served by `events`, and what it produced.
    fn messages_stream(events: &str) -> (Response, bool, Vec<Msg>) {
        let port = mock(events, 200, true);
        let c = cfg(format!("http://127.0.0.1:{port}"), true);
        let token = crate::cancel::CancelToken::new();
        let cx = make_ctx();
        let rec = Recorder::default();
        let (resp, interrupted, _) = block_on(call_api(&c, &cx, &token, &rec)).unwrap();
        (resp, interrupted, rec.msgs())
    }

    #[test]
    fn a_two_hundred_answer_with_an_error_is_reported_in_the_wires_own_words() {
        // The status says the request was fine; the body says otherwise.
        // The wire's own words are what comes back.
        let (port, reqs) = mock_seq(vec![
            (200, r#"{"error":{"message":"the model is gone"}}"#.into()),
        ]);
        let c = cfg(format!("http://127.0.0.1:{port}"), false);
        let cx = make_ctx();
        let err = block_on(call_api(&c, &cx, &crate::cancel::CancelToken::new(), &sink())).unwrap_err();
        assert!(err.to_string().contains("the model is gone"), "the wire's own words: {err}");
        assert_eq!(reqs.lock().unwrap().len(), 1, "a wire error is the request's own: no resend");
    }

    #[test]
    fn a_stream_that_ends_with_the_done_sentinel_stops_there() {
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#, "\n\n",
            "data: [DONE]\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"never read"}}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        assert_eq!(resp.blocks[0].text().unwrap(), "hi", "the sentinel ended the stream");
    }

    #[test]
    fn a_stream_line_that_is_not_json_is_skipped() {
        let (resp, _, _) = messages_stream(concat!(
            "data: this is not json\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        assert_eq!(resp.blocks[0].text().unwrap(), "hi", "the junk line changed nothing");
    }

    #[test]
    fn a_block_that_arrives_out_of_order_lands_in_its_own_slot() {
        // The wire numbers its blocks; a block that shows up at index 2
        // without its predecessors is still the third block.
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"third"}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1, "the empty slots before it are not blocks");
        assert_eq!(resp.blocks[0].text().unwrap(), "third");
    }

    #[test]
    fn a_thinking_block_opens_a_think_and_closes_it_at_the_stop() {
        let (resp, _, msgs) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"pondering"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        match &resp.blocks[0] {
            Block::Thinking { text, .. } => assert_eq!(text, "pondering"),
            other => panic!("expected Thinking, got {other:?}"),
        }
        let shape: Vec<&str> = msgs.iter().map(|m| match m {
            Msg::Think(_) => "think", Msg::ThinkEnd => "think_end", Msg::Done => "done", _ => "other",
        }).collect();
        assert_eq!(shape, vec!["think", "think", "think_end", "done"],
            "the block opens a think and the stop closes it: {msgs:?}");
    }

    #[test]
    fn a_block_of_an_unknown_kind_is_ignored() {
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"xx"}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert!(resp.blocks.is_empty(), "an unknown block kind is not a block");
    }

    #[test]
    fn a_delta_for_a_block_that_never_started_is_ignored() {
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"nowhere"}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert!(resp.blocks.is_empty(), "a delta with no block is dropped");
    }

    #[test]
    fn a_text_delta_with_no_text_changes_nothing() {
        let (resp, _, msgs) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        assert_eq!(resp.blocks[0].text().unwrap(), "", "no text arrived: the block stays empty");
        assert!(!msgs.iter().any(|m| matches!(m, Msg::Text(_))), "{msgs:?}");
    }

    #[test]
    fn a_thinking_delta_with_no_thinking_changes_nothing() {
        let (resp, _, msgs) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        match &resp.blocks[0] {
            Block::Thinking { text, .. } => assert_eq!(text, "", "no thinking arrived"),
            other => panic!("expected Thinking, got {other:?}"),
        }
        assert_eq!(msgs.iter().filter(|m| matches!(m, Msg::Think(_))).count(), 1,
            "only the block start opened a think: {msgs:?}");
    }

    #[test]
    fn a_tool_delta_with_no_partial_json_changes_nothing() {
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"bash","input":{}}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta"}}"#, "\n\n",
            r#"data: {"type":"content_block_stop","index":0}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1);
        assert!(resp.blocks[0].input().is_null(), "no arguments arrived: the tool has no input");
    }

    #[test]
    fn a_message_delta_with_no_output_tokens_reports_the_usage_it_has() {
        let (resp, _, msgs) = messages_stream(concat!(
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":7}}}"#, "\n\n",
            r#"data: {"type":"message_delta","usage":{"input_tokens":7}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.usage["input_tokens"], 7);
        assert!(!msgs.iter().any(|m| matches!(m, Msg::OutTokens(_))),
            "no output count was reported: {msgs:?}");
    }

    #[test]
    fn an_event_of_an_unknown_kind_is_ignored() {
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"ping"}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert!(resp.blocks.is_empty(), "an unknown event is not a block");
    }

    /// A body that is not JSON at all is refused with the parser's own
    /// words: there is no answer to read, and the turn cannot go on.
    #[test]
    fn an_answer_that_is_not_json_says_so() {
        let (port, _) = mock_seq(vec![(200, "this is not json at all".into())]);
        let c = cfg(format!("http://127.0.0.1:{port}"), false);
        let cx = make_ctx();
        let err = block_on(call_api(&c, &cx, &crate::cancel::CancelToken::new(), &sink())).unwrap_err();
        assert!(matches!(err, crate::Error::Io(_)), "the body was not json: {err:?}");
        assert!(err.to_string().contains("JSON"), "the parser says why: {err}");
    }

    /// A block that starts twice in the same slot is one block: the wire
    /// numbers its blocks, and a slot holds one of them.
    #[test]
    fn a_block_that_starts_twice_in_the_same_slot_keeps_one_block() {
        let (resp, _, _) = messages_stream(concat!(
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#, "\n\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#, "\n\n",
            r#"data: {"type":"message_stop"}"#, "\n\n"));
        assert_eq!(resp.blocks.len(), 1, "the slot holds one block: {resp:?}");
        assert_eq!(resp.blocks[0].text().unwrap(), "hi");
    }
}
