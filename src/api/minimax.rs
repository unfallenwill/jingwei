// The MiniMax vendor: the Messages wire (content blocks, tool_use/result
// pairs, thinking with a signature, interleaved reasoning, cache_control
// breakpoints — plus the dialect's own policy words). Probed live before
// this was written: `thinking: adaptive` and the Anthropic `enabled`
// spelling both accepted, `disabled` honored on M3, budget_tokens accepted
// beside either, and the docs' echo mandate — full content back every
// turn, thinking and signature included — is enforced leniently today;
// the adapter echoes anyway, the documented contract being the safe side.

use super::{empty_usage, merge_usage, request, response_from_value, show_turn, sse_channel, strip_thinking,
            CacheMode, Thinking};
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::display::{Msg, Show};
use crate::ir::{Block, Message, Response};
use crate::{Error, Result};
use serde_json::{json, Value};

/// The minimax adapter's one entry into the provider port. Strip is a
/// wire rule wearing a policy flag: this wire carries thinking blocks in
/// its history verbatim, so stripping them is this adapter's job, never
/// the core's.
pub(super) async fn turn(cfg: &Config, history: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let messages: Vec<Message> = if cfg.thinking == Thinking::Strip { strip_thinking(history) } else { history.to_vec() };
    if cfg.streaming {
        streaming(cfg, &messages, schemas, token, sink).await
    } else {
        blocking(cfg, &messages, schemas, token, sink).await
    }
}

pub(super) fn body(cfg: &Config, messages: &[Message], schemas: &[Value], stream: bool) -> Value {
    let active = cfg.cache == CacheMode::Active;
    let system = if active {
        json!([{"type": "text", "text": crate::SYSTEM, "cache_control": {"type": "ephemeral"}}])
    } else {
        json!(crate::SYSTEM)
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

async fn blocking(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = body(cfg, messages, schemas, false);
    let key = cfg.api_key.clone();
    let v: Value = request(token, &key, &url, &body, true, |r| {
        let v: Value = r.into_json()?;
        if let Some(err) = v.get("error") { return Err(Error::Msg(err.to_string())); }
        Ok(v)
    }).await?;
    let resp = response_from_value(&v);
    show_turn(&resp, sink);
    Ok((resp, token.is_cancelled()))
}

async fn streaming(cfg: &Config, messages: &[Message], schemas: &[Value], token: &CancelToken, sink: &dyn Show) -> Result<(Response, bool)> {
    let url = format!("{}/v1/messages", cfg.base_url.trim_end_matches('/'));
    let body = body(cfg, messages, schemas, true);
    let key = cfg.api_key.clone();
    let resp = request(token, &key, &url, &body, true, Ok).await?;
    // Blocks arrive one at a time, indexed; a tool_use's arguments stream as
    // partial JSON, so they accumulate in `tool_json` beside the typed block
    // until the block closes and the assembled JSON parses into `input`.
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
        match v["type"].as_str() {
            Some("message_start") => {
                // The size of v["message"]["content"] here is unreliable —
                // the wire reports blocks progressively via content_block_start,
                // and message_start's `content` may be empty until the first
                // block arrives. Let content_block_start grow `blocks`.
                // What is reliable here: the *usage* ledger — minimax opens
                // the turn with the real input/cache counts already filled in,
                // and a later `message_delta` only carries output_tokens. If
                // we don't merge this now, status.turn.context_in() stays at
                // zero for the whole turn, and the bar's ctx gauge disappears.
                // The display port is wired so any later merge (turn-end
                // `message_delta`) replaces — so merging twice with the same
                // fields is a no-op rather than a double-count.
                merge_usage(&mut usage, v["message"]["usage"].as_object());
            }
            Some("content_block_start") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                if i >= blocks.len() { blocks.resize_with(i + 1, || None); tool_json.resize_with(i + 1, String::new); }
                match v["content_block"]["type"].as_str() {
                    Some("text") => blocks[i] = Some(Block::Text(String::new())),
                    Some("thinking") => blocks[i] = Some(Block::Thinking { text: String::new(), signature: None }),
                    Some("tool_use") => blocks[i] = Some(Block::ToolUse {
                        id: v["content_block"]["id"].as_str().unwrap_or("").to_string(),
                        name: v["content_block"]["name"].as_str().unwrap_or("").to_string(),
                        input: Value::Null,
                    }),
                    _ => {}
                }
                if matches!(v["content_block"]["type"].as_str(), Some("thinking")) {
                    sink.show(Msg::Think(String::new()));
                }
            }
            Some("content_block_delta") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                let delta = &v["delta"];
                match blocks.get_mut(i).and_then(|b| b.as_mut()) {
                    Some(Block::Text(t)) => {
                        if let Some(s) = delta["text"].as_str() {
                            t.push_str(s);
                            sink.show(Msg::Text(s.into()));
                        }
                    }
                    Some(Block::Thinking { text, .. }) => {
                        if let Some(s) = delta["thinking"].as_str() {
                            text.push_str(s);
                            sink.show(Msg::Think(s.into()));
                        }
                    }
                    Some(Block::ToolUse { .. }) => {
                        if let Some(s) = delta["partial_json"].as_str() {
                            tool_json[i].push_str(s);
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let i = v["index"].as_u64().unwrap_or(0) as usize;
                match blocks.get_mut(i).and_then(|b| b.as_mut()) {
                    Some(Block::Thinking { .. }) => sink.show(Msg::ThinkEnd),
                    Some(Block::ToolUse { input, .. }) => {
                        *input = serde_json::from_str(&tool_json[i]).unwrap_or(Value::Null);
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                merge_usage(&mut usage, v["usage"].as_object());
                sink.show(Msg::Usage(crate::display::Usage::from_value(&usage)));
                if let Some(n) = v["usage"]["output_tokens"].as_u64() {
                    sink.show(Msg::OutTokens(n));
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
#[cfg(test)]
mod tests {

    use crate::api::{call_api, Protocol};
    use crate::api::{Effort, Thinking as CThinking};
    use crate::test_util::{block_on, cfg, mock, mock_seq, sink, BlockExt};
    use serde_json::json;

    #[test]
    fn body_thinking_toggle_and_verbatim_echo() {
        let mut c = cfg("https://x".into(), true);
        c.max_tokens = 65_536;  // enough headroom for Effort::Max to be visibly different from High
        assert_eq!(super::body(&c, &[], &[], false)["thinking"], json!({"type": "adaptive"}));
        c.thinking = CThinking::Strip;
        assert_eq!(super::body(&c, &[], &[], true)["thinking"], json!({"type": "disabled"}));
        c.thinking = CThinking::Preserve;
        c.effort = Some(Effort::High);
        assert_eq!(super::body(&c, &[], &[], false)["thinking"],
            json!({"type": "adaptive", "budget_tokens": 32768}));
        c.effort = Some(Effort::Max);
        let budget = super::body(&c, &[], &[], false)["thinking"]["budget_tokens"].as_u64().unwrap();
        assert!(budget > 0 && (budget as u32) < c.max_tokens,
            "max clamps below max_tokens: {budget}");
    }

    #[test]
    fn body_marks_cache_breakpoints_only_when_active() {
        let mut c = cfg("https://x".into(), false);
        c.cache = crate::api::CacheMode::Auto;
        let b = super::body(&c, &[], &[json!({"name":"t","description":"d","input_schema":{"type":"object"}})], false);
        assert!(b["system"].is_string(), "no breakpoints on auto");
        assert!(b["tools"][0].get("cache_control").is_none());
        c.cache = crate::api::CacheMode::Active;
        let b = super::body(&c, &[], &[json!({"name":"t","description":"d","input_schema":{"type":"object"}})], false);
        assert!(b["system"].as_array().unwrap()[0].get("cache_control").is_some());
        assert!(b["tools"][0].get("cache_control").is_some());
    }

    #[test]
    fn blocking_roundtrip_and_error() {
        let good = json!({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 2}}).to_string().into_boxed_str();
        let port = mock(&good, 200, false);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.protocol = Protocol::MINIMAX;
        let token = crate::cancel::CancelToken::new();
        let (resp, _) = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap();
        assert_eq!(resp.blocks.len(), 1);
        let bad = mock(&json!({"error": {"message": "boom"}}).to_string(), 400, false);
        let mut c = cfg(format!("http://127.0.0.1:{bad}"), false);
        c.protocol = Protocol::MINIMAX;
        let err = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap_err();
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
        let (resp, _) = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap();
        eprintln!("got blocks: {:?}
usage: {:?}", resp.blocks, resp.usage);
        assert_eq!(resp.blocks.len(), 2);
        assert_eq!(resp.blocks[0].text().unwrap(), "hello world");
        assert_eq!(resp.blocks[1].name(), "bash");
        assert_eq!(resp.blocks[1].input(), &json!({"command": "echo hi"}));
    }

    #[test]
    fn streaming_merges_message_start_usage_so_ctx_is_not_zero() {
        // minimax opens the turn with input_tokens/cache counts already in
        // message_start.usage, and message_delta only carries output_tokens.
        // If the adapter drops message_start's usage, status.turn.context_in()
        // is zero for the whole turn and the bar's ctx gauge disappears.
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
        let (resp, _) = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap();
        // message_start's usage must reach the response — both input and
        // cache_read survive even though message_delta carries neither.
        assert_eq!(resp.usage["input_tokens"], 1234, "message_start input must persist");
        assert_eq!(resp.usage["cache_read_input_tokens"], 800, "message_start cache_read must persist");
        assert_eq!(resp.usage["output_tokens"], 5);
        // And the ctx gauge can read it: context_in() = input + cache_read + cache_write.
        let u = crate::display::Usage::from_value(&resp.usage);
        assert_eq!(u.context_in(), 2034, "ctx_in must include input + cache_read");
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
        let err = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap_err();
        assert!(err.to_string().contains("rate limit"), "got: {err}");
    }

    #[test]
    fn wire_paths_and_auth_headers_per_protocol() {
        let token = crate::cancel::CancelToken::new();
        let body = json!({"content": [], "usage": {}}).to_string();
        let (port, seen) = mock_seq(vec![(200, body)]);
        let c = cfg(format!("http://127.0.0.1:{port}"), false);
        let _ = block_on(call_api(&c, &[], &[], &token, &sink())).unwrap();
        let raw = seen.lock().unwrap()[0].clone();
        assert!(raw.contains("POST /v1/messages"), "minimax hits /v1/messages: {raw}");
        assert!(raw.contains("x-api-key: test-key") && raw.contains("anthropic-version: 2023-06-01"),
            "minimax carries both auth headers: {raw}");
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
}
