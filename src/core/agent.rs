//! The agent loop: the coroutine that composes everything else. Each
//! run is [`agent_turn`] — cancellation semantics wrapped around
//! [`agent_loop`], which drives [`run_turn`]: one model call, the tool
//! batch it asked for, and around again, until the model answers in
//! plain text or a bound (`--max-turns`, Ctrl-C, an error) stops it.
//!
//! The loop owns no resources: the config, history, context, cancel
//! token, MCP hub, and output sink are all handed in by the caller
//! (`crate::run_with_hub` or a frontend), which is exactly why every
//! piece can be tested with mocks and why new frontends cost nothing.

use crate::api::call_api as api_call_api;
use crate::cancel::CancelToken;
use crate::config::Config;
use crate::core::context::{self, Context};
use crate::core::ir::{Block, Message, ToolResult};
use crate::core::turn::{Turn, TurnOutcome, TurnState};
use crate::display::{Msg, Sev, Show};
use crate::mcp::Hub;
use crate::tool_runtime::run_tool;
use crate::tools::print_tool_call;
use crate::Error;

const MAX_TOOL_OUTPUT: usize = 50_000;

/// Drive one agent run as an interruptible coroutine.
///
/// The first Ctrl-C cancels the token: the agent coroutine notices at its
/// next suspension point (a streamed line, a tool poll, a turn boundary),
/// tidies the history so every tool call stays paired, and unwinds — the
/// REPL prompt comes back with everything the agent already carried still in
/// place. A second Ctrl-C while it is unwinding exits immediately, for when
/// even graceful is too slow.
///
/// `turn` is the per-run state machine: the caller hands in a fresh
/// `Turn` (in `Queued`) and the wrapper drives it through the table via
/// [`agent_loop`]. The caller can inspect `turn` after the wrapper
/// returns to read the terminal state — the wrapper owns the cancel
/// select (`tokio::select!` against Ctrl-C) and the double-signal
/// semantics, not to hide the turn from anyone.
pub(crate) async fn agent_turn(cfg: &Config, ctx: &mut Context, history: &mut Vec<Message>, turn: &mut Turn, token: &CancelToken, hub: &Hub, sink: &dyn Show) -> crate::Result<()> {
    let agent = agent_loop(cfg, ctx, history, turn, token, hub, sink);
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

/// The agent itself, as a coroutine. Every await is a suspension point and a
/// cancellation point: between turns, on every streamed line, while each tool
/// runs. On cancellation it returns `Err(Interrupted)` after leaving the
/// history valid — interrupted tools get results, unfinished tool calls are
/// dropped from the turn, and the REPL prompt simply returns.
///
/// `turn` is the per-run state machine the wrapper owns; this function
/// drives it through the table — `Queued → InProgress` at the entry,
/// the terminal transition on the way out. The mechanics of "do API
/// calls and run tools" live in [`run_turn`], kept separate so the
/// mapping between [`Result`] and [`TurnState`] has exactly one home
/// (caller translates to [`TurnOutcome`], [`Turn::finish`] maps that).
///
/// `hub` is the MCP hub — its `definitions()` is the source of the schemas
/// the model sees, and its `call()` is what runs when the model asks for
/// a tool whose name starts with `mcp__`. Re-fetching definitions every
/// turn is what lets `/mcp enable|disable|reconnect` take effect mid-session.
///
/// `ctx` is the model-side context — system text + tool list, owned by
/// `crate::core::context`. The agent loop asks `ctx.refresh(hub)` each turn so
/// the next request sees whatever MCP servers reported.
async fn agent_loop(cfg: &Config, ctx: &mut Context, history: &mut Vec<Message>, turn: &mut Turn, token: &CancelToken, hub: &Hub, sink: &dyn Show) -> crate::Result<()> {
    // The only legal first move. An Illegal here would mean the caller
    // handed us a non-fresh turn — a programming bug — and we surface it
    // as a regular error rather than a panic so a misconfigured caller
    // doesn't kill the agent mid-sentence.
    turn.transition(TurnState::InProgress, None)
        .map_err(|e| Error::Msg(e.to_string()))?;

    let result = run_turn(cfg, ctx, history, token, hub, sink).await;
    // The Result → TurnOutcome translation lives here because `Error`
    // is a crate-root type and turn.rs deliberately does not depend on it.
    // From TurnOutcome onward, the mapping is the turn's own concern.
    let outcome = match &result {
        Ok(()) => TurnOutcome::Completed,
        Err(Error::Interrupted) => TurnOutcome::Interrupted,
        Err(e) => TurnOutcome::Failed(e.to_string()),
    };
    turn.finish(outcome);
    result
}

/// Drive one turn's worth of API calls and tool runs. Pure mechanics — the
/// [`Turn`] state machine is owned by the caller. Returns whatever
/// happened; the caller maps it onto a terminal state via [`Turn::finish`].
async fn run_turn(cfg: &Config, ctx: &mut Context, history: &mut Vec<Message>, token: &CancelToken, hub: &Hub, sink: &dyn Show) -> crate::Result<()> {
    for step in 0..cfg.max_turns {
        if token.is_cancelled() { return Err(Error::Interrupted); }
        ctx.refresh(hub).await;
        if context::fit_context(history, cfg.context_size) {
            sink.show(Msg::Note { sev: Sev::Warn, text: " trimmed history to fit --context-size ".into() });
        }
        let (resp, interrupted, _body) = api_call_api(cfg, ctx, history, token, sink).await?;
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
        // Echo the full assistant turn back — including the final
        // text-only turn, so follow-up questions keep context.
        if !content.is_empty() {
            history.push(Message::Assistant(content));
        }
        if calls.is_empty() { return Ok(()); }

        // Run the tools. Cancellation stops the batch: the interrupted
        // call reports as much as it got, the rest are skipped, and
        // every tool_use still leaves with its tool_result.
        let mut results = Vec::with_capacity(calls.len());
        let mut stopped = false;
        for c in &calls {
            let (id, name, input) = c.tool_use().expect("filtered to tool calls");
            let out = if stopped {
                "skipped: this run was interrupted before the tool ran".into()
            } else {
                run_tool(name, input, token, hub).await
            };
            let out = truncate(&out);
            print_tool_call(name, input, &out, sink);
            results.push(ToolResult { id: id.to_string(), content: out });
            if token.is_cancelled() { stopped = true; }
        }
        history.push(Message::ToolResults(results));
        if stopped { return Err(Error::Interrupted); }

        if step + 1 == cfg.max_turns {
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

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel::CancelToken;
    use crate::mcp::Hub;
    use crate::test_util::{block_on, cfg, mock_seq, mock_stall, sink};
    use serde_json::json;
    use serde_json::Value;
    use std::sync::Arc;

    /// A typed history from JSON literals — the tests still speak the wire
    /// shape, and `ir` owns the only translation of it.
    fn hv(values: Vec<Value>) -> Vec<Message> {
        values.iter().map(Message::from_value).collect()
    }
    /// History back as JSON, for the assertions the tests have always made.
    fn hist(history: &[Message]) -> Value {
        crate::core::ir::history_value(history)
    }
    /// A response's blocks as JSON, read like a `content` array.
    fn blocks(b: &[Block]) -> Value {
        Value::Array(b.iter().map(Block::to_value).collect())
    }

    // ---- truncate: the bounded-output safety net --------------------------

    #[test]
    fn truncate_passes_through_short_strings() {
        assert_eq!(truncate("hello"), "hello");
        assert_eq!(truncate(""), "");
        assert_eq!(truncate(&"x".repeat(MAX_TOOL_OUTPUT)), "x".repeat(MAX_TOOL_OUTPUT));
    }

    #[test]
    fn truncate_strips_after_cap_with_marker() {
        let big = "a".repeat(MAX_TOOL_OUTPUT + 100);
        let t = truncate(&big);
        assert!(t.contains(&format!("…[truncated, {} bytes total]", big.len())));
        assert!(t.len() <= MAX_TOOL_OUTPUT + 64, "the marker is bounded: {t}");
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

    // ---- agent loop ----

    #[test]
    fn agent_loop_runs_tool_then_stops_on_plain_text() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"echo hi"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let done = r#"{"content":[{"type":"text","text":"done"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, reqs) = mock_seq(vec![(200, tool.into()), (200, done.into())]);
        let mut history = hv(vec![json!({"role": "user", "content": "run echo"})]);
        let mut turn = Turn::new(0);
        block_on(async {
            // Hub::empty() spawns an actor task — the test needs a runtime
            // for the spawn to land, even though the hub itself has no work
            // to do in this test.
            let hub = Hub::empty();
            let mut cx = crate::test_util::ctx();
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut cx, &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await.unwrap();
            hub.shutdown().await;
        });
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
        // the turn reached Completed — the natural exit on a final-text response
        assert_eq!(turn.state, TurnState::Completed);
        assert!(turn.started_at.is_some() && turn.ended_at.is_some());
    }

    #[test]
    fn agent_loop_stops_at_max_turns() {
        let tool = r#"{"content":[{"type":"tool_use","id":"t1","name":"bash","input":{"command":"true"}}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, _) = mock_seq(vec![(200, tool.into())]);
        let mut c = cfg(format!("http://127.0.0.1:{port}"), false);
        c.max_turns = 1;
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        block_on(async {
            let hub = Hub::empty();
            let mut cx = crate::test_util::ctx();
            agent_loop(&c, &mut cx, &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await.unwrap(); // must return instead of spinning
            hub.shutdown().await;
        });
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
        // hitting max_turns is a normal exit, not a failure — Completed, not Failed
        assert_eq!(turn.state, TurnState::Completed);
    }

    // ---- coroutine cancellation ----

    // bash kill/cancel tests live in tool_runtime::tests (the natural
    // home for them) with tighter timing ceilings.

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
        let mut turn = Turn::new(0);
        let err = block_on(async {
            let hub = Hub::empty();
            let mut cx = crate::test_util::ctx();
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            let res = agent_loop(&cfg(format!("http://127.0.0.1:{port}"), true), &mut cx, &mut history, &mut turn, &token, &hub, &sink()).await;
            hub.shutdown().await;
            res
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
        // Ctrl-C mid-stream lands the turn on `Interrupted`, not `Failed` —
        // interruption is a stop, not an error, and the table keeps the two
        // separate so the banner can read "interrupted" without "error".
        assert_eq!(turn.state, TurnState::Interrupted);
        assert!(turn.error.is_none(), "interrupted is not a failure");
    }

    #[test]
    fn agent_loop_interrupted_mid_tool_finishes_pairing() {
        let cmd = if cfg!(windows) { "ping -n 30 127.0.0.1 > nul" } else { "sleep 30" };
        let tool = format!(
            r#"{{"content":[{{"type":"tool_use","id":"t1","name":"bash","input":{{"command":"{cmd}"}}}}],"usage":{{"input_tokens":1,"output_tokens":1}}}}"#);
        let (port, _) = mock_seq(vec![(200, tool)]);
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        let err = block_on(async {
            let hub = Hub::empty();
            let mut cx = crate::test_util::ctx();
            let token = Arc::new(CancelToken::new());
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                t.cancel();
            });
            let res = agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut cx, &mut history, &mut turn, &token, &hub, &sink()).await;
            hub.shutdown().await;
            res
        }).unwrap_err();
        assert!(matches!(err, Error::Interrupted), "got: {err}");
        assert_eq!(history.len(), 3); // user + assistant(tool_use) + user(tool_result)
        let result = hist(&history)[2]["content"][0]["content"].as_str().unwrap().to_string();
        assert!(result.contains("[interrupted by user]"), "got: {result}");
        assert_eq!(hist(&history)[2]["content"][0]["tool_use_id"], json!("t1"));
        // mid-tool cancel also lands on `Interrupted` — same state, same reason.
        assert_eq!(turn.state, TurnState::Interrupted);
    }

    /// A 500 from the wire is *not* `Failed`: the transport retries it
    /// (`is_transient`), so the user only sees the eventual result. We
    /// exhaust `RETRY_MAX` and serve another 500 on every attempt — once
    /// retries are gone, the turn ends in `Failed` with the API status
    /// text in `error`.
    #[test]
    fn agent_loop_records_failed_when_api_rejects() {
        // 5xx is transient; serve enough 500s to exhaust retries, and the
        // final error surfaces as `Failed`. Each retry is a fresh
        // connection on the mock listener, so N retries need N responses.
        let five_hundred = r#"{"type":"error","error":{"type":"api_error","message":"overloaded"}}"#;
        let mut responses = Vec::new();
        for _ in 0..=crate::api::RETRY_ATTEMPTS {
            responses.push((500, five_hundred.into()));
        }
        let (port, _) = mock_seq(responses);
        let mut history = hv(vec![json!({"role": "user", "content": "go"})]);
        let mut turn = Turn::new(0);
        let err = block_on(async {
            let hub = Hub::empty();
            let mut cx = crate::test_util::ctx();
            agent_loop(&cfg(format!("http://127.0.0.1:{port}"), false), &mut cx, &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await
        }).unwrap_err();
        // the error reaches the caller — the same one we record on the turn
        assert!(matches!(err, Error::Api(500, _)), "got: {err}");
        assert_eq!(turn.state, TurnState::Failed, "5xx after retries is Failed, not Interrupted");
        assert!(turn.error.as_deref().unwrap_or("").contains("500"), "error carries the api status: got {:?}", turn.error);
        assert!(turn.started_at.is_some() && turn.ended_at.is_some(), "timestamps stamp both ends");
    }

    // bash+grandchild test lives in tool_runtime::tests; see cmd/conn
    // there for a deeper timeout note.

    /// The wrapper owns the cancel select; the loop-level tests above
    /// exercise the mechanics without it. This one drives the wrapper
    /// end-to-end and asserts on the same Turn the caller owns — so a
    /// future change that forgets to plumb `&mut turn` through agent_turn,
    /// or hands a non-fresh turn to agent_loop, fails here.
    #[test]
    fn agent_turn_drives_a_fresh_turn_through_to_completed() {
        let done = r#"{"content":[{"type":"text","text":"all done"}],"usage":{"input_tokens":1,"output_tokens":1}}"#;
        let (port, _) = mock_seq(vec![(200, done.into())]);
        let mut turn = Turn::new(7);
        // a wrapper-level run starts in Queued — that's the caller's job
        // to construct, and the wrapper must not silently rewind it.
        assert_eq!(turn.state, TurnState::Queued);
        let mut history = hv(vec![json!({"role": "user", "content": "say hi"})]);
        block_on(async {
            let hub = Hub::empty();
            let mut cx = crate::test_util::ctx();
            agent_turn(&cfg(format!("http://127.0.0.1:{port}"), false), &mut cx, &mut history, &mut turn, &CancelToken::new(), &hub, &sink()).await.unwrap();
            hub.shutdown().await;
        });
        // the wrapper drove Queued → InProgress → Completed on the turn
        // the caller handed in; the id survives, the timestamps stamp
        // both ends, and the loop's mechanics ran at least once.
        assert_eq!(turn.id, 7, "wrapper must not rewind turn id");
        assert_eq!(turn.state, TurnState::Completed);
        assert!(turn.started_at.is_some() && turn.ended_at.is_some());
        assert_eq!(hist(&history).as_array().unwrap().last().unwrap()["role"], json!("assistant"));
    }
}
