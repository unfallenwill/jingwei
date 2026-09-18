// The one thing every test module needs: a fresh, empty temp dir scoped to
// this process. The pid keeps two jingwei test runs from colliding; the name
// keeps one run's tests from colliding with each other. `pub(crate)` because
// tests live in the modules they exercise, and those modules live under the
// crate root.

use std::fs;
use std::path::PathBuf;

/// Serializes the env-mutating tests so NO_COLOR set
/// by one test cannot race a sibling reading them. Test-only.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take [`ENV_LOCK`], ignoring poisoning so one failed test does not cascade
/// into "poisoned" panics for every other env-touching test.
#[cfg(test)]
pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
pub(crate) fn temp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("jingwei_test_{}_{}", std::process::id(), name));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}

/// Drive a coroutine to completion on its own little scheduler. Test modules
/// that need to drive an async fn synchronously — tool runtime, cancel
/// races, agent turn stubs — use this so they do not have to depend on the
/// whole tokio runtime the binary already runs.
#[cfg(test)]
pub(crate) fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all().build().unwrap().block_on(f)
}
// Shared test helpers that used to live in main.rs's mod tests, where
// every test could `use super::*` and see them. Once the tests are split
// across modules — main.rs still runs the agent-loop and transport tests,
// the api/* modules each run their own vendor tests — those helpers live
// here so any module's `mod tests` can import them. cfg(test) gates the
// whole thing: the binary never sees it.

use crate::api::{CacheMode, Protocol, Thinking};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A sink that drops everything — the tests assert on state and on the
/// mock wire, never on what a frontend shows. The real frontends
/// (`plain::PlainSink`, the TUI's `ChannelSink`) are exercised through
/// the binary instead.
#[cfg(test)]
pub(crate) struct NullSink;
#[cfg(test)]
impl crate::display::Show for NullSink {
    fn show(&self, _m: crate::display::Msg) {}
}
#[cfg(test)]
pub(crate) fn sink() -> NullSink {
    NullSink
}

/// The agent's resolved config used by the tests: `test-key` is the api_key,
/// `model` is a placeholder, and `minimax` is the protocol because every
/// history/value assertion the tests make has Messages-wire semantics — the
/// IR is the Messages wire's dialect, so minimax is what round-trips without
/// translation.
#[cfg(test)]
pub(crate) fn cfg(base: String, streaming: bool) -> crate::config::Config {
    crate::config::Config {
        api_key: "test-key".into(), base_url: base, model: "test-model".into(),
        protocol: Protocol::MINIMAX, cache: CacheMode::Auto, thinking: Thinking::Preserve,
        effort: None,
        max_tokens: 1024, context_size: crate::config::DEFAULT_CONTEXT_SIZE,
        max_turns: crate::config::DEFAULT_MAX_TURNS, streaming,
    }
}

/// A bare `Context` for tests that don't care about AGENTS.md or the
/// tool list. Vendors take `&Context`; tests that exercise `body()` or
/// `call_api()` need a handle, and wiring one by hand is noise. The empty
/// AGENTS.md context gives us a system_text equal to `crate::SYSTEM`,
/// which is what nearly every test wants anyway.
#[cfg(test)]
pub(crate) fn ctx() -> crate::context::Context {
    let md = crate::agents_md::AgentsMdContext::empty(&std::path::PathBuf::from("."));
    crate::context::Context::new(&md)
}

/// Serve one HTTP response on a fresh port: `response` as JSON (status 200)
/// or SSE (status 200 + `Content-Type: text/event-stream`) when `sse` is set.
#[cfg(test)]
pub(crate) fn mock(response: &str, status: u16, sse: bool) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    // The mock serves the response body from the spawned thread; the
    // `&str` would otherwise need to be 'static, but the body is read
    // before the thread exits — so a local string works as long as we
    // own it into the closure.
    let response = response.to_owned();
    std::thread::spawn(move || {
        let Ok((mut s, _)) = listener.accept() else { return };
        let mut buf = [0u8; 8192];
        let _ = s.read(&mut buf);
        let (ctype, body) = if sse {
            ("text/event-stream", format!("{}\n\n", response))
        } else {
            ("application/json", response)
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
#[cfg(test)]
pub(crate) fn mock_seq(responses: Vec<(u16, String)>) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for (status, body) in responses {
            let Ok((mut s, _)) = listener.accept() else { break };
            let mut buf = vec![0u8; 16 * 1024];
            let _ = s.read(&mut buf);
            let raw = String::from_utf8_lossy(&buf[..]).to_string();
            log.lock().unwrap().push(raw);
            let resp = format!("HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            let _ = s.write_all(resp.as_bytes());
            let _ = s.flush();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    });
    (port, seen)
}

/// Two-stage mock: serves the first response, then a *stalled* second
/// response whose SSE bytes stop short of `[DONE]`, so the streaming
/// consumer sits on an open connection waiting for content that never
/// arrives. The test cancels from outside to land the wedge test exactly.
#[cfg(test)]
pub(crate) fn mock_stall(first: &'static str, stalled: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for (i, body) in [first, stalled].into_iter().enumerate() {
            let Ok((mut s, _)) = listener.accept() else { return };
            let mut buf = [0u8; 8192];
            let _ = s.read(&mut buf);
            // The first response is complete (Content-Length set, Connection
            // close); the second is *deliberately* open-ended — no
            // Content-Length, no Connection: close — and the thread sleeps
            // so the client sits on the stream waiting for content that
            // never arrives. Cancellation races against this wait.
            let head = if i == 0 {
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())
            } else {
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_string()
            };
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(body.as_bytes());
            let _ = s.flush();
            if i == 1 {
                std::thread::sleep(std::time::Duration::from_secs(10));
            }
        }
    });
    port
}

#[cfg(test)]
pub(crate) fn block_text(b: &crate::ir::Block) -> Option<&str> {
    match b { crate::ir::Block::Text(t) => Some(t), _ => None }
}

#[cfg(test)]
pub(crate) fn block_name(b: &crate::ir::Block) -> &str {
    match b { crate::ir::Block::ToolUse { name, .. } => name, _ => "" }
}

#[cfg(test)]
pub(crate) fn block_input(b: &crate::ir::Block) -> &serde_json::Value {
    match b { crate::ir::Block::ToolUse { input, .. } => input, _ => &serde_json::Value::Null }
}


/// Test-only extension trait — `resp.blocks[i].text()` etc. with no
/// `test_util::` prefix inside the tests. Vendor tests use these because
/// the production code has no reason to reach into a Block variant's
/// inner fields — these getters are test ergonomics, not API.
#[cfg(test)]
pub(crate) trait BlockExt {
    fn text(&self) -> Option<&str>;
    fn name(&self) -> &str;
    fn input(&self) -> &serde_json::Value;
}
#[cfg(test)]
impl BlockExt for crate::ir::Block {
    fn text(&self) -> Option<&str> { block_text(self) }
    fn name(&self) -> &str { block_name(self) }
    fn input(&self) -> &serde_json::Value { block_input(self) }
}
