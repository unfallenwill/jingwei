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

/// The name of the variable that points at the home directory.
#[cfg(test)]
pub(crate) fn home_var() -> &'static str {
    if cfg!(windows) { "USERPROFILE" } else { "HOME" }
}

/// A home directory a test may write into, and the name of the variable
/// that points at it.
///
/// The guard owns the dir as well: the session's path points into it, so
/// the dir has to outlive the writes that use that path and be removed
/// once they are done. A test that takes the guard also takes the env
/// lock, so a sibling test that moves the same variable at the same time
/// cannot see the wrong home.
#[cfg(test)]
pub(crate) struct HomeOverride {
    pub(crate) prev: Option<std::ffi::OsString>,
    pub(crate) dir: PathBuf,
}

#[cfg(test)]
impl HomeOverride {
    pub(crate) fn new(tag: &str) -> Self {
        let dir = temp_dir(&format!("home_{tag}"));
        let prev = std::env::var_os(home_var());
        std::env::set_var(home_var(), &dir);
        Self { prev, dir }
    }
}

#[cfg(test)]
impl Drop for HomeOverride {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(home_var(), v),
            None => std::env::remove_var(home_var()),
        }
        let _ = fs::remove_dir_all(&self.dir);
    }
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
impl crate::ui::Show for NullSink {
    fn show(&self, _m: crate::ui::Msg) {}
}
#[cfg(test)]
pub(crate) fn sink() -> NullSink {
    NullSink
}

/// A sink that keeps every message: the tests assert on the sequence the
/// port received, not on what a frontend would draw with it.
///
/// One recorder for the whole crate. The api layer, the run boundary, the
/// tool tier and the plain frontend each wanted one, and four copies of
/// the same ten lines is four places to fix the same bug.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct Recorder(Arc<Mutex<Vec<crate::ui::Msg>>>);

#[cfg(test)]
impl Recorder {
    /// Every message, in the order it arrived.
    pub(crate) fn msgs(&self) -> Vec<crate::ui::Msg> {
        self.0.lock().unwrap().clone()
    }
    /// The last message — and it leaves the record.
    pub(crate) fn pop(&self) -> Option<crate::ui::Msg> {
        self.0.lock().unwrap().pop()
    }
}

#[cfg(test)]
impl crate::ui::Show for Recorder {
    fn show(&self, m: crate::ui::Msg) {
        self.0.lock().unwrap().push(m);
    }
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
        max_agents: crate::config::DEFAULT_MAX_AGENTS,
        agent_wait_ms: crate::config::DEFAULT_AGENT_WAIT_MS,
    }
}

/// A bare `Context` for tests that don't care about AGENTS.md or the
/// tool list. Vendors take `&Context`; tests that exercise `body()` or
/// `call_api()` need a handle, and wiring one by hand is noise. The empty
/// AGENTS.md context gives us a system_text equal to `crate::core::prompt::SYSTEM`,
/// which is what nearly every test wants anyway.
#[cfg(test)]
pub(crate) fn ctx() -> crate::core::context::Context {
    let md = crate::core::agents_md::AgentsMdContext::empty(&std::path::PathBuf::from("."));
    crate::core::context::Context::new(&md, crate::config::DEFAULT_CONTEXT_SIZE)
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
    mock_seq_with(responses, || {})
}

/// Like [`mock_seq`], but with a hook: it fires after a request has been
/// read and *before* that request's response goes out. A test that wants
/// to look at the world at the exact moment the model is called — the
/// file on disk, the transcript in memory — uses it to look, instead of
/// guessing what the timing would have been.
#[cfg(test)]
pub(crate) fn mock_seq_with<F: Fn() + Send + 'static>(
    responses: Vec<(u16, String)>,
    on_request: F,
) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for (status, body) in responses {
            let Ok((mut s, _)) = listener.accept() else { break };
            // A request split across TCP packets (common under parallel
            // test load) still comes through whole when we drain until
            // EOF instead of trusting a single read.
            let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(200)));
            let mut req = Vec::new();
            let mut buf = [0u8; 16_384];
            while let Ok(n) = s.read(&mut buf) {
                if n == 0 { break }
                req.extend_from_slice(&buf[..n]);
            }
            log.lock().unwrap().push(String::from_utf8_lossy(&req).into_owned());
            on_request();
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
/// Like [`mock_seq`] but the second response stalls: the head and the
/// given events are written, then the connection is *held open* — the
/// client waits for content that never arrives. Returns the port plus a
/// flag that turns true once the stalled bytes are on the wire, so a
/// cancellation test can wait for the exact moment to interrupt instead
/// of racing a fixed delay against scheduler load.
pub(crate) fn mock_stall(first: &'static str, stalled: &'static str) -> (u16, std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let stalled_written = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = stalled_written.clone();
    std::thread::spawn(move || {
        for (i, body) in [first, stalled].into_iter().enumerate() {
            let Ok((mut s, _)) = listener.accept() else { return };
            let mut buf = [0u8; 8192];
            let _ = s.read(&mut buf);
            // The first response is complete (Content-Length set, Connection
            // close); the second is open-ended — no Content-Length — and the
            // thread sleeps so the client sits on the stream waiting for
            // content that never arrives. Cancellation lands against this
            // wait.
            let head = if i == 0 {
                format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())
            } else {
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n".to_string()
            };
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(body.as_bytes());
            let _ = s.flush();
            if i == 0 {
                // Drain until the client closes: dropping the socket while
                // the client may still be reading (or still writing its
                // request) races a reset into its stream — the flake this
                // guard exists for. EOF is the client's move, not ours.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while std::time::Instant::now() < deadline {
                    match s.read(&mut buf) {
                        Ok(0) => break,
                        Err(_) => break,
                        Ok(_) => {}
                    }
                }
            } else {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_secs(10));
            }
        }
    });
    (port, stalled_written)
}

#[cfg(test)]
pub(crate) fn block_text(b: &crate::core::ir::Block) -> Option<&str> {
    match b { crate::core::ir::Block::Text(t) => Some(t), _ => None }
}

#[cfg(test)]
pub(crate) fn block_name(b: &crate::core::ir::Block) -> &str {
    match b { crate::core::ir::Block::ToolUse { name, .. } => name, _ => "" }
}

#[cfg(test)]
pub(crate) fn block_input(b: &crate::core::ir::Block) -> &serde_json::Value {
    match b { crate::core::ir::Block::ToolUse { input, .. } => input, _ => &serde_json::Value::Null }
}


/// The JSON body of a raw HTTP request as the mock recorded it: the bytes
/// after the head/body separator. A test that wants to assert on what
/// went on the wire slices the body out of the raw text and parses it.
#[cfg(test)]
pub(crate) fn request_body(raw: &str) -> serde_json::Value {
    let body = raw.split_once("\r\n\r\n").map(|(_, b)| b).unwrap_or(raw);
    serde_json::from_str(body).unwrap_or_else(|e| panic!("the request body is json: {e}\n{raw}"))
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
impl BlockExt for crate::core::ir::Block {
    fn text(&self) -> Option<&str> { block_text(self) }
    fn name(&self) -> &str { block_name(self) }
    fn input(&self) -> &serde_json::Value { block_input(self) }
}
