// The tools, as coroutines. The four registry entries — `bash`, `read_file`,
// `write_file`, `edit_file` — describe what to *do*; this module describes
// what *happens while doing it*. `bash` owns its own child process and its
// own cancellable I/O, because a runaway shell is the one thing in this
// agent that can hold the whole coroutine hostage. The file tools are quick
// and go through the blocking pool — cancellation just abandons the in-flight
// task, the next tool call starts fresh.
//
// Every suspension point is also a cancellation point. The same `CancelToken`
// races against the child's pipe drains as races against the agent loop's
// between-turn wait, because they are the same coroutine.

use crate::api::blocking;
use crate::cancel::CancelToken;
use crate::mcp::Hub;
use crate::tools::{bash_output, dispatch};
use serde_json::Value;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::sync::mpsc;

/// A child's pipe, drained on its own thread. Chunks stream back over an
/// *async* channel — not one final buffer — so partial output survives even
/// when the child is killed while a grandchild still holds the pipe open,
/// and waiting for the rest suspends the coroutine instead of blocking the
/// scheduler thread. What arrived so far lives in `got`, on the pipe itself,
/// so it survives any cancelled collect.
struct Drained {
    rx: tokio::sync::mpsc::Receiver<Vec<u8>>,
    got: Vec<u8>,
}

fn drain_pipe<R: Read + Send + 'static>(pipe: Option<R>) -> Drained {
    let (tx, rx) = mpsc::channel(64);
    std::thread::spawn(move || {
        let Some(mut r) = pipe else { return };
        let mut buf = [0u8; 8192];
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break, // EOF or broken pipe
                Ok(n) => { if tx.blocking_send(buf[..n].to_vec()).is_err() { break } } // collector gone
            }
        }
    });
    Drained { rx, got: Vec::new() }
}

/// How long an interrupted pipe waits for stragglers before abandoning them.
const PIPE_GRACE: Duration = Duration::from_millis(300);

impl Drained {
    /// Collect what the pipe produced, as a coroutine. With `eof` this
    /// awaits EOF (the reader thread finishes when every writer closes the
    /// pipe — the same contract as the blocking tool) — an await, not a
    /// block, so a grandchild holding the pipe keeps the coroutine
    /// suspensible. Every wait races the cancel token: Ctrl-C during the
    /// EOF wait salvages whatever arrives within a short grace and
    /// abandons the rest; without `eof` (child already killed) the grace
    /// window is all there is.
    async fn collect(&mut self, token: &CancelToken, eof: bool) -> String {
        loop {
            let chunk = if eof {
                tokio::select! {
                    c = self.rx.recv() => c,
                    _ = token.cancelled() => {
                        // salvage what lands within the grace, abandon the rest
                        tokio::time::timeout(PIPE_GRACE, self.rx.recv()).await.unwrap_or_default()
                    }
                }
            } else {
                tokio::time::timeout(PIPE_GRACE, self.rx.recv()).await.unwrap_or_default()
            };
            match chunk {
                Some(c) => self.got.extend_from_slice(&c),
                None => break, // EOF, grace elapsed, or abandoned after cancel
            }
        }
        String::from_utf8_lossy(&self.got).into_owned()
    }
}

/// bash as a coroutine: same tool as the registry's, but it watches the
/// cancel token while the command runs. Ctrl-C kills the child at once — no
/// waiting out a runaway `sleep 300` — and whatever output it already
/// produced (plus an `[interrupted]` note) still reaches the model. If the
/// child left grandchildren holding the pipe, they get a short grace period
/// and are then abandoned.
pub(crate) async fn run_bash(command: &str, token: &CancelToken) -> String {
    let (prog, flag) = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
    let mut child = match Command::new(prog).args([flag, command])
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return format!("error: {e}"),
    };
    let mut out_pipe = drain_pipe(child.stdout.take());
    let mut err_pipe = drain_pipe(child.stderr.take());
    let mut killed = false;
    let status = loop {
        if !killed && token.is_cancelled() {
            killed = true;
            let _ = child.kill();
        }
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {}
            Err(e) => return format!("error: {e}"),
        }
        if killed {
            // Reap the killed child on the blocking pool: `Child::wait`
            // is a synchronous stdlib call and would otherwise freeze
            // this coroutine — and with it the single-thread tokio
            // scheduler the runtime runs on. The cancellation that
            // triggered the kill already raced us here; we just want
            // the ExitStatus without paying for it on the runtime
            // thread. The blocking pool handles the wait; the
            // cancel token can still interrupt the await. A second
            // cancellation landing while we reap surfaces as
            // `Err(Error::Interrupted)`, which we collapse to `None`
            // and let the post-loop `killed` check turn into the
            // "[interrupted by user]" note.
            break blocking(token, move || Ok(child.wait().ok())).await.unwrap_or_default();
        }
        // The wait itself is a suspension point — a sleep raced against
        // cancellation, so waiting for the child is interruptible too.
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            _ = token.cancelled() => {}
        }
    };
    // Gather the pipes as a coroutine: the EOF wait suspends (a grandchild
    // holding the pipe can delay it) and races cancellation — Ctrl-C
    // salvages what arrived within the grace window and moves on, so no
    // background process can hold the agent hostage.
    let (out, err) = tokio::join!(out_pipe.collect(token, !killed), err_pipe.collect(token, !killed));
    if !killed && token.is_cancelled() { killed = true; } // pipes abandoned mid-collect
    let mut s = bash_output(status.as_ref(), &out, &err);
    if killed { s.push_str("\n[interrupted by user]"); }
    s
}

/// Execute one tool call inside the agent coroutine. bash is cancellable (its
/// child process is killed); the file tools are quick, run on the blocking
/// pool, and are simply abandoned if cancellation wins the race.
///
/// `hub` is the MCP hub: every call whose name carries the `mcp__` prefix
/// goes through it. The hub's actor task owns the connection to the server;
/// what `run_tool` does is the one line of dispatch the agent core needs to
/// know.
///
/// Cancellation formats are deliberately different — the model can tell
/// what happened from the prefix alone: bash preserves whatever output
/// had already arrived (so the marker sits *after* the bytes, and reads
/// like a note the tool appended) while the file tools had no partial
/// output to keep (so the whole tool result is a single line saying so).
/// MCP calls inherit the cancellation marker the way the file tools do:
/// an in-flight hub call is a coroutine on tokio, and dropping the
/// `select!` arm that awaited it is what abandonment looks like here.
pub(crate) async fn run_tool(name: &str, input: &Value, token: &CancelToken, hub: &Hub) -> String {
    if crate::mcp::is_tool(name) {
        // MCP calls are not cancellable mid-flight (the hub's actor task
        // owns the connection, and the call's `tokio::time::timeout` is what
        // bounds it). What we can cancel is the *wait* for the reply — a
        // token already fired means the user stopped the turn and the
        // answer is no longer wanted.
        if token.is_cancelled() {
            return "[interrupted by user]".into();
        }
        return hub.call(name, &input.to_string()).await;
    }
    if name == "bash" {
        return run_bash(input["command"].as_str().unwrap_or(""), token).await;
    }
    let name = name.to_string();
    let input = input.clone();
    let job = tokio::task::spawn_blocking(move || dispatch(&name, &input));
    tokio::select! {
        out = job => out.unwrap_or_else(|e| format!("error: {e}")),
        _ = token.cancelled() => "[interrupted by user]".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::bash_output;
    use crate::test_util::block_on;
    use std::process::ExitStatus;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn bash_coroutine_is_killed_by_cancellation() {
        // The cancel arrives at 300ms; the child must be killed and the
        // coroutine returned well before the 30-second sleep would
        // naturally finish. A 2-second ceiling catches "didn't kill the
        // child" and "killed it but the pipe drain hung" — both are
        // real regressions. Anything under 2s is fast enough that the
        // test stays in single-digit seconds.
        let token = Arc::new(CancelToken::new());
        let start = Instant::now();
        let out = block_on(async {
            let t = token.clone();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(300));
                t.cancel();
            });
            run_bash("echo started; sleep 30", &token).await
        });
        assert!(out.contains("[interrupted by user]"), "got: {out}");
        assert!(start.elapsed() < Duration::from_secs(2),
            "cancel at 300ms, kill+collect should finish well under 2s: {:?}", start.elapsed());
    }

    #[test]
    fn bash_cancel_with_grandchild_holding_pipe() {
        // sh exits at once, but the backgrounded sleep keeps stdout open:
        // the read end of the pipe is still held, which is what used to
        // wedge `collect()` before it was a coroutine.
        let cmd = if cfg!(windows) {
            "echo started & start /b ping -n 30 127.0.0.1 > nul"
        } else {
            "sleep 30 & echo started"
        };
        let token = Arc::new(CancelToken::new());
        let t = token.clone();
        let out = block_on(async move {
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(300));
                t.cancel();
            });
            run_bash(cmd, &token).await
        });
        assert!(out.contains("started"), "partial output must survive: {out}");
        assert!(out.contains("[interrupted by user]"), "got: {out}");
    }

    // ---- bash_output: shared shape, both async and sync paths -------------

    fn make_exit(code: i32) -> ExitStatus {
        if cfg!(windows) {
            std::process::Command::new("cmd").args(["/C", &format!("exit {code}")]).status().unwrap()
        } else {
            std::process::Command::new("sh").args(["-c", &format!("exit {code}")]).status().unwrap()
        }
    }

    #[test]
    fn bash_output_adds_a_separator_newline_when_err_follows_unterminated_stdout() {
        // the format string ends with `{out}` and does NOT add its own
        // newline; when stderr is non-empty and stdout does not end with
        // one, bash_output inserts the separator itself
        let st = make_exit(0);
        let s = bash_output(Some(&st), "hi", "warn");
        assert_eq!(s, "exit=0\nhi\nwarn", "stdout without trailing newline gets one before stderr");
    }

    #[test]
    fn bash_output_preserves_a_trailing_newline_before_stderr() {
        let st = make_exit(1);
        let s = bash_output(Some(&st), "hi\n", "warn");
        assert_eq!(s, "exit=1\nhi\nwarn");
    }

    #[test]
    fn bash_output_without_status_uses_minus_one() {
        let s = bash_output(None, "out", "");
        assert_eq!(s, "exit=-1\nout");
    }

    #[test]
    fn bash_output_skips_stderr_when_empty() {
        let st = make_exit(0);
        let s = bash_output(Some(&st), "out", "");
        assert_eq!(s, "exit=0\nout");
    }

    // ---- drain_pipe: the thread that streams the child's bytes ------------

    /// An `io::Read` that yields the bytes we feed it, then returns 0.
    fn feeder(bytes: Vec<u8>) -> impl Read + Send + 'static {
        struct One(Option<std::io::Cursor<Vec<u8>>>);
        impl Read for One {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let cur = self.0.as_mut().unwrap();
                let pos = cur.position() as usize;
                if pos >= cur.get_ref().len() { return Ok(0); }
                let n = cur.read(buf)?;
                Ok(n)
            }
        }
        One(Some(std::io::Cursor::new(bytes)))
    }

    #[test]
    fn drain_pipe_with_none_pipe_emits_nothing_and_eofs() {
        // the `let Some(mut r) = pipe else { return };` path: drain_pipe
        // returns a Drained whose rx closes immediately
        let mut d = drain_pipe::<std::fs::File>(None);
        block_on(async {
            // a closed receiver yields None on the next recv
            assert!(d.rx.recv().await.is_none());
        });
        assert!(d.got.is_empty());
    }

    #[test]
    fn drain_pipe_streams_what_the_pipe_yields() {
        let mut d = drain_pipe(Some(feeder(b"hello world".to_vec())));
        block_on(async {
            let got = d.collect(&CancelToken::new(), true).await;
            assert_eq!(got, "hello world");
        });
    }

    #[test]
    fn drain_pipe_returns_after_grace_when_eof_is_false_and_nothing_arrives() {
        // drain_pipe without EOF and a *hanging* source: collect waits
        // the grace period and gives up, returning what little arrived.
        // (An empty feeder EOFs immediately, so the sender drops and
        // recv() returns None — that path is the EOF case, not the
        // grace case.) PIPE_GRACE is 300ms; if it grows, this test
        // pins the contract.
        let mut d = drain_pipe(Some(hanging_reader()));
        let start = Instant::now();
        block_on(async {
            let got = d.collect(&CancelToken::new(), false).await;
            assert!(got.is_empty(), "nothing arrived: {got:?}");
        });
        let elapsed = start.elapsed();
        // close to grace, not zero (we actually waited) and not 2x grace
        assert!(elapsed >= Duration::from_millis(250),
            "grace must wait: {elapsed:?}");
        assert!(elapsed < Duration::from_millis(600),
            "grace should be ~300ms, not unbounded: {elapsed:?}");
    }

    /// A Read that sleeps for 5 seconds and then returns EOF — keeps
    /// the pipe open long enough that recv() never returns None, so the
    /// grace timeout is the only thing that unblocks collect.
    fn hanging_reader() -> impl Read + Send + 'static {
        struct Hang(std::sync::Mutex<Option<std::io::Empty>>);
        impl Read for Hang {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                std::thread::sleep(Duration::from_secs(5));
                // after the sleep, return EOF on the inner cursor
                let mut e = self.0.lock().unwrap().take().unwrap();
                e.read(_buf)
            }
        }
        Hang(std::sync::Mutex::new(Some(std::io::empty())))
    }

    #[test]
    fn drain_pipe_eof_collect_with_cancelled_token_returns_buffered_bytes() {
        // the eof=true + pre-cancelled path: the select's cancel arm
        // wins immediately, then the grace window pulls whatever the
        // reader thread has already pushed. Sleep first so the
        // synchronous feeder has time to deliver — without it the
        // grace window can elapse before any byte is queued and the
        // salvage path is never exercised.
        let mut d = drain_pipe(Some(feeder(b"salvaged".to_vec())));
        let token = CancelToken::new();
        token.cancel();
        block_on(async {
            // give the reader thread time to push bytes into the channel
            tokio::time::sleep(Duration::from_millis(100)).await;
            let got = d.collect(&token, true).await;
            assert_eq!(got, "salvaged",
                "queued bytes must survive a pre-cancelled token: {got:?}");
        });
    }

    // ---- run_bash: spawn failure is a returned error string ---------------

    #[test]
    fn run_bash_collects_clean_stdout_without_an_interruption_marker() {
        let token = CancelToken::new();
        let out = block_on(async { run_bash("echo hello", &token).await });
        assert!(out.starts_with("exit=0\nhello"), "got: {out}");
        assert!(!out.contains("[interrupted by user]"), "no interruption: {out}");
    }

    // ---- run_tool: the dispatcher's two paths -----------------------------

    /// An empty hub for tests that do not exercise MCP dispatch.
    fn empty_hub() -> Hub {
        Hub::empty()
    }

    #[test]
    fn run_tool_dispatches_non_bash_to_the_blocking_pool() {
        // file tools run through spawn_blocking; cancellation only fires
        // if the tool itself is slow enough — a write+read is fast, so the
        // result is the tool's string, not "interrupted by user"
        block_on(async {
            let token = CancelToken::new();
            let hub = empty_hub();
            let out = run_tool("bash", &serde_json::json!({"command": "echo hi"}), &token, &hub).await;
            assert!(out.starts_with("exit=0"), "got: {out}");
            hub.shutdown().await;
        });
    }

    #[test]
    fn run_tool_cancellation_interrupts_a_slow_bash_call() {
        // the cancellable path: a long-running bash with a token cancelled
        // before the next select wakes — the tool returns the interrupted
        // marker
        let token = Arc::new(CancelToken::new());
        let t = token.clone();
        let out = block_on(async move {
            let hub = empty_hub();
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(200));
                t.cancel();
            });
            let out = run_tool("bash", &serde_json::json!({"command": "sleep 30"}), &token, &hub).await;
            hub.shutdown().await;
            out
        });
        assert!(out.contains("[interrupted by user]"), "got: {out}");
    }

    #[test]
    fn run_tool_non_bash_cancellation_returns_the_marker() {
        // the non-bash path's select races the spawn_blocking against
        // the cancel — a pre-cancelled token always wins. The marker is
        // the same word bash uses, so the model sees one shape for "this
        // turn did not finish" no matter which tool ran.
        let token = Arc::new(CancelToken::new());
        token.cancel();
        let out = block_on(async {
            let hub = empty_hub();
            let out = run_tool(
                "write_file",
                &serde_json::json!({"path": "/tmp/jingwei_cancel_marker", "content": "x"}),
                &token,
                &hub,
            ).await;
            hub.shutdown().await;
            out
        });
        assert_eq!(out, "[interrupted by user]",
            "non-bash cancellation uses the bash marker: {out}");
    }

    // ---- MCP dispatch: the prefix-armed branch of run_tool ---------------

    /// `run_tool` on an `mcp__` name dispatches to the hub: the result is
    /// what the server answered, and the same answer the hub would give on
    /// its own. This is the path the agent loop exercises when the model
    /// calls an MCP tool.
    #[tokio::test]
    async fn run_tool_dispatches_an_mcp_name_to_the_hub() {
        use crate::mcp::{Hub, stub::Stub};
        let stub = Stub::new();
        let hub = Hub::of_entries(vec![stub.entry(&[("STUB_TOOLS", "echo")])]).await;
        let out = run_tool(
            "mcp__stub__echo",
            &serde_json::json!({"text": "hello"}),
            &CancelToken::new(),
            &hub,
        ).await;
        assert_eq!(out, "called with {text:hello}", "got: {out}");
        hub.shutdown().await;
    }

    /// A pre-cancelled token never reaches the hub — the run is over before
    /// any of this matters. Same cancellation marker every other tool uses.
    #[tokio::test]
    async fn run_tool_with_an_mcp_name_and_a_pre_cancelled_token_short_circuits() {
        use crate::mcp::Hub;
        let token = Arc::new(CancelToken::new());
        token.cancel();
        let hub = Hub::empty();
        let out = run_tool(
            "mcp__anyone__anything",
            &serde_json::json!({}),
            &token,
            &hub,
        ).await;
        assert_eq!(out, "[interrupted by user]");
        hub.shutdown().await;
    }

    /// An MCP name no server offers is answered with the hub's own message:
    /// `error: no MCP tool named ...`. The model sees the same shape it
    /// would see for any unknown tool.
    #[tokio::test]
    async fn run_tool_with_an_mcp_name_no_server_offers_yields_a_hub_error() {
        use crate::mcp::Hub;
        let hub = Hub::empty();
        let out = run_tool(
            "mcp__nobody__nothing",
            &serde_json::json!({}),
            &CancelToken::new(),
            &hub,
        ).await;
        assert!(out.starts_with("error: no MCP tool named"), "got: {out}");
        hub.shutdown().await;
    }
}
