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

use crate::cancel::CancelToken;
use crate::tools::dispatch;
use serde_json::Value;
use std::io::Read;
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;
use tokio::sync::mpsc;

/// Combined-output shape shared by the blocking registry tool and the
/// cancellable coroutine version below.
fn bash_output(status: Option<&ExitStatus>, out: &str, err: &str) -> String {
    let mut s = format!("exit={}\n{out}", status.and_then(|st| st.code()).unwrap_or(-1));
    if !err.is_empty() {
        if !s.ends_with('\n') { s.push('\n'); }
        s.push_str(err);
    }
    s
}

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
        if killed { break child.wait().ok(); } // kill sent: this returns promptly
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
pub(crate) async fn run_tool(name: &str, input: &Value, token: &CancelToken) -> String {
    if name == "bash" {
        return run_bash(input["command"].as_str().unwrap_or(""), token).await;
    }
    let name = name.to_string();
    let input = input.clone();
    let job = tokio::task::spawn_blocking(move || dispatch(&name, &input));
    tokio::select! {
        out = job => out.unwrap_or_else(|e| format!("error: {e}")),
        _ = token.cancelled() => "error: interrupted by user".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::block_on;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn bash_coroutine_is_killed_by_cancellation() {
        let token = Arc::new(CancelToken::new());
        let start = Instant::now();
        let out = block_on(async {
            let t = token.clone();
            // Cancel from a separate blocking thread, like Ctrl-C would.
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(300));
                t.cancel();
            });
            run_bash("echo started; sleep 30", &token).await
        });
        assert!(out.contains("[interrupted by user]"), "got: {out}");
        assert!(start.elapsed() < Duration::from_secs(5), "took {:?}", start.elapsed());
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

}
