// jingwei — 精卫填海. A minimal coding agent: speak a task, and a small agent
// carries stones — one tool call at a time — until the sea is land.
//
// The binary is a thin process shim: it owns the tokio runtime, the exit
// codes, and the red error banner — everything agent-shaped lives in the
// library (`src/lib.rs` and below), where tests reach it without spawning
// a process. If you are adding behavior, it almost certainly does not
// belong in this file.

use jingwei::ui::{ERR_BG, paint};
use jingwei::{run, Error};
use std::time::Duration;

fn main() {
    // The agent runs as a coroutine on a single-threaded cooperative
    // scheduler: blocking work (HTTP, file tools) is farmed to the
    // blocking pool so every await is a real suspension point; Ctrl-C
    // rides the same machinery (see `CancelToken` and `agent::runner::run_agent`).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("runtime init: {e}"));
    let result = rt.block_on(run());
    // Abandoned blocking workers (an interrupted HTTP call still inside its
    // read timeout) must not stall shutdown — give them one second to
    // wind down, then drop the runtime regardless.
    rt.shutdown_timeout(Duration::from_secs(1));
    match result {
        Ok(()) => {}
        // interrupted by Ctrl-C — exit the way a shell command would
        Err(Error::Interrupted) => std::process::exit(130),
        Err(e) => {
            eprintln!("{}", paint(&format!(" jingwei: {e} "), ERR_BG, true));
            std::process::exit(1);
        }
    }
}
