//! The display port: the agent core never touches a terminal. It emits
//! [`Msg`]s through [`disp`], and a *frontend* interprets them — the plain
//! frontend folds them into a log, the TUI folds them into application
//! state. Both frontends are pure functions of the message stream, which is
//! what makes the UI testable without a terminal.
//!
//! With no frontend installed, `disp` routes to a default plain instance —
//! agent-core tests exercise the protocols without driving any UI.

use crate::Usage;
use std::io::IsTerminal;
use std::sync::{Mutex, OnceLock};
use tokio::sync::mpsc::UnboundedSender;

/// Severity of a note row (warnings scroll with the transcript).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sev {
    Warn,
    Err,
}

/// Everything the agent core wants the human to see.
#[derive(Clone, Debug)]
pub enum Msg {
    /// A dim information line (the banner).
    Banner(String),
    /// The user's task is being worked on: echoed, status bar starts.
    TaskBegin(String),
    /// The run finished (normally, or interrupted).
    TaskEnd,
    /// Streamed answer text; may carry partial lines and newlines.
    Text(String),
    /// Streamed reasoning delta — folded away, never shown inline.
    Think(String),
    /// The reasoning block closed: fold it into a marker.
    ThinkEnd,
    /// A tool call completed: name, argument summary, full output.
    Tool { name: String, summary: String, output: String },
    /// A warning or error line.
    Note { sev: Sev, text: String },
    /// The latest request's usage (replaces the live turn).
    Usage(Usage),
    /// Streaming output-token count (ticks up while text flows).
    OutTokens(u64),
    /// The request finished: fold the turn into session totals.
    Done,
}

/// The installed frontend. TUI installs a channel; plain is the default.
static FRONT: OnceLock<Front> = OnceLock::new();

enum Front {
    Chan(UnboundedSender<Msg>),
    Plain(Mutex<Plain>),
}

/// Install the TUI frontend: messages flow to its event loop.
pub fn install_chan(tx: UnboundedSender<Msg>) {
    let _ = FRONT.set(Front::Chan(tx));
}

/// Route one message to the installed frontend. Infallible and quiet: a
/// dropped TUI receiver (or no frontend at all) must never take the agent
/// down — the sea does not care whether anyone is watching.
pub fn disp(m: Msg) {
    match FRONT.get_or_init(|| Front::Plain(Mutex::new(Plain::new()))) {
        Front::Chan(tx) => {
            let _ = tx.send(m);
        }
        Front::Plain(p) => {
            let mut p = p.lock().unwrap();
            for (stream, line) in p.feed(m) {
                match stream {
                    Stream::Out => println!("{line}"),
                    Stream::Err => eprintln!("{line}"),
                }
            }
        }
    }
}

// ---- plain frontend ---------------------------------------------------------

/// Where a rendered plain line goes.
#[derive(PartialEq, Eq, Debug)]
pub enum Stream {
    Out,
    Err,
}

/// How many lines of tool output the plain log echoes per call.
const PLAIN_TOOL_LINES: usize = 20;

/// The plain frontend as a *pure* fold: `feed` consumes one message and
/// returns the finished lines it produced (stdout or stderr). Partial
/// streamed text and partial thinking stay inside until a newline or end
/// closes them — tests pin the exact output.
#[derive(Default)]
pub struct Plain {
    text: String,
    text_open: bool,
    think: String,
    thoughts: usize,
    out: bool,
}

impl Plain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one message into zero or more finished lines.
    pub fn feed(&mut self, m: Msg) -> Vec<(Stream, String)> {
        let mut lines = vec![];
        let emit = |s: Stream, l: String, v: &mut Vec<(Stream, String)>| v.push((s, l));
        match m {
            Msg::Banner(b) => emit(Stream::Out, b, &mut lines),
            Msg::TaskBegin(t) => emit(Stream::Out, format!("jingwei ❯ {t}"), &mut lines),
            Msg::TaskEnd => {
                self.flush(&mut lines);
                self.fold_think(&mut lines);
            }
            Msg::Text(t) => {
                self.out = true;
                self.text.push_str(&t);
                self.flush(&mut lines);
            }
            Msg::Think(t) => self.think.push_str(&t),
            Msg::ThinkEnd => self.fold_think(&mut lines),
            Msg::Tool { name, summary, output } => {
                self.flush(&mut lines);
                emit(Stream::Out, format!("[{name}] {summary}"), &mut lines);
                let all: Vec<&str> = output.lines().collect();
                let shown = all.len().min(PLAIN_TOOL_LINES);
                for l in &all[..shown] {
                    emit(Stream::Out, format!("│ {l}"), &mut lines);
                }
                if all.len() > shown {
                    emit(Stream::Out, format!("… +{} more lines", all.len() - shown), &mut lines);
                }
            }
            Msg::Note { sev, text } => emit(match sev {
                Sev::Warn | Sev::Err => Stream::Err,
            }, text, &mut lines),
            // usage lives in the TUI's status bar; the plain log has none
            Msg::Usage(_) | Msg::OutTokens(_) | Msg::Done => {
                if let Msg::Done = m {
                    self.flush(&mut lines);
                    self.fold_think(&mut lines);
                }
            }
        }
        lines
    }

    /// Land complete lines of streamed text; the partial tail stays buffered.
    fn flush(&mut self, lines: &mut Vec<(Stream, String)>) {
        while let Some(i) = self.text.find('\n') {
            let line: String = self.text.drain(..=i).collect();
            lines.push((Stream::Out, line.trim_end_matches('\n').to_string()));
        }
        self.text_open = !self.text.is_empty();
    }

    /// Fold accumulated reasoning into a marker line; interrupted thinking
    /// still folds, so nothing reasoned is lost.
    fn fold_think(&mut self, lines: &mut Vec<(Stream, String)>) {
        if self.think.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.think);
        self.thoughts += 1;
        let n = text.lines().count().max(1);
        lines.push((Stream::Out, format!("▸ thought #{} · {} lines", self.thoughts, n)));
    }

    /// Terminate any open streamed line (plain streams are line-oriented).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn close(&mut self) -> Vec<(Stream, String)> {
        let mut lines = vec![];
        self.flush(&mut lines);
        if self.text_open || !self.text.is_empty() {
            let rest = std::mem::take(&mut self.text);
            self.text_open = false;
            lines.push((Stream::Out, rest));
        }
        self.fold_think(&mut lines);
        lines
    }
}

/// Drive the plain REPL: read lines, run the agent, print via the port.
/// Used when stdout is not a terminal (pipes, one-shot runs) or when
/// `JINGWEI_NO_TUI` asks for the log form.
pub async fn plain_repl(cfg: &crate::Config) -> crate::Result<()> {
    use crate::{agent_turn, CancelToken};
    let tty = std::io::stdin().is_terminal();
    println!("jingwei — 精卫填海，一石一石 · type a task, Ctrl-C interrupts, Ctrl-D rests");
    println!("{} · {} · {}", cfg.protocol_label(), cfg.model, cfg.base_url);
    let mut history: Vec<serde_json::Value> = vec![];
    let stdin = std::io::stdin();
    loop {
        if tty {
            print!("jingwei> ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            // bad bytes: warn and carry on, the way the old readline did
            Err(e) => {
                disp(Msg::Note { sev: Sev::Warn, text: format!("warning: input ({e})") });
                continue;
            }
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        history.push(serde_json::json!({"role": "user", "content": line}));
        disp(Msg::TaskBegin(line));
        let token = CancelToken::new();
        match agent_turn(cfg, &mut history, &token).await {
            Err(crate::Error::Interrupted) => {}
            Err(e) => disp(Msg::Note { sev: Sev::Err, text: format!(" error: {e} ") }),
            Ok(()) => {}
        }
        disp(Msg::TaskEnd);
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn run(msgs: Vec<Msg>) -> Vec<(Stream, String)> {
        let mut p = Plain::new();
        let mut out = vec![];
        for m in msgs {
            out.extend(p.feed(m));
        }
        out.extend(p.close());
        out
    }

    #[test]
    fn plain_folds_streamed_text_and_thinking() {
        let out = run(vec![
            Msg::TaskBegin("count files".into()),
            Msg::Think("step one\nstep two".into()),
            Msg::ThinkEnd,
            Msg::Text("there are 3".into()),
            Msg::Text(" files\n".into()),
            Msg::Done,
            Msg::TaskEnd,
        ]);
        let texts: Vec<String> = out.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(texts, vec![
            "jingwei ❯ count files",
            "▸ thought #1 · 2 lines",
            "there are 3 files",
        ]);
        assert!(out.iter().all(|(s, _)| *s == Stream::Out), "no stderr in a clean run");
    }

    #[test]
    fn plain_tool_echo_is_capped() {
        let out = run(vec![Msg::Tool {
            name: "bash".into(),
            summary: "$ seq 1 30".into(),
            output: (1..=30).map(|i| i.to_string()).collect::<Vec<_>>().join("\n"),
        }]);
        assert_eq!(out.len(), 1 + 20 + 1);
        assert_eq!(out[0].1, "[bash] $ seq 1 30");
        assert_eq!(out[1].1, "│ 1");
        assert_eq!(out.last().unwrap().1, "… +10 more lines");
    }

    #[test]
    fn interrupted_thinking_still_folds_on_done() {
        let out = run(vec![Msg::Think("half a thought".into()), Msg::Done]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, "▸ thought #1 · 1 lines");
    }

    #[test]
    fn notes_go_to_stderr() {
        let out = run(vec![Msg::Note { sev: Sev::Warn, text: " trimmed history ".into() }]);
        assert_eq!(out, vec![(Stream::Err, " trimmed history ".to_string())]);
    }
}
