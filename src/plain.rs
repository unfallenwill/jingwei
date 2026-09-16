//! The plain frontend: a pure fold over [`Msg`]s into a line-oriented log,
//! plus the dumb line reader that drives it (pipes, one-shot runs,
//! `JINGWEI_NO_TUI=1`). The TUI's sibling — same port, same vocabulary
//! (`crate::display`), none of the terminal machinery.

use crate::display::{self, disp, Msg, Sev};
use std::io::{IsTerminal, Write};

/// Where a rendered plain line goes.
#[derive(PartialEq, Eq, Debug)]
pub enum Stream {
    Out,
    Err,
}

/// How many body lines the plain log echoes per tool call. The log has no
/// unfolding, so its echo must be self-sufficient — hence 20 lines, where
/// the TUI's collapsed fold shows only [`crate::tui::view::TOOL_TAIL_SHOWN`]
/// because Ctrl-O exists to unfold the rest.
const PLAIN_TOOL_LINES: usize = 20;

/// The plain frontend as a *pure* fold: `feed` consumes one message and
/// returns the finished lines it produced (stdout or stderr). Partial
/// streamed text stays buffered until a newline closes it or the turn ends
/// (`Done`/`TaskEnd` land the tail even without a trailing newline) —
/// tests pin the exact output.
#[derive(Default)]
pub struct Plain {
    text: String,
    think: String,
    thoughts: usize,
}

impl Plain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one message into zero or more finished lines.
    pub fn feed(&mut self, m: Msg) -> Vec<(Stream, String)> {
        let mut lines = vec![];
        match m {
            Msg::Banner(b) => lines.push((Stream::Out, b)),
            Msg::TaskBegin(t) => {
                lines.push((Stream::Out, format!("{}{}{t}", display::PROMPT_HEAD, display::PROMPT_GUTTER)))
            }
            Msg::TaskEnd => self.end_of_turn(&mut lines),
            Msg::Text(t) => {
                self.text.push_str(&t);
                self.flush(&mut lines);
            }
            Msg::Think(t) => self.think.push_str(&t),
            Msg::ThinkEnd => self.fold_think(&mut lines),
            Msg::Tool { name, summary, output } => {
                self.end_of_text(&mut lines);
                lines.push((Stream::Out, format!("[{name}] {summary}")));
                let all: Vec<&str> = output.lines().collect();
                let shown = all.len().min(PLAIN_TOOL_LINES);
                for l in &all[..shown] {
                    lines.push((Stream::Out, format!("│ {l}")));
                }
                if all.len() > shown {
                    lines.push((Stream::Out, format!("{}… +{} more lines", display::FOLD_GUTTER, all.len() - shown)));
                }
            }
            // both severities ride stderr: notes are asides to the answer,
            // and the answer itself is stdout
            Msg::Note { text, .. } => lines.push((Stream::Err, text)),
            // usage lives in the TUI's status bar; the plain log has none
            Msg::Usage(_) | Msg::OutTokens(_) => {}
            Msg::Done => self.end_of_turn(&mut lines),
        }
        lines
    }

    /// A turn ended: land the open text line (with or without its newline)
    /// and fold whatever reasoning was still arriving.
    fn end_of_turn(&mut self, lines: &mut Vec<(Stream, String)>) {
        self.end_of_text(lines);
        self.fold_think(lines);
    }

    /// Land complete lines of streamed text; the partial tail stays buffered.
    fn flush(&mut self, lines: &mut Vec<(Stream, String)>) {
        while let Some(i) = self.text.find('\n') {
            let line: String = self.text.drain(..=i).collect();
            lines.push((Stream::Out, line.trim_end_matches('\n').to_string()));
        }
    }

    /// Close the streamed text: complete lines, then the unterminated tail
    /// as one last line. The wire's final chunk owes us no newline.
    fn end_of_text(&mut self, lines: &mut Vec<(Stream, String)>) {
        self.flush(lines);
        if !self.text.is_empty() {
            let rest = std::mem::take(&mut self.text);
            lines.push((Stream::Out, rest));
        }
    }

    /// Fold accumulated reasoning into a marker line; interrupted thinking
    /// still folds, so nothing reasoned is lost.
    fn fold_think(&mut self, lines: &mut Vec<(Stream, String)>) {
        if self.think.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.think);
        self.thoughts += 1;
        let mut all = text.lines();
        let first = all.next().unwrap_or("");
        lines.push((Stream::Out, display::thought_folded(self.thoughts, first, all.count())));
    }
}

/// Drive the plain REPL: read lines, run the agent, print via the port.
/// Used when stdout is not a terminal (pipes, one-shot runs) or when
/// `JINGWEI_NO_TUI` asks for the log form.
///
/// No editor here, so no input-history file: recall is a TUI feature; this
/// path keeps the shell's own history (up-arrow) and line editing.
pub async fn plain_repl(cfg: &crate::Config) -> crate::Result<()> {
    use crate::{agent_turn, CancelToken};
    let tty = std::io::stdin().is_terminal();
    display::disp(Msg::Banner("jingwei — 精卫填海，一石一石 · type a task, /exit or Ctrl-D rests, Ctrl-C interrupts".into()));
    disp(Msg::Banner(cfg.identity()));
    let mut history: Vec<serde_json::Value> = vec![];
    let stdin = std::io::stdin();
    loop {
        if tty {
            print!("{}{} ", display::PROMPT_HEAD, display::PROMPT_GUTTER.trim());
            let _ = std::io::stdout().flush();
        }
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            // bad bytes on stdin: warn and carry on
            Err(e) => {
                disp(Msg::Note { sev: Sev::Warn, text: format!("warning: input ({e})") });
                continue;
            }
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        // `/exit` leaves the log REPL too — the same word the TUI speaks,
        // checked at the same place in the flow: after the trim, before
        // the line becomes a task
        if display::is_exit(&line) {
            break;
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
            format!("{}{}count files", display::PROMPT_HEAD, display::PROMPT_GUTTER),
            "▸ thought #1 · step one … +1".to_string(),
            "there are 3 files".to_string(),
        ]);
        assert!(out.iter().all(|(s, _)| *s == Stream::Out), "no stderr in a clean run");
    }

    #[test]
    fn trailing_partial_line_lands_on_done() {
        // production never had a close() call — drive feed exactly like the
        // wire does; the tail without a newline must still print
        let mut p = Plain::new();
        let mut out = vec![];
        for m in [
            Msg::TaskBegin("t".into()),
            Msg::Text("hello".into()),
            Msg::Text(" world".into()), // no trailing newline anywhere
            Msg::Done,
            Msg::TaskEnd,
        ] {
            out.extend(p.feed(m));
        }
        let texts: Vec<String> = out.iter().map(|(_, l)| l.clone()).collect();
        assert!(texts.iter().any(|l| l == "hello world"), "final line without \\n must still print: {texts:?}");
    }

    #[test]
    fn trailing_partial_lands_on_taskend_too() {
        let out = run(vec![
            Msg::TaskBegin("t".into()),
            Msg::Text("tail".into()),
            Msg::TaskEnd,
        ]);
        assert!(out.iter().any(|(_, l)| l == "tail"), "{out:?}");
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
        assert_eq!(out.last().unwrap().1, "│ … +10 more lines");
    }

    #[test]
    fn interrupted_thinking_still_folds_on_done() {
        let out = run(vec![Msg::Think("half a thought".into()), Msg::Done]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, "▸ thought #1 · half a thought");
    }

    #[test]
    fn notes_go_to_stderr() {
        let out = run(vec![Msg::Note { sev: Sev::Warn, text: " trimmed history ".into() }]);
        assert_eq!(out, vec![(Stream::Err, " trimmed history ".to_string())]);
    }
}
