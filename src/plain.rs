//! The plain frontend: a pure fold over [`Msg`]s into a line-oriented log,
//! plus the dumb line reader that drives it (pipes, one-shot runs). The
//! TUI's sibling — same port, same vocabulary (`crate::display`), none of
//! the terminal machinery.

use crate::display::{self, Msg, Sev, Show};
use crate::mcp::Hub;
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

/// The plain frontend's sink: fold each message and write the finished lines
/// (stdout for the answer, stderr for notes). The fold is pure ([`Plain`]);
/// the write is the sink's, so the agent coroutine and the shell share one
/// handle (`&self`) — the `Mutex` is that sharing, not logic.
pub struct PlainSink(std::sync::Mutex<Plain>);

impl PlainSink {
    pub fn new() -> Self {
        Self(std::sync::Mutex::new(Plain::new()))
    }
}

impl Default for PlainSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Show for PlainSink {
    fn show(&self, m: Msg) {
        for (stream, line) in self.0.lock().unwrap().feed(m) {
            match stream {
                Stream::Out => println!("{line}"),
                Stream::Err => eprintln!("{line}"),
            }
        }
    }
}

/// Drive the plain REPL: read lines, run the agent, print via the port.
/// Used when stdout is not a terminal (pipes, one-shot runs). The TUI
/// gate is `tui::wanted()` — when stdout is a tty, the TUI takes over.
///
/// No editor here, so no input-history file: recall is a TUI feature; this
/// path keeps the shell's own history (up-arrow) and line editing.
pub async fn plain_repl(
    cfg: &crate::config::Config,
    mut convo: crate::session::Convo,
    banners: Vec<String>,
    hub: Hub,
) -> crate::Result<()> {
    use crate::{agent_turn};
    let sink = PlainSink::new();
    let tty = std::io::stdin().is_terminal();
    sink.show(Msg::Banner("jingwei — 精卫填海，一石一石 · type a task, /exit or Ctrl-D rests, Ctrl-C interrupts".into()));
    sink.show(Msg::Banner(cfg.identity()));
    for b in banners {
        sink.show(Msg::Banner(b));
    }
    // The hub's notes are also a banner: which servers came up, which did
    // not, and what to look at. They land before the prompt so the user
    // knows MCP is wired before their first turn.
    for note in hub.notes().await {
        sink.show(Msg::Banner(note));
    }
    let stdin = std::io::stdin();
    loop {
        if tty {
            // The interactive prompt must match the TaskBegin banner
            // character-for-character — both prefix the same `<head> <gutter> `
            // string. Earlier versions `.trim()`'d the gutter to drop
            // the leading space; the result drifted from the banner and
            // from what `prompt_w()` (TUI-aligned) measures, so a
            // screenshot of the prompt and the task line did not line
            // up. Plain keeps the leading space.
            print!("{}{} ", display::PROMPT_HEAD, display::PROMPT_GUTTER);
            let _ = std::io::stdout().flush();
        }
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            // bad bytes on stdin: warn and carry on
            Err(e) => {
                sink.show(Msg::Note { sev: Sev::Warn, text: format!("warning: input ({e})") });
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
        // `/mcp` is its own little command family: handled before the
        // line becomes a task. Sub-commands (`list`, `enable <n>`, ...)
        // are routed through the hub and printed via the same port the
        // agent uses, so the user sees them the same way.
        if let Some(rest) = line.strip_prefix("/mcp").map(str::trim) {
            handle_mcp_command(rest, &hub, &sink).await;
            continue;
        }
        convo.history.push(crate::user_message(&line));
        if let Err(e) = convo.persist() {
            sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
        }
        // the task is on disk before the first stone moves
        sink.show(Msg::TaskBegin(line));
        let token = crate::cancel::CancelToken::new();
        match agent_turn(cfg, &mut convo.history, &token, &hub, &sink).await {
            Err(crate::Error::Interrupted) => {}
            Err(e) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" error: {e} ") }),
            Ok(()) => {}
        }
        if let Err(e) = convo.persist() {
            sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
        }
        // run boundary: the file never ends mid-run
        sink.show(Msg::TaskEnd);
    }
    Ok(())
}

/// One `/mcp` subcommand. The bare `/mcp` (called as the empty `rest`)
/// shows the long report; `list` shows the per-server row; `enable`,
/// `disable`, `reconnect`, `disconnect` operate on a named server. The
/// routing mirrors the TUI's, so the two frontends say the same thing.
pub(crate) async fn handle_mcp_command(rest: &str, hub: &Hub, sink: &dyn Show) {
    let mut it = rest.split_whitespace();
    let sub = it.next().unwrap_or("");
    let name = it.next();
    match sub {
        "" => {
            // bare /mcp: long report — one line per server, then tools
            for line in hub.report().await {
                sink.show(Msg::Banner(line));
            }
        }
        "list" => {
            // one row per server: name, state, tools_count
            for status in hub.list().await {
                sink.show(Msg::Banner(format_mcp_status(&status)));
            }
        }
        "enable" => {
            let Some(name) = name else {
                sink.show(Msg::Note { sev: Sev::Err, text: " mcp: missing server name — try /mcp list ".into() });
                return;
            };
            match hub.enable(name).await {
                Ok(()) => sink.show(Msg::Banner(format!("mcp: enable {name}: done"))),
                Err(why) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" mcp: enable {name}: {why} ") }),
            }
        }
        "disable" => {
            let Some(name) = name else {
                sink.show(Msg::Note { sev: Sev::Err, text: " mcp: missing server name — try /mcp list ".into() });
                return;
            };
            match hub.disable(name).await {
                Ok(()) => sink.show(Msg::Banner(format!("mcp: disable {name}: done"))),
                Err(why) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" mcp: disable {name}: {why} ") }),
            }
        }
        "reconnect" => {
            let Some(name) = name else {
                sink.show(Msg::Note { sev: Sev::Err, text: " mcp: missing server name — try /mcp list ".into() });
                return;
            };
            match hub.reconnect(name).await {
                Ok(()) => sink.show(Msg::Banner(format!("mcp: reconnect {name}: done"))),
                Err(why) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" mcp: reconnect {name}: {why} ") }),
            }
        }
        "disconnect" => {
            let Some(name) = name else {
                sink.show(Msg::Note { sev: Sev::Err, text: " mcp: missing server name — try /mcp list ".into() });
                return;
            };
            match hub.disconnect(name).await {
                Ok(()) => sink.show(Msg::Banner(format!("mcp: disconnect {name}: done"))),
                Err(why) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" mcp: disconnect {name}: {why} ") }),
            }
        }
        other => sink.show(Msg::Note {
            sev: Sev::Warn,
            text: format!(" mcp: unknown subcommand: {other:?} · try /mcp list "),
        }),
    }
}

/// One row of `/mcp list`: the server name, the state it stands in, and
/// the count of tools being offered. Shared with the TUI so the two
/// frontends print the same shape.
pub(crate) fn format_mcp_status(status: &crate::mcp::ServerStatus) -> String {
    use crate::mcp::ServerState;
    let state = match &status.state {
        ServerState::Ready => "ready".to_string(),
        ServerState::Failed(why) => format!("failed ({why})"),
        ServerState::Disabled => "disabled".into(),
        ServerState::Disconnected => "disconnected".into(),
    };
    format!("{:<20} {:<14} {} tools", status.name, state, status.tools_count)
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

    #[test]
    fn banner_message_is_a_plain_stdout_line() {
        let out = run(vec![Msg::Banner("jingwei".into())]);
        assert_eq!(out, vec![(Stream::Out, "jingwei".into())]);
    }

    #[test]
    fn usage_and_out_tokens_are_swallowed() {
        // the plain log has no status bar — usage messages emit nothing
        let out = run(vec![
            Msg::Usage(crate::display::Usage { input: 1, output: 2, cache_read: 0, cache_write: 0 }),
            Msg::OutTokens(42),
        ]);
        assert!(out.is_empty(), "usage / OutTokens do not produce output: {out:?}");
    }

    #[test]
    fn think_with_no_end_emits_nothing_until_done() {
        // thinking that arrives but never gets ThinkEnd: the partial
        // accumulator stays buffered; a later Done folds it. Both phases
        // run on the same Plain so the buffer carries across.
        let mut p = Plain::new();
        let mut out = vec![];
        out.extend(p.feed(Msg::Think("line one\n".into())));
        assert!(out.is_empty(), "no ThinkEnd yet: {out:?}");
        out.extend(p.feed(Msg::Think("line two".into())));
        out.extend(p.feed(Msg::Done));
        // the fold shows the first line and counts the rest
        assert!(out.iter().any(|(_, l)| l == "▸ thought #1 · line one … +1"),
            "got: {out:?}");
    }

    #[test]
    fn tool_call_with_no_output_lines_just_prints_the_summary() {
        // fewer than PLAIN_TOOL_LINES lines: only the summary header, no
        // fold marker at the end
        let out = run(vec![Msg::Tool {
            name: "read_file".into(),
            summary: "f.txt".into(),
            output: "just one line".into(),
        }]);
        assert_eq!(out.len(), 2, "{out:?}");
        assert_eq!(out[0].1, "[read_file] f.txt");
        assert_eq!(out[1].1, "│ just one line");
    }

    #[test]
    fn consecutive_thinks_concatenate_into_one_marker() {
        // multiple Think deltas before a single ThinkEnd: one marker,
        // not several — a run-packing reader doesn't try to fold half
        // a thought
        let out = run(vec![
            Msg::Think("first\n".into()),
            Msg::Think("second".into()),
            Msg::ThinkEnd,
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, "▸ thought #1 · first … +1");
    }

    #[test]
    fn text_split_across_two_messages_folds_at_the_newline() {
        // "hello" and " world\n" arrive as two deltas: only one line lands
        let out = run(vec![
            Msg::Text("hello".into()),
            Msg::Text(" world\n".into()),
        ]);
        assert_eq!(out, vec![(Stream::Out, "hello world".into())]);
    }

    #[test]
    fn tool_call_after_open_text_ends_the_open_line_first() {
        // text mid-line, then a tool call: the partial line must land
        // before the tool's header, so the tool output reads clean
        let out = run(vec![
            Msg::Text("thinking aloud".into()),
            Msg::Tool { name: "bash".into(), summary: "$ ls".into(), output: "f\n".into() },
        ]);
        assert_eq!(out[0], (Stream::Out, "thinking aloud".into()));
        assert_eq!(out[1].1, "[bash] $ ls");
    }

    // ---- /mcp command handling: same path the REPL takes ------------------

    use crate::display::Show;

    /// Capture sink that records banner / note messages verbatim.
    struct BagSink(std::sync::Mutex<Vec<Msg>>);
    impl Show for BagSink {
        fn show(&self, m: Msg) {
            self.0.lock().unwrap().push(m);
        }
    }

    /// The bare `/mcp` on an empty hub surfaces the "no MCP servers"
    /// message — the only thing the report function returns.
    #[tokio::test]
    async fn mcp_bare_lists_the_default_message_when_nothing_is_configured() {
        let hub = crate::mcp::Hub::empty();
        let sink = BagSink(Default::default());
        handle_mcp_command("", &hub, &sink).await;
        let msgs = sink.0.lock().unwrap();
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        match &msgs[0] {
            Msg::Banner(s) => assert!(s.contains("no MCP servers"), "{s}"),
            other => panic!("expected a banner, got {other:?}"),
        }
        hub.shutdown().await;
    }

    /// `/mcp list` on an empty hub says so without crashing.
    #[tokio::test]
    async fn mcp_list_on_an_empty_hub_emits_nothing_visible() {
        let hub = crate::mcp::Hub::empty();
        let sink = BagSink(Default::default());
        handle_mcp_command("list", &hub, &sink).await;
        // empty hub → empty list → no banners
        assert!(sink.0.lock().unwrap().is_empty());
        hub.shutdown().await;
    }

    /// An unknown subcommand is a warning, not a silent failure.
    #[tokio::test]
    async fn mcp_unknown_subcommand_emits_a_warning() {
        let hub = crate::mcp::Hub::empty();
        let sink = BagSink(Default::default());
        handle_mcp_command("frobnicate", &hub, &sink).await;
        let msgs = sink.0.lock().unwrap();
        assert_eq!(msgs.len(), 1);
        match &msgs[0] {
            Msg::Note { text, .. } => assert!(text.contains("unknown subcommand"), "{text}"),
            other => panic!("expected a note, got {other:?}"),
        }
        hub.shutdown().await;
    }

    /// Subcommands that need a server name say so when none is given.
    #[tokio::test]
    async fn mcp_enable_without_a_name_is_an_error() {
        let hub = crate::mcp::Hub::empty();
        let sink = BagSink(Default::default());
        handle_mcp_command("enable", &hub, &sink).await;
        let msgs = sink.0.lock().unwrap();
        assert_eq!(msgs.len(), 1);
        match &msgs[0] {
            Msg::Note { sev, text } => {
                assert_eq!(*sev, Sev::Err);
                assert!(text.contains("missing server name"), "{text}");
            }
            other => panic!("expected a note, got {other:?}"),
        }
        hub.shutdown().await;
    }

    /// `format_mcp_status` produces one row with the columns a list needs:
    /// the name, the state, the tool count.
    #[test]
    fn format_mcp_status_has_the_columns_a_list_needs() {
        use crate::mcp::{ServerState, ServerStatus};
        let row = format_mcp_status(&ServerStatus {
            name: "files".into(),
            state: ServerState::Ready,
            tools_count: 3,
        });
        assert!(row.contains("files"), "{row}");
        assert!(row.contains("ready"), "{row}");
        assert!(row.contains("3 tools"), "{row}");
        let failed = format_mcp_status(&ServerStatus {
            name: "broken".into(),
            state: ServerState::Failed("nope".into()),
            tools_count: 0,
        });
        assert!(failed.contains("failed"), "{failed}");
        assert!(failed.contains("nope"), "{failed}");
    }
}
