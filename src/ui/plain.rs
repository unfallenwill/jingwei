//! The plain frontend: a pure fold over [`Msg`]s into a line-oriented log,
//! plus the dumb line reader that drives it (pipes, one-shot runs). The
//! TUI's sibling — same port, same vocabulary (`crate::ui`), none of
//! the terminal machinery.

use crate::ui::{self, AgentId, AgentMsg, Msg, Sev, Show, TurnState};
use crate::mcp::Hub;
use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};

/// Where a rendered plain line goes.
#[derive(PartialEq, Eq, Debug)]
pub enum Stream {
    Out,
    Err,
}

/// How many body lines the plain log echoes per tool call. The log has no
/// unfolding, so its echo must be self-sufficient — hence 20 lines, where
/// the TUI's collapsed fold shows only [`crate::ui::tui::view::TOOL_TAIL_SHOWN`]
/// because Ctrl-O exists to unfold the rest.
const PLAIN_TOOL_LINES: usize = 20;

/// The plain frontend as a *pure* fold: `feed` consumes one message and
/// returns the finished lines it produced (stdout or stderr). Partial
/// streamed text stays buffered until a newline closes it or the turn ends
/// (`Done`/`TaskEnd` land the tail even without a trailing newline) —
/// tests pin the exact output.
///
/// The fold is per agent: the conversation has a stream of its own, and
/// every background agent has one of its own. A worker's lines carry the
/// agent's tag (`[a1] `); the conversation's carry nothing.
#[derive(Default)]
pub struct Plain {
    /// The conversation's own stream.
    main: PlainStream,
    /// The background agents' streams, by id.
    workers: BTreeMap<AgentId, PlainStream>,
}

/// One stream's state — the conversation's, or one background agent's.
/// The fold is the same for both; only the tag on the lines differs.
#[derive(Default)]
struct PlainStream {
    /// Streamed text, not yet a line.
    text: String,
    /// Reasoning, folded when the block closes.
    think: String,
    /// How many reasoning blocks have been folded.
    thoughts: usize,
    /// The label the fleet gave it.
    label: String,
    /// How many tools it has called.
    tools: usize,
    /// Its state.
    state: TurnState,
    /// Its report.
    report: String,
}

impl PlainStream {
    /// Land complete lines of streamed text; the partial tail stays buffered.
    fn flush(&mut self, tag: &str, lines: &mut Vec<(Stream, String)>) {
        while let Some(i) = self.text.find('\n') {
            let line: String = self.text.drain(..=i).collect();
            lines.push((Stream::Out, format!("{tag}{}", line.trim_end_matches('\n'))));
        }
    }

    /// Close the streamed text: complete lines, then the unterminated
    /// tail as one last line. The wire's final chunk owes us no newline.
    fn end_of_text(&mut self, tag: &str, lines: &mut Vec<(Stream, String)>) {
        self.flush(tag, lines);
        if !self.text.is_empty() {
            let rest = std::mem::take(&mut self.text);
            lines.push((Stream::Out, format!("{tag}{rest}")));
        }
    }

    /// A tool call: the head — tagged, for a worker — and the tail on
    /// the shared gutter. The gutter is the fold's shape, not the
    /// speaker's: the head directly above it already carries the tag.
    fn tool(&mut self, tag: &str, name: &str, summary: &str, output: &str, lines: &mut Vec<(Stream, String)>) {
        self.end_of_text(tag, lines);
        lines.push((Stream::Out, format!("{tag}[{name}] {summary}")));
        let all: Vec<&str> = output.lines().collect();
        let shown = all.len().min(PLAIN_TOOL_LINES);
        for l in &all[..shown] {
            lines.push((Stream::Out, format!("│ {l}")));
        }
        if all.len() > shown {
            lines.push((Stream::Out, format!("{}… +{} more lines", ui::FOLD_GUTTER, all.len() - shown)));
        }
    }

    /// Fold accumulated reasoning into a marker line; interrupted thinking
    /// still folds, so nothing reasoned is lost.
    fn fold_think(&mut self, tag: &str, lines: &mut Vec<(Stream, String)>) {
        if self.think.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.think);
        self.thoughts += 1;
        let mut all = text.lines();
        let first = all.next().unwrap_or("");
        lines.push((Stream::Out, format!("{tag}{}", ui::thought_folded(self.thoughts, first, all.count()))));
    }

    /// A turn ended: land the open text line (with or without its newline)
    /// and fold whatever reasoning was still arriving.
    fn end_of_turn(&mut self, tag: &str, lines: &mut Vec<(Stream, String)>) {
        self.end_of_text(tag, lines);
        self.fold_think(tag, lines);
    }
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
                lines.push((Stream::Out, format!("{}{}{t}", ui::PROMPT_HEAD, ui::PROMPT_GUTTER)))
            }
            Msg::TaskEnd => self.main.end_of_turn("", &mut lines),
            Msg::Text(t) => {
                self.main.text.push_str(&t);
                self.main.flush("", &mut lines);
            }
            Msg::Think(t) => self.main.think.push_str(&t),
            Msg::ThinkEnd => self.main.fold_think("", &mut lines),
            Msg::Tool { name, summary, output } => {
                self.main.tool("", &name, &summary, &output, &mut lines);
            }
            // both severities ride stderr: notes are asides to the answer,
            // and the answer itself is stdout
            Msg::Note { text, .. } => lines.push((Stream::Err, text)),
            // usage lives in the TUI's status bar; the plain log has none
            Msg::Usage(_) | Msg::OutTokens(_) => {}
            Msg::Done => self.main.end_of_turn("", &mut lines),
            Msg::FromAgent { id, event } => self.agent_event(id, *event, &mut lines),
        }
        lines
    }

    /// A background agent's event. Its lines carry the agent's tag; the
    /// conversation's carry nothing.
    ///
    /// The plain log *streams* a worker's prose — a line lands when its
    /// newline arrives, exactly as the conversation's does. (The TUI
    /// defers a worker's prose to the agent's end, so two speakers'
    /// prose never interleaves in its transcript.) The log has no live
    /// pane to interleave with, and a log that showed nothing until the
    /// worker ended would make a working agent look like an idle one.
    fn agent_event(&mut self, id: AgentId, ev: AgentMsg, lines: &mut Vec<(Stream, String)>) {
        let tag = ui::agent_tag(id);
        let stream = self.workers.entry(id).or_default();
        match ev {
            // the start names the agent and nothing else
            AgentMsg::Started { label } => {
                stream.label = label;
                stream.state = TurnState::InProgress;
            }
            AgentMsg::Text(t) => {
                stream.text.push_str(&t);
                stream.flush(&tag, lines);
            }
            AgentMsg::Think(t) => stream.think.push_str(&t),
            AgentMsg::ThinkEnd => stream.fold_think(&tag, lines),
            AgentMsg::Tool { name, summary, output } => {
                stream.tools += 1;
                stream.tool(&tag, &name, &summary, &output, lines);
            }
            // a note is an aside to the answer, and rides stderr as the
            // conversation's own notes do
            AgentMsg::Note { text, .. } => lines.push((Stream::Err, format!("{tag}{text}"))),
            AgentMsg::Finished { state, report } => {
                stream.state = state;
                stream.report = report;
                // the prose tail lands, then the reasoning folds
                stream.end_of_turn(&tag, lines);
                // the completion line: the tool count, but no elapsed
                // time — the plain log has no clock and does not fake one
                lines.push((Stream::Out, ui::agent_summary(id, stream.state, &stream.label, None, stream.tools)));
                // the report's lines, tagged
                for line in stream.report.lines() {
                    lines.push((Stream::Out, format!("{tag}{line}")));
                }
            }
        }
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
    mut ctx: crate::core::context::Context,
    mut convo: crate::core::session::Convo,
    banners: Vec<String>,
    hub: Hub,
    router: std::sync::Arc<crate::core::ports::ToolRouter>,
    fleet: std::sync::Arc<crate::workflow::exec::Fleet>,
) -> crate::Result<()> {
    // The sink is shared: the run shows through it, and so does every
    // worker the fleet spawns under it.
    let sink: std::sync::Arc<dyn Show> = std::sync::Arc::new(PlainSink::new());
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
            print!("{}{} ", ui::PROMPT_HEAD, ui::PROMPT_GUTTER);
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
        if ui::is_exit(&line) {
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
        // `/agents` is the human's own surface on the fleet: the workers
        // and their reports, without going through the model. It is
        // handled before the line becomes a task, exactly like `/mcp`.
        if let Some(rest) = line.strip_prefix("/agents") {
            // the verb stands alone: `/agentsfoo` is a task, not a command
            if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                handle_agents_command(rest.trim(), &fleet, &sink);
                continue;
            }
        }
        // The run boundary: the REPL hands the line to the run, which
        // owns the begin, the saves, the end — and the note a failure
        // earns, said inside the run's bracket so it belongs to the
        // task it is about.
        let scope = crate::cancel::CancelScope::new();
        let _ = crate::workflow::run_task(&line, cfg, &mut ctx, &mut convo, &router, &scope, &fleet, &sink, &ui::error_note).await;
    }
    Ok(())
}

/// The human's `/agents` verb: the workers and their reports, printed
/// through the same port the agent uses.
///
/// One implementation, two frontends: the TUI's `/agents` reaches the
/// same handler, so the two say the same thing for the same input.
pub(crate) fn handle_agents_command(rest: &str, fleet: &crate::workflow::exec::Fleet, sink: &dyn Show) {
    for m in crate::agent::cmd::handle(rest, fleet).unwrap_or_default() {
        sink.show(m);
    }
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
        "enable" | "disable" | "reconnect" | "disconnect" => {
            named_server_op(sub, name, hub, sink).await;
        }
        other => sink.show(Msg::Note {
            sev: Sev::Warn,
            text: format!(" mcp: unknown subcommand: {other:?} · try /mcp list "),
        }),
    }
}

/// The four verbs that name a server: one missing-name note, one
/// done/error shape — the verb picks the hub call, everything else is
/// shared. The TUI rides this same path through its own `/mcp` spawn.
async fn named_server_op(verb: &str, name: Option<&str>, hub: &Hub, sink: &dyn Show) {
    let Some(name) = name else {
        sink.show(Msg::Note { sev: Sev::Err, text: " mcp: missing server name — try /mcp list ".into() });
        return;
    };
    let out = match verb {
        "enable" => hub.enable(name).await,
        "disable" => hub.disable(name).await,
        "reconnect" => hub.reconnect(name).await,
        _ => hub.disconnect(name).await,
    };
    match out {
        Ok(()) => sink.show(Msg::Banner(format!("mcp: {verb} {name}: done"))),
        Err(why) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" mcp: {verb} {name}: {why} ") }),
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
            format!("{}{}count files", ui::PROMPT_HEAD, ui::PROMPT_GUTTER),
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
    fn the_conversation_is_unchanged() {
        // a mixed sequence: the conversation's own line is exactly what
        // it was, and the worker's lines carry the worker's tag
        let id = ui::AgentId::next();
        let out = run(vec![
            Msg::TaskBegin("count files".into()),
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Text("a worker spoke\n".into())) },
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Tool {
                name: "bash".into(), summary: "$ ls".into(), output: "a\n".into() }) },
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Note {
                sev: Sev::Err, text: " it failed ".into() }) },
            Msg::Text("the conversation's answer".into()),
            Msg::Done,
            Msg::TaskEnd,
        ]);
        // the routing is part of the contract, not an accident of the
        // fold: the worker's note is the only line that rides stderr,
        // and the conversation's own answer rides stdout untagged
        assert_eq!(out, vec![
            (Stream::Out, format!("{}{}count files", ui::PROMPT_HEAD, ui::PROMPT_GUTTER)),
            (Stream::Out, format!("[{id}] a worker spoke")),
            (Stream::Out, format!("[{id}] [bash] $ ls")),
            (Stream::Out, "│ a".to_string()),
            (Stream::Err, format!("[{id}]  it failed ")),
            (Stream::Out, "the conversation's answer".to_string()),
        ], "the conversation's line is unchanged, the worker's lines are tagged: {out:?}");
    }

    #[test]
    fn a_workers_lines_carry_its_tag() {
        let id = ui::AgentId::next();
        let out = run(vec![
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Started { label: "build".into() }) },
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Text("the worker's prose\n".into())) },
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Tool {
                name: "bash".into(), summary: "$ ls".into(), output: "a\n".into() }) },
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Finished {
                state: ui::TurnState::Completed, report: "found three files".into() }) },
        ]);
        let texts: Vec<String> = out.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(texts, vec![
            format!("[{id}] the worker's prose"),
            format!("[{id}] [bash] $ ls"),
            "│ a".to_string(),
            // the tool count, but no elapsed time: the plain log has no clock
            format!("[{id}] ✓ build · 1 tool"),
            format!("[{id}] found three files"),
        ], "{texts:?}");
    }

    #[test]
    fn a_workers_tool_tail_keeps_the_gutter() {
        // the tag is on the head; the tail hangs from the fold's gutter
        let id = ui::AgentId::next();
        let out = run(vec![Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Tool {
            name: "bash".into(), summary: "$ ls".into(), output: "a\nb\n".into() }) }]);
        let texts: Vec<String> = out.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(texts, vec![
            format!("[{id}] [bash] $ ls"),
            "│ a".to_string(),
            "│ b".to_string(),
        ], "{texts:?}");
    }

    #[test]
    fn two_workers_do_not_share_a_buffer() {
        // a1's half-line and a2's half-line never merge: each stream
        // keeps its own text
        let a = ui::AgentId::next();
        let b = ui::AgentId::next();
        let out = run(vec![
            Msg::FromAgent { id: a, event: Box::new(ui::AgentMsg::Text("hello".into())) },
            Msg::FromAgent { id: b, event: Box::new(ui::AgentMsg::Text("world\n".into())) },
            Msg::FromAgent { id: a, event: Box::new(ui::AgentMsg::Text(" there\n".into())) },
        ]);
        let texts: Vec<String> = out.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(texts, vec![
            format!("[{b}] world"),
            format!("[{a}] hello there"),
        ], "two streams, two buffers: {texts:?}");
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
            Msg::Usage(crate::ui::Usage { input: 1, output: 2, cache_read: 0, cache_write: 0 }),
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

    use crate::test_util::Recorder;

    /// The bare `/mcp` on an empty hub surfaces the "no MCP servers"
    /// message — the only thing the report function returns.
    #[tokio::test]
    async fn mcp_bare_lists_the_default_message_when_nothing_is_configured() {
        let hub = crate::mcp::Hub::empty();
        let sink = Recorder::default();
        handle_mcp_command("", &hub, &sink).await;
        // The recorder hands back a copy, so no lock is held across the
        // await — `hub.shutdown()` cannot deadlock behind one.
        let msgs = sink.msgs();
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
        let sink = Recorder::default();
        handle_mcp_command("list", &hub, &sink).await;
        // empty hub → empty list → no banners
        assert!(sink.msgs().is_empty());
        hub.shutdown().await;
    }

    /// An unknown subcommand is a warning, not a silent failure.
    #[tokio::test]
    async fn mcp_unknown_subcommand_emits_a_warning() {
        let hub = crate::mcp::Hub::empty();
        let sink = Recorder::default();
        handle_mcp_command("frobnicate", &hub, &sink).await;
        let msgs = sink.msgs();
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
        let sink = Recorder::default();
        handle_mcp_command("enable", &hub, &sink).await;
        let msgs = sink.msgs();
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

    /// The human's own surface on the fleet: the plain REPL prints the
    /// same table the TUI prints, because both go through one handler.
    #[test]
    fn the_plain_repl_prints_the_agents_table() {
        let rec = crate::test_util::Recorder::default();
        let fleet = crate::workflow::exec::Fleet::new(crate::workflow::exec::AgentCfg::default());
        handle_agents_command("", &fleet, &rec);
        // the same port the agent uses, and the plain frontend's fold on
        // top of it: the human sees one line
        let out = run(rec.msgs());
        let texts: Vec<String> = out.iter().map(|(_, l)| l.clone()).collect();
        assert_eq!(texts, vec!["no background agents".to_string()]);
    }

    /// A worker's reasoning folds the same way the conversation's does —
    /// into one marker line — but the worker's tag rides in front of it.
    #[test]
    fn a_workers_thinking_folds_into_a_tagged_marker() {
        let id = ui::AgentId::next();
        let out = run(vec![
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::Think("worker reasoning\n".into())) },
            Msg::FromAgent { id, event: Box::new(ui::AgentMsg::ThinkEnd) },
        ]);
        assert_eq!(
            out,
            vec![(Stream::Out, format!("[{id}] ▸ thought #1 · worker reasoning"))],
            "{out:?}",
        );
    }

    /// The sink is the fold plus the write: it holds nothing of its own
    /// beyond the fold, and a fresh one has folded nothing yet.
    #[test]
    fn a_sink_folds_what_it_is_shown() {
        let sink = PlainSink::new();
        assert_eq!(sink.0.lock().unwrap().main.thoughts, 0, "a fresh sink has folded nothing");
        sink.show(Msg::Think("a thought".into()));
        sink.show(Msg::ThinkEnd);
        assert_eq!(sink.0.lock().unwrap().main.thoughts, 1, "the sink folded the thought");
        // `default` is the same sink, and it is fresh too
        let other = PlainSink::default();
        assert_eq!(other.0.lock().unwrap().main.thoughts, 0, "the default sink is fresh");
        // a note is the sink's other stream: it rides stderr, not stdout
        sink.show(Msg::Note { sev: Sev::Warn, text: " a note ".into() });
        assert_eq!(sink.0.lock().unwrap().main.thoughts, 1, "a note is not a thought");
    }

    /// A hub with one server whose configuration is a failure: the hub
    /// knows the server, so `/mcp list` has a row to show and every verb
    /// has a server to act on.
    async fn a_hub_with_a_broken_server(name: &str) -> crate::mcp::Hub {
        let mut entry = crate::mcp::stub::Stub::new().entry(&[]);
        entry.name = name.to_string();
        entry.config = Err("no such command".into());
        crate::mcp::Hub::of_entries(vec![entry]).await
    }

    /// A server that cannot start is still a server the user can see:
    /// `/mcp list` gives it a row and says why it is not there.
    #[tokio::test]
    async fn mcp_list_shows_a_server_that_cannot_start() {
        let hub = a_hub_with_a_broken_server("alpha").await;
        let sink = Recorder::default();
        handle_mcp_command("list", &hub, &sink).await;
        let msgs = sink.msgs();
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        match &msgs[0] {
            Msg::Banner(s) => {
                assert!(s.contains("alpha"), "{s}");
                assert!(s.contains("failed"), "{s}");
                assert!(s.contains("no such command"), "{s}");
            }
            other => panic!("expected a banner, got {other:?}"),
        }
        hub.shutdown().await;
    }

    /// Each of the four verbs names the server it acted on and says what
    /// came of it: the hub's own words, in the note.
    #[tokio::test]
    async fn the_four_server_verbs_say_what_the_hub_said() {
        let hub = a_hub_with_a_broken_server("alpha").await;
        for verb in ["enable", "disable", "reconnect", "disconnect"] {
            let sink = Recorder::default();
            handle_mcp_command(&format!("{verb} alpha"), &hub, &sink).await;
            let msgs = sink.msgs();
            assert_eq!(msgs.len(), 1, "{verb}: got {msgs:?}");
            match &msgs[0] {
                Msg::Note { sev, text } => {
                    assert_eq!(*sev, Sev::Err, "{verb}");
                    assert!(text.contains(verb), "{verb}: {text}");
                    assert!(text.contains("alpha"), "{verb}: {text}");
                }
                other => panic!("{verb}: expected a note, got {other:?}"),
            }
        }
        hub.shutdown().await;
    }

    /// A server that goes offline on request says the verb is done: the
    /// banner names the server and the verb.
    #[tokio::test]
    async fn a_server_that_goes_offline_says_the_verb_is_done() {
        let stub = crate::mcp::stub::Stub::new();
        let mut entry = stub.entry(&[("STUB_TOOLS", "echo")]);
        entry.name = "alpha".into();
        let hub = crate::mcp::Hub::of_entries(vec![entry]).await;
        let sink = Recorder::default();
        handle_mcp_command("disable alpha", &hub, &sink).await;
        let msgs = sink.msgs();
        assert_eq!(msgs.len(), 1, "got: {msgs:?}");
        match &msgs[0] {
            Msg::Banner(s) => assert_eq!(s, "mcp: disable alpha: done"),
            other => panic!("expected a banner, got {other:?}"),
        }
        hub.shutdown().await;
    }

    /// A server that is out of service is summarised by what it is, not
    /// by a failure: the row says "disabled" or "disconnected".
    #[test]
    fn a_disabled_or_a_disconnected_server_is_summarised_as_such() {
        use crate::mcp::{ServerState, ServerStatus};
        let disabled = format_mcp_status(&ServerStatus {
            name: "alpha".into(),
            state: ServerState::Disabled,
            tools_count: 0,
        });
        assert!(disabled.contains("alpha"), "{disabled}");
        assert!(disabled.contains("disabled"), "{disabled}");
        let gone = format_mcp_status(&ServerStatus {
            name: "alpha".into(),
            state: ServerState::Disconnected,
            tools_count: 0,
        });
        assert!(gone.contains("disconnected"), "{gone}");
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
