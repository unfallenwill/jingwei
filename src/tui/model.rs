//! TUI state: the model half of model/update/view. Everything here is plain
//! data — no instants, no terminal, no channels — so every rule the UI obeys
//! (folding, scrolling, editing, the status bar) is a fact about this data
//! that a test can pin.

use crate::Usage;
use std::time::Duration;

/// The one heartbeat interval; the spinner advances one frame per tick and
/// the elapsed clock accumulates ticks. Keeping time *in the state* (not
/// read from the wall clock at render time) is what makes views
/// deterministic and testable.
pub const TICK: Duration = Duration::from_millis(120);

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// The two modes. `Input` is the REPL (editing a line); `Browse` is the
/// expanded review mode: every fold is open and the transcript scrolls.
/// Ctrl-O toggles between them; expansion itself is a ratchet — folds that
/// have been expanded never fold again.
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Mode {
    Input,
    Browse,
}

/// One transcript row. Rows reference folds by index; the fold's
/// `expanded` flag decides how the row renders.
#[derive(Clone, Debug)]
pub enum Row {
    /// The echoed task this run is working on.
    Task(String),
    /// A finished line of answer text (plain).
    Line(String),
    /// A folded reasoning block (`thought #k`).
    Thought(usize),
    /// A tool call: header line plus a foldable output tail.
    Tool { head: String, id: usize },
    /// A warning or error.
    Note(crate::display::Sev, String),
}

/// A foldable block. `expanded` only ever flips false → true: Ctrl-O opens
/// everything, and what has been open stays open.
#[derive(Clone, Debug)]
pub struct Fold {
    /// 1-based number among folds of the same kind (thought #3).
    pub n: usize,
    /// "thought" | "tool" — decides the marker's wording.
    pub kind: &'static str,
    /// The full body, kept verbatim.
    pub body: String,
    pub expanded: bool,
}

impl Fold {
    pub fn lines(&self) -> usize {
        self.body.lines().count().max(1)
    }
}

/// The line editor: text plus a byte-cursor kept on grapheme boundaries.
/// History navigation swaps the draft out and back.
#[derive(Clone, Debug, Default)]
pub struct Input {
    pub text: String,
    pub cursor: usize,
    pub history: Vec<String>,
    /// The in-progress draft parked while walking history.
    pub draft: Option<String>,
    pub hist_pos: usize,
}

/// The status bar's state. `elapsed` accumulates ticks, not wall time.
#[derive(Clone, Debug)]
pub struct Status {
    pub task: Option<Task>,
    pub turn: Usage,
    pub total: Usage,
    pub spin: usize,
}

#[derive(Clone, Debug)]
pub struct Task {
    pub text: String,
    pub elapsed: Duration,
}

/// The application state — the single source of truth the view renders.
#[derive(Clone, Debug)]
pub struct App {
    pub mode: Mode,
    pub rows: Vec<Row>,
    /// Partial streamed line, shown live as the last transcript row.
    pub partial: String,
    /// Reasoning arriving now, folded away until `ThinkEnd`.
    pub think_buf: String,
    pub folds: Vec<Fold>,
    pub input: Input,
    pub status: Status,
    /// Ctrl-C was sent to the running agent; a second one exits.
    pub cancel_sent: bool,
    pub quit: bool,
    /// Rows scrolled up from the bottom (0 = following the tail).
    pub scroll: u16,
    /// Whether the transcript follows new content (cleared by scrolling up,
    /// restored by reaching the bottom or submitting a task).
    pub follow: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            mode: Mode::Input,
            rows: vec![],
            partial: String::new(),
            think_buf: String::new(),
            folds: vec![],
            input: Input::default(),
            status: Status { task: None, turn: Usage::default(), total: Usage::default(), spin: 0 },
            cancel_sent: false,
            quit: false,
            scroll: 0,
            follow: true,
        }
    }

    pub fn working(&self) -> bool {
        self.status.task.is_some()
    }

    /// Land finished lines of streamed text; the partial tail stays live.
    pub fn push_text(&mut self, t: &str) {
        self.partial.push_str(t);
        while let Some(i) = self.partial.find('\n') {
            let line: String = self.partial.drain(..=i).collect();
            self.rows.push(Row::Line(line.trim_end_matches('\n').to_string()));
        }
    }

    /// Flush the partial line into the transcript.
    pub fn flush_partial(&mut self) {
        if !self.partial.is_empty() {
            let line = std::mem::take(&mut self.partial);
            self.rows.push(Row::Line(line));
        }
    }

    /// Fold the reasoning accumulated so far into a marker row. Interrupted
    /// thinking folds too — nothing reasoned is lost.
    pub fn fold_thought(&mut self) {
        if self.think_buf.is_empty() {
            return;
        }
        self.flush_partial();
        let body = std::mem::take(&mut self.think_buf);
        let n = self.folds.iter().filter(|f| f.kind == "thought").count() + 1;
        let id = self.folds.len();
        self.folds.push(Fold { n, kind: "thought", body, expanded: false });
        self.rows.push(Row::Thought(id));
    }

    /// Ctrl-O's ratchet: open every fold that is still closed.
    pub fn expand_all(&mut self) {
        for f in &mut self.folds {
            f.expanded = true;
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn closed_folds(&self) -> usize {
        self.folds.iter().filter(|f| !f.expanded).count()
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
