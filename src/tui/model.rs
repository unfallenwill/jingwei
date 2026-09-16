//! TUI state: the model half of model/update/view. Everything here is plain
//! data — no instants, no terminal, no channels — so every rule the UI obeys
//! (folding, scrolling, editing, the status bar) is a fact about this data
//! that a test can pin.

use crate::display::{Sev, Usage};
use std::time::Duration;

/// The one heartbeat cadence; the spinner advances one frame per tick and
/// the elapsed clock accumulates the *measured* deltas each tick carries
/// (see [`crate::tui::update::Ev::Tick`]). Keeping time *in the state* (not
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

/// One transcript row. A row that folds *owns* its fold — no index into a
/// side table, so a fold can never be lost, duplicated, or rendered with
/// the wrong kind. What kind of fold a row carries *is* the variant:
/// `Thought` holds reasoning, `Tool` a head line plus an output tail —
/// there is no second copy of that fact inside `Fold` to disagree with it.
#[derive(Clone, Debug)]
pub enum Row {
    /// The echoed task this run is working on.
    Task(String),
    /// A finished line of answer text (plain).
    Line(String),
    /// A folded reasoning block (`thought #k`).
    Thought(Fold),
    /// A tool call: header line plus a foldable output tail.
    Tool { head: String, fold: Fold },
    /// A warning or error.
    Note(Sev, String),
    /// A blank separator between runs — layout named as what it is, not a
    /// smuggled `Line("")`.
    Sep,
}

/// A foldable block. `expanded` only ever flips false → true: Ctrl-O opens
/// everything, and what has been open stays open. The kind of block this is
/// lives in the [`Row`] variant carrying it, never here.
#[derive(Clone, Debug)]
pub struct Fold {
    /// 1-based number among folds of the same kind (thought #3).
    pub n: usize,
    /// The full body, kept verbatim.
    pub body: String,
    pub expanded: bool,
    /// How long the model spent inside this block — measured from the
    /// ticks that elapsed between the first delta and the fold. `None`
    /// when no task was running to keep time.
    pub duration: Option<Duration>,
}

impl Fold {
    pub fn lines(&self) -> usize {
        self.body.lines().count().max(1)
    }
}

/// The line editor: text plus a byte-cursor kept on grapheme boundaries.
/// The text may contain '\n' — Ctrl-J and Shift-Enter break the line, and
/// the view renders one row per line with the caret's row tracked.
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

impl Input {
    /// The cursor's byte offset, floored to a char boundary. Every editor
    /// op goes through here; a stray non-boundary index is a bug in one of
    /// them — caught loudly in debug, healed forward in release.
    pub fn caret(&self) -> usize {
        let clamped = self.cursor.min(self.text.len());
        debug_assert!(
            self.text.is_char_boundary(clamped),
            "cursor {} left the grapheme grid — an editor op is broken",
            self.cursor
        );
        floor_boundary(&self.text, clamped)
    }
}

/// The healing itself, separate so it is testable: a non-boundary index
/// moves forward to the next boundary.
fn floor_boundary(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// The status bar's state. `elapsed` accumulates measured tick deltas.
#[derive(Clone, Debug)]
pub struct Status {
    /// Mirrors the last `Row::Task` — both are written together, in
    /// `TaskBegin`, and nowhere else.
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

/// Where the review overlay looks. One enum for what used to be a
/// `scroll: u16` plus a `follow: bool` that had to agree by hand:
/// `Tail` *is* "following", `Up(n)` *is* "n rows off the bottom", and the
/// impossible combinations no longer exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scroll {
    /// Pinned to the bottom: new content arrives into view.
    Tail,
    /// Lifted `n` rendered rows off the bottom; clamped against the
    /// content at render time, so `Up(u16::MAX)` means "top".
    Up(u16),
}

impl Scroll {
    pub const TOP: Scroll = Scroll::Up(u16::MAX);

    /// Named for what the update tests assert; the enum makes it a fact
    /// rather than a flag to maintain.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_tail(&self) -> bool {
        matches!(self, Self::Tail)
    }

    /// Rendered-rows offset from the bottom (0 at the tail).
    pub fn offset(&self) -> usize {
        match self {
            Self::Tail => 0,
            Self::Up(n) => *n as usize,
        }
    }

    /// Lift `n` rows off the bottom; any lift stops following.
    pub fn lift(&self, n: u16) -> Self {
        Self::Up(self.offset().saturating_add(n as usize).min(u16::MAX as usize) as u16)
    }

    /// Sink `n` rows toward the bottom; reaching it resumes following.
    pub fn sink(&self, n: u16) -> Self {
        match self.offset().saturating_sub(n as usize) {
            0 => Self::Tail,
            n => Self::Up(n as u16),
        }
    }
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
    /// The task clock's reading when the current reasoning block opened —
    /// the anchor `duration` is measured against. Set by the first delta
    /// of a block, taken by the fold that closes it.
    think_start: Option<Duration>,
    pub input: Input,
    pub status: Status,
    /// Ctrl-C was sent to the running agent; a second one exits.
    pub cancel_sent: bool,
    pub quit: bool,
    /// Where the review overlay looks (see [`Scroll`]). The REPL itself
    /// never scrolls — its transcript lives in the terminal's scrollback.
    pub scroll: Scroll,
}

impl App {
    pub fn new() -> Self {
        Self {
            mode: Mode::Input,
            rows: vec![],
            partial: String::new(),
            think_buf: String::new(),
            think_start: None,
            input: Input::default(),
            status: Status { task: None, turn: Usage::default(), total: Usage::default(), spin: 0 },
            cancel_sent: false,
            quit: false,
            scroll: Scroll::Tail,
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

    /// Reasoning delta in; the first delta of a block anchors the clock.
    pub fn push_think(&mut self, t: &str) {
        if self.think_buf.is_empty() {
            self.think_start = self.status.task.as_ref().map(|t| t.elapsed);
        }
        self.think_buf.push_str(t);
    }

    /// How long the in-flight reasoning block has been running, against
    /// the same task clock [`Fold::duration`] is measured with.
    pub fn thinking_for(&self) -> Option<Duration> {
        match (self.think_start, self.status.task.as_ref()) {
            (Some(start), Some(t)) if t.elapsed >= start => Some(t.elapsed - start),
            _ => None,
        }
    }

    /// 1-based number the next fold of one kind will carry — counted over
    /// the rows themselves, so it can never drift from what was rendered.
    /// Also the number the *live* reasoning block previews.
    pub fn next_thought_n(&self) -> usize {
        self.rows.iter().filter(|r| matches!(r, Row::Thought(_))).count() + 1
    }

    fn next_tool_n(&self) -> usize {
        self.rows.iter().filter(|r| matches!(r, Row::Tool { .. })).count() + 1
    }

    /// Fold the reasoning accumulated so far into a marker row. Interrupted
    /// thinking folds too — nothing reasoned is lost.
    pub fn fold_thought(&mut self) {
        if self.think_buf.is_empty() {
            return;
        }
        self.flush_partial();
        let body = std::mem::take(&mut self.think_buf);
        let n = self.next_thought_n();
        let duration = match (self.think_start.take(), self.status.task.as_ref()) {
            (Some(start), Some(t)) if t.elapsed >= start => Some(t.elapsed - start),
            _ => None,
        };
        self.rows.push(Row::Thought(Fold { n, body, expanded: false, duration }));
    }

    /// Fold a tool call's output into a header row plus a tail.
    pub fn fold_tool(&mut self, head: String, output: String) {
        self.flush_partial();
        self.fold_thought();
        let n = self.next_tool_n();
        self.rows.push(Row::Tool { head, fold: Fold { n, body: output, expanded: false, duration: None } });
    }

    /// Ctrl-O's ratchet: open every fold that is still closed.
    pub fn expand_all(&mut self) {
        for f in self.rows.iter_mut().filter_map(|r| match r {
            Row::Thought(f) => Some(f),
            Row::Tool { fold, .. } => Some(fold),
            _ => None,
        }) {
            f.expanded = true;
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn closed_folds(&self) -> usize {
        self.rows
            .iter()
            .filter_map(|r| match r {
                Row::Thought(f) | Row::Tool { fold: f, .. } => Some(f),
                _ => None,
            })
            .filter(|f| !f.expanded)
            .count()
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_cannot_encode_the_impossible() {
        assert!(Scroll::Tail.is_tail());
        assert_eq!(Scroll::Tail.offset(), 0);
        // lifting leaves the tail; sinking back resumes it
        let up = Scroll::Tail.lift(10);
        assert_eq!(up, Scroll::Up(10));
        assert!(!up.is_tail());
        assert_eq!(up.sink(3), Scroll::Up(7));
        assert_eq!(up.sink(10), Scroll::Tail);
        // sinking at the tail stays the tail; lift saturates
        assert_eq!(Scroll::Tail.sink(5), Scroll::Tail);
        assert_eq!(Scroll::Up(u16::MAX).lift(1), Scroll::Up(u16::MAX));
    }

    #[test]
    fn caret_floors_to_boundaries_and_heals_forward() {
        let mut input = Input { text: "精卫".into(), cursor: 0, ..Default::default() };
        // the healing itself: a mid-char byte moves to the next boundary
        assert_eq!(floor_boundary(&input.text, 1), 3);
        assert_eq!(floor_boundary(&input.text, 99), "精卫".len(), "clamped to the end");
        // a healthy cursor passes through untouched
        input.cursor = 3;
        assert_eq!(input.caret(), 3);
        input.cursor = 6;
        assert_eq!(input.caret(), 6);
    }
}
