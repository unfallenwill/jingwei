//! The TUI frontend: the only layer allowed to touch a terminal. Everything
//! above this line is pure (model/update/view); this module owns raw mode,
//! the inline pane, the transcript's flush into the terminal's own
//! scrollback, the Ctrl-O overlay, the event stream, and the agent task —
//! and does nothing else. If it has logic, that logic belongs in update.rs
//! (or, for the scroll-and-paint geometry, in [`flush_plan`], which is pure
//! and tested).
//!
//! There is deliberately no alternate screen and no mouse capture while the
//! REPL runs: the transcript is appended above a small framed pane on the
//! primary screen, so the wheel, text selection, and everything that was on
//! screen before jingwei started keep working. (An alternate screen would
//! also make wheel events arrive as ↑/↓ keys on many terminals — which the
//! input row would eat as history recall.) The one full-screen view, the
//! Ctrl-O review overlay, takes the alternate screen only while it is open
//! and restores the primary screen on return.
//!
//! [`Stage`] draws the pane without ratatui's inline viewport on purpose:
//! that viewport answers height changes and resizes with a cursor-position
//! query (ESC[6n) read from stdin — which would race the keyboard reader
//! thread for the same bytes. Instead the stage tracks its own top row and
//! scrolls by printing newlines on the last row (see [`Stage::scroll`]), the
//! way the transcript has always ridden the terminal. The one query of the
//! session happens in [`Stage::new`], before the reader thread exists.
//!
//! A repaint reaches the terminal as *one burst*: everything the frame
//! wants — cursor hide, scrolls, transcript rows, the pane, the caret — is
//! staged into a buffer (see [`Stage::frame`]) and written by a single
//! flush, bracketed by synchronized output (CSI ? 2026 h/l). Terminals
//! render on their own clock, and a frame that arrives in pieces lends
//! them intermediate states to show: a scroll whose freed rows are not
//! painted yet, or a pane cleared of its old rows before the new ones
//! land, each reads as a flash. The brackets ask the terminal to hold
//! presentation until the frame closes (terminals without the mode ignore
//! it, and still take the whole frame as one contiguous write).

pub mod model;
pub mod update;
pub mod view;

use crate::display::{ChannelSink, Msg, Sev, Show as DisplayShow};
use crate::session::Convo;
use crate::{agent_turn, home_dir, user_message, CancelToken, Config, Error};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::style::Print;
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate};
use crossterm::Command;
use model::{App, Mode};
use ratatui::backend::CrosstermBackend;
use ratatui::backend::Backend as _;
use ratatui::buffer::{Buffer, Cell, CellDiffOption, CellWidth};
use ratatui::layout::Rect;
use ratatui::text::{Line, Text};
use ratatui::widgets::Widget;
use std::io::{self, IsTerminal};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex as AsyncMutex;
use update::{update as step, Action, Ev};

type Backend = CrosstermBackend<io::Stdout>;

/// The running agent: its cancellation token and its coroutine. A struct,
/// not a tuple — the shell reads it by name.
struct Agent {
    token: Arc<CancelToken>,
    job: tokio::task::JoinHandle<()>,
}

impl Agent {
    /// Reap the coroutine after it signalled `TaskEnd`. Its panic (dev
    /// builds; release aborts the process) becomes a visible error row
    /// instead of silence.
    async fn reap(self, sink: &dyn DisplayShow) {
        if let Err(e) = self.job.await {
            sink.show(Msg::Note { sev: Sev::Err, text: format!(" agent task panicked: {e} ") });
        }
    }
}

/// The plan for handing `n` finished lines to the scrollback above a pane
/// at row `top`, `height` rows tall, on a screen of `screen` rows: a list
/// of (scroll_up, paint_at, count) steps. Pure arithmetic — the one piece
/// of the stage that can be wrong in interesting ways, and therefore the
/// one piece that is unit-tested.
///
/// Invariants every plan preserves: every row is painted exactly once, in
/// order; after each step the pane still ends on or above the screen's
/// last row; the pane's new top row is the row after the last painted one.
fn flush_plan(top: u16, height: u16, screen: u16, n: usize) -> Vec<(u16, u16, usize)> {
    let mut plan = vec![];
    let mut drawn = top as i32;
    let mut rest = n as i32;
    let vp = height as i32;
    let scr = screen as i32;
    while rest + vp > scr {
        let to_draw = rest.min(scr) as usize;
        let scroll_up = 0.max(drawn + to_draw as i32 - scr) as u16;
        let at = (drawn - scroll_up as i32) as u16;
        plan.push((scroll_up, at, to_draw));
        drawn += to_draw as i32 - scroll_up as i32;
        rest -= to_draw as i32;
    }
    let scroll_up = 0.max(drawn + rest + vp - scr) as u16;
    let at = (drawn - scroll_up as i32) as u16;
    plan.push((scroll_up, at, rest as usize));
    plan
}

/// The bytes that scroll the screen one row at a time: CR+LF pairs, printed
/// with the cursor parked on the last row. Pure, so the choice it pins —
/// line feeds, never `CSI S` — is a testable fact and not just a comment.
/// See [`Stage::scroll`] for why that choice matters.
fn scroll_bytes(n: u16) -> String {
    "\r\n".repeat(n as usize)
}

/// Append a command's escape sequence to a writer as bytes — commands
/// speak `fmt::Write`, the wire speaks `io::Write`; this is the one
/// adapter. (fmt::Write into a String cannot fail, so neither can this.)
fn put_cmd<W: Write>(w: &mut W, cmd: impl Command) -> io::Result<()> {
    let mut s = String::new();
    let _ = cmd.write_ansi(&mut s);
    w.write_all(s.as_bytes())
}

/// Write one composed frame as a single synchronized burst: the body
/// wrapped in Begin/End Synchronized Update (CSI ? 2026 h/l), then one
/// flush. The brackets are the terminal's promise to present nothing
/// until the frame closes — so the scroll inside a frame never shows
/// without the rows that fill the gap it opened (the flash this pins).
/// Terminals without the mode ignore it and still take one contiguous
/// write. An empty frame writes nothing at all.
fn send_frame<W: Write>(w: &mut W, frame: &[u8]) -> io::Result<()> {
    if frame.is_empty() {
        return Ok(());
    }
    put_cmd(w, BeginSynchronizedUpdate)?;
    w.write_all(frame)?;
    put_cmd(w, EndSynchronizedUpdate)?;
    w.flush()
}

/// The exit burst: erase the pane's rows, park the caret where the
/// shell's prompt will land, and roll back the session's modes —
/// composed like any frame (see [`send_frame`]) so the pane dissolves
/// in one render instead of row by row.
fn retire_bytes(top: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::new();
    for y in top..top + height {
        let _ = put_cmd(&mut out, MoveTo(0, y));
        let _ = put_cmd(&mut out, Clear(ClearType::UntilNewLine));
    }
    let _ = put_cmd(&mut out, MoveTo(0, top));
    let _ = put_cmd(&mut out, Show);
    let _ = put_cmd(&mut out, DisableBracketedPaste);
    let _ = put_cmd(&mut out, PopKeyboardEnhancementFlags);
    out
}

/// The bottom of the screen: the pane's rows and where they sit. All
/// geometry is tracked, never queried — see the module docs.
struct Stage {
    /// The frame under construction. Everything one repaint wants on the
    /// wire — Hide, scrolls, transcript rows, height-change clears, pane
    /// rows, the caret — is staged here as raw ANSI until [`Stage::send`]
    /// writes it in one burst. Nothing paints outside a frame, so a frame
    /// never carries state across repaints (a resize's scroll is staged
    /// only when the size changed, which forces the repaint that sends it).
    frame: Vec<u8>,
    /// The pane's top row on the primary screen.
    top: u16,
    /// The pane's current row count (it grows with the draft).
    height: u16,
    width: u16,
    screen: u16,
}

impl Stage {
    /// Anchor the pane at the bottom of the screen — where a REPL's input
    /// belongs, and where the README always said it was. The one
    /// cursor-position query of the session (made before the keyboard reader
    /// exists) is not for the anchor but for safety: it tells us whether the
    /// shell's own last lines would fall inside the pane, and if so we scroll
    /// them up first, so nothing already on the screen is overwritten.
    fn new(height: u16) -> io::Result<Self> {
        let (w, h) = crossterm::terminal::size()?;
        let (_, cursor_row) = crossterm::cursor::position()
            .map_err(|e| io::Error::other(format!("cursor position: {e}")))?;
        let screen = h.max(1);
        let height = height.min(screen);
        let top = screen - height;
        let me = Stage { frame: Vec::new(), top, height, width: w.max(8), screen };
        if cursor_row > top {
            let s = cursor_row - top;
            // before the first frame there is nothing of ours on screen,
            // so no intermediate state to hide — a direct write is fine
            crossterm::execute!(io::stdout(), MoveTo(0, screen - 1), Print(scroll_bytes(s)))?;
        }
        Ok(me)
    }

    /// Adopt a new terminal size and re-anchor for a resize redraw. The
    /// screen has just reflowed our rows (and our pane) on its own — an
    /// event we cannot undo and must not try to reason about, so a resize is
    /// never a scroll: the caller clears the screen and reprints the visible
    /// tail instead (see the event loop). This only records the new geometry:
    /// the pane goes to the bottom (`pane_h` rows), with `count` transcript
    /// rows reprinted above it — nothing that already rode into the
    /// scrollback is among them, and the blank above stays blank.
    fn begin_resize(&mut self, w: u16, h: u16, pane_h: u16, count: u16) {
        self.width = w.max(8);
        self.screen = h.max(1);
        self.height = pane_h.clamp(1, self.screen);
        self.top = self.screen.saturating_sub(self.height).saturating_sub(count);
    }

    /// Grow or shrink the pane. Growth scrolls the screen (old transcript
    /// rides into the scrollback, nothing is lost); shrink clears the rows
    /// the pane gives up. Both are staged — the clears and the rows that
    /// replace them ride the same burst, so the pane never spends a render
    /// with its old bottom erased and its new one not yet drawn.
    fn set_height(&mut self, new_h: u16) -> io::Result<()> {
        let new_h = new_h.min(self.screen).max(1);
        if new_h < self.height {
            for y in (self.top + new_h)..(self.top + self.height) {
                self.put(MoveTo(0, y));
                self.put(Clear(ClearType::UntilNewLine));
            }
        } else if new_h > self.height && self.top + new_h > self.screen {
            let s = self.top + new_h - self.screen;
            self.scroll(s)?;
            self.top = self.screen - new_h;
        }
        self.height = new_h;
        Ok(())
    }

    /// Scroll the whole screen up `n` rows: the top rows land in the
    /// terminal's own scrollback. Done by printing newlines on the last
    /// row — the same path a shell's own output takes — and NOT by
    /// `CSI S` (ScrollUp): that scrolls too, but on xterm.js terminals
    /// (VS Code, Cursor) it *deletes* the top row instead of archiving
    /// it, so the transcript left the screen and was gone.
    fn scroll(&mut self, n: u16) -> io::Result<()> {
        if n > 0 {
            // Raw mode has OPOST off, so "\n" is a bare line feed; each LF
            // on the bottom row scrolls the screen (and archives the top)
            // and the CR keeps the cursor honest for the next one.
            self.put(MoveTo(0, self.screen - 1));
            self.frame.extend_from_slice(scroll_bytes(n).as_bytes());
        }
        Ok(())
    }

    /// Hand the rows rendered since the last flush to the terminal: they
    /// are painted above the pane (overwriting its old position) and, once
    /// the screen is full, carried into the native scrollback. Same
    /// chunked scroll-and-paint arithmetic ratatui's inline viewport uses,
    /// done here so the heights stay ours — the arithmetic lives in
    /// [`flush_plan`], where tests can reach it.
    fn flush(&mut self, lines: &[Line<'static>]) -> io::Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let plan = flush_plan(self.top, self.height, self.screen, lines.len());
        let mut idx = 0usize;
        let mut top = self.top;
        for (scroll_up, at, count) in plan {
            self.scroll(scroll_up)?;
            self.paint_at(at, &lines[idx..idx + count])?;
            idx += count;
            top = at + count as u16;
        }
        self.top = top;
        Ok(())
    }

    /// Write whole rows at `at` — every column, spaces included, so shorter
    /// rows clear what they draw over (the pane is repainted in place). The
    /// one exception is the trailing half of a wide grapheme, which
    /// [`row_cells`] steps over. The rows are staged into the frame through
    /// a backend over its bytes: ratatui's own run-packing (MoveTo only
    /// between non-adjacent cells) keeps working, and nothing reaches the
    /// terminal until [`Stage::send`].
    fn paint_at(&mut self, at: u16, lines: &[Line<'static>]) -> io::Result<()> {
        let n = lines.len() as u16;
        if n == 0 {
            return Ok(());
        }
        let w = self.width;
        let area = Rect { x: 0, y: 0, width: w, height: n };
        let mut buf = Buffer::empty(area);
        Text::from(lines.to_vec()).render(area, &mut buf);
        let cells = row_cells(&buf, at);
        let mut backend = CrosstermBackend::new(&mut self.frame);
        backend.draw(cells.iter().map(|(x, y, c)| (*x, *y, c)))?;
        Ok(())
    }

    /// The pane itself.
    fn paint(&mut self, lines: &[Line<'static>]) -> io::Result<()> {
        self.paint_at(self.top, lines)
    }

    /// Append a command's escape sequence to the frame under construction.
    fn put(&mut self, cmd: impl Command) {
        let _ = put_cmd(&mut self.frame, cmd);
    }

    /// Open a frame: hide the caret for the composition to come — the
    /// writes of a frame walk a visible caret across rows that are only
    /// half-repainted, and the caret reappears (with [`Stage::send`]) only
    /// once the rows are in place, where it belongs.
    fn open_frame(&mut self) {
        self.put(Hide);
    }

    /// Put the composed frame on the wire: one synchronized burst, one
    /// flush — every flush is a render opportunity, so a frame offers the
    /// terminal exactly one. (Stdout is line-buffered and the frame carries
    /// the scroll's newlines, so the tty may still split the write at the
    /// last one; the sync brackets make that split invisible wherever the
    /// mode is known, and microseconds-wide where it is not.)
    fn send(&mut self) -> io::Result<()> {
        send_frame(&mut io::stdout().lock(), &self.frame)?;
        self.frame.clear();
        Ok(())
    }
}

/// The cells of a rendered block, offset to start at row `at`: one per
/// column, except the trailing halves of wide graphemes. A wide glyph
/// covers its neighbor column, so the buffer parks a blank cell there —
/// but the terminal's cursor already skips past both halves when the glyph
/// is printed, and the backend only repositions between *non-adjacent*
/// cells. Emitting the blank would land one column further right and shove
/// the rest of the row sideways (every CJK char spaced a column apart, the
/// caret off by that much). Stepping over the shadow, the way ratatui's
/// own buffer diff does, keeps runs contiguous and columns exact.
fn row_cells(buf: &Buffer, at: u16) -> Vec<(u16, u16, Cell)> {
    let w = buf.area.width;
    let mut cells = Vec::with_capacity(buf.content.len());
    for y in 0..buf.area.height {
        let mut x = 0u16;
        while x < w {
            let c = &buf[(x, y)];
            let width = c.cell_width().max(1);
            if c.diff_option != CellDiffOption::Skip {
                cells.push((x, at + y, c.clone()));
            }
            x += width;
        }
    }
    cells
}

/// Whether this frame must repaint the pane: the transcript has rows to
/// hand to the scrollback, nothing has been painted yet, the terminal
/// resized, or the pane's rows or caret differ from what is on screen.
///
/// The comparison renders the candidate pane first and diffs it against
/// the painted one — dirty-tracking at the render boundary rather than in
/// every update arm. Its cost is bounded by the pane (≤ screen rows), and
/// the alternative (per-field dirty flags) would tax every future rule
/// added to `update` with remembering to flag itself.
///
/// When none hold the frame is skipped wholesale — no cells written, no
/// `Hide`, no `Show`. That is not merely an optimization: the tick fires
/// every 120 ms even while the prompt sits idle and the state does not
/// change, and hiding then re-showing the cursor that often resets the
/// terminal's own blink phase, so the caret flickers eight times a second
/// instead of blinking at the terminal's pace. An unchanged frame leaves
/// the caret — and its blinking — entirely to the terminal.
fn needs_paint(
    grew: bool,
    painted: Option<&((u16, u16), view::Pane)>,
    w: u16,
    h: u16,
    next: &view::Pane,
) -> bool {
    if grew {
        return true;
    }
    match painted {
        None => true,
        Some(((pw, ph), p)) => *pw != w || *ph != h || p.lines != next.lines || p.cursor != next.cursor,
    }
}

/// Open the TUI: terminal setup, the event loop, guaranteed restore.
/// `convo` is the conversation to run on (history + its session); the
/// banners come from the composition root and land beside ours.
pub async fn run(cfg: &Config, convo: Convo, banners: Vec<String>) -> crate::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let sink = ChannelSink::new(tx);
    sink.show(Msg::Banner(format!(
        "jingwei — 精卫填海，一石一石 · {}",
        cfg.identity())));
    sink.show(Msg::Banner(
        "type a task · Ctrl-J / Shift-Enter breaks the line · Ctrl-O unfolds · Ctrl-C interrupts (twice exits) · /exit or Ctrl-D rests".into()));
    for b in banners {
        sink.show(Msg::Banner(b));
    }

    let mut app = App::new();
    app.input.history = load_history();
    app.info = model::Info {
        model: cfg.model.clone(),
        effort: cfg.effort_label().map(str::to_owned),
        context_limit: cfg.context_size,
    };

    let convo = Arc::new(AsyncMutex::new(convo));
    let mut agent: Option<Agent> = None;

    // Raw mode, bracketed paste, and the kitty keyboard protocol (which
    // makes Shift-Enter distinguishable from Enter on terminals that
    // support it; the rest ignore it). No alternate screen, no mouse.
    crossterm::terminal::enable_raw_mode().map_err(|e| Error::Msg(format!("raw mode: {e}")))?;
    crossterm::execute!(
        io::stdout(),
        EnableBracketedPaste,
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
                | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
        )
    )
    .map_err(|e| Error::Msg(format!("keyboard protocol: {e}")))?;
    let mut stage = Stage::new(4)
        .map_err(|e| Error::Msg(format!("terminal: {e}")))?;

    // only now does anything else read stdin
    let mut events = key_channel();
    let mut tick = tokio::time::interval(model::TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // the clock the ticks carry: measured, so a busy loop that skipped
    // ticks still bills the time it actually took
    let mut clock = Instant::now();

    // How many transcript rows have already been handed to the scrollback.
    let mut flushed = 0usize;
    // The Ctrl-O overlay; present only while the review view is open.
    let mut overlay: Option<ratatui::Terminal<Backend>> = None;
    // What the primary screen currently shows: the size the pane was
    // painted at, and the pane. Frames that would paint the very same
    // pane are skipped wholesale — see [`needs_paint`].
    let mut painted: Option<((u16, u16), view::Pane)> = None;

    let mut done = false;
    while !done {
        match app.mode {
            Mode::Input => {
                if overlay.take().is_some() {
                    // back on the primary screen: it still holds the pane
                    // and the whole scrollback, exactly as we left them.
                    // The mode switch rides the frame's first bytes and
                    // goes out with the repaint in the same burst (a
                    // repaint always follows — painted is forgotten), so
                    // no render catches the screen switched but the pane
                    // still stale. The overlay left the terminal's cursor
                    // hidden; the frame's own Show restores it even if
                    // the pane itself is unchanged.
                    stage.put(crossterm::terminal::LeaveAlternateScreen);
                    painted = None;
                }
                // The terminal's size, polled each frame (the same fallback
                // ratatui's autoresize runs per draw). On a change the screen
                // has already reflowed our rows and our pane by its own rules
                // — a width change rewraps everything, a height change adds or
                // drops rows — so we do not reuse the old geometry: we clear
                // and reprint the visible tail at the new width, the pane
                // re-anchored to the bottom.
                let (sw, sh) = crossterm::terminal::size().unwrap_or((stage.width, stage.screen));
                let w = sw.max(8);
                let h = sh.max(1);
                let resized = (w, h) != (stage.width, stage.screen);
                let p = view::pane(&app, w, h);
                let lines = if resized {
                    // Reprint only what fits, and no more than what was on
                    // screen before — rows that already rode into the
                    // scrollback must not be shown a second time.
                    let was_visible = stage.screen.saturating_sub(stage.height);
                    let count = was_visible.min(h.saturating_sub(p.lines.len() as u16));
                    stage.begin_resize(w, h, p.lines.len() as u16, count);
                    view::transcript_tail(&app, w as usize, count as usize)
                } else {
                    view::flush_lines(&app, w as usize, flushed)
                };
                let grew = !lines.is_empty();
                flushed = app.rows.len();
                if needs_paint(grew, painted.as_ref(), w, h, &p) {
                    // The whole frame is composed first and sent as one
                    // synchronized burst (see [`Stage::send`]): scrolls,
                    // transcript rows, and the pane land together, so the
                    // terminal is never handed the intermediate state —
                    // a scrolled screen with its freed rows still blank —
                    // that read as a flash. Cells go out as contiguous
                    // runs that walk a visible caret across the pane, and
                    // a growing pane scrolls the screen under it — the
                    // caret only exists once the rows are in place, so it
                    // is shown then, where it belongs.
                    stage.open_frame();
                    if resized {
                        // wipe the terminal's own reflow of the old rows
                        // before reprinting ours over it (the scrollback is
                        // untouched — only the screen clears)
                        stage.put(Clear(ClearType::All));
                    }
                    stage.flush(&lines)?;
                    stage.set_height(p.lines.len() as u16)?;
                    stage.paint(&p.lines)?;
                    if let Some(c) = p.cursor.as_ref() {
                        stage.put(Show);
                        stage.put(MoveTo(c.col.min(w.saturating_sub(1)), stage.top + c.row));
                    }
                    stage.send()?;
                    painted = Some(((w, h), p));
                }
            }
            Mode::Browse => {
                if overlay.is_none() {
                    crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
                    match ratatui::Terminal::new(CrosstermBackend::new(io::stdout())) {
                        Ok(t) => overlay = Some(t),
                        Err(e) => {
                            let _ = crossterm::execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen);
                            return Err(Error::Msg(format!("terminal: {e}")));
                        }
                    }
                }
                if let Some(t) = overlay.as_mut() {
                    // the overlay obeys the same one-burst law: ratatui's
                    // diff flushes on its own, so the brackets go around
                    // its whole draw — a resize or a first paint rewrites
                    // the entire screen, the widest tearing window there
                    // is. The End goes out even when the draw fails: a
                    // terminal left inside the brackets would present
                    // nothing at all.
                    let _ = crossterm::execute!(io::stdout(), BeginSynchronizedUpdate);
                    let drawn = draw_overlay(t, &app);
                    let _ = crossterm::execute!(io::stdout(), EndSynchronizedUpdate);
                    drawn?;
                }
            }
        }

        tokio::select! {
            maybe = events.recv() => {
                match maybe {
                    Some(ev) => match ev {
                        crossterm::event::Event::Key(k) => {
                            handle(step(&mut app, Ev::Key(k)), cfg, &convo, &mut agent, &sink);
                        }
                        crossterm::event::Event::Paste(p) => {
                            handle(step(&mut app, Ev::Paste(p)), cfg, &convo, &mut agent, &sink);
                        }
                        // a resize needs no arm of its own: the Input arm
                        // polls the size every frame and redraws the viewport
                        // when it changed, so the event just wakes the loop —
                        // which also covers the resize that happened while
                        // the overlay held the screen (that one reconciles on
                        // the iteration that closes the overlay).
                        _ => {}
                    },
                    None => done = true, // stdin gone
                }
            }
            maybe = rx.recv() => {
                // every message goes through update; reaping the agent
                // is an effect *after* the state transition, not a
                // second interpretation path beside it
                if let Some(m) = maybe {
                    let ended = matches!(m, Msg::TaskEnd);
                    step(&mut app, Ev::Msg(m));
                    if ended {
                        if let Some(a) = agent.take() {
                            a.reap(&sink).await;
                        }
                    }
                }
            }
            _ = tick.tick() => {
                let now = Instant::now();
                step(&mut app, Ev::Tick(now - clock));
                clock = now;
            }
        }
        done = done || app.quit;
    }

    // Restore the terminal no matter how we left the loop. The transcript
    // already lives in the scrollback; the pane's rows are erased and the
    // cursor parks on the first of them for the shell's prompt.
    if let Some(a) = agent.take() {
        a.token.cancel(); // don't leave a coroutine carrying stones behind
        a.job.abort();
    }
    if overlay.take().is_some() {
        let _ = crossterm::execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen);
    }
    // One burst to dissolve: the pane's rows erase, the caret parks on
    // the first of them for the shell's prompt, and the session's modes
    // roll back — a render between the per-row clears used to show the
    // pane dissolving row by row. Raw mode is termios, not escape
    // bytes; it goes back after the burst.
    let _ = send_frame(&mut io::stdout().lock(), &retire_bytes(stage.top, stage.height));
    let _ = crossterm::terminal::disable_raw_mode();
    save_history(&app.input.history);
    Ok(())
}

/// One overlay frame: the full-screen review view, cursor hidden.
fn draw_overlay(term: &mut ratatui::Terminal<Backend>, app: &App) -> io::Result<()> {
    term.draw(|f| {
        let area = f.area();
        let screen = view::browse(app, area.width, area.height);
        f.render_widget(ratatui::text::Text::from(screen.lines), area);
    })?;
    Ok(())
}

/// Perform an update's side effect: submit spawns the agent coroutine,
/// cancel fires its token.
fn handle(
    action: Action,
    cfg: &Config,
    convo: &Arc<AsyncMutex<Convo>>,
    agent: &mut Option<Agent>,
    sink: &ChannelSink,
) {
    match action {
        Action::None => {}
        Action::Cancel => {
            if let Some(a) = agent {
                a.token.cancel();
            }
        }
        Action::Submit(line) => {
            let cfg = cfg.clone();
            let convo = convo.clone();
            let token = Arc::new(CancelToken::new());
            let tok = token.clone();
            sink.show(Msg::TaskBegin(line.clone()));
            let sink = sink.clone(); // one for the spawned task, one for the shell
            let job = tokio::spawn(async move {
                let mut c = convo.lock().await;
                c.history.push(user_message(&line));
                c.persist(&sink); // the task is on disk before the first stone moves
                match agent_turn(&cfg, &mut c.history, &tok, &sink).await {
                    Err(Error::Interrupted) => {}
                    Err(e) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" error: {e} ") }),
                    Ok(()) => {}
                }
                c.persist(&sink); // run boundary: the file never ends mid-run
                sink.show(Msg::TaskEnd);
            });
            *agent = Some(Agent { token, job });
        }
    }
}

/// Should the REPL open the TUI? A terminal on stdout and no opt-out.
pub fn wanted() -> bool {
    io::stdout().is_terminal() && std::env::var_os("JINGWEI_NO_TUI").is_none()
}

fn load_history() -> Vec<String> {
    home_dir()
        .map(|d| d.join(".jingwei_history"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).map(unescape).collect())
        .unwrap_or_default()
}

fn save_history(history: &[String]) {
    if let Some(dir) = home_dir() {
        let body = history.iter().map(|s| escape(s)).collect::<Vec<_>>().join("\n");
        let _ = std::fs::write(dir.join(".jingwei_history"), body + "\n");
    }
}

/// One history entry per physical line, so the file stays greppable and
/// diffable. Backslashes and newlines are escaped — the same scheme the
/// old rustyline frontend wrote — which buys two things at once: a
/// multi-line task survives a restart as one recallable entry, and a
/// `~/.jingwei_history` from before the TUI rewrite loads as it was saved.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some(c) => {
                out.push('\\');
                out.push(c);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Terminal events on their own thread: crossterm's blocking `read()` is a
/// syscall, and syscalls don't suspend coroutines — so the reader gets a
/// thread and the event loop receives through a channel, exactly like the
/// SSE lines of a streamed response. Losing the receiver just ends the
/// thread; quitting never waits on a keystroke.
fn key_channel() -> tokio::sync::mpsc::UnboundedReceiver<crossterm::event::Event> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        // a read error ends the thread exactly like the receiver leaving
        while let Ok(ev) = crossterm::event::read() {
            if tx.send(ev).is_err() {
                break; // the event loop is gone
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    #[test]
    fn scrolling_is_line_feeds_never_csi_s() {
        // xterm.js (VS Code, Cursor) *deletes* the row CSI S scrolls off
        // instead of archiving it — the transcript left the screen and was
        // gone. Line feeds on the bottom row archive everywhere, the way a
        // shell's own output always has.
        let b = scroll_bytes(3);
        assert_eq!(b, "\r\n\r\n\r\n");
        assert!(!b.contains('\x1b'), "no escape sequences in the scroll primitive");
    }

    /// A sink that records the wire: every write, and every flush. Each
    /// flush is a render opportunity — what this counts is how many
    /// chances a frame gives the terminal to show a half-painted screen.
    #[derive(Default)]
    struct Wire {
        writes: Vec<Vec<u8>>,
        flushes: usize,
    }

    impl io::Write for Wire {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.writes.push(buf.to_vec());
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn a_frame_leaves_as_one_synchronized_burst() {
        // the bug this pins: a repaint used to arrive as several separate
        // flushes — Hide, then the scroll, then the rows it made room for,
        // then the caret — and a terminal that rendered between the scroll
        // and the rows showed the pane displaced over blank rows: a flash
        let mut w = Wire::default();
        send_frame(&mut w, b"").unwrap();
        assert_eq!((w.writes.len(), w.flushes), (0, 0), "an empty frame never touches the wire");

        let mut w = Wire::default();
        send_frame(&mut w, b"\x1b[?25l\r\nrows").unwrap();
        let flat = w.writes.concat();
        assert_eq!(flat, b"\x1b[?2026h\x1b[?25l\r\nrows\x1b[?2026l".to_vec(), "begin, body, end — in order");
        assert_eq!(w.flushes, 1, "one flush per frame, the one render opportunity it offers");
    }

    #[test]
    fn retiring_dissolves_the_pane_in_one_burst() {
        // the exit used to clear the pane with one execute! per row — a
        // render between them showed the pane dissolving row by row
        let s = String::from_utf8(retire_bytes(20, 4)).unwrap();
        assert_eq!(s.matches("\x1b[K").count(), 4, "one erase per pane row: {s:?}");
        // rows clear top-down …
        let rows: Vec<usize> =
            ["\x1b[21;1H", "\x1b[22;1H", "\x1b[23;1H", "\x1b[24;1H"].iter().map(|m| s.find(m).expect("a move per row")).collect();
        assert!(rows.windows(2).all(|w| w[0] < w[1]), "top-down: {s:?}");
        // … then the caret parks on the pane's first row for the shell's
        // prompt, and the session's modes roll back in the same burst
        assert!(s.ends_with("\x1b[21;1H\x1b[?25h\x1b[?2004l\x1b[<1u"), "caret parks, modes roll back: {s:?}");
    }

    #[test]
    fn history_round_trips_one_line_per_entry() {
        let entries = vec![
            "count *.rs".to_string(),
            "replace \\d+ with [0-9]".to_string(), // backslashes doubled, restored on load
            "one\ntwo\nthree".to_string(),          // multi-line stays one entry
            "C:\\path\\to\\file".to_string(),
            "trailing backslash \\".to_string(),
        ];
        let body = entries.iter().map(|e| escape(e)).collect::<Vec<_>>().join("\n") + "\n";
        assert!(body.lines().count() == entries.len(), "each entry is one physical line");
        let loaded: Vec<String> = body.lines().filter(|l| !l.trim().is_empty()).map(unescape).collect();
        assert_eq!(loaded, entries);
    }

    #[test]
    fn history_reads_the_rustyline_era_format() {
        // a file written by the old rustyline frontend: escaped backslashes
        // (linenoise scheme), never multi-line
        let old = "rename \\\\d to \\\\w and grep \\\\bword";
        assert_eq!(unescape(old), "rename \\d to \\w and grep \\bword");
        // a lone backslash at line's end survives
        assert_eq!(unescape("tail \\"), "tail \\");
    }


    /// A fake terminal with the two behaviors the paint path depends on:
    /// printing a glyph advances the cursor by its width, and the backend
    /// only repositions between non-adjacent cells. Wide glyphs fill every
    /// column they cover, so a row reads back with each one doubled.
    fn replay(cells: &[(u16, u16, Cell)], w: u16, h: u16) -> Vec<String> {
        let mut grid = vec![vec![' '; w as usize]; h as usize];
        let mut cur = (0u16, 0u16);
        let mut last: Option<(u16, u16)> = None;
        for (x, y, c) in cells {
            if !matches!(last, Some(p) if *x == p.0 + 1 && *y == p.1) {
                cur = (*x, *y);
            }
            let width = c.cell_width().max(1) as usize;
            let ch = c.symbol().chars().next().unwrap_or(' ');
            let row = &mut grid[cur.1 as usize];
            for dx in 0..width {
                if let Some(cell) = row.get_mut(cur.0 as usize + dx) {
                    *cell = ch;
                }
            }
            cur.0 += width as u16;
            last = Some((*x, *y));
        }
        grid.into_iter().map(|r| r.into_iter().collect()).collect()
    }

    fn render_buf(lines: &[Line<'static>], w: u16) -> Buffer {
        let area = Rect { x: 0, y: 0, width: w, height: lines.len() as u16 };
        let mut buf = Buffer::empty(area);
        Text::from(lines.to_vec()).render(area, &mut buf);
        buf
    }

    #[test]
    fn wide_graphemes_render_compact_not_spaced() {
        let buf = render_buf(&[Line::from("精卫填海 abc")], 40);
        let cells = row_cells(&buf, 0);
        // no cell is written into the right half a wide glyph covers: the
        // stream jumps 0,2,4,6 over the CJK, then runs one per column
        let xs: Vec<u16> = cells.iter().map(|(x, _, _)| *x).collect();
        assert_eq!(&xs[..8], &[0, 2, 4, 6, 8, 9, 10, 11], "trailing halves skipped: {xs:?}");
        let row = &replay(&cells, 40, 1)[0];
        assert_eq!(row.as_str(), format!("精精卫卫填填海海 abc{}", " ".repeat(28)), "no gaps between CJK: {row:?}");
    }

    #[test]
    fn pane_input_row_cjk_ends_where_the_caret_sits() {
        let mut a = App::new();
        for ch in "精卫 fill".chars() {
            step(&mut a, Ev::Key(KeyEvent {
                code: KeyCode::Char(ch),
                modifiers: KeyModifiers::NONE,
                kind: KeyEventKind::Press,
                state: KeyEventState::NONE,
            }));
        }
        let p = view::pane(&a, 40, 24);
        let caret = p.cursor.expect("caret in input mode");
        let buf = render_buf(&p.lines, 40);
        let rows = replay(&row_cells(&buf, 0), 40, buf.area.height);
        let input = &rows[1]; // rule, input, rule, bar
        let draft: String = input.chars().skip(10).take(caret.col as usize - 10).collect();
        assert_eq!(draft, "精精卫卫 fill", "the draft renders compact under the prompt: {input:?}");
        assert_eq!(input.chars().nth(caret.col as usize), Some(' '), "the column the caret sits on is free");
    }

    fn press(c: char) -> Ev {
        Ev::Key(KeyEvent {
            code: KeyCode::Char(c),
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    #[test]
    fn idle_ticks_leave_the_painted_pane_alone() {
        // the bug this pins: while the prompt sits idle a tick changed
        // nothing, yet the frame still hid and re-showed the caret every
        // 120ms — resetting the terminal's blink phase, so the caret
        // flickered instead of blinking at its own pace
        let mut a = App::new();
        for c in "hi".chars() {
            step(&mut a, press(c));
        }
        let painted = ((80u16, 24u16), view::pane(&a, 80, 24));
        for _ in 0..25 {
            step(&mut a, Ev::Tick(model::TICK));
            assert!(
                !needs_paint(false, Some(&painted), 80, 24, &view::pane(&a, 80, 24)),
                "an idle tick must not repaint — that is what reset the caret's blink"
            );
        }
    }

    #[test]
    fn real_changes_still_repaint() {
        let mut a = App::new();
        // the first frame (and the first after leaving the overlay)
        assert!(needs_paint(false, None, 80, 24, &view::pane(&a, 80, 24)));
        let painted = ((80u16, 24u16), view::pane(&a, 80, 24));
        // a resize, even one that renders identical rows
        assert!(needs_paint(false, Some(&painted), 60, 24, &view::pane(&a, 60, 24)));
        // the caret moves: typing changes the pane
        step(&mut a, press('x'));
        assert!(needs_paint(false, Some(&painted), 80, 24, &view::pane(&a, 80, 24)));
        // rows landed for the scrollback
        step(&mut a, Ev::Msg(Msg::Text("row\n".into())));
        assert!(needs_paint(true, Some(&painted), 80, 24, &view::pane(&a, 80, 24)));
        // the live thinking block's clock: a tick moves its "· 1s" label,
        // so the pane still repaints while reasoning streams
        step(&mut a, Ev::Msg(Msg::TaskBegin("t".into())));
        step(&mut a, Ev::Msg(Msg::Think("reasoning".into())));
        let thinking = ((80u16, 24u16), view::pane(&a, 80, 24));
        step(&mut a, Ev::Tick(model::TICK * 9));
        assert!(needs_paint(false, Some(&thinking), 80, 24, &view::pane(&a, 80, 24)));
    }

    // ---- the scroll-and-paint plan: pure, and pinned -----------------------

    /// Simulate the terminal executing a plan: a scroll drops the top rows
    /// and pads the bottom with blanks; a paint writes its lines in order.
    /// Returns the final screen (each cell: which line index sits there)
    /// and the pane's new top row. Asserts along the way that paints stay
    /// on-screen and never overwrite content a missing scroll should have
    /// carried away.
    fn run_plan(plan: &[(u16, u16, usize)], screen: u16) -> (Vec<Vec<Option<usize>>>, u16) {
        let mut grid = vec![vec![None; 1]; screen as usize];
        let mut next_line = 0usize;
        let mut pane_top = 0u16;
        for &(scroll_up, at, count) in plan {
            for _ in 0..scroll_up {
                grid.remove(0);
                grid.push(vec![None]);
            }
            for i in 0..count {
                let y = at as usize + i;
                assert!(y < screen as usize, "paint off-screen: row {y} of {screen}");
                assert!(grid[y][0].is_none(), "row {y} painted over surviving content — a scroll is missing");
                grid[y][0] = Some(next_line);
                next_line += 1;
            }
            pane_top = at + count as u16;
        }
        (grid, pane_top)
    }

    #[test]
    fn flush_plan_fits_without_scrolling_when_there_is_room() {
        // pane at row 5 of 24, 6 lines to flush, pane 4 tall: 5+6+4 <= 24
        let plan = flush_plan(5, 4, 24, 6);
        assert_eq!(plan.len(), 1, "one step: {plan:?}");
        assert_eq!(plan[0], (0, 5, 6), "no scroll, painted in place");
        let (grid, pane_top) = run_plan(&plan, 24);
        assert_eq!(pane_top, 11, "pane moves below the new rows");
        assert_eq!(grid[5..11], (0..6).map(|i| vec![Some(i)]).collect::<Vec<_>>());
    }

    #[test]
    fn flush_plan_chunks_when_the_transcript_overflows_the_screen() {
        // 100 lines on a 24-row screen: everything painted exactly once in
        // order, the pane stays on-screen, and the lines that remain
        // visible are the freshest ones
        let plan = flush_plan(20, 4, 24, 100);
        assert!(plan.iter().any(|&(s, _, _)| s > 0), "overflow scrolled: {plan:?}");
        let (grid, pane_top) = run_plan(&plan, 24);
        assert!(pane_top + 4 <= 24, "pane ends on-screen: pane_top {pane_top}");
        let visible: Vec<usize> = grid.iter().flatten().flatten().copied().collect();
        // 100 lines went through; the screen keeps the freshest ones above
        // the pane — exactly the rows the pane did not take
        assert_eq!(visible.len(), pane_top as usize);
        assert_eq!(visible, (100 - pane_top as usize..100).collect::<Vec<_>>(), "the freshest lines stay visible");
        assert!(grid[pane_top as usize..].iter().all(|c| c[0].is_none()), "the pane's rows are clear");
    }

    #[test]
    fn flush_plan_scrolls_a_bottom_pane_into_view() {
        // pane anchored at the very bottom: flushing must scroll the pane
        // up enough to stay visible
        let plan = flush_plan(19, 4, 24, 3);
        let (_, pane_top) = run_plan(&plan, 24);
        assert!(pane_top + 4 <= 24, "pane ends on-screen: pane_top {pane_top}");
    }

    #[test]
    fn flush_plan_zero_lines_paints_nothing() {
        assert_eq!(flush_plan(20, 4, 24, 0), vec![(0, 20, 0)]);
    }

    /// The invariant a resize reprint rests on: anchored so the pane sits at
    /// the bottom with `count` rows above it (`top = screen - pane - count`),
    /// the flush plan paints *in place* — it never scrolls. That is what
    /// makes the clear-and-reprint safe: no row is pushed into the
    /// scrollback, so rows that already rode there are not shown twice.
    #[test]
    fn resize_reprint_plan_never_scrolls() {
        for screen in [3u16, 10, 24, 60] {
            for pane_h in 1..=screen {
                for count in 0..=(screen - pane_h) {
                    let top = screen - pane_h - count;
                    let plan = flush_plan(top, pane_h, screen, count as usize);
                    let scrolled: u16 = plan.iter().map(|(s, _, _)| *s).sum();
                    assert_eq!(scrolled, 0, "screen={screen} pane={pane_h} count={count}: {plan:?}");
                    // and the pane's top after the reprint is where we anchored it
                    let end = plan.last().map(|(_, at, c)| at + *c as u16).unwrap_or(top);
                    assert_eq!(end, screen - pane_h, "screen={screen} pane={pane_h} count={count}");
                }
            }
        }
    }
}
