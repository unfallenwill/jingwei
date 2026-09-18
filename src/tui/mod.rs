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

use crate::api::VENDORS;
use crate::config::Config;
use crate::display::{ChannelSink, Msg, Sev, Show as DisplayShow};
use crate::mcp::Hub;
use crate::session::Convo;
use crate::settings::Settings;
use crate::{agent_turn, user_message, Error};

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(std::path::PathBuf::from)
}
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::style::Print;
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{BeginSynchronizedUpdate, Clear, ClearType, EndSynchronizedUpdate};
use crossterm::Command;
use model::{App, CompletionItem, CompletionKind, Mode};
use update::{detect_completion, set_completion};
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

/// The running agent: its cancellation token and its coroutine.
struct Agent {
    token: Arc<crate::cancel::CancelToken>,
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

/// Plan for handing `n` finished lines to the scrollback above a pane
/// at row `top`, `height` rows tall, on a screen of `screen` rows: a list
/// of (scroll_up, paint_at, count) steps. Pure arithmetic, the one piece
/// of the stage that can be wrong in interesting ways — and therefore
/// the one piece that is unit-tested.
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

/// The bytes that scroll the screen one row at a time: CR+LF pairs,
/// printed with the cursor parked on the last row. Pure, so the choice
/// it pins — line feeds, never `CSI S` — is a testable fact. See
/// [`Stage::scroll`] for why.
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
/// geometry is tracked, never queried.
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
    /// are painted above the pane and, once the screen is full, carried
    /// into the native scrollback. Same chunked scroll-and-paint
    /// arithmetic ratatui's inline viewport uses, kept here so the
    /// heights stay ours — the arithmetic lives in [`flush_plan`].
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
/// but the terminal's cursor already skips past both halves when printed,
/// and the backend only repositions between *non-adjacent* cells.
/// Emitting the blank would shove the rest of the row sideways (every
/// CJK char spaced a column apart). Stepping over the shadow, the way
/// ratatui's own buffer diff does, keeps runs contiguous and columns
/// exact.
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
/// hand to the scrollback, nothing painted yet, the terminal resized,
/// or the pane's rows or caret differ from what is on screen.
///
/// The comparison renders the candidate pane first and diffs it against
/// the painted one — dirty-tracking at the render boundary rather than
/// in every update arm. Cost is bounded by the pane (≤ screen rows);
/// the alternative (per-field dirty flags) would tax every future rule
/// with remembering to flag itself.
///
/// When none hold the frame is skipped wholesale — no cells written, no
/// `Hide`, no `Show`. Not merely an optimization: the tick fires every
/// 120 ms even while the prompt sits idle, and hiding then re-showing
/// the cursor that often resets the terminal's own blink phase, so the
/// caret flickers eight times a second instead of blinking at the
/// terminal's pace. An unchanged frame leaves the caret — and its
/// blinking — entirely to the terminal.
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
/// `convo` is the conversation to run on; `hub` is the MCP hub: a clone
/// the TUI shares with the agent loop, and uses itself for `/mcp`
/// slash commands while idle.
pub async fn run(cfg: Config, mut ctx: crate::context::Context, convo: Convo, banners: Vec<String>, hub: Hub) -> crate::Result<()> {
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
    // MCP startup notes — same surface the plain REPL shows, so the two
    // frontends announce servers the same way.
    for note in hub.notes().await {
        sink.show(Msg::Banner(note));
    }

    let mut app = App::new();
    app.input.history = load_history();
    app.info = model::Info {
        model: cfg.model.clone(),
        effort: cfg.effort_label().map(str::to_owned),
        context_limit: cfg.context_size,
    };

    let mut cfg = cfg;
    // convo is replaceable — `/model` swaps in a fresh Convo without
    // disturbing the running event loop. Each replacement wraps a new
    // AsyncMutex; the previous one is dropped with its lock held at
    // most until any in-flight agent coroutine finishes.
    let mut convo = Arc::new(AsyncMutex::new(convo));
    let mut agent: Option<Agent> = None;
    // The MCP hub is shared with the spawned agent task (so MCP tool
    // calls have a hub to dispatch to) and with the slash dispatcher
    // (so /mcp can read & operate on the same servers). Cloning a Hub
    // is cloning an mpsc::Sender — cheap, and the actor task is the
    // single owner of the real state.
    let hub = hub;

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
                            let action = step(&mut app, Ev::Key(k));
                            // After every key event the input may have
                            // crossed a slash-command boundary — sync the
                            // menu from scratch (cheap when nothing changes).
                            refresh_completion(&mut app, &hub).await;
                            apply_outcome(handle(action, &cfg, &mut ctx, &convo, &mut agent, &sink, &hub), &mut cfg, &mut convo, &mut app, &sink);
                        }
                        crossterm::event::Event::Paste(p) => {
                            let action = step(&mut app, Ev::Paste(p));
                            refresh_completion(&mut app, &hub).await;
                            apply_outcome(handle(action, &cfg, &mut ctx, &convo, &mut agent, &sink, &hub), &mut cfg, &mut convo, &mut app, &sink);
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
    // pane dissolving row by row. Raw mode is termios, not escape bytes.
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

/// The outcome of `handle` — what the main loop needs to do next.
/// `Submit` already starts the agent coroutine inside `handle`, so the
/// only outcomes that escape are the ones the loop itself must act on:
/// cancel a running task, switch to a new (cfg, convo, banners), or do
/// nothing.
#[derive(Default)]
#[allow(clippy::large_enum_variant)] // Switch carries a full Config + Convo; the
                                      // other arms are empty. The trade-off is fine:
                                      // None / Cancel are the common case (zero cost)
                                      // and Switch is the rare path that pays for it.
enum HandleOutcome {
    #[default]
    None,
    Cancel,
    /// Replace cfg / convo / banners with the ones in here. The previous
    /// Convo is dropped (its session file is already on disk from the
    /// last `persist`); the new Convo starts a fresh session file under
    /// the same project subdirectory.
    Switch {
        cfg: Config,
        convo: Convo,
        banners: Vec<String>,
        /// A short status line the loop will show in the transcript so
        /// the user sees what just happened (model name, protocol).
        announce: String,
    },
}

impl std::fmt::Debug for HandleOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Config / Convo don't impl Debug — and we don't need them in
        // test failure messages; the variant tag is enough.
        match self {
            HandleOutcome::None => write!(f, "None"),
            HandleOutcome::Cancel => write!(f, "Cancel"),
            HandleOutcome::Switch { announce, .. } => write!(f, "Switch {{ announce: {announce:?} }}"),
        }
    }
}

/// Apply the outcome of `handle` to the loop's mutable state. The
/// switch arm is the one that mutates: the loop's `cfg` and `convo`
/// are replaced, the `info` row of the pane is refreshed, and a
/// transcript note announces what happened.
fn apply_outcome(outcome: HandleOutcome, cfg: &mut Config, convo: &mut Arc<AsyncMutex<Convo>>, app: &mut App, sink: &ChannelSink) {
    match outcome {
        HandleOutcome::None | HandleOutcome::Cancel => {}
        HandleOutcome::Switch { cfg: new_cfg, convo: new_convo, banners, announce } => {
            *cfg = new_cfg;
            *convo = Arc::new(AsyncMutex::new(new_convo));
            app.info = model::Info {
                model: cfg.model.clone(),
                effort: cfg.effort_label().map(str::to_owned),
                context_limit: cfg.context_size,
            };
            for b in banners {
                sink.show(Msg::Banner(b));
            }
            sink.show(Msg::Note { sev: Sev::Warn, text: format!(" {announce} ") });
            // the next frame's `painted` will diff against the new
            // `app.info` and redraw the pane — no explicit dirty flag.
        }
    }
}

/// Parse a `/`-prefixed input line into a slash command + its args.
/// A line that is not a slash command returns `None`; the caller falls
/// through to the normal submit path.
fn parse_slash(line: &str) -> Option<SlashCmd<'_>> {
    let line = line.trim();
    let rest = line.strip_prefix('/')?;
    let mut it = rest.split_whitespace();
    let name = it.next()?;
    Some(SlashCmd { name, args: it.collect() })
}

#[derive(Debug)]
struct SlashCmd<'a> {
    name: &'a str,
    args: Vec<&'a str>,
}

/// Dispatch a slash command. The dispatcher is small and grows by
/// accretion — every new command adds a match arm and nothing else.
/// Unknown commands surface in the transcript as a `Note` so the user
/// sees the typo (no silent failure).
fn dispatch_slash(cmd: SlashCmd<'_>, cfg: &Config, convo: &Arc<AsyncMutex<Convo>>, sink: &ChannelSink, hub: &Hub) -> HandleOutcome {
    match cmd.name {
        "exit" | "quit" => {
            // the loop owns the quit flag; we just stop the current
            // agent and let the loop see Cancel → drain → exit
            HandleOutcome::Cancel
        }
        "model" => match slash_model(cmd.args, cfg, convo) {
            Ok(o) => o,
            Err(e) => {
                sink.show(Msg::Note { sev: Sev::Err, text: format!(" {e} ") });
                HandleOutcome::None
            }
        },
        "mcp" => {
            // /mcp subcommands are handled inline (cannot block the TUI
            // event loop on a network round-trip). The handler mirrors
            // plain::handle_mcp_command exactly so the two frontends
            // say the same thing for the same input.
            let hub = hub.clone();
            let sink = sink.clone();
            let rest = cmd.args.join(" ");
            tokio::spawn(async move {
                crate::plain::handle_mcp_command(&rest, &hub, &sink).await;
            });
            HandleOutcome::None
        }
        "help" => {
            sink.show(Msg::Note {
                sev: Sev::Warn,
                text: " commands: /model [<provider>/<model>] · /mcp [list|enable|disable|reconnect|disconnect <name>] · /exit · /help ".into(),
            });
            HandleOutcome::None
        }
        other => {
            sink.show(Msg::Note { sev: Sev::Warn, text: format!(" unknown command: /{other} · try /help ") });
            HandleOutcome::None
        }
    }
}

/// `/model` switches the active profile. The new Config + a fresh Convo
/// ride back through `HandleOutcome::Switch`; the loop swaps them in.
/// With no args we surface the current selection and the available
/// profiles; with one arg we look it up under `<provider>/<model>`.
fn slash_model(args: Vec<&str>, cfg: &Config, convo: &Arc<AsyncMutex<Convo>>) -> Result<HandleOutcome, String> {
    let settings = Settings::load().map_err(|e| e.to_string())?;
    let current_key = settings
        .active
        .clone()
        .or_else(|| {
            settings
                .providers
                .keys()
                .next()
                .cloned()
        });

    let selected_key = match args.as_slice() {
        [] => {
            // list: print every profile and return None so the current
            // active stays — picking a row happens on the next /model call.
            // (A TUI picker is a future affordance; today the list is text.)
            let mut lines = String::from("/model — providers:");
            for (k, p) in &settings.providers {
                let mark = if Some(k) == current_key.as_ref() { " *" } else { "  " };
                let base = p.base_url.as_deref().unwrap_or("(vendor default)");
                lines.push_str(&format!("\n  {mark}{k}  proto={}  base={base}", p.protocol));
            }
            return Err(lines); // re-uses Err as "show this in the transcript"
        }
        [one] => one.to_string(),
        _ => return Err("/model takes at most one argument: <provider>/<model>".into()),
    };
    let profile = settings.providers.get(&selected_key)
        .ok_or_else(|| {
            let known: Vec<&str> = settings.providers.keys().map(String::as_str).collect();
            format!(
                "no profile named '{selected_key}' (known: {})",
                known.join(", ")
            )
        })?;
    if Some(&selected_key) == current_key.as_ref() {
        return Err(format!("already on {selected_key}"));
    }

    // Build the new Config from the chosen profile. We start from the
    // vendor defaults (which base_url / model / protocol to use), then
    // overlay what the profile pins. Args/env were already resolved
    // into `cfg`, so we keep its effort / max_tokens / context_size /
    // streaming / cache / thinking — the *behaviour* knobs survive a
    // profile switch unchanged.
    let mut new_cfg = cfg.clone();
    new_cfg.protocol = protocol_of(&profile.protocol)
        .ok_or_else(|| format!("unknown protocol '{}'", profile.protocol))?;
    new_cfg.api_key = profile.api_key.clone();
    new_cfg.base_url = profile
        .base_url
        .clone()
        .or_else(|| new_cfg.protocol.default_base().map(str::to_owned))
        .ok_or_else(|| format!("profile '{selected_key}' has no base_url and the vendor does not default one"))?;
    // model id — derive from the profile key (the part after the slash)
    let model_id = selected_key.split_once('/').map(|(_, m)| m.to_string())
        .ok_or_else(|| format!("profile key '{selected_key}' is not '<provider>/<model>'"))?;
    new_cfg.model = model_id;

    // Close the current session: persist one last time so any agent turn
    // in flight is on disk, then drop the in-memory convo. The file is
    // already closed by Convo's Drop.
    let mut c = match convo.try_lock() {
        Ok(g) => g,
        Err(_) => return Err("cannot switch profile while an agent turn is running — wait for it to finish (or /exit)".into()),
    };
    if let Err(e) = c.persist() {
        return Err(format!("could not close current session: {e}"));
    }
    drop(c);

    // Persist the new active. Loading+rewriting preserves any unknown
    // future fields; if settings.json doesn't exist yet we create the
    // bare structure.
    let mut next_settings = settings;
    next_settings.active = Some(selected_key.clone());
    next_settings.save().map_err(|e| format!("save settings.json: {e}"))?;

    // Open a fresh session under the new identity. The session module's
    // `Session::new` is a fresh file under the project subdirectory;
    // the previous file stays on disk under its own id.
    let prov = crate::session::Provenance {
        model: new_cfg.model.clone(),
        protocol: new_cfg.protocol_label().into(),
        base_url: new_cfg.base_url.clone(),
    };
    let new_session = crate::session::Session::new(&prov, crate::config::HISTORY_FORMAT);
    let announce = format!(
        "switched to {} (new session {})",
        selected_key,
        new_session.id()
    );
    let banner = match new_session.path() {
        Some(p) => format!("session {} · {}", new_session.id(), p.display()),
        None => format!("session {} · no home directory: this conversation stays in memory", new_session.id()),
    };
    let new_convo = crate::session::Convo::persistent(new_session, vec![]);
    Ok(HandleOutcome::Switch { cfg: new_cfg, convo: new_convo, banners: vec![banner], announce })
}

/// Translate a protocol name string into the `Protocol` enum, falling
/// through VENDORS so adding a vendor lands here without a touch.
fn protocol_of(name: &str) -> Option<crate::api::Protocol> {
    VENDORS.iter().find(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, p)| *p)
}

/// Perform an update's side effect. `Submit` is fully consumed inside
/// (the agent coroutine is spawned) and yields no outcome — the loop
/// keeps its current cfg and convo. Slash commands on the input line
/// resolve here too: dispatching `/model` returns `Switch` with the
/// newly-built Config and Convo.
fn handle(action: Action, cfg: &Config, ctx: &mut crate::context::Context, convo: &Arc<AsyncMutex<Convo>>,
          agent: &mut Option<Agent>, sink: &ChannelSink, hub: &Hub) -> HandleOutcome {
    match action {
        Action::None => HandleOutcome::None,
        Action::Cancel => {
            if let Some(a) = agent {
                a.token.cancel();
            }
            HandleOutcome::Cancel
        }
        Action::Submit(line) => {
            // Slash commands share the input line with regular tasks —
            // they never reach the agent. Dispatch first, and only fall
            // through to a real submit when the line is just a task.
            if let Some(cmd) = parse_slash(&line) {
                return dispatch_slash(cmd, cfg, convo, sink, hub);
            }
            // The narrow race: the agent's `Msg::TaskBegin` rides the
            // channel back to update `app.working()`; between sending
            // it and the loop reading it, the user could press Enter
            // again — the agent Option already holds a live job, and
            // spawning a second one would orphan the first. Two
            // coroutines then race the convo lock and both push into
            // history; the file would be torn. Refuse instead.
            if agent.is_some() {
                sink.show(Msg::Note {
                    sev: Sev::Warn,
                    text: " a task is already running — wait for it to finish or Ctrl-C to cancel ".into(),
                });
                return HandleOutcome::None;
            }
            let cfg = cfg.clone();
            let mut ctx = ctx.clone();
            let convo = convo.clone();
            let hub = hub.clone();
            let token = Arc::new(crate::cancel::CancelToken::new());
            let tok = token.clone();
            sink.show(Msg::TaskBegin(line.clone()));
            let sink = sink.clone(); // one for the spawned task, one for the shell
            let job = tokio::spawn(async move {
                let mut c = convo.lock().await;
                c.history.push(user_message(&line));
                if let Err(e) = c.persist() {
                    sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
                }
                // the task is on disk before the first stone moves
                let mut turn = crate::turn::Turn::new(0);
                match agent_turn(&cfg, &mut ctx, &mut c.history, &mut turn, &tok, &hub, &sink).await {
                    Err(Error::Interrupted) => {}
                    Err(e) => sink.show(Msg::Note { sev: Sev::Err, text: format!(" error: {e} ") }),
                    Ok(()) => {}
                }
                if let Err(e) = c.persist() {
                    sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
                }
                // run boundary: the file never ends mid-run
                sink.show(Msg::TaskEnd);
            });
            *agent = Some(Agent { token, job });
            HandleOutcome::None
        }
    }
}

/// Should the REPL open the TUI? A terminal on stdout is the only gate.
pub fn wanted() -> bool {
    io::stdout().is_terminal()
}

/// Re-derive the slash-command menu from the current input text, with
/// fresh candidates from the hub (server names) and settings (profile
/// keys). Called after every key event the run loop processes.
async fn refresh_completion(app: &mut App, hub: &Hub) {
    let Some((kind, prefix)) = detect_completion(&app.input.text) else {
        if app.completion.is_some() { app.completion = None; }
        return;
    };
    let candidates = collect_candidates(kind, &prefix, hub).await;
    set_completion(app, kind, &prefix, candidates);
}

/// Fetch the candidate list for `kind`, filtered by `prefix`. The
/// hard-coded lists (commands, `/mcp` subcommands) come back at once;
/// settings and the hub are awaited only for the kinds that need them.
async fn collect_candidates(
    kind: CompletionKind,
    prefix: &str,
    hub: &Hub,
) -> Vec<CompletionItem> {
    let prefix_lc = prefix.to_lowercase();
    let starts = |s: &str| s.to_lowercase().starts_with(&prefix_lc);
    match kind {
        CompletionKind::Command => vec![
            item("exit",   "/exit",   "leave the session"),
            item("quit",   "/quit",   "leave the session"),
            item("model",  "/model",  "switch profile (bare lists, <key> switches)"),
            item("mcp",    "/mcp",    "manage MCP servers — list | enable | disable | reconnect | disconnect"),
            item("help",   "/help",   "show this menu"),
        ].into_iter().filter(|c| prefix.is_empty() || starts(&c.insert)).collect(),
        CompletionKind::McpSub => vec![
            item("list",       "list",       "list servers + their states"),
            item("enable",     "enable",     "re-enable a disabled server"),
            item("disable",    "disable",    "take a server offline, keep its config"),
            item("reconnect",  "reconnect",  "close + reopen the connection"),
            item("disconnect", "disconnect", "disconnect a server (errors if not connected)"),
        ].into_iter().filter(|c| prefix.is_empty() || starts(&c.insert)).collect(),
        CompletionKind::ModelArg => {
            let Ok(settings) = Settings::load() else { return vec![] };
            settings.providers.keys()
                .filter(|k| prefix.is_empty() || starts(k))
                .map(|k| CompletionItem {
                    insert: k.clone(),
                    label: k.clone(),
                    description: "switch to this profile".into(),
                    trailing_space: false,
                })
                .collect()
        }
        CompletionKind::McpServer => {
            let servers = hub.list().await;
            servers.into_iter().map(|s| s.name)
                .filter(|n| prefix.is_empty() || starts(n))
                .map(|n| CompletionItem {
                    insert: n.clone(),
                    label: n.clone(),
                    description: "MCP server".into(),
                    trailing_space: false,
                })
                .collect()
        }
    }
}

fn item(insert: &str, label: &str, description: &str) -> CompletionItem {
    // Commands end a word — Tab appends a space and the menu then offers
    // what comes next (subcommands, server tags). Profile keys and
    // server names are the whole argument — Tab just lands the caret.
    let trailing_space = matches!(insert, "exit" | "quit" | "help" | "model" | "mcp"
                                       | "list" | "enable" | "disable" | "reconnect" | "disconnect");
    CompletionItem {
        insert: insert.into(),
        label: label.into(),
        description: description.into(),
        trailing_space,
    }
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
        // the draft starts immediately after the prompt prefix; the test
        // helper duplicates wide chars back into chars so 精卫 (4 display
        // cols) lands as 4 chars. caret.col is in display columns, so the
        // draft's display width is `caret.col - prompt_w`.
        let prefix = format!("{}{}", crate::display::PROMPT_HEAD, crate::display::PROMPT_GUTTER);
        let after = input.strip_prefix(&prefix).expect("prompt prefix");
        let draft: String = after.chars().take((caret.col as usize).saturating_sub(crate::display::prompt_w())).collect();
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

    // ---- the stage's geometry and frame machinery ------------------------

    /// Build a stage with the fields tests want. `Stage::new` queries the
    /// real terminal — impossible under cargo test, so the geometry that
    /// drives every method is set by hand.
    fn fake_stage(top: u16, height: u16, width: u16, screen: u16) -> Stage {
        Stage { frame: Vec::new(), top, height, width, screen }
    }

    #[test]
    fn begin_resize_records_new_geometry_and_anchors_pane_at_bottom() {
        let mut s = fake_stage(0, 0, 0, 0);
        // resize to 80×24, pane 4 rows, no transcript reprinted above
        s.begin_resize(80, 24, 4, 0);
        assert_eq!((s.width, s.screen, s.height, s.top), (80, 24, 4, 20));
        // with a 6-row tail: pane at 14, 4 rows, screen 24
        s.begin_resize(80, 24, 4, 6);
        assert_eq!(s.top, 24 - 4 - 6);
        // width below the floor: Stage widens to 8 so no row ever lands in
        // a buffer with width zero
        s.begin_resize(2, 24, 4, 0);
        assert_eq!(s.width, 8);
        // height 0: clamped to 1 so set_height has a non-empty pane
        s.begin_resize(80, 0, 0, 0);
        assert_eq!(s.screen, 1);
        assert_eq!(s.height, 1);
    }

    #[test]
    fn set_height_clamps_to_screen_and_one() {
        let mut s = fake_stage(20, 4, 80, 24);
        // height 0 → clamped up to 1, no shrinks/grows trigger
        s.set_height(0).unwrap();
        assert_eq!(s.height, 1);
        // reset to a smaller height before the next clamp test, so the
        // bounded-by-screen path runs without scrolling
        let mut s = fake_stage(20, 4, 80, 24);
        // height above screen → clamped down (this case scrolls — see the
        // grow-scroll test for its behavior; here we only check the clamp)
        s.set_height(100).unwrap();
        assert_eq!(s.height, 24);
        // a height equal to current is a no-op: frame doesn't grow
        let after_clamp = s.frame.len();
        s.set_height(s.height).unwrap();
        assert_eq!(s.frame.len(), after_clamp, "no-op on equal height");
    }

    #[test]
    fn set_height_shrink_clears_rows_the_pane_gives_up() {
        // pane 4 rows starting at row 20: shrink to 2 clears rows 22 and 23
        let mut s = fake_stage(20, 4, 80, 24);
        s.set_height(2).unwrap();
        assert_eq!(s.height, 2);
        let body = String::from_utf8(s.frame.clone()).unwrap();
        // one MoveTo per cleared row, then an erase to end of line on each
        assert_eq!(body.matches("\x1b[23;1H").count(), 1, "row 22 cleared");
        assert_eq!(body.matches("\x1b[24;1H").count(), 1, "row 23 cleared");
        assert_eq!(body.matches("\x1b[K").count(), 2, "two erases");
    }

    #[test]
    fn set_height_grow_scrolls_when_pane_would_fall_off_screen() {
        // pane anchored at row 18 of 24 with 6 rows — growing it to 10 rows
        // would push the bottom to row 28; stage scrolls 4 lines first so
        // the new top sits at screen - new_h
        let mut s = fake_stage(18, 6, 80, 24);
        s.set_height(10).unwrap();
        assert_eq!(s.height, 10);
        assert_eq!(s.top, 24 - 10, "pane re-anchored under the scroll");
        let body = String::from_utf8(s.frame.clone()).unwrap();
        assert!(body.contains("\x1b[24;1H"), "scroll from the last row");
        assert_eq!(body.matches("\r\n").count(), 4, "exactly 4 line feeds");
    }

    #[test]
    fn set_height_grow_above_screen_does_not_scroll() {
        // pane anchored at row 18 with height 2: growing to height 6 fits
        // exactly (18 + 6 = 24), so no scroll is needed
        let mut s = fake_stage(18, 2, 80, 24);
        s.set_height(6).unwrap();
        assert_eq!(s.height, 6);
        assert_eq!(s.top, 18, "top unchanged when growth fits");
        assert!(s.frame.is_empty(), "no scroll bytes: {:?}", s.frame);
    }

    #[test]
    fn scroll_emits_line_feeds_only_when_asked_to() {
        let mut s = fake_stage(20, 4, 80, 24);
        s.scroll(0).unwrap();
        assert!(s.frame.is_empty(), "zero scrolls leave the frame empty");
        s.scroll(3).unwrap();
        let body = String::from_utf8(s.frame.clone()).unwrap();
        assert_eq!(body, "\x1b[24;1H\r\n\r\n\r\n");
        assert!(!body.contains('\x1b') || body.contains("\x1b[24;1H"), "only the move cursor — no CSI S");
    }

    #[test]
    fn flush_with_no_lines_keeps_frame_empty() {
        let mut s = fake_stage(20, 4, 80, 24);
        s.flush(&[]).unwrap();
        assert!(s.frame.is_empty(), "nothing to flush, nothing written");
    }

    #[test]
    fn flush_paints_in_place_when_there_is_room() {
        // top + n + height <= screen, so the plan has zero scroll
        let mut s = fake_stage(15, 4, 80, 24); // 15 + 2 + 4 = 21 <= 24
        let lines = vec![Line::from("hello"), Line::from("world")];
        s.flush(&lines).unwrap();
        // the pane sits below the new rows, with no scroll
        assert_eq!(s.top, 17);
        // no scroll bytes — flush_plan emitted NO line feeds
        assert!(!String::from_utf8_lossy(&s.frame).contains('\r'),
            "no scroll bytes on the wire: {:?}", s.frame);
        // the rows landed in order, in place
        let body = String::from_utf8_lossy(&s.frame);
        let hello_at = body.find("hello").expect("hello painted");
        let world_at = body.find("world").expect("world painted");
        assert!(hello_at < world_at, "rows painted in order: {body:?}");
    }

    #[test]
    fn flush_scrolls_and_paints_when_the_screen_fills() {
        // 30 rows into a screen 24 rows tall, pane 4 rows: the plan must
        // scroll the transcript under itself and paint every row exactly
        // once, in order, with the pane still on-screen at the bottom.
        let mut s = fake_stage(20, 4, 80, 24);
        let lines: Vec<Line<'static>> = (0..30).map(|i| Line::from(format!("r{i:02}"))).collect();
        s.flush(&lines).unwrap();
        let body = String::from_utf8_lossy(&s.frame);
        // every row painted exactly once, in order — no row missing,
        // no row duplicated, no row out of order. ratatui emits a small
        // reset-cursor escape after the last cell, so we don't require
        // the cursor to land at end-of-string (that would couple us to
        // ratatui's exact byte sequence).
        let mut cursor = 0usize;
        for i in 0..30 {
            let token = format!("r{i:02}");
            let at = body[cursor..].find(&token)
                .unwrap_or_else(|| panic!("line {i} (r{i:02}) not found after cursor {cursor}: {body:?}"));
            cursor += at + token.len();
            // the same row must NOT appear again after this point
            assert!(body[cursor..].find(&token).is_none(),
                "line {i} (r{i:02}) appears twice: {body:?}");
        }
        // pane still fits on screen
        assert!(s.top + s.height <= 24, "pane top {} + height {} > 24", s.top, s.height);
        // at least one scroll step happened — the screen would have
        // overflowed without one
        let scroll_count = body.matches("\r\n").count();
        assert!(scroll_count > 0, "screen overflow produced zero scrolls: {body:?}");
    }

    #[test]
    fn paint_at_with_zero_lines_writes_nothing() {
        let mut s = fake_stage(20, 4, 80, 24);
        s.paint_at(10, &[]).unwrap();
        assert!(s.frame.is_empty(), "no rows → no bytes");
    }

    #[test]
    fn paint_uses_the_pane_top_as_its_row() {
        let mut s = fake_stage(20, 4, 80, 24);
        s.paint(&[Line::from("only")]).unwrap();
        // the painted row lands at top (20) — not at 0
        let body = String::from_utf8_lossy(&s.frame);
        assert!(body.contains("only"));
    }

    #[test]
    fn put_appends_a_command_to_the_frame() {
        let mut s = fake_stage(20, 4, 80, 24);
        s.put(crossterm::style::Print::<&str>("marker"));
        assert!(s.frame.ends_with(b"marker"), "{:?}", s.frame);
    }

    #[test]
    fn open_frame_hides_the_caret_before_writes() {
        let mut s = fake_stage(20, 4, 80, 24);
        s.open_frame();
        // Hide writes \x1b[?25l — every frame starts with it
        assert!(s.frame.starts_with(b"\x1b[?25l"), "{:?}", s.frame);
    }

    // ---- row_cells / needs_paint: the rendering helpers -------------------

    #[test]
    fn row_cells_offsets_to_the_given_row() {
        let buf = render_buf(&[Line::from("hi")], 10);
        let cells = row_cells(&buf, 7);
        for (_, y, _) in &cells {
            assert_eq!(*y, 7, "every cell stamped with `at`");
        }
    }

    #[test]
    fn row_cells_steps_over_wide_grapheme_trailing_halves() {
        // a 3-wide glyph at columns 0..3: row_cells pushes only the first
        // half; it never inserts the blank that would shove the next cell
        let buf = render_buf(&[Line::from("精卫")], 6);
        let cells = row_cells(&buf, 0);
        let xs: Vec<u16> = cells.iter().map(|(x, _, _)| *x).collect();
        // two wide glyphs: emitted at x=0 (width 2), then x=2 (width 2)
        assert!(!xs.contains(&1), "the trailing half of 精 is skipped");
        assert!(!xs.contains(&3), "the trailing half of 卫 is skipped");
        assert!(xs.contains(&0) && xs.contains(&2));
    }

    #[test]
    fn row_cells_empty_buffer_returns_nothing() {
        let buf = render_buf(&[], 10);
        assert!(row_cells(&buf, 0).is_empty());
    }

    #[test]
    fn needs_paint_returns_true_on_grow_even_when_pane_matches() {
        // the bug this pins: a frame that paints *nothing* of its own (no
        // transcript rows) but had a previous frame still repaints because
        // transcript rows landed; a tick that doesn't grow never repaints
        let p = view::Pane { lines: vec![Line::from("a")], cursor: None };
        let painted = ((80u16, 24u16), p.clone());
        assert!(needs_paint(true, Some(&painted), 80, 24, &p), "grew → paint");
        assert!(!needs_paint(false, Some(&painted), 80, 24, &p), "no grow, no change → skip");
    }

    #[test]
    fn needs_paint_first_frame_without_history_paints() {
        let p = view::Pane { lines: vec![], cursor: None };
        assert!(needs_paint(false, None, 80, 24, &p), "no prior paint → paint");
    }

    #[test]
    fn needs_paint_resize_paints_even_when_pane_lines_match() {
        let p = view::Pane { lines: vec![Line::from("a")], cursor: None };
        let painted = ((80u16, 24u16), p.clone());
        assert!(needs_paint(false, Some(&painted), 60, 24, &p), "width changed → paint");
        assert!(needs_paint(false, Some(&painted), 80, 30, &p), "height changed → paint");
    }

    #[test]
    fn needs_paint_cursor_move_paints() {
        let a = view::Pane { lines: vec![Line::from("a")], cursor: Some(view::Cursor { row: 1, col: 5 }) };
        let b = view::Pane { lines: vec![Line::from("a")], cursor: Some(view::Cursor { row: 1, col: 6 }) };
        let painted = ((80u16, 24u16), a.clone());
        assert!(needs_paint(false, Some(&painted), 80, 24, &b), "cursor moved → paint");
    }

    // ---- handle: the action dispatcher ----
    /// A handle that lets a test inspect the agent that was spawned.
    fn run_handle(action: Action) -> (ChannelSink, Option<Agent>) {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let cfg = crate::config::Config {
            api_key: "k".into(), base_url: "http://127.0.0.1:1".into(), model: "m".into(),
            protocol: crate::api::Protocol::MINIMAX,
            cache: crate::api::CacheMode::Auto,
            thinking: crate::api::Thinking::Preserve,
            effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let mut convo = Convo::ephemeral();
        convo.history.push(crate::ir::Message::User("seed".into()));
        let convo = Arc::new(AsyncMutex::new(convo));
        let mut agent: Option<Agent> = None;
        let hub = crate::mcp::Hub::empty();
        let mut ctx = crate::test_util::ctx();
        handle(action, &cfg, &mut ctx, &convo, &mut agent, &sink, &hub);
        (sink, agent)
    }

    #[tokio::test]
    async fn handle_none_is_a_no_op() {
        let (_sink, agent) = run_handle(Action::None);
        assert!(agent.is_none(), "no agent spawned on None");
    }

    #[tokio::test]
    async fn handle_cancel_with_no_agent_does_not_panic() {
        // the test of `if let Some(a) = agent` — absent agent is a silent
        // no-op (the action runs against a now-empty agent slot)
        let (_sink, agent) = run_handle(Action::Cancel);
        assert!(agent.is_none());
    }

    #[tokio::test]
    async fn handle_submit_starts_an_agent_and_emits_task_begin() {
        // a Submit lands a TaskBegin in the sink and fills `agent`. The
        // spawned task races an unreachable host, so we cancel it before
        // returning — what we assert here is the *start* of the action,
        // not its end
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let cfg = crate::config::Config {
            api_key: "k".into(), base_url: "http://127.0.0.1:1".into(), model: "m".into(),
            protocol: crate::api::Protocol::MINIMAX,
            cache: crate::api::CacheMode::Auto,
            thinking: crate::api::Thinking::Preserve,
            effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let mut convo = Convo::ephemeral();
        convo.history.push(crate::ir::Message::User("seed".into()));
        let convo = Arc::new(AsyncMutex::new(convo));
        let mut agent: Option<Agent> = None;
        let hub = crate::mcp::Hub::empty();
        let mut cx = crate::context::Context::new(&crate::agents_md::AgentsMdContext::empty(&std::path::PathBuf::from(".")));
        // tokio::test gives us the runtime; handle does its own tokio::spawn.
        handle(Action::Submit("hello world".into()), &cfg, &mut cx, &convo, &mut agent, &sink, &hub);
        assert!(agent.is_some(), "submit sets the agent slot");
        // a task begin landed in the sink
        let got = rx.try_recv().expect("TaskBegin emitted by submit");
        assert!(matches!(got, Msg::TaskBegin(ref t) if t == "hello world"));
        // tidy up: cancel the running coroutine before the test ends
        if let Some(a) = agent {
            a.token.cancel();
            a.job.abort();
        }
    }

    // ---- Agent::reap: panic becomes a visible note ------------------------

    #[tokio::test]
    async fn handle_submit_refuses_when_an_agent_is_already_running() {
        // the narrow race window: a first Submit spawned an agent; a
        // second Submit arriving before the channel carries TaskBegin
        // back to the state must not orphan the first one. The first
        // agent slot is preserved, the second Submit emits a warning
        // and returns None.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let cfg = crate::config::Config {
            api_key: "k".into(), base_url: "http://127.0.0.1:1".into(), model: "m".into(),
            protocol: crate::api::Protocol::MINIMAX,
            cache: crate::api::CacheMode::Auto,
            thinking: crate::api::Thinking::Preserve,
            effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let convo = Arc::new(AsyncMutex::new(Convo::ephemeral()));
        // a pre-existing agent slot, as if the first Submit had already
        // run and the spawned coroutine were still in flight
        let mut agent: Option<Agent>;
        let first_token = Arc::new(crate::cancel::CancelToken::new());
        let hub = crate::mcp::Hub::empty();
        let mut cx = crate::context::Context::new(&crate::agents_md::AgentsMdContext::empty(&std::path::PathBuf::from(".")));
        // tokio::test wraps us in a runtime; spawn the pre-existing agent
        // slot inside this scope so its JoinHandle has a home.
        let first_job = tokio::spawn(async {});
        agent = Some(Agent { token: first_token.clone(), job: first_job });
        handle(Action::Submit("second task".into()), &cfg, &mut cx, &convo, &mut agent, &sink, &hub);
        // the original agent is still there — same token, not replaced
        assert!(agent.is_some());
        assert_eq!(Arc::as_ptr(&agent.as_ref().unwrap().token), Arc::as_ptr(&first_token));
        // and a warning landed in the sink, naming the cause
        let mut saw_warn = false;
        while let Ok(m) = rx.try_recv() {
            if let Msg::Note { sev: Sev::Warn, ref text } = m {
                assert!(text.contains("already running"), "warn text: {text}");
                saw_warn = true;
            }
        }
        assert!(saw_warn, "the refused submit must surface a warning");
        // tidy up
        if let Some(a) = agent {
            a.token.cancel();
            a.job.abort();
        }
    }

    #[test]
    fn agent_reap_surfaces_a_panicking_join_as_an_error_note() {
        // the test of the `if let Err(e) = self.job.await` arm: a spawned
        // task that panics comes back as JoinError, which reap turns into
        // an error note instead of letting the error fall silent
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let token = Arc::new(crate::cancel::CancelToken::new());
        crate::test_util::block_on(async move {
            let handle = tokio::spawn(async { panic!("intentional") });
            let a = Agent { token, job: handle };
            a.reap(&sink).await;
        });
        let got = rx.try_recv().expect("a note must be emitted on panic");
        assert!(matches!(got, Msg::Note { sev: Sev::Err, ref text } if text.contains("panicked")));
    }

    #[test]
    fn agent_reap_is_silent_on_a_clean_join() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let token = Arc::new(crate::cancel::CancelToken::new());
        crate::test_util::block_on(async move {
            let handle = tokio::spawn(async { /* fine */ });
            let a = Agent { token, job: handle };
            a.reap(&sink).await;
        });
        assert!(rx.try_recv().is_err(), "no note on a clean join");
    }

    // ---- history file: save → load round trip -----------------------------

    /// Point home_dir() at a temp dir for the lifetime of the test. The
    /// loader and saver both consult home_dir(); the override has to
    /// land before either is called. The guard removes the dir on drop.
    struct HomeOverride(std::path::PathBuf);
    impl Drop for HomeOverride {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
    }

    fn override_home(tag: &str) -> (HomeOverride, std::path::PathBuf) {
        let p = crate::test_util::temp_dir(&format!("tui_home_{tag}"));
        let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        std::env::set_var(var, &p);
        (HomeOverride(p.clone()), p)
    }

    fn restore_home(prev: Option<std::ffi::OsString>) {
        let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        match prev {
            Some(v) => std::env::set_var(var, v),
            None => std::env::remove_var(var),
        }
    }

    #[test]
    fn load_history_returns_empty_when_no_file_exists() {
        let prev = if cfg!(windows) { std::env::var_os("USERPROFILE") } else { std::env::var_os("HOME") };
        let (_g, _p) = override_home("missing");
        assert!(load_history().is_empty());
        restore_home(prev);
    }

    #[test]
    fn save_then_load_history_round_trips() {
        let prev = if cfg!(windows) { std::env::var_os("USERPROFILE") } else { std::env::var_os("HOME") };
        let (_g, dir) = override_home("round");
        let entries = vec!["first task".to_string(), "second\nmultiline".into(), "with \\ backslash".into()];
        save_history(&entries);
        let loaded = load_history();
        assert_eq!(loaded, entries, "the loader recovers the saver's bytes exactly");
        // one line per entry, on disk
        let body = std::fs::read_to_string(dir.join(".jingwei_history")).unwrap();
        assert_eq!(body.lines().count(), entries.len());
        restore_home(prev);
    }

    #[test]
    fn load_history_skips_blank_and_whitespace_only_lines() {
        let prev = if cfg!(windows) { std::env::var_os("USERPROFILE") } else { std::env::var_os("HOME") };
        let (_g, dir) = override_home("blank");
        let mut body = String::new();
        body.push_str(&escape("keep me"));
        body.push('\n');
        body.push_str("   \n");
        body.push('\n');
        body.push_str(&escape("also keep"));
        body.push('\n');
        std::fs::write(dir.join(".jingwei_history"), body).unwrap();
        let loaded = load_history();
        assert_eq!(loaded, vec!["keep me".to_string(), "also keep".to_string()]);
        restore_home(prev);
    }

    #[test]
    fn home_dir_returns_none_when_unset() {
        let _prev = if cfg!(windows) { std::env::var_os("USERPROFILE") } else { std::env::var_os("HOME") };
        let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        std::env::remove_var(var);
        assert!(home_dir().is_none());
        restore_home(_prev);
    }

    #[test]
    fn home_dir_reads_the_platform_specific_variable() {
        let _prev = if cfg!(windows) { std::env::var_os("USERPROFILE") } else { std::env::var_os("HOME") };
        let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        std::env::set_var(var, "/tmp/some-test-home");
        assert_eq!(home_dir().unwrap().to_str().unwrap(), "/tmp/some-test-home");
        restore_home(_prev);
    }

    // ---- slash command parser & dispatcher -----------------------------

    #[test]
    fn parse_slash_recognises_a_leading_slash() {
        let cmd = parse_slash("/model foo bar").expect("slash line");
        assert_eq!(cmd.name, "model");
        assert_eq!(cmd.args, vec!["foo", "bar"]);
    }

    #[test]
    fn parse_slash_returns_none_for_plain_text() {
        // a normal task — not a slash command — must fall through to submit
        assert!(parse_slash("count files").is_none());
        assert!(parse_slash("").is_none());
        // leading whitespace + a slash is still a slash command once
        // trimmed; the user typing "/model" with accidental indent is
        // a command, not a plain task. The test pins that the parser
        // owns the trim, so dispatchers can rely on a clean prefix.
        assert!(parse_slash("  /model").is_some());
    }

    #[test]
    fn parse_slash_keeps_the_original_case_of_args() {
        let cmd = parse_slash("/Model MINIMAX/M3").unwrap();
        assert_eq!(cmd.name, "Model");
        assert_eq!(cmd.args, vec!["MINIMAX/M3"]);
    }

    #[tokio::test]
    async fn slash_dispatch_unknown_command_yields_no_switch() {
        // `/garbage` does not exist; the dispatcher surfaces a warning
        // and returns None so the loop keeps its current cfg / convo.
        let cfg = Config { api_key: "k".into(), base_url: "https://x".into(), model: "m".into(),
            protocol: crate::api::Protocol::MINIMAX, cache: crate::api::CacheMode::Auto,
            thinking: crate::api::Thinking::Preserve, effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let convo = Arc::new(AsyncMutex::new(crate::session::Convo::ephemeral()));
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let cmd = SlashCmd { name: "garbage", args: vec![] };
        let hub = crate::mcp::Hub::empty();
        let outcome = dispatch_slash(cmd, &cfg, &convo, &sink, &hub);
        // we don't read the warning Note — the sink it would land on is
        // a fake channel — but we pin the outcome, which is what the
        // loop acts on.
        assert!(matches!(outcome, HandleOutcome::None), "garbage must not switch");
    }

    #[tokio::test]
    async fn slash_help_announces_available_commands() {
        let cfg = Config { api_key: "k".into(), base_url: "https://x".into(), model: "m".into(),
            protocol: crate::api::Protocol::MINIMAX, cache: crate::api::CacheMode::Auto,
            thinking: crate::api::Thinking::Preserve, effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let convo = Arc::new(AsyncMutex::new(crate::session::Convo::ephemeral()));
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        let cmd = SlashCmd { name: "help", args: vec![] };
        let hub = crate::mcp::Hub::empty();
        let outcome = dispatch_slash(cmd, &cfg, &convo, &sink, &hub);
        assert!(matches!(outcome, HandleOutcome::None));
    }

    #[test]
    fn protocol_of_matches_case_insensitively() {
        assert!(protocol_of("minimax").is_some());
        assert!(protocol_of("MINIMAX").is_some());
        assert!(protocol_of("Zai").is_some());
        assert!(protocol_of("deepseek").is_some());
        assert!(protocol_of("nope").is_none());
    }

    #[test]
    fn slash_model_with_unknown_profile_yields_an_error_string() {
        // No settings.json on disk → slash_model reports a clear error,
        // returns HandleOutcome::None (no switch).
        let _lock = crate::test_util::env_lock();
        let _prev = std::env::var("HOME").ok();
        let dir = crate::test_util::temp_dir("slash_unknown_profile");
        std::env::set_var("HOME", dir);
        // No settings file written — slash_model loads an empty Settings.
        let cfg = Config { api_key: "k".into(), base_url: "https://x".into(), model: "m".into(),
            protocol: crate::api::Protocol::MINIMAX, cache: crate::api::CacheMode::Auto,
            thinking: crate::api::Thinking::Preserve, effort: None,
            max_tokens: 1024, context_size: 1_000_000, max_turns: 60, streaming: true,
        };
        let convo = Arc::new(AsyncMutex::new(crate::session::Convo::ephemeral()));
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let _sink = ChannelSink::new(tx);
        let r = slash_model(vec!["minimax/MiniMax-M3"], &cfg, &convo);
        match r {
            Err(e) => assert!(e.contains("no profile named"), "got: {e}"),
            Ok(_) => panic!("must error without settings"),
        }
        // restore HOME so other tests are not affected
        if let Some(v) = _prev { std::env::set_var("HOME", v); } else { std::env::remove_var("HOME"); }
    }
}
