//! TUI view: pure functions of the state. The REPL owns only a small pane at
//! the bottom of the *primary* screen — [`pane`] renders it. The transcript
//! is not a screen the app scrolls: finished rows are rendered once by
//! [`flush_lines`] and appended above the pane into the terminal's own
//! scrollback, where the wheel (and everything that was on screen before
//! jingwei started) already works. [`browse`] renders the Ctrl-O overlay:
//! the one full-screen view, on the alternate screen, that may scroll.
//!
//! Views never touch a terminal, a clock, or a random number; tests assert
//! on the produced rows directly. Layout is two-faced by design ([`Lay`]):
//! what is flushed into the scrollback can never be rewritten, so it
//! truncates — every row exactly one physical row, rendered deterministically
//! once — while the review overlay re-renders every frame and wraps, because
//! a review that hides the end of the line reviews nothing.
//!
//! Styling honors [`crate::display::color_on`]: with NO_COLOR the same
//! structure renders without color, exactly like the plain frontend.

use super::model::{App, Fold, Mode, Row, THINK_TAIL_SHOWN};
use crate::display::{color_on, disp_width, elapsed_str, prompt_w, thought_folded, truncate_cols, wrap_cols, Usage, ERR_BG, FOLD_GUTTER, PROMPT_GUTTER, PROMPT_HEAD, WARN_BG};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// How many body lines a *collapsed* tool tail shows before the "+N" marker.
/// Small on purpose: the folded REPL shows a preview, Ctrl-O (or the plain
/// log's [`crate::plain`] 20-line echo) exists to see the rest.
pub const TOOL_TAIL_SHOWN: usize = 3;

// ---- the status bar's text (moved out of main.rs: view vocabulary, all of it)

/// 26156 → "26.2k"; 2_600_000 → "2.6M"; small counts stay exact.
fn humanize(n: u64) -> String {
    if n < 1000 { format!("{n}") }
    else if n < 1_000_000 { format!("{:.1}k", n as f64 / 1000.0) }
    else { format!("{:.1}M", n as f64 / 1_000_000.0) }
}

/// Session cache hit rate: of everything the model has read, the share
/// that came from cache. `None` until cache traffic exists — an endpoint
/// that never caches would otherwise carry a permanent "0%", which is
/// noise; the first write (cache being built, nothing hit yet) does count,
/// because "0%" then is the truth and the change.
fn cache_pct(total: &Usage) -> Option<u64> {
    let read = total.context_in();
    (read > 0 && (total.cache_read > 0 || total.cache_write > 0))
        .then(|| total.cache_read * 100 / read)
}

/// The status bar's text within `max_w` display columns: the session's
/// ledger, flush right, pure — so tests pin it directly.
///
/// The ledger answers "who am I talking to, how much room is left" —
/// checked occasionally, so it sheds as the pane narrows: model first
/// (the banner already said it), then effort, then the cache rate, then
/// the ctx gauge — with the browse hint pinned (in review, the user is
/// reading history, not checking accounts — the hint is the one thing
/// the bar must still say). When nothing unpinned is left to shed, the
/// bar truncates, keeping its one-physical-row law.
pub fn bar_text(app: &App, max_w: usize) -> String {
    let mut segs = bar_segs(app);
    loop {
        let r = segs.iter().map(|s| s.content.clone()).collect::<Vec<_>>().join(" · ");
        if r.is_empty() {
            return String::new();
        }
        let rw = disp_width(&r);
        if rw <= max_w || segs.iter().all(|s| s.pinned) {
            // flush right: where the ledger has always sat
            return if rw >= max_w { truncate_cols(&r, max_w) } else { format!("{}{r}", " ".repeat(max_w - rw)) };
        }
        let i = segs.iter().position(|s| !s.pinned).expect("an unpinned segment exists");
        segs.remove(i);
    }
}

/// One ledger segment: its text, and whether it may be shed when the pane
/// narrows. Pinned: the browse hint (in review, the user is reading
/// history, not checking accounts — the hint is the one thing the bar must
/// still say).
struct Seg {
    content: String,
    pinned: bool,
}

/// The bar's segments (see [`bar_text`]). They appear as their data
/// arrives: no cache traffic — no cache rate; no request read yet — no
/// ctx gauge; unknown limits (the default `Info`) — no ctx gauge either.
fn bar_segs(app: &App) -> Vec<Seg> {
    let st = &app.status;
    let i = &app.info;
    let mut right = vec![];
    if !i.model.is_empty() { right.push(Seg { content: i.model.clone(), pinned: false }); }
    if let Some(e) = &i.effort { right.push(Seg { content: format!("effort {e}"), pinned: false }); }
    if let Some(p) = cache_pct(&st.total) {
        right.push(Seg { content: format!("cache {p}%"), pinned: false });
    }
    if i.context_limit > 0 && st.turn.context_in() > 0 {
        right.push(Seg { content: format!("ctx {}/{}", humanize(st.turn.context_in()), humanize(i.context_limit)), pinned: false });
    }
    if app.mode == Mode::Browse {
        right.push(Seg { content: "browse: ↑↓ PgUp/PgDn g/G · Ctrl-O returns".into(), pinned: true });
    }
    right
}

/// The inline pane: the only part of the primary screen jingwei draws.
///
/// ```text
/// (streaming tail — only while a line is in flight)
/// ───────────────────────────── rule
/// jingwei ❯ the input, one row per line while composing
/// ───────────────────────────── rule
/// the ledger bar, flush right
/// ```
#[derive(Debug, Clone)]
pub struct Pane {
    pub lines: Vec<Line<'static>>,
    /// The terminal cursor: row within the pane (0-based) and column, on
    /// the input row the caret sits on; `None` while a task runs.
    pub cursor: Option<Cursor>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cursor {
    pub row: u16,
    pub col: u16,
}

/// The Ctrl-O overlay: a full-screen unfolded review of the whole session.
#[derive(Debug)]
pub struct Screen {
    pub lines: Vec<Line<'static>>,
}

impl Screen {
    /// The plain text of each row, for assertions.
    #[cfg(test)]
    pub fn texts(&self) -> Vec<String> {
        self.lines.iter().map(|l| l.spans.iter().map(|s| s.content.clone()).collect::<String>()).collect()
    }
}

/// One dim separator rule spanning the width — the pane's frame.
fn rule(w: usize) -> Line<'static> {
    Line::from(Span::styled("─".repeat(w), dim()))
}

/// Render the pane `w` columns wide inside a screen `h` rows tall (the
/// input block is capped to what fits). Row count varies with the state:
/// the tail row appears while text streams, the input block grows one row
/// per line being composed — and within a task the count only ever grows:
/// blank padding holds the pane at its high-water mark
/// ([`Status::pane_floor`](super::model::Status::pane_floor)), so the
/// status bar keeps its row while folds swap tail rows for reserve.
pub fn pane(app: &App, w: u16, h: u16) -> Pane {
    let w = w.max(8) as usize;
    let mut lines: Vec<Line<'static>> = vec![];
    think_tail(app, w, Lay::Flush, &mut lines);
    if !app.partial.is_empty() {
        lines.push(Line::from(truncate_cols(&app.partial, w)));
    }
    for _ in lines.len()..app.status.pane_floor {
        lines.push(Line::from(""));
    }
    lines.push(rule(w));

    let cursor = match &app.status.task {
        // while a task runs, one locked row shows the task; no caret
        Some(t) => {
            let mut text = t.text.as_str();
            let more = t.text.contains('\n');
            if more {
                text = t.text.split('\n').next().unwrap_or("");
            }
            let shown = truncate_cols(text, w.saturating_sub(prompt_w()));
            let label = if more { format!("{shown} …") } else { shown.to_string() };
            lines.push(prompt_line(&label, dim()));
            None
        }
        None => {
            // leave room for tail + two rules + the bar
            let cap = (h as usize).saturating_sub(lines.len() + 2).max(1);
            let (rows, caret_row, caret_col) = input_rows(app, w, cap);
            let caret_row = lines.len() + caret_row;
            lines.extend(rows);
            Some(Cursor { row: caret_row as u16, col: caret_col as u16 })
        }
    };

    lines.push(rule(w));
    lines.push(status_row(app, w));
    Pane { lines, cursor }
}

/// The prompt row: `jingwei ❯ <label>`, continuation rows indented by the
/// same gutter width.
fn prompt_line(label: &str, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(PROMPT_HEAD, prompt_style()),
        Span::styled(PROMPT_GUTTER, gutter_style()),
        Span::styled(label.to_string(), style),
    ])
}

/// The input block: one row per line of the draft (the prompt gutter only
/// on the first), horizontally scrolled on the caret's row, vertically
/// windowed to `cap` rows with the caret kept visible. Returns the rows,
/// the caret's row within the block, and its column.
fn input_rows(app: &App, w: usize, cap: usize) -> (Vec<Line<'static>>, usize, usize) {
    let text = &app.input.text;
    let i = app.input.caret();
    let lines: Vec<&str> = text.split('\n').collect();
    let line_idx = text[..i].matches('\n').count();
    let line_start = text[..i].rfind('\n').map(|p| p + 1).unwrap_or(0);
    let off = i - line_start; // byte offset of the caret within its line

    // the vertical window: the last `cap` rows, slid up to keep the caret
    let start = if lines.len() <= cap { 0 } else { (lines.len() - cap).min(line_idx) };

    let avail = w.saturating_sub(prompt_w());
    let mut rows = vec![];
    let mut caret_row = 0;
    let mut caret_col = prompt_w();
    for (j, l) in lines.iter().enumerate().skip(start).take(cap) {
        let head = if j == 0 { Span::styled(PROMPT_HEAD, prompt_style()) } else { Span::raw(" ".repeat(prompt_w())) };
        let gutter = if j == 0 { Span::styled(PROMPT_GUTTER, gutter_style()) } else { Span::raw("") };
        if j == line_idx {
            let (shown, col) = line_window(l, off, avail);
            caret_row = j - start;
            caret_col = prompt_w() + col;
            rows.push(Line::from(vec![head, gutter, Span::raw(shown)]));
        } else {
            rows.push(Line::from(vec![head, gutter, Span::raw(truncate_cols(l, avail))]));
        }
    }
    (rows, caret_row, caret_col)
}

/// The caret's row, horizontally windowed: when the text before the caret
/// is wider than the row, drop whole graphemes from the front so the caret
/// stays visible. Returns the shown text and the caret's column within it.
fn line_window(line: &str, off: usize, avail: usize) -> (String, usize) {
    use unicode_segmentation::UnicodeSegmentation;
    let before = &line[..off.min(line.len())];
    if disp_width(before) <= avail.saturating_sub(1) {
        let col = disp_width(before);
        return (truncate_cols(line, avail), col);
    }
    let mut start = off;
    let mut width = 0;
    for (s, g) in line[..off].grapheme_indices(true).rev() {
        let gw = disp_width(g).max(1);
        if width + gw > avail.saturating_sub(4) {
            break;
        }
        width += gw;
        start = s;
    }
    (format!("…{}", truncate_cols(&line[start..], avail.saturating_sub(1))), width + 1)
}

/// How a row may lay out its text. The same row renders under both:
/// truncation is the scrollback's ruler — what is flushed is immutable, so
/// it must be exactly one physical row, deterministically — while the
/// review overlay re-renders every frame and owes the reader the whole
/// line. One enum, because the two modes are one fact with two faces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lay {
    /// Truncate with an ellipsis: one row, printed once, never rewritten.
    Flush,
    /// Wrap: reading beats row counts where rewriting is free.
    Review,
}

/// Render transcript rows `from..` as they should land in the scrollback.
/// The caller only ever advances `from` past rows already flushed, so each
/// row is rendered exactly once, with the fold state it has right now —
/// what the ratchet has opened by then prints open, later folds print as
/// markers.
pub fn flush_lines(app: &App, w: usize, from: usize) -> Vec<Line<'static>> {
    let mut out = vec![];
    for row in &app.rows[from.min(app.rows.len())..] {
        row_lines(row, w, &mut out, Lay::Flush);
    }
    out
}

/// Render the Ctrl-O overlay `w`×`h`: every fold open, the window over the
/// rendered rows offset by the scroll position, the hint bar at the bottom.
pub fn browse(app: &App, w: u16, h: u16) -> Screen {
    let w = w.max(8) as usize;
    let body = render_rows(app, w);
    let area_h = (h as usize).saturating_sub(1);

    // The visible window: the last `area_h` rendered rows, offset by the
    // scroll position (rows scrolled up from the bottom). `Scroll::TOP`
    // means "top" — clamped against the actual content.
    let scroll = app.scroll.offset().min(body.len().saturating_sub(area_h));
    let end = body.len() - scroll;
    let start = end.saturating_sub(area_h);
    let mut lines: Vec<Line<'static>> = body[start..end].to_vec();
    while lines.len() < area_h {
        lines.push(Line::from(""));
    }
    lines.push(status_row(app, w));
    Screen { lines }
}

/// Flatten the transcript rows into rendered lines, expanding open folds.
/// One transcript row may render as several lines (a fold's body); the
/// scroll offset counts *rendered* lines, so what you see is what scrolls.
fn render_rows(app: &App, w: usize) -> Vec<Line<'static>> {
    let mut out = vec![];
    for row in &app.rows {
        row_lines(row, w, &mut out, Lay::Review);
    }
    if !app.partial.is_empty() {
        for chunk in wrap_cols(&app.partial, w) {
            out.push(Line::from(chunk));
        }
    }
    think_tail(app, w, Lay::Review, &mut out);
    out
}

/// Render one transcript row (with its fold, if any) into `out`.
fn row_lines(row: &Row, w: usize, out: &mut Vec<Line<'static>>, lay: Lay) {
    match row {
        // a multi-line task echoes as one row per line, the gutter
        // aligning continuation lines under the prompt; under Review a
        // wrapped chunk of the task aligns there too
        Row::Task(t) => {
            let avail = w.saturating_sub(prompt_w());
            for (j, l) in t.split('\n').enumerate() {
                match lay {
                    Lay::Flush => {
                        let (head, gutter) = task_spans(j == 0);
                        out.push(Line::from(vec![head, gutter, Span::raw(truncate_cols(l, avail))]));
                    }
                    Lay::Review => {
                        for (k, chunk) in wrap_cols(l, avail).into_iter().enumerate() {
                            let (head, gutter) = task_spans(j == 0 && k == 0);
                            out.push(Line::from(vec![head, gutter, Span::raw(chunk)]));
                        }
                    }
                }
            }
        }
        Row::Line(l) => flat(l, w, lay, out, Style::default()),
        Row::Sep => out.push(Line::from("")),
        // the fold travels inside the row: its kind is the variant arm the
        // renderer is already in — there is nothing to disagree with
        Row::Thought(f) => thought_lines(f, w, lay, out),
        Row::Tool { head, fold } => {
            // the head carries the fold-state glyph, like a thought marker:
            // one left rail of ▸/▾ down the transcript, scan it to read
            // the whole fold tree's state
            out.push(fold_head(head, fold.expanded, w));
            tool_lines(fold, w, lay, out);
        }
        Row::Note(sev, text) => {
            let (bg, err) = match sev {
                crate::display::Sev::Warn => (WARN_BG, false),
                crate::display::Sev::Err => (ERR_BG, true),
            };
            let style = note_style(bg, err);
            match lay {
                Lay::Flush => out.push(Line::from(Span::styled(truncate_cols(text, w), style))),
                Lay::Review => for chunk in wrap_cols(text, w) {
                    out.push(Line::from(Span::styled(chunk, style)));
                },
            }
        }
    }
}

/// The task row's two lead spans: the prompt on the first physical row,
/// blank alignment everywhere a continuation lands under it.
fn task_spans(first: bool) -> (Span<'static>, Span<'static>) {
    if first {
        (Span::styled(PROMPT_HEAD, prompt_style()), Span::styled(PROMPT_GUTTER, gutter_style()))
    } else {
        (Span::raw(" ".repeat(prompt_w())), Span::raw(""))
    }
}

/// A flat (unguttered) row's content: one truncated row when flushed, one
/// row per wrapped chunk under review.
fn flat(l: &str, w: usize, lay: Lay, out: &mut Vec<Line<'static>>, style: Style) {
    match lay {
        Lay::Flush => out.push(Line::from(Span::styled(truncate_cols(l, w), style))),
        Lay::Review => for chunk in wrap_cols(l, w) {
            out.push(Line::from(Span::styled(chunk, style)));
        },
    }
}

/// A reasoning fold: its marker when closed, marker + full body when open.
/// The open marker splits in two — the title is a header, its metadata a
/// footnote — so the eye lands on "thought #3" and skips the counts. The
/// body hangs from the shared fold gutter and leans italic: reasoning is
/// the model's asides, not the answer.
fn thought_lines(f: &Fold, w: usize, lay: Lay, out: &mut Vec<Line<'static>>) {
    if f.expanded {
        let n = f.lines();
        let meta = format!(" · {n} line{}", if n == 1 { "" } else { "s" });
        let meta = match f.duration {
            Some(d) => format!("{meta} · {}", elapsed_str(d)),
            None => meta,
        };
        let title_w = disp_width(&format!("▾ thought #{}", f.n));
        out.push(Line::from(vec![
            Span::styled(truncate_cols(&format!("▾ thought #{}", f.n), w), head_style()),
            Span::styled(truncate_cols(&meta, w.saturating_sub(title_w)), dim()),
        ]));
        for l in f.body.lines() {
            guttered(l, w, lay, thought_style(), out);
        }
    } else {
        let mut all = f.body.lines();
        let first = all.next().unwrap_or("");
        let marker = thought_folded(f.n, first, all.count());
        out.push(Line::from(Span::styled(truncate_cols(&marker, w), dim())));
    }
}

/// A tool fold: the call is visible even folded — a short tail preview,
/// the full output when open. Bodies hang from the same gutter as
/// thoughts; the output is fact, so it stays upright (no italic).
fn tool_lines(f: &Fold, w: usize, lay: Lay, out: &mut Vec<Line<'static>>) {
    let all: Vec<&str> = f.body.lines().collect();
    if f.expanded {
        for l in all {
            guttered(l, w, lay, dim(), out);
        }
    } else {
        let shown = all.len().min(TOOL_TAIL_SHOWN);
        for l in &all[..shown] {
            guttered(l, w, lay, dim(), out);
        }
        if all.len() > shown {
            guttered(&format!("… +{} more lines", all.len() - shown), w, lay, dim(), out);
        }
    }
}

/// A fold's head line: the state glyph on the rail, the head text beside
/// it. The glyph's style *is* the state — dim when there is more hidden,
/// bold when the body is open below.
fn fold_head(head: &str, expanded: bool, w: usize) -> Line<'static> {
    let glyph = if expanded { "▾" } else { "▸" };
    let gstyle = if expanded { head_style() } else { dim() };
    Line::from(vec![
        Span::styled(glyph, gstyle),
        Span::raw(" "),
        Span::styled(truncate_cols(head, w.saturating_sub(2)), head_style()),
    ])
}

/// The live reasoning block: reasoning streams for seconds, and a frozen
/// pane cannot tell "thinking" from "hung". So the pane (and the review
/// overlay) show the block as it arrives — the number it will fold into,
/// how long it has been running, and the tail that still moves. It never
/// lands in the scrollback: ThinkEnd folds it once, with its final state.
fn think_tail(app: &App, w: usize, lay: Lay, out: &mut Vec<Line<'static>>) {
    if app.think_buf.is_empty() {
        return;
    }
    let all: Vec<&str> = app.think_buf.lines().collect();
    let n = all.len().max(1);
    let mut head = format!("◌ thought #{} · {n} line{}", app.next_thought_n(), if n == 1 { "" } else { "s" });
    if let Some(d) = app.thinking_for() {
        head.push_str(&format!(" · {}", elapsed_str(d)));
    }
    out.push(Line::from(Span::styled(truncate_cols(&head, w), dim())));
    for l in all.iter().rev().take(THINK_TAIL_SHOWN).rev() {
        guttered(l, w, lay, thought_style(), out);
    }
}

/// One body line of a fold: the shared gutter, then the content — the
/// structure that survives a DIM-blind terminal. Under Review a wrapped
/// chunk keeps hanging from the gutter, so the block stays a block.
fn guttered(l: &str, w: usize, lay: Lay, style: Style, out: &mut Vec<Line<'static>>) {
    let avail = w.saturating_sub(disp_width(FOLD_GUTTER));
    match lay {
        Lay::Flush => out.push(Line::from(vec![
            Span::styled(FOLD_GUTTER, gutter_style()),
            Span::styled(truncate_cols(l, avail), style),
        ])),
        Lay::Review => for chunk in wrap_cols(l, avail) {
            out.push(Line::from(vec![
                Span::styled(FOLD_GUTTER, gutter_style()),
                Span::styled(chunk, style),
            ]));
        },
    }
}

/// The status bar row: [`bar_text`]'s ledger, dim, truncated to one
/// physical row. Empty string when there is nothing to say.
fn status_row(app: &App, w: usize) -> Line<'static> {
    let bar = bar_text(app, w);
    if bar.is_empty() {
        Line::from("")
    } else {
        Line::from(Span::styled(bar, dim()))
    }
}

// ---- styles: one gate for color, so NO_COLOR covers the TUI like the log --

fn prompt_style() -> Style {
    if color_on() { Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD) } else { Style::default().add_modifier(Modifier::BOLD) }
}

fn gutter_style() -> Style {
    if color_on() { Style::default().fg(Color::DarkGray) } else { Style::default() }
}

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

/// Reasoning renders as an aside: dimmed *and* italic — a second style
/// axis, not a second hue, so the palette stays at two foreground colors.
fn thought_style() -> Style {
    Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC)
}

fn head_style() -> Style {
    if color_on() { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().add_modifier(Modifier::BOLD) }
}

fn note_style(bg: (u8, u8, u8), err: bool) -> Style {
    if !color_on() {
        return if err { Style::default().add_modifier(Modifier::BOLD) } else { Style::default() };
    }
    let mut s = Style::default().bg(Color::Rgb(bg.0, bg.1, bg.2));
    if err {
        s = s.fg(Color::White).add_modifier(Modifier::BOLD);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::super::model::App;
    use crate::display::Usage;
    use super::super::update::{update, Ev, Action};
    use super::*;
    use crate::display::Msg;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, KeyEventKind, KeyEventState};

    /// The width bar_text was pinned against before it learned the pane's.
    const BAR_W: usize = 76;

    fn key(code: KeyCode, mods: KeyModifiers) -> Ev {
        Ev::Key(KeyEvent { code, modifiers: mods, kind: KeyEventKind::Press, state: KeyEventState::NONE })
    }

    fn type_str(a: &mut App, s: &str) {
        for c in s.chars() {
            update(a, key(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    fn texts_of(lines: &[Line<'static>]) -> Vec<String> {
        lines.iter().map(|l| l.spans.iter().map(|s| s.content.clone()).collect::<String>()).collect()
    }

    fn seeded() -> App {
        let mut a = App::new();
        for ev in [
            Ev::Msg(Msg::TaskBegin("count files".into())),
            Ev::Msg(Msg::Think("line one\nline two\nline three".into())),
            Ev::Msg(Msg::ThinkEnd),
            Ev::Msg(Msg::Text("found 3 files".into())),
            Ev::Msg(Msg::Done),
            Ev::Msg(Msg::TaskEnd),
        ] {
            update(&mut a, ev);
        }
        a
    }

    // ---- the status bar's text ----

    /// An app whose static facts are known: model, effort, both limits.
    fn informed(a: &mut App) -> &mut App {
        a.info = super::super::model::Info {
            model: "MiniMax-M3".into(),
            effort: Some("high".into()),
            context_limit: 1_000_000,
        };
        a
    }

    #[test]
    fn bar_segments_appear_as_data_arrives() {
        // nothing known, nothing happening: the bar says nothing
        assert_eq!(bar_text(&App::new(), BAR_W), "");
        // identity arrives with the frontend's facts — the ledger flushes
        // right, and with no traffic yet it is all there is to say
        let mut a = App::new();
        informed(&mut a);
        assert_eq!(bar_text(&a, BAR_W), format!("{}MiniMax-M3 · effort high", " ".repeat(BAR_W - disp_width("MiniMax-M3 · effort high"))));
        // traffic without cache: still no cache segment (a permanent
        // "cache 0%" on non-caching endpoints is noise) — and the session
        // totals are not bar material at all
        a.status.total = Usage { input: 4600, output: 336, ..Usage::default() };
        assert!(!bar_text(&a, BAR_W).contains("cache"), "{}", bar_text(&a, BAR_W));
        assert!(!bar_text(&a, BAR_W).contains("total"), "{}", bar_text(&a, BAR_W));
        a.status.total = Usage { input: 26_156, output: 336, cache_read: 25_344, cache_write: 1188 };
        let bar = bar_text(&a, BAR_W);
        assert!(bar.contains("cache 48%"), "25344/(26156+25344+1188) ≈ 48: {bar}");
        assert!(!bar.contains("total"), "session totals left the bar: {bar}");
        // a finished request sizes the next one: the ctx gauge reads the
        // last request's context against the limit
        a.status.turn = Usage { input: 2000, cache_read: 43_000, ..Usage::default() };
        assert!(bar_text(&a, BAR_W).contains("ctx 45.0k/1.0M"), "{}", bar_text(&a, BAR_W));
    }

    #[test]
    fn bar_sheds_identity_then_cache_then_ctx() {
        let mut a = App::new();
        informed(&mut a);
        a.status.turn = Usage { input: 45_000, output: 900, ..Usage::default() };
        a.status.total = Usage { input: 45_000, output: 900, cache_read: 20_000, ..Usage::default() };

        // wide: everything, flush right
        let wide = bar_text(&a, 120);
        assert!(wide.contains("MiniMax-M3") && wide.contains("effort high")
            && wide.contains("cache 30%") && wide.contains("ctx 45.0k/1.0M"), "{wide}");
        assert!(wide.starts_with(' '), "flush right: {wide}");

        // narrower: identity sheds first — the banner already said it
        let mid = bar_text(&a, 50);
        assert!(!mid.contains("MiniMax-M3"), "model shed: {mid}");
        assert!(mid.contains("effort high") && mid.contains("cache 30%")
            && mid.contains("ctx 45.0k/1.0M"), "{mid}");

        // narrower still: effort, then the cache rate — the ctx gauge is
        // the ledger's last survivor, the one reading that changes what
        // you do next
        let tighter = bar_text(&a, 35);
        assert!(!tighter.contains("effort"), "effort sheds next: {tighter}");
        assert!(tighter.contains("cache 30%") && tighter.contains("ctx"), "{tighter}");
        let tight = bar_text(&a, 20);
        assert!(!tight.contains("cache"), "cache sheds before ctx: {tight}");
        assert!(tight.contains("ctx 45.0k/1.0M"), "{tight}");

        // nothing left to shed: the bar says nothing rather than lie
        assert_eq!(bar_text(&a, 10), "");

        // the one-physical-row law holds through every width
        for w in [10, 20, 35, 50, 120] {
            assert!(disp_width(&truncate_cols(&bar_text(&a, w), w)) <= w, "w={w}");
        }
    }

    #[test]
    fn pane_pads_to_its_floor_so_the_bar_holds_its_row_through_a_fold() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::TaskBegin("t".into())));
        update(&mut a, Ev::Msg(Msg::Think("l1\nl2\nl3\nl4".into())));
        let streaming = texts_of(&pane(&a, 80, 24).lines);
        assert_eq!(streaming.len(), 8, "head + 3 tail rows + the 4-row frame: {streaming:?}");
        assert!(streaming[0].starts_with('◌'), "{streaming:?}");

        // the fold swaps tail rows for blank padding: same height, the
        // bar never moves — this is the fix for the blank line that used
        // to open under the bar each time a thought folded
        update(&mut a, Ev::Msg(Msg::ThinkEnd));
        let folded = texts_of(&pane(&a, 80, 24).lines);
        assert_eq!(folded.len(), 8, "padded to the floor: {folded:?}");
        assert_eq!(&folded[..4], vec!["", "", "", ""], "blank reserve above the rule: {folded:?}");
        assert!(folded[4].starts_with("─────"), "the frame follows: {folded:?}");

        // idle keeps the padded pane (the bar keeps its row at the
        // bottom); the next task starts tight again
        update(&mut a, Ev::Msg(Msg::TaskEnd));
        assert_eq!(pane(&a, 80, 24).lines.len(), 8, "idle keeps the reserve");
        update(&mut a, Ev::Msg(Msg::TaskBegin("next".into())));
        assert_eq!(pane(&a, 80, 24).lines.len(), 4, "each task starts tight");
    }

    #[test]
    fn browse_hint_rides_the_ledgers_tail() {
        let mut a = App::new();
        informed(&mut a);
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let bar = bar_text(&a, 120);
        assert!(bar.ends_with("browse: ↑↓ PgUp/PgDn g/G · Ctrl-O returns"), "{bar}");
        // pinned: it survives widths that shed the whole ledger before it
        let narrow = bar_text(&a, 48);
        assert!(narrow.contains("Ctrl-O returns"), "{narrow}");
    }

    // ---- the transcript ----

    #[test]
    fn flushed_rows_fold_thoughts_and_tool_tails() {
        let a = seeded();
        let texts = texts_of(&flush_lines(&a, 80, 0));
        assert!(texts.iter().any(|t| t.contains("▸ thought #1 · line one … +2")), "{texts:?}");
        assert!(texts.iter().any(|t| t == "found 3 files"));
        // the folded marker previews the first line and hides the rest
        assert!(texts.iter().all(|t| !t.contains("line two") && !t.contains("line three")),
            "body stays folded behind the preview: {texts:?}");
    }

    #[test]
    fn open_thought_marker_splits_title_from_metadata_and_carries_duration() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::TaskBegin("t".into())));
        update(&mut a, Ev::Msg(Msg::Think("only line".into())));
        for _ in 0..25 {
            update(&mut a, Ev::Tick(super::super::model::TICK));
        }
        update(&mut a, Ev::Msg(Msg::ThinkEnd));
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let row = browse(&a, 80, 10).lines.into_iter().find(|l| {
            l.spans.iter().any(|sp| sp.content.contains("thought #1"))
        }).expect("marker rendered");
        // title span carries no metadata; the metadata span carries the
        // count (singular) and the measured duration
        let title = &row.spans[0];
        assert_eq!(title.content, "▾ thought #1");
        assert!(title.style.add_modifier.contains(ratatui::style::Modifier::BOLD), "title is a header");
        let meta = row.spans.iter().map(|s| s.content.clone()).collect::<String>();
        assert!(meta.contains("1 line · 3s"), "singular, with duration: {meta}");
    }

    #[test]
    fn flush_renders_only_the_suffix_once() {
        let mut a = seeded();
        for i in 0..40 {
            update(&mut a, Ev::Msg(Msg::Text(format!("row {i}\n"))));
        }
        let first = texts_of(&flush_lines(&a, 80, 0));
        assert_eq!(first.len(), a.rows.len(), "from 0 renders every row exactly once");
        let delta = texts_of(&flush_lines(&a, 80, a.rows.len() - 2));
        assert_eq!(delta.len(), 2, "only unflushed rows render: {delta:?}");
        assert!(delta.iter().any(|t| t == "row 39"));
    }

    #[test]
    fn browse_shows_bodies_and_new_folds_arrive_closed() {
        let mut a = seeded();
        let flushed_then = a.rows.len();
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = browse(&a, 80, 24).texts();
        assert!(texts.iter().any(|t| t.contains("▾ thought #1 · 3 lines")));
        assert!(texts.iter().any(|t| t == "│ line one"), "browse shows the body on the gutter");
        assert_eq!(texts.len(), 24, "browse fills the screen");
        assert!(texts.last().unwrap().contains("Ctrl-O returns"));
        // back to the REPL: the scrollback is immutable; rows that arrive
        // afterwards print as markers again until the next Ctrl-O
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        update(&mut a, Ev::Msg(Msg::Think("after\n".into())));
        update(&mut a, Ev::Msg(Msg::ThinkEnd));
        let delta = texts_of(&flush_lines(&a, 80, flushed_then));
        assert!(delta.iter().any(|t| t.contains("thought #2")), "{delta:?}");
        assert!(delta.iter().all(|t| t != "after"), "new folds arrive closed: {delta:?}");
    }

    #[test]
    fn collapsed_tool_shows_short_tail_expanded_shows_all() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Tool {
            name: "bash".into(),
            summary: "$ seq 1 10".into(),
            output: (1..=10).map(|i| i.to_string()).collect::<Vec<_>>().join("\n"),
        }));
        let texts = texts_of(&flush_lines(&a, 80, 0));
        assert!(texts.iter().any(|t| t == "▸ [bash] $ seq 1 10"), "head carries the fold glyph: {texts:?}");
        assert!(texts.iter().any(|t| t == "│ 1"), "preview rides the gutter: {texts:?}");
        assert!(texts.iter().any(|t| t == "│ … +7 more lines"), "{texts:?}");
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = browse(&a, 80, 24).texts();
        assert!(texts.iter().any(|t| t == "│ 10"), "expanded tail reaches the end");
        assert!(texts.iter().all(|t| !t.contains("more lines")));
        assert!(texts.iter().any(|t| t == "▾ [bash] $ seq 1 10"), "glyph flips with the fold");
    }

    #[test]
    fn pane_is_framed_by_rules_tail_input_bar() {
        let mut a = App::new();
        let p = pane(&a, 80, 24);
        let texts = texts_of(&p.lines);
        assert_eq!(texts.len(), 4, "rule, input, rule, bar: {texts:?}");
        assert!(texts[0].starts_with("─────"), "top rule: {texts:?}");
        assert!(texts[1].starts_with("jingwei ❯ "), "input row carries the prompt");
        assert!(texts[2].starts_with("─────"), "bottom rule: {texts:?}");
        assert_eq!(p.cursor, Some(Cursor { row: 1, col: prompt_w() as u16 }));

        update(&mut a, Ev::Msg(Msg::Text("partial tail".into())));
        let p = pane(&a, 80, 24);
        let texts = texts_of(&p.lines);
        assert_eq!(texts[0], "partial tail", "the streamed tail rides above the top rule");
        assert_eq!(p.cursor.as_ref().unwrap().row, 2, "caret row follows the tail row");

        update(&mut a, Ev::Msg(Msg::TaskBegin("busy\nsecond line".into())));
        let p = pane(&a, 80, 24);
        let texts = texts_of(&p.lines);
        let locked = &texts[texts.len() - 3];
        assert!(locked.starts_with("jingwei ❯ busy …"), "multi-line task locks one row: {locked:?}");
        assert!(p.cursor.is_none(), "no caret while the row is locked");
    }

    #[test]
    fn multi_line_draft_renders_one_row_per_line_with_caret() {
        let mut a = App::new();
        type_str(&mut a, "one");
        update(&mut a, key(KeyCode::Char('j'), KeyModifiers::CONTROL));
        type_str(&mut a, "tw");
        let p = pane(&a, 40, 24);
        let texts = texts_of(&p.lines);
        assert_eq!(texts[1], "jingwei ❯ one", "first line carries the prompt: {texts:?}");
        assert_eq!(texts[2], format!("{}tw", " ".repeat(prompt_w())), "continuation aligns under the prompt");
        assert_eq!(p.cursor, Some(Cursor { row: 2, col: (prompt_w() + 2) as u16 }));
    }

    #[test]
    fn input_block_windows_to_fit_the_screen_and_keeps_the_caret() {
        let mut a = App::new();
        for i in 0..30 {
            type_str(&mut a, &format!("l{i}"));
            update(&mut a, key(KeyCode::Char('j'), KeyModifiers::CONTROL));
        }
        let p = pane(&a, 40, 10);
        let texts = texts_of(&p.lines);
        let block = &texts[1..texts.len() - 2];
        assert!(block.len() < 31, "the block is capped: {block:?}");
        assert_eq!(block[block.len() - 2].trim(), "l29", "the tail of the draft is visible: {block:?}");
        assert_eq!(block.last().unwrap().trim(), "", "the caret rides the trailing empty row: {block:?}");
        assert_eq!(p.cursor.unwrap().row as usize, 1 + block.len() - 1, "caret row is inside the block");
        // walk the caret to the very front: the window follows it up
        a.input.cursor = 0;
        let texts = texts_of(&pane(&a, 40, 10).lines);
        let block = &texts[1..texts.len() - 2];
        assert!(block[0].ends_with("l0"), "the window slid up to the caret: {block:?}");
    }

    #[test]
    fn pane_input_row_tracks_cursor_for_latins_and_cjk() {
        let mut a = App::new();
        type_str(&mut a, "精卫 fill");
        let p = pane(&a, 40, 24);
        assert_eq!(p.cursor, Some(Cursor { row: 1, col: (prompt_w() + disp_width("精卫 fill")) as u16 }));
        // walk back over a wide char: the column drops by 2
        update(&mut a, key(KeyCode::Left, KeyModifiers::NONE));
        let p = pane(&a, 40, 24);
        assert_eq!(p.cursor, Some(Cursor { row: 1, col: (prompt_w() + disp_width("精卫 fil")) as u16 }));
    }

    #[test]
    fn pane_bar_is_the_ledger() {
        let mut a = seeded();
        informed(&mut a);
        update(&mut a, Ev::Msg(Msg::Usage(Usage { input: 100, output: 7, ..Usage::default() })));
        let texts = texts_of(&pane(&a, 80, 24).lines);
        let bar = texts.last().unwrap();
        assert!(bar.contains("ctx 100/1.0M"), "the finished request sizes the gauge: {bar}");
        assert!(!bar.contains('#'), "no step counter: {bar}");
        assert!(!bar.contains("total"), "session totals are not bar material: {bar}");
        // the bar ignores task state entirely: working changes nothing it says
        let before = bar.clone();
        update(&mut a, Ev::Msg(Msg::TaskBegin("run".into())));
        update(&mut a, Ev::Tick(super::super::model::TICK));
        let after = texts_of(&pane(&a, 80, 24).lines).pop().unwrap();
        assert_eq!(after, before, "a running task leaves the ledger alone");
    }

    #[test]
    fn browse_scrolling_reveals_older_rows_and_clamps() {
        let mut a = seeded();
        for i in 0..40 {
            update(&mut a, Ev::Msg(Msg::Text(format!("row {i}\n"))));
        }
        let bottom = browse(&a, 40, 6).texts();
        assert!(bottom.iter().any(|t| t == "row 39"), "{bottom:?}");
        update(&mut a, key(KeyCode::PageUp, KeyModifiers::NONE)); // enter browse, +10
        update(&mut a, key(KeyCode::PageUp, KeyModifiers::NONE));
        let top = browse(&a, 40, 6).texts();
        assert!(top.iter().any(|t| t == "row 19"), "window moved up: {top:?}");
        assert!(!top.iter().any(|t| t == "row 38"), "scrolled away from the tail: {top:?}");
        // g clamps to top; Scroll::TOP never overflows the window math
        update(&mut a, key(KeyCode::Char('g'), KeyModifiers::NONE));
        let _ = browse(&a, 40, 6); // must not panic
    }

    #[test]
    fn flushed_lines_truncate_rather_than_wrap() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Text(format!("{}\n", "x".repeat(200)))));
        for l in flush_lines(&a, 40, 0) {
            let t = l.spans.iter().map(|s| s.content.clone()).collect::<String>();
            assert!(disp_width(&t) <= 40);
        }
    }

    #[test]
    fn review_wraps_what_flush_must_truncate() {
        let long = format!("{} {}", "word".repeat(30), "精卫填海");
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Think(format!("{long}\n"))));
        update(&mut a, Ev::Msg(Msg::ThinkEnd));
        update(&mut a, Ev::Msg(Msg::Text(format!("{long}\n"))));

        // flush: one physical row, cut with an ellipsis — the scrollback's ruler
        let flushed = texts_of(&flush_lines(&a, 40, 0));
        assert!(flushed.iter().any(|t| t.starts_with("word") && t.ends_with('…')), "{flushed:?}");
        assert_eq!(flushed.len(), a.rows.len(), "one row each, still");

        // review: the whole line is readable, every chunk fits, wrapped
        // body chunks stay on the fold's gutter
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = browse(&a, 40, 30).texts();
        assert!(texts.iter().all(|t| disp_width(t) <= 40), "{texts:?}");
        let rejoined = texts.iter()
            .filter_map(|t| t.strip_prefix("│ "))
            .collect::<String>();
        assert_eq!(rejoined, long, "the body wraps whole: {texts:?}");
        let answer = texts.iter()
            .filter(|t| !t.starts_with("│ ") && !t.contains("thought #") && !t.is_empty() && !t.contains("browse:"))
            .cloned().collect::<String>();
        assert_eq!(answer, long, "the answer wraps whole: {texts:?}");
        assert!(!texts[..texts.len() - 1].iter().any(|t| t.ends_with('…')),
            "review never ellipsizes content (the status bar may still cut): {texts:?}");
    }

    #[test]
    fn fold_hierarchy_is_structural_not_chromatic() {
        // the claim of the gutter: strip every color and modifier and the
        // block still reads as a block — DIM-blind terminals lose nothing
        std::env::set_var("JINGWEI_NO_COLOR", "1");
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Think("reasoned".into())));
        update(&mut a, Ev::Msg(Msg::ThinkEnd));
        update(&mut a, Ev::Msg(Msg::Tool {
            name: "bash".into(), summary: "$ ls".into(), output: "a\nb".into(),
        }));
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = browse(&a, 80, 12).texts();
        assert!(texts.iter().any(|t| t.starts_with("▾ thought #1")), "glyph survives: {texts:?}");
        assert!(texts.iter().any(|t| t == "│ reasoned"), "gutter survives: {texts:?}");
        assert!(texts.iter().any(|t| t == "│ b"), "tool body too: {texts:?}");
        std::env::remove_var("JINGWEI_NO_COLOR");
    }

    #[test]
    fn notes_render_with_severity_background() {
        // this row is *about* color — force it on regardless of the harness
        std::env::set_var("JINGWEI_COLOR", "always");
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Note { sev: crate::display::Sev::Err, text: "api 429".into() }));
        let row = flush_lines(&a, 60, 0).into_iter().find(|l| {
            l.spans.iter().any(|sp| sp.content.contains("api 429"))
        }).expect("error row rendered");
        assert!(row.spans.iter().all(|sp| sp.style.bg.is_some()), "error row carries a background");
        std::env::remove_var("JINGWEI_COLOR");
    }

    #[test]
    fn no_color_renders_the_same_rows_without_color() {
        std::env::set_var("JINGWEI_NO_COLOR", "1");
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Note { sev: crate::display::Sev::Err, text: "api 429".into() }));
        let row = flush_lines(&a, 60, 0).into_iter().find(|l| {
            l.spans.iter().any(|sp| sp.content.contains("api 429"))
        }).expect("error row rendered");
        assert!(row.spans.iter().all(|sp| sp.style.bg.is_none()), "NO_COLOR strips backgrounds");
        std::env::remove_var("JINGWEI_NO_COLOR");
    }

    #[test]
    fn live_thinking_streams_into_the_pane_and_never_flushes_early() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::TaskBegin("refactor".into())));
        update(&mut a, Ev::Msg(Msg::Think("step one\nstep two\nstep three\nstep four".into())));
        for _ in 0..10 { update(&mut a, Ev::Tick(super::super::model::TICK)); }

        // the pane shows the block as it arrives: its future number, its
        // running clock, and the tail that still moves
        let texts = texts_of(&pane(&a, 72, 24).lines);
        assert!(texts.iter().any(|t| t.starts_with("◌ thought #1 · 4 lines · 1s")), "{texts:?}");
        assert_eq!(texts.iter().filter(|t| t.starts_with("│ step")).count(), THINK_TAIL_SHOWN,
            "only the tail moves: {texts:?}");
        assert!(!texts.iter().any(|t| t.contains("step one")), "the head of the body is not bulk: {texts:?}");

        // nothing of the block lands in the scrollback while it is live
        let flushed = texts_of(&flush_lines(&a, 72, 0));
        assert!(flushed.iter().all(|t| !t.contains("◌") && !t.contains("step")), "the live block never flushes: {flushed:?}");

        // browse reviews the live block too, wrapped
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = browse(&a, 72, 12).texts();
        assert!(texts.iter().any(|t| t.contains("◌ thought #1")), "{texts:?}");
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));

        // ThinkEnd folds it once, with its final state and measured duration
        update(&mut a, Ev::Msg(Msg::ThinkEnd));
        let texts = texts_of(&flush_lines(&a, 72, 0));
        assert!(texts.iter().any(|t| t == "▸ thought #1 · step one … +3"), "folded once: {texts:?}");
    }

    #[test]
    fn actions_still_flow_through() {
        let mut a = App::new();
        type_str(&mut a, "hi");
        assert!(matches!(update(&mut a, key(KeyCode::Enter, KeyModifiers::NONE)), Action::Submit(_)));
    }
}
