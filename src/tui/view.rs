//! TUI view: pure functions of the state. The REPL owns only a small pane at
//! the bottom of the *primary* screen — [`pane`] renders it. The transcript
//! is not a screen the app scrolls: finished rows are rendered once by
//! [`flush_lines`] and appended above the pane into the terminal's own
//! scrollback, where the wheel (and everything that was on screen before
//! jingwei started) already works. [`browse`] renders the Ctrl-O overlay:
//! the one full-screen view, on the alternate screen, that may scroll.
//!
//! Views never touch a terminal, a clock, or a random number; tests assert
//! on the produced rows directly. Long lines are truncated with an ellipsis
//! rather than wrapped, so every rendered row is exactly one physical row —
//! what was flushed into the scrollback can never be rewritten, so rows
//! must render deterministically once.
//!
//! Styling honors [`crate::display::color_on`]: with NO_COLOR the same
//! structure renders without color, exactly like the plain frontend.

use super::model::{App, Fold, FoldKind, Mode, Row};
use crate::display::{color_on, disp_width, prompt_w, thought_marker, truncate_cols, Usage, ERR_BG, PROMPT_GUTTER, PROMPT_HEAD, WARN_BG};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::time::Duration;

/// How many body lines a *collapsed* tool tail shows before the "+N" marker.
/// Small on purpose: the folded REPL shows a preview, Ctrl-O (or the plain
/// log's [`crate::plain`] 20-line echo) exists to see the rest.
pub const TOOL_TAIL_SHOWN: usize = 3;

// ---- the status bar's text (moved out of main.rs: view vocabulary, all of it)

/// 26156 → "26.2k"; small counts stay exact.
fn humanize(n: u64) -> String {
    if n < 1000 { format!("{n}") } else { format!("{:.1}k", n as f64 / 1000.0) }
}

fn elapsed_str(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 { format!("{s}s") } else { format!("{}m{:02}s", s / 60, s % 60) }
}

/// The status bar's content within `max_w` display columns. Pure — the
/// spinner glyph and elapsed label are passed in already rendered so tests
/// can pin them. Segments appear as their data arrives; when the row gets
/// too wide, cache goes first, then the turn — session totals and the
/// spinner never do.
pub fn bar_text(working: Option<(&str, &str)>, turn: &Usage, total: &Usage, max_w: usize) -> String {
    let head: Vec<String> = working.map(|(g, e)| vec![format!("{g} {e}")]).unwrap_or_default();
    let turn_seg = (!turn.is_zero())
        .then(|| format!("turn in {} out {}", humanize(turn.context_in()), humanize(turn.output)));
    let total_seg = (!total.is_zero())
        .then(|| format!("total in {} out {}", humanize(total.context_in()), humanize(total.output)));
    let assemble = |with_turn: bool| {
        let mut segs = head.clone();
        if with_turn { if let Some(t) = &turn_seg { segs.push(t.clone()) } }
        if let Some(t) = &total_seg { segs.push(t.clone()) }
        segs.join(" · ")
    };
    let mut out = assemble(true);
    if disp_width(&out) > max_w && turn_seg.is_some() {
        out = assemble(false);
    }
    let cache = if total.cache_write > 0 {
        format!("cache {}+{}", humanize(total.cache_read), humanize(total.cache_write))
    } else if total.cache_read > 0 {
        format!("cache {}", humanize(total.cache_read))
    } else {
        String::new()
    };
    if !cache.is_empty() && disp_width(&out) + 3 + disp_width(&cache) <= max_w {
        out = if out.is_empty() { cache } else { format!("{out} · {cache}") };
    }
    out
}

/// The inline pane: the only part of the primary screen jingwei draws.
///
/// ```text
/// (streaming tail — only while a line is in flight)
/// ───────────────────────────── rule
/// jingwei ❯ the input, one row per line while composing
/// ───────────────────────────── rule
/// spinner · tokens · elapsed
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
/// per line being composed.
pub fn pane(app: &App, w: u16, h: u16) -> Pane {
    let w = w.max(8) as usize;
    let mut lines: Vec<Line<'static>> = vec![];
    if !app.partial.is_empty() {
        lines.push(Line::from(truncate_cols(&app.partial, w)));
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

/// Render transcript rows `from..` as they should land in the scrollback.
/// The caller only ever advances `from` past rows already flushed, so each
/// row is rendered exactly once, with the fold state it has right now —
/// what the ratchet has opened by then prints open, later folds print as
/// markers.
pub fn flush_lines(app: &App, w: usize, from: usize) -> Vec<Line<'static>> {
    let mut out = vec![];
    for row in &app.rows[from.min(app.rows.len())..] {
        row_lines(row, w, &mut out);
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
        row_lines(row, w, &mut out);
    }
    if !app.partial.is_empty() {
        out.push(Line::from(truncate_cols(&app.partial, w)));
    }
    out
}

/// Render one transcript row (with its fold, if any) into `out`.
fn row_lines(row: &Row, w: usize, out: &mut Vec<Line<'static>>) {
    match row {
        // a multi-line task echoes as one row per line, the gutter
        // aligning continuation lines under the prompt
        Row::Task(t) => {
            for (j, l) in t.split('\n').enumerate() {
                let head = if j == 0 { Span::styled(PROMPT_HEAD, prompt_style()) } else { Span::raw(" ".repeat(prompt_w())) };
                let gutter = if j == 0 { Span::styled(PROMPT_GUTTER, gutter_style()) } else { Span::raw("") };
                out.push(Line::from(vec![head, gutter, Span::raw(truncate_cols(l, w.saturating_sub(prompt_w())))]));
            }
        }
        Row::Line(l) => out.push(Line::from(truncate_cols(l, w))),
        Row::Sep => out.push(Line::from("")),
        // the fold travels inside the row: its kind is the row's variant,
        // not a string that might disagree with it
        Row::Thought(f) => fold_lines(f, w, out),
        Row::Tool { head, fold } => {
            out.push(Line::from(Span::styled(truncate_cols(head, w), head_style())));
            fold_lines(fold, w, out);
        }
        Row::Note(sev, text) => {
            let (bg, err) = match sev {
                crate::display::Sev::Warn => (WARN_BG, false),
                crate::display::Sev::Err => (ERR_BG, true),
            };
            out.push(Line::from(Span::styled(format!(" {text} ").trim_end().to_string(), note_style(bg, err))));
        }
    }
}

/// A fold renders as its marker when closed, header + full body when open.
/// Tool tails show a short preview even when closed — the call is visible,
/// its bulk is not.
fn fold_lines(f: &Fold, w: usize, out: &mut Vec<Line<'static>>) {
    let n = f.lines();
    match f.kind {
        FoldKind::Thought => {
            let (glyph, style) = if f.expanded { ("▾", head_style()) } else { ("▸", dim()) };
            out.push(Line::from(Span::styled(truncate_cols(&thought_marker(glyph, f.n, n), w), style)));
            if f.expanded {
                for l in f.body.lines() {
                    out.push(Line::from(Span::styled(truncate_cols(l, w), dim())));
                }
            }
        }
        FoldKind::Tool => {
            let all: Vec<&str> = f.body.lines().collect();
            if f.expanded {
                for l in all {
                    out.push(Line::from(Span::styled(truncate_cols(l, w), dim())));
                }
            } else {
                let shown = all.len().min(TOOL_TAIL_SHOWN);
                for l in &all[..shown] {
                    out.push(Line::from(Span::styled(truncate_cols(l, w), dim())));
                }
                if all.len() > shown {
                    out.push(Line::from(Span::styled(
                        truncate_cols(&format!("… +{} more lines", all.len() - shown), w),
                        dim(),
                    )));
                }
            }
        }
    }
}

/// The status bar: the turn/total usage within this pane's width, plus the
/// mode hint in browse.
fn status_row(app: &App, w: usize) -> Line<'static> {
    let head = app.status.task.as_ref()
        .map(|t| (super::model::SPINNER[app.status.spin % super::model::SPINNER.len()], elapsed_str(t.elapsed)));
    let mut bar = bar_text(head.as_ref().map(|(g, e)| (*g, e.as_str())), &app.status.turn, &app.status.total, w);
    if app.mode == Mode::Browse {
        let hint = " · browse: ↑↓ PgUp/PgDn g/G · Ctrl-O returns";
        if !bar.is_empty() {
            bar.push_str(hint);
        } else {
            bar = hint.trim_start_matches(" · ").to_string();
        }
    }
    if bar.is_empty() {
        Line::from("")
    } else {
        Line::from(Span::styled(truncate_cols(&bar, w), dim()))
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

    #[test]
    fn bar_text_segments_appear_as_data_arrives() {
        let z = Usage::default();
        assert_eq!(bar_text(None, &z, &z, BAR_W), "");
        let turn = Usage { input: 4600, output: 120, ..Usage::default() };
        assert_eq!(bar_text(None, &turn, &z, BAR_W), "turn in 4.6k out 120");
        assert_eq!(bar_text(Some(("⠋", "3s")), &turn, &z, BAR_W), "⠋ 3s · turn in 4.6k out 120");
        let total = Usage { input: 26_156, output: 336, cache_read: 25_344, cache_write: 1188 };
        assert_eq!(bar_text(None, &turn, &total, BAR_W),
            "turn in 4.6k out 120 · total in 52.7k out 336 · cache 25.3k+1.2k");
        let total2 = Usage { input: 1, output: 1, cache_read: 500, ..Usage::default() };
        assert_eq!(bar_text(None, &z, &total2, BAR_W), "total in 501 out 1 · cache 500");
    }

    #[test]
    fn bar_text_drops_turn_to_fit_and_reattaches_cache_when_room() {
        let huge = Usage { input: 9_999_999, output: 9_999_999, cache_read: 9_999_999, cache_write: 9_999_999 };
        let bar = bar_text(Some(("⠋", "999m59s")), &huge, &huge, 60);
        assert!(!bar.contains("turn in"), "turn dropped to fit: {bar}");
        assert!(bar.contains("total in"), "totals never drop: {bar}");
        assert!(!bar.contains("cache"), "cache dropped to fit: {bar}");
        // a wide pane keeps everything, a narrow one sheds segments
        let wide = bar_text(Some(("⠋", "999m59s")), &huge, &huge, 200);
        assert!(wide.contains("turn in") && wide.contains("cache"), "{wide}");
        let narrow = bar_text(Some(("⠋", "9s")), &huge, &huge, 20);
        assert!(!narrow.contains("turn in") && !narrow.contains("cache"), "segments shed as the pane narrows: {narrow}");
        // the one-physical-row guarantee is the caller's truncate; bar_text
        // itself never sheds totals by design
        assert!(disp_width(&truncate_cols(&narrow, 20)) <= 20, "{narrow}");
    }

    #[test]
    fn elapsed_str_minutes_and_seconds() {
        assert_eq!(elapsed_str(Duration::from_secs(3)), "3s");
        assert_eq!(elapsed_str(Duration::from_secs(59)), "59s");
        assert_eq!(elapsed_str(Duration::from_secs(63)), "1m03s");
    }

    // ---- the transcript ----

    #[test]
    fn flushed_rows_fold_thoughts_and_tool_tails() {
        let a = seeded();
        let texts = texts_of(&flush_lines(&a, 80, 0));
        assert!(texts.iter().any(|t| t.contains("▸ thought #1 · 3 lines")), "{texts:?}");
        assert!(texts.iter().any(|t| t == "found 3 files"));
        assert!(texts.iter().all(|t| !t.contains("line one")), "body stays folded: {texts:?}");
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
        assert!(texts.iter().any(|t| t == "line one"), "browse shows the body");
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
        assert!(texts.iter().any(|t| t == "[bash] $ seq 1 10"));
        assert!(texts.iter().any(|t| t == "… +7 more lines"), "{texts:?}");
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = browse(&a, 80, 24).texts();
        assert!(texts.iter().any(|t| t == "10"), "expanded tail reaches the end");
        assert!(texts.iter().all(|t| !t.contains("more lines")));
    }

    #[test]
    fn pane_is_framed_by_rules_tail_input_bar() {
        let mut a = seeded();
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
    fn working_pane_bar_shows_spinner_elapsed_and_usage() {
        let mut a = seeded();
        update(&mut a, Ev::Msg(Msg::TaskBegin("run".into())));
        for _ in 0..25 {
            update(&mut a, Ev::Tick(super::super::model::TICK));
        }
        update(&mut a, Ev::Msg(Msg::Usage(Usage { input: 100, output: 7, ..Usage::default() })));
        let texts = texts_of(&pane(&a, 80, 24).lines);
        let bar = texts.last().unwrap();
        assert!(bar.contains("s ·"), "elapsed present: {bar}");
        assert!(bar.contains("turn in 100 out 7"), "{bar}");
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
    fn actions_still_flow_through() {
        let mut a = App::new();
        type_str(&mut a, "hi");
        assert!(matches!(update(&mut a, key(KeyCode::Enter, KeyModifiers::NONE)), Action::Submit(_)));
    }
}
