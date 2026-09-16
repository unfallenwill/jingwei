//! TUI view: a pure function of the state. `view` renders the whole screen
//! into plain data — rows of styled spans plus the input cursor's column —
//! and never touches a terminal, a clock, or a random number. Tests assert
//! on the produced rows directly; the backend only blits them.
//!
//! Long lines are truncated with an ellipsis rather than wrapped: every
//! rendered row is then exactly one physical row, which keeps the scroll
//! arithmetic in this file trivially correct.

use super::model::{App, Fold, Mode, Row};
use crate::{bar_text, disp_width, elapsed_str, truncate_cols};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// How many body lines a *collapsed* tool tail shows before the "+N" marker.
const TOOL_TAIL_SHOWN: usize = 3;

const WARN_BG: (u8, u8, u8) = (255, 220, 100);
const ERR_BG: (u8, u8, u8) = (198, 40, 40);

/// The rendered screen: one `Line` per physical row, top to bottom, plus —
/// in input mode — the column the terminal cursor should sit at (on the
/// input row, the second-to-last row).
#[derive(Debug)]
pub struct Screen {
    pub lines: Vec<Line<'static>>,
    pub cursor: Option<u16>,
}

impl Screen {
    /// The plain text of each row, for assertions.
    #[cfg(test)]
    pub fn texts(&self) -> Vec<String> {
        self.lines.iter().map(|l| l.spans.iter().map(|s| s.content.clone()).collect::<String>()).collect()
    }
}

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

fn head_style() -> Style {
    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
}

fn note_style(bg: (u8, u8, u8), err: bool) -> Style {
    let mut s = Style::default().bg(Color::Rgb(bg.0, bg.1, bg.2));
    if err {
        s = s.fg(Color::White).add_modifier(Modifier::BOLD);
    }
    s
}

/// Render the app into a screen `w` columns by `h` rows.
///
/// Layout — input mode:
/// ```text
/// ─ transcript (h-2 rows, scrolled) ─
/// ─ input row ─
/// ─ status bar ─
/// ```
/// browse mode drops the input row and gives the transcript its space.
pub fn view(app: &App, w: u16, h: u16) -> Screen {
    let w = w.max(8) as usize;
    let body = render_rows(app, w);
    let (area_h, cursor) = match app.mode {
        // while a task runs, the input row shows the locked task, dim
        Mode::Input if app.working() => (h.saturating_sub(2), None),
        Mode::Input => (h.saturating_sub(2), Some(input_cursor_col(app, w))),
        Mode::Browse => (h.saturating_sub(1), None),
    };
    let area_h = area_h as usize;

    // The visible window: the last `area_h` rendered rows, offset by the
    // scroll position (rows scrolled up from the bottom). `u16::MAX` means
    // "top" — clamp it against the actual content.
    let scroll = (app.scroll as usize).min(body.len().saturating_sub(area_h));
    let end = body.len() - scroll;
    let start = end.saturating_sub(area_h);
    let mut lines: Vec<Line<'static>> = body[start..end].to_vec();
    // pad short content so the layout keeps its shape
    while lines.len() < area_h {
        lines.push(Line::from(""));
    }

    let mut out = lines;
    if app.mode == Mode::Input {
        out.push(match &app.status.task {
            Some(t) => Line::from(vec![
                Span::styled("jingwei", Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD)),
                Span::styled(" ❯ ", Style::default().fg(Color::DarkGray)),
                Span::styled(truncate_cols(&t.text, w.saturating_sub(10)), dim()),
            ]),
            None => input_row(app, w),
        });
    }
    out.push(status_row(app, w));
    Screen { lines: out, cursor: cursor.map(|c| c as u16) }
}

/// Flatten the transcript rows into rendered lines, expanding open folds.
/// One transcript row may render as several lines (a fold's body); the
/// scroll offset counts *rendered* lines, so what you see is what scrolls.
fn render_rows(app: &App, w: usize) -> Vec<Line<'static>> {
    let mut out = vec![];
    let push = |l: Line<'static>, out: &mut Vec<Line<'static>>| out.push(l);
    for row in &app.rows {
        match row {
            Row::Task(t) => push(
                Line::from(vec![
                    Span::styled("jingwei", Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD)),
                    Span::raw(" ❯ "),
                    Span::raw(truncate_cols(t, w.saturating_sub(10))),
                ]),
                &mut out,
            ),
            Row::Line(l) => push(Line::from(truncate_cols(l, w)), &mut out),
            Row::Thought(id) => fold_lines(app.folds.get(*id), w, &mut out),
            Row::Tool { head, id } => {
                push(Line::from(Span::styled(truncate_cols(head, w), head_style())), &mut out);
                fold_lines(app.folds.get(*id), w, &mut out);
            }
            Row::Note(sev, text) => {
                let (bg, err) = match sev {
                    crate::display::Sev::Warn => (WARN_BG, false),
                    crate::display::Sev::Err => (ERR_BG, true),
                };
                push(Line::from(Span::styled(format!(" {text} ").trim_end().to_string(), note_style(bg, err))), &mut out);
            }
        }
    }
    if !app.partial.is_empty() {
        push(Line::from(truncate_cols(&app.partial, w)), &mut out);
    }
    out
}

/// A fold renders as its marker when closed, header + full body when open.
/// Tool tails show a short preview even when closed — the call is visible,
/// its bulk is not.
fn fold_lines(fold: Option<&Fold>, w: usize, out: &mut Vec<Line<'static>>) {
    let Some(f) = fold else { return };
    let n = f.lines();
    if f.kind == "thought" {
        let (glyph, style) = if f.expanded { ("▾", head_style()) } else { ("▸", dim()) };
        out.push(Line::from(Span::styled(
            truncate_cols(&format!("{glyph} thought #{} · {} lines", f.n, n), w),
            style,
        )));
        if f.expanded {
            for l in f.body.lines() {
                out.push(Line::from(Span::styled(truncate_cols(l, w), dim())));
            }
        }
    } else {
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

/// The input row: prompt plus a horizontal window of the edit line centered
/// on the cursor when the text is wider than the row.
fn input_row(app: &App, w: usize) -> Line<'static> {
    let prompt_w = 10; // "jingwei ❯ "
    let avail = w.saturating_sub(prompt_w);
    let before = &app.input.text[..app.input.cursor.min(app.input.text.len())];
    let (off, shown) = if disp_width(before) > avail.saturating_sub(1) {
        // keep the cursor visible: drop whole graphemes from the front
        let mut off = app.input.text.len();
        let mut width = 0;
        for g in app.input.text[..app.input.cursor].graphemes_rev() {
            let gw = disp_width(g);
            if width + gw > avail.saturating_sub(4) {
                break;
            }
            width += gw;
            off -= g.len();
        }
        (off, truncate_cols(&app.input.text[off..], avail))
    } else {
        (0, truncate_cols(&app.input.text, avail))
    };
    Line::from(vec![
        Span::styled("jingwei", Style::default().fg(Color::LightBlue).add_modifier(Modifier::BOLD)),
        Span::styled(" ❯ ", Style::default().fg(Color::DarkGray)),
        Span::raw(if off > 0 { format!("…{shown}") } else { shown }),
    ])
}

/// The cursor's column on the input row (prompt width + displayed width of
/// the text before the cursor, minus any horizontal scroll offset).
fn input_cursor_col(app: &App, w: usize) -> usize {
    let prompt_w = 10;
    let avail = w.saturating_sub(prompt_w);
    let i = app.input.cursor.min(app.input.text.len());
    let before = &app.input.text[..i];
    let col = disp_width(before);
    if col > avail.saturating_sub(1) {
        // the view scrolled; the cursor rides at the right edge
        avail.saturating_sub(1)
    } else {
        prompt_w + col
    }
}

/// The status bar: the same pure `bar_text` the log UI used, plus the mode
/// hint in browse.
fn status_row(app: &App, w: usize) -> Line<'static> {
    let head = app.status.task.as_ref()
        .map(|t| (super::model::SPINNER[app.status.spin % super::model::SPINNER.len()], elapsed_str(t.elapsed)));
    let mut bar = bar_text(head.as_ref().map(|(g, e)| (*g, e.as_str())), &app.status.turn, &app.status.total);
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

/// Reverse grapheme iteration for the input row's horizontal scroll.
trait GraphemesRev {
    fn graphemes_rev(&self) -> Vec<&str>;
}

impl GraphemesRev for str {
    fn graphemes_rev(&self) -> Vec<&str> {
        use unicode_segmentation::UnicodeSegmentation;
        self.graphemes(true).collect::<Vec<_>>().into_iter().rev().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::App;
    use crate::Usage;
    use super::super::update::{update, Ev, Action};
    use super::*;
    use crate::display::Msg;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, KeyEventKind, KeyEventState};

    fn key(code: KeyCode, mods: KeyModifiers) -> Ev {
        Ev::Key(KeyEvent { code, modifiers: mods, kind: KeyEventKind::Press, state: KeyEventState::NONE })
    }

    fn type_str(a: &mut App, s: &str) {
        for c in s.chars() {
            update(a, key(KeyCode::Char(c), KeyModifiers::NONE));
        }
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

    #[test]
    fn folded_thought_renders_as_one_marker_not_its_body() {
        let a = seeded();
        let s = view(&a, 80, 24);
        let texts = s.texts();
        assert!(texts.iter().any(|t| t.contains("▸ thought #1 · 3 lines")), "{texts:?}");
        assert!(texts.iter().any(|t| t == "found 3 files"));
        assert!(texts.iter().all(|t| !t.contains("line one")), "body stays folded: {texts:?}");
    }

    #[test]
    fn ctrl_o_expansion_shows_bodies_and_keeps_them_after_return() {
        let mut a = seeded();
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = view(&a, 80, 24).texts();
        assert!(texts.iter().any(|t| t.contains("▾ thought #1 · 3 lines")));
        assert!(texts.iter().any(|t| t == "line one"), "browse shows the body");
        // back to input mode: still expanded, input row is back
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let s = view(&a, 80, 24);
        let texts = s.texts();
        assert!(texts.iter().any(|t| t == "line one"), "ratchet: expanded stays expanded");
        assert!(s.cursor.is_some(), "input mode owns the cursor");
    }

    #[test]
    fn collapsed_tool_shows_short_tail_expanded_shows_all() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Tool {
            name: "bash".into(),
            summary: "$ seq 1 10".into(),
            output: (1..=10).map(|i| i.to_string()).collect::<Vec<_>>().join("\n"),
        }));
        let texts = view(&a, 80, 24).texts();
        assert!(texts.iter().any(|t| t == "[bash] $ seq 1 10"));
        assert!(texts.iter().any(|t| t == "… +7 more lines"), "{texts:?}");
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let texts = view(&a, 80, 24).texts();
        assert!(texts.iter().any(|t| t == "10"), "expanded tail reaches the end");
        assert!(texts.iter().all(|t| !t.contains("more lines")));
    }

    #[test]
    fn input_row_tracks_cursor_for_latins_and_cjk() {
        let mut a = App::new();
        type_str(&mut a, "精卫 fill");
        let s = view(&a, 40, 10);
        assert_eq!(s.cursor, Some(10 + disp_width("精卫 fill") as u16));
        // walk back over a wide char: the column drops by 2
        update(&mut a, key(KeyCode::Left, KeyModifiers::NONE));
        let s = view(&a, 40, 10);
        assert_eq!(s.cursor, Some((10 + disp_width("精卫 fil")) as u16));
    }

    #[test]
    fn working_status_bar_shows_spinner_elapsed_and_usage() {
        let mut a = seeded();
        update(&mut a, Ev::Msg(Msg::TaskBegin("run".into())));
        for _ in 0..25 {
            update(&mut a, Ev::Tick);
        }
        update(&mut a, Ev::Msg(Msg::Usage(Usage { input: 100, output: 7, ..Usage::default() })));
        let texts = view(&a, 80, 24).texts();
        let bar = texts.last().unwrap();
        assert!(bar.contains("s ·"), "elapsed present: {bar}");
        assert!(bar.contains("turn in 100 out 7"), "{bar}");
    }

    #[test]
    fn browse_mode_drops_input_row_and_adds_hint() {
        let mut a = seeded();
        update(&mut a, key(KeyCode::Char('o'), KeyModifiers::CONTROL));
        let s = view(&a, 80, 24);
        assert_eq!(s.lines.len(), 24);
        assert!(s.cursor.is_none());
        assert!(s.texts().last().unwrap().contains("Ctrl-O returns"));
    }

    #[test]
    fn scrolling_reveals_older_rows_and_clamps() {
        let mut a = seeded();
        for i in 0..40 {
            update(&mut a, Ev::Msg(Msg::Text(format!("row {i}\n"))));
        }
        let bottom = view(&a, 40, 6).texts();
        assert!(bottom.iter().any(|t| t == "row 39"), "{bottom:?}");
        update(&mut a, key(KeyCode::PageUp, KeyModifiers::NONE)); // enter browse, +10
        update(&mut a, key(KeyCode::PageUp, KeyModifiers::NONE));
        let top = view(&a, 40, 6).texts();
        assert!(top.iter().any(|t| t == "row 19"), "window moved up: {top:?}");
        assert!(!top.iter().any(|t| t == "row 38"), "scrolled away from the tail: {top:?}");
        // G clamps to top; u16::MAX never overflows the window math
        update(&mut a, key(KeyCode::Char('g'), KeyModifiers::NONE));
        let _ = view(&a, 40, 6); // must not panic
    }

    #[test]
    fn long_lines_truncate_rather_than_wrap() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Text(format!("{}\n", "x".repeat(200)))));
        let s = view(&a, 40, 6);
        assert!(s.lines.iter().all(|l| {
            let t = l.spans.iter().map(|s| s.content.clone()).collect::<String>();
            disp_width(&t) <= 40
        }));
    }

    #[test]
    fn notes_render_with_severity_background() {
        let mut a = App::new();
        update(&mut a, Ev::Msg(Msg::Note { sev: crate::display::Sev::Err, text: "api 429".into() }));
        let s = view(&a, 60, 6);
        let row = s.lines.iter().find(|l| {
            l.spans.iter().any(|sp| sp.content.contains("api 429"))
        }).expect("error row rendered");
        assert!(row.spans.iter().all(|sp| sp.style.bg.is_some()), "error row carries a background");
    }

    #[test]
    fn actions_still_flow_through() {
        let mut a = App::new();
        type_str(&mut a, "hi");
        assert!(matches!(update(&mut a, key(KeyCode::Enter, KeyModifiers::NONE)), Action::Submit(_)));
    }
}

