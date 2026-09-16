//! TUI update: the event-to-state transition, pure. `update` takes the app
//! and one event and returns whatever action the outside world must perform
//! (submit a task, cancel the agent, nothing). Every rule below — editing,
//! history recall, scroll clamping, follow-the-tail, the fold ratchet — is
//! unit-tested without a terminal.

use super::model::{App, Fold, Mode, Row, Task, TICK};
use crate::display::Msg;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

/// What the run loop should do after an update.
#[derive(PartialEq, Eq, Debug)]
pub enum Action {
    /// The user submitted this task text.
    Submit(String),
    /// Ctrl-C: cancel the running agent (first press) / exit (second).
    Cancel,
    /// Nothing — just redraw.
    None,
}

pub enum Ev {
    Key(KeyEvent),
    Paste(String),
    Msg(Msg),
    Tick,
}

pub fn update(app: &mut App, ev: Ev) -> Action {
    match ev {
        Ev::Tick => tick(app),
        Ev::Msg(m) => msg(app, m),
        Ev::Paste(p) => {
            // pasted newlines are flattened: the editor is one line
            let flat = p.replace(['\r', '\n'], " ");
            input_insert(app, &flat);
            Action::None
        }
        Ev::Key(k) => key(app, k),
    }
}

fn tick(app: &mut App) -> Action {
    if let Some(t) = &mut app.status.task {
        t.elapsed += TICK;
        app.status.spin = (app.status.spin + 1) % super::model::SPINNER.len();
    }
    Action::None
}

fn msg(app: &mut App, m: Msg) -> Action {
    match m {
        Msg::Banner(b) => app.rows.push(Row::Line(b)),
        Msg::TaskBegin(t) => {
            app.status.task = Some(Task { text: t.clone(), elapsed: Duration::ZERO });
            app.status.spin = 0;
            app.cancel_sent = false;
            app.follow = true;
            app.scroll = 0;
            app.flush_partial();
            app.rows.push(Row::Task(t));
        }
        Msg::TaskEnd => {
            app.flush_partial();
            app.fold_thought();
            app.status.task = None;
            app.cancel_sent = false;
            app.rows.push(Row::Line(String::new()));
        }
        Msg::Text(t) => app.push_text(&t),
        Msg::Think(t) => app.think_buf.push_str(&t),
        Msg::ThinkEnd => app.fold_thought(),
        Msg::Tool { name, summary, output } => {
            app.flush_partial();
            app.fold_thought();
            let n = app.folds.iter().filter(|f| f.kind == "tool").count() + 1;
            let id = app.folds.len();
            app.folds.push(Fold { n, kind: "tool", body: output, expanded: false });
            app.rows.push(Row::Tool { head: format!("[{name}] {summary}"), id });
        }
        Msg::Note { sev, text } => {
            app.flush_partial();
            app.rows.push(Row::Note(sev, text));
        }
        Msg::Usage(u) => app.status.turn = u,
        Msg::OutTokens(n) => app.status.turn.output = n,
        Msg::Done => {
            app.status.total.add(&app.status.turn);
            app.flush_partial();
            app.fold_thought();
        }
    }
    if app.follow {
        app.scroll = 0;
    }
    Action::None
}

fn key(app: &mut App, k: KeyEvent) -> Action {
    if k.kind != crossterm::event::KeyEventKind::Press {
        return Action::None;
    }
    match app.mode {
        Mode::Input => input_mode(app, k),
        Mode::Browse => browse_mode(app, k),
    }
}

fn input_mode(app: &mut App, k: KeyEvent) -> Action {
    // Ctrl-C outranks mode: while a task runs it cancels, twice exits.
    if matches!(k.code, KeyCode::Char('c')) && k.modifiers.contains(KeyModifiers::CONTROL) {
        if app.working() {
            if app.cancel_sent {
                app.quit = true;
            } else {
                app.cancel_sent = true;
                return Action::Cancel;
            }
        } else {
            input_clear(app);
        }
        return Action::None;
    }
    // While working, the input row shows the locked task — no editing.
    if app.working() {
        return match k.code {
            KeyCode::Char('o') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                browse(app);
                Action::None
            }
            _ => Action::None,
        };
    }
    match k.code {
        KeyCode::Enter => {
            let line = app.input.text.trim().to_string();
            if line.is_empty() {
                return Action::None;
            }
            app.input.history.push(line.clone());
            app.input.hist_pos = app.input.history.len();
            app.input.draft = None;
            input_clear(app);
            app.follow = true;
            app.scroll = 0;
            Action::Submit(line)
        }
        KeyCode::Char(c) if !k.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            input_insert(app, &c.to_string());
            Action::None
        }
        KeyCode::Backspace => {
            if k.modifiers.contains(KeyModifiers::CONTROL) {
                delete_word_back(app);
            } else {
                delete_grapheme_back(app);
            }
            Action::None
        }
        KeyCode::Delete => {
            delete_grapheme_forward(app);
            Action::None
        }
        KeyCode::Left => {
            move_cursor(app, -1);
            Action::None
        }
        KeyCode::Right => {
            move_cursor(app, 1);
            Action::None
        }
        KeyCode::Home => {
            app.input.cursor = 0;
            Action::None
        }
        KeyCode::End => {
            app.input.cursor = app.input.text.len();
            Action::None
        }
        KeyCode::Up => {
            history(app, -1);
            Action::None
        }
        KeyCode::Down => {
            history(app, 1);
            Action::None
        }
        KeyCode::Char('u') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            input_clear(app);
            Action::None
        }
        KeyCode::Char('w') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            delete_word_back(app);
            Action::None
        }
        KeyCode::Char('d') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.input.text.is_empty() {
                app.quit = true;
            } else {
                delete_grapheme_forward(app);
            }
            Action::None
        }
        KeyCode::Char('o') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            browse(app);
            Action::None
        }
        KeyCode::PageUp => {
            browse(app);
            app.follow = false;
            app.scroll = app.scroll.saturating_add(10);
            Action::None
        }
        _ => Action::None,
    }
}

fn browse_mode(app: &mut App, k: KeyEvent) -> Action {
    if matches!(k.code, KeyCode::Char('c')) && k.modifiers.contains(KeyModifiers::CONTROL) {
        if app.working() {
            if app.cancel_sent {
                app.quit = true;
            } else {
                app.cancel_sent = true;
                return Action::Cancel;
            }
        } else {
            app.mode = Mode::Input;
        }
        return Action::None;
    }
    match k.code {
        // Ctrl-O (and q, Esc) return to the REPL; folds stay expanded.
        KeyCode::Char('o') if k.modifiers.contains(KeyModifiers::CONTROL) => app.mode = Mode::Input,
        KeyCode::Char('q') | KeyCode::Esc => app.mode = Mode::Input,
        KeyCode::Up | KeyCode::Char('k') => scroll(app, 1),
        KeyCode::Down | KeyCode::Char('j') => scroll_down(app, 1),
        KeyCode::PageUp => scroll(app, 10),
        KeyCode::PageDown => scroll_down(app, 10),
        KeyCode::Char('g') | KeyCode::Home => {
            app.scroll = u16::MAX; // clamped against content at render time
            app.follow = false;
        }
        KeyCode::Char('G') | KeyCode::End => {
            app.scroll = 0;
            app.follow = true;
        }
        _ => {}
    }
    Action::None
}

/// Enter the expanded review mode: open every fold (the ratchet), take the
/// user's eyes with us.
fn browse(app: &mut App) {
    app.expand_all();
    app.mode = Mode::Browse;
}

fn scroll(app: &mut App, n: u16) {
    app.scroll = app.scroll.saturating_add(n);
    app.follow = false;
}

fn scroll_down(app: &mut App, n: u16) {
    app.scroll = app.scroll.saturating_sub(n);
    if app.scroll == 0 {
        app.follow = true;
    }
}

// ---- the line editor --------------------------------------------------------

/// Split the editor's text at the cursor (grapheme-safe: cursor is kept on
/// char boundaries by every operation below).
fn split_at_cursor(app: &App) -> (String, String) {
    let i = app.input.cursor.min(app.input.text.len());
    // back to a char boundary if a stray byte index ever lands mid-char
    let mut i = i;
    while i < app.input.text.len() && !app.input.text.is_char_boundary(i) {
        i += 1;
    }
    (app.input.text[..i].to_string(), app.input.text[i..].to_string())
}

fn input_insert(app: &mut App, s: &str) {
    let (a, b) = split_at_cursor(app);
    let cursor = a.len() + s.len();
    app.input.text = format!("{a}{s}{b}");
    app.input.cursor = cursor;
}

fn input_clear(app: &mut App) {
    app.input.text.clear();
    app.input.cursor = 0;
}

fn delete_grapheme_back(app: &mut App) {
    let (a, b) = split_at_cursor(app);
    let cut = a.grapheme_cut(true);
    app.input.text = format!("{cut}{b}");
    app.input.cursor = cut.len();
}

fn delete_grapheme_forward(app: &mut App) {
    let (a, b) = split_at_cursor(app);
    let keep = b.grapheme_cut(false);
    app.input.text = format!("{a}{keep}");
}

/// Trim the last/first grapheme (family emoji, CJK, combining marks count as
/// one — the cursor must never split what the terminal renders as one).
trait GraphemeCut {
    fn grapheme_cut(&self, back: bool) -> String;
}

impl GraphemeCut for String {
    fn grapheme_cut(&self, back: bool) -> String {
        use unicode_segmentation::UnicodeSegmentation;
        let gs: Vec<&str> = self.graphemes(true).collect();
        if back {
            gs[..gs.len().saturating_sub(1)].concat()
        } else {
            gs[gs.len().min(1)..].concat()
        }
    }
}

/// Move by whole graphemes; `d` is -1 (left) or 1 (right).
fn move_cursor(app: &mut App, d: i32) {
    let (a, b) = split_at_cursor(app);
    app.input.cursor = if d < 0 {
        grapheme_offset(&a, -1)
    } else {
        a.len() + grapheme_offset(&b, 1)
    };
}

/// Byte offset of the next/previous grapheme boundary in `s`.
fn grapheme_offset(s: &str, d: i32) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    if d < 0 {
        let gs: Vec<&str> = s.graphemes(true).collect();
        gs[..gs.len().saturating_sub(1)].iter().map(|g| g.len()).sum()
    } else {
        s.graphemes(true).next().map_or(0, |g| g.len())
    }
}

/// Ctrl-W: kill the previous word together with the whitespace before it —
/// repeated presses eat the line word by word.
fn delete_word_back(app: &mut App) {
    let (a, b) = split_at_cursor(app);
    let trimmed = a.trim_end();
    let cut = match trimmed.rfind(char::is_whitespace) {
        Some(i) => &a[..i],
        None => "",
    };
    app.input.text = format!("{cut}{b}");
    app.input.cursor = cut.len();
}

fn history(app: &mut App, d: i32) {
    let h = &mut app.input.history;
    if h.is_empty() {
        return;
    }
    if app.input.draft.is_none() {
        app.input.draft = Some(app.input.text.clone());
    }
    let max = h.len();
    let pos = app.input.hist_pos.min(max);
    let next = if d < 0 { pos.saturating_sub(1) } else { (pos + 1).min(max) };
    app.input.hist_pos = next;
    app.input.text = if next == max {
        app.input.draft.clone().unwrap_or_default()
    } else {
        h[next].clone()
    };
    app.input.cursor = app.input.text.len();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Usage;
    use crossterm::event::KeyEventKind;

    fn key(code: KeyCode, mods: KeyModifiers) -> Ev {
        Ev::Key(KeyEvent { code, modifiers: mods, kind: KeyEventKind::Press, state: KeyEventState::NONE })
    }
    use crossterm::event::KeyEventState;

    fn drive(app: &mut App, evs: Vec<Ev>) -> Vec<Action> {
        evs.into_iter().map(|e| update(app, e)).collect()
    }

    fn type_str(app: &mut App, s: &str) {
        for c in s.chars() {
            drive(app, vec![key(KeyCode::Char(c), KeyModifiers::NONE)]);
        }
    }

    // ---- the editor ----

    #[test]
    fn editor_inserts_moves_and_deletes_by_grapheme() {
        let mut a = App::new();
        type_str(&mut a, "hé精卫");
        assert_eq!(a.input.text, "hé精卫");
        drive(&mut a, vec![key(KeyCode::Left, KeyModifiers::NONE)]); // before 卫
        drive(&mut a, vec![key(KeyCode::Backspace, KeyModifiers::NONE)]); // kill 精
        assert_eq!(a.input.text, "hé卫");
        assert_eq!(a.input.cursor, "hé".len());
        drive(&mut a, vec![key(KeyCode::Delete, KeyModifiers::NONE)]); // kill 卫
        assert_eq!(a.input.text, "hé");
        drive(&mut a, vec![key(KeyCode::Home, KeyModifiers::NONE)]);
        type_str(&mut a, "X");
        assert_eq!(a.input.text, "Xhé");
    }

    #[test]
    fn editor_word_and_line_deletes() {
        let mut a = App::new();
        type_str(&mut a, "one two  three");
        drive(&mut a, vec![key(KeyCode::Char('w'), KeyModifiers::CONTROL)]);
        assert_eq!(a.input.text, "one two ", "Ctrl-W kills 'three'");
        drive(&mut a, vec![key(KeyCode::Char('w'), KeyModifiers::CONTROL)]);
        assert_eq!(a.input.text, "one", "Ctrl-W kills 'two' with the space before it");
        drive(&mut a, vec![key(KeyCode::Char('u'), KeyModifiers::CONTROL)]);
        assert_eq!(a.input.text, "");
    }

    #[test]
    fn enter_submits_clears_and_remembers_history() {
        let mut a = App::new();
        type_str(&mut a, "count *.rs");
        let acts = drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(acts, vec![Action::Submit("count *.rs".into())]);
        assert_eq!(a.input.text, "");
        // start a fresh draft, recall, then come back: the draft was parked
        type_str(&mut a, "abc");
        drive(&mut a, vec![key(KeyCode::Up, KeyModifiers::NONE)]);
        assert_eq!(a.input.text, "count *.rs");
        drive(&mut a, vec![key(KeyCode::Down, KeyModifiers::NONE)]);
        assert_eq!(a.input.text, "abc", "down returns to the parked draft");
    }

    #[test]
    fn empty_enter_and_ctrl_d_on_text_are_gentle() {
        let mut a = App::new();
        assert_eq!(drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]), vec![Action::None]);
        type_str(&mut a, "ab");
        drive(&mut a, vec![key(KeyCode::Left, KeyModifiers::NONE)]);
        drive(&mut a, vec![key(KeyCode::Char('d'), KeyModifiers::CONTROL)]); // deletes forward
        assert_eq!(a.input.text, "a");
        assert!(!a.quit);
        drive(&mut a, vec![key(KeyCode::Char('c'), KeyModifiers::CONTROL)]); // clears, not quit
        assert_eq!(a.input.text, "");
        assert!(!a.quit);
        drive(&mut a, vec![key(KeyCode::Char('d'), KeyModifiers::CONTROL)]); // empty: quit
        assert!(a.quit);
    }

    // ---- the fold ratchet (Ctrl-O) ----

    fn seeded() -> App {
        let mut a = App::new();
        drive(&mut a, vec![
            Ev::Msg(Msg::TaskBegin("t".into())),
            Ev::Msg(Msg::Think("l1\nl2\nl3".into())),
            Ev::Msg(Msg::ThinkEnd),
            Ev::Msg(Msg::Tool { name: "bash".into(), summary: "$ ls".into(), output: "a\nb".into() }),
            Ev::Msg(Msg::TaskEnd),
        ]);
        a
    }

    #[test]
    fn ctrl_o_expands_all_then_returns_and_never_refolds() {
        let mut a = seeded();
        assert_eq!(a.closed_folds(), 2);
        drive(&mut a, vec![key(KeyCode::Char('o'), KeyModifiers::CONTROL)]);
        assert_eq!(a.mode, Mode::Browse);
        assert_eq!(a.closed_folds(), 0, "every kind of fold opens: thought and tool");
        // back to the REPL — and the ratchet holds
        drive(&mut a, vec![key(KeyCode::Char('o'), KeyModifiers::CONTROL)]);
        assert_eq!(a.mode, Mode::Input);
        assert_eq!(a.closed_folds(), 0, "expanded stays expanded");
        // a new thought arrives folded; Ctrl-O opens only what is closed
        drive(&mut a, vec![Ev::Msg(Msg::Think("later".into())), Ev::Msg(Msg::ThinkEnd)]);
        assert_eq!(a.closed_folds(), 1);
        drive(&mut a, vec![key(KeyCode::Char('o'), KeyModifiers::CONTROL), key(KeyCode::Char('o'), KeyModifiers::CONTROL)]);
        assert_eq!(a.closed_folds(), 0);
    }

    #[test]
    fn q_and_esc_also_leave_browse() {
        let mut a = seeded();
        drive(&mut a, vec![key(KeyCode::Char('o'), KeyModifiers::CONTROL)]);
        assert_eq!(a.mode, Mode::Browse);
        drive(&mut a, vec![key(KeyCode::Char('q'), KeyModifiers::NONE)]);
        assert_eq!(a.mode, Mode::Input);
        drive(&mut a, vec![key(KeyCode::PageUp, KeyModifiers::NONE)]);
        assert_eq!(a.mode, Mode::Browse, "PgUp from the prompt is a shortcut into browse");
        drive(&mut a, vec![key(KeyCode::Esc, KeyModifiers::NONE)]);
        assert_eq!(a.mode, Mode::Input);
    }

    // ---- scrolling & following ----

    #[test]
    fn scrolling_up_stops_following_bottom_restores_it() {
        let mut a = seeded();
        drive(&mut a, vec![key(KeyCode::PageUp, KeyModifiers::NONE)]);
        assert!(a.scroll >= 10);
        assert!(!a.follow);
        // new content must not yank the view while the user reads
        drive(&mut a, vec![Ev::Msg(Msg::Text("fresh\n".into()))]);
        assert!(!a.follow);
        // walking back to the bottom resumes following
        drive(&mut a, vec![key(KeyCode::Char('G'), KeyModifiers::SHIFT)]) ; // 'G'
        assert_eq!(a.scroll, 0);
        assert!(a.follow);
    }

    // ---- messages ----

    #[test]
    fn streamed_text_lands_lines_and_keeps_partial_live() {
        let mut a = App::new();
        drive(&mut a, vec![
            Ev::Msg(Msg::Text("hello\nwor".into())),
        ]);
        assert_eq!(a.rows.len(), 1);
        assert_eq!(a.partial, "wor");
        drive(&mut a, vec![Ev::Msg(Msg::Text("ld\n".into())), Ev::Msg(Msg::Done)]);
        assert_eq!(a.partial, "");
        assert!(matches!(a.rows.last(), Some(Row::Line(l)) if l == "world"), "the tail landed");
    }

    #[test]
    fn usage_folds_into_totals_on_done() {
        let mut a = App::new();
        let u = Usage { input: 10, output: 5, ..Usage::default() };
        drive(&mut a, vec![Ev::Msg(Msg::Usage(u))]);
        assert_eq!(a.status.turn.output, 5);
        drive(&mut a, vec![Ev::Msg(Msg::OutTokens(9))]);
        assert_eq!(a.status.turn.output, 9);
        drive(&mut a, vec![Ev::Msg(Msg::Done)]);
        assert_eq!((a.status.total.input, a.status.total.output), (10, 9));
    }

    #[test]
    fn ctrl_c_cancels_when_working_clears_when_idle_twice_exits() {
        let mut a = seeded();
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("running".into()))]);
        let acts = drive(&mut a, vec![key(KeyCode::Char('c'), KeyModifiers::CONTROL)]);
        assert_eq!(acts, vec![Action::Cancel]);
        assert!(a.cancel_sent);
        assert!(!a.quit);
        drive(&mut a, vec![key(KeyCode::Char('c'), KeyModifiers::CONTROL)]);
        assert!(a.quit, "second Ctrl-C while cancelling exits");
        // idle: Ctrl-C just clears the line
        let mut b = App::new();
        type_str(&mut b, "draft");
        drive(&mut b, vec![key(KeyCode::Char('c'), KeyModifiers::CONTROL)]);
        assert_eq!(b.input.text, "");
        assert!(!b.quit);
    }

    #[test]
    fn input_is_locked_while_working() {
        let mut a = App::new();
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("busy".into()))]);
        type_str(&mut a, "xyz");
        assert_eq!(a.input.text, "");
        drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        // Enter does not submit while a task runs
    }
}
