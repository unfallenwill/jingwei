//! TUI update: the event-to-state transition, pure. `update` takes the app
//! and one event and returns whatever action the outside world must perform
//! (submit a task, cancel the agent, nothing). Every rule below — editing,
//! history recall, scroll clamping, follow-the-tail, the fold ratchet — is
//! unit-tested without a terminal.

use super::model::{App, Completion, CompletionItem, CompletionKind, Mode, Row, Scroll, Task};
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
    /// A heartbeat carrying the *measured* time since the previous tick.
    /// The shell owns the clock and reads it once; the update only
    /// accumulates — so elapsed time stays true under load (missed ticks
    /// carry bigger deltas) while remaining a pure function of events.
    Tick(Duration),
}

pub fn update(app: &mut App, ev: Ev) -> Action {
    match ev {
        Ev::Tick(dt) => tick(app, dt),
        Ev::Msg(m) => msg(app, m),
        Ev::Paste(p) => {
            // pasted line breaks are content now; while a task runs the
            // editor is locked
            if !app.working() {
                input_insert(app, &p);
            }
            Action::None
        }
        Ev::Key(k) => key(app, k),
    }
}

fn tick(app: &mut App, dt: Duration) -> Action {
    if let Some(t) = &mut app.status.task {
        t.elapsed += dt;
    }
    Action::None
}

fn msg(app: &mut App, m: Msg) -> Action {
    match m {
        Msg::Banner(b) => app.rows.push(Row::Line(b)),
        Msg::TaskBegin(t) => {
            app.status.task = Some(Task { text: t.clone(), elapsed: Duration::ZERO });
            // each task starts with a tight pane; the floor grows with
            // its live rows and holds through the folds between
            app.status.pane_floor = 0;
            app.cancel_sent = false;
            app.scroll = Scroll::Tail;
            // a working task locks the editor — drop any open menu
            dismiss_completion(app);
            app.flush_partial();
            app.rows.push(Row::Task(t));
        }
        Msg::TaskEnd => {
            app.flush_partial();
            app.fold_thought();
            app.status.task = None;
            app.cancel_sent = false;
            app.rows.push(Row::Sep);
        }
        Msg::Text(t) => app.push_text(&t),
        Msg::Think(t) => app.push_think(&t),
        Msg::ThinkEnd => app.fold_thought(),
        Msg::Tool { name, summary, output } => {
            app.fold_tool(format!("[{name}] {summary}"), output);
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

/// What the menu should currently offer — derived from the input text.
/// Returns `None` when the menu should hide (regular task, complete
/// command with no args expected). The `String` is the substring the
/// menu is filtering on: the last whitespace-separated word of the
/// relevant position (or `""` when the user just typed a space).
pub fn detect_completion(text: &str) -> Option<(CompletionKind, String)> {
    let text = text.trim_start();
    if text.is_empty() || !text.starts_with('/') { return None; }
    let body = &text[1..];
    let trimmed = body.trim_end();
    let trailing_space = body.len() > trimmed.len();
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();

    if tokens.is_empty() {
        // `/` alone or with trailing whitespace: list every command.
        return Some((CompletionKind::Command, String::new()));
    }

    let cmd = tokens[0];
    let rest = tokens.len() - 1;

    match cmd {
        // Commands that take no arguments: once the user has typed the
        // full name (and a trailing space, signalling they mean it),
        // the menu closes — there is nothing to complete.
        "exit" | "quit" | "help" if trailing_space || rest >= 1 => None,
        // `/model [arg]` — one optional profile key.
        "model" => match (rest, trailing_space) {
            (0, false) => Some((CompletionKind::Command, cmd.to_string())),
            (0, true) => Some((CompletionKind::ModelArg, String::new())),
            (1, false) => Some((CompletionKind::ModelArg, tokens[1].to_string())),
            (1, true) => None,
            _ => None,
        },
        // `/mcp <sub> [<server>]` — subcommand name, then server tag.
        "mcp" => match (rest, trailing_space) {
            (0, false) => Some((CompletionKind::Command, cmd.to_string())),
            (0, true) => Some((CompletionKind::McpSub, String::new())),
            (1, false) => Some((CompletionKind::McpSub, tokens[1].to_string())),
            (1, true) => Some((CompletionKind::McpServer, String::new())),
            (2, false) => Some((CompletionKind::McpServer, tokens[2].to_string())),
            _ => None,
        },
        // Unknown command being typed: keep offering command names as
        // long as the user has not yet typed a space.
        other => {
            if trailing_space {
                None
            } else {
                Some((CompletionKind::Command, other.to_string()))
            }
        }
    }
}

/// Sync `app.completion` with what the input currently asks for.
/// `candidates_for` supplies the filtered list for whatever
/// kind/prefix the menu now shows; the closure captures settings/hub so
/// `App` stays pure of its own devices.
///
/// Selection is preserved across same-kind+same-prefix refreshes (so
/// typing the next letter of a filter does not reset the highlight);
/// any change in kind, or in prefix, resets selection to the top — what
/// the user is now looking for is different.
pub fn set_completion(
    app: &mut App,
    kind: CompletionKind,
    prefix: &str,
    candidates: Vec<CompletionItem>,
) {
    let prefix = prefix.to_string();
    match app.completion.as_mut() {
        Some(c) if c.kind == kind && c.prefix == prefix => {
            c.candidates = candidates;
            if c.selected >= c.candidates.len() {
                c.selected = if c.candidates.is_empty() {
                    0
                } else {
                    c.candidates.len() - 1
                };
            }
        }
        _ => {
            app.completion = Some(Completion::new(kind, &prefix, candidates));
        }
    }
}

/// Hide the menu — Esc, or any input that has decided completion is no
/// longer relevant.
pub fn dismiss_completion(app: &mut App) {
    app.completion = None;
}

/// Move the highlight one row up (wrapping); no-op when the menu is
/// empty.
pub fn completion_up(app: &mut App) {
    if let Some(c) = app.completion.as_mut() {
        if !c.candidates.is_empty() {
            c.selected = (c.selected + c.candidates.len() - 1) % c.candidates.len();
        }
    }
}

/// Move the highlight one row down (wrapping); no-op when empty.
pub fn completion_down(app: &mut App) {
    if let Some(c) = app.completion.as_mut() {
        if !c.candidates.is_empty() {
            c.selected = (c.selected + 1) % c.candidates.len();
        }
    }
}

/// Apply the highlighted completion: replace the prefix the menu is
/// filtering on with the candidate's `insert` text, optionally append a
/// trailing space, and leave the caret at the end of what was just
/// inserted. Returns whether anything happened — the caller can ignore
/// it (Tab when nothing is selected is just a no-op).
pub fn apply_completion(app: &mut App) -> bool {
    let Some(c) = app.completion.as_ref() else { return false };
    let Some(item) = c.candidates.get(c.selected).cloned() else { return false };
    let prefix = c.prefix.clone();

    // The prefix lives at the caret. For command completion the prefix
    // starts after the leading `/`; for arg completion it starts after
    // the last whitespace. Both are exactly `prefix.len()` bytes back
    // from the caret — provided the menu has been kept in sync with
    // the input, which `set_completion` (called after every key event)
    // enforces.
    let cur = app.input.caret();
    let replace_start = cur.saturating_sub(prefix.len());
    let trailing_space = item.trailing_space;

    let mut new_text = String::with_capacity(app.input.text.len() + item.insert.len() + 1);
    new_text.push_str(&app.input.text[..replace_start]);
    new_text.push_str(&item.insert);
    if trailing_space { new_text.push(' '); }
    new_text.push_str(&app.input.text[cur..]);
    app.input.text = new_text;
    app.input.cursor = replace_start + item.insert.len() + if trailing_space { 1 } else { 0 };

    // The TUI loop's next refresh recomputes the menu from the new
    // input. We drop the stale one now so a highlighted row does not
    // linger one keystroke behind.
    app.completion = None;
    true
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
    // Completion keys (Tab / Esc / arrows) outrank the editor: while the
    // menu is up, these keys belong to it, not to the cursor.
    if app.completion.is_some() {
        match k.code {
            KeyCode::Tab => return if apply_completion(app) { Action::None } else { Action::None },
            KeyCode::BackTab => { completion_up(app); return Action::None; }
            KeyCode::Esc => { dismiss_completion(app); return Action::None; }
            KeyCode::Up => { completion_up(app); return Action::None; }
            KeyCode::Down => { completion_down(app); return Action::None; }
            KeyCode::Enter => {
                // Enter still submits; if a row is highlighted, apply it first.
                apply_completion(app);
            }
            _ => {}
        }
    } else {
        // No menu up — Tab inserts a literal tab if the cursor is past
        // column 1 of an empty line and nothing else applies. We have
        // no use for that today, so swallow it silently.
        if matches!(k.code, KeyCode::Tab) {
            return Action::None;
        }
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
        // Shift-Enter (and Ctrl-Enter, where the terminal reports it)
        // break the line instead of submitting
        KeyCode::Enter if k.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::CONTROL) => {
            input_insert(app, "\n");
            Action::None
        }
        KeyCode::Enter => {
            let line = app.input.text.trim().to_string();
            if line.is_empty() {
                return Action::None;
            }
            // `/exit` is a command, not a task: it submits nothing and
            // recalls nothing — the line clears and the REPL quits
            if crate::display::is_exit(&line) {
                input_clear(app);
                app.quit = true;
                return Action::None;
            }
            app.input.history.push(line.clone());
            app.input.hist_pos = app.input.history.len();
            app.input.draft = None;
            // the editor empties for the next task; Up recalls this one
            input_clear(app);
            app.scroll = Scroll::Tail;
            // submitting a slash command closes the menu — its work is done
            dismiss_completion(app);
            Action::Submit(line)
        }
        KeyCode::Char(c) if !k.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            input_insert(app, &c.to_string());
            Action::None
        }
        // Ctrl-J is the line break that works on every terminal: in raw
        // mode it arrives as exactly this key event
        KeyCode::Char('j') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            input_insert(app, "\n");
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
        // Home/End are line-wise: they clamp to the current line's bounds
        KeyCode::Home => {
            let i = app.input.caret();
            let start = app.input.text[..i].rfind('\n').map_or(0, |p| p + 1);
            app.input.cursor = start;
            Action::None
        }
        KeyCode::End => {
            let i = app.input.caret();
            let end = app.input.text[i..].find('\n').map_or(app.input.text.len(), |p| i + p);
            app.input.cursor = end;
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
            // entering browse dismisses any menu — the overlay has its
            // own affordances, the completion has nothing to point at
            dismiss_completion(app);
            browse(app);
            Action::None
        }
        KeyCode::PageUp => {
            dismiss_completion(app);
            browse(app);
            app.scroll = app.scroll.lift(10);
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
        KeyCode::Up | KeyCode::Char('k') => app.scroll = app.scroll.lift(1),
        KeyCode::Down | KeyCode::Char('j') => app.scroll = app.scroll.sink(1),
        KeyCode::PageUp => app.scroll = app.scroll.lift(10),
        KeyCode::PageDown => app.scroll = app.scroll.sink(10),
        KeyCode::Char('g') | KeyCode::Home => app.scroll = Scroll::TOP,
        KeyCode::Char('G') | KeyCode::End => app.scroll = Scroll::Tail,
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

// ---- the line editor --------------------------------------------------------
//
// All ops work in place on byte offsets taken from `Input::caret` — no
// prefix/suffix clones, no format!-reassembly: one allocation at most (the
// growth of the String itself).

/// Byte offset of the start of the grapheme before `i` (0 at the front).
fn prev_grapheme_start(text: &str, i: usize) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    text[..i].grapheme_indices(true).next_back().map_or(0, |(s, _)| s)
}

/// Byte offset just past the grapheme at `i` (`i` when at the end).
fn next_grapheme_end(text: &str, i: usize) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    text[i..].graphemes(true).next().map_or(i, |g| i + g.len())
}

fn input_insert(app: &mut App, s: &str) {
    let i = app.input.caret();
    app.input.text.insert_str(i, s);
    app.input.cursor = i + s.len();
}

fn input_clear(app: &mut App) {
    app.input.text.clear();
    app.input.cursor = 0;
}

fn delete_grapheme_back(app: &mut App) {
    let i = app.input.caret();
    let start = prev_grapheme_start(&app.input.text, i);
    app.input.text.replace_range(start..i, "");
    app.input.cursor = start;
}

fn delete_grapheme_forward(app: &mut App) {
    let i = app.input.caret();
    let end = next_grapheme_end(&app.input.text, i);
    app.input.text.replace_range(i..end, "");
}

/// Move by whole graphemes; `d` is -1 (left) or 1 (right).
fn move_cursor(app: &mut App, d: i32) {
    let i = app.input.caret();
    app.input.cursor = if d < 0 {
        prev_grapheme_start(&app.input.text, i)
    } else {
        next_grapheme_end(&app.input.text, i)
    };
}

/// Ctrl-W: kill the previous word together with the whitespace before it —
/// repeated presses eat the line word by word.
fn delete_word_back(app: &mut App) {
    let i = app.input.caret();
    let before = &app.input.text[..i];
    let cut = before.trim_end().rfind(char::is_whitespace).unwrap_or(0);
    app.input.text.replace_range(cut..i, "");
    app.input.cursor = cut;
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
    use super::super::model::TICK;
    use crate::display::Usage;
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
    fn editor_editing_combining_marks_keeps_them_whole() {
        let mut a = App::new();
        type_str(&mut a, "e\u{301}x"); // é as e + combining acute, then x
        drive(&mut a, vec![key(KeyCode::Left, KeyModifiers::NONE)]); // caret after the é, before x
        drive(&mut a, vec![key(KeyCode::Backspace, KeyModifiers::NONE)]);
        assert_eq!(a.input.text, "x", "the whole grapheme went, not its mark alone");
        assert_eq!(a.input.cursor, 0);
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
    fn enter_submits_clears_the_line_and_remembers_history() {
        let mut a = App::new();
        type_str(&mut a, "count *.rs");
        let acts = drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(acts, vec![Action::Submit("count *.rs".into())]);
        assert_eq!(a.input.text, "", "the line clears for the next task");
        assert_eq!(a.input.cursor, 0);
        // history recall parks and returns a fresh draft
        type_str(&mut a, "abc");
        drive(&mut a, vec![key(KeyCode::Up, KeyModifiers::NONE)]);
        assert_eq!(a.input.text, "count *.rs");
        drive(&mut a, vec![key(KeyCode::Down, KeyModifiers::NONE)]);
        assert_eq!(a.input.text, "abc", "down returns to the parked draft");
    }

    #[test]
    fn ctrl_j_and_shift_enter_break_lines_backspace_joins() {
        let mut a = App::new();
        type_str(&mut a, "one");
        drive(&mut a, vec![key(KeyCode::Char('j'), KeyModifiers::CONTROL)]);
        assert_eq!(a.input.text, "one\n", "Ctrl-J breaks the line");
        type_str(&mut a, "two");
        drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::SHIFT)]);
        assert_eq!(a.input.text, "one\ntwo\n", "Shift-Enter breaks the line");
        // a multi-line draft submits whole
        type_str(&mut a, "three");
        let acts = drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(acts, vec![Action::Submit("one\ntwo\nthree".into())]);
        // Home is line-wise: the caret stops at the start of its line
        let mut b = App::new();
        type_str(&mut b, "aa\nbb");
        drive(&mut b, vec![key(KeyCode::Home, KeyModifiers::NONE)]);
        assert_eq!(b.input.cursor, "aa\n".len());
        // Backspace over the line break joins the lines
        drive(&mut b, vec![key(KeyCode::Backspace, KeyModifiers::NONE)]);
        assert_eq!(b.input.text, "aabb");
    }

    #[test]
    fn paste_keeps_line_breaks_and_is_locked_while_working() {
        let mut a = App::new();
        drive(&mut a, vec![Ev::Paste("one\ntwo".into())]);
        assert_eq!(a.input.text, "one\ntwo", "pasted breaks survive");
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("busy".into()))]);
        drive(&mut a, vec![Ev::Paste("x".into())]);
        assert_eq!(a.input.text, "one\ntwo", "the editor is locked while working");
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

    #[test]
    fn slash_exit_quits_without_submitting_or_recall() {
        let mut a = App::new();
        type_str(&mut a, "/exit");
        let acts = drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(acts, vec![Action::None], "a command submits no task");
        assert!(a.quit);
        assert_eq!(a.input.text, "", "the line clears on the way out");
        assert!(a.input.history.is_empty(), "commands are not tasks to recall");
        // surrounded by whitespace it is still the command
        let mut b = App::new();
        type_str(&mut b, " /exit ");
        drive(&mut b, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert!(b.quit, "trimmed to the command word");
        // a task that merely mentions /exit is a task
        let mut c = App::new();
        type_str(&mut c, "run /exit in bash");
        let acts = drive(&mut c, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert_eq!(acts, vec![Action::Submit("run /exit in bash".into())]);
        assert!(!c.quit);
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
        assert_eq!(a.scroll, Scroll::Up(10));
        assert!(!a.scroll.is_tail());
        // new content must not yank the view while the user reads
        drive(&mut a, vec![Ev::Msg(Msg::Text("fresh\n".into()))]);
        assert!(!a.scroll.is_tail(), "still reading");
        // walking back to the bottom resumes following
        drive(&mut a, vec![key(KeyCode::Char('G'), KeyModifiers::SHIFT)]); // 'G'
        assert_eq!(a.scroll, Scroll::Tail);
    }

    // ---- messages ----

    #[test]
    fn thought_duration_is_measured_between_first_delta_and_fold() {
        let mut a = App::new();
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("t".into()))]);
        drive(&mut a, vec![Ev::Tick(TICK * 2)]); // thinking has not started yet
        drive(&mut a, vec![Ev::Msg(Msg::Think("l1".into()))]);
        drive(&mut a, vec![Ev::Tick(TICK), Ev::Tick(TICK * 9)]);
        drive(&mut a, vec![Ev::Msg(Msg::ThinkEnd)]);
        match a.rows.last() {
            Some(Row::Thought(f)) => assert_eq!(f.duration, Some(TICK * 10), "anchored at the first delta"),
            r => panic!("thought folded: {r:?}"),
        }
        // outside a task there is no clock to read — no duration claimed
        let mut b = App::new();
        drive(&mut b, vec![Ev::Msg(Msg::Think("musing".into())), Ev::Msg(Msg::ThinkEnd)]);
        match b.rows.last() {
            Some(Row::Thought(f)) => assert_eq!(f.duration, None),
            r => panic!("thought folded: {r:?}"),
        }
    }

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
    fn a_run_ends_with_a_separator_row() {
        let a = seeded();
        assert!(matches!(a.rows.last(), Some(Row::Sep)), "TaskEnd lands a named separator");
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
    fn pane_floor_is_a_high_water_mark_reset_per_task() {
        let mut a = App::new();
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("t".into()))]);
        assert_eq!(a.status.pane_floor, 0, "a task starts tight");
        drive(&mut a, vec![Ev::Msg(Msg::Think("l1\nl2\nl3\nl4\nl5".into()))]);
        assert_eq!(a.status.pane_floor, 4, "the head plus the capped tail");
        drive(&mut a, vec![Ev::Msg(Msg::ThinkEnd)]);
        assert_eq!(a.status.pane_floor, 4, "a fold does not lower the floor");
        drive(&mut a, vec![Ev::Msg(Msg::Think("one line".into()))]);
        assert_eq!(a.status.pane_floor, 4, "a smaller block does not either");
        drive(&mut a, vec![Ev::Msg(Msg::Text("partial".into()))]);
        assert_eq!(a.status.pane_floor, 4, "the partial row fits under the mark");
        drive(&mut a, vec![Ev::Msg(Msg::TaskEnd)]);
        assert_eq!(a.status.pane_floor, 4, "idle keeps it — the bar keeps its row");
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("next".into()))]);
        assert_eq!(a.status.pane_floor, 0, "the next task starts tight again");
    }

    #[test]
    fn ticks_accumulate_measured_deltas_not_fixed_steps() {
        let mut a = App::new();
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("t".into()))]);
        // a busy loop skips ticks; the next tick carries the whole gap
        drive(&mut a, vec![Ev::Tick(TICK), Ev::Tick(TICK * 7)]);
        assert_eq!(a.status.task.as_ref().unwrap().elapsed, TICK * 8);
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

    // ---- the slash-command menu ----

    fn fake_candidates(kind: CompletionKind, prefix: &str) -> Vec<CompletionItem> {
        // A standalone candidate source for tests: nothing profile- or
        // hub-specific — a fixed map the test then asserts on.
        match kind {
            CompletionKind::Command => vec![
                CompletionItem { insert: "exit".into(), label: "/exit".into(), description: "leave".into(), trailing_space: false },
                CompletionItem { insert: "model".into(), label: "/model".into(), description: "switch profile".into(), trailing_space: true },
                CompletionItem { insert: "mcp".into(), label: "/mcp".into(), description: "manage servers".into(), trailing_space: true },
            ],
            CompletionKind::ModelArg => vec![
                CompletionItem { insert: "minimax/MiniMax-M3".into(), label: "minimax/MiniMax-M3".into(), description: "default".into(), trailing_space: false },
                CompletionItem { insert: "zai/glm-5.3-flash".into(), label: "zai/glm-5.3-flash".into(), description: "zai".into(), trailing_space: false },
            ],
            CompletionKind::McpSub => vec![
                CompletionItem { insert: "list".into(), label: "list".into(), description: "list servers".into(), trailing_space: true },
                CompletionItem { insert: "enable".into(), label: "enable".into(), description: "re-enable".into(), trailing_space: true },
            ],
            CompletionKind::McpServer => vec![
                CompletionItem { insert: "alpha".into(), label: "alpha".into(), description: "first server".into(), trailing_space: false },
                CompletionItem { insert: "beta".into(), label: "beta".into(), description: "second server".into(), trailing_space: false },
            ],
            _ => vec![],
        }
        .into_iter()
        .filter(|c| c.insert.starts_with(prefix) || c.label.trim_start_matches('/').starts_with(prefix) || prefix.is_empty())
        .collect()
    }

    #[test]
    fn detect_lists_commands_for_empty_or_partial_slash() {
        assert_eq!(detect_completion(""), None);
        assert_eq!(detect_completion("count files"), None);
        assert_eq!(detect_completion("/"), Some((CompletionKind::Command, String::new())));
        assert_eq!(detect_completion("/mod"), Some((CompletionKind::Command, "mod".into())));
        assert_eq!(detect_completion("/model"), Some((CompletionKind::Command, "model".into())));
        assert_eq!(detect_completion("/xyz"), Some((CompletionKind::Command, "xyz".into())));
    }

    #[test]
    fn detect_transitions_to_model_arg_on_space() {
        assert_eq!(detect_completion("/model "), Some((CompletionKind::ModelArg, String::new())));
        assert_eq!(detect_completion("/model m"), Some((CompletionKind::ModelArg, "m".into())));
        assert_eq!(detect_completion("/model m "), None, "no completion after the arg + space");
    }

    #[test]
    fn detect_walks_through_mcp_levels() {
        assert_eq!(detect_completion("/mcp "), Some((CompletionKind::McpSub, String::new())));
        assert_eq!(detect_completion("/mcp e"), Some((CompletionKind::McpSub, "e".into())));
        assert_eq!(detect_completion("/mcp enable "), Some((CompletionKind::McpServer, String::new())));
        assert_eq!(detect_completion("/mcp enable a"), Some((CompletionKind::McpServer, "a".into())));
        assert_eq!(detect_completion("/mcp enable alpha "), None);
    }

    #[test]
    fn detect_closes_the_menu_after_no_arg_commands() {
        assert_eq!(detect_completion("/exit "), None);
        assert_eq!(detect_completion("/help "), None);
    }

    #[test]
    fn set_completion_keeps_selection_across_same_kind_same_prefix() {
        let mut a = App::new();
        let cands = fake_candidates(CompletionKind::Command, "");
        set_completion(&mut a, CompletionKind::Command, "", cands.clone());
        a.completion.as_mut().unwrap().selected = 1;
        // refresh with the same kind + prefix: selection survives
        set_completion(&mut a, CompletionKind::Command, "", cands);
        assert_eq!(a.completion.as_ref().unwrap().selected, 1);
    }

    #[test]
    fn set_completion_resets_selection_on_prefix_change() {
        let mut a = App::new();
        set_completion(&mut a, CompletionKind::Command, "", fake_candidates(CompletionKind::Command, ""));
        a.completion.as_mut().unwrap().selected = 2;
        // the prefix narrowed: a fresh menu — selection returns to the top
        set_completion(&mut a, CompletionKind::Command, "mo", fake_candidates(CompletionKind::Command, "mo"));
        assert_eq!(a.completion.as_ref().unwrap().selected, 0);
    }

    #[test]
    fn set_completion_clamps_selection_when_candidates_shrink() {
        let mut a = App::new();
        set_completion(&mut a, CompletionKind::Command, "", fake_candidates(CompletionKind::Command, ""));
        a.completion.as_mut().unwrap().selected = 3; // past the end after filtering
        set_completion(&mut a, CompletionKind::Command, "mo", fake_candidates(CompletionKind::Command, "mo"));
        let c = a.completion.as_ref().unwrap();
        assert!(c.selected < c.candidates.len(), "selected clamps into the new list");
    }

    #[test]
    fn completion_up_and_down_wrap_within_candidates() {
        let mut a = App::new();
        set_completion(&mut a, CompletionKind::Command, "", fake_candidates(CompletionKind::Command, ""));
        let c = a.completion.as_mut().unwrap();
        assert_eq!(c.selected, 0);
        completion_up(&mut a);
        let c = a.completion.as_ref().unwrap();
        assert_eq!(c.selected, c.candidates.len() - 1, "Up from the top wraps to the bottom");
        completion_down(&mut a);
        assert_eq!(a.completion.as_ref().unwrap().selected, 0, "Down from the bottom wraps to the top");
    }

    #[test]
    fn apply_completion_replaces_prefix_and_appends_trailing_space() {
        let mut a = App::new();
        type_str(&mut a, "/mod");
        set_completion(&mut a, CompletionKind::Command, "mod", fake_candidates(CompletionKind::Command, "mod"));
        assert!(apply_completion(&mut a));
        assert_eq!(a.input.text, "/model ", "the prefix is replaced and a space follows");
        assert_eq!(a.input.cursor, "/model ".len(), "caret sits after the inserted text");
    }

    #[test]
    fn apply_completion_for_arg_does_not_add_trailing_space() {
        let mut a = App::new();
        type_str(&mut a, "/model mini");
        set_completion(&mut a, CompletionKind::ModelArg, "mini", fake_candidates(CompletionKind::ModelArg, "mini"));
        assert!(apply_completion(&mut a));
        assert_eq!(a.input.text, "/model minimax/MiniMax-M3", "profile key replaces prefix in place");
        assert_eq!(a.input.cursor, "/model minimax/MiniMax-M3".len());
    }

    #[test]
    fn tab_with_no_menu_is_a_no_op() {
        let mut a = App::new();
        type_str(&mut a, "/mod");
        a.completion = None;
        let before = a.input.text.clone();
        drive(&mut a, vec![key(KeyCode::Tab, KeyModifiers::NONE)]);
        assert_eq!(a.input.text, before, "Tab without a menu does nothing");
    }

    #[test]
    fn tab_while_menu_open_applies_the_highlighted_row() {
        let mut a = App::new();
        type_str(&mut a, "/mod");
        set_completion(&mut a, CompletionKind::Command, "mod", fake_candidates(CompletionKind::Command, "mod"));
        let acts = drive(&mut a, vec![key(KeyCode::Tab, KeyModifiers::NONE)]);
        assert_eq!(acts, vec![Action::None]);
        assert!(a.input.text.starts_with("/model "), "Tab applied the selected row");
    }

    #[test]
    fn arrows_while_menu_open_navigate_instead_of_recalling_history() {
        let mut a = App::new();
        type_str(&mut a, "/");
        set_completion(&mut a, CompletionKind::Command, "", fake_candidates(CompletionKind::Command, ""));
        // history is empty; Up/Down would otherwise be no-ops on history too,
        // so instead seed the menu and assert selection moves
        drive(&mut a, vec![key(KeyCode::Down, KeyModifiers::NONE)]);
        assert_eq!(a.completion.as_ref().unwrap().selected, 1, "Down advanced the highlight");
        drive(&mut a, vec![key(KeyCode::Up, KeyModifiers::NONE)]);
        assert_eq!(a.completion.as_ref().unwrap().selected, 0, "Up went back");
    }

    #[test]
    fn esc_dismisses_the_menu_without_changing_input() {
        let mut a = App::new();
        type_str(&mut a, "/mod");
        set_completion(&mut a, CompletionKind::Command, "mod", fake_candidates(CompletionKind::Command, "mod"));
        drive(&mut a, vec![key(KeyCode::Esc, KeyModifiers::NONE)]);
        assert!(a.completion.is_none());
        assert_eq!(a.input.text, "/mod", "Esc closes the menu, not the input");
    }

    #[test]
    fn submitting_a_command_clears_the_menu() {
        let mut a = App::new();
        type_str(&mut a, "/help");
        set_completion(&mut a, CompletionKind::Command, "help", fake_candidates(CompletionKind::Command, "help"));
        let acts = drive(&mut a, vec![key(KeyCode::Enter, KeyModifiers::NONE)]);
        assert!(a.completion.is_none(), "the menu closes when Enter fires");
        assert_eq!(acts, vec![Action::Submit("/help".into())]);
    }

    #[test]
    fn working_drops_any_open_menu() {
        let mut a = App::new();
        type_str(&mut a, "/");
        set_completion(&mut a, CompletionKind::Command, "", fake_candidates(CompletionKind::Command, ""));
        drive(&mut a, vec![Ev::Msg(Msg::TaskBegin("busy".into()))]);
        assert!(a.completion.is_none(), "TaskBegin locks the editor and the menu");
    }
}
