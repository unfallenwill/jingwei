//! The display port: the agent core never touches a terminal. It emits
//! [`Msg`]s through [`disp`], and a *frontend* interprets them — the plain
//! frontend (`crate::plain`) folds them into a log, the TUI (`crate::tui`)
//! folds them into application state.
//!
//! This module is only the contract: the message type, the usage shape it
//! carries, and the small vocabulary both frontends render with (the
//! prompt, the thought marker, measurement). No frontend lives here — the
//! default plain instance is constructed by `disp` on demand, nothing else.
//!
//! With no frontend installed, `disp` routes to a default plain instance —
//! agent-core tests exercise the protocols without driving any UI.

use serde_json::Value;
use std::env;
use std::time::Duration;
use std::io::{self, IsTerminal};
use std::sync::{Mutex, OnceLock};
use tokio::sync::mpsc::UnboundedSender;

/// Severity of a note row (warnings scroll with the transcript).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sev {
    Warn,
    Err,
}

/// Everything the agent core wants the human to see.
///
/// The protocol's ordering contract — frontends are free to rely on it:
///
/// ```text
/// TaskBegin(text)
///   ( Think.. ThinkEnd | Text.. | Tool | Note | Usage | OutTokens )*   one run
///   Done                                                               fold turn usage into totals
/// TaskEnd
/// ```
///
/// `Usage` replaces the live turn's counters wholesale; `OutTokens` ticks
/// the output count while text streams and is superseded by the next
/// `Usage`. A run interrupted mid-flight still ends with `Done`/`TaskEnd`
/// (whatever was reasoned is folded first), so frontends never dangle.
#[derive(Clone, Debug)]
pub enum Msg {
    /// A dim information line (the banner).
    Banner(String),
    /// The user's task is being worked on: echoed, status bar starts.
    TaskBegin(String),
    /// The run finished (normally, or interrupted).
    TaskEnd,
    /// Streamed answer text; may carry partial lines and newlines. The
    /// final chunk of a turn need not end in `\n` — frontends land it on
    /// `Done`.
    Text(String),
    /// Streamed reasoning delta — folded away, never shown inline.
    Think(String),
    /// The reasoning block closed: fold it into a marker.
    ThinkEnd,
    /// A tool call completed: name, argument summary, full output.
    Tool { name: String, summary: String, output: String },
    /// A warning or error line.
    Note { sev: Sev, text: String },
    /// The latest request's usage (replaces the live turn).
    Usage(Usage),
    /// Streaming output-token count (ticks up while text flows).
    OutTokens(u64),
    /// The request finished: fold the turn into session totals.
    Done,
}

/// Token usage of one API request — or session totals. Part of the [`Msg`]
/// contract, so it lives with it.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
}

impl Usage {
    /// From an internal-shape usage object (Anthropic keys, or OpenAI mapped).
    pub fn from_value(v: &Value) -> Self {
        let g = |k: &str| v[k].as_u64().unwrap_or(0);
        Self {
            input: g("input_tokens"),
            output: g("output_tokens"),
            cache_read: g("cache_read_input_tokens"),
            cache_write: g("cache_creation_input_tokens"),
        }
    }
    /// Everything the model read this request: plain input plus cache traffic.
    pub fn context_in(&self) -> u64 {
        self.input + self.cache_read + self.cache_write
    }
    pub fn add(&mut self, o: &Usage) {
        self.input += o.input;
        self.output += o.output;
        self.cache_read += o.cache_read;
        self.cache_write += o.cache_write;
    }
    pub fn is_zero(&self) -> bool {
        self.input == 0 && self.output == 0 && self.cache_read == 0 && self.cache_write == 0
    }
}

// ---- shared display vocabulary ---------------------------------------------
//
// Both frontends render the same prompt, the same thought marker, and the
// same severity colors. The strings live here, once, so the log and the TUI
// cannot drift apart.

/// The prompt's head and gutter; [`prompt_w`] is derived, never restated.
pub const PROMPT_HEAD: &str = "jingwei";
pub const PROMPT_GUTTER: &str = " ❯ ";

/// Display columns the prompt occupies — computed from the strings above so
/// editing the prompt cannot silently break alignment.
pub fn prompt_w() -> usize {
    disp_width(PROMPT_HEAD) + disp_width(PROMPT_GUTTER)
}

/// The gutter a fold's body hangs from — `│ ` — shared by both frontends
/// so the block's visual grouping reads the same in the log and the TUI.
/// Structural, not chromatic: it survives NO_COLOR and DIM-blind terminals.
pub const FOLD_GUTTER: &str = "│ ";

/// Backgrounds for short, high-attention signals. Warn is honey, Err is red.
pub const WARN_BG: (u8, u8, u8) = (255, 220, 100);
pub const ERR_BG: (u8, u8, u8) = (198, 40, 40);

/// Paint a short, high-attention badge: a truecolor background with the
/// text in white (or near-black). The only place raw ANSI escape codes are
/// spelled out — the entry point's fatal errors use it before any frontend
/// exists; everything else renders through a frontend.
pub fn paint(s: &str, bg: (u8, u8, u8), white: bool) -> String {
    if color_on() {
        format!("\x1b[{}m\x1b[48;2;{};{};{}m{s}\x1b[49m\x1b[39m", if white { 97 } else { 30 }, bg.0, bg.1, bg.2)
    } else {
        s.into()
    }
}

/// The one-line marker a *folded* reasoning block shows — the one form
/// both frontends render: `▸ thought #3 · checking Cargo.toml first … +13`.
/// Content beats counts for the expand-or-skip decision, so the collapsed
/// marker previews the block's first line and names how much is hidden
/// behind it. The state glyph lives here, not in the caller.
pub fn thought_folded(n: usize, first: &str, more: usize) -> String {
    let first = first.trim();
    match (first.is_empty(), more) {
        (true, _) => format!("▸ thought #{n}"),
        (false, 0) => format!("▸ thought #{n} · {first}"),
        (false, m) => format!("▸ thought #{n} · {first} … +{m}"),
    }
}

/// Elapsed time, the way both the status bar and the fold markers read it:
/// `3s`, then `1m03s`. A shared vocabulary word, not two spellings.
pub fn elapsed_str(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 { format!("{s}s") } else { format!("{}m{:02}s", s / 60, s % 60) }
}

/// The one REPL command both frontends speak: `/exit` leaves the session.
/// One spelling, defined once — the TUI's editor and the plain reader
/// agree on exactly this word, so the command works wherever a prompt is.
pub const EXIT_COMMAND: &str = "/exit";

/// Does a submitted line ask to leave? Both frontends trim before they
/// look, so a trailing space is not a typo — but a task that merely
/// *mentions* the word is still a task.
pub fn is_exit(line: &str) -> bool {
    line.trim() == EXIT_COMMAND
}

/// Should we color at all? Honors `NO_COLOR`/`JINGWEI_NO_COLOR`, requires a
/// terminal, and `JINGWEI_COLOR=always|1|true` forces color on (useful when
/// jingwei's output rides a pipe into a color-aware pager).
pub fn color_on() -> bool {
    if matches!(env::var("JINGWEI_COLOR").as_deref(), Ok("always" | "1" | "true")) {
        return true;
    }
    env::var_os("NO_COLOR").is_none()
        && env::var_os("JINGWEI_NO_COLOR").is_none()
        && io::stdout().is_terminal()
        && io::stderr().is_terminal()
}

/// Display width of a string, per Unicode (east-asian wide = 2, combining
/// marks = 0). Both frontends measure with the same ruler.
pub fn disp_width(s: &str) -> usize {
    use unicode_width::UnicodeWidthStr;
    s.width()
}

/// Cut to `max` display columns, marking the cut with an ellipsis. Walks
/// graphemes so a combining mark is never severed from its base.
pub fn truncate_cols(s: &str, max: usize) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    let mut out = String::new();
    let mut w = 0;
    for g in s.graphemes(true) {
        let gw = disp_width(g).max(1);
        if w + gw > max.saturating_sub(2) {
            out.push('…');
            return out;
        }
        out.push_str(g);
        w += gw;
    }
    out
}

/// Wrap to `max` display columns, grapheme-greedy: every chunk fits, no
/// ellipsis — the review mode's ruler, where reading the whole line beats
/// keeping the row count. Empty input yields one empty chunk, so a wrapped
/// line never loses its row.
pub fn wrap_cols(s: &str, max: usize) -> Vec<String> {
    use unicode_segmentation::UnicodeSegmentation;
    let max = max.max(1);
    let mut out = vec![];
    let mut cur = String::new();
    let mut w = 0;
    for g in s.graphemes(true) {
        let gw = disp_width(g).max(1);
        if w + gw > max && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            w = 0;
        }
        cur.push_str(g);
        w += gw;
    }
    out.push(cur);
    out
}

/// The installed frontend. TUI installs a channel; plain is the default.
static FRONT: OnceLock<Front> = OnceLock::new();

enum Front {
    Chan(UnboundedSender<Msg>),
    Plain(Mutex<crate::plain::Plain>),
}

/// Install the TUI frontend: messages flow to its event loop.
pub fn install_chan(tx: UnboundedSender<Msg>) {
    let _ = FRONT.set(Front::Chan(tx));
}

/// Route one message to the installed frontend. Infallible and quiet: a
/// dropped TUI receiver (or no frontend at all) must never take the agent
/// down — the sea does not care whether anyone is watching.
pub fn disp(m: Msg) {
    match FRONT.get_or_init(|| Front::Plain(Mutex::new(crate::plain::Plain::new()))) {
        Front::Chan(tx) => {
            let _ = tx.send(m);
        }
        Front::Plain(p) => {
            let mut p = p.lock().unwrap();
            for (stream, line) in p.feed(m) {
                match stream {
                    crate::plain::Stream::Out => println!("{line}"),
                    crate::plain::Stream::Err => eprintln!("{line}"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thought_folded_previews_content_and_counts_the_hidden() {
        assert_eq!(thought_folded(1, "half a thought", 0), "▸ thought #1 · half a thought");
        assert_eq!(thought_folded(3, "step one", 13), "▸ thought #3 · step one … +13");
        // an empty first line previews nothing rather than a dangling dot
        assert_eq!(thought_folded(2, "  ", 4), "▸ thought #2");
    }

    #[test]
    fn elapsed_str_reads_clock_like_a_person() {
        assert_eq!(elapsed_str(Duration::from_secs(3)), "3s");
        assert_eq!(elapsed_str(Duration::from_secs(59)), "59s");
        assert_eq!(elapsed_str(Duration::from_secs(63)), "1m03s");
    }

    #[test]
    fn is_exit_matches_the_one_word_not_its_mentions() {
        assert!(is_exit("/exit"));
        assert!(is_exit("  /exit  "), "the trim both frontends apply counts");
        assert!(!is_exit("/exit now"), "an argument makes it a task");
        assert!(!is_exit("run /exit in a shell"), "mentioning the word is not asking");
        assert!(!is_exit(""));
    }

    #[test]
    fn prompt_w_matches_the_strings_it_is_derived_from() {
        assert_eq!(prompt_w(), disp_width(&format!("{PROMPT_HEAD}{PROMPT_GUTTER}")));
    }

    #[test]
    fn truncate_cols_respects_wide_chars() {
        assert_eq!(truncate_cols("abc", 10), "abc");
        assert_eq!(truncate_cols("精卫填海", 5), "精…");
        let long = "x".repeat(80);
        let t = truncate_cols(&long, 60);
        assert!(t.ends_with('…') && t.chars().count() == 59, "cut to ~60 columns: {t}");
    }

    #[test]
    fn wrap_cols_fits_every_chunk_and_keeps_empty_rows() {
        assert_eq!(wrap_cols("", 10), vec!["".to_string()], "an empty line keeps its row");
        assert_eq!(wrap_cols("ab", 10), vec!["ab".to_string()]);
        assert_eq!(wrap_cols("abcd", 2), vec!["ab".to_string(), "cd".to_string()]);
        // wide chars never split across chunks
        let chunks = wrap_cols("精卫填海", 4);
        assert_eq!(chunks, vec!["精卫".to_string(), "填海".to_string()]);
        // every chunk fits, nothing is lost, and the join is the original
        let long = format!("{} 精卫 {}", "x".repeat(37), "y".repeat(41));
        let chunks = wrap_cols(&long, 20);
        assert!(chunks.iter().all(|c| disp_width(c) <= 20), "{chunks:?}");
        assert_eq!(chunks.concat(), long);
    }

    #[test]
    fn disp_width_counts_combining_marks_as_zero() {
        // e + combining acute is ONE column, not three — the ruler the caret
        // math relies on
        assert_eq!(disp_width("e\u{301}"), 1);
        assert_eq!(disp_width("精卫"), 4);
    }

    #[test]
    fn truncate_cols_never_severs_a_grapheme() {
        // cutting near "é" (e + U+0301) takes or leaves the whole grapheme — never the mark alone
        assert_eq!(truncate_cols("ab\u{301}cd", 4), "ab\u{301}…");
        assert_eq!(truncate_cols("ab\u{301}cd", 5), "ab\u{301}c…");
    }

    #[test]
    fn usage_maps_internal_shape_and_folds_totals() {
        let v = serde_json::json!({"input_tokens": 10, "output_tokens": 0,
            "cache_read_input_tokens": 5, "cache_creation_input_tokens": 2});
        let mut u = Usage::from_value(&v);
        assert_eq!(u.context_in(), 17); // the model read all of it
        u.output = 3;
        let mut total = Usage::default();
        total.add(&u);
        total.add(&u);
        assert_eq!((total.input, total.output, total.cache_read, total.cache_write), (20, 6, 10, 4));
    }
}
