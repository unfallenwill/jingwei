//! The display port: the agent core never touches a terminal. It emits
//! [`Msg`]s through a [`Show`] sink, and a *frontend* interprets them — the
//! plain frontend (`crate::ui::plain`) folds them into a log, the TUI
//! (`crate::ui::tui`) folds them into application state, and a remote adapter (a
//! web socket, say) would ship them as JSON to a browser. No frontend lives
//! here: the port is the contract ([`Msg`], [`Usage`], the [`Show`] trait)
//! plus the small rendering vocabulary the terminal frontends share. Nothing
//! imports a concrete frontend; there is no process-global sink — the
//! composition root hands the core whichever frontend owns this run's
//! transport.

use serde_json::Value;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use std::io::{self, IsTerminal};
use unicode_segmentation::UnicodeSegmentation;
use tokio::sync::mpsc::UnboundedSender;

pub(crate) mod plain;
pub(crate) mod tui;

// The port names the agent's identity and the run's lifecycle, and both
// live below it: `AgentId` in `core::ids`, `TurnState` in `core::turn`.
// The port is public, so it re-exports them — a contract that hands a
// consumer a type it cannot name is not a contract.
pub use crate::core::ids::AgentId;
pub use crate::core::turn::TurnState;

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
///   ( Think.. ThinkEnd | Text.. | Tool | Note | Usage | OutTokens )*   one request
///   Done                                                               close that request's reading
/// TaskEnd
/// ```
///
/// `Usage` opens a request's usage reading; `Done` closes it, and the
/// reading is folded into the session totals exactly once. A `Done` with
/// no `Usage` since the last `Done` is not an error: it closes nothing,
/// and the accounting leaves the numbers alone. A stream that reports no
/// usage ends with exactly that — a `Done` and no reading to close.
///
/// `Usage` replaces the live turn's counters wholesale; `OutTokens` ticks
/// the output count while text streams and refines the reading the
/// `Usage` carried. A request cut short mid-stream still sends its
/// `Done` — the stream ends and the pair closes either way — while one
/// cut short before it reported anything closes nothing. A run
/// interrupted mid-flight still ends with `TaskEnd` (whatever was
/// reasoned is folded first), so frontends never dangle.
///
/// > A background agent's events arrive **wrapped**: `FromAgent { id, event }`.
/// > The wrapper carries the attribution and nothing else — the inner events
/// > keep the same relative order they would have had unwrapped, and the
/// > inner `Usage` / `Done` belong to *that* agent, never to the session's
/// > totals. A worker's events never arrive unwrapped, and the main
/// > conversation's events never arrive wrapped.
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
    /// The latest request's usage (replaces the live turn, and opens the
    /// reading this request's `Done` will close).
    Usage(Usage),
    /// Streaming output-token count (ticks up while text flows) — the
    /// refinement of the reading the last `Usage` opened.
    OutTokens(u64),
    /// The request finished: close its usage reading (fold it into the
    /// session totals exactly once).
    Done,
    /// A background agent's event. The wrapper is the attribution: the
    /// inner event belongs to `id`, not to the main conversation.
    FromAgent { id: AgentId, event: Box<AgentMsg> },
}

/// What a background agent can say.
///
/// A closed vocabulary, and deliberately narrower than [`Msg`]: a worker
/// never brackets a run (`TaskBegin` / `TaskEnd`), never ends the human's
/// turn (`Done`), and never reports session usage (`Usage` /
/// `OutTokens`). Those belong to the main conversation. A worker's own
/// tokens are its own business — the main session's totals do not grow by
/// a worker's spending, and this vocabulary is what makes that structural
/// rather than a rule somebody has to remember.
#[derive(Clone, Debug)]
pub enum AgentMsg {
    /// The agent started, with the label the human sees.
    Started { label: String },
    /// A chunk of answer text.
    Text(String),
    /// A chunk of reasoning.
    Think(String),
    /// The reasoning block closed: fold it into a marker.
    ThinkEnd,
    /// A tool call completed: name, argument summary, full output.
    Tool { name: String, summary: String, output: String },
    /// A note — a warning, an error, the final report.
    Note { sev: Sev, text: String },
    /// The agent reached a terminal state, with its report.
    Finished { state: TurnState, report: String },
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
}

// ---- shared display vocabulary ---------------------------------------------
//
// Both frontends render the same prompt, the same thought marker, and the
// same severity colors. The strings live here, once, so the log and the TUI
// cannot drift apart.

/// What a failed run looks like, in the two interactive frontends.
///
/// One error line — and silence for an interrupted run: the user asked
/// for that one, so it is not a failure to report. The one-shot does
/// not use this; it propagates the error instead, so the process exit
/// code stays right and the binary paints the banner.
///
/// The run boundary calls this *inside* the run's bracket. A note that
/// lands after the run has ended reads as a footnote to nothing.
pub fn error_note(e: &crate::core::error::Error) -> Option<Msg> {
    match e {
        crate::core::error::Error::Interrupted => None,
        e => Some(Msg::Note { sev: Sev::Err, text: format!(" error: {e} ") }),
    }
}

/// The prompt's gutter; [`prompt_w`] is derived, never restated.
/// Just `❯` — no name prefix, the editor pane says where it is by where it sits.
pub const PROMPT_HEAD: &str = "";
pub const PROMPT_GUTTER: &str = "❯ ";

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

/// The one glyph per state, in the vocabulary the rest of the port
/// already prints. Ambiguous-width like `▸`/`❯`, which the tree uses.
///
/// The state itself is `core::turn`'s — the port does not keep a second
/// copy of "is this agent still running" — but the *glyph* is a display
/// concern, so it lives here with the rest of the vocabulary.
pub fn state_glyph(state: TurnState) -> &'static str {
    match state {
        TurnState::Queued => "○",
        TurnState::InProgress => "▸",
        TurnState::Completed => "✓",
        TurnState::Failed => "✗",
        TurnState::Interrupted => "⊘",
        TurnState::Canceled => "◌",
    }
}

/// The tag every line a background agent produces carries: the agent's
/// id in brackets — `[a1] `.
///
/// One spelling, defined once, in the port both frontends render
/// through: a log line and a transcript row can never disagree about
/// who spoke. The main agent's own lines carry no tag — the
/// conversation *is* the transcript, and it needs no speaker labels.
pub fn agent_tag(id: AgentId) -> String {
    format!("[{id}] ")
}

/// The one-line summary a finished background agent leaves behind:
/// `[a1] ✓ build · 48s · 23 tools`.
///
/// The parts are the facts a frontend actually has: the tag, the state
/// glyph, the label the fleet gave the agent, the elapsed time, and the
/// number of tools it called. The elapsed time is optional — the TUI
/// owns a clock and the plain log has none, and a frontend with no clock
/// does not fake one. An empty label is left out, so the line says only
/// what it knows.
pub fn agent_summary(
    id: AgentId,
    state: TurnState,
    label: &str,
    elapsed: Option<Duration>,
    tools: usize,
) -> String {
    let mut head = format!("{}{}", agent_tag(id), state_glyph(state));
    if !label.is_empty() {
        head.push(' ');
        head.push_str(label);
    }
    let mut parts = vec![head];
    if let Some(d) = elapsed {
        parts.push(elapsed_str(d));
    }
    parts.push(if tools == 1 { "1 tool".to_string() } else { format!("{tools} tools") });
    parts.join(" · ")
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

/// Should we color at all? Honors `NO_COLOR` (the industry-standard opt-out)
/// and gates on `stderr` being a terminal — the only stream `paint` writes
/// to (the fatal-error banner in `main`). `stdout` is intentionally not in
/// the gate: this function is about one specific banner, and a tty-stderr /
/// piped-stdout layout should still paint a coloured fatal line.
pub fn color_on() -> bool {
    env::var_os("NO_COLOR").is_none()
        && io::stderr().is_terminal()
}

// Re-exported so existing callers keep working; the canonical home is
// `crate::format` (where session.rs can reach them without depending on
// the presentation layer).
pub use crate::format::{disp_width, truncate_cols};

/// Wrap to `max` display columns, grapheme-greedy: every chunk fits, no
/// ellipsis — the review mode's ruler, where reading the whole line beats
/// keeping the row count. Empty input yields one empty chunk, so a wrapped
/// line never loses its row.
pub fn wrap_cols(s: &str, max: usize) -> Vec<String> {
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
/// The display port's sink — the one thing the agent core knows about a
/// frontend. `&self` so one handle serves the coroutine and the shell at
/// once; `Send + Sync` so the same shape carries a terminal, a pipe, and (one
/// day) a web socket. The core is handed a `&dyn Show` and never learns
/// which one — the frontend is a detail the composition root plugs in.
pub trait Show: Send + Sync {
    /// Show one message. Infallible and quiet by contract: a dropped receiver
    /// (or a frontend that stopped listening) must never take the agent down
    /// — the sea does not care whether anyone is watching.
    fn show(&self, m: Msg);
}

/// An `Arc` of a sink is a sink.
///
/// The fleet hands the same sink to the main run and to every worker it
/// spawns, and it has to keep that sink alive while the run it owns is
/// gone. An `Arc` is the shape that does both: one allocation, any number
/// of owners, no clone of the sink's own state. `PlainSink` is not
/// `Clone` — its state is a folded log — so this is the only way a
/// frontend can hand the same log to a worker.
impl<T: Show + ?Sized> Show for Arc<T> {
    fn show(&self, m: Msg) {
        (**self).show(m)
    }
}

/// The common sink: hand each message to a channel another task drains. The
/// terminal TUI and any remote frontend (a web socket, say) work this way —
/// the adapter owns the receiver and the transport, the core owns neither.
/// `Clone`, so a frontend can hand one copy to a spawned agent task and keep
/// another for itself.
#[derive(Clone)]
pub struct ChannelSink(UnboundedSender<Msg>);

impl ChannelSink {
    pub fn new(tx: UnboundedSender<Msg>) -> Self {
        Self(tx)
    }
}

impl Show for ChannelSink {
    fn show(&self, m: Msg) {
        let _ = self.0.send(m);
    }
}

/// A sink that attributes everything it carries to one background agent.
///
/// A worker gets one of these instead of the main sink, so its events
/// reach the frontend already wrapped. The messages that mean nothing in
/// a worker — the run bracketing, the session accounting, the banner —
/// are dropped here, and that is the whole point of the drop list: a
/// worker cannot end the human's turn or spend the session's tokens.
///
/// `Clone`, like the sink it wraps: the shared sink is shared, and the
/// frontend may hold one copy while the worker holds another.
#[derive(Clone)]
pub struct AgentSink {
    inner: Arc<dyn Show>,
    id: AgentId,
}

impl AgentSink {
    /// Wrap a sink so everything it carries belongs to `id`.
    pub fn new(inner: impl Show + 'static, id: AgentId) -> Self {
        Self { inner: Arc::new(inner), id }
    }

    /// The agent this sink speaks for.
    pub fn id(&self) -> AgentId {
        self.id
    }

    /// Announce that the agent started.
    pub fn started(&self, label: String) {
        self.inner.show(Msg::FromAgent {
            id: self.id,
            event: Box::new(AgentMsg::Started { label }),
        });
    }

    /// Announce that the agent reached a terminal state.
    pub fn finished(&self, state: TurnState, report: String) {
        self.inner.show(Msg::FromAgent {
            id: self.id,
            event: Box::new(AgentMsg::Finished { state, report }),
        });
    }
}

impl Show for AgentSink {
    /// The mapping — and the drop list.
    ///
    /// | `Msg` | becomes | why |
    /// |---|---|---|
    /// | `Text(t)` | `AgentMsg::Text(t)` | the worker's prose |
    /// | `Think(t)` | `AgentMsg::Think(t)` | the worker's reasoning |
    /// | `ThinkEnd` | `AgentMsg::ThinkEnd` | the block closed |
    /// | `Tool { name, summary, output }` | `AgentMsg::Tool { … }` | the worker's tool call |
    /// | `Note { sev, text }` | `AgentMsg::Note { sev, text }` | the worker's warning or error |
    /// | `Banner(_)` | *dropped* | a banner is the conversation's announcement; a worker announces nothing |
    /// | `TaskBegin(_)` | *dropped* | a worker does not begin the human's run |
    /// | `TaskEnd` | *dropped* | a worker does not end it either |
    /// | `Usage(_)` / `OutTokens(_)` | *dropped* | a worker's tokens are not the session's tokens |
    /// | `Done` | *dropped* | `Done` closes the *main's* request; the worker's own accounting is its own |
    /// | `FromAgent { .. }` | *dropped* | no nesting: a wrapped event belongs to another agent, not to this one |
    ///
    /// The drop list is the design, not an omission — a worker cannot
    /// end the human's turn, spend the session's tokens, or open a run.
    fn show(&self, m: Msg) {
        let event = match m {
            Msg::Text(t) => AgentMsg::Text(t),
            Msg::Think(t) => AgentMsg::Think(t),
            Msg::ThinkEnd => AgentMsg::ThinkEnd,
            Msg::Tool { name, summary, output } => AgentMsg::Tool { name, summary, output },
            Msg::Note { sev, text } => AgentMsg::Note { sev, text },
            Msg::Banner(_) => return,
            Msg::TaskBegin(_) => return,
            Msg::TaskEnd => return,
            Msg::Usage(_) | Msg::OutTokens(_) => return,
            Msg::Done => return,
            Msg::FromAgent { .. } => return,
        };
        self.inner.show(Msg::FromAgent { id: self.id, event: Box::new(event) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::block_on;

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
    fn paint_passes_through_when_color_is_off() {
        // NO_COLOR on → strict passthrough, no escape sequences either way
        let _env = crate::test_util::env_lock();
        std::env::set_var("NO_COLOR", "1");
        let off = paint("err", ERR_BG, true);
        assert_eq!(off, "err");
        let dark = paint("warn", WARN_BG, false);
        assert_eq!(dark, "warn");
        std::env::remove_var("NO_COLOR");
    }

    #[test]
    fn channel_sink_forwards_messages_through_the_sender() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = ChannelSink::new(tx);
        sink.show(Msg::Text("hi".into()));
        sink.show(Msg::Done);
        // drop the sink so the sender closes — otherwise `rx.recv().await`
        // would block forever, the test would hang, and the whole suite stalls.
        drop(sink);
        let got = block_on(async {
            let mut out = vec![];
            while let Some(m) = rx.recv().await {
                out.push(m);
            }
            out
        });
        assert!(matches!(got[0], Msg::Text(ref t) if t == "hi"));
        assert!(matches!(got.last(), Some(Msg::Done)));
    }

    #[test]
    fn error_note_says_nothing_about_an_interrupted_run() {
        // the user asked for it: it is a stop, not a failure to report
        assert!(error_note(&crate::core::error::Error::Interrupted).is_none());
        // anything else is one error line
        let note = error_note(&crate::core::error::Error::Api(429, "limited".into()))
            .expect("a failed run says something");
        match note {
            Msg::Note { sev, text } => {
                assert_eq!(sev, Sev::Err);
                assert!(text.contains("api 429"), "{text}");
            }
            other => panic!("expected a note, got {other:?}"),
        }
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

    // ---- the attribution: a worker's events arrive wrapped ----------------

    /// A worker id of its own — the main agent's id is spoken for.
    fn worker_id() -> AgentId {
        AgentId::next()
    }

    #[test]
    fn a_workers_events_are_wrapped() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::Text("the worker's prose".into()));
        let msgs = rec.msgs();
        assert_eq!(msgs.len(), 1, "one event, and it is the worker's: {msgs:?}");
        match &msgs[0] {
            Msg::FromAgent { id, event } => {
                assert_eq!(*id, sink.id(), "the wrapper carries the attribution");
                match event.as_ref() {
                    AgentMsg::Text(t) => assert_eq!(t, "the worker's prose"),
                    other => panic!("expected the worker's text, got {other:?}"),
                }
            }
            other => panic!("a worker's event arrives wrapped, never bare: {other:?}"),
        }
    }

    #[test]
    fn the_conversations_events_are_not_wrapped() {
        // the same recorder, reached directly: the wrapper is the
        // adapter's doing, and the main conversation is untouched by it
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::Text("from the worker".into()));
        rec.show(Msg::Text("from the conversation".into()));
        let msgs = rec.msgs();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert!(matches!(&msgs[0], Msg::FromAgent { .. }), "the worker's event is wrapped: {msgs:?}");
        assert!(
            matches!(&msgs[1], Msg::Text(ref t) if t == "from the conversation"),
            "the conversation's event is not: {msgs:?}"
        );
    }

    #[test]
    fn a_worker_cannot_end_the_conversation() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::TaskEnd);
        sink.show(Msg::Done);
        assert!(rec.msgs().is_empty(), "a worker ends nothing: {:?}", rec.msgs());
    }

    #[test]
    fn a_worker_cannot_spend_the_sessions_tokens() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::Usage(Usage { input: 10, output: 5, ..Usage::default() }));
        sink.show(Msg::OutTokens(9));
        assert!(rec.msgs().is_empty(), "a worker's tokens are its own: {:?}", rec.msgs());
    }

    #[test]
    fn a_workers_lifecycle_is_announced() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.started("scout".into());
        sink.finished(TurnState::Completed, "found three files".into());
        let msgs = rec.msgs();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        match &msgs[0] {
            Msg::FromAgent { id, event } => {
                assert_eq!(*id, sink.id());
                match event.as_ref() {
                    AgentMsg::Started { label } => assert_eq!(label, "scout"),
                    other => panic!("expected a start, got {other:?}"),
                }
            }
            other => panic!("expected a wrapped start, got {other:?}"),
        }
        match &msgs[1] {
            Msg::FromAgent { id, event } => {
                assert_eq!(*id, sink.id());
                match event.as_ref() {
                    AgentMsg::Finished { state, report } => {
                        assert_eq!(*state, TurnState::Completed);
                        assert_eq!(report, "found three files");
                    }
                    other => panic!("expected a finish, got {other:?}"),
                }
            }
            other => panic!("expected a wrapped finish, got {other:?}"),
        }
    }

    #[test]
    fn a_workers_tool_call_keeps_its_three_parts() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::Tool {
            name: "read_file".into(),
            summary: "Cargo.toml".into(),
            output: "the file's contents".into(),
        });
        let msgs = rec.msgs();
        match &msgs[0] {
            Msg::FromAgent { event, .. } => match event.as_ref() {
                AgentMsg::Tool { name, summary, output } => {
                    assert_eq!(name, "read_file");
                    assert_eq!(summary, "Cargo.toml");
                    assert_eq!(output, "the file's contents");
                }
                other => panic!("expected a tool event, got {other:?}"),
            },
            other => panic!("expected a wrapped tool event, got {other:?}"),
        }
    }

    #[test]
    fn a_workers_note_keeps_its_severity() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::Note { sev: Sev::Warn, text: "watch out".into() });
        let msgs = rec.msgs();
        match &msgs[0] {
            Msg::FromAgent { event, .. } => match event.as_ref() {
                AgentMsg::Note { sev, text } => {
                    assert_eq!(*sev, Sev::Warn, "the severity is not flattened away");
                    assert_eq!(text, "watch out");
                }
                other => panic!("expected a note event, got {other:?}"),
            },
            other => panic!("expected a wrapped note, got {other:?}"),
        }
    }

    #[test]
    fn a_worker_cannot_open_a_run_or_announce_a_banner() {
        let rec = crate::test_util::Recorder::default();
        let sink = AgentSink::new(rec.clone(), worker_id());
        sink.show(Msg::Banner("jingwei".into()));
        sink.show(Msg::TaskBegin("the human's task".into()));
        assert!(
            rec.msgs().is_empty(),
            "a worker brackets nothing and announces nothing: {:?}",
            rec.msgs()
        );
    }

    #[test]
    fn the_state_glyphs_are_distinct() {
        let all = [
            (TurnState::Queued, "○"),
            (TurnState::InProgress, "▸"),
            (TurnState::Completed, "✓"),
            (TurnState::Failed, "✗"),
            (TurnState::Interrupted, "⊘"),
            (TurnState::Canceled, "◌"),
        ];
        for (state, glyph) in all {
            assert_eq!(state_glyph(state), glyph, "{state:?}");
        }
        // one glyph per state: no two states wear the same face
        let mut seen: Vec<&str> = all.iter().map(|(_, g)| *g).collect();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(seen.len(), before, "the six states wear six glyphs");
    }

    #[test]
    fn the_tag_and_the_summary_name_the_agent() {
        let id = AgentId::next();
        assert_eq!(agent_tag(id), format!("[{id}] "), "the id in brackets");
        // the summary reads: tag, glyph, label, elapsed, tools
        assert_eq!(
            agent_summary(id, TurnState::Completed, "build", Some(Duration::from_secs(48)), 23),
            format!("[{id}] ✓ build · 48s · 23 tools"),
        );
        // a frontend with no clock leaves the time out — it does not fake one
        assert_eq!(
            agent_summary(id, TurnState::Completed, "build", None, 23),
            format!("[{id}] ✓ build · 23 tools"),
        );
        // one tool is one tool
        assert_eq!(
            agent_summary(id, TurnState::Failed, "", None, 1),
            format!("[{id}] ✗ · 1 tool"),
        );
    }

    #[test]
    fn an_arc_of_a_sink_is_a_sink() {
        let rec = crate::test_util::Recorder::default();
        let sink: Arc<dyn Show> = Arc::new(rec.clone());
        sink.show(Msg::Text("through the arc".into()));
        assert_eq!(rec.msgs().len(), 1, "the arc carried the message to the sink under it");
    }
}
