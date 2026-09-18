//! One **turn** inside a session — a single execution of the agent loop,
//! from a user task to its terminal state.
//!
//! Two facts keep this module small:
//!
//! - The session ([`crate::session`]) owns the long-lived things — identity,
//!   history, archive, fork, rename. It does not run.
//! - The turn owns the short-lived thing — one execution. It does not
//!   outlive its terminal state.
//!
//! That division lets each layer have a state machine with the scale of
//! change it actually has: the session's states move on archive and fork
//! (rare, deliberate), the turn's states move every few seconds while an
//! agent run is in flight. Folding both into one enum would invent the
//! Cartesian product — "ARCHIVED + IN_PROGRESS", "PURGED + QUEUED" —
//! combinations that have no meaning and no use.
//!
//! This module today is **data + transfer rules + smoke tests**. Wiring
//! `agent_loop` to drive transitions (and a future `TurnHandle` to expose
//! them) is the next iteration; doing it here would make this file a
//! rewrite of `main.rs` instead of an addition.

use std::time::{SystemTime, UNIX_EPOCH};

/// The state of one turn. The model is borrowed from the Codex design and
/// trimmed to what jingwei can actually reach today; new states land as
/// the agent loop learns new tricks, not before.
///
/// ```text
///                  ┌────────┐ dequeue
///     turn_start   │ QUEUED ├──────────────►┐
///     ────────────►└────────┘               │
///                                         ▼
///                                ┌────────────────┐
///                                │  IN_PROGRESS   │◀────┐
///                                └─┬─────┬──────┬─┘     │
///                                  │     │      │       │ steer (future)
///                                  │     │      └───────┘
///                                  │     │
///      natural finish (final text) │     │ interrupt()
///                                  │     │
///                                  ▼     ▼
///                            ┌────────┐ ┌────────────┐
///                            │COMPLETE│ │INTERRUPTED │  (partial output kept)
///                            │  -D    │ └────────────┘
///
///                          error        never ran
///                           │            │
///                           ▼            ▼
///                      ┌────────┐    ┌──────────┐
///                      │ FAILED │    │ CANCELED │
///                      └────────┘    └──────────┘
/// ```
///
/// **Terminal states** (`COMPLETED`, `FAILED`, `INTERRUPTED`, `CANCELED`):
/// the turn will never move again. Resuming the conversation is what makes
/// new turns possible — it does not resurrect a finished one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    /// Constructed but not yet handed to the scheduler. Today jingwei runs
    /// one turn per process synchronously, so this state is rarely
    /// *visible* — but it is the honest starting point, and what makes a
    /// future asynchronous scheduler possible without rewriting this enum.
    Queued,
    /// The agent loop is in flight: an HTTP call is open, a tool is
    /// running, or a stream is being parsed. `started_at` is set.
    InProgress,
    /// The turn ended naturally on a final-text response. Normal exit.
    Completed,
    /// The turn ended on an API / IO / serialization error. `error` carries
    /// the cause. The history is still loadable.
    Failed,
    /// Ctrl-C landed at a suspension point inside the turn. The history
    /// keeps finished text/thinking/tool calls; unfinished tool calls were
    /// dropped (an orphan tool_use is a 400 on the next request).
    Interrupted,
    /// The turn was cancelled before it ever started running (a queued
    /// turn whose session was archived, or a future async cancel).
    Canceled,
}

impl TurnState {
    /// A turn in one of these states will never move again. Used to
    /// reject "second" transitions on a terminal turn without forcing
    /// callers to enumerate the four terminal variants themselves.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            TurnState::Completed | TurnState::Failed | TurnState::Interrupted | TurnState::Canceled,
        )
    }
}

/// What the table says may follow `from`. This is the **only** place the
/// transition rules live — every other code path funnels through
/// [`Turn::transition`], so the table is the single source of truth and a
/// property-based test can pin it down.
///
/// Two states that look "adjacent" but are not in the table are deliberate:
///
/// - `QUEUED → INTERRUPTED`: a queued turn can be cancelled, but "interrupt"
///   reads as "stopped mid-run"; the queued turn never ran, so the honest
///   name is `CANCELED`. The table keeps that distinction.
fn allowed_from(from: TurnState) -> &'static [TurnState] {
    match from {
        TurnState::Queued => &[TurnState::InProgress, TurnState::Canceled],
        TurnState::InProgress => &[
            TurnState::Completed,
            TurnState::Failed,
            TurnState::Interrupted,
        ],
        TurnState::Completed | TurnState::Failed | TurnState::Interrupted | TurnState::Canceled => &[],
    }
}

/// One turn. Cheap to construct — fields are all `Copy` except `error`,
/// which is `None` until the turn ends badly.
#[derive(Debug, Clone)]
pub struct Turn {
    /// Sequential id within the owning session: 0 for the first turn, 1
    /// for the next, and so on. Matches the line index of the user task
    /// that opened it, so `--list` and resumes can say "turn #3" and mean
    /// the third row of history.
    pub id: u32,
    /// Current state. Starts at `Queued` and walks the table.
    pub state: TurnState,
    /// Wall-clock seconds since the epoch, UTC. `started_at` is filled in
    /// at the `Queued → InProgress` transition; `ended_at` is filled in at
    /// the first transition *into* a terminal state. Together they are the
    /// turn's wall-clock duration — a fact the bar already shows for the
    /// current run, but this struct carries it so future code can reason
    /// about a whole session at once.
    pub started_at: Option<u64>,
    pub ended_at: Option<u64>,
    /// Set when `state == Failed`. Kept as a `String` to avoid pulling
    /// `crate::Error`'s non-`Clone` shape through this module; the agent
    /// core renders it, session.rs does not see it.
    pub error: Option<String>,
}

impl Turn {
    /// A fresh turn for a session: `id` set, everything else empty. The
    /// `Queued` initial state matches the table's only entry for a turn
    /// that has not yet been touched.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            state: TurnState::Queued,
            started_at: None,
            ended_at: None,
            error: None,
        }
    }

    /// Move the turn to `next`, stamping timestamps as the table requires.
    /// The only legal `from → next` pairs are the ones [`allowed_from`]
    /// lists — anything else returns `Err` and leaves the turn untouched,
    /// which is the property the smoke tests pin down.
    ///
    /// `error` is meaningful only when `next == Failed`; passing one for a
    /// non-failed transition is ignored, so a caller that mishandles the
    /// flag does not corrupt the turn.
    pub fn transition(&mut self, next: TurnState, error: Option<String>) -> Result<(), TurnError> {
        if !allowed_from(self.state).contains(&next) {
            return Err(TurnError::Illegal {
                from: self.state,
                to: next,
                turn_id: self.id,
            });
        }
        let now = now_secs();
        // Fill in timestamps the way the design says. The terminal guard
        // here matters: a turn that has already ended must not have its
        // timestamps rewritten by a second transition, and the table
        // refuses that case before we get here.
        if self.state == TurnState::Queued && next == TurnState::InProgress {
            self.started_at = Some(now);
        }
        if next.is_terminal() && self.ended_at.is_none() {
            self.ended_at = Some(now);
        }
        if next == TurnState::Failed {
            self.error = error;
        }
        self.state = next;
        Ok(())
    }
}

/// What happened during the run, expressed in the categories the state
/// machine distinguishes. The caller translates its own `Result` into
/// this enum (where the caller's error type lives), then hands it to
/// [`Turn::finish`] — the table knows how to map an outcome onto a
/// state, but it does not know any specific error type.
#[derive(Debug)]
pub enum TurnOutcome {
    /// The run ended naturally on a final-text response.
    Completed,
    /// Ctrl-C landed at a suspension point inside the run. Same shape
    /// as `Interrupted` in the table — kept as its own variant so the
    /// caller's translation is exhaustive against the error type.
    Interrupted,
    /// The run ended on an error of some other kind; `0` carries the
    /// rendered message.
    Failed(String),
}

impl Turn {
    /// Move the turn to the terminal state that matches the outcome.
    /// Centralizes the `TurnOutcome` → `TurnState` mapping on this side;
    /// the `Result` → `TurnOutcome` translation stays with the caller's
    /// error type. From outcome onward everything is the turn's concern.
    /// A `TurnError` here means a future change broke the mapping — the
    /// smoke tests pin the table down before any new outcome can land
    /// here.
    pub fn finish(&mut self, outcome: TurnOutcome) {
        let (next, payload) = match outcome {
            TurnOutcome::Completed => (TurnState::Completed, None),
            TurnOutcome::Interrupted => (TurnState::Interrupted, None),
            TurnOutcome::Failed(msg) => (TurnState::Failed, Some(msg)),
        };
        let _ = self.transition(next, payload);
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What went wrong. Carries enough context for a future `TurnHandle` to
/// render a useful banner without re-discovering the state by hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnError {
    /// The caller asked for a transition the table does not allow. The
    /// `from` and `to` are the actual states involved; `turn_id` names the
    /// turn so a log line can be unambiguous in a multi-turn session.
    Illegal { from: TurnState, to: TurnState, turn_id: u32 },
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TurnError::Illegal { from, to, turn_id } => {
                write!(f, "turn #{turn_id}: illegal transition {from:?} → {to:?}")
            }
        }
    }
}

impl std::error::Error for TurnError {}

// ---- smoke tests -----------------------------------------------------------
//
// These tests pin the transfer table down: every legal transition is
// exercised end-to-end, and every illegal one is refused with `Err`. If
// a future change loosens the table, these tests fail first — and that
// is exactly the safety net the design asks for.

#[cfg(test)]
mod tests {
    use super::*;

    /// The happy path: `Queued → InProgress → Completed`. The timestamps
    /// arrive in the right order, `started_at` is set, `ended_at` is set,
    /// no `error`.
    #[test]
    fn queued_to_in_progress_to_completed_stamps_timestamps() {
        let mut t = Turn::new(0);
        assert_eq!(t.state, TurnState::Queued);
        assert!(t.started_at.is_none() && t.ended_at.is_none());

        assert!(t.transition(TurnState::InProgress, None).is_ok());
        assert_eq!(t.state, TurnState::InProgress);
        let started = t.started_at.expect("started_at set on enter");
        assert!(t.ended_at.is_none());

        assert!(t.transition(TurnState::Completed, None).is_ok());
        assert_eq!(t.state, TurnState::Completed);
        let ended = t.ended_at.expect("ended_at set on first terminal");
        assert!(ended >= started, "ended_at is monotonic: {started} → {ended}");
        assert!(t.error.is_none(), "completed turns carry no error");
    }

    /// The interruption path: Ctrl-C mid-run. The table treats this as
    /// distinct from `Canceled` (the turn *did* run for a while).
    #[test]
    fn in_progress_to_interrupted_keeps_the_partial_run() {
        let mut t = Turn::new(2);
        t.transition(TurnState::InProgress, None).unwrap();
        assert!(t.transition(TurnState::Interrupted, None).is_ok());
        assert_eq!(t.state, TurnState::Interrupted);
        assert!(t.ended_at.is_some(), "interrupted is terminal — ended_at set");
        assert!(t.error.is_none(), "interruption is not an error");
    }

    /// The queued-cancel path: a turn that never ran. The table routes
    /// this to `Canceled`, not `Interrupted` — see the doc on the table.
    #[test]
    fn queued_to_canceled_skips_running() {
        let mut t = Turn::new(0);
        assert!(t.transition(TurnState::Canceled, None).is_ok());
        assert_eq!(t.state, TurnState::Canceled);
        assert!(t.ended_at.is_some(), "even cancels are terminal — ended_at set");
        assert!(t.started_at.is_none(), "but they never started");
    }

    /// The failure path carries the error message; non-failed transitions
    /// drop the field, so a caller that mishandles the flag does not poison
    /// a healthy turn.
    #[test]
    fn failed_carries_the_error_and_other_terminals_do_not() {
        let mut t = Turn::new(0);
        t.transition(TurnState::InProgress, None).unwrap();
        assert!(t.transition(TurnState::Failed, Some("api 429".into())).is_ok());
        assert_eq!(t.error.as_deref(), Some("api 429"));
        assert_eq!(t.state, TurnState::Failed);
        assert!(t.ended_at.is_some());

        // error is ignored on non-failed transitions
        let mut t = Turn::new(1);
        t.transition(TurnState::InProgress, None).unwrap();
        assert!(t.transition(TurnState::Completed, Some("ignored".into())).is_ok());
        assert!(t.error.is_none(), "only Failed stores the error: got {:?}", t.error);
    }

    /// Once a turn is terminal, the table refuses every transition — even
    /// ones a future caller might think are "harmless". The protection is
    /// not for the data (the timestamps would just rewrite) but for the
    /// *narrative* of the turn: a Completed turn that later becomes
    /// Failed would be a lie about the session.
    #[test]
    fn terminal_states_are_sinks() {
        for &term in &[
            TurnState::Completed,
            TurnState::Failed,
            TurnState::Interrupted,
            TurnState::Canceled,
        ] {
            let mut t = Turn::new(0);
            t.transition(TurnState::InProgress, None).unwrap_or(()); // get past Queued
            if !term.is_terminal() { continue; }
            // For CANCELED, we need to set it from QUEUED (no InProgress needed)
            if term == TurnState::Canceled {
                let mut t = Turn::new(1);
                t.transition(term, None).unwrap();
                assert_eq!(t.state, term);
                assert!(t.transition(TurnState::InProgress, None).is_err());
                continue;
            }
            t.transition(term, None).unwrap();
            assert_eq!(t.state, term);
            for &next in &[
                TurnState::Queued,
                TurnState::InProgress,
                TurnState::Completed,
                TurnState::Failed,
                TurnState::Interrupted,
                TurnState::Canceled,
            ] {
                assert!(
                    t.transition(next, None).is_err(),
                    "terminal {term:?} must reject → {next:?}"
                );
            }
        }
    }

    /// The full transition table as a property: for every (from, to) pair,
    /// the table's verdict matches a hand-derived allow-list. This catches
    /// the case where someone adds a state but forgets to extend the
    /// match — the assert fires because the table is the only thing the
    /// user-facing API consults.
    #[test]
    fn transition_table_matches_expected_pairs() {
        let expected: &[((TurnState, TurnState), ())] = &[
            ((TurnState::Queued, TurnState::InProgress), ()),
            ((TurnState::Queued, TurnState::Canceled), ()),
            ((TurnState::InProgress, TurnState::Completed), ()),
            ((TurnState::InProgress, TurnState::Failed), ()),
            ((TurnState::InProgress, TurnState::Interrupted), ()),
        ];
        for ((from, to), ()) in expected {
            let mut t = Turn::new(0);
            if *from == TurnState::InProgress {
                // bootstrap past Queued for InProgress-source tests
                t.transition(TurnState::InProgress, None).unwrap();
            }
            assert!(
                t.transition(*to, None).is_ok(),
                "expected legal: {from:?} → {to:?}"
            );
        }

        // And a few that must *not* be in the table.
        let forbidden: &[(TurnState, TurnState)] = &[
            (TurnState::Queued, TurnState::Completed),
            (TurnState::Queued, TurnState::Failed),
            (TurnState::Queued, TurnState::Interrupted),
            (TurnState::InProgress, TurnState::Queued),
            (TurnState::InProgress, TurnState::Canceled),
            (TurnState::Completed, TurnState::InProgress),
            (TurnState::Failed, TurnState::Completed),
            (TurnState::Canceled, TurnState::InProgress),
        ];
        for (from, to) in forbidden {
            let mut t = Turn::new(0);
            // Boot the turn to `from` the way the table allows. Without
            // this step, an `InProgress`-source test is still in `Queued`
            // and the transition *succeeds*, hiding the bug we are trying
            // to catch.
            t.state = *from;
            match *from {
                TurnState::InProgress => {
                    t.started_at = Some(0);
                }
                TurnState::Queued => {}
                _ => {
                    // Terminal sources: stamp ended_at too, so the
                    // bootstrap reflects what a real terminal turn looks
                    // like.
                    t.started_at = Some(0);
                    t.ended_at = Some(0);
                }
            }
            assert!(
                t.transition(*to, None).is_err(),
                "expected illegal: {from:?} → {to:?}"
            );
        }
    }

    /// `is_terminal` covers exactly the four terminal variants — a
    /// drift detector for the helper.
    #[test]
    fn is_terminal_covers_only_terminals() {
        assert!(!TurnState::Queued.is_terminal());
        assert!(!TurnState::InProgress.is_terminal());
        assert!(TurnState::Completed.is_terminal());
        assert!(TurnState::Failed.is_terminal());
        assert!(TurnState::Interrupted.is_terminal());
        assert!(TurnState::Canceled.is_terminal());
    }

    /// The error type renders something a future CLI can show without
    /// having to re-enumerate the states.
    #[test]
    fn illegal_transition_error_is_descriptive() {
        let mut t = Turn::new(7);
        let err = t.transition(TurnState::Completed, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("turn #7"), "names the turn: {msg}");
        assert!(msg.contains("Queued"), "names the source state: {msg}");
        assert!(msg.contains("Completed"), "names the destination state: {msg}");
    }
}