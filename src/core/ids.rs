//! Who is running: the id the agent-side tables key on.
//!
//! The id lives here, at L0, and not up in the agent module, because three
//! modules *below* the agent layer have to be able to name it: the cancel
//! scope (one leaf per agent), the ledger (keyed by agent *and* path), and
//! the tool port. An id only the top layer can spell is an id the lower
//! layers cannot use — and using it is the whole point of having one.
//!
//! It is a plain newtype: comparable, hashable, printable. Nothing here
//! does I/O, and nothing here knows what an agent *is* — only how one is
//! told apart from another.

use std::sync::atomic::{AtomicU64, Ordering};

/// Who is running. A plain newtype: comparable, hashable, printable.
///
/// The number underneath is the identity; the [`Display`](std::fmt::Display)
/// form is only the tag a human reads.
///
/// The type is `pub` so the display port — which is public — can name it
/// in its contract. The constructors stay inside the crate: only this
/// process mints ids.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AgentId(u64);

impl AgentId {
    /// The main agent — the conversation the human is talking to.
    ///
    /// Minted literally, never by [`AgentId::next`], so it is always 0.
    pub(crate) const MAIN: AgentId = AgentId(0);

    /// A fresh id, distinct from every other id this process has handed out.
    #[allow(dead_code)] // the fleet slice mints worker ids; until then only the tests do
    pub(crate) fn next() -> AgentId {
        AgentId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Read an id back from the tag a human sees.
    ///
    /// The fleet's tools speak ids in the display form — the model writes
    /// `a1`, the human types `a1` — so something has to turn that string
    /// back into the identity underneath it. Only the worker form parses:
    /// `main` is the conversation, not a background agent, and `a0` is the
    /// one number the counter never hands out.
    #[allow(dead_code)] // the fleet slice hands ids back; until then only the tests do
    pub(crate) fn parse(s: &str) -> Option<AgentId> {
        let n: u64 = s.strip_prefix('a')?.parse().ok()?;
        if n == 0 { return None; }
        Some(AgentId(n))
    }
}

/// The process-wide counter of minted ids.
///
/// It starts at **1**, not 0: `MAIN` is the literal 0, so a counter that
/// started at 0 would hand `MAIN` out a second time and two agents would
/// share an identity. The wrap at 2^64 is theoretical — a process that
/// minted 18 quintillion agents would have other problems first.
static NEXT: AtomicU64 = AtomicU64::new(1);

/// The tag a human reads: `main` for the main agent, `a1`, `a2`, … for
/// the rest. The number is the id's *display*, not its identity — the
/// identity is the `u64` underneath.
impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == AgentId::MAIN {
            write!(f, "main")
        } else {
            write!(f, "a{}", self.0)
        }
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_is_zero_and_reads_as_main() {
        assert_eq!(AgentId::MAIN.0, 0, "MAIN is minted literally, so it is 0");
        assert_eq!(AgentId::MAIN.to_string(), "main");
    }

    #[test]
    fn a_fresh_id_is_never_main_and_never_repeats() {
        let a = AgentId::next();
        let b = AgentId::next();
        assert_ne!(a, b, "two fresh ids must not be the same");
        assert_ne!(a, AgentId::MAIN, "the counter starts at 1, so MAIN is never handed out again");
        assert_ne!(b, AgentId::MAIN);
        assert!(a.0 >= 1 && b.0 >= 1, "the counter starts at 1 — 0 belongs to MAIN");
    }

    #[test]
    fn a_worker_id_reads_as_a1() {
        assert_eq!(AgentId(1).to_string(), "a1");
        assert_eq!(AgentId(42).to_string(), "a42");
    }

    #[test]
    fn an_id_parses_from_its_display_form() {
        assert_eq!(AgentId::parse("a1"), Some(AgentId(1)));
        assert_eq!(AgentId::parse("a42"), Some(AgentId(42)));
        // a round trip: what a human reads is what a model can hand back
        let id = AgentId::next();
        assert_eq!(AgentId::parse(&id.to_string()), Some(id));
        // the main agent is the conversation, not a background worker
        assert_eq!(AgentId::parse("main"), None);
        // a0 is the number the counter never hands out
        assert_eq!(AgentId::parse("a0"), None);
        // and a name is not an id
        assert_eq!(AgentId::parse("build"), None);
        assert_eq!(AgentId::parse(""), None);
        assert_eq!(AgentId::parse("a"), None);
    }
}
