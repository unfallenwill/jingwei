//! System-side reminders: extra instructions injected into the model
//! request, distinct from the static `crate::SYSTEM` constant. Every
//! non-static piece of system-side text rides here — AGENTS.md content
//! (commit 1), runtime context, future runtime injections — and reaches
//! the model as a system-role message.
//!
//! ## Why a separate type from `system_text`
//!
//! `Context.system_text: String` holds *only* `crate::SYSTEM` — the
//! stable identity contract the model reads once per session. Every
//! other piece of system-side text rides in `Context.reminders`, where
//! every [`SystemReminder`] carries a stable [`id`] used for dedupe and
//! cache-breakpoint reuse.
//!
//! The vendor renders `reminders` as extra system blocks (Anthropic's
//! Messages wire) or appended to the single system message (chat-
//! completions); both are still system role, just shaped differently per
//! protocol. Either way the model sees the same system-side text.
//!
//! ## Lifetime
//!
//! A reminder, once enqueued, is a message — it stays for the rest of
//! the session, or until something replaces it by id. We do not model
//! "ephemeral" at the type level: if a reminder is too noisy, the
//! caller (the hook that enqueues it) dedupes by id. The model just
//! reads messages; it doesn't know which ones were "supposed to be
//! temporary".
//!
//! ## Cache behavior
//!
//! [`ReminderId::is_session_stable`] is the only place a reminder's
//! cache semantics are decided. `true` ⇒ the text is byte-stable across
//! requests within the session, vendors attach a cache marker to it.
//! `false` ⇒ the text may change per request, vendors skip the marker.
//! Adding a new variant requires a one-line answer here; no other
//! code needs to change.

use std::path::PathBuf;

/// One injected system-side instruction.
#[derive(Debug, Clone)]
pub(crate) struct SystemReminder {
    /// Stable identity. Two reminders with the same id are the same
    /// reminder: dedupe keeps one, replace-by-id keeps the seat but
    /// changes the body. Different ids ⇒ different seats.
    pub id: ReminderId,
    /// Rendered body. The reminder owns its own leading separator
    /// (`"\n\n"` when meant to follow another system-side block) — the
    /// composer for chat-completions vendors concatenates with no
    /// extra separator, keeping the wire shape byte-stable against the
    /// old `compose_system` output.
    pub text: String,
}


/// The kind of reminder. Variants correspond to "what kind of extra
/// instruction this is". Adding a new kind is an enum-variant addition
/// — the primitive stays the same, the vendor render loop stays the
/// same, and `is_session_stable()` is the only place a new case must
/// be answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReminderId {
    /// The body of the closest AGENTS.md, eager at session start.
    /// `path` is the file the body came from — kept in the id so two
    /// sessions loading different AGENTS.md files don't collide in
    /// cache, and so the model can tell *where* the body came from.
    AgentsMdClosest { path: PathBuf },

    /// Path-only index of AGENTS.md files below the workspace root
    /// but on the path from repo root to CWD (excluding the closest
    /// one, which is already represented as `AgentsMdClosest`).
    /// Eager at session start. The model reads any of these on
    /// demand via `read_file` — we list paths, not bodies, so the
    /// cache hit rate of the closest AGENTS.md is unaffected by
    /// how many nested files exist.
    AgentsMdIndex { paths: Vec<PathBuf> },
}

impl ReminderId {
    /// True when this reminder's text is byte-stable across requests
    /// within the same session — vendors attach a cache marker to it
    /// so the model sees a stable cache segment. False when the text
    /// may change between requests or is per-turn; vendors skip the
    /// marker.
    ///
    /// Changing the answer here is a cache-behavior change for that
    /// reminder — make it deliberate, not a default match arm.
    pub fn is_session_stable(&self) -> bool {
        match self {
            // AGENTS.md is loaded once at session start and not edited
            // mid-session. The body, the path, the surrounding header
            // are all byte-stable.
            Self::AgentsMdClosest { .. } => true,
            // The path index is also loaded once at session start and
            // doesn't change between requests. Bodies of the listed
            // files are *not* inlined — the model reads them via
            // `read_file` when it wants them, which keeps the cached
            // prefix independent of how many nested files exist.
            Self::AgentsMdIndex { .. } => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agents_md_variants_are_session_stable() {
        // Both AGENTS.md variants are loaded once at session start
        // and do not change between requests — they ride in the
        // cached prefix.
        assert!(ReminderId::AgentsMdClosest { path: PathBuf::from("/r/AGENTS.md") }
            .is_session_stable());
        assert!(ReminderId::AgentsMdIndex { paths: vec![PathBuf::from("/r/x/AGENTS.md")] }
            .is_session_stable());
    }

    #[test]
    fn reminder_carries_id_and_text_independently() {
        // the struct is data — id and text are independently addressable
        let r = SystemReminder {
            id: ReminderId::AgentsMdClosest { path: PathBuf::from("/r/AGENTS.md") },
            text: "use pnpm".into(),
        };
        match &r.id {
            ReminderId::AgentsMdClosest { path } => {
                assert_eq!(path, &PathBuf::from("/r/AGENTS.md"));
            }
            other => panic!("expected AgentsMdClosest, got {other:?}"),
        }
        assert_eq!(r.text, "use pnpm");
    }

    #[test]
    fn reminder_ids_eq_on_full_content() {
        // Two reminders with the same full id value are equal — this
        // is what makes dedupe by id work without needing to look at
        // the text body. Path and Vec contents both participate in
        // the comparison.
        let a = ReminderId::AgentsMdClosest { path: PathBuf::from("/r/AGENTS.md") };
        let b = ReminderId::AgentsMdClosest { path: PathBuf::from("/r/AGENTS.md") };
        let c = ReminderId::AgentsMdClosest { path: PathBuf::from("/other/AGENTS.md") };
        assert_eq!(a, b);
        assert_ne!(a, c);

        let p = ReminderId::AgentsMdIndex { paths: vec![PathBuf::from("/r/x")] };
        let q = ReminderId::AgentsMdIndex { paths: vec![PathBuf::from("/r/x")] };
        let r = ReminderId::AgentsMdIndex { paths: vec![PathBuf::from("/r/y")] };
        assert_eq!(p, q);
        assert_ne!(p, r);
        // mixed kinds are different even if one is empty
        assert_ne!(p, ReminderId::AgentsMdIndex { paths: vec![] });
    }
}
