//! The registry of live agents.
//!
//! The roster sits at L5 because it holds an agent's cancel scope and
//! names it by its id, and because it sits beside the `Agent` value it
//! registers: a roster that held `Agent` values would reach upward into
//! the agent layer — the exact edge the architecture test exists to
//! catch.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::cancel::CancelScope;
use crate::core::ids::AgentId;

/// The registry of live agents.
///
/// It holds state and it does not act: registering an agent is not
/// starting it, and removing one is not stopping it. The executor
/// (`workflow::run_task`) does the acting; the roster is what it registers
/// with, and what a stop request reaches through.
///
/// Keyed by id in a `BTreeMap`, so the listing is in the order the ids
/// were minted — the order the agents started.
#[derive(Clone)]
pub(crate) struct Roster {
    inner: Arc<Mutex<BTreeMap<AgentId, CancelScope>>>,
}

impl Roster {
    /// A roster with nobody in it.
    pub(crate) fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(BTreeMap::new())) }
    }

    /// Register a live agent, with the scope that stops it.
    ///
    /// The scope is the agent's *own* scope, not a leaf: cancelling a
    /// scope cancels everything beneath it, so a scope is the right
    /// handle for "stop this agent" once agents can have agents.
    pub(crate) fn register(&self, id: AgentId, scope: CancelScope) {
        let mut live = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        live.insert(id, scope);
    }

    /// The agent is no longer live. Removing an agent that is not there
    /// is a no-op, not a failure.
    pub(crate) fn remove(&self, id: AgentId) {
        let mut live = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        live.remove(&id);
    }

    /// The live agents and the scopes that stop them — the roster's own
    /// view of who is running.
    fn live(&self) -> Vec<(AgentId, CancelScope)> {
        let live = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        live.iter().map(|(id, scope)| (*id, scope.clone())).collect()
    }

    /// The ids of the live agents, in the order the ids were minted.
    fn ids(&self) -> Vec<AgentId> {
        self.live().into_iter().map(|(id, _)| id).collect()
    }

    /// The ids of the live agents, for the tests that need to see the
    /// roster from the outside.
    ///
    /// Test-only on purpose: the production API is `new` / `register` /
    /// `remove`, and the fleet slice grows the readers the frontends
    /// need.
    #[cfg(test)]
    pub(crate) fn live_ids(&self) -> Vec<AgentId> {
        self.ids()
    }

    /// The scope the roster holds for one agent — the handle a stop
    /// request reaches.
    #[cfg(test)]
    pub(crate) fn scope_of(&self, id: AgentId) -> Option<CancelScope> {
        self.live().into_iter().find(|(i, _)| *i == id).map(|(_, scope)| scope)
    }
}

impl std::fmt::Debug for Roster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live: Vec<AgentId> = self.ids();
        f.debug_struct("Roster").field("live", &live).finish()
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The roster is shared, not owned: one registry answers "who is
    /// running" for the whole process, and the reader that answers it
    /// may be a different task from the one that registered.
    #[test]
    fn the_roster_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Roster>();
    }

    #[test]
    fn the_roster_holds_a_live_agent() {
        let roster = Roster::new();
        let id = AgentId::next();
        let scope = CancelScope::new();
        roster.register(id, scope.clone());
        assert_eq!(roster.live_ids(), vec![id], "the agent is live");
        // The roster holds the *scope*, not a snapshot of it: cancelling
        // the scope that was registered reaches the one the roster holds
        // — a leaf minted under the held scope is born cancelled, which
        // is exactly what a handle on the same tree buys.
        scope.cancel();
        let held = roster.scope_of(id).expect("the agent is still live");
        assert!(
            held.token().is_cancelled(),
            "the roster holds a handle on the same cancel tree"
        );
    }

    #[test]
    fn a_finished_agent_leaves_the_roster() {
        let roster = Roster::new();
        let id = AgentId::next();
        roster.register(id, CancelScope::new());
        assert_eq!(roster.live_ids().len(), 1);
        roster.remove(id);
        assert!(roster.live_ids().is_empty(), "the agent is gone");
        // removing an agent that is not there is a no-op, not a panic
        roster.remove(id);
        roster.remove(AgentId::next());
        assert!(roster.live_ids().is_empty());
    }

    #[test]
    fn two_agents_are_two_roster_entries() {
        let roster = Roster::new();
        let a = AgentId::next();
        let b = AgentId::next();
        roster.register(a, CancelScope::new());
        roster.register(b, CancelScope::new());
        let ids = roster.live_ids();
        assert_eq!(ids.len(), 2, "ids do not collide: {ids:?}");
        assert!(ids.contains(&a) && ids.contains(&b), "both are there: {ids:?}");
        // and one agent leaving does not take the other with it
        roster.remove(a);
        assert_eq!(roster.live_ids(), vec![b]);
    }
}
