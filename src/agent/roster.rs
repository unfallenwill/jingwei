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

    /// The live *background* agents, in the order the ids were minted.
    ///
    /// The main agent is not one of them: the conversation the human is
    /// talking to is not a background agent, and the fleet's limits and
    /// its listing both talk about workers.
    pub(crate) fn list(&self) -> Vec<AgentId> {
        self.ids().into_iter().filter(|id| *id != AgentId::MAIN).collect()
    }

    /// Stop one agent, and say whether it was live.
    ///
    /// The scope the roster holds is cancelled, so the agent's own leaf
    /// is born cancelled and the run under it unwinds. The main agent is
    /// never stopped this way — the human ends the conversation, not a
    /// stop request.
    pub(crate) fn stop(&self, id: AgentId) -> bool {
        if id == AgentId::MAIN { return false; }
        let scope = {
            let live = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            live.get(&id).cloned()
        };
        match scope {
            Some(scope) => { scope.cancel(); true }
            None => false,
        }
    }

    /// Stop every background agent, and leave the main one alone.
    pub(crate) fn stop_all(&self) {
        for (id, scope) in self.live() {
            if id != AgentId::MAIN { scope.cancel(); }
        }
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
        assert_eq!(roster.list(), vec![id], "the agent is live");
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
        assert_eq!(roster.list().len(), 1);
        roster.remove(id);
        assert!(roster.list().is_empty(), "the agent is gone");
        // removing an agent that is not there is a no-op, not a panic
        roster.remove(id);
        roster.remove(AgentId::next());
        assert!(roster.list().is_empty());
    }

    #[test]
    fn two_agents_are_two_roster_entries() {
        let roster = Roster::new();
        let a = AgentId::next();
        let b = AgentId::next();
        roster.register(a, CancelScope::new());
        roster.register(b, CancelScope::new());
        let ids = roster.list();
        assert_eq!(ids.len(), 2, "ids do not collide: {ids:?}");
        assert!(ids.contains(&a) && ids.contains(&b), "both are there: {ids:?}");
        // and one agent leaving does not take the other with it
        roster.remove(a);
        assert_eq!(roster.list(), vec![b]);
    }

    #[test]
    fn the_roster_lists_the_live_agents() {
        let roster = Roster::new();
        let a = AgentId::next();
        let b = AgentId::next();
        // the main agent registers itself for the length of its run, and
        // it is the conversation - not a background agent
        roster.register(AgentId::MAIN, CancelScope::new());
        roster.register(a, CancelScope::new());
        roster.register(b, CancelScope::new());
        assert_eq!(roster.list(), vec![a, b], "the main agent is not on the list");
    }

    #[test]
    fn stopping_one_agent_leaves_the_others() {
        let roster = Roster::new();
        let a = AgentId::next();
        let b = AgentId::next();
        let scope_a = CancelScope::new();
        roster.register(a, scope_a.clone());
        roster.register(b, CancelScope::new());
        assert!(roster.stop(a), "a live agent can be stopped");
        assert!(scope_a.is_cancelled(), "and the scope that stops it is cancelled");
        assert!(!roster.scope_of(b).expect("b is live").is_cancelled(), "b is untouched");
        // an agent that is not there cannot be stopped
        assert!(!roster.stop(AgentId::next()), "an id the roster never held cannot be stopped");
        // and the main agent is the conversation's, not the fleet's
        roster.register(AgentId::MAIN, CancelScope::new());
        assert!(!roster.stop(AgentId::MAIN), "the main agent is never stopped this way");
    }

    #[test]
    fn stop_all_leaves_the_main_alone() {
        let roster = Roster::new();
        let main_scope = CancelScope::new();
        let worker_scope = CancelScope::new();
        roster.register(AgentId::MAIN, main_scope.clone());
        roster.register(AgentId::next(), worker_scope.clone());
        roster.stop_all();
        assert!(worker_scope.is_cancelled(), "the background agent is stopped");
        assert!(!main_scope.is_cancelled(), "the conversation is not the fleet's to stop");
    }
}
