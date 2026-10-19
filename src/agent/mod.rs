//! The agent, as a value.
//!
//! Until now "the agent" was an adjective: the thing that was currently
//! talking to the model, named only by the call that happened to be
//! running. This module gives it a name, an id, a role, a task, and a
//! lifecycle — a plain value, with no I/O, no async, and no handles — so
//! that the UI can clone and render it, the ledger and the cancel scope
//! can key on it, and two callers can finally be told apart.
//!
//! The lifecycle is **not** duplicated here. The agent *owns* a
//! [`Turn`] — the state machine in [`crate::core::turn`] — and reads it
//! through it. A second copy of "is this agent still running" would be
//! exactly the kind of two-sources-of-truth this refactor exists to
//! remove.
//!
//! The fleet lives in [`crate::workflow::exec`]; this module holds the
//! pieces it is made of: the [`roster`] of live agents, the [`runner`]
//! that drives one agent, the [`worker`] that runs one in the
//! background, the [`tools`] the main agent uses to manage them, and
//! the [`cmd`] the human uses.
//!
//! The id lives in [`crate::core::ids`] (L0), not here, because the
//! modules *below* this one — the cancel scope, the ledger, the tool port
//! — have to be able to name it. It is re-exported so the agent module and
//! the rest of the crate can say `agent::AgentId` where that reads better.

use crate::core::turn::{Turn, TurnState};
use crate::ui::Usage;

pub(crate) mod cmd;
pub(crate) mod roster;
pub(crate) mod runner;
pub(crate) mod tools;
pub(crate) mod worker;

pub(crate) use crate::core::ids::AgentId;

/// What an agent is for. The main agent talks to the human; a worker is
/// one the fleet spawns to work a task of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AgentRole {
    /// The conversation the human is talking to.
    Main,
    /// An agent the fleet spawned.
    Worker,
}

/// One agent: who it is, what it was asked, how far it got, and what it
/// cost.
///
/// The fields the frontends and the tables key on are `pub(crate)` so the
/// modules that need them can read them; the lifecycle stays behind
/// [`Agent::state`] and friends, so there is exactly one copy of it.
#[derive(Clone)]
pub(crate) struct Agent {
    /// Who this is. The ledger keys on it; the cancel scope holds one
    /// token per id; the UI shows it.
    pub(crate) id: AgentId,
    /// A short, human-facing name ("main", or a worker's label).
    pub(crate) name: String,
    /// Main or Worker.
    ///
    /// The depth-one rule is checked on the *id*, in the dispatch, where
    /// the call arrives — so nothing reads the role yet. It is here
    /// because the value says what the agent *is*, and the fleet's
    /// record is a projection of it, not the other way round.
    #[allow(dead_code)]
    pub(crate) role: AgentRole,
    /// The task this agent was given. The fleet's table does not show
    /// it; the value carries it so a frontend that wants it can read it
    /// without asking the fleet to remember a second copy.
    #[allow(dead_code)]
    pub(crate) task: String,
    /// The lifecycle state, owned by `core::turn`.
    pub(crate) lifecycle: Turn,
    /// What the agent produced, once it is done.
    pub(crate) report: Option<String>,
    /// The tools it called, in order.
    pub(crate) tools_used: Vec<String>,
    /// What it spent.
    #[allow(dead_code)] // per-worker cost accounting is a named follow-up
    pub(crate) usage: Usage,
}

impl Agent {
    /// A fresh agent: an identity, a name, a role, a task, and a lifecycle
    /// that has not started yet. The main agent is the one the human talks
    /// to; a worker is the one the fleet spawns.
    pub(crate) fn new(
        id: AgentId,
        name: String,
        role: AgentRole,
        task: String,
        lifecycle: Turn,
    ) -> Self {
        Self {
            id,
            name,
            role,
            task,
            lifecycle,
            report: None,
            tools_used: Vec::new(),
            usage: Usage::default(),
        }
    }

    /// The main agent: the conversation the human is talking to. Its id is
    /// [`AgentId::MAIN`], its name is "main", and it starts queued.
    pub(crate) fn main(task: &str) -> Self {
        Self::new(
            AgentId::MAIN,
            "main".to_string(),
            AgentRole::Main,
            task.to_string(),
            Turn::new(0),
        )
    }

    /// Where the agent is in its lifecycle.
    pub(crate) fn state(&self) -> TurnState {
        self.lifecycle.state
    }

    /// When the agent started, if it has. Wall-clock seconds, so they
    /// say *when* and not *how long* — the elapsed time is a duration,
    /// and the fleet measures it with a clock of its own.
    #[allow(dead_code)] // the frontends keep their own clock for the elapsed time
    pub(crate) fn started_at(&self) -> Option<u64> {
        self.lifecycle.started_at
    }

    /// When the agent ended, if it has.
    #[allow(dead_code)] // the frontends keep their own clock for the elapsed time
    pub(crate) fn ended_at(&self) -> Option<u64> {
        self.lifecycle.ended_at
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_main_agent_starts_queued_with_its_task_and_no_report() {
        let agent = Agent::main("count the files");
        assert_eq!(agent.id, AgentId::MAIN);
        assert_eq!(agent.name, "main");
        assert_eq!(agent.role, AgentRole::Main);
        assert_eq!(agent.task, "count the files");
        assert_eq!(agent.state(), TurnState::Queued);
        assert!(agent.report.is_none());
        assert!(agent.tools_used.is_empty());
        assert_eq!(agent.usage, Usage::default());
        assert!(agent.started_at().is_none() && agent.ended_at().is_none());
    }

    #[test]
    fn the_agent_reads_its_lifecycle_through_the_turn_it_owns() {
        let mut agent = Agent::main("go");
        agent
            .lifecycle
            .transition(TurnState::InProgress, None)
            .unwrap();
        assert_eq!(agent.state(), TurnState::InProgress);
        assert_eq!(agent.started_at(), agent.lifecycle.started_at);
        agent.lifecycle.finish(crate::core::turn::TurnOutcome::Completed);
        assert_eq!(agent.state(), TurnState::Completed);
        assert_eq!(agent.ended_at(), agent.lifecycle.ended_at);
        assert!(agent.ended_at().is_some());
    }
}
