//! The agent's essence, kept apart from everything that serves it.
//!
//! A coding agent is, first, a conversation: a canonical history
//! ([`ir`]) that persists across runs ([`session`]) and is driven one
//! run at a time ([`turn`]). Each request renders that history into
//! what the model actually sees ([`context`]) — the static system text
//! ([`prompt`]), the tool list, in-band [`reminder`]s, and the project
//! conventions discovered by [`agents_md`]. And the loop that carries it
//! all — model call, tool run, model call — lives above this cluster, in
//! `crate::agent`.
//!
//! The rest of the crate serves this core from the outside: `api`
//! speaks the wire, `tools` and `mcp` act on the world, `ui` shows it
//! happening. Nothing in `core` imports those; the loop takes them as
//! parameters.

pub(crate) mod agents_md;
pub(crate) mod context;
pub(crate) mod error;
pub(crate) mod ids;
pub(crate) mod ir;
pub(crate) mod ports;
pub(crate) mod prompt;
pub(crate) mod reminder;
pub(crate) mod session;
pub(crate) mod turn;
