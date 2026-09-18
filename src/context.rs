//! The model-side context: every turn, jingwei renders one prompt and ships
//! it down a wire. This module owns the parts of the prompt the model sees
//! but the agent core doesn't think about — system text (just `SYSTEM`),
//! the active reminder list (every non-static system-side instruction:
//! AGENTS.md today, runtime context and compaction anchors tomorrow), and
//! the trim policy that keeps the history inside the configured budget.
//!
//! Why this is its own module: before, the same work was scattered across
//! three places — `agents_md::system_prompt_extras` (now retired), the
//! per-vendor `body()` functions (each running its own composition), and
//! a clump of trim helpers in `main::run_turn` that read tool schemas and
//! called `fit_context` inline. Three call sites, three implicit
//! contracts, no name. Here the work has one home and one name
//! (`Context`); vendors consume what `refresh` has already assembled
//! and never reach into AGENTS.md or the tool registry themselves.
//!
//! ## Architectural room for v2
//!
//! - `refresh()` already accepts a `&Hub`; the next obvious additions are
//!   an AGENTS.md reload (when a session detects a write to the loaded
//!   file) and a history-summarization pass that hands context to an LLM
//!   and rebuilds history in place. Both have an obvious home here.
//! - `reminders` is currently `Vec<SystemReminder>` and adds to the wire
//!   via the vendor render loop; a future multi-message form (nested
//!   AGENTS.md index, runtime context) is just more `SystemReminder`s
//!   with different ids. The field type stays the same.
//! - The trim policy is currently `bytes/3` and "oldest tool_result
//!   first". A pluggable `TrimPolicy` is straightforward when there's
//!   more than one policy worth offering; for now one policy lives here.

use crate::agents_md::AgentsMdContext;
use crate::ir::{history_value, Block, Message};
use crate::mcp::Hub;
use crate::reminder::{ReminderId, SystemReminder};
use crate::tools::tools;
use serde_json::{json, Value};

/// The per-turn context the model sees. Vendors consume it; the agent
/// loop mutates `history` and asks `refresh()` to update the parts that
/// can change between requests (tool list — and, in v2, the reminder
/// list after a runtime injection).
#[derive(Clone)]
pub(crate) struct Context {
    /// The static system prompt — exactly `crate::SYSTEM`, nothing else.
    /// Vendors send this as the first system block on every wire. It is
    /// never augmented: AGENTS.md and runtime context ride in `reminders`,
    /// not here. That single rule is what keeps cache-breakpoint shape
    /// stable across requests within a session.
    pub(crate) system_text: String,
    /// The list of non-static system-side instructions to send on every
    /// request. Built once at session start by [`Context::new`]; the
    /// field is `pub(crate)` mut so future hooks (mid-session AGENTS.md
    /// refresh, runtime context, compaction anchors) can extend it.
    /// Each reminder carries a stable id for dedupe and an
    /// `is_session_stable()` predicate the vendor uses to decide whether
    /// to attach a cache marker.
    pub(crate) reminders: Vec<SystemReminder>,
    /// Built-in tool schemas joined with whatever MCP servers reported at
    /// the last `refresh()`. Vendors consume this list verbatim.
    pub(crate) tools: Vec<Value>,
}

impl Context {
    /// Build the per-session context. Compose the static system text
    /// from `crate::SYSTEM` alone (AGENTS.md content rides in a
    /// reminder, not in this string), enqueue AGENTS.md reminders when
    /// the loader found files, and seed the tool list with the built-in
    /// registry. MCP tools join in on the first `refresh()` — they
    /// require a runtime, which this constructor deliberately doesn't see.
    pub fn new(agents_md: &AgentsMdContext) -> Self {
        let mut reminders = Vec::new();
        if let Some(path) = agents_md.found_path.as_ref() {
            if !agents_md.content.is_empty() {
                // The reminder text owns its own leading "\n\n": it
                // stands as its own block under a multi-block render
                // (Anthropic), and concatenates verbatim when the
                // chat-completions family collapses everything into one
                // string — both end up with `SYSTEM` followed by the
                // AGENTS.md section separated by blank lines.
                let text = format!(
                    "\n\n# Project conventions (AGENTS.md at {})\n\n{}",
                    path.display(),
                    agents_md.content,
                );
                reminders.push(SystemReminder {
                    id: ReminderId::AgentsMdClosest { path: path.clone() },
                    text,
                });
            }
        }
        // Nested AGENTS.md index: every AGENTS.md under the workspace
        // (the cwd where jingwei was invoked), excluding the closest,
        // already represented above. Bodies are *not* inlined — the
        // model reads them on demand via `read_file`. This makes the
        // cache hit rate of the closest AGENTS.md independent of how
        // many nested files exist.
        if !agents_md.nested_paths.is_empty() {
            let body = agents_md
                .nested_paths
                .iter()
                .map(|p| format!("- {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n");
            let text = format!(
                "\n\n# Additional AGENTS.md files in this workspace\n{}\n\n\
                 Read them with read_file when working in those directories.",
                body,
            );
            reminders.push(SystemReminder {
                id: ReminderId::AgentsMdIndex { paths: agents_md.nested_paths.clone() },
                text,
            });
        }
        Self {
            system_text: crate::SYSTEM.to_string(),
            reminders,
            tools: builtin_tools(),
        }
    }

    /// Refresh the parts of the context that change per turn. Today only
    /// the tool list moves (MCP servers can come and go between turns);
    /// `system_text` and `reminders` are stable for the session.
    /// `refresh()` is async because `Hub::definitions()` is — vendoring
    /// it as a plain sync call would force the rest of the agent loop to
    /// drop await points that exist for cancellation, which we don't
    /// want to give back.
    pub async fn refresh(&mut self, hub: &Hub) {
        let mut schemas = builtin_tools();
        schemas.extend(hub.definitions().await);
        self.tools = schemas;
    }
}

/// Tool schemas for the built-in registry: `bash`, `read_file`, `write_file`,
/// `edit_file`. Pure over the static `tools()` table — MCP entries join in
/// via `Context::refresh`.
pub(crate) fn builtin_tools() -> Vec<Value> {
    tools().iter()
        .map(|t| json!({"name": t.name, "description": t.desc, "input_schema": t.schema}))
        .collect()
}

// ---- history trim ----------------------------------------------------------

/// Rough token estimate (bytes/3 — conservative for CJK-heavy content).
/// Counts the serialized history alone; system prompt and tool schemas
/// are part of the request yet sit outside the budget the `--context-size`
/// knob controls. Vendors stop at `cfg.context_size` history tokens; the
/// system + tools layers are assumed to fit their own budgets (they're
/// static and small in practice).
pub(crate) fn est_tokens(history: &[Message]) -> u64 {
    serde_json::to_string(&history_value(history)).map_or(0, |s| (s.len() / 3) as u64)
}

/// Shrink history until the estimate fits `limit`, trimming the oldest
/// tool_result first. Both shrinks keep tool_use/result pairing valid: an
/// in-place cut touches only the result's text, and a minimal result leaves
/// together with its paired tool_use (an orphaned tool_use is a 400 on the
/// next request), along with any message left holding no blocks.
pub(crate) fn fit_context(history: &mut Vec<Message>, limit: u64) -> bool {
    let mut changed = false;
    while est_tokens(history) > limit {
        let Some((mi, bi)) = oldest_tool_result(history) else { break };
        let Message::ToolResults(rs) = &mut history[mi] else { break };
        if rs[bi].content.chars().count() > 200 {
            let cut: String = rs[bi].content.chars().take(200).collect();
            rs[bi].content = format!("{cut}…[trimmed to fit context]");
            changed = true;
            continue;
        }
        drop_result_and_pair(history, mi, bi);
        changed = true;
    }
    changed
}

/// The first (oldest) tool_result in the history, as (message, block) indices.
fn oldest_tool_result(history: &[Message]) -> Option<(usize, usize)> {
    history.iter().position(|m| matches!(m, Message::ToolResults(rs) if !rs.is_empty())).map(|mi| (mi, 0))
}

/// Delete the tool_result at (mi, bi) and its tool_use — which sits in the
/// assistant message just before — so neither survives unpaired. A thinking
/// block that only led up to that call goes too; messages emptied of blocks
/// are removed outright (an empty content array is its own API error).
fn drop_result_and_pair(history: &mut Vec<Message>, mi: usize, bi: usize) {
    let id = match &history[mi] {
        Message::ToolResults(rs) => rs[bi].id.clone(),
        _ => return,
    };
    if let Message::ToolResults(rs) = &mut history[mi] { rs.remove(bi); }
    if mi > 0 {
        if let Message::Assistant(blocks) = &mut history[mi - 1] {
            blocks.retain(|b| match b.tool_use() { Some((bid, _, _)) => bid != id, None => true });
            // thinking whose tool_use is gone: nothing left to reason towards
            if !blocks.is_empty() && blocks.iter().all(Block::is_thinking) { blocks.clear(); }
        }
    }
    let empty = |m: &Message| matches!(m, Message::Assistant(b) if b.is_empty())
        || matches!(m, Message::ToolResults(r) if r.is_empty());
    if empty(&history[mi]) {
        history.remove(mi);
        if mi > 0 && empty(&history[mi - 1]) {
            history.remove(mi - 1);
        }
    }
}



// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents_md::AgentsMdContext;
    use crate::ir::{Block, Message, ToolResult};
    use crate::test_util::temp_dir;
    use serde_json::json;

    // JSON fixtures + serializing helpers shared by the trim tests below.
    // `hv` builds a typed history from JSON literals (the wire shape the
    // model sees); `hist` serializes a typed history back to JSON so the
    // assertions can read `j[1]["content"][0]["id"]` the way the wire does.
    fn hv(values: Vec<Value>) -> Vec<Message> {
        values.iter().map(Message::from_value).collect()
    }
    fn hist(history: &[Message]) -> Value {
        history_value(history)
    }

    // Every tool_use keeps exactly one tool_result and vice versa, and no
    // message is left with an empty content array.
    fn assert_pairing(history: &[Message]) {
        let uses: Vec<String> = history.iter().flat_map(|m| match m {
            Message::Assistant(blocks) => blocks.iter()
                .filter_map(|b| b.tool_use().map(|(id, _, _)| id.to_string())).collect::<Vec<_>>(),
            _ => vec![],
        }).collect();
        let results: Vec<String> = history.iter().flat_map(|m| match m {
            Message::ToolResults(rs) => rs.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
            _ => vec![],
        }).collect();
        for u in &uses { assert!(results.contains(u), "tool_use {u} lost its tool_result"); }
        for r in &results { assert!(uses.contains(r), "tool_result {r} lost its tool_use"); }
        for m in history {
            match m {
                Message::Assistant(b) => assert!(!b.is_empty(), "empty message left behind"),
                Message::ToolResults(r) => assert!(!r.is_empty(), "empty message left behind"),
                Message::User(_) => {}
            }
        }
    }

    // ---- Context::new + builtin_tools --------------------------------------

    #[test]
    fn context_new_reads_empty_when_no_file_loaded() {
        // a workspace with no AGENTS.md: system_text is the bare constant,
        // reminders is empty — the wire is unchanged from before reminders
        // existed, and `agents_md.found_path` is None
        let dir = temp_dir("ctx_none");
        let md = AgentsMdContext::load(&dir);
        let ctx = Context::new(&md);
        assert_eq!(ctx.system_text, crate::SYSTEM);
        assert!(md.found_path.is_none());
        assert!(ctx.reminders.is_empty(),
            "no AGENTS.md ⇒ no reminder enqueued; the wire is byte-equivalent to the old behavior");
    }

    #[test]
    fn context_new_carries_extras_into_a_reminder() {
        // a workspace with AGENTS.md: the body rides in an AgentsMdClosest
        // reminder, not in `system_text`. The split keeps the static
        // system prompt free of any project content.
        let dir = temp_dir("ctx_with_md");
        std::fs::write(dir.join("AGENTS.md"), "use rustfmt").unwrap();
        let md = AgentsMdContext::load(&dir);
        let ctx = Context::new(&md);

        // system_text is just SYSTEM — AGENTS.md does NOT leak in here
        assert!(ctx.system_text.contains(crate::SYSTEM));
        assert!(!ctx.system_text.contains("use rustfmt"),
            "AGENTS.md body must not live in system_text; it rides in reminders");

        // exactly one reminder, of the AgentsMdClosest kind, with the body
        assert_eq!(ctx.reminders.len(), 1, "one AgentsMdClosest reminder");
        let r = &ctx.reminders[0];
        match &r.id {
            ReminderId::AgentsMdClosest { path } => {
                assert_eq!(path, &dir.join("AGENTS.md"),
                    "the id carries the path so two AGENTS.md files in different sessions don't collide");
            }
            other => panic!("expected AgentsMdClosest, got {other:?}"),
        }
        assert!(r.id.is_session_stable(), "AGENTS.md content rides in the cached prefix");
        assert!(r.text.contains("# Project conventions"), "reminder is labelled");
        assert!(r.text.contains("use rustfmt"), "AGENTS.md body is in the reminder text");
        assert!(r.text.contains(&dir.display().to_string()),
            "banner path is in the reminder text so the model knows where the content came from");
        assert!(r.text.starts_with("\n\n"),
            "reminder owns its leading separator (verbatim carry-over from the old `compose_system` shape)");
    }

    #[test]
    fn context_new_skips_reminder_when_closest_content_is_empty() {
        // the loader can produce a found_path with empty content (degenerate
        // / future cases). No reminder is enqueued rather than an empty
        // block — empty reminders are noise on the wire.
        let dir = temp_dir("ctx_empty_md");
        std::fs::write(dir.join("AGENTS.md"), "").unwrap();
        let md = AgentsMdContext::load(&dir);
        let ctx = Context::new(&md);
        assert!(ctx.reminders.is_empty(),
            "an AGENTS.md with no body produces no reminder; only `system_text` is sent");
    }

    #[test]
    fn context_new_keeps_system_text_pure() {
        // The whole point of the split: `system_text` is `SYSTEM` and
        // nothing else. No concatenation, no extras, no surprise.
        let dir = temp_dir("ctx_pure");
        std::fs::write(dir.join("AGENTS.md"), "use rustfmt").unwrap();
        let md = AgentsMdContext::load(&dir);
        let ctx = Context::new(&md);
        assert_eq!(ctx.system_text, crate::SYSTEM,
            "system_text is the bare SYSTEM constant; all AGENTS.md content rides in reminders");
    }

    #[test]
    fn context_new_enqueues_nested_index_when_present() {
        // Workspace = cwd. The loader does a tree-wide BFS from cwd;
        // the closest is the workspace's ancestor (sub/AGENTS.md);
        // anything else under cwd surfaces in the AgentsMdIndex
        // reminder.
        //
        // The workspace is dir/sub/deeper. For BFS to find a third
        // AGENTS.md, we add one *under* deeper (sibling/deeper/AGENTS.md).
        let dir = temp_dir("ctx_nested");
        std::fs::write(dir.join("AGENTS.md"), "from root").unwrap();
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("AGENTS.md"), "from sub").unwrap();
        let workspace = dir.join("sub").join("deeper");
        std::fs::create_dir_all(&workspace).unwrap();
        let sibling = workspace.join("sibling");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("AGENTS.md"), "from sibling").unwrap();

        let md = AgentsMdContext::load(&workspace);
        let ctx = Context::new(&md);

        // closest walks up from deeper; finds sub/AGENTS.md first
        assert_eq!(md.found_path, Some(sub.join("AGENTS.md")));
        // BFS from cwd (deeper) walks: deeper/AGENTS.md? No. sibling/AGENTS.md? Yes.
        // sibling/AGENTS.md is not the closest, so it lands in nested_paths.
        assert_eq!(md.nested_paths, vec![sibling.join("AGENTS.md")]);

        assert_eq!(ctx.reminders.len(), 2,
            "one AgentsMdClosest + one AgentsMdIndex");
        // AgentsMdClosest first (closest is the most relevant), then the index
        match &ctx.reminders[0].id {
            ReminderId::AgentsMdClosest { path } => {
                assert_eq!(path, &sub.join("AGENTS.md"));
            }
            other => panic!("expected AgentsMdClosest first, got {other:?}"),
        }
        let idx = &ctx.reminders[1];
        match &idx.id {
            ReminderId::AgentsMdIndex { paths } => {
                assert_eq!(paths, &vec![sibling.join("AGENTS.md")]);
            }
            other => panic!("expected AgentsMdIndex, got {other:?}"),
        }
        assert!(idx.id.is_session_stable(),
            "the path list is byte-stable for the session → cache marker");
        assert!(idx.text.contains("Additional AGENTS.md files"),
            "the index is labelled so the model recognizes it as a path list, not body content");
        assert!(idx.text.contains(&sibling.join("AGENTS.md").display().to_string()),
            "the listed path shows up in the reminder text");
        assert!(idx.text.contains("read_file"),
            "the reminder hints at how to load the bodies — model uses read_file on demand");
        assert!(!idx.text.contains("from sibling"),
            "the index lists PATHS only; bodies are loaded by the model via read_file, not inlined here");
        assert!(idx.text.starts_with("\n\n"),
            "reminder owns its leading separator");
    }

    #[test]
    fn context_new_skips_index_when_no_nested_paths() {
        // Only the closest AGENTS.md exists; no other AGENTS.md in
        // the workspace tree ⇒ no AgentsMdIndex reminder. The wire is
        // the same as before nested enumeration shipped.
        let dir = temp_dir("ctx_no_nested");
        std::fs::write(dir.join("AGENTS.md"), "from root").unwrap();
        let ctx = Context::new(&AgentsMdContext::load(&dir));
        assert_eq!(ctx.reminders.len(), 1,
            "no nested paths ⇒ no index reminder, only the closest");
        assert!(matches!(ctx.reminders[0].id, ReminderId::AgentsMdClosest { .. }));
    }

    #[test]
    fn builtin_tools_lists_the_four_shipped_tools() {
        let ts = builtin_tools();
        let names: Vec<&str> = ts.iter()
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert_eq!(names, vec!["bash", "read_file", "write_file", "edit_file"]);
        for t in &ts {
            // the wire shape: { name, description, input_schema }
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            assert_eq!(t["input_schema"]["type"], "object");
            assert!(t["input_schema"]["properties"].is_object());
            assert!(t["input_schema"]["required"].is_array());
        }
    }

    // ---- est_tokens --------------------------------------------------------

    #[test]
    fn est_tokens_is_zero_when_history_cannot_be_serialized() {
        // the or-default path: history that fails to serialize still
        // returns 0, not a panic
        let h: Vec<Message> = vec![];
        assert_eq!(est_tokens(&h), 0);
    }

    #[test]
    fn est_tokens_grows_with_content_size() {
        // a longer user message bulks a larger estimate
        let s = "x".repeat(3_000);
        let h = vec![Message::User(s)];
        let small = est_tokens(&[Message::User("hi".into())]);
        let big = est_tokens(&h);
        assert!(big > small, "more bytes → more tokens: small={small}, big={big}");
    }

    // ---- trim helpers: low-level (typed Message, no JSON roundtrip) -------

    #[test]
    fn oldest_tool_result_returns_none_when_history_has_no_results() {
        let h = vec![Message::User("t".into())];
        assert!(oldest_tool_result(&h).is_none());
    }

    #[test]
    fn oldest_tool_result_skips_empty_tool_results() {
        // a ToolResults with no entries does not count
        let h = vec![
            Message::User("go".into()),
            Message::Assistant(vec![Block::Text("ok".into())]),
            Message::ToolResults(vec![]),
            Message::ToolResults(vec![ToolResult { id: "t".into(), content: "ok".into() }]),
        ];
        let (mi, _) = oldest_tool_result(&h).expect("the second ToolResults has entries");
        assert_eq!(mi, 3);
    }

    #[test]
    fn drop_result_and_pair_drops_both_use_and_result() {
        let mut h = vec![
            Message::User("go".into()),
            Message::Assistant(vec![Block::ToolUse {
                id: "t1".into(), name: "bash".into(), input: json!({}),
            }]),
            Message::ToolResults(vec![ToolResult { id: "t1".into(), content: "ok".into() }]),
        ];
        drop_result_and_pair(&mut h, 2, 0);
        // the tool_use and its tool_result both gone; the user message survives
        assert_eq!(h.len(), 1, "the empty assistant + result messages collapse: {h:?}");
        assert!(matches!(&h[0], Message::User(_)));
    }

    #[test]
    fn drop_result_and_pair_keeps_thinking_blocks_for_other_calls() {
        // a batch where thinking was reasoning towards *two* calls:
        // dropping the first result must not lose the thinking block —
        // it's still the reasoning for the second call
        let mut h = vec![
            Message::User("go".into()),
            Message::Assistant(vec![
                Block::Thinking { text: "two calls".into(), signature: None },
                Block::ToolUse { id: "t1".into(), name: "bash".into(), input: json!({}) },
                Block::ToolUse { id: "t2".into(), name: "read_file".into(), input: json!({"path": "x"}) },
            ]),
            Message::ToolResults(vec![
                ToolResult { id: "t1".into(), content: "ok".into() },
                ToolResult { id: "t2".into(), content: "y".into() },
            ]),
        ];
        // find t1's position in the result message
        let (mi, _) = oldest_tool_result(&h).unwrap();
        // bi is hard-coded to 0 in `oldest_tool_result`, so the first
        // result is always the one dropped
        drop_result_and_pair(&mut h, mi, 0);
        let j = hist(&h).to_string();
        assert!(!j.contains("\"id\": \"t1\""), "t1's tool_use is gone");
        assert!(j.contains("t2"), "t2's pair survives");
        assert!(j.contains("two calls"), "the thinking block survives t2's sake");
    }

    #[test]
    fn drop_result_and_pair_returns_silently_on_wrong_index() {
        // mi points at a non-ToolResults: the function must not panic
        let mut h = vec![Message::User("go".into())];
        let before = h.clone();
        drop_result_and_pair(&mut h, 0, 0);
        assert_eq!(hist(&h), hist(&before));
    }

    #[test]
    fn fit_context_returns_false_when_history_already_fits() {
        let h = vec![Message::User("hi".into())];
        assert!(!fit_context(&mut h.clone(), u64::MAX));
        let mut h2 = h.clone();
        assert!(!fit_context(&mut h2, est_tokens(&h) + 1000));
    }

    #[test]
    fn fit_context_returns_false_when_there_is_nothing_to_trim() {
        // an over-limit history with no tool_results: nothing changes
        let mut h = vec![Message::User("go".into())];
        assert!(!fit_context(&mut h, 0));
    }

    // ---- fit_context: integration (JSON wire shape end-to-end) ------------

    #[test]
    fn fit_context_trims_oldest_tool_result_and_keeps_pairing() {
        let big = "x".repeat(10_000);
        let mut history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]}),
        ]);
        assert!(fit_context(&mut history, 500));
        let j = hist(&history);
        assert!(j[2]["content"][0]["content"].as_str().unwrap().contains("trimmed"));
        assert_eq!(j[1]["content"][0]["id"], j[2]["content"][0]["tool_use_id"]);
        let mut tiny = hv(vec![json!({"role": "user", "content": "t"})]);
        assert!(!fit_context(&mut tiny, 10_000));
    }

    #[test]
    fn fit_context_drops_short_tool_result_messages_when_still_over() {
        let mut history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "one call, one thought"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}), // too short to trim in place
            json!({"role": "assistant", "content": [{"type": "text", "text": "padding to keep the estimate high"}]}),
        ]);
        let limit = est_tokens(&history) - 1; // guarantee the first pass is over
        assert!(fit_context(&mut history, limit));
        // the whole exchange is gone — the assistant message held only the call
        assert_eq!(history.len(), 2);
        assert!(!hist(&history).to_string().contains("tool_result"));
        assert_pairing(&history);
    }

    #[test]
    fn fit_context_keeps_unpaired_blocks_of_partially_dropped_batches() {
        // a batch of two calls where only the first result is minimal: the
        // second call/result pair must survive the first one's removal
        let mut history = hv(vec![
            json!({"role": "user", "content": "go"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "plan: run two"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "true"}},
                {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "x"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},        // minimal → dropped with t1
                {"type": "tool_result", "tool_use_id": "t2", "content": "y".repeat(500)}]}), // trimmed in place
            json!({"role": "assistant", "content": [{"type": "text", "text": "padding to keep the estimate high"}]}),
        ]);
        let limit = est_tokens(&history) - 1;
        assert!(fit_context(&mut history, limit));
        assert_pairing(&history);
        let body = hist(&history).to_string();
        assert!(!body.contains("\"id\": \"t1\""), "t1's tool_use must not survive its result: {body}");
        assert!(body.contains("t2"), "t2's pair must both survive: {body}");
        assert!(body.contains("thinking"), "thinking led up to t2 as well — it stays: {body}");
    }
}