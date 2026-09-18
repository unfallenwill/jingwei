//! Project conventions loaded from AGENTS.md files (https://agents.md/).
//!
//! v1: walk up from workspace to the filesystem root and load the closest
//! AGENTS.md (cap 64 KB). The body goes into the system prompt's prefix
//! cache as one labelled section; the banner names the path so the user
//! sees what got loaded.
//!
//! ## Architectural room for v2
//!
//! The struct fields are stable for comparison:
//! - `load()` is pure and side-effect free — calling it a second time
//!   gives a fresh context for diff/refresh purposes.
//! - `found_path` + `content` are addressable; a future `diff_since(&self,
//!   other: &AgentsMdContext)` is straightforward to add without breaking
//!   callers.
//! - A future context message (for nested AGENTS.md paths that aren't
//!   loaded eagerly) can be built by adding a `nested: Vec<PathBuf>`
//!   field and a sibling `as_context_message()` method.
//!
//! v1 deliberately does **not** implement refresh, nested enumeration, or
//! the diff method. Each lives in its own v2 commit.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hard cap on the AGENTS.md body. 64 KB ≈ 20K tokens — enough for real
/// projects, small enough that the cached prefix stays cheap. A file
/// over the cap is truncated with a marker line so the model knows it
/// saw a slice, not the whole thing.
const MAX_ROOT_BYTES: usize = 64 * 1024;

/// The filename the AGENTS.md spec mandates. Case-sensitive on purpose:
/// the spec spells it exactly that way.
const FILENAME: &str = "AGENTS.md";

/// The "project conventions" half of the conversation context: zero, one,
/// or (in future) many AGENTS.md files the agent has discovered. v1 only
/// loads the closest one walking up; the rest of the spec (nested paths,
/// refresh, context messages) lives behind the v2 surface.
#[derive(Debug, Clone)]
pub struct AgentsMdContext {
    /// The directory we started walking up from. Kept for diagnostics
    /// ("loaded from <workspace>") and so a future refresh can re-walk
    /// without the caller having to remember it. Currently the load
    /// path never reads it back, but v2 refresh needs it.
    #[allow(dead_code)]
    pub workspace: PathBuf,
    /// Path of the AGENTS.md we loaded, walking up. `None` when no
    /// AGENTS.md was found at any ancestor — agent sees no conventions.
    pub found_path: Option<PathBuf>,
    /// The body, capped at [`MAX_ROOT_BYTES`]. Empty when `found_path`
    /// is `None`.
    pub content: String,
    /// Unix seconds at load time. The banner and any future context
    /// message embed this so the model knows how fresh the view is —
    /// and v2 diff needs it for "what changed since".
    #[allow(dead_code)]
    pub loaded_at: u64,
}

impl AgentsMdContext {
    /// Walk up from `workspace` to the filesystem root. The first
    /// directory that has an `AGENTS.md` is the one we use (closest
    /// wins, per the AGENTS.md spec).
    ///
    /// Pure: no IO side effects beyond the one file read on the found
    /// path. A missing or unreadable file is silently skipped — the next
    /// ancestor is tried in its place.
    pub fn load(workspace: &Path) -> Self {
        let mut cur = Some(workspace.to_path_buf());
        let mut found_path = None;
        let mut content = String::new();
        while let Some(d) = cur {
            let candidate = d.join(FILENAME);
            if let Ok(text) = std::fs::read_to_string(&candidate) {
                found_path = Some(candidate);
                content = cap(text);
                break;
            }
            cur = d.parent().map(|p| p.to_path_buf());
        }
        Self {
            workspace: workspace.to_path_buf(),
            found_path,
            content,
            loaded_at: now_secs(),
        }
    }

    /// The empty context: same shape, but no file found. The `--no-agents-md`
    /// flag uses this so the rest of the agent runs as if the user simply
    /// had no AGENTS.md — no banner, no extras, no surprise.
    pub fn empty(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            found_path: None,
            content: String::new(),
            loaded_at: now_secs(),
        }
    }

    /// The text to append after `crate::SYSTEM` in the system prompt.
    /// Returns `""` when no AGENTS.md was loaded, so callers don't have
    /// to special-case the empty path. Already wrapped in a labelled
    /// section so the model can recognize where the content came from.
    pub fn system_prompt_extras(&self) -> String {
        if self.content.is_empty() {
            return String::new();
        }
        let path = self.found_path.as_deref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        format!(
            "\n\n# Project conventions (AGENTS.md at {})\n\n{}",
            path, self.content
        )
    }

    /// One short banner line, or `None` when nothing was loaded. The
    /// point is transparency: a user who sees this banner knows what the
    /// agent is about to be told about their project.
    pub fn banner(&self) -> Option<String> {
        let path = self.found_path.as_ref()?.display().to_string();
        Some(format!(
            "jingwei · loaded AGENTS.md at {path} ({} bytes)",
            self.content.len()
        ))
    }
}

/// Compose the full system prompt: base `SYSTEM` plus the AGENTS.md
/// extras, separated by a blank line. Returns a `String` (the extras
/// force ownership) — vendors are free to intern or share the result
/// if the cache key works out.
pub fn full_system_prompt(extras: &str) -> String {
    if extras.is_empty() {
        return crate::SYSTEM.to_string();
    }
    format!("{}\n\n{}", crate::SYSTEM, extras)
}

/// Truncate `text` to at most [`MAX_ROOT_BYTES`] bytes on a char
/// boundary, then append a marker line so the model can tell it saw a
/// slice. The marker is in the body so it lands in the same prompt
/// segment as the truncated text.
fn cap(mut text: String) -> String {
    if text.len() <= MAX_ROOT_BYTES {
        return text;
    }
    let mut end = MAX_ROOT_BYTES;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str(&format!(
        "\n\n[…truncated, AGENTS.md exceeded {} bytes]",
        MAX_ROOT_BYTES
    ));
    text
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::temp_dir;

    #[test]
    fn load_returns_empty_when_no_agents_md_exists() {
        let dir = temp_dir("agents_md_none");
        let ctx = AgentsMdContext::load(&dir);
        assert!(ctx.found_path.is_none());
        assert!(ctx.content.is_empty());
        assert!(ctx.system_prompt_extras().is_empty());
        assert!(ctx.banner().is_none());
    }

    #[test]
    fn load_finds_agents_md_in_workspace() {
        let dir = temp_dir("agents_md_root");
        std::fs::write(dir.join("AGENTS.md"), "use pnpm").unwrap();
        let ctx = AgentsMdContext::load(&dir);
        assert_eq!(ctx.found_path, Some(dir.join("AGENTS.md")));
        assert_eq!(ctx.content, "use pnpm");
        let extra = ctx.system_prompt_extras();
        assert!(extra.contains("# Project conventions"));
        assert!(extra.contains("use pnpm"));
        assert!(extra.contains(&dir.display().to_string()));
        let b = ctx.banner().unwrap();
        assert!(b.contains("AGENTS.md"));
        assert!(b.contains("use pnpm".len().to_string().as_str())
            || b.contains("9 bytes")); // "use pnpm" is 9 bytes
    }

    #[test]
    fn load_walks_up_to_find_closest_ancestor() {
        let root = temp_dir("agents_md_walkup");
        std::fs::write(root.join("AGENTS.md"), "from root").unwrap();
        let nested = root.join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();
        let ctx = AgentsMdContext::load(&nested);
        assert_eq!(ctx.found_path, Some(root.join("AGENTS.md")));
        assert_eq!(ctx.content, "from root");
    }

    #[test]
    fn load_closest_wins_over_ancestor() {
        // Both the workspace and an ancestor have AGENTS.md — closest wins.
        let root = temp_dir("agents_md_closest");
        std::fs::write(root.join("AGENTS.md"), "from root").unwrap();
        let inner = root.join("pkg");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(inner.join("AGENTS.md"), "from pkg").unwrap();
        let ctx = AgentsMdContext::load(&inner);
        assert_eq!(ctx.found_path, Some(inner.join("AGENTS.md")));
        assert_eq!(ctx.content, "from pkg");
    }

    #[test]
    fn cap_truncates_oversize_files_with_a_marker() {
        let big = "x".repeat(MAX_ROOT_BYTES + 100);
        let capped = cap(big);
        assert!(capped.contains("truncated"), "should mark truncation");
        // the original was MAX + 100 bytes; the cap marker adds text but
        // the leading prefix is still bounded
        let prefix_end = capped.find("…").unwrap();
        assert!(prefix_end <= MAX_ROOT_BYTES + 50);
    }

    #[test]
    fn cap_lands_on_a_char_boundary() {
        // a multi-byte char straddles the boundary — must back off, not slice
        let mut big = "x".repeat(MAX_ROOT_BYTES - 2);
        big.push_str("精卫填海");
        let capped = cap(big.clone());
        let marker = "truncated,";
        let marker_pos = capped.find(marker).unwrap();
        let kept = &capped[..marker_pos];
        assert!(kept.is_char_boundary(kept.len()),
            "kept prefix must end on a char boundary, got {} bytes ending in {:?}",
            kept.len(),
            kept.chars().last());
    }

    #[test]
    fn empty_constructor_has_no_path_no_content() {
        let dir = temp_dir("agents_md_empty");
        let ctx = AgentsMdContext::empty(&dir);
        assert!(ctx.found_path.is_none());
        assert!(ctx.content.is_empty());
        assert!(ctx.banner().is_none());
    }

    #[test]
    fn full_system_prompt_returns_base_when_extras_empty() {
        let s = full_system_prompt("");
        assert_eq!(s, crate::SYSTEM);
    }

    #[test]
    fn full_system_prompt_appends_extras_with_a_blank_separator() {
        let s = full_system_prompt("\n\n# extra");
        assert!(s.starts_with(crate::SYSTEM));
        assert!(s.contains("\n\n# extra"));
        // exactly one blank line between base and extras — not two, not zero
        assert!(s.ends_with("# extra"));
    }

    #[test]
    fn system_prompt_extras_is_empty_when_nothing_loaded() {
        let dir = temp_dir("agents_md_no_extras");
        let ctx = AgentsMdContext::load(&dir);
        assert_eq!(ctx.system_prompt_extras(), "");
    }
}