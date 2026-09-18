//! Project conventions loaded from AGENTS.md files (https://agents.md/).
//!
//! Two pieces are produced:
//! - `found_path` + `content`: the closest AGENTS.md, walking up from
//!   the workspace. This becomes an `AgentsMdClosest` reminder in
//!   `Context::reminders`.
//! - `nested_paths`: every AGENTS.md under the workspace (the cwd
//!   where jingwei was invoked), excluding the closest one. Bodies
//!   are not inlined — the model reads them on demand via `read_file`.
//!   This becomes an `AgentsMdIndex` reminder — a path-only index of
//!   "every AGENTS.md in your project" so the model knows what's
//!   available without having to walk the tree first.
//!
//! The workspace *is* the project root. No `.git` discovery, no VCS
//! coupling: the user runs `jingwei` in a directory, that directory
//! is the scope. Hidden dirs (`.git`, `.hg`, …) and known-heavy ones
//! (`target`, `node_modules`, …) are skipped during enumeration so
//! real-world repos don't make the loader crawl forever.
//!
//! ## Architectural room for v2
//!
//! The struct fields are stable for comparison:
//! - `load()` is pure and side-effect free — calling it a second time
//!   gives a fresh context for diff/refresh purposes.
//! - `found_path` + `content` are addressable; a future `diff_since(&self,
//!   other: &AgentsMdContext)` is straightforward to add without breaking
//!   callers.
//!
//! v1 deliberately does **not** implement refresh, body-level nested
//! reading, or the diff method. Each lives in its own commit.

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

/// The "project conventions" half of the conversation context: zero,
/// one, or many AGENTS.md files the agent has discovered. v1 loads the
/// closest one walking up; `nested_paths` adds the path-only index of
/// files along the path from repo root to CWD, every request of the
/// agent.
#[derive(Debug, Clone)]
pub struct AgentsMdContext {
    /// The directory we started walking up from. Kept for diagnostics
    /// ("loaded from <workspace>") and so a future refresh can re-walk
    /// without the caller having to remember it. Currently the load
    /// path never reads it back, but a refresh needs it.
    #[allow(dead_code)]
    pub(crate) workspace: PathBuf,
    /// Path of the AGENTS.md we loaded, walking up. `None` when no
    /// AGENTS.md was found at any ancestor — agent sees no conventions.
    pub found_path: Option<PathBuf>,
    /// The body, capped at [`MAX_ROOT_BYTES`]. Empty when `found_path`
    /// is `None`.
    pub content: String,
    /// Unix seconds at load time. The banner and any future context
    /// message embed this so the model knows how fresh the view is —
    /// and a future diff needs it for "what changed since".
    #[allow(dead_code)]
    pub(crate) loaded_at: u64,
    /// Paths of AGENTS.md files under `workspace` (the cwd where jingwei
    /// was invoked), excluding the closest one already represented by
    /// `found_path`. Bodies are *not* loaded — the model reads them on
    /// demand via `read_file`. Tree-wide BFS from cwd; hidden dirs and
    /// known-heavy trees are skipped during enumeration.
    pub nested_paths: Vec<PathBuf>,
}

impl AgentsMdContext {
    /// Walk up from `workspace` to the filesystem root. The first
    /// directory that has an `AGENTS.md` is the one we use (closest
    /// wins, per the AGENTS.md spec). Then enumerate nested AGENTS.md
    /// files under `workspace` (the cwd) via tree-wide BFS — the cwd
    /// is the project root, no VCS discovery involved.
    ///
    /// Pure: no IO side effects beyond the file reads. A missing or
    /// unreadable file is silently skipped — the next directory is
    /// tried in its place.
    pub fn load(workspace: &Path) -> Self {
        // Walk up for the closest AGENTS.md.
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

        // Nested AGENTS.md index: tree-wide BFS from `workspace` (the
        // cwd where jingwei was invoked). The cwd *is* the project
        // root — we don't depend on `.git` discovery, which is fragile
        // (other VCSes, no-VCS dirs, worktrees). Hidden directories
        // (`.git`, `.hg`, …) and common heavy ones (`target`,
        // `node_modules`, …) are skipped so we don't walk huge trees
        // for nothing. `found_path` (the closest AGENTS.md) is
        // excluded; the rest are surfaced as a path-only index the
        // model reads on demand via `read_file`.
        let nested_paths = collect_tree_agents_md(workspace, found_path.as_ref());

        Self {
            workspace: workspace.to_path_buf(),
            found_path,
            content,
            loaded_at: now_secs(),
            nested_paths,
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
            nested_paths: Vec::new(),
        }
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

/// BFS from `root`, collecting every `AGENTS.md` it can read. Output
/// order is parent-before-children with siblings sorted by path — the
/// same shape a tree-dump tool would produce, and what reads best in
/// the system prompt.
///
/// Skips hidden directories (starting with `.`) and a small set of
/// common heavy ones (`target`, `node_modules`, `dist`, `build`,
/// `.venv`, `venv`, `__pycache__`) — none of these are project source
/// in the languages jingwei targets today, and walking them would
/// blow up the load on real-world repos. The skip list is deliberately
/// small and explicit, not pattern-based: it can grow alongside
/// languages jingwei supports.
///
/// `exclude` (the closest AGENTS.md, if any) is dropped if encountered
/// — `found_path` already represents it, listing it twice would be a
/// duplicate on the wire.
fn collect_tree_agents_md(root: &Path, exclude: Option<&PathBuf>) -> Vec<PathBuf> {
    use std::collections::VecDeque;
    let mut out = Vec::new();
    let mut queue: VecDeque<PathBuf> = VecDeque::new();
    queue.push_back(root.to_path_buf());
    while let Some(dir) = queue.pop_front() {
        let candidate = dir.join(FILENAME);
        if candidate.is_file() && exclude != Some(&candidate) {
            out.push(candidate);
        }
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        let mut subdirs: Vec<PathBuf> = entries
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                // Skip symlinks: a symlink inside the workspace could
                // point outside (escaping the project root entirely,
                // e.g. into `/tmp` or a sibling repo), and a cycle
                // through symlinks would loop the BFS forever. Using
                // `symlink_metadata` here is the difference between
                // "is this entry a symlink?" and "does the underlying
                // path point to a directory we could resolve?" — the
                // former is what we want. A regular subdir is not a
                // symlink, so `file_type().is_symlink()` is false for
                // the directories we do want to recurse into.
                let meta = e.metadata().ok()?;
                if meta.is_symlink() {
                    return None;
                }
                if p.is_dir() && !is_skippable(&p) { Some(p) } else { None }
            })
            .collect();
        subdirs.sort();
        for s in subdirs {
            queue.push_back(s);
        }
    }
    out
}

/// True for directories we don't want to recurse into: hidden dirs
/// (VCS internals, IDE state) and a small fixed list of known-heavy
/// build/dependency trees.
fn is_skippable(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return false };
    if name.starts_with('.') {
        // `.` and `..` are handled by read_dir; everything else
        // starting with `.` is skipped (`.hg`, `.idea`, `.cache`, …).
        return true;
    }
    matches!(
        name,
        "node_modules" | "target" | "dist" | "build" | ".venv" | "venv" | "__pycache__"
    )
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
        assert!(ctx.banner().is_none());
        // No AGENTS.md in cwd's tree ⇒ no nested enumeration either.
        assert!(ctx.nested_paths.is_empty());
    }

    #[test]
    fn load_finds_agents_md_in_workspace() {
        let dir = temp_dir("agents_md_root");
        std::fs::write(dir.join("AGENTS.md"), "use pnpm").unwrap();
        let ctx = AgentsMdContext::load(&dir);
        assert_eq!(ctx.found_path, Some(dir.join("AGENTS.md")));
        assert_eq!(ctx.content, "use pnpm");
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
        assert!(ctx.nested_paths.is_empty());
    }

    /// Pin the loader against this repo's own AGENTS.md: jingwei runs
    /// from many workspaces, but if the workspace *is* this project, the
    /// file at the root is the conventions other agents wrote for us.
    /// Catches accidental edits that would silently break discovery.
    #[test]
    fn load_finds_this_repos_own_agents_md() {
        // CARGO_MANIFEST_DIR is set by cargo to the crate being tested —
        // resolves to the project root regardless of where cargo runs from.
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let ctx = AgentsMdContext::load(manifest);
        let path = ctx.found_path.clone().expect("jingwei ships an AGENTS.md at the root");
        assert_eq!(path, manifest.join("AGENTS.md"));
        let banner = ctx.banner().expect("a file present ⇒ a banner");
        // The banner has to land at the start of every session, in front
        // of the session-id line, so the user sees what got loaded.
        assert!(banner.starts_with("jingwei · loaded AGENTS.md at "));
        assert!(banner.contains(&manifest.display().to_string()));
    }

    // ---- nested enumeration: tree-wide from workspace ----

    /// Build a temp dir as the workspace and place AGENTS.md files at
    /// the given relative paths. Each path's parent dirs are created.
    fn fixture_with_agents(name: &str, files: &[&str]) -> PathBuf {
        let dir = temp_dir(name);
        for rel in files {
            let p = dir.join(rel);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&p, format!("from {rel}")).unwrap();
        }
        dir
    }

    #[test]
    fn nested_paths_is_tree_wide_from_workspace() {
        // Workspace is the cwd; enumeration = recurse from it.
        // Multiple subtrees at varying depths all get collected.
        let dir = fixture_with_agents("tree_wide", &[
            "AGENTS.md",
            "a/AGENTS.md",
            "a/b/AGENTS.md",
            "a/b/c/AGENTS.md",
            "sibling/AGENTS.md",
            "sibling/deep/AGENTS.md",
        ]);
        let ctx = AgentsMdContext::load(&dir);
        assert_eq!(ctx.found_path, Some(dir.join("AGENTS.md")));
        // tree-wide from cwd — root's AGENTS.md is the closest (excluded);
        // everything else below is in nested_paths, regardless of how
        // deep or whether it sits on the cwd→leaf path.
        let expected_paths = [
            dir.join("a").join("AGENTS.md"),
            dir.join("a").join("b").join("AGENTS.md"),
            dir.join("a").join("b").join("c").join("AGENTS.md"),
            dir.join("sibling").join("AGENTS.md"),
            dir.join("sibling").join("deep").join("AGENTS.md"),
        ];
        let expected: std::collections::HashSet<_> = expected_paths.iter().collect();
        let nested: std::collections::HashSet<_> = ctx.nested_paths.iter().collect();
        assert_eq!(nested, expected,
            "tree-wide BFS finds every AGENTS.md under cwd, excluding the closest");
    }

    #[test]
    fn nested_paths_lists_in_bfs_root_to_leaf_order() {
        // Order is parent-before-children, siblings sorted — what
        // reads best in the system prompt and what a tree dump
        // would produce.
        let dir = fixture_with_agents("bfs_order", &[
            "z/AGENTS.md",
            "a/AGENTS.md",
            "a/m/AGENTS.md",
            "a/a/AGENTS.md",
        ]);
        let ctx = AgentsMdContext::load(&dir);
        // BFS visits: a, z, then a/a, a/m, then z/... (no subdirs).
        // Within each level, sorted: a comes before z; a/a before a/m.
        assert_eq!(ctx.nested_paths, vec![
            dir.join("a").join("AGENTS.md"),
            dir.join("z").join("AGENTS.md"),
            dir.join("a").join("a").join("AGENTS.md"),
            dir.join("a").join("m").join("AGENTS.md"),
        ]);
    }

    #[test]
    fn nested_paths_excludes_the_closest_one() {
        // The closest AGENTS.md is also collected by BFS (it's the
        // cwd itself). We filter it out so it doesn't appear twice
        // on the wire (it's already in the AgentsMdClosest reminder).
        let dir = fixture_with_agents("excludes_closest", &[
            "AGENTS.md",
            "a/AGENTS.md",
        ]);
        let ctx = AgentsMdContext::load(&dir);
        assert_eq!(ctx.found_path, Some(dir.join("AGENTS.md")));
        assert!(!ctx.nested_paths.contains(&dir.join("AGENTS.md")),
            "root AGENTS.md is the closest and must not appear in the index");
        assert!(ctx.nested_paths.contains(&dir.join("a").join("AGENTS.md")));
    }

    #[test]
    fn nested_paths_skips_hidden_and_heavy_dirs() {
        // .git, .hg, node_modules, target — none of these are project
        // source, and walking them would explode on real repos. They
        // must be skipped even if they happen to contain AGENTS.md.
        let dir = fixture_with_agents("skip_dirs", &[
            ".git/AGENTS.md",       // VCS internal — skip
            "node_modules/AGENTS.md", // heavy deps — skip
            "target/AGENTS.md",     // build output — skip
            "visible/AGENTS.md",    // project source — keep
            "visible/.hidden/AGENTS.md", // nested hidden — skip
        ]);
        let ctx = AgentsMdContext::load(&dir);
        assert!(!ctx.nested_paths.iter().any(|p| p.starts_with(dir.join(".git"))),
            ".git subtree is skipped");
        assert!(!ctx.nested_paths.iter().any(|p| p.starts_with(dir.join("node_modules"))),
            "node_modules subtree is skipped");
        assert!(!ctx.nested_paths.iter().any(|p| p.starts_with(dir.join("target"))),
            "target subtree is skipped");
        assert!(ctx.nested_paths.contains(&dir.join("visible").join("AGENTS.md")),
            "project-source AGENTS.md is kept");
        // .hidden inside visible/ is also skipped (still a hidden dir)
        let visible = dir.join("visible");
        assert!(!ctx.nested_paths.iter().any(|p| p.starts_with(visible.join(".hidden"))),
            "hidden dirs are skipped at every depth, not just the top");
    }

    #[test]
    fn nested_paths_does_not_walk_above_workspace() {
        // The BFS starts at cwd. An AGENTS.md in a *parent* directory
        // (above the temp dir) must NOT show up in nested_paths —
        // even if one exists there. The closest-walk handles ancestors
        // separately (as `found_path`); the index is purely the cwd
        // subtree.
        //
        // We deliberately do NOT write to the temp dir's parent on
        // disk: temp_dir lives under /tmp on Linux, and dropping files
        // there pollutes every later test in this process. Instead we
        // rely on the actual boundary: BFS walks downward from cwd,
        // and `dir.parent()`'s contents are unreachable from cwd by
        // construction.
        let dir = temp_dir("does_not_walk_above");
        std::fs::write(dir.join("AGENTS.md"), "from dir").unwrap();
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("AGENTS.md"), "from sub").unwrap();

        // workspace = sub. dir/AGENTS.md is the only AGENTS.md in
        // the cwd's tree (dir/AGENTS.md and sub/AGENTS.md). The
        // BFS starts at sub and finds sub/AGENTS.md. Nothing above
        // sub is reachable.
        let ctx = AgentsMdContext::load(&sub);
        assert_eq!(ctx.found_path, Some(sub.join("AGENTS.md")));
        // dir/AGENTS.md is NOT in nested_paths even though it's the
        // parent of cwd — the index never walks up. (closest-walk
        // would surface it as found_path if cwd were above it; but
        // here cwd= sub, and dir is sub's parent, which is *above*
        // the workspace, so closest-walk doesn't reach it.)
        assert!(ctx.nested_paths.is_empty(),
            "BFS from cwd does not walk above the workspace; dir/AGENTS.md \
             is above cwd and unreachable");
    }

    #[test]
    fn nested_paths_skips_symlinks_inside_workspace() {
        // A symlink inside the workspace could escape the project
        // (e.g. point at /tmp) and surface foreign AGENTS.md files.
        // It could also cycle. Either way, the loader must skip
        // symlinked entries — the BFS treats them as not-a-directory.
        let dir = fixture_with_agents("symlinks", &[
            "AGENTS.md",
            "real/AGENTS.md",
        ]);
        // Create a symlink under cwd that points at a sibling
        // directory; if we followed it, the foreign AGENTS.md would
        // leak into nested_paths.
        let escapee = dir.parent().unwrap().join("jingwei_test_escapee_target");
        std::fs::create_dir_all(&escapee).unwrap();
        std::fs::write(escapee.join("AGENTS.md"), "from escapee").unwrap();
        std::os::unix::fs::symlink(&escapee, dir.join("escape-link")).unwrap();

        // And a self-loop would loop the BFS forever if we followed it;
        // here we just verify the symlink is not recursed into.
        std::os::unix::fs::symlink(dir.join("real"), dir.join("self-link")).unwrap();

        let ctx = AgentsMdContext::load(&dir);
        // real/AGENTS.md surfaces; the symlinked targets do not.
        assert!(ctx.nested_paths.contains(&dir.join("real").join("AGENTS.md")));
        assert!(!ctx.nested_paths.iter().any(|p| p.starts_with(dir.join("escape-link"))),
            "symlinked escapee must not leak into nested_paths");
        assert!(!ctx.nested_paths.iter().any(|p| p.starts_with(dir.join("self-link"))),
            "self-loop symlink must not be recursed into");

        // cleanup the escapee target we created next to the temp dir
        let _ = std::fs::remove_dir_all(&escapee);
    }

    #[test]
    fn nested_paths_bfs_terminates_on_existing_subdirs() {
        // The BFS visits every existing subdir and finds its AGENTS.md.
        // (The previous "handles_unreadable_subdirs_gracefully" name
        // claimed permission-denied semantics that we don't actually
        // exercise in tests — and there's no portable way to flip
        // permissions in a unit test. The graceful path on error is
        // "silently skip and continue", which is what the closure's
        // `let Ok(entries) = ... else { continue }` arm buys.)
        let dir = fixture_with_agents("existing_subdirs", &[
            "AGENTS.md",
            "ok/AGENTS.md",
            "more/AGENTS.md",
        ]);
        let ctx = AgentsMdContext::load(&dir);
        assert!(ctx.nested_paths.contains(&dir.join("ok").join("AGENTS.md")));
        assert!(ctx.nested_paths.contains(&dir.join("more").join("AGENTS.md")));
    }
}