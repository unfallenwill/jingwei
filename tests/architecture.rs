// Architecture test: enforce jingwei's layering rules. Pure stdlib —
// walks `src/**/*.rs`, parses `use crate::xxx::...` at the top of each
// file, builds a directed graph, and asserts no upward edge.
//
// Layering (lower → higher, deps go downward only):
//
//   L0  display, ir, session, turn                  — pure data types + state machines (turn is per-run, no IO)
//   L1  cancel, file_io, format, ledger, agents_md — infra primitives (agents_md walks the FS — IO, no other crate deps)
//   L3  config, settings, context, edit, tools, tool_runtime, login, mcp + submodules
//                                                  — tool/config tier; reaches into api (L4) for the Vendor port + policy enums
//   L4  api (+ api::minimax/zai/deepseek)          — vendor + transport
//   L5  main, plain, tui (+ tui::model/update/view)— composition + frontends
//   T   test_util                                  — test-only helpers
//
// Rules:
//   - L0..L5 may not import from any higher-numbered layer (no upward).
//   - T may only be imported by other modules' `#[cfg(test)] mod tests`
//     blocks (we can't enforce that here, so T → anything is allowed by
//     the production `use` lines, with the caveat that test_util is itself
//     `cfg(test)` and so its symbols don't reach production binaries).
//   - L5 (main/plain/tui) may import anything.
//
// What this catches: circular-feeling imports (like L2 → L4), dead
// modules that take a heavy dep, and accidental coupling between layers
// that should stay independent. Pure stdlib, no extra deps.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layer { L0, L1, L3, L4, L5, T }

impl Layer {
    /// Lower layers (smaller number) may not import from higher-numbered ones.
    /// T (test) is a special case — it lives behind `cfg(test)`, so any
    /// production-code import of it is a bug.
    fn rank(self) -> u8 {
        match self { Self::L0=>0, Self::L1=>1, Self::L3=>3, Self::L4=>4, Self::L5=>5, Self::T=>99 }
    }
}

fn layer_of(mod_name: &str) -> Option<Layer> {
    let l = match mod_name {
        "display" | "ir" | "session" | "turn" => Layer::L0, // turn is the pure per-run state machine; data + transfer rules, no IO
        "cancel" | "file_io" | "format" | "ledger" | "agents_md" | "reminder" => Layer::L1, // agents_md walks the FS — IO primitive, no other crate deps; reminder is pure data the agent core threads through Context
        "config" | "settings" | "context" => Layer::L3, // tool/config tier; may reach into api for the Vendor port + policy enums; context builds the per-turn model state
        "edit" | "tools" | "tool_runtime" | "login" | "mcp" | "mcp::wire" | "mcp::health"
            | "mcp::config" | "mcp::client" | "mcp::stdio" | "mcp::http" | "mcp::inner"
            | "mcp::stub" => Layer::L3, // tool/config tier; MCP is a tool-provider sibling to `tools`
        "api" | "api::minimax" | "api::zai" | "api::deepseek" => Layer::L4,
        "main" | "plain" | "tui" | "tui::model" | "tui::update" | "tui::view" => Layer::L5,
        "test_util" => Layer::T,
        _ => return None,
    };
    Some(l)
}

/// `src/api/mod.rs` is the `api` module; `src/api/deepseek.rs` is the
/// `api::deepseek` submodule. `src/main.rs` is just `main`. This function
/// turns a file path into its module path.
fn file_to_module(path: &Path) -> Option<String> {
    let rel = path.strip_prefix("src/").ok()?.to_str()?;
    let rel = rel.trim_end_matches(".rs").replace(['/', '\\'], "::");
    if rel == "main" { return Some("main".into()); }
    if let Some(stripped) = rel.strip_suffix("::mod") {
        return Some(stripped.to_string());
    }
    Some(rel)
}

/// Remove every `#[cfg(test)] mod tests { ... }` block from `text`,
/// including its outer attribute. Brace-counting handles nested braces.
/// Used so that `use` statements inside test modules don't pollute the
/// production-layer dependency graph.
fn strip_cfg_test_mod_tests(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Look for `#[cfg(test)]` (possibly with other whitespace) followed
        // (with anything between) by `mod tests` and a `{`.
        if bytes[i..].starts_with(b"#[cfg(test)]") || bytes[i..].starts_with(b"#[ cfg (test) ]") {
            // Find the matching `{` and brace-balance.
            // Advance to the `{`.
            let mut j = i;
            while j < bytes.len() && bytes[j] != b'{' { j += 1; }
            if j == bytes.len() { out.push_str(&text[i..]); break; }
            let mut depth = 1usize;
            j += 1;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            // Append everything between the previous content and `start`
            // unchanged; skip from `start` to `j`.
            // (We only handle top-level `#[cfg(test)]` — nested cfg(test)
            // mod blocks are unusual and not produced by this codebase.)
            i = j;
        } else {
            // Copy one char and advance.
            // We can't easily push a single UTF-8 char; just push the byte
            // and let the next iteration handle the rest. (Works because
            // we're not actually parsing — just stripping.)
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Walk src/ and build: module name → list of imported top-level modules.
fn collect_edges() -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for entry in walkdir("src") {
        let path = match entry { Ok(p) => p, Err(_) => continue };
        if path.extension().and_then(|e| e.to_str()) != Some("rs") { continue; }
        let Some(mod_name) = file_to_module(&path) else { continue; };
        let Ok(text) = fs::read_to_string(&path) else { continue; };
        // Skip `#[cfg(test)] mod tests { ... }` blocks — those imports
        // are test-only and never reach the production binary.
        let prod_text = strip_cfg_test_mod_tests(&text);
        let mut deps = Vec::new();
        for line in prod_text.lines() {
            // Only `use crate::xxx::...` — not function bodies, not `super::*`.
            let Some(rest) = line.trim_start().strip_prefix("use crate::") else { continue; };
            // The first `::`-separated segment is the target module (possibly
            // followed by a sub-item). Strip sub-items: `api::minimax::Effort`
            // → target `api::minimax`.
            let top = rest.split("::").next().unwrap_or("");
            // Special-case api/mod.rs: top-level `use crate::api` from outside
            // api means "the api module". Use crate::api::minimax → "api::minimax".
            let mut target = top.to_string();
            if target == "api" {
                // Check if the next segment is one of the known submodules.
                let next = rest.split("::").nth(1).unwrap_or("");
                if matches!(next, "minimax" | "zai" | "deepseek") {
                    target = format!("api::{next}");
                }
            }
            if target != mod_name {
                deps.push(target);
            }
        }
        deps.sort();
        deps.dedup();
        out.insert(mod_name, deps);
    }
    out
}

/// Tiny recursive directory walker (avoids pulling in the `walkdir` crate).
fn walkdir<P: AsRef<Path>>(root: P) -> Vec<std::io::Result<PathBuf>> {
    let mut out = Vec::new();
    fn visit(p: &Path, out: &mut Vec<std::io::Result<PathBuf>>) {
        let entries = match fs::read_dir(p) {
            Ok(e) => e,
            Err(e) => { out.push(Err(e)); return; }
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => { out.push(Err(e)); continue; }
            };
            let path = entry.path();
            if path.is_dir() { visit(&path, out); } else { out.push(Ok(path)); }
        }
    }
    visit(root.as_ref(), &mut out);
    out
}

#[test]
fn no_upward_layer_dependencies() {
    let edges = collect_edges();
    let mut violations: Vec<(String, String, Layer, Layer)> = Vec::new();
    for (fr, deps) in &edges {
        let Some(lf) = layer_of(fr) else { continue; };
        for to in deps {
            let Some(lt) = layer_of(to) else { continue; };  // outside-project dep — skip
            // Upward = target strictly higher than source.
            // T → non-T is a different bug (test util leaking into production).
            // T (test_util) is `cfg(test)`-only — it may use any production
            // module, since the symbols it references are stripped from
            // release binaries. The interesting check is upward between
            // production layers, with a one-step carve-out: tools/tool_runtime/config
            // (L3) reach into api (L4) for the Vendor type and its Policy
            // enums — that's the policy boundary, not a layering violation.
            let lf_n = lf.rank();
            let lt_n = lt.rank();
            let upward = lt_n > lf_n
                && lf != Layer::L5
                && lf != Layer::T
                && !(lf == Layer::L3 && lt == Layer::L4);
            if upward {
                violations.push((fr.clone(), to.clone(), lf, lt));
            }
        }
    }
    if !violations.is_empty() {
        let msg: Vec<String> = violations.iter().map(|(fr, to, lf, lt)| {
            format!("  {:?}/{:?} -> {:?}/{:?}", lf, fr, lt, to)
        }).collect();
        panic!("architecture violations ({}):\n{}", violations.len(), msg.join("\n"));
    }
}

#[test]
fn every_module_is_in_a_known_layer() {
    let edges = collect_edges();
    let unknown: Vec<_> = edges.keys().filter(|m| layer_of(m).is_none()).collect();
    assert!(unknown.is_empty(), "unknown module names: {unknown:?}");
}
