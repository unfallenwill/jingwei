// Architecture test: enforce jingwei's layering rules. Pure stdlib —
// walks `src/**/*.rs`, resolves every crate-internal import to the module
// that owns it, builds a directed graph, and asserts no upward edge.
//
// Layering (lower → higher, deps go downward only):
//
//   L0  ui (the port), core (mod, ir, session, turn, error, ids), format
//                                                  — pure data + state machines + the display port
//   L1  cancel, file_io, core::agents_md, core::reminder
//                                                  — infra primitives
//   L2  core::context, core::ports, core::prompt
//                                                  — per-agent state + the ports
//   L3  config (+ settings/login), tools (+ edit/ledger/runtime), mcp + submodules
//                                                  — the user's wishes and the tool tier
//   L4  api (+ api::minimax/zai/deepseek)          — vendor + transport
//   L5  lib, main, agent (+ runner), ui::plain, ui::tui (+ model/update/view)
//                                                  — composition root + the loop + the port's adapters
//   T   test_util                                  — test-only helpers
//
// Rules:
//   - L0..L5 may not import from any higher-numbered layer (no upward).
//   - L5 may import anything.
//   - T is test-only: it may import anything (its symbols live behind
//     `cfg(test)`), and nothing may import it in production.
//
// The resolver: every `use` item is expanded (brace groups, nested
// groups, `self`, renames, globs), every inline `crate::…` path counts
// as an edge, and a path resolves to the *longest known module prefix* —
// `crate::core::context` is the module `core::context`, not `core`, and
// the rest of that path is an item inside it. A re-export resolves to
// the module that defines the item, so `crate::Error` is the module
// that holds `Error`, never a module named "Error".
//
// The honest limits: the net sees `use` declarations and inline `crate::…`
// paths — not `pub use` chains through intermediate modules (one level is
// followed), not macro-generated code, and not paths that do not start
// with `crate::`, except a relative `use` that names a child of the
// module it is written in.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layer { L0, L1, L2, L3, L4, L5, T }

impl Layer {
    /// Lower layers (smaller number) may not import from higher-numbered ones.
    /// T (test) is a special case — it lives behind `cfg(test)`, so any
    /// production-code import of it is a bug.
    fn rank(self) -> u8 {
        match self {
            Self::L0 => 0, Self::L1 => 1, Self::L2 => 2, Self::L3 => 3,
            Self::L4 => 4, Self::L5 => 5, Self::T => 99,
        }
    }
}

/// The layer table: one flat list, so a new module is one row and the
/// resolver doubles the table as its known-module list.
///
/// A row for a module that does not exist yet is fine — the known-layer
/// test walks the tree, not the table — and is how a module's layer is
/// declared before the module lands.
const LAYERS: &[(&str, Layer)] = &[
    // L0 — pure data, state machines, and the display port.
    ("ui", Layer::L0),                 // the port: Msg, Usage, Show
    ("core", Layer::L0),               // the cluster's root
    ("core::ir", Layer::L0),           // the typed internal shape
    ("core::session", Layer::L0),      // the JSON-lines transcript
    ("core::turn", Layer::L0),         // the per-run state machine
    ("core::error", Layer::L0),        // Error and Result's home
    ("core::ids", Layer::L0),          // the agent identity: who is running
    ("format", Layer::L0),             // string measurement; it imports
                                        // nothing, and the display port (L0)
                                        // and the session (L0) both reach it
    // L1 — infra primitives.
    ("cancel", Layer::L1),
    ("file_io", Layer::L1),
    ("core::agents_md", Layer::L1),    // walks the FS
    ("core::reminder", Layer::L1),
    // L2 — per-agent state and the ports.
    ("core::context", Layer::L2),      // the per-session model state: the
                                        // system text, the reminders, the
                                        // tool list, the transcript, the
                                        // budget. It takes the tool list
                                        // from the router, so it no longer
                                        // reaches into the tool tier
    ("core::ports", Layer::L2),        // the tool boundary
    ("core::prompt", Layer::L2),       // the prompt's own module
    // L3 — the tool/config tier. May reach into api (L4); see the
    // carve-out in `upward_edges`.
    ("config", Layer::L3),
    ("config::settings", Layer::L3),
    ("config::login", Layer::L3),
    ("tools", Layer::L3),
    ("tools::edit", Layer::L3),
    ("tools::ledger", Layer::L3),
    ("tools::runtime", Layer::L3),
    ("mcp", Layer::L3),
    ("mcp::wire", Layer::L3),
    ("mcp::health", Layer::L3),
    ("mcp::config", Layer::L3),
    ("mcp::client", Layer::L3),
    ("mcp::stdio", Layer::L3),
    ("mcp::http", Layer::L3),
    ("mcp::inner", Layer::L3),
    ("mcp::stub", Layer::L3),
    // L4 — the model boundary.
    ("api", Layer::L4),
    ("api::minimax", Layer::L4),
    ("api::zai", Layer::L4),
    ("api::deepseek", Layer::L4),
    // L5 — composition and the frontends.
    ("lib", Layer::L5),
    ("main", Layer::L5),
    ("agent", Layer::L5),               // the agent as a value
    ("agent::runner", Layer::L5),       // the turn loop
    ("ui::plain", Layer::L5),
    ("ui::tui", Layer::L5),
    ("ui::tui::model", Layer::L5),
    ("ui::tui::update", Layer::L5),
    ("ui::tui::view", Layer::L5),
    // T — test-only helpers.
    ("test_util", Layer::T),
];

fn layer_of(mod_name: &str) -> Option<Layer> {
    LAYERS.iter().find(|(m, _)| *m == mod_name).map(|(_, l)| *l)
}

/// The crate root's module name, and the scope a `crate::…` path starts
/// in. The root is its own scope: `crate::api` is a child of the root,
/// not of a module called "lib".
const ROOT_MODULE: &str = "lib";
const ROOT_SCOPE: &str = "";

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

/// A module's path (`core::ir`) and its source.
type Modules = BTreeMap<String, String>;

/// `scope` → `item name` → the module that defines the item.
///
/// The crate root's scope is the empty string: a `crate::…` path starts
/// there, and the root's own items — `Error`, `Result`, and the private
/// `use`s that make `crate::Error` nameable — are its aliases.
type Aliases = HashMap<String, HashMap<String, String>>;

/// The scope a module's own aliases live in: the root's aliases are the
/// root scope's aliases.
fn scope_key(module: &str) -> &str {
    if module == ROOT_MODULE { ROOT_SCOPE } else { module }
}

// ---- the source scanner ----------------------------------------------------

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

fn read_ident(b: &[u8], at: usize) -> Option<String> {
    if at >= b.len() || !is_ident_byte(b[at]) { return None; }
    let mut end = at;
    while end < b.len() && is_ident_byte(b[end]) { end += 1; }
    Some(String::from_utf8_lossy(&b[at..end]).into_owned())
}

/// Blank a span, keeping the newlines so the text keeps its shape.
fn blank(out: &mut [u8], from: usize, to: usize) {
    let to = to.min(out.len());
    for i in from..to {
        if out[i] != b'\n' { out[i] = b' '; }
    }
}

/// If a string literal starts at `i`, the index just past its closing
/// quote. Handles `"…"`, `r"…"`, `r#"…"#` and the `b`/`br` prefixes.
fn string_literal_end(b: &[u8], i: usize) -> Option<usize> {
    if b[i] == b'"' {
        let mut j = i + 1;
        while j < b.len() {
            match b[j] {
                b'\\' => j += 2,
                b'"' => return Some(j + 1),
                _ => j += 1,
            }
        }
        return Some(b.len());
    }
    // A raw string: r"…", r#"…"#, r##"…"##
    let mut j = i + 1;
    let mut hashes = 0;
    while j < b.len() && b[j] == b'#' { hashes += 1; j += 1; }
    if j >= b.len() || b[j] != b'"' { return None; }
    j += 1;
    while j < b.len() {
        if b[j] == b'"' {
            let mut k = j + 1;
            let mut h = 0;
            while h < hashes && k < b.len() && b[k] == b'#' { h += 1; k += 1; }
            if h == hashes { return Some(k); }
        }
        j += 1;
    }
    Some(b.len())
}

/// If a char literal starts at `i`, the index just past its closing quote.
///
/// A lifetime (`'a`) is not a char literal: it is a quote, one name, and
/// no closing quote. That is the difference this looks for.
fn char_literal_end(b: &[u8], i: usize) -> Option<usize> {
    let mut j = i + 1;
    if j >= b.len() { return None; }
    if b[j] == b'\\' {
        j += 1;
        if j < b.len() && b[j] == b'u' {
            j += 1;
            if j < b.len() && b[j] == b'{' {
                while j < b.len() && b[j] != b'}' { j += 1; }
                if j < b.len() { j += 1; }
            }
        } else {
            j += 1;
        }
    } else {
        j += 1;
        while j < b.len() && (b[j] & 0xC0) == 0x80 { j += 1; }
    }
    if j < b.len() && b[j] == b'\'' { Some(j + 1) } else { None }
}

/// Blank out comments, string literals and char literals, keeping the
/// text's length. A mention of `crate::api::VENDORS` inside a comment or
/// a string is not an import.
fn blank_comments_and_literals(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' { out[i] = b' '; i += 1; }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let mut depth = 0usize;
            while i < b.len() {
                if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                    depth += 1; out[i] = b' '; out[i + 1] = b' '; i += 2; continue;
                }
                if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
                    depth -= 1; out[i] = b' '; out[i + 1] = b' '; i += 2;
                    if depth == 0 { break; }
                    continue;
                }
                if b[i] != b'\n' { out[i] = b' '; }
                i += 1;
            }
        } else if b[i] == b'r' && i + 1 < b.len() && (b[i + 1] == b'"' || b[i + 1] == b'#') {
            match string_literal_end(b, i) {
                Some(end) => { blank(&mut out, i, end); i = end; }
                None => i += 1,
            }
        } else if b[i] == b'"' {
            let end = string_literal_end(b, i).unwrap_or(b.len());
            blank(&mut out, i, end);
            i = end;
        } else if b[i] == b'\'' {
            match char_literal_end(b, i) {
                Some(end) => { blank(&mut out, i, end); i = end; }
                None => i += 1,
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

/// The end of the `#[cfg(test)]` attribute at `i`, if that is what it is.
fn cfg_test_attr_end(b: &[u8], i: usize) -> Option<usize> {
    if !b[i..].starts_with(b"#[") { return None; }
    let mut j = i + 2;
    while j < b.len() && b[j] != b']' { j += 1; }
    if j >= b.len() { return None; }
    let norm: String = b[i + 2..j].iter()
        .filter(|c| !c.is_ascii_whitespace())
        .map(|c| *c as char)
        .collect();
    if norm == "cfg(test)" { Some(j + 1) } else { None }
}

/// The index just past the `}` that closes the `{` at `open`.
fn matching_brace(b: &[u8], open: usize) -> usize {
    let mut depth = 0usize;
    let mut i = open;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => { depth -= 1; if depth == 0 { return i + 1; } }
            _ => {}
        }
        i += 1;
    }
    b.len()
}

/// The end of the item that starts at `at`: its body, or its `;`.
fn item_end(b: &[u8], at: usize) -> usize {
    let mut j = at;
    while j < b.len() {
        match b[j] {
            b'{' => return matching_brace(b, j),
            b';' => return j + 1,
            _ => j += 1,
        }
    }
    b.len()
}

/// Blank the test-only items: `#[cfg(test)] mod … { … }`,
/// `#[cfg(test)] mod …;` and `#[cfg(test)] fn … { … }`. Their imports
/// never reach the production binary, so they are not edges.
fn strip_cfg_test_items(text: &str) -> String {
    let b = text.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'#' {
            if let Some(attr_end) = cfg_test_attr_end(b, i) {
                let mut j = attr_end;
                while j < b.len() && (b[j] as char).is_whitespace() { j += 1; }
                let is_mod = b[j..].starts_with(b"mod")
                    && (j + 3 >= b.len() || !is_ident_byte(b[j + 3]));
                if is_mod {
                    j += 3;
                    while j < b.len() && (b[j] as char).is_whitespace() { j += 1; }
                    while j < b.len() && is_ident_byte(b[j]) { j += 1; }
                    while j < b.len() && (b[j] as char).is_whitespace() { j += 1; }
                    if j < b.len() && b[j] == b'{' {
                        let end = matching_brace(b, j);
                        blank(&mut out, i, end);
                        i = end;
                        continue;
                    }
                }
                let end = item_end(b, j);
                blank(&mut out, i, end);
                i = end;
                continue;
            }
        }
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

// ---- the use tree ----------------------------------------------------------

/// One leaf of a `use` tree.
#[derive(Debug, Clone)]
struct Leaf {
    /// The path as written: `crate::…`, `self::…`, `super::…`, or a
    /// relative path that names a child of the current module. A glob
    /// carries the path of the module it is written in.
    path: Vec<String>,
    /// The name the item is bound to, when it is renamed.
    alias: Option<String>,
}

struct TreeParser<'a> { b: &'a [u8], i: usize }

impl<'a> TreeParser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_whitespace() { self.i += 1; }
    }

    fn ident(&mut self) -> Option<String> {
        let id = read_ident(self.b, self.i)?;
        self.i += id.len();
        Some(id)
    }

    /// A comma-separated list of trees.
    fn list(&mut self, prefix: &[String], in_group: bool) -> Vec<Leaf> {
        let mut out = Vec::new();
        loop {
            self.ws();
            if self.i >= self.b.len() { break; }
            let c = self.b[self.i];
            if c == b',' { self.i += 1; continue; }
            if c == b'}' || c == b';' { break; }
            let before = self.i;
            out.extend(self.tree(prefix, in_group));
            if self.i == before { break; }
            self.ws();
            if self.i < self.b.len() && self.b[self.i] == b',' { self.i += 1; continue; }
            break;
        }
        out
    }

    /// One tree: a path, a group, or a glob.
    fn tree(&mut self, prefix: &[String], in_group: bool) -> Vec<Leaf> {
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'*' {
            self.i += 1;
            let alias = self.opt_as();
            // A glob names the module it is written in, not an item inside
            // it: `use crate::api::*;` is an import of `api`.
            return vec![Leaf { path: prefix.to_vec(), alias }];
        }
        let Some(first) = self.ident() else { return Vec::new() };
        let mut segs: Vec<String> = prefix.to_vec();
        if in_group && first == "self" {
            // `use a::b::{self, c}` — `self` names the module `a::b`.
            let alias = self.opt_as();
            return vec![Leaf { path: segs, alias }];
        }
        segs.push(first);
        loop {
            self.ws();
            if self.i + 1 < self.b.len() && self.b[self.i] == b':' && self.b[self.i + 1] == b':' {
                self.i += 2;
                self.ws();
                if self.i < self.b.len() && self.b[self.i] == b'*' {
                    // A glob names the module it is written in, not an
                    // item inside it: `use crate::api::*;` imports `api`.
                    self.i += 1;
                    break;
                }
                let Some(next) = self.ident() else { break };
                segs.push(next);
                continue;
            }
            break;
        }
        self.ws();
        if self.i < self.b.len() && self.b[self.i] == b'{' {
            self.i += 1;
            let children = self.list(&segs, true);
            self.ws();
            if self.i < self.b.len() && self.b[self.i] == b'}' { self.i += 1; }
            return children;
        }
        let alias = self.opt_as();
        vec![Leaf { path: segs, alias }]
    }

    /// The `as name` rename, if there is one.
    fn opt_as(&mut self) -> Option<String> {
        self.ws();
        if self.b[self.i..].starts_with(b"as")
            && (self.i + 2 >= self.b.len() || !is_ident_byte(self.b[self.i + 2]))
        {
            self.i += 2;
            self.ws();
            return self.ident();
        }
        None
    }
}

/// The `use` items in `text`: the span of the keyword and the span of the
/// item that follows it.
fn find_use_items(text: &str) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'u' && b[i..].starts_with(b"use")
            && (i == 0 || !is_ident_byte(b[i - 1]))
            && (i + 3 >= b.len() || !is_ident_byte(b[i + 3]))
        {
            let mut p = TreeParser { b, i: i + 3 };
            p.list(&[], false);
            p.ws();
            let end = if p.i < b.len() && b[p.i] == b';' { p.i + 1 } else { p.i };
            out.push((i, end));
            i = end.max(i + 3);
            continue;
        }
        if is_ident_byte(b[i]) {
            while i < b.len() && is_ident_byte(b[i]) { i += 1; }
            continue;
        }
        i += 1;
    }
    out
}

/// The leaves of every `use` item in `text`.
fn use_leaves(text: &str) -> Vec<Leaf> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    for (start, _) in find_use_items(text) {
        let mut p = TreeParser { b, i: start + 3 };
        out.extend(p.list(&[], false));
    }
    out
}

/// The `crate::…` paths written in an expression or a type position — the
/// ones that are not part of a `use` item.
fn inline_paths(text: &str) -> Vec<Vec<String>> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'c' && b[i..].starts_with(b"crate")
            && (i == 0 || !is_ident_byte(b[i - 1]))
            && (i + 5 >= b.len() || !is_ident_byte(b[i + 5]))
            && i + 7 <= b.len() && &b[i + 5..i + 7] == b"::"
        {
            let mut j = i + 7;
            let mut segs = Vec::new();
            loop {
                let Some(id) = read_ident(b, j) else { break };
                j += id.len();
                segs.push(id);
                if j + 1 < b.len() && b[j] == b':' && b[j + 1] == b':' {
                    j += 2;
                    continue;
                }
                break;
            }
            if !segs.is_empty() { out.push(segs); }
            i = j.max(i + 7);
            continue;
        }
        if is_ident_byte(b[i]) {
            while i < b.len() && is_ident_byte(b[i]) { i += 1; }
            continue;
        }
        i += 1;
    }
    out
}

/// The items a module defines for the rest of the crate.
fn definitions(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'p' && b[i..].starts_with(b"pub")
            && (i == 0 || !is_ident_byte(b[i - 1]))
            && (i + 3 >= b.len() || !is_ident_byte(b[i + 3]))
        {
            let mut j = i + 3;
            while j < b.len() && (b[j] as char).is_whitespace() { j += 1; }
            if j < b.len() && b[j] == b'(' {
                let mut k = j + 1;
                while k < b.len() && b[k] != b')' { k += 1; }
                j = (k + 1).min(b.len());
                while j < b.len() && (b[j] as char).is_whitespace() { j += 1; }
            }
            let Some(kw) = read_ident(b, j) else { i += 1; continue };
            if matches!(kw.as_str(), "struct" | "enum" | "type" | "const" | "fn"
                | "trait" | "static" | "union" | "macro_rules")
            {
                let mut k = j + kw.len();
                while k < b.len() && (b[k] as char).is_whitespace() { k += 1; }
                if let Some(name) = read_ident(b, k) { out.push(name); }
            }
        }
        if is_ident_byte(b[i]) {
            while i < b.len() && is_ident_byte(b[i]) { i += 1; }
            continue;
        }
        i += 1;
    }
    out
}

// ---- the resolver ----------------------------------------------------------

/// The longest known module prefix of `path`, and the tail that is left
/// over — the items inside that module.
///
/// Only the known modules are consulted: this is the pass that builds the
/// alias map, so it must not depend on the map it is building.
fn known_prefix<'a>(
    path: &'a [String],
    base: &str,
    known: &HashSet<String>,
) -> (String, &'a [String]) {
    let mut segs = path;
    let mut cur = base.to_string();
    if let Some(first) = segs.first() {
        match first.as_str() {
            "crate" => { cur = String::new(); segs = &segs[1..]; }
            "self" => { segs = &segs[1..]; }
            "super" => { cur = parent_of(base); segs = &segs[1..]; }
            _ => {}
        }
    }
    for (i, seg) in segs.iter().enumerate() {
        let candidate = if cur.is_empty() { seg.clone() } else { format!("{cur}::{seg}") };
        if known.contains(&candidate) {
            cur = candidate;
            continue;
        }
        // `seg` is an item inside `cur`.
        return (cur, &segs[i..]);
    }
    (cur, &segs[segs.len()..])
}

/// The parent of a module path: `core::ir` → `core`, `core` → the root.
fn parent_of(module: &str) -> String {
    match module.rfind("::") {
        Some(at) => module[..at].to_string(),
        None => String::new(),
    }
}

/// The module that owns the thing `path` names, or `None` when the path
/// leaves the crate (an external crate, or a name the net does not know).
///
/// The longest known module prefix wins: `crate::core::context` is the
/// module `core::context`, and the rest of the path is an item inside it.
/// An item is then resolved through the alias map, so a re-export points
/// at the module that defines it.
fn owner_of(
    path: &[String],
    base: &str,
    known: &HashSet<String>,
    aliases: &Aliases,
) -> Option<String> {
    let (cur, tail) = known_prefix(path, base, known);
    if tail.is_empty() {
        return if cur.is_empty() { None } else { Some(cur) };
    }
    let name = &tail[0];
    let scope = scope_key(&cur);
    if let Some(home) = aliases.get(scope).and_then(|m| m.get(name)) {
        return Some(home.clone());
    }
    if cur.is_empty() { None } else { Some(cur) }
}

/// Build the alias map: for every module, the names it offers and the
/// module that defines them.
fn build_aliases(mods: &Modules, known: &HashSet<String>) -> Aliases {
    let mut aliases: Aliases = HashMap::new();
    for (module, text) in mods {
        let scope = scope_key(module).to_string();
        // Comments and string literals are not declarations, and a
        // test-only item is not part of the production surface.
        let stripped = strip_cfg_test_items(&blank_comments_and_literals(text));
        let entry = aliases.entry(scope.clone()).or_default();
        // What the module defines itself.
        for name in definitions(&stripped) {
            entry.insert(name, module.clone());
        }
        // What it re-exports: `use` and `pub use` alike — a private `use`
        // in the crate root is still nameable as `crate::the_name`.
        for leaf in use_leaves(&stripped) {
            let (home, tail) = known_prefix(&leaf.path, &scope, known);
            if home.is_empty() { continue; }
            let name = match tail.first() {
                Some(first) => leaf.alias.clone().unwrap_or_else(|| first.clone()),
                None => leaf.path.last().cloned().unwrap_or_default(),
            };
            if name.is_empty() { continue; }
            entry.insert(name, home);
        }
    }
    // Follow one level of `pub use` chain: if the module a name was found
    // in re-exports it from somewhere else, the name really lives there.
    let updates: Vec<(String, String, String)> = {
        let mut v = Vec::new();
        for (scope, names) in &aliases {
            for (name, home) in names {
                if let Some(deeper) = aliases.get(home).and_then(|m| m.get(name)) {
                    if deeper != home { v.push((scope.clone(), name.clone(), deeper.clone())); }
                }
            }
        }
        v
    };
    for (scope, name, deeper) in updates {
        if let Some(m) = aliases.get_mut(&scope) { m.insert(name, deeper); }
    }
    aliases
}

/// The module → the modules it imports, for one tree.
fn edges_of(mods: &Modules, known: &HashSet<String>, aliases: &Aliases) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for (module, text) in mods {
        let scope = scope_key(module).to_string();
        let stripped = strip_cfg_test_items(&blank_comments_and_literals(text));
        let mut deps: Vec<String> = Vec::new();

        // The `use` items.
        for leaf in use_leaves(&stripped) {
            if let Some(owner) = owner_of(&leaf.path, &scope, known, aliases) {
                if owner != *module { deps.push(owner); }
            }
        }

        // The inline `crate::…` paths, with the `use` items blanked out so
        // they are not counted twice.
        let mut bytes = stripped.clone().into_bytes();
        for (start, end) in find_use_items(&stripped) {
            blank(&mut bytes, start, end);
        }
        let scan = String::from_utf8(bytes).unwrap_or_default();
        for segs in inline_paths(&scan) {
            if let Some(owner) = owner_of(&segs, ROOT_SCOPE, known, aliases) {
                if owner != *module { deps.push(owner); }
            }
        }

        deps.sort();
        deps.dedup();
        out.insert(module.clone(), deps);
    }
    out
}

/// The upward edges in a tree.
fn upward_edges(edges: &HashMap<String, Vec<String>>) -> Vec<(String, String, Layer, Layer)> {
    let mut violations = Vec::new();
    for (fr, deps) in edges {
        let Some(lf) = layer_of(fr) else { continue };
        for to in deps {
            let Some(lt) = layer_of(to) else { continue };  // outside-project dep — skip
            // Upward = target strictly higher than source.
            // T → non-T is a different bug (test util leaking into production).
            // T (test_util) is `cfg(test)`-only — it may use any production
            // module, since the symbols it references are stripped from
            // release binaries. The interesting check is upward between
            // production layers, with one carve-out: L3 may reach L4, and
            // only for two reasons — the vendor table and the policy enums.
            // `config`, `config::settings` and `config::login` need
            // `api::Protocol`, `api::VENDORS`, `api::CacheMode`,
            // `api::Thinking` and `api::Effort`; `tools::runtime` needs
            // `api::blocking`, which is a *function*, not a type. Nothing
            // else in L3 may name anything in L4.
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
    violations
}

/// Walk src/ and build the module → its imported modules map.
fn collect_edges() -> HashMap<String, Vec<String>> {
    let mods = collect_modules();
    let known = known_modules(&mods);
    let aliases = build_aliases(&mods, &known);
    edges_of(&mods, &known, &aliases)
}

/// Every module in the tree, with its source.
fn collect_modules() -> Modules {
    let mut out = Modules::new();
    for entry in walkdir("src") {
        let path = match entry { Ok(p) => p, Err(_) => continue };
        if path.extension().and_then(|e| e.to_str()) != Some("rs") { continue; }
        let Some(mod_name) = file_to_module(&path) else { continue };
        let Ok(text) = fs::read_to_string(&path) else { continue };
        out.insert(mod_name, text);
    }
    out
}

/// The modules the resolver knows: the ones on disk, plus the ones the
/// layer table declares.
fn known_modules(mods: &Modules) -> HashSet<String> {
    let mut known: HashSet<String> = mods.keys().cloned().collect();
    known.extend(LAYERS.iter().map(|(m, _)| (*m).to_string()));
    known
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

// ---- the gate --------------------------------------------------------------

#[test]
fn no_upward_layer_dependencies() {
    let edges = collect_edges();
    let violations = upward_edges(&edges);
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

// ---- the resolver, on a synthetic tree -------------------------------------
//
// The fixtures are string literals, not files: the real tree stays clean,
// and the resolver is exercised against trees that would not compile.

/// A synthetic tree: a module's source, and the edges it produces.
fn fixture(source: &str) -> HashMap<String, Vec<String>> {
    let mut mods = Modules::new();
    mods.insert("api".to_string(), "pub(crate) const VENDORS: &[&str] = &[];\n".to_string());
    mods.insert("core::ir".to_string(), source.to_string());
    let known = known_modules(&mods);
    let aliases = build_aliases(&mods, &known);
    edges_of(&mods, &known, &aliases)
}

/// The modules a fixture imports, sorted.
fn targets(source: &str) -> Vec<String> {
    let mut t = fixture(source).get("core::ir").cloned().unwrap_or_default();
    t.sort();
    t
}

#[test]
fn a_violation_behind_a_brace_group_is_seen() {
    // `use crate::{api::VENDORS};` — the old resolver read the first `::`
    // segment and saw "{api", which is not a module: the edge was lost.
    assert_eq!(targets("use crate::{api::VENDORS};\n"), vec!["api"]);
    // `use crate::api::{VENDORS};` — the same import, the other way round.
    assert_eq!(targets("use crate::api::{VENDORS};\n"), vec!["api"]);
}

#[test]
fn a_violation_in_an_inline_path_is_seen() {
    assert_eq!(targets("fn f() { let v = crate::api::VENDORS; }\n"), vec!["api"]);
    assert_eq!(targets("fn f(v: crate::api::Protocol) {}\n"), vec!["api"]);
}

#[test]
fn a_violation_in_a_function_body_is_seen() {
    assert_eq!(targets("fn f() { use crate::api::VENDORS; }\n"), vec!["api"]);
}

#[test]
fn a_mention_in_a_comment_or_a_string_is_not_an_edge() {
    assert!(targets("// see crate::api::VENDORS\n").is_empty());
    assert!(targets("/* crate::api::VENDORS */\n").is_empty());
    assert!(targets("fn f() { let s = \"crate::api::VENDORS\"; }\n").is_empty());
    assert!(targets("fn f() { let s = r#\"crate::api::VENDORS\"#; }\n").is_empty());
    // A lifetime is not a char literal, and a char literal is not a path.
    assert!(targets("fn f<'a>(x: &'a str) { let c = 'x'; }\n").is_empty());
}

#[test]
fn a_declaration_mentioned_in_a_comment_is_not_a_declaration() {
    let mut mods = Modules::new();
    mods.insert("api".to_string(), "pub(crate) const VENDORS: &[&str] = &[];\n".to_string());
    mods.insert(
        "core::ir".to_string(),
        "/// pub use crate::api::VENDORS;\n/// pub enum VENDORS { }\n".to_string(),
    );
    let known = known_modules(&mods);
    let aliases = build_aliases(&mods, &known);
    let offered = aliases.get("core::ir").cloned().unwrap_or_default();
    assert!(!offered.contains_key("VENDORS"), "a doc comment is not a declaration");
}

#[test]
fn a_submodule_resolves_by_the_longest_known_prefix() {
    // `core::ports` is the module, not `core` — the tail is an item in it.
    assert_eq!(targets("use crate::core::ports::ToolProvider;\n"), vec!["core::ports"]);
    assert_eq!(targets("use crate::core::context::Context;\n"), vec!["core::context"]);
}

#[test]
fn nested_brace_groups_expand() {
    let t = targets("use crate::api::{minimax::Wire, zai::Wire};\n");
    assert_eq!(t, vec!["api::minimax", "api::zai"]);
}

#[test]
fn a_glob_names_the_module_itself() {
    assert_eq!(targets("use crate::api::*;\n"), vec!["api"]);
}

#[test]
fn a_root_re_export_resolves_to_the_module_that_defines_it() {
    let mut mods = Modules::new();
    mods.insert("api".to_string(), "pub(crate) const VENDORS: &[&str] = &[];\n".to_string());
    mods.insert("lib".to_string(), "pub use crate::api::VENDORS;\n".to_string());
    mods.insert("core::ir".to_string(), "use crate::VENDORS;\n".to_string());
    let known = known_modules(&mods);
    let aliases = build_aliases(&mods, &known);
    let edges = edges_of(&mods, &known, &aliases);
    assert_eq!(edges.get("core::ir"), Some(&vec!["api".to_string()]));
    // The name is not a module of its own.
    assert!(layer_of("VENDORS").is_none());
}

#[test]
fn error_and_result_resolve_to_their_home_not_to_a_module_named_error() {
    // The real shape: `core::error` defines the type, the root re-exports
    // it, and a caller names it as `crate::Error`. The net must see the
    // edge to the module that *defines* it — never to the root, and never
    // to a module named "Error".
    let mut mods = Modules::new();
    mods.insert("core::error".to_string(), concat!(
        "pub enum Error { Interrupted }\n",
        "pub type Result<T> = std::result::Result<T, Error>;\n",
    ).to_string());
    mods.insert("lib".to_string(), "pub use crate::core::error::{Error, Result};\n".to_string());
    mods.insert("core::ir".to_string(), "use crate::{Error, Result};\n".to_string());
    let known = known_modules(&mods);
    let aliases = build_aliases(&mods, &known);
    let edges = edges_of(&mods, &known, &aliases);
    assert_eq!(edges.get("core::ir"), Some(&vec!["core::error".to_string()]));
    // There is no module called "Error" and no module called "Result".
    assert!(layer_of("Error").is_none());
    assert!(layer_of("Result").is_none());
}

#[test]
fn the_real_tree_has_no_upward_edge() {
    // The synthetic fixtures above prove the resolver sees what the old
    // one missed; this one holds the real tree to the same rule.
    let edges = collect_edges();
    let violations = upward_edges(&edges);
    assert!(violations.is_empty(), "architecture violations: {violations:?}");
}
