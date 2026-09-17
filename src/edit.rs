// ---- edit_file: one transaction, however many hunks ------------------------
//
// A search/replace tool has two failure modes that each cost a whole turn:
// the model's `old` did not match byte-for-byte (a tab where it wrote spaces,
// a line ending it did not see), or it matched more than once and the tool
// refused — whereupon the only way forward is to paste more context, byte for
// byte, and try again. Both are the harness failing to negotiate, not the
// model failing to understand: it knows what to change and is being asked to
// prove it by reproducing text it already read.
//
// So this one negotiates. An exact match wins, always. Failing that, whole
// lines are aligned ignoring indentation, and `new` is shifted to the
// indentation the file actually uses. A repeated `old` is not refused
// outright but *named*: the line numbers of every occurrence, plus the two
// ways out — widen `old` with a line of context, or ask for every occurrence
// with replace_all. The write is temp+rename (as session.rs has always done)
// so a crash mid-edit cannot leave a truncated file behind, and the result
// carries the line numbers and a diff, so neither the model nor the user has
// to re-read the file to learn what happened.
//
// The tolerance is bounded on purpose: the line-wise pass must match *every*
// line of `old` and must land in exactly one place, so it rescues whitespace
// drift without excusing a wrong target. Ambiguity is reported, never guessed.
//
// And it takes a batch. `edits` carries as many pairs as the job needs —
// a rename, a repeated idiom with three different shapes, a small refactor —
// and they are resolved against one buffer, in order, so a pair may edit what
// an earlier pair wrote. The transaction is what makes it safe: any pair that
// cannot be placed takes the whole call down with the pair's number and its
// reason, and nothing is written. One call, one write, one diff — instead of
// K calls and K chances to fall back to the whole-file overwrite, which is the
// blunt tool this one exists to keep out of the conversation.
//
// The read ledger that backs the "file changed on disk" refusal lives in the
// crate root for now (it is shared with `read_file` and `write_file`); once
// the tool registry moves to its own module this file will import it from
// there.

use crate::ledger::{ledger_note, stale};
use serde_json::Value;

/// The most diff lines one replacement may print before the report falls back
/// to line counts, and the most a whole call may print. A diff is for
/// orientation; the file is still on disk for the rest.
const DIFF_ROWS: usize = 12;
const DIFF_TOTAL: usize = 40;

/// The entry point called by the tool registry: parse `old`/`new` vs `edits`,
/// then hand the batch to `apply_edits`. Two ways to say the same thing and
/// saying both at once is a mistake worth naming rather than silently
/// preferring one.
pub(crate) fn edit_tool(i: &Value) -> String {
    let p = i["path"].as_str().unwrap_or("");
    let replace_all = i["replace_all"].as_bool().unwrap_or(false);
    let edits: Vec<(String, String)> = match &i["edits"] {
        Value::Array(_) if i["old"].is_string() =>
            return format!("error: send either `old`/`new` or `edits`, not both — `edits` already \
                            carries the pairs to apply (in {p})"),
        Value::Array(a) if a.is_empty() =>
            return format!("error: `edits` is empty — send the pairs to apply, or use `old`/`new` (in {p})"),
        Value::Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for (n, e) in a.iter().enumerate() {
                match (e["old"].as_str(), e["new"].as_str()) {
                    (Some(old), Some(new)) => out.push((old.to_string(), new.to_string())),
                    _ => return format!("error: edits[{}] needs `old` and `new` strings (in {p})", n + 1),
                }
            }
            out
        }
        // One pair, and a missing `new` is an empty string: deleting text is
        // an edit too.
        _ => match i["old"].as_str() {
            Some(old) => vec![(old.to_string(), i["new"].as_str().unwrap_or("").to_string())],
            None => return format!("error: send `old`/`new` or `edits` — there is nothing to apply (in {p})"),
        },
    };
    apply_edits(p, &edits, replace_all)
}

/// The whole edit tool: read once, resolve every pair against the buffer (each
/// seeing what the ones before it wrote), write once. All or nothing — a batch
/// that fails at pair 3 writes nothing at all and says which pair failed and
/// why, so the model re-plans instead of discovering a half-applied file.
fn apply_edits(p: &str, edits: &[(String, String)], replace_all: bool) -> String {
    let content = match std::fs::read_to_string(p) {
        Ok(c) => c,
        Err(e) => return format!("error: {e} (in {p})"),
    };
    // The bytes just read are the ones the edit is written against — if they
    // are not the bytes we last saw, this file moved under us and the edit is
    // not ours to make.
    if let Some(e) = stale(p) { return e; }
    let mut buffer = content;
    let mut marks: Vec<(usize, usize)> = Vec::new();
    let mut body = String::new();
    let mut shown = 0;
    for (n, (old, new)) in edits.iter().enumerate() {
        let n = n + 1;
        let resolved = match resolve(&buffer, old, new, replace_all) {
            Ok(r) => r,
            Err(e) => return format!("error: edit {n}/{}: {e} (in {p}; nothing was written)",
                edits.len()),
        };
        let spans: Vec<Span> = resolved.units.iter().map(|(s, e, text)| Span {
            start: *s, end: *e, line: line_at(&buffer, *s), text: text.clone(),
        }).collect();
        let updated = splice(&buffer, &resolved.units);
        if resolved.fuzzy {
            body.push_str("\n  (matched line by line, ignoring indentation — `new` was re-indented to \
                           where those lines sit; nothing else about their whitespace changed)");
        }
        let prefix = if edits.len() > 1 { format!("edit {n} ") } else { String::new() };
        body.push_str(&edit_diff(&buffer, &updated, &spans, &prefix, &mut shown));
        for s in &spans { marks.push((n, s.line)); }
        buffer = updated;
    }
    if let Err(e) = crate::file_io::write_atomic(p, &buffer) { return format!("error: {e} (in {p})"); }
    ledger_note(p, buffer.as_bytes());
    let mut out = edit_header(p, &marks, edits.len());
    out.push_str(&body);
    out
}

/// The resolution ladder for one pair: an exact match wins; failing that whole
/// lines are aligned ignoring indentation; a repeated `old` is refused *with
/// its line numbers* unless `replace_all` says every occurrence is meant. This
/// is the only place ambiguity is allowed to stop an edit.
fn resolve(content: &str, old: &str, new: &str, replace_all: bool) -> std::result::Result<Resolution, String> {
    if old.is_empty() {
        return Err("`old` is empty — there is nothing to find; use write_file to create or replace a \
                    file wholesale".into());
    }
    let exact: Vec<(usize, usize)> = content.match_indices(old).map(|(i, m)| (i, i + m.len())).collect();
    let (units, fuzzy) = match exact.len() {
        1 => (vec![(exact[0].0, exact[0].1, new.to_string())], false),
        0 => (fuzzy_hits(content, old, new), true),
        _ if replace_all => (exact.iter().map(|&(s, e)| (s, e, new.to_string())).collect(), false),
        _ => return Err(ambiguous(content, &exact)),
    };
    if units.is_empty() { return Err(NOT_FOUND.into()); }
    // The tolerant pass found more than one place: that is exactly the
    // ambiguity replace_all exists to resolve, so it gets the same message.
    if units.len() > 1 && !replace_all && fuzzy {
        let ranges: Vec<(usize, usize)> = units.iter().map(|(s, e, _)| (*s, *e)).collect();
        return Err(ambiguous(content, &ranges));
    }
    Ok(Resolution { units, fuzzy })
}

/// What one `old`/`new` pair resolved to against the buffer it is applied to:
/// the byte ranges to replace (with the text that goes in), and whether the
/// tolerant pass is what found them.
struct Resolution {
    units: Vec<(usize, usize, String)>,
    fuzzy: bool,
}

/// One replacement, resolved against the file as it was: the byte range it
/// covers, the line it starts on, and what that range becomes.
struct Span {
    start: usize,
    end: usize,
    /// 1-based, in the file *before* the edit.
    line: usize,
    text: String,
}

/// The way out of an ambiguous `old`, in the model's own terms: where the
/// occurrences are, and the two moves that resolve it. A bare "must be
/// unique" makes the model guess; the line numbers let it widen the right
/// side in one shot.
fn ambiguous(content: &str, ranges: &[(usize, usize)]) -> String {
    let mut lines: Vec<String> = ranges.iter().take(8).map(|&(s, _)| line_at(content, s).to_string()).collect();
    if ranges.len() > lines.len() { lines.push("…".into()); }
    format!("`old` appears {} times (lines {}) — widen `old` with a line of context around the \
             one you mean, or set replace_all: true to replace all {}",
            ranges.len(), lines.join(", "), ranges.len())
}

const NOT_FOUND: &str = "`old` not found — not as written, and not line by line with indentation \
                         ignored either. Re-read the file and copy the lines you mean verbatim";

/// Splice replacements into the content, in order.
fn splice(content: &str, units: &[(usize, usize, String)]) -> String {
    let mut out = String::with_capacity(content.len());
    let mut at = 0;
    for (start, end, text) in units {
        out.push_str(&content[at..*start]);
        out.push_str(text);
        at = *end;
    }
    out.push_str(&content[at..]);
    out
}

/// The header: how many replacements landed, and where. A batch reports each
/// edit's own lines — the buffer that edit saw, in order — and says so, rather
/// than leaving the model to work out whose numbering it is reading.
fn edit_header(p: &str, marks: &[(usize, usize)], edits: usize) -> String {
    if edits == 1 {
        return if marks.len() == 1 {
            format!("ok: edited {p} — 1 replacement at line {}", marks[0].1)
        } else {
            let lines: Vec<String> = marks.iter().map(|m| m.1.to_string()).collect();
            format!("ok: edited {p} — {} replacements at lines {} (numbering from before the edit)",
                marks.len(), lines.join(", "))
        };
    }
    let mut per: Vec<String> = Vec::new();
    let mut i = 0;
    while i < marks.len() {
        let n = marks[i].0;
        let mut lines = Vec::new();
        while i < marks.len() && marks[i].0 == n { lines.push(marks[i].1.to_string()); i += 1; }
        per.push(format!("edit {n} at line{} {}", if lines.len() > 1 { "s" } else { "" }, lines.join(", ")));
    }
    format!("ok: edited {p} — {edits} edits, {} replacements; lines are as of each edit, in order\n  {}",
        marks.len(), per.join("; "))
}

/// The diff for one edit: the whole lines it landed on, before and after — so
/// the change reads in its place, indentation and all, rather than as the bare
/// fragment that happened to be matched. Capped twice over: past DIFF_ROWS for
/// one replacement, or DIFF_TOTAL for the call, the report falls back to counts
/// — a diff is for orientation, the file itself is one tool call away.
fn edit_diff(before: &str, after: &str, spans: &[Span], prefix: &str, shown: &mut usize) -> String {
    let mut out = String::new();
    let mut shift = 0isize;
    for s in spans {
        let (a, b) = line_bounds(before, s.start, s.end);
        // The same lines after the edit: everything before `start` is where it
        // was, shifted by the replacements already spliced in.
        let (c, d) = line_bounds(after, (a as isize + shift) as usize,
                                        (s.start as isize + shift + s.text.len() as isize) as usize);
        let old: Vec<&str> = lines_of(&before[a..b]).into_iter().map(|(_, body, _)| body).collect();
        let new: Vec<&str> = lines_of(&after[c..d]).into_iter().map(|(_, body, _)| body).collect();
        shift += s.text.len() as isize - (s.end - s.start) as isize;
        let label = format!("@@ {prefix}line {}", s.line);
        if old.len() + new.len() > DIFF_ROWS || *shown + old.len() + new.len() > DIFF_TOTAL {
            out.push_str(&format!("\n{label} — {} line(s) to {} line(s), diff omitted", old.len(), new.len()));
            continue;
        }
        out.push_str(&format!("\n{label}"));
        for l in &old { out.push_str(&format!("\n-{l}")); }
        for l in &new { out.push_str(&format!("\n+{l}")); }
        *shown += old.len() + new.len();
    }
    out
}

/// The tolerant pass: align `old` to whole lines, ignoring leading and
/// trailing whitespace on each, and return every place the block fits, with
/// `new` re-indented to the indentation found there. An all-whitespace `old`
/// matches nowhere — there would be nothing anchoring it — and returning
/// every fit is what lets the caller report ambiguity instead of picking one.
fn fuzzy_hits(content: &str, old: &str, new: &str) -> Vec<(usize, usize, String)> {
    let mut pat: Vec<&str> = old.split('\n').collect();
    if old.ends_with('\n') { pat.pop(); } // the final newline ends the block
    let pat: Vec<&str> = pat.into_iter().map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();
    if pat.iter().all(|l| l.trim().is_empty()) { return Vec::new(); }
    let file = lines_of(content);
    if pat.len() > file.len() { return Vec::new(); }
    let from = &pat[0][..pat[0].len() - pat[0].trim_start().len()];
    let mut hits = Vec::new();
    for i in 0..=file.len() - pat.len() {
        if (0..pat.len()).all(|k| file[i + k].1.trim() == pat[k].trim()) {
            let body = file[i].1;
            let last = file[i + pat.len() - 1];
            let (start, end) = (file[i].0, last.0 + last.2);
            let mut text = reindent(new, from, &body[..body.len() - body.trim_start().len()]);
            // The block matched to the end of its last line, so the line
            // terminator is part of what is being replaced: the replacement
            // has to carry one too, or the following line rides up into it.
            let term = if content[..end].ends_with("\r\n") { "\r\n" }
                       else if content[..end].ends_with('\n') { "\n" } else { "" };
            if !text.is_empty() && !term.is_empty() && !text.ends_with('\n') { text.push_str(term); }
            hits.push((start, end, text));
        }
    }
    hits
}

/// Shift `new` from the indentation `old` was written with to the indentation
/// the file actually uses — the same edit, landing where the code lives. Only
/// the block's own base indent moves; deeper lines keep their relative shape.
fn reindent(new: &str, from: &str, to: &str) -> String {
    if from == to { return new.to_string(); }
    let mut out = String::with_capacity(new.len() + 16);
    for (i, line) in new.split('\n').enumerate() {
        if i > 0 { out.push('\n'); }
        match line.strip_prefix(from) {
            Some(rest) if !line.trim().is_empty() => { out.push_str(to); out.push_str(rest); }
            _ => out.push_str(line),
        }
    }
    out
}

/// The file's lines as (byte offset, body without its terminator, byte length
/// including it). Bodies keep their indentation: the tolerant pass trims for
/// comparison, and the difference is what re-indents `new`.
fn lines_of(content: &str) -> Vec<(usize, &str, usize)> {
    let mut out = Vec::new();
    let mut at = 0;
    for raw in content.split_inclusive('\n') {
        let body = raw.strip_suffix('\n').unwrap_or(raw);
        out.push((at, body.strip_suffix('\r').unwrap_or(body), raw.len()));
        at += raw.len();
    }
    out
}

/// 1-based line number of a byte offset.
fn line_at(content: &str, at: usize) -> usize {
    content[..at].matches('\n').count() + 1
}

/// The byte range of the whole lines a range touches. A replacement is
/// usually a fragment of a line, and a diff of the fragment alone reads as if
/// the indentation had been stripped — so the report widens to the lines the
/// edit actually lands on. A range that ends where a line began (a whole-line
/// replacement, a deletion) covers no line of its own there, and must not
/// swallow the next one.
fn line_bounds(content: &str, from: usize, to: usize) -> (usize, usize) {
    let start = content[..from].rfind('\n').map_or(0, |i| i + 1);
    let end = if to <= start || content[..to].ends_with('\n') {
        to.max(start)
    } else {
        content[to..].find('\n').map_or(content.len(), |i| to + i + 1)
    };
    (start, end)
}

#[cfg(test)]
mod tests {
    use crate::test_util::temp_dir;
    use crate::tools::dispatch;
    use serde_json::json;
    use std::fs;

    #[test]
    fn edit_reports_where_it_landed_and_leaves_no_temp_behind() {
        let dir = temp_dir("edit_report");
        let p = dir.join("f.rs");
        fs::write(&p, "fn main() {\n    let a = 1;\n    let b = 2;\n}\n").unwrap();
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "let b = 2;", "new": "let b = 3;"}));
        assert!(out.contains("line 3"), "report names the line: {out}");
        assert!(out.contains("-    let b = 2;") && out.contains("+    let b = 3;"), "report carries a diff: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "fn main() {\n    let a = 1;\n    let b = 3;\n}\n");
        // the write is temp+rename: the temp file must not outlive it
        let leftovers: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.contains("jingwei")).collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
    }

    #[test]
    fn edit_ignores_indentation_drift_but_not_a_wrong_target() {
        let dir = temp_dir("edit_fuzzy");
        let p = dir.join("f.rs");
        let file = "fn main() {\n    if ready {\n        go();\n    }\n}\n";
        // `old` copied without the file's indentation — the tolerant pass lands
        // it and shifts `new` to where the lines actually sit
        fs::write(&p, file).unwrap();
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "if ready {\n    go();\n}", "new": "if ready {\n    run();\n}"}));
        assert!(out.starts_with("ok:"), "got: {out}");
        assert!(out.contains("ignoring indentation"), "says why it matched: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "fn main() {\n    if ready {\n        run();\n    }\n}\n");
        // the tolerant pass still refuses to guess between two placements
        fs::write(&p, "go();\n\ngo();\n").unwrap();
        dispatch("read_file", &json!({"path": p.to_str().unwrap()})); // the bytes above are not the ones we saw
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "  go();", "new": "run();"}));
        assert!(out.contains("appears 2 times") && out.contains("lines 1, 3"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "go();\n\ngo();\n", "an ambiguous edit writes nothing");
    }

    #[test]
    fn edit_replace_all_covers_a_repeated_idiom() {
        let dir = temp_dir("edit_all");
        let p = dir.join("f.rs");
        fs::write(&p, "let a = old(\"x\");\nlet b = old(\"x\");\nlet c = old(\"x\");\n").unwrap();
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "old(\"x\")", "new": "new(\"x\")", "replace_all": true}));
        assert!(out.contains("3 replacements at lines 1, 2, 3"), "got: {out}");
        assert!(!fs::read_to_string(&p).unwrap().contains("old("), "every occurrence replaced");
        // and the two-arg form still refuses, naming the lines
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "new(\"x\")", "new": "z"}));
        assert!(out.contains("replace_all: true"), "the refusal offers the way out: {out}");
    }

    #[test]
    fn edit_empty_old_says_what_to_do_instead() {
        let dir = temp_dir("edit_empty");
        let p = dir.join("f.txt");
        fs::write(&p, "x").unwrap();
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "", "new": "y"}));
        assert!(out.contains("write_file"), "points at the right tool: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "x");
    }

    #[test]
    fn edits_in_one_call_land_together_or_not_at_all() {
        let dir = temp_dir("edit_batch");
        let p = dir.join("f.rs");
        fs::write(&p, "old_a();\nkeep();\nold_b();\n").unwrap();
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "edits": [
            {"old": "old_a()", "new": "new_a()"}, {"old": "old_b()", "new": "new_b()"}]}));
        assert!(out.contains("2 edits, 2 replacements"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "new_a();\nkeep();\nnew_b();\n");
        // a pair may edit what an earlier pair wrote: the batch is in order
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "edits": [
            {"old": "new_a()", "new": "tmp_a()"}, {"old": "tmp_a()", "new": "final_a()"}]}));
        assert!(out.starts_with("ok:"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "final_a();\nkeep();\nnew_b();\n");
        // one pair that cannot be placed takes the whole call down, names
        // itself, and writes nothing
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "edits": [
            {"old": "keep()", "new": "gone()"}, {"old": "missing()", "new": "x"}]}));
        assert!(out.contains("edit 2/2") && out.contains("nothing was written"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "final_a();\nkeep();\nnew_b();\n", "all or nothing");
        // the two spellings do not mix, and neither does an empty batch
        let both = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "a", "new": "b",
            "edits": [{"old": "a", "new": "b"}]}));
        assert!(both.contains("not both"), "got: {both}");
        let none = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "edits": []}));
        assert!(none.contains("`edits` is empty"), "got: {none}");
        let malformed = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "edits": [{"old": "a"}]}));
        assert!(malformed.contains("edits[1] needs"), "got: {malformed}");
    }

    #[test]
    fn an_edit_refuses_a_file_that_moved_since_it_was_read() {
        let dir = temp_dir("edit_stale");
        let p = dir.join("f.txt");
        fs::write(&p, "one\ntwo\n").unwrap();
        assert!(dispatch("read_file", &json!({"path": p.to_str().unwrap()})).starts_with("one"));
        // someone else's write — the user's editor, a formatter — lands after
        // our read: the ledger knows these are not the bytes we saw
        fs::write(&p, "one\nTWO\n").unwrap();
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "two", "new": "three"}));
        assert!(out.contains("changed on disk"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "one\nTWO\n", "nothing was written");
        // the same guard covers the blunt tool
        let out = dispatch("write_file", &json!({"path": p.to_str().unwrap(), "content": "clobbered"}));
        assert!(out.contains("changed on disk"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "one\nTWO\n");
        // re-reading is the whole fix — now the edit lands
        assert!(dispatch("read_file", &json!({"path": p.to_str().unwrap()})).contains("TWO"));
        let out = dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "TWO", "new": "three"}));
        assert!(out.starts_with("ok:"), "got: {out}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "one\nthree\n");
        // a file jingwei has never seen has nothing to compare against, and is
        // written as before — this is a change detector, not a read-first gate
        let fresh = dir.join("fresh.txt");
        fs::write(&fresh, "hello").unwrap();
        assert!(dispatch("edit_file", &json!({"path": fresh.to_str().unwrap(), "old": "hello", "new": "bye"})).starts_with("ok:"));
    }
}
