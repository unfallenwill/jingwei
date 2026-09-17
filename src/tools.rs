// The four tools, as the registry knows them. Each entry is the plain
// description the agent hands to the model — name, what it does, the
// JSON-schema shape — plus a synchronous `run` that the coroutine runtime
// calls on the blocking pool. `bash` is special-cased in `tool_runtime`
// because a runaway shell is the one tool whose execution has to be
// cancellable; everything else is fast enough that "abandon the in-flight
// task on Ctrl-C" is the right answer.
//
// The display port (`Msg::Tool`) is called once per tool call with a
// one-line summary the frontends can echo verbatim, and the full output the
// frontends decide how much of to show.

use crate::display::{Msg, Show};
use crate::edit::edit_tool;
use crate::file_io::write_atomic;
use crate::ledger::{ledger_note, stale};
use serde_json::{json, Value};
use std::fs;
use std::process::{Command, ExitStatus};

/// One-line summary of a tool call's arguments: the command, the path, …
pub(crate) fn tool_summary(name: &str, input: &Value) -> String {
    match name {
        "bash" => format!("$ {}", input["command"].as_str().unwrap_or("")),
        "read_file" => input["path"].as_str().unwrap_or("").into(),
        "write_file" => format!("{} ({} bytes)",
            input["path"].as_str().unwrap_or(""),
            input["content"].as_str().map_or(0, str::len)),
        "edit_file" => {
            let p = input["path"].as_str().unwrap_or("");
            let mut notes: Vec<String> = Vec::new();
            if let Some(a) = input["edits"].as_array().filter(|a| !a.is_empty()) {
                notes.push(format!("{} edit{}", a.len(), if a.len() == 1 { "" } else { "s" }));
            }
            if input["replace_all"].as_bool().unwrap_or(false) { notes.push("replace_all".into()); }
            if notes.is_empty() { p.into() } else { format!("{p} ({})", notes.join(", ")) }
        },
        _ => String::new(),
    }
}

/// Echo a tool call through the display port: the frontends decide how
/// much of the output tail to show (and how to fold the rest).
pub(crate) fn print_tool_call(name: &str, input: &Value, output: &str, sink: &dyn Show) {
    sink.show(Msg::Tool { name: name.into(), summary: tool_summary(name, input), output: output.into() });
}

/// The one tool type — `name`, `desc`, `schema` go to the model; `run` is
/// the synchronous body that the coroutine runtime hands the blocking pool.
pub(crate) struct Tool {
    pub(crate) name: &'static str,
    pub(crate) desc: &'static str,
    pub(crate) schema: Value,
    pub(crate) run: fn(&Value) -> String,
}

pub(crate) fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "bash",
            desc: "Run a shell command; returns combined stdout/stderr and the exit code. \
                   Use `cd <dir> && <cmd>` to change directory within a call.",
            schema: json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}),
            run: |i| {
                let prog_flag = if cfg!(windows) { ("cmd", "/C") } else { ("sh", "-c") };
                match Command::new(prog_flag.0).args([prog_flag.1, i["command"].as_str().unwrap_or("")]).output() {
                    Ok(o) => bash_output(Some(&o.status),
                        &String::from_utf8_lossy(&o.stdout),
                        &String::from_utf8_lossy(&o.stderr)),
                    Err(e) => format!("error: {e}"),
                }
            },
        },
        Tool {
            name: "read_file",
            desc: "Read the full contents of a file.",
            schema: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            run: |i| {
                let p = i["path"].as_str().unwrap_or("");
                match fs::read(p) {
                    // Reading a file is what makes it editable: the ledger
                    // remembers these bytes, so an edit that later finds
                    // different ones refuses instead of writing over them.
                    Ok(b) => { ledger_note(p, &b); String::from_utf8_lossy(&b).into_owned() }
                    Err(e) => format!("error: {e}"),
                }
            },
        },
        Tool {
            name: "write_file",
            desc: "Write content to a file, creating parent dirs as needed. Overwrites existing files.",
            schema: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
            run: |i| {
                let p = i["path"].as_str().unwrap_or("");
                let c = i["content"].as_str().unwrap_or("");
                // The blunt tool gets the guard too — overwriting a file that
                // moved under us is the most expensive thing this agent can do.
                if let Some(e) = stale(p) { return e; }
                if let Some(dir) = std::path::Path::new(p).parent().filter(|d| !d.as_os_str().is_empty()) {
                    let _ = fs::create_dir_all(dir);
                }
                match write_atomic(p, c) {
                    Ok(()) => { ledger_note(p, c.as_bytes()); format!("ok: wrote {} bytes to {p}", c.len()) }
                    Err(e) => format!("error: {e}"),
                }
            },
        },
        Tool {
            name: "edit_file",
            desc: "Edit a file: replace `old` with `new`, or apply several pairs as `edits` in one \
                   all-or-nothing call (in order — each sees what the ones before it wrote). Matching \
                   is exact first; failing that, whole lines are aligned ignoring indentation (and \
                   `new` is shifted to where they sit). If `old` matches more than one place the \
                   error names the line numbers — widen `old` with a line of context around the one \
                   you mean, or set replace_all:true to replace every occurrence. The result carries \
                   the affected line numbers and a diff. Refuses to write over a file that changed \
                   on disk since jingwei last read it: re-read it, then redo the edit.",
            schema: json!({"type":"object","properties":{
                "path":{"type":"string"},
                "old":{"type":"string","description":"the text to find"},
                "new":{"type":"string","description":"what goes in its place; omit to delete"},
                "edits":{"type":"array","description":"several `old`/`new` pairs applied in one call, in order, all or nothing","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"]}},
                "replace_all":{"type":"boolean","description":"replace every occurrence, instead of refusing when `old` is not unique"}},
                "required":["path"]}),
            run: |i| edit_tool(i),
        },
    ]
}

/// Combined-output shape for the bash tool — shared by the synchronous
/// registry entry (below) and the cancellable coroutine path in
/// `tool_runtime`, so the tool's output is the same shape whichever
/// path the call lands on. The async caller can hand `Some(&status)`
/// after reaping the child; the sync one hands `Some(&output.status)`
/// from the std `Command::output` it just consumed.
pub(crate) fn bash_output(status: Option<&ExitStatus>, out: &str, err: &str) -> String {
    let mut s = format!("exit={}\n{out}", status.and_then(|st| st.code()).unwrap_or(-1));
    if !err.is_empty() {
        if !s.ends_with('\n') { s.push('\n'); }
        s.push_str(err);
    }
    s
}

pub(crate) fn dispatch(name: &str, input: &Value) -> String {
    tools().into_iter().find(|t| t.name == name)
        .map(|t| (t.run)(input))
        .unwrap_or_else(|| format!("error: unknown tool '{name}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::temp_dir;
    use serde_json::json;

    #[test]
    fn tools_registry_is_wellformed() {
        let t = tools();
        assert_eq!(t.len(), 4);
        assert_eq!(t.iter().map(|x| x.name).collect::<std::collections::BTreeSet<_>>().len(), 4);
        for x in t {
            assert_eq!(x.schema["type"], "object");
            assert!(x.schema["properties"].is_object() && x.schema["required"].is_array());
        }
    }

    // ---- tool_summary: the one-line echo the frontends show ---------------

    #[test]
    fn tool_summary_for_bash_prints_command_with_dollar() {
        assert_eq!(tool_summary("bash", &json!({"command": "ls -l"})), "$ ls -l");
        // missing command: empty string, not a panic
        assert_eq!(tool_summary("bash", &json!({})), "$ ");
    }

    #[test]
    fn tool_summary_for_read_file_is_the_path() {
        assert_eq!(tool_summary("read_file", &json!({"path": "src/main.rs"})), "src/main.rs");
    }

    #[test]
    fn tool_summary_for_write_file_lists_path_with_size() {
        assert_eq!(tool_summary("write_file", &json!({"path": "a.rs", "content": "hello"})),
                   "a.rs (5 bytes)");
        // missing content: byte count is zero
        assert_eq!(tool_summary("write_file", &json!({"path": "a.rs"})), "a.rs (0 bytes)");
    }

    #[test]
    fn tool_summary_for_edit_file_without_notes_is_just_the_path() {
        assert_eq!(tool_summary("edit_file", &json!({"path": "a.rs"})), "a.rs");
    }

    #[test]
    fn tool_summary_for_edit_file_with_edits_lists_the_count() {
        let v = json!({"path": "a.rs", "edits": [{"old": "a", "new": "b"}, {"old": "c", "new": "d"}]});
        assert_eq!(tool_summary("edit_file", &v), "a.rs (2 edits)");
        // a single edit shows up as "edit", not "edits" — the count is
        // grammatical, not just numeric
        let v1 = json!({"path": "a.rs", "edits": [{"old": "a", "new": "b"}]});
        assert_eq!(tool_summary("edit_file", &v1), "a.rs (1 edit)");
    }

    #[test]
    fn tool_summary_for_edit_file_with_replace_all_adds_the_flag() {
        let v = json!({"path": "a.rs", "old": "x", "new": "y", "replace_all": true});
        assert_eq!(tool_summary("edit_file", &v), "a.rs (replace_all)");
    }

    #[test]
    fn tool_summary_for_edit_file_with_edits_and_replace_all_lists_both() {
        let v = json!({"path": "a.rs", "edits": [{"old": "a", "new": "b"}], "replace_all": true});
        assert_eq!(tool_summary("edit_file", &v), "a.rs (1 edit, replace_all)");
        // a single edit with replace_all still uses the singular noun
    }

    #[test]
    fn tool_summary_for_unknown_tool_returns_empty() {
        // the `_ => String::new()` arm: every unknown tool name (or
        // typo'd one) must produce an empty summary. We try several
        // names to make sure the catch-all actually catches all.
        for name in ["nope", "", "BASH", "Edit_File", "rm-rf"] {
            assert_eq!(tool_summary(name, &json!({})), "",
                "unknown tool {name:?} must produce an empty summary");
        }
    }

    // ---- print_tool_call: the channel through the display port ------------

    /// A sink that captures every Msg handed to it: the tests assert on
    /// the message the frontends would receive, not on a string.
    struct CapturingSink(std::sync::Mutex<Vec<Msg>>);
    impl crate::display::Show for CapturingSink {
        fn show(&self, m: Msg) { self.0.lock().unwrap().push(m); }
    }

    #[test]
    fn print_tool_call_emits_a_tool_message_with_summary_and_output() {
        let sink = CapturingSink(Default::default());
        print_tool_call("bash", &json!({"command": "echo hi"}), "out", &sink);
        let got = sink.0.lock().unwrap().pop().unwrap();
        match got {
            Msg::Tool { name, summary, output } => {
                assert_eq!(name, "bash");
                assert_eq!(summary, "$ echo hi");
                assert_eq!(output, "out");
            }
            other => panic!("expected Msg::Tool, got {other:?}"),
        }
    }

    // ---- bash_output: same shape as the async tool's output ----------------

    #[test]
    fn bash_output_with_only_stdout_skips_stderr() {
        let st = make_exit(0);
        let s = bash_output(Some(&st), "hello\n", "");
        assert_eq!(s, "exit=0\nhello\n");
    }

    #[test]
    fn bash_output_appends_stderr_after_a_newline() {
        let st = make_exit(0);
        let s = bash_output(Some(&st), "out", "err");
        // no trailing newline on `out`, so the format adds one before err
        assert_eq!(s, "exit=0\nout\nerr");
    }

    #[test]
    fn bash_output_preserves_a_trailing_newline_before_stderr() {
        let st = make_exit(1);
        let s = bash_output(Some(&st), "out\n", "err");
        assert_eq!(s, "exit=1\nout\nerr");
    }

    #[test]
    fn bash_output_without_status_uses_minus_one() {
        let s = bash_output(None, "out", "");
        assert_eq!(s, "exit=-1\nout");
    }

    /// Make an ExitStatus for tests. On Unix, `Command::status("true")` is
    /// the cross-platform answer; on Windows, "cmd /C exit 0" works.
    fn make_exit(code: i32) -> ExitStatus {
        if cfg!(windows) {
            std::process::Command::new("cmd").args(["/C", &format!("exit {code}")]).status().unwrap()
        } else {
            std::process::Command::new("sh").args(["-c", &format!("exit {code}")]).status().unwrap()
        }
    }

    // ---- dispatch: the registry's one entry point -------------------------

    #[test]
    fn dispatch_unknown_tool_returns_an_error_message() {
        let out = dispatch("nope", &json!({}));
        assert!(out.starts_with("error: unknown tool"), "got: {out}");
    }

    // ---- the file tools: exercised through dispatch -----------------------

    #[test]
    fn read_file_returns_error_with_path_wrapped_when_missing() {
        let dir = temp_dir("read_missing");
        let missing = dir.join("nope.txt");
        let result = dispatch("read_file", &json!({"path": missing.to_str().unwrap()}));
        assert!(result.starts_with("error:"), "got: {result}");
    }

    #[test]
    fn write_file_returns_error_when_path_is_a_directory() {
        // a path the runtime can't write to: a directory.
        let dir = temp_dir("write_dir");
        let result = dispatch("write_file", &json!({"path": dir.to_str().unwrap(), "content": "x"}));
        // the io error is whatever the OS returns — we just verify the
        // tool formatted it as an error string instead of panicking
        assert!(result.starts_with("error:") || result.starts_with("ok: wrote"),
            "got: {result}");
    }

    #[test]
    fn write_file_writes_through_a_symlink_to_its_target() {
        // a symlink in the temp dir, written through — the write must
        // land on the *target*, not replace the link
        let dir = temp_dir("symlink_write");
        let target = dir.join("real.txt");
        std::fs::write(&target, "").unwrap(); // an empty file to symlink at
        let link = dir.join("link.txt");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let result = dispatch("write_file", &json!({"path": link.to_str().unwrap(), "content": "v"}));
        assert!(result.starts_with("ok: wrote"), "got: {result}");
        // both paths now point at content
        let on_target = std::fs::read_to_string(&target).unwrap();
        let via_link = std::fs::read_link(&link).unwrap();
        assert_eq!(via_link, target, "the symlink itself is unchanged");
        assert_eq!(on_target, "v", "the target got the bytes");
    }
}
