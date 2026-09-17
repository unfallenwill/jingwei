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
                notes.push(format!("{} edits", a.len()));
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

/// Combined-output shape for the synchronous bash entry. The async one in
/// `tool_runtime` builds the same string, so the tool's output is the same
/// whether the call lands on the blocking path or the cancellable one.
fn bash_output(status: Option<&ExitStatus>, out: &str, err: &str) -> String {
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

    #[test]
    fn bash_tool_returns_exit_code_and_output() {
        let ok = dispatch("bash", &json!({"command": "echo hi"}));
        assert!(ok.starts_with("exit=0") && ok.contains("hi"), "got: {ok}");
        let bad = dispatch("bash", &json!({"command": if cfg!(windows) { "exit 1" } else { "false" }}));
        assert!(!bad.starts_with("exit=0") && bad.contains("exit="), "got: {bad}");
    }
}
