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

mod edit;
mod ledger;
pub(crate) mod runtime;

use crate::ui::{Msg, Show};
use crate::cancel::CancelToken;
use crate::core::ports::{BoxFuture, ToolProvider};
use crate::file_io::write_atomic;
use crate::tools::edit::edit_tool;
use crate::tools::ledger::{ledger_note, stale};
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
            desc: "Run a shell command (`sh -c` on Unix, `cmd /C` on Windows). Output shape is \
                   `exit=N\\n<stdout><optional stderr>` (N is the child's status; `-1` means the process \
                   was terminated by a signal with no normal exit code). Output longer than 50,000 bytes \
                   is truncated with a `…[truncated, N bytes total]` marker — tail the file or pipe \
                   through `head` / `tail` / `sed` if you need more. The shell inherits the agent's \
                   current working directory; relative paths resolve there, not against the user's \
                   shell. Use `cd <dir> && <cmd>` to change directory within a single call. Long-running \
                   commands are killed on Ctrl-C; partial output still reaches the model with an \
                   `[interrupted by user]` marker.",
            schema: json!({"type":"object","properties":{
                "command":{"type":"string","description":"Shell command string; passed to `sh -c` on Unix or `cmd /C` on Windows. Multi-line commands use `;` or `&&`."}
            },"required":["command"]}),
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
            desc: "Read the full contents of a file. Use this — not `cat` via bash — before any edit. \
                   A read also loads the file into the agent's ledger: edit_file and write_file \
                   then refuse to touch the file if it changed on disk, so re-read after any \
                   external edit (rustfmt, git pull, sed, …).",
            schema: json!({"type":"object","properties":{
                "path":{"type":"string","description":"Absolute or relative path; resolves against the agent's current working directory."}
            },"required":["path"]}),
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
            desc: "Write content to a file, creating parent dirs as needed. Overwrites existing files. \
                   Prefer edit_file for small or targeted changes — this is the blunt tool and \
                   will silently replace uncommitted work. The write is atomic (temp+rename): \
                   a crash mid-call cannot leave a truncated file. Refuses to overwrite if the \
                   file changed since jingwei last read or wrote it; re-read first, then write.",
            schema: json!({"type":"object","properties":{
                "path":{"type":"string","description":"Absolute or relative path; resolves against the agent's current working directory. Parent dirs are created if they do not exist."},
                "content":{"type":"string","description":"Full file contents — the tool always overwrites. For a small or targeted change, use edit_file instead."}
            },"required":["path","content"]}),
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
                   on disk since jingwei last read it: re-read it, then redo the edit. Passing an \
                   empty `old` is refused — use write_file to create or replace a file wholesale.",
            schema: json!({"type":"object","properties":{
                "path":{"type":"string","description":"Absolute or relative path; resolves against the agent's current working directory."},
                "old":{"type":"string","description":"the text to find"},
                "new":{"type":"string","description":"what goes in its place; omit to delete"},
                "edits":{"type":"array","description":"several `old`/`new` pairs applied in one call, in order, all or nothing. Omitting `new` in an item is a deletion, mirroring the outer `old`/`new` shape.","items":{"type":"object","properties":{
                    "old":{"type":"string","description":"the text to find"},
                    "new":{"type":"string","description":"what goes in its place; omit to delete (matches the outer `old`/`new` shape)"}
                },"required":["old"]}},
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

// ---- the built-in provider --------------------------------------------------

/// Tool schemas for the built-in registry: `bash`, `read_file`,
/// `write_file`, `edit_file`. Pure over the static `tools()` table, and
/// the exact expression the model's cached prefix is built from — the
/// bytes and the order are a cache contract, so this body is not free to
/// be reworded.
pub(crate) fn builtin_schemas() -> Vec<Value> {
    tools().iter()
        .map(|t| json!({"name": t.name, "description": t.desc, "input_schema": t.schema}))
        .collect()
}

/// The four shipped tools, as a tool provider.
///
/// The provider is the seam the router talks to; the registry below it is
/// unchanged. `owns` is "one of the four", `definitions` is the four
/// schemas in the registry's order, and `call` runs the same bodies the
/// tool runtime ran before the provider existed.
///
/// Cancellation formats are deliberately different — the model can tell
/// what happened from the prefix alone: bash preserves whatever output
/// had already arrived (so the marker sits *after* the bytes, and reads
/// like a note the tool appended) while the file tools had no partial
/// output to keep (so the whole tool result is a single line saying so).
pub(crate) struct Builtins;

impl ToolProvider for Builtins {
    fn owns(&self, name: &str) -> bool {
        matches!(name, "bash" | "read_file" | "write_file" | "edit_file")
    }

    fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>> {
        // the built-ins are synchronous: a ready future is the honest
        // shape for them, and it keeps the trait dyn-compatible.
        Box::pin(std::future::ready(builtin_schemas()))
    }

    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a Value,
        token: &'a CancelToken,
    ) -> BoxFuture<'a, String> {
        Box::pin(async move {
            if name == "bash" {
                // bash owns its own child process: it is the one tool
                // whose execution has to be cancellable, so it takes the
                // coroutine path, not the registry's synchronous one.
                return crate::tools::runtime::run_bash(
                    input["command"].as_str().unwrap_or(""), token).await;
            }
            let name = name.to_string();
            let input = input.clone();
            let job = tokio::task::spawn_blocking(move || dispatch(&name, &input));
            tokio::select! {
                out = job => out.unwrap_or_else(|e| format!("error: {e}")),
                _ = token.cancelled() => "[interrupted by user]".into(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::temp_dir;
    use serde_json::json;
    use std::fs;

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

    #[test]
    fn write_read_edit_file_contract() {
        let dir = temp_dir("files");
        let p = dir.join("f.txt");
        // write (with missing parent dirs) + read roundtrip
        assert!(dispatch("write_file", &json!({"path": dir.join("a/b.txt").to_str().unwrap(), "content": "v1"})).starts_with("ok:"));
        assert_eq!(dispatch("read_file", &json!({"path": dir.join("a/b.txt").to_str().unwrap()})), "v1");
        // edit unique
        fs::write(&p, "hello world").unwrap();
        assert!(dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "hello", "new": "hi"})).starts_with("ok:"));
        assert_eq!(fs::read_to_string(&p).unwrap(), "hi world");
        // edit missing
        assert!(dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "zzz", "new": "x"})).contains("not found"));
        // edit ambiguous — re-read so the ledger knows the current bytes,
        // and the ambiguity message is the one we test (the ledger check
        // fires first and would otherwise steal the message).
        fs::write(&p, "ab ab").unwrap();
        dispatch("read_file", &json!({"path": p.to_str().unwrap()}));
        assert!(dispatch("edit_file", &json!({"path": p.to_str().unwrap(), "old": "ab", "new": "cd"})).contains("appears 2 times"));
        // read missing
        assert!(dispatch("read_file", &json!({"path": dir.join("nope").to_str().unwrap()})).starts_with("error:"));
        // unknown tool
        assert!(dispatch("nope", &json!({})).starts_with("error: unknown tool"));
    }

    // ---- the built-in provider: the four shipped tools --------------------

    /// The composed list the model saw before the router existed: the four
    /// built-in schemas, byte for byte. The cached prefix is system + tools,
    /// so a byte here is a byte that invalidates the whole prefix — this
    /// literal is the proof that the move changed nothing. Captured from the
    /// tree *before* the provider existed.
    const GOLDEN_BUILTIN_SCHEMAS: &str = r#"[{"description":"Run a shell command (`sh -c` on Unix, `cmd /C` on Windows). Output shape is `exit=N\\n<stdout><optional stderr>` (N is the child's status; `-1` means the process was terminated by a signal with no normal exit code). Output longer than 50,000 bytes is truncated with a `…[truncated, N bytes total]` marker — tail the file or pipe through `head` / `tail` / `sed` if you need more. The shell inherits the agent's current working directory; relative paths resolve there, not against the user's shell. Use `cd <dir> && <cmd>` to change directory within a single call. Long-running commands are killed on Ctrl-C; partial output still reaches the model with an `[interrupted by user]` marker.","input_schema":{"properties":{"command":{"description":"Shell command string; passed to `sh -c` on Unix or `cmd /C` on Windows. Multi-line commands use `;` or `&&`.","type":"string"}},"required":["command"],"type":"object"},"name":"bash"},{"description":"Read the full contents of a file. Use this — not `cat` via bash — before any edit. A read also loads the file into the agent's ledger: edit_file and write_file then refuse to touch the file if it changed on disk, so re-read after any external edit (rustfmt, git pull, sed, …).","input_schema":{"properties":{"path":{"description":"Absolute or relative path; resolves against the agent's current working directory.","type":"string"}},"required":["path"],"type":"object"},"name":"read_file"},{"description":"Write content to a file, creating parent dirs as needed. Overwrites existing files. Prefer edit_file for small or targeted changes — this is the blunt tool and will silently replace uncommitted work. The write is atomic (temp+rename): a crash mid-call cannot leave a truncated file. Refuses to overwrite if the file changed since jingwei last read or wrote it; re-read first, then write.","input_schema":{"properties":{"content":{"description":"Full file contents — the tool always overwrites. For a small or targeted change, use edit_file instead.","type":"string"},"path":{"description":"Absolute or relative path; resolves against the agent's current working directory. Parent dirs are created if they do not exist.","type":"string"}},"required":["path","content"],"type":"object"},"name":"write_file"},{"description":"Edit a file: replace `old` with `new`, or apply several pairs as `edits` in one all-or-nothing call (in order — each sees what the ones before it wrote). Matching is exact first; failing that, whole lines are aligned ignoring indentation (and `new` is shifted to where they sit). If `old` matches more than one place the error names the line numbers — widen `old` with a line of context around the one you mean, or set replace_all:true to replace every occurrence. The result carries the affected line numbers and a diff. Refuses to write over a file that changed on disk since jingwei last read it: re-read it, then redo the edit. Passing an empty `old` is refused — use write_file to create or replace a file wholesale.","input_schema":{"properties":{"edits":{"description":"several `old`/`new` pairs applied in one call, in order, all or nothing. Omitting `new` in an item is a deletion, mirroring the outer `old`/`new` shape.","items":{"properties":{"new":{"description":"what goes in its place; omit to delete (matches the outer `old`/`new` shape)","type":"string"},"old":{"description":"the text to find","type":"string"}},"required":["old"],"type":"object"},"type":"array"},"new":{"description":"what goes in its place; omit to delete","type":"string"},"old":{"description":"the text to find","type":"string"},"path":{"description":"Absolute or relative path; resolves against the agent's current working directory.","type":"string"},"replace_all":{"description":"replace every occurrence, instead of refusing when `old` is not unique","type":"boolean"}},"required":["path"],"type":"object"},"name":"edit_file"}]"#;

    #[test]
    fn the_built_in_schemas_are_byte_identical_to_the_pre_move_list() {
        assert_eq!(
            serde_json::to_string(&builtin_schemas()).unwrap(),
            GOLDEN_BUILTIN_SCHEMAS,
            "the built-in schema list is a cache contract: its bytes and its order must not move",
        );
    }

    #[test]
    fn the_built_in_provider_offers_the_four_schemas_in_todays_order() {
        let ts = builtin_schemas();
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

    #[test]
    fn the_built_in_provider_owns_only_the_four_names() {
        let b = Builtins;
        for name in ["bash", "read_file", "write_file", "edit_file"] {
            assert!(b.owns(name), "{name} is one of the four");
        }
        for name in ["mcp__x__y", "nope", "", "BASH"] {
            assert!(!b.owns(name), "{name:?} is not a built-in");
        }
    }

    #[test]
    fn the_built_in_provider_runs_a_tool() {
        let dir = temp_dir("builtin_provider");
        let p = dir.join("f.txt");
        std::fs::write(&p, "hello").unwrap();
        let out = crate::test_util::block_on(
            Builtins.call("read_file", &json!({"path": p.to_str().unwrap()}),
                          &crate::cancel::CancelToken::new()),
        );
        assert_eq!(out, "hello", "got: {out}");
    }

    /// bash takes the coroutine path, not the registry's synchronous one:
    /// it owns its child process so a runaway shell can be killed.
    #[test]
    fn the_built_in_provider_runs_bash_through_the_cancellable_path() {
        let out = crate::test_util::block_on(
            Builtins.call("bash", &json!({"command": "echo hi"}),
                          &crate::cancel::CancelToken::new()),
        );
        assert!(out.starts_with("exit=0"), "got: {out}");
        assert!(out.contains("hi"), "got: {out}");
    }

    /// A long-running bash with a token that fires mid-call: the child is
    /// killed and the tool answers with the interruption marker.
    #[test]
    fn the_built_in_provider_is_interrupted_when_the_token_fires() {
        use std::sync::Arc;
        use std::time::Duration;
        let token = Arc::new(crate::cancel::CancelToken::new());
        let t = token.clone();
        let out = crate::test_util::block_on(async move {
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(200));
                t.cancel();
            });
            Builtins.call("bash", &json!({"command": "sleep 30"}), &token).await
        });
        assert!(out.contains("[interrupted by user]"), "got: {out}");
    }

    /// The non-bash path's select races the blocking pool against the
    /// cancel — a pre-cancelled token always wins. The marker is the same
    /// word bash uses, so the model sees one shape for "this turn did not
    /// finish" no matter which tool ran.
    #[test]
    fn the_built_in_provider_answers_a_pre_cancelled_call_with_the_marker() {
        let dir = temp_dir("builtin_cancel");
        let token = crate::cancel::CancelToken::new();
        token.cancel();
        let out = crate::test_util::block_on(
            Builtins.call("write_file",
                          &json!({"path": dir.join("f.txt").to_str().unwrap(), "content": "x"}),
                          &token),
        );
        assert_eq!(out, "[interrupted by user]",
            "non-bash cancellation uses the bash marker: {out}");
    }

    /// The order of the composed list is a cache contract: the built-ins
    /// come first, then the MCP tools, exactly as the context's own refresh
    /// used to produce them. The router reproduces that order.
    #[test]
    fn the_router_composes_the_built_ins_before_the_mcp_tools() {
        /// A stand-in for the MCP provider: it owns an `mcp__` name and
        /// offers one schema.
        struct FakeMcp;
        impl crate::core::ports::ToolProvider for FakeMcp {
            fn owns(&self, name: &str) -> bool { name.starts_with("mcp__") }
            fn definitions<'a>(&'a self) -> crate::core::ports::BoxFuture<'a, Vec<Value>> {
                Box::pin(std::future::ready(vec![json!({"name": "mcp__fake__tool"})]))
            }
            fn call<'a>(&'a self, name: &'a str, _input: &'a Value,
                        _token: &'a crate::cancel::CancelToken)
                -> crate::core::ports::BoxFuture<'a, String> {
                Box::pin(std::future::ready(format!("mcp:{name}")))
            }
        }
        let router = crate::core::ports::ToolRouter::new(vec![
            Box::new(Builtins),
            Box::new(FakeMcp),
        ]);
        let defs = crate::test_util::block_on(router.definitions());
        let names: Vec<&str> = defs.iter().filter_map(|v| v["name"].as_str()).collect();
        assert_eq!(names, vec!["bash", "read_file", "write_file", "edit_file", "mcp__fake__tool"]);
        // and the built-in half is still the golden bytes
        let builtins = serde_json::to_string(&defs[..4]).unwrap();
        assert_eq!(builtins, GOLDEN_BUILTIN_SCHEMAS);
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
    impl crate::ui::Show for CapturingSink {
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
