// Integration: the REPL persists its conversation under its project's
// directory; --list shows this project's; -c resumes this project's newest
// (strictly); an unknown --resume is an error that names the fix.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn temp_home(tag: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    d.push(format!("jingwei_sessions_it_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// jingwei with a sandboxed home and a dead endpoint: config parses, no
/// request can succeed — exactly what persistence tests want.
fn jingwei(home: &Path) -> Command {
    let bin = env!("CARGO_BIN_EXE_jingwei");
    let mut c = Command::new(bin);
    // both spellings, so the sandbox holds on every platform
    c.env("HOME", home).env("USERPROFILE", home)
        .env("JINGWEI_API_KEY", "dummy")
        .env("JINGWEI_BASE_URL", "http://127.0.0.1:1")
        .env("JINGWEI_MODEL", "it-model");
    c
}

/// A jingwei invocation: sandboxed home, given cwd, given args.
fn cmd(home: &Path, cwd: &Path, args: &[&str]) -> Command {
    let mut c = jingwei(home);
    c.current_dir(cwd);
    for a in args {
        c.arg(a);
    }
    c
}

/// Pipe lines into an interactive (plain-frontend) run and collect output.
fn pipe(mut c: Command, lines: &str) -> Output {
    let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(lines.as_bytes()).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn repl_persists_list_shows_and_continue_resumes() {
    let home = temp_home("flow");

    // 1. an interactive run: the task fails against the dead endpoint, but
    //    the task itself is already persisted — under this project's dir
    let out = pipe(cmd(&home, &home, &[]), "hello sessions\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exit {:?}\nstdout: {stdout}", out.status.code());
    assert!(stdout.contains("session "), "a fresh run announces its session: {stdout}");
    assert!(
        stdout.contains(".jingwei/projects/"),
        "the banner shows the project-scoped session path: {stdout}"
    );

    // 2. --list shows this project's sessions — no credentials needed
    let out = cmd(&home, &home, &["--list"]).output().unwrap();
    let list = String::from_utf8_lossy(&out.stdout);
    assert!(list.contains("hello sessions"), "--list shows the first task: {list}");
    assert!(list.contains("it-model"), "--list shows the model: {list}");
    assert!(list.contains("flow"), "the scope footer names this project: {list}");

    // 3. -c in the same project resumes it; /exit leaves before any task
    let out = pipe(cmd(&home, &home, &["-c"]), "/exit\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("resumed session"), "-c announces the resumed session: {stdout}");
    assert!(stdout.contains("1 message"), "the persisted task came back: {stdout}");
    assert!(!stdout.contains("tools now run from"), "same project: no moved-tree note: {stdout}");

    // 4. --list --all mixes projects, with the DIR column to tell them apart
    let other = home.join("other-proj");
    std::fs::create_dir_all(&other).unwrap();
    pipe(cmd(&home, &other, &[]), "a task elsewhere\n");
    let out = cmd(&home, &home, &["--list", "--all"]).output().unwrap();
    let list = String::from_utf8_lossy(&out.stdout);
    assert!(list.contains("hello sessions") && list.contains("a task elsewhere"), "every project: {list}");
    assert!(list.contains("DIR"), "projects mixed: the DIR column appears: {list}");
    assert!(list.contains("other-proj"), "the column names the other project: {list}");
}

#[test]
fn continue_is_strictly_this_projects() {
    let home = temp_home("strict");
    pipe(cmd(&home, &home, &[]), "task from home\n");

    // a different project has no sessions of its own: -c refuses, it does
    // not reach for another project's conversation
    let elsewhere = home.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let out = pipe(cmd(&home, &elsewhere, &["-c"]), "/exit\n");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no sessions yet"),
        "-c is scoped to this project's directory: {stderr}"
    );
    assert!(stderr.contains("projects"), "the error names the project directory: {stderr}");
}

#[test]
fn dash_collision_shares_a_project_and_the_note_fires() {
    // the documented wrinkle of the project-path encoding: `/x-y` and
    // `/x/y` flatten to the same project directory — the session resumes
    // (the layout is the scope), and the header's cwd says the tree moved
    let home = temp_home("collide");
    let with_sep = home.join("x").join("y");
    let with_dash = home.join("x-y");
    std::fs::create_dir_all(&with_sep).unwrap();
    std::fs::create_dir_all(&with_dash).unwrap();
    pipe(cmd(&home, &with_sep, &[]), "task in the slashed tree\n");

    let out = pipe(cmd(&home, &with_dash, &["-c"]), "/exit\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("resumed session"), "the shared encoding is one project: {stdout}");
    assert!(stdout.contains("tools now run from"), "the header's cwd fires the moved-tree note: {stdout}");
}

#[test]
fn resume_unknown_id_is_an_error_naming_the_fix() {
    let home = temp_home("none");
    let out = cmd(&home, &home, &["--resume", "zzz"]).output().unwrap();
    assert!(!out.status.success(), "an unknown id exits non-zero");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no session starts with 'zzz'"), "{err}");
    assert!(err.contains("--list"), "{err}");
}
