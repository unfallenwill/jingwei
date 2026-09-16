// Integration test: pipe invalid UTF-8 bytes into the REPL and verify it
// warns instead of crashing.

use std::io::Write;
use std::process::{Command, Stdio};

fn spawn_repl() -> std::process::Child {
    let bin = env!("CARGO_BIN_EXE_jingwei");
    Command::new(bin)
        .env("JINGWEI_API_KEY", "dummy")
        .env("JINGWEI_BASE_URL", "http://127.0.0.1:1") // never reached
        .env("JINGWEI_MODEL", "dummy")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jingwei")
}

#[test]
fn repl_survives_invalid_utf8_then_eof() {
    let mut child = spawn_repl();

    // Invalid UTF-8 followed by a newline, then EOF.
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(&[0xff, 0xfe, 0xfd]).unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.write_all(b"a clean second line\n").unwrap();
    drop(stdin); // close → EOF

    let out = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(out.status.success(), "non-zero exit {:?}\nstderr: {stderr}", out.status);
    assert!(
        stderr.contains("warning: input"),
        "invalid UTF-8 should warn, not crash; stderr: {stderr}"
    );
    assert!(
        stderr.contains("Connection refused") || stderr.contains("Connection Failed"),
        "expected network error on second line; stderr: {stderr}"
    );
}

#[test]
fn repl_exit_command_quits_before_the_next_line() {
    let mut child = spawn_repl();

    // `/exit` on the first line: the session ends before the second line
    // is ever read — so it is never attempted as a task.
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"/exit\n").unwrap();
    stdin.write_all(b"this line must never run\n").unwrap();
    drop(stdin);

    let out = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "non-zero exit {:?}\nstderr: {stderr}", out.status);
    assert!(
        !stderr.contains("Connection refused") && !stderr.contains("Connection Failed"),
        "/exit quits before the next line becomes a task; stderr: {stderr}"
    );
}
