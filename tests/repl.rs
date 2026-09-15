// Integration test: pipe invalid UTF-8 bytes into the REPL and verify it
// warns instead of crashing.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn repl_survives_invalid_utf8_then_eof() {
    let bin = env!("CARGO_BIN_EXE_jingwei");
    let mut child = Command::new(bin)
        .env("JINGWEI_API_KEY", "dummy")
        .env("JINGWEI_BASE_URL", "http://127.0.0.1:1") // never reached
        .env("JINGWEI_MODEL", "dummy")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jingwei");

    // Invalid UTF-8 followed by a newline, then EOF.
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(&[0xff, 0xfe, 0xfd]).unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.write_all(b"a clean second line\n").unwrap();
    drop(stdin); // close → EOF

    let out = child.wait_with_output().expect("wait");
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(out.status.success(), "non-zero exit {:?}\nstderr: {stderr}", out.status);
    assert!(stderr.contains("readline"), "no readline warning; stderr: {stderr}");
    assert!(
        stderr.contains("Connection refused") || stderr.contains("Connection Failed"),
        "expected network error on second line; stderr: {stderr}"
    );
}
