// `jingwei login` — configure providers / api keys / active model.
// The wizard is intentionally small and stdin-driven: it does not depend
// on the TUI being available (so it runs through pipes, scripts, ssh
// without a tty, etc.), and the same code path is what a future
// in-REPL `/login` would shell out to.
//
// Today the wizard's surface is "add a profile and make it active" —
// the other moves (list, switch, delete) live in `jingwei login --show`
// and the REPL's `/model`. Adding a profile is the most-common case and
// the only one that *requires* a wizard (the others are answered from
// what's already on disk).

use crate::api::VENDORS;
use crate::config::settings::{Profile, Settings};
use crate::{Error, Result};
use std::io::{self, BufRead, IsTerminal, Write};

/// Run a non-interactive `jingwei login` invocation. `--show` prints
/// the current settings, `--reset` deletes the file, and bare `login`
/// starts the wizard.
pub(crate) fn run(show: bool, reset: bool, reveal: bool) -> Result<()> {
    if reset {
        Settings::reset()?;
        println!("settings.json removed");
        return Ok(());
    }
    if show {
        let s = Settings::load()?;
        print!("{}", s.describe(reveal));
        return Ok(());
    }
    // bare `jingwei login`: the wizard.
    let stdin_tty = io::stdin().is_terminal();
    let stdout_tty = io::stdout().is_terminal();
    if !stdin_tty || !stdout_tty {
        return Err(Error::Msg(
            "jingwei login needs an interactive terminal — neither stdin nor stdout is a tty".into(),
        ));
    }
    let mut settings = Settings::load().unwrap_or_else(|_| Settings::empty());
    let new_key = wizard(&mut io::stdin().lock(), &mut io::stdout().lock(), &mut settings)?;
    settings.active = Some(new_key.clone());
    settings.save()?;
    println!("\nsaved {} to {}", new_key, Settings::path().map(|p| p.display().to_string()).unwrap_or_default());
    println!("run `jingwei` to start a session, or `jingwei login --show` to review");
    Ok(())
}

/// Run the wizard's Q&A loop against the given input / output streams.
/// On success returns the profile key (`<provider>/<model>`) that was
/// added to `settings`. The caller is responsible for setting `active`
/// and saving the file.
fn wizard(
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    settings: &mut Settings,
) -> Result<String> {
    let protocol = pick_protocol(input, output)?;
    let default_model = default_model_for(&protocol).unwrap_or("").to_string();
    let model = ask(output, input, "model", Some(&default_model))?;
    if model.is_empty() {
        return Err(Error::Msg("model cannot be empty (the chosen vendor has no default)".into()));
    }
    let key = format!("{protocol}/{model}");
    let default_base = VENDORS.iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(&protocol))
        .and_then(|(_, p)| p.default_base())
        .unwrap_or("")
        .to_string();
    let base_url = ask_optional(output, input, "base URL (Enter for vendor default)", Some(&default_base))?;
    let api_key = ask_secret_api_key(output, input, "api key")?;
    if api_key.is_empty() {
        return Err(Error::Msg("api key cannot be empty".into()));
    }
    settings.providers.insert(key.clone(), Profile {
        protocol: protocol.to_string(),
        api_key,
        base_url,
    });
    Ok(key)
}

/// Pick a protocol from the VENDORS table by index. The number-keyed
/// list keeps a future addition a no-op for the wizard.
fn pick_protocol(input: &mut dyn BufRead, output: &mut dyn Write) -> Result<String> {
    writeln!(output, "\nprovider:")?;
    for (i, (name, _)) in VENDORS.iter().enumerate() {
        let default = if *name == "minimax" { " (default)" } else { "" };
        writeln!(output, "  {}) {}{}", i + 1, name, default)?;
    }
    loop {
        let raw = ask(output, input, "choice", Some("1"))?;
        if let Some((name, _)) = parse_pick(&raw) {
            return Ok(name.to_string());
        }
        writeln!(output, "  invalid choice — pick a number from the list")?;
    }
}

fn parse_pick(raw: &str) -> Option<(&'static str, crate::api::Protocol)> {
    let raw = raw.trim();
    // accept either the number or the protocol name itself
    if let Ok(n) = raw.parse::<usize>() {
        if n >= 1 && n <= VENDORS.len() {
            return Some(VENDORS[n - 1]);
        }
    }
    VENDORS.iter().find(|(n, _)| n.eq_ignore_ascii_case(raw)).copied()
}

/// The vendor's default model name (the same string the vendor uses
/// when none is given — e.g. minimax → MiniMax-M3).
fn default_model_for(protocol: &str) -> Option<&'static str> {
    parse_pick(protocol).and_then(|(_, p)| p.default_model())
}

/// Print `prompt` (with the bracketed `hint`, when non-empty) and read
/// one line from `input`, trimmed. The two ask-variants decide what an
/// empty line means: a default or a skip.
fn prompt_line(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str, hint: Option<&str>) -> Result<String> {
    let suffix = match hint { Some(h) if !h.is_empty() => format!(" [{h}]"), _ => String::new() };
    write!(output, "{prompt}{suffix}: ")?;
    output.flush()?;
    let mut buf = String::new();
    input.read_line(&mut buf).map_err(|e| Error::Msg(format!("read: {e}")))?;
    Ok(buf.trim().to_string())
}

/// Print `prompt` with an optional `default` and read a line from
/// `input`. Returns the trimmed value, or the default if the line is
/// empty. An *empty* default and an *empty* line collapse to `None`
/// at the call site — useful for "press Enter to skip".
fn ask(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str, default: Option<&str>) -> Result<String> {
    let line = prompt_line(output, input, prompt, default)?;
    Ok(if line.is_empty() { default.unwrap_or("").to_string() } else { line })
}

/// Print `prompt` and read a line. Returns `Some(trimmed input)` on
/// non-empty input, `None` when the user pressed Enter (the "skip"
/// gesture). The optional `hint` is shown in brackets for context
/// ("base URL [https://api.x]") but does not auto-fill.
fn ask_optional(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str, hint: Option<&str>) -> Result<Option<String>> {
    let line = prompt_line(output, input, prompt, hint)?;
    Ok(if line.is_empty() { None } else { Some(line) })
}

/// Like `ask`, but turns off echo on stdin so the API key is not
/// reflected back to the terminal. Reads from `input` (the wizard's
/// `&mut dyn BufRead`, which is a `StdinLock` in production) — going
/// through `io::stdin()` directly would deadlock on stdin's internal
/// `Mutex`: `StdinLock` holds that mutex for its lifetime, and
/// `Stdin::read_line` re-acquires it, so the second acquisition never
/// returns. The termios toggle only needs the fd number, which we
/// pull from `io::stdin().as_raw_fd()` (a constant `STDIN_FILENO` that
/// doesn't touch the mutex). The test path skips the echo toggle
/// entirely — `input` is a `Cursor` over a script of lines, with no
/// real terminal to talk to.
#[cfg(not(test))]
fn ask_secret_api_key(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str) -> Result<String> {
    write!(output, "{prompt}: ")?;
    output.flush().map_err(|e| Error::Msg(format!("flush: {e}")))?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = io::stdin().as_raw_fd();
        // piped / redirected stdin: tcgetattr fails with ENOTTY → fall through
        if unsafe { set_echo(fd, false) }.is_ok() {
            let mut buf = String::new();
            let res = input.read_line(&mut buf);
            let _ = unsafe { set_echo(fd, true) };
            println!();
            return res.map(|_| buf.trim().to_string()).map_err(|e| Error::Msg(format!("read: {e}")));
        }
    }
    let mut buf = String::new();
    input.read_line(&mut buf).map_err(|e| Error::Msg(format!("read: {e}")))?;
    Ok(buf.trim().to_string())
}

#[cfg(test)]
fn ask_secret_api_key(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str) -> Result<String> {
    // In tests we drive the wizard with a Cursor over a script of
    // inputs — the real Stdin is irrelevant. Skip the termios dance
    // entirely and read straight off the supplied `input`.
    prompt_line(output, input, prompt, None)
}

#[cfg(all(unix, not(test)))]
unsafe fn set_echo(fd: std::os::unix::io::RawFd, on: bool) -> std::io::Result<()> {
    // The libc crate's `termios` binding is why `libc` is a dependency at
    // all: it owns the per-platform layout. (The MCP stdio transport's
    // `kill_on_drop` is tokio's, and needs no dependency.) The hand-rolled
    // struct this replaces was 60 bytes where macOS's kernel writes 72
    // (tcflag_t is u64 there, c_cc is [u8; 20]) — tcgetattr ran 12 bytes
    // past the end of the object, so every hidden-key prompt either
    // scrambled the terminal settings or smashed the stack outright:
    // `jingwei login` segfaulted right after the api key was read.
    let mut t: libc::termios = std::mem::zeroed();
    if libc::tcgetattr(fd, &mut t) != 0 { return Err(std::io::Error::last_os_error()); }
    if on { t.c_lflag |= libc::ECHO } else { t.c_lflag &= !libc::ECHO }
    if libc::tcsetattr(fd, libc::TCSAFLUSH, &t) != 0 { return Err(std::io::Error::last_os_error()); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Drive the wizard against an in-memory script of inputs.
    fn run_wizard_against(inputs: &[&str]) -> (Result<String>, Settings) {
        let joined: String = inputs_to_script(inputs);
        let mut input = Cursor::new(joined.into_bytes());
        let mut output: Vec<u8> = Vec::new();
        let mut settings = Settings::empty();
        let r = wizard(&mut input, &mut output, &mut settings);
        // The wizard's prompts are the most informative debug surface when
        // a wizard test fails ("which question did it stop at?"), so keep
        // the capture around. Print on failure only — running seven
        // wizard tests prints seven dialogs otherwise.
        if r.is_err() {
            eprintln!("--- wizard output ---\n{}\n---", String::from_utf8_lossy(&output));
        }
        (r, settings)
    }

    #[test]
    fn wizard_happy_path_minimax_minimax_m3() {
        let (r, s) = run_wizard_against(&["1", "", "", "sk-test"]);
        let key = r.expect("wizard should succeed");
        assert_eq!(key, "minimax/MiniMax-M3");
        let p = &s.providers[&key];
        assert_eq!(p.protocol, "minimax");
        assert_eq!(p.api_key, "sk-test");
        assert!(p.base_url.is_none(), "Enter on base_url means vendor default");
    }

    #[test]
    fn wizard_accepts_protocol_by_name_not_number() {
        let (r, s) = run_wizard_against(&["zai", "", "", "sk-z"]);
        let key = r.expect("wizard accepts the protocol name");
        assert!(key.starts_with("zai/"), "got key {key}");
        assert_eq!(s.providers[&key].api_key, "sk-z");
    }

    #[test]
    fn wizard_rejects_invalid_then_accepts() {
        let (r, _) = run_wizard_against(&["nope", "2", "", "", "sk-d"]);
        assert!(r.is_ok(), "second pick is valid: {r:?}");
    }

    #[test]
    fn wizard_rejects_empty_api_key() {
        // 1 = minimax (default model + base url), then bare-Enter on model,
        // base url, and api key — the empty key must error.
        let (r, _) = run_wizard_against(&["1", "", "", ""]);
        match r {
            Err(e) => assert!(e.to_string().contains("api key"), "got: {e}"),
            Ok(_) => panic!("empty api key must error"),
        }
    }

    #[test]
    fn wizard_explicit_base_url_overrides_vendor_default() {
        let (r, s) = run_wizard_against(&["1", "M3", "https://my-proxy", "sk-x"]);
        let key = r.unwrap();
        assert_eq!(s.providers[&key].base_url.as_deref(), Some("https://my-proxy"));
    }

    #[test]
    fn wizard_explicit_model_replaces_default() {
        let (r, _s) = run_wizard_against(&["zai", "glm-4.6", "", "sk-z"]);
        let key = r.unwrap();
        assert_eq!(key, "zai/glm-4.6");
    }

    #[test]
    fn ask_returns_default_when_input_is_empty() {
        let mut input = Cursor::new(b"\n".to_vec());
        let mut output = Vec::new();
        let got = ask(&mut output, &mut input, "p", Some("default-val")).unwrap();
        assert_eq!(got, "default-val");
    }

    #[test]
    fn ask_returns_trimmed_input_when_nonempty() {
        let mut input = Cursor::new(b"  hello  \n".to_vec());
        let mut output = Vec::new();
        let got = ask(&mut output, &mut input, "p", None).unwrap();
        assert_eq!(got, "hello");
    }

    #[test]
    fn parse_pick_accepts_number_and_name() {
        assert_eq!(parse_pick("1").unwrap().0, "minimax");
        assert_eq!(parse_pick("ZAI").unwrap().0, "zai");
        assert_eq!(parse_pick("DeepSeek").unwrap().0, "deepseek");
        assert!(parse_pick("0").is_none());
        assert!(parse_pick("99").is_none());
        assert!(parse_pick("nope").is_none());
    }

    // ---- the wizard's own failure paths ---------------------------------

    /// A script of answers that breaks on the `fail_at`-th read. The
    /// wizard asks one question per read — provider, model, base url, api
    /// key — so the read number *is* the question number.
    struct BreakingReader {
        lines: Vec<Vec<u8>>,
        buf: Vec<u8>,
        reads: usize,
        fail_at: usize,
    }

    impl BreakingReader {
        fn new(answers: &[&str], fail_at: usize) -> Self {
            Self {
                lines: answers.iter().map(|a| format!("{a}\n").into_bytes()).collect(),
                buf: Vec::new(),
                reads: 0,
                fail_at,
            }
        }
    }

    impl std::io::Read for BreakingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = {
                let available = match self.fill_buf() { Ok(a) => a, Err(e) => return Err(e) };
                let n = available.len().min(buf.len());
                buf[..n].copy_from_slice(&available[..n]);
                n
            };
            self.consume(n);
            Ok(n)
        }
    }

    impl std::io::BufRead for BreakingReader {
        fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
            self.reads += 1;
            if self.reads == self.fail_at {
                return Err(std::io::Error::other("the pipe broke"));
            }
            if self.buf.is_empty() {
                if self.lines.is_empty() { return Ok(&[]); }
                self.buf = self.lines.remove(0);
            }
            Ok(&self.buf)
        }
        fn consume(&mut self, amt: usize) {
            let amt = amt.min(self.buf.len());
            self.buf.drain(..amt);
        }
    }

    /// A writer that fails as soon as its output would carry `needle` —
    /// the terminal that goes away in the middle of a question.
    struct FailingWriter { needle: &'static str, seen: String }

    impl std::io::Write for FailingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.seen.push_str(&String::from_utf8_lossy(buf));
            if self.seen.contains(self.needle) {
                return Err(std::io::Error::other("the terminal went away"));
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }

    /// A writer that takes every byte and fails only at the flush — the
    /// terminal that dies between the question and the answer.
    struct FailingFlush;

    impl std::io::Write for FailingFlush {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> { Ok(buf.len()) }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("the terminal went away"))
        }
    }

    /// Drive the wizard with a reader of our own making.
    fn run_wizard_with<R: std::io::BufRead>(input: &mut R) -> (Result<String>, Settings) {
        let mut output: Vec<u8> = Vec::new();
        let mut settings = Settings::empty();
        let r = wizard(input, &mut output, &mut settings);
        if r.is_err() {
            eprintln!("--- wizard output ---\n{}\n---", String::from_utf8_lossy(&output));
        }
        (r, settings)
    }

    /// Drive the wizard with a writer of our own making.
    fn run_wizard_writing_to<W: std::io::Write>(input: &[&str], output: &mut W) -> Result<String> {
        let joined: String = inputs_to_script(input);
        let mut input = Cursor::new(joined.into_bytes());
        let mut settings = Settings::empty();
        wizard(&mut input, output, &mut settings)
    }

    fn inputs_to_script(inputs: &[&str]) -> String {
        inputs.iter().flat_map(|s| s.bytes().chain(std::iter::once(b'\n'))).map(|b| b as char).collect()
    }

    #[test]
    fn a_vendor_with_no_default_model_cannot_be_chosen() {
        // deepseek names no default model, so the model question offers
        // an empty default and Enter leaves the model empty — the wizard
        // has to say so rather than save a nameless profile.
        let (r, _) = run_wizard_against(&["deepseek", "", "", "sk-d"]);
        match r {
            Err(e) => assert_eq!(e.to_string(), "model cannot be empty (the chosen vendor has no default)"),
            Ok(k) => panic!("deepseek has no default model; the wizard must refuse, got {k}"),
        }
    }

    #[test]
    fn a_broken_pipe_stops_the_wizard_at_the_provider_question() {
        let mut input = BreakingReader::new(&["1"], 1);
        let (r, _) = run_wizard_with(&mut input);
        assert_eq!(r.unwrap_err().to_string(), "read: the pipe broke");
    }

    #[test]
    fn a_broken_pipe_stops_the_wizard_at_the_model_question() {
        let mut input = BreakingReader::new(&["1"], 2);
        let (r, _) = run_wizard_with(&mut input);
        assert_eq!(r.unwrap_err().to_string(), "read: the pipe broke");
    }

    #[test]
    fn a_broken_pipe_stops_the_wizard_at_the_base_url_question() {
        let mut input = BreakingReader::new(&["1"], 3);
        let (r, _) = run_wizard_with(&mut input);
        assert_eq!(r.unwrap_err().to_string(), "read: the pipe broke");
    }

    #[test]
    fn a_broken_pipe_stops_the_wizard_at_the_api_key_question() {
        let mut input = BreakingReader::new(&["1"], 4);
        let (r, _) = run_wizard_with(&mut input);
        assert_eq!(r.unwrap_err().to_string(), "read: the pipe broke");
    }

    #[test]
    fn a_broken_terminal_stops_the_wizard_at_the_provider_list() {
        let mut output = FailingWriter { needle: "provider:", seen: String::new() };
        let e = run_wizard_writing_to(&["1"], &mut output).unwrap_err();
        assert!(matches!(&e, Error::Io(_)), "the failure is the terminal's: {e:?}");
        assert_eq!(e.to_string(), "the terminal went away");
    }

    #[test]
    fn a_broken_terminal_stops_the_wizard_at_the_vendor_list() {
        let mut output = FailingWriter { needle: "zai", seen: String::new() };
        let e = run_wizard_writing_to(&["1"], &mut output).unwrap_err();
        assert!(matches!(&e, Error::Io(_)), "the failure is the terminal's: {e:?}");
        assert_eq!(e.to_string(), "the terminal went away");
    }

    #[test]
    fn a_broken_terminal_stops_the_wizard_at_the_invalid_choice_notice() {
        let mut output = FailingWriter { needle: "invalid choice", seen: String::new() };
        let e = run_wizard_writing_to(&["nope"], &mut output).unwrap_err();
        assert!(matches!(&e, Error::Io(_)), "the failure is the terminal's: {e:?}");
        assert_eq!(e.to_string(), "the terminal went away");
    }

    #[test]
    fn a_broken_terminal_stops_the_wizard_at_the_question_itself() {
        // the question never reaches the screen, so no answer can come
        let mut output = FailingWriter { needle: "choice", seen: String::new() };
        let e = run_wizard_writing_to(&["1"], &mut output).unwrap_err();
        assert!(matches!(&e, Error::Io(_)), "the failure is the terminal's: {e:?}");
        assert_eq!(e.to_string(), "the terminal went away");
    }

    #[test]
    fn a_broken_terminal_stops_the_wizard_at_the_prompt_flush() {
        // the question is written, but the terminal dies before the
        // answer is read
        let mut output = FailingFlush;
        let e = run_wizard_writing_to(&["1"], &mut output).unwrap_err();
        assert!(matches!(&e, Error::Io(_)), "the failure is the terminal's: {e:?}");
        assert_eq!(e.to_string(), "the terminal went away");
    }

    // ---- the login command ----------------------------------------------

    /// Point `$HOME` at a temp dir for the length of a test. The settings
    /// file lives under the home, so the command's own effects are only
    /// observable against a home we own.
    struct HomeGuard { prev: Option<String> }
    impl HomeGuard {
        fn set(dir: &std::path::Path) -> Self {
            let prev = std::env::var("HOME").ok();
            std::env::set_var("HOME", dir);
            Self { prev }
        }
    }
    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn the_login_command_removes_the_settings_file() {
        let _lock = crate::test_util::env_lock();
        let dir = crate::test_util::temp_dir("login_reset");
        let _home = HomeGuard::set(&dir);
        let path = dir.join(".jingwei").join("settings.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{}").unwrap();
        run(false, true, false).expect("the reset succeeds");
        assert!(!path.exists(), "the settings file is gone");
    }

    #[test]
    fn the_login_command_reports_a_settings_file_that_is_not_there() {
        let _lock = crate::test_util::env_lock();
        let dir = crate::test_util::temp_dir("login_reset_missing");
        let _home = HomeGuard::set(&dir);
        // nothing on disk: the reset has nothing to remove, and says so
        let e = run(false, true, false).unwrap_err().to_string();
        assert_eq!(e, "settings.json not found — nothing to reset");
    }

    /// The show arm prints the settings to stdout, and libtest swallows a
    /// test's own stdout — so the printing happens in a child process
    /// (this same test binary, told apart by an env marker) whose stdout
    /// is a pipe the parent reads.
    const SHOW_MARKER: &str = "JINGWEI_TEST_LOGIN_SHOW";

    #[test]
    #[ignore = "runs only inside the child process the show test spawns"]
    fn the_login_show_probe() {
        if std::env::var_os(SHOW_MARKER).is_none() { return; }
        run(true, false, false).expect("the settings show");
        std::io::stdout().flush().expect("the settings are flushed out");
    }

    #[test]
    fn the_login_command_shows_the_settings() {
        let dir = crate::test_util::temp_dir("login_show");
        let path = dir.join(".jingwei").join("settings.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{"$schema_version":2,"active":"minimax/M","providers":{}}"#).unwrap();
        // the child gets a home of its own, so the settings it shows are
        // the ones this test wrote
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "config::login::tests::the_login_show_probe", "--ignored", "--nocapture"])
            .env(SHOW_MARKER, "1")
            .env(if cfg!(windows) { "USERPROFILE" } else { "HOME" }, &dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .expect("the child runs");
        let text = format!("{}{}",
            String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success(), "the child ran: {text}");
        assert!(text.contains("$schema_version: 2"), "the settings are shown: {text}");
        assert!(text.contains("active: minimax/M"), "the active profile is named: {text}");
        assert!(text.contains("providers: (none)"), "and the profiles are counted: {text}");
    }

    #[test]
    fn the_login_command_fails_when_the_settings_file_is_broken() {
        // the show arm reads the file: a broken one is the visible proof
        let _lock = crate::test_util::env_lock();
        let dir = crate::test_util::temp_dir("login_show_broken");
        let _home = HomeGuard::set(&dir);
        let path = dir.join(".jingwei").join("settings.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "not json at all").unwrap();
        let e = run(true, false, false).unwrap_err().to_string();
        assert!(e.starts_with("settings.json:"), "got: {e}");
    }

    /// The wizard arm of `run` needs a terminal on *both* ends; a unit
    /// test has neither, and the moment one of them is a terminal the
    /// wizard would open and block on the real stdin. So the gate is
    /// checked in a child process whose stdin and stdout are pipes — the
    /// same test binary, told apart by an env marker.
    const GATE_MARKER: &str = "JINGWEI_TEST_LOGIN_GATE";

    #[test]
    #[ignore = "runs only inside the child process the gate test spawns"]
    fn the_login_gate_probe() {
        if std::env::var_os(GATE_MARKER).is_none() { return; }
        let e = run(false, false, false).unwrap_err();
        eprintln!("login gate: {e}");
    }

    #[test]
    fn the_login_command_needs_a_terminal() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "config::login::tests::the_login_gate_probe", "--ignored", "--nocapture"])
            .env(GATE_MARKER, "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .expect("the child runs");
        let text = format!("{}{}",
            String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success(), "the child ran: {text}");
        assert!(text.contains("jingwei login needs an interactive terminal"),
            "the gate says why: {text}");
    }

    // ---- the wizard, behind a terminal ----------------------------------

    /// The gate above is only half the story: the wizard behind it needs
    /// a terminal on *both* ends, and it reads the process's own stdin.
    /// So the wizard is reached by handing a **child process** a pty —
    /// the child is this same test binary, told apart by an env
    /// marker, and its stdin/stdout are the two ends of a pseudo
    /// terminal. `libc` is already a dependency (the crate uses it for
    /// the terminal's own settings), so a pty costs no new one.
    #[cfg(unix)]
    struct Pty {
        /// the child's three standard streams — the same end of the pty
        streams: (std::process::Stdio, std::process::Stdio, std::process::Stdio),
        /// the parent's end: the answers go in, the child's output comes out
        master: std::os::unix::io::RawFd,
        /// the parent's own copy of the child's end. It has to be closed
        /// once the child is spawned, or the terminal never hangs up and
        /// the parent waits for output that will never end.
        slave: std::os::unix::io::RawFd,
    }

    #[cfg(unix)]
    impl Pty {
        fn open() -> Pty {
            use std::os::unix::io::FromRawFd;
            unsafe {
                let mut master: libc::c_int = -1;
                let mut slave: libc::c_int = -1;
                let mut size: libc::winsize = std::mem::zeroed();
                size.ws_row = 40;
                size.ws_col = 120;
                let rc = libc::openpty(
                    &mut master, &mut slave,
                    std::ptr::null_mut(), // the pty has no name
                    std::ptr::null_mut(), // and keeps the default settings
                    &mut size,
                );
                assert_eq!(rc, 0, "a pty opens: {}", std::io::Error::last_os_error());
                // the child should not keep either end
                libc::fcntl(master, libc::F_SETFD, libc::FD_CLOEXEC);
                libc::fcntl(slave, libc::F_SETFD, libc::FD_CLOEXEC);
                // the child's three standard streams are the same end of
                // the pty, each on its own descriptor
                let end = || {
                    let fd = libc::dup(slave);
                    assert!(fd >= 0, "the pty end is duplicated");
                    std::process::Stdio::from(std::fs::File::from_raw_fd(fd))
                };
                Pty { streams: (end(), end(), end()), master, slave }
            }
        }
    }

    #[cfg(unix)]
    const WIZARD_MARKER: &str = "JINGWEI_TEST_LOGIN_WIZARD";

    #[cfg(unix)]
    #[test]
    #[ignore = "runs only inside the child process the pty test spawns"]
    fn the_login_wizard_probe() {
        if std::env::var_os(WIZARD_MARKER).is_none() { return; }
        match run(false, false, false) {
            Ok(()) => eprintln!("login wizard: ok"),
            Err(e) => eprintln!("login wizard: {e:?}"),
        }
    }

    /// Run `jingwei login` in a child process that has a terminal on
    /// both ends, hand it the given answers, and give back what it
    /// printed. The child is this same test binary, told apart by an
    /// env marker; its stdin and stdout are the two ends of a pseudo
    /// terminal, so the tty gate opens and the wizard runs for real.
    ///
    /// "For real" except for the secret prompt. The child is a *test*
    /// binary, so it runs the `#[cfg(test)]` variant of
    /// `ask_secret_api_key` — the one that reads the answer straight
    /// off the input and skips the termios echo toggle. A test build
    /// cannot reach the `#[cfg(not(test))]` code at all, so the echo
    /// toggle and `set_echo` are the two things here that no unit test
    /// executes. That is the price of the echo toggle being the code
    /// that used to segfault `jingwei login` right after the key was
    /// read: it is the one piece of the wizard that only the real
    /// binary has.
    #[cfg(unix)]
    fn login_in_a_terminal(home: &std::path::Path, answers: &[u8]) -> String {
        use std::io::{Read, Write};
        use std::os::unix::io::FromRawFd;

        let Pty { streams: (stdin, stdout, stderr), master, slave } = Pty::open();
        let mut master = unsafe { std::fs::File::from_raw_fd(master) };
        let mut reader = master.try_clone().expect("the parent end clones");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "config::login::tests::the_login_wizard_probe", "--ignored", "--nocapture"])
            .env("HOME", home)
            .env(WIZARD_MARKER, "1")
            .stdin(stdin)
            .stdout(stdout)
            .stderr(stderr)
            .spawn()
            .expect("the child runs");
        // from here on the child is the only one holding the terminal's
        // far end, so the terminal hangs up when the child is done
        unsafe { libc::close(slave) };
        // the answers, in the order the wizard asks them: the provider
        // (1 = minimax), the model (Enter takes the vendor's default),
        // the base URL (Enter takes the vendor's default), the key. The
        // parent's end stays open, so the terminal never sees an
        // end-of-file that would throw the answers away.
        master.write_all(answers).expect("the answers reach the terminal");
        let seen = std::thread::spawn(move || {
            let mut out = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                match reader.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => out.extend_from_slice(&chunk[..n]),
                }
            }
            String::from_utf8_lossy(&out).into_owned()
        });
        // the wizard asks four questions and four answers are on the
        // terminal, so it finishes; the deadline is here so that a
        // wizard asking one question more fails the test instead of
        // hanging the suite
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        let status = loop {
            match child.try_wait().expect("the child is still ours") {
                Some(status) => break status,
                None if std::time::Instant::now() > deadline => {
                    let _ = child.kill();
                    panic!("the wizard never finished");
                }
                None => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        };
        let text = seen.join().expect("the reader finishes");
        assert!(status.success(), "the wizard ran: {text}");
        text
    }

    #[cfg(unix)]
    #[test]
    fn the_login_command_saves_the_profile_it_collected() {
        let home = crate::test_util::temp_dir("login_wizard");
        let text = login_in_a_terminal(&home, b"1\n\n\nsk-test\n");
        assert!(text.contains("saved minimax/MiniMax-M3 to"), "the profile was saved: {text}");
        let saved = std::fs::read_to_string(home.join(".jingwei").join("settings.json"))
            .expect("the settings file is there");
        assert!(saved.contains("sk-test"), "the key was kept: {saved}");
        assert!(saved.contains("minimax/MiniMax-M3"), "the profile is named after it: {saved}");
    }

    #[cfg(unix)]
    #[test]
    fn the_login_command_reports_a_wizard_that_collected_no_key() {
        let home = crate::test_util::temp_dir("login_wizard_nokey");
        // the last answer is the api key, and it is empty
        let text = login_in_a_terminal(&home, b"1\n\n\n\n");
        assert!(text.contains("api key cannot be empty"), "the wizard says why: {text}");
    }

    #[cfg(unix)]
    #[test]
    fn the_login_command_starts_over_when_the_settings_file_cannot_be_read() {
        let home = crate::test_util::temp_dir("login_wizard_broken");
        // a settings file that is not json: the wizard starts from an
        // empty set of profiles instead of refusing to start
        std::fs::create_dir_all(home.join(".jingwei")).unwrap();
        std::fs::write(home.join(".jingwei").join("settings.json"), "not json at all").unwrap();
        let text = login_in_a_terminal(&home, b"1\n\n\n\n");
        assert!(text.contains("api key cannot be empty"), "the wizard asked its questions: {text}");
    }

    #[cfg(unix)]
    #[test]
    fn the_login_command_reports_a_home_it_cannot_save_to() {
        let home = crate::test_util::temp_dir("login_wizard_unsaveable");
        // ".jingwei" is a file, so the settings cannot be saved there
        std::fs::write(home.join(".jingwei"), "not a directory").unwrap();
        let text = login_in_a_terminal(&home, b"1\n\n\nsk-test\n");
        assert!(text.contains("AlreadyExists"), "the save says why: {text}");
    }
}
