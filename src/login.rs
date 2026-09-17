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
use crate::settings::{Profile, Settings};
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

/// Print `prompt` with an optional `default` and read a line from
/// `input`. Returns the trimmed value, or the default if the line is
/// empty. An *empty* default and an *empty* line collapse to `None`
/// at the call site — useful for "press Enter to skip".
fn ask(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str, default: Option<&str>) -> Result<String> {
    let suffix = match default {
        Some(d) if !d.is_empty() => format!(" [{d}]"),
        _ => String::new(),
    };
    write!(output, "{prompt}{suffix}: ")?;
    output.flush()?;
    let mut buf = String::new();
    input.read_line(&mut buf).map_err(|e| Error::Msg(format!("read: {e}")))?;
    let line = buf.trim().to_string();
    Ok(if line.is_empty() { default.unwrap_or("").to_string() } else { line })
}

/// Print `prompt` and read a line. Returns `Some(trimmed input)` on
/// non-empty input, `None` when the user pressed Enter (the "skip"
/// gesture). The optional `hint` is shown in brackets for context
/// ("base URL [https://api.x]") but does not auto-fill.
fn ask_optional(output: &mut dyn Write, input: &mut dyn BufRead, prompt: &str, hint: Option<&str>) -> Result<Option<String>> {
    let suffix = match hint {
        Some(h) if !h.is_empty() => format!(" [{h}]"),
        _ => String::new(),
    };
    write!(output, "{prompt}{suffix}: ")?;
    output.flush()?;
    let mut buf = String::new();
    input.read_line(&mut buf).map_err(|e| Error::Msg(format!("read: {e}")))?;
    let line = buf.trim().to_string();
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
        if unsafe { turn_echo_off(fd) }.is_ok() {
            let mut buf = String::new();
            let res = input.read_line(&mut buf);
            let _ = unsafe { turn_echo_on(fd) };
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
    write!(output, "{prompt}: ")?;
    output.flush()?;
    let mut buf = String::new();
    input.read_line(&mut buf).map_err(|e| Error::Msg(format!("read: {e}")))?;
    Ok(buf.trim().to_string())
}

#[cfg(all(unix, not(test)))]
unsafe fn turn_echo_off(fd: std::os::unix::io::RawFd) -> std::io::Result<()> {
    // termios magic numbers — ECHO is bit 0o10 on the local-mode word
    extern "C" { fn tcgetattr(fd: i32, termios: *mut Termios) -> i32; fn tcsetattr(fd: i32, when: i32, termios: *const Termios) -> i32; }
    static TCSAFLUSH: i32 = 2;
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Termios {
        c_iflag: u32, c_oflag: u32, c_cflag: u32, c_lflag: u32,
        c_line: u8, c_cc: [u8; 32], c_ispeed: u32, c_ospeed: u32,
    }
    let mut t = Termios::default();
    if tcgetattr(fd, &mut t) != 0 { return Err(std::io::Error::last_os_error()); }
    t.c_lflag &= !0o10u32; // clear ECHO
    if tcsetattr(fd, TCSAFLUSH, &t) != 0 { return Err(std::io::Error::last_os_error()); }
    Ok(())
}

#[cfg(all(unix, not(test)))]
unsafe fn turn_echo_on(fd: std::os::unix::io::RawFd) -> std::io::Result<()> {
    extern "C" { fn tcgetattr(fd: i32, termios: *mut Termios) -> i32; fn tcsetattr(fd: i32, when: i32, termios: *const Termios) -> i32; }
    static TCSAFLUSH: i32 = 2;
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Termios {
        c_iflag: u32, c_oflag: u32, c_cflag: u32, c_lflag: u32,
        c_line: u8, c_cc: [u8; 32], c_ispeed: u32, c_ospeed: u32,
    }
    let mut t = Termios::default();
    if tcgetattr(fd, &mut t) != 0 { return Err(std::io::Error::last_os_error()); }
    t.c_lflag |= 0o10u32; // set ECHO
    if tcsetattr(fd, TCSAFLUSH, &t) != 0 { return Err(std::io::Error::last_os_error()); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Drive the wizard against an in-memory script of inputs.
    fn run_wizard_against(inputs: &[&str]) -> (Result<String>, Settings) {
        let joined: String = inputs.iter().flat_map(|s| s.bytes().chain(std::iter::once(b'\n'))).map(|b| b as char).collect();
        let mut input = Cursor::new(joined.into_bytes());
        let mut output: Vec<u8> = Vec::new();
        let mut settings = Settings::empty();
        let r = wizard(&mut input, &mut output, &mut settings);
        eprintln!("--- wizard output ---\n{}\n---", String::from_utf8_lossy(&output));
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
}
