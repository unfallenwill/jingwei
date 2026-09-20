// The read ledger: the one strict thing here that pays for itself. Every
// `read_file` records a fingerprint of the bytes it saw; every write records
// what it left behind. Before `edit_file` or `write_file` touches a file,
// that memory is compared against the disk, and a file that moved — the
// user's editor, a formatter, another agent — stops the write. Nobody is
// allowed to write over content they have never seen, because that is how
// someone's uncommitted work disappears.
//
// It is deliberately *not* a read-before-edit gate. opencode removed theirs,
// and the reasons hold here too: the gate burned a turn on every workflow
// that read a file through bash, and it protected nothing — the edit still
// writes against content the tool just loaded. A fingerprint guards the real
// hazard (the file moved under us) without punishing the common case (never
// read it, go ahead). The honest limit: a file jingwei has never seen has
// nothing to compare against, and is written as before.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// The ledger: path → fingerprint of the content last seen there by this
/// process. A session is long-lived and one file is re-read rarely, so this
/// stays small; it never leaves the process.
pub(crate) fn ledger() -> MutexGuard<'static, HashMap<PathBuf, u64>> {
    static LEDGER: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner())
}

/// FNV-1a over the bytes: not a security hash, a change detector — cheap, no
/// dependency, and a file that moved reads as different.
pub(crate) fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes { h = (h ^ *b as u64).wrapping_mul(0x0000_0100_0000_01b3); }
    h
}

/// One key per file, however the model spelled the path: the parent resolved
/// (so `./a.rs`, `a.rs` and `/abs/a.rs` agree) with the name kept as written.
/// An unresolvable parent — a directory that does not exist yet — falls back to
/// the path as given.
pub(crate) fn ledger_key(p: &str) -> PathBuf {
    let path = Path::new(p);
    let resolve = |d: &Path| fs::canonicalize(d).unwrap_or_else(|_| d.to_path_buf());
    match path.parent().filter(|d| !d.as_os_str().is_empty()) {
        Some(dir) => resolve(dir).join(path.file_name().unwrap_or_default()),
        None => resolve(path),
    }
}

pub(crate) fn ledger_note(p: &str, bytes: &[u8]) {
    ledger().insert(ledger_key(p), fingerprint(bytes));
}

/// Has this file moved since we last saw it? `None` — never seen, or not
/// readable, so not ours to judge; `Some(error)` — the bytes are not the bytes
/// we read, and the message says what to do about it.
pub(crate) fn stale(p: &str) -> Option<String> {
    let seen = *ledger().get(&ledger_key(p))?;
    let now = fs::read(p).ok()?;
    (fingerprint(&now) != seen).then(|| format!(
        "error: {p} changed on disk since jingwei last read or wrote it — re-read it, then make \
         the change again against what it actually says (writing over content you have not seen \
         is how someone's uncommitted work disappears)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::temp_dir;

    /// The ledger is process-global; tests that touch it must serialize
    /// with each other so one test's ledger entry does not bleed into
    /// another's path lookup.
    fn ledger_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_util::env_lock()
    }

    #[test]
    fn ledger_default_is_empty() {
        let _g = ledger_lock();
        // a path we know nothing about: the ledger has never seen it
        let path = temp_dir("fresh").join("never_seen.txt");
        assert!(ledger().get(&path).is_none());
    }

    #[test]
    fn ledger_note_stores_and_recalls_fingerprints() {
        let _g = ledger_lock();
        let path = temp_dir("note").join("f.txt");
        ledger_note(path.to_str().unwrap(), b"hello world");
        let got = ledger().get(&ledger_key(path.to_str().unwrap())).copied();
        assert!(got.is_some(), "the path now maps to a fingerprint");
        assert_eq!(got.unwrap(), fingerprint(b"hello world"));
    }

    #[test]
    fn fingerprint_distinguishes_changes() {
        let a = fingerprint(b"hello");
        let b = fingerprint(b"world");
        assert_ne!(a, b, "different bytes yield different fingerprints");
        // same bytes: same fingerprint (the test this guards)
        assert_eq!(fingerprint(b"x"), fingerprint(b"x"));
    }

    #[test]
    fn ledger_key_resolves_a_relative_path_against_cwd() {
        // `./a.rs` joins the resolved cwd with the file name
        let k = ledger_key("./f.txt");
        let cwd = std::env::current_dir().unwrap();
        assert!(k.starts_with(&cwd), "./a path resolves under cwd: {k:?}");
    }

    #[test]
    fn ledger_key_keeps_the_filename_as_written_when_no_parent() {
        // a bare name: the parent is empty, falls back to the path as given
        let k = ledger_key("file.rs");
        assert!(k.ends_with("file.rs"), "the file name is preserved: {k:?}");
    }

    #[test]
    fn ledger_key_canonicalizes_an_existing_absolute_parent() {
        // an absolute path's parent canonicalizes to itself; the file
        // name is kept as written
        let dir = temp_dir("absolute");
        let k = ledger_key(dir.join("f.txt").to_str().unwrap());
        assert!(k.starts_with(dir.canonicalize().unwrap().to_str().unwrap()), "abs path: {k:?}");
    }

    #[test]
    fn stale_is_none_for_a_file_we_have_never_seen() {
        let _g = ledger_lock();
        let dir = temp_dir("stale_fresh");
        let p = dir.join("fresh.txt");
        std::fs::write(&p, "v").unwrap();
        // the ledger has no entry for this path
        assert!(stale(p.to_str().unwrap()).is_none());
    }

    #[test]
    fn stale_is_none_for_a_missing_file_even_after_a_read() {
        // a read of a file that doesn't exist leaves nothing in the
        // ledger; a follow-up check finds nothing to compare
        let _g = ledger_lock();
        let dir = temp_dir("stale_missing");
        let p = dir.join("never_existed.txt");
        // the read call here is the one the agent makes
        assert!(crate::tools::dispatch("read_file", &serde_json::json!({"path": p.to_str().unwrap()})).starts_with("error:"));
        assert!(stale(p.to_str().unwrap()).is_none(),
            "a file we never saw has nothing to check: {:?}",
            stale(p.to_str().unwrap()));
    }

    #[test]
    fn stale_fires_when_the_file_moved_since_we_last_saw_it() {
        let _g = ledger_lock();
        let dir = temp_dir("stale_moved");
        let p = dir.join("f.txt");
        std::fs::write(&p, "first version").unwrap();
        // the ledger now remembers the bytes
        ledger_note(p.to_str().unwrap(), b"first version");
        // the file moves under us
        std::fs::write(&p, "second version").unwrap();
        let msg = stale(p.to_str().unwrap());
        assert!(msg.is_some(), "the file changed: {msg:?}");
        let s = msg.unwrap();
        assert!(s.starts_with("error:"));
        assert!(s.contains(p.to_str().unwrap()));
    }

    #[test]
    fn stale_silent_when_nothing_changed() {
        let _g = ledger_lock();
        let dir = temp_dir("stale_same");
        let p = dir.join("f.txt");
        std::fs::write(&p, "stable").unwrap();
        ledger_note(p.to_str().unwrap(), b"stable");
        assert!(stale(p.to_str().unwrap()).is_none(),
            "the bytes match what we read: stable has not moved");
    }

    #[test]
    fn ledger_keeps_two_different_files_independent() {
        let _g = ledger_lock();
        let dir = temp_dir("independent");
        let a = dir.join("a.txt");
        let b = dir.join("b.txt");
        std::fs::write(&a, "alpha").unwrap();
        std::fs::write(&b, "beta").unwrap();
        ledger_note(a.to_str().unwrap(), b"alpha");
        ledger_note(b.to_str().unwrap(), b"beta");
        assert_eq!(ledger().get(&ledger_key(a.to_str().unwrap())).copied(), Some(fingerprint(b"alpha")));
        assert_eq!(ledger().get(&ledger_key(b.to_str().unwrap())).copied(), Some(fingerprint(b"beta")));
    }
}
