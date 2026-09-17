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
