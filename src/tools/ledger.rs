// The read ledger: the one strict thing here that pays for itself. Every
// `read_file` records a fingerprint of the bytes it saw; every write records
// what it left behind. Before `edit_file` or `write_file` touches a file,
// that memory is compared against the disk, and a file that moved — the
// user's editor, a formatter, another agent — stops the write. Nobody is
// allowed to write over content they have never seen, because that is how
// someone's uncommitted work disappears.
//
// The ledger keeps **two views** of every file: what *this agent* has seen,
// and what *this process* has seen. A write is checked against the writing
// agent's own view — the bytes on disk must still be the bytes that agent
// last saw or wrote — and falls back to the process's view only for an
// agent that has never touched the file. With one agent the two views hold
// the same value, so the fallback never fires and the ledger behaves
// exactly as it always has.
//
// The check and the write are one critical section, and the bytes the check
// compares are the bytes the write applies to: an edit approved against one
// moment of a file is not applied to another. See `with_ledger`.
//
// It is deliberately *not* a read-before-edit gate. opencode removed theirs,
// and the reasons hold here too: the gate burned a turn on every workflow
// that read a file through bash, and it protected nothing — the edit still
// writes against content the tool just loaded. A fingerprint guards the real
// hazard (the file moved under us) without punishing the common case (never
// read it, go ahead). The honest limit: a file jingwei has never seen has
// nothing to compare against, and is written as before.

use crate::core::ids::AgentId;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// What the ledger knows. Two views of every file: what this process has
/// seen, and what each agent has seen.
pub(crate) struct Ledger {
    /// (agent, path) → the fingerprint of the bytes that agent last saw or
    /// wrote. This is the view a write is checked against.
    seen: HashMap<(AgentId, PathBuf), u64>,
    /// path → the fingerprint of the bytes this process last saw or wrote.
    /// The fallback for a writer that has never touched the file.
    any: HashMap<PathBuf, u64>,
}

impl Ledger {
    /// Record the bytes an agent has just seen or written.
    ///
    /// Both views move: the agent's own, and the process's. The process's
    /// view is the fallback for a writer that has never touched the file,
    /// so it has to track whatever any agent last saw.
    pub(crate) fn note(&mut self, agent: AgentId, p: &str, bytes: &[u8]) {
        let key = ledger_key(p);
        let fp = fingerprint(bytes);
        self.seen.insert((agent, key.clone()), fp);
        self.any.insert(key, fp);
    }

    /// The fingerprint a write is checked against: the agent's own view of
    /// the file, or the process's, for an agent that has never touched it.
    fn view(&self, agent: AgentId, key: &PathBuf) -> Option<u64> {
        self.seen.get(&(agent, key.clone())).or_else(|| self.any.get(key)).copied()
    }

    /// Are these the bytes *this agent* last saw? `None` — never seen, so
    /// not ours to judge; `Some(error)` — they are not the bytes we saw,
    /// and the message says what to do about it.
    ///
    /// The bytes are the ones the caller is about to write, so the check
    /// approves exactly what the write applies to. The agent's own view
    /// wins. The process's view is the fallback, for the writer that has
    /// never touched the file at all.
    pub(crate) fn stale_bytes(&self, agent: AgentId, p: &str, bytes: &[u8]) -> Option<String> {
        let key = ledger_key(p);
        let seen = self.view(agent, &key)?;
        (fingerprint(bytes) != seen).then(|| refusal(p))
    }

    /// `stale_bytes`, against what is on the disk right now: the check for
    /// a write of *new* content, where there is no buffer to compare. The
    /// read happens under the lock, with the write that follows it.
    pub(crate) fn stale(&self, agent: AgentId, p: &str) -> Option<String> {
        let key = ledger_key(p);
        let seen = self.view(agent, &key)?;
        let now = fs::read(p).ok()?;
        (fingerprint(&now) != seen).then(|| refusal(p))
    }
}

/// The refusal, byte for byte. It is model-visible text, so it is written
/// once here and pinned by a test: one shape, one message. It does not say
/// "another agent" — the mover may be the user's editor.
fn refusal(p: &str) -> String {
    format!("error: {p} changed on disk since jingwei last read or wrote it — re-read it, then make \
             the change again against what it actually says (writing over content you have not seen \
             is how someone's uncommitted work disappears)")
}

/// The ledger, locked. The lock is the discipline; see [`with_ledger`].
fn ledger() -> MutexGuard<'static, Ledger> {
    static LEDGER: OnceLock<Mutex<Ledger>> = OnceLock::new();
    LEDGER
        .get_or_init(|| Mutex::new(Ledger { seen: HashMap::new(), any: HashMap::new() }))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Run `f` with the ledger locked.
///
/// The lock is the ledger's whole discipline, and it says one thing: **the
/// check and the write that follows it belong in one call to this.** A check
/// compares a file against a view of it; a write puts bytes on disk. Between
/// the two there is a window, and a file that moves in that window is written
/// over without anybody finding out. Holding the lock across both closes it:
/// two agents cannot both pass the check and then both write.
///
/// The read belongs inside too, for the same reason — the bytes the check
/// compares must be the bytes the write applies to. A file read of a few KB
/// under a mutex is cheap; the alternative is a check that is not a check.
///
/// The lock is not reentrant — never call `with_ledger` from inside
/// `with_ledger`.
pub(crate) fn with_ledger<R>(f: impl FnOnce(&mut Ledger) -> R) -> R {
    let mut led = ledger();
    f(&mut led)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ids::AgentId;
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
        let key = ledger_key(path.to_str().unwrap());
        with_ledger(|led| {
            assert!(led.any.get(&key).is_none(), "the process has no view of it");
            assert!(led.seen.get(&(AgentId::MAIN, key)).is_none(),
                "and the main agent has no view of it either");
        });
    }

    #[test]
    fn ledger_note_stores_and_recalls_fingerprints() {
        let _g = ledger_lock();
        let path = temp_dir("note").join("f.txt");
        let p = path.to_str().unwrap();
        with_ledger(|led| led.note(AgentId::MAIN, p, b"hello world"));
        with_ledger(|led| {
            let key = ledger_key(p);
            assert_eq!(led.any.get(&key).copied(), Some(fingerprint(b"hello world")),
                "the process remembers the bytes it saw");
            assert_eq!(led.seen.get(&(AgentId::MAIN, key)).copied(), Some(fingerprint(b"hello world")),
                "and so does the agent that saw them");
        });
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
        assert!(with_ledger(|led| led.stale(AgentId::MAIN, p.to_str().unwrap())).is_none());
    }

    #[test]
    fn stale_is_none_for_a_missing_file_even_after_a_read() {
        // a read of a file that doesn't exist leaves nothing in the
        // ledger; a follow-up check finds nothing to compare
        let _g = ledger_lock();
        let dir = temp_dir("stale_missing");
        let p = dir.join("never_existed.txt");
        // the read call here is the one the agent makes
        assert!(crate::tools::dispatch("read_file", &serde_json::json!({"path": p.to_str().unwrap()}), AgentId::MAIN).starts_with("error:"));
        assert!(with_ledger(|led| led.stale(AgentId::MAIN, p.to_str().unwrap())).is_none(),
            "a file we never saw has nothing to check: {:?}",
            with_ledger(|led| led.stale(AgentId::MAIN, p.to_str().unwrap())));
    }

    #[test]
    fn stale_fires_when_the_file_moved_since_we_last_saw_it() {
        let _g = ledger_lock();
        let dir = temp_dir("stale_moved");
        let p = dir.join("f.txt");
        std::fs::write(&p, "first version").unwrap();
        // the ledger now remembers the bytes
        with_ledger(|led| led.note(AgentId::MAIN, p.to_str().unwrap(), b"first version"));
        // the file moves under us
        std::fs::write(&p, "second version").unwrap();
        let msg = with_ledger(|led| led.stale(AgentId::MAIN, p.to_str().unwrap()));
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
        with_ledger(|led| led.note(AgentId::MAIN, p.to_str().unwrap(), b"stable"));
        assert!(with_ledger(|led| led.stale(AgentId::MAIN, p.to_str().unwrap())).is_none(),
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
        with_ledger(|led| led.note(AgentId::MAIN, a.to_str().unwrap(), b"alpha"));
        with_ledger(|led| led.note(AgentId::MAIN, b.to_str().unwrap(), b"beta"));
        with_ledger(|led| {
            assert_eq!(led.any.get(&ledger_key(a.to_str().unwrap())).copied(), Some(fingerprint(b"alpha")));
            assert_eq!(led.any.get(&ledger_key(b.to_str().unwrap())).copied(), Some(fingerprint(b"beta")));
        });
    }

    // ---- two agents, one file ---------------------------------------------

    /// The bug this slice exists to fix: the ledger compared the disk
    /// against the *most recent* entry for the path, so one agent's write
    /// told another agent's check that the file was fine — and the second
    /// write clobbered the first without anybody finding out.
    #[test]
    fn two_agents_cannot_overwrite_each_other() {
        let _g = ledger_lock();
        let dir = temp_dir("two_agents");
        let path = dir.join("f.txt");
        std::fs::write(&path, "original").unwrap();
        let p = path.to_str().unwrap();
        let a = AgentId::next();
        let b = AgentId::next();
        // both agents read the file
        crate::tools::dispatch("read_file", &serde_json::json!({"path": p}), a);
        crate::tools::dispatch("read_file", &serde_json::json!({"path": p}), b);
        // A writes: its own view matches the disk, so the write lands
        let out = crate::tools::dispatch("write_file", &serde_json::json!({"path": p, "content": "A"}), a);
        assert!(out.starts_with("ok:"), "A's write lands: {out}");
        // B writes the same file: the disk is not the bytes B saw
        let out = crate::tools::dispatch("write_file", &serde_json::json!({"path": p, "content": "B"}), b);
        assert!(out.contains("changed on disk"), "B is refused: {out}");
        assert_eq!(std::fs::read_to_string(p).unwrap(), "A", "A's bytes are still there");
    }

    /// An agent's own write moves its own view: the second write is
    /// against the bytes it just wrote, not against the bytes it read.
    #[test]
    fn an_agents_own_write_does_not_stale_itself() {
        let _g = ledger_lock();
        let dir = temp_dir("own_write");
        let path = dir.join("f.txt");
        std::fs::write(&path, "original").unwrap();
        let p = path.to_str().unwrap();
        let a = AgentId::next();
        crate::tools::dispatch("read_file", &serde_json::json!({"path": p}), a);
        let out = crate::tools::dispatch("write_file", &serde_json::json!({"path": p, "content": "v1"}), a);
        assert!(out.starts_with("ok:"), "the first write lands: {out}");
        let out = crate::tools::dispatch("write_file", &serde_json::json!({"path": p, "content": "v2"}), a);
        assert!(out.starts_with("ok:"), "an agent's own write does not stale it: {out}");
        assert_eq!(std::fs::read_to_string(p).unwrap(), "v2");
    }

    /// Two agents, released at the same moment, writing the same file:
    /// exactly one wins. The critical section is what makes that true —
    /// without it both could pass the check and then both write.
    ///
    /// The barrier releases both threads at the same instant; nothing
    /// sleeps. The assertion is deterministic under the fix, because the
    /// lock serializes the two writes, and it fails without the fix,
    /// because then both checks can land before either write.
    #[test]
    fn two_agents_writing_at_once_leave_exactly_one_winner() {
        let _g = ledger_lock();
        let dir = temp_dir("one_winner");
        let path = dir.join("f.txt");
        std::fs::write(&path, "original").unwrap();
        let p = path.to_str().unwrap();
        let a = AgentId::next();
        let b = AgentId::next();
        crate::tools::dispatch("read_file", &serde_json::json!({"path": p}), a);
        crate::tools::dispatch("read_file", &serde_json::json!({"path": p}), b);
        let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = [(a, "A"), (b, "B")]
            .into_iter()
            .map(|(agent, content)| {
                let gate = gate.clone();
                let p = p.to_string();
                std::thread::spawn(move || {
                    gate.wait();
                    crate::tools::dispatch("write_file", &serde_json::json!({"path": p, "content": content}), agent)
                })
            })
            .collect();
        let results: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.starts_with("ok:")).count(), 1,
            "exactly one write lands: {results:?}");
        assert_eq!(results.iter().filter(|r| r.contains("changed on disk")).count(), 1,
            "the other is refused, not ignored: {results:?}");
        let on_disk = std::fs::read_to_string(p).unwrap();
        assert!(on_disk == "A" || on_disk == "B", "the file holds one of the two: {on_disk:?}");
    }

    /// The limit, pinned deliberately: an agent that has never touched a
    /// file is not stopped by the ledger. This is the documented policy —
    /// "never read it, go ahead" — and the per-agent view does not change
    /// it. Pinning it here makes changing it a decision rather than an
    /// accident.
    #[test]
    fn a_writer_that_never_read_the_file_still_writes_it() {
        let _g = ledger_lock();
        let dir = temp_dir("never_read");
        let path = dir.join("f.txt");
        std::fs::write(&path, "original").unwrap();
        let p = path.to_str().unwrap();
        let a = AgentId::next();
        let out = crate::tools::dispatch("write_file", &serde_json::json!({"path": p, "content": "written"}), a);
        assert!(out.starts_with("ok:"), "never read it, go ahead: {out}");
        assert_eq!(std::fs::read_to_string(p).unwrap(), "written");
    }

    /// The refusal is model-visible text: one shape, one message. This
    /// pins the bytes, so a change to them is a decision and not an
    /// accident. (The message does not say "another agent" — the mover
    /// may be the user's editor.)
    #[test]
    fn the_refusal_message_is_byte_identical() {
        let _g = ledger_lock();
        let dir = temp_dir("refusal_bytes");
        let path = dir.join("f.txt");
        std::fs::write(&path, "first").unwrap();
        let p = path.to_str().unwrap();
        let a = AgentId::next();
        crate::tools::dispatch("read_file", &serde_json::json!({"path": p}), a);
        std::fs::write(&path, "second").unwrap();
        let expected = format!("error: {p} changed on disk since jingwei last read or wrote it — re-read it, then make the change again against what it actually says (writing over content you have not seen is how someone's uncommitted work disappears)");
        assert_eq!(with_ledger(|led| led.stale(a, p)), Some(expected));
    }

    /// The check is a comparison against the agent's view, and the bytes it
    /// compares are the ones the caller is about to write. That is what
    /// makes the edit path safe: the bytes the check approves are the bytes
    /// the edit is applied to, not a second read of the same file.
    #[test]
    fn the_check_compares_the_bytes_the_write_applies_to() {
        let _g = ledger_lock();
        let dir = temp_dir("stale_bytes");
        let path = dir.join("f.txt");
        std::fs::write(&path, "first").unwrap();
        let p = path.to_str().unwrap();
        with_ledger(|led| led.note(AgentId::MAIN, p, b"first"));
        with_ledger(|led| {
            assert!(led.stale_bytes(AgentId::MAIN, p, b"first").is_none(),
                "the bytes we saw are the bytes we write: the check approves them");
            let msg = led.stale_bytes(AgentId::MAIN, p, b"second").unwrap();
            assert!(msg.contains("changed on disk"), "other bytes are refused: {msg}");
        });
    }
}
