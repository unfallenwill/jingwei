//! Session persistence — the conversation outliving the process.
//!
//! A session is one JSONL file under `~/.jingwei/projects/<project>/<id>.jsonl`
//! — one subdirectory per project: the working
//! directory's canonical path flattened to `-` is the physical index, so
//! `-c` means "this project's newest" without any filtering. The scheme's
//! known wrinkle is accepted deliberately: a literal `-` in a
//! directory name can collide with a separator (`/a-b` and `/a/b` share an
//! encoding) — the header's `cwd` keeps the truth, and the
//! resumed-elsewhere note fires on the rare clash. Deep paths fuse to a
//! readable prefix plus a short hash, staying under filesystem limits. The
//! first line
//! is a header (id, start time, payload format version, model, protocol,
//! endpoint, working directory — provenance for `--list`, not a contract),
//! and every following line is one message of the internal history, verbatim.
//!
//! Division of labor — this is the whole design:
//!
//! - The **agent core owns the payload**: the internal history shape and
//!   every rule that makes it sendable (tool_use/result pairing, thinking
//!   continuity). This module never interprets it — a line is one JSON
//!   value; the only corruption it defends against is a *torn tail* (a
//!   write killed mid-line), which is framing damage and shows up as a
//!   parse failure.
//! - The **session owns the framing**: header, one-value-per-line,
//!   append/rewrite, the [`Session::sync`] counter.
//! - The **shell owns the policy** (which session, resume or fresh,
//!   ephemeral one-shots) and calls [`Convo::persist`] at *run boundaries*
//!   only: nothing inside a run is ever synced, so the file can never end
//!   mid-run (an assistant `tool_use` without its `tool_result` would 400
//!   the next request). A hard kill loses the run in flight, never the
//!   runs before it.
//!
//! Appends carry only the tail; a history that *shrank* — the context trim
//! dropped a tool_use/result pair — rewrites the file whole, atomically
//! (temp file + rename). The payload's version stamp (`format`) belongs to
//! its owner and is embedded here as an opaque integer.
//!
//! The payload round-trips **byte-stably**: serde_json's sorted keys make
//! serialization canonical, so reloaded history rebuilds byte-identical
//! request bodies — the property provider prefix caches live on. Resume
//! with the same binary and the same flags, and the cache stays warm.
//!
//! There is deliberately no relationship to providers: sessions persist
//! the internal representation, which every wire adapter already speaks —
//! a session started on one protocol resumes on another for free.

use crate::display::{self, Msg, Sev, Show};
use crate::ir::Message;
use crate::Error;

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(std::path::PathBuf::from)
}
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Where session files live: `~/.jingwei/projects/` — one subdirectory
/// per project.
pub fn projects_root() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".jingwei").join("projects"))
}

/// The project subdirectory name: the absolute path with every separator
/// (and the Windows drive colon) flattened to `-`, so a
/// `~/.jingwei/projects` listing reads like a path tree. Paths deeper than
/// the fuse limit melt to a readable head plus a short hash of the whole,
/// so the name stays unique and under filesystem limits.
pub fn encode_project(cwd: &str) -> String {
    let flat: String = cwd.chars().map(|c| if matches!(c, '/' | '\\' | ':') { '-' } else { c }).collect();
    if flat.chars().count() <= 120 {
        return if flat.is_empty() { "default".into() } else { flat };
    }
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    cwd.hash(&mut h);
    let head: String = flat.chars().take(100).collect();
    format!("{head}-{:08x}", h.finish() as u32)
}

/// The current project's session directory: the working directory,
/// canonicalized (so `/.`, repeated separators, and symlinks spell one
/// project one way), then encoded.
pub fn project_dir() -> Option<PathBuf> {
    project_dir_of(&current_dir_string())
}

/// [`project_dir`] for an explicit path — the one seam the tests reach.
pub fn project_dir_of(cwd: &str) -> Option<PathBuf> {
    let canonical = fs::canonicalize(cwd)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| cwd.to_string());
    projects_root().map(|root| root.join(encode_project(&canonical)))
}

// ---- the conversation handle ----------------------------------------------

/// The conversation a process runs on: the history the agent core borrows,
/// plus the session that remembers it. The bridge is one call,
/// [`Convo::persist`], made at run boundaries only — the file-never-ends-
/// mid-run invariant is enforced by *who calls when*, not by either side
/// checking (see the module docs).
pub struct Convo {
    /// The internal history — the agent core's own shape, borrowed as `&mut`.
    pub history: Vec<Message>,
    session: Option<Session>,
}

impl Convo {
    /// A conversation that lives in memory only (a one-shot run).
    pub fn ephemeral() -> Self {
        Self { history: vec![], session: None }
    }

    /// A conversation persisted to `session` — fresh, or resumed with its
    /// history loaded from the file.
    pub fn persistent(session: Session, history: Vec<Message>) -> Self {
        Self { history, session: Some(session) }
    }

    /// Land everything not yet on disk. Run boundaries only. A failure
    /// here must never take the agent down: the note says what happened,
    /// the conversation continues in memory.
    pub fn persist(&mut self, sink: &dyn Show) {
        let Some(s) = self.session.as_mut() else { return };
        if let Err(e) = s.sync(&self.history) {
            sink.show(Msg::Note { sev: Sev::Warn, text: format!(" warning: session not saved ({e}) ") });
        }
    }
}

// ---- the session file -------------------------------------------------------

/// One session file: where it lives, its identity, the header line (kept
/// raw so a rewrite reproduces it byte for byte), and how many messages
/// are already on disk.
#[derive(Debug)]
pub struct Session {
    path: Option<PathBuf>,
    id: String,
    created: u64,
    cwd: String,
    model: String,
    header: String,
    persisted: usize,
}

/// Everything `--list` shows about one saved session. `first` comes from a
/// peek, not a contract: display-only, best-effort — if the payload shape
/// ever changes, the column goes blank and nothing breaks.
#[derive(Debug)]
pub struct Meta {
    pub id: String,
    pub created: u64,
    pub cwd: String,
    pub model: String,
    pub protocol: String,
    pub msgs: usize,
    pub first: String,
}

/// The connection facts the header stamps: model, protocol label, endpoint.
/// Opaque strings here — stamped and re-displayed, never interpreted. This
/// module knows the wire protocols exist only as a word in a header; a
/// third one lands without touching it. The composition root translates
/// `Config` into this at the one boundary where both are known.
pub struct Provenance {
    pub model: String,
    pub protocol: String,
    pub base_url: String,
}

impl Session {
    /// A fresh session under `dir` (`None`: no home — persistence quietly
    /// off), identified by its start time. `format` is the payload's
    /// version, owned by the agent core and embedded here as an opaque
    /// integer. The file itself appears only when the first run persists.
    fn create(dir: Option<&Path>, prov: &Provenance, format: u32, created: u64) -> Self {
        let cwd = current_dir_string();
        let id = format!("{}-{}", id_stem(created), rand4());
        let header = json!({
            "type": "session", "id": id, "created": created, "format": format,
            "model": prov.model, "protocol": prov.protocol, "base_url": prov.base_url,
            "cwd": cwd,
        })
        .to_string();
        Self {
            path: dir.map(|d| d.join(format!("{id}.jsonl"))),
            id,
            created,
            cwd,
            model: prov.model.clone(),
            header,
            persisted: 0,
        }
    }

    /// A fresh session in the current project's directory.
    pub fn new(prov: &Provenance, format: u32) -> Self {
        Self::create(project_dir().as_deref(), prov, format, now_secs())
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn created(&self) -> u64 {
        self.created
    }

    /// Where this session's working tree was when it started — provenance
    /// for the "resumed elsewhere" note and the directory filter.
    pub fn cwd(&self) -> &str {
        &self.cwd
    }

    /// The model that wrote this session — provenance for the
    /// model-changed note (a different model means a different request,
    /// and a prefix cache that starts cold).
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Put everything not yet on disk there: append the tail or — when the
    /// history *shrank* below what the file holds (the context trim dropped
    /// tool_use/result pairs) — rewrite the file whole. A same-length trim
    /// (an in-place content cut) appends nothing: the file keeps the fuller
    /// record, and the trim simply happens again on load. An empty tail
    /// writes nothing, so a session that never ran never lands on disk.
    pub fn sync(&mut self, history: &[Message]) -> io::Result<()> {
        if self.path.is_none() {
            return Ok(()); // no home directory: persistence quietly off
        }
        if self.persisted > history.len() {
            return self.rewrite(history);
        }
        let tail = &history[self.persisted..];
        if tail.is_empty() {
            return Ok(());
        }
        if self.persisted == 0 {
            if let Some(p) = self.path.as_deref().and_then(Path::parent) {
                fs::create_dir_all(p)?;
            }
        }
        let mut f = OpenOptions::new().create(true).append(true).open(self.path.as_deref().unwrap())?;
        if self.persisted == 0 {
            writeln!(f, "{}", self.header)?;
        }
        for m in tail {
            writeln!(f, "{}", m.to_value())?;
        }
        f.flush()?;
        self.persisted = history.len();
        Ok(())
    }

    /// Write the whole file anew — temp file, then rename, so a crash never
    /// leaves a half-rewritten session.
    fn rewrite(&mut self, history: &[Message]) -> io::Result<()> {
        let tmp = self.path.as_deref().unwrap().with_extension("tmp");
        if let Some(p) = tmp.parent() {
            fs::create_dir_all(p)?;
        }
        let mut f = fs::File::create(&tmp)?;
        writeln!(f, "{}", self.header)?;
        for m in history {
            writeln!(f, "{}", m.to_value())?;
        }
        f.flush()?;
        fs::rename(&tmp, self.path.as_deref().unwrap())?;
        self.persisted = history.len();
        Ok(())
    }

    /// Resume: the newest session in this project (`None`), or the one
    /// whose id starts with `selector`. Scoping comes from the layout —
    /// the project subdirectory *is* the filter. The returned session
    /// keeps the file's identity — new turns append to it.
    pub fn load(selector: Option<&str>) -> crate::Result<(Session, Vec<Message>)> {
        let dir = project_dir()
            .ok_or_else(|| Error::Msg("no home directory — cannot find ~/.jingwei/projects".into()))?;
        load_in(&dir, selector)
    }

    /// This project's saved sessions, newest first.
    pub fn list() -> crate::Result<Vec<Meta>> {
        let dir = project_dir()
            .ok_or_else(|| Error::Msg("no home directory — cannot find ~/.jingwei/projects".into()))?;
        list_in(&dir)
    }

    /// Every project's sessions, newest first (`--list --all`).
    pub fn list_all() -> crate::Result<Vec<Meta>> {
        let root = projects_root()
            .ok_or_else(|| Error::Msg("no home directory — cannot find ~/.jingwei/projects".into()))?;
        let rd = match fs::read_dir(&root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        let mut all = vec![];
        for proj in rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
            all.extend(list_in(&proj)?);
        }
        all.sort_by(|a, b| (b.created, &b.id).cmp(&(a.created, &a.id))); // newest first, across projects
        Ok(all)
    }
}

// ---- directory scan ---------------------------------------------------------

fn load_in(dir: &Path, selector: Option<&str>) -> crate::Result<(Session, Vec<Message>)> {
    let metas = list_in(dir)?;
    let chosen = match selector {
        Some(s) => {
            let mut hits = metas.iter().filter(|m| m.id.starts_with(s));
            let one = hits.next().ok_or_else(|| {
                Error::Msg(format!("no session starts with '{s}' ({} saved; try --list)", metas.len()))
            })?;
            if let Some(two) = hits.next() {
                return Err(Error::Msg(format!(
                    "'{s}' matches several sessions: {}, {}",
                    one.id, two.id
                )));
            }
            one
        }
        None => metas.first().ok_or_else(|| {
            Error::Msg(format!("no sessions yet in {} — an interactive run creates one", dir.display()))
        })?,
    };
    let path = dir.join(format!("{}.jsonl", chosen.id));
    let text = fs::read_to_string(&path)
        .map_err(|e| Error::Msg(format!("reading {}: {e}", path.display())))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default().to_string();
    let mut history = vec![];
    for line in lines {
        // payload-blind: any line that parses is kept as-is. The only
        // corruption this defends against is framing damage — a write
        // killed mid-line — and that is a parse failure.
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            history.push(Message::from_value(&v));
        }
    }
    Ok((
        Session {
            path: Some(path),
            id: chosen.id.clone(),
            created: chosen.created,
            cwd: chosen.cwd.clone(),
            model: chosen.model.clone(),
            header,
            persisted: history.len(),
        },
        history,
    ))
}

fn list_in(dir: &Path) -> crate::Result<Vec<Meta>> {
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e.into()),
    };
    let mut out: Vec<Meta> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
        .filter_map(|p| read_meta(&p))
        .collect();
    out.sort_by(|a, b| (b.created, &b.id).cmp(&(a.created, &a.id))); // newest first
    Ok(out)
}

/// Header + counts from one session file; `None` when the file isn't a
/// session (no header, unparseable) — a foreign file in the directory is
/// skipped, not fatal.
fn read_meta(path: &Path) -> Option<Meta> {
    let text = fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let header: Value = serde_json::from_str(lines.next()?).ok()?;
    if header["type"] != "session" {
        return None;
    }
    let mut msgs = 0;
    let mut first = String::new();
    for line in lines {
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue }; // torn tail
        msgs += 1;
        if first.is_empty() && v["role"] == "user" {
            // a peek, not a contract — display-only, best-effort
            if let Some(s) = v["content"].as_str() {
                first = s.lines().next().unwrap_or("").to_owned();
            }
        }
    }
    Some(Meta {
        id: header["id"].as_str()?.to_owned(),
        created: header["created"].as_u64().unwrap_or(0),
        cwd: header["cwd"].as_str().unwrap_or_default().to_owned(),
        model: header["model"].as_str().unwrap_or("?").to_owned(),
        protocol: header["protocol"].as_str().unwrap_or("?").to_owned(),
        msgs,
        first,
    })
}

// ---- --list -------------------------------------------------------------------

/// `--list`: this project's sessions (`all: false`) or every project's —
/// newest first. Needs no config — this runs before credentials are even
/// read.
pub fn print_list(all: bool) -> crate::Result<()> {
    let metas = if all { Session::list_all()? } else { Session::list()? };
    let scope = if all { projects_root() } else { project_dir() }
        .ok_or_else(|| Error::Msg("no home directory — cannot find ~/.jingwei/projects".into()))?;
    if metas.is_empty() {
        println!("no sessions yet in {} — an interactive run creates one", scope.display());
        return Ok(());
    }
    print!("{}", format_table(&metas, &scope, all));
    Ok(())
}

/// The `--list` table. Columns pad to their widest entry, the first task
/// is clipped (wide chars counted double — the same ruler the frontends
/// measure with), and a footer names the scope. The DIR column appears
/// only when projects are mixed (`--all`) — within one project every row
/// would carry the same value.
pub fn format_table(metas: &[Meta], scope: &Path, show_dir: bool) -> String {
    let mut heads = vec!["ID", "STARTED", "PROTOCOL", "MODEL", "MSGS"];
    if show_dir {
        heads.push("DIR");
    }
    heads.push("FIRST TASK");
    let cells = |m: &Meta| {
        let mut v = vec![
            m.id.clone(),
            utc(m.created),
            m.protocol.clone(),
            display::truncate_cols(&m.model, 18),
            m.msgs.to_string(),
        ];
        if show_dir {
            v.push(clip_head(&dir_tail(&m.cwd), 26));
        }
        v.push(display::truncate_cols(m.first.trim(), 40));
        v
    };
    let rows: Vec<Vec<String>> = metas.iter().map(cells).collect();
    let mut widths: Vec<usize> = heads.iter().map(|h| display::disp_width(h)).collect();
    for r in &rows {
        for (i, c) in r.iter().enumerate() {
            widths[i] = widths[i].max(display::disp_width(c));
        }
    }
    let row = |cells: &[String]| {
        cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let pad = " ".repeat(widths[i].saturating_sub(display::disp_width(c)));
                format!("{c}{pad}")
            })
            .collect::<Vec<_>>()
            .join("  ")
    };
    let mut out = String::new();
    out.push_str(&row(&heads.iter().map(|s| s.to_string()).collect::<Vec<_>>()));
    out.push('\n');
    for r in &rows {
        out.push_str(&row(r));
        out.push('\n');
    }
    out.push_str(&format!(
        "\n{} {} in {}\nresume one with: jingwei --resume <ID> · the newest with: jingwei -c\n",
        metas.len(),
        if metas.len() == 1 { "session" } else { "sessions" },
        scope.display()
    ));
    out
}

// ---- time, without a dependency ----------------------------------------------

/// Days since 1970-01-01 → (year, month, day) — Howard Hinnant's
/// `civil_from_days`, the whole of the calendar this module needs. Pure,
/// so the dates it produces are a unit-tested fact and not a dependency.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// (y, m, d, hh, mm, ss) from unix seconds — UTC, the one clock that needs
/// no locale to read back later.
fn utc_parts(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let (y, mo, d) = civil_from_days((secs / 86_400) as i64);
    let rem = secs % 86_400;
    (y, mo, d, (rem / 3600) as u32, (rem % 3600 / 60) as u32, (rem % 60) as u32)
}

/// "YYYYMMDD-HHMMSS" — a session id's time part: sorts chronologically as
/// a plain string, names the file, stays readable in `--list`.
pub fn id_stem(secs: u64) -> String {
    let (y, mo, d, hh, mm, ss) = utc_parts(secs);
    format!("{y:04}{mo:02}{d:02}-{hh:02}{mm:02}{ss:02}")
}

/// "YYYY-MM-DD HH:MM" (UTC) — human time for banners and `--list`.
pub fn utc(secs: u64) -> String {
    let (y, mo, d, hh, mm, _) = utc_parts(secs);
    format!("{y:04}-{mo:02}-{d:02} {hh:02}:{mm:02}")
}

/// "1 message" / "3 messages" — banners read like a person counts.
pub fn msg_word(n: usize) -> &'static str {
    if n == 1 { "message" } else { "messages" }
}

/// The process's working directory, as a string — provenance for the
/// header and the scope for "resume from here".
pub fn current_dir_string() -> String {
    std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default()
}

/// Are these the same directory? Canonicalizes both when it can (so `.`,
/// symlinks, and repeated separators agree); paths that no longer exist
/// fall back to string equality. The directory filter and the
/// resumed-elsewhere note both ask this one question.
pub fn same_dir(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The readable tail of a path — the last two components, with an ellipsis
/// when there is more above them: the `--list` DIR column and the
/// resumed-elsewhere note, where the leaf names the project and the root
/// is noise. Both separators, so a Windows path renders too.
pub fn dir_tail(path: &str) -> String {
    let parts: Vec<&str> = path.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    match parts.len() {
        0 => path.to_string(),
        1 => parts[0].to_string(),
        n => format!("…/{}", parts[n - 2..].join("/")),
    }
}

/// Clip to `max` display columns keeping the *end* — paths name things at
/// their leaf, so the DIR column drops the front; the mirror of
/// `display::truncate_cols`, which drops the end.
fn clip_head(s: &str, max: usize) -> String {
    use unicode_segmentation::UnicodeSegmentation;
    if display::disp_width(s) <= max {
        return s.to_string();
    }
    let mut kept: Vec<&str> = vec![];
    let mut w = 1; // the ellipsis rides inside the budget
    for g in s.graphemes(true).rev() {
        let gw = display::disp_width(g).max(1);
        if w + gw > max {
            break;
        }
        kept.push(g);
        w += gw;
    }
    kept.reverse();
    format!("…{}", kept.concat())
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Four random hex digits, so two sessions started in the same second still
/// get their own files. `RandomState` is seeded from the OS; hashing pid
/// and clock through it is entropy enough for a filename suffix, without
/// a dependency for it.
fn rand4() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(now_secs());
    h.write_u32(std::process::id());
    h.write_u64(0x5A17_E4E1);
    format!("{:04x}", h.finish() as u16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("jingwei_session_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// The header's stamp: opaque strings, exactly what session.rs sees of
    /// the connection — never a Config, never a Protocol.
    fn prov() -> Provenance {
        Provenance { model: "test-model".into(), protocol: "test-proto".into(), base_url: "https://x".into() }
    }

    fn history() -> Vec<Message> {
        vec![
            crate::user_message("count files"),
            Message::from_value(&json!({"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]})),
            Message::from_value(&json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a\nb"}]})),
            Message::from_value(&json!({"role": "assistant", "content": [{"type": "text", "text": "there are 2"}]})),
        ]
    }

    // ---- the calendar ----

    #[test]
    fn utc_covers_epoch_and_both_kinds_of_leap_day() {
        assert_eq!(utc(0), "1970-01-01 00:00");
        assert_eq!(id_stem(0), "19700101-000000");
        assert_eq!(utc(951_782_400), "2000-02-29 00:00"); // century leap day
        assert_eq!(utc(1_709_210_096), "2024-02-29 12:34"); // quadrennial leap day
        assert_eq!(id_stem(1_709_210_096), "20240229-123456");
        assert_eq!(msg_word(1), "message");
        assert_eq!(msg_word(2), "messages");
    }

    #[test]
    fn ids_sort_chronologically_as_plain_strings() {
        assert!(id_stem(1_700_000_000) < id_stem(1_700_000_001));
    }

    // ---- sync & the file ----

    #[test]
    fn file_appears_only_on_first_sync_and_holds_header_plus_messages() {
        let d = dir("lazy");
        let mut s = Session::create(Some(&d), &prov(), 1, 0);
        assert!(fs::read_dir(&d).unwrap().next().is_none(), "nothing before the first run");
        let h = history();
        s.sync(&h[..2]).unwrap();
        let body = fs::read_to_string(s.path().unwrap()).unwrap();
        let header: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(header["type"], "session");
        assert_eq!(header["format"], 1); // embedded opaque, not interpreted
        assert_eq!(header["id"], json!(s.id()));
        assert_eq!(header["model"], "test-model");
        assert_eq!(header["protocol"], "test-proto");
        assert!(header["cwd"].as_str().is_some_and(|s| !s.is_empty()), "provenance: where it started");
        assert!(body.ends_with(&format!("{}\n", h[1].to_value())), "messages land verbatim, one per line");
        s.sync(&h).unwrap(); // appending continues from the counter
        let body = fs::read_to_string(s.path().unwrap()).unwrap();
        assert_eq!(body.lines().count(), 1 + 4);
    }

    #[test]
    fn sync_appends_only_the_tail_and_load_round_trips() {
        let d = dir("round");
        let mut s = Session::create(Some(&d), &prov(), 1, 0);
        let h = history();
        s.sync(&h).unwrap();
        let (mut loaded, hist) = load_in(&d, Some(s.id())).unwrap();
        assert_eq!(loaded.id(), s.id());
        assert_eq!(loaded.created(), s.created());
        assert_eq!(hist, h, "payload-blind: the same values, byte for byte");
        // the loaded session keeps the file's identity — new turns append
        let mut h2 = h.clone();
        h2.push(crate::user_message("again"));
        loaded.sync(&h2).unwrap();
        let (_, hist2) = load_in(&d, Some(s.id())).unwrap();
        assert_eq!(hist2, h2);
    }

    #[test]
    fn shrink_rewrites_the_whole_file() {
        let d = dir("shrink");
        let mut s = Session::create(Some(&d), &prov(), 1, 0);
        let h = history();
        s.sync(&h).unwrap();
        // the context trim dropped the middle pair: 4 → 2 messages
        let trimmed = vec![h[0].clone(), h[3].clone()];
        s.sync(&trimmed).unwrap();
        let body = fs::read_to_string(s.path().unwrap()).unwrap();
        assert_eq!(body.lines().count(), 1 + 2);
        assert!(!body.contains("t1"), "the dropped pair is gone from the file");
        let (_, hist) = load_in(&d, Some(s.id())).unwrap();
        assert_eq!(hist, trimmed);
        assert!(fs::read_dir(&d).unwrap().count() >= 1, "rename left no .tmp behind");
    }

    #[test]
    fn torn_tail_loads_up_to_the_last_whole_line() {
        let d = dir("torn");
        let mut s = Session::create(Some(&d), &prov(), 1, 0);
        let h = history();
        s.sync(&h).unwrap();
        // a hard kill mid-write: the last line never finished
        let path = s.path().unwrap().to_owned();
        let mut body = fs::read_to_string(&path).unwrap();
        body.push_str(r#"{"role":"assistant","content":[{"type":"te"#);
        fs::write(&path, body).unwrap();
        let (_, hist) = load_in(&d, Some(s.id())).unwrap();
        assert_eq!(hist, h, "everything whole survives; the torn line does not");
    }

    // ---- choosing a session ----

    #[test]
    fn load_resolves_newest_unique_prefix_and_ambiguity() {
        let d = dir("resolve");
        let mut older = Session::create(Some(&d), &prov(), 1, 1000);
        older.sync(&[crate::user_message("old")]).unwrap();
        let mut newer = Session::create(Some(&d), &prov(), 1, 2000);
        newer.sync(&[crate::user_message("new")]).unwrap();

        let (got, hist) = load_in(&d, None).unwrap();
        assert_eq!(got.id(), newer.id(), "no selector: the newest");
        assert_eq!(hist, vec![crate::user_message("new")]);

        let (got, _) = load_in(&d, Some(older.id())).unwrap();
        assert_eq!(got.id(), older.id(), "a unique prefix resolves");

        // two ids sharing a prefix: the error names both, not a guess
        let amb = format!("{}9", older.id());
        let body = fs::read_to_string(older.path().unwrap())
            .unwrap()
            .replacen(&format!("\"id\":\"{}\"", older.id()), &format!("\"id\":\"{amb}\""), 1);
        fs::write(d.join(format!("{amb}.jsonl")), body).unwrap();
        let err = load_in(&d, Some(older.id())).unwrap_err().to_string();
        assert!(err.contains("matches several"), "{err}");
        assert!(err.contains(&amb), "{err}");

        let err = load_in(&d, Some("zzz")).unwrap_err().to_string();
        assert!(err.contains("no session starts with 'zzz'"), "{err}");
        assert!(err.contains("--list"), "{err}");
    }

    #[test]
    fn list_is_newest_first_with_first_task_and_counts() {
        let d = dir("list");
        let mut a = Session::create(Some(&d), &prov(), 1, 1000);
        a.sync(&[crate::user_message("first task")]).unwrap();
        let mut b = Session::create(Some(&d), &prov(), 1, 2000);
        b.sync(&[
            crate::user_message("second task"),
            Message::from_value(&json!({"role": "assistant", "content": [{"type": "text", "text": "done"}]})),
        ])
        .unwrap();
        // a foreign file in the directory is skipped, not fatal
        fs::write(d.join("notes.jsonl"), "not a session\n").unwrap();

        let metas = list_in(&d).unwrap();
        assert_eq!(metas.len(), 2);
        assert_eq!(metas[0].id, b.id(), "newest first");
        assert_eq!(metas[0].first, "second task");
        assert_eq!(metas[0].msgs, 2);
        assert_eq!(metas[1].first, "first task");
        assert!(!metas[1].cwd.is_empty(), "the header's cwd is carried through");
        assert_eq!(metas[1].model, "test-model");
        assert_eq!(metas[1].protocol, "test-proto");
    }

    // ---- --list & the convo handle ----

    #[test]
    fn table_pads_to_the_widest_and_clips_the_long() {
        let metas = vec![Meta {
            id: "20250916-191205-a1b2".into(),
            created: 1_758_000_000,
            cwd: "/home/u/精卫项目".into(),
            model: "a-model-name-longer-than-the-clamp-allows".into(),
            protocol: "test-proto".into(),
            msgs: 12,
            first: "x".repeat(100),
        }];
        // one project: no DIR column — every row would say the same thing
        let table = format_table(&metas, Path::new("/tmp/here"), false);
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines[0].contains("ID") && lines[0].contains("FIRST TASK"), "{table}");
        assert!(!lines[0].contains("DIR"), "{table}");
        assert!(lines[1].starts_with("20250916-191205-a1b2"), "{table}");
        assert!(lines[1].contains("2025-09-16 05:20"), "{table}");
        assert!(table.contains('…'), "the long prompt and model clip: {table}");
        assert!(table.contains("1 session"), "{table}");
        // projects mixed (--all): the DIR column names each one
        let table = format_table(&metas, Path::new("/tmp/here"), true);
        assert!(table.contains("…/u/精卫项目"), "the DIR column shows the path's tail: {table}");
    }

    #[test]
    fn encode_project_flattens_separators_and_fuses_deep_paths() {
        assert_eq!(encode_project("/home/u/proj"), "-home-u-proj");
        assert_eq!(encode_project("C:\\Users\\u\\p"), "C--Users-u-p", "drive colon and backslashes each flatten to a dash");
        assert_eq!(encode_project("/"), "-");
        assert_eq!(encode_project(""), "default");
        // the accepted wrinkle: a literal '-' collides with a separator —
        // the header's cwd keeps the truth, the resumed-elsewhere note fires
        assert_eq!(encode_project("/a-b/c"), encode_project("/a/b/c"), "documented collision");
        // deep paths fuse: readable head + hash of the whole, so two long
        // paths sharing their head still get distinct directories
        let deep = |tail: &str| format!("/root/{}{tail}", "x".repeat(150));
        let a = encode_project(&deep("one"));
        let b = encode_project(&deep("two"));
        assert!(a.chars().count() < 130, "under filesystem limits: {a}");
        assert_ne!(a, b, "the hash keeps deep paths distinct");
        assert!(a.starts_with("-root-xxxx"), "the head stays readable: {a}");
    }

    #[test]
    fn project_dir_canonicalizes_before_encoding() {
        let d = dir("projdir");
        let clean = project_dir_of(&d.display().to_string()).unwrap();
        let spelled = project_dir_of(&format!("{}/./", d.display())).unwrap();
        assert_eq!(clean, spelled, "one project, one directory name");
        let name = clean.file_name().unwrap().to_string_lossy().into_owned();
        assert!(!name.contains('/'), "the encoded name is one path component");
    }

    #[test]
    fn dir_tail_keeps_the_project_leaf_and_drops_the_root_noise() {
        assert_eq!(dir_tail("/home/u/proj"), "…/u/proj");
        assert_eq!(dir_tail("proj"), "proj");
        assert_eq!(dir_tail(""), "");
        assert_eq!(dir_tail("/a/b/c/d"), "…/c/d");
        assert_eq!(dir_tail("C:\\work\\repo"), "…/work/repo", "windows separators render too");
    }

    #[test]
    fn clip_head_keeps_the_leaf_not_the_root() {
        assert_eq!(clip_head("abcdefghij", 5), "…ghij");
        assert_eq!(clip_head("short", 10), "short", "fits: untouched");
        assert_eq!(clip_head("…/tmp/精卫项目", 9), "…精卫项目", "wide chars counted, graphemes kept whole");
    }

    #[test]
    fn same_dir_agrees_through_the_filesystem() {
        let d = dir("samedir");
        let plain = d.display().to_string();
        let padded = format!("{plain}/"); // a trailing separator is the same place
        assert!(same_dir(&plain, &padded), "canonicalization sees through spelling");
        assert!(!same_dir(&plain, &format!("{plain}x")), "a different place is different");
        assert!(!same_dir("/no/such/one", "/no/such/two"), "nonexistent paths don't accidentally agree");
    }

    #[test]
    fn round_trip_is_byte_stable_for_prefix_caches() {
        // providers' prefix caches match exact request bytes: a resumed
        // session must rebuild byte-identical history or the whole cache
        // lineage is lost at the process boundary. serde_json's sorted
        // keys make serialization canonical — the file's bytes are a
        // fixed point — and numbers round-trip exactly.
        let h = vec![
            crate::user_message("count 精卫 \"quoted\" \\backslash\\ \nnewline\ttab 🪨"),
            Message::from_value(&json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "考虑\n多行推理 é\u{301}"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {
                    "command": "printf 'a\\b'", "n": -12, "f": 2.5,
                    "big": 9007199254740993i64,
                    "nested": {"k": [1, 2, {"z": null}]}}},
                {"type": "text", "text": "done"}]})),
        ];
        let d = dir("bytes");
        let mut s = Session::create(Some(&d), &prov(), 1, 0);
        s.sync(&h).unwrap();
        let (_, loaded) = load_in(&d, Some(s.id())).unwrap();
        assert_eq!(h, loaded, "the value tree comes back identical");
        assert_eq!(
            serde_json::to_string(&crate::ir::history_value(&h)).unwrap(),
            serde_json::to_string(&crate::ir::history_value(&loaded)).unwrap(),
            "bytes identical — the resumed request replays into the same prefix"
        );
    }

    #[test]
    fn ephemeral_convo_persists_nothing_and_never_panics() {
        let mut c = Convo::ephemeral();
        c.history.push(crate::user_message("hi"));
        c.persist(&crate::plain::PlainSink::new()); // a quiet no-op — one-shots leave no session behind
        assert_eq!(c.history.len(), 1);
    }
}
