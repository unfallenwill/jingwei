// File-level primitives that don't fit any one tool: atomic write, and
// someday, more. They live at the crate root so any tool can use them
// without an awkward path.

use std::io::{self, Write};
use std::path::Path;
use std::process;

/// Replace the file's bytes without ever letting a reader — or a crash —
/// see it truncated: a temp file in the same directory (same filesystem, so
/// the rename is atomic), written, flushed and fsynced, then renamed over
/// the target. The temp name carries our pid so two jingwei processes
/// cannot collide, and the target's permissions are carried over — a fresh
/// temp file would otherwise hand the file 0644 on the way through.
pub(crate) fn write_atomic(p: &str, content: &str) -> io::Result<()> {
    let path = Path::new(p);
    // Write *through* a symlink rather than over it: rename replaces the link
    // itself, which is not what "edit this file" means.
    let target = if path.is_symlink() { std::fs::canonicalize(path)? } else { path.to_path_buf() };
    let name = target.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "file".into());
    let tmp = target.with_file_name(format!(".{name}.jingwei-{}.tmp", process::id()));
    let mut f = std::fs::File::create(&tmp)?;
    let written = f.write_all(content.as_bytes()).and_then(|_| f.sync_all());
    drop(f);
    if let Err(e) = written { let _ = std::fs::remove_file(&tmp); return Err(e); }
    if let Ok(meta) = std::fs::metadata(&target) { let _ = std::fs::set_permissions(&tmp, meta.permissions()); }
    if let Err(e) = std::fs::rename(&tmp, &target) { let _ = std::fs::remove_file(&tmp); return Err(e); }
    Ok(())
}
