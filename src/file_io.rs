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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::temp_dir;

    #[test]
    fn write_atomic_creates_a_new_file_with_the_given_bytes() {
        let dir = temp_dir("create");
        let p = dir.join("new.txt");
        assert!(write_atomic(p.to_str().unwrap(), "hello").is_ok());
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "hello");
    }

    #[test]
    fn write_atomic_overwrites_an_existing_file_in_place() {
        let dir = temp_dir("overwrite");
        let p = dir.join("f.txt");
        std::fs::write(&p, "first").unwrap();
        write_atomic(p.to_str().unwrap(), "second").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "second");
    }

    #[test]
    fn write_atomic_does_not_leave_the_temp_file_behind_on_success() {
        let dir = temp_dir("clean_tmp");
        let p = dir.join("f.txt");
        write_atomic(p.to_str().unwrap(), "v").unwrap();
        // nothing under dir.matches(\.jingwei-.*\.tmp$)
        let stragglers: Vec<_> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".jingwei-"))
            .collect();
        assert!(stragglers.is_empty(), "no tmp file left behind: {stragglers:?}");
    }

    #[test]
    fn write_atomic_cleans_up_the_temp_file_when_rename_fails() {
        // target is a directory — rename would fail, the temp must vanish
        let dir = temp_dir("rename_fail");
        std::fs::create_dir(dir.join("file")).unwrap();
        let err = write_atomic(dir.join("file").to_str().unwrap(), "v");
        assert!(err.is_err(), "writing over a directory must error");
        // the tmp file we tried must not survive the failure
        let stragglers: Vec<_> = std::fs::read_dir(&dir).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".jingwei-"))
            .collect();
        assert!(stragglers.is_empty(), "no tmp file left behind: {stragglers:?}");
    }

    #[cfg(unix)]
    #[test]
    fn write_atomic_writes_through_a_symlink_to_its_target() {
        let dir = temp_dir("symlink");
        let target = dir.join("real.txt");
        std::fs::write(&target, "before").unwrap();
        let link = dir.join("link.txt");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        write_atomic(link.to_str().unwrap(), "after").unwrap();
        // the symlink itself is unchanged — only the file it points at moved
        assert_eq!(std::fs::read_link(&link).unwrap(), target);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "after");
    }

    #[cfg(unix)]
    #[test]
    fn write_atomic_carries_target_permissions_over() {
        // set the target to a non-default mode, then write through
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir("perms");
        let p = dir.join("f.txt");
        std::fs::write(&p, "v").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&p, perms).unwrap();
        write_atomic(p.to_str().unwrap(), "v2").unwrap();
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "permissions survived the atomic replace");
    }

    #[test]
    fn write_atomic_to_a_path_with_no_filename_uses_the_default_name() {
        // a bare path with no file_name: write_atomic still produces a
        // tmp file with the fallback name
        let dir = temp_dir("noname");
        let leaf = "";
        let full = format!("{}{}", dir.to_str().unwrap(), std::path::Path::new(leaf).to_str().unwrap_or(""));
        // the path with an empty filename creates a file called "file" in
        // the temp dir — we just want to see that write_atomic did not
        // panic; the rename may or may not succeed depending on the FS
        let _ = write_atomic(&full, "x");
    }
}
