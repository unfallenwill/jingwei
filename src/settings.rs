// Persistent settings for jingwei: the user's multi-provider profile set,
// the currently active one, and the small set of shared knobs that travel
// with every run. Lives at `~/.jingwei/settings.json`. The CLI / env
// layers still resolve a per-run `Config`; this module just supplies the
// defaults between runs so the user does not retype api keys every time.
//
// Schema lives at the call site (serde) and is bumped via
// `$schema_version` whenever the field set changes. v1 was the
// single-profile pre-multi-provider shape; v2 introduces the `providers`
// map. The migration lives in `migrate_v1_to_v2` so older settings keep
// working — the field set is the only thing that moved.

use crate::file_io::write_atomic;
use crate::{Error, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;

/// Schema version we write today. Older files migrate up at load.
pub(crate) const CURRENT_SCHEMA_VERSION: u32 = 2;

/// One entry under `providers`. The key in the parent map is
/// `<provider>/<model>` — the same string the `/model` list shows and the
/// `active` field references. We do not denormalise it here: the key is
/// the source of truth.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Profile {
    /// Wire protocol: `minimax` / `zai` / `deepseek`.
    pub(crate) protocol: String,
    pub(crate) api_key: String,
    /// `None` means "use the vendor default base_url" (see Vendor::default_base).
    pub(crate) base_url: Option<String>,
}

/// The whole settings file. Cheap to clone — callers treat it as a view
/// over the file on disk and re-save through `save` after edits.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Settings {
    /// Currently active `<provider>/<model>` key. `None` when the
    /// providers map is empty — `login` then prompts the user.
    pub(crate) active: Option<String>,
    pub(crate) providers: BTreeMap<String, Profile>,
    /// Shared behaviour knobs. `None` means "fall through to the
    /// CLI / env / vendor default chain".
    pub(crate) effort: Option<String>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) context_size: Option<u64>,
}

impl Settings {
    pub(crate) fn empty() -> Self {
        Self { active: None, providers: BTreeMap::new(), effort: None, max_tokens: None, context_size: None }
    }

    /// The `~/.jingwei` directory, if a home is available. The directory
    /// itself is created lazily by callers — this is only the location.
    pub(crate) fn home_dir() -> Option<std::path::PathBuf> {
        env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(std::path::PathBuf::from)
    }

    pub(crate) fn path() -> Option<std::path::PathBuf> {
        Self::home_dir().map(|h| h.join(".jingwei").join("settings.json"))
    }

    /// Find the active profile, or the only one if `active` is unset —
    /// the latter covers a freshly-migrated v1 file that named no active.
    pub(crate) fn active_profile(&self) -> Option<(&str, &Profile)> {
        if let Some(k) = &self.active {
            return self.providers.get_key_value(k).map(|(k, v)| (k.as_str(), v));
        }
        if self.providers.len() == 1 {
            return self.providers.iter().next().map(|(k, v)| (k.as_str(), v));
        }
        None
    }

    /// Persist to `~/.jingwei/settings.json` via the atomic write
    /// primitive — a half-written settings file would lock the user out.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn save(&self) -> Result<()> {
        let Some(path) = Self::path() else {
            return Err(Error::Msg("no home directory — cannot save ~/.jingwei/settings.json".into()));
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(Error::Io)?;
        }
        write_atomic(path.to_str().unwrap(), &self.to_json_string())
            .map_err(|e| Error::Msg(format!("write settings.json: {e}")))?;
        // The api_key field lives in this file; tighten permissions so
        // other users on the box cannot read it.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path).map_err(Error::Io)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&path, perms).map_err(Error::Io)?;
        }
        Ok(())
    }

    /// Remove the on-disk file. Errors when home is unknown or the file
    /// is missing — those are user-visible "cannot reset" conditions.
    pub(crate) fn reset() -> Result<()> {
        let Some(path) = Self::path() else {
            return Err(Error::Msg("no home directory — nothing to reset".into()));
        };
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(Error::Msg("settings.json not found — nothing to reset".into())),
            Err(e) => Err(Error::Io(e)),
        }
    }

    /// Read & migrate up. A missing file is not an error — the caller
    /// asks `active_args` and gets nothing back, exactly like the env
    /// being unset.
    pub(crate) fn load() -> Result<Self> {
        let Some(path) = Self::path() else { return Ok(Self::empty()); };
        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::empty()),
            Err(e) => return Err(Error::Io(e)),
        };
        Self::from_json(&raw)
    }

    /// `~/.jingwei/settings.json`, formatted for humans. Bumping
    /// `$schema_version` does not change the formatter — every valid
    /// version emits a v2-shaped file.
    #[cfg_attr(not(test), allow(dead_code))]
    fn to_json_string(&self) -> String {
        let mut providers = serde_json::Map::new();
        for (k, p) in &self.providers {
            let mut v = serde_json::Map::new();
            v.insert("protocol".into(), json!(p.protocol));
            v.insert("api_key".into(), json!(p.api_key));
            v.insert("base_url".into(), p.base_url.as_ref().map(|s| json!(s)).unwrap_or(Value::Null));
            providers.insert(k.clone(), Value::Object(v));
        }
        let obj = json!({
            "$schema_version": CURRENT_SCHEMA_VERSION,
            "active": self.active,
            "providers": providers,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "context_size": self.context_size,
        });
        serde_json::to_string_pretty(&obj).expect("settings always serialise")
    }

    fn from_json(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s).map_err(|e| Error::Msg(format!("settings.json: {e}")))?;
        let version = v.get("$schema_version").and_then(Value::as_u64).unwrap_or(1) as u32;
        match version {
            2 => parse_v2(&v),
            1 => parse_v1_then_migrate(&v),
            other => Err(Error::Msg(format!("settings.json: unsupported $schema_version {other}"))),
        }
    }

    /// Build a settings layer that, when merged on top of CLI/env, picks
    /// the active profile's protocol/api_key/base_url. Returns a tuple
    /// of (protocol, base_url, model, api_key) all wrapped in Option so
    /// the caller can decide whether to use each.
    pub(crate) fn active_args(&self) -> Option<ActiveArgs> {
        let (key, p) = self.active_profile()?;
        Some(ActiveArgs {
            key: key.to_string(),
            protocol: p.protocol.clone(),
            api_key: p.api_key.clone(),
            base_url: p.base_url.clone(),
        })
    }

    /// Pretty-print active profile + knobs for `jingwei login --show`.
    /// `api_key` is redacted to keep screenshots / shell history safe;
    /// `--reveal` re-enables the raw value.
    pub(crate) fn describe(&self, reveal_api_key: bool) -> String {
        let mut out = String::new();
        out.push_str(&format!("$schema_version: {}\n", CURRENT_SCHEMA_VERSION));
        out.push_str(&format!("active: {}\n", self.active.as_deref().unwrap_or("(none)")));
        out.push_str(&format!("effort: {}\n", self.effort.as_deref().unwrap_or("(unset)")));
        out.push_str(&format!("max_tokens: {}\n", self.max_tokens.map(|n| n.to_string()).unwrap_or_else(|| "(unset)".into())));
        out.push_str(&format!("context_size: {}\n", self.context_size.map(|n| n.to_string()).unwrap_or_else(|| "(unset)".into())));
        if self.providers.is_empty() {
            out.push_str("providers: (none)\n");
            return out;
        }
        for (k, p) in &self.providers {
            let key = if reveal_api_key { p.api_key.clone() } else { redact(&p.api_key) };
            let base = p.base_url.as_deref().unwrap_or("(vendor default)");
            out.push_str(&format!("  {k}: protocol={}, api_key={key}, base_url={base}\n", p.protocol));
        }
        out
    }
}

/// The shape `active_args` lifts off settings — the protocol/api_key/
/// base_url the active profile pins, ready to feed into `args` before
/// `build_config` runs.
pub(crate) struct ActiveArgs {
    /// The `<provider>/<model>` key from the providers map; carries
    /// display-only today, but reserved for the upcoming `/model` list
    /// to highlight the active row by key without another map lookup.
    #[allow(dead_code)]
    pub(crate) key: String,
    pub(crate) protocol: String,
    pub(crate) api_key: String,
    pub(crate) base_url: Option<String>,
}

fn redact(s: &str) -> String {
    if s.len() <= 4 { return "****".into(); }
    // keep the last 4 chars so the user can tell keys apart at a glance
    let tail = &s[s.len() - 4..];
    format!("****{tail}")
}

/// v2 → typed struct. We hand-parse rather than `derive(Deserialize)` so
/// unknown fields are kept out of the struct (forward-compat) and bad
/// types surface with the field name in the error.
fn parse_v2(v: &Value) -> Result<Settings> {
    let mut s = Settings::empty();
    if let Some(a) = v.get("active").and_then(Value::as_str) { s.active = Some(a.to_string()); }
    if let Some(e) = v.get("effort").and_then(Value::as_str) { s.effort = Some(e.to_string()); }
    if let Some(m) = v.get("max_tokens").and_then(Value::as_u64) { s.max_tokens = Some(m as u32); }
    if let Some(c) = v.get("context_size").and_then(Value::as_u64) { s.context_size = Some(c); }
    let providers = v.get("providers").and_then(Value::as_object).ok_or_else(|| {
        Error::Msg("settings.json: missing `providers` map".into())
    })?;
    for (k, pv) in providers {
        let p = parse_profile(pv)?;
        s.providers.insert(k.clone(), p);
    }
    Ok(s)
}

fn parse_profile(v: &Value) -> Result<Profile> {
    let protocol = v.get("protocol").and_then(Value::as_str).ok_or_else(|| {
        Error::Msg("settings.json profile: missing `protocol`".into())
    })?.to_string();
    let api_key = v.get("api_key").and_then(Value::as_str).ok_or_else(|| {
        Error::Msg("settings.json profile: missing `api_key`".into())
    })?.to_string();
    let base_url = v.get("base_url").and_then(Value::as_str).map(str::to_string);
    Ok(Profile { protocol, api_key, base_url })
}

/// v1 was the single-profile shape:
///   { "api_key", "base_url", "model", "protocol", "effort", "max_tokens", "context_size", "streaming", "cache", "thinking" }
/// We collapse the eight single fields into one profile under key
/// `<protocol>/<model>` and drop the three we no longer carry
/// (`streaming`, `cache`, `thinking`).
fn parse_v1_then_migrate(v: &Value) -> Result<Settings> {
    let protocol = v.get("protocol").and_then(Value::as_str).unwrap_or("minimax").to_string();
    let model = v.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let api_key = v.get("api_key").and_then(Value::as_str).unwrap_or("").to_string();
    let base_url = v.get("base_url").and_then(Value::as_str).map(str::to_string);
    if api_key.is_empty() {
        return Err(Error::Msg("settings.json (v1): missing `api_key` — cannot migrate".into()));
    }
    // Fall back to a vendor-default model so the key has a value; the
    // active profile's model is whatever was on disk.
    let model = if model.is_empty() {
        match protocol.as_str() {
            "minimax" => "MiniMax-M3".to_string(),
            "zai" => "glm-5.3-flash".to_string(),
            _ => "deepseek-chat".to_string(),
        }
    } else { model };
    let key = format!("{protocol}/{model}");
    let mut providers = BTreeMap::new();
    providers.insert(key.clone(), Profile { protocol, api_key, base_url });
    Ok(Settings {
        active: Some(key),
        providers,
        effort: v.get("effort").and_then(Value::as_str).map(str::to_string),
        max_tokens: v.get("max_tokens").and_then(Value::as_u64).map(|n| n as u32),
        context_size: v.get("context_size").and_then(Value::as_u64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::temp_dir;

    /// Home-override seam: the settings module's `path()` is hard-wired
    /// to `$HOME`, but tests need to point it at a temp dir without
    /// mutating the environment. We save & restore HOME around each test
    /// via the same lock the env-touching tests use.
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
    fn missing_settings_is_an_empty_value() {
        // no file on disk: load returns an empty Settings, not an error
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_empty");
        let _home = HomeGuard::set(&dir);
        let s = Settings::load().unwrap();
        assert!(s.active.is_none());
        assert!(s.providers.is_empty());
        assert!(s.effort.is_none());
    }

    #[test]
    fn save_then_load_round_trips_profiles_and_active() {
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_roundtrip");
        let _home = HomeGuard::set(&dir);
        let mut s = Settings::empty();
        s.active = Some("zai/glm-5.3-flash".into());
        s.providers.insert("minimax/MiniMax-M3".into(), Profile {
            protocol: "minimax".into(), api_key: "sk-m".into(), base_url: None,
        });
        s.providers.insert("zai/glm-5.3-flash".into(), Profile {
            protocol: "zai".into(), api_key: "sk-z".into(), base_url: Some("https://z".into()),
        });
        s.effort = Some("high".into());
        s.save().unwrap();
        let s2 = Settings::load().unwrap();
        assert_eq!(s2.active.as_deref(), Some("zai/glm-5.3-flash"));
        assert_eq!(s2.providers.len(), 2);
        assert_eq!(s2.effort.as_deref(), Some("high"));
        assert_eq!(s2.providers["zai/glm-5.3-flash"].api_key, "sk-z");
        assert_eq!(s2.providers["zai/glm-5.3-flash"].base_url.as_deref(), Some("https://z"));
    }

    #[test]
    fn save_sets_0600_permissions_on_unix() {
        #[cfg(unix)] {
            use std::os::unix::fs::PermissionsExt;
            let _lock = crate::test_util::env_lock();
            let dir = temp_dir("settings_perms");
            let _home = HomeGuard::set(&dir);
            let mut s = Settings::empty();
            s.active = Some("minimax/M".into());
            s.providers.insert("minimax/M".into(), Profile {
                protocol: "minimax".into(), api_key: "sk-secret".into(), base_url: None,
            });
            let path_before = Settings::path().expect("HOME must be set");
            s.save().unwrap();
            eprintln!("saved to: {}", path_before.display());
            eprintln!("exists after save: {}", path_before.exists());
            let mode = std::fs::metadata(&path_before).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "api_key file must be owner-only: {mode:o}");
        }
    }

    #[test]
    fn reset_removes_the_file() {
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_reset");
        let _home = HomeGuard::set(&dir);
        let mut s = Settings::empty();
        s.active = Some("minimax/M".into());
        s.providers.insert("minimax/M".into(), Profile {
            protocol: "minimax".into(), api_key: "k".into(), base_url: None,
        });
        s.save().unwrap();
        assert!(Settings::path().unwrap().exists());
        Settings::reset().unwrap();
        assert!(!Settings::path().unwrap().exists());
    }

    #[test]
    fn reset_on_missing_file_is_a_clear_error() {
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_reset_missing");
        let _home = HomeGuard::set(&dir);
        let err = Settings::reset().unwrap_err().to_string();
        assert!(err.contains("nothing to reset"), "got: {err}");
    }

    #[test]
    fn v1_single_profile_migrates_into_the_providers_map() {
        // the old shape had eight flat fields; the migration folds them
        // into one profile under `<protocol>/<model>`, drops streaming /
        // cache / thinking (we no longer carry them), and carries effort /
        // max_tokens / context_size through to the new shape.
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_migrate");
        let _home = HomeGuard::set(&dir);
        let v1 = r#"{
            "api_key": "sk-old",
            "base_url": "https://x",
            "model": "MiniMax-M3",
            "protocol": "minimax",
            "effort": "high",
            "max_tokens": 8192,
            "context_size": 500000,
            "streaming": false,
            "cache": "auto",
            "thinking": "preserve"
        }"#;
        std::fs::create_dir_all(Settings::path().unwrap().parent().unwrap()).unwrap();
        std::fs::write(Settings::path().unwrap(), v1).unwrap();
        let s = Settings::load().unwrap();
        assert_eq!(s.active.as_deref(), Some("minimax/MiniMax-M3"));
        let p = &s.providers["minimax/MiniMax-M3"];
        assert_eq!(p.protocol, "minimax");
        assert_eq!(p.api_key, "sk-old");
        assert_eq!(p.base_url.as_deref(), Some("https://x"));
        assert_eq!(s.effort.as_deref(), Some("high"));
        assert_eq!(s.max_tokens, Some(8192));
        assert_eq!(s.context_size, Some(500_000));
    }

    #[test]
    fn v1_deepseek_with_no_model_fills_a_default() {
        // deepseek's default_model() is None — the migration falls back to
        // a sane name so the key under providers is well-formed and the
        // profile is selectable.
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_migrate_deepseek");
        let _home = HomeGuard::set(&dir);
        let v1 = r#"{ "api_key": "sk-d", "protocol": "deepseek" }"#;
        std::fs::create_dir_all(Settings::path().unwrap().parent().unwrap()).unwrap();
        std::fs::write(Settings::path().unwrap(), v1).unwrap();
        let s = Settings::load().unwrap();
        assert!(s.active.is_some(), "v1 without model still produces a usable profile");
        let key = s.active.as_deref().unwrap();
        assert!(s.providers[key].api_key == "sk-d");
        assert_eq!(s.providers[key].protocol, "deepseek");
    }

    #[test]
    fn unknown_schema_version_is_an_error() {
        let _lock = crate::test_util::env_lock();
        let dir = temp_dir("settings_bad_version");
        let _home = HomeGuard::set(&dir);
        std::fs::create_dir_all(Settings::path().unwrap().parent().unwrap()).unwrap();
        std::fs::write(Settings::path().unwrap(),
            r#"{ "$schema_version": 99, "providers": {} }"#).unwrap();
        let err = Settings::load().unwrap_err().to_string();
        assert!(err.contains("unsupported $schema_version"), "got: {err}");
    }

    #[test]
    fn active_profile_picks_active_then_falls_back_to_only_one() {
        // two cases the lookup has to handle cleanly: explicit active,
        // and the lone-profile case where active was never set (e.g. by
        // a v1 migration that didn't populate it).
        let mut s = Settings::empty();
        s.providers.insert("a/1".into(), Profile { protocol: "minimax".into(), api_key: "k1".into(), base_url: None });
        s.providers.insert("b/2".into(), Profile { protocol: "zai".into(), api_key: "k2".into(), base_url: None });
        s.active = Some("a/1".into());
        assert_eq!(s.active_profile().unwrap().0, "a/1");
        s.active = None;
        // two profiles, no active → ambiguous → None
        assert!(s.active_profile().is_none());
        s.providers.remove("b/2");
        // one profile, no active → falls back to the survivor
        assert_eq!(s.active_profile().unwrap().0, "a/1");
    }

    #[test]
    fn describe_redacts_by_default_and_reveals_on_request() {
        let mut s = Settings::empty();
        s.active = Some("minimax/MiniMax-M3".into());
        s.providers.insert("minimax/MiniMax-M3".into(), Profile {
            protocol: "minimax".into(),
            api_key: "sk-1234567890abcdef".into(),
            base_url: None,
        });
        let d = s.describe(false);
        assert!(d.contains("****"), "redacted by default: {d}");
        assert!(!d.contains("1234567890"), "tail not visible: {d}");
        let d = s.describe(true);
        assert!(d.contains("sk-1234567890abcdef"), "reveal shows it: {d}");
    }
}

