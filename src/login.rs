// `jingwei login` — configure providers / api keys / active model. Today
// this module is the read-side: `--show` prints the current settings,
// `--reset` deletes the file. The interactive wizard lives in a future
// commit — it is the bigger piece, and keeping it separate keeps this
// commit's review surface small.

use crate::{Error, Result};
use crate::settings::Settings;

/// Run a non-interactive `jingwei login` invocation. The interactive
/// wizard is not yet wired in, so a bare `jingwei login` reports a hint
/// instead of starting it.
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
    // bare `jingwei login`: tell the user the wizard is on its way and
    // show what they can already do today.
    if Settings::missing() {
        eprintln!("no settings.json found — `jingwei login` will create one");
    } else {
        eprintln!("settings.json already exists at {}", Settings::path().map(|p| p.display().to_string()).unwrap_or_default());
    }
    eprintln!("(interactive wizard is not wired in this build — use --show or --reset)");
    Err(Error::Msg("login wizard not yet implemented".into()))
}
