//! `/agents` — the human's surface on the fleet.
//!
//! The human must be able to see and stop the workers **without going
//! through the model**: a runaway worker is usually one the model
//! mis-specified, and asking that model to stop it costs a turn that may
//! never come.
//!
//! One implementation, two frontends — the same shape as the MCP command
//! handler: [`handle`] parses the verb and the frontend prints what it
//! returns, and [`render_table`] draws the table both `/agents` and
//! `list_agents` show.
//!
//! There is no `/agents spawn` — spawning is a decomposition decision,
//! and the human does not have the decomposition in their head. There is
//! no `/agents steer` and no `/agents wait`: the workers' progress is
//! already in the transcript, and a verb with no counterpart in the tool
//! set is a second way to do the same thing.

use std::time::Duration;

use crate::core::ids::AgentId;
use crate::ui::Msg;
use crate::workflow::exec::Fleet;

/// One row of the agents table.
///
/// This is the agent table's own row. It is not the TUI's
/// `ui::tui::model::Row`, and it must not borrow that name's meaning: a
/// name that means two things in one crate is a name somebody will read
/// wrong.
#[derive(Debug)]
pub(crate) struct Row {
    /// Who the worker is.
    pub(crate) id: AgentId,
    /// The name the human gave it.
    pub(crate) label: String,
    /// `running`, `stopping`, `done`, `failed` or `stopped`.
    pub(crate) state: &'static str,
    /// How long it has run, or how long it ran.
    pub(crate) elapsed: Duration,
    /// The tool calls it made.
    pub(crate) tools: usize,
    /// Its report, while it has not been taken.
    pub(crate) report: Option<String>,
    /// What went wrong, if it failed.
    pub(crate) error: Option<String>,
    /// The report has been taken.
    pub(crate) delivered: bool,
}

/// The human's agent verb. `rest` is everything after `/agents`.
///
/// Returns the lines to print, or `None` when the line was not an
/// `/agents` command at all.
pub(crate) fn handle(rest: &str, fleet: &Fleet) -> Option<Vec<Msg>> {
    let mut it = rest.split_whitespace();
    match it.next() {
        // the bare verb: the table
        None | Some("") => Some(table_lines(fleet.list())),
        Some("stop") => {
            let Some(raw) = it.next() else {
                return Some(vec![Msg::Banner("usage: /agents stop <id>".into())]);
            };
            match AgentId::parse(raw) {
                Some(id) => Some(vec![Msg::Banner(fleet.stop(id))]),
                None => Some(vec![Msg::Banner(format!("'{raw}' is not an agent id"))]),
            }
        }
        Some(other) => Some(vec![Msg::Banner(format!(
            "unknown /agents verb: {other} · try /agents or /agents stop <id>"
        ))]),
    }
}

/// The table, one banner per line.
///
/// A banner is a line in both frontends, and a row that wrapped into two
/// is a row nobody can read: the same reason `/mcp list` emits one
/// banner per server.
fn table_lines(text: String) -> Vec<Msg> {
    text.lines().map(|l| Msg::Banner(l.to_string())).collect()
}

/// The table of workers — the same text for `/agents` and for the
/// `list_agents` tool.
pub(crate) fn render_table(rows: &[Row]) -> String {
    if rows.is_empty() {
        return "no background agents".into();
    }
    let mut out = String::new();
    for r in rows {
        let detail = if let Some(e) = &r.error {
            e.clone()
        } else if r.delivered {
            "reported".into()
        } else {
            format!("{} tools", r.tools)
        };
        out.push_str(&format!(
            "{:<4}{:<11}{:<9}{:>4}   {}\n",
            r.id,
            r.label,
            r.state,
            crate::ui::elapsed_str(r.elapsed),
            detail
        ));
        // A report that has not been taken is printed under its row: the
        // human should not have to ask the model for it.
        if let Some(report) = &r.report {
            if !report.trim().is_empty() {
                for line in report.lines() {
                    out.push_str(&format!("    {line}\n"));
                }
            }
        }
    }
    out.trim_end().to_string()
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: AgentId, label: &str, state: &'static str) -> Row {
        Row {
            id,
            label: label.into(),
            state,
            elapsed: Duration::from_secs(12),
            tools: 7,
            report: None,
            error: None,
            delivered: false,
        }
    }

    #[test]
    fn the_table_lists_the_workers() {
        let a = AgentId::next();
        let b = AgentId::next();
        let text = render_table(&[row(a, "build", "running"), row(b, "test-fix", "done")]);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "one line per worker: {text}");
        assert!(lines[0].starts_with(&format!("{a}  build")), "got: {text}");
        assert!(lines[0].contains("running"), "got: {text}");
        assert!(lines[0].contains("12s"), "got: {text}");
        assert!(lines[0].contains("7 tools"), "got: {text}");
        assert!(lines[1].contains("done"), "got: {text}");
    }

    #[test]
    fn the_table_shows_a_finished_report() {
        let a = AgentId::next();
        let mut r = row(a, "build", "done");
        r.report = Some("the build is green".into());
        let text = render_table(&[r]);
        assert!(text.contains("the build is green"), "an undelivered report is under its row: {text}");
        // a delivered one is marked, and not printed again
        let mut r = row(a, "build", "done");
        r.delivered = true;
        let text = render_table(&[r]);
        assert!(text.contains("reported"), "a delivered report is marked: {text}");
    }

    #[test]
    fn an_empty_table_says_so() {
        assert_eq!(render_table(&[]), "no background agents");
    }

    #[test]
    fn the_table_comes_back_one_banner_per_line() {
        // A banner is a line: a row that wrapped into two is a row
        // nobody can read.
        let fleet = Fleet::new(crate::workflow::exec::AgentCfg::default());
        let msgs = handle("", &fleet).expect("the bare verb has a table");
        assert_eq!(msgs.len(), 1, "one banner for the one line: {msgs:?}");
        assert!(
            matches!(&msgs[0], Msg::Banner(t) if t == "no background agents"),
            "got: {msgs:?}"
        );
    }
}
