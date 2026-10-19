//! The runtime contract with the model.
//!
//! One string, shipped on every request, in every session, for every
//! vendor. It is static on purpose: the cache breakpoint sits on the
//! first system block, so anything that varies per session (AGENTS.md,
//! the agent-tools reminder) rides in `Context::reminders` and never
//! here.
//!
//! ## Six paragraphs
//!
//! The prompt is six paragraphs, and the names are the ones the prompt
//! already implies: who jingwei is, the tools and how to pick between
//! them, the craft, the safety rules, the finishing rules, and the
//! tail. A diff of the prompt is a diff of a paragraph, not of one
//! and a half kilobytes on a line.
//!
//! ## The bytes are a contract
//!
//! The system block is the first segment of every session's cached
//! prefix (see `api::minimax`), so a change here invalidates every
//! cache the moment it lands. `the_prompt_is_the_bytes_it_was` pins the
//! bytes: it is the evidence that a *structural* refactor of the
//! prompt did not change the prompt's *words*. If the prompt is ever
//! deliberately reworded, that pin is updated in the same commit as
//! the reword, and that commit's body says why.

/// The prompt, in paragraphs.
///
/// `concat!` reads literals only — it cannot take a `const` — so the
/// paragraphs and the concatenation are declared together, out of the
/// same literals. The names are the handles the tests hold the prompt
/// by; the bytes are the ones `SYSTEM` ships. They cannot drift: one
/// invocation, one text.
macro_rules! prompt {
    (
        $(
            $(#[$doc:meta])*
            $name:ident = $text:literal
        ),* $(,)?
    ) => {
        /// The whole prompt, in the order it is sent.
        pub(crate) const SYSTEM: &str = concat!($($text),*);

        $(
            $(#[$doc])*
            // Named for the tests that hold the prompt apart; the
            // shipped bytes are `concat!`ed from the same literals.
            #[allow(dead_code)]
            const $name: &str = $text;
        )*
    };
}

prompt! {
    /// Who jingwei is, and that it is a coding agent with tools.
    IDENTITY = "You are jingwei (精卫), a coding agent with these tools: ",
    /// The built-in tools, and how to pick between them.
    TOOLS = "bash (run a shell command), read_file, write_file, edit_file (search/replace; exact match with line-aligned indentation fallback). \
        Pick the smallest tool that fits: read_file to inspect (not cat via bash); edit_file for a few lines; write_file only to create or wholesale-replace; bash for the rest — running tests, git, grep, ls. ",
    /// Read before you edit; prefer small, targeted commands.
    CRAFT = "Read a file before you edit it — never guess its contents. \
        Prefer small, targeted commands. ",
    /// Nothing interactive, nothing destructive, no git without an ask.
    SAFETY = "Never start a command that waits for input or never returns (editors, pagers, watch or server modes) — everything you run must terminate. \
        Never run destructive commands unless the user explicitly asks. \
        Never run git commit, push, reset, checkout, clean, or rebase unless the user explicitly asks. ",
    /// How the answer is written, and when jingwei may stop.
    VERIFY = "Write for a terminal, not a Markdown renderer: plain text, no headings/tables/bold markers, short lines, with exact copy-pasteable paths and commands. And since your reasoning is folded away and unseen, put every conclusion in the visible answer. \
        Don't stop until the task is actually delivered: make the change, then verify it by running the build, the tests, or the exact command, fix what fails, and repeat until it passes — only then summarize. If you cannot finish, say exactly what is done, what is left, and what blocked you. ",
    /// The name question, and the summary rule.
    TAIL = "When asked your name, say you are jingwei (精卫). \
        When finished, reply with a concise 1-3 sentence summary of what you did.",
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prompt's bytes before this slice, verbatim.
    ///
    /// The pin. It is the evidence that a *structural* refactor of the
    /// prompt did not change the prompt's *words*. If the prompt is
    /// ever deliberately reworded, this pin is updated in the same
    /// commit as the reword — and that commit's body says why.
    const PINNED: &str = "You are jingwei (精卫), a coding agent with these tools: \
        bash (run a shell command), read_file, write_file, edit_file (search/replace; exact match with line-aligned indentation fallback). \
        Pick the smallest tool that fits: read_file to inspect (not cat via bash); edit_file for a few lines; write_file only to create or wholesale-replace; bash for the rest — running tests, git, grep, ls. \
        Read a file before you edit it — never guess its contents. \
        Prefer small, targeted commands. \
        Never start a command that waits for input or never returns (editors, pagers, watch or server modes) — everything you run must terminate. \
        Never run destructive commands unless the user explicitly asks. \
        Never run git commit, push, reset, checkout, clean, or rebase unless the user explicitly asks. \
        Write for a terminal, not a Markdown renderer: plain text, no headings/tables/bold markers, short lines, with exact copy-pasteable paths and commands. And since your reasoning is folded away and unseen, put every conclusion in the visible answer. \
        Don't stop until the task is actually delivered: make the change, then verify it by running the build, the tests, or the exact command, fix what fails, and repeat until it passes — only then summarize. If you cannot finish, say exactly what is done, what is left, and what blocked you. \
        When asked your name, say you are jingwei (精卫). \
        When finished, reply with a concise 1-3 sentence summary of what you did.";

    /// The six paragraphs, in the order they are sent.
    const PARAGRAPHS: [(&str, &str); 6] = [
        ("IDENTITY", IDENTITY),
        ("TOOLS", TOOLS),
        ("CRAFT", CRAFT),
        ("SAFETY", SAFETY),
        ("VERIFY", VERIFY),
        ("TAIL", TAIL),
    ];

    #[test]
    fn the_prompt_is_the_bytes_it_was() {
        assert_eq!(SYSTEM, PINNED,
            "the prompt's bytes changed; a reword is its own commit, and it says why");
    }

    #[test]
    fn every_paragraph_is_not_empty() {
        for (name, text) in PARAGRAPHS {
            assert!(!text.is_empty(), "{name} is blank");
            assert!(!text.starts_with(' '), "{name} starts with a space");
            assert!(!text.contains("  "), "{name} carries a double space");
        }
        // The separator belongs to the end of every paragraph but the
        // last: it is the single space the old one-line string used to
        // carry between two sentences.
        for (name, text) in &PARAGRAPHS[..PARAGRAPHS.len() - 1] {
            assert!(text.ends_with(' '), "{name} lost its separator space");
            assert!(!text.ends_with("  "), "{name} carries two separator spaces");
        }
        let (name, text) = PARAGRAPHS[PARAGRAPHS.len() - 1];
        assert!(!text.ends_with(' '), "{name} must not end with a space");
        assert!(!SYSTEM.starts_with(' ') && !SYSTEM.ends_with(' '),
            "the prompt starts and ends on a word");
    }

    #[test]
    fn the_paragraphs_concatenate_to_the_prompt() {
        // The drift guard: the six paragraphs, in order, are the
        // prompt. A paragraph added, renamed or moved breaks here
        // before it breaks the pin.
        let joined: String = PARAGRAPHS.iter().map(|(_, text)| *text).collect();
        assert_eq!(joined, SYSTEM);
    }

    #[test]
    fn the_prompt_names_jingwei() {
        // Two claims, two paragraphs: the identity says who jingwei
        // is, the tail answers the name question. Each is asserted
        // against the paragraph that carries it — asserted against
        // `SYSTEM` instead, the first would be subsumed by the
        // second, because the tail's sentence contains the first
        // sentence's words, and a guard that cannot fail is not a
        // guard.
        assert!(IDENTITY.contains("jingwei (精卫)"),
            "the identity paragraph names jingwei");
        assert!(TAIL.contains("say you are jingwei (精卫)"),
            "the tail answers the name question");
    }
}
