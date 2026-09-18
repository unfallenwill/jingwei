# AGENTS.md — jingwei (精卫)

This file is for **agents developing jingwei itself** — not for jingwei
at runtime. The runtime prompt is the `SYSTEM` constant in `src/main.rs`,
which is a separate contract (see [Don'ts](#donts)).

Keep this file to **rules and constraints that the code does not already
encode**: build/test gates, what counts as a good commit, what not to
touch, and where the two prompts diverge. Anything you can learn from
`ls`, `grep`, or `README.md` does not belong here, and any list that
goes stale with the next commit (file names, test counts, line numbers)
definitely does not.

## Build & test gates

```sh
cargo check --tests          # pre-commit: cheap, catches signature drift
cargo test --bin jingwei     # pre-push: full suite, must be green
```

## Commits

- Lowercase, terse, optional `scope:` prefix matching the touched module.
- Body explains *why*, not *what*.
- One logical change per commit. Split unrelated edits — reviewability
  beats commit-count aesthetics.

## Don'ts

- **Never run `git push`, `git reset --hard`, `git checkout`, `git clean`,
  or `git rebase` without an explicit ask.** This is a hard rule even
  when the conversation makes the intent obvious.
- **Don't edit `SYSTEM` lightly.** It rides into every API call; every
  jingwei user sees the new behavior. Tweak only with intent.
- **Don't bypass `tests/architecture.rs` with `#[cfg(test)]` trickery.**
  The test exists to enforce layering; work with it, not around it.
- **Don't introduce external dependencies casually.** New deps deserve
  their own commit with a rationale in the body. Vendoring a YAML
  parser or a directory walker is fine on its own merits; vendoring any
  HTTP / async runtime is not.
- **Commit `Cargo.lock` changes.** It is tracked (only `target/` and
  `Cargo.lock.bak` are gitignored).

## The two agent prompts

They look similar but are two unrelated contracts:

| Audience | Lives in | Why it exists |
| --- | --- | --- |
| jingwei at runtime, given a task | `SYSTEM` in `src/main.rs` | Shipped with every API call; models weight it heavily; users tune behavior by editing it. |
| An agent developing jingwei itself | this file | Lets Cursor / Codex / Aider / etc. pick up the conventions without an internal monologue. |

Editing one does not affect the other. Don't conflate them.