# AGENTS.md — jingwei (精卫)

A minimal coding agent in Rust. Single binary, three vendor adapters
(Messages wire / two Chat-Completions dialects), plain + TUI frontends,
MCP integration via an actor-style hub. The project *is* a coding agent,
so this file is read by **other coding agents working on the agent's
own source** — not by jingwei at runtime (that one's prompt lives in
the `SYSTEM` constant in `src/main.rs`).

## Build & test

```sh
cargo check --tests          # quick — exercises tests so a cold run catches signature mismatches
cargo test --bin jingwei    # 470 unit + integration tests, ~5s
cargo build                 # the actual binary
```

`cargo check --tests` is the cheap pre-commit gate. `cargo test --bin jingwei`
is the full pre-push gate. Don't push without the latter passing.

## Commit messages

Lowercase, terse, optional `scope:` prefix matching the touched module.
Body explains *why*, not *what*. Examples from the history:

```
tui: hang slash-menu above the rule so the input frame stays put
drop JINGWEI_* env var support, config is flags-only
config: bump DEFAULT_MAX_TURNS to 100 (60 was tight for 2026 workloads)
```

One logical change per commit. If you're touching two unrelated things,
split them — reviewability beats commit-count aesthetics.

## Module map

```
src/main.rs            ← root: declares every module, owns the entry points
src/turn.rs            ← per-run state machine (Queued / InProgress / terminal).
                         Pure data + transfer rules + 8 smoke tests.
                         Depends on std only. Don't break the leaf property.
src/agents_md.rs       ← AGENTS.md loader (closest-wins walk up, 64 KB cap,
                         full_system_prompt helper). Just added.
src/ir.rs              ← the typed Message/Block shape. Vendors translate to/from
                         this; the agent core speaks it natively.
src/session.rs         ← long-lived identity (history, archive, fork, rename).
                         Does not run.
src/api/{mod,minimax,zai,deepseek}.rs   ← three vendor adapters; body() is the
                         single composition point. cfg.agents_md_extra rides
                         into the system prompt here.
src/mcp/               ← MCP actor hub. mcp::Hub is an mpsc::Sender clone;
                         mcp::HubInner lives only inside the actor.
src/cancel.rs          ← cancel token; agent_turn owns the select! against it.
src/tool_runtime.rs    ← tool dispatch (bash, read_file, write_file, edit_file,
                         mcp__.* routing).
src/tui/, src/plain.rs  ← two frontends. tui for tty, plain otherwise.
src/config.rs          ← Args + Config. Resolves flags → resolved settings.
src/test_util.rs       ← test-only helpers (mock servers, tempdirs, NullSink).
tests/                  ← integration tests (architecture.rs enforces layering).
docs/                   ← rendered architecture diagrams (jingwei-deps.svg).
```

## Architecture rules

The agent loop is **three layers**, each with one job:

```
agent_turn   — wrapper: own the cancel select! against Ctrl-C, construct Turn.
agent_loop   — driver: walk Turn through the table (Queued → InProgress →
                terminal). Translate Result<()> to TurnOutcome, then Turn::finish
                maps that onto a state. No API / tool calls here.
run_turn     — mechanics: the for-step loop that calls api_call_api and runs
                tools. Pure mechanics, knows nothing about Turn.
```

Adding a new agent capability goes in one of three places:

- **New tool**: `src/tools.rs` (definition) + `src/tool_runtime.rs` (dispatch).
- **New vendor**: copy `src/api/zai.rs` to `src/api/<name>.rs`, register in
  `src/api/mod.rs`, add to `config::VENDORS`.
- **New terminal state on the turn**: `src/turn.rs` (state + transition table +
  smoke tests); `src/main.rs` (Result → TurnOutcome mapping in agent_loop).

`tests/architecture.rs` enforces layering. Any new file should fit
the L0..L5 layers it documents; running it (`cargo test`) catches
upward imports. Don't bypass it with `#[cfg(test)]` trickery — the
test lives for that reason.

## Don'ts

- **Never run `git push`, `git reset --hard`, `git checkout`, `git clean`,
  or `git rebase` without an explicit ask.** `SYSTEM` says the same thing.
  The dev policy above also enforces this, but the project should not
  depend on that.
- **Don't edit `SYSTEM` lightly.** It's the runtime agent's prompt
  contract — every jingwei run injects it into every API call. Any
  change is a behavior change for every user.
- **Don't add files to `src/` that the existing modules don't need.**
  `src/turn.rs` is a leaf because it imports nothing from the crate;
  preserve that.
- **Don't commit `target/` or `Cargo.lock.bak`** — both gitignored.
  `Cargo.lock` itself *is* tracked; commit its changes.
- **Don't introduce new external dependencies casually.** Vendoring a
  YAML parser or a directory walker deserves its own commit and a
  mention in the commit body. Vendoring any HTTP / async runtime
  doesn't.

## The two "agent" prompts

jingwei has two unrelated audiences for instructions. They look similar
but live in different files for different reasons:

| Audience | File | Why |
|----------|------|-----|
| jingwei **at runtime**, given a task | `SYSTEM` in `src/main.rs` | Every API call carries it; models weight it heavily; users tune behavior by editing it. |
| An agent **developing jingwei itself** | this file | Lets Cursor / Codex / Aider / etc. pick up the conventions without an internal monologue. |

Editing one does not affect the other. Don't conflate them.

## Tools available

Per `SYSTEM`, the running jingwei agent has: `bash`, `read_file`,
`write_file`, `edit_file` (search/replace with line-aligned fallback).
The same four are what *you* should reach for when working on this
project:

- `read_file` before editing — never guess contents
- `edit_file` for a few lines
- `write_file` only to create / wholesale-replace
- `bash` for everything else (build, test, grep, ls)