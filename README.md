# jingwei · 精卫

> 炎帝之少女，名曰女娃。女娃游于东海，溺而不返，故为精卫，
> 常衔西山之木石，以堙于东海。 ——《山海经·北山经》

**jingwei** is a minimal coding agent. Give it a task and it carries stones —
one command, one edit at a time — until the sea is land.

**Providers live behind a port.** Pick the wire protocol and jingwei speaks
it: minimax (default — the Messages wire, where the vendor names both its
endpoint `https://api.minimax.cn/anthropic` and its flagship `MiniMax-M3`,
so a key alone is a whole config), zai (Chat Completions with preserved
thinking — endpoint and flagship `glm-5.3-flash` named too), and deepseek
(the Chat Completions dialect plus a thinking toggle; it names its own
endpoint too). Every other Messages- or Chat-Completions-compatible
endpoint still works by pointing `--base-url` at it.

Model/view/update TUI on ratatui + crossterm; agent core speaks the wire.
Cross-platform (Linux / macOS / Windows). Release binary ~3 MB.

## Setup

Everything is a CLI flag — no environment variables to set. At minimum you
need an API key; the endpoint and model default to each vendor's own:

```sh
jingwei --api-key <your-api-key> "task"
```

Override the defaults when you want a different endpoint or model:

```sh
jingwei --api-key <key> --base-url https://your-endpoint -m your-model "task"
jingwei --protocol zai    --effort high "task"
jingwei --protocol deepseek --thinking strip "task"
```

Other knobs:

```sh
--cache auto|active           # server cache mode (active = cache_control breakpoints on Messages wire)
--thinking preserve|strip     # whether to keep reasoning in sent history
--effort low|medium|high|max  # reasoning effort; unset sends nothing
--max-tokens 131072           # per-turn output cap
--context-size 1000000        # trim history when the estimate exceeds this
--max-turns 100               # stop the agent loop after N turns
--max-agents 4                # how many background agents may run at once (0 disables spawning)
--agent-wait-ms 120000        # how long wait_agents waits before it reports "still running"
NO_COLOR=1                    # disable colors (both frontends honor the industry standard)
```

Build:

```sh
cargo build --release
```

The binary is at `./target/release/jingwei` (or `jingwei.exe` on Windows).

Cross-compile to Windows from Linux:

```sh
rustup target add x86_64-pc-windows-gnu
cargo build --release --target x86_64-pc-windows-gnu
```

## Usage

```sh
# DeepSeek's own endpoint (deepseek protocol knows it — no --base-url needed)
jingwei --protocol deepseek -m deepseek-flash "count *.rs files"

# MiniMax — the default; endpoint and flagship known, active cache + effort
jingwei --cache active --effort high "count *.rs files"

# Zai (智谱) — endpoint and flagship known; preserved thinking on by default
jingwei --protocol zai --effort high "count *.rs files"

# DeepSeek, thinking off (the wire's honest strip: thinking disabled)
jingwei --protocol deepseek --thinking strip "quick question"

jingwei    # interactive REPL
jingwei -c            # resume the newest session from this project
jingwei --resume 20260916-1156    # resume by id prefix
jingwei --list        # this project's sessions (--all: every project's)
jingwei --help    # full flag list
```

### REPL

With no prompt, jingwei opens a REPL. `/exit` quits; Ctrl-C clears the
current line, Ctrl-D exits; an API error is reported but doesn't kill the
session. In the TUI,
input history is saved to `~/.jingwei_history` and reloaded on start; the
plain log (pipes, one-shots) has no editor, so it keeps
your shell's own line editing and history instead. Each answer (including the
final text turn of an agent run) stays in the conversation, so follow-ups keep
context.

`/agents` lists the session's background agents — one row each: id, label,
state, elapsed, tools called, and the report of any agent that has
finished and not been read yet. `/agents stop <id>` stops one. Both
frontends share one implementation (`src/agent/cmd.rs`), the way `/mcp`
does, so the human can see and stop a runaway worker without going
through the model.

The interactive UI is an inline TUI (crossterm, with ratatui only for the
review overlay) that owns a small framed pane — separator rules above and
below the input row, a live status bar underneath — at the bottom of the
screen you already had. The bar is the session's ledger, flush right
(model, effort tier, cache hit rate, `ctx used/--context-size` — checked
occasionally, shed in that reverse order as the pane narrows). While a
task runs the pane only ever grows, and when a reasoning block folds the
rows it held become blank reserve — so the status bar never walks between
rows and nothing opens under the bar as thoughts fold and restream. The
reserve is blank, never the folded reasoning: only a live, streaming block
is ever drawn above the input. Everything
above is the terminal's own scrollback:
finished lines are appended to it, so the mouse wheel, text selection, and
whatever was on screen before jingwei started keep working; there is no
alternate screen and no mouse capture while the REPL runs. The editor is
ours — grapheme-safe caret, **multi-line composing** (Ctrl-J breaks the
line everywhere; Shift-Enter too, where the terminal speaks the kitty
keyboard protocol), line-wise Home/End, history recall
(`~/.jingwei_history`), word deletes — no readline dependency. Submitting
clears the input line for the next task; Up recalls the last one. Pipes,
one-shot runs get the plain line-oriented log
instead.

Reasoning arrives *live and folded*: while a block streams, the pane shows
it as it arrives (`◌ thought #2 · 4 lines · 12s` plus its moving tail) — a
frozen pane cannot tell "thinking" from "hung". When it closes, it lands
as one dim marker that previews its first line (`▸ thought #3 · checking
Cargo.toml … +13`) instead of a wall of text. **Ctrl-O unfolds everything**
— thoughts and tool-output tails alike — into a scrollable full-screen
review (the one place an alternate screen is used, and only while it is
open); Ctrl-O again returns to the prompt, and what has been unfolded never
refolds. Folded bodies hang from a `│` gutter — structure, not color, so
the hierarchy survives NO_COLOR and DIM-blind terminals. Long lines wrap:
the scrollback wraps at the width each row was flushed with (the wrap is
baked, the way a shell's own output is), and the review overlay re-wraps
at the live width as the terminal is resized. The pane itself is resized
by clearing the screen and reprinting the visible tail at the new width —
the pane stays anchored at the bottom, and nothing that already rode into
the scrollback is reprinted — because the terminal's own reflow of a width
change is not something we try to preserve, only to overwrite. Only the
pane's live rows —
the streaming tail, the status bar — stay one physical row each. There are
no display switches: folding is simply how reasoning is shown, in the TUI
and in the plain log alike.

### Sessions

Interactive conversations persist across processes, one subdirectory per
project: `~/.jingwei/projects/<project>/<id>.jsonl`,
where `<project>` is the working directory's canonical path flattened to
`-`. The file's header records the true `cwd`, model, and protocol; every
following line is one message of the internal history, appended **at run
boundaries only** (so the file never ends mid-run: an interrupted run
leaves it at its last complete run, a hard kill never corrupts what came
before, and a history the context trim shrank is rewritten atomically).
`-c`/`--continue` resumes **this project's** newest session; `--resume
<ID>` resumes by id prefix within the project; `--list` shows this
project's sessions (newest first), `--list --all` every project's (with a
DIR column). One-shot runs stay ephemeral unless resumed. Sessions store
the internal history, not any vendor's wire format, so a session resumes
on a different model, protocol, or endpoint than it started with (the
banner notes a model change). History round-trips **byte-identically** —
canonical JSON — so resuming with the same binary and flags keeps provider
prefix caches warm. Two
wrinkles of the encoding are accepted: a literal `-` in a directory name
collides with a separator (`/a-b` and `/a/b` share a project — the
header's cwd still tells them apart, and resuming across the clash notes
that the tree moved), and renaming a project directory starts a new
namespace (the old files stay where they were).

## Architecture

The crate is a library plus a thin binary: `src/lib.rs` composes, and
`src/main.rs` is 37 lines — the tokio runtime, the exit codes, the error
banner, nothing else.

The agent core never touches a terminal: it emits `Msg`s through a display
port (`src/ui/mod.rs`), and a frontend interprets them —

- **TUI** (`src/ui/tui/`, on a terminal): an Elm-style single state tree.
  `model.rs` is plain data (no clocks, no I/O), `update.rs` is a pure
  event→state function, `view.rs` is a pure state→screen function, and
  `mod.rs` is the only impure shell — raw mode, the event loop, the blit.
  Every UI rule (grapheme-safe editing, history recall, scroll clamping,
  follow-the-tail, the fold ratchet) is a unit-tested fact about pure
  functions; no terminal needed to test the UI.
- **plain** (`src/ui/plain.rs`; pipes, one-shot runs):
  a pure fold over the same messages into a line-oriented log, plus the
  dumb line reader that drives it.

The port module is only the contract: the `Msg` type (with its documented
ordering), the usage shape it carries, the `Show` sink the core emits
through, and the vocabulary both frontends render with — the prompt, the
thought marker, width measurement, the color gate (`NO_COLOR` honored
everywhere; `NO_COLOR` opts out). The core is handed a
`&dyn Show` and never learns which frontend is listening — there is no
process-global sink, so a run's frontend is a detail the composition root
plugs in: the terminal TUI, the plain log, or (one day) a web socket that
ships the same `Msg`s as JSON. The port imports no frontend; the frontends
import the port.

Under the port the agent became a value, and the run became a function. An
**agent** (`src/agent/`) is a plain value — id, name, role, task,
lifecycle, report, the tools it called, what it spent — with no I/O and
no handles, so the frontends can clone and render it, the file ledger and
the cancel scope can key on it, and two callers can finally be told apart.
Its **role** is `Main` (the conversation the human is talking to) or
`Worker` (one the fleet spawned). A **context**
(`src/core/context.rs`) is one agent's model-visible state: the system
text, the in-band reminders, the tool surface, the transcript it owns,
and the trim budget that trims it — so two agents can never trim each
other's messages. A **run** is one function, `workflow::run_task`
(`src/workflow/mod.rs`): the task goes on disk before the first model
call, the run is bracketed by exactly one begin and one end, and the
fleet is opened and closed with the run. The one-shot, the plain REPL and
the TUI no longer each spell that sequence out.

The **fleet** (`src/workflow/exec.rs`) is the one runtime capability the
round added: the main agent can start background agents that work in
parallel, each with its own transcript, its own context budget, its own
cancel token and its own sink. They run the *same* turn code with the
*same* tools; what makes them safe is not a smaller toolbox but the
ledger (each agent keeps its own view of the files it has seen) and the
bounds the fleet puts on the fan-out. The main agent collects the
reports itself, with `wait_agents` — nothing is slipped into its
transcript behind its back — and a run that ends with workers still live
stops them and lets them go.

The port grew exactly one variant for this: `Msg::FromAgent { id, event }`
carries a background agent's events **wrapped**. The wrapper is the
attribution — the inner usage and the inner end-of-turn belong to that
agent and never to the session's totals, so a worker's spending cannot
inflate the bar, and a worker can never end the human's turn.

Providers sit behind a port of their own. The core speaks one internal
history and asks the composition root which wire to hand it to; each wire
is a self-contained adapter that owns its translation *and* its rules —
what a vendor accepts (`deepseek` and `zai` reject `--cache active`: their
caches are implicit, nothing to mark; all three refuse `--effort` with
`--thinking strip`: their wires want every past turn's reasoning back once
tools are in play), how a policy flag becomes wire fields (`--thinking
strip` erases history blocks on minimax and parks the model at `thinking:
disabled` on deepseek; on zai it is simply *not kept* — the flagship
thinks whether asked or not, `disabled` is a hard 400 — while `preserve`
there spells `clear_thinking: false`, *preserved thinking*, the vendor's
own recommendation for agents because the echoed reasoning rides the
cached prefix), and how the usage ledger maps back (each vendor's cache
fields folded into the one internal "input is non-cached input" contract
the ctx gauge and cache% read). Vendors that share the Chat Completions
dialect share its neutral machinery (message translation, SSE assembly) —
the dialect names no vendor, so no vendor's details can leak into
another's.

## The three knobs people confuse

| Knob | Layer | What it controls |
| --- | --- | --- |
| `--thinking preserve\|strip` | client policy | whether reasoning blocks from earlier turns stay in the history you send back |
| `--effort low\|medium\|high\|max` | request | reasoning effort, when the endpoint offers the knob: `reasoning_effort` on the zai wire (low/high/max are its words; medium folds into high — the word is a hard 400 there), `thinking` `adaptive` + a `budget_tokens` tier on the minimax wire (low 1024 · medium 8k · high 32k · max = max-tokens minus a floor for the reply), `reasoning_effort` on the deepseek wire (low/high/max are its words; it maps medium→high itself). Unset sends nothing — the endpoint's default rules |
| `--cache auto\|active` | cost/latency | `auto`: rely on the server's implicit cache — minimax, zai and deepseek all report their hits in `usage`, so the bar's cache% and the ctx gauge read true on every wire. `active`: mark `cache_control` breakpoints (system + last tool) — minimax's Messages wire, natively. zai's and deepseek's caches need no switch at all |
| interleaved thinking | model capability | whether the model can reason between tool calls within a turn — not a client switch; jingwei just keeps turn structure intact so it can happen |

They interact (preserved thinking makes the cached prefix longer and more
stable), but they are different axes: thinking decides what is in the context,
caching decides how the server bills and accelerates that context.

## Tools

| Tool         | What it does                                             |
| ------------ | -------------------------------------------------------- |
| `bash`       | Runs a shell command (`sh -c` on Unix, `cmd /C` on Win). |
| `read_file`  | Reads a file.                                            |
| `write_file` | Writes a file, creating parent dirs.                     |
| `edit_file`  | Replaces `old` with `new` — exactly, then line by line ignoring indentation. Names the line numbers when `old` is ambiguous; `replace_all` replaces every occurrence; `edits` applies several pairs in one all-or-nothing call. |

Those four are the whole built-in registry, and it stays four. The fleet
adds four more, in their own table, and they are a *separate provider*:
only the main agent sees them, and a worker never does.

| Tool          | What it does                                             |
| ------------- | -------------------------------------------------------- |
| `spawn_agent` | Starts a background agent on one self-contained task and returns its id. The agent has its own conversation, its own context window and its own cancel token, and shares this session's workspace, shell and file ledger. |
| `wait_agents` | Waits for background agents to finish and takes their reports. A report is delivered once; an agent that is still running comes back as still running. |
| `stop_agent`  | Stops one background agent. Whatever it already produced is kept and reported back. |
| `list_agents` | Lists this session's background agents: id, label, state, elapsed, tools called — and a finished agent's report, if it has not been taken yet. |

jingwei also speaks the **Model Context Protocol** — external servers can
expose their own tools, and jingwei offers them to the model with the same
shape as the built-in ones. See [MCP](#mcp) below.

`edit_file` is the tool with the most policy, all of it aimed at keeping the
work in one trip. It matches exactly first, then line by line with indentation
ignored — the model may copy a line rather than every byte of it — and shifts
`new` to the indentation the file actually uses. An ambiguous `old` is refused
*with the line numbers of every occurrence and the two ways out* (widen the
context, or `replace_all`) instead of the bare demand that it be unique, which
only buys a retry with more text pasted in. `edits` carries several pairs in one
call, resolved in order against one buffer and either all applied or none — that
is the way to do a rename or a repeated idiom without reaching for `write_file`,
which is the blunt tool that can quietly overwrite uncommitted work.

The write itself is a temp file plus a rename (as `session.rs` has always done),
so no crash can leave a truncated file behind. And both `edit_file` and
`write_file` keep a ledger of the bytes they last saw in each path: if the file
on disk is not those bytes — the user's editor, a formatter, another agent — the
write is refused and the tool says to re-read first. That is the one strict
check that pays for itself. It is deliberately not a read-before-edit gate: a
file jingwei has never seen is written as before, because a gate that fires on
"you read it through bash" costs a turn and protects nothing.

Every tool call is echoed to the terminal: the tool name, a summary of its
arguments (the command, the path, …) and the first 20 lines of its output.
Tool outputs above `MAX_TOOL_OUTPUT` (50 KB) are truncated before they go back
to the model. The agent loop stops after `--max-turns` (default 100) iterations —
even 精卫 has a budget.

## MCP

The Model Context Protocol is how jingwei extends its tool set without
forking: an external process speaks JSON-RPC over stdio (or HTTP), jingwei
asks it for its tools on startup, and the model sees them with the same
`{name, description, input_schema}` shape it sees the built-ins.

### Configuration

Two places the configuration can live:

- `~/.jingwei/mcp.json` — every workspace picks it up.
- `<cwd>/.mcp.json` — this project's own servers. A name defined in
  both files is the project's (it is the more specific of the two).

Both files use the shape every MCP client (Claude Code, Cursor, VS Code,
…) writes — servers under `mcpServers`:

```json
{
  "mcpServers": {
    "files":  {"command": "npx",  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    "github": {"url": "https://api.githubcopilot.com/mcp/", "headers": {"Authorization": "Bearer ..."}}
  }
}
```

`command` (with `args`, `env`) speaks stdio; `url` (with `headers`) speaks
streamable HTTP. `${VAR}` is expanded in every string field, with the
usual `${VAR:-default}` for optional ones. A server that did not come up
at startup does not cost the session — the model only sees the tools of
the ones that did, and `/mcp` says which was missing and why.

### `/mcp` in the REPL

Both frontends handle a `/mcp` family of commands, the same way:

| command              | what it does                                              |
| -------------------- | --------------------------------------------------------- |
| `/mcp`               | one line per server, plus the tools it offers             |
| `/mcp list`          | one row per server: name, state (`ready`/`failed`/…), tools |
| `/mcp enable <name>` | reconnect a server that was disabled                      |
| `/mcp disable <name>`| take a server offline without losing its configuration    |
| `/mcp reconnect <name>` | close any live connection and reopen with the stored config |
| `/mcp disconnect <name>` | close the live connection, configuration preserved       |

A server's tools are advertised to the model as `mcp__<server>__<tool>` —
the same convention every other MCP client uses. A call that the server
cannot answer, or one to a server that was disabled mid-session, comes
back as a sentence the model can read: `error: mcp server <name> is
disabled; enable it with /mcp enable <name>`.

A command takes effect on the registry at once — the tool list changes at
once, so the model stops seeing a disabled server's tools on the next turn
— and on the server's connection at that server's next safe point. A call
already running on that server is allowed to finish: the command does not
cancel it. Calls to different servers overlap; calls to one server are one
at a time, because the hub serializes a server's calls on purpose, so one
server's calls keep one order.

`/mcp disable` and `/mcp disconnect` need no new connection, so they are
answered at once — whatever that server is doing. `/mcp enable` and
`/mcp reconnect` do need one, so they are answered when it is up, which is
behind anything that server was already doing.

Two commands on one server never wait on each other's effect on the
registry — that happens as soon as the command is handled — so their
connections can come up in either order. The later command is the one that
stands — the registry's view is the later command's — and the earlier one
is told it was superseded: a connection that comes up after the server was
taken offline does not bring it back.

1. POST `{base}/v1/messages` (minimax — base defaulting to
   `https://api.minimax.cn/anthropic`, model to `MiniMax-M3`) or
   `{base}/chat/completions` (zai — base defaulting to
   `https://open.bigmodel.cn/api/paas/v4`, model to `glm-5.3-flash` — and
   deepseek, base defaulting to `https://api.deepseek.com`) with `model`,
   `system`, `tools`, `messages` — streaming by default (`-S` to block).
2. Each turn the model returns content. Text is printed as it streams, tool
   calls are executed, results go back in the next request. Reasoning blocks
   are preserved or stripped per `--thinking`. Every assistant turn — including
   the final answer — is kept in history, so follow-up questions keep context.
3. Caching: `--cache auto` relies on the server's passive prefix cache;
   `--cache active` adds `cache_control` breakpoints (system + last tool)
   on the Messages wire. DeepSeek's disk cache is neither: always on, no
   breakpoints, its hit/miss reported in every response's `usage`
   (`prompt_cache_hit_tokens`) and folded into the cache% the bar shows.
   Cache stats are printed from each response's `usage`.

## Tests

```sh
cargo check --tests    # the gate: cheap, catches signature drift
cargo test             # 638 unit tests + 22 integration tests
cargo build            # the production binary — the gates do not build it
```

`cargo build` earns its line because a production import of the test-only
helper is invisible to the other two gates: `test_util` lives behind
`#[cfg(test)]`, so such an import compiles under `cargo check --tests`
and `cargo test` and breaks only `cargo build`. The architecture test
forbids it.

## License

MIT — see [LICENSE](LICENSE).
