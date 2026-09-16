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
Cross-platform (Linux / macOS / Windows). Release binary ~2 MB.

## Setup

```sh
export JINGWEI_API_KEY=<your-api-key>
# optional
export JINGWEI_BASE_URL=...
export JINGWEI_MODEL=...
# Defaults: --max-tokens 131072 · --context-size 1000000 · --max-turns 60
export JINGWEI_PROTOCOL=minimax     # or zai, or deepseek
export JINGWEI_CACHE=auto           # or active (minimax's cache_control breakpoints)
export JINGWEI_THINKING=preserve     # or strip
export JINGWEI_EFFORT=high           # low | medium | high | max — reasoning effort knob
export NO_COLOR=1                    # disable colors, both frontends (JINGWEI_NO_COLOR too)
export JINGWEI_COLOR=always          # force colors on (e.g. through a pipe into a pager)
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
plain log (pipes, one-shots, `JINGWEI_NO_TUI=1`) has no editor, so it keeps
your shell's own line editing and history instead. Each answer (including the
final text turn of an agent run) stays in the conversation, so follow-ups keep
context.

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
one-shot runs, and `JINGWEI_NO_TUI=1` get the plain line-oriented log
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

The agent core never touches a terminal: it emits `Msg`s through a display
port (`src/display.rs`), and a frontend interprets them —

- **TUI** (`src/tui/`, on a terminal): an Elm-style single state tree.
  `model.rs` is plain data (no clocks, no I/O), `update.rs` is a pure
  event→state function, `view.rs` is a pure state→screen function, and
  `mod.rs` is the only impure shell — raw mode, the event loop, the blit.
  Every UI rule (grapheme-safe editing, history recall, scroll clamping,
  follow-the-tail, the fold ratchet) is a unit-tested fact about pure
  functions; no terminal needed to test the UI.
- **plain** (`src/plain.rs`; pipes, one-shot runs, `JINGWEI_NO_TUI=1`):
  a pure fold over the same messages into a line-oriented log, plus the
  dumb line reader that drives it.

The port module is only the contract: the `Msg` type (with its documented
ordering), the usage shape it carries, the `Show` sink the core emits
through, and the vocabulary both frontends render with — the prompt, the
thought marker, width measurement, the color gate (`NO_COLOR` honored
everywhere; `JINGWEI_COLOR=always` forces it on). The core is handed a
`&dyn Show` and never learns which frontend is listening — there is no
process-global sink, so a run's frontend is a detail the composition root
plugs in: the terminal TUI, the plain log, or (one day) a web socket that
ships the same `Msg`s as JSON. The port imports no frontend; the frontends
import the port.

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
| `edit_file`  | Replaces the first exact occurrence of a string.         |

Every tool call is echoed to the terminal: the tool name, a summary of its
arguments (the command, the path, …) and the first 20 lines of its output.
Tool outputs above `MAX_TOOL_OUTPUT` (50 KB) are truncated before they go back
to the model. The agent loop stops after `--max-turns` (default 60) iterations —
even 精卫 has a budget.

## How it works

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
cargo test            # 128 unit tests + 6 integration tests
cargo clippy --all-targets
```

## License

MIT — see [LICENSE](LICENSE).
