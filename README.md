# jingwei · 精卫

> 炎帝之少女，名曰女娃。女娃游于东海，溺而不返，故为精卫，
> 常衔西山之木石，以堙于东海。 ——《山海经·北山经》

**jingwei** is a minimal coding agent. Give it a task and it carries stones —
one command, one edit at a time — until the sea is land.

**No built-in providers.** You bring an endpoint and pick the wire protocol;
jingwei speaks both Anthropic Messages and OpenAI Chat Completions.

Model/view/update TUI on ratatui + crossterm; agent core speaks the wire.
Cross-platform (Linux / macOS / Windows). Release binary ~2 MB.

## Setup

```sh
export JINGWEI_API_KEY=<your-api-key>
# optional
export JINGWEI_BASE_URL=...
export JINGWEI_MODEL=...
# Defaults: --max-tokens 131072 · --context-size 1000000 · --max-turns 60
export JINGWEI_PROTOCOL=anthropic    # or openai
export JINGWEI_CACHE=auto            # or active (anthropic protocol only)
export JINGWEI_THINKING=preserve     # or strip
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
# Anthropic-protocol endpoint (MiniMax)
jingwei --base-url https://api.minimax.cn/anthropic -m MiniMax-M3 "count *.rs files"

# OpenAI-protocol endpoint (GLM native API), bigger output budget + context trim
jingwei --protocol openai --base-url https://open.bigmodel.cn/api/paas/v4 \
        -m glm-5.3 --context-size 100000 "refactor this"

# Anthropic-protocol endpoint (GLM), active prompt caching + preserved thinking
jingwei --base-url https://open.bigmodel.cn/api/anthropic -m glm-5.3 \
        --cache active --thinking preserve "quick question"

jingwei    # interactive REPL
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
below the input row, a live status bar underneath (spinner, elapsed,
per-turn and session token usage, cache traffic) — at the bottom of the
screen you already had. Everything above is the terminal's own scrollback:
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
spinner alone cannot tell "thinking" from "hung". When it closes, it lands
as one dim marker that previews its first line (`▸ thought #3 · checking
Cargo.toml … +13`) instead of a wall of text. **Ctrl-O unfolds everything**
— thoughts and tool-output tails alike — into a scrollable full-screen
review (the one place an alternate screen is used, and only while it is
open); Ctrl-O again returns to the prompt, and what has been unfolded never
refolds. Folded bodies hang from a `│` gutter — structure, not color, so
the hierarchy survives NO_COLOR and DIM-blind terminals — and the review
mode wraps where the scrollback truncates: what is flushed is immutable and
renders once, what is reviewed re-renders every frame and owes the reader
the whole line. There are no display switches: folding is simply how
reasoning is shown, in the TUI and in the plain log alike.

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
ordering), the usage shape it carries, and the vocabulary both frontends
render with — the prompt, the thought marker, width measurement, the color
gate (`NO_COLOR` honored everywhere; `JINGWEI_COLOR=always` forces it on).

## The three knobs people confuse

| Knob | Layer | What it controls |
| --- | --- | --- |
| `--thinking preserve\|strip` | client policy | whether reasoning blocks from earlier turns stay in the history you send back |
| `--cache auto\|active` | cost/latency | `auto`: rely on the server's passive cache. `active`: mark Anthropic `cache_control` breakpoints (system + last tool) — anthropic protocol only |
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

1. POST `{base}/v1/messages` (anthropic) or `{base}/chat/completions` (openai)
   with `model`, `system`, `tools`, `messages` — streaming by default (`-S` to
   block).
2. Each turn the model returns content. Text is printed as it streams, tool
   calls are executed, results go back in the next request. Reasoning blocks
   are preserved or stripped per `--thinking`. Every assistant turn — including
   the final answer — is kept in history, so follow-up questions keep context.
3. Caching: `--cache auto` relies on the server's passive prefix cache;
   `--cache active` adds Anthropic `cache_control` breakpoints (system + last
   tool). Cache stats are printed from each response's `usage`.

## Tests

```sh
cargo test            # 95 unit tests + 2 REPL integration tests
cargo clippy --all-targets
```

## License

MIT — see [LICENSE](LICENSE).
