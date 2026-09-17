//! A server to talk to, for the tests.
//!
//! What the tests are about is the protocol, and a process is the subject of
//! only some of them — so the other end of the pipe is a shell script: it
//! answers the handshake and the two tool methods over its standard input and
//! output, and what it does with a call follows from the tool's name. Every
//! stub is written under a directory of its own and removed when the fixture is
//! dropped, so a test that fails leaves nothing behind.
//!
//! What it does is chosen through the environment the entry hands it, which is
//! also how a real server is configured — so the tests exercise the config path
//! as well:
//!
//! | variable | what it does |
//! |---|---|
//! | `STUB_TOOLS` | the tools `tools/list` offers, comma separated (default `echo`) |
//! | `STUB_VERSION` | the protocol version it answers `initialize` with |
//! | `STUB_SILENT` | answers nothing at all, the server that never comes up |
//! | `STUB_PAGES` | hands `tools/list` out in this many pages |
//! | `STUB_FOREVER` | always hands back another cursor |
//! | `STUB_JUNK` | offers entries that are not tools |
//! | `STUB_LOG` | appends every line it reads, and `eof` when its input ends |
//!
//! The tool a call names decides what the call does:
//!
//! | tool | what it answers |
//! |---|---|
//! | `echo` | the request line it was sent, as the text of one content block |
//! | `fail` | `isError`, with the server's own words |
//! | `empty` | no content at all |
//! | `rich` | one block of every kind a result may carry |
//! | `structured` | structure and no prose |
//! | `big` | more text than a result is allowed to carry |
//! | `slow` | nothing for a while, so that a budget runs out |
//! | `die` | nothing, because the server exits |
//! | `noise` | a banner where a message belongs, then an answer |
//! | `ping` | a request of its own to answer first, then the result |

// The fixture is a public-API affordance for tests in the binary crate and
// in this module's own unit tests; the production binary never constructs a
// stub. `dead_code` allows silence for the parts production does not reach.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::config::{Entry, ServerConfig};

/// Where a stub's script and its log live: a directory of the test's own.
pub struct Stub {
    dir: PathBuf,
}

impl Default for Stub {
    fn default() -> Self {
        Self::new()
    }
}

impl Stub {
    /// A stub in a directory of its own.
    pub fn new() -> Self {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "jingwei-mcp-stub-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("server.sh"), SCRIPT).unwrap();
        Self { dir }
    }

    /// The script a test writes an entry for, for the hub test that builds its
    /// own `.mcp.json` and never touches this fixture's entry.
    pub fn script(&self) -> PathBuf {
        self.dir.join("server.sh")
    }

    /// Where the server writes down what it was sent, for the tests that are
    /// about what reached it.
    pub fn log(&self) -> PathBuf {
        self.dir.join("received.log")
    }

    /// What the server was sent, one line each, and `eof` when its input ended.
    pub fn received(&self) -> Vec<String> {
        std::fs::read_to_string(self.log())
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// The entry for this stub, with `vars` added to its environment.
    pub fn entry(&self, vars: &[(&str, &str)]) -> Entry {
        let mut env: BTreeMap<String, String> = BTreeMap::from([(
            "STUB_LOG".to_string(),
            self.log().to_string_lossy().into_owned(),
        )]);
        for (key, value) in vars {
            env.insert(key.to_string(), value.to_string());
        }
        Entry {
            name: "stub".into(),
            from: "test".into(),
            config: Ok(ServerConfig {
                command: Some("bash".into()),
                args: vec![self.script().to_string_lossy().into_owned()],
                env,
                ..Default::default()
            }),
        }
    }

    /// The same, as a bare config: the tests that build a connection themselves.
    pub fn config(&self, vars: &[(&str, &str)]) -> ServerConfig {
        self.entry(vars)
            .config
            .expect("the test's entry is readable")
    }

    /// The same, with the entry's own timeout set: the tests that are about a
    /// request running out of time rather than about what a server does.
    pub fn config_timed(&self, vars: &[(&str, &str)], seconds: u64) -> ServerConfig {
        let mut config = self.config(vars);
        config.timeout = Some(seconds);
        config
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The script itself: a server that is as small as the protocol allows and as
/// talkative as the tests need.
const SCRIPT: &str = r#"#!/usr/bin/env bash
# An MCP server over stdio, for jingwei's tests. See src/mcp/stub.rs.
set -u

version=${STUB_VERSION:-2025-06-18}
tools=${STUB_TOOLS:-echo}

# A number field of the request line.
number() {
  local value=${1#*\"id\":}
  printf '%s' "${value%%,*}"
}

# A string field of the request line, read from the end: the protocol's own
# `name` is the last one on the line, whatever the arguments happen to carry.
string() {
  local value=${2##*\"$1\":\"}
  printf '%s' "${value%%\"*}"
}

# The arguments object of a call: everything after `"arguments":` up to the
# `"name"` that follows it (the fields are written in order).
arguments() {
  local value=${1#*\"arguments\":}
  printf '%s' "${value%%,\"name\":*}"
}

# The tools named in the first argument, as the JSON list tools/list answers.
listed() {
  local out= name
  for name in ${1//,/ }; do
    out="$out,{\"name\":\"$name\",\"description\":\"the $name tool\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}}"
  done
  printf '%s' "${out#,}"
}

# A result carrying one text block.
text_result() {
  printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"%s"}]}}\n' "$1" "$2"
}

answer_call() {
  local id=$1 line=$2 name=$3
  case "$name" in
    echo)
      local args
      args=$(arguments "$line")
      text_result "$id" "called with ${args//\"/}" ;;
    fail)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"the stub could not do it"}],"isError":true}}\n' "$id" ;;
    empty) printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[]}}\n' "$id" ;;
    rich)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"a picture"},{"type":"image","data":"aGk=","mimeType":"image/png"},{"type":"audio","data":"aGk=","mimeType":"audio/wav"},{"type":"resource_link","uri":"file:///tmp/notes.md","name":"notes.md"},{"type":"resource","resource":{"uri":"file:///tmp/x.txt","mimeType":"text/plain","text":"the embedded text"}},{"type":"resource","resource":{"uri":"file:///tmp/blob.bin","blob":"aGk="}},{"type":"something_new","value":1}],"isError":false}}\n' "$id" ;;
    structured) printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[],"structuredContent":{"count":2,"of":"things"}}}\n' "$id" ;;
    big) text_result "$id" "$(head -c 60000 /dev/zero | tr '\0' 'x')" ;;
    slow) sleep 5; text_result "$id" "eventually" ;;
    die) exit 3 ;;
    noise)
      printf 'Stub MCP server ready.\n'
      text_result "$id" "answered after the banner" ;;
    ping)
      printf '{"jsonrpc":"2.0","id":"s1","method":"ping"}\n'
      text_result "$id" "answered after asking" ;;
    *) text_result "$id" "the stub has no tool called $name" ;;
  esac
}

while IFS= read -r line; do
  if [ -n "${STUB_LOG:-}" ]; then
    printf '%s\n' "$line" >> "$STUB_LOG"
  fi
  if [ -n "${STUB_SILENT:-}" ]; then
    continue
  fi
  case "$line" in
    *'"method":"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"stub","version":"0.0.0"}}}\n' "$(number "$line")" "$version" ;;
    *'"method":"notifications/initialized"'*) ;;
    *'"method":"tools/list"'*)
      id=$(number "$line")
      junk=
      if [ -n "${STUB_JUNK:-}" ]; then
        junk=',{"description":"a tool with no name"},{"name":"","description":"a tool with an empty name"}'
      fi
      if [ -n "${STUB_FOREVER:-}" ] || [ -n "${STUB_PAGES:-}" ]; then
        # Paged: page 1 is what a request with no cursor asks for, and every
        # cursor names the page after the one it was handed out with.
        page=1
        if [[ "$line" == *'"cursor"'* ]]; then
          page=$(printf '%s' "$line" | sed -n 's/.*"cursor":"\([0-9]*\)".*/\1/p')
        fi
        if [ -n "${STUB_FOREVER:-}" ] || [ "$page" -lt "${STUB_PAGES:-1}" ]; then
          printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[%s%s],"nextCursor":"%s"}}\n' "$id" "$(listed "page$page")" "$junk" "$((page + 1))"
        else
          printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[%s%s]}}\n' "$id" "$(listed "page$page")" "$junk"
        fi
      else
        printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[%s%s]}}\n' "$id" "$(listed "$tools")" "$junk"
      fi ;;
    *'"method":"tools/call"'*)
      answer_call "$(number "$line")" "$line" "$(string name "$line")" ;;
    *)
      printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"no such method"}}\n' "$(number "$line")" ;;
  esac
done

if [ -n "${STUB_LOG:-}" ]; then
  printf 'eof\n' >> "$STUB_LOG"
fi
"#;
