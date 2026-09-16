//! The internal IR, typed.
//!
//! The agent core's conversation used to be `Vec<serde_json::Value>` — every
//! read a string subscript (`m["content"][0]["type"]`), every typo a runtime
//! `null`. This module gives that shape names: a [`Message`] is a turn, a
//! [`Block`] is one piece of assistant content, a [`ToolResult`] is a
//! result. [`Response`] is what a wire hands back.
//!
//! The JSON is still the boundary — sessions persist it, wires speak it —
//! but the string subscripts live *only here*, in [`Message::from_value`] /
//! [`Message::to_value`] and their block counterparts. Everything else is
//! typed.
//!
//! `to_value` reproduces the exact JSON the core used to spell by hand
//! (serde_json's sorted keys make it canonical), so session files round-trip
//! byte for byte and prefix caches stay warm. `Block::Other` carries any
//! block shape the core does not name — verbatim, so an echoed turn is never
//! narrowed to what this version happens to know.

use serde_json::{json, Value};

/// One block of assistant content — the model's own vocabulary, typed.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Text(String),
    Thinking { text: String, signature: Option<String> },
    ToolUse { id: String, name: String, input: Value },
    /// A block shape the core does not name (a vendor's exotic block, a
    /// newer API's addition): kept whole so the echo stays faithful.
    Other(Value),
}

impl Block {
    /// Decode one wire block. Unknown shapes fall through to [`Block::Other`].
    pub fn from_value(v: &Value) -> Block {
        match v["type"].as_str() {
            Some("text") => Block::Text(v["text"].as_str().unwrap_or("").to_string()),
            Some("thinking") => Block::Thinking {
                text: v["thinking"].as_str().unwrap_or("").to_string(),
                signature: v.get("signature").and_then(Value::as_str).map(str::to_owned),
            },
            Some("tool_use") => Block::ToolUse {
                id: v["id"].as_str().unwrap_or("").to_string(),
                name: v["name"].as_str().unwrap_or("").to_string(),
                input: v.get("input").cloned().unwrap_or(Value::Null),
            },
            _ => Block::Other(v.clone()),
        }
    }

    /// Encode back to the wire shape. The four named block kinds emit
    /// exactly the keys the core always wrote; `Other` is verbatim.
    pub fn to_value(&self) -> Value {
        match self {
            Block::Text(t) => json!({"type": "text", "text": t}),
            Block::Thinking { text, signature } => {
                let mut o = json!({"type": "thinking", "thinking": text});
                if let Some(s) = signature {
                    o["signature"] = json!(s);
                }
                o
            }
            Block::ToolUse { id, name, input } => {
                json!({"type": "tool_use", "id": id, "name": name, "input": input})
            }
            Block::Other(v) => v.clone(),
        }
    }

    /// The tool call this block is, if it is one: (id, name, input).
    pub fn tool_use(&self) -> Option<(&str, &str, &Value)> {
        match self {
            Block::ToolUse { id, name, input } => Some((id, name, input)),
            _ => None,
        }
    }

    pub fn is_thinking(&self) -> bool {
        matches!(self, Block::Thinking { .. })
    }
}

/// A tool result block — a `user` turn that feeds tool output back.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolResult {
    pub id: String,
    pub content: String,
}

/// One turn of internal history.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    /// The user's task — plain text.
    User(String),
    /// The model's turn: text, reasoning, and/or tool calls.
    Assistant(Vec<Block>),
    /// Tool output going back to the model (role `user`, tool_result blocks).
    ToolResults(Vec<ToolResult>),
}

impl Message {
    /// Decode one persisted/wire message. Roles the core never writes (a
    /// stray foreign line) decode to an empty assistant turn rather than
    /// panicking — payload-blind loading is session.rs's contract.
    pub fn from_value(v: &Value) -> Message {
        match v["role"].as_str() {
            Some("assistant") => Message::Assistant(
                v["content"]
                    .as_array()
                    .map(|a| a.iter().map(Block::from_value).collect())
                    .unwrap_or_default(),
            ),
            _ => match v["content"].as_str() {
                Some(s) => Message::User(s.to_string()),
                None => Message::ToolResults(
                    v["content"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|b| b["type"] == "tool_result")
                        .map(|b| ToolResult {
                            id: b["tool_use_id"].as_str().unwrap_or("").to_string(),
                            content: b["content"].as_str().unwrap_or("").to_string(),
                        })
                        .collect(),
                ),
            },
        }
    }

    /// Encode to the persistent/wire shape.
    pub fn to_value(&self) -> Value {
        match self {
            Message::User(s) => json!({"role": "user", "content": s}),
            Message::Assistant(blocks) => json!({
                "role": "assistant",
                "content": blocks.iter().map(Block::to_value).collect::<Vec<_>>(),
            }),
            Message::ToolResults(results) => json!({
                "role": "user",
                "content": results.iter()
                    .map(|r| json!({"type": "tool_result", "tool_use_id": r.id, "content": r.content}))
                    .collect::<Vec<_>>(),
            }),
        }
    }
}

/// The whole history as one JSON array — the shape `est_tokens` measures and
/// session tests compare.
pub fn history_value(history: &[Message]) -> Value {
    Value::Array(history.iter().map(Message::to_value).collect())
}

/// One wire turn's answer: the assistant's blocks and the request's usage
/// ledger (internal shape, Anthropic keys — what `Usage::from_value` reads).
#[derive(Clone, Debug)]
pub struct Response {
    pub blocks: Vec<Block>,
    pub usage: Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_round_trips_every_shape() {
        let values = vec![
            json!({"role": "user", "content": "task"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "exit=0"}]}),
        ];
        for v in &values {
            assert_eq!(&Message::from_value(v).to_value(), v, "round-trips byte for byte");
        }
    }

    #[test]
    fn thinking_without_signature_omits_the_key() {
        let v = json!({"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]});
        assert_eq!(Message::from_value(&v).to_value(), v);
    }

    #[test]
    fn unknown_blocks_survive_verbatim() {
        let v = json!({"role": "assistant", "content": [{"type": "redacted_thinking", "data": "zz"}]});
        assert_eq!(Message::from_value(&v).to_value(), v, "Other carries the block whole");
    }

    #[test]
    fn block_tool_use_accessor() {
        let b = Block::from_value(&json!({"type": "tool_use", "id": "t", "name": "bash", "input": {}}));
        assert_eq!(b.tool_use(), Some(("t", "bash", &json!({}))));
        assert!(!Block::Text("x".into()).is_thinking());
    }
}
