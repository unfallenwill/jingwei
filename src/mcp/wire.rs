//! The JSON-RPC 2.0 vocabulary MCP is carried in.
//!
//! Everything that can be said about a message by looking at the message is
//! said here: what it is ([`classify`]), how one is built, what to answer when
//! the server is the one asking ([`serve`]), and which protocol versions this
//! client can run ([`negotiate`]). Nothing here holds a connection or a byte:
//! the transports move values of this shape, and the two of them share these
//! rules rather than each having their own.
//!
//! Two of the spec's rules are why the shapes are built rather than spelled out
//! at each call site: a request carries an id and a notification must not, and
//! a response carries exactly one of `result` and `error`.

use serde_json::{Value, json};

/// The version this client asks for: the latest it knows, which is what the
/// specification asks a client to send.
///
/// It is not the only one it will run: a server that answers with an older one
/// it supports is a server to talk to, not one to hang up on. See
/// [`SUPPORTED_VERSIONS`].
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// The versions this client can run.
///
/// The tool surface is the same in all three — `initialize`, then
/// `tools/list`, then `tools/call` — which is the whole of what this client
/// uses, so the versions differ for it in nothing that matters. A version
/// outside this list is a server that may speak a shape this client would
/// misread, and that is a connection to refuse rather than to guess at.
pub const SUPPORTED_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// The only value the protocol version field takes.
pub const JSONRPC: &str = "2.0";

/// The server was asked for something it does not serve.
pub const METHOD_NOT_FOUND: i64 = -32601;

/// What a refusal is given when the server's message carries no code of its
/// own: "something went wrong at the far end" is the only thing left to say.
pub const INTERNAL_ERROR: i64 = -32603;

/// One request, as it goes out.
pub fn request(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": JSONRPC, "id": id, "method": method, "params": params})
}

/// One notification, as it goes out: a method and no id, which is what makes it
/// a notification rather than a request.
pub fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": JSONRPC, "method": method, "params": params})
}

/// One answer to something the server asked.
pub fn success(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": JSONRPC, "id": id, "result": result})
}

/// One refusal, in the shape a server reads: a code and a sentence.
pub fn failure(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": JSONRPC, "id": id, "error": {"code": code, "message": message}})
}

/// What a message that arrived from the server is.
#[derive(Debug, Clone, PartialEq)]
pub enum Incoming<'a> {
    /// The answer to a request this client sent: the id says which. A `None`
    /// id is one that does not name a request this client has open (a server
    /// that answers a question no one asked, or echoes back in a shape we did
    /// not send), and is read past the same way noise is.
    Answer {
        id: Option<u64>,
        outcome: Outcome<'a>,
    },
    /// A request *from* the server, which it is waiting on an answer for: a
    /// client that says nothing leaves the server waiting forever. The id is
    /// the server's own and may be a string or a number — the protocol
    /// allows both — so it is kept as a `Value` and echoed back unchanged.
    Ask { id: &'a Value, method: &'a str },
    /// A notification: it says something happened and expects nothing back.
    Notice,
    /// Not a message this vocabulary defines. A server that writes one is
    /// broken, and what a client can do about it is nothing at all.
    Noise,
}

/// The two ways a request can come back.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome<'a> {
    /// The result, as the method's own shape defines it.
    Ok(&'a Value),
    /// The server refused, in its own words — which are the words to report.
    Failed { code: i64, message: String },
}

impl Outcome<'_> {
    /// The refusal as one line: what the model (and the user) is told.
    pub fn reason(&self) -> String {
        match self {
            Outcome::Failed { code, message } => format!("{message} (JSON-RPC error {code})"),
            Outcome::Ok(_) => String::new(),
        }
    }
}

/// Read one message for what it is.
///
/// The three fields do the whole of the work: `jsonrpc` says it is one of ours,
/// `id` says something is being answered or asked, and `method` says a request
/// while only `result`/`error` say an answer.
///
/// A response carries the same id as the request it answers, and ids in MCP are
/// numbers *or* strings — the protocol was widened past the JSON-RPC default of
/// numbers when it had to be, and a server that answers a request this client
/// sent with a string id is one we still read as an answer. An id that does not
/// parse as `u64` arrives as `Incoming::Answer { id: None }`: the request map
/// looks the id up by number and misses, the same way it would for an answer
/// nobody is waiting on, and the message is read past. The same reasoning
/// covers fractional, null, and absent ids: not an answer to anything we asked,
/// not noise worth reporting — just a line to skip.
pub fn classify(value: &Value) -> Incoming<'_> {
    let Some(object) = value.as_object() else {
        return Incoming::Noise;
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some(JSONRPC) {
        return Incoming::Noise;
    }
    let id = object.get("id");
    if let Some(method) = object.get("method").and_then(Value::as_str) {
        return match id {
            Some(id) => Incoming::Ask { id, method },
            None => Incoming::Notice,
        };
    }
    // A response is one with `result` or `error`; the id may be missing entirely
    // (a server that lost its place), null (a JSON-RPC default this protocol
    // forbids), a string (legal in MCP), or a number (the JSON-RPC default).
    // What it cannot be is `0`, the JSON-RPC error for "invalid request" — a
    // server that sent `id: 0` is one we should already have noticed, and the
    // map lookup would have dropped it anyway. We accept all of them and let
    // the caller's pending map decide which, if any, are this client's.
    let id = id.and_then(Value::as_u64);
    // An `error` is read before a `result`: a message carrying both is a
    // server contradicting itself, and the refusal is the half that says a
    // request did not succeed.
    if let Some(error) = object.get("error") {
        let code = error
            .get("code")
            .and_then(Value::as_i64)
            .unwrap_or(INTERNAL_ERROR);
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the server refused the request and said nothing about why")
            .to_string();
        return Incoming::Answer {
            id,
            outcome: Outcome::Failed { code, message },
        };
    }
    match object.get("result") {
        Some(result) => Incoming::Answer {
            id,
            outcome: Outcome::Ok(result),
        },
        None => Incoming::Noise,
    }
}

/// What to answer a server that asked us something.
///
/// `ping` is the only request this client serves: it is the one every server is
/// allowed to send at any time and the only one whose answer asks nothing of
/// us. The rest of what a server may ask — `roots/list`, `sampling/createMessage`,
/// `elicitation/create` — is refused by name rather than ignored, because the
/// alternative is a server waiting for an answer that is not coming. The client
/// advertises none of them (see `client::initialize`), so a server that asks
/// anyway has misread the handshake, and being told so is how it finds out.
pub fn serve(id: &Value, method: &str) -> Value {
    match method {
        "ping" => success(id, json!({})),
        other => failure(
            id,
            METHOD_NOT_FOUND,
            &format!("this client does not serve {other}; it offers tools only"),
        ),
    }
}

/// The protocol version a server answered with, or why this client cannot run
/// it.
///
/// The server picks: it answers with the version asked for when it supports it
/// and with its own latest otherwise, and a client that does not support the
/// answer should disconnect rather than carry on in a shape it may misread.
pub fn negotiate(answered: &Value) -> Result<String, String> {
    let Some(version) = answered.get("protocolVersion").and_then(Value::as_str) else {
        return Err("the server's answer to initialize names no protocolVersion".into());
    };
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(format!(
            "the server speaks protocol {version}, which this client does not \
             (it speaks {})",
            SUPPORTED_VERSIONS.join(", ")
        ));
    }
    Ok(version.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_carries_an_id_and_a_notification_does_not() {
        let asked = request(7, "tools/list", json!({}));
        assert_eq!(asked["id"], json!(7));
        assert_eq!(asked["method"], json!("tools/list"));
        let told = notification("notifications/initialized", json!({}));
        assert!(told.get("id").is_none(), "{told}");
        assert_eq!(told["jsonrpc"], json!(JSONRPC));
    }

    #[test]
    fn an_answer_carries_exactly_one_of_result_and_error() {
        let ok = success(&json!(3), json!({"tools": []}));
        assert_eq!(ok["id"], json!(3));
        assert!(ok.get("error").is_none());
        let no = failure(&json!(3), METHOD_NOT_FOUND, "no such method");
        assert_eq!(no["error"]["code"], json!(METHOD_NOT_FOUND));
        assert!(no.get("result").is_none());
    }

    #[test]
    fn an_answer_is_read_by_its_id() {
        let message = json!({"jsonrpc": "2.0", "id": 4, "result": {"tools": []}});
        assert_eq!(
            classify(&message),
            Incoming::Answer {
                id: Some(4),
                outcome: Outcome::Ok(&json!({"tools": []}))
            }
        );
    }

    #[test]
    fn a_string_id_is_still_an_answer_to_a_known_request() {
        // The protocol allows string ids (basic.md §responses): a server that
        // echoes our number as a string is allowed, and one that answers a
        // request we never sent with a string is a string id that does not
        // parse as u64 — the pending map will miss it. Either way, it is read
        // as an answer, not as noise.
        let ok = json!({"jsonrpc": "2.0", "id": "s1", "result": {}});
        assert_eq!(
            classify(&ok),
            Incoming::Answer {
                id: None,
                outcome: Outcome::Ok(&json!({}))
            }
        );
        let bad = json!({"jsonrpc": "2.0", "id": "s1", "error": {"code": -32602, "message": "x"}});
        let Incoming::Answer { id, outcome } = classify(&bad) else {
            panic!("not read as an answer");
        };
        assert_eq!(id, None);
        assert!(matches!(outcome, Outcome::Failed { .. }));
    }

    #[test]
    fn a_refusal_is_read_with_its_code_and_its_words() {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32602, "message": "no such tool"}
        });
        let Incoming::Answer { id, outcome } = classify(&message) else {
            panic!("not read as an answer");
        };
        assert_eq!(id, Some(1));
        assert_eq!(
            outcome,
            Outcome::Failed {
                code: -32602,
                message: "no such tool".into()
            }
        );
        assert_eq!(outcome.reason(), "no such tool (JSON-RPC error -32602)");
    }

    #[test]
    fn a_refusal_with_nothing_to_say_still_says_something() {
        // The code and the message are both required by the protocol; a server
        // that leaves one out is still a server whose answer has to be read.
        let message = json!({"jsonrpc": "2.0", "id": 1, "error": {}});
        let Incoming::Answer { outcome, .. } = classify(&message) else {
            panic!("not read as an answer");
        };
        assert!(
            outcome.reason().contains("said nothing about why"),
            "{outcome:?}"
        );
        assert!(outcome.reason().contains(&INTERNAL_ERROR.to_string()));
    }

    #[test]
    fn an_error_wins_over_a_result_beside_it() {
        let message = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {},
            "error": {"code": -32603, "message": "both"}
        });
        let Incoming::Answer { outcome, .. } = classify(&message) else {
            panic!("not read as an answer");
        };
        assert!(matches!(outcome, Outcome::Failed { .. }), "{outcome:?}");
        assert_eq!(Outcome::Ok(&json!(null)).reason(), "");
    }

    #[test]
    fn a_server_asking_is_told_apart_from_a_server_announcing() {
        let asking = json!({"jsonrpc": "2.0", "id": 9, "method": "ping"});
        assert_eq!(
            classify(&asking),
            Incoming::Ask {
                id: &json!(9),
                method: "ping"
            }
        );
        let telling = json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"});
        assert_eq!(classify(&telling), Incoming::Notice);
    }

    #[test]
    fn what_is_not_a_message_is_noise() {
        for junk in [
            json!("a string the server meant to be JSON"),
            json!({"id": 1, "result": {}}), // no jsonrpc
            json!({"jsonrpc": "1.0", "id": 1, "result": {}}), // not 2.0
            json!({"jsonrpc": "2.0", "id": 1}), // neither result nor error
        ] {
            assert_eq!(classify(&junk), Incoming::Noise, "{junk}");
        }
        // A message that parses as an answer but with an id we never sent.
        // It is an answer — to nobody this client asked — and is read past
        // by the pending map rather than reported as a wire-format failure.
        let unknown = json!({"jsonrpc": "2.0", "id": "x", "result": {}});
        assert_eq!(
            classify(&unknown),
            Incoming::Answer {
                id: None,
                outcome: Outcome::Ok(&json!({}))
            }
        );
        // A null id is not a valid JSON-RPC 2.0 id, but the protocol allows it
        // through the same door: an answer with nobody on the other end.
        let null_id = json!({"jsonrpc": "2.0", "id": null, "result": {}});
        assert_eq!(
            classify(&null_id),
            Incoming::Answer {
                id: None,
                outcome: Outcome::Ok(&json!({}))
            }
        );
    }

    #[test]
    fn ping_is_the_one_the_client_serves() {
        let answer = serve(&json!(5), "ping");
        assert_eq!(answer["result"], json!({}));
        let refused = serve(&json!(5), "sampling/createMessage");
        assert_eq!(refused["error"]["code"], json!(METHOD_NOT_FOUND));
        assert!(
            refused["error"]["message"]
                .as_str()
                .unwrap()
                .contains("sampling/createMessage"),
            "{refused}"
        );
    }

    #[test]
    fn only_the_versions_this_client_runs_are_negotiated() {
        assert_eq!(
            negotiate(&json!({"protocolVersion": PROTOCOL_VERSION})).unwrap(),
            PROTOCOL_VERSION
        );
        // A server that answered with an older one it supports is one to talk
        // to: the tool surface it will see is the same.
        assert_eq!(
            negotiate(&json!({"protocolVersion": "2024-11-05"})).unwrap(),
            "2024-11-05"
        );
        let newer = negotiate(&json!({"protocolVersion": "2099-01-01"})).unwrap_err();
        assert!(newer.contains("2099-01-01"), "{newer}");
        let silent = negotiate(&json!({})).unwrap_err();
        assert!(silent.contains("protocolVersion"), "{silent}");
    }
}
