//! The crate's one error type, and the `Result` alias that rides with it.
//!
//! It lives here (L0) rather than at the crate root because the root is
//! the composition root: an L4 module naming `crate::Error` would look
//! like an upward edge into the composition layer, and the layering net
//! cannot see a type that has no home of its own. The root re-exports
//! both, so every call site keeps its spelling — `crate::Error` now
//! resolves to *this* module.
//!
//! The variants are a closed vocabulary the whole crate speaks: the
//! wire's own failures (`Api`), a sentence we invented (`Msg`), the
//! transport's (`Http`), the JSON parser's (`Json`), the filesystem's
//! (`Io`), and the one that is not a failure at all — `Interrupted`,
//! the user pressing Ctrl-C. The frontends match on `Interrupted` to
//! stay silent instead of printing an error, so it must keep being its
//! own variant.

use std::io;

/// The crate's one error type.
#[derive(Debug)]
pub enum Error {
    Api(u16, String),
    Msg(String),
    Http(Box<ureq::Error>),
    Json(serde_json::Error),
    Io(io::Error),
    /// The coroutine was cancelled (Ctrl-C). Not a failure — a stop.
    Interrupted,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Api(c, b) => write!(f, "api {c}: {b}"),
            Self::Msg(m) => write!(f, "{m}"),
            Self::Http(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Interrupted => write!(f, "interrupted"),
        }
    }
}
impl std::error::Error for Error {}

impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self {
        match e {
            ureq::Error::Status(c, r) => Self::Api(c, r.into_string().unwrap_or_default()),
            other => Self::Http(Box::new(other)),
        }
    }
}
impl From<serde_json::Error> for Error { fn from(e: serde_json::Error) -> Self { Self::Json(e) } }
impl From<io::Error> for Error { fn from(e: io::Error) -> Self { Self::Io(e) } }

impl Error {
    /// Is this worth sending again? A 429 or a 5xx, or a socket that
    /// never answered, is not the request's fault — and the answer is
    /// the same on every wire, so it lives on the error type. Pure, so
    /// a test reaches it.
    pub(crate) fn is_transient(&self) -> bool {
        match self {
            Error::Http(_) => true,
            Error::Api(s, _) => *s == 408 || *s == 429 || *s >= 500,
            _ => false,
        }
    }
}

/// The crate's one result type.
pub type Result<T> = std::result::Result<T, Error>;

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        assert_eq!(Error::Api(429, "limited".into()).to_string(), "api 429: limited");
        assert_eq!(Error::Msg("bad".into()).to_string(), "bad");
        assert_eq!(Error::Interrupted.to_string(), "interrupted");
    }

    // ---- retry: the transport's own rule ----
    #[test]
    fn only_transient_errors_are_resent() {
        // the retryable kinds: a rate limit, a timeout, a server that is down
        for code in [408, 429, 500, 502, 503, 504] {
            assert!(Error::Api(code, "x".into()).is_transient(), "{code} should be resent");
        }
        // the request's own fault, or a stop: never resent
        for code in [400, 401, 403, 404, 422] {
            assert!(!Error::Api(code, "x".into()).is_transient(), "{code} must not be resent");
        }
        assert!(!Error::Msg("x".into()).is_transient());
        assert!(!Error::Interrupted.is_transient(), "a cancellation is not a failure to retry");
    }

    // ---- Error: the type's display & conversions -------------------------

    #[test]
    fn error_display_strings_are_actually_distinct() {
        // each variant has its own Display shape — the test pins them
        // so a refactor can't silently turn two variants into the same
        // string (which would defeat the user's ability to tell what
        // kind of failure they're looking at).
        assert_eq!(format!("{}", Error::Interrupted), "interrupted");
        // The passthrough variants delegate to their inner Display; the
        // outer prefix is the variant's own word — we just check the
        // inner value shows up verbatim, no "error: " or similar wrapper
        // added by Error's Display.
        let io = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        assert_eq!(format!("{}", Error::Io(io)), "denied");
        // Error::Json's body is the inner error verbatim
        let json_err = serde_json::from_str::<u32>("abc").unwrap_err();
        let json_str = format!("{}", json_err);
        assert_eq!(format!("{}", Error::Json(json_err)), json_str);
        // Error::Msg shows the wrapped message verbatim
        assert_eq!(format!("{}", Error::Msg("hi".into())), "hi");
        // Error::Api prefix error shape: "api {code}: {body}"
        assert_eq!(format!("{}", Error::Api(503, "unavailable".into())),
            "api 503: unavailable");
    }

    #[test]
    fn error_from_ureq_wraps_a_status_into_api() {
        // the Status arm of From<ureq::Error>: the status code and body
        // ride along as the Error::Api variant
        let body = r#"{"error":"model gone"}"#;
        let r = ureq::Response::new(404, "Not Found", body).unwrap();
        let err: Error = ureq::Error::Status(404, r).into();
        match err {
            Error::Api(code, ref b) => {
                assert_eq!(code, 404);
                assert!(b.contains("model gone"), "body carries: {b}");
            }
            other => panic!("expected Error::Api, got {other:?}"),
        }
    }

    #[test]
    fn error_from_ureq_wraps_other_kinds_into_http() {
        // transport / DNS / decode errors: they ride as Error::Http.
        // Transport cannot be constructed outside ureq, so we trigger a
        // real one — a request to a closed port.
        let err: Error = ureq::get("http://127.0.0.1:1/nope").call().unwrap_err().into();
        assert!(matches!(err, Error::Http(_)), "got {err:?}");
    }

    #[test]
    fn error_from_io_preserves_the_io_kind() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope");
        let err: Error = io.into();
        assert!(matches!(err, Error::Io(_)));
    }

    // ---- the variants nothing else reaches yet --------------------------

    /// A transport failure, without touching the network: a URL that does
    /// not parse fails inside `ureq` before a socket is ever opened.
    fn transport_error() -> ureq::Error {
        ureq::get("relative/without/a/base").call().unwrap_err()
    }

    #[test]
    fn the_http_variant_shows_the_transport_failure_verbatim() {
        // The Display arm for Http: the inner error's own text, with no
        // wrapper of ours around it — the user is reading the transport's
        // complaint, not ours.
        let inner = transport_error();
        let expected = inner.to_string();
        let err: Error = inner.into();
        assert_eq!(err.to_string(), expected,
            "Error::Http shows the transport error exactly as it came");
        assert!(!expected.is_empty(), "the transport error says something");
        assert_ne!(err.to_string(), "interrupted",
            "and it is not another variant's text");
    }

    #[test]
    fn a_json_failure_becomes_the_json_variant() {
        // The From<serde_json::Error> conversion: a parse failure keeps
        // the parser's own message, and stays a Json error (not Io, not
        // Http) so the frontends can tell "the answer was not json" from
        // "the wire broke".
        let inner = serde_json::from_str::<u32>("not json at all").unwrap_err();
        let expected = inner.to_string();
        let err: Error = inner.into();
        assert!(matches!(err, Error::Json(_)), "got {err:?}");
        assert_eq!(err.to_string(), expected, "the parser's message rides along");
    }

    #[test]
    fn a_transport_failure_is_worth_sending_again() {
        // The Http arm of is_transient: a request that never made it out
        // (or never came back) is not the request's fault — resend it.
        let err: Error = transport_error().into();
        assert!(err.is_transient(), "a transport failure is retried: {err}");
        // and the other variants stay where they are
        assert!(!Error::Msg("x".into()).is_transient());
        assert!(!Error::Api(404, "x".into()).is_transient());
    }
}
