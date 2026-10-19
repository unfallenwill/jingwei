//! The tool boundary: the seam a new tool family plugs into.
//!
//! A *provider* supplies tools; a *router* holds providers, asks them for
//! their schemas, and dispatches a call to the one that owns the name.
//! Neither name mentions a concrete tool. The built-ins (`tools`), the MCP
//! servers (`mcp`), and — one day — the agent family each implement
//! [`ToolProvider`], and the router composes them. Nothing in `core` ever
//! learns what a provider is made of: it asks for schemas and asks for a
//! call, and that is the whole conversation.
//!
//! ## Why the futures are boxed
//!
//! The built-in tools are synchronous; the MCP path is asynchronous (the
//! hub's actor task owns the connection and answers a call). A provider
//! list is therefore heterogeneous — `Vec<Box<dyn ToolProvider>>` — and
//! `async fn` in a trait is **not** dyn-compatible. The trait hands back a
//! boxed future instead, which keeps the list heterogeneous without
//! giving up the ability to await a call. A provider whose work is
//! synchronous returns a ready future; that is the honest shape for it,
//! and it costs one allocation.
//!
//! The router is L2: it knows the trait, the cancel token and the JSON
//! value, and nothing else. It may not name `tools` or `mcp` — that is
//! the whole point of the seam.

use crate::cancel::CancelToken;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

/// A boxed future — the shape every async method on [`ToolProvider`]
/// hands back.
pub(crate) type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A source of tools.
pub(crate) trait ToolProvider: Send + Sync {
    /// Does this provider own this tool name? The router asks every
    /// provider, in order, until one says yes.
    fn owns(&self, name: &str) -> bool;

    /// The schemas the model sees for this provider's tools.
    fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>>;

    /// Run one of this provider's tools. The result is text: a failure is
    /// text that starts with `error:` — the shape the model already reads.
    fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a Value,
        token: &'a CancelToken,
    ) -> BoxFuture<'a, String>;
}

/// The composed tool surface: the providers, in the order their schemas
/// should appear.
///
/// The order is a contract, not a preference. The provider list's order
/// is the order of the combined schema list the model is shown, and the
/// cached prefix is system + tools — a byte change anywhere in that list
/// invalidates the whole prefix. The built-ins come first and the MCP
/// tools after them, exactly as they did before the router existed.
pub(crate) struct ToolRouter {
    providers: Vec<Box<dyn ToolProvider>>,
}

impl ToolRouter {
    /// Compose a router over a list of providers. The list's order is
    /// the order of the schemas.
    pub(crate) fn new(providers: Vec<Box<dyn ToolProvider>>) -> Self {
        Self { providers }
    }

    /// Every provider's schemas, in provider order. One await per
    /// provider, in sequence: the MCP provider's list comes from the hub's
    /// actor, so this is one round trip — the same one the context's own
    /// refresh used to cost.
    pub(crate) fn definitions(&self) -> BoxFuture<'_, Vec<Value>> {
        Box::pin(async move {
            let mut out = Vec::new();
            for p in &self.providers {
                out.extend(p.definitions().await);
            }
            out
        })
    }

    /// Run one tool: the first provider that owns the name runs it. A
    /// name nobody owns is answered in the shape the model already reads
    /// for an unknown tool.
    pub(crate) fn call<'a>(
        &'a self,
        name: &'a str,
        input: &'a Value,
        token: &'a CancelToken,
    ) -> BoxFuture<'a, String> {
        Box::pin(async move {
            for p in &self.providers {
                if p.owns(name) {
                    return p.call(name, input, token).await;
                }
            }
            format!("error: unknown tool '{name}'")
        })
    }
}

// ---- tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A provider that owns two names and answers with its own name — and
    /// counts how often it was asked for its schemas.
    struct Stub {
        names: Vec<String>,
        schema: String,
        asked: Arc<AtomicUsize>,
    }

    impl Stub {
        fn new(names: &[&str], schema: &str) -> (Self, Arc<AtomicUsize>) {
            let asked = Arc::new(AtomicUsize::new(0));
            (Stub { names: names.iter().map(|s| s.to_string()).collect(), schema: schema.into(), asked: asked.clone() }, asked)
        }
    }

    impl ToolProvider for Stub {
        fn owns(&self, name: &str) -> bool {
            self.names.iter().any(|n| n == name)
        }
        fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            Box::pin(std::future::ready(vec![serde_json::json!({"name": self.schema})]))
        }
        fn call<'a>(
            &'a self,
            name: &'a str,
            _input: &'a Value,
            _token: &'a CancelToken,
        ) -> BoxFuture<'a, String> {
            let out = format!("{}:{}", self.schema, name);
            Box::pin(std::future::ready(out))
        }
    }

    fn router(providers: Vec<Box<dyn ToolProvider>>) -> ToolRouter {
        ToolRouter::new(providers)
    }

    #[test]
    fn the_router_asks_the_providers_in_order() {
        let (a, asked_a) = Stub::new(&["x"], "a");
        let (b, asked_b) = Stub::new(&["y"], "b");
        let r = router(vec![Box::new(a), Box::new(b)]);
        let defs = crate::test_util::block_on(r.definitions());
        // the composed list keeps the providers' order
        let names: Vec<&str> = defs.iter().filter_map(|v| v["name"].as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(asked_a.load(Ordering::SeqCst), 1);
        assert_eq!(asked_b.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_router_routes_a_call_to_the_provider_that_owns_the_name() {
        let (a, _) = Stub::new(&["x"], "a");
        let (b, _) = Stub::new(&["y"], "b");
        let r = router(vec![Box::new(a), Box::new(b)]);
        let token = CancelToken::new();
        let out = crate::test_util::block_on(r.call("y", &serde_json::json!({}), &token));
        assert_eq!(out, "b:y", "the call reached the owner of the name");
    }

    #[test]
    fn the_router_answers_an_unknown_name_with_the_model_facing_error() {
        let (a, _) = Stub::new(&["x"], "a");
        let r = router(vec![Box::new(a)]);
        let token = CancelToken::new();
        let out = crate::test_util::block_on(r.call("nope", &serde_json::json!({}), &token));
        assert_eq!(out, "error: unknown tool 'nope'");
    }

    #[test]
    fn the_router_does_not_call_a_provider_that_owns_nothing() {
        // a provider that owns nothing must never be asked to run a tool:
        // the router walks the list by `owns`, and stops at the first
        // owner. A provider that owns nothing contributes only its schemas.
        struct Never;
        impl ToolProvider for Never {
            fn owns(&self, _name: &str) -> bool { false }
            fn definitions<'a>(&'a self) -> BoxFuture<'a, Vec<Value>> {
                Box::pin(std::future::ready(vec![]))
            }
            fn call<'a>(
                &'a self,
                name: &'a str,
                _input: &'a Value,
                _token: &'a CancelToken,
            ) -> BoxFuture<'a, String> {
                Box::pin(std::future::ready(format!("never:{name}")))
            }
        }
        let (a, _) = Stub::new(&["x"], "a");
        let r = router(vec![Box::new(Never), Box::new(a)]);
        let token = CancelToken::new();
        let out = crate::test_util::block_on(r.call("nope", &serde_json::json!({}), &token));
        assert_eq!(out, "error: unknown tool 'nope'", "the provider that owns nothing was not called");
    }
}
