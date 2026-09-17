// One-shot cooperative cancellation. `cancel()` fires when the user hits
// Ctrl-C; code inside the coroutine races its real work against
// `cancelled().await` (suspends until cancelled) or peeks with
// `is_cancelled()` (no suspension). The agent is an async coroutine: a
// computation that suspends at every `await` and can be abandoned at any of
// those suspension points. Suspension points double as cancellation points —
// `CancelToken` is the cooperative channel between the outside world
// (Ctrl-C) and the running coroutine, so the agent can be interrupted at
// any moment, at a safe point of its own choosing, with the conversation
// history left valid.
//
// Cloning is cheap: the token wraps an `Arc`, so a copy handed to a child
// coroutine is the same cancel state, and any holder may cancel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

#[derive(Debug, Clone)]
pub(crate) struct CancelToken {
    flag: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Default for CancelToken {
    fn default() -> Self { Self::new() }
}

impl CancelToken {
    pub(crate) fn new() -> Self {
        Self { flag: Arc::new(AtomicBool::new(false)), notify: Arc::new(Notify::new()) }
    }

    /// Request cancellation. Idempotent; wakes every suspended `cancelled()`.
    pub(crate) fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Resolves once the token is cancelled. Never spins — it suspends.
    pub(crate) async fn cancelled(&self) {
        loop {
            if self.is_cancelled() { return; }
            // Register the waiter *before* re-checking the flag, so a cancel
            // landing in between can never be missed.
            let notified = self.notify.notified();
            if self.is_cancelled() { return; }
            notified.await;
        }
    }
}
