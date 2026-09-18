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
            // Register the waiter before re-checking the flag, so a cancel
            // landing in between can never be missed.
            let notified = self.notify.notified();
            if self.is_cancelled() { return; }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_default_both_yield_an_uncancelled_token() {
        // Default::default() delegates to Self::new(), so the two
        // constructors are equivalent — one test covers both.
        let direct = CancelToken::new();
        let defaulted: CancelToken = Default::default();
        assert!(!direct.is_cancelled());
        assert!(!defaulted.is_cancelled());
    }

    #[test]
    fn cancel_is_idempotent() {
        let t = CancelToken::new();
        t.cancel();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn clones_share_the_same_cancel_state() {
        // the CancelToken holds Arc<AtomicBool> + Arc<Notify>; the derived
        // Clone must produce another handle to the same state, not a
        // separate one. We check both directions (cancel-from-original,
        // cancel-from-clone) so a buggy manual Clone that pointed each
        // clone at its own flag would fail.
        let a = CancelToken::new();
        let b = a.clone();
        a.cancel();
        assert!(b.is_cancelled(), "a clone sees the cancellation");
        // also: cancelling through a clone flips the original's flag
        let x = CancelToken::new();
        let y = x.clone();
        y.cancel();
        assert!(x.is_cancelled(), "cancelling through a clone flips the original");
    }

    #[test]
    fn cancelled_returns_immediately_when_already_cancelled() {
        // the `if self.is_cancelled() { return; }` short-circuit at the
        // top of the loop: a token that was cancelled before the call
        // must resolve without ever awaiting the Notify. 5ms is generous
        // — the check is a single atomic load.
        let t = CancelToken::new();
        t.cancel();
        let started = std::time::Instant::now();
        crate::test_util::block_on(async { t.cancelled().await });
        assert!(started.elapsed() < std::time::Duration::from_millis(5),
            "an already-cancelled token resolves immediately: {:?}", started.elapsed());
    }

    #[test]
    fn cancelled_suspends_and_wakes_on_cancel() {
        // the second branch of the loop: register the waiter, re-check the
        // flag, await the notification. A cancel arriving during the await
        // must wake the coroutine.
        let t = Arc::new(CancelToken::new());
        let tc = t.clone();
        crate::test_util::block_on(async move {
            let waiter = {
                let t = t.clone();
                tokio::spawn(async move { t.cancelled().await })
            };
            // give the waiter a beat to reach the notify
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            tc.cancel();
            let start = std::time::Instant::now();
            waiter.await.unwrap();
            assert!(start.elapsed() < std::time::Duration::from_millis(100),
                "the waiter woke promptly: {:?}", start.elapsed());
        });
    }

    #[test]
    fn cancelled_wakes_only_after_the_flag_is_set() {
        // the first branch of the loop: `if self.is_cancelled() { return; }`
        let t = CancelToken::new();
        crate::test_util::block_on(async {
            // immediate: the flag is still false, so we have to wait —
            // a sibling cancel wakes us up
            let tc = t.clone();
            let waiter = tokio::spawn(async move { t.cancelled().await });
            tokio::task::spawn_blocking(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                tc.cancel();
            });
            waiter.await.unwrap();
        });
    }
}
