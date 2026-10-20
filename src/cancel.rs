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
//
// A token is a *leaf*. The tree above it is the [`CancelScope`]: a node
// that owns what is beneath it. Cancelling a scope cancels everything
// under it; cancelling a leaf leaves its siblings alone. A leaf minted
// under an already-cancelled scope is born cancelled — the race a bare
// token cannot cover.
//
// The cancellable call onto the blocking pool lives here too: it is a
// cancellation helper, not a transport concern, and it is shared by the
// api transport (every HTTP body) and the bash tool (reaping the killed
// child).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

use crate::{Error, Result};

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

/// A node in the cancel tree. Cancelling a scope cancels everything
/// beneath it. Cloning a scope gives another handle on the same tree.
///
/// The scope is the *parent*; a [`CancelToken`] is the leaf. A token is
/// never a parent of another token — the "clones share one flag" rule
/// above is untouched — so "stop this but not that" and "stop everything
/// under me" are both expressible, and neither is a special case of the
/// other.
#[derive(Clone)]
pub(crate) struct CancelScope {
    inner: Arc<ScopeInner>,
}

/// What one scope owns: its own cancelled bit, and the children it has
/// minted. The child list is behind a mutex because the mint and the
/// cancel must not interleave — see [`CancelScope::token`].
struct ScopeInner {
    flag: AtomicBool,
    children: Mutex<Vec<Child>>,
}

/// One child of a scope: a leaf token, or a nested scope.
enum Child {
    Token(CancelToken),
    Scope(CancelScope),
}

impl CancelScope {
    /// A fresh scope: not cancelled, with no children yet.
    pub(crate) fn new() -> Self {
        Self { inner: Arc::new(ScopeInner { flag: AtomicBool::new(false), children: Mutex::new(Vec::new()) }) }
    }

    /// A leaf under this scope.
    ///
    /// A leaf minted under an already-cancelled scope is **born
    /// cancelled**: the scope's flag is read and the leaf is registered
    /// under one lock, so a cancel that lands in between can neither be
    /// missed nor let the leaf through. A `check the flag; … then push`
    /// that let go of the lock in between is the bug this shape exists
    /// to prevent.
    pub(crate) fn token(&self) -> CancelToken {
        let token = CancelToken::new();
        let mut children = self.inner.children.lock().unwrap();
        children.push(Child::Token(token.clone()));
        if self.inner.flag.load(Ordering::Acquire) { token.cancel(); }
        token
    }

    /// A nested scope: a node under this scope. A node minted under an
    /// already-cancelled scope is born cancelled, exactly as a leaf is.
    pub(crate) fn child(&self) -> CancelScope {
        let scope = CancelScope::new();
        let mut children = self.inner.children.lock().unwrap();
        children.push(Child::Scope(scope.clone()));
        if self.inner.flag.load(Ordering::Acquire) { scope.cancel(); }
        scope
    }

    /// Cancel this scope and everything under it.
    ///
    /// The walk happens under the children lock, so a mint that arrives
    /// during it waits, and finds the flag set when it gets in — the
    /// leaf it wanted is born cancelled rather than live. The lock order
    /// is strictly parent → child and no child ever reaches up, so the
    /// walk cannot deadlock.
    pub(crate) fn cancel(&self) {
        let children = self.inner.children.lock().unwrap();
        if self.inner.flag.load(Ordering::Acquire) { return; }  // idempotent
        self.inner.flag.store(true, Ordering::Release);
        for child in children.iter() {
            match child {
                Child::Token(t) => t.cancel(),
                Child::Scope(s) => s.cancel(),   // cascades into the grandchildren
            }
        }
    }

    /// Has this scope been cancelled? A cancelled *child* does not set
    /// it: only this scope's own cancel does.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.inner.flag.load(Ordering::Acquire)
    }
}

/// Run a synchronous closure on the blocking pool with the same
/// cancellation shape the streaming callers already use: the work
/// runs on a worker thread, the await is a real suspension point, and
/// the token races it so a Ctrl-C mid-reap returns `Interrupted`.
/// Shared between the api transport (every HTTP body) and the bash tool
/// (reaping the killed child).
pub(crate) async fn blocking<T, F>(token: &CancelToken, f: F) -> Result<T>
where F: FnOnce() -> Result<T> + Send + 'static, T: Send + 'static,
{
    tokio::select! {
        res = tokio::task::spawn_blocking(f) => res.unwrap_or_else(|e| Err(Error::Msg(format!("worker: {e}")))),
        _ = token.cancelled() => Err(Error::Interrupted),
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
        // and the re-check after the waiter is registered. A `cancelled()`
        // that came back while the flag was still false would be a waiter
        // waking for nothing — so the test watches both halves of the
        // name: while the flag is false the waiter is still parked, and
        // the flag is true the moment the waiter comes back.
        let t = CancelToken::new();
        let tc = t.clone();
        crate::test_util::block_on(async move {
            let waiter = {
                let t = t.clone();
                tokio::spawn(async move { t.cancelled().await })
            };
            // give the waiter a beat to reach the notify
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert!(!t.is_cancelled(), "the flag is still false");
            assert!(!waiter.is_finished(), "the waiter is still parked on the flag");
            tc.cancel();
            waiter.await.unwrap();
            assert!(t.is_cancelled(), "the waiter came back with the flag set");
        });
    }

    // ---- the scope: the tree above the leaf -----------------------------

    #[test]
    fn a_token_under_a_scope_is_cancelled_when_the_scope_is() {
        let scope = CancelScope::new();
        let t = scope.token();
        assert!(!t.is_cancelled(), "a fresh leaf is live");
        scope.cancel();
        assert!(t.is_cancelled(), "cancelling the scope reaches the leaf");
    }

    #[test]
    fn a_token_minted_under_a_cancelled_scope_is_born_cancelled() {
        let scope = CancelScope::new();
        scope.cancel();
        let t = scope.token();
        assert!(t.is_cancelled(), "a leaf minted under a dead scope is not live");
    }

    #[test]
    fn cancelling_one_sibling_token_leaves_the_other_and_the_scope_alone() {
        let scope = CancelScope::new();
        let a = scope.token();
        let b = scope.token();
        a.cancel();
        assert!(a.is_cancelled());
        assert!(!b.is_cancelled(), "the sibling leaf is untouched");
        assert!(!scope.is_cancelled(), "the scope is not cancelled by a leaf");
    }

    #[test]
    fn a_parent_cancel_reaches_a_grandchild_token() {
        // the subtle one: the parent does not hold its children's *tokens*,
        // it holds their *scopes* and cancels them by calling their cancel,
        // which cascades.
        let parent = CancelScope::new();
        let child = parent.child();
        let grandchild = child.token();
        parent.cancel();
        assert!(child.is_cancelled(), "the parent reaches the child scope");
        assert!(grandchild.is_cancelled(), "the parent reaches the grandchild leaf");
    }

    #[test]
    fn a_child_cancel_leaves_the_parent_and_the_parents_other_children_alone() {
        let parent = CancelScope::new();
        let child = parent.child();
        let sibling = parent.token();
        let under_child = child.token();
        child.cancel();
        assert!(under_child.is_cancelled(), "the child cancels its own subtree");
        assert!(!parent.is_cancelled(), "the parent is not cancelled");
        assert!(!sibling.is_cancelled(), "the parent's other leaf is untouched");
        assert!(!parent.token().is_cancelled(), "the parent still mints live leaves");
    }

    #[test]
    fn clones_of_a_scope_share_one_tree() {
        let a = CancelScope::new();
        let b = a.clone();
        // a leaf minted through one handle is a leaf of the same tree
        let t = b.token();
        a.cancel();
        assert!(t.is_cancelled(), "the original's cancel reaches the clone's leaf");
        assert!(b.is_cancelled(), "the clone sees the cancellation");
        // and the other way round
        let c = CancelScope::new();
        let d = c.clone();
        d.cancel();
        assert!(c.is_cancelled(), "cancelling through a clone cancels the original");
    }

    #[test]
    fn a_scope_is_cancelled_only_after_its_own_cancel() {
        let scope = CancelScope::new();
        assert!(!scope.is_cancelled());
        let child = scope.child();
        child.cancel();
        assert!(!scope.is_cancelled(), "a cancelled child does not cancel the parent");
        scope.cancel();
        assert!(scope.is_cancelled());
    }

    #[test]
    fn a_child_scope_minted_under_a_cancelled_scope_is_born_cancelled() {
        // A nested scope is a node under its parent: minted under a dead
        // parent it comes up dead, exactly like a leaf does.
        let parent = CancelScope::new();
        parent.cancel();
        let child = parent.child();
        assert!(child.is_cancelled(), "a scope minted under a dead scope is not live");
        assert!(child.token().is_cancelled(), "and its leaves are born dead too");
    }

    #[test]
    fn a_panicking_worker_is_reported_as_a_worker_error() {
        // The blocking pool's join failed — the closure panicked on the
        // worker thread. That is neither an interruption nor the
        // caller's own error: it comes back as a worker error, named.
        // (The panic prints its own message to stderr; that noise is the
        // panic hook doing its job, not a failing test.)
        let t = CancelToken::new();
        let err = crate::test_util::block_on(blocking(&t, || -> crate::Result<()> {
            panic!("the worker fell over")
        }))
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("worker: "), "the failure is named as the worker's: {msg}");
        assert!(msg.contains("the worker fell over"), "and it says what happened: {msg}");
    }

    #[test]
    fn minting_and_cancelling_take_the_same_lock() {
        // The race the tree has to close: a leaf minted while a cancel is
        // landing must not slip through. Both the mint and the cancel take
        // the children lock, so a mint that gets in after the cancel is
        // born cancelled, and one that got in before is cancelled by the
        // walk. A `check the flag; … then push` that let go of the lock
        // in between would let the leaf through, and this is what catches
        // it.
        let scope = CancelScope::new();
        // Hold the lock: both threads below block on it.
        let held = scope.inner.children.lock().unwrap();
        let minted: Arc<Mutex<Option<CancelToken>>> = Arc::new(Mutex::new(None));
        let mint_done = Arc::new(AtomicBool::new(false));
        let cancel_done = Arc::new(AtomicBool::new(false));

        let s1 = scope.clone();
        let m1 = minted.clone();
        let d1 = mint_done.clone();
        let minter = std::thread::spawn(move || {
            let t = s1.token();
            *m1.lock().unwrap() = Some(t);
            d1.store(true, Ordering::Release);
        });

        let s2 = scope.clone();
        let d2 = cancel_done.clone();
        let canceller = std::thread::spawn(move || {
            s2.cancel();
            d2.store(true, Ordering::Release);
        });

        // Let both threads reach the lock, then check that neither got
        // through: the lock is held, so neither can.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!mint_done.load(Ordering::Acquire), "the mint is waiting for the lock");
        assert!(!cancel_done.load(Ordering::Acquire), "the cancel is waiting for the lock");
        assert!(!scope.is_cancelled(), "the scope is not cancelled yet");

        drop(held);   // the lock is free: the two threads race for it now
        minter.join().unwrap();
        canceller.join().unwrap();

        // Whoever won the lock, the leaf ends up cancelled — born
        // cancelled if the cancel won, cancelled by the walk if the mint
        // did.
        let token = minted.lock().unwrap().clone().expect("the mint finished");
        assert!(token.is_cancelled(), "a leaf minted across a cancel is never left live");
        assert!(scope.is_cancelled());
    }

    #[test]
    fn every_token_minted_under_a_scope_is_cancelled_once_the_scope_is() {
        // The same rule at scale: however the mint and the cancel
        // interleave, no leaf minted under the scope is left live once the
        // scope is cancelled.
        let scope = CancelScope::new();
        let s = scope.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(5));
            s.cancel();
        });
        let mut tokens = Vec::new();
        for _ in 0..1_000 {
            tokens.push(scope.token());
        }
        canceller.join().unwrap();
        assert!(scope.is_cancelled());
        let live = tokens.iter().filter(|t| !t.is_cancelled()).count();
        assert_eq!(live, 0, "no leaf minted under a cancelled scope is left live");
    }
}
