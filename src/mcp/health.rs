//! What a connection says about itself once it stops working.
//!
//! A transport's reader is the only thing that learns a connection is over —
//! the pipe ended, the process exited, the socket closed — and the caller that
//! needs to know is on the other side of a request that will never be answered.
//! This is where the two meet: the reader writes the reason down once, and every
//! caller that finds the connection dead reads it back, so that a failure never
//! says less than what happened.
//!
//! The first reason wins. A reader that has stopped because the process exited
//! then fails to write a reply and would have a second thing to say about it;
//! the second one is a consequence of the first, and reporting it would bury
//! the cause.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether the other end can still answer, and why it cannot when it cannot.
#[derive(Debug, Default)]
pub struct Health {
    alive: AtomicBool,
    detail: std::sync::Mutex<Option<String>>,
}

impl Health {
    /// A connection that has just opened: alive, with nothing to say.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            alive: AtomicBool::new(true),
            detail: std::sync::Mutex::new(None),
        })
    }

    /// Whether the connection can be asked anything.
    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// The connection is over, for `why`. Called by the reader that found it
    /// out; the first reason is the one that stands.
    pub fn dies(&self, why: impl Into<String>) {
        self.alive.store(false, Ordering::SeqCst);
        let mut detail = self.detail.lock().expect("not held across an await");
        if detail.is_none() {
            *detail = Some(why.into());
        }
    }

    /// The reason the connection is over, or `None` while it is not.
    pub fn reason(&self) -> Option<String> {
        self.detail
            .lock()
            .expect("not held across an await")
            .clone()
    }

    /// The reason, as a sentence that can stand alone: what a failure reports
    /// when it has nothing better to say.
    pub fn why(&self) -> String {
        self.reason()
            .unwrap_or_else(|| "the connection is closed".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_connection_is_alive_and_says_nothing() {
        let health = Health::new();
        assert!(health.alive());
        assert_eq!(health.reason(), None);
        assert_eq!(health.why(), "the connection is closed");
    }

    #[test]
    fn the_first_reason_is_the_one_that_stands() {
        let health = Health::new();
        health.dies("the server exited with status 1");
        assert!(!health.alive());
        assert_eq!(health.why(), "the server exited with status 1");
        // The write that failed because of it is not a second reason.
        health.dies("writing to the server failed: broken pipe");
        assert_eq!(health.why(), "the server exited with status 1");
    }
}
