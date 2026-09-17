//! The stdio transport: the server is a child process, and the protocol is what
//! the two of them write to each other's pipes.
//!
//! A message is one line of JSON — the specification forbids a newline inside
//! one — so the framing is the line, and a reader task is what turns lines into
//! answers for whoever is waiting. That task is the only reader: two of them
//! would take turns stealing each other's messages, so the waiting side hands
//! its request to the map the reader settles, and never touches the pipe.
//!
//! What comes back is not always an answer. A server may ask something of its
//! own (and a client that says nothing leaves it waiting), and may send a
//! notification, which expects nothing at all. Both are read here, and both are
//! why the reader is a task rather than a loop inside the request: a
//! notification that arrives while nobody is waiting has to be read too, or the
//! pipe fills and the server blocks on a write nobody will ever take.

use std::collections::HashMap;
use std::process::Stdio as ProcessStdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, oneshot};

use super::config::ServerConfig;
use super::health::Health;
use super::wire;

/// How long a server is given to leave on its own once its input is closed,
/// before it is killed. A server that reads its input — every one that speaks
/// this transport does — is gone well inside this; the grace is for the ones
/// that were in the middle of something when the session ended.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// The server's own input, behind the `None` that closing it is.
///
/// A clone of this is what the reader answers server requests with, and the
/// close that ends a session has to be visible to that clone rather than only
/// to the handle the caller holds: taking the writer out of here is what drops
/// it, and dropping it is what closes the pipe.
#[derive(Clone)]
pub struct Sink(Arc<Mutex<Option<Box<dyn AsyncWrite + Send + Unpin>>>>);

impl Sink {
    fn new(writer: impl AsyncWrite + Send + Unpin + 'static) -> Self {
        Self(Arc::new(Mutex::new(Some(Box::new(writer)))))
    }

    /// Write one message: compact JSON, one line, flushed. A newline inside a
    /// message would be read as the start of the next one, which is why the
    /// spec forbids it and why the serialization here is the compact one.
    async fn send(&self, message: &Value) -> Result<(), String> {
        let mut line =
            serde_json::to_string(message).map_err(|e| format!("cannot be written: {e}"))?;
        line.push('\n');
        let mut guard = self.0.lock().await;
        let Some(writer) = guard.as_mut() else {
            return Err("the server's input is closed".into());
        };
        writer
            .write_all(line.as_bytes())
            .await
            .map_err(|e| format!("writing to the server failed: {e}"))?;
        writer
            .flush()
            .await
            .map_err(|e| format!("writing to the server failed: {e}"))
    }

    /// Close the server's input. What the protocol says a client does to end a
    /// session, and what a server that reads its input reads as the end.
    async fn close(&self) {
        let _ = self.0.lock().await.take();
    }
}

/// What a waiting caller is handed: the server's answer, or the sentence that
/// says why there will not be one — a refusal, a timeout, or a connection that
/// ended under it. Both are text, because both are something to tell the model.
type Waiter = oneshot::Sender<Result<Value, String>>;

/// The requests this client is waiting on, by the id they were sent with.
///
/// The id is the client's to pick and the server's to echo, which is the whole
/// of how an answer finds its request in a pipe that carries one line at a time.
#[derive(Clone, Default)]
pub struct Pending(Arc<Mutex<HashMap<u64, Waiter>>>);

impl Pending {
    fn new() -> Self {
        Self::default()
    }

    /// Start waiting on `id`. The receiver is the caller's: it either arrives
    /// with the answer, or is dropped by [`Pending::fail_all`] when the
    /// connection ends, and the caller reads why from the health.
    async fn wait(&self, id: u64) -> oneshot::Receiver<Result<Value, String>> {
        let (tx, rx) = oneshot::channel();
        self.0.lock().await.insert(id, tx);
        rx
    }

    /// Hand `id`'s answer to whoever is waiting for it. An answer nobody is
    /// waiting for — a request that timed out, was cancelled, or came back
    /// with an id this client never sent (a string in a protocol that allows
    /// both, for instance) — is dropped: the request is over, and its answer
    /// cannot change that. A `None` id is the same situation with a step less
    /// in front of it: classify could not read a number out of it, and the
    /// map has no number to look it up by.
    async fn settle(&self, id: Option<u64>, outcome: wire::Outcome<'_>) {
        let Some(id) = id else { return };
        let Some(waiting) = self.0.lock().await.remove(&id) else {
            return;
        };
        let result = match outcome {
            wire::Outcome::Ok(value) => Ok(value.clone()),
            wire::Outcome::Failed { .. } => Err(outcome.reason()),
        };
        let _ = waiting.send(result);
    }

    /// Stop waiting on `id` without an answer: the caller has given up on it.
    async fn forget(&self, id: u64) {
        self.0.lock().await.remove(&id);
    }

    /// Tell everyone still waiting that the connection is over, and why.
    async fn fail_all(&self, why: &str) {
        for (_, waiting) in self.0.lock().await.drain() {
            let _ = waiting.send(Err(why.to_string()));
        }
    }
}

/// One server started as a child process, and the pipes to it.
pub struct Stdio {
    sink: Sink,
    pending: Pending,
    health: Arc<Health>,
    child: Mutex<Child>,
    /// The last thing the server wrote where it should not have: its standard
    /// output is the protocol and nothing else, and a server that prints a
    /// banner there has a banner to answer for.
    noise: Arc<std::sync::Mutex<Option<String>>>,
    /// The last line the server logged on its standard error, which is where
    /// the specification says a server may say anything at all. It is kept
    /// because a server that fails to start usually says why there and nowhere
    /// else.
    log: Arc<std::sync::Mutex<Option<String>>>,
    next_id: AtomicU64,
}

impl Stdio {
    /// Start the server the entry names and wire up its pipes.
    ///
    /// The child inherits this process's environment (that is how `npx` and
    /// `uvx` are found) with the entry's own variables added to it.
    pub fn spawn(config: &ServerConfig) -> Result<Self, String> {
        let command = config.command.clone().unwrap_or_default();
        let mut child = Command::new(&command);
        child
            .args(&config.args)
            .stdin(ProcessStdio::piped())
            .stdout(ProcessStdio::piped())
            .stderr(ProcessStdio::piped())
            // The backstop for a session that ends without closing this
            // connection: a server must not outlive the client that started it.
            .kill_on_drop(true);
        for (key, value) in &config.env {
            child.env(key, value);
        }
        let mut child = child
            .spawn()
            .map_err(|e| format!("failed to start {command:?}: {e}"))?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");

        let sink = Sink::new(stdin);
        let pending = Pending::new();
        let health = Health::new();
        let noise = Arc::new(std::sync::Mutex::new(None));
        let log = Arc::new(std::sync::Mutex::new(None));
        tokio::spawn(pump(
            BufReader::new(stdout),
            sink.clone(),
            pending.clone(),
            Arc::clone(&health),
            Arc::clone(&noise),
        ));
        tokio::spawn(log_lines(BufReader::new(stderr), Arc::clone(&log)));
        Ok(Self {
            sink,
            pending,
            health,
            child: Mutex::new(child),
            noise,
            log,
            next_id: AtomicU64::new(1),
        })
    }

    /// One request, and the answer to it.
    pub async fn ask(
        &self,
        method: &str,
        params: Value,
        budget: Duration,
    ) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        if !self.health.alive() {
            return Err(self.health.why());
        }
        let waiting = self.pending.wait(id).await;
        if let Err(e) = self.sink.send(&wire::request(id, method, params)).await {
            self.health.dies(&e);
            self.pending.forget(id).await;
            return Err(e);
        }
        match tokio::time::timeout(budget, waiting).await {
            Ok(Ok(Ok(value))) => Ok(value),
            // The server refused the request: its own words are the answer,
            // and they are not a broken connection.
            Ok(Ok(Err(refused))) => Err(refused),
            // The sender is gone, so the reader gave up on this connection and
            // wrote down why.
            Ok(Err(_)) => Err(self.health.why()),
            Err(_) => {
                self.pending.forget(id).await;
                Err(format!(
                    "no answer to {method} within {}s{}",
                    budget.as_secs(),
                    self.trailing()
                ))
            }
        }
    }

    /// One notification: no id, and nothing to wait for.
    pub async fn tell(&self, method: &str, params: Value) -> Result<(), String> {
        if !self.health.alive() {
            return Err(self.health.why());
        }
        self.sink.send(&wire::notification(method, params)).await
    }

    /// End the connection the way the specification says a client ends one:
    /// close the server's input, give it a moment to leave on its own, and kill
    /// it if it does not.
    pub async fn shutdown(&self) {
        self.sink.close().await;
        let mut child = self.child.lock().await;
        if tokio::time::timeout(SHUTDOWN_GRACE, child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
    }

    /// What the server was last seen saying, for a failure that has nothing
    /// else to report: its own last log line, or the text it wrote where a
    /// message belongs.
    fn trailing(&self) -> String {
        let log = self.log.lock().expect("not held across an await").clone();
        let noise = self.noise.lock().expect("not held across an await").clone();
        match (log, noise) {
            (None, None) => String::new(),
            (Some(line), None) => format!(" (the server last logged: {line})"),
            (None, Some(line)) => format!(" (the server last wrote: {line})"),
            (Some(log), Some(noise)) => {
                format!(" (the server last logged: {log}; it last wrote: {noise})")
            }
        }
    }
}

/// Read the server's output until it ends, settling answers and serving the
/// requests that come the other way.
///
/// Generic over its reader so that a test can drive it with a pipe of its own:
/// what this function has to get right is the protocol, and none of that needs
/// a process on the other end.
async fn pump<R>(
    reader: R,
    sink: Sink,
    pending: Pending,
    health: Arc<Health>,
    noise: Arc<std::sync::Mutex<Option<String>>>,
) where
    R: AsyncBufRead + Unpin + Send + 'static,
{
    let mut lines = reader.lines();
    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => {
                health.dies("the server closed its output, so it is no longer there to answer");
                break;
            }
            Err(e) => {
                health.dies(format!("reading the server's output failed: {e}"));
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            // The specification forbids anything but a message on this pipe,
            // and a server that breaks that rule is still one whose output has
            // to be read: the line is kept as the last thing it said, and the
            // session carries on.
            *noise.lock().expect("not held across an await") = Some(line);
            continue;
        };
        match wire::classify(&message) {
            wire::Incoming::Answer { id, outcome } => pending.settle(id, outcome).await,
            wire::Incoming::Ask { id, method } => {
                let answer = wire::serve(id, method);
                if let Err(e) = sink.send(&answer).await {
                    // The server asked and cannot be answered: the connection
                    // is over, and the caller waiting on it finds out here.
                    health.dies(e);
                    break;
                }
            }
            // A notification is a server saying something happened; nothing in
            // this client acts on one, and a message that is not a message is
            // read past for the same reason an unknown event is.
            wire::Incoming::Notice | wire::Incoming::Noise => {}
        }
    }
    pending.fail_all(&health.why()).await;
}

/// Keep the server's last log line, and nothing else: standard error is the
/// server's to talk on, and a session that echoed it would be a session with
/// two voices.
async fn log_lines<R>(reader: R, log: Arc<std::sync::Mutex<Option<String>>>)
where
    R: AsyncBufRead + Unpin + Send + 'static,
{
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if !line.trim().is_empty() {
            *log.lock().expect("not held across an await") = Some(line);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, BufReader, DuplexStream, duplex};

    /// A pump reading from a pipe the test writes into, and the pipe its own
    /// writes come out of: a server on the other end of two pipes, made of
    /// memory rather than of a process.
    struct Peer {
        /// What the server writes: the client reads it.
        to_client: DuplexStream,
        /// What the client writes: the server reads it.
        from_client: DuplexStream,
        pending: Pending,
        health: Arc<Health>,
        noise: Arc<std::sync::Mutex<Option<String>>>,
    }

    impl Peer {
        fn new() -> Self {
            let (to_client, client_reads) = duplex(8 * 1024);
            let (client_writes, from_client) = duplex(8 * 1024);
            let pending = Pending::new();
            let health = Health::new();
            let noise = Arc::new(std::sync::Mutex::new(None));
            tokio::spawn(pump(
                BufReader::new(client_reads),
                Sink::new(client_writes),
                pending.clone(),
                Arc::clone(&health),
                Arc::clone(&noise),
            ));
            Self {
                to_client,
                from_client,
                pending,
                health,
                noise,
            }
        }

        /// The server sends `message`, as a line of JSON.
        async fn says(&mut self, message: Value) {
            self.to_client
                .write_all(format!("{message}\n").as_bytes())
                .await
                .unwrap();
            self.to_client.flush().await.unwrap();
        }

        /// The server says something that is not a message.
        async fn splutters(&mut self, text: &str) {
            self.to_client
                .write_all(format!("{text}\n").as_bytes())
                .await
                .unwrap();
        }

        /// What the client wrote back, as one line of JSON.
        async fn hears(&mut self) -> Value {
            let mut buf = [0u8; 1];
            let mut line = Vec::new();
            loop {
                let n = self.from_client.read(&mut buf).await.unwrap();
                assert_ne!(n, 0, "the client closed its side");
                if buf[0] == b'\n' {
                    break;
                }
                line.push(buf[0]);
            }
            serde_json::from_slice(&line).unwrap()
        }

        /// The pipe from the server ends: the server is gone.
        async fn closes(&mut self) {
            let _ = self.to_client.shutdown().await;
        }
    }

    #[tokio::test]
    async fn an_answer_finds_the_request_that_waited_for_it() {
        let mut peer = Peer::new();
        let waiting = peer.pending.wait(4).await;
        // The second request is answered first: the id is what pairs them, not
        // the order they were sent in.
        let other = peer.pending.wait(5).await;
        peer.says(json!({"jsonrpc": "2.0", "id": 5, "result": {"which": 5}}))
            .await;
        peer.says(json!({"jsonrpc": "2.0", "id": 4, "result": {"which": 4}}))
            .await;
        assert_eq!(waiting.await.unwrap().unwrap()["which"], json!(4));
        assert_eq!(other.await.unwrap().unwrap()["which"], json!(5));
    }

    #[tokio::test]
    async fn a_refusal_comes_back_as_the_servers_own_words() {
        let mut peer = Peer::new();
        let waiting = peer.pending.wait(1).await;
        peer.says(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "no such tool"}
        }))
        .await;
        let refused = waiting.await.unwrap().unwrap_err();
        assert!(refused.contains("no such tool"), "{refused}");
    }

    #[tokio::test]
    async fn an_answer_nobody_waits_for_is_dropped() {
        let mut peer = Peer::new();
        // A request that timed out and was forgotten: its answer arrives late,
        // and there is nothing to do with it.
        peer.says(json!({"jsonrpc": "2.0", "id": 9, "result": {}}))
            .await;
        assert!(peer.pending.0.lock().await.is_empty());
        peer.says(json!({"jsonrpc": "2.0", "method": "notifications/message"}))
            .await;
        // Still reading: the late answer and the notification were read past
        // rather than taken for the end of the stream.
        let waiting = peer.pending.wait(1).await;
        peer.says(json!({"jsonrpc": "2.0", "id": 1, "result": {"still": "here"}}))
            .await;
        assert_eq!(waiting.await.unwrap().unwrap()["still"], json!("here"));
    }

    #[tokio::test]
    async fn the_server_asking_is_answered_on_the_same_pipe() {
        let mut peer = Peer::new();
        peer.says(json!({"jsonrpc": "2.0", "id": "s1", "method": "ping"}))
            .await;
        let answer = peer.hears().await;
        assert_eq!(answer["id"], json!("s1"));
        assert_eq!(answer["result"], json!({}));
        // The id is echoed as it arrived: it is the server's, not ours.
        peer.says(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "sampling/createMessage"
        }))
        .await;
        let refused = peer.hears().await;
        assert_eq!(refused["error"]["code"], json!(wire::METHOD_NOT_FOUND));
    }

    #[tokio::test]
    async fn what_is_not_a_message_is_kept_as_the_last_thing_it_said() {
        let mut peer = Peer::new();
        peer.splutters("Some server banner nobody asked for").await;
        peer.splutters("   ").await;
        // The pump reads lines in order, so an answer that arrives after the
        // noise is proof that the noise was read too.
        let waiting = peer.pending.wait(1).await;
        peer.says(json!({"jsonrpc": "2.0", "id": 1, "result": {}}))
            .await;
        waiting.await.unwrap().unwrap();
        assert_eq!(
            peer.noise.lock().unwrap().as_deref(),
            Some("Some server banner nobody asked for")
        );
        assert!(peer.health.alive());
    }

    #[tokio::test]
    async fn the_output_ending_is_the_connection_ending() {
        let mut peer = Peer::new();
        let waiting = peer.pending.wait(1).await;
        peer.closes().await;
        // The request that was waiting is told, and that is the moment the
        // connection is known to be over: the pump writes it down before it
        // fails what it was holding.
        let told = waiting.await.unwrap().unwrap_err();
        assert!(told.contains("closed its output"), "{told}");
        assert!(!peer.health.alive());
        assert!(peer.health.why().contains("closed its output"));
    }

    #[tokio::test]
    async fn a_reply_that_cannot_be_written_ends_the_connection() {
        let (mut to_client, client_reads) = duplex(8 * 1024);
        // The server is gone: nothing reads what the client writes, and the
        // reply it owes the ping goes nowhere.
        let (client_writes, nobody_reads) = duplex(8 * 1024);
        drop(nobody_reads);
        let health = Health::new();
        tokio::spawn(pump(
            BufReader::new(client_reads),
            Sink::new(client_writes),
            Pending::new(),
            Arc::clone(&health),
            Arc::new(std::sync::Mutex::new(None)),
        ));
        to_client
            .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")
            .await
            .unwrap();
        to_client.flush().await.unwrap();
        for _ in 0..200 {
            if !health.alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(!health.alive(), "a reply that cannot be written is the end");
        assert!(
            health.why().contains("writing to the server failed"),
            "{}",
            health.why()
        );
    }

    #[tokio::test]
    async fn every_waiting_request_is_told_when_the_connection_ends() {
        let pending = Pending::new();
        let first = pending.wait(1).await;
        let second = pending.wait(2).await;
        pending.fail_all("the server is gone").await;
        assert_eq!(first.await.unwrap().unwrap_err(), "the server is gone");
        assert_eq!(second.await.unwrap().unwrap_err(), "the server is gone");
        assert!(pending.0.lock().await.is_empty());
    }

    #[tokio::test]
    async fn a_write_after_the_sink_is_closed_is_refused() {
        let (writer, _reader) = duplex(8 * 1024);
        let sink = Sink::new(writer);
        sink.send(&json!({"jsonrpc": "2.0"})).await.unwrap();
        sink.close().await;
        let refused = sink.send(&json!({"jsonrpc": "2.0"})).await.unwrap_err();
        assert!(refused.contains("input is closed"), "{refused}");
    }
}
