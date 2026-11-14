//! The background process, and the screen's connection to it.
//!
//! The harness itself (the conversation, the agent loop, the scheduler, the
//! timeline file) runs in one long-lived process, the daemon. The TUI is a
//! client: it attaches over a local socket, is sent what has happened so
//! far, then every event as it happens, and sends back what the user does.
//! Closing the screen does not stop the work, so scheduled jobs start with
//! no screen open.
//!
//! The protocol is one JSON object per line, in both directions (`ToDaemon`,
//! `ToClient`). The socket is a Unix socket only the user can open; nothing
//! listens on the network.
//!
//! Everything about the process's life is a fixed rule, decided here:
//! - one daemon per data folder; a second start finds the first and exits;
//! - it ends by itself once no screen is attached, nothing is running or
//!   waiting to be remembered, and no job is scheduled (`LINGER`);
//! - `stop` is refused while a task is running unless forced.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::demo::Command;
use crate::event::{EventKind, MemoryNode, Severity, Tagged};
use crate::live::Live;
use crate::timeline::{Log, Recorded};

/// How often the daemon looks for new events when nothing is sent to it.
const TICK: Duration = Duration::from_millis(80);
/// How long the daemon stays once there is nothing left for it to do.
const LINGER: Duration = Duration::from_secs(20);
/// Events from this run kept for a screen that attaches later. Older ones
/// are still in the timeline file.
const MAX_SINCE: usize = 600;
/// The longest a socket address may safely be on Linux and macOS.
const MAX_SOCKET_PATH: usize = 100;
/// How long a slow screen may hold up the daemon before it is dropped.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// What a client sends.
#[derive(Debug, Serialize, Deserialize)]
pub enum ToDaemon {
    /// Become a screen: be sent `Hello`, then every event.
    Attach,
    /// Something the user did on an attached screen.
    Command(Command),
    /// One question, one `Status` back.
    Status,
    /// List the scheduled jobs, or cancel one. One `Said` back.
    Schedule { cancel: Option<u32> },
    /// End the daemon. One `Said` back.
    Stop { force: bool },
}

/// What the daemon sends.
#[derive(Debug, Serialize, Deserialize)]
pub enum ToClient {
    Hello(Box<Hello>),
    Event(Recorded),
    /// Streaming speed in tokens a second, for the throughput graph.
    Throughput(u64),
    /// The reply a task is writing right now, as far as it has come. Empty
    /// when it is no longer writing. Never part of the timeline.
    Draft {
        task: u32,
        text: String,
    },
    Status(Status),
    Said {
        ok: bool,
        text: String,
    },
    /// The daemon is ending; why.
    Closing(String),
}

/// The state of the daemon, in a few facts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Status {
    /// Which build of the program it is (`build_id`).
    pub build: String,
    pub pid: u32,
    /// The folder it works in.
    pub folder: String,
    /// The core model as it was named (`backend[:model]`), and as shown.
    pub spec: String,
    pub core: String,
    /// The core model's context window, in tokens.
    pub window: u32,
    /// Whether a task is running.
    pub busy: bool,
    /// Screens attached.
    pub screens: usize,
    /// Jobs scheduled.
    pub jobs: usize,
    /// How the Telegram bot stands, in a line; empty when none is
    /// connected. While it is unpaired this carries the pairing code.
    #[serde(default)]
    pub telegram: String,
}

/// What a screen needs to draw itself when it attaches.
#[derive(Debug, Serialize, Deserialize)]
pub struct Hello {
    pub status: Status,
    /// Events from before this daemon started: history, nothing in motion.
    pub before: Vec<Recorded>,
    /// Events since it started, the most recent `MAX_SINCE`: what is true
    /// now, including a call still waiting for approval.
    pub since: Vec<Recorded>,
    /// What is remembered, for the memory view.
    pub memory: Vec<MemoryNode>,
}

/// What does not change while a daemon runs.
#[derive(Clone, Debug)]
pub struct Identity {
    pub folder: String,
    pub spec: String,
}

/// Which build of the program this process is: its version plus the size
/// and time of its file. A rebuilt or reinstalled program has another.
pub fn build_id() -> String {
    let file = std::env::current_exe()
        .ok()
        .and_then(|exe| std::fs::metadata(exe).ok());
    let (len, changed) = file.map_or((0, 0), |m| {
        let changed = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        (m.len(), changed)
    });
    format!("{}-{len}-{changed}", env!("CARGO_PKG_VERSION"))
}

/// Where the daemon listens: `daemon.sock` in the data folder, or, when
/// that path is too long for a socket address, a file named after the data
/// folder in the runtime (else temporary) folder.
pub fn socket_path() -> Option<PathBuf> {
    let dir = crate::learn::default_path()?.parent()?.to_path_buf();
    let direct = dir.join("daemon.sock");
    if direct.as_os_str().len() <= MAX_SOCKET_PATH {
        return Some(direct);
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in dir.as_os_str().as_encoded_bytes() {
        hash = (hash ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
    }
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir);
    Some(base.join(format!("conscious-{hash:016x}.sock")))
}

/// Where the daemon's own notes and errors go.
pub fn log_path() -> Option<PathBuf> {
    Some(crate::learn::default_path()?.parent()?.join("daemon.log"))
}

/// Takes the socket for a daemon about to start. Fails if one is already
/// listening there; a socket file left by a daemon that died is replaced.
pub fn bind(path: &Path) -> Result<UnixListener, String> {
    use std::os::unix::fs::PermissionsExt;
    if UnixStream::connect(path).is_ok() {
        return Err("conscious is already running in the background".into());
    }
    let _ = std::fs::remove_file(path);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let listener = UnixListener::bind(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // Only the user may talk to it.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(listener)
}

fn write_line<T: Serialize>(stream: &mut UnixStream, message: &T) -> std::io::Result<()> {
    let mut line = serde_json::to_string(message)?;
    line.push('\n');
    stream.write_all(line.as_bytes())
}

enum In {
    Joined(UnixStream),
    Said(ToDaemon),
    Gone,
}

/// Accepts connections and reads what each one sends, handing everything
/// to the daemon's one loop, which alone touches the harness.
fn listen(listener: UnixListener, tx: Sender<(u64, In)>) {
    thread::spawn(move || {
        for (id, stream) in (1u64..).zip(listener.incoming().flatten()) {
            let Ok(reader) = stream.try_clone() else {
                continue;
            };
            let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
            if tx.send((id, In::Joined(stream))).is_err() {
                return;
            }
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(reader).lines().map_while(Result::ok) {
                    // A line that cannot be read is from another version
                    // of the program; it is ignored, not fatal.
                    if let Ok(message) = serde_json::from_str::<ToDaemon>(&line)
                        && tx.send((id, In::Said(message))).is_err()
                    {
                        return;
                    }
                }
                let _ = tx.send((id, In::Gone));
            });
        }
    });
}

struct Server {
    live: Live,
    log: Log,
    identity: Identity,
    build: String,
    before: Vec<Recorded>,
    since: VecDeque<Recorded>,
    /// Every open connection, and which of them are screens.
    open: HashMap<u64, UnixStream>,
    screens: HashSet<u64>,
    sent_throughput: u64,
    /// Since when there has been nothing to stay for.
    quiet: Option<Instant>,
    linger: Duration,
}

impl Drop for Server {
    /// Closes every connection for good when the daemon ends. The threads
    /// reading from them hold their own handles, which would otherwise keep
    /// a client waiting on a daemon that is no longer serving (as when it
    /// stays alive a while to start its successor).
    fn drop(&mut self) {
        for stream in self.open.values() {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

impl Server {
    fn status(&self) -> Status {
        let core = self.live.router().core();
        Status {
            build: self.build.clone(),
            pid: std::process::id(),
            folder: self.identity.folder.clone(),
            spec: self.identity.spec.clone(),
            core: core.label(),
            window: core.context_window(),
            busy: self.live.busy(),
            screens: self.screens.len(),
            jobs: self.jobs(),
            telegram: self.live.telegram_state(),
        }
    }

    fn jobs(&self) -> usize {
        self.live
            .schedule()
            .map_or(0, |s| s.lock().unwrap().jobs().len())
    }

    /// Sends to one connection; one that cannot be written to is dropped.
    fn send(&mut self, id: u64, message: &ToClient) {
        let failed = match self.open.get_mut(&id) {
            Some(stream) => write_line(stream, message).is_err(),
            None => false,
        };
        if failed {
            self.open.remove(&id);
            self.screens.remove(&id);
        }
    }

    fn broadcast(&mut self, message: &ToClient) {
        let screens: Vec<u64> = self.screens.iter().copied().collect();
        for id in screens {
            self.send(id, message);
        }
    }

    /// Records an event and shows it on every screen.
    fn publish(&mut self, event: Tagged) {
        self.log.append(event.task, &event.kind);
        let recorded = Recorded {
            wall: crate::learn::now(),
            task: event.task,
            kind: event.kind,
        };
        if self.since.len() == MAX_SINCE {
            self.since.pop_front();
        }
        self.since.push_back(recorded.clone());
        self.broadcast(&ToClient::Event(recorded));
    }

    /// Handles one message. Returns false when the daemon is to end.
    fn receive(&mut self, id: u64, message: In) -> bool {
        match message {
            In::Joined(stream) => {
                self.open.insert(id, stream);
            }
            In::Gone => {
                self.open.remove(&id);
                self.screens.remove(&id);
            }
            In::Said(ToDaemon::Attach) => {
                let hello = Hello {
                    status: self.status(),
                    before: self.before.clone(),
                    since: self.since.iter().cloned().collect(),
                    memory: self.live.memory_nodes(),
                };
                // Hello and joining the screens happen in one step of this
                // loop, so no event falls between them.
                self.screens.insert(id);
                self.send(id, &ToClient::Hello(Box::new(hello)));
            }
            In::Said(ToDaemon::Command(cmd)) => {
                // Only an attached screen may act for the user.
                if !self.screens.contains(&id) {
                    return true;
                }
                if let Command::Restart = cmd {
                    self.live = self.live.restarted();
                    return true;
                }
                for event in self.live.handle(cmd) {
                    self.publish(event);
                }
            }
            In::Said(ToDaemon::Status) => {
                let status = self.status();
                self.send(id, &ToClient::Status(status));
            }
            In::Said(ToDaemon::Schedule { cancel }) => {
                let answer = match self.live.schedule() {
                    None => Err("the scheduler is switched off".to_string()),
                    Some(schedule) => {
                        let mut schedule = schedule.lock().unwrap();
                        match cancel {
                            Some(job) => schedule.cancel(job),
                            None => Ok(schedule.report(crate::learn::now())),
                        }
                    }
                };
                let (ok, text) = match answer {
                    Ok(text) => (true, text),
                    Err(text) => (false, text),
                };
                self.send(id, &ToClient::Said { ok, text });
            }
            In::Said(ToDaemon::Stop { force }) => {
                if self.live.busy() && !force {
                    self.send(
                        id,
                        &ToClient::Said {
                            ok: false,
                            text: "a task is running; `conscious stop --force` ends it anyway"
                                .into(),
                        },
                    );
                    return true;
                }
                self.send(
                    id,
                    &ToClient::Said {
                        ok: true,
                        text: "conscious has stopped".into(),
                    },
                );
                self.broadcast(&ToClient::Closing("it was stopped".into()));
                return false;
            }
        }
        true
    }

    /// Whether there is still something to stay for. With nothing for
    /// `linger`, the daemon ends.
    fn wanted(&mut self) -> bool {
        if !self.screens.is_empty()
            || !self.live.settled()
            || self.jobs() > 0
            || self.live.connected()
        {
            self.quiet = None;
            return true;
        }
        self.quiet.get_or_insert_with(Instant::now).elapsed() < self.linger
    }
}

/// Runs the harness until it is stopped or has nothing left to do. `before`
/// is the timeline as read from the file when it started.
pub fn serve(
    live: Live,
    log: Log,
    before: Vec<Recorded>,
    listener: UnixListener,
    identity: Identity,
) -> String {
    serve_for(live, log, before, listener, identity, LINGER)
}

fn serve_for(
    live: Live,
    log: Log,
    before: Vec<Recorded>,
    listener: UnixListener,
    identity: Identity,
    linger: Duration,
) -> String {
    let (tx, inbox) = mpsc::channel();
    listen(listener, tx);
    let mut server = Server {
        live,
        log,
        identity,
        build: build_id(),
        before,
        since: VecDeque::new(),
        open: HashMap::new(),
        screens: HashSet::new(),
        sent_throughput: 0,
        quiet: None,
        linger,
    };
    loop {
        let mut next = match inbox.recv_timeout(TICK) {
            Ok(message) => Some(message),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => return "its listener ended".into(),
        };
        while let Some((id, message)) = next {
            if !server.receive(id, message) {
                return "it was stopped".into();
            }
            next = inbox.try_recv().ok();
        }
        for event in server.live.poll() {
            server.publish(event);
        }
        for (task, text) in server.live.drafts() {
            server.broadcast(&ToClient::Draft { task, text });
        }
        let speed = server.live.throughput();
        if speed != server.sent_throughput {
            server.sent_throughput = speed;
            server.broadcast(&ToClient::Throughput(speed));
        }
        // A self-update was adopted and nothing is running: this process
        // ends so that the new build can take its place. Screens are told,
        // and reconnect by themselves.
        if let Some(why) = server.live.take_restart() {
            server.broadcast(&ToClient::Closing(format!("restarting: {why}")));
            return format!("restart: {why}");
        }
        if !server.wanted() {
            return "no screen was attached, nothing was running and no job was scheduled".into();
        }
    }
}

/// Asks the daemon one thing and returns its one answer.
pub fn ask(path: &Path, message: &ToDaemon, wait: Duration) -> Result<ToClient, String> {
    let mut stream = UnixStream::connect(path).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(wait))
        .map_err(|e| e.to_string())?;
    write_line(&mut stream, message).map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new(stream)
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    if line.is_empty() {
        return Err("the background process closed the connection".into());
    }
    serde_json::from_str(&line).map_err(|e| format!("unreadable answer: {e}"))
}

/// The daemon's status, or why it could not be had (it is not running).
pub fn status(path: &Path, wait: Duration) -> Result<Status, String> {
    match ask(path, &ToDaemon::Status, wait)? {
        ToClient::Status(status) => Ok(status),
        other => Err(format!("unexpected answer: {other:?}")),
    }
}

/// Starts the daemon as a process of its own, detached from this terminal:
/// in its own process group, so Ctrl-C and closing the terminal do not
/// reach it, with its output going to `log`.
pub fn spawn(args: &[String], log: &Path) -> std::io::Result<std::process::Child> {
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let out = std::fs::File::create(log)?;
    let err = out.try_clone()?;
    std::process::Command::new(std::env::current_exe()?)
        .arg("daemon")
        .args(args)
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .process_group(0)
        .spawn()
}

/// A screen's connection to the daemon. Has the same surface as the
/// in-process feeds: events out, commands in.
pub struct Remote {
    out: UnixStream,
    rx: Receiver<ToClient>,
    throughput: u64,
    /// Replies being written, as last sent, not yet handed to the screen.
    drafts: Vec<(u32, String)>,
    /// Whether the loss of the connection has been said.
    lost: bool,
    /// Where the daemon listens, to reconnect after it restarts.
    path: PathBuf,
    /// Since when the daemon has been restarting into a new build.
    restarting: Option<Instant>,
}

/// How long a screen waits for a daemon that said it is restarting.
const RECONNECT_FOR: Duration = Duration::from_secs(180);

impl Remote {
    /// Attaches as a screen. Returns the connection and what to draw first.
    pub fn attach(path: &Path) -> Result<(Self, Hello), String> {
        let mut out = UnixStream::connect(path).map_err(|e| e.to_string())?;
        write_line(&mut out, &ToDaemon::Attach).map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(out.try_clone().map_err(|e| e.to_string())?);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let hello = match serde_json::from_str::<ToClient>(&line) {
            Ok(ToClient::Hello(hello)) => *hello,
            Ok(other) => return Err(format!("unexpected answer: {other:?}")),
            Err(e) => {
                return Err(format!(
                    "the background process answered in a way this version cannot read ({e}); \
                     run `conscious stop --force` and start again"
                ));
            }
        };
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(message) = serde_json::from_str::<ToClient>(&line)
                    && tx.send(message).is_err()
                {
                    return;
                }
            }
        });
        Ok((
            Self {
                out,
                rx,
                throughput: 0,
                drafts: Vec::new(),
                lost: false,
                path: path.to_path_buf(),
                restarting: None,
            },
            hello,
        ))
    }

    fn lost(&mut self, why: &str) -> Option<Tagged> {
        if self.lost {
            return None;
        }
        self.lost = true;
        Some(Tagged::system(EventKind::Alert {
            severity: Severity::Error,
            text: format!(
                "The background process is gone ({why}), so nothing sent from this screen \
                 will be acted on. Quit (q) and run `conscious` again."
            ),
        }))
    }

    pub fn poll(&mut self) -> Vec<Tagged> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(ToClient::Event(recorded)) => out.push(Tagged {
                    task: recorded.task,
                    kind: recorded.kind,
                }),
                Ok(ToClient::Throughput(speed)) => self.throughput = speed,
                Ok(ToClient::Draft { task, text }) => {
                    self.drafts.retain(|(t, _)| *t != task);
                    self.drafts.push((task, text));
                }
                // A daemon that restarts into a new build comes back: the
                // screen waits for it instead of giving up.
                Ok(ToClient::Closing(why)) if why.starts_with("restarting") => {
                    self.restarting = Some(Instant::now());
                    out.push(Tagged::system(EventKind::Alert {
                        severity: Severity::Info,
                        text: format!(
                            "The harness is {why}. This screen reconnects by itself when the \
                             new build is up."
                        ),
                    }));
                }
                Ok(ToClient::Closing(why)) => out.extend(self.lost(&why)),
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.throughput = 0;
                    match self.restarting {
                        Some(since) if since.elapsed() < RECONNECT_FOR => {}
                        Some(_) => {
                            self.restarting = None;
                            out.extend(self.lost("it did not come back after restarting"));
                        }
                        None => out.extend(self.lost("the connection closed")),
                    }
                    break;
                }
            }
        }
        out
    }

    /// While the daemon is restarting, tries to attach to the new one.
    /// Returns what to draw once it is back; the connection is then the
    /// new one.
    pub fn reattach(&mut self) -> Option<Hello> {
        self.restarting?;
        let (fresh, hello) = Self::attach(&self.path).ok()?;
        *self = fresh;
        Some(hello)
    }

    /// Sends what the user did. What comes of it arrives through `poll`.
    pub fn handle(&mut self, cmd: Command) -> Vec<Tagged> {
        match write_line(&mut self.out, &ToDaemon::Command(cmd)) {
            Ok(()) => Vec::new(),
            Err(_) => self
                .lost("it could not be written to")
                .into_iter()
                .collect(),
        }
    }

    pub fn throughput(&mut self) -> u64 {
        self.throughput
    }

    /// The replies being written that have changed since last asked.
    pub fn drafts(&mut self) -> Vec<(u32, String)> {
        std::mem::take(&mut self.drafts)
    }
}

impl Drop for Remote {
    /// Closes the connection for good. The thread reading from it holds
    /// its own handle, which would otherwise keep this screen counted as
    /// attached until the whole program ended.
    fn drop(&mut self) {
        let _ = self.out.shutdown(std::net::Shutdown::Both);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::testing::Fixed;
    use crate::router::Router;
    use crate::schedule::{Missed, Schedule};
    use std::sync::{Arc, Mutex};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("csd-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Starts a daemon on its own thread; returns its socket and the
    /// thread, which ends with the reason the daemon gave for ending.
    fn start(
        name: &str,
        live: Live,
        before: Vec<Recorded>,
        linger: Duration,
    ) -> (PathBuf, thread::JoinHandle<String>) {
        let path = scratch(name).join("d.sock");
        let listener = bind(&path).unwrap();
        let identity = Identity {
            folder: "/work".into(),
            spec: "fixed".into(),
        };
        let handle = thread::spawn(move || {
            serve_for(live, Log::disabled(), before, listener, identity, linger)
        });
        (path, handle)
    }

    /// Starts a daemon on a socket path that was used before.
    fn start_at(path: &Path, live: Live) -> (PathBuf, thread::JoinHandle<String>) {
        let _ = std::fs::remove_file(path);
        let listener = bind(path).unwrap();
        let identity = Identity {
            folder: "/work".into(),
            spec: "fixed".into(),
        };
        let handle = thread::spawn(move || {
            serve_for(live, Log::disabled(), Vec::new(), listener, identity, LONG)
        });
        (path.to_path_buf(), handle)
    }

    fn live(answer: &'static str) -> Live {
        Live::new(Arc::new(Router::single(Arc::new(Fixed(answer)))))
    }

    /// Polls a screen until an event matches, or gives up.
    fn wait_for(remote: &mut Remote, found: impl Fn(&EventKind) -> bool) -> Vec<EventKind> {
        let mut seen: Vec<EventKind> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            seen.extend(remote.poll().into_iter().map(|e| e.kind));
            if seen.iter().any(&found) {
                return seen;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("not seen within 5 s; got {seen:?}");
    }

    const LONG: Duration = Duration::from_secs(60);
    const WAIT: Duration = Duration::from_secs(5);

    #[test]
    fn a_screen_attaches_sends_a_message_and_sees_the_reply() {
        let (path, daemon) = start("chat", live("hello there"), Vec::new(), LONG);
        let (mut screen, hello) = Remote::attach(&path).unwrap();
        assert_eq!(hello.status.core, "fixed");
        assert_eq!(hello.status.window, 8_000);
        assert_eq!(hello.status.folder, "/work");
        assert_eq!(hello.status.screens, 0, "counted before this screen joined");

        assert!(screen.handle(Command::UserMessage("hi".into())).is_empty());
        let seen = wait_for(
            &mut screen,
            |e| matches!(e, EventKind::Message { agent: Some(_), text } if text == "hello there"),
        );
        assert!(
            seen.iter()
                .any(|e| matches!(e, EventKind::Message { agent: None, text } if text == "hi"))
        );
        assert_eq!(status(&path, WAIT).unwrap().screens, 1);
        // A screen that closes is no longer counted.
        let (second, _) = Remote::attach(&path).unwrap();
        assert_eq!(status(&path, WAIT).unwrap().screens, 2);
        drop(second);
        let deadline = Instant::now() + WAIT;
        while status(&path, WAIT).unwrap().screens != 1 {
            assert!(
                Instant::now() < deadline,
                "the closed screen is still counted"
            );
            thread::sleep(Duration::from_millis(10));
        }

        let said = ask(&path, &ToDaemon::Stop { force: false }, WAIT).unwrap();
        assert!(matches!(said, ToClient::Said { ok: true, .. }));
        assert_eq!(daemon.join().unwrap(), "it was stopped");
        // The screen is told, once, that what it was attached to is gone.
        let after = wait_for(&mut screen, |e| {
            matches!(
                e,
                EventKind::Alert {
                    severity: Severity::Error,
                    ..
                }
            )
        });
        assert_eq!(after.len(), 1);
        assert!(screen.poll().is_empty());
    }

    #[test]
    fn work_goes_on_with_no_screen_and_a_later_screen_sees_what_happened() {
        let earlier = vec![Recorded {
            wall: 1_000,
            task: None,
            kind: EventKind::Message {
                agent: None,
                text: "from an earlier run".into(),
            },
        }];
        let (path, _daemon) = start("detach", live("done while away"), earlier, LONG);
        let (mut first, _) = Remote::attach(&path).unwrap();
        first.handle(Command::UserMessage("carry on without me".into()));
        // The screen closes straight away, before the answer.
        drop(first);

        let deadline = Instant::now() + WAIT;
        let hello = loop {
            let (_, hello) = Remote::attach(&path).unwrap();
            let answered = hello.since.iter().any(|r| {
                matches!(&r.kind, EventKind::Message { agent: Some(_), text } if text == "done while away")
            });
            if answered {
                break hello;
            }
            assert!(Instant::now() < deadline, "no answer: {:?}", hello.since);
            thread::sleep(Duration::from_millis(10));
        };
        // History from before the daemon is kept apart from what is live.
        assert_eq!(hello.before.len(), 1);
        assert!(hello.since.iter().any(|r| {
            matches!(&r.kind, EventKind::Message { agent: None, text } if text == "carry on without me")
        }));
        let _ = ask(&path, &ToDaemon::Stop { force: true }, WAIT);
    }

    #[test]
    fn a_scheduled_job_runs_with_no_screen_attached() {
        let now = crate::learn::now();
        let jobs = Arc::new(Mutex::new(Schedule::in_memory(0)));
        jobs.lock()
            .unwrap()
            .add("say the build status", "in 1m", Missed::Run, now - 300)
            .unwrap();
        let live = live("The build is green.").with_schedule(Some(jobs.clone()));
        let (path, _daemon) = start("job", live, Vec::new(), LONG);

        let deadline = Instant::now() + WAIT;
        while !jobs.lock().unwrap().jobs().is_empty() {
            assert!(Instant::now() < deadline, "the job never started");
            thread::sleep(Duration::from_millis(10));
        }
        let (mut screen, hello) = Remote::attach(&path).unwrap();
        let mut seen: Vec<EventKind> = hello.since.into_iter().map(|r| r.kind).collect();
        let answered = |e: &EventKind| matches!(e, EventKind::Message { agent: Some(_), text } if text == "The build is green.");
        if !seen.iter().any(answered) {
            seen.extend(wait_for(&mut screen, answered));
        }
        assert!(seen.iter().any(|e| matches!(
            e,
            EventKind::Message { agent: None, text } if text.starts_with("[Routine #1")
        )));
        let _ = ask(&path, &ToDaemon::Stop { force: true }, WAIT);
    }

    #[test]
    fn a_call_that_needs_approval_waits_for_a_screen_and_stop_is_refused_meanwhile() {
        use crate::guard::Guard;
        use crate::provider::testing::Queue;
        let root = scratch("approve");
        let core = Queue::new(
            "core",
            &[
                r#"CALL write_file {"path": "note.txt", "content": "done\n"}"#,
                "Wrote the note.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let jobs = Arc::new(Mutex::new(Schedule::in_memory(0)));
        jobs.lock()
            .unwrap()
            .add(
                "write the note",
                "in 1m",
                Missed::Run,
                crate::learn::now() - 300,
            )
            .unwrap();
        let live = Live::new(Arc::new(Router::single(Arc::new(core))))
            .with_guard(Some(guard))
            .with_schedule(Some(jobs));
        let (path, _daemon) = start("approve", live, Vec::new(), LONG);

        // The job starts with nobody watching and stops at the write.
        let asked = |e: &EventKind| match e {
            EventKind::ToolRequested {
                id,
                needs_approval: true,
                ..
            } => Some(*id),
            _ => None,
        };
        let deadline = Instant::now() + WAIT;
        let (mut screen, id) = loop {
            let (screen, hello) = Remote::attach(&path).unwrap();
            if let Some(id) = hello.since.iter().find_map(|r| asked(&r.kind)) {
                assert!(hello.status.busy);
                break (screen, id);
            }
            assert!(Instant::now() < deadline, "the write never asked");
            thread::sleep(Duration::from_millis(10));
        };
        assert!(
            !root.join("note.txt").exists(),
            "nothing is written unasked"
        );

        // A task is in progress, so a plain stop is refused.
        let refused = ask(&path, &ToDaemon::Stop { force: false }, WAIT).unwrap();
        assert!(matches!(refused, ToClient::Said { ok: false, text } if text.contains("--force")));

        // The screen that attached later gives the answer.
        screen.handle(Command::Decide { id, approve: true });
        wait_for(
            &mut screen,
            |e| matches!(e, EventKind::Message { agent: Some(_), text } if text == "Wrote the note."),
        );
        assert_eq!(
            std::fs::read_to_string(root.join("note.txt")).unwrap(),
            "done\n"
        );
        let _ = ask(&path, &ToDaemon::Stop { force: true }, WAIT);
    }

    #[test]
    fn it_ends_to_make_way_for_a_new_build_and_screens_wait_for_it() {
        let mut waiting = live("x");
        waiting.restart_due("self-update #3 was adopted");
        let (path, daemon) = start("restart", waiting, Vec::new(), LONG);
        // It restarts only once it is reached; a screen that was attached
        // is told it is coming back, not that it is gone.
        let attached = Remote::attach(&path);
        let reason = daemon.join().unwrap();
        assert_eq!(reason, "restart: self-update #3 was adopted");
        if let Ok((mut screen, _)) = attached {
            let seen = wait_for(&mut screen, |e| {
                matches!(e, EventKind::Alert { severity: Severity::Info, text }
                    if text.contains("restarting: self-update #3 was adopted"))
            });
            assert!(!seen.iter().any(|e| matches!(
                e,
                EventKind::Alert {
                    severity: Severity::Error,
                    ..
                }
            )));
            // Nothing is listening yet, so there is nothing to attach to.
            assert!(screen.reattach().is_none());
            assert!(screen.poll().is_empty(), "no error while it waits");
            // The new build comes up on the same socket: the screen attaches.
            let (_, _next) = start_at(&path, live("y"));
            let deadline = Instant::now() + WAIT;
            let hello = loop {
                if let Some(hello) = screen.reattach() {
                    break hello;
                }
                assert!(Instant::now() < deadline, "did not reconnect");
                thread::sleep(Duration::from_millis(10));
            };
            assert_eq!(hello.status.core, "fixed");
            assert!(screen.reattach().is_none(), "attached: no longer waiting");
            let _ = ask(&path, &ToDaemon::Stop { force: true }, WAIT);
        }
    }

    #[test]
    fn it_ends_by_itself_only_when_there_is_nothing_to_stay_for() {
        // Nothing attached, running or scheduled: it ends.
        let (_, daemon) = start("idle", live("x"), Vec::new(), Duration::from_millis(50));
        assert!(daemon.join().unwrap().starts_with("no screen was attached"));

        // A scheduled job keeps it.
        let jobs = Arc::new(Mutex::new(Schedule::in_memory(0)));
        jobs.lock()
            .unwrap()
            .add("later", "in 1h", Missed::Run, crate::learn::now())
            .unwrap();
        let kept = live("x").with_schedule(Some(jobs.clone()));
        let (path, daemon) = start("kept", kept, Vec::new(), Duration::from_millis(50));
        thread::sleep(Duration::from_millis(400));
        assert_eq!(status(&path, WAIT).unwrap().jobs, 1);
        assert!(!daemon.is_finished());

        // The schedule is read and changed through the daemon, which owns it.
        let listed = ask(&path, &ToDaemon::Schedule { cancel: None }, WAIT).unwrap();
        assert!(matches!(listed, ToClient::Said { ok: true, text } if text.contains("#1 once at")));
        let gone = ask(&path, &ToDaemon::Schedule { cancel: Some(9) }, WAIT).unwrap();
        assert!(matches!(gone, ToClient::Said { ok: false, .. }));
        let done = ask(&path, &ToDaemon::Schedule { cancel: Some(1) }, WAIT).unwrap();
        assert!(matches!(done, ToClient::Said { ok: true, .. }));
        // With the last job cancelled there is nothing to stay for.
        assert!(daemon.join().unwrap().starts_with("no screen was attached"));
    }

    #[test]
    fn only_one_daemon_listens_and_only_a_screen_may_send_commands() {
        let (path, _daemon) = start("one", live("x"), Vec::new(), LONG);
        assert!(bind(&path).is_err());

        // A connection that has not attached cannot act for the user.
        let mut stranger = UnixStream::connect(&path).unwrap();
        write_line(
            &mut stranger,
            &ToDaemon::Command(Command::UserMessage("do something".into())),
        )
        .unwrap();
        // Lines that are not the protocol are ignored.
        stranger.write_all(b"not json\n").unwrap();
        thread::sleep(Duration::from_millis(300));
        let now = status(&path, WAIT).unwrap();
        assert!(!now.busy);
        let (_, hello) = Remote::attach(&path).unwrap();
        assert!(
            !hello
                .since
                .iter()
                .any(|r| matches!(r.kind, EventKind::Message { .. }))
        );
        let _ = ask(&path, &ToDaemon::Stop { force: true }, WAIT);
    }

    #[test]
    fn a_dead_daemons_socket_file_is_replaced() {
        let path = scratch("stale").join("d.sock");
        drop(bind(&path).unwrap());
        assert!(path.exists(), "the file outlives its listener");
        assert!(status(&path, WAIT).is_err());
        assert!(bind(&path).is_ok());
    }
}
