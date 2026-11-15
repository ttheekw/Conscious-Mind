//! Live feed: the same events-out / commands-in surface as `demo`, backed by
//! real model providers.
//!
//! Every message the user sends opens a **task**: a quick answer or hours of
//! work. Tasks run side by side, each on its own worker thread with its own
//! conversation (`Task::thread`), so one task's context never carries
//! another's. What a task needs from earlier work reaches it through the
//! knowledge lookup, not by being carried along. Every event a task
//! produces is tagged with it (`Tagged`), which is how the screen shows one
//! tab per task.
//!
//! Fixed rules, decided here and not by a model: at most `MAX_RUNNING` tasks
//! run at once and the rest wait their turn in the order they were asked;
//! a scheduled job is a task like any other and takes a slot.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::agent;
use crate::demo::Command;
use crate::event::{
    AgentState, ContextSection, EventKind, LookupPick, LookupStep, MemKind, MemoryNode, Severity,
    Tagged, TaskState,
};
use crate::guard::Guard;
use crate::knowledge::{Digest, KnowledgeBase, MemoryKind, NodeId, digest};
use crate::provider::{Message, Request, Response, Role, estimate_tokens};
use crate::router::Router;
use crate::schedule::Schedule;
use crate::skills::Skills;

const SYSTEM: &str = "You are the model behind `conscious`, a goal-driven agent harness. \
No tools are connected in this mode, so you cannot run commands, read files or browse; \
answer from what is in the conversation and say so when a request needs tools.";

const CORE: u32 = 0;
/// Working memory of one task: about this many tokens of its own
/// conversation are sent to the model verbatim. Older turns, and everything
/// from other tasks, are recalled from the knowledge graph when relevant.
const HISTORY_BUDGET: u32 = 6_000;
/// Memories drawn in the memory view; more would not be readable.
const MAX_SHOWN_MEMORIES: usize = 40;
/// Characters of a memory's text sent to the screen with it.
const MEMORY_SHOWN_CHARS: usize = 600;
/// Earlier memories a model re-reads per start.
const MAX_REREAD: usize = 25;
/// Tasks that may run at the same time. More wait their turn.
pub const MAX_RUNNING: usize = 3;
/// The main conversation: the one task that is always there. Every message
/// goes to it; other tasks are branches it opened, or runs of a routine.
pub const MAIN: u32 = 0;
/// The folder, in the project, where files sent over Telegram are saved.
pub const TELEGRAM_INBOX: &str = "telegram-inbox";
/// What main is told about itself.
const MAIN_BRIEF: &str = "You are the main conversation with the user: one continuous thread \
    that keeps going. Here you answer, look things up and decide; you cannot change files \
    here. When what is wanted changes anything, or is a piece of work of its own (more than \
    a few steps, long, or separate from what is being talked about), you decide to open a \
    branch for it with the `branch` tool: it has every tool, runs by itself alongside, the \
    user can watch it in its own tab, and its result is merged back into this conversation \
    when it ends, marked as such. Open one without asking; do not do a branch's work here as \
    well, and do not wait for it. While branches run you stay free to talk. For work that \
    should repeat or happen later, set a routine with the `schedule` tool: each run of a \
    routine is a branch that merges back here too.";
/// What a branch is told about itself.
const BRANCH_BRIEF: &str = "You are working on a branch off the main conversation. You were \
    given this one piece of work and have nothing else from that conversation. When you end, \
    your final answer is merged into the main conversation: make it a complete account of \
    what was done and found, readable by itself.";
/// How often change-triggered jobs look at what they watch.
const WATCH_EVERY: Duration = Duration::from_secs(5);
/// How often the local time zone is read again, to follow daylight saving.
const ZONE_EVERY: Duration = Duration::from_secs(600);

/// How a kind of memory is shown in the memory view.
fn mem_kind(kind: Option<MemoryKind>) -> MemKind {
    match kind {
        Some(MemoryKind::Identity | MemoryKind::Instruction) => MemKind::Procedural,
        Some(MemoryKind::Fact) => MemKind::Semantic,
        Some(MemoryKind::Event | MemoryKind::Nothing) | None => MemKind::Episodic,
    }
}

enum Update {
    Event(EventKind),
    /// Text the model has just written, as it writes it.
    Streamed(String),
    /// Which model family ran the task and at what rated complexity, so a
    /// later rating of this task counts for the right one.
    Routed(Option<(String, u8)>),
    /// The call finished; `Some` carries the assistant's reply for history.
    Finished(Option<Message>),
    /// A model has read an exchange and said what it is (or failed to), so
    /// it can be remembered.
    Understood {
        question: String,
        answer: String,
        digest: Option<Digest>,
    },
    /// The same, for a memory stored before a model had read it.
    Reread {
        id: NodeId,
        digest: Digest,
    },
    /// The idle-time pass over memory has ended.
    Tidied,
    /// What the run's own checks showed about how it went.
    Outcome(agent::Outcome),
    /// The task's success criteria as they stood against the finished work.
    Checked(Vec<crate::goal::Checked>),
    /// The learner that works out what the user meant has ended.
    Learned,
    /// A task asked for the harness's own code to be changed.
    ChangeAsked {
        problem: String,
        change: String,
    },
    /// Something that arrived from outside the screen (a file sent over
    /// Telegram, once saved), to be said to main.
    Heard(String),
    /// Main asked for a branch to be opened.
    BranchAsked {
        title: String,
        task: String,
    },
    /// A self-update ended. `Some` says it was adopted, and why a restart
    /// is due.
    ChangeEnded(Option<String>),
    /// The skills the task that just ended had loaded, for rating them.
    SkillsUsed(Vec<String>),
}

/// The connection to a Telegram bot: what it has heard comes in on one
/// channel, what is to be sent goes out on another, each with a thread of
/// its own, so the feed never waits on the network.
struct Bridge {
    pairing: crate::telegram::Pairing,
    /// Where the pairing is saved, if anywhere.
    path: Option<std::path::PathBuf>,
    /// The code that makes whoever sends it the bot's owner.
    code: String,
    heard: Receiver<crate::telegram::Incoming>,
    out: Sender<Out2>,
    /// The bot itself, for fetching a file the owner sent.
    bot: Arc<dyn crate::telegram::Bot>,
    /// Calls put to the owner with buttons and not yet decided: the call,
    /// and what the message said.
    pending: HashMap<u32, String>,
    /// When the pairing code was last said again on the screen.
    reminded: Option<Instant>,
    /// When "typing…" was last shown.
    typed: Option<Instant>,
    /// Wrong pairing codes sent so far. At `PAIR_TRIES` the code is thrown
    /// away, so nobody can pair until a new one is issued at the screen.
    wrong: u32,
    /// Cleared when the bridge goes, so its reader thread ends.
    alive: Arc<AtomicBool>,
}

/// Something to send over Telegram.
enum Out2 {
    Say(i64, String),
    /// A call that needs approval, with Approve and Deny under it.
    Ask {
        chat: i64,
        tool: u32,
        text: String,
    },
    /// The call was decided: its message says so and loses its buttons.
    Settle {
        tool: u32,
        text: String,
    },
    /// One of the owner's messages is marked as seen.
    Seen {
        chat: i64,
        message: i64,
    },
    /// "typing…" is shown while work goes on.
    Typing(i64),
    /// A pressed button is acknowledged.
    Pressed {
        button: String,
        note: String,
    },
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
    }
}

impl Bridge {
    fn save(&self) {
        if let (Some(path), Ok(text)) = (&self.path, serde_json::to_string_pretty(&self.pairing)) {
            let _ = std::fs::write(path, text);
        }
    }
}

/// A reply as it is sent to a phone: plain text, and as short as the
/// cheapest model can make it without losing what matters. One that is
/// already short and plain goes as it is; if the model cannot be reached,
/// the reply goes with its markup taken out and nothing else changed.
fn for_phone(router: &Router, text: &str) -> String {
    use crate::telegram::{FOR_PHONE, already_brief, plain};
    if already_brief(text) {
        return plain(text);
    }
    let req = Request {
        system: FOR_PHONE.to_string(),
        messages: vec![Message::user(format!("<message>\n{text}\n</message>"))],
        ..Default::default()
    };
    match router.cheapest().complete(&req, &mut |_| {}) {
        Ok(resp) if !resp.text.trim().is_empty() => crate::guard::redact(&plain(&resp.text)),
        _ => plain(text),
    }
}

/// Who is waiting for a secret to be typed: (task, name, where to answer).
type SecretWaiters = Arc<Mutex<Vec<(u32, String, Sender<bool>)>>>;

/// Where a worker thread reports to: the feed, with the task it works for.
#[derive(Clone)]
struct Out {
    task: Option<u32>,
    tx: Sender<(Option<u32>, Update)>,
}

impl Out {
    fn send(&self, update: Update) {
        let _ = self.tx.send((self.task, update));
    }
}

/// One thing that was asked for, and the conversation about it.
struct Task {
    id: u32,
    /// What it is called on its tab.
    title: String,
    /// The reply being written right now, as far as it has come, and
    /// whether the screens have been sent it as it stands.
    draft: String,
    draft_sent: bool,
    /// Its own conversation: the only turns sent to the model verbatim.
    thread: Vec<Message>,
    /// The message being answered, to remember with its answer.
    asked: Option<String>,
    /// Remembered exchanges that are still in `thread`, oldest first, so a
    /// lookup does not hand the model what it already has.
    in_window: VecDeque<NodeId>,
    /// A message waiting for a free slot.
    queued: Option<String>,
    running: bool,
    /// Where the user's approve/deny decisions go while it runs.
    decisions: Option<Sender<(u32, bool)>>,
    /// Set to end it early.
    stop: Arc<AtomicBool>,
    /// The skills its last run loaded, which the user's rating applies to.
    skills: Vec<String>,
    /// The scheduled job it is a run of, if it is one.
    job: Option<u32>,
    /// The model family and complexity of its last run, and whether the
    /// user has rated that run.
    routed: Option<(String, u8)>,
    rated: bool,
    /// Messages that have left its working memory.
    dropped: usize,
    /// The lessons its current run has been given, which its rating counts
    /// for. Shared with the run, which adds what `recall` hands it.
    given: Arc<Mutex<Vec<NodeId>>>,
    /// Whether the user's rating of this run has been counted for lessons.
    judged: bool,
    /// How many standing rules were in front of its current run.
    rules: usize,
    /// The success criteria of its last run, as checked against the work,
    /// what was asked and what was answered: what the user's response is
    /// compared with.
    criteria: Vec<crate::goal::Checked>,
    last_asked: String,
    answer: String,
}

pub struct Live {
    router: Arc<Router>,
    /// The knowledge store: looked up for every message, and where each
    /// exchange is remembered.
    knowledge: Option<Arc<Mutex<KnowledgeBase>>>,
    tasks: Vec<Task>,
    next_task: u32,
    /// The task that finished most recently, which an unaddressed rating
    /// applies to.
    last_finished: Option<u32>,
    tx: Sender<(Option<u32>, Update)>,
    rx: Receiver<(Option<u32>, Update)>,
    announced: bool,
    /// Results of the start-up checks, shown once: (is a warning, text).
    notices: Vec<(bool, String)>,
    /// (when, characters) for text streamed in the last second.
    recent: Arc<Mutex<VecDeque<(Instant, usize)>>>,
    /// Lessons learned from feedback this session, for memory node ids.
    lessons: u32,
    /// Whether this session continues an earlier one.
    resumed: bool,
    /// The guardrails. With them the model is given tools and works as an
    /// agent; without them it can only talk.
    guard: Option<Arc<Guard>>,
    /// Source of tool-call ids, so they never repeat within a session.
    tool_ids: Arc<AtomicU32>,
    /// Source of ids for sub-agents; the core agent is 0.
    agent_ids: Arc<AtomicU32>,
    /// The procedures the agent has written for itself.
    skills: Option<Arc<Mutex<Skills>>>,
    /// Jobs set to start by themselves; checked whenever a slot is free.
    schedule: Option<Arc<Mutex<Schedule>>>,
    /// When the change-triggered jobs last looked at what they watch, and
    /// when the local time zone was last read.
    watched_at: Instant,
    zone_read_at: Instant,
    /// Exchanges a model is still reading before they are remembered.
    reading: usize,
    /// Which task is changing which file.
    claims: Arc<Mutex<crate::claims::Claims>>,
    /// How to load the model that finds passages by meaning; taken when it
    /// is started, once.
    meaning: Option<MeaningLoader>,
    /// Whether each task is given success criteria, checked against them,
    /// and the criteria judged by how the user responds.
    goals: bool,
    /// Learners still working out what the user meant.
    learning: usize,
    /// Whether the harness may rewrite its own source when a task asks.
    self_update: bool,
    /// Whether a self-update is under way (one at a time).
    updating: bool,
    /// Whether a message is shown before its context is assembled.
    echo_first: bool,
    /// Tasks waiting for the user to type a secret: (task, the secret's
    /// name, where the answer goes).
    secret_waiters: SecretWaiters,
    /// The Telegram bot the main conversation can also be reached by.
    telegram: Option<Bridge>,
    /// Set when a self-update was adopted: the harness should restart into
    /// the new build once nothing is running. Says which update.
    restart: Option<String>,
    /// Where the harness's source and its self-update folder are, when not
    /// the real ones (tests).
    update_dirs: Option<(std::path::PathBuf, std::path::PathBuf)>,
    /// Since when nothing has been running, and when memory was last gone
    /// over in idle time. `tidying` is set while a pass is under way.
    idle_since: Option<Instant>,
    tidied_at: Option<Instant>,
    tidying: bool,
    /// How long nothing must have been running before memory is gone over.
    tidy_after: Duration,
}

/// How long the harness must have been idle before it goes over memory.
const IDLE_BEFORE_TIDY: Duration = Duration::from_secs(60);
/// The least time between two idle-time passes over memory.
const TIDY_EVERY: Duration = Duration::from_secs(30 * 60);
/// The model calls one idle-time pass may make, by what they are for:
/// earlier memories read, pairs of near-repeats judged, summaries written.
/// Together they are the pass's whole budget.
const TIDY_REREAD: usize = 10;
const TIDY_JUDGE: usize = 5;
const TIDY_SUMMARIES: usize = 2;

/// Loads the embedding model. A function, because loading can take a while
/// (the first time, it downloads the model) and so happens off the main
/// thread.
pub type MeaningLoader =
    Box<dyn FnOnce() -> Result<Arc<dyn crate::knowledge::Embedder>, String> + Send>;

/// How often the meaning index is brought up to date with what was added.
const EMBED_EVERY: Duration = Duration::from_secs(2);

fn tag(task: u32, kinds: Vec<EventKind>) -> Vec<Tagged> {
    kinds
        .into_iter()
        .map(|kind| Tagged {
            task: Some(task),
            kind,
        })
        .collect()
}

/// The first line of a message, cut to fit a tab.
/// A memory as the screen is sent it: its title, and what the store holds
/// behind it, so that opening it on the screen shows the real thing.
fn memory_node(kb: &KnowledgeBase, id: NodeId, label: String, kind: MemKind) -> MemoryNode {
    let (text, about) = match kb.node(id) {
        Some(node) => {
            let mut text: String = node.text.chars().take(MEMORY_SHOWN_CHARS).collect();
            if text.len() < node.text.len() {
                text.push_str(" […]");
            }
            let trust = match node.provenance.trust {
                crate::knowledge::Trust::Inferred => "not confirmed yet",
                crate::knowledge::Trust::Experience => "confirmed by how tasks went",
                crate::knowledge::Trust::Verified => "verified",
            };
            let record = &node.record;
            let mut about = format!(
                "{trust} · {} · {}",
                KnowledgeBase::origin(node),
                KnowledgeBase::privacy(node).label()
            );
            if record.given > 0 {
                about.push_str(&format!(
                    " · given to {} task{}",
                    record.given,
                    if record.given == 1 { "" } else { "s" }
                ));
            }
            if record.good + record.bad > 0 {
                about.push_str(&format!(
                    " · rated good {} and bad {}",
                    record.good, record.bad
                ));
            }
            (text, about)
        }
        None => (String::new(), String::new()),
    };
    MemoryNode {
        id: id as u32,
        kind,
        label,
        text,
        about,
    }
}

/// Adds what the user side says to a conversation. Two such messages in a
/// row become one, since some models only take turns that alternate.
fn push_user(thread: &mut Vec<Message>, text: String) {
    match thread.last_mut() {
        Some(last) if last.role == crate::provider::Role::User => {
            last.content.push_str("\n\n");
            last.content.push_str(&text);
        }
        _ => thread.push(Message::user(text)),
    }
}

/// What main's conversation holds of a branch that has ended. A long
/// result is cut to half of main's working memory; all of it is in the
/// branch's own tab and in memory, where a lookup finds it.
fn merge_note(branch: u32, title: &str, result: Option<&str>) -> String {
    let Some(result) = result else {
        return format!(
            "[Branch #{branch} \"{title}\" ended without an answer, so nothing was merged.]"
        );
    };
    let room = HISTORY_BUDGET as usize * 2;
    let mut kept: String = result.chars().take(room).collect();
    if kept.len() < result.len() {
        kept.push_str("\n[cut here; the rest is in the branch and in memory]");
    }
    format!(
        "[Branch #{branch} \"{title}\" has ended and its result is merged into this \
         conversation. The user has been shown it. Its result:]\n{kept}"
    )
}

fn title_of(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default()
        .chars()
        .take(90)
        .collect()
}

impl Live {
    #[cfg(test)]
    pub fn new(router: Arc<Router>) -> Self {
        Self::with_knowledge(router, None)
    }

    pub fn with_knowledge(
        router: Arc<Router>,
        knowledge: Option<Arc<Mutex<KnowledgeBase>>>,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            router,
            knowledge,
            tasks: Vec::new(),
            next_task: 1,
            last_finished: None,
            tx,
            rx,
            announced: false,
            recent: Arc::default(),
            lessons: 0,
            resumed: false,
            guard: None,
            tool_ids: Arc::new(AtomicU32::new(1)),
            agent_ids: Arc::new(AtomicU32::new(1)),
            skills: None,
            notices: Vec::new(),
            schedule: None,
            watched_at: Instant::now(),
            zone_read_at: Instant::now(),
            reading: 0,
            claims: Arc::default(),
            meaning: None,
            goals: false,
            learning: 0,
            self_update: false,
            updating: false,
            echo_first: false,
            secret_waiters: Arc::default(),
            telegram: None,
            restart: None,
            update_dirs: None,
            idle_since: None,
            tidied_at: None,
            tidying: false,
            tidy_after: IDLE_BEFORE_TIDY,
        }
    }

    fn out(&self, task: Option<u32>) -> Out {
        Out {
            task,
            tx: self.tx.clone(),
        }
    }

    pub fn router(&self) -> Arc<Router> {
        self.router.clone()
    }

    pub fn knowledge(&self) -> Option<Arc<Mutex<KnowledgeBase>>> {
        self.knowledge.clone()
    }

    /// Gives the model tools, under these guardrails.
    pub fn with_guard(mut self, guard: Option<Arc<Guard>>) -> Self {
        self.guard = guard;
        self
    }

    pub fn guard(&self) -> Option<Arc<Guard>> {
        self.guard.clone()
    }

    /// Lets the agent keep and reuse procedures it writes for itself.
    pub fn with_skills(mut self, skills: Option<Arc<Mutex<Skills>>>) -> Self {
        self.skills = skills;
        self
    }

    /// Lets the harness rewrite its own source when a task asks for it
    /// (`selfupdate`): in a copy, adopted only through the gate.
    pub fn with_self_update(mut self, on: bool) -> Self {
        self.self_update = on;
        self
    }

    /// Marks a restart as due, as an adopted self-update would.
    #[cfg(test)]
    pub fn restart_due(&mut self, why: &str) {
        self.restart = Some(why.to_string());
    }

    /// Why the harness should restart now, if a self-update was adopted
    /// and nothing is running that a restart would cut off. Asking takes
    /// the reason: it is given once.
    pub fn take_restart(&mut self) -> Option<String> {
        if self.settled() {
            self.restart.take()
        } else {
            None
        }
    }

    /// Starts a self-update as a task of its own, or says why not.
    fn start_self_update(&mut self, problem: String, change: String) -> Vec<Tagged> {
        let refuse = |why: &str| {
            vec![Tagged::system(EventKind::Alert {
                severity: Severity::Warn,
                text: format!(
                    "A change to the harness's own code was asked for, but not started: {why}."
                ),
            })]
        };
        if !self.self_update {
            return refuse("self-update is switched off (--no-self-update)");
        }
        if self.updating || self.restart.is_some() {
            return refuse("another self-update is already under way");
        }
        let dirs = self
            .update_dirs
            .clone()
            .or_else(|| Some((crate::selfupdate::source_dir()?, crate::selfupdate::home()?)));
        let Some((source, home)) = dirs else {
            return refuse(
                "the source this program was built from is not on this machine any more",
            );
        };
        // The copy builds every dependency for itself the first time.
        if self.update_dirs.is_none()
            && let Some(free) = crate::selfupdate::free_bytes(&home)
            && free < crate::selfupdate::NEEDS_FREE
        {
            return refuse(&format!(
                "only {:.1} GB of disk is free and building a copy of the harness needs {} GB",
                free as f64 / 1e9,
                crate::selfupdate::NEEDS_FREE / 1_000_000_000
            ));
        }
        let title = format!("Change the harness itself: {change}");
        let (id, mut events) = self.open(&title, "the harness itself");
        let at = self.slot(id);
        self.updating = true;
        self.tasks[at].running = true;
        self.tasks[at].stop = Arc::default();
        let (decide, decisions) = mpsc::channel();
        self.tasks[at].decisions = Some(decide);
        let env = UpdateEnv {
            router: self.router.clone(),
            tool_ids: self.tool_ids.clone(),
            agent_ids: self.agent_ids.clone(),
            task: id,
            source,
            home,
            problem: problem.clone(),
            change,
        };
        let (tx, stop) = (self.out(Some(id)), self.tasks[at].stop.clone());
        thread::spawn(move || {
            let adopted = run_self_update(&env, &tx, &decisions, &stop);
            tx.send(Update::ChangeEnded(adopted));
            tx.send(Update::Finished(None));
        });
        events.extend(tag(
            id,
            vec![
                EventKind::Message {
                    agent: None,
                    text: format!("[Asked for by a task, not typed by you.]\n{problem}"),
                },
                EventKind::TaskUpdate {
                    state: TaskState::Working,
                    note: "copying its own source to work on".into(),
                },
            ],
        ));
        events
    }

    /// Has every task given success criteria before it starts, checked
    /// against them when it ends, and the criteria themselves judged by the
    /// user's response: where the two disagree, a learner works out what
    /// the user meant, for the next time criteria are written.
    pub fn with_goals(mut self, on: bool) -> Self {
        self.goals = on;
        self
    }

    /// Compares the user's response to a finished task with the success
    /// criteria it was judged by. If they disagree (or the response is in
    /// words, which only a model can read), a learner is started to work
    /// out what the user wanted. Returns what to say about it now.
    fn weigh_criteria(&mut self, at: usize, said: crate::goal::Said) -> Vec<EventKind> {
        use crate::goal::{Agreement, Said, agreement};
        let task = &self.tasks[at];
        if !self.goals || task.criteria.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        if let Said::Rated(good) = &said {
            let how = agreement(&task.criteria, *good);
            let line = match how {
                Agreement::Unclear => return Vec::new(),
                Agreement::Agree => None,
                Agreement::MissedTheIntent => Some(
                    "Every success criterion was met, yet you rated the result bad: the \
                     criteria measured the wrong thing. Working out what you wanted.",
                ),
                Agreement::AskedTooMuch => Some(
                    "A success criterion was not met, yet you rated the result good: the \
                     criteria asked for something you did not need. Working out what matters \
                     to you.",
                ),
            };
            if let Some(kb) = &self.knowledge {
                kb.lock().unwrap().criteria_judged(how == Agreement::Agree);
            }
            match line {
                None => return Vec::new(),
                Some(text) => events.push(EventKind::Alert {
                    severity: Severity::Warn,
                    text: text.into(),
                }),
            }
        }
        // The learner: a small model reads the task, the criteria, the
        // answer and the response, off the main thread.
        let (router, kb, tx) = (
            self.router.clone(),
            self.knowledge.clone(),
            self.out(Some(task.id)),
        );
        let (id, asked, checked, answer) = (
            task.id,
            task.last_asked.clone(),
            task.criteria.clone(),
            task.answer.clone(),
        );
        self.learning += 1;
        thread::spawn(move || {
            learn_what_was_meant(
                &router,
                kb.as_deref(),
                &tx,
                id,
                &asked,
                &checked,
                &answer,
                &said,
            );
            tx.send(Update::Learned);
        });
        events
    }

    /// Switches on lookup by meaning. The model is loaded in the
    /// background once the session starts; until it is ready, and if it
    /// cannot be had, lookups are by words and links.
    pub fn with_meaning(mut self, loader: Option<MeaningLoader>) -> Self {
        self.meaning = loader;
        self
    }

    /// Loads the meaning model off the main thread, then keeps the meaning
    /// index up to date with whatever is added, for as long as the
    /// knowledge store is in use.
    fn start_meaning(&mut self) {
        let (Some(loader), Some(kb)) = (self.meaning.take(), &self.knowledge) else {
            return;
        };
        let (kb, tx) = (Arc::downgrade(kb), self.out(None));
        let say = move |severity: Severity, text: String| {
            tx.send(Update::Event(EventKind::Alert { severity, text }));
        };
        thread::spawn(move || {
            let model = match loader() {
                Ok(model) => model,
                Err(why) => {
                    say(
                        Severity::Warn,
                        format!(
                            "Lookup by meaning is off, so lookups are by words and links \
                             only: the model could not be loaded ({why})."
                        ),
                    );
                    return;
                }
            };
            match kb.upgrade() {
                Some(kb) => kb.lock().unwrap().set_embedder(model.clone()),
                None => return,
            }
            let mut first = true;
            // Ends when the store is no longer in use.
            while let Some(kb) = kb.upgrade() {
                match crate::knowledge::embed_pending(&kb, model.as_ref()) {
                    Ok(made) if first || made >= 20 => say(
                        Severity::Info,
                        format!(
                            "Lookup by meaning is on{}: {} passages and memories can now be \
                             found by what they mean, not only by their words.",
                            if made > 0 {
                                format!(" ({made} newly read by the local model)")
                            } else {
                                String::new()
                            },
                            kb.lock().unwrap().stats().embedded
                        ),
                    ),
                    Ok(_) => {}
                    Err(why) => {
                        say(
                            Severity::Warn,
                            format!("The meaning index could not be brought up to date: {why}"),
                        );
                        return;
                    }
                }
                first = false;
                drop(kb);
                thread::sleep(EMBED_EVERY);
            }
        });
    }

    /// Uses idle time on memory, the way sleep is used: once nothing has
    /// been running for `tidy_after`, and at most every `TIDY_EVERY`:
    /// notes that plainly say the same thing are merged (no model); pairs
    /// that may or may not say the same are put to a model, and its answer
    /// is kept either way; memories stored before a model had read them
    /// are read; and memories that have gathered on one subject get one
    /// summary, the memories themselves being kept. The model calls are
    /// capped (`TIDY_JUDGE`, `TIDY_REREAD`, `TIDY_SUMMARIES`) and go to the
    /// cheapest model. It runs off the main thread and says what it did
    /// only if it did something.
    fn tidy_when_idle(&mut self) {
        if self.busy() || self.reading > 0 {
            self.idle_since = None;
            return;
        }
        let idle = self.idle_since.get_or_insert_with(Instant::now).elapsed();
        let due = self.tidied_at.is_none_or(|at| at.elapsed() >= TIDY_EVERY);
        let Some(kb) = &self.knowledge else {
            return;
        };
        if self.tidying || idle < self.tidy_after || !due {
            return;
        }
        self.tidying = true;
        self.tidied_at = Some(Instant::now());
        let (kb, router, tx) = (kb.clone(), self.router.clone(), self.out(None));
        thread::spawn(move || {
            let merged = kb.lock().unwrap().merge_repeats();
            for group in &merged {
                tx.send(Update::Event(EventKind::MemoryConsolidated {
                    agent: CORE,
                    from: group.from.iter().map(|id| *id as u32).collect(),
                    into: MemoryNode {
                        id: group.into as u32,
                        kind: MemKind::Semantic,
                        label: group.title.clone(),
                        text: String::new(),
                        about: String::new(),
                    },
                }));
            }
            let consolidated = |from: Vec<NodeId>, into: NodeId, label: String| {
                tx.send(Update::Event(EventKind::MemoryConsolidated {
                    agent: CORE,
                    from: from.iter().map(|id| *id as u32).collect(),
                    into: MemoryNode {
                        id: into as u32,
                        kind: MemKind::Semantic,
                        label,
                        text: String::new(),
                        about: String::new(),
                    },
                }));
            };
            let model = router.cheapest();
            let mut repeats: usize = merged.iter().map(|g| g.from.len()).sum();
            // Pairs too alike to be sure of either way: a model says which.
            let near = kb.lock().unwrap().near_repeats(TIDY_JUDGE);
            for (keep, other, a, b) in near {
                let Ok((Some(same), call)) =
                    crate::knowledge::same_statement(model.as_ref(), &a, &b)
                else {
                    continue;
                };
                tx.send(Update::Event(model_call(&call)));
                if let Some(group) = kb.lock().unwrap().settle_pair(keep, other, same) {
                    repeats += group.from.len();
                    consolidated(group.from, group.into, group.title);
                }
            }
            let pending: Vec<_> = kb
                .lock()
                .unwrap()
                .undigested()
                .into_iter()
                .take(TIDY_REREAD)
                .collect();
            let mut read = 0;
            for (id, question, answer) in pending {
                if let Ok((Some(d), call)) = digest(model.as_ref(), &question, &answer) {
                    read += 1;
                    tx.send(Update::Event(model_call(&call)));
                    tx.send(Update::Reread { id, digest: d });
                }
            }
            // Memories that have gathered on one subject get one summary.
            let groups = kb.lock().unwrap().related_groups(TIDY_SUMMARIES);
            let mut summaries = 0;
            for group in groups {
                let (ids, texts): (Vec<NodeId>, Vec<String>) = group.into_iter().unzip();
                let Ok((summary, call)) = crate::knowledge::summarise(model.as_ref(), &texts)
                else {
                    continue;
                };
                tx.send(Update::Event(model_call(&call)));
                let summary = summary.map(|(title, said)| {
                    (crate::guard::redact(&title), crate::guard::redact(&said))
                });
                let made = kb.lock().unwrap().add_summary(
                    &ids,
                    summary.as_ref().map(|(t, s)| (t.as_str(), s.as_str())),
                );
                if let (Some(node), Some((title, _))) = (made, summary) {
                    summaries += 1;
                    consolidated(ids, node, format!("summary: {title}"));
                }
            }
            if repeats + read + summaries > 0 {
                let mut did = Vec::new();
                if summaries > 0 {
                    did.push(format!(
                        "wrote {summaries} summar{} of memories that had gathered on one subject",
                        if summaries == 1 { "y" } else { "ies" }
                    ));
                }
                if repeats > 0 {
                    did.push(format!(
                        "merged {repeats} memor{} that repeated another",
                        if repeats == 1 { "y" } else { "ies" }
                    ));
                }
                if read > 0 {
                    did.push(format!(
                        "worked out what {read} earlier memor{} {}",
                        if read == 1 { "y" } else { "ies" },
                        if read == 1 { "is" } else { "are" }
                    ));
                }
                tx.send(Update::Event(EventKind::Alert {
                    severity: Severity::Info,
                    text: format!("While idle, went over memory: {}.", did.join(" and ")),
                }));
            }
            tx.send(Update::Tidied);
        });
    }

    /// What the start-up checks found, to say once in the timeline.
    pub fn with_notices(mut self, notices: Vec<(bool, String)>) -> Self {
        self.notices.splice(0..0, notices);
        self
    }

    /// Lets the agent set jobs for later, and starts them when due. Runs
    /// that were missed while the harness was off are settled here, by each
    /// job's own rule.
    pub fn with_schedule(mut self, schedule: Option<Arc<Mutex<Schedule>>>) -> Self {
        if let Some(schedule) = &schedule {
            let mut s = schedule.lock().unwrap();
            let now = crate::learn::now();
            s.skip_unseen(&crate::schedule::stamp);
            for note in s.settle(now) {
                self.notices.push((true, note));
            }
            if !s.jobs().is_empty() {
                self.notices.push((false, s.report(now)));
            }
        }
        self.schedule = schedule;
        self
    }

    pub fn schedule(&self) -> Option<Arc<Mutex<Schedule>>> {
        self.schedule.clone()
    }

    /// How many branches are running. Main is not counted: it always has
    /// its own place, so the conversation is never held up by branches.
    fn running(&self) -> usize {
        self.tasks
            .iter()
            .filter(|t| t.running && t.id != MAIN)
            .count()
    }

    /// Whether any task is running or waiting its turn.
    pub fn busy(&self) -> bool {
        self.tasks.iter().any(|t| t.running || t.queued.is_some())
    }

    /// Whether nothing is running and nothing is waiting to be remembered,
    /// so stopping now would lose nothing.
    pub fn settled(&self) -> bool {
        !self.busy() && self.reading == 0 && self.learning == 0 && !self.tidying
    }

    /// A fresh start with the same models, memory, guardrails, skills and
    /// schedule, for the restart command. Task numbers carry on, since the
    /// earlier ones are still in the timeline.
    pub fn restarted(&self) -> Live {
        let mut fresh = Live::with_knowledge(self.router(), self.knowledge())
            .with_goals(self.goals)
            .with_guard(self.guard())
            .with_skills(self.skills())
            .with_schedule(self.schedule());
        fresh.next_task = self.next_task;
        fresh.echo_first = self.echo_first;
        fresh
    }

    /// What is remembered, as the memory view shows it.
    pub fn memory_nodes(&self) -> Vec<MemoryNode> {
        match &self.knowledge {
            Some(kb) => Self::shown_memories(&kb.lock().unwrap()),
            None => Vec::new(),
        }
    }

    fn shown_memories(kb: &KnowledgeBase) -> Vec<MemoryNode> {
        kb.memories()
            .into_iter()
            .take(MAX_SHOWN_MEMORIES)
            .map(|(id, title, kind)| memory_node(kb, id, title, mem_kind(kind)))
            .collect()
    }

    /// Starts the job that is due, if there is one and a slot is free.
    fn start_due_job(&mut self, now: u64) -> Vec<Tagged> {
        if self.running() >= MAX_RUNNING {
            return Vec::new();
        }
        let Some(schedule) = &self.schedule else {
            return Vec::new();
        };
        let (job, rule) = {
            let mut s = schedule.lock().unwrap();
            // Daylight saving moves local time; daily jobs follow it.
            if self.zone_read_at.elapsed() >= ZONE_EVERY {
                self.zone_read_at = Instant::now();
                s.set_offset(crate::schedule::utc_offset(), now);
            }
            if self.watched_at.elapsed() >= WATCH_EVERY {
                self.watched_at = Instant::now();
                s.watch(now, &crate::schedule::stamp);
            }
            let Some(job) = s.due(now).cloned() else {
                return Vec::new();
            };
            s.started(job.id, now);
            let rule = job.rule(s.offset());
            (job, rule)
        };
        let mut out = vec![Tagged::system(EventKind::Alert {
            severity: Severity::Info,
            text: format!(
                "Routine #{} ({rule}) is due; running it on a branch.",
                job.id
            ),
        })];
        let text = format!(
            "[Routine #{}, set earlier to run {rule}. It has started by itself; \
             nobody may be at the screen to answer questions.]\n{}",
            job.id, job.task
        );
        let (id, opened) = self.open(&job.task, &format!("routine #{}", job.id));
        out.extend(opened);
        if let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) {
            task.job = Some(job.id);
        }
        out.extend(self.begin(id, text));
        out
    }

    pub fn skills(&self) -> Option<Arc<Mutex<Skills>>> {
        self.skills.clone()
    }

    /// Continues from earlier sessions: each task recorded in the timeline
    /// gets its conversation back, as far as the budget allows, so a reply
    /// to it carries on where it stopped.
    pub fn resume(&mut self, recorded: &[crate::timeline::Recorded]) {
        self.resumed = !recorded.is_empty();
        // Tool calls from earlier sessions are back on screen with their
        // numbers, so new calls must be numbered after them. Otherwise an
        // approval prompt would open on an old call with the same number.
        let last = recorded
            .iter()
            .filter_map(|r| match &r.kind {
                EventKind::ToolRequested { id, .. } => Some(*id),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        self.tool_ids.fetch_max(last + 1, Ordering::Relaxed);
        // The same goes for sub-agents shown in the restored timeline.
        let last_agent = recorded
            .iter()
            .filter_map(|r| match &r.kind {
                EventKind::AgentSpawned { id, .. } => Some(*id),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        self.agent_ids.fetch_max(last_agent + 1, Ordering::Relaxed);
        for r in recorded {
            let Some(id) = r.task else {
                continue;
            };
            self.next_task = self.next_task.max(id + 1);
            if let EventKind::Merged {
                branch,
                title,
                result,
                ok,
            } = &r.kind
            {
                let at = self.slot(id);
                let note = merge_note(*branch, title, ok.then_some(result.as_str()));
                push_user(&mut self.tasks[at].thread, note);
            }
            if let EventKind::Message { agent, text } = &r.kind {
                let at = self.slot(id);
                self.tasks[at].thread.push(match agent {
                    None => Message::user(text.clone()),
                    Some(_) => Message::assistant(text.clone(), Vec::new()),
                });
            }
        }
        for at in 0..self.tasks.len() {
            self.trim_thread(at);
            self.tasks[at].dropped = 0;
        }
    }

    /// The position of task `id`, creating it if it is not known yet.
    fn slot(&mut self, id: u32) -> usize {
        if let Some(at) = self.tasks.iter().position(|t| t.id == id) {
            return at;
        }
        self.tasks.push(Task {
            id,
            title: String::new(),
            draft: String::new(),
            draft_sent: true,
            thread: Vec::new(),
            asked: None,
            in_window: VecDeque::new(),
            queued: None,
            running: false,
            decisions: None,
            stop: Arc::default(),
            skills: Vec::new(),
            job: None,
            routed: None,
            rated: false,
            dropped: 0,
            given: Arc::default(),
            judged: false,
            rules: 0,
            criteria: Vec::new(),
            last_asked: String::new(),
            answer: String::new(),
        });
        self.tasks.len() - 1
    }

    /// Creates a task and announces it.
    fn open(&mut self, text: &str, origin: &str) -> (u32, Vec<Tagged>) {
        let id = self.next_task;
        self.next_task += 1;
        let at = self.slot(id);
        self.tasks[at].title = title_of(text);
        let opened = tag(
            id,
            vec![EventKind::TaskOpened {
                title: title_of(text),
                origin: origin.to_string(),
            }],
        );
        (id, opened)
    }

    /// Returns every event produced since the last poll.
    pub fn poll(&mut self) -> Vec<Tagged> {
        let mut tagged: Vec<Tagged> = Vec::new();
        let mut out: Vec<EventKind> = Vec::new();
        let announced_before = self.announced;
        if !self.announced {
            self.announced = true;
            // The restored timeline already has the core agent in it.
            if !self.resumed {
                out.push(EventKind::AgentSpawned {
                    id: CORE,
                    parent: None,
                    name: "core".into(),
                    purpose: format!("talks to {}", self.router.core().label()),
                });
            }
            out.push(EventKind::Alert {
                severity: Severity::Info,
                text: format!(
                    "{}Core model: {}. {} {} Press i to talk to it: it answers here, and opens a branch when something is work of its own. + or - rates a result.",
                    if self.resumed {
                        "Picking up where the last session ended. "
                    } else {
                        ""
                    },
                    self.router.core().label(),
                    if self.router.routes() {
                        format!(
                            "It rates each task and routes it across {} models.",
                            self.router.candidates().len()
                        )
                    } else {
                        "Routing is off.".to_string()
                    },
                    match &self.guard {
                        Some(g) => g.describe(),
                        None => "It has no tools in this session, so it can only talk.".into(),
                    }
                ),
            });
        }
        if !announced_before {
            // Say plainly what is switched on, so an empty panel is not
            // mistaken for a missing feature.
            for (warning, text) in std::mem::take(&mut self.notices) {
                out.push(EventKind::Alert {
                    severity: if warning {
                        Severity::Warn
                    } else {
                        Severity::Info
                    },
                    text,
                });
            }
            out.push(EventKind::Alert {
                severity: Severity::Info,
                text: format!("Routing policy: {}", self.router.policy_line()),
            });
            out.push(EventKind::Alert {
                severity: Severity::Info,
                text: match &self.knowledge {
                    Some(kb) => kb.lock().unwrap().summary(),
                    None => "Knowledge graph: switched off for this run (--no-knowledge), so \
                             nothing is looked up or remembered."
                        .into(),
                },
            });
        }
        if !announced_before {
            self.start_meaning();
        }
        if !announced_before && let Some(kb) = &self.knowledge {
            let kb = kb.lock().unwrap();
            // The memory view shows what is remembered, not project files.
            out.push(EventKind::MemoryIndexed {
                nodes: Self::shown_memories(&kb),
            });
            // Memories stored before a model had read them are read now, in
            // the background, so they can be found by what they mean.
            let pending: Vec<_> = kb.undigested().into_iter().take(MAX_REREAD).collect();
            if !pending.is_empty() {
                let (router, tx) = (self.router.clone(), self.out(None));
                thread::spawn(move || {
                    let model = router.cheapest();
                    let mut read = 0;
                    for (id, question, answer) in pending {
                        if let Ok((Some(d), _)) = digest(model.as_ref(), &question, &answer) {
                            read += 1;
                            tx.send(Update::Reread { id, digest: d });
                        }
                    }
                    tx.send(Update::Event(EventKind::Alert {
                        severity: Severity::Info,
                        text: format!(
                            "Went back over {read} earlier memor{} to work out what each one is.",
                            if read == 1 { "y" } else { "ies" }
                        ),
                    }));
                });
            }
        }
        tagged.extend(out.into_iter().map(Tagged::system));
        while let Ok((task, update)) = self.rx.try_recv() {
            let at = task.and_then(|id| self.tasks.iter().position(|t| t.id == id));
            match update {
                Update::Event(kind) => {
                    // A model call has ended: what it was writing is now
                    // an event of its own (or was a tool request).
                    if let (Some(at), EventKind::ModelCall { .. } | EventKind::Message { .. }) =
                        (at, &kind)
                    {
                        self.set_draft(at, String::new());
                    }
                    tagged.push(Tagged { task, kind });
                }
                Update::Streamed(text) => {
                    self.recent
                        .lock()
                        .unwrap()
                        .push_back((Instant::now(), text.chars().count()));
                    if let Some(at) = at {
                        let so_far = format!("{}{text}", self.tasks[at].draft);
                        self.set_draft(at, so_far);
                    }
                }
                Update::Routed(routed) => {
                    if let Some(at) = at {
                        self.tasks[at].routed = routed;
                        self.tasks[at].rated = false;
                    }
                }
                Update::Finished(reply) => {
                    let Some(at) = at else {
                        continue;
                    };
                    self.tasks[at].running = false;
                    self.tasks[at].decisions = None;
                    self.set_draft(at, String::new());
                    // The files it was changing are free for other tasks.
                    self.claims.lock().unwrap().release(self.tasks[at].id);
                    self.last_finished = task;
                    // What a change-triggered job changed itself does not
                    // set it off again.
                    if let (Some(job), Some(schedule)) = (self.tasks[at].job, &self.schedule) {
                        schedule
                            .lock()
                            .unwrap()
                            .rebase(job, &crate::schedule::stamp);
                    }
                    let asked = self.tasks[at].asked.take();
                    let answer = reply.as_ref().map(|r| r.content.clone());
                    if let Some(reply) = reply {
                        self.tasks[at].last_asked = asked.clone().unwrap_or_default();
                        self.tasks[at].answer = reply.content.clone();
                        self.understand(task, asked, reply.content.clone());
                        self.tasks[at].thread.push(reply);
                    }
                    // A branch that has ended is merged into main.
                    if self.tasks[at].id != MAIN {
                        let (branch, title) = (self.tasks[at].id, self.tasks[at].title.clone());
                        tagged.extend(self.merge(branch, title, answer));
                    }
                }
                // A file sent over Telegram has been saved: main is told.
                Update::Heard(text) => {
                    tagged.extend(self.say(text));
                }
                Update::BranchAsked { title, task } => {
                    tagged.extend(self.branch(&title, task));
                }
                Update::Understood {
                    question,
                    answer,
                    digest,
                } => {
                    self.reading = self.reading.saturating_sub(1);
                    if let Some(kind) = self.remember(at, &question, &answer, digest.as_ref()) {
                        tagged.push(Tagged { task, kind });
                    }
                }
                Update::SkillsUsed(names) => {
                    if let Some(at) = at {
                        self.tasks[at].skills = names;
                    }
                }
                Update::Tidied => self.tidying = false,
                Update::Learned => self.learning = self.learning.saturating_sub(1),
                Update::ChangeAsked { problem, change } => {
                    tagged.extend(self.start_self_update(problem, change));
                }
                Update::ChangeEnded(adopted) => {
                    self.updating = false;
                    self.restart = adopted;
                }
                Update::Checked(checked) => {
                    if let Some(at) = at {
                        self.tasks[at].criteria = checked;
                    }
                }
                // The run's own checks count for the lessons it used and
                // wrote, without waiting for the user to rate it.
                Update::Outcome(outcome) => {
                    let (Some(at), Some(kb)) = (at, &self.knowledge) else {
                        continue;
                    };
                    let (passed, said) = match outcome {
                        agent::Outcome::Verified(check) => {
                            (true, format!("this run's last check passed (`{check}`)"))
                        }
                        agent::Outcome::Failed(why) => (false, why),
                        agent::Outcome::Unknown => continue,
                    };
                    let given = self.tasks[at].given.lock().unwrap().clone();
                    let judged = {
                        let mut kb = kb.lock().unwrap();
                        kb.task_went(self.tasks[at].rules, passed);
                        kb.judge_outcome(self.tasks[at].id, &given, passed)
                    };
                    for line in judged.confirmed {
                        tagged.push(Tagged {
                            task,
                            kind: EventKind::Alert {
                                severity: Severity::Success,
                                text: format!("Confirmed, because {said}: {line}"),
                            },
                        });
                    }
                    for line in judged.retired {
                        tagged.push(Tagged {
                            task,
                            kind: EventKind::Alert {
                                severity: Severity::Warn,
                                text: format!(
                                    "Retired, and not to be used again, because {said} and \
                                     nothing has gone its way: {line}"
                                ),
                            },
                        });
                    }
                }
                Update::Reread { id, digest } => {
                    if let Some(kb) = &self.knowledge {
                        let mut kb = kb.lock().unwrap();
                        kb.redigest(id, &digest);
                        let _ = kb.commit();
                    }
                }
            }
        }
        // What came in over Telegram, and what goes back out over it.
        if !announced_before {
            tagged.extend(self.start_telegram());
        }
        tagged.extend(self.telegram_heard());
        self.telegram_tell(&tagged);
        self.telegram_typing();
        // Whatever the limits on what is learned have just changed is said,
        // so the learning can be seen.
        if !tagged.is_empty()
            && let Some(kb) = &self.knowledge
        {
            for text in kb.lock().unwrap().take_tuning_notes() {
                tagged.push(Tagged::system(EventKind::Alert {
                    severity: Severity::Info,
                    text,
                }));
            }
        }
        // A slot that came free goes to the task that has waited longest.
        // Main takes what was said to it while it was busy as soon as it
        // is free, whatever the branches are doing.
        loop {
            let free = self.running() < MAX_RUNNING;
            let Some(at) = self
                .tasks
                .iter()
                .position(|t| t.queued.is_some() && !t.running && (t.id == MAIN || free))
            else {
                break;
            };
            let text = self.tasks[at].queued.take().unwrap_or_default();
            let id = self.tasks[at].id;
            tagged.extend(tag(id, self.launch(at, text)));
        }
        // Then to a scheduled job that is due.
        if announced_before {
            tagged.extend(self.start_due_job(crate::learn::now()));
            // And with nothing to do, memory is gone over.
            self.tidy_when_idle();
        }
        tagged
    }

    pub fn handle(&mut self, cmd: Command) -> Vec<Tagged> {
        match cmd {
            Command::UserMessage(text) => self.say(text),
            Command::Reply { task, text } if task == MAIN => self.say(text),
            Command::Reply { task, text } => {
                let Some(at) = self.tasks.iter().position(|t| t.id == task) else {
                    // Its conversation is no longer held; it starts anew.
                    return self.handle(Command::UserMessage(text));
                };
                if self.tasks[at].running || self.tasks[at].queued.is_some() {
                    return tag(
                        task,
                        vec![EventKind::Alert {
                            severity: Severity::Warn,
                            text: "This branch is still being worked on; reply in it when it \
                                   has answered, or say it in main (n)."
                                .into(),
                        }],
                    );
                }
                // What the user writes after a result is a response to it:
                // it may show the criteria judged the result wrongly.
                let mut events = tag(
                    task,
                    self.weigh_criteria(at, crate::goal::Said::Replied(text.clone())),
                );
                self.tasks[at].criteria.clear();
                events.extend(self.begin(task, text));
                events
            }
            Command::Feedback { good, task } => self.feedback(good, task),
            // The user's verdict on a tool call goes to whichever task is
            // waiting for it; the others ignore a number that is not theirs.
            Command::Decide { id, approve } => {
                for task in &self.tasks {
                    if let Some(decisions) = &task.decisions {
                        let _ = decisions.send((id, approve));
                    }
                }
                Vec::new()
            }
            Command::Stop { task } => {
                let Some(at) = self.tasks.iter().position(|t| t.id == task) else {
                    return Vec::new();
                };
                let t = &mut self.tasks[at];
                if t.queued.take().is_some() {
                    return tag(
                        task,
                        vec![EventKind::TaskUpdate {
                            state: TaskState::Stopped,
                            note: "stopped by you before it started".into(),
                        }],
                    );
                }
                if !t.running {
                    return Vec::new();
                }
                // The loop sees the flag at its next step; a call waiting
                // for approval is answered no, and so is a secret it was
                // waiting for.
                t.stop.store(true, Ordering::Relaxed);
                t.decisions = None;
                self.secret_waiters
                    .lock()
                    .unwrap()
                    .retain(|(waiting, _, _)| *waiting != task);
                tag(
                    task,
                    vec![EventKind::Alert {
                        severity: Severity::Warn,
                        text: "Stopping: the step in progress is the last.".into(),
                    }],
                )
            }
            Command::Secret { name, value } => self.take_secret(&name, value),
            Command::Restart => Vec::new(),
        }
    }

    /// Takes what the user said, in main. Said while main is still
    /// answering, it is kept and taken up, together with anything else said
    /// meanwhile, as soon as main is free.
    fn say(&mut self, text: String) -> Vec<Tagged> {
        // Main is opened by the first thing ever said; after that it is
        // always there (a restart finds it in the timeline).
        let mut opened = Vec::new();
        if !self.tasks.iter().any(|t| t.id == MAIN) {
            opened = tag(
                MAIN,
                vec![EventKind::TaskOpened {
                    title: "main".into(),
                    origin: "you".into(),
                }],
            );
        }
        let at = self.slot(MAIN);
        if self.tasks[at].running || self.tasks[at].queued.is_some() {
            let waiting = match self.tasks[at].queued.take() {
                Some(earlier) => format!("{earlier}\n\n{text}"),
                None => text.clone(),
            };
            self.tasks[at].queued = Some(waiting);
            opened.extend(tag(MAIN, vec![EventKind::Message { agent: None, text }]));
            return opened;
        }
        // What the user says after an answer is a response to it: it may
        // show the criteria judged that answer wrongly.
        let mut events = opened;
        events.extend(tag(
            MAIN,
            self.weigh_criteria(at, crate::goal::Said::Replied(text.clone())),
        ));
        self.tasks[at].criteria.clear();
        events.extend(self.begin(MAIN, text));
        events
    }

    /// Takes a secret the user typed into its window (or their refusal to
    /// give one). The value goes to the store and nowhere else: it is in no
    /// event, and whoever was waiting is told only that it is there.
    fn take_secret(&mut self, name: &str, value: Option<crate::secrets::Hidden>) -> Vec<Tagged> {
        let stored = match value {
            Some(value) => crate::secrets::store()
                .lock()
                .unwrap()
                .set(name, &value.0)
                .map(|()| true),
            None => Ok(false),
        };
        let mut waiting = Vec::new();
        self.secret_waiters
            .lock()
            .unwrap()
            .retain(|(task, asked, tx)| {
                if asked == name {
                    let _ = tx.send(matches!(stored, Ok(true)));
                    waiting.push(*task);
                    false
                } else {
                    true
                }
            });
        let answered = EventKind::SecretAnswered {
            name: name.to_string(),
            stored: matches!(stored, Ok(true)),
        };
        let said = EventKind::Alert {
            severity: match &stored {
                Ok(true) => Severity::Success,
                Ok(false) => Severity::Info,
                Err(_) => Severity::Error,
            },
            text: match &stored {
                Ok(true) => format!(
                    "Secret {name} is stored. Its value is not shown anywhere and is never \
                     sent to a model; a command that uses it asks you first."
                ),
                Ok(false) => format!("Secret {name} was not given."),
                Err(why) => format!("Secret {name} could not be stored: {why}"),
            },
        };
        let mut events = Vec::new();
        if waiting.is_empty() {
            events.push(Tagged::system(answered));
            events.push(Tagged::system(said));
        } else {
            for task in waiting {
                events.extend(tag(task, vec![answered.clone(), said.clone()]));
            }
        }
        events.extend(self.telegram_follow(name));
        events
    }

    /// Whether something outside keeps this harness needed with no screen
    /// attached: a connected bot can bring a message at any time.
    pub fn connected(&self) -> bool {
        self.telegram.is_some()
    }

    /// How the Telegram bot stands, in a line, for `conscious status`. The
    /// pairing code is in it while nobody is paired, so it can always be
    /// found again.
    pub fn telegram_state(&self) -> String {
        match &self.telegram {
            None => String::new(),
            Some(bridge) if bridge.pairing.owner.is_some() => {
                "connected and paired with your account".into()
            }
            Some(bridge) if bridge.code.is_empty() => {
                "connected, pairing locked after too many wrong codes: run `conscious stop`, \
                 then `conscious`, for a new code"
                    .into()
            }
            Some(bridge) => format!(
                "connected, not paired: send your bot the message /pair {}",
                bridge.code
            ),
        }
    }

    /// Connects the Telegram bot if the user has given its token.
    fn start_telegram(&mut self) -> Vec<Tagged> {
        let Some(token) = crate::secrets::get(crate::telegram::TOKEN) else {
            return Vec::new();
        };
        let path = crate::telegram::pairing_path();
        self.connect_telegram(Arc::new(crate::telegram::Api::new(token)), path)
    }

    /// When the secret just entered is the bot's token, (re)connects it.
    fn telegram_follow(&mut self, name: &str) -> Vec<Tagged> {
        if name != crate::telegram::TOKEN {
            return Vec::new();
        }
        self.telegram = None;
        self.start_telegram()
    }

    /// Starts listening to `bot` and says how to pair with it.
    fn connect_telegram(
        &mut self,
        bot: Arc<dyn crate::telegram::Bot>,
        path: Option<std::path::PathBuf>,
    ) -> Vec<Tagged> {
        let pairing: crate::telegram::Pairing = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        let code = crate::telegram::new_code();
        let alive = Arc::new(AtomicBool::new(true));
        let (heard_tx, heard) = mpsc::channel();
        let (out, out_rx) = mpsc::channel::<Out2>();
        // Reading: one long request after another.
        let files = bot.clone();
        let (reader, living, report) = (bot.clone(), alive.clone(), self.out(None));
        let mut after = pairing.after;
        thread::spawn(move || {
            let mut said = String::new();
            while living.load(Ordering::Relaxed) {
                match reader.updates(after, crate::telegram::WAIT_SECS) {
                    Ok((messages, last)) => {
                        after = after.max(last);
                        said.clear();
                        for message in messages {
                            if heard_tx.send(message).is_err() {
                                return;
                            }
                        }
                    }
                    Err(why) => {
                        // Said once, not every few seconds.
                        if why != said {
                            report.send(Update::Event(EventKind::Alert {
                                severity: Severity::Warn,
                                text: format!("Telegram: {why}"),
                            }));
                            said = why.clone();
                        }
                        // A token Telegram refuses will not start working.
                        if why.contains("does not accept the bot token") {
                            return;
                        }
                        thread::sleep(Duration::from_secs(10));
                    }
                }
            }
        });
        // Sending.
        let (sender, report) = (bot.clone(), self.out(None));
        thread::spawn(move || {
            // Where each call's question was sent: (chat, message).
            let mut asked: HashMap<u32, (i64, i64)> = HashMap::new();
            for item in out_rx {
                let sent = match item {
                    Out2::Say(chat, text) => sender.send(chat, &text),
                    Out2::Ask { chat, tool, text } => {
                        let buttons = [
                            ("Approve".to_string(), crate::telegram::decision(tool, true)),
                            ("Deny".to_string(), crate::telegram::decision(tool, false)),
                        ];
                        sender.ask(chat, &text, &buttons).map(|message| {
                            asked.insert(tool, (chat, message));
                        })
                    }
                    Out2::Settle { tool, text } => match asked.remove(&tool) {
                        Some((chat, message)) => sender.settle(chat, message, &text),
                        None => Ok(()),
                    },
                    Out2::Pressed { button, note } => sender.pressed(&button, &note),
                    // Courtesies: if one cannot be sent it is not worth a
                    // warning, least of all one every few seconds.
                    Out2::Seen { chat, message } => {
                        let _ = sender.seen(chat, message);
                        Ok(())
                    }
                    Out2::Typing(chat) => {
                        let _ = sender.typing(chat);
                        Ok(())
                    }
                };
                if let Err(why) = sent {
                    report.send(Update::Event(EventKind::Alert {
                        severity: Severity::Warn,
                        text: format!("Telegram: a message could not be sent: {why}"),
                    }));
                }
            }
        });
        let unpaired = pairing.owner.is_none();
        let text = match pairing.owner {
            Some(_) => "Telegram: the bot is connected and paired. What you send it is said \
                        to main, and main's replies are sent back to you there."
                .to_string(),
            None => format!(
                "Telegram: the bot is connected but answers to nobody yet. To make it \
                 yours, send it this message from your Telegram account: /pair {code}"
            ),
        };
        self.telegram = Some(Bridge {
            pairing,
            path,
            code,
            heard,
            out,
            bot: files,
            pending: HashMap::new(),
            reminded: None,
            typed: None,
            wrong: 0,
            alive,
        });
        vec![Tagged::system(EventKind::Alert {
            // Something for the user to do is shown in the conversation;
            // a plain "connected" is not.
            severity: if unpaired {
                Severity::Warn
            } else {
                Severity::Info
            },
            text,
        })]
    }

    /// Takes what the bot has heard: pairs, says it to main, decides a
    /// call the owner pressed a button on, or fetches a file they sent.
    fn telegram_heard(&mut self) -> Vec<Tagged> {
        use crate::telegram::Heard;
        let Some(bridge) = &mut self.telegram else {
            return Vec::new();
        };
        // First what each message is, by the rules; then acting on it,
        // which needs the rest of the feed.
        let messages: Vec<_> = bridge.heard.try_iter().collect();
        let mut heard = Vec::new();
        let mut locked = false;
        for message in messages {
            bridge.pairing.after = bridge.pairing.after.max(message.update);
            let what = crate::telegram::hear(&bridge.pairing, &bridge.code, &message);
            if let Heard::Pair(owner) = &what {
                bridge.pairing.owner = Some(*owner);
            }
            // A wrong code counts. After a few the code is thrown away:
            // six digits could otherwise simply be tried one by one.
            if bridge.pairing.owner.is_none()
                && !bridge.code.is_empty()
                && crate::telegram::tries_to_pair(&message)
            {
                bridge.wrong += 1;
                if bridge.wrong >= crate::telegram::PAIR_TRIES {
                    bridge.code.clear();
                    locked = true;
                }
            }
            heard.push((message.chat, message.message, what));
        }
        if !heard.is_empty() {
            bridge.save();
        }
        let mut events = Vec::new();
        if locked {
            events.push(Tagged::system(EventKind::Alert {
                severity: Severity::Warn,
                text: format!(
                    "Telegram: {} wrong pairing codes were sent to your bot, so pairing is \
                     locked and the code no longer works. If that was not you, someone is \
                     guessing. For a new code run `conscious stop`, then `conscious`.",
                    crate::telegram::PAIR_TRIES
                ),
            }));
        }
        for (chat, message, what) in heard {
            let Some(bridge) = &mut self.telegram else {
                break;
            };
            // What the owner sent is marked as seen straight away, before
            // any work on it starts.
            if matches!(what, Heard::Say(_) | Heard::File { .. }) && message != 0 {
                let _ = bridge.out.send(Out2::Seen { chat, message });
                let _ = bridge.out.send(Out2::Typing(chat));
                bridge.typed = Some(Instant::now());
            }
            match what {
                Heard::Nothing => {}
                Heard::Reply(text) => {
                    // To the owner, always. To someone writing to a bot
                    // that is nobody's yet, at most once a minute, and not
                    // at all once pairing is locked: a stranger must not be
                    // able to make it send without end.
                    let unpaired = bridge.pairing.owner.is_none();
                    let due = bridge
                        .reminded
                        .is_none_or(|at| at.elapsed() >= Duration::from_secs(60));
                    if !unpaired || (due && !bridge.code.is_empty()) {
                        let _ = bridge.out.send(Out2::Say(chat, text));
                    }
                    // Most likely it is the user, looking for the code: it
                    // is said again where they will see it.
                    if unpaired && due && !bridge.code.is_empty() {
                        bridge.reminded = Some(Instant::now());
                        events.push(Tagged::system(EventKind::Alert {
                            severity: Severity::Warn,
                            text: format!(
                                "Telegram: someone wrote to your bot, which is not paired \
                                 yet. If that was you, send it this message: /pair {}",
                                bridge.code
                            ),
                        }));
                    }
                }
                Heard::Pair(_) => {
                    let _ = bridge.out.send(Out2::Say(
                        chat,
                        "Paired. What you send here is said to conscious's main conversation, \
                         and its replies come back here. A call that needs your approval comes \
                         with Approve and Deny buttons."
                            .into(),
                    ));
                    events.push(Tagged::system(EventKind::Alert {
                        severity: Severity::Success,
                        text: "Telegram: paired with the account that sent the code. Only \
                               that account is heard from now on."
                            .into(),
                    }));
                }
                Heard::Say(text) => events.extend(self.say(text)),
                Heard::Decide {
                    tool,
                    approve,
                    button,
                } => {
                    // Only a call that was put to them, and is still open.
                    let open = bridge.pending.contains_key(&tool);
                    let _ = bridge.out.send(Out2::Pressed {
                        button,
                        note: match (open, approve) {
                            (false, _) => "This is no longer waiting.".into(),
                            (true, true) => "Approved.".into(),
                            (true, false) => "Denied.".into(),
                        },
                    });
                    if open {
                        events.push(Tagged::system(EventKind::Alert {
                            severity: Severity::Info,
                            text: format!(
                                "Telegram: call #{tool} was {} from your Telegram account.",
                                if approve { "approved" } else { "denied" }
                            ),
                        }));
                        events.extend(self.handle(Command::Decide { id: tool, approve }));
                    }
                }
                Heard::File { media, caption } => {
                    let root = self.guard.as_ref().map(|g| g.root().to_path_buf());
                    let (bot, out, heard) =
                        (bridge.bot.clone(), bridge.out.clone(), self.out(None));
                    // Fetched on a thread of its own: a video takes a while.
                    thread::spawn(move || {
                        let saved = root
                            .ok_or_else(|| {
                                "conscious has no folder to keep files in this session".to_string()
                            })
                            .and_then(|root| {
                                if media.size > crate::telegram::MAX_FILE {
                                    return Err(
                                        "it is larger than the 20 MB a bot may fetch".to_string()
                                    );
                                }
                                let (bytes, path) = bot.fetch(&media.file)?;
                                let folder = root.join(TELEGRAM_INBOX);
                                std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
                                let name =
                                    crate::telegram::file_name(&media, &path, crate::learn::now());
                                std::fs::write(folder.join(&name), &bytes)
                                    .map_err(|e| e.to_string())?;
                                Ok((name, bytes.len()))
                            });
                        match saved {
                            Ok((name, size)) => heard.send(Update::Heard(format!(
                                "[Sent from Telegram: a {}, saved as {TELEGRAM_INBOX}/{name} \
                                 ({:.1} MB). You are given text only, so you cannot see or hear \
                                 it; work on the file with your tools if that is needed, and \
                                 say so plainly if what is asked needs it to be seen or heard.]{}",
                                media.kind,
                                size as f64 / 1_048_576.0,
                                if caption.is_empty() {
                                    String::new()
                                } else {
                                    format!("\n{caption}")
                                }
                            ))),
                            Err(why) => {
                                let _ = out.send(Out2::Say(
                                    chat,
                                    format!("That {} could not be taken: {why}.", media.kind),
                                ));
                            }
                        }
                    });
                }
            }
        }
        events
    }

    /// Sends the owner what main said and what a branch brought back, puts
    /// a call that needs approval to them with buttons, and says when
    /// something can only be answered at the screen.
    fn telegram_tell(&mut self, events: &[Tagged]) {
        let Some(bridge) = &mut self.telegram else {
            return;
        };
        let Some(owner) = bridge.pairing.owner else {
            return;
        };
        // Anything secret is taken out before it leaves the machine.
        let clean = |text: &str| crate::guard::redact(text);
        // Replies are written for the screen: long, and in markdown. For
        // the phone each is made plain and as short as it can be, by the
        // cheapest model, off this thread.
        let brief = |text: String, out: &Sender<Out2>, router: &Arc<Router>| {
            let (out, router) = (out.clone(), router.clone());
            thread::spawn(move || {
                let _ = out.send(Out2::Say(owner, for_phone(&router, &text)));
            });
        };
        for event in events {
            let item = match &event.kind {
                EventKind::Message {
                    agent: Some(_),
                    text,
                } if event.task == Some(MAIN) => {
                    brief(clean(text), &bridge.out, &self.router);
                    continue;
                }
                EventKind::Merged {
                    branch,
                    title,
                    result,
                    ok,
                } => {
                    if *ok {
                        brief(
                            clean(&format!(
                                "Branch #{branch} ({title}) has ended. Its result:\n\n{result}"
                            )),
                            &bridge.out,
                            &self.router,
                        );
                        continue;
                    }
                    Out2::Say(
                        owner,
                        format!("Branch #{branch} ({title}) ended without an answer."),
                    )
                }
                EventKind::ToolRequested {
                    id,
                    tool,
                    args,
                    risk,
                    needs_approval: true,
                    policy,
                    ..
                } => {
                    let what: String = args.chars().take(800).collect();
                    let text = clean(&format!(
                        "Approval needed · risk {}\n\n{tool}: {what}\n\nWhy it asks: {policy}",
                        risk.label().to_lowercase()
                    ));
                    // What changes the agent's own limits is decided at
                    // the screen, where all of it can be read.
                    if crate::telegram::SCREEN_ONLY.contains(&tool.as_str()) {
                        Out2::Say(
                            owner,
                            format!("{text}\n\nThis one can only be decided at the screen."),
                        )
                    } else {
                        bridge.pending.insert(*id, text.clone());
                        Out2::Ask {
                            chat: owner,
                            tool: *id,
                            text,
                        }
                    }
                }
                EventKind::ToolDecision { id, approved, by } => {
                    let Some(text) = bridge.pending.remove(id) else {
                        continue;
                    };
                    Out2::Settle {
                        tool: *id,
                        text: format!(
                            "{text}\n\n{} (by {by}).",
                            if *approved { "Approved" } else { "Denied" }
                        ),
                    }
                }
                // What went wrong is said on the phone too: a task that
                // stopped would otherwise just fall silent there. (Not the
                // bridge's own trouble with Telegram, which could not be
                // sent anyway and would only repeat itself.)
                EventKind::Alert {
                    severity: Severity::Error,
                    text,
                } if !text.starts_with("Telegram:") => {
                    Out2::Say(owner, crate::telegram::plain(&clean(text)))
                }
                EventKind::SecretAsked { name, .. } => Out2::Say(
                    owner,
                    format!(
                        "Waiting for you to type the secret {name} at the screen. Do not \
                         send it here."
                    ),
                ),
                _ => continue,
            };
            let _ = bridge.out.send(item);
        }
    }

    /// Keeps "typing…" showing in the owner's chat for as long as anything
    /// is being worked on.
    fn telegram_typing(&mut self) {
        let busy = self.busy();
        let Some(bridge) = &mut self.telegram else {
            return;
        };
        let Some(owner) = bridge.pairing.owner else {
            return;
        };
        let due = bridge
            .typed
            .is_none_or(|at| at.elapsed() >= crate::telegram::TYPING_EVERY);
        if busy && due {
            bridge.typed = Some(Instant::now());
            let _ = bridge.out.send(Out2::Typing(owner));
        }
    }

    /// Opens a branch off main for one piece of work. It starts at once, or
    /// waits its turn when as many branches as may run at once are running.
    fn branch(&mut self, title: &str, text: String) -> Vec<Tagged> {
        let (id, mut events) = self.open(title, "main");
        events.extend(tag(
            MAIN,
            vec![EventKind::Alert {
                severity: Severity::Info,
                text: format!("Opened branch #{id}: {}", title_of(title)),
            }],
        ));
        events.extend(self.begin(id, text));
        events
    }

    /// Brings an ended branch's result into main: main's conversation holds
    /// it from now on, and the timeline shows the merge. `None` is a branch
    /// that ended with no answer.
    fn merge(&mut self, branch: u32, title: String, result: Option<String>) -> Vec<Tagged> {
        let at = self.slot(MAIN);
        push_user(
            &mut self.tasks[at].thread,
            merge_note(branch, &title, result.as_deref()),
        );
        tag(
            MAIN,
            vec![EventKind::Merged {
                branch,
                title,
                ok: result.is_some(),
                result: result.unwrap_or_default(),
            }],
        )
    }

    fn set_draft(&mut self, at: usize, text: String) {
        if self.tasks[at].draft != text {
            self.tasks[at].draft = text;
            self.tasks[at].draft_sent = false;
        }
    }

    /// The replies being written right now that have grown (or ended)
    /// since this was last asked: (task, the text so far). Empty text means
    /// nothing is being written any more. These are for the screen only and
    /// are never events: the finished reply is the event. A request for a
    /// tool that the model writes out as text is not shown.
    pub fn drafts(&mut self) -> Vec<(u32, String)> {
        self.tasks
            .iter_mut()
            .filter(|t| !t.draft_sent)
            .map(|t| {
                t.draft_sent = true;
                let shown = t.draft.split("<tool_call").next().unwrap_or_default();
                (t.id, shown.trim_end().to_string())
            })
            .collect()
    }

    /// Has what the user says shown before the work on it starts: a
    /// message is announced by `handle` and its context is assembled (the
    /// knowledge lookup, which takes a moment) by the next `poll`.
    pub fn with_echo_first(mut self, on: bool) -> Self {
        self.echo_first = on;
        self
    }

    /// Takes a message for a task: starts on it, or queues it when every
    /// slot is taken.
    fn begin(&mut self, id: u32, text: String) -> Vec<Tagged> {
        let at = self.slot(id);
        let mut events = vec![EventKind::Message {
            agent: None,
            text: text.clone(),
        }];
        if id != MAIN && self.running() >= MAX_RUNNING {
            self.tasks[at].queued = Some(text);
            events.push(EventKind::TaskUpdate {
                state: TaskState::Queued,
                note: format!(
                    "waiting its turn: {MAX_RUNNING} branches are running, the most at once"
                ),
            });
        } else if self.echo_first {
            // Started by the next poll, so that the message is on screen
            // before the lookup for it is made.
            self.tasks[at].queued = Some(text);
            events.push(EventKind::TaskUpdate {
                state: TaskState::Working,
                note: "looking up what it knows about this".into(),
            });
        } else {
            events.extend(self.launch(at, text));
        }
        tag(id, events)
    }

    /// Assembles the context for a task's message and starts a worker on it.
    fn launch(&mut self, at: usize, text: String) -> Vec<EventKind> {
        let id = self.tasks[at].id;
        self.tasks[at].running = true;
        self.tasks[at].stop = Arc::default();
        self.tasks[at].asked = Some(text.clone());
        push_user(&mut self.tasks[at].thread, text.clone());
        let dropped = self.trim_thread(at);
        // Graph lookup: what the store holds that bears on this message.
        let in_window: Vec<NodeId> = self.tasks[at].in_window.iter().copied().collect();
        let recalled = self.knowledge.as_ref().and_then(|kb| {
            let kb = kb.lock().unwrap();
            let (retrieval, block) = kb.recall(&text, &in_window)?;
            let label = |id: NodeId| {
                kb.node(id)
                    .map(|n| (n.kind.label().to_string(), n.title.clone()))
            };
            let picks: Vec<LookupPick> = retrieval
                .hits
                .iter()
                .filter_map(|h| {
                    let node = kb.node(h.node)?;
                    Some(LookupPick {
                        memory: node.id as u32,
                        title: node.title.clone(),
                        // In words, not as a file path.
                        source: KnowledgeBase::origin(node).to_string(),
                        trust: node.provenance.trust.label().to_string(),
                        space: node.space.clone(),
                        score: h.score,
                        tokens: h.tokens,
                        hops: h.hops,
                        matched: h.matched.clone(),
                        path: h
                            .path
                            .iter()
                            .filter_map(|step| {
                                let (kind, title) = label(step.node)?;
                                Some(LookupStep {
                                    id: step.node,
                                    kind,
                                    title,
                                    relation: step.relation.clone(),
                                })
                            })
                            .collect(),
                        text: node.text.clone(),
                    })
                })
                .collect();
            Some((retrieval, block, picks))
        });
        // The lessons among what was found are this run's to answer for.
        let given: Vec<NodeId> = match (&self.knowledge, &recalled) {
            (Some(kb), Some((retrieval, _, _))) => {
                let nodes: Vec<NodeId> = retrieval.hits.iter().map(|h| h.node).collect();
                kb.lock().unwrap().given(&nodes)
            }
            _ => Vec::new(),
        };
        // The standing rules are in front of every run, so every run's
        // outcome and rating counts for them too.
        let given = match &self.knowledge {
            Some(kb) => {
                let mut kb = kb.lock().unwrap();
                let rules = kb.standing_rule_ids();
                self.tasks[at].rules = rules.len();
                let mut all = given;
                for id in kb.given(&rules) {
                    if !all.contains(&id) {
                        all.push(id);
                    }
                }
                all
            }
            None => given,
        };
        self.tasks[at].given = Arc::new(Mutex::new(given));
        self.tasks[at].judged = false;
        let mut system = match &self.guard {
            Some(guard) => agent::system_prompt(guard),
            None => SYSTEM.to_string(),
        };
        // Where it stands: the conversation that keeps going, or a branch
        // off it.
        if self.guard.is_some() {
            system.push_str("\n\n");
            system.push_str(if id == MAIN { MAIN_BRIEF } else { BRANCH_BRIEF });
        }
        let instructions = estimate_tokens(&system);
        // The project's own rules for agents, and the skills written so far.
        let project = self
            .guard
            .as_ref()
            .map(|g| agent::project_instructions(g, &text))
            .unwrap_or_default();
        let skill_list = self
            .skills
            .as_ref()
            .filter(|_| self.guard.is_some())
            .map(|s| s.lock().unwrap().index())
            .unwrap_or_default();
        for block in [&project, &skill_list] {
            if !block.is_empty() {
                system.push_str("\n\n");
                system.push_str(block);
            }
        }
        // Standing memory: who the user is and how they want things done.
        let standing = self
            .knowledge
            .as_ref()
            .map(|kb| kb.lock().unwrap().standing())
            .unwrap_or_default();
        if !standing.is_empty() {
            system.push_str("\n\n");
            system.push_str(&standing);
        }
        // The rules the agent drew for itself that have held up: how to
        // behave whatever the task. Few, and always there.
        let rules = self
            .knowledge
            .as_ref()
            .map(|kb| kb.lock().unwrap().rules())
            .unwrap_or_default();
        if !rules.is_empty() {
            system.push_str("\n\n");
            system.push_str(&rules);
        }
        let found_block = recalled
            .as_ref()
            .map(|(_, block, _)| block.clone())
            .unwrap_or_default();
        if let Some((_, block, _)) = &recalled {
            system.push_str("\n\n");
            system.push_str(block);
        }
        let req = Request {
            system,
            messages: self.tasks[at].thread.clone(),
            tools: Vec::new(),
            max_tokens: None,
        };
        let mut sections = vec![ContextSection {
            name: "system".into(),
            tokens: instructions,
            why: "harness instructions (estimated size)".into(),
        }];
        if !project.is_empty() {
            sections.push(ContextSection {
                name: "project".into(),
                tokens: estimate_tokens(&project),
                why: "this project's own instructions for agents (a long file is cut down \
                      to the parts that bear on this task)"
                    .into(),
            });
        }
        if !skill_list.is_empty() {
            sections.push(ContextSection {
                name: "skills".into(),
                tokens: estimate_tokens(&skill_list),
                why: "the list of procedures the agent wrote after earlier tasks".into(),
            });
        }
        if !standing.is_empty() {
            sections.push(ContextSection {
                name: "memory".into(),
                tokens: estimate_tokens(&standing),
                why: "what is always kept in mind: who you are and how you want things done".into(),
            });
        }
        if !rules.is_empty() {
            sections.push(ContextSection {
                name: "rules".into(),
                tokens: estimate_tokens(&rules),
                why: "how it has learned to work whatever the task: the few rules it drew \
                      itself that have held up; know-how for one kind of task comes by lookup"
                    .into(),
            });
        }
        let mut recall_event = None;
        if let Some((retrieval, _, picks)) = recalled {
            sections.push(ContextSection {
                name: "knowledge".into(),
                tokens: retrieval.tokens,
                why: KnowledgeBase::describe(&retrieval),
            });
            recall_event = Some(EventKind::KnowledgeLookup {
                agent: CORE,
                query: text.clone(),
                words: retrieval.words.clone(),
                seeds: retrieval.seeds,
                reached: retrieval.reached,
                dropped: retrieval.dropped,
                tokens: retrieval.tokens,
                limits: retrieval.limits.clone(),
                picks,
            });
        }
        let turns = self.tasks[at].thread.len();
        sections.push(ContextSection {
            name: "task".into(),
            tokens: req
                .messages
                .iter()
                .map(|m| estimate_tokens(&m.content))
                .sum(),
            why: if dropped == 0 && self.tasks[at].dropped == 0 {
                format!(
                    "this task's own conversation ({turns} message{}); nothing from other \
                     tasks is carried along",
                    if turns == 1 { "" } else { "s" }
                )
            } else {
                format!(
                    "this task's most recent turns; {} older message{} left working memory \
                     and are recalled from the knowledge graph when relevant",
                    self.tasks[at].dropped,
                    if self.tasks[at].dropped == 1 { "" } else { "s" }
                )
            },
        });
        let context = EventKind::Context {
            window: self.router.core().context_window(),
            sections,
        };
        let (router, tx) = (self.router.clone(), self.out(Some(id)));
        let stop = self.tasks[at].stop.clone();
        // What memory holds on how this user judges results: the standing
        // memories and whatever the lookup found for this message.
        let intent_notes = [standing.as_str(), found_block.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut events = vec![EventKind::TaskUpdate {
            state: TaskState::Working,
            note: "choosing a model for it".into(),
        }];
        match self.guard.clone() {
            Some(guard) => {
                // Its goal is what was asked.
                events.push(EventKind::GoalSet {
                    title: title_of(&text),
                    criteria: Vec::new(),
                    token_budget: agent::TOKEN_BUDGET,
                });
                events.push(EventKind::PlanUpdated {
                    version: 0,
                    steps: Vec::new(),
                });
                let (decide, decisions) = mpsc::channel();
                self.tasks[at].decisions = Some(decide);
                let env = TaskEnv {
                    router,
                    guard,
                    knowledge: self.knowledge.clone(),
                    skills: self.skills.clone(),
                    schedule: self.schedule.clone(),
                    tool_ids: self.tool_ids.clone(),
                    agent_ids: self.agent_ids.clone(),
                    claims: self.claims.clone(),
                    task: id,
                    secret_waiters: self.secret_waiters.clone(),
                    main: id == MAIN,
                    given: self.tasks[at].given.clone(),
                    goals: self.goals,
                    intent: intent_notes,
                };
                thread::spawn(move || run_task(&env, &text, req, &tx, &decisions, &stop));
            }
            None => {
                thread::spawn(move || run_call(&router, &text, &req, &tx));
            }
        }
        events.extend(recall_event);
        events.extend([
            context,
            EventKind::AgentStatus {
                id: CORE,
                state: AgentState::Thinking,
            },
        ]);
        events
    }

    /// Keeps a task's working memory within `HISTORY_BUDGET` by dropping
    /// its oldest turns. They are not lost: each was remembered in the
    /// knowledge graph when it happened. Returns how many were dropped.
    fn trim_thread(&mut self, at: usize) -> usize {
        let task = &mut self.tasks[at];
        let size = |h: &[Message]| -> u32 { h.iter().map(|m| estimate_tokens(&m.content)).sum() };
        let mut dropped = 0;
        while task.thread.len() > 1 && size(&task.thread) > HISTORY_BUDGET {
            let gone = task.thread.remove(0);
            dropped += 1;
            // An answer leaving the window frees its memory for recall.
            if gone.role == Role::Assistant {
                task.in_window.pop_front();
            }
        }
        task.dropped += dropped;
        dropped
    }

    /// Has a small model read the exchange and say what it is, in the
    /// background, before it is remembered. If the model cannot be reached
    /// or answers unusably, the exchange is remembered as it was said.
    fn understand(&mut self, task: Option<u32>, asked: Option<String>, answer: String) {
        let (Some(_), Some(question)) = (&self.knowledge, asked) else {
            return;
        };
        let (router, tx) = (self.router.clone(), self.out(task));
        self.reading += 1;
        thread::spawn(move || {
            let result = digest(router.cheapest().as_ref(), &question, &answer);
            let (understood, call) = match result {
                Ok((d, call)) => (d, Some(call)),
                Err(_) => (None, None),
            };
            if let Some(call) = &call {
                tx.send(Update::Event(model_call(call)));
            }
            tx.send(Update::Understood {
                question,
                answer,
                digest: understood,
            });
        });
    }

    /// Stores an exchange in the knowledge graph and reports it.
    fn remember(
        &mut self,
        at: Option<usize>,
        question: &str,
        answer: &str,
        digest: Option<&Digest>,
    ) -> Option<EventKind> {
        // An exchange a model read and found nothing in (a greeting, thanks)
        // is not filed in memory. It is still in the timeline, which keeps
        // everything.
        if digest.is_some_and(|d| d.kind == MemoryKind::Nothing) {
            return None;
        }
        let mut kb = self.knowledge.as_ref()?.lock().unwrap();
        let id = kb.remember(question, answer, digest);
        // A failed save costs this memory only on the next run.
        let _ = kb.commit();
        let node = memory_node(
            &kb,
            id,
            match digest {
                Some(d) => d.title.clone(),
                None => question.chars().take(40).collect(),
            },
            mem_kind(digest.map(|d| d.kind)),
        );
        drop(kb);
        if let Some(at) = at {
            self.tasks[at].in_window.push_back(id);
        }
        Some(EventKind::MemoryWrite { agent: CORE, node })
    }

    /// Teaches routing from the user's verdict on a task's answer: the task
    /// named, or else the one that finished most recently. What was learned
    /// is posted as a procedural memory.
    fn feedback(&mut self, good: bool, task: Option<u32>) -> Vec<Tagged> {
        let id = task.or(self.last_finished);
        let at = id.and_then(|id| self.tasks.iter().position(|t| t.id == id));
        let wrap = |kinds: Vec<EventKind>| -> Vec<Tagged> {
            kinds
                .into_iter()
                .map(|kind| Tagged { task: id, kind })
                .collect()
        };
        // The rating also counts for the skills the task relied on.
        let used = at.map(|at| std::mem::take(&mut self.tasks[at].skills));
        let retired = match (&self.skills, used) {
            (Some(skills), Some(used)) if !used.is_empty() => {
                skills.lock().unwrap().rate(&used, good)
            }
            _ => Vec::new(),
        };
        let mut retired: Vec<EventKind> = retired
            .into_iter()
            .map(|name| EventKind::Alert {
                severity: Severity::Warn,
                text: format!(
                    "Retired the skill `{name}`: the tasks it was used for keep being rated bad."
                ),
            })
            .collect();
        // And for the lessons it was given and the lessons it wrote: a good
        // rating confirms them, a bad one counts against them.
        if let (Some(at), Some(kb)) = (at, self.knowledge.clone())
            && !self.tasks[at].running
            && !self.tasks[at].judged
        {
            self.tasks[at].judged = true;
            let given = self.tasks[at].given.lock().unwrap().clone();
            // The rating is also a verdict on the success criteria.
            retired.extend(self.weigh_criteria(at, crate::goal::Said::Rated(good)));
            let mut kb = kb.lock().unwrap();
            let judged = kb.rate_lessons(self.tasks[at].id, &given, good);
            // The rating also teaches the limits on what is learned.
            kb.task_went(self.tasks[at].rules, good);
            let moved = kb.take_tuning_notes();
            drop(kb);
            for text in moved {
                retired.push(EventKind::Alert {
                    severity: Severity::Info,
                    text,
                });
            }
            for line in judged.confirmed {
                retired.push(EventKind::Alert {
                    severity: Severity::Success,
                    text: format!("Confirmed by this rating: {line}"),
                });
            }
            for line in judged.retired {
                retired.push(EventKind::Alert {
                    severity: Severity::Warn,
                    text: format!("Retired by this rating, and not to be used again: {line}"),
                });
            }
        }
        let learned = match at {
            None => self.router.feedback(good),
            Some(at) if self.tasks[at].running => Err("that task has not answered yet".to_string()),
            Some(at) if self.tasks[at].rated => Err("that task has already been rated".to_string()),
            Some(at) => match self.tasks[at].routed.clone() {
                Some((family, level)) => self.router.rate(&family, level, good),
                None => Err("that task was not routed, so there is nothing to rate".to_string()),
            },
        };
        match learned {
            Ok(learned) => {
                if let Some(at) = at {
                    self.tasks[at].rated = true;
                }
                self.lessons += 1;
                let mut events = vec![EventKind::MemoryWrite {
                    agent: CORE,
                    node: MemoryNode {
                        // Clear of the ids used by ingested memories.
                        id: 10_000 + self.lessons,
                        kind: MemKind::Procedural,
                        label: format!("routing: {}", if good { "worked" } else { "too weak" }),
                        text: String::new(),
                        about: String::new(),
                    },
                }];
                // The first line is what was learned; any further line is a
                // policy version being rolled back.
                for (i, line) in learned.iter().enumerate() {
                    events.push(EventKind::Alert {
                        severity: if i == 0 {
                            Severity::Success
                        } else {
                            Severity::Warn
                        },
                        text: if i == 0 {
                            format!("Routing learned: {line}")
                        } else {
                            line.clone()
                        },
                    });
                }
                // Let the core reconsider the policy script, off the UI thread.
                if self.router.review_due() {
                    let (router, tx) = (self.router.clone(), self.out(None));
                    thread::spawn(move || run_review(&router, &tx));
                }
                events.extend(retired);
                wrap(events)
            }
            Err(why) => {
                let mut events = vec![EventKind::Alert {
                    severity: Severity::Info,
                    text: format!("Nothing learned: {why}."),
                }];
                events.extend(retired);
                wrap(events)
            }
        }
    }

    /// Streaming speed in tokens/sec over the last second, 0 when idle.
    pub fn throughput(&mut self) -> u64 {
        let mut recent = self.recent.lock().unwrap();
        let cutoff = Instant::now() - Duration::from_secs(1);
        while recent.front().is_some_and(|(at, _)| *at < cutoff) {
            recent.pop_front();
        }
        let chars: usize = recent.iter().map(|(_, n)| n).sum();
        (chars as u64).div_ceil(4)
    }
}

fn model_call(resp: &Response) -> EventKind {
    EventKind::ModelCall {
        agent: CORE,
        model: resp.model.clone(),
        tokens_in: resp.usage.input,
        cached_in: resp.usage.cached,
        tokens_out: resp.usage.output,
        latency_ms: resp.latency.as_millis() as u64,
    }
}

/// Has the core review the policy script and reports what came of it.
fn run_review(router: &Router, tx: &Out) {
    let outcome = router.review();
    let send = |kind: EventKind| {
        tx.send(Update::Event(kind));
    };
    if let Some(call) = &outcome.call {
        send(model_call(call));
    }
    send(EventKind::Alert {
        severity: if outcome.adopted.is_some() {
            Severity::Success
        } else {
            Severity::Info
        },
        text: outcome.message,
    });
}

/// Decides which model runs a task, and shows the decision and its cost.
fn route(router: &Router, task: &str, tx: &Out) -> crate::router::Choice {
    let send = |kind: EventKind| {
        tx.send(Update::Event(kind));
    };
    let choice = router.choose(task);
    for call in &choice.calls {
        send(model_call(call));
    }
    for note in &choice.notes {
        send(EventKind::Alert {
            severity: Severity::Warn,
            text: note.clone(),
        });
    }
    if router.routes() {
        send(EventKind::Thought {
            agent: CORE,
            text: format!(
                "Routing to {}: {}",
                router.candidates()[choice.index].spec,
                choice.reason
            ),
        });
    }
    choice
}

/// What a self-update needs.
struct UpdateEnv {
    router: Arc<Router>,
    tool_ids: Arc<AtomicU32>,
    agent_ids: Arc<AtomicU32>,
    task: u32,
    /// The real source, and where copies, backups and history are kept.
    source: std::path::PathBuf,
    home: std::path::PathBuf,
    problem: String,
    change: String,
}

/// Times the model is sent back to fix what the gate found.
const GATE_ROUNDS: usize = 3;

/// A self-update from start to finish: a copy of the source, the model's
/// change in it, the gate, the user's approval where protected files are
/// touched, adoption and the build of the real binary. Returns why a
/// restart is due if the change was adopted. Every step is said in the
/// task's own tab, and the attempt is recorded whatever comes of it.
fn run_self_update(
    env: &UpdateEnv,
    tx: &Out,
    decisions: &Receiver<(u32, bool)>,
    stop: &AtomicBool,
) -> Option<String> {
    use crate::selfupdate::{self, Attempt, History, Runner, Sandbox};
    let send = |kind: EventKind| tx.send(Update::Event(kind));
    let working = |note: String| {
        send(EventKind::TaskUpdate {
            state: TaskState::Working,
            note,
        })
    };
    let approve = |id: u32| loop {
        match decisions.recv() {
            Ok((decided, approve)) if decided == id => return approve,
            Ok(_) => continue,
            Err(_) => return false,
        }
    };
    let mut history = History::load(&env.home);
    let id = history.next_id();
    let work = env.home.join("work");
    let record =
        |history: &mut History, changes: Vec<selfupdate::Change>, outcome: &str, why: &str| {
            history.record(Attempt {
                id,
                at: crate::learn::now(),
                problem: env.problem.clone(),
                changes,
                outcome: outcome.into(),
                why: why.into(),
            });
        };
    // Ends the task without adopting anything, saying why.
    let give_up = |history: &mut History,
                   changes: Vec<selfupdate::Change>,
                   outcome: &str,
                   state: TaskState,
                   why: String| {
        record(history, changes, outcome, &why);
        send(EventKind::Message {
            agent: Some(CORE),
            text: format!(
                "The harness's code was **not** changed (self-update #{id}, {outcome}).\n\n{why}"
            ),
        });
        send(EventKind::TaskUpdate {
            state,
            note: format!("not adopted: {}", why.lines().next().unwrap_or_default()),
        });
        None
    };

    if let Err(why) = selfupdate::prepare(&env.source, &work) {
        return give_up(
            &mut history,
            Vec::new(),
            "rejected",
            TaskState::Failed,
            format!("its source could not be copied: {why}"),
        );
    }
    let guard = match crate::guard::Guard::for_scratch(&work) {
        Ok(guard) if guard.can_sandbox() => guard,
        Ok(_) => {
            return give_up(
                &mut history,
                Vec::new(),
                "rejected",
                TaskState::Failed,
                "The sandbox is not available on this machine, and code the harness wrote \
                 itself is never built or tested outside one. Install bubblewrap to allow \
                 self-updates."
                    .into(),
            );
        }
        Err(why) => {
            return give_up(&mut history, Vec::new(), "rejected", TaskState::Failed, why);
        }
    };

    // The model makes the change in the copy, with the ordinary tools.
    let brief = selfupdate::brief(&env.problem, &env.change);
    let choice = route(&env.router, &brief, tx);
    let hooks = agent::Hooks {
        emit: &send,
        approve: &approve,
        on_text: &|text| tx.send(Update::Streamed(text.to_string())),
        ids: &env.tool_ids,
        agents: &env.agent_ids,
        // Its own notes on this work do not belong in the user's memory.
        knowledge: None,
        skills: None,
        schedule: None,
        files: None,
        task: env.task,
        given: &|_| {},
        self_update: &|_, _| "a self-update cannot ask for another".to_string(),
        branch: None,
        secret: &|_, _| false,
        web: &crate::research::Web,
        stop,
        limits: agent::Limits::default(),
    };
    let mut messages = vec![Message::user("Make the change described above.")];
    let mut passed = None;
    let mut refused = String::new();
    for round in 1..=GATE_ROUNDS {
        working(format!(
            "writing the change in a copy of its source, with {} (round {round} of {GATE_ROUNDS})",
            env.router.candidates()[choice.index].spec
        ));
        let req = Request {
            system: format!("{}\n\n{brief}", agent::system_prompt(&guard)),
            messages: messages.clone(),
            ..Default::default()
        };
        let done = match agent::run(&env.router, &guard, &choice, req, &hooks) {
            Ok(done) => done,
            Err(e) => {
                let changes = selfupdate::changes(&env.source, &work);
                return give_up(
                    &mut history,
                    changes,
                    "rejected",
                    TaskState::Failed,
                    format!("the model could not be reached: {e}"),
                );
            }
        };
        if done.ended_early {
            let changes = selfupdate::changes(&env.source, &work);
            let state = if stop.load(Ordering::Relaxed) {
                TaskState::Stopped
            } else {
                TaskState::Failed
            };
            return give_up(&mut history, changes, "rejected", state, done.text);
        }
        send(EventKind::Thought {
            agent: CORE,
            text: done.text.chars().take(600).collect(),
        });
        // The gate, run by the harness itself.
        match selfupdate::gate(&Sandbox, &env.source, &work, &working) {
            Ok(said) => {
                passed = Some((said, done.text));
                break;
            }
            Err(why) => {
                send(EventKind::Alert {
                    severity: Severity::Warn,
                    text: format!("The gate refused the change: {why}"),
                });
                messages = done.transcript;
                messages.push(Message::user(format!(
                    "The harness ran its gate on your change and it did not pass:\n{why}\n\nFix \
                     this, then reply without a tool call."
                )));
                refused = why;
            }
        }
    }
    let changes = selfupdate::changes(&env.source, &work);
    let Some((gate_said, model_said)) = passed else {
        return give_up(
            &mut history,
            changes,
            "rejected",
            TaskState::Failed,
            format!("The gate still refused it after {GATE_ROUNDS} rounds:\n{refused}"),
        );
    };
    let listed: Vec<String> = changes
        .iter()
        .map(|c| format!("{} ({:?})", c.path, c.kind).to_lowercase())
        .collect();

    // A change to a protected file is the user's to allow, always.
    let protected = selfupdate::protected(&changes);
    if !protected.is_empty() {
        let ask = env.tool_ids.fetch_add(1, Ordering::Relaxed);
        send(EventKind::ToolRequested {
            id: ask,
            agent: CORE,
            tool: "adopt self-update".into(),
            // The change itself, so the consent is to what will be done
            // and not to a description of it.
            args: format!(
                "Adopt self-update #{id}: {}\nFiles changed: {}\nThe copy to look at: {}\n\n\
                 The change to the protected files:\n{}",
                env.change,
                listed.join(", "),
                work.display(),
                selfupdate::diff(&env.source, &work, &changes, &protected)
            ),
            risk: crate::event::Risk::High,
            needs_approval: true,
            policy: format!(
                "needs your approval: it changes protected parts of the harness ({}). The gate passed: {gate_said}.",
                protected.join(", ")
            ),
        });
        send(EventKind::TaskUpdate {
            state: TaskState::Working,
            note: format!("waiting for you: it changes {}", protected.join(", ")),
        });
        let approved = approve(ask);
        send(EventKind::ToolDecision {
            id: ask,
            approved,
            by: "you".into(),
        });
        if !approved {
            return give_up(
                &mut history,
                changes,
                "declined",
                TaskState::Stopped,
                format!(
                    "It passed the gate, but it changes protected parts of the harness ({}) \
                     and you declined it.",
                    protected.join(", ")
                ),
            );
        }
    }

    // Adoption: the change goes into the real source and the real binary
    // is built from it. If that build fails, everything is put back.
    working("adopting: copying the change into the real source and building it".into());
    let backup = selfupdate::backup_dir(&env.home, id);
    if let Err(why) = selfupdate::adopt(&env.source, &work, &changes, &backup) {
        let _ = selfupdate::restore(&env.source, &backup);
        return give_up(
            &mut history,
            changes,
            "rejected",
            TaskState::Failed,
            format!("the change could not be copied into the source, which was put back: {why}"),
        );
    }
    let built = Sandbox
        .run(&env.source, selfupdate::build_command())
        .and_then(|_| selfupdate::install_binary(&env.source, &backup));
    if let Err(why) = built {
        let _ = selfupdate::restore(&env.source, &backup);
        let _ = Sandbox.run(&env.source, selfupdate::build_command());
        let why: String = why
            .lines()
            .rev()
            .take(15)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return give_up(
            &mut history,
            changes,
            "rejected",
            TaskState::Failed,
            format!("the real binary could not be built, so the source was put back:\n{why}"),
        );
    }
    record(&mut history, changes, "adopted", &gate_said);
    send(EventKind::Message {
        agent: Some(CORE),
        text: format!(
            "The harness changed its own code (self-update #{id}).\n\n**Why:** {}\n\n\
             **What changed:** {}\n\n**The gate:** {gate_said}.\n\n{model_said}\n\n\
             It restarts into the new build as soon as nothing else is running. \
             `conscious selfupdate rollback` takes this change back.",
            env.problem,
            listed.join(", ")
        ),
    });
    send(EventKind::TaskUpdate {
        state: TaskState::Done,
        note: format!(
            "adopted {} changed file{}; restarting into the new build",
            listed.len(),
            if listed.len() == 1 { "" } else { "s" }
        ),
    });
    Some(format!("self-update #{id} was adopted"))
}

/// The learner: works out, from how the user responded to a finished task,
/// what they wanted that its success criteria got wrong, and keeps it in
/// memory as a note on the user's intent, where the next task's criteria
/// are written from. Says what it learned, or that it could not tell.
#[allow(clippy::too_many_arguments)]
fn learn_what_was_meant(
    router: &Router,
    kb: Option<&Mutex<KnowledgeBase>>,
    tx: &Out,
    task: u32,
    asked: &str,
    checked: &[crate::goal::Checked],
    answer: &str,
    said: &crate::goal::Said,
) {
    use crate::goal::Said;
    use crate::knowledge::{Kept, Learned};
    let send = |severity: Severity, text: String| {
        tx.send(Update::Event(EventKind::Alert { severity, text }));
    };
    let rated = matches!(said, Said::Rated(_));
    let Ok((intent, call)) =
        crate::goal::learn_intent(router.cheapest().as_ref(), asked, checked, answer, said)
    else {
        return;
    };
    tx.send(Update::Event(model_call(&call)));
    let Some(intent) = intent else {
        // A rating alone does not always show what was wanted.
        if rated {
            send(
                Severity::Info,
                "The rating alone does not show what you wanted. Reply in this task with what \
                 was missing or not needed, and it will be learned from."
                    .into(),
            );
        }
        return;
    };
    let Some(kb) = kb else {
        return;
    };
    let (text, when) = (
        crate::guard::redact(&intent.text),
        crate::guard::redact(&intent.when),
    );
    let mut kb = kb.lock().unwrap();
    // A reply that corrects the result means the criteria did not judge it
    // the way the user does. (A rating was already counted.)
    if !rated {
        kb.criteria_judged(false);
    }
    let kept = kb.add_learned(
        task,
        Learned::Intent {
            text: &text,
            when: &when,
        },
        true,
    );
    let _ = kb.commit();
    drop(kb);
    let applies = if when.is_empty() {
        String::new()
    } else {
        format!(" (applies when: {when})")
    };
    match kept {
        Kept::New(id) => {
            tx.send(Update::Event(EventKind::MemoryWrite {
                agent: CORE,
                node: MemoryNode {
                    id: id as u32,
                    kind: MemKind::Procedural,
                    label: format!(
                        "what you want: {}",
                        text.chars().take(40).collect::<String>()
                    ),
                    text: String::new(),
                    about: String::new(),
                },
            }));
            send(
                Severity::Success,
                format!(
                    "Learned what you meant, for setting better success criteria next time: \
                     {text}{applies}"
                ),
            );
        }
        Kept::Known(_) => send(
            Severity::Info,
            format!("This confirms what was already noted about what you want: {text}"),
        ),
        Kept::Refused(why) => send(
            Severity::Info,
            format!("What it made of your response was not kept: {why}"),
        ),
    }
}

/// What a task needs from the session it runs in.
struct TaskEnv {
    router: Arc<Router>,
    guard: Arc<Guard>,
    knowledge: Option<Arc<Mutex<KnowledgeBase>>>,
    skills: Option<Arc<Mutex<Skills>>>,
    schedule: Option<Arc<Mutex<Schedule>>>,
    tool_ids: Arc<AtomicU32>,
    agent_ids: Arc<AtomicU32>,
    claims: Arc<Mutex<crate::claims::Claims>>,
    task: u32,
    secret_waiters: SecretWaiters,
    /// Whether this is the main conversation, which may open branches.
    main: bool,
    given: Arc<Mutex<Vec<NodeId>>>,
    /// Whether to set and check success criteria, and what is known about
    /// how the user judges results, to write them with.
    goals: bool,
    intent: String,
}

/// Works on a task as an agent: the model may use tools, each call checked
/// by the guardrails and, where they say so, approved by the user.
fn run_task(
    env: &TaskEnv,
    task: &str,
    mut req: Request,
    tx: &Out,
    decisions: &Receiver<(u32, bool)>,
    stop: &AtomicBool,
) {
    let (router, guard) = (env.router.as_ref(), env.guard.as_ref());
    let send = |kind: EventKind| {
        tx.send(Update::Event(kind));
    };
    let status = |state: AgentState| EventKind::AgentStatus { id: CORE, state };
    let choice = route(router, task, tx);
    // A branch is not started here: it is handed to the feed, which runs
    // it as a task of its own and merges its result back.
    let branched = std::cell::Cell::new(false);
    let open_branch = |title: &str, work: &str| -> String {
        branched.set(true);
        tx.send(Update::BranchAsked {
            title: title.to_string(),
            task: work.to_string(),
        });
        format!(
            "A branch was opened for \"{title}\". It works by itself alongside this \
             conversation and its result will be merged in here when it ends. Do not do that \
             work yourself and do not wait for it: tell the user in a line that the branch \
             was opened, and carry on."
        )
    };
    let hooks = agent::Hooks {
        emit: &send,
        // Waits for the user's verdict on this call. If the session ends
        // first, the answer is no.
        approve: &|id| loop {
            match decisions.recv() {
                Ok((decided, approve)) if decided == id => return approve,
                Ok(_) => continue,
                Err(_) => return false,
            }
        },
        on_text: &|text| {
            tx.send(Update::Streamed(text.to_string()));
        },
        ids: &env.tool_ids,
        agents: &env.agent_ids,
        knowledge: env.knowledge.as_deref(),
        skills: env.skills.as_deref(),
        schedule: env.schedule.as_deref(),
        files: Some(&env.claims),
        task: env.task,
        given: &|nodes| {
            if let Some(kb) = &env.knowledge {
                let lessons = kb.lock().unwrap().given(nodes);
                let mut given = env.given.lock().unwrap();
                for id in lessons {
                    if !given.contains(&id) {
                        given.push(id);
                    }
                }
            }
        },
        // A change to the harness itself is not made here: it is handed to
        // the feed, which runs it as a task of its own.
        self_update: &|problem, change| {
            tx.send(Update::ChangeAsked {
                problem: problem.to_string(),
                change: change.to_string(),
            });
            "A self-update has been asked for. It runs as a task of its own, in a copy of the \
             harness's source, and is adopted only if the harness's own formatting, lint and \
             tests pass; the harness then restarts into the new build. This task cannot use \
             the change. Finish now: tell the user what could not be done, and that a change \
             to the harness was started for it."
                .to_string()
        },
        // The user is asked on the screen; this waits for the answer. The
        // value itself goes to the store, never through here.
        secret: &|name, why| {
            let (answer, answered) = mpsc::channel();
            env.secret_waiters
                .lock()
                .unwrap()
                .push((env.task, name.to_string(), answer));
            send(EventKind::SecretAsked {
                name: name.to_string(),
                why: why.to_string(),
            });
            answered.recv().unwrap_or(false)
        },
        branch: env.main.then_some(&open_branch as &agent::OpenBranch<'_>),
        web: &crate::research::Web,
        stop,
        limits: agent::Limits::default(),
    };
    // Before the work: how its success will be judged. The model that
    // does the work is given the criteria, so it works towards them.
    let mut criteria = Vec::new();
    if env.goals {
        send(EventKind::TaskUpdate {
            state: TaskState::Working,
            note: "setting out how success will be judged".into(),
        });
        if let Ok((written, call)) =
            crate::goal::write(router.cheapest().as_ref(), task, &env.intent)
        {
            send(model_call(&call));
            if !written.is_empty() {
                send(EventKind::GoalSet {
                    title: title_of(task),
                    criteria: written.clone(),
                    token_budget: agent::TOKEN_BUDGET,
                });
                req.system.push_str("\n\n");
                req.system.push_str(&crate::goal::brief(&written));
                criteria = written;
            }
        }
    }
    send(EventKind::TaskUpdate {
        state: TaskState::Working,
        note: format!("working, with {}", router.candidates()[choice.index].spec),
    });
    tx.send(Update::Routed(router.decision(&choice)));
    let reply = match agent::run(router, guard, &choice, req, &hooks) {
        Ok(done) => {
            send(EventKind::Message {
                agent: Some(CORE),
                text: done.text.clone(),
            });
            send(EventKind::Alert {
                severity: Severity::Success,
                text: format!(
                    "Task ended after {} model call{}, {} tokens in and {} out.",
                    done.steps,
                    if done.steps == 1 { "" } else { "s" },
                    done.usage.input,
                    done.usage.output
                ),
            });
            send(status(AgentState::Idle));
            // After the work: how it stands against the criteria.
            // A turn of main that handed the work to a branch has not done
            // it: the branch is judged against its own criteria, and this
            // turn against none.
            let mut judged = String::new();
            if branched.get() {
                judged = "; the work is on a branch, which is judged by its own criteria".into();
            } else if !criteria.is_empty() {
                let record = crate::provider::cli::transcript(&Request {
                    messages: done.transcript.clone(),
                    ..Default::default()
                });
                let observed = match &done.outcome {
                    agent::Outcome::Verified(check) => format!("its last check passed (`{check}`)"),
                    agent::Outcome::Failed(why) => why.clone(),
                    agent::Outcome::Unknown => String::new(),
                };
                if let Ok((checked, call)) = crate::goal::check(
                    router.cheapest().as_ref(),
                    &criteria,
                    &record,
                    &done.text,
                    &observed,
                ) {
                    send(model_call(&call));
                    judged = format!("; {}", crate::goal::tally(&checked));
                    send(EventKind::CriteriaChecked {
                        checked: checked.clone(),
                    });
                    tx.send(Update::Checked(checked));
                }
            }
            let stopped = stop.load(Ordering::Relaxed);
            send(EventKind::TaskUpdate {
                state: if stopped {
                    TaskState::Stopped
                } else {
                    TaskState::Done
                },
                note: if stopped {
                    "stopped by you".into()
                } else {
                    format!(
                        "answered after {} model call{}{judged}{}",
                        done.steps,
                        if done.steps == 1 { "" } else { "s" },
                        match &done.outcome {
                            agent::Outcome::Verified(check) =>
                                format!("; its last check passed (`{check}`)"),
                            agent::Outcome::Failed(why) => format!("; {why}"),
                            agent::Outcome::Unknown => String::new(),
                        }
                    )
                },
            });
            // Learn from it: if the task took real work, keep the approach.
            if let Some((kept, text)) = agent::reflect(router, &choice, &done, &hooks) {
                send(EventKind::Alert {
                    severity: if kept {
                        Severity::Success
                    } else {
                        Severity::Info
                    },
                    text,
                });
            }
            tx.send(Update::SkillsUsed(done.skills_used.clone()));
            // After reflection, so a lesson this run wrote is judged too.
            tx.send(Update::Outcome(done.outcome.clone()));
            Some(Message::assistant(done.text, Vec::new()))
        }
        Err(e) => {
            send(EventKind::Alert {
                severity: Severity::Error,
                text: format!("The task stopped: {e}"),
            });
            send(status(AgentState::Failed));
            send(EventKind::TaskUpdate {
                state: TaskState::Failed,
                note: format!("stopped: {e}"),
            });
            None
        }
    };
    tx.send(Update::Finished(reply));
}

fn run_call(router: &Router, task: &str, req: &Request, tx: &Out) {
    let send = |kind: EventKind| {
        tx.send(Update::Event(kind));
    };
    let status = |state: AgentState| EventKind::AgentStatus { id: CORE, state };
    let choice = route(router, task, tx);
    send(EventKind::TaskUpdate {
        state: TaskState::Working,
        note: format!("answering, with {}", router.candidates()[choice.index].spec),
    });
    tx.send(Update::Routed(router.decision(&choice)));
    let (_, fallback, result) = router.complete(&choice, req, &mut |text| {
        tx.send(Update::Streamed(text.to_string()));
    });
    if let Some(note) = fallback {
        send(EventKind::Alert {
            severity: Severity::Warn,
            text: note,
        });
    }
    let reply = match result {
        Ok(resp) => {
            send(model_call(&resp));
            for w in &resp.warnings {
                send(EventKind::Alert {
                    severity: Severity::Warn,
                    text: w.clone(),
                });
            }
            send(EventKind::Message {
                agent: Some(CORE),
                text: resp.text.clone(),
            });
            send(status(AgentState::Idle));
            send(EventKind::TaskUpdate {
                state: TaskState::Done,
                note: "answered".into(),
            });
            Some(Message::assistant(resp.text, Vec::new()))
        }
        Err(e) => {
            send(EventKind::Alert {
                severity: Severity::Error,
                text: format!("Model call failed: {e}"),
            });
            send(status(AgentState::Failed));
            send(EventKind::TaskUpdate {
                state: TaskState::Failed,
                note: format!("the model call failed: {e}"),
            });
            None
        }
    };
    tx.send(Update::Finished(reply));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::Knowledge;
    use crate::provider::Provider;
    use crate::provider::testing::{Fixed, Scripted};
    use crate::router::Candidate;

    /// The events without the tasks they belong to.
    fn kinds(events: Vec<Tagged>) -> Vec<EventKind> {
        events.into_iter().map(|e| e.kind).collect()
    }

    /// A message in the one conversation these tests hold: the first opens
    /// a task, the rest are replies in it.
    fn msg(live: &Live, text: String) -> Command {
        match live.tasks.last() {
            Some(task) => Command::Reply {
                task: task.id,
                text,
            },
            None => Command::UserMessage(text),
        }
    }

    impl Live {
        /// Every task's conversation, in the order the tasks were opened.
        fn history(&self) -> Vec<Message> {
            self.tasks.iter().flat_map(|t| t.thread.clone()).collect()
        }

        fn in_window(&self) -> Vec<NodeId> {
            self.tasks
                .iter()
                .flat_map(|t| t.in_window.iter().copied())
                .collect()
        }
    }

    /// Opens a branch for one piece of work, as main does when it decides
    /// to, without a model having to ask for it.
    fn ask(live: &mut Live, text: String) -> Vec<Tagged> {
        let (id, mut events) = live.open(&text, "main");
        events.extend(live.begin(id, text));
        events
    }

    fn single(p: impl Provider + 'static) -> Arc<Router> {
        Arc::new(Router::single(Arc::new(p)))
    }

    /// Polls until the answer has arrived and, where there is a knowledge
    /// store, the exchange has been remembered.
    fn settle(live: &mut Live) -> Vec<EventKind> {
        let mut out = Vec::new();
        let remembered = live.in_window().len();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut answered = !live.busy();
        while Instant::now() < deadline {
            out.extend(kinds(live.poll()));
            answered |= out.iter().any(|e| {
                matches!(
                    e,
                    EventKind::Message { agent: Some(_), .. }
                        | EventKind::AgentStatus {
                            state: AgentState::Failed,
                            ..
                        }
                )
            });
            let stored = live.knowledge.is_none()
                || live.in_window().len() > remembered
                || out.iter().any(|e| {
                    matches!(
                        e,
                        EventKind::AgentStatus {
                            state: AgentState::Failed,
                            ..
                        }
                    )
                });
            if !live.busy() && answered && stored {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        out
    }

    #[test]
    fn user_message_gets_a_model_reply_and_history_grows() {
        let mut live = Live::new(single(Fixed("hello there")));
        let first = kinds(live.poll());
        assert!(matches!(first[0], EventKind::AgentSpawned { .. }));

        let sent = kinds(live.handle(msg(&live, "hi".into())));
        assert!(matches!(&sent[0], EventKind::TaskOpened { title, .. } if title == "main"));
        assert!(matches!(&sent[1], EventKind::Message { agent: None, text } if text == "hi"));
        assert!(
            sent.iter()
                .any(|e| matches!(e, EventKind::Context { window: 8_000, .. }))
        );

        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::ModelCall { model, tokens_out: 3, .. } if model == "fixed-1"
        )));
        assert!(events.iter().any(
            |e| matches!(e, EventKind::Message { agent: Some(0), text } if text == "hello there")
        ));
        assert!(!live.busy());
        assert_eq!(live.history().len(), 2);
    }

    #[test]
    fn live_reply_renders_in_the_tui_without_a_goal() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut live = Live::new(single(Fixed("hello there")));
        let mut app = crate::app::App::new(false);
        let mut events = kinds(live.poll());
        events.extend(kinds(live.handle(msg(&live, "hi model".into()))));
        events.extend(settle(&mut live));
        for kind in events {
            app.apply(kind);
        }
        assert_eq!(app.last_model, "fixed-1");
        assert_eq!(app.model_calls, 1);

        let mut term = Terminal::new(TestBackend::new(160, 46)).unwrap();
        term.draw(|f| crate::ui::render(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let mut screen = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        assert!(screen.contains("hi model"), "{screen}");
        assert!(screen.contains("hello there"), "{screen}");
    }

    #[test]
    fn each_message_is_routed_and_a_failed_model_falls_back_to_the_core() {
        let pool = |provider: Scripted| {
            vec![Candidate {
                spec: "helper".into(),
                family: "helper".into(),
                good_for: "easy things".into(),
                tier: Some(1),
                provider: Arc::new(provider),
            }]
        };
        let core = || Arc::new(Scripted::ok("core", "COMPLEXITY: 1\nWHY: easy one"));
        let with = |provider: Scripted| {
            Arc::new(Router::with(core(), pool(provider), Knowledge::in_memory()))
        };

        let router = with(Scripted::ok("helper", "from helper"));
        let mut live = Live::new(router);
        kinds(live.poll());
        kinds(live.handle(msg(&live, "hi".into())));
        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Thought { text, .. }
                if text.starts_with("Routing to helper: complexity 1 (easy one)")
        )));
        let models: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                EventKind::ModelCall { model, .. } => Some(model.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(models, ["core", "helper"]);
        assert!(events.iter().any(
            |e| matches!(e, EventKind::Message { agent: Some(0), text } if text == "from helper")
        ));

        // The chosen model is down: the core answers and a warning says why.
        let mut live = Live::new(with(Scripted::err("helper", "usage limit")));
        kinds(live.poll());
        kinds(live.handle(msg(&live, "hi".into())));
        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Alert { severity: Severity::Warn, text } if text.contains("helper failed")
        )));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventKind::Message { agent: Some(0), .. }))
        );
    }

    #[test]
    fn feedback_is_learned_and_posted_as_a_memory() {
        let router = Router::with(
            Arc::new(Scripted::ok("core", "COMPLEXITY: 4\nWHY: hard")),
            vec![Candidate {
                spec: "mid".into(),
                family: "mid".into(),
                good_for: String::new(),
                tier: Some(2),
                provider: Arc::new(Scripted::ok("mid", "an answer")),
            }],
            Knowledge::in_memory(),
        );
        let mut live = Live::new(Arc::new(router));
        kinds(live.poll());

        // Nothing has been answered yet, so there is nothing to rate.
        let early = kinds(live.handle(Command::Feedback {
            good: false,
            task: None,
        }));
        assert!(
            matches!(&early[0], EventKind::Alert { text, .. } if text.starts_with("Nothing learned"))
        );

        kinds(live.handle(msg(&live, "hard one".into())));
        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Thought { text, .. } if text.starts_with("Routing to mid")
        )));

        let learned = kinds(live.handle(Command::Feedback {
            good: false,
            task: None,
        }));
        assert!(matches!(
            &learned[0],
            EventKind::MemoryWrite { node, .. } if node.kind == MemKind::Procedural
        ));
        assert!(matches!(
            &learned[1],
            EventKind::Alert { text, .. } if text.contains("mid at complexity 4: bad")
        ));
        // A bad rating also sends the policy to the core for review. The
        // scripted core's reply is not a script, so the policy is kept.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut reviewed = Vec::new();
        while reviewed.is_empty() && Instant::now() < deadline {
            reviewed = kinds(live.poll());
            thread::sleep(Duration::from_millis(5));
        }
        assert!(reviewed.iter().any(|e| matches!(
            e,
            EventKind::Alert { text, .. } if text.contains("policy review reply was unreadable")
        )));

        // The same kind of task now goes to the stronger model.
        kinds(live.handle(msg(&live, "another hard one".into())));
        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Thought { text, .. } if text.starts_with("Routing to core")
        )));
    }

    #[test]
    fn a_message_is_answered_with_knowledge_from_the_graph() {
        let mut kb = KnowledgeBase::in_memory();
        kb.ingest(
            "proj",
            "docs/policy.md",
            "# Policy\n\nA rewrite that misbehaves is rolled back by the `PolicyStore`.\n",
        );
        kb.commit().unwrap();
        let mut live = Live::with_knowledge(single(Fixed("ok")), Some(Arc::new(Mutex::new(kb))));
        let first = kinds(live.poll());
        // The memory view lists what is remembered, never project files.
        assert!(
            first
                .iter()
                .any(|e| matches!(e, EventKind::MemoryIndexed { nodes } if nodes.is_empty()))
        );

        let sent = kinds(live.handle(msg(&live, "what if a rewrite misbehaves?".into())));
        assert!(sent.iter().any(|e| matches!(
            e,
            EventKind::KnowledgeLookup { picks, .. } if picks.len() == 1 && picks[0].title == "Policy"
        )));
        let knowledge = sent
            .iter()
            .find_map(|e| match e {
                EventKind::Context { sections, .. } => {
                    sections.iter().find(|s| s.name == "knowledge")
                }
                _ => None,
            })
            .expect("a knowledge section in the context");
        assert!(knowledge.tokens > 0);
        assert!(
            knowledge
                .why
                .starts_with("1 passage from the knowledge graph")
        );
        // The exchange is remembered once answered.
        let done = settle(&mut live);
        assert!(done.iter().any(|e| matches!(
            e,
            EventKind::MemoryWrite { node, .. }
                if node.kind == MemKind::Episodic && node.label.starts_with("what if a rewrite")
        )));
        assert_eq!(live.knowledge().unwrap().lock().unwrap().stats().notes, 1);

        // Asked again, the passage is recalled but not the exchange itself,
        // which the model still has in working memory.
        let sent = kinds(live.handle(msg(&live, "what if a rewrite misbehaves?".into())));
        assert!(sent.iter().any(|e| matches!(
            e,
            EventKind::KnowledgeLookup { picks, .. } if picks.len() == 1 && picks[0].title == "Policy"
        )));
        settle(&mut live);

        // A message the store has nothing on adds no knowledge section.
        let sent = kinds(live.handle(msg(&live, "zzz qqq".into())));
        assert!(
            !sent
                .iter()
                .any(|e| matches!(e, EventKind::MemoryRead { .. }))
        );
    }

    #[test]
    fn old_turns_leave_working_memory_and_are_recalled_from_the_graph() {
        let kb = Arc::new(Mutex::new(KnowledgeBase::in_memory()));
        let mut live =
            Live::with_knowledge(single(Fixed("the launch code is tangerine")), Some(kb));
        kinds(live.poll());
        kinds(live.handle(msg(&live, "what is the launch code?".into())));
        settle(&mut live);
        assert_eq!(live.history().len(), 2);

        // A very long message pushes the first exchange out of the window.
        let filler = "filler ".repeat(HISTORY_BUDGET as usize);
        kinds(live.handle(msg(&live, filler)));
        settle(&mut live);
        assert!(live.history().len() <= 2, "{}", live.history().len());
        assert!(live.in_window().len() <= 1);

        // It is no longer in working memory, so the lookup brings it back.
        let sent = kinds(live.handle(msg(&live, "remind me of the launch code".into())));
        assert!(sent.iter().any(|e| matches!(
            e,
            EventKind::KnowledgeLookup { picks, .. } if picks.iter().any(|p| p.title.contains("launch code"))
        )));
        let timeline = sent
            .iter()
            .find_map(|e| match e {
                EventKind::Context { sections, .. } => sections.iter().find(|s| s.name == "task"),
                _ => None,
            })
            .unwrap();
        assert!(
            timeline.why.contains("left working memory"),
            "{}",
            timeline.why
        );
    }

    #[test]
    fn a_model_reads_each_exchange_before_it_is_remembered() {
        use crate::provider::testing::Queue;
        // The answer, then the model's reading of the exchange.
        let core = Queue::new(
            "core",
            &[
                "Nice to meet you, Ada.",
                "KIND: identity\nTITLE: The user's name\nSUMMARY: The user's name is Ada.\nTERMS: name",
                "Your name is Ada.",
                "KIND: event\nTITLE: Asked for own name\nSUMMARY: The user asked what their name is.\nTERMS: name",
            ],
        );
        let kb = Arc::new(Mutex::new(KnowledgeBase::in_memory()));
        let mut live = Live::with_knowledge(Arc::new(Router::single(Arc::new(core))), Some(kb));
        kinds(live.poll());
        kinds(live.handle(msg(&live, "I am Ada".into())));
        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::MemoryWrite { node, .. }
                if node.label == "The user's name" && node.kind == MemKind::Procedural
        )));

        // From now on the model is always told, whatever words are used.
        let sent = kinds(live.handle(msg(&live, "who am I?".into())));
        let memory = sent
            .iter()
            .find_map(|e| match e {
                EventKind::Context { sections, .. } => sections.iter().find(|s| s.name == "memory"),
                _ => None,
            })
            .expect("a standing-memory section in the context");
        assert!(memory.tokens > 0);
        settle(&mut live);
        let kb = live.knowledge().unwrap();
        assert!(
            kb.lock()
                .unwrap()
                .standing()
                .contains("The user's name is Ada.")
        );
    }

    #[test]
    fn a_restart_resumes_the_conversation_from_the_timeline() {
        use crate::timeline::Recorded;
        let said = |agent: Option<u32>, text: &str| Recorded {
            wall: 1_794_579_000,
            task: Some(1),
            kind: EventKind::Message {
                agent,
                text: text.into(),
            },
        };
        let recorded = vec![
            said(None, "I am Ada"),
            said(Some(0), "Nice to meet you."),
            Recorded {
                wall: 1_794_579_001,
                task: None,
                kind: EventKind::Alert {
                    severity: Severity::Info,
                    text: "an old status line".into(),
                },
            },
        ];
        let mut live = Live::new(single(Fixed("ok")));
        live.resume(&recorded);
        assert_eq!(live.history().len(), 2);
        assert_eq!(live.history()[0].content, "I am Ada");
        // The core agent is already in the restored timeline.
        let first = kinds(live.poll());
        assert!(
            !first
                .iter()
                .any(|e| matches!(e, EventKind::AgentSpawned { .. }))
        );
        assert!(first.iter().any(|e| matches!(
            e,
            EventKind::Alert { text, .. } if text.starts_with("Picking up where the last session ended.")
        )));

        // And the screen shows the earlier exchange, stamped with its time.
        let mut app = crate::app::App::new(false);
        app.restore(recorded);
        assert_eq!(app.events.len(), 2, "the stale status line is not replayed");
        assert!(
            app.events
                .iter()
                .all(|e| e.restored && e.wall == 1_794_579_000)
        );
        assert_eq!(app.agents.len(), 0);
    }

    #[test]
    fn a_task_uses_tools_and_waits_for_the_user_before_changing_anything() {
        use crate::provider::testing::Queue;
        let root =
            std::env::temp_dir().join(format!("conscious-live-agent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("notes.txt"), "remember the milk\n").unwrap();
        let core = Queue::new(
            "core",
            &[
                r#"CALL read_file {"path": "notes.txt"}"#,
                r#"CALL write_file {"path": "copy.txt", "content": "remember the milk\n"}"#,
                "Copied the note into copy.txt.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core)))).with_guard(Some(guard));
        kinds(live.poll());
        let sent = kinds(ask(&mut live, "copy my note".into()));
        assert!(
            sent.iter()
                .any(|e| matches!(e, EventKind::GoalSet { title, .. } if title == "copy my note"))
        );

        // It reads by itself, then stops at the write and waits.
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut pending = None;
        while pending.is_none() && Instant::now() < deadline {
            events.extend(kinds(live.poll()));
            pending = events.iter().find_map(|e| match e {
                EventKind::ToolRequested {
                    id,
                    needs_approval: true,
                    ..
                } => Some(*id),
                _ => None,
            });
            thread::sleep(Duration::from_millis(5));
        }
        let id = pending.expect("the write asked for approval");
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::ToolFinished { ok: true, output, .. } if output.contains("remember the milk")
        )));
        assert!(
            !root.join("copy.txt").exists(),
            "nothing is written before approval"
        );
        assert!(live.busy());

        // Approved: the file is written and the task finishes.
        kinds(live.handle(Command::Decide { id, approve: true }));
        let events = settle(&mut live);
        assert!(events.iter().any(
            |e| matches!(e, EventKind::ToolDecision { approved: true, by, .. } if by == "you")
        ));
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Message { agent: Some(0), text } if text == "Copied the note into copy.txt."
        )));
        assert_eq!(
            std::fs::read_to_string(root.join("copy.txt")).unwrap(),
            "remember the milk\n"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn approval_still_works_after_a_restart_with_earlier_tool_calls_on_screen() {
        use crate::app::{App, Modal, ToolStatus};
        use crate::provider::testing::Queue;
        use crate::timeline::Recorded;
        use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

        let root =
            std::env::temp_dir().join(format!("conscious-live-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // The last session ended with tool call #1 waiting for an answer.
        let recorded = vec![Recorded {
            wall: 1_794_579_000,
            task: Some(1),
            kind: EventKind::ToolRequested {
                id: 1,
                agent: 0,
                tool: "write_file".into(),
                args: "old.txt".into(),
                risk: crate::event::Risk::Medium,
                needs_approval: true,
                policy: "needs your approval: changes a file".into(),
            },
        }];
        let core = Queue::new(
            "core",
            &[
                r#"CALL write_file {"path": "new.txt", "content": "hi\n"}"#,
                "Written.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core)))).with_guard(Some(guard));
        live.resume(&recorded);
        let mut app = App::new(false);
        app.restore(recorded);
        assert_eq!(
            app.tools[0].status,
            ToolStatus::Denied,
            "the old call is closed"
        );

        for kind in kinds(live.poll()) {
            app.apply(kind);
        }
        for kind in kinds(ask(&mut live, "write it".into())) {
            app.apply(kind);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.pending().is_empty() && Instant::now() < deadline {
            for kind in kinds(live.poll()) {
                app.apply(kind);
            }
            thread::sleep(Duration::from_millis(5));
        }

        // The prompt is for the new call, not the old one with the same number.
        let waiting = app.pending()[0].id;
        assert_ne!(waiting, 1);
        assert_eq!(app.modal, Some(Modal::Tool(waiting)));
        let shown = app.tool(waiting).unwrap();
        assert_eq!(shown.status, ToolStatus::AwaitingApproval);
        assert!(shown.args.starts_with("new.txt"));

        // Pressing `a` approves it and the file is written.
        app.on_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        for cmd in std::mem::take(&mut app.outbox) {
            for kind in kinds(live.handle(cmd)) {
                app.apply(kind);
            }
        }
        for kind in settle(&mut live) {
            app.apply(kind);
        }
        assert_eq!(app.tool(waiting).unwrap().status, ToolStatus::Ok);
        assert!(root.join("new.txt").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_due_job_starts_by_itself_when_nothing_is_running() {
        use crate::schedule::Missed;
        let now = crate::learn::now();
        let jobs = Arc::new(Mutex::new(Schedule::in_memory(0)));
        // Set five minutes ago to run a minute later: overdue by now.
        jobs.lock()
            .unwrap()
            .add("say the build status", "in 1m", Missed::Run, now - 300)
            .unwrap();
        jobs.lock()
            .unwrap()
            .add("not yet", "in 1h", Missed::Run, now)
            .unwrap();
        let mut live =
            Live::new(single(Fixed("The build is green."))).with_schedule(Some(jobs.clone()));

        // The first poll says what is scheduled; nothing starts before that.
        let first = kinds(live.poll());
        assert!(first.iter().any(|e| matches!(
            e,
            EventKind::Alert { text, .. } if text.starts_with("2 scheduled jobs")
        )));
        assert!(!live.busy());

        let started = kinds(live.poll());
        assert!(started.iter().any(|e| matches!(
            e,
            EventKind::Alert { text, .. } if text.starts_with("Routine #1 (once at ")
        )));
        assert!(live.busy());
        assert!(live.history()[0].content.contains("say the build status"));
        assert!(live.history()[0].content.starts_with("[Routine #1"));

        let events = settle(&mut live);
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Message { agent: Some(_), text } if text == "The build is green."
        )));
        // A routine's run is a branch: what it found is merged into main.
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Merged { branch: 1, ok: true, result, .. } if result == "The build is green."
        )));
        let main = live.tasks.iter().find(|t| t.id == MAIN).unwrap();
        assert!(main.thread[0].content.contains("The build is green."));
        // The one-off job is finished; the other waits for its time.
        let left: Vec<u32> = jobs.lock().unwrap().jobs().iter().map(|j| j.id).collect();
        assert_eq!(left, [2]);
        assert!(!live.busy() && kinds(live.poll()).is_empty());
    }

    #[test]
    fn the_agent_sets_a_job_for_itself_once_the_user_approves() {
        use crate::provider::testing::Queue;
        let root =
            std::env::temp_dir().join(format!("conscious-live-schedule-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let core = Queue::new(
            "core",
            &[
                r#"CALL schedule {"action": "create", "task": "Run the tests and report failures.", "when": "every 1h"}"#,
                r#"CALL schedule {"action": "create", "task": "too often", "when": "every 10s"}"#,
                "Scheduled the hourly test run.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let jobs = Arc::new(Mutex::new(Schedule::in_memory(0)));
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core))))
            .with_guard(Some(guard))
            .with_schedule(Some(jobs.clone()));
        kinds(live.poll());
        let mut events = kinds(live.handle(msg(&live, "check the tests every hour".into())));
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut decided = 0;
        while live.busy() && Instant::now() < deadline {
            let new = kinds(live.poll());
            for e in &new {
                // In careful mode setting a job is the user's decision.
                if let EventKind::ToolRequested {
                    id,
                    tool,
                    needs_approval,
                    ..
                } = e
                {
                    assert_eq!((tool.as_str(), *needs_approval), ("schedule", true));
                    kinds(live.handle(Command::Decide {
                        id: *id,
                        approve: true,
                    }));
                    decided += 1;
                }
            }
            events.extend(new);
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(decided, 2);
        let jobs = jobs.lock().unwrap();
        // The second request broke the limit and was refused by the scheduler.
        assert_eq!(jobs.jobs().len(), 1);
        assert_eq!(jobs.jobs()[0].rule(0), "every 1 hour");
        assert_eq!(jobs.jobs()[0].task, "Run the tests and report failures.");
        assert!(events.iter().any(|e| matches!(
            e,
            EventKind::Alert { text, .. } if text.starts_with("The agent scheduled as job #1: every 1 hour")
        )));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, EventKind::ToolFinished { ok: false, .. }))
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A model that answers only once it is let go, and records what it
    /// was sent.
    struct Held {
        open: Arc<AtomicBool>,
        seen: Arc<Mutex<Vec<Vec<String>>>>,
    }

    impl Provider for Held {
        fn label(&self) -> String {
            "held".into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            _: &mut dyn FnMut(&str),
        ) -> Result<Response, crate::provider::ProviderError> {
            self.seen
                .lock()
                .unwrap()
                .push(req.messages.iter().map(|m| m.content.clone()).collect());
            while !self.open.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(2));
            }
            let asked = req.messages.last().map_or("", |m| m.content.as_str());
            Ok(Response {
                text: format!("answer to: {asked}"),
                tool_calls: Vec::new(),
                usage: crate::provider::Usage {
                    input: 5,
                    cached: 0,
                    output: 3,
                },
                latency: Duration::from_millis(1),
                model: "held".into(),
                warnings: Vec::new(),
            })
        }
    }

    type Seen = Arc<Mutex<Vec<Vec<String>>>>;

    fn held() -> (Live, Arc<AtomicBool>, Seen) {
        let open = Arc::new(AtomicBool::new(false));
        let seen: Seen = Arc::default();
        let live = Live::new(single(Held {
            open: open.clone(),
            seen: seen.clone(),
        }));
        (live, open, seen)
    }

    /// Polls until `done` holds for everything seen so far.
    fn until(live: &mut Live, done: impl Fn(&[Tagged]) -> bool) -> Vec<Tagged> {
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            out.extend(live.poll());
            if done(&out) {
                return out;
            }
            thread::sleep(Duration::from_millis(5));
        }
        panic!("not seen within 5 s: {out:?}");
    }

    fn state_of(events: &[Tagged], task: u32) -> Option<TaskState> {
        events.iter().rev().find_map(|e| match &e.kind {
            EventKind::TaskUpdate { state, .. } if e.task == Some(task) => Some(*state),
            _ => None,
        })
    }

    #[test]
    fn each_message_is_its_own_task_and_they_run_side_by_side() {
        let (mut live, open, seen) = held();
        live.poll();
        let first = ask(&mut live, "count the files".into());
        let second = ask(&mut live, "what is a tide".into());
        // Each opens a task, and everything it produces carries its number.
        assert!(matches!(
            &first[0],
            Tagged { task: Some(1), kind: EventKind::TaskOpened { title, origin } }
                if title == "count the files" && origin == "main"
        ));
        assert!(first.iter().all(|e| e.task == Some(1)));
        assert!(second.iter().all(|e| e.task == Some(2)));
        assert_eq!(state_of(&first, 1), Some(TaskState::Working));
        // Both are with the model at the same time.
        let deadline = Instant::now() + Duration::from_secs(5);
        while seen.lock().unwrap().len() < 2 {
            assert!(Instant::now() < deadline, "the second task did not start");
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(live.running(), 2);
        // A task's context holds its own conversation and nothing of the
        // other's.
        let mut sent = seen.lock().unwrap().clone();
        sent.sort();
        assert_eq!(sent, [vec!["count the files"], vec!["what is a tide"]]);

        // A reply to a task that has not answered yet is refused.
        let refused = live.handle(Command::Reply {
            task: 1,
            text: "and the folders?".into(),
        });
        assert!(matches!(
            &refused[..],
            [Tagged {
                task: Some(1),
                kind: EventKind::Alert {
                    severity: Severity::Warn,
                    ..
                }
            }]
        ));

        open.store(true, Ordering::Relaxed);
        let events = until(&mut live, |e| {
            state_of(e, 1) == Some(TaskState::Done) && state_of(e, 2) == Some(TaskState::Done)
        });
        for (task, answer) in [
            (1, "answer to: count the files"),
            (2, "answer to: what is a tide"),
        ] {
            assert!(events.iter().any(|e| e.task == Some(task)
                && matches!(&e.kind, EventKind::Message { agent: Some(_), text } if text == answer)));
        }
        assert!(!live.busy());

        // A reply carries on that task's conversation, and only that one.
        seen.lock().unwrap().clear();
        let replied = live.handle(Command::Reply {
            task: 2,
            text: "and a neap tide?".into(),
        });
        assert!(replied.iter().all(|e| e.task == Some(2)));
        assert!(
            !replied
                .iter()
                .any(|e| matches!(e.kind, EventKind::TaskOpened { .. }))
        );
        until(&mut live, |e| state_of(e, 2) == Some(TaskState::Done));
        assert_eq!(
            seen.lock().unwrap()[0],
            [
                "what is a tide",
                "answer to: what is a tide",
                "and a neap tide?"
            ]
        );
    }

    /// A model that, as main, opens a branch for the report and answers
    /// everything else itself; as the branch, it writes the report once it
    /// is let go. It records who it was asked as and whether it was offered
    /// the `branch` tool.
    struct Brancher {
        open: Arc<AtomicBool>,
        offered: Arc<Mutex<Vec<(bool, bool)>>>,
    }

    impl Provider for Brancher {
        fn label(&self) -> String {
            "brancher".into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            on_text: &mut dyn FnMut(&str),
        ) -> Result<Response, crate::provider::ProviderError> {
            let main = req.system.contains(MAIN_BRIEF);
            assert_ne!(main, req.system.contains(BRANCH_BRIEF));
            let has = |name: &str| req.tools.iter().any(|t| t.name == name);
            assert_ne!(main, has("write_file"));
            assert_ne!(main, has("edit_file"));
            assert!(has("read_file") && has("bash"));
            self.offered.lock().unwrap().push((main, has("branch")));
            let say =
                |text: &'static str| Scripted::ok("brancher", text).complete(req, &mut |_| {});
            let _ = on_text;
            if !main {
                while !self.open.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(2));
                }
                return say("The report is written: 3 sections.");
            }
            let last = req.messages.last().unwrap();
            if last.role == crate::provider::Role::Tool {
                return say("I opened a branch for the report.");
            }
            if last.content.ends_with("write the quarterly report") {
                let mut resp = say("")?;
                resp.tool_calls = vec![crate::provider::ToolCall {
                    id: "call_0".into(),
                    name: "branch".into(),
                    arguments: serde_json::json!({
                        "title": "quarterly report",
                        "task": "Write the quarterly report from the figures in the project.",
                    }),
                }];
                return Ok(resp);
            }
            if last.content.contains("what did it find") {
                let merged = last
                    .content
                    .contains("[Branch #1 \"quarterly report\" has ended")
                    && last.content.contains("3 sections");
                return say(if merged {
                    "It wrote 3 sections."
                } else {
                    "I do not know."
                });
            }
            say("A tide is the sea rising and falling.")
        }
    }

    #[test]
    fn main_decides_to_branch_keeps_going_and_the_result_is_merged_back() {
        let root =
            std::env::temp_dir().join(format!("conscious-live-branch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let (open, offered) = (Arc::new(AtomicBool::new(false)), Arc::default());
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(single(Brancher {
            open: open.clone(),
            offered: Arc::clone(&offered),
        }))
        .with_guard(Some(guard));
        live.poll();
        let answer_of = |events: &[Tagged], task: u32| -> Option<String> {
            events.iter().rev().find_map(|e| match &e.kind {
                EventKind::Message {
                    agent: Some(_),
                    text,
                } if e.task == Some(task) => Some(text.clone()),
                _ => None,
            })
        };

        // What is said goes to main; nothing else is opened for it.
        let sent = live.handle(Command::UserMessage("write the quarterly report".into()));
        assert!(matches!(
            &sent[0],
            Tagged { task: Some(MAIN), kind: EventKind::TaskOpened { origin, .. } } if origin == "you"
        ));
        assert!(sent.iter().all(|e| e.task == Some(MAIN)));

        // Main decides this is work of its own: it opens a branch, says so,
        // and is done with its turn while the branch works.
        let events = until(&mut live, |e| {
            state_of(e, MAIN) == Some(TaskState::Done) && state_of(e, 1) == Some(TaskState::Working)
        });
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::TaskOpened { title, origin }
                if e.task == Some(1) && title == "quarterly report" && origin == "main"
        )));
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { text, .. }
                if e.task == Some(MAIN) && text == "Opened branch #1: quarterly report"
        )));
        assert_eq!(
            answer_of(&events, MAIN).as_deref(),
            Some("I opened a branch for the report.")
        );
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::TaskUpdate { state: TaskState::Done, note }
                if e.task == Some(MAIN) && note.contains("the work is on a branch")
        )));
        // The branch was given the brief main wrote, and nothing else.
        let branch = live.tasks.iter().find(|t| t.id == 1).unwrap();
        assert_eq!(
            branch.thread[0].content,
            "Write the quarterly report from the figures in the project."
        );
        assert_eq!(live.running(), 1);

        // Main keeps going while the branch is still at work.
        let sent = live.handle(Command::UserMessage("what is a tide".into()));
        assert!(
            !sent
                .iter()
                .any(|e| matches!(e.kind, EventKind::TaskOpened { .. }))
        );
        let events = until(&mut live, |e| state_of(e, MAIN) == Some(TaskState::Done));
        assert_eq!(
            answer_of(&events, MAIN).as_deref(),
            Some("A tide is the sea rising and falling.")
        );
        assert_eq!(live.running(), 1);

        // When the branch ends, its result is merged into main.
        open.store(true, Ordering::Relaxed);
        let events = until(&mut live, |e| {
            e.iter().any(|e| matches!(e.kind, EventKind::Merged { .. }))
        });
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Merged { branch: 1, title, result, ok: true }
                if e.task == Some(MAIN)
                    && title == "quarterly report"
                    && result == "The report is written: 3 sections."
        )));
        assert_eq!(live.running(), 0);

        // From then on main's own conversation holds it.
        live.handle(Command::UserMessage("what did it find?".into()));
        let events = until(&mut live, |e| state_of(e, MAIN) == Some(TaskState::Done));
        assert_eq!(
            answer_of(&events, MAIN).as_deref(),
            Some("It wrote 3 sections.")
        );
        // Its turns still alternate, for models that take nothing else.
        let main = live.tasks.iter().find(|t| t.id == MAIN).unwrap();
        assert!(main.thread.windows(2).all(|w| w[0].role != w[1].role));

        // Only main is offered the tool that opens a branch, and main is not
        // offered the tools that change files: that work is a branch's.
        let offered = offered.lock().unwrap();
        assert!(offered.iter().any(|(main, _)| !main));
        assert!(offered.iter().all(|(main, branch)| main == branch));
        // No message was opened as a task of its own.
        assert_eq!(live.tasks.len(), 2);
    }

    #[test]
    fn what_is_said_while_main_is_answering_is_taken_up_together_when_it_is_free() {
        let (mut live, open, seen) = held();
        live.poll();
        live.handle(Command::UserMessage("one".into()));
        // Main is busy: these are shown at once and kept for its next turn.
        for text in ["two", "three"] {
            let sent = live.handle(Command::UserMessage(text.into()));
            assert!(matches!(
                &sent[..],
                [Tagged { task: Some(MAIN), kind: EventKind::Message { agent: None, text: said } }]
                    if said == text
            ));
        }
        // Main does not take a branch's place: as many branches as may run
        // at once still start beside it.
        for n in 1..=MAX_RUNNING {
            ask(&mut live, format!("branch {n}"));
        }
        assert_eq!(live.running(), MAX_RUNNING);
        assert!(live.tasks.iter().all(|t| t.running));

        open.store(true, Ordering::Relaxed);
        let events = until(&mut live, |e| {
            e.iter().any(|e| {
                matches!(
                    &e.kind,
                    EventKind::Message { agent: Some(_), text }
                        if e.task == Some(MAIN) && text.ends_with("two\n\nthree")
                )
            })
        });
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Message { agent: Some(_), text }
                if e.task == Some(MAIN) && text == "answer to: one"
        )));
        // The second turn was one request, made after the first was
        // answered (branches that merged meanwhile are in it too).
        assert!(seen.lock().unwrap().iter().any(|m| m.len() == 3
            && m[0].starts_with("one")
            && m[1] == "answer to: one"
            && m[2].ends_with("two\n\nthree")));
        let answers = events
            .iter()
            .filter(|e| {
                e.task == Some(MAIN) && matches!(e.kind, EventKind::Message { agent: Some(_), .. })
            })
            .count();
        assert_eq!(answers, 2);
    }

    #[test]
    fn a_secret_is_typed_by_the_user_and_its_value_reaches_no_event_and_no_model() {
        use crate::provider::testing::Queue;
        use crate::secrets::Hidden;
        let root =
            std::env::temp_dir().join(format!("conscious-live-secret-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let core = Queue::new(
            "core",
            &[
                r#"CALL secret {"action": "request", "name": "LIVE_TEST_KEY", "why": "to publish the package"}"#,
                "The key is stored; I will use it as $LIVE_TEST_KEY.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core)))).with_guard(Some(guard));
        live.poll();
        ask(&mut live, "publish the package".into());
        // The agent asks; the request names the secret and says why, and
        // nothing more. It does not need approving: the user answers it by
        // typing, or not.
        let mut events = until(&mut live, |e| {
            e.iter()
                .any(|e| matches!(e.kind, EventKind::SecretAsked { .. }))
        });
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::SecretAsked { name, why }
                if e.task == Some(1) && name == "LIVE_TEST_KEY" && why == "to publish the package"
        )));
        assert!(live.busy(), "the task waits for the user");

        // The user types it into its window.
        let stored = live.handle(Command::Secret {
            name: "LIVE_TEST_KEY".into(),
            value: Some(Hidden("val-0123456789-live".into())),
        });
        assert!(stored.iter().any(|e| matches!(
            &e.kind,
            EventKind::SecretAnswered { name, stored: true } if e.task == Some(1) && name == "LIVE_TEST_KEY"
        )));
        events.extend(stored);
        events.extend(until(&mut live, |e| {
            state_of(e, 1) == Some(TaskState::Done)
        }));
        // It is in the store, and the model was told only that it is there.
        assert_eq!(
            crate::secrets::get("LIVE_TEST_KEY").as_deref(),
            Some("val-0123456789-live")
        );
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::ToolFinished { ok: true, output, .. } if output.contains("The user entered LIVE_TEST_KEY")
        )));
        // The value is in no event at all, and so not in the timeline, on
        // the screen, or in anything sent to a model or to Telegram.
        assert!(!format!("{events:?}").contains("val-0123456789-live"));
        let thread: String = live
            .tasks
            .iter()
            .flat_map(|t| t.thread.iter().map(|m| m.content.clone()))
            .collect();
        assert!(!thread.contains("val-0123456789-live"));

        // Declining tells the agent no, and stores nothing.
        let core = Queue::new(
            "core",
            &[
                r#"CALL secret {"action": "request", "name": "LIVE_TEST_REFUSED"}"#,
                "I cannot go on without it.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core)))).with_guard(Some(guard));
        live.poll();
        ask(&mut live, "publish the package".into());
        until(&mut live, |e| {
            e.iter()
                .any(|e| matches!(e.kind, EventKind::SecretAsked { .. }))
        });
        live.handle(Command::Secret {
            name: "LIVE_TEST_REFUSED".into(),
            value: None,
        });
        let events = until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::ToolFinished { ok: false, output, .. } if output.contains("did not enter")
        )));
        assert_eq!(crate::secrets::get("LIVE_TEST_REFUSED"), None);
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A bot that hears what a test puts in and keeps what it was sent.
    #[derive(Default)]
    struct FakeBot {
        incoming: Mutex<VecDeque<crate::telegram::Incoming>>,
        sent: Mutex<Vec<(i64, String)>>,
        /// Questions sent with buttons: (the text, what each button carries).
        asked: Mutex<Vec<(String, Vec<String>)>>,
        /// (message, what it was changed to say).
        settled: Mutex<Vec<(i64, String)>>,
        /// The note each pressed button was acknowledged with.
        notes: Mutex<Vec<String>>,
        /// Messages marked as seen, and how often "typing…" was shown.
        seen: Mutex<Vec<i64>>,
        typing: AtomicU32,
        update: AtomicU32,
    }

    impl FakeBot {
        fn incoming(&self, from: i64, private: bool, text: &str) -> crate::telegram::Incoming {
            crate::telegram::Incoming {
                update: i64::from(self.update.fetch_add(1, Ordering::Relaxed)) + 1,
                chat: from,
                from,
                private,
                message: 40 + i64::from(self.update.load(Ordering::Relaxed)),
                text: text.into(),
                media: None,
                button: None,
            }
        }

        fn hears(&self, from: i64, private: bool, text: &str) {
            let message = self.incoming(from, private, text);
            self.incoming.lock().unwrap().push_back(message);
        }

        fn presses(&self, from: i64, data: &str) {
            let mut message = self.incoming(from, true, "");
            message.button = Some(crate::telegram::Button {
                id: "press".into(),
                data: data.into(),
            });
            self.incoming.lock().unwrap().push_back(message);
        }

        fn sends_file(&self, from: i64, kind: &str, name: &str, caption: &str) {
            let mut message = self.incoming(from, true, caption);
            message.media = Some(crate::telegram::Media {
                kind: kind.into(),
                file: "file-1".into(),
                name: name.into(),
                size: 11,
            });
            self.incoming.lock().unwrap().push_back(message);
        }

        fn sent(&self) -> Vec<(i64, String)> {
            self.sent.lock().unwrap().clone()
        }
    }

    impl crate::telegram::Bot for FakeBot {
        fn updates(
            &self,
            after: i64,
            _: u32,
        ) -> Result<(Vec<crate::telegram::Incoming>, i64), String> {
            let got: Vec<_> = self.incoming.lock().unwrap().drain(..).collect();
            if got.is_empty() {
                thread::sleep(Duration::from_millis(3));
            }
            let last = got.iter().map(|m| m.update).max().unwrap_or(after);
            Ok((got, last))
        }

        fn send(&self, chat: i64, text: &str) -> Result<(), String> {
            self.sent.lock().unwrap().push((chat, text.to_string()));
            Ok(())
        }

        fn ask(&self, _: i64, text: &str, buttons: &[(String, String)]) -> Result<i64, String> {
            let mut asked = self.asked.lock().unwrap();
            asked.push((
                text.to_string(),
                buttons.iter().map(|(_, data)| data.clone()).collect(),
            ));
            Ok(100 + asked.len() as i64)
        }

        fn settle(&self, _: i64, message: i64, text: &str) -> Result<(), String> {
            self.settled
                .lock()
                .unwrap()
                .push((message, text.to_string()));
            Ok(())
        }

        fn pressed(&self, _: &str, note: &str) -> Result<(), String> {
            self.notes.lock().unwrap().push(note.to_string());
            Ok(())
        }

        fn fetch(&self, _: &str) -> Result<(Vec<u8>, String), String> {
            Ok((b"image bytes".to_vec(), "photos/file_7.jpg".to_string()))
        }

        fn seen(&self, _: i64, message: i64) -> Result<(), String> {
            self.seen.lock().unwrap().push(message);
            Ok(())
        }

        fn typing(&self, _: i64) -> Result<(), String> {
            self.typing.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn a_call_is_approved_from_telegram_with_a_button_and_a_file_sent_there_is_saved() {
        use crate::provider::testing::Queue;
        let root =
            std::env::temp_dir().join(format!("conscious-live-telegram-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let core = Queue::new(
            "core",
            &[
                r#"CALL write_file {"path": "note.txt", "content": "hello"}"#,
                "Written.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core)))).with_guard(Some(guard));
        live.poll();
        let bot = Arc::new(FakeBot::default());
        live.connect_telegram(bot.clone(), None);
        live.telegram.as_mut().unwrap().pairing.owner = Some(7);
        let wait = |live: &mut Live, done: &dyn Fn() -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !done() {
                assert!(Instant::now() < deadline, "not seen within 5 s");
                live.poll();
                thread::sleep(Duration::from_millis(3));
            }
        };

        // A branch wants to write a file: the owner is asked, with buttons.
        ask(&mut live, "write the note".into());
        wait(&mut live, &|| !bot.asked.lock().unwrap().is_empty());
        let (text, buttons) = bot.asked.lock().unwrap()[0].clone();
        assert!(text.starts_with("Approval needed"), "{text}");
        assert!(
            text.contains("write_file") && text.contains("note.txt"),
            "{text}"
        );
        let id: u32 = buttons[0].strip_prefix("ok:").unwrap().parse().unwrap();
        assert_eq!(buttons, [format!("ok:{id}"), format!("no:{id}")]);
        assert!(!root.join("note.txt").exists());

        // Someone else pressing Approve decides nothing.
        bot.presses(8, &format!("ok:{id}"));
        // The owner pressing it does: the call runs, the press is
        // acknowledged, and the message says what was decided.
        bot.presses(7, &format!("ok:{id}"));
        wait(&mut live, &|| !bot.settled.lock().unwrap().is_empty());
        assert_eq!(
            std::fs::read_to_string(root.join("note.txt")).unwrap(),
            "hello"
        );
        assert_eq!(*bot.notes.lock().unwrap(), ["Approved."]);
        let (message, settled) = bot.settled.lock().unwrap()[0].clone();
        assert_eq!(message, 101);
        assert!(settled.ends_with("Approved (by you)."), "{settled}");
        // Pressing again on a call that is over changes nothing.
        bot.presses(7, &format!("no:{id}"));
        wait(&mut live, &|| bot.notes.lock().unwrap().len() == 2);
        assert_eq!(bot.notes.lock().unwrap()[1], "This is no longer waiting.");

        // A call that widens the agent's own access is put to the owner
        // without buttons: it is decided at the screen only.
        live.telegram_tell(&[Tagged {
            task: Some(1),
            kind: EventKind::ToolRequested {
                id: 900,
                agent: 0,
                tool: "access".into(),
                args: "set_mode trusted".into(),
                risk: crate::event::Risk::High,
                needs_approval: true,
                policy: "widens what the agent may do".into(),
            },
        }]);
        wait(&mut live, &|| {
            bot.sent()
                .iter()
                .any(|(_, text)| text.ends_with("can only be decided at the screen."))
        });
        assert_eq!(bot.asked.lock().unwrap().len(), 1);

        // A photo the owner sends is saved in the project and main is told
        // where; one from anyone else is not fetched.
        bot.sends_file(8, "photo", "", "from a stranger");
        bot.sends_file(7, "photo", "", "what is on the whiteboard?");
        wait(&mut live, &|| {
            live_main_thread(&root).is_some_and(|saved| saved.len() == 1)
        });
        let saved = live_main_thread(&root).unwrap();
        assert!(saved[0].ends_with("-photo.jpg"), "{saved:?}");
        assert_eq!(
            std::fs::read(root.join(TELEGRAM_INBOX).join(&saved[0])).unwrap(),
            b"image bytes"
        );
        let told_main = |live: &Live| {
            live.tasks.iter().any(|t| {
                t.id == MAIN
                    && t.thread
                        .iter()
                        .any(|m| m.content.contains("what is on the whiteboard?"))
            })
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !told_main(&live) {
            assert!(Instant::now() < deadline, "main was not told of the photo");
            live.poll();
            thread::sleep(Duration::from_millis(3));
        }
        let main = live.tasks.iter().find(|t| t.id == MAIN).unwrap();
        let told = &main.thread[0].content;
        assert!(
            told.contains(&format!("a photo, saved as {TELEGRAM_INBOX}/{}", saved[0])),
            "{told}"
        );
        assert!(told.contains("you cannot see or hear it"), "{told}");
        assert!(!told.contains("from a stranger"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The files saved from Telegram under `root`, if the folder exists.
    fn live_main_thread(root: &std::path::Path) -> Option<Vec<String>> {
        let mut names: Vec<String> = std::fs::read_dir(root.join(TELEGRAM_INBOX))
            .ok()?
            .filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned()))
            .collect();
        names.sort();
        Some(names)
    }

    #[test]
    fn a_task_that_stops_with_an_error_says_so_on_telegram_too() {
        let bot = Arc::new(FakeBot::default());
        let mut live = Live::new(single(Scripted {
            label: "limited",
            reply: Err("You've hit your 5-hour limit · resets 5am (UTC)"),
        }));
        live.poll();
        live.connect_telegram(bot.clone(), None);
        live.telegram.as_mut().unwrap().pairing.owner = Some(7);
        bot.hears(7, true, "ok so");
        let deadline = Instant::now() + Duration::from_secs(5);
        while bot.sent().is_empty() {
            assert!(
                Instant::now() < deadline,
                "the error did not reach Telegram"
            );
            live.poll();
            thread::sleep(Duration::from_millis(3));
        }
        let (chat, text) = bot.sent()[0].clone();
        assert_eq!(chat, 7);
        assert!(text.contains("5-hour limit"), "{text}");
        // The bridge's own warnings about Telegram are not sent to Telegram.
        live.telegram_tell(&[Tagged::system(EventKind::Alert {
            severity: Severity::Error,
            text: "Telegram: a message could not be sent: timeout".into(),
        })]);
        thread::sleep(Duration::from_millis(30));
        assert!(bot.sent().iter().all(|(_, t)| !t.starts_with("Telegram:")));
    }

    /// A model that answers at length, in markdown, and rewrites a reply
    /// for a phone when that is what it is asked.
    struct Wordy;

    impl Provider for Wordy {
        fn label(&self) -> String {
            "wordy".into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            _: &mut dyn FnMut(&str),
        ) -> Result<Response, crate::provider::ProviderError> {
            let reply = if req.system == crate::telegram::FOR_PHONE {
                // Even if the small model slips some markup in.
                "The build **passed**: 268 tests ran, none failed."
            } else {
                // Long enough to keep the task running for a moment.
                thread::sleep(Duration::from_millis(40));
                "## Build result\n\nThe build **passed**.\n\n| Check | Result |\n|---|---|\n| tests | 268 passed |\n| lint | clean |\n\n- `cargo fmt --check` was clean\n- nothing failed\n\nSee `target/` for the output, and say if you want the full log."
            };
            Scripted::ok("wordy", reply).complete(req, &mut |_| {})
        }
    }

    #[test]
    fn telegram_shows_seen_and_typing_and_gets_a_short_plain_reply_while_the_screen_keeps_markdown()
    {
        let bot = Arc::new(FakeBot::default());
        let mut live = Live::new(single(Wordy));
        live.poll();
        live.connect_telegram(bot.clone(), None);
        live.telegram.as_mut().unwrap().pairing.owner = Some(7);

        bot.hears(7, true, "how did the build go?");
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut events = Vec::new();
        while bot.sent().is_empty() {
            assert!(Instant::now() < deadline, "no reply reached Telegram");
            events.extend(live.poll());
            thread::sleep(Duration::from_millis(3));
        }
        // The message was marked as seen, and "typing…" shown, before the
        // answer came.
        assert_eq!(bot.seen.lock().unwrap().len(), 1);
        assert!(bot.typing.load(Ordering::Relaxed) >= 1);
        // The phone gets one short plain message: no markup, no table.
        let (chat, text) = bot.sent()[0].clone();
        assert_eq!(chat, 7);
        assert_eq!(text, "The build passed: 268 tests ran, none failed.");
        // The screen still has the reply as it was written.
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Message { agent: Some(_), text }
                if e.task == Some(MAIN) && text.starts_with("## Build result") && text.contains("| tests |")
        )));
    }

    #[test]
    fn a_paired_telegram_account_talks_to_main_and_nobody_else_is_heard() {
        let bot = Arc::new(FakeBot::default());
        let mut live = Live::new(single(Fixed("It is high tide.")));
        live.poll();
        assert!(!live.connected());
        let said = live.connect_telegram(bot.clone(), None);
        assert!(live.connected());
        let code = live.telegram.as_ref().unwrap().code.clone();
        // The way to pair is said on the harness's own screen only.
        assert!(matches!(
            &said[..],
            [Tagged { task: None, kind: EventKind::Alert { text, .. } }]
                if text.contains(&format!("/pair {code}"))
        ));
        type Sent = [(i64, String)];
        let wait_for = |live: &mut Live, done: &dyn Fn(&Sent) -> bool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut events = Vec::new();
            while !done(&bot.sent()) {
                assert!(Instant::now() < deadline, "sent: {:?}", bot.sent());
                events.extend(live.poll());
                thread::sleep(Duration::from_millis(3));
            }
            events
        };

        // The code can always be found again, without the screen.
        assert_eq!(
            live.telegram_state(),
            format!("connected, not paired: send your bot the message /pair {code}")
        );

        // Before pairing, nothing anyone sends reaches the conversation.
        // But the code is said again on the screen, where the user who is
        // trying to pair will see it.
        bot.hears(9, true, "delete everything");
        let events = wait_for(&mut live, &|sent| sent.len() == 1);
        assert!(bot.sent()[0].1.contains("not paired"));
        assert!(live.tasks.is_empty());
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { severity: Severity::Warn, text }
                if e.task.is_none() && text.ends_with(&format!("/pair {code}"))
        )));

        // Guessing the code gets nowhere: a few wrong ones lock pairing,
        // after which even the right code does nothing.
        let locked_bot = Arc::new(FakeBot::default());
        let mut guessed = Live::new(single(Fixed("ok")));
        guessed.poll();
        guessed.connect_telegram(locked_bot.clone(), None);
        let real = guessed.telegram.as_ref().unwrap().code.clone();
        for n in 0..crate::telegram::PAIR_TRIES {
            let wrong = format!("{:06}", (real.parse::<u32>().unwrap() + 1 + n) % 1_000_000);
            locked_bot.hears(9, true, &format!("/pair {wrong}"));
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut said = Vec::new();
        while !guessed.telegram_state().contains("pairing locked") {
            assert!(Instant::now() < deadline, "pairing was not locked");
            said.extend(guessed.poll());
            thread::sleep(Duration::from_millis(3));
        }
        assert!(said.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { severity: Severity::Warn, text } if text.contains("pairing is locked")
        )));
        locked_bot.hears(9, true, &format!("/pair {real}"));
        for _ in 0..40 {
            guessed.poll();
            thread::sleep(Duration::from_millis(3));
        }
        assert!(guessed.telegram.as_ref().unwrap().pairing.owner.is_none());
        // And a stranger cannot make the bot send without end.
        assert!(locked_bot.sent().len() <= 1, "{:?}", locked_bot.sent());

        // The account that sends the code becomes the owner.
        bot.hears(7, true, &format!("/pair {code}"));
        let events = wait_for(&mut live, &|sent| sent.len() == 2);
        assert_eq!(bot.sent()[1].0, 7);
        assert_eq!(
            live.telegram_state(),
            "connected and paired with your account"
        );
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { severity: Severity::Success, text } if text.contains("paired")
        )));

        // What the owner says goes to main, and main's reply goes back.
        bot.hears(8, true, "what is the tide doing");
        bot.hears(7, false, "said in a group");
        bot.hears(7, true, "what is the tide doing");
        let events = wait_for(&mut live, &|sent| sent.len() == 3);
        assert_eq!(bot.sent()[2], (7, "It is high tide.".to_string()));
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Message { agent: None, text }
                if e.task == Some(MAIN) && text == "what is the tide doing"
        )));
        // The stranger and the group were heard by nobody: one message
        // reached main, and one answer left.
        let main = live.tasks.iter().find(|t| t.id == MAIN).unwrap();
        assert_eq!(main.thread.len(), 2);
        assert_eq!(live.tasks.len(), 1);
    }

    /// A model that writes its reply in pieces, then waits to be let go.
    struct Streamer {
        open: Arc<AtomicBool>,
    }

    impl Provider for Streamer {
        fn label(&self) -> String {
            "streamer".into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            on_text: &mut dyn FnMut(&str),
        ) -> Result<Response, crate::provider::ProviderError> {
            for piece in ["Hel", "lo there.", "\n<tool_call>{\"name\": \"x\""] {
                on_text(piece);
            }
            while !self.open.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(2));
            }
            Scripted::ok("streamer", "Hello there.").complete(req, &mut |_| {})
        }
    }

    #[test]
    fn a_message_shows_before_its_lookup_and_the_reply_shows_as_it_is_written() {
        let open = Arc::new(AtomicBool::new(false));
        let mut live = Live::new(single(Streamer { open: open.clone() })).with_echo_first(true);
        live.poll();
        assert!(live.drafts().is_empty());

        // What was typed comes straight back, before any context is put
        // together for it.
        let sent = kinds(live.handle(Command::UserMessage("hi".into())));
        assert!(matches!(&sent[1], EventKind::Message { agent: None, text } if text == "hi"));
        assert!(!sent.iter().any(|e| matches!(e, EventKind::Context { .. })));
        // The next poll starts the work.
        let started = kinds(live.poll());
        assert!(
            started
                .iter()
                .any(|e| matches!(e, EventKind::Context { .. }))
        );

        // The reply is handed over as it grows. A tool request the model
        // writes out as text is not part of it.
        let mut seen = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while seen.last().map(String::as_str) != Some("Hello there.") {
            assert!(Instant::now() < deadline, "drafts seen: {seen:?}");
            live.poll();
            for (task, text) in live.drafts() {
                assert_eq!(task, MAIN);
                seen.push(text);
            }
            thread::sleep(Duration::from_millis(2));
        }
        // Nothing new to hand over until it changes.
        assert!(live.drafts().is_empty());

        // When the reply is finished it is an event, and the draft ends.
        open.store(true, Ordering::Relaxed);
        let events = until(&mut live, |e| state_of(e, MAIN) == Some(TaskState::Done));
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::Message { agent: Some(_), text } if text == "Hello there."
        )));
        assert_eq!(live.drafts(), [(MAIN, String::new())]);
    }

    #[test]
    fn tasks_beyond_the_limit_wait_their_turn_and_one_can_be_stopped_while_waiting() {
        let (mut live, open, seen) = held();
        live.poll();
        let mut events = Vec::new();
        for n in 1..=MAX_RUNNING + 2 {
            events.extend(ask(&mut live, format!("task {n}")));
        }
        let (fourth, fifth) = (MAX_RUNNING as u32 + 1, MAX_RUNNING as u32 + 2);
        assert_eq!(live.running(), MAX_RUNNING);
        assert_eq!(state_of(&events, 1), Some(TaskState::Working));
        assert_eq!(state_of(&events, fourth), Some(TaskState::Queued));
        assert_eq!(state_of(&events, fifth), Some(TaskState::Queued));
        assert!(live.busy());

        // One that is still waiting can be taken back.
        let stopped = live.handle(Command::Stop { task: fifth });
        assert_eq!(state_of(&stopped, fifth), Some(TaskState::Stopped));

        // When the running ones finish, the one that waited starts.
        open.store(true, Ordering::Relaxed);
        let events = until(&mut live, |e| state_of(e, fourth) == Some(TaskState::Done));
        assert!(events.iter().any(|e| e.task == Some(fourth)
            && matches!(&e.kind, EventKind::Message { agent: Some(_), text } if text == &format!("answer to: task {fourth}"))));
        // The stopped one never reached the model.
        assert!(
            !seen
                .lock()
                .unwrap()
                .iter()
                .flatten()
                .any(|m| m == &format!("task {fifth}"))
        );
        assert!(!live.busy());
    }

    #[test]
    fn a_running_task_can_be_stopped_and_a_rating_goes_to_the_task_named() {
        use crate::provider::testing::Queue;
        let root = std::env::temp_dir().join(format!("conscious-live-stop-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let core = Queue::new(
            "core",
            &[
                r#"CALL write_file {"path": "a.txt", "content": "x"}"#,
                "This reply is never asked for.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core)))).with_guard(Some(guard));
        live.poll();
        ask(&mut live, "write a file".into());
        // It stops at the write and waits for the user.
        until(&mut live, |e| {
            e.iter().any(|e| {
                matches!(
                    e.kind,
                    EventKind::ToolRequested {
                        needs_approval: true,
                        ..
                    }
                )
            })
        });
        let said = live.handle(Command::Stop { task: 1 });
        assert!(matches!(
            &said[..],
            [Tagged {
                task: Some(1),
                kind: EventKind::Alert { .. }
            }]
        ));
        // The waiting call is answered no, and the task ends as stopped.
        let events = until(&mut live, |e| state_of(e, 1) == Some(TaskState::Stopped));
        assert!(events.iter().any(|e| matches!(
            e.kind,
            EventKind::ToolDecision {
                approved: false,
                ..
            }
        )));
        assert!(!root.join("a.txt").exists());
        assert!(!live.busy());

        // With one model nothing was routed, so there is nothing to rate,
        // and saying so is filed under the task that was named.
        let rated = live.handle(Command::Feedback {
            good: true,
            task: Some(1),
        });
        assert!(matches!(
            &rated[..],
            [Tagged { task: Some(1), kind: EventKind::Alert { text, .. } }] if text.contains("was not routed")
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_memory_is_found_by_meaning_once_the_local_model_is_ready() {
        use crate::knowledge::embed::testing::Ideas;
        let mut kb = KnowledgeBase::in_memory();
        kb.add_note(
            crate::knowledge::MEMORY_SPACE,
            "user",
            "The boat is kept in the old harbour.",
        );
        kb.commit().unwrap();
        let kb = Arc::new(Mutex::new(kb));
        let loader: MeaningLoader =
            Box::new(|| Ok(Arc::new(Ideas) as Arc<dyn crate::knowledge::Embedder>));
        let mut live =
            Live::with_knowledge(single(Fixed("ok")), Some(kb.clone())).with_meaning(Some(loader));
        // The model is loaded in the background and says when it is ready.
        let events = until(&mut live, |e| {
            e.iter().any(|e| matches!(
                &e.kind,
                EventKind::Alert { text, .. } if text.starts_with("Lookup by meaning is on (1 newly read")
            ))
        });
        assert!(events.iter().all(|e| e.task.is_none()));

        // A question in other words is now given the memory.
        let sent = ask(&mut live, "which vessel do we sail".into());
        let found = sent.iter().find_map(|e| match &e.kind {
            EventKind::KnowledgeLookup { picks, limits, .. } => {
                Some((picks.clone(), limits.clone()))
            }
            _ => None,
        });
        let (picks, limits) = found.expect("the lookup found it");
        assert!(picks[0].text.contains("old harbour"));
        assert!(picks[0].matched[0].starts_with("≈ meaning"));
        assert!(limits.starts_with("by words and by meaning"));

        // If the model cannot be had, that is said and lookups go on by words.
        let failing: MeaningLoader = Box::new(|| Err("no network".into()));
        let mut live =
            Live::with_knowledge(single(Fixed("ok")), Some(kb)).with_meaning(Some(failing));
        until(&mut live, |e| {
            e.iter().any(|e| matches!(
                &e.kind,
                EventKind::Alert { severity: Severity::Warn, text } if text.contains("no network")
            ))
        });
    }

    #[test]
    fn a_rating_counts_for_the_lessons_a_task_was_given_and_the_ones_it_wrote() {
        let mut kb = KnowledgeBase::in_memory();
        let useful = match kb.add_lesson(90, "the launch checklist lives in the wiki", true) {
            crate::knowledge::Kept::New(id) => id,
            other => panic!("{other:?}"),
        };
        kb.commit().unwrap();
        let kb = Arc::new(Mutex::new(kb));
        let mut live = Live::with_knowledge(single(Fixed("It is in the wiki.")), Some(kb.clone()));
        live.poll();

        // The lookup hands the task the lesson; that is on record.
        let sent = ask(&mut live, "where is the launch checklist".into());
        assert!(sent.iter().any(|e| matches!(
            &e.kind,
            EventKind::KnowledgeLookup { picks, .. } if picks.iter().any(|p| p.text.contains("wiki"))
        )));
        assert_eq!(*live.tasks[0].given.lock().unwrap(), [useful]);
        assert_eq!(kb.lock().unwrap().node(useful).unwrap().record.given, 1);
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));

        // Rated good: the lesson it relied on is confirmed, once.
        let rated = live.handle(Command::Feedback {
            good: true,
            task: Some(1),
        });
        assert!(rated.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { severity: Severity::Success, text }
                if text == "Confirmed by this rating: Lesson: the launch checklist lives in the wiki"
        )));
        let again = live.handle(Command::Feedback {
            good: true,
            task: Some(1),
        });
        assert!(!again.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { text, .. } if text.starts_with("Confirmed")
        )));
        {
            let kb = kb.lock().unwrap();
            let node = kb.node(useful).unwrap();
            assert_eq!(node.provenance.trust, crate::knowledge::Trust::Experience);
            assert_eq!((node.record.good, node.record.bad), (1, 0));
        }

        // A lesson a task wrote is retired when that task is rated bad.
        kb.lock()
            .unwrap()
            .add_lesson(2, "always delete the lock file first", true);
        ask(&mut live, "tidy the build folder".into());
        until(&mut live, |e| state_of(e, 2) == Some(TaskState::Done));
        let rated = live.handle(Command::Feedback {
            good: false,
            task: Some(2),
        });
        assert!(rated.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { severity: Severity::Warn, text }
                if text.ends_with("Lesson: always delete the lock file first")
        )));
        assert!(
            kb.lock()
                .unwrap()
                .lessons_report()
                .contains("retired (rated bad 1 time")
        );
    }

    #[test]
    fn idle_time_is_used_to_merge_repeated_memories_and_read_unread_ones() {
        let mut kb = KnowledgeBase::in_memory();
        // The same exchange remembered twice, as said (no model had read it).
        let first = kb.remember("the gate code is 4417", "Noted.", None);
        let second = kb.remember("the gate code is 4417", "Noted.", None);
        kb.commit().unwrap();
        let kb = Arc::new(Mutex::new(kb));
        let reader = Scripted::ok(
            "core",
            "KIND: fact\nTITLE: Gate code\nSUMMARY: The gate code is 4417.\nTERMS: gate code",
        );
        let mut live = Live::with_knowledge(single(reader), Some(kb.clone()));
        // Not while it has only just gone quiet.
        live.tidy_after = Duration::from_secs(3600);
        live.poll();
        live.poll();
        assert!(!live.tidying);

        live.tidy_after = Duration::ZERO;
        let events = until(&mut live, |e| {
            e.iter().any(|e| matches!(
                &e.kind,
                EventKind::Alert { text, .. } if text.starts_with("While idle, went over memory")
            ))
        });
        // The repeat was merged into the newer one, and shown as that.
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::MemoryConsolidated { from, into, .. }
                if from == &[first as u32] && into.id == second as u32
        )));
        assert!(events.iter().all(|e| e.task.is_none()));
        until(&mut live, |_| true);
        {
            let kb = kb.lock().unwrap();
            assert!(kb.node(first).unwrap().retired.is_some());
            // What was left was read by the model and filed under its meaning.
            assert_eq!(kb.node(second).unwrap().title, "Gate code");
            assert!(kb.undigested().is_empty());
        }
        // It does not run again straight away, and never while a task runs.
        let deadline = Instant::now() + Duration::from_millis(300);
        while live.tidying {
            live.poll();
            assert!(Instant::now() < deadline);
        }
        assert!(live.poll().is_empty());
        assert!(!live.tidying);
    }

    #[test]
    fn a_run_whose_own_check_passes_confirms_the_lesson_it_used_without_a_rating() {
        use crate::provider::testing::Queue;
        let root =
            std::env::temp_dir().join(format!("conscious-live-outcome-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("check.sh"), "exit 0\n").unwrap();
        let mut kb = KnowledgeBase::in_memory();
        let lesson = match kb.add_lesson(40, "run check.sh to verify the release folder", true) {
            crate::knowledge::Kept::New(id) => id,
            other => panic!("{other:?}"),
        };
        kb.commit().unwrap();
        let kb = Arc::new(Mutex::new(kb));
        let core = Queue::new(
            "core",
            &[
                r#"CALL bash {"command": "sh check.sh"}"#,
                "The release folder is verified.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live =
            Live::with_knowledge(Arc::new(Router::single(Arc::new(core))), Some(kb.clone()))
                .with_guard(Some(guard));
        live.poll();
        ask(&mut live, "verify the release folder with check.sh".into());
        // Approve the command if the guardrails ask (they do without a sandbox).
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while state_of(&events, 1) != Some(TaskState::Done) {
            assert!(Instant::now() < deadline, "{events:?}");
            let new = live.poll();
            for e in &new {
                if let EventKind::ToolRequested {
                    id,
                    needs_approval: true,
                    ..
                } = &e.kind
                {
                    live.handle(Command::Decide {
                        id: *id,
                        approve: true,
                    });
                }
            }
            events.extend(new);
            thread::sleep(Duration::from_millis(5));
        }
        // The task's state says what vouches for it.
        assert!(
            events.iter().any(|e| matches!(
                &e.kind,
                EventKind::TaskUpdate { state: TaskState::Done, note }
                    if note.ends_with("its last check passed (`sh check.sh`)")
            )),
            "{events:?}"
        );
        // And the lesson it was given is confirmed by that, with no rating.
        let confirmed = |e: &Tagged| {
            matches!(
                &e.kind,
                EventKind::Alert { severity: Severity::Success, text }
                    if text.starts_with("Confirmed, because this run's last check passed (`sh check.sh`)")
            )
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !events.iter().any(confirmed) {
            assert!(Instant::now() < deadline, "{events:?}");
            events.extend(live.poll());
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            events
                .iter()
                .filter(|e| confirmed(e))
                .all(|e| e.task == Some(1))
        );
        let kb = kb.lock().unwrap();
        let node = kb.node(lesson).unwrap();
        assert_eq!(node.provenance.trust, crate::knowledge::Trust::Experience);
        assert_eq!(
            (node.record.given, node.record.passed, node.record.good),
            (1, 1, 0)
        );
        drop(kb);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn idle_time_settles_near_repeats_with_a_model_and_summarises_what_has_gathered() {
        use crate::knowledge::embed::testing::Ideas;
        let mut kb = KnowledgeBase::in_memory();
        kb.set_embedder(Arc::new(Ideas));
        let space = crate::knowledge::MEMORY_SPACE;
        // Two notes too alike to be sure of.
        let older = kb.add_note(
            space,
            "conversation",
            "money money money money money money money rain rain rain\n\nKind: fact",
        );
        let newer = kb.add_note(
            space,
            "conversation",
            "money money money money money money money rain rain rain rain rain rain rain\n\nKind: fact",
        );
        // Four memories on one subject.
        let trip: Vec<NodeId> = [
            "boat",
            "boat ship cat",
            "boat vessel money",
            "boat sail rain",
        ]
        .iter()
        .map(|said| {
            kb.add_note(
                space,
                "conversation",
                &format!("Trip: {said}.\n\nKind: event"),
            )
        })
        .collect();
        kb.commit().unwrap();
        kb.embed_some(20).unwrap();
        let kb = Arc::new(Mutex::new(kb));
        // One reply that answers both questions the pass asks.
        let reader = Scripted::ok(
            "core",
            "SAME\nTITLE: The boat trip\nSUMMARY: The boat sails from the harbour.",
        );
        let mut live = Live::with_knowledge(single(reader), Some(kb.clone()));
        live.tidy_after = Duration::ZERO;
        let events = until(&mut live, |e| {
            e.iter().any(|e| matches!(
                &e.kind,
                EventKind::Alert { text, .. } if text.starts_with("While idle, went over memory")
            ))
        });
        let said = events
            .iter()
            .find_map(|e| match &e.kind {
                EventKind::Alert { text, .. } if text.starts_with("While idle") => {
                    Some(text.clone())
                }
                _ => None,
            })
            .unwrap();
        assert!(said.contains("wrote 1 summary"), "{said}");
        assert!(said.contains("merged 1 memory"), "{said}");
        // Each is shown as a consolidation: the pair into the newer note,
        // the group into its summary.
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::MemoryConsolidated { from, into, .. }
                if from == &[older as u32] && into.id == newer as u32
        )));
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::MemoryConsolidated { from, into, .. }
                if from.len() == 4 && into.label == "summary: The boat trip"
        )));
        let kb = kb.lock().unwrap();
        assert!(kb.node(older).unwrap().retired.is_some());
        // The memories behind the summary are all still there.
        assert!(
            trip.iter()
                .all(|id| kb.node(*id).unwrap().retired.is_none())
        );
        assert!(
            kb.memories()
                .iter()
                .any(|(_, title, _)| title == "The boat trip")
        );
    }

    #[test]
    fn an_exchange_with_nothing_in_it_is_not_filed_in_memory() {
        let kb = Arc::new(Mutex::new(KnowledgeBase::in_memory()));
        // The same model answers and then reads the exchange.
        let mut live = Live::with_knowledge(
            single(Scripted::ok("core", "KIND: nothing")),
            Some(kb.clone()),
        );
        live.poll();
        ask(&mut live, "thanks!".into());
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        // The reading comes back, and nothing is written.
        let deadline = Instant::now() + Duration::from_secs(5);
        while live.reading > 0 {
            assert!(Instant::now() < deadline);
            let events = live.poll();
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e.kind, EventKind::MemoryWrite { .. }))
            );
            thread::sleep(Duration::from_millis(5));
        }
        assert!(kb.lock().unwrap().memories().is_empty());
    }

    #[test]
    fn a_standing_rule_is_in_every_tasks_context_and_answers_for_every_task() {
        use crate::knowledge::Learned;
        let mut kb = KnowledgeBase::in_memory();
        let text = "Run the failed check again after a fix before saying it is fixed.";
        let rule = match kb.add_learned(1, Learned::Rule(text), true) {
            crate::knowledge::Kept::New(id) => id,
            other => panic!("{other:?}"),
        };
        kb.judge_outcome(1, &[], true);
        kb.add_learned(2, Learned::Rule(text), true);
        kb.commit().unwrap();
        let kb = Arc::new(Mutex::new(kb));
        let (mut live, open, seen) = held();
        live.knowledge = Some(kb.clone());
        live.poll();
        // A task about something else entirely is still given the rule.
        let sent = ask(&mut live, "what is a tide".into());
        let sections = sent
            .iter()
            .find_map(|e| match &e.kind {
                EventKind::Context { sections, .. } => Some(sections.clone()),
                _ => None,
            })
            .unwrap();
        assert!(sections.iter().any(|s| s.name == "rules"), "{sections:?}");
        assert_eq!(*live.tasks[0].given.lock().unwrap(), [rule]);
        let _ = seen;
        open.store(true, Ordering::Relaxed);
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        // So that task's rating counts for it.
        live.handle(Command::Feedback {
            good: true,
            task: Some(1),
        });
        assert_eq!(kb.lock().unwrap().node(rule).unwrap().record.good, 1);
    }

    #[test]
    fn ratings_move_the_limits_on_what_is_learned_and_the_timeline_says_so() {
        use crate::knowledge::Learned;
        let mut kb = KnowledgeBase::in_memory();
        let text = "Run the failed check again after a fix before saying it is fixed.";
        let rule = match kb.add_learned(1, Learned::Rule(text), true) {
            crate::knowledge::Kept::New(id) => id,
            other => panic!("{other:?}"),
        };
        kb.judge_outcome(1, &[], true);
        kb.add_learned(2, Learned::Rule(text), true);
        kb.commit().unwrap();
        assert_eq!(kb.standing_rule_ids(), [rule]);
        let kb = Arc::new(Mutex::new(kb));
        let mut live = Live::with_knowledge(single(Fixed("ok")), Some(kb.clone()));
        live.poll();

        // The rule stands in front of a task, and the user rates it bad:
        // the rule is retired, and rules need more behind them from now on.
        ask(&mut live, "what is a tide".into());
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        let rated = live.handle(Command::Feedback {
            good: false,
            task: Some(1),
        });
        assert!(rated.iter().any(|e| matches!(
            &e.kind,
            EventKind::Alert { text, .. } if text.starts_with("Retired by this rating")
        )));
        let said: Vec<&String> = rated
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::Alert { text, .. } if text.starts_with("Learned from what happened") => {
                    Some(text)
                }
                _ => None,
            })
            .collect();
        assert_eq!(said.len(), 1, "{rated:?}");
        assert!(said[0].contains("is now 4 (was 3)"), "{}", said[0]);
        assert!(said[0].contains("turned out badly"));
        let report = kb.lock().unwrap().tuning_report(0);
        assert!(report.contains("3 → 4"), "{report}");
        assert!(
            report.contains("tasks with standing rules: 0 went well, 1 badly"),
            "{report}"
        );
    }

    /// A model for the whole criteria loop: it answers each kind of call by
    /// what the call asks for.
    struct Judge {
        /// What the checker says of the criteria.
        verdicts: &'static str,
        seen: Arc<Mutex<Vec<String>>>,
    }

    impl Provider for Judge {
        fn label(&self) -> String {
            "judge".into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            _: &mut dyn FnMut(&str),
        ) -> Result<Response, crate::provider::ProviderError> {
            self.seen.lock().unwrap().push(req.system.clone());
            let text = if req.system.starts_with("You set out how success") {
                "- The answer names the tide.\n- The answer is one sentence."
            } else if req.system.starts_with("You check an agent's finished work") {
                self.verdicts
            } else if req.system.starts_with("An agent finished a task") {
                "INTENT: The user wants the reason behind an answer, not only the answer.\nWHEN: explaining something"
            } else if req.system.starts_with("You file memories") {
                "KIND: nothing"
            } else {
                "A spring tide."
            };
            Ok(Response {
                text: text.into(),
                tool_calls: Vec::new(),
                usage: crate::provider::Usage {
                    input: 5,
                    cached: 0,
                    output: 3,
                },
                latency: Duration::from_millis(1),
                model: "judge".into(),
                warnings: Vec::new(),
            })
        }
    }

    #[allow(clippy::type_complexity)]
    fn judged(
        verdicts: &'static str,
    ) -> (
        Live,
        Arc<Mutex<KnowledgeBase>>,
        Arc<Mutex<Vec<String>>>,
        std::path::PathBuf,
    ) {
        let root = std::env::temp_dir().join(format!(
            "conscious-live-goal-{}-{:?}",
            std::process::id(),
            thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let kb = Arc::new(Mutex::new(KnowledgeBase::in_memory()));
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let live = Live::with_knowledge(
            single(Judge {
                verdicts,
                seen: seen.clone(),
            }),
            Some(kb.clone()),
        )
        .with_guard(Some(Arc::new(Guard::new(&root).unwrap())))
        .with_goals(true);
        (live, kb, seen, root)
    }

    fn alerts(events: &[Tagged]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match &e.kind {
                EventKind::Alert { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_task_is_given_success_criteria_works_to_them_and_is_checked_against_them() {
        let (mut live, _, seen, root) =
            judged("1. MET - it says spring tide\n2. MET - one sentence");
        live.poll();
        ask(&mut live, "which tide is highest".into());
        let events = until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        // The criteria are set before the work and shown as the task's goal.
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::GoalSet { criteria, .. }
                if criteria == &["The answer names the tide.", "The answer is one sentence."]
        )));
        // The model that did the work was given them.
        assert!(seen.lock().unwrap().iter().any(|system| {
            system.contains("How success on this task will be judged")
                && system.contains("- The answer names the tide.")
        }));
        // And the finished work was checked against them.
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::CriteriaChecked { checked }
                if checked.len() == 2 && checked[0].evidence == "it says spring tide"
        )));
        assert!(events.iter().any(|e| matches!(
            &e.kind,
            EventKind::TaskUpdate { state: TaskState::Done, note }
                if note.contains("2 of 2 success criteria met")
        )));
        assert_eq!(live.tasks[0].criteria.len(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_response_that_disagrees_with_the_criteria_starts_the_learner_and_one_that_agrees_does_not()
    {
        // All met, and the user rates it good: the criteria were right.
        let (mut live, kb, _, root) = judged("1. MET - yes\n2. MET - yes");
        live.poll();
        ask(&mut live, "which tide is highest".into());
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        let rated = live.handle(Command::Feedback {
            good: true,
            task: Some(1),
        });
        assert!(
            !alerts(&rated).iter().any(|a| a.contains("criteri")),
            "{rated:?}"
        );
        assert_eq!(live.learning, 0);
        assert!(
            kb.lock()
                .unwrap()
                .tuning_report(0)
                .contains("the way you did 1 time, differently 0 times")
        );
        let _ = std::fs::remove_dir_all(&root);

        // All met, and the user rates it bad: they measured the wrong
        // thing, and the learner works out what was wanted.
        let (mut live, kb, _, root) = judged("1. MET - yes\n2. MET - yes");
        live.poll();
        ask(&mut live, "which tide is highest".into());
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        let rated = live.handle(Command::Feedback {
            good: false,
            task: Some(1),
        });
        assert!(
            alerts(&rated)
                .iter()
                .any(|a| a
                    .starts_with("Every success criterion was met, yet you rated the result bad")),
            "{rated:?}"
        );
        let events = until(&mut live, |e| {
            alerts(e)
                .iter()
                .any(|a| a.starts_with("Learned what you meant"))
        });
        let said = alerts(&events);
        assert!(said.iter().any(|a| a.contains(
            "The user wants the reason behind an answer, not only the answer. (applies when: explaining something)"
        )), "{said:?}");
        assert!(events.iter().all(|e| e.task == Some(1)));
        {
            let kb = kb.lock().unwrap();
            // Kept as the agent's own note on what the user wants, to be
            // confirmed or retired like any lesson.
            assert!(
                kb.lessons_report()
                    .contains("what the user wants, not yet confirmed")
            );
            assert!(kb.tuning_report(0).contains("differently 1 time"));
        }
        // The next task of that kind is given it when its criteria are set.
        let sent = ask(
            &mut live,
            "explaining something: why are tides higher at full moon".into(),
        );
        assert!(
            sent.iter().any(|e| matches!(
                &e.kind,
                EventKind::KnowledgeLookup { picks, .. }
                    if picks.iter().any(|p| p.text.starts_with("Intent: The user wants the reason"))
            )),
            "{sent:?}"
        );
        let _ = std::fs::remove_dir_all(&root);

        // Something not met, and the user rates it good: they asked too much.
        let (mut live, _, _, root) = judged("1. MET - yes\n2. NOT MET - three sentences");
        live.poll();
        ask(&mut live, "which tide is highest".into());
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        let rated = live.handle(Command::Feedback {
            good: true,
            task: Some(1),
        });
        assert!(alerts(&rated).iter().any(|a| {
            a.starts_with("A success criterion was not met, yet you rated the result good")
        }));
        // Not met and rated bad is agreement: the work fell short, not the criteria.
        let (mut live2, _, _, root2) = judged("1. MET - yes\n2. NOT MET - three sentences");
        live2.poll();
        ask(&mut live2, "which tide is highest".into());
        until(&mut live2, |e| state_of(e, 1) == Some(TaskState::Done));
        let rated = live2.handle(Command::Feedback {
            good: false,
            task: Some(1),
        });
        assert!(!alerts(&rated).iter().any(|a| a.contains("criteri")));
        assert_eq!(live2.learning, 0);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&root2);
    }

    #[test]
    fn a_reply_that_corrects_the_result_is_learned_from_too() {
        let (mut live, kb, _, root) = judged("1. MET - yes\n2. MET - yes");
        live.poll();
        ask(&mut live, "which tide is highest".into());
        until(&mut live, |e| state_of(e, 1) == Some(TaskState::Done));
        // The user writes back in the same task.
        live.handle(Command::Reply {
            task: 1,
            text: "but why? you did not say why".into(),
        });
        let events = until(&mut live, |e| {
            alerts(e)
                .iter()
                .any(|a| a.starts_with("Learned what you meant"))
                && state_of(e, 1) == Some(TaskState::Done)
        });
        assert!(!alerts(&events).is_empty());
        let kb = kb.lock().unwrap();
        assert!(
            kb.lessons_report()
                .contains("Intent: The user wants the reason behind an answer")
        );
        assert!(kb.tuning_report(0).contains("differently 1 time"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A small source tree standing in for the harness's own, and an empty
    /// folder for its self-update work.
    fn own_source(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "conscious-live-selfupdate-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let (source, home) = (base.join("source"), base.join("home"));
        std::fs::create_dir_all(source.join("src")).unwrap();
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(source.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        std::fs::write(source.join("src/main.rs"), "fn main() {}\n").unwrap();
        (source, home)
    }

    #[test]
    fn a_task_can_ask_for_the_harness_to_be_changed_and_it_runs_as_a_task_of_its_own() {
        use crate::provider::testing::Queue;
        let (source, home) = own_source("ask");
        let root = home.join("project");
        std::fs::create_dir_all(&root).unwrap();
        let core = Queue::new(
            "core",
            &[
                r#"CALL self_update {"problem": "there is no tool to read a PDF", "change": "add a tool that extracts text from a PDF"}"#,
                "I could not read the PDF; a change to the harness was started for it.",
                // The self-update's own run: it changes nothing.
                "I looked and changed nothing.",
                "Still nothing.",
                "Nothing again.",
            ],
        );
        let guard = Arc::new(Guard::new(&root).unwrap());
        let mut live = Live::new(Arc::new(Router::single(Arc::new(core))))
            .with_guard(Some(guard))
            .with_self_update(true);
        live.update_dirs = Some((source.clone(), home.clone()));
        live.poll();
        ask(&mut live, "summarise report.pdf".into());

        // Asking for it is a high-risk request the user must approve.
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(10);
        while state_of(&events, 2).is_none_or(|s| s.is_open()) {
            assert!(Instant::now() < deadline, "{events:?}");
            let new = live.poll();
            for e in &new {
                if let EventKind::ToolRequested {
                    id,
                    needs_approval: true,
                    tool,
                    risk,
                    ..
                } = &e.kind
                {
                    assert_eq!(tool, "self_update");
                    assert_eq!(*risk, crate::event::Risk::High);
                    live.handle(Command::Decide {
                        id: *id,
                        approve: true,
                    });
                }
            }
            events.extend(new);
            thread::sleep(Duration::from_millis(5));
        }
        // The asking task was told it cannot use the change, and finished.
        assert!(events.iter().any(|e| e.task == Some(1)
            && matches!(&e.kind, EventKind::ToolFinished { ok: true, output, .. }
                if output.starts_with("A self-update has been asked for."))));
        // The change runs as its own task, marked as the harness's own.
        assert!(events.iter().any(|e| e.task == Some(2)
            && matches!(&e.kind, EventKind::TaskOpened { title, origin }
                if title.starts_with("Change the harness itself: add a tool")
                    && origin == "the harness itself")));
        // It was not adopted (no sandbox here, or nothing was changed), and
        // says so. Either way the real source is untouched and nothing
        // restarts.
        let said = events.iter().find_map(|e| match &e.kind {
            EventKind::Message {
                agent: Some(_),
                text,
            } if e.task == Some(2) => Some(text.clone()),
            _ => None,
        });
        let said = said.expect("the self-update reports what came of it");
        assert!(
            said.starts_with("The harness's code was **not** changed"),
            "{said}"
        );
        assert_eq!(state_of(&events, 2), Some(TaskState::Failed));
        until(&mut live, |_| true);
        assert!(!live.updating);
        assert!(live.take_restart().is_none());
        assert_eq!(
            std::fs::read_to_string(source.join("src/main.rs")).unwrap(),
            "fn main() {}\n"
        );
        // The attempt is on record.
        let history = crate::selfupdate::History::load(&home);
        assert_eq!(history.attempts.len(), 1);
        assert_eq!(history.attempts[0].outcome, "rejected");
        assert_eq!(
            history.attempts[0].problem,
            "there is no tool to read a PDF"
        );
        assert!(history.last_adopted().is_none());
        let _ = std::fs::remove_dir_all(home.parent().unwrap());
    }

    #[test]
    fn a_self_update_is_not_started_when_switched_off_or_while_one_is_under_way() {
        let (source, home) = own_source("off");
        let mut live = Live::new(single(Fixed("ok")));
        live.update_dirs = Some((source, home.clone()));
        let refused = live.start_self_update("p".into(), "c".into());
        assert!(matches!(
            &refused[..],
            [Tagged { task: None, kind: EventKind::Alert { text, .. } }] if text.contains("switched off")
        ));
        live.self_update = true;
        live.updating = true;
        let refused = live.start_self_update("p".into(), "c".into());
        assert!(alerts(&refused)[0].contains("already under way"));
        // A restart that is due waits until nothing is running.
        live.updating = false;
        live.restart_due("self-update #1 was adopted");
        live.reading = 1;
        assert!(live.take_restart().is_none());
        live.reading = 0;
        assert_eq!(
            live.take_restart().as_deref(),
            Some("self-update #1 was adopted")
        );
        assert!(live.take_restart().is_none(), "given once");
        let _ = std::fs::remove_dir_all(home.parent().unwrap());
    }
}
