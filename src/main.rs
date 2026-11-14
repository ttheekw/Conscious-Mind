mod agent;
mod app;
mod claims;
mod daemon;
mod demo;
mod event;
mod goal;
mod guard;
mod knowledge;
mod learn;
mod live;
mod markdown;
mod policy;
mod preflight;
mod provider;
mod research;
mod router;
mod schedule;
mod secrets;
mod selfupdate;
mod skills;
#[cfg(test)]
mod snapshot;
mod telegram;
mod theme;
mod timeline;
mod tools;
mod ui;
mod web;

use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{
    self as term, DisableMouseCapture, EnableMouseCapture, KeyEventKind,
};
use ratatui::crossterm::execute;

use app::App;
use demo::{Command, Demo};
use event::{EventKind, Tagged};
use live::Live;
use provider::{Message, Provider, Request};
use router::Router;

/// How often state advances (animations, the demo script, polling the model).
const TICK: Duration = Duration::from_millis(80);
/// How often the screen is redrawn. Faster than `TICK` so that anything
/// timed by the clock, such as the scrolling charts, moves smoothly.
const FRAME: Duration = Duration::from_millis(33);

const USAGE: &str = r#"conscious                          start: everything on, nothing else to run first
conscious --model SPEC             the same, with a core model of your choosing
conscious --demo                   the scripted demo, with no model
conscious ask PROMPT               send one prompt and print the reply
conscious models --model SPEC      list the models in the pool and what routing has learned
conscious feedback good|bad        rate the last `ask` answer, so routing learns from it
conscious policy                   show the routing policy version and its history
conscious policy show              print the active policy script
conscious policy rollback          go back to the previous policy version
conscious policy propose FILE      put a script through the regression gate; adopted if it passes
conscious policy review --model SPEC   have the core review the policy now
conscious check [--call]           what this machine has and which model would be used;
                                   --call also sends each ready backend one real request
conscious schedule                 the jobs the agent has set to start by themselves
conscious schedule cancel ID       remove one of them
conscious status                   whether conscious is running in the background, and on what
conscious stop [--force]           end the background process (--force: even mid-task)
conscious --no-daemon              run everything inside this window only; it ends when you quit
conscious --no-semantic            look things up by words and links only, without the local meaning model
conscious --no-criteria            do not set success criteria for each task (saves two small model calls a task)
conscious --no-self-update         never let it change its own source code
conscious selfupdate               what it has tried to change in its own code, and what came of each
conscious selfupdate rollback      take the newest adopted change to its own code back
conscious secrets [remove NAME]    the names of the secrets you have stored (enter one with K in the TUI)
conscious skills                   the procedures the agent has written for itself
conscious lessons                  the lessons it has drawn, whether each is confirmed, and how it has fared
conscious tuning                   the limits on what it learns: what each is now, and why it moved
conscious knowledge ...            the knowledge graph: ingest, search, add, stats (run it for help)

conscious runs in the background and the screen attaches to it: plain
`conscious` starts the background process if it is not running, then shows
it. Quitting the screen (q) leaves it working, so a task carries on and
scheduled jobs start with no screen open. It ends by itself once nothing is
running and nothing is scheduled.

There is one main conversation that keeps going (0). What you say goes to it;
it answers there, and when something is a piece of work of its own it opens a
branch: a tab that works by itself and is merged back into main when it ends.
Several branches run at once (b lists them). Routines are jobs it set to
repeat; each run is a branch too. A task that needs facts conscious does not have is
shown as paused for knowledge research: it searches the web, keeps what it
reads in the local knowledge graph, and carries on.

Plain `conscious` first checks the machine and picks a core model by itself (the
first that is ready: Claude Code signed in, else Codex signed in, else a local
server), brings the knowledge graph up to date with the project in the
current folder, looks every message up in it, remembers every exchange, routes
each task by complexity and learns from your ratings.

SPEC is backend[:model] and names the core model. CONSCIOUS_MODEL sets the default.
For each task the core rates its complexity (1-5) and learned knowledge picks
the cheapest available model expected to handle that level; feedback updates
that knowledge. The choice is made by an interpreted script that the core
may rewrite; a rewrite is adopted only if it passes a regression suite and is
rolled back if it misbehaves. --no-route sends everything to the core.
Each message is also looked up in the knowledge graph and the passages found
are given to the model as reference data; --no-knowledge turns that off. Only the core is
fixed: the other models are the latest of each family that the installed
backends (Claude Code, Codex, a local server) offer right now.
~/.config/conscious/models.json replaces that pool:
  [{"spec": "openrouter:<model-id>", "good_for": "what it is", "tier": 1}]
"#;

/// How long to wait for a quick answer from the background process.
const QUICK: Duration = Duration::from_secs(5);
/// How long the background process may take to start: it checks the
/// machine, asks the backends for their models and syncs the project.
const START_WAIT: Duration = Duration::from_secs(180);
/// How often the screen reads the local time zone again (daylight saving).
const ZONE_EVERY: Duration = Duration::from_secs(600);

/// Where the TUI's events come from.
enum Feed {
    Demo(Demo),
    /// The harness inside this process (`--no-daemon`).
    Live(Box<Live>),
    /// The harness in the background process; this is the normal case.
    Remote(daemon::Remote, daemon::Status),
}

/// What the screen shows before anything new happens.
#[derive(Default)]
struct Start {
    /// Earlier sessions: history, nothing in motion.
    before: Vec<timeline::Recorded>,
    /// When attaching to a running harness: what has happened in it since
    /// it started, and what it remembers now.
    since: Vec<timeline::Recorded>,
    memory: Vec<event::MemoryNode>,
    /// Things to say once that concern this screen only.
    notices: Vec<String>,
}

impl Feed {
    fn poll(&mut self) -> Vec<Tagged> {
        match self {
            Feed::Demo(d) => d.poll().into_iter().map(Tagged::system).collect(),
            Feed::Live(l) => l.poll(),
            Feed::Remote(r, _) => r.poll(),
        }
    }

    fn handle(&mut self, cmd: Command) -> Vec<Tagged> {
        match self {
            Feed::Demo(d) => d.handle(cmd).into_iter().map(Tagged::system).collect(),
            Feed::Live(l) => l.handle(cmd),
            Feed::Remote(r, _) => r.handle(cmd),
        }
    }

    /// The replies being written right now that have changed.
    fn drafts(&mut self) -> Vec<(u32, String)> {
        match self {
            Feed::Demo(_) => Vec::new(),
            Feed::Live(l) => l.drafts(),
            Feed::Remote(r, _) => r.drafts(),
        }
    }

    fn throughput(&mut self) -> u64 {
        match self {
            Feed::Demo(d) => d.throughput(),
            Feed::Live(l) => l.throughput(),
            Feed::Remote(r, _) => r.throughput(),
        }
    }

    /// Starts the conversation afresh, for the restart command.
    fn restart(&mut self) {
        match self {
            Feed::Demo(d) => *d = Demo::new(),
            Feed::Live(l) => **l = l.restarted(),
            Feed::Remote(r, _) => {
                r.handle(Command::Restart);
            }
        }
    }

    fn new_app(&self, with_boot: bool) -> App {
        match self {
            // The boot screen describes the demo's pretend model.
            Feed::Demo(_) => App::new(with_boot),
            Feed::Live(l) => {
                let mut app = App::new(false);
                app.live = true;
                let core = l.router().core();
                app.last_model = core.label();
                app.window = core.context_window();
                app
            }
            Feed::Remote(_, status) => {
                let mut app = App::new(false);
                app.live = true;
                app.last_model = status.core.clone();
                app.window = status.window;
                app
            }
        }
    }
}

struct Args {
    ask: bool,
    list: bool,
    feedback: bool,
    policy: bool,
    route: bool,
    knowledge: bool,
    demo: bool,
    /// Find passages by meaning as well as by words (a local model).
    semantic: bool,
    /// Set success criteria for each task and learn from how the user
    /// responds to the result.
    goals: bool,
    /// Let the harness rewrite its own source when a task asks for it.
    self_update: bool,
    /// Be the background process (started by the screen, not by hand).
    daemon: bool,
    /// Run the harness inside this process instead of in the background.
    local: bool,
    model: Option<String>,
    rest: Vec<String>,
}

impl Args {
    /// Whether a subcommand has been seen already.
    fn command(&self) -> bool {
        self.ask || self.list || self.feedback || self.policy || self.daemon
    }
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        ask: false,
        list: false,
        feedback: false,
        policy: false,
        route: true,
        knowledge: true,
        demo: false,
        semantic: true,
        goals: true,
        self_update: true,
        daemon: false,
        local: false,
        model: std::env::var("CONSCIOUS_MODEL")
            .ok()
            .filter(|m| !m.is_empty()),
        rest: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--model" | "-m" => {
                args.model = Some(it.next().ok_or("--model needs a value")?);
            }
            "--no-route" => args.route = false,
            "--no-knowledge" => args.knowledge = false,
            "--no-semantic" => args.semantic = false,
            "--no-criteria" => args.goals = false,
            "--no-self-update" => args.self_update = false,
            "--demo" => args.demo = true,
            "--no-daemon" => args.local = true,
            "daemon" if !args.command() => args.daemon = true,
            "ask" if !args.command() => args.ask = true,
            "models" if !args.command() => args.list = true,
            "feedback" if !args.command() => args.feedback = true,
            "policy" if !args.command() => args.policy = true,
            _ if a.starts_with("--model=") => args.model = Some(a["--model=".len()..].to_string()),
            _ if a.starts_with('-') && !args.ask => return Err(format!("unknown option `{a}`")),
            _ if args.list => return Err(format!("unexpected argument `{a}`")),
            _ => args.rest.push(a),
        }
    }
    if !args.ask && !args.feedback && !args.policy && !args.rest.is_empty() {
        return Err(format!("unexpected argument `{}`", args.rest[0]));
    }
    Ok(args)
}

fn main() -> ExitCode {
    // The knowledge store has its own options, so it parses its own arguments.
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().is_some_and(|a| a == "knowledge") {
        return knowledge::cli::run(&raw[1..]);
    }
    if raw.first().is_some_and(|a| a == "secrets") {
        return secrets_command(&raw[1..]);
    }
    if raw.first().is_some_and(|a| a == "skills") {
        match skills::Skills::default_dir() {
            Some(dir) => println!("{}", skills::Skills::load(&dir).report().trim_end()),
            None => eprintln!("error: no home directory to keep skills in"),
        }
        return ExitCode::SUCCESS;
    }
    if raw.first().is_some_and(|a| a == "lessons") {
        let opened = knowledge::KnowledgeBase::default_dir()
            .ok_or_else(|| "no home directory to keep memory in".to_string())
            .and_then(|dir| knowledge::KnowledgeBase::open(&dir));
        return match opened {
            Ok(kb) => {
                println!("{}", kb.lessons_report());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if raw.first().is_some_and(|a| a == "selfupdate") {
        return selfupdate_command(&raw[1..]);
    }
    if raw.first().is_some_and(|a| a == "tuning") {
        let opened = knowledge::KnowledgeBase::default_dir()
            .ok_or_else(|| "no home directory to keep memory in".to_string())
            .and_then(|dir| knowledge::KnowledgeBase::open(&dir));
        return match opened {
            Ok(kb) => {
                println!("{}", kb.tuning_report(schedule::utc_offset()));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    if raw.first().is_some_and(|a| a == "schedule") {
        return schedule_command(&raw[1..]);
    }
    if raw.first().is_some_and(|a| a == "check") {
        return check(&raw[1..]);
    }
    if raw.first().is_some_and(|a| a == "status") {
        return status_command();
    }
    if raw.first().is_some_and(|a| a == "stop") {
        return stop_command(&raw[1..]);
    }
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}\n");
            }
            eprint!("{USAGE}\nBackends:\n{}\n", provider::SPEC_HELP);
            return if msg.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            };
        }
    };
    if args.feedback {
        return feedback(&args.rest);
    }
    let review_now = match (args.policy, args.rest.first().map(String::as_str)) {
        (false, _) => false,
        (true, Some("review")) => true,
        (true, action) => return policy_command(action, args.rest.get(1)),
    };
    if args.demo {
        return match tui(
            Feed::Demo(Demo::new()),
            timeline::Log::disabled(),
            Start::default(),
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    // The screen attaches to the harness running in the background, and
    // starts it first if need be.
    if !(args.ask || args.list || review_now || args.daemon || args.local) {
        match daemon::socket_path() {
            Some(path) => return screen(&args, &path),
            None => eprintln!(
                "note: no home directory for the background process to live in; \
                 running inside this window only"
            ),
        }
    }
    // The background process takes its socket before the slow part of the
    // start, so a second one started meanwhile finds it and gives way.
    let listener = if args.daemon {
        let taken = daemon::socket_path()
            .ok_or_else(|| "no home directory to keep the socket in".to_string())
            .and_then(|path| Ok((daemon::bind(&path)?, path)));
        match taken {
            Ok(taken) => Some(taken),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        None
    };
    // Check the machine first. Which model is the core follows from what
    // is found, by fixed rules; a start that cannot work stops here and
    // says what to do.
    let report = preflight::judge(&preflight::probe(args.model.clone()));
    let Some(spec) = report.core.clone() else {
        eprint!("{}", report.problems());
        eprintln!(
            "\n`conscious check` shows everything that was looked at; \
             `conscious --demo` runs the scripted demo without a model."
        );
        return ExitCode::from(2);
    };
    let core: Arc<dyn Provider> = match provider::from_spec(&spec) {
        Ok(p) => Arc::from(p),
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let router = if args.route {
        let (router, notes) = Router::discover(core);
        for n in notes {
            eprintln!("note: {n}");
        }
        router
    } else {
        Router::single(core)
    };

    // The knowledge store is opened and brought up to date with the project
    // in the current folder on every start, so there is no separate step.
    let kb = args
        .knowledge
        .then(knowledge::KnowledgeBase::default_dir)
        .flatten()
        .and_then(|dir| match knowledge::KnowledgeBase::open(&dir) {
            Ok(mut kb) => {
                if let Err(e) = knowledge::cli::sync_project(&mut kb, std::path::Path::new(".")) {
                    eprintln!("note: project not synced into the knowledge graph: {e}");
                }
                Some(Arc::new(std::sync::Mutex::new(kb)))
            }
            Err(e) => {
                eprintln!("note: knowledge store not loaded: {e}");
                None
            }
        });

    if args.ask || args.list || review_now {
        if args.list {
            print!("policy {}", router.policy_summary());
            for (i, c) in router.candidates().iter().enumerate() {
                let role = if i == 0 { "  (core)" } else { "" };
                println!("{}. {}{role}", i + 1, c.spec);
                if !c.good_for.is_empty() {
                    println!("   {}", c.good_for);
                }
                println!("   {}", router.profile(i));
            }
            return ExitCode::SUCCESS;
        }
        if review_now {
            return review(&router);
        }
        return ask(&router, kb.as_deref(), &args.rest.join(" "));
    }

    // The timeline is saved as it happens and read back here, so a restart
    // continues the last session instead of starting blank.
    let (log, recorded) = match timeline::default_path() {
        Some(path) => {
            let (log, recorded, note) = timeline::Log::open(&path);
            if let Some(n) = note {
                eprintln!("note: {n}");
            }
            (log, recorded)
        }
        None => (timeline::Log::disabled(), Vec::new()),
    };
    // The guardrails give the model its tools, confined to this folder.
    // What the user granted in earlier sessions is kept beside the other data.
    let granted = learn::default_path().and_then(|p| Some(p.parent()?.join("access.json")));
    let guard = match guard::Guard::open(std::path::Path::new("."), granted) {
        Ok(g) => Some(Arc::new(g)),
        Err(e) => {
            eprintln!("note: no tools this session: {e}");
            None
        }
    };
    // The procedures the agent has written for itself in earlier sessions.
    let skills = skills::Skills::default_dir()
        .map(|dir| Arc::new(std::sync::Mutex::new(skills::Skills::load(&dir))));
    // The jobs the agent has set for itself; due ones start once the
    // screen is up.
    let jobs = schedule::Schedule::default_path().map(|path| {
        let (jobs, note) = schedule::Schedule::load(&path, schedule::utc_offset());
        if let Some(n) = note {
            eprintln!("note: {n}");
        }
        Arc::new(std::sync::Mutex::new(jobs))
    });
    // The local model that finds passages by meaning. Loaded in the
    // background; the first time, its files are downloaded.
    let meaning: Option<live::MeaningLoader> = args.semantic.then(|| {
        Box::new(|| {
            let dir = knowledge::embed::Local::default_dir()
                .ok_or_else(|| "no home directory for its files".to_string())?;
            let model = knowledge::embed::Local::load(&dir)?;
            Ok(Arc::new(model) as Arc<dyn knowledge::Embedder>)
        }) as live::MeaningLoader
    });
    let mut live = Live::with_knowledge(Arc::new(router), kb)
        .with_guard(guard)
        .with_skills(skills)
        .with_schedule(jobs)
        .with_notices(report.notices())
        .with_meaning(meaning)
        .with_goals(args.goals)
        .with_self_update(args.self_update)
        .with_echo_first(true);
    live.resume(&recorded);
    if let Some((listener, path)) = listener {
        let identity = daemon::Identity {
            folder: std::env::current_dir()
                .map(|d| d.display().to_string())
                .unwrap_or_default(),
            spec,
        };
        eprintln!("serving in {} with {}", identity.folder, identity.spec);
        let why = daemon::serve(live, log, recorded, listener, identity);
        let _ = std::fs::remove_file(&path);
        eprintln!("ended: {why}");
        // A self-update was adopted: the new build takes this one's place.
        if why.starts_with("restart") {
            return restart_into_new_build(&path);
        }
        return ExitCode::SUCCESS;
    }
    let start = Start {
        before: recorded,
        ..Start::default()
    };
    match tui(Feed::Live(Box::new(live)), log, start) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// After a self-update: starts the new build as the background process
/// and waits for it to answer. If it does not come up, the update is taken
/// back (`selfupdate::rollback`) and the previous build is started again.
/// Fixed rules, no model: a build that cannot start never stays.
fn restart_into_new_build(path: &std::path::Path) -> ExitCode {
    // The same arguments this process was started with, after `daemon`.
    let args: Vec<String> = std::env::args().skip(2).collect();
    let log = daemon::log_path().unwrap_or_else(|| std::env::temp_dir().join("conscious.log"));
    let comes_up = |what: &str| -> bool {
        let mut child = match daemon::spawn(&args, &log) {
            Ok(child) => child,
            Err(e) => {
                eprintln!("{what} could not be started: {e}");
                return false;
            }
        };
        let deadline = Instant::now() + START_WAIT;
        loop {
            if daemon::status(path, START_WAIT).is_ok() {
                return true;
            }
            let ended = !matches!(child.try_wait(), Ok(None));
            if ended || Instant::now() >= deadline {
                let _ = child.kill();
                eprintln!(
                    "{what} did not come up:\n{}",
                    std::fs::read_to_string(&log).unwrap_or_default().trim_end()
                );
                return false;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    if comes_up("the new build") {
        return ExitCode::SUCCESS;
    }
    // The new build cannot start: back to the one before.
    match selfupdate::rollback(&selfupdate::Sandbox, "the new build did not come up") {
        Ok(said) => eprintln!("{said}"),
        Err(why) => eprintln!("the self-update could not be taken back: {why}"),
    }
    if comes_up("the previous build") {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// `conscious selfupdate [rollback]`: what the harness has tried to change
/// in its own code, or take the newest adopted change back.
fn selfupdate_command(rest: &[String]) -> ExitCode {
    let Some(home) = selfupdate::home() else {
        eprintln!("error: no home directory");
        return ExitCode::FAILURE;
    };
    match rest {
        [] => {
            println!(
                "{}",
                selfupdate::History::load(&home).report(schedule::utc_offset())
            );
            ExitCode::SUCCESS
        }
        [action] if action == "rollback" => {
            match selfupdate::rollback(&selfupdate::Sandbox, "taken back by hand") {
                Ok(said) => {
                    println!(
                        "{said}\nIf conscious is running in the background, `conscious stop` and \
                         start it again to run the restored build."
                    );
                    ExitCode::SUCCESS
                }
                Err(why) => {
                    eprintln!("error: {why}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("error: use `conscious selfupdate` or `conscious selfupdate rollback`");
            ExitCode::from(2)
        }
    }
}

/// Shows the harness that runs in the background, starting it first if it
/// is not running. Quitting the screen leaves it running.
fn screen(args: &Args, path: &std::path::Path) -> ExitCode {
    let mut notices = Vec::new();
    let mut running = daemon::status(path, START_WAIT).ok();
    // One left from an older build of the program is replaced, when that
    // interrupts nothing and nobody.
    if let Some(found) = &running
        && found.build != daemon::build_id()
    {
        if !found.busy && found.screens == 0 {
            let _ = daemon::ask(path, &daemon::ToDaemon::Stop { force: false }, QUICK);
            let deadline = Instant::now() + QUICK;
            while daemon::status(path, QUICK).is_ok() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(50));
            }
            running = None;
        } else {
            notices.push(
                "The background process was started by an older build of conscious and is in \
                 use, so it was left running. To use the new build: `conscious stop`, then \
                 start again."
                    .to_string(),
            );
        }
    }
    let started_here = running.is_none();
    if started_here {
        // Check the machine here, in view, so a start that cannot work
        // says why instead of failing out of sight.
        let report = preflight::judge(&preflight::probe(args.model.clone()));
        let Some(spec) = report.core.clone() else {
            eprint!("{}", report.problems());
            eprintln!(
                "\n`conscious check` shows everything that was looked at; \
                 `conscious --demo` runs the scripted demo without a model."
            );
            return ExitCode::from(2);
        };
        // The very first start checks, in view, that a model answers.
        let spec = match first_run(&report, spec) {
            Ok(spec) => spec,
            Err(code) => return code,
        };
        let mut pass = vec!["--model".to_string(), spec];
        if !args.route {
            pass.push("--no-route".into());
        }
        if !args.knowledge {
            pass.push("--no-knowledge".into());
        }
        if !args.semantic {
            pass.push("--no-semantic".into());
        }
        if !args.goals {
            pass.push("--no-criteria".into());
        }
        if !args.self_update {
            pass.push("--no-self-update".into());
        }
        let log = daemon::log_path().unwrap_or_else(|| std::env::temp_dir().join("conscious.log"));
        eprintln!("Starting conscious in the background...");
        let mut child = match daemon::spawn(&pass, &log) {
            Ok(child) => child,
            Err(e) => {
                eprintln!(
                    "error: the background process could not be started: {e}\n\
                     `conscious --no-daemon` runs everything inside this window instead."
                );
                return ExitCode::FAILURE;
            }
        };
        let deadline = Instant::now() + START_WAIT;
        loop {
            if daemon::status(path, START_WAIT).is_ok() {
                break;
            }
            let ended = !matches!(child.try_wait(), Ok(None));
            if ended || Instant::now() >= deadline {
                eprintln!(
                    "error: the background process {}. What it said:\n{}\n\
                     `conscious --no-daemon` runs everything inside this window instead.",
                    if ended {
                        "ended while starting"
                    } else {
                        "did not come up in time"
                    },
                    std::fs::read_to_string(&log).unwrap_or_default().trim_end()
                );
                return ExitCode::FAILURE;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let (remote, hello) = match daemon::Remote::attach(path) {
        Ok(attached) => attached,
        Err(e) => {
            eprintln!("error: could not attach to the background process: {e}");
            return ExitCode::FAILURE;
        }
    };
    let status = hello.status.clone();
    if !started_here {
        // It was already running, with choices made when it started.
        let here = std::env::current_dir()
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        if here != status.folder {
            notices.push(format!(
                "conscious is already running, and works in {}, not in this folder. To work \
                 here instead: quit, `conscious stop`, and start it from here.",
                status.folder
            ));
        }
        if let Some(named) = args.model.as_ref().filter(|m| **m != status.spec) {
            notices.push(format!(
                "conscious is already running with {} as its core model, not {named}. To \
                 change it: quit, `conscious stop`, and start again.",
                status.spec
            ));
        }
    }
    let start = Start {
        before: hello.before,
        since: hello.since,
        memory: hello.memory,
        notices,
    };
    if let Err(e) = tui(
        Feed::Remote(remote, status),
        timeline::Log::disabled(),
        start,
    ) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    // Say what is left behind, so a process nobody remembers is no surprise.
    // (A moment first, for the daemon to see that this screen has closed.)
    std::thread::sleep(Duration::from_millis(250));
    if let Ok(left) = daemon::status(path, QUICK) {
        let mut doing = Vec::new();
        if left.busy {
            doing.push("a task is still running".to_string());
        }
        if left.jobs > 0 {
            doing.push(format!(
                "{} scheduled job{}",
                left.jobs,
                if left.jobs == 1 { "" } else { "s" }
            ));
        }
        if left.screens > 0 {
            doing.push("another screen is attached".to_string());
        }
        if doing.is_empty() {
            println!(
                "Nothing is running or scheduled, so conscious ends by itself in a few seconds."
            );
        } else {
            println!(
                "conscious keeps running in the background ({}). `conscious` shows it again; \
                 `conscious stop` ends it.",
                doing.join(", ")
            );
        }
    }
    ExitCode::SUCCESS
}

/// `conscious status`: whether the background process is running.
fn status_command() -> ExitCode {
    let running = daemon::socket_path().and_then(|path| daemon::status(&path, QUICK).ok());
    match running {
        Some(s) => {
            println!(
                "conscious is running in the background (process {}).\n\
                 folder   {}\ncore     {}\ntask     {}\nscreens  {}\njobs     {}",
                s.pid,
                s.folder,
                s.spec,
                if s.busy {
                    "one is running"
                } else {
                    "none running"
                },
                s.screens,
                s.jobs
            );
            ExitCode::SUCCESS
        }
        None => {
            println!("conscious is not running. `conscious` starts it.");
            ExitCode::FAILURE
        }
    }
}

/// `conscious stop [--force]`: ends the background process.
fn stop_command(rest: &[String]) -> ExitCode {
    let force = match rest {
        [] => false,
        [flag] if flag == "--force" => true,
        _ => {
            eprintln!("error: use `conscious stop` or `conscious stop --force`");
            return ExitCode::from(2);
        }
    };
    let answer = daemon::socket_path()
        .ok_or_else(|| "no home directory".to_string())
        .and_then(|path| daemon::ask(&path, &daemon::ToDaemon::Stop { force }, QUICK));
    match answer {
        Ok(daemon::ToClient::Said { ok: true, text }) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Ok(daemon::ToClient::Said { ok: false, text }) => {
            eprintln!("not stopped: {text}");
            ExitCode::FAILURE
        }
        _ => {
            println!("conscious is not running.");
            ExitCode::SUCCESS
        }
    }
}

/// `conscious secrets` lists the names of the stored secrets, never their
/// values; `remove NAME` forgets one. Secrets are entered in the TUI (K),
/// not here, so that they are never typed into a shell history.
fn secrets_command(args: &[String]) -> ExitCode {
    let mut store = secrets::store().lock().unwrap();
    match args {
        [] => {
            let names = store.names();
            if names.is_empty() {
                println!(
                    "No secrets are stored. In conscious, press K to type one into its own window."
                );
            } else {
                println!("Stored secrets (names only; values are never shown):");
                for name in names {
                    println!("  {name}");
                }
            }
            ExitCode::SUCCESS
        }
        [action, name] if action == "remove" => match store.remove(name) {
            Ok(true) => {
                println!(
                    "Removed {name}. If conscious is running in the background, run \
                     `conscious stop` so that it forgets it too."
                );
                ExitCode::SUCCESS
            }
            Ok(false) => {
                eprintln!("error: there is no secret called {name}");
                ExitCode::FAILURE
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("usage: conscious secrets [remove NAME]");
            ExitCode::from(2)
        }
    }
}

/// The check a first start makes, in view, before anything is started:
/// says what conscious works through, shows what this machine has, and
/// sends one small real request to make sure a model really answers
/// (being signed in does not show a used-up allowance). The backends are
/// tried in a fixed order; the first that answers is the core. Returns
/// that core, or the exit code to stop with.
fn first_run(report: &preflight::Report, chosen: String) -> Result<String, ExitCode> {
    let Some(marker) = preflight::first_run_marker() else {
        return Ok(chosen);
    };
    if marker.exists() {
        return Ok(chosen);
    }
    println!("{}", report.welcome());
    for spec in report.first_run_order() {
        println!("Asking {spec} for a one-word reply, to see that it really answers...");
        match preflight::call(&spec) {
            Ok(how) => {
                println!("ok    {how}\n");
                let _ = std::fs::write(&marker, format!("{spec}\n"));
                return Ok(spec);
            }
            Err(why) => println!("FAIL  {spec}: {why}\n"),
        }
    }
    eprintln!(
        "error: no model answered, so conscious cannot start.\n\
         Sign in to Claude Code (run `claude`) or Codex (run `codex login`), then run \
         conscious again.\n\
         `conscious check --call` repeats this check; `conscious --demo` runs without a model."
    );
    Err(ExitCode::from(2))
}

/// `conscious schedule` lists the scheduled jobs; `cancel ID` removes one.
/// Needs no model. While the background process runs, it holds the
/// schedule, so it is the one asked.
fn schedule_command(rest: &[String]) -> ExitCode {
    let cancel = match rest {
        [] => None,
        [action, id] if action == "cancel" => match id.trim_start_matches('#').parse::<u32>() {
            Ok(id) => Some(id),
            Err(_) => {
                eprintln!("error: `{id}` is not a job number");
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("error: use `conscious schedule` or `conscious schedule cancel ID`");
            return ExitCode::from(2);
        }
    };
    let asked = daemon::socket_path()
        .and_then(|path| daemon::ask(&path, &daemon::ToDaemon::Schedule { cancel }, QUICK).ok());
    if let Some(daemon::ToClient::Said { ok, text }) = asked {
        return if ok {
            println!("{text}");
            ExitCode::SUCCESS
        } else {
            eprintln!("error: {text}");
            ExitCode::FAILURE
        };
    }
    let Some(path) = schedule::Schedule::default_path() else {
        eprintln!("error: no home directory to keep the schedule in");
        return ExitCode::FAILURE;
    };
    let (mut jobs, note) = schedule::Schedule::load(&path, schedule::utc_offset());
    if let Some(n) = note {
        eprintln!("note: {n}");
    }
    match cancel {
        None => {
            println!("{}", jobs.report(learn::now()));
            ExitCode::SUCCESS
        }
        Some(id) => match jobs.cancel(id) {
            Ok(said) => {
                println!("{said}");
                ExitCode::SUCCESS
            }
            Err(why) => {
                eprintln!("error: {why}");
                ExitCode::FAILURE
            }
        },
    }
}

/// `conscious check [--model SPEC] [--call]`: the start-up checks, printed.
/// Succeeds when a model can be used (and, with `--call`, when the core
/// answered a real request).
fn check(rest: &[String]) -> ExitCode {
    let mut named = std::env::var("CONSCIOUS_MODEL")
        .ok()
        .filter(|m| !m.is_empty());
    let mut call = false;
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--call" => call = true,
            "--model" | "-m" => match it.next() {
                Some(m) => named = Some(m.clone()),
                None => {
                    eprintln!("error: --model needs a value");
                    return ExitCode::from(2);
                }
            },
            _ if a.starts_with("--model=") => named = Some(a["--model=".len()..].to_string()),
            other => {
                eprintln!(
                    "error: unexpected argument `{other}`; use `conscious check [--model SPEC] [--call]`"
                );
                return ExitCode::from(2);
            }
        }
    }
    let report = preflight::judge(&preflight::probe(named));
    print!("{}", report.render());
    let mut ok = report.core.is_some();
    if call {
        println!();
        for spec in &report.ready {
            let _ = io::stdout().flush();
            match preflight::call(spec) {
                Ok(how) => println!("ok    call {spec}: {how}"),
                Err(why) => {
                    println!("FAIL  call {spec}: {why}");
                    if report.core.as_deref() == Some(spec) {
                        ok = false;
                    }
                }
            }
        }
    }
    println!(
        "\n{}",
        match (&report.core, ok) {
            (Some(core), true) => format!("Ready: `conscious` will start with {core}."),
            (Some(core), false) =>
                format!("Not ready: {core} would be the core model, but it did not answer."),
            (None, _) => "Not ready: no model can be used.".to_string(),
        }
    );
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// `conscious feedback good|bad`: rates the last routed answer. Needs no
/// model, only the saved routing knowledge.
fn feedback(rest: &[String]) -> ExitCode {
    let good = match rest {
        [v] if v == "good" => true,
        [v] if v == "bad" => false,
        _ => {
            eprintln!("error: use `conscious feedback good` or `conscious feedback bad`");
            return ExitCode::from(2);
        }
    };
    let Some(path) = learn::default_path() else {
        eprintln!("error: no home directory to keep routing knowledge in");
        return ExitCode::FAILURE;
    };
    let (mut knowledge, mut store) = load_learning(&path);
    match router::apply_feedback(&mut knowledge, &mut store, good) {
        Ok(learned) => {
            for (i, line) in learned.iter().enumerate() {
                println!("{}{line}", if i == 0 { "learned: " } else { "" });
            }
            if store.review_due() {
                println!("the core will review the routing policy on the next `ask`");
            }
            ExitCode::SUCCESS
        }
        Err(why) => {
            eprintln!("nothing learned: {why}");
            ExitCode::FAILURE
        }
    }
}

/// Loads routing knowledge and the policy versions kept beside it.
fn load_learning(path: &std::path::Path) -> (learn::Knowledge, policy::PolicyStore) {
    let (knowledge, note) = learn::Knowledge::load(path);
    let mut notes: Vec<String> = note.into_iter().collect();
    let store = match knowledge.policy_dir() {
        Some(dir) => {
            let (store, more) = policy::PolicyStore::load(&dir);
            notes.extend(more);
            store
        }
        None => policy::PolicyStore::in_memory(),
    };
    for n in notes {
        eprintln!("note: {n}");
    }
    (knowledge, store)
}

/// `conscious policy [show|rollback]`: inspect or step back the policy.
fn policy_command(action: Option<&str>, file: Option<&String>) -> ExitCode {
    let Some(path) = learn::default_path() else {
        eprintln!("error: no home directory to keep the routing policy in");
        return ExitCode::FAILURE;
    };
    let (knowledge, mut store) = load_learning(&path);
    match action {
        Some("propose") => {
            let Some(file) = file else {
                eprintln!("error: use `conscious policy propose FILE`");
                return ExitCode::from(2);
            };
            let source = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: {file}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            match store.propose(&source, "proposed by hand", &knowledge.rated_facts()) {
                Ok(version) => println!("policy v{version} adopted after passing regression"),
                Err(problems) => {
                    println!("rejected by regression, keeping v{}:", store.version());
                    for p in problems {
                        println!("  - {p}");
                    }
                    return ExitCode::FAILURE;
                }
            }
        }
        None => print!("{}", policy::summary(&store)),
        Some("show") => println!("{}", store.source()),
        Some("rollback") => println!("{}", store.rollback("rolled back by hand")),
        Some(other) => {
            eprintln!("error: unknown policy action `{other}` (show, rollback, propose, review)");
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

/// Has the core review the policy script and prints what came of it.
fn review(router: &Router) -> ExitCode {
    let outcome = router.review();
    if let Some(call) = &outcome.call {
        eprintln!(
            "[policy review: {} · {} in · {} out · {:.1}s]",
            call.model,
            call.usage.input,
            call.usage.output,
            call.latency.as_secs_f32()
        );
    }
    eprintln!("[{}]", outcome.message);
    ExitCode::SUCCESS
}

/// One-shot prompt, streamed to stdout. Usage goes to stderr.
fn ask(
    router: &Router,
    kb: Option<&std::sync::Mutex<knowledge::KnowledgeBase>>,
    prompt: &str,
) -> ExitCode {
    if prompt.trim().is_empty() {
        eprintln!("error: `ask` needs a prompt");
        return ExitCode::from(2);
    }
    let mut req = Request {
        messages: vec![Message::user(prompt)],
        ..Default::default()
    };
    if let Some((retrieval, block)) = kb.and_then(|kb| kb.lock().unwrap().recall(prompt, &[])) {
        eprintln!("[{}]", knowledge::KnowledgeBase::describe(&retrieval));
        req.system = block;
    }
    let choice = router.choose(prompt);
    for call in &choice.calls {
        eprintln!(
            "[routing call: {} · {} in · {} out · {:.1}s]",
            call.model,
            call.usage.input,
            call.usage.output,
            call.latency.as_secs_f32()
        );
    }
    for note in &choice.notes {
        eprintln!("warning: {note}");
    }
    if router.routes() {
        eprintln!(
            "[routing to {}: {}]",
            router.candidates()[choice.index].spec,
            choice.reason
        );
    }
    let mut out = io::stdout();
    let (_, fallback, result) = router.complete(&choice, &req, &mut |text| {
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    });
    if let Some(note) = fallback {
        eprintln!("warning: {note}");
    }
    match result {
        Ok(resp) => {
            println!();
            for w in &resp.warnings {
                eprintln!("warning: {w}");
            }
            eprintln!(
                "[{} · {} in ({} cached) · {} out · {:.1}s]",
                resp.model,
                resp.usage.input,
                resp.usage.cached,
                resp.usage.output,
                resp.latency.as_secs_f32()
            );
            // A rating given since the last run may have left a review due.
            if router.review_due() {
                review(router);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn tui(feed: Feed, log: timeline::Log, start: Start) -> io::Result<()> {
    let offset = schedule::utc_offset();
    let mut terminal = ratatui::init();
    execute!(io::stdout(), EnableMouseCapture)?;
    let result = run(&mut terminal, feed, log, start, offset);
    let _ = execute!(io::stdout(), DisableMouseCapture);
    ratatui::restore();
    result
}

fn run(
    terminal: &mut ratatui::DefaultTerminal,
    mut feed: Feed,
    mut log: timeline::Log,
    start: Start,
    mut utc_offset: i64,
) -> io::Result<()> {
    let mut app = feed.new_app(true);
    app.utc_offset = utc_offset;
    app.restore(start.before);
    if let Feed::Remote(_, status) = &feed {
        app.attach(start.since, start.memory, &status.core, status.busy);
    }
    for text in start.notices {
        app.apply(EventKind::Alert {
            severity: event::Severity::Warn,
            text,
        });
    }
    let mut last_tick = Instant::now();
    let mut zone_read = Instant::now();

    while !app.quit {
        terminal.draw(|f| ui::render(f, &mut app))?;

        if term::poll(TICK.saturating_sub(last_tick.elapsed()).min(FRAME))? {
            match term::read()? {
                term::Event::Key(k) if k.kind == KeyEventKind::Press => app.on_key(k),
                term::Event::Mouse(m) => app.on_mouse(m),
                _ => {}
            }
        }

        let commands: Vec<Command> = std::mem::take(&mut app.outbox);
        for cmd in commands {
            if let Command::Restart = cmd {
                feed.restart();
                app = feed.new_app(false);
                app.utc_offset = utc_offset;
                continue;
            }
            for event in feed.handle(cmd) {
                log.append(event.task, &event.kind);
                app.apply_in(event.task, event.kind);
            }
        }

        if last_tick.elapsed() >= TICK {
            last_tick = Instant::now();
            if !app.booting() {
                for event in feed.poll() {
                    log.append(event.task, &event.kind);
                    app.apply_in(event.task, event.kind);
                }
            }
            // The background process restarted into a build it made itself:
            // attach to the new one and draw from what it sends.
            if let Feed::Remote(remote, status) = &mut feed
                && let Some(hello) = remote.reattach()
            {
                *status = hello.status.clone();
                app = feed.new_app(false);
                app.utc_offset = utc_offset;
                app.restore(hello.before);
                app.attach(
                    hello.since,
                    hello.memory,
                    &hello.status.core,
                    hello.status.busy,
                );
                app.apply(EventKind::Alert {
                    severity: event::Severity::Success,
                    text: "Reconnected: the harness is now running the build it made.".into(),
                });
            }
            // Inside this window there is nothing to restart into.
            if let Feed::Live(live) = &mut feed
                && let Some(why) = live.take_restart()
            {
                app.apply(EventKind::Alert {
                    severity: event::Severity::Warn,
                    text: format!("{why}. Quit and start conscious again to run the new build."),
                });
            }
            for (task, text) in feed.drafts() {
                app.set_draft(task, text);
            }
            app.on_tick(feed.throughput());
            // Local time moves against UTC with daylight saving.
            if zone_read.elapsed() >= ZONE_EVERY {
                zone_read = Instant::now();
                utc_offset = schedule::utc_offset();
                app.utc_offset = utc_offset;
            }
        }
    }
    Ok(())
}
