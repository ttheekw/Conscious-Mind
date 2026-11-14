//! Offscreen render tests. Drive the demo deterministically, render into a
//! `TestBackend`, and check state + interactions. With `SNAPSHOT_HTML=1` the
//! rendered frames are also written to `target/snapshots/*.html` for review.

use std::fmt::Write as _;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::style::{Color, Modifier};

use crate::app::{App, Button, Modal, ToolStatus};
use crate::demo::{Command, Demo};
use crate::ui;

const W: u16 = 160;
const H: u16 = 46;

fn render(app: &mut App) -> Buffer {
    let mut term = Terminal::new(TestBackend::new(W, H)).unwrap();
    term.draw(|f| ui::render(f, app)).unwrap();
    term.backend().buffer().clone()
}

fn text(buf: &Buffer) -> String {
    let mut s = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            s.push_str(buf[(x, y)].symbol());
        }
        s.push('\n');
    }
    s
}

fn css(c: Color, default: &str) -> String {
    match c {
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        _ => default.to_string(),
    }
}

fn write_html(name: &str, buf: &Buffer) {
    if std::env::var("SNAPSHOT_HTML").is_err() {
        return;
    }
    let mut html = String::from(
        "<!doctype html><meta charset=utf-8><body style=\"margin:0;background:#05080d\">\
         <pre style=\"margin:0;padding:12px;font:14px/1.2 'DejaVu Sans Mono',monospace;\">",
    );
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            let cell = &buf[(x, y)];
            let sym = match cell.symbol() {
                "<" => "&lt;",
                ">" => "&gt;",
                "&" => "&amp;",
                s => s,
            };
            let bold = if cell.modifier.contains(Modifier::BOLD) {
                "font-weight:bold;"
            } else {
                ""
            };
            let ital = if cell.modifier.contains(Modifier::ITALIC) {
                "font-style:italic;"
            } else {
                ""
            };
            let _ = write!(
                html,
                "<span style=\"display:inline-block;width:1ch;overflow:visible;white-space:pre;color:{};background:{};{bold}{ital}\">{sym}</span>",
                css(cell.fg, "#b6ced8"),
                css(cell.bg, "#05080d"),
            );
        }
        html.push('\n');
    }
    html.push_str("</pre>");
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/target/snapshots");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(format!("{dir}/{name}.html"), html).unwrap();
    std::fs::write(format!("{dir}/{name}.txt"), text(buf)).unwrap();
}

fn key(app: &mut App, code: KeyCode) {
    app.on_key(KeyEvent::new(code, KeyModifiers::NONE));
}

fn click(app: &mut App, x: u16, y: u16) {
    app.on_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: x,
        row: y,
        modifiers: KeyModifiers::NONE,
    });
}

fn pump(app: &mut App, demo: &mut Demo) {
    for cmd in std::mem::take(&mut app.outbox) {
        for kind in demo.handle(cmd) {
            app.apply(kind);
        }
    }
}

/// App + demo advanced to the approval gate, with the approval modal open.
fn at_gate() -> (App, Demo) {
    let mut app = App::new(false);
    let mut demo = Demo::new();
    for kind in demo.drain_until_gate() {
        app.apply(kind);
    }
    for i in 0..60 {
        app.on_tick(2 + (i % 5));
    }
    (app, demo)
}

#[test]
fn approval_modal_opens_and_accept_button_works() {
    let (mut app, mut demo) = at_gate();
    assert_eq!(app.modal, Some(Modal::Tool(4)));
    assert_eq!(app.tool(4).unwrap().status, ToolStatus::AwaitingApproval);

    let buf = render(&mut app);
    write_html("1-approval", &buf);
    let screen = text(&buf);
    assert!(screen.contains("TOOL CALL #4"));
    assert!(screen.contains("ACCEPT"));

    let (rect, _) = *app
        .hits
        .buttons
        .iter()
        .find(|(_, b)| *b == Button::Accept)
        .expect("accept button drawn");
    click(&mut app, rect.x + 1, rect.y);
    pump(&mut app, &mut demo);
    assert_eq!(app.modal, None);
    assert_eq!(app.tool(4).unwrap().status, ToolStatus::Running);
    assert_eq!(app.tool(4).unwrap().decided_by.as_deref(), Some("you"));
}

#[test]
fn dashboard_shows_all_panels() {
    let (mut app, _demo) = at_gate();
    key(&mut app, KeyCode::Esc); // close the approval modal: "later"
    let buf = render(&mut app);
    write_html("2-dashboard", &buf);
    let screen = text(&buf);
    for needle in [
        "CONSCIOUS",
        "MIND",
        "AGENTS",
        "ACTIVITY",
        "CONTEXT WINDOW",
        "TOOLS",
        "SIGNALS",
        "APPROVAL PENDING",
        "research",
        "edit_file",
    ] {
        assert!(screen.contains(needle), "missing {needle:?} in:\n{screen}");
    }
}

#[test]
fn the_map_shows_real_memories_and_a_click_opens_one() {
    use crate::app::Modal;
    use crate::event::{EventKind as E, MemKind, MemoryNode};
    let mut app = App::new(false);
    let node = |id: u32, kind: MemKind, label: &str, text: &str| MemoryNode {
        id,
        kind,
        label: label.into(),
        text: text.into(),
        about: "not confirmed yet · something said in an earlier conversation".into(),
    };
    app.apply(E::MemoryIndexed {
        nodes: vec![
            node(
                3,
                MemKind::Semantic,
                "where the boat is kept",
                "The boat is kept in the old harbour, on the north pontoon.",
            ),
            node(
                9,
                MemKind::Procedural,
                "how replies should be written",
                "The user wants short replies with no headings.",
            ),
        ],
    });
    let screen = text(&render(&mut app));
    // The gauges nothing ever fed are gone; what the marks mean is said.
    for gone in ["CONF", "SURPRISE", "FOCUS"] {
        assert!(!screen.contains(gone), "{gone} is still shown:\n{screen}");
    }
    assert!(screen.contains("◆ fact"), "{screen}");
    // One mark per memory, each where a click can reach it.
    assert_eq!(app.hits.memories.len(), 2);
    let (spot, id) = app
        .hits
        .memories
        .iter()
        .find(|(_, id)| *id == 3)
        .copied()
        .unwrap();
    assert_eq!(id, 3);
    click(&mut app, spot.x + 1, spot.y);
    assert_eq!(app.modal, Some(Modal::Memory(3)));
    let screen = text(&render(&mut app));
    for expected in [
        "where the boat is kept",
        "◆ a fact",
        "something said in an earlier conversation",
        "The boat is kept in the old harbour, on the north pontoon.",
        "2 of 2",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in:\n{screen}"
        );
    }
    // The arrows step through the memories, newest first; esc closes.
    key(&mut app, KeyCode::Up);
    assert_eq!(app.modal, Some(Modal::Memory(9)));
    let screen = text(&render(&mut app));
    assert!(screen.contains("The user wants short replies with no headings."));
    assert!(screen.contains("always kept in mind"));
    key(&mut app, KeyCode::Esc);
    assert_eq!(app.modal, None);
    // m opens the newest one without the mouse.
    key(&mut app, KeyCode::Char('m'));
    assert_eq!(app.modal, Some(Modal::Memory(9)));
}

#[test]
fn clicking_a_tool_row_opens_its_details() {
    let (mut app, _demo) = at_gate();
    key(&mut app, KeyCode::Esc);
    render(&mut app);
    let (rect, i) = app.hits.tools[0];
    click(&mut app, rect.x + 2, rect.y);
    assert_eq!(app.modal, Some(Modal::Tool(app.tools[i].id)));
    let buf = render(&mut app);
    write_html("3-tool-detail", &buf);
    assert!(text(&buf).contains("OUTPUT"));
}

#[test]
fn deny_blocks_the_goal_and_raw_events_tab_lists_everything() {
    let (mut app, mut demo) = at_gate();
    key(&mut app, KeyCode::Char('d'));
    pump(&mut app, &mut demo);
    for kind in demo.drain_until_gate() {
        app.apply(kind);
    }
    assert_eq!(app.tool(4).unwrap().status, ToolStatus::Denied);
    assert_eq!(app.goal_state().0, "BLOCKED");

    key(&mut app, KeyCode::Char('2'));
    let buf = render(&mut app);
    write_html("4-raw-events", &buf);
    assert!(text(&buf).contains("ToolDecision"));
}

#[test]
fn user_message_reaches_timeline() {
    let (mut app, mut demo) = at_gate();
    key(&mut app, KeyCode::Esc);
    key(&mut app, KeyCode::Char('i'));
    for c in "hello".chars() {
        key(&mut app, KeyCode::Char(c));
    }
    key(&mut app, KeyCode::Enter);
    assert!(matches!(app.outbox.first(), Some(Command::UserMessage(m)) if m == "hello"));
    pump(&mut app, &mut demo);
    let buf = render(&mut app);
    assert!(text(&buf).contains("hello"));
}

#[test]
fn boot_screen_renders() {
    let mut app = App::new(true);
    app.boot = Some(std::time::Instant::now() - std::time::Duration::from_millis(820));
    app.tick = 40;
    assert!(app.booting());
    let buf = render(&mut app);
    write_html("0-boot", &buf);
    assert!(text(&buf).contains("event log"));
}

#[test]
fn memory_recall_learning_and_reflection_update_the_mind() {
    let (mut app, mut demo) = at_gate();
    // The recall during diagnosis pulled memory #7 into working memory.
    let m7 = app.memory.iter().find(|m| m.node.id == 7).unwrap();
    assert!(m7.touched.is_some() && m7.recalls == 1);
    assert_eq!(app.memory.len(), 14);

    // Approve: the agent learns a lesson, then reflects and merges it with #7.
    key(&mut app, KeyCode::Char('a'));
    pump(&mut app, &mut demo);
    for kind in demo.drain_until_gate() {
        app.apply(kind);
    }
    assert!(app.memory.iter().any(|m| m.node.id == 100));
    for kind in demo.run_idle_cycle() {
        app.apply(kind);
    }
    let merged = app
        .memory
        .iter()
        .find(|m| m.node.id == 102)
        .expect("consolidated memory");
    assert!(merged.absorbed.is_none());
    assert!(app.memory.iter().filter(|m| m.absorbed.is_some()).count() == 2);
    assert_eq!(app.goal_state().0, "VERIFIED");
    assert!(app.confidence > 0.85);

    let buf = render(&mut app);
    write_html("5-reflecting", &buf);
    let screen = text(&buf);
    assert!(screen.contains("MIND"));
    // What memory did is in the knowledge view, not in the conversation.
    assert!(!screen.contains("MERGE"));
    key(&mut app, KeyCode::Char('3'));
    assert!(text(&render(&mut app)).contains("MERGE"));
}

/// Not a correctness test: prints how long one frame takes to render.
/// `cargo test frame_time -- --ignored --nocapture` (add `--release` to compare).
#[test]
#[ignore = "timing probe"]
fn frame_time() {
    let (mut app, _demo) = at_gate();
    key(&mut app, KeyCode::Esc);
    let mut term = Terminal::new(TestBackend::new(W, H)).unwrap();
    let frames = 200;
    let started = std::time::Instant::now();
    for _ in 0..frames {
        app.on_tick(3);
        term.draw(|f| ui::render(f, &mut app)).unwrap();
    }
    let per = started.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!("frame: {per:.2} ms at {W}x{H}");
}

/// A live session with a small knowledge graph: one question answered.
fn after_a_lookup() -> App {
    use crate::knowledge::KnowledgeBase;
    use crate::live::Live;
    use crate::provider::testing::Fixed;
    use crate::router::Router;
    use std::sync::{Arc, Mutex};

    let mut kb = KnowledgeBase::in_memory();
    kb.ingest(
        "proj",
        "docs/policy.md",
        "# Policy\n\n## Rollback\n\nA rewrite that misbehaves is rolled back by the `PolicyStore`, which keeps every version on disk.\n",
    );
    kb.ingest(
        "proj",
        "docs/store.md",
        "# Store internals\n\n## Versions\n\nThe `PolicyStore` writes each adopted script as a numbered file beside `history.json`.\n",
    );
    kb.commit().unwrap();
    let reply = "## Rollback\n\nA rewrite is rolled back when it **misbehaves**:\n\n- its file no longer loads\n- it returns an *invalid* index\n\n| Check | Limit |\n|---|---|\n| bad ratings | 2 |\n\n```rust\nstore.rollback(\"why\");\n```\nSee `PolicyStore`.";
    let router = Arc::new(Router::single(Arc::new(Fixed(reply))));
    let mut live = Live::with_knowledge(router, Some(Arc::new(Mutex::new(kb))));
    let mut app = App::new(false);
    let mut events = live.poll();
    events.extend(live.handle(Command::UserMessage(
        "what happens when a rewrite misbehaves".into(),
    )));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !events
        .iter()
        .any(|e| matches!(e.kind, crate::event::EventKind::MemoryWrite { .. }))
        && std::time::Instant::now() < deadline
    {
        events.extend(live.poll());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    for event in events {
        app.apply_in(event.task, event.kind);
    }
    // Open the task's own tab, where its conversation is.
    app.select_task(Some(1));
    app
}

#[test]
fn replies_are_rendered_as_markdown_in_the_timeline() {
    let mut app = after_a_lookup();
    let buf = render(&mut app);
    write_html("6-markdown", &buf);
    let screen = text(&buf);
    // Markup is gone; structure is drawn.
    assert!(!screen.contains("**misbehaves**"), "{screen}");
    assert!(!screen.contains("```"), "{screen}");
    assert!(
        screen.contains("A rewrite is rolled back when it misbehaves:"),
        "{screen}"
    );
    assert!(screen.contains("• its file no longer loads"));
    assert!(screen.contains("Check       │ Limit"));
    assert!(screen.contains("store.rollback(\"why\");"));
    // The lookup is not part of the conversation: it is in the knowledge
    // view.
    assert!(!screen.contains("2 passages from the knowledge graph"));
    key(&mut app, KeyCode::Char('3'));
    let screen = text(&render(&mut app));
    assert!(screen.contains("2 passages from the knowledge graph"));
    assert!(screen.contains("press g to see how"));
}

#[test]
fn the_lookup_view_draws_the_routes_as_a_graph() {
    let mut app = after_a_lookup();
    key(&mut app, KeyCode::Char('g'));
    assert!(matches!(app.modal, Some(Modal::Lookup(_))));
    let buf = render(&mut app);
    write_html("7-lookup", &buf);
    let screen = text(&buf);
    for needle in [
        "KNOWLEDGE LOOKUP",
        "happens · misbehaves · rewrite",
        "node matched your words",
        "2 passages given to the model",
        // The graph: the question, the shared term, both passages.
        "YOUR QUESTION",
        "#1 Policy > Rollback",
        "policystore",
        "#2 Store internals > Versions",
        // The selected passage, in words.
        "PASSAGE 1 OF 2",
        "your question → Policy > Rollback",
        "its own words matched your question (words found: misbehaves, rewrite)",
        "the project's documentation · verified",
        "rolled back by the `PolicyStore`",
    ] {
        assert!(screen.contains(needle), "missing {needle:?} in:\n{screen}");
    }
    // File names are not shown.
    assert!(!screen.contains(".md"), "{screen}");
    // The links are solid lines that turn with rounded corners.
    assert!(screen.contains('╭') || screen.contains('╰'), "{screen}");
    assert!(screen.contains('◉'));

    // Stepping to the second passage shows the links that led to it.
    key(&mut app, KeyCode::Down);
    let buf = render(&mut app);
    write_html("8-lookup-linked", &buf);
    let screen = text(&buf);
    for needle in [
        "PASSAGE 2 OF 2",
        "your question → Policy > Rollback → policystore → Store internals > Versions",
        "reached 2 links away from the words that matched",
        "numbered file beside",
    ] {
        assert!(screen.contains(needle), "missing {needle:?} in:\n{screen}");
    }
    // It cannot step past the last passage, and Esc closes the view.
    key(&mut app, KeyCode::Down);
    assert_eq!(app.lookup_sel, 1);
    key(&mut app, KeyCode::Esc);
    assert_eq!(app.modal, None);
}

#[test]
fn a_screen_attaching_to_a_running_harness_shows_what_is_true_now() {
    use crate::event::{EventKind, MemKind, MemoryNode, Risk};
    use crate::timeline::Recorded;
    let at = |wall: u64, kind: EventKind| Recorded {
        wall,
        task: Some(1),
        kind,
    };
    let asked = |id: u32, needs_approval: bool| EventKind::ToolRequested {
        id,
        agent: 0,
        tool: "write_file".into(),
        args: "notes.txt".into(),
        risk: Risk::Medium,
        needs_approval,
        policy: "changes a file".into(),
    };
    // Before the harness started: a call nobody ever decided.
    let before = vec![at(100, asked(1, true))];
    // Since: one call decided and finished, one still waiting.
    let since = vec![
        at(
            200,
            EventKind::Message {
                agent: None,
                text: "tidy the notes".into(),
            },
        ),
        at(201, asked(2, true)),
        at(
            202,
            EventKind::ToolDecision {
                id: 2,
                approved: true,
                by: "you".into(),
            },
        ),
        at(
            203,
            EventKind::ToolFinished {
                id: 2,
                ok: true,
                output: "written".into(),
                duration_ms: 3,
            },
        ),
        at(204, asked(3, true)),
    ];
    let memory = vec![MemoryNode {
        id: 7,
        kind: MemKind::Semantic,
        label: "the notes live in notes.txt".into(),
        text: String::new(),
        about: String::new(),
    }];

    let mut app = App::new(false);
    app.restore(before);
    app.attach(since, memory, "fixed-1", true);

    // The old call is history; the one waiting now is waiting here too,
    // and its prompt is the one that is open.
    assert_eq!(app.tool(1).unwrap().status, ToolStatus::Denied);
    assert_eq!(app.tool(2).unwrap().status, ToolStatus::Ok);
    let waiting: Vec<u32> = app.pending().iter().map(|t| t.id).collect();
    assert_eq!(waiting, [3]);
    assert!(matches!(app.modal, Some(Modal::Tool(3))));
    // Deciding it on this screen sends the decision to the harness.
    key(&mut app, KeyCode::Char('a'));
    assert!(matches!(
        app.outbox[..],
        [Command::Decide {
            id: 3,
            approve: true
        }]
    ));
    // The core agent is there although its start is further back than
    // what was sent, and memory is what the store holds now.
    assert_eq!(app.agents.len(), 1);
    assert_eq!(app.agents[0].purpose, "talks to fixed-1");
    assert_eq!(app.memory.len(), 1);
    // Events keep the time they happened at.
    assert_eq!(app.events.last().unwrap().wall, 204);
    assert!(text(&render(&mut app)).contains("tidy the notes"));

    // With nothing waiting, no prompt is opened.
    let mut idle = App::new(false);
    idle.attach(vec![at(201, asked(2, false))], Vec::new(), "fixed-1", false);
    assert!(idle.modal.is_none());
}

/// A screen with three tasks: one finished, one paused for research, one
/// waiting its turn.
fn with_tasks() -> App {
    use crate::event::{ContextSection, EventKind as E, Risk, TaskState};
    let mut app = App::new(false);
    app.apply(E::AgentSpawned {
        id: 0,
        parent: None,
        name: "core".into(),
        purpose: "talks to fixed-1".into(),
    });
    let mut say = |task: u32, kind: E| app.apply_in(Some(task), kind);
    // Main: the conversation, which opened a branch for the work.
    say(
        0,
        E::TaskOpened {
            title: "main".into(),
            origin: "you".into(),
        },
    );
    say(
        0,
        E::Message {
            agent: None,
            text: "how many python files are there? and look into tides for me".into(),
        },
    );
    say(
        0,
        E::Message {
            agent: Some(0),
            text: "I opened a branch for each.".into(),
        },
    );
    say(
        0,
        E::TaskUpdate {
            state: TaskState::Done,
            note: "answered after 3 model calls".into(),
        },
    );
    say(
        1,
        E::TaskOpened {
            title: "count the python files".into(),
            origin: "main".into(),
        },
    );
    say(
        1,
        E::Message {
            agent: None,
            text: "count the python files".into(),
        },
    );
    say(
        1,
        E::ToolRequested {
            id: 1,
            agent: 0,
            tool: "bash".into(),
            args: "find . -name '*.py' | wc -l".into(),
            risk: Risk::Low,
            needs_approval: false,
            policy: "read-only".into(),
        },
    );
    say(
        1,
        E::Message {
            agent: Some(0),
            text: "There are **4** python files.".into(),
        },
    );
    say(
        1,
        E::TaskUpdate {
            state: TaskState::Done,
            note: "answered after 2 model calls".into(),
        },
    );
    say(
        0,
        E::Merged {
            branch: 1,
            title: "count the python files".into(),
            result: "There are **4** python files.".into(),
            ok: true,
        },
    );
    say(
        2,
        E::TaskOpened {
            title: "explain how spring tides differ from neap tides".into(),
            origin: "main".into(),
        },
    );
    say(
        2,
        E::Message {
            agent: None,
            text: "explain how spring tides differ from neap tides".into(),
        },
    );
    say(
        2,
        E::Context {
            window: 8_000,
            sections: vec![
                ContextSection {
                    name: "system".into(),
                    tokens: 700,
                    why: "harness instructions".into(),
                },
                ContextSection {
                    name: "task".into(),
                    tokens: 12,
                    why: "this task's own conversation (1 message)".into(),
                },
            ],
        },
    );
    say(
        2,
        E::TaskUpdate {
            state: TaskState::Researching,
            note: "finding out: how often do spring tides come".into(),
        },
    );
    say(
        2,
        E::Research {
            phase: "search".into(),
            text: "searched the web for \"spring tide frequency\": 9 results".into(),
            source: String::new(),
        },
    );
    say(
        2,
        E::Research {
            phase: "kept".into(),
            text: "kept 6 passages from: Spring tides".into(),
            source: "https://tides.example/spring".into(),
        },
    );
    say(
        3,
        E::TaskOpened {
            title: "tidy the notes folder".into(),
            origin: "routine #2".into(),
        },
    );
    say(
        3,
        E::TaskUpdate {
            state: TaskState::Queued,
            note: "waiting its turn: 3 tasks are running, the most at once".into(),
        },
    );
    app
}

#[test]
fn main_keeps_going_and_each_branch_has_a_tab_and_merges_back() {
    let mut app = with_tasks();
    // The screen opens on main: the conversation, with what its branches
    // brought back merged into it.
    assert_eq!(app.task_sel, Some(0));
    let buf = render(&mut app);
    write_html("8-main", &buf);
    let screen = text(&buf);
    for expected in [
        "0 MAIN",
        "b BRANCHES 2",
        "MAIN :: ",
        "READY",
        "2 branches open",
        "I opened a branch for each.",
        "branch #1 ended and is merged into main: count the python files",
        "There are 4 python files.",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in:\n{screen}"
        );
    }
    // What a branch did on the way is in its own tab, not in main.
    assert!(!screen.contains("kept 6 passages"), "{screen}");

    // A reply shows while it is being written, at the bottom, in the tab of
    // the task writing it, and gives way to the finished reply.
    app.set_draft(0, "Counting the files now".into());
    app.set_draft(2, "Spring tides come".into());
    let screen = text(&render(&mut app));
    assert!(screen.contains("writing"), "{screen}");
    assert!(screen.contains("Counting the files now▌"), "{screen}");
    assert!(!screen.contains("Spring tides come"), "{screen}");
    app.apply_in(
        Some(0),
        crate::event::EventKind::Message {
            agent: Some(0),
            text: "There are 4.".into(),
        },
    );
    let screen = text(&render(&mut app));
    assert!(!screen.contains("writing"), "{screen}");
    assert!(screen.contains("There are 4."), "{screen}");
    // A tab for main, one for the list of branches, then one per branch
    // as far as they fit, with a count of the rest.
    assert_eq!(app.hits.task_tabs.len(), 3, "{screen}");
    assert!(screen.contains("#3 ○ tidy the no…"), "{screen}");
    assert!(screen.contains("+1"), "{screen}");
    // Only branches still going are counted and listed: the one that ended
    // is merged into main and has no tab unless asked for.
    assert!(screen.contains("b BRANCHES 2"), "{screen}");

    // b lists the branches, newest first, with each one's state in words.
    key(&mut app, KeyCode::Char('b'));
    assert_eq!(app.task_sel, None);
    let buf = render(&mut app);
    write_html("8-branches-board", &buf);
    let screen = text(&buf);
    assert!(
        screen.contains("1 running · 1 queued · 1 ended, hidden (e shows them)"),
        "{screen}"
    );
    assert!(!screen.contains("#1   ✓ DONE"), "{screen}");
    // e lists the ended ones too (and hides them again).
    key(&mut app, KeyCode::Char('e'));
    let screen = text(&render(&mut app));
    assert!(screen.contains("1 ended (e hides them)"), "{screen}");
    for expected in [
        "#3   ○ QUEUED",
        "#2   ◈ RESEARCH",
        "#1   ✓ DONE",
        "tidy the notes folder",
        "finding out: how often do spring tides",
        "1 page researched",
    ] {
        assert!(
            screen.contains(expected),
            "missing {expected:?} in:\n{screen}"
        );
    }

    // Enter opens the selected row's branch; the newest is selected first.
    key(&mut app, KeyCode::Enter);
    assert_eq!(app.task_sel, Some(3));
    assert!(text(&render(&mut app)).contains("opened by routine #2"));
    // Stepping through the tabs reaches the one being researched.
    key(&mut app, KeyCode::Char(']'));
    assert_eq!(app.task_sel, Some(2));
    let buf = render(&mut app);
    write_html("9-task-research", &buf);
    let screen = text(&buf);
    assert!(screen.contains("BRANCH #2 :: "), "{screen}");
    assert!(screen.contains("PAUSED FOR KNOWLEDGE RESEARCH"), "{screen}");
    assert!(screen.contains("merges into main when it ends"), "{screen}");
    // Its activity is the conversation only: what was asked of it, not
    // each step it took, and nothing of another branch's.
    assert!(
        screen.contains("explain how spring tides differ"),
        "{screen}"
    );
    assert!(!screen.contains("kept 6 passages"), "{screen}");
    assert!(!screen.contains("There are 4 python files."), "{screen}");

    // Its knowledge view has the research steps, and says what the model
    // was given, and why.
    key(&mut app, KeyCode::Char('3'));
    let screen = text(&render(&mut app));
    assert!(screen.contains("kept 6 passages from: Spring tides"));
    assert!(screen.contains("https://tides.example/spring"));
    assert!(screen.contains("712 tokens given to the model"), "{screen}");
    assert!(screen.contains("this task's own conversation"));

    // Typing here while the branch is still going is said to main; in a
    // branch that has answered it is a reply to that branch.
    key(&mut app, KeyCode::Char('i'));
    assert_eq!(app.input_task, None);
    assert!(text(&render(&mut app)).contains("TO MAIN"));
    key(&mut app, KeyCode::Esc);
    key(&mut app, KeyCode::Char(']'));
    assert_eq!(app.task_sel, Some(1));
    key(&mut app, KeyCode::Char('i'));
    assert_eq!(app.input_task, Some(1));
    for c in "and the tests?".chars() {
        key(&mut app, KeyCode::Char(c));
    }
    assert!(text(&render(&mut app)).contains("REPLY IN BRANCH #1"));
    key(&mut app, KeyCode::Enter);
    assert!(matches!(
        &app.outbox[..],
        [Command::Reply { task: 1, text }] if text == "and the tests?"
    ));
    // n always talks to main. Nothing opens a tab for it: whether it
    // becomes a branch is the harness's decision, and the screen stays put.
    key(&mut app, KeyCode::Esc);
    key(&mut app, KeyCode::Char('n'));
    assert_eq!(app.input_task, None);
    key(&mut app, KeyCode::Char('x'));
    key(&mut app, KeyCode::Enter);
    assert!(matches!(&app.outbox[1], Command::UserMessage(m) if m == "x"));
    app.apply_in(
        Some(4),
        crate::event::EventKind::TaskOpened {
            title: "x".into(),
            origin: "main".into(),
        },
    );
    assert_eq!(app.task_sel, Some(1));
    // 0 goes back to main, where typing is always to main.
    key(&mut app, KeyCode::Esc);
    key(&mut app, KeyCode::Char('0'));
    assert_eq!(app.task_sel, Some(0));
    key(&mut app, KeyCode::Char('i'));
    assert_eq!(app.input_task, None);
    key(&mut app, KeyCode::Esc);
    // X stops the branch whose tab is open, if it is still going.
    app.select_task(Some(2));
    key(&mut app, KeyCode::Char('X'));
    assert!(matches!(app.outbox[2], Command::Stop { task: 2 }));
}
