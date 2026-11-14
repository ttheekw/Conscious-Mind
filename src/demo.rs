//! Scripted event feed that stands in for the real harness while the TUI is
//! being designed. Nothing here talks to a model; all data is simulated.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::event::{
    AgentState, ContextSection, EventKind, MemKind, MemoryNode, Risk, Severity, StepStatus,
    ToolCallId,
};

/// Commands the TUI sends back to the harness.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Command {
    Decide {
        id: ToolCallId,
        approve: bool,
    },
    /// A new task, in the user's words.
    UserMessage(String),
    /// A further message in a task that already exists.
    Reply {
        task: u32,
        text: String,
    },
    /// The user's verdict on a task's answer (the last finished one when
    /// none is named), for routing to learn from.
    Feedback {
        good: bool,
        #[serde(default)]
        task: Option<u32>,
    },
    /// End a task that is still running.
    Stop {
        task: u32,
    },
    /// A secret the user typed into its own window, or (`None`) their
    /// refusal to give one that was asked for. Never an event.
    Secret {
        name: String,
        value: Option<crate::secrets::Hidden>,
    },
    Restart,
}

enum Step {
    Emit(u64, EventKind),
    /// Block the script until the user decides on this tool call.
    Gate {
        tool: ToolCallId,
        on_approve: Vec<Step>,
        on_deny: Vec<Step>,
    },
}

pub struct Demo {
    queue: VecDeque<Step>,
    next_at: Instant,
    gate: Option<(ToolCallId, Vec<Step>, Vec<Step>)>,
    rng: u64,
    busy: bool,
    /// How the approval gate was decided; shapes what the agent reflects on.
    approved: Option<bool>,
    idle_cycle: u32,
    /// Whether `poll` queues another idle cycle once the script runs out.
    /// Tests turn this off while draining, or the queue would never empty.
    keep_alive: bool,
}

fn e(ms: u64, kind: EventKind) -> Step {
    Step::Emit(ms, kind)
}

fn status(id: u32, state: AgentState) -> EventKind {
    EventKind::AgentStatus { id, state }
}

fn ctx(sections: &[(&str, u32, &str)]) -> EventKind {
    EventKind::Context {
        window: 32_768,
        sections: sections
            .iter()
            .map(|(n, t, w)| ContextSection {
                name: n.to_string(),
                tokens: *t,
                why: w.to_string(),
            })
            .collect(),
    }
}

fn model(agent: u32, tin: u32, cached: u32, tout: u32, ms: u64) -> EventKind {
    EventKind::ModelCall {
        agent,
        model: "gemma-4-e4b".into(),
        tokens_in: tin,
        cached_in: cached,
        tokens_out: tout,
        latency_ms: ms,
    }
}

fn tool(
    id: u32,
    agent: u32,
    name: &str,
    args: &str,
    risk: Risk,
    ask: bool,
    policy: &str,
) -> EventKind {
    EventKind::ToolRequested {
        id,
        agent,
        tool: name.into(),
        args: args.into(),
        risk,
        needs_approval: ask,
        policy: policy.into(),
    }
}

fn auto(id: u32) -> EventKind {
    EventKind::ToolDecision {
        id,
        approved: true,
        by: "policy:auto".into(),
    }
}

fn done(id: u32, ok: bool, output: &str, ms: u64) -> EventKind {
    EventKind::ToolFinished {
        id,
        ok,
        output: output.into(),
        duration_ms: ms,
    }
}

fn alert(severity: Severity, text: &str) -> EventKind {
    EventKind::Alert {
        severity,
        text: text.into(),
    }
}

fn step(index: usize, status: StepStatus) -> EventKind {
    EventKind::StepUpdate { index, status }
}

fn node(id: u32, kind: MemKind, label: &str) -> MemoryNode {
    MemoryNode {
        id,
        kind,
        label: label.into(),
        text: String::new(),
        about: String::new(),
    }
}

fn vitals(agent: u32, confidence: f32, surprise: f32, focus: &str) -> EventKind {
    EventKind::Vitals {
        agent,
        confidence,
        surprise,
        focus: focus.into(),
    }
}

fn thought(agent: u32, text: &str) -> EventKind {
    EventKind::Thought {
        agent,
        text: text.into(),
    }
}

/// The long-term memory the agent starts the session with.
fn memory_index() -> Vec<MemoryNode> {
    use MemKind::*;
    vec![
        node(1, Semantic, "parser crate: crates/parser"),
        node(2, Semantic, "tokenizer owns list parsing"),
        node(3, Procedural, "run tests before patching"),
        node(4, Episodic, "11-07 flaky io_timeout test"),
        node(5, Semantic, "CI runs clippy -D warnings"),
        node(6, Procedural, "prefer small reviewable diffs"),
        node(7, Episodic, "11-02 trailing ',' broke lists"),
        node(8, Semantic, "grammar v2 allows nesting"),
        node(9, Episodic, "11-09 user denied force-push"),
        node(10, Procedural, "delegate reading to sub-agents"),
        node(11, Semantic, "user timezone UTC+5:30"),
        node(12, Episodic, "11-10 nightly tests scheduled"),
        node(13, Procedural, "summarise long tool output"),
        node(14, Semantic, "model gemma-4-e4b on cpu x4"),
    ]
}

fn script() -> Vec<Step> {
    use AgentState::*;
    vec![
        e(300, EventKind::GoalSet {
            title: "Make `cargo test -p parser` pass".into(),
            criteria: vec![
                "cargo test -p parser exits 0".into(),
                "no new clippy warnings".into(),
            ],
            token_budget: 200_000,
        }),
        e(200, EventKind::AgentSpawned {
            id: 0,
            parent: None,
            name: "core".into(),
            purpose: "owns the goal: plans, delegates, verifies".into(),
        }),
        e(250, EventKind::MemoryIndexed {
            nodes: memory_index(),
        }),
        e(300, ctx(&[
            ("system", 1_180, "harness rules + tool schemas (stable, cached)"),
            ("goal", 420, "active goal and its success criteria"),
            ("memory", 0, "nothing retrieved yet"),
            ("recent", 260, "last turns in the timeline"),
        ])),
        e(200, status(0, Thinking)),
        e(1400, model(0, 1_860, 1_180, 14, 2_900)),
        e(100, EventKind::Thought {
            agent: 0,
            text: "Criteria are checkable by running the test suite. Start by seeing what fails.".into(),
        }),
        e(150, vitals(0, 0.40, 0.05, "crates/parser test suite")),
        e(400, EventKind::PlanUpdated {
            version: 1,
            steps: vec![
                "Run the parser test suite".into(),
                "Diagnose the failures".into(),
                "Patch the parser".into(),
                "Verify: tests + clippy".into(),
            ],
        }),
        e(200, step(0, StepStatus::Active)),
        e(200, status(0, Acting)),
        e(300, tool(1, 0, "shell", "cargo test -p parser", Risk::Low, false, "allow: read-only build/test command inside workspace")),
        e(150, auto(1)),
        e(2200, done(1, false, "running 18 tests\ntest tokenizer::nested_goal ... FAILED\ntest tokenizer::trailing_comma ... FAILED\ntest grammar::list_items ... FAILED\n\ntest result: FAILED. 15 passed; 3 failed", 2_140)),
        e(200, step(0, StepStatus::Done)),
        e(150, step(1, StepStatus::Active)),
        e(200, alert(Severity::Warn, "3 tests failing in parser: nested_goal, trailing_comma, list_items")),
        e(150, vitals(0, 0.30, 0.85, "3 failing tests: tokenizer, grammar")),
        e(300, EventKind::MemoryRead {
            agent: 0,
            query: "parser trailing comma".into(),
            hits: vec![
                (7, "11-02 trailing ',' broke lists: tokenizer consumed ',' before the ')' check".into()),
                (2, "tokenizer owns list parsing".into()),
            ],
        }),
        e(150, vitals(0, 0.52, 0.30, "memory #7: the 11-02 trailing-comma bug")),
        e(300, ctx(&[
            ("system", 1_180, "harness rules + tool schemas (stable, cached)"),
            ("goal", 520, "goal, criteria and plan v1"),
            ("memory", 640, "1 episodic hit: earlier trailing-comma bug"),
            ("tool", 980, "failing test output from call #1"),
            ("recent", 410, "last turns in the timeline"),
        ])),
        e(300, status(0, Thinking)),
        e(1200, model(0, 3_730, 1_180, 17, 3_400)),
        e(100, EventKind::Thought {
            agent: 0,
            text: "Memory points at the tokenizer. Split the work: one sub-agent reads the tokenizer, another maps the failing tests.".into(),
        }),
        e(300, EventKind::AgentSpawned {
            id: 1,
            parent: Some(0),
            name: "research".into(),
            purpose: "read tokenizer.rs and locate trailing-comma handling".into(),
        }),
        e(150, EventKind::AgentSpawned {
            id: 2,
            parent: Some(0),
            name: "tests".into(),
            purpose: "map each failing test to the code path it exercises".into(),
        }),
        e(100, status(0, Waiting)),
        e(100, status(1, Acting)),
        e(100, status(2, Acting)),
        e(300, tool(2, 1, "read_file", "src/tokenizer.rs", Risk::Low, false, "allow: read inside workspace")),
        e(100, auto(2)),
        e(200, tool(3, 2, "grep", "-n \"fn (nested_goal|trailing_comma|list_items)\" tests/", Risk::Low, false, "allow: read inside workspace")),
        e(100, auto(3)),
        e(500, done(2, true, "// 412 lines\n...\n198 |     Token::Comma => { self.advance(); continue; }\n199 |     Token::RParen => break,\n...", 40)),
        e(200, done(3, true, "tests/tokenizer.rs:12: fn nested_goal()\ntests/tokenizer.rs:31: fn trailing_comma()\ntests/grammar.rs:8: fn list_items()", 25)),
        e(200, status(1, Thinking)),
        e(200, status(2, Thinking)),
        e(1300, model(1, 2_410, 0, 15, 3_100)),
        e(100, EventKind::Message {
            agent: Some(1),
            text: "Found it: line 198 consumes ',' then `continue`s, so a ',' directly before ')' leaves the list open. Same root cause as the 2026-11-02 bug.".into(),
        }),
        e(150, vitals(1, 0.80, 0.10, "src/tokenizer.rs:198")),
        e(200, status(1, Done)),
        e(700, model(2, 1_120, 0, 11, 2_200)),
        e(100, EventKind::Message {
            agent: Some(2),
            text: "All 3 failures go through tokenizer list handling; no other code paths involved.".into(),
        }),
        e(200, status(2, Done)),
        e(200, step(1, StepStatus::Done)),
        e(150, step(2, StepStatus::Active)),
        e(200, status(0, Thinking)),
        e(1100, model(0, 4_020, 1_180, 18, 3_600)),
        e(100, EventKind::Thought {
            agent: 0,
            text: "Check for ')' before consuming ','. Small patch, but it edits source, so policy will ask.".into(),
        }),
        e(150, vitals(0, 0.82, 0.05, "patch src/tokenizer.rs:198")),
        e(200, status(0, Acting)),
        e(300, tool(4, 0, "edit_file", "path: src/tokenizer.rs\n@@ -196,6 +196,10 @@\n-    Token::Comma => { self.advance(); continue; }\n+    Token::Comma => {\n+        self.advance();\n+        if self.peek() == Token::RParen { break; }\n+        continue;\n+    }", Risk::Medium, true, "ask: edits a tracked source file (not in scratch dir)")),
        e(100, status(0, Waiting)),
        e(100, alert(Severity::Warn, "approval needed: #4 edit_file src/tokenizer.rs")),
        Step::Gate {
            tool: 4,
            on_approve: vec![
                e(100, status(0, Acting)),
                e(600, done(4, true, "patched src/tokenizer.rs (+4 −1)", 12)),
                e(200, step(2, StepStatus::Done)),
                e(150, step(3, StepStatus::Active)),
                e(300, tool(5, 0, "shell", "cargo test -p parser", Risk::Low, false, "allow: read-only build/test command inside workspace")),
                e(100, auto(5)),
                e(2300, done(5, true, "running 18 tests\n..................\ntest result: ok. 18 passed; 0 failed", 2_210)),
                e(150, vitals(0, 0.95, 0.02, "verify: clippy")),
                e(300, tool(6, 0, "shell", "cargo clippy -p parser -- -D warnings", Risk::Low, false, "allow: read-only build/test command inside workspace")),
                e(100, auto(6)),
                e(2000, done(6, true, "Finished `dev` profile in 1.9s (0 warnings)", 1_930)),
                e(200, step(3, StepStatus::Done)),
                e(200, EventKind::MemoryWrite {
                    agent: 0,
                    node: node(100, MemKind::Procedural, "lists: check ')' before ','"),
                }),
                e(200, status(0, Done)),
                e(100, alert(Severity::Success, "GOAL VERIFIED — 2/2 criteria met (tests pass, clippy clean)")),
                e(100, EventKind::Message {
                    agent: Some(0),
                    text: "Done. Patched the trailing-comma handling in src/tokenizer.rs; all 18 parser tests pass and clippy is clean. Saved the fix as a lesson so it is recognised next time.".into(),
                }),
            ],
            on_deny: vec![
                e(200, status(0, Thinking)),
                e(900, model(0, 4_300, 1_180, 14, 2_800)),
                e(100, EventKind::Thought {
                    agent: 0,
                    text: "Edit denied. I can't meet the criteria without changing the file, so stop and hand the patch over instead of retrying.".into(),
                }),
                e(150, vitals(0, 0.55, 0.60, "blocked: edit permission")),
                e(200, step(2, StepStatus::Failed)),
                e(200, EventKind::MemoryWrite {
                    agent: 0,
                    node: node(101, MemKind::Procedural, "propose diffs before editing src"),
                }),
                e(200, status(0, Waiting)),
                e(100, alert(Severity::Error, "GOAL BLOCKED — needs permission to edit src/tokenizer.rs")),
                e(100, EventKind::Message {
                    agent: Some(0),
                    text: "Blocked: the fix needs an edit to src/tokenizer.rs. The patch is in tool call #4 — approve a re-run or apply it yourself, then I'll verify.".into(),
                }),
            ],
        },
    ]
}

impl Demo {
    pub fn new() -> Self {
        Self {
            queue: script().into(),
            next_at: Instant::now() + Duration::from_millis(400),
            gate: None,
            rng: 0x2545_F491_4F6C_DD1D,
            busy: false,
            approved: None,
            idle_cycle: 0,
            keep_alive: true,
        }
    }

    /// Returns every scripted event that is due now.
    pub fn poll(&mut self) -> Vec<EventKind> {
        let mut out = Vec::new();
        let now = Instant::now();
        while self.gate.is_none() && now >= self.next_at {
            match self.queue.pop_front() {
                Some(Step::Emit(_, kind)) => {
                    if let EventKind::AgentStatus { state, .. } = &kind {
                        self.busy = state.is_active();
                    }
                    out.push(kind);
                    let delay = match self.queue.front() {
                        Some(Step::Emit(ms, _)) => *ms,
                        _ => 0,
                    };
                    self.next_at = now + Duration::from_millis(delay);
                }
                Some(Step::Gate {
                    tool,
                    on_approve,
                    on_deny,
                }) => {
                    self.busy = false;
                    self.gate = Some((tool, on_approve, on_deny));
                }
                None => {
                    self.busy = false;
                    if self.keep_alive && self.approved.is_some() {
                        // Script finished: keep living. Reflect, consolidate, wait.
                        let steps = self.idle_steps();
                        self.queue.extend(steps);
                        self.next_at = now + Duration::from_millis(1500);
                    }
                    break;
                }
            }
        }
        out
    }

    pub fn handle(&mut self, cmd: Command) -> Vec<EventKind> {
        match cmd {
            // The scripted demo keeps no secrets.
            Command::Secret { .. } => Vec::new(),
            Command::Decide { id, approve } => {
                let Some((tool, on_approve, on_deny)) = self.gate.take() else {
                    return Vec::new();
                };
                if tool != id {
                    self.gate = Some((tool, on_approve, on_deny));
                    return Vec::new();
                }
                self.approved = Some(approve);
                let branch = if approve { on_approve } else { on_deny };
                for s in branch.into_iter().rev() {
                    self.queue.push_front(s);
                }
                self.next_at = Instant::now() + Duration::from_millis(150);
                vec![EventKind::ToolDecision {
                    id,
                    approved: approve,
                    by: "you".into(),
                }]
            }
            Command::UserMessage(text) => {
                let reply = if self.gate.is_some() {
                    "Noted. I'm paused on the pending approval; I'll fold this in once it's decided."
                } else {
                    "Noted — added to the goal's context. (Demo: no model is connected yet.)"
                };
                self.queue.push_front(e(
                    500,
                    EventKind::Message {
                        agent: Some(0),
                        text: reply.into(),
                    },
                ));
                self.next_at = Instant::now() + Duration::from_millis(500);
                vec![EventKind::Message { agent: None, text }]
            }
            Command::Reply { text, .. } => self.handle(Command::UserMessage(text)),
            Command::Feedback { .. } | Command::Stop { .. } | Command::Restart => Vec::new(),
        }
    }

    /// Emits every scripted event up to the next approval gate, ignoring delays.
    #[cfg(test)]
    pub fn drain_until_gate(&mut self) -> Vec<EventKind> {
        // Without this, `poll` refills the queue with idle steps every time it
        // empties and this loop grows `out` until the machine runs out of memory.
        self.keep_alive = false;
        let mut out = Vec::new();
        while self.gate.is_none() && !self.queue.is_empty() {
            self.next_at = Instant::now();
            out.extend(self.poll());
        }
        self.keep_alive = true;
        out
    }

    /// Runs one idle reflection cycle to completion, ignoring delays.
    #[cfg(test)]
    pub fn run_idle_cycle(&mut self) -> Vec<EventKind> {
        let steps = self.idle_steps();
        self.queue.extend(steps);
        self.drain_until_gate()
    }

    /// What the agent does between goals: replay the episode, consolidate
    /// what it learned, revisit older memories, then rest.
    fn idle_steps(&mut self) -> Vec<Step> {
        use AgentState::*;
        let cycle = self.idle_cycle;
        self.idle_cycle += 1;
        if cycle == 0 {
            let (from, into, why) = if self.approved == Some(true) {
                (
                    vec![7, 100],
                    node(102, MemKind::Procedural, "comma bug x2: check ')' first"),
                    "Replaying the run. Memory #7 and today's lesson describe the same bug; merging them and raising its weight.",
                )
            } else {
                (
                    vec![6, 101],
                    node(103, MemKind::Procedural, "small diffs, proposed first"),
                    "Replaying the run. The denied edit and my note on small diffs say the same thing: propose before editing. Merging them.",
                )
            };
            return vec![
                e(1500, status(0, Reflecting)),
                e(900, thought(0, why)),
                e(
                    1400,
                    EventKind::MemoryRead {
                        agent: 0,
                        query: "self-review: today's episode".into(),
                        hits: from
                            .iter()
                            .map(|id| (*id, format!("memory #{id}")))
                            .collect(),
                    },
                ),
                e(
                    1600,
                    EventKind::MemoryConsolidated {
                        agent: 0,
                        from,
                        into,
                    },
                ),
                e(400, vitals(0, 0.90, 0.0, "memory consolidation")),
                e(
                    1800,
                    thought(
                        0,
                        "Nothing scheduled until the nightly test run at 02:00. Watching src/ for changes.",
                    ),
                ),
                e(600, status(0, Idle)),
            ];
        }
        const MUSINGS: [(&str, u32); 3] = [
            (
                "Revisiting the flaky io_timeout note from 11-07. Still unexplained; keeping it, flagged for the nightly run.",
                4,
            ),
            (
                "'Run tests before patching' paid off again today. Strengthening it.",
                3,
            ),
            (
                "Checked the schedule: nightly tests at 02:00. Nothing else pending.",
                12,
            ),
        ];
        let (text, id) = MUSINGS[(cycle as usize - 1) % MUSINGS.len()];
        vec![
            e(7000, status(0, Reflecting)),
            e(
                900,
                EventKind::MemoryRead {
                    agent: 0,
                    query: "idle review".into(),
                    hits: vec![(id, format!("memory #{id}"))],
                },
            ),
            e(700, thought(0, text)),
            e(500, vitals(0, 0.90, 0.0, &format!("memory #{id}"))),
            e(2500, status(0, Idle)),
        ]
    }

    /// Simulated generation speed in tokens/sec, 0 when nothing is generating.
    pub fn throughput(&mut self) -> u64 {
        if !self.busy {
            return 0;
        }
        // xorshift: cheap jitter so the sparkline looks alive.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        2 + self.rng % 5
    }
}
