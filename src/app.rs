//! TUI state. Harness state is derived from the event log via `apply`;
//! everything else here is view state (focus, selection, modals, hit areas).

use std::time::{Duration, Instant};

use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;

use crate::demo::Command;
use crate::event::{
    AgentId, AgentState, ContextSection, Event, EventKind, MemoryId, MemoryNode, Risk, Severity,
    StepStatus, TaskId, TaskState, ToolCallId,
};
use crate::live::MAIN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Panel {
    Agents,
    Center,
    Tools,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Timeline,
    Events,
    Memory,
}

impl Tab {
    pub const ALL: [Tab; 3] = [Tab::Timeline, Tab::Events, Tab::Memory];

    pub fn label(self) -> &'static str {
        match self {
            Tab::Timeline => "ACTIVITY",
            Tab::Events => "RAW EVENTS",
            Tab::Memory => "KNOWLEDGE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    AwaitingApproval,
    Running,
    Ok,
    Failed,
    Denied,
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: ToolCallId,
    /// The task that asked for it.
    pub task: Option<TaskId>,
    pub agent: AgentId,
    pub tool: String,
    pub args: String,
    pub risk: Risk,
    pub policy: String,
    pub status: ToolStatus,
    pub decided_by: Option<String>,
    pub output: Option<String>,
    pub duration_ms: Option<u64>,
    pub requested_at: Duration,
}

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: AgentId,
    pub parent: Option<AgentId>,
    pub name: String,
    pub purpose: String,
    pub state: AgentState,
    pub tokens: u64,
}

/// The window a secret is typed into.
#[derive(Clone, Debug, Default)]
pub struct SecretForm {
    pub name: String,
    pub value: String,
    /// What the agent said it is for, when the agent asked.
    pub why: String,
    /// Whether the agent asked for it (the name is then fixed, and closing
    /// the window tells the agent it was not given).
    pub asked: bool,
    /// Which of the two fields is being typed into.
    pub on_value: bool,
    pub error: String,
}

/// One task as the screen tracks it: a tab.
#[derive(Clone, Debug)]
pub struct TaskView {
    pub id: TaskId,
    pub title: String,
    /// Who asked: `you`, or a scheduled job.
    pub origin: String,
    pub state: TaskState,
    /// What it is doing, in a line.
    pub note: String,
    /// Unix seconds when it was asked and when it ended.
    pub opened: u64,
    pub ended: Option<u64>,
    pub goal: Option<Goal>,
    pub plan: Vec<PlanStep>,
    pub plan_version: u32,
    pub context: Vec<ContextSection>,
    pub tokens: u64,
    pub calls: u32,
    pub tools: u32,
    pub lookups: u32,
    /// Pages read and kept by knowledge research.
    pub researched: u32,
    /// Finished while another tab was open, and not looked at since.
    pub unread: bool,
}

#[derive(Clone, Debug)]
pub struct Goal {
    pub title: String,
    pub criteria: Vec<String>,
    pub token_budget: u64,
}

#[derive(Clone, Debug)]
pub struct PlanStep {
    pub text: String,
    pub status: StepStatus,
}

/// A long-term memory as the TUI tracks it.
#[derive(Clone, Debug)]
pub struct MemState {
    pub node: MemoryNode,
    pub born: Instant,
    /// Last time it was recalled or written.
    pub touched: Option<Instant>,
    pub recalls: u32,
    /// Set when consolidation merged it into another memory.
    pub absorbed: Option<Instant>,
}

impl MemState {
    fn new(node: MemoryNode, born: Instant) -> Self {
        Self {
            node,
            born,
            touched: None,
            recalls: 0,
            absorbed: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modal {
    Tool(ToolCallId),
    Event(usize),
    /// How a knowledge lookup found its passages; the index is its event.
    Lookup(usize),
    Context(usize),
    Agent(usize),
    /// One remembered thing, opened from the map of memories.
    Memory(crate::event::MemoryId),
    Help,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Accept,
    Deny,
    Close,
    OpenPending,
}

/// Screen regions recorded during rendering so mouse clicks can be resolved.
#[derive(Default)]
pub struct Hits {
    pub panels: Vec<(Rect, Panel)>,
    pub tabs: Vec<(Rect, Tab)>,
    /// The task tabs; `None` is the overview.
    pub task_tabs: Vec<(Rect, Option<TaskId>)>,
    /// Rows of the overview's task board.
    pub board: Vec<(Rect, TaskId)>,
    pub agents: Vec<(Rect, usize)>,
    pub tools: Vec<(Rect, usize)>,
    pub context: Vec<(Rect, usize)>,
    /// The memories drawn on the map.
    pub memories: Vec<(Rect, crate::event::MemoryId)>,
    pub center: Vec<(Rect, usize)>,
    pub buttons: Vec<(Rect, Button)>,
    pub modal: Option<Rect>,
    /// Panel interiors whose fully empty rows get the dot-grid texture.
    pub texture: Vec<Rect>,
}

fn inside(r: Rect, col: u16, row: u16) -> bool {
    col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
}

fn find<T: Copy>(list: &[(Rect, T)], col: u16, row: u16) -> Option<T> {
    list.iter()
        .find(|(r, _)| inside(*r, col, row))
        .map(|(_, v)| *v)
}

pub const BOOT_LINES: [(&str, &str); 8] = [
    ("event log", "mounted  ~/.conscious/events.log"),
    ("guardrails", "policy loaded (12 rules, 3 protected paths)"),
    ("sandbox", "landlock + seccomp profile armed"),
    ("model", "gemma-4-e4b q4_k_m · cpu ×4"),
    ("memory", "1,204 entries indexed (hybrid: bm25 + vector)"),
    ("mcp", "2 servers connected (filesystem, github)"),
    ("scheduler", "3 schedules armed"),
    ("core agent", "online"),
];
pub const BOOT_STEP: Duration = Duration::from_millis(140);

pub struct App {
    pub started: Instant,
    pub boot: Option<Instant>,
    pub quit: bool,
    pub tick: u64,

    // Harness state, derived from events.
    pub events: Vec<Event>,
    /// Every task, oldest first.
    pub tasks: Vec<TaskView>,
    /// The task whose tab is open; `None` is the overview.
    pub task_sel: Option<TaskId>,
    /// The row selected on the overview's task board.
    pub board_sel: usize,
    /// Open the next task this screen asks for as soon as it exists.
    pub follow_next: bool,
    /// Whether the main conversation has been shown once: its tab is the
    /// one a screen opens on.
    pub main_seen: bool,
    /// The reply each task is writing right now, as far as it has come.
    pub drafts: Vec<(TaskId, String)>,
    /// The window a secret is typed into, when it is open. What is typed
    /// there is held here only until it is sent to the secret store: it is
    /// never put in the input line, an event or the timeline.
    pub secret_form: Option<SecretForm>,
    /// Set while earlier sessions' events are being put back on screen.
    pub restoring: bool,
    /// Whether branches that have ended are listed too. Off to begin with:
    /// what they found is merged into main, so only work in hand is shown.
    pub show_ended: bool,
    /// The task the input line will reply to; `None` starts a new one.
    pub input_task: Option<TaskId>,
    /// Whether a real harness is behind the screen, not the scripted demo.
    pub live: bool,
    pub goal: Option<Goal>,
    pub plan: Vec<PlanStep>,
    pub plan_version: u32,
    pub agents: Vec<Agent>,
    pub tools: Vec<ToolCall>,
    pub context: Vec<ContextSection>,
    pub window: u32,
    pub signals: Vec<(Duration, Severity, String)>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub tokens_cached: u64,
    pub model_calls: u64,
    pub last_model: String,
    pub throughput: Vec<u64>,
    /// When the newest throughput sample arrived, so the chart can slide
    /// smoothly toward the next one.
    pub sampled_at: Instant,

    // Inner life, derived from events.
    pub memory: Vec<MemState>,
    /// Consolidation links (from, into, when).
    pub links: Vec<(MemoryId, MemoryId, Instant)>,
    pub confidence: f32,
    pub surprise: Vec<f32>,
    /// What the agent says it is attending to right now.
    pub attention: String,
    /// Latest thought or message: (agent, text, arrived).
    pub thought: Option<(AgentId, String, Instant)>,
    /// Arrival times of recent events, for the heart rate.
    pub pulse_log: Vec<Instant>,

    // View state.
    pub focus: Panel,
    pub tab: Tab,
    pub agent_sel: usize,
    pub tool_sel: usize,
    pub center_sel: Option<usize>,
    pub center_scroll: usize,
    /// Set when keyboard selection moves, so the renderer scrolls it into view.
    pub ensure_sel: bool,
    pub visible_center: Vec<usize>,
    pub agent_filter: Option<AgentId>,
    pub modal: Option<Modal>,
    pub modal_scroll: u16,
    /// Seconds to add to UTC for local time, for stamping earlier events.
    pub utc_offset: i64,
    /// Which passage is selected in the lookup view.
    pub lookup_sel: usize,
    pub input: String,
    pub hits: Hits,
    pub outbox: Vec<Command>,
}

impl App {
    /// Seconds between throughput samples: one every third tick of 80 ms.
    pub const SAMPLE_SECS: f64 = 0.24;

    pub fn new(with_boot: bool) -> Self {
        let now = Instant::now();
        Self {
            started: now,
            boot: with_boot.then_some(now),
            quit: false,
            tick: 0,
            events: Vec::new(),
            tasks: Vec::new(),
            task_sel: None,
            board_sel: 0,
            follow_next: false,
            main_seen: false,
            drafts: Vec::new(),
            show_ended: false,
            secret_form: None,
            restoring: false,
            input_task: None,
            live: false,
            goal: None,
            plan: Vec::new(),
            plan_version: 0,
            agents: Vec::new(),
            tools: Vec::new(),
            context: Vec::new(),
            window: 32_768,
            signals: Vec::new(),
            tokens_in: 0,
            tokens_out: 0,
            tokens_cached: 0,
            model_calls: 0,
            last_model: "gemma-4-e4b".into(),
            throughput: vec![0; 120],
            sampled_at: now,
            memory: Vec::new(),
            links: Vec::new(),
            confidence: 0.0,
            surprise: vec![0.0; 40],
            attention: String::new(),
            thought: None,
            pulse_log: Vec::new(),
            focus: Panel::Center,
            tab: Tab::Timeline,
            agent_sel: 0,
            tool_sel: 0,
            center_sel: None,
            center_scroll: 0,
            ensure_sel: false,
            visible_center: Vec::new(),
            agent_filter: None,
            modal: None,
            modal_scroll: 0,
            utc_offset: 0,
            lookup_sel: 0,
            input: String::new(),
            hits: Hits::default(),
            outbox: Vec::new(),
        }
    }

    pub fn booting(&self) -> bool {
        match self.boot {
            Some(t) => t.elapsed() < BOOT_STEP * (BOOT_LINES.len() as u32 + 5),
            None => false,
        }
    }

    pub fn agent_name(&self, id: AgentId) -> String {
        self.agents
            .iter()
            .find(|a| a.id == id)
            .map(|a| a.name.clone())
            .unwrap_or_else(|| format!("agent{id}"))
    }

    /// The tool call with this id. If an id was ever reused, the newest
    /// call wins: that is the one something can still happen to.
    pub fn tool(&self, id: ToolCallId) -> Option<&ToolCall> {
        self.tools.iter().rev().find(|t| t.id == id)
    }

    pub fn pending(&self) -> Vec<&ToolCall> {
        self.tools
            .iter()
            .filter(|t| t.status == ToolStatus::AwaitingApproval)
            .collect()
    }

    pub fn context_used(&self) -> u32 {
        self.context.iter().map(|s| s.tokens).sum()
    }

    /// A key pressed while the secret window is open. Everything typed
    /// goes to the window and nowhere else.
    fn secret_key(&mut self, k: KeyEvent) {
        let Some(form) = &mut self.secret_form else {
            return;
        };
        match k.code {
            // Closing it without a value tells a waiting agent "no".
            KeyCode::Esc => {
                if form.asked {
                    self.outbox.push(Command::Secret {
                        name: form.name.clone(),
                        value: None,
                    });
                }
                self.secret_form = None;
            }
            KeyCode::Tab | KeyCode::BackTab | KeyCode::Up | KeyCode::Down => {
                if !form.asked {
                    form.on_value = !form.on_value;
                }
            }
            KeyCode::Enter => {
                if !form.on_value {
                    form.on_value = true;
                    return;
                }
                if let Err(why) = crate::secrets::valid_name(&form.name) {
                    form.error = why;
                    form.on_value = false;
                    return;
                }
                if form.value.trim().is_empty() {
                    form.error = "type the secret's value, or press esc to leave".into();
                    return;
                }
                self.outbox.push(Command::Secret {
                    name: form.name.clone(),
                    value: Some(crate::secrets::Hidden(std::mem::take(&mut form.value))),
                });
                self.secret_form = None;
            }
            KeyCode::Backspace => {
                form.error.clear();
                if form.on_value {
                    form.value.pop();
                } else {
                    form.name.pop();
                }
            }
            KeyCode::Char(c) => {
                form.error.clear();
                if form.on_value {
                    form.value.push(c);
                } else {
                    // Names are capitals: typed in lower case, they are raised.
                    form.name.push(c.to_ascii_uppercase());
                }
            }
            _ => {}
        }
    }

    /// The memories on the map, newest first: those not merged away.
    pub fn shown_memories(&self) -> Vec<&MemState> {
        let mut shown: Vec<&MemState> = self
            .memory
            .iter()
            .filter(|m| m.absorbed.is_none())
            .collect();
        shown.sort_by_key(|m| std::cmp::Reverse(m.node.id));
        shown
    }

    /// Moves the open memory to the next or previous one.
    fn step_memory(&mut self, from: crate::event::MemoryId, delta: i32) {
        let ids: Vec<_> = self.shown_memories().iter().map(|m| m.node.id).collect();
        if ids.is_empty() {
            self.modal = None;
            return;
        }
        let at = ids.iter().position(|id| *id == from).unwrap_or(0) as i32;
        let next = (at + delta).clamp(0, ids.len() as i32 - 1) as usize;
        self.modal = Some(Modal::Memory(ids[next]));
        self.modal_scroll = 0;
    }

    /// Takes the reply a task is writing, as far as it has come; empty
    /// text means it is no longer writing.
    pub fn set_draft(&mut self, task: TaskId, text: String) {
        self.drafts.retain(|(t, _)| *t != task);
        if !text.is_empty() {
            self.drafts.push((task, text));
        }
    }

    /// What the task whose tab is open is writing right now.
    pub fn draft(&self) -> Option<&str> {
        let open = self.task_sel?;
        self.drafts
            .iter()
            .find(|(t, _)| *t == open)
            .map(|(_, text)| text.as_str())
    }

    pub fn task(&self, id: TaskId) -> Option<&TaskView> {
        self.tasks.iter().find(|t| t.id == id)
    }

    /// The task the header, plan and context panels describe: the open tab,
    /// else the task that is still going (the newest), else the last one.
    pub fn shown_task(&self) -> Option<&TaskView> {
        self.task_sel
            .and_then(|id| self.task(id))
            .or_else(|| {
                self.tasks
                    .iter()
                    .rev()
                    .find(|t| t.state.is_open() && t.state != TaskState::Queued)
            })
            .or_else(|| self.tasks.iter().rev().find(|t| t.state.is_open()))
            .or(self.tasks.last())
    }

    /// Whether a task has a tool call waiting for the user.
    pub fn task_needs_you(&self, id: TaskId) -> bool {
        self.tools
            .iter()
            .any(|t| t.task == Some(id) && t.status == ToolStatus::AwaitingApproval)
    }

    /// The branches (and routine runs): every task but the main
    /// conversation, oldest first.
    pub fn branches(&self) -> impl DoubleEndedIterator<Item = &TaskView> {
        self.tasks.iter().filter(|t| t.id != MAIN)
    }

    /// The branches that have a tab and a row on the board: those still
    /// going, the one whose tab is open, and the ended ones too when the
    /// user has asked to see them (`e`).
    pub fn listed_branches(&self) -> Vec<&TaskView> {
        self.branches()
            .filter(|t| self.show_ended || t.state.is_open() || self.task_sel == Some(t.id))
            .collect()
    }

    /// Ended branches that are not listed just now.
    pub fn hidden_branches(&self) -> usize {
        self.branches().count() - self.listed_branches().len()
    }

    /// Branches that have not finished.
    pub fn open_tasks(&self) -> usize {
        self.branches().filter(|t| t.state.is_open()).count()
    }

    /// Opens a task's tab, or the overview.
    pub fn select_task(&mut self, id: Option<TaskId>) {
        self.task_sel = id.filter(|id| self.task(*id).is_some());
        self.center_sel = None;
        self.center_scroll = 0;
        if let Some(task) = self
            .task_sel
            .and_then(|id| self.tasks.iter_mut().find(|t| t.id == id))
        {
            task.unread = false;
        }
        self.show();
    }

    /// Steps through the tabs: overview, then tasks newest first.
    fn step_task(&mut self, delta: i32) {
        let mut order: Vec<Option<TaskId>> = Vec::new();
        if self.task(MAIN).is_some() {
            order.push(Some(MAIN));
        }
        order.push(None);
        order.extend(self.listed_branches().iter().rev().map(|t| Some(t.id)));
        let at = order.iter().position(|t| *t == self.task_sel).unwrap_or(0) as i32;
        let next = (at + delta).rem_euclid(order.len() as i32) as usize;
        self.select_task(order[next]);
    }

    /// Puts the shown task's goal, plan and context in the panels.
    fn show(&mut self) {
        let Some(task) = self.shown_task() else {
            return;
        };
        let (goal, plan, version, context) = (
            // A task that was only talked through has no goal of its own:
            // what was asked stands in for it.
            task.goal.clone().or_else(|| {
                Some(Goal {
                    title: task.title.clone(),
                    criteria: Vec::new(),
                    token_budget: crate::agent::TOKEN_BUDGET,
                })
            }),
            task.plan.clone(),
            task.plan_version,
            task.context.clone(),
        );
        self.goal = goal;
        self.plan = plan;
        self.plan_version = version;
        self.context = context;
    }

    pub fn goal_state(&self) -> (&'static str, bool) {
        if let Some(task) = self.shown_task() {
            if self.task_needs_you(task.id) {
                return ("AWAITING YOU", true);
            }
            return match task.state {
                TaskState::Queued => ("QUEUED", false),
                TaskState::Working => ("RUNNING", false),
                TaskState::Researching => ("RESEARCHING", false),
                TaskState::Done if task.id == MAIN => ("READY", false),
                TaskState::Done => ("FINISHED", false),
                TaskState::Failed => ("FAILED", true),
                TaskState::Stopped => ("STOPPED", true),
            };
        }
        let core = self.agents.iter().find(|a| a.parent.is_none());
        if self.goal.is_none() {
            return ("IDLE", false);
        }
        if !self.pending().is_empty() {
            return ("AWAITING YOU", true);
        }
        if self.plan.iter().any(|s| s.status == StepStatus::Failed) {
            return ("BLOCKED", true);
        }
        if !self.plan.is_empty() && self.plan.iter().all(|s| s.status == StepStatus::Done) {
            return ("VERIFIED", false);
        }
        match core.map(|a| a.state) {
            Some(AgentState::Done) => ("VERIFIED", false),
            Some(AgentState::Idle) => ("FINISHED", false),
            Some(AgentState::Failed) => ("STOPPED", true),
            _ => ("RUNNING", false),
        }
    }

    /// Applies an event that belongs to no task.
    pub fn apply(&mut self, kind: EventKind) {
        self.apply_in(None, kind);
    }

    /// Keeps the task an event belongs to up to date with it.
    fn track(&mut self, id: TaskId, kind: &EventKind) {
        let wall = crate::learn::now();
        if !self.tasks.iter().any(|t| t.id == id) {
            // An event of a task whose opening is further back than what
            // this screen was sent.
            self.tasks.push(TaskView {
                id,
                title: String::new(),
                origin: String::new(),
                state: TaskState::Working,
                note: String::new(),
                opened: wall,
                ended: None,
                goal: None,
                plan: Vec::new(),
                plan_version: 0,
                context: Vec::new(),
                tokens: 0,
                calls: 0,
                tools: 0,
                lookups: 0,
                researched: 0,
                unread: false,
            });
        }
        let selected = self.task_sel == Some(id);
        let Some(task) = self.tasks.iter_mut().find(|t| t.id == id) else {
            return;
        };
        match kind {
            EventKind::TaskOpened { title, origin } => {
                task.title = title.clone();
                task.origin = origin.clone();
            }
            EventKind::TaskUpdate { state, note } => {
                let was_open = task.state.is_open();
                task.state = *state;
                task.note = note.clone();
                if state.is_open() {
                    task.ended = None;
                } else {
                    task.ended = Some(wall);
                    task.unread = was_open && !selected;
                }
            }
            // Main is named by the latest thing said in it.
            EventKind::Message { agent: None, text } if task.title.is_empty() || id == MAIN => {
                task.title = text
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .chars()
                    .take(90)
                    .collect();
            }
            EventKind::GoalSet {
                title,
                criteria,
                token_budget,
            } => {
                task.goal = Some(Goal {
                    title: title.clone(),
                    criteria: criteria.clone(),
                    token_budget: *token_budget,
                });
            }
            EventKind::PlanUpdated { version, steps } => {
                task.plan_version = *version;
                task.plan = steps
                    .iter()
                    .map(|s| PlanStep {
                        text: s.clone(),
                        status: StepStatus::Pending,
                    })
                    .collect();
            }
            EventKind::StepUpdate { index, status } => {
                if let Some(s) = task.plan.get_mut(*index) {
                    s.status = *status;
                }
            }
            EventKind::Context { sections, .. } => task.context = sections.clone(),
            EventKind::ModelCall {
                tokens_in,
                tokens_out,
                ..
            } => {
                task.calls += 1;
                task.tokens += (*tokens_in + *tokens_out) as u64;
            }
            EventKind::ToolRequested { tool, args, .. } => {
                task.tools += 1;
                if task.state == TaskState::Working {
                    task.note = format!("{tool} {}", args.lines().next().unwrap_or_default());
                }
            }
            EventKind::KnowledgeLookup { .. } => task.lookups += 1,
            EventKind::Research { phase, .. } if phase == "kept" => task.researched += 1,
            _ => {}
        }
    }

    /// Applies an event, with the task it belongs to.
    pub fn apply_in(&mut self, task: Option<TaskId>, kind: EventKind) {
        let at = self.started.elapsed();
        let now = Instant::now();
        if let Some(id) = task {
            // The finished reply takes the place of what was being written.
            if matches!(&kind, EventKind::Message { agent: Some(_), .. }) {
                self.set_draft(id, String::new());
            }
            self.track(id, &kind);
            // A screen opens on the main conversation. Branches open in
            // the background: their tabs appear, and the user goes to them.
            if id == MAIN && !self.main_seen {
                self.main_seen = true;
                self.follow_next = false;
                self.task_sel = Some(id);
                self.center_sel = None;
                self.center_scroll = 0;
            }
        }
        match &kind {
            // The agent asks for a secret: its window opens. (Not for one
            // asked in an earlier session, whose task is gone.)
            EventKind::SecretAsked { name, why } => {
                if !self.restoring {
                    self.secret_form = Some(SecretForm {
                        name: name.clone(),
                        value: String::new(),
                        why: why.clone(),
                        asked: true,
                        on_value: true,
                        error: String::new(),
                    });
                }
            }
            EventKind::SecretAnswered { name, .. } => {
                if self
                    .secret_form
                    .as_ref()
                    .is_some_and(|f| f.asked && f.name == *name)
                {
                    self.secret_form = None;
                }
            }
            EventKind::TaskOpened { .. }
            | EventKind::Merged { .. }
            | EventKind::TaskUpdate { .. }
            | EventKind::CriteriaChecked { .. }
            | EventKind::Research { .. } => {}
            EventKind::GoalSet {
                title,
                criteria,
                token_budget,
            } => {
                self.goal = Some(Goal {
                    title: title.clone(),
                    criteria: criteria.clone(),
                    token_budget: *token_budget,
                });
            }
            EventKind::PlanUpdated { version, steps } => {
                self.plan_version = *version;
                self.plan = steps
                    .iter()
                    .map(|s| PlanStep {
                        text: s.clone(),
                        status: StepStatus::Pending,
                    })
                    .collect();
            }
            EventKind::StepUpdate { index, status } => {
                if let Some(s) = self.plan.get_mut(*index) {
                    s.status = *status;
                }
            }
            EventKind::AgentSpawned {
                id,
                parent,
                name,
                purpose,
            } => self.agents.push(Agent {
                id: *id,
                parent: *parent,
                name: name.clone(),
                purpose: purpose.clone(),
                state: AgentState::Idle,
                tokens: 0,
            }),
            EventKind::AgentStatus { id, state } => {
                if let Some(a) = self.agents.iter_mut().find(|a| a.id == *id) {
                    a.state = *state;
                }
            }
            EventKind::ModelCall {
                agent,
                model,
                tokens_in,
                cached_in,
                tokens_out,
                ..
            } => {
                self.tokens_in += *tokens_in as u64;
                self.tokens_out += *tokens_out as u64;
                self.tokens_cached += *cached_in as u64;
                self.model_calls += 1;
                self.last_model = model.clone();
                if let Some(a) = self.agents.iter_mut().find(|a| a.id == *agent) {
                    a.tokens += (*tokens_in + *tokens_out) as u64;
                }
            }
            EventKind::ToolRequested {
                id,
                agent,
                tool,
                args,
                risk,
                needs_approval,
                policy,
            } => {
                self.tools.push(ToolCall {
                    id: *id,
                    task,
                    agent: *agent,
                    tool: tool.clone(),
                    args: args.clone(),
                    risk: *risk,
                    policy: policy.clone(),
                    status: if *needs_approval {
                        ToolStatus::AwaitingApproval
                    } else {
                        ToolStatus::Running
                    },
                    decided_by: None,
                    output: None,
                    duration_ms: None,
                    requested_at: at,
                });
                if *needs_approval && self.modal.is_none() {
                    self.open(Modal::Tool(*id));
                }
            }
            EventKind::ToolDecision { id, approved, by } => {
                if let Some(t) = self.tools.iter_mut().rev().find(|t| t.id == *id) {
                    t.decided_by = Some(by.clone());
                    t.status = if *approved {
                        ToolStatus::Running
                    } else {
                        ToolStatus::Denied
                    };
                }
            }
            EventKind::ToolFinished {
                id,
                ok,
                output,
                duration_ms,
            } => {
                if let Some(t) = self.tools.iter_mut().rev().find(|t| t.id == *id) {
                    t.status = if *ok {
                        ToolStatus::Ok
                    } else {
                        ToolStatus::Failed
                    };
                    t.output = Some(output.clone());
                    t.duration_ms = Some(*duration_ms);
                }
            }
            EventKind::Context { sections, window } => {
                self.context = sections.clone();
                self.window = *window;
            }
            EventKind::Alert { severity, text } => {
                self.signals.push((at, *severity, text.clone()));
            }
            EventKind::Thought { agent, text } => {
                self.thought = Some((*agent, text.clone(), now));
            }
            EventKind::Message {
                agent: Some(agent),
                text,
            } => {
                self.thought = Some((*agent, text.clone(), now));
            }
            EventKind::Message { agent: None, .. } => {}
            EventKind::MemoryIndexed { nodes } => {
                // Already-known memories: no birth flash.
                let settled = now.checked_sub(Duration::from_secs(30)).unwrap_or(now);
                for n in nodes {
                    if !self.memory.iter().any(|m| m.node.id == n.id) {
                        self.memory.push(MemState::new(n.clone(), settled));
                    }
                }
            }
            EventKind::MemoryRead { hits, .. } => {
                for (id, _) in hits {
                    if let Some(m) = self.memory.iter_mut().find(|m| m.node.id == *id) {
                        m.touched = Some(now);
                        m.recalls += 1;
                    }
                }
            }
            EventKind::KnowledgeLookup { picks, .. } => {
                for pick in picks {
                    if let Some(m) = self.memory.iter_mut().find(|m| m.node.id == pick.memory) {
                        m.touched = Some(now);
                        m.recalls += 1;
                    }
                }
            }
            EventKind::MemoryWrite { node, .. } => {
                let mut m = MemState::new(node.clone(), now);
                m.touched = Some(now);
                self.memory.push(m);
            }
            EventKind::MemoryConsolidated { from, into, .. } => {
                for m in self.memory.iter_mut().filter(|m| from.contains(&m.node.id)) {
                    m.absorbed = Some(now);
                }
                let mut merged = MemState::new(into.clone(), now);
                merged.touched = Some(now);
                self.memory.push(merged);
                for id in from {
                    self.links.push((*id, into.id, now));
                }
            }
            EventKind::Vitals {
                confidence,
                surprise,
                focus,
                ..
            } => {
                self.confidence = *confidence;
                self.surprise.remove(0);
                self.surprise.push(*surprise);
                self.attention = focus.clone();
            }
        }
        self.pulse_log.push(now);
        self.pulse_log
            .retain(|t| now.duration_since(*t) < Duration::from_secs(60));
        self.events.push(Event {
            seq: self.events.len() as u64 + 1,
            at,
            wall: crate::learn::now(),
            task,
            restored: false,
            kind,
        });
        self.show();
    }

    /// Gives the event just applied the time it really happened at.
    fn backdate(&mut self, task: Option<TaskId>, wall: u64) {
        if let Some(event) = self.events.last_mut() {
            event.at = Duration::ZERO;
            event.wall = wall;
            event.restored = true;
        }
        if let Some(view) = task.and_then(|id| self.tasks.iter_mut().find(|t| t.id == id)) {
            if view.opened > wall || view.calls + view.tools == 0 {
                view.opened = view.opened.min(wall);
            }
            if view.ended.is_some() {
                view.ended = Some(wall);
            }
        }
    }

    /// Replays events from earlier sessions, so the timeline, memory view
    /// and counters continue instead of starting blank. Nothing in them is
    /// re-run; they only rebuild what is shown.
    pub fn restore(&mut self, recorded: Vec<crate::timeline::Recorded>) {
        self.restoring = true;
        self.restore_events(recorded);
        self.restoring = false;
    }

    fn restore_events(&mut self, recorded: Vec<crate::timeline::Recorded>) {
        for r in recorded {
            // Start-of-session status lines and index loads describe a
            // session that is over; replaying them would only add clutter.
            let stale = r.task.is_none()
                && matches!(
                    r.kind,
                    EventKind::MemoryIndexed { .. }
                        | EventKind::Alert {
                            severity: Severity::Info,
                            ..
                        }
                );
            if stale {
                continue;
            }
            self.apply_in(r.task, r.kind);
            self.backdate(r.task, r.wall);
        }
        // A task that was still going when the last session ended did not
        // finish.
        for task in &mut self.tasks {
            if task.state.is_open() {
                task.state = TaskState::Stopped;
                task.note = "the session ended before it finished".into();
                task.ended.get_or_insert(task.opened);
            }
            task.unread = false;
        }
        // What was in motion when the last session ended is not in motion now.
        self.thought = None;
        self.pulse_log.clear();
        self.modal = None;
        self.links.clear();
        for agent in &mut self.agents {
            agent.state = AgentState::Idle;
        }
        // A tool call left waiting when the session ended was never run.
        for tool in &mut self.tools {
            if matches!(
                tool.status,
                ToolStatus::AwaitingApproval | ToolStatus::Running
            ) {
                tool.status = ToolStatus::Denied;
                tool.decided_by = Some("session ended".into());
            }
        }
        // The memory view is rebuilt from the store itself on start, which
        // knows what each memory has since been understood to be.
        self.memory.clear();
    }

    /// Catches up with a harness that is already running in the background:
    /// `since` is what has happened since it started, and unlike `restore`
    /// it is taken as true now. A call still waiting for approval is
    /// waiting on this screen too, and agents are in the state they were
    /// last put in. `memory` is what is remembered at this moment.
    pub fn attach(
        &mut self,
        since: Vec<crate::timeline::Recorded>,
        memory: Vec<MemoryNode>,
        core: &str,
        busy: bool,
    ) {
        for r in since {
            self.apply_in(r.task, r.kind);
            self.backdate(r.task, r.wall);
        }
        for task in &mut self.tasks {
            task.unread = false;
        }
        // None of that happened just now, so nothing should flash.
        self.pulse_log.clear();
        self.links.clear();
        if !busy {
            self.thought = None;
        }
        // The only prompt worth opening is one that is still undecided.
        let waiting = self.pending().first().map(|t| t.id);
        self.modal = waiting.map(Modal::Tool);
        // The core agent was announced when the harness started, which may
        // be further back than `since` reaches.
        if !self.agents.iter().any(|a| a.parent.is_none()) {
            self.agents.insert(
                0,
                Agent {
                    id: 0,
                    parent: None,
                    name: "core".into(),
                    purpose: format!("talks to {core}"),
                    state: AgentState::Idle,
                    tokens: 0,
                },
            );
        }
        // The store itself says what is remembered now.
        self.memory.clear();
        let settled = Instant::now()
            .checked_sub(Duration::from_secs(30))
            .unwrap_or_else(Instant::now);
        self.memory
            .extend(memory.into_iter().map(|n| MemState::new(n, settled)));
    }

    /// Real event rate over the last minute: the agent's "heart rate".
    pub fn events_per_min(&self) -> usize {
        self.pulse_log.len()
    }

    pub fn on_tick(&mut self, tok_s: u64) {
        self.tick = self.tick.wrapping_add(1);
        if self.tick.is_multiple_of(3) {
            self.throughput.remove(0);
            self.throughput.push(tok_s);
            self.sampled_at = Instant::now();
        }
        // Surprise is a spike that decays, like a startle response.
        if self.tick.is_multiple_of(5) {
            let last = *self.surprise.last().unwrap_or(&0.0);
            self.surprise.remove(0);
            self.surprise.push(last * 0.82);
        }
    }

    fn open(&mut self, modal: Modal) {
        self.modal = Some(modal);
        self.modal_scroll = 0;
        self.lookup_sel = 0;
    }

    fn decide(&mut self, id: ToolCallId, approve: bool) {
        if self.tool(id).map(|t| t.status) == Some(ToolStatus::AwaitingApproval) {
            self.outbox.push(Command::Decide { id, approve });
            self.modal = None;
        }
    }

    fn modal_pending_tool(&self) -> Option<ToolCallId> {
        match self.modal {
            Some(Modal::Tool(id))
                if self.tool(id).map(|t| t.status) == Some(ToolStatus::AwaitingApproval) =>
            {
                Some(id)
            }
            _ => None,
        }
    }

    fn submit_input(&mut self) {
        let text = self.input.trim().to_string();
        self.input.clear();
        if !text.is_empty() {
            match self.input_task {
                Some(task) => self.outbox.push(Command::Reply { task, text }),
                None => {
                    self.outbox.push(Command::UserMessage(text));
                    self.follow_next = true;
                }
            }
            self.center_scroll = 0;
            self.center_sel = None;
        }
    }

    fn open_center(&mut self, idx: usize) {
        self.center_sel = Some(idx);
        let modal = match self.events.get(idx).map(|e| &e.kind) {
            Some(EventKind::ToolRequested { id, .. }) if self.tab == Tab::Timeline => {
                Modal::Tool(*id)
            }
            Some(EventKind::KnowledgeLookup { .. }) => Modal::Lookup(idx),
            Some(_) => Modal::Event(idx),
            None => return,
        };
        self.open(modal);
    }

    fn move_selection(&mut self, delta: i32) {
        let step = |cur: usize, len: usize| -> usize {
            if len == 0 {
                return 0;
            }
            (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize
        };
        match self.focus {
            Panel::Agents => self.agent_sel = step(self.agent_sel, self.agents.len()),
            Panel::Tools => self.tool_sel = step(self.tool_sel, self.tools.len()),
            Panel::Center | Panel::Input if self.on_board() => {
                self.board_sel = step(self.board_sel, self.tasks.len());
            }
            Panel::Center | Panel::Input => {
                let vis = &self.visible_center;
                if vis.is_empty() {
                    return;
                }
                let pos = self
                    .center_sel
                    .and_then(|s| vis.iter().position(|&v| v == s))
                    .unwrap_or(vis.len() - 1);
                let new = step(pos, vis.len());
                self.center_sel = Some(vis[new]);
                self.ensure_sel = true;
                if new == vis.len() - 1 && delta > 0 {
                    self.center_scroll = 0;
                }
            }
        }
    }

    /// Whether the centre shows the overview's task board.
    pub fn on_board(&self) -> bool {
        self.task_sel.is_none() && self.tab == Tab::Timeline && self.branches().next().is_some()
    }

    /// The task on the selected row of the board (newest first).
    pub fn board_task(&self) -> Option<TaskId> {
        self.listed_branches()
            .iter()
            .rev()
            .nth(self.board_sel)
            .map(|t| t.id)
    }

    /// Opens the input line: to reply in the open task if it has answered,
    /// otherwise for a new task.
    fn start_input(&mut self, new: bool) {
        let reply_to = self
            .task_sel
            .filter(|id| *id != MAIN && self.task(*id).is_some_and(|t| !t.state.is_open()));
        self.input_task = if new { None } else { reply_to };
        self.focus = Panel::Input;
    }

    fn toggle_filter(&mut self) {
        let id = self.agents.get(self.agent_sel).map(|a| a.id);
        self.agent_filter = if self.agent_filter == id { None } else { id };
        self.center_scroll = 0;
        self.center_sel = None;
    }

    fn cycle_focus(&mut self, back: bool) {
        let order = [Panel::Agents, Panel::Center, Panel::Tools, Panel::Input];
        let i = order.iter().position(|p| *p == self.focus).unwrap_or(1);
        let n = order.len();
        self.focus = order[if back { (i + n - 1) % n } else { (i + 1) % n }];
    }

    pub fn on_key(&mut self, k: KeyEvent) {
        if k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }
        if self.booting() {
            self.boot = None;
            return;
        }
        if let Some(Modal::Lookup(idx)) = self.modal {
            let picks = match self.events.get(idx).map(|e| &e.kind) {
                Some(EventKind::KnowledgeLookup { picks, .. }) => picks.len(),
                _ => 0,
            };
            match k.code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => self.modal = None,
                KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                    self.lookup_sel = (self.lookup_sel + 1).min(picks.saturating_sub(1));
                    self.modal_scroll = 0;
                }
                KeyCode::Up | KeyCode::Char('k') | KeyCode::BackTab => {
                    self.lookup_sel = self.lookup_sel.saturating_sub(1);
                    self.modal_scroll = 0;
                }
                // The passage text can be longer than the view.
                KeyCode::PageDown | KeyCode::Char(' ') => self.modal_scroll += 5,
                KeyCode::PageUp => self.modal_scroll = self.modal_scroll.saturating_sub(5),
                _ => {}
            }
            return;
        }
        if self.secret_form.is_some() {
            self.secret_key(k);
            return;
        }
        if let Some(Modal::Memory(id)) = self.modal {
            match k.code {
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter => self.modal = None,
                KeyCode::Down | KeyCode::Right | KeyCode::Char('j') | KeyCode::Tab => {
                    self.step_memory(id, 1)
                }
                KeyCode::Up | KeyCode::Left | KeyCode::Char('k') | KeyCode::BackTab => {
                    self.step_memory(id, -1)
                }
                _ => {}
            }
            return;
        }
        if let Some(modal) = self.modal {
            match k.code {
                KeyCode::Esc | KeyCode::Char('q') => self.modal = None,
                KeyCode::Char('a') | KeyCode::Char('y') | KeyCode::Enter => {
                    if let Some(id) = self.modal_pending_tool() {
                        self.decide(id, true);
                    } else if k.code == KeyCode::Enter {
                        self.modal = None;
                    }
                }
                KeyCode::Char('d') | KeyCode::Char('n') => {
                    if let Some(id) = self.modal_pending_tool() {
                        self.decide(id, false);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => self.modal_scroll += 1,
                KeyCode::Up | KeyCode::Char('k') => {
                    self.modal_scroll = self.modal_scroll.saturating_sub(1)
                }
                _ => {}
            }
            let _ = modal;
            return;
        }
        if self.focus == Panel::Input {
            match k.code {
                KeyCode::Esc => self.focus = Panel::Center,
                KeyCode::Enter => self.submit_input(),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Tab => self.cycle_focus(false),
                KeyCode::Char(c) => self.input.push(c),
                _ => {}
            }
            return;
        }
        match k.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Tab => self.cycle_focus(false),
            KeyCode::BackTab => self.cycle_focus(true),
            KeyCode::Char('1') => self.tab = Tab::Timeline,
            KeyCode::Char('2') => self.tab = Tab::Events,
            KeyCode::Char('3') => self.tab = Tab::Memory,
            KeyCode::Char('i') | KeyCode::Char(':') => self.start_input(false),
            KeyCode::Char('n') => self.start_input(true),
            // A secret of the user's own choosing: its window opens empty.
            KeyCode::Char('K') => {
                self.secret_form = Some(SecretForm::default());
            }
            // The newest memory; the arrows then step through the rest.
            KeyCode::Char('m') => {
                if let Some(id) = self.shown_memories().first().map(|m| m.node.id) {
                    self.open(Modal::Memory(id));
                }
            }
            KeyCode::Char('0') => self.select_task(Some(MAIN)),
            KeyCode::Char('b') => self.select_task(None),
            // Ended branches: listed or not.
            KeyCode::Char('e') => {
                self.show_ended = !self.show_ended;
                self.board_sel = 0;
            }
            KeyCode::Char(']') | KeyCode::Right => self.step_task(1),
            KeyCode::Char('[') | KeyCode::Left => self.step_task(-1),
            // End the open task. A capital, so it is not pressed by accident.
            KeyCode::Char('X') => {
                if let Some(task) = self
                    .task_sel
                    .filter(|id| self.task(*id).is_some_and(|t| t.state.is_open()))
                {
                    self.outbox.push(Command::Stop { task });
                }
            }
            KeyCode::Char('?') => self.open(Modal::Help),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::PageUp => self.center_scroll += 10,
            KeyCode::PageDown => self.center_scroll = self.center_scroll.saturating_sub(10),
            KeyCode::End | KeyCode::Char('G') => {
                self.center_scroll = 0;
                self.center_sel = None;
            }
            KeyCode::Char('f') => self.toggle_filter(),
            KeyCode::Char('r') => self.outbox.push(Command::Restart),
            // Show how the most recent knowledge lookup found its passages.
            KeyCode::Char('g') => {
                let latest = self.events.iter().rposition(|e| {
                    matches!(e.kind, EventKind::KnowledgeLookup { .. })
                        && (self.task_sel.is_none() || e.task == self.task_sel)
                });
                if let Some(idx) = latest {
                    self.open(Modal::Lookup(idx));
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => self.outbox.push(Command::Feedback {
                good: true,
                task: self.task_sel,
            }),
            KeyCode::Char('-') => self.outbox.push(Command::Feedback {
                good: false,
                task: self.task_sel,
            }),
            KeyCode::Char('a') => {
                if let Some(id) = self.pending().first().map(|t| t.id) {
                    self.decide(id, true);
                }
            }
            KeyCode::Char('d') => {
                if let Some(id) = self.pending().first().map(|t| t.id) {
                    self.decide(id, false);
                }
            }
            KeyCode::Enter => match self.focus {
                Panel::Agents => {
                    if self.agent_sel < self.agents.len() {
                        self.open(Modal::Agent(self.agent_sel));
                    }
                }
                Panel::Tools => {
                    if let Some(t) = self.tools.get(self.tool_sel) {
                        let id = t.id;
                        self.open(Modal::Tool(id));
                    }
                }
                _ if self.on_board() => {
                    let task = self.board_task();
                    self.select_task(task);
                }
                _ => {
                    if let Some(idx) = self.center_sel.or(self.visible_center.last().copied()) {
                        self.open_center(idx);
                    }
                }
            },
            _ => {}
        }
    }

    pub fn on_mouse(&mut self, m: MouseEvent) {
        // The secret window takes the keyboard and nothing else is live.
        if self.secret_form.is_some() {
            return;
        }
        let (col, row) = (m.column, m.row);
        match m.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                if self.booting() {
                    self.boot = None;
                    return;
                }
                if self.modal.is_some() {
                    match find(&self.hits.buttons, col, row) {
                        Some(Button::Accept) => {
                            if let Some(id) = self.modal_pending_tool() {
                                self.decide(id, true);
                            }
                        }
                        Some(Button::Deny) => {
                            if let Some(id) = self.modal_pending_tool() {
                                self.decide(id, false);
                            }
                        }
                        Some(Button::Close) => self.modal = None,
                        _ => {
                            if let Some(r) = self.hits.modal
                                && !inside(r, col, row)
                            {
                                self.modal = None;
                            }
                        }
                    }
                    return;
                }
                if let Some(button) = find(&self.hits.buttons, col, row) {
                    if let Some(id) = self.pending().first().map(|t| t.id) {
                        match button {
                            Button::Accept => self.decide(id, true),
                            Button::Deny => self.decide(id, false),
                            _ => self.open(Modal::Tool(id)),
                        }
                    }
                    return;
                }
                if let Some(task) = find(&self.hits.task_tabs, col, row) {
                    self.focus = Panel::Center;
                    self.select_task(task);
                    return;
                }
                if let Some(task) = find(&self.hits.board, col, row) {
                    self.focus = Panel::Center;
                    self.select_task(Some(task));
                    return;
                }
                if let Some(tab) = find(&self.hits.tabs, col, row) {
                    self.tab = tab;
                    self.focus = Panel::Center;
                    self.center_sel = None;
                    self.center_scroll = 0;
                    return;
                }
                if let Some(i) = find(&self.hits.agents, col, row) {
                    self.focus = Panel::Agents;
                    if self.agent_sel == i {
                        self.toggle_filter();
                    } else {
                        self.agent_sel = i;
                    }
                    return;
                }
                if let Some(i) = find(&self.hits.tools, col, row) {
                    self.focus = Panel::Tools;
                    self.tool_sel = i;
                    if let Some(t) = self.tools.get(i) {
                        let id = t.id;
                        self.open(Modal::Tool(id));
                    }
                    return;
                }
                if let Some(id) = find(&self.hits.memories, col, row) {
                    self.open(Modal::Memory(id));
                    return;
                }
                if let Some(i) = find(&self.hits.context, col, row) {
                    self.open(Modal::Context(i));
                    return;
                }
                if let Some(idx) = find(&self.hits.center, col, row) {
                    self.focus = Panel::Center;
                    self.open_center(idx);
                    return;
                }
                if let Some(p) = find(&self.hits.panels, col, row) {
                    self.focus = p;
                }
            }
            MouseEventKind::ScrollUp => {
                if self.modal.is_some() {
                    self.modal_scroll = self.modal_scroll.saturating_sub(2);
                } else if find(&self.hits.panels, col, row) == Some(Panel::Tools) {
                    self.tool_sel = self.tool_sel.saturating_sub(1);
                } else {
                    self.center_scroll += 3;
                }
            }
            MouseEventKind::ScrollDown => {
                if self.modal.is_some() {
                    self.modal_scroll += 2;
                } else if find(&self.hits.panels, col, row) == Some(Panel::Tools) {
                    self.tool_sel = (self.tool_sel + 1).min(self.tools.len().saturating_sub(1));
                } else {
                    self.center_scroll = self.center_scroll.saturating_sub(3);
                }
            }
            _ => {}
        }
    }
}
