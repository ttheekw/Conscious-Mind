//! Core event types. Every step the harness takes is recorded as an `Event`;
//! the TUI (and later replay/debugging) is rendered purely from this log.

// Some variants are part of the schema but not yet emitted by the demo feed.
#![allow(dead_code)]

use std::time::Duration;

use serde::{Deserialize, Serialize};

pub type AgentId = u32;
pub type ToolCallId = u32;
pub type TaskId = u32;

/// Where a task stands. A task is one thing the user (or a schedule) asked
/// for: a quick answer or hours of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskState {
    /// Waiting for a free slot; others are running.
    Queued,
    Working,
    /// Paused while the knowledge it lacks is gathered.
    Researching,
    Done,
    Failed,
    /// Ended by the user before it finished.
    Stopped,
}

impl TaskState {
    pub fn label(self) -> &'static str {
        match self {
            TaskState::Queued => "QUEUED",
            TaskState::Working => "WORKING",
            TaskState::Researching => "RESEARCH",
            TaskState::Done => "DONE",
            TaskState::Failed => "FAILED",
            TaskState::Stopped => "STOPPED",
        }
    }

    /// Whether the task is still to finish.
    pub fn is_open(self) -> bool {
        matches!(
            self,
            TaskState::Queued | TaskState::Working | TaskState::Researching
        )
    }
}

/// An event together with the task it belongs to. Events about the harness
/// itself (start-up, the scheduler, policy reviews) belong to none.
#[derive(Clone, Debug)]
pub struct Tagged {
    pub task: Option<TaskId>,
    pub kind: EventKind,
}

impl Tagged {
    pub fn system(kind: EventKind) -> Self {
        Self { task: None, kind }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Thinking,
    Acting,
    Waiting,
    Done,
    Failed,
    /// Idle but working on itself: replaying episodes, consolidating memory.
    Reflecting,
}

impl AgentState {
    pub fn label(self) -> &'static str {
        match self {
            AgentState::Idle => "IDLE",
            AgentState::Thinking => "THINK",
            AgentState::Acting => "ACT",
            AgentState::Waiting => "WAIT",
            AgentState::Done => "DONE",
            AgentState::Failed => "FAIL",
            AgentState::Reflecting => "REFLECT",
        }
    }

    pub fn is_active(self) -> bool {
        matches!(self, AgentState::Thinking | AgentState::Acting)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Active,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Success,
    Warn,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Risk {
    Low,
    Medium,
    High,
}

impl Risk {
    pub fn label(self) -> &'static str {
        match self {
            Risk::Low => "LOW",
            Risk::Medium => "MEDIUM",
            Risk::High => "HIGH",
        }
    }
}

pub type MemoryId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemKind {
    /// Something that happened (a run, a failure, a decision).
    Episodic,
    /// A fact about the world or the codebase.
    Semantic,
    /// A learned way of doing things (a lesson, a skill).
    Procedural,
}

impl MemKind {
    pub fn label(self) -> &'static str {
        match self {
            MemKind::Episodic => "episodic",
            MemKind::Semantic => "semantic",
            MemKind::Procedural => "procedural",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryNode {
    pub id: MemoryId,
    pub kind: MemKind,
    pub label: String,
    /// What is remembered, as the store holds it (cut if very long), and a
    /// line on what kind of memory it is, how far it is trusted and where
    /// it came from. Empty when the sender had neither.
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub about: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextSection {
    pub name: String,
    pub tokens: u32,
    /// Why context assembly included this section.
    pub why: String,
}

/// One node on the route a knowledge lookup took to a passage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookupStep {
    /// The node's id in the knowledge graph, so routes that pass through
    /// the same node can be drawn as one.
    pub id: u64,
    /// `passage`, `term`, `document`, `section` or `note`.
    pub kind: String,
    pub title: String,
    /// The link followed to get here; `None` for the node whose words matched.
    pub relation: Option<String>,
}

/// One passage a knowledge lookup handed to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookupPick {
    /// The memory-view node to light up (the passage's document).
    pub memory: MemoryId,
    pub title: String,
    pub source: String,
    pub trust: String,
    pub space: String,
    pub score: f32,
    pub tokens: u32,
    /// Links between the matched node and this passage; 0 is a direct match.
    pub hops: u8,
    /// The question's words that started this route.
    pub matched: Vec<String>,
    pub path: Vec<LookupStep>,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventKind {
    /// A task was created: the main conversation (task 0), a branch that
    /// main opened, or a routine's run.
    TaskOpened {
        title: String,
        /// Who asked: `you` (main), `main` (a branch), `routine #2`.
        origin: String,
    },
    /// A task changed state; `note` says what it is doing, in a line.
    TaskUpdate {
        state: TaskState,
        note: String,
    },
    /// A step of knowledge research: a search made, a page read, what was
    /// kept in the knowledge graph.
    Research {
        /// `search`, `read`, `kept`, `known` or `failed`.
        phase: String,
        text: String,
        /// Where it came from: a web address, or empty.
        source: String,
    },
    /// A branch ended and its result was brought into the main conversation
    /// (carried by main). `ok` is false when it ended with nothing to merge.
    Merged {
        branch: TaskId,
        title: String,
        result: String,
        ok: bool,
    },
    /// The agent needs a secret (an API key, a token) and asks the user to
    /// type it into the secret window. Only its name is ever in an event.
    SecretAsked {
        name: String,
        why: String,
    },
    /// The user answered: the secret was stored, or they declined.
    SecretAnswered {
        name: String,
        stored: bool,
    },
    /// The task's success criteria, checked against the finished work.
    CriteriaChecked {
        checked: Vec<crate::goal::Checked>,
    },
    GoalSet {
        title: String,
        criteria: Vec<String>,
        token_budget: u64,
    },
    PlanUpdated {
        version: u32,
        steps: Vec<String>,
    },
    StepUpdate {
        index: usize,
        status: StepStatus,
    },
    AgentSpawned {
        id: AgentId,
        parent: Option<AgentId>,
        name: String,
        purpose: String,
    },
    AgentStatus {
        id: AgentId,
        state: AgentState,
    },
    /// A visible message. `agent: None` means the user wrote it.
    Message {
        agent: Option<AgentId>,
        text: String,
    },
    /// A short reasoning summary from an agent.
    Thought {
        agent: AgentId,
        text: String,
    },
    ModelCall {
        agent: AgentId,
        model: String,
        tokens_in: u32,
        cached_in: u32,
        tokens_out: u32,
        latency_ms: u64,
    },
    ToolRequested {
        id: ToolCallId,
        agent: AgentId,
        tool: String,
        args: String,
        risk: Risk,
        needs_approval: bool,
        /// The guardrail policy's reasoning for its verdict.
        policy: String,
    },
    ToolDecision {
        id: ToolCallId,
        approved: bool,
        by: String,
    },
    ToolFinished {
        id: ToolCallId,
        ok: bool,
        output: String,
        duration_ms: u64,
    },
    /// Long-term memory index loaded or refreshed.
    MemoryIndexed {
        nodes: Vec<MemoryNode>,
    },
    MemoryRead {
        agent: AgentId,
        query: String,
        hits: Vec<(MemoryId, String)>,
    },
    /// A lookup in the knowledge graph, with enough detail to show how each
    /// passage was found.
    KnowledgeLookup {
        agent: AgentId,
        query: String,
        /// The question reduced to the words searched for.
        words: Vec<String>,
        /// Nodes whose words matched.
        seeds: usize,
        /// Nodes reached in all, counting those only passed through.
        reached: usize,
        /// Passages that qualified but did not fit the limits.
        dropped: usize,
        tokens: u32,
        /// The limits the lookup ran under, in words.
        limits: String,
        picks: Vec<LookupPick>,
    },
    MemoryWrite {
        agent: AgentId,
        node: MemoryNode,
    },
    /// Several memories merged into one (consolidation during reflection).
    MemoryConsolidated {
        agent: AgentId,
        from: Vec<MemoryId>,
        into: MemoryNode,
    },
    /// The agent's self-assessment: how sure it is of its current course
    /// (0..1) and how far the last outcome was from what it predicted (0..1).
    Vitals {
        agent: AgentId,
        confidence: f32,
        surprise: f32,
        /// What it is attending to right now.
        focus: String,
    },
    Context {
        sections: Vec<ContextSection>,
        window: u32,
    },
    Alert {
        severity: Severity,
        text: String,
    },
}

impl EventKind {
    /// Variant name, for the raw event view.
    pub fn name(&self) -> &'static str {
        match self {
            EventKind::TaskOpened { .. } => "TaskOpened",
            EventKind::TaskUpdate { .. } => "TaskUpdate",
            EventKind::Research { .. } => "Research",
            EventKind::Merged { .. } => "Merged",
            EventKind::SecretAsked { .. } => "SecretAsked",
            EventKind::SecretAnswered { .. } => "SecretAnswered",
            EventKind::CriteriaChecked { .. } => "CriteriaChecked",
            EventKind::GoalSet { .. } => "GoalSet",
            EventKind::PlanUpdated { .. } => "PlanUpdated",
            EventKind::StepUpdate { .. } => "StepUpdate",
            EventKind::AgentSpawned { .. } => "AgentSpawned",
            EventKind::AgentStatus { .. } => "AgentStatus",
            EventKind::Message { .. } => "Message",
            EventKind::Thought { .. } => "Thought",
            EventKind::ModelCall { .. } => "ModelCall",
            EventKind::ToolRequested { .. } => "ToolRequested",
            EventKind::ToolDecision { .. } => "ToolDecision",
            EventKind::ToolFinished { .. } => "ToolFinished",
            EventKind::MemoryIndexed { .. } => "MemoryIndexed",
            EventKind::MemoryRead { .. } => "MemoryRead",
            EventKind::KnowledgeLookup { .. } => "KnowledgeLookup",
            EventKind::MemoryWrite { .. } => "MemoryWrite",
            EventKind::MemoryConsolidated { .. } => "MemoryConsolidated",
            EventKind::Vitals { .. } => "Vitals",
            EventKind::Context { .. } => "Context",
            EventKind::Alert { .. } => "Alert",
        }
    }

    /// The agent this event belongs to, if any (used for filtering).
    pub fn agent(&self) -> Option<AgentId> {
        match self {
            EventKind::AgentSpawned { id, .. } | EventKind::AgentStatus { id, .. } => Some(*id),
            EventKind::Message { agent, .. } => *agent,
            EventKind::Thought { agent, .. }
            | EventKind::ModelCall { agent, .. }
            | EventKind::ToolRequested { agent, .. }
            | EventKind::MemoryRead { agent, .. }
            | EventKind::KnowledgeLookup { agent, .. }
            | EventKind::MemoryWrite { agent, .. }
            | EventKind::MemoryConsolidated { agent, .. }
            | EventKind::Vitals { agent, .. } => Some(*agent),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Event {
    pub seq: u64,
    /// Time since the session started. Zero for events from earlier sessions.
    pub at: Duration,
    /// Unix seconds when it happened.
    pub wall: u64,
    /// The task it belongs to, if any.
    pub task: Option<TaskId>,
    /// Whether this event was read back from the timeline file, not produced
    /// in this session.
    pub restored: bool,
    pub kind: EventKind,
}
