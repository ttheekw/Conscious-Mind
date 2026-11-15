//! How the agent's own lessons earn their place, and how memory is kept
//! free of repeats.
//!
//! A lesson is a note the agent wrote for itself after a task. Stored
//! carelessly, such notes do harm: an agent tends to repeat what a similar
//! past record did, so a wrong lesson is replayed into later tasks. The
//! rules here are fixed and take no model:
//!
//! - **A lesson starts unconfirmed** (`Trust::Inferred`), and counts for
//!   less in a lookup until it is confirmed.
//! - **A rating settles it.** When the user rates a task, the rating counts
//!   for every lesson that task was given, and for the lessons it wrote. A
//!   good rating confirms an unconfirmed lesson (`Trust::Experience`); a bad
//!   one retires it. A confirmed lesson is retired once it has at least
//!   `RETIRE_AFTER_BAD` bad ratings and more bad than good.
//! - **A run that did not finish** (stopped, or ended by a limit) can only
//!   have its lessons confirmed by a later task that used them: its own
//!   good rating does not vouch for something learned on the way to not
//!   finishing.
//! - **A run's own checks count too**, without waiting for the user. When
//!   a run ended by itself with its last test, build or lint command
//!   passing and nothing changed after it (`agent::Outcome::Verified`),
//!   that confirms the lessons it used and the lessons it wrote. A run
//!   that was cut short or whose last check failed counts against the
//!   lessons it used; an unconfirmed lesson is retired after
//!   `RETIRE_AFTER_FAILED` such runs if nothing has ever gone its way. The
//!   user's word outweighs this: a lesson no rating of theirs has backed
//!   is retired by a single bad rating, however many checks passed.
//! - **Nothing is stored twice.** A lesson that says what one already held
//!   says is not added.
//! - **Retired is not deleted.** A retired note stays in the store with the
//!   reason, and is never found by a lookup again.
//!
//! **Two kinds, and a limit on each** (after TextReg, which names the
//! failure this guards against: learned text that keeps growing and fills
//! with narrow rules that fit one case; its cost is length times
//! narrowness). What the agent learns about how to work is one of:
//!
//! - a **rule**: how to behave whatever the task, with no knowledge of any
//!   one project. A rule that has earned it (`RULE_SUPPORT`) is put in
//!   front of the model on every task, so there may only be a few
//!   (`MAX_RULES`, `RULES_BUDGET`). Until then it is found by lookup like
//!   any lesson. Something that names a file, a value or one project is
//!   not a rule, whatever the model calls it (`looks_specific`): it is
//!   kept as know-how.
//! - **know-how**: something that matters again for one kind of task. It
//!   is only ever found by lookup, when a task is about that, and says
//!   when it applies.
//!
//! What matters to one task only is not kept at all; reflection is told so
//! and the limits below enforce it. Each entry is short
//! (`MAX_LESSON_CHARS`: a longer one is refused, not cut), and there are at
//! most `MAX_LESSONS` in use: beyond that the least useful is retired.
//! Support for a rule is counted, not judged: the separate tasks that drew
//! the same thing, and the runs and ratings that went well with it.
//!
//! "Says the same" is decided by `same`: equal once case and punctuation
//! are set aside, or at least `SAME` close in meaning. With the model in
//! use, rewordings of one statement score about 0.93 to 0.97, but two
//! opposite statements on one subject ("prefers tabs" / "prefers spaces")
//! reach 0.925, so the bar is set above that and some rewordings are kept
//! as separate notes rather than risk merging a statement with its
//! opposite.

use super::graph::{Node, NodeId, NodeKind, Trust};
use super::{KnowledgeBase, MEMORY_SPACE, MemoryKind};

// How long a lesson may be, how many may be in use, how much support a
// rule needs, how many rules may stand, and how close "the same" is are
// not fixed here: they are learned, within bounds (`tuning.rs`).
/// The most the standing rules may take together, in estimated tokens.
pub const RULES_BUDGET: usize = 600;
/// Bad ratings after which a confirmed lesson is retired, if it has more
/// bad than good.
pub const RETIRE_AFTER_BAD: u32 = 2;
/// Failed runs after which an unconfirmed lesson that nothing has gone
/// well for is retired.
pub const RETIRE_AFTER_FAILED: u32 = 2;
/// Closeness from which two notes may say the same thing, but could as
/// well say opposite things: a model is asked which (`near_repeats`).
pub const NEAR: f32 = 0.90;
/// Closeness at which memories are about the same subject. Measured with
/// the model in use: related memories 0.83 to 0.87, the same subject from
/// another side about 0.71, unrelated ones 0.52 to 0.63.
pub const RELATED: f32 = 0.70;
/// Memories on one subject before they are worth a summary, and the most
/// one summary is made from.
pub const MIN_GROUP: usize = 4;
pub const MAX_GROUP: usize = 8;
/// The link from a summary to each memory it was made from.
const SUMMARISES: &str = "summarises";
/// Notes compared with each other in one pass over memory, newest first.
const MAX_COMPARED: usize = 1_500;

/// What became of a lesson offered for keeping.
#[derive(Debug, PartialEq)]
pub enum Kept {
    New(NodeId),
    /// One saying the same is already held: this is it.
    Known(NodeId),
    /// Not kept, and why.
    Refused(String),
}

/// Something the agent learned about how to work, as offered for keeping.
#[derive(Clone, Copy, Debug)]
pub enum Learned<'a> {
    /// How to behave whatever the task.
    Rule(&'a str),
    /// What matters again for one kind of task, and when it applies.
    KnowHow { text: &'a str, when: &'a str },
    /// What the user wants from one kind of task, learned when the success
    /// criteria set for a task turned out not to judge it the way the user
    /// does. Found by lookup, like know-how, and used to write criteria.
    Intent { text: &'a str, when: &'a str },
}

/// Who may see something the harness has learned. There are two kinds of
/// memory (the user set this out, 2026-11-15, after their name turned up
/// where it should not have):
///
/// 1. **About the user and what they want**: who they are, how they like
///    things done, what they meant by a request, what happened in their
///    projects. This is theirs. It is `Private`: it stays on their machine
///    and is in nothing that is exported or published.
/// 2. **How the system itself should work to solve a problem**, whatever
///    the request and whoever asks. This is not about anyone, and may be
///    shared so that other copies of the harness start off knowing it. It
///    is `Shareable`.
///
/// Everything is private unless it is plainly of the second kind, so a
/// mistake errs towards keeping something back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Privacy {
    Private,
    Shareable,
}

impl Privacy {
    pub fn label(self) -> &'static str {
        match self {
            Privacy::Private => "private: about you or your work, stays on this machine",
            Privacy::Shareable => "shareable: how the system works in general, about nobody",
        }
    }
}

/// Whether a statement speaks of a person: the user, or anyone addressed
/// or named as one. Such a thing is about someone, so it is never shared.
pub fn speaks_of_someone(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .any(|word| {
            matches!(
                word,
                "user"
                    | "user's"
                    | "users"
                    | "you"
                    | "your"
                    | "yours"
                    | "i"
                    | "my"
                    | "me"
                    | "he"
                    | "she"
                    | "his"
                    | "her"
                    | "him"
                    | "their"
                    | "they"
                    | "them"
                    | "name"
                    | "named"
                    | "owner"
                    | "client"
                    | "customer"
                    | "boss"
                    | "team"
            )
        })
}

/// Whether a statement is about one project, file or value rather than
/// about how to work in general: it quotes a name, gives a path or a
/// number, or speaks of "this project". Such a thing cannot be a rule for
/// every task.
pub fn looks_specific(text: &str) -> bool {
    let lower = text.to_lowercase();
    let named = text.contains('`') || text.contains('/') || text.contains('\\');
    let numbered = text.chars().any(|c| c.is_ascii_digit());
    let file = lower
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .any(|word| {
            let word = word.trim_matches(|c: char| !c.is_alphanumeric());
            word.rsplit_once('.').is_some_and(|(name, ext)| {
                !name.is_empty()
                    && (1..=5).contains(&ext.len())
                    && ext.chars().all(char::is_alphanumeric)
            })
        });
    let here = [
        "this project",
        "this repo",
        "this codebase",
        "this machine",
        "in this folder",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase));
    named || numbered || file || here
}

/// What a rating did to the lessons it reached.
#[derive(Debug, Default, PartialEq)]
pub struct Judged {
    /// Lessons this rating confirmed, by their first line.
    pub confirmed: Vec<String>,
    /// Lessons this rating retired.
    pub retired: Vec<String>,
}

/// Notes merged into one that says the same.
#[derive(Debug, PartialEq)]
pub struct Merged {
    pub from: Vec<NodeId>,
    pub into: NodeId,
    pub title: String,
}

fn plain(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn author(task: u32, finished: bool) -> String {
    if finished {
        format!("agent, from task #{task}")
    } else {
        format!("agent, from task #{task}, which did not finish")
    }
}

impl KnowledgeBase {
    /// Whether a node is a lesson the agent wrote for itself after a task.
    pub fn is_lesson(node: &Node) -> bool {
        node.kind == NodeKind::Note
            && node.space == MEMORY_SPACE
            && node.provenance.source.starts_with("agent, from")
    }

    fn live_lessons(&self) -> Vec<NodeId> {
        self.graph
            .nodes()
            .filter(|n| Self::is_lesson(n) && n.retired.is_none())
            .map(|n| n.id)
            .collect()
    }

    /// Whether two notes say the same thing. `b_vector` is the second
    /// note's vector, when it has one.
    fn same(&self, a: &Node, b_text: &str, b_vector: Option<&[f32]>) -> bool {
        if plain(&a.text) == plain(b_text) {
            return true;
        }
        match (self.vectors.get(a.id), b_vector) {
            (Some(va), Some(vb)) if va.len() == vb.len() => {
                let dot: f32 = va.iter().zip(vb).map(|(x, y)| x * y).sum();
                let length = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
                length > 0.0 && dot / length >= self.tuning.same()
            }
            _ => false,
        }
    }

    /// Who may see this memory. Only one thing is shareable: a rule the
    /// agent drew for itself about how to work whatever the task, that has
    /// been confirmed, is still in use, and names no project, file, value
    /// or person. Who the user is, what they asked for, what they meant
    /// (`Intent`), know-how for their kind of work, and every remembered
    /// fact and event are private.
    pub fn privacy(node: &Node) -> Privacy {
        let general = Self::is_rule(node)
            && node.retired.is_none()
            && node.provenance.trust != Trust::Inferred
            && !looks_specific(&node.text)
            && !speaks_of_someone(&node.text);
        if general {
            Privacy::Shareable
        } else {
            Privacy::Private
        }
    }

    /// The rules that may be shared, as plain statements, best supported
    /// first. Nothing private is ever in this.
    pub fn shareable_rules(&self) -> Vec<String> {
        let mut rules: Vec<&Node> = self
            .graph
            .nodes()
            .filter(|n| Self::privacy(n) == Privacy::Shareable)
            .collect();
        rules.sort_by_key(|n| std::cmp::Reverse((Self::support(n), n.id)));
        rules
            .into_iter()
            .map(|n| {
                n.text
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .trim_start_matches("Rule: ")
                    .to_string()
            })
            .collect()
    }

    /// Takes rules shared by another copy of the harness, one a line (a
    /// leading `- ` is allowed). Each comes in as a rule of the agent's own
    /// that nothing has confirmed yet, so it has to earn its place here
    /// like any other; one that names a project, a value or a person is
    /// not a general rule and is refused. Returns (taken, already known,
    /// refused).
    pub fn import_rules(&mut self, text: &str) -> (usize, usize, usize) {
        let (mut taken, mut known, mut refused) = (0, 0, 0);
        for line in text.lines() {
            let rule = line.trim().trim_start_matches("- ").trim();
            if rule.is_empty() || rule.starts_with('#') {
                continue;
            }
            if looks_specific(rule) || speaks_of_someone(rule) {
                refused += 1;
                continue;
            }
            match self.add_learned(0, Learned::Rule(rule), true) {
                Kept::New(_) => taken += 1,
                Kept::Known(_) => known += 1,
                Kept::Refused(_) => refused += 1,
            }
        }
        (taken, known, refused)
    }

    /// Whether a lesson is a rule (how to behave whatever the task) rather
    /// than know-how for one kind of task.
    pub fn is_rule(node: &Node) -> bool {
        Self::is_lesson(node) && node.text.starts_with("Rule: ")
    }

    /// What stands behind a lesson: the separate tasks that drew it, and
    /// the runs and ratings that went well with it.
    fn support(node: &Node) -> u32 {
        node.drawn_by.len() as u32 + node.record.good + node.record.passed
    }

    /// The rules that have earned a place in front of every task: confirmed,
    /// with `RULE_SUPPORT` behind them, the best supported first, within
    /// `MAX_RULES` and `RULES_BUDGET`.
    fn standing_rules(&self) -> Vec<&Node> {
        let mut rules: Vec<&Node> = self
            .graph
            .nodes()
            .filter(|n| Self::is_rule(n) && n.retired.is_none())
            .filter(|n| n.provenance.trust >= Trust::Experience)
            .filter(|n| Self::support(n) >= self.tuning.rule_support())
            .collect();
        rules.sort_by_key(|n| std::cmp::Reverse((Self::support(n), n.provenance.at, n.id)));
        let mut used = 0;
        rules
            .into_iter()
            .take(self.tuning.max_rules())
            .take_while(|n| {
                used += n.text.chars().count().div_ceil(4);
                used <= RULES_BUDGET
            })
            .collect()
    }

    /// The nodes of the rules that are in front of every task.
    pub fn standing_rule_ids(&self) -> Vec<NodeId> {
        self.standing_rules().iter().map(|n| n.id).collect()
    }

    /// The standing rules as a block for a model's context, or empty.
    pub fn rules(&self) -> String {
        let rules = self.standing_rules();
        if rules.is_empty() {
            return String::new();
        }
        let mut s = String::from(
            "How you have learned to work. These are rules you drew yourself from earlier \
             tasks and that have held up since. They apply to any task; what the user asks \
             and the harness's rules come first.\n",
        );
        for n in rules {
            s.push_str(&format!("- {}\n", n.text.trim_start_matches("Rule: ")));
        }
        s
    }

    /// Keeps know-how learned in `task`: see `add_learned`.
    #[cfg(test)]
    pub fn add_lesson(&mut self, task: u32, lesson: &str, finished: bool) -> Kept {
        self.add_learned(
            task,
            Learned::KnowHow {
                text: lesson,
                when: "",
            },
            finished,
        )
    }

    /// Keeps something learned in `task`, unless it is too long, or one
    /// saying the same is already held (which then counts `task` as one
    /// more that drew it). What is called a rule but speaks of one project,
    /// file or value is kept as know-how. It starts unconfirmed. `finished`
    /// says whether the run it came from reached its end. If this takes the
    /// lessons in use past `MAX_LESSONS`, the least useful is retired. Call
    /// `commit` afterwards.
    pub fn add_learned(&mut self, task: u32, learned: Learned, finished: bool) -> Kept {
        let (said, when, rule) = match learned {
            Learned::Rule(text) => (text.trim(), "", !looks_specific(text)),
            Learned::KnowHow { text, when } | Learned::Intent { text, when } => {
                (text.trim(), when.trim(), false)
            }
        };
        let length = said.chars().count() + when.chars().count();
        if said.is_empty() {
            return Kept::Refused("it was empty".into());
        }
        let longest = self.tuning.max_lesson_chars();
        if length > longest {
            // Counted: if this keeps happening and long lessons do no worse
            // than short ones, the limit is raised.
            self.tuning.refused_for_length();
            return Kept::Refused(format!(
                "it runs to {length} characters, and a lesson is at most {longest}"
            ));
        }
        let mut text = if rule {
            format!("Rule: {said}")
        } else if matches!(learned, Learned::Intent { .. }) {
            format!("Intent: {said}")
        } else {
            format!("Lesson: {said}")
        };
        if !when.is_empty() {
            text.push_str(&format!("\nApplies when: {when}"));
        }
        // The new lesson as a vector, to compare by meaning, made the way
        // stored notes are (their first line stands as their title).
        let title: String = text
            .lines()
            .next()
            .unwrap_or_default()
            .chars()
            .take(60)
            .collect();
        let vector = self.embedder.as_ref().and_then(|model| {
            model
                .passages(&[format!("{title}\n{text}")])
                .ok()?
                .into_iter()
                .next()
        });
        for id in self.live_lessons() {
            let same = self
                .graph
                .node(id)
                .is_some_and(|held| self.same(held, &text, vector.as_deref()));
            if same {
                // Drawn again, by another task: that is support for it.
                if let Some(held) = self.graph.node_mut(id)
                    && !held.drawn_by.contains(&task)
                    && held.drawn_by.len() < 16
                {
                    held.drawn_by.push(task);
                }
                return Kept::Known(id);
            }
        }
        // Learned again after being retired only to make room: it was
        // wanted after all. It comes back, and there is more room from now.
        let made_way: Vec<NodeId> = self
            .graph
            .nodes()
            .filter(|n| {
                Self::is_lesson(n)
                    && n.retired
                        .as_deref()
                        .is_some_and(|why| why.starts_with("made room"))
            })
            .map(|n| n.id)
            .collect();
        for id in made_way {
            let same = self
                .graph
                .node(id)
                .is_some_and(|held| self.same(held, &text, vector.as_deref()));
            if same {
                if let Some(held) = self.graph.node_mut(id) {
                    held.retired = None;
                    if !held.drawn_by.contains(&task) {
                        held.drawn_by.push(task);
                    }
                }
                self.tuning.wanted_again();
                self.keep_within_limit(id);
                return Kept::New(id);
            }
        }
        let id = self.add_note(MEMORY_SPACE, &author(task, finished), &text);
        if let Some(node) = self.graph.node_mut(id) {
            node.provenance.trust = Trust::Inferred;
            node.drawn_by.push(task);
        }
        self.keep_within_limit(id);
        Kept::New(id)
    }

    /// Retires the least useful lessons until no more are in use than there
    /// is room for (a learned limit). The one just added (`fresh`) is never the one to go.
    /// Least useful: the worst record, then the least used, then the oldest.
    fn keep_within_limit(&mut self, fresh: NodeId) {
        let mut live: Vec<(i64, u32, u64, NodeId)> = self
            .graph
            .nodes()
            .filter(|n| Self::is_lesson(n) && n.retired.is_none() && n.id != fresh)
            .map(|n| {
                let merit = i64::from(Self::support(n)) - i64::from(n.record.bad + n.record.failed)
                    + i64::from(n.provenance.trust >= Trust::Experience);
                (merit, n.record.given, n.provenance.at, n.id)
            })
            .collect();
        live.sort();
        let room = self.tuning.max_lessons();
        let over = (live.len() + 1).saturating_sub(room);
        let unused = live.iter().filter(|(_, given, _, _)| *given == 0).count();
        let unused_share = unused as f32 / live.len().max(1) as f32;
        for (merit, given, _, id) in live.into_iter().take(over) {
            if let Some(node) = self.graph.node_mut(id) {
                node.retired = Some(format!(
                    "made room: more than {room} lessons were in use and this was the least \
                     useful"
                ));
            }
            // What had to go says whether the room is the right size.
            self.tuning.made_room(given > 0 && merit > 1, unused_share);
        }
    }

    /// Records that these nodes were put in front of a task, and returns
    /// the lessons among them: the ones that task's rating will count for.
    pub fn given(&mut self, nodes: &[NodeId]) -> Vec<NodeId> {
        let mut lessons = Vec::new();
        for id in nodes {
            if let Some(node) = self.graph.node_mut(*id)
                && Self::is_lesson(node)
                && node.retired.is_none()
                && !lessons.contains(id)
            {
                node.record.given += 1;
                lessons.push(*id);
            }
        }
        lessons
    }

    /// Counts the user's rating of `task` for the lessons it was `given`
    /// and the lessons it wrote, confirming and retiring by the rules at
    /// the top of this file. Saves.
    pub fn rate_lessons(&mut self, task: u32, given: &[NodeId], good: bool) -> Judged {
        let wrote = format!("agent, from task #{task}");
        let wrote_unfinished = format!("{wrote}, which did not finish");
        let mut judged = Judged::default();
        let mut touched = false;
        let standing = self.standing_rule_ids();
        let mut went = Vec::new();
        let mut standing_failed = 0;
        for id in self.live_lessons() {
            let Some(node) = self.graph.node_mut(id) else {
                continue;
            };
            let own = node.provenance.source == wrote || node.provenance.source == wrote_unfinished;
            let used = given.contains(&id);
            if !own && !used {
                continue;
            }
            touched = true;
            went.push((node.text.chars().count(), good));
            let line = node.text.lines().next().unwrap_or_default().to_string();
            if good {
                node.record.good += 1;
                // A run that did not finish cannot vouch for its own lesson.
                let vouches = used || node.provenance.source == wrote;
                if vouches && node.provenance.trust == Trust::Inferred {
                    node.provenance.trust = Trust::Experience;
                    judged.confirmed.push(line);
                }
            } else {
                node.record.bad += 1;
                let unconfirmed = node.provenance.trust == Trust::Inferred;
                // Confirmed only by passing checks: the user's word, now
                // that there is one, outweighs them.
                let never_backed = node.record.good == 0;
                let against =
                    node.record.bad >= RETIRE_AFTER_BAD && node.record.bad > node.record.good;
                if unconfirmed || never_backed || against {
                    node.retired = Some(format!(
                        "rated bad {} time{} and good {}",
                        node.record.bad,
                        if node.record.bad == 1 { "" } else { "s" },
                        node.record.good
                    ));
                    judged.retired.push(line);
                    standing_failed += usize::from(standing.contains(&id));
                }
            }
        }
        self.learn_from(went, standing_failed);
        if touched {
            let _ = self.commit();
        }
        judged
    }

    /// Passes on to the learned limits how lessons of each length fared and
    /// whether a rule that stood in front of every task had to be retired.
    fn learn_from(&mut self, went: Vec<(usize, bool)>, standing_failed: usize) {
        for (chars, good) in went {
            self.tuning.lesson_went(chars, good);
        }
        for _ in 0..standing_failed {
            self.tuning.standing_rule_failed();
        }
    }

    /// A task has ended and how it went is known (the user's rating, or the
    /// run's own checks). `rules` is how many standing rules were in front
    /// of it. The limits on rules learn from this.
    pub fn task_went(&mut self, rules: usize, good: bool) {
        let full = rules >= self.tuning.max_rules();
        self.tuning.task_went(rules, full, good);
    }

    /// Records whether the success criteria set for a task judged it the
    /// way the user did.
    pub fn criteria_judged(&mut self, agreed: bool) {
        self.tuning.criteria_judged(agreed);
    }

    /// What the learned limits have just changed, in words, for the
    /// timeline.
    pub fn take_tuning_notes(&mut self) -> Vec<String> {
        self.tuning.take_notes()
    }

    /// The learned limits, their bounds, the evidence and the moves so far.
    pub fn tuning_report(&self, offset: i64) -> String {
        self.tuning.report(offset)
    }

    /// Counts what a run's own checks showed (`passed`, or not) for the
    /// lessons `task` was `given` and the lessons it wrote, by the rules at
    /// the top of this file. Saves.
    pub fn judge_outcome(&mut self, task: u32, given: &[NodeId], passed: bool) -> Judged {
        let wrote = format!("agent, from task #{task}");
        let mut judged = Judged::default();
        let mut touched = false;
        let standing = self.standing_rule_ids();
        let mut went = Vec::new();
        let mut standing_failed = 0;
        for id in self.live_lessons() {
            let Some(node) = self.graph.node_mut(id) else {
                continue;
            };
            // A lesson from a run that did not finish is left to later runs.
            let own = node.provenance.source == wrote;
            if !own && !given.contains(&id) {
                continue;
            }
            touched = true;
            went.push((node.text.chars().count(), passed));
            let line = node.text.lines().next().unwrap_or_default().to_string();
            if passed {
                node.record.passed += 1;
                if node.provenance.trust == Trust::Inferred {
                    node.provenance.trust = Trust::Experience;
                    judged.confirmed.push(line);
                }
            } else {
                node.record.failed += 1;
                let nothing_for_it = node.record.passed == 0 && node.record.good == 0;
                if node.provenance.trust == Trust::Inferred
                    && nothing_for_it
                    && node.record.failed >= RETIRE_AFTER_FAILED
                {
                    node.retired = Some(format!(
                        "the {} runs it was used in did not succeed",
                        node.record.failed
                    ));
                    judged.retired.push(line);
                    standing_failed += usize::from(standing.contains(&id));
                }
            }
        }
        self.learn_from(went, standing_failed);
        if touched {
            let _ = self.commit();
        }
        judged
    }

    fn closeness(&self, a: NodeId, b: NodeId) -> Option<f32> {
        let (va, vb) = (self.vectors.get(a)?, self.vectors.get(b)?);
        (va.len() == vb.len()).then(|| va.iter().zip(vb).map(|(x, y)| x * y).sum())
    }

    /// Pairs of notes that are close enough to say the same thing but not
    /// close enough to be sure (between `NEAR` and `SAME`), which a model
    /// should be asked about: up to `limit` of them, as (the one to keep,
    /// the other, their texts). Pairs already judged different are left
    /// out. The one to keep is chosen as in `merge_repeats`.
    pub fn near_repeats(&self, limit: usize) -> Vec<(NodeId, NodeId, String, String)> {
        let mut notes: Vec<&Node> = self
            .graph
            .nodes()
            .filter(|n| n.kind == NodeKind::Note && n.space == MEMORY_SPACE && n.retired.is_none())
            .filter(|n| self.vectors.get(n.id).is_some())
            .collect();
        let merit = |n: &Node| {
            if Self::is_lesson(n) {
                i64::from(n.record.good + n.record.passed)
                    - i64::from(n.record.bad + n.record.failed)
                    + i64::from(n.provenance.trust == Trust::Experience)
            } else {
                0
            }
        };
        notes.sort_by_key(|n| std::cmp::Reverse((merit(n), n.provenance.at, n.id)));
        notes.truncate(MAX_COMPARED);
        let mut pairs = Vec::new();
        let mut taken: Vec<NodeId> = Vec::new();
        for (i, keep) in notes.iter().enumerate() {
            if pairs.len() >= limit {
                break;
            }
            if taken.contains(&keep.id) {
                continue;
            }
            for other in &notes[i + 1..] {
                let alike = Self::is_lesson(keep) == Self::is_lesson(other)
                    && Self::kind_of(keep) == Self::kind_of(other);
                let judged = keep.apart.contains(&other.id) || other.apart.contains(&keep.id);
                if !alike || judged || taken.contains(&other.id) {
                    continue;
                }
                if self
                    .closeness(keep.id, other.id)
                    .is_some_and(|c| (NEAR..self.tuning.same()).contains(&c))
                {
                    pairs.push((keep.id, other.id, keep.text.clone(), other.text.clone()));
                    taken.extend([keep.id, other.id]);
                    break;
                }
            }
        }
        pairs
    }

    /// Settles a pair from `near_repeats`: if they were judged to say the
    /// same, `other` is merged into `keep`; if not, that is remembered so
    /// the pair is not asked about again. Saves.
    pub fn settle_pair(&mut self, keep: NodeId, other: NodeId, same: bool) -> Option<Merged> {
        // What was said about a pair this close teaches where "the same"
        // begins.
        if let Some(closeness) = self.closeness(keep, other) {
            self.tuning.verdict(closeness, same);
        }
        if !same {
            if let Some(node) = self.graph.node_mut(keep)
                && !node.apart.contains(&other)
            {
                node.apart.push(other);
            }
            let _ = self.commit();
            return None;
        }
        let record = self.graph.node(other)?.record;
        self.graph.node_mut(other)?.retired =
            Some(format!("says the same as #{keep}; merged into it"));
        let kept = self.graph.node_mut(keep)?;
        kept.record.given += record.given;
        kept.record.good += record.good;
        kept.record.bad += record.bad;
        kept.record.passed += record.passed;
        kept.record.failed += record.failed;
        let title = kept.title.clone();
        let _ = self.commit();
        Some(Merged {
            from: vec![other],
            into: keep,
            title,
        })
    }

    /// Whether a memory already has a summary made from it (or an attempt
    /// at one that came to nothing).
    fn summarised(&self, id: NodeId) -> bool {
        self.graph.neighbours(id).any(|(_, edge, _)| {
            matches!(&edge.relation, super::graph::Relation::Related(label) if label == SUMMARISES)
        })
    }

    /// Groups of memories on one subject that have no summary yet: up to
    /// `limit` groups of `MIN_GROUP` to `MAX_GROUP` remembered facts and
    /// events, each as (node, text), oldest first. Standing memories and
    /// the agent's own notes are left out: the first are always in front
    /// of the model, the second are not conversation.
    pub fn related_groups(&self, limit: usize) -> Vec<Vec<(NodeId, String)>> {
        let mut notes: Vec<&Node> = self
            .graph
            .nodes()
            .filter(|n| {
                n.kind == NodeKind::Note
                    && n.space == MEMORY_SPACE
                    && n.retired.is_none()
                    && n.provenance.source == "conversation"
                    && matches!(Self::kind_of(n), Some(MemoryKind::Fact | MemoryKind::Event))
            })
            .filter(|n| self.vectors.get(n.id).is_some() && !self.summarised(n.id))
            .collect();
        notes.sort_by_key(|n| (n.provenance.at, n.id));
        notes.truncate(MAX_COMPARED);
        let mut grouped: Vec<NodeId> = Vec::new();
        let mut groups = Vec::new();
        for seed in &notes {
            if groups.len() >= limit {
                break;
            }
            if grouped.contains(&seed.id) {
                continue;
            }
            let members: Vec<&&Node> = notes
                .iter()
                .filter(|n| !grouped.contains(&n.id))
                .filter(|n| {
                    n.id == seed.id || self.closeness(seed.id, n.id).is_some_and(|c| c >= RELATED)
                })
                .take(MAX_GROUP)
                .collect();
            if members.len() >= MIN_GROUP {
                grouped.extend(members.iter().map(|n| n.id));
                groups.push(members.iter().map(|n| (n.id, n.text.clone())).collect());
            }
        }
        groups
    }

    /// Records what came of summarising a group from `related_groups`. With
    /// a summary, it becomes a memory of its own, unconfirmed, linked to
    /// the memories it was made from, which are kept as they are. Without
    /// one, an empty marker is kept (retired, so never found) so the same
    /// group is not tried again. Saves. Returns the summary's node.
    pub fn add_summary(
        &mut self,
        members: &[NodeId],
        summary: Option<(&str, &str)>,
    ) -> Option<NodeId> {
        let text = match summary {
            Some((_, said)) => format!(
                "{}\n\nKind: fact\nA summary made while idle of {} earlier memories, which are \
                 kept and say more.",
                said.trim(),
                members.len()
            ),
            None => "No one summary could be made of these memories.".to_string(),
        };
        let id = self.add_note(MEMORY_SPACE, "agent, while idle", &text);
        if let Some(node) = self.graph.node_mut(id) {
            node.provenance.trust = Trust::Inferred;
            match summary {
                Some((title, _)) => node.title = title.trim().chars().take(60).collect(),
                None => {
                    node.retired = Some("not one subject; kept so it is not tried again".into())
                }
            }
        }
        for member in members {
            self.graph.add_edge(
                id,
                *member,
                super::graph::Relation::Related(SUMMARISES.to_string()),
                Trust::Inferred,
            );
        }
        let _ = self.commit();
        summary.map(|_| id)
    }

    /// Merges notes in memory that say the same thing, so a lookup is not
    /// spent on repeats. Of each set the keeper is, for lessons, the one
    /// with the best record, and otherwise the newest (a later statement
    /// stands for an earlier one). The others are retired with a pointer to
    /// it; a lesson's ratings are added to the keeper's. Only notes of the
    /// same kind are compared. Saves if anything changed.
    pub fn merge_repeats(&mut self) -> Vec<Merged> {
        struct Entry {
            id: NodeId,
            lesson: bool,
            kind: Option<MemoryKind>,
            /// Best first.
            rank: (i64, u64, NodeId),
        }
        let mut notes: Vec<Entry> = self
            .graph
            .nodes()
            .filter(|n| n.kind == NodeKind::Note && n.space == MEMORY_SPACE && n.retired.is_none())
            .map(|n| {
                let lesson = Self::is_lesson(n);
                let merit = if lesson {
                    i64::from(n.record.good + n.record.passed)
                        - i64::from(n.record.bad + n.record.failed)
                        + i64::from(n.provenance.trust == Trust::Experience)
                } else {
                    0
                };
                Entry {
                    id: n.id,
                    lesson,
                    kind: Self::kind_of(n),
                    rank: (merit, n.provenance.at, n.id),
                }
            })
            .collect();
        notes.sort_by_key(|e| std::cmp::Reverse(e.rank));
        notes.truncate(MAX_COMPARED);

        let mut gone = vec![false; notes.len()];
        let mut merged = Vec::new();
        for i in 0..notes.len() {
            if gone[i] {
                continue;
            }
            let mut from = Vec::new();
            for j in i + 1..notes.len() {
                if gone[j] || notes[j].lesson != notes[i].lesson || notes[j].kind != notes[i].kind {
                    continue;
                }
                let (Some(keeper), Some(other)) =
                    (self.graph.node(notes[i].id), self.graph.node(notes[j].id))
                else {
                    continue;
                };
                if self.same(keeper, &other.text, self.vectors.get(other.id)) {
                    gone[j] = true;
                    from.push(notes[j].id);
                }
            }
            if from.is_empty() {
                continue;
            }
            let into = notes[i].id;
            let mut sum = super::graph::Record::default();
            for id in &from {
                if let Some(node) = self.graph.node_mut(*id) {
                    sum.given += node.record.given;
                    sum.good += node.record.good;
                    sum.bad += node.record.bad;
                    sum.passed += node.record.passed;
                    sum.failed += node.record.failed;
                    node.retired = Some(format!("says the same as #{into}; merged into it"));
                }
            }
            let mut title = String::new();
            if let Some(keeper) = self.graph.node_mut(into) {
                keeper.record.given += sum.given;
                keeper.record.good += sum.good;
                keeper.record.bad += sum.bad;
                keeper.record.passed += sum.passed;
                keeper.record.failed += sum.failed;
                title = keeper.title.clone();
            }
            merged.push(Merged { from, into, title });
        }
        if !merged.is_empty() {
            let _ = self.commit();
        }
        merged
    }

    /// Every lesson, in use first and newest first, one per line: for the
    /// `conscious lessons` command.
    pub fn lessons_report(&self) -> String {
        let mut lessons: Vec<&Node> = self.graph.nodes().filter(|n| Self::is_lesson(n)).collect();
        if lessons.is_empty() {
            return "No lessons have been written yet. One is written after a task that took \
                    real work, when the model finds something worth keeping."
                .into();
        }
        lessons.sort_by_key(|n| {
            (
                n.retired.is_some(),
                std::cmp::Reverse((n.provenance.at, n.id)),
            )
        });
        let mut out = String::new();
        let standing = self.standing_rule_ids();
        for n in lessons {
            let kind = if n.text.starts_with("Intent: ") {
                "what the user wants"
            } else if !Self::is_rule(n) {
                "know-how"
            } else if standing.contains(&n.id) {
                "rule, in front of every task"
            } else {
                "rule, found by lookup until it has more support"
            };
            let state = match (&n.retired, n.provenance.trust) {
                (Some(why), _) => format!("retired ({why})"),
                (None, Trust::Inferred) => "not yet confirmed".to_string(),
                (None, _) => "confirmed".to_string(),
            };
            out.push_str(&format!(
                "#{} {kind}, {state} · {} · drawn by {} task{} · given to {} task{} · rated good {}, bad {} · runs passed {}, failed {} · {}\n    {}\n",
                n.id,
                match Self::privacy(n) {
                    Privacy::Private => "private",
                    Privacy::Shareable => "shareable",
                },
                n.drawn_by.len(),
                if n.drawn_by.len() == 1 { "" } else { "s" },
                n.record.given,
                if n.record.given == 1 { "" } else { "s" },
                n.record.good,
                n.record.bad,
                n.record.passed,
                n.record.failed,
                n.provenance.source.trim_start_matches("agent, "),
                n.text.lines().next().unwrap_or_default()
            ));
        }
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::super::embed::testing::Ideas;
    use super::super::{Query, Trust, tuning};
    use super::*;

    const MAX_RULES: usize = tuning::MAX_RULES.1;
    const MAX_LESSONS: usize = tuning::MAX_LESSONS.1;
    const SAME: f32 = tuning::SAME.1;

    fn found(kb: &KnowledgeBase, question: &str) -> Vec<NodeId> {
        kb.retrieve(&Query::new(question))
            .hits
            .iter()
            .map(|h| h.node)
            .collect()
    }

    fn id(kept: Kept) -> NodeId {
        match kept {
            Kept::New(id) | Kept::Known(id) => id,
            Kept::Refused(why) => panic!("refused: {why}"),
        }
    }

    #[test]
    fn only_general_confirmed_rules_are_shareable_and_everything_about_the_user_is_private() {
        let mut kb = KnowledgeBase::in_memory();
        // The first kind: about the user and what they want.
        let name = kb.remember("I am Ada", "Nice to meet you.", None);
        let intent = match kb.add_learned(
            1,
            Learned::Intent {
                text: "wants the answer in one sentence",
                when: "asked a factual question",
            },
            true,
        ) {
            Kept::New(id) => id,
            other => panic!("{other:?}"),
        };
        let know_how = match kb.add_lesson(2, "the seed script needs the staging flag", true) {
            Kept::New(id) => id,
            other => panic!("{other:?}"),
        };
        // The second kind: how to work, whatever the task.
        let new_rule = |kb: &mut KnowledgeBase, task: u32, text: &str| match kb.add_learned(
            task,
            Learned::Rule(text),
            true,
        ) {
            Kept::New(id) => id,
            other => panic!("{text}: {other:?}"),
        };
        let general = new_rule(&mut kb, 3, "read the failing test before changing the code");
        let unproven = new_rule(&mut kb, 4, "run the whole suite after every change");
        let about_user = new_rule(
            &mut kb,
            5,
            "keep replies short because the user dislikes long ones",
        );
        // Rated good: confirmed.
        for task in [1, 2, 3, 5] {
            kb.rate_lessons(task, &[], true);
        }

        let privacy = |kb: &KnowledgeBase, id| KnowledgeBase::privacy(kb.node(id).unwrap());
        assert_eq!(privacy(&kb, general), Privacy::Shareable);
        for (what, id) in [
            ("who the user is", name),
            ("what the user meant", intent),
            ("know-how from their work", know_how),
            ("a rule nothing has confirmed", unproven),
            ("a rule that speaks of the user", about_user),
        ] {
            assert_eq!(privacy(&kb, id), Privacy::Private, "{what}");
        }
        // Only the one general, confirmed rule is ever handed out.
        assert_eq!(
            kb.shareable_rules(),
            ["read the failing test before changing the code"]
        );
        let exported = kb.shareable_rules().join("\n");
        for private in ["Ada", "one sentence", "staging", "user"] {
            assert!(!exported.contains(private), "{private} was exported");
        }
        // The report says which each is.
        let report = kb.lessons_report();
        assert_eq!(report.matches(" · shareable · ").count(), 1, "{report}");

        // A retired rule is no longer shared.
        kb.rate_lessons(3, &[], false);
        kb.rate_lessons(3, &[], false);
        assert!(kb.shareable_rules().is_empty());

        // Rules shared by another copy come in unconfirmed, and anything
        // about a project or a person is refused at the door.
        let mut other = KnowledgeBase::in_memory();
        let (taken, known, refused) = other.import_rules(
            "# shared rules\n\
             - read the failing test before changing the code\n\
             - Read the failing test before changing the code.\n\
             - always tell Ada's boss when the build fails\n\
             - the config lives in conf.d/90-local.ini\n",
        );
        assert_eq!((taken, known, refused), (1, 1, 2));
        // Unconfirmed here, so not passed on again until it proves itself.
        assert!(other.shareable_rules().is_empty());
    }

    #[test]
    fn a_lesson_starts_unconfirmed_and_a_rating_confirms_or_retires_it() {
        let mut kb = KnowledgeBase::in_memory();
        let cause = id(kb.add_lesson(
            4,
            "the failing test was caused by the local override file",
            true,
        ));
        let wrong = id(kb.add_lesson(4, "the deploy script must be run twice", true));
        kb.commit().unwrap();
        for lesson in [cause, wrong] {
            assert_eq!(kb.node(lesson).unwrap().provenance.trust, Trust::Inferred);
        }
        // It is found, and shown to the model as what it is.
        let lookup = kb.retrieve(&Query::new("why is the test failing, local override"));
        assert_eq!(lookup.hits[0].node, cause);
        assert!(
            kb.render(&lookup)
                .contains("a lesson of your own, not yet confirmed")
        );

        // Task 9 is given the first lesson, and is rated good: confirmed.
        let given = kb.given(&[cause, 9_999]);
        assert_eq!(given, [cause]);
        let judged = kb.rate_lessons(9, &given, true);
        assert_eq!(judged.confirmed.len(), 1);
        assert!(judged.retired.is_empty());
        let node = kb.node(cause).unwrap();
        assert_eq!(node.provenance.trust, Trust::Experience);
        assert_eq!(
            (node.record.given, node.record.good, node.record.bad),
            (1, 1, 0)
        );
        let lookup = kb.retrieve(&Query::new("why is the test failing, local override"));
        assert!(
            kb.render(&lookup)
                .contains("a lesson of your own, confirmed")
        );

        // Task 4, which wrote both, is rated bad. The unconfirmed one goes
        // at once; the confirmed one survives a single bad rating.
        let judged = kb.rate_lessons(4, &[], false);
        assert_eq!(
            judged.retired,
            ["Lesson: the deploy script must be run twice"]
        );
        assert!(kb.node(wrong).unwrap().retired.is_some());
        assert!(kb.node(cause).unwrap().retired.is_none());
        // Retired is kept but never found again, and counts for nothing.
        assert!(found(&kb, "deploy script run twice").is_empty());
        assert!(kb.given(&[wrong]).is_empty());

        // A second bad rating, with one good: more bad than good, retired.
        let judged = kb.rate_lessons(12, &[cause], false);
        assert_eq!(judged.retired.len(), 1);
        assert!(found(&kb, "why is the test failing, local override").is_empty());
        let report = kb.lessons_report();
        assert!(
            report.contains("retired (rated bad 2 times and good 1)"),
            "{report}"
        );
    }

    #[test]
    fn a_run_that_did_not_finish_cannot_vouch_for_its_own_lesson() {
        let mut kb = KnowledgeBase::in_memory();
        let lesson = id(kb.add_lesson(7, "the cache folder must exist before the build", false));
        kb.commit().unwrap();
        assert!(
            kb.node(lesson)
                .unwrap()
                .provenance
                .source
                .ends_with("did not finish")
        );
        // Its own task rated good: still unconfirmed.
        assert!(kb.rate_lessons(7, &[], true).confirmed.is_empty());
        assert_eq!(kb.node(lesson).unwrap().provenance.trust, Trust::Inferred);
        // A later task that used it, rated good: confirmed.
        let given = kb.given(&[lesson]);
        assert_eq!(kb.rate_lessons(8, &given, true).confirmed.len(), 1);
        // Its own task rated bad would have retired it while unconfirmed.
        let mut kb = KnowledgeBase::in_memory();
        let lesson = id(kb.add_lesson(7, "the cache folder must exist before the build", false));
        assert_eq!(kb.rate_lessons(7, &[], false).retired.len(), 1);
        assert!(kb.node(lesson).unwrap().retired.is_some());
    }

    #[test]
    fn a_lesson_already_held_is_not_stored_twice() {
        let mut kb = KnowledgeBase::in_memory();
        let first = kb.add_lesson(1, "The cat door sticks in wet weather.", true);
        assert!(matches!(first, Kept::New(_)));
        kb.commit().unwrap();
        // The same words, differently dressed.
        assert_eq!(
            kb.add_lesson(2, "  the cat door sticks, in wet weather", true),
            Kept::Known(id(first))
        );
        // With the model, the same meaning in other words.
        kb.set_embedder(std::sync::Arc::new(Ideas));
        kb.embed_some(10).unwrap();
        assert!(matches!(
            kb.add_lesson(3, "feline, wet storm", true),
            Kept::Known(_)
        ));
        // Something else is new.
        assert!(matches!(
            kb.add_lesson(3, "wages are paid monthly", true),
            Kept::New(_)
        ));
        // A retired lesson does not stand in the way of learning it afresh.
        kb.rate_lessons(1, &[], false);
        assert!(matches!(
            kb.add_lesson(5, "The cat door sticks in wet weather.", true),
            Kept::New(_)
        ));
    }

    #[test]
    fn unconfirmed_knowledge_counts_for_less_than_confirmed() {
        let mut kb = KnowledgeBase::in_memory();
        let a = id(kb.add_lesson(
            1,
            "restart the worker after changing the queue settings",
            true,
        ));
        let b = id(kb.add_lesson(
            2,
            "restart the worker after changing the queue settings file",
            true,
        ));
        kb.commit().unwrap();
        let given = kb.given(&[b]);
        kb.rate_lessons(3, &given, true);
        let lookup = kb.retrieve(&Query::new("restart worker queue settings"));
        let score = |node: NodeId| lookup.hits.iter().find(|h| h.node == node).unwrap().score;
        assert!(score(b) > score(a), "{} vs {}", score(b), score(a));
    }

    #[test]
    fn repeats_in_memory_are_merged_into_the_newest_and_lessons_into_the_best() {
        let mut kb = KnowledgeBase::in_memory();
        let digest = |summary: &str, kind: &str| {
            format!("{summary}\n\nKind: {kind}\nUser said: x\n\nAnswer given: y")
        };
        let old = kb.add_note(
            MEMORY_SPACE,
            "conversation",
            &digest("The boat is in the harbour.", "fact"),
        );
        let new = kb.add_note(
            MEMORY_SPACE,
            "conversation",
            &digest("The boat is in the harbour.", "fact"),
        );
        // The same words under another kind are not the same memory.
        let event = kb.add_note(
            MEMORY_SPACE,
            "conversation",
            &digest("The boat is in the harbour.", "event"),
        );
        let other = kb.add_note(
            MEMORY_SPACE,
            "conversation",
            &digest("Wages are paid on Fridays.", "fact"),
        );
        // Two lessons that say the same; the older one has the better record.
        let proven = id(kb.add_lesson(1, "the storm drain floods in heavy rain", true));
        let given = kb.given(&[proven]);
        kb.rate_lessons(2, &given, true);
        kb.commit().unwrap();
        kb.set_embedder(std::sync::Arc::new(Ideas));
        kb.embed_some(20).unwrap();
        // Stored directly, as if written before repeats were checked for.
        let repeat = kb.add_note(
            MEMORY_SPACE,
            "agent, from task #3",
            "Lesson: umbrella weather, wet",
        );
        kb.commit().unwrap();
        kb.embed_some(20).unwrap();

        let merged = kb.merge_repeats();
        assert_eq!(merged.len(), 2, "{merged:?}");
        let lesson = merged
            .iter()
            .find(|m| m.into == proven)
            .expect("lessons merged");
        assert_eq!(lesson.from, [repeat]);
        let memory = merged.iter().find(|m| m.into != proven).unwrap();
        // For plain memories the newest stands; ids rise with time here.
        assert_eq!(
            (memory.into, memory.from.clone()),
            (new.max(old), vec![new.min(old)])
        );
        for kept in [event, other, proven] {
            assert!(kb.node(kept).unwrap().retired.is_none());
        }
        assert!(
            kb.node(repeat)
                .unwrap()
                .retired
                .as_deref()
                .unwrap()
                .contains("merged into")
        );
        // Merged notes are gone from lookups and from the memory view.
        assert!(
            !kb.memories()
                .iter()
                .any(|(n, _, _)| *n == repeat || *n == new.min(old))
        );
        // A second pass finds nothing left to do.
        assert!(kb.merge_repeats().is_empty());
    }

    #[test]
    fn a_runs_own_checks_confirm_and_retire_lessons_but_the_users_word_outweighs_them() {
        let mut kb = KnowledgeBase::in_memory();
        let used = id(kb.add_lesson(1, "the migration must run before the seed script", true));
        let own = id(kb.add_lesson(5, "the seed script needs the staging flag", true));
        let shaky = id(kb.add_lesson(2, "skipping the cache step is safe", true));
        kb.commit().unwrap();

        // Task 5 used the first lesson and wrote the second; its tests passed.
        let given = kb.given(&[used]);
        let judged = kb.judge_outcome(5, &given, true);
        assert_eq!(judged.confirmed.len(), 2);
        for lesson in [used, own] {
            let node = kb.node(lesson).unwrap();
            assert_eq!(node.provenance.trust, Trust::Experience);
            assert_eq!((node.record.passed, node.record.good), (1, 0));
        }
        // A run that failed counts against what it used: once is not enough
        // to retire, twice with nothing for it is.
        assert!(kb.judge_outcome(6, &[shaky], false).retired.is_empty());
        assert_eq!(kb.judge_outcome(7, &[shaky], false).retired.len(), 1);
        assert_eq!(
            kb.node(shaky).unwrap().retired.as_deref(),
            Some("the 2 runs it was used in did not succeed")
        );
        // A lesson that has passed before is not retired by failures alone.
        kb.judge_outcome(8, &[used], false);
        kb.judge_outcome(9, &[used], false);
        assert!(kb.node(used).unwrap().retired.is_none());

        // Checks passed, but the user rates the task that wrote it bad: no
        // rating of theirs ever backed it, so it goes.
        let judged = kb.rate_lessons(5, &[], false);
        assert_eq!(
            judged.retired,
            ["Lesson: the seed script needs the staging flag"]
        );
        let report = kb.lessons_report();
        assert!(report.contains("runs passed 1, failed 2"), "{report}");
    }

    #[test]
    fn near_repeats_are_put_to_a_model_and_its_answer_is_remembered() {
        let mut kb = KnowledgeBase::in_memory();
        kb.set_embedder(std::sync::Arc::new(Ideas));
        let note = |kb: &mut KnowledgeBase, text: &str| {
            kb.add_note(
                MEMORY_SPACE,
                "conversation",
                &format!("{text}\n\nKind: fact"),
            )
        };
        // The stand-in model places these two near each other, not on top.
        let a = note(&mut kb, "cat cat cat cat cat cat cat boat boat boat");
        let b = note(
            &mut kb,
            "cat cat cat cat cat cat cat boat boat boat boat boat boat boat",
        );
        note(&mut kb, "money wages salary");
        kb.commit().unwrap();
        kb.embed_some(10).unwrap();
        let close = kb.closeness(a, b).unwrap();
        assert!((NEAR..SAME).contains(&close), "{close}");
        // Not merged on closeness alone.
        assert!(kb.merge_repeats().is_empty());

        let pairs = kb.near_repeats(5);
        assert_eq!(pairs.len(), 1);
        let (keep, other) = (pairs[0].0, pairs[0].1);
        assert_eq!((keep, other), (b, a), "the newer one is the one to keep");
        // Judged different: remembered, and not asked again.
        assert!(kb.settle_pair(keep, other, false).is_none());
        assert!(kb.near_repeats(5).is_empty());
        assert!(kb.node(a).unwrap().retired.is_none());

        // Judged the same: merged into the one to keep.
        note(
            &mut kb,
            "money money money money money money money rain rain rain",
        );
        note(
            &mut kb,
            "money money money money money money money rain rain rain rain rain rain rain",
        );
        kb.commit().unwrap();
        kb.embed_some(10).unwrap();
        let pairs = kb.near_repeats(5);
        assert_eq!(pairs.len(), 1, "{pairs:?}");
        let merged = kb.settle_pair(pairs[0].0, pairs[0].1, true).unwrap();
        assert_eq!(merged.into, pairs[0].0);
        assert!(kb.node(pairs[0].1).unwrap().retired.is_some());
    }

    #[test]
    fn memories_on_one_subject_get_one_summary_and_are_kept() {
        let mut kb = KnowledgeBase::in_memory();
        kb.set_embedder(std::sync::Arc::new(Ideas));
        let mut trip = Vec::new();
        for (n, said) in ["boat", "ship harbour", "vessel sails", "sail boat harbour"]
            .iter()
            .enumerate()
        {
            trip.push(kb.add_note(
                MEMORY_SPACE,
                "conversation",
                &format!("Trip note {n}: {said}.\n\nKind: event\nUser said: {said}"),
            ));
        }
        // Too few on this subject, a standing memory, and an agent's note.
        kb.add_note(
            MEMORY_SPACE,
            "conversation",
            "Wages are paid monthly.\n\nKind: fact",
        );
        kb.add_note(
            MEMORY_SPACE,
            "conversation",
            "The user sails a boat.\n\nKind: identity",
        );
        kb.add_note(MEMORY_SPACE, "agent", "boat ship vessel");
        kb.commit().unwrap();
        kb.embed_some(20).unwrap();

        let groups = kb.related_groups(3);
        assert_eq!(groups.len(), 1);
        let members: Vec<NodeId> = groups[0].iter().map(|(id, _)| *id).collect();
        assert_eq!(members, trip);

        let summary = kb
            .add_summary(
                &members,
                Some(("The boat trip", "The boat sails from the harbour.")),
            )
            .unwrap();
        let node = kb.node(summary).unwrap();
        assert_eq!(node.title, "The boat trip");
        assert_eq!(node.provenance.trust, Trust::Inferred);
        assert!(
            node.text
                .contains("summary made while idle of 4 earlier memories")
        );
        // The memories it was made from are kept.
        assert!(
            trip.iter()
                .all(|id| kb.node(*id).unwrap().retired.is_none())
        );
        // The same group is not summarised twice.
        assert!(kb.related_groups(3).is_empty());

        // A group the model found to be no one subject is not tried again.
        let mut kb2 = KnowledgeBase::in_memory();
        kb2.set_embedder(std::sync::Arc::new(Ideas));
        let ids: Vec<NodeId> = (0..4)
            .map(|n| {
                kb2.add_note(
                    MEMORY_SPACE,
                    "conversation",
                    &format!("Storm {n}: rain and wet weather.\n\nKind: fact"),
                )
            })
            .collect();
        kb2.commit().unwrap();
        kb2.embed_some(10).unwrap();
        assert_eq!(kb2.related_groups(3).len(), 1);
        assert!(kb2.add_summary(&ids, None).is_none());
        assert!(kb2.related_groups(3).is_empty());
        assert_eq!(
            kb2.memories().len(),
            4,
            "the marker is not shown as a memory"
        );
    }

    #[test]
    fn a_rule_is_general_and_know_how_is_for_one_kind_of_task() {
        for specific in [
            "The config is read from `settings.ini`.",
            "Check conf.d/90-local.ini first.",
            "Retry the upload 3 times.",
            "In this project the tests need the seed data.",
            "Look in settings.ini before the code.",
        ] {
            assert!(looks_specific(specific), "{specific}");
        }
        for general in [
            "Run the check that failed again after a fix, before saying it is fixed.",
            "Read the error message in full before changing anything.",
            "When two sources disagree, say so rather than picking one silently.",
        ] {
            assert!(!looks_specific(general), "{general}");
        }

        let mut kb = KnowledgeBase::in_memory();
        let rule = id(kb.add_learned(
            1,
            Learned::Rule("Run the failed check again after a fix before saying it is fixed."),
            true,
        ));
        // Called a rule, but it is about one file: kept as know-how.
        let not_a_rule = id(kb.add_learned(
            1,
            Learned::Rule("Always check `conf.d/90-local.ini` before the code."),
            true,
        ));
        let how = id(kb.add_learned(
            2,
            Learned::KnowHow {
                text: "The fixtures live in `data/seed`.",
                when: "writing or fixing tests here",
            },
            true,
        ));
        assert!(KnowledgeBase::is_rule(kb.node(rule).unwrap()));
        assert!(!KnowledgeBase::is_rule(kb.node(not_a_rule).unwrap()));
        assert!(
            kb.node(how)
                .unwrap()
                .text
                .ends_with("Applies when: writing or fixing tests here")
        );
        // Too long, or empty, is refused and not cut.
        let long = "word ".repeat(120);
        assert!(matches!(
            kb.add_learned(3, Learned::KnowHow { text: &long, when: "" }, true),
            Kept::Refused(why) if why.contains("at most 500")
        ));
        assert!(matches!(
            kb.add_learned(3, Learned::Rule("  "), true),
            Kept::Refused(_)
        ));
    }

    #[test]
    fn a_rule_is_put_in_front_of_every_task_only_once_it_has_earned_it() {
        let mut kb = KnowledgeBase::in_memory();
        let text = "Run the failed check again after a fix before saying it is fixed.";
        let rule = id(kb.add_learned(1, Learned::Rule(text), true));
        kb.commit().unwrap();
        // Unconfirmed and drawn once: found by lookup, not standing.
        assert_eq!(kb.rules(), "");
        assert!(found(&kb, "failed check after a fix").contains(&rule));

        // Confirmed by a passing run: still only two things behind it.
        kb.judge_outcome(1, &[], true);
        assert_eq!(kb.rules(), "");
        // A second task draws the same thing: that is the third.
        assert_eq!(
            kb.add_learned(6, Learned::Rule(text), true),
            Kept::Known(rule)
        );
        // The same task drawing it again adds nothing.
        assert_eq!(
            kb.add_learned(6, Learned::Rule(text), true),
            Kept::Known(rule)
        );
        assert_eq!(kb.node(rule).unwrap().drawn_by, [1, 6]);
        let block = kb.rules();
        assert!(
            block.starts_with("How you have learned to work."),
            "{block}"
        );
        assert!(block.contains("- Run the failed check again after a fix"));
        // Standing, so no longer handed out by lookup as well.
        assert!(kb.recall("failed check after a fix", &[]).is_none());
        assert_eq!(kb.standing_rule_ids(), [rule]);
        assert!(kb.lessons_report().contains("rule, in front of every task"));

        // Know-how never stands, however well it does.
        let how = id(kb.add_lesson(2, "the fixtures live in the seed folder", true));
        for task in 10..16 {
            kb.judge_outcome(task, &[how], true);
        }
        assert_eq!(kb.standing_rule_ids(), [rule]);

        // A standing rule that turns out badly is retired like any lesson.
        kb.rate_lessons(20, &[rule], false);
        assert_eq!(kb.rules(), "");

        // There is room for only so many, the best supported first.
        let mut kb = KnowledgeBase::in_memory();
        for n in 0..MAX_RULES + 5 {
            let words = [
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
                "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec",
            ];
            let r = id(kb.add_learned(
                n as u32,
                Learned::Rule(&format!("Always {} first.", words[n])),
                true,
            ));
            for task in 0..(3 + n as u32) {
                kb.judge_outcome(100 + task, &[r], true);
            }
        }
        assert_eq!(kb.standing_rule_ids().len(), MAX_RULES);
        // The best supported are the ones kept in front.
        assert!(kb.rules().contains("quebec") && !kb.rules().contains("alpha"));
    }

    #[test]
    fn lessons_in_use_are_capped_and_the_least_useful_makes_room() {
        let mut kb = KnowledgeBase::in_memory();
        let mut ids = Vec::new();
        for n in 0..MAX_LESSONS {
            ids.push(id(kb.add_lesson(
                n as u32,
                &format!("finding number {n} about widget {n}x"),
                true,
            )));
        }
        // The first has done well; the second has done badly.
        kb.judge_outcome(500, &[ids[0]], true);
        kb.judge_outcome(501, &[ids[1]], false);
        let live = |kb: &KnowledgeBase| kb.live_lessons().len();
        assert_eq!(live(&kb), MAX_LESSONS);

        let newest = id(kb.add_lesson(900, "one more finding about the gadget", true));
        assert_eq!(live(&kb), MAX_LESSONS);
        // The one with the worst record made room; the good one and the new
        // one are still there.
        let why = kb.node(ids[1]).unwrap().retired.clone().unwrap();
        assert!(why.starts_with("made room"), "{why}");
        assert!(kb.node(ids[0]).unwrap().retired.is_none());
        assert!(kb.node(newest).unwrap().retired.is_none());
    }
}
