//! The limits on what is learned, themselves learned from what happens.
//!
//! How many rules may stand in front of every task, how much support a
//! rule needs, how many lessons may be in use, how long one may be, and how
//! close two notes must be to count as the same: none of these has a right
//! value known in advance. Each starts at a default and is moved by
//! evidence: the user's ratings, how runs went, what was lost to a limit
//! and then wanted again, what a model said when asked whether two notes
//! say the same.
//!
//! The split is the harness's usual one. *What the value is* is learned.
//! *How it may move* is compiled and fixed: every setting has a floor and
//! a ceiling it can never leave, moves one step at a time, and only on a
//! stated amount of evidence, which is then spent. No model decides any of
//! it. Every move is recorded with its reason (`history`) and said in the
//! timeline (`take_notes`), so the learning is visible and a bad move can
//! be traced.
//!
//! Kept in `tuning.json` beside the graph.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The settings, their starting values and the bounds they stay within.
pub const RULE_SUPPORT: (u32, u32, u32) = (2, 3, 6);
pub const MAX_RULES: (usize, usize, usize) = (4, 12, 20);
pub const MAX_LESSONS: (usize, usize, usize) = (50, 150, 500);
pub const MAX_LESSON_CHARS: (usize, usize, usize) = (200, 500, 800);
pub const SAME: (f32, f32, f32) = (0.90, 0.95, 0.95);

/// Good outcomes in a row, with standing rules in front of the task and
/// none of them retired, before a rule is trusted on less support.
const EASE_SUPPORT_AFTER: u32 = 12;
/// Good outcomes in a row with every rule slot taken before one is added.
const MORE_RULES_AFTER: u32 = 15;
/// Outcomes needed on each side before tasks with and without standing
/// rules, or long and short lessons, are compared.
const ENOUGH: u32 = 8;
/// How much worse tasks with standing rules must do than tasks without
/// before there are fewer rule slots.
const RULES_HURT_BY: f32 = 0.15;
/// Lessons refused for length before the limit is reconsidered.
const REFUSALS_BEFORE_LONGER: u32 = 3;
/// A lesson over this share of the length limit counts as long.
const LONG_SHARE: f32 = 0.6;
/// Verdicts kept on pairs of near-repeats, and how many must agree, with
/// none against, before a closeness is taken to mean "the same".
const VERDICTS_KEPT: usize = 200;
const VERDICTS_NEEDED: usize = 5;
/// History entries kept.
const HISTORY_KEPT: usize = 60;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Values {
    rule_support: u32,
    max_rules: usize,
    max_lessons: usize,
    max_lesson_chars: usize,
    same: f32,
}

impl Default for Values {
    fn default() -> Self {
        Self {
            rule_support: RULE_SUPPORT.1,
            max_rules: MAX_RULES.1,
            max_lessons: MAX_LESSONS.1,
            max_lesson_chars: MAX_LESSON_CHARS.1,
            same: SAME.1,
        }
    }
}

/// The evidence gathered since each setting last moved.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Evidence {
    /// Good outcomes in a row for tasks that had standing rules.
    #[serde(default)]
    good_streak: u32,
    /// (good, bad) for tasks with standing rules, and for tasks without.
    #[serde(default)]
    with_rules: (u32, u32),
    #[serde(default)]
    without_rules: (u32, u32),
    /// Lessons refused for being too long.
    #[serde(default)]
    refused: u32,
    /// (went well, went badly) for long lessons, and for short ones.
    #[serde(default)]
    long: (u32, u32),
    #[serde(default)]
    short: (u32, u32),
    /// What a model said about pairs of near-repeats: (closeness, same).
    #[serde(default)]
    verdicts: Vec<(f32, bool)>,
    /// How often the success criteria set for a task judged it the way the
    /// user did, and how often they did not.
    #[serde(default)]
    criteria: (u32, u32),
}

/// One move of one setting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Change {
    /// Unix seconds.
    pub at: u64,
    pub setting: String,
    pub from: String,
    pub to: String,
    pub why: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Saved {
    #[serde(default)]
    values: Values,
    #[serde(default)]
    evidence: Evidence,
    #[serde(default)]
    history: Vec<Change>,
}

#[derive(Default)]
pub struct Tuning {
    path: Option<PathBuf>,
    values: Values,
    evidence: Evidence,
    history: Vec<Change>,
    /// Moves not yet said in the timeline.
    notes: Vec<String>,
}

fn rate(counts: (u32, u32)) -> f32 {
    counts.1 as f32 / (counts.0 + counts.1).max(1) as f32
}

impl Tuning {
    /// Reads the saved settings. A missing or unreadable file gives the
    /// defaults; values are pulled back inside their bounds whatever the
    /// file says.
    pub fn load(path: &Path) -> Self {
        let saved: Saved = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        let mut tuning = Self {
            path: Some(path.to_path_buf()),
            values: saved.values,
            evidence: saved.evidence,
            history: saved.history,
            notes: Vec::new(),
        };
        let v = &mut tuning.values;
        v.rule_support = v.rule_support.clamp(RULE_SUPPORT.0, RULE_SUPPORT.2);
        v.max_rules = v.max_rules.clamp(MAX_RULES.0, MAX_RULES.2);
        v.max_lessons = v.max_lessons.clamp(MAX_LESSONS.0, MAX_LESSONS.2);
        v.max_lesson_chars = v
            .max_lesson_chars
            .clamp(MAX_LESSON_CHARS.0, MAX_LESSON_CHARS.2);
        v.same = if v.same.is_finite() {
            v.same.clamp(SAME.0, SAME.2)
        } else {
            SAME.1
        };
        tuning
    }

    fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        let saved = Saved {
            values: self.values.clone(),
            evidence: self.evidence.clone(),
            history: self.history.clone(),
        };
        if let Ok(text) = serde_json::to_string_pretty(&saved) {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }

    pub fn rule_support(&self) -> u32 {
        self.values.rule_support
    }
    pub fn max_rules(&self) -> usize {
        self.values.max_rules
    }
    pub fn max_lessons(&self) -> usize {
        self.values.max_lessons
    }
    pub fn max_lesson_chars(&self) -> usize {
        self.values.max_lesson_chars
    }
    /// Closeness at which two notes are taken to say the same thing.
    pub fn same(&self) -> f32 {
        self.values.same
    }

    /// The moves made since this was last asked, in words, for the timeline.
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    fn moved(&mut self, setting: &str, from: String, to: String, why: String) {
        self.notes.push(format!(
            "Learned from what happened: {setting} is now {to} (was {from}), because {why}."
        ));
        self.history.push(Change {
            at: crate::learn::now(),
            setting: setting.to_string(),
            from,
            to,
            why,
        });
        if self.history.len() > HISTORY_KEPT {
            let extra = self.history.len() - HISTORY_KEPT;
            self.history.drain(..extra);
        }
    }

    /// A task has ended and how it went is known, from the user's rating
    /// or from the run's own checks. `rules` is how many standing rules
    /// were in front of it, `slots_full` whether every rule slot was taken.
    pub fn task_went(&mut self, rules: usize, slots_full: bool, good: bool) {
        let e = &mut self.evidence;
        let side = if rules > 0 {
            &mut e.with_rules
        } else {
            &mut e.without_rules
        };
        if good {
            side.0 += 1;
        } else {
            side.1 += 1;
        }
        if rules > 0 {
            e.good_streak = if good { e.good_streak + 1 } else { 0 };
        }
        // Rules keep going well: one may be trusted on less.
        if self.evidence.good_streak >= EASE_SUPPORT_AFTER
            && self.values.rule_support > RULE_SUPPORT.0
        {
            let from = self.values.rule_support;
            self.values.rule_support -= 1;
            self.evidence.good_streak = 0;
            self.moved(
                "the support a rule needs before it stands in front of every task",
                from.to_string(),
                self.values.rule_support.to_string(),
                format!("{EASE_SUPPORT_AFTER} tasks in a row went well with the standing rules"),
            );
        }
        // Every slot is taken and things keep going well: room for one more.
        if slots_full
            && self.evidence.good_streak >= MORE_RULES_AFTER
            && self.values.max_rules < MAX_RULES.2
        {
            let from = self.values.max_rules;
            self.values.max_rules += 1;
            self.evidence.good_streak = 0;
            self.moved(
                "the number of rules that may stand in front of every task",
                from.to_string(),
                self.values.max_rules.to_string(),
                format!("every slot was taken and {MORE_RULES_AFTER} tasks in a row went well"),
            );
        }
        // Tasks with standing rules do clearly worse than tasks without:
        // fewer slots.
        let (with, without) = (self.evidence.with_rules, self.evidence.without_rules);
        if with.0 + with.1 >= ENOUGH
            && without.0 + without.1 >= ENOUGH
            && rate(with) > rate(without) + RULES_HURT_BY
            && self.values.max_rules > MAX_RULES.0
        {
            let from = self.values.max_rules;
            self.values.max_rules = (from - 2).max(MAX_RULES.0);
            let why = format!(
                "tasks with standing rules went badly {:.0}% of the time against {:.0}% without",
                rate(with) * 100.0,
                rate(without) * 100.0
            );
            self.evidence.with_rules = (0, 0);
            self.evidence.without_rules = (0, 0);
            self.moved(
                "the number of rules that may stand in front of every task",
                from.to_string(),
                self.values.max_rules.to_string(),
                why,
            );
        }
        self.save();
    }

    /// A rule that stood in front of every task has been retired for doing
    /// badly: rules need more behind them before they stand.
    pub fn standing_rule_failed(&mut self) {
        self.evidence.good_streak = 0;
        if self.values.rule_support < RULE_SUPPORT.2 {
            let from = self.values.rule_support;
            self.values.rule_support += 1;
            self.moved(
                "the support a rule needs before it stands in front of every task",
                from.to_string(),
                self.values.rule_support.to_string(),
                "a rule that stood in front of every task turned out badly".into(),
            );
        }
        self.save();
    }

    /// A lesson had to be retired to make room. `useful` says it had been
    /// given to tasks and had more for it than against it: the room is too
    /// small. `unused_share` is the share of lessons in use that no task
    /// has ever been given: when most are dead weight, the room is too big.
    pub fn made_room(&mut self, useful: bool, unused_share: f32) {
        let from = self.values.max_lessons;
        if useful && from < MAX_LESSONS.2 {
            self.values.max_lessons = (from + 20).min(MAX_LESSONS.2);
            self.moved(
                "the number of lessons in use",
                from.to_string(),
                self.values.max_lessons.to_string(),
                "a lesson that had been doing well had to go to make room".into(),
            );
        } else if !useful && unused_share > 0.6 && from > MAX_LESSONS.0 {
            self.values.max_lessons = (from - 10).max(MAX_LESSONS.0);
            self.moved(
                "the number of lessons in use",
                from.to_string(),
                self.values.max_lessons.to_string(),
                format!(
                    "{:.0}% of the lessons in use have never been given to a task",
                    unused_share * 100.0
                ),
            );
        }
        self.save();
    }

    /// Something was learned again that had been retired only to make
    /// room: it was wanted after all, so there is more room from now on.
    pub fn wanted_again(&mut self) {
        let from = self.values.max_lessons;
        if from < MAX_LESSONS.2 {
            self.values.max_lessons = (from + 20).min(MAX_LESSONS.2);
            self.moved(
                "the number of lessons in use",
                from.to_string(),
                self.values.max_lessons.to_string(),
                "something retired to make room was learned again".into(),
            );
        }
        self.save();
    }

    /// The user's response to a task has been compared with the success
    /// criteria set for it: they judged it the same way, or not.
    pub fn criteria_judged(&mut self, agreed: bool) {
        if agreed {
            self.evidence.criteria.0 += 1;
        } else {
            self.evidence.criteria.1 += 1;
        }
        self.save();
    }

    /// A lesson was refused for being too long.
    pub fn refused_for_length(&mut self) {
        self.evidence.refused += 1;
        self.weigh_length();
        self.save();
    }

    /// A lesson of `chars` characters went well or badly (a rating, or a
    /// run's own checks).
    pub fn lesson_went(&mut self, chars: usize, good: bool) {
        let long = chars as f32 > self.values.max_lesson_chars as f32 * LONG_SHARE;
        let side = if long {
            &mut self.evidence.long
        } else {
            &mut self.evidence.short
        };
        if good {
            side.0 += 1;
        } else {
            side.1 += 1;
        }
        self.weigh_length();
        self.save();
    }

    fn weigh_length(&mut self) {
        let (long, short) = (self.evidence.long, self.evidence.short);
        let from = self.values.max_lesson_chars;
        let enough = long.0 + long.1 >= ENOUGH / 2 && short.0 + short.1 >= ENOUGH / 2;
        // Long lessons do clearly worse than short ones: shorter limit.
        if enough && rate(long) > rate(short) + 0.2 && from > MAX_LESSON_CHARS.0 {
            self.values.max_lesson_chars = (from - 50).max(MAX_LESSON_CHARS.0);
            let why = format!(
                "long lessons went badly {:.0}% of the time against {:.0}% for short ones",
                rate(long) * 100.0,
                rate(short) * 100.0
            );
            self.evidence.long = (0, 0);
            self.evidence.short = (0, 0);
            self.evidence.refused = 0;
            self.moved(
                "the longest a lesson may be, in characters",
                from.to_string(),
                self.values.max_lesson_chars.to_string(),
                why,
            );
        // Lessons keep being refused for length, and long ones do no worse
        // than short ones: longer limit.
        } else if self.evidence.refused >= REFUSALS_BEFORE_LONGER
            && long.0 + long.1 >= ENOUGH / 2
            && rate(long) <= rate(short) + 0.1
            && from < MAX_LESSON_CHARS.2
        {
            self.values.max_lesson_chars = (from + 50).min(MAX_LESSON_CHARS.2);
            let why = format!(
                "{} lessons were refused for length, and the long ones kept do no worse than \
                 the short ones",
                self.evidence.refused
            );
            self.evidence.refused = 0;
            self.evidence.long = (0, 0);
            self.evidence.short = (0, 0);
            self.moved(
                "the longest a lesson may be, in characters",
                from.to_string(),
                self.values.max_lesson_chars.to_string(),
                why,
            );
        }
    }

    /// A model was asked whether two notes this close say the same thing,
    /// and answered. The closeness taken to mean "the same" is the lowest
    /// (in steps of 0.01) at or above which every verdict so far was "the
    /// same", with enough of them and at least one at that very step; one
    /// "different" at or above a closeness keeps the bar above it.
    pub fn verdict(&mut self, closeness: f32, same: bool) {
        self.evidence.verdicts.push((closeness, same));
        if self.evidence.verdicts.len() > VERDICTS_KEPT {
            let extra = self.evidence.verdicts.len() - VERDICTS_KEPT;
            self.evidence.verdicts.drain(..extra);
        }
        let mut learned = SAME.2;
        for step in (0..=5).rev() {
            let bar = SAME.0 + step as f32 * 0.01;
            let above: Vec<bool> = self
                .evidence
                .verdicts
                .iter()
                .filter(|(c, _)| *c >= bar)
                .map(|(_, s)| *s)
                .collect();
            if above.iter().any(|s| !s) {
                break;
            }
            // The bar only goes as low as pairs have actually been judged:
            // there must be a verdict at this very step, not just above it.
            let here = self
                .evidence
                .verdicts
                .iter()
                .any(|(c, _)| *c >= bar && *c < bar + 0.01);
            if here && above.len() >= VERDICTS_NEEDED {
                learned = bar;
            }
        }
        let from = self.values.same;
        if (learned - from).abs() > 0.001 {
            self.values.same = learned;
            let why = if learned < from {
                format!(
                    "every pair at least that close that a model was asked about ({} of them) \
                     said the same thing",
                    self.evidence
                        .verdicts
                        .iter()
                        .filter(|(c, _)| *c >= learned)
                        .count()
                )
            } else {
                "a pair that close turned out to say different things".to_string()
            };
            self.moved(
                "how close two notes must be to count as the same",
                format!("{from:.2}"),
                format!("{learned:.2}"),
                why,
            );
        }
        self.save();
    }

    /// The settings, their bounds and how they came to be what they are:
    /// for the `conscious tuning` command.
    pub fn report(&self, offset: i64) -> String {
        let v = &self.values;
        let e = &self.evidence;
        let mut out = String::from(
            "Limits on what is learned. Each starts at a default and is moved by what happens \
             (your ratings, how runs went), one step at a time, inside fixed bounds.\n\n",
        );
        let line = |name: &str, now: String, bounds: String, start: String| {
            format!("{name:<46} {now:>6}   (starts at {start}, stays within {bounds})\n")
        };
        out.push_str(&line(
            "support a rule needs to stand in every task",
            v.rule_support.to_string(),
            format!("{} to {}", RULE_SUPPORT.0, RULE_SUPPORT.2),
            RULE_SUPPORT.1.to_string(),
        ));
        out.push_str(&line(
            "rules that may stand in every task",
            v.max_rules.to_string(),
            format!("{} to {}", MAX_RULES.0, MAX_RULES.2),
            MAX_RULES.1.to_string(),
        ));
        out.push_str(&line(
            "lessons in use at once",
            v.max_lessons.to_string(),
            format!("{} to {}", MAX_LESSONS.0, MAX_LESSONS.2),
            MAX_LESSONS.1.to_string(),
        ));
        out.push_str(&line(
            "longest a lesson may be (characters)",
            v.max_lesson_chars.to_string(),
            format!("{} to {}", MAX_LESSON_CHARS.0, MAX_LESSON_CHARS.2),
            MAX_LESSON_CHARS.1.to_string(),
        ));
        out.push_str(&line(
            "closeness at which two notes are the same",
            format!("{:.2}", v.same),
            format!("{:.2} to {:.2}", SAME.0, SAME.2),
            format!("{:.2}", SAME.1),
        ));
        out.push_str(&format!(
            "\nEvidence since each last moved:\n  tasks with standing rules: {} went well, {} badly ({} well in a row)\n  tasks without: {} went well, {} badly\n  long lessons: {} went well, {} badly · short: {} well, {} badly · {} refused for length\n  near-repeat pairs judged: {}\n  success criteria: judged the result the way you did {} time{}, differently {} time{}\n",
            e.with_rules.0, e.with_rules.1, e.good_streak,
            e.without_rules.0, e.without_rules.1,
            e.long.0, e.long.1, e.short.0, e.short.1, e.refused,
            e.verdicts.len(),
            e.criteria.0, if e.criteria.0 == 1 { "" } else { "s" },
            e.criteria.1, if e.criteria.1 == 1 { "" } else { "s" },
        ));
        if self.history.is_empty() {
            out.push_str("\nNothing has moved yet.");
        } else {
            out.push_str("\nMoves, newest first:\n");
            for change in self.history.iter().rev() {
                out.push_str(&format!(
                    "  {} · {}: {} → {}, because {}\n",
                    crate::schedule::local(change.at, offset),
                    change.setting,
                    change.from,
                    change.to,
                    change.why
                ));
            }
        }
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_are_trusted_on_less_when_they_keep_going_well_and_on_more_when_one_fails() {
        let mut t = Tuning::default();
        assert_eq!(t.rule_support(), 3);
        // Tasks without standing rules teach nothing about rules.
        for _ in 0..40 {
            t.task_went(0, false, true);
        }
        assert_eq!(t.rule_support(), 3);
        // A bad one breaks the run of good ones.
        for _ in 0..11 {
            t.task_went(2, false, true);
        }
        t.task_went(2, false, false);
        for _ in 0..11 {
            t.task_went(2, false, true);
        }
        assert_eq!(t.rule_support(), 3);
        t.task_went(2, false, true);
        assert_eq!(t.rule_support(), 2);
        let notes = t.take_notes();
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("is now 2 (was 3)"), "{}", notes[0]);
        assert!(t.take_notes().is_empty());
        // It never goes below its floor, however well things go.
        for _ in 0..100 {
            t.task_went(2, false, true);
        }
        assert_eq!(t.rule_support(), RULE_SUPPORT.0);
        // A standing rule turning out badly raises it, up to the ceiling.
        for _ in 0..10 {
            t.standing_rule_failed();
        }
        assert_eq!(t.rule_support(), RULE_SUPPORT.2);
    }

    #[test]
    fn rule_slots_grow_when_full_and_going_well_and_shrink_when_rules_hurt() {
        let mut t = Tuning::default();
        // Going well, but the slots are not all taken: no need for more.
        for _ in 0..40 {
            t.task_went(3, false, true);
        }
        assert_eq!(t.max_rules(), 12);
        for _ in 0..15 {
            t.task_went(12, true, true);
        }
        assert_eq!(t.max_rules(), 13);

        // Tasks with rules go badly much more often than tasks without.
        let mut t = Tuning::default();
        for _ in 0..8 {
            t.task_went(0, false, true);
        }
        for n in 0..8 {
            t.task_went(4, false, n % 2 == 0);
        }
        assert_eq!(t.max_rules(), 10);
        assert!(t.take_notes().iter().any(|n| n.contains("went badly 50%")));
        // The evidence was spent: it does not shrink again at once.
        t.task_went(4, false, false);
        assert_eq!(t.max_rules(), 10);
        for _ in 0..200 {
            for _ in 0..8 {
                t.task_went(0, false, true);
            }
            for _ in 0..8 {
                t.task_went(4, false, false);
            }
        }
        assert_eq!(t.max_rules(), MAX_RULES.0);
    }

    #[test]
    fn room_for_lessons_follows_what_is_lost_and_what_is_never_used() {
        let mut t = Tuning::default();
        // A dull lesson made room and most are in use: nothing to learn.
        t.made_room(false, 0.2);
        assert_eq!(t.max_lessons(), 150);
        t.made_room(true, 0.2);
        assert_eq!(t.max_lessons(), 170);
        t.wanted_again();
        assert_eq!(t.max_lessons(), 190);
        // Most of what is kept is never used: less room.
        t.made_room(false, 0.7);
        assert_eq!(t.max_lessons(), 180);
        for _ in 0..100 {
            t.made_room(false, 0.9);
        }
        assert_eq!(t.max_lessons(), MAX_LESSONS.0);
        for _ in 0..100 {
            t.wanted_again();
        }
        assert_eq!(t.max_lessons(), MAX_LESSONS.2);
    }

    #[test]
    fn the_length_limit_follows_how_long_and_short_lessons_fare() {
        // Long ones do much worse: shorter.
        let mut t = Tuning::default();
        for _ in 0..4 {
            t.lesson_went(100, true);
            t.lesson_went(450, false);
        }
        assert_eq!(t.max_lesson_chars(), 450);
        // Refusals alone are not enough; long ones must be doing no worse.
        let mut t = Tuning::default();
        for _ in 0..5 {
            t.refused_for_length();
        }
        assert_eq!(t.max_lesson_chars(), 500);
        for _ in 0..4 {
            t.lesson_went(450, true);
            t.lesson_went(100, true);
        }
        assert_eq!(t.max_lesson_chars(), 550);
        assert!(t.take_notes()[0].contains("refused for length"));
    }

    #[test]
    fn the_same_bar_is_learned_from_what_a_model_says_about_close_pairs() {
        let mut t = Tuning::default();
        assert_eq!(t.same(), 0.95);
        // Four agreeing verdicts are not enough.
        for _ in 0..4 {
            t.verdict(0.935, true);
        }
        assert_eq!(t.same(), 0.95);
        t.verdict(0.932, true);
        assert!((t.same() - 0.93).abs() < 0.001, "{}", t.same());
        // A "different" lower down does not disturb it.
        t.verdict(0.91, false);
        assert!((t.same() - 0.93).abs() < 0.001);
        // A "different" at or above the bar puts it back above that pair.
        t.verdict(0.94, false);
        assert_eq!(t.same(), 0.95);
        assert!(
            t.take_notes()
                .last()
                .unwrap()
                .contains("turned out to say different things")
        );
    }

    #[test]
    fn what_was_learned_survives_a_restart_and_a_bad_file_cannot_break_the_bounds() {
        let dir = std::env::temp_dir().join(format!("cs-tuning-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("tuning.json");
        let mut t = Tuning::load(&path);
        t.standing_rule_failed();
        t.wanted_again();
        let again = Tuning::load(&path);
        assert_eq!((again.rule_support(), again.max_lessons()), (4, 170));
        assert_eq!(again.history.len(), 2);
        let report = again.report(0);
        assert!(
            report.contains("3 → 4, because a rule that stood"),
            "{report}"
        );

        std::fs::write(
            &path,
            r#"{"values":{"rule_support":0,"max_rules":9999,"max_lessons":1,"max_lesson_chars":99999,"same":0.1}}"#,
        )
        .unwrap();
        let t = Tuning::load(&path);
        assert_eq!(
            (
                t.rule_support(),
                t.max_rules(),
                t.max_lessons(),
                t.max_lesson_chars()
            ),
            (
                RULE_SUPPORT.0,
                MAX_RULES.2,
                MAX_LESSONS.0,
                MAX_LESSON_CHARS.2
            )
        );
        assert_eq!(t.same(), SAME.0);
        std::fs::write(&path, "not json").unwrap();
        assert_eq!(Tuning::load(&path).max_rules(), MAX_RULES.1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
