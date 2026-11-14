//! Skills: the agent's procedural memory, written by the agent itself.
//!
//! After a task that took real work, the model that did it is asked whether
//! the way it was solved would help next time. If so it writes a skill: a
//! short name, when to use it, and the steps. Later tasks are shown the list
//! of skills (name and when-to-use only, to keep context small) and load the
//! one that fits. A skill is refined by saving it again under the same name.
//!
//! The harness curates them from evidence: each skill counts how often it
//! was loaded and how the tasks that used it were rated. One that keeps
//! being used in badly rated tasks is retired; if there are too many, the
//! least useful are retired first. Nothing here is executable: a skill is
//! text the model reads, and every action it then takes still goes through
//! the guardrails.
//!
//! Stored as one Markdown file per skill, so they can be read and edited by
//! hand.

use std::path::{Path, PathBuf};

use crate::learn::now;

/// Skills listed to the model at the start of a task.
const MAX_LISTED: usize = 20;
/// Skills kept active; beyond this the least useful are retired.
const MAX_ACTIVE: usize = 40;
/// Bad ratings (outnumbering good ones) at which a skill is retired.
const RETIRE_AFTER_BAD: u32 = 2;
const MAX_NAME_CHARS: usize = 48;
const MAX_WHEN_CHARS: usize = 200;
const MAX_STEPS_CHARS: usize = 4_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    /// Short kebab-case name; also the file name.
    pub name: String,
    /// One line: the kind of task this is for.
    pub when: String,
    /// The procedure, as Markdown.
    pub steps: String,
    /// Times it was loaded for a task.
    pub uses: u32,
    /// Ratings of the tasks it was loaded for.
    pub good: u32,
    pub bad: u32,
    pub updated: u64,
    /// Retired skills are kept on disk but no longer offered.
    pub retired: bool,
}

impl Skill {
    /// How much a skill has earned its place: rated use counts most.
    fn worth(&self) -> i64 {
        self.good as i64 * 4 - self.bad as i64 * 4 + self.uses as i64
    }

    fn to_file(&self) -> String {
        format!(
            "---\nname: {}\nwhen: {}\nuses: {}\ngood: {}\nbad: {}\nupdated: {}\nretired: {}\n---\n{}\n",
            self.name,
            self.when,
            self.uses,
            self.good,
            self.bad,
            self.updated,
            self.retired,
            self.steps.trim()
        )
    }

    fn from_file(text: &str) -> Option<Self> {
        let rest = text.strip_prefix("---\n")?;
        let (header, steps) = rest.split_once("\n---\n")?;
        let field = |key: &str| {
            header
                .lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix(": "))
        };
        let number = |key: &str| field(key).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
        Some(Skill {
            name: slug(field("name")?),
            when: field("when")?.trim().to_string(),
            steps: steps.trim().to_string(),
            uses: number("uses"),
            good: number("good"),
            bad: number("bad"),
            updated: field("updated")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0),
            retired: field("retired").is_some_and(|v| v.trim() == "true"),
        })
    }
}

/// A name reduced to lowercase words joined by dashes.
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-')
        .chars()
        .take(MAX_NAME_CHARS)
        .collect()
}

#[derive(Default)]
pub struct Skills {
    dir: Option<PathBuf>,
    items: Vec<Skill>,
}

impl Skills {
    /// Skills that are never written to disk.
    #[cfg(test)]
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Where skills live: beside the routing knowledge.
    pub fn default_dir() -> Option<PathBuf> {
        Some(crate::learn::default_path()?.parent()?.join("skills"))
    }

    /// Reads every skill file in `dir`. Files that are not skills are ignored.
    pub fn load(dir: &Path) -> Self {
        let mut items: Vec<Skill> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .filter_map(|text| Skill::from_file(&text))
            .collect();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            dir: Some(dir.to_path_buf()),
            items,
        }
    }

    fn write(&self, skill: &Skill) {
        let Some(dir) = &self.dir else {
            return;
        };
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{}.md", skill.name)), skill.to_file());
    }

    /// Active skills, most useful first.
    pub fn active(&self) -> Vec<&Skill> {
        let mut list: Vec<&Skill> = self.items.iter().filter(|s| !s.retired).collect();
        list.sort_by(|a, b| b.worth().cmp(&a.worth()).then(a.name.cmp(&b.name)));
        list
    }

    /// The list shown to the model: names and when to use them, nothing more.
    pub fn index(&self) -> String {
        let active = self.active();
        if active.is_empty() {
            return String::new();
        }
        let mut s = String::from(
            "Skills you wrote after earlier tasks. If one fits this task, load it with the \
             `skill` tool before you start and follow it; if it turns out wrong or incomplete, \
             save a corrected version under the same name.\n",
        );
        for skill in active.into_iter().take(MAX_LISTED) {
            s.push_str(&format!("- {}: {}\n", skill.name, skill.when));
        }
        s
    }

    /// The full text of a skill, counting the use.
    pub fn open(&mut self, name: &str) -> Option<String> {
        let wanted = slug(name);
        let skill = self
            .items
            .iter_mut()
            .find(|s| s.name == wanted && !s.retired)?;
        skill.uses += 1;
        let text = format!("# {}\nWhen: {}\n\n{}", skill.name, skill.when, skill.steps);
        let skill = skill.clone();
        self.write(&skill);
        Some(text)
    }

    /// Creates a skill, or replaces the text of an existing one (keeping its
    /// history). Returns its name and whether it was new.
    pub fn save(&mut self, name: &str, when: &str, steps: &str) -> Result<(String, bool), String> {
        let name = slug(name);
        let when: String = when
            .trim()
            .lines()
            .next()
            .unwrap_or_default()
            .chars()
            .take(MAX_WHEN_CHARS)
            .collect();
        let steps: String = steps.trim().chars().take(MAX_STEPS_CHARS).collect();
        if name.len() < 3 {
            return Err("a skill needs a name of a few words".into());
        }
        if when.is_empty() || steps.is_empty() {
            return Err("a skill needs both when to use it and its steps".into());
        }
        let (skill, new) = match self.items.iter_mut().find(|s| s.name == name) {
            Some(existing) => {
                existing.when = when;
                existing.steps = steps;
                existing.updated = now();
                // A rewrite gets a fresh chance.
                existing.retired = false;
                existing.bad = 0;
                (existing.clone(), false)
            }
            None => {
                let skill = Skill {
                    name: name.clone(),
                    when,
                    steps,
                    uses: 0,
                    good: 0,
                    bad: 0,
                    updated: now(),
                    retired: false,
                };
                self.items.push(skill.clone());
                (skill, true)
            }
        };
        self.write(&skill);
        self.curate();
        Ok((name, new))
    }

    /// Counts the rating of a task against the skills it loaded, and retires
    /// any that this tips over. Returns the names retired.
    pub fn rate(&mut self, used: &[String], good: bool) -> Vec<String> {
        let mut changed = Vec::new();
        for skill in self.items.iter_mut().filter(|s| used.contains(&s.name)) {
            if good {
                skill.good += 1;
            } else {
                skill.bad += 1;
            }
            changed.push(skill.clone());
        }
        for skill in &changed {
            self.write(skill);
        }
        self.curate()
    }

    /// Retires skills the evidence is against, and the least useful ones if
    /// there are too many. Returns the names retired.
    fn curate(&mut self) -> Vec<String> {
        let mut retired = Vec::new();
        for skill in self.items.iter_mut().filter(|s| !s.retired) {
            if skill.bad >= RETIRE_AFTER_BAD && skill.bad > skill.good {
                skill.retired = true;
                retired.push(skill.name.clone());
            }
        }
        let active = self.items.iter().filter(|s| !s.retired).count();
        if active > MAX_ACTIVE {
            let mut order: Vec<(i64, u64, String)> = self
                .items
                .iter()
                .filter(|s| !s.retired)
                .map(|s| (s.worth(), s.updated, s.name.clone()))
                .collect();
            order.sort();
            for (_, _, name) in order.into_iter().take(active - MAX_ACTIVE) {
                if let Some(skill) = self.items.iter_mut().find(|s| s.name == name) {
                    skill.retired = true;
                    retired.push(name);
                }
            }
        }
        for name in &retired {
            if let Some(skill) = self.items.iter().find(|s| &s.name == name) {
                self.write(skill);
            }
        }
        retired
    }

    /// Every skill, for display: active first.
    pub fn report(&self) -> String {
        if self.items.is_empty() {
            return "no skills yet: one is written after a task that takes several steps".into();
        }
        let mut s = String::new();
        let mut all: Vec<&Skill> = self.items.iter().collect();
        all.sort_by(|a, b| a.retired.cmp(&b.retired).then(b.worth().cmp(&a.worth())));
        for skill in all {
            s.push_str(&format!(
                "{}{} · used {} · {} good, {} bad\n   {}\n",
                skill.name,
                if skill.retired { " (retired)" } else { "" },
                skill.uses,
                skill.good,
                skill.bad,
                skill.when
            ));
        }
        s
    }
}

/// What the model is asked once a task that took real work has ended.
pub const DISTILL: &str = "The task above is finished and it took several steps. Decide what, \
if anything, a future task should not have to work out again. Keep little: what is written here \
is carried into later work, and most of what happened in one task is that task's own detail. \
Four answers are possible.\n\
1. A skill: the way it was solved is a reusable procedure that would save time or avoid a \
mistake on similar tasks. If a skill you loaded was wrong or incomplete, write the corrected \
version under the same name.\n\
2. A rule: something about how to work that would hold for any task, in any project, with no \
knowledge of this one. It names no file, value, tool or project. Rare.\n\
3. Know-how: something you had to find out that will matter again for one kind of task: where \
something lives, how this project or machine behaves, a cause that was not the obvious one, a \
dead end. Say when it applies.\n\
4. Nothing: the work was routine, everything needed was given or obvious, or what you found \
matters only to this one task (a value that was wrong today, a file that was missing).\n\
Never write down what memory or an existing skill already holds (both are listed below).\n\
Reply in exactly one of these forms and nothing else:\n\
SKILL: <three to six words, lowercase, dashes>\n\
WHEN: <one line: the kind of task this is for>\n\
STEPS:\n\
<the procedure as a short numbered list: what to check first, what to do, how to verify, and \
any dead end to avoid. General, not tied to this one task's names.>\n\
or\n\
RULE: <one sentence, at most 30 words, that stands on its own>\n\
or\n\
KNOWHOW: <one or two sentences, at most 60 words, with exact names in `backticks`>\n\
WHEN: <one line: the kind of task it applies to>\n\
or\n\
SKIP: <one line: why nothing here is worth keeping>";

/// What a model made of a finished task.
#[derive(Debug, PartialEq)]
pub enum Distilled {
    Skill {
        name: String,
        when: String,
        steps: String,
    },
    /// How to work, whatever the task: holds with no knowledge of any
    /// one project.
    Rule(String),
    /// One thing found out that matters again for one kind of task, and
    /// when it applies (empty if the model did not say).
    KnowHow {
        text: String,
        when: String,
    },
    /// Nothing kept, and the model's reason (empty if it gave none).
    Skip(String),
    Unreadable,
}

/// Reads the reply to `DISTILL`.
pub fn read_distilled(reply: &str) -> Distilled {
    if let Some((name, when, steps)) = parse_distilled(reply) {
        return Distilled::Skill { name, when, steps };
    }
    let clean = reply.trim().trim_start_matches(['*', '#', ' ']);
    let upper = clean.to_ascii_uppercase();
    let rest = |word: &str| {
        upper.starts_with(word).then(|| {
            clean[word.len()..]
                .trim_start_matches(['*', ':', ' '])
                .trim()
                .to_string()
        })
    };
    // Know-how may carry a `WHEN:` line after its text.
    let split = |body: String| -> (String, String) {
        let mut text = Vec::new();
        let mut when = String::new();
        for line in body.lines() {
            let trimmed = line.trim().trim_start_matches(['*', ' ']);
            match trimmed.get(..4) {
                Some(start) if start.eq_ignore_ascii_case("WHEN") => {
                    when = trimmed[4..]
                        .trim_start_matches(['*', ':', ' '])
                        .trim()
                        .to_string();
                }
                _ => text.push(line.trim()),
            }
        }
        (text.join(" ").trim().to_string(), when)
    };
    if let Some(rule) = rest("RULE").filter(|r| !r.is_empty()) {
        return Distilled::Rule(split(rule).0);
    }
    // `LESSON` is the name know-how went by before the two were told apart.
    if let Some(body) = rest("KNOWHOW")
        .or_else(|| rest("KNOW-HOW"))
        .or_else(|| rest("LESSON"))
        .filter(|b| !b.is_empty())
    {
        let (text, when) = split(body);
        return Distilled::KnowHow { text, when };
    }
    match rest("SKIP") {
        Some(why) => Distilled::Skip(why.lines().next().unwrap_or_default().to_string()),
        None => Distilled::Unreadable,
    }
}

/// Reads the reply to `DISTILL`: a skill's name, when and steps, or nothing.
pub fn parse_distilled(reply: &str) -> Option<(String, String, String)> {
    let field = |name: &str| {
        reply.lines().find_map(|l| {
            let l = l.trim().trim_start_matches(['*', '#', ' ']);
            let rest = l
                .to_ascii_uppercase()
                .starts_with(name)
                .then(|| &l[name.len()..])?;
            Some(rest.trim_start_matches(['*', ':', ' ']).trim().to_string())
        })
    };
    let name = field("SKILL").filter(|n| !n.is_empty())?;
    let when = field("WHEN").filter(|w| !w.is_empty())?;
    let at = reply.to_ascii_uppercase().find("STEPS")?;
    let steps = reply[at + 5..]
        .trim_start_matches(['*', ':', ' '])
        .trim()
        .to_string();
    (!steps.is_empty()).then_some((name, when, steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_skill_is_saved_listed_loaded_and_refined() {
        let mut skills = Skills::in_memory();
        assert_eq!(skills.index(), "");
        let (name, new) = skills
            .save(
                "Fix a failing Python test",
                "A Python test fails",
                "1. Run it.\n2. Read the trace.",
            )
            .unwrap();
        assert_eq!((name.as_str(), new), ("fix-a-failing-python-test", true));
        assert!(
            skills
                .index()
                .ends_with("- fix-a-failing-python-test: A Python test fails\n")
        );

        let text = skills.open("fix-a-failing-python-test").unwrap();
        assert!(text.contains("When: A Python test fails") && text.contains("2. Read the trace."));
        assert_eq!(skills.active()[0].uses, 1);
        assert!(skills.open("no-such-skill").is_none());

        // Saving under the same name refines it and keeps its history.
        let (_, new) = skills
            .save(
                "fix-a-failing-python-test",
                "A Python test fails",
                "1. Run it.\n2. Read the trace.\n3. Fix, rerun.",
            )
            .unwrap();
        assert!(!new);
        assert_eq!(skills.active().len(), 1);
        assert_eq!(skills.active()[0].uses, 1);
        assert!(skills.active()[0].steps.ends_with("3. Fix, rerun."));

        assert!(skills.save("x", "when", "steps").is_err());
        assert!(skills.save("a real name", "", "steps").is_err());
    }

    #[test]
    fn ratings_rank_skills_and_retire_the_ones_that_keep_failing() {
        let mut skills = Skills::in_memory();
        skills.save("good one", "w", "s").unwrap();
        skills.save("bad one", "w", "s").unwrap();
        skills.save("unused", "w", "s").unwrap();
        let used = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();

        assert!(skills.rate(&used(&["good-one"]), true).is_empty());
        assert!(skills.rate(&used(&["bad-one"]), false).is_empty());
        let names: Vec<&str> = skills.active().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["good-one", "unused", "bad-one"]);

        // A second bad rating retires it; it is no longer offered or loadable.
        assert_eq!(skills.rate(&used(&["bad-one"]), false), ["bad-one"]);
        assert!(!skills.index().contains("bad-one"));
        assert!(skills.open("bad-one").is_none());
        assert!(skills.report().contains("bad-one (retired)"));

        // Rewriting it gives it a fresh chance.
        skills.save("bad one", "w", "better steps").unwrap();
        assert!(skills.open("bad-one").is_some());
    }

    #[test]
    fn too_many_skills_retires_the_least_useful() {
        let mut skills = Skills::in_memory();
        for i in 0..MAX_ACTIVE {
            skills.save(&format!("skill number {i}"), "w", "s").unwrap();
        }
        skills.rate(&["skill-number-0".to_string()], true);
        skills.save("one too many", "w", "s").unwrap();
        assert_eq!(skills.active().len(), MAX_ACTIVE);
        // The rated one survives; an unrated one went.
        assert!(skills.active().iter().any(|s| s.name == "skill-number-0"));
    }

    #[test]
    fn skills_survive_a_restart_as_readable_files() {
        let dir =
            std::env::temp_dir().join(format!("conscious-skills-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut skills = Skills::load(&dir);
        skills
            .save(
                "check before deploy",
                "Before shipping",
                "1. Test.\n2. Tag.",
            )
            .unwrap();
        skills.open("check-before-deploy");
        skills.rate(&["check-before-deploy".to_string()], true);
        std::fs::write(dir.join("notes.md"), "not a skill").unwrap();

        let file = std::fs::read_to_string(dir.join("check-before-deploy.md")).unwrap();
        assert!(file.starts_with(
            "---\nname: check-before-deploy\nwhen: Before shipping\nuses: 1\ngood: 1\n"
        ));
        assert!(file.ends_with("---\n1. Test.\n2. Tag.\n"));

        let again = Skills::load(&dir);
        assert_eq!(again.active().len(), 1);
        let s = again.active()[0];
        assert_eq!(
            (s.uses, s.good, s.steps.as_str()),
            (1, 1, "1. Test.\n2. Tag.")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn distilled_replies_are_read_or_skipped() {
        let reply = "SKILL: fix-failing-test\nWHEN: A test fails and the cause is unknown\nSTEPS:\n1. Run the test.\n2. Read the code it names.";
        assert_eq!(
            parse_distilled(reply),
            Some((
                "fix-failing-test".into(),
                "A test fails and the cause is unknown".into(),
                "1. Run the test.\n2. Read the code it names.".into()
            ))
        );
        assert_eq!(parse_distilled("SKIP"), None);
        assert_eq!(read_distilled("SKIP"), Distilled::Skip(String::new()));
        assert_eq!(
            read_distilled("SKIP: a routine edit\nnothing more"),
            Distilled::Skip("a routine edit".into())
        );
        assert_eq!(
            read_distilled("**LESSON:** Tests need `make seed` first."),
            Distilled::KnowHow {
                text: "Tests need `make seed` first.".into(),
                when: String::new()
            }
        );
        assert_eq!(
            read_distilled(
                "KNOWHOW: The fixtures live in `data/seed`,\nnot in the tests folder.\nWHEN: writing or fixing tests here"
            ),
            Distilled::KnowHow {
                text: "The fixtures live in `data/seed`, not in the tests folder.".into(),
                when: "writing or fixing tests here".into()
            }
        );
        assert_eq!(
            read_distilled(
                "RULE: Run the check that failed again after a fix, before saying it is fixed."
            ),
            Distilled::Rule(
                "Run the check that failed again after a fix, before saying it is fixed.".into()
            )
        );
        assert_eq!(
            read_distilled("I think it went well"),
            Distilled::Unreadable
        );
        assert!(matches!(read_distilled(reply), Distilled::Skill { .. }));
        assert_eq!(parse_distilled("SKILL: x\nWHEN: y"), None);
        assert_eq!(slug("  Fix: the BUILD!! "), "fix-the-build");
    }
}
