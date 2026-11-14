//! Learned routing evidence: what has been observed about each model family.
//!
//! This module only records facts: for every (family, complexity level) pair,
//! how many tasks were rated good and bad, plus latency and availability. It
//! takes no decisions. Turning these facts into a choice is the job of the
//! interpreted policy script (`policy`), which the core model can rewrite;
//! the evidence it reads cannot be rewritten by anyone.
//!
//! Evidence is keyed by family ("codex:gpt-sol", "claude-code:haiku"), not by
//! model version, so what was learned carries over when a backend ships a new
//! version of the same family.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::policy::Facts;

/// Complexity levels, 1 (trivial) to `LEVELS` (expert).
pub const LEVELS: usize = 5;

/// How much one outcome also counts for neighbouring levels: success at a
/// level is evidence for easier levels, failure for harder ones.
const SPILL: f64 = 0.5;
/// Weight of the newest latency sample in the running average.
const LATENCY_ALPHA: f64 = 0.3;
/// Rated tasks remembered individually, for the core's policy reviews.
const RATED_KEPT: usize = 30;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FamilyStats {
    /// Size tier: 1 small, 2 medium, 3 large. `None` until someone says.
    pub tier: Option<u8>,
    /// Good outcomes observed per level (fractional because of `SPILL`).
    pub good: [f64; LEVELS],
    pub bad: [f64; LEVELS],
    /// Running average seconds per call.
    pub latency_s: Option<f64>,
    pub calls: u64,
    /// Unix time until which this family is not offered, after it failed
    /// (a usage limit, an outage). Kept on disk so a new run does not retry
    /// a model that failed a minute ago.
    #[serde(default)]
    pub down_until: Option<u64>,
}

/// The last routed task, so feedback knows what it is about.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub family: String,
    /// 1..=LEVELS
    pub complexity: u8,
    /// Set once feedback has been given, so one answer is not counted twice.
    #[serde(default)]
    pub rated: bool,
}

/// One rated task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rated {
    pub family: String,
    pub complexity: u8,
    pub good: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Knowledge {
    families: BTreeMap<String, FamilyStats>,
    last: Option<Decision>,
    /// The most recent rated tasks, oldest first.
    #[serde(default)]
    rated: Vec<Rated>,
    /// Models the user has said not to use: a name, or part of one, and the
    /// Unix time the ban ends (`None` for until further notice).
    #[serde(default)]
    banned: BTreeMap<String, Option<u64>>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

/// Where routing knowledge lives unless `CONSCIOUS_KNOWLEDGE_FILE` says otherwise.
pub fn default_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("CONSCIOUS_KNOWLEDGE_FILE") {
        return Some(PathBuf::from(p));
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(data.join("conscious").join("routing.json"))
}

/// Seconds since the Unix epoch.
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Knowledge {
    /// Knowledge that is never written to disk.
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Loads knowledge from `path`. A missing file starts empty; an unreadable
    /// one also starts empty and says so, without overwriting it until the
    /// next save.
    pub fn load(path: &Path) -> (Self, Option<String>) {
        let (mut k, note) = match std::fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<Knowledge>(&text) {
                Ok(k) => (k, None),
                Err(e) => (
                    Knowledge::default(),
                    Some(format!(
                        "{}: {e}; starting with no routing knowledge",
                        path.display()
                    )),
                ),
            },
            Err(_) => (Knowledge::default(), None),
        };
        k.path = Some(path.to_path_buf());
        (k, note)
    }

    /// Writes knowledge back, via a temp file so a crash cannot truncate it.
    pub fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Where the policy versions live: next to the knowledge file.
    pub fn policy_dir(&self) -> Option<PathBuf> {
        Some(self.path.as_ref()?.parent()?.join("policy"))
    }

    pub fn tier(&self, family: &str) -> Option<u8> {
        self.families.get(family).and_then(|f| f.tier)
    }

    pub fn set_tier(&mut self, family: &str, tier: u8) {
        self.families.entry(family.to_string()).or_default().tier = Some(tier.clamp(1, 3));
    }

    /// Takes a family out of the pool for `seconds` from now.
    pub fn mark_down(&mut self, family: &str, seconds: u64) {
        self.families
            .entry(family.to_string())
            .or_default()
            .down_until = Some(now() + seconds);
    }

    /// Seconds until a family that failed may be offered again; 0 if it can
    /// be offered now.
    pub fn down_for(&self, family: &str) -> u64 {
        self.families
            .get(family)
            .and_then(|f| f.down_until)
            .map_or(0, |until| until.saturating_sub(now()))
    }

    /// Stops models whose name contains `name` from being used, for `days`
    /// or until allowed again.
    pub fn ban(&mut self, name: &str, days: Option<u64>) {
        self.banned
            .insert(name.trim().to_lowercase(), days.map(|d| now() + d * 86_400));
    }

    /// Lifts every ban that covers `name`. Returns how many were lifted.
    pub fn allow(&mut self, name: &str) -> usize {
        let name = name.trim().to_lowercase();
        let before = self.banned.len();
        self.banned
            .retain(|ban, _| !(ban.contains(&name) || name.contains(ban.as_str())));
        before - self.banned.len()
    }

    /// Whether the user has ruled this model out for now.
    pub fn is_banned(&self, spec: &str) -> bool {
        let spec = spec.to_lowercase();
        self.banned
            .iter()
            .any(|(ban, until)| spec.contains(ban.as_str()) && until.is_none_or(|t| now() < t))
    }

    /// The bans in force, as (name, days left if it expires).
    pub fn bans(&self) -> Vec<(String, Option<u64>)> {
        self.banned
            .iter()
            .filter(|(_, until)| until.is_none_or(|t| now() < t))
            .map(|(ban, until)| (ban.clone(), until.map(|t| (t - now()).div_ceil(86_400))))
            .collect()
    }

    pub fn stats(&self, family: &str) -> Option<&FamilyStats> {
        self.families.get(family)
    }

    /// The last routed task, if it has not been rated yet.
    pub fn pending(&self) -> Option<&Decision> {
        self.last.as_ref().filter(|d| !d.rated)
    }

    #[cfg(test)]
    pub fn last(&self) -> Option<&Decision> {
        self.last.as_ref()
    }

    /// What the policy is told about one family.
    pub fn facts(&self, family: &str) -> Facts {
        let stats = self.families.get(family).cloned().unwrap_or_default();
        Facts {
            family: family.to_string(),
            tier: stats.tier,
            good: stats.good,
            bad: stats.bad,
            calls: stats.calls,
            latency_s: stats.latency_s,
        }
    }

    /// Facts for every family that has been rated at least once: the recorded
    /// outcomes a policy rewrite is replayed against.
    pub fn rated_facts(&self) -> Vec<Facts> {
        self.families
            .iter()
            .filter(|(_, s)| s.good.iter().chain(&s.bad).any(|x| *x > 0.0))
            .map(|(family, _)| self.facts(family))
            .collect()
    }

    /// The recent rated tasks as text, oldest first.
    pub fn recent_ratings(&self) -> String {
        self.rated
            .iter()
            .map(|r| {
                format!(
                    "- {} at complexity {}: {}\n",
                    r.family,
                    r.complexity,
                    if r.good { "good" } else { "bad" }
                )
            })
            .collect()
    }

    /// Records that `family` ran a task of this level, making it the task the
    /// next feedback applies to.
    pub fn observe(&mut self, family: &str, level: u8, latency_s: f64) {
        let stats = self.families.entry(family.to_string()).or_default();
        stats.calls += 1;
        stats.latency_s = Some(match stats.latency_s {
            Some(avg) => avg + LATENCY_ALPHA * (latency_s - avg),
            None => latency_s,
        });
        self.last = Some(Decision {
            family: family.to_string(),
            complexity: level.clamp(1, LEVELS as u8),
            rated: false,
        });
    }

    /// Makes a particular run the one the next feedback applies to. Used
    /// when several tasks have run and the user rates one of them.
    pub fn point_at(&mut self, family: &str, level: u8) {
        self.last = Some(Decision {
            family: family.to_string(),
            complexity: level.clamp(1, LEVELS as u8),
            rated: false,
        });
    }

    /// Applies feedback to the last routed task. Returns what was rated, or
    /// why nothing was.
    pub fn feedback(&mut self, good: bool) -> Result<Rated, String> {
        let Some(last) = self.last.as_mut() else {
            return Err("no routed task to rate yet".into());
        };
        if last.rated {
            return Err("the last task has already been rated".into());
        }
        last.rated = true;
        let rated = Rated {
            family: last.family.clone(),
            complexity: last.complexity,
            good,
        };
        let l = rated.complexity as usize - 1;
        let stats = self.families.entry(rated.family.clone()).or_default();
        if good {
            stats.good[l] += 1.0;
            // Doing a hard task well is evidence for easier ones too.
            for easier in &mut stats.good[..l] {
                *easier += SPILL;
            }
        } else {
            stats.bad[l] += 1.0;
            // Failing an easy task is evidence against harder ones too.
            for harder in &mut stats.bad[l + 1..] {
                *harder += SPILL;
            }
        }
        self.rated.push(rated.clone());
        let excess = self.rated.len().saturating_sub(RATED_KEPT);
        self.rated.drain(..excess);
        Ok(rated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_counts_at_its_level_and_spills_to_neighbours() {
        let mut k = Knowledge::in_memory();
        k.observe("small", 3, 1.0);
        let rated = k.feedback(true).unwrap();
        assert_eq!(
            (rated.family.as_str(), rated.complexity, rated.good),
            ("small", 3, true)
        );
        // Success at level 3 is half-evidence for levels 1 and 2, none above.
        assert_eq!(k.stats("small").unwrap().good, [0.5, 0.5, 1.0, 0.0, 0.0]);

        k.observe("small", 2, 1.0);
        k.feedback(false).unwrap();
        // Failure at level 2 is half-evidence against levels 3 to 5, none below.
        assert_eq!(k.stats("small").unwrap().bad, [0.0, 1.0, 0.5, 0.5, 0.5]);
        assert_eq!(
            k.recent_ratings(),
            "- small at complexity 3: good\n- small at complexity 2: bad\n"
        );
    }

    #[test]
    fn feedback_needs_a_task_and_counts_once() {
        let mut k = Knowledge::in_memory();
        assert!(k.feedback(true).is_err());
        k.observe("small", 2, 1.0);
        assert!(k.feedback(true).is_ok());
        assert!(k.feedback(true).is_err());
        assert_eq!(k.stats("small").unwrap().good[1], 1.0);
    }

    #[test]
    fn only_the_recent_ratings_are_kept_individually() {
        let mut k = Knowledge::in_memory();
        for _ in 0..RATED_KEPT + 5 {
            k.observe("m", 1, 1.0);
            k.feedback(true).unwrap();
        }
        assert_eq!(k.rated.len(), RATED_KEPT);
        assert_eq!(k.stats("m").unwrap().good[0], (RATED_KEPT + 5) as f64);
    }

    #[test]
    fn facts_carry_the_evidence_and_only_rated_families_are_replayed() {
        let mut k = Knowledge::in_memory();
        k.set_tier("seen", 2);
        k.set_tier("unrated", 1);
        k.observe("seen", 4, 2.0);
        k.feedback(false).unwrap();
        let f = k.facts("seen");
        assert_eq!((f.tier, f.calls, f.latency_s), (Some(2), 1, Some(2.0)));
        assert_eq!(f.bad[3], 1.0);
        assert_eq!(
            k.facts("never-seen"),
            Facts {
                family: "never-seen".into(),
                ..Default::default()
            }
        );
        let rated: Vec<String> = k.rated_facts().into_iter().map(|f| f.family).collect();
        assert_eq!(rated, ["seen"]);
    }

    #[test]
    fn banned_models_stay_out_until_allowed_or_the_ban_runs_out() {
        let mut k = Knowledge::in_memory();
        k.ban("Codex", None);
        k.ban("opus", Some(3));
        assert!(k.is_banned("codex:gpt-6.1-sol") && k.is_banned("claude-code:opus"));
        assert!(!k.is_banned("claude-code:haiku"));
        assert_eq!(k.bans(), [("codex".into(), None), ("opus".into(), Some(3))]);
        assert_eq!(k.allow("codex"), 1);
        assert!(!k.is_banned("codex:gpt-6.1-sol"));
        // A ban that has run out no longer applies.
        k.banned.insert("opus".into(), Some(now() - 1));
        assert!(!k.is_banned("claude-code:opus") && k.bans().is_empty());
    }

    #[test]
    fn latency_is_a_running_average() {
        let mut k = Knowledge::in_memory();
        k.observe("small", 1, 2.0);
        k.observe("small", 1, 4.0);
        let s = k.stats("small").unwrap();
        assert_eq!(s.calls, 2);
        assert!((s.latency_s.unwrap() - 2.6).abs() < 1e-9);
    }

    #[test]
    fn knowledge_survives_a_save_and_load() {
        let dir = std::env::temp_dir().join(format!("conscious-learn-test-{}", std::process::id()));
        let path = dir.join("nested").join("routing.json");
        let (mut k, note) = Knowledge::load(&path);
        assert!(note.is_none());
        assert_eq!(k.policy_dir().unwrap(), dir.join("nested").join("policy"));
        k.set_tier("small", 1);
        k.observe("small", 3, 1.5);
        k.feedback(false).unwrap();
        k.mark_down("small", 600);
        k.save().unwrap();

        let (again, note) = Knowledge::load(&path);
        assert!(note.is_none());
        assert!((590..=600).contains(&again.down_for("small")));
        assert_eq!(again.down_for("never-seen"), 0);
        assert_eq!(again.tier("small"), Some(1));
        assert_eq!(again.stats("small").unwrap().bad[2], 1.0);
        assert_eq!(again.last().unwrap().family, "small");
        assert!(again.last().unwrap().rated);
        assert_eq!(again.recent_ratings(), "- small at complexity 3: bad\n");

        std::fs::write(&path, "{broken").unwrap();
        let (empty, note) = Knowledge::load(&path);
        assert!(note.unwrap().contains("starting with no routing knowledge"));
        assert!(empty.stats("small").is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
