//! The interpreted decision policy.
//!
//! How a model is chosen for a task is not compiled into the harness: it is a
//! Rhai script (`policy/default.rhai` is version 0) that the harness
//! interprets. The core model may rewrite it. That is the non-deterministic
//! half. Everything around it is deterministic:
//!
//! - **Sandbox.** A script sees only the facts passed in and returns a choice.
//!   It has no file, network or module access and a hard step limit.
//! - **Regression gate.** A rewrite is adopted only if `check` finds no
//!   problem: it must be small, never fail, give the same answer twice, keep
//!   a set of behaviours that must always hold, and do no worse than the
//!   current policy on the outcomes recorded so far. No approval is asked.
//! - **Rollback.** Every adopted version is kept on disk. A version that
//!   errors at run time, or collects bad ratings during its probation, is
//!   retired and the previous one restored, down to the built-in script.
//!
//! Guardrails and this gate stay in compiled code; a script cannot reach them.

use std::path::{Path, PathBuf};

use rhai::{AST, Array, Dynamic, Engine, Map, Scope};
use serde::{Deserialize, Serialize};

use crate::learn::{LEVELS, now};

const DEFAULT_SOURCE: &str = include_str!("policy/default.rhai");

/// Largest script accepted, in bytes. Keeps rewrites concise and reviewable.
pub const MAX_SOURCE_BYTES: usize = 8_000;
/// Interpreter steps one call may take before it is stopped.
const MAX_OPERATIONS: u64 = 200_000;
/// How much worse than the current policy a rewrite may score on recorded
/// outcomes (as a fraction of the evidence) and still be adopted.
const REPLAY_TOLERANCE: f64 = 0.02;
/// A new version is on probation for this many rated tasks.
const PROBATION: u32 = 6;
/// During probation, this many bad ratings (and more bad than good) send it back.
const PROBATION_BAD: u32 = 2;
/// A review is due after this many rated tasks, or sooner after a bad one.
const REVIEW_EVERY: u32 = 5;
/// Rejected rewrites remembered, so the core can see what did not pass.
const REJECTED_KEPT: usize = 5;

/// What a script is told about one candidate model family.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Facts {
    pub family: String,
    /// 1 small, 2 medium, 3 large; `None` if not known.
    pub tier: Option<u8>,
    pub good: [f64; LEVELS],
    pub bad: [f64; LEVELS],
    pub calls: u64,
    pub latency_s: Option<f64>,
}

impl Facts {
    fn to_dynamic(&self) -> Dynamic {
        let floats =
            |v: &[f64; LEVELS]| -> Array { v.iter().map(|x| Dynamic::from_float(*x)).collect() };
        let mut m = Map::new();
        m.insert("family".into(), self.family.clone().into());
        m.insert(
            "tier".into(),
            Dynamic::from_int(self.tier.unwrap_or(0) as i64),
        );
        m.insert("good".into(), Dynamic::from_array(floats(&self.good)));
        m.insert("bad".into(), Dynamic::from_array(floats(&self.bad)));
        m.insert("calls".into(), Dynamic::from_int(self.calls as i64));
        m.insert(
            "latency".into(),
            Dynamic::from_float(self.latency_s.unwrap_or(-1.0)),
        );
        Dynamic::from_map(m)
    }
}

/// One compiled policy script.
pub struct Policy {
    engine: Engine,
    ast: AST,
    source: String,
}

fn sandbox() -> Engine {
    let mut engine = Engine::new();
    engine
        .set_max_operations(MAX_OPERATIONS)
        .set_max_call_levels(24)
        .set_max_expr_depths(64, 32)
        .set_max_string_size(4_000)
        .set_max_array_size(2_000)
        .set_max_map_size(200)
        .set_max_modules(0)
        .set_module_resolver(rhai::module_resolvers::DummyModuleResolver::new())
        .disable_symbol("eval")
        // Scripts must not write into the terminal the TUI is drawing on.
        .on_print(|_| {})
        .on_debug(|_, _, _| {});
    engine
}

impl Policy {
    /// The policy compiled into the binary: the last resort of every rollback.
    pub fn builtin() -> Self {
        Self::compile(DEFAULT_SOURCE).expect("the built-in policy compiles")
    }

    pub fn compile(source: &str) -> Result<Self, String> {
        let engine = sandbox();
        let ast = engine
            .compile(source)
            .map_err(|e| format!("does not compile: {e}"))?;
        Ok(Self {
            engine,
            ast,
            source: source.to_string(),
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// The script's estimate that `facts` handles a task of this level well.
    pub fn estimate(&self, facts: &Facts, level: u8) -> Result<f64, String> {
        let out: Dynamic = self
            .engine
            .call_fn(
                &mut Scope::new(),
                &self.ast,
                "estimate",
                (facts.to_dynamic(), level as i64),
            )
            .map_err(|e| format!("estimate failed: {e}"))?;
        let p = out
            .as_float()
            .or_else(|_| out.as_int().map(|i| i as f64))
            .map_err(|t| format!("estimate returned {t}, not a number"))?;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(format!("estimate returned {p}, outside 0..1"));
        }
        Ok(p)
    }

    /// Which candidate the script picks, as a position in `candidates`, and
    /// its reason. Any answer that is not a valid position is an error.
    pub fn select(&self, level: u8, candidates: &[Facts]) -> Result<(usize, String), String> {
        let list: Array = candidates.iter().map(Facts::to_dynamic).collect();
        let out: Dynamic = self
            .engine
            .call_fn(&mut Scope::new(), &self.ast, "select", (level as i64, list))
            .map_err(|e| format!("select failed: {e}"))?;
        let (index, reason) = match out.as_int() {
            Ok(i) => (i, String::new()),
            Err(_) => {
                let map = out
                    .try_cast::<Map>()
                    .ok_or("select returned neither an index nor a map")?;
                let index = map
                    .get("index")
                    .and_then(|d| d.as_int().ok())
                    .ok_or("select returned no integer `index`")?;
                let reason = map.get("reason").map(|d| d.to_string()).unwrap_or_default();
                (index, reason)
            }
        };
        if index < 0 || index as usize >= candidates.len() {
            return Err(format!(
                "select returned index {index} for {} candidates",
                candidates.len()
            ));
        }
        let reason: String = reason.chars().take(160).collect();
        Ok((
            index as usize,
            if reason.trim().is_empty() {
                "policy gave no reason".into()
            } else {
                reason
            },
        ))
    }
}

// ───────────────────────── regression suite ─────────────────────────

fn family(name: &str, tier: u8) -> Facts {
    Facts {
        family: name.into(),
        tier: (tier > 0).then_some(tier),
        ..Default::default()
    }
}

fn with_evidence(mut f: Facts, level: u8, good: f64, bad: f64) -> Facts {
    f.good[level as usize - 1] = good;
    f.bad[level as usize - 1] = bad;
    f
}

/// How well a policy's choices line up with the outcomes actually recorded:
/// for each level, the observed success rate (Laplace-smoothed) of the family
/// it picks, weighted by how much was observed at that level.
fn replay_score(policy: &Policy, history: &[Facts]) -> Result<(f64, f64), String> {
    let mut score = 0.0;
    let mut weight = 0.0;
    for level in 1..=LEVELS as u8 {
        let l = level as usize - 1;
        let seen: f64 = history.iter().map(|f| f.good[l] + f.bad[l]).sum();
        if seen == 0.0 {
            continue;
        }
        let (pick, _) = policy.select(level, history)?;
        let f = &history[pick];
        score += seen * (f.good[l] + 1.0) / (f.good[l] + f.bad[l] + 2.0);
        weight += seen;
    }
    Ok((score, weight))
}

/// Runs the regression suite on a candidate policy. `baseline` is the policy
/// in use and `history` the real recorded facts. Returns every problem found;
/// empty means the candidate may be adopted. Fully deterministic.
pub fn check(candidate: &Policy, baseline: &Policy, history: &[Facts]) -> Vec<String> {
    let mut problems = Vec::new();
    if candidate.source().len() > MAX_SOURCE_BYTES {
        problems.push(format!(
            "script is {} bytes; the limit is {MAX_SOURCE_BYTES}",
            candidate.source().len()
        ));
    }

    // 1. It always answers, validly, and the same way twice.
    let pools: [Vec<Facts>; 5] = [
        vec![family("only", 2)],
        vec![family("small", 1), family("medium", 2), family("large", 3)],
        vec![family("large", 3), family("medium", 2), family("small", 1)],
        vec![family("unknown-a", 0), family("unknown-b", 0)],
        vec![
            with_evidence(family("small", 1), 2, 3.0, 1.0),
            with_evidence(family("medium", 2), 4, 0.5, 2.0),
            with_evidence(family("large", 3), 5, 4.0, 0.0),
        ],
    ];
    for (p, pool) in pools.iter().enumerate() {
        for level in 1..=LEVELS as u8 {
            match (candidate.select(level, pool), candidate.select(level, pool)) {
                (Ok(a), Ok(b)) if a == b => {}
                (Ok(_), Ok(_)) => problems.push(format!(
                    "pool {p}, level {level}: two runs gave different answers"
                )),
                (Err(e), _) | (_, Err(e)) => problems.push(format!("pool {p}, level {level}: {e}")),
            }
            for f in pool {
                if let Err(e) = candidate.estimate(f, level) {
                    problems.push(format!("pool {p}, level {level}, {}: {e}", f.family));
                }
            }
        }
    }
    if !problems.is_empty() {
        // The behavioural checks below would only repeat the same failures.
        return problems;
    }

    // 2. Behaviours that must always hold.
    let pick = |level: u8, pool: &[Facts]| -> String {
        candidate
            .select(level, pool)
            .map(|(i, _)| pool[i].family.clone())
            .unwrap_or_default()
    };
    let tiers = &pools[1];
    if pick(1, tiers) == "large" {
        problems.push("a trivial task with no evidence went to the large model".into());
    }
    if pick(LEVELS as u8, tiers) == "small" {
        problems.push("an expert task with no evidence went to the small model".into());
    }
    if pick(3, &pools[0]) != "only" {
        problems.push("with one candidate, it was not chosen".into());
    }
    for level in 1..=LEVELS as u8 {
        let proven = [
            with_evidence(family("failing", 1), level, 0.0, 10.0),
            with_evidence(family("working", 3), level, 10.0, 0.0),
        ];
        if pick(level, &proven) != "working" {
            problems.push(format!(
                "level {level}: chose a model with 10 bad ratings over one with 10 good"
            ));
        }
        let both_good = [
            with_evidence(family("costly", 3), level, 10.0, 0.0),
            with_evidence(family("cheap", 1), level, 10.0, 0.0),
        ];
        if pick(level, &both_good) != "cheap" {
            problems.push(format!(
                "level {level}: two models proven equally good, but the costlier was chosen"
            ));
        }
        let base = family("m", 2);
        let est = |f: &Facts| candidate.estimate(f, level).unwrap_or(f64::NAN);
        if est(&with_evidence(base.clone(), level, 5.0, 0.0)) < est(&base) {
            problems.push(format!("level {level}: good ratings lowered the estimate"));
        }
        if est(&with_evidence(base.clone(), level, 0.0, 5.0)) > est(&base) {
            problems.push(format!("level {level}: bad ratings raised the estimate"));
        }
    }

    // 3. No worse than the policy in use on what has actually been observed.
    if !history.is_empty() {
        match (
            replay_score(candidate, history),
            replay_score(baseline, history),
        ) {
            (Ok((new, weight)), Ok((old, _))) if weight > 0.0 => {
                if new < old - REPLAY_TOLERANCE * weight {
                    problems.push(format!(
                        "on recorded outcomes it scores {:.0}% against the current policy's {:.0}%",
                        100.0 * new / weight,
                        100.0 * old / weight
                    ));
                }
            }
            (Err(e), _) => problems.push(format!("replay of recorded outcomes: {e}")),
            // No recorded outcomes yet, or the baseline itself cannot replay.
            _ => {}
        }
    }
    problems
}

// ───────────────────────── versions on disk ─────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Active,
    /// Replaced by a newer version; a rollback may restore it.
    Retired,
    /// Withdrawn after misbehaving; never restored.
    RolledBack,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: u32,
    /// The core's stated reason for the rewrite.
    pub why: String,
    pub adopted_at: u64,
    /// Ratings received while this version was active.
    pub good: u32,
    pub bad: u32,
    pub status: Status,
    /// Why it was rolled back, if it was.
    #[serde(default)]
    pub ended: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rejected {
    pub at: u64,
    pub why: String,
    pub problems: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct State {
    /// 0 is the built-in policy.
    active: u32,
    versions: Vec<VersionInfo>,
    rejected: Vec<Rejected>,
    review_due: bool,
    rated_since_review: u32,
}

/// The policy in use, its history, and the rules for changing it.
pub struct PolicyStore {
    dir: Option<PathBuf>,
    policy: Policy,
    state: State,
}

impl PolicyStore {
    /// A store that starts on the built-in policy and writes nothing.
    pub fn in_memory() -> Self {
        Self {
            dir: None,
            policy: Policy::builtin(),
            state: State::default(),
        }
    }

    /// Loads the active version from `dir`. If its file is missing or no
    /// longer compiles, rolls back until something loads. The second value
    /// lists anything worth telling the user.
    pub fn load(dir: &Path) -> (Self, Vec<String>) {
        let mut notes = Vec::new();
        let state = match std::fs::read_to_string(dir.join("history.json")) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                notes.push(format!(
                    "policy history unreadable ({e}); using the built-in policy"
                ));
                State::default()
            }),
            Err(_) => State::default(),
        };
        let mut store = Self {
            dir: Some(dir.to_path_buf()),
            policy: Policy::builtin(),
            state,
        };
        while store.state.active != 0 {
            match store.read_version(store.state.active) {
                Ok(policy) => {
                    store.policy = policy;
                    break;
                }
                Err(e) => notes.push(store.rollback(&format!("could not be loaded: {e}"))),
            }
        }
        (store, notes)
    }

    fn version_path(&self, version: u32) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|d| d.join(format!("v{version:04}.rhai")))
    }

    fn read_version(&self, version: u32) -> Result<Policy, String> {
        let path = self.version_path(version).ok_or("no policy directory")?;
        let source =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Policy::compile(&source)
    }

    fn save(&self) -> Result<(), String> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let text = serde_json::to_string_pretty(&self.state).map_err(|e| e.to_string())?;
        let tmp = dir.join("history.json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, dir.join("history.json")).map_err(|e| e.to_string())
    }

    pub fn version(&self) -> u32 {
        self.state.active
    }

    pub fn source(&self) -> &str {
        self.policy.source()
    }

    pub fn versions(&self) -> &[VersionInfo] {
        &self.state.versions
    }

    pub fn rejected(&self) -> &[Rejected] {
        &self.state.rejected
    }

    fn info_mut(&mut self, version: u32) -> Option<&mut VersionInfo> {
        self.state
            .versions
            .iter_mut()
            .find(|v| v.version == version)
    }

    /// Withdraws the active version and restores the newest earlier one that
    /// was never rolled back, or the built-in policy. Returns what happened.
    pub fn rollback(&mut self, reason: &str) -> String {
        let from = self.state.active;
        if from == 0 {
            return "already on the built-in policy; nothing to roll back".into();
        }
        if let Some(info) = self.info_mut(from) {
            info.status = Status::RolledBack;
            info.ended = Some(reason.to_string());
        }
        // Walk down until a version loads; the built-in one always does.
        let mut to = 0;
        let mut policy = Policy::builtin();
        let earlier: Vec<u32> = self
            .state
            .versions
            .iter()
            .rev()
            .filter(|v| v.version < from && v.status == Status::Retired)
            .map(|v| v.version)
            .collect();
        for version in earlier {
            match self.read_version(version) {
                Ok(p) => {
                    to = version;
                    policy = p;
                    break;
                }
                Err(e) => {
                    if let Some(info) = self.info_mut(version) {
                        info.status = Status::RolledBack;
                        info.ended = Some(format!("could not be loaded: {e}"));
                    }
                }
            }
        }
        if let Some(info) = self.info_mut(to) {
            info.status = Status::Active;
        }
        self.state.active = to;
        self.policy = policy;
        let _ = self.save();
        format!(
            "policy v{from} rolled back to {}: {reason}",
            if to == 0 {
                "the built-in policy".to_string()
            } else {
                format!("v{to}")
            }
        )
    }

    /// Picks a candidate with the active policy. If the script fails or gives
    /// an invalid answer it is rolled back on the spot and the previous one
    /// asked, down to the built-in policy; if even that fails, the first
    /// candidate (the core) is used. Returns the position, the reason, and a
    /// note per rollback.
    pub fn select(&mut self, level: u8, candidates: &[Facts]) -> (usize, String, Vec<String>) {
        let mut notes = Vec::new();
        loop {
            match self.policy.select(level, candidates) {
                Ok((index, reason)) => return (index, reason, notes),
                Err(e) if self.state.active == 0 => {
                    notes.push(format!("the built-in policy failed ({e}); using the core"));
                    return (0, "policy unavailable".into(), notes);
                }
                Err(e) => notes.push(self.rollback(&e)),
            }
        }
    }

    /// The active policy's estimate, falling back to the built-in one if the
    /// script cannot give a number. Used for display only.
    pub fn estimate(&self, facts: &Facts, level: u8) -> f64 {
        self.policy
            .estimate(facts, level)
            .or_else(|_| Policy::builtin().estimate(facts, level))
            .unwrap_or(0.0)
    }

    /// Runs the regression suite on a proposed rewrite and adopts it if it
    /// passes. `Ok` is the new version number; `Err` lists the problems,
    /// which are also remembered so the core can see them next time.
    pub fn propose(
        &mut self,
        source: &str,
        why: &str,
        history: &[Facts],
    ) -> Result<u32, Vec<String>> {
        let problems = match Policy::compile(source) {
            Ok(candidate) if candidate.source().trim() == self.policy.source().trim() => {
                vec!["identical to the current policy".to_string()]
            }
            Ok(candidate) => {
                let problems = check(&candidate, &self.policy, history);
                if problems.is_empty() {
                    return Ok(self.adopt(candidate, why));
                }
                problems
            }
            Err(e) => vec![e],
        };
        self.state.rejected.push(Rejected {
            at: now(),
            why: why.to_string(),
            problems: problems.clone(),
        });
        let excess = self.state.rejected.len().saturating_sub(REJECTED_KEPT);
        self.state.rejected.drain(..excess);
        let _ = self.save();
        Err(problems)
    }

    fn adopt(&mut self, candidate: Policy, why: &str) -> u32 {
        let version = self
            .state
            .versions
            .iter()
            .map(|v| v.version)
            .max()
            .unwrap_or(0)
            + 1;
        if let Some(path) = self.version_path(version) {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&path, candidate.source());
        }
        let previous = self.state.active;
        if let Some(info) = self.info_mut(previous) {
            info.status = Status::Retired;
        }
        self.state.versions.push(VersionInfo {
            version,
            why: why.chars().take(300).collect(),
            adopted_at: now(),
            good: 0,
            bad: 0,
            status: Status::Active,
            ended: None,
        });
        self.state.active = version;
        self.policy = candidate;
        let _ = self.save();
        version
    }

    /// Counts a rating against the active version. A rewrite that collects
    /// bad ratings during its probation is rolled back; the returned note
    /// says so. Also decides whether the core should review the policy.
    pub fn record_feedback(&mut self, good: bool) -> Option<String> {
        self.state.rated_since_review += 1;
        if !good || self.state.rated_since_review >= REVIEW_EVERY {
            self.state.review_due = true;
        }
        let active = self.state.active;
        let verdict = self.info_mut(active).map(|info| {
            if good {
                info.good += 1;
            } else {
                info.bad += 1;
            }
            (info.good, info.bad)
        });
        let note = match verdict {
            Some((good, bad)) if good + bad <= PROBATION && bad >= PROBATION_BAD && bad > good => {
                Some(self.rollback(&format!(
                    "{bad} bad and {good} good ratings in its first {} rated tasks",
                    good + bad
                )))
            }
            _ => None,
        };
        let _ = self.save();
        note
    }

    pub fn review_due(&self) -> bool {
        self.state.review_due
    }

    pub fn review_done(&mut self) {
        self.state.review_due = false;
        self.state.rated_since_review = 0;
        let _ = self.save();
    }
}

// ───────────────────────── the core's review ─────────────────────────

pub const REVIEW_SYSTEM: &str = "You maintain the routing policy of an agent harness. The policy \
is a Rhai script that decides which model family runs a task of a given complexity. You are shown \
the current script, the outcomes recorded so far, and recent ratings.\n\n\
Decide whether the script should change. Change it only if the evidence shows a better rule; \
otherwise keep it. Reply in one of two ways:\n\
- `KEEP` and nothing else, or\n\
- a first line `WHY: <one sentence>`, then the complete new script in one ```rhai code block.\n\n\
A rewrite is adopted automatically only if it passes a regression suite, and is rolled back if \
it then performs badly. It must: keep the contract in the script's header (functions `estimate` \
and `select`); stay under 8000 bytes; never error; be deterministic (no randomness, no time); \
never send a trivial task to the largest model or an expert task to the smallest when nothing is \
known; prefer a family with 10 good ratings over one with 10 bad; prefer the cheaper of two \
equally proven families; never let good ratings lower an estimate or bad ones raise it; and do \
no worse than the current script on the recorded outcomes.\n\
Rhai notes: functions cannot read variables defined outside them; write floats with a decimal \
point; maps are `#{ key: value }`; no imports or I/O.";

/// What the core is shown when asked to review the policy.
pub fn review_prompt(store: &PolicyStore, history: &[Facts], recent: &str) -> String {
    let mut s = format!(
        "Current policy (version {}):\n```rhai\n{}\n```\n\n",
        store.version(),
        store.source()
    );
    s.push_str("Recorded outcomes per family (good/bad ratings at complexity 1..5):\n");
    for f in history {
        let cells: Vec<String> = (0..LEVELS)
            .map(|l| format!("{}/{}", trim(f.good[l]), trim(f.bad[l])))
            .collect();
        s.push_str(&format!(
            "- {} (tier {}, {} calls): {}\n",
            f.family,
            f.tier.map_or("unknown".into(), |t| t.to_string()),
            f.calls,
            cells.join("  ")
        ));
    }
    s.push_str("\nRecent ratings, oldest first:\n");
    s.push_str(if recent.is_empty() {
        "(none)\n"
    } else {
        recent
    });
    let failed: Vec<&VersionInfo> = store
        .versions()
        .iter()
        .filter(|v| v.status == Status::RolledBack)
        .collect();
    if !failed.is_empty() || !store.rejected().is_empty() {
        s.push_str("\nEarlier rewrites that did not last (do not repeat them):\n");
        for v in failed.iter().rev().take(3) {
            s.push_str(&format!(
                "- v{} \"{}\" was rolled back: {}\n",
                v.version,
                v.why,
                v.ended.as_deref().unwrap_or("no reason recorded")
            ));
        }
        for r in store.rejected().iter().rev().take(3) {
            s.push_str(&format!(
                "- \"{}\" was rejected: {}\n",
                r.why,
                r.problems.join("; ")
            ));
        }
    }
    s
}

fn trim(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{x:.0}")
    } else {
        format!("{x:.1}")
    }
}

/// The policy in use and how it got there, for display.
pub fn summary(store: &PolicyStore) -> String {
    let mut s = format!(
        "active: {}\n",
        match store.version() {
            0 => "v0 (built in)".to_string(),
            v => format!("v{v}"),
        }
    );
    for v in store.versions() {
        s.push_str(&format!(
            "  v{} · {} · {} good, {} bad · {}{}\n",
            v.version,
            match v.status {
                Status::Active => "active",
                Status::Retired => "replaced",
                Status::RolledBack => "rolled back",
            },
            v.good,
            v.bad,
            v.why,
            v.ended
                .as_ref()
                .map(|e| format!(" · ended: {e}"))
                .unwrap_or_default()
        ));
    }
    for r in store.rejected() {
        s.push_str(&format!(
            "  rejected · {} · {}\n",
            r.why,
            r.problems.join("; ")
        ));
    }
    s
}

#[derive(Debug, PartialEq)]
pub enum Review {
    Keep,
    Rewrite {
        why: String,
        source: String,
    },
    /// Neither `KEEP` nor a script: treated as keep, and reported.
    Unreadable,
}

/// Reads the core's reply to a review.
pub fn parse_review(reply: &str) -> Review {
    let Some(open) = reply.find("```") else {
        return if reply.to_ascii_uppercase().contains("KEEP") {
            Review::Keep
        } else {
            Review::Unreadable
        };
    };
    // Skip the rest of the opening fence line (the language tag).
    let body_start = reply[open..]
        .find('\n')
        .map_or(reply.len(), |n| open + n + 1);
    let Some(close) = reply[body_start..].find("```") else {
        return Review::Unreadable;
    };
    let source = reply[body_start..body_start + close].trim().to_string();
    if source.is_empty() {
        return Review::Unreadable;
    }
    let why = reply[..open]
        .lines()
        .find_map(|l| {
            let at = l.to_ascii_uppercase().find("WHY")?;
            Some(
                l[at + 3..]
                    .trim_start_matches([':', ' ', '-', '*'])
                    .trim()
                    .to_string(),
            )
        })
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| "no reason given".into());
    Review::Rewrite { why, source }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiers() -> Vec<Facts> {
        vec![family("small", 1), family("medium", 2), family("large", 3)]
    }

    fn pick(policy: &Policy, level: u8, pool: &[Facts]) -> String {
        let (i, _) = policy.select(level, pool).unwrap();
        pool[i].family.clone()
    }

    /// A valid rewrite: always the lowest tier whose estimate is at least 0.6.
    const REWRITE: &str = r#"
fn prior(tier, level) {
    let t = if tier < 1 { 2 } else if tier > 3 { 3 } else { tier };
    let table = [[0.9, 0.8, 0.5, 0.3, 0.15], [0.95, 0.9, 0.85, 0.65, 0.45], [0.97, 0.95, 0.92, 0.88, 0.85]];
    table[t - 1][level - 1]
}
fn estimate(c, level) {
    (2.0 * prior(c.tier, level) + c.good[level - 1]) / (2.0 + c.good[level - 1] + c.bad[level - 1])
}
fn select(level, candidates) {
    let pick = -1; let pick_tier = 99; let best = 0; let best_mean = -1.0;
    for i in 0..candidates.len() {
        let c = candidates[i];
        let mean = estimate(c, level);
        let tier = if c.tier < 1 { 2 } else { c.tier };
        if mean >= 0.6 && tier < pick_tier { pick = i; pick_tier = tier; }
        if mean > best_mean { best = i; best_mean = mean; }
    }
    if pick >= 0 { #{ index: pick, reason: "lowest tier at 60%" } } else { #{ index: best, reason: "best available" } }
}
"#;

    #[test]
    fn builtin_policy_maps_complexity_to_tier() {
        let p = Policy::builtin();
        let pool = tiers();
        assert_eq!(pick(&p, 1, &pool), "small");
        assert_eq!(pick(&p, 2, &pool), "small");
        assert_eq!(pick(&p, 3, &pool), "medium");
        // Optimism lets the cheaper model try level 4 first.
        assert_eq!(pick(&p, 4, &pool), "medium");
        assert_eq!(pick(&p, 5, &pool), "large");
        let (_, reason) = p.select(1, &pool).unwrap();
        assert_eq!(reason, "cheapest model expected to handle it");
    }

    #[test]
    fn builtin_policy_follows_the_evidence() {
        let p = Policy::builtin();
        let mut pool = tiers();
        // One bad rating moves level 4 off the medium model.
        pool[1].bad[3] = 1.0;
        assert_eq!(pick(&p, 4, &pool), "large");
        // Good ratings let the small model take level 3.
        pool[0].good[2] = 4.0;
        assert_eq!(pick(&p, 3, &pool), "small");
        let est = p.estimate(&pool[0], 3).unwrap();
        assert!((est - 5.0 / 6.0).abs() < 1e-9, "{est}");
    }

    #[test]
    fn builtin_policy_passes_its_own_regression_suite() {
        let p = Policy::builtin();
        assert_eq!(check(&p, &p, &tiers()), Vec::<String>::new());
    }

    #[test]
    fn scripts_are_sandboxed() {
        // Runaway loop: stopped by the step limit, not hung.
        let spin =
            Policy::compile("fn estimate(c, l) { 0.5 } fn select(l, c) { loop { } }").unwrap();
        assert!(
            spin.select(1, &tiers())
                .unwrap_err()
                .contains("select failed")
        );
        // No modules, no eval.
        assert!(
            Policy::compile("import \"os\" as os; fn select(l, c) { 0 }")
                .and_then(|p| p.select(1, &tiers()).map(|_| ()))
                .is_err()
        );
        assert!(Policy::compile("fn select(l, c) { eval(\"0\") }").is_err());
        // An out-of-range answer is refused.
        let wild = Policy::compile("fn estimate(c, l) { 0.5 } fn select(l, c) { 7 }").unwrap();
        assert!(wild.select(1, &tiers()).unwrap_err().contains("index 7"));
        let odd = Policy::compile("fn estimate(c, l) { 1.5 } fn select(l, c) { 0 }").unwrap();
        assert!(
            odd.estimate(&family("x", 1), 1)
                .unwrap_err()
                .contains("outside 0..1")
        );
    }

    #[test]
    fn regression_suite_rejects_bad_rewrites() {
        let base = Policy::builtin();
        let problems = |source: &str| check(&Policy::compile(source).unwrap(), &base, &[]);

        let always_large = "fn estimate(c, l) { 0.9 }
            fn select(l, cs) { let b = 0; for i in 0..cs.len() { if cs[i].tier > cs[b].tier { b = i; } } #{ index: b, reason: \"big\" } }";
        let found = problems(always_large);
        assert!(
            found.iter().any(|p| p.contains("trivial task")),
            "{found:?}"
        );
        assert!(found.iter().any(|p| p.contains("costlier")), "{found:?}");

        let ignores_evidence = "fn estimate(c, l) { 0.5 } fn select(l, cs) { 0 }";
        let found = problems(ignores_evidence);
        assert!(
            found.iter().any(|p| p.contains("10 bad ratings")),
            "{found:?}"
        );

        let crashes = "fn estimate(c, l) { 0.5 } fn select(l, cs) { cs[99].tier }";
        assert!(
            problems(crashes)
                .iter()
                .any(|p| p.contains("select failed"))
        );

        let missing = "fn select(l, cs) { 0 }";
        assert!(
            problems(missing)
                .iter()
                .any(|p| p.contains("estimate failed"))
        );

        let long = format!("{REWRITE}\n// {}", "x".repeat(MAX_SOURCE_BYTES));
        assert!(problems(&long).iter().any(|p| p.contains("the limit is")));
    }

    #[test]
    fn regression_suite_replays_recorded_outcomes() {
        // History: the small model keeps failing level 3; the medium one works.
        let history = vec![
            with_evidence(family("small", 1), 3, 0.0, 6.0),
            with_evidence(family("medium", 2), 3, 6.0, 0.0),
        ];
        // Passes every fixed check, but insists on the first candidate at level 3.
        let stubborn = REWRITE.replace(
            "fn select(level, candidates) {",
            "fn select(level, candidates) {\n    if level == 3 && candidates.len() == 2 && candidates[0].bad[2] == 6.0 { return #{ index: 0, reason: \"stubborn\" }; }",
        );
        let found = check(
            &Policy::compile(&stubborn).unwrap(),
            &Policy::builtin(),
            &history,
        );
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("on recorded outcomes"), "{found:?}");
        assert!(
            check(
                &Policy::compile(REWRITE).unwrap(),
                &Policy::builtin(),
                &history
            )
            .is_empty()
        );
    }

    #[test]
    fn a_passing_rewrite_is_adopted_and_a_failing_one_is_remembered() {
        let mut store = PolicyStore::in_memory();
        assert_eq!(store.version(), 0);
        let rejected = store
            .propose("fn select(l, c) { 0 }", "shorter", &[])
            .unwrap_err();
        assert!(!rejected.is_empty());
        assert_eq!(store.version(), 0);
        assert_eq!(store.rejected().len(), 1);
        assert!(
            store
                .propose("this is not rhai (", "broken", &[])
                .unwrap_err()[0]
                .contains("compile")
        );
        assert_eq!(
            store.propose(DEFAULT_SOURCE, "same", &[]).unwrap_err()[0],
            "identical to the current policy"
        );

        assert_eq!(store.propose(REWRITE, "simpler threshold", &[]), Ok(1));
        assert_eq!(store.version(), 1);
        // The new rule is in force: 0.65 at level 4 now clears the bar outright.
        let (i, reason, notes) = store.select(4, &tiers());
        assert_eq!((i, reason.as_str()), (1, "lowest tier at 60%"));
        assert!(notes.is_empty());
    }

    #[test]
    fn bad_ratings_during_probation_roll_a_rewrite_back() {
        let mut store = PolicyStore::in_memory();
        store.propose(REWRITE, "simpler threshold", &[]).unwrap();
        assert!(store.record_feedback(true).is_none());
        assert!(store.record_feedback(false).is_none());
        let note = store.record_feedback(false).unwrap();
        assert!(
            note.contains("policy v1 rolled back to the built-in policy"),
            "{note}"
        );
        assert!(note.contains("2 bad and 1 good"), "{note}");
        assert_eq!(store.version(), 0);
        assert_eq!(store.versions()[0].status, Status::RolledBack);
        // The built-in policy is never rolled back, whatever the ratings.
        assert!(store.record_feedback(false).is_none());
        assert!(store.record_feedback(false).is_none());
    }

    #[test]
    fn a_rewrite_that_survives_probation_stays() {
        let mut store = PolicyStore::in_memory();
        store.propose(REWRITE, "simpler threshold", &[]).unwrap();
        for _ in 0..PROBATION {
            assert!(store.record_feedback(true).is_none());
        }
        // Past probation, later bad ratings do not roll it back by themselves.
        for _ in 0..PROBATION + 2 {
            assert!(store.record_feedback(false).is_none());
        }
        assert_eq!(store.version(), 1);
    }

    #[test]
    fn versions_persist_and_a_broken_file_rolls_back_on_load() {
        let dir =
            std::env::temp_dir().join(format!("conscious-policy-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (mut store, notes) = PolicyStore::load(&dir);
        assert!(notes.is_empty());
        assert_eq!(store.propose(REWRITE, "first", &[]), Ok(1));
        let second = REWRITE.replace("0.6", "0.7").replace("60%", "70%");
        assert_eq!(store.propose(&second, "second", &[]), Ok(2));

        let (again, notes) = PolicyStore::load(&dir);
        assert!(notes.is_empty());
        assert_eq!(again.version(), 2);
        assert!(again.source().contains("70%"));

        // Version 2's file is damaged: loading falls back to version 1.
        std::fs::write(dir.join("v0002.rhai"), "fn select( {").unwrap();
        let (recovered, notes) = PolicyStore::load(&dir);
        assert_eq!(recovered.version(), 1);
        assert!(
            notes[0].contains("policy v2 rolled back to v1"),
            "{notes:?}"
        );
        assert!(recovered.source().contains("60%"));

        // And with version 1 gone too, to the built-in policy.
        std::fs::remove_file(dir.join("v0001.rhai")).unwrap();
        let (recovered, _) = PolicyStore::load(&dir);
        assert_eq!(recovered.version(), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_version_that_breaks_at_run_time_is_rolled_back_on_the_spot() {
        let mut store = PolicyStore::in_memory();
        store.propose(REWRITE, "fine", &[]).unwrap();
        // Simulate a script that passed the suite but fails on real input.
        store.policy =
            Policy::compile("fn estimate(c, l) { 0.5 } fn select(l, c) { c[42].tier }").unwrap();
        let (i, reason, notes) = store.select(1, &tiers());
        assert_eq!(store.version(), 0);
        assert_eq!(
            (i, reason.as_str()),
            (0, "cheapest model expected to handle it")
        );
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("policy v1 rolled back"), "{notes:?}");
    }

    #[test]
    fn reviews_come_due_after_a_bad_rating_or_several_ratings() {
        let mut store = PolicyStore::in_memory();
        for _ in 0..REVIEW_EVERY - 1 {
            store.record_feedback(true);
            assert!(!store.review_due());
        }
        store.record_feedback(true);
        assert!(store.review_due());
        store.review_done();
        assert!(!store.review_due());
        store.record_feedback(false);
        assert!(store.review_due());
    }

    #[test]
    fn review_replies_are_parsed() {
        assert_eq!(parse_review("KEEP"), Review::Keep);
        assert_eq!(parse_review("I would keep it as is."), Review::Keep);
        assert_eq!(parse_review("not sure"), Review::Unreadable);
        assert_eq!(
            parse_review(
                "WHY: level 3 keeps failing on small\n```rhai\nfn select(l, c) { 0 }\n```\nthanks"
            ),
            Review::Rewrite {
                why: "level 3 keeps failing on small".into(),
                source: "fn select(l, c) { 0 }".into(),
            }
        );
        assert_eq!(parse_review("```rhai\nfn a() {}"), Review::Unreadable);
    }

    #[test]
    fn review_prompt_shows_the_script_the_evidence_and_past_failures() {
        let mut store = PolicyStore::in_memory();
        store
            .propose("fn select(l, c) { 0 }", "too short", &[])
            .unwrap_err();
        let history = vec![with_evidence(family("small", 1), 3, 0.5, 2.0)];
        let p = review_prompt(&store, &history, "- small at complexity 3: bad\n");
        assert!(p.contains("Current policy (version 0)"));
        assert!(p.contains("fn select(level, candidates)"));
        assert!(p.contains("- small (tier 1, 0 calls): 0/0  0/0  0.5/2  0/0  0/0"));
        assert!(p.contains("- small at complexity 3: bad"));
        assert!(p.contains("\"too short\" was rejected"));
    }
}
