//! Per-task model routing.
//!
//! The user names one model: the core. Everything else is decided per task:
//!
//! 1. The core rates the task's complexity, 1 (trivial) to 5 (expert).
//! 2. The policy script (`policy`), interpreted at run time, reads the
//!    recorded evidence (`learn`) and picks a model family for that level.
//! 3. Feedback on the answer adds to the evidence, on disk.
//! 4. Now and then the core reviews the policy and may rewrite the script.
//!    A rewrite is adopted only if it passes the regression suite, and is
//!    rolled back if it then misbehaves; nobody is asked to approve it.
//!
//! Only the core is fixed. The pool is whatever the backends on this machine
//! offer at startup (`provider::available_models`): the latest model of each
//! family, never a superseded one, and no model version is named in this
//! code. A model that fails is taken out of the pool for a while. Anything
//! that goes wrong lands back on the core. `models.json` (`~/.config/conscious/`
//! or `CONSCIOUS_MODELS_FILE`) replaces the discovered pool when present.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::learn::{self, Knowledge, LEVELS};
use crate::policy::{self, Facts, PolicyStore, Review};
use crate::provider::{self, Message, ModelInfo, Provider, ProviderError, Request, Response};

/// How much of a task the core is shown when rating it. Enough to judge
/// difficulty without paying to send a whole document twice.
const TASK_PREVIEW_CHARS: usize = 2_000;

/// How long a model that failed stays out of the pool. Long enough not to
/// keep hitting a rate or usage limit, short enough to recover by itself.
const COOLDOWN: Duration = Duration::from_secs(600);

const RATE_SYSTEM: &str = "You rate how hard a task is for a language model, so it can be sent to a suitably sized one. Do not answer the task. Scale:\n\
1 trivial: lookup, arithmetic, reformatting\n\
2 simple: short factual answer, small edit, short summary\n\
3 moderate: everyday coding, explanation, analysis\n\
4 hard: multi-step reasoning, design, subtle debugging\n\
5 expert: novel, long or high-stakes reasoning\n\
Reply with exactly two lines:\nCOMPLEXITY: <1-5>\nWHY: <one short sentence>";

const TIER_SYSTEM: &str = "You sort language models into size tiers from their names and descriptions. 1 = small, fast, cheap. 2 = balanced general-purpose. 3 = largest, most capable, most expensive. Reply with one line per model and nothing else, in the form `<model number>: <tier>`.";

pub struct Candidate {
    /// `backend[:model]`, as accepted by `provider::from_spec`.
    pub spec: String,
    /// What routing knowledge is keyed by; stable across model versions.
    pub family: String,
    /// The backend's description, shown when the core sorts models into tiers.
    pub good_for: String,
    /// Size tier if the backend states it (1 small, 2 medium, 3 large).
    pub tier: Option<u8>,
    pub provider: Arc<dyn Provider>,
}

pub struct Choice {
    /// Index into `Router::candidates`; 0 is the core.
    pub index: usize,
    /// The rated complexity, when the task was rated.
    pub complexity: Option<u8>,
    pub reason: String,
    /// The core calls made to decide, so their cost is visible.
    pub calls: Vec<Response>,
    /// Things that happened while deciding that the user should see, such as
    /// a policy version being rolled back.
    pub notes: Vec<String>,
}

/// What a policy review led to.
pub struct ReviewOutcome {
    /// The core call, if it was made.
    pub call: Option<Response>,
    /// The new version number, if a rewrite was adopted.
    pub adopted: Option<u32>,
    pub message: String,
}

pub struct Router {
    /// `[0]` is always the core model.
    candidates: Vec<Candidate>,
    /// Per candidate: when a failed model may be offered again.
    down_until: Mutex<Vec<Option<Instant>>>,
    knowledge: Mutex<Knowledge>,
    policy: Mutex<PolicyStore>,
}

#[derive(Deserialize)]
struct FileEntry {
    spec: String,
    #[serde(default)]
    good_for: String,
    #[serde(default)]
    tier: Option<u8>,
}

impl Router {
    /// No routing: every task goes to the core.
    pub fn single(core: Arc<dyn Provider>) -> Self {
        Self::with(core, Vec::new(), Knowledge::in_memory())
    }

    /// The core plus an explicit pool. Entries equal to the core are dropped.
    pub fn with(core: Arc<dyn Provider>, pool: Vec<Candidate>, knowledge: Knowledge) -> Self {
        Self::build(core, pool, knowledge, PolicyStore::in_memory())
    }

    fn build(
        core: Arc<dyn Provider>,
        pool: Vec<Candidate>,
        mut knowledge: Knowledge,
        policy: PolicyStore,
    ) -> Self {
        let spec = core.label();
        // If the pool knows the core, reuse what it says about it.
        let known = pool.iter().find(|c| c.spec == spec);
        let mut candidates = vec![Candidate {
            family: known
                .map(|c| c.family.clone())
                .unwrap_or_else(|| spec.clone()),
            good_for: known.map(|c| c.good_for.clone()).unwrap_or_default(),
            // The user picked the core to be trusted with anything.
            tier: known.and_then(|c| c.tier).or(Some(3)),
            spec,
            provider: core,
        }];
        for c in pool {
            if !candidates.iter().any(|k| k.spec == c.spec) {
                candidates.push(c);
            }
        }
        for c in &candidates {
            if let (Some(tier), None) = (c.tier, knowledge.tier(&c.family)) {
                knowledge.set_tier(&c.family, tier);
            }
        }
        // Models that failed recently, in this run or an earlier one, start
        // out of the pool.
        let down_until = candidates
            .iter()
            .map(|c| match knowledge.down_for(&c.family) {
                0 => None,
                seconds => Some(Instant::now() + Duration::from_secs(seconds)),
            })
            .collect();
        Self {
            down_until: Mutex::new(down_until),
            candidates,
            knowledge: Mutex::new(knowledge),
            policy: Mutex::new(policy),
        }
    }

    /// The core plus every model found on this machine, with the routing
    /// knowledge learned so far. The second value lists anything that was
    /// skipped and why.
    pub fn discover(core: Arc<dyn Provider>) -> (Self, Vec<String>) {
        let mut notes = Vec::new();
        let from_file = match models_file() {
            Some(path) if path.exists() => match read_file(&path) {
                Ok(entries) => Some(entries),
                Err(e) => {
                    notes.push(format!("{}: {e}; using discovered models", path.display()));
                    None
                }
            },
            _ => None,
        };
        let entries = from_file.unwrap_or_else(|| {
            let (models, problems) = provider::available_models();
            notes.extend(problems);
            models
        });
        let mut pool = Vec::new();
        for e in entries {
            match provider::from_spec(&e.spec) {
                Ok(p) => {
                    let p: Arc<dyn Provider> = Arc::from(p);
                    pool.push(Candidate {
                        spec: p.label(),
                        family: e.family,
                        good_for: e.description,
                        tier: e.tier,
                        provider: p,
                    });
                }
                Err(err) => notes.push(format!("skipped `{}`: {err}", e.spec)),
            }
        }
        let knowledge = match learn::default_path() {
            Some(path) => {
                let (k, note) = Knowledge::load(&path);
                notes.extend(note);
                k
            }
            None => Knowledge::in_memory(),
        };
        let policy = match knowledge.policy_dir() {
            Some(dir) => {
                let (store, policy_notes) = PolicyStore::load(&dir);
                notes.extend(policy_notes);
                store
            }
            None => PolicyStore::in_memory(),
        };
        (Self::build(core, pool, knowledge, policy), notes)
    }

    pub fn core(&self) -> Arc<dyn Provider> {
        self.candidates[0].provider.clone()
    }

    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// The `models` tool: lets the agent report on the pool and carry out the
    /// user's wishes about which models to use. `action` is `list`,
    /// `disable` or `enable`. Returns what to tell the model, and whether it
    /// worked.
    pub fn manage_models(&self, action: &str, name: &str, days: Option<u64>) -> (bool, String) {
        let name = name.trim();
        let matching: Vec<&str> = self
            .candidates
            .iter()
            .map(|c| c.spec.as_str())
            .filter(|spec| !name.is_empty() && spec.to_lowercase().contains(&name.to_lowercase()))
            .collect();
        let outcome = match action {
            "list" => (true, String::new()),
            "disable" if name.is_empty() => (false, "say which model to stop using".into()),
            // Nothing to take out now, but the wish stands: keep it out if
            // it turns up in a later session.
            "disable" if matching.is_empty() => {
                let mut k = self.knowledge.lock().unwrap();
                k.ban(name, days);
                let _ = k.save();
                (
                    true,
                    format!(
                        "No model with `{name}` in its name is in the pool right now, so nothing \
                         changed today. It is recorded, and such a model will be kept out if it \
                         appears."
                    ),
                )
            }
            "disable" => {
                let mut k = self.knowledge.lock().unwrap();
                k.ban(name, days);
                let _ = k.save();
                let core = &self.candidates[0].spec;
                let mut said = format!(
                    "Stopped using {} {}.",
                    matching.join(", "),
                    match days {
                        Some(d) => format!("for {d} day{}", if d == 1 { "" } else { "s" }),
                        None => "until told otherwise".into(),
                    }
                );
                if matching.contains(&core.as_str()) {
                    said.push_str(&format!(
                        " Note: {core} is the core model and stays in use; it is chosen when \
                         the harness starts."
                    ));
                }
                (true, said)
            }
            "enable" => {
                let mut k = self.knowledge.lock().unwrap();
                let lifted = k.allow(name);
                let _ = k.save();
                if lifted == 0 {
                    (false, format!("nothing matching `{name}` was switched off"))
                } else {
                    (true, format!("Models matching `{name}` may be used again."))
                }
            }
            other => (
                false,
                format!("unknown action `{other}`: use list, disable or enable"),
            ),
        };
        if !outcome.0 {
            return outcome;
        }
        // Always end with the state of the pool, so the model can report it.
        let open = self.available();
        let k = self.knowledge.lock().unwrap();
        let mut report = outcome.1;
        report.push_str("\nModel pool now:\n");
        for (i, c) in self.candidates.iter().enumerate() {
            let state = if i == 0 {
                "core, always used"
            } else if k.is_banned(&c.spec) {
                "switched off by the user"
            } else if !open.contains(&i) {
                "resting after a failure"
            } else {
                "in use"
            };
            report.push_str(&format!("- {} ({state})\n", c.spec));
        }
        for (name, days) in k.bans() {
            report.push_str(&match days {
                Some(d) => format!(
                    "`{name}` is switched off for {d} more day{}.\n",
                    if d == 1 { "" } else { "s" }
                ),
                None => format!("`{name}` is switched off until told otherwise.\n"),
            });
        }
        (true, report.trim().to_string())
    }

    /// The smallest model available right now, for cheap housekeeping calls
    /// such as working out what a memory is. Falls back to the core.
    pub fn cheapest(&self) -> Arc<dyn Provider> {
        // `available` takes the knowledge lock itself, so ask it first.
        let open = self.available();
        let k = self.knowledge.lock().unwrap();
        let pick = open
            .into_iter()
            .min_by_key(|&i| k.tier(&self.candidates[i].family).unwrap_or(2))
            .unwrap_or(0);
        self.candidates[pick].provider.clone()
    }

    /// Whether there is anything to choose between.
    pub fn routes(&self) -> bool {
        self.candidates.len() > 1
    }

    /// What is known about a candidate: its tier and the learned chance of
    /// success at each complexity level.
    pub fn profile(&self, index: usize) -> String {
        let k = self.knowledge.lock().unwrap();
        let policy = self.policy.lock().unwrap();
        let family = &self.candidates[index].family;
        let tier = match k.tier(family) {
            Some(t) => format!("tier {t}"),
            None => "tier not yet rated".into(),
        };
        let seen = match k.stats(family) {
            Some(s) if s.calls > 0 => format!(
                " · {} call{}, avg {:.1}s",
                s.calls,
                if s.calls == 1 { "" } else { "s" },
                s.latency_s.unwrap_or(0.0)
            ),
            _ => String::new(),
        };
        let facts = k.facts(family);
        let estimates: Vec<String> = (1..=LEVELS as u8)
            .map(|level| format!("c{level} {:.0}%", policy.estimate(&facts, level) * 100.0))
            .collect();
        format!("{tier} · {}{seen}", estimates.join("  "))
    }

    /// One line on the policy in use.
    pub fn policy_line(&self) -> String {
        let policy = self.policy.lock().unwrap();
        let rewrites = policy.versions().len();
        format!(
            "v{}{}, interpreted script; {} rewrite{} tried so far. Rate answers with + or - to teach it.",
            policy.version(),
            if policy.version() == 0 {
                " (built in)"
            } else {
                ""
            },
            rewrites,
            if rewrites == 1 { "" } else { "s" }
        )
    }

    /// The policy in use, for display: its version and history.
    pub fn policy_summary(&self) -> String {
        policy::summary(&self.policy.lock().unwrap())
    }

    /// Indices of the models that can be offered right now. The core is
    /// always included: it is the fallback, so it is never taken out.
    fn available(&self) -> Vec<usize> {
        let now = Instant::now();
        let down = self.down_until.lock().unwrap();
        let k = self.knowledge.lock().unwrap();
        (0..self.candidates.len())
            .filter(|&i| {
                i == 0
                    || (down[i].is_none_or(|until| now >= until)
                        && !k.is_banned(&self.candidates[i].spec))
            })
            .collect()
    }

    /// Whether a model is currently out of the pool after failing.
    #[cfg(test)]
    pub fn is_down(&self, index: usize) -> bool {
        !self.available().contains(&index)
    }

    fn ask_core(&self, system: &str, prompt: String) -> Result<Response, ProviderError> {
        let req = Request {
            system: system.into(),
            messages: vec![Message::user(prompt)],
            ..Default::default()
        };
        self.candidates[0].provider.complete(&req, &mut |_| {})
    }

    /// Has the core sort any family of unknown size into a tier, once. The
    /// answer is kept in knowledge, so this costs a call only when a backend
    /// offers a family that has not been seen before.
    fn rate_new_families(&self, open: &[usize], calls: &mut Vec<Response>) {
        let unknown: Vec<usize> = {
            let k = self.knowledge.lock().unwrap();
            open.iter()
                .copied()
                .filter(|&i| k.tier(&self.candidates[i].family).is_none())
                .collect()
        };
        if unknown.is_empty() {
            return;
        }
        let mut prompt = String::from("Models:\n");
        for (n, &i) in unknown.iter().enumerate() {
            let c = &self.candidates[i];
            prompt.push_str(&format!("{}. {}: {}\n", n + 1, c.spec, c.good_for));
        }
        // If the call fails the tiers stay unknown (treated as medium) and
        // are asked for again next time.
        let Ok(resp) = self.ask_core(TIER_SYSTEM, prompt) else {
            return;
        };
        let tiers = parse_tiers(&resp.text, unknown.len());
        let mut k = self.knowledge.lock().unwrap();
        for (n, &i) in unknown.iter().enumerate() {
            // A model the core did not place is taken to be medium.
            k.set_tier(&self.candidates[i].family, tiers[n].unwrap_or(2));
        }
        let _ = k.save();
        calls.push(resp);
    }

    /// Decides which model should run `task`. Never fails: any problem
    /// resolves to the core, with the reason saying why.
    pub fn choose(&self, task: &str) -> Choice {
        let open = self.available();
        let core = |reason: String, calls: Vec<Response>| Choice {
            index: 0,
            complexity: None,
            reason,
            calls,
            notes: Vec::new(),
        };
        if open.len() < 2 {
            return core(
                if self.routes() {
                    "no other model is available right now".into()
                } else {
                    "only one model available".into()
                },
                Vec::new(),
            );
        }
        let mut calls = Vec::new();
        self.rate_new_families(&open, &mut calls);

        let resp = match self.ask_core(RATE_SYSTEM, task_prompt(task)) {
            Ok(resp) => resp,
            Err(e) => {
                return core(
                    format!("could not rate the task ({e}); using the core"),
                    calls,
                );
            }
        };
        let rated = parse_tagged(&resp.text, "COMPLEXITY", LEVELS);
        calls.push(resp);
        let Some((level, why)) = rated else {
            return core(
                "the task rating was unreadable; using the core".into(),
                calls,
            );
        };
        let level = level as u8 + 1;

        let facts: Vec<Facts> = {
            let k = self.knowledge.lock().unwrap();
            open.iter()
                .map(|&i| k.facts(&self.candidates[i].family))
                .collect()
        };
        let mut policy = self.policy.lock().unwrap();
        let (pick, how, notes) = policy.select(level, &facts);
        let l = level as usize - 1;
        let seen = facts[pick].good[l] + facts[pick].bad[l];
        Choice {
            index: open[pick],
            complexity: Some(level),
            reason: format!(
                "complexity {level} ({why}); {how} (est. {:.0}% from {} rated, policy v{})",
                policy.estimate(&facts[pick], level) * 100.0,
                if seen.fract() == 0.0 {
                    format!("{seen:.0}")
                } else {
                    format!("{seen:.1}")
                },
                policy.version()
            ),
            calls,
            notes,
        }
    }

    /// Runs `req` on the chosen candidate. If that model fails and it was not
    /// the core, it is taken out of the pool for `COOLDOWN` and the core gets
    /// the request instead. Returns the index of the model that produced the
    /// result, and a note when a fallback happened.
    ///
    /// A failure here is about availability (limits, outages), so it is not
    /// counted as evidence about what the model is capable of.
    pub fn complete(
        &self,
        choice: &Choice,
        req: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> (usize, Option<String>, Result<Response, ProviderError>) {
        let index = choice.index;
        let chosen = &self.candidates[index];
        let (used, note, result) = match chosen.provider.complete(req, on_text) {
            Err(e) if index != 0 => {
                self.down_until.lock().unwrap()[index] = Some(Instant::now() + COOLDOWN);
                {
                    let mut k = self.knowledge.lock().unwrap();
                    k.mark_down(&chosen.family, COOLDOWN.as_secs());
                    let _ = k.save();
                }
                let note = format!(
                    "{} failed ({e}); falling back to {} and not offering it for {} min",
                    chosen.spec,
                    self.candidates[0].spec,
                    COOLDOWN.as_secs() / 60
                );
                (
                    0,
                    Some(note),
                    self.candidates[0].provider.complete(req, on_text),
                )
            }
            result => (index, None, result),
        };
        if let (Ok(resp), Some(level)) = (&result, choice.complexity) {
            let mut k = self.knowledge.lock().unwrap();
            k.observe(
                &self.candidates[used].family,
                level,
                resp.latency.as_secs_f64(),
            );
            let _ = k.save();
        }
        (used, note, result)
    }

    /// Applies the user's verdict on the last routed answer and saves what
    /// was learned. `Ok` lists what changed; `Err` says why nothing did.
    pub fn feedback(&self, good: bool) -> Result<Vec<String>, String> {
        let mut k = self.knowledge.lock().unwrap();
        let mut policy = self.policy.lock().unwrap();
        apply_feedback(&mut k, &mut policy, good)
    }

    /// The model family a choice sends a task to and the task's rated
    /// complexity: what a rating of that task is evidence about. `None`
    /// when the task was not rated, as with a single model.
    pub fn decision(&self, choice: &Choice) -> Option<(String, u8)> {
        Some((
            self.candidates[choice.index].family.clone(),
            choice.complexity?,
        ))
    }

    /// Applies the user's verdict on one particular run (`decision`), which
    /// need not be the most recent one.
    pub fn rate(&self, family: &str, level: u8, good: bool) -> Result<Vec<String>, String> {
        let mut k = self.knowledge.lock().unwrap();
        let mut policy = self.policy.lock().unwrap();
        k.point_at(family, level);
        apply_feedback(&mut k, &mut policy, good)
    }

    /// Whether the core should be asked to review the policy.
    pub fn review_due(&self) -> bool {
        self.routes() && self.policy.lock().unwrap().review_due()
    }

    /// Asks the core whether the policy script should change, and if it
    /// offers a rewrite, runs it through the regression suite and adopts it
    /// if it passes. Makes one core call; nothing is held locked meanwhile.
    pub fn review(&self) -> ReviewOutcome {
        let prompt = {
            let k = self.knowledge.lock().unwrap();
            let policy = self.policy.lock().unwrap();
            policy::review_prompt(&policy, &k.rated_facts(), &k.recent_ratings())
        };
        let resp = match self.ask_core(policy::REVIEW_SYSTEM, prompt) {
            Ok(resp) => resp,
            // Left due, so it is tried again after the next rating.
            Err(e) => {
                return ReviewOutcome {
                    call: None,
                    adopted: None,
                    message: format!("policy review failed ({e}); it will be retried"),
                };
            }
        };
        let history = self.knowledge.lock().unwrap().rated_facts();
        let mut policy = self.policy.lock().unwrap();
        let (adopted, message) = match policy::parse_review(&resp.text) {
            Review::Keep => (None, format!("policy v{} reviewed: kept", policy.version())),
            Review::Unreadable => (
                None,
                format!(
                    "policy review reply was unreadable; keeping v{}",
                    policy.version()
                ),
            ),
            Review::Rewrite { why, source } => match policy.propose(&source, &why, &history) {
                Ok(version) => (
                    Some(version),
                    format!("policy v{version} adopted after passing regression: {why}"),
                ),
                Err(problems) => (
                    None,
                    format!(
                        "policy rewrite rejected by regression, keeping v{} ({why}): {}",
                        policy.version(),
                        problems.join("; ")
                    ),
                ),
            },
        };
        policy.review_done();
        ReviewOutcome {
            call: Some(resp),
            adopted,
            message,
        }
    }
}

/// Counts a rating as evidence, lets the policy store judge the version on
/// probation, and saves. Shared by the router and the `feedback` command.
pub fn apply_feedback(
    k: &mut Knowledge,
    policy: &mut PolicyStore,
    good: bool,
) -> Result<Vec<String>, String> {
    let before = k
        .pending()
        .map(|d| policy.estimate(&k.facts(&d.family), d.complexity));
    let rated = k.feedback(good)?;
    let after = policy.estimate(&k.facts(&rated.family), rated.complexity);
    let mut learned = vec![format!(
        "{} at complexity {}: {} → estimate {:.0}% → {:.0}%",
        rated.family,
        rated.complexity,
        if good { "good" } else { "bad" },
        before.unwrap_or(after) * 100.0,
        after * 100.0
    )];
    learned.extend(policy.record_feedback(good));
    if let Err(e) = k.save() {
        learned.push(format!("not saved: {e}"));
    }
    Ok(learned)
}

fn task_prompt(task: &str) -> String {
    let mut s = String::from("Task:\n");
    s.extend(task.chars().take(TASK_PREVIEW_CHARS));
    if task.chars().count() > TASK_PREVIEW_CHARS {
        s.push_str("\n[task truncated]");
    }
    s
}

fn first_number(s: &str) -> Option<usize> {
    let digits: String = s
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Reads `<TAG>: <n>` / `WHY: <text>` leniently. Small models wander from the
/// format, so a bare number anywhere in the reply is accepted too. `n` must
/// be in `1..=count`; the result is zero-based.
fn parse_tagged(reply: &str, tag: &str, count: usize) -> Option<(usize, String)> {
    let tagged = reply
        .lines()
        .find(|l| l.to_ascii_uppercase().contains(tag))
        .and_then(first_number);
    let n = tagged.or_else(|| first_number(reply))?;
    if n == 0 || n > count {
        return None;
    }
    let why = reply
        .lines()
        .find_map(|l| {
            let upper = l.to_ascii_uppercase();
            let at = upper.find("WHY")?;
            Some(
                l[at + 3..]
                    .trim_start_matches([':', ' ', '-', '*'])
                    .trim()
                    .to_string(),
            )
        })
        .filter(|w| !w.is_empty())
        .unwrap_or_else(|| "no reason given".into());
    Some((n - 1, why.chars().take(200).collect()))
}

/// Reads `<model number>: <tier>` lines. Returns one entry per model, `None`
/// where the reply did not place it.
fn parse_tiers(reply: &str, count: usize) -> Vec<Option<u8>> {
    let mut tiers = vec![None; count];
    for line in reply.lines() {
        let mut numbers = line
            .split(|c: char| !c.is_ascii_digit())
            .filter(|t| !t.is_empty())
            .filter_map(|t| t.parse::<usize>().ok());
        if let (Some(n), Some(tier)) = (numbers.next(), numbers.next())
            && (1..=count).contains(&n)
            && (1..=3).contains(&tier)
        {
            tiers[n - 1] = Some(tier as u8);
        }
    }
    tiers
}

fn models_file() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("CONSCIOUS_MODELS_FILE") {
        return Some(PathBuf::from(p));
    }
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(config.join("conscious").join("models.json"))
}

fn read_file(path: &Path) -> Result<Vec<ModelInfo>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let entries: Vec<FileEntry> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    Ok(entries
        .into_iter()
        .map(|e| ModelInfo {
            family: e.spec.clone(),
            spec: e.spec,
            description: e.good_for,
            tier: e.tier,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::testing::{Queue, Scripted};

    fn candidate(spec: &'static str, tier: u8, reply: &'static str) -> Candidate {
        Candidate {
            spec: spec.into(),
            family: spec.into(),
            good_for: format!("{spec} things"),
            tier: Some(tier),
            provider: Arc::new(Scripted::ok(spec, reply)),
        }
    }

    fn failing(spec: &'static str, tier: u8) -> Candidate {
        Candidate {
            provider: Arc::new(Scripted::err(spec, "usage limit")),
            ..candidate(spec, tier, "")
        }
    }

    /// Core (tier 3) plus a small and a medium model.
    fn router(core_reply: &'static str) -> Router {
        Router::with(
            Arc::new(Scripted::ok("core", core_reply)),
            vec![candidate("small", 1, "s"), candidate("medium", 2, "m")],
            Knowledge::in_memory(),
        )
    }

    fn spec(r: &Router, c: &Choice) -> String {
        r.candidates()[c.index].spec.clone()
    }

    #[test]
    fn tagged_replies_are_parsed_leniently() {
        assert_eq!(
            parse_tagged("COMPLEXITY: 2\nWHY: simple lookup", "COMPLEXITY", 5),
            Some((1, "simple lookup".into()))
        );
        assert_eq!(
            parse_tagged(
                "**Complexity:** 3\n**Why** - needs planning",
                "COMPLEXITY",
                5
            ),
            Some((2, "needs planning".into()))
        );
        assert_eq!(
            parse_tagged("I'd say 1.", "COMPLEXITY", 5),
            Some((0, "no reason given".into()))
        );
        assert_eq!(parse_tagged("COMPLEXITY: 7\nWHY: x", "COMPLEXITY", 5), None);
        assert_eq!(parse_tagged("COMPLEXITY: 0", "COMPLEXITY", 5), None);
        assert_eq!(parse_tagged("no idea", "COMPLEXITY", 5), None);
    }

    #[test]
    fn tier_lines_are_parsed_and_gaps_left_unplaced() {
        assert_eq!(
            parse_tiers("1: 3\n2 - tier 1\n4: 9\nnonsense", 4),
            [Some(3), Some(1), None, None]
        );
    }

    #[test]
    fn complexity_decides_the_model() {
        let pick = |reply: &'static str| {
            let r = router(reply);
            let c = r.choose("task");
            (spec(&r, &c), c.complexity)
        };
        assert_eq!(
            pick("COMPLEXITY: 1\nWHY: trivial"),
            ("small".into(), Some(1))
        );
        assert_eq!(
            pick("COMPLEXITY: 3\nWHY: moderate"),
            ("medium".into(), Some(3))
        );
        assert_eq!(pick("COMPLEXITY: 5\nWHY: expert"), ("core".into(), Some(5)));
    }

    #[test]
    fn the_decision_explains_itself_and_reports_its_call() {
        let r = router("COMPLEXITY: 3\nWHY: everyday coding");
        let c = r.choose("write a parser");
        assert!(
            c.reason
                .starts_with("complexity 3 (everyday coding); cheapest model expected to handle it (est. 85% from 0 rated, policy v0)"),
            "{}",
            c.reason
        );
        assert_eq!(c.calls.len(), 1);
    }

    #[test]
    fn feedback_changes_which_model_gets_that_complexity() {
        let r = router("COMPLEXITY: 4\nWHY: hard");
        let req = Request::default();

        // Unproven, the medium model is tried on level 4 first.
        let c = r.choose("task");
        assert_eq!(spec(&r, &c), "medium");
        let _ = r.complete(&c, &req, &mut |_| {});
        let learned = r.feedback(false).unwrap();
        assert!(
            learned[0].contains("medium at complexity 4: bad"),
            "{learned:?}"
        );

        // After a bad verdict, level 4 goes to the stronger model.
        let c = r.choose("task");
        assert_eq!(spec(&r, &c), "core");
        // One answer cannot be rated twice.
        assert!(r.feedback(false).is_err());
    }

    #[test]
    fn unreadable_or_failed_rating_falls_back_to_the_core() {
        let c = router("the weather is nice").choose("task");
        assert_eq!((c.index, c.complexity), (0, None));
        assert!(c.reason.contains("unreadable"));

        let failing_core = Router::with(
            Arc::new(Scripted::err("core", "boom")),
            vec![candidate("small", 1, "s")],
            Knowledge::in_memory(),
        );
        let c = failing_core.choose("task");
        assert_eq!(c.index, 0);
        assert!(c.reason.contains("boom"));
        assert!(c.calls.is_empty());
    }

    #[test]
    fn a_failing_choice_falls_back_to_the_core_without_blaming_its_ability() {
        let r = Router::with(
            Arc::new(Scripted::ok("core", "COMPLEXITY: 1\nWHY: x")),
            vec![failing("flaky", 1)],
            Knowledge::in_memory(),
        );
        let req = Request::default();
        let c = r.choose("task");
        assert_eq!(spec(&r, &c), "flaky");
        let (used, note, result) = r.complete(&c, &req, &mut |_| {});
        assert_eq!(used, 0);
        assert!(note.unwrap().contains("flaky failed"));
        assert!(result.is_ok());
        assert!(r.is_down(1) && !r.is_down(0));

        // The outage is not held against the model: nothing was recorded for
        // it, and the task that feedback applies to is the core's.
        let k = r.knowledge.lock().unwrap();
        assert_eq!(k.stats("flaky").unwrap().calls, 0);
        assert_eq!(k.last().unwrap().family, "core");
    }

    #[test]
    fn a_failing_core_has_nothing_to_fall_back_to() {
        let dead = Router::single(Arc::new(Scripted::err("core", "down")));
        let c = dead.choose("task");
        let (_, note, result) = dead.complete(&c, &Request::default(), &mut |_| {});
        assert!(note.is_none() && result.is_err());
    }

    #[test]
    fn a_model_that_failed_is_not_offered_until_its_cooldown_ends() {
        let r = Router::with(
            Arc::new(Scripted::ok("core", "COMPLEXITY: 1\nWHY: x")),
            vec![failing("flaky", 1), candidate("steady", 1, "ok")],
            Knowledge::in_memory(),
        );
        let c = r.choose("t");
        assert_eq!(spec(&r, &c), "flaky");
        let _ = r.complete(&c, &Request::default(), &mut |_| {});

        // With `flaky` out, the other small model takes its place.
        assert_eq!(r.available(), [0, 2]);
        assert_eq!(spec(&r, &r.choose("t")), "steady");

        // Once the cooldown has passed it is offered again.
        r.down_until.lock().unwrap()[1] = Some(Instant::now() - Duration::from_secs(1));
        assert_eq!(r.available(), [0, 1, 2]);

        // If everything but the core is down, the core is not even asked.
        let mut down = r.down_until.lock().unwrap();
        down[1] = Some(Instant::now() + COOLDOWN);
        down[2] = Some(Instant::now() + COOLDOWN);
        drop(down);
        let c = r.choose("t");
        assert_eq!(c.index, 0);
        assert!(c.calls.is_empty());
        assert!(c.reason.contains("no other model is available"));
    }

    #[test]
    fn a_recent_failure_is_remembered_by_the_next_run() {
        let mut knowledge = Knowledge::in_memory();
        knowledge.mark_down("flaky", 600);
        let r = Router::with(
            Arc::new(Scripted::ok("core", "COMPLEXITY: 1\nWHY: x")),
            vec![failing("flaky", 1), candidate("steady", 1, "ok")],
            knowledge,
        );
        assert!(r.is_down(1));
        assert_eq!(spec(&r, &r.choose("t")), "steady");
    }

    #[test]
    fn the_user_can_have_a_model_taken_out_of_the_pool_and_put_back() {
        let r = Router::with(
            Arc::new(Scripted::ok("core", "COMPLEXITY: 1\nWHY: x")),
            vec![candidate("codex:fast", 1, "c"), candidate("steady", 1, "s")],
            Knowledge::in_memory(),
        );
        assert_eq!(spec(&r, &r.choose("t")), "codex:fast");

        let (ok, said) = r.manage_models("disable", "Codex", Some(3));
        assert!(ok, "{said}");
        assert!(
            said.starts_with("Stopped using codex:fast for 3 days."),
            "{said}"
        );
        assert!(
            said.contains("- codex:fast (switched off by the user)"),
            "{said}"
        );
        assert!(said.contains("- core (core, always used)"), "{said}");
        assert_eq!(spec(&r, &r.choose("t")), "steady");

        let (ok, said) = r.manage_models("enable", "codex", None);
        assert!(ok && said.contains("- codex:fast (in use)"), "{said}");
        assert_eq!(spec(&r, &r.choose("t")), "codex:fast");

        // A model that is not in the pool today is still recorded as unwanted.
        let (ok, said) = r.manage_models("disable", "llama", None);
        assert!(ok && said.contains("kept out if it appears"), "{said}");
        assert!(r.knowledge.lock().unwrap().is_banned("local:llama-3"));
        assert!(!r.manage_models("enable", "codex", None).0);
        assert!(
            r.manage_models("list", "", None)
                .1
                .starts_with("Model pool now:")
        );
        // The core cannot be switched off this way, and says so.
        let (_, said) = r.manage_models("disable", "core", None);
        assert!(
            said.contains("core is the core model and stays in use"),
            "{said}"
        );
    }

    #[test]
    fn a_single_model_is_never_asked_to_route_and_teaches_nothing() {
        // An error core proves no call was made.
        let r = Router::single(Arc::new(Scripted::err("core", "must not be called")));
        assert!(!r.routes());
        let c = r.choose("task");
        assert_eq!((c.index, c.complexity), (0, None));
        assert!(c.calls.is_empty());
        assert!(r.feedback(true).is_err());
    }

    #[test]
    fn families_of_unknown_size_are_tiered_by_the_core_once() {
        let unknown = |spec: &'static str| Candidate {
            tier: None,
            ..candidate(spec, 1, "x")
        };
        // First reply answers the tier question, the next two rate tasks.
        let core = Queue::new(
            "core",
            &[
                "1: 1\n2: 3",
                "COMPLEXITY: 1\nWHY: trivial",
                "COMPLEXITY: 1\nWHY: trivial",
            ],
        );
        let r = Router::with(
            Arc::new(core),
            vec![unknown("mystery-a"), unknown("mystery-b")],
            Knowledge::in_memory(),
        );
        let c = r.choose("task");
        assert_eq!(c.calls.len(), 2, "tier call + rating call");
        assert_eq!(spec(&r, &c), "mystery-a");
        assert!(r.profile(1).starts_with("tier 1"));
        assert!(r.profile(2).starts_with("tier 3"));

        // The tiers are remembered, so the second task costs one call.
        assert_eq!(r.choose("task").calls.len(), 1);
    }

    /// A rewrite that passes regression: lowest tier whose estimate is 60%.
    const REWRITE: &str = r#"WHY: a flat 60% bar is simpler
```rhai
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
```"#;

    fn reviewing(replies: &[&'static str]) -> Router {
        Router::with(
            Arc::new(Queue::new("core", replies)),
            vec![candidate("small", 1, "s"), candidate("medium", 2, "m")],
            Knowledge::in_memory(),
        )
    }

    #[test]
    fn a_bad_rating_makes_a_review_due_and_a_passing_rewrite_is_adopted() {
        let r = reviewing(&[
            "COMPLEXITY: 4\nWHY: hard",
            REWRITE,
            "COMPLEXITY: 4\nWHY: hard",
        ]);
        assert!(!r.review_due());
        let c = r.choose("task");
        assert!(c.reason.ends_with("policy v0)"), "{}", c.reason);
        let _ = r.complete(&c, &Request::default(), &mut |_| {});
        r.feedback(false).unwrap();
        assert!(r.review_due());

        let outcome = r.review();
        assert_eq!(outcome.adopted, Some(1));
        assert!(
            outcome.message.contains("policy v1 adopted"),
            "{}",
            outcome.message
        );
        assert!(outcome.message.contains("a flat 60% bar is simpler"));
        assert!(outcome.call.is_some());
        assert!(!r.review_due());

        // The rewritten script now makes the decisions.
        let c = r.choose("task");
        assert!(c.reason.contains("policy v1"), "{}", c.reason);
        assert!(r.policy_summary().contains("v1"));
    }

    #[test]
    fn a_rewrite_that_fails_regression_is_rejected_and_the_policy_kept() {
        let bad = "WHY: always use the biggest\n```rhai\nfn estimate(c, l) { 0.9 }\nfn select(l, cs) { let b = 0; for i in 0..cs.len() { if cs[i].tier > cs[b].tier { b = i; } } #{ index: b, reason: \"big\" } }\n```";
        let r = reviewing(&[bad]);
        let outcome = r.review();
        assert_eq!(outcome.adopted, None);
        assert!(
            outcome
                .message
                .contains("rejected by regression, keeping v0"),
            "{}",
            outcome.message
        );
        assert!(
            outcome.message.contains("trivial task"),
            "{}",
            outcome.message
        );
    }

    #[test]
    fn keep_unreadable_and_failed_reviews_change_nothing() {
        assert!(
            reviewing(&["KEEP"])
                .review()
                .message
                .contains("reviewed: kept")
        );
        assert!(reviewing(&["hmm"]).review().message.contains("unreadable"));
        let r = Router::with(
            Arc::new(Scripted::err("core", "down")),
            vec![candidate("small", 1, "s")],
            Knowledge::in_memory(),
        );
        let outcome = r.review();
        assert!(outcome.call.is_none());
        assert!(outcome.message.contains("will be retried"));
    }

    #[test]
    fn an_adopted_rewrite_that_draws_bad_ratings_is_rolled_back() {
        let r = reviewing(&[REWRITE, "COMPLEXITY: 3\nWHY: moderate"]);
        assert_eq!(r.review().adopted, Some(1));
        let req = Request::default();
        let mut notes = Vec::new();
        for _ in 0..2 {
            let c = r.choose("task");
            let _ = r.complete(&c, &req, &mut |_| {});
            notes = r.feedback(false).unwrap();
        }
        assert!(
            notes
                .iter()
                .any(|n| n.contains("policy v1 rolled back to the built-in policy")),
            "{notes:?}"
        );
        assert!(r.choose("task").reason.contains("policy v0"));
    }

    #[test]
    fn long_tasks_are_truncated_for_rating() {
        let p = task_prompt(&"x".repeat(TASK_PREVIEW_CHARS + 50));
        assert!(p.ends_with("[task truncated]"));
        assert_eq!(task_prompt("short"), "Task:\nshort");
    }

    #[test]
    fn the_core_is_not_listed_twice_and_keeps_its_tier() {
        let r = Router::with(
            Arc::new(Scripted::ok("small", "")),
            vec![candidate("small", 1, "s"), candidate("big", 3, "b")],
            Knowledge::in_memory(),
        );
        let specs: Vec<&str> = r.candidates().iter().map(|c| c.spec.as_str()).collect();
        assert_eq!(specs, ["small", "big"]);
        assert!(r.profile(0).starts_with("tier 1"));
    }

    #[test]
    fn models_file_entries_are_read() {
        let dir =
            std::env::temp_dir().join(format!("conscious-router-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("models.json");
        std::fs::write(
            &path,
            r#"[{"spec": "local:tiny", "good_for": "offline drafts", "tier": 1}, {"spec": "codex"}]"#,
        )
        .unwrap();
        let entries = read_file(&path).unwrap();
        assert_eq!(entries[0].spec, "local:tiny");
        assert_eq!(entries[0].description, "offline drafts");
        assert_eq!(entries[0].tier, Some(1));
        assert_eq!(entries[1].tier, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
