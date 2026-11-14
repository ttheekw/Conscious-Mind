//! The scheduler: tasks the agent (or the user, through it) has set to start
//! later, once or repeatedly.
//!
//! A job is a task in words plus a rule for when it is due. Everything here
//! is deterministic and takes the time as an argument: parsing a rule,
//! working out the next due time, choosing which job runs, and what happens
//! to runs that were missed while the harness was not running. No model is
//! involved in any of it; the model only asks for a job with the `schedule`
//! tool, and the guardrails decide whether it may.
//!
//! Jobs are kept in `schedule.json` beside the routing knowledge and survive
//! restarts. They fire while the background process (`daemon`) is running,
//! with or without a screen attached; a job that came due while it was not
//! is handled on the next start by its own missed-run rule.
//!
//! A job can also be due when something changes on disk (`When::Changed`).
//! Looking at the disk is the one piece of I/O here (`stamp`); deciding what
//! a changed stamp means is not, and takes the stamp as an argument.
//!
//! Hard limits, enforced here: `MAX_JOBS` active jobs, and a repeating job
//! no more often than `MIN_EVERY`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Jobs that may exist at once.
pub const MAX_JOBS: usize = 20;
/// The shortest interval a repeating job may have, in seconds.
pub const MIN_EVERY: u64 = 5 * 60;
/// The shortest delay for a one-off job, in seconds.
const MIN_DELAY: u64 = 10;
const DAY: u64 = 86_400;

/// When a job is due.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum When {
    /// One time, at this moment (seconds since the Unix epoch).
    Once { at: u64 },
    /// Again and again, this many seconds after each start.
    Every { secs: u64 },
    /// Every day at this local time, in minutes after midnight.
    Daily { minute: u32 },
    /// Whenever this file or folder changes. `seen` is its `stamp` when the
    /// job last looked.
    Changed { path: PathBuf, seen: u64 },
}

/// The due time of a job that is waiting for a change, not for a moment.
const NEVER: u64 = u64::MAX;
/// Entries looked at when stamping a folder; a larger folder is stamped by
/// its first this many, in name order.
const MAX_STAMPED: usize = 2_000;

/// What to do with a run that came due while the harness was not running.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Missed {
    /// Run it once on the next start, however many times it was missed.
    Run,
    /// Let it go; wait for the next time it is due.
    Skip,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: u32,
    /// The task, in the words it will be given to the agent.
    pub task: String,
    pub when: When,
    /// When it is next due.
    pub next: u64,
    pub paused: bool,
    pub missed: Missed,
    pub created: u64,
    /// How many times it has started.
    pub runs: u32,
    pub last_run: Option<u64>,
}

#[derive(Default, Serialize, Deserialize)]
struct Saved {
    next_id: u32,
    jobs: Vec<Job>,
}

pub struct Schedule {
    path: Option<PathBuf>,
    jobs: Vec<Job>,
    next_id: u32,
    /// Seconds to add to UTC for local time.
    offset: i64,
}

/// Seconds to add to UTC for local time, right now. The standard library
/// cannot tell, so the system's `date` is asked; UTC if that fails. It
/// changes with daylight saving, so long-running callers ask again from time
/// to time (`Schedule::set_offset`).
pub fn utc_offset() -> i64 {
    let Ok(out) = std::process::Command::new("date").arg("+%z").output() else {
        return 0;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let text = text.trim();
    let (sign, digits) = match text.split_at_checked(1) {
        Some(("-", d)) => (-1, d),
        Some(("+", d)) => (1, d),
        _ => return 0,
    };
    match (
        digits.get(..2).and_then(|h| h.parse::<i64>().ok()),
        digits.get(2..4).and_then(|m| m.parse::<i64>().ok()),
    ) {
        (Some(h), Some(m)) => sign * (h * 3600 + m * 60),
        _ => 0,
    }
}

/// Days since 1970-01-01 of a calendar date (proleptic Gregorian).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The calendar date of a day count since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// A moment as local date and time, `2026-11-15 09:00`.
pub fn local(at: u64, offset: i64) -> String {
    let t = at as i64 + offset;
    let (y, m, d) = civil_from_days(t.div_euclid(DAY as i64));
    let s = t.rem_euclid(DAY as i64);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", s / 3600, s % 3600 / 60)
}

/// A length of time in words: `45 minutes`, `2 hours`.
fn span(secs: u64) -> String {
    let (n, unit) = match secs {
        s if s % DAY == 0 => (s / DAY, "day"),
        s if s % 3600 == 0 => (s / 3600, "hour"),
        s if s % 60 == 0 => (s / 60, "minute"),
        s => (s, "second"),
    };
    format!("{n} {unit}{}", if n == 1 { "" } else { "s" })
}

/// `10m`, `2h`, `1d`, `90s`, also with the unit spelled out.
fn parse_span(text: &str) -> Result<u64, String> {
    let text = text.trim().to_lowercase().replace(' ', "");
    let digits = text.chars().take_while(char::is_ascii_digit).count();
    let n: u64 = text[..digits]
        .parse()
        .map_err(|_| format!("`{text}` is not a length of time; use e.g. 10m, 2h or 1d"))?;
    let unit = match &text[digits..] {
        "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3600,
        "d" | "day" | "days" => DAY,
        other => {
            return Err(format!("`{other}` is not a unit of time; use s, m, h or d"));
        }
    };
    n.checked_mul(unit)
        .filter(|s| *s > 0 && *s <= 366 * DAY)
        .ok_or_else(|| "a length of time must be between 1 second and a year".to_string())
}

/// `09:00` as minutes after midnight.
fn parse_clock(text: &str) -> Result<u32, String> {
    let bad = || format!("`{text}` is not a time of day; use HH:MM, e.g. 09:00");
    let (h, m) = text.trim().split_once(':').ok_or_else(bad)?;
    match (h.parse::<u32>(), m.parse::<u32>()) {
        (Ok(h), Ok(m)) if h < 24 && m < 60 && m.to_string().len() <= 2 => Ok(h * 60 + m),
        _ => Err(bad()),
    }
}

/// The first moment after `now` at which local time is `minute` past midnight.
fn next_daily(minute: u32, now: u64, offset: i64) -> u64 {
    let local_now = now as i64 + offset;
    let midnight = local_now.div_euclid(DAY as i64) * DAY as i64;
    let mut at = midnight + i64::from(minute) * 60;
    if at <= local_now {
        at += DAY as i64;
    }
    (at - offset) as u64
}

/// A fingerprint of a file or folder that differs whenever its contents
/// have changed: sizes and modification times, never the contents. A folder
/// is walked (not into `.git`, `target` or `node_modules`, and not through
/// symlinks). Something that does not exist stamps as 0.
pub fn stamp(path: &Path) -> u64 {
    fn mix(hash: &mut u64, bytes: &[u8]) {
        for b in bytes {
            *hash = (*hash ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn walk(path: &Path, hash: &mut u64, left: &mut usize) {
        if *left == 0 {
            return;
        }
        *left -= 1;
        let Ok(meta) = std::fs::symlink_metadata(path) else {
            return;
        };
        mix(hash, path.as_os_str().as_encoded_bytes());
        if meta.is_dir() {
            let mut inside: Vec<PathBuf> = std::fs::read_dir(path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    !p.file_name()
                        .is_some_and(|n| n == ".git" || n == "target" || n == "node_modules")
                })
                .collect();
            inside.sort();
            for p in inside {
                walk(&p, hash, left);
            }
        } else {
            mix(hash, &meta.len().to_le_bytes());
            let changed = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            mix(hash, &changed.to_le_bytes());
        }
    }
    if std::fs::symlink_metadata(path).is_err() {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325;
    let mut left = MAX_STAMPED;
    walk(path, &mut hash, &mut left);
    // 0 is kept for "does not exist".
    hash.max(1)
}

/// The path in a rule of the form `when PATH changes`, if it is one.
pub fn watched_path(rule: &str) -> Option<&str> {
    let rest = rule.trim().strip_prefix("when ")?;
    let path = rest
        .trim()
        .strip_suffix("changes")
        .or_else(|| rest.trim().strip_suffix("change"))?
        .trim();
    (!path.is_empty()).then_some(path)
}

/// Reads a rule for when a job is due, and gives its first due time.
///
/// Forms: `in 10m` (once, after a delay), `every 2h` (repeating),
/// `daily 09:00` (every day at that local time), `at 09:00` (once, the next
/// time it is that time) and `at 2026-11-15 09:00` (once, then).
pub fn parse_when(text: &str, now: u64, offset: i64) -> Result<(When, u64), String> {
    let text = text.trim();
    let (word, rest) = text
        .split_once(char::is_whitespace)
        .map(|(w, r)| (w.to_lowercase(), r.trim()))
        .unwrap_or((text.to_lowercase(), ""));
    match word.as_str() {
        "in" => {
            let secs = parse_span(rest)?;
            if secs < MIN_DELAY {
                return Err(format!("a job cannot be due in under {MIN_DELAY} seconds"));
            }
            Ok((When::Once { at: now + secs }, now + secs))
        }
        "every" => {
            let secs = parse_span(rest)?;
            if secs < MIN_EVERY {
                return Err(format!(
                    "a repeating job cannot run more often than every {}",
                    span(MIN_EVERY)
                ));
            }
            Ok((When::Every { secs }, now + secs))
        }
        "daily" => {
            let minute = parse_clock(rest.trim_start_matches("at").trim())?;
            Ok((When::Daily { minute }, next_daily(minute, now, offset)))
        }
        "at" => {
            let at = match rest.split_once(char::is_whitespace) {
                None => next_daily(parse_clock(rest)?, now, offset),
                Some((date, clock)) => {
                    let bad = || format!("`{date}` is not a date; use YYYY-MM-DD");
                    let parts: Vec<i64> = date
                        .split('-')
                        .map(|p| p.parse::<i64>().map_err(|_| bad()))
                        .collect::<Result<_, _>>()?;
                    let [y, m, d] = parts[..] else {
                        return Err(bad());
                    };
                    if !(1..=12).contains(&m)
                        || !(1..=31).contains(&d)
                        || !(1970..=9999).contains(&y)
                    {
                        return Err(bad());
                    }
                    let days = days_from_civil(y, m, d);
                    if civil_from_days(days) != (y, m, d) {
                        return Err(bad());
                    }
                    let minute = i64::from(parse_clock(clock)?);
                    let at = days * DAY as i64 + minute * 60 - offset;
                    if at <= now as i64 {
                        return Err(format!(
                            "{} is in the past",
                            local(at.max(0) as u64, offset)
                        ));
                    }
                    at as u64
                }
            };
            Ok((When::Once { at }, at))
        }
        _ => Err(format!(
            "`{text}` is not a rule for when. Use one of: `in 10m`, `every 2h`, `daily 09:00`, \
             `at 09:00`, `at 2026-11-15 09:00`, `when PATH changes`"
        )),
    }
}

impl Job {
    /// The rule in words: `every 2 hours`, `once at 2026-11-15 09:00`.
    pub fn rule(&self, offset: i64) -> String {
        match &self.when {
            When::Once { at } => format!("once at {}", local(*at, offset)),
            When::Every { secs } => format!("every {}", span(*secs)),
            When::Daily { minute } => format!("daily at {:02}:{:02}", minute / 60, minute % 60),
            When::Changed { path, .. } => format!("when {} changes", path.display()),
        }
    }

    /// When it is next due, in words.
    pub fn due_in_words(&self, offset: i64) -> String {
        if self.next == NEVER {
            "when a change is seen".to_string()
        } else {
            local(self.next, offset)
        }
    }
}

impl Schedule {
    pub fn in_memory(offset: i64) -> Self {
        Self {
            path: None,
            jobs: Vec::new(),
            next_id: 1,
            offset,
        }
    }

    /// Where the schedule is kept: beside the routing knowledge.
    pub fn default_path() -> Option<PathBuf> {
        Some(
            crate::learn::default_path()?
                .parent()?
                .join("schedule.json"),
        )
    }

    /// Reads the saved schedule. A missing file is an empty schedule; an
    /// unreadable one is reported and left untouched on disk until the next
    /// change.
    pub fn load(path: &Path, offset: i64) -> (Self, Option<String>) {
        let mut schedule = Self {
            path: Some(path.to_path_buf()),
            ..Self::in_memory(offset)
        };
        let note = match std::fs::read_to_string(path) {
            Err(_) => None,
            Ok(text) => match serde_json::from_str::<Saved>(&text) {
                Ok(saved) => {
                    schedule.next_id = saved.next_id.max(1);
                    schedule.jobs = saved.jobs;
                    None
                }
                Err(e) => Some(format!(
                    "the saved schedule could not be read ({e}); starting with none"
                )),
            },
        };
        (schedule, note)
    }

    fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let saved = Saved {
            next_id: self.next_id,
            jobs: self.jobs.clone(),
        };
        let text = serde_json::to_string_pretty(&saved).map_err(|e| e.to_string())?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        // Written beside, then renamed, so a crash never leaves half a file.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    pub fn jobs(&self) -> &[Job] {
        &self.jobs
    }

    pub fn offset(&self) -> i64 {
        self.offset
    }

    /// Adds a job. Refused when the rule cannot be read, the task is empty,
    /// or the limit on jobs is reached.
    pub fn add(&mut self, task: &str, when: &str, missed: Missed, now: u64) -> Result<Job, String> {
        let task = task.trim();
        if task.is_empty() {
            return Err("say what the job should do".into());
        }
        if self.jobs.len() >= MAX_JOBS {
            return Err(format!(
                "there are already {MAX_JOBS} scheduled jobs, the most allowed; cancel one first"
            ));
        }
        let (when, next) = parse_when(when, now, self.offset)?;
        self.push(task, when, next, missed, now)
    }

    /// Adds a job that is due whenever `path` changes from how it looks now
    /// (`seen`, its `stamp`). It starts at most once per `MIN_EVERY`.
    pub fn add_watch(
        &mut self,
        task: &str,
        path: &Path,
        seen: u64,
        missed: Missed,
        now: u64,
    ) -> Result<Job, String> {
        let task = task.trim();
        if task.is_empty() {
            return Err("say what the job should do".into());
        }
        if self.jobs.len() >= MAX_JOBS {
            return Err(format!(
                "there are already {MAX_JOBS} scheduled jobs, the most allowed; cancel one first"
            ));
        }
        let when = When::Changed {
            path: path.to_path_buf(),
            seen,
        };
        self.push(task, when, NEVER, missed, now)
    }

    fn push(
        &mut self,
        task: &str,
        when: When,
        next: u64,
        missed: Missed,
        now: u64,
    ) -> Result<Job, String> {
        let job = Job {
            id: self.next_id,
            task: task.to_string(),
            when,
            next,
            paused: false,
            missed,
            created: now,
            runs: 0,
            last_run: None,
        };
        self.next_id += 1;
        self.jobs.push(job.clone());
        self.save()?;
        Ok(job)
    }

    /// The job to start now, if one is due: the one due longest, then the
    /// lowest id. Does not change anything; call `started` when it starts.
    pub fn due(&self, now: u64) -> Option<&Job> {
        self.jobs
            .iter()
            .filter(|j| !j.paused && j.next <= now)
            .min_by_key(|j| (j.next, j.id))
    }

    /// Records that a job has started and works out when it is next due. A
    /// one-off job is finished and removed. A repeating job counts from this
    /// start, not from when it was due, so a late run is never followed by a
    /// burst of catch-up runs.
    pub fn started(&mut self, id: u32, now: u64) {
        let offset = self.offset;
        let Some(at) = self.jobs.iter().position(|j| j.id == id) else {
            return;
        };
        let job = &mut self.jobs[at];
        job.runs += 1;
        job.last_run = Some(now);
        match job.when {
            When::Once { .. } => {
                self.jobs.remove(at);
            }
            When::Every { secs } => job.next = now + secs,
            When::Daily { minute } => job.next = next_daily(minute, now, offset),
            When::Changed { .. } => job.next = NEVER,
        }
        let _ = self.save();
    }

    /// Looks at what the change-triggered jobs are watching (`look` gives a
    /// path's `stamp`) and makes due each one whose path has changed. A job
    /// that started less than `MIN_EVERY` ago stays waiting, and fires once
    /// for everything that changed in the meantime.
    pub fn watch(&mut self, now: u64, look: &dyn Fn(&Path) -> u64) {
        let mut any = false;
        for job in &mut self.jobs {
            let rested = job.last_run.is_none_or(|last| now >= last + MIN_EVERY);
            if job.paused || job.next != NEVER || !rested {
                continue;
            }
            if let When::Changed { path, seen } = &mut job.when {
                let found = look(path);
                if found != *seen {
                    *seen = found;
                    job.next = now;
                    any = true;
                }
            }
        }
        if any {
            let _ = self.save();
        }
    }

    /// Has a change-triggered job take what it watches as it is now for
    /// unchanged. Used after the job's own run (so its own edits do not set
    /// it off again), on resuming it, and for missed changes it is to skip.
    pub fn rebase(&mut self, id: u32, look: &dyn Fn(&Path) -> u64) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id)
            && let When::Changed { path, seen } = &mut job.when
        {
            let found = look(path);
            if found != *seen {
                *seen = found;
                let _ = self.save();
            }
        }
    }

    /// Changes that happened while the harness was not running do not count
    /// for jobs set to skip missed runs.
    pub fn skip_unseen(&mut self, look: &dyn Fn(&Path) -> u64) {
        let ids: Vec<u32> = self
            .jobs
            .iter()
            .filter(|j| j.missed == Missed::Skip && matches!(j.when, When::Changed { .. }))
            .map(|j| j.id)
            .collect();
        for id in ids {
            self.rebase(id, look);
        }
    }

    /// Local time has moved against UTC (daylight saving, or the machine's
    /// time zone was changed): daily jobs are re-timed to the new clock. A
    /// job whose time today has, by the new clock, already passed without
    /// it having run is made due now instead of losing the day.
    pub fn set_offset(&mut self, offset: i64, now: u64) {
        if offset == self.offset {
            return;
        }
        self.offset = offset;
        for job in &mut self.jobs {
            if let When::Daily { minute } = job.when
                && job.next > now
            {
                let retimed = next_daily(minute, now, offset);
                job.next = if retimed > job.next + DAY / 2 {
                    now
                } else {
                    retimed
                };
            }
        }
        let _ = self.save();
    }

    /// Applies each job's missed-run rule at the start of a session: a job
    /// that is overdue and set to `Skip` is moved to its next time (or
    /// dropped, if it was a one-off). Jobs set to `Run` are left due and
    /// will start once. Returns what was done, in words.
    pub fn settle(&mut self, now: u64) -> Vec<String> {
        let offset = self.offset;
        let mut notes = Vec::new();
        self.jobs.retain_mut(|job| {
            if job.paused || job.next > now || job.missed == Missed::Run {
                return true;
            }
            match job.when {
                When::Once { .. } => {
                    notes.push(format!(
                        "Scheduled job #{} was due at {} while conscious was not running; it is set to skip missed runs, so it was dropped.",
                        job.id,
                        local(job.next, offset)
                    ));
                    false
                }
                When::Every { secs } => {
                    notes.push(format!(
                        "Scheduled job #{} missed its run at {}; skipped, next in {}.",
                        job.id,
                        local(job.next, offset),
                        span(secs)
                    ));
                    job.next = now + secs;
                    true
                }
                When::Daily { minute } => {
                    notes.push(format!(
                        "Scheduled job #{} missed its run at {}; skipped until tomorrow's.",
                        job.id,
                        local(job.next, offset)
                    ));
                    job.next = next_daily(minute, now, offset);
                    true
                }
                When::Changed { .. } => {
                    notes.push(format!(
                        "Scheduled job #{} was set off by a change while conscious was not running; skipped.",
                        job.id
                    ));
                    job.next = NEVER;
                    true
                }
            }
        });
        if !notes.is_empty() {
            let _ = self.save();
        }
        notes
    }

    /// Pauses or resumes a job. A resumed repeating job that is overdue
    /// waits for its next time instead of starting at once.
    pub fn set_paused(&mut self, id: u32, paused: bool, now: u64) -> Result<String, String> {
        let offset = self.offset;
        let job = self
            .jobs
            .iter_mut()
            .find(|j| j.id == id)
            .ok_or_else(|| format!("there is no scheduled job #{id}"))?;
        job.paused = paused;
        if !paused && job.next <= now {
            match job.when {
                When::Once { .. } => {}
                When::Every { secs } => job.next = now + secs,
                When::Daily { minute } => job.next = next_daily(minute, now, offset),
                When::Changed { .. } => job.next = NEVER,
            }
        }
        let said = if paused {
            format!("job #{id} is paused")
        } else {
            format!(
                "job #{id} is running again; next due {}",
                job.due_in_words(offset)
            )
        };
        self.save()?;
        Ok(said)
    }

    pub fn cancel(&mut self, id: u32) -> Result<String, String> {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        if self.jobs.len() == before {
            return Err(format!("there is no scheduled job #{id}"));
        }
        self.save()?;
        Ok(format!("job #{id} is cancelled"))
    }

    /// Every job, one per line, soonest first.
    pub fn report(&self, now: u64) -> String {
        if self.jobs.is_empty() {
            return "No jobs are scheduled.".into();
        }
        let mut jobs: Vec<&Job> = self.jobs.iter().collect();
        jobs.sort_by_key(|j| (j.paused, j.next, j.id));
        let mut out = format!(
            "{} scheduled job{} (they start by themselves, with or without the screen open):\n",
            jobs.len(),
            if jobs.len() == 1 { "" } else { "s" }
        );
        for j in jobs {
            let state = if j.paused {
                "paused".to_string()
            } else if j.next <= now {
                "due now".to_string()
            } else {
                format!("next {}", j.due_in_words(self.offset))
            };
            out.push_str(&format!(
                "#{} {} · {state} · run {} time{}{} · {}\n",
                j.id,
                j.rule(self.offset),
                j.runs,
                if j.runs == 1 { "" } else { "s" },
                if j.missed == Missed::Skip {
                    " · skips missed runs"
                } else {
                    ""
                },
                j.task.lines().next().unwrap_or_default()
            ));
        }
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-11-14 12:00:00 UTC.
    const NOON: u64 = 1_794_657_600;
    /// UTC+5:30.
    const IST: i64 = 19_800;

    #[test]
    fn dates_convert_both_ways() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
        assert_eq!(
            civil_from_days(days_from_civil(2026, 12, 31) + 1),
            (2027, 1, 1)
        );
        assert_eq!(local(NOON, 0), "2026-11-14 12:00");
        assert_eq!(local(NOON, IST), "2026-11-14 17:30");
        assert_eq!(local(NOON, -8 * 3600), "2026-11-14 04:00");
    }

    #[test]
    fn rules_are_read_into_a_first_due_time() {
        assert_eq!(
            parse_when("in 10m", NOON, 0),
            Ok((When::Once { at: NOON + 600 }, NOON + 600))
        );
        assert_eq!(
            parse_when("every 2 hours", NOON, 0),
            Ok((When::Every { secs: 7200 }, NOON + 7200))
        );
        // 17:30 local now; 09:00 local is tomorrow, 15h30m away.
        assert_eq!(
            parse_when("daily 09:00", NOON, IST),
            Ok((When::Daily { minute: 540 }, NOON + 15 * 3600 + 1800))
        );
        // 18:00 local is still today.
        assert_eq!(
            parse_when("at 18:00", NOON, IST),
            Ok((When::Once { at: NOON + 1800 }, NOON + 1800))
        );
        let (when, next) = parse_when("at 2026-11-15 09:00", NOON, IST).unwrap();
        assert_eq!(when, When::Once { at: next });
        assert_eq!(local(next, IST), "2026-11-15 09:00");
    }

    #[test]
    fn rules_that_make_no_sense_or_break_the_limits_are_refused() {
        for (rule, why) in [
            ("every 1m", "more often than every 5 minutes"),
            ("in 2s", "under 10 seconds"),
            ("at 2026-11-13 09:00", "in the past"),
            ("at 2026-02-30 09:00", "not a date"),
            ("daily 25:00", "not a time of day"),
            ("every fortnight", "not a length of time"),
            ("tomorrow", "not a rule for when"),
            ("", "not a rule for when"),
        ] {
            let err = parse_when(rule, NOON, 0).unwrap_err();
            assert!(err.contains(why), "{rule}: {err}");
        }
    }

    #[test]
    fn the_job_due_longest_runs_first_and_repeats_from_its_start() {
        let mut s = Schedule::in_memory(0);
        let hourly = s
            .add("check the build", "every 1h", Missed::Run, NOON)
            .unwrap();
        let once = s
            .add("send the summary", "in 30m", Missed::Run, NOON)
            .unwrap();
        assert_eq!(s.due(NOON + 60), None);
        assert_eq!(s.due(NOON + 1800).map(|j| j.id), Some(once.id));

        // Both overdue: the one due first goes first.
        let late = NOON + 2 * 3600;
        assert_eq!(s.due(late).map(|j| j.id), Some(once.id));
        s.started(once.id, late);
        // A one-off job is finished; the repeating one is next.
        assert_eq!(s.jobs().len(), 1);
        assert_eq!(s.due(late).map(|j| j.id), Some(hourly.id));
        s.started(hourly.id, late);
        // No burst of catch-up runs: the next is an hour after this start.
        assert_eq!(s.due(late + 3599), None);
        assert_eq!(s.due(late + 3600).map(|j| j.id), Some(hourly.id));
        assert_eq!(s.jobs()[0].runs, 1);
    }

    #[test]
    fn missed_runs_follow_each_jobs_own_rule() {
        let mut s = Schedule::in_memory(0);
        let run = s.add("a", "every 1h", Missed::Run, NOON).unwrap();
        let skip = s.add("b", "every 1h", Missed::Skip, NOON).unwrap();
        let gone = s.add("c", "in 1h", Missed::Skip, NOON).unwrap();
        let daily = s.add("d", "daily 13:00", Missed::Skip, NOON).unwrap();

        // The harness was off for two days.
        let back = NOON + 2 * DAY;
        let notes = s.settle(back);
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert!(
            notes
                .iter()
                .any(|n| n.contains(&format!("#{} was due", gone.id)))
        );
        assert!(s.jobs().iter().all(|j| j.id != gone.id));
        // Only the job that asked for it runs, and only once.
        assert_eq!(s.due(back).map(|j| j.id), Some(run.id));
        s.started(run.id, back);
        assert_eq!(s.due(back), None);
        let next = |id: u32| s.jobs().iter().find(|j| j.id == id).unwrap().next;
        assert_eq!(next(skip.id), back + 3600);
        assert_eq!(next(daily.id), back + 3600);
        // Settling again changes nothing.
        assert!(s.settle(back).is_empty());
    }

    #[test]
    fn jobs_are_limited_paused_cancelled_and_listed() {
        let mut s = Schedule::in_memory(0);
        let job = s
            .add("tidy the logs", "every 1h", Missed::Run, NOON)
            .unwrap();
        assert!(s.add("  ", "in 1h", Missed::Run, NOON).is_err());
        s.set_paused(job.id, true, NOON).unwrap();
        assert_eq!(s.due(NOON + DAY), None);
        assert!(
            s.report(NOON)
                .contains("#1 every 1 hour · paused · run 0 times · tidy the logs")
        );
        // Resumed late, it waits for its next time.
        s.set_paused(job.id, false, NOON + DAY).unwrap();
        assert_eq!(s.due(NOON + DAY), None);
        assert_eq!(s.due(NOON + DAY + 3600).map(|j| j.id), Some(job.id));
        assert!(s.set_paused(99, true, NOON).is_err());

        for n in 1..MAX_JOBS {
            s.add(&format!("job {n}"), "in 1h", Missed::Run, NOON)
                .unwrap();
        }
        let err = s
            .add("one too many", "in 1h", Missed::Run, NOON)
            .unwrap_err();
        assert!(err.contains("already 20 scheduled jobs"), "{err}");
        assert_eq!(s.cancel(job.id), Ok("job #1 is cancelled".into()));
        assert!(s.cancel(job.id).is_err());
        assert_eq!(
            Schedule::in_memory(0).report(NOON),
            "No jobs are scheduled."
        );
    }

    #[test]
    fn the_schedule_survives_a_restart() {
        let dir = std::env::temp_dir().join(format!("conscious-schedule-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("schedule.json");
        let (mut s, note) = Schedule::load(&path, IST);
        assert!(note.is_none() && s.jobs().is_empty());
        s.add("water the plants", "daily 07:30", Missed::Skip, NOON)
            .unwrap();
        let second = s.add("b", "in 1h", Missed::Run, NOON).unwrap();
        s.cancel(second.id).unwrap();

        let (again, note) = Schedule::load(&path, IST);
        assert!(note.is_none());
        assert_eq!(again.jobs(), s.jobs());
        assert_eq!(again.jobs()[0].rule(IST), "daily at 07:30");
        // Ids are never reused.
        let (mut again, _) = Schedule::load(&path, IST);
        assert_eq!(again.add("c", "in 1h", Missed::Run, NOON).unwrap().id, 3);

        std::fs::write(&path, "not json").unwrap();
        let (broken, note) = Schedule::load(&path, IST);
        assert!(broken.jobs().is_empty() && note.unwrap().contains("could not be read"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_job_starts_when_what_it_watches_changes_and_not_more_often_than_allowed() {
        use std::cell::Cell;
        let mut s = Schedule::in_memory(0);
        let looks = Cell::new(7u64);
        let look = |_: &Path| looks.get();
        let job = s
            .add_watch("rerun the tests", Path::new("/p/src"), 7, Missed::Run, NOON)
            .unwrap();
        assert_eq!(job.rule(0), "when /p/src changes");
        assert_eq!(job.due_in_words(0), "when a change is seen");
        assert!(s.report(NOON).contains("next when a change is seen"));

        // Nothing has changed: it is not due, however long it waits.
        s.watch(NOON + 10, &look);
        assert!(s.due(NOON + 10 * DAY).is_none());

        // A change makes it due at once.
        looks.set(8);
        s.watch(NOON + 20, &look);
        assert_eq!(s.due(NOON + 20).map(|j| j.id), Some(job.id));
        s.started(job.id, NOON + 20);
        assert!(s.due(NOON + 30).is_none());

        // What the run itself changed does not set it off again.
        looks.set(9);
        s.rebase(job.id, &look);
        s.watch(NOON + 20 + MIN_EVERY, &look);
        assert!(s.due(NOON + 20 + MIN_EVERY).is_none());

        // Changes soon after a run wait, and then count once.
        looks.set(10);
        s.watch(NOON + 60, &look);
        assert!(s.due(NOON + 60).is_none());
        looks.set(11);
        s.watch(NOON + 20 + MIN_EVERY, &look);
        assert_eq!(s.due(NOON + 20 + MIN_EVERY).map(|j| j.id), Some(job.id));
        assert_eq!(s.jobs()[0].runs, 1);

        // A paused job does not look.
        s.started(job.id, NOON + 20 + MIN_EVERY);
        s.set_paused(job.id, true, NOON).unwrap();
        looks.set(12);
        s.watch(NOON + DAY, &look);
        assert!(s.due(NOON + DAY).is_none());
    }

    #[test]
    fn changes_made_while_it_was_off_follow_the_missed_run_rule() {
        let mut s = Schedule::in_memory(0);
        let run = s
            .add_watch("a", Path::new("/p/a"), 1, Missed::Run, NOON)
            .unwrap();
        let skip = s
            .add_watch("b", Path::new("/p/b"), 1, Missed::Skip, NOON)
            .unwrap();
        // Both changed while nothing was running.
        let look = |_: &Path| 2u64;
        s.skip_unseen(&look);
        assert!(s.settle(NOON + DAY).is_empty());
        s.watch(NOON + DAY, &look);
        assert_eq!(s.due(NOON + DAY).map(|j| j.id), Some(run.id));
        s.started(run.id, NOON + DAY);
        assert!(s.due(NOON + DAY).is_none(), "job {} skipped it", skip.id);
    }

    #[test]
    fn a_real_file_stamps_differently_once_it_changes() {
        let dir = std::env::temp_dir().join(format!("cs-stamp-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let file = dir.join("src/a.txt");
        assert_eq!(stamp(&file), 0, "what is not there stamps as 0");
        std::fs::write(&file, "one").unwrap();
        let (first, folder) = (stamp(&file), stamp(&dir));
        assert_ne!(first, 0);
        assert_eq!(first, stamp(&file), "unchanged: the same stamp");
        std::fs::write(&file, "one and more").unwrap();
        assert_ne!(stamp(&file), first);
        assert_ne!(
            stamp(&dir),
            folder,
            "a folder changes when a file in it does"
        );
        // Build output and version control are not watched.
        let quiet = stamp(&dir);
        std::fs::create_dir_all(dir.join("target")).unwrap();
        std::fs::write(dir.join("target/out"), "x").unwrap();
        assert_eq!(stamp(&dir), quiet);
        std::fs::remove_file(&file).unwrap();
        assert_eq!(stamp(&file), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn watch_rules_are_told_from_time_rules() {
        assert_eq!(
            watched_path("when src/main.rs changes"),
            Some("src/main.rs")
        );
        assert_eq!(watched_path(" when my notes/ change "), Some("my notes/"));
        assert_eq!(watched_path("when changes"), None);
        assert_eq!(watched_path("every 2h"), None);
        assert!(
            parse_when("whenever", NOON, 0)
                .unwrap_err()
                .contains("when PATH changes")
        );
    }

    #[test]
    fn daily_jobs_follow_the_clock_when_daylight_saving_moves_it() {
        const HOUR: u64 = 3600;
        // 08:00 local at UTC+1; the job is for 09:00 local, an hour away.
        let mut s = Schedule::in_memory(HOUR as i64);
        let now = NOON - 5 * HOUR;
        let job = s.add("standup", "daily 09:00", Missed::Run, now).unwrap();
        assert_eq!(job.next, now + HOUR);

        // The clocks go back an hour: it is 07:00 now, 09:00 is two hours off.
        s.set_offset(0, now);
        assert_eq!(s.jobs()[0].next, now + 2 * HOUR);
        assert_eq!(local(s.jobs()[0].next, 0), "2026-11-14 09:00");

        // The clocks go forward two hours: it is 09:00 already past by the
        // new clock (09:30), and the job has not run today. It runs now
        // rather than losing the day.
        let later = now + 30 * 60;
        s.set_offset(2 * HOUR as i64, later);
        assert_eq!(s.jobs()[0].next, later);
        s.started(job.id, later);
        assert_eq!(local(s.jobs()[0].next, 2 * HOUR as i64), "2026-11-15 09:00");

        // The same offset again changes nothing.
        let before = s.jobs()[0].next;
        s.set_offset(2 * HOUR as i64, later + 60);
        assert_eq!(s.jobs()[0].next, before);
    }
}
