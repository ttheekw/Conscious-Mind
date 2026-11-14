//! The harness rewriting its own Rust source, rebuilding and restarting.
//!
//! When a task cannot meet its success criteria because of how this
//! harness itself is built (a missing ability, a limit, a bug), the model
//! may ask for a change to the source with the `self_update` tool. What
//! happens then is fixed here, not left to the model:
//!
//! 1. **A copy.** The source is copied to a work folder. The model edits
//!    the copy with the ordinary tools, under a guard that confines it to
//!    that folder. The real source is not touched until the end.
//! 2. **The gate**, run by the harness in the sandbox (no network, writes
//!    confined to the copy): `cargo fmt --check`, `cargo clippy -D
//!    warnings`, and the whole test suite must pass. No existing test may
//!    be removed, and the change must bring at least one new test
//!    (`test_fns`). The harness runs these itself; the model saying they
//!    pass counts for nothing.
//! 3. **Protected files.** Nothing in the harness is beyond changing,
//!    the guardrails included, but a change that touches the guardrails,
//!    the agent loop that enforces them, this file, the routing gate, the
//!    learned limits' bounds, the web rules, the socket, the build files or
//!    the installer (`PROTECTED`) is adopted only with the user's consent,
//!    whatever mode the harness is in. The user is shown the change itself
//!    (`diff`), not a description of it. Anything else that passes the gate
//!    is adopted by itself.
//! 4. **Adoption.** The files that changed are backed up and copied into
//!    the real source, the real binary is rebuilt there, and the harness
//!    restarts into it once nothing else is running.
//! 5. **Rollback.** If the new build cannot be made, or the new process
//!    does not come up, the backup is put back and the old one carries on.
//!    `conscious selfupdate rollback` does the same by hand.
//!
//! Without a working sandbox none of this runs: code a model wrote is
//! never built or tested outside one.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Files a self-update may not change without the user's approval, by
/// their path in the source tree.
pub const PROTECTED: [&str; 12] = [
    "src/guard.rs",
    "src/agent.rs",
    "src/selfupdate.rs",
    "src/policy.rs",
    "src/policy/default.rhai",
    "src/knowledge/tuning.rs",
    "src/web.rs",
    "src/daemon.rs",
    "Cargo.toml",
    "Cargo.lock",
    "install.sh",
    "build.rs",
];
/// Folders that are not source and are never copied or compared.
const SKIPPED: [&str; 4] = ["target", ".git", "graphify-out", "node_modules"];
/// A source file larger than this is not copied: it is not source.
const MAX_FILE_BYTES: u64 = 2_000_000;
/// How long each gate step may take. A first build compiles every
/// dependency.
const STEP_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// Disk that must be free before a self-update starts: the copy builds
/// every dependency for itself the first time.
pub const NEEDS_FREE: u64 = 6_000_000_000;
/// Attempts kept in the history.
const HISTORY_KEPT: usize = 40;

/// Where the source this binary was built from lives, if it is still
/// there. The path is fixed when the program is compiled.
pub fn source_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    (dir.join("Cargo.toml").is_file() && dir.join("src/main.rs").is_file()).then_some(dir)
}

/// Where the copies, backups and history are kept: a cache folder, apart
/// from the harness's data (which the agent can never write to).
pub fn home() -> Option<PathBuf> {
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(cache.join("conscious").join("selfupdate"))
}

/// Bytes free on the disk that holds `path` (or its nearest existing
/// parent), as `df` reports it. `None` if that cannot be found out, in
/// which case nothing is assumed.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let mut at = path;
    while !at.exists() {
        at = at.parent()?;
    }
    let out = std::process::Command::new("df")
        .arg("-Pk")
        .arg(at)
        .output()
        .ok()?;
    read_df(&String::from_utf8_lossy(&out.stdout))
}

/// The "available" column of `df -Pk`, in bytes.
fn read_df(output: &str) -> Option<u64> {
    let line = output.lines().nth(1)?;
    let kilobytes: u64 = line.split_whitespace().nth(3)?.parse().ok()?;
    Some(kilobytes * 1024)
}

/// Every source file under `root`, as paths relative to it, sorted.
fn files(root: &Path) -> Vec<PathBuf> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                if !SKIPPED.iter().any(|s| name == *s) {
                    walk(root, &path, out);
                }
            } else if meta.is_file()
                && meta.len() <= MAX_FILE_BYTES
                && let Ok(rel) = path.strip_prefix(root)
            {
                out.push(rel.to_path_buf());
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out.sort();
    out
}

/// Makes `work` a fresh copy of the source at `source`. Whatever was in
/// `work` before is replaced, except its build output, which is kept so a
/// second attempt does not compile every dependency again.
pub fn prepare(source: &Path, work: &Path) -> Result<usize, String> {
    let io = |e: std::io::Error| e.to_string();
    std::fs::create_dir_all(work).map_err(io)?;
    for old in files(work) {
        let _ = std::fs::remove_file(work.join(old));
    }
    let all = files(source);
    for rel in &all {
        let to = work.join(rel);
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        std::fs::copy(source.join(rel), &to).map_err(io)?;
    }
    Ok(all.len())
}

/// What a self-update did to one file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Added,
    Modified,
    Removed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Change {
    /// The file, relative to the source root, with `/` between parts.
    pub path: String,
    pub kind: Kind,
}

fn slashed(path: &Path) -> String {
    path.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// How the copy at `work` differs from the source at `source`.
pub fn changes(source: &Path, work: &Path) -> Vec<Change> {
    let before: BTreeSet<PathBuf> = files(source).into_iter().collect();
    let after: BTreeSet<PathBuf> = files(work).into_iter().collect();
    let mut out = Vec::new();
    for rel in before.union(&after) {
        let kind = match (before.contains(rel), after.contains(rel)) {
            (true, false) => Kind::Removed,
            (false, true) => Kind::Added,
            _ => {
                let same =
                    std::fs::read(source.join(rel)).ok() == std::fs::read(work.join(rel)).ok();
                if same {
                    continue;
                }
                Kind::Modified
            }
        };
        out.push(Change {
            path: slashed(rel),
            kind,
        });
    }
    out
}

/// Characters of diff shown to the user when asked to approve a change.
const MAX_DIFF_CHARS: usize = 12_000;

/// The change to `paths`, line by line, as the user is shown it before
/// approving: what would be taken out (`-`) and put in (`+`), with a few
/// lines around each. Cut off, and saying so, if very long.
pub fn diff(source: &Path, work: &Path, changed: &[Change], paths: &[String]) -> String {
    let mut out = String::new();
    for change in changed.iter().filter(|c| paths.contains(&c.path)) {
        let (old, new) = (source.join(&change.path), work.join(&change.path));
        let missing = Path::new("/dev/null");
        let (from, to) = match change.kind {
            Kind::Added => (missing, new.as_path()),
            Kind::Removed => (old.as_path(), missing),
            Kind::Modified => (old.as_path(), new.as_path()),
        };
        let shown = std::process::Command::new("diff")
            .arg("-u")
            .arg("--label")
            .arg(format!("{} (now)", change.path))
            .arg("--label")
            .arg(format!("{} (proposed)", change.path))
            .arg(from)
            .arg(to)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        if shown.trim().is_empty() {
            out.push_str(&format!(
                "{}: changed (the difference could not be shown)\n",
                change.path
            ));
        } else {
            out.push_str(&shown);
        }
        out.push('\n');
    }
    if out.chars().count() > MAX_DIFF_CHARS {
        let cut: String = out.chars().take(MAX_DIFF_CHARS).collect();
        return format!(
            "{cut}\n[… the change is longer than shown here; the whole of it is in {} …]",
            work.display()
        );
    }
    out.trim_end().to_string()
}

/// The changed files that need the user's approval before adoption.
pub fn protected(changes: &[Change]) -> Vec<String> {
    changes
        .iter()
        .filter(|c| PROTECTED.contains(&c.path.as_str()))
        .map(|c| c.path.clone())
        .collect()
}

/// Every test in the source under `root`, as `file: name`. Read from the
/// source text: an attribute line `#[test]` and the `fn` that follows it.
pub fn test_fns(root: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for rel in files(root) {
        if rel.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(root.join(&rel)) else {
            continue;
        };
        let mut pending = false;
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("#[test]") || line.starts_with("#[tokio::test") {
                pending = true;
            } else if pending && !line.starts_with("#[") && !line.is_empty() {
                pending = false;
                let rest = line
                    .trim_start_matches("pub ")
                    .trim_start_matches("async ")
                    .strip_prefix("fn ");
                if let Some(rest) = rest {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    out.insert(format!("{}: {name}", slashed(&rel)));
                }
            }
        }
    }
    out
}

/// What the test rules make of a change: the existing tests that are gone,
/// and how many new ones there are.
pub fn tests_kept(source: &Path, work: &Path) -> (Vec<String>, usize) {
    let (before, after) = (test_fns(source), test_fns(work));
    let gone = before.difference(&after).cloned().collect();
    (gone, after.difference(&before).count())
}

/// Something that can run a build command in a folder: the sandbox, or a
/// stand-in for tests.
pub trait Runner {
    /// Runs `command` in `dir`. `Ok` is its output when it succeeds, `Err`
    /// its output (or why it could not be run) when it does not.
    fn run(&self, dir: &Path, command: &str) -> Result<String, String>;
}

/// The real runner: the shell sandbox, with writes confined to `dir` and
/// no network.
pub struct Sandbox;

impl Runner for Sandbox {
    fn run(&self, dir: &Path, command: &str) -> Result<String, String> {
        let guard = crate::guard::Guard::for_scratch(dir)?;
        if !guard.can_sandbox() {
            return Err(
                "the sandbox is not available on this machine, and code the harness wrote \
                 itself is never built or tested outside one"
                    .into(),
            );
        }
        let mut cmd = guard.shell(command, false);
        cmd.env("TERM", "dumb").env("CARGO_TERM_COLOR", "never");
        let mut lines = Vec::new();
        let exit = crate::provider::cli::run(cmd, String::new(), STEP_TIMEOUT, |l| {
            lines.push(l.to_string())
        })
        .map_err(|e| e.to_string())?;
        let mut out = lines.join("\n");
        if !exit.stderr.trim().is_empty() {
            out.push('\n');
            out.push_str(exit.stderr.trim());
        }
        if exit.success { Ok(out) } else { Err(out) }
    }
}

/// The end of a command's output, which is where a failure is explained.
fn tail(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    lines[lines.len().saturating_sub(25)..].join("\n")
}

/// The steps of the gate, with what each is called when it fails.
const GATE: [(&str, &str); 3] = [
    ("formatting", "cargo fmt --check"),
    (
        "lint",
        "cargo clippy --all-targets --locked --offline -- -D warnings",
    ),
    ("tests", "cargo test --locked --offline"),
];

/// Runs the gate on the copy at `work`, against the source at `source`.
/// `Ok` carries a line on what passed; `Err` says which step failed and
/// shows the end of its output. `said` is told each step as it starts.
pub fn gate(
    runner: &dyn Runner,
    source: &Path,
    work: &Path,
    said: &dyn Fn(String),
) -> Result<String, String> {
    let changed = changes(source, work);
    if changed.is_empty() {
        return Err("nothing in the source was changed".into());
    }
    let (gone, new) = tests_kept(source, work);
    if !gone.is_empty() {
        return Err(format!(
            "existing tests were removed or renamed, which a self-update may not do: {}",
            gone.join(", ")
        ));
    }
    if new == 0 {
        return Err(
            "the change brings no new test; a self-update must add one for what it changes".into(),
        );
    }
    for (name, command) in GATE {
        said(format!("gate: {name} (`{command}`)"));
        if let Err(output) = runner.run(work, command) {
            return Err(format!("{name} failed (`{command}`):\n{}", tail(&output)));
        }
    }
    Ok(format!(
        "formatting, lint and the whole test suite pass; no existing test was removed and {new} \
         new test{} came with the change",
        if new == 1 { "" } else { "s" }
    ))
}

/// One attempt at a self-update, as kept in the history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub id: u32,
    /// Unix seconds.
    pub at: u64,
    /// What the harness could not do, in the words of the model that asked.
    pub problem: String,
    pub changes: Vec<Change>,
    /// `adopted`, `rejected`, `declined` or `rolled back`, and why.
    pub outcome: String,
    pub why: String,
}

#[derive(Default, Serialize, Deserialize)]
pub struct History {
    #[serde(default)]
    pub attempts: Vec<Attempt>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

impl History {
    pub fn load(home: &Path) -> Self {
        let path = home.join("history.json");
        let mut history: History = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        history.path = Some(path);
        history
    }

    pub fn next_id(&self) -> u32 {
        self.attempts.iter().map(|a| a.id).max().unwrap_or(0) + 1
    }

    /// Records an attempt, or replaces the record of the same attempt.
    pub fn record(&mut self, attempt: Attempt) {
        self.attempts.retain(|a| a.id != attempt.id);
        self.attempts.push(attempt);
        if self.attempts.len() > HISTORY_KEPT {
            let extra = self.attempts.len() - HISTORY_KEPT;
            self.attempts.drain(..extra);
        }
        if let Some(path) = &self.path
            && let Ok(text) = serde_json::to_string_pretty(self)
        {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(path, text);
        }
    }

    /// The newest attempt that was adopted and not rolled back since.
    pub fn last_adopted(&self) -> Option<&Attempt> {
        self.attempts.iter().rev().find(|a| a.outcome == "adopted")
    }

    pub fn report(&self, offset: i64) -> String {
        if self.attempts.is_empty() {
            return "The harness has not tried to change its own code yet.".into();
        }
        let mut out = String::new();
        for a in self.attempts.iter().rev() {
            out.push_str(&format!(
                "#{} {} · {} · {}\n    asked because: {}\n    changed: {}\n",
                a.id,
                crate::schedule::local(a.at, offset),
                a.outcome,
                a.why.lines().next().unwrap_or_default(),
                a.problem.lines().next().unwrap_or_default(),
                if a.changes.is_empty() {
                    "nothing".to_string()
                } else {
                    a.changes
                        .iter()
                        .map(|c| c.path.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ));
        }
        out.trim_end().to_string()
    }
}

/// Where attempt `id` keeps the files it replaced.
pub fn backup_dir(home: &Path, id: u32) -> PathBuf {
    home.join("backups").join(id.to_string())
}

/// Puts the changes from the copy into the real source, first saving
/// every file it replaces or removes under `backup` (and a note of the
/// files it adds, so they can be taken out again).
pub fn adopt(source: &Path, work: &Path, changed: &[Change], backup: &Path) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    let _ = std::fs::remove_dir_all(backup);
    std::fs::create_dir_all(backup.join("files")).map_err(io)?;
    // The backup is complete before the source is touched.
    for change in changed.iter().filter(|c| c.kind != Kind::Added) {
        let to = backup.join("files").join(&change.path);
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir).map_err(io)?;
        }
        std::fs::copy(source.join(&change.path), &to).map_err(io)?;
    }
    let list = serde_json::to_string_pretty(changed).map_err(|e| e.to_string())?;
    std::fs::write(backup.join("changes.json"), list).map_err(io)?;
    for change in changed {
        let to = source.join(&change.path);
        match change.kind {
            Kind::Removed => std::fs::remove_file(&to).map_err(io)?,
            Kind::Added | Kind::Modified => {
                if let Some(dir) = to.parent() {
                    std::fs::create_dir_all(dir).map_err(io)?;
                }
                std::fs::copy(work.join(&change.path), &to).map_err(io)?;
            }
        }
    }
    Ok(())
}

/// Undoes `adopt`: the source is put back as `backup` recorded it. Returns
/// how many files were restored or removed.
pub fn restore(source: &Path, backup: &Path) -> Result<usize, String> {
    let io = |e: std::io::Error| e.to_string();
    let list = std::fs::read_to_string(backup.join("changes.json")).map_err(io)?;
    let changed: Vec<Change> = serde_json::from_str(&list).map_err(|e| e.to_string())?;
    for change in &changed {
        let at = source.join(&change.path);
        match change.kind {
            Kind::Added => {
                let _ = std::fs::remove_file(&at);
            }
            Kind::Modified | Kind::Removed => {
                if let Some(dir) = at.parent() {
                    std::fs::create_dir_all(dir).map_err(io)?;
                }
                std::fs::copy(backup.join("files").join(&change.path), &at).map_err(io)?;
            }
        }
    }
    Ok(changed.len())
}

/// The command that builds the real binary, in the same way this one was
/// built.
pub fn build_command() -> &'static str {
    if cfg!(debug_assertions) {
        "cargo build --locked --offline"
    } else {
        "cargo build --release --locked --offline"
    }
}

/// Where a build in `source` leaves the binary.
pub fn built_binary(source: &Path) -> PathBuf {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    source.join("target").join(profile).join("conscious")
}

/// Puts the freshly built binary where this program runs from, if that is
/// somewhere else (an installed copy), keeping the old one under `backup`.
/// Returns whether anything had to be copied.
pub fn install_binary(source: &Path, backup: &Path) -> Result<bool, String> {
    let io = |e: std::io::Error| e.to_string();
    let built = built_binary(source);
    let running = std::env::current_exe().map_err(io)?;
    let same = running.canonicalize().ok() == built.canonicalize().ok();
    if same {
        return Ok(false);
    }
    std::fs::create_dir_all(backup).map_err(io)?;
    std::fs::copy(&running, backup.join("conscious.previous")).map_err(io)?;
    // Beside the destination, then renamed: a running copy is never half
    // overwritten.
    let staged = running.with_extension("new");
    std::fs::copy(&built, &staged).map_err(io)?;
    std::fs::rename(&staged, &running).map_err(io)?;
    Ok(true)
}

/// Undoes `install_binary`, if it did anything.
pub fn restore_binary(backup: &Path) -> Result<bool, String> {
    let io = |e: std::io::Error| e.to_string();
    let previous = backup.join("conscious.previous");
    if !previous.is_file() {
        return Ok(false);
    }
    let running = std::env::current_exe().map_err(io)?;
    let staged = running.with_extension("old");
    std::fs::copy(&previous, &staged).map_err(io)?;
    std::fs::rename(&staged, &running).map_err(io)?;
    Ok(true)
}

/// Takes back the newest adopted self-update: the source and the binary
/// are put back as they were before it, and the binary is rebuilt from
/// the restored source. Returns what was done, in words.
pub fn rollback(runner: &dyn Runner, why: &str) -> Result<String, String> {
    let home = home().ok_or("no home directory")?;
    let source = source_dir().ok_or("the source this program was built from is not there")?;
    let mut history = History::load(&home);
    let attempt = history
        .last_adopted()
        .cloned()
        .ok_or("no adopted self-update to take back")?;
    let backup = backup_dir(&home, attempt.id);
    let files = restore(&source, &backup)?;
    let binary = restore_binary(&backup)?;
    // Where the program runs straight from the build folder, the binary
    // is rebuilt from the restored source.
    let rebuilt = if binary {
        String::new()
    } else {
        match runner.run(&source, build_command()) {
            Ok(_) => " The binary was rebuilt from it.".to_string(),
            Err(output) => format!(
                " Rebuilding the binary failed; run `cargo build` in {}:\n{}",
                source.display(),
                tail(&output)
            ),
        }
    };
    history.record(Attempt {
        outcome: "rolled back".into(),
        why: why.to_string(),
        ..attempt.clone()
    });
    Ok(format!(
        "Self-update #{} was taken back ({why}): {files} file{} restored{}.{rebuilt}",
        attempt.id,
        if files == 1 { "" } else { "s" },
        if binary {
            " and the previous binary put back"
        } else {
            ""
        }
    ))
}

/// What the model is told when it is set to work on the copy.
pub fn brief(problem: &str, change: &str) -> String {
    format!(
        "You are changing the source code of `conscious`, the agent harness you are running \
         in. The folder you are in is a COPY of its source; the harness will take your change \
         from here only if it passes the gate described below.\n\n\
         Why a change is wanted: {problem}\n\
         What was asked for: {change}\n\n\
         How to work:\n\
         1. Read `AGENTS.md` first: it says how the code is laid out and what must stay as it \
         is. Then read the files you will touch.\n\
         2. Make the smallest change that removes the problem. Keep to the module boundaries \
         described there. Update `AGENTS.md` where it describes what you changed.\n\
         3. Add at least one test for the new behaviour. Do not remove, rename or weaken any \
         existing test.\n\
         4. Run `cargo fmt`, then `cargo clippy --all-targets --offline -- -D warnings` and \
         `cargo test --offline`, and fix what they report. There is no network.\n\
         5. When they pass, reply without a tool call: say what you changed and why.\n\n\
         The harness then runs the gate itself: formatting, lint and the whole test suite \
         must pass, no existing test may be gone, and there must be a new one. Your saying \
         they pass counts for nothing. A change to any of these files needs the user's \
         approval before it is adopted, so leave them alone unless the problem cannot be \
         solved otherwise: {}.",
        PROTECTED.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn tree(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cs-selfupdate-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        dir
    }

    fn write(root: &Path, rel: &str, text: &str) {
        let at = root.join(rel);
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        std::fs::write(at, text).unwrap();
    }

    /// A small source tree with two tests, and a fresh copy of it.
    fn project(name: &str) -> (PathBuf, PathBuf) {
        let source = tree(&format!("{name}-src"));
        write(&source, "Cargo.toml", "[package]\nname = \"x\"\n");
        write(&source, "src/main.rs", "fn main() {}\n");
        write(
            &source,
            "src/tools.rs",
            "pub fn double(n: u32) -> u32 { n * 2 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn doubles() {}\n\n    #[test]\n    #[ignore]\n    fn slow_one() {}\n}\n",
        );
        write(&source, "src/guard.rs", "pub fn check() {}\n");
        // Build output is not source.
        write(&source, "target/debug/junk", "binary");
        let work = tree(&format!("{name}-work"));
        prepare(&source, &work).unwrap();
        (source, work)
    }

    /// A runner that passes everything except the commands it is told to
    /// fail, and remembers what it was asked to run.
    struct Fake {
        fails: Vec<&'static str>,
        ran: RefCell<Vec<String>>,
    }

    impl Fake {
        fn passing() -> Self {
            Self::failing(&[])
        }
        fn failing(fails: &[&'static str]) -> Self {
            Self {
                fails: fails.to_vec(),
                ran: RefCell::default(),
            }
        }
    }

    impl Runner for Fake {
        fn run(&self, _: &Path, command: &str) -> Result<String, String> {
            self.ran.borrow_mut().push(command.to_string());
            if self.fails.iter().any(|f| command.contains(f)) {
                Err("line 1\nerror: something is wrong".into())
            } else {
                Ok("ok".into())
            }
        }
    }

    #[test]
    fn the_copy_holds_the_source_and_not_the_build_output() {
        let (source, work) = project("copy");
        assert!(work.join("src/tools.rs").is_file());
        assert!(!work.join("target").exists());
        assert!(changes(&source, &work).is_empty());
        // The copy's own build output survives a fresh copy; stale source
        // in it does not.
        write(&work, "target/debug/deps", "kept");
        write(&work, "src/stale.rs", "old");
        prepare(&source, &work).unwrap();
        assert!(work.join("target/debug/deps").is_file());
        assert!(!work.join("src/stale.rs").exists());
    }

    #[test]
    fn what_changed_is_listed_and_protected_files_are_singled_out() {
        let (source, work) = project("changes");
        write(
            &work,
            "src/tools.rs",
            "pub fn double(n: u32) -> u32 { n + n }\n",
        );
        write(&work, "src/extra.rs", "pub fn new() {}\n");
        write(&work, "src/guard.rs", "pub fn check() { /* looser */ }\n");
        std::fs::remove_file(work.join("src/main.rs")).unwrap();
        let changed = changes(&source, &work);
        let listed: Vec<(&str, &Kind)> =
            changed.iter().map(|c| (c.path.as_str(), &c.kind)).collect();
        assert_eq!(
            listed,
            [
                ("src/extra.rs", &Kind::Added),
                ("src/guard.rs", &Kind::Modified),
                ("src/main.rs", &Kind::Removed),
                ("src/tools.rs", &Kind::Modified),
            ]
        );
        assert_eq!(protected(&changed), ["src/guard.rs"]);
        // Every file the gate depends on is on the list.
        for file in [
            "src/guard.rs",
            "src/agent.rs",
            "src/selfupdate.rs",
            "Cargo.toml",
        ] {
            assert!(PROTECTED.contains(&file));
        }
    }

    #[test]
    fn the_gate_wants_every_old_test_kept_a_new_one_added_and_all_three_checks_to_pass() {
        let said = |_: String| {};
        // Nothing changed.
        let (source, work) = project("gate");
        assert!(
            gate(&Fake::passing(), &source, &work, &said)
                .unwrap_err()
                .contains("nothing in the source was changed")
        );

        // A change with no new test.
        write(
            &work,
            "src/tools.rs",
            &std::fs::read_to_string(source.join("src/tools.rs"))
                .unwrap()
                .replace("n * 2", "n + n"),
        );
        assert!(
            gate(&Fake::passing(), &source, &work, &said)
                .unwrap_err()
                .contains("brings no new test")
        );

        // A test removed (here: renamed), even with a new one added.
        let text = std::fs::read_to_string(source.join("src/tools.rs")).unwrap();
        write(
            &work,
            "src/tools.rs",
            &text.replace("fn doubles()", "fn doubles_differently()"),
        );
        let refused = gate(&Fake::passing(), &source, &work, &said).unwrap_err();
        assert!(
            refused.contains("existing tests were removed or renamed"),
            "{refused}"
        );
        assert!(refused.contains("src/tools.rs: doubles"), "{refused}");

        // Old tests kept and a new one added: the three checks run, in order.
        write(
            &work,
            "src/tools.rs",
            &text.replace(
                "    #[test]\n    fn doubles() {}",
                "    #[test]\n    fn doubles() {}\n\n    #[test]\n    fn doubles_zero() {}",
            ),
        );
        let runner = Fake::passing();
        let passed = gate(&runner, &source, &work, &said).unwrap();
        assert!(
            passed.contains("1 new test came with the change"),
            "{passed}"
        );
        let ran = runner.ran.borrow();
        assert_eq!(ran.len(), 3);
        assert!(
            ran[0].contains("fmt --check")
                && ran[1].contains("clippy")
                && ran[2].contains("cargo test")
        );
        // The checks never reach the network.
        assert!(ran[1].contains("--offline") && ran[2].contains("--offline"));

        // A failing check stops the gate and shows why.
        for (fails, named) in [
            ("fmt", "formatting failed"),
            ("clippy", "lint failed"),
            ("cargo test", "tests failed"),
        ] {
            let runner = Fake::failing(&[fails]);
            let refused = gate(&runner, &source, &work, &said).unwrap_err();
            assert!(refused.starts_with(named), "{refused}");
            assert!(refused.contains("error: something is wrong"));
        }
        // An ignored test still counts as a test that must be kept.
        assert!(test_fns(&source).contains("src/tools.rs: slow_one"));
    }

    #[test]
    fn adopting_copies_the_change_in_and_restoring_puts_everything_back() {
        let (source, work) = project("adopt");
        let before_tools = std::fs::read_to_string(source.join("src/tools.rs")).unwrap();
        write(
            &work,
            "src/tools.rs",
            "pub fn double(n: u32) -> u32 { n + n }\n",
        );
        write(&work, "src/extra.rs", "pub fn new() {}\n");
        std::fs::remove_file(work.join("src/guard.rs")).unwrap();
        let changed = changes(&source, &work);
        let backup = tree("adopt-backup");
        adopt(&source, &work, &changed, &backup).unwrap();
        assert!(
            changes(&source, &work).is_empty(),
            "the source now equals the copy"
        );
        assert!(source.join("src/extra.rs").is_file());
        assert!(!source.join("src/guard.rs").exists());
        // Build output in the source was left alone.
        assert!(source.join("target/debug/junk").is_file());

        assert_eq!(restore(&source, &backup).unwrap(), 3);
        assert_eq!(
            std::fs::read_to_string(source.join("src/tools.rs")).unwrap(),
            before_tools
        );
        assert!(source.join("src/guard.rs").is_file());
        assert!(!source.join("src/extra.rs").exists());
    }

    #[test]
    fn attempts_are_remembered_and_the_newest_adopted_one_is_the_one_to_take_back() {
        let home = tree("history");
        let mut history = History::load(&home);
        assert_eq!(history.next_id(), 1);
        assert!(history.report(0).contains("has not tried"));
        let attempt = |id: u32, outcome: &str| Attempt {
            id,
            at: 1_794_657_600,
            problem: "no way to read a PDF".into(),
            changes: vec![Change {
                path: "src/tools.rs".into(),
                kind: Kind::Modified,
            }],
            outcome: outcome.into(),
            why: "the gate passed".into(),
        };
        history.record(attempt(1, "adopted"));
        history.record(attempt(2, "rejected"));
        let again = History::load(&home);
        assert_eq!(again.next_id(), 3);
        assert_eq!(again.last_adopted().map(|a| a.id), Some(1));
        let report = again.report(0);
        assert!(
            report.contains("#2 2026-11-14 12:00 · rejected"),
            "{report}"
        );
        assert!(report.contains("asked because: no way to read a PDF"));
        // Once taken back, it is no longer the one to take back.
        let mut again = again;
        again.record(attempt(1, "rolled back"));
        assert!(again.last_adopted().is_none());
    }

    #[test]
    fn the_user_is_shown_the_change_itself_before_approving_it() {
        let (source, work) = project("diff");
        write(&work, "src/guard.rs", "pub fn check() { /* looser */ }\n");
        write(
            &work,
            "src/tools.rs",
            "pub fn double(n: u32) -> u32 { n + n }\n",
        );
        write(&work, "build.rs", "fn main() {}\n");
        let changed = changes(&source, &work);
        let needs = protected(&changed);
        assert_eq!(needs, ["build.rs", "src/guard.rs"]);
        let shown = diff(&source, &work, &changed, &needs);
        // What goes and what comes, for the protected files only.
        assert!(shown.contains("--- src/guard.rs (now)"), "{shown}");
        assert!(shown.contains("-pub fn check() {}"));
        assert!(shown.contains("+pub fn check() { /* looser */ }"));
        assert!(shown.contains("+++ build.rs (proposed)"));
        assert!(shown.contains("+fn main() {}"));
        assert!(
            !shown.contains("n + n"),
            "unprotected files are not part of what is asked"
        );
        // A very long change is cut, and says where the rest is.
        write(&work, "src/guard.rs", &"pub fn more() {}\n".repeat(2_000));
        let changed = changes(&source, &work);
        let shown = diff(&source, &work, &changed, &needs);
        assert!(shown.contains("the change is longer than shown here"));
        assert!(shown.chars().count() < MAX_DIFF_CHARS + 400);
    }

    #[test]
    fn free_disk_space_is_read_from_df() {
        let output = "Filesystem     1024-blocks      Used Available Capacity Mounted on\n/dev/nvme0n1p6   262144000 247000000   2726297      99% /\n";
        assert_eq!(read_df(output), Some(2_726_297 * 1024));
        assert_eq!(read_df("nonsense"), None);
        // On a real machine it gives some number, also for a folder that
        // does not exist yet.
        let missing = std::env::temp_dir().join("cs-not-there/deeper");
        assert!(free_bytes(&missing).is_some_and(|free| free > 0));
    }

    #[test]
    fn the_model_is_told_the_rules_it_works_under() {
        let text = brief(
            "it cannot read PDF files",
            "add a tool that extracts text from a PDF",
        );
        for said in [
            "COPY of its source",
            "it cannot read PDF files",
            "Read `AGENTS.md` first",
            "Add at least one test",
            "Your saying they pass counts for nothing",
            "src/guard.rs",
        ] {
            assert!(text.contains(said), "{said}");
        }
    }
}
