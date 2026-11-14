//! Start-up checks: what this machine has, and whether a model can be used.
//!
//! Two halves, kept apart on purpose. `probe` looks at the machine: which
//! backends are installed and signed in (asked of each CLI's own status
//! command), whether a local server answers, whether the data folder can be
//! written, whether a sandbox works. `judge` turns those facts into a report
//! and a choice of core model. `judge` is pure: the same facts always give
//! the same report and the same core, and no model is called to decide
//! anything. Only `call`, which the user asks for with `check --call`, sends
//! a real request.

use std::io::IsTerminal;
use std::path::Path;
use std::time::Instant;

use crate::provider::cli::{self, Login};
use crate::provider::{self, Message, Request};

/// What `probe` found. Everything `judge` decides comes from here.
#[derive(Clone, Debug)]
pub struct Facts {
    /// The core model the user named (`--model` or `CONSCIOUS_MODEL`).
    pub named: Option<String>,
    /// `None`: not installed.
    pub claude: Option<Login>,
    pub codex: Option<Login>,
    pub local_url: String,
    /// Specs of the models a local server offers; empty if none answers.
    pub local: Vec<String>,
    pub openai_key: bool,
    pub openrouter_key: bool,
    /// The data folder, or why it cannot be written.
    pub data: Result<String, String>,
    pub sandbox: bool,
    pub terminal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Level {
    Ok,
    /// Absent, and not needed.
    Off,
    Warn,
    Fail,
}

#[derive(Clone, Debug)]
pub struct Check {
    pub name: &'static str,
    pub level: Level,
    pub detail: String,
    /// What to do about it, when something can be done.
    pub fix: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub checks: Vec<Check>,
    /// The core model to start with. `None`: nothing usable.
    pub core: Option<String>,
    /// One spec per backend that is ready, for `check --call`.
    pub ready: Vec<String>,
}

fn env_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.trim().is_empty())
}

/// Whether an API key is to hand: in the environment, or among the secrets
/// the user typed into the secret window.
fn key_set(name: &str) -> bool {
    env_set(name) || crate::secrets::get(name).is_some()
}

/// Whether the folder the harness keeps its data in can be written.
fn data_folder() -> Result<String, String> {
    let file = crate::learn::default_path().ok_or("there is no home directory")?;
    let dir = file.parent().ok_or("the data path has no folder")?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let probe = dir.join(format!(".write-check-{}", std::process::id()));
    std::fs::write(&probe, b"").map_err(|e| format!("{}: {e}", dir.display()))?;
    let _ = std::fs::remove_file(&probe);
    Ok(dir.display().to_string())
}

/// Looks at the machine. Runs each installed CLI's status command and asks
/// the local server for its model list; calls no model.
pub fn probe(named: Option<String>) -> Facts {
    Facts {
        named,
        claude: cli::login("claude"),
        codex: cli::login("codex"),
        local_url: std::env::var("CONSCIOUS_LOCAL_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "http://127.0.0.1:8080/v1".into()),
        local: provider::openai::list_local_models()
            .into_iter()
            .map(|m| m.spec)
            .collect(),
        openai_key: key_set("OPENAI_API_KEY"),
        openrouter_key: key_set("OPENROUTER_API_KEY"),
        data: data_folder(),
        sandbox: crate::guard::sandbox_works(Path::new(".")),
        terminal: std::io::stdout().is_terminal() && std::io::stdin().is_terminal(),
    }
}

fn cli_check(name: &'static str, program: &str, state: Option<Login>) -> Check {
    let (level, detail, fix) = match state {
        Some(Login::Yes) => (Level::Ok, "installed and signed in".to_string(), None),
        Some(Login::Unknown) => (
            Level::Warn,
            "installed; it would not say whether it is signed in".to_string(),
            Some(format!(
                "run `conscious check --call` to try a real request, or update `{program}`"
            )),
        ),
        Some(Login::No) => (
            Level::Fail,
            "installed but signed out".to_string(),
            Some(match program {
                "claude" => "run `claude auth login`".to_string(),
                _ => format!("run `{program} login`"),
            }),
        ),
        None => (Level::Off, "not installed".to_string(), None),
    };
    Check {
        name,
        level,
        detail,
        fix,
    }
}

/// The report and the core model for these facts. Pure and total: no I/O,
/// no model, and the same facts always give the same answer.
///
/// With no model named the core is the first of: Claude Code signed in,
/// Codex signed in, a local server's first model, Claude Code installed
/// with unknown sign-in, Codex likewise. A named model is used as named,
/// unless its backend is known to be unusable.
pub fn judge(facts: &Facts) -> Report {
    let mut checks = vec![
        cli_check("Claude Code", "claude", facts.claude),
        cli_check("Codex", "codex", facts.codex),
        match facts.local.first() {
            Some(first) => Check {
                name: "Local server",
                level: Level::Ok,
                detail: format!(
                    "answering at {} with {} model{} ({first})",
                    facts.local_url,
                    facts.local.len(),
                    if facts.local.len() == 1 { "" } else { "s" }
                ),
                fix: None,
            },
            None => Check {
                name: "Local server",
                level: Level::Off,
                detail: format!("nothing answering at {}", facts.local_url),
                fix: None,
            },
        },
    ];

    let usable = |state: Option<Login>| matches!(state, Some(Login::Yes | Login::Unknown));
    // Why a named model cannot be used, if that is already certain.
    let blocked = |spec: &str| -> Option<(String, String)> {
        let backend = spec.split(':').next().unwrap_or(spec).trim();
        let cli_state = |what: &str, program: &str, state: Option<Login>| match state {
            None => Some((
                format!("{what} is not installed"),
                format!(
                    "install it, or name another model with --model (`{program}` was not found on PATH)"
                ),
            )),
            Some(Login::No) => Some((
                format!("{what} is signed out"),
                match program {
                    "claude" => "run `claude auth login`".to_string(),
                    _ => format!("run `{program} login`"),
                },
            )),
            _ => None,
        };
        match backend {
            "claude-code" => cli_state("Claude Code", "claude", facts.claude),
            "codex" => cli_state("Codex", "codex", facts.codex),
            "local" if facts.local.is_empty() => Some((
                format!("no local server is answering at {}", facts.local_url),
                "start llama-server or Ollama, or set CONSCIOUS_LOCAL_URL".to_string(),
            )),
            "openai" if !facts.openai_key => Some((
                "OPENAI_API_KEY is not set".to_string(),
                "export OPENAI_API_KEY before starting".to_string(),
            )),
            "openrouter" if !facts.openrouter_key => Some((
                "OPENROUTER_API_KEY is not set".to_string(),
                "export OPENROUTER_API_KEY before starting".to_string(),
            )),
            // Anything else, including an unknown backend, is for the
            // provider layer to accept or refuse.
            _ => None,
        }
    };

    let mut ready = Vec::new();
    if usable(facts.claude) {
        ready.push("claude-code".to_string());
    }
    if usable(facts.codex) {
        ready.push("codex".to_string());
    }
    ready.extend(facts.local.first().cloned());

    let (core, model) = match &facts.named {
        Some(spec) => match blocked(spec) {
            Some((why, fix)) => (
                None,
                Check {
                    name: "Core model",
                    level: Level::Fail,
                    detail: format!("`{spec}` was asked for, but {why}"),
                    fix: Some(fix),
                },
            ),
            None => (
                Some(spec.clone()),
                Check {
                    name: "Core model",
                    level: Level::Ok,
                    detail: format!("{spec} (the one you named)"),
                    fix: None,
                },
            ),
        },
        None => {
            let pick = [
                (facts.claude == Some(Login::Yes)).then(|| "claude-code".to_string()),
                (facts.codex == Some(Login::Yes)).then(|| "codex".to_string()),
                facts.local.first().cloned(),
                (facts.claude == Some(Login::Unknown)).then(|| "claude-code".to_string()),
                (facts.codex == Some(Login::Unknown)).then(|| "codex".to_string()),
            ]
            .into_iter()
            .flatten()
            .next();
            match pick {
                Some(spec) => (
                    Some(spec.clone()),
                    Check {
                        name: "Core model",
                        level: Level::Ok,
                        detail: format!("{spec} (the first backend that is ready)"),
                        fix: None,
                    },
                ),
                None => (
                    None,
                    Check {
                        name: "Core model",
                        level: Level::Fail,
                        detail: "no model can be used".to_string(),
                        fix: Some(
                            "install Claude Code or Codex and sign in, start a local model \
                             server, or name a model with --model (`conscious --help` lists \
                             the backends)"
                                .to_string(),
                        ),
                    },
                ),
            }
        }
    };
    if let Some(spec) = &facts.named
        && core.is_some()
        && !ready.contains(spec)
    {
        ready.insert(0, spec.clone());
    }
    checks.push(model);

    checks.push(match &facts.data {
        Ok(dir) => Check {
            name: "Data folder",
            level: Level::Ok,
            detail: format!("{dir} can be written"),
            fix: None,
        },
        Err(why) => Check {
            name: "Data folder",
            level: Level::Warn,
            detail: format!("cannot be written ({why})"),
            fix: Some(
                "nothing will be remembered between runs until this folder is writable".into(),
            ),
        },
    });
    checks.push(if facts.sandbox {
        Check {
            name: "Sandbox",
            level: Level::Ok,
            detail: "shell commands run confined to the project, with no network".into(),
            fix: None,
        }
    } else {
        Check {
            name: "Sandbox",
            level: Level::Warn,
            detail: "bubblewrap is missing or not allowed; every shell command will ask first"
                .into(),
            fix: Some("install it, e.g. `sudo apt install bubblewrap`".into()),
        }
    });
    checks.push(if facts.terminal {
        Check {
            name: "Terminal",
            level: Level::Ok,
            detail: "interactive".into(),
            fix: None,
        }
    } else {
        Check {
            name: "Terminal",
            level: Level::Warn,
            detail: "not interactive; the full screen needs a real terminal".into(),
            fix: Some("`conscious ask PROMPT` works without one".into()),
        }
    });
    Report {
        checks,
        core,
        ready,
    }
}

impl Report {
    /// The whole report, one check per line, for `conscious check`.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for c in &self.checks {
            let mark = match c.level {
                Level::Ok => "ok  ",
                Level::Off => "--  ",
                Level::Warn => "warn",
                Level::Fail => "FAIL",
            };
            out.push_str(&format!("{mark}  {:<13} {}\n", c.name, c.detail));
            if let Some(fix) = &c.fix {
                out.push_str(&format!("      {:<13} -> {fix}\n", ""));
            }
        }
        out
    }

    /// Only what is wrong, for a start that cannot go ahead.
    pub fn problems(&self) -> String {
        let mut out = String::new();
        for c in self.checks.iter().filter(|c| c.level == Level::Fail) {
            out.push_str(&format!("error: {}: {}\n", c.name, c.detail));
            if let Some(fix) = &c.fix {
                out.push_str(&format!("       -> {fix}\n"));
            }
        }
        out
    }

    /// One line for the timeline at start, and the warnings worth showing.
    pub fn notices(&self) -> Vec<(bool, String)> {
        let state = |name: &str| {
            self.checks
                .iter()
                .find(|c| c.name == name)
                .map_or(String::new(), |c| c.detail.clone())
        };
        let mut out = vec![(
            false,
            format!(
                "Start-up checks: core model {}. Claude Code {}; Codex {}; local server: {}.",
                state("Core model"),
                state("Claude Code"),
                state("Codex"),
                state("Local server"),
            ),
        )];
        for c in &self.checks {
            if matches!(c.level, Level::Warn | Level::Fail)
                && matches!(c.name, "Data folder" | "Sandbox" | "Claude Code" | "Codex")
            {
                out.push((
                    true,
                    format!(
                        "{}: {}.{}",
                        c.name,
                        c.detail,
                        c.fix
                            .as_ref()
                            .map_or(String::new(), |f| format!(" To fix: {f}."))
                    ),
                ));
            }
        }
        out
    }
}

impl Report {
    /// What a first start says before anything else: that conscious brings
    /// no model of its own, which ones it works through, and what this
    /// machine has. Pure: the same report gives the same text.
    pub fn welcome(&self) -> String {
        let mut out = String::from(
            "conscious does not come with a model. It works through one you already have:\n\n\
             \x20 Claude Code   the `claude` command, installed and signed in\n\
             \x20 Codex         the `codex` command, installed and signed in\n\n\
             It can also use a local server that is running (llama-server, Ollama), or an\n\
             OpenAI or OpenRouter key when you name that model with --model. Nothing else\n\
             is supported.\n\n\
             Checking this machine:\n\n",
        );
        out.push_str(&self.render());
        out
    }

    /// The backends to try on a first start, in a fixed order: the one the
    /// rules chose as the core, then every other one that looks ready.
    pub fn first_run_order(&self) -> Vec<String> {
        let mut order: Vec<String> = self.core.iter().cloned().collect();
        for spec in &self.ready {
            if !order.contains(spec) {
                order.push(spec.clone());
            }
        }
        order
    }
}

/// Where it is noted that the first-start check has passed on this machine.
pub fn first_run_marker() -> Option<std::path::PathBuf> {
    Some(
        crate::learn::default_path()?
            .parent()?
            .join("first-run-checked"),
    )
}

/// Sends one small real request to `spec` and says how it went. The only
/// part of the checks that uses a model, and only when asked for.
pub fn call(spec: &str) -> Result<String, String> {
    let provider = provider::from_spec(spec).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let req = Request {
        messages: vec![Message::user("Reply with the single word OK.")],
        ..Default::default()
    };
    let resp = provider
        .complete(&req, &mut |_| {})
        .map_err(|e| e.to_string())?;
    Ok(format!(
        "{} answered in {:.1}s",
        resp.model,
        started.elapsed().as_secs_f32()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Facts {
        Facts {
            named: None,
            claude: None,
            codex: None,
            local_url: "http://127.0.0.1:8080/v1".into(),
            local: Vec::new(),
            openai_key: false,
            openrouter_key: false,
            data: Ok("/data".into()),
            sandbox: true,
            terminal: true,
        }
    }

    fn level(report: &Report, name: &str) -> Level {
        report.checks.iter().find(|c| c.name == name).unwrap().level
    }

    #[test]
    fn a_first_start_says_what_is_supported_and_tries_the_core_before_the_rest() {
        // Nothing usable: the welcome still says what would be.
        let none = judge(&facts());
        let text = none.welcome();
        for needed in ["Claude Code", "Codex", "does not come with a model", "FAIL"] {
            assert!(text.contains(needed), "missing {needed:?} in:\n{text}");
        }
        assert!(none.first_run_order().is_empty());

        // Both signed in: Claude Code is the core, so it is asked first,
        // and Codex is what is tried if it does not answer.
        let both = judge(&Facts {
            claude: Some(Login::Yes),
            codex: Some(Login::Yes),
            ..facts()
        });
        assert_eq!(both.first_run_order(), ["claude-code", "codex"]);
        assert_eq!(both.welcome(), both.welcome());

        // A named model is the only one tried first; the others follow.
        let named = judge(&Facts {
            named: Some("codex".into()),
            claude: Some(Login::Yes),
            codex: Some(Login::Yes),
            ..facts()
        });
        assert_eq!(named.first_run_order()[0], "codex");
    }

    #[test]
    fn with_nothing_installed_there_is_no_core_and_the_report_says_what_to_do() {
        let report = judge(&facts());
        assert_eq!(report.core, None);
        assert!(report.ready.is_empty());
        assert_eq!(level(&report, "Core model"), Level::Fail);
        assert_eq!(level(&report, "Claude Code"), Level::Off);
        assert!(report.problems().contains("no model can be used"));
        assert!(
            report
                .problems()
                .contains("-> install Claude Code or Codex")
        );
    }

    #[test]
    fn the_core_is_the_first_backend_that_is_ready() {
        let all = Facts {
            claude: Some(Login::Yes),
            codex: Some(Login::Yes),
            local: vec!["local:gemma".into()],
            ..facts()
        };
        assert_eq!(judge(&all).core.as_deref(), Some("claude-code"));
        assert_eq!(judge(&all).ready, ["claude-code", "codex", "local:gemma"]);

        // Installed is not enough: a signed-out backend is passed over.
        let signed_out = Facts {
            claude: Some(Login::No),
            ..all.clone()
        };
        let report = judge(&signed_out);
        assert_eq!(report.core.as_deref(), Some("codex"));
        assert_eq!(level(&report, "Claude Code"), Level::Fail);
        assert!(report.render().contains("-> run `claude auth login`"));

        let only_local = Facts {
            claude: Some(Login::No),
            codex: None,
            ..all.clone()
        };
        assert_eq!(judge(&only_local).core.as_deref(), Some("local:gemma"));
    }

    #[test]
    fn a_backend_that_will_not_say_is_used_only_after_those_that_are_sure() {
        let unsure = Facts {
            claude: Some(Login::Unknown),
            ..facts()
        };
        let report = judge(&unsure);
        assert_eq!(report.core.as_deref(), Some("claude-code"));
        assert_eq!(level(&report, "Claude Code"), Level::Warn);

        let codex_sure = Facts {
            codex: Some(Login::Yes),
            ..unsure
        };
        assert_eq!(judge(&codex_sure).core.as_deref(), Some("codex"));
    }

    #[test]
    fn a_named_model_is_used_as_named_unless_its_backend_cannot_work() {
        let named = |spec: &str, base: Facts| {
            judge(&Facts {
                named: Some(spec.into()),
                ..base
            })
        };
        let ready = Facts {
            claude: Some(Login::Yes),
            ..facts()
        };
        assert_eq!(
            named("claude-code:opus", ready.clone()).core.as_deref(),
            Some("claude-code:opus")
        );
        // Nothing else is substituted for what the user asked for.
        let report = named("codex", ready.clone());
        assert_eq!(report.core, None);
        assert!(report.problems().contains("Codex is not installed"));

        let report = named("openrouter:some/model", ready.clone());
        assert!(report.problems().contains("OPENROUTER_API_KEY is not set"));
        let keyed = Facts {
            openrouter_key: true,
            ..ready.clone()
        };
        let report = named("openrouter:some/model", keyed);
        assert_eq!(report.core.as_deref(), Some("openrouter:some/model"));
        assert_eq!(report.ready[0], "openrouter:some/model");

        assert!(
            named("local", ready.clone())
                .problems()
                .contains("no local server")
        );
        // An unknown backend is left for the provider layer to refuse.
        assert_eq!(named("other:x", ready).core.as_deref(), Some("other:x"));
    }

    #[test]
    fn the_same_facts_always_give_the_same_report() {
        let f = Facts {
            claude: Some(Login::Unknown),
            codex: Some(Login::No),
            local: vec!["local:a".into(), "local:b".into()],
            sandbox: false,
            terminal: false,
            data: Err("read-only".into()),
            ..facts()
        };
        let (a, b) = (judge(&f), judge(&f));
        assert_eq!(a.render(), b.render());
        assert_eq!(a.core, b.core);
        assert_eq!(a.core.as_deref(), Some("local:a"));
        // Missing extras are warnings, never a reason not to start.
        assert_eq!(level(&a, "Sandbox"), Level::Warn);
        assert_eq!(level(&a, "Data folder"), Level::Warn);
        assert_eq!(level(&a, "Terminal"), Level::Warn);
        assert_eq!(a.notices().iter().filter(|(warn, _)| *warn).count(), 4);
    }

    #[test]
    fn status_output_of_the_clis_is_read() {
        assert_eq!(
            cli::claude_login("{\n  \"loggedIn\": true,\n  \"authMethod\": \"claude.ai\"\n}"),
            Login::Yes
        );
        assert_eq!(cli::claude_login("{\"loggedIn\": false}"), Login::No);
        assert_eq!(cli::claude_login("error: unknown command"), Login::Unknown);
        assert_eq!(
            cli::codex_login(true, "Logged in using ChatGPT"),
            Login::Yes
        );
        assert_eq!(cli::codex_login(false, "Not logged in"), Login::No);
        assert_eq!(
            cli::codex_login(false, "unrecognized subcommand"),
            Login::Unknown
        );
    }
}
