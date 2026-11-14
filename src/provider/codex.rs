//! Codex CLI as a model-only backend.
//!
//! Runs `codex exec --json` in an empty directory with a read-only sandbox and
//! no saved session. Unlike Claude Code, Codex has no flag to remove its
//! built-in tools, so the model is told not to use them and any use is
//! reported back as a warning. It authenticates with the CLI's existing login.

use std::process::Command;
use std::time::Instant;

use serde_json::Value;

use super::cli;
use super::{
    MAX_MODELS_PER_BACKEND, ModelInfo, Provider, ProviderError, Request, Response, Usage,
    family_and_version, latest_per_family,
};

/// The models this Codex login can use, from the CLI's own catalog.
pub fn list_models() -> Result<Vec<ModelInfo>, ProviderError> {
    if !cli::on_path("codex") {
        return Ok(Vec::new());
    }
    let mut cmd = Command::new("codex");
    cmd.args(["debug", "models"]);
    let mut json = String::new();
    let exit = cli::run(
        cmd,
        String::new(),
        std::time::Duration::from_secs(15),
        |l| {
            json.push_str(l);
            json.push('\n');
        },
    )?;
    if !exit.success {
        return Err(ProviderError::Protocol(cli::tail(&exit.stderr)));
    }
    parse_catalog(&json)
}

/// Reads `codex debug models`: the models Codex itself lists to users, in its
/// priority order, keeping only the newest of each family. Codex does not
/// state a size tier, so the core is asked to judge that from the
/// descriptions.
fn parse_catalog(json: &str) -> Result<Vec<ModelInfo>, ProviderError> {
    let v: Value = serde_json::from_str(json)
        .map_err(|e| ProviderError::Protocol(format!("model catalog is not JSON ({e})")))?;
    let models = v
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| ProviderError::Protocol("model catalog has no `models` list".into()))?;
    let mut listed: Vec<&Value> = models
        .iter()
        .filter(|m| m.get("visibility").and_then(Value::as_str) == Some("list"))
        .filter(|m| m.get("slug").and_then(Value::as_str).is_some())
        .collect();
    listed.sort_by_key(|m| {
        m.get("priority")
            .and_then(Value::as_i64)
            .unwrap_or(i64::MAX)
    });
    let slug = |m: &&Value| {
        m.get("slug")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    Ok(latest_per_family(listed, slug)
        .into_iter()
        .take(MAX_MODELS_PER_BACKEND)
        .map(|m| {
            let text = |key: &str| m.get(key).and_then(Value::as_str).unwrap_or_default();
            let mut description = text("description").trim().to_string();
            if description.is_empty() {
                description = text("display_name").to_string();
            }
            if let Some(window) = m.get("context_window").and_then(Value::as_u64) {
                description.push_str(&format!(" Context: {}k tokens.", window / 1000));
            }
            ModelInfo {
                spec: format!("codex:{}", text("slug")),
                family: format!("codex:{}", family_and_version(text("slug")).0),
                description,
                tier: None,
            }
        })
        .collect())
}

pub struct Codex {
    pub model: Option<String>,
    pub window: u32,
}

impl Provider for Codex {
    fn label(&self) -> String {
        match &self.model {
            Some(m) => format!("codex:{m}"),
            None => "codex".into(),
        }
    }

    fn context_window(&self) -> u32 {
        self.window
    }

    fn complete(
        &self,
        req: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Response, ProviderError> {
        let started = Instant::now();
        let dir = cli::workdir()?;
        let mut cmd = Command::new("codex");
        cmd.arg("exec")
            .arg("--json")
            .arg("--skip-git-repo-check")
            .arg("--ephemeral")
            .args(["--sandbox", "read-only"])
            .arg("-C")
            .arg(&dir);
        if let Some(m) = &self.model {
            cmd.args(["-m", m]);
        }
        // `-` reads the prompt from stdin. Codex has no system-prompt flag, so
        // the system text leads the prompt.
        cmd.arg("-");
        let prompt = format!("{}\n\n{}", cli::system_prompt(req), cli::transcript(req));

        let mut parser = Parser::default();
        let exit = cli::run(cmd, prompt, cli::TIMEOUT, |line| parser.line(line))?;

        if let Some(message) = parser.error {
            return Err(ProviderError::Api {
                status: None,
                message,
            });
        }
        let Some(message) = parser.message else {
            return Err(ProviderError::Protocol(format!(
                "codex exited without a reply: {}",
                cli::tail(&exit.stderr)
            )));
        };
        // Tool requests come back written out in the reply text.
        let (text, tool_calls) = if req.tools.is_empty() {
            (message, Vec::new())
        } else {
            cli::parse_reply(&message, &req.tools)?
        };
        on_text(&text);
        Ok(Response {
            text,
            tool_calls,
            usage: parser.usage,
            model: self.model.clone().unwrap_or_else(|| "codex".into()),
            latency: started.elapsed(),
            warnings: parser.warnings,
        })
    }
}

/// Reads `codex exec --json` output, one JSON event per line.
#[derive(Default)]
struct Parser {
    /// The last agent message; earlier ones are progress notes.
    message: Option<String>,
    usage: Usage,
    error: Option<String>,
    warnings: Vec<String>,
}

impl Parser {
    fn line(&mut self, line: &str) {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            return;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("item.completed") => {
                let kind = v.pointer("/item/type").and_then(Value::as_str);
                match kind {
                    Some("agent_message") => {
                        self.message = v
                            .pointer("/item/text")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                    }
                    Some(
                        k @ ("command_execution" | "file_change" | "mcp_tool_call" | "web_search"),
                    ) => self
                        .warnings
                        .push(format!("codex used its own tool ({k}) outside the harness")),
                    _ => {}
                }
            }
            Some("turn.completed") => {
                let n = |key: &str| {
                    v.pointer(&format!("/usage/{key}"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32
                };
                self.usage = Usage {
                    input: n("input_tokens"),
                    cached: n("cached_input_tokens"),
                    output: n("output_tokens"),
                };
            }
            Some("error") => {
                self.error = v.get("message").and_then(Value::as_str).map(str::to_string);
            }
            Some("turn.failed") => {
                if let Some(m) = v.pointer("/error/message").and_then(Value::as_str) {
                    self.error = Some(m.to_string());
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_keeps_listed_models_in_priority_order() {
        let json = r#"{"models": [
            {"slug": "b-model", "description": "Second.", "visibility": "list", "priority": 2, "context_window": 272000},
            {"slug": "internal", "description": "Hidden.", "visibility": "hide", "priority": 0},
            {"slug": "a-model", "description": "First.", "visibility": "list", "priority": 1, "context_window": 500000},
            {"slug": "c-model", "display_name": "C Model", "visibility": "list", "priority": 3}
        ]}"#;
        let models = parse_catalog(json).unwrap();
        let specs: Vec<&str> = models.iter().map(|m| m.spec.as_str()).collect();
        assert_eq!(specs, ["codex:a-model", "codex:b-model", "codex:c-model"]);
        assert_eq!(models[0].description, "First. Context: 500k tokens.");
        assert_eq!(models[2].description, "C Model");
        assert!(parse_catalog("{}").is_err());
    }

    #[test]
    fn catalog_drops_superseded_versions_of_a_family() {
        let json = r#"{"models": [
            {"slug": "gpt-6.1-sol", "description": "Latest.", "visibility": "list", "priority": 1},
            {"slug": "gpt-6-sol", "description": "Previous.", "visibility": "list", "priority": 2},
            {"slug": "gpt-6-luna", "description": "Fast.", "visibility": "list", "priority": 3},
            {"slug": "gpt-5.6-luna", "description": "Older.", "visibility": "list", "priority": 4}
        ]}"#;
        let models = parse_catalog(json).unwrap();
        let specs: Vec<&str> = models.iter().map(|m| m.spec.as_str()).collect();
        assert_eq!(specs, ["codex:gpt-6.1-sol", "codex:gpt-6-luna"]);
        assert_eq!(models[0].family, "codex:gpt-sol");
        assert_eq!(models[0].tier, None);
    }

    #[test]
    fn reply_usage_and_stray_tool_use_are_read() {
        let mut p = Parser::default();
        for l in [
            r#"{"type":"thread.started","thread_id":"t"}"#,
            r#"{"type":"turn.started"}"#,
            r#"{"type":"item.completed","item":{"id":"item_0","type":"reasoning","text":"thinking"}}"#,
            r#"{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"ls","exit_code":0}}"#,
            r#"{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"pong"}}"#,
            r#"{"type":"turn.completed","usage":{"input_tokens":900,"cached_input_tokens":800,"output_tokens":7}}"#,
        ] {
            p.line(l);
        }
        assert_eq!(p.message.as_deref(), Some("pong"));
        assert_eq!(
            p.usage,
            Usage {
                input: 900,
                cached: 800,
                output: 7
            }
        );
        assert_eq!(p.warnings.len(), 1);
        assert!(p.error.is_none());
    }

    // Event shapes as emitted by codex-cli 0.167.0 when the account was over its limit.
    #[test]
    fn usage_limit_is_reported_as_an_error() {
        let mut p = Parser::default();
        for l in [
            r#"{"type":"thread.started","thread_id":"t"}"#,
            r#"{"type":"turn.started"}"#,
            r#"{"type":"error","message":"You've hit your usage limit."}"#,
            r#"{"type":"turn.failed","error":{"message":"You've hit your usage limit."}}"#,
        ] {
            p.line(l);
        }
        assert_eq!(p.error.as_deref(), Some("You've hit your usage limit."));
        assert!(p.message.is_none());
    }
}
