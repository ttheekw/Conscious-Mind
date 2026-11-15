//! Claude Code CLI as a model-only backend.
//!
//! Runs `claude -p` with its tools, MCP servers, settings, slash commands and
//! session persistence all switched off and our own system prompt, in an empty
//! directory. It authenticates with whatever login the CLI already has.

use std::process::Command;
use std::time::Instant;

use serde_json::Value;

use super::cli;
use super::{ModelInfo, Provider, ProviderError, Request, Response, ToolCall, Usage};

/// Claude Code has no command that lists models, so the pool is its three
/// model families. These are aliases the CLI resolves to the latest model of
/// each family, not model versions; a family the account cannot use fails on
/// first use and the router stops offering it.
pub fn list_models() -> Vec<ModelInfo> {
    if !cli::on_path("claude") {
        return Vec::new();
    }
    [
        ("haiku", 1, "Claude's small, fastest and cheapest family."),
        ("sonnet", 2, "Claude's balanced middle family."),
        ("opus", 3, "Claude's large, slower and costlier family."),
    ]
    .into_iter()
    .map(|(name, tier, description)| ModelInfo {
        spec: format!("claude-code:{name}"),
        // The alias already means "latest of this family".
        family: format!("claude-code:{name}"),
        description: description.into(),
        tier: Some(tier),
    })
    .collect()
}

/// The effort levels the Claude Code CLI takes.
pub const EFFORTS: [&str; 5] = ["low", "medium", "high", "xhigh", "max"];

pub struct ClaudeCode {
    pub model: Option<String>,
    /// How hard the model is asked to think (`low` … `max`), when set.
    /// Written after the model in a spec: `claude-code:opus@high`.
    pub effort: Option<String>,
    pub window: u32,
}

impl Provider for ClaudeCode {
    fn label(&self) -> String {
        let name = match &self.model {
            Some(m) => format!("claude-code:{m}"),
            None => "claude-code".into(),
        };
        match &self.effort {
            Some(effort) => format!("{name}@{effort}"),
            None => name,
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
        let mut cmd = Command::new("claude");
        cmd.current_dir(cli::workdir()?)
            .args(["-p", "--output-format", "stream-json", "--verbose"])
            .arg("--include-partial-messages")
            .args(["--system-prompt", &cli::system_prompt(req)])
            .args(["--tools", ""])
            .arg("--strict-mcp-config")
            .args(["--setting-sources", ""])
            .arg("--no-session-persistence")
            .arg("--disable-slash-commands");
        if let Some(m) = &self.model {
            cmd.args(["--model", m]);
        }
        if let Some(effort) = &self.effort {
            cmd.args(["--effort", effort]);
        }

        let mut parser = Parser::default();
        let exit = cli::run(cmd, cli::transcript(req), cli::TIMEOUT, |line| {
            parser.line(line, on_text);
        })?;

        let Some(result) = parser.result else {
            return Err(ProviderError::Protocol(format!(
                "claude exited without a result: {}",
                cli::tail(&exit.stderr)
            )));
        };
        if result.is_error {
            return Err(ProviderError::Api {
                status: None,
                message: result.text,
            });
        }
        // Tool requests come back written out in the reply text.
        let (text, tool_calls): (String, Vec<ToolCall>) = if req.tools.is_empty() {
            (result.text, Vec::new())
        } else {
            cli::parse_reply(&result.text, &req.tools)?
        };
        Ok(Response {
            text,
            tool_calls,
            usage: result.usage,
            model: parser
                .model
                .or_else(|| self.model.clone())
                .unwrap_or_else(|| "claude".into()),
            latency: started.elapsed(),
            warnings: parser.warnings,
        })
    }
}

#[derive(Debug)]
struct ResultLine {
    is_error: bool,
    text: String,
    usage: Usage,
}

/// Reads Claude Code's `stream-json` output, one JSON object per line.
#[derive(Default)]
struct Parser {
    model: Option<String>,
    result: Option<ResultLine>,
    warnings: Vec<String>,
}

impl Parser {
    fn line(&mut self, line: &str, on_text: &mut dyn FnMut(&str)) {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            return;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("system") if v.get("subtype").and_then(Value::as_str) == Some("init") => {
                self.model = v.get("model").and_then(Value::as_str).map(str::to_string);
                // The CLI is started with no tools; any it reports having
                // means one slipped through.
                let extra: Vec<&str> = v
                    .get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect();
                if !extra.is_empty() {
                    self.warnings.push(format!(
                        "claude-code started with its own tools enabled: {}",
                        extra.join(", ")
                    ));
                }
            }
            Some("stream_event") => {
                if v.pointer("/event/delta/type").and_then(Value::as_str) == Some("text_delta")
                    && let Some(t) = v.pointer("/event/delta/text").and_then(Value::as_str)
                {
                    on_text(t);
                }
            }
            Some("result") => {
                let n = |key: &str| {
                    v.pointer(&format!("/usage/{key}"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u32
                };
                let cached = n("cache_read_input_tokens");
                self.result = Some(ResultLine {
                    is_error: v.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                    text: v
                        .get("result")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    usage: Usage {
                        input: n("input_tokens") + n("cache_creation_input_tokens") + cached,
                        cached,
                        output: n("output_tokens"),
                    },
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lines captured from `claude -p --output-format stream-json` (2.1.323),
    // trimmed to the fields the parser reads.
    const PLAIN: &[&str] = &[
        r#"{"type":"system","subtype":"init","tools":[],"mcp_servers":[],"model":"claude-fable-5-2"}"#,
        r#"{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}}"#,
        r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"po"}}}"#,
        r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ng"}}}"#,
        r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed"}}"#,
        r#"{"type":"result","subtype":"success","is_error":false,"result":"pong","usage":{"input_tokens":2,"cache_creation_input_tokens":581,"cache_read_input_tokens":40,"output_tokens":4}}"#,
    ];

    #[test]
    fn plain_reply_streams_text_and_reports_usage() {
        let mut p = Parser::default();
        let mut seen = String::new();
        for l in PLAIN {
            p.line(l, &mut |t| seen.push_str(t));
        }
        assert_eq!(seen, "pong");
        assert_eq!(p.model.as_deref(), Some("claude-fable-5-2"));
        assert!(p.warnings.is_empty());
        let r = p.result.unwrap();
        assert!(!r.is_error);
        assert_eq!(r.text, "pong");
        assert_eq!(
            r.usage,
            Usage {
                input: 623,
                cached: 40,
                output: 4
            }
        );
    }

    #[test]
    fn a_cli_started_with_its_own_tools_is_reported() {
        let mut p = Parser::default();
        p.line(
            r#"{"type":"system","subtype":"init","tools":["Bash","Read"],"model":"m"}"#,
            &mut |_| {},
        );
        assert_eq!(p.warnings.len(), 1);
        assert!(p.warnings[0].contains("Bash, Read"));
    }

    /// End-to-end check against the installed CLI. Uses the real login, so it
    /// is opt-in: `cargo test -- --ignored real_claude`.
    #[test]
    #[ignore = "calls the real claude CLI"]
    fn real_claude_requests_an_offered_tool() {
        use crate::provider::{Message, ToolSpec};
        let req = Request {
            system: "Use the tools you are offered.".into(),
            messages: vec![Message::user("What is in Cargo.toml?")],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "Read a file from the working directory".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            }],
            max_tokens: None,
        };
        let backend = ClaudeCode {
            model: None,
            effort: None,
            window: 200_000,
        };
        let resp = backend.complete(&req, &mut |_| {}).unwrap();
        assert!(resp.warnings.is_empty(), "{:?}", resp.warnings);
        assert_eq!(resp.tool_calls.len(), 1, "{resp:?}");
        assert_eq!(resp.tool_calls[0].name, "read_file");
        assert_eq!(resp.tool_calls[0].arguments["path"], "Cargo.toml");
    }

    #[test]
    fn error_result_is_flagged() {
        let mut p = Parser::default();
        p.line(
            r#"{"type":"result","subtype":"error_during_execution","is_error":true,"result":"Not logged in"}"#,
            &mut |_| {},
        );
        let r = p.result.unwrap();
        assert!(r.is_error);
        assert_eq!(r.text, "Not logged in");
    }
}
