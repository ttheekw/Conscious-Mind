//! Model provider layer: one trait over every LLM backend.
//!
//! Backends:
//! - `openai` / `openrouter` / `local`: any OpenAI-compatible chat-completions
//!   endpoint over HTTP (`openai.rs`).
//! - `claude-code` / `codex`: the vendor CLIs run as subprocesses in
//!   *model-only* mode (`claude_code.rs`, `codex.rs`). Their built-in tools are
//!   switched off or confined, so every tool call still goes through this
//!   harness and its guardrails.
//!
//! A provider turns one assembled `Request` into one `Response`. It never runs
//! tools and never keeps conversation state; the harness owns both.

// The tool-calling half of this API is exercised by tests only until the goal
// loop lands and starts sending tool specs.
#![allow(dead_code)]

mod claude_code;
pub(crate) mod cli;
mod codex;
pub(crate) mod openai;

use std::fmt;
use std::time::Duration;

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// The result of a tool call, fed back to the model.
    Tool,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Tool calls the assistant made in this message.
    pub tool_calls: Vec<ToolCall>,
    /// For `Role::Tool`: which call this is the result of.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: text.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(text: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: text.into(),
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: output.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// A tool the model may request. `parameters` is a JSON Schema object.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Always a JSON object (checked by `check_tool_calls`).
    pub arguments: Value,
}

#[derive(Clone, Debug, Default)]
pub struct Request {
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// All input tokens, including cached ones.
    pub input: u32,
    pub cached: u32,
    pub output: u32,
}

#[derive(Clone, Debug)]
pub struct Response {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    /// The model that actually answered, as reported by the backend.
    pub model: String,
    pub latency: Duration,
    /// Things the harness should surface, e.g. a CLI backend that used one of
    /// its own tools despite being told not to.
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub enum ProviderError {
    /// Bad spec, missing API key, CLI not installed.
    Config(String),
    /// Could not reach the backend, or it timed out.
    Transport(String),
    /// The backend answered with an error (bad key, rate limit, usage limit).
    Api {
        status: Option<u16>,
        message: String,
    },
    /// The backend answered with something we could not parse.
    Protocol(String),
    /// The model asked for a tool that does not exist or with malformed
    /// arguments. Kept separate so the goal loop can feed it back to the model.
    BadToolCall { name: String, reason: String },
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Config(m) => write!(f, "configuration: {m}"),
            ProviderError::Transport(m) => write!(f, "transport: {m}"),
            ProviderError::Api {
                status: Some(s),
                message,
            } => write!(f, "backend error ({s}): {message}"),
            ProviderError::Api {
                status: None,
                message,
            } => write!(f, "backend error: {message}"),
            ProviderError::Protocol(m) => write!(f, "unexpected backend output: {m}"),
            ProviderError::BadToolCall { name, reason } => {
                write!(f, "invalid tool call `{name}`: {reason}")
            }
        }
    }
}

impl std::error::Error for ProviderError {}

pub trait Provider: Send + Sync {
    /// Short name for the UI, e.g. `openrouter:meta/llama` or `claude-code`.
    fn label(&self) -> String;

    /// Context window to budget against, in tokens. A per-backend default;
    /// override with `CONSCIOUS_CONTEXT`.
    fn context_window(&self) -> u32;

    /// Runs one model call. `on_text` receives visible text as it streams
    /// (backends that cannot stream call it once, at the end).
    fn complete(
        &self,
        req: &Request,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Response, ProviderError>;
}

pub const SPEC_HELP: &str = "\
  claude-code[:MODEL]   Claude Code CLI, using your existing login
  codex[:MODEL]         Codex CLI, using your existing login
  openai:MODEL          OpenAI API            (OPENAI_API_KEY, OPENAI_BASE_URL)
  openrouter:MODEL      OpenRouter            (OPENROUTER_API_KEY)
  local[:MODEL]         llama-server / Ollama (CONSCIOUS_LOCAL_URL, default
                        http://127.0.0.1:8080/v1)";

/// A model some backend on this machine can run right now.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// `backend:model`, as accepted by `from_spec`.
    pub spec: String,
    /// The model line this belongs to, without its version (`codex:gpt-sol`).
    /// Routing knowledge is keyed by this, so it survives version bumps.
    pub family: String,
    /// What it is good for, in the backend's own words where it provides them.
    pub description: String,
    /// Size tier when the backend states it: 1 small, 2 medium, 3 large.
    pub tier: Option<u8>,
}

/// Splits a model id into its family and version: `gpt-6.1-sol` is family
/// `gpt-sol`, version `[6, 1]`. A dash-separated part counts as version when
/// it is made only of digits and dots.
pub(crate) fn family_and_version(id: &str) -> (String, Vec<u32>) {
    let mut family = Vec::new();
    let mut version = Vec::new();
    for part in id.split('-') {
        let numeric = part.chars().any(|c| c.is_ascii_digit())
            && part.chars().all(|c| c.is_ascii_digit() || c == '.');
        if numeric {
            version.extend(part.split('.').filter_map(|n| n.parse::<u32>().ok()));
        } else {
            family.push(part);
        }
    }
    (family.join("-"), version)
}

/// Keeps only the newest model of each family, dropping superseded ones.
/// Families stay in the order they first appear.
pub(crate) fn latest_per_family<T>(items: Vec<T>, id: impl Fn(&T) -> String) -> Vec<T> {
    let mut kept: Vec<(String, Vec<u32>, T)> = Vec::new();
    for item in items {
        let (family, version) = family_and_version(&id(&item));
        match kept.iter_mut().find(|(f, _, _)| *f == family) {
            Some(slot) if version > slot.1 => *slot = (family, version, item),
            Some(_) => {}
            None => kept.push((family, version, item)),
        }
    }
    kept.into_iter().map(|(_, _, item)| item).collect()
}

/// Most models taken from one backend, in that backend's own priority order,
/// so the routing prompt stays short.
pub const MAX_MODELS_PER_BACKEND: usize = 6;

/// Asks every backend present on this machine which models it offers: the
/// current model of each family, never a superseded one. Nothing here names a
/// model version: the lists come from the backends themselves.
/// The second value lists backends that could not be queried, and why.
pub fn available_models() -> (Vec<ModelInfo>, Vec<String>) {
    let mut models = Vec::new();
    let mut notes = Vec::new();
    // A backend that says it is signed out is left out: every call to it
    // would fail and fall back to the core.
    if cli::login("claude") != Some(cli::Login::No) {
        models.extend(claude_code::list_models());
    }
    if cli::login("codex") != Some(cli::Login::No) {
        match codex::list_models() {
            Ok(found) => models.extend(found),
            Err(e) => notes.push(format!("could not list Codex models: {e}")),
        }
    }
    models.extend(openai::list_local_models());
    (models, notes)
}

/// Builds a provider from a `backend[:model]` spec.
pub fn from_spec(spec: &str) -> Result<Box<dyn Provider>, ProviderError> {
    let (backend, model) = match spec.split_once(':') {
        Some((b, m)) => (b.trim(), Some(m.trim()).filter(|m| !m.is_empty())),
        None => (spec.trim(), None),
    };
    let window = std::env::var("CONSCIOUS_CONTEXT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok());
    let need_model = |name: &str| {
        model.map(str::to_string).ok_or_else(|| {
            ProviderError::Config(format!("`{name}` needs a model, e.g. `{name}:<model-id>`"))
        })
    };
    // A key comes from the environment, or from the secrets the user typed
    // into the secret window. It is held by the provider and sent only to
    // that provider's own endpoint.
    let need_key = |var: &str| {
        std::env::var(var)
            .ok()
            .filter(|k| !k.trim().is_empty())
            .or_else(|| crate::secrets::get(var))
            .ok_or_else(|| {
                ProviderError::Config(format!(
                    "{var} is not set (set it in the environment, or press K in conscious \
                     and enter it as a secret)"
                ))
            })
    };
    let env_or = |var: &str, default: &str| {
        std::env::var(var)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| default.to_string())
    };
    Ok(match backend {
        "claude-code" | "claude" => Box::new(claude_code::ClaudeCode {
            model: model.map(str::to_string),
            window: window.unwrap_or(200_000),
        }),
        "codex" => Box::new(codex::Codex {
            model: model.map(str::to_string),
            window: window.unwrap_or(128_000),
        }),
        "openai" => Box::new(openai::OpenAiCompat {
            name: "openai",
            base_url: env_or("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key: Some(need_key("OPENAI_API_KEY")?),
            model: need_model("openai")?,
            max_tokens_field: "max_completion_tokens",
            window: window.unwrap_or(128_000),
        }),
        "openrouter" => Box::new(openai::OpenAiCompat {
            name: "openrouter",
            base_url: env_or("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key: Some(need_key("OPENROUTER_API_KEY")?),
            model: need_model("openrouter")?,
            max_tokens_field: "max_tokens",
            window: window.unwrap_or(128_000),
        }),
        "local" => Box::new(openai::OpenAiCompat {
            name: "local",
            base_url: env_or("CONSCIOUS_LOCAL_URL", "http://127.0.0.1:8080/v1"),
            api_key: std::env::var("CONSCIOUS_LOCAL_KEY").ok(),
            model: model.unwrap_or("gemma-4-e4b").to_string(),
            max_tokens_field: "max_tokens",
            window: window.unwrap_or(32_768),
        }),
        other => {
            return Err(ProviderError::Config(format!(
                "unknown backend `{other}`. Available:\n{SPEC_HELP}"
            )));
        }
    })
}

/// Rejects calls to tools that were not offered, or whose arguments are not a
/// JSON object. Full schema validation belongs to the tool layer; this is the
/// part every backend can check.
pub(crate) fn check_tool_calls(
    tools: &[ToolSpec],
    calls: &[ToolCall],
) -> Result<(), ProviderError> {
    for call in calls {
        if !tools.iter().any(|t| t.name == call.name) {
            return Err(ProviderError::BadToolCall {
                name: call.name.clone(),
                reason: "no such tool was offered".into(),
            });
        }
        if !call.arguments.is_object() {
            return Err(ProviderError::BadToolCall {
                name: call.name.clone(),
                reason: "arguments are not a JSON object".into(),
            });
        }
    }
    Ok(())
}

/// Parses a tool call's JSON-encoded arguments. An empty string means `{}`.
pub(crate) fn parse_arguments(name: &str, raw: &str) -> Result<Value, ProviderError> {
    if raw.trim().is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    serde_json::from_str(raw).map_err(|e| ProviderError::BadToolCall {
        name: name.to_string(),
        reason: format!("arguments are not valid JSON ({e})"),
    })
}

/// Removes a secret from text that may end up in the event log or the TUI.
pub(crate) fn redact(text: &str, secret: Option<&str>) -> String {
    match secret {
        Some(s) if s.len() >= 8 => text.replace(s, "[redacted]"),
        _ => text.to_string(),
    }
}

/// Rough token estimate (about four characters per token) for budgeting
/// before a backend reports real usage.
pub fn estimate_tokens(text: &str) -> u32 {
    (text.chars().count() as u32).div_ceil(4)
}

#[cfg(test)]
pub mod testing {
    use super::*;

    /// A provider that returns a fixed reply, for harness tests.
    pub struct Fixed(pub &'static str);

    impl Provider for Fixed {
        fn label(&self) -> String {
            "fixed".into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            on_text: &mut dyn FnMut(&str),
        ) -> Result<Response, ProviderError> {
            on_text(self.0);
            Ok(Response {
                text: self.0.to_string(),
                tool_calls: Vec::new(),
                usage: Usage {
                    input: req
                        .messages
                        .iter()
                        .map(|m| estimate_tokens(&m.content))
                        .sum(),
                    cached: 0,
                    output: estimate_tokens(self.0),
                },
                model: "fixed-1".into(),
                latency: Duration::from_millis(5),
                warnings: Vec::new(),
            })
        }
    }

    /// A named provider that gives its replies in order, repeating the last.
    pub struct Queue {
        label: &'static str,
        replies: std::sync::Mutex<std::collections::VecDeque<&'static str>>,
    }

    impl Queue {
        pub fn new(label: &'static str, replies: &[&'static str]) -> Self {
            Self {
                label,
                replies: std::sync::Mutex::new(replies.iter().copied().collect()),
            }
        }
    }

    impl Provider for Queue {
        fn label(&self) -> String {
            self.label.into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            req: &Request,
            on_text: &mut dyn FnMut(&str),
        ) -> Result<Response, ProviderError> {
            let mut replies = self.replies.lock().unwrap();
            let text = if replies.len() > 1 {
                replies.pop_front().unwrap()
            } else {
                replies.front().copied().unwrap_or_default()
            };
            // `CALL name {json}` scripts a tool call instead of text.
            if let Some(call) = text.strip_prefix("CALL ") {
                let (name, arguments) = call.split_once(' ').unwrap_or((call, "{}"));
                let mut resp = Scripted::ok(self.label, "").complete(req, on_text)?;
                resp.tool_calls = vec![ToolCall {
                    id: "call_0".into(),
                    name: name.into(),
                    arguments: serde_json::from_str(arguments)
                        .expect("scripted arguments are JSON"),
                }];
                return Ok(resp);
            }
            Scripted::ok(self.label, text).complete(req, on_text)
        }
    }

    /// A named provider with a canned reply or a canned failure.
    pub struct Scripted {
        pub label: &'static str,
        pub reply: Result<&'static str, &'static str>,
    }

    impl Scripted {
        pub fn ok(label: &'static str, reply: &'static str) -> Self {
            Self {
                label,
                reply: Ok(reply),
            }
        }

        pub fn err(label: &'static str, message: &'static str) -> Self {
            Self {
                label,
                reply: Err(message),
            }
        }
    }

    impl Provider for Scripted {
        fn label(&self) -> String {
            self.label.into()
        }
        fn context_window(&self) -> u32 {
            8_000
        }
        fn complete(
            &self,
            _req: &Request,
            on_text: &mut dyn FnMut(&str),
        ) -> Result<Response, ProviderError> {
            let text = self.reply.map_err(|m| ProviderError::Api {
                status: None,
                message: m.into(),
            })?;
            on_text(text);
            Ok(Response {
                text: text.to_string(),
                tool_calls: Vec::new(),
                usage: Usage {
                    input: 10,
                    cached: 0,
                    output: estimate_tokens(text),
                },
                model: self.label.into(),
                latency: Duration::from_millis(5),
                warnings: Vec::new(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn spec_selects_backend_and_model() {
        assert_eq!(from_spec("claude-code").unwrap().label(), "claude-code");
        assert_eq!(
            from_spec("claude-code:sonnet").unwrap().label(),
            "claude-code:sonnet"
        );
        assert_eq!(from_spec("codex:gpt-x").unwrap().label(), "codex:gpt-x");
        assert_eq!(from_spec("local:qwen").unwrap().label(), "local:qwen");
        assert!(matches!(
            from_spec("nope:model"),
            Err(ProviderError::Config(_))
        ));
    }

    #[test]
    fn tool_calls_must_name_an_offered_tool_with_object_arguments() {
        let tools = [ToolSpec {
            name: "read_file".into(),
            description: String::new(),
            parameters: json!({"type": "object"}),
        }];
        let call = |name: &str, arguments: Value| ToolCall {
            id: "c1".into(),
            name: name.into(),
            arguments,
        };
        assert!(check_tool_calls(&tools, &[call("read_file", json!({"path": "a"}))]).is_ok());
        assert!(check_tool_calls(&tools, &[call("rm", json!({}))]).is_err());
        assert!(check_tool_calls(&tools, &[call("read_file", json!("a"))]).is_err());
        assert_eq!(parse_arguments("t", "  ").unwrap(), json!({}));
        assert!(parse_arguments("t", "{oops").is_err());
    }

    #[test]
    fn only_the_newest_model_of_each_family_is_kept() {
        assert_eq!(
            family_and_version("gpt-6.1-sol"),
            ("gpt-sol".into(), vec![6, 1])
        );
        assert_eq!(
            family_and_version("gpt-6-astra"),
            ("gpt-astra".into(), vec![6])
        );
        assert_eq!(
            family_and_version("qwen2.5-coder-7b"),
            ("qwen2.5-coder-7b".into(), vec![])
        );

        let ids = vec![
            "gpt-6-sol",
            "gpt-6.1-sol",
            "gpt-6-astra",
            "gpt-5.6-sol",
            "gpt-5.6-luna",
            "gpt-6-luna",
            "plain",
        ];
        assert_eq!(
            latest_per_family(ids, |id| id.to_string()),
            ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna", "plain"]
        );
    }

    #[test]
    fn secrets_are_redacted() {
        assert_eq!(
            redact("bad key sk-abcdef123456", Some("sk-abcdef123456")),
            "bad key [redacted]"
        );
    }
}
