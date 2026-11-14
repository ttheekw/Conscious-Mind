//! OpenAI-compatible chat-completions backend (OpenAI, OpenRouter, and local
//! servers such as llama-server or Ollama), with SSE streaming.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::{
    MAX_MODELS_PER_BACKEND, Message, ModelInfo, Provider, ProviderError, Request, Response, Role,
    ToolCall, Usage, check_tool_calls, family_and_version, latest_per_family, parse_arguments,
    redact,
};

pub struct OpenAiCompat {
    pub name: &'static str,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// OpenAI wants `max_completion_tokens`; most compatible servers still
    /// only know `max_tokens`.
    pub max_tokens_field: &'static str,
    pub window: u32,
}

impl OpenAiCompat {
    fn body(&self, req: &Request) -> Value {
        let mut messages = Vec::new();
        if !req.system.is_empty() {
            messages.push(json!({"role": "system", "content": req.system}));
        }
        messages.extend(req.messages.iter().map(message_json));
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if !req.tools.is_empty() {
            body["tools"] = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        },
                    })
                })
                .collect();
        }
        if let Some(n) = req.max_tokens {
            body[self.max_tokens_field] = json!(n);
        }
        body
    }
}

fn message_json(m: &Message) -> Value {
    match m.role {
        Role::User => json!({"role": "user", "content": m.content}),
        Role::Tool => json!({
            "role": "tool",
            "tool_call_id": m.tool_call_id.as_deref().unwrap_or_default(),
            "content": m.content,
        }),
        Role::Assistant => {
            let mut v = json!({"role": "assistant", "content": m.content});
            if !m.tool_calls.is_empty() {
                v["tool_calls"] = m
                    .tool_calls
                    .iter()
                    .map(|c| {
                        json!({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments.to_string(),
                            },
                        })
                    })
                    .collect();
            }
            v
        }
    }
}

impl Provider for OpenAiCompat {
    fn label(&self) -> String {
        format!("{}:{}", self.name, self.model)
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
        let key = self.api_key.as_deref();
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_response(Some(Duration::from_secs(300)))
            .build()
            .into();
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let mut call = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");
        if let Some(k) = key {
            call = call.header("Authorization", format!("Bearer {k}"));
        }
        let resp = call
            .send(self.body(req).to_string())
            .map_err(|e| ProviderError::Transport(redact(&format!("{url}: {e}"), key)))?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let text = resp.into_body().read_to_string().unwrap_or_default();
            return Err(ProviderError::Api {
                status: Some(status),
                message: redact(&error_message(&text), key),
            });
        }

        let stream =
            read_stream(BufReader::new(resp.into_body().into_reader()), on_text).map_err(|e| {
                match e {
                    ProviderError::Api { status, message } => ProviderError::Api {
                        status,
                        message: redact(&message, key),
                    },
                    other => other,
                }
            })?;
        let mut tool_calls = Vec::new();
        for (i, raw) in stream.calls.into_iter().enumerate() {
            tool_calls.push(ToolCall {
                arguments: parse_arguments(&raw.name, &raw.arguments)?,
                id: if raw.id.is_empty() {
                    format!("call_{i}")
                } else {
                    raw.id
                },
                name: raw.name,
            });
        }
        check_tool_calls(&req.tools, &tool_calls)?;
        Ok(Response {
            text: stream.text,
            tool_calls,
            usage: stream.usage,
            model: stream.model.unwrap_or_else(|| self.model.clone()),
            latency: started.elapsed(),
            warnings: Vec::new(),
        })
    }
}

/// The models a local OpenAI-compatible server has loaded, or nothing if no
/// server is listening.
pub fn list_local_models() -> Vec<ModelInfo> {
    let base = std::env::var("CONSCIOUS_LOCAL_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8080/v1".into());
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(2)))
        .build()
        .into();
    let Ok(resp) = agent
        .get(format!("{}/models", base.trim_end_matches('/')))
        .call()
    else {
        return Vec::new();
    };
    if resp.status().as_u16() != 200 {
        return Vec::new();
    }
    let body = resp.into_body().read_to_string().unwrap_or_default();
    latest_per_family(parse_model_list(&body), String::clone)
        .into_iter()
        .take(MAX_MODELS_PER_BACKEND)
        .map(|id| ModelInfo {
            spec: format!("local:{id}"),
            family: format!("local:{}", family_and_version(&id).0),
            description:
                "Runs on this machine: free and private, but smaller and slower than hosted models."
                    .into(),
            // What fits on a local CPU is small next to hosted models.
            tier: Some(1),
        })
        .collect()
}

/// Model ids from an OpenAI-style `GET /models` body.
fn parse_model_list(body: &str) -> Vec<String> {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    parsed
        .as_ref()
        .and_then(|v| v.get("data"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

/// Pulls a readable message out of an error body (`{"error": {"message"}}`,
/// `{"error": "..."}`, or raw text).
fn error_message(body: &str) -> String {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let msg = parsed.as_ref().and_then(|v| {
        let e = v.get("error")?;
        e.get("message")
            .and_then(Value::as_str)
            .or_else(|| e.as_str())
            .map(str::to_string)
    });
    let msg = msg.unwrap_or_else(|| body.trim().to_string());
    if msg.is_empty() {
        "empty error response".into()
    } else {
        msg.chars().take(600).collect()
    }
}

#[derive(Debug, Default, PartialEq)]
pub(crate) struct RawCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Default)]
pub(crate) struct Stream {
    pub text: String,
    pub calls: Vec<RawCall>,
    pub usage: Usage,
    pub model: Option<String>,
}

/// Reads a chat-completions SSE stream to its end.
pub(crate) fn read_stream(
    reader: impl BufRead,
    on_text: &mut dyn FnMut(&str),
) -> Result<Stream, ProviderError> {
    let mut out = Stream::default();
    for line in reader.lines() {
        let line =
            line.map_err(|e| ProviderError::Transport(format!("stream interrupted: {e}")))?;
        // Blank lines separate events; lines starting with ':' are keep-alives.
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(data)
            .map_err(|e| ProviderError::Protocol(format!("bad stream chunk ({e})")))?;
        if let Some(err) = chunk.get("error") {
            return Err(ProviderError::Api {
                status: err.get("code").and_then(Value::as_u64).map(|c| c as u16),
                message: error_message(&chunk.to_string()),
            });
        }
        if out.model.is_none() {
            out.model = chunk
                .get("model")
                .and_then(Value::as_str)
                .filter(|m| !m.is_empty())
                .map(str::to_string);
        }
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            let n = |v: Option<&Value>| v.and_then(Value::as_u64).unwrap_or(0) as u32;
            out.usage = Usage {
                input: n(u.get("prompt_tokens")),
                cached: n(u.pointer("/prompt_tokens_details/cached_tokens")),
                output: n(u.get("completion_tokens")),
            };
        }
        let Some(delta) = chunk.pointer("/choices/0/delta") else {
            continue;
        };
        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            out.text.push_str(text);
            on_text(text);
        }
        for part in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = part.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            if out.calls.len() <= index {
                out.calls.resize_with(index + 1, RawCall::default);
            }
            let call = &mut out.calls[index];
            if let Some(id) = part.get("id").and_then(Value::as_str) {
                call.id.push_str(id);
            }
            if let Some(name) = part.pointer("/function/name").and_then(Value::as_str) {
                call.name.push_str(name);
            }
            if let Some(args) = part.pointer("/function/arguments").and_then(Value::as_str) {
                call.arguments.push_str(args);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolSpec;

    fn backend() -> OpenAiCompat {
        OpenAiCompat {
            name: "openai",
            base_url: "http://unused".into(),
            api_key: None,
            model: "m".into(),
            max_tokens_field: "max_completion_tokens",
            window: 1000,
        }
    }

    #[test]
    fn stream_collects_text_tool_calls_and_usage() {
        let sse = concat!(
            ": keep-alive\n\n",
            "data: {\"model\":\"m-1\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Rea\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ding.\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"a.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":9,\"prompt_tokens_details\":{\"cached_tokens\":100}}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut seen = String::new();
        let s = read_stream(sse.as_bytes(), &mut |t| seen.push_str(t)).unwrap();
        assert_eq!(s.text, "Reading.");
        assert_eq!(seen, "Reading.");
        assert_eq!(s.model.as_deref(), Some("m-1"));
        assert_eq!(
            s.usage,
            Usage {
                input: 120,
                cached: 100,
                output: 9
            }
        );
        assert_eq!(
            s.calls,
            vec![RawCall {
                id: "call_a".into(),
                name: "read_file".into(),
                arguments: "{\"path\":\"a.rs\"}".into(),
            }]
        );
    }

    #[test]
    fn stream_error_chunk_becomes_api_error() {
        let sse = "data: {\"error\":{\"code\":429,\"message\":\"slow down\"}}\n\n";
        let err = read_stream(sse.as_bytes(), &mut |_| {}).unwrap_err();
        assert!(
            matches!(err, ProviderError::Api { status: Some(429), ref message } if message == "slow down")
        );
    }

    #[test]
    fn body_carries_system_history_tools_and_limit() {
        let req = Request {
            system: "sys".into(),
            messages: vec![
                Message::user("hi"),
                Message::assistant(
                    "",
                    vec![ToolCall {
                        id: "c1".into(),
                        name: "read_file".into(),
                        arguments: json!({"path": "a"}),
                    }],
                ),
                Message::tool_result("c1", "contents"),
            ],
            tools: vec![ToolSpec {
                name: "read_file".into(),
                description: "read".into(),
                parameters: json!({"type": "object"}),
            }],
            max_tokens: Some(64),
        };
        let body = backend().body(&req);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a\"}"
        );
        assert_eq!(body["messages"][3]["tool_call_id"], "c1");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["max_completion_tokens"], 64);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn model_list_ids_are_read() {
        let body =
            r#"{"object":"list","data":[{"id":"qwen-small","object":"model"},{"id":"gemma"}]}"#;
        assert_eq!(parse_model_list(body), ["qwen-small", "gemma"]);
        assert!(parse_model_list("not json").is_empty());
    }

    #[test]
    fn error_bodies_are_reduced_to_a_message() {
        assert_eq!(
            error_message("{\"error\":{\"message\":\"bad key\"}}"),
            "bad key"
        );
        assert_eq!(error_message("{\"error\":\"nope\"}"), "nope");
        assert_eq!(error_message("gateway timeout"), "gateway timeout");
    }
}
