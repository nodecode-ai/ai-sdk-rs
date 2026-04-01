#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ai_sdk_rs::core::error::TransportError;
use ai_sdk_rs::core::transport::{HttpTransport, TransportConfig, TransportStream};
use ai_sdk_rs::providers::openai::config::OpenAIConfig;
use ai_sdk_rs::providers::openai::OpenAIResponsesLanguageModel;
use ai_sdk_rs::types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream;
use serde_json::{json, Value};

#[derive(Clone)]
pub struct FixtureTransport {
    json_response: Arc<Value>,
    chunks: Arc<Vec<Bytes>>,
}

pub struct FixtureStreamResponse {
    chunks: Arc<Vec<Bytes>>,
}

impl FixtureTransport {
    pub fn new(json_response: Value, chunks: Vec<Bytes>) -> Self {
        Self {
            json_response: Arc::new(json_response),
            chunks: Arc::new(chunks),
        }
    }
}

#[async_trait]
impl HttpTransport for FixtureTransport {
    type StreamResponse = FixtureStreamResponse;

    fn into_stream(resp: Self::StreamResponse) -> (TransportStream, Vec<(String, String)>) {
        let chunks = (*resp.chunks).clone();
        (
            Box::pin(stream::iter(
                chunks.into_iter().map(Ok::<Bytes, TransportError>),
            )),
            vec![],
        )
    }

    async fn post_json_stream(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        Ok(FixtureStreamResponse {
            chunks: Arc::clone(&self.chunks),
        })
    }

    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        Ok(((*self.json_response).clone(), vec![]))
    }
}

pub fn benchmark_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark tokio runtime")
}

pub fn openai_model_with_json_response(
    model_id: &str,
    response: Value,
) -> OpenAIResponsesLanguageModel<FixtureTransport> {
    openai_model(model_id, FixtureTransport::new(response, Vec::new()))
}

pub fn openai_model_with_stream_fixture(
    model_id: &str,
    fixture_name: &str,
) -> OpenAIResponsesLanguageModel<FixtureTransport> {
    openai_model(
        model_id,
        FixtureTransport::new(minimal_text_response(), read_fixture_chunks(fixture_name)),
    )
}

pub fn minimal_text_response() -> Value {
    json!({
        "id": "resp_bench",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.1-mini",
        "output": [
            {
                "id": "msg_bench",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                        "logprobs": []
                    }
                ]
            }
        ],
        "usage": {
            "input_tokens": 32,
            "output_tokens": 4,
            "total_tokens": 36
        }
    })
}

pub fn stream_fixture_size_bytes(fixture_name: &str) -> usize {
    read_fixture_chunks(fixture_name)
        .iter()
        .map(Bytes::len)
        .sum()
}

pub fn stream_fixture_chunks(fixture_name: &str) -> Vec<Bytes> {
    read_fixture_chunks(fixture_name)
}

pub fn simple_call_options() -> v2t::CallOptions {
    let mut options = v2t::CallOptions::new(vec![
        v2t::PromptMessage::System {
            content: "You are a terse assistant.".into(),
            provider_options: None,
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Summarize what the SDK is responsible for in one sentence.".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
    ]);
    options.max_output_tokens = Some(128);
    options.temperature = Some(0.2);
    options.provider_options.insert(
        "openai".into(),
        HashMap::from([
            ("serviceTier".into(), json!("default")),
            ("textVerbosity".into(), json!("low")),
        ]),
    );
    options
}

pub fn tool_heavy_call_options() -> v2t::CallOptions {
    let mut options = v2t::CallOptions::new(vec![
        v2t::PromptMessage::System {
            content: "Use tools when they reduce latency or token cost.".into(),
            provider_options: None,
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Check docs, inspect a CSV, and return a structured answer.".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
        v2t::PromptMessage::Assistant {
            content: vec![
                v2t::AssistantPart::Reasoning {
                    text: "I need to inspect documentation and a file before answering.".into(),
                    provider_options: None,
                },
                v2t::AssistantPart::ToolCall(v2t::ToolCallPart {
                    tool_call_id: "call_lookup_docs".into(),
                    tool_name: "lookup_docs".into(),
                    input: json!({
                        "query": "ai-sdk-rs streaming normalization"
                    })
                    .to_string(),
                    provider_executed: false,
                    provider_metadata: None,
                    dynamic: false,
                    provider_options: None,
                }),
            ],
            provider_options: None,
        },
        v2t::PromptMessage::Tool {
            content: vec![v2t::ToolMessagePart::ToolResult(v2t::ToolResultPart {
                r#type: v2t::ToolResultPartType::ToolResult,
                tool_call_id: "call_lookup_docs".into(),
                tool_name: "lookup_docs".into(),
                output: v2t::ToolResultOutput::Json {
                    value: json!({
                        "sections": ["streaming", "providers", "request translation"]
                    }),
                },
                provider_options: None,
            })],
            provider_options: None,
        },
    ]);
    options.max_output_tokens = Some(512);
    options.temperature = Some(0.1);
    options.response_format = Some(v2t::ResponseFormat::Json {
        schema: Some(json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" },
                "sources": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["summary", "sources"],
            "additionalProperties": false
        })),
        name: Some("benchmark_result".into()),
        description: Some("Normalized tool-assisted answer.".into()),
    });
    options.tool_choice = Some(v2t::ToolChoice::Auto);
    options.tools = vec![
        v2t::Tool::Function(v2t::FunctionTool {
            r#type: v2t::FunctionToolType::Function,
            name: "lookup_docs".into(),
            description: Some("Look up local SDK documentation.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            strict: Some(true),
            provider_options: None,
        }),
        v2t::Tool::Provider(v2t::ProviderTool {
            r#type: v2t::ProviderToolType::Provider,
            id: "openai.web_search".into(),
            name: "web_search".into(),
            args: json!({
                "externalWebAccess": true,
                "searchContextSize": "high",
                "userLocation": {
                    "type": "approximate",
                    "country": "US",
                    "region": "CA",
                    "city": "San Francisco",
                    "timezone": "America/Los_Angeles"
                }
            }),
        }),
        v2t::Tool::Provider(v2t::ProviderTool {
            r#type: v2t::ProviderToolType::Provider,
            id: "openai.code_interpreter".into(),
            name: "python".into(),
            args: json!({
                "container": {
                    "fileIds": ["file-bench-a", "file-bench-b"]
                }
            }),
        }),
    ];
    options.provider_options.insert(
        "openai".into(),
        HashMap::from([
            ("parallelToolCalls".into(), json!(true)),
            ("store".into(), json!(true)),
            ("serviceTier".into(), json!("priority")),
            ("reasoningEffort".into(), json!("medium")),
            ("reasoningSummary".into(), json!("auto")),
            ("textVerbosity".into(), json!("medium")),
            (
                "include".into(),
                json!([
                    "reasoning.encrypted_content",
                    "message.output_text.logprobs"
                ]),
            ),
        ]),
    );
    options
}

pub fn stream_call_options(fixture_name: &str) -> v2t::CallOptions {
    let mut options = v2t::CallOptions::new(vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: format!("Replay benchmark fixture {fixture_name}"),
            provider_options: None,
        }],
        provider_options: None,
    }]);
    options.max_output_tokens = Some(512);
    options.tools = match fixture_name {
        "openai-web-search-tool.1" => vec![v2t::Tool::Provider(v2t::ProviderTool {
            r#type: v2t::ProviderToolType::Provider,
            id: "openai.web_search".into(),
            name: "web_search".into(),
            args: json!({
                "externalWebAccess": true,
                "searchContextSize": "medium"
            }),
        })],
        "openai-mcp-tool-approval.1" => vec![v2t::Tool::Provider(v2t::ProviderTool {
            r#type: v2t::ProviderToolType::Provider,
            id: "openai.mcp".into(),
            name: "MCP".into(),
            args: json!({
                "serverLabel": "zip1",
                "serverUrl": "https://zip1.io/mcp",
                "serverDescription": "Link shortener",
                "requireApproval": "always"
            }),
        })],
        "openai-code-interpreter-tool.1" => vec![v2t::Tool::Provider(v2t::ProviderTool {
            r#type: v2t::ProviderToolType::Provider,
            id: "openai.code_interpreter".into(),
            name: "python".into(),
            args: json!({
                "container": "auto"
            }),
        })],
        _ => Vec::new(),
    };
    options.tool_choice = Some(v2t::ToolChoice::Auto);
    options
}

fn openai_model(
    model_id: &str,
    transport: FixtureTransport,
) -> OpenAIResponsesLanguageModel<FixtureTransport> {
    OpenAIResponsesLanguageModel::new(
        model_id,
        OpenAIConfig {
            provider_name: "openai.responses".into(),
            provider_scope_name: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            endpoint_path: "/responses".into(),
            headers: vec![],
            query_params: vec![],
            supported_urls: HashMap::new(),
            file_id_prefixes: Some(vec!["file-".into()]),
            default_options: None,
            request_defaults: None,
        },
        transport,
        TransportConfig::default(),
    )
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("providers")
        .join("openai")
        .join("tests")
        .join("fixtures")
}

fn read_fixture_chunks(name: &str) -> Vec<Bytes> {
    let raw = std::fs::read_to_string(fixture_dir().join(format!("{name}.chunks.txt")))
        .unwrap_or_else(|err| panic!("missing fixture {name}: {err}"));
    let mut chunks = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        chunks.push(Bytes::from(format!("data: {trimmed}\n\n")));
    }
    chunks.push(Bytes::from_static(b"data: [DONE]\n\n"));
    chunks
}
