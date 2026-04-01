use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use bytes::Bytes;
use futures_util::{future::join_all, StreamExt, TryStreamExt};
use serde_json::{json, Value};

use super::ai_sdk_rs::core::transport::TransportConfig;
use super::ai_sdk_rs::core::LanguageModel;
use super::ai_sdk_rs::providers::amazon_bedrock::config::{BedrockAuth, BedrockConfig};
use super::ai_sdk_rs::providers::amazon_bedrock::language_model::BedrockLanguageModel;
use super::ai_sdk_rs::providers::anthropic::messages::language_model::AnthropicMessagesConfig;
use super::ai_sdk_rs::providers::anthropic::AnthropicMessagesLanguageModel;
use super::ai_sdk_rs::providers::gateway::config::{GatewayAuth, GatewayAuthMethod, GatewayConfig};
use super::ai_sdk_rs::providers::gateway::GatewayLanguageModel;
use super::ai_sdk_rs::providers::google::gen_ai::language_model::{
    GoogleGenAiConfig, GoogleGenAiLanguageModel,
};
use super::ai_sdk_rs::providers::google_vertex::{GoogleVertexConfig, GoogleVertexLanguageModel};
use super::ai_sdk_rs::providers::openai::config::OpenAIConfig;
use super::ai_sdk_rs::providers::openai::OpenAIResponsesLanguageModel;
use super::ai_sdk_rs::providers::openai_compatible::chat::language_model::{
    OpenAICompatibleChatConfig, OpenAICompatibleChatLanguageModel,
};
use super::ai_sdk_rs::types::v2 as v2t;
use super::ai_sdk_rs::types::{Event, TokenUsage};
use super::fixture_replay::FixtureTransport;
use super::{
    minimal_text_response, openai_responses_config, simple_call_options, stream_call_options,
    stream_fixture_chunks,
};

pub type ScenarioFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

pub struct JsonParseScenario {
    pub name: &'static str,
    pub payload: String,
}

pub struct EventMappingScenario {
    pub name: &'static str,
    pub events: Vec<Event>,
}

pub struct StreamCollectionScenario {
    pub name: &'static str,
    pub parts: Vec<v2t::StreamPart>,
}

pub struct AsyncProviderParseScenario {
    pub name: &'static str,
    pub bytes: u64,
    pub run: fn() -> ScenarioFuture,
}

pub struct AsyncConcurrentReplayScenario {
    pub name: &'static str,
    pub bytes: u64,
    pub run: fn() -> ScenarioFuture,
}

pub fn provider_matrix_families() -> &'static [&'static str] {
    &[
        "openai",
        "azure",
        "anthropic",
        "google",
        "google-vertex",
        "bedrock",
        "gateway",
        "openai-compatible",
    ]
}

pub fn json_parse_scenarios() -> Vec<JsonParseScenario> {
    vec![
        JsonParseScenario {
            name: "strict_json",
            payload: "{\"summary\":\"ok\",\"sources\":[\"streaming\",\"providers\",\"json\"]}"
                .to_string(),
        },
        JsonParseScenario {
            name: "noisy_fragment",
            payload: "tool-call-start<<<{\"summary\":\"ok\",\"sources\":[\"streaming\",\"providers\",\"json\"],\"nested\":{\"a\":[1,2,3],\"b\":{\"c\":\"d\"}}}>>>tool-call-end".to_string(),
        },
        JsonParseScenario {
            name: "large_wrapped_json",
            payload: large_wrapped_json(),
        },
        JsonParseScenario {
            name: "truncated_large_wrapped_json",
            payload: truncated_large_wrapped_json(),
        },
    ]
}

pub fn event_mapping_scenarios() -> Vec<EventMappingScenario> {
    vec![
        EventMappingScenario {
            name: "mixed_stream_small",
            events: small_provider_events(),
        },
        EventMappingScenario {
            name: "mixed_stream_scale",
            events: scaled_provider_events(),
        },
        EventMappingScenario {
            name: "interleaved_tool_fragments",
            events: interleaved_provider_events(),
        },
    ]
}

pub fn stream_collection_scenarios() -> Vec<StreamCollectionScenario> {
    vec![
        StreamCollectionScenario {
            name: "tool_rich_parts_small",
            parts: small_stream_parts(),
        },
        StreamCollectionScenario {
            name: "tool_rich_parts_scale",
            parts: scaled_stream_parts(),
        },
    ]
}

pub fn provider_parse_scenarios() -> Vec<AsyncProviderParseScenario> {
    vec![
        AsyncProviderParseScenario {
            name: "openai_fragmented_real_fixture",
            bytes: bytes_len(openai_fragmented_fixture_chunks()),
            run: || Box::pin(run_openai_fragmented_fixture_stream()),
        },
        AsyncProviderParseScenario {
            name: "anthropic_scale_fragmented",
            bytes: bytes_len(anthropic_scale_chunks()),
            run: || Box::pin(run_anthropic_scale_stream()),
        },
        AsyncProviderParseScenario {
            name: "gateway_scale_fragmented",
            bytes: bytes_len(gateway_scale_chunks()),
            run: || Box::pin(run_gateway_scale_stream()),
        },
        AsyncProviderParseScenario {
            name: "openai_compatible_adversarial",
            bytes: bytes_len(openai_compatible_adversarial_chunks()),
            run: || Box::pin(run_openai_compatible_adversarial_stream()),
        },
    ]
}

pub fn concurrent_replay_scenarios() -> Vec<AsyncConcurrentReplayScenario> {
    vec![
        AsyncConcurrentReplayScenario {
            name: "openai_parallel_4",
            bytes: bytes_len(openai_fragmented_fixture_chunks()) * 4,
            run: || Box::pin(run_openai_parallel_replay()),
        },
        AsyncConcurrentReplayScenario {
            name: "anthropic_parallel_4",
            bytes: bytes_len(anthropic_scale_chunks()) * 4,
            run: || Box::pin(run_anthropic_parallel_replay()),
        },
        AsyncConcurrentReplayScenario {
            name: "gateway_backpressure_4",
            bytes: bytes_len(gateway_scale_chunks()) * 4,
            run: || Box::pin(run_gateway_backpressure_replay()),
        },
        AsyncConcurrentReplayScenario {
            name: "openai_compatible_backpressure_4",
            bytes: bytes_len(openai_compatible_adversarial_chunks()) * 4,
            run: || Box::pin(run_openai_compatible_backpressure_replay()),
        },
    ]
}

pub async fn run_openai_generate() {
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.1-mini",
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
        FixtureTransport::new(minimal_text_response(), Vec::new()),
        TransportConfig::default(),
    );
    let response = model
        .do_generate(simple_call_options())
        .await
        .expect("openai provider matrix generate");
    std::hint::black_box(response);
}

pub async fn run_azure_generate() {
    let mut options = v2t::CallOptions::new(vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Summarize the provider matrix in one sentence.".into(),
            provider_options: None,
        }],
        provider_options: None,
    }]);
    options.max_output_tokens = Some(96);
    options.provider_options.insert(
        "azure".into(),
        HashMap::from([("serviceTier".into(), json!("default"))]),
    );

    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4.1-mini",
        OpenAIConfig {
            provider_name: "azure.responses".into(),
            provider_scope_name: "azure".into(),
            base_url: "https://example.openai.azure.com/openai".into(),
            endpoint_path: "/v1/responses".into(),
            headers: vec![("api-key".into(), "test-key".into())],
            query_params: vec![("api-version".into(), "2025-03-01-preview".into())],
            supported_urls: HashMap::new(),
            file_id_prefixes: Some(vec!["assistant-".into()]),
            default_options: None,
            request_defaults: None,
        },
        FixtureTransport::new(minimal_text_response(), Vec::new()),
        TransportConfig::default(),
    );
    let response = model
        .do_generate(options)
        .await
        .expect("azure provider matrix generate");
    std::hint::black_box(response);
}

pub async fn run_anthropic_stream() {
    let cfg = AnthropicMessagesConfig {
        provider_name: "anthropic",
        provider_scope_name: "anthropic".into(),
        base_url: "https://api.example.com".into(),
        headers: vec![],
        http: FixtureTransport::new(json!({}), anthropic_stream_chunks()),
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        default_options: None,
    };
    let model = AnthropicMessagesLanguageModel::new("claude-3-5-sonnet-20241022".to_string(), cfg);
    let response = model
        .do_stream(v2t::CallOptions::new(matrix_prompt()))
        .await
        .expect("anthropic provider matrix stream");
    let parts = response
        .stream
        .try_collect::<Vec<_>>()
        .await
        .expect("anthropic provider matrix stream parts");
    std::hint::black_box(parts);
}

pub async fn run_google_generate() {
    let cfg = GoogleGenAiConfig {
        provider_name: "google.gen-ai",
        provider_scope_name: "google".into(),
        base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
        headers: vec![],
        http: FixtureTransport::new(google_generate_response(), Vec::new()),
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        query_params: vec![],
        default_options: None,
        warn_on_include_thoughts: true,
    };
    let model = GoogleGenAiLanguageModel::new("gemini-2.5-flash", cfg);
    let response = model
        .do_generate(v2t::CallOptions::new(google_matrix_prompt()))
        .await
        .expect("google provider matrix generate");
    std::hint::black_box(response);
}

pub async fn run_google_vertex_generate() {
    let cfg = GoogleVertexConfig {
        provider_name: "google-vertex",
        provider_scope_name: "google-vertex".into(),
        base_url: "https://us-central1-aiplatform.googleapis.com/v1/projects/test/locations/us-central1/publishers/google/models".into(),
        headers: vec![],
        http: FixtureTransport::new(google_generate_response(), Vec::new()),
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        query_params: vec![],
        default_options: None,
    };
    let model = GoogleVertexLanguageModel::new("gemini-2.5-flash", cfg);
    let response = model
        .do_generate(v2t::CallOptions::new(google_matrix_prompt()))
        .await
        .expect("google vertex provider matrix generate");
    std::hint::black_box(response);
}

pub async fn run_bedrock_generate() {
    let cfg = BedrockConfig {
        provider_name: "amazon-bedrock.converse",
        provider_scope_name: "bedrock".into(),
        base_url: "https://bedrock.example".into(),
        headers: vec![],
        http: FixtureTransport::new(bedrock_generate_response(), Vec::new()),
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        default_options: None,
        auth: BedrockAuth::ApiKey {
            token: "test-token".into(),
        },
    };
    let model = BedrockLanguageModel::new("anthropic.claude-3-sonnet", cfg);
    let response = model
        .do_generate(v2t::CallOptions::new(matrix_prompt()))
        .await
        .expect("bedrock provider matrix generate");
    std::hint::black_box(response);
}

pub async fn run_gateway_generate() {
    let cfg = GatewayConfig {
        provider_name: "gateway",
        provider_scope_name: "gateway".into(),
        base_url: "https://ai-gateway.vercel.sh/v1/ai".into(),
        endpoint_path: Some("/language-model".into()),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        transport_cfg: TransportConfig::default(),
        default_options: None,
        request_defaults: None,
        auth: Some(GatewayAuth {
            token: "test-key".into(),
            method: GatewayAuthMethod::ApiKey,
        }),
    };
    let model = GatewayLanguageModel::new(
        "openai/gpt-4.1-mini",
        cfg,
        FixtureTransport::new(gateway_generate_response(), Vec::new()),
    );
    let response = model
        .do_generate(v2t::CallOptions::new(matrix_prompt()))
        .await
        .expect("gateway provider matrix generate");
    std::hint::black_box(response);
}

pub async fn run_openai_compatible_stream() {
    let cfg = OpenAICompatibleChatConfig {
        provider_scope_name: "openai-compatible".into(),
        base_url: "https://compat.example/v1".into(),
        headers: vec![("authorization".into(), "Bearer test-key".into())],
        http: FixtureTransport::new(json!({}), openai_compatible_stream_chunks()),
        transport_cfg: TransportConfig::default(),
        include_usage: true,
        supported_urls: HashMap::new(),
        query_params: vec![],
        supports_structured_outputs: true,
        default_options: None,
    };
    let model = OpenAICompatibleChatLanguageModel::new("grok-beta", cfg);
    let response = model
        .do_stream(v2t::CallOptions::new(matrix_prompt()))
        .await
        .expect("openai-compatible provider matrix stream");
    let parts = response
        .stream
        .try_collect::<Vec<_>>()
        .await
        .expect("openai-compatible provider matrix stream parts");
    std::hint::black_box(parts);
}

fn matrix_prompt() -> v2t::Prompt {
    vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Summarize the provider benchmark matrix.".into(),
            provider_options: None,
        }],
        provider_options: None,
    }]
}

fn google_matrix_prompt() -> v2t::Prompt {
    let assistant_provider_opts = Some(v2t::ProviderOptions::from([
        (
            "google".into(),
            HashMap::from([("thoughtSignature".into(), json!("sig-google"))]),
        ),
        (
            "google-vertex".into(),
            HashMap::from([("thoughtSignature".into(), json!("sig-vertex"))]),
        ),
    ]));
    vec![
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".to_string(),
                provider_options: None,
            }],
            provider_options: None,
        },
        v2t::PromptMessage::Assistant {
            content: vec![v2t::AssistantPart::Text {
                text: "world".to_string(),
                provider_options: assistant_provider_opts,
            }],
            provider_options: None,
        },
    ]
}

fn bedrock_generate_response() -> Value {
    json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    { "text": "Hello, World!" }
                ]
            }
        },
        "stopReason": "stop",
        "usage": {
            "inputTokens": 4,
            "outputTokens": 8,
            "totalTokens": 12
        }
    })
}

fn google_generate_response() -> Value {
    json!({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "ok"
                }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 2,
            "totalTokenCount": 6
        }
    })
}

fn gateway_generate_response() -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": "ok"
            }
        ],
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6
        }
    })
}

fn anthropic_stream_chunks() -> Vec<Bytes> {
    vec![
        anthropic_sse_chunk(
            Some("message_start"),
            json!({
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 0
                    }
                }
            }),
        ),
        anthropic_sse_chunk(
            Some("content_block_start"),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"}
            }),
        ),
        anthropic_sse_chunk(
            Some("content_block_delta"),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "ponder"
                }
            }),
        ),
        anthropic_sse_chunk(
            Some("content_block_delta"),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "signature_delta",
                    "signature": "sig-1"
                }
            }),
        ),
        anthropic_sse_chunk(
            Some("content_block_delta"),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "hello"
                }
            }),
        ),
        anthropic_sse_chunk(
            Some("content_block_start"),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "weather"
                }
            }),
        ),
        anthropic_sse_chunk(
            Some("content_block_delta"),
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"city\":\"SF\"}"
                }
            }),
        ),
        anthropic_sse_chunk(
            Some("message_delta"),
            json!({
                "type": "message_delta",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3
                }
            }),
        ),
        anthropic_sse_chunk(Some("message_stop"), json!({"type": "message_stop"})),
    ]
}

fn anthropic_sse_chunk(event: Option<&str>, payload: Value) -> Bytes {
    let mut chunk = String::new();
    if let Some(event) = event {
        chunk.push_str(&format!("event: {event}\n"));
    }
    chunk.push_str(&format!("data: {payload}\n\n"));
    Bytes::from(chunk)
}

fn openai_compatible_stream_chunks() -> Vec<Bytes> {
    vec![
        Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "id":"chat-1",
                "model":"grok-beta",
                "created":123,
                "choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]
            })
        )),
        Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "id":"chat-1",
                "choices":[{"index":0,"delta":{"content":"Hello"}}]
            })
        )),
        Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "choices":[{"delta":{},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
            })
        )),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ]
}

async fn collect_openai_fragmented_fixture_parts() -> Vec<v2t::StreamPart> {
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.1-mini",
        openai_responses_config(),
        FixtureTransport::new(
            minimal_text_response(),
            openai_fragmented_fixture_chunks().clone(),
        ),
        TransportConfig::default(),
    );
    let response = model
        .do_stream(stream_call_options("openai-mcp-tool-approval.1"))
        .await
        .expect("openai fragmented fixture stream");
    response
        .stream
        .try_collect::<Vec<_>>()
        .await
        .expect("openai fragmented fixture parts")
}

async fn run_openai_fragmented_fixture_stream() {
    let parts = collect_openai_fragmented_fixture_parts().await;
    std::hint::black_box(parts);
}

async fn collect_anthropic_scale_parts() -> Vec<v2t::StreamPart> {
    let cfg = AnthropicMessagesConfig {
        provider_name: "anthropic",
        provider_scope_name: "anthropic".into(),
        base_url: "https://api.example.com".into(),
        headers: vec![],
        http: FixtureTransport::new(json!({}), anthropic_scale_chunks().clone()),
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        default_options: None,
    };
    let model = AnthropicMessagesLanguageModel::new("claude-3-5-sonnet-20241022".to_string(), cfg);
    let response = model
        .do_stream(v2t::CallOptions::new(matrix_prompt()))
        .await
        .expect("anthropic scale stream");
    response
        .stream
        .try_collect::<Vec<_>>()
        .await
        .expect("anthropic scale parts")
}

async fn run_anthropic_scale_stream() {
    let parts = collect_anthropic_scale_parts().await;
    std::hint::black_box(parts);
}

async fn collect_gateway_scale_parts(backpressure: bool) -> Vec<v2t::StreamPart> {
    let cfg = GatewayConfig {
        provider_name: "gateway",
        provider_scope_name: "gateway".into(),
        base_url: "https://ai-gateway.vercel.sh/v1/ai".into(),
        endpoint_path: Some("/language-model".into()),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        transport_cfg: TransportConfig::default(),
        default_options: None,
        request_defaults: None,
        auth: Some(GatewayAuth {
            token: "test-key".into(),
            method: GatewayAuthMethod::ApiKey,
        }),
    };
    let model = GatewayLanguageModel::new(
        "openai/gpt-4.1-mini",
        cfg,
        FixtureTransport::new(json!({}), gateway_scale_chunks().clone()),
    );
    let mut options = v2t::CallOptions::new(matrix_prompt());
    options.include_raw_chunks = true;
    let response = model
        .do_stream(options)
        .await
        .expect("gateway scale stream");
    if backpressure {
        collect_stream_with_yields(response.stream, 32).await
    } else {
        response
            .stream
            .try_collect::<Vec<_>>()
            .await
            .expect("gateway scale parts")
    }
}

async fn run_gateway_scale_stream() {
    let parts = collect_gateway_scale_parts(false).await;
    std::hint::black_box(parts);
}

async fn collect_openai_compatible_adversarial_parts(backpressure: bool) -> Vec<v2t::StreamPart> {
    let cfg = OpenAICompatibleChatConfig {
        provider_scope_name: "openai-compatible".into(),
        base_url: "https://compat.example/v1".into(),
        headers: vec![("authorization".into(), "Bearer test-key".into())],
        http: FixtureTransport::new(json!({}), openai_compatible_adversarial_chunks().clone()),
        transport_cfg: TransportConfig::default(),
        include_usage: true,
        supported_urls: HashMap::new(),
        query_params: vec![],
        supports_structured_outputs: true,
        default_options: None,
    };
    let model = OpenAICompatibleChatLanguageModel::new("grok-beta", cfg);
    let mut options = v2t::CallOptions::new(matrix_prompt());
    options.tools = vec![v2t::Tool::Function(v2t::FunctionTool {
        r#type: v2t::FunctionToolType::Function,
        name: "test-tool".into(),
        description: Some("Benchmark tool".into()),
        input_schema: json!({
            "type":"object",
            "properties":{"value":{"type":"string"}},
            "required":["value"],
            "additionalProperties": false
        }),
        strict: Some(true),
        provider_options: None,
    })];
    let response = model
        .do_stream(options)
        .await
        .expect("openai-compatible adversarial stream");
    if backpressure {
        collect_stream_with_yields(response.stream, 24).await
    } else {
        response
            .stream
            .try_collect::<Vec<_>>()
            .await
            .expect("openai-compatible adversarial parts")
    }
}

async fn run_openai_compatible_adversarial_stream() {
    let parts = collect_openai_compatible_adversarial_parts(false).await;
    std::hint::black_box(parts);
}

async fn run_openai_parallel_replay() {
    let responses = join_all((0..4).map(|_| collect_openai_fragmented_fixture_parts())).await;
    std::hint::black_box(responses);
}

async fn run_anthropic_parallel_replay() {
    let responses = join_all((0..4).map(|_| collect_anthropic_scale_parts())).await;
    std::hint::black_box(responses);
}

async fn run_gateway_backpressure_replay() {
    let responses = join_all((0..4).map(|_| collect_gateway_scale_parts(true))).await;
    std::hint::black_box(responses);
}

async fn run_openai_compatible_backpressure_replay() {
    let responses = join_all((0..4).map(|_| collect_openai_compatible_adversarial_parts(true))).await;
    std::hint::black_box(responses);
}

async fn collect_stream_with_yields(
    stream: super::ai_sdk_rs::core::PartStream,
    yield_every: usize,
) -> Vec<v2t::StreamPart> {
    let mut parts = Vec::new();
    let mut seen = 0usize;
    futures_util::pin_mut!(stream);
    while let Some(part) = stream.next().await {
        parts.push(part.expect("stream part"));
        seen += 1;
        if seen % yield_every == 0 {
            tokio::task::yield_now().await;
        }
    }
    parts
}

fn small_provider_events() -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..32 {
        events.push(Event::TextDelta {
            delta: "chunk ".into(),
        });
    }
    events.push(Event::ReasoningStart {
        id: "reasoning:0".into(),
    });
    for _ in 0..16 {
        events.push(Event::ReasoningDelta {
            delta: "step ".into(),
        });
    }
    events.push(Event::ReasoningEnd);
    events.push(Event::ToolCallStart {
        id: "call_0".into(),
        name: "lookup_docs".into(),
    });
    for delta in ["{\"query\":", "\"streaming pipeline\"", ",\"limit\":", "5}"] {
        events.push(Event::ToolCallDelta {
            id: "call_0".into(),
            args_json: delta.into(),
        });
    }
    events.push(Event::ToolCallEnd {
        id: "call_0".into(),
    });
    events.push(Event::Usage {
        usage: TokenUsage::new(128, 32),
    });
    events.push(Event::Done);
    events
}

fn scaled_provider_events() -> Vec<Event> {
    let mut events = Vec::new();
    for idx in 0..1024 {
        events.push(Event::TextDelta {
            delta: format!("chunk-{idx:04}-{}", "alpha ".repeat(4)),
        });
    }
    events.push(Event::ReasoningStart {
        id: "reasoning:scale".into(),
    });
    for idx in 0..256 {
        events.push(Event::ReasoningDelta {
            delta: format!("reason-{idx:03}-{}", "trace ".repeat(3)),
        });
    }
    events.push(Event::ReasoningEnd);
    for call_idx in 0..8 {
        let id = format!("call_{call_idx}");
        events.push(Event::ToolCallStart {
            id: id.clone(),
            name: format!("tool_{call_idx}"),
        });
        for segment in 0..32 {
            events.push(Event::ToolCallDelta {
                id: id.clone(),
                args_json: format!(
                    "\"segment_{segment}\":\"{}\",",
                    "value".repeat(4)
                ),
            });
        }
        events.push(Event::ToolCallEnd { id });
    }
    events.push(Event::Usage {
        usage: TokenUsage::new(8192, 2048),
    });
    events.push(Event::Done);
    events
}

fn interleaved_provider_events() -> Vec<Event> {
    let mut events = Vec::new();
    events.push(Event::ReasoningStart {
        id: "reasoning:interleave".into(),
    });
    for idx in 0..192 {
        if idx % 3 == 0 {
            let id = format!("call_interleave_{idx}");
            events.push(Event::ToolCallStart {
                id: id.clone(),
                name: "lookup_docs".into(),
            });
            events.push(Event::ToolCallDelta {
                id: id.clone(),
                args_json: format!("{{\"query\":\"edge-{idx}\"}}"),
            });
            events.push(Event::ToolCallEnd { id });
        }
        events.push(Event::ReasoningDelta {
            delta: format!("branch-{idx}"),
        });
        events.push(Event::TextDelta {
            delta: format!("text-{idx}-{}", "beta".repeat(2)),
        });
    }
    events.push(Event::ReasoningEnd);
    events.push(Event::Usage {
        usage: TokenUsage::new(4096, 1024),
    });
    events.push(Event::Done);
    events
}

fn small_stream_parts() -> Vec<v2t::StreamPart> {
    vec![
        v2t::StreamPart::StreamStart {
            warnings: Vec::new(),
        },
        v2t::StreamPart::TextStart {
            id: "text:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::TextDelta {
            id: "text:0".into(),
            delta: "SDK ".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::TextDelta {
            id: "text:0".into(),
            delta: "overhead".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::TextEnd {
            id: "text:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningStart {
            id: "reasoning:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningDelta {
            id: "reasoning:0".into(),
            delta: "trace the translation path".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningSignature {
            signature: "sig_bench".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningEnd {
            id: "reasoning:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ToolCall(v2t::ToolCallPart {
            tool_call_id: "call_0".into(),
            tool_name: "lookup_docs".into(),
            input: json!({
                "query": "streaming pipeline"
            })
            .to_string(),
            provider_executed: false,
            provider_metadata: None,
            dynamic: false,
            provider_options: None,
        }),
        v2t::StreamPart::ToolResult {
            tool_call_id: "call_0".into(),
            tool_name: "lookup_docs".into(),
            result: json!({
                "matches": 5
            }),
            is_error: false,
            preliminary: false,
            provider_metadata: None,
        },
        v2t::StreamPart::Finish {
            usage: v2t::Usage {
                input_tokens: Some(128),
                output_tokens: Some(64),
                total_tokens: Some(192),
                reasoning_tokens: Some(16),
                cached_input_tokens: Some(32),
            },
            finish_reason: v2t::FinishReason::Stop,
            provider_metadata: None,
        },
    ]
}

fn scaled_stream_parts() -> Vec<v2t::StreamPart> {
    let mut parts = vec![
        v2t::StreamPart::StreamStart {
            warnings: Vec::new(),
        },
        v2t::StreamPart::TextStart {
            id: "text:scale".into(),
            provider_metadata: None,
        },
    ];
    for idx in 0..1536 {
        parts.push(v2t::StreamPart::TextDelta {
            id: "text:scale".into(),
            delta: format!("chunk-{idx:04}-{}", "payload ".repeat(3)),
            provider_metadata: None,
        });
    }
    parts.push(v2t::StreamPart::TextEnd {
        id: "text:scale".into(),
        provider_metadata: None,
    });
    parts.push(v2t::StreamPart::ReasoningStart {
        id: "reasoning:scale".into(),
        provider_metadata: None,
    });
    for idx in 0..384 {
        parts.push(v2t::StreamPart::ReasoningDelta {
            id: "reasoning:scale".into(),
            delta: format!("reason-{idx:03}-{}", "detail ".repeat(2)),
            provider_metadata: None,
        });
    }
    parts.push(v2t::StreamPart::ReasoningSignature {
        signature: "sig_scale".into(),
        provider_metadata: None,
    });
    parts.push(v2t::StreamPart::ReasoningEnd {
        id: "reasoning:scale".into(),
        provider_metadata: None,
    });
    for idx in 0..24 {
        parts.push(v2t::StreamPart::ToolCall(v2t::ToolCallPart {
            tool_call_id: format!("call_{idx}"),
            tool_name: "lookup_docs".into(),
            input: json!({
                "query": format!("streaming-{idx}")
            })
            .to_string(),
            provider_executed: false,
            provider_metadata: None,
            dynamic: false,
            provider_options: None,
        }));
        parts.push(v2t::StreamPart::ToolResult {
            tool_call_id: format!("call_{idx}"),
            tool_name: "lookup_docs".into(),
            result: json!({
                "matches": idx + 1,
                "sources": ["docs", "fixtures", "bench"]
            }),
            is_error: false,
            preliminary: idx % 2 == 0,
            provider_metadata: None,
        });
    }
    parts.push(v2t::StreamPart::Finish {
        usage: v2t::Usage {
            input_tokens: Some(8192),
            output_tokens: Some(4096),
            total_tokens: Some(12288),
            reasoning_tokens: Some(768),
            cached_input_tokens: Some(1024),
        },
        finish_reason: v2t::FinishReason::Stop,
        provider_metadata: None,
    });
    parts
}

fn large_wrapped_json() -> String {
    let mut payload = String::from("tool-call-start<<<{\"summary\":\"");
    payload.push_str(&"bench-overhead ".repeat(128));
    payload.push_str("\",\"sources\":[");
    for idx in 0..256 {
        if idx > 0 {
            payload.push(',');
        }
        payload.push('"');
        payload.push_str(&format!("source-{idx:03}"));
        payload.push('"');
    }
    payload.push_str("],\"nested\":{\"segments\":[");
    for idx in 0..128 {
        if idx > 0 {
            payload.push(',');
        }
        payload.push_str(&format!(
            "{{\"id\":{idx},\"text\":\"{}\"}}",
            "delta ".repeat(12)
        ));
    }
    payload.push_str("]}}>>>tool-call-end");
    payload
}

fn truncated_large_wrapped_json() -> String {
    let mut payload = large_wrapped_json();
    payload.truncate(payload.len().saturating_sub(57));
    payload
}

fn openai_fragmented_fixture_chunks() -> &'static Vec<Bytes> {
    static CHUNKS: OnceLock<Vec<Bytes>> = OnceLock::new();
    CHUNKS.get_or_init(|| {
        fragment_chunks(
            stream_fixture_chunks("openai-mcp-tool-approval.1"),
            &[1, 2, 3, 5, 8, 13, 21],
        )
    })
}

fn anthropic_scale_chunks() -> &'static Vec<Bytes> {
    static CHUNKS: OnceLock<Vec<Bytes>> = OnceLock::new();
    CHUNKS.get_or_init(|| {
        let mut frames = vec![
            anthropic_sse_chunk(
                Some("message_start"),
                json!({
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 64, "output_tokens": 0}}
                }),
            ),
            anthropic_sse_chunk(
                Some("content_block_start"),
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking"}
                }),
            ),
        ];

        for idx in 0..192 {
            frames.push(anthropic_sse_chunk(
                Some("content_block_delta"),
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": format!("ponder-{idx:03}-{}", "trace ".repeat(6))
                    }
                }),
            ));
        }
        frames.push(anthropic_sse_chunk(
            Some("content_block_delta"),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "signature_delta",
                    "signature": "sig-scale"
                }
            }),
        ));
        for idx in 0..384 {
            frames.push(anthropic_sse_chunk(
                Some("content_block_delta"),
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": format!("segment-{idx:03}-{}", "hello ".repeat(8))
                    }
                }),
            ));
        }
        frames.push(anthropic_sse_chunk(
            Some("content_block_start"),
            json!({
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-scale-1",
                    "name": "weather"
                }
            }),
        ));
        for idx in 0..96 {
            frames.push(anthropic_sse_chunk(
                Some("content_block_delta"),
                json!({
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": format!("\"segment_{idx}\":\"{}\",", "value".repeat(6))
                    }
                }),
            ));
        }
        frames.push(anthropic_sse_chunk(
            Some("message_delta"),
            json!({
                "type": "message_delta",
                "usage": {"input_tokens": 64, "output_tokens": 512}
            }),
        ));
        frames.push(anthropic_sse_chunk(
            Some("message_stop"),
            json!({"type": "message_stop"}),
        ));

        fragment_chunks(frames, &[2, 3, 5, 8, 13, 21, 34])
    })
}

fn gateway_scale_chunks() -> &'static Vec<Bytes> {
    static CHUNKS: OnceLock<Vec<Bytes>> = OnceLock::new();
    CHUNKS.get_or_init(|| {
        let mut frames = vec![
            gateway_sse_chunk(json!({
                "type": "stream-start",
                "warnings": [{"type": "other", "message": "gateway stress"}]
            })),
            gateway_sse_chunk(json!({
                "type": "response-metadata",
                "id": "resp-gateway-scale",
                "modelId": "gateway-model",
                "timestamp": "2026-01-02T03:04:05Z"
            })),
        ];

        for idx in 0..128 {
            frames.push(gateway_sse_chunk(json!({
                "type": "reasoning-delta",
                "delta": format!("reason-{idx:03}-{}", "think ".repeat(4)),
                "providerMetadata": {"gateway": {"phase": "reasoning"}}
            })));
        }
        for idx in 0..512 {
            frames.push(gateway_sse_chunk(json!({
                "type": "text-delta",
                "delta": format!("text-{idx:03}-{}", "segment ".repeat(5)),
                "providerMetadata": {"gateway": {"phase": "text"}}
            })));
            if idx % 64 == 0 {
                frames.push(gateway_sse_chunk(json!({
                    "type": "raw",
                    "rawValue": {"upstream": format!("frame-{idx}")}
                })));
            }
        }

        frames.push(gateway_sse_chunk(json!({
            "type": "tool-input-start",
            "id": "call-scale",
            "toolName": "weather",
            "providerExecuted": false
        })));
        for idx in 0..96 {
            frames.push(gateway_sse_chunk(json!({
                "type": "tool-input-delta",
                "id": "call-scale",
                "delta": format!("{{\"segment\":{idx},\"value\":\"{}\"}}", "x".repeat(12)),
                "providerExecuted": false
            })));
        }
        frames.push(gateway_sse_chunk(json!({
            "type": "tool-input-end",
            "id": "call-scale",
            "providerExecuted": false
        })));
        frames.push(gateway_sse_chunk(json!({
            "type": "tool-call",
            "toolCallId": "call-scale",
            "toolName": "weather",
            "input": {"city": "SF"},
            "providerExecuted": false
        })));
        frames.push(gateway_sse_chunk(json!({
            "type": "finish",
            "finishReason": "tool-calls",
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": 256,
                "total_tokens": 384
            },
            "providerMetadata": {"gateway": {"finishSource": "gateway"}}
        })));
        frames.push(Bytes::from_static(b"data: [DONE]\n\n"));

        fragment_chunks(frames, &[1, 4, 7, 11, 17, 29])
    })
}

fn openai_compatible_adversarial_chunks() -> &'static Vec<Bytes> {
    static CHUNKS: OnceLock<Vec<Bytes>> = OnceLock::new();
    CHUNKS.get_or_init(|| {
        let mut frames = vec![Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "id":"chat-adversarial",
                "model":"grok-beta",
                "created":456,
                "choices":[{
                    "index":0,
                    "delta":{
                        "role":"assistant",
                        "content":"",
                        "tool_calls":[{
                            "index":0,
                            "id":"call_adversarial",
                            "type":"function",
                            "function":{"name":"test-tool","arguments":""}
                        }]
                    },
                    "finish_reason":null
                }]
            })
        ))];

        for segment in [
            "{\"",
            "value",
            "\":\"",
            "adv",
            "ersarial",
            " payload",
            "\"}",
        ] {
            frames.push(Bytes::from(format!(
                "data: {}\n\n",
                json!({
                    "choices":[{
                        "delta":{"tool_calls":[{"index":0,"function":{"arguments":segment}}]},
                        "finish_reason":null
                    }]
                })
            )));
        }

        for idx in 0..128 {
            frames.push(Bytes::from(format!(
                "data: {}\n\n",
                json!({
                    "choices":[{"delta":{"content": format!("chunk-{idx:03}-{}", "text ".repeat(4))},"finish_reason":null}]
                })
            )));
        }

        frames.push(Bytes::from_static(b"data: {not-json}\n\n"));
        frames.push(Bytes::from(format!(
            "data: {}\n\n",
            json!({
                "choices":[{"delta":{},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":32,"completion_tokens":64,"total_tokens":96}
            })
        )));
        frames.push(Bytes::from_static(b"data: [DONE]\n\n"));

        fragment_chunks(frames, &[2, 5, 9, 14, 23])
    })
}

fn gateway_sse_chunk(payload: Value) -> Bytes {
    Bytes::from(format!("data: {payload}\n\n"))
}

fn fragment_chunks(chunks: Vec<Bytes>, pattern: &[usize]) -> Vec<Bytes> {
    let total_len = chunks.iter().map(Bytes::len).sum();
    let mut raw = Vec::with_capacity(total_len);
    for chunk in chunks {
        raw.extend_from_slice(&chunk);
    }
    fragment_bytes(&raw, pattern)
}

fn fragment_bytes(raw: &[u8], pattern: &[usize]) -> Vec<Bytes> {
    let mut fragments = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while start < raw.len() {
        let len = pattern[idx % pattern.len()].min(raw.len() - start);
        fragments.push(Bytes::copy_from_slice(&raw[start..start + len]));
        start += len;
        idx += 1;
    }
    fragments
}

fn bytes_len(chunks: &[Bytes]) -> u64 {
    chunks.iter().map(|chunk| chunk.len() as u64).sum()
}
