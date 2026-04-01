use std::collections::HashMap;

use bytes::Bytes;
use futures_util::TryStreamExt;
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
use super::fixture_replay::FixtureTransport;
use super::{minimal_text_response, simple_call_options};

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
