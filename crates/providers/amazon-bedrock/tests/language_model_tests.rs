use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::LanguageModel;
use crate::ai_sdk_providers_amazon_bedrock::config::{BedrockAuth, BedrockConfig};
use crate::ai_sdk_providers_amazon_bedrock::language_model::BedrockLanguageModel;
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::stream;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct TestTransport {
    response_json: Arc<Mutex<serde_json::Value>>,
}

impl TestTransport {
    fn new(response_json: serde_json::Value) -> Self {
        Self {
            response_json: Arc::new(Mutex::new(response_json)),
        }
    }
}

struct TestStreamResponse;

#[async_trait]
impl HttpTransport for TestTransport {
    type StreamResponse = TestStreamResponse;

    fn into_stream(
        _resp: Self::StreamResponse,
    ) -> (
        Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
        Vec<(String, String)>,
    ) {
        (Box::pin(stream::iter(vec![])), vec![])
    }

    async fn post_json_stream(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &serde_json::Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        Err(TransportError::Other("post_json_stream unused".into()))
    }

    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &serde_json::Value,
        _cfg: &TransportConfig,
    ) -> Result<(serde_json::Value, Vec<(String, String)>), TransportError> {
        Ok((self.response_json.lock().unwrap().clone(), vec![]))
    }
}

fn build_model(response_json: serde_json::Value) -> BedrockLanguageModel<TestTransport> {
    let transport = TestTransport::new(response_json);
    let cfg = BedrockConfig {
        provider_name: "amazon-bedrock.converse",
        provider_scope_name: "bedrock".into(),
        base_url: "https://bedrock.example".into(),
        headers: vec![],
        http: transport,
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        default_options: None,
        auth: BedrockAuth::ApiKey {
            token: "test-token".into(),
        },
    };
    BedrockLanguageModel::new("anthropic.claude-3-sonnet", cfg)
}

fn base_prompt() -> Vec<v2t::PromptMessage> {
    vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }]
}

#[tokio::test]
async fn text_response_maps_to_text_content() {
    let response = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    { "text": "Hello, World!" }
                ]
            }
        },
        "stopReason": "stop"
    });
    let model = build_model(response);

    let result = model
        .do_generate(v2t::CallOptions {
            prompt: base_prompt(),
            ..Default::default()
        })
        .await
        .expect("generate");

    assert_eq!(result.content.len(), 1);
    match &result.content[0] {
        v2t::Content::Text {
            text,
            provider_metadata,
        } => {
            assert_eq!(text, "Hello, World!");
            assert!(provider_metadata.is_none());
        }
        other => panic!("unexpected content variant: {:?}", other),
    }
}

#[tokio::test]
async fn tool_use_maps_to_tool_call() {
    let response = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    { "toolUse": { "toolUseId": "call-1", "name": "demo", "input": { "foo": "bar" } } }
                ]
            }
        },
        "stopReason": "tool_use"
    });
    let model = build_model(response);

    let result = model
        .do_generate(v2t::CallOptions {
            prompt: base_prompt(),
            ..Default::default()
        })
        .await
        .expect("generate");

    assert_eq!(result.content.len(), 1);
    match &result.content[0] {
        v2t::Content::ToolCall(call) => {
            assert_eq!(call.tool_call_id, "call-1");
            assert_eq!(call.tool_name, "demo");
            assert_eq!(call.input, "{\"foo\":\"bar\"}");
            assert!(!call.provider_executed);
            assert!(call.provider_options.is_none());
        }
        other => panic!("unexpected content variant: {:?}", other),
    }
}

#[tokio::test]
async fn tool_use_maps_to_text_when_json_response_tool_enabled() {
    let response = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    { "toolUse": { "toolUseId": "call-1", "name": "json", "input": { "foo": "bar" } } }
                ]
            }
        },
        "stopReason": "stop"
    });
    let model = build_model(response);

    let result = model
        .do_generate(v2t::CallOptions {
            prompt: base_prompt(),
            response_format: Some(v2t::ResponseFormat::Json {
                schema: Some(json!({"type": "object"})),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("generate");

    assert_eq!(result.content.len(), 1);
    match &result.content[0] {
        v2t::Content::Text {
            text,
            provider_metadata,
        } => {
            assert_eq!(text, "{\"foo\":\"bar\"}");
            assert!(provider_metadata.is_none());
        }
        other => panic!("unexpected content variant: {:?}", other),
    }
}

#[tokio::test]
async fn usage_and_finish_reason_are_mapped() {
    let response = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    { "text": "Done" }
                ]
            }
        },
        "stopReason": "stop_sequence",
        "usage": {
            "inputTokens": 4,
            "outputTokens": 34,
            "totalTokens": 38,
            "cacheReadInputTokens": 2
        }
    });
    let model = build_model(response);

    let result = model
        .do_generate(v2t::CallOptions {
            prompt: base_prompt(),
            ..Default::default()
        })
        .await
        .expect("generate");

    assert!(matches!(result.finish_reason, v2t::FinishReason::Stop));
    assert_eq!(result.usage.input_tokens, Some(4));
    assert_eq!(result.usage.output_tokens, Some(34));
    assert_eq!(result.usage.total_tokens, Some(38));
    assert_eq!(result.usage.cached_input_tokens, Some(2));
    assert!(result.usage.reasoning_tokens.is_none());
}

#[tokio::test]
async fn provider_metadata_includes_trace_usage_and_json_flag() {
    let trace = json!({
        "guardrail": {
            "inputAssessment": {
                "id": {
                    "filters": [{ "action": "BLOCKED" }]
                }
            }
        }
    });
    let response = json!({
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    { "text": "Done" }
                ]
            }
        },
        "stopReason": "stop",
        "trace": trace,
        "usage": {
            "inputTokens": 4,
            "outputTokens": 34,
            "totalTokens": 38,
            "cacheWriteInputTokens": 3
        }
    });
    let model = build_model(response);

    let result = model
        .do_generate(v2t::CallOptions {
            prompt: base_prompt(),
            response_format: Some(v2t::ResponseFormat::Json {
                schema: Some(json!({"type": "object"})),
                name: None,
                description: None,
            }),
            ..Default::default()
        })
        .await
        .expect("generate");

    let bedrock = result
        .provider_metadata
        .as_ref()
        .and_then(|m| m.get("bedrock"))
        .expect("bedrock metadata");

    assert_eq!(
        bedrock.get("trace"),
        Some(&json!({
            "guardrail": {
                "inputAssessment": {
                    "id": { "filters": [{ "action": "BLOCKED" }] }
                }
            }
        }))
    );

    let usage_value = bedrock
        .get("usage")
        .and_then(|value| value.as_object())
        .expect("usage object expected");
    assert_eq!(usage_value.get("cacheWriteInputTokens"), Some(&json!(3)));

    assert_eq!(bedrock.get("isJsonResponseFromTool"), Some(&json!(true)));
}
