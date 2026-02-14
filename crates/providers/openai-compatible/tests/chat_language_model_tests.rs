use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::json::without_null_fields;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_providers_openai_compatible::chat::language_model::{
    OpenAICompatibleChatConfig, OpenAICompatibleChatLanguageModel,
};
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::stream;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

const NEW_PROVIDER_SCOPES: &[&str] = &[
    "groq",
    "xai",
    "deepseek",
    "mistral",
    "togetherai",
    "fireworks-ai",
    "deepinfra",
    "openrouter",
    "perplexity",
];

#[derive(Clone)]
struct TestTransport {
    chunks: Arc<Vec<Bytes>>,
    response_headers: Arc<Mutex<Vec<(String, String)>>>,
    last_body: Arc<Mutex<Option<serde_json::Value>>>,
    last_headers: Arc<Mutex<Option<Vec<(String, String)>>>>,
    stream_error: Arc<Mutex<Option<TransportError>>>,
}

impl TestTransport {
    fn new(chunks: Vec<Bytes>) -> Self {
        Self {
            chunks: Arc::new(chunks),
            response_headers: Arc::new(Mutex::new(vec![])),
            last_body: Arc::new(Mutex::new(None)),
            last_headers: Arc::new(Mutex::new(None)),
            stream_error: Arc::new(Mutex::new(None)),
        }
    }

    fn with_response_headers(self, headers: Vec<(String, String)>) -> Self {
        *self.response_headers.lock().unwrap() = headers;
        self
    }

    fn with_stream_error(self, error: TransportError) -> Self {
        *self.stream_error.lock().unwrap() = Some(error);
        self
    }

    fn last_body(&self) -> Option<serde_json::Value> {
        self.last_body.lock().unwrap().clone()
    }
}

struct TestStreamResponse {
    headers: Vec<(String, String)>,
    chunks: Vec<Bytes>,
}

#[async_trait]
impl HttpTransport for TestTransport {
    type StreamResponse = TestStreamResponse;

    fn into_stream(
        resp: Self::StreamResponse,
    ) -> (
        Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
        Vec<(String, String)>,
    ) {
        let stream = stream::iter(resp.chunks.into_iter().map(Ok));
        (Box::pin(stream), resp.headers)
    }

    async fn post_json_stream(
        &self,
        _url: &str,
        headers: &[(String, String)],
        body: &serde_json::Value,
        cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        if let Some(err) = self.stream_error.lock().unwrap().take() {
            return Err(err);
        }
        let cleaned = if cfg.strip_null_fields {
            without_null_fields(body)
        } else {
            body.clone()
        };
        *self.last_body.lock().unwrap() = Some(cleaned);
        *self.last_headers.lock().unwrap() = Some(headers.to_vec());
        Ok(TestStreamResponse {
            headers: self.response_headers.lock().unwrap().clone(),
            chunks: self.chunks.as_ref().clone(),
        })
    }

    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &serde_json::Value,
        _cfg: &TransportConfig,
    ) -> Result<(serde_json::Value, Vec<(String, String)>), TransportError> {
        Err(TransportError::Other("post_json unused".into()))
    }
}

fn json_chunk(val: serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {val}\n\n"))
}

fn build_model(
    chunks: Vec<Bytes>,
    supports_structured_outputs: bool,
) -> (
    OpenAICompatibleChatLanguageModel<TestTransport>,
    TestTransport,
) {
    build_model_for_scope(chunks, supports_structured_outputs, "test-provider")
}

fn build_model_for_scope(
    chunks: Vec<Bytes>,
    supports_structured_outputs: bool,
    provider_scope_name: &str,
) -> (
    OpenAICompatibleChatLanguageModel<TestTransport>,
    TestTransport,
) {
    let transport = TestTransport::new(chunks);
    let cfg = OpenAICompatibleChatConfig {
        provider_scope_name: provider_scope_name.into(),
        base_url: "https://my.api.com/v1".into(),
        headers: vec![("authorization".into(), "Bearer test-api-key".into())],
        http: transport.clone(),
        transport_cfg: TransportConfig::default(),
        include_usage: true,
        supported_urls: HashMap::new(),
        query_params: vec![],
        supports_structured_outputs,
        default_options: None,
    };
    (
        OpenAICompatibleChatLanguageModel::new("grok-beta", cfg),
        transport,
    )
}

#[tokio::test]
async fn builds_request_body_with_user_and_messages() {
    let (model, transport) = build_model(vec![], false);

    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "test-provider".into(),
        HashMap::from([("user".into(), json!("test-user-id"))]),
    );

    let _ = model
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "Hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options,
            ..Default::default()
        })
        .await
        .expect("stream response");

    let body = transport.last_body().expect("sent body");
    assert_eq!(body.get("model"), Some(&json!("grok-beta")));
    assert_eq!(body.get("user"), Some(&json!("test-user-id")));
    assert_eq!(
        body.get("messages"),
        Some(&json!([{ "role": "user", "content": "Hello"}]))
    );
    assert_eq!(body.get("stream"), Some(&json!(true)));
    assert_eq!(
        body.get("stream_options"),
        Some(&json!({"include_usage": true}))
    );
}

#[tokio::test]
async fn response_format_json_schema_when_structured_outputs_enabled() {
    let chunks = vec![
        json_chunk(json!({
            "choices":[{"delta":{"content":""},"finish_reason":null}]
        })),
        json_chunk(json!({
            "choices":[{"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
        })),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let (model, transport) = build_model(chunks, true);
    let options = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        response_format: Some(v2t::ResponseFormat::Json {
            schema: Some(json!({"type":"object","properties":{"value":{"type":"string"}}})),
            name: None,
            description: None,
        }),
        ..Default::default()
    };

    let _ = model.do_generate(options).await.expect("generate");
    let body = transport.last_body().expect("sent body");

    assert_eq!(
        body.get("response_format"),
        Some(&json!({
            "type":"json_schema",
            "json_schema":{
                "name":"response",
                "schema":{"type":"object","properties":{"value":{"type":"string"}}
                }
            }
        }))
    );
}

#[tokio::test]
async fn response_format_json_object_and_warning_when_structured_outputs_disabled() {
    let chunks = vec![
        json_chunk(json!({
            "choices":[{"delta":{"content":"Hello"},"finish_reason":null}]
        })),
        json_chunk(json!({
            "choices":[{"delta":{},"finish_reason":"stop"}]
        })),
        Bytes::from_static(b"data: [DONE]\n\n"),
    ];
    let (model, transport) = build_model(chunks, false);
    let options = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        response_format: Some(v2t::ResponseFormat::Json {
            schema: Some(json!({"type":"object"})),
            name: None,
            description: None,
        }),
        ..Default::default()
    };

    let res = model.do_generate(options).await.expect("generate");
    let body = transport.last_body().expect("sent body");
    assert_eq!(
        body.get("response_format"),
        Some(&json!({"type":"json_object"}))
    );
    assert!(
        res.warnings
            .iter()
            .any(|w| matches!(w, v2t::CallWarning::UnsupportedSetting { setting, .. } if setting == "responseFormat"))
    );
}

#[tokio::test]
async fn includes_tools_and_tool_choice_in_request_body() {
    let (model, transport) = build_model(vec![], false);
    let tool = v2t::FunctionTool {
        r#type: v2t::FunctionToolType::Function,
        name: "test-tool".into(),
        description: None,
        input_schema: json!({
            "type":"object",
            "properties":{"value":{"type":"string"}},
            "required":["value"],
            "additionalProperties": false
        }),
        strict: None,
        provider_options: None,
    };

    let _ = model
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "Hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(tool.clone())],
            tool_choice: Some(v2t::ToolChoice::Tool {
                name: "test-tool".into(),
            }),
            ..Default::default()
        })
        .await
        .expect("stream response");

    let body = transport.last_body().expect("sent body");
    assert_eq!(
        body.get("tools"),
        Some(&json!([{
            "type":"function",
            "function":{
                "name":"test-tool",
                "parameters":{
                    "type":"object",
                    "properties":{"value":{"type":"string"}},
                    "required":["value"],
                    "additionalProperties": false
                }
            }
        }]))
    );
    assert_eq!(
        body.get("tool_choice"),
        Some(&json!({"type":"function","function":{"name":"test-tool"}}))
    );
}

#[tokio::test]
async fn merges_openai_compatible_and_provider_specific_options() {
    let (model, transport) = build_model(vec![], false);
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai-compatible".into(),
        HashMap::from([
            ("user".into(), json!("base-user")),
            ("reasoningEffort".into(), json!("low")),
            ("textVerbosity".into(), json!("low")),
            ("baseOnly".into(), json!("ignored")),
        ]),
    );
    provider_options.insert(
        "test-provider".into(),
        HashMap::from([
            ("reasoningEffort".into(), json!("high")),
            ("textVerbosity".into(), json!("high")),
            ("someCustomOption".into(), json!("test-value")),
        ]),
    );
    provider_options.insert(
        "other-provider".into(),
        HashMap::from([("ignored".into(), json!(true))]),
    );

    let _ = model
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "Hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options,
            ..Default::default()
        })
        .await
        .expect("stream response");

    let body = transport.last_body().expect("sent body");
    assert_eq!(body.get("user"), Some(&json!("base-user")));
    assert_eq!(body.get("reasoning_effort"), Some(&json!("high")));
    assert_eq!(body.get("verbosity"), Some(&json!("high")));
    assert_eq!(body.get("someCustomOption"), Some(&json!("test-value")));
    assert!(!body.as_object().unwrap().contains_key("baseOnly"));
    assert!(!body.as_object().unwrap().contains_key("textVerbosity"));
    assert!(!body.as_object().unwrap().contains_key("ignored"));
}

#[tokio::test]
async fn newly_routed_scopes_serialize_request_body_with_identical_precedence() {
    for scope in NEW_PROVIDER_SCOPES {
        let (model, transport) = build_model_for_scope(vec![], false, scope);

        let mut provider_options = v2t::ProviderOptions::new();
        provider_options.insert(
            "openai-compatible".into(),
            HashMap::from([
                ("user".into(), json!("base-user")),
                ("reasoningEffort".into(), json!("low")),
                ("textVerbosity".into(), json!("low")),
                ("baseOnly".into(), json!("ignored")),
            ]),
        );
        provider_options.insert(
            (*scope).into(),
            HashMap::from([
                ("reasoningEffort".into(), json!("high")),
                ("textVerbosity".into(), json!("high")),
                ("providerTag".into(), json!(scope)),
            ]),
        );

        let _ = model
            .do_stream(v2t::CallOptions {
                prompt: vec![v2t::PromptMessage::User {
                    content: vec![v2t::UserPart::Text {
                        text: "Hello".into(),
                        provider_options: None,
                    }],
                    provider_options: None,
                }],
                provider_options,
                ..Default::default()
            })
            .await
            .unwrap_or_else(|_| panic!("expected stream request serialization for {scope}"));

        let body = transport.last_body().expect("sent body");
        assert_eq!(body.get("model"), Some(&json!("grok-beta")));
        assert_eq!(body.get("user"), Some(&json!("base-user")));
        assert_eq!(body.get("reasoning_effort"), Some(&json!("high")));
        assert_eq!(body.get("verbosity"), Some(&json!("high")));
        assert_eq!(body.get("providerTag"), Some(&json!(scope)));
        assert!(!body.as_object().unwrap().contains_key("baseOnly"));
    }
}

#[tokio::test]
async fn newly_routed_scopes_map_transport_errors_identically() {
    for scope in NEW_PROVIDER_SCOPES {
        let transport = TestTransport::new(vec![]).with_stream_error(TransportError::HttpStatus {
            status: 400,
            body: format!(r#"{{"error":{{"message":"{scope} bad request"}}}}"#),
            retry_after_ms: None,
            sanitized: "http status 400".into(),
            headers: Vec::new(),
        });
        let cfg = OpenAICompatibleChatConfig {
            provider_scope_name: (*scope).into(),
            base_url: "https://my.api.com/v1".into(),
            headers: vec![("authorization".into(), "Bearer test-api-key".into())],
            http: transport,
            transport_cfg: TransportConfig::default(),
            include_usage: true,
            supported_urls: HashMap::new(),
            query_params: vec![],
            supports_structured_outputs: false,
            default_options: None,
        };
        let model = OpenAICompatibleChatLanguageModel::new("grok-beta", cfg);
        let result = model
            .do_stream(v2t::CallOptions {
                prompt: vec![v2t::PromptMessage::User {
                    content: vec![v2t::UserPart::Text {
                        text: "Hello".into(),
                        provider_options: None,
                    }],
                    provider_options: None,
                }],
                ..Default::default()
            })
            .await;

        let err = match result {
            Ok(_) => panic!("expected upstream error for {scope}"),
            Err(err) => err,
        };

        match err {
            SdkError::Upstream {
                status, message, ..
            } => {
                assert_eq!(status, 400);
                assert_eq!(message, format!("{scope} bad request"));
            }
            other => panic!("expected upstream error for {scope}, got {other:?}"),
        }
    }
}
