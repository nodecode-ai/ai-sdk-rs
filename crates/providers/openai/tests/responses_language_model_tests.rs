use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::LanguageModel;
use crate::ai_sdk_providers_openai::config::OpenAIConfig;
use crate::ai_sdk_providers_openai::responses::language_model::OpenAIResponsesLanguageModel;
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::stream;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct TestTransport {
    last_body: Arc<Mutex<Option<Value>>>,
    json_response: Arc<Mutex<Option<Value>>>,
}

impl TestTransport {
    fn new() -> Self {
        Self {
            last_body: Arc::new(Mutex::new(None)),
            json_response: Arc::new(Mutex::new(None)),
        }
    }

    fn last_body(&self) -> Option<Value> {
        self.last_body.lock().unwrap().clone()
    }

    fn with_json_response(self, response: Value) -> Self {
        *self.json_response.lock().unwrap() = Some(response);
        self
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
        body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        *self.last_body.lock().unwrap() = Some(body.clone());
        Ok(TestStreamResponse)
    }

    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        *self.last_body.lock().unwrap() = Some(body.clone());
        let response = self
            .json_response
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| TransportError::Other("post_json unused".into()))?;
        Ok((response, vec![]))
    }
}

fn provider_options_fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/responses_provider_options_request.json"
    ))
    .expect("provider options fixture")
}

fn item_reference_fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/responses_item_reference_request.json"
    ))
    .expect("item reference fixture")
}

fn reasoning_summary_fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/responses_reasoning_summary_request.json"
    ))
    .expect("reasoning summary fixture")
}

fn provider_tool_outputs_fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/responses_provider_tool_outputs_request.json"
    ))
    .expect("provider tool outputs fixture")
}

fn openai_error_fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/openai-error.1.json"))
        .expect("openai error fixture")
}

fn local_shell_response_fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/openai-local-shell-tool.1.json"))
        .expect("local shell response fixture")
}

fn function_tool_for_strict_passthrough(strict: Option<bool>) -> v2t::FunctionTool {
    let provider_options = strict.map(|value| {
        v2t::ProviderOptions::from([(
            "openai".into(),
            HashMap::from([("strict".into(), json!(value))]),
        )])
    });
    v2t::FunctionTool {
        r#type: v2t::FunctionToolType::Function,
        name: "strict-tool".into(),
        description: Some("strict passthrough".into()),
        input_schema: json!({
            "type":"object",
            "properties":{"value":{"type":"string"}},
            "required":["value"],
            "additionalProperties": false
        }),
        strict,
        provider_options,
    }
}

async fn request_body_for_function_tool(function_tool: v2t::FunctionTool) -> Value {
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        tools: vec![v2t::Tool::Function(function_tool)],
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    transport.last_body().expect("request body")
}

#[tokio::test]
async fn request_body_includes_responses_provider_options() {
    let prompt = vec![
        v2t::PromptMessage::System {
            content: "System prompt".into(),
            provider_options: None,
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
    ];
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([
            ("conversation".into(), json!("conv_123")),
            ("maxToolCalls".into(), json!(2)),
            ("promptCacheKey".into(), json!("cache-key")),
            ("promptCacheRetention".into(), json!("24h")),
            ("truncation".into(), json!("disabled")),
            ("systemMessageMode".into(), json!("system")),
            ("forceReasoning".into(), json!(true)),
            ("reasoningEffort".into(), json!("low")),
        ]),
    );
    let opts = v2t::CallOptions {
        prompt,
        response_format: Some(v2t::ResponseFormat::Json {
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "ok": { "type": "boolean" }
                },
                "required": ["ok"],
                "additionalProperties": false
            })),
            name: None,
            description: Some("Return a JSON object".into()),
        }),
        provider_options,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body, provider_options_fixture());
}

#[tokio::test]
async fn request_body_includes_provider_tools_and_tool_choice() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Search please".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let provider_tool = v2t::ProviderTool {
        r#type: v2t::ProviderToolType::Provider,
        id: "openai.web_search".into(),
        name: "search".into(),
        args: json!({
            "searchContextSize": "medium",
            "userLocation": { "type": "approximate", "country": "US" },
            "externalWebAccess": true,
            "filters": { "allowedDomains": ["example.com"] }
        }),
    };
    let function_tool = v2t::FunctionTool {
        r#type: v2t::FunctionToolType::Function,
        name: "test-tool".into(),
        description: Some("test".into()),
        input_schema: json!({
            "type":"object",
            "properties":{"value":{"type":"string"}},
            "required":["value"],
            "additionalProperties": false
        }),
        strict: None,
        provider_options: None,
    };
    let opts = v2t::CallOptions {
        prompt,
        tools: vec![
            v2t::Tool::Provider(provider_tool),
            v2t::Tool::Function(function_tool),
        ],
        tool_choice: Some(v2t::ToolChoice::Tool {
            name: "search".into(),
        }),
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body.get("tool_choice"), Some(&json!({"type":"web_search"})));
    assert_eq!(
        body.get("tools"),
        Some(&json!([
            {
                "type":"web_search",
                "filters": { "allowed_domains": ["example.com"] },
                "external_web_access": true,
                "search_context_size": "medium",
                "user_location": { "type": "approximate", "country": "US" }
            },
            {
                "type":"function",
                "name":"test-tool",
                "description":"test",
                "parameters":{
                    "type":"object",
                    "properties":{"value":{"type":"string"}},
                    "required":["value"],
                    "additionalProperties": false
                }
            }
        ]))
    );
}

#[tokio::test]
async fn request_body_function_tool_strict_true_passthrough() {
    let body = request_body_for_function_tool(function_tool_for_strict_passthrough(Some(true))).await;
    let function_tool = body
        .get("tools")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .expect("function tool payload");
    assert_eq!(
        function_tool.get("strict"),
        Some(&json!(true)),
        "function tool strict=true must serialize to tools[0].strict=true"
    );
}

#[tokio::test]
async fn request_body_function_tool_strict_false_passthrough() {
    let body = request_body_for_function_tool(function_tool_for_strict_passthrough(Some(false))).await;
    let function_tool = body
        .get("tools")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .expect("function tool payload");
    assert_eq!(
        function_tool.get("strict"),
        Some(&json!(false)),
        "function tool strict=false must serialize to tools[0].strict=false"
    );
}

#[tokio::test]
async fn request_body_function_tool_strict_omitted_when_unspecified() {
    let body = request_body_for_function_tool(function_tool_for_strict_passthrough(None)).await;
    let function_tool = body
        .get("tools")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .expect("function tool payload");
    assert!(
        function_tool.get("strict").is_none(),
        "function tool strict field must be omitted when not explicitly specified"
    );
}

#[tokio::test]
async fn request_body_auto_includes_stream_extras() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([
            ("store".into(), json!(false)),
            ("logprobs".into(), json!(true)),
        ]),
    );
    let opts = v2t::CallOptions {
        prompt,
        tools: vec![
            v2t::Tool::Provider(v2t::ProviderTool {
                r#type: v2t::ProviderToolType::Provider,
                id: "openai.web_search".into(),
                name: "webSearch".into(),
                args: json!({}),
            }),
            v2t::Tool::Provider(v2t::ProviderTool {
                r#type: v2t::ProviderToolType::Provider,
                id: "openai.code_interpreter".into(),
                name: "codeExecution".into(),
                args: json!({}),
            }),
        ],
        provider_options,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5-mini",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    let include = body
        .get("include")
        .and_then(|v| v.as_array())
        .expect("include");
    assert_eq!(
        include,
        &vec![
            json!("message.output_text.logprobs"),
            json!("web_search_call.action.sources"),
            json!("code_interpreter_call.outputs"),
            json!("reasoning.encrypted_content"),
        ]
    );
}

#[tokio::test]
async fn provider_tool_args_validation_errors() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Search please".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let provider_tool = v2t::ProviderTool {
        r#type: v2t::ProviderToolType::Provider,
        id: "openai.file_search".into(),
        name: "search".into(),
        args: json!({}),
    };
    let opts = v2t::CallOptions {
        prompt,
        tools: vec![v2t::Tool::Provider(provider_tool)],
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model =
        OpenAIResponsesLanguageModel::new("gpt-4o", cfg, transport, TransportConfig::default());

    let err = match model.do_stream(opts).await {
        Ok(_) => panic!("invalid args should fail"),
        Err(err) => err,
    };
    match err {
        SdkError::InvalidArgument { message } => {
            assert!(message.contains("vectorStoreIds"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn non_stream_response_error_returns_error() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let opts = v2t::CallOptions {
        prompt,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new().with_json_response(openai_error_fixture());
    let model =
        OpenAIResponsesLanguageModel::new("gpt-4o", cfg, transport, TransportConfig::default());

    let err = model
        .do_generate(opts)
        .await
        .expect_err("response.error should propagate as SdkError");
    match err {
        SdkError::Upstream { message, .. } => {
            assert!(message.contains("You exceeded your current quota"));
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[tokio::test]
async fn non_stream_usage_maps_nested_details() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let opts = v2t::CallOptions {
        prompt,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new().with_json_response(local_shell_response_fixture());
    let model =
        OpenAIResponsesLanguageModel::new("gpt-5-codex", cfg, transport, TransportConfig::default());

    let result = model.do_generate(opts).await.expect("generate response");
    assert_eq!(result.usage.input_tokens, Some(407));
    assert_eq!(result.usage.cached_input_tokens, Some(0));
    assert_eq!(result.usage.output_tokens, Some(24));
    assert_eq!(result.usage.reasoning_tokens, Some(0));
    assert_eq!(result.usage.total_tokens, Some(431));
}

#[tokio::test]
async fn non_stream_mcp_approval_requests_emit_content() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let response = json!({
        "output": [
            {
                "type": "mcp_approval_request",
                "id": "approval-item",
                "name": "test-tool",
                "arguments": "{\"foo\":\"bar\"}",
                "approval_request_id": "approval-123"
            }
        ]
    });
    let opts = v2t::CallOptions {
        prompt,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new().with_json_response(response);
    let model =
        OpenAIResponsesLanguageModel::new("gpt-4o", cfg, transport, TransportConfig::default());

    let result = model.do_generate(opts).await.expect("generate response");
    assert_eq!(result.content.len(), 2);

    let (tool_call_id, tool_name, input, provider_executed) = match &result.content[0] {
        v2t::Content::ToolCall(call) => (
            call.tool_call_id.clone(),
            call.tool_name.clone(),
            call.input.clone(),
            call.provider_executed,
        ),
        other => panic!("unexpected content: {other:?}"),
    };
    assert_eq!(tool_name, "mcp.test-tool");
    assert_eq!(input, "{\"foo\":\"bar\"}");
    assert!(provider_executed);

    match &result.content[1] {
        v2t::Content::ToolApprovalRequest {
            approval_id,
            tool_call_id: approval_tool_call_id,
            ..
        } => {
            assert_eq!(approval_id, "approval-123");
            assert_eq!(approval_tool_call_id, &tool_call_id);
        }
        other => panic!("unexpected content: {other:?}"),
    }
}

#[tokio::test]
async fn request_body_includes_item_references_and_tool_approvals() {
    let prompt = vec![
        v2t::PromptMessage::Assistant {
            content: vec![v2t::AssistantPart::Text {
                text: "Hello".into(),
                provider_options: Some(v2t::ProviderOptions::from([(
                    "openai".into(),
                    HashMap::from([("itemId".into(), json!("msg_123"))]),
                )])),
            }],
            provider_options: None,
        },
        v2t::PromptMessage::Assistant {
            content: vec![v2t::AssistantPart::ToolCall(v2t::ToolCallPart {
                tool_call_id: "call-1".into(),
                tool_name: "local_shell".into(),
                input: json!({"action": {"command": ["ls"]}}).to_string(),
                provider_executed: false,
                provider_metadata: None,
                dynamic: false,
                provider_options: Some(v2t::ProviderOptions::from([(
                    "openai".into(),
                    HashMap::from([("itemId".into(), json!("lsh_123"))]),
                )])),
            })],
            provider_options: None,
        },
        v2t::PromptMessage::Tool {
            content: vec![
                v2t::ToolMessagePart::ToolResult(v2t::ToolResultPart {
                    r#type: v2t::ToolResultPartType::ToolResult,
                    tool_call_id: "call-1".into(),
                    tool_name: "local_shell".into(),
                    output: v2t::ToolResultOutput::Json {
                        value: json!({"output": "ok"}),
                    },
                    provider_options: None,
                }),
                v2t::ToolMessagePart::ToolApprovalResponse(v2t::ToolApprovalResponsePart {
                    r#type: v2t::ToolApprovalResponsePartType::ToolApprovalResponse,
                    approval_id: "mcpr_1".into(),
                    approved: true,
                    provider_options: None,
                }),
            ],
            provider_options: None,
        },
    ];
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([("store".into(), json!(true))]),
    );
    let opts = v2t::CallOptions {
        prompt,
        tools: vec![v2t::Tool::Provider(v2t::ProviderTool {
            r#type: v2t::ProviderToolType::Provider,
            id: "openai.local_shell".into(),
            name: "local_shell".into(),
            args: json!({}),
        })],
        provider_options,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body, item_reference_fixture());
}

#[tokio::test]
async fn request_body_includes_reasoning_summary_when_store_false() {
    let prompt = vec![v2t::PromptMessage::Assistant {
        content: vec![v2t::AssistantPart::Reasoning {
            text: "step one".into(),
            provider_options: Some(v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([
                    ("itemId".into(), json!("rs_1")),
                    ("reasoningEncryptedContent".into(), json!("enc")),
                ]),
            )])),
        }],
        provider_options: None,
    }];
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([("store".into(), json!(false))]),
    );
    let opts = v2t::CallOptions {
        prompt,
        provider_options,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body, reasoning_summary_fixture());
}

#[tokio::test]
async fn request_body_includes_provider_tool_outputs() {
    let prompt = vec![v2t::PromptMessage::Tool {
        content: vec![
            v2t::ToolMessagePart::ToolResult(v2t::ToolResultPart {
                r#type: v2t::ToolResultPartType::ToolResult,
                tool_call_id: "call-shell".into(),
                tool_name: "shell".into(),
                output: v2t::ToolResultOutput::Json {
                    value: json!({
                        "output": [
                            {
                                "stdout": "hi\n",
                                "stderr": "",
                                "outcome": { "type": "exit", "exitCode": 0 }
                            }
                        ]
                    }),
                },
                provider_options: None,
            }),
            v2t::ToolMessagePart::ToolResult(v2t::ToolResultPart {
                r#type: v2t::ToolResultPartType::ToolResult,
                tool_call_id: "call-apply".into(),
                tool_name: "apply_patch".into(),
                output: v2t::ToolResultOutput::Json {
                    value: json!({
                        "status": "completed",
                        "output": "patched"
                    }),
                },
                provider_options: None,
            }),
        ],
        provider_options: None,
    }];
    let opts = v2t::CallOptions {
        prompt,
        tools: vec![
            v2t::Tool::Provider(v2t::ProviderTool {
                r#type: v2t::ProviderToolType::Provider,
                id: "openai.shell".into(),
                name: "shell".into(),
                args: json!({}),
            }),
            v2t::Tool::Provider(v2t::ProviderTool {
                r#type: v2t::ProviderToolType::Provider,
                id: "openai.apply_patch".into(),
                name: "apply_patch".into(),
                args: json!({}),
            }),
        ],
        ..Default::default()
    };
    let cfg = OpenAIConfig {
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
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body, provider_tool_outputs_fixture());
}

#[tokio::test]
async fn request_defaults_do_not_override_explicit_reasoning_effort() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([("reasoningEffort".into(), json!("xhigh"))]),
    );
    let opts = v2t::CallOptions {
        prompt,
        provider_options,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://api.openai.com/v1".into(),
        endpoint_path: "/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: Some(json!({
            "openai": {
                "reasoning": {
                    "effort": "low",
                    "summary": "auto"
                }
            }
        })),
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(
        body.get("reasoning").and_then(|r| r.get("effort")),
        Some(&json!("xhigh"))
    );
    assert_eq!(
        body.get("reasoning").and_then(|r| r.get("summary")),
        Some(&json!("auto"))
    );
}

#[tokio::test]
async fn request_defaults_apply_reasoning_when_effort_not_explicit() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];
    let opts = v2t::CallOptions {
        prompt,
        ..Default::default()
    };
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://api.openai.com/v1".into(),
        endpoint_path: "/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: Some(json!({
            "openai": {
                "reasoning": {
                    "effort": "low",
                    "summary": "auto"
                }
            }
        })),
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(
        body.get("reasoning").and_then(|r| r.get("effort")),
        Some(&json!("low"))
    );
    assert_eq!(
        body.get("reasoning").and_then(|r| r.get("summary")),
        Some(&json!("auto"))
    );
}
