use crate::core::error::TransportError;
use crate::core::json::without_null_fields;
use crate::core::transport::{
    set_transport_observer, HttpTransport, TransportBody, TransportConfig, TransportEvent,
    TransportObserver,
};
use crate::core::{LanguageModel, SdkError};
use crate::provider::{registry, Credentials};
use crate::providers::anthropic::messages::language_model::AnthropicMessagesConfig;
use crate::providers::anthropic::AnthropicMessagesLanguageModel;
use crate::types::catalog::{ModelInfo, ProviderDefinition, SdkType};
use crate::types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::{stream, TryStreamExt};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::net::TcpListener;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<TransportEvent>>,
}

impl RecordingObserver {
    fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    fn request_json_for_url(&self, url: &str) -> Option<serde_json::Value> {
        self.events.lock().unwrap().iter().rev().find_map(|event| {
            if event.url != url {
                return None;
            }
            match event.request_body.clone() {
                Some(TransportBody::Json(body)) => Some(body),
                _ => None,
            }
        })
    }
}

impl TransportObserver for RecordingObserver {
    fn on_event(&self, event: TransportEvent) {
        self.events.lock().unwrap().push(event);
    }
}

fn transport_observer() -> Arc<RecordingObserver> {
    static OBSERVER: OnceLock<Arc<RecordingObserver>> = OnceLock::new();
    OBSERVER
        .get_or_init(|| {
            let observer = Arc::new(RecordingObserver::default());
            assert!(
                set_transport_observer(observer.clone()),
                "transport observer was already installed before Anthropic null-path tests ran"
            );
            observer
        })
        .clone()
}

#[derive(Clone, Default)]
struct TestTransport {
    last_body: Arc<Mutex<Option<serde_json::Value>>>,
    last_headers: Arc<Mutex<Option<Vec<(String, String)>>>>,
    stream_chunks: Arc<Vec<Bytes>>,
}

impl TestTransport {
    fn with_stream_chunks(chunks: Vec<Bytes>) -> Self {
        Self {
            last_body: Arc::new(Mutex::new(None)),
            last_headers: Arc::new(Mutex::new(None)),
            stream_chunks: Arc::new(chunks),
        }
    }

    fn last_body(&self) -> Option<serde_json::Value> {
        self.last_body.lock().unwrap().clone()
    }

    fn last_headers(&self) -> Option<Vec<(String, String)>> {
        self.last_headers.lock().unwrap().clone()
    }
}

struct TestStreamResponse {
    headers: Vec<(String, String)>,
    chunks: Vec<Result<Bytes, TransportError>>,
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
        (Box::pin(stream::iter(resp.chunks)), resp.headers)
    }

    async fn post_json_stream(
        &self,
        _url: &str,
        headers: &[(String, String)],
        body: &serde_json::Value,
        cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        let sent_body = if cfg.strip_null_fields {
            without_null_fields(body)
        } else {
            body.clone()
        };
        *self.last_body.lock().unwrap() = Some(sent_body);
        *self.last_headers.lock().unwrap() = Some(headers.to_vec());
        Ok(TestStreamResponse {
            headers: vec![],
            chunks: self.stream_chunks.iter().cloned().map(Ok).collect(),
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

    async fn post_multipart(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _form: &crate::core::transport::MultipartForm,
        _cfg: &TransportConfig,
    ) -> Result<(serde_json::Value, Vec<(String, String)>), TransportError> {
        Err(TransportError::Other("post_multipart unused".into()))
    }

    async fn get_bytes(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _cfg: &TransportConfig,
    ) -> Result<(Bytes, Vec<(String, String)>), TransportError> {
        Err(TransportError::Other("get_bytes unused".into()))
    }
}

fn provider_tool(id: &str, args: serde_json::Value) -> v2t::Tool {
    v2t::Tool::Provider(v2t::ProviderTool {
        r#type: v2t::ProviderToolType::Provider,
        id: id.to_string(),
        name: id.to_string(),
        args,
    })
}

fn basic_prompt() -> v2t::Prompt {
    vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "hi".into(),
            provider_options: None,
        }],
        provider_options: None,
    }]
}

fn anthropic_cache_control_options() -> Option<v2t::ProviderOptions> {
    let mut scope = HashMap::new();
    scope.insert("cacheControl".to_string(), json!({ "type": "ephemeral" }));
    let mut opts = v2t::ProviderOptions::new();
    opts.insert("anthropic".to_string(), scope);
    Some(opts)
}

fn build_model(transport: TestTransport) -> AnthropicMessagesLanguageModel<TestTransport> {
    let cfg = AnthropicMessagesConfig {
        provider_name: "anthropic",
        provider_scope_name: "anthropic".into(),
        base_url: "https://api.example.com".into(),
        headers: vec![],
        http: transport,
        transport_cfg: TransportConfig::default(),
        supported_urls: HashMap::new(),
        default_options: None,
    };
    AnthropicMessagesLanguageModel::new("claude-3-5-sonnet-20241022".into(), cfg)
}

fn anthropic_provider_definition(base_url: String) -> ProviderDefinition {
    ProviderDefinition {
        name: "anthropic".into(),
        display_name: "Anthropic".into(),
        sdk_type: SdkType::Anthropic,
        base_url,
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/messages".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models: HashMap::<String, ModelInfo>::new(),
        preserve_model_prefix: true,
    }
}

fn unused_local_base_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let port = listener.local_addr().expect("listener addr").port();
    drop(listener);
    format!("http://127.0.0.1:{port}")
}

fn contains_object_null(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(map) => map
            .values()
            .any(|entry| entry.is_null() || contains_object_null(entry)),
        serde_json::Value::Array(items) => items.iter().any(contains_object_null),
        _ => false,
    }
}

fn sse_chunk(event: Option<&str>, payload: serde_json::Value) -> Bytes {
    let mut chunk = String::new();
    if let Some(event) = event {
        chunk.push_str(&format!("event: {event}\n"));
    }
    chunk.push_str(&format!("data: {payload}\n\n"));
    Bytes::from(chunk)
}

#[tokio::test]
async fn provider_tool_sets_code_execution_beta() {
    let transport = TestTransport::default();
    let model = build_model(transport.clone());

    let mut options = v2t::CallOptions::new(basic_prompt());
    options.tools = vec![provider_tool(
        "anthropic.code_execution_20250522",
        json!({}),
    )];

    let response = model.do_stream(options).await.expect("stream response");
    let body = response.request_body.expect("request body");
    assert_eq!(
        body.get("tools").cloned(),
        Some(json!([
            {"type": "code_execution_20250522", "name": "code_execution"}
        ]))
    );

    let headers = transport.last_headers().expect("headers captured");
    let beta_header = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let betas: HashSet<String> = beta_header
        .split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect();
    assert!(betas.contains("code-execution-2025-05-22"));
}

#[tokio::test]
async fn provider_tool_maps_web_fetch() {
    let transport = TestTransport::default();
    let model = build_model(transport.clone());

    let mut options = v2t::CallOptions::new(basic_prompt());
    options.tools = vec![provider_tool(
        "anthropic.web_fetch_20250910",
        json!({
            "maxUses": 3,
            "allowedDomains": ["example.com"],
            "blockedDomains": ["bad.com"],
            "citations": {"enabled": true},
            "maxContentTokens": 123
        }),
    )];

    let response = model.do_stream(options).await.expect("stream response");
    let body = response.request_body.expect("request body");
    assert_eq!(
        body.get("tools").cloned(),
        Some(json!([
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "max_uses": 3,
                "allowed_domains": ["example.com"],
                "blocked_domains": ["bad.com"],
                "citations": {"enabled": true},
                "max_content_tokens": 123
            }
        ]))
    );

    let headers = transport.last_headers().expect("headers captured");
    let beta_header = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    let betas: HashSet<String> = beta_header
        .split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect();
    assert!(betas.contains("web-fetch-2025-09-10"));
}

#[tokio::test]
async fn adjacent_system_text_entries_with_matching_cache_policy_are_collapsed() {
    let transport = TestTransport::default();
    let model = build_model(transport);

    let prompt = vec![
        v2t::PromptMessage::System {
            content: "cached system".into(),
            provider_options: anthropic_cache_control_options(),
        },
        v2t::PromptMessage::System {
            content: "plain one".into(),
            provider_options: None,
        },
        v2t::PromptMessage::System {
            content: "plain two".into(),
            provider_options: None,
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hi".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
    ];
    let response = model
        .do_stream(v2t::CallOptions::new(prompt))
        .await
        .expect("stream response");
    let body = response.request_body.expect("request body");
    let system = body
        .get("system")
        .and_then(|value| value.as_array())
        .expect("system array");
    assert_eq!(system.len(), 2);
    assert_eq!(
        system[0].get("text").and_then(|value| value.as_str()),
        Some("cached system")
    );
    assert_eq!(
        system[0]
            .get("cache_control")
            .and_then(|value| value.get("type"))
            .and_then(|value| value.as_str()),
        Some("ephemeral")
    );
    assert_eq!(
        system[1].get("text").and_then(|value| value.as_str()),
        Some("plain one\n\nplain two")
    );
    assert!(system[1].get("cache_control").is_none());
}

#[tokio::test]
async fn request_body_can_still_contain_object_nulls_before_transport_pruning() {
    let transport = TestTransport::default();
    let model = build_model(transport);

    let prompt = vec![
        v2t::PromptMessage::System {
            content: "system".into(),
            provider_options: anthropic_cache_control_options(),
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hi".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
    ];
    let mut options = v2t::CallOptions::new(prompt);
    options.tools = vec![provider_tool(
        "anthropic.web_fetch_20250910",
        json!({
            "maxUses": 1,
            "allowedDomains": ["example.com"],
            "citations": {"enabled": true}
        }),
    )];

    let response = model.do_stream(options).await.expect("stream response");
    let body = response.request_body.expect("request body");
    assert!(
        contains_object_null(&body),
        "anthropic builder output should continue documenting object nulls until a later cleanup lineage removes them: {body}"
    );
}

#[tokio::test]
async fn transport_prunes_object_nulls_from_stream_payload_when_enabled() {
    let transport = TestTransport::default();
    let model = build_model(transport.clone());

    let prompt = vec![
        v2t::PromptMessage::System {
            content: "system".into(),
            provider_options: anthropic_cache_control_options(),
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hi".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
    ];
    let mut options = v2t::CallOptions::new(prompt);
    options.tools = vec![provider_tool(
        "anthropic.web_fetch_20250910",
        json!({
            "maxUses": 1,
            "allowedDomains": ["example.com"],
            "citations": {"enabled": true}
        }),
    )];

    let response = model.do_stream(options).await.expect("stream response");
    let builder_body = response.request_body.expect("builder request body");
    assert!(
        contains_object_null(&builder_body),
        "builder request body should still show the pre-transport null evidence: {builder_body}"
    );

    let wire_body = transport.last_body().expect("wire request body");
    assert!(
        !contains_object_null(&wire_body),
        "transport should strip object nulls before sending the Anthropic stream payload: {wire_body}"
    );
}

#[tokio::test]
async fn registry_built_anthropic_stream_payload_is_null_free_on_wire() {
    let observer = transport_observer();
    observer.clear();

    let base_url = unused_local_base_url();
    let wire_url = format!("{base_url}/messages");
    let definition = anthropic_provider_definition(base_url);
    let registration = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("anthropic"))
        .expect("anthropic registration");
    let model = (registration.build)(
        &definition,
        "claude-3-5-sonnet-20241022",
        &Credentials::ApiKey("test-key".into()),
    )
    .expect("build anthropic model");

    let mut options = v2t::CallOptions::new(basic_prompt());
    options.tools = vec![provider_tool(
        "anthropic.web_fetch_20250910",
        json!({
            "maxUses": 1,
            "allowedDomains": ["example.com"],
            "citations": {"enabled": true}
        }),
    )];

    let err = match model.do_stream(options).await {
        Ok(_) => panic!("closed localhost port should fail"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        SdkError::Transport(TransportError::Network(_))
    ));

    let wire_body = observer
        .request_json_for_url(&wire_url)
        .expect("captured Anthropic wire request");
    assert!(
        !contains_object_null(&wire_body),
        "registry-built Anthropic transport should keep object nulls off the wire: {wire_body}"
    );
}

#[tokio::test]
async fn shared_event_mapper_streams_reasoning_text_tool_and_finish() {
    let transport = TestTransport::with_stream_chunks(vec![
        sse_chunk(
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
        sse_chunk(
            Some("content_block_start"),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"}
            }),
        ),
        sse_chunk(
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
        sse_chunk(
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
        sse_chunk(
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
        sse_chunk(
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
        sse_chunk(
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
        sse_chunk(
            Some("message_delta"),
            json!({
                "type": "message_delta",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3
                }
            }),
        ),
        sse_chunk(Some("message_stop"), json!({"type": "message_stop"})),
    ]);
    let model = build_model(transport);

    let response = model
        .do_stream(v2t::CallOptions::new(basic_prompt()))
        .await
        .expect("stream response");
    let parts: Vec<v2t::StreamPart> = response
        .stream
        .try_collect()
        .await
        .expect("collect stream parts");

    assert!(matches!(
        &parts[0],
        v2t::StreamPart::StreamStart { warnings } if warnings.is_empty()
    ));

    let reasoning_start_idx = parts
        .iter()
        .position(|part| matches!(part, v2t::StreamPart::ReasoningStart { id, .. } if id == "0"))
        .expect("reasoning start");
    let reasoning_delta_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::ReasoningDelta { id, delta, .. }
                    if id == "0" && delta == "ponder"
            )
        })
        .expect("reasoning delta");
    let reasoning_signature_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::ReasoningSignature { signature, .. } if signature == "sig-1"
            )
        })
        .expect("reasoning signature");
    let text_start_idx = parts
        .iter()
        .position(|part| matches!(part, v2t::StreamPart::TextStart { id, .. } if id == "text-1"))
        .expect("text start");
    let text_delta_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::TextDelta { id, delta, .. }
                    if id == "text-1" && delta == "hello"
            )
        })
        .expect("text delta");
    let tool_input_start_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::ToolInputStart { id, tool_name, provider_executed, .. }
                    if id == "tool-1" && tool_name == "weather" && !provider_executed
            )
        })
        .expect("tool input start");
    let tool_input_delta_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::ToolInputDelta { id, delta, provider_executed, .. }
                    if id == "tool-1" && delta == "{\"city\":\"SF\"}" && !provider_executed
            )
        })
        .expect("tool input delta");
    let tool_input_end_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::ToolInputEnd { id, provider_executed, .. }
                    if id == "tool-1" && !provider_executed
            )
        })
        .expect("tool input end");
    let tool_call_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::ToolCall(call)
                    if call.tool_call_id == "tool-1"
                        && call.tool_name == "weather"
                        && call.input == "{\"city\":\"SF\"}"
                        && !call.provider_executed
            )
        })
        .expect("tool call");
    let text_end_idx = parts
        .iter()
        .position(|part| matches!(part, v2t::StreamPart::TextEnd { id, .. } if id == "text-1"))
        .expect("text end");
    let reasoning_end_idx = parts
        .iter()
        .position(|part| matches!(part, v2t::StreamPart::ReasoningEnd { id, .. } if id == "0"))
        .expect("reasoning end");
    let finish_idx = parts
        .iter()
        .position(|part| {
            matches!(
                part,
                v2t::StreamPart::Finish {
                    usage,
                    finish_reason,
                    ..
                } if usage.input_tokens == Some(2)
                    && usage.output_tokens == Some(3)
                    && usage.total_tokens == Some(5)
                    && matches!(finish_reason, v2t::FinishReason::ToolCalls)
            )
        })
        .expect("finish");

    assert!(reasoning_start_idx < reasoning_delta_idx);
    assert!(reasoning_delta_idx < reasoning_signature_idx);
    assert!(reasoning_signature_idx < text_start_idx);
    assert!(text_start_idx < text_delta_idx);
    assert!(text_delta_idx < tool_input_start_idx);
    assert!(tool_input_start_idx < tool_input_delta_idx);
    assert!(tool_input_delta_idx < tool_input_end_idx);
    assert!(tool_input_end_idx < tool_call_idx);
    assert!(tool_call_idx < finish_idx);
    assert!(text_start_idx < text_end_idx);
    assert!(reasoning_start_idx < reasoning_end_idx);
    assert!(text_end_idx < finish_idx);
    assert!(reasoning_end_idx < finish_idx);
}
