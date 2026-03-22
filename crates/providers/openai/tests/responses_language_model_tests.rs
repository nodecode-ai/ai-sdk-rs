use crate::core::error::{SdkError, TransportError};
use crate::core::transport::{
    HttpTransport, JsonStreamWebsocketConnection, TransportConfig, TransportStream,
};
use crate::core::LanguageModel;
use crate::providers::openai::config::OpenAIConfig;
use crate::providers::openai::responses::language_model::OpenAIResponsesLanguageModel;
use crate::types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream;
use futures_util::stream::BoxStream;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message as WsMessage;

enum StreamBehavior {
    Chunks(Vec<Result<Bytes, TransportError>>),
    Error(TransportError),
}

#[derive(Clone)]
struct TestTransport {
    last_body: Arc<Mutex<Option<Value>>>,
    last_url: Arc<Mutex<Option<String>>>,
    last_headers: Arc<Mutex<Vec<(String, String)>>>,
    stream_urls: Arc<Mutex<Vec<String>>>,
    websocket_connect_urls: Arc<Mutex<Vec<String>>>,
    websocket_request_bodies: Arc<Mutex<Vec<Value>>>,
    websocket_response_headers: Arc<Mutex<Vec<(String, String)>>>,
    close_websocket_after_send: Arc<AtomicBool>,
    json_response: Arc<Mutex<Option<Value>>>,
    stream_behaviors: Arc<Mutex<VecDeque<StreamBehavior>>>,
}

impl TestTransport {
    fn new() -> Self {
        Self {
            last_body: Arc::new(Mutex::new(None)),
            last_url: Arc::new(Mutex::new(None)),
            last_headers: Arc::new(Mutex::new(Vec::new())),
            stream_urls: Arc::new(Mutex::new(Vec::new())),
            websocket_connect_urls: Arc::new(Mutex::new(Vec::new())),
            websocket_request_bodies: Arc::new(Mutex::new(Vec::new())),
            websocket_response_headers: Arc::new(Mutex::new(Vec::new())),
            close_websocket_after_send: Arc::new(AtomicBool::new(false)),
            json_response: Arc::new(Mutex::new(None)),
            stream_behaviors: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn last_body(&self) -> Option<Value> {
        self.last_body.lock().unwrap().clone()
    }

    fn last_url(&self) -> Option<String> {
        self.last_url.lock().unwrap().clone()
    }

    fn last_headers(&self) -> Vec<(String, String)> {
        self.last_headers.lock().unwrap().clone()
    }

    fn stream_urls(&self) -> Vec<String> {
        self.stream_urls.lock().unwrap().clone()
    }

    fn websocket_connect_urls(&self) -> Vec<String> {
        self.websocket_connect_urls.lock().unwrap().clone()
    }

    fn websocket_request_bodies(&self) -> Vec<Value> {
        self.websocket_request_bodies.lock().unwrap().clone()
    }

    fn with_json_response(self, response: Value) -> Self {
        *self.json_response.lock().unwrap() = Some(response);
        self
    }

    fn with_stream_behavior(self, behavior: StreamBehavior) -> Self {
        self.stream_behaviors.lock().unwrap().push_back(behavior);
        self
    }

    fn with_websocket_response_headers(self, headers: Vec<(String, String)>) -> Self {
        *self.websocket_response_headers.lock().unwrap() = headers;
        self
    }

    fn with_close_websocket_after_send(self) -> Self {
        self.close_websocket_after_send
            .store(true, Ordering::SeqCst);
        self
    }
}

struct TestStreamResponse {
    stream: BoxStream<'static, Result<Bytes, TransportError>>,
}

struct TestWebsocketConnection {
    transport: TestTransport,
    closed: Arc<AtomicBool>,
}

#[async_trait]
impl JsonStreamWebsocketConnection for TestWebsocketConnection {
    async fn send_json_stream(
        &self,
        body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<TransportStream, TransportError> {
        *self.transport.last_body.lock().unwrap() = Some(body.clone());
        self.transport
            .websocket_request_bodies
            .lock()
            .unwrap()
            .push(body.clone());
        let behavior = self
            .transport
            .stream_behaviors
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| {
                StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
                    b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-default\"}}\n\n",
                ))])
            });
        if self
            .transport
            .close_websocket_after_send
            .load(Ordering::SeqCst)
        {
            self.closed.store(true, Ordering::SeqCst);
        }
        match behavior {
            StreamBehavior::Chunks(chunks) => Ok(Box::pin(stream::iter(chunks))),
            StreamBehavior::Error(err) => {
                self.closed.store(true, Ordering::SeqCst);
                Err(err)
            }
        }
    }

    fn response_headers(&self) -> Vec<(String, String)> {
        self.transport
            .websocket_response_headers
            .lock()
            .unwrap()
            .clone()
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl HttpTransport for TestTransport {
    type StreamResponse = TestStreamResponse;

    fn into_stream(resp: Self::StreamResponse) -> (TransportStream, Vec<(String, String)>) {
        (resp.stream, vec![])
    }

    async fn post_json_stream(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        *self.last_body.lock().unwrap() = Some(body.clone());
        *self.last_url.lock().unwrap() = Some(url.to_string());
        *self.last_headers.lock().unwrap() = headers.to_vec();
        self.stream_urls.lock().unwrap().push(url.to_string());
        let behavior = self
            .stream_behaviors
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| {
                StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
                    b"data: {\"type\":\"response.completed\"}\n\n",
                ))])
            });
        match behavior {
            StreamBehavior::Chunks(chunks) => Ok(TestStreamResponse {
                stream: Box::pin(stream::iter(chunks)),
            }),
            StreamBehavior::Error(err) => Err(err),
        }
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

    async fn connect_json_stream_websocket(
        &self,
        url: &str,
        headers: &[(String, String)],
        _cfg: &TransportConfig,
    ) -> Result<Box<dyn JsonStreamWebsocketConnection>, TransportError> {
        *self.last_url.lock().unwrap() = Some(url.to_string());
        *self.last_headers.lock().unwrap() = headers.to_vec();
        self.websocket_connect_urls
            .lock()
            .unwrap()
            .push(url.to_string());
        Ok(Box::new(TestWebsocketConnection {
            transport: self.clone(),
            closed: Arc::new(AtomicBool::new(false)),
        }))
    }
}

fn provider_options_fixture() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/responses_provider_options_request.json"
    ))
    .expect("provider options fixture")
}

fn websocket_transport_options() -> v2t::ProviderOptions {
    v2t::ProviderOptions::from([(
        "openai".into(),
        HashMap::from([(
            "transport".into(),
            json!({
                "mode": "websocket",
                "fallback": "http",
            }),
        )]),
    )])
}

struct TestWebsocketServer {
    base_url: String,
    connect_headers: Arc<Mutex<Vec<HashMap<String, String>>>>,
    request_records: Arc<Mutex<Vec<(usize, Value)>>>,
    listener_task: JoinHandle<()>,
    connection_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl TestWebsocketServer {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind websocket listener");
        let addr = listener.local_addr().expect("listener addr");
        let connect_headers = Arc::new(Mutex::new(Vec::new()));
        let request_records = Arc::new(Mutex::new(Vec::new()));
        let connection_tasks = Arc::new(Mutex::new(Vec::new()));
        let next_connection_id = Arc::new(AtomicUsize::new(1));
        let listener_connect_headers = Arc::clone(&connect_headers);
        let listener_request_records = Arc::clone(&request_records);
        let listener_connection_tasks = Arc::clone(&connection_tasks);
        let listener_task = tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(stream) => stream,
                    Err(_) => break,
                };
                let connection_id = next_connection_id.fetch_add(1, Ordering::SeqCst);
                let task_connect_headers = Arc::clone(&listener_connect_headers);
                let task_request_records = Arc::clone(&listener_request_records);
                let connection_task = tokio::spawn(async move {
                    let captured_headers = Arc::new(Mutex::new(None::<HashMap<String, String>>));
                    let callback_headers = Arc::clone(&captured_headers);
                    let mut websocket =
                        accept_hdr_async(stream, move |request: &Request, response: Response| {
                            let headers = request
                                .headers()
                                .iter()
                                .map(|(name, value)| {
                                    (
                                        name.as_str().to_string(),
                                        value.to_str().unwrap_or_default().to_string(),
                                    )
                                })
                                .collect::<HashMap<_, _>>();
                            *callback_headers.lock().unwrap() = Some(headers);
                            Ok(response)
                        })
                        .await
                        .expect("accept websocket");
                    task_connect_headers
                        .lock()
                        .unwrap()
                        .push(captured_headers.lock().unwrap().take().unwrap_or_default());

                    while let Some(message) = websocket.next().await {
                        let message = message.expect("websocket message");
                        let text = match message {
                            WsMessage::Text(text) => text,
                            WsMessage::Binary(binary) => {
                                String::from_utf8(binary.to_vec()).expect("utf8 websocket body")
                            }
                            WsMessage::Close(_) => break,
                            _ => continue,
                        };
                        let body: Value =
                            serde_json::from_str(&text).expect("websocket request body");
                        task_request_records
                            .lock()
                            .unwrap()
                            .push((connection_id, body));
                        websocket
                            .send(WsMessage::Text(
                                json!({
                                    "type": "response.completed",
                                    "response": {
                                        "id": format!("resp-{connection_id}"),
                                    },
                                })
                                .to_string()
                                .into(),
                            ))
                            .await
                            .expect("send websocket response");
                        websocket.close(None).await.expect("close websocket");
                        break;
                    }
                });
                listener_connection_tasks
                    .lock()
                    .unwrap()
                    .push(connection_task);
            }
        });

        Self {
            base_url: format!("http://{addr}"),
            connect_headers,
            request_records,
            listener_task,
            connection_tasks,
        }
    }

    async fn wait_for_connection_count(&self, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if self.connect_headers.lock().unwrap().len() >= expected {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {expected} websocket connection(s); saw {}",
                self.connect_headers.lock().unwrap().len()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    fn connect_headers(&self) -> Vec<HashMap<String, String>> {
        self.connect_headers.lock().unwrap().clone()
    }

    fn request_records(&self) -> Vec<(usize, Value)> {
        self.request_records.lock().unwrap().clone()
    }
}

impl Drop for TestWebsocketServer {
    fn drop(&mut self) {
        self.listener_task.abort();
        let mut tasks = self.connection_tasks.lock().unwrap();
        for task in tasks.iter() {
            task.abort();
        }
        tasks.clear();
    }
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

fn transport_http_status(status: u16) -> TransportError {
    TransportError::HttpStatus {
        status,
        body: "{}".to_string(),
        retry_after_ms: Some(250),
        sanitized: format!("http status {status}"),
        headers: vec![],
    }
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

async fn drain_stream_response(response: crate::core::StreamResponse) {
    let mut stream = response.stream;
    while let Some(item) = stream.next().await {
        let _ = item.expect("stream part");
    }
}

async fn consume_until_finish_and_drop(response: crate::core::StreamResponse) {
    let mut stream = response.stream;
    while let Some(item) = stream.next().await {
        let item = item.expect("stream part");
        if matches!(item, v2t::StreamPart::Finish { .. }) {
            break;
        }
    }
}

fn request_shape_without_incremental_fields(value: &Value) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("input");
        object.remove("previous_response_id");
        object.remove("generate");
    }
    value
}

#[tokio::test]
async fn stream_uses_wss_for_codex_oauth_endpoint_path() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        ..Default::default()
    };

    let _ = model.do_stream(opts).await.expect("stream response");
    let url = transport.last_url().expect("stream url");
    assert_eq!(url, "wss://chatgpt.com/backend-api/codex/responses");
    assert_eq!(
        transport.last_body().expect("request body"),
        json!({
            "type": "response.create",
            "model": "gpt-5.3-codex",
            "input": [{"role":"user","content":[{"type":"input_text","text":"hello"}]}],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "stream": true
        })
    );
}

#[tokio::test]
async fn codex_websocket_request_body_defaults_tool_settings_when_callers_omit_them() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        ..Default::default()
    };

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body.get("tool_choice"), Some(&json!("auto")));
    assert_eq!(body.get("parallel_tool_calls"), Some(&json!(true)));
}

#[tokio::test]
async fn codex_websocket_request_body_keeps_explicit_tool_settings() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        tool_choice: Some(v2t::ToolChoice::None),
        provider_options: v2t::ProviderOptions::from([(
            "openai".into(),
            HashMap::from([("parallelToolCalls".into(), json!(false))]),
        )]),
        ..Default::default()
    };

    let _ = model.do_stream(opts).await.expect("stream response");
    let body = transport.last_body().expect("request body");
    assert_eq!(body.get("tool_choice"), Some(&json!("none")));
    assert_eq!(body.get("parallel_tool_calls"), Some(&json!(false)));
}

#[tokio::test]
async fn codex_websocket_forwards_call_headers_and_client_metadata() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new();
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "second".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        provider_options: v2t::ProviderOptions::from([(
            "openai".into(),
            HashMap::from([
                ("previousResponseId".into(), json!("resp-1")),
                (
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                ),
            ]),
        )]),
        headers: HashMap::from([
            (
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            ),
            (
                "x-codex-beta-features".to_string(),
                "multi_agent".to_string(),
            ),
        ]),
        ..Default::default()
    };

    let _ = model.do_stream(opts).await.expect("stream response");
    assert_eq!(
        transport.last_body().expect("request body"),
        json!({
            "type": "response.create",
            "model": "gpt-5.3-codex",
            "input": [{"role":"user","content":[{"type":"input_text","text":"second"}]}],
            "previous_response_id": "resp-1",
            "client_metadata": {
                "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
            },
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "stream": true
        })
    );
    assert!(transport.last_headers().iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-codex-turn-metadata")
            && value == "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
    }));
    assert!(transport.last_headers().iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("x-codex-beta-features") && value == "multi_agent"
    }));
}

#[tokio::test]
async fn stream_uses_websocket_when_provider_transport_requests_it() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://example.invalid/v1".into(),
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
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        provider_options: websocket_transport_options(),
        ..Default::default()
    };

    let _ = model.do_stream(opts).await.expect("stream response");
    let url = transport.last_url().expect("stream url");
    assert_eq!(url, "wss://example.invalid/v1/responses");
    assert_eq!(
        transport.last_body().expect("request body"),
        json!({
            "type": "response.create",
            "response": {
                "model": "gpt-4o",
                "input": [{"role":"user","content":[{"type":"input_text","text":"hello"}]}],
                "stream": true
            }
        })
    );
    assert!(transport
        .last_headers()
        .iter()
        .any(|(name, value)| name == "OpenAI-Beta" && value == "responses_websockets=2026-02-06"));
}

#[tokio::test]
async fn stream_falls_back_to_http_when_websocket_stream_closes_before_first_event() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://example.invalid/v1".into(),
        endpoint_path: "/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_stream_behavior(StreamBehavior::Chunks(vec![]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\"}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        provider_options: websocket_transport_options(),
        ..Default::default()
    };

    let response = model.do_stream(opts).await.expect("stream response");
    assert_eq!(
        response
            .response_headers
            .as_ref()
            .and_then(|headers| headers.get("x-ai-sdk-effective-transport"))
            .map(String::as_str),
        Some("http")
    );
    assert_eq!(
        response
            .response_headers
            .as_ref()
            .and_then(|headers| headers.get("x-ai-sdk-transport-fallback"))
            .map(String::as_str),
        Some("websocket->http")
    );
    assert_eq!(
        transport.stream_urls(),
        vec![
            "wss://example.invalid/v1/responses".to_string(),
            "https://example.invalid/v1/responses".to_string()
        ]
    );
}

#[tokio::test]
async fn stream_does_not_fallback_to_http_when_websocket_is_rate_limited() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://example.invalid/v1".into(),
        endpoint_path: "/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_stream_behavior(StreamBehavior::Error(transport_http_status(429)))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\"}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-4o",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let err = match model
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options: websocket_transport_options(),
            ..Default::default()
        })
        .await
    {
        Ok(_) => panic!("rate limit should be surfaced"),
        Err(err) => err,
    };

    match err {
        SdkError::RateLimited { retry_after_ms, .. } => assert_eq!(retry_after_ms, Some(250)),
        other => panic!("expected rate-limited error, got {other:?}"),
    }
    assert_eq!(
        transport.stream_urls(),
        vec!["wss://example.invalid/v1/responses".to_string()]
    );
}

#[tokio::test]
async fn codex_websocket_http_fallback_drops_explicit_previous_response_id_after_reset() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_stream_behavior(StreamBehavior::Chunks(vec![]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http\"}}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let mut session = model.new_turn_session();
    let response = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "third".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([
                    (
                        "transport".into(),
                        json!({
                            "mode": "websocket",
                            "fallback": "http",
                        }),
                    ),
                    ("previousResponseId".into(), json!("resp-stale")),
                ]),
            )]),
            ..Default::default()
        })
        .await
        .expect("fallback stream response");
    let headers = response.response_headers.clone().expect("fallback headers");
    drain_stream_response(response).await;

    assert_eq!(
        headers
            .get("x-ai-sdk-effective-transport")
            .map(String::as_str),
        Some("http")
    );
    assert_eq!(
        headers
            .get("x-ai-sdk-transport-fallback")
            .map(String::as_str),
        Some("websocket->http")
    );
    assert_eq!(
        headers
            .get("x-ai-sdk-provider-session-reset-reason")
            .map(String::as_str),
        Some("websocket_http_fallback")
    );
    assert_eq!(
        transport.last_url().as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
    assert_eq!(
        transport
            .last_body()
            .as_ref()
            .and_then(|body| body.get("previous_response_id"))
            .and_then(|value| value.as_str()),
        None
    );
    assert_eq!(
        transport.websocket_connect_urls(),
        vec!["wss://chatgpt.com/backend-api/codex/responses".to_string()]
    );
    assert_eq!(
        transport.stream_urls(),
        vec!["https://chatgpt.com/backend-api/codex/responses".to_string()]
    );
}

#[tokio::test]
async fn codex_websocket_cold_rate_limit_retries_with_prewarm_without_http_fallback() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_stream_behavior(StreamBehavior::Error(transport_http_status(429)))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"warm-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\"}}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );
    let mut session = model.new_turn_session();
    let response = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "third".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options: websocket_transport_options(),
            ..Default::default()
        })
        .await
        .expect("cold websocket rate limit should retry via warmup");
    let response_headers = response.response_headers.clone().expect("response headers");
    drain_stream_response(response).await;

    let request_bodies = transport.websocket_request_bodies();
    assert_eq!(request_bodies.len(), 3);
    assert_eq!(
        request_bodies[0]
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        None
    );
    assert_eq!(
        request_bodies[1]
            .get("generate")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        request_bodies[2]
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        Some("warm-1")
    );
    assert_eq!(transport.stream_urls(), Vec::<String>::new());
    assert_eq!(
        transport.websocket_connect_urls(),
        vec![
            "wss://chatgpt.com/backend-api/codex/responses".to_string(),
            "wss://chatgpt.com/backend-api/codex/responses".to_string(),
        ]
    );
    assert_eq!(
        response_headers
            .get("x-ai-sdk-provider-session-prewarmed")
            .map(String::as_str),
        Some("true")
    );
}

#[tokio::test]
async fn cold_codex_turn_session_skips_prewarm_when_first_request_succeeds() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_websocket_response_headers(vec![(
            "x-codex-turn-state".to_string(),
            "ts-1".to_string(),
        )])
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"warm-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-2\"}}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let mut session = model.new_turn_session();
    let first = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(function_tool_for_strict_passthrough(None))],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([(
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                )]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("first stream response");
    let first_headers = first.response_headers.clone().expect("first headers");
    drain_stream_response(first).await;

    let second = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "second".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(function_tool_for_strict_passthrough(None))],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([(
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                )]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("second stream response");
    let second_headers = second.response_headers.clone().expect("second headers");
    drain_stream_response(second).await;

    assert_eq!(
        transport.websocket_connect_urls(),
        vec!["wss://chatgpt.com/backend-api/codex/responses".to_string()]
    );
    let request_bodies = transport.websocket_request_bodies();
    assert_eq!(
        request_bodies.len(),
        2,
        "expected real request and follow-up frame"
    );
    let first_real = &request_bodies[0];
    let second_real = &request_bodies[1];
    assert_eq!(
        first_real
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        None
    );
    assert_eq!(
        second_real
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        Some("warm-1")
    );
    assert_eq!(second_real.get("generate"), None);
    assert_eq!(
        first_headers
            .get("x-ai-sdk-provider-session-prewarmed")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        first_headers
            .get("x-ai-sdk-provider-session-request-count")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(
        first_headers
            .get("x-ai-sdk-provider-previous-response-id-used")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        first_headers
            .get("x-ai-sdk-provider-session-warmup-response-id-used")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        first_headers
            .get("x-ai-sdk-provider-session-reused")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-session-connections")
            .map(String::as_str),
        Some("1")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-session-prewarmed")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-session-request-count")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-session-reused")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-previous-response-id-used")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-session-warmup-response-id-used")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        second_headers.get("x-codex-turn-state").map(String::as_str),
        Some("ts-1")
    );
}

#[tokio::test]
async fn explicit_previous_response_id_is_ignored_after_websocket_reconnect_reset() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_close_websocket_after_send()
        .with_websocket_response_headers(vec![(
            "x-codex-turn-state".to_string(),
            "ts-1".to_string(),
        )])
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"warm-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-2\"}}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let mut session = model.new_turn_session();
    let first = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options: websocket_transport_options(),
            ..Default::default()
        })
        .await
        .expect("first stream response");
    drain_stream_response(first).await;

    let second = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "third".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([
                    (
                        "transport".into(),
                        json!({
                            "mode": "websocket",
                            "fallback": "http",
                        }),
                    ),
                    ("previousResponseId".into(), json!("resp-1")),
                ]),
            )]),
            ..Default::default()
        })
        .await
        .expect("second stream response");
    let second_headers = second.response_headers.clone().expect("second headers");
    drain_stream_response(second).await;

    assert_eq!(
        transport.websocket_connect_urls(),
        vec![
            "wss://chatgpt.com/backend-api/codex/responses".to_string(),
            "wss://chatgpt.com/backend-api/codex/responses".to_string(),
        ]
    );
    let request_bodies = transport.websocket_request_bodies();
    assert_eq!(
        request_bodies[1]
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        None
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-session-reset-reason")
            .map(String::as_str),
        Some("websocket_reconnect")
    );
    assert_eq!(
        second_headers
            .get("x-ai-sdk-provider-previous-response-id-used")
            .map(String::as_str),
        Some("false")
    );
}

#[tokio::test]
async fn explicit_previous_response_id_wins_over_cached_warmup_id_on_reused_socket() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_websocket_response_headers(vec![(
            "x-codex-turn-state".to_string(),
            "ts-1".to_string(),
        )])
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"warm-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-2\"}}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let mut session = model.new_turn_session();
    let first = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(function_tool_for_strict_passthrough(None))],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([(
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                )]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("first stream response");
    drain_stream_response(first).await;

    let second = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "second".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(function_tool_for_strict_passthrough(None))],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([
                    (
                        "clientMetadata".into(),
                        json!({
                            "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                        }),
                    ),
                    ("previousResponseId".into(), json!("explicit-final")),
                ]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("second stream response");
    drain_stream_response(second).await;

    let request_bodies = transport.websocket_request_bodies();
    assert_eq!(
        request_bodies[1]
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        Some("explicit-final")
    );
}

#[tokio::test]
async fn codex_turn_session_updates_previous_response_id_before_stream_drain() {
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://chatgpt.com".into(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport = TestTransport::new()
        .with_websocket_response_headers(vec![(
            "x-codex-turn-state".to_string(),
            "ts-1".to_string(),
        )])
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"warm-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\"}}\n\n",
        ))]))
        .with_stream_behavior(StreamBehavior::Chunks(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-2\"}}\n\n",
        ))]));
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5.3-codex",
        cfg,
        transport.clone(),
        TransportConfig::default(),
    );

    let mut session = model.new_turn_session();
    let first = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(function_tool_for_strict_passthrough(None))],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([(
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                )]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("first stream response");
    consume_until_finish_and_drop(first).await;

    let second = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "second".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            tools: vec![v2t::Tool::Function(function_tool_for_strict_passthrough(None))],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([(
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                )]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("second stream response");
    consume_until_finish_and_drop(second).await;

    assert_eq!(
        transport.websocket_connect_urls(),
        vec!["wss://chatgpt.com/backend-api/codex/responses".to_string()]
    );
    let request_bodies = transport.websocket_request_bodies();
    assert_eq!(
        request_bodies[1]
            .get("previous_response_id")
            .and_then(|value| value.as_str()),
        Some("warm-1")
    );
}

#[tokio::test]
async fn codex_first_request_with_upgrade_headers_skips_headerless_preconnect() {
    let server = TestWebsocketServer::start().await;
    let cfg = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: server.base_url.clone(),
        endpoint_path: "/backend-api/codex/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    };
    let transport_cfg = TransportConfig::default();
    let transport = crate::transport_reqwest::ReqwestTransport::new(&transport_cfg);
    let mut model =
        OpenAIResponsesLanguageModel::new("gpt-5.3-codex", cfg, transport, transport_cfg);
    model.start_codex_websocket_preconnect();
    server.wait_for_connection_count(1).await;

    let mut session = model.new_turn_session();
    let response = session
        .do_stream(v2t::CallOptions {
            prompt: vec![v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "hello".into(),
                    provider_options: None,
                }],
                provider_options: None,
            }],
            provider_options: v2t::ProviderOptions::from([(
                "openai".into(),
                HashMap::from([(
                    "clientMetadata".into(),
                    json!({
                        "x-codex-turn-metadata": "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}"
                    }),
                )]),
            )]),
            headers: HashMap::from([(
                "x-codex-turn-metadata".to_string(),
                "{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}".to_string(),
            )]),
            ..Default::default()
        })
        .await
        .expect("stream response");
    drain_stream_response(response).await;

    let connect_headers = server.connect_headers();
    assert_eq!(
        connect_headers.len(),
        2,
        "expected preconnect and fresh request socket"
    );
    assert!(
        !connect_headers[0].contains_key("x-codex-turn-metadata"),
        "preconnect should remain headerless"
    );
    assert_eq!(
        connect_headers[1]
            .get("x-codex-turn-metadata")
            .map(String::as_str),
        Some("{\"cwd\":\"/tmp/project\",\"approval_policy\":\"never\"}")
    );
    let request_records = server.request_records();
    assert_eq!(
        request_records.len(),
        1,
        "expected a single first-request frame"
    );
    assert_eq!(
        request_records[0].0, 2,
        "the first request must use the fresh header-bearing connection"
    );
}

#[tokio::test]
async fn stream_keeps_https_for_standard_openai_responses_path() {
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
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        ..Default::default()
    };

    let _ = model.do_stream(opts).await.expect("stream response");
    let url = transport.last_url().expect("stream url");
    assert_eq!(url, "https://api.openai.com/v1/responses");
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
    let body =
        request_body_for_function_tool(function_tool_for_strict_passthrough(Some(true))).await;
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
    let body =
        request_body_for_function_tool(function_tool_for_strict_passthrough(Some(false))).await;
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
async fn request_body_function_tool_strict_from_provider_options_when_field_unset() {
    let mut function_tool = function_tool_for_strict_passthrough(None);
    function_tool.provider_options = Some(v2t::ProviderOptions::from([(
        "openai".into(),
        HashMap::from([("strict".into(), json!(true))]),
    )]));
    let body = request_body_for_function_tool(function_tool).await;
    let function_tool = body
        .get("tools")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .expect("function tool payload");
    assert_eq!(
        function_tool.get("strict"),
        Some(&json!(true)),
        "function tool strict should fall back to provider option when typed strict is unset"
    );
}

#[tokio::test]
async fn request_body_function_tool_strict_prefers_typed_field_over_provider_options() {
    let mut function_tool = function_tool_for_strict_passthrough(Some(false));
    function_tool.provider_options = Some(v2t::ProviderOptions::from([(
        "openai".into(),
        HashMap::from([("strict".into(), json!(true))]),
    )]));
    let body = request_body_for_function_tool(function_tool).await;
    let function_tool = body
        .get("tools")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .expect("function tool payload");
    assert_eq!(
        function_tool.get("strict"),
        Some(&json!(false)),
        "typed function tool strict should take precedence over provider option strict"
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
    let error_fixture = openai_error_fixture();
    let expected_message = error_fixture
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .expect("openai error fixture message")
        .to_owned();
    let transport = TestTransport::new().with_json_response(error_fixture);
    let model =
        OpenAIResponsesLanguageModel::new("gpt-4o", cfg, transport, TransportConfig::default());

    let err = model
        .do_generate(opts)
        .await
        .expect_err("response.error should propagate as SdkError");
    match err {
        SdkError::Upstream {
            status,
            message,
            source,
        } => {
            assert_eq!(status, 400);
            assert_eq!(message, expected_message);
            assert!(source.is_none());
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
    let model = OpenAIResponsesLanguageModel::new(
        "gpt-5-codex",
        cfg,
        transport,
        TransportConfig::default(),
    );

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
