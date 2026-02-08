use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::LanguageModel;
use crate::ai_sdk_providers_anthropic::messages::language_model::AnthropicMessagesConfig;
use crate::ai_sdk_providers_anthropic::AnthropicMessagesLanguageModel;
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::stream;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct TestTransport {
    last_headers: Arc<Mutex<Option<Vec<(String, String)>>>>,
}

impl TestTransport {
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
        _body: &serde_json::Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        *self.last_headers.lock().unwrap() = Some(headers.to_vec());
        Ok(TestStreamResponse {
            headers: vec![],
            chunks: vec![],
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
        _form: &crate::ai_sdk_core::transport::MultipartForm,
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
