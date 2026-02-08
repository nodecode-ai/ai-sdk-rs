use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::json::without_null_fields;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::EmbeddingModel;
use crate::ai_sdk_providers_openai_compatible::embedding::embedding_model::{
    OpenAICompatibleEmbeddingConfig, OpenAICompatibleEmbeddingModel,
};
use crate::ai_sdk_types::embedding::EmbedOptions;
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::stream;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

const DEFAULT_MAX_EMBEDDINGS_PER_CALL: usize = 2048;

#[derive(Clone)]
struct TestTransport {
    response_json: Arc<Mutex<serde_json::Value>>,
    response_headers: Arc<Mutex<Vec<(String, String)>>>,
    stream_chunks: Arc<Mutex<Vec<Result<Bytes, TransportError>>>>,
    last_body: Arc<Mutex<Option<serde_json::Value>>>,
    last_headers: Arc<Mutex<Option<Vec<(String, String)>>>>,
}

impl TestTransport {
    fn new(response_json: serde_json::Value) -> Self {
        Self {
            response_json: Arc::new(Mutex::new(response_json)),
            response_headers: Arc::new(Mutex::new(vec![])),
            stream_chunks: Arc::new(Mutex::new(vec![])),
            last_body: Arc::new(Mutex::new(None)),
            last_headers: Arc::new(Mutex::new(None)),
        }
    }

    fn with_response_headers(self, headers: Vec<(String, String)>) -> Self {
        *self.response_headers.lock().unwrap() = headers;
        self
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
        let cleaned = if cfg.strip_null_fields {
            without_null_fields(body)
        } else {
            body.clone()
        };
        *self.last_body.lock().unwrap() = Some(cleaned);
        *self.last_headers.lock().unwrap() = Some(headers.to_vec());
        let mut guard = self.stream_chunks.lock().unwrap();
        let chunks = std::mem::take(&mut *guard);
        Ok(TestStreamResponse {
            headers: self.response_headers.lock().unwrap().clone(),
            chunks,
        })
    }

    async fn post_json(
        &self,
        _url: &str,
        headers: &[(String, String)],
        body: &serde_json::Value,
        cfg: &TransportConfig,
    ) -> Result<(serde_json::Value, Vec<(String, String)>), TransportError> {
        let cleaned = if cfg.strip_null_fields {
            without_null_fields(body)
        } else {
            body.clone()
        };
        *self.last_body.lock().unwrap() = Some(cleaned);
        *self.last_headers.lock().unwrap() = Some(headers.to_vec());
        Ok((
            self.response_json.lock().unwrap().clone(),
            self.response_headers.lock().unwrap().clone(),
        ))
    }
}

fn build_model(transport: TestTransport) -> OpenAICompatibleEmbeddingModel<TestTransport> {
    let cfg = OpenAICompatibleEmbeddingConfig {
        provider_scope_name: "test-provider".into(),
        base_url: "https://my.api.com/v1".into(),
        headers: vec![("authorization".into(), "Bearer test-api-key".into())],
        http: transport,
        transport_cfg: TransportConfig::default(),
        query_params: vec![],
        max_embeddings_per_call: Some(DEFAULT_MAX_EMBEDDINGS_PER_CALL),
        supports_parallel_calls: true,
        default_options: None,
    };
    OpenAICompatibleEmbeddingModel::new("text-embedding-3-large", cfg)
}

#[tokio::test]
async fn extracts_embeddings_usage_and_headers() {
    let response = json!({
        "object": "list",
        "data": [
            {"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},
            {"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]}
        ],
        "model":"text-embedding-3-large",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "providerMetadata": {"test-provider": {"foo":"bar"}}
    });
    let transport = TestTransport::new(response).with_response_headers(vec![
        ("content-length".into(), "236".into()),
        ("content-type".into(), "application/json".into()),
        ("test-header".into(), "test-value".into()),
    ]);
    let model = build_model(transport);

    let result = model
        .do_embed(EmbedOptions::new(vec![
            "sunny day at the beach".into(),
            "rainy day in the city".into(),
        ]))
        .await
        .expect("embed response");

    assert_eq!(
        result.embeddings,
        vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]
    );
    let usage = result.usage.as_ref().expect("usage");
    assert_eq!(usage.tokens, Some(8));
    assert_eq!(
        result.response_headers,
        Some(HashMap::from([
            ("content-length".into(), "236".into()),
            ("content-type".into(), "application/json".into()),
            ("test-header".into(), "test-value".into())
        ]))
    );
    assert_eq!(
        result.provider_metadata.expect("provider metadata")["test-provider"]["foo"],
        json!("bar")
    );
}

#[tokio::test]
async fn passes_model_and_values_in_request_body() {
    let response = json!({
        "object": "list",
        "data": [
            {"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}
        ],
        "model":"text-embedding-3-large"
    });
    let transport = TestTransport::new(response.clone());
    let model = build_model(transport.clone());

    let result = model
        .do_embed(EmbedOptions::new(vec![
            "sunny day at the beach".into(),
            "rainy day in the city".into(),
        ]))
        .await
        .expect("embed response");

    assert_eq!(
        result.request_body,
        Some(json!({
            "model": "text-embedding-3-large",
            "input": ["sunny day at the beach","rainy day in the city"],
            "encoding_format": "float"
        }))
    );

    assert_eq!(
        transport.last_body().unwrap(),
        json!({
            "model": "text-embedding-3-large",
            "input": ["sunny day at the beach","rainy day in the city"],
            "encoding_format": "float"
        })
    );
}

#[tokio::test]
async fn includes_dimensions_from_provider_options() {
    let response = json!({
        "object": "list",
        "data": [
            {"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}
        ],
        "model":"text-embedding-3-large"
    });
    let transport = TestTransport::new(response);
    let model = build_model(transport.clone());

    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai-compatible".into(),
        HashMap::from([
            ("user".into(), json!("base-user")),
            ("baseOnly".into(), json!("ignored")),
        ]),
    );
    provider_options.insert(
        "test-provider".into(),
        HashMap::from([("dimensions".into(), json!(64))]),
    );

    let _ = model
        .do_embed(EmbedOptions {
            values: vec!["sunny day at the beach".into()],
            headers: HashMap::new(),
            provider_options,
        })
        .await
        .expect("embed response");

    assert_eq!(
        transport.last_body().unwrap(),
        json!({
            "model": "text-embedding-3-large",
            "input": ["sunny day at the beach"],
            "encoding_format": "float",
            "dimensions": 64,
            "user": "base-user"
        })
    );
}

#[tokio::test]
async fn merges_provider_and_request_headers() {
    let response = json!({
        "object": "list",
        "data": [
            {"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}
        ],
        "model":"text-embedding-3-large"
    });
    let transport = TestTransport::new(response);
    let cfg = OpenAICompatibleEmbeddingConfig {
        provider_scope_name: "test-provider".into(),
        base_url: "https://my.api.com/v1".into(),
        headers: vec![
            ("authorization".into(), "Bearer test-api-key".into()),
            (
                "custom-provider-header".into(),
                "provider-header-value".into(),
            ),
        ],
        http: transport.clone(),
        transport_cfg: TransportConfig::default(),
        query_params: vec![],
        max_embeddings_per_call: Some(DEFAULT_MAX_EMBEDDINGS_PER_CALL),
        supports_parallel_calls: true,
        default_options: None,
    };
    let model = OpenAICompatibleEmbeddingModel::new("text-embedding-3-large", cfg);

    let _ = model
        .do_embed(EmbedOptions {
            values: vec!["sunny day at the beach".into()],
            headers: HashMap::from([(
                "Custom-Request-Header".into(),
                "request-header-value".into(),
            )]),
            provider_options: HashMap::new(),
        })
        .await
        .expect("embed response");

    let headers = transport.last_headers().unwrap();
    let as_map: HashMap<String, String> = headers
        .into_iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect();
    assert_eq!(
        as_map,
        HashMap::from([
            ("authorization".into(), "Bearer test-api-key".into()),
            ("content-type".into(), "application/json".into()),
            ("accept".into(), "application/json".into()),
            (
                "custom-provider-header".into(),
                "provider-header-value".into(),
            ),
            (
                "custom-request-header".into(),
                "request-header-value".into(),
            )
        ])
    );
}
