use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::json::without_null_fields;
use crate::ai_sdk_core::transport::{HttpTransport, MultipartForm, MultipartValue, TransportConfig};
use crate::ai_sdk_core::ImageModel;
use crate::ai_sdk_providers_openai_compatible::image::image_model::{
    OpenAICompatibleImageConfig, OpenAICompatibleImageModel,
};
use crate::ai_sdk_types::image::{ImageData, ImageFile, ImageOptions, ImageWarning};
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::stream;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

#[derive(Clone)]
struct TestTransport {
    json_response: Arc<Mutex<serde_json::Value>>,
    multipart_response: Arc<Mutex<serde_json::Value>>,
    response_headers: Arc<Mutex<Vec<(String, String)>>>,
    last_body: Arc<Mutex<Option<serde_json::Value>>>,
    last_headers: Arc<Mutex<Option<Vec<(String, String)>>>>,
    last_url: Arc<Mutex<Option<String>>>,
    last_form: Arc<Mutex<Option<MultipartForm>>>,
    download_payload: Arc<Mutex<Option<(Vec<u8>, Vec<(String, String)>)>>>,
}

impl TestTransport {
    fn new(response_json: serde_json::Value) -> Self {
        Self {
            json_response: Arc::new(Mutex::new(response_json.clone())),
            multipart_response: Arc::new(Mutex::new(response_json)),
            response_headers: Arc::new(Mutex::new(vec![])),
            last_body: Arc::new(Mutex::new(None)),
            last_headers: Arc::new(Mutex::new(None)),
            last_url: Arc::new(Mutex::new(None)),
            last_form: Arc::new(Mutex::new(None)),
            download_payload: Arc::new(Mutex::new(None)),
        }
    }

    fn with_response_headers(self, headers: Vec<(String, String)>) -> Self {
        *self.response_headers.lock().unwrap() = headers;
        self
    }

    fn with_multipart_response(self, response_json: serde_json::Value) -> Self {
        *self.multipart_response.lock().unwrap() = response_json;
        self
    }

    fn with_download_payload(self, bytes: Vec<u8>, headers: Vec<(String, String)>) -> Self {
        *self.download_payload.lock().unwrap() = Some((bytes, headers));
        self
    }

    fn last_body(&self) -> Option<serde_json::Value> {
        self.last_body.lock().unwrap().clone()
    }

    fn last_headers(&self) -> Option<Vec<(String, String)>> {
        self.last_headers.lock().unwrap().clone()
    }

    fn last_url(&self) -> Option<String> {
        self.last_url.lock().unwrap().clone()
    }

    fn last_form(&self) -> Option<MultipartForm> {
        self.last_form.lock().unwrap().clone()
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
        _headers: &[(String, String)],
        _body: &serde_json::Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        Err(TransportError::Other("post_json_stream unused".into()))
    }

    async fn post_json(
        &self,
        url: &str,
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
        *self.last_url.lock().unwrap() = Some(url.to_string());
        Ok((
            self.json_response.lock().unwrap().clone(),
            self.response_headers.lock().unwrap().clone(),
        ))
    }

    async fn post_multipart(
        &self,
        url: &str,
        headers: &[(String, String)],
        form: &MultipartForm,
        _cfg: &TransportConfig,
    ) -> Result<(serde_json::Value, Vec<(String, String)>), TransportError> {
        *self.last_headers.lock().unwrap() = Some(headers.to_vec());
        *self.last_url.lock().unwrap() = Some(url.to_string());
        *self.last_form.lock().unwrap() = Some(form.clone());
        Ok((
            self.multipart_response.lock().unwrap().clone(),
            self.response_headers.lock().unwrap().clone(),
        ))
    }

    async fn get_bytes(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _cfg: &TransportConfig,
    ) -> Result<(Bytes, Vec<(String, String)>), TransportError> {
        let payload = self
            .download_payload
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| TransportError::Other("download payload not set".into()))?;
        Ok((Bytes::from(payload.0), payload.1))
    }
}

fn build_model(transport: TestTransport) -> OpenAICompatibleImageModel<TestTransport> {
    let cfg = OpenAICompatibleImageConfig {
        provider_scope_name: "test-provider".into(),
        base_url: "https://api.example.com/v1".into(),
        headers: vec![("authorization".into(), "Bearer test-api-key".into())],
        http: transport,
        transport_cfg: TransportConfig::default(),
        query_params: vec![],
        default_options: None,
    };
    OpenAICompatibleImageModel::new("dall-e-3", cfg)
}

#[tokio::test]
async fn passes_generation_body_with_provider_options() {
    let response = json!({
        "data": [
            {"b64_json": "test1234"},
            {"b64_json": "test5678"}
        ]
    });
    let transport = TestTransport::new(response);
    let model = build_model(transport.clone());

    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([
            ("user".into(), json!("base-user")),
            ("quality".into(), json!("hd")),
        ]),
    );

    let options = ImageOptions {
        prompt: Some("A photorealistic astronaut riding a horse".into()),
        n: 2,
        size: Some("1024x1024".into()),
        provider_options,
        ..Default::default()
    };

    let result = model.do_generate(options).await.expect("generate response");
    assert_eq!(result.images.len(), 2);

    assert_eq!(
        transport.last_body().unwrap(),
        json!({
            "model": "dall-e-3",
            "prompt": "A photorealistic astronaut riding a horse",
            "n": 2,
            "size": "1024x1024",
            "user": "base-user",
            "quality": "hd",
            "response_format": "b64_json"
        })
    );
}

#[tokio::test]
async fn adds_warnings_for_unsupported_settings() {
    let response = json!({
        "data": [
            {"b64_json": "test1234"}
        ]
    });
    let transport = TestTransport::new(response);
    let model = build_model(transport);

    let options = ImageOptions {
        prompt: Some("A photorealistic astronaut riding a horse".into()),
        aspect_ratio: Some("16:9".into()),
        seed: Some(123),
        ..Default::default()
    };

    let result = model.do_generate(options).await.expect("generate response");
    assert_eq!(
        result.warnings,
        vec![
            ImageWarning::Unsupported {
                feature: "aspectRatio".into(),
                details: Some(
                    "This model does not support aspect ratio. Use `size` instead.".into()
                ),
            },
            ImageWarning::Unsupported {
                feature: "seed".into(),
                details: None,
            },
        ]
    );
}

#[tokio::test]
async fn merges_provider_and_request_headers() {
    let response = json!({
        "data": [
            {"b64_json": "test1234"}
        ]
    });
    let transport = TestTransport::new(response);
    let cfg = OpenAICompatibleImageConfig {
        provider_scope_name: "test-provider".into(),
        base_url: "https://api.example.com/v1".into(),
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
        default_options: None,
    };
    let model = OpenAICompatibleImageModel::new("dall-e-3", cfg);

    let options = ImageOptions {
        prompt: Some("A photorealistic astronaut riding a horse".into()),
        headers: HashMap::from([(
            "Custom-Request-Header".into(),
            "request-header-value".into(),
        )]),
        ..Default::default()
    };

    let _ = model.do_generate(options).await.expect("generate response");

    let headers = transport.last_headers().unwrap();
    let as_map: HashMap<String, String> = headers
        .into_iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect();
    assert_eq!(
        as_map.get("custom-provider-header"),
        Some(&"provider-header-value".to_string())
    );
    assert_eq!(
        as_map.get("custom-request-header"),
        Some(&"request-header-value".to_string())
    );
    assert_eq!(
        as_map.get("content-type"),
        Some(&"application/json".to_string())
    );
}

#[tokio::test]
async fn uses_edit_endpoint_with_files_and_mask() {
    let response = json!({
        "data": [
            {"b64_json": "edited-image-base64"}
        ]
    });
    let transport = TestTransport::new(json!({ "data": [] }))
        .with_multipart_response(response.clone())
        .with_download_payload(
            vec![1, 2, 3],
            vec![("content-type".into(), "image/png".into())],
        );
    let model = build_model(transport.clone());

    let options = ImageOptions {
        prompt: Some("Edit this image".into()),
        files: vec![ImageFile::Url {
            url: "https://example.com/image.png".into(),
        }],
        mask: Some(ImageFile::File {
            media_type: "image/png".into(),
            data: ImageData::Bytes {
                bytes: vec![4, 5, 6],
            },
        }),
        ..Default::default()
    };

    let result = model.do_generate(options).await.expect("edit response");
    assert_eq!(
        result.images,
        vec![ImageData::Base64("edited-image-base64".into())]
    );
    assert_eq!(
        transport.last_url().unwrap(),
        "https://api.example.com/v1/images/edits"
    );

    let form = transport.last_form().expect("multipart form");
    let image_fields: Vec<_> = form
        .fields
        .iter()
        .filter(|field| field.name == "image")
        .collect();
    assert_eq!(image_fields.len(), 1);
    if let MultipartValue::Bytes { data, .. } = &image_fields[0].value {
        assert_eq!(data, &vec![1, 2, 3]);
    } else {
        panic!("expected image bytes");
    }

    let mask_fields: Vec<_> = form
        .fields
        .iter()
        .filter(|field| field.name == "mask")
        .collect();
    assert_eq!(mask_fields.len(), 1);
}

#[tokio::test]
async fn response_metadata_has_timestamp_and_model_id() {
    let response = json!({
        "data": [
            {"b64_json": "test1234"}
        ]
    });
    let transport = TestTransport::new(response)
        .with_response_headers(vec![("content-type".into(), "application/json".into())]);
    let model = build_model(transport);

    let before = SystemTime::now();
    let result = model
        .do_generate(ImageOptions::new(Some("test".into())))
        .await
        .expect("generate response");
    let after = SystemTime::now();

    assert!(result.response.timestamp >= before);
    assert!(result.response.timestamp <= after);
    assert_eq!(result.response.model_id, "dall-e-3");
    assert_eq!(
        result.response.headers,
        Some(HashMap::from([(
            "content-type".into(),
            "application/json".into()
        )]))
    );
}
