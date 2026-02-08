use crate::core::error::TransportError;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use serde_json::Value;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

#[derive(Clone, Debug)]
pub struct TransportConfig {
    /// Overall request timeout (optional; if None, rely on connect + idle)
    pub request_timeout: Option<Duration>,
    /// TCP connect timeout
    pub connect_timeout: Duration,
    /// Per-chunk idle read timeout
    pub idle_read_timeout: Duration,
    /// Whether to strip object fields with null values from JSON bodies before sending
    pub strip_null_fields: bool,
}

/// Body payload captured for transport observers.
#[derive(Debug, Clone)]
pub enum TransportBody {
    Json(Value),
    Text(String),
}

#[derive(Debug, Clone)]
pub struct MultipartForm {
    pub fields: Vec<MultipartField>,
}

impl MultipartForm {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn push_text(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.fields.push(MultipartField {
            name: name.into(),
            value: MultipartValue::Text(value.into()),
        });
    }

    pub fn push_bytes(
        &mut self,
        name: impl Into<String>,
        data: Vec<u8>,
        filename: Option<String>,
        content_type: Option<String>,
    ) {
        self.fields.push(MultipartField {
            name: name.into(),
            value: MultipartValue::Bytes {
                data,
                filename,
                content_type,
            },
        });
    }
}

#[derive(Debug, Clone)]
pub struct MultipartField {
    pub name: String,
    pub value: MultipartValue,
}

#[derive(Debug, Clone)]
pub enum MultipartValue {
    Text(String),
    Bytes {
        data: Vec<u8>,
        filename: Option<String>,
        content_type: Option<String>,
    },
}

/// Structured event emitted by transport implementations.
#[derive(Debug, Clone)]
pub struct TransportEvent {
    pub started_at: SystemTime,
    pub latency: Option<Duration>,
    pub method: String,
    pub url: String,
    pub status: Option<u16>,
    pub request_headers: Vec<(String, String)>,
    pub response_headers: Vec<(String, String)>,
    pub request_body: Option<TransportBody>,
    pub response_body: Option<TransportBody>,
    pub response_size: Option<usize>,
    pub error: Option<String>,
    pub is_stream: bool,
}

/// Observer hook for transport events.
pub trait TransportObserver: Send + Sync {
    fn on_event(&self, event: TransportEvent);
}

static TRANSPORT_OBSERVER: OnceLock<Arc<dyn TransportObserver>> = OnceLock::new();

/// Register a transport observer (one-time).
pub fn set_transport_observer(observer: Arc<dyn TransportObserver>) -> bool {
    TRANSPORT_OBSERVER.set(observer).is_ok()
}

/// Emit a transport event if an observer is registered.
pub fn emit_transport_event(event: TransportEvent) {
    if let Some(observer) = TRANSPORT_OBSERVER.get() {
        observer.on_event(event);
    }
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            request_timeout: None,
            connect_timeout: Duration::from_secs(10),
            idle_read_timeout: Duration::from_secs(45),
            strip_null_fields: true,
        }
    }
}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    /// Response for a successful streaming HTTP request.
    /// Contains the response headers and the byte stream body.
    type StreamResponse: Send;

    /// Extract the underlying byte stream from the transport-specific response wrapper.
    fn into_stream(
        resp: Self::StreamResponse,
    ) -> (
        Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
        Vec<(String, String)>,
    );

    async fn post_json_stream(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError>;

    /// Perform a JSON POST request and return the parsed JSON body along with response headers.
    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError>;

    /// Perform a multipart/form-data POST request and return the parsed JSON body along with response headers.
    async fn post_multipart(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _form: &MultipartForm,
        _cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        Err(TransportError::Other(
            "multipart form-data is not supported by this transport".into(),
        ))
    }

    /// Perform a GET request and return the response bytes along with headers.
    async fn get_bytes(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _cfg: &TransportConfig,
    ) -> Result<(Bytes, Vec<(String, String)>), TransportError> {
        Err(TransportError::Other(
            "byte downloads are not supported by this transport".into(),
        ))
    }
}
