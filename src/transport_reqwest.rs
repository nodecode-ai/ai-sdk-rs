use crate::core::error::{display_body_for_error, TransportError};
use crate::core::transport::{
    emit_transport_event, HttpTransport, JsonStreamWebsocketConnection, MultipartForm,
    MultipartValue, TransportBody, TransportConfig, TransportEvent, TransportStream,
};
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde_json::Value;
use std::borrow::Cow;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error as WsError, Message as WsMessage};
use tokio_tungstenite::{client_async_tls_with_config, connect_async};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tracing::debug;
use url::Url;

#[derive(Clone)]
pub struct ReqwestTransport {
    client: Client,
}

type ReqwestWebsocketStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedProxy {
    host: String,
    port: u16,
    authorization: Option<String>,
}

struct ReqwestJsonStreamWebsocketConnection {
    socket: Arc<Mutex<ReqwestWebsocketStream>>,
    response_headers: Vec<(String, String)>,
    closed: Arc<AtomicBool>,
}

impl ReqwestTransport {
    fn json_request_body<'a>(body: &'a Value, cfg: &TransportConfig) -> Cow<'a, Value> {
        if cfg.strip_null_fields {
            Cow::Owned(crate::core::json::without_null_fields(body))
        } else {
            Cow::Borrowed(body)
        }
    }

    fn configure_builder(
        mut builder: reqwest::ClientBuilder,
        cfg: &TransportConfig,
    ) -> reqwest::ClientBuilder {
        builder = builder
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .pool_idle_timeout(Duration::from_secs(90))
            // reqwest 0.12 no longer exposes `http2_keep_alive_interval`; use
            // the cross-protocol TCP keepalive interval instead.
            .tcp_keepalive_interval(Duration::from_secs(30));
        if let Some(req_timeout) = cfg.request_timeout {
            builder = builder.timeout(req_timeout);
        }
        // connect timeout
        builder.connect_timeout(cfg.connect_timeout)
    }

    fn try_new_with_builder(
        cfg: &TransportConfig,
        builder: reqwest::ClientBuilder,
    ) -> Result<Self, TransportError> {
        let builder = Self::configure_builder(builder, cfg);
        let client = builder.build().map_err(|err| {
            TransportError::Other(format!(
                "reqwest client build failed: {}",
                format_reqwest_error_chain(&err)
            ))
        })?;
        Ok(Self { client })
    }

    fn new_with_builder(cfg: &TransportConfig, builder: reqwest::ClientBuilder) -> Self {
        // Keep compatibility with existing call sites while removing panics.
        match Self::try_new_with_builder(cfg, builder) {
            Ok(transport) => transport,
            Err(err) => {
                debug!(
                    target: "ai_sdk::transport::reqwest",
                    error = %err,
                    "falling back to reqwest::Client::new after transport init failure"
                );
                Self {
                    client: Client::new(),
                }
            }
        }
    }

    pub fn try_new(cfg: &TransportConfig) -> Result<Self, TransportError> {
        Self::try_new_with_builder(cfg, Client::builder())
    }

    pub fn new(cfg: &TransportConfig) -> Self {
        Self::new_with_builder(cfg, Client::builder())
    }

    fn is_websocket_url(url: &str) -> bool {
        url.starts_with("ws://") || url.starts_with("wss://")
    }

    async fn connect_websocket_stream(
        &self,
        request: http::Request<()>,
        cfg: &TransportConfig,
    ) -> Result<(ReqwestWebsocketStream, http::Response<Option<Vec<u8>>>), TransportError> {
        let request_url = request.uri().to_string();
        if let Some(proxy) = resolve_proxy_for_websocket_url(&request_url)? {
            let tunnel = open_http_proxy_tunnel(request.uri(), &proxy, cfg).await?;
            return client_async_tls_with_config(request, tunnel, None, None)
                .await
                .map_err(|err| {
                    let (mapped, _, _) = map_websocket_connect_error(err);
                    mapped
                });
        }

        connect_async(request).await.map_err(|err| {
            let (mapped, _, _) = map_websocket_connect_error(err);
            mapped
        })
    }

    async fn open_json_stream_websocket(
        &self,
        url: &str,
        headers: &[(String, String)],
        cfg: &TransportConfig,
    ) -> Result<ReqwestJsonStreamWebsocketConnection, TransportError> {
        let started_at = SystemTime::now();
        let start_instant = Instant::now();
        let request_headers = headers.to_vec();
        let request_body = None;

        let mut request = url.into_client_request().map_err(|err| {
            TransportError::Other(format!("invalid websocket url '{url}': {err}"))
        })?;
        for (name, value) in headers {
            if should_skip_websocket_header(name) {
                continue;
            }
            let Ok(header_name) = name.parse::<http::header::HeaderName>() else {
                continue;
            };
            let Ok(header_value) = value.parse::<http::header::HeaderValue>() else {
                continue;
            };
            request.headers_mut().insert(header_name, header_value);
        }

        let connect_result = tokio::time::timeout(
            cfg.connect_timeout,
            self.connect_websocket_stream(request, cfg),
        )
        .await;
        let (socket, response) = match connect_result {
            Err(_) => {
                let err = TransportError::ConnectTimeout(cfg.connect_timeout);
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method: "GET".to_string(),
                    url: url.to_string(),
                    status: None,
                    request_headers,
                    response_headers: Vec::new(),
                    request_body,
                    response_body: None,
                    response_size: None,
                    error: Some(err.to_string()),
                    is_stream: true,
                });
                return Err(err);
            }
            Ok(Err(err)) => {
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method: "GET".to_string(),
                    url: url.to_string(),
                    status: err.status(),
                    request_headers,
                    response_headers: websocket_connect_error_headers(&err),
                    request_body,
                    response_body: None,
                    response_size: None,
                    error: Some(err.to_string()),
                    is_stream: true,
                });
                return Err(err);
            }
            Ok(Ok(ok)) => ok,
        };

        let response_headers = header_pairs(response.headers());
        emit_transport_event(TransportEvent {
            started_at,
            latency: Some(start_instant.elapsed()),
            method: "GET".to_string(),
            url: url.to_string(),
            status: Some(response.status().as_u16()),
            request_headers,
            response_headers: response_headers.clone(),
            request_body,
            response_body: None,
            response_size: None,
            error: None,
            is_stream: true,
        });

        Ok(ReqwestJsonStreamWebsocketConnection {
            socket: Arc::new(Mutex::new(socket)),
            response_headers,
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    async fn post_json_stream_websocket(
        &self,
        url: &str,
        headers: &[(String, String)],
        cleaned_body: &Value,
        cfg: &TransportConfig,
    ) -> Result<(TransportStream, Vec<(String, String)>), TransportError> {
        let connection = self.open_json_stream_websocket(url, headers, cfg).await?;
        let response_headers = connection.response_headers();
        let stream = connection.send_json_stream(cleaned_body, cfg).await?;
        Ok((stream, response_headers))
    }
}

#[async_trait]
impl JsonStreamWebsocketConnection for ReqwestJsonStreamWebsocketConnection {
    async fn send_json_stream(
        &self,
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<TransportStream, TransportError> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(TransportError::StreamClosed);
        }

        let payload = serde_json::to_string(body).map_err(|err| {
            TransportError::Other(format!("failed to encode websocket payload: {err}"))
        })?;
        let idle = cfg.idle_read_timeout;
        let socket = Arc::clone(&self.socket);
        let closed = Arc::clone(&self.closed);
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Bytes, TransportError>>(32);

        tokio::spawn(async move {
            let mut socket = socket.lock().await;
            if closed.load(Ordering::SeqCst) {
                let _ = tx.send(Err(TransportError::StreamClosed)).await;
                return;
            }

            if let Err(err) = socket.send(WsMessage::Text(payload)).await {
                closed.store(true, Ordering::SeqCst);
                let _ = tx.send(Err(map_websocket_stream_error(err, idle))).await;
                return;
            }

            loop {
                let next = tokio::time::timeout(idle, socket.next()).await;
                let outcome = match next {
                    Err(_) => {
                        closed.store(true, Ordering::SeqCst);
                        let _ = tx.send(Err(TransportError::IdleReadTimeout(idle))).await;
                        break;
                    }
                    Ok(None) => {
                        closed.store(true, Ordering::SeqCst);
                        let _ = tx.send(Err(TransportError::StreamClosed)).await;
                        break;
                    }
                    Ok(Some(Err(err))) => {
                        closed.store(true, Ordering::SeqCst);
                        let _ = tx.send(Err(map_websocket_stream_error(err, idle))).await;
                        break;
                    }
                    Ok(Some(Ok(message))) => websocket_message_to_sse_chunk(message),
                };

                match outcome {
                    Ok(WebsocketMessageOutcome { chunk, terminal }) => {
                        if let Some(chunk) = chunk {
                            if tx.send(Ok(chunk)).await.is_err() {
                                closed.store(true, Ordering::SeqCst);
                                break;
                            }
                        }
                        if terminal {
                            break;
                        }
                    }
                    Err(err) => {
                        if matches!(err, TransportError::StreamClosed) {
                            closed.store(true, Ordering::SeqCst);
                        }
                        let _ = tx.send(Err(err)).await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(async_stream::stream! {
            while let Some(item) = rx.recv().await {
                yield item;
            }
        }))
    }

    fn response_headers(&self) -> Vec<(String, String)> {
        self.response_headers.clone()
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new(&TransportConfig::default())
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    type StreamResponse = (TransportStream, Vec<(String, String)>);

    fn into_stream(resp: Self::StreamResponse) -> (TransportStream, Vec<(String, String)>) {
        resp
    }

    async fn post_json_stream(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        let cleaned_body = Self::json_request_body(body, cfg);

        if Self::is_websocket_url(url) {
            return self
                .post_json_stream_websocket(url, headers, cleaned_body.as_ref(), cfg)
                .await;
        }

        // Build request
        let mut req = self.client.post(url).json(cleaned_body.as_ref());
        for (k, v) in headers {
            // Skip Content-Type as .json() already sets it
            if !k.eq_ignore_ascii_case("content-type") {
                req = req.header(k, v);
            }
        }

        let started_at = SystemTime::now();
        let start_instant = Instant::now();
        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let (method, request_url, request_headers) = if let Some(req) = request_snapshot.as_ref() {
            (
                req.method().to_string(),
                req.url().to_string(),
                header_pairs(req.headers()),
            )
        } else {
            ("POST".to_string(), url.to_string(), headers.to_vec())
        };
        let request_body = Some(TransportBody::Json(cleaned_body.as_ref().clone()));

        // Send
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method,
                    url: request_url,
                    status: None,
                    request_headers,
                    response_headers: Vec::new(),
                    request_body,
                    response_body: None,
                    response_size: None,
                    error: Some(detail.clone()),
                    is_stream: true,
                });
                return Err(if e.is_connect() {
                    TransportError::Network(format!("connect: {detail}"))
                } else if e.is_timeout() {
                    TransportError::ConnectTimeout(cfg.connect_timeout)
                } else {
                    TransportError::Network(detail)
                });
            }
        };

        let status = resp.status();
        // Error: read body (already done in current code) and log it
        if !status.is_success() {
            let retry_after_ms = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(parse_retry_after_ms);

            // Get headers before consuming response
            let res_headers = resp
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
                .collect::<Vec<_>>();

            let body_text = resp.text().await.unwrap_or_default();
            let sanitized = display_body_for_error(&body_text);
            emit_transport_event(TransportEvent {
                started_at,
                latency: Some(start_instant.elapsed()),
                method,
                url: request_url,
                status: Some(status.as_u16()),
                request_headers,
                response_headers: res_headers.clone(),
                request_body,
                response_body: Some(TransportBody::Text(body_text.clone())),
                response_size: Some(body_text.len()),
                error: Some(format!("HTTP {}: {}", status.as_u16(), sanitized)),
                is_stream: true,
            });
            return Err(TransportError::HttpStatus {
                status: status.as_u16(),
                body: body_text,
                retry_after_ms,
                sanitized,
                headers: res_headers,
            });
        }

        // Success: stream the bytes with idle timeout enforcement
        let idle = cfg.idle_read_timeout;
        // Collect response headers for return
        let res_headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
            .collect::<Vec<_>>();

        emit_transport_event(TransportEvent {
            started_at,
            latency: Some(start_instant.elapsed()),
            method,
            url: request_url,
            status: Some(status.as_u16()),
            request_headers,
            response_headers: res_headers.clone(),
            request_body,
            response_body: None,
            response_size: None,
            error: None,
            is_stream: true,
        });

        let mut inner = resp.bytes_stream();

        let s = async_stream::try_stream! {
            loop {
                let next = tokio::time::timeout(idle, inner.next()).await;
                match next {
                    Err(_) => Err(TransportError::IdleReadTimeout(idle))?,
                    Ok(None) => break,
                    Ok(Some(Err(e))) => {
                        if e.is_timeout() { Err(TransportError::IdleReadTimeout(idle))?; }
                        else { Err(TransportError::BodyRead(e.to_string()))?; }
                    }
                    Ok(Some(Ok(bytes))) => { yield bytes; }
                }
            }
        };
        Ok((Box::pin(s), res_headers))
    }

    async fn connect_json_stream_websocket(
        &self,
        url: &str,
        headers: &[(String, String)],
        cfg: &TransportConfig,
    ) -> Result<Box<dyn JsonStreamWebsocketConnection>, TransportError> {
        Ok(Box::new(
            self.open_json_stream_websocket(url, headers, cfg).await?,
        ))
    }

    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        let cleaned_body = Self::json_request_body(body, cfg);
        let mut req = self.client.post(url).json(cleaned_body.as_ref());
        for (k, v) in headers {
            if !k.eq_ignore_ascii_case("content-type") {
                req = req.header(k, v);
            }
        }

        let started_at = SystemTime::now();
        let start_instant = Instant::now();
        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let (method, request_url, request_headers) = if let Some(req) = request_snapshot.as_ref() {
            (
                req.method().to_string(),
                req.url().to_string(),
                header_pairs(req.headers()),
            )
        } else {
            ("POST".to_string(), url.to_string(), headers.to_vec())
        };
        let request_body = Some(TransportBody::Json(cleaned_body.as_ref().clone()));

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method,
                    url: request_url,
                    status: None,
                    request_headers,
                    response_headers: Vec::new(),
                    request_body,
                    response_body: None,
                    response_size: None,
                    error: Some(detail.clone()),
                    is_stream: false,
                });
                return Err(if e.is_connect() {
                    TransportError::Network(format!("connect: {detail}"))
                } else if e.is_timeout() {
                    TransportError::ConnectTimeout(cfg.connect_timeout)
                } else {
                    TransportError::Network(detail)
                });
            }
        };

        let status = resp.status();
        let res_headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
            .collect::<Vec<_>>();

        if !status.is_success() {
            let retry_after_ms = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(parse_retry_after_ms);
            let body_text = resp.text().await.unwrap_or_default();
            let sanitized = display_body_for_error(&body_text);
            emit_transport_event(TransportEvent {
                started_at,
                latency: Some(start_instant.elapsed()),
                method,
                url: request_url,
                status: Some(status.as_u16()),
                request_headers,
                response_headers: res_headers.clone(),
                request_body,
                response_body: Some(TransportBody::Text(body_text.clone())),
                response_size: Some(body_text.len()),
                error: Some(format!("HTTP {}: {}", status.as_u16(), sanitized)),
                is_stream: false,
            });
            return Err(TransportError::HttpStatus {
                status: status.as_u16(),
                body: body_text,
                retry_after_ms,
                sanitized,
                headers: res_headers,
            });
        }

        // Success: parse JSON
        let text = resp
            .text()
            .await
            .map_err(|e| TransportError::BodyRead(e.to_string()))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|_| TransportError::BodyRead("invalid json".into()))?;
        emit_transport_event(TransportEvent {
            started_at,
            latency: Some(start_instant.elapsed()),
            method,
            url: request_url,
            status: Some(status.as_u16()),
            request_headers,
            response_headers: res_headers.clone(),
            request_body,
            response_body: Some(TransportBody::Json(json.clone())),
            response_size: Some(text.len()),
            error: None,
            is_stream: false,
        });
        Ok((json, res_headers))
    }

    async fn post_multipart(
        &self,
        url: &str,
        headers: &[(String, String)],
        form: &MultipartForm,
        cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        let mut req_form = Form::new();
        for field in &form.fields {
            match &field.value {
                MultipartValue::Text(text) => {
                    req_form = req_form.text(field.name.clone(), text.clone());
                }
                MultipartValue::Bytes {
                    data,
                    filename,
                    content_type,
                } => {
                    let mut part = Part::bytes(data.clone());
                    if let Some(name) = filename {
                        part = part.file_name(name.clone());
                    }
                    if let Some(ct) = content_type {
                        part = part
                            .mime_str(ct)
                            .map_err(|e| TransportError::Other(e.to_string()))?;
                    }
                    req_form = req_form.part(field.name.clone(), part);
                }
            }
        }

        let mut req = self.client.post(url).multipart(req_form);
        for (k, v) in headers {
            if !k.eq_ignore_ascii_case("content-type") {
                req = req.header(k, v);
            }
        }

        let started_at = SystemTime::now();
        let start_instant = Instant::now();
        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let (method, request_url, request_headers) = if let Some(req) = request_snapshot.as_ref() {
            (
                req.method().to_string(),
                req.url().to_string(),
                header_pairs(req.headers()),
            )
        } else {
            ("POST".to_string(), url.to_string(), headers.to_vec())
        };
        let request_body = None;

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method,
                    url: request_url,
                    status: None,
                    request_headers,
                    response_headers: Vec::new(),
                    request_body,
                    response_body: None,
                    response_size: None,
                    error: Some(detail.clone()),
                    is_stream: false,
                });
                return Err(if e.is_connect() {
                    TransportError::Network(format!("connect: {detail}"))
                } else if e.is_timeout() {
                    TransportError::ConnectTimeout(cfg.connect_timeout)
                } else {
                    TransportError::Network(detail)
                });
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let retry_after_ms = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(parse_retry_after_ms);
            let res_headers = resp
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
                .collect::<Vec<_>>();
            let body_text = resp.text().await.unwrap_or_default();
            let sanitized = display_body_for_error(&body_text);
            emit_transport_event(TransportEvent {
                started_at,
                latency: Some(start_instant.elapsed()),
                method,
                url: request_url,
                status: Some(status.as_u16()),
                request_headers,
                response_headers: res_headers.clone(),
                request_body,
                response_body: Some(TransportBody::Text(body_text.clone())),
                response_size: Some(body_text.len()),
                error: Some(format!("HTTP {}: {}", status.as_u16(), sanitized)),
                is_stream: false,
            });
            return Err(TransportError::HttpStatus {
                status: status.as_u16(),
                body: body_text,
                retry_after_ms,
                sanitized,
                headers: res_headers,
            });
        }

        let res_headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
            .collect::<Vec<_>>();
        let text = resp
            .text()
            .await
            .map_err(|e| TransportError::BodyRead(e.to_string()))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|_| TransportError::BodyRead("invalid json".into()))?;

        emit_transport_event(TransportEvent {
            started_at,
            latency: Some(start_instant.elapsed()),
            method,
            url: request_url,
            status: Some(status.as_u16()),
            request_headers,
            response_headers: res_headers.clone(),
            request_body,
            response_body: Some(TransportBody::Json(json.clone())),
            response_size: Some(text.len()),
            error: None,
            is_stream: false,
        });

        Ok((json, res_headers))
    }

    async fn get_bytes(
        &self,
        url: &str,
        headers: &[(String, String)],
        cfg: &TransportConfig,
    ) -> Result<(Bytes, Vec<(String, String)>), TransportError> {
        let mut req = self.client.get(url);
        for (k, v) in headers {
            req = req.header(k, v);
        }

        let started_at = SystemTime::now();
        let start_instant = Instant::now();
        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let (method, request_url, request_headers) = if let Some(req) = request_snapshot.as_ref() {
            (
                req.method().to_string(),
                req.url().to_string(),
                header_pairs(req.headers()),
            )
        } else {
            ("GET".to_string(), url.to_string(), headers.to_vec())
        };

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method,
                    url: request_url,
                    status: None,
                    request_headers,
                    response_headers: Vec::new(),
                    request_body: None,
                    response_body: None,
                    response_size: None,
                    error: Some(detail.clone()),
                    is_stream: false,
                });
                return Err(if e.is_connect() {
                    TransportError::Network(format!("connect: {detail}"))
                } else if e.is_timeout() {
                    TransportError::ConnectTimeout(cfg.connect_timeout)
                } else {
                    TransportError::Network(detail)
                });
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let retry_after_ms = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(parse_retry_after_ms);
            let res_headers = resp
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
                .collect::<Vec<_>>();
            let body_text = resp.text().await.unwrap_or_default();
            let sanitized = display_body_for_error(&body_text);
            emit_transport_event(TransportEvent {
                started_at,
                latency: Some(start_instant.elapsed()),
                method,
                url: request_url,
                status: Some(status.as_u16()),
                request_headers,
                response_headers: res_headers.clone(),
                request_body: None,
                response_body: Some(TransportBody::Text(body_text.clone())),
                response_size: Some(body_text.len()),
                error: Some(format!("HTTP {}: {}", status.as_u16(), sanitized)),
                is_stream: false,
            });
            return Err(TransportError::HttpStatus {
                status: status.as_u16(),
                body: body_text,
                retry_after_ms,
                sanitized,
                headers: res_headers,
            });
        }

        let res_headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
            .collect::<Vec<_>>();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| TransportError::BodyRead(e.to_string()))?;

        emit_transport_event(TransportEvent {
            started_at,
            latency: Some(start_instant.elapsed()),
            method,
            url: request_url,
            status: Some(status.as_u16()),
            request_headers,
            response_headers: res_headers.clone(),
            request_body: None,
            response_body: None,
            response_size: Some(bytes.len()),
            error: None,
            is_stream: false,
        });

        Ok((bytes, res_headers))
    }
}

fn should_skip_websocket_header(name: &str) -> bool {
    if name.eq_ignore_ascii_case("host")
        || name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("content-length")
        || name.eq_ignore_ascii_case("content-type")
    {
        return true;
    }
    name.to_ascii_lowercase().starts_with("sec-websocket-")
}

fn map_websocket_connect_error(
    err: WsError,
) -> (TransportError, Option<u16>, Vec<(String, String)>) {
    match err {
        WsError::Http(response) => {
            let status = response.status().as_u16();
            let headers = header_pairs(response.headers());
            let retry_after_ms = response
                .headers()
                .get(http::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(parse_retry_after_ms);
            (
                TransportError::HttpStatus {
                    status,
                    body: String::new(),
                    retry_after_ms,
                    sanitized: format!("http status {status}"),
                    headers: headers.clone(),
                },
                Some(status),
                headers,
            )
        }
        other => (
            TransportError::Network(format!("websocket connect failed: {other}")),
            None,
            Vec::new(),
        ),
    }
}

fn websocket_connect_error_headers(err: &TransportError) -> Vec<(String, String)> {
    match err {
        TransportError::HttpStatus { headers, .. } => headers.clone(),
        _ => Vec::new(),
    }
}

fn map_websocket_stream_error(err: WsError, idle_timeout: Duration) -> TransportError {
    match err {
        WsError::ConnectionClosed | WsError::AlreadyClosed => TransportError::StreamClosed,
        WsError::Io(io_err) if io_err.kind() == std::io::ErrorKind::TimedOut => {
            TransportError::IdleReadTimeout(idle_timeout)
        }
        WsError::Http(response) => {
            let status = response.status().as_u16();
            let headers = header_pairs(response.headers());
            let retry_after_ms = response
                .headers()
                .get(http::header::RETRY_AFTER)
                .and_then(|h| h.to_str().ok())
                .and_then(parse_retry_after_ms);
            TransportError::HttpStatus {
                status,
                body: String::new(),
                retry_after_ms,
                sanitized: format!("http status {status}"),
                headers,
            }
        }
        other => TransportError::Network(format!("websocket stream failed: {other}")),
    }
}

#[derive(Debug)]
struct WebsocketMessageOutcome {
    chunk: Option<Bytes>,
    terminal: bool,
}

fn websocket_text_to_sse_chunk(text: &str) -> Option<Bytes> {
    if text.trim().is_empty() {
        return None;
    }
    let payload = if looks_like_sse_payload(text) {
        ensure_sse_terminator(text)
    } else {
        sse_data_frame(text)
    };
    Some(Bytes::from(payload))
}

fn websocket_message_to_sse_chunk(
    message: WsMessage,
) -> Result<WebsocketMessageOutcome, TransportError> {
    match message {
        WsMessage::Text(text) => Ok(WebsocketMessageOutcome {
            chunk: websocket_text_to_sse_chunk(&text),
            terminal: websocket_text_has_terminal_event(&text),
        }),
        WsMessage::Binary(binary) => {
            if binary.is_empty() {
                return Ok(WebsocketMessageOutcome {
                    chunk: None,
                    terminal: false,
                });
            }
            let text = String::from_utf8_lossy(&binary);
            Ok(WebsocketMessageOutcome {
                chunk: websocket_text_to_sse_chunk(&text),
                terminal: websocket_text_has_terminal_event(&text),
            })
        }
        WsMessage::Ping(_) | WsMessage::Pong(_) => Ok(WebsocketMessageOutcome {
            chunk: None,
            terminal: false,
        }),
        WsMessage::Close(_) => Err(TransportError::StreamClosed),
        _ => Ok(WebsocketMessageOutcome {
            chunk: None,
            terminal: false,
        }),
    }
}

fn looks_like_sse_payload(text: &str) -> bool {
    text.lines().any(|line| {
        line.starts_with("data:")
            || line.starts_with("event:")
            || line.starts_with("id:")
            || line.starts_with("retry:")
    })
}

fn websocket_text_has_terminal_event(text: &str) -> bool {
    if looks_like_sse_payload(text) {
        return text
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
            .any(websocket_payload_has_terminal_event);
    }
    websocket_payload_has_terminal_event(text.trim())
}

fn websocket_payload_has_terminal_event(payload: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(payload) else {
        return false;
    };
    matches!(
        value.get("type").and_then(|value| value.as_str()),
        Some("response.completed" | "response.incomplete" | "response.failed" | "error")
    )
}

fn ensure_sse_terminator(payload: &str) -> String {
    if payload.ends_with("\n\n") || payload.ends_with("\r\n\r\n") {
        payload.to_string()
    } else if payload.ends_with('\n') {
        format!("{payload}\n")
    } else {
        format!("{payload}\n\n")
    }
}

fn sse_data_frame(payload: &str) -> String {
    let mut out = String::new();
    let mut wrote_line = false;
    for line in payload.lines() {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
        wrote_line = true;
    }
    if !wrote_line {
        out.push_str("data: ");
        out.push_str(payload);
        out.push('\n');
    }
    out.push('\n');
    out
}

fn header_pairs(headers: &http::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

fn resolve_proxy_for_websocket_url(url: &str) -> Result<Option<ResolvedProxy>, TransportError> {
    let parsed = Url::parse(url)
        .map_err(|err| TransportError::Other(format!("invalid websocket url '{url}': {err}")))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| TransportError::Other(format!("websocket url '{url}' is missing host")))?;
    if is_no_proxy_host(host) {
        return Ok(None);
    }
    let Some(proxy_value) = websocket_proxy_env_value(parsed.scheme()) else {
        return Ok(None);
    };
    let proxy = Url::parse(&proxy_value).map_err(|err| {
        TransportError::Other(format!("invalid proxy url '{proxy_value}': {err}"))
    })?;
    if !proxy.scheme().eq_ignore_ascii_case("http") {
        return Err(TransportError::Other(format!(
            "unsupported websocket proxy scheme '{}': only http:// proxies are supported",
            proxy.scheme()
        )));
    }
    let proxy_host = proxy.host_str().ok_or_else(|| {
        TransportError::Other(format!("proxy url '{proxy_value}' is missing host"))
    })?;
    let port = proxy.port_or_known_default().ok_or_else(|| {
        TransportError::Other(format!("proxy url '{proxy_value}' is missing port"))
    })?;
    let authorization = if proxy.username().is_empty() {
        None
    } else {
        let credentials = format!(
            "{}:{}",
            proxy.username(),
            proxy.password().unwrap_or_default()
        );
        Some(format!(
            "Basic {}",
            BASE64_STANDARD.encode(credentials.as_bytes())
        ))
    };
    Ok(Some(ResolvedProxy {
        host: proxy_host.to_string(),
        port,
        authorization,
    }))
}

fn websocket_proxy_env_value(scheme: &str) -> Option<String> {
    let keys = if scheme.eq_ignore_ascii_case("wss") {
        [
            "WSS_PROXY",
            "wss_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
        ]
    } else {
        [
            "WS_PROXY",
            "ws_proxy",
            "HTTP_PROXY",
            "http_proxy",
            "ALL_PROXY",
            "all_proxy",
            "",
            "",
        ]
    };
    keys.iter().filter(|key| !key.is_empty()).find_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn is_no_proxy_host(host: &str) -> bool {
    let Some(raw) = std::env::var("NO_PROXY")
        .ok()
        .or_else(|| std::env::var("no_proxy").ok())
    else {
        return false;
    };
    raw.split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .any(|entry| no_proxy_entry_matches(host, entry))
}

fn no_proxy_entry_matches(host: &str, entry: &str) -> bool {
    if entry == "*" {
        return true;
    }
    let entry = entry
        .strip_prefix('.')
        .unwrap_or(entry)
        .split(':')
        .next()
        .unwrap_or(entry);
    host.eq_ignore_ascii_case(entry)
        || host
            .strip_suffix(entry)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

async fn open_http_proxy_tunnel(
    uri: &http::Uri,
    proxy: &ResolvedProxy,
    cfg: &TransportConfig,
) -> Result<TcpStream, TransportError> {
    let target_host = uri
        .host()
        .ok_or_else(|| TransportError::Other(format!("websocket uri '{uri}' is missing host")))?;
    let target_port = uri.port_u16().unwrap_or_else(|| {
        if uri.scheme_str() == Some("wss") {
            443
        } else {
            80
        }
    });
    let mut socket = TcpStream::connect((proxy.host.as_str(), proxy.port))
        .await
        .map_err(|err| TransportError::Network(format!("proxy connect failed: {err}")))?;
    let authority = format!("{target_host}:{target_port}");
    let connect_request = build_http_proxy_connect_request(&authority, proxy);
    tokio::time::timeout(
        cfg.connect_timeout,
        socket.write_all(connect_request.as_bytes()),
    )
    .await
    .map_err(|_| TransportError::ConnectTimeout(cfg.connect_timeout))?
    .map_err(|err| TransportError::Network(format!("proxy connect write failed: {err}")))?;
    tokio::time::timeout(cfg.connect_timeout, socket.flush())
        .await
        .map_err(|_| TransportError::ConnectTimeout(cfg.connect_timeout))?
        .map_err(|err| TransportError::Network(format!("proxy connect flush failed: {err}")))?;

    let mut response = Vec::new();
    loop {
        let mut chunk = [0_u8; 1024];
        let bytes_read = tokio::time::timeout(cfg.connect_timeout, socket.read(&mut chunk))
            .await
            .map_err(|_| TransportError::ConnectTimeout(cfg.connect_timeout))?
            .map_err(|err| TransportError::Network(format!("proxy connect read failed: {err}")))?;
        if bytes_read == 0 {
            return Err(TransportError::Network(
                "proxy closed the CONNECT tunnel before sending a response".into(),
            ));
        }
        response.extend_from_slice(&chunk[..bytes_read]);
        if response.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if response.len() > 64 * 1024 {
            return Err(TransportError::Other(
                "proxy CONNECT response headers exceeded 64KiB".into(),
            ));
        }
    }

    validate_http_proxy_connect_response(&response)
        .map_err(|err| TransportError::Other(format!("proxy CONNECT failed: {err}")))?;
    Ok(socket)
}

fn build_http_proxy_connect_request(authority: &str, proxy: &ResolvedProxy) -> String {
    let mut request = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n");
    if let Some(authorization) = proxy.authorization.as_ref() {
        request.push_str("Proxy-Authorization: ");
        request.push_str(authorization);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    request
}

fn validate_http_proxy_connect_response(response: &[u8]) -> Result<(), String> {
    let text = String::from_utf8_lossy(response);
    let header_end = text.find("\r\n\r\n").unwrap_or(text.len());
    let header_text = &text[..header_end];
    let mut lines = header_text.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| "empty proxy response".to_string())?;
    let mut parts = status_line.split_whitespace();
    let _http_version = parts
        .next()
        .ok_or_else(|| format!("malformed proxy response status line: {status_line}"))?;
    let status = parts
        .next()
        .ok_or_else(|| format!("malformed proxy response status line: {status_line}"))?
        .parse::<u16>()
        .map_err(|_| format!("invalid proxy response status line: {status_line}"))?;
    if (200..300).contains(&status) {
        return Ok(());
    }
    Err(format!("http status {status}: {status_line}"))
}

fn parse_retry_after_ms(s: &str) -> Option<u64> {
    // RFC 7231: either delta-seconds or HTTP date; support simple delta only
    if let Ok(secs) = s.trim().parse::<u64>() {
        return Some(secs * 1000);
    }
    None
}

fn format_reqwest_error_chain(err: &reqwest::Error) -> String {
    let mut out = err.to_string();
    let mut current = err.source();
    while let Some(src) = current {
        out.push_str(": ");
        out.push_str(&src.to_string());
        current = src.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    #[test]
    fn try_new_returns_transport_error_when_client_build_fails() {
        let cfg = TransportConfig::default();
        let err = match ReqwestTransport::try_new_with_builder(
            &cfg,
            Client::builder().user_agent("bad\nagent"),
        ) {
            Ok(_) => panic!("invalid user-agent should fail reqwest client build"),
            Err(err) => err,
        };
        match err {
            TransportError::Other(message) => {
                assert!(
                    message.contains("reqwest client build failed"),
                    "unexpected message: {message}"
                );
            }
            other => panic!("unexpected transport error variant: {other:?}"),
        }
    }

    #[test]
    fn new_with_builder_does_not_panic_when_client_build_fails() {
        let cfg = TransportConfig::default();
        let _transport =
            ReqwestTransport::new_with_builder(&cfg, Client::builder().user_agent("bad\nagent"));
    }

    #[test]
    fn websocket_json_text_is_wrapped_as_sse_data_frame() {
        let payload = r#"{"type":"response.output_text.delta","delta":"hi"}"#;
        let chunk = websocket_text_to_sse_chunk(payload).expect("chunk");
        let text = String::from_utf8(chunk.to_vec()).expect("utf8");
        assert_eq!(text, format!("data: {payload}\n\n"));
    }

    #[test]
    fn websocket_sse_payload_keeps_event_shape() {
        let payload = "event: response\ndata: {\"type\":\"response.completed\"}";
        let chunk = websocket_text_to_sse_chunk(payload).expect("chunk");
        let text = String::from_utf8(chunk.to_vec()).expect("utf8");
        assert_eq!(
            text,
            "event: response\ndata: {\"type\":\"response.completed\"}\n\n"
        );
    }

    #[test]
    fn websocket_close_frame_maps_to_stream_closed_error() {
        let err = websocket_message_to_sse_chunk(WsMessage::Close(None))
            .expect_err("close frame should terminate the stream with an error");
        assert!(matches!(err, TransportError::StreamClosed));
    }

    #[test]
    fn websocket_header_filter_skips_upgrade_headers() {
        assert!(should_skip_websocket_header("connection"));
        assert!(should_skip_websocket_header("upgrade"));
        assert!(should_skip_websocket_header("sec-websocket-key"));
        assert!(should_skip_websocket_header("content-type"));
        assert!(!should_skip_websocket_header("authorization"));
    }

    #[test]
    fn websocket_proxy_resolution_prefers_secure_proxy_for_wss() {
        let _guard = env_test_lock();
        std::env::set_var("WSS_PROXY", "");
        std::env::set_var("HTTPS_PROXY", "http://user:pass@127.0.0.1:8080");
        std::env::remove_var("ALL_PROXY");
        std::env::remove_var("NO_PROXY");
        let proxy =
            resolve_proxy_for_websocket_url("wss://chatgpt.com/backend-api/codex/responses")
                .expect("proxy resolution")
                .expect("proxy configured");
        assert_eq!(proxy.host, "127.0.0.1");
        assert_eq!(proxy.port, 8080);
        assert_eq!(
            proxy.authorization,
            Some(format!(
                "Basic {}",
                BASE64_STANDARD.encode("user:pass".as_bytes())
            ))
        );
    }

    #[test]
    fn websocket_proxy_resolution_honors_no_proxy() {
        let _guard = env_test_lock();
        std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:8080");
        std::env::set_var("NO_PROXY", "chatgpt.com,.example.invalid");
        let proxy =
            resolve_proxy_for_websocket_url("wss://chatgpt.com/backend-api/codex/responses")
                .expect("proxy resolution");
        assert!(proxy.is_none());
    }

    #[test]
    fn builds_http_proxy_connect_request_with_basic_auth() {
        let request = build_http_proxy_connect_request(
            "chatgpt.com:443",
            &ResolvedProxy {
                host: "127.0.0.1".to_string(),
                port: 8080,
                authorization: Some("Basic dXNlcjpwYXNz".to_string()),
            },
        );
        assert!(request.starts_with("CONNECT chatgpt.com:443 HTTP/1.1\r\n"));
        assert!(request.contains("Host: chatgpt.com:443\r\n"));
        assert!(request.contains("Proxy-Authorization: Basic dXNlcjpwYXNz\r\n"));
        assert!(request.ends_with("\r\n\r\n"));
    }

    #[test]
    fn proxy_connect_response_accepts_success_status() {
        assert!(validate_http_proxy_connect_response(
            b"HTTP/1.1 200 Connection established\r\nProxy-Agent: mitmproxy\r\n\r\n"
        )
        .is_ok());
    }

    #[test]
    fn proxy_connect_response_rejects_non_success_status() {
        let err = validate_http_proxy_connect_response(
            b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n",
        )
        .expect_err("non-success proxy CONNECT should fail");
        assert!(err.contains("http status 407"));
    }
}
