use crate::ai_sdk_core::error::{display_body_for_error, TransportError};
use crate::ai_sdk_core::transport::{
    emit_transport_event, HttpTransport, MultipartForm, MultipartValue, TransportBody,
    TransportConfig, TransportEvent,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::{SinkExt, StreamExt};
use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde_json::Value;
use std::error::Error as StdError;
use std::pin::Pin;
use std::time::{Duration, Instant, SystemTime};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error as WsError, Message as WsMessage};
use tracing::debug;

pub struct ReqwestTransport {
    client: Client,
}

impl ReqwestTransport {
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

    async fn post_json_stream_websocket(
        &self,
        url: &str,
        headers: &[(String, String)],
        cleaned_body: &Value,
        cfg: &TransportConfig,
    ) -> Result<
        (
            Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
            Vec<(String, String)>,
        ),
        TransportError,
    > {
        let started_at = SystemTime::now();
        let start_instant = Instant::now();
        let request_headers = headers.to_vec();
        let request_body = Some(TransportBody::Json(cleaned_body.clone()));

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

        let connect_result =
            tokio::time::timeout(cfg.connect_timeout, connect_async(request)).await;
        let (mut socket, response) = match connect_result {
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
                let (mapped, status, response_headers) = map_websocket_connect_error(err);
                emit_transport_event(TransportEvent {
                    started_at,
                    latency: Some(start_instant.elapsed()),
                    method: "GET".to_string(),
                    url: url.to_string(),
                    status,
                    request_headers,
                    response_headers,
                    request_body,
                    response_body: None,
                    response_size: None,
                    error: Some(mapped.to_string()),
                    is_stream: true,
                });
                return Err(mapped);
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

        let payload = serde_json::to_string(cleaned_body).map_err(|err| {
            TransportError::Other(format!("failed to encode websocket payload: {err}"))
        })?;
        socket
            .send(WsMessage::Text(payload))
            .await
            .map_err(|err| map_websocket_stream_error(err, cfg.idle_read_timeout))?;

        let idle = cfg.idle_read_timeout;
        let s = async_stream::try_stream! {
            loop {
                let next = tokio::time::timeout(idle, socket.next()).await;
                match next {
                    Err(_) => Err(TransportError::IdleReadTimeout(idle))?,
                    Ok(None) => break,
                    Ok(Some(Err(err))) => Err(map_websocket_stream_error(err, idle))?,
                    Ok(Some(Ok(message))) => {
                        match message {
                            WsMessage::Text(text) => {
                                if let Some(chunk) = websocket_text_to_sse_chunk(&text) {
                                    yield chunk;
                                }
                            }
                            WsMessage::Binary(binary) => {
                                if binary.is_empty() {
                                    continue;
                                }
                                let text = String::from_utf8_lossy(&binary);
                                if let Some(chunk) = websocket_text_to_sse_chunk(&text) {
                                    yield chunk;
                                }
                            }
                            WsMessage::Ping(_) | WsMessage::Pong(_) => {}
                            WsMessage::Close(_) => break,
                            _ => {}
                        }
                    }
                }
            }
        };

        Ok((Box::pin(s), response_headers))
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new(&TransportConfig::default())
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    type StreamResponse = (
        Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
        Vec<(String, String)>,
    );

    fn into_stream(
        resp: Self::StreamResponse,
    ) -> (
        Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
        Vec<(String, String)>,
    ) {
        resp
    }

    async fn post_json_stream(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        // Clean body by stripping null fields if configured
        let cleaned_body: Value = if cfg.strip_null_fields {
            crate::ai_sdk_core::json::without_null_fields(body)
        } else {
            body.clone()
        };

        if Self::is_websocket_url(url) {
            return self
                .post_json_stream_websocket(url, headers, &cleaned_body, cfg)
                .await;
        }

        // Build request
        let mut req = self.client.post(url).json(&cleaned_body);
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
        let request_body = Some(TransportBody::Json(cleaned_body.clone()));

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

    async fn post_json(
        &self,
        url: &str,
        headers: &[(String, String)],
        body: &Value,
        cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        // Clean body by stripping null fields if configured
        let cleaned_body: Value = if cfg.strip_null_fields {
            crate::ai_sdk_core::json::without_null_fields(body)
        } else {
            body.clone()
        };
        let mut req = self.client.post(url).json(&cleaned_body);
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
        let request_body = Some(TransportBody::Json(cleaned_body.clone()));

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

fn looks_like_sse_payload(text: &str) -> bool {
    text.lines().any(|line| {
        line.starts_with("data:")
            || line.starts_with("event:")
            || line.starts_with("id:")
            || line.starts_with("retry:")
    })
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
    fn websocket_header_filter_skips_upgrade_headers() {
        assert!(should_skip_websocket_header("connection"));
        assert!(should_skip_websocket_header("upgrade"));
        assert!(should_skip_websocket_header("sec-websocket-key"));
        assert!(should_skip_websocket_header("content-type"));
        assert!(!should_skip_websocket_header("authorization"));
    }
}
