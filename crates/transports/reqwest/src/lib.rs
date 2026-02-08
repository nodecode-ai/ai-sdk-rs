use crate::ai_sdk_core::error::{display_body_for_error, TransportError};
use crate::ai_sdk_core::transport::{
    emit_transport_event, HttpTransport, MultipartForm, MultipartValue, TransportBody,
    TransportConfig, TransportEvent,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::multipart::{Form, Part};
use reqwest::Client;
use serde_json::Value;
use std::error::Error as StdError;
use std::pin::Pin;
use std::time::{Duration, Instant, SystemTime};
use tracing::debug;

pub struct ReqwestTransport {
    client: Client,
}

impl ReqwestTransport {
    pub fn new(cfg: &TransportConfig) -> Self {
        let mut builder = Client::builder()
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .pool_idle_timeout(Duration::from_secs(90))
            // reqwest 0.12 no longer exposes `http2_keep_alive_interval`; use
            // the cross-protocol TCP keepalive interval instead.
            .tcp_keepalive_interval(Duration::from_secs(30));
        if let Some(req_timeout) = cfg.request_timeout {
            builder = builder.timeout(req_timeout);
        }
        // connect timeout
        builder = builder.connect_timeout(cfg.connect_timeout);
        Self {
            client: builder.build().expect("reqwest client build"),
        }
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

fn header_pairs(headers: &reqwest::header::HeaderMap) -> Vec<(String, String)> {
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
