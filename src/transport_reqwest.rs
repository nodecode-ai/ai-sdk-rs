use crate::core::error::TransportError;
use crate::core::transport::{
    emit_transport_event, HttpTransport, JsonStreamWebsocketConnection, MultipartForm,
    MultipartValue, TransportBody, TransportConfig, TransportEvent, TransportStream,
};
use crate::transport_http_common::{
    emit_response_success_event, emit_send_error_event, header_pairs, map_http_status_error,
    parse_retry_after_ms, RequestContext,
};
use crate::transport_websocket_common::{
    map_websocket_connect_error, map_websocket_stream_error, open_http_proxy_tunnel,
    resolve_proxy_for_websocket_url, should_skip_websocket_header, websocket_connect_error_headers,
    websocket_message_to_sse_chunk, WebsocketMessageOutcome,
};
use async_trait::async_trait;
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
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{client_async_tls_with_config, connect_async};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tracing::debug;

#[derive(Clone)]
pub struct LegacyReqwestTransport {
    client: Client,
}

pub use crate::transport_hyper::HyperTransport as ReqwestTransport;

type ReqwestWebsocketStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

struct ReqwestJsonStreamWebsocketConnection {
    socket: Arc<Mutex<ReqwestWebsocketStream>>,
    response_headers: Vec<(String, String)>,
    closed: Arc<AtomicBool>,
}

impl LegacyReqwestTransport {
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

impl Default for LegacyReqwestTransport {
    fn default() -> Self {
        Self::new(&TransportConfig::default())
    }
}

fn request_context(
    request_snapshot: Option<&reqwest::Request>,
    fallback_method: &str,
    fallback_url: &str,
    fallback_headers: &[(String, String)],
    request_body: Option<TransportBody>,
    is_stream: bool,
) -> RequestContext {
    if let Some(request) = request_snapshot {
        RequestContext::new(
            request.method().to_string(),
            request.url().to_string(),
            header_pairs(request.headers()),
            request_body,
            is_stream,
        )
    } else {
        RequestContext::new(
            fallback_method.to_string(),
            fallback_url.to_string(),
            fallback_headers.to_vec(),
            request_body,
            is_stream,
        )
    }
}

#[async_trait]
impl HttpTransport for LegacyReqwestTransport {
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

        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let request_body = Some(TransportBody::Json(cleaned_body.as_ref().clone()));
        let context = request_context(
            request_snapshot.as_ref(),
            "POST",
            url,
            headers,
            request_body,
            true,
        );

        // Send
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_send_error_event(&context, detail.clone());
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
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                res_headers,
                body_text,
            ));
        }

        // Success: stream the bytes with idle timeout enforcement
        let idle = cfg.idle_read_timeout;
        // Collect response headers for return
        let res_headers = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| v.to_str().ok().map(|s| (k.to_string(), s.to_string())))
            .collect::<Vec<_>>();

        emit_response_success_event(&context, status.as_u16(), res_headers.clone(), None, None);

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

        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let request_body = Some(TransportBody::Json(cleaned_body.as_ref().clone()));
        let context = request_context(
            request_snapshot.as_ref(),
            "POST",
            url,
            headers,
            request_body,
            false,
        );

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_send_error_event(&context, detail.clone());
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
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                res_headers,
                body_text,
            ));
        }

        // Success: parse JSON
        let text = resp
            .text()
            .await
            .map_err(|e| TransportError::BodyRead(e.to_string()))?;
        let json: Value = serde_json::from_str(&text)
            .map_err(|_| TransportError::BodyRead("invalid json".into()))?;
        emit_response_success_event(
            &context,
            status.as_u16(),
            res_headers.clone(),
            Some(TransportBody::Json(json.clone())),
            Some(text.len()),
        );
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

        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let context = request_context(request_snapshot.as_ref(), "POST", url, headers, None, false);

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_send_error_event(&context, detail.clone());
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
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                res_headers,
                body_text,
            ));
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

        emit_response_success_event(
            &context,
            status.as_u16(),
            res_headers.clone(),
            Some(TransportBody::Json(json.clone())),
            Some(text.len()),
        );

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

        let request_snapshot = req.try_clone().and_then(|r| r.build().ok());
        let context = request_context(request_snapshot.as_ref(), "GET", url, headers, None, false);

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let detail = format_reqwest_error_chain(&e);
                debug!(target: "ai_sdk::transport::reqwest", %detail, "reqwest send failed");
                emit_send_error_event(&context, detail.clone());
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
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                res_headers,
                body_text,
            ));
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

        emit_response_success_event(
            &context,
            status.as_u16(),
            res_headers.clone(),
            None,
            Some(bytes.len()),
        );

        Ok((bytes, res_headers))
    }
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
        let err = match LegacyReqwestTransport::try_new_with_builder(
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
        let _transport = LegacyReqwestTransport::new_with_builder(
            &cfg,
            Client::builder().user_agent("bad\nagent"),
        );
    }
}
