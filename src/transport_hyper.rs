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
use http::header::{CONTENT_TYPE, RETRY_AFTER};
use http::{Method, Request, Uri};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Empty, Full};
use hyper::body::Incoming;
use hyper_rustls::{ConfigBuilderExt, HttpsConnector};
use hyper_util::client::legacy::connect::proxy::{SocksV4, SocksV5, Tunnel};
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::client::proxy::matcher::{Intercept, Matcher};
use hyper_util::rt::{TokioExecutor, TokioTimer};
use serde_json::Value;
use std::borrow::Cow;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{client_async_tls_with_config, connect_async};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use tower_service::Service;
use tracing::debug;
use uuid::Uuid;

type BoxError = Box<dyn StdError + Send + Sync>;
type RequestBody = BoxBody<Bytes, Infallible>;
type HyperConnector = HttpsConnector<ProxyAwareConnector>;
type HyperClient = Client<HyperConnector, RequestBody>;
type HyperIo = <HttpConnector as Service<Uri>>::Response;
type HyperWebsocketStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Clone)]
pub struct HyperTransport {
    client: HyperClient,
}

pub use HyperTransport as ReqwestTransport;

#[derive(Clone)]
struct ProxyAwareConnector {
    http: HttpConnector,
    matcher: Arc<Matcher>,
}

struct HyperJsonStreamWebsocketConnection {
    socket: Arc<Mutex<HyperWebsocketStream>>,
    response_headers: Vec<(String, String)>,
    closed: Arc<AtomicBool>,
}

impl ProxyAwareConnector {
    fn new(cfg: &TransportConfig) -> Self {
        let mut http = HttpConnector::new();
        http.enforce_http(false);
        http.set_keepalive(Some(Duration::from_secs(60)));
        http.set_keepalive_interval(Some(Duration::from_secs(30)));
        http.set_connect_timeout(Some(cfg.connect_timeout));
        http.set_nodelay(true);
        Self {
            http,
            matcher: Arc::new(Matcher::from_env()),
        }
    }
}

impl Service<Uri> for ProxyAwareConnector {
    type Response = HyperIo;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.http.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, dst: Uri) -> Self::Future {
        let intercept = self.matcher.intercept(&dst);
        let http = self.http.clone();
        Box::pin(async move {
            match intercept {
                Some(proxy) => connect_via_proxy(http, dst, proxy).await,
                None => {
                    let mut connector = http;
                    connector.call(dst).await.map_err(Into::into)
                }
            }
        })
    }
}

async fn connect_via_proxy(
    http: HttpConnector,
    dst: Uri,
    intercept: Intercept,
) -> Result<HyperIo, BoxError> {
    let proxy_uri = intercept.uri().clone();
    match proxy_uri.scheme_str().unwrap_or("http") {
        "http" => {
            let mut connector = Tunnel::new(proxy_uri, http);
            if let Some(auth) = intercept.basic_auth().cloned() {
                connector = connector.with_auth(auth);
            }
            connector
                .call(dst)
                .await
                .map_err(|err| Box::new(err) as BoxError)
        }
        "https" => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "https proxies are not supported by HyperTransport",
        ))),
        "socks4" => {
            let mut connector = SocksV4::new(proxy_uri, http).local_dns(true);
            connector
                .call(dst)
                .await
                .map_err(|err| Box::new(err) as BoxError)
        }
        "socks4a" => {
            let mut connector = SocksV4::new(proxy_uri, http).local_dns(false);
            connector
                .call(dst)
                .await
                .map_err(|err| Box::new(err) as BoxError)
        }
        "socks5" => {
            let mut connector = SocksV5::new(proxy_uri, http).local_dns(true);
            if let Some((user, pass)) = intercept.raw_auth() {
                connector = connector.with_auth(user.to_string(), pass.to_string());
            }
            connector
                .call(dst)
                .await
                .map_err(|err| Box::new(err) as BoxError)
        }
        "socks5h" => {
            let mut connector = SocksV5::new(proxy_uri, http).local_dns(false);
            if let Some((user, pass)) = intercept.raw_auth() {
                connector = connector.with_auth(user.to_string(), pass.to_string());
            }
            connector
                .call(dst)
                .await
                .map_err(|err| Box::new(err) as BoxError)
        }
        other => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("unsupported proxy scheme {other}"),
        ))),
    }
}

impl HyperTransport {
    fn json_request_body<'a>(body: &'a Value, cfg: &TransportConfig) -> Cow<'a, Value> {
        if cfg.strip_null_fields {
            Cow::Owned(crate::core::json::without_null_fields(body))
        } else {
            Cow::Borrowed(body)
        }
    }

    fn install_rustls_provider() {
        static INSTALLED: OnceLock<()> = OnceLock::new();
        INSTALLED.get_or_init(|| {
            let _ = rustls::crypto::ring::default_provider().install_default();
        });
    }

    fn build_client(cfg: &TransportConfig) -> Result<HyperClient, TransportError> {
        Self::install_rustls_provider();
        let tls = rustls::ClientConfig::builder()
            .with_native_roots()
            .map_err(|err| TransportError::Other(format!("failed to load native roots: {err}")))?
            .with_no_client_auth();
        let connector = HttpsConnector::from((ProxyAwareConnector::new(cfg), tls));
        let mut builder = Client::builder(TokioExecutor::new());
        builder.pool_idle_timeout(Duration::from_secs(90));
        builder.pool_timer(TokioTimer::new());
        Ok(builder.build(connector))
    }

    pub fn try_new(cfg: &TransportConfig) -> Result<Self, TransportError> {
        Ok(Self {
            client: Self::build_client(cfg)?,
        })
    }

    pub fn new(cfg: &TransportConfig) -> Self {
        match Self::try_new(cfg) {
            Ok(transport) => transport,
            Err(err) => {
                debug!(
                    target: "ai_sdk::transport::hyper",
                    error = %err,
                    "falling back to default hyper transport config after init failure"
                );
                Self::try_new(&TransportConfig::default()).unwrap_or_else(|fallback_err| {
                    panic!("hyper transport init failed: {fallback_err}")
                })
            }
        }
    }

    fn is_websocket_url(url: &str) -> bool {
        url.starts_with("ws://") || url.starts_with("wss://")
    }

    fn build_request(
        method: Method,
        url: &str,
        headers: &[(String, String)],
        body: RequestBody,
        content_type: Option<&str>,
        skip_content_type_header: bool,
        request_body: Option<TransportBody>,
        is_stream: bool,
    ) -> Result<(Request<RequestBody>, RequestContext), TransportError> {
        let uri = url
            .parse::<Uri>()
            .map_err(|err| TransportError::Other(format!("invalid request url '{url}': {err}")))?;
        let mut builder = Request::builder().method(method.clone()).uri(uri);
        if let Some(content_type) = content_type {
            builder = builder.header(CONTENT_TYPE, content_type);
        }
        for (name, value) in headers {
            if skip_content_type_header && name.eq_ignore_ascii_case("content-type") {
                continue;
            }
            builder = builder.header(name, value);
        }
        let request = builder.body(body).map_err(|err| {
            TransportError::Other(format!("failed to build hyper request: {err}"))
        })?;
        let context = RequestContext::new(
            method.as_str().to_string(),
            url.to_string(),
            header_pairs(request.headers()),
            request_body,
            is_stream,
        );
        Ok((request, context))
    }

    async fn send_request(
        &self,
        request: Request<RequestBody>,
        cfg: &TransportConfig,
    ) -> Result<hyper::Response<Incoming>, TransportError> {
        let response = if let Some(request_timeout) = cfg.request_timeout {
            tokio::time::timeout(request_timeout, self.client.request(request))
                .await
                .map_err(|_| TransportError::ConnectTimeout(cfg.connect_timeout))?
        } else {
            self.client.request(request).await
        };
        response.map_err(|err| map_hyper_request_error(err, cfg))
    }

    async fn collect_body_bytes(body: Incoming) -> Result<Bytes, TransportError> {
        body.collect()
            .await
            .map(|collected| collected.to_bytes())
            .map_err(|err| TransportError::BodyRead(format_error_chain(&err)))
    }

    async fn connect_websocket_stream(
        &self,
        request: http::Request<()>,
        cfg: &TransportConfig,
    ) -> Result<(HyperWebsocketStream, http::Response<Option<Vec<u8>>>), TransportError> {
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
    ) -> Result<HyperJsonStreamWebsocketConnection, TransportError> {
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

        Ok(HyperJsonStreamWebsocketConnection {
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

    fn build_multipart_body(form: &MultipartForm) -> Result<(Bytes, String), TransportError> {
        let boundary = format!("----ai-sdk-rs-{}", Uuid::new_v4().simple());
        let mut body = Vec::new();

        for field in &form.fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            match &field.value {
                MultipartValue::Text(text) => {
                    body.extend_from_slice(
                        format!(
                            "Content-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n",
                            field.name, text
                        )
                        .as_bytes(),
                    );
                }
                MultipartValue::Bytes {
                    data,
                    filename,
                    content_type,
                } => {
                    let mut disposition =
                        format!("Content-Disposition: form-data; name=\"{}\"", field.name);
                    if let Some(filename) = filename {
                        disposition.push_str(&format!("; filename=\"{filename}\""));
                    }
                    body.extend_from_slice(disposition.as_bytes());
                    body.extend_from_slice(b"\r\n");
                    if let Some(content_type) = content_type {
                        body.extend_from_slice(
                            format!("Content-Type: {content_type}\r\n").as_bytes(),
                        );
                    }
                    body.extend_from_slice(b"\r\n");
                    body.extend_from_slice(data);
                    body.extend_from_slice(b"\r\n");
                }
            }
        }

        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        Ok((
            Bytes::from(body),
            format!("multipart/form-data; boundary={boundary}"),
        ))
    }
}

impl Default for HyperTransport {
    fn default() -> Self {
        Self::new(&TransportConfig::default())
    }
}

#[async_trait]
impl JsonStreamWebsocketConnection for HyperJsonStreamWebsocketConnection {
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

        {
            let mut guard = socket.lock().await;
            if let Err(err) = guard.send(WsMessage::Text(payload.into())).await {
                closed.store(true, Ordering::SeqCst);
                let _ = tx.send(Err(TransportError::StreamClosed)).await;
                return Err(TransportError::Network(format!(
                    "websocket send failed: {err}"
                )));
            }
        }

        tokio::spawn(async move {
            let mut saw_terminal_event = false;
            loop {
                let next_item = {
                    let mut guard = socket.lock().await;
                    tokio::time::timeout(idle, guard.next()).await
                };

                let outcome = match next_item {
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
                            saw_terminal_event |= terminal;
                            if tx.send(Ok(chunk)).await.is_err() {
                                break;
                            }
                            if saw_terminal_event {
                                closed.store(true, Ordering::SeqCst);
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        closed.store(true, Ordering::SeqCst);
                        let terminal = matches!(err, TransportError::StreamClosed);
                        let _ = tx.send(Err(err)).await;
                        if terminal {
                            break;
                        }
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

#[async_trait]
impl HttpTransport for HyperTransport {
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

        let request_body = Some(TransportBody::Json(cleaned_body.as_ref().clone()));
        let body_bytes = serde_json::to_vec(cleaned_body.as_ref()).map_err(|err| {
            TransportError::Other(format!("failed to encode request body: {err}"))
        })?;
        let (request, context) = Self::build_request(
            Method::POST,
            url,
            headers,
            Full::new(Bytes::from(body_bytes)).boxed(),
            Some("application/json"),
            true,
            request_body,
            true,
        )?;

        let response = match self.send_request(request, cfg).await {
            Ok(response) => response,
            Err(err) => {
                let detail = err.to_string();
                debug!(target: "ai_sdk::transport::hyper", %detail, "hyper request failed");
                emit_send_error_event(&context, detail);
                return Err(err);
            }
        };

        let status = response.status();
        if !status.is_success() {
            let retry_after_ms = response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(parse_retry_after_ms);
            let response_headers = header_pairs(response.headers());
            let body_text =
                String::from_utf8_lossy(&Self::collect_body_bytes(response.into_body()).await?)
                    .to_string();
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                response_headers,
                body_text,
            ));
        }

        let response_headers = header_pairs(response.headers());
        emit_response_success_event(
            &context,
            status.as_u16(),
            response_headers.clone(),
            None,
            None,
        );

        let mut body = response.into_body().into_data_stream();
        let idle = cfg.idle_read_timeout;
        let stream = async_stream::try_stream! {
            loop {
                let next = tokio::time::timeout(idle, body.next()).await;
                match next {
                    Err(_) => Err(TransportError::IdleReadTimeout(idle))?,
                    Ok(None) => break,
                    Ok(Some(Err(err))) => Err(TransportError::BodyRead(format_error_chain(&err)))?,
                    Ok(Some(Ok(bytes))) => yield bytes,
                }
            }
        };
        Ok((Box::pin(stream), response_headers))
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
        let request_body = Some(TransportBody::Json(cleaned_body.as_ref().clone()));
        let body_bytes = serde_json::to_vec(cleaned_body.as_ref()).map_err(|err| {
            TransportError::Other(format!("failed to encode request body: {err}"))
        })?;
        let (request, context) = Self::build_request(
            Method::POST,
            url,
            headers,
            Full::new(Bytes::from(body_bytes)).boxed(),
            Some("application/json"),
            true,
            request_body,
            false,
        )?;

        let response = match self.send_request(request, cfg).await {
            Ok(response) => response,
            Err(err) => {
                let detail = err.to_string();
                debug!(target: "ai_sdk::transport::hyper", %detail, "hyper request failed");
                emit_send_error_event(&context, detail);
                return Err(err);
            }
        };

        let status = response.status();
        let response_headers = header_pairs(response.headers());
        let body_bytes = Self::collect_body_bytes(response.into_body()).await?;

        if !status.is_success() {
            let retry_after_ms = response_headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(RETRY_AFTER.as_str()))
                .and_then(|(_, value)| parse_retry_after_ms(value));
            let body_text = String::from_utf8_lossy(&body_bytes).to_string();
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                response_headers,
                body_text,
            ));
        }

        let text = String::from_utf8_lossy(&body_bytes).to_string();
        let json: Value = serde_json::from_str(&text)
            .map_err(|_| TransportError::BodyRead("invalid json".into()))?;
        emit_response_success_event(
            &context,
            status.as_u16(),
            response_headers.clone(),
            Some(TransportBody::Json(json.clone())),
            Some(text.len()),
        );
        Ok((json, response_headers))
    }

    async fn post_multipart(
        &self,
        url: &str,
        headers: &[(String, String)],
        form: &MultipartForm,
        cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        let (body_bytes, content_type) = Self::build_multipart_body(form)?;
        let (request, context) = Self::build_request(
            Method::POST,
            url,
            headers,
            Full::new(body_bytes).boxed(),
            Some(&content_type),
            true,
            None,
            false,
        )?;

        let response = match self.send_request(request, cfg).await {
            Ok(response) => response,
            Err(err) => {
                let detail = err.to_string();
                debug!(target: "ai_sdk::transport::hyper", %detail, "hyper request failed");
                emit_send_error_event(&context, detail);
                return Err(err);
            }
        };

        let status = response.status();
        let response_headers = header_pairs(response.headers());
        let body_bytes = Self::collect_body_bytes(response.into_body()).await?;

        if !status.is_success() {
            let retry_after_ms = response_headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(RETRY_AFTER.as_str()))
                .and_then(|(_, value)| parse_retry_after_ms(value));
            let body_text = String::from_utf8_lossy(&body_bytes).to_string();
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                response_headers,
                body_text,
            ));
        }

        let text = String::from_utf8_lossy(&body_bytes).to_string();
        let json: Value = serde_json::from_str(&text)
            .map_err(|_| TransportError::BodyRead("invalid json".into()))?;
        emit_response_success_event(
            &context,
            status.as_u16(),
            response_headers.clone(),
            Some(TransportBody::Json(json.clone())),
            Some(text.len()),
        );
        Ok((json, response_headers))
    }

    async fn get_bytes(
        &self,
        url: &str,
        headers: &[(String, String)],
        cfg: &TransportConfig,
    ) -> Result<(Bytes, Vec<(String, String)>), TransportError> {
        let (request, context) = Self::build_request(
            Method::GET,
            url,
            headers,
            Empty::<Bytes>::new().boxed(),
            None,
            false,
            None,
            false,
        )?;

        let response = match self.send_request(request, cfg).await {
            Ok(response) => response,
            Err(err) => {
                let detail = err.to_string();
                debug!(target: "ai_sdk::transport::hyper", %detail, "hyper request failed");
                emit_send_error_event(&context, detail);
                return Err(err);
            }
        };

        let status = response.status();
        let response_headers = header_pairs(response.headers());
        let body_bytes = Self::collect_body_bytes(response.into_body()).await?;

        if !status.is_success() {
            let retry_after_ms = response_headers
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(RETRY_AFTER.as_str()))
                .and_then(|(_, value)| parse_retry_after_ms(value));
            let body_text = String::from_utf8_lossy(&body_bytes).to_string();
            return Err(map_http_status_error(
                &context,
                status.as_u16(),
                retry_after_ms,
                response_headers,
                body_text,
            ));
        }

        emit_response_success_event(
            &context,
            status.as_u16(),
            response_headers.clone(),
            None,
            Some(body_bytes.len()),
        );
        Ok((body_bytes, response_headers))
    }
}

fn format_error_chain(err: &(dyn StdError + 'static)) -> String {
    let mut out = err.to_string();
    let mut current = err.source();
    while let Some(source) = current {
        out.push_str(": ");
        out.push_str(&source.to_string());
        current = source.source();
    }
    out
}

fn map_hyper_request_error(
    err: impl StdError + Send + Sync + 'static,
    cfg: &TransportConfig,
) -> TransportError {
    let detail = format_error_chain(&err);
    if detail.contains("deadline has elapsed") || detail.contains("timed out") {
        TransportError::ConnectTimeout(cfg.connect_timeout)
    } else {
        TransportError::Network(detail)
    }
}
