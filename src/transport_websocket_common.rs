use crate::core::error::TransportError;
use crate::core::transport::TransportConfig;
use crate::transport_http_common::{header_pairs, parse_retry_after_ms};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use bytes::Bytes;
use serde_json::Value;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::{Error as WsError, Message as WsMessage};
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedProxy {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) authorization: Option<String>,
}

#[derive(Debug)]
pub(crate) struct WebsocketMessageOutcome {
    pub(crate) chunk: Option<Bytes>,
    pub(crate) terminal: bool,
}

pub(crate) fn should_skip_websocket_header(name: &str) -> bool {
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

pub(crate) fn map_websocket_connect_error(
    err: WsError,
) -> (TransportError, Option<u16>, Vec<(String, String)>) {
    match err {
        WsError::Http(response) => {
            let status = response.status().as_u16();
            let headers = header_pairs(response.headers());
            let retry_after_ms = response
                .headers()
                .get(http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
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

pub(crate) fn websocket_connect_error_headers(err: &TransportError) -> Vec<(String, String)> {
    match err {
        TransportError::HttpStatus { headers, .. } => headers.clone(),
        _ => Vec::new(),
    }
}

pub(crate) fn map_websocket_stream_error(err: WsError, idle_timeout: Duration) -> TransportError {
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
                .and_then(|value| value.to_str().ok())
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

pub(crate) fn websocket_text_to_sse_chunk(text: &str) -> Option<Bytes> {
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

pub(crate) fn websocket_message_to_sse_chunk(
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

pub(crate) fn resolve_proxy_for_websocket_url(
    url: &str,
) -> Result<Option<ResolvedProxy>, TransportError> {
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

pub(crate) async fn open_http_proxy_tunnel(
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

pub(crate) fn build_http_proxy_connect_request(authority: &str, proxy: &ResolvedProxy) -> String {
    let mut request = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n");
    if let Some(authorization) = proxy.authorization.as_ref() {
        request.push_str("Proxy-Authorization: ");
        request.push_str(authorization);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    request
}

pub(crate) fn validate_http_proxy_connect_response(response: &[u8]) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
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
