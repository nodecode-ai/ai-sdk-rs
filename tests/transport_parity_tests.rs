use ::ai_sdk_rs::ai_sdk_core::error::TransportError;
use ::ai_sdk_rs::ai_sdk_core::transport::{
    set_transport_observer, HttpTransport, MultipartForm, TransportBody, TransportConfig,
    TransportEvent, TransportObserver,
};
use ::ai_sdk_rs::transport_hyper::HyperTransport;
use ::ai_sdk_rs::transport_reqwest::LegacyReqwestTransport;
use bytes::Bytes;
use futures_util::TryStreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<TransportEvent>>,
}

impl RecordingObserver {
    fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    fn last_event(&self) -> TransportEvent {
        self.events
            .lock()
            .unwrap()
            .last()
            .cloned()
            .expect("transport event")
    }
}

impl TransportObserver for RecordingObserver {
    fn on_event(&self, event: TransportEvent) {
        self.events.lock().unwrap().push(event);
    }
}

fn transport_observer() -> Arc<RecordingObserver> {
    static OBSERVER: OnceLock<Arc<RecordingObserver>> = OnceLock::new();
    OBSERVER
        .get_or_init(|| {
            let observer = Arc::new(RecordingObserver::default());
            assert!(
                set_transport_observer(observer.clone()),
                "transport observer was already installed before transport parity tests ran",
            );
            observer
        })
        .clone()
}

#[derive(Clone, Debug)]
struct RecordedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[derive(Clone)]
enum ResponseBody {
    Full(Vec<u8>),
    Chunked(Vec<Vec<u8>>),
}

#[derive(Clone)]
struct ResponseSpec {
    status: u16,
    headers: Vec<(String, String)>,
    body: ResponseBody,
}

impl ResponseSpec {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), "application/json".into())],
            body: ResponseBody::Full(value.to_string().into_bytes()),
        }
    }

    fn bytes(status: u16, bytes: Vec<u8>, content_type: &str) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), content_type.into())],
            body: ResponseBody::Full(bytes),
        }
    }

    fn chunked(status: u16, chunks: Vec<Vec<u8>>, content_type: &str) -> Self {
        Self {
            status,
            headers: vec![("content-type".into(), content_type.into())],
            body: ResponseBody::Chunked(chunks),
        }
    }

    fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

struct TestServer {
    base_url: String,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    task: JoinHandle<()>,
}

impl TestServer {
    async fn spawn(response: ResponseSpec) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind transport parity listener");
        let addr = listener.local_addr().expect("listener addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured_requests = Arc::clone(&requests);
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept request");
            let request = read_request(&mut stream).await;
            captured_requests.lock().unwrap().push(request);
            write_response(&mut stream, response).await;
        });
        Self {
            base_url: format!("http://{addr}"),
            requests,
            task,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    async fn finish(self) -> Vec<RecordedRequest> {
        let Self { requests, task, .. } = self;
        task.await.expect("server task");
        let recorded_requests = requests.lock().unwrap().clone();
        recorded_requests
    }
}

fn lower_header_map(headers: &[(String, String)]) -> HashMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect()
}

fn concat_bytes(chunks: &[Bytes]) -> Vec<u8> {
    let total = chunks.iter().map(Bytes::len).sum();
    let mut out = Vec::with_capacity(total);
    for chunk in chunks {
        out.extend_from_slice(chunk);
    }
    out
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn test_transport_config() -> TransportConfig {
    TransportConfig {
        request_timeout: Some(Duration::from_secs(2)),
        connect_timeout: Duration::from_secs(2),
        idle_read_timeout: Duration::from_secs(2),
        strip_null_fields: true,
    }
}

fn reqwest_transport(cfg: &TransportConfig) -> LegacyReqwestTransport {
    LegacyReqwestTransport::try_new(cfg).expect("build reqwest transport")
}

fn hyper_transport(cfg: &TransportConfig) -> HyperTransport {
    HyperTransport::try_new(cfg).expect("build hyper transport")
}

async fn assert_post_json_stream_contract<T: HttpTransport>(transport: &T, cfg: &TransportConfig) {
    let observer = transport_observer();
    observer.clear();
    let response_chunks = vec![
        b"data: {\"type\":\"response.started\"}\n\n".to_vec(),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let server = TestServer::spawn(
        ResponseSpec::chunked(200, response_chunks.clone(), "text/event-stream")
            .with_header("x-response-id", "stream-1"),
    )
    .await;
    let body = json!({
        "message": "hello",
        "drop": null,
    });
    let response = transport
        .post_json_stream(
            &server.url("/stream"),
            &[
                ("x-test".into(), "stream-contract".into()),
                ("content-type".into(), "application/json".into()),
            ],
            &body,
            cfg,
        )
        .await
        .expect("stream response");
    let (stream, response_headers) = T::into_stream(response);
    let chunks: Vec<Bytes> = stream.try_collect().await.expect("stream body");
    let requests = server.finish().await;
    let request = requests.first().expect("captured request");
    let request_headers = lower_header_map(&request.headers);
    let response_headers = lower_header_map(&response_headers);
    let event = observer.last_event();

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/stream");
    assert_eq!(
        response_headers.get("x-response-id"),
        Some(&"stream-1".to_string())
    );
    assert_eq!(
        response_headers.get("content-type"),
        Some(&"text/event-stream".to_string())
    );
    assert_eq!(
        request_headers.get("x-test"),
        Some(&"stream-contract".to_string())
    );
    assert!(
        request_headers
            .get("content-type")
            .is_some_and(|value| value.starts_with("application/json")),
        "expected application/json content-type, got {:?}",
        request_headers.get("content-type"),
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&request.body).expect("request json"),
        json!({ "message": "hello" })
    );
    assert_eq!(
        concat_bytes(&chunks),
        concat_bytes(
            &response_chunks
                .into_iter()
                .map(Bytes::from)
                .collect::<Vec<_>>()
        )
    );
    assert!(event.is_stream);
    assert_eq!(event.status, Some(200));
    assert_eq!(event.error, None);
    match event.request_body {
        Some(TransportBody::Json(sent_body)) => {
            assert_eq!(sent_body, json!({ "message": "hello" }));
        }
        other => panic!("unexpected stream request body: {other:?}"),
    }
}

async fn assert_post_json_contract<T: HttpTransport>(transport: &T, cfg: &TransportConfig) {
    let observer = transport_observer();
    observer.clear();
    let response_json = json!({
        "ok": true,
        "mode": "json",
    });
    let server = TestServer::spawn(
        ResponseSpec::json(200, response_json.clone()).with_header("x-response-id", "json-1"),
    )
    .await;
    let body = json!({
        "message": "hello",
        "drop": null,
    });
    let (json_body, response_headers) = transport
        .post_json(
            &server.url("/json"),
            &[("x-test".into(), "json-contract".into())],
            &body,
            cfg,
        )
        .await
        .expect("json response");
    let requests = server.finish().await;
    let request = requests.first().expect("captured request");
    let request_headers = lower_header_map(&request.headers);
    let response_headers = lower_header_map(&response_headers);
    let event = observer.last_event();

    assert_eq!(json_body, response_json);
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/json");
    assert_eq!(
        serde_json::from_slice::<Value>(&request.body).expect("request json"),
        json!({ "message": "hello" })
    );
    assert_eq!(
        request_headers.get("x-test"),
        Some(&"json-contract".to_string())
    );
    assert_eq!(
        response_headers.get("x-response-id"),
        Some(&"json-1".to_string())
    );
    assert!(!event.is_stream);
    assert_eq!(event.status, Some(200));
    assert_eq!(event.error, None);
    match event.request_body {
        Some(TransportBody::Json(sent_body)) => {
            assert_eq!(sent_body, json!({ "message": "hello" }));
        }
        other => panic!("unexpected json request body: {other:?}"),
    }
    match event.response_body {
        Some(TransportBody::Json(response_body)) => {
            assert_eq!(response_body, response_json);
        }
        other => panic!("unexpected json response body: {other:?}"),
    }
}

async fn assert_retry_after_contract<T: HttpTransport>(transport: &T, cfg: &TransportConfig) {
    let observer = transport_observer();
    observer.clear();
    let server = TestServer::spawn(
        ResponseSpec::json(429, json!({ "error": "slow down" }))
            .with_header("retry-after", "3")
            .with_header("x-response-id", "json-rate-limit"),
    )
    .await;
    let err = transport
        .post_json(
            &server.url("/json-rate-limit"),
            &[("x-test".into(), "retry-after-contract".into())],
            &json!({ "message": "hello" }),
            cfg,
        )
        .await
        .expect_err("rate-limited response");
    let requests = server.finish().await;
    let request = requests.first().expect("captured request");
    let event = observer.last_event();

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/json-rate-limit");
    match err {
        TransportError::HttpStatus {
            status,
            retry_after_ms,
            body,
            headers,
            ..
        } => {
            let headers = lower_header_map(&headers);
            assert_eq!(status, 429);
            assert_eq!(retry_after_ms, Some(3000));
            assert_eq!(body, r#"{"error":"slow down"}"#);
            assert_eq!(
                headers.get("x-response-id"),
                Some(&"json-rate-limit".to_string())
            );
        }
        other => panic!("unexpected transport error: {other:?}"),
    }
    assert!(!event.is_stream);
    assert_eq!(event.status, Some(429));
    assert!(
        event
            .error
            .as_deref()
            .is_some_and(|message| message.contains("HTTP 429")),
        "expected HTTP 429 event error, got {:?}",
        event.error,
    );
}

async fn assert_post_multipart_contract<T: HttpTransport>(transport: &T, cfg: &TransportConfig) {
    let observer = transport_observer();
    observer.clear();
    let response_json = json!({ "uploaded": true });
    let server = TestServer::spawn(
        ResponseSpec::json(200, response_json.clone()).with_header("x-response-id", "upload-1"),
    )
    .await;
    let mut form = MultipartForm::new();
    form.push_text("purpose", "transport-parity");
    form.push_bytes(
        "file",
        b"fixture-file-contents".to_vec(),
        Some("fixture.txt".into()),
        Some("text/plain".into()),
    );
    let (json_body, response_headers) = transport
        .post_multipart(
            &server.url("/multipart"),
            &[("x-test".into(), "multipart-contract".into())],
            &form,
            cfg,
        )
        .await
        .expect("multipart response");
    let requests = server.finish().await;
    let request = requests.first().expect("captured request");
    let request_headers = lower_header_map(&request.headers);
    let response_headers = lower_header_map(&response_headers);
    let request_body = String::from_utf8_lossy(&request.body);
    let event = observer.last_event();

    assert_eq!(json_body, response_json);
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/multipart");
    assert_eq!(
        request_headers.get("x-test"),
        Some(&"multipart-contract".to_string())
    );
    assert!(
        request_headers
            .get("content-type")
            .is_some_and(|value| value.starts_with("multipart/form-data; boundary=")),
        "expected multipart/form-data content-type, got {:?}",
        request_headers.get("content-type"),
    );
    assert!(request_body.contains("name=\"purpose\""));
    assert!(request_body.contains("transport-parity"));
    assert!(request_body.contains("name=\"file\""));
    assert!(request_body.contains("filename=\"fixture.txt\""));
    assert!(request_body.contains("Content-Type: text/plain"));
    assert!(request_body.contains("fixture-file-contents"));
    assert_eq!(
        response_headers.get("x-response-id"),
        Some(&"upload-1".to_string())
    );
    assert!(!event.is_stream);
    assert_eq!(event.status, Some(200));
    assert!(event.request_body.is_none());
}

async fn assert_get_bytes_contract<T: HttpTransport>(transport: &T, cfg: &TransportConfig) {
    let observer = transport_observer();
    observer.clear();
    let expected = b"transport-byte-fixture".to_vec();
    let server = TestServer::spawn(
        ResponseSpec::bytes(200, expected.clone(), "application/octet-stream")
            .with_header("x-response-id", "bytes-1"),
    )
    .await;
    let (bytes, response_headers) = transport
        .get_bytes(
            &server.url("/bytes"),
            &[("x-test".into(), "bytes-contract".into())],
            cfg,
        )
        .await
        .expect("bytes response");
    let requests = server.finish().await;
    let request = requests.first().expect("captured request");
    let response_headers = lower_header_map(&response_headers);
    let event = observer.last_event();

    assert_eq!(bytes, Bytes::from(expected.clone()));
    assert_eq!(request.method, "GET");
    assert_eq!(request.path, "/bytes");
    assert!(request.body.is_empty());
    assert_eq!(
        response_headers.get("x-response-id"),
        Some(&"bytes-1".to_string())
    );
    assert!(!event.is_stream);
    assert_eq!(event.status, Some(200));
    assert!(event.request_body.is_none());
    assert!(event.response_body.is_none());
    assert_eq!(event.response_size, Some(expected.len()));
}

async fn read_request(stream: &mut TcpStream) -> RecordedRequest {
    let mut buffer = Vec::new();
    let header_end = loop {
        if let Some(position) = find_bytes(&buffer, b"\r\n\r\n") {
            break position + 4;
        }
        read_more(stream, &mut buffer).await;
    };
    let header_text = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().expect("request line");
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().expect("request method").to_string();
    let path = request_parts.next().expect("request path").to_string();
    let headers = lines
        .take_while(|line| !line.is_empty())
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_string(), value.trim().to_string()))
        .collect::<Vec<_>>();
    let header_map = lower_header_map(&headers);
    let mut body_buffer = buffer[header_end..].to_vec();
    let body = if header_map
        .get("transfer-encoding")
        .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
    {
        read_chunked_body(stream, &mut body_buffer).await
    } else if let Some(content_length) = header_map
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
    {
        read_exact_body(stream, &mut body_buffer, content_length).await
    } else {
        Vec::new()
    };
    RecordedRequest {
        method,
        path,
        headers,
        body,
    }
}

async fn write_response(stream: &mut TcpStream, response: ResponseSpec) {
    let mut head = format!(
        "HTTP/1.1 {} {}\r\n",
        response.status,
        reason_phrase(response.status)
    );
    let mut has_connection = false;
    match response.body {
        ResponseBody::Full(body) => {
            for (name, value) in &response.headers {
                if name.eq_ignore_ascii_case("connection") {
                    has_connection = true;
                }
                head.push_str(name);
                head.push_str(": ");
                head.push_str(value);
                head.push_str("\r\n");
            }
            if !has_connection {
                head.push_str("Connection: close\r\n");
            }
            head.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
            stream
                .write_all(head.as_bytes())
                .await
                .expect("write headers");
            stream.write_all(&body).await.expect("write body");
        }
        ResponseBody::Chunked(chunks) => {
            for (name, value) in &response.headers {
                if name.eq_ignore_ascii_case("connection") {
                    has_connection = true;
                }
                head.push_str(name);
                head.push_str(": ");
                head.push_str(value);
                head.push_str("\r\n");
            }
            if !has_connection {
                head.push_str("Connection: close\r\n");
            }
            head.push_str("Transfer-Encoding: chunked\r\n\r\n");
            stream
                .write_all(head.as_bytes())
                .await
                .expect("write headers");
            for chunk in chunks {
                let prefix = format!("{:X}\r\n", chunk.len());
                stream
                    .write_all(prefix.as_bytes())
                    .await
                    .expect("write chunk size");
                stream.write_all(&chunk).await.expect("write chunk");
                stream.write_all(b"\r\n").await.expect("write chunk end");
                stream.flush().await.expect("flush chunk");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            stream
                .write_all(b"0\r\n\r\n")
                .await
                .expect("write final chunk");
        }
    }
    stream.flush().await.expect("flush response");
}

async fn read_chunked_body(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Vec<u8> {
    let mut body = Vec::new();
    loop {
        let size_line = read_line(stream, buffer).await;
        let size =
            usize::from_str_radix(size_line.split(';').next().unwrap_or_default().trim(), 16)
                .expect("chunk size");
        if size == 0 {
            loop {
                if read_line(stream, buffer).await.is_empty() {
                    return body;
                }
            }
        }
        ensure_buffer_len(stream, buffer, size + 2).await;
        body.extend_from_slice(&buffer[..size]);
        buffer.drain(..size + 2);
    }
}

async fn read_exact_body(
    stream: &mut TcpStream,
    buffer: &mut Vec<u8>,
    content_length: usize,
) -> Vec<u8> {
    ensure_buffer_len(stream, buffer, content_length).await;
    buffer[..content_length].to_vec()
}

async fn read_line(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> String {
    loop {
        if let Some(position) = find_bytes(buffer, b"\r\n") {
            let line = String::from_utf8_lossy(&buffer[..position]).to_string();
            buffer.drain(..position + 2);
            return line;
        }
        read_more(stream, buffer).await;
    }
}

async fn ensure_buffer_len(stream: &mut TcpStream, buffer: &mut Vec<u8>, wanted: usize) {
    while buffer.len() < wanted {
        read_more(stream, buffer).await;
    }
}

async fn read_more(stream: &mut TcpStream, buffer: &mut Vec<u8>) {
    let mut chunk = [0_u8; 4096];
    let read = stream.read(&mut chunk).await.expect("read socket");
    assert!(read > 0, "unexpected socket close");
    buffer.extend_from_slice(&chunk[..read]);
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[tokio::test(flavor = "current_thread")]
async fn reqwest_transport_locks_post_json_stream_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = reqwest_transport(&cfg);
    assert_post_json_stream_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reqwest_transport_locks_post_json_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = reqwest_transport(&cfg);
    assert_post_json_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reqwest_transport_preserves_retry_after_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = reqwest_transport(&cfg);
    assert_retry_after_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reqwest_transport_locks_multipart_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = reqwest_transport(&cfg);
    assert_post_multipart_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn reqwest_transport_locks_get_bytes_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = reqwest_transport(&cfg);
    assert_get_bytes_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn hyper_transport_locks_post_json_stream_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = hyper_transport(&cfg);
    assert_post_json_stream_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn hyper_transport_locks_post_json_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = hyper_transport(&cfg);
    assert_post_json_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn hyper_transport_preserves_retry_after_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = hyper_transport(&cfg);
    assert_retry_after_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn hyper_transport_locks_multipart_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = hyper_transport(&cfg);
    assert_post_multipart_contract(&transport, &cfg).await;
}

#[tokio::test(flavor = "current_thread")]
async fn hyper_transport_locks_get_bytes_contract() {
    let _guard = test_lock();
    let cfg = test_transport_config();
    let transport = hyper_transport(&cfg);
    assert_get_bytes_contract(&transport, &cfg).await;
}
