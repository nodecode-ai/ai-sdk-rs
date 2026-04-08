use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_core::transport::{
    HttpTransport, JsonStreamWebsocketConnection, TransportConfig,
};
use crate::ai_sdk_core::{
    map_events_to_parts, GenerateResponse, LanguageModel, LanguageModelTurnSession, StreamResponse,
};
use crate::ai_sdk_streaming_sse::{PipelineBuilder, ProviderChunk, SseEvent};
use crate::ai_sdk_types::v2 as v2t;
use crate::ai_sdk_types::{Event, TokenUsage};
use futures_core::Stream;
use futures_util::{stream, StreamExt};
use serde_json::{json, Map, Value};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio::time::{sleep_until, Duration, Instant};
use url::Url;
use uuid::Uuid;

use super::provider_tools::{
    build_tool_name_mapping, provider_tool_data_from_output_item, provider_tool_parts_from_data,
    ProviderToolParts, ToolNameMapping,
};
use super::request_translation::{
    build_request_body, parse_openai_provider_options, OpenAIProviderOptionsParsed,
};
use crate::provider_openai::config::OpenAIConfig;
use crate::provider_openai::error::map_transport_error;

type EventStream = Pin<Box<dyn Stream<Item = Result<Event, SdkError>> + Send>>;
type ByteStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, SdkError>> + Send>>;
type RawByteStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, TransportError>> + Send>>;
type SessionWebsocketRequest = (Value, Value, ByteStream, Vec<(String, String)>, bool, bool);

const OPENAI_WS_BETA_HEADER: &str = "OpenAI-Beta";
const OPENAI_WS_BETA_VALUE: &str = "responses_websockets=2026-02-06";
const CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
const TRANSPORT_HEADER_EFFECTIVE: &str = "x-ai-sdk-effective-transport";
const TRANSPORT_HEADER_REQUESTED: &str = "x-ai-sdk-requested-transport";
const TRANSPORT_HEADER_FALLBACK: &str = "x-ai-sdk-transport-fallback";
const PROVIDER_SESSION_CONNECTIONS_HEADER: &str = "x-ai-sdk-provider-session-connections";
const PROVIDER_SESSION_PREWARMED_HEADER: &str = "x-ai-sdk-provider-session-prewarmed";
const PROVIDER_SESSION_REQUEST_COUNT_HEADER: &str = "x-ai-sdk-provider-session-request-count";
const PROVIDER_SESSION_REUSED_HEADER: &str = "x-ai-sdk-provider-session-reused";
const PROVIDER_SESSION_RESET_REASON_HEADER: &str = "x-ai-sdk-provider-session-reset-reason";
const PROVIDER_PREVIOUS_RESPONSE_ID_USED_HEADER: &str =
    "x-ai-sdk-provider-previous-response-id-used";
const PROVIDER_WARMUP_RESPONSE_ID_USED_HEADER: &str =
    "x-ai-sdk-provider-session-warmup-response-id-used";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ResponseTransportMode {
    Http,
    Websocket,
}

impl ResponseTransportMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Websocket => "websocket",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResponseTransportSelection {
    requested: ResponseTransportMode,
    fallback_http: bool,
}

#[derive(Debug, Default)]
struct OpenAIResponsesTurnSessionState {
    turn_state: Option<String>,
    last_request: Option<Value>,
    last_response_id: Option<String>,
    last_response_id_from_warmup: bool,
    warmup_completed: bool,
    connection_count: usize,
    request_count: usize,
    last_reset_reason: Option<String>,
    force_http: bool,
}

#[derive(Debug, Clone)]
struct OpenAIResponsesTurnSessionTelemetry {
    connections: usize,
    prewarmed: bool,
    request_count: usize,
    reused: bool,
    previous_response_id_used: bool,
    warmup_response_id_used: bool,
    reset_reason: Option<String>,
}

#[derive(Default)]
struct WebsocketPreconnectState {
    headers: Vec<(String, String)>,
    ready: Option<Box<dyn JsonStreamWebsocketConnection>>,
    pending: Option<JoinHandle<Result<Box<dyn JsonStreamWebsocketConnection>, SdkError>>>,
}

struct OpenAIResponsesTurnSession<'a, T: HttpTransport> {
    model: &'a OpenAIResponsesLanguageModel<T>,
    websocket: Option<Box<dyn JsonStreamWebsocketConnection>>,
    state: Arc<Mutex<OpenAIResponsesTurnSessionState>>,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct DefaultClock;

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct InMemoryState;

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct NotKeyed;

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Quota {
    interval: Duration,
}

impl Quota {
    #[doc(hidden)]
    pub fn per_second(rps: NonZeroU32) -> Self {
        Self {
            interval: Duration::from_secs_f64(1.0 / f64::from(rps.get())),
        }
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct RateLimiter<K = NotKeyed, S = InMemoryState, C = DefaultClock> {
    interval: Duration,
    next_ready: AsyncMutex<Instant>,
    marker: PhantomData<(K, S, C)>,
}

impl RateLimiter<NotKeyed, InMemoryState, DefaultClock> {
    #[doc(hidden)]
    pub fn direct(quota: Quota) -> Self {
        Self {
            interval: quota.interval,
            next_ready: AsyncMutex::new(Instant::now()),
            marker: PhantomData,
        }
    }
}

impl<K, S, C> RateLimiter<K, S, C> {
    #[doc(hidden)]
    pub async fn until_ready(&self) {
        let scheduled = {
            let mut next_ready = self.next_ready.lock().await;
            reserve_rate_limiter_slot(&mut next_ready, Instant::now(), self.interval)
        };
        sleep_until(scheduled).await;
    }
}

fn reserve_rate_limiter_slot(
    next_ready: &mut Instant,
    now: Instant,
    interval: Duration,
) -> Instant {
    let scheduled = if *next_ready > now { *next_ready } else { now };
    *next_ready = scheduled + interval;
    scheduled
}

pub struct OpenAIResponsesLanguageModel<
    T: HttpTransport = crate::reqwest_transport::ReqwestTransport,
> {
    pub model_id: String,
    pub config: OpenAIConfig,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    websocket_preconnect: Arc<Mutex<WebsocketPreconnectState>>,
}

impl Default for OpenAIResponsesLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    fn default() -> Self {
        let cfg = TransportConfig::default();
        Self {
            model_id: String::new(),
            config: OpenAIConfig {
                provider_name: "openai.responses".into(),
                provider_scope_name: "openai".into(),
                base_url: "https://api.openai.com/v1".into(),
                endpoint_path: "/responses".into(),
                headers: vec![],
                query_params: vec![],
                supported_urls: HashMap::new(),
                file_id_prefixes: Some(vec!["file-".into()]),
                default_options: None,
                request_defaults: None,
            },
            http: crate::reqwest_transport::ReqwestTransport::new(&cfg),
            transport_cfg: cfg,
            limiter: None,
            websocket_preconnect: Arc::new(Mutex::new(WebsocketPreconnectState::default())),
        }
    }
}

impl<T: HttpTransport> OpenAIResponsesLanguageModel<T> {
    pub fn new(
        model_id: impl Into<String>,
        config: OpenAIConfig,
        http: T,
        transport_cfg: TransportConfig,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            config,
            http,
            transport_cfg,
            limiter: None,
            websocket_preconnect: Arc::new(Mutex::new(WebsocketPreconnectState::default())),
        }
    }

    pub fn with_rate_limit_per_sec(mut self, rps: u32) -> Self {
        if let Some(nz) = NonZeroU32::new(rps) {
            let q = Quota::per_second(nz);
            self.limiter = Some(Arc::new(RateLimiter::direct(q)));
        }
        self
    }

    fn endpoint_url(&self) -> String {
        self.config.endpoint_url()
    }

    fn compact_endpoint_url(&self) -> String {
        let endpoint = self.endpoint_url();
        if let Ok(mut url) = Url::parse(&endpoint) {
            let path = format!("{}/compact", url.path().trim_end_matches('/'));
            url.set_path(&path);
            return url.to_string();
        }
        format!("{}/compact", endpoint.trim_end_matches('/'))
    }

    async fn take_preconnected_websocket(&self) -> Option<Box<dyn JsonStreamWebsocketConnection>> {
        let pending = {
            let mut state = self.websocket_preconnect.lock().unwrap();
            if let Some(connection) = state.ready.take() {
                state.headers.clear();
                return Some(connection);
            }
            state.pending.take()
        };
        let Some(handle) = pending else {
            return None;
        };
        match handle.await {
            Ok(Ok(connection)) => {
                self.websocket_preconnect.lock().unwrap().headers.clear();
                Some(connection)
            }
            Ok(Err(err)) => {
                self.websocket_preconnect.lock().unwrap().headers.clear();
                tracing::debug!(error = %err, "openai websocket preconnect failed");
                None
            }
            Err(err) => {
                self.websocket_preconnect.lock().unwrap().headers.clear();
                tracing::debug!(error = %err, "openai websocket preconnect task cancelled");
                None
            }
        }
    }

    fn preconnected_websocket_headers_match(&self, headers: &[(String, String)]) -> Option<bool> {
        let state = self.websocket_preconnect.lock().unwrap();
        if state.ready.is_none() && state.pending.is_none() {
            return None;
        }
        Some(state.headers == headers)
    }

    fn discard_preconnected_websocket(&self) {
        let mut state = self.websocket_preconnect.lock().unwrap();
        if let Some(handle) = state.pending.take() {
            handle.abort();
        }
        state.ready = None;
        state.headers.clear();
    }

    fn canonicalize_header(lc: &str) -> String {
        if lc.eq_ignore_ascii_case("openai-beta") {
            return OPENAI_WS_BETA_HEADER.to_string();
        }
        lc.split('-')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    None => String::new(),
                    Some(f) => {
                        f.to_ascii_uppercase().to_string() + &chars.as_str().to_ascii_lowercase()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    fn request_headers(&self, extra: &HashMap<String, String>) -> BTreeMap<String, String> {
        let mut hdrs: BTreeMap<String, String> = BTreeMap::new();
        hdrs.insert("content-type".into(), "application/json".into());
        hdrs.insert("accept".into(), "application/json".into());
        for (k, v) in &self.config.headers {
            if crate::ai_sdk_core::options::is_internal_sdk_header(k) {
                continue;
            }
            hdrs.insert(k.to_lowercase(), v.clone());
        }
        for (k, v) in extra {
            if crate::ai_sdk_core::options::is_internal_sdk_header(k) {
                continue;
            }
            hdrs.insert(k.to_lowercase(), v.clone());
        }
        hdrs
    }

    pub async fn compact_history_json(&self, options: v2t::CallOptions) -> Result<Value, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        let (body, _warnings) = build_request_body(&options, &self.model_id, &self.config)?;
        let compact_body = build_compaction_request_body(body)?;
        let url = self.compact_endpoint_url();
        let hdrs = self.request_headers(&options.headers);
        let headers: Vec<(String, String)> = hdrs
            .into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect();
        let (json, _res_headers) = self
            .http
            .post_json(&url, &headers, &compact_body, &self.transport_cfg)
            .await
            .map_err(map_transport_error)?;

        if let Some(error) = json.get("error").filter(|v| !v.is_null()) {
            let message = error
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| error.to_string());
            return Err(SdkError::Upstream {
                status: 400,
                message,
                source: None,
            });
        }

        Ok(json)
    }

    async fn post_response_json(
        &self,
        body: &Value,
        extra_headers: &HashMap<String, String>,
    ) -> Result<Value, SdkError> {
        let url = self.endpoint_url();
        let headers: Vec<(String, String)> = self
            .request_headers(extra_headers)
            .into_iter()
            .map(|(key, value)| (Self::canonicalize_header(&key), value))
            .collect();
        let (json, _response_headers) = self
            .http
            .post_json(&url, &headers, body, &self.transport_cfg)
            .await
            .map_err(map_transport_error)?;
        Ok(json)
    }

    async fn send(
        &self,
        body: serde_json::Value,
        transport: ResponseTransportSelection,
        extra_headers: &HashMap<String, String>,
    ) -> Result<(ByteStream, v2t::Headers), SdkError> {
        let requested = transport.requested;
        let hdrs = self.request_headers(extra_headers);
        if let Some(l) = &self.limiter {
            let _ = l.until_ready().await;
        }

        match self.send_once(&hdrs, &body, requested).await {
            Ok((stream, res_headers)) => {
                self.finish_send(
                    stream,
                    res_headers,
                    requested,
                    transport.fallback_http,
                    &hdrs,
                    &body,
                )
                .await
            }
            Err(err) => {
                self.handle_send_error(err, requested, transport.fallback_http, &hdrs, &body)
                    .await
            }
        }
    }

    async fn finish_send(
        &self,
        stream: RawByteStream,
        res_headers: Vec<(String, String)>,
        requested: ResponseTransportMode,
        fallback_http: bool,
        headers: &BTreeMap<String, String>,
        body: &serde_json::Value,
    ) -> Result<(ByteStream, v2t::Headers), SdkError> {
        if requested != ResponseTransportMode::Websocket {
            return Ok(map_raw_transport_response(
                stream,
                res_headers,
                requested,
                requested,
                None,
            ));
        }

        match prefetch_stream(stream).await {
            Ok(stream) => Ok((
                stream,
                response_headers_with_transport(res_headers, requested, requested, None),
            )),
            Err(err) if fallback_http && should_fallback_to_http_after_websocket_error(&err) => {
                self.send_http_fallback(headers, body, requested).await
            }
            Err(err) => Err(err),
        }
    }

    async fn handle_send_error(
        &self,
        err: SdkError,
        requested: ResponseTransportMode,
        fallback_http: bool,
        headers: &BTreeMap<String, String>,
        body: &serde_json::Value,
    ) -> Result<(ByteStream, v2t::Headers), SdkError> {
        if requested == ResponseTransportMode::Websocket
            && fallback_http
            && should_fallback_to_http_after_websocket_error(&err)
        {
            return self.send_http_fallback(headers, body, requested).await;
        }

        Err(err)
    }

    async fn send_http_fallback(
        &self,
        headers: &BTreeMap<String, String>,
        body: &serde_json::Value,
        requested: ResponseTransportMode,
    ) -> Result<(ByteStream, v2t::Headers), SdkError> {
        let (stream, res_headers) = self
            .send_once(headers, body, ResponseTransportMode::Http)
            .await?;
        Ok(map_raw_transport_response(
            stream,
            res_headers,
            requested,
            ResponseTransportMode::Http,
            Some(requested),
        ))
    }

    async fn send_once(
        &self,
        base_headers: &BTreeMap<String, String>,
        body: &serde_json::Value,
        transport: ResponseTransportMode,
    ) -> Result<(RawByteStream, Vec<(String, String)>), SdkError> {
        let mut url = self.endpoint_url();
        if transport == ResponseTransportMode::Websocket {
            url = to_websocket_url(&url)?;
        }

        let mut hdrs = base_headers.clone();
        if transport == ResponseTransportMode::Websocket {
            hdrs.entry("openai-beta".into())
                .or_insert_with(|| OPENAI_WS_BETA_VALUE.to_string());
        }
        let headers: Vec<(String, String)> = hdrs
            .into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect();
        let request_body = transport_request_body(body, transport, &self.config.endpoint_path);

        match self
            .http
            .post_json_stream(&url, &headers, &request_body, &self.transport_cfg)
            .await
        {
            Ok(resp) => {
                let (stream, res_headers) = <T as HttpTransport>::into_stream(resp);
                Ok((stream, res_headers))
            }
            Err(te) => Err(map_transport_error(te)),
        }
    }
}

fn map_raw_transport_response(
    stream: RawByteStream,
    headers: Vec<(String, String)>,
    requested: ResponseTransportMode,
    effective: ResponseTransportMode,
    fallback_from: Option<ResponseTransportMode>,
) -> (ByteStream, v2t::Headers) {
    (
        map_transport_stream(stream),
        response_headers_with_transport(headers, requested, effective, fallback_from),
    )
}

#[cfg(test)]
mod tests {
    use super::{reserve_rate_limiter_slot, OpenAIResponsesLanguageModel};
    use tokio::time::{Duration, Instant};

    #[test]
    fn rate_limiter_slot_is_immediate_when_idle() {
        let base = Instant::now();
        let interval = Duration::from_millis(100);
        let mut next_ready = base;

        let scheduled = reserve_rate_limiter_slot(&mut next_ready, base, interval);

        assert_eq!(scheduled, base);
        assert_eq!(next_ready, base + interval);
    }

    #[test]
    fn rate_limiter_slot_queues_when_previous_slot_is_reserved() {
        let base = Instant::now();
        let interval = Duration::from_millis(100);
        let mut next_ready = base + interval;

        let scheduled =
            reserve_rate_limiter_slot(&mut next_ready, base + Duration::from_millis(40), interval);

        assert_eq!(scheduled, base + interval);
        assert_eq!(next_ready, base + Duration::from_millis(200));
    }

    #[test]
    fn zero_rps_keeps_rate_limiter_disabled() {
        let model =
            OpenAIResponsesLanguageModel::<crate::reqwest_transport::ReqwestTransport>::default()
                .with_rate_limit_per_sec(0);

        assert!(model.limiter.is_none());
    }
}

impl OpenAIResponsesLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn start_codex_websocket_preconnect(&mut self) {
        if !should_use_codex_oauth_websocket_transport(&self.config.endpoint_path) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let url = match to_websocket_url(&self.endpoint_url()) {
            Ok(url) => url,
            Err(err) => {
                tracing::debug!(error = %err, "openai websocket preconnect url resolution failed");
                return;
            }
        };
        let headers = {
            let mut hdrs = self.request_headers(&HashMap::new());
            hdrs.entry("openai-beta".into())
                .or_insert_with(|| OPENAI_WS_BETA_VALUE.to_string());
            hdrs.into_iter()
                .map(|(k, v)| (Self::canonicalize_header(&k), v))
                .collect::<Vec<_>>()
        };
        let http = self.http.clone();
        let transport_cfg = self.transport_cfg.clone();
        let task_headers = headers.clone();
        let task = runtime.spawn(async move {
            http.connect_json_stream_websocket(&url, &task_headers, &transport_cfg)
                .await
                .map_err(map_transport_error)
        });
        let mut state = self.websocket_preconnect.lock().unwrap();
        if state.ready.is_none() && state.pending.is_none() {
            state.headers = headers;
            state.pending = Some(task);
        }
    }
}

impl<'a, T: HttpTransport + Send + Sync + 'static> OpenAIResponsesTurnSession<'a, T> {
    fn new(model: &'a OpenAIResponsesLanguageModel<T>) -> Self {
        Self {
            model,
            websocket: None,
            state: Arc::new(Mutex::new(OpenAIResponsesTurnSessionState::default())),
        }
    }

    fn clear_incremental_state(&self, reason: &str) {
        let mut state = self.state.lock().unwrap();
        state.turn_state = None;
        state.last_request = None;
        state.last_response_id = None;
        state.last_response_id_from_warmup = false;
        state.warmup_completed = false;
        state.request_count = 0;
        state.last_reset_reason = Some(reason.to_string());
    }

    fn activate_http_fallback(&mut self, reason: &str) {
        self.websocket = None;
        let mut state = self.state.lock().unwrap();
        state.turn_state = None;
        state.last_request = None;
        state.last_response_id = None;
        state.last_response_id_from_warmup = false;
        state.warmup_completed = false;
        state.request_count = 0;
        state.force_http = true;
        state.last_reset_reason = Some(reason.to_string());
    }

    fn websocket_headers(&self, extra_headers: &HashMap<String, String>) -> Vec<(String, String)> {
        let turn_state = self.state.lock().unwrap().turn_state.clone();
        let mut merged = extra_headers.clone();
        if let Some(turn_state) = turn_state {
            merged
                .entry(CODEX_TURN_STATE_HEADER.to_string())
                .or_insert(turn_state);
        }
        let mut hdrs = self.model.request_headers(&merged);
        hdrs.entry("openai-beta".into())
            .or_insert_with(|| OPENAI_WS_BETA_VALUE.to_string());
        hdrs.into_iter()
            .map(|(k, v)| {
                (
                    OpenAIResponsesLanguageModel::<T>::canonicalize_header(&k),
                    v,
                )
            })
            .collect()
    }

    async fn ensure_websocket_connection(
        &mut self,
        headers: &[(String, String)],
    ) -> Result<bool, SdkError> {
        let needs_new = self
            .websocket
            .as_ref()
            .map(|connection| connection.is_closed())
            .unwrap_or(true);
        if !needs_new {
            return Ok(true);
        }

        if self.websocket.is_some() {
            self.clear_incremental_state("websocket_reconnect");
            self.websocket = None;
        }

        match self.model.preconnected_websocket_headers_match(headers) {
            Some(true) => {
                if let Some(connection) = self.model.take_preconnected_websocket().await {
                    {
                        let mut state = self.state.lock().unwrap();
                        state.connection_count = state.connection_count.saturating_add(1);
                    }
                    self.websocket = Some(connection);
                    return Ok(false);
                }
            }
            Some(false) => self.model.discard_preconnected_websocket(),
            None => {}
        }

        let url = to_websocket_url(&self.model.endpoint_url())?;
        let connection = self
            .model
            .http
            .connect_json_stream_websocket(&url, headers, &self.model.transport_cfg)
            .await
            .map_err(map_transport_error)?;
        {
            let mut state = self.state.lock().unwrap();
            state.connection_count = state.connection_count.saturating_add(1);
        }
        self.websocket = Some(connection);
        Ok(false)
    }

    fn should_prewarm_websocket(&self, body: &Value) -> bool {
        let state = self.state.lock().unwrap();
        !state.warmup_completed
            && state.request_count == 0
            && state.last_response_id.is_none()
            && body.get("previous_response_id").is_none()
    }

    fn warmup_websocket_body(&self, body: &Value) -> (Value, Value) {
        let mut warmup_body = body.clone();
        if let Some(object) = warmup_body.as_object_mut() {
            object.insert("generate".to_string(), Value::Bool(false));
            object.insert("input".to_string(), Value::Array(Vec::new()));
            object.remove("previous_response_id");
        }
        let transport_body = transport_request_body(
            &warmup_body,
            ResponseTransportMode::Websocket,
            &self.model.config.endpoint_path,
        );
        (warmup_body, transport_body)
    }

    async fn send_websocket_request(
        &mut self,
        transport_body: &Value,
    ) -> Result<(ByteStream, Vec<(String, String)>), SdkError> {
        let transport_headers = self
            .websocket
            .as_ref()
            .map(|connection| connection.response_headers())
            .unwrap_or_default();
        let raw_stream = match self
            .websocket
            .as_ref()
            .ok_or_else(|| SdkError::Transport(TransportError::StreamClosed))?
            .send_json_stream(transport_body, &self.model.transport_cfg)
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                self.clear_incremental_state("websocket_reconnect");
                self.websocket = None;
                return Err(map_transport_error(err));
            }
        };
        let stream = match prefetch_stream(raw_stream).await {
            Ok(stream) => stream,
            Err(err) => {
                self.clear_incremental_state("websocket_reconnect");
                self.websocket = None;
                return Err(err);
            }
        };
        let mut state = self.state.lock().unwrap();
        state.request_count = state.request_count.saturating_add(1);
        Ok((stream, transport_headers))
    }

    async fn prewarm_websocket_session(&mut self, body: &Value) -> Result<(), SdkError> {
        let (warmup_body, warmup_transport_body) = self.warmup_websocket_body(body);
        let (stream, _) = self.send_websocket_request(&warmup_transport_body).await?;
        let pipeline = PipelineBuilder::<OpenAIResponsesChunk>::new()
            .with_provider("openai_official")
            .include_raw(false)
            .build(stream);
        let mut parts = map_events_to_parts(
            Box::pin(pipeline),
            super::stream_hooks::build_stream_mapper_config(
                Vec::new(),
                build_tool_name_mapping(&[]),
                HashMap::new(),
                false,
                false,
            ),
        );
        let mut completed_response_id = None;
        while let Some(item) = parts.next().await {
            match item {
                Ok(v2t::StreamPart::ResponseMetadata { meta }) => {
                    completed_response_id = meta.id.clone();
                }
                Ok(v2t::StreamPart::Finish {
                    provider_metadata, ..
                }) => {
                    completed_response_id =
                        provider_response_id_from_metadata(provider_metadata.as_ref())
                            .or(completed_response_id);
                }
                Ok(_) => {}
                Err(err) => return Err(err),
            }
        }
        let Some(response_id) = completed_response_id else {
            return Err(SdkError::Transport(TransportError::Other(
                "codex websocket warmup completed without a response id".into(),
            )));
        };
        let mut state = self.state.lock().unwrap();
        state.last_request = Some(warmup_body);
        state.last_response_id = Some(response_id);
        state.last_response_id_from_warmup = true;
        state.warmup_completed = true;
        Ok(())
    }

    async fn send_session_websocket_request(
        &mut self,
        body: Value,
    ) -> Result<(Value, Value, ByteStream, Vec<(String, String)>, bool, bool), SdkError> {
        let (session_body, transport_body, previous_response_id_used, warmup_response_id_used) =
            self.prepared_websocket_body(body);
        let (stream, transport_headers) = self.send_websocket_request(&transport_body).await?;
        Ok((
            session_body,
            transport_body,
            stream,
            transport_headers,
            previous_response_id_used,
            warmup_response_id_used,
        ))
    }

    fn prepared_websocket_body(&self, mut body: Value) -> (Value, Value, bool, bool) {
        let mut previous_response_id_used = false;
        let mut warmup_response_id_used = false;
        let explicit_previous_response_id = body
            .get("previous_response_id")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string);
        if let Some(object) = body.as_object_mut() {
            object.remove("previous_response_id");
        }
        let (last_response_id, last_request, last_response_id_from_warmup, reset_reason) = {
            let state = self.state.lock().unwrap();
            (
                state.last_response_id.clone(),
                state.last_request.clone(),
                state.last_response_id_from_warmup,
                state.last_reset_reason.clone(),
            )
        };
        if let Some(explicit_previous_response_id) = explicit_previous_response_id {
            if reset_reason.is_some() {
                tracing::info!(
                    reason = ?reset_reason,
                    "preserving explicit previous_response_id across provider-session reset"
                );
            }
            if let Some(object) = body.as_object_mut() {
                object.insert(
                    "previous_response_id".to_string(),
                    Value::String(explicit_previous_response_id),
                );
                previous_response_id_used = true;
            }
        } else if let Some(last_response_id) = last_response_id {
            if request_shape_matches_previous(last_request.as_ref(), &body) {
                if let Some(object) = body.as_object_mut() {
                    object.insert(
                        "previous_response_id".to_string(),
                        Value::String(last_response_id),
                    );
                    previous_response_id_used = true;
                    warmup_response_id_used = last_response_id_from_warmup;
                }
            }
        }
        let transport_body = transport_request_body(
            &body,
            ResponseTransportMode::Websocket,
            &self.model.config.endpoint_path,
        );
        (
            body,
            transport_body,
            previous_response_id_used,
            warmup_response_id_used,
        )
    }

    fn response_headers(
        &self,
        transport_headers: Vec<(String, String)>,
        requested: ResponseTransportMode,
        effective: ResponseTransportMode,
        fallback_from: Option<ResponseTransportMode>,
        reused: bool,
        previous_response_id_used: bool,
        warmup_response_id_used: bool,
    ) -> v2t::Headers {
        let mut headers =
            response_headers_with_transport(transport_headers, requested, effective, fallback_from);
        let telemetry = {
            let state = self.state.lock().unwrap();
            if let Some(turn_state) = state.turn_state.as_ref() {
                headers
                    .entry(CODEX_TURN_STATE_HEADER.to_string())
                    .or_insert_with(|| turn_state.clone());
            }
            OpenAIResponsesTurnSessionTelemetry {
                connections: state.connection_count,
                prewarmed: state.warmup_completed,
                request_count: state.request_count,
                reused,
                previous_response_id_used,
                warmup_response_id_used,
                reset_reason: state.last_reset_reason.clone(),
            }
        };
        headers.insert(
            PROVIDER_SESSION_CONNECTIONS_HEADER.to_string(),
            telemetry.connections.to_string(),
        );
        headers.insert(
            PROVIDER_SESSION_PREWARMED_HEADER.to_string(),
            telemetry.prewarmed.to_string(),
        );
        headers.insert(
            PROVIDER_SESSION_REQUEST_COUNT_HEADER.to_string(),
            telemetry.request_count.to_string(),
        );
        headers.insert(
            PROVIDER_SESSION_REUSED_HEADER.to_string(),
            telemetry.reused.to_string(),
        );
        headers.insert(
            PROVIDER_PREVIOUS_RESPONSE_ID_USED_HEADER.to_string(),
            telemetry.previous_response_id_used.to_string(),
        );
        headers.insert(
            PROVIDER_WARMUP_RESPONSE_ID_USED_HEADER.to_string(),
            telemetry.warmup_response_id_used.to_string(),
        );
        if let Some(reason) = telemetry.reset_reason {
            headers.insert(PROVIDER_SESSION_RESET_REASON_HEADER.to_string(), reason);
        }
        if let Some(turn_state) = headers.get(CODEX_TURN_STATE_HEADER).cloned() {
            self.state.lock().unwrap().turn_state = Some(turn_state);
        }
        headers
    }

    fn wrap_stream_state(
        &self,
        parts: crate::ai_sdk_core::PartStream,
        request_body: Value,
        track_incremental_state: bool,
    ) -> crate::ai_sdk_core::PartStream {
        let state = Arc::clone(&self.state);
        Box::pin(async_stream::stream! {
            let mut stream = parts;
            let mut completed_response_id = None;
            let mut completed = false;
            let mut state_committed = false;
            while let Some(item) = stream.next().await {
                match &item {
                    Ok(v2t::StreamPart::ResponseMetadata { meta }) => {
                        completed_response_id = meta.id.clone();
                    }
                    Ok(v2t::StreamPart::Finish { provider_metadata, .. }) => {
                        completed = true;
                        completed_response_id =
                            provider_response_id_from_metadata(provider_metadata.as_ref())
                                .or(completed_response_id.clone());
                        write_turn_session_incremental_state(
                            &state,
                            &request_body,
                            track_incremental_state,
                            completed_response_id.clone(),
                        );
                        state_committed = true;
                    }
                    Err(_) => {
                        completed = false;
                        completed_response_id = None;
                        write_turn_session_incremental_state(
                            &state,
                            &request_body,
                            track_incremental_state,
                            None,
                        );
                        state_committed = true;
                    }
                    _ => {}
                }
                yield item;
            }

            if !state_committed {
                write_turn_session_incremental_state(
                    &state,
                    &request_body,
                    track_incremental_state,
                    if completed { completed_response_id } else { None },
                );
            }
        })
    }

    async fn stream_http_request(
        &self,
        mut body: Value,
        include_raw: bool,
        requested: ResponseTransportMode,
        extra_headers: &HashMap<String, String>,
        warnings: Vec<v2t::CallWarning>,
        tool_name_mapping: ToolNameMapping,
        approval_request_id_map: HashMap<String, String>,
        store_for_stream: bool,
        logprobs_enabled: bool,
    ) -> Result<StreamResponse, SdkError> {
        if let Some(limiter) = &self.model.limiter {
            let _ = limiter.until_ready().await;
        }
        if let Some(object) = body.as_object_mut() {
            object.remove("previous_response_id");
        }
        let base_headers = self.model.request_headers(extra_headers);
        let request_body = transport_request_body(
            &body,
            ResponseTransportMode::Http,
            &self.model.config.endpoint_path,
        );
        let (stream, transport_headers) = self
            .model
            .send_once(&base_headers, &body, ResponseTransportMode::Http)
            .await?;
        let response_headers = self.response_headers(
            transport_headers,
            requested,
            ResponseTransportMode::Http,
            Some(ResponseTransportMode::Websocket),
            false,
            false,
            false,
        );
        let pipeline = PipelineBuilder::<OpenAIResponsesChunk>::new()
            .with_provider("openai_official")
            .include_raw(include_raw)
            .build(map_transport_stream(stream));
        let parts = map_events_to_parts(
            Box::pin(pipeline),
            super::stream_hooks::build_stream_mapper_config(
                warnings,
                tool_name_mapping,
                approval_request_id_map,
                store_for_stream,
                logprobs_enabled,
            ),
        );
        Ok(StreamResponse {
            stream: self.wrap_stream_state(parts, request_body.clone(), false),
            request_body: Some(request_body),
            response_headers: Some(response_headers),
        })
    }

    async fn stream_http_on_websocket_error(
        &mut self,
        err: SdkError,
        transport_selection: ResponseTransportSelection,
        body: Value,
        include_raw: bool,
        extra_headers: &HashMap<String, String>,
        warnings: Vec<v2t::CallWarning>,
        tool_name_mapping: ToolNameMapping,
        approval_request_id_map: HashMap<String, String>,
        store_for_stream: bool,
        logprobs_enabled: bool,
    ) -> Result<StreamResponse, SdkError> {
        if transport_selection.fallback_http && should_fallback_to_http_after_websocket_error(&err)
        {
            self.activate_http_fallback("websocket_http_fallback");
            self.stream_http_request(
                body,
                include_raw,
                transport_selection.requested,
                extra_headers,
                warnings,
                tool_name_mapping,
                approval_request_id_map,
                store_for_stream,
                logprobs_enabled,
            )
            .await
        } else {
            Err(err)
        }
    }

    async fn send_session_websocket_request_with_prewarm(
        &mut self,
        body: &Value,
        websocket_headers: &[(String, String)],
    ) -> Result<SessionWebsocketRequest, SdkError> {
        if !self.should_prewarm_websocket(body) {
            return self.send_session_websocket_request(body.clone()).await;
        }

        match self.send_session_websocket_request(body.clone()).await {
            Ok(result) => Ok(result),
            Err(SdkError::RateLimited { .. }) => {
                self.ensure_websocket_connection(websocket_headers).await?;
                self.prewarm_websocket_session(body).await?;
                self.send_session_websocket_request(body.clone()).await
            }
            Err(err) => Err(err),
        }
    }
}

#[async_trait::async_trait]
impl<T: HttpTransport + Send + Sync + 'static> LanguageModelTurnSession
    for OpenAIResponsesTurnSession<'_, T>
{
    fn provider_name(&self) -> &'static str {
        "OpenAI"
    }

    fn model_id(&self) -> &str {
        &self.model.model_id
    }

    async fn do_stream(&mut self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.model.config.provider_scope_name,
            self.model.config.default_options.as_ref(),
        );
        let prov = parse_openai_provider_options(
            &options.provider_options,
            &self.model.config.provider_scope_name,
        );
        let transport_selection =
            resolve_transport_selection(&self.model.config.endpoint_path, &prov);
        if transport_selection.requested != ResponseTransportMode::Websocket
            || !should_use_codex_oauth_websocket_transport(&self.model.config.endpoint_path)
        {
            return self.model.do_stream(options).await;
        }

        let tool_name_mapping = build_tool_name_mapping(&options.tools);
        let (mut body, warnings) =
            build_request_body(&options, &self.model.model_id, &self.model.config)?;
        body["stream"] = Value::Bool(true);
        let store_for_stream = prov.store.unwrap_or(false);
        let logprobs_enabled =
            prov.logprobs_bool.unwrap_or(false) || prov.logprobs_n.unwrap_or(0) > 0;
        let approval_request_id_map = extract_approval_request_id_to_tool_call_id(
            &options.prompt,
            &self.model.config.provider_scope_name,
        );

        if self.state.lock().unwrap().force_http {
            return self
                .stream_http_request(
                    body,
                    options.include_raw_chunks,
                    transport_selection.requested,
                    &options.headers,
                    warnings,
                    tool_name_mapping,
                    approval_request_id_map,
                    store_for_stream,
                    logprobs_enabled,
                )
                .await;
        }

        if let Some(limiter) = &self.model.limiter {
            let _ = limiter.until_ready().await;
        }

        let websocket_headers = self.websocket_headers(&options.headers);
        let reused = match self.ensure_websocket_connection(&websocket_headers).await {
            Ok(reused) => reused,
            Err(err) => {
                return self
                    .stream_http_on_websocket_error(
                        err,
                        transport_selection,
                        body,
                        options.include_raw_chunks,
                        &options.headers,
                        warnings,
                        tool_name_mapping,
                        approval_request_id_map,
                        store_for_stream,
                        logprobs_enabled,
                    )
                    .await;
            }
        };
        let (
            session_body,
            transport_body,
            stream,
            transport_headers,
            previous_response_id_used,
            warmup_response_id_used,
        ) = match self
            .send_session_websocket_request_with_prewarm(&body, &websocket_headers)
            .await
        {
            Ok(result) => result,
            Err(err) => {
                return self
                    .stream_http_on_websocket_error(
                        err,
                        transport_selection,
                        body,
                        options.include_raw_chunks,
                        &options.headers,
                        warnings,
                        tool_name_mapping,
                        approval_request_id_map,
                        store_for_stream,
                        logprobs_enabled,
                    )
                    .await;
            }
        };
        let response_headers = self.response_headers(
            transport_headers,
            transport_selection.requested,
            ResponseTransportMode::Websocket,
            None,
            reused,
            previous_response_id_used,
            warmup_response_id_used,
        );
        let pipeline = PipelineBuilder::<OpenAIResponsesChunk>::new()
            .with_provider("openai_official")
            .include_raw(options.include_raw_chunks)
            .build(stream);
        let parts = map_events_to_parts(
            Box::pin(pipeline),
            super::stream_hooks::build_stream_mapper_config(
                warnings,
                tool_name_mapping,
                approval_request_id_map,
                store_for_stream,
                logprobs_enabled,
            ),
        );
        Ok(StreamResponse {
            stream: self.wrap_stream_state(parts, session_body, true),
            request_body: Some(transport_body),
            response_headers: Some(response_headers),
        })
    }
}

fn request_shape_matches_previous(previous: Option<&Value>, current: &Value) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    request_shape_without_input(previous) == request_shape_without_input(current)
}

fn request_shape_without_input(value: &Value) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("input");
        object.remove("previous_response_id");
        object.remove("generate");
    }
    value
}

fn provider_response_id_from_metadata(metadata: Option<&v2t::ProviderMetadata>) -> Option<String> {
    metadata
        .and_then(|outer| outer.get("openai"))
        .and_then(|inner| inner.get("responseId"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn write_turn_session_incremental_state(
    state: &Arc<Mutex<OpenAIResponsesTurnSessionState>>,
    request_body: &Value,
    track_incremental_state: bool,
    response_id: Option<String>,
) {
    let mut state = state.lock().unwrap();
    if track_incremental_state {
        state.last_request = Some(request_body.clone());
        state.last_response_id = response_id;
    } else {
        state.last_request = None;
        state.last_response_id = None;
    }
    state.last_response_id_from_warmup = false;
    state.last_reset_reason = None;
}

pub(super) fn should_use_codex_oauth_websocket_transport(endpoint_path: &str) -> bool {
    endpoint_path
        .trim()
        .trim_end_matches('/')
        .eq_ignore_ascii_case("/backend-api/codex/responses")
}

fn to_websocket_url(url: &str) -> Result<String, SdkError> {
    let mut parsed = Url::parse(url).map_err(|err| SdkError::InvalidArgument {
        message: format!("invalid endpoint url '{url}': {err}"),
    })?;
    let new_scheme = match parsed.scheme() {
        "https" => "wss",
        "http" => "ws",
        "wss" | "ws" => return Ok(parsed.into()),
        scheme => {
            return Err(SdkError::InvalidArgument {
                message: format!("unsupported endpoint scheme '{scheme}' for websocket stream"),
            });
        }
    };
    parsed
        .set_scheme(new_scheme)
        .map_err(|_| SdkError::InvalidArgument {
            message: format!("failed to convert endpoint scheme to '{new_scheme}'"),
        })?;
    Ok(parsed.into())
}

fn transport_request_body(
    body: &Value,
    transport: ResponseTransportMode,
    endpoint_path: &str,
) -> Value {
    if transport != ResponseTransportMode::Websocket {
        return body.clone();
    }

    if should_use_codex_oauth_websocket_transport(endpoint_path) {
        let mut frame = serde_json::Map::new();
        frame.insert("type".into(), Value::String("response.create".into()));
        if let Some(body_obj) = body.as_object() {
            frame.extend(body_obj.clone());
            return Value::Object(frame);
        }
    }

    json!({
        "type": "response.create",
        "response": body,
    })
}

async fn prefetch_stream(stream: RawByteStream) -> Result<ByteStream, SdkError> {
    let mut stream = stream;
    let Some(first) = stream.next().await else {
        return Err(map_transport_stream_error(TransportError::StreamClosed));
    };
    let first = first.map_err(map_transport_stream_error)?;
    let rest = stream.map(|chunk| chunk.map_err(map_transport_stream_error));
    Ok(Box::pin(stream::once(async move { Ok(first) }).chain(rest)))
}

fn map_transport_stream(stream: RawByteStream) -> ByteStream {
    Box::pin(stream.map(|chunk| chunk.map_err(map_transport_stream_error)))
}

fn map_transport_stream_error(err: TransportError) -> SdkError {
    match err {
        TransportError::IdleReadTimeout(_) => SdkError::Timeout,
        TransportError::ConnectTimeout(_) => SdkError::Timeout,
        other => SdkError::Transport(other),
    }
}

fn should_fallback_to_http_after_websocket_error(err: &SdkError) -> bool {
    !matches!(
        err,
        SdkError::RateLimited { .. }
            | SdkError::Upstream {
                status: 401 | 403,
                ..
            }
    )
}

fn response_headers_with_transport(
    headers: Vec<(String, String)>,
    requested: ResponseTransportMode,
    effective: ResponseTransportMode,
    fallback_from: Option<ResponseTransportMode>,
) -> v2t::Headers {
    let mut out = v2t::Headers::new();
    for (name, value) in headers {
        out.insert(name.to_ascii_lowercase(), value);
    }
    out.insert(
        TRANSPORT_HEADER_REQUESTED.to_string(),
        requested.as_str().to_string(),
    );
    out.insert(
        TRANSPORT_HEADER_EFFECTIVE.to_string(),
        effective.as_str().to_string(),
    );
    if let Some(from) = fallback_from {
        out.insert(
            TRANSPORT_HEADER_FALLBACK.to_string(),
            format!("{}->{}", from.as_str(), effective.as_str()),
        );
    }
    out
}

// Convenience constructor for default reqwest transport
impl OpenAIResponsesLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn builder(
        model_id: impl Into<String>,
    ) -> crate::provider_openai::provider::OpenAIResponsesBuilder {
        crate::provider_openai::provider::OpenAIResponsesBuilder::new(model_id)
    }

    pub fn create_simple(
        model_id: impl Into<String>,
        base_url: Option<String>,
        api_key: String,
    ) -> Self {
        let mut builder = Self::builder(model_id);
        if let Some(base_url) = base_url {
            builder = builder.with_base_url(base_url);
        }
        if !api_key.is_empty() {
            builder = builder.with_api_key(api_key);
        }
        builder
            .build()
            .expect("openai builder should create the default reqwest transport")
    }
}

#[async_trait::async_trait]
impl<T: HttpTransport + Send + Sync + 'static> LanguageModel for OpenAIResponsesLanguageModel<T> {
    fn provider_name(&self) -> &'static str {
        "OpenAI"
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn new_turn_session(&self) -> crate::ai_sdk_core::BoxedLanguageModelTurnSession<'_> {
        Box::new(OpenAIResponsesTurnSession::new(self))
    }

    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.config.supported_urls.clone()
    }

    async fn do_generate(&self, options: v2t::CallOptions) -> Result<GenerateResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        let tool_name_mapping = build_tool_name_mapping(&options.tools);
        let (body, warnings) = build_request_body(&options, &self.model_id, &self.config)?;
        let json = self.post_response_json(&body, &options.headers).await?;
        maybe_openai_response_error(&json)?;

        let approval_request_id_map = extract_approval_request_id_to_tool_call_id(
            &options.prompt,
            &self.config.provider_scope_name,
        );
        let (content, has_function_calls) =
            extract_response_content(&json, &tool_name_mapping, &approval_request_id_map);
        let usage = extract_openai_generate_usage(&json);
        let finish_reason = extract_openai_finish_reason(&json, has_function_calls);
        let provider_metadata = extract_openai_generate_provider_metadata(&json);

        Ok(GenerateResponse {
            content,
            finish_reason,
            usage,
            provider_metadata,
            request_body: None,
            response_headers: None,
            response_body: Some(json),
            warnings,
        })
    }

    async fn do_stream(&self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        let prov = parse_openai_provider_options(
            &options.provider_options,
            &self.config.provider_scope_name,
        );
        let transport_selection = resolve_transport_selection(&self.config.endpoint_path, &prov);
        let tool_name_mapping = build_tool_name_mapping(&options.tools);
        let (mut body, warnings) = build_request_body(&options, &self.model_id, &self.config)?;
        body["stream"] = Value::Bool(true);
        let store_for_stream = prov.store.unwrap_or(false);
        let logprobs_enabled =
            prov.logprobs_bool.unwrap_or(false) || prov.logprobs_n.unwrap_or(0) > 0;
        let approval_request_id_map = extract_approval_request_id_to_tool_call_id(
            &options.prompt,
            &self.config.provider_scope_name,
        );
        let request_body = transport_request_body(
            &body,
            transport_selection.requested,
            &self.config.endpoint_path,
        );
        let (stream, response_headers) = self
            .stream_with_body(
                body,
                options.include_raw_chunks,
                transport_selection,
                &options.headers,
            )
            .await?;
        let parts = map_events_to_parts(
            stream,
            super::stream_hooks::build_stream_mapper_config(
                warnings,
                tool_name_mapping,
                approval_request_id_map,
                store_for_stream,
                logprobs_enabled,
            ),
        );
        Ok(StreamResponse {
            stream: parts,
            request_body: Some(request_body),
            response_headers: Some(response_headers),
        })
    }
}

// ----- Helpers: request building and SSE mapping -----

pub(super) fn parse_openai_usage(u: &serde_json::Value) -> Option<TokenUsage> {
    let obj = u.as_object()?;
    let input = obj
        .get("input_tokens")
        .or_else(|| obj.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let output = obj
        .get("output_tokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let total = obj
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(input + output);
    let cache_read_tokens = obj
        .get("cache_read_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| parse_openai_cached_input_tokens(u).map(|v| v as usize));
    let cache_write_tokens = obj
        .get("cache_write_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| {
            obj.get("cache_creation")
                .and_then(|cc| cc.as_object())
                .and_then(|co| {
                    let a = co
                        .get("ephemeral_5m_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let b = co
                        .get("ephemeral_1h_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let sum = a + b;
                    if sum > 0 {
                        Some(sum as usize)
                    } else {
                        None
                    }
                })
        });
    Some(TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        cache_read_tokens,
        cache_write_tokens,
    })
}

fn maybe_openai_response_error(json: &Value) -> Result<(), SdkError> {
    let Some(error) = json.get("error").filter(|value| !value.is_null()) else {
        return Ok(());
    };

    let message = error
        .get("message")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| error.to_string());
    Err(SdkError::Upstream {
        status: 400,
        message,
        source: None,
    })
}

fn extract_openai_generate_usage(json: &Value) -> v2t::Usage {
    let mut usage = v2t::Usage::default();
    let usage_val = json.get("usage").or_else(|| {
        json.get("response")
            .and_then(|response| response.get("usage"))
    });

    if let Some(usage_tokens) = usage_val.and_then(parse_openai_usage) {
        usage.input_tokens = Some(usage_tokens.input_tokens as u64);
        usage.output_tokens = Some(usage_tokens.output_tokens as u64);
        usage.total_tokens = Some(usage_tokens.total_tokens as u64);
        if let Some(cached_input_tokens) = usage_tokens.cache_read_tokens {
            usage.cached_input_tokens = Some(cached_input_tokens as u64);
        }
    }
    if let Some(raw_usage) = usage_val {
        apply_openai_usage_details(raw_usage, &mut usage);
    }

    usage
}

fn extract_openai_finish_reason(json: &Value, has_function_calls: bool) -> v2t::FinishReason {
    let finish_hint = json
        .get("incomplete_details")
        .and_then(|value| value.get("reason"))
        .and_then(|value| value.as_str());
    map_finish_reason(finish_hint, has_function_calls)
}

fn extract_openai_generate_provider_metadata(json: &Value) -> Option<v2t::ProviderMetadata> {
    let response_id = openai_response_field(json, "id");
    let service_tier = openai_response_field(json, "service_tier");
    if response_id.is_none() && service_tier.is_none() {
        return None;
    }

    let mut outer = HashMap::new();
    let mut inner = HashMap::new();
    if let Some(response_id) = response_id {
        inner.insert("responseId".into(), json!(response_id));
    }
    if let Some(service_tier) = service_tier {
        inner.insert("serviceTier".into(), json!(service_tier));
    }
    outer.insert("openai".into(), inner);
    Some(outer)
}

fn openai_response_field<'a>(json: &'a Value, field: &str) -> Option<&'a str> {
    json.get(field)
        .and_then(|value| value.as_str())
        .or_else(|| {
            json.get("response")
                .and_then(|response| response.get(field))
                .and_then(|value| value.as_str())
        })
}

fn parse_openai_cached_input_tokens(u: &serde_json::Value) -> Option<u64> {
    u.get("input_tokens_details")
        .and_then(|v| v.get("cached_tokens"))
        .or_else(|| {
            u.get("prompt_tokens_details")
                .and_then(|v| v.get("cached_tokens"))
        })
        .and_then(|v| v.as_u64())
}

fn parse_openai_reasoning_tokens(u: &serde_json::Value) -> Option<u64> {
    u.get("output_tokens_details")
        .and_then(|v| v.get("reasoning_tokens"))
        .or_else(|| {
            u.get("completion_tokens_details")
                .and_then(|v| v.get("reasoning_tokens"))
        })
        .and_then(|v| v.as_u64())
}

pub(super) fn apply_openai_usage_details(u: &serde_json::Value, usage: &mut v2t::Usage) {
    if let Some(cached) = parse_openai_cached_input_tokens(u) {
        usage.cached_input_tokens = Some(cached);
    }
    if let Some(reasoning) = parse_openai_reasoning_tokens(u) {
        usage.reasoning_tokens = Some(reasoning);
    }
}

// Ensure tool schemas always have a top-level "type":"object".
pub(super) fn normalize_object_schema(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|v| v.as_str()).is_some() {
                return schema.clone();
            }
            if map.contains_key("properties") || !map.is_empty() {
                let mut out = serde_json::Map::new();
                out.insert("type".into(), serde_json::Value::String("object".into()));
                if let Some(props) = map.get("properties") {
                    out.insert("properties".into(), props.clone());
                } else {
                    out.insert("properties".into(), serde_json::Value::Object(map.clone()));
                }
                if let Some(req) = map.get("required") {
                    out.insert("required".into(), req.clone());
                }
                return serde_json::Value::Object(out);
            }
            json!({"type":"object"})
        }
        _ => json!({"type":"object"}),
    }
}

fn extract_approval_request_id_to_tool_call_id(
    prompt: &[v2t::PromptMessage],
    provider_scope_name: &str,
) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    for message in prompt {
        if let v2t::PromptMessage::Assistant { content, .. } = message {
            for part in content {
                if let v2t::AssistantPart::ToolCall(call) = part {
                    if let Some(opts) = call.provider_options.as_ref() {
                        if let Some(scope) = opts.get(provider_scope_name) {
                            let approval_id =
                                scope.get("approvalRequestId").and_then(|v| v.as_str());
                            if let Some(approval_id) = approval_id {
                                mapping.insert(approval_id.to_string(), call.tool_call_id.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    mapping
}

pub(super) fn escape_json_delta(delta: &str) -> String {
    if delta
        .as_bytes()
        .iter()
        .all(|b| *b >= 0x20 && *b != b'"' && *b != b'\\')
    {
        return delta.to_string();
    }
    let encoded = serde_json::to_string(delta).unwrap_or_else(|_| "\"\"".into());
    if encoded.len() >= 2 && encoded.starts_with('"') && encoded.ends_with('"') {
        encoded[1..encoded.len() - 1].to_string()
    } else {
        encoded
    }
}

#[derive(Default)]
pub(crate) struct OpenAIResponsesChunk {
    tool_calls: HashMap<usize, OpenAIToolCallState>,
    pending_deltas: HashMap<usize, Vec<String>>,
}

struct OpenAIToolCallState {
    id: String,
}

impl OpenAIResponsesChunk {
    fn output_index(json: &Value) -> Option<usize> {
        json.get("output_index")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
    }

    fn push_data(events: &mut Vec<Event>, key: impl Into<String>, value: Value) {
        events.push(Event::Data {
            key: key.into(),
            value,
        });
    }

    fn flush_pending_tool_call_deltas(
        &mut self,
        output_index: usize,
        call_id: &str,
        events: &mut Vec<Event>,
    ) {
        if let Some(pending) = self.pending_deltas.remove(&output_index) {
            for delta in pending {
                events.push(Event::ToolCallDelta {
                    id: call_id.to_string(),
                    args_json: delta,
                });
            }
        }
    }

    fn finish_tool_call(&mut self, output_index: usize, events: &mut Vec<Event>) {
        if let Some(state) = self.tool_calls.remove(&output_index) {
            self.flush_pending_tool_call_deltas(output_index, &state.id, events);
            events.push(Event::ToolCallEnd { id: state.id });
        }
    }

    fn close_open_tool_calls(&mut self, events: &mut Vec<Event>) {
        if self.tool_calls.is_empty() {
            return;
        }
        for (_output_index, state) in self.tool_calls.drain() {
            events.push(Event::ToolCallEnd { id: state.id });
        }
        self.pending_deltas.clear();
    }

    fn handle_response_created(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(response) = json.get("response") else {
            return;
        };
        let id = response.get("id").cloned().unwrap_or(Value::Null);
        let model = response.get("model").cloned().unwrap_or(Value::Null);
        let created_at = response
            .get("created_at")
            .and_then(|value| value.as_i64().or_else(|| value.as_u64().map(|n| n as i64)))
            .map(|value| value * 1000);
        Self::push_data(
            events,
            "openai.response_metadata",
            json!({
                "id": id,
                "model": model,
                "created_at": created_at,
            }),
        );
    }

    fn handle_output_text_event(&self, event_type: &str, json: &Value, events: &mut Vec<Event>) {
        match event_type {
            "response.output_text.delta" => self.handle_output_text_delta(json, events),
            "response.output_text.annotation.added" => {
                self.handle_output_text_annotation(json, events);
            }
            _ => {}
        }
    }

    fn handle_output_text_delta(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(item_id) = json.get("item_id").and_then(|value| value.as_str()) else {
            return;
        };
        let delta = json
            .get("delta")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if delta.is_empty() {
            return;
        }
        let logprobs = json.get("logprobs").cloned().unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.text_delta",
            json!({
                "item_id": item_id,
                "delta": delta,
                "logprobs": logprobs,
            }),
        );
    }

    fn handle_output_text_annotation(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(item_id) = json.get("item_id").and_then(|value| value.as_str()) else {
            return;
        };
        let annotation = json.get("annotation").cloned().unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.text_annotation",
            json!({
                "item_id": item_id,
                "annotation": annotation,
            }),
        );
    }

    fn handle_output_item_event(
        &mut self,
        event_type: &str,
        json: &Value,
        events: &mut Vec<Event>,
    ) {
        let Some(item) = json.get("item").and_then(|value| value.as_object()) else {
            return;
        };
        match event_type {
            "response.output_item.added" => self.handle_output_item_added(json, item, events),
            "response.output_item.done" => self.handle_output_item_done(json, item, events),
            _ => {}
        }
    }

    fn handle_output_item_added(
        &mut self,
        json: &Value,
        item: &Map<String, Value>,
        events: &mut Vec<Event>,
    ) {
        let Some(item_type) = item.get("type").and_then(|value| value.as_str()) else {
            return;
        };
        match item_type {
            "function_call" => self.handle_function_call_added(json, item, events),
            "apply_patch_call" => self.handle_apply_patch_call_added(json, item, events),
            "message" => Self::handle_message_added(item, events),
            "reasoning" => Self::handle_reasoning_added(item, events),
            "web_search_call" => {
                Self::handle_simple_tool_item_added(item, "openai.web_search_call.added", events)
            }
            "file_search_call" => {
                Self::handle_simple_tool_item_added(item, "openai.file_search_call.added", events)
            }
            "image_generation_call" => Self::handle_simple_tool_item_added(
                item,
                "openai.image_generation_call.added",
                events,
            ),
            "code_interpreter_call" => self.handle_code_interpreter_call_added(json, item, events),
            "computer_call" => {
                Self::handle_simple_tool_item_added(item, "openai.computer_call.added", events)
            }
            _ => {}
        }
    }

    fn handle_function_call_added(
        &mut self,
        json: &Value,
        item: &Map<String, Value>,
        events: &mut Vec<Event>,
    ) {
        let name = item
            .get("name")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let call_id = item
            .get("call_id")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let output_index = Self::output_index(json);

        if let (Some(call_id), Some(item_id)) = (
            call_id.as_ref(),
            item.get("id").and_then(|value| value.as_str()),
        ) {
            Self::push_data(
                events,
                format!("openai.tool_item_id.{call_id}"),
                json!({ "item_id": item_id }),
            );
        }

        if let (Some(output_index), Some(call_id), Some(tool_name)) = (output_index, call_id, name)
        {
            self.tool_calls.insert(
                output_index,
                OpenAIToolCallState {
                    id: call_id.clone(),
                },
            );
            events.push(Event::ToolCallStart {
                id: call_id.clone(),
                name: tool_name,
            });
            self.flush_pending_tool_call_deltas(output_index, &call_id, events);
        }
    }

    fn handle_apply_patch_call_added(
        &self,
        json: &Value,
        item: &Map<String, Value>,
        events: &mut Vec<Event>,
    ) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let Some(call_id) = item.get("call_id").and_then(|value| value.as_str()) else {
            return;
        };
        let operation = item.get("operation").cloned().unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.apply_patch_call.added",
            json!({
                "output_index": output_index,
                "call_id": call_id,
                "operation": operation,
            }),
        );
    }

    fn handle_message_added(item: &Map<String, Value>, events: &mut Vec<Event>) {
        let Some(item_id) = item.get("id").and_then(|value| value.as_str()) else {
            return;
        };
        Self::push_data(
            events,
            "openai.message_added",
            json!({ "item_id": item_id }),
        );
    }

    fn handle_reasoning_added(item: &Map<String, Value>, events: &mut Vec<Event>) {
        let Some(item_id) = item.get("id").and_then(|value| value.as_str()) else {
            return;
        };
        let encrypted_content = item
            .get("encrypted_content")
            .cloned()
            .unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.reasoning_added",
            json!({
                "item_id": item_id,
                "encrypted_content": encrypted_content,
            }),
        );
    }

    fn handle_simple_tool_item_added(
        item: &Map<String, Value>,
        event_key: &str,
        events: &mut Vec<Event>,
    ) {
        let Some(tool_call_id) = item.get("id").and_then(|value| value.as_str()) else {
            return;
        };
        Self::push_data(events, event_key, json!({ "tool_call_id": tool_call_id }));
    }

    fn handle_code_interpreter_call_added(
        &self,
        json: &Value,
        item: &Map<String, Value>,
        events: &mut Vec<Event>,
    ) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let Some(tool_call_id) = item.get("id").and_then(|value| value.as_str()) else {
            return;
        };
        let container_id = item
            .get("container_id")
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        Self::push_data(
            events,
            "openai.code_interpreter_call.added",
            json!({
                "output_index": output_index,
                "tool_call_id": tool_call_id,
                "container_id": container_id,
            }),
        );
    }

    fn handle_output_item_done(
        &mut self,
        json: &Value,
        item: &Map<String, Value>,
        events: &mut Vec<Event>,
    ) {
        match item.get("type").and_then(|value| value.as_str()) {
            Some("apply_patch_call") => self.handle_apply_patch_call_done(json, item, events),
            Some("message") => Self::handle_message_done(item, events),
            Some("reasoning") => Self::handle_reasoning_done(item, events),
            Some("function_call") => {
                Self::push_data(events, "openai.function_call_done", json!({}));
            }
            _ => {}
        }

        if let Some(tool_data) = provider_tool_data_from_output_item(item) {
            Self::push_data(events, "openai.provider_tool", tool_data);
        }

        if let Some(output_index) = Self::output_index(json) {
            self.finish_tool_call(output_index, events);
        }
    }

    fn handle_apply_patch_call_done(
        &self,
        json: &Value,
        item: &Map<String, Value>,
        events: &mut Vec<Event>,
    ) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let call_id = item.get("call_id").and_then(|value| value.as_str());
        let operation = item.get("operation").cloned().unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.apply_patch_call.done",
            json!({
                "output_index": output_index,
                "call_id": call_id,
                "operation": operation,
            }),
        );
    }

    fn handle_message_done(item: &Map<String, Value>, events: &mut Vec<Event>) {
        let Some(item_id) = item.get("id").and_then(|value| value.as_str()) else {
            return;
        };
        Self::push_data(events, "openai.message_done", json!({ "item_id": item_id }));
    }

    fn handle_reasoning_done(item: &Map<String, Value>, events: &mut Vec<Event>) {
        let Some(item_id) = item.get("id").and_then(|value| value.as_str()) else {
            return;
        };
        let encrypted_content = item
            .get("encrypted_content")
            .cloned()
            .unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.reasoning_done",
            json!({
                "item_id": item_id,
                "encrypted_content": encrypted_content,
            }),
        );
    }

    fn handle_function_call_arguments_event(
        &mut self,
        event_type: &str,
        json: &Value,
        events: &mut Vec<Event>,
    ) {
        if event_type != "response.function_call_arguments.delta" {
            return;
        }
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let delta = json
            .get("delta")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(state) = self.tool_calls.get(&output_index) {
            events.push(Event::ToolCallDelta {
                id: state.id.clone(),
                args_json: delta,
            });
        } else {
            self.pending_deltas
                .entry(output_index)
                .or_default()
                .push(delta);
        }
    }

    fn handle_code_interpreter_event(
        &self,
        event_type: &str,
        json: &Value,
        events: &mut Vec<Event>,
    ) {
        match event_type {
            "response.code_interpreter_call_code.delta" => {
                self.handle_code_interpreter_code_delta(json, events);
            }
            "response.code_interpreter_call_code.done" => {
                self.handle_code_interpreter_code_done(json, events);
            }
            _ => {}
        }
    }

    fn handle_code_interpreter_code_delta(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let delta = json
            .get("delta")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        if delta.is_empty() {
            return;
        }
        Self::push_data(
            events,
            "openai.code_interpreter_call.code_delta",
            json!({
                "output_index": output_index,
                "delta": delta,
            }),
        );
    }

    fn handle_code_interpreter_code_done(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let code = json
            .get("code")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        Self::push_data(
            events,
            "openai.code_interpreter_call.code_done",
            json!({
                "output_index": output_index,
                "code": code,
            }),
        );
    }

    fn handle_apply_patch_event(&self, event_type: &str, json: &Value, events: &mut Vec<Event>) {
        match event_type {
            "response.apply_patch_call_operation_diff.delta" => {
                self.handle_apply_patch_diff_delta(json, events);
            }
            "response.apply_patch_call_operation_diff.done" => {
                self.handle_apply_patch_diff_done(json, events);
            }
            _ => {}
        }
    }

    fn handle_apply_patch_diff_delta(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let delta = json
            .get("delta")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        if delta.is_empty() {
            return;
        }
        Self::push_data(
            events,
            "openai.apply_patch_call.diff.delta",
            json!({
                "output_index": output_index,
                "delta": delta,
            }),
        );
    }

    fn handle_apply_patch_diff_done(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(output_index) = Self::output_index(json) else {
            return;
        };
        let diff = json
            .get("diff")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        Self::push_data(
            events,
            "openai.apply_patch_call.diff.done",
            json!({
                "output_index": output_index,
                "diff": diff,
            }),
        );
    }

    fn handle_reasoning_summary_event(
        &self,
        event_type: &str,
        json: &Value,
        events: &mut Vec<Event>,
    ) {
        match event_type {
            "response.reasoning_summary_part.added" => {
                self.handle_reasoning_summary_added(json, events);
            }
            "response.reasoning_summary_text.delta" => {
                self.handle_reasoning_summary_delta(json, events);
            }
            "response.reasoning_summary_part.done" => {
                self.handle_reasoning_summary_done(json, events);
            }
            _ => {}
        }
    }

    fn handle_reasoning_summary_added(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(item_id) = json.get("item_id").and_then(|value| value.as_str()) else {
            return;
        };
        let Some(summary_index) = json.get("summary_index").and_then(|value| value.as_u64()) else {
            return;
        };
        Self::push_data(
            events,
            "openai.reasoning_summary_added",
            json!({
                "item_id": item_id,
                "summary_index": summary_index,
            }),
        );
    }

    fn handle_reasoning_summary_delta(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(item_id) = json.get("item_id").and_then(|value| value.as_str()) else {
            return;
        };
        let Some(summary_index) = json.get("summary_index").and_then(|value| value.as_u64()) else {
            return;
        };
        let delta = json
            .get("delta")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if delta.is_empty() {
            return;
        }
        Self::push_data(
            events,
            "openai.reasoning_summary_delta",
            json!({
                "item_id": item_id,
                "summary_index": summary_index,
                "delta": delta,
            }),
        );
    }

    fn handle_reasoning_summary_done(&self, json: &Value, events: &mut Vec<Event>) {
        let Some(item_id) = json.get("item_id").and_then(|value| value.as_str()) else {
            return;
        };
        let Some(summary_index) = json.get("summary_index").and_then(|value| value.as_u64()) else {
            return;
        };
        Self::push_data(
            events,
            "openai.reasoning_summary_done",
            json!({
                "item_id": item_id,
                "summary_index": summary_index,
            }),
        );
    }

    fn handle_image_generation_partial(&self, json: &Value, events: &mut Vec<Event>) {
        let (Some(tool_call_id), Some(partial_image_b64)) = (
            json.get("item_id").and_then(|value| value.as_str()),
            json.get("partial_image_b64")
                .and_then(|value| value.as_str()),
        ) else {
            return;
        };
        Self::push_data(
            events,
            "openai.image_generation_call.partial",
            json!({
                "tool_call_id": tool_call_id,
                "partial_image_b64": partial_image_b64,
            }),
        );
    }

    fn handle_terminal_event(&mut self, event_type: &str, json: &Value, events: &mut Vec<Event>) {
        match event_type {
            "response.completed" | "response.incomplete" => {
                self.handle_response_terminal(json, events);
            }
            "response.failed" => self.handle_response_failed(json, events),
            "error" => Self::push_data(events, "openai.error", json.clone()),
            _ => {}
        }
    }

    fn handle_response_terminal(&mut self, json: &Value, events: &mut Vec<Event>) {
        if let Some(response) = json.get("response") {
            self.handle_response_terminal_metadata(response, events);
        }
        self.close_open_tool_calls(events);
        events.push(Event::Done);
    }

    fn handle_response_terminal_metadata(&self, response: &Value, events: &mut Vec<Event>) {
        if let Some(usage_value) = response.get("usage") {
            if let Some(usage) = parse_openai_usage(usage_value) {
                events.push(Event::Usage { usage });
            }
            Self::push_data(events, "usage", usage_value.clone());
        }
        let incomplete_reason = response
            .get("incomplete_details")
            .and_then(|value| value.get("reason"))
            .cloned()
            .unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.finish",
            json!({ "incomplete_reason": incomplete_reason }),
        );
        let response_id = response.get("id").cloned().unwrap_or(Value::Null);
        let service_tier = response.get("service_tier").cloned().unwrap_or(Value::Null);
        Self::push_data(
            events,
            "openai.response",
            json!({
                "id": response_id,
                "service_tier": service_tier,
            }),
        );
    }

    fn handle_response_failed(&mut self, json: &Value, events: &mut Vec<Event>) {
        let failed_payload = json
            .get("response")
            .map(|response| {
                json!({
                    "id": response.get("id").cloned().unwrap_or(Value::Null),
                })
            })
            .unwrap_or_else(|| json!({}));
        Self::push_data(events, "openai.failed", failed_payload);
        self.close_open_tool_calls(events);
        events.push(Event::Done);
    }
}

impl ProviderChunk for OpenAIResponsesChunk {
    fn try_from_sse(&mut self, event: &SseEvent) -> Result<Option<Vec<Event>>, SdkError> {
        let trimmed = std::str::from_utf8(&event.data).unwrap_or("").trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return Ok(None);
        }
        let json: serde_json::Value = match serde_json::from_slice(&event.data) {
            Ok(v) => v,
            Err(err) => {
                return Ok(Some(vec![Event::Error {
                    message: format!("Invalid JSON chunk: {}", err),
                }]));
            }
        };
        let t = match json.get("type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return Ok(Some(vec![Event::Error {
                    message: "Invalid chunk: missing type".into(),
                }]));
            }
        };
        let mut events = Vec::new();
        if t.starts_with("response.output_text.") {
            self.handle_output_text_event(t, &json, &mut events);
        } else if t.starts_with("response.output_item.") {
            self.handle_output_item_event(t, &json, &mut events);
        } else if t.starts_with("response.function_call_arguments.") {
            self.handle_function_call_arguments_event(t, &json, &mut events);
        } else if t.starts_with("response.code_interpreter_call_code.") {
            self.handle_code_interpreter_event(t, &json, &mut events);
        } else if t.starts_with("response.apply_patch_call_operation_diff.") {
            self.handle_apply_patch_event(t, &json, &mut events);
        } else if t.starts_with("response.reasoning_summary_") {
            self.handle_reasoning_summary_event(t, &json, &mut events);
        } else {
            match t {
                "response.created" => self.handle_response_created(&json, &mut events),
                "response.image_generation_call.partial_image" => {
                    self.handle_image_generation_partial(&json, &mut events);
                }
                "response.completed" | "response.incomplete" | "response.failed" | "error" => {
                    self.handle_terminal_event(t, &json, &mut events);
                }
                _ => {}
            }
        }
        if events.is_empty() {
            Ok(None)
        } else {
            Ok(Some(events))
        }
    }
}

pub(super) fn openai_item_metadata(
    item_id: &str,
    extras: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> v2t::ProviderMetadata {
    let mut inner = HashMap::new();
    inner.insert("itemId".into(), serde_json::json!(item_id));
    for (key, value) in extras {
        inner.insert(key, value);
    }
    let mut outer = HashMap::new();
    outer.insert("openai".into(), inner);
    outer
}

pub(super) fn map_finish_reason(hint: Option<&str>, has_function_calls: bool) -> v2t::FinishReason {
    match hint {
        None => {
            if has_function_calls {
                v2t::FinishReason::ToolCalls
            } else {
                v2t::FinishReason::Stop
            }
        }
        Some("max_output_tokens") => v2t::FinishReason::Length,
        Some("content_filter") => v2t::FinishReason::ContentFilter,
        Some(_) => {
            if has_function_calls {
                v2t::FinishReason::ToolCalls
            } else {
                v2t::FinishReason::Other
            }
        }
    }
}

struct ResponseContentAccumulator {
    content: Vec<v2t::Content>,
    has_function_calls: bool,
    approval_request_id_map: HashMap<String, String>,
}

fn extract_response_content(
    json: &serde_json::Value,
    tool_name_mapping: &ToolNameMapping,
    approval_request_id_map: &HashMap<String, String>,
) -> (Vec<v2t::Content>, bool) {
    let mut state = ResponseContentAccumulator {
        content: Vec::new(),
        has_function_calls: false,
        approval_request_id_map: approval_request_id_map.clone(),
    };
    let output = match json.get("output").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return (state.content, false),
    };

    for item in output {
        let item_obj = match item.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        extract_response_output_item(item_obj, tool_name_mapping, &mut state);
    }

    (state.content, state.has_function_calls)
}

fn extract_response_output_item(
    item: &Map<String, Value>,
    tool_name_mapping: &ToolNameMapping,
    state: &mut ResponseContentAccumulator,
) {
    match item
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("")
    {
        "message" => push_response_message_content(item, &mut state.content),
        "function_call" => push_response_function_call(item, state),
        _ => push_response_provider_tool_content(item, tool_name_mapping, state),
    }
}

fn push_response_message_content(item: &Map<String, Value>, content: &mut Vec<v2t::Content>) {
    let Some(parts) = item.get("content").and_then(|value| value.as_array()) else {
        return;
    };

    let mut text_acc = String::new();
    for part in parts {
        if part.get("type").and_then(|value| value.as_str()) != Some("output_text") {
            continue;
        }
        if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
            text_acc.push_str(text);
        }
    }

    if !text_acc.is_empty() {
        content.push(v2t::Content::Text {
            text: text_acc,
            provider_metadata: None,
        });
    }
}

fn push_response_function_call(item: &Map<String, Value>, state: &mut ResponseContentAccumulator) {
    let (Some(call_id), Some(name)) = (
        item.get("call_id").and_then(|value| value.as_str()),
        item.get("name").and_then(|value| value.as_str()),
    ) else {
        return;
    };

    let args = item
        .get("arguments")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .to_string();
    let provider_metadata = item
        .get("id")
        .and_then(|value| value.as_str())
        .map(|id| openai_item_metadata(id, []));
    state
        .content
        .push(v2t::Content::ToolCall(v2t::ToolCallPart {
            tool_call_id: call_id.to_string(),
            tool_name: name.to_string(),
            input: args,
            provider_executed: false,
            provider_metadata,
            dynamic: false,
            provider_options: None,
        }));
    state.has_function_calls = true;
}

fn push_response_provider_tool_content(
    item: &Map<String, Value>,
    tool_name_mapping: &ToolNameMapping,
    state: &mut ResponseContentAccumulator,
) {
    let Some(tool_data) = provider_tool_data_from_output_item(item) else {
        return;
    };
    let Some(mut parts) = provider_tool_parts_from_data(&tool_data, tool_name_mapping) else {
        return;
    };

    if push_response_tool_approval_request(&parts, state) {
        return;
    }

    remap_response_tool_call_id(&mut parts, &state.approval_request_id_map);
    let (tool_call_metadata, tool_result_metadata) = response_provider_tool_metadata(&parts);
    state
        .content
        .push(v2t::Content::ToolCall(v2t::ToolCallPart {
            tool_call_id: parts.tool_call_id.clone(),
            tool_name: parts.tool_name.clone(),
            input: parts.input,
            provider_executed: parts.provider_executed,
            provider_metadata: tool_call_metadata,
            dynamic: parts.dynamic,
            provider_options: None,
        }));
    if let Some(result) = parts.result {
        state.content.push(v2t::Content::ToolResult {
            tool_call_id: parts.tool_call_id,
            tool_name: parts.tool_name,
            result,
            is_error: parts.is_error,
            provider_metadata: tool_result_metadata,
        });
    }
}

fn push_response_tool_approval_request(
    parts: &ProviderToolParts,
    state: &mut ResponseContentAccumulator,
) -> bool {
    if !parts.is_approval_request {
        return false;
    }

    let approval_id = parts
        .approval_request_id
        .clone()
        .unwrap_or_else(|| parts.tool_call_id.clone());
    let tool_call_id = Uuid::new_v4().to_string();
    state
        .approval_request_id_map
        .insert(approval_id.clone(), tool_call_id.clone());
    state
        .content
        .push(v2t::Content::ToolCall(v2t::ToolCallPart {
            tool_call_id: tool_call_id.clone(),
            tool_name: parts.tool_name.clone(),
            input: parts.input.clone(),
            provider_executed: parts.provider_executed,
            provider_metadata: None,
            dynamic: parts.dynamic,
            provider_options: None,
        }));
    state.content.push(v2t::Content::ToolApprovalRequest {
        approval_id,
        tool_call_id,
        provider_metadata: None,
    });
    true
}

fn remap_response_tool_call_id(
    parts: &mut ProviderToolParts,
    approval_request_id_map: &HashMap<String, String>,
) {
    if let Some(approval_id) = parts.approval_request_id.as_ref() {
        if let Some(mapped) = approval_request_id_map.get(approval_id) {
            parts.tool_call_id = mapped.clone();
        }
    }
}

fn response_provider_tool_metadata(
    parts: &ProviderToolParts,
) -> (Option<v2t::ProviderMetadata>, Option<v2t::ProviderMetadata>) {
    let tool_call_metadata = match parts.tool_type.as_str() {
        "apply_patch" | "local_shell" | "shell" => parts.provider_metadata.clone(),
        _ => None,
    };
    let tool_result_metadata = if parts.tool_type == "mcp" {
        parts.provider_metadata.clone()
    } else {
        None
    };
    (tool_call_metadata, tool_result_metadata)
}

fn resolve_transport_selection(
    endpoint_path: &str,
    provider_options: &OpenAIProviderOptionsParsed,
) -> ResponseTransportSelection {
    let requested = provider_options.transport_mode.unwrap_or_else(|| {
        if should_use_codex_oauth_websocket_transport(endpoint_path) {
            ResponseTransportMode::Websocket
        } else {
            ResponseTransportMode::Http
        }
    });
    ResponseTransportSelection {
        requested,
        fallback_http: provider_options.transport_fallback_http,
    }
}

fn build_compaction_request_body(body: Value) -> Result<Value, SdkError> {
    let body_obj = body.as_object().ok_or_else(|| SdkError::InvalidArgument {
        message: "openai responses request body must be an object".into(),
    })?;

    let mut compact = Map::new();
    for key in [
        "model",
        "instructions",
        "input",
        "tools",
        "parallel_tool_calls",
        "reasoning",
        "text",
    ] {
        if let Some(value) = body_obj.get(key) {
            compact.insert(key.to_string(), value.clone());
        }
    }

    if !compact.contains_key("model") || !compact.contains_key("input") {
        return Err(SdkError::InvalidArgument {
            message: "openai compact request body requires model and input".into(),
        });
    }

    Ok(Value::Object(compact))
}

impl<T: HttpTransport> OpenAIResponsesLanguageModel<T> {
    async fn stream_with_body(
        &self,
        body: Value,
        include_raw: bool,
        transport: ResponseTransportSelection,
        extra_headers: &HashMap<String, String>,
    ) -> Result<(EventStream, v2t::Headers), SdkError> {
        // Build headers for logging
        let (bytes, response_headers) = match self.send(body, transport, extra_headers).await {
            Ok(ok) => ok,
            Err(e) => {
                return Err(e);
            }
        };

        let pipeline = PipelineBuilder::<OpenAIResponsesChunk>::new()
            .with_provider("openai_official")
            .include_raw(include_raw)
            .build(bytes);
        Ok((Box::pin(pipeline), response_headers))
    }
}
