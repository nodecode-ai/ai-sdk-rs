use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_core::transport::{
    HttpTransport, JsonStreamWebsocketConnection, TransportConfig,
};
use crate::ai_sdk_core::{
    map_events_to_parts, EventMapperConfig, EventMapperHooks, EventMapperState, GenerateResponse,
    LanguageModel, LanguageModelTurnSession, StreamResponse,
};
use crate::ai_sdk_streaming_sse::{PipelineBuilder, ProviderChunk, SseEvent};
use crate::ai_sdk_types::v2 as v2t;
use crate::ai_sdk_types::{Event, TokenUsage};
use futures_core::Stream;
use futures_util::{stream, StreamExt};
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use serde_json::{json, Map, Value};
use tokio::task::JoinHandle;
use url::Url;
use uuid::Uuid;

use super::provider_tools::{
    build_tool_name_mapping, provider_tool_data_from_output_item, provider_tool_parts_from_data,
    ToolNameMapping,
};
use super::request_translation::{
    build_request_body, parse_openai_provider_options, OpenAIProviderOptionsParsed,
};
use crate::provider_openai::config::OpenAIConfig;
use crate::provider_openai::error::map_transport_error;

type EventStream = Pin<Box<dyn Stream<Item = Result<Event, SdkError>> + Send>>;
type ByteStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, SdkError>> + Send>>;
type RawByteStream = Pin<Box<dyn Stream<Item = Result<bytes::Bytes, TransportError>> + Send>>;

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

        let primary = self.send_once(&hdrs, &body, requested).await;
        match primary {
            Ok((stream, res_headers)) => {
                if requested == ResponseTransportMode::Websocket {
                    match prefetch_stream(stream).await {
                        Ok(stream) => Ok((
                            stream,
                            response_headers_with_transport(
                                res_headers,
                                requested,
                                requested,
                                None,
                            ),
                        )),
                        Err(err)
                            if transport.fallback_http
                                && should_fallback_to_http_after_websocket_error(&err) =>
                        {
                            let (stream, res_headers) = self
                                .send_once(&hdrs, &body, ResponseTransportMode::Http)
                                .await?;
                            Ok((
                                map_transport_stream(stream),
                                response_headers_with_transport(
                                    res_headers,
                                    requested,
                                    ResponseTransportMode::Http,
                                    Some(requested),
                                ),
                            ))
                        }
                        Err(err) => Err(err),
                    }
                } else {
                    Ok((
                        map_transport_stream(stream),
                        response_headers_with_transport(res_headers, requested, requested, None),
                    ))
                }
            }
            Err(err)
                if requested == ResponseTransportMode::Websocket
                    && transport.fallback_http
                    && should_fallback_to_http_after_websocket_error(&err) =>
            {
                let (stream, res_headers) = self
                    .send_once(&hdrs, &body, ResponseTransportMode::Http)
                    .await?;
                Ok((
                    map_transport_stream(stream),
                    response_headers_with_transport(
                        res_headers,
                        requested,
                        ResponseTransportMode::Http,
                        Some(requested),
                    ),
                ))
            }
            Err(err) => Err(err),
        }
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
            if reset_reason.is_none() {
                if let Some(object) = body.as_object_mut() {
                    object.insert(
                        "previous_response_id".to_string(),
                        Value::String(explicit_previous_response_id),
                    );
                    previous_response_id_used = true;
                }
            } else {
                tracing::info!(
                    reason = ?reset_reason,
                    "ignoring explicit previous_response_id after provider-session reset"
                );
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
            Err(err)
                if transport_selection.fallback_http
                    && should_fallback_to_http_after_websocket_error(&err) =>
            {
                self.activate_http_fallback("websocket_http_fallback");
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
            Err(err) => return Err(err),
        };
        let should_prewarm = self.should_prewarm_websocket(&body);
        let websocket_result = if should_prewarm {
            match self.send_session_websocket_request(body.clone()).await {
                Ok(result) => Ok(result),
                Err(SdkError::RateLimited { .. }) => {
                    match self.ensure_websocket_connection(&websocket_headers).await {
                        Ok(_) => {}
                        Err(err)
                            if transport_selection.fallback_http
                                && should_fallback_to_http_after_websocket_error(&err) =>
                        {
                            self.activate_http_fallback("websocket_http_fallback");
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
                        Err(err) => return Err(err),
                    }
                    match self.prewarm_websocket_session(&body).await {
                        Ok(()) => self.send_session_websocket_request(body.clone()).await,
                        Err(err)
                            if transport_selection.fallback_http
                                && should_fallback_to_http_after_websocket_error(&err) =>
                        {
                            self.activate_http_fallback("websocket_http_fallback");
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
                        Err(err) => return Err(err),
                    }
                }
                Err(err)
                    if transport_selection.fallback_http
                        && should_fallback_to_http_after_websocket_error(&err) =>
                {
                    self.activate_http_fallback("websocket_http_fallback");
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
                Err(err) => return Err(err),
            }
        } else {
            self.send_session_websocket_request(body.clone()).await
        };
        let (
            session_body,
            transport_body,
            stream,
            transport_headers,
            previous_response_id_used,
            warmup_response_id_used,
        ) = match websocket_result {
            Ok(result) => result,
            Err(err)
                if transport_selection.fallback_http
                    && should_fallback_to_http_after_websocket_error(&err) =>
            {
                self.activate_http_fallback("websocket_http_fallback");
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
            Err(err) => return Err(err),
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
        // Use non-streaming JSON call to Responses API
        let url = self.endpoint_url();
        let hdrs = self.request_headers(&options.headers);
        let headers: Vec<(String, String)> = hdrs
            .into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect();
        let (json, _res_headers) = self
            .http
            .post_json(&url, &headers, &body, &self.transport_cfg)
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

        let approval_request_id_map = extract_approval_request_id_to_tool_call_id(
            &options.prompt,
            &self.config.provider_scope_name,
        );
        let (content, has_function_calls) =
            extract_response_content(&json, &tool_name_mapping, &approval_request_id_map);

        // Usage best-effort
        let mut usage = v2t::Usage::default();
        let usage_val = json
            .get("usage")
            .or_else(|| json.get("response").and_then(|r| r.get("usage")));
        if let Some(u) = usage_val.and_then(parse_openai_usage) {
            usage.input_tokens = Some(u.input_tokens as u64);
            usage.output_tokens = Some(u.output_tokens as u64);
            usage.total_tokens = Some(u.total_tokens as u64);
            if let Some(v) = u.cache_read_tokens {
                usage.cached_input_tokens = Some(v as u64);
            }
        }
        if let Some(raw_usage) = usage_val {
            apply_openai_usage_details(raw_usage, &mut usage);
        }

        // Finish reason mapping
        let finish_hint = json
            .get("incomplete_details")
            .and_then(|v| v.get("reason"))
            .and_then(|v| v.as_str());
        let finish_reason = map_finish_reason(finish_hint, has_function_calls);

        // Provider metadata: responseId and serviceTier
        let mut provider_metadata: Option<
            std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
        > = None;
        let resp_id = json.get("id").and_then(|v| v.as_str()).or_else(|| {
            json.get("response")
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
        });
        let tier = json
            .get("service_tier")
            .and_then(|v| v.as_str())
            .or_else(|| {
                json.get("response")
                    .and_then(|r| r.get("service_tier"))
                    .and_then(|v| v.as_str())
            });
        if resp_id.is_some() || tier.is_some() {
            let mut outer = std::collections::HashMap::new();
            let mut inner = std::collections::HashMap::new();
            if let Some(rid) = resp_id {
                inner.insert("responseId".into(), serde_json::json!(rid));
            }
            if let Some(st) = tier {
                inner.insert("serviceTier".into(), serde_json::json!(st));
            }
            outer.insert("openai".into(), inner);
            provider_metadata = Some(outer);
        }

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

#[allow(dead_code)]
mod legacy_provider_tools_request_side {
    use super::*;

    fn invalid_tool_args(tool: &v2t::ProviderTool, message: impl Into<String>) -> SdkError {
        SdkError::InvalidArgument {
            message: format!(
                "provider tool {} ({}): {}",
                tool.name,
                tool.id,
                message.into()
            ),
        }
    }

    fn require_args_object(tool: &v2t::ProviderTool) -> Result<&Map<String, Value>, SdkError> {
        tool.args
            .as_object()
            .ok_or_else(|| invalid_tool_args(tool, "args must be an object"))
    }

    fn require_field<'a>(
        tool: &v2t::ProviderTool,
        args: &'a Map<String, Value>,
        key: &str,
    ) -> Result<&'a Value, SdkError> {
        args.get(key)
            .ok_or_else(|| invalid_tool_args(tool, format!("args.{key} is required")))
    }

    fn expect_string(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
        if value.as_str().is_some() {
            Ok(())
        } else {
            Err(invalid_tool_args(tool, format!("{path} must be a string")))
        }
    }

    fn expect_bool(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
        if value.as_bool().is_some() {
            Ok(())
        } else {
            Err(invalid_tool_args(tool, format!("{path} must be a boolean")))
        }
    }

    fn expect_number(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
        if value.as_f64().is_some() {
            Ok(())
        } else {
            Err(invalid_tool_args(tool, format!("{path} must be a number")))
        }
    }

    fn expect_int_range(
        tool: &v2t::ProviderTool,
        value: &Value,
        path: &str,
        min: i64,
        max: i64,
    ) -> Result<(), SdkError> {
        let raw = value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
            .or_else(|| {
                value.as_f64().filter(|v| v.fract() == 0.0).and_then(|v| {
                    if v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                        Some(v as i64)
                    } else {
                        None
                    }
                })
            })
            .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an integer")))?;
        if raw < min || raw > max {
            return Err(invalid_tool_args(
                tool,
                format!("{path} must be between {min} and {max}"),
            ));
        }
        Ok(())
    }

    fn expect_enum(
        tool: &v2t::ProviderTool,
        value: &Value,
        path: &str,
        allowed: &[&str],
    ) -> Result<(), SdkError> {
        match value.as_str() {
            Some(val) if allowed.contains(&val) => Ok(()),
            Some(_) => Err(invalid_tool_args(
                tool,
                format!("{path} must be one of {}", allowed.join(", ")),
            )),
            None => Err(invalid_tool_args(
                tool,
                format!("{path} must be one of {}", allowed.join(", ")),
            )),
        }
    }

    fn expect_string_array(
        tool: &v2t::ProviderTool,
        value: &Value,
        path: &str,
    ) -> Result<(), SdkError> {
        let arr = value
            .as_array()
            .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an array")))?;
        if arr.iter().all(|item| item.as_str().is_some()) {
            Ok(())
        } else {
            Err(invalid_tool_args(
                tool,
                format!("{path} must be an array of strings"),
            ))
        }
    }

    fn expect_object<'a>(
        tool: &v2t::ProviderTool,
        value: &'a Value,
        path: &str,
    ) -> Result<&'a Map<String, Value>, SdkError> {
        value
            .as_object()
            .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an object")))
    }

    fn ensure_known_keys(
        tool: &v2t::ProviderTool,
        args: &Map<String, Value>,
        allowed: &[&str],
    ) -> Result<(), SdkError> {
        let mut unknown = args
            .keys()
            .filter(|key| !allowed.contains(&key.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if unknown.is_empty() {
            return Ok(());
        }
        unknown.sort();
        Err(invalid_tool_args(
            tool,
            format!("args contains unsupported keys: {}", unknown.join(", ")),
        ))
    }

    fn validate_user_location(
        tool: &v2t::ProviderTool,
        value: &Value,
        path: &str,
    ) -> Result<(), SdkError> {
        let obj = expect_object(tool, value, path)?;
        let loc_type = obj.get("type").ok_or_else(|| {
            invalid_tool_args(tool, format!("{path}.type must be \"approximate\""))
        })?;
        expect_enum(tool, loc_type, &format!("{path}.type"), &["approximate"])?;
        for key in ["country", "city", "region", "timezone"] {
            if let Some(val) = obj.get(key) {
                expect_string(tool, val, &format!("{path}.{key}"))?;
            }
        }
        Ok(())
    }

    fn validate_file_search_filter(
        tool: &v2t::ProviderTool,
        value: &Value,
        path: &str,
    ) -> Result<(), SdkError> {
        let obj = expect_object(tool, value, path)?;
        let filter_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| invalid_tool_args(tool, format!("{path}.type is required")))?;
        match filter_type {
            "and" | "or" => {
                let filters = obj.get("filters").ok_or_else(|| {
                    invalid_tool_args(tool, format!("{path}.filters is required"))
                })?;
                let arr = filters.as_array().ok_or_else(|| {
                    invalid_tool_args(tool, format!("{path}.filters must be an array"))
                })?;
                for (idx, entry) in arr.iter().enumerate() {
                    validate_file_search_filter(tool, entry, &format!("{path}.filters[{idx}]"))?;
                }
                Ok(())
            }
            "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "nin" => {
                let key = obj
                    .get("key")
                    .ok_or_else(|| invalid_tool_args(tool, format!("{path}.key is required")))?;
                expect_string(tool, key, &format!("{path}.key"))?;
                let val = obj
                    .get("value")
                    .ok_or_else(|| invalid_tool_args(tool, format!("{path}.value is required")))?;
                if val.as_str().is_some() || val.as_bool().is_some() || val.as_f64().is_some() {
                    return Ok(());
                }
                if let Some(arr) = val.as_array() {
                    if arr.iter().all(|item| item.as_str().is_some()) {
                        return Ok(());
                    }
                }
                Err(invalid_tool_args(
                    tool,
                    format!("{path}.value must be a string, number, boolean, or array of strings"),
                ))
            }
            _ => Err(invalid_tool_args(
                tool,
                format!("{path}.type has an invalid value"),
            )),
        }
    }

    pub(super) fn validate_openai_provider_tool_args(
        tool_type: &str,
        tool: &v2t::ProviderTool,
    ) -> Result<(), SdkError> {
        match tool_type {
            "file_search" => {
                let args = require_args_object(tool)?;
                let ids = require_field(tool, args, "vectorStoreIds")?;
                expect_string_array(tool, ids, "args.vectorStoreIds")?;
                if let Some(max) = args.get("maxNumResults") {
                    expect_number(tool, max, "args.maxNumResults")?;
                }
                if let Some(rank) = args.get("ranking") {
                    let rank_obj = expect_object(tool, rank, "args.ranking")?;
                    if let Some(ranker) = rank_obj.get("ranker") {
                        expect_string(tool, ranker, "args.ranking.ranker")?;
                    }
                    if let Some(score) = rank_obj.get("scoreThreshold") {
                        expect_number(tool, score, "args.ranking.scoreThreshold")?;
                    }
                }
                if let Some(filters) = args.get("filters") {
                    validate_file_search_filter(tool, filters, "args.filters")?;
                }
                Ok(())
            }
            "web_search_preview" => {
                let args = require_args_object(tool)?;
                if let Some(size) = args.get("searchContextSize") {
                    expect_enum(
                        tool,
                        size,
                        "args.searchContextSize",
                        &["low", "medium", "high"],
                    )?;
                }
                if let Some(loc) = args.get("userLocation") {
                    validate_user_location(tool, loc, "args.userLocation")?;
                }
                Ok(())
            }
            "web_search" => {
                let args = require_args_object(tool)?;
                if let Some(access) = args.get("externalWebAccess") {
                    expect_bool(tool, access, "args.externalWebAccess")?;
                }
                if let Some(filters) = args.get("filters") {
                    let obj = expect_object(tool, filters, "args.filters")?;
                    if let Some(domains) = obj.get("allowedDomains") {
                        expect_string_array(tool, domains, "args.filters.allowedDomains")?;
                    }
                }
                if let Some(size) = args.get("searchContextSize") {
                    expect_enum(
                        tool,
                        size,
                        "args.searchContextSize",
                        &["low", "medium", "high"],
                    )?;
                }
                if let Some(loc) = args.get("userLocation") {
                    validate_user_location(tool, loc, "args.userLocation")?;
                }
                Ok(())
            }
            "code_interpreter" => {
                let args = require_args_object(tool)?;
                if let Some(container) = args.get("container") {
                    if let Some(obj) = container.as_object() {
                        if let Some(file_ids) = obj.get("fileIds") {
                            expect_string_array(tool, file_ids, "args.container.fileIds")?;
                        }
                    } else if container.as_str().is_none() {
                        return Err(invalid_tool_args(
                            tool,
                            "args.container must be a string or object",
                        ));
                    }
                }
                Ok(())
            }
            "image_generation" => {
                let args = require_args_object(tool)?;
                ensure_known_keys(
                    tool,
                    args,
                    &[
                        "background",
                        "inputFidelity",
                        "inputImageMask",
                        "model",
                        "moderation",
                        "outputCompression",
                        "outputFormat",
                        "partialImages",
                        "quality",
                        "size",
                    ],
                )?;
                if let Some(background) = args.get("background") {
                    expect_enum(
                        tool,
                        background,
                        "args.background",
                        &["auto", "opaque", "transparent"],
                    )?;
                }
                if let Some(fidelity) = args.get("inputFidelity") {
                    expect_enum(tool, fidelity, "args.inputFidelity", &["low", "high"])?;
                }
                if let Some(mask) = args.get("inputImageMask") {
                    let mask_obj = expect_object(tool, mask, "args.inputImageMask")?;
                    if let Some(file_id) = mask_obj.get("fileId") {
                        expect_string(tool, file_id, "args.inputImageMask.fileId")?;
                    }
                    if let Some(image_url) = mask_obj.get("imageUrl") {
                        expect_string(tool, image_url, "args.inputImageMask.imageUrl")?;
                    }
                }
                if let Some(model) = args.get("model") {
                    expect_string(tool, model, "args.model")?;
                }
                if let Some(moderation) = args.get("moderation") {
                    expect_enum(tool, moderation, "args.moderation", &["auto"])?;
                }
                if let Some(output_compression) = args.get("outputCompression") {
                    expect_int_range(tool, output_compression, "args.outputCompression", 0, 100)?;
                }
                if let Some(output_format) = args.get("outputFormat") {
                    expect_enum(
                        tool,
                        output_format,
                        "args.outputFormat",
                        &["png", "jpeg", "webp"],
                    )?;
                }
                if let Some(partial_images) = args.get("partialImages") {
                    expect_int_range(tool, partial_images, "args.partialImages", 0, 3)?;
                }
                if let Some(quality) = args.get("quality") {
                    expect_enum(
                        tool,
                        quality,
                        "args.quality",
                        &["auto", "low", "medium", "high"],
                    )?;
                }
                if let Some(size) = args.get("size") {
                    expect_enum(
                        tool,
                        size,
                        "args.size",
                        &["1024x1024", "1024x1536", "1536x1024", "auto"],
                    )?;
                }
                Ok(())
            }
            "mcp" => {
                let args = require_args_object(tool)?;
                let server_label = require_field(tool, args, "serverLabel")?;
                expect_string(tool, server_label, "args.serverLabel")?;
                let server_url = args.get("serverUrl");
                let connector_id = args.get("connectorId");
                if let Some(url) = server_url {
                    expect_string(tool, url, "args.serverUrl")?;
                }
                if let Some(connector) = connector_id {
                    expect_string(tool, connector, "args.connectorId")?;
                }
                if server_url.is_none() && connector_id.is_none() {
                    return Err(invalid_tool_args(
                        tool,
                        "args.serverUrl or args.connectorId is required",
                    ));
                }
                if let Some(allowed) = args.get("allowedTools") {
                    if let Some(arr) = allowed.as_array() {
                        for (idx, entry) in arr.iter().enumerate() {
                            expect_string(tool, entry, &format!("args.allowedTools[{idx}]"))?;
                        }
                    } else if let Some(obj) = allowed.as_object() {
                        if let Some(read_only) = obj.get("readOnly") {
                            expect_bool(tool, read_only, "args.allowedTools.readOnly")?;
                        }
                        if let Some(tool_names) = obj.get("toolNames") {
                            expect_string_array(tool, tool_names, "args.allowedTools.toolNames")?;
                        }
                    } else {
                        return Err(invalid_tool_args(
                            tool,
                            "args.allowedTools must be an array or object",
                        ));
                    }
                }
                if let Some(authorization) = args.get("authorization") {
                    expect_string(tool, authorization, "args.authorization")?;
                }
                if let Some(headers) = args.get("headers") {
                    let obj = expect_object(tool, headers, "args.headers")?;
                    for (key, val) in obj {
                        expect_string(tool, val, &format!("args.headers.{key}"))?;
                    }
                }
                if let Some(require_approval) = args.get("requireApproval") {
                    if let Some(val) = require_approval.as_str() {
                        if val != "always" && val != "never" {
                            return Err(invalid_tool_args(
                                tool,
                                "args.requireApproval must be \"always\" or \"never\"",
                            ));
                        }
                    } else if let Some(obj) = require_approval.as_object() {
                        if let Some(never) = obj.get("never") {
                            let never_obj =
                                expect_object(tool, never, "args.requireApproval.never")?;
                            if let Some(tool_names) = never_obj.get("toolNames") {
                                expect_string_array(
                                    tool,
                                    tool_names,
                                    "args.requireApproval.never.toolNames",
                                )?;
                            }
                        }
                    } else {
                        return Err(invalid_tool_args(
                            tool,
                            "args.requireApproval must be a string or object",
                        ));
                    }
                }
                if let Some(server_description) = args.get("serverDescription") {
                    expect_string(tool, server_description, "args.serverDescription")?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    #[derive(Clone, Default)]
    pub(super) struct ToolNameMapping {
        custom_to_provider: HashMap<String, String>,
        provider_to_custom: HashMap<String, String>,
        web_search_tool_name: Option<String>,
    }

    impl ToolNameMapping {
        pub(super) fn to_provider_tool_name<'a>(&'a self, custom_tool_name: &'a str) -> &'a str {
            self.custom_to_provider
                .get(custom_tool_name)
                .map(|s| s.as_str())
                .unwrap_or(custom_tool_name)
        }

        pub(super) fn to_custom_tool_name<'a>(&'a self, provider_tool_name: &'a str) -> &'a str {
            self.provider_to_custom
                .get(provider_tool_name)
                .map(|s| s.as_str())
                .unwrap_or(provider_tool_name)
        }
    }

    pub(super) fn openai_provider_tool_name(id: &str) -> Option<&'static str> {
        match id {
            "openai.file_search" => Some("file_search"),
            "openai.local_shell" => Some("local_shell"),
            "openai.shell" => Some("shell"),
            "openai.apply_patch" => Some("apply_patch"),
            "openai.web_search_preview" => Some("web_search_preview"),
            "openai.web_search" => Some("web_search"),
            "openai.code_interpreter" => Some("code_interpreter"),
            "openai.image_generation" => Some("image_generation"),
            "openai.mcp" => Some("mcp"),
            _ => None,
        }
    }

    pub(super) fn build_tool_name_mapping(tools: &[v2t::Tool]) -> ToolNameMapping {
        let mut mapping = ToolNameMapping::default();
        for tool in tools {
            if let v2t::Tool::Provider(provider_tool) = tool {
                if let Some(provider_name) = openai_provider_tool_name(&provider_tool.id) {
                    mapping
                        .custom_to_provider
                        .insert(provider_tool.name.clone(), provider_name.to_string());
                    mapping
                        .provider_to_custom
                        .insert(provider_name.to_string(), provider_tool.name.clone());
                    if matches!(
                        provider_tool.id.as_str(),
                        "openai.web_search" | "openai.web_search_preview"
                    ) && mapping.web_search_tool_name.is_none()
                    {
                        mapping.web_search_tool_name = Some(provider_tool.name.clone());
                    }
                }
            }
        }
        mapping
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

#[allow(dead_code)]
mod legacy_provider_tools_output_side {
    use super::legacy_provider_tools_request_side::{
        openai_provider_tool_name, validate_openai_provider_tool_args,
    };
    use super::*;

    fn map_web_search_output(action: &serde_json::Value) -> Option<serde_json::Value> {
        let obj = action.as_object()?;
        let action_type = obj.get("type")?.as_str()?;
        match action_type {
            "search" => {
                let mut out = serde_json::Map::new();
                let mut inner = serde_json::Map::new();
                inner.insert("type".into(), serde_json::Value::String("search".into()));
                if let Some(query) = obj.get("query") {
                    if !query.is_null() {
                        inner.insert("query".into(), query.clone());
                    }
                }
                out.insert("action".into(), serde_json::Value::Object(inner));
                if let Some(sources) = obj.get("sources") {
                    if !sources.is_null() {
                        out.insert("sources".into(), sources.clone());
                    }
                }
                Some(serde_json::Value::Object(out))
            }
            "open_page" => Some(json!({
                "action": {
                    "type": "openPage",
                    "url": obj.get("url").cloned().unwrap_or(serde_json::Value::Null),
                }
            })),
            "find_in_page" => Some(json!({
                "action": {
                    "type": "findInPage",
                    "url": obj.get("url").cloned().unwrap_or(serde_json::Value::Null),
                    "pattern": obj.get("pattern").cloned().unwrap_or(serde_json::Value::Null),
                }
            })),
            _ => None,
        }
    }

    pub(super) fn build_openai_provider_tool(
        tool: &v2t::ProviderTool,
    ) -> Result<Option<serde_json::Value>, SdkError> {
        let empty = serde_json::Map::new();
        let tool_type = match openai_provider_tool_name(&tool.id) {
            Some(tool_type) => tool_type,
            None => return Ok(None),
        };
        validate_openai_provider_tool_args(tool_type, tool)?;
        let args = tool.args.as_object().unwrap_or(&empty);
        let val = match tool_type {
            "file_search" => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("file_search"));
                if let Some(ids) = args.get("vectorStoreIds") {
                    obj.insert("vector_store_ids".into(), ids.clone());
                }
                if let Some(max) = args.get("maxNumResults") {
                    obj.insert("max_num_results".into(), max.clone());
                }
                if let Some(rank) = args.get("ranking").and_then(|v| v.as_object()) {
                    let mut opts = serde_json::Map::new();
                    if let Some(ranker) = rank.get("ranker") {
                        opts.insert("ranker".into(), ranker.clone());
                    }
                    if let Some(score) = rank.get("scoreThreshold") {
                        opts.insert("score_threshold".into(), score.clone());
                    }
                    if !opts.is_empty() {
                        obj.insert("ranking_options".into(), serde_json::Value::Object(opts));
                    }
                }
                if let Some(filters) = args.get("filters") {
                    obj.insert("filters".into(), filters.clone());
                }
                Some(serde_json::Value::Object(obj))
            }
            "local_shell" => Some(json!({"type":"local_shell"})),
            "shell" => Some(json!({"type":"shell"})),
            "apply_patch" => Some(json!({"type":"apply_patch"})),
            "web_search_preview" => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("web_search_preview"));
                if let Some(size) = args.get("searchContextSize") {
                    obj.insert("search_context_size".into(), size.clone());
                }
                if let Some(loc) = args.get("userLocation") {
                    obj.insert("user_location".into(), loc.clone());
                }
                Some(serde_json::Value::Object(obj))
            }
            "web_search" => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("web_search"));
                if let Some(filters) = args.get("filters").and_then(|v| v.as_object()) {
                    if let Some(allowed_domains) = filters.get("allowedDomains") {
                        obj.insert(
                            "filters".into(),
                            json!({"allowed_domains": allowed_domains}),
                        );
                    }
                }
                if let Some(access) = args.get("externalWebAccess") {
                    obj.insert("external_web_access".into(), access.clone());
                }
                if let Some(size) = args.get("searchContextSize") {
                    obj.insert("search_context_size".into(), size.clone());
                }
                if let Some(loc) = args.get("userLocation") {
                    obj.insert("user_location".into(), loc.clone());
                }
                Some(serde_json::Value::Object(obj))
            }
            "code_interpreter" => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("code_interpreter"));
                let container = match args.get("container") {
                    None | Some(serde_json::Value::Null) => json!({"type":"auto"}),
                    Some(serde_json::Value::String(val)) => json!(val),
                    Some(serde_json::Value::Object(map)) => {
                        let mut c = serde_json::Map::new();
                        c.insert("type".into(), json!("auto"));
                        if let Some(file_ids) = map.get("fileIds") {
                            c.insert("file_ids".into(), file_ids.clone());
                        }
                        serde_json::Value::Object(c)
                    }
                    Some(other) => other.clone(),
                };
                obj.insert("container".into(), container);
                Some(serde_json::Value::Object(obj))
            }
            "image_generation" => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("image_generation"));
                for (src, dst) in [
                    ("background", "background"),
                    ("inputFidelity", "input_fidelity"),
                    ("model", "model"),
                    ("moderation", "moderation"),
                    ("partialImages", "partial_images"),
                    ("quality", "quality"),
                    ("outputCompression", "output_compression"),
                    ("outputFormat", "output_format"),
                    ("size", "size"),
                ] {
                    if let Some(val) = args.get(src) {
                        obj.insert(dst.into(), val.clone());
                    }
                }
                if let Some(mask) = args.get("inputImageMask").and_then(|v| v.as_object()) {
                    let mut mask_obj = serde_json::Map::new();
                    if let Some(file_id) = mask.get("fileId") {
                        mask_obj.insert("file_id".into(), file_id.clone());
                    }
                    if let Some(image_url) = mask.get("imageUrl") {
                        mask_obj.insert("image_url".into(), image_url.clone());
                    }
                    if !mask_obj.is_empty() {
                        obj.insert(
                            "input_image_mask".into(),
                            serde_json::Value::Object(mask_obj),
                        );
                    }
                }
                Some(serde_json::Value::Object(obj))
            }
            "mcp" => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".into(), json!("mcp"));
                if let Some(val) = args.get("serverLabel") {
                    obj.insert("server_label".into(), val.clone());
                }
                if let Some(val) = args.get("authorization") {
                    obj.insert("authorization".into(), val.clone());
                }
                if let Some(val) = args.get("connectorId") {
                    obj.insert("connector_id".into(), val.clone());
                }
                if let Some(val) = args.get("headers") {
                    obj.insert("headers".into(), val.clone());
                }
                if let Some(val) = args.get("serverDescription") {
                    obj.insert("server_description".into(), val.clone());
                }
                if let Some(val) = args.get("serverUrl") {
                    obj.insert("server_url".into(), val.clone());
                }
                if let Some(allowed) = args.get("allowedTools") {
                    if let Some(list) = allowed.as_array() {
                        obj.insert(
                            "allowed_tools".into(),
                            serde_json::Value::Array(list.clone()),
                        );
                    } else if let Some(filter) = allowed.as_object() {
                        let mut allowed_obj = serde_json::Map::new();
                        if let Some(read_only) = filter.get("readOnly") {
                            allowed_obj.insert("read_only".into(), read_only.clone());
                        }
                        if let Some(tool_names) = filter.get("toolNames") {
                            allowed_obj.insert("tool_names".into(), tool_names.clone());
                        }
                        if !allowed_obj.is_empty() {
                            obj.insert(
                                "allowed_tools".into(),
                                serde_json::Value::Object(allowed_obj),
                            );
                        }
                    }
                }
                let require_approval = match args.get("requireApproval") {
                    None | Some(serde_json::Value::Null) => None,
                    Some(serde_json::Value::String(val)) => {
                        Some(serde_json::Value::String(val.clone()))
                    }
                    Some(serde_json::Value::Object(map)) => map.get("never").map(|never| {
                        if let Some(filter) = never.as_object() {
                            let mut filter_obj = serde_json::Map::new();
                            if let Some(tool_names) = filter.get("toolNames") {
                                filter_obj.insert("tool_names".into(), tool_names.clone());
                            }
                            json!({"never": filter_obj})
                        } else {
                            json!({"never": {}})
                        }
                    }),
                    Some(other) => Some(other.clone()),
                };
                obj.insert(
                    "require_approval".into(),
                    require_approval.unwrap_or_else(|| json!("never")),
                );
                Some(serde_json::Value::Object(obj))
            }
            _ => None,
        };
        Ok(val)
    }

    fn provider_tool_data_from_output_item(
        item: &serde_json::Map<String, Value>,
    ) -> Option<serde_json::Value> {
        let item_type = item.get("type")?.as_str()?;
        let item_id = item
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        match item_type {
            "web_search_call" => {
                let tool_call_id = item_id.clone()?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("web_search"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                obj.insert("input".into(), json!({}));
                if let Some(action) = item.get("action").and_then(map_web_search_output) {
                    obj.insert("result".into(), action);
                }
                Some(serde_json::Value::Object(obj))
            }
            "file_search_call" => {
                let tool_call_id = item_id.clone()?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("file_search"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                obj.insert("input".into(), json!({}));
                let results_val = item.get("results").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|entry| entry.as_object())
                        .map(|entry| {
                            let mut mapped = serde_json::Map::new();
                            if let Some(attributes) = entry.get("attributes") {
                                mapped.insert("attributes".into(), attributes.clone());
                            }
                            if let Some(file_id) = entry.get("file_id") {
                                mapped.insert("fileId".into(), file_id.clone());
                            }
                            if let Some(filename) = entry.get("filename") {
                                mapped.insert("filename".into(), filename.clone());
                            }
                            if let Some(score) = entry.get("score") {
                                mapped.insert("score".into(), score.clone());
                            }
                            if let Some(text) = entry.get("text") {
                                mapped.insert("text".into(), text.clone());
                            }
                            serde_json::Value::Object(mapped)
                        })
                        .collect::<Vec<_>>()
                });
                let result = json!({
                    "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                    "results": results_val.map(serde_json::Value::Array).unwrap_or(serde_json::Value::Null),
                });
                obj.insert("result".into(), result);
                Some(serde_json::Value::Object(obj))
            }
            "code_interpreter_call" => {
                let tool_call_id = item_id.clone()?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("code_interpreter"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                let input = json!({
                    "code": item.get("code").cloned().unwrap_or(serde_json::Value::Null),
                    "containerId": item.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                });
                obj.insert("input".into(), input);
                let result = json!({
                    "outputs": item.get("outputs").cloned().unwrap_or(serde_json::Value::Null),
                });
                obj.insert("result".into(), result);
                Some(serde_json::Value::Object(obj))
            }
            "image_generation_call" => {
                let tool_call_id = item_id.clone()?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("image_generation"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                obj.insert("input".into(), json!({}));
                let result = json!({
                    "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
                });
                obj.insert("result".into(), result);
                Some(serde_json::Value::Object(obj))
            }
            "computer_call" => {
                let tool_call_id = item_id.clone()?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("computer_use"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                obj.insert("input".into(), json!(""));
                let status = item
                    .get("status")
                    .cloned()
                    .unwrap_or_else(|| json!("completed"));
                obj.insert(
                    "result".into(),
                    json!({
                        "type": "computer_use_tool_result",
                        "status": status,
                    }),
                );
                Some(serde_json::Value::Object(obj))
            }
            "local_shell_call" => {
                let tool_call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| item_id.clone())?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("local_shell"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(false));
                let action = item.get("action").and_then(|v| v.as_object());
                let mut action_obj = serde_json::Map::new();
                if let Some(action) = action {
                    if let Some(command) = action.get("command") {
                        action_obj.insert("command".into(), command.clone());
                    }
                    if let Some(timeout) = action.get("timeout_ms") {
                        action_obj.insert("timeoutMs".into(), timeout.clone());
                    }
                    if let Some(user) = action.get("user") {
                        action_obj.insert("user".into(), user.clone());
                    }
                    if let Some(dir) = action.get("working_directory") {
                        action_obj.insert("workingDirectory".into(), dir.clone());
                    }
                    if let Some(env) = action.get("env") {
                        action_obj.insert("env".into(), env.clone());
                    }
                }
                if !action_obj.is_empty() {
                    obj.insert("input".into(), json!({ "action": action_obj }));
                } else {
                    obj.insert("input".into(), json!({}));
                }
                Some(serde_json::Value::Object(obj))
            }
            "shell_call" => {
                let tool_call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| item_id.clone())?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("shell"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(false));
                let action = item.get("action").and_then(|v| v.as_object());
                let mut action_obj = serde_json::Map::new();
                if let Some(action) = action {
                    if let Some(commands) = action.get("commands") {
                        action_obj.insert("commands".into(), commands.clone());
                    }
                    if let Some(timeout) = action.get("timeout_ms") {
                        action_obj.insert("timeoutMs".into(), timeout.clone());
                    }
                    if let Some(max_len) = action.get("max_output_length") {
                        action_obj.insert("maxOutputLength".into(), max_len.clone());
                    }
                }
                if !action_obj.is_empty() {
                    obj.insert("input".into(), json!({ "action": action_obj }));
                } else {
                    obj.insert("input".into(), json!({}));
                }
                Some(serde_json::Value::Object(obj))
            }
            "apply_patch_call" => {
                let tool_call_id = item
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .or_else(|| item_id.clone())?;
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("apply_patch"));
                obj.insert("tool_call_id".into(), json!(tool_call_id.clone()));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(false));
                let input = json!({
                    "callId": tool_call_id,
                    "operation": item.get("operation").cloned().unwrap_or(serde_json::Value::Null),
                });
                obj.insert("input".into(), input);
                Some(serde_json::Value::Object(obj))
            }
            "mcp_call" => {
                let tool_call_id = item_id.clone()?;
                let name = item.get("name").and_then(|v| v.as_str())?.to_string();
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("mcp"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                obj.insert(
                    "input".into(),
                    item.get("arguments").cloned().unwrap_or_else(|| json!("")),
                );
                obj.insert("mcp_name".into(), json!(name.clone()));
                if let Some(approval_request_id) =
                    item.get("approval_request_id").and_then(|v| v.as_str())
                {
                    obj.insert("approval_request_id".into(), json!(approval_request_id));
                }
                if let Some(server_label) = item.get("server_label") {
                    obj.insert("server_label".into(), server_label.clone());
                }
                let mut result = serde_json::Map::new();
                result.insert("type".into(), json!("call"));
                if let Some(server_label) = item.get("server_label") {
                    result.insert("serverLabel".into(), server_label.clone());
                }
                result.insert("name".into(), json!(name));
                result.insert(
                    "arguments".into(),
                    item.get("arguments").cloned().unwrap_or_else(|| json!("")),
                );
                if let Some(output) = item.get("output") {
                    result.insert("output".into(), output.clone());
                }
                if let Some(error) = item.get("error") {
                    result.insert("error".into(), error.clone());
                }
                obj.insert("result".into(), serde_json::Value::Object(result));
                Some(serde_json::Value::Object(obj))
            }
            "mcp_approval_request" => {
                let tool_call_id = item_id.clone()?;
                let name = item.get("name").and_then(|v| v.as_str())?.to_string();
                let mut obj = serde_json::Map::new();
                obj.insert("tool_type".into(), json!("mcp"));
                obj.insert("tool_call_id".into(), json!(tool_call_id));
                if let Some(id) = item_id.as_ref() {
                    obj.insert("item_id".into(), json!(id));
                }
                obj.insert("provider_executed".into(), json!(true));
                obj.insert(
                    "input".into(),
                    item.get("arguments").cloned().unwrap_or_else(|| json!("")),
                );
                obj.insert("mcp_name".into(), json!(name));
                let approval_request_id = item
                    .get("approval_request_id")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string())
                    .or_else(|| item_id.clone());
                if let Some(approval_request_id) = approval_request_id {
                    obj.insert("approval_request_id".into(), json!(approval_request_id));
                }
                obj.insert("approval_request".into(), json!(true));
                if let Some(server_label) = item.get("server_label") {
                    obj.insert("server_label".into(), server_label.clone());
                }
                Some(serde_json::Value::Object(obj))
            }
            _ => None,
        }
    }

    struct ProviderToolParts {
        tool_call_id: String,
        tool_name: String,
        tool_type: String,
        input: String,
        provider_executed: bool,
        dynamic: bool,
        result: Option<serde_json::Value>,
        is_error: bool,
        provider_metadata: Option<v2t::ProviderMetadata>,
        approval_request_id: Option<String>,
        is_approval_request: bool,
    }

    fn provider_tool_parts_from_data(
        data: &serde_json::Value,
        tool_name_mapping: &ToolNameMapping,
    ) -> Option<ProviderToolParts> {
        let obj = data.as_object()?;
        let tool_type = obj.get("tool_type").and_then(|v| v.as_str()).unwrap_or("");
        let tool_call_id = obj
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if tool_call_id.is_empty() {
            return None;
        }
        let input_val = obj.get("input").cloned().unwrap_or(serde_json::Value::Null);
        let input = match input_val {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        };
        let dynamic = tool_type == "mcp";
        let tool_name = if tool_type == "mcp" {
            obj.get("mcp_name")
                .and_then(|v| v.as_str())
                .map(|name| format!("mcp.{name}"))
                .unwrap_or_else(|| "mcp".into())
        } else if tool_type == "web_search" {
            tool_name_mapping
                .web_search_tool_name
                .clone()
                .unwrap_or_else(|| {
                    tool_name_mapping
                        .to_custom_tool_name("web_search")
                        .to_string()
                })
        } else {
            tool_name_mapping.to_custom_tool_name(tool_type).to_string()
        };
        let provider_executed = obj
            .get("provider_executed")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let result = obj.get("result").cloned().filter(|v| !v.is_null());
        let is_error = obj
            .get("is_error")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| {
                result
                    .as_ref()
                    .and_then(|val| val.get("error"))
                    .map(|v| !v.is_null())
                    .unwrap_or(false)
            });
        let provider_metadata = obj.get("item_id").and_then(|v| v.as_str()).map(|id| {
            let mut inner = HashMap::new();
            inner.insert("itemId".into(), serde_json::json!(id));
            let mut outer = HashMap::new();
            outer.insert("openai".into(), inner);
            outer
        });
        let approval_request_id = obj
            .get("approval_request_id")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string());
        let is_approval_request = obj
            .get("approval_request")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Some(ProviderToolParts {
            tool_call_id,
            tool_name,
            tool_type: tool_type.to_string(),
            input,
            provider_executed,
            dynamic,
            result,
            is_error,
            provider_metadata,
            approval_request_id,
            is_approval_request,
        })
    }
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
        match t {
            "response.created" => {
                if let Some(resp) = json.get("response") {
                    let id = resp.get("id").cloned().unwrap_or(serde_json::Value::Null);
                    let model = resp
                        .get("model")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let created_at = resp
                        .get("created_at")
                        .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)))
                        .map(|v| v * 1000);
                    events.push(Event::Data {
                        key: "openai.response_metadata".into(),
                        value: serde_json::json!({
                            "id": id,
                            "model": model,
                            "created_at": created_at,
                        }),
                    });
                }
            }
            "response.output_text.delta" => {
                let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                let item_id = json.get("item_id").and_then(|v| v.as_str());
                if let Some(item_id) = item_id {
                    if !delta.is_empty() {
                        let logprobs = json
                            .get("logprobs")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        events.push(Event::Data {
                            key: "openai.text_delta".into(),
                            value: serde_json::json!({
                                "item_id": item_id,
                                "delta": delta,
                                "logprobs": logprobs,
                            }),
                        });
                    }
                }
            }
            "response.output_text.annotation.added" => {
                let item_id = json.get("item_id").and_then(|v| v.as_str());
                let annotation = json
                    .get("annotation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if let Some(item_id) = item_id {
                    events.push(Event::Data {
                        key: "openai.text_annotation".into(),
                        value: serde_json::json!({
                            "item_id": item_id,
                            "annotation": annotation,
                        }),
                    });
                }
            }
            "response.output_item.added" => {
                if let Some(item) = json.get("item").and_then(|v| v.as_object()) {
                    if let Some(typ) = item.get("type").and_then(|v| v.as_str()) {
                        match typ {
                            "function_call" => {
                                let name = item
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let call_id = item
                                    .get("call_id")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let index = json
                                    .get("output_index")
                                    .and_then(|v| v.as_u64())
                                    .map(|n| n as usize);
                                if let (Some(cid), Some(id)) =
                                    (call_id.as_ref(), item.get("id").and_then(|v| v.as_str()))
                                {
                                    events.push(Event::Data {
                                        key: format!("openai.tool_item_id.{}", cid),
                                        value: serde_json::json!({"item_id": id}),
                                    });
                                }
                                if let (Some(idx), Some(cid), Some(tool_name)) =
                                    (index, call_id.clone(), name.clone())
                                {
                                    self.tool_calls
                                        .insert(idx, OpenAIToolCallState { id: cid.clone() });
                                    events.push(Event::ToolCallStart {
                                        id: cid.clone(),
                                        name: tool_name,
                                    });
                                    if let Some(pending) = self.pending_deltas.remove(&idx) {
                                        for delta in pending {
                                            events.push(Event::ToolCallDelta {
                                                id: cid.clone(),
                                                args_json: delta,
                                            });
                                        }
                                    }
                                }
                            }
                            "apply_patch_call" => {
                                let index = json
                                    .get("output_index")
                                    .and_then(|v| v.as_u64())
                                    .map(|n| n as usize);
                                let call_id = item.get("call_id").and_then(|v| v.as_str());
                                let operation = item
                                    .get("operation")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                if let (Some(idx), Some(call_id)) = (index, call_id) {
                                    events.push(Event::Data {
                                        key: "openai.apply_patch_call.added".into(),
                                        value: serde_json::json!({
                                            "output_index": idx,
                                            "call_id": call_id,
                                            "operation": operation,
                                        }),
                                    });
                                }
                            }
                            "message" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.message_added".into(),
                                        value: serde_json::json!({"item_id": id}),
                                    });
                                }
                            }
                            "reasoning" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    let enc = item
                                        .get("encrypted_content")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null);
                                    events.push(Event::Data {
                                        key: "openai.reasoning_added".into(),
                                        value: serde_json::json!({
                                            "item_id": id,
                                            "encrypted_content": enc,
                                        }),
                                    });
                                }
                            }
                            "web_search_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.web_search_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            "file_search_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.file_search_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            "image_generation_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.image_generation_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            "code_interpreter_call" => {
                                let output_index = json
                                    .get("output_index")
                                    .and_then(|v| v.as_u64())
                                    .map(|n| n as usize);
                                if let (Some(id), Some(idx)) =
                                    (item.get("id").and_then(|v| v.as_str()), output_index)
                                {
                                    let container_id = item
                                        .get("container_id")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string());
                                    events.push(Event::Data {
                                        key: "openai.code_interpreter_call.added".into(),
                                        value: serde_json::json!({
                                            "output_index": idx,
                                            "tool_call_id": id,
                                            "container_id": container_id,
                                        }),
                                    });
                                }
                            }
                            "computer_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.computer_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "response.output_item.done" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                if let Some(item) = json.get("item").and_then(|v| v.as_object()) {
                    if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                        if item_type == "apply_patch_call" {
                            if let Some(idx) = index {
                                let call_id = item.get("call_id").and_then(|v| v.as_str());
                                let operation = item
                                    .get("operation")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                events.push(Event::Data {
                                    key: "openai.apply_patch_call.done".into(),
                                    value: serde_json::json!({
                                        "output_index": idx,
                                        "call_id": call_id,
                                        "operation": operation,
                                    }),
                                });
                            }
                        }
                        if item_type == "message" {
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                events.push(Event::Data {
                                    key: "openai.message_done".into(),
                                    value: serde_json::json!({"item_id": id}),
                                });
                            }
                        }
                        if item_type == "reasoning" {
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                let enc = item
                                    .get("encrypted_content")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                events.push(Event::Data {
                                    key: "openai.reasoning_done".into(),
                                    value: serde_json::json!({
                                        "item_id": id,
                                        "encrypted_content": enc,
                                    }),
                                });
                            }
                        }
                        if item_type == "function_call" {
                            events.push(Event::Data {
                                key: "openai.function_call_done".into(),
                                value: serde_json::json!({}),
                            });
                        }
                    }
                    if let Some(tool_data) = provider_tool_data_from_output_item(item) {
                        events.push(Event::Data {
                            key: "openai.provider_tool".into(),
                            value: tool_data,
                        });
                    }
                }
                if let Some(idx) = index {
                    if let Some(state) = self.tool_calls.remove(&idx) {
                        if let Some(pending) = self.pending_deltas.remove(&idx) {
                            for delta in pending {
                                events.push(Event::ToolCallDelta {
                                    id: state.id.clone(),
                                    args_json: delta,
                                });
                            }
                        }
                        events.push(Event::ToolCallEnd { id: state.id });
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let delta = json
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(idx) = index {
                    if let Some(state) = self.tool_calls.get(&idx) {
                        events.push(Event::ToolCallDelta {
                            id: state.id.clone(),
                            args_json: delta,
                        });
                    } else {
                        self.pending_deltas.entry(idx).or_default().push(delta);
                    }
                }
            }
            "response.code_interpreter_call_code.delta" => {
                if let Some(idx) = json.get("output_index").and_then(|v| v.as_u64()) {
                    let delta = json
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !delta.is_empty() {
                        events.push(Event::Data {
                            key: "openai.code_interpreter_call.code_delta".into(),
                            value: serde_json::json!({
                                "output_index": idx as usize,
                                "delta": delta,
                            }),
                        });
                    }
                }
            }
            "response.code_interpreter_call_code.done" => {
                if let Some(idx) = json.get("output_index").and_then(|v| v.as_u64()) {
                    let code = json
                        .get("code")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    events.push(Event::Data {
                        key: "openai.code_interpreter_call.code_done".into(),
                        value: serde_json::json!({
                            "output_index": idx as usize,
                            "code": code,
                        }),
                    });
                }
            }
            "response.image_generation_call.partial_image" => {
                if let (Some(id), Some(b64)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("partial_image_b64").and_then(|v| v.as_str()),
                ) {
                    events.push(Event::Data {
                        key: "openai.image_generation_call.partial".into(),
                        value: serde_json::json!({
                            "tool_call_id": id,
                            "partial_image_b64": b64,
                        }),
                    });
                }
            }
            "response.apply_patch_call_operation_diff.delta" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let delta = json
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(idx) = index {
                    if !delta.is_empty() {
                        events.push(Event::Data {
                            key: "openai.apply_patch_call.diff.delta".into(),
                            value: serde_json::json!({
                                "output_index": idx,
                                "delta": delta,
                            }),
                        });
                    }
                }
            }
            "response.apply_patch_call_operation_diff.done" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let diff = json
                    .get("diff")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(idx) = index {
                    events.push(Event::Data {
                        key: "openai.apply_patch_call.diff.done".into(),
                        value: serde_json::json!({
                            "output_index": idx,
                            "diff": diff,
                        }),
                    });
                }
            }
            "response.reasoning_summary_part.added" => {
                if let (Some(id), Some(idx)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("summary_index").and_then(|v| v.as_u64()),
                ) {
                    events.push(Event::Data {
                        key: "openai.reasoning_summary_added".into(),
                        value: serde_json::json!({
                            "item_id": id,
                            "summary_index": idx,
                        }),
                    });
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let (Some(id), Some(idx)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("summary_index").and_then(|v| v.as_u64()),
                ) {
                    let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                    if !delta.is_empty() {
                        events.push(Event::Data {
                            key: "openai.reasoning_summary_delta".into(),
                            value: serde_json::json!({
                                "item_id": id,
                                "summary_index": idx,
                                "delta": delta,
                            }),
                        });
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                if let (Some(id), Some(idx)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("summary_index").and_then(|v| v.as_u64()),
                ) {
                    events.push(Event::Data {
                        key: "openai.reasoning_summary_done".into(),
                        value: serde_json::json!({
                            "item_id": id,
                            "summary_index": idx,
                        }),
                    });
                }
            }
            "response.completed" | "response.incomplete" => {
                if let Some(resp) = json.get("response") {
                    if let Some(usage_val) = resp.get("usage") {
                        if let Some(usage) = parse_openai_usage(usage_val) {
                            events.push(Event::Usage { usage });
                        }
                        events.push(Event::Data {
                            key: "usage".into(),
                            value: usage_val.clone(),
                        });
                    }
                    let fin = resp
                        .get("incomplete_details")
                        .and_then(|v| v.get("reason"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    events.push(Event::Data {
                        key: "openai.finish".into(),
                        value: serde_json::json!({"incomplete_reason": fin}),
                    });
                    let rid = resp.get("id").cloned().unwrap_or(serde_json::Value::Null);
                    let st = resp
                        .get("service_tier")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    events.push(Event::Data {
                        key: "openai.response".into(),
                        value: serde_json::json!({"id": rid, "service_tier": st}),
                    });
                }
                if !self.tool_calls.is_empty() {
                    for (_idx, state) in self.tool_calls.drain() {
                        events.push(Event::ToolCallEnd { id: state.id });
                    }
                    self.pending_deltas.clear();
                }
                events.push(Event::Done);
            }
            "response.failed" => {
                // TS parity: failed chunks are not treated as "response finished" chunks.
                // They should not set incomplete_reason/service_tier driven finish metadata.
                let failed_payload = json
                    .get("response")
                    .map(|resp| {
                        serde_json::json!({
                            "id": resp.get("id").cloned().unwrap_or(serde_json::Value::Null),
                        })
                    })
                    .unwrap_or_else(|| serde_json::json!({}));
                events.push(Event::Data {
                    key: "openai.failed".into(),
                    value: failed_payload,
                });
                if !self.tool_calls.is_empty() {
                    for (_idx, state) in self.tool_calls.drain() {
                        events.push(Event::ToolCallEnd { id: state.id });
                    }
                    self.pending_deltas.clear();
                }
                events.push(Event::Done);
            }
            "error" => {
                events.push(Event::Data {
                    key: "openai.error".into(),
                    value: json.clone(),
                });
            }
            _ => {}
        }
        if events.is_empty() {
            Ok(None)
        } else {
            Ok(Some(events))
        }
    }
}

#[derive(Debug, Clone)]
struct OpenAIApplyPatchState {
    tool_call_id: String,
    operation_path: Option<String>,
    has_diff: bool,
    end_emitted: bool,
}

#[derive(Debug, Clone)]
struct OpenAICodeInterpreterState {
    tool_call_id: String,
    container_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningSummaryStatus {
    Active,
    CanConclude,
    Concluded,
}

#[derive(Debug, Clone, Default)]
struct OpenAIReasoningState {
    encrypted_content: Option<serde_json::Value>,
    summary_parts: HashMap<u32, ReasoningSummaryStatus>,
}

#[derive(Default)]
struct OpenAIStreamExtras {
    finish_hint: Option<String>,
    response_id: Option<String>,
    service_tier: Option<String>,
    saw_response_failed: bool,
    store: bool,
    logprobs_enabled: bool,
    has_function_calls: bool,
    logprobs: Vec<serde_json::Value>,
    message_annotations: HashMap<String, Vec<serde_json::Value>>,
    active_reasoning: HashMap<String, OpenAIReasoningState>,
    open_tool_inputs: HashSet<String>,
    tool_item_ids: HashMap<String, String>, // call_id -> item_id
    approval_request_id_map: HashMap<String, String>,
    apply_patch_calls: HashMap<usize, OpenAIApplyPatchState>,
    code_interpreter_calls: HashMap<usize, OpenAICodeInterpreterState>,
    emitted_tool_calls: HashSet<String>,
    tool_name_mapping: ToolNameMapping,
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

fn build_stream_mapper_config(
    warnings: Vec<v2t::CallWarning>,
    tool_name_mapping: ToolNameMapping,
    approval_request_id_map: HashMap<String, String>,
    store: bool,
    logprobs_enabled: bool,
) -> EventMapperConfig<OpenAIStreamExtras> {
    let mut hooks: EventMapperHooks<OpenAIStreamExtras> = EventMapperHooks::default();

    hooks.tool_end_metadata = Some(Box::new(
        |state: &mut EventMapperState<OpenAIStreamExtras>, id| {
            state
                .extra
                .tool_item_ids
                .get(id)
                .map(|iid| openai_item_metadata(iid, []))
        },
    ));

    hooks.data = Some(Box::new(
        |state: &mut EventMapperState<OpenAIStreamExtras>, key, value| {
            if key == "usage" {
                if let Some(usage) = parse_openai_usage(value) {
                    state.usage.input_tokens = Some(usage.input_tokens as u64);
                    state.usage.output_tokens = Some(usage.output_tokens as u64);
                    state.usage.total_tokens = Some(usage.total_tokens as u64);
                    state.usage.cached_input_tokens = usage.cache_read_tokens.map(|v| v as u64);
                }
                apply_openai_usage_details(value, &mut state.usage);
                return None;
            } else if key == "openai.response_metadata" {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let model_id = value
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let timestamp_ms = value
                    .get("created_at")
                    .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)));
                if let Some(rid) = id.as_ref() {
                    if state.extra.response_id.is_none() {
                        state.extra.response_id = Some(rid.clone());
                    }
                }
                let meta = v2t::ResponseMetadata {
                    id,
                    timestamp_ms,
                    model_id,
                };
                return Some(vec![v2t::StreamPart::ResponseMetadata { meta }]);
            } else if key == "openai.message_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                state
                    .extra
                    .message_annotations
                    .insert(item_id.to_string(), Vec::new());
                if state.text_open.as_deref() != Some(item_id) {
                    return Some(
                        state.open_text(
                            item_id.to_string(),
                            Some(openai_item_metadata(item_id, [])),
                        ),
                    );
                }
            } else if key == "openai.text_delta" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                let start_metadata = if state.text_open.as_deref() != Some(item_id) {
                    state
                        .extra
                        .message_annotations
                        .entry(item_id.to_string())
                        .or_default();
                    Some(openai_item_metadata(item_id, []))
                } else {
                    None
                };
                if state.extra.logprobs_enabled {
                    if let Some(logprobs) = value.get("logprobs").filter(|v| !v.is_null()) {
                        state.extra.logprobs.push(logprobs.clone());
                    }
                }
                return Some(state.push_text_delta(
                    Some(item_id.to_string()),
                    item_id,
                    delta.to_string(),
                    start_metadata,
                    None,
                ));
            } else if key == "openai.text_annotation" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let annotation = value.get("annotation")?.clone();
                state
                    .extra
                    .message_annotations
                    .entry(item_id.to_string())
                    .or_default()
                    .push(annotation.clone());
                let annotation_obj = annotation.as_object()?;
                let annotation_type = annotation_obj.get("type")?.as_str()?;
                let make_provider_metadata = |vals: Vec<(&str, serde_json::Value)>| {
                    let mut inner = HashMap::new();
                    for (key, val) in vals {
                        inner.insert(key.into(), val);
                    }
                    let mut outer = HashMap::new();
                    outer.insert("openai".into(), inner);
                    outer
                };
                match annotation_type {
                    "url_citation" => {
                        let url = annotation_obj.get("url")?.as_str()?;
                        let title = annotation_obj
                            .get("title")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: url.to_string(),
                            title,
                            provider_metadata: None,
                        }]);
                    }
                    "file_citation" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let title = annotation_obj
                            .get("quote")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                annotation_obj
                                    .get("filename")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .or_else(|| Some(file_id.to_string()));
                        let mut metadata_vals = vec![("fileId", json!(file_id))];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title,
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    "container_file_citation" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let container_id = annotation_obj.get("container_id")?.as_str()?;
                        let title = annotation_obj
                            .get("filename")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| Some(file_id.to_string()));
                        let mut metadata_vals = vec![
                            ("fileId", json!(file_id)),
                            ("containerId", json!(container_id)),
                        ];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title,
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    "file_path" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let mut metadata_vals = vec![("fileId", json!(file_id))];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title: Some(file_id.to_string()),
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    _ => {}
                }
            } else if key == "openai.message_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let annotations = state
                    .extra
                    .message_annotations
                    .remove(item_id)
                    .unwrap_or_default();
                let md = if annotations.is_empty() {
                    openai_item_metadata(item_id, [])
                } else {
                    openai_item_metadata(
                        item_id,
                        [("annotations".into(), serde_json::Value::Array(annotations))],
                    )
                };
                if state.text_open.as_deref() == Some(item_id) {
                    return state.close_text(Some(md)).map(|part| vec![part]);
                }
                return Some(vec![state.text_end_part(item_id.to_string(), Some(md))]);
            } else if key == "openai.error" {
                state.extra.finish_hint = Some("error".into());
                return Some(vec![v2t::StreamPart::Error {
                    error: value.clone(),
                }]);
            } else if key == "openai.reasoning_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let enc = value
                    .get("encrypted_content")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let mut state_entry = OpenAIReasoningState {
                    encrypted_content: Some(enc.clone()),
                    summary_parts: HashMap::new(),
                };
                state_entry
                    .summary_parts
                    .insert(0, ReasoningSummaryStatus::Active);
                state
                    .extra
                    .active_reasoning
                    .insert(item_id.to_string(), state_entry);
                return Some(state.open_reasoning(
                    format!("{item_id}:0"),
                    Some(openai_item_metadata(
                        item_id,
                        [("reasoningEncryptedContent".into(), enc)],
                    )),
                ));
            } else if key == "openai.reasoning_summary_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                if summary_index == 0 {
                    return None;
                }
                let (concluded_ids, enc) = {
                    let reasoning_state = state.extra.active_reasoning.get_mut(item_id)?;
                    let mut concluded_ids = Vec::new();
                    for (idx, status) in reasoning_state.summary_parts.iter_mut() {
                        if matches!(status, ReasoningSummaryStatus::CanConclude) {
                            concluded_ids.push(*idx);
                            *status = ReasoningSummaryStatus::Concluded;
                        }
                    }
                    reasoning_state
                        .summary_parts
                        .insert(summary_index as u32, ReasoningSummaryStatus::Active);
                    let enc = reasoning_state
                        .encrypted_content
                        .clone()
                        .unwrap_or(serde_json::Value::Null);
                    (concluded_ids, enc)
                };
                let mut out = Vec::new();
                for idx in concluded_ids {
                    let reasoning_id = format!("{item_id}:{idx}");
                    if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                        if let Some(part) =
                            state.close_reasoning(Some(openai_item_metadata(item_id, [])))
                        {
                            out.push(part);
                        }
                    } else {
                        out.push(state.reasoning_end_part(
                            reasoning_id,
                            Some(openai_item_metadata(item_id, [])),
                        ));
                    }
                }
                out.extend(state.open_reasoning(
                    format!("{item_id}:{summary_index}"),
                    Some(openai_item_metadata(
                        item_id,
                        [("reasoningEncryptedContent".into(), enc)],
                    )),
                ));
                return Some(out);
            } else if key == "openai.reasoning_summary_delta" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                return Some(vec![state.push_reasoning_delta(
                    &format!("{item_id}:{summary_index}"),
                    delta.to_string(),
                    Some(openai_item_metadata(item_id, [])),
                )]);
            } else if key == "openai.reasoning_summary_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                let should_close =
                    if let Some(reasoning_state) = state.extra.active_reasoning.get_mut(item_id) {
                        if state.extra.store {
                            reasoning_state
                                .summary_parts
                                .insert(summary_index as u32, ReasoningSummaryStatus::Concluded);
                            Some(format!("{item_id}:{summary_index}"))
                        } else {
                            reasoning_state
                                .summary_parts
                                .insert(summary_index as u32, ReasoningSummaryStatus::CanConclude);
                            None
                        }
                    } else {
                        None
                    };
                if let Some(reasoning_id) = should_close {
                    if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                        return state
                            .close_reasoning(Some(openai_item_metadata(item_id, [])))
                            .map(|part| vec![part]);
                    }
                    return Some(vec![state.reasoning_end_part(
                        reasoning_id,
                        Some(openai_item_metadata(item_id, [])),
                    )]);
                }
            } else if key == "openai.reasoning_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let enc = value
                    .get("encrypted_content")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if let Some(reasoning_state) = state.extra.active_reasoning.remove(item_id) {
                    let md =
                        openai_item_metadata(item_id, [("reasoningEncryptedContent".into(), enc)]);
                    let mut out = Vec::new();
                    for (idx, status) in reasoning_state.summary_parts {
                        if matches!(
                            status,
                            ReasoningSummaryStatus::Active | ReasoningSummaryStatus::CanConclude
                        ) {
                            let reasoning_id = format!("{item_id}:{idx}");
                            if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                                if let Some(part) = state.close_reasoning(Some(md.clone())) {
                                    out.push(part);
                                }
                            } else {
                                out.push(state.reasoning_end_part(reasoning_id, Some(md.clone())));
                            }
                        }
                    }
                    if !out.is_empty() {
                        return Some(out);
                    }
                }
            } else if key == "openai.web_search_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .web_search_tool_name
                    .clone()
                    .unwrap_or_else(|| {
                        state
                            .extra
                            .tool_name_mapping
                            .to_custom_tool_name("web_search")
                            .to_string()
                    });
                state.has_tool_calls = true;
                state
                    .extra
                    .emitted_tool_calls
                    .insert(tool_call_id.to_string());
                let tool_call_id = tool_call_id.to_string();
                let mut out =
                    vec![state.start_tool_call(tool_call_id.clone(), tool_name, true, None)];
                state.tool_args.insert(tool_call_id.clone(), "{}".into());
                out.extend(state.finish_tool_call(tool_call_id, true, None, None, false, None));
                return Some(out);
            } else if key == "openai.file_search_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("file_search")
                    .to_string();
                state.has_tool_calls = true;
                state
                    .extra
                    .emitted_tool_calls
                    .insert(tool_call_id.to_string());
                return Some(vec![state.tool_call_part(
                    tool_call_id.to_string(),
                    tool_name,
                    "{}".into(),
                    true,
                    None,
                    false,
                    None,
                )]);
            } else if key == "openai.image_generation_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("image_generation")
                    .to_string();
                state.has_tool_calls = true;
                state
                    .extra
                    .emitted_tool_calls
                    .insert(tool_call_id.to_string());
                return Some(vec![state.tool_call_part(
                    tool_call_id.to_string(),
                    tool_name,
                    "{}".into(),
                    true,
                    None,
                    false,
                    None,
                )]);
            } else if key == "openai.image_generation_call.partial" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let partial = value.get("partial_image_b64").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("image_generation")
                    .to_string();
                return Some(vec![v2t::StreamPart::ToolResult {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name,
                    result: json!({ "result": partial }),
                    is_error: false,
                    preliminary: true,
                    provider_metadata: None,
                }]);
            } else if key == "openai.code_interpreter_call.added" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let container_id = value
                    .get("container_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                state.extra.code_interpreter_calls.insert(
                    output_index,
                    OpenAICodeInterpreterState {
                        tool_call_id: tool_call_id.to_string(),
                        container_id: container_id.clone(),
                    },
                );
                state.has_tool_calls = true;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("code_interpreter")
                    .to_string();
                let cid = container_id.unwrap_or_default();
                return Some(vec![
                    state.start_tool_call(tool_call_id.to_string(), tool_name, true, None),
                    state.push_tool_call_delta(
                        tool_call_id.to_string(),
                        format!(
                            "{{\"containerId\":\"{}\",\"code\":\"",
                            escape_json_delta(&cid)
                        ),
                        true,
                        None,
                    ),
                ]);
            } else if key == "openai.code_interpreter_call.code_delta" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                if let Some(call_state) = state.extra.code_interpreter_calls.get(&output_index) {
                    return Some(vec![state.push_tool_call_delta(
                        call_state.tool_call_id.clone(),
                        escape_json_delta(delta),
                        true,
                        None,
                    )]);
                }
            } else if key == "openai.code_interpreter_call.code_done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                if let Some(call_state) = state.extra.code_interpreter_calls.remove(&output_index) {
                    let mut out = vec![state.push_tool_call_delta(
                        call_state.tool_call_id.clone(),
                        "\"}".into(),
                        true,
                        None,
                    )];
                    out.extend(state.finish_tool_call(
                        call_state.tool_call_id.clone(),
                        true,
                        None,
                        None,
                        false,
                        None,
                    ));
                    state
                        .extra
                        .emitted_tool_calls
                        .insert(call_state.tool_call_id);
                    return Some(out);
                }
            } else if key == "openai.computer_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                state
                    .extra
                    .open_tool_inputs
                    .insert(tool_call_id.to_string());
                state.has_tool_calls = true;
                return Some(vec![state.start_tool_call(
                    tool_call_id.to_string(),
                    state
                        .extra
                        .tool_name_mapping
                        .to_custom_tool_name("computer_use")
                        .to_string(),
                    true,
                    None,
                )]);
            } else if key == "openai.apply_patch_call.added" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let call_id = value.get("call_id").and_then(|v| v.as_str())?;
                let operation = value
                    .get("operation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let operation_type = operation.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if operation_type.is_empty() {
                    return None;
                }
                let operation_path = operation
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string());
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("apply_patch")
                    .to_string();
                let mut call_state = OpenAIApplyPatchState {
                    tool_call_id: call_id.to_string(),
                    operation_path,
                    has_diff: false,
                    end_emitted: false,
                };
                let mut out =
                    vec![state.start_tool_call(call_id.to_string(), tool_name, false, None)];
                if operation_type == "delete_file" {
                    let input = json!({
                        "callId": call_id,
                        "operation": operation,
                    })
                    .to_string();
                    out.push(state.push_tool_call_delta(call_id.to_string(), input, false, None));
                    out.push(state.tool_input_end_part(call_id.to_string(), false, None));
                    call_state.has_diff = true;
                    call_state.end_emitted = true;
                } else {
                    let path = call_state.operation_path.as_deref().unwrap_or("");
                    let delta = format!(
                        "{{\"callId\":\"{}\",\"operation\":{{\"type\":\"{}\",\"path\":\"{}\",\"diff\":\"",
                        escape_json_delta(call_id),
                        escape_json_delta(operation_type),
                        escape_json_delta(path)
                    );
                    out.push(state.push_tool_call_delta(call_id.to_string(), delta, false, None));
                }
                state.has_tool_calls = true;
                state
                    .extra
                    .apply_patch_calls
                    .insert(output_index, call_state);
                return Some(out);
            } else if key == "openai.apply_patch_call.diff.delta" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(call_state) = state.extra.apply_patch_calls.get_mut(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    if !delta.is_empty() {
                        call_state.has_diff = true;
                        let tool_call_id = call_state.tool_call_id.clone();
                        return Some(vec![state.push_tool_call_delta(
                            tool_call_id,
                            escape_json_delta(delta),
                            false,
                            None,
                        )]);
                    }
                }
            } else if key == "openai.apply_patch_call.diff.done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let diff = value.get("diff").and_then(|v| v.as_str()).unwrap_or("");
                if let Some((tool_call_id, should_emit_diff)) = state
                    .extra
                    .apply_patch_calls
                    .get_mut(&output_index)
                    .and_then(|call_state| {
                        if call_state.end_emitted {
                            return None;
                        }
                        let should_emit_diff = !call_state.has_diff;
                        if should_emit_diff {
                            call_state.has_diff = true;
                        }
                        call_state.end_emitted = true;
                        Some((call_state.tool_call_id.clone(), should_emit_diff))
                    })
                {
                    let mut out = Vec::new();
                    if should_emit_diff {
                        out.push(state.push_tool_call_delta(
                            tool_call_id.clone(),
                            escape_json_delta(diff),
                            false,
                            None,
                        ));
                    }
                    out.push(state.push_tool_call_delta(
                        tool_call_id.clone(),
                        "\"}}".into(),
                        false,
                        None,
                    ));
                    out.push(state.tool_input_end_part(tool_call_id, false, None));
                    return Some(out);
                }
            } else if key == "openai.apply_patch_call.done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                if let Some(mut call_state) = state.extra.apply_patch_calls.remove(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    let mut out = Vec::new();
                    if !call_state.has_diff {
                        let diff = value
                            .get("operation")
                            .and_then(|v| v.get("diff"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        call_state.has_diff = true;
                        out.push(state.push_tool_call_delta(
                            call_state.tool_call_id.clone(),
                            escape_json_delta(diff),
                            false,
                            None,
                        ));
                    }
                    out.push(state.push_tool_call_delta(
                        call_state.tool_call_id.clone(),
                        "\"}}".into(),
                        false,
                        None,
                    ));
                    out.push(state.tool_input_end_part(
                        call_state.tool_call_id.clone(),
                        false,
                        None,
                    ));
                    call_state.end_emitted = true;
                    return Some(out);
                }
            } else if key == "openai.provider_tool" {
                if let Some(mut parts) =
                    provider_tool_parts_from_data(value, &state.extra.tool_name_mapping)
                {
                    state.has_tool_calls = true;
                    let tool_type = parts.tool_type.clone();
                    if parts.is_approval_request {
                        let approval_id = parts
                            .approval_request_id
                            .clone()
                            .unwrap_or_else(|| parts.tool_call_id.clone());
                        let tool_call_id = Uuid::new_v4().to_string();
                        state
                            .extra
                            .approval_request_id_map
                            .insert(approval_id.clone(), tool_call_id.clone());
                        return Some(vec![
                            state.tool_call_part(
                                tool_call_id.clone(),
                                parts.tool_name,
                                parts.input,
                                parts.provider_executed,
                                None,
                                parts.dynamic,
                                None,
                            ),
                            v2t::StreamPart::ToolApprovalRequest {
                                approval_id,
                                tool_call_id,
                                provider_metadata: None,
                            },
                        ]);
                    }
                    if let Some(approval_id) = parts.approval_request_id.as_ref() {
                        if let Some(mapped) = state.extra.approval_request_id_map.get(approval_id) {
                            parts.tool_call_id = mapped.clone();
                        }
                    }
                    let tool_call_id = parts.tool_call_id.clone();
                    let tool_call_metadata = match tool_type.as_str() {
                        "apply_patch" | "local_shell" | "shell" => parts.provider_metadata.clone(),
                        _ => None,
                    };
                    let tool_result_metadata = if tool_type == "mcp" {
                        parts.provider_metadata.clone()
                    } else {
                        None
                    };
                    let mut out = Vec::new();
                    if tool_type == "computer_use" {
                        if state.extra.open_tool_inputs.remove(&tool_call_id) {
                            out.push(state.tool_input_end_part(tool_call_id.clone(), true, None));
                        }
                    }
                    let skip_tool_call =
                        matches!(
                            tool_type.as_str(),
                            "web_search" | "file_search" | "image_generation" | "code_interpreter"
                        ) && state.extra.emitted_tool_calls.contains(&tool_call_id);
                    if !skip_tool_call {
                        out.push(state.tool_call_part(
                            tool_call_id.clone(),
                            parts.tool_name.clone(),
                            parts.input,
                            parts.provider_executed,
                            tool_call_metadata,
                            parts.dynamic,
                            None,
                        ));
                    }
                    if let Some(result) = parts.result.take() {
                        out.push(v2t::StreamPart::ToolResult {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: parts.tool_name,
                            result,
                            is_error: parts.is_error,
                            preliminary: false,
                            provider_metadata: tool_result_metadata,
                        });
                    }
                    if !out.is_empty() {
                        return Some(out);
                    }
                }
            } else if key.starts_with("openai.tool_item_id.") {
                if let Some(iid) = value.get("item_id").and_then(|v| v.as_str()) {
                    let call_id = key.trim_start_matches("openai.tool_item_id.").to_string();
                    state.extra.tool_item_ids.insert(call_id, iid.to_string());
                }
            } else if key == "openai.function_call_done" {
                state.extra.has_function_calls = true;
            } else if key == "openai.finish" {
                if let Some(r) = value.get("incomplete_reason").and_then(|v| v.as_str()) {
                    state.extra.finish_hint = Some(r.to_string());
                }
            } else if key == "openai.failed" {
                state.extra.saw_response_failed = true;
                if state.extra.response_id.is_none() {
                    if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                        state.extra.response_id = Some(id.to_string());
                    }
                }
            } else if key == "openai.response" {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    state.extra.response_id = Some(id.to_string());
                }
                if let Some(st) = value.get("service_tier").and_then(|v| v.as_str()) {
                    state.extra.service_tier = Some(st.to_string());
                }
            }
            None
        },
    ));

    hooks.finish = Some(Box::new(|state: &EventMapperState<OpenAIStreamExtras>| {
        let reason = if state.extra.saw_response_failed {
            // TS mapper keeps default "other" for response.failed terminal trajectories.
            v2t::FinishReason::Other
        } else {
            map_finish_reason(
                state.extra.finish_hint.as_deref(),
                state.extra.has_function_calls,
            )
        };
        let mut inner = HashMap::new();
        if let Some(rid) = &state.extra.response_id {
            inner.insert("responseId".into(), serde_json::json!(rid));
        }
        if !state.extra.saw_response_failed {
            if let Some(st) = &state.extra.service_tier {
                inner.insert("serviceTier".into(), serde_json::json!(st));
            }
        }
        if !state.extra.logprobs.is_empty() {
            inner.insert(
                "logprobs".into(),
                serde_json::Value::Array(state.extra.logprobs.clone()),
            );
        }
        let metadata = if inner.is_empty() {
            None
        } else {
            let mut outer = HashMap::new();
            outer.insert("openai".into(), inner);
            Some(outer)
        };
        (reason, metadata)
    }));

    EventMapperConfig {
        warnings,
        treat_tool_names_as_text: HashSet::new(),
        default_text_id: "text-1",
        finish_reason_fallback: v2t::FinishReason::Stop,
        initial_extra: OpenAIStreamExtras {
            tool_name_mapping,
            approval_request_id_map,
            store,
            logprobs_enabled,
            ..Default::default()
        },
        hooks,
    }
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

fn extract_response_content(
    json: &serde_json::Value,
    tool_name_mapping: &ToolNameMapping,
    approval_request_id_map: &HashMap<String, String>,
) -> (Vec<v2t::Content>, bool) {
    let mut content: Vec<v2t::Content> = Vec::new();
    let mut has_function_calls = false;
    let mut approval_request_id_map = approval_request_id_map.clone();
    let output = match json.get("output").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return (content, false),
    };

    for item in output {
        let item_obj = match item.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        let item_type = item_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match item_type {
            "message" => {
                if let Some(parts) = item_obj.get("content").and_then(|v| v.as_array()) {
                    let mut text_acc = String::new();
                    for part in parts {
                        if part.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                text_acc.push_str(text);
                            }
                        }
                    }
                    if !text_acc.is_empty() {
                        content.push(v2t::Content::Text {
                            text: text_acc,
                            provider_metadata: None,
                        });
                    }
                }
            }
            "function_call" => {
                if let (Some(call_id), Some(name)) = (
                    item_obj.get("call_id").and_then(|v| v.as_str()),
                    item_obj.get("name").and_then(|v| v.as_str()),
                ) {
                    let args = item_obj
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let provider_metadata = item_obj
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|id| openai_item_metadata(id, []));
                    content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                        tool_call_id: call_id.to_string(),
                        tool_name: name.to_string(),
                        input: args,
                        provider_executed: false,
                        provider_metadata,
                        dynamic: false,
                        provider_options: None,
                    }));
                    has_function_calls = true;
                }
            }
            _ => {
                if let Some(tool_data) = provider_tool_data_from_output_item(item_obj) {
                    if let Some(mut parts) =
                        provider_tool_parts_from_data(&tool_data, tool_name_mapping)
                    {
                        let tool_type = parts.tool_type.clone();
                        if parts.is_approval_request {
                            let approval_id = parts
                                .approval_request_id
                                .clone()
                                .unwrap_or_else(|| parts.tool_call_id.clone());
                            let tool_call_id = Uuid::new_v4().to_string();
                            approval_request_id_map
                                .insert(approval_id.clone(), tool_call_id.clone());
                            content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                                tool_call_id: tool_call_id.clone(),
                                tool_name: parts.tool_name,
                                input: parts.input,
                                provider_executed: parts.provider_executed,
                                provider_metadata: None,
                                dynamic: parts.dynamic,
                                provider_options: None,
                            }));
                            content.push(v2t::Content::ToolApprovalRequest {
                                approval_id,
                                tool_call_id,
                                provider_metadata: None,
                            });
                            continue;
                        }
                        if let Some(approval_id) = parts.approval_request_id.as_ref() {
                            if let Some(mapped) = approval_request_id_map.get(approval_id) {
                                parts.tool_call_id = mapped.clone();
                            }
                        }
                        let tool_call_metadata = match tool_type.as_str() {
                            "apply_patch" | "local_shell" | "shell" => {
                                parts.provider_metadata.clone()
                            }
                            _ => None,
                        };
                        let tool_result_metadata = if tool_type == "mcp" {
                            parts.provider_metadata.clone()
                        } else {
                            None
                        };
                        content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                            tool_call_id: parts.tool_call_id.clone(),
                            tool_name: parts.tool_name.clone(),
                            input: parts.input,
                            provider_executed: parts.provider_executed,
                            provider_metadata: tool_call_metadata,
                            dynamic: parts.dynamic,
                            provider_options: None,
                        }));
                        if let Some(result) = parts.result {
                            content.push(v2t::Content::ToolResult {
                                tool_call_id: parts.tool_call_id,
                                tool_name: parts.tool_name,
                                result,
                                is_error: parts.is_error,
                                provider_metadata: tool_result_metadata,
                            });
                        }
                    }
                }
            }
        }
    }

    (content, has_function_calls)
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
