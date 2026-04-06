use std::collections::{BTreeMap, HashMap};

use crate::ai_sdk_core::request_builder::defaults::{
    build_call_options, request_overrides_from_json,
};
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{
    GenerateResponse, LanguageModel, PartStream, SdkError, StreamNormalizationState, StreamResponse,
};
use crate::ai_sdk_streaming_sse::SseDecoder;
use crate::ai_sdk_types::v2 as v2t;
use async_stream::try_stream;
use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value as JsonValue;
use tracing::instrument;

use crate::provider_gateway::config::GatewayConfig;
use crate::provider_gateway::error::map_transport_error;

const SPEC_VERSION_HEADER: &str = "ai-language-model-specification-version";
const MODEL_ID_HEADER: &str = "ai-language-model-id";
const STREAMING_HEADER: &str = "ai-language-model-streaming";

pub struct GatewayLanguageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport> {
    pub model_id: String,
    pub config: GatewayConfig,
    pub http: T,
}

impl Default for GatewayLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    fn default() -> Self {
        let transport_cfg = TransportConfig::default();
        Self {
            model_id: String::new(),
            config: GatewayConfig {
                provider_name: "gateway",
                provider_scope_name: "gateway".into(),
                base_url: "https://ai-gateway.vercel.sh/v1/ai".into(),
                endpoint_path: None,
                headers: Vec::new(),
                query_params: Vec::new(),
                supported_urls: HashMap::new(),
                transport_cfg: transport_cfg.clone(),
                default_options: None,
                request_defaults: None,
                auth: None,
            },
            http: crate::reqwest_transport::ReqwestTransport::new(&transport_cfg),
        }
    }
}

impl<T: HttpTransport> GatewayLanguageModel<T> {
    pub fn new(model_id: impl Into<String>, config: GatewayConfig, http: T) -> Self {
        Self {
            model_id: model_id.into(),
            config,
            http,
        }
    }

    fn transport_config(&self) -> &TransportConfig {
        &self.config.transport_cfg
    }

    fn endpoint_url(&self) -> String {
        let mut url = self.config.language_endpoint();
        if !self.config.query_params.is_empty() {
            let query = self
                .config
                .query_params
                .iter()
                .fold(String::new(), |mut acc, (k, v)| {
                    if acc.is_empty() {
                        acc.push('?');
                    } else {
                        acc.push('&');
                    }
                    acc.push_str(&urlencoding::encode(k));
                    acc.push('=');
                    acc.push_str(&urlencoding::encode(v));
                    acc
                });
            url.push_str(&query);
        }
        url
    }

    fn canonicalize_header(lc: &str) -> String {
        lc.split('-')
            .map(|segment| {
                let mut chars = segment.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first
                        .to_ascii_uppercase()
                        .to_string()
                        .chars()
                        .chain(chars.as_str().to_ascii_lowercase().chars())
                        .collect(),
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    fn collect_o11y_headers() -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if let Ok(deployment) = std::env::var("VERCEL_DEPLOYMENT_ID") {
            if !deployment.is_empty() {
                headers.push(("ai-o11y-deployment-id".into(), deployment));
            }
        }
        if let Ok(env) = std::env::var("VERCEL_ENV") {
            if !env.is_empty() {
                headers.push(("ai-o11y-environment".into(), env));
            }
        }
        if let Ok(region) = std::env::var("VERCEL_REGION") {
            if !region.is_empty() {
                headers.push(("ai-o11y-region".into(), region));
            }
        }
        if let Ok(request_id) = std::env::var("VERCEL_REQUEST_ID") {
            if !request_id.is_empty() {
                headers.push(("ai-o11y-request-id".into(), request_id));
            }
        } else if let Ok(request_id) = std::env::var("X_VERCEL_ID") {
            if !request_id.is_empty() {
                headers.push(("ai-o11y-request-id".into(), request_id));
            }
        }
        headers
    }

    fn merge_headers(
        &self,
        call_headers: &HashMap<String, String>,
        streaming: bool,
    ) -> Vec<(String, String)> {
        let mut merged: BTreeMap<String, String> = BTreeMap::new();
        merged.insert("content-type".into(), "application/json".into());
        merged.insert("accept".into(), "application/json".into());
        for (k, v) in &self.config.headers {
            merged.insert(k.to_ascii_lowercase(), v.clone());
        }
        merged.insert(SPEC_VERSION_HEADER.into(), "2".into());
        merged.insert(MODEL_ID_HEADER.into(), self.model_id.clone());
        merged.insert(STREAMING_HEADER.into(), streaming.to_string());
        if let Some(auth) = &self.config.auth {
            let mut token = auth.token.clone();
            if !token.to_ascii_lowercase().starts_with("bearer ") {
                token = format!("Bearer {}", token);
            }
            merged.insert("authorization".into(), token);
            merged.insert(
                "ai-gateway-auth-method".into(),
                auth.method.as_header_value().to_string(),
            );
        }
        for (k, v) in Self::collect_o11y_headers() {
            merged.insert(k.to_ascii_lowercase(), v);
        }
        for (k, v) in call_headers {
            if v.trim().is_empty() {
                continue;
            }
            merged.insert(k.to_ascii_lowercase(), v.clone());
        }
        merged
            .into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect()
    }

    fn encode_file_parts(options: &mut v2t::CallOptions) {
        for message in options.prompt.iter_mut() {
            match message {
                v2t::PromptMessage::User { content, .. } => {
                    for part in content.iter_mut() {
                        if let v2t::UserPart::File {
                            data, media_type, ..
                        } = part
                        {
                            Self::encode_data_content(media_type, data);
                        }
                    }
                }
                v2t::PromptMessage::Assistant { content, .. } => {
                    for part in content.iter_mut() {
                        if let v2t::AssistantPart::File {
                            data, media_type, ..
                        } = part
                        {
                            Self::encode_data_content(media_type, data);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn encode_data_content(media_type: &str, data: &mut v2t::DataContent) {
        match data {
            v2t::DataContent::Bytes { bytes } => {
                if bytes.is_empty() {
                    return;
                }
                let encoded = BASE64.encode(bytes);
                let url = format!("data:{};base64,{}", media_type, encoded);
                *data = v2t::DataContent::Url { url };
            }
            v2t::DataContent::Base64 { base64 } => {
                if base64.starts_with("data:") {
                    return;
                }
                let url = format!("data:{};base64,{}", media_type, base64);
                *data = v2t::DataContent::Url { url };
            }
            v2t::DataContent::Url { .. } => {}
        }
    }

    fn headers_vec_to_map(headers: Vec<(String, String)>) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for (k, v) in headers {
            map.insert(k, v);
        }
        map
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync> LanguageModel for GatewayLanguageModel<T>
where
    T::StreamResponse: Send,
{
    fn provider_name(&self) -> &'static str {
        self.config.provider_name
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.config.supported_urls.clone()
    }

    #[instrument(name = "gateway.do_generate", skip_all, fields(model = %self.model_id))]
    async fn do_generate(&self, options: v2t::CallOptions) -> Result<GenerateResponse, SdkError> {
        let mut options = build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        Self::encode_file_parts(&mut options);
        let mut body = serde_json::to_value(&options)?;
        if let Some(defaults) = self.config.request_defaults.as_ref() {
            if let Some(overrides) =
                request_overrides_from_json(&self.config.provider_scope_name, defaults)
            {
                let disallow = ["model", "prompt", "stream", "tools", "input"];
                crate::ai_sdk_core::options::merge_options_with_disallow(
                    &mut body, &overrides, &disallow,
                );
            }
        }

        let headers = self.merge_headers(&options.headers, false);
        match self
            .http
            .post_json(
                &self.endpoint_url(),
                &headers,
                &body,
                self.transport_config(),
            )
            .await
        {
            Ok((response_body, response_headers)) => {
                let content = content_from_value(response_body.get("content"))?;
                let finish_reason = parse_finish_reason(
                    response_body
                        .get("finish_reason")
                        .or_else(|| response_body.get("finishReason")),
                );
                let usage = parse_usage(response_body.get("usage"));
                let provider_metadata = provider_metadata_from_value(
                    response_body
                        .get("provider_metadata")
                        .or_else(|| response_body.get("providerMetadata")),
                );
                let warnings = response_body
                    .get("warnings")
                    .map(parse_call_warnings)
                    .unwrap_or_default();

                Ok(GenerateResponse {
                    content,
                    finish_reason,
                    usage,
                    provider_metadata,
                    request_body: Some(body),
                    response_headers: Some(Self::headers_vec_to_map(response_headers.clone())),
                    response_body: Some(response_body.clone()),
                    warnings,
                })
            }
            Err(err) => Err(map_transport_error(err)),
        }
    }

    #[instrument(name = "gateway.do_stream", skip_all, fields(model = %self.model_id))]
    async fn do_stream(&self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let mut options = build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        let include_raw = options.include_raw_chunks;
        Self::encode_file_parts(&mut options);
        let mut body = serde_json::to_value(&options)?;
        if let Some(defaults) = self.config.request_defaults.as_ref() {
            if let Some(overrides) =
                request_overrides_from_json(&self.config.provider_scope_name, defaults)
            {
                let disallow = ["model", "prompt", "stream", "tools", "input"];
                crate::ai_sdk_core::options::merge_options_with_disallow(
                    &mut body, &overrides, &disallow,
                );
            }
        }
        let headers = self.merge_headers(&options.headers, true);
        match self
            .http
            .post_json_stream(
                &self.endpoint_url(),
                &headers,
                &body,
                self.transport_config(),
            )
            .await
        {
            Ok(resp) => {
                let (stream, response_headers) = T::into_stream(resp);
                let mapped_stream = stream.map(|chunk| chunk.map_err(SdkError::from));
                let part_stream = decode_gateway_stream(mapped_stream, include_raw);
                Ok(StreamResponse {
                    stream: part_stream,
                    request_body: Some(body),
                    response_headers: Some(Self::headers_vec_to_map(response_headers)),
                })
            }
            Err(err) => Err(map_transport_error(err)),
        }
    }
}

fn decode_gateway_stream<S>(bytes: S, include_raw: bool) -> PartStream
where
    S: Stream<Item = Result<Bytes, SdkError>> + Send + 'static,
{
    Box::pin(try_stream! {
        let mut decoder = SseDecoder::new();
        let mut state = GatewayStreamState::default();
        futures_util::pin_mut!(bytes);

        while let Some(chunk) = bytes.next().await {
            let chunk = chunk?;
            for event in decoder.push(&chunk) {
                for part in decode_gateway_event_data(&mut state, &event.data, include_raw) {
                    yield part;
                }
            }
        }

        for event in decoder.finish() {
            for part in decode_gateway_event_data(&mut state, &event.data, include_raw) {
                yield part;
            }
        }
    })
}

fn decode_gateway_event_data(
    state: &mut GatewayStreamState,
    data: &[u8],
    include_raw: bool,
) -> Vec<v2t::StreamPart> {
    if data.is_empty() || data == b"[DONE]" {
        return Vec::new();
    }

    match serde_json::from_slice::<JsonValue>(data) {
        Ok(raw_value) => state.process_chunk(raw_value, include_raw),
        Err(_) if include_raw => vec![v2t::StreamPart::Raw {
            raw_value: JsonValue::String(String::from_utf8_lossy(data).to_string()),
        }],
        Err(_) => Vec::new(),
    }
}

struct GatewayStreamState {
    stream_started: bool,
    normalizer: StreamNormalizationState<()>,
    pending_tool_end_metadata: HashMap<String, Option<v2t::ProviderMetadata>>,
    text_counter: usize,
    current_text_id: Option<String>,
    reasoning_counter: usize,
    current_reasoning_id: Option<String>,
    finished_emitted: bool,
}

impl Default for GatewayStreamState {
    fn default() -> Self {
        Self {
            stream_started: false,
            normalizer: StreamNormalizationState::new(()),
            pending_tool_end_metadata: HashMap::new(),
            text_counter: 0,
            current_text_id: None,
            reasoning_counter: 0,
            current_reasoning_id: None,
            finished_emitted: false,
        }
    }
}

impl GatewayStreamState {
    fn stream_start_parts(&mut self, value: &JsonValue) -> Vec<v2t::StreamPart> {
        self.stream_started = true;
        let warnings = parse_call_warnings(value.get("warnings").unwrap_or(&JsonValue::Null));
        vec![v2t::StreamPart::StreamStart { warnings }]
    }

    fn ensure_stream_started(&mut self, parts: &mut Vec<v2t::StreamPart>) {
        if !self.stream_started {
            self.stream_started = true;
            parts.push(v2t::StreamPart::StreamStart {
                warnings: Vec::new(),
            });
        }
    }

    fn process_text_chunk(&mut self, chunk_type: &str, value: &JsonValue) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        match chunk_type {
            "text-start" => {
                let id = self.ensure_text_id(value.get("id").and_then(|v| v.as_str()));
                if self.normalizer.text_open.as_ref() != Some(&id) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.extend(self.normalizer.open_text(id, metadata));
                }
            }
            "text-delta" => {
                let delta = value
                    .get("textDelta")
                    .or_else(|| value.get("delta"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if delta.is_empty() {
                    return parts;
                }
                let id = self.ensure_text_id(value.get("id").and_then(|v| v.as_str()));
                let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                let start_metadata = if self.normalizer.text_open.as_ref() != Some(&id) {
                    metadata.clone()
                } else {
                    None
                };
                parts.extend(self.normalizer.push_text_delta(
                    Some(id),
                    "text-1",
                    delta,
                    start_metadata,
                    metadata,
                ));
            }
            "text-end" => {
                let id = self.ensure_text_id(value.get("id").and_then(|v| v.as_str()));
                if self.normalizer.text_open.as_deref() == Some(id.as_str()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    if let Some(part) = self.normalizer.close_text(metadata) {
                        parts.push(part);
                    }
                }
                self.current_text_id = None;
            }
            _ => {}
        }
        parts
    }

    fn process_reasoning_chunk(
        &mut self,
        chunk_type: &str,
        value: &JsonValue,
    ) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        match chunk_type {
            "reasoning-start" => {
                let id = self.ensure_reasoning_id(value.get("id").and_then(|v| v.as_str()));
                if self.normalizer.reasoning_open.as_ref() != Some(&id) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.extend(self.normalizer.open_reasoning(id, metadata));
                }
            }
            "reasoning-delta" => {
                let delta = value
                    .get("reasoningDelta")
                    .or_else(|| value.get("delta"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if delta.is_empty() {
                    return parts;
                }
                let id = self.ensure_reasoning_id(value.get("id").and_then(|v| v.as_str()));
                let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                if self.normalizer.reasoning_open.as_ref() != Some(&id) {
                    parts.extend(self.normalizer.open_reasoning(id, metadata.clone()));
                }
                parts.push(
                    self.normalizer
                        .push_reasoning_delta("reasoning-1", delta, metadata),
                );
            }
            "reasoning-end" => {
                let id = self.ensure_reasoning_id(value.get("id").and_then(|v| v.as_str()));
                if self.normalizer.reasoning_open.as_deref() == Some(id.as_str()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    if let Some(part) = self.normalizer.close_reasoning(metadata) {
                        parts.push(part);
                    }
                }
                self.current_reasoning_id = None;
            }
            _ => {}
        }
        parts
    }

    fn process_tool_chunk(&mut self, chunk_type: &str, value: &JsonValue) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        match chunk_type {
            "tool-input-start" => {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    let tool_name = value
                        .get("toolName")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let provider_executed = value
                        .get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(self.normalizer.start_tool_call(
                        id.to_string(),
                        tool_name,
                        provider_executed,
                        metadata,
                    ));
                }
            }
            "tool-input-delta" => {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    let delta = value
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if delta.is_empty() {
                        return parts;
                    }
                    let provider_executed = value
                        .get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(self.normalizer.push_tool_call_delta(
                        id.to_string(),
                        delta,
                        provider_executed,
                        metadata,
                    ));
                }
            }
            "tool-input-end" => {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    self.pending_tool_end_metadata
                        .insert(id.to_string(), metadata);
                }
            }
            "tool-call" => {
                if let Some(tool_call_id) = value.get("toolCallId").and_then(|v| v.as_str()) {
                    let tool_name = value
                        .get("toolName")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let input_json = match value.get("input") {
                        Some(JsonValue::String(s)) => s.clone(),
                        Some(other) => serde_json::to_string(other).unwrap_or_default(),
                        None => String::new(),
                    };
                    let provider_executed = value
                        .get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let tool_call_id = tool_call_id.to_string();
                    if !self.normalizer.tool_names.contains_key(&tool_call_id) {
                        self.normalizer
                            .tool_names
                            .insert(tool_call_id.clone(), tool_name.clone());
                    }
                    self.normalizer
                        .tool_args
                        .insert(tool_call_id.clone(), input_json);
                    let end_metadata = self
                        .pending_tool_end_metadata
                        .remove(&tool_call_id)
                        .unwrap_or(None);
                    parts.extend(self.normalizer.finish_tool_call(
                        tool_call_id,
                        provider_executed,
                        end_metadata,
                        None,
                        false,
                        None,
                    ));
                }
            }
            "tool-result" => {
                if let Some(tool_call_id) = value.get("toolCallId").and_then(|v| v.as_str()) {
                    let tool_name = value
                        .get("toolName")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let result = value.get("result").cloned().unwrap_or(JsonValue::Null);
                    let is_error = value
                        .get("isError")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::ToolResult {
                        tool_call_id: tool_call_id.to_string(),
                        tool_name,
                        result,
                        is_error,
                        preliminary: false,
                        provider_metadata: metadata,
                    });
                }
            }
            _ => {}
        }
        parts
    }

    fn process_file_chunk(&self, value: &JsonValue, include_raw: bool) -> Vec<v2t::StreamPart> {
        if let (Some(media_type), Some(data_str)) = (
            value.get("mediaType").and_then(|v| v.as_str()),
            value.get("data").and_then(|v| v.as_str()),
        ) {
            return vec![v2t::StreamPart::File {
                media_type: media_type.to_string(),
                data: data_str.to_string(),
            }];
        }
        if include_raw {
            vec![v2t::StreamPart::Raw {
                raw_value: value.clone(),
            }]
        } else {
            Vec::new()
        }
    }

    fn process_source_chunk(&self, value: &JsonValue, include_raw: bool) -> Vec<v2t::StreamPart> {
        let Some(source_type) = value.get("sourceType").and_then(|v| v.as_str()) else {
            return Vec::new();
        };

        match source_type {
            "url" => {
                if let (Some(id), Some(url)) = (
                    value.get("id").and_then(|v| v.as_str()),
                    value.get("url").and_then(|v| v.as_str()),
                ) {
                    let title = value
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    return vec![v2t::StreamPart::SourceUrl {
                        id: id.to_string(),
                        url: url.to_string(),
                        title,
                        provider_metadata: metadata,
                    }];
                }
                Vec::new()
            }
            _ if include_raw => vec![v2t::StreamPart::Raw {
                raw_value: value.clone(),
            }],
            _ => Vec::new(),
        }
    }

    fn process_finish_chunk(&mut self, value: &JsonValue) -> Vec<v2t::StreamPart> {
        if self.finished_emitted {
            return Vec::new();
        }
        let usage = parse_usage(value.get("usage"));
        let finish_reason = parse_finish_reason(
            value
                .get("finishReason")
                .or_else(|| value.get("finish_reason")),
        );
        let metadata = provider_metadata_from_value(value.get("providerMetadata"));
        self.normalizer.usage = usage;
        self.finished_emitted = true;
        vec![self.normalizer.finish_part(finish_reason, metadata)]
    }

    fn process_raw_chunk(&self, value: &JsonValue, include_raw: bool) -> Vec<v2t::StreamPart> {
        if !include_raw {
            return Vec::new();
        }
        value
            .get("rawValue")
            .cloned()
            .map(|raw_value| vec![v2t::StreamPart::Raw { raw_value }])
            .unwrap_or_default()
    }

    fn process_chunk(&mut self, value: JsonValue, include_raw: bool) -> Vec<v2t::StreamPart> {
        let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if chunk_type == "stream-start" {
            return self.stream_start_parts(&value);
        }

        let mut parts = Vec::new();
        self.ensure_stream_started(&mut parts);
        let chunk_parts = match chunk_type {
            "text-start" | "text-delta" | "text-end" => self.process_text_chunk(chunk_type, &value),
            "reasoning-start" | "reasoning-delta" | "reasoning-end" => {
                self.process_reasoning_chunk(chunk_type, &value)
            }
            "tool-input-start" | "tool-input-delta" | "tool-input-end" | "tool-call"
            | "tool-result" => self.process_tool_chunk(chunk_type, &value),
            "file" => self.process_file_chunk(&value, include_raw),
            "source" => self.process_source_chunk(&value, include_raw),
            "response-metadata" => {
                let meta = parse_response_metadata(&value);
                vec![v2t::StreamPart::ResponseMetadata { meta }]
            }
            "finish" => self.process_finish_chunk(&value),
            "error" => {
                let payload = value
                    .get("error")
                    .cloned()
                    .unwrap_or(JsonValue::String("Gateway error".into()));
                vec![v2t::StreamPart::Error { error: payload }]
            }
            "raw" => self.process_raw_chunk(&value, include_raw),
            _ if include_raw => vec![v2t::StreamPart::Raw {
                raw_value: value.clone(),
            }],
            _ => Vec::new(),
        };
        parts.extend(chunk_parts);
        parts
    }

    fn ensure_text_id(&mut self, provided: Option<&str>) -> String {
        if let Some(id) = provided {
            let id = id.to_string();
            self.current_text_id = Some(id.clone());
            return id;
        }
        if let Some(current) = self.current_text_id.clone() {
            return current;
        }
        self.text_counter += 1;
        let id = format!("text-{}", self.text_counter);
        self.current_text_id = Some(id.clone());
        id
    }

    fn ensure_reasoning_id(&mut self, provided: Option<&str>) -> String {
        if let Some(id) = provided {
            let id = id.to_string();
            self.current_reasoning_id = Some(id.clone());
            return id;
        }
        if let Some(current) = self.current_reasoning_id.clone() {
            return current;
        }
        self.reasoning_counter += 1;
        let id = format!("reasoning-{}", self.reasoning_counter);
        self.current_reasoning_id = Some(id.clone());
        id
    }
}

fn provider_metadata_from_value(value: Option<&JsonValue>) -> Option<v2t::ProviderMetadata> {
    value.and_then(parse_provider_metadata)
}

fn parse_provider_metadata(value: &JsonValue) -> Option<v2t::ProviderMetadata> {
    let obj = value.as_object()?;
    let mut out: v2t::ProviderMetadata = HashMap::new();
    for (k, v) in obj {
        if let Some(inner) = v.as_object() {
            let mut inner_map: HashMap<String, JsonValue> = HashMap::new();
            for (ik, iv) in inner {
                inner_map.insert(ik.clone(), iv.clone());
            }
            out.insert(k.clone(), inner_map);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn parse_finish_reason(value: Option<&JsonValue>) -> v2t::FinishReason {
    if let Some(val) = value {
        if let Some(s) = val.as_str() {
            return match s.to_ascii_lowercase().as_str() {
                "stop" => v2t::FinishReason::Stop,
                "length" => v2t::FinishReason::Length,
                "content_filter" | "content-filter" => v2t::FinishReason::ContentFilter,
                "tool-calls" | "tool_calls" => v2t::FinishReason::ToolCalls,
                "error" => v2t::FinishReason::Error,
                _ => v2t::FinishReason::Other,
            };
        }
    }
    v2t::FinishReason::Unknown
}

fn parse_usage(value: Option<&JsonValue>) -> v2t::Usage {
    let mut usage = v2t::Usage::default();
    if let Some(JsonValue::Object(map)) = value {
        if let Some(v) = map.get("prompt_tokens").and_then(|v| v.as_u64()) {
            usage.input_tokens = Some(v);
        }
        if let Some(v) = map.get("completion_tokens").and_then(|v| v.as_u64()) {
            usage.output_tokens = Some(v);
        }
        if let Some(v) = map.get("total_tokens").and_then(|v| v.as_u64()) {
            usage.total_tokens = Some(v);
        }
        if let Some(v) = map.get("reasoning_tokens").and_then(|v| v.as_u64()) {
            usage.reasoning_tokens = Some(v);
        }
        if let Some(v) = map.get("cached_input_tokens").and_then(|v| v.as_u64()) {
            usage.cached_input_tokens = Some(v);
        }
    }
    usage
}

fn parse_call_warnings(value: &JsonValue) -> Vec<v2t::CallWarning> {
    let mut warnings = Vec::new();
    if let Some(array) = value.as_array() {
        for item in array {
            if let Some(kind) = item.get("type").and_then(|v| v.as_str()) {
                match kind {
                    "unsupported-setting" => {
                        if let Some(setting) = item.get("setting").and_then(|v| v.as_str()) {
                            warnings.push(v2t::CallWarning::UnsupportedSetting {
                                setting: setting.to_string(),
                                details: item
                                    .get("details")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string()),
                            });
                        }
                    }
                    "unsupported-tool" => {
                        if let Some(tool) = item
                            .get("tool")
                            .and_then(|t| t.get("name"))
                            .and_then(|v| v.as_str())
                        {
                            warnings.push(v2t::CallWarning::UnsupportedTool {
                                tool_name: tool.to_string(),
                                details: item
                                    .get("details")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string()),
                            });
                        }
                    }
                    "other" => {
                        if let Some(message) = item.get("message").and_then(|v| v.as_str()) {
                            warnings.push(v2t::CallWarning::Other {
                                message: message.to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    warnings
}

fn parse_response_metadata(value: &JsonValue) -> v2t::ResponseMetadata {
    let mut meta = v2t::ResponseMetadata::default();
    if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
        meta.id = Some(id.to_string());
    }
    if let Some(model_id) = value.get("modelId").and_then(|v| v.as_str()) {
        meta.model_id = Some(model_id.to_string());
    }
    if let Some(ts_val) = value.get("timestamp") {
        if let Some(ts_str) = ts_val.as_str() {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts_str) {
                meta.timestamp_ms = Some(dt.timestamp_millis());
            }
        } else if let Some(ms) = ts_val.as_i64() {
            meta.timestamp_ms = Some(ms);
        }
    }
    meta
}

fn content_from_value(value: Option<&JsonValue>) -> Result<Vec<v2t::Content>, SdkError> {
    match value {
        Some(JsonValue::Array(arr)) => {
            serde_json::from_value(JsonValue::Array(arr.clone())).map_err(SdkError::from)
        }
        Some(val) => serde_json::from_value(val.clone())
            .map(|single| vec![single])
            .map_err(SdkError::from),
        None => Ok(vec![]),
    }
}

#[cfg(test)]
mod tests {
    use super::decode_gateway_stream;
    use crate::ai_sdk_core::SdkError;
    use crate::ai_sdk_types::v2 as v2t;
    use bytes::Bytes;
    use futures_util::{stream, TryStreamExt};
    use serde_json::json;

    fn sse_chunk(value: serde_json::Value) -> Result<Bytes, SdkError> {
        Ok(Bytes::from(format!("data: {value}\n\n")))
    }

    fn gateway_stream_fixture() -> Vec<Result<Bytes, SdkError>> {
        vec![
            sse_chunk(json!({
                "type": "stream-start",
                "warnings": [{"type": "other", "message": "gateway warning"}]
            })),
            sse_chunk(json!({
                "type": "response-metadata",
                "id": "resp-gateway-1",
                "modelId": "gateway-model",
                "timestamp": "2026-01-02T03:04:05Z"
            })),
            sse_chunk(json!({
                "type": "reasoning-delta",
                "delta": "thinking",
                "providerMetadata": {"gateway": {"phase": "reasoning"}}
            })),
            sse_chunk(json!({
                "type": "text-delta",
                "delta": "hello",
                "providerMetadata": {"gateway": {"phase": "text"}}
            })),
            sse_chunk(json!({
                "type": "tool-input-start",
                "id": "call-1",
                "toolName": "weather",
                "providerExecuted": false
            })),
            sse_chunk(json!({
                "type": "tool-input-delta",
                "id": "call-1",
                "delta": "{\"city\":\"SF\"}",
                "providerExecuted": false
            })),
            sse_chunk(json!({
                "type": "tool-input-end",
                "id": "call-1",
                "providerExecuted": false
            })),
            sse_chunk(json!({
                "type": "tool-call",
                "toolCallId": "call-1",
                "toolName": "weather",
                "input": {"city": "SF"},
                "providerExecuted": false
            })),
            sse_chunk(json!({
                "type": "raw",
                "rawValue": {"upstream": "frame-9"}
            })),
            sse_chunk(json!({
                "type": "finish",
                "finishReason": "tool-calls",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5
                },
                "providerMetadata": {"gateway": {"finishSource": "gateway"}}
            })),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ]
    }

    fn assert_gateway_stream_start(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::StreamStart { warnings }
                if warnings.len() == 1
                    && matches!(
                        warnings[0],
                        v2t::CallWarning::Other { ref message } if message == "gateway warning"
                    )
        ));
    }

    fn assert_gateway_response_metadata(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ResponseMetadata { meta }
                if meta.id.as_deref() == Some("resp-gateway-1")
                    && meta.model_id.as_deref() == Some("gateway-model")
        ));
    }

    fn assert_gateway_reasoning_start(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ReasoningStart { id, provider_metadata }
                if id == "reasoning-1"
                    && provider_metadata
                        .as_ref()
                        .and_then(|meta| meta.get("gateway"))
                        .and_then(|inner| inner.get("phase"))
                        == Some(&json!("reasoning"))
        ));
    }

    fn assert_gateway_reasoning_delta(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ReasoningDelta { id, delta, .. }
                if id == "reasoning-1" && delta == "thinking"
        ));
    }

    fn assert_gateway_text_start(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::TextStart { id, provider_metadata }
                if id == "text-1"
                    && provider_metadata
                        .as_ref()
                        .and_then(|meta| meta.get("gateway"))
                        .and_then(|inner| inner.get("phase"))
                        == Some(&json!("text"))
        ));
    }

    fn assert_gateway_text_delta(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::TextDelta { id, delta, .. }
                if id == "text-1" && delta == "hello"
        ));
    }

    fn assert_gateway_tool_input_start(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ToolInputStart { id, tool_name, provider_executed, .. }
                if id == "call-1" && tool_name == "weather" && !provider_executed
        ));
    }

    fn assert_gateway_tool_input_delta(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ToolInputDelta { id, delta, provider_executed, .. }
                if id == "call-1" && delta == "{\"city\":\"SF\"}" && !provider_executed
        ));
    }

    fn assert_gateway_tool_input_end(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ToolInputEnd { id, provider_executed, .. }
                if id == "call-1" && !provider_executed
        ));
    }

    fn assert_gateway_tool_call(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::ToolCall(call)
                if call.tool_call_id == "call-1"
                    && call.tool_name == "weather"
                    && call.input == "{\"city\":\"SF\"}"
                    && !call.provider_executed
        ));
    }

    fn assert_gateway_raw(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::Raw { raw_value }
                if raw_value == &json!({"upstream": "frame-9"})
        ));
    }

    fn assert_gateway_finish(part: &v2t::StreamPart) {
        assert!(matches!(
            part,
            v2t::StreamPart::Finish {
                usage,
                finish_reason,
                provider_metadata,
            } if usage.input_tokens == Some(2)
                && usage.output_tokens == Some(3)
                && usage.total_tokens == Some(5)
                && matches!(finish_reason, v2t::FinishReason::ToolCalls)
                && provider_metadata
                    .as_ref()
                    .and_then(|meta| meta.get("gateway"))
                    .and_then(|inner| inner.get("finishSource"))
                    == Some(&json!("gateway"))
        ));
    }

    fn assert_gateway_stream_parts(parts: &[v2t::StreamPart]) {
        assert_gateway_stream_start(&parts[0]);
        assert_gateway_response_metadata(&parts[1]);
        assert_gateway_reasoning_start(&parts[2]);
        assert_gateway_reasoning_delta(&parts[3]);
        assert_gateway_text_start(&parts[4]);
        assert_gateway_text_delta(&parts[5]);
        assert_gateway_tool_input_start(&parts[6]);
        assert_gateway_tool_input_delta(&parts[7]);
        assert_gateway_tool_input_end(&parts[8]);
        assert_gateway_tool_call(&parts[9]);
        assert_gateway_raw(&parts[10]);
        assert_gateway_finish(&parts[11]);
    }

    #[tokio::test]
    async fn decode_gateway_stream_normalizes_text_reasoning_tool_raw_and_finish() {
        let parts: Vec<v2t::StreamPart> =
            decode_gateway_stream(stream::iter(gateway_stream_fixture()), true)
                .try_collect()
                .await
                .expect("gateway stream parts");

        assert_gateway_stream_parts(&parts);
    }
}
