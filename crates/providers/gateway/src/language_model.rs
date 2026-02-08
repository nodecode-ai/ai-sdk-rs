use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ai_sdk_core::request_builder::defaults::{build_call_options, request_overrides_from_json};
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{GenerateResponse, LanguageModel, PartStream, SdkError, StreamResponse};
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
                crate::ai_sdk_core::options::merge_options_with_disallow(&mut body, &overrides, &disallow);
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
                crate::ai_sdk_core::options::merge_options_with_disallow(&mut body, &overrides, &disallow);
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
                if event.data.is_empty() {
                    continue;
                }
                if event.data.as_ref() == b"[DONE]" {
                    continue;
                }

                let raw_value = match serde_json::from_slice::<JsonValue>(&event.data) {
                    Ok(v) => v,
                    Err(_) => {
                        if include_raw {
                            yield v2t::StreamPart::Raw {
                                raw_value: JsonValue::String(String::from_utf8_lossy(&event.data).to_string()),
                            };
                        }
                        continue;
                    }
                };

                for part in state.process_chunk(raw_value.clone(), include_raw) {
                    yield part;
                }
            }
        }

        for event in decoder.finish() {
            if event.data.is_empty() {
                continue;
            }
            if event.data.as_ref() == b"[DONE]" {
                continue;
            }
            match serde_json::from_slice::<JsonValue>(&event.data) {
                Ok(raw_value) => {
                    for part in state.process_chunk(raw_value, include_raw) {
                        yield part;
                    }
                }
                Err(_) => {
                    if include_raw {
                        yield v2t::StreamPart::Raw {
                            raw_value: JsonValue::String(
                                String::from_utf8_lossy(&event.data).to_string(),
                            ),
                        };
                    }
                }
            }
        }
    })
}

#[derive(Default)]
struct GatewayStreamState {
    stream_started: bool,
    text_counter: usize,
    current_text_id: Option<String>,
    active_text: HashSet<String>,
    reasoning_counter: usize,
    current_reasoning_id: Option<String>,
    active_reasoning: HashSet<String>,
    finished_emitted: bool,
}

impl GatewayStreamState {
    fn process_chunk(&mut self, value: JsonValue, include_raw: bool) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        let chunk_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if chunk_type == "stream-start" {
            self.stream_started = true;
            let warnings = parse_call_warnings(value.get("warnings").unwrap_or(&JsonValue::Null));
            parts.push(v2t::StreamPart::StreamStart { warnings });
            return parts;
        }

        if !self.stream_started {
            self.stream_started = true;
            parts.push(v2t::StreamPart::StreamStart {
                warnings: Vec::new(),
            });
        }

        match chunk_type {
            "text-start" => {
                let id = self.ensure_text_id(value.get("id").and_then(|v| v.as_str()));
                if self.active_text.insert(id.clone()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::TextStart {
                        id,
                        provider_metadata: metadata,
                    });
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
                if self.active_text.insert(id.clone()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::TextStart {
                        id: id.clone(),
                        provider_metadata: metadata,
                    });
                }
                let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                parts.push(v2t::StreamPart::TextDelta {
                    id,
                    delta,
                    provider_metadata: metadata,
                });
            }
            "text-end" => {
                let id = self.ensure_text_id(value.get("id").and_then(|v| v.as_str()));
                if self.active_text.remove(&id) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::TextEnd {
                        id,
                        provider_metadata: metadata,
                    });
                }
                self.current_text_id = None;
            }
            "reasoning-start" => {
                let id = self.ensure_reasoning_id(value.get("id").and_then(|v| v.as_str()));
                if self.active_reasoning.insert(id.clone()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::ReasoningStart {
                        id,
                        provider_metadata: metadata,
                    });
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
                if self.active_reasoning.insert(id.clone()) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::ReasoningStart {
                        id: id.clone(),
                        provider_metadata: metadata,
                    });
                }
                let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                parts.push(v2t::StreamPart::ReasoningDelta {
                    id,
                    delta,
                    provider_metadata: metadata,
                });
            }
            "reasoning-end" => {
                let id = self.ensure_reasoning_id(value.get("id").and_then(|v| v.as_str()));
                if self.active_reasoning.remove(&id) {
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::ReasoningEnd {
                        id,
                        provider_metadata: metadata,
                    });
                }
                self.current_reasoning_id = None;
            }
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
                    parts.push(v2t::StreamPart::ToolInputStart {
                        id: id.to_string(),
                        tool_name,
                        provider_executed,
                        provider_metadata: metadata,
                    });
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
                    parts.push(v2t::StreamPart::ToolInputDelta {
                        id: id.to_string(),
                        delta,
                        provider_executed,
                        provider_metadata: metadata,
                    });
                }
            }
            "tool-input-end" => {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    let provider_executed = value
                        .get("providerExecuted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                    parts.push(v2t::StreamPart::ToolInputEnd {
                        id: id.to_string(),
                        provider_executed,
                        provider_metadata: metadata,
                    });
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
                    parts.push(v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                        tool_call_id: tool_call_id.to_string(),
                        tool_name,
                        input: input_json,
                        provider_executed,
                        provider_metadata: None,
                        dynamic: false,
                        provider_options: None,
                    }));
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
            "file" => {
                if let (Some(media_type), Some(data_str)) = (
                    value.get("mediaType").and_then(|v| v.as_str()),
                    value.get("data").and_then(|v| v.as_str()),
                ) {
                    parts.push(v2t::StreamPart::File {
                        media_type: media_type.to_string(),
                        data: data_str.to_string(),
                    });
                } else if include_raw {
                    parts.push(v2t::StreamPart::Raw { raw_value: value });
                }
            }
            "source" => {
                if let Some(source_type) = value.get("sourceType").and_then(|v| v.as_str()) {
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
                                let metadata =
                                    provider_metadata_from_value(value.get("providerMetadata"));
                                parts.push(v2t::StreamPart::SourceUrl {
                                    id: id.to_string(),
                                    url: url.to_string(),
                                    title,
                                    provider_metadata: metadata,
                                });
                            }
                        }
                        _ => {
                            if include_raw {
                                parts.push(v2t::StreamPart::Raw { raw_value: value });
                            }
                        }
                    }
                }
            }
            "response-metadata" => {
                let meta = parse_response_metadata(&value);
                parts.push(v2t::StreamPart::ResponseMetadata { meta });
            }
            "finish" => {
                if self.finished_emitted {
                    return parts;
                }
                let usage = parse_usage(value.get("usage"));
                let finish_reason = parse_finish_reason(
                    value
                        .get("finishReason")
                        .or_else(|| value.get("finish_reason")),
                );
                let metadata = provider_metadata_from_value(value.get("providerMetadata"));
                parts.push(v2t::StreamPart::Finish {
                    usage,
                    finish_reason,
                    provider_metadata: metadata,
                });
                self.finished_emitted = true;
            }
            "error" => {
                let payload = value
                    .get("error")
                    .cloned()
                    .unwrap_or(JsonValue::String("Gateway error".into()));
                parts.push(v2t::StreamPart::Error { error: payload });
            }
            "raw" => {
                if include_raw {
                    if let Some(raw_value) = value.get("rawValue").cloned() {
                        parts.push(v2t::StreamPart::Raw { raw_value });
                    }
                }
            }
            _ => {
                if include_raw {
                    parts.push(v2t::StreamPart::Raw { raw_value: value });
                }
            }
        }
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
