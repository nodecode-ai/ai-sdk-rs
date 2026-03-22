use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{SdkError, StreamNormalizationState, StreamResponse};
use crate::ai_sdk_streaming_sse::SseDecoder;
use crate::ai_sdk_types::json::parse_json_loose;
use crate::ai_sdk_types::v2 as v2t;
use async_stream::try_stream;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::provider_openai_compatible::completion::finish_reason::map_openai_compatible_finish_reason;
use crate::provider_openai_compatible::error::map_transport_error_to_sdk_error;

#[derive(Clone, Copy)]
pub enum StreamMode {
    Chat,
    Completion,
}

pub struct StreamSettings {
    pub warnings: Vec<v2t::CallWarning>,
    pub include_raw: bool,
    pub include_usage: bool,
    pub provider_scope_name: String,
}

#[derive(Default)]
struct ToolCallState {
    id: Option<String>,
    name: Option<String>,
    finished: bool,
    started: bool,
}

struct ChatState {
    normalizer: StreamNormalizationState<()>,
    tool_calls: Vec<ToolCallState>,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            normalizer: StreamNormalizationState::new(()),
            tool_calls: Vec::new(),
        }
    }
}

pub fn build_stream<S>(
    bytes_stream: S,
    settings: StreamSettings,
    mode: StreamMode,
) -> crate::ai_sdk_core::PartStream
where
    S: Stream<Item = Result<Bytes, SdkError>> + Send + 'static,
{
    Box::pin(try_stream! {
        yield v2t::StreamPart::StreamStart { warnings: settings.warnings.clone() };
        let mut decoder = SseDecoder::new();
        let mut usage = v2t::Usage::default();
        let mut finish_reason = v2t::FinishReason::Unknown;
        let mut first_chunk = true;
        let mut chat_state = ChatState::default();
        let mut completion_state = StreamNormalizationState::new(());
        let mut completion_started = false;
        let mut provider_metadata: Option<v2t::ProviderMetadata> = None;

        macro_rules! handle_sse_event {
            ($ev:expr) => {{
                let ev = $ev;
                if ev.data.as_ref() == b"[DONE]" {
                    for part in emit_finish(
                        mode,
                        &mut chat_state,
                        &mut completion_state,
                        completion_started,
                        usage.clone(),
                        finish_reason,
                        provider_metadata.clone(),
                    ) {
                        yield part;
                    }
                    return;
                }

                let val: JsonValue = match serde_json::from_slice(&ev.data) {
                    Ok(v) => v,
                    Err(_) => {
                        yield v2t::StreamPart::Error {
                            error: serde_json::json!({"message":"invalid json chunk"}),
                        };
                        continue;
                    }
                };

                if settings.include_raw {
                    yield v2t::StreamPart::Raw { raw_value: val.clone() };
                }

                if first_chunk {
                    first_chunk = false;
                    yield v2t::StreamPart::ResponseMetadata {
                        meta: response_metadata_from_chunk(&val),
                    };
                    if matches!(mode, StreamMode::Completion) {
                        for part in completion_state.open_text("0".into(), None) {
                            yield part;
                        }
                        completion_started = true;
                    }
                }

                update_usage(
                    settings.include_usage,
                    &mut usage,
                    &mut provider_metadata,
                    &settings.provider_scope_name,
                    &val,
                );

                let parts = match mode {
                    StreamMode::Chat => match handle_chat_delta(&val, &mut chat_state, &mut finish_reason) {
                        Ok(parts) => parts,
                        Err(err) => {
                            yield err.into_part();
                            return;
                        }
                    },
                    StreamMode::Completion => {
                        handle_completion_delta(&val, &mut completion_state, &mut finish_reason)
                    }
                };
                for part in parts {
                    yield part;
                }
            }};
        }

        futures_util::pin_mut!(bytes_stream);
        while let Some(chunk_res) = bytes_stream.next().await {
            let chunk = match chunk_res {
                Ok(b) => b,
                Err(te) => {
                    yield v2t::StreamPart::Error { error: serde_json::json!({"message": te.to_string()}) };
                    break;
                }
            };

            for ev in decoder.push(&chunk) {
                handle_sse_event!(ev);
            }
        }

        for ev in decoder.finish() {
            handle_sse_event!(ev);
        }

        for part in emit_finish(
            mode,
            &mut chat_state,
            &mut completion_state,
            completion_started,
            usage,
            finish_reason,
            provider_metadata,
        ) {
            yield part;
        }
    })
}

fn emit_finish(
    mode: StreamMode,
    state: &mut ChatState,
    completion_state: &mut StreamNormalizationState<()>,
    completion_started: bool,
    usage: v2t::Usage,
    finish_reason: v2t::FinishReason,
    provider_metadata: Option<v2t::ProviderMetadata>,
) -> Vec<v2t::StreamPart> {
    let mut parts = Vec::new();
    match mode {
        StreamMode::Chat => {
            state.normalizer.usage = usage;
            for tool_call in state
                .tool_calls
                .iter_mut()
                .filter(|tc| tc.started && !tc.finished)
            {
                if let Some(id) = &tool_call.id {
                    parts.extend(state.normalizer.finish_tool_call(
                        id.clone(),
                        false,
                        None,
                        None,
                        false,
                        None,
                    ));
                    tool_call.finished = true;
                }
            }
            if let Some(part) = state.normalizer.close_reasoning(None) {
                parts.push(part);
            }
            if let Some(part) = state.normalizer.close_text(None) {
                parts.push(part);
            }
            parts.push(
                state
                    .normalizer
                    .finish_part(finish_reason, provider_metadata),
            );
        }
        StreamMode::Completion => {
            if completion_started {
                completion_state.usage = usage;
                parts.extend(completion_state.finish_stream(
                    Some((finish_reason, provider_metadata)),
                    v2t::FinishReason::Unknown,
                ));
            }
        }
    }
    parts
}

pub async fn start_streaming<T: HttpTransport + Send + Sync>(
    http: &T,
    url: String,
    mut headers: Vec<(String, String)>,
    mut body: JsonValue,
    transport_cfg: &TransportConfig,
    settings: StreamSettings,
    mode: StreamMode,
) -> Result<StreamResponse, SdkError> {
    if let Some(map) = body.as_object_mut() {
        map.insert("stream".into(), JsonValue::Bool(true));
        if settings.include_usage {
            map.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": true}),
            );
        }
    }

    // Ensure content-type for SSE
    if !headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
    {
        headers.push(("content-type".into(), "application/json".into()));
    }

    let resp = http
        .post_json_stream(&url, &headers, &body, transport_cfg)
        .await
        .map_err(map_transport_error_to_sdk_error)?;

    let (bytes_stream, resp_headers) = <T as HttpTransport>::into_stream(resp);
    let headers_map: std::collections::HashMap<String, String> = resp_headers.into_iter().collect();

    let mapped_stream = bytes_stream.map(|res| res.map_err(map_transport_error_to_sdk_error));
    let part_stream = build_stream(mapped_stream, settings, mode);
    Ok(StreamResponse {
        stream: part_stream,
        request_body: Some(body),
        response_headers: Some(headers_map),
    })
}

fn update_usage(
    include_usage: bool,
    usage: &mut v2t::Usage,
    provider_metadata: &mut Option<v2t::ProviderMetadata>,
    provider_scope_name: &str,
    val: &JsonValue,
) {
    if !include_usage {
        return;
    }
    if let Some(u) = val.get("usage") {
        if let Some(u2) = crate::ai_sdk_types::usage::from_openai(u) {
            usage.input_tokens = Some(u2.input_tokens as u64);
            usage.output_tokens = Some(u2.output_tokens as u64);
            usage.total_tokens = Some(u2.total_tokens as u64);
            if let Some(v) = u2.cache_read_tokens {
                usage.cached_input_tokens = Some(v as u64);
            }
        }

        if let Some(cache_read) = u
            .get("prompt_tokens_details")
            .and_then(|v| v.get("cached_tokens"))
            .and_then(|v| v.as_u64())
        {
            usage.cached_input_tokens = Some(cache_read);
        }

        if let Some(reasoning) = u
            .get("completion_tokens_details")
            .and_then(|v| v.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
        {
            usage.reasoning_tokens = Some(reasoning);
        }

        if let Some(accepted) = u
            .get("completion_tokens_details")
            .and_then(|v| v.get("accepted_prediction_tokens"))
            .and_then(|v| v.as_u64())
        {
            set_provider_metadata_value(
                provider_metadata,
                provider_scope_name,
                "acceptedPredictionTokens",
                accepted,
            );
        }
        if let Some(rejected) = u
            .get("completion_tokens_details")
            .and_then(|v| v.get("rejected_prediction_tokens"))
            .and_then(|v| v.as_u64())
        {
            set_provider_metadata_value(
                provider_metadata,
                provider_scope_name,
                "rejectedPredictionTokens",
                rejected,
            );
        }
    }
}

fn set_provider_metadata_value(
    provider_metadata: &mut Option<v2t::ProviderMetadata>,
    provider_scope_name: &str,
    key: &str,
    value: u64,
) {
    let outer = provider_metadata.get_or_insert_with(HashMap::new);
    let inner = outer
        .entry(provider_scope_name.to_string())
        .or_insert_with(HashMap::new);
    inner.insert(key.to_string(), serde_json::json!(value));
}

fn handle_completion_delta(
    val: &JsonValue,
    state: &mut StreamNormalizationState<()>,
    finish_reason: &mut v2t::FinishReason,
) -> Vec<v2t::StreamPart> {
    let mut parts = Vec::new();
    if let Some(choice0) = val
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.get(0))
    {
        if let Some(fr) = choice0.get("finish_reason").and_then(|v| v.as_str()) {
            *finish_reason = map_openai_compatible_finish_reason(Some(fr));
        }
        if let Some(text) = choice0.get("text").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                parts.extend(state.push_text_delta(
                    Some("0".into()),
                    "0",
                    text.to_string(),
                    None,
                    None,
                ));
            }
        }
    }
    parts
}

fn handle_chat_delta(
    val: &JsonValue,
    state: &mut ChatState,
    finish_reason: &mut v2t::FinishReason,
) -> Result<Vec<v2t::StreamPart>, ToolDeltaError> {
    let mut parts = Vec::new();
    let Some(choice0) = val
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.get(0))
    else {
        return Ok(parts);
    };

    if let Some(fr) = choice0.get("finish_reason").and_then(|v| v.as_str()) {
        *finish_reason = map_openai_compatible_finish_reason(Some(fr));
    }

    let Some(delta) = choice0.get("delta").and_then(|d| d.as_object()) else {
        return Ok(parts);
    };

    if let Some(reasoning) = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(|v| v.as_str())
    {
        if state.normalizer.reasoning_open.is_none() {
            parts.extend(state.normalizer.open_reasoning("reasoning-0".into(), None));
        }
        if !reasoning.is_empty() {
            parts.push(state.normalizer.push_reasoning_delta(
                "reasoning-0",
                reasoning.to_string(),
                None,
            ));
        }
    }

    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
        let text_id = "txt-0".to_string();
        if state.normalizer.text_open.is_none() {
            parts.extend(state.normalizer.open_text(text_id.clone(), None));
        }
        if !text.is_empty() {
            parts.extend(state.normalizer.push_text_delta(
                Some(text_id),
                "txt-0",
                text.to_string(),
                None,
                None,
            ));
        }
    }

    if let Some(tc_arr) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tc_arr {
            let index = tc
                .get("index")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ToolDeltaError::new("Expected 'index' to be a number."))?
                as usize;
            if state.tool_calls.len() <= index {
                state
                    .tool_calls
                    .resize_with(index + 1, ToolCallState::default);
            }
            let slot = &mut state.tool_calls[index];
            let func = tc
                .get("function")
                .and_then(|v| v.as_object())
                .ok_or_else(|| ToolDeltaError::new("Expected 'function.name' to be a string."))?;
            let args_fragment = func.get("arguments").and_then(|v| v.as_str());

            if !slot.started {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| ToolDeltaError::new("Expected 'id' to be a string."))?;
                let name = func.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
                    ToolDeltaError::new("Expected 'function.name' to be a string.")
                })?;

                slot.id = Some(id.to_string());
                slot.name = Some(name.to_string());
                slot.started = true;

                parts.push(state.normalizer.start_tool_call(
                    id.to_string(),
                    name.to_string(),
                    false,
                    None,
                ));

                if let Some(fragment) = args_fragment {
                    if fragment.is_empty() {
                        let normalized_args = state
                            .normalizer
                            .tool_args
                            .get(id)
                            .map(String::as_str)
                            .unwrap_or("");
                        if !slot.finished && parse_json_loose(normalized_args).is_some() {
                            parts.extend(state.normalizer.finish_tool_call(
                                id.to_string(),
                                false,
                                None,
                                None,
                                false,
                                None,
                            ));
                            slot.finished = true;
                        }
                        continue;
                    }
                    parts.push(state.normalizer.push_tool_call_delta(
                        id.to_string(),
                        fragment.to_string(),
                        false,
                        None,
                    ));
                }

                let normalized_args = state
                    .normalizer
                    .tool_args
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or("");
                if !slot.finished && parse_json_loose(normalized_args).is_some() {
                    parts.extend(state.normalizer.finish_tool_call(
                        id.to_string(),
                        false,
                        None,
                        None,
                        false,
                        None,
                    ));
                    slot.finished = true;
                }
                continue;
            }

            let id = slot
                .id
                .clone()
                .ok_or_else(|| ToolDeltaError::new("Expected 'id' to be a string."))?;
            let name = slot
                .name
                .clone()
                .ok_or_else(|| ToolDeltaError::new("Expected 'function.name' to be a string."))?;
            parts.push(state.normalizer.push_tool_call_delta(
                id.clone(),
                args_fragment.unwrap_or("").to_string(),
                false,
                None,
            ));
            let normalized_args = state
                .normalizer
                .tool_args
                .get(&id)
                .map(String::as_str)
                .unwrap_or("");
            if !slot.finished && parse_json_loose(normalized_args).is_some() {
                let _ = name;
                parts.extend(
                    state
                        .normalizer
                        .finish_tool_call(id, false, None, None, false, None),
                );
                slot.finished = true;
            }
        }
    }

    Ok(parts)
}

fn response_metadata_from_chunk(val: &JsonValue) -> v2t::ResponseMetadata {
    let id = val
        .get("id")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let model_id = val
        .get("model")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let created = val.get("created").and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_u64().and_then(|u| i64::try_from(u).ok()))
    });
    let timestamp_ms = created.map(|v| v.saturating_mul(1000));
    v2t::ResponseMetadata {
        id,
        timestamp_ms,
        model_id,
    }
}

struct ToolDeltaError {
    message: String,
}

impl ToolDeltaError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn into_part(self) -> v2t::StreamPart {
        v2t::StreamPart::Error {
            error: serde_json::json!({ "message": self.message }),
        }
    }
}
