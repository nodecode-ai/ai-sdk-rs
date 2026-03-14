use std::collections::{HashMap, HashSet};
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value as JsonValue;

use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::{PartStream, StreamNormalizationState};
use crate::ai_sdk_streaming_sse::SseDecoder;
use crate::ai_sdk_types::v2 as v2t;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>;

pub fn build_google_stream_part_stream(
    mut inner: ByteStream,
    warnings: Vec<v2t::CallWarning>,
    include_raw: bool,
    provider_scope: &'static str,
) -> PartStream {
    Box::pin(async_stream::try_stream! {
        use serde_json::json;
        yield v2t::StreamPart::StreamStart { warnings };
        let mut decoder = SseDecoder::new();
        let mut normalizer = StreamNormalizationState::new(());
        let mut last_code_tool_id: Option<String> = None;
        let mut emitted_source_urls: HashSet<String> = HashSet::new();
        let mut finish_reason: v2t::FinishReason = v2t::FinishReason::Unknown;
        let mut provider_metadata: Option<v2t::ProviderMetadata> = None;
        let mut block_counter: u64 = 0;

        macro_rules! handle_sse_event {
            ($ev:expr) => {{
                let ev = $ev;
                let data = String::from_utf8_lossy(&ev.data).to_string();
                if data.trim().is_empty() {
                    continue;
                }
                let parsed: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if include_raw {
                    yield v2t::StreamPart::Raw {
                        raw_value: parsed.clone(),
                    };
                }

                if let Some(u) = parsed.get("usageMetadata") {
                    normalizer.usage = v2t::Usage {
                        input_tokens: u.get("promptTokenCount").and_then(|v| v.as_u64()),
                        output_tokens: u.get("candidatesTokenCount").and_then(|v| v.as_u64()),
                        total_tokens: u.get("totalTokenCount").and_then(|v| v.as_u64()),
                        reasoning_tokens: u.get("thoughtsTokenCount").and_then(|v| v.as_u64()),
                        cached_input_tokens: u.get("cachedContentTokenCount").and_then(|v| v.as_u64()),
                    };
                }

                if let Some(cands) = parsed.get("candidates").and_then(|v| v.as_array()) {
                    for cand in cands {
                        if let Some(content) = cand.get("content").and_then(|v| v.as_object()) {
                            let parts = content.get("parts").and_then(|p| p.as_array()).cloned().unwrap_or_default();
                            for p in parts {
                                if let Some(ec) = p.get("executableCode").and_then(|v| v.as_object()) {
                                    if ec.get("code").and_then(|v| v.as_str()).is_some() {
                                        let id = uuid::Uuid::new_v4().to_string();
                                        last_code_tool_id = Some(id.clone());
                                        yield normalizer.start_tool_call(
                                            id.clone(),
                                            "code_execution".into(),
                                            true,
                                            None,
                                        );
                                        let delta = serde_json::to_string(ec).unwrap_or("{}".into());
                                        yield normalizer.push_tool_call_delta(
                                            id.clone(),
                                            delta,
                                            true,
                                            None,
                                        );
                                        for part in normalizer.finish_tool_call(
                                            id,
                                            true,
                                            None,
                                            None,
                                            false,
                                            None,
                                        ) {
                                            yield part;
                                        }
                                        continue;
                                    }
                                }
                                if let Some(res) = p.get("codeExecutionResult").and_then(|v| v.as_object()) {
                                    if let Some(id) = last_code_tool_id.take() {
                                        yield v2t::StreamPart::ToolResult { tool_call_id: id, tool_name: "code_execution".into(), result: json!({"outcome": res.get("outcome"), "output": res.get("output")}), is_error: false, preliminary: false, provider_metadata: None };
                                        continue;
                                    }
                                }
                                if let Some(txt) = p.get("text").and_then(|v| v.as_str()) {
                                    if !txt.is_empty() {
                                        let is_thought = p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                                        let thought_sig = p.get("thoughtSignature").and_then(|v| v.as_str());
                                        let pm = thought_sig.map(|sig| {
                                            let mut outer = HashMap::new();
                                            let mut inner = HashMap::new();
                                            inner.insert("thoughtSignature".into(), JsonValue::String(sig.to_string()));
                                            outer.insert(provider_scope.into(), inner);
                                            outer
                                        });
                                        if is_thought {
                                            if let Some(part) = normalizer.close_text(None) {
                                                yield part;
                                            }
                                            if normalizer.reasoning_open.is_none() {
                                                let id = format!("r-{}", block_counter);
                                                block_counter += 1;
                                                for part in normalizer.open_reasoning(id, pm.clone()) {
                                                    yield part;
                                                }
                                            }
                                            yield normalizer.push_reasoning_delta(
                                                "reasoning-1",
                                                txt.to_string(),
                                                pm,
                                            );
                                        } else {
                                            if let Some(part) = normalizer.close_reasoning(None) {
                                                yield part;
                                            }
                                            let start_metadata = if normalizer.text_open.is_none() {
                                                pm.clone()
                                            } else {
                                                None
                                            };
                                            let id = normalizer.text_open.clone().unwrap_or_else(|| {
                                                let id = format!("t-{}", block_counter);
                                                block_counter += 1;
                                                id
                                            });
                                            for part in normalizer.push_text_delta(
                                                Some(id),
                                                "t-0",
                                                txt.to_string(),
                                                start_metadata,
                                                pm,
                                            ) {
                                                yield part;
                                            }
                                        }
                                        continue;
                                    }
                                }
                                if let Some(inline) = p.get("inlineData").and_then(|v| v.as_object()) {
                                    let media_type = inline.get("mimeType").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let data = inline.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    yield v2t::StreamPart::File { media_type, data };
                                    continue;
                                }
                                if let Some(fc) = p.get("functionCall").and_then(|v| v.as_object()) {
                                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let args = fc.get("args").cloned().unwrap_or(json!({})).to_string();
                                    let thought_sig = p.get("thoughtSignature").and_then(|v| v.as_str());
                                    let provider_options = thought_sig.map(|sig| {
                                        let mut outer = HashMap::new();
                                        let mut inner = HashMap::new();
                                        inner.insert("thoughtSignature".into(), JsonValue::String(sig.to_string()));
                                        outer.insert(provider_scope.into(), inner);
                                        outer
                                    });
                                    let id = uuid::Uuid::new_v4().to_string();
                                    yield normalizer.start_tool_call(
                                        id.clone(),
                                        name,
                                        false,
                                        None,
                                    );
                                    yield normalizer.push_tool_call_delta(
                                        id.clone(),
                                        args,
                                        false,
                                        None,
                                    );
                                    for part in normalizer.finish_tool_call(
                                        id,
                                        false,
                                        None,
                                        None,
                                        false,
                                        provider_options,
                                    ) {
                                        yield part;
                                    }
                                    continue;
                                }
                            }
                        }

                        if let Some(gm) = cand.get("groundingMetadata") {
                            if let Some(chunks) = gm.get("groundingChunks").and_then(|v| v.as_array()) {
                                for ch in chunks {
                                    if let Some(web) = ch.get("web").and_then(|v| v.as_object()) {
                                        if let Some(url) = web.get("uri").and_then(|v| v.as_str()) {
                                            if !emitted_source_urls.contains(url) {
                                                emitted_source_urls.insert(url.to_string());
                                                let id = uuid::Uuid::new_v4().to_string();
                                                let title = web.get("title").and_then(|v| v.as_str()).map(|s| s.to_string());
                                                yield v2t::StreamPart::SourceUrl { id, url: url.to_string(), title, provider_metadata: None };
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(fr) = cand.get("finishReason").and_then(|v| v.as_str()) {
                            finish_reason = match fr {
                                "STOP" => if normalizer.has_tool_calls { v2t::FinishReason::ToolCalls } else { v2t::FinishReason::Stop },
                                "MAX_TOKENS" => v2t::FinishReason::Length,
                                "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => v2t::FinishReason::ContentFilter,
                                "FINISH_REASON_UNSPECIFIED" | "OTHER" => v2t::FinishReason::Other,
                                "MALFORMED_FUNCTION_CALL" => v2t::FinishReason::Error,
                                _ => v2t::FinishReason::Unknown,
                            };
                            let mut inner_map = HashMap::new();
                            inner_map.insert("groundingMetadata".into(), cand.get("groundingMetadata").cloned().unwrap_or(JsonValue::Null));
                            inner_map.insert("urlContextMetadata".into(), cand.get("urlContextMetadata").cloned().unwrap_or(JsonValue::Null));
                            inner_map.insert("safetyRatings".into(), cand.get("safetyRatings").cloned().unwrap_or(JsonValue::Null));
                            let usage_has = normalizer.usage.input_tokens.is_some()
                                || normalizer.usage.output_tokens.is_some()
                                || normalizer.usage.total_tokens.is_some()
                                || normalizer.usage.reasoning_tokens.is_some()
                                || normalizer.usage.cached_input_tokens.is_some();
                            if usage_has {
                                inner_map.insert("usageMetadata".into(), json!({
                                    "promptTokenCount": normalizer.usage.input_tokens,
                                    "candidatesTokenCount": normalizer.usage.output_tokens,
                                    "totalTokenCount": normalizer.usage.total_tokens,
                                    "thoughtsTokenCount": normalizer.usage.reasoning_tokens,
                                    "cachedContentTokenCount": normalizer.usage.cached_input_tokens,
                                }));
                            }
                            let mut outer = HashMap::new();
                            outer.insert(provider_scope.into(), inner_map);
                            provider_metadata = Some(outer);
                        }
                    }
                }
            }};
        }

        while let Some(chunk_res) = inner.next().await {
            match chunk_res {
                Ok(chunk) => {
                    for ev in decoder.push(&chunk) {
                        handle_sse_event!(ev);
                    }
                }
                Err(te) => {
                    let e = crate::provider_google::shared::error::map_transport_error_to_sdk_error(te);
                    yield v2t::StreamPart::Error { error: serde_json::json!({"message": e.to_string()}) };
                    break;
                }
            }
        }

        for ev in decoder.finish() {
            handle_sse_event!(ev);
        }

        for part in normalizer.finish_stream(
            Some((finish_reason, provider_metadata)),
            v2t::FinishReason::Unknown,
        ) {
            yield part;
        }
    })
}
