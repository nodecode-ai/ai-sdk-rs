use std::collections::{HashMap, HashSet};
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value as JsonValue;

use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::PartStream;
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
        let mut current_text_id: Option<String> = None;
        let mut current_reasoning_id: Option<String> = None;
        let mut last_code_tool_id: Option<String> = None;
        let mut emitted_source_urls: HashSet<String> = HashSet::new();
        let mut usage: v2t::Usage = v2t::Usage::default();
        let mut finish_reason: v2t::FinishReason = v2t::FinishReason::Unknown;
        let mut provider_metadata: Option<v2t::ProviderMetadata> = None;
        let mut has_tool_calls: bool = false;
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
                    usage = v2t::Usage {
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
                                        yield v2t::StreamPart::ToolInputStart { id: id.clone(), tool_name: "code_execution".into(), provider_executed: true, provider_metadata: None };
                                        let delta = serde_json::to_string(ec).unwrap_or("{}".into());
                                        yield v2t::StreamPart::ToolInputDelta {
                                            id: id.clone(),
                                            delta,
                                            provider_executed: true,
                                            provider_metadata: None,
                                        };
                                        yield v2t::StreamPart::ToolInputEnd {
                                            id: id.clone(),
                                            provider_executed: true,
                                            provider_metadata: None,
                                        };
                                        yield v2t::StreamPart::ToolCall(v2t::ToolCallPart { tool_call_id: id, tool_name: "code_execution".into(), input: serde_json::to_string(ec).unwrap_or("{}".into()), provider_executed: true, provider_metadata: None, dynamic: false, provider_options: None });
                                        has_tool_calls = true;
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
                                            if let Some(id) = current_text_id.take() {
                                                yield v2t::StreamPart::TextEnd {
                                                    id,
                                                    provider_metadata: None,
                                                };
                                            }
                                            if current_reasoning_id.is_none() {
                                                let id = format!("r-{}", block_counter);
                                                block_counter += 1;
                                                current_reasoning_id = Some(id.clone());
                                                yield v2t::StreamPart::ReasoningStart {
                                                    id,
                                                    provider_metadata: pm.clone(),
                                                };
                                            }
                                            let id = current_reasoning_id.clone().unwrap();
                                            yield v2t::StreamPart::ReasoningDelta {
                                                id,
                                                delta: txt.to_string(),
                                                provider_metadata: pm,
                                            };
                                        } else {
                                            if let Some(id) = current_reasoning_id.take() {
                                                yield v2t::StreamPart::ReasoningEnd {
                                                    id,
                                                    provider_metadata: None,
                                                };
                                            }
                                            if current_text_id.is_none() {
                                                let id = format!("t-{}", block_counter);
                                                block_counter += 1;
                                                current_text_id = Some(id.clone());
                                                yield v2t::StreamPart::TextStart {
                                                    id,
                                                    provider_metadata: pm.clone(),
                                                };
                                            }
                                            let id = current_text_id.clone().unwrap();
                                            yield v2t::StreamPart::TextDelta {
                                                id,
                                                delta: txt.to_string(),
                                                provider_metadata: pm,
                                            };
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
                                    yield v2t::StreamPart::ToolInputStart { id: id.clone(), tool_name: name.clone(), provider_executed: false, provider_metadata: None };
                                    yield v2t::StreamPart::ToolInputDelta {
                                        id: id.clone(),
                                        delta: args.clone(),
                                        provider_executed: false,
                                        provider_metadata: None,
                                    };
                                    yield v2t::StreamPart::ToolInputEnd {
                                        id: id.clone(),
                                        provider_executed: false,
                                        provider_metadata: None,
                                    };
                                    yield v2t::StreamPart::ToolCall(v2t::ToolCallPart { tool_call_id: id, tool_name: name, input: args, provider_executed: false, provider_metadata: None, dynamic: false, provider_options });
                                    has_tool_calls = true;
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
                                "STOP" => if has_tool_calls { v2t::FinishReason::ToolCalls } else { v2t::FinishReason::Stop },
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
                            let usage_has = usage.input_tokens.is_some()
                                || usage.output_tokens.is_some()
                                || usage.total_tokens.is_some()
                                || usage.reasoning_tokens.is_some()
                                || usage.cached_input_tokens.is_some();
                            if usage_has {
                                inner_map.insert("usageMetadata".into(), json!({
                                    "promptTokenCount": usage.input_tokens,
                                    "candidatesTokenCount": usage.output_tokens,
                                    "totalTokenCount": usage.total_tokens,
                                    "thoughtsTokenCount": usage.reasoning_tokens,
                                    "cachedContentTokenCount": usage.cached_input_tokens,
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

        if let Some(id) = current_text_id.take() {
            yield v2t::StreamPart::TextEnd {
                id,
                provider_metadata: None,
            };
        }
        if let Some(id) = current_reasoning_id.take() {
            yield v2t::StreamPart::ReasoningEnd {
                id,
                provider_metadata: None,
            };
        }

        yield v2t::StreamPart::Finish {
            usage,
            finish_reason,
            provider_metadata,
        };
    })
}
