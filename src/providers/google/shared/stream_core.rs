use std::collections::{HashMap, HashSet};
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::{json, Value as JsonValue};

use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::{PartStream, StreamNormalizationState};
use crate::ai_sdk_streaming_sse::SseDecoder;
use crate::ai_sdk_types::v2 as v2t;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>;

struct GoogleStreamState {
    normalizer: StreamNormalizationState<()>,
    last_code_tool_id: Option<String>,
    emitted_source_urls: HashSet<String>,
    finish_reason: v2t::FinishReason,
    provider_metadata: Option<v2t::ProviderMetadata>,
    block_counter: u64,
}

impl Default for GoogleStreamState {
    fn default() -> Self {
        Self {
            normalizer: StreamNormalizationState::new(()),
            last_code_tool_id: None,
            emitted_source_urls: HashSet::new(),
            finish_reason: v2t::FinishReason::Unknown,
            provider_metadata: None,
            block_counter: 0,
        }
    }
}

impl GoogleStreamState {
    fn decode_event(
        &mut self,
        data: &[u8],
        include_raw: bool,
        provider_scope: &'static str,
    ) -> Vec<v2t::StreamPart> {
        let data = String::from_utf8_lossy(data).to_string();
        if data.trim().is_empty() {
            return Vec::new();
        }
        let parsed = match serde_json::from_str::<JsonValue>(&data) {
            Ok(value) => value,
            Err(_) => return Vec::new(),
        };

        let mut parts = Vec::new();
        if include_raw {
            parts.push(v2t::StreamPart::Raw {
                raw_value: parsed.clone(),
            });
        }

        self.apply_usage(&parsed);
        self.push_candidate_parts(&parsed, provider_scope, &mut parts);
        parts
    }

    fn apply_usage(&mut self, parsed: &JsonValue) {
        if let Some(usage) = parsed.get("usageMetadata") {
            self.normalizer.usage = v2t::Usage {
                input_tokens: usage.get("promptTokenCount").and_then(|v| v.as_u64()),
                output_tokens: usage.get("candidatesTokenCount").and_then(|v| v.as_u64()),
                total_tokens: usage.get("totalTokenCount").and_then(|v| v.as_u64()),
                reasoning_tokens: usage.get("thoughtsTokenCount").and_then(|v| v.as_u64()),
                cached_input_tokens: usage
                    .get("cachedContentTokenCount")
                    .and_then(|v| v.as_u64()),
            };
        }
    }

    fn push_candidate_parts(
        &mut self,
        parsed: &JsonValue,
        provider_scope: &'static str,
        parts: &mut Vec<v2t::StreamPart>,
    ) {
        let Some(candidates) = parsed.get("candidates").and_then(|v| v.as_array()) else {
            return;
        };

        for candidate in candidates {
            self.push_candidate_content(candidate, provider_scope, parts);
            self.push_grounding_chunks(candidate, parts);
            self.capture_candidate_finish(candidate, provider_scope);
        }
    }

    fn push_candidate_content(
        &mut self,
        candidate: &JsonValue,
        provider_scope: &'static str,
        parts: &mut Vec<v2t::StreamPart>,
    ) {
        let Some(content) = candidate.get("content").and_then(|v| v.as_object()) else {
            return;
        };
        let candidate_parts = content
            .get("parts")
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();

        for part in candidate_parts {
            if self.push_executable_code(&part, parts) {
                continue;
            }
            if self.push_code_execution_result(&part, parts) {
                continue;
            }
            if self.push_text_part(&part, provider_scope, parts) {
                continue;
            }
            if self.push_inline_data(&part, parts) {
                continue;
            }
            if self.push_function_call(&part, provider_scope, parts) {
                continue;
            }
        }
    }

    fn push_executable_code(&mut self, part: &JsonValue, parts: &mut Vec<v2t::StreamPart>) -> bool {
        let Some(executable_code) = part.get("executableCode").and_then(|v| v.as_object()) else {
            return false;
        };
        if executable_code
            .get("code")
            .and_then(|v| v.as_str())
            .is_none()
        {
            return false;
        }

        let id = uuid::Uuid::new_v4().to_string();
        self.last_code_tool_id = Some(id.clone());
        parts.push(self.normalizer.start_tool_call(
            id.clone(),
            "code_execution".into(),
            true,
            None,
        ));
        let delta = serde_json::to_string(executable_code).unwrap_or_else(|_| "{}".into());
        parts.push(
            self.normalizer
                .push_tool_call_delta(id.clone(), delta, true, None),
        );
        parts.extend(
            self.normalizer
                .finish_tool_call(id, true, None, None, false, None),
        );
        true
    }

    fn push_code_execution_result(
        &mut self,
        part: &JsonValue,
        parts: &mut Vec<v2t::StreamPart>,
    ) -> bool {
        let Some(result) = part.get("codeExecutionResult").and_then(|v| v.as_object()) else {
            return false;
        };
        let Some(id) = self.last_code_tool_id.take() else {
            return false;
        };

        parts.push(v2t::StreamPart::ToolResult {
            tool_call_id: id,
            tool_name: "code_execution".into(),
            result: json!({
                "outcome": result.get("outcome"),
                "output": result.get("output")
            }),
            is_error: false,
            preliminary: false,
            provider_metadata: None,
        });
        true
    }

    fn push_text_part(
        &mut self,
        part: &JsonValue,
        provider_scope: &'static str,
        parts: &mut Vec<v2t::StreamPart>,
    ) -> bool {
        let Some(text) = part.get("text").and_then(|v| v.as_str()) else {
            return false;
        };
        if text.is_empty() {
            return true;
        }

        let is_thought = part
            .get("thought")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let metadata = thought_signature_metadata(part, provider_scope);
        if is_thought {
            if let Some(close_text) = self.normalizer.close_text(None) {
                parts.push(close_text);
            }
            if self.normalizer.reasoning_open.is_none() {
                let id = self.next_block_id("r");
                parts.extend(self.normalizer.open_reasoning(id, metadata.clone()));
            }
            parts.push(self.normalizer.push_reasoning_delta(
                "reasoning-1",
                text.to_string(),
                metadata,
            ));
        } else {
            if let Some(close_reasoning) = self.normalizer.close_reasoning(None) {
                parts.push(close_reasoning);
            }
            let start_metadata = if self.normalizer.text_open.is_none() {
                metadata.clone()
            } else {
                None
            };
            let id = self
                .normalizer
                .text_open
                .clone()
                .unwrap_or_else(|| self.next_block_id("t"));
            parts.extend(self.normalizer.push_text_delta(
                Some(id),
                "t-0",
                text.to_string(),
                start_metadata,
                metadata,
            ));
        }
        true
    }

    fn push_inline_data(&mut self, part: &JsonValue, parts: &mut Vec<v2t::StreamPart>) -> bool {
        let Some(inline_data) = part.get("inlineData").and_then(|v| v.as_object()) else {
            return false;
        };
        let media_type = inline_data
            .get("mimeType")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let data = inline_data
            .get("data")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        parts.push(v2t::StreamPart::File { media_type, data });
        true
    }

    fn push_function_call(
        &mut self,
        part: &JsonValue,
        provider_scope: &'static str,
        parts: &mut Vec<v2t::StreamPart>,
    ) -> bool {
        let Some(function_call) = part.get("functionCall").and_then(|v| v.as_object()) else {
            return false;
        };

        let name = function_call
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args = function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({}))
            .to_string();
        let provider_options = thought_signature_metadata(part, provider_scope);
        let id = uuid::Uuid::new_v4().to_string();
        parts.push(
            self.normalizer
                .start_tool_call(id.clone(), name, false, None),
        );
        parts.push(
            self.normalizer
                .push_tool_call_delta(id.clone(), args, false, None),
        );
        parts.extend(self.normalizer.finish_tool_call(
            id,
            false,
            None,
            None,
            false,
            provider_options,
        ));
        true
    }

    fn push_grounding_chunks(&mut self, candidate: &JsonValue, parts: &mut Vec<v2t::StreamPart>) {
        let Some(chunks) = candidate
            .get("groundingMetadata")
            .and_then(|v| v.get("groundingChunks"))
            .and_then(|v| v.as_array())
        else {
            return;
        };

        for chunk in chunks {
            let Some(web) = chunk.get("web").and_then(|v| v.as_object()) else {
                continue;
            };
            let Some(url) = web.get("uri").and_then(|v| v.as_str()) else {
                continue;
            };
            if self.emitted_source_urls.insert(url.to_string()) {
                let id = uuid::Uuid::new_v4().to_string();
                let title = web
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                parts.push(v2t::StreamPart::SourceUrl {
                    id,
                    url: url.to_string(),
                    title,
                    provider_metadata: None,
                });
            }
        }
    }

    fn capture_candidate_finish(&mut self, candidate: &JsonValue, provider_scope: &'static str) {
        let Some(finish_reason) = candidate.get("finishReason").and_then(|v| v.as_str()) else {
            return;
        };

        self.finish_reason = match finish_reason {
            "STOP" => {
                if self.normalizer.has_tool_calls {
                    v2t::FinishReason::ToolCalls
                } else {
                    v2t::FinishReason::Stop
                }
            }
            "MAX_TOKENS" => v2t::FinishReason::Length,
            "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT"
            | "SPII" => v2t::FinishReason::ContentFilter,
            "FINISH_REASON_UNSPECIFIED" | "OTHER" => v2t::FinishReason::Other,
            "MALFORMED_FUNCTION_CALL" => v2t::FinishReason::Error,
            _ => v2t::FinishReason::Unknown,
        };

        let mut inner_map = HashMap::new();
        inner_map.insert(
            "groundingMetadata".into(),
            candidate
                .get("groundingMetadata")
                .cloned()
                .unwrap_or(JsonValue::Null),
        );
        inner_map.insert(
            "urlContextMetadata".into(),
            candidate
                .get("urlContextMetadata")
                .cloned()
                .unwrap_or(JsonValue::Null),
        );
        inner_map.insert(
            "safetyRatings".into(),
            candidate
                .get("safetyRatings")
                .cloned()
                .unwrap_or(JsonValue::Null),
        );

        let usage_has = self.normalizer.usage.input_tokens.is_some()
            || self.normalizer.usage.output_tokens.is_some()
            || self.normalizer.usage.total_tokens.is_some()
            || self.normalizer.usage.reasoning_tokens.is_some()
            || self.normalizer.usage.cached_input_tokens.is_some();
        if usage_has {
            inner_map.insert(
                "usageMetadata".into(),
                json!({
                    "promptTokenCount": self.normalizer.usage.input_tokens,
                    "candidatesTokenCount": self.normalizer.usage.output_tokens,
                    "totalTokenCount": self.normalizer.usage.total_tokens,
                    "thoughtsTokenCount": self.normalizer.usage.reasoning_tokens,
                    "cachedContentTokenCount": self.normalizer.usage.cached_input_tokens,
                }),
            );
        }

        let mut outer = HashMap::new();
        outer.insert(provider_scope.into(), inner_map);
        self.provider_metadata = Some(outer);
    }

    fn next_block_id(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}-{}", self.block_counter);
        self.block_counter += 1;
        id
    }

    fn finish_parts(&mut self) -> Vec<v2t::StreamPart> {
        self.normalizer.finish_stream(
            Some((self.finish_reason.clone(), self.provider_metadata.clone())),
            v2t::FinishReason::Unknown,
        )
    }
}

fn thought_signature_metadata(
    part: &JsonValue,
    provider_scope: &'static str,
) -> Option<v2t::ProviderMetadata> {
    let signature = part.get("thoughtSignature").and_then(|v| v.as_str())?;
    let mut outer = HashMap::new();
    let mut inner = HashMap::new();
    inner.insert(
        "thoughtSignature".into(),
        JsonValue::String(signature.to_string()),
    );
    outer.insert(provider_scope.into(), inner);
    Some(outer)
}

pub fn build_google_stream_part_stream(
    mut inner: ByteStream,
    warnings: Vec<v2t::CallWarning>,
    include_raw: bool,
    provider_scope: &'static str,
) -> PartStream {
    Box::pin(async_stream::try_stream! {
        yield v2t::StreamPart::StreamStart { warnings };
        let mut decoder = SseDecoder::new();
        let mut state = GoogleStreamState::default();

        while let Some(chunk_res) = inner.next().await {
            match chunk_res {
                Ok(chunk) => {
                    for ev in decoder.push(&chunk) {
                        for part in state.decode_event(&ev.data, include_raw, provider_scope) {
                            yield part;
                        }
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
            for part in state.decode_event(&ev.data, include_raw, provider_scope) {
                yield part;
            }
        }

        for part in state.finish_parts() {
            yield part;
        }
    })
}
