use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;

use crate::types::v2 as v2t;

pub(crate) struct ParsedGoogleGenerateResponse {
    pub content: Vec<v2t::Content>,
    pub finish_reason: v2t::FinishReason,
    pub usage: v2t::Usage,
    pub provider_metadata: Option<v2t::ProviderMetadata>,
}

#[derive(Default)]
struct ContentAccumulator {
    content: Vec<v2t::Content>,
    pending_code_execution_id: Option<String>,
    pending_function_response_id: Option<String>,
}

impl ContentAccumulator {
    fn push_google_gen_ai_part(&mut self, part: &JsonValue) {
        if self.push_executable_code(part)
            || self.push_code_execution_result(part)
            || self.push_text_or_reasoning(part, "google")
            || self.push_function_call(part, "google", false, false)
            || self.push_inline_data(part)
        {}
    }

    fn push_google_vertex_part(&mut self, part: &JsonValue) {
        if self.push_text_or_reasoning(part, "google-vertex")
            || self.push_function_call(part, "google-vertex", true, true)
            || self.push_function_response(part)
            || self.push_inline_data(part)
        {}
    }

    fn push_executable_code(&mut self, part: &JsonValue) -> bool {
        let Some(executable_code) = part
            .get("executableCode")
            .and_then(|value| value.as_object())
        else {
            return false;
        };
        if executable_code
            .get("code")
            .and_then(|value| value.as_str())
            .is_none()
        {
            return false;
        }

        let tool_call_id = uuid::Uuid::new_v4().to_string();
        self.pending_code_execution_id = Some(tool_call_id.clone());
        self.content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
            tool_call_id,
            tool_name: "code_execution".into(),
            input: serde_json::to_string(executable_code).unwrap_or_else(|_| "{}".into()),
            provider_executed: true,
            provider_metadata: None,
            dynamic: false,
            provider_options: None,
        }));
        true
    }

    fn push_code_execution_result(&mut self, part: &JsonValue) -> bool {
        let Some(result) = part
            .get("codeExecutionResult")
            .and_then(|value| value.as_object())
        else {
            return false;
        };
        let Some(tool_call_id) = self.pending_code_execution_id.take() else {
            return false;
        };

        self.content.push(v2t::Content::ToolResult {
            tool_call_id,
            tool_name: "code_execution".into(),
            result: json!({
                "outcome": result.get("outcome"),
                "output": result.get("output"),
            }),
            is_error: false,
            provider_metadata: None,
        });
        true
    }

    fn push_text_or_reasoning(&mut self, part: &JsonValue, provider_scope: &'static str) -> bool {
        let Some(text) = part.get("text").and_then(|value| value.as_str()) else {
            return false;
        };
        if text.is_empty() {
            return true;
        }

        let provider_metadata = thought_signature_metadata(part, provider_scope);
        if part
            .get("thought")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            self.content.push(v2t::Content::Reasoning {
                text: text.to_string(),
                provider_metadata,
            });
        } else {
            self.content.push(v2t::Content::Text {
                text: text.to_string(),
                provider_metadata,
            });
        }
        true
    }

    fn push_function_call(
        &mut self,
        part: &JsonValue,
        provider_scope: &'static str,
        remember_for_response: bool,
        null_args_as_empty_object: bool,
    ) -> bool {
        let Some(function_call) = part.get("functionCall").and_then(|value| value.as_object())
        else {
            return false;
        };

        let tool_name = function_call
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let args = function_call
            .get("args")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let input = if null_args_as_empty_object && args.is_null() {
            "{}".to_string()
        } else {
            args.to_string()
        };
        let tool_call_id = uuid::Uuid::new_v4().to_string();
        if remember_for_response {
            self.pending_function_response_id = Some(tool_call_id.clone());
        }

        self.content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
            tool_call_id,
            tool_name,
            input,
            provider_executed: false,
            provider_metadata: None,
            dynamic: false,
            provider_options: thought_signature_metadata(part, provider_scope),
        }));
        true
    }

    fn push_function_response(&mut self, part: &JsonValue) -> bool {
        let Some(function_response) = part
            .get("functionResponse")
            .and_then(|value| value.as_object())
        else {
            return false;
        };

        let tool_name = function_response
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let response_value = function_response
            .get("response")
            .cloned()
            .unwrap_or(JsonValue::Null);
        let result = response_value
            .get("content")
            .cloned()
            .unwrap_or(response_value);

        self.content.push(v2t::Content::ToolResult {
            tool_call_id: self
                .pending_function_response_id
                .clone()
                .unwrap_or_else(|| tool_name.clone()),
            tool_name,
            result,
            is_error: false,
            provider_metadata: None,
        });
        true
    }

    fn push_inline_data(&mut self, part: &JsonValue) -> bool {
        let Some(inline_data) = part.get("inlineData").and_then(|value| value.as_object()) else {
            return false;
        };

        let media_type = inline_data
            .get("mimeType")
            .and_then(|value| value.as_str())
            .unwrap_or("application/octet-stream")
            .to_string();
        let data = inline_data
            .get("data")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        self.content.push(v2t::Content::File { media_type, data });
        true
    }

    fn push_grounding_sources(&mut self, candidate: &JsonValue) {
        let Some(chunks) = candidate
            .get("groundingMetadata")
            .and_then(|value| value.get("groundingChunks"))
            .and_then(|value| value.as_array())
        else {
            return;
        };

        for chunk in chunks {
            let Some(web) = chunk.get("web").and_then(|value| value.as_object()) else {
                continue;
            };
            let Some(url) = web.get("uri").and_then(|value| value.as_str()) else {
                continue;
            };

            self.content.push(v2t::Content::SourceUrl {
                id: uuid::Uuid::new_v4().to_string(),
                url: url.to_string(),
                title: web
                    .get("title")
                    .and_then(|value| value.as_str())
                    .map(|value| value.to_string()),
                provider_metadata: None,
            });
        }
    }
}

pub(crate) fn parse_google_gen_ai_generate_response(
    resp_json: &JsonValue,
) -> ParsedGoogleGenerateResponse {
    let candidate = first_candidate(resp_json);
    let mut content = ContentAccumulator::default();
    if let Some(parts) = candidate_parts(candidate) {
        for part in parts {
            content.push_google_gen_ai_part(part);
        }
    }
    if let Some(candidate) = candidate {
        content.push_grounding_sources(candidate);
    }
    let content = content.content;
    let usage_metadata = resp_json.get("usageMetadata");

    ParsedGoogleGenerateResponse {
        finish_reason: google_finish_reason(candidate, &content),
        provider_metadata: google_provider_metadata(candidate, usage_metadata, "google"),
        usage: google_usage(usage_metadata),
        content,
    }
}

pub(crate) fn parse_google_vertex_generate_response(
    resp_json: &JsonValue,
) -> ParsedGoogleGenerateResponse {
    let candidate = first_candidate(resp_json);
    let mut content = ContentAccumulator::default();
    if let Some(parts) = candidate_parts(candidate) {
        for part in parts {
            content.push_google_vertex_part(part);
        }
    }

    ParsedGoogleGenerateResponse {
        content: content.content,
        finish_reason: v2t::FinishReason::Stop,
        provider_metadata: None,
        usage: google_vertex_usage(resp_json.get("usageMetadata")),
    }
}

fn first_candidate(resp_json: &JsonValue) -> Option<&JsonValue> {
    resp_json
        .get("candidates")
        .and_then(|value| value.as_array())
        .and_then(|candidates| candidates.first())
}

fn candidate_parts(candidate: Option<&JsonValue>) -> Option<&[JsonValue]> {
    candidate
        .and_then(|value| value.get("content"))
        .and_then(|value| value.get("parts"))
        .and_then(|value| value.as_array())
        .map(Vec::as_slice)
}

fn thought_signature_metadata(
    part: &JsonValue,
    provider_scope: &'static str,
) -> Option<v2t::ProviderMetadata> {
    let signature = part
        .get("thoughtSignature")
        .and_then(|value| value.as_str())?;
    let mut outer = HashMap::new();
    let mut inner = HashMap::new();
    inner.insert(
        "thoughtSignature".into(),
        JsonValue::String(signature.to_string()),
    );
    outer.insert(provider_scope.into(), inner);
    Some(outer)
}

fn google_finish_reason(
    candidate: Option<&JsonValue>,
    content: &[v2t::Content],
) -> v2t::FinishReason {
    let has_tool_calls = content
        .iter()
        .any(|part| matches!(part, v2t::Content::ToolCall(_)));
    match candidate
        .and_then(|value| value.get("finishReason"))
        .and_then(|value| value.as_str())
    {
        Some("STOP") => {
            if has_tool_calls {
                v2t::FinishReason::ToolCalls
            } else {
                v2t::FinishReason::Stop
            }
        }
        Some("MAX_TOKENS") => v2t::FinishReason::Length,
        Some(
            "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII",
        ) => v2t::FinishReason::ContentFilter,
        Some("FINISH_REASON_UNSPECIFIED" | "OTHER") => v2t::FinishReason::Other,
        Some("MALFORMED_FUNCTION_CALL") => v2t::FinishReason::Error,
        Some(_) => v2t::FinishReason::Unknown,
        None => v2t::FinishReason::Unknown,
    }
}

fn google_usage(usage_metadata: Option<&JsonValue>) -> v2t::Usage {
    let Some(usage_metadata) = usage_metadata else {
        return v2t::Usage::default();
    };

    v2t::Usage {
        input_tokens: usage_metadata
            .get("promptTokenCount")
            .and_then(|value| value.as_u64()),
        output_tokens: usage_metadata
            .get("candidatesTokenCount")
            .and_then(|value| value.as_u64()),
        total_tokens: usage_metadata
            .get("totalTokenCount")
            .and_then(|value| value.as_u64()),
        reasoning_tokens: usage_metadata
            .get("thoughtsTokenCount")
            .and_then(|value| value.as_u64()),
        cached_input_tokens: usage_metadata
            .get("cachedContentTokenCount")
            .and_then(|value| value.as_u64()),
    }
}

fn google_vertex_usage(usage_metadata: Option<&JsonValue>) -> v2t::Usage {
    let Some(usage_metadata) = usage_metadata else {
        return v2t::Usage::default();
    };

    let mut usage = v2t::Usage::default();
    usage.input_tokens = usage_metadata
        .get("promptTokenCount")
        .and_then(|value| value.as_u64());
    usage.output_tokens = usage_metadata
        .get("candidatesTokenCount")
        .and_then(|value| value.as_u64());
    usage.total_tokens = usage_metadata
        .get("totalTokenCount")
        .and_then(|value| value.as_u64());
    usage
}

fn google_provider_metadata(
    candidate: Option<&JsonValue>,
    usage_metadata: Option<&JsonValue>,
    provider_scope: &'static str,
) -> Option<v2t::ProviderMetadata> {
    let Some(candidate) = candidate.and_then(JsonValue::as_object) else {
        return None;
    };

    let mut inner = HashMap::new();
    inner.insert(
        "groundingMetadata".into(),
        candidate
            .get("groundingMetadata")
            .cloned()
            .unwrap_or(JsonValue::Null),
    );
    inner.insert(
        "urlContextMetadata".into(),
        candidate
            .get("urlContextMetadata")
            .cloned()
            .unwrap_or(JsonValue::Null),
    );
    inner.insert(
        "safetyRatings".into(),
        candidate
            .get("safetyRatings")
            .cloned()
            .unwrap_or(JsonValue::Null),
    );
    if let Some(usage_metadata) = usage_metadata.cloned() {
        inner.insert("usageMetadata".into(), usage_metadata);
    }

    let mut outer = HashMap::new();
    outer.insert(provider_scope.into(), inner);
    Some(outer)
}
