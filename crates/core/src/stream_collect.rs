use std::collections::HashMap;

use futures_util::StreamExt;

use crate::ai_sdk_types::v2 as v2t;
use crate::core::SdkError;
use crate::core::{GenerateResponse, StreamResponse};

/// Controls how stream parts are collapsed into a `GenerateResponse`.
#[derive(Debug, Clone, Copy)]
pub struct StreamCollectorConfig {
    /// Collect reasoning parts into `Content::Reasoning`.
    pub allow_reasoning: bool,
    /// If set, attach the reasoning signature under this provider scope.
    pub reasoning_metadata_scope: Option<&'static str>,
    /// Collect `Content::ToolCall` and `Content::ToolApprovalRequest` parts.
    pub allow_tool_calls: bool,
    /// Collect `Content::ToolResult` parts.
    pub allow_tool_results: bool,
    /// Collect `Content::File` parts.
    pub allow_files: bool,
    /// Collect `Content::SourceUrl` parts.
    pub allow_source_urls: bool,
    /// When true, propagate `StreamPart::Error` as an upstream failure.
    pub fail_on_error: bool,
}

impl Default for StreamCollectorConfig {
    fn default() -> Self {
        Self {
            allow_reasoning: false,
            reasoning_metadata_scope: None,
            allow_tool_calls: false,
            allow_tool_results: false,
            allow_files: false,
            allow_source_urls: false,
            fail_on_error: false,
        }
    }
}

/// Collapse a `StreamResponse` into a `GenerateResponse`, honoring the provided config.
pub async fn collect_stream_to_response(
    stream_resp: StreamResponse,
    cfg: StreamCollectorConfig,
) -> Result<GenerateResponse, SdkError> {
    let mut content: Vec<v2t::Content> = Vec::new();
    let mut text_buf: HashMap<String, String> = HashMap::new();
    let mut reasoning_buf: HashMap<String, String> = HashMap::new();
    let mut reasoning_signature: Option<String> = None;
    let mut usage = v2t::Usage::default();
    let mut finish_reason = v2t::FinishReason::Unknown;
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();

    let stream = stream_resp.stream;
    futures_util::pin_mut!(stream);
    while let Some(item) = stream.next().await {
        let part = item?;
        match part {
            v2t::StreamPart::StreamStart { warnings: w } => {
                warnings = w;
            }
            v2t::StreamPart::TextStart { id, .. } => {
                text_buf.entry(id).or_default();
            }
            v2t::StreamPart::TextDelta { id, delta, .. } => {
                text_buf
                    .entry(id)
                    .and_modify(|s| s.push_str(&delta))
                    .or_insert(delta);
            }
            v2t::StreamPart::TextEnd {
                id,
                provider_metadata,
            } => {
                if let Some(text) = text_buf.remove(&id) {
                    if !text.is_empty() {
                        content.push(v2t::Content::Text {
                            text,
                            provider_metadata,
                        });
                    }
                }
            }
            v2t::StreamPart::ReasoningStart { id, .. } if cfg.allow_reasoning => {
                reasoning_buf.entry(id).or_default();
            }
            v2t::StreamPart::ReasoningDelta { id, delta, .. } if cfg.allow_reasoning => {
                reasoning_buf
                    .entry(id)
                    .and_modify(|s| s.push_str(&delta))
                    .or_insert(delta);
            }
            v2t::StreamPart::ReasoningEnd { id, .. } if cfg.allow_reasoning => {
                if let Some(text) = reasoning_buf.remove(&id) {
                    if !text.is_empty() {
                        let provider_metadata = reasoning_signature.as_ref().and_then(|sig| {
                            cfg.reasoning_metadata_scope.map(|scope| {
                                let mut inner = std::collections::HashMap::new();
                                inner.insert(
                                    "signature".into(),
                                    serde_json::Value::String(sig.clone()),
                                );
                                let mut outer = std::collections::HashMap::new();
                                outer.insert(scope.to_string(), inner);
                                outer
                            })
                        });
                        content.push(v2t::Content::Reasoning {
                            text,
                            provider_metadata,
                        });
                    }
                }
            }
            v2t::StreamPart::ReasoningSignature { signature, .. } if cfg.allow_reasoning => {
                reasoning_signature = Some(signature);
            }
            v2t::StreamPart::ToolCall(tc) if cfg.allow_tool_calls => {
                content.push(v2t::Content::ToolCall(tc));
            }
            v2t::StreamPart::ToolApprovalRequest {
                approval_id,
                tool_call_id,
                provider_metadata,
            } if cfg.allow_tool_calls => {
                content.push(v2t::Content::ToolApprovalRequest {
                    approval_id,
                    tool_call_id,
                    provider_metadata,
                });
            }
            v2t::StreamPart::ToolResult {
                tool_call_id,
                tool_name,
                result,
                is_error,
                preliminary: _,
                provider_metadata,
            } if cfg.allow_tool_results => {
                content.push(v2t::Content::ToolResult {
                    tool_call_id,
                    tool_name,
                    result,
                    is_error,
                    provider_metadata,
                });
            }
            v2t::StreamPart::File { media_type, data } if cfg.allow_files => {
                content.push(v2t::Content::File { media_type, data });
            }
            v2t::StreamPart::SourceUrl {
                id,
                url,
                title,
                provider_metadata,
            } if cfg.allow_source_urls => {
                content.push(v2t::Content::SourceUrl {
                    id,
                    url,
                    title,
                    provider_metadata,
                });
            }
            v2t::StreamPart::Finish {
                usage: u,
                finish_reason: fr,
                ..
            } => {
                usage = u;
                finish_reason = fr;
                break;
            }
            v2t::StreamPart::Error { error } if cfg.fail_on_error => {
                return Err(SdkError::Upstream {
                    status: 500,
                    message: error.to_string(),
                    source: None,
                });
            }
            _ => {}
        }
    }

    Ok(GenerateResponse {
        content,
        finish_reason,
        usage,
        provider_metadata: None,
        request_body: None,
        response_headers: stream_resp.response_headers,
        response_body: None,
        warnings,
    })
}
