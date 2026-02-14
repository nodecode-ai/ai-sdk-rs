use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::ai_sdk_core::error::{
    map_http_status_to_rate_limited_error, map_http_status_to_upstream_error, SdkError,
    TransportError,
};

// Mirrors packages/openai-compatible/src/openai-compatible-error.ts

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatibleInnerError {
    pub message: String,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub param: Option<JsonValue>,
    #[serde(default)]
    pub code: Option<JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatibleErrorData {
    pub error: OpenAICompatibleInnerError,
}

/// Attempt to parse an OpenAI-compatible JSON error body and map to an SdkError.
pub fn map_transport_error_to_sdk_error(te: TransportError) -> SdkError {
    match te {
        TransportError::HttpStatus {
            status,
            body,
            retry_after_ms,
            headers,
            ..
        } => {
            // Unauthorized and rate limit have special mapping
            if status == 401 {
                return map_http_status_to_upstream_error(
                    status,
                    body,
                    retry_after_ms,
                    headers,
                    None,
                );
            }
            if status == 429 {
                return map_http_status_to_rate_limited_error(
                    status,
                    body,
                    retry_after_ms,
                    headers,
                );
            }

            let parsed_message = serde_json::from_str::<OpenAICompatibleErrorData>(&body)
                .ok()
                .map(|err| err.error.message);
            map_http_status_to_upstream_error(status, body, retry_after_ms, headers, parsed_message)
        }
        other => SdkError::Transport(other),
    }
}

#[cfg(test)]
mod tests {
    use super::map_transport_error_to_sdk_error;
    use crate::ai_sdk_core::error::{SdkError, TransportError};

    fn http_status(status: u16, body: &str, retry_after_ms: Option<u64>) -> TransportError {
        TransportError::HttpStatus {
            status,
            body: body.to_string(),
            retry_after_ms,
            sanitized: "redacted".to_string(),
            headers: Vec::new(),
        }
    }

    #[test]
    fn openai_compatible_parser_message_is_preserved() {
        let mapped = map_transport_error_to_sdk_error(http_status(
            400,
            r#"{"error":{"message":"bad request"}}"#,
            None,
        ));
        match mapped {
            SdkError::Upstream { message, .. } => assert_eq!(message, "bad request"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn openai_compatible_rate_limit_mapping_keeps_retry_after() {
        let mapped = map_transport_error_to_sdk_error(http_status(429, "{}", Some(1234)));
        match mapped {
            SdkError::RateLimited { retry_after_ms, .. } => assert_eq!(retry_after_ms, Some(1234)),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn openai_compatible_fallback_message_uses_http_status() {
        let mapped = map_transport_error_to_sdk_error(http_status(500, "not-json", None));
        match mapped {
            SdkError::Upstream {
                message, source, ..
            } => {
                assert_eq!(message, "http status 500");
                match source {
                    Some(source) => match source.as_ref() {
                        TransportError::HttpStatus { sanitized, .. } => {
                            assert_eq!(sanitized, "http status 500")
                        }
                        other => panic!("unexpected source: {other:?}"),
                    },
                    None => panic!("expected source"),
                }
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
