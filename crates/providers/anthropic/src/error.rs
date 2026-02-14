use serde::{Deserialize, Serialize};

use crate::ai_sdk_core::error::{map_http_status_to_upstream_error, SdkError, TransportError};

// Mirrors packages/anthropic/src/anthropic-error.ts

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorItem {
    pub r#type: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorData {
    pub r#type: String, // 'error'
    pub error: AnthropicErrorItem,
}

/// Attempt to parse an Anthropic JSON error body and map to an SdkError::Upstream.
pub fn map_transport_error_to_sdk_error(te: TransportError) -> SdkError {
    match te {
        TransportError::HttpStatus {
            status,
            body,
            retry_after_ms,
            headers,
            ..
        } => {
            let parsed_message = serde_json::from_str::<AnthropicErrorData>(&body)
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

    fn http_status(body: &str) -> TransportError {
        TransportError::HttpStatus {
            status: 400,
            body: body.to_string(),
            retry_after_ms: None,
            sanitized: "redacted".to_string(),
            headers: Vec::new(),
        }
    }

    #[test]
    fn anthropic_parser_message_is_preserved() {
        let mapped = map_transport_error_to_sdk_error(http_status(
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}"#,
        ));

        match mapped {
            SdkError::Upstream { message, .. } => assert_eq!(message, "bad input"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn anthropic_fallback_message_uses_http_status() {
        let mapped = map_transport_error_to_sdk_error(http_status("not-json"));
        match mapped {
            SdkError::Upstream {
                message, source, ..
            } => {
                assert_eq!(message, "http status 400");
                match source {
                    Some(source) => match source.as_ref() {
                        TransportError::HttpStatus { sanitized, .. } => {
                            assert_eq!(sanitized, "http status 400")
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
