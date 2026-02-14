use serde::Deserialize;

use crate::ai_sdk_core::error::{map_http_status_to_upstream_error, SdkError, TransportError};

#[derive(Debug, Deserialize)]
pub struct BedrockErrorPayload {
    pub message: Option<String>,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
}

/// Map a transport error into an `SdkError`, attempting to surface the
/// Bedrock error payload if available.
pub fn map_transport_error(te: TransportError) -> SdkError {
    match te {
        TransportError::HttpStatus {
            status,
            body,
            retry_after_ms,
            headers,
            ..
        } => {
            let parsed_message = serde_json::from_str::<BedrockErrorPayload>(&body)
                .ok()
                .and_then(|err| err.message.or(err.error_type));
            map_http_status_to_upstream_error(status, body, retry_after_ms, headers, parsed_message)
        }
        other => SdkError::Transport(other),
    }
}

#[cfg(test)]
mod tests {
    use super::map_transport_error;
    use crate::ai_sdk_core::error::{SdkError, TransportError};

    fn http_status(body: &str) -> TransportError {
        TransportError::HttpStatus {
            status: 502,
            body: body.to_string(),
            retry_after_ms: None,
            sanitized: "redacted".to_string(),
            headers: Vec::new(),
        }
    }

    #[test]
    fn bedrock_parser_message_is_preserved() {
        let mapped = map_transport_error(http_status(r#"{"message":"model unavailable"}"#));
        match mapped {
            SdkError::Upstream { message, .. } => assert_eq!(message, "model unavailable"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn bedrock_fallback_message_uses_http_status() {
        let mapped = map_transport_error(http_status("not-json"));
        match mapped {
            SdkError::Upstream {
                message, source, ..
            } => {
                assert_eq!(message, "http status 502");
                match source {
                    Some(source) => match source.as_ref() {
                        TransportError::HttpStatus { sanitized, .. } => {
                            assert_eq!(sanitized, "http status 502")
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
