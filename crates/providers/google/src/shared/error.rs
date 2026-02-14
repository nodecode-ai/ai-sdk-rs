use crate::ai_sdk_core::error::{map_http_status_to_upstream_error, SdkError, TransportError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleErrorInner {
    pub code: Option<i64>,
    pub message: String,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleErrorData {
    pub error: GoogleErrorInner,
}

/// Attempt to parse a Google-family JSON error body and map to SdkError.
pub fn map_transport_error_to_sdk_error(te: TransportError) -> SdkError {
    match te {
        TransportError::HttpStatus {
            status,
            body,
            retry_after_ms,
            headers,
            ..
        } => {
            let parsed_message = serde_json::from_str::<GoogleErrorData>(&body)
                .ok()
                .map(|parsed| parsed.error.message);
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
            status: 403,
            body: body.to_string(),
            retry_after_ms: None,
            sanitized: "redacted".to_string(),
            headers: Vec::new(),
        }
    }

    #[test]
    fn google_parser_message_is_preserved() {
        let mapped = map_transport_error_to_sdk_error(http_status(
            r#"{"error":{"code":403,"message":"permission denied","status":"PERMISSION_DENIED"}}"#,
        ));
        match mapped {
            SdkError::Upstream { message, .. } => assert_eq!(message, "permission denied"),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn google_fallback_message_uses_http_status() {
        let mapped = map_transport_error_to_sdk_error(http_status("not-json"));
        match mapped {
            SdkError::Upstream {
                message, source, ..
            } => {
                assert_eq!(message, "http status 403");
                match source {
                    Some(source) => match source.as_ref() {
                        TransportError::HttpStatus { sanitized, .. } => {
                            assert_eq!(sanitized, "http status 403")
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
