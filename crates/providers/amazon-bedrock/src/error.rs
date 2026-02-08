use serde::Deserialize;

use crate::ai_sdk_core::error::{SdkError, TransportError};

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
            if let Ok(err) = serde_json::from_str::<BedrockErrorPayload>(&body) {
                let message = err
                    .message
                    .or(err.error_type)
                    .unwrap_or_else(|| format!("http status {}", status));
                return SdkError::Upstream {
                    status,
                    message,
                    source: Some(Box::new(TransportError::HttpStatus {
                        status,
                        body,
                        retry_after_ms,
                        sanitized: format!("http status {}", status),
                        headers,
                    })),
                };
            }
            SdkError::Upstream {
                status,
                message: format!("http status {}", status),
                source: Some(Box::new(TransportError::HttpStatus {
                    status,
                    body,
                    retry_after_ms,
                    sanitized: format!("http status {}", status),
                    headers,
                })),
            }
        }
        other => SdkError::Transport(other),
    }
}
