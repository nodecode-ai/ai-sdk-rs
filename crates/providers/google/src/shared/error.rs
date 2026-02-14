use crate::ai_sdk_core::error::{SdkError, TransportError};
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
            if let Ok(parsed) = serde_json::from_str::<GoogleErrorData>(&body) {
                return SdkError::Upstream {
                    status,
                    message: parsed.error.message,
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
