use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::ai_sdk_core::error::{SdkError, TransportError};

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
            // Try to parse body to extract message
            if let Ok(v) = serde_json::from_str::<JsonValue>(&body) {
                if let Ok(err) = serde_json::from_value::<AnthropicErrorData>(v.clone()) {
                    return SdkError::Upstream {
                        status,
                        message: err.error.message,
                        source: Some(Box::new(TransportError::HttpStatus {
                            status,
                            body,
                            retry_after_ms,
                            sanitized: format!("http status {}", status),
                            headers,
                        })),
                    };
                }
            }
            // Fallback to generic upstream with sanitized message
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
