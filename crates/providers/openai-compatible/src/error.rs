use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::ai_sdk_core::error::{SdkError, TransportError};

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
                return SdkError::Upstream {
                    status,
                    message: format!("http status {}", status),
                    source: Some(Box::new(TransportError::HttpStatus {
                        status,
                        body,
                        retry_after_ms,
                        sanitized: format!("http status {}", status),
                        headers,
                    })),
                };
            }
            if status == 429 {
                return SdkError::RateLimited {
                    retry_after_ms,
                    source: Some(Box::new(TransportError::HttpStatus {
                        status,
                        body,
                        retry_after_ms,
                        sanitized: format!("http status {}", status),
                        headers,
                    })),
                };
            }

            // Try to parse body to extract message
            if let Ok(v) = serde_json::from_str::<JsonValue>(&body) {
                if let Ok(err) = serde_json::from_value::<OpenAICompatibleErrorData>(v.clone()) {
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
