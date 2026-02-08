use crate::ai_sdk_core::error::{SdkError, TransportError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct OpenAIErrorEnvelope {
    pub error: OpenAIError,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    pub message: String,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub param: Option<serde_json::Value>,
    #[serde(default)]
    pub code: Option<serde_json::Value>,
}

/// Parse a typical OpenAI-style error JSON body and return the message if present.
pub fn parse_openai_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<OpenAIErrorEnvelope>(body)
        .ok()
        .map(|e| e.error.message)
}

/// Map TransportError to SdkError using OpenAI error conventions.
pub fn map_transport_error(te: TransportError) -> SdkError {
    match &te {
        TransportError::HttpStatus {
            status,
            retry_after_ms,
            body,
            ..
        } => match status {
            401 => {
                let message = parse_openai_error_message(body)
                    .unwrap_or_else(|| crate::ai_sdk_core::error::display_body_for_error(body));
                SdkError::Upstream {
                    status: 401,
                    message,
                    source: Some(Box::new(te)),
                }
            }
            429 => SdkError::RateLimited {
                retry_after_ms: *retry_after_ms,
                source: Some(Box::new(te)),
            },
            s => {
                let message = parse_openai_error_message(body)
                    .unwrap_or_else(|| crate::ai_sdk_core::error::display_body_for_error(body));
                SdkError::Upstream {
                    status: *s,
                    message,
                    source: Some(Box::new(te)),
                }
            }
        },
        TransportError::IdleReadTimeout(_) | TransportError::ConnectTimeout(_) => SdkError::Timeout,
        _ => SdkError::Transport(te),
    }
}
