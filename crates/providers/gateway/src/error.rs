use crate::ai_sdk_core::error::{display_body_for_error, SdkError, TransportError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GatewayErrorEnvelope {
    error: GatewayErrorBody,
}

#[derive(Debug, Deserialize)]
struct GatewayErrorBody {
    #[serde(default)]
    message: Option<String>,
    #[serde(default, rename = "type")]
    error_type: Option<String>,
    #[serde(default)]
    _code: Option<serde_json::Value>,
    #[serde(default)]
    _param: Option<serde_json::Value>,
}

fn parse_gateway_error(body: &str) -> Option<GatewayErrorBody> {
    serde_json::from_str::<GatewayErrorEnvelope>(body)
        .ok()
        .map(|envelope| envelope.error)
}

pub fn map_transport_error(te: TransportError) -> SdkError {
    match &te {
        TransportError::HttpStatus {
            status,
            retry_after_ms,
            body,
            ..
        } => {
            let parsed = parse_gateway_error(body);
            let message = parsed
                .as_ref()
                .and_then(|err| err.message.as_ref().map(ToString::to_string))
                .or_else(|| {
                    parsed
                        .as_ref()
                        .and_then(|err| err.error_type.as_ref())
                        .map(|t| format!("{} error", t))
                })
                .unwrap_or_else(|| display_body_for_error(body));
            match *status {
                401 | 403 => SdkError::Unauthorized,
                408 => SdkError::Timeout,
                425 | 429 => SdkError::RateLimited {
                    retry_after_ms: *retry_after_ms,
                    source: Some(Box::new(te)),
                },
                _ => SdkError::Upstream {
                    status: *status,
                    message,
                    source: Some(Box::new(te)),
                },
            }
        }
        TransportError::IdleReadTimeout(_) | TransportError::ConnectTimeout(_) => SdkError::Timeout,
        _ => SdkError::Transport(te),
    }
}
