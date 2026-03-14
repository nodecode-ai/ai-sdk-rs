use serde_json::Value;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SdkError {
    #[error("unauthorized")]
    Unauthorized,
    #[error("rate limited")]
    RateLimited {
        /// Milliseconds suggested by Retry-After if present
        retry_after_ms: Option<u64>,
        #[source]
        source: Option<Box<TransportError>>,
    },
    #[error("timeout")]
    Timeout,
    #[error("cancelled")]
    Cancelled,
    #[error("upstream error (status {status}): {message}")]
    Upstream {
        status: u16,
        message: String,
        #[source]
        source: Option<Box<TransportError>>,
    },
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
}

// Retry behavior is handled by caller-specific logic; removed Retryable trait impl.

impl SdkError {
    /// Format error details for better debugging visibility
    pub fn format_details(&self) -> String {
        match self {
            SdkError::RateLimited {
                retry_after_ms,
                source,
                ..
            } => {
                let mut msg = String::from("rate limited");

                // Add retry-after if present
                if let Some(ms) = retry_after_ms {
                    msg.push_str(&format!(" (retry after {}ms)", ms));
                }

                // Add source error details if available
                if let Some(src) = source {
                    // Extract the body from HttpStatus error if available
                    if let TransportError::HttpStatus { body, status, .. } = src.as_ref() {
                        msg = format!("http status {}: {}", status, body);
                    }
                }

                msg
            }
            SdkError::Upstream {
                status,
                message,
                source,
            } => {
                let mut msg = format!("http status {}: {}", status, message);

                // Add source body if different from message
                if let Some(src) = source {
                    if let TransportError::HttpStatus { body, .. } = src.as_ref() {
                        // Only add body if it's not already in the message
                        if !message.contains(body) && !body.is_empty() {
                            msg.push_str(&format!(" [body: {}]", body));
                        }
                    }
                }

                msg
            }
            SdkError::Timeout => "timeout".to_string(),
            SdkError::Unauthorized => "unauthorized".to_string(),
            SdkError::Cancelled => "cancelled".to_string(),
            SdkError::Transport(te) => format!("transport error: {}", te),
            SdkError::Serde(se) => format!("serde error: {}", se),
            SdkError::InvalidArgument { message } => format!("invalid argument: {}", message),
        }
    }
}

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("http status {status}: {sanitized}")]
    HttpStatus {
        status: u16,
        /// upstream body (should be treated as sensitive; only log sanitized)
        body: String,
        /// Retry-After header (ms) if available
        retry_after_ms: Option<u64>,
        /// Sanitized message for display
        sanitized: String,
        /// Upstream response headers (lowercased keys where possible)
        headers: Vec<(String, String)>,
    },
    #[error("network: {0}")]
    Network(String),
    #[error("connect timeout after {0:?}")]
    ConnectTimeout(Duration),
    #[error("idle read timeout after {0:?}")]
    IdleReadTimeout(Duration),
    #[error("body read error: {0}")]
    BodyRead(String),
    #[error("stream closed")]
    StreamClosed,
    #[error("other: {0}")]
    Other(String),
}

impl TransportError {
    pub fn status(&self) -> Option<u16> {
        match self {
            TransportError::HttpStatus { status, .. } => Some(*status),
            _ => None,
        }
    }
    pub fn retry_after_ms(&self) -> Option<u64> {
        match self {
            TransportError::HttpStatus { retry_after_ms, .. } => *retry_after_ms,
            _ => None,
        }
    }
    pub fn sanitized_message(&self) -> String {
        match self {
            TransportError::HttpStatus { status, .. } => format!("http status {status}"),
            _ => self.to_string(),
        }
    }
}

pub fn http_status_fallback_message(status: u16) -> String {
    format!("http status {status}")
}

pub fn build_http_status_transport_error(
    status: u16,
    body: String,
    retry_after_ms: Option<u64>,
    headers: Vec<(String, String)>,
) -> TransportError {
    TransportError::HttpStatus {
        status,
        body,
        retry_after_ms,
        sanitized: http_status_fallback_message(status),
        headers,
    }
}

pub fn map_http_status_to_upstream_error(
    status: u16,
    body: String,
    retry_after_ms: Option<u64>,
    headers: Vec<(String, String)>,
    message: Option<String>,
) -> SdkError {
    let fallback = http_status_fallback_message(status);
    let source = build_http_status_transport_error(status, body, retry_after_ms, headers);
    SdkError::Upstream {
        status,
        message: message.unwrap_or(fallback),
        source: Some(Box::new(source)),
    }
}

pub fn map_http_status_to_rate_limited_error(
    status: u16,
    body: String,
    retry_after_ms: Option<u64>,
    headers: Vec<(String, String)>,
) -> SdkError {
    let source = build_http_status_transport_error(status, body, retry_after_ms, headers);
    SdkError::RateLimited {
        retry_after_ms,
        source: Some(Box::new(source)),
    }
}

pub fn display_body_for_error(body: &str) -> String {
    let trimmed = body.trim();
    let looks_like_json = trimmed.starts_with('{') || trimmed.starts_with('[');
    if looks_like_json {
        match serde_json::from_str::<Value>(trimmed) {
            Ok(v) => v.to_string(), // minified JSON
            Err(_) => format!("{} bytes", body.len()),
        }
    } else {
        format!("{} bytes", body.len())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_http_status_transport_error, http_status_fallback_message,
        map_http_status_to_rate_limited_error, map_http_status_to_upstream_error, SdkError,
        TransportError,
    };

    #[test]
    fn upstream_helper_uses_parsed_message_when_present() {
        let mapped = map_http_status_to_upstream_error(
            418,
            "{\"error\":\"teapot\"}".into(),
            Some(1500),
            vec![("x-test".into(), "1".into())],
            Some("custom message".into()),
        );

        match mapped {
            SdkError::Upstream {
                status,
                message,
                source,
            } => {
                assert_eq!(status, 418);
                assert_eq!(message, "custom message");
                match source {
                    Some(source) => match source.as_ref() {
                        TransportError::HttpStatus { sanitized, .. } => {
                            assert_eq!(sanitized, "http status 418")
                        }
                        other => panic!("unexpected source: {other:?}"),
                    },
                    None => panic!("expected source transport error"),
                }
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn upstream_helper_uses_http_status_fallback_when_message_missing() {
        let mapped = map_http_status_to_upstream_error(
            503,
            "upstream unavailable".into(),
            None,
            Vec::new(),
            None,
        );

        match mapped {
            SdkError::Upstream { message, .. } => {
                assert_eq!(message, "http status 503");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn rate_limited_helper_preserves_retry_after() {
        let mapped =
            map_http_status_to_rate_limited_error(429, "slow down".into(), Some(2500), Vec::new());

        match mapped {
            SdkError::RateLimited {
                retry_after_ms,
                source,
            } => {
                assert_eq!(retry_after_ms, Some(2500));
                match source {
                    Some(source) => match source.as_ref() {
                        TransportError::HttpStatus { sanitized, .. } => {
                            assert_eq!(sanitized, "http status 429")
                        }
                        other => panic!("unexpected source: {other:?}"),
                    },
                    None => panic!("expected source transport error"),
                }
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn fallback_message_and_builder_are_consistent() {
        assert_eq!(http_status_fallback_message(404), "http status 404");
        let built =
            build_http_status_transport_error(404, "not found".into(), Some(10), Vec::new());
        match built {
            TransportError::HttpStatus {
                status,
                retry_after_ms,
                sanitized,
                ..
            } => {
                assert_eq!(status, 404);
                assert_eq!(retry_after_ms, Some(10));
                assert_eq!(sanitized, "http status 404");
            }
            other => panic!("unexpected transport variant: {other:?}"),
        }
    }
}
