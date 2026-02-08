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
