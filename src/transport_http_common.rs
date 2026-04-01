use crate::core::error::{display_body_for_error, TransportError};
use crate::core::transport::{emit_transport_event, TransportBody, TransportEvent};
use std::time::{Instant, SystemTime};

pub(crate) struct RequestContext {
    started_at: SystemTime,
    start_instant: Instant,
    method: String,
    url: String,
    request_headers: Vec<(String, String)>,
    request_body: Option<TransportBody>,
    is_stream: bool,
}

impl RequestContext {
    pub(crate) fn new(
        method: String,
        url: String,
        request_headers: Vec<(String, String)>,
        request_body: Option<TransportBody>,
        is_stream: bool,
    ) -> Self {
        Self {
            started_at: SystemTime::now(),
            start_instant: Instant::now(),
            method,
            url,
            request_headers,
            request_body,
            is_stream,
        }
    }
}

pub(crate) fn emit_send_error_event(context: &RequestContext, detail: String) {
    emit_transport_event(TransportEvent {
        started_at: context.started_at,
        latency: Some(context.start_instant.elapsed()),
        method: context.method.clone(),
        url: context.url.clone(),
        status: None,
        request_headers: context.request_headers.clone(),
        response_headers: Vec::new(),
        request_body: context.request_body.clone(),
        response_body: None,
        response_size: None,
        error: Some(detail),
        is_stream: context.is_stream,
    });
}

pub(crate) fn emit_response_success_event(
    context: &RequestContext,
    status: u16,
    response_headers: Vec<(String, String)>,
    response_body: Option<TransportBody>,
    response_size: Option<usize>,
) {
    emit_transport_event(TransportEvent {
        started_at: context.started_at,
        latency: Some(context.start_instant.elapsed()),
        method: context.method.clone(),
        url: context.url.clone(),
        status: Some(status),
        request_headers: context.request_headers.clone(),
        response_headers,
        request_body: context.request_body.clone(),
        response_body,
        response_size,
        error: None,
        is_stream: context.is_stream,
    });
}

pub(crate) fn map_http_status_error(
    context: &RequestContext,
    status: u16,
    retry_after_ms: Option<u64>,
    response_headers: Vec<(String, String)>,
    body: String,
) -> TransportError {
    let sanitized = display_body_for_error(&body);
    emit_transport_event(TransportEvent {
        started_at: context.started_at,
        latency: Some(context.start_instant.elapsed()),
        method: context.method.clone(),
        url: context.url.clone(),
        status: Some(status),
        request_headers: context.request_headers.clone(),
        response_headers: response_headers.clone(),
        request_body: context.request_body.clone(),
        response_body: Some(TransportBody::Text(body.clone())),
        response_size: Some(body.len()),
        error: Some(format!("HTTP {status}: {sanitized}")),
        is_stream: context.is_stream,
    });
    TransportError::HttpStatus {
        status,
        body,
        retry_after_ms,
        sanitized,
        headers: response_headers,
    }
}

pub(crate) fn header_pairs(headers: &http::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                name.to_string(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

pub(crate) fn parse_retry_after_ms(value: &str) -> Option<u64> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .map(|seconds| seconds * 1000)
}
