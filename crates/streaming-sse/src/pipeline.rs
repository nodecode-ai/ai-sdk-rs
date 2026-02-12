//! Generic SSE to Event streaming pipeline
//!
//! This module provides a reusable pipeline for converting raw SSE byte streams
//! into typed Event streams using provider-specific parsers.

use crate::ai_sdk_core::SdkError;
use crate::ai_sdk_types::Event;
use crate::streaming_sse::{ProviderChunk, SseDecoder};
use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use std::marker::PhantomData;

/// Convert a raw SSE byte stream into typed Events using a provider-specific parser
///
/// This function provides a standardized pipeline that:
/// 1. Accepts a byte stream from HTTP/transport layer
/// 2. Decodes SSE frames using SseDecoder
/// 3. Parses frames using the provider's ProviderChunk implementation
/// 4. Yields Events to the consumer
///
/// # Type Parameters
/// - `S`: The input byte stream
/// - `P`: The provider-specific chunk parser implementing ProviderChunk
/// - `E`: The error type from the transport layer
///
/// # Example
/// ```ignore
/// struct MyProviderChunk;
/// impl ProviderChunk for MyProviderChunk { ... }
///
/// let byte_stream = transport.stream_bytes().await?;
/// let events = sse_to_events::<_, MyProviderChunk, _>(byte_stream);
/// ```
pub fn sse_to_events<S, P, E>(bytes: S) -> impl Stream<Item = Result<Event, SdkError>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    P: ProviderChunk + Default + Send + 'static,
    E: Into<SdkError> + Send + 'static,
{
    async_stream::try_stream! {
        let mut decoder = SseDecoder::new();
        let mut parser = P::default();
        futures_util::pin_mut!(bytes);
        // Per-stream transformation state (across all chunks)
        static REASONING_ID: &str = "reasoning:0";
        let mut reasoning_started = false;

        let mut map_event = |event: Event| -> (Vec<Event>, bool) {
            match event {
                Event::ReasoningStart { .. } => {
                    if !reasoning_started {
                        reasoning_started = true;
                    }
                    (vec![event], false)
                }
                Event::ReasoningDelta { delta } => {
                    if !reasoning_started {
                        reasoning_started = true;
                        return (
                            vec![
                                Event::ReasoningStart {
                                    id: REASONING_ID.to_string(),
                                },
                                Event::ReasoningDelta { delta },
                            ],
                            false,
                        );
                    }
                    (vec![Event::ReasoningDelta { delta }], false)
                }
                Event::ReasoningEnd => {
                    if reasoning_started {
                        reasoning_started = false;
                    }
                    (vec![Event::ReasoningEnd], false)
                }
                Event::Done => {
                    if reasoning_started {
                        reasoning_started = false;
                        return (vec![Event::ReasoningEnd, Event::Done], true);
                    }
                    (vec![Event::Done], true)
                }
                event @ Event::Error { .. } => (vec![event], true),
                other => (vec![other], false),
            }
        };

        while let Some(chunk_result) = bytes.next().await {
            let chunk = chunk_result.map_err(|e| e.into())?;

            // Process all complete SSE events from this chunk
            for sse_event in decoder.push(&chunk) {
                let events = parser.try_from_sse(&sse_event)?.unwrap_or_default();
                for event in events {
                    let (out, stop) = map_event(event);
                    for ev in out {
                        yield ev;
                    }
                    if stop {
                        return;
                    }
                }
            }
        }

        for sse_event in decoder.finish() {
            let events = parser.try_from_sse(&sse_event)?.unwrap_or_default();
            for event in events {
                let (out, stop) = map_event(event);
                for ev in out {
                    yield ev;
                }
                if stop {
                    return;
                }
            }
        }

        yield Event::Error {
            message: "Unexpected EOF".into(),
        };
    }
}

/// Builder for configuring SSE to Event pipeline
pub struct PipelineBuilder<P> {
    provider_name: Option<&'static str>,
    include_raw: bool,
    _phantom: PhantomData<P>,
}

impl<P: ProviderChunk + Default + Send + 'static> PipelineBuilder<P> {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            provider_name: None,
            include_raw: false,
            _phantom: PhantomData,
        }
    }

    /// Set the provider name for metrics
    pub fn with_provider(mut self, name: &'static str) -> Self {
        self.provider_name = Some(name);
        self
    }

    /// Include provider raw chunks (JSON if possible, else string) in the event stream.
    /// Default is false.
    pub fn include_raw(mut self, include: bool) -> Self {
        self.include_raw = include;
        self
    }

    /// Build the pipeline for the given byte stream
    pub fn build<S, E>(self, bytes: S) -> impl Stream<Item = Result<Event, SdkError>>
    where
        S: Stream<Item = Result<Bytes, E>> + Send + 'static,
        E: Into<SdkError> + Send + 'static,
    {
        let include_raw = self.include_raw;
        match self.provider_name {
            Some(_provider_name) => {
                // Inline pipeline with raw support
                async_stream::try_stream! {
                    let mut decoder = SseDecoder::new();
                    let mut parser = P::default();
                    futures_util::pin_mut!(bytes);
                    static REASONING_ID: &str = "reasoning:0";
                    let mut reasoning_started = false;

                    while let Some(chunk_result) = bytes.next().await {
                        let chunk = chunk_result.map_err(|e| e.into())?;

                        for sse_event in decoder.push(&chunk) {
                            if include_raw {
                                let raw_val: serde_json::Value = match serde_json::from_slice(&sse_event.data) {
                                    Ok(v) => v,
                                    Err(_) => serde_json::Value::String(String::from_utf8_lossy(&sse_event.data).to_string()),
                                };
                                yield Event::Raw { raw_value: raw_val };
                            }

                            if let Some(events) = parser.try_from_sse(&sse_event)? {
                                for event in events {
                                    match event {
                                        Event::ReasoningStart { .. } => {
                                            if !reasoning_started {
                                                reasoning_started = true;
                                            }
                                            yield event;
                                        }
                                        Event::ReasoningDelta { delta } => {
                                            if !reasoning_started {
                                                reasoning_started = true;
                                                yield Event::ReasoningStart { id: REASONING_ID.to_string() };
                                            }
                                            yield Event::ReasoningDelta { delta };
                                        }
                                        Event::ReasoningEnd => {
                                            if reasoning_started {
                                                reasoning_started = false;
                                            }
                                            yield Event::ReasoningEnd;
                                        }
                                        event @ Event::Error { .. } => {
                                            yield event;
                                            return;
                                        }
                                        Event::Done => {
                                            if reasoning_started {
                                                yield Event::ReasoningEnd;
                                            }
                                            yield Event::Done;
                                            return;
                                        }
                                        other => {
                                            yield other;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    yield Event::Error {
                        message: "Unexpected EOF".into(),
                    };
                }.boxed()
            }
            None => {
                if !include_raw {
                    sse_to_events::<S, P, E>(bytes).boxed()
                } else {
                    // Inline pipeline without metrics + raw support
                    async_stream::try_stream! {
                        let mut decoder = SseDecoder::new();
                        let mut parser = P::default();
                        futures_util::pin_mut!(bytes);
                        static REASONING_ID: &str = "reasoning:0";
                        let mut reasoning_started = false;

                        while let Some(chunk_result) = bytes.next().await {
                            let chunk = chunk_result.map_err(|e| e.into())?;
                            for sse_event in decoder.push(&chunk) {
                                if include_raw {
                                    let raw_val: serde_json::Value = match serde_json::from_slice(&sse_event.data) {
                                        Ok(v) => v,
                                        Err(_) => serde_json::Value::String(String::from_utf8_lossy(&sse_event.data).to_string()),
                                    };
                                    yield Event::Raw { raw_value: raw_val };
                                }
                                if let Some(events) = parser.try_from_sse(&sse_event)? {
                                    for event in events {
                                        match event {
                                            Event::ReasoningStart { .. } => {
                                                if !reasoning_started {
                                                    reasoning_started = true;
                                                }
                                                yield event;
                                            }
                                            Event::ReasoningDelta { delta } => {
                                                if !reasoning_started {
                                                    reasoning_started = true;
                                                    yield Event::ReasoningStart { id: REASONING_ID.to_string() };
                                                }
                                                yield Event::ReasoningDelta { delta };
                                            }
                                            Event::ReasoningEnd => {
                                                if reasoning_started {
                                                    reasoning_started = false;
                                                }
                                                yield Event::ReasoningEnd;
                                            }
                                            event @ Event::Error { .. } => {
                                                yield event;
                                                return;
                                            }
                                            Event::Done => {
                                                if reasoning_started {
                                                    yield Event::ReasoningEnd;
                                                }
                                                yield Event::Done;
                                                return;
                                            }
                                            other => {
                                                yield other;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        yield Event::Error {
                            message: "Unexpected EOF".into(),
                        };
                    }.boxed()
                }
            }
        }
    }
}

impl<P: ProviderChunk + Default + Send + 'static> Default for PipelineBuilder<P> {
    fn default() -> Self {
        Self::new()
    }
}
