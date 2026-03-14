//! # Server-Sent Events (SSE) Parsing Library
//!
//! Transport-agnostic SSE parsing for the ai-sdk-rs ecosystem.
//!
//! This crate provides:
//! - `SseEvent`: Core SSE event representation
//! - `SseDecoder`: Incremental SSE frame decoder
//! - `ProviderChunk`: Trait for provider-specific event parsing

use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_types::Event;
use bytes::Bytes;
use std::collections::VecDeque;

/// Represents a single Server-Sent Event
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// Event type (optional)
    pub event: Option<String>,
    /// Event data payload
    pub data: Bytes,
    /// Event ID (optional)
    pub id: Option<String>,
    /// Retry timeout in milliseconds (optional)
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create a new SSE event with just data
    pub fn data(data: impl Into<Bytes>) -> Self {
        Self {
            event: None,
            data: data.into(),
            id: None,
            retry: None,
        }
    }

    /// Set the event type
    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    /// Set the event ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// Incremental SSE decoder that handles chunk boundaries correctly
pub struct SseDecoder {
    /// Internal buffer for incomplete frames
    buffer: Vec<u8>,
    /// Current event being built
    current_event: EventBuilder,
    /// Queue of completed events ready to be yielded
    event_queue: VecDeque<SseEvent>,
}

impl SseDecoder {
    /// Create a new SSE decoder
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_event: EventBuilder::new(),
            event_queue: VecDeque::new(),
        }
    }

    /// Push new data chunk and get any complete events
    ///
    /// This method handles partial chunks correctly and will buffer
    /// incomplete events until they are complete.
    pub fn push(&mut self, chunk: &[u8]) -> impl Iterator<Item = SseEvent> + '_ {
        self.buffer.extend_from_slice(chunk);
        self.process_buffer();
        self.event_queue.drain(..)
    }

    /// Process the internal buffer to extract complete events
    fn process_buffer(&mut self) {
        // Process all complete frames (ending with a blank line)
        while let Some(frame_end) = find_event_terminator(&self.buffer) {
            // Extract frame including the terminator characters
            let frame = self.buffer.drain(..=frame_end).collect::<Vec<_>>();

            // Process the frame line by line
            self.process_frame(&frame);
        }
    }

    /// Process a complete SSE frame
    fn process_frame(&mut self, frame: &[u8]) {
        let frame_str = String::from_utf8_lossy(frame);

        for line in frame_str.lines() {
            if line.is_empty() {
                // Empty line signals end of event
                if self.current_event.has_data() {
                    let builder = std::mem::replace(&mut self.current_event, EventBuilder::new());
                    if let Some(event) = builder.build() {
                        self.event_queue.push_back(event);
                    }
                }
            } else if line.starts_with(':') {
                // Comment line, ignore
                continue;
            } else {
                // Parse field:value
                let (field, value) = if let Some(colon_pos) = line.find(':') {
                    let (field, rest) = line.split_at(colon_pos);
                    let value = rest.trim_start_matches(':').trim_start();
                    (field, value)
                } else {
                    (line, "")
                };

                match field {
                    "data" => self.current_event.append_data(value),
                    "event" => self.current_event.set_event(value),
                    "id" => self.current_event.set_id(value),
                    "retry" => {
                        if let Ok(retry) = value.parse::<u64>() {
                            self.current_event.set_retry(retry);
                        }
                    }
                    _ => {} // Unknown field, ignore
                }
            }
        }

        // Handle any remaining event at frame end
        if self.current_event.has_data() {
            let builder = std::mem::replace(&mut self.current_event, EventBuilder::new());
            if let Some(event) = builder.build() {
                self.event_queue.push_back(event);
            }
        }
    }

    /// Check if decoder has buffered data that might be incomplete
    pub fn has_buffered_data(&self) -> bool {
        !self.buffer.is_empty() || self.current_event.has_data()
    }

    /// Finalize the stream and flush any buffered event fragments.
    ///
    /// Some providers close the connection immediately after the final `data:`
    /// payload without sending the trailing blank line that normally
    /// terminates an SSE event. Appending a synthetic blank line ensures the
    /// final event is emitted instead of being dropped on stream shutdown.
    pub fn finish(&mut self) -> impl Iterator<Item = SseEvent> + '_ {
        if self.has_buffered_data() {
            // Append a synthetic blank line (`\n\n`) and process it like a
            // normal frame boundary. This handles `\r`, `\n`, and `\r\n`
            // mixtures because `process_frame` trims newline variants when
            // splitting lines.
            self.buffer.extend_from_slice(b"\n\n");
            self.process_buffer();
        }
        self.event_queue.drain(..)
    }
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal builder for SSE events
struct EventBuilder {
    event: Option<String>,
    data_parts: Vec<String>,
    id: Option<String>,
    retry: Option<u64>,
}

impl EventBuilder {
    fn new() -> Self {
        Self {
            event: None,
            data_parts: Vec::new(),
            id: None,
            retry: None,
        }
    }

    fn append_data(&mut self, data: &str) {
        self.data_parts.push(data.to_string());
    }

    fn set_event(&mut self, event: &str) {
        self.event = Some(event.to_string());
    }

    fn set_id(&mut self, id: &str) {
        self.id = Some(id.to_string());
    }

    fn set_retry(&mut self, retry: u64) {
        self.retry = Some(retry);
    }

    fn has_data(&self) -> bool {
        !self.data_parts.is_empty()
    }

    fn build(self) -> Option<SseEvent> {
        if self.data_parts.is_empty() {
            return None;
        }

        let data = self.data_parts.join("\n");
        Some(SseEvent {
            event: self.event,
            data: Bytes::from(data),
            id: self.id,
            retry: self.retry,
        })
    }
}

/// Find the index of the final newline character that terminates an SSE event.
fn find_event_terminator(buf: &[u8]) -> Option<usize> {
    let mut idx = 0;
    let mut line_start = 0;
    while idx < buf.len() {
        match buf[idx] {
            b'\n' => {
                if idx == line_start {
                    return Some(idx);
                }
                idx += 1;
                line_start = idx;
            }
            b'\r' => {
                // Need at least one more byte to know if this is CR or CRLF.
                if idx + 1 >= buf.len() {
                    return None;
                }
                let mut terminator_len = 1;
                if buf[idx + 1] == b'\n' {
                    terminator_len = 2;
                }
                if idx == line_start {
                    return Some(idx + terminator_len - 1);
                }
                idx += terminator_len;
                line_start = idx;
            }
            _ => {
                idx += 1;
            }
        }
    }
    None
}

/// Trait for provider-specific chunk parsing
///
/// Implement this trait to convert SSE events into unified Event types.
pub trait ProviderChunk {
    /// Try to parse an SSE event into unified events
    ///
    /// Returns:
    /// - `Ok(Some(events))` if the SSE event was successfully parsed
    /// - `Ok(None)` if the SSE event should be ignored (e.g., heartbeat)
    /// - `Err(e)` if parsing failed
    fn try_from_sse(&mut self, event: &SseEvent) -> Result<Option<Vec<Event>>, SdkError>;
}

// Optional: Stream support when the feature is enabled
#[cfg(feature = "stream")]
pub mod stream;

#[cfg(feature = "stream")]
pub use stream::{SseStream, SseStreamExt};

// Pipeline module for unified SSE to Event conversion
pub mod pipeline;
pub use pipeline::{sse_to_events, PipelineBuilder};
