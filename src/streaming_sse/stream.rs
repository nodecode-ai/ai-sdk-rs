//! Async Stream support for SSE parsing

use crate::streaming_sse::{SseDecoder, SseEvent};
use bytes::Bytes;
use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Adapter that turns a byte stream into an SSE event stream
pub struct SseStream<S> {
    inner: S,
    decoder: SseDecoder,
    pending_events: Vec<SseEvent>,
    current_index: usize,
}

impl<S> SseStream<S> {
    /// Create a new SSE stream from a byte stream
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            decoder: SseDecoder::new(),
            pending_events: Vec::new(),
            current_index: 0,
        }
    }
}

impl<S, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error,
{
    type Item = Result<SseEvent, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // First, yield any pending events from previous chunks
        if self.current_index < self.pending_events.len() {
            let event = self.pending_events[self.current_index].clone();
            self.current_index += 1;
            return Poll::Ready(Some(Ok(event)));
        }

        // Clear the pending events buffer for reuse
        self.pending_events.clear();
        self.current_index = 0;

        // Poll the inner stream for more data
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                // Process the chunk through the decoder
                self.pending_events = self.decoder.push(&chunk).collect();

                // If we got events, return the first one
                if !self.pending_events.is_empty() {
                    let event = self.pending_events[0].clone();
                    self.current_index = 1;
                    Poll::Ready(Some(Ok(event)))
                } else {
                    // No complete events yet, need more data
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                // Stream ended - check if decoder has any buffered data
                // In a well-formed SSE stream, this shouldn't happen,
                // but we handle it gracefully
                if self.decoder.has_buffered_data() {
                    // Force flush with double newline
                    self.pending_events = self.decoder.push(b"\n\n").collect();
                    if !self.pending_events.is_empty() {
                        let event = self.pending_events[0].clone();
                        self.current_index = 1;
                        return Poll::Ready(Some(Ok(event)));
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Extension trait for byte streams to easily convert to SSE streams
pub trait SseStreamExt: Stream {
    /// Convert this byte stream into an SSE event stream
    fn into_sse_stream(self) -> SseStream<Self>
    where
        Self: Sized,
    {
        SseStream::new(self)
    }
}

impl<S: Stream> SseStreamExt for S {}
