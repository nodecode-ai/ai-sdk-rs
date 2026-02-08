//! Example of how providers can use the streaming-sse crate

use crate::ai_sdk_core::{Event, SdkError};
use crate::ai_sdk_streaming_sse::{ProviderChunk, SseDecoder, SseEvent};
use bytes::Bytes;

/// Example provider implementation
#[derive(Default)]
struct OpenAIProvider;

impl ProviderChunk for OpenAIProvider {
    fn try_from_sse(&mut self, event: &SseEvent) -> Result<Option<Vec<Event>>, SdkError> {
        // Check for special "[DONE]" marker
        if event.data == Bytes::from("[DONE]") {
            return Ok(Some(vec![Event::Done]));
        }

        // Parse JSON from event data
        let json: serde_json::Value =
            serde_json::from_slice(&event.data).map_err(SdkError::Serde)?;

        // Extract text deltas from OpenAI response format
        if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta").and_then(|v| v.as_object()) {
                    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                        if !content.is_empty() {
                            return Ok(Some(vec![Event::TextDelta {
                                delta: content.to_string(),
                            }]));
                        }
                    }
                }
            }
        }

        // Ignore events we don't recognize
        Ok(None)
    }
}

fn main() {
    // Example of using the decoder with chunked input
    let mut decoder = SseDecoder::new();

    // Simulating chunks received from network
    let chunk1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n";
    let chunk2 = b"\ndata: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n";
    let chunk3 = b"data: [DONE]\n\n";

    // Process first chunk (incomplete)
    let events1: Vec<_> = decoder.push(chunk1).collect();
    println!("After chunk1: {} events", events1.len());

    // Process second chunk (completes first event)
    let events2: Vec<_> = decoder.push(chunk2).collect();
    println!("After chunk2: {} events", events2.len());
    let mut parser = OpenAIProvider::default();
    for event in events2 {
        if let Ok(Some(parsed)) = parser.try_from_sse(&event) {
            for ev in parsed {
                println!("Parsed event: {:?}", ev);
            }
        }
    }

    // Process final chunk
    let events3: Vec<_> = decoder.push(chunk3).collect();
    println!("After chunk3: {} events", events3.len());
    for event in events3 {
        if let Ok(Some(parsed)) = parser.try_from_sse(&event) {
            for ev in parsed {
                println!("Parsed event: {:?}", ev);
            }
        }
    }
}

#[cfg(feature = "stream")]
async fn example_with_stream() {
    use crate::ai_sdk_streaming_sse::SseStreamExt;
    use futures_util::StreamExt;

    // Simulate a byte stream from HTTP response
    let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
        Ok(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Streaming\"}}]}\n\n",
        )),
        Ok(Bytes::from(
            "data: {\"choices\":[{\"delta\":{\"content\":\" example\"}}]}\n\n",
        )),
        Ok(Bytes::from("data: [DONE]\n\n")),
    ];

    let byte_stream = futures_util::stream::iter(chunks);
    let mut sse_stream = byte_stream.into_sse_stream();

    let mut parser = OpenAIProvider::default();
    while let Some(result) = sse_stream.next().await {
        match result {
            Ok(event) => {
                if let Ok(Some(parsed)) = parser.try_from_sse(&event) {
                    for ev in parsed {
                        println!("Streamed event: {:?}", ev);
                    }
                }
            }
            Err(e) => eprintln!("Stream error: {:?}", e),
        }
    }
}
