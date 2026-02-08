use crate::ai_sdk_core::SdkError;
use crate::ai_sdk_streaming_sse::{sse_to_events, PipelineBuilder, ProviderChunk, SseEvent};
use crate::ai_sdk_types::Event;
use bytes::Bytes;
use futures_util::stream;
use futures_util::StreamExt;

#[derive(Default)]
struct TestProvider;

impl ProviderChunk for TestProvider {
    fn try_from_sse(&mut self, event: &SseEvent) -> Result<Option<Vec<Event>>, SdkError> {
        let data = std::str::from_utf8(&event.data).unwrap_or_default();
        let event = match data {
            "done" => Event::Done,
            "error" => Event::Error {
                message: "boom".into(),
            },
            other => Event::TextDelta {
                delta: other.to_string(),
            },
        };
        Ok(Some(vec![event]))
    }
}

fn sse_stream(lines: &[&str]) -> impl futures_core::Stream<Item = Result<Bytes, SdkError>> {
    let chunks = lines
        .iter()
        .map(|line| Ok::<Bytes, SdkError>(Bytes::from(format!("data: {}\n\n", line))))
        .collect::<Vec<_>>();
    stream::iter(chunks)
}

#[tokio::test]
async fn unexpected_eof_emits_error() {
    let events: Vec<Event> = sse_to_events::<_, TestProvider, _>(sse_stream(&["hello"]))
        .filter_map(|res| async { res.ok() })
        .collect()
        .await;

    assert!(matches!(
        events.first(),
        Some(Event::TextDelta { delta }) if delta == "hello"
    ));
    assert!(matches!(
        events.last(),
        Some(Event::Error { message }) if message == "Unexpected EOF"
    ));
    assert!(!events.iter().any(|event| matches!(event, Event::Done)));
}

#[tokio::test]
async fn eof_flushes_partial_event() {
    let chunks = vec![Ok::<Bytes, SdkError>(Bytes::from("data: hello\n"))];
    let events: Vec<Event> = sse_to_events::<_, TestProvider, _>(stream::iter(chunks))
        .filter_map(|res| async { res.ok() })
        .collect()
        .await;

    assert!(matches!(
        events.first(),
        Some(Event::TextDelta { delta }) if delta == "hello"
    ));
    assert!(matches!(
        events.last(),
        Some(Event::Error { message }) if message == "Unexpected EOF"
    ));
}

#[tokio::test]
async fn error_terminates_without_done() {
    let pipeline = PipelineBuilder::<TestProvider>::new()
        .with_provider("test")
        .build(sse_stream(&["error", "done"]));
    let events: Vec<Event> = pipeline
        .filter_map(|res| async { res.ok() })
        .collect()
        .await;

    assert_eq!(events.len(), 1);
    assert!(matches!(
        events.first(),
        Some(Event::Error { message }) if message == "boom"
    ));
}

#[tokio::test]
async fn done_passes_through() {
    let events: Vec<Event> = sse_to_events::<_, TestProvider, _>(sse_stream(&["done"]))
        .filter_map(|res| async { res.ok() })
        .collect()
        .await;

    assert_eq!(events.len(), 1);
    assert!(matches!(events.first(), Some(Event::Done)));
}
