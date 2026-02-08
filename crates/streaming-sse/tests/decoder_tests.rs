use crate::ai_sdk_streaming_sse::SseDecoder;
use bytes::Bytes;

#[test]
fn test_simple_event() {
    let mut decoder = SseDecoder::new();
    let events: Vec<_> = decoder.push(b"data: hello world\n\n").collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, Bytes::from("hello world"));
    assert_eq!(events[0].event, None);
}

#[test]
fn test_complete_event() {
    let mut decoder = SseDecoder::new();
    let data = b"id: 123\nevent: message\ndata: test data\nretry: 5000\n\n";
    let events: Vec<_> = decoder.push(data).collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].id, Some("123".to_string()));
    assert_eq!(events[0].event, Some("message".to_string()));
    assert_eq!(events[0].data, Bytes::from("test data"));
    assert_eq!(events[0].retry, Some(5000));
}

#[test]
fn test_multiline_data() {
    let mut decoder = SseDecoder::new();
    let data = b"data: line 1\ndata: line 2\ndata: line 3\n\n";
    let events: Vec<_> = decoder.push(data).collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, Bytes::from("line 1\nline 2\nline 3"));
}

#[test]
fn test_chunked_input() {
    let mut decoder = SseDecoder::new();

    let events1: Vec<_> = decoder.push(b"data: hello").collect();
    assert_eq!(events1.len(), 0);

    let events2: Vec<_> = decoder.push(b" world\n\n").collect();
    assert_eq!(events2.len(), 1);
    assert_eq!(events2[0].data, Bytes::from("hello world"));
}

#[test]
fn test_crlf_terminated_event() {
    let mut decoder = SseDecoder::new();
    let data = b"data: hello world\r\n\r\n";
    let events: Vec<_> = decoder.push(data).collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, Bytes::from("hello world"));
}

#[test]
fn test_crlf_split_across_chunks() {
    let mut decoder = SseDecoder::new();
    let first = b"data: chunked\r\n";
    let second = b"\r\n";

    let events_first: Vec<_> = decoder.push(first).collect();
    assert!(events_first.is_empty());

    let events_second: Vec<_> = decoder.push(second).collect();
    assert_eq!(events_second.len(), 1);
    assert_eq!(events_second[0].data, Bytes::from("chunked"));
}

#[test]
fn test_multiple_events() {
    let mut decoder = SseDecoder::new();
    let data = b"data: event1\n\ndata: event2\n\n";
    let events: Vec<_> = decoder.push(data).collect();

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].data, Bytes::from("event1"));
    assert_eq!(events[1].data, Bytes::from("event2"));
}

#[test]
fn test_finish_flushes_partial_event() {
    let mut decoder = SseDecoder::new();
    let data = b"data: trailing-only\r\n";
    let events: Vec<_> = decoder.push(data).collect();
    assert!(events.is_empty());

    let flushed: Vec<_> = decoder.finish().collect();
    assert_eq!(flushed.len(), 1);
    assert_eq!(flushed[0].data, Bytes::from("trailing-only"));
}

#[test]
fn test_comment_lines() {
    let mut decoder = SseDecoder::new();
    let data = b": this is a comment\ndata: actual data\n: another comment\n\n";
    let events: Vec<_> = decoder.push(data).collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, Bytes::from("actual data"));
}

#[test]
fn test_field_without_colon() {
    let mut decoder = SseDecoder::new();
    let data = b"data\nevent\n\n";
    let events: Vec<_> = decoder.push(data).collect();

    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data, Bytes::from(""));
    assert_eq!(events[0].event, Some("".to_string()));
}

#[test]
fn test_very_chunked_multiline() {
    let mut decoder = SseDecoder::new();

    assert_eq!(decoder.push(b"da").count(), 0);
    assert_eq!(decoder.push(b"ta: li").count(), 0);
    assert_eq!(decoder.push(b"ne 1\nda").count(), 0);
    assert_eq!(decoder.push(b"ta: line 2").count(), 0);
    assert_eq!(decoder.push(b"\n\n").count(), 1);

    let events: Vec<_> = decoder.push(b"").collect();
    assert_eq!(events.len(), 0);
}
