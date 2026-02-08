use crate::ai_sdk_streaming_sse::stream::SseStreamExt;
use bytes::Bytes;
use futures_util::stream;
use futures_util::StreamExt;

#[tokio::test]
async fn test_stream_adapter() {
    use std::io;

    let chunks: Vec<Result<Bytes, io::Error>> = vec![
        Ok(Bytes::from("data: hello")),
        Ok(Bytes::from(" world\n\n")),
        Ok(Bytes::from("data: second event\n\n")),
    ];

    let byte_stream = stream::iter(chunks);
    let mut sse_stream = byte_stream.into_sse_stream();

    let event1 = sse_stream.next().await.unwrap().unwrap();
    assert_eq!(event1.data, Bytes::from("hello world"));

    let event2 = sse_stream.next().await.unwrap().unwrap();
    assert_eq!(event2.data, Bytes::from("second event"));

    assert!(sse_stream.next().await.is_none());
}

#[tokio::test]
async fn test_stream_with_errors() {
    use std::io;

    let chunks: Vec<Result<Bytes, io::Error>> = vec![
        Ok(Bytes::from("data: event1\n\n")),
        Err(io::Error::new(io::ErrorKind::Other, "network error")),
        Ok(Bytes::from("data: event2\n\n")),
    ];

    let byte_stream = stream::iter(chunks);
    let mut sse_stream = byte_stream.into_sse_stream();

    let event1 = sse_stream.next().await.unwrap().unwrap();
    assert_eq!(event1.data, Bytes::from("event1"));

    let error = sse_stream.next().await.unwrap();
    assert!(error.is_err());

    let event2 = sse_stream.next().await.unwrap().unwrap();
    assert_eq!(event2.data, Bytes::from("event2"));

    assert!(sse_stream.next().await.is_none());
}
