use crate::ai_sdk_core::SdkError;
use crate::ai_sdk_providers_openai_compatible::{build_stream, StreamMode, StreamSettings};
use crate::ai_sdk_types::v2 as v2t;
use bytes::Bytes;
use futures_util::stream;
use futures_util::TryStreamExt;
use serde_json::json;

fn chunk(data: impl Into<String>) -> Result<Bytes, SdkError> {
    Ok(Bytes::from(data.into()))
}

fn json_chunk(value: serde_json::Value) -> Result<Bytes, SdkError> {
    chunk(format!("data: {}\n\n", value))
}

#[tokio::test]
async fn streams_text_and_usage() {
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "model":"grok-beta",
                "created":123,
                "choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]
            })),
            json_chunk(json!({
                "id":"chat-1",
                "choices":[{"index":0,"delta":{"content":"Hello"}}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: true,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    assert_eq!(parts.len(), 6);
    match &parts[0] {
        v2t::StreamPart::StreamStart { warnings } => assert!(warnings.is_empty()),
        other => panic!("expected stream start, got {other:?}"),
    }
    match &parts[1] {
        v2t::StreamPart::ResponseMetadata { meta } => {
            assert_eq!(meta.id.as_deref(), Some("chat-1"));
            assert_eq!(meta.model_id.as_deref(), Some("grok-beta"));
            assert_eq!(meta.timestamp_ms, Some(123000));
        }
        other => panic!("expected response metadata, got {other:?}"),
    }
    match &parts[2] {
        v2t::StreamPart::TextStart { id, .. } => assert_eq!(id, "txt-0"),
        other => panic!("expected text start, got {other:?}"),
    }
    match &parts[3] {
        v2t::StreamPart::TextDelta { id, delta, .. } => {
            assert_eq!(id, "txt-0");
            assert_eq!(delta, "Hello");
        }
        other => panic!("expected text delta, got {other:?}"),
    }
    match &parts[4] {
        v2t::StreamPart::TextEnd { id, .. } => assert_eq!(id, "txt-0"),
        other => panic!("expected text end, got {other:?}"),
    }
    match &parts[5] {
        v2t::StreamPart::Finish {
            usage,
            finish_reason,
            ..
        } => {
            match finish_reason {
                v2t::FinishReason::Stop => {}
                other => panic!("expected stop finish reason, got {other:?}"),
            }
            assert_eq!(usage.input_tokens, Some(3));
            assert_eq!(usage.output_tokens, Some(4));
            assert_eq!(usage.total_tokens, Some(7));
        }
        other => panic!("expected finish, got {other:?}"),
    }
}

#[tokio::test]
async fn streams_reasoning_content_before_text() {
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "model":"grok-beta",
                "created":456,
                "choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"From reasoning_content","reasoning":"From reasoning"},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"content":"Final response"},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{},"finish_reason":"stop"}]
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: false,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    assert_eq!(parts.len(), 9);
    match &parts[0] {
        v2t::StreamPart::StreamStart { .. } => {}
        other => panic!("expected stream start, got {other:?}"),
    }
    match &parts[1] {
        v2t::StreamPart::ResponseMetadata { meta } => {
            assert_eq!(meta.id.as_deref(), Some("chat-1"));
            assert_eq!(meta.model_id.as_deref(), Some("grok-beta"));
            assert_eq!(meta.timestamp_ms, Some(456000));
        }
        other => panic!("expected response metadata, got {other:?}"),
    }
    match &parts[2] {
        v2t::StreamPart::ReasoningStart { id, .. } => assert_eq!(id, "reasoning-0"),
        other => panic!("expected reasoning start, got {other:?}"),
    }
    match &parts[3] {
        v2t::StreamPart::ReasoningDelta { id, delta, .. } => {
            assert_eq!(id, "reasoning-0");
            assert_eq!(delta, "From reasoning_content");
        }
        other => panic!("expected reasoning delta, got {other:?}"),
    }
    match &parts[4] {
        v2t::StreamPart::TextStart { id, .. } => assert_eq!(id, "txt-0"),
        other => panic!("expected text start, got {other:?}"),
    }
    match &parts[5] {
        v2t::StreamPart::TextDelta { id, delta, .. } => {
            assert_eq!(id, "txt-0");
            assert_eq!(delta, "Final response");
        }
        other => panic!("expected text delta, got {other:?}"),
    }
    match &parts[6] {
        v2t::StreamPart::ReasoningEnd { id, .. } => assert_eq!(id, "reasoning-0"),
        other => panic!("expected reasoning end, got {other:?}"),
    }
    match &parts[7] {
        v2t::StreamPart::TextEnd { id, .. } => assert_eq!(id, "txt-0"),
        other => panic!("expected text end, got {other:?}"),
    }
    match &parts[8] {
        v2t::StreamPart::Finish { finish_reason, .. } => match finish_reason {
            v2t::FinishReason::Stop => {}
            other => panic!("expected stop finish reason, got {other:?}"),
        },
        other => panic!("expected finish, got {other:?}"),
    }
}

#[tokio::test]
async fn streams_tool_calls_and_finish_reason() {
    let arg_segments = ["", "{\"", "value", "\":\"", "Spark", "le", " Day", "\"}"];
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments": arg_segments[0]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[1]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[2]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[3]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[4]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[5]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[6]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[7]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{},"finish_reason":"tool_calls"}],
                "usage":{"prompt_tokens":18,"completion_tokens":439,"total_tokens":457}
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: true,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    let tool_call = parts
        .iter()
        .find_map(|p| match p {
            v2t::StreamPart::ToolCall(tc) => Some(tc.clone()),
            _ => None,
        })
        .expect("tool call");
    assert_eq!(tool_call.tool_call_id, "call_O17Uplv4lJvD6DVdIvFFeRMw");
    assert_eq!(tool_call.tool_name, "test-tool");
    assert_eq!(tool_call.input, r#"{"value":"Sparkle Day"}"#);

    let finish = parts
        .iter()
        .find_map(|p| match p {
            v2t::StreamPart::Finish {
                finish_reason,
                usage,
                ..
            } => Some((finish_reason.clone(), usage.clone())),
            _ => None,
        })
        .expect("finish part");
    match finish.0 {
        v2t::FinishReason::ToolCalls => {}
        other => panic!("expected tool_calls finish reason, got {other:?}"),
    }
    assert_eq!(finish.1.input_tokens, Some(18));
    assert_eq!(finish.1.output_tokens, Some(439));
    assert_eq!(finish.1.total_tokens, Some(457));
}

#[tokio::test]
async fn streams_tool_input_deltas_as_fragments() {
    let arg_segments = ["", "{\"", "value", "\":\"", "Spark", "le", " Day", "\"}"];
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_fragmented","type":"function","function":{"name":"test-tool","arguments": arg_segments[0]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[1]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[2]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[3]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[4]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[5]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[6]}}]},"finish_reason":null}]
            })),
            json_chunk(json!({
                "choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments": arg_segments[7]}}]},"finish_reason":null}]
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: false,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    let starts = parts
        .iter()
        .filter(|part| matches!(part, v2t::StreamPart::ToolInputStart { id, .. } if id == "call_fragmented"))
        .count();
    assert_eq!(starts, 1);

    let deltas: Vec<String> = parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::ToolInputDelta { id, delta, .. } if id == "call_fragmented" => {
                Some(delta.clone())
            }
            _ => None,
        })
        .collect();
    let expected: Vec<String> = arg_segments[1..].iter().map(|s| s.to_string()).collect();
    assert_eq!(deltas, expected);
}

#[tokio::test]
async fn errors_on_missing_tool_call_id() {
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":null,"function":{"name":"test-tool","arguments":"{}"}}]},"finish_reason":null}]
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: false,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    let error = parts.iter().find_map(|part| match part {
        v2t::StreamPart::Error { error } => error.get("message").and_then(|v| v.as_str()),
        _ => None,
    });
    assert_eq!(error, Some("Expected 'id' to be a string."));
    assert!(!parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::Finish { .. })));
}

#[tokio::test]
async fn errors_on_missing_tool_call_name() {
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_missing_name","function":{"name":null,"arguments":"{}"}}]},"finish_reason":null}]
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: false,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    let error = parts.iter().find_map(|part| match part {
        v2t::StreamPart::Error { error } => error.get("message").and_then(|v| v.as_str()),
        _ => None,
    });
    assert_eq!(error, Some("Expected 'function.name' to be a string."));
    assert!(!parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::Finish { .. })));
}

#[tokio::test]
async fn streams_usage_with_cache_reasoning_and_prediction_tokens() {
    let parts: Vec<v2t::StreamPart> = build_stream(
        stream::iter(vec![
            json_chunk(json!({
                "id":"chat-1",
                "model":"grok-beta",
                "created":789,
                "choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
                "usage":{
                    "prompt_tokens":10,
                    "completion_tokens":5,
                    "total_tokens":15,
                    "prompt_tokens_details":{"cached_tokens":4},
                    "completion_tokens_details":{
                        "reasoning_tokens":2,
                        "accepted_prediction_tokens":3,
                        "rejected_prediction_tokens":1
                    }
                }
            })),
            chunk("data: [DONE]\n\n"),
        ]),
        StreamSettings {
            warnings: vec![],
            include_raw: false,
            include_usage: true,
            provider_scope_name: "openai-compatible".into(),
        },
        StreamMode::Chat,
    )
    .try_collect()
    .await
    .expect("stream parts");

    let finish = parts.iter().find_map(|part| match part {
        v2t::StreamPart::Finish {
            usage,
            provider_metadata,
            ..
        } => Some((usage.clone(), provider_metadata.clone())),
        _ => None,
    });
    let (usage, provider_metadata) = finish.expect("finish part");
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.output_tokens, Some(5));
    assert_eq!(usage.total_tokens, Some(15));
    assert_eq!(usage.cached_input_tokens, Some(4));
    assert_eq!(usage.reasoning_tokens, Some(2));

    let provider_metadata = provider_metadata.expect("provider metadata");
    let openai_meta = provider_metadata
        .get("openai-compatible")
        .expect("openai-compatible metadata");
    assert_eq!(openai_meta.get("acceptedPredictionTokens"), Some(&json!(3)));
    assert_eq!(openai_meta.get("rejectedPredictionTokens"), Some(&json!(1)));
}
