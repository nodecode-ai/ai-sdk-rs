use std::collections::HashMap;
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::json;

use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_types::v2 as v2t;
use crate::provider_google::shared::generate_response::{
    parse_google_gen_ai_generate_response, parse_google_vertex_generate_response,
};
use crate::provider_google::shared::options::{
    parse_google_provider_options_for_scopes, GoogleProviderOptions,
};
use crate::provider_google::shared::prepare_tools::prepare_tools as prepare_google_family_tools;
use crate::provider_google::shared::prompt::{
    convert_to_google_prompt_with_scopes, GoogleContent, GoogleContentPart, GooglePrompt,
};
use crate::provider_google::shared::stream_core::build_google_stream_part_stream;

const GOOGLE_SCOPES: &[&str] = &["google"];
const GOOGLE_VERTEX_SCOPES: &[&str] = &["google-vertex", "google"];

fn parse_google_provider_options(opts: &v2t::ProviderOptions) -> Option<GoogleProviderOptions> {
    parse_google_provider_options_for_scopes(opts, GOOGLE_SCOPES)
}

fn parse_google_vertex_provider_options(
    opts: &v2t::ProviderOptions,
) -> Option<GoogleProviderOptions> {
    parse_google_provider_options_for_scopes(opts, GOOGLE_VERTEX_SCOPES)
}

fn convert_google_prompt(prompt: &v2t::Prompt, is_gemma: bool) -> Result<GooglePrompt, SdkError> {
    convert_to_google_prompt_with_scopes(prompt, is_gemma, GOOGLE_SCOPES)
}

fn convert_vertex_prompt(prompt: &v2t::Prompt, is_gemma: bool) -> Result<GooglePrompt, SdkError> {
    convert_to_google_prompt_with_scopes(prompt, is_gemma, GOOGLE_VERTEX_SCOPES)
}

fn provider_tool(id: &str, args: serde_json::Value) -> v2t::Tool {
    v2t::Tool::Provider(v2t::ProviderTool {
        r#type: v2t::ProviderToolType::Provider,
        id: id.to_string(),
        name: id.to_string(),
        args,
    })
}

fn function_tool(name: &str) -> v2t::Tool {
    v2t::Tool::Function(v2t::FunctionTool {
        r#type: v2t::FunctionToolType::Function,
        name: name.to_string(),
        description: Some("test tool".to_string()),
        input_schema: json!({
            "type":"object",
            "properties":{"value":{"type":"string"}},
            "required":["value"],
            "additionalProperties": false
        }),
        strict: None,
        provider_options: None,
    })
}

#[test]
fn prepare_tools_function_path_parity_between_google_and_vertex() {
    let tools = vec![function_tool("calc")];
    let choice = Some(v2t::ToolChoice::Tool {
        name: "calc".to_string(),
    });

    let google = prepare_google_family_tools(&tools, &choice, "gemini-2.5-flash");
    let vertex = prepare_google_family_tools(&tools, &choice, "gemini-2.5-flash");

    assert_eq!(google.tools, vertex.tools);
    assert_eq!(google.tool_config, vertex.tool_config);
    assert_eq!(
        format!("{:?}", google.tool_warnings),
        format!("{:?}", vertex.tool_warnings)
    );
}

#[test]
fn prepare_tools_provider_path_parity_between_google_and_vertex() {
    let tools = vec![provider_tool(
        "google.google_search",
        json!({
            "mode": "MODE_DYNAMIC",
            "dynamicThreshold": 0.7
        }),
    )];

    let google = prepare_google_family_tools(&tools, &None, "gemini-1.5-flash");
    let vertex = prepare_google_family_tools(&tools, &None, "gemini-1.5-flash");

    assert_eq!(google.tools, vertex.tools);
    assert_eq!(google.tool_config, vertex.tool_config);
    assert_eq!(
        format!("{:?}", google.tool_warnings),
        format!("{:?}", vertex.tool_warnings)
    );
}

#[test]
fn provider_option_scope_resolution_matches_expected_behavior() {
    let mut opts = v2t::ProviderOptions::new();
    opts.insert(
        "google".into(),
        HashMap::from([("responseModalities".into(), json!(["TEXT"]))]),
    );
    opts.insert(
        "google-vertex".into(),
        HashMap::from([("responseModalities".into(), json!(["IMAGE"]))]),
    );

    let google = parse_google_provider_options(&opts).expect("google options");
    assert_eq!(google.response_modalities, Some(vec!["TEXT".to_string()]));

    let vertex = parse_google_vertex_provider_options(&opts).expect("vertex options");
    assert_eq!(vertex.response_modalities, Some(vec!["IMAGE".to_string()]));

    let mut fallback_opts = v2t::ProviderOptions::new();
    fallback_opts.insert(
        "google".into(),
        HashMap::from([("responseModalities".into(), json!(["TEXT"]))]),
    );
    let vertex_fallback =
        parse_google_vertex_provider_options(&fallback_opts).expect("vertex fallback options");
    assert_eq!(
        vertex_fallback.response_modalities,
        Some(vec!["TEXT".to_string()])
    );
}

fn first_model_text_signature(prompt: &GooglePrompt) -> Option<String> {
    prompt.contents.iter().find_map(|content| match content {
        GoogleContent::Model { parts } => parts.iter().find_map(|part| match part {
            GoogleContentPart::Text {
                thought_signature, ..
            } => thought_signature.clone(),
            _ => None,
        }),
        _ => None,
    })
}

#[test]
fn prompt_thought_signature_is_scoped_per_provider() {
    let assistant_provider_opts = Some(v2t::ProviderOptions::from([
        (
            "google".into(),
            HashMap::from([("thoughtSignature".into(), json!("sig-google"))]),
        ),
        (
            "google-vertex".into(),
            HashMap::from([("thoughtSignature".into(), json!("sig-vertex"))]),
        ),
    ]));
    let prompt = vec![
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "hello".to_string(),
                provider_options: None,
            }],
            provider_options: None,
        },
        v2t::PromptMessage::Assistant {
            content: vec![v2t::AssistantPart::Text {
                text: "world".to_string(),
                provider_options: assistant_provider_opts,
            }],
            provider_options: None,
        },
    ];

    let google_prompt = convert_google_prompt(&prompt, false).expect("google prompt");
    let vertex_prompt = convert_vertex_prompt(&prompt, false).expect("vertex prompt");

    assert_eq!(
        first_model_text_signature(&google_prompt).as_deref(),
        Some("sig-google")
    );
    assert_eq!(
        first_model_text_signature(&vertex_prompt).as_deref(),
        Some("sig-vertex")
    );
}

fn stream_input(
    payload: serde_json::Value,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>> {
    let data = format!("data: {}\n\n", payload);
    Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(data))]))
}

fn stream_inputs(
    payloads: Vec<serde_json::Value>,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>> {
    let items = payloads
        .into_iter()
        .map(|payload| Ok(Bytes::from(format!("data: {payload}\n\n"))));
    Box::pin(futures_util::stream::iter(items))
}

fn metadata_scope(meta: &Option<v2t::ProviderMetadata>) -> Option<String> {
    meta.as_ref().and_then(|m| m.keys().next().cloned())
}

async fn collect_parts<S, E>(mut stream: S, label: &str) -> Vec<v2t::StreamPart>
where
    S: Stream<Item = Result<v2t::StreamPart, E>> + Unpin,
    E: std::fmt::Debug,
{
    let mut parts = Vec::new();
    while let Some(part) = stream.next().await {
        parts.push(part.unwrap_or_else(|err| panic!("{label} stream part: {err:?}")));
    }
    parts
}

#[tokio::test]
async fn shared_stream_core_preserves_provider_scope_namespace() {
    let payload = json!({
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "hello",
                    "thoughtSignature": "sig-123"
                }]
            },
            "groundingMetadata": {
                "groundingChunks": [{
                    "web": {"uri": "https://example.com", "title": "Example"}
                }]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 2,
            "totalTokenCount": 3
        }
    });

    let mut google_stream =
        build_google_stream_part_stream(stream_input(payload.clone()), vec![], false, "google");
    let mut vertex_stream =
        build_google_stream_part_stream(stream_input(payload), vec![], false, "google-vertex");

    let google_parts = collect_parts(&mut google_stream, "google").await;
    let vertex_parts = collect_parts(&mut vertex_stream, "google-vertex").await;

    let google_text_scope = google_parts.iter().find_map(|part| match part {
        v2t::StreamPart::TextStart {
            provider_metadata, ..
        } => metadata_scope(provider_metadata),
        _ => None,
    });
    let vertex_text_scope = vertex_parts.iter().find_map(|part| match part {
        v2t::StreamPart::TextStart {
            provider_metadata, ..
        } => metadata_scope(provider_metadata),
        _ => None,
    });

    assert_eq!(google_text_scope.as_deref(), Some("google"));
    assert_eq!(vertex_text_scope.as_deref(), Some("google-vertex"));

    let google_finish = google_parts.iter().find_map(|part| match part {
        v2t::StreamPart::Finish {
            finish_reason,
            provider_metadata,
            ..
        } => Some((finish_reason, metadata_scope(provider_metadata))),
        _ => None,
    });
    let vertex_finish = vertex_parts.iter().find_map(|part| match part {
        v2t::StreamPart::Finish {
            finish_reason,
            provider_metadata,
            ..
        } => Some((finish_reason, metadata_scope(provider_metadata))),
        _ => None,
    });

    let (google_finish_reason, google_scope) = google_finish.expect("google finish");
    let (vertex_finish_reason, vertex_scope) = vertex_finish.expect("vertex finish");
    assert!(matches!(google_finish_reason, v2t::FinishReason::Stop));
    assert!(matches!(vertex_finish_reason, v2t::FinishReason::Stop));
    assert_eq!(google_scope.as_deref(), Some("google"));
    assert_eq!(vertex_scope.as_deref(), Some("google-vertex"));
}

#[tokio::test]
async fn shared_stream_core_normalizes_reasoning_tool_raw_and_finish() {
    let payload = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "text": "hidden reasoning",
                        "thought": true,
                        "thoughtSignature": "sig-456"
                    },
                    {
                        "text": "visible answer"
                    },
                    {
                        "functionCall": {
                            "name": "weather",
                            "args": {"city": "SF"}
                        }
                    }
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 3,
            "totalTokenCount": 5,
            "thoughtsTokenCount": 1
        }
    });

    let mut google_stream =
        build_google_stream_part_stream(stream_inputs(vec![payload]), vec![], true, "google");
    let parts = collect_parts(&mut google_stream, "google").await;

    let [v2t::StreamPart::StreamStart { warnings }, v2t::StreamPart::Raw { .. }, v2t::StreamPart::ReasoningStart {
        provider_metadata: reasoning_metadata,
        ..
    }, v2t::StreamPart::ReasoningDelta {
        delta: reasoning_delta,
        ..
    }, v2t::StreamPart::ReasoningEnd { .. }, v2t::StreamPart::TextStart { .. }, v2t::StreamPart::TextDelta {
        delta: visible_answer,
        ..
    }, v2t::StreamPart::ToolInputStart {
        tool_name,
        provider_executed: tool_started_by_provider,
        ..
    }, v2t::StreamPart::ToolInputDelta {
        delta: tool_delta,
        provider_executed: tool_delta_from_provider,
        ..
    }, v2t::StreamPart::ToolInputEnd {
        provider_executed: tool_ended_by_provider,
        ..
    }, v2t::StreamPart::ToolCall(call), v2t::StreamPart::TextEnd { .. }, v2t::StreamPart::Finish {
        usage,
        finish_reason,
        provider_metadata: finish_metadata,
    }] = parts.as_slice()
    else {
        panic!("unexpected stream sequence: {parts:?}");
    };

    assert!(warnings.is_empty());
    assert_eq!(
        metadata_scope(reasoning_metadata).as_deref(),
        Some("google")
    );
    assert_eq!(reasoning_delta, "hidden reasoning");
    assert_eq!(visible_answer, "visible answer");
    assert_eq!(tool_name, "weather");
    assert!(!tool_started_by_provider);
    assert_eq!(tool_delta, "{\"city\":\"SF\"}");
    assert!(!tool_delta_from_provider);
    assert!(!tool_ended_by_provider);
    assert_eq!(call.tool_name, "weather");
    assert_eq!(call.input, "{\"city\":\"SF\"}");
    assert!(!call.provider_executed);
    assert_eq!(usage.input_tokens, Some(2));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.total_tokens, Some(5));
    assert_eq!(usage.reasoning_tokens, Some(1));
    assert!(matches!(finish_reason, v2t::FinishReason::ToolCalls));
    assert_eq!(metadata_scope(finish_metadata).as_deref(), Some("google"));
}

#[test]
fn shared_generate_response_parser_preserves_google_gen_ai_fields() {
    let parsed = parse_google_gen_ai_generate_response(&json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "executableCode": {
                            "language": "PYTHON",
                            "code": "print('hi')"
                        }
                    },
                    {
                        "codeExecutionResult": {
                            "outcome": "OUTCOME_OK",
                            "output": "hi"
                        }
                    },
                    {
                        "text": "internal thought",
                        "thought": true,
                        "thoughtSignature": "sig-reasoning"
                    },
                    {
                        "text": "visible answer",
                        "thoughtSignature": "sig-text"
                    },
                    {
                        "functionCall": {
                            "name": "weather",
                            "args": {"city": "SF"}
                        },
                        "thoughtSignature": "sig-tool"
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "abc123"
                        }
                    }
                ]
            },
            "groundingMetadata": {
                "groundingChunks": [{
                    "web": {"uri": "https://example.com", "title": "Example"}
                }]
            },
            "urlContextMetadata": {"status": "ok"},
            "safetyRatings": [{"category": "SAFE"}],
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 5,
            "totalTokenCount": 8,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 1
        }
    }));

    let [v2t::Content::ToolCall(code_call), v2t::Content::ToolResult {
        tool_call_id: code_result_id,
        tool_name: code_result_name,
        result: code_result,
        ..
    }, v2t::Content::Reasoning {
        text: reasoning_text,
        provider_metadata: reasoning_metadata,
    }, v2t::Content::Text {
        text: visible_text,
        provider_metadata: text_metadata,
    }, v2t::Content::ToolCall(function_call), v2t::Content::File { media_type, data }, v2t::Content::SourceUrl { url, title, .. }] =
        parsed.content.as_slice()
    else {
        panic!("unexpected google gen-ai content: {:?}", parsed.content);
    };

    assert_eq!(code_call.tool_name, "code_execution");
    assert!(code_call.provider_executed);
    assert_eq!(code_result_id, &code_call.tool_call_id);
    assert_eq!(code_result_name, "code_execution");
    assert_eq!(code_result["output"], json!("hi"));
    assert_eq!(reasoning_text, "internal thought");
    assert_eq!(
        metadata_scope(reasoning_metadata).as_deref(),
        Some("google")
    );
    assert_eq!(visible_text, "visible answer");
    assert_eq!(metadata_scope(text_metadata).as_deref(), Some("google"));
    assert_eq!(function_call.tool_name, "weather");
    assert_eq!(function_call.input, "{\"city\":\"SF\"}");
    assert_eq!(
        function_call
            .provider_options
            .as_ref()
            .and_then(|options| options.keys().next())
            .map(String::as_str),
        Some("google")
    );
    assert_eq!(media_type, "image/png");
    assert_eq!(data, "abc123");
    assert_eq!(url, "https://example.com");
    assert_eq!(title.as_deref(), Some("Example"));
    assert!(matches!(parsed.finish_reason, v2t::FinishReason::ToolCalls));
    assert_eq!(parsed.usage.input_tokens, Some(3));
    assert_eq!(parsed.usage.output_tokens, Some(5));
    assert_eq!(parsed.usage.total_tokens, Some(8));
    assert_eq!(parsed.usage.reasoning_tokens, Some(2));
    assert_eq!(parsed.usage.cached_input_tokens, Some(1));
    let provider_metadata = parsed.provider_metadata.expect("google provider metadata");
    assert!(provider_metadata.contains_key("google"));
    assert_eq!(
        provider_metadata["google"]["usageMetadata"]["thoughtsTokenCount"],
        json!(2)
    );
}

#[test]
fn shared_generate_response_parser_preserves_google_vertex_fields() {
    let parsed = parse_google_vertex_generate_response(&json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "text": "vertex thought",
                        "thought": true,
                        "thoughtSignature": "sig-vertex"
                    },
                    {
                        "functionCall": {
                            "name": "weather",
                            "args": null
                        },
                        "thoughtSignature": "sig-tool"
                    },
                    {
                        "functionResponse": {
                            "name": "weather",
                            "response": {
                                "content": {"tempC": 20}
                            }
                        }
                    },
                    {
                        "inlineData": {
                            "mimeType": "application/pdf",
                            "data": "xyz"
                        }
                    }
                ]
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 6,
            "totalTokenCount": 10,
            "thoughtsTokenCount": 99,
            "cachedContentTokenCount": 77
        }
    }));

    let [v2t::Content::Reasoning {
        text: reasoning_text,
        provider_metadata: reasoning_metadata,
    }, v2t::Content::ToolCall(tool_call), v2t::Content::ToolResult {
        tool_call_id: tool_result_id,
        tool_name: tool_result_name,
        result: tool_result,
        ..
    }, v2t::Content::File { media_type, data }] = parsed.content.as_slice()
    else {
        panic!("unexpected google vertex content: {:?}", parsed.content);
    };

    assert_eq!(reasoning_text, "vertex thought");
    assert_eq!(
        metadata_scope(reasoning_metadata).as_deref(),
        Some("google-vertex")
    );
    assert_eq!(tool_call.tool_name, "weather");
    assert_eq!(tool_call.input, "{}");
    assert_eq!(tool_result_id, &tool_call.tool_call_id);
    assert_eq!(tool_result_name, "weather");
    assert_eq!(tool_result["tempC"], json!(20));
    assert_eq!(media_type, "application/pdf");
    assert_eq!(data, "xyz");
    assert!(matches!(parsed.finish_reason, v2t::FinishReason::Stop));
    assert_eq!(parsed.usage.input_tokens, Some(4));
    assert_eq!(parsed.usage.output_tokens, Some(6));
    assert_eq!(parsed.usage.total_tokens, Some(10));
    assert_eq!(parsed.usage.reasoning_tokens, None);
    assert_eq!(parsed.usage.cached_input_tokens, None);
    assert!(parsed.provider_metadata.is_none());
}
