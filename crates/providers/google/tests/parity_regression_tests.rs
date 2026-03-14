use std::collections::HashMap;
use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::json;

use crate::core::error::{SdkError, TransportError};
use crate::types::v2 as v2t;
use crate::providers::google::shared::options::{
    parse_google_provider_options_for_scopes, GoogleProviderOptions,
};
use crate::providers::google::shared::prepare_tools::prepare_tools as prepare_google_family_tools;
use crate::providers::google::shared::prompt::{
    convert_to_google_prompt_with_scopes, GoogleContent, GoogleContentPart, GooglePrompt,
};
use crate::providers::google::shared::stream_core::build_google_stream_part_stream;

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

    let mut google_parts = Vec::new();
    let mut vertex_parts = Vec::new();

    while let Some(part) = google_stream.next().await {
        google_parts.push(part.expect("google stream part"));
    }
    while let Some(part) = vertex_stream.next().await {
        vertex_parts.push(part.expect("vertex stream part"));
    }

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
    let mut parts = Vec::new();
    while let Some(part) = google_stream.next().await {
        parts.push(part.expect("google stream part"));
    }

    assert!(matches!(&parts[0], v2t::StreamPart::StreamStart { warnings } if warnings.is_empty()));
    assert!(matches!(&parts[1], v2t::StreamPart::Raw { .. }));
    assert!(matches!(
        &parts[2],
        v2t::StreamPart::ReasoningStart { provider_metadata, .. }
            if metadata_scope(provider_metadata).as_deref() == Some("google")
    ));
    assert!(matches!(
        &parts[3],
        v2t::StreamPart::ReasoningDelta { delta, .. } if delta == "hidden reasoning"
    ));
    assert!(matches!(&parts[4], v2t::StreamPart::ReasoningEnd { .. }));
    assert!(matches!(&parts[5], v2t::StreamPart::TextStart { .. }));
    assert!(matches!(
        &parts[6],
        v2t::StreamPart::TextDelta { delta, .. } if delta == "visible answer"
    ));
    assert!(matches!(
        &parts[7],
        v2t::StreamPart::ToolInputStart { tool_name, provider_executed, .. }
            if tool_name == "weather" && !provider_executed
    ));
    assert!(matches!(
        &parts[8],
        v2t::StreamPart::ToolInputDelta { delta, provider_executed, .. }
            if delta == "{\"city\":\"SF\"}" && !provider_executed
    ));
    assert!(matches!(
        &parts[9],
        v2t::StreamPart::ToolInputEnd { provider_executed, .. } if !provider_executed
    ));
    assert!(matches!(
        &parts[10],
        v2t::StreamPart::ToolCall(call)
            if call.tool_name == "weather"
                && call.input == "{\"city\":\"SF\"}"
                && !call.provider_executed
    ));
    assert!(matches!(&parts[11], v2t::StreamPart::TextEnd { .. }));
    assert!(matches!(
        &parts[12],
        v2t::StreamPart::Finish {
            usage,
            finish_reason,
            provider_metadata,
        } if usage.input_tokens == Some(2)
            && usage.output_tokens == Some(3)
            && usage.total_tokens == Some(5)
            && usage.reasoning_tokens == Some(1)
            && matches!(finish_reason, v2t::FinishReason::ToolCalls)
            && metadata_scope(provider_metadata).as_deref() == Some("google")
    ));
}
