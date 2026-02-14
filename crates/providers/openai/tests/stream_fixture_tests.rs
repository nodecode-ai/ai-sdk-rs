use crate::ai_sdk_core::error::TransportError;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::LanguageModel;
use crate::ai_sdk_providers_openai::config::OpenAIConfig;
use crate::ai_sdk_providers_openai::responses::language_model::OpenAIResponsesLanguageModel;
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use futures_util::{stream, TryStreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct FixtureTransport {
    chunks: Arc<Vec<Bytes>>,
}

struct FixtureStreamResponse {
    chunks: Arc<Vec<Bytes>>,
}

impl FixtureTransport {
    fn from_fixture(name: &str) -> Self {
        Self {
            chunks: Arc::new(read_fixture_chunks(name)),
        }
    }
}

fn read_fixture_chunks(name: &str) -> Vec<Bytes> {
    let fixture_name = format!("{name}.chunks.txt");
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(&fixture_name),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("crates")
            .join("providers")
            .join("openai")
            .join("tests")
            .join("fixtures")
            .join(&fixture_name),
    ];
    let path = candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .unwrap_or_else(|| candidates[0].clone());
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("missing fixture {path:?}"));
    let mut chunks = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        chunks.push(Bytes::from(format!("data: {trimmed}\n\n")));
    }
    chunks.push(Bytes::from_static(b"data: [DONE]\n\n"));
    chunks
}

fn openai_error_fixture_message() -> String {
    serde_json::from_str::<Value>(include_str!("fixtures/openai-error.1.json"))
        .expect("openai error fixture json")
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .expect("openai error fixture message")
        .to_owned()
}

#[async_trait]
impl HttpTransport for FixtureTransport {
    type StreamResponse = FixtureStreamResponse;

    fn into_stream(
        resp: Self::StreamResponse,
    ) -> (
        Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>,
        Vec<(String, String)>,
    ) {
        let chunks = (*resp.chunks).clone();
        (Box::pin(stream::iter(chunks.into_iter().map(Ok))), vec![])
    }

    async fn post_json_stream(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        Ok(FixtureStreamResponse {
            chunks: self.chunks.clone(),
        })
    }

    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        Err(TransportError::Other("post_json unused".into()))
    }
}

fn test_config() -> OpenAIConfig {
    OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: "openai".into(),
        base_url: "https://api.openai.com/v1".into(),
        endpoint_path: "/responses".into(),
        headers: vec![],
        query_params: vec![],
        supported_urls: HashMap::new(),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options: None,
        request_defaults: None,
    }
}

fn provider_tool(id: &str, name: &str, args: Value) -> v2t::Tool {
    v2t::Tool::Provider(v2t::ProviderTool {
        r#type: v2t::ProviderToolType::Provider,
        id: id.into(),
        name: name.into(),
        args,
    })
}

async fn collect_parts(
    fixture: &str,
    model_id: &str,
    tools: Vec<v2t::Tool>,
    provider_options: Option<v2t::ProviderOptions>,
) -> Vec<v2t::StreamPart> {
    let transport = FixtureTransport::from_fixture(fixture);
    let model = OpenAIResponsesLanguageModel::new(
        model_id,
        test_config(),
        transport,
        TransportConfig::default(),
    );
    let opts = v2t::CallOptions {
        prompt: vec![v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Hello".into(),
                provider_options: None,
            }],
            provider_options: None,
        }],
        tools,
        provider_options: provider_options.unwrap_or_default(),
        ..Default::default()
    };
    let resp = model.do_stream(opts).await.expect("stream");
    resp.stream
        .try_collect()
        .await
        .expect("collect stream parts")
}

fn has_finish(parts: &[v2t::StreamPart]) -> bool {
    parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::Finish { .. }))
}

fn has_error(parts: &[v2t::StreamPart]) -> bool {
    parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::Error { .. }))
}

fn assert_ok_stream(parts: &[v2t::StreamPart]) {
    assert!(!has_error(parts));
    assert!(has_finish(parts));
}

fn finish_usage(parts: &[v2t::StreamPart]) -> &v2t::Usage {
    parts
        .iter()
        .find_map(|part| match part {
            v2t::StreamPart::Finish { usage, .. } => Some(usage),
            _ => None,
        })
        .expect("finish part")
}

fn tool_calls<'a>(parts: &'a [v2t::StreamPart], name: &str) -> Vec<&'a v2t::ToolCallPart> {
    parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::ToolCall(call) if call.tool_name == name => Some(call),
            _ => None,
        })
        .collect()
}

fn tool_results<'a>(parts: &'a [v2t::StreamPart], name: &str) -> Vec<(&'a Value, bool)> {
    parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::ToolResult {
                tool_name,
                result,
                preliminary,
                ..
            } if tool_name == name => Some((result, *preliminary)),
            _ => None,
        })
        .collect()
}

fn tool_input_starts(parts: &[v2t::StreamPart], name: &str) -> Vec<bool> {
    parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::ToolInputStart {
                tool_name,
                provider_executed,
                ..
            } if tool_name == name => Some(*provider_executed),
            _ => None,
        })
        .collect()
}

fn mcp_tool(label: &str, url: &str, description: &str, require_approval: bool) -> v2t::Tool {
    let mut args = serde_json::Map::new();
    args.insert("serverLabel".into(), json!(label));
    args.insert("serverUrl".into(), json!(url));
    args.insert("serverDescription".into(), json!(description));
    if require_approval {
        args.insert("requireApproval".into(), json!("always"));
    }
    provider_tool("openai.mcp", "MCP", Value::Object(args))
}

#[tokio::test]
async fn stream_web_search_fixture() {
    let tools = vec![provider_tool("openai.web_search", "webSearch", json!({}))];
    let parts = collect_parts("openai-web-search-tool.1", "gpt-5-nano", tools, None).await;

    assert_ok_stream(&parts);

    let inputs = tool_input_starts(&parts, "webSearch");
    assert!(!inputs.is_empty());
    assert!(inputs.iter().all(|exec| *exec));

    let calls = tool_calls(&parts, "webSearch");
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| call.provider_executed));

    let results = tool_results(&parts, "webSearch");
    assert!(!results.is_empty());
    assert!(results.iter().any(|(result, _)| {
        result.get("action").is_some()
            && result
                .get("sources")
                .and_then(|v| v.as_array())
                .map(|arr| !arr.is_empty())
                .unwrap_or(false)
    }));
}

#[tokio::test]
async fn stream_file_search_without_results_fixture() {
    let tools = vec![provider_tool(
        "openai.file_search",
        "fileSearch",
        json!({ "vectorStoreIds": ["vs_68caad8bd5d88191ab766cf043d89a18"] }),
    )];
    let parts = collect_parts("openai-file-search-tool.1", "gpt-5-nano", tools, None).await;

    assert_ok_stream(&parts);

    let calls = tool_calls(&parts, "fileSearch");
    assert!(!calls.is_empty());

    let results = tool_results(&parts, "fileSearch");
    assert!(!results.is_empty());
    let (result, _) = results[0];
    let results_val = result.get("results").expect("results field");
    assert!(results_val.is_null());
}

#[tokio::test]
async fn stream_file_search_with_results_fixture() {
    let tools = vec![provider_tool(
        "openai.file_search",
        "fileSearch",
        json!({ "vectorStoreIds": ["vs_68caad8bd5d88191ab766cf043d89a18"] }),
    )];
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai".into(),
        HashMap::from([("include".into(), json!(["file_search_call.results"]))]),
    );
    let parts = collect_parts(
        "openai-file-search-tool.2",
        "gpt-5-nano",
        tools,
        Some(provider_options),
    )
    .await;

    assert_ok_stream(&parts);

    let results = tool_results(&parts, "fileSearch");
    assert!(!results.is_empty());
    let (result, _) = results[0];
    let results_arr = result
        .get("results")
        .and_then(|v| v.as_array())
        .expect("results array");
    assert!(!results_arr.is_empty());
}

#[tokio::test]
async fn stream_code_interpreter_fixture() {
    let tools = vec![provider_tool(
        "openai.code_interpreter",
        "codeExecution",
        json!({}),
    )];
    let parts = collect_parts("openai-code-interpreter-tool.1", "gpt-5-nano", tools, None).await;

    assert_ok_stream(&parts);

    let inputs = tool_input_starts(&parts, "codeExecution");
    assert!(!inputs.is_empty());

    let calls = tool_calls(&parts, "codeExecution");
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| call.provider_executed));

    let sources: Vec<_> = parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::SourceUrl {
                provider_metadata, ..
            } => provider_metadata.as_ref(),
            _ => None,
        })
        .collect();
    assert!(!sources.is_empty());
    let has_file = sources.iter().any(|meta| {
        meta.get("openai")
            .and_then(|inner| {
                Some(inner.get("fileId").is_some() && inner.get("containerId").is_some())
            })
            .unwrap_or(false)
    });
    assert!(has_file);
}

#[tokio::test]
async fn stream_image_generation_fixture() {
    let tools = vec![provider_tool(
        "openai.image_generation",
        "generateImage",
        json!({}),
    )];
    let parts = collect_parts("openai-image-generation-tool.1", "gpt-5-nano", tools, None).await;

    assert_ok_stream(&parts);

    let calls = tool_calls(&parts, "generateImage");
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| call.provider_executed));

    let results = tool_results(&parts, "generateImage");
    assert!(!results.is_empty());
    assert!(results.iter().any(|(_, preliminary)| *preliminary));
}

#[tokio::test]
async fn stream_local_shell_fixture() {
    let tools = vec![provider_tool("openai.local_shell", "shell", json!({}))];
    let parts = collect_parts("openai-local-shell-tool.1", "gpt-5-codex", tools, None).await;

    assert_ok_stream(&parts);

    let calls = tool_calls(&parts, "shell");
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| !call.provider_executed));
    assert!(calls.iter().all(|call| call.provider_metadata.is_some()));
}

#[tokio::test]
async fn stream_usage_maps_nested_details() {
    let tools = vec![provider_tool("openai.local_shell", "shell", json!({}))];
    let parts = collect_parts("openai-local-shell-tool.1", "gpt-5-codex", tools, None).await;

    assert_ok_stream(&parts);
    let usage = finish_usage(&parts);
    assert_eq!(usage.input_tokens, Some(407));
    assert_eq!(usage.cached_input_tokens, Some(0));
    assert_eq!(usage.output_tokens, Some(151));
    assert_eq!(usage.reasoning_tokens, Some(128));
    assert_eq!(usage.total_tokens, Some(558));
}

#[tokio::test]
async fn stream_shell_fixture() {
    let tools = vec![provider_tool("openai.shell", "shell", json!({}))];
    let parts = collect_parts("openai-shell-tool.1", "gpt-5.1", tools, None).await;

    assert_ok_stream(&parts);

    let calls = tool_calls(&parts, "shell");
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| !call.provider_executed));
    assert!(calls.iter().all(|call| call.provider_metadata.is_some()));
}

#[tokio::test]
async fn stream_mcp_fixture() {
    let tools = vec![mcp_tool(
        "dmcp",
        "https://mcp.exa.ai/mcp",
        "A web-search API for AI agents",
        false,
    )];
    let parts = collect_parts("openai-mcp-tool.1", "gpt-5-mini", tools, None).await;

    assert_ok_stream(&parts);

    let mcp_calls: Vec<_> = parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::ToolCall(call) if call.tool_name.starts_with("mcp.") => Some(call),
            _ => None,
        })
        .collect();
    assert!(!mcp_calls.is_empty());

    let mcp_results: Vec<_> = parts
        .iter()
        .filter_map(|part| match part {
            v2t::StreamPart::ToolResult {
                tool_name,
                provider_metadata,
                ..
            } if tool_name.starts_with("mcp.") => provider_metadata.as_ref(),
            _ => None,
        })
        .collect();
    assert!(!mcp_results.is_empty());
}

#[tokio::test]
async fn stream_mcp_approval_turn1_fixture() {
    let tools = vec![mcp_tool(
        "zip1",
        "https://zip1.io/mcp",
        "Link shortener",
        true,
    )];
    let parts = collect_parts("openai-mcp-tool-approval.1", "gpt-5-mini", tools, None).await;

    assert_ok_stream(&parts);
    assert!(parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::ToolApprovalRequest { .. })));
}

#[tokio::test]
async fn stream_mcp_approval_turn2_fixture() {
    let tools = vec![mcp_tool(
        "zip1",
        "https://zip1.io/mcp",
        "Link shortener",
        true,
    )];
    let parts = collect_parts("openai-mcp-tool-approval.2", "gpt-5-mini", tools, None).await;

    assert_ok_stream(&parts);
    assert!(parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::TextDelta { .. })));
}

#[tokio::test]
async fn stream_mcp_approval_turn3_fixture() {
    let tools = vec![mcp_tool(
        "zip1",
        "https://zip1.io/mcp",
        "Link shortener",
        true,
    )];
    let parts = collect_parts("openai-mcp-tool-approval.3", "gpt-5-mini", tools, None).await;

    assert_ok_stream(&parts);
    assert!(parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::ToolApprovalRequest { .. })));
}

#[tokio::test]
async fn stream_mcp_approval_turn4_fixture() {
    let tools = vec![mcp_tool(
        "zip1",
        "https://zip1.io/mcp",
        "Link shortener",
        true,
    )];
    let parts = collect_parts("openai-mcp-tool-approval.4", "gpt-5-mini", tools, None).await;

    assert_ok_stream(&parts);
    let has_result = parts
        .iter()
        .any(|part| matches!(part, v2t::StreamPart::ToolResult { .. }));
    assert!(has_result);
}

#[tokio::test]
async fn stream_error_fixture() {
    let parts = collect_parts("openai-error.1", "gpt-4o-mini", vec![], None).await;
    let error_count = parts
        .iter()
        .filter(|part| matches!(part, v2t::StreamPart::Error { .. }))
        .count();
    assert_eq!(
        error_count, 1,
        "openai-error fixture should emit exactly one error part"
    );
    let finish_count = parts
        .iter()
        .filter(|part| matches!(part, v2t::StreamPart::Finish { .. }))
        .count();
    assert_eq!(
        finish_count, 1,
        "openai-error fixture should emit exactly one finish part"
    );
    assert!(
        has_error(&parts),
        "openai-error fixture must emit an error part before stream termination"
    );
    assert!(
        has_finish(&parts),
        "openai-error fixture must still emit a finish part after terminal failure"
    );

    let error_idx = parts
        .iter()
        .position(|part| matches!(part, v2t::StreamPart::Error { .. }))
        .expect("error part index");
    let (finish_idx, finish_reason, provider_metadata) = parts
        .iter()
        .enumerate()
        .find_map(|(idx, part)| match part {
            v2t::StreamPart::Finish {
                finish_reason,
                provider_metadata,
                ..
            } => Some((idx, finish_reason.clone(), provider_metadata.clone())),
            _ => None,
        })
        .expect("finish part details");

    assert!(
        finish_idx > error_idx,
        "finish part must be emitted after the error part for openai-error fixture"
    );
    assert!(
        matches!(finish_reason, v2t::FinishReason::Other),
        "TS baseline treats response.failed as non-finished chunk; finish reason should remain Other"
    );

    let response_metadata_id = parts
        .iter()
        .find_map(|part| match part {
            v2t::StreamPart::ResponseMetadata { meta } => meta.id.as_deref(),
            _ => None,
        })
        .expect("response metadata id");
    let openai_meta = provider_metadata
        .as_ref()
        .and_then(|meta| meta.get("openai"))
        .expect("finish part should include openai provider metadata");
    let finish_response_id = openai_meta
        .get("responseId")
        .and_then(Value::as_str)
        .expect("finish metadata responseId");
    assert_eq!(
        finish_response_id, response_metadata_id,
        "finish responseId should match stream response metadata id"
    );
    assert!(
        openai_meta.get("serviceTier").is_none(),
        "TS baseline does not include serviceTier when terminal chunk is response.failed"
    );

    let error_message = parts
        .iter()
        .find_map(|part| match part {
            v2t::StreamPart::Error { error } => error
                .get("error")
                .and_then(|value| value.get("message"))
                .and_then(Value::as_str),
            _ => None,
        })
        .expect("stream error message");
    assert_eq!(
        error_message,
        openai_error_fixture_message(),
        "stream error message should match openai-error fixture payload"
    );
}

#[tokio::test]
async fn stream_apply_patch_fixture() {
    let tools = vec![provider_tool(
        "openai.apply_patch",
        "apply_patch",
        json!({}),
    )];
    let parts = collect_parts(
        "openai-apply-patch-tool.1",
        "gpt-5.1-2025-11-13",
        tools,
        None,
    )
    .await;

    assert_ok_stream(&parts);

    let inputs = tool_input_starts(&parts, "apply_patch");
    assert!(!inputs.is_empty());
    assert!(inputs.iter().all(|exec| !*exec));

    let calls = tool_calls(&parts, "apply_patch");
    assert!(!calls.is_empty());
    assert!(calls.iter().all(|call| !call.provider_executed));
    for call in calls {
        let input: Value = serde_json::from_str(&call.input).expect("tool input json");
        assert!(input.get("callId").is_some());
        assert!(input.get("operation").is_some());
    }
}

#[tokio::test]
async fn stream_apply_patch_delete_fixture() {
    let tools = vec![provider_tool(
        "openai.apply_patch",
        "apply_patch",
        json!({}),
    )];
    let parts = collect_parts(
        "openai-apply-patch-tool-delete.1",
        "gpt-5.1-2025-11-13",
        tools,
        None,
    )
    .await;

    assert_ok_stream(&parts);

    let inputs = tool_input_starts(&parts, "apply_patch");
    assert!(!inputs.is_empty());
    assert!(inputs.iter().all(|exec| !*exec));
}

#[tokio::test]
async fn stream_fixture_perf_smoke() {
    let tools = vec![mcp_tool(
        "dmcp",
        "https://mcp.exa.ai/mcp",
        "A web-search API for AI agents",
        false,
    )];
    let start = Instant::now();
    let parts = collect_parts("openai-mcp-tool.1", "gpt-5-mini", tools, None).await;
    assert!(!parts.is_empty());
    assert!(start.elapsed() < Duration::from_secs(5));
}
