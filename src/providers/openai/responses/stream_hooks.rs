use std::collections::{HashMap, HashSet};

use crate::ai_sdk_core::{EventMapperConfig, EventMapperHooks, EventMapperState};
use crate::ai_sdk_types::v2 as v2t;
use serde_json::json;
use uuid::Uuid;

use super::language_model::{
    apply_openai_usage_details, escape_json_delta, map_finish_reason, openai_item_metadata,
    parse_openai_usage,
};
use super::provider_tools::{provider_tool_parts_from_data, ProviderToolParts, ToolNameMapping};

#[derive(Debug, Clone)]
struct OpenAIApplyPatchState {
    tool_call_id: String,
    operation_path: Option<String>,
    has_diff: bool,
    end_emitted: bool,
}

#[derive(Debug, Clone)]
struct OpenAICodeInterpreterState {
    tool_call_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningSummaryStatus {
    Active,
    CanConclude,
    Concluded,
}

#[derive(Debug, Clone, Default)]
struct OpenAIReasoningState {
    encrypted_content: Option<serde_json::Value>,
    summary_parts: HashMap<u32, ReasoningSummaryStatus>,
}

#[derive(Default)]
pub(super) struct OpenAIStreamExtras {
    finish_hint: Option<String>,
    response_id: Option<String>,
    service_tier: Option<String>,
    saw_response_failed: bool,
    store: bool,
    logprobs_enabled: bool,
    has_function_calls: bool,
    logprobs: Vec<serde_json::Value>,
    message_annotations: HashMap<String, Vec<serde_json::Value>>,
    active_reasoning: HashMap<String, OpenAIReasoningState>,
    open_tool_inputs: HashSet<String>,
    tool_item_ids: HashMap<String, String>,
    approval_request_id_map: HashMap<String, String>,
    apply_patch_calls: HashMap<usize, OpenAIApplyPatchState>,
    code_interpreter_calls: HashMap<usize, OpenAICodeInterpreterState>,
    emitted_tool_calls: HashSet<String>,
    tool_name_mapping: ToolNameMapping,
}

type StreamDataHandlerResult = Option<Option<Vec<v2t::StreamPart>>>;

fn handled_none() -> StreamDataHandlerResult {
    Some(None)
}

fn handled_part(part: v2t::StreamPart) -> StreamDataHandlerResult {
    Some(Some(vec![part]))
}

fn handle_usage_or_response_metadata_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    match key {
        "usage" => {
            if let Some(usage) = parse_openai_usage(value) {
                state.usage.input_tokens = Some(usage.input_tokens as u64);
                state.usage.output_tokens = Some(usage.output_tokens as u64);
                state.usage.total_tokens = Some(usage.total_tokens as u64);
                state.usage.cached_input_tokens = usage.cache_read_tokens.map(|v| v as u64);
            }
            apply_openai_usage_details(value, &mut state.usage);
            handled_none()
        }
        "openai.response_metadata" => {
            let id = value
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let model_id = value
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let timestamp_ms = value
                .get("created_at")
                .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)));
            if let Some(rid) = id.as_ref() {
                if state.extra.response_id.is_none() {
                    state.extra.response_id = Some(rid.clone());
                }
            }
            handled_part(v2t::StreamPart::ResponseMetadata {
                meta: v2t::ResponseMetadata {
                    id,
                    timestamp_ms,
                    model_id,
                },
            })
        }
        _ => None,
    }
}

fn make_openai_provider_metadata(values: Vec<(&str, serde_json::Value)>) -> v2t::ProviderMetadata {
    let mut inner = HashMap::new();
    for (key, value) in values {
        inner.insert(key.into(), value);
    }
    let mut outer = HashMap::new();
    outer.insert("openai".into(), inner);
    outer
}

fn build_source_url_part(
    url: String,
    title: Option<String>,
    provider_metadata: Option<v2t::ProviderMetadata>,
) -> v2t::StreamPart {
    v2t::StreamPart::SourceUrl {
        id: Uuid::new_v4().to_string(),
        url,
        title,
        provider_metadata,
    }
}

fn handle_message_added_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    let item_id = value.get("item_id").and_then(|v| v.as_str())?;
    state
        .extra
        .message_annotations
        .insert(item_id.to_string(), Vec::new());
    if state.text_open.as_deref() != Some(item_id) {
        return Some(Some(state.open_text(
            item_id.to_string(),
            Some(openai_item_metadata(item_id, [])),
        )));
    }
    handled_none()
}

fn text_delta_start_metadata(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    item_id: &str,
) -> Option<v2t::ProviderMetadata> {
    if state.text_open.as_deref() == Some(item_id) {
        return None;
    }
    state
        .extra
        .message_annotations
        .entry(item_id.to_string())
        .or_default();
    Some(openai_item_metadata(item_id, []))
}

fn collect_text_logprobs(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    value: &serde_json::Value,
) {
    if !state.extra.logprobs_enabled {
        return;
    }
    if let Some(logprobs) = value.get("logprobs").filter(|v| !v.is_null()) {
        state.extra.logprobs.push(logprobs.clone());
    }
}

fn handle_text_delta_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    let item_id = value.get("item_id").and_then(|v| v.as_str())?;
    let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
    if delta.is_empty() {
        return handled_none();
    }
    let start_metadata = text_delta_start_metadata(state, item_id);
    collect_text_logprobs(state, value);
    Some(Some(state.push_text_delta(
        Some(item_id.to_string()),
        item_id,
        delta.to_string(),
        start_metadata,
        None,
    )))
}

fn annotation_title(
    annotation_obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<String> {
    keys.iter().find_map(|key| {
        annotation_obj
            .get(*key)
            .and_then(|value| value.as_str())
            .map(|value| value.to_string())
    })
}

fn push_annotation_index(
    metadata: &mut Vec<(&'static str, serde_json::Value)>,
    annotation_obj: &serde_json::Map<String, serde_json::Value>,
) {
    if let Some(index) = annotation_obj.get("index").filter(|value| !value.is_null()) {
        metadata.push(("index", index.clone()));
    }
}

fn handle_url_citation_annotation(
    annotation_obj: &serde_json::Map<String, serde_json::Value>,
) -> StreamDataHandlerResult {
    let url = annotation_obj.get("url")?.as_str()?;
    handled_part(build_source_url_part(
        url.to_string(),
        annotation_title(annotation_obj, &["title"]),
        None,
    ))
}

fn handle_file_citation_annotation(
    annotation_obj: &serde_json::Map<String, serde_json::Value>,
) -> StreamDataHandlerResult {
    let file_id = annotation_obj.get("file_id")?.as_str()?;
    let mut metadata = vec![("fileId", json!(file_id))];
    push_annotation_index(&mut metadata, annotation_obj);
    handled_part(build_source_url_part(
        file_id.to_string(),
        annotation_title(annotation_obj, &["quote", "filename"])
            .or_else(|| Some(file_id.to_string())),
        Some(make_openai_provider_metadata(metadata)),
    ))
}

fn handle_container_file_citation_annotation(
    annotation_obj: &serde_json::Map<String, serde_json::Value>,
) -> StreamDataHandlerResult {
    let file_id = annotation_obj.get("file_id")?.as_str()?;
    let container_id = annotation_obj.get("container_id")?.as_str()?;
    let mut metadata = vec![
        ("fileId", json!(file_id)),
        ("containerId", json!(container_id)),
    ];
    push_annotation_index(&mut metadata, annotation_obj);
    handled_part(build_source_url_part(
        file_id.to_string(),
        annotation_title(annotation_obj, &["filename"]).or_else(|| Some(file_id.to_string())),
        Some(make_openai_provider_metadata(metadata)),
    ))
}

fn handle_file_path_annotation(
    annotation_obj: &serde_json::Map<String, serde_json::Value>,
) -> StreamDataHandlerResult {
    let file_id = annotation_obj.get("file_id")?.as_str()?;
    let mut metadata = vec![("fileId", json!(file_id))];
    push_annotation_index(&mut metadata, annotation_obj);
    handled_part(build_source_url_part(
        file_id.to_string(),
        Some(file_id.to_string()),
        Some(make_openai_provider_metadata(metadata)),
    ))
}

fn handle_text_annotation_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    let item_id = value.get("item_id").and_then(|v| v.as_str())?;
    let annotation = value.get("annotation")?.clone();
    state
        .extra
        .message_annotations
        .entry(item_id.to_string())
        .or_default()
        .push(annotation.clone());
    let annotation_obj = annotation.as_object()?;
    let annotation_type = annotation_obj.get("type")?.as_str()?;
    match annotation_type {
        "url_citation" => handle_url_citation_annotation(annotation_obj),
        "file_citation" => handle_file_citation_annotation(annotation_obj),
        "container_file_citation" => handle_container_file_citation_annotation(annotation_obj),
        "file_path" => handle_file_path_annotation(annotation_obj),
        _ => handled_none(),
    }
}

fn message_done_metadata(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    item_id: &str,
) -> v2t::ProviderMetadata {
    let annotations = state
        .extra
        .message_annotations
        .remove(item_id)
        .unwrap_or_default();
    if annotations.is_empty() {
        openai_item_metadata(item_id, [])
    } else {
        openai_item_metadata(
            item_id,
            [("annotations".into(), serde_json::Value::Array(annotations))],
        )
    }
}

fn handle_message_done_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    let item_id = value.get("item_id").and_then(|v| v.as_str())?;
    let metadata = message_done_metadata(state, item_id);
    if state.text_open.as_deref() == Some(item_id) {
        return Some(state.close_text(Some(metadata)).map(|part| vec![part]));
    }
    handled_part(state.text_end_part(item_id.to_string(), Some(metadata)))
}

fn handle_message_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    match key {
        "openai.message_added" => handle_message_added_event(state, value),
        "openai.text_delta" => handle_text_delta_event(state, value),
        "openai.text_annotation" => handle_text_annotation_event(state, value),
        "openai.message_done" => handle_message_done_event(state, value),
        "openai.error" => {
            state.extra.finish_hint = Some("error".into());
            handled_part(v2t::StreamPart::Error {
                error: value.clone(),
            })
        }
        _ => None,
    }
}

fn handle_reasoning_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    match key {
        "openai.reasoning_added" => {
            let item_id = value.get("item_id").and_then(|v| v.as_str())?;
            let encrypted_content = value
                .get("encrypted_content")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let mut reasoning_state = OpenAIReasoningState {
                encrypted_content: Some(encrypted_content.clone()),
                summary_parts: HashMap::new(),
            };
            reasoning_state
                .summary_parts
                .insert(0, ReasoningSummaryStatus::Active);
            state
                .extra
                .active_reasoning
                .insert(item_id.to_string(), reasoning_state);
            Some(Some(state.open_reasoning(
                format!("{item_id}:0"),
                Some(openai_item_metadata(
                    item_id,
                    [("reasoningEncryptedContent".into(), encrypted_content)],
                )),
            )))
        }
        "openai.reasoning_summary_added" => {
            let item_id = value.get("item_id").and_then(|v| v.as_str())?;
            let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
            if summary_index == 0 {
                return handled_none();
            }
            let (concluded_ids, encrypted_content) = {
                let reasoning_state = state.extra.active_reasoning.get_mut(item_id)?;
                let mut concluded_ids = Vec::new();
                for (idx, status) in reasoning_state.summary_parts.iter_mut() {
                    if matches!(status, ReasoningSummaryStatus::CanConclude) {
                        concluded_ids.push(*idx);
                        *status = ReasoningSummaryStatus::Concluded;
                    }
                }
                reasoning_state
                    .summary_parts
                    .insert(summary_index as u32, ReasoningSummaryStatus::Active);
                let encrypted_content = reasoning_state
                    .encrypted_content
                    .clone()
                    .unwrap_or(serde_json::Value::Null);
                (concluded_ids, encrypted_content)
            };
            let mut out = Vec::new();
            for idx in concluded_ids {
                let reasoning_id = format!("{item_id}:{idx}");
                if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                    if let Some(part) =
                        state.close_reasoning(Some(openai_item_metadata(item_id, [])))
                    {
                        out.push(part);
                    }
                } else {
                    out.push(
                        state.reasoning_end_part(
                            reasoning_id,
                            Some(openai_item_metadata(item_id, [])),
                        ),
                    );
                }
            }
            out.extend(state.open_reasoning(
                format!("{item_id}:{summary_index}"),
                Some(openai_item_metadata(
                    item_id,
                    [("reasoningEncryptedContent".into(), encrypted_content)],
                )),
            ));
            Some(Some(out))
        }
        "openai.reasoning_summary_delta" => {
            let item_id = value.get("item_id").and_then(|v| v.as_str())?;
            let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
            let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return handled_none();
            }
            handled_part(state.push_reasoning_delta(
                &format!("{item_id}:{summary_index}"),
                delta.to_string(),
                Some(openai_item_metadata(item_id, [])),
            ))
        }
        "openai.reasoning_summary_done" => {
            let item_id = value.get("item_id").and_then(|v| v.as_str())?;
            let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
            let should_close =
                if let Some(reasoning_state) = state.extra.active_reasoning.get_mut(item_id) {
                    if state.extra.store {
                        reasoning_state
                            .summary_parts
                            .insert(summary_index as u32, ReasoningSummaryStatus::Concluded);
                        Some(format!("{item_id}:{summary_index}"))
                    } else {
                        reasoning_state
                            .summary_parts
                            .insert(summary_index as u32, ReasoningSummaryStatus::CanConclude);
                        None
                    }
                } else {
                    None
                };
            if let Some(reasoning_id) = should_close {
                if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                    return Some(
                        state
                            .close_reasoning(Some(openai_item_metadata(item_id, [])))
                            .map(|part| vec![part]),
                    );
                }
                return handled_part(
                    state.reasoning_end_part(reasoning_id, Some(openai_item_metadata(item_id, []))),
                );
            }
            handled_none()
        }
        "openai.reasoning_done" => {
            let item_id = value.get("item_id").and_then(|v| v.as_str())?;
            let encrypted_content = value
                .get("encrypted_content")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            if let Some(reasoning_state) = state.extra.active_reasoning.remove(item_id) {
                let metadata = openai_item_metadata(
                    item_id,
                    [("reasoningEncryptedContent".into(), encrypted_content)],
                );
                let mut out = Vec::new();
                for (idx, status) in reasoning_state.summary_parts {
                    if matches!(
                        status,
                        ReasoningSummaryStatus::Active | ReasoningSummaryStatus::CanConclude
                    ) {
                        let reasoning_id = format!("{item_id}:{idx}");
                        if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                            if let Some(part) = state.close_reasoning(Some(metadata.clone())) {
                                out.push(part);
                            }
                        } else {
                            out.push(
                                state.reasoning_end_part(reasoning_id, Some(metadata.clone())),
                            );
                        }
                    }
                }
                if !out.is_empty() {
                    return Some(Some(out));
                }
            }
            handled_none()
        }
        _ => None,
    }
}

fn handle_search_and_image_tool_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    match key {
        "openai.web_search_call.added" => {
            let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
            let tool_name = state
                .extra
                .tool_name_mapping
                .web_search_tool_name
                .clone()
                .unwrap_or_else(|| {
                    state
                        .extra
                        .tool_name_mapping
                        .to_custom_tool_name("web_search")
                        .to_string()
                });
            state.has_tool_calls = true;
            state
                .extra
                .emitted_tool_calls
                .insert(tool_call_id.to_string());
            let tool_call_id = tool_call_id.to_string();
            let mut out = vec![state.start_tool_call(tool_call_id.clone(), tool_name, true, None)];
            state.tool_args.insert(tool_call_id.clone(), "{}".into());
            out.extend(state.finish_tool_call(tool_call_id, true, None, None, false, None));
            Some(Some(out))
        }
        "openai.file_search_call.added" => {
            let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
            let tool_name = state
                .extra
                .tool_name_mapping
                .to_custom_tool_name("file_search")
                .to_string();
            state.has_tool_calls = true;
            state
                .extra
                .emitted_tool_calls
                .insert(tool_call_id.to_string());
            handled_part(state.tool_call_part(
                tool_call_id.to_string(),
                tool_name,
                "{}".into(),
                true,
                None,
                false,
                None,
            ))
        }
        "openai.image_generation_call.added" => {
            let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
            let tool_name = state
                .extra
                .tool_name_mapping
                .to_custom_tool_name("image_generation")
                .to_string();
            state.has_tool_calls = true;
            state
                .extra
                .emitted_tool_calls
                .insert(tool_call_id.to_string());
            handled_part(state.tool_call_part(
                tool_call_id.to_string(),
                tool_name,
                "{}".into(),
                true,
                None,
                false,
                None,
            ))
        }
        "openai.image_generation_call.partial" => {
            let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
            let partial = value.get("partial_image_b64").and_then(|v| v.as_str())?;
            let tool_name = state
                .extra
                .tool_name_mapping
                .to_custom_tool_name("image_generation")
                .to_string();
            handled_part(v2t::StreamPart::ToolResult {
                tool_call_id: tool_call_id.to_string(),
                tool_name,
                result: json!({ "result": partial }),
                is_error: false,
                preliminary: true,
                provider_metadata: None,
            })
        }
        _ => None,
    }
}

fn handle_code_interpreter_tool_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    match key {
        "openai.code_interpreter_call.added" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
            let container_id = value
                .get("container_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            state.extra.code_interpreter_calls.insert(
                output_index,
                OpenAICodeInterpreterState {
                    tool_call_id: tool_call_id.to_string(),
                },
            );
            state.has_tool_calls = true;
            let tool_name = state
                .extra
                .tool_name_mapping
                .to_custom_tool_name("code_interpreter")
                .to_string();
            let container_id = container_id.unwrap_or_default();
            Some(Some(vec![
                state.start_tool_call(tool_call_id.to_string(), tool_name, true, None),
                state.push_tool_call_delta(
                    tool_call_id.to_string(),
                    format!(
                        "{{\"containerId\":\"{}\",\"code\":\"",
                        escape_json_delta(&container_id)
                    ),
                    true,
                    None,
                ),
            ]))
        }
        "openai.code_interpreter_call.code_delta" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            if delta.is_empty() {
                return handled_none();
            }
            if let Some(call_state) = state.extra.code_interpreter_calls.get(&output_index) {
                return handled_part(state.push_tool_call_delta(
                    call_state.tool_call_id.clone(),
                    escape_json_delta(delta),
                    true,
                    None,
                ));
            }
            handled_none()
        }
        "openai.code_interpreter_call.code_done" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            if let Some(call_state) = state.extra.code_interpreter_calls.remove(&output_index) {
                let mut out = vec![state.push_tool_call_delta(
                    call_state.tool_call_id.clone(),
                    "\"}".into(),
                    true,
                    None,
                )];
                out.extend(state.finish_tool_call(
                    call_state.tool_call_id.clone(),
                    true,
                    None,
                    None,
                    false,
                    None,
                ));
                state
                    .extra
                    .emitted_tool_calls
                    .insert(call_state.tool_call_id);
                return Some(Some(out));
            }
            handled_none()
        }
        _ => None,
    }
}

fn handle_computer_and_apply_patch_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    match key {
        "openai.computer_call.added" => {
            let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
            state
                .extra
                .open_tool_inputs
                .insert(tool_call_id.to_string());
            state.has_tool_calls = true;
            Some(Some(vec![state.start_tool_call(
                tool_call_id.to_string(),
                state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("computer_use")
                    .to_string(),
                true,
                None,
            )]))
        }
        "openai.apply_patch_call.added" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            let call_id = value.get("call_id").and_then(|v| v.as_str())?;
            let operation = value
                .get("operation")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let operation_type = operation.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if operation_type.is_empty() {
                return handled_none();
            }
            let operation_path = operation
                .get("path")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string());
            let tool_name = state
                .extra
                .tool_name_mapping
                .to_custom_tool_name("apply_patch")
                .to_string();
            let mut call_state = OpenAIApplyPatchState {
                tool_call_id: call_id.to_string(),
                operation_path,
                has_diff: false,
                end_emitted: false,
            };
            let mut out = vec![state.start_tool_call(call_id.to_string(), tool_name, false, None)];
            if operation_type == "delete_file" {
                let input = json!({
                    "callId": call_id,
                    "operation": operation,
                })
                .to_string();
                out.push(state.push_tool_call_delta(call_id.to_string(), input, false, None));
                out.push(state.tool_input_end_part(call_id.to_string(), false, None));
                call_state.has_diff = true;
                call_state.end_emitted = true;
            } else {
                let path = call_state.operation_path.as_deref().unwrap_or("");
                let delta = format!(
                    "{{\"callId\":\"{}\",\"operation\":{{\"type\":\"{}\",\"path\":\"{}\",\"diff\":\"",
                    escape_json_delta(call_id),
                    escape_json_delta(operation_type),
                    escape_json_delta(path)
                );
                out.push(state.push_tool_call_delta(call_id.to_string(), delta, false, None));
            }
            state.has_tool_calls = true;
            state
                .extra
                .apply_patch_calls
                .insert(output_index, call_state);
            Some(Some(out))
        }
        "openai.apply_patch_call.diff.delta" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
            let tool_call_id =
                if let Some(call_state) = state.extra.apply_patch_calls.get_mut(&output_index) {
                    if call_state.end_emitted {
                        return handled_none();
                    }
                    if !delta.is_empty() {
                        call_state.has_diff = true;
                        Some(call_state.tool_call_id.clone())
                    } else {
                        None
                    }
                } else {
                    None
                };
            if let Some(tool_call_id) = tool_call_id {
                return handled_part(state.push_tool_call_delta(
                    tool_call_id,
                    escape_json_delta(delta),
                    false,
                    None,
                ));
            }
            handled_none()
        }
        "openai.apply_patch_call.diff.done" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            if let Some((tool_call_id, should_emit_diff)) = state
                .extra
                .apply_patch_calls
                .get_mut(&output_index)
                .and_then(|call_state| {
                    if call_state.end_emitted {
                        return None;
                    }
                    let should_emit_diff = !call_state.has_diff;
                    if should_emit_diff {
                        call_state.has_diff = true;
                    }
                    call_state.end_emitted = true;
                    Some((call_state.tool_call_id.clone(), should_emit_diff))
                })
            {
                let mut out = Vec::new();
                if should_emit_diff {
                    let diff = value.get("diff").and_then(|v| v.as_str()).unwrap_or("");
                    out.push(state.push_tool_call_delta(
                        tool_call_id.clone(),
                        escape_json_delta(diff),
                        false,
                        None,
                    ));
                }
                out.push(state.push_tool_call_delta(
                    tool_call_id.clone(),
                    "\"}}".into(),
                    false,
                    None,
                ));
                out.push(state.tool_input_end_part(tool_call_id, false, None));
                return Some(Some(out));
            }
            handled_none()
        }
        "openai.apply_patch_call.done" => {
            let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
            if let Some(call_state) = state.extra.apply_patch_calls.remove(&output_index) {
                if call_state.end_emitted {
                    return handled_none();
                }
                let mut out = Vec::new();
                if !call_state.has_diff {
                    let diff = value
                        .get("operation")
                        .and_then(|v| v.get("diff"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    out.push(state.push_tool_call_delta(
                        call_state.tool_call_id.clone(),
                        escape_json_delta(diff),
                        false,
                        None,
                    ));
                }
                out.push(state.push_tool_call_delta(
                    call_state.tool_call_id.clone(),
                    "\"}}".into(),
                    false,
                    None,
                ));
                out.push(state.tool_input_end_part(call_state.tool_call_id, false, None));
                return Some(Some(out));
            }
            handled_none()
        }
        _ => None,
    }
}

fn handle_provider_tool_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    if key != "openai.provider_tool" {
        return None;
    }

    if let Some(mut parts) = provider_tool_parts_from_data(value, &state.extra.tool_name_mapping) {
        state.has_tool_calls = true;
        if let Some(out) = emit_provider_tool_approval_request(state, &parts) {
            return Some(Some(out));
        }

        remap_provider_tool_call_id(&mut parts, &state.extra.approval_request_id_map);
        let tool_call_id = parts.tool_call_id.clone();
        let (tool_call_metadata, tool_result_metadata) = provider_tool_stream_metadata(&parts);
        let mut out = Vec::new();
        maybe_close_provider_tool_input(state, &parts, &tool_call_id, &mut out);
        if !should_skip_provider_tool_call(state, &parts, &tool_call_id) {
            out.push(state.tool_call_part(
                tool_call_id.clone(),
                parts.tool_name.clone(),
                parts.input,
                parts.provider_executed,
                tool_call_metadata,
                parts.dynamic,
                None,
            ));
        }
        if let Some(result) = parts.result.take() {
            out.push(v2t::StreamPart::ToolResult {
                tool_call_id,
                tool_name: parts.tool_name,
                result,
                is_error: parts.is_error,
                preliminary: false,
                provider_metadata: tool_result_metadata,
            });
        }
        if !out.is_empty() {
            return Some(Some(out));
        }
    }
    handled_none()
}

fn emit_provider_tool_approval_request(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    parts: &ProviderToolParts,
) -> Option<Vec<v2t::StreamPart>> {
    if !parts.is_approval_request {
        return None;
    }

    let approval_id = parts
        .approval_request_id
        .clone()
        .unwrap_or_else(|| parts.tool_call_id.clone());
    let tool_call_id = Uuid::new_v4().to_string();
    state
        .extra
        .approval_request_id_map
        .insert(approval_id.clone(), tool_call_id.clone());
    Some(vec![
        state.tool_call_part(
            tool_call_id.clone(),
            parts.tool_name.clone(),
            parts.input.clone(),
            parts.provider_executed,
            None,
            parts.dynamic,
            None,
        ),
        v2t::StreamPart::ToolApprovalRequest {
            approval_id,
            tool_call_id,
            provider_metadata: None,
        },
    ])
}

fn remap_provider_tool_call_id(
    parts: &mut ProviderToolParts,
    approval_request_id_map: &HashMap<String, String>,
) {
    if let Some(approval_id) = parts.approval_request_id.as_ref() {
        if let Some(mapped) = approval_request_id_map.get(approval_id) {
            parts.tool_call_id = mapped.clone();
        }
    }
}

fn provider_tool_stream_metadata(
    parts: &ProviderToolParts,
) -> (Option<v2t::ProviderMetadata>, Option<v2t::ProviderMetadata>) {
    let tool_call_metadata = match parts.tool_type.as_str() {
        "apply_patch" | "local_shell" | "shell" => parts.provider_metadata.clone(),
        _ => None,
    };
    let tool_result_metadata = if parts.tool_type == "mcp" {
        parts.provider_metadata.clone()
    } else {
        None
    };
    (tool_call_metadata, tool_result_metadata)
}

fn maybe_close_provider_tool_input(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    parts: &ProviderToolParts,
    tool_call_id: &str,
    out: &mut Vec<v2t::StreamPart>,
) {
    if parts.tool_type == "computer_use" && state.extra.open_tool_inputs.remove(tool_call_id) {
        out.push(state.tool_input_end_part(tool_call_id.to_string(), true, None));
    }
}

fn should_skip_provider_tool_call(
    state: &EventMapperState<OpenAIStreamExtras>,
    parts: &ProviderToolParts,
    tool_call_id: &str,
) -> bool {
    matches!(
        parts.tool_type.as_str(),
        "web_search" | "file_search" | "image_generation" | "code_interpreter"
    ) && state.extra.emitted_tool_calls.contains(tool_call_id)
}

fn handle_tail_state_event(
    state: &mut EventMapperState<OpenAIStreamExtras>,
    key: &str,
    value: &serde_json::Value,
) -> StreamDataHandlerResult {
    if key.starts_with("openai.tool_item_id.") {
        if let Some(item_id) = value.get("item_id").and_then(|v| v.as_str()) {
            let call_id = key.trim_start_matches("openai.tool_item_id.").to_string();
            state
                .extra
                .tool_item_ids
                .insert(call_id, item_id.to_string());
        }
        return handled_none();
    }

    match key {
        "openai.function_call_done" => {
            state.extra.has_function_calls = true;
            handled_none()
        }
        "openai.finish" => {
            if let Some(reason) = value.get("incomplete_reason").and_then(|v| v.as_str()) {
                state.extra.finish_hint = Some(reason.to_string());
            }
            handled_none()
        }
        "openai.failed" => {
            state.extra.saw_response_failed = true;
            if state.extra.response_id.is_none() {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    state.extra.response_id = Some(id.to_string());
                }
            }
            handled_none()
        }
        "openai.response" => {
            if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                state.extra.response_id = Some(id.to_string());
            }
            if let Some(service_tier) = value.get("service_tier").and_then(|v| v.as_str()) {
                state.extra.service_tier = Some(service_tier.to_string());
            }
            handled_none()
        }
        _ => None,
    }
}

pub(super) fn build_stream_mapper_config(
    warnings: Vec<v2t::CallWarning>,
    tool_name_mapping: ToolNameMapping,
    approval_request_id_map: HashMap<String, String>,
    store: bool,
    logprobs_enabled: bool,
) -> EventMapperConfig<OpenAIStreamExtras> {
    let mut hooks: EventMapperHooks<OpenAIStreamExtras> = EventMapperHooks::default();

    hooks.tool_end_metadata = Some(Box::new(
        |state: &mut EventMapperState<OpenAIStreamExtras>, id| {
            state
                .extra
                .tool_item_ids
                .get(id)
                .map(|iid| openai_item_metadata(iid, []))
        },
    ));

    hooks.data = Some(Box::new(
        |state: &mut EventMapperState<OpenAIStreamExtras>, key, value| {
            if let Some(result) = handle_usage_or_response_metadata_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_message_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_reasoning_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_search_and_image_tool_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_code_interpreter_tool_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_computer_and_apply_patch_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_provider_tool_event(state, key, value) {
                return result;
            }
            if let Some(result) = handle_tail_state_event(state, key, value) {
                return result;
            }
            None
        },
    ));

    hooks.finish = Some(Box::new(|state: &EventMapperState<OpenAIStreamExtras>| {
        let reason = if state.extra.saw_response_failed {
            v2t::FinishReason::Other
        } else {
            map_finish_reason(
                state.extra.finish_hint.as_deref(),
                state.extra.has_function_calls,
            )
        };
        let mut inner = HashMap::new();
        if let Some(rid) = &state.extra.response_id {
            inner.insert("responseId".into(), serde_json::json!(rid));
        }
        if !state.extra.saw_response_failed {
            if let Some(st) = &state.extra.service_tier {
                inner.insert("serviceTier".into(), serde_json::json!(st));
            }
        }
        if !state.extra.logprobs.is_empty() {
            inner.insert(
                "logprobs".into(),
                serde_json::Value::Array(state.extra.logprobs.clone()),
            );
        }
        let metadata = if inner.is_empty() {
            None
        } else {
            let mut outer = HashMap::new();
            outer.insert("openai".into(), inner);
            Some(outer)
        };
        (reason, metadata)
    }));

    EventMapperConfig {
        warnings,
        treat_tool_names_as_text: HashSet::new(),
        default_text_id: "text-1",
        finish_reason_fallback: v2t::FinishReason::Stop,
        initial_extra: OpenAIStreamExtras {
            tool_name_mapping,
            approval_request_id_map,
            store,
            logprobs_enabled,
            ..Default::default()
        },
        hooks,
    }
}
