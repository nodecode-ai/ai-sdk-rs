use std::collections::{HashMap, HashSet};

use crate::ai_sdk_core::{EventMapperConfig, EventMapperHooks, EventMapperState};
use crate::ai_sdk_types::v2 as v2t;
use serde_json::json;
use uuid::Uuid;

use super::language_model::{
    apply_openai_usage_details, escape_json_delta, map_finish_reason, openai_item_metadata,
    parse_openai_usage,
};
use super::provider_tools::{provider_tool_parts_from_data, ToolNameMapping};

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
            if key == "usage" {
                if let Some(usage) = parse_openai_usage(value) {
                    state.usage.input_tokens = Some(usage.input_tokens as u64);
                    state.usage.output_tokens = Some(usage.output_tokens as u64);
                    state.usage.total_tokens = Some(usage.total_tokens as u64);
                    state.usage.cached_input_tokens = usage.cache_read_tokens.map(|v| v as u64);
                }
                apply_openai_usage_details(value, &mut state.usage);
                return None;
            } else if key == "openai.response_metadata" {
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
                let meta = v2t::ResponseMetadata {
                    id,
                    timestamp_ms,
                    model_id,
                };
                return Some(vec![v2t::StreamPart::ResponseMetadata { meta }]);
            } else if key == "openai.message_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                state
                    .extra
                    .message_annotations
                    .insert(item_id.to_string(), Vec::new());
                if state.text_open.as_deref() != Some(item_id) {
                    return Some(
                        state.open_text(
                            item_id.to_string(),
                            Some(openai_item_metadata(item_id, [])),
                        ),
                    );
                }
            } else if key == "openai.text_delta" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                let start_metadata = if state.text_open.as_deref() != Some(item_id) {
                    state
                        .extra
                        .message_annotations
                        .entry(item_id.to_string())
                        .or_default();
                    Some(openai_item_metadata(item_id, []))
                } else {
                    None
                };
                if state.extra.logprobs_enabled {
                    if let Some(logprobs) = value.get("logprobs").filter(|v| !v.is_null()) {
                        state.extra.logprobs.push(logprobs.clone());
                    }
                }
                return Some(state.push_text_delta(
                    Some(item_id.to_string()),
                    item_id,
                    delta.to_string(),
                    start_metadata,
                    None,
                ));
            } else if key == "openai.text_annotation" {
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
                let make_provider_metadata = |vals: Vec<(&str, serde_json::Value)>| {
                    let mut inner = HashMap::new();
                    for (key, val) in vals {
                        inner.insert(key.into(), val);
                    }
                    let mut outer = HashMap::new();
                    outer.insert("openai".into(), inner);
                    outer
                };
                match annotation_type {
                    "url_citation" => {
                        let url = annotation_obj.get("url")?.as_str()?;
                        let title = annotation_obj
                            .get("title")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: url.to_string(),
                            title,
                            provider_metadata: None,
                        }]);
                    }
                    "file_citation" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let title = annotation_obj
                            .get("quote")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                annotation_obj
                                    .get("filename")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .or_else(|| Some(file_id.to_string()));
                        let mut metadata_vals = vec![("fileId", json!(file_id))];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title,
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    "container_file_citation" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let container_id = annotation_obj.get("container_id")?.as_str()?;
                        let title = annotation_obj
                            .get("filename")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| Some(file_id.to_string()));
                        let mut metadata_vals = vec![
                            ("fileId", json!(file_id)),
                            ("containerId", json!(container_id)),
                        ];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title,
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    "file_path" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let mut metadata_vals = vec![("fileId", json!(file_id))];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title: Some(file_id.to_string()),
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    _ => {}
                }
            } else if key == "openai.message_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let annotations = state
                    .extra
                    .message_annotations
                    .remove(item_id)
                    .unwrap_or_default();
                let md = if annotations.is_empty() {
                    openai_item_metadata(item_id, [])
                } else {
                    openai_item_metadata(
                        item_id,
                        [("annotations".into(), serde_json::Value::Array(annotations))],
                    )
                };
                if state.text_open.as_deref() == Some(item_id) {
                    return state.close_text(Some(md)).map(|part| vec![part]);
                }
                return Some(vec![state.text_end_part(item_id.to_string(), Some(md))]);
            } else if key == "openai.error" {
                state.extra.finish_hint = Some("error".into());
                return Some(vec![v2t::StreamPart::Error {
                    error: value.clone(),
                }]);
            } else if key == "openai.reasoning_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let enc = value
                    .get("encrypted_content")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let mut state_entry = OpenAIReasoningState {
                    encrypted_content: Some(enc.clone()),
                    summary_parts: HashMap::new(),
                };
                state_entry
                    .summary_parts
                    .insert(0, ReasoningSummaryStatus::Active);
                state
                    .extra
                    .active_reasoning
                    .insert(item_id.to_string(), state_entry);
                return Some(state.open_reasoning(
                    format!("{item_id}:0"),
                    Some(openai_item_metadata(
                        item_id,
                        [("reasoningEncryptedContent".into(), enc)],
                    )),
                ));
            } else if key == "openai.reasoning_summary_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                if summary_index == 0 {
                    return None;
                }
                let (concluded_ids, enc) = {
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
                    let enc = reasoning_state
                        .encrypted_content
                        .clone()
                        .unwrap_or(serde_json::Value::Null);
                    (concluded_ids, enc)
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
                        out.push(state.reasoning_end_part(
                            reasoning_id,
                            Some(openai_item_metadata(item_id, [])),
                        ));
                    }
                }
                out.extend(state.open_reasoning(
                    format!("{item_id}:{summary_index}"),
                    Some(openai_item_metadata(
                        item_id,
                        [("reasoningEncryptedContent".into(), enc)],
                    )),
                ));
                return Some(out);
            } else if key == "openai.reasoning_summary_delta" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                return Some(vec![state.push_reasoning_delta(
                    &format!("{item_id}:{summary_index}"),
                    delta.to_string(),
                    Some(openai_item_metadata(item_id, [])),
                )]);
            } else if key == "openai.reasoning_summary_done" {
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
                        return state
                            .close_reasoning(Some(openai_item_metadata(item_id, [])))
                            .map(|part| vec![part]);
                    }
                    return Some(vec![state.reasoning_end_part(
                        reasoning_id,
                        Some(openai_item_metadata(item_id, [])),
                    )]);
                }
            } else if key == "openai.reasoning_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let enc = value
                    .get("encrypted_content")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if let Some(reasoning_state) = state.extra.active_reasoning.remove(item_id) {
                    let md =
                        openai_item_metadata(item_id, [("reasoningEncryptedContent".into(), enc)]);
                    let mut out = Vec::new();
                    for (idx, status) in reasoning_state.summary_parts {
                        if matches!(
                            status,
                            ReasoningSummaryStatus::Active | ReasoningSummaryStatus::CanConclude
                        ) {
                            let reasoning_id = format!("{item_id}:{idx}");
                            if state.reasoning_open.as_deref() == Some(reasoning_id.as_str()) {
                                if let Some(part) = state.close_reasoning(Some(md.clone())) {
                                    out.push(part);
                                }
                            } else {
                                out.push(state.reasoning_end_part(reasoning_id, Some(md.clone())));
                            }
                        }
                    }
                    if !out.is_empty() {
                        return Some(out);
                    }
                }
            } else if key == "openai.web_search_call.added" {
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
                let mut out =
                    vec![state.start_tool_call(tool_call_id.clone(), tool_name, true, None)];
                state.tool_args.insert(tool_call_id.clone(), "{}".into());
                out.extend(state.finish_tool_call(tool_call_id, true, None, None, false, None));
                return Some(out);
            } else if key == "openai.file_search_call.added" {
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
                return Some(vec![state.tool_call_part(
                    tool_call_id.to_string(),
                    tool_name,
                    "{}".into(),
                    true,
                    None,
                    false,
                    None,
                )]);
            } else if key == "openai.image_generation_call.added" {
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
                return Some(vec![state.tool_call_part(
                    tool_call_id.to_string(),
                    tool_name,
                    "{}".into(),
                    true,
                    None,
                    false,
                    None,
                )]);
            } else if key == "openai.image_generation_call.partial" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let partial = value.get("partial_image_b64").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("image_generation")
                    .to_string();
                return Some(vec![v2t::StreamPart::ToolResult {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name,
                    result: json!({ "result": partial }),
                    is_error: false,
                    preliminary: true,
                    provider_metadata: None,
                }]);
            } else if key == "openai.code_interpreter_call.added" {
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
                let cid = container_id.unwrap_or_default();
                return Some(vec![
                    state.start_tool_call(tool_call_id.to_string(), tool_name, true, None),
                    state.push_tool_call_delta(
                        tool_call_id.to_string(),
                        format!(
                            "{{\"containerId\":\"{}\",\"code\":\"",
                            escape_json_delta(&cid)
                        ),
                        true,
                        None,
                    ),
                ]);
            } else if key == "openai.code_interpreter_call.code_delta" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                if let Some(call_state) = state.extra.code_interpreter_calls.get(&output_index) {
                    return Some(vec![state.push_tool_call_delta(
                        call_state.tool_call_id.clone(),
                        escape_json_delta(delta),
                        true,
                        None,
                    )]);
                }
            } else if key == "openai.code_interpreter_call.code_done" {
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
                    return Some(out);
                }
            } else if key == "openai.computer_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                state
                    .extra
                    .open_tool_inputs
                    .insert(tool_call_id.to_string());
                state.has_tool_calls = true;
                return Some(vec![state.start_tool_call(
                    tool_call_id.to_string(),
                    state
                        .extra
                        .tool_name_mapping
                        .to_custom_tool_name("computer_use")
                        .to_string(),
                    true,
                    None,
                )]);
            } else if key == "openai.apply_patch_call.added" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let call_id = value.get("call_id").and_then(|v| v.as_str())?;
                let operation = value
                    .get("operation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let operation_type = operation.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if operation_type.is_empty() {
                    return None;
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
                let mut out =
                    vec![state.start_tool_call(call_id.to_string(), tool_name, false, None)];
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
                return Some(out);
            } else if key == "openai.apply_patch_call.diff.delta" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(call_state) = state.extra.apply_patch_calls.get_mut(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    if !delta.is_empty() {
                        call_state.has_diff = true;
                        let tool_call_id = call_state.tool_call_id.clone();
                        return Some(vec![state.push_tool_call_delta(
                            tool_call_id,
                            escape_json_delta(delta),
                            false,
                            None,
                        )]);
                    }
                }
            } else if key == "openai.apply_patch_call.diff.done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let diff = value.get("diff").and_then(|v| v.as_str()).unwrap_or("");
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
                    return Some(out);
                }
            } else if key == "openai.apply_patch_call.done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                if let Some(mut call_state) = state.extra.apply_patch_calls.remove(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    let mut out = Vec::new();
                    if !call_state.has_diff {
                        let diff = value
                            .get("operation")
                            .and_then(|v| v.get("diff"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        call_state.has_diff = true;
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
                    out.push(state.tool_input_end_part(
                        call_state.tool_call_id.clone(),
                        false,
                        None,
                    ));
                    call_state.end_emitted = true;
                    return Some(out);
                }
            } else if key == "openai.provider_tool" {
                if let Some(mut parts) =
                    provider_tool_parts_from_data(value, &state.extra.tool_name_mapping)
                {
                    state.has_tool_calls = true;
                    let tool_type = parts.tool_type.clone();
                    if parts.is_approval_request {
                        let approval_id = parts
                            .approval_request_id
                            .clone()
                            .unwrap_or_else(|| parts.tool_call_id.clone());
                        let tool_call_id = Uuid::new_v4().to_string();
                        state
                            .extra
                            .approval_request_id_map
                            .insert(approval_id.clone(), tool_call_id.clone());
                        return Some(vec![
                            state.tool_call_part(
                                tool_call_id.clone(),
                                parts.tool_name,
                                parts.input,
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
                        ]);
                    }
                    if let Some(approval_id) = parts.approval_request_id.as_ref() {
                        if let Some(mapped) = state.extra.approval_request_id_map.get(approval_id) {
                            parts.tool_call_id = mapped.clone();
                        }
                    }
                    let tool_call_id = parts.tool_call_id.clone();
                    let tool_call_metadata = match tool_type.as_str() {
                        "apply_patch" | "local_shell" | "shell" => parts.provider_metadata.clone(),
                        _ => None,
                    };
                    let tool_result_metadata = if tool_type == "mcp" {
                        parts.provider_metadata.clone()
                    } else {
                        None
                    };
                    let mut out = Vec::new();
                    if tool_type == "computer_use" {
                        if state.extra.open_tool_inputs.remove(&tool_call_id) {
                            out.push(state.tool_input_end_part(tool_call_id.clone(), true, None));
                        }
                    }
                    let skip_tool_call =
                        matches!(
                            tool_type.as_str(),
                            "web_search" | "file_search" | "image_generation" | "code_interpreter"
                        ) && state.extra.emitted_tool_calls.contains(&tool_call_id);
                    if !skip_tool_call {
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
                            tool_call_id: tool_call_id.clone(),
                            tool_name: parts.tool_name,
                            result,
                            is_error: parts.is_error,
                            preliminary: false,
                            provider_metadata: tool_result_metadata,
                        });
                    }
                    if !out.is_empty() {
                        return Some(out);
                    }
                }
            } else if key.starts_with("openai.tool_item_id.") {
                if let Some(iid) = value.get("item_id").and_then(|v| v.as_str()) {
                    let call_id = key.trim_start_matches("openai.tool_item_id.").to_string();
                    state.extra.tool_item_ids.insert(call_id, iid.to_string());
                }
            } else if key == "openai.function_call_done" {
                state.extra.has_function_calls = true;
            } else if key == "openai.finish" {
                if let Some(r) = value.get("incomplete_reason").and_then(|v| v.as_str()) {
                    state.extra.finish_hint = Some(r.to_string());
                }
            } else if key == "openai.failed" {
                state.extra.saw_response_failed = true;
                if state.extra.response_id.is_none() {
                    if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                        state.extra.response_id = Some(id.to_string());
                    }
                }
            } else if key == "openai.response" {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    state.extra.response_id = Some(id.to_string());
                }
                if let Some(st) = value.get("service_tier").and_then(|v| v.as_str()) {
                    state.extra.service_tier = Some(st.to_string());
                }
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
