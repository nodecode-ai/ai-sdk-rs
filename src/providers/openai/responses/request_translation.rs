use std::collections::{HashMap, HashSet};

use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_core::options::merge_options_with_disallow;
use crate::ai_sdk_core::request_builder::defaults::request_overrides_from_json;
use crate::ai_sdk_types::v2 as v2t;
use base64::Engine;
use serde_json::{json, Value};

use super::language_model::{
    normalize_object_schema, should_use_codex_oauth_websocket_transport, ResponseTransportMode,
};
use super::provider_tools::{build_openai_provider_tool, build_tool_name_mapping, ToolNameMapping};
use crate::provider_openai::config::OpenAIConfig;

const TOP_LOGPROBS_MAX: u32 = 20;

#[derive(Clone, Copy, Debug)]
pub(super) enum SystemMessageMode {
    Remove,
    System,
    Developer,
}

fn parse_system_message_mode(value: &str) -> Option<SystemMessageMode> {
    match value {
        "remove" => Some(SystemMessageMode::Remove),
        "system" => Some(SystemMessageMode::System),
        "developer" => Some(SystemMessageMode::Developer),
        _ => None,
    }
}

#[derive(Clone)]
struct ResponsesModelConfig {
    is_reasoning_model: bool,
    system_message_mode: SystemMessageMode,
    required_auto_truncation: bool,
    supports_flex_processing: bool,
    supports_priority_processing: bool,
    supports_non_reasoning_parameters: bool,
}

fn get_responses_model_config(model_id: &str) -> ResponsesModelConfig {
    let supports_flex = model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
        || (model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat"));
    let supports_priority = model_id.starts_with("gpt-4")
        || model_id.starts_with("gpt-5-mini")
        || (model_id.starts_with("gpt-5")
            && !model_id.starts_with("gpt-5-nano")
            && !model_id.starts_with("gpt-5-chat"))
        || model_id.starts_with("o3")
        || model_id.starts_with("o4-mini");
    let is_reasoning = model_id.starts_with("o1")
        || model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
        || model_id.starts_with("codex-mini")
        || model_id.starts_with("computer-use-preview")
        || (model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat"));
    let system_mode = if is_reasoning {
        SystemMessageMode::Developer
    } else {
        SystemMessageMode::System
    };
    let supports_non_reasoning_parameters =
        model_id.starts_with("gpt-5.1") || model_id.starts_with("gpt-5.2");

    ResponsesModelConfig {
        is_reasoning_model: is_reasoning,
        system_message_mode: system_mode,
        required_auto_truncation: false,
        supports_flex_processing: supports_flex,
        supports_priority_processing: supports_priority,
        supports_non_reasoning_parameters,
    }
}

#[derive(Default, Debug, Clone)]
pub(super) struct OpenAIProviderOptionsParsed {
    pub(super) conversation: Option<String>,
    pub(super) client_metadata: Option<serde_json::Value>,
    pub(super) metadata: Option<serde_json::Value>,
    pub(super) max_tool_calls: Option<u32>,
    pub(super) parallel_tool_calls: Option<bool>,
    pub(super) previous_response_id: Option<String>,
    pub(super) store: Option<bool>,
    pub(super) user: Option<String>,
    pub(super) instructions: Option<String>,
    pub(super) service_tier: Option<String>,
    pub(super) include: Option<Vec<String>>,
    pub(super) text_verbosity: Option<String>,
    pub(super) prompt_cache_key: Option<String>,
    pub(super) prompt_cache_retention: Option<String>,
    pub(super) safety_identifier: Option<String>,
    pub(super) system_message_mode: Option<SystemMessageMode>,
    pub(super) force_reasoning: Option<bool>,
    pub(super) strict_json_schema: Option<bool>,
    pub(super) truncation: Option<String>,
    pub(super) reasoning_effort: Option<String>,
    pub(super) reasoning_summary: Option<String>,
    pub(super) logprobs_bool: Option<bool>,
    pub(super) logprobs_n: Option<u32>,
    pub(super) transport_mode: Option<ResponseTransportMode>,
    pub(super) transport_fallback_http: bool,
}

#[derive(Debug, Clone)]
struct OpenAIRequestToolSettings {
    parallel_tool_calls: Option<bool>,
    tool_choice: Option<v2t::ToolChoice>,
}

pub(super) fn parse_openai_provider_options(
    opts: &v2t::ProviderOptions,
    provider_scope: &str,
) -> OpenAIProviderOptionsParsed {
    let mut parsed = OpenAIProviderOptionsParsed::default();
    let map = match opts.get(provider_scope) {
        Some(map) => map,
        None => return parsed,
    };
    let get_bool = |k: &str| map.get(k).and_then(|v| v.as_bool());
    let get_str = |k: &str| map.get(k).and_then(|v| v.as_str().map(|s| s.to_string()));
    let get_arr = |k: &str| {
        map.get(k).and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
    };

    parsed.conversation = get_str("conversation");
    parsed.client_metadata = map
        .get("clientMetadata")
        .cloned()
        .or_else(|| map.get("client_metadata").cloned());
    parsed.metadata = map.get("metadata").cloned();
    parsed.max_tool_calls = map
        .get("maxToolCalls")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok());
    parsed.parallel_tool_calls = get_bool("parallelToolCalls");
    parsed.previous_response_id = get_str("previousResponseId");
    parsed.store = get_bool("store");
    parsed.user = get_str("user");
    parsed.instructions = get_str("instructions");
    parsed.service_tier = get_str("serviceTier");
    parsed.include = get_arr("include");
    parsed.text_verbosity = get_str("textVerbosity");
    parsed.prompt_cache_key = get_str("promptCacheKey");
    parsed.prompt_cache_retention = get_str("promptCacheRetention");
    parsed.safety_identifier = get_str("safetyIdentifier");
    parsed.system_message_mode = map
        .get("systemMessageMode")
        .and_then(|v| v.as_str())
        .and_then(parse_system_message_mode);
    parsed.force_reasoning = get_bool("forceReasoning");
    parsed.strict_json_schema = get_bool("strictJsonSchema");
    parsed.truncation = get_str("truncation");
    parsed.reasoning_effort = get_str("reasoningEffort");
    parsed.reasoning_summary = get_str("reasoningSummary");

    if let Some(v) = map.get("logprobs") {
        if let Some(b) = v.as_bool() {
            parsed.logprobs_bool = Some(b);
        }
        if let Some(n) = v.as_u64() {
            parsed.logprobs_n = Some(n as u32);
        }
    }
    if let Some((mode, fallback_http)) = parse_transport_provider_options(opts, provider_scope) {
        parsed.transport_mode = Some(mode);
        parsed.transport_fallback_http = fallback_http;
    }
    parsed
}

fn resolve_request_tool_settings(
    endpoint_path: &str,
    provider_options: &OpenAIProviderOptionsParsed,
    tool_choice: &Option<v2t::ToolChoice>,
) -> OpenAIRequestToolSettings {
    let codex_defaults = should_use_codex_oauth_websocket_transport(endpoint_path);
    OpenAIRequestToolSettings {
        parallel_tool_calls: provider_options
            .parallel_tool_calls
            .or_else(|| codex_defaults.then_some(true)),
        tool_choice: tool_choice
            .clone()
            .or_else(|| codex_defaults.then_some(v2t::ToolChoice::Auto)),
    }
}

fn parse_transport_provider_options(
    opts: &v2t::ProviderOptions,
    provider_scope: &str,
) -> Option<(ResponseTransportMode, bool)> {
    for scope_name in [provider_scope, "openai", "openai.responses"] {
        let Some(scope) = opts.get(scope_name) else {
            continue;
        };
        let Some(transport) = scope.get("transport").and_then(|value| value.as_object()) else {
            continue;
        };
        let mode = transport
            .get("mode")
            .and_then(|value| value.as_str())
            .and_then(parse_transport_mode)?;
        let fallback_http = transport
            .get("fallback")
            .and_then(|value| value.as_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("http"));
        return Some((mode, fallback_http));
    }
    None
}

fn parse_transport_mode(value: &str) -> Option<ResponseTransportMode> {
    if value.eq_ignore_ascii_case("websocket") {
        Some(ResponseTransportMode::Websocket)
    } else if value.eq_ignore_ascii_case("http") {
        Some(ResponseTransportMode::Http)
    } else {
        None
    }
}

fn get_provider_option_value<'a>(
    provider_options: &'a Option<v2t::ProviderOptions>,
    provider_scope: &str,
    key: &str,
) -> Option<&'a serde_json::Value> {
    provider_options
        .as_ref()
        .and_then(|opts| opts.get(provider_scope))
        .and_then(|opts| opts.get(key))
}

fn get_provider_option_string(
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope: &str,
    key: &str,
) -> Option<String> {
    get_provider_option_value(provider_options, provider_scope, key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn openai_item_id_from_provider_options(
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope: &str,
) -> Option<String> {
    get_provider_option_string(provider_options, provider_scope, "itemId")
}

fn openai_reasoning_encrypted_content_from_provider_options(
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope: &str,
) -> Option<String> {
    get_provider_option_string(
        provider_options,
        provider_scope,
        "reasoningEncryptedContent",
    )
}

fn convert_to_openai_messages(
    prompt: &v2t::Prompt,
    system_mode: SystemMessageMode,
    file_id_prefixes: Option<&[String]>,
    provider_scope_name: &str,
    store: bool,
    tool_name_mapping: &ToolNameMapping,
    has_local_shell_tool: bool,
    has_shell_tool: bool,
    has_apply_patch_tool: bool,
) -> (Vec<Value>, Vec<v2t::CallWarning>) {
    let mut messages: Vec<Value> = Vec::new();
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();
    let mut processed_approval_ids = HashSet::new();

    fn to_data_uri(media_type: &str, data: &v2t::DataContent) -> Option<String> {
        match data {
            v2t::DataContent::Base64 { base64 } => {
                Some(format!("data:{};base64,{}", media_type, base64))
            }
            v2t::DataContent::Bytes { bytes } => Some(format!(
                "data:{};base64,{}",
                media_type,
                base64::engine::general_purpose::STANDARD.encode(bytes)
            )),
            v2t::DataContent::Url { url } => Some(url.clone()),
        }
    }

    for msg in prompt {
        match msg {
            v2t::PromptMessage::System { content, .. } => match system_mode {
                SystemMessageMode::Remove => warnings.push(v2t::CallWarning::Other {
                    message: "system messages are removed for this model".into(),
                }),
                SystemMessageMode::System => {
                    messages.push(json!({"role":"system","content": content }))
                }
                SystemMessageMode::Developer => {
                    messages.push(json!({"role":"developer","content": content }))
                }
            },
            v2t::PromptMessage::User { content, .. } => {
                let mut parts: Vec<Value> = Vec::new();
                for (idx, part) in content.iter().enumerate() {
                    match part {
                        v2t::UserPart::Text { text, .. } => {
                            parts.push(json!({"type":"input_text","text": text}))
                        }
                        v2t::UserPart::File {
                            filename,
                            data,
                            media_type,
                            ..
                        } => {
                            if media_type.starts_with("image/") {
                                let mt = if media_type == "image/*" {
                                    "image/jpeg"
                                } else {
                                    media_type
                                };
                                if let Some(url) = to_data_uri(mt, data) {
                                    let is_http =
                                        url.starts_with("http://") || url.starts_with("https://");
                                    if !is_http {
                                        if let Some(prefixes) = file_id_prefixes {
                                            if prefixes.iter().any(|p| url.starts_with(p)) {
                                                parts.push(
                                                    json!({"type":"input_image","file_id": url }),
                                                );
                                                continue;
                                            }
                                        }
                                    }
                                    parts.push(json!({"type":"input_image","image_url": url }));
                                }
                            } else if media_type == "application/pdf" {
                                match data {
                                    v2t::DataContent::Url { url } => {
                                        parts.push(json!({"type":"input_file","file_url": url}))
                                    }
                                    _ => {
                                        if let v2t::DataContent::Base64 { base64 } = data {
                                            if let Some(prefixes) = file_id_prefixes {
                                                if prefixes.iter().any(|p| base64.starts_with(p)) {
                                                    parts.push(json!({"type":"input_file","file_id": base64}));
                                                    continue;
                                                }
                                            }
                                        }
                                        let fname = filename
                                            .clone()
                                            .unwrap_or_else(|| format!("part-{}.pdf", idx));
                                        if let Some(uri) = to_data_uri("application/pdf", data) {
                                            parts.push(json!({"type":"input_file","filename": fname, "file_data": uri}));
                                        }
                                    }
                                }
                            } else {
                                warnings.push(v2t::CallWarning::Other {
                                    message: format!("unsupported file media type: {}", media_type),
                                });
                            }
                        }
                    }
                }
                if !parts.is_empty() {
                    messages.push(json!({"role":"user","content": parts}));
                }
            }
            v2t::PromptMessage::Assistant { content, .. } => {
                let mut reasoning_message_idx = HashMap::new();
                let mut reasoning_item_refs = HashSet::new();
                for part in content {
                    match part {
                        v2t::AssistantPart::Text {
                            text,
                            provider_options,
                        } => {
                            let item_id = openai_item_id_from_provider_options(
                                provider_options,
                                provider_scope_name,
                            );
                            if store && item_id.is_some() {
                                messages.push(json!({"type":"item_reference","id": item_id}));
                                continue;
                            }
                            let mut message = serde_json::Map::new();
                            message.insert("role".into(), json!("assistant"));
                            message.insert(
                                "content".into(),
                                json!([{"type":"output_text","text": text}]),
                            );
                            if let Some(id) = item_id {
                                message.insert("id".into(), json!(id));
                            }
                            messages.push(Value::Object(message));
                        }
                        v2t::AssistantPart::ToolCall(tc) => {
                            let item_id = openai_item_id_from_provider_options(
                                &tc.provider_options,
                                provider_scope_name,
                            );
                            if tc.provider_executed {
                                if store && item_id.is_some() {
                                    messages.push(json!({"type":"item_reference","id": item_id}));
                                }
                                continue;
                            }
                            if store && item_id.is_some() {
                                messages.push(json!({"type":"item_reference","id": item_id}));
                                continue;
                            }
                            let resolved_tool_name =
                                tool_name_mapping.to_provider_tool_name(&tc.tool_name);
                            let input_json: serde_json::Value =
                                serde_json::from_str(&tc.input).unwrap_or(json!({}));
                            if has_local_shell_tool && resolved_tool_name == "local_shell" {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("local_shell_call"));
                                obj.insert("call_id".into(), json!(tc.tool_call_id));
                                if let Some(id) = item_id {
                                    obj.insert("id".into(), json!(id));
                                }
                                let mut action = serde_json::Map::new();
                                action.insert("type".into(), json!("exec"));
                                if let Some(action_obj) =
                                    input_json.get("action").and_then(|v| v.as_object())
                                {
                                    if let Some(command) = action_obj.get("command") {
                                        action.insert("command".into(), command.clone());
                                    }
                                    if let Some(timeout) = action_obj
                                        .get("timeoutMs")
                                        .or_else(|| action_obj.get("timeout_ms"))
                                    {
                                        action.insert("timeout_ms".into(), timeout.clone());
                                    }
                                    if let Some(user) = action_obj.get("user") {
                                        action.insert("user".into(), user.clone());
                                    }
                                    if let Some(dir) = action_obj
                                        .get("workingDirectory")
                                        .or_else(|| action_obj.get("working_directory"))
                                    {
                                        action.insert("working_directory".into(), dir.clone());
                                    }
                                    if let Some(env) = action_obj.get("env") {
                                        action.insert("env".into(), env.clone());
                                    }
                                }
                                obj.insert("action".into(), Value::Object(action));
                                messages.push(Value::Object(obj));
                                continue;
                            }
                            if has_shell_tool && resolved_tool_name == "shell" {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("shell_call"));
                                obj.insert("call_id".into(), json!(tc.tool_call_id));
                                if let Some(id) = item_id {
                                    obj.insert("id".into(), json!(id));
                                }
                                obj.insert("status".into(), json!("completed"));
                                let mut action = serde_json::Map::new();
                                if let Some(action_obj) =
                                    input_json.get("action").and_then(|v| v.as_object())
                                {
                                    if let Some(commands) = action_obj.get("commands") {
                                        action.insert("commands".into(), commands.clone());
                                    }
                                    if let Some(timeout) = action_obj
                                        .get("timeoutMs")
                                        .or_else(|| action_obj.get("timeout_ms"))
                                    {
                                        action.insert("timeout_ms".into(), timeout.clone());
                                    }
                                    if let Some(max_len) = action_obj
                                        .get("maxOutputLength")
                                        .or_else(|| action_obj.get("max_output_length"))
                                    {
                                        action.insert("max_output_length".into(), max_len.clone());
                                    }
                                }
                                obj.insert("action".into(), Value::Object(action));
                                messages.push(Value::Object(obj));
                                continue;
                            }
                            if has_apply_patch_tool && resolved_tool_name == "apply_patch" {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("apply_patch_call"));
                                obj.insert("call_id".into(), json!(tc.tool_call_id));
                                if let Some(id) = item_id {
                                    obj.insert("id".into(), json!(id));
                                }
                                obj.insert("status".into(), json!("completed"));
                                if let Some(operation) = input_json.get("operation").cloned() {
                                    obj.insert("operation".into(), operation);
                                }
                                messages.push(Value::Object(obj));
                                continue;
                            }
                            let mut obj = serde_json::Map::new();
                            obj.insert("type".into(), json!("function_call"));
                            obj.insert("call_id".into(), json!(tc.tool_call_id));
                            obj.insert(
                                "name".into(),
                                json!(tool_name_mapping.to_provider_tool_name(&tc.tool_name)),
                            );
                            obj.insert("arguments".into(), json!(tc.input));
                            if let Some(id) = item_id {
                                obj.insert("id".into(), json!(id));
                            }
                            messages.push(Value::Object(obj));
                        }
                        v2t::AssistantPart::ToolResult(tr) => {
                            if store {
                                let item_id = openai_item_id_from_provider_options(
                                    &tr.provider_options,
                                    provider_scope_name,
                                )
                                .unwrap_or_else(|| tr.tool_call_id.clone());
                                messages.push(json!({"type":"item_reference","id": item_id}));
                            } else {
                                warnings.push(v2t::CallWarning::Other {
                                    message: format!(
                                        "Results for OpenAI tool {} are not sent to the API when store is false",
                                        tr.tool_name
                                    ),
                                });
                            }
                        }
                        v2t::AssistantPart::Reasoning {
                            text,
                            provider_options,
                        } => {
                            let item_id = openai_item_id_from_provider_options(
                                provider_options,
                                provider_scope_name,
                            );
                            let encrypted =
                                openai_reasoning_encrypted_content_from_provider_options(
                                    provider_options,
                                    provider_scope_name,
                                );
                            let Some(reasoning_id) = item_id else {
                                warnings.push(v2t::CallWarning::Other {
                                    message: format!(
                                        "Non-OpenAI reasoning parts are not supported. Skipping reasoning part: {}",
                                        text
                                    ),
                                });
                                continue;
                            };
                            if store {
                                if reasoning_item_refs.insert(reasoning_id.clone()) {
                                    messages
                                        .push(json!({"type":"item_reference","id": reasoning_id}));
                                }
                                continue;
                            }
                            let summary_entry = if text.is_empty() {
                                None
                            } else {
                                Some(json!({"type":"summary_text","text": text}))
                            };
                            if let Some(idx) = reasoning_message_idx.get(&reasoning_id).copied() {
                                if let Some(entry) = summary_entry {
                                    if let Some(obj) = messages
                                        .get_mut(idx)
                                        .and_then(|v: &mut Value| v.as_object_mut())
                                    {
                                        if let Some(summary) = obj
                                            .get_mut("summary")
                                            .and_then(|v: &mut Value| v.as_array_mut())
                                        {
                                            summary.push(entry);
                                        }
                                        if let Some(enc) = encrypted.clone() {
                                            obj.insert("encrypted_content".into(), json!(enc));
                                        }
                                    }
                                } else {
                                    warnings.push(v2t::CallWarning::Other {
                                        message: format!(
                                            "Cannot append empty reasoning part to existing reasoning sequence. Skipping reasoning part: {}",
                                            reasoning_id
                                        ),
                                    });
                                }
                            } else {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("reasoning"));
                                obj.insert("id".into(), json!(reasoning_id.clone()));
                                let summary = summary_entry
                                    .map(|entry| Value::Array(vec![entry]))
                                    .unwrap_or_else(|| Value::Array(Vec::new()));
                                obj.insert("summary".into(), summary);
                                if let Some(enc) = encrypted {
                                    obj.insert("encrypted_content".into(), json!(enc));
                                }
                                messages.push(Value::Object(obj));
                                reasoning_message_idx.insert(reasoning_id, messages.len() - 1);
                            }
                        }
                        v2t::AssistantPart::File { .. } => {}
                    }
                }
            }
            v2t::PromptMessage::Tool { content, .. } => {
                for part in content {
                    match part {
                        v2t::ToolMessagePart::ToolApprovalResponse(resp) => {
                            if !processed_approval_ids.insert(resp.approval_id.clone()) {
                                continue;
                            }
                            if store {
                                messages
                                    .push(json!({"type":"item_reference","id": resp.approval_id}));
                            }
                            messages.push(json!({
                                "type":"mcp_approval_response",
                                "approval_request_id": resp.approval_id,
                                "approve": resp.approved,
                            }));
                        }
                        v2t::ToolMessagePart::ToolResult(tr) => {
                            let resolved_tool_name =
                                tool_name_mapping.to_provider_tool_name(&tr.tool_name);
                            if has_local_shell_tool && resolved_tool_name == "local_shell" {
                                if let v2t::ToolResultOutput::Json { value } = &tr.output {
                                    if let Some(obj) = value.as_object() {
                                        if let Some(output) = obj.get("output") {
                                            messages.push(json!({
                                                "type":"local_shell_call_output",
                                                "call_id": tr.tool_call_id,
                                                "output": output.clone(),
                                            }));
                                            continue;
                                        }
                                    }
                                }
                            }
                            if has_shell_tool && resolved_tool_name == "shell" {
                                if let v2t::ToolResultOutput::Json { value } = &tr.output {
                                    if let Some(obj) = value.as_object() {
                                        if let Some(output) =
                                            obj.get("output").and_then(|v| v.as_array())
                                        {
                                            let mapped = output
                                                .iter()
                                                .filter_map(|entry| {
                                                    let entry = entry.as_object()?;
                                                    let stdout = entry.get("stdout")?.clone();
                                                    let stderr = entry.get("stderr")?.clone();
                                                    let outcome_obj = entry.get("outcome")?.as_object()?;
                                                    let outcome_type = outcome_obj.get("type")?.as_str()?;
                                                    let outcome = if outcome_type == "timeout" {
                                                        json!({"type":"timeout"})
                                                    } else if outcome_type == "exit" {
                                                        let exit_code = outcome_obj
                                                            .get("exitCode")
                                                            .or_else(|| outcome_obj.get("exit_code"))?;
                                                        json!({"type":"exit","exit_code": exit_code})
                                                    } else {
                                                        return None;
                                                    };
                                                    Some(json!({
                                                        "stdout": stdout,
                                                        "stderr": stderr,
                                                        "outcome": outcome,
                                                    }))
                                                })
                                                .collect::<Vec<_>>();
                                            messages.push(json!({
                                                "type":"shell_call_output",
                                                "call_id": tr.tool_call_id,
                                                "output": mapped,
                                            }));
                                            continue;
                                        }
                                    }
                                }
                            }
                            if has_apply_patch_tool && resolved_tool_name == "apply_patch" {
                                if let v2t::ToolResultOutput::Json { value } = &tr.output {
                                    if let Some(obj) = value.as_object() {
                                        if let Some(status) = obj.get("status") {
                                            let mut out = serde_json::Map::new();
                                            out.insert(
                                                "type".into(),
                                                json!("apply_patch_call_output"),
                                            );
                                            out.insert("call_id".into(), json!(tr.tool_call_id));
                                            out.insert("status".into(), status.clone());
                                            if let Some(output) = obj.get("output") {
                                                out.insert("output".into(), output.clone());
                                            }
                                            messages.push(Value::Object(out));
                                            continue;
                                        }
                                    }
                                }
                            }
                            let out_val = tool_output_to_value(&tr.output);
                            messages.push(json!({
                                "type":"function_call_output",
                                "call_id": tr.tool_call_id,
                                "output": out_val,
                            }));
                        }
                    }
                }
            }
        }
    }

    (messages, warnings)
}

fn tool_output_to_value(output: &v2t::ToolResultOutput) -> Value {
    match output {
        v2t::ToolResultOutput::Text { value } => Value::String(value.clone()),
        v2t::ToolResultOutput::ErrorText { value } => Value::String(value.clone()),
        v2t::ToolResultOutput::Json { value } => Value::String(value.to_string()),
        v2t::ToolResultOutput::ErrorJson { value } => Value::String(value.to_string()),
        v2t::ToolResultOutput::Content { value } => {
            let mut parts: Vec<Value> = Vec::new();
            for item in value {
                match item {
                    v2t::ToolResultInlineContent::Text { text } => {
                        parts.push(json!({"type":"input_text","text": text}));
                    }
                    v2t::ToolResultInlineContent::Media { data, media_type } => {
                        if media_type.starts_with("image/") {
                            parts.push(json!({
                                "type":"input_image",
                                "image_url": format!("data:{};base64,{}", media_type, data),
                            }));
                        } else {
                            parts.push(json!({
                                "type":"input_file",
                                "filename": "data",
                                "file_data": format!("data:{};base64,{}", media_type, data),
                            }));
                        }
                    }
                }
            }
            Value::Array(parts)
        }
    }
}

fn is_openai_builtin_tool(name: &str) -> bool {
    matches!(
        name,
        "file_search"
            | "local_shell"
            | "shell"
            | "apply_patch"
            | "web_search_preview"
            | "web_search"
            | "code_interpreter"
            | "image_generation"
            | "mcp"
    )
}

fn map_tool_choice(
    choice: &Option<v2t::ToolChoice>,
    tool_name_mapping: &ToolNameMapping,
) -> Option<Value> {
    match choice {
        None => None,
        Some(v2t::ToolChoice::Auto) => Some(Value::String("auto".into())),
        Some(v2t::ToolChoice::None) => Some(Value::String("none".into())),
        Some(v2t::ToolChoice::Required) => Some(Value::String("required".into())),
        Some(v2t::ToolChoice::Tool { name }) => {
            let mapped = tool_name_mapping.to_provider_tool_name(name);
            if is_openai_builtin_tool(mapped) {
                Some(json!({"type": mapped}))
            } else {
                Some(json!({"type":"function","name": mapped}))
            }
        }
    }
}

fn function_tool_strict(tool: &v2t::FunctionTool) -> Option<bool> {
    tool.strict.or_else(|| {
        tool.provider_options.as_ref().and_then(|opts| {
            opts.get("openai")
                .or_else(|| opts.get("openai.responses"))
                .and_then(|scope| scope.get("strict"))
                .and_then(|value| value.as_bool())
        })
    })
}

pub(super) fn build_request_body(
    options: &v2t::CallOptions,
    model_id: &str,
    cfg: &OpenAIConfig,
) -> Result<(Value, Vec<v2t::CallWarning>), SdkError> {
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();
    if options.top_k.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "topK".into(),
            details: None,
        });
    }
    if options.seed.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "seed".into(),
            details: None,
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "presencePenalty".into(),
            details: None,
        });
    }
    if options.frequency_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "frequencyPenalty".into(),
            details: None,
        });
    }
    if options.stop_sequences.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "stopSequences".into(),
            details: None,
        });
    }

    let prov = parse_openai_provider_options(&options.provider_options, &cfg.provider_scope_name);
    let request_tool_settings =
        resolve_request_tool_settings(&cfg.endpoint_path, &prov, &options.tool_choice);
    let model_cfg = get_responses_model_config(model_id);
    let base_is_reasoning_model = model_cfg.is_reasoning_model;
    let is_reasoning_model = prov.force_reasoning.unwrap_or(base_is_reasoning_model);
    let system_message_mode = prov.system_message_mode.unwrap_or_else(|| {
        if is_reasoning_model {
            SystemMessageMode::Developer
        } else {
            model_cfg.system_message_mode
        }
    });
    let tool_name_mapping = build_tool_name_mapping(&options.tools);
    let has_local_shell_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.local_shell"));
    let has_shell_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.shell"));
    let has_apply_patch_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.apply_patch"));
    let has_web_search_tool = options.tools.iter().any(|tool| {
        matches!(
            tool,
            v2t::Tool::Provider(t)
                if t.id == "openai.web_search" || t.id == "openai.web_search_preview"
        )
    });
    let has_code_interpreter_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.code_interpreter"));
    let (messages, mut message_warnings) = convert_to_openai_messages(
        &options.prompt,
        system_message_mode,
        cfg.file_id_prefixes.as_deref(),
        &cfg.provider_scope_name,
        prov.store.unwrap_or(true),
        &tool_name_mapping,
        has_local_shell_tool,
        has_shell_tool,
        has_apply_patch_tool,
    );
    warnings.append(&mut message_warnings);

    if prov.conversation.is_some() && prov.previous_response_id.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "conversation".into(),
            details: Some("conversation and previousResponseId cannot be used together".into()),
        });
    }
    let store_value = prov.store;
    let top_logprobs = if let Some(n) = prov.logprobs_n {
        Some(n)
    } else if prov.logprobs_bool == Some(true) {
        Some(TOP_LOGPROBS_MAX)
    } else {
        None
    };
    let logprobs_requested = prov.logprobs_bool == Some(true) || prov.logprobs_n.unwrap_or(0) > 0;
    let mut include = prov.include.clone();
    let mut add_include = |key: &str| {
        if let Some(list) = include.as_mut() {
            if !list.iter().any(|s| s == key) {
                list.push(key.to_string());
            }
        } else {
            include = Some(vec![key.to_string()]);
        }
    };
    if logprobs_requested {
        add_include("message.output_text.logprobs");
    }
    if has_web_search_tool {
        add_include("web_search_call.action.sources");
    }
    if has_code_interpreter_tool {
        add_include("code_interpreter_call.outputs");
    }
    if store_value == Some(false) && is_reasoning_model {
        add_include("reasoning.encrypted_content");
    }

    let mut text_obj: Option<Value> = None;
    if let Some(v2t::ResponseFormat::Json {
        schema,
        name,
        description,
    }) = &options.response_format
    {
        let mut format_obj = json!({"type":"json_object"});
        if let Some(s) = schema {
            format_obj = json!({
                "type": "json_schema",
                "strict": prov.strict_json_schema.unwrap_or(true),
                "name": name.clone().unwrap_or_else(|| "response".into()),
                "description": description.clone(),
                "schema": s,
            });
        }
        text_obj = Some(json!({"format": format_obj}));
    }
    if let Some(v) = prov.text_verbosity {
        let base = text_obj.take().unwrap_or_else(|| json!({}));
        let mut obj = base.as_object().cloned().unwrap_or_default();
        obj.insert("verbosity".into(), Value::String(v));
        text_obj = Some(Value::Object(obj));
    }

    let mut body_map = serde_json::Map::new();
    body_map.insert("model".into(), json!(model_id));
    if let Some(i) = prov.instructions.as_deref() {
        body_map.insert("instructions".into(), json!(i));
    }
    body_map.insert("input".into(), json!(messages));
    if let Some(t) = options.temperature {
        body_map.insert("temperature".into(), json!(t));
    }
    if let Some(tp) = options.top_p {
        body_map.insert("top_p".into(), json!(tp));
    }
    if let Some(mx) = options.max_output_tokens {
        body_map.insert("max_output_tokens".into(), json!(mx));
    }
    let mut body = Value::Object(body_map);
    if let Some(t) = text_obj {
        body["text"] = t;
    }

    if let Some(m) = prov.metadata {
        body["metadata"] = m;
    }
    if let Some(c) = prov.conversation {
        body["conversation"] = json!(c);
    }
    if let Some(n) = prov.max_tool_calls {
        body["max_tool_calls"] = json!(n);
    }
    if let Some(b) = request_tool_settings.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(b);
    }
    if let Some(s) = prov.previous_response_id {
        body["previous_response_id"] = json!(s);
    }
    if let Some(client_metadata) = prov.client_metadata {
        body["client_metadata"] = client_metadata;
    }
    if let Some(b) = store_value {
        body["store"] = json!(b);
    }
    if let Some(u) = prov.user {
        body["user"] = json!(u);
    }
    if let Some(k) = prov.prompt_cache_key {
        body["prompt_cache_key"] = json!(k);
    }
    if let Some(r) = prov.prompt_cache_retention {
        body["prompt_cache_retention"] = json!(r);
    }
    if let Some(s) = prov.safety_identifier {
        body["safety_identifier"] = json!(s);
    }
    if let Some(t) = prov.truncation {
        body["truncation"] = json!(t);
    }

    if !options.tools.is_empty() {
        let mut tools: Vec<Value> = Vec::new();
        for tool in &options.tools {
            match tool {
                v2t::Tool::Function(t) => {
                    let params = normalize_object_schema(&t.input_schema);
                    let mut function_tool = json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": params
                    });
                    if let Some(strict) = function_tool_strict(t) {
                        function_tool["strict"] = json!(strict);
                    }
                    tools.push(function_tool);
                }
                v2t::Tool::Provider(t) => {
                    if let Some(val) = build_openai_provider_tool(t)? {
                        tools.push(val);
                    } else {
                        warnings.push(v2t::CallWarning::UnsupportedTool {
                            tool_name: t.name.clone(),
                            details: Some(format!("unsupported provider tool id {}", t.id)),
                        });
                    }
                }
            }
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
    }

    if let Some(tc) = map_tool_choice(&request_tool_settings.tool_choice, &tool_name_mapping) {
        body["tool_choice"] = tc;
    }
    if let Some(n) = top_logprobs {
        body["top_logprobs"] = json!(n);
    }
    if let Some(incl) = include {
        body["include"] = json!(incl);
    }

    if is_reasoning_model {
        let allow_non_reasoning = prov.reasoning_effort.as_deref() == Some("none")
            && model_cfg.supports_non_reasoning_parameters;
        if !allow_non_reasoning {
            if body.get("temperature").is_some() {
                body.as_object_mut().unwrap().remove("temperature");
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "temperature".into(),
                    details: Some("temperature is not supported for reasoning models".into()),
                });
            }
            if body.get("top_p").is_some() {
                body.as_object_mut().unwrap().remove("top_p");
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "topP".into(),
                    details: Some("topP is not supported for reasoning models".into()),
                });
            }
        }
        if prov.reasoning_effort.is_some() || prov.reasoning_summary.is_some() {
            let mut r = serde_json::Map::new();
            if let Some(e) = prov.reasoning_effort.as_ref() {
                r.insert("effort".into(), Value::String(e.clone()));
            }
            if let Some(s) = prov.reasoning_summary.as_ref() {
                r.insert("summary".into(), Value::String(s.clone()));
            }
            body["reasoning"] = Value::Object(r);
        }
    } else {
        if prov.reasoning_effort.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "reasoningEffort".into(),
                details: Some("reasoningEffort is not supported for non-reasoning models".into()),
            });
        }
        if prov.reasoning_summary.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "reasoningSummary".into(),
                details: Some("reasoningSummary is not supported for non-reasoning models".into()),
            });
        }
    }

    if model_cfg.required_auto_truncation {
        body["truncation"] = json!("auto");
    }

    if let Some(defaults) = cfg.request_defaults.as_ref() {
        if let Some(overrides) = request_overrides_from_json(&cfg.provider_scope_name, defaults) {
            tracing::debug!(
                provider_scope = %cfg.provider_scope_name,
                override_keys = ?json_object_keys(&overrides),
                "openai request defaults resolved"
            );
            let disallow = ["model", "input", "stream", "tools", "tool_choice"];
            merge_options_with_disallow(&mut body, &overrides, &disallow);
            tracing::debug!(
                provider_scope = %cfg.provider_scope_name,
                has_reasoning_effort = body.get("reasoning_effort").is_some(),
                has_reasoning = body.get("reasoning").is_some(),
                has_reasoning_effort_nested = body
                    .get("reasoning")
                    .and_then(|v| v.get("effort"))
                    .is_some(),
                "openai request defaults merged"
            );
            if is_reasoning_model {
                if let Some(explicit_effort) = prov.reasoning_effort.as_ref() {
                    if let Some(body_obj) = body.as_object_mut() {
                        let reasoning = body_obj
                            .entry("reasoning".to_string())
                            .or_insert_with(|| Value::Object(serde_json::Map::new()));
                        if !reasoning.is_object() {
                            *reasoning = Value::Object(serde_json::Map::new());
                        }
                        if let Some(reasoning_obj) = reasoning.as_object_mut() {
                            reasoning_obj.insert(
                                "effort".to_string(),
                                Value::String(explicit_effort.clone()),
                            );
                        }
                    }
                }
            }
        } else {
            tracing::debug!(
                provider_scope = %cfg.provider_scope_name,
                "openai request defaults present but no overrides matched"
            );
        }
    }

    if let Some(tier) = prov.service_tier {
        match tier.as_str() {
            "flex" if !model_cfg.supports_flex_processing => {
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "serviceTier".into(),
                    details: Some(
                        "flex processing is only available for o3, o4-mini, and gpt-5 models"
                            .into(),
                    ),
                });
            }
            "priority" if !model_cfg.supports_priority_processing => {
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "serviceTier".into(),
                    details: Some(
                        "priority processing is only available for supported models and requires Enterprise access"
                            .into(),
                    ),
                });
            }
            _ => {
                body["service_tier"] = json!(tier);
            }
        }
    }

    Ok((body, warnings))
}

fn json_object_keys(value: &Value) -> Vec<String> {
    value
        .as_object()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_default()
}
