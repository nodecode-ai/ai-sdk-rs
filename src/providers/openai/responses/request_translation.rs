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

fn is_non_chat_gpt_5_model(model_id: &str) -> bool {
    model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat")
}

fn supports_flex_processing(model_id: &str) -> bool {
    model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
        || is_non_chat_gpt_5_model(model_id)
}

fn supports_priority_processing(model_id: &str) -> bool {
    model_id.starts_with("gpt-4")
        || model_id.starts_with("gpt-5-mini")
        || (is_non_chat_gpt_5_model(model_id) && !model_id.starts_with("gpt-5-nano"))
        || model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
}

fn is_reasoning_model(model_id: &str) -> bool {
    model_id.starts_with("o1")
        || model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
        || model_id.starts_with("codex-mini")
        || model_id.starts_with("computer-use-preview")
        || is_non_chat_gpt_5_model(model_id)
}

fn system_message_mode_for_model(model_id: &str) -> SystemMessageMode {
    if is_reasoning_model(model_id) {
        SystemMessageMode::Developer
    } else {
        SystemMessageMode::System
    }
}

fn supports_non_reasoning_parameters(model_id: &str) -> bool {
    model_id.starts_with("gpt-5.1") || model_id.starts_with("gpt-5.2")
}

fn get_responses_model_config(model_id: &str) -> ResponsesModelConfig {
    let is_reasoning_model = is_reasoning_model(model_id);

    ResponsesModelConfig {
        is_reasoning_model,
        system_message_mode: system_message_mode_for_model(model_id),
        required_auto_truncation: false,
        supports_flex_processing: supports_flex_processing(model_id),
        supports_priority_processing: supports_priority_processing(model_id),
        supports_non_reasoning_parameters: supports_non_reasoning_parameters(model_id),
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

struct OpenAIMessageTranslationSettings<'a> {
    file_id_prefixes: Option<&'a [String]>,
    provider_scope_name: &'a str,
    store: bool,
    tool_name_mapping: &'a ToolNameMapping,
    has_local_shell_tool: bool,
    has_shell_tool: bool,
    has_apply_patch_tool: bool,
}

#[derive(Default)]
struct OpenAIAssistantMessageState {
    reasoning_message_idx: HashMap<String, usize>,
    reasoning_item_refs: HashSet<String>,
}

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

fn starts_with_file_id_prefix(value: &str, file_id_prefixes: Option<&[String]>) -> bool {
    file_id_prefixes.is_some_and(|prefixes| prefixes.iter().any(|prefix| value.starts_with(prefix)))
}

fn handle_system_prompt_message(
    messages: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    content: &str,
    system_mode: SystemMessageMode,
) {
    match system_mode {
        SystemMessageMode::Remove => warnings.push(v2t::CallWarning::Other {
            message: "system messages are removed for this model".into(),
        }),
        SystemMessageMode::System => messages.push(json!({"role":"system","content": content })),
        SystemMessageMode::Developer => {
            messages.push(json!({"role":"developer","content": content }))
        }
    }
}

fn push_user_file_part(
    parts: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    filename: &Option<String>,
    data: &v2t::DataContent,
    media_type: &str,
    idx: usize,
    file_id_prefixes: Option<&[String]>,
) {
    if media_type.starts_with("image/") {
        let media_type = if media_type == "image/*" {
            "image/jpeg"
        } else {
            media_type
        };
        if let Some(url) = to_data_uri(media_type, data) {
            let is_http = url.starts_with("http://") || url.starts_with("https://");
            if !is_http && starts_with_file_id_prefix(&url, file_id_prefixes) {
                parts.push(json!({"type":"input_image","file_id": url }));
                return;
            }
            parts.push(json!({"type":"input_image","image_url": url }));
        }
        return;
    }

    if media_type == "application/pdf" {
        match data {
            v2t::DataContent::Url { url } => {
                parts.push(json!({"type":"input_file","file_url": url}))
            }
            _ => {
                if let v2t::DataContent::Base64 { base64 } = data {
                    if starts_with_file_id_prefix(base64, file_id_prefixes) {
                        parts.push(json!({"type":"input_file","file_id": base64}));
                        return;
                    }
                }
                let filename = filename
                    .clone()
                    .unwrap_or_else(|| format!("part-{}.pdf", idx));
                if let Some(uri) = to_data_uri("application/pdf", data) {
                    parts.push(json!({"type":"input_file","filename": filename, "file_data": uri}));
                }
            }
        }
        return;
    }

    warnings.push(v2t::CallWarning::Other {
        message: format!("unsupported file media type: {}", media_type),
    });
}

fn handle_user_prompt_message(
    messages: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    content: &[v2t::UserPart],
    file_id_prefixes: Option<&[String]>,
) {
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
            } => push_user_file_part(
                &mut parts,
                warnings,
                filename,
                data,
                media_type,
                idx,
                file_id_prefixes,
            ),
        }
    }
    if !parts.is_empty() {
        messages.push(json!({"role":"user","content": parts}));
    }
}

fn build_local_shell_call_message(
    tc: &v2t::ToolCallPart,
    item_id: Option<String>,
    input_json: &Value,
) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("local_shell_call"));
    obj.insert("call_id".into(), json!(tc.tool_call_id));
    if let Some(id) = item_id {
        obj.insert("id".into(), json!(id));
    }
    let mut action = serde_json::Map::new();
    action.insert("type".into(), json!("exec"));
    if let Some(action_obj) = input_json.get("action").and_then(|v| v.as_object()) {
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
    Value::Object(obj)
}

fn build_shell_call_message(
    tc: &v2t::ToolCallPart,
    item_id: Option<String>,
    input_json: &Value,
) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("shell_call"));
    obj.insert("call_id".into(), json!(tc.tool_call_id));
    if let Some(id) = item_id {
        obj.insert("id".into(), json!(id));
    }
    obj.insert("status".into(), json!("completed"));
    let mut action = serde_json::Map::new();
    if let Some(action_obj) = input_json.get("action").and_then(|v| v.as_object()) {
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
    Value::Object(obj)
}

fn build_apply_patch_call_message(
    tc: &v2t::ToolCallPart,
    item_id: Option<String>,
    input_json: &Value,
) -> Value {
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
    Value::Object(obj)
}

fn build_assistant_tool_call_override(
    tc: &v2t::ToolCallPart,
    settings: &OpenAIMessageTranslationSettings<'_>,
    resolved_tool_name: &str,
    item_id: Option<String>,
    input_json: &Value,
) -> Option<Value> {
    if settings.has_local_shell_tool && resolved_tool_name == "local_shell" {
        return Some(build_local_shell_call_message(tc, item_id, input_json));
    }
    if settings.has_shell_tool && resolved_tool_name == "shell" {
        return Some(build_shell_call_message(tc, item_id, input_json));
    }
    if settings.has_apply_patch_tool && resolved_tool_name == "apply_patch" {
        return Some(build_apply_patch_call_message(tc, item_id, input_json));
    }
    None
}

fn handle_assistant_text_part(
    messages: &mut Vec<Value>,
    text: &str,
    provider_options: &Option<v2t::ProviderOptions>,
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    let item_id =
        openai_item_id_from_provider_options(provider_options, settings.provider_scope_name);
    if settings.store && item_id.is_some() {
        messages.push(json!({"type":"item_reference","id": item_id}));
        return;
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

fn handle_assistant_tool_call_part(
    messages: &mut Vec<Value>,
    tc: &v2t::ToolCallPart,
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    let item_id =
        openai_item_id_from_provider_options(&tc.provider_options, settings.provider_scope_name);
    if tc.provider_executed {
        if settings.store && item_id.is_some() {
            messages.push(json!({"type":"item_reference","id": item_id}));
        }
        return;
    }
    if settings.store && item_id.is_some() {
        messages.push(json!({"type":"item_reference","id": item_id}));
        return;
    }

    let resolved_tool_name = settings
        .tool_name_mapping
        .to_provider_tool_name(&tc.tool_name);
    let input_json: serde_json::Value = serde_json::from_str(&tc.input).unwrap_or(json!({}));
    if let Some(message) = build_assistant_tool_call_override(
        tc,
        settings,
        resolved_tool_name,
        item_id.clone(),
        &input_json,
    ) {
        messages.push(message);
        return;
    }

    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("function_call"));
    obj.insert("call_id".into(), json!(tc.tool_call_id));
    obj.insert(
        "name".into(),
        json!(settings
            .tool_name_mapping
            .to_provider_tool_name(&tc.tool_name)),
    );
    obj.insert("arguments".into(), json!(tc.input));
    if let Some(id) = item_id {
        obj.insert("id".into(), json!(id));
    }
    messages.push(Value::Object(obj));
}

fn append_reasoning_summary_entry(
    messages: &mut [Value],
    warnings: &mut Vec<v2t::CallWarning>,
    reasoning_id: &str,
    encrypted: Option<String>,
    summary_entry: Option<Value>,
    reasoning_message_idx: &HashMap<String, usize>,
) {
    let Some(entry) = summary_entry else {
        warnings.push(v2t::CallWarning::Other {
            message: format!(
                "Cannot append empty reasoning part to existing reasoning sequence. Skipping reasoning part: {}",
                reasoning_id
            ),
        });
        return;
    };

    let Some(idx) = reasoning_message_idx.get(reasoning_id).copied() else {
        return;
    };
    let Some(obj) = messages
        .get_mut(idx)
        .and_then(|value: &mut Value| value.as_object_mut())
    else {
        return;
    };
    if let Some(summary) = obj
        .get_mut("summary")
        .and_then(|value: &mut Value| value.as_array_mut())
    {
        summary.push(entry);
    }
    if let Some(enc) = encrypted {
        obj.insert("encrypted_content".into(), json!(enc));
    }
}

fn handle_assistant_reasoning_part(
    messages: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    assistant_state: &mut OpenAIAssistantMessageState,
    text: &str,
    provider_options: &Option<v2t::ProviderOptions>,
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    let item_id =
        openai_item_id_from_provider_options(provider_options, settings.provider_scope_name);
    let encrypted = openai_reasoning_encrypted_content_from_provider_options(
        provider_options,
        settings.provider_scope_name,
    );
    let Some(reasoning_id) = item_id else {
        warnings.push(v2t::CallWarning::Other {
            message: format!(
                "Non-OpenAI reasoning parts are not supported. Skipping reasoning part: {}",
                text
            ),
        });
        return;
    };

    if settings.store {
        if assistant_state
            .reasoning_item_refs
            .insert(reasoning_id.clone())
        {
            messages.push(json!({"type":"item_reference","id": reasoning_id}));
        }
        return;
    }

    let summary_entry = (!text.is_empty()).then(|| json!({"type":"summary_text","text": text}));
    if assistant_state
        .reasoning_message_idx
        .contains_key(&reasoning_id)
    {
        append_reasoning_summary_entry(
            messages,
            warnings,
            &reasoning_id,
            encrypted,
            summary_entry,
            &assistant_state.reasoning_message_idx,
        );
        return;
    }

    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("reasoning"));
    obj.insert("id".into(), json!(reasoning_id.clone()));
    obj.insert(
        "summary".into(),
        summary_entry
            .map(|entry| Value::Array(vec![entry]))
            .unwrap_or_else(|| Value::Array(Vec::new())),
    );
    if let Some(enc) = encrypted {
        obj.insert("encrypted_content".into(), json!(enc));
    }
    messages.push(Value::Object(obj));
    assistant_state
        .reasoning_message_idx
        .insert(reasoning_id, messages.len() - 1);
}

fn handle_assistant_tool_result_part(
    messages: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    tr: &v2t::ToolResultPart,
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    if settings.store {
        let item_id = openai_item_id_from_provider_options(
            &tr.provider_options,
            settings.provider_scope_name,
        )
        .unwrap_or_else(|| tr.tool_call_id.clone());
        messages.push(json!({"type":"item_reference","id": item_id}));
        return;
    }

    warnings.push(v2t::CallWarning::Other {
        message: format!(
            "Results for OpenAI tool {} are not sent to the API when store is false",
            tr.tool_name
        ),
    });
}

fn handle_assistant_prompt_message(
    messages: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    content: &[v2t::AssistantPart],
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    let mut assistant_state = OpenAIAssistantMessageState::default();
    for part in content {
        match part {
            v2t::AssistantPart::Text {
                text,
                provider_options,
            } => handle_assistant_text_part(messages, text, provider_options, settings),
            v2t::AssistantPart::ToolCall(tc) => {
                handle_assistant_tool_call_part(messages, tc, settings)
            }
            v2t::AssistantPart::ToolResult(tr) => {
                handle_assistant_tool_result_part(messages, warnings, tr, settings)
            }
            v2t::AssistantPart::Reasoning {
                text,
                provider_options,
            } => handle_assistant_reasoning_part(
                messages,
                warnings,
                &mut assistant_state,
                text,
                provider_options,
                settings,
            ),
            v2t::AssistantPart::File { .. } => {}
        }
    }
}

fn build_shell_call_output_message(tool_call_id: &str, output: &[Value]) -> Option<Value> {
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
    Some(json!({
        "type":"shell_call_output",
        "call_id": tool_call_id,
        "output": mapped,
    }))
}

fn build_tool_result_override(
    tr: &v2t::ToolResultPart,
    settings: &OpenAIMessageTranslationSettings<'_>,
    resolved_tool_name: &str,
) -> Option<Value> {
    if settings.has_local_shell_tool && resolved_tool_name == "local_shell" {
        if let v2t::ToolResultOutput::Json { value } = &tr.output {
            if let Some(output) = value.as_object().and_then(|obj| obj.get("output")) {
                return Some(json!({
                    "type":"local_shell_call_output",
                    "call_id": tr.tool_call_id,
                    "output": output.clone(),
                }));
            }
        }
    }

    if settings.has_shell_tool && resolved_tool_name == "shell" {
        if let v2t::ToolResultOutput::Json { value } = &tr.output {
            if let Some(output) = value
                .as_object()
                .and_then(|obj| obj.get("output"))
                .and_then(|value| value.as_array())
            {
                return build_shell_call_output_message(&tr.tool_call_id, output);
            }
        }
    }

    if settings.has_apply_patch_tool && resolved_tool_name == "apply_patch" {
        if let v2t::ToolResultOutput::Json { value } = &tr.output {
            if let Some(obj) = value.as_object() {
                if let Some(status) = obj.get("status") {
                    let mut out = serde_json::Map::new();
                    out.insert("type".into(), json!("apply_patch_call_output"));
                    out.insert("call_id".into(), json!(tr.tool_call_id));
                    out.insert("status".into(), status.clone());
                    if let Some(output) = obj.get("output") {
                        out.insert("output".into(), output.clone());
                    }
                    return Some(Value::Object(out));
                }
            }
        }
    }

    None
}

fn handle_tool_result_message(
    messages: &mut Vec<Value>,
    tr: &v2t::ToolResultPart,
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    let resolved_tool_name = settings
        .tool_name_mapping
        .to_provider_tool_name(&tr.tool_name);
    if let Some(message) = build_tool_result_override(tr, settings, resolved_tool_name) {
        messages.push(message);
        return;
    }

    let out_val = tool_output_to_value(&tr.output);
    messages.push(json!({
        "type":"function_call_output",
        "call_id": tr.tool_call_id,
        "output": out_val,
    }));
}

fn handle_tool_prompt_message(
    messages: &mut Vec<Value>,
    processed_approval_ids: &mut HashSet<String>,
    content: &[v2t::ToolMessagePart],
    settings: &OpenAIMessageTranslationSettings<'_>,
) {
    for part in content {
        match part {
            v2t::ToolMessagePart::ToolApprovalResponse(resp) => {
                if !processed_approval_ids.insert(resp.approval_id.clone()) {
                    continue;
                }
                if settings.store {
                    messages.push(json!({"type":"item_reference","id": resp.approval_id}));
                }
                messages.push(json!({
                    "type":"mcp_approval_response",
                    "approval_request_id": resp.approval_id,
                    "approve": resp.approved,
                }));
            }
            v2t::ToolMessagePart::ToolResult(tr) => {
                handle_tool_result_message(messages, tr, settings)
            }
        }
    }
}

fn handle_prompt_message(
    messages: &mut Vec<Value>,
    warnings: &mut Vec<v2t::CallWarning>,
    processed_approval_ids: &mut HashSet<String>,
    msg: &v2t::PromptMessage,
    settings: &OpenAIMessageTranslationSettings<'_>,
    system_mode: SystemMessageMode,
) {
    match msg {
        v2t::PromptMessage::System { content, .. } => {
            handle_system_prompt_message(messages, warnings, content, system_mode)
        }
        v2t::PromptMessage::User { content, .. } => {
            handle_user_prompt_message(messages, warnings, content, settings.file_id_prefixes)
        }
        v2t::PromptMessage::Assistant { content, .. } => {
            handle_assistant_prompt_message(messages, warnings, content, settings)
        }
        v2t::PromptMessage::Tool { content, .. } => {
            handle_tool_prompt_message(messages, processed_approval_ids, content, settings)
        }
    }
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
    let settings = OpenAIMessageTranslationSettings {
        file_id_prefixes,
        provider_scope_name,
        store,
        tool_name_mapping,
        has_local_shell_tool,
        has_shell_tool,
        has_apply_patch_tool,
    };

    for msg in prompt {
        handle_prompt_message(
            &mut messages,
            &mut warnings,
            &mut processed_approval_ids,
            msg,
            &settings,
            system_mode,
        );
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

#[derive(Default)]
struct OpenAIResponseToolPresence {
    has_local_shell_tool: bool,
    has_shell_tool: bool,
    has_apply_patch_tool: bool,
    has_web_search_tool: bool,
    has_code_interpreter_tool: bool,
}

fn push_unsupported_responses_option_warning(warnings: &mut Vec<v2t::CallWarning>, setting: &str) {
    warnings.push(v2t::CallWarning::UnsupportedSetting {
        setting: setting.into(),
        details: None,
    });
}

fn collect_unsupported_responses_option_warnings(
    options: &v2t::CallOptions,
    warnings: &mut Vec<v2t::CallWarning>,
) {
    if options.top_k.is_some() {
        push_unsupported_responses_option_warning(warnings, "topK");
    }
    if options.seed.is_some() {
        push_unsupported_responses_option_warning(warnings, "seed");
    }
    if options.presence_penalty.is_some() {
        push_unsupported_responses_option_warning(warnings, "presencePenalty");
    }
    if options.frequency_penalty.is_some() {
        push_unsupported_responses_option_warning(warnings, "frequencyPenalty");
    }
    if options.stop_sequences.is_some() {
        push_unsupported_responses_option_warning(warnings, "stopSequences");
    }
}

fn detect_openai_response_tool_presence(tools: &[v2t::Tool]) -> OpenAIResponseToolPresence {
    let mut presence = OpenAIResponseToolPresence::default();
    for tool in tools {
        let v2t::Tool::Provider(tool) = tool else {
            continue;
        };
        match tool.id.as_str() {
            "openai.local_shell" => presence.has_local_shell_tool = true,
            "openai.shell" => presence.has_shell_tool = true,
            "openai.apply_patch" => presence.has_apply_patch_tool = true,
            "openai.web_search" | "openai.web_search_preview" => {
                presence.has_web_search_tool = true
            }
            "openai.code_interpreter" => presence.has_code_interpreter_tool = true,
            _ => {}
        }
    }
    presence
}

fn append_unique_include(include: &mut Option<Vec<String>>, key: &str) {
    if let Some(list) = include.as_mut() {
        if !list.iter().any(|existing| existing == key) {
            list.push(key.to_string());
        }
    } else {
        *include = Some(vec![key.to_string()]);
    }
}

fn build_responses_include(
    prov: &OpenAIProviderOptionsParsed,
    tool_presence: &OpenAIResponseToolPresence,
    is_reasoning_model: bool,
) -> Option<Vec<String>> {
    let mut include = prov.include.clone();
    let logprobs_requested = prov.logprobs_bool == Some(true) || prov.logprobs_n.unwrap_or(0) > 0;
    if logprobs_requested {
        append_unique_include(&mut include, "message.output_text.logprobs");
    }
    if tool_presence.has_web_search_tool {
        append_unique_include(&mut include, "web_search_call.action.sources");
    }
    if tool_presence.has_code_interpreter_tool {
        append_unique_include(&mut include, "code_interpreter_call.outputs");
    }
    if prov.store == Some(false) && is_reasoning_model {
        append_unique_include(&mut include, "reasoning.encrypted_content");
    }
    include
}

fn build_responses_text_object(
    options: &v2t::CallOptions,
    prov: &OpenAIProviderOptionsParsed,
) -> Option<Value> {
    let mut text_obj: Option<Value> = None;
    if let Some(v2t::ResponseFormat::Json {
        schema,
        name,
        description,
    }) = &options.response_format
    {
        let mut format_obj = json!({"type":"json_object"});
        if let Some(schema) = schema {
            format_obj = json!({
                "type": "json_schema",
                "strict": prov.strict_json_schema.unwrap_or(true),
                "name": name.clone().unwrap_or_else(|| "response".into()),
                "description": description.clone(),
                "schema": schema,
            });
        }
        text_obj = Some(json!({"format": format_obj}));
    }
    if let Some(verbosity) = prov.text_verbosity.as_ref() {
        let base = text_obj.take().unwrap_or_else(|| json!({}));
        let mut obj = base.as_object().cloned().unwrap_or_default();
        obj.insert("verbosity".into(), Value::String(verbosity.clone()));
        text_obj = Some(Value::Object(obj));
    }
    text_obj
}

fn build_responses_tools(
    options: &v2t::CallOptions,
    warnings: &mut Vec<v2t::CallWarning>,
) -> Result<Vec<Value>, SdkError> {
    let mut tools = Vec::new();
    for tool in &options.tools {
        match tool {
            v2t::Tool::Function(tool) => {
                let params = normalize_object_schema(&tool.input_schema);
                let mut function_tool = json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params
                });
                if let Some(strict) = function_tool_strict(tool) {
                    function_tool["strict"] = json!(strict);
                }
                tools.push(function_tool);
            }
            v2t::Tool::Provider(tool) => {
                if let Some(value) = build_openai_provider_tool(tool)? {
                    tools.push(value);
                } else {
                    warnings.push(v2t::CallWarning::UnsupportedTool {
                        tool_name: tool.name.clone(),
                        details: Some(format!("unsupported provider tool id {}", tool.id)),
                    });
                }
            }
        }
    }
    Ok(tools)
}

fn apply_responses_base_body(
    body: &mut Value,
    prov: &OpenAIProviderOptionsParsed,
    request_tool_settings: &OpenAIRequestToolSettings,
    options: &v2t::CallOptions,
) {
    if let Some(metadata) = prov.metadata.as_ref() {
        body["metadata"] = metadata.clone();
    }
    if let Some(conversation) = prov.conversation.as_ref() {
        body["conversation"] = json!(conversation);
    }
    if let Some(max_tool_calls) = prov.max_tool_calls {
        body["max_tool_calls"] = json!(max_tool_calls);
    }
    if let Some(parallel_tool_calls) = request_tool_settings.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(parallel_tool_calls);
    }
    if let Some(previous_response_id) = prov.previous_response_id.as_ref() {
        body["previous_response_id"] = json!(previous_response_id);
    }
    if let Some(client_metadata) = prov.client_metadata.as_ref() {
        body["client_metadata"] = client_metadata.clone();
    }
    if let Some(store) = prov.store {
        body["store"] = json!(store);
    }
    if let Some(user) = prov.user.as_ref() {
        body["user"] = json!(user);
    }
    if let Some(prompt_cache_key) = prov.prompt_cache_key.as_ref() {
        body["prompt_cache_key"] = json!(prompt_cache_key);
    }
    if let Some(prompt_cache_retention) = prov.prompt_cache_retention.as_ref() {
        body["prompt_cache_retention"] = json!(prompt_cache_retention);
    }
    if let Some(safety_identifier) = prov.safety_identifier.as_ref() {
        body["safety_identifier"] = json!(safety_identifier);
    }
    if let Some(truncation) = prov.truncation.as_ref() {
        body["truncation"] = json!(truncation);
    }
    if let Some(tool_choice) = map_tool_choice(
        &request_tool_settings.tool_choice,
        &build_tool_name_mapping(&options.tools),
    ) {
        body["tool_choice"] = tool_choice;
    }
}

fn apply_reasoning_model_settings(
    body: &mut Value,
    warnings: &mut Vec<v2t::CallWarning>,
    prov: &OpenAIProviderOptionsParsed,
    model_cfg: &ResponsesModelConfig,
    is_reasoning_model: bool,
) {
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
            let mut reasoning = serde_json::Map::new();
            if let Some(effort) = prov.reasoning_effort.as_ref() {
                reasoning.insert("effort".into(), Value::String(effort.clone()));
            }
            if let Some(summary) = prov.reasoning_summary.as_ref() {
                reasoning.insert("summary".into(), Value::String(summary.clone()));
            }
            body["reasoning"] = Value::Object(reasoning);
        }
        return;
    }

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

fn merge_openai_request_defaults(
    body: &mut Value,
    cfg: &OpenAIConfig,
    prov: &OpenAIProviderOptionsParsed,
    is_reasoning_model: bool,
) {
    let Some(defaults) = cfg.request_defaults.as_ref() else {
        return;
    };
    let Some(overrides) = request_overrides_from_json(&cfg.provider_scope_name, defaults) else {
        tracing::debug!(
            provider_scope = %cfg.provider_scope_name,
            "openai request defaults present but no overrides matched"
        );
        return;
    };

    tracing::debug!(
        provider_scope = %cfg.provider_scope_name,
        override_keys = ?json_object_keys(&overrides),
        "openai request defaults resolved"
    );
    let disallow = ["model", "input", "stream", "tools", "tool_choice"];
    merge_options_with_disallow(body, &overrides, &disallow);
    tracing::debug!(
        provider_scope = %cfg.provider_scope_name,
        has_reasoning_effort = body.get("reasoning_effort").is_some(),
        has_reasoning = body.get("reasoning").is_some(),
        has_reasoning_effort_nested = body
            .get("reasoning")
            .and_then(|value| value.get("effort"))
            .is_some(),
        "openai request defaults merged"
    );
    if !is_reasoning_model {
        return;
    }
    let Some(explicit_effort) = prov.reasoning_effort.as_ref() else {
        return;
    };
    let Some(body_obj) = body.as_object_mut() else {
        return;
    };
    let reasoning = body_obj
        .entry("reasoning".to_string())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if !reasoning.is_object() {
        *reasoning = Value::Object(serde_json::Map::new());
    }
    if let Some(reasoning_obj) = reasoning.as_object_mut() {
        reasoning_obj.insert("effort".to_string(), Value::String(explicit_effort.clone()));
    }
}

fn apply_service_tier(
    body: &mut Value,
    warnings: &mut Vec<v2t::CallWarning>,
    model_cfg: &ResponsesModelConfig,
    prov: &OpenAIProviderOptionsParsed,
) {
    let Some(service_tier) = prov.service_tier.as_ref() else {
        return;
    };
    match service_tier.as_str() {
        "flex" if !model_cfg.supports_flex_processing => {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "serviceTier".into(),
                details: Some(
                    "flex processing is only available for o3, o4-mini, and gpt-5 models".into(),
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
            body["service_tier"] = json!(service_tier);
        }
    }
}

struct OpenAIRequestBuildState {
    prov: OpenAIProviderOptionsParsed,
    request_tool_settings: OpenAIRequestToolSettings,
    model_cfg: ResponsesModelConfig,
    is_reasoning_model: bool,
    system_message_mode: SystemMessageMode,
    tool_name_mapping: ToolNameMapping,
    tool_presence: OpenAIResponseToolPresence,
}

fn build_request_state(
    options: &v2t::CallOptions,
    model_id: &str,
    cfg: &OpenAIConfig,
) -> OpenAIRequestBuildState {
    let prov = parse_openai_provider_options(&options.provider_options, &cfg.provider_scope_name);
    let request_tool_settings =
        resolve_request_tool_settings(&cfg.endpoint_path, &prov, &options.tool_choice);
    let model_cfg = get_responses_model_config(model_id);
    let is_reasoning_model = prov.force_reasoning.unwrap_or(model_cfg.is_reasoning_model);
    let system_message_mode = prov.system_message_mode.unwrap_or_else(|| {
        if is_reasoning_model {
            SystemMessageMode::Developer
        } else {
            model_cfg.system_message_mode
        }
    });
    let tool_name_mapping = build_tool_name_mapping(&options.tools);
    let tool_presence = detect_openai_response_tool_presence(&options.tools);

    OpenAIRequestBuildState {
        prov,
        request_tool_settings,
        model_cfg,
        is_reasoning_model,
        system_message_mode,
        tool_name_mapping,
        tool_presence,
    }
}

fn build_request_input_messages(
    options: &v2t::CallOptions,
    cfg: &OpenAIConfig,
    state: &OpenAIRequestBuildState,
) -> (Vec<Value>, Vec<v2t::CallWarning>) {
    convert_to_openai_messages(
        &options.prompt,
        state.system_message_mode,
        cfg.file_id_prefixes.as_deref(),
        &cfg.provider_scope_name,
        state.prov.store.unwrap_or(true),
        &state.tool_name_mapping,
        state.tool_presence.has_local_shell_tool,
        state.tool_presence.has_shell_tool,
        state.tool_presence.has_apply_patch_tool,
    )
}

fn build_initial_request_body(
    model_id: &str,
    instructions: Option<&str>,
    messages: Vec<Value>,
    options: &v2t::CallOptions,
) -> Value {
    let mut body_map = serde_json::Map::new();
    body_map.insert("model".into(), json!(model_id));
    if let Some(instructions) = instructions {
        body_map.insert("instructions".into(), json!(instructions));
    }
    body_map.insert("input".into(), json!(messages));
    if let Some(temperature) = options.temperature {
        body_map.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = options.top_p {
        body_map.insert("top_p".into(), json!(top_p));
    }
    if let Some(max_output_tokens) = options.max_output_tokens {
        body_map.insert("max_output_tokens".into(), json!(max_output_tokens));
    }
    Value::Object(body_map)
}

pub(super) fn build_request_body(
    options: &v2t::CallOptions,
    model_id: &str,
    cfg: &OpenAIConfig,
) -> Result<(Value, Vec<v2t::CallWarning>), SdkError> {
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();
    collect_unsupported_responses_option_warnings(options, &mut warnings);

    let state = build_request_state(options, model_id, cfg);
    let (messages, mut message_warnings) = build_request_input_messages(options, cfg, &state);
    warnings.append(&mut message_warnings);

    if state.prov.conversation.is_some() && state.prov.previous_response_id.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "conversation".into(),
            details: Some("conversation and previousResponseId cannot be used together".into()),
        });
    }
    let top_logprobs = state
        .prov
        .logprobs_n
        .or_else(|| (state.prov.logprobs_bool == Some(true)).then_some(TOP_LOGPROBS_MAX));
    let include =
        build_responses_include(&state.prov, &state.tool_presence, state.is_reasoning_model);
    let text_obj = build_responses_text_object(options, &state.prov);

    let mut body = build_initial_request_body(
        model_id,
        state.prov.instructions.as_deref(),
        messages,
        options,
    );
    if let Some(t) = text_obj {
        body["text"] = t;
    }
    apply_responses_base_body(
        &mut body,
        &state.prov,
        &state.request_tool_settings,
        options,
    );

    if !options.tools.is_empty() {
        let tools = build_responses_tools(options, &mut warnings)?;
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
    }

    if let Some(n) = top_logprobs {
        body["top_logprobs"] = json!(n);
    }
    if let Some(incl) = include {
        body["include"] = json!(incl);
    }

    apply_reasoning_model_settings(
        &mut body,
        &mut warnings,
        &state.prov,
        &state.model_cfg,
        state.is_reasoning_model,
    );

    if state.model_cfg.required_auto_truncation {
        body["truncation"] = json!("auto");
    }

    merge_openai_request_defaults(&mut body, cfg, &state.prov, state.is_reasoning_model);
    apply_service_tier(&mut body, &mut warnings, &state.model_cfg, &state.prov);

    Ok((body, warnings))
}

fn json_object_keys(value: &Value) -> Vec<String> {
    value
        .as_object()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_default()
}
