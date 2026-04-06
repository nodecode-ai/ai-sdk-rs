use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use base64::Engine;
use futures_util::StreamExt;
use serde_json::{json, Value as JsonValue};

use crate::ai_sdk_core::options;
use crate::ai_sdk_core::stream_collect::{collect_stream_to_response, StreamCollectorConfig};
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{
    map_events_to_parts, EventMapperConfig, EventMapperHooks, LanguageModel, SdkError,
};
use crate::ai_sdk_streaming_sse::{sse_to_events, ProviderChunk, SseEvent};
use crate::ai_sdk_types::v2 as v2t;
use crate::ai_sdk_types::Event as ProviderEvent;

use crate::provider_anthropic::error::map_transport_error_to_sdk_error;
use crate::provider_anthropic::messages::options::{
    parse_anthropic_file_part_options, parse_anthropic_provider_options, AnthropicMessagesModelId,
    AnthropicProviderOptions, ThinkingOption,
};

const TRACE_PREFIX: &str = "[ANTHROPIC-V2]";
const REQ_TRACE_PREFIX: &str = "[REQTRACE]";

/// Configuration for the Anthropic Messages model.
pub struct AnthropicMessagesConfig<T: HttpTransport> {
    pub provider_name: &'static str,
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub default_options: Option<v2t::ProviderOptions>,
}

pub struct AnthropicMessagesLanguageModel<
    T: HttpTransport = crate::reqwest_transport::ReqwestTransport,
> {
    model_id: AnthropicMessagesModelId,
    cfg: AnthropicMessagesConfig<T>,
}

struct BuiltAnthropicRequest {
    body: JsonValue,
    warnings: Vec<v2t::CallWarning>,
    betas: HashSet<String>,
    uses_json_response_tool: bool,
}

enum PromptBlock<'a> {
    System(Vec<&'a v2t::PromptMessage>),
    Assistant(Vec<&'a v2t::PromptMessage>),
    User(Vec<&'a v2t::PromptMessage>),
    Tool(Vec<&'a v2t::PromptMessage>),
}

struct PromptPayloadBuild {
    system: Option<Vec<JsonValue>>,
    messages: Vec<JsonValue>,
    missing_thinking_reasoning: bool,
}

fn collect_unsupported_option_warnings(
    options: &v2t::CallOptions,
    warnings: &mut Vec<v2t::CallWarning>,
) {
    if options.frequency_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "frequencyPenalty".into(),
            details: None,
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "presencePenalty".into(),
            details: None,
        });
    }
    if options.seed.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "seed".into(),
            details: None,
        });
    }
}

fn build_json_response_tool(
    options: &v2t::CallOptions,
    warnings: &mut Vec<v2t::CallWarning>,
) -> Option<v2t::FunctionTool> {
    let Some(v2t::ResponseFormat::Json { schema, .. }) = &options.response_format else {
        return None;
    };
    match schema {
        Some(schema) => Some(v2t::FunctionTool {
            r#type: v2t::FunctionToolType::Function,
            name: "json".into(),
            description: Some("Respond with a JSON object.".into()),
            input_schema: schema.clone(),
            strict: None,
            provider_options: None,
        }),
        None => {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "responseFormat".into(),
                details: Some(
                    "JSON response format requires a schema. The response format is ignored."
                        .into(),
                ),
            });
            None
        }
    }
}

fn get_cache_control(opts: &Option<v2t::ProviderOptions>) -> Option<JsonValue> {
    let map = opts.as_ref()?.get("anthropic")?;
    map.get("cacheControl")
        .or_else(|| map.get("cache_control"))
        .cloned()
}

fn push_system_text_entry(
    entries: &mut Vec<JsonValue>,
    content: &str,
    cache_control: Option<JsonValue>,
) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    if let Some(last) = entries.last_mut() {
        let last_cache_control = last.get("cache_control").cloned();
        if last_cache_control == cache_control {
            if let Some(last_text) = last
                .as_object()
                .and_then(|obj| obj.get("text"))
                .and_then(JsonValue::as_str)
                .map(str::to_string)
            {
                if let Some(obj) = last.as_object_mut() {
                    obj.insert(
                        "text".into(),
                        JsonValue::String(format!("{last_text}\n\n{trimmed}")),
                    );
                }
                return;
            }
        }
    }
    let mut obj = json!({"type":"text","text": trimmed});
    if let Some(cc) = cache_control {
        obj.as_object_mut()
            .expect("system text entry should be an object")
            .insert("cache_control".into(), cc);
    }
    entries.push(obj);
}

fn to_base64(data: &v2t::DataContent) -> Option<String> {
    match data {
        v2t::DataContent::Base64 { base64 } => Some(base64.clone()),
        v2t::DataContent::Bytes { bytes } => {
            Some(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        v2t::DataContent::Url { .. } => None,
    }
}

type ReasoningMetadata = (Option<String>, Option<String>);

fn reasoning_metadata_from_scope(scope: &HashMap<String, JsonValue>) -> ReasoningMetadata {
    let signature = scope
        .get("signature")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let redacted = scope
        .get("redactedData")
        .or_else(|| scope.get("redacted_data"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    (signature, redacted)
}

fn has_reasoning_metadata(metadata: &ReasoningMetadata) -> bool {
    metadata.0.is_some() || metadata.1.is_some()
}

fn scope_keys(scope: &HashMap<String, JsonValue>) -> Vec<&str> {
    scope.keys().map(|k| k.as_str()).collect()
}

fn log_reasoning_metadata(
    scope_kind: &str,
    scope_key: Option<&str>,
    metadata: &ReasoningMetadata,
    keys: Vec<&str>,
) {
    match scope_key {
        Some(key) => tracing::info!(
            "[THINK_DIAG]: extract_reasoning_metadata scope={} key={} signature_present={} redacted_present={} keys={:?}",
            scope_kind,
            key,
            metadata.0.is_some(),
            metadata.1.is_some(),
            keys
        ),
        None => tracing::info!(
            "[THINK_DIAG]: extract_reasoning_metadata scope={} signature_present={} redacted_present={} keys={:?}",
            scope_kind,
            metadata.0.is_some(),
            metadata.1.is_some(),
            keys
        ),
    }
}

fn find_reasoning_metadata_in_scope(
    scope_kind: &str,
    scope_key: Option<&str>,
    scope: &HashMap<String, JsonValue>,
) -> Option<ReasoningMetadata> {
    let metadata = reasoning_metadata_from_scope(scope);
    if !has_reasoning_metadata(&metadata) {
        return None;
    }
    log_reasoning_metadata(scope_kind, scope_key, &metadata, scope_keys(scope));
    Some(metadata)
}

fn extract_reasoning_metadata(opts: &Option<v2t::ProviderOptions>) -> ReasoningMetadata {
    let Some(map) = opts.as_ref() else {
        return (None, None);
    };
    if let Some(scope) = map.get("anthropic") {
        if let Some(metadata) = find_reasoning_metadata_in_scope("primary", None, scope) {
            return metadata;
        }
    }
    for (key, scope) in map {
        if key.eq_ignore_ascii_case("anthropic") || key.to_ascii_lowercase().contains("anthropic") {
            if let Some(metadata) = find_reasoning_metadata_in_scope("alias", Some(key), scope) {
                return metadata;
            }
        }
    }
    for scope in map.values() {
        if let Some(metadata) = find_reasoning_metadata_in_scope("fallback", None, scope) {
            return metadata;
        }
    }
    let provider_keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
    tracing::info!(
        "[THINK_DIAG]: extract_reasoning_metadata scope=missing signature_present=false redacted_present=false keys={:?}",
        provider_keys
    );
    (None, None)
}

fn extract_persisted_reasoning(
    opts: &Option<v2t::ProviderOptions>,
) -> Option<(String, Option<String>)> {
    let map = opts.as_ref()?;
    let mut scope = map.get("anthropic");
    if scope.is_none() {
        if let Some(key) = map.keys().find(|k| k.eq_ignore_ascii_case("anthropic")) {
            scope = map.get(key);
        }
    }
    let scope = scope?;
    let text = scope
        .get("persistedReasoningText")
        .or_else(|| scope.get("persisted_reasoning_text"))
        .and_then(|v| v.as_str())?
        .trim()
        .to_string();
    if text.is_empty() {
        return None;
    }
    let signature = scope
        .get("persistedReasoningSignature")
        .or_else(|| scope.get("persisted_reasoning_signature"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    Some((text, signature))
}

fn prompt_message_tag(message: &v2t::PromptMessage) -> u8 {
    match message {
        v2t::PromptMessage::System { .. } => 0,
        v2t::PromptMessage::Assistant { .. } => 1,
        v2t::PromptMessage::User { .. } => 2,
        v2t::PromptMessage::Tool { .. } => 3,
    }
}

fn push_prompt_block<'a>(
    blocks: &mut Vec<PromptBlock<'a>>,
    tag: u8,
    messages: Vec<&'a v2t::PromptMessage>,
) {
    match tag {
        0 => blocks.push(PromptBlock::System(messages)),
        1 => blocks.push(PromptBlock::Assistant(messages)),
        2 => blocks.push(PromptBlock::User(messages)),
        _ => blocks.push(PromptBlock::Tool(messages)),
    }
}

fn group_prompt_blocks(prompt: &[v2t::PromptMessage]) -> Vec<PromptBlock<'_>> {
    let mut blocks = Vec::new();
    let mut current_tag: Option<u8> = None;
    let mut current_messages = Vec::new();

    for message in prompt {
        let tag = prompt_message_tag(message);
        match current_tag {
            Some(existing) if existing == tag => current_messages.push(message),
            Some(existing) => {
                push_prompt_block(&mut blocks, existing, std::mem::take(&mut current_messages));
                current_messages.push(message);
                current_tag = Some(tag);
            }
            None => {
                current_messages.push(message);
                current_tag = Some(tag);
            }
        }
    }

    if let Some(tag) = current_tag {
        push_prompt_block(&mut blocks, tag, current_messages);
    }

    blocks
}

fn build_system_entries(messages: &[&v2t::PromptMessage]) -> Vec<JsonValue> {
    let mut entries = Vec::new();
    for message in messages {
        if let v2t::PromptMessage::System {
            content,
            provider_options,
        } = message
        {
            if content.is_empty() {
                continue;
            }
            push_system_text_entry(&mut entries, content, get_cache_control(provider_options));
        }
    }
    entries
}

fn user_part_cache_control(
    part: &v2t::UserPart,
    is_last: bool,
    provider_options: &Option<v2t::ProviderOptions>,
) -> Option<JsonValue> {
    let part_cache_control = match part {
        v2t::UserPart::Text {
            provider_options, ..
        }
        | v2t::UserPart::File {
            provider_options, ..
        } => get_cache_control(provider_options),
    };
    let message_cache_control = if is_last {
        get_cache_control(provider_options)
    } else {
        None
    };
    part_cache_control.or(message_cache_control)
}

fn build_image_source(data: &v2t::DataContent, media_type: &str) -> JsonValue {
    match data {
        v2t::DataContent::Url { url } => json!({"type":"url","url": url}),
        _ => json!({
            "type": "base64",
            "media_type": if media_type == "image/*" {
                "image/jpeg"
            } else {
                media_type
            },
            "data": to_base64(data).unwrap_or_default(),
        }),
    }
}

fn build_pdf_source(data: &v2t::DataContent) -> JsonValue {
    match data {
        v2t::DataContent::Url { url } => json!({"type":"url","url": url}),
        _ => json!({
            "type": "base64",
            "media_type": "application/pdf",
            "data": to_base64(data).unwrap_or_default(),
        }),
    }
}

fn build_text_source(data: &v2t::DataContent) -> JsonValue {
    match data {
        v2t::DataContent::Url { url } => json!({"type":"url","url": url}),
        v2t::DataContent::Base64 { base64 } => json!({
            "type": "text",
            "media_type": "text/plain",
            "data": String::from_utf8(
                base64::engine::general_purpose::STANDARD.decode(base64).unwrap_or_default(),
            )
            .unwrap_or_default(),
        }),
        v2t::DataContent::Bytes { bytes } => json!({
            "type": "text",
            "media_type": "text/plain",
            "data": String::from_utf8(bytes.clone()).unwrap_or_default(),
        }),
    }
}

fn apply_document_metadata(
    obj: &mut JsonValue,
    title: Option<String>,
    context: Option<String>,
    citations_enabled: bool,
) {
    if let Some(title) = title {
        obj.as_object_mut()
            .expect("document entry should be an object")
            .insert("title".into(), json!(title));
    }
    if let Some(context) = context {
        obj.as_object_mut()
            .expect("document entry should be an object")
            .insert("context".into(), json!(context));
    }
    if citations_enabled {
        obj.as_object_mut()
            .expect("document entry should be an object")
            .insert("citations".into(), json!({"enabled": true}));
    }
}

fn build_user_file_entry(
    filename: &Option<String>,
    data: &v2t::DataContent,
    media_type: &str,
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope_name: &str,
    cache_control: Option<JsonValue>,
    warnings: &mut Vec<v2t::CallWarning>,
    betas: &mut HashSet<String>,
) -> Option<JsonValue> {
    if media_type.starts_with("image/") {
        return Some(json!({
            "type": "image",
            "source": build_image_source(data, media_type),
            "cache_control": cache_control,
        }));
    }

    if media_type == "application/pdf" {
        betas.insert("pdfs-2024-09-25".into());
        let file_options = parse_anthropic_file_part_options(provider_options, provider_scope_name);
        let metadata_title = file_options.as_ref().and_then(|o| o.title.clone());
        let metadata_context = file_options.as_ref().and_then(|o| o.context.clone());
        let citations_enabled = file_options
            .as_ref()
            .and_then(|o| o.citations.as_ref().map(|c| c.enabled))
            .unwrap_or(false);
        let mut obj = json!({
            "type": "document",
            "source": build_pdf_source(data),
            "cache_control": cache_control,
        });
        apply_document_metadata(
            &mut obj,
            metadata_title.or_else(|| filename.clone()),
            metadata_context,
            citations_enabled,
        );
        return Some(obj);
    }

    if media_type == "text/plain" {
        let file_options = parse_anthropic_file_part_options(provider_options, provider_scope_name);
        let metadata_title = file_options.as_ref().and_then(|o| o.title.clone());
        let metadata_context = file_options.as_ref().and_then(|o| o.context.clone());
        let mut obj = json!({
            "type": "document",
            "source": build_text_source(data),
            "cache_control": cache_control,
        });
        apply_document_metadata(
            &mut obj,
            metadata_title.or_else(|| filename.clone()),
            metadata_context,
            false,
        );
        return Some(obj);
    }

    warnings.push(v2t::CallWarning::Other {
        message: format!("unsupported media type: {media_type}"),
    });
    None
}

fn build_user_block_content(
    messages: &[&v2t::PromptMessage],
    provider_scope_name: &str,
    warnings: &mut Vec<v2t::CallWarning>,
    betas: &mut HashSet<String>,
) -> Vec<JsonValue> {
    let mut anthropic_content = Vec::new();

    for message in messages {
        if let v2t::PromptMessage::User {
            content,
            provider_options,
        } = message
        {
            for (index, part) in content.iter().enumerate() {
                let cache_control =
                    user_part_cache_control(part, index + 1 == content.len(), provider_options);
                match part {
                    v2t::UserPart::Text { text, .. } => {
                        anthropic_content.push(json!({
                            "type": "text",
                            "text": text,
                            "cache_control": cache_control,
                        }));
                    }
                    v2t::UserPart::File {
                        filename,
                        data,
                        media_type,
                        provider_options,
                    } => {
                        if let Some(entry) = build_user_file_entry(
                            filename,
                            data,
                            media_type,
                            provider_options,
                            provider_scope_name,
                            cache_control,
                            warnings,
                            betas,
                        ) {
                            anthropic_content.push(entry);
                        }
                    }
                }
            }
        }
    }

    anthropic_content
}

fn assistant_part_cache_control(
    part: &v2t::AssistantPart,
    is_last: bool,
    provider_options: &Option<v2t::ProviderOptions>,
) -> Option<JsonValue> {
    let part_cache_control = match part {
        v2t::AssistantPart::Text {
            provider_options, ..
        }
        | v2t::AssistantPart::Reasoning {
            provider_options, ..
        }
        | v2t::AssistantPart::File {
            provider_options, ..
        } => get_cache_control(provider_options),
        v2t::AssistantPart::ToolCall(part) => part.provider_options.as_ref().and_then(|opts| {
            opts.get("anthropic").and_then(|map| {
                map.get("cacheControl")
                    .or_else(|| map.get("cache_control"))
                    .cloned()
            })
        }),
        v2t::AssistantPart::ToolResult(part) => part.provider_options.as_ref().and_then(|opts| {
            opts.get("anthropic").and_then(|map| {
                map.get("cacheControl")
                    .or_else(|| map.get("cache_control"))
                    .cloned()
            })
        }),
    };
    let message_cache_control = if is_last {
        get_cache_control(provider_options)
    } else {
        None
    };
    part_cache_control.or(message_cache_control)
}

fn log_assistant_part_diagnostics(
    content: &[v2t::AssistantPart],
    provider_options: &Option<v2t::ProviderOptions>,
) {
    tracing::info!(
        "[THINK_DIAG]: assistant_block parts={} provider_opts_present={}",
        content.len(),
        provider_options.is_some()
    );
    for (index, part) in content.iter().enumerate() {
        match part {
            v2t::AssistantPart::Text {
                provider_options, ..
            } => tracing::info!(
                "[THINK_DIAG]: assistant_part index={} type=text provider_opts_present={}",
                index,
                provider_options.is_some()
            ),
            v2t::AssistantPart::Reasoning {
                provider_options, ..
            } => tracing::info!(
                "[THINK_DIAG]: assistant_part index={} type=reasoning provider_opts_present={}",
                index,
                provider_options.is_some()
            ),
            v2t::AssistantPart::File {
                provider_options, ..
            } => tracing::info!(
                "[THINK_DIAG]: assistant_part index={} type=file provider_opts_present={}",
                index,
                provider_options.is_some()
            ),
            v2t::AssistantPart::ToolCall(part) => tracing::info!(
                "[THINK_DIAG]: assistant_part index={} type=tool_call provider_opts_present={}",
                index,
                part.provider_options.is_some()
            ),
            v2t::AssistantPart::ToolResult(part) => tracing::info!(
                "[THINK_DIAG]: assistant_part index={} type=tool_result provider_opts_present={}",
                index,
                part.provider_options.is_some()
            ),
        }
    }
}

fn build_reasoning_entry(
    text: &str,
    provider_options: &Option<v2t::ProviderOptions>,
    cache_control: Option<JsonValue>,
) -> JsonValue {
    let (signature, redacted) = extract_reasoning_metadata(provider_options);
    let signature_present = signature.is_some();
    let redacted_present = redacted.is_some();

    if let Some(data) = redacted {
        let mut obj = json!({
            "type": "redacted_thinking",
            "data": data,
            "cache_control": cache_control,
        });
        if let Some(signature) = signature {
            obj.as_object_mut()
                .expect("reasoning entry should be an object")
                .insert("signature".into(), json!(signature));
        }
        tracing::info!(
            provider_opts = ?provider_options,
            "[THINK_DIAG]: reasoning_entry type=redacted signature_present={} redacted_present={} text_len={} cache_control_present={}",
            signature_present,
            redacted_present,
            text.len(),
            cache_control.is_some()
        );
        return obj;
    }

    let mut obj = json!({
        "type": "thinking",
        "thinking": text,
        "cache_control": cache_control,
    });
    if let Some(signature) = signature {
        obj.as_object_mut()
            .expect("reasoning entry should be an object")
            .insert("signature".into(), json!(signature));
        tracing::info!(
            provider_opts = ?provider_options,
            "[THINK_DIAG]: reasoning_entry type=thinking signature_present=true redacted_present={} text_len={} cache_control_present={}",
            redacted_present,
            text.len(),
            cache_control.is_some()
        );
    } else {
        tracing::debug!(
            provider_opts = ?provider_options,
            "[THINK_DIAG]: reasoning_entry type=thinking signature_present=false redacted_present={} text_len={} cache_control_present={}",
            redacted_present,
            text.len(),
            cache_control.is_some()
        );
    }
    obj
}

fn build_assistant_tool_call_entry(
    part: &v2t::ToolCallPart,
    cache_control: Option<JsonValue>,
) -> JsonValue {
    let input = serde_json::from_str::<JsonValue>(&part.input).unwrap_or_else(|_| json!({}));
    let mut obj = json!({
        "type": "tool_use",
        "id": part.tool_call_id,
        "name": part.tool_name,
        "input": input,
    });
    if let Some(cache_control) = cache_control {
        obj.as_object_mut()
            .expect("tool call entry should be an object")
            .insert("cache_control".into(), cache_control);
    }
    obj
}

fn build_assistant_message_content(
    content: &[v2t::AssistantPart],
    provider_options: &Option<v2t::ProviderOptions>,
    missing_thinking_reasoning: &mut bool,
) -> Vec<JsonValue> {
    log_assistant_part_diagnostics(content, provider_options);

    let mut reasoning_entries = Vec::new();
    let mut other_entries = Vec::new();

    for (index, part) in content.iter().enumerate() {
        let cache_control =
            assistant_part_cache_control(part, index + 1 == content.len(), provider_options);
        match part {
            v2t::AssistantPart::Text { text, .. } => {
                other_entries.push(json!({
                    "type": "text",
                    "text": text,
                    "cache_control": cache_control,
                }));
            }
            v2t::AssistantPart::Reasoning {
                text,
                provider_options,
            } => {
                reasoning_entries.push(build_reasoning_entry(
                    text,
                    provider_options,
                    cache_control,
                ));
            }
            v2t::AssistantPart::File {
                data, media_type, ..
            } => {
                if media_type.starts_with("image/") {
                    other_entries.push(json!({
                        "type": "image",
                        "source": build_image_source(data, media_type),
                        "cache_control": cache_control,
                    }));
                }
            }
            v2t::AssistantPart::ToolCall(part) => {
                other_entries.push(build_assistant_tool_call_entry(part, cache_control));
            }
            v2t::AssistantPart::ToolResult(_) => {}
        }
    }

    if reasoning_entries.is_empty() {
        if let Some((text, signature)) = extract_persisted_reasoning(provider_options) {
            let mut obj = json!({
                "type": "thinking",
                "thinking": text,
            });
            let has_signature = if let Some(signature) = signature {
                obj.as_object_mut()
                    .expect("persisted reasoning entry should be an object")
                    .insert("signature".into(), json!(signature));
                true
            } else {
                false
            };
            other_entries.insert(0, obj);
            tracing::debug!(
                "[THINK_DIAG]: reasoning_entry source=persisted_reasoning signature_present={} text_len={}",
                has_signature,
                text.len()
            );
        } else if provider_options.is_some() {
            tracing::warn!(
                "[THINK_DIAG]: missing_flag source=persisted_reasoning_missing provider_opts_present={}",
                provider_options.is_some()
            );
            *missing_thinking_reasoning = true;
        } else {
            tracing::info!(
                "[THINK_DIAG]: skip_missing_flag source=persisted_reasoning_missing provider_opts_present=false"
            );
        }
    }

    let mut anthropic_content = reasoning_entries;
    anthropic_content.extend(other_entries);
    anthropic_content
}

fn build_assistant_block_content(
    messages: &[&v2t::PromptMessage],
    missing_thinking_reasoning: &mut bool,
) -> Vec<JsonValue> {
    let mut anthropic_content = Vec::new();

    for message in messages {
        if let v2t::PromptMessage::Assistant {
            content,
            provider_options,
        } = message
        {
            anthropic_content.extend(build_assistant_message_content(
                content,
                provider_options,
                missing_thinking_reasoning,
            ));
        }
    }

    anthropic_content
}

fn build_tool_result_value(output: &v2t::ToolResultOutput) -> JsonValue {
    match output {
        v2t::ToolResultOutput::Text { value } => json!(value),
        v2t::ToolResultOutput::Json { value } => json!(value),
        v2t::ToolResultOutput::ErrorText { value } => json!(value),
        v2t::ToolResultOutput::ErrorJson { value } => json!(value),
        v2t::ToolResultOutput::Content { value } => json!(value),
    }
}

fn build_tool_result_entry(part: &v2t::ToolResultPart) -> JsonValue {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("tool_result"));
    obj.insert("tool_use_id".into(), json!(part.tool_call_id.clone()));
    obj.insert("content".into(), build_tool_result_value(&part.output));
    if matches!(
        part.output,
        v2t::ToolResultOutput::ErrorText { .. } | v2t::ToolResultOutput::ErrorJson { .. }
    ) {
        obj.insert("is_error".into(), json!(true));
    }
    JsonValue::Object(obj)
}

fn build_tool_block_entries(content: &[v2t::ToolMessagePart]) -> Vec<JsonValue> {
    let mut entries = Vec::new();
    for part in content {
        let v2t::ToolMessagePart::ToolResult(part) = part else {
            continue;
        };
        entries.push(build_tool_result_entry(part));
    }
    entries
}

fn append_tool_entries(messages: &mut Vec<JsonValue>, mut entries: Vec<JsonValue>) {
    if entries.is_empty() {
        return;
    }

    let mut appended = false;
    if let Some(last_msg) = messages.last_mut() {
        if last_msg
            .get("role")
            .and_then(|v| v.as_str())
            .map(|role| role.eq_ignore_ascii_case("user"))
            .unwrap_or(false)
        {
            if let Some(content) = last_msg.get_mut("content").and_then(|v| v.as_array_mut()) {
                let can_append = content.iter().all(|existing| {
                    existing
                        .get("type")
                        .and_then(|v| v.as_str())
                        .map(|ty| ty.eq_ignore_ascii_case("tool_result"))
                        .unwrap_or(false)
                });
                if can_append {
                    content.extend(entries.drain(..));
                    appended = true;
                }
            }
        }
    }

    if !appended {
        messages.push(json!({"role": "user", "content": entries}));
    }
}

fn append_tool_block_messages(messages: &mut Vec<JsonValue>, block: &[&v2t::PromptMessage]) {
    for message in block {
        if let v2t::PromptMessage::Tool { content, .. } = message {
            append_tool_entries(messages, build_tool_block_entries(content));
        }
    }
}

fn build_prompt_payload(
    prompt: &[v2t::PromptMessage],
    provider_scope_name: &str,
    warnings: &mut Vec<v2t::CallWarning>,
    betas: &mut HashSet<String>,
) -> PromptPayloadBuild {
    let mut system: Option<Vec<JsonValue>> = None;
    let mut messages = Vec::new();
    let mut missing_thinking_reasoning = false;

    for block in group_prompt_blocks(prompt) {
        match block {
            PromptBlock::System(block_messages) => {
                let entries = build_system_entries(&block_messages);
                if !entries.is_empty() {
                    match &mut system {
                        Some(existing) => existing.extend(entries),
                        None => system = Some(entries),
                    }
                }
            }
            PromptBlock::User(block_messages) => {
                let content =
                    build_user_block_content(&block_messages, provider_scope_name, warnings, betas);
                if !content.is_empty() {
                    messages.push(json!({"role":"user","content": content}));
                }
            }
            PromptBlock::Assistant(block_messages) => {
                let content =
                    build_assistant_block_content(&block_messages, &mut missing_thinking_reasoning);
                if !content.is_empty() {
                    messages.push(json!({"role":"assistant","content": content}));
                }
            }
            PromptBlock::Tool(block_messages) => {
                append_tool_block_messages(&mut messages, &block_messages);
            }
        }
    }

    PromptPayloadBuild {
        system,
        messages,
        missing_thinking_reasoning,
    }
}

fn reorder_last_assistant_reasoning(messages: &mut [JsonValue]) -> bool {
    let Some(last_assistant) = messages
        .iter_mut()
        .rev()
        .find(|message| message.get("role").and_then(|v| v.as_str()) == Some("assistant"))
    else {
        return false;
    };

    let Some(content) = last_assistant
        .get_mut("content")
        .and_then(|v| v.as_array_mut())
    else {
        tracing::warn!("[THINK_DIAG]: missing_flag source=reorder reason=no_content_array");
        return true;
    };

    let mut reasoning_items = Vec::new();
    let original_len = content.len();
    let mut index = 0;
    while index < content.len() {
        let ty = content[index]
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if ty == "thinking" || ty == "redacted_thinking" {
            reasoning_items.push(content.remove(index));
        } else {
            index += 1;
        }
    }

    tracing::info!(
        "[THINK_DIAG]: reasoning_reorder extracted={} original_content_len={} remaining_content_len={}",
        reasoning_items.len(),
        original_len,
        content.len()
    );

    if reasoning_items.is_empty() {
        tracing::warn!(
            "[THINK_DIAG]: missing_flag source=reorder reason=empty_extracted original_content_len={}",
            original_len
        );
        return true;
    }

    let reinserted = reasoning_items.len();
    for (index, item) in reasoning_items.into_iter().enumerate() {
        content.insert(index, item);
    }
    tracing::info!(
        "[THINK_DIAG]: reasoning_reorder reinserted={} final_content_len={}",
        reinserted,
        content.len()
    );
    false
}

fn arg_u32(args: &JsonValue, key: &str) -> Option<u32> {
    args.get(key)
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
}

fn arg_string_vec(args: &JsonValue, key: &str) -> Option<Vec<String>> {
    let values = args.get(key)?.as_array()?;
    Some(
        values
            .iter()
            .filter_map(|value| value.as_str().map(|value| value.to_string()))
            .collect(),
    )
}

fn build_anthropic_provider_tool(
    tool: &v2t::ProviderTool,
    betas: &mut HashSet<String>,
) -> Option<JsonValue> {
    match tool.id.as_str() {
        "anthropic.code_execution_20250522" => {
            betas.insert("code-execution-2025-05-22".into());
            Some(json!({
                "type": "code_execution_20250522",
                "name": "code_execution"
            }))
        }
        "anthropic.code_execution_20250825" => {
            betas.insert("code-execution-2025-08-25".into());
            Some(json!({
                "type": "code_execution_20250825",
                "name": "code_execution"
            }))
        }
        "anthropic.computer_20250124" => {
            betas.insert("computer-use-2025-01-24".into());
            let mut map = serde_json::Map::new();
            map.insert("type".into(), json!("computer_20250124"));
            map.insert("name".into(), json!("computer"));
            if let Some(value) = arg_u32(&tool.args, "displayWidthPx") {
                map.insert("display_width_px".into(), json!(value));
            }
            if let Some(value) = arg_u32(&tool.args, "displayHeightPx") {
                map.insert("display_height_px".into(), json!(value));
            }
            if let Some(value) = arg_u32(&tool.args, "displayNumber") {
                map.insert("display_number".into(), json!(value));
            }
            Some(JsonValue::Object(map))
        }
        "anthropic.computer_20241022" => {
            betas.insert("computer-use-2024-10-22".into());
            let mut map = serde_json::Map::new();
            map.insert("type".into(), json!("computer_20241022"));
            map.insert("name".into(), json!("computer"));
            if let Some(value) = arg_u32(&tool.args, "displayWidthPx") {
                map.insert("display_width_px".into(), json!(value));
            }
            if let Some(value) = arg_u32(&tool.args, "displayHeightPx") {
                map.insert("display_height_px".into(), json!(value));
            }
            if let Some(value) = arg_u32(&tool.args, "displayNumber") {
                map.insert("display_number".into(), json!(value));
            }
            Some(JsonValue::Object(map))
        }
        "anthropic.text_editor_20250124" => {
            betas.insert("computer-use-2025-01-24".into());
            Some(json!({
                "type": "text_editor_20250124",
                "name": "str_replace_editor"
            }))
        }
        "anthropic.text_editor_20241022" => {
            betas.insert("computer-use-2024-10-22".into());
            Some(json!({
                "type": "text_editor_20241022",
                "name": "str_replace_editor"
            }))
        }
        "anthropic.text_editor_20250429" => {
            betas.insert("computer-use-2025-01-24".into());
            Some(json!({
                "type": "text_editor_20250429",
                "name": "str_replace_based_edit_tool"
            }))
        }
        "anthropic.text_editor_20250728" => {
            let mut map = serde_json::Map::new();
            map.insert("type".into(), json!("text_editor_20250728"));
            map.insert("name".into(), json!("str_replace_based_edit_tool"));
            if let Some(value) = arg_u32(&tool.args, "maxCharacters") {
                map.insert("max_characters".into(), json!(value));
            }
            Some(JsonValue::Object(map))
        }
        "anthropic.bash_20250124" => {
            betas.insert("computer-use-2025-01-24".into());
            Some(json!({
                "type": "bash_20250124",
                "name": "bash"
            }))
        }
        "anthropic.bash_20241022" => {
            betas.insert("computer-use-2024-10-22".into());
            Some(json!({
                "type": "bash_20241022",
                "name": "bash"
            }))
        }
        "anthropic.memory_20250818" => {
            betas.insert("context-management-2025-06-27".into());
            Some(json!({
                "type": "memory_20250818",
                "name": "memory"
            }))
        }
        "anthropic.web_fetch_20250910" => {
            betas.insert("web-fetch-2025-09-10".into());
            let mut map = serde_json::Map::new();
            map.insert("type".into(), json!("web_fetch_20250910"));
            map.insert("name".into(), json!("web_fetch"));
            if let Some(value) = arg_u32(&tool.args, "maxUses") {
                map.insert("max_uses".into(), json!(value));
            }
            if let Some(value) = arg_string_vec(&tool.args, "allowedDomains") {
                map.insert("allowed_domains".into(), json!(value));
            }
            if let Some(value) = arg_string_vec(&tool.args, "blockedDomains") {
                map.insert("blocked_domains".into(), json!(value));
            }
            if let Some(citations) = tool.args.get("citations") {
                map.insert("citations".into(), citations.clone());
            }
            if let Some(value) = arg_u32(&tool.args, "maxContentTokens") {
                map.insert("max_content_tokens".into(), json!(value));
            }
            Some(JsonValue::Object(map))
        }
        "anthropic.web_search_20250305" => {
            let mut map = serde_json::Map::new();
            map.insert("type".into(), json!("web_search_20250305"));
            map.insert("name".into(), json!("web_search"));
            if let Some(value) = arg_u32(&tool.args, "maxUses") {
                map.insert("max_uses".into(), json!(value));
            }
            if let Some(value) = arg_string_vec(&tool.args, "allowedDomains") {
                map.insert("allowed_domains".into(), json!(value));
            }
            if let Some(value) = arg_string_vec(&tool.args, "blockedDomains") {
                map.insert("blocked_domains".into(), json!(value));
            }
            if let Some(user_location) = tool.args.get("userLocation") {
                map.insert("user_location".into(), user_location.clone());
            }
            Some(JsonValue::Object(map))
        }
        "anthropic.tool_search_regex_20251119" => {
            betas.insert("advanced-tool-use-2025-11-20".into());
            Some(json!({
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex"
            }))
        }
        "anthropic.tool_search_bm25_20251119" => {
            betas.insert("advanced-tool-use-2025-11-20".into());
            Some(json!({
                "type": "tool_search_tool_bm25_20251119",
                "name": "tool_search_tool_bm25"
            }))
        }
        _ => None,
    }
}

fn build_anthropic_tools(
    options: &v2t::CallOptions,
    json_response_tool: Option<&v2t::FunctionTool>,
    warnings: &mut Vec<v2t::CallWarning>,
    betas: &mut HashSet<String>,
) -> Vec<JsonValue> {
    let mut tools = Vec::new();

    if let Some(tool) = json_response_tool {
        tools.push(json!({
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema
        }));
        if !options.tools.is_empty() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "tools".into(),
                details: Some(
                    "JSON response format does not support tools. The provided tools are ignored."
                        .into(),
                ),
            });
        }
        return tools;
    }

    for tool in &options.tools {
        match tool {
            v2t::Tool::Function(tool) => {
                tools.push(json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }));
            }
            v2t::Tool::Provider(tool) => {
                if let Some(value) = build_anthropic_provider_tool(tool, betas) {
                    tools.push(value);
                } else {
                    warnings.push(v2t::CallWarning::UnsupportedTool {
                        tool_name: tool.id.clone(),
                        details: None,
                    });
                }
            }
        }
    }

    tools
}

fn maybe_apply_disable_parallel_tool_use(
    obj: &mut JsonValue,
    provider_opts: Option<&AnthropicProviderOptions>,
) {
    if let Some(disable_parallel_tool_use) =
        provider_opts.and_then(|opts| opts.disable_parallel_tool_use)
    {
        obj.as_object_mut()
            .expect("tool choice should be an object")
            .insert(
                "disable_parallel_tool_use".into(),
                json!(disable_parallel_tool_use),
            );
    }
}

fn apply_tools_and_tool_choice(
    body: &mut JsonValue,
    tools: Vec<JsonValue>,
    options: &v2t::CallOptions,
    provider_opts: Option<&AnthropicProviderOptions>,
    uses_json_response_tool: bool,
) {
    let has_tools = !tools.is_empty();
    if has_tools {
        body["tools"] = JsonValue::Array(tools);
    }

    if uses_json_response_tool {
        body["tool_choice"] =
            json!({"type":"tool","name":"json","disable_parallel_tool_use": true});
        return;
    }

    let Some(tool_choice) = &options.tool_choice else {
        return;
    };

    match tool_choice {
        v2t::ToolChoice::Auto => {
            let mut obj = json!({"type":"auto"});
            maybe_apply_disable_parallel_tool_use(&mut obj, provider_opts);
            body["tool_choice"] = obj;
        }
        v2t::ToolChoice::Required => {
            let mut obj = json!({"type":"any"});
            maybe_apply_disable_parallel_tool_use(&mut obj, provider_opts);
            body["tool_choice"] = obj;
        }
        v2t::ToolChoice::Tool { name } => {
            let mut obj = json!({"type":"tool","name": name});
            maybe_apply_disable_parallel_tool_use(&mut obj, provider_opts);
            body["tool_choice"] = obj;
        }
        v2t::ToolChoice::None => {
            if has_tools {
                let _ = body.as_object_mut().map(|m| {
                    m.remove("tools");
                });
            }
        }
    }
}

fn apply_sampling_settings(body: &mut JsonValue, options: &v2t::CallOptions) {
    if let Some(temperature) = options.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = options.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(top_k) = options.top_k {
        body["top_k"] = json!(top_k);
    }
    if let Some(stop_sequences) = &options.stop_sequences {
        body["stop_sequences"] = json!(stop_sequences);
    }
}

fn last_assistant_content(prompt: &[v2t::PromptMessage]) -> Option<&[v2t::AssistantPart]> {
    prompt.iter().rev().find_map(|message| match message {
        v2t::PromptMessage::Assistant { content, .. } => Some(content.as_slice()),
        _ => None,
    })
}

fn reasoning_part_count(parts: &[v2t::AssistantPart]) -> usize {
    parts
        .iter()
        .filter(|part| matches!(part, v2t::AssistantPart::Reasoning { .. }))
        .count()
}

fn anthropic_reasoning_scope(
    provider_options: &v2t::ProviderOptions,
) -> Option<&HashMap<String, JsonValue>> {
    provider_options.get("anthropic").or_else(|| {
        provider_options.iter().find_map(|(key, value)| {
            if key.eq_ignore_ascii_case("anthropic") {
                Some(value)
            } else {
                None
            }
        })
    })
}

fn scope_has_reasoning_signature(scope: &HashMap<String, JsonValue>) -> bool {
    ["signature", "persistedReasoningSignature"]
        .iter()
        .any(|expected| scope.keys().any(|key| key.eq_ignore_ascii_case(expected)))
}

fn reasoning_part_has_signature(part: &v2t::AssistantPart) -> bool {
    let v2t::AssistantPart::Reasoning {
        provider_options, ..
    } = part
    else {
        return false;
    };
    provider_options
        .as_ref()
        .and_then(anthropic_reasoning_scope)
        .is_some_and(scope_has_reasoning_signature)
}

fn reasoning_signature_count(parts: &[v2t::AssistantPart]) -> usize {
    parts
        .iter()
        .filter(|part| reasoning_part_has_signature(part))
        .count()
}

fn assistant_reasoning_signature_counts(options: &v2t::CallOptions) -> (usize, usize) {
    last_assistant_content(&options.prompt)
        .map(|parts| {
            (
                reasoning_part_count(parts),
                reasoning_signature_count(parts),
            )
        })
        .unwrap_or((0, 0))
}

fn maybe_warn_missing_reasoning_signature(
    options: &v2t::CallOptions,
    provider_name: &str,
    model_id: &str,
    thinking_cfg: Option<&ThinkingOption>,
    missing_thinking_reasoning: bool,
    warnings: &mut Vec<v2t::CallWarning>,
) {
    if !missing_thinking_reasoning || !matches!(thinking_cfg, Some(ThinkingOption::Enabled { .. }))
    {
        return;
    }

    let (reasoning_parts, reasoning_with_signature) = assistant_reasoning_signature_counts(options);
    tracing::warn!(
        "[THINK_DIAG]: missing reasoning signature; provider={} model={} prompt_messages={} assistant_reasoning_parts={} assistant_reasoning_with_signature={}",
        provider_name,
        model_id,
        options.prompt.len(),
        reasoning_parts,
        reasoning_with_signature
    );
    warnings.push(v2t::CallWarning::Other {
        message: "Anthropic thinking is enabled but the latest assistant message has no reasoning content with a signature. The request continues and the upstream API may reject it.".into(),
    });
}

fn apply_thinking_settings(body: &mut JsonValue, thinking_cfg: Option<&ThinkingOption>) {
    match thinking_cfg {
        Some(ThinkingOption::Enabled { budget_tokens }) => {
            body["thinking"] = json!({"type": "enabled", "budget_tokens": budget_tokens});
            body["max_tokens"] = json!((*budget_tokens as u64) + 1);
            let _ = body.as_object_mut().map(|map| {
                map.remove("temperature");
                map.remove("top_p");
                map.remove("top_k");
            });
        }
        Some(ThinkingOption::Disabled) => {
            body["thinking"] = json!({"type": "disabled"});
        }
        None => {}
    }
}

fn log_payload_summary(body: &JsonValue) {
    let messages_count = body
        .get("messages")
        .and_then(|v| v.as_array())
        .map_or(0, |arr| arr.len());
    let (thinking_count, redacted_count) =
        if let Some(messages_arr) = body.get("messages").and_then(|v| v.as_array()) {
            let mut thinking = 0;
            let mut redacted = 0;
            for msg in messages_arr {
                if let Some(content) = msg.get("content").and_then(|v| v.as_array()) {
                    for item in content {
                        if let Some(ty) = item.get("type").and_then(|v| v.as_str()) {
                            if ty.eq_ignore_ascii_case("thinking") {
                                thinking += 1;
                            } else if ty.eq_ignore_ascii_case("redacted_thinking") {
                                redacted += 1;
                            }
                        }
                    }
                }
            }
            (thinking, redacted)
        } else {
            (0, 0)
        };
    tracing::info!(
        "[THINK_DIAG]: anthropic_payload messages={} thinking_entries={} redacted_entries={} tool_choice_present={} thinking_field={:?}",
        messages_count,
        thinking_count,
        redacted_count,
        body.get("tool_choice").is_some(),
        body.get("thinking")
    );
}

impl<T: HttpTransport> AnthropicMessagesLanguageModel<T> {
    pub fn new(model_id: AnthropicMessagesModelId, cfg: AnthropicMessagesConfig<T>) -> Self {
        Self { model_id, cfg }
    }

    fn build_request_url(&self, streaming: bool) -> String {
        let base = self.cfg.base_url.trim_end_matches('/');
        let path = if streaming { "/messages" } else { "/messages" };
        format!("{}{}", base, path)
    }

    fn build_request_body(
        &self,
        options: &v2t::CallOptions,
    ) -> Result<BuiltAnthropicRequest, SdkError> {
        let mut warnings = Vec::new();
        collect_unsupported_option_warnings(options, &mut warnings);

        let json_response_tool = build_json_response_tool(options, &mut warnings);
        let provider_opts = parse_anthropic_provider_options(
            &options.provider_options,
            &self.cfg.provider_scope_name,
        );
        let mut betas = HashSet::new();

        let PromptPayloadBuild {
            system,
            mut messages,
            mut missing_thinking_reasoning,
        } = build_prompt_payload(
            &options.prompt,
            &self.cfg.provider_scope_name,
            &mut warnings,
            &mut betas,
        );

        let thinking_cfg = provider_opts
            .as_ref()
            .and_then(|opts| opts.thinking.as_ref());
        if matches!(thinking_cfg, Some(ThinkingOption::Enabled { .. }))
            && reorder_last_assistant_reasoning(&mut messages)
        {
            missing_thinking_reasoning = true;
        }

        let tools = build_anthropic_tools(
            options,
            json_response_tool.as_ref(),
            &mut warnings,
            &mut betas,
        );

        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": options.max_output_tokens.unwrap_or(1024),
        });
        if let Some(system) = system {
            body["system"] = JsonValue::Array(system);
        }

        apply_tools_and_tool_choice(
            &mut body,
            tools,
            options,
            provider_opts.as_ref(),
            json_response_tool.is_some(),
        );
        apply_sampling_settings(&mut body, options);
        maybe_warn_missing_reasoning_signature(
            options,
            self.cfg.provider_name,
            &self.model_id,
            thinking_cfg,
            missing_thinking_reasoning,
            &mut warnings,
        );
        apply_thinking_settings(&mut body, thinking_cfg);
        log_payload_summary(&body);

        Ok(BuiltAnthropicRequest {
            body,
            warnings,
            betas,
            uses_json_response_tool: json_response_tool.is_some(),
        })
    }
}

impl AnthropicMessagesLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn builder(
        model_id: impl Into<AnthropicMessagesModelId>,
    ) -> crate::provider_anthropic::provider::AnthropicMessagesBuilder {
        crate::provider_anthropic::provider::AnthropicMessagesBuilder::new(model_id.into())
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync> LanguageModel for AnthropicMessagesLanguageModel<T> {
    fn provider_name(&self) -> &'static str {
        self.cfg.provider_name
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.cfg.supported_urls.clone()
    }

    async fn do_generate(
        &self,
        options: v2t::CallOptions,
    ) -> Result<crate::ai_sdk_core::GenerateResponse, SdkError> {
        let stream_resp = self.do_stream(options).await?;
        collect_stream_to_response(
            stream_resp,
            StreamCollectorConfig {
                allow_reasoning: true,
                reasoning_metadata_scope: Some("anthropic"),
                allow_tool_calls: true,
                allow_tool_results: true,
                allow_files: true,
                allow_source_urls: true,
                fail_on_error: true,
            },
        )
        .await
    }

    async fn do_stream(
        &self,
        options: v2t::CallOptions,
    ) -> Result<crate::ai_sdk_core::StreamResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );
        let BuiltAnthropicRequest {
            body,
            warnings,
            betas,
            uses_json_response_tool: uses_json_tool,
        } = self.build_request_body(&options)?;
        let url = self.build_request_url(true);
        let mut headers: Vec<(String, String)> = self
            .cfg
            .headers
            .iter()
            .filter(|(k, _)| !options::is_internal_sdk_header(k))
            .cloned()
            .collect();

        let mut beta_values: Vec<String> = Vec::new();
        let mut beta_seen: HashSet<String> = HashSet::new();
        headers.retain(|(k, v)| {
            if k.eq_ignore_ascii_case("anthropic-beta") {
                for token in v.split(',') {
                    let trimmed = token.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let value = trimmed.to_string();
                    if beta_seen.insert(value.clone()) {
                        beta_values.push(value);
                    }
                }
                false
            } else {
                true
            }
        });

        for beta in betas.into_iter() {
            if beta.is_empty() {
                continue;
            }
            if beta_seen.insert(beta.clone()) {
                beta_values.push(beta);
            }
        }

        headers.retain(|(k, _)| !k.eq_ignore_ascii_case("accept"));
        headers.push(("accept".into(), "text/event-stream".into()));

        // Always set content-type
        if !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
        {
            headers.push(("content-type".into(), "application/json".into()));
        }

        if !beta_values.is_empty() {
            headers.push(("anthropic-beta".into(), beta_values.join(",")));
        }

        // Ensure Anthropic returns SSE by setting stream: true in the request body
        let mut body = body;
        body["stream"] = serde_json::json!(true);
        let summary = summarize_anthropic_request(&body);
        tracing::debug!(
            "{}: anthropic_request system_entries={} message_entries={} thinking_entries={}",
            REQ_TRACE_PREFIX,
            summary.system_entries,
            summary.message_entries,
            summary.thinking_entries
        );
        tracing::info!(
            "{}: do_stream started model={} stream=true",
            TRACE_PREFIX,
            self.model_id()
        );

        let resp = match self
            .cfg
            .http
            .post_json_stream(&url, &headers, &body, &self.cfg.transport_cfg)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let mapped = map_transport_error_to_sdk_error(e);
                tracing::info!(
                    "{}: transport error during stream start: {}",
                    TRACE_PREFIX,
                    mapped
                );
                return Err(mapped);
            }
        };

        let (bytes_stream, _res_headers) = <T as HttpTransport>::into_stream(resp);
        tracing::info!("{}: SSE stream acquired; decoding events", TRACE_PREFIX);

        // Decode SSE and convert to provider-agnostic events, then map to v2 parts
        let events = sse_to_events::<_, AnthropicChunk, SdkError>(bytes_stream.map(|chunk_res| {
            chunk_res.map_err(|e| match e {
                crate::ai_sdk_core::error::TransportError::IdleReadTimeout(_) => SdkError::Timeout,
                crate::ai_sdk_core::error::TransportError::ConnectTimeout(_) => SdkError::Timeout,
                other => SdkError::Transport(other),
            })
        }));

        let mut hooks = EventMapperHooks::default();
        hooks.data = Some(Box::new(
            |_state: &mut crate::ai_sdk_core::EventMapperState<()>, key, value| {
                if key == "reasoning_signature" {
                    if let Some(sig) = value.get("signature").and_then(|s| s.as_str()) {
                        return Some(vec![v2t::StreamPart::ReasoningSignature {
                            signature: sig.to_string(),
                            provider_metadata: None,
                        }]);
                    }
                }
                None
            },
        ));

        let parts = map_events_to_parts(
            Box::pin(events),
            EventMapperConfig {
                warnings,
                treat_tool_names_as_text: if uses_json_tool {
                    HashSet::from(["json".to_string()])
                } else {
                    HashSet::new()
                },
                default_text_id: "text-1",
                finish_reason_fallback: v2t::FinishReason::Unknown,
                initial_extra: (),
                hooks,
            },
        );

        Ok(crate::ai_sdk_core::StreamResponse {
            stream: parts,
            request_body: Some(body),
            response_headers: None,
        })
    }
}

#[derive(Default)]
struct AnthropicRequestSummary {
    system_entries: usize,
    message_entries: usize,
    thinking_entries: usize,
}

fn summarize_anthropic_request(body: &JsonValue) -> AnthropicRequestSummary {
    let system_entries = body
        .get("system")
        .and_then(|v| v.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);
    let message_entries = body
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);
    let thinking_entries = body
        .get("messages")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|msg| msg.get("content").and_then(|c| c.as_array()))
                .flatten()
                .filter(|part| part.get("type").and_then(|t| t.as_str()) == Some("thinking"))
                .count()
        })
        .unwrap_or(0);

    AnthropicRequestSummary {
        system_entries,
        message_entries,
        thinking_entries,
    }
}

// --- SSE parsing: Anthropic provider chunk to crate::ai_sdk_types::Event ---

#[derive(Default)]
struct AnthropicChunk {
    tool_calls: HashMap<usize, AnthropicToolCallState>,
    pending_deltas: HashMap<usize, Vec<String>>,
}

struct AnthropicToolCallState {
    id: String,
}

impl ProviderChunk for AnthropicChunk {
    fn try_from_sse(&mut self, event: &SseEvent) -> Result<Option<Vec<ProviderEvent>>, SdkError> {
        // Fast path: end of message
        if let Some(ev) = &event.event {
            if ev == "message_stop" {
                let mut out = self.drain_tool_calls();
                out.push(ProviderEvent::Done);
                return Ok(Some(out));
            }
        }

        // Try to parse JSON payload; ignore non-JSON heartbeats
        let v: JsonValue = match serde_json::from_slice(&event.data) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        Ok(Some(self.parse_anthropic_frame(&v)))
    }
}

impl AnthropicChunk {
    fn drain_tool_calls(&mut self) -> Vec<ProviderEvent> {
        let mut out = Vec::new();
        for (_idx, state) in self.tool_calls.drain() {
            out.push(ProviderEvent::ToolCallEnd { id: state.id });
        }
        self.pending_deltas.clear();
        out
    }

    fn parse_anthropic_frame(&mut self, v: &JsonValue) -> Vec<ProviderEvent> {
        let mut out = Vec::new();
        if let Some(t) = v.get("type").and_then(|s| s.as_str()) {
            match t {
                "message_start" => {
                    if let Some(usage) = v.get("message").and_then(|m| m.get("usage")) {
                        push_anthropic_usage(&mut out, usage);
                    }
                }
                "message_delta" => {
                    if let Some(usage) = v.get("usage") {
                        push_anthropic_usage(&mut out, usage);
                    }
                }
                "content_block_delta" => self.push_content_block_delta(v, &mut out),
                "content_block_start" => self.push_content_block_start(v, &mut out),
                "message_stop" => {
                    out.extend(self.drain_tool_calls());
                    out.push(ProviderEvent::Done);
                }
                _ => {}
            }
        }
        out
    }

    fn push_content_block_delta(&mut self, v: &JsonValue, out: &mut Vec<ProviderEvent>) {
        let Some(delta) = v.get("delta") else {
            return;
        };

        match delta.get("type").and_then(|s| s.as_str()) {
            Some("text_delta") => {
                if let Some(text) = delta.get("text").and_then(|s| s.as_str()) {
                    out.push(ProviderEvent::TextDelta {
                        delta: text.to_string(),
                    });
                }
            }
            Some("thinking_delta") => {
                if let Some(thinking) = delta.get("thinking").and_then(|s| s.as_str()) {
                    if !thinking.is_empty() {
                        out.push(ProviderEvent::ReasoningDelta {
                            delta: thinking.to_string(),
                        });
                    }
                }
            }
            Some("signature_delta") => {
                if let Some(signature) = delta.get("signature").and_then(|s| s.as_str()) {
                    out.push(ProviderEvent::Data {
                        key: "reasoning_signature".to_string(),
                        value: json!({"signature": signature}),
                    });
                }
            }
            Some("input_json_delta") => self.push_tool_call_delta(v, delta, out),
            _ => {}
        }
    }

    fn push_tool_call_delta(
        &mut self,
        v: &JsonValue,
        delta: &JsonValue,
        out: &mut Vec<ProviderEvent>,
    ) {
        let idx = v.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
        let arg = delta
            .get("partial_json")
            .or_else(|| delta.get("json"))
            .or_else(|| delta.get("delta"))
            .and_then(|s| s.as_str())
            .unwrap_or_default()
            .to_string();

        if let Some(idx) = idx {
            if let Some(state) = self.tool_calls.get(&idx) {
                out.push(ProviderEvent::ToolCallDelta {
                    id: state.id.clone(),
                    args_json: arg,
                });
            } else {
                self.pending_deltas.entry(idx).or_default().push(arg);
            }
        }
    }

    fn push_content_block_start(&mut self, v: &JsonValue, out: &mut Vec<ProviderEvent>) {
        let idx = v.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
        let Some(content_block) = v.get("content_block") else {
            return;
        };

        match content_block.get("type").and_then(|s| s.as_str()) {
            Some("tool_use") => self.push_tool_call_start(idx, content_block, out),
            Some("thinking") | Some("redacted_thinking") => {
                let id = idx
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "0".to_string());
                out.push(ProviderEvent::ReasoningStart { id });
            }
            _ => {}
        }
    }

    fn push_tool_call_start(
        &mut self,
        idx: Option<usize>,
        content_block: &JsonValue,
        out: &mut Vec<ProviderEvent>,
    ) {
        let id = content_block
            .get("id")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        let name = content_block
            .get("name")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());

        if let (Some(idx), Some(id), Some(name)) = (idx, id, name) {
            self.tool_calls
                .insert(idx, AnthropicToolCallState { id: id.clone() });
            out.push(ProviderEvent::ToolCallStart {
                id: id.clone(),
                name,
            });
            if let Some(pending) = self.pending_deltas.remove(&idx) {
                for delta in pending {
                    out.push(ProviderEvent::ToolCallDelta {
                        id: id.clone(),
                        args_json: delta,
                    });
                }
            }
        }
    }
}

fn push_anthropic_usage(out: &mut Vec<ProviderEvent>, usage: &JsonValue) {
    use crate::ai_sdk_types::TokenUsage;
    let norm = crate::ai_sdk_types::usage::normalize_anthropic(usage);
    let input = norm
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let output = norm
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let total = norm
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(input + output);
    let cache_read_tokens = norm
        .get("cache_read_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let cache_write_tokens = norm
        .get("cache_write_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    out.push(ProviderEvent::Usage {
        usage: TokenUsage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: total,
            cache_read_tokens,
            cache_write_tokens,
        },
    });
    out.push(ProviderEvent::Data {
        key: "usage".to_string(),
        value: norm,
    });
}
