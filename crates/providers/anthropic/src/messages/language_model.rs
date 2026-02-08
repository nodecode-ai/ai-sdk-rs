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

pub struct AnthropicMessagesLanguageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport> {
    model_id: AnthropicMessagesModelId,
    cfg: AnthropicMessagesConfig<T>,
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
    ) -> Result<
        (
            JsonValue,
            Vec<v2t::CallWarning>,
            Option<AnthropicProviderOptions>,
            std::collections::HashSet<String>,
            bool,
        ),
        SdkError,
    > {
        let mut warnings: Vec<v2t::CallWarning> = vec![];

        // Unsupported knobs parity
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

        // JSON response format -> inject json tool
        let mut json_response_tool: Option<v2t::FunctionTool> = None;
        if let Some(v2t::ResponseFormat::Json { schema, .. }) = &options.response_format {
            match schema {
                Some(s) => {
                    json_response_tool = Some(v2t::FunctionTool {
                        r#type: v2t::FunctionToolType::Function,
                        name: "json".into(),
                        description: Some("Respond with a JSON object.".into()),
                        input_schema: s.clone(),
                        provider_options: None,
                    });
                }
                None => {
                    warnings.push(v2t::CallWarning::UnsupportedSetting { setting: "responseFormat".into(), details: Some("JSON response format requires a schema. The response format is ignored.".into()) });
                }
            }
        }

        // Provider options merged upstream (config defaults + call-level overrides)
        let provider_opts = parse_anthropic_provider_options(
            &options.provider_options,
            &self.cfg.provider_scope_name,
        );
        let mut betas: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Helpers
        fn get_cache_control(opts: &Option<v2t::ProviderOptions>) -> Option<JsonValue> {
            let map = opts.as_ref()?.get("anthropic")?;
            if let Some(v) = map.get("cacheControl").or_else(|| map.get("cache_control")) {
                Some(v.clone())
            } else {
                None
            }
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
        fn extract_reasoning_metadata(
            opts: &Option<v2t::ProviderOptions>,
        ) -> (Option<String>, Option<String>) {
            fn from_scope(
                scope: &std::collections::HashMap<String, serde_json::Value>,
            ) -> (Option<String>, Option<String>) {
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
            let Some(map) = opts.as_ref() else {
                return (None, None);
            };
            if let Some(scope) = map.get("anthropic") {
                let res = from_scope(scope);
                if res.0.is_some() || res.1.is_some() {
                    let scope_keys: Vec<&str> = scope.keys().map(|k| k.as_str()).collect();
                    tracing::info!(
                        "[THINK_DIAG]: extract_reasoning_metadata scope=primary signature_present={} redacted_present={} keys={:?}",
                        res.0.is_some(),
                        res.1.is_some(),
                        scope_keys
                    );
                    return res;
                }
            }
            for (key, scope) in map.iter() {
                if key.eq_ignore_ascii_case("anthropic")
                    || key.to_ascii_lowercase().contains("anthropic")
                {
                    let res = from_scope(scope);
                    if res.0.is_some() || res.1.is_some() {
                        let scope_keys: Vec<&str> = scope.keys().map(|k| k.as_str()).collect();
                        tracing::info!(
                            "[THINK_DIAG]: extract_reasoning_metadata scope=alias key={} signature_present={} redacted_present={} keys={:?}",
                            key,
                            res.0.is_some(),
                            res.1.is_some(),
                            scope_keys
                        );
                        return res;
                    }
                }
            }
            for scope in map.values() {
                let res = from_scope(scope);
                if res.0.is_some() || res.1.is_some() {
                    let scope_keys: Vec<&str> = scope.keys().map(|k| k.as_str()).collect();
                    tracing::info!(
                        "[THINK_DIAG]: extract_reasoning_metadata scope=fallback signature_present={} redacted_present={} keys={:?}",
                        res.0.is_some(),
                        res.1.is_some(),
                        scope_keys
                    );
                    return res;
                }
            }
            let scope_keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
            tracing::info!(
                "[THINK_DIAG]: extract_reasoning_metadata scope=missing signature_present=false redacted_present=false keys={:?}",
                scope_keys
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

        // Group messages into blocks to support system arrays and user/assistant coalescing
        enum Block<'a> {
            System(Vec<&'a v2t::PromptMessage>),
            Assistant(Vec<&'a v2t::PromptMessage>),
            User(Vec<&'a v2t::PromptMessage>),
            Tool(Vec<&'a v2t::PromptMessage>),
        }
        let mut blocks: Vec<Block> = Vec::new();
        let mut acc: Option<(u8, Vec<&v2t::PromptMessage>)> = None;
        for m in &options.prompt {
            let tag = match m {
                v2t::PromptMessage::System { .. } => 0u8,
                v2t::PromptMessage::Assistant { .. } => 1u8,
                v2t::PromptMessage::User { .. } => 2u8,
                v2t::PromptMessage::Tool { .. } => 3u8,
            };
            match &mut acc {
                Some((t, v)) if *t == tag => v.push(m),
                Some((_t, v)) => {
                    let saved = std::mem::replace(v, vec![m]);
                    let saved_tag = std::mem::replace(&mut acc.as_mut().unwrap().0, tag);
                    match saved_tag {
                        0 => blocks.push(Block::System(saved)),
                        1 => blocks.push(Block::Assistant(saved)),
                        2 => blocks.push(Block::User(saved)),
                        _ => blocks.push(Block::Tool(saved)),
                    }
                }
                None => acc = Some((tag, vec![m])),
            }
        }
        if let Some((t, v)) = acc.take() {
            match t {
                0 => blocks.push(Block::System(v)),
                1 => blocks.push(Block::Assistant(v)),
                2 => blocks.push(Block::User(v)),
                _ => blocks.push(Block::Tool(v)),
            }
        }

        let mut system: Option<Vec<JsonValue>> = None;
        let mut messages: Vec<JsonValue> = Vec::new();

        let mut missing_thinking_reasoning = false;

        for block in blocks {
            match block {
                Block::System(msgs) => {
                    let mut sys_vec: Vec<JsonValue> = Vec::new();
                    for sys in msgs {
                        if let v2t::PromptMessage::System {
                            content,
                            provider_options,
                        } = sys
                        {
                            if content.is_empty() {
                                continue;
                            }
                            let mut obj = json!({"type":"text","text": content});
                            if let Some(cc) = get_cache_control(provider_options) {
                                obj.as_object_mut()
                                    .unwrap()
                                    .insert("cache_control".into(), cc);
                            }
                            sys_vec.push(obj);
                        }
                    }
                    if !sys_vec.is_empty() {
                        match &mut system {
                            Some(existing) => existing.extend(sys_vec),
                            None => system = Some(sys_vec),
                        }
                    }
                }
                Block::User(msgs_in_block) => {
                    let mut anthropic_content: Vec<JsonValue> = Vec::new();
                    for message in msgs_in_block {
                        if let v2t::PromptMessage::User {
                            content,
                            provider_options,
                        } = message
                        {
                            for (idx, part) in content.iter().enumerate() {
                                let is_last = idx + 1 == content.len();
                                let part_cc = match part {
                                    v2t::UserPart::Text {
                                        provider_options, ..
                                    }
                                    | v2t::UserPart::File {
                                        provider_options, ..
                                    } => get_cache_control(provider_options),
                                };
                                let msg_cc = if is_last {
                                    get_cache_control(provider_options)
                                } else {
                                    None
                                };
                                let cache_control = part_cc.or(msg_cc);
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
                                        if media_type.starts_with("image/") {
                                            let source = match data {
                                                v2t::DataContent::Url { url } => {
                                                    json!({"type":"url","url": url})
                                                }
                                                _ => {
                                                    json!({
                                                        "type": "base64",
                                                        "media_type": if media_type == "image/*" {
                                                            "image/jpeg"
                                                        } else {
                                                            media_type
                                                        },
                                                        "data": to_base64(data).unwrap_or_default(),
                                                    })
                                                }
                                            };
                                            anthropic_content.push(json!({
                                                "type": "image",
                                                "source": source,
                                                "cache_control": cache_control,
                                            }));
                                        } else if media_type == "application/pdf" {
                                            betas.insert("pdfs-2024-09-25".into());
                                            let fopts = parse_anthropic_file_part_options(
                                                provider_options,
                                                &self.cfg.provider_scope_name,
                                            );
                                            let metadata_title =
                                                fopts.as_ref().and_then(|o| o.title.clone());
                                            let metadata_ctx =
                                                fopts.as_ref().and_then(|o| o.context.clone());
                                            let citations = fopts
                                                .as_ref()
                                                .and_then(|o| {
                                                    o.citations.as_ref().map(|c| c.enabled)
                                                })
                                                .unwrap_or(false);
                                            let source = match data {
                                                v2t::DataContent::Url { url } => {
                                                    json!({"type":"url","url": url})
                                                }
                                                _ => {
                                                    json!({
                                                        "type": "base64",
                                                        "media_type": "application/pdf",
                                                        "data": to_base64(data).unwrap_or_default(),
                                                    })
                                                }
                                            };
                                            let mut obj = json!({
                                                "type": "document",
                                                "source": source,
                                                "cache_control": cache_control,
                                            });
                                            if let Some(ti) =
                                                metadata_title.or_else(|| filename.clone())
                                            {
                                                obj.as_object_mut()
                                                    .unwrap()
                                                    .insert("title".into(), json!(ti));
                                            }
                                            if let Some(ctx) = metadata_ctx {
                                                obj.as_object_mut()
                                                    .unwrap()
                                                    .insert("context".into(), json!(ctx));
                                            }
                                            if citations {
                                                obj.as_object_mut().unwrap().insert(
                                                    "citations".into(),
                                                    json!({"enabled": true}),
                                                );
                                            }
                                            anthropic_content.push(obj);
                                        } else if media_type == "text/plain" {
                                            let fopts = parse_anthropic_file_part_options(
                                                provider_options,
                                                &self.cfg.provider_scope_name,
                                            );
                                            let metadata_title =
                                                fopts.as_ref().and_then(|o| o.title.clone());
                                            let metadata_ctx =
                                                fopts.as_ref().and_then(|o| o.context.clone());
                                            let source = match data {
                                                v2t::DataContent::Url { url } => {
                                                    json!({"type":"url","url": url})
                                                }
                                                v2t::DataContent::Base64 { base64 } => {
                                                    json!({
                                                        "type": "text",
                                                        "media_type": "text/plain",
                                                        "data": String::from_utf8(
                                                            base64::engine::general_purpose::STANDARD.decode(base64).unwrap_or_default(),
                                                        )
                                                        .unwrap_or_default(),
                                                    })
                                                }
                                                v2t::DataContent::Bytes { bytes } => {
                                                    json!({
                                                        "type": "text",
                                                        "media_type": "text/plain",
                                                        "data": String::from_utf8(bytes.clone())
                                                            .unwrap_or_default(),
                                                    })
                                                }
                                            };
                                            let mut obj = json!({
                                                "type": "document",
                                                "source": source,
                                                "cache_control": cache_control,
                                            });
                                            if let Some(ti) =
                                                metadata_title.or_else(|| filename.clone())
                                            {
                                                obj.as_object_mut()
                                                    .unwrap()
                                                    .insert("title".into(), json!(ti));
                                            }
                                            if let Some(ctx) = metadata_ctx {
                                                obj.as_object_mut()
                                                    .unwrap()
                                                    .insert("context".into(), json!(ctx));
                                            }
                                            anthropic_content.push(obj);
                                        } else {
                                            warnings.push(v2t::CallWarning::Other {
                                                message: format!(
                                                    "unsupported media type: {}",
                                                    media_type
                                                ),
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !anthropic_content.is_empty() {
                        messages.push(json!({"role":"user","content": anthropic_content}));
                    }
                }
                Block::Assistant(msgs_in_block) => {
                    let mut anthropic_content: Vec<JsonValue> = Vec::new();
                    for message in msgs_in_block {
                        if let v2t::PromptMessage::Assistant {
                            content,
                            provider_options,
                        } = message
                        {
                            tracing::info!(
                                "[THINK_DIAG]: assistant_block parts={} provider_opts_present={}",
                                content.len(),
                                provider_options.is_some()
                            );
                            for (i, part) in content.iter().enumerate() {
                                match part {
                                    v2t::AssistantPart::Text { provider_options, .. } => tracing::info!("[THINK_DIAG]: assistant_part index={} type=text provider_opts_present={}", i, provider_options.is_some()),
                                    v2t::AssistantPart::Reasoning { provider_options, .. } => tracing::info!("[THINK_DIAG]: assistant_part index={} type=reasoning provider_opts_present={}", i, provider_options.is_some()),
                                    v2t::AssistantPart::File { provider_options, .. } => tracing::info!("[THINK_DIAG]: assistant_part index={} type=file provider_opts_present={}", i, provider_options.is_some()),
                                    v2t::AssistantPart::ToolCall(part) => tracing::info!("[THINK_DIAG]: assistant_part index={} type=tool_call provider_opts_present={}", i, part.provider_options.is_some()),
                                    v2t::AssistantPart::ToolResult(part) => tracing::info!("[THINK_DIAG]: assistant_part index={} type=tool_result provider_opts_present={}", i, part.provider_options.is_some()),
                                }
                            }
                            let mut reasoning_entries: Vec<JsonValue> = Vec::new();
                            let mut other_entries: Vec<JsonValue> = Vec::new();
                            for (idx, part) in content.iter().enumerate() {
                                let is_last = idx + 1 == content.len();
                                let part_cc = match part {
                                    v2t::AssistantPart::Text {
                                        provider_options, ..
                                    }
                                    | v2t::AssistantPart::Reasoning {
                                        provider_options, ..
                                    }
                                    | v2t::AssistantPart::File {
                                        provider_options, ..
                                    } => get_cache_control(provider_options),
                                    v2t::AssistantPart::ToolCall(p) => {
                                        p.provider_options.as_ref().and_then(|o| {
                                            o.get("anthropic").and_then(|m| {
                                                m.get("cacheControl")
                                                    .or_else(|| m.get("cache_control"))
                                                    .cloned()
                                            })
                                        })
                                    }
                                    v2t::AssistantPart::ToolResult(p) => {
                                        p.provider_options.as_ref().and_then(|o| {
                                            o.get("anthropic").and_then(|m| {
                                                m.get("cacheControl")
                                                    .or_else(|| m.get("cache_control"))
                                                    .cloned()
                                            })
                                        })
                                    }
                                };
                                let msg_cc = if is_last {
                                    get_cache_control(provider_options)
                                } else {
                                    None
                                };
                                let cache_control = part_cc.or(msg_cc);
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
                                        let (signature, redacted) =
                                            extract_reasoning_metadata(provider_options);
                                        let signature_present = signature.is_some();
                                        let redacted_present = redacted.is_some();
                                        if let Some(data) = redacted {
                                            let mut obj = json!({
                                                "type": "redacted_thinking",
                                                "data": data,
                                                "cache_control": cache_control,
                                            });
                                            if let Some(sig) = signature {
                                                obj.as_object_mut()
                                                    .unwrap()
                                                    .insert("signature".into(), json!(sig));
                                            }
                                            reasoning_entries.push(obj);
                                            tracing::info!(
                                                provider_opts = ?provider_options,
                                                "[THINK_DIAG]: reasoning_entry type=redacted signature_present={} redacted_present={} text_len={} cache_control_present={}",
                                                signature_present,
                                                redacted_present,
                                                text.len(),
                                                cache_control.is_some()
                                            );
                                            continue;
                                        }

                                        let mut obj = json!({
                                            "type": "thinking",
                                            "thinking": text,
                                            "cache_control": cache_control,
                                        });
                                        if let Some(sig) = signature {
                                            obj.as_object_mut()
                                                .unwrap()
                                                .insert("signature".into(), json!(sig));
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
                                        reasoning_entries.push(obj);
                                    }
                                    v2t::AssistantPart::File {
                                        filename: _,
                                        data,
                                        media_type,
                                        ..
                                    } => {
                                        if media_type.starts_with("image/") {
                                            let source = match data {
                                                v2t::DataContent::Url { url } => json!({
                                                    "type": "url",
                                                    "url": url,
                                                }),
                                                _ => json!({
                                                    "type": "base64",
                                                    "media_type": if media_type == "image/*" {
                                                        "image/jpeg"
                                                    } else {
                                                        media_type
                                                    },
                                                    "data": to_base64(data).unwrap_or_default(),
                                                }),
                                            };
                                            other_entries.push(json!({
                                                "type": "image",
                                                "source": source,
                                                "cache_control": cache_control,
                                            }));
                                        }
                                    }
                                    v2t::AssistantPart::ToolCall(tc) => {
                                        let input = serde_json::from_str::<JsonValue>(&tc.input)
                                            .unwrap_or_else(|_| json!({}));
                                        let mut obj = json!({
                                            "type": "tool_use",
                                            "id": tc.tool_call_id,
                                            "name": tc.tool_name,
                                            "input": input,
                                        });
                                        if let Some(cc) = cache_control {
                                            obj.as_object_mut()
                                                .unwrap()
                                                .insert("cache_control".into(), cc);
                                        }
                                        other_entries.push(obj);
                                    }
                                    v2t::AssistantPart::ToolResult(_tr) => {
                                        // Provider executed tool results would require structured mapping; not implemented here.
                                    }
                                }
                            }
                            if !reasoning_entries.is_empty() {
                                anthropic_content.extend(reasoning_entries);
                            } else if let Some((text, signature)) =
                                extract_persisted_reasoning(provider_options)
                            {
                                let mut obj = json!({
                                    "type": "thinking",
                                    "thinking": text,
                                });
                                let has_signature = if let Some(sig) = signature {
                                    obj.as_object_mut()
                                        .unwrap()
                                        .insert("signature".into(), json!(sig));
                                    true
                                } else {
                                    false
                                };
                                anthropic_content.insert(0, obj);
                                tracing::debug!(
                                    "[THINK_DIAG]: reasoning_entry source=persisted_reasoning signature_present={} text_len={}",
                                    has_signature,
                                    text.len()
                                );
                            } else {
                                let provider_opts_present = provider_options.is_some();
                                if provider_opts_present {
                                    tracing::warn!(
                                        "[THINK_DIAG]: missing_flag source=persisted_reasoning_missing provider_opts_present={}",
                                        provider_opts_present
                                    );
                                    missing_thinking_reasoning = true;
                                } else {
                                    tracing::info!(
                                        "[THINK_DIAG]: skip_missing_flag source=persisted_reasoning_missing provider_opts_present=false"
                                    );
                                }
                            }
                            anthropic_content.extend(other_entries);
                        }
                    }
                    if !anthropic_content.is_empty() {
                        messages.push(json!({"role":"assistant","content": anthropic_content}));
                    }
                }
                Block::Tool(msgs_in_block) => {
                    for message in msgs_in_block {
                        if let v2t::PromptMessage::Tool { content, .. } = message {
                            let mut entries: Vec<JsonValue> = Vec::new();
                            for part in content {
                                let part = match part {
                                    v2t::ToolMessagePart::ToolResult(part) => part,
                                    v2t::ToolMessagePart::ToolApprovalResponse(_) => {
                                        continue;
                                    }
                                };
                                let val = match &part.output {
                                    v2t::ToolResultOutput::Text { value } => json!(value),
                                    v2t::ToolResultOutput::Json { value } => json!(value),
                                    v2t::ToolResultOutput::ErrorText { value } => json!(value),
                                    v2t::ToolResultOutput::ErrorJson { value } => json!(value),
                                    v2t::ToolResultOutput::Content { value } => json!(value),
                                };
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("tool_result"));
                                obj.insert("tool_use_id".into(), json!(part.tool_call_id.clone()));
                                obj.insert("content".into(), val);
                                if matches!(
                                    part.output,
                                    v2t::ToolResultOutput::ErrorText { .. }
                                        | v2t::ToolResultOutput::ErrorJson { .. }
                                ) {
                                    obj.insert("is_error".into(), json!(true));
                                }
                                entries.push(JsonValue::Object(obj));
                            }
                            if !entries.is_empty() {
                                let mut entries = entries;
                                let mut appended = false;
                                if let Some(last_msg) = messages.last_mut() {
                                    if last_msg
                                        .get("role")
                                        .and_then(|v| v.as_str())
                                        .map(|r| r.eq_ignore_ascii_case("user"))
                                        .unwrap_or(false)
                                    {
                                        if let Some(arr) = last_msg
                                            .get_mut("content")
                                            .and_then(|v| v.as_array_mut())
                                        {
                                            let can_append = arr.iter().all(|existing| {
                                                existing
                                                    .get("type")
                                                    .and_then(|v| v.as_str())
                                                    .map(|ty| {
                                                        ty.eq_ignore_ascii_case("tool_result")
                                                    })
                                                    .unwrap_or(false)
                                            });
                                            if can_append {
                                                arr.extend(entries.drain(..));
                                                appended = true;
                                            }
                                        }
                                    }
                                }
                                if !appended {
                                    messages.push(json!({"role": "user", "content": entries}));
                                }
                            }
                        }
                    }
                }
            }
        }
        if matches!(
            provider_opts.as_ref().and_then(|o| o.thinking.as_ref()),
            Some(ThinkingOption::Enabled { .. })
        ) {
            if let Some(last_assistant) = messages
                .iter_mut()
                .rev()
                .find(|msg| msg.get("role").and_then(|v| v.as_str()) == Some("assistant"))
            {
                if let Some(content) = last_assistant
                    .get_mut("content")
                    .and_then(|v| v.as_array_mut())
                {
                    let mut reasoning_items: Vec<JsonValue> = Vec::new();
                    let mut idx = 0;
                    let original_len = content.len();
                    while idx < content.len() {
                        let ty = content[idx]
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if ty == "thinking" || ty == "redacted_thinking" {
                            reasoning_items.push(content.remove(idx));
                        } else {
                            idx += 1;
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
                        missing_thinking_reasoning = true;
                    } else {
                        let mut insert_idx = 0;
                        for item in reasoning_items {
                            content.insert(insert_idx, item);
                            insert_idx += 1;
                        }
                        tracing::info!(
                            "[THINK_DIAG]: reasoning_reorder reinserted={} final_content_len={}",
                            insert_idx,
                            content.len()
                        );
                    }
                } else {
                    tracing::warn!(
                        "[THINK_DIAG]: missing_flag source=reorder reason=no_content_array"
                    );
                    missing_thinking_reasoning = true;
                }
            }
        }

        // Build tools array
        // Anthropic function tools do NOT include a "type" field.
        let mut tools: Vec<JsonValue> = Vec::new();
        let mut tool_warnings: Vec<v2t::CallWarning> = vec![];
        fn arg_u32(args: &JsonValue, key: &str) -> Option<u32> {
            args.get(key)
                .and_then(|v| v.as_u64())
                .and_then(|v| u32::try_from(v).ok())
        }
        fn arg_string_vec(args: &JsonValue, key: &str) -> Option<Vec<String>> {
            let arr = args.get(key)?.as_array()?;
            let values: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            Some(values)
        }
        if let Some(ref t) = json_response_tool {
            // Inject the JSON response tool as a plain function tool (no "type")
            tools.push(json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema
            }));
            // When JSON response format is used, ignore additional user-provided tools for Anthropic.
            if !options.tools.is_empty() {
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "tools".into(),
                    details: Some("JSON response format does not support tools. The provided tools are ignored.".into()),
                });
            }
        } else {
            for tool in &options.tools {
                match tool {
                    v2t::Tool::Function(t) => {
                        tools.push(json!({
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.input_schema
                        }));
                    }
                    v2t::Tool::Provider(tool) => match tool.id.as_str() {
                        "anthropic.code_execution_20250522" => {
                            betas.insert("code-execution-2025-05-22".into());
                            tools.push(json!({
                                "type": "code_execution_20250522",
                                "name": "code_execution"
                            }));
                        }
                        "anthropic.code_execution_20250825" => {
                            betas.insert("code-execution-2025-08-25".into());
                            tools.push(json!({
                                "type": "code_execution_20250825",
                                "name": "code_execution"
                            }));
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
                            tools.push(JsonValue::Object(map));
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
                            tools.push(JsonValue::Object(map));
                        }
                        "anthropic.text_editor_20250124" => {
                            betas.insert("computer-use-2025-01-24".into());
                            tools.push(json!({
                                "type": "text_editor_20250124",
                                "name": "str_replace_editor"
                            }));
                        }
                        "anthropic.text_editor_20241022" => {
                            betas.insert("computer-use-2024-10-22".into());
                            tools.push(json!({
                                "type": "text_editor_20241022",
                                "name": "str_replace_editor"
                            }));
                        }
                        "anthropic.text_editor_20250429" => {
                            betas.insert("computer-use-2025-01-24".into());
                            tools.push(json!({
                                "type": "text_editor_20250429",
                                "name": "str_replace_based_edit_tool"
                            }));
                        }
                        "anthropic.text_editor_20250728" => {
                            let mut map = serde_json::Map::new();
                            map.insert("type".into(), json!("text_editor_20250728"));
                            map.insert("name".into(), json!("str_replace_based_edit_tool"));
                            if let Some(value) = arg_u32(&tool.args, "maxCharacters") {
                                map.insert("max_characters".into(), json!(value));
                            }
                            tools.push(JsonValue::Object(map));
                        }
                        "anthropic.bash_20250124" => {
                            betas.insert("computer-use-2025-01-24".into());
                            tools.push(json!({
                                "type": "bash_20250124",
                                "name": "bash"
                            }));
                        }
                        "anthropic.bash_20241022" => {
                            betas.insert("computer-use-2024-10-22".into());
                            tools.push(json!({
                                "type": "bash_20241022",
                                "name": "bash"
                            }));
                        }
                        "anthropic.memory_20250818" => {
                            betas.insert("context-management-2025-06-27".into());
                            tools.push(json!({
                                "type": "memory_20250818",
                                "name": "memory"
                            }));
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
                            tools.push(JsonValue::Object(map));
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
                            tools.push(JsonValue::Object(map));
                        }
                        "anthropic.tool_search_regex_20251119" => {
                            betas.insert("advanced-tool-use-2025-11-20".into());
                            tools.push(json!({
                                "type": "tool_search_tool_regex_20251119",
                                "name": "tool_search_tool_regex"
                            }));
                        }
                        "anthropic.tool_search_bm25_20251119" => {
                            betas.insert("advanced-tool-use-2025-11-20".into());
                            tools.push(json!({
                                "type": "tool_search_tool_bm25_20251119",
                                "name": "tool_search_tool_bm25"
                            }));
                        }
                        _ => {
                            tool_warnings.push(v2t::CallWarning::UnsupportedTool {
                                tool_name: tool.id.clone(),
                                details: None,
                            });
                        }
                    },
                }
            }
        }

        warnings.extend(tool_warnings.drain(..));

        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "max_tokens": options.max_output_tokens.unwrap_or(1024),
        });
        if let Some(sys) = system {
            body["system"] = JsonValue::Array(sys);
        }
        // tools + tool_choice
        let has_tools = !tools.is_empty();
        if has_tools {
            body["tools"] = JsonValue::Array(tools);
        }
        // toolChoice mapping
        if json_response_tool.is_some() {
            body["tool_choice"] =
                json!({"type":"tool","name":"json","disable_parallel_tool_use": true});
        } else {
            if let Some(tc) = &options.tool_choice {
                match tc {
                    v2t::ToolChoice::Auto => {
                        let mut obj = json!({"type":"auto"});
                        if let Some(p) = provider_opts
                            .as_ref()
                            .and_then(|o| o.disable_parallel_tool_use)
                        {
                            obj.as_object_mut()
                                .unwrap()
                                .insert("disable_parallel_tool_use".into(), json!(p));
                        }
                        body["tool_choice"] = obj;
                    }
                    v2t::ToolChoice::Required => {
                        let mut obj = json!({"type":"any"});
                        if let Some(p) = provider_opts
                            .as_ref()
                            .and_then(|o| o.disable_parallel_tool_use)
                        {
                            obj.as_object_mut()
                                .unwrap()
                                .insert("disable_parallel_tool_use".into(), json!(p));
                        }
                        body["tool_choice"] = obj;
                    }
                    v2t::ToolChoice::Tool { name } => {
                        let mut obj = json!({"type":"tool","name": name});
                        if let Some(p) = provider_opts
                            .as_ref()
                            .and_then(|o| o.disable_parallel_tool_use)
                        {
                            obj.as_object_mut()
                                .unwrap()
                                .insert("disable_parallel_tool_use".into(), json!(p));
                        }
                        body["tool_choice"] = obj;
                    }
                    v2t::ToolChoice::None => {
                        // If tools exist but disabled, drop tools from body
                        if has_tools {
                            body.as_object_mut().map(|m| {
                                m.remove("tools");
                            });
                        }
                    }
                }
            }
        }
        if let Some(t) = options.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(t) = options.top_p {
            body["top_p"] = json!(t);
        }
        if let Some(t) = options.top_k {
            body["top_k"] = json!(t);
        }
        if let Some(stops) = &options.stop_sequences {
            body["stop_sequences"] = json!(stops);
        }

        let thinking_cfg = provider_opts.as_ref().and_then(|o| o.thinking.clone());

        if missing_thinking_reasoning
            && matches!(thinking_cfg, Some(ThinkingOption::Enabled { .. }))
        {
            let assistant_snapshot = options
                .prompt
                .iter()
                .rev()
                .find_map(|m| match m {
                    v2t::PromptMessage::Assistant { content, .. } => Some(content),
                    _ => None,
                })
                .map(|parts| {
                    let reasoning_parts = parts
                        .iter()
                        .filter(|p| matches!(p, v2t::AssistantPart::Reasoning { .. }))
                        .count();
                    let reasoning_with_signature = parts
                        .iter()
                        .filter_map(|p| match p {
                            v2t::AssistantPart::Reasoning {
                                provider_options, ..
                            } => provider_options.as_ref(),
                            _ => None,
                        })
                        .filter(|opts| {
                            let opts = *opts;
                            let scope = opts.get("anthropic").or_else(|| {
                                opts.iter().find_map(|(k, v)| {
                                    if k.eq_ignore_ascii_case("anthropic") {
                                        Some(v)
                                    } else {
                                        None
                                    }
                                })
                            });
                            scope
                                .map(|m| {
                                    m.contains_key("signature")
                                        || m.keys().any(|k| k.eq_ignore_ascii_case("signature"))
                                        || m.contains_key("persistedReasoningSignature")
                                        || m.keys().any(|k| {
                                            k.eq_ignore_ascii_case("persistedReasoningSignature")
                                        })
                                })
                                .unwrap_or(false)
                        })
                        .count();
                    (reasoning_parts, reasoning_with_signature)
                });
            let (reasoning_parts, reasoning_with_signature) = assistant_snapshot.unwrap_or((0, 0));
            tracing::warn!(
                "[THINK_DIAG]: missing reasoning signature; provider={} model={} prompt_messages={} assistant_reasoning_parts={} assistant_reasoning_with_signature={}",
                self.cfg.provider_name,
                self.model_id,
                options.prompt.len(),
                reasoning_parts,
                reasoning_with_signature
            );
            warnings.push(v2t::CallWarning::Other {
                message: "Anthropic thinking is enabled but the latest assistant message has no reasoning content with a signature. The request continues and the upstream API may reject it.".into(),
            });
        }

        if let Some(thinking) = thinking_cfg {
            match thinking {
                ThinkingOption::Enabled { budget_tokens } => {
                    body["thinking"] = json!({"type": "enabled", "budget_tokens": budget_tokens});
                    // Anthropic requires max_tokens strictly greater than thinking.budget_tokens
                    let thinking_max_tokens = (budget_tokens as u64) + 1;
                    body["max_tokens"] = json!(thinking_max_tokens);
                    body.as_object_mut().map(|m| {
                        m.remove("temperature");
                        m.remove("top_p");
                        m.remove("top_k");
                    });
                }
                ThinkingOption::Disabled => {
                    body["thinking"] = json!({"type": "disabled"});
                }
            }
        }

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

        Ok((
            body,
            warnings,
            provider_opts,
            betas,
            json_response_tool.is_some(),
        ))
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
        let (body, warnings, _opts, betas, uses_json_tool) = self.build_request_body(&options)?;
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
                "content_block_delta" => {
                    if v.get("delta")
                        .and_then(|d| d.get("type").and_then(|s| s.as_str()))
                        == Some("text_delta")
                    {
                        if let Some(txt) = v
                            .get("delta")
                            .and_then(|d| d.get("text").and_then(|s| s.as_str()))
                        {
                            out.push(ProviderEvent::TextDelta {
                                delta: txt.to_string(),
                            });
                        }
                    }
                    if v.get("delta")
                        .and_then(|d| d.get("type").and_then(|s| s.as_str()))
                        == Some("thinking_delta")
                    {
                        if let Some(thinking) = v
                            .get("delta")
                            .and_then(|d| d.get("thinking").and_then(|s| s.as_str()))
                        {
                            if !thinking.is_empty() {
                                out.push(ProviderEvent::ReasoningDelta {
                                    delta: thinking.to_string(),
                                });
                            }
                        }
                    }
                    if v.get("delta")
                        .and_then(|d| d.get("type").and_then(|s| s.as_str()))
                        == Some("signature_delta")
                    {
                        if let Some(sig) = v
                            .get("delta")
                            .and_then(|d| d.get("signature").and_then(|s| s.as_str()))
                        {
                            out.push(ProviderEvent::Data {
                                key: "reasoning_signature".to_string(),
                                value: json!({"signature": sig}),
                            });
                        }
                    }
                    if v.get("delta")
                        .and_then(|d| d.get("type").and_then(|s| s.as_str()))
                        == Some("input_json_delta")
                    {
                        let idx = v.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
                        let arg = v
                            .get("delta")
                            .and_then(|d| {
                                d.get("partial_json")
                                    .or_else(|| d.get("json"))
                                    .or_else(|| d.get("delta"))
                            })
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
                }
                "content_block_start" => {
                    let idx = v.get("index").and_then(|i| i.as_u64()).map(|i| i as usize);
                    if let Some(cb) = v.get("content_block") {
                        match cb.get("type").and_then(|s| s.as_str()) {
                            Some("tool_use") => {
                                let id =
                                    cb.get("id").and_then(|s| s.as_str()).map(|s| s.to_string());
                                let name = cb
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
                            Some("thinking") | Some("redacted_thinking") => {
                                let rid = idx
                                    .map(|i| i.to_string())
                                    .unwrap_or_else(|| "0".to_string());
                                out.push(ProviderEvent::ReasoningStart { id: rid });
                            }
                            _ => {}
                        }
                    }
                }
                "message_stop" => {
                    out.extend(self.drain_tool_calls());
                    out.push(ProviderEvent::Done);
                }
                _ => {}
            }
        }
        out
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
