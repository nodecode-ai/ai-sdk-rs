use crate::ai_sdk_core::SdkError;
use crate::ai_sdk_types::v2 as v2t;
use base64::Engine as _;
use serde_json::{json, Value as JsonValue};

use crate::provider_amazon_bedrock::options::{has_cache_point, parse_reasoning_metadata};

#[derive(Debug, Clone)]
pub struct ConvertedPrompt {
    pub system: Vec<JsonValue>,
    pub messages: Vec<JsonValue>,
}

pub fn convert_prompt(prompt: &[v2t::PromptMessage]) -> Result<ConvertedPrompt, SdkError> {
    let blocks = group_into_blocks(prompt);

    let mut system: Vec<JsonValue> = Vec::new();
    let mut messages: Vec<JsonValue> = Vec::new();

    for (block_index, block) in blocks.iter().enumerate() {
        match block {
            Block::System(msgs) => {
                if !messages.is_empty() {
                    return Err(unsupported(
                        "Multiple system message blocks separated by other roles are not supported by Amazon Bedrock",
                    ));
                }
                for m in msgs {
                    if let v2t::PromptMessage::System {
                        content,
                        provider_options,
                    } = m
                    {
                        if content.trim().is_empty() {
                            continue;
                        }
                        system.push(json!({ "text": content }));
                        if has_cache_point(provider_options) {
                            system.push(cache_point_value());
                        }
                    }
                }
            }
            Block::User(msgs) => {
                let mut content_blocks: Vec<JsonValue> = Vec::new();
                for message in msgs {
                    match message {
                        v2t::PromptMessage::User {
                            content,
                            provider_options,
                        } => {
                            for part in content {
                                match part {
                                    v2t::UserPart::Text { text, .. } => {
                                        if !text.is_empty() {
                                            content_blocks.push(json!({ "text": text }));
                                        }
                                    }
                                    v2t::UserPart::File {
                                        filename,
                                        data,
                                        media_type,
                                        ..
                                    } => {
                                        if matches!(data, v2t::DataContent::Url { .. }) {
                                            return Err(unsupported(
                                                "Bedrock does not support file parts by URL; provide inlined bytes or base64",
                                            ));
                                        }
                                        let mime = media_type.clone();
                                        if mime.is_empty() {
                                            return Err(unsupported(
                                                "File message parts require a MIME type for Amazon Bedrock",
                                            ));
                                        }
                                        let bytes = data_content_to_base64(data)?;
                                        if mime.starts_with("image/") {
                                            let format = bedrock_image_format(&mime)?;
                                            content_blocks.push(json!({
                                                "image": {
                                                    "format": format,
                                                    "source": { "bytes": bytes },
                                                }
                                            }));
                                        } else {
                                            let format = bedrock_document_format(&mime)?;
                                            let name = filename.clone().unwrap_or_else(|| {
                                                // Bedrock expects a name; synthesize deterministic placeholder
                                                format!("document-{}", content_blocks.len() + 1)
                                            });
                                            content_blocks.push(json!({
                                                "document": {
                                                    "format": format,
                                                    "name": name,
                                                    "source": { "bytes": bytes },
                                                }
                                            }));
                                        }
                                    }
                                }
                            }
                            if has_cache_point(provider_options) {
                                content_blocks.push(cache_point_value());
                            }
                        }
                        v2t::PromptMessage::Tool {
                            content,
                            provider_options,
                        } => {
                            for part in content {
                                let part = match part {
                                    v2t::ToolMessagePart::ToolResult(part) => part,
                                    v2t::ToolMessagePart::ToolApprovalResponse(_) => {
                                        continue;
                                    }
                                };
                                let mut tool_content: Vec<JsonValue> = Vec::new();
                                match &part.output {
                                    v2t::ToolResultOutput::Content { value } => {
                                        for item in value {
                                            match item {
                                                v2t::ToolResultInlineContent::Text { text } => {
                                                    tool_content.push(json!({"text": text}));
                                                }
                                                v2t::ToolResultInlineContent::Media {
                                                    data,
                                                    media_type,
                                                } => {
                                                    if !media_type.starts_with("image/") {
                                                        return Err(unsupported(
                                                            &format!(
                                                                "Unsupported media type in tool result: {}",
                                                                media_type
                                                            ),
                                                        ));
                                                    }
                                                    tool_content.push(json!({
                                                        "image": {
                                                            "format": bedrock_image_format(media_type)?,
                                                            "source": { "bytes": data },
                                                        }
                                                    }));
                                                }
                                            }
                                        }
                                    }
                                    v2t::ToolResultOutput::Text { value }
                                    | v2t::ToolResultOutput::ErrorText { value } => {
                                        tool_content.push(json!({ "text": value }));
                                    }
                                    v2t::ToolResultOutput::Json { value }
                                    | v2t::ToolResultOutput::ErrorJson { value } => {
                                        tool_content.push(json!({ "text": value.to_string() }));
                                    }
                                }
                                content_blocks.push(json!({
                                    "toolResult": {
                                        "toolUseId": part.tool_call_id,
                                        "content": tool_content,
                                    }
                                }));
                            }
                            if has_cache_point(provider_options) {
                                content_blocks.push(cache_point_value());
                            }
                        }
                        _ => {}
                    }
                }
                if !content_blocks.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": content_blocks,
                    }));
                }
            }
            Block::Assistant(msgs) => {
                let mut content_blocks: Vec<JsonValue> = Vec::new();
                for (message_idx, message) in msgs.iter().enumerate() {
                    if let v2t::PromptMessage::Assistant {
                        content,
                        provider_options,
                    } = message
                    {
                        for (part_idx, part) in content.iter().enumerate() {
                            match part {
                                v2t::AssistantPart::Text { text, .. } => {
                                    if text.trim().is_empty() {
                                        continue;
                                    }
                                    let trimmed = trim_if_last(
                                        block_index,
                                        message_idx,
                                        part_idx,
                                        blocks.len(),
                                        msgs.len(),
                                        content.len(),
                                        text,
                                    );
                                    if trimmed.is_empty() {
                                        continue;
                                    }
                                    content_blocks.push(json!({
                                        "text": trimmed,
                                    }));
                                }
                                v2t::AssistantPart::Reasoning {
                                    text,
                                    provider_options,
                                } => {
                                    let trimmed = trim_if_last(
                                        block_index,
                                        message_idx,
                                        part_idx,
                                        blocks.len(),
                                        msgs.len(),
                                        content.len(),
                                        text,
                                    );
                                    if let Some(meta) = parse_reasoning_metadata(provider_options) {
                                        if let Some(sig) = meta.signature {
                                            content_blocks.push(json!({
                                                "reasoningContent": {
                                                    "reasoningText": {
                                                        "text": trimmed,
                                                        "signature": sig,
                                                    }
                                                }
                                            }));
                                            continue;
                                        }
                                        if let Some(redacted) = meta.redacted_data {
                                            content_blocks.push(json!({
                                                "reasoningContent": {
                                                    "redactedReasoning": {
                                                        "data": redacted,
                                                    }
                                                }
                                            }));
                                            continue;
                                        }
                                    }
                                    if !trimmed.is_empty() {
                                        content_blocks.push(json!({
                                            "reasoningContent": {
                                                "reasoningText": {
                                                    "text": trimmed,
                                                }
                                            }
                                        }));
                                    }
                                }
                                v2t::AssistantPart::ToolCall(part) => {
                                    let input = if part.input.trim().is_empty() {
                                        JsonValue::Null
                                    } else {
                                        serde_json::from_str(&part.input).unwrap_or_else(|_| {
                                            JsonValue::String(part.input.clone())
                                        })
                                    };
                                    content_blocks.push(json!({
                                        "toolUse": {
                                            "toolUseId": part.tool_call_id,
                                            "name": part.tool_name,
                                            "input": input,
                                        }
                                    }));
                                }
                                v2t::AssistantPart::File { .. } => {
                                    // Assistant file parts are not supported for prefill today.
                                    return Err(unsupported(
                                        "Assistant file content is not supported when pre-filling Bedrock conversations",
                                    ));
                                }
                                v2t::AssistantPart::ToolResult(_) => {
                                    // Not expected in assistant messages.
                                }
                            }
                        }
                        if has_cache_point(provider_options) {
                            content_blocks.push(cache_point_value());
                        }
                    }
                }
                if !content_blocks.is_empty() {
                    messages.push(json!({
                        "role": "assistant",
                        "content": content_blocks,
                    }));
                }
            }
        }
    }

    Ok(ConvertedPrompt { system, messages })
}

enum Block<'a> {
    System(Vec<&'a v2t::PromptMessage>),
    Assistant(Vec<&'a v2t::PromptMessage>),
    User(Vec<&'a v2t::PromptMessage>),
}

fn group_into_blocks(prompt: &[v2t::PromptMessage]) -> Vec<Block<'_>> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut current: Option<Block> = None;

    for message in prompt {
        match message {
            v2t::PromptMessage::System { .. } => match current.as_mut() {
                Some(Block::System(msgs)) => msgs.push(message),
                _ => {
                    if let Some(b) = current.take() {
                        blocks.push(b);
                    }
                    current = Some(Block::System(vec![message]));
                }
            },
            v2t::PromptMessage::Assistant { .. } => match current.as_mut() {
                Some(Block::Assistant(msgs)) => msgs.push(message),
                _ => {
                    if let Some(b) = current.take() {
                        blocks.push(b);
                    }
                    current = Some(Block::Assistant(vec![message]));
                }
            },
            v2t::PromptMessage::User { .. } | v2t::PromptMessage::Tool { .. } => {
                match current.as_mut() {
                    Some(Block::User(msgs)) => msgs.push(message),
                    _ => {
                        if let Some(b) = current.take() {
                            blocks.push(b);
                        }
                        current = Some(Block::User(vec![message]));
                    }
                }
            }
        }
    }

    if let Some(b) = current {
        blocks.push(b);
    }

    blocks
}

fn cache_point_value() -> JsonValue {
    json!({ "cachePoint": { "type": "default" } })
}

fn data_content_to_base64(data: &v2t::DataContent) -> Result<String, SdkError> {
    Ok(match data {
        v2t::DataContent::Base64 { base64 } => base64.clone(),
        v2t::DataContent::Bytes { bytes } => {
            base64::engine::general_purpose::STANDARD.encode(bytes)
        }
        v2t::DataContent::Url { .. } => {
            return Err(unsupported(
                "Amazon Bedrock does not support file content by URL references",
            ))
        }
    })
}

fn bedrock_image_format(mime: &str) -> Result<&'static str, SdkError> {
    match mime {
        "image/jpeg" => Ok("jpeg"),
        "image/png" => Ok("png"),
        "image/gif" => Ok("gif"),
        "image/webp" => Ok("webp"),
        other => Err(unsupported(&format!(
            "Unsupported image MIME type for Amazon Bedrock: {}",
            other
        ))),
    }
}

fn bedrock_document_format(mime: &str) -> Result<&'static str, SdkError> {
    match mime {
        "application/pdf" => Ok("pdf"),
        "text/csv" => Ok("csv"),
        "application/msword" => Ok("doc"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Ok("docx"),
        "application/vnd.ms-excel" => Ok("xls"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Ok("xlsx"),
        "text/html" => Ok("html"),
        "text/plain" => Ok("txt"),
        "text/markdown" => Ok("md"),
        other => Err(unsupported(&format!(
            "Unsupported document MIME type for Amazon Bedrock: {}",
            other
        ))),
    }
}

fn trim_if_last(
    block_index: usize,
    message_index: usize,
    part_index: usize,
    total_blocks: usize,
    block_messages: usize,
    message_parts: usize,
    text: &str,
) -> String {
    let is_last_block = block_index + 1 == total_blocks;
    let is_last_message = message_index + 1 == block_messages;
    let is_last_part = part_index + 1 == message_parts;
    if is_last_block && is_last_message && is_last_part {
        text.trim().to_string()
    } else {
        text.to_string()
    }
}

fn unsupported(message: &str) -> SdkError {
    SdkError::Upstream {
        status: 400,
        message: message.to_string(),
        source: None,
    }
}
