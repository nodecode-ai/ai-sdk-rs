use crate::core::SdkError;
use crate::types::v2 as v2t;
use base64::Engine as _;
use serde_json::{json, Value as JsonValue};

use crate::providers::amazon_bedrock::options::{has_cache_point, parse_reasoning_metadata};

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
            Block::System(msgs) => append_system_block(msgs, !messages.is_empty(), &mut system)?,
            Block::User(msgs) => append_user_block(msgs, &mut messages)?,
            Block::Assistant(msgs) => {
                append_assistant_block(block_index, blocks.len(), msgs, &mut messages)?
            }
        }
    }

    Ok(ConvertedPrompt { system, messages })
}

fn append_system_block(
    msgs: &[&v2t::PromptMessage],
    has_messages: bool,
    system: &mut Vec<JsonValue>,
) -> Result<(), SdkError> {
    if has_messages {
        return Err(unsupported(
            "Multiple system message blocks separated by other roles are not supported by Amazon Bedrock",
        ));
    }

    for message in msgs {
        if let v2t::PromptMessage::System {
            content,
            provider_options,
        } = message
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

    Ok(())
}

fn append_user_block(
    msgs: &[&v2t::PromptMessage],
    messages: &mut Vec<JsonValue>,
) -> Result<(), SdkError> {
    let mut content_blocks = Vec::new();
    for message in msgs {
        match message {
            v2t::PromptMessage::User {
                content,
                provider_options,
            } => {
                append_user_parts(content, &mut content_blocks)?;
                if has_cache_point(provider_options) {
                    content_blocks.push(cache_point_value());
                }
            }
            v2t::PromptMessage::Tool {
                content,
                provider_options,
            } => {
                append_tool_results(content, &mut content_blocks)?;
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
    Ok(())
}

fn append_user_parts(
    content: &[v2t::UserPart],
    content_blocks: &mut Vec<JsonValue>,
) -> Result<(), SdkError> {
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
                content_blocks.push(user_file_part_to_block(
                    filename.as_ref(),
                    data,
                    media_type,
                    content_blocks.len() + 1,
                )?);
            }
        }
    }
    Ok(())
}

fn user_file_part_to_block(
    filename: Option<&String>,
    data: &v2t::DataContent,
    media_type: &str,
    index: usize,
) -> Result<JsonValue, SdkError> {
    if matches!(data, v2t::DataContent::Url { .. }) {
        return Err(unsupported(
            "Bedrock does not support file parts by URL; provide inlined bytes or base64",
        ));
    }
    if media_type.is_empty() {
        return Err(unsupported(
            "File message parts require a MIME type for Amazon Bedrock",
        ));
    }

    let bytes = data_content_to_base64(data)?;
    if media_type.starts_with("image/") {
        let format = bedrock_image_format(media_type)?;
        Ok(json!({
            "image": {
                "format": format,
                "source": { "bytes": bytes },
            }
        }))
    } else {
        let format = bedrock_document_format(media_type)?;
        let name = filename
            .cloned()
            .unwrap_or_else(|| format!("document-{index}"));
        Ok(json!({
            "document": {
                "format": format,
                "name": name,
                "source": { "bytes": bytes },
            }
        }))
    }
}

fn append_tool_results(
    content: &[v2t::ToolMessagePart],
    content_blocks: &mut Vec<JsonValue>,
) -> Result<(), SdkError> {
    for part in content {
        let v2t::ToolMessagePart::ToolResult(part) = part else {
            continue;
        };
        content_blocks.push(tool_result_to_block(part)?);
    }
    Ok(())
}

fn tool_result_to_block(part: &v2t::ToolResultPart) -> Result<JsonValue, SdkError> {
    let mut tool_content = Vec::new();
    match &part.output {
        v2t::ToolResultOutput::Content { value } => {
            for item in value {
                match item {
                    v2t::ToolResultInlineContent::Text { text } => {
                        tool_content.push(json!({"text": text}));
                    }
                    v2t::ToolResultInlineContent::Media { data, media_type } => {
                        if !media_type.starts_with("image/") {
                            return Err(unsupported(&format!(
                                "Unsupported media type in tool result: {}",
                                media_type
                            )));
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
        v2t::ToolResultOutput::Text { value } | v2t::ToolResultOutput::ErrorText { value } => {
            tool_content.push(json!({ "text": value }));
        }
        v2t::ToolResultOutput::Json { value } | v2t::ToolResultOutput::ErrorJson { value } => {
            tool_content.push(json!({ "text": value.to_string() }));
        }
    }

    Ok(json!({
        "toolResult": {
            "toolUseId": part.tool_call_id,
            "content": tool_content,
        }
    }))
}

fn append_assistant_block(
    block_index: usize,
    total_blocks: usize,
    msgs: &[&v2t::PromptMessage],
    messages: &mut Vec<JsonValue>,
) -> Result<(), SdkError> {
    let mut content_blocks = Vec::new();
    for (message_idx, message) in msgs.iter().enumerate() {
        if let v2t::PromptMessage::Assistant {
            content,
            provider_options,
        } = message
        {
            append_assistant_parts(
                block_index,
                total_blocks,
                message_idx,
                msgs.len(),
                content,
                &mut content_blocks,
            )?;
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
    Ok(())
}

fn append_assistant_parts(
    block_index: usize,
    total_blocks: usize,
    message_idx: usize,
    total_messages: usize,
    content: &[v2t::AssistantPart],
    content_blocks: &mut Vec<JsonValue>,
) -> Result<(), SdkError> {
    for (part_idx, part) in content.iter().enumerate() {
        match part {
            v2t::AssistantPart::Text { text, .. } => {
                let trimmed = trim_if_last(
                    block_index,
                    message_idx,
                    part_idx,
                    total_blocks,
                    total_messages,
                    content.len(),
                    text,
                );
                if !trimmed.is_empty() {
                    content_blocks.push(json!({ "text": trimmed }));
                }
            }
            v2t::AssistantPart::Reasoning {
                text,
                provider_options,
            } => {
                let trimmed = trim_if_last(
                    block_index,
                    message_idx,
                    part_idx,
                    total_blocks,
                    total_messages,
                    content.len(),
                    text,
                );
                if let Some(block) = assistant_reasoning_block(&trimmed, provider_options) {
                    content_blocks.push(block);
                }
            }
            v2t::AssistantPart::ToolCall(part) => {
                let input = if part.input.trim().is_empty() {
                    JsonValue::Null
                } else {
                    serde_json::from_str(&part.input)
                        .unwrap_or_else(|_| JsonValue::String(part.input.clone()))
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
                return Err(unsupported(
                    "Assistant file content is not supported when pre-filling Bedrock conversations",
                ));
            }
            v2t::AssistantPart::ToolResult(_) => {}
        }
    }
    Ok(())
}

fn assistant_reasoning_block(
    text: &str,
    provider_options: &Option<v2t::ProviderOptions>,
) -> Option<JsonValue> {
    if let Some(meta) = parse_reasoning_metadata(provider_options) {
        if let Some(sig) = meta.signature {
            return Some(json!({
                "reasoningContent": {
                    "reasoningText": {
                        "text": text,
                        "signature": sig,
                    }
                }
            }));
        }
        if let Some(redacted) = meta.redacted_data {
            return Some(json!({
                "reasoningContent": {
                    "redactedReasoning": {
                        "data": redacted,
                    }
                }
            }));
        }
    }

    if text.is_empty() {
        None
    } else {
        Some(json!({
            "reasoningContent": {
                "reasoningText": {
                    "text": text,
                }
            }
        }))
    }
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
