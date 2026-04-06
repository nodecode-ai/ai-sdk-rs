use std::collections::HashMap;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::core::error::{SdkError, TransportError};
use crate::types::v2 as v2t;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSystemInstruction {
    pub parts: Vec<GoogleTextPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleTextPart {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum GoogleContent {
    #[serde(rename = "user")]
    User { parts: Vec<GoogleContentPart> },
    #[serde(rename = "model")]
    Model { parts: Vec<GoogleContentPart> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GoogleContentPart {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
        thought_signature: Option<String>,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GoogleInlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GoogleFunctionCall,
        #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GoogleFunctionResponse,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: GoogleFileData,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleFileData {
    pub mime_type: String,
    pub file_uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GooglePrompt {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "systemInstruction")]
    pub system_instruction: Option<GoogleSystemInstruction>,
    pub contents: Vec<GoogleContent>,
}

fn to_base64(data: &v2t::DataContent) -> String {
    match data {
        v2t::DataContent::Base64 { base64 } => base64.clone(),
        v2t::DataContent::Bytes { bytes } => STANDARD.encode(bytes),
        v2t::DataContent::Url { .. } => String::new(),
    }
}

fn extract_thought_signature_provider(
    opts: &Option<v2t::ProviderOptions>,
    provider_scopes: &[&str],
) -> Option<String> {
    let map = opts
        .as_ref()
        .and_then(|m| provider_scopes.iter().find_map(|scope| m.get(*scope)))?;
    map.get("thoughtSignature")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn system_message_order_error() -> SdkError {
    SdkError::Transport(TransportError::Other(
        "system messages are only supported at the beginning of the conversation".into(),
    ))
}

fn assistant_file_support_error() -> SdkError {
    SdkError::Transport(TransportError::Other(
        "Only PNG images are supported in assistant messages".into(),
    ))
}

fn assistant_file_url_error() -> SdkError {
    SdkError::Transport(TransportError::Other(
        "File data URLs in assistant messages are not supported".into(),
    ))
}

fn convert_user_parts(content: &[v2t::UserPart]) -> Vec<GoogleContentPart> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            v2t::UserPart::Text { text, .. } => parts.push(GoogleContentPart::Text {
                text: text.clone(),
                thought: None,
                thought_signature: None,
            }),
            v2t::UserPart::File {
                data, media_type, ..
            } => {
                if let v2t::DataContent::Url { url } = data {
                    parts.push(GoogleContentPart::FileData {
                        file_data: GoogleFileData {
                            mime_type: media_type.clone(),
                            file_uri: url.clone(),
                        },
                    });
                } else {
                    let mime_type = if media_type == "image/*" {
                        "image/jpeg".to_string()
                    } else {
                        media_type.clone()
                    };
                    parts.push(GoogleContentPart::InlineData {
                        inline_data: GoogleInlineData {
                            mime_type,
                            data: to_base64(data),
                        },
                    });
                }
            }
        }
    }
    parts
}

fn convert_assistant_text_part(
    text: &str,
    thought: Option<bool>,
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scopes: &[&str],
) -> Option<GoogleContentPart> {
    if text.is_empty() {
        return None;
    }
    Some(GoogleContentPart::Text {
        text: text.to_string(),
        thought,
        thought_signature: extract_thought_signature_provider(provider_options, provider_scopes),
    })
}

fn convert_assistant_file_part(
    media_type: &str,
    data: &v2t::DataContent,
) -> Result<GoogleContentPart, SdkError> {
    if media_type != "image/png" {
        return Err(assistant_file_support_error());
    }
    match data {
        v2t::DataContent::Url { .. } => Err(assistant_file_url_error()),
        _ => Ok(GoogleContentPart::InlineData {
            inline_data: GoogleInlineData {
                mime_type: media_type.to_string(),
                data: to_base64(data),
            },
        }),
    }
}

fn convert_assistant_parts(
    content: &[v2t::AssistantPart],
    provider_scopes: &[&str],
    tool_call_names: &mut HashMap<String, String>,
) -> Result<Vec<GoogleContentPart>, SdkError> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            v2t::AssistantPart::Text {
                text,
                provider_options,
                ..
            } => {
                if let Some(part) =
                    convert_assistant_text_part(text, None, provider_options, provider_scopes)
                {
                    parts.push(part);
                }
            }
            v2t::AssistantPart::Reasoning {
                text,
                provider_options,
                ..
            } => {
                if let Some(part) =
                    convert_assistant_text_part(text, Some(true), provider_options, provider_scopes)
                {
                    parts.push(part);
                }
            }
            v2t::AssistantPart::File {
                media_type, data, ..
            } => parts.push(convert_assistant_file_part(media_type, data)?),
            v2t::AssistantPart::ToolCall(tool_call) => {
                tool_call_names
                    .entry(tool_call.tool_call_id.clone())
                    .or_insert_with(|| tool_call.tool_name.clone());
                let args = serde_json::from_str::<serde_json::Value>(&tool_call.input)
                    .unwrap_or(json!({}));
                parts.push(GoogleContentPart::FunctionCall {
                    function_call: GoogleFunctionCall {
                        name: tool_call.tool_name.clone(),
                        args,
                    },
                    thought_signature: extract_thought_signature_provider(
                        &tool_call.provider_options,
                        provider_scopes,
                    ),
                });
            }
            v2t::AssistantPart::ToolResult(_) => {}
        }
    }
    Ok(parts)
}

fn resolve_tool_name(
    tool_name: &str,
    tool_call_id: &str,
    tool_call_names: &HashMap<String, String>,
) -> String {
    if tool_name.trim().is_empty() {
        tool_call_names
            .get(tool_call_id)
            .cloned()
            .unwrap_or_else(|| tool_call_id.to_string())
    } else {
        tool_name.to_string()
    }
}

fn convert_tool_parts(
    content: &[v2t::ToolMessagePart],
    tool_call_names: &mut HashMap<String, String>,
) -> Vec<GoogleContentPart> {
    let mut parts = Vec::new();
    for part in content {
        let part = match part {
            v2t::ToolMessagePart::ToolResult(part) => part,
            v2t::ToolMessagePart::ToolApprovalResponse(_) => continue,
        };
        if !part.tool_name.is_empty() {
            tool_call_names
                .entry(part.tool_call_id.clone())
                .or_insert_with(|| part.tool_name.clone());
        }
        let resolved_tool_name =
            resolve_tool_name(&part.tool_name, &part.tool_call_id, tool_call_names);
        let payload = match &part.output {
            v2t::ToolResultOutput::Text { value } => json!(value),
            v2t::ToolResultOutput::Json { value } => json!(value),
            v2t::ToolResultOutput::ErrorText { value } => json!(value),
            v2t::ToolResultOutput::ErrorJson { value } => json!(value),
            v2t::ToolResultOutput::Content { value } => json!(value),
        };
        parts.push(GoogleContentPart::FunctionResponse {
            function_response: GoogleFunctionResponse {
                name: resolved_tool_name.clone(),
                response: json!({
                    "name": resolved_tool_name,
                    "content": payload
                }),
            },
        });
    }
    parts
}

fn apply_gemma_system_instruction(
    is_gemma: bool,
    system_instruction_parts: &[GoogleTextPart],
    contents: &mut [GoogleContent],
) {
    if !is_gemma || system_instruction_parts.is_empty() || contents.is_empty() {
        return;
    }
    if let Some(GoogleContent::User { parts }) = contents.get_mut(0) {
        let system_text = system_instruction_parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        parts.insert(
            0,
            GoogleContentPart::Text {
                text: format!("{system_text}\n\n"),
                thought: None,
                thought_signature: None,
            },
        );
    }
}

pub fn convert_to_google_prompt_with_scopes(
    prompt: &v2t::Prompt,
    is_gemma: bool,
    provider_scopes: &[&str],
) -> Result<GooglePrompt, SdkError> {
    let mut system_instruction_parts: Vec<GoogleTextPart> = Vec::new();
    let mut contents: Vec<GoogleContent> = Vec::new();
    let mut system_allowed = true;
    let mut tool_call_names: HashMap<String, String> = HashMap::new();

    for msg in prompt {
        match msg {
            v2t::PromptMessage::System { content, .. } => {
                if !system_allowed {
                    return Err(system_message_order_error());
                }
                system_instruction_parts.push(GoogleTextPart {
                    text: content.clone(),
                });
            }
            v2t::PromptMessage::User { content, .. } => {
                system_allowed = false;
                contents.push(GoogleContent::User {
                    parts: convert_user_parts(content),
                });
            }
            v2t::PromptMessage::Assistant { content, .. } => {
                system_allowed = false;
                contents.push(GoogleContent::Model {
                    parts: convert_assistant_parts(content, provider_scopes, &mut tool_call_names)?,
                });
            }
            v2t::PromptMessage::Tool { content, .. } => {
                system_allowed = false;
                contents.push(GoogleContent::User {
                    parts: convert_tool_parts(content, &mut tool_call_names),
                });
            }
        }
    }

    apply_gemma_system_instruction(is_gemma, &system_instruction_parts, &mut contents);

    let system_instruction = if !system_instruction_parts.is_empty() && !is_gemma {
        Some(GoogleSystemInstruction {
            parts: system_instruction_parts,
        })
    } else {
        None
    };

    Ok(GooglePrompt {
        system_instruction,
        contents,
    })
}
