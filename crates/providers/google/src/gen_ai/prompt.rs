use std::collections::HashMap;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_types::v2 as v2t;

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

fn extract_thought_signature_provider(opts: &Option<v2t::ProviderOptions>) -> Option<String> {
    let map = opts.as_ref()?.get("google")?;
    map.get("thoughtSignature")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Convert V2 prompt into Google GenAI prompt shape.
pub fn convert_to_google_prompt(
    prompt: &v2t::Prompt,
    is_gemma: bool,
) -> Result<GooglePrompt, SdkError> {
    let mut system_instruction_parts: Vec<GoogleTextPart> = Vec::new();
    let mut contents: Vec<GoogleContent> = Vec::new();
    let mut system_allowed = true;
    let mut tool_call_names: HashMap<String, String> = HashMap::new();

    for msg in prompt {
        match msg {
            v2t::PromptMessage::System { content, .. } => {
                if !system_allowed {
                    return Err(SdkError::Transport(TransportError::Other(
                        "system messages are only supported at the beginning of the conversation"
                            .into(),
                    )));
                }
                system_instruction_parts.push(GoogleTextPart {
                    text: content.clone(),
                });
            }
            v2t::PromptMessage::User { content, .. } => {
                system_allowed = false;
                let mut parts: Vec<GoogleContentPart> = Vec::new();
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
                                let mime = if media_type == "image/*" {
                                    "image/jpeg".to_string()
                                } else {
                                    media_type.clone()
                                };
                                parts.push(GoogleContentPart::InlineData {
                                    inline_data: GoogleInlineData {
                                        mime_type: mime,
                                        data: to_base64(data),
                                    },
                                });
                            }
                        }
                    }
                }
                contents.push(GoogleContent::User { parts });
            }
            v2t::PromptMessage::Assistant {
                content,
                provider_options: _provider_options,
            } => {
                system_allowed = false;
                let mut parts: Vec<GoogleContentPart> = Vec::new();
                for part in content {
                    match part {
                        v2t::AssistantPart::Text {
                            text,
                            provider_options: popts,
                            ..
                        } => {
                            if !text.is_empty() {
                                parts.push(GoogleContentPart::Text {
                                    text: text.clone(),
                                    thought: None,
                                    thought_signature: extract_thought_signature_provider(popts),
                                });
                            }
                        }
                        v2t::AssistantPart::Reasoning {
                            text,
                            provider_options: popts,
                            ..
                        } => {
                            if !text.is_empty() {
                                parts.push(GoogleContentPart::Text {
                                    text: text.clone(),
                                    thought: Some(true),
                                    thought_signature: extract_thought_signature_provider(popts),
                                });
                            }
                        }
                        v2t::AssistantPart::File {
                            media_type, data, ..
                        } => {
                            if media_type != "image/png" {
                                return Err(SdkError::Transport(TransportError::Other(
                                    "Only PNG images are supported in assistant messages".into(),
                                )));
                            }
                            match data {
                                v2t::DataContent::Url { .. } => {
                                    return Err(SdkError::Transport(TransportError::Other(
                                        "File data URLs in assistant messages are not supported"
                                            .into(),
                                    )));
                                }
                                _ => {
                                    parts.push(GoogleContentPart::InlineData {
                                        inline_data: GoogleInlineData {
                                            mime_type: media_type.clone(),
                                            data: to_base64(data),
                                        },
                                    });
                                }
                            }
                        }
                        v2t::AssistantPart::ToolCall(tc) => {
                            tool_call_names
                                .entry(tc.tool_call_id.clone())
                                .or_insert_with(|| tc.tool_name.clone());
                            let args = serde_json::from_str::<serde_json::Value>(&tc.input)
                                .unwrap_or(json!({}));
                            parts.push(GoogleContentPart::FunctionCall {
                                function_call: GoogleFunctionCall {
                                    name: tc.tool_name.clone(),
                                    args,
                                },
                                thought_signature: extract_thought_signature_provider(
                                    &tc.provider_options,
                                ),
                            });
                        }
                        v2t::AssistantPart::ToolResult(_tr) => { /* not represented in assistant block */
                        }
                    }
                }
                contents.push(GoogleContent::Model { parts });
            }
            v2t::PromptMessage::Tool { content, .. } => {
                system_allowed = false;
                let mut parts: Vec<GoogleContentPart> = Vec::new();
                for part in content {
                    let part = match part {
                        v2t::ToolMessagePart::ToolResult(part) => part,
                        v2t::ToolMessagePart::ToolApprovalResponse(_) => {
                            continue;
                        }
                    };
                    if !part.tool_name.is_empty() {
                        tool_call_names
                            .entry(part.tool_call_id.clone())
                            .or_insert_with(|| part.tool_name.clone());
                    }
                    let resolved_tool_name = if part.tool_name.trim().is_empty() {
                        tool_call_names
                            .get(&part.tool_call_id)
                            .cloned()
                            .unwrap_or_else(|| part.tool_call_id.clone())
                    } else {
                        part.tool_name.clone()
                    };
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
                contents.push(GoogleContent::User { parts });
            }
        }
    }

    // Gemma: push system text into first user as prefix
    if is_gemma && !system_instruction_parts.is_empty() && !contents.is_empty() {
        if let Some(GoogleContent::User { parts }) = contents.get_mut(0) {
            let sys_text = system_instruction_parts
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            parts.insert(
                0,
                GoogleContentPart::Text {
                    text: format!("{}\n\n", sys_text),
                    thought: None,
                    thought_signature: None,
                },
            );
        }
    }

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
