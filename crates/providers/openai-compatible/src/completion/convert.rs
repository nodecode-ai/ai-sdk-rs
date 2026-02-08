use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_types::v2 as v2t;

/// Convert a V2 prompt into an OpenAI-compatible completion prompt string.
/// Follows the TS implementation semantics.
pub fn convert_to_openai_compatible_completion_prompt(
    prompt: &v2t::Prompt,
    user_label: &str,
    assistant_label: &str,
) -> Result<(String, Option<Vec<String>>), SdkError> {
    let mut text = String::new();

    let mut idx = 0usize;
    if let Some(v2t::PromptMessage::System { content, .. }) = prompt.get(0) {
        text.push_str(content);
        text.push_str("\n\n");
        idx = 1;
    }

    for m in &prompt[idx..] {
        match m {
            v2t::PromptMessage::System { content, .. } => {
                return Err(SdkError::Upstream {
                    status: 400,
                    message: format!("Unexpected system message in prompt: {}", content),
                    source: None,
                });
            }
            v2t::PromptMessage::User { content, .. } => {
                let mut buf = String::new();
                for part in content {
                    match part {
                        v2t::UserPart::Text { text, .. } => buf.push_str(text),
                        v2t::UserPart::File { .. } => {
                            return Err(SdkError::Upstream {
                                status: 400,
                                message:
                                    "Unsupported functionality: file parts in completion prompt"
                                        .into(),
                                source: None,
                            });
                        }
                    }
                }
                text.push_str(user_label);
                text.push_str(":\n");
                text.push_str(&buf);
                text.push_str("\n\n");
            }
            v2t::PromptMessage::Assistant { content, .. } => {
                let mut buf = String::new();
                for part in content {
                    match part {
                        v2t::AssistantPart::Text { text, .. } => buf.push_str(text),
                        v2t::AssistantPart::Reasoning { text, .. } => buf.push_str(text),
                        v2t::AssistantPart::File { .. } => {
                            return Err(SdkError::Upstream {
                                status: 400,
                                message:
                                    "Unsupported functionality: file parts in assistant message"
                                        .into(),
                                source: None,
                            });
                        }
                        v2t::AssistantPart::ToolCall(_) | v2t::AssistantPart::ToolResult(_) => {
                            return Err(SdkError::Upstream { status: 400, message: "Unsupported functionality: tool-call messages in completion prompt".into(), source: None });
                        }
                    }
                }
                text.push_str(assistant_label);
                text.push_str(":\n");
                text.push_str(&buf);
                text.push_str("\n\n");
            }
            v2t::PromptMessage::Tool { .. } => {
                return Err(SdkError::Upstream {
                    status: 400,
                    message: "Unsupported functionality: tool messages in completion prompt".into(),
                    source: None,
                });
            }
        }
    }

    // Assistant message prefix for the model to complete
    text.push_str(assistant_label);
    text.push_str(":\n");

    let stop_sequences = vec![format!("\n{}:", user_label)];
    Ok((text, Some(stop_sequences)))
}
