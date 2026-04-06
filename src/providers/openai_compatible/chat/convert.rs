use crate::types::v2 as v2t;
use serde_json::{Map, Value};

fn get_openai_metadata(
    provider_scope: &str,
    map: &Option<v2t::ProviderOptions>,
) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(po) = map {
        if let Some(obj) = po.get(provider_scope) {
            for (k, v) in obj.iter() {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    out
}

fn extend_metadata(target: &mut Map<String, Value>, metadata: Map<String, Value>) {
    for (key, value) in metadata {
        target.insert(key, value);
    }
}

fn object_with_metadata(value: Value, metadata: Map<String, Value>) -> Value {
    let mut object = value
        .as_object()
        .expect("message builders must produce JSON objects")
        .clone();
    extend_metadata(&mut object, metadata);
    Value::Object(object)
}

fn build_system_message(
    provider_scope: &str,
    content: &str,
    provider_options: &Option<v2t::ProviderOptions>,
) -> Value {
    object_with_metadata(
        serde_json::json!({"role":"system","content": content}),
        get_openai_metadata(provider_scope, provider_options),
    )
}

fn build_collapsed_user_text_message(
    provider_scope: &str,
    content: &[v2t::UserPart],
    provider_options: &Option<v2t::ProviderOptions>,
) -> Option<Value> {
    let [v2t::UserPart::Text {
        text,
        provider_options: part_options,
    }] = content
    else {
        return None;
    };

    let mut metadata = get_openai_metadata(provider_scope, provider_options);
    extend_metadata(
        &mut metadata,
        get_openai_metadata(provider_scope, part_options),
    );
    Some(object_with_metadata(
        serde_json::json!({"role":"user","content": text}),
        metadata,
    ))
}

fn user_part_metadata(provider_scope: &str, part: &v2t::UserPart) -> Map<String, Value> {
    match part {
        v2t::UserPart::Text {
            provider_options, ..
        }
        | v2t::UserPart::File {
            provider_options, ..
        } => get_openai_metadata(provider_scope, provider_options),
    }
}

fn file_data_to_url(media_type: &str, data: &v2t::DataContent) -> String {
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;

    match data {
        v2t::DataContent::Url { url } => url.clone(),
        v2t::DataContent::Base64 { base64 } => {
            format!("data:{};base64,{}", media_type, base64)
        }
        v2t::DataContent::Bytes { bytes } => {
            format!("data:{};base64,{}", media_type, B64.encode(bytes))
        }
    }
}

fn convert_user_part(provider_scope: &str, part: &v2t::UserPart) -> Option<Value> {
    let metadata = user_part_metadata(provider_scope, part);
    match part {
        v2t::UserPart::Text { text, .. } => Some(object_with_metadata(
            serde_json::json!({"type":"text","text": text}),
            metadata,
        )),
        v2t::UserPart::File {
            data, media_type, ..
        } => {
            if !media_type.starts_with("image/") {
                return None;
            }
            Some(object_with_metadata(
                serde_json::json!({
                    "type":"image_url",
                    "image_url":{"url": file_data_to_url(media_type, data)}
                }),
                metadata,
            ))
        }
    }
}

fn build_user_message(
    provider_scope: &str,
    content: &[v2t::UserPart],
    provider_options: &Option<v2t::ProviderOptions>,
) -> Value {
    if let Some(message) =
        build_collapsed_user_text_message(provider_scope, content, provider_options)
    {
        return message;
    }

    let content = content
        .iter()
        .filter_map(|part| convert_user_part(provider_scope, part))
        .collect::<Vec<_>>();
    object_with_metadata(
        serde_json::json!({"role":"user","content": content}),
        get_openai_metadata(provider_scope, provider_options),
    )
}

fn tool_call_to_openai(tool_call: &v2t::ToolCallPart) -> Value {
    serde_json::json!({
        "type":"function",
        "id": tool_call.tool_call_id,
        "function": {"name": tool_call.tool_name, "arguments": tool_call.input}
    })
}

fn build_assistant_message(
    provider_scope: &str,
    content: &[v2t::AssistantPart],
    provider_options: &Option<v2t::ProviderOptions>,
) -> Value {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for part in content {
        match part {
            v2t::AssistantPart::Text { text: value, .. }
            | v2t::AssistantPart::Reasoning { text: value, .. } => text.push_str(value),
            v2t::AssistantPart::ToolCall(tool_call) => {
                tool_calls.push(tool_call_to_openai(tool_call));
            }
            v2t::AssistantPart::File { .. } | v2t::AssistantPart::ToolResult(_) => {}
        }
    }

    let mut assistant = serde_json::json!({"role":"assistant"})
        .as_object()
        .expect("assistant seed must be an object")
        .clone();
    if !text.is_empty() {
        assistant.insert("content".into(), Value::String(text));
    }
    if !tool_calls.is_empty() {
        assistant.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    extend_metadata(
        &mut assistant,
        get_openai_metadata(provider_scope, provider_options),
    );
    Value::Object(assistant)
}

fn tool_result_output_to_string(output: &v2t::ToolResultOutput) -> String {
    match output {
        v2t::ToolResultOutput::Text { value } | v2t::ToolResultOutput::ErrorText { value } => {
            value.clone()
        }
        v2t::ToolResultOutput::Json { value } | v2t::ToolResultOutput::ErrorJson { value } => {
            value.to_string()
        }
        v2t::ToolResultOutput::Content { value } => serde_json::json!(value).to_string(),
    }
}

fn build_tool_messages(
    provider_scope: &str,
    content: &[v2t::ToolMessagePart],
    provider_options: &Option<v2t::ProviderOptions>,
) -> Vec<Value> {
    let metadata = get_openai_metadata(provider_scope, provider_options);
    content
        .iter()
        .filter_map(|part| match part {
            v2t::ToolMessagePart::ToolResult(tool_result) => Some(object_with_metadata(
                serde_json::json!({
                    "role":"tool",
                    "tool_call_id": tool_result.tool_call_id,
                    "content": tool_result_output_to_string(&tool_result.output)
                }),
                metadata.clone(),
            )),
            v2t::ToolMessagePart::ToolApprovalResponse(_) => None,
        })
        .collect()
}

fn convert_prompt_message(provider_scope: &str, message: &v2t::PromptMessage) -> Vec<Value> {
    match message {
        v2t::PromptMessage::System {
            content,
            provider_options,
        } => vec![build_system_message(
            provider_scope,
            content,
            provider_options,
        )],
        v2t::PromptMessage::User {
            content,
            provider_options,
        } => vec![build_user_message(
            provider_scope,
            content,
            provider_options,
        )],
        v2t::PromptMessage::Assistant {
            content,
            provider_options,
        } => vec![build_assistant_message(
            provider_scope,
            content,
            provider_options,
        )],
        v2t::PromptMessage::Tool {
            content,
            provider_options,
        } => build_tool_messages(provider_scope, content, provider_options),
    }
}

pub fn convert_to_openai_compatible_chat_messages(
    provider_scope: &str,
    prompt: &v2t::Prompt,
) -> Vec<Value> {
    let mut messages = Vec::new();
    for message in prompt {
        messages.extend(convert_prompt_message(provider_scope, message));
    }
    messages
}
