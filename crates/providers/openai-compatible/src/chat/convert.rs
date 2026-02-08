use crate::ai_sdk_types::v2 as v2t;

fn get_openai_metadata(
    provider_scope: &str,
    map: &Option<v2t::ProviderOptions>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut out = serde_json::Map::new();
    if let Some(po) = map {
        if let Some(obj) = po.get(provider_scope) {
            for (k, v) in obj.iter() {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    out
}

pub fn convert_to_openai_compatible_chat_messages(
    provider_scope: &str,
    prompt: &v2t::Prompt,
) -> Vec<serde_json::Value> {
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;
    let mut messages: Vec<serde_json::Value> = Vec::new();
    for m in prompt {
        match m {
            v2t::PromptMessage::System {
                content,
                provider_options,
            } => {
                let meta = get_openai_metadata(provider_scope, provider_options);
                let mut obj = serde_json::json!({"role":"system","content": content})
                    .as_object()
                    .unwrap()
                    .clone();
                for (k, v) in meta {
                    obj.insert(k, v);
                }
                messages.push(serde_json::Value::Object(obj));
            }
            v2t::PromptMessage::User {
                content,
                provider_options,
            } => {
                if content.len() == 1 {
                    if let v2t::UserPart::Text {
                        text,
                        provider_options: part_opts,
                    } = &content[0]
                    {
                        let mut meta = get_openai_metadata(provider_scope, provider_options);
                        // merge part metadata too
                        let part_meta = get_openai_metadata(provider_scope, part_opts);
                        for (k, v) in part_meta {
                            meta.insert(k, v);
                        }
                        let mut obj = serde_json::json!({"role":"user","content": text})
                            .as_object()
                            .unwrap()
                            .clone();
                        for (k, v) in meta {
                            obj.insert(k, v);
                        }
                        messages.push(serde_json::Value::Object(obj));
                        continue;
                    }
                }

                let mut parts_json: Vec<serde_json::Value> = Vec::new();
                for part in content {
                    let meta = match part {
                        v2t::UserPart::Text {
                            provider_options, ..
                        } => get_openai_metadata(provider_scope, provider_options),
                        v2t::UserPart::File {
                            provider_options, ..
                        } => get_openai_metadata(provider_scope, provider_options),
                    };
                    match part {
                        v2t::UserPart::Text { text, .. } => {
                            let mut obj = serde_json::json!({"type":"text","text": text})
                                .as_object()
                                .unwrap()
                                .clone();
                            for (k, v) in meta {
                                obj.insert(k, v);
                            }
                            parts_json.push(serde_json::Value::Object(obj));
                        }
                        v2t::UserPart::File {
                            filename: _,
                            data,
                            media_type,
                            ..
                        } => {
                            if media_type.starts_with("image/") {
                                let url_val = match data {
                                    v2t::DataContent::Url { url } => url.clone(),
                                    v2t::DataContent::Base64 { base64 } => {
                                        format!("data:{};base64,{}", media_type, base64)
                                    }
                                    v2t::DataContent::Bytes { bytes } => {
                                        format!("data:{};base64,{}", media_type, B64.encode(bytes))
                                    }
                                };
                                let mut obj = serde_json::json!({"type":"image_url","image_url":{"url": url_val}}).as_object().unwrap().clone();
                                for (k, v) in meta {
                                    obj.insert(k, v);
                                }
                                parts_json.push(serde_json::Value::Object(obj));
                            }
                        }
                    }
                }
                let mut obj = serde_json::json!({"role":"user","content": parts_json})
                    .as_object()
                    .unwrap()
                    .clone();
                let meta = get_openai_metadata(provider_scope, provider_options);
                for (k, v) in meta {
                    obj.insert(k, v);
                }
                messages.push(serde_json::Value::Object(obj));
            }
            v2t::PromptMessage::Assistant {
                content,
                provider_options,
            } => {
                let mut text = String::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();
                for part in content {
                    match part {
                        v2t::AssistantPart::Text { text: t, .. } => text.push_str(t),
                        v2t::AssistantPart::Reasoning { text: t, .. } => text.push_str(t),
                        v2t::AssistantPart::File { .. } => {}
                        v2t::AssistantPart::ToolCall(tc) => {
                            let args_str = tc.input.clone();
                            tool_calls.push(serde_json::json!({
                                "type":"function",
                                "id": tc.tool_call_id,
                                "function": {"name": tc.tool_name, "arguments": args_str}
                            }));
                        }
                        v2t::AssistantPart::ToolResult(_) => {}
                    }
                }
                let mut obj = serde_json::json!({"role":"assistant"})
                    .as_object()
                    .unwrap()
                    .clone();
                if !text.is_empty() {
                    obj.insert("content".into(), serde_json::Value::String(text));
                }
                if !tool_calls.is_empty() {
                    obj.insert("tool_calls".into(), serde_json::Value::Array(tool_calls));
                }
                let meta = get_openai_metadata(provider_scope, provider_options);
                for (k, v) in meta {
                    obj.insert(k, v);
                }
                messages.push(serde_json::Value::Object(obj));
            }
            v2t::PromptMessage::Tool {
                content,
                provider_options,
            } => {
                for part in content {
                    let tr = match part {
                        v2t::ToolMessagePart::ToolResult(tr) => tr,
                        v2t::ToolMessagePart::ToolApprovalResponse(_) => {
                            continue;
                        }
                    };
                    let value_str = match &tr.output {
                        v2t::ToolResultOutput::Text { value } => value.clone(),
                        v2t::ToolResultOutput::Json { value } => value.to_string(),
                        v2t::ToolResultOutput::ErrorText { value } => value.clone(),
                        v2t::ToolResultOutput::ErrorJson { value } => value.to_string(),
                        v2t::ToolResultOutput::Content { value } => {
                            serde_json::json!(value).to_string()
                        }
                    };
                    let mut obj = serde_json::json!({"role":"tool","tool_call_id": tr.tool_call_id, "content": value_str}).as_object().unwrap().clone();
                    let meta = get_openai_metadata(provider_scope, provider_options);
                    for (k, v) in meta {
                        obj.insert(k, v);
                    }
                    messages.push(serde_json::Value::Object(obj));
                }
            }
        }
    }
    messages
}
