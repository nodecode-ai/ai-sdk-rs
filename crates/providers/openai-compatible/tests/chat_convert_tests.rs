use crate::ai_sdk_providers_openai_compatible::chat::convert::convert_to_openai_compatible_chat_messages;
use crate::ai_sdk_types::v2 as v2t;
use serde_json::json;

#[test]
fn user_text_collapses_to_string_content() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::Text {
            text: "Hello".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];

    let result = convert_to_openai_compatible_chat_messages("test-provider", &prompt);

    assert_eq!(result, vec![json!({"role":"user","content":"Hello"})]);
}

#[test]
fn user_images_from_bytes_become_data_urls() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![
            v2t::UserPart::Text {
                text: "Hello".into(),
                provider_options: None,
            },
            v2t::UserPart::File {
                filename: None,
                data: v2t::DataContent::Bytes {
                    bytes: vec![0, 1, 2, 3],
                },
                media_type: "image/png".into(),
                provider_options: None,
            },
        ],
        provider_options: None,
    }];

    let result = convert_to_openai_compatible_chat_messages("test-provider", &prompt);

    assert_eq!(
        result,
        vec![json!({
            "role":"user",
            "content": [
                {"type":"text","text":"Hello"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAECAw==" }}
            ]
        })]
    );
}

#[test]
fn user_images_from_url_passthrough() {
    let prompt = vec![v2t::PromptMessage::User {
        content: vec![v2t::UserPart::File {
            filename: None,
            data: v2t::DataContent::Url {
                url: "https://example.com/image.jpg".into(),
            },
            media_type: "image/*".into(),
            provider_options: None,
        }],
        provider_options: None,
    }];

    let result = convert_to_openai_compatible_chat_messages("test-provider", &prompt);

    assert_eq!(
        result,
        vec![json!({
            "role":"user",
            "content": [
                {"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}
            ]
        })]
    );
}

#[test]
fn tool_calls_and_results_are_stringified() {
    let prompt = vec![
        v2t::PromptMessage::Assistant {
            content: vec![v2t::AssistantPart::ToolCall(v2t::ToolCallPart {
                tool_call_id: "quux".into(),
                tool_name: "thwomp".into(),
                input: json!({"foo":"bar123"}).to_string(),
                provider_executed: false,
                provider_metadata: None,
                dynamic: false,
                provider_options: None,
            })],
            provider_options: None,
        },
        v2t::PromptMessage::Tool {
            content: vec![v2t::ToolMessagePart::ToolResult(v2t::ToolResultPart {
                r#type: v2t::ToolResultPartType::ToolResult,
                tool_call_id: "quux".into(),
                tool_name: "thwomp".into(),
                output: v2t::ToolResultOutput::Json {
                    value: json!({"oof":"321rab"}),
                },
                provider_options: None,
            })],
            provider_options: None,
        },
    ];

    let result = convert_to_openai_compatible_chat_messages("test-provider", &prompt);

    assert_eq!(
        result,
        vec![
            json!({
                "role":"assistant",
                "tool_calls":[
                    {
                        "type":"function",
                        "id":"quux",
                        "function":{"name":"thwomp","arguments": json!({"foo":"bar123"}).to_string()}
                    }
                ]
            }),
            json!({
                "role":"tool",
                "tool_call_id":"quux",
                "content": json!({"oof":"321rab"}).to_string()
            })
        ]
    );
}
