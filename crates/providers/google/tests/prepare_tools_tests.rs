use crate::ai_sdk_providers_google::prepare_tools::prepare_tools;
use crate::ai_sdk_types::v2 as v2t;
use serde_json::json;

fn provider_tool(id: &str, args: serde_json::Value) -> v2t::Tool {
    v2t::Tool::Provider(v2t::ProviderTool {
        r#type: v2t::ProviderToolType::Provider,
        id: id.to_string(),
        name: id.to_string(),
        args,
    })
}

#[test]
fn provider_google_search_dynamic_retrieval() {
    let tool = provider_tool(
        "google.google_search",
        json!({
            "mode": "MODE_DYNAMIC",
            "dynamicThreshold": 0.7
        }),
    );

    let prepared = prepare_tools(&[tool], &None, "gemini-1.5-flash");

    assert_eq!(
        prepared.tools,
        Some(json!([
            {
                "googleSearchRetrieval": {
                    "dynamicRetrievalConfig": {
                        "mode": "MODE_DYNAMIC",
                        "dynamicThreshold": 0.7
                    }
                }
            }
        ]))
    );
    assert!(prepared.tool_config.is_none());
    assert!(prepared.tool_warnings.is_empty());
}

#[test]
fn provider_enterprise_search_requires_gemini2() {
    let tool = provider_tool("google.enterprise_web_search", json!({}));

    let prepared = prepare_tools(&[tool], &None, "gemini-1.5-pro");

    assert!(prepared.tools.is_none());
    assert_eq!(prepared.tool_warnings.len(), 1);
    match &prepared.tool_warnings[0] {
        v2t::CallWarning::UnsupportedTool { tool_name, details } => {
            assert_eq!(tool_name, "google.enterprise_web_search");
            assert!(details
                .as_deref()
                .unwrap_or_default()
                .contains("Gemini 2.0 or newer"));
        }
        other => panic!("unexpected warning: {other:?}"),
    }
}
