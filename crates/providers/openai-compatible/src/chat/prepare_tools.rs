use crate::ai_sdk_types::v2 as v2t;

pub struct PreparedTools {
    pub tools: Option<Vec<serde_json::Value>>,
    pub tool_choice: Option<serde_json::Value>,
    pub warnings: Vec<v2t::CallWarning>,
}

pub fn prepare_tools(tools: &[v2t::Tool], tool_choice: &Option<v2t::ToolChoice>) -> PreparedTools {
    let mut warnings: Vec<v2t::CallWarning> = vec![];
    let function_tools: Vec<&v2t::FunctionTool> = tools
        .iter()
        .filter_map(|tool| match tool {
            v2t::Tool::Function(f) => Some(f),
            v2t::Tool::Provider(p) => {
                warnings.push(v2t::CallWarning::UnsupportedTool {
                    tool_name: p.name.clone(),
                    details: Some("provider tools are not supported".into()),
                });
                None
            }
        })
        .collect();

    let openai_tools = if function_tools.is_empty() {
        None
    } else {
        Some(
            function_tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect::<Vec<_>>(),
        )
    };

    let tool_choice_val = match tool_choice {
        None => None,
        Some(v2t::ToolChoice::Auto) => Some(serde_json::json!("auto")),
        Some(v2t::ToolChoice::None) => Some(serde_json::json!("none")),
        Some(v2t::ToolChoice::Required) => Some(serde_json::json!("required")),
        Some(v2t::ToolChoice::Tool { name }) => {
            Some(serde_json::json!({"type":"function","function": {"name": name}}))
        }
    };

    PreparedTools {
        tools: openai_tools,
        tool_choice: tool_choice_val,
        warnings,
    }
}
