use crate::types::v2 as v2t;
use serde_json::{json, Value as JsonValue};

/// Convert JSON Schema (draft-07) into the OpenAPI-like schema expected by Google models.
pub fn convert_json_schema_to_openapi_schema(schema: &JsonValue) -> JsonValue {
    if is_empty_object_schema(schema) {
        return JsonValue::Null;
    }
    match schema {
        JsonValue::Bool(b) => json!({"type": "boolean", "properties": {}, "const": b}),
        JsonValue::Object(map) => {
            let mut out = serde_json::Map::new();

            if let Some(desc) = map.get("description").cloned() {
                out.insert("description".into(), desc);
            }
            if let Some(req) = map.get("required").cloned() {
                out.insert("required".into(), req);
            }
            if let Some(fmt) = map.get("format").cloned() {
                out.insert("format".into(), fmt);
            }

            if let Some(cv) = map.get("const").cloned() {
                out.insert("enum".into(), json!([cv]));
            }

            if let Some(t) = map.get("type") {
                match t {
                    JsonValue::Array(arr) => {
                        if arr.iter().any(|v| v == "null") {
                            let first_non_null = arr
                                .iter()
                                .find(|v| **v != JsonValue::String("null".into()))
                                .cloned()
                                .unwrap_or(JsonValue::String("object".into()));
                            out.insert("type".into(), first_non_null);
                            out.insert("nullable".into(), JsonValue::Bool(true));
                        } else {
                            out.insert("type".into(), JsonValue::Array(arr.clone()));
                        }
                    }
                    JsonValue::String(s) => {
                        out.insert("type".into(), JsonValue::String(s.clone()));
                    }
                    _ => {}
                }
            }

            if let Some(ev) = map.get("enum").cloned() {
                out.insert("enum".into(), ev);
            }

            if let Some(props) = map.get("properties").and_then(|v| v.as_object()) {
                let mut props_out = serde_json::Map::new();
                for (k, v) in props {
                    let converted = convert_json_schema_to_openapi_schema(v);
                    if !converted.is_null() {
                        props_out.insert(k.clone(), converted);
                    }
                }
                out.insert("properties".into(), JsonValue::Object(props_out));
            }

            if let Some(items) = map.get("items") {
                let converted = match items {
                    JsonValue::Array(arr) => JsonValue::Array(
                        arr.iter()
                            .map(convert_json_schema_to_openapi_schema)
                            .collect(),
                    ),
                    other => convert_json_schema_to_openapi_schema(other),
                };
                out.insert("items".into(), converted);
            }

            if let Some(all) = map.get("allOf").and_then(|v| v.as_array()) {
                out.insert(
                    "allOf".into(),
                    JsonValue::Array(
                        all.iter()
                            .map(convert_json_schema_to_openapi_schema)
                            .collect(),
                    ),
                );
            }

            if let Some(any) = map.get("anyOf").and_then(|v| v.as_array()) {
                let has_null = any.iter().any(|s| {
                    matches!(s, JsonValue::Object(m) if m.get("type").and_then(|v| v.as_str()) == Some("null"))
                });
                let non_null: Vec<JsonValue> = any
                    .iter()
                    .filter(|s| {
                        !matches!(s, JsonValue::Object(m) if m.get("type").and_then(|v| v.as_str()) == Some("null"))
                    })
                    .map(convert_json_schema_to_openapi_schema)
                    .collect();
                if has_null {
                    if non_null.len() == 1 {
                        if let Some(obj) = non_null.into_iter().next() {
                            if let JsonValue::Object(mut om) = obj {
                                om.insert("nullable".into(), JsonValue::Bool(true));
                                return JsonValue::Object(om);
                            }
                        }
                    } else {
                        out.insert("anyOf".into(), JsonValue::Array(non_null));
                        out.insert("nullable".into(), JsonValue::Bool(true));
                    }
                } else {
                    out.insert("anyOf".into(), JsonValue::Array(non_null));
                }
            }

            if let Some(one) = map.get("oneOf").and_then(|v| v.as_array()) {
                out.insert(
                    "oneOf".into(),
                    JsonValue::Array(
                        one.iter()
                            .map(convert_json_schema_to_openapi_schema)
                            .collect(),
                    ),
                );
            }

            if let Some(minl) = map.get("minLength").cloned() {
                out.insert("minLength".into(), minl);
            }

            JsonValue::Object(out)
        }
        _ => JsonValue::Null,
    }
}

fn is_empty_object_schema(v: &JsonValue) -> bool {
    if let JsonValue::Object(m) = v {
        let t_obj = m.get("type").and_then(|v| v.as_str()) == Some("object");
        let props_empty = m
            .get("properties")
            .map(|p| p.as_object().map(|o| o.is_empty()).unwrap_or(true))
            .unwrap_or(true);
        let addl = m.get("additionalProperties").is_some();
        return t_obj && props_empty && !addl;
    }
    false
}

pub struct PreparedTools {
    pub tools: Option<JsonValue>,
    pub tool_config: Option<JsonValue>,
    pub tool_warnings: Vec<v2t::CallWarning>,
}

struct GoogleToolSupport {
    is_gemini2_or_newer: bool,
    supports_dynamic_retrieval: bool,
    supports_file_search: bool,
}

fn detect_google_tool_support(model_id: &str) -> GoogleToolSupport {
    let model_id = model_id.to_ascii_lowercase();
    let is_latest = matches!(
        model_id.as_str(),
        "gemini-flash-latest" | "gemini-flash-lite-latest" | "gemini-pro-latest"
    );

    GoogleToolSupport {
        is_gemini2_or_newer: model_id.contains("gemini-2")
            || model_id.contains("gemini-3")
            || is_latest,
        supports_dynamic_retrieval: model_id.contains("gemini-1.5-flash")
            && !model_id.contains("-8b"),
        supports_file_search: model_id.contains("gemini-2.5"),
    }
}

fn push_unsupported_google_tool_warning(
    tool_warnings: &mut Vec<v2t::CallWarning>,
    tool_name: &str,
    details: Option<&str>,
) {
    tool_warnings.push(v2t::CallWarning::UnsupportedTool {
        tool_name: tool_name.into(),
        details: details.map(str::to_string),
    });
}

fn build_google_provider_tool(
    tool: &v2t::ProviderTool,
    support: &GoogleToolSupport,
    tool_warnings: &mut Vec<v2t::CallWarning>,
) -> Option<JsonValue> {
    match tool.id.as_str() {
        "google.google_search" => {
            if support.is_gemini2_or_newer {
                Some(json!({ "googleSearch": {} }))
            } else if support.supports_dynamic_retrieval {
                let mut dynamic_config = serde_json::Map::new();
                if let Some(mode) = tool.args.get("mode").and_then(|v| v.as_str()) {
                    dynamic_config.insert("mode".into(), json!(mode));
                }
                if let Some(threshold) = tool.args.get("dynamicThreshold").and_then(|v| v.as_f64())
                {
                    dynamic_config.insert("dynamicThreshold".into(), json!(threshold));
                }
                Some(json!({
                    "googleSearchRetrieval": {
                        "dynamicRetrievalConfig": JsonValue::Object(dynamic_config),
                    }
                }))
            } else {
                Some(json!({ "googleSearchRetrieval": {} }))
            }
        }
        "google.enterprise_web_search" => {
            if support.is_gemini2_or_newer {
                Some(json!({ "enterpriseWebSearch": {} }))
            } else {
                push_unsupported_google_tool_warning(
                    tool_warnings,
                    &tool.id,
                    Some("Enterprise Web Search requires Gemini 2.0 or newer."),
                );
                None
            }
        }
        "google.url_context" => {
            if support.is_gemini2_or_newer {
                Some(json!({ "urlContext": {} }))
            } else {
                push_unsupported_google_tool_warning(
                    tool_warnings,
                    &tool.id,
                    Some(
                        "The URL context tool is not supported with other Gemini models than Gemini 2.",
                    ),
                );
                None
            }
        }
        "google.code_execution" => {
            if support.is_gemini2_or_newer {
                Some(json!({ "codeExecution": {} }))
            } else {
                push_unsupported_google_tool_warning(
                    tool_warnings,
                    &tool.id,
                    Some(
                        "The code execution tools is not supported with other Gemini models than Gemini 2.",
                    ),
                );
                None
            }
        }
        "google.file_search" => {
            if support.supports_file_search {
                let args = tool.args.as_object().cloned().unwrap_or_default();
                Some(json!({ "fileSearch": args }))
            } else {
                push_unsupported_google_tool_warning(
                    tool_warnings,
                    &tool.id,
                    Some("The file search tool is only supported with Gemini 2.5 models."),
                );
                None
            }
        }
        "google.vertex_rag_store" => {
            if support.is_gemini2_or_newer {
                let mut rag_resources = serde_json::Map::new();
                if let Some(rag_corpus) = tool.args.get("ragCorpus").and_then(|v| v.as_str()) {
                    rag_resources.insert("rag_corpus".into(), json!(rag_corpus));
                }
                let mut vertex_rag_store = serde_json::Map::new();
                vertex_rag_store.insert("rag_resources".into(), JsonValue::Object(rag_resources));
                if let Some(top_k) = tool.args.get("topK").and_then(|v| v.as_u64()) {
                    vertex_rag_store.insert("similarity_top_k".into(), json!(top_k));
                }
                Some(json!({
                    "retrieval": { "vertex_rag_store": vertex_rag_store }
                }))
            } else {
                push_unsupported_google_tool_warning(
                    tool_warnings,
                    &tool.id,
                    Some(
                        "The RAG store tool is not supported with other Gemini models than Gemini 2.",
                    ),
                );
                None
            }
        }
        "google.google_maps" => {
            if support.is_gemini2_or_newer {
                Some(json!({ "googleMaps": {} }))
            } else {
                push_unsupported_google_tool_warning(
                    tool_warnings,
                    &tool.id,
                    Some(
                        "The Google Maps grounding tool is not supported with Gemini models other than Gemini 2 or newer.",
                    ),
                );
                None
            }
        }
        _ => {
            push_unsupported_google_tool_warning(tool_warnings, &tool.id, None);
            None
        }
    }
}

fn build_google_provider_tools(
    tools: &[v2t::Tool],
    model_id: &str,
    tool_warnings: &mut Vec<v2t::CallWarning>,
) -> Vec<JsonValue> {
    let support = detect_google_tool_support(model_id);

    tools
        .iter()
        .filter_map(|tool| match tool {
            v2t::Tool::Provider(tool) => build_google_provider_tool(tool, &support, tool_warnings),
            _ => None,
        })
        .collect()
}

fn collect_function_tools(tools: &[v2t::Tool]) -> Vec<&v2t::FunctionTool> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            v2t::Tool::Function(tool) => Some(tool),
            _ => None,
        })
        .collect()
}

fn build_google_function_declarations(function_tools: &[&v2t::FunctionTool]) -> JsonValue {
    let mut declarations = Vec::new();
    for tool in function_tools {
        let params = convert_json_schema_to_openapi_schema(&tool.input_schema);
        declarations.push(json!({
            "name": tool.name,
            "description": tool.description.clone().unwrap_or_default(),
            "parameters": if params.is_null() { JsonValue::Null } else { params },
        }));
    }
    json!([{ "functionDeclarations": declarations }])
}

fn build_google_function_tool_config(tool_choice: &Option<v2t::ToolChoice>) -> Option<JsonValue> {
    match tool_choice {
        None => None,
        Some(v2t::ToolChoice::Auto) => Some(json!({"functionCallingConfig": {"mode": "AUTO"}})),
        Some(v2t::ToolChoice::None) => Some(json!({"functionCallingConfig": {"mode": "NONE"}})),
        Some(v2t::ToolChoice::Required) => Some(json!({"functionCallingConfig": {"mode": "ANY"}})),
        Some(v2t::ToolChoice::Tool { name }) => {
            Some(json!({"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}))
        }
    }
}

pub fn prepare_tools(
    tools: &[v2t::Tool],
    tool_choice: &Option<v2t::ToolChoice>,
    model_id: &str,
) -> PreparedTools {
    let mut tool_warnings = vec![];

    if tools.is_empty() {
        return PreparedTools {
            tools: None,
            tool_config: None,
            tool_warnings,
        };
    }

    let has_function_tools = tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Function(_)));
    let has_provider_tools = tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(_)));

    if has_function_tools && has_provider_tools {
        tool_warnings.push(v2t::CallWarning::Other {
            message: "Unsupported combination of function and provider-defined tools.".into(),
        });
    }

    if has_provider_tools {
        let google_tools = build_google_provider_tools(tools, model_id, &mut tool_warnings);

        return PreparedTools {
            tools: if google_tools.is_empty() {
                None
            } else {
                Some(JsonValue::Array(google_tools))
            },
            tool_config: None,
            tool_warnings,
        };
    }

    let function_tools = collect_function_tools(tools);

    if function_tools.is_empty() {
        return PreparedTools {
            tools: None,
            tool_config: None,
            tool_warnings,
        };
    }

    PreparedTools {
        tools: Some(build_google_function_declarations(&function_tools)),
        tool_config: build_google_function_tool_config(tool_choice),
        tool_warnings,
    }
}
