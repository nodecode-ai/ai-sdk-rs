use std::collections::HashMap;

use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_types::v2 as v2t;
use serde_json::{json, Map, Value};

fn invalid_tool_args(tool: &v2t::ProviderTool, message: impl Into<String>) -> SdkError {
    SdkError::InvalidArgument {
        message: format!(
            "provider tool {} ({}): {}",
            tool.name,
            tool.id,
            message.into()
        ),
    }
}

fn require_args_object(tool: &v2t::ProviderTool) -> Result<&Map<String, Value>, SdkError> {
    tool.args
        .as_object()
        .ok_or_else(|| invalid_tool_args(tool, "args must be an object"))
}

fn require_field<'a>(
    tool: &v2t::ProviderTool,
    args: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a Value, SdkError> {
    args.get(key)
        .ok_or_else(|| invalid_tool_args(tool, format!("args.{key} is required")))
}

fn expect_string(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
    if value.as_str().is_some() {
        Ok(())
    } else {
        Err(invalid_tool_args(tool, format!("{path} must be a string")))
    }
}

fn expect_bool(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
    if value.as_bool().is_some() {
        Ok(())
    } else {
        Err(invalid_tool_args(tool, format!("{path} must be a boolean")))
    }
}

fn expect_number(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
    if value.as_f64().is_some() {
        Ok(())
    } else {
        Err(invalid_tool_args(tool, format!("{path} must be a number")))
    }
}

fn expect_int_range(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
    min: i64,
    max: i64,
) -> Result<(), SdkError> {
    let raw = value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| {
            value.as_f64().filter(|v| v.fract() == 0.0).and_then(|v| {
                if v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                    Some(v as i64)
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an integer")))?;
    if raw < min || raw > max {
        return Err(invalid_tool_args(
            tool,
            format!("{path} must be between {min} and {max}"),
        ));
    }
    Ok(())
}

fn expect_enum(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
    allowed: &[&str],
) -> Result<(), SdkError> {
    match value.as_str() {
        Some(val) if allowed.contains(&val) => Ok(()),
        Some(_) | None => Err(invalid_tool_args(
            tool,
            format!("{path} must be one of {}", allowed.join(", ")),
        )),
    }
}

fn expect_string_array(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an array")))?;
    if arr.iter().all(|item| item.as_str().is_some()) {
        Ok(())
    } else {
        Err(invalid_tool_args(
            tool,
            format!("{path} must be an array of strings"),
        ))
    }
}

fn expect_object<'a>(
    tool: &v2t::ProviderTool,
    value: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, SdkError> {
    value
        .as_object()
        .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an object")))
}

fn ensure_known_keys(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
    allowed: &[&str],
) -> Result<(), SdkError> {
    let mut unknown = args
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    Err(invalid_tool_args(
        tool,
        format!("args contains unsupported keys: {}", unknown.join(", ")),
    ))
}

fn validate_user_location(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    let obj = expect_object(tool, value, path)?;
    let loc_type = obj
        .get("type")
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.type must be \"approximate\"")))?;
    expect_enum(tool, loc_type, &format!("{path}.type"), &["approximate"])?;
    for key in ["country", "city", "region", "timezone"] {
        if let Some(val) = obj.get(key) {
            expect_string(tool, val, &format!("{path}.{key}"))?;
        }
    }
    Ok(())
}

fn require_filter_type<'a>(
    tool: &v2t::ProviderTool,
    obj: &'a Map<String, Value>,
    path: &str,
) -> Result<&'a str, SdkError> {
    obj.get("type")
        .and_then(|value| value.as_str())
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.type is required")))
}

fn validate_nested_file_search_filters(
    tool: &v2t::ProviderTool,
    obj: &Map<String, Value>,
    path: &str,
) -> Result<(), SdkError> {
    let filters = obj
        .get("filters")
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.filters is required")))?;
    let arr = filters
        .as_array()
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.filters must be an array")))?;
    for (idx, entry) in arr.iter().enumerate() {
        validate_file_search_filter(tool, entry, &format!("{path}.filters[{idx}]"))?;
    }
    Ok(())
}

fn validate_file_search_filter_value(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    if value.as_str().is_some() || value.as_bool().is_some() || value.as_f64().is_some() {
        return Ok(());
    }
    if let Some(arr) = value.as_array() {
        if arr.iter().all(|item| item.as_str().is_some()) {
            return Ok(());
        }
    }
    Err(invalid_tool_args(
        tool,
        format!("{path} must be a string, number, boolean, or array of strings"),
    ))
}

fn validate_comparison_file_search_filter(
    tool: &v2t::ProviderTool,
    obj: &Map<String, Value>,
    path: &str,
) -> Result<(), SdkError> {
    let key = obj
        .get("key")
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.key is required")))?;
    expect_string(tool, key, &format!("{path}.key"))?;
    let value = obj
        .get("value")
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.value is required")))?;
    validate_file_search_filter_value(tool, value, &format!("{path}.value"))
}

fn validate_file_search_filter(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    let obj = expect_object(tool, value, path)?;
    match require_filter_type(tool, obj, path)? {
        "and" | "or" => validate_nested_file_search_filters(tool, obj, path),
        "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "nin" => {
            validate_comparison_file_search_filter(tool, obj, path)
        }
        _ => Err(invalid_tool_args(
            tool,
            format!("{path}.type has an invalid value"),
        )),
    }
}

fn validate_file_search_tool_args(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
) -> Result<(), SdkError> {
    let ids = require_field(tool, args, "vectorStoreIds")?;
    expect_string_array(tool, ids, "args.vectorStoreIds")?;
    if let Some(max) = args.get("maxNumResults") {
        expect_number(tool, max, "args.maxNumResults")?;
    }
    if let Some(rank) = args.get("ranking") {
        let rank_obj = expect_object(tool, rank, "args.ranking")?;
        if let Some(ranker) = rank_obj.get("ranker") {
            expect_string(tool, ranker, "args.ranking.ranker")?;
        }
        if let Some(score) = rank_obj.get("scoreThreshold") {
            expect_number(tool, score, "args.ranking.scoreThreshold")?;
        }
    }
    if let Some(filters) = args.get("filters") {
        validate_file_search_filter(tool, filters, "args.filters")?;
    }
    Ok(())
}

fn validate_web_search_preview_tool_args(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
) -> Result<(), SdkError> {
    if let Some(size) = args.get("searchContextSize") {
        expect_enum(
            tool,
            size,
            "args.searchContextSize",
            &["low", "medium", "high"],
        )?;
    }
    if let Some(loc) = args.get("userLocation") {
        validate_user_location(tool, loc, "args.userLocation")?;
    }
    Ok(())
}

fn validate_web_search_tool_args(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
) -> Result<(), SdkError> {
    if let Some(access) = args.get("externalWebAccess") {
        expect_bool(tool, access, "args.externalWebAccess")?;
    }
    if let Some(filters) = args.get("filters") {
        let obj = expect_object(tool, filters, "args.filters")?;
        if let Some(domains) = obj.get("allowedDomains") {
            expect_string_array(tool, domains, "args.filters.allowedDomains")?;
        }
    }
    if let Some(size) = args.get("searchContextSize") {
        expect_enum(
            tool,
            size,
            "args.searchContextSize",
            &["low", "medium", "high"],
        )?;
    }
    if let Some(loc) = args.get("userLocation") {
        validate_user_location(tool, loc, "args.userLocation")?;
    }
    Ok(())
}

fn validate_code_interpreter_tool_args(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
) -> Result<(), SdkError> {
    if let Some(container) = args.get("container") {
        if let Some(obj) = container.as_object() {
            if let Some(file_ids) = obj.get("fileIds") {
                expect_string_array(tool, file_ids, "args.container.fileIds")?;
            }
        } else if container.as_str().is_none() {
            return Err(invalid_tool_args(
                tool,
                "args.container must be a string or object",
            ));
        }
    }
    Ok(())
}

fn validate_image_generation_tool_args(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
) -> Result<(), SdkError> {
    ensure_known_keys(
        tool,
        args,
        &[
            "background",
            "inputFidelity",
            "inputImageMask",
            "model",
            "moderation",
            "outputCompression",
            "outputFormat",
            "partialImages",
            "quality",
            "size",
        ],
    )?;
    if let Some(background) = args.get("background") {
        expect_enum(
            tool,
            background,
            "args.background",
            &["auto", "opaque", "transparent"],
        )?;
    }
    if let Some(fidelity) = args.get("inputFidelity") {
        expect_enum(tool, fidelity, "args.inputFidelity", &["low", "high"])?;
    }
    if let Some(mask) = args.get("inputImageMask") {
        let mask_obj = expect_object(tool, mask, "args.inputImageMask")?;
        if let Some(file_id) = mask_obj.get("fileId") {
            expect_string(tool, file_id, "args.inputImageMask.fileId")?;
        }
        if let Some(image_url) = mask_obj.get("imageUrl") {
            expect_string(tool, image_url, "args.inputImageMask.imageUrl")?;
        }
    }
    if let Some(model) = args.get("model") {
        expect_string(tool, model, "args.model")?;
    }
    if let Some(moderation) = args.get("moderation") {
        expect_enum(tool, moderation, "args.moderation", &["auto"])?;
    }
    if let Some(output_compression) = args.get("outputCompression") {
        expect_int_range(tool, output_compression, "args.outputCompression", 0, 100)?;
    }
    if let Some(output_format) = args.get("outputFormat") {
        expect_enum(
            tool,
            output_format,
            "args.outputFormat",
            &["png", "jpeg", "webp"],
        )?;
    }
    if let Some(partial_images) = args.get("partialImages") {
        expect_int_range(tool, partial_images, "args.partialImages", 0, 3)?;
    }
    if let Some(quality) = args.get("quality") {
        expect_enum(
            tool,
            quality,
            "args.quality",
            &["auto", "low", "medium", "high"],
        )?;
    }
    if let Some(size) = args.get("size") {
        expect_enum(
            tool,
            size,
            "args.size",
            &["1024x1024", "1024x1536", "1536x1024", "auto"],
        )?;
    }
    Ok(())
}

fn validate_mcp_allowed_tools(tool: &v2t::ProviderTool, allowed: &Value) -> Result<(), SdkError> {
    if let Some(arr) = allowed.as_array() {
        for (idx, entry) in arr.iter().enumerate() {
            expect_string(tool, entry, &format!("args.allowedTools[{idx}]"))?;
        }
        return Ok(());
    }

    let obj = allowed
        .as_object()
        .ok_or_else(|| invalid_tool_args(tool, "args.allowedTools must be an array or object"))?;
    if let Some(read_only) = obj.get("readOnly") {
        expect_bool(tool, read_only, "args.allowedTools.readOnly")?;
    }
    if let Some(tool_names) = obj.get("toolNames") {
        expect_string_array(tool, tool_names, "args.allowedTools.toolNames")?;
    }
    Ok(())
}

fn validate_mcp_require_approval(
    tool: &v2t::ProviderTool,
    require_approval: &Value,
) -> Result<(), SdkError> {
    if let Some(value) = require_approval.as_str() {
        if matches!(value, "always" | "never") {
            return Ok(());
        }
        return Err(invalid_tool_args(
            tool,
            "args.requireApproval must be \"always\" or \"never\"",
        ));
    }

    let obj = require_approval.as_object().ok_or_else(|| {
        invalid_tool_args(tool, "args.requireApproval must be a string or object")
    })?;
    if let Some(never) = obj.get("never") {
        let never_obj = expect_object(tool, never, "args.requireApproval.never")?;
        if let Some(tool_names) = never_obj.get("toolNames") {
            expect_string_array(tool, tool_names, "args.requireApproval.never.toolNames")?;
        }
    }
    Ok(())
}

fn validate_mcp_tool_args(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
) -> Result<(), SdkError> {
    let server_label = require_field(tool, args, "serverLabel")?;
    expect_string(tool, server_label, "args.serverLabel")?;

    let server_url = args.get("serverUrl");
    let connector_id = args.get("connectorId");
    if let Some(url) = server_url {
        expect_string(tool, url, "args.serverUrl")?;
    }
    if let Some(connector) = connector_id {
        expect_string(tool, connector, "args.connectorId")?;
    }
    if server_url.is_none() && connector_id.is_none() {
        return Err(invalid_tool_args(
            tool,
            "args.serverUrl or args.connectorId is required",
        ));
    }

    if let Some(allowed) = args.get("allowedTools") {
        validate_mcp_allowed_tools(tool, allowed)?;
    }
    if let Some(authorization) = args.get("authorization") {
        expect_string(tool, authorization, "args.authorization")?;
    }
    if let Some(headers) = args.get("headers") {
        let obj = expect_object(tool, headers, "args.headers")?;
        for (key, val) in obj {
            expect_string(tool, val, &format!("args.headers.{key}"))?;
        }
    }
    if let Some(require_approval) = args.get("requireApproval") {
        validate_mcp_require_approval(tool, require_approval)?;
    }
    if let Some(server_description) = args.get("serverDescription") {
        expect_string(tool, server_description, "args.serverDescription")?;
    }
    Ok(())
}

fn validate_openai_provider_tool_args(
    tool_type: &str,
    tool: &v2t::ProviderTool,
) -> Result<(), SdkError> {
    match tool_type {
        "file_search" => validate_file_search_tool_args(tool, require_args_object(tool)?),
        "web_search_preview" => {
            validate_web_search_preview_tool_args(tool, require_args_object(tool)?)
        }
        "web_search" => validate_web_search_tool_args(tool, require_args_object(tool)?),
        "code_interpreter" => validate_code_interpreter_tool_args(tool, require_args_object(tool)?),
        "image_generation" => validate_image_generation_tool_args(tool, require_args_object(tool)?),
        "mcp" => validate_mcp_tool_args(tool, require_args_object(tool)?),
        _ => Ok(()),
    }
}

#[derive(Clone, Default)]
pub(super) struct ToolNameMapping {
    custom_to_provider: HashMap<String, String>,
    provider_to_custom: HashMap<String, String>,
    pub(super) web_search_tool_name: Option<String>,
}

impl ToolNameMapping {
    pub(super) fn to_provider_tool_name<'a>(&'a self, custom_tool_name: &'a str) -> &'a str {
        self.custom_to_provider
            .get(custom_tool_name)
            .map(|s| s.as_str())
            .unwrap_or(custom_tool_name)
    }

    pub(super) fn to_custom_tool_name<'a>(&'a self, provider_tool_name: &'a str) -> &'a str {
        self.provider_to_custom
            .get(provider_tool_name)
            .map(|s| s.as_str())
            .unwrap_or(provider_tool_name)
    }
}

fn openai_provider_tool_name(id: &str) -> Option<&'static str> {
    match id {
        "openai.file_search" => Some("file_search"),
        "openai.local_shell" => Some("local_shell"),
        "openai.shell" => Some("shell"),
        "openai.apply_patch" => Some("apply_patch"),
        "openai.web_search_preview" => Some("web_search_preview"),
        "openai.web_search" => Some("web_search"),
        "openai.code_interpreter" => Some("code_interpreter"),
        "openai.image_generation" => Some("image_generation"),
        "openai.mcp" => Some("mcp"),
        _ => None,
    }
}

pub(super) fn build_tool_name_mapping(tools: &[v2t::Tool]) -> ToolNameMapping {
    let mut mapping = ToolNameMapping::default();
    for tool in tools {
        if let v2t::Tool::Provider(provider_tool) = tool {
            if let Some(provider_name) = openai_provider_tool_name(&provider_tool.id) {
                mapping
                    .custom_to_provider
                    .insert(provider_tool.name.clone(), provider_name.to_string());
                mapping
                    .provider_to_custom
                    .insert(provider_name.to_string(), provider_tool.name.clone());
                if matches!(
                    provider_tool.id.as_str(),
                    "openai.web_search" | "openai.web_search_preview"
                ) && mapping.web_search_tool_name.is_none()
                {
                    mapping.web_search_tool_name = Some(provider_tool.name.clone());
                }
            }
        }
    }
    mapping
}

fn map_web_search_output(action: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = action.as_object()?;
    let action_type = obj.get("type")?.as_str()?;
    match action_type {
        "search" => {
            let mut out = serde_json::Map::new();
            let mut inner = serde_json::Map::new();
            inner.insert("type".into(), serde_json::Value::String("search".into()));
            if let Some(query) = obj.get("query") {
                if !query.is_null() {
                    inner.insert("query".into(), query.clone());
                }
            }
            out.insert("action".into(), serde_json::Value::Object(inner));
            if let Some(sources) = obj.get("sources") {
                if !sources.is_null() {
                    out.insert("sources".into(), sources.clone());
                }
            }
            Some(serde_json::Value::Object(out))
        }
        "open_page" => Some(json!({
            "action": {
                "type": "openPage",
                "url": obj.get("url").cloned().unwrap_or(serde_json::Value::Null),
            }
        })),
        "find_in_page" => Some(json!({
            "action": {
                "type": "findInPage",
                "url": obj.get("url").cloned().unwrap_or(serde_json::Value::Null),
                "pattern": obj.get("pattern").cloned().unwrap_or(serde_json::Value::Null),
            }
        })),
        _ => None,
    }
}

fn build_file_search_provider_tool(args: &Map<String, Value>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("file_search"));
    if let Some(ids) = args.get("vectorStoreIds") {
        obj.insert("vector_store_ids".into(), ids.clone());
    }
    if let Some(max) = args.get("maxNumResults") {
        obj.insert("max_num_results".into(), max.clone());
    }
    if let Some(rank) = args.get("ranking").and_then(|v| v.as_object()) {
        let mut opts = serde_json::Map::new();
        if let Some(ranker) = rank.get("ranker") {
            opts.insert("ranker".into(), ranker.clone());
        }
        if let Some(score) = rank.get("scoreThreshold") {
            opts.insert("score_threshold".into(), score.clone());
        }
        if !opts.is_empty() {
            obj.insert("ranking_options".into(), serde_json::Value::Object(opts));
        }
    }
    if let Some(filters) = args.get("filters") {
        obj.insert("filters".into(), filters.clone());
    }
    Value::Object(obj)
}

fn build_web_search_preview_provider_tool(args: &Map<String, Value>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("web_search_preview"));
    if let Some(size) = args.get("searchContextSize") {
        obj.insert("search_context_size".into(), size.clone());
    }
    if let Some(loc) = args.get("userLocation") {
        obj.insert("user_location".into(), loc.clone());
    }
    Value::Object(obj)
}

fn build_web_search_provider_tool(args: &Map<String, Value>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("web_search"));
    if let Some(filters) = args.get("filters").and_then(|v| v.as_object()) {
        if let Some(allowed_domains) = filters.get("allowedDomains") {
            obj.insert(
                "filters".into(),
                json!({"allowed_domains": allowed_domains}),
            );
        }
    }
    if let Some(access) = args.get("externalWebAccess") {
        obj.insert("external_web_access".into(), access.clone());
    }
    if let Some(size) = args.get("searchContextSize") {
        obj.insert("search_context_size".into(), size.clone());
    }
    if let Some(loc) = args.get("userLocation") {
        obj.insert("user_location".into(), loc.clone());
    }
    Value::Object(obj)
}

fn build_code_interpreter_provider_tool(args: &Map<String, Value>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("code_interpreter"));
    let container = match args.get("container") {
        None | Some(serde_json::Value::Null) => json!({"type":"auto"}),
        Some(serde_json::Value::String(val)) => json!(val),
        Some(serde_json::Value::Object(map)) => {
            let mut container = serde_json::Map::new();
            container.insert("type".into(), json!("auto"));
            if let Some(file_ids) = map.get("fileIds") {
                container.insert("file_ids".into(), file_ids.clone());
            }
            serde_json::Value::Object(container)
        }
        Some(other) => other.clone(),
    };
    obj.insert("container".into(), container);
    Value::Object(obj)
}

fn build_image_generation_provider_tool(args: &Map<String, Value>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("image_generation"));
    for (src, dst) in [
        ("background", "background"),
        ("inputFidelity", "input_fidelity"),
        ("model", "model"),
        ("moderation", "moderation"),
        ("partialImages", "partial_images"),
        ("quality", "quality"),
        ("outputCompression", "output_compression"),
        ("outputFormat", "output_format"),
        ("size", "size"),
    ] {
        if let Some(val) = args.get(src) {
            obj.insert(dst.into(), val.clone());
        }
    }
    if let Some(mask) = args.get("inputImageMask").and_then(|v| v.as_object()) {
        let mut mask_obj = serde_json::Map::new();
        if let Some(file_id) = mask.get("fileId") {
            mask_obj.insert("file_id".into(), file_id.clone());
        }
        if let Some(image_url) = mask.get("imageUrl") {
            mask_obj.insert("image_url".into(), image_url.clone());
        }
        if !mask_obj.is_empty() {
            obj.insert(
                "input_image_mask".into(),
                serde_json::Value::Object(mask_obj),
            );
        }
    }
    Value::Object(obj)
}

fn build_mcp_provider_tool(args: &Map<String, Value>) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("type".into(), json!("mcp"));
    if let Some(val) = args.get("serverLabel") {
        obj.insert("server_label".into(), val.clone());
    }
    if let Some(val) = args.get("authorization") {
        obj.insert("authorization".into(), val.clone());
    }
    if let Some(val) = args.get("connectorId") {
        obj.insert("connector_id".into(), val.clone());
    }
    if let Some(val) = args.get("headers") {
        obj.insert("headers".into(), val.clone());
    }
    if let Some(val) = args.get("serverDescription") {
        obj.insert("server_description".into(), val.clone());
    }
    if let Some(val) = args.get("serverUrl") {
        obj.insert("server_url".into(), val.clone());
    }
    if let Some(allowed) = args.get("allowedTools") {
        if let Some(list) = allowed.as_array() {
            obj.insert(
                "allowed_tools".into(),
                serde_json::Value::Array(list.clone()),
            );
        } else if let Some(filter) = allowed.as_object() {
            let mut allowed_obj = serde_json::Map::new();
            if let Some(read_only) = filter.get("readOnly") {
                allowed_obj.insert("read_only".into(), read_only.clone());
            }
            if let Some(tool_names) = filter.get("toolNames") {
                allowed_obj.insert("tool_names".into(), tool_names.clone());
            }
            if !allowed_obj.is_empty() {
                obj.insert(
                    "allowed_tools".into(),
                    serde_json::Value::Object(allowed_obj),
                );
            }
        }
    }
    let require_approval = match args.get("requireApproval") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(val)) => Some(serde_json::Value::String(val.clone())),
        Some(serde_json::Value::Object(map)) => map.get("never").map(|never| {
            if let Some(filter) = never.as_object() {
                let mut filter_obj = serde_json::Map::new();
                if let Some(tool_names) = filter.get("toolNames") {
                    filter_obj.insert("tool_names".into(), tool_names.clone());
                }
                json!({"never": filter_obj})
            } else {
                json!({"never": {}})
            }
        }),
        Some(other) => Some(other.clone()),
    };
    obj.insert(
        "require_approval".into(),
        require_approval.unwrap_or_else(|| json!("never")),
    );
    Value::Object(obj)
}

pub(super) fn build_openai_provider_tool(
    tool: &v2t::ProviderTool,
) -> Result<Option<serde_json::Value>, SdkError> {
    let empty = serde_json::Map::new();
    let tool_type = match openai_provider_tool_name(&tool.id) {
        Some(tool_type) => tool_type,
        None => return Ok(None),
    };
    validate_openai_provider_tool_args(tool_type, tool)?;
    let args = tool.args.as_object().unwrap_or(&empty);
    let val = match tool_type {
        "file_search" => Some(build_file_search_provider_tool(args)),
        "local_shell" => Some(json!({"type":"local_shell"})),
        "shell" => Some(json!({"type":"shell"})),
        "apply_patch" => Some(json!({"type":"apply_patch"})),
        "web_search_preview" => Some(build_web_search_preview_provider_tool(args)),
        "web_search" => Some(build_web_search_provider_tool(args)),
        "code_interpreter" => Some(build_code_interpreter_provider_tool(args)),
        "image_generation" => Some(build_image_generation_provider_tool(args)),
        "mcp" => Some(build_mcp_provider_tool(args)),
        _ => None,
    };
    Ok(val)
}

fn build_output_item_base(
    tool_type: &str,
    tool_call_id: String,
    item_id: Option<&String>,
    provider_executed: bool,
) -> serde_json::Map<String, Value> {
    let mut obj = serde_json::Map::new();
    obj.insert("tool_type".into(), json!(tool_type));
    obj.insert("tool_call_id".into(), json!(tool_call_id));
    if let Some(id) = item_id {
        obj.insert("item_id".into(), json!(id));
    }
    obj.insert("provider_executed".into(), json!(provider_executed));
    obj
}

fn output_item_call_id(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<String> {
    item.get("call_id")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
        .or_else(|| item_id.clone())
}

fn map_file_search_results(item: &serde_json::Map<String, Value>) -> Value {
    let results_val = item.get("results").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|entry| entry.as_object())
            .map(|entry| {
                let mut mapped = serde_json::Map::new();
                if let Some(attributes) = entry.get("attributes") {
                    mapped.insert("attributes".into(), attributes.clone());
                }
                if let Some(file_id) = entry.get("file_id") {
                    mapped.insert("fileId".into(), file_id.clone());
                }
                if let Some(filename) = entry.get("filename") {
                    mapped.insert("filename".into(), filename.clone());
                }
                if let Some(score) = entry.get("score") {
                    mapped.insert("score".into(), score.clone());
                }
                if let Some(text) = entry.get("text") {
                    mapped.insert("text".into(), text.clone());
                }
                serde_json::Value::Object(mapped)
            })
            .collect::<Vec<_>>()
    });
    json!({
        "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
        "results": results_val.map(serde_json::Value::Array).unwrap_or(serde_json::Value::Null),
    })
}

fn map_local_shell_input(item: &serde_json::Map<String, Value>) -> Value {
    let action = item.get("action").and_then(|value| value.as_object());
    let mut action_obj = serde_json::Map::new();
    if let Some(action) = action {
        if let Some(command) = action.get("command") {
            action_obj.insert("command".into(), command.clone());
        }
        if let Some(timeout) = action.get("timeout_ms") {
            action_obj.insert("timeoutMs".into(), timeout.clone());
        }
        if let Some(user) = action.get("user") {
            action_obj.insert("user".into(), user.clone());
        }
        if let Some(dir) = action.get("working_directory") {
            action_obj.insert("workingDirectory".into(), dir.clone());
        }
        if let Some(env) = action.get("env") {
            action_obj.insert("env".into(), env.clone());
        }
    }
    if action_obj.is_empty() {
        json!({})
    } else {
        json!({ "action": action_obj })
    }
}

fn map_shell_input(item: &serde_json::Map<String, Value>) -> Value {
    let action = item.get("action").and_then(|value| value.as_object());
    let mut action_obj = serde_json::Map::new();
    if let Some(action) = action {
        if let Some(commands) = action.get("commands") {
            action_obj.insert("commands".into(), commands.clone());
        }
        if let Some(timeout) = action.get("timeout_ms") {
            action_obj.insert("timeoutMs".into(), timeout.clone());
        }
        if let Some(max_len) = action.get("max_output_length") {
            action_obj.insert("maxOutputLength".into(), max_len.clone());
        }
    }
    if action_obj.is_empty() {
        json!({})
    } else {
        json!({ "action": action_obj })
    }
}

fn provider_tool_data_for_web_search(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let mut obj = build_output_item_base("web_search", tool_call_id, item_id.as_ref(), true);
    obj.insert("input".into(), json!({}));
    if let Some(action) = item.get("action").and_then(map_web_search_output) {
        obj.insert("result".into(), action);
    }
    Some(Value::Object(obj))
}

fn provider_tool_data_for_file_search(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let mut obj = build_output_item_base("file_search", tool_call_id, item_id.as_ref(), true);
    obj.insert("input".into(), json!({}));
    obj.insert("result".into(), map_file_search_results(item));
    Some(Value::Object(obj))
}

fn provider_tool_data_for_code_interpreter(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let mut obj = build_output_item_base("code_interpreter", tool_call_id, item_id.as_ref(), true);
    obj.insert(
        "input".into(),
        json!({
            "code": item.get("code").cloned().unwrap_or(serde_json::Value::Null),
            "containerId": item.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
        }),
    );
    obj.insert(
        "result".into(),
        json!({
            "outputs": item.get("outputs").cloned().unwrap_or(serde_json::Value::Null),
        }),
    );
    Some(Value::Object(obj))
}

fn provider_tool_data_for_image_generation(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let mut obj = build_output_item_base("image_generation", tool_call_id, item_id.as_ref(), true);
    obj.insert("input".into(), json!({}));
    obj.insert(
        "result".into(),
        json!({
            "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
        }),
    );
    Some(Value::Object(obj))
}

fn provider_tool_data_for_computer_call(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let mut obj = build_output_item_base("computer_use", tool_call_id, item_id.as_ref(), true);
    obj.insert("input".into(), json!(""));
    let status = item
        .get("status")
        .cloned()
        .unwrap_or_else(|| json!("completed"));
    obj.insert(
        "result".into(),
        json!({
            "type": "computer_use_tool_result",
            "status": status,
        }),
    );
    Some(Value::Object(obj))
}

fn provider_tool_data_for_local_shell(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = output_item_call_id(item, item_id)?;
    let mut obj = build_output_item_base("local_shell", tool_call_id, item_id.as_ref(), false);
    obj.insert("input".into(), map_local_shell_input(item));
    Some(Value::Object(obj))
}

fn provider_tool_data_for_shell(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = output_item_call_id(item, item_id)?;
    let mut obj = build_output_item_base("shell", tool_call_id, item_id.as_ref(), false);
    obj.insert("input".into(), map_shell_input(item));
    Some(Value::Object(obj))
}

fn provider_tool_data_for_apply_patch(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = output_item_call_id(item, item_id)?;
    let mut obj =
        build_output_item_base("apply_patch", tool_call_id.clone(), item_id.as_ref(), false);
    obj.insert(
        "input".into(),
        json!({
            "callId": tool_call_id,
            "operation": item.get("operation").cloned().unwrap_or(serde_json::Value::Null),
        }),
    );
    Some(Value::Object(obj))
}

fn provider_tool_data_for_mcp_call(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let name = item
        .get("name")
        .and_then(|value| value.as_str())?
        .to_string();
    let mut obj = build_output_item_base("mcp", tool_call_id, item_id.as_ref(), true);
    obj.insert(
        "input".into(),
        item.get("arguments").cloned().unwrap_or_else(|| json!("")),
    );
    obj.insert("mcp_name".into(), json!(name.clone()));
    if let Some(approval_request_id) = item
        .get("approval_request_id")
        .and_then(|value| value.as_str())
    {
        obj.insert("approval_request_id".into(), json!(approval_request_id));
    }
    if let Some(server_label) = item.get("server_label") {
        obj.insert("server_label".into(), server_label.clone());
    }
    let mut result = serde_json::Map::new();
    result.insert("type".into(), json!("call"));
    if let Some(server_label) = item.get("server_label") {
        result.insert("serverLabel".into(), server_label.clone());
    }
    result.insert("name".into(), json!(name));
    result.insert(
        "arguments".into(),
        item.get("arguments").cloned().unwrap_or_else(|| json!("")),
    );
    if let Some(output) = item.get("output") {
        result.insert("output".into(), output.clone());
    }
    if let Some(error) = item.get("error") {
        result.insert("error".into(), error.clone());
    }
    obj.insert("result".into(), Value::Object(result));
    Some(Value::Object(obj))
}

fn provider_tool_data_for_mcp_approval_request(
    item: &serde_json::Map<String, Value>,
    item_id: &Option<String>,
) -> Option<Value> {
    let tool_call_id = item_id.clone()?;
    let name = item
        .get("name")
        .and_then(|value| value.as_str())?
        .to_string();
    let mut obj = build_output_item_base("mcp", tool_call_id, item_id.as_ref(), true);
    obj.insert(
        "input".into(),
        item.get("arguments").cloned().unwrap_or_else(|| json!("")),
    );
    obj.insert("mcp_name".into(), json!(name));
    let approval_request_id = item
        .get("approval_request_id")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
        .or_else(|| item_id.clone());
    if let Some(approval_request_id) = approval_request_id {
        obj.insert("approval_request_id".into(), json!(approval_request_id));
    }
    obj.insert("approval_request".into(), json!(true));
    if let Some(server_label) = item.get("server_label") {
        obj.insert("server_label".into(), server_label.clone());
    }
    Some(Value::Object(obj))
}

type OutputItemMapper = fn(&serde_json::Map<String, Value>, &Option<String>) -> Option<Value>;

const OUTPUT_ITEM_MAPPERS: &[(&str, OutputItemMapper)] = &[
    ("web_search_call", provider_tool_data_for_web_search),
    ("file_search_call", provider_tool_data_for_file_search),
    (
        "code_interpreter_call",
        provider_tool_data_for_code_interpreter,
    ),
    (
        "image_generation_call",
        provider_tool_data_for_image_generation,
    ),
    ("computer_call", provider_tool_data_for_computer_call),
    ("local_shell_call", provider_tool_data_for_local_shell),
    ("shell_call", provider_tool_data_for_shell),
    ("apply_patch_call", provider_tool_data_for_apply_patch),
    ("mcp_call", provider_tool_data_for_mcp_call),
    (
        "mcp_approval_request",
        provider_tool_data_for_mcp_approval_request,
    ),
];

pub(super) fn provider_tool_data_from_output_item(
    item: &serde_json::Map<String, Value>,
) -> Option<serde_json::Value> {
    let item_type = item.get("type")?.as_str()?;
    let item_id = item
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    OUTPUT_ITEM_MAPPERS
        .iter()
        .find(|(candidate, _)| *candidate == item_type)
        .and_then(|(_, mapper)| mapper(item, &item_id))
}

pub(super) struct ProviderToolParts {
    pub(super) tool_call_id: String,
    pub(super) tool_name: String,
    pub(super) tool_type: String,
    pub(super) input: String,
    pub(super) provider_executed: bool,
    pub(super) dynamic: bool,
    pub(super) result: Option<serde_json::Value>,
    pub(super) is_error: bool,
    pub(super) provider_metadata: Option<v2t::ProviderMetadata>,
    pub(super) approval_request_id: Option<String>,
    pub(super) is_approval_request: bool,
}

pub(super) fn provider_tool_parts_from_data(
    data: &serde_json::Value,
    tool_name_mapping: &ToolNameMapping,
) -> Option<ProviderToolParts> {
    let obj = data.as_object()?;
    let tool_type = obj.get("tool_type").and_then(|v| v.as_str()).unwrap_or("");
    let tool_call_id = obj
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if tool_call_id.is_empty() {
        return None;
    }
    let input_val = obj.get("input").cloned().unwrap_or(serde_json::Value::Null);
    let input = match input_val {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    };
    let dynamic = tool_type == "mcp";
    let tool_name = if tool_type == "mcp" {
        obj.get("mcp_name")
            .and_then(|v| v.as_str())
            .map(|name| format!("mcp.{name}"))
            .unwrap_or_else(|| "mcp".into())
    } else if tool_type == "web_search" {
        tool_name_mapping
            .web_search_tool_name
            .clone()
            .unwrap_or_else(|| {
                tool_name_mapping
                    .to_custom_tool_name("web_search")
                    .to_string()
            })
    } else {
        tool_name_mapping.to_custom_tool_name(tool_type).to_string()
    };
    let provider_executed = obj
        .get("provider_executed")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let result = obj.get("result").cloned().filter(|v| !v.is_null());
    let is_error = obj
        .get("is_error")
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| {
            result
                .as_ref()
                .and_then(|val| val.get("error"))
                .map(|v| !v.is_null())
                .unwrap_or(false)
        });
    let provider_metadata = obj.get("item_id").and_then(|v| v.as_str()).map(|id| {
        let mut inner = HashMap::new();
        inner.insert("itemId".into(), serde_json::json!(id));
        let mut outer = HashMap::new();
        outer.insert("openai".into(), inner);
        outer
    });
    let approval_request_id = obj
        .get("approval_request_id")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let is_approval_request = obj
        .get("approval_request")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(ProviderToolParts {
        tool_call_id,
        tool_name,
        tool_type: tool_type.to_string(),
        input,
        provider_executed,
        dynamic,
        result,
        is_error,
        provider_metadata,
        approval_request_id,
        is_approval_request,
    })
}
