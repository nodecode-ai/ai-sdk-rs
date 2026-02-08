use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::ai_sdk_types::v2::ProviderOptions;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BedrockProviderOptions {
    #[serde(
        default,
        rename = "additionalModelRequestFields",
        alias = "additional_model_request_fields",
        skip_serializing_if = "Option::is_none"
    )]
    pub additional_model_request_fields: Option<JsonMap<String, JsonValue>>,
    #[serde(
        default,
        rename = "reasoningConfig",
        alias = "reasoning_config",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_config: Option<BedrockReasoningConfig>,
    #[serde(
        default,
        rename = "guardrailConfig",
        alias = "guardrail_config",
        skip_serializing_if = "Option::is_none"
    )]
    pub guardrail_config: Option<JsonMap<String, JsonValue>>,
    #[serde(
        default,
        rename = "guardrailStreamConfig",
        alias = "guardrail_stream_config",
        skip_serializing_if = "Option::is_none"
    )]
    pub guardrail_stream_config: Option<JsonMap<String, JsonValue>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BedrockReasoningConfig {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(rename = "budgetTokens", alias = "budget_tokens")]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BedrockReasoningMetadata {
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(rename = "redactedData", alias = "redacted_data")]
    pub redacted_data: Option<String>,
}

/// Extract the `bedrock` provider option block from call options.
pub fn parse_bedrock_provider_options(opts: &ProviderOptions) -> Option<BedrockProviderOptions> {
    let map = opts.get("bedrock")?;
    let value = JsonValue::Object(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    serde_json::from_value(value).ok()
}

/// Parse reasoning metadata for assistant reasoning parts coming from provider options.
pub fn parse_reasoning_metadata(
    opts: &Option<ProviderOptions>,
) -> Option<BedrockReasoningMetadata> {
    let map = opts.as_ref()?.get("bedrock")?;
    let value = JsonValue::Object(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    serde_json::from_value(value).ok()
}

/// Determine whether the provider options include a cache point marker.
pub fn has_cache_point(opts: &Option<ProviderOptions>) -> bool {
    let Some(map) = opts.as_ref() else {
        return false;
    };
    let Some(bedrock) = map.get("bedrock") else {
        return false;
    };
    match bedrock
        .get("cachePoint")
        .or_else(|| bedrock.get("cache_point"))
    {
        Some(JsonValue::Object(obj)) => obj.get("type").is_some(),
        _ => false,
    }
}

/// Helper to merge JSON maps (shallow, overriding destination keys).
pub fn merge_json_maps(
    target: &mut JsonMap<String, JsonValue>,
    additional: &JsonMap<String, JsonValue>,
) {
    for (k, v) in additional.iter() {
        target.insert(k.clone(), v.clone());
    }
}

/// Convert an optional JSON map into a plain hashmap of serde_json::Value (clone helper).
pub fn map_to_owned(
    map: &Option<JsonMap<String, JsonValue>>,
) -> Option<JsonMap<String, JsonValue>> {
    map.as_ref()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}
