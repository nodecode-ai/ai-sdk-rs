use crate::ai_sdk_types::v2::ProviderOptions;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingConfig {
    #[serde(
        rename = "thinkingBudget",
        alias = "thinking_budget",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub thinking_budget: Option<u32>,
    #[serde(
        rename = "includeThoughts",
        alias = "include_thoughts",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub include_thoughts: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoogleProviderOptions {
    #[serde(
        rename = "responseModalities",
        alias = "response_modalities",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub response_modalities: Option<Vec<String>>,
    #[serde(
        rename = "thinkingConfig",
        alias = "thinking_config",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub thinking_config: Option<ThinkingConfig>,
    #[serde(
        rename = "cachedContent",
        alias = "cached_content",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cached_content: Option<String>,
    #[serde(
        rename = "structuredOutputs",
        alias = "structured_outputs",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub structured_outputs: Option<bool>,
    #[serde(
        rename = "safetySettings",
        alias = "safety_settings",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub safety_settings: Option<Vec<SafetySetting>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<String>,
    #[serde(
        rename = "audioTimestamp",
        alias = "audio_timestamp",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub audio_timestamp: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub labels: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

pub fn parse_google_provider_options_for_scopes(
    opts: &ProviderOptions,
    provider_scopes: &[&str],
) -> Option<GoogleProviderOptions> {
    let map = provider_scopes.iter().find_map(|scope| opts.get(*scope))?;
    let v = JsonValue::Object(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    serde_json::from_value::<GoogleProviderOptions>(v).ok()
}
