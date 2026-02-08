use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::ai_sdk_types::v2::ProviderOptions;

pub type AnthropicMessagesModelId = String;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicFilePartProviderOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<Citations>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citations {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ThinkingOption {
    Enabled {
        #[serde(rename = "budgetTokens", alias = "budget_tokens")]
        budget_tokens: u32,
    },
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicProviderOptions {
    #[serde(
        default,
        rename = "sendReasoning",
        alias = "send_reasoning",
        skip_serializing_if = "Option::is_none"
    )]
    pub send_reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingOption>,
    #[serde(
        default,
        rename = "disableParallelToolUse",
        alias = "disable_parallel_tool_use",
        skip_serializing_if = "Option::is_none"
    )]
    pub disable_parallel_tool_use: Option<bool>,
}

fn provider_scope_from_options<'a>(
    opts: &'a ProviderOptions,
    provider_scope_name: &str,
) -> Option<&'a std::collections::HashMap<String, JsonValue>> {
    opts.get(provider_scope_name)
}

/// Extracts and deserializes the provider-scoped section of provider options.
///
/// Supports custom provider ids (e.g. `"newcli"`) by reading options from the
/// exact `provider_scope_name` key (no fallback).
pub fn parse_anthropic_provider_options(
    opts: &ProviderOptions,
    provider_scope_name: &str,
) -> Option<AnthropicProviderOptions> {
    let map = provider_scope_from_options(opts, provider_scope_name)?;
    let v = JsonValue::Object(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    serde_json::from_value::<AnthropicProviderOptions>(v).ok()
}

/// Extract file-level provider options for Anthropic (citations/title/context).
pub fn parse_anthropic_file_part_options(
    opts: &Option<ProviderOptions>,
    provider_scope_name: &str,
) -> Option<AnthropicFilePartProviderOptions> {
    let map = provider_scope_from_options(opts.as_ref()?, provider_scope_name)?;
    let v = JsonValue::Object(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
    serde_json::from_value::<AnthropicFilePartProviderOptions>(v).ok()
}
