use crate::ai_sdk_types::image::ImageOptions as ImageCallOptions;
use crate::ai_sdk_types::v2 as v2t;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAICompatibleImageProviderOptions {
    pub user: Option<String>,
}

/// Parse providerOptions for the OpenAI scope only.
/// Returns typed options and a map of extra keys not covered by typed fields.
pub fn parse_openai_compatible_image_provider_options(
    provider_options: &v2t::ProviderOptions,
) -> (
    OpenAICompatibleImageProviderOptions,
    Option<JsonMap<String, JsonValue>>,
) {
    let Some(map) = provider_options.get("openai") else {
        return (OpenAICompatibleImageProviderOptions::default(), None);
    };

    let user = map
        .get("user")
        .and_then(|v| v.as_str())
        .map(|value| value.to_string());

    let mut extras = JsonMap::new();
    for (k, v) in map.iter() {
        if k == "user" {
            continue;
        }
        extras.insert(k.clone(), v.clone());
    }

    let extras = if extras.is_empty() {
        None
    } else {
        Some(extras)
    };

    (OpenAICompatibleImageProviderOptions { user }, extras)
}

/// Merge provider defaults into the given image options.
pub fn apply_provider_defaults(
    opts: ImageCallOptions,
    provider_scope_name: &str,
    defaults: Option<&v2t::ProviderOptions>,
) -> ImageCallOptions {
    crate::ai_sdk_core::request_builder::defaults::build_image_options(opts, provider_scope_name, defaults)
}
