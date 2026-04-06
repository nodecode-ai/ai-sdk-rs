use crate::ai_sdk_types::embedding::EmbedOptions as EmbeddingCallOptions;
use crate::ai_sdk_types::v2 as v2t;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAICompatibleEmbeddingProviderOptions {
    pub dimensions: Option<u32>,
    pub user: Option<String>,
}

/// Parse providerOptions for the given provider scope names.
/// Later scopes override earlier scopes; extras are taken from the last scope.
pub fn parse_openai_compatible_embedding_provider_options(
    provider_options: &v2t::ProviderOptions,
    provider_scope_names: &[&str],
) -> (
    OpenAICompatibleEmbeddingProviderOptions,
    Option<JsonMap<String, JsonValue>>,
) {
    let mut merged = OpenAICompatibleEmbeddingProviderOptions::default();
    let mut found = false;
    for name in provider_scope_names {
        if let Some(map) = provider_options.get(*name) {
            found = true;
            if let Some(dimensions) = map
                .get("dimensions")
                .and_then(|v| v.as_u64())
                .and_then(|n| u32::try_from(n).ok())
            {
                merged.dimensions = Some(dimensions);
            }
            if let Some(user) = map.get("user").and_then(|v| v.as_str()) {
                merged.user = Some(user.to_string());
            }
        }
    }

    if !found {
        return (OpenAICompatibleEmbeddingProviderOptions::default(), None);
    }

    let extras = provider_scope_names
        .last()
        .and_then(|name| provider_options.get(*name))
        .map(|map| {
            let mut extras = JsonMap::new();
            for (k, v) in map.iter() {
                if k == "dimensions" || k == "user" {
                    continue;
                }
                extras.insert(k.clone(), v.clone());
            }
            extras
        });

    (merged, extras)
}

/// Merge provider defaults into the given embedding options.
pub fn apply_provider_defaults(
    opts: EmbeddingCallOptions,
    provider_scope_name: &str,
    defaults: Option<&v2t::ProviderOptions>,
) -> EmbeddingCallOptions {
    crate::ai_sdk_core::request_builder::defaults::build_embed_options(
        opts,
        provider_scope_name,
        defaults,
    )
}
