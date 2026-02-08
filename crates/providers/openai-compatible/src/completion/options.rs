use crate::ai_sdk_types::v2 as v2t;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAICompatibleCompletionProviderOptions {
    pub echo: Option<bool>,
    pub logit_bias: Option<HashMap<String, f64>>, // token id -> bias
    pub suffix: Option<String>,
    pub user: Option<String>,
}

/// Parse providerOptions for the given provider scope names.
/// Later scopes override earlier scopes; extras are taken from the last scope.
pub fn parse_openai_compatible_completion_provider_options(
    provider_options: &v2t::ProviderOptions,
    provider_scope_names: &[&str],
) -> (
    OpenAICompatibleCompletionProviderOptions,
    Option<JsonMap<String, JsonValue>>,
) {
    let mut merged = OpenAICompatibleCompletionProviderOptions::default();
    let mut found = false;
    for name in provider_scope_names {
        if let Some(map) = provider_options.get(*name) {
            found = true;
            if let Some(echo) = map.get("echo").and_then(|v| v.as_bool()) {
                merged.echo = Some(echo);
            }
            if let Some(logit_bias) = map
                .get("logitBias")
                .and_then(|v| serde_json::from_value::<HashMap<String, f64>>(v.clone()).ok())
            {
                merged.logit_bias = Some(logit_bias);
            }
            if let Some(suffix) = map.get("suffix").and_then(|v| v.as_str()) {
                merged.suffix = Some(suffix.to_string());
            }
            if let Some(user) = map.get("user").and_then(|v| v.as_str()) {
                merged.user = Some(user.to_string());
            }
        }
    }

    if !found {
        return (OpenAICompatibleCompletionProviderOptions::default(), None);
    }

    let extras = provider_scope_names
        .last()
        .and_then(|name| provider_options.get(*name))
        .map(|map| {
            let mut extras = JsonMap::new();
            for (k, v) in map.iter() {
                if k == "echo" || k == "logitBias" || k == "suffix" || k == "user" {
                    continue;
                }
                extras.insert(k.clone(), v.clone());
            }
            extras
        });

    (merged, extras)
}
