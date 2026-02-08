use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::ai_sdk_types::v2 as v2t;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAICompatibleChatProviderOptions {
    pub user: Option<String>,
    pub reasoning_effort: Option<String>,
    pub text_verbosity: Option<String>,
}

/// Parse providerOptions for the given provider scope names.
/// Later scopes override earlier scopes; extras are taken from the last scope.
pub fn parse_openai_compatible_chat_provider_options(
    provider_options: &v2t::ProviderOptions,
    provider_scope_names: &[&str],
) -> (
    OpenAICompatibleChatProviderOptions,
    Option<JsonMap<String, JsonValue>>,
) {
    let mut merged = OpenAICompatibleChatProviderOptions::default();
    let mut found = false;
    for name in provider_scope_names {
        if let Some(map) = provider_options.get(*name) {
            found = true;
            if let Some(user) = map.get("user").and_then(|v| v.as_str()) {
                merged.user = Some(user.to_string());
            }
            if let Some(reasoning_effort) = map.get("reasoningEffort").and_then(|v| v.as_str()) {
                merged.reasoning_effort = Some(reasoning_effort.to_string());
            }
            if let Some(text_verbosity) = map.get("textVerbosity").and_then(|v| v.as_str()) {
                merged.text_verbosity = Some(text_verbosity.to_string());
            }
        }
    }

    if !found {
        return (OpenAICompatibleChatProviderOptions::default(), None);
    }

    let extras = provider_scope_names
        .last()
        .and_then(|name| provider_options.get(*name))
        .map(|map| {
            let mut extras = JsonMap::new();
            for (k, v) in map.iter() {
                if k == "user" || k == "reasoningEffort" || k == "textVerbosity" {
                    continue;
                }
                extras.insert(k.clone(), v.clone());
            }
            extras
        });

    (merged, extras)
}
