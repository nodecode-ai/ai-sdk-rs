use crate::types::v2 as v2t;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::HashMap;

const KNOWN_COMPLETION_PROVIDER_OPTION_KEYS: &[&str] = &["echo", "logitBias", "suffix", "user"];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAICompatibleCompletionProviderOptions {
    pub echo: Option<bool>,
    #[serde(rename = "logitBias", alias = "logit_bias")]
    pub logit_bias: Option<HashMap<String, f64>>, // token id -> bias
    pub suffix: Option<String>,
    pub user: Option<String>,
}

fn parse_completion_scope_options(
    map: &HashMap<String, JsonValue>,
) -> OpenAICompatibleCompletionProviderOptions {
    let value = JsonValue::Object(
        map.iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    );
    serde_json::from_value(value).unwrap_or_default()
}

fn merge_completion_scope_options(
    merged: &mut OpenAICompatibleCompletionProviderOptions,
    parsed: OpenAICompatibleCompletionProviderOptions,
) {
    if let Some(echo) = parsed.echo {
        merged.echo = Some(echo);
    }
    if let Some(logit_bias) = parsed.logit_bias {
        merged.logit_bias = Some(logit_bias);
    }
    if let Some(suffix) = parsed.suffix {
        merged.suffix = Some(suffix);
    }
    if let Some(user) = parsed.user {
        merged.user = Some(user);
    }
}

fn collect_completion_provider_extras(
    provider_options: &v2t::ProviderOptions,
    provider_scope_names: &[&str],
) -> Option<JsonMap<String, JsonValue>> {
    provider_scope_names
        .last()
        .and_then(|name| provider_options.get(*name))
        .map(|map| {
            map.iter()
                .filter(|(key, _)| !KNOWN_COMPLETION_PROVIDER_OPTION_KEYS.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
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
        let Some(map) = provider_options.get(*name) else {
            continue;
        };
        found = true;
        merge_completion_scope_options(&mut merged, parse_completion_scope_options(map));
    }

    if !found {
        return (OpenAICompatibleCompletionProviderOptions::default(), None);
    }

    (
        merged,
        collect_completion_provider_extras(provider_options, provider_scope_names),
    )
}
