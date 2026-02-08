//! Provider defaults and request overrides.
//!
//! Precedence order:
//! 1) Explicit call options take priority.
//! 2) Provider defaults fill missing values for the exact provider scope.
//! 3) Request overrides are merged last into the request body (minus disallowed keys).
//!
//! Only exact provider scope keys are accepted; inline or aliased scopes are ignored.

use std::collections::HashMap;

use crate::ai_sdk_types::embedding::EmbedOptions;
use crate::ai_sdk_types::image::ImageOptions;
use crate::ai_sdk_types::v2 as v2t;
use serde_json::Value as JsonValue;

fn json_object_to_hashmap(map: &serde_json::Map<String, JsonValue>) -> HashMap<String, JsonValue> {
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

fn scoped_object<'a>(
    provider_scope: &str,
    value: &'a JsonValue,
) -> Option<&'a serde_json::Map<String, JsonValue>> {
    value.as_object()?.get(provider_scope)?.as_object()
}

fn merge_json_defaults(target: &mut JsonValue, defaults: &JsonValue) {
    match (target, defaults) {
        (JsonValue::Object(target_map), JsonValue::Object(defaults_map)) => {
            for (k, v) in defaults_map {
                match target_map.get_mut(k) {
                    Some(existing) => merge_json_defaults(existing, v),
                    None => {
                        target_map.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        _ => {}
    }
}

fn merge_scope_defaults(
    target: &mut v2t::ProviderOptions,
    provider_scope: &str,
    defaults: &HashMap<String, JsonValue>,
) {
    let entry = target.entry(provider_scope.to_string()).or_default();
    for (key, val) in defaults {
        match entry.get_mut(key) {
            Some(existing) => merge_json_defaults(existing, val),
            None => {
                entry.insert(key.clone(), val.clone());
            }
        }
    }
}

/// Merge provider default options into a target provider options map without
/// overriding explicit values.
pub fn merge_provider_defaults(target: &mut v2t::ProviderOptions, defaults: &v2t::ProviderOptions) {
    for (scope, defs) in defaults {
        merge_scope_defaults(target, scope, defs);
    }
}

/// Parse provider defaults from a JSON blob (e.g., X-AI-SDK-Options header).
///
/// Only entries under the exact provider scope are accepted.
pub fn provider_defaults_from_json(
    provider_scope: &str,
    value: &JsonValue,
) -> Option<v2t::ProviderOptions> {
    let scoped = scoped_object(provider_scope, value)?;
    let mut wrapped: v2t::ProviderOptions = HashMap::new();
    wrapped.insert(provider_scope.to_string(), json_object_to_hashmap(scoped));
    Some(wrapped)
}

/// Resolve provider-scoped request overrides from a JSON blob (e.g., X-AI-SDK-Options).
/// Only entries under the exact provider scope are accepted.
pub fn request_overrides_from_json(provider_scope: &str, value: &JsonValue) -> Option<JsonValue> {
    let scoped = scoped_object(provider_scope, value)?;
    Some(JsonValue::Object(scoped.clone()))
}

/// Merge config defaults into call options for the exact provider scope.
/// Returns the merged options for chaining convenience.
pub fn build_call_options(
    mut opts: v2t::CallOptions,
    provider_scope: &str,
    config_defaults: Option<&v2t::ProviderOptions>,
) -> v2t::CallOptions {
    if let Some(defaults) = config_defaults {
        if let Some(scope_defaults) = defaults.get(provider_scope) {
            merge_scope_defaults(&mut opts.provider_options, provider_scope, scope_defaults);
        }
    }
    opts
}

/// Merge config defaults into embedding options for the exact provider scope.
pub fn build_embed_options(
    mut opts: EmbedOptions,
    provider_scope: &str,
    config_defaults: Option<&v2t::ProviderOptions>,
) -> EmbedOptions {
    if let Some(defaults) = config_defaults {
        if let Some(scope_defaults) = defaults.get(provider_scope) {
            merge_scope_defaults(&mut opts.provider_options, provider_scope, scope_defaults);
        }
    }
    opts
}

/// Merge config defaults into image options for the exact provider scope.
pub fn build_image_options(
    mut opts: ImageOptions,
    provider_scope: &str,
    config_defaults: Option<&v2t::ProviderOptions>,
) -> ImageOptions {
    if let Some(defaults) = config_defaults {
        if let Some(scope_defaults) = defaults.get(provider_scope) {
            merge_scope_defaults(&mut opts.provider_options, provider_scope, scope_defaults);
        }
    }
    opts
}
