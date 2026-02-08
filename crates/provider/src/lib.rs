//! Provider builder types and registration for ai-sdk-rs.
//!
//! This crate defines lightweight credentials/settings types and an optional
//! registration mechanism that provider crates can use to self-register
//! builders. The factory crate can then either consume registered builders or
//! fall back to sdk_type-based construction.

use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_types::{
    catalog::{ProviderDefinition, SdkType},
    v2::ProviderOptions as V2ProviderOptions,
};
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use url::Url;

/// Credentials provided by the application layer.
#[derive(Debug, Clone)]
pub enum Credentials {
    /// API key string; interpreted per provider (often used as a bearer token).
    ApiKey(String),
    /// Bearer token (OAuth or equivalent), with or without the "Bearer " prefix.
    Bearer(String),
    /// No credentials provided; provider may fall back to environment.
    None,
}

impl Credentials {
    pub fn as_bearer(&self) -> Option<String> {
        match self {
            Credentials::Bearer(s) => Some(if s.to_lowercase().starts_with("bearer ") {
                s.clone()
            } else {
                format!("Bearer {}", s)
            }),
            _ => None,
        }
    }
    pub fn as_api_key(&self) -> Option<String> {
        match self {
            Credentials::ApiKey(s) => Some(s.clone()),
            _ => None,
        }
    }
}

/// Optional registration of provider builders.
///
/// Providers may `inventory::submit!` a registration to avoid factory matches.
pub mod registry {
    use super::*;
    use crate::ai_sdk_types::catalog::SdkType;

    /// Static registration record for a provider builder.
    pub struct ProviderRegistration {
        /// Identifier for this builder (e.g., "openai", "anthropic", "google", "groq").
        pub id: &'static str,
        /// Canonical SDK type for this provider implementation.
        pub sdk_type: SdkType,
        /// A predicate that decides if this builder wants to handle the given definition.
        /// If `None`, the factory will match by `id` vs `def.sdk_type`.
        pub matches: Option<fn(&ProviderDefinition) -> bool>,
        /// Build a language model given the definition, model id and credentials.
        pub build: fn(
            def: &ProviderDefinition,
            model: &str,
            creds: &Credentials,
        ) -> Result<Arc<dyn LanguageModel>, SdkError>,
        /// Optional reasoning scope alias resolver.
        pub reasoning_scope: Option<fn(&ReasoningScopeContext) -> Option<Vec<String>>>,
    }

    inventory::collect!(ProviderRegistration);

    /// Iterate over all registered builders.
    pub fn iter() -> inventory::iter<ProviderRegistration> {
        inventory::iter::<ProviderRegistration>
    }
}

/// Resolve an `SdkType` from a provider id using registered builders.
///
/// This treats the provider registry as the single source of truth for
/// id→sdk_type mapping, avoiding ad-hoc parsers.
pub fn sdk_type_from_id(id: &str) -> Option<SdkType> {
    let needle = id.trim();
    if needle.is_empty() {
        return None;
    }
    let needle_lc = needle.to_ascii_lowercase();
    for reg in registry::iter() {
        if reg.id.eq_ignore_ascii_case(&needle_lc) {
            return Some(reg.sdk_type.clone());
        }
    }
    None
}

/// Apply a per-provider streaming idle timeout override to the transport config.
pub fn apply_stream_idle_timeout_ms(def: &ProviderDefinition, cfg: &mut TransportConfig) {
    if let Some(ms) = def.stream_idle_timeout_ms {
        if ms > 0 {
            cfg.idle_read_timeout = Duration::from_millis(ms);
        }
    }
}

/// Context passed to reasoning scope hooks registered by providers.
pub struct ReasoningScopeContext<'a> {
    pub provider_id: &'a str,
    pub sdk_type: &'a SdkType,
    pub model_id: Option<&'a str>,
    pub base_url: Option<&'a str>,
}

impl<'a> ReasoningScopeContext<'a> {
    fn normalized_provider_id(&self) -> String {
        normalize_provider_id(self.provider_id)
    }

    fn base_url_host(&self) -> Option<String> {
        let value = self.base_url?.trim();
        if value.is_empty() {
            return None;
        }
        Url::parse(value)
            .ok()
            .and_then(|parsed| parsed.host_str().map(|h| h.to_ascii_lowercase()))
    }
}

fn normalize_provider_id(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    let mut last_dash = false;
    for ch in name.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

/// Resolve provider-specific reasoning scope aliases using registry hooks.
pub fn reasoning_scope_aliases(
    provider_id: &str,
    sdk_type: &SdkType,
    model_id: Option<&str>,
    base_url: Option<&str>,
) -> Option<Vec<String>> {
    let ctx = ReasoningScopeContext {
        provider_id,
        sdk_type,
        model_id,
        base_url,
    };

    let mut aliases = Vec::new();
    let mut seen = HashSet::new();
    let mut handled = false;

    let mut push = |value: &str| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return;
        }
        let lower = trimmed.to_ascii_lowercase();
        if seen.insert(lower) {
            aliases.push(trimmed.to_string());
        }
    };

    for reg in registry::iter() {
        if !(reg.id.eq_ignore_ascii_case(provider_id) || &reg.sdk_type == sdk_type) {
            continue;
        }
        if let Some(handler) = reg.reasoning_scope {
            if let Some(extra) = handler(&ctx) {
                for alias in extra {
                    push(&alias);
                }
                handled = true;
            }
        }
    }

    if !handled {
        return None;
    }

    push(provider_id);
    push(&ctx.normalized_provider_id());
    if let Some(host) = ctx.base_url_host() {
        push(&host);
    }

    Some(aliases)
}

fn build_options_from_scope(
    aliases: &[String],
    scope: &JsonMap<String, JsonValue>,
) -> Option<V2ProviderOptions> {
    if aliases.is_empty() || scope.is_empty() {
        return None;
    }
    let mut opts = V2ProviderOptions::new();
    let scope_map: HashMap<String, JsonValue> =
        scope.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    for alias in aliases {
        opts.insert(alias.clone(), scope_map.clone());
    }
    Some(opts)
}

/// Build provider options for streaming reasoning metadata (signature/redacted data).
pub fn reasoning_stream_options(
    provider_id: &str,
    sdk_type: &SdkType,
    model_id: Option<&str>,
    base_url: Option<&str>,
    signature: Option<&str>,
    redacted_data: Option<&str>,
) -> Option<V2ProviderOptions> {
    let aliases = reasoning_scope_aliases(provider_id, sdk_type, model_id, base_url)?;
    let mut scope = JsonMap::new();
    if let Some(sig) = signature {
        if !sig.is_empty() {
            scope.insert("signature".to_string(), JsonValue::String(sig.to_string()));
        }
    }
    if let Some(data) = redacted_data {
        if !data.is_empty() {
            scope.insert(
                "redactedData".to_string(),
                JsonValue::String(data.to_string()),
            );
        }
    }
    build_options_from_scope(&aliases, &scope)
}

/// Build provider options for persisted reasoning metadata stored on messages.
pub fn persisted_reasoning_options(
    provider_id: &str,
    sdk_type: &SdkType,
    model_id: Option<&str>,
    base_url: Option<&str>,
    text: &str,
    signature: Option<&str>,
) -> Option<V2ProviderOptions> {
    let aliases = reasoning_scope_aliases(provider_id, sdk_type, model_id, base_url)?;
    if text.trim().is_empty() {
        return None;
    }
    let mut scope = JsonMap::new();
    scope.insert(
        "persistedReasoningText".to_string(),
        JsonValue::String(text.to_string()),
    );
    if let Some(sig) = signature {
        if !sig.is_empty() {
            scope.insert(
                "persistedReasoningSignature".to_string(),
                JsonValue::String(sig.to_string()),
            );
        }
    }
    build_options_from_scope(&aliases, &scope)
}
