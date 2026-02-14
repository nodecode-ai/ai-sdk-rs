use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_provider::{
    build_provider_transport_config, registry::ProviderRegistration, Credentials,
    ReasoningScopeContext,
};
use crate::ai_sdk_types::catalog::{ProviderDefinition, SdkType};
use crate::ai_sdk_types::v2 as v2t;
use serde_json::Value as JsonValue;

use crate::provider_anthropic::messages::language_model::{
    AnthropicMessagesConfig, AnthropicMessagesLanguageModel,
};
const OAUTH_BETA_HEADER_VALUE: &str = "oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14";
const DEFAULT_IDLE_READ_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_STAINLESS_TIMEOUT_SECS: u64 = 600;

fn oauth_required_headers() -> Vec<(String, String)> {
    let pkg_version = env!("CARGO_PKG_VERSION");
    let arch = std::env::consts::ARCH;
    let os = std::env::consts::OS;

    vec![
        ("accept".to_string(), "application/json".to_string()),
        ("accept-language".to_string(), "*".to_string()),
        (
            "anthropic-beta".to_string(),
            OAUTH_BETA_HEADER_VALUE.to_string(),
        ),
        (
            "anthropic-dangerous-direct-browser-access".to_string(),
            "true".to_string(),
        ),
        ("connection".to_string(), "keep-alive".to_string()),
        ("sec-fetch-mode".to_string(), "cors".to_string()),
        (
            "user-agent".to_string(),
            format!("clixode-ai-sdk/{pkg_version} (oauth)"),
        ),
        ("x-app".to_string(), "cli".to_string()),
        ("x-stainless-arch".to_string(), arch.to_string()),
        ("x-stainless-lang".to_string(), "rust".to_string()),
        ("x-stainless-os".to_string(), os.to_string()),
        (
            "x-stainless-package-version".to_string(),
            pkg_version.to_string(),
        ),
        ("x-stainless-retry-count".to_string(), "0".to_string()),
        ("x-stainless-runtime".to_string(), "rust".to_string()),
        (
            "x-stainless-runtime-version".to_string(),
            pkg_version.to_string(),
        ),
        (
            "x-stainless-timeout".to_string(),
            DEFAULT_STAINLESS_TIMEOUT_SECS.to_string(),
        ),
    ]
}

pub fn default_headers_from_creds(
    api_key: Option<String>,
    bearer: Option<String>,
) -> Vec<(String, String)> {
    let mut h = vec![
        ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ("content-type".to_string(), "application/json".to_string()),
        (
            "x-stainless-timeout".to_string(),
            DEFAULT_STAINLESS_TIMEOUT_SECS.to_string(),
        ),
    ];
    if let Some(b) = bearer {
        h.push((
            "authorization".into(),
            if b.to_lowercase().starts_with("bearer ") {
                b
            } else {
                format!("Bearer {}", b)
            },
        ));
        h.extend(oauth_required_headers());
    } else if let Some(k) = api_key {
        h.push(("x-api-key".into(), k));
    }
    h
}

fn match_anthropic(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::Anthropic)
}

fn build_anthropic(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    // Resolve credentials
    let api_key = creds
        .as_api_key()
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());
    let bearer = creds.as_bearer();

    let base_url = if def.base_url.trim().is_empty() {
        "https://api.anthropic.com/v1".to_string()
    } else {
        def.base_url.clone()
    };

    let mut header_map: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (k, v) in default_headers_from_creds(api_key, bearer) {
        header_map.insert(k.to_ascii_lowercase(), (k, v));
    }
    for (k, v) in &def.headers {
        header_map.insert(k.to_ascii_lowercase(), (k.clone(), v.clone()));
    }

    let mut default_options: Option<v2t::ProviderOptions> = None;
    let mut headers: Vec<(String, String)> = Vec::new();
    for (_lc, (orig, value)) in header_map.into_iter() {
        if sdkopt::is_internal_sdk_header(&orig) {
            if default_options.is_none() {
                if let Ok(json) = serde_json::from_str::<JsonValue>(&value) {
                    if let Some(parsed) = provider_defaults_from_json(&def.name, &json) {
                        default_options = Some(parsed);
                    }
                }
            }
            continue;
        }
        headers.push((orig, value));
    }

    let transport_cfg = build_provider_transport_config(def, Some(DEFAULT_IDLE_READ_TIMEOUT));

    let http = crate::reqwest_transport::ReqwestTransport::try_new(&transport_cfg)
        .map_err(SdkError::Transport)?;

    let supported_urls =
        HashMap::from([("image/*".to_string(), vec![r"^https?://.*$".to_string()])]);

    let lm = AnthropicMessagesLanguageModel::new(
        model.to_string(),
        AnthropicMessagesConfig {
            provider_name: "anthropic.messages",
            provider_scope_name: def.name.clone(),
            base_url,
            headers,
            http,
            transport_cfg,
            supported_urls,
            default_options,
        },
    );

    Ok(Arc::new(lm))
}

fn anthropic_reasoning_scope(_ctx: &ReasoningScopeContext) -> Option<Vec<String>> {
    Some(vec!["anthropic".to_string()])
}

inventory::submit! {
    ProviderRegistration {
        id: "anthropic",
        sdk_type: SdkType::Anthropic,
        matches: Some(match_anthropic),
        build: build_anthropic,
        reasoning_scope: Some(anthropic_reasoning_scope),
    }
}
