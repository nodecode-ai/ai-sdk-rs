use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_provider::{
    apply_stream_idle_timeout_ms, registry::ProviderRegistration, Credentials,
};
use crate::ai_sdk_types::catalog::{ProviderDefinition, SdkType};
use crate::ai_sdk_types::v2 as v2t;
use serde_json::Value as JsonValue;

use crate::provider_gateway::config::{GatewayAuth, GatewayAuthMethod, GatewayConfig};
use crate::provider_gateway::language_model::GatewayLanguageModel;

const DEFAULT_BASE_URL: &str = "https://ai-gateway.vercel.sh/v1/ai";
const AI_GATEWAY_PROTOCOL_VERSION: &str = "0.0.1";

fn match_gateway(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::Gateway)
}

fn normalize_endpoint_path(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        None
    } else if trimmed.starts_with('/') {
        Some(trimmed.to_string())
    } else {
        Some(format!("/{}", trimmed))
    }
}

fn resolve_auth(creds: &Credentials) -> Option<GatewayAuth> {
    match creds {
        Credentials::ApiKey(value) => to_api_key_auth(value.clone()),
        Credentials::Bearer(value) => to_oidc_auth(value.clone()),
        Credentials::None => {
            if let Ok(key) = std::env::var("AI_GATEWAY_API_KEY") {
                if !key.trim().is_empty() {
                    return to_api_key_auth(key);
                }
            }
            if let Ok(token) = std::env::var("VERCEL_OIDC_TOKEN") {
                if !token.trim().is_empty() {
                    return to_oidc_auth(token);
                }
            }
            None
        }
    }
}

fn to_api_key_auth(value: String) -> Option<GatewayAuth> {
    let token = value.trim();
    if token.is_empty() {
        None
    } else {
        Some(GatewayAuth {
            token: token.to_string(),
            method: GatewayAuthMethod::ApiKey,
        })
    }
}

fn to_oidc_auth(value: String) -> Option<GatewayAuth> {
    let token = value.trim();
    if token.is_empty() {
        None
    } else {
        let stripped = token
            .strip_prefix("Bearer ")
            .or_else(|| token.strip_prefix("bearer "))
            .unwrap_or(token);
        Some(GatewayAuth {
            token: stripped.to_string(),
            method: GatewayAuthMethod::Oidc,
        })
    }
}

fn filter_headers(
    headers: &HashMap<String, String>,
    provider_scope: &str,
) -> (
    Vec<(String, String)>,
    Option<v2t::ProviderOptions>,
    Option<JsonValue>,
) {
    let mut filtered: Vec<(String, String)> = Vec::new();
    let mut defaults: Option<v2t::ProviderOptions> = None;
    let mut raw: Option<JsonValue> = None;

    for (k, v) in headers {
        if sdkopt::is_internal_sdk_header(k) {
            if raw.is_none() {
                if let Ok(json) = serde_json::from_str::<JsonValue>(v) {
                    if defaults.is_none() {
                        defaults = provider_defaults_from_json(provider_scope, &json);
                    }
                    raw = Some(json);
                }
            }
            continue;
        }
        let key = k.to_ascii_lowercase();
        if matches!(
            key.as_str(),
            "content-type"
                | "accept"
                | "authorization"
                | "x-api-key"
                | "ai-gateway-auth-method"
                | "ai-gateway-protocol-version"
        ) {
            continue;
        }
        filtered.push((key, v.clone()));
    }

    (filtered, defaults, raw)
}

fn build_gateway(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let auth = resolve_auth(creds);

    let base_url = if def.base_url.trim().is_empty() {
        DEFAULT_BASE_URL.to_string()
    } else {
        def.base_url.trim_end_matches('/').to_string()
    };

    let (extra_headers, default_options, request_defaults) =
        filter_headers(&def.headers, &def.name);

    let mut headers = Vec::new();
    headers.push((
        "ai-gateway-protocol-version".to_string(),
        AI_GATEWAY_PROTOCOL_VERSION.to_string(),
    ));
    headers.extend(extra_headers);

    let endpoint_path = match normalize_endpoint_path(&def.endpoint_path) {
        Some(path) => Some(path),
        None => {
            let base = base_url.trim_end_matches('/');
            if base.ends_with("/language-model") {
                None
            } else {
                Some("/language-model".to_string())
            }
        }
    };

    let mut transport_cfg = TransportConfig::default();
    transport_cfg.idle_read_timeout = Duration::from_secs(45);
    apply_stream_idle_timeout_ms(def, &mut transport_cfg);

    let http = crate::reqwest_transport::ReqwestTransport::new(&transport_cfg);

    let supported_urls = HashMap::from([("*/*".to_string(), vec![r"^.*$".to_string()])]);

    let config = GatewayConfig {
        provider_name: "gateway",
        provider_scope_name: def.name.clone(),
        base_url,
        endpoint_path,
        headers,
        query_params: def
            .query_params
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        supported_urls,
        transport_cfg: transport_cfg.clone(),
        default_options,
        request_defaults,
        auth,
    };

    let lm = GatewayLanguageModel::new(model.to_string(), config, http);
    Ok(Arc::new(lm))
}

inventory::submit! {
    ProviderRegistration {
        id: "gateway",
        sdk_type: SdkType::Gateway,
        matches: Some(match_gateway),
        build: build_gateway,
        reasoning_scope: None,
    }
}
