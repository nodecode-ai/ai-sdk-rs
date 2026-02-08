use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_core::{LanguageModel, SdkError, TransportError};
use crate::ai_sdk_provider::{apply_stream_idle_timeout_ms, registry::ProviderRegistration, Credentials};
use crate::ai_sdk_providers_openai::config::OpenAIConfig;
use crate::ai_sdk_providers_openai::responses::language_model::OpenAIResponsesLanguageModel;
use crate::ai_sdk_types::catalog::{ProviderDefinition, SdkType};
use crate::ai_sdk_types::v2 as v2t;
use serde_json::Value as JsonValue;
use tracing::info;

const TRACE_PREFIX: &str = "[AZURE]";
const DEFAULT_ENDPOINT_PATH: &str = "/v1/responses";
const DEFAULT_BASE_URL_FMT: &str = "https://{resource}.openai.azure.com/openai";
const DEFAULT_API_VERSION: &str = "v1";
const RESOURCE_ENV: &str = "AZURE_RESOURCE_NAME";
const API_KEY_ENV: &str = "AZURE_API_KEY";
const API_KEY_ENV_FALLBACK: &str = "AZURE_OPENAI_API_KEY";
const TOKEN_ENV: &str = "AZURE_BEARER_TOKEN";
const ENDPOINT_ENV: &str = "AZURE_OPENAI_ENDPOINT";

fn match_azure(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::Azure) || def.name.eq_ignore_ascii_case("azure")
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn default_headers_from_auth(
    api_key: Option<String>,
    bearer: Option<String>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "application/json".to_string()),
    ];
    if let Some(key) = api_key {
        if !key.trim().is_empty() {
            headers.push(("api-key".into(), key));
        }
    }
    if let Some(token) = bearer {
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            let value = if trimmed.to_ascii_lowercase().starts_with("bearer ") {
                trimmed.to_string()
            } else {
                format!("Bearer {}", trimmed)
            };
            headers.push(("authorization".into(), value));
        }
    }
    headers
}

fn filter_headers(
    headers: &HashMap<String, String>,
    provider_scope: &str,
) -> (
    Vec<(String, String)>,
    Option<v2t::ProviderOptions>,
    Option<JsonValue>,
    Option<bool>,
) {
    let mut filtered = Vec::new();
    let mut defaults: Option<v2t::ProviderOptions> = None;
    let mut raw: Option<JsonValue> = None;
    let mut use_deployments: Option<bool> = None;

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
            "content-type" | "accept" | "authorization" | "api-key" | "x-api-key"
        ) {
            continue;
        }
        if key == "x-azure-use-deployment-urls" || key == "azure-use-deployment-urls" {
            if use_deployments.is_none() {
                use_deployments = parse_bool(v);
            }
            continue;
        }
        filtered.push((key, v.clone()));
    }

    (filtered, defaults, raw, use_deployments)
}

fn resolve_api_key(creds: &Credentials) -> Option<String> {
    match creds {
        Credentials::ApiKey(v) => Some(v.clone()),
        Credentials::Bearer(v) => Some(
            v.trim()
                .trim_start_matches("Bearer ")
                .trim_start_matches("bearer ")
                .to_string(),
        ),
        Credentials::None => None,
    }
    .filter(|v| !v.trim().is_empty())
    .or_else(|| std::env::var(API_KEY_ENV).ok())
    .or_else(|| std::env::var(API_KEY_ENV_FALLBACK).ok())
    .filter(|v| !v.trim().is_empty())
}

fn resolve_bearer_token(creds: &Credentials) -> Option<String> {
    match creds {
        Credentials::Bearer(v) => Some(v.clone()),
        _ => std::env::var(TOKEN_ENV).ok(),
    }
    .filter(|v| !v.trim().is_empty())
}

fn resolve_api_version(def: &ProviderDefinition) -> String {
    let mut version = None;
    for (k, v) in def.query_params.iter() {
        if k.eq_ignore_ascii_case("api-version") && !v.trim().is_empty() {
            version = Some(v.clone());
            break;
        }
    }
    version
        .or_else(|| std::env::var("AZURE_API_VERSION").ok())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_API_VERSION.to_string())
}

fn normalize_endpoint_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        DEFAULT_ENDPOINT_PATH.to_string()
    } else if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{}", trimmed)
    }
}

fn strip_v1_prefix(path: &str) -> String {
    let trimmed = path.trim_start_matches('/');
    if let Some(rest) = trimmed.strip_prefix("v1/") {
        format!("/{}", rest)
    } else {
        format!("/{}", trimmed)
    }
}

fn resolve_base_prefix(def: &ProviderDefinition) -> Result<String, SdkError> {
    if !def.base_url.trim().is_empty() {
        return Ok(def.base_url.trim_end_matches('/').to_string());
    }
    if let Ok(endpoint) = std::env::var(ENDPOINT_ENV) {
        let trimmed = endpoint.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.trim_end_matches('/').to_string());
        }
    }

    let mut env_candidates: Vec<String> = Vec::new();
    if let Some(env_name) = def.env.as_ref() {
        env_candidates.push(env_name.clone());
    }
    env_candidates.push(RESOURCE_ENV.to_string());

    for env_name in env_candidates {
        if let Ok(resource) = std::env::var(&env_name) {
            let trimmed = resource.trim();
            if !trimmed.is_empty() {
                return Ok(DEFAULT_BASE_URL_FMT.replace("{resource}", trimmed));
            }
        }
    }

    Err(SdkError::Transport(TransportError::Other(
        "Azure base URL not configured; set base_url, AZURE_OPENAI_ENDPOINT, or AZURE_RESOURCE_NAME"
            .into(),
    )))
}

fn build_azure(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let api_key = resolve_api_key(creds);
    let bearer = resolve_bearer_token(creds);
    let (extra_headers, default_options, request_defaults, use_deployments_from_headers) =
        filter_headers(&def.headers, &def.name);

    let use_deployment_urls = use_deployments_from_headers
        .or_else(|| {
            std::env::var("AZURE_USE_DEPLOYMENT_URLS")
                .ok()
                .and_then(|v| parse_bool(&v))
        })
        .unwrap_or(false);

    let base_prefix = resolve_base_prefix(def)?;
    let mut endpoint_path = normalize_endpoint_path(&def.endpoint_path);
    let mut base_url = base_prefix.trim_end_matches('/').to_string();
    let base_has_deployments = base_url.contains("/deployments/");

    if use_deployment_urls
        && !base_has_deployments
        && !endpoint_path
            .trim_start_matches('/')
            .starts_with("deployments/")
    {
        base_url = format!("{}/deployments/{}", base_url, model);
        endpoint_path = strip_v1_prefix(&endpoint_path);
    }

    let api_version = resolve_api_version(def);
    let mut query_params: Vec<(String, String)> = def
        .query_params
        .iter()
        .filter(|(k, _)| !k.eq_ignore_ascii_case("api-version"))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    query_params.push(("api-version".into(), api_version.clone()));

    let mut headers = default_headers_from_auth(api_key, bearer);
    headers.extend(extra_headers);

    let supported_urls = HashMap::from([
        ("image/*".to_string(), vec![r"^https?://.*$".to_string()]),
        (
            "application/pdf".to_string(),
            vec![r"^https?://.*$".to_string()],
        ),
    ]);

    let mut transport_cfg = TransportConfig::default();
    transport_cfg.idle_read_timeout = Duration::from_secs(45);
    apply_stream_idle_timeout_ms(def, &mut transport_cfg);
    let http = crate::reqwest_transport::ReqwestTransport::new(&transport_cfg);

    let config = OpenAIConfig {
        provider_name: "azure.responses".into(),
        provider_scope_name: def.name.clone(),
        base_url,
        endpoint_path,
        headers,
        query_params,
        supported_urls,
        file_id_prefixes: Some(vec!["assistant-".into()]),
        default_options,
        request_defaults,
    };

    info!(
        "{}: configured Azure scope={} endpoint={} api_version={} deployment_mode={}",
        TRACE_PREFIX,
        config.provider_scope_name,
        config.endpoint_url(),
        api_version,
        use_deployment_urls
    );

    let lm = OpenAIResponsesLanguageModel::new(model.to_string(), config, http, transport_cfg);
    Ok(Arc::new(lm))
}

inventory::submit! {
    ProviderRegistration {
        id: "azure",
        sdk_type: SdkType::Azure,
        matches: Some(match_azure),
        build: build_azure,
        reasoning_scope: None,
    }
}
