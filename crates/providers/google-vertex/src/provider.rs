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
use crate::provider_google_vertex::language_model::{
    GoogleVertexConfig, GoogleVertexLanguageModel,
};
use serde_json::Value as JsonValue;

const DEFAULT_API_VERSION: &str = "v1beta1";

fn default_headers(bearer: Option<String>, api_key: Option<String>) -> Vec<(String, String)> {
    let mut h = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "application/json".to_string()),
    ];
    if let Some(token) = bearer {
        let v = if token.to_lowercase().starts_with("bearer ") {
            token
        } else {
            format!("Bearer {}", token)
        };
        h.push(("authorization".into(), v));
    } else if let Some(key) = api_key {
        if !key.is_empty() {
            h.push(("x-goog-api-key".into(), key));
        }
    }
    h
}

fn filter_headers(
    headers: &HashMap<String, String>,
    provider_scope: &str,
) -> (Vec<(String, String)>, Option<v2t::ProviderOptions>) {
    let mut filtered: Vec<(String, String)> = Vec::new();
    let mut defaults: Option<v2t::ProviderOptions> = None;
    for (k, v) in headers {
        if sdkopt::is_internal_sdk_header(k) {
            if defaults.is_none() {
                if let Ok(json) = serde_json::from_str::<JsonValue>(v) {
                    if let Some(parsed) = provider_defaults_from_json(provider_scope, &json) {
                        defaults = Some(parsed);
                    }
                }
            }
            continue;
        }
        let kl = k.to_ascii_lowercase();
        if kl == "content-type"
            || kl == "accept"
            || kl == "authorization"
            || kl == "x-api-key"
            || kl == "x-goog-api-key"
        {
            continue;
        }
        filtered.push((kl, v.clone()));
    }
    (filtered, defaults)
}

fn resolve_base_url(def: &ProviderDefinition) -> Result<String, SdkError> {
    let configured = def.base_url.trim();
    if !configured.is_empty() {
        return Ok(configured.trim_end_matches('/').to_string());
    }

    let project = std::env::var("GOOGLE_VERTEX_PROJECT").unwrap_or_default();
    let location = std::env::var("GOOGLE_VERTEX_LOCATION").unwrap_or_default();
    let project = project.trim();
    let location = location.trim();

    if project.is_empty() || location.is_empty() {
        return Err(SdkError::Upstream {
            status: 400,
            message: "Missing Google Vertex configuration: set provider base_url or GOOGLE_VERTEX_PROJECT and GOOGLE_VERTEX_LOCATION".into(),
            source: None,
        });
    }

    let host = if location.eq_ignore_ascii_case("global") {
        "aiplatform.googleapis.com".to_string()
    } else {
        format!("{}-aiplatform.googleapis.com", location)
    };

    Ok(format!(
        "https://{}/{}/projects/{}/locations/{}/publishers/google",
        host, DEFAULT_API_VERSION, project, location
    ))
}

fn match_google_vertex(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::GoogleVertex)
}

fn build_google_vertex(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let bearer = creds
        .as_bearer()
        .or_else(|| std::env::var("GOOGLE_VERTEX_ACCESS_TOKEN").ok())
        .or_else(|| std::env::var("GOOGLE_CLOUD_ACCESS_TOKEN").ok());
    let api_key = creds
        .as_api_key()
        .or_else(|| std::env::var("GOOGLE_VERTEX_API_KEY").ok());

    let base_url = resolve_base_url(def)?;
    let mut headers = default_headers(bearer, api_key);
    let (extra_headers, default_options) = filter_headers(&def.headers, &def.name);
    headers.extend(extra_headers);

    let supported_urls = HashMap::from([(
        "*".to_string(),
        vec![String::from(r"^https?://.*$"), String::from(r"^gs://.*$")],
    )]);

    let mut transport_cfg = TransportConfig::default();
    transport_cfg.idle_read_timeout = Duration::from_secs(45);
    apply_stream_idle_timeout_ms(def, &mut transport_cfg);

    let http = crate::reqwest_transport::ReqwestTransport::new(&transport_cfg);

    let cfg = GoogleVertexConfig {
        provider_name: "google.vertex",
        provider_scope_name: def.name.clone(),
        base_url,
        headers,
        http,
        transport_cfg,
        supported_urls,
        query_params: def
            .query_params
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        default_options,
    };

    let lm = GoogleVertexLanguageModel::new(model.to_string(), cfg);
    Ok(Arc::new(lm))
}

inventory::submit! {
    ProviderRegistration {
        id: "google-vertex",
        sdk_type: SdkType::GoogleVertex,
        matches: Some(match_google_vertex),
        build: build_google_vertex,
        reasoning_scope: None,
    }
}
