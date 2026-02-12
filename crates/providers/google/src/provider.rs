use std::collections::HashMap;
use std::sync::Arc;

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

use crate::provider_google::gen_ai::language_model::{GoogleGenAiConfig, GoogleGenAiLanguageModel};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

fn default_headers_from_api_key(api_key: Option<String>) -> Vec<(String, String)> {
    let mut h = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "application/json".to_string()),
    ];
    if let Some(key) = api_key {
        if !key.is_empty() {
            h.push(("x-goog-api-key".into(), key));
        }
    }
    h
}

fn filter_headers(
    headers: &std::collections::HashMap<String, String>,
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

fn match_google(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::Google)
}

fn build_google(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    // Resolve API key from credentials or env
    let api_key = creds
        .as_api_key()
        .or_else(|| std::env::var("GOOGLE_GENERATIVE_AI_API_KEY").ok());

    let base_url = if def.base_url.trim().is_empty() {
        DEFAULT_BASE_URL.to_string()
    } else {
        def.base_url.clone()
    };
    let mut headers = default_headers_from_api_key(api_key);
    let (extra_headers, default_options) = filter_headers(&def.headers, &def.name);
    headers.extend(extra_headers);

    let supported_urls = HashMap::from([(
        "*".to_string(),
        vec![
            // files endpoint under base_url
            format!(r"^{}{}$", regex::escape(&base_url), "/files/.*"),
            // YouTube URLs
            String::from(r"^https://(?:www\.)?youtube\.com/watch\?v=[\w-]+(?:&[\w=&.-]*)?$"),
            String::from(r"^https://youtu\.be/[\w-]+(?:\?[\w=&.-]*)?$"),
        ],
    )]);

    let mut transport_cfg = TransportConfig::default();
    transport_cfg.idle_read_timeout = std::time::Duration::from_secs(45);
    apply_stream_idle_timeout_ms(def, &mut transport_cfg);

    let http = crate::reqwest_transport::ReqwestTransport::new(&transport_cfg);

    let cfg = GoogleGenAiConfig {
        provider_name: "google.gen-ai",
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
        warn_on_include_thoughts: true,
    };

    let lm = GoogleGenAiLanguageModel::new(model.to_string(), cfg);
    Ok(Arc::new(lm))
}

inventory::submit! {
    ProviderRegistration {
        id: "google",
        sdk_type: SdkType::Google,
        matches: Some(match_google),
        build: build_google,
        reasoning_scope: None,
    }
}
