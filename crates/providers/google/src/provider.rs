use std::collections::HashMap;
use std::sync::Arc;

use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_provider::{
    build_provider_transport_config, collect_query_params, filter_provider_bootstrap_headers,
    registry::ProviderRegistration, Credentials,
};
use crate::ai_sdk_types::catalog::{ProviderDefinition, SdkType};

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
    let bootstrap_headers = filter_provider_bootstrap_headers(
        &def.headers,
        &def.name,
        &[
            "content-type",
            "accept",
            "authorization",
            "x-api-key",
            "x-goog-api-key",
        ],
    );
    headers.extend(bootstrap_headers.headers);

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

    let transport_cfg =
        build_provider_transport_config(def, Some(std::time::Duration::from_secs(45)));

    let http = crate::reqwest_transport::ReqwestTransport::try_new(&transport_cfg)
        .map_err(SdkError::Transport)?;

    let cfg = GoogleGenAiConfig {
        provider_name: "google.gen-ai",
        provider_scope_name: def.name.clone(),
        base_url,
        headers,
        http,
        transport_cfg,
        supported_urls,
        query_params: collect_query_params(def),
        default_options: bootstrap_headers.default_options,
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
