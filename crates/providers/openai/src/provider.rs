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

use crate::provider_openai::config::OpenAIConfig;
use crate::provider_openai::responses::language_model::OpenAIResponsesLanguageModel;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_ENDPOINT_PATH: &str = "/responses";

fn default_headers_from_creds(
    api_key: Option<String>,
    bearer: Option<String>,
) -> Vec<(String, String)> {
    let mut h = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "application/json".to_string()),
    ];
    if let Some(b) = bearer {
        let v = if b.to_lowercase().starts_with("bearer ") {
            b
        } else {
            format!("Bearer {}", b)
        };
        h.push(("authorization".into(), v));
    } else if let Some(k) = api_key {
        h.push(("authorization".into(), format!("Bearer {}", k)));
    }
    h
}

fn filter_headers(
    headers: &std::collections::HashMap<String, String>,
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
                    raw = Some(json.clone());
                    if defaults.is_none() {
                        defaults = provider_defaults_from_json(provider_scope, &json);
                    }
                }
            }
            continue;
        }
        let kl = k.to_ascii_lowercase();
        if kl == "content-type" || kl == "accept" || kl == "authorization" || kl == "x-api-key" {
            continue;
        }
        filtered.push((kl, v.clone()));
    }
    (filtered, defaults, raw)
}

fn match_openai(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAI)
}

fn build_openai(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    // Resolve credentials
    let api_key = creds
        .as_api_key()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());
    let bearer = creds.as_bearer();

    // Core config
    let base_url = if def.base_url.trim().is_empty() {
        DEFAULT_BASE_URL.to_string()
    } else {
        def.base_url.clone()
    };
    let endpoint_path = if def.endpoint_path.trim().is_empty() {
        DEFAULT_ENDPOINT_PATH.to_string()
    } else {
        def.endpoint_path.clone()
    };

    let mut headers = default_headers_from_creds(api_key, bearer);
    let (extra_headers, default_options, request_defaults) =
        filter_headers(&def.headers, &def.name);
    headers.extend(extra_headers);

    let config = OpenAIConfig {
        provider_name: "openai.responses".into(),
        provider_scope_name: def.name.clone(),
        base_url,
        endpoint_path,
        headers,
        query_params: def
            .query_params
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        supported_urls: HashMap::from([
            ("image/*".to_string(), vec![r"^https?://.*$".to_string()]),
            (
                "application/pdf".to_string(),
                vec![r"^https?://.*$".to_string()],
            ),
        ]),
        file_id_prefixes: Some(vec!["file-".into()]),
        default_options,
        request_defaults,
    };

    let mut transport_cfg = TransportConfig::default();
    // Keep generous idle read timeout for SSE streams
    // Align with previous implementation
    transport_cfg.idle_read_timeout = std::time::Duration::from_secs(45);
    apply_stream_idle_timeout_ms(def, &mut transport_cfg);

    let http = crate::reqwest_transport::ReqwestTransport::try_new(&transport_cfg)
        .map_err(SdkError::Transport)?;

    let lm = OpenAIResponsesLanguageModel::new(model.to_string(), config, http, transport_cfg);
    Ok(Arc::new(lm))
}

inventory::submit! {
    ProviderRegistration {
        id: "openai",
        sdk_type: SdkType::OpenAI,
        matches: Some(match_openai),
        build: build_openai,
        reasoning_scope: None,
    }
}
