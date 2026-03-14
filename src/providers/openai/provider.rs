use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::core::transport::TransportConfig;
use crate::core::{LanguageModel, SdkError};
use crate::provider::{
    build_provider_transport_config, collect_query_params, filter_provider_bootstrap_headers,
    registry::ProviderRegistration, Credentials,
};
use crate::types::catalog::{ProviderDefinition, SdkType};
use crate::types::v2 as v2t;
use serde_json::Value as JsonValue;

use crate::providers::openai::config::OpenAIConfig;
use crate::providers::openai::responses::language_model::OpenAIResponsesLanguageModel;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_ENDPOINT_PATH: &str = "/responses";

fn default_transport_config() -> TransportConfig {
    let mut cfg = TransportConfig::default();
    cfg.idle_read_timeout = Duration::from_secs(45);
    cfg
}

fn supported_urls() -> HashMap<String, Vec<String>> {
    HashMap::from([
        ("image/*".to_string(), vec![r"^https?://.*$".to_string()]),
        (
            "application/pdf".to_string(),
            vec![r"^https?://.*$".to_string()],
        ),
    ])
}

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

#[derive(Clone, Debug)]
pub struct OpenAIResponsesBuilder {
    model_id: String,
    provider_scope_name: String,
    base_url: String,
    endpoint_path: String,
    api_key: Option<String>,
    bearer: Option<String>,
    headers: Vec<(String, String)>,
    query_params: Vec<(String, String)>,
    default_options: Option<v2t::ProviderOptions>,
    request_defaults: Option<JsonValue>,
    transport_cfg: TransportConfig,
}

impl OpenAIResponsesBuilder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            provider_scope_name: "openai".into(),
            base_url: DEFAULT_BASE_URL.into(),
            endpoint_path: DEFAULT_ENDPOINT_PATH.into(),
            api_key: None,
            bearer: None,
            headers: Vec::new(),
            query_params: Vec::new(),
            default_options: None,
            request_defaults: None,
            transport_cfg: default_transport_config(),
        }
    }

    pub fn with_provider_scope_name(mut self, provider_scope_name: impl Into<String>) -> Self {
        self.provider_scope_name = provider_scope_name.into();
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_endpoint_path(mut self, endpoint_path: impl Into<String>) -> Self {
        self.endpoint_path = endpoint_path.into();
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_bearer(mut self, bearer: impl Into<String>) -> Self {
        self.bearer = Some(bearer.into());
        self
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    pub fn with_headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers.extend(
            headers
                .into_iter()
                .map(|(key, value)| (key.into(), value.into())),
        );
        self
    }

    pub fn with_query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.push((key.into(), value.into()));
        self
    }

    pub fn with_query_params<I, K, V>(mut self, query_params: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.query_params.extend(
            query_params
                .into_iter()
                .map(|(key, value)| (key.into(), value.into())),
        );
        self
    }

    pub fn with_default_options(mut self, default_options: v2t::ProviderOptions) -> Self {
        self.default_options = Some(default_options);
        self
    }

    pub fn with_request_defaults(mut self, request_defaults: JsonValue) -> Self {
        self.request_defaults = Some(request_defaults);
        self
    }

    pub fn with_transport_config(mut self, transport_cfg: TransportConfig) -> Self {
        self.transport_cfg = transport_cfg;
        self
    }

    pub fn build(
        self,
    ) -> Result<OpenAIResponsesLanguageModel<crate::transport_reqwest::ReqwestTransport>, SdkError>
    {
        let mut headers = default_headers_from_creds(self.api_key, self.bearer);
        headers.extend(self.headers);

        let config = OpenAIConfig {
            provider_name: "openai.responses".into(),
            provider_scope_name: self.provider_scope_name,
            base_url: self.base_url,
            endpoint_path: self.endpoint_path,
            headers,
            query_params: self.query_params,
            supported_urls: supported_urls(),
            file_id_prefixes: Some(vec!["file-".into()]),
            default_options: self.default_options,
            request_defaults: self.request_defaults,
        };

        let http = crate::transport_reqwest::ReqwestTransport::try_new(&self.transport_cfg)
            .map_err(SdkError::Transport)?;
        let mut lm =
            OpenAIResponsesLanguageModel::new(self.model_id, config, http, self.transport_cfg);
        lm.start_codex_websocket_preconnect();
        Ok(lm)
    }
}

fn match_openai(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAI)
}

fn build_openai(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let api_key = creds
        .as_api_key()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());
    let bearer = creds.as_bearer();

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

    let bootstrap_headers = filter_provider_bootstrap_headers(
        &def.headers,
        &def.name,
        &["content-type", "accept", "authorization", "x-api-key"],
    );
    let extra_headers = bootstrap_headers.headers;
    let default_options = bootstrap_headers.default_options;
    let request_defaults = bootstrap_headers.request_defaults;

    let transport_cfg = build_provider_transport_config(def, Some(Duration::from_secs(45)));

    let mut builder = OpenAIResponsesBuilder::new(model)
        .with_provider_scope_name(def.name.clone())
        .with_base_url(base_url)
        .with_endpoint_path(endpoint_path)
        .with_headers(extra_headers)
        .with_query_params(collect_query_params(def))
        .with_transport_config(transport_cfg);
    if let Some(default_options) = default_options {
        builder = builder.with_default_options(default_options);
    }
    if let Some(request_defaults) = request_defaults {
        builder = builder.with_request_defaults(request_defaults);
    }
    if let Some(bearer) = bearer {
        builder = builder.with_bearer(bearer);
    } else if let Some(api_key) = api_key {
        builder = builder.with_api_key(api_key);
    }
    Ok(Arc::new(builder.build()?))
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
