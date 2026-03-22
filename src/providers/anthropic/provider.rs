use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::transport::TransportConfig;
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
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

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

#[derive(Clone, Debug)]
pub struct AnthropicMessagesBuilder {
    model_id: String,
    provider_scope_name: String,
    base_url: String,
    api_key: Option<String>,
    bearer: Option<String>,
    headers: Vec<(String, String)>,
    default_options: Option<v2t::ProviderOptions>,
    transport_cfg: TransportConfig,
}

impl AnthropicMessagesBuilder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            provider_scope_name: "anthropic".into(),
            base_url: DEFAULT_BASE_URL.into(),
            api_key: None,
            bearer: None,
            headers: Vec::new(),
            default_options: None,
            transport_cfg: {
                let mut cfg = TransportConfig::default();
                cfg.idle_read_timeout = DEFAULT_IDLE_READ_TIMEOUT;
                cfg
            },
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

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_bearer(mut self, bearer: impl Into<String>) -> Self {
        self.bearer = Some(bearer.into());
        self
    }

    pub fn with_oauth_bearer(self, bearer: impl Into<String>) -> Self {
        self.with_bearer(bearer)
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

    pub fn with_default_options(mut self, default_options: v2t::ProviderOptions) -> Self {
        self.default_options = Some(default_options);
        self
    }

    pub fn with_transport_config(mut self, transport_cfg: TransportConfig) -> Self {
        self.transport_cfg = transport_cfg;
        self
    }

    pub fn build(
        self,
    ) -> Result<AnthropicMessagesLanguageModel<crate::reqwest_transport::ReqwestTransport>, SdkError>
    {
        let mut header_map: BTreeMap<String, (String, String)> = BTreeMap::new();
        for (key, value) in default_headers_from_creds(self.api_key, self.bearer) {
            header_map.insert(key.to_ascii_lowercase(), (key, value));
        }
        for (key, value) in self.headers {
            header_map.insert(key.to_ascii_lowercase(), (key, value));
        }

        let mut headers = Vec::new();
        for (_lc, (orig, value)) in header_map.into_iter() {
            if sdkopt::is_internal_sdk_header(&orig) {
                continue;
            }
            headers.push((orig, value));
        }

        let http = crate::reqwest_transport::ReqwestTransport::try_new(&self.transport_cfg)
            .map_err(SdkError::Transport)?;
        let supported_urls =
            HashMap::from([("image/*".to_string(), vec![r"^https?://.*$".to_string()])]);

        Ok(AnthropicMessagesLanguageModel::new(
            self.model_id,
            AnthropicMessagesConfig {
                provider_name: "anthropic.messages",
                provider_scope_name: self.provider_scope_name,
                base_url: self.base_url,
                headers,
                http,
                transport_cfg: self.transport_cfg,
                supported_urls,
                default_options: self.default_options,
            },
        ))
    }
}

fn match_anthropic(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::Anthropic)
}

fn build_anthropic(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let api_key = creds
        .as_api_key()
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());
    let bearer = creds.as_bearer();

    let base_url = if def.base_url.trim().is_empty() {
        DEFAULT_BASE_URL.to_string()
    } else {
        def.base_url.clone()
    };

    let mut header_map: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (k, v) in default_headers_from_creds(api_key.clone(), bearer.clone()) {
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
    let mut builder = AnthropicMessagesBuilder::new(model)
        .with_provider_scope_name(def.name.clone())
        .with_base_url(base_url)
        .with_headers(headers)
        .with_transport_config(transport_cfg);
    if let Some(default_options) = default_options {
        builder = builder.with_default_options(default_options);
    }
    if let Some(bearer) = bearer {
        builder = builder.with_bearer(bearer);
    } else if let Some(api_key) = api_key {
        builder = builder.with_api_key(api_key);
    }

    Ok(Arc::new(builder.build()?))
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
