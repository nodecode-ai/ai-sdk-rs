use std::collections::HashMap;

use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_types::v2::ProviderOptions;

#[derive(Clone, Debug)]
pub struct GatewayConfig {
    pub provider_name: &'static str,
    pub provider_scope_name: String,
    pub base_url: String,
    pub endpoint_path: Option<String>,
    pub headers: Vec<(String, String)>,
    pub query_params: Vec<(String, String)>,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub transport_cfg: TransportConfig,
    pub default_options: Option<ProviderOptions>,
    pub request_defaults: Option<serde_json::Value>,
    pub auth: Option<GatewayAuth>,
}

impl GatewayConfig {
    pub fn language_endpoint(&self) -> String {
        match self.endpoint_path.as_deref() {
            Some(path) if !path.is_empty() => {
                format!("{}{}", self.base_url.trim_end_matches('/'), path)
            }
            _ => self.base_url.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GatewayAuth {
    pub token: String,
    pub method: GatewayAuthMethod,
}

#[derive(Clone, Debug)]
pub enum GatewayAuthMethod {
    ApiKey,
    Oidc,
}

impl GatewayAuthMethod {
    pub fn as_header_value(&self) -> &'static str {
        match self {
            GatewayAuthMethod::ApiKey => "api-key",
            GatewayAuthMethod::Oidc => "oidc",
        }
    }
}
