use std::collections::HashMap;

use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_types::v2::ProviderOptions;

/// Authentication configuration for Amazon Bedrock requests.
#[derive(Debug, Clone)]
pub enum BedrockAuth {
    /// Use a static bearer token (aka API key) for `Authorization: Bearer ...`.
    ApiKey { token: String },
    /// Use AWS Signature Version 4 signing with the provided credentials.
    SigV4(SigV4Config),
}

/// AWS SigV4 signing material.
#[derive(Debug, Clone)]
pub struct SigV4Config {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub region: String,
}

/// Core configuration shared by the language model implementation.
#[derive(Debug, Clone)]
pub struct BedrockConfig<T: HttpTransport> {
    pub provider_name: &'static str,
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub default_options: Option<ProviderOptions>,
    pub auth: BedrockAuth,
}

impl<T: HttpTransport> BedrockConfig<T> {
    pub fn endpoint_for_model(&self, model_id: &str, suffix: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let encoded_model = urlencoding::encode(model_id);
        format!("{}/model/{}{}", base, encoded_model, suffix)
    }
}
