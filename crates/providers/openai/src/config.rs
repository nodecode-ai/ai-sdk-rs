use std::collections::HashMap;

use crate::ai_sdk_types::v2 as v2t;
use serde_json::Value as JsonValue;

/// OpenAI provider configuration used by the Responses language model.
#[derive(Clone, Debug)]
pub struct OpenAIConfig {
    pub provider_name: String,
    pub provider_scope_name: String,
    pub base_url: String,
    /// Path after base_url (default: "/v1/responses").
    pub endpoint_path: String,
    /// Static headers to include for every request (lower-case keys preferred).
    pub headers: Vec<(String, String)>,
    /// Query params appended to the request URL.
    pub query_params: Vec<(String, String)>,
    /// Supported URL regex patterns by media type.
    pub supported_urls: HashMap<String, Vec<String>>,
    /// File ID prefixes used to identify file IDs in Responses API.
    /// When undefined, all file data is treated as base64 content.
    pub file_id_prefixes: Option<Vec<String>>,
    /// Provider-scoped default options parsed from configuration headers.
    pub default_options: Option<v2t::ProviderOptions>,
    /// Raw request-level overrides parsed from configuration headers (pre-merged).
    pub request_defaults: Option<JsonValue>,
}

impl OpenAIConfig {
    pub fn endpoint_url(&self) -> String {
        let base_trimmed = self.base_url.trim_end_matches('/');
        let mut ep = self.endpoint_path.trim_start_matches('/');
        // Guard against double "/v1" if caller supplied base_url ending with /v1
        if base_trimmed.ends_with("/v1") && ep.starts_with("v1/") {
            ep = &ep[3..]; // drop leading "v1/"
        }
        let mut url = format!("{}/{}", base_trimmed, ep);
        if !self.query_params.is_empty() {
            let qp = self
                .query_params
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            url.push('?');
            url.push_str(&qp);
        }
        url
    }
}
