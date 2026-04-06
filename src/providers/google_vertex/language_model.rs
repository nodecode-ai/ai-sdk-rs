use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::core::transport::{HttpTransport, TransportConfig};
use crate::core::{GenerateResponse, LanguageModel, SdkError, StreamResponse};
use crate::types::v2 as v2t;

use crate::providers::google::shared::error::map_transport_error_to_sdk_error;
use crate::providers::google::shared::generate_response::parse_google_vertex_generate_response;
use crate::providers::google::shared::request_body::{
    build_google_request_body, GoogleRequestBodyBuildConfig,
};
use crate::providers::google::shared::stream_core::build_google_stream_part_stream;

const TRACE_PREFIX: &str = "[GOOGLE-VERTEX]";

pub struct GoogleVertexConfig<T: HttpTransport = crate::transport_reqwest::ReqwestTransport> {
    pub provider_name: &'static str,
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub query_params: Vec<(String, String)>,
    pub default_options: Option<v2t::ProviderOptions>,
}

pub struct GoogleVertexLanguageModel<T: HttpTransport = crate::transport_reqwest::ReqwestTransport>
{
    pub model_id: String,
    pub cfg: GoogleVertexConfig<T>,
}

impl<T: HttpTransport> GoogleVertexLanguageModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: GoogleVertexConfig<T>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg,
        }
    }

    fn model_path(&self) -> String {
        let id = &self.model_id;
        if id.contains('/') {
            id.clone()
        } else {
            format!("models/{}", id)
        }
    }

    fn url_generate(&self) -> String {
        let mut url = format!(
            "{}/{}:generateContent",
            self.cfg.base_url.trim_end_matches('/'),
            self.model_path()
        );
        if !self.cfg.query_params.is_empty() {
            let qp: Vec<String> = self
                .cfg
                .query_params
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect();
            url.push('?');
            url.push_str(&qp.join("&"));
        }
        url
    }

    fn url_stream(&self) -> String {
        let base = format!(
            "{}/{}:streamGenerateContent?alt=sse",
            self.cfg.base_url.trim_end_matches('/'),
            self.model_path()
        );
        if self.cfg.query_params.is_empty() {
            base
        } else {
            let qp: Vec<String> = self
                .cfg
                .query_params
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect();
            format!("{}&{}", base, qp.join("&"))
        }
    }

    fn is_gemma(&self) -> bool {
        self.model_id.to_ascii_lowercase().starts_with("gemma-")
    }

    fn build_body(
        &self,
        options: &v2t::CallOptions,
    ) -> Result<(JsonValue, Vec<v2t::CallWarning>), SdkError> {
        build_google_request_body(
            GoogleRequestBodyBuildConfig {
                scope_names: &["google-vertex", "google"],
                raw_provider_option_keys: &["google-vertex", "google"],
                model_id: &self.model_id,
                is_gemma: self.is_gemma(),
                trace_prefix: TRACE_PREFIX,
                include_thoughts_warning: None,
            },
            options,
        )
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync + 'static> LanguageModel for GoogleVertexLanguageModel<T> {
    fn provider_name(&self) -> &'static str {
        self.cfg.provider_name
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.cfg.supported_urls.clone()
    }

    async fn do_generate(&self, options: v2t::CallOptions) -> Result<GenerateResponse, SdkError> {
        let options = crate::core::request_builder::defaults::build_call_options(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );
        let (body, warnings) = self.build_body(&options)?;
        let url = self.url_generate();

        let headers: Vec<(String, String)> = self
            .cfg
            .headers
            .iter()
            .filter(|(k, _)| !crate::core::options::is_internal_sdk_header(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let (resp_json, resp_headers) = match self
            .cfg
            .http
            .post_json(&url, &headers, &body, &self.cfg.transport_cfg)
            .await
        {
            Ok(ok) => ok,
            Err(te) => {
                return Err(map_transport_error_to_sdk_error(te));
            }
        };

        let parsed = parse_google_vertex_generate_response(&resp_json);

        Ok(GenerateResponse {
            content: parsed.content,
            finish_reason: parsed.finish_reason,
            usage: parsed.usage,
            provider_metadata: parsed.provider_metadata,
            request_body: Some(body),
            response_headers: Some(resp_headers.into_iter().collect()),
            response_body: Some(resp_json),
            warnings,
        })
    }

    async fn do_stream(&self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let options = crate::core::request_builder::defaults::build_call_options(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );
        let (body, warnings) = self.build_body(&options)?;
        let url = self.url_stream();
        let headers: Vec<(String, String)> = self
            .cfg
            .headers
            .iter()
            .filter(|(k, _)| !crate::core::options::is_internal_sdk_header(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let resp = match self
            .cfg
            .http
            .post_json_stream(&url, &headers, &body, &self.cfg.transport_cfg)
            .await
        {
            Ok(r) => r,
            Err(te) => {
                return Err(map_transport_error_to_sdk_error(te));
            }
        };
        let (inner, resp_headers) = <T as HttpTransport>::into_stream(resp);

        let include_raw = options.include_raw_chunks;
        let stream = build_google_stream_part_stream(inner, warnings, include_raw, "google-vertex");

        Ok(StreamResponse {
            stream,
            request_body: Some(body),
            response_headers: Some(resp_headers.into_iter().collect()),
        })
    }
}
