use std::collections::{BTreeMap, HashMap};

use crate::ai_sdk_core::embedding::{EmbedResponse, EmbeddingModel};
use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_core::options::is_internal_sdk_header;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_types::embedding::{EmbedOptions, EmbedUsage};
use crate::ai_sdk_types::v2 as v2t;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

use crate::provider_openai_compatible::embedding::options::{
    apply_provider_defaults, parse_openai_compatible_embedding_provider_options,
    OpenAICompatibleEmbeddingProviderOptions,
};
use crate::provider_openai_compatible::error::map_transport_error_to_sdk_error;

pub(crate) const DEFAULT_MAX_EMBEDDINGS_PER_CALL: usize = 2048;

#[derive(Clone)]
pub struct OpenAICompatibleEmbeddingConfig<T: HttpTransport> {
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub query_params: Vec<(String, String)>,
    pub max_embeddings_per_call: Option<usize>,
    pub supports_parallel_calls: bool,
    pub default_options: Option<v2t::ProviderOptions>,
}

pub struct OpenAICompatibleEmbeddingModel<
    T: HttpTransport = crate::reqwest_transport::ReqwestTransport,
> {
    model_id: String,
    cfg: OpenAICompatibleEmbeddingConfig<T>,
}

impl<T: HttpTransport> OpenAICompatibleEmbeddingModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: OpenAICompatibleEmbeddingConfig<T>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg,
        }
    }

    fn build_request_url(&self) -> String {
        let base = self.cfg.base_url.trim_end_matches('/');
        let mut url = format!("{}/embeddings", base);
        if !self.cfg.query_params.is_empty() {
            let qp = self
                .cfg
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

    fn canonicalize_header(lc: &str) -> String {
        lc.split('-')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    None => String::new(),
                    Some(f) => {
                        f.to_ascii_uppercase().to_string() + &chars.as_str().to_ascii_lowercase()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    fn build_headers(&self, extra: &HashMap<String, String>) -> Vec<(String, String)> {
        let mut hdrs: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in &self.cfg.headers {
            if is_internal_sdk_header(k) {
                continue;
            }
            hdrs.insert(k.to_ascii_lowercase(), v.clone());
        }
        for (k, v) in extra {
            if is_internal_sdk_header(k) {
                continue;
            }
            hdrs.insert(k.to_ascii_lowercase(), v.clone());
        }
        hdrs.entry("content-type".into())
            .or_insert_with(|| "application/json".into());
        hdrs.entry("accept".into())
            .or_insert_with(|| "application/json".into());
        hdrs.into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect()
    }

    fn build_request_body(&self, options: &EmbedOptions) -> Result<JsonValue, SdkError> {
        // Provider options and extras
        let scope_names = ["openai-compatible", self.cfg.provider_scope_name.as_str()];
        let (prov_opts, prov_extras) = parse_openai_compatible_embedding_provider_options(
            &options.provider_options,
            &scope_names,
        );

        // Send null for model if empty (some APIs require null rather than empty string)
        let model_value: JsonValue = if self.model_id.is_empty() {
            JsonValue::Null
        } else {
            JsonValue::String(self.model_id.clone())
        };

        let OpenAICompatibleEmbeddingProviderOptions { dimensions, user } = prov_opts;

        let mut body_map = serde_json::Map::new();
        body_map.insert("model".into(), model_value);
        body_map.insert("input".into(), json!(options.values));
        body_map.insert("encoding_format".into(), JsonValue::String("float".into()));
        if let Some(dimensions) = dimensions {
            body_map.insert("dimensions".into(), json!(dimensions));
        }
        if let Some(user) = user {
            body_map.insert("user".into(), json!(user));
        }

        if let Some(extras) = prov_extras {
            for (k, v) in extras {
                body_map.insert(k, v);
            }
        }

        Ok(JsonValue::Object(body_map))
    }

    fn too_many_values_error(&self, limit: usize, actual: usize) -> SdkError {
        SdkError::Upstream {
            status: 400,
            message: format!(
                "too many embedding values: {} (max {} per call)",
                actual, limit
            ),
            source: None,
        }
    }
}

#[async_trait::async_trait]
impl<T: HttpTransport + Send + Sync> EmbeddingModel for OpenAICompatibleEmbeddingModel<T> {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn max_embeddings_per_call(&self) -> Option<usize> {
        self.cfg.max_embeddings_per_call
    }

    fn supports_parallel_calls(&self) -> bool {
        self.cfg.supports_parallel_calls
    }

    async fn do_embed(&self, options: EmbedOptions) -> Result<EmbedResponse, SdkError> {
        let options = apply_provider_defaults(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );

        if let Some(limit) = self.cfg.max_embeddings_per_call {
            if options.values.len() > limit {
                return Err(self.too_many_values_error(limit, options.values.len()));
            }
        }

        let body = self.build_request_body(&options)?;
        let headers = self.build_headers(&options.headers);
        let url = self.build_request_url();

        let (json, res_headers) = self
            .cfg
            .http
            .post_json(&url, &headers, &body, &self.cfg.transport_cfg)
            .await
            .map_err(map_transport_error_to_sdk_error)?;

        let parsed: EmbeddingResponse =
            serde_json::from_value(json.clone()).map_err(|se| SdkError::Serde(se))?;

        let embeddings = parsed
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect::<Vec<_>>();

        let usage = parsed.usage.map(|u| EmbedUsage {
            tokens: u.prompt_tokens,
        });

        let response_headers: Option<v2t::Headers> = if res_headers.is_empty() {
            None
        } else {
            Some(
                res_headers
                    .into_iter()
                    .map(|(k, v)| (k.to_ascii_lowercase(), v))
                    .collect(),
            )
        };

        Ok(EmbedResponse {
            embeddings,
            usage,
            provider_metadata: parsed.provider_metadata,
            response_headers,
            response_body: Some(json),
            request_body: Some(body),
        })
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    usage: Option<EmbeddingUsage>,
    #[serde(default, rename = "providerMetadata")]
    provider_metadata: Option<v2t::ProviderMetadata>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
}
