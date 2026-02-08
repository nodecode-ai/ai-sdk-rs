use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{json, Value as JsonValue};

use crate::ai_sdk_core::stream_collect::{collect_stream_to_response, StreamCollectorConfig};
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_types::v2 as v2t;

use crate::provider_openai_compatible::completion::convert::convert_to_openai_compatible_completion_prompt;
use crate::provider_openai_compatible::completion::options::{
    parse_openai_compatible_completion_provider_options, OpenAICompatibleCompletionProviderOptions,
};

pub struct OpenAICompatibleCompletionConfig<T: HttpTransport> {
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub include_usage: bool,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub query_params: Vec<(String, String)>,
    pub default_options: Option<v2t::ProviderOptions>,
}

pub struct OpenAICompatibleCompletionLanguageModel<
    T: HttpTransport = crate::reqwest_transport::ReqwestTransport,
> {
    model_id: String,
    cfg: OpenAICompatibleCompletionConfig<T>,
}

impl<T: HttpTransport> OpenAICompatibleCompletionLanguageModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: OpenAICompatibleCompletionConfig<T>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg,
        }
    }

    fn build_request_url(&self) -> String {
        let base = self.cfg.base_url.trim_end_matches('/');
        let mut url = format!("{}/completions", base);
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

    fn build_request_body(
        &self,
        options: &v2t::CallOptions,
    ) -> Result<(JsonValue, Vec<v2t::CallWarning>), SdkError> {
        let mut warnings: Vec<v2t::CallWarning> = vec![];

        // Unsupported knobs for completion
        if options.top_k.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "topK".into(),
                details: None,
            });
        }
        if !options.tools.is_empty() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "tools".into(),
                details: None,
            });
        }
        if options.tool_choice.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "toolChoice".into(),
                details: None,
            });
        }
        if matches!(
            options.response_format,
            Some(v2t::ResponseFormat::Json { .. })
        ) {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "responseFormat".into(),
                details: Some("JSON response format is not supported.".into()),
            });
        }

        // Provider options (scoped by provider name)
        let scope_names = ["openai-compatible", self.cfg.provider_scope_name.as_str()];
        let (prov_opts, prov_extras) = parse_openai_compatible_completion_provider_options(
            &options.provider_options,
            &scope_names,
        );
        tracing::info!(
            "[PROVOPTS]: completion scopes={:?} has_suffix={:?}",
            scope_names,
            prov_opts.suffix
        );
        tracing::info!(
            "[PROVOPTS]: completion extras_keys={:?}",
            prov_extras
                .as_ref()
                .map(|m| m.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        );

        // Convert prompt
        let (completion_prompt, mut stop_from_convert) =
            convert_to_openai_compatible_completion_prompt(&options.prompt, "user", "assistant")?;

        // Combine stop sequences
        if let Some(user_stops) = &options.stop_sequences {
            if stop_from_convert.is_none() {
                stop_from_convert = Some(vec![]);
            }
            if let Some(stops) = stop_from_convert.as_mut() {
                for s in user_stops {
                    stops.push(s.clone());
                }
            }
        }

        // Base args
        let OpenAICompatibleCompletionProviderOptions {
            echo,
            logit_bias,
            suffix,
            user,
        } = prov_opts;

        let mut body_map = serde_json::Map::new();
        body_map.insert("model".into(), json!(self.model_id));
        if let Some(echo) = echo {
            body_map.insert("echo".into(), json!(echo));
        }
        if let Some(logit_bias) = logit_bias {
            body_map.insert("logit_bias".into(), json!(logit_bias));
        }
        if let Some(suffix) = suffix {
            body_map.insert("suffix".into(), json!(suffix));
        }
        if let Some(user) = user {
            body_map.insert("user".into(), json!(user));
        }
        if let Some(mt) = options.max_output_tokens {
            body_map.insert("max_tokens".into(), json!(mt));
        }
        if let Some(t) = options.temperature {
            body_map.insert("temperature".into(), json!(t));
        }
        if let Some(tp) = options.top_p {
            body_map.insert("top_p".into(), json!(tp));
        }
        if let Some(fp) = options.frequency_penalty {
            body_map.insert("frequency_penalty".into(), json!(fp));
        }
        if let Some(pp) = options.presence_penalty {
            body_map.insert("presence_penalty".into(), json!(pp));
        }
        if let Some(seed) = options.seed {
            body_map.insert("seed".into(), json!(seed));
        }

        // Merge provider extras (shallow)
        if let Some(extras) = prov_extras {
            for (k, v) in extras {
                tracing::info!("[PROVOPTS]: completion extra {}", k);
                body_map.insert(k, v);
            }
        }

        // Append prompt and stop (avoid override by extras)
        body_map.insert("prompt".into(), JsonValue::String(completion_prompt));
        if let Some(stops) = stop_from_convert {
            if !stops.is_empty() {
                body_map.insert("stop".into(), json!(stops));
            }
        }

        Ok((JsonValue::Object(body_map), warnings))
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync> LanguageModel for OpenAICompatibleCompletionLanguageModel<T> {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.cfg.supported_urls.clone()
    }

    async fn do_generate(
        &self,
        options: v2t::CallOptions,
    ) -> Result<crate::ai_sdk_core::GenerateResponse, SdkError> {
        let stream_resp = self.do_stream(options).await?;
        collect_stream_to_response(stream_resp, StreamCollectorConfig::default()).await
    }

    async fn do_stream(
        &self,
        options: v2t::CallOptions,
    ) -> Result<crate::ai_sdk_core::StreamResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );
        let (mut body, warnings) = self.build_request_body(&options)?;
        // Add stream controls
        if let Some(map) = body.as_object_mut() {
            map.insert("stream".into(), JsonValue::Bool(true));
            if self.cfg.include_usage {
                map.insert("stream_options".into(), json!({"include_usage": true}));
            }
        }

        let url = self.build_request_url();
        let headers = self.cfg.headers.clone();

        crate::provider_openai_compatible::stream::start_streaming(
            &self.cfg.http,
            url,
            headers,
            body,
            &self.cfg.transport_cfg,
            crate::provider_openai_compatible::stream::StreamSettings {
                warnings,
                include_raw: options.include_raw_chunks,
                include_usage: self.cfg.include_usage,
                provider_scope_name: self.cfg.provider_scope_name.clone(),
            },
            crate::provider_openai_compatible::stream::StreamMode::Completion,
        )
        .await
    }
}
