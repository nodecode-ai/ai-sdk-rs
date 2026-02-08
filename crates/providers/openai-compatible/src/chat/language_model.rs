use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{json, Value as JsonValue};

use crate::ai_sdk_core::stream_collect::{collect_stream_to_response, StreamCollectorConfig};
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_types::v2 as v2t;

use crate::provider_openai_compatible::chat::convert::convert_to_openai_compatible_chat_messages as convert_messages;
use crate::provider_openai_compatible::chat::options::{
    parse_openai_compatible_chat_provider_options, OpenAICompatibleChatProviderOptions,
};
use crate::provider_openai_compatible::chat::prepare_tools::prepare_tools;

pub struct OpenAICompatibleChatConfig<T: HttpTransport> {
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub include_usage: bool,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub query_params: Vec<(String, String)>,
    pub supports_structured_outputs: bool,
    pub default_options: Option<v2t::ProviderOptions>,
}

pub struct OpenAICompatibleChatLanguageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport>
{
    model_id: String,
    cfg: OpenAICompatibleChatConfig<T>,
}

impl<T: HttpTransport> OpenAICompatibleChatLanguageModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: OpenAICompatibleChatConfig<T>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg,
        }
    }

    fn build_request_url(&self) -> String {
        let base = self.cfg.base_url.trim_end_matches('/');
        let mut url = format!("{}/chat/completions", base);
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
        if options.top_k.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "topK".into(),
                details: None,
            });
        }

        // Response format JSON handling (structured outputs)
        let response_format = match &options.response_format {
            Some(v2t::ResponseFormat::Json {
                schema,
                name,
                description,
            }) => {
                if self.cfg.supports_structured_outputs {
                    if let Some(s) = schema {
                        Some(
                            json!({"type":"json_schema","json_schema": {"schema": s, "name": name.clone().unwrap_or_else(|| "response".into()), "description": description } }),
                        )
                    } else {
                        Some(json!({"type":"json_object"}))
                    }
                } else {
                    warnings.push(v2t::CallWarning::UnsupportedSetting {
                        setting: "responseFormat".into(),
                        details: Some(
                            "JSON response format schema is only supported with structuredOutputs"
                                .into(),
                        ),
                    });
                    Some(json!({"type":"json_object"}))
                }
            }
            _ => None,
        };

        // Provider options and extras
        let scope_names = ["openai-compatible", self.cfg.provider_scope_name.as_str()];
        let (prov_opts, prov_extras) =
            parse_openai_compatible_chat_provider_options(&options.provider_options, &scope_names);
        tracing::info!(
            "[PROVOPTS]: chat scopes={:?} has_reasoning_effort={:?}",
            scope_names,
            prov_opts.reasoning_effort
        );
        tracing::info!(
            "[PROVOPTS]: chat extras_keys={:?}",
            prov_extras
                .as_ref()
                .map(|m| m.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        );

        // Prepare tools & tool_choice
        let prep = prepare_tools(&options.tools, &options.tool_choice);
        warnings.extend(prep.warnings.into_iter());

        let OpenAICompatibleChatProviderOptions {
            user,
            reasoning_effort,
            text_verbosity,
        } = prov_opts;

        let mut body_map = serde_json::Map::new();
        body_map.insert("model".into(), json!(self.model_id));
        body_map.insert(
            "messages".into(),
            json!(convert_messages(
                &self.cfg.provider_scope_name,
                &options.prompt
            )),
        );
        if let Some(u) = user {
            body_map.insert("user".into(), json!(u));
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
        if let Some(stop) = options.stop_sequences.as_ref() {
            body_map.insert("stop".into(), json!(stop));
        }
        if let Some(seed) = options.seed {
            body_map.insert("seed".into(), json!(seed));
        }
        if let Some(tools) = prep.tools {
            body_map.insert("tools".into(), json!(tools));
        }
        if let Some(choice) = prep.tool_choice {
            body_map.insert("tool_choice".into(), choice);
        }
        if let Some(rf) = response_format {
            body_map.insert("response_format".into(), rf);
        }
        if let Some(re) = reasoning_effort {
            body_map.insert("reasoning_effort".into(), JsonValue::String(re));
        }
        if let Some(verbosity) = text_verbosity {
            body_map.insert("verbosity".into(), JsonValue::String(verbosity));
        }
        if let Some(extras) = prov_extras {
            for (k, v) in extras {
                tracing::info!("[PROVOPTS]: chat extra {}", k);
                body_map.insert(k, v);
            }
        }
        Ok((JsonValue::Object(body_map), warnings))
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync> LanguageModel for OpenAICompatibleChatLanguageModel<T> {
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
        collect_stream_to_response(
            stream_resp,
            StreamCollectorConfig {
                allow_reasoning: true,
                allow_tool_calls: true,
                ..StreamCollectorConfig::default()
            },
        )
        .await
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
            crate::provider_openai_compatible::stream::StreamMode::Chat,
        )
        .await
    }
}
