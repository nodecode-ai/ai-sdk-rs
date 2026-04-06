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

pub struct OpenAICompatibleChatLanguageModel<
    T: HttpTransport = crate::reqwest_transport::ReqwestTransport,
> {
    model_id: String,
    cfg: OpenAICompatibleChatConfig<T>,
}

fn insert_json_value(
    body_map: &mut serde_json::Map<String, JsonValue>,
    key: &str,
    value: Option<JsonValue>,
) {
    if let Some(value) = value {
        body_map.insert(key.into(), value);
    }
}

fn build_response_format(
    supports_structured_outputs: bool,
    response_format: &Option<v2t::ResponseFormat>,
    warnings: &mut Vec<v2t::CallWarning>,
) -> Option<JsonValue> {
    match response_format {
        Some(v2t::ResponseFormat::Json {
            schema,
            name,
            description,
        }) => {
            if supports_structured_outputs {
                Some(match schema {
                    Some(schema) => json!({
                        "type": "json_schema",
                        "json_schema": {
                            "schema": schema,
                            "name": name.clone().unwrap_or_else(|| "response".into()),
                            "description": description,
                        }
                    }),
                    None => json!({"type":"json_object"}),
                })
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
    }
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

        let response_format = build_response_format(
            self.cfg.supports_structured_outputs,
            &options.response_format,
            &mut warnings,
        );

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
        warnings.extend(prep.warnings);

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
        insert_json_value(&mut body_map, "user", user.map(JsonValue::String));
        insert_json_value(
            &mut body_map,
            "max_tokens",
            options.max_output_tokens.map(|value| json!(value)),
        );
        insert_json_value(
            &mut body_map,
            "temperature",
            options.temperature.map(|value| json!(value)),
        );
        insert_json_value(
            &mut body_map,
            "top_p",
            options.top_p.map(|value| json!(value)),
        );
        insert_json_value(
            &mut body_map,
            "frequency_penalty",
            options.frequency_penalty.map(|value| json!(value)),
        );
        insert_json_value(
            &mut body_map,
            "presence_penalty",
            options.presence_penalty.map(|value| json!(value)),
        );
        insert_json_value(
            &mut body_map,
            "stop",
            options.stop_sequences.as_ref().map(|value| json!(value)),
        );
        insert_json_value(
            &mut body_map,
            "seed",
            options.seed.map(|value| json!(value)),
        );
        insert_json_value(&mut body_map, "tools", prep.tools.map(|value| json!(value)));
        insert_json_value(&mut body_map, "tool_choice", prep.tool_choice);
        insert_json_value(&mut body_map, "response_format", response_format);
        insert_json_value(
            &mut body_map,
            "reasoning_effort",
            reasoning_effort.map(JsonValue::String),
        );
        insert_json_value(
            &mut body_map,
            "verbosity",
            text_verbosity.map(JsonValue::String),
        );
        if let Some(extras) = prov_extras {
            for (k, v) in extras {
                tracing::info!("[PROVOPTS]: chat extra {}", k);
                body_map.insert(k, v);
            }
        }
        Ok((JsonValue::Object(body_map), warnings))
    }
}

impl OpenAICompatibleChatLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn builder(
        model_id: impl Into<String>,
    ) -> crate::provider_openai_compatible::provider::OpenAICompatibleChatBuilder {
        crate::provider_openai_compatible::provider::OpenAICompatibleChatBuilder::new(model_id)
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
