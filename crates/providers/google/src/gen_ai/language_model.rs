use async_trait::async_trait;
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;

use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{GenerateResponse, LanguageModel, SdkError, StreamResponse};
use crate::ai_sdk_types::v2 as v2t;

use super::options::{parse_google_provider_options, GoogleProviderOptions, ThinkingConfig};
use super::prompt::{convert_to_google_prompt, GooglePrompt};
use crate::provider_google::error::map_transport_error_to_sdk_error;
use crate::provider_google::prepare_tools::{convert_json_schema_to_openapi_schema, prepare_tools};
use crate::provider_google::shared::stream_core::build_google_stream_part_stream;

const TRACE_PREFIX: &str = "[GOOGLE-V2]";

pub struct GoogleGenAiConfig<T: HttpTransport = crate::reqwest_transport::ReqwestTransport> {
    pub provider_name: &'static str,
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub supported_urls: HashMap<String, Vec<String>>,
    pub query_params: Vec<(String, String)>,
    pub default_options: Option<v2t::ProviderOptions>,
    /// Emit warnings for options that are only supported on Vertex (e.g., includeThoughts).
    pub warn_on_include_thoughts: bool,
}

pub struct GoogleGenAiLanguageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport> {
    pub model_id: String,
    pub cfg: GoogleGenAiConfig<T>,
}

impl<T: HttpTransport> GoogleGenAiLanguageModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: GoogleGenAiConfig<T>) -> Self {
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
        let mut warnings: Vec<v2t::CallWarning> = Vec::new();

        // provider options
        let google_opts: Option<GoogleProviderOptions> =
            parse_google_provider_options(&options.provider_options);

        // thinking include thoughts warning for non-vertex provider
        if self.cfg.warn_on_include_thoughts {
            if let Some(ref g) = google_opts {
                if let Some(ThinkingConfig {
                    include_thoughts: Some(true),
                    ..
                }) = &g.thinking_config
                {
                    warnings.push(v2t::CallWarning::Other { message: "The 'includeThoughts' option is only supported with the Google Vertex provider and might not be supported or could behave unexpectedly with the current Google provider (google.gen-ai).".into() });
                }
            }
        } else if let Some(ref g) = google_opts {
            if let Some(ThinkingConfig {
                include_thoughts: Some(true),
                ..
            }) = &g.thinking_config
            {
                // Vertex supports includeThoughts; no warning needed.
            }
        }

        // prompt conversion
        let prompt = convert_to_google_prompt(&options.prompt, self.is_gemma())?;
        let GooglePrompt {
            system_instruction,
            contents,
        } = prompt;

        // tools
        let prepared = prepare_tools(&options.tools, &options.tool_choice, &self.model_id);
        warnings.extend(prepared.tool_warnings.into_iter());

        // response format
        let (response_mime_type, response_schema) = match &options.response_format {
            Some(v2t::ResponseFormat::Json { schema, .. }) => {
                let mime = Some("application/json".to_string());
                let resp_schema = match schema {
                    Some(s) => {
                        let conv = convert_json_schema_to_openapi_schema(s);
                        if conv.is_null() {
                            None
                        } else {
                            Some(conv)
                        }
                    }
                    None => None,
                };
                (mime, resp_schema)
            }
            _ => (None, None),
        };

        let mut generation_config = json!({
            "maxOutputTokens": options.max_output_tokens,
            "temperature": options.temperature,
            "topK": options.top_k,
            "topP": options.top_p,
            "frequencyPenalty": options.frequency_penalty,
            "presencePenalty": options.presence_penalty,
            "stopSequences": options.stop_sequences,
            "seed": options.seed,
        });

        if let Some(m) = response_mime_type {
            generation_config["responseMimeType"] = json!(m);
        }
        if let Some(s) = response_schema {
            generation_config["responseSchema"] = s;
        }

        let mut request_body_overrides: Option<serde_json::Map<String, JsonValue>> = None;
        let mut threshold_override: Option<JsonValue> = None;
        if let Some(raw_google_opts) = options.provider_options.get("google") {
            let mut extras = raw_google_opts.clone();
            if let Some(gen_cfg_extra) = extras
                .remove("generationConfig")
                .or_else(|| extras.remove("generation_config"))
            {
                crate::ai_sdk_core::options::deep_merge(&mut generation_config, &gen_cfg_extra);
            }

            for key in [
                "responseModalities",
                "response_modalities",
                "thinkingConfig",
                "thinking_config",
                "cachedContent",
                "cached_content",
                "structuredOutputs",
                "structured_outputs",
                "safetySettings",
                "safety_settings",
                "audioTimestamp",
                "audio_timestamp",
                "labels",
            ] {
                extras.remove(key);
            }

            if let Some(thresh) = extras.remove("threshold") {
                threshold_override = Some(thresh);
            }

            if !extras.is_empty() {
                let mut map = serde_json::Map::new();
                for (k, v) in extras.into_iter() {
                    map.insert(k, v);
                }
                request_body_overrides = Some(map);
            }
        }

        if let Some(g) = &google_opts {
            if let Some(v) = &g.response_modalities {
                generation_config["responseModalities"] = json!(v);
            }
            if let Some(v) = &g.thinking_config {
                generation_config["thinkingConfig"] =
                    serde_json::to_value(v).unwrap_or(JsonValue::Null);
            }
            if let Some(v) = &g.audio_timestamp {
                generation_config["audioTimestamp"] = json!(v);
            }
        }

        let mut body = json!({
            "generationConfig": generation_config,
            "contents": contents,
        });
        if !self.is_gemma() {
            body["systemInstruction"] =
                serde_json::to_value(system_instruction).unwrap_or(JsonValue::Null);
        }

        if let Some(t) = google_opts.as_ref().and_then(|g| g.threshold.clone()) {
            body["threshold"] = json!(t);
        }
        if let Some(thresh) = threshold_override {
            body["threshold"] = thresh;
        }

        if let Some(g) = &google_opts {
            if let Some(v) = &g.safety_settings {
                body["safetySettings"] = json!(v);
            }
            if let Some(v) = &g.cached_content {
                body["cachedContent"] = json!(v);
            }
            if let Some(v) = &g.labels {
                body["labels"] = json!(v);
            }
        }

        if let Some(t) = prepared.tools {
            body["tools"] = t;
        }
        if let Some(tc) = prepared.tool_config {
            body["toolConfig"] = tc;
        }

        if let Some(overrides) = request_body_overrides {
            if !overrides.is_empty() {
                tracing::info!(
                    "{}: applying provider request overrides {:?}",
                    TRACE_PREFIX,
                    overrides.keys().collect::<Vec<_>>()
                );
            }
            crate::ai_sdk_core::options::merge_options_with_disallow(
                &mut body,
                &JsonValue::Object(overrides),
                &["contents", "systemInstruction"],
            );
        }

        Ok((body, warnings))
    }
}

impl GoogleGenAiLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn provider_name_static() -> &'static str {
        "google.gen-ai"
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync + 'static> LanguageModel for GoogleGenAiLanguageModel<T> {
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
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );
        let (body, warnings) = self.build_body(&options)?;
        let url = self.url_generate();

        // lowercase, canonical headers; skip internal
        let headers: Vec<(String, String)> = self
            .cfg
            .headers
            .iter()
            .filter(|(k, _)| !crate::ai_sdk_core::options::is_internal_sdk_header(k))
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

        // Extract candidate
        let candidate = resp_json
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(0))
            .cloned()
            .unwrap_or(JsonValue::Null);
        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();
        let usage_md = resp_json.get("usageMetadata").cloned();

        // Build content
        let mut content_out: Vec<v2t::Content> = Vec::new();
        let mut last_code_tool_id: Option<String> = None;
        for p in parts.iter() {
            // executable code
            if let Some(exe) = p.get("executableCode").and_then(|v| v.as_object()) {
                if exe.get("code").and_then(|v| v.as_str()).is_some() {
                    let id = uuid::Uuid::new_v4().to_string();
                    last_code_tool_id = Some(id.clone());
                    content_out.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                        tool_call_id: id,
                        tool_name: "code_execution".into(),
                        input: serde_json::to_string(exe).unwrap_or("{}".into()),
                        provider_executed: true,
                        provider_metadata: None,
                        dynamic: false,
                        provider_options: None,
                    }));
                    continue;
                }
            }
            if let Some(res) = p.get("codeExecutionResult").and_then(|v| v.as_object()) {
                if let Some(id) = last_code_tool_id.take() {
                    content_out.push(v2t::Content::ToolResult {
                        tool_call_id: id,
                        tool_name: "code_execution".into(),
                        result: json!({"outcome": res.get("outcome"), "output": res.get("output")}),
                        is_error: false,
                        provider_metadata: None,
                    });
                    continue;
                }
            }
            // text/thought
            if let Some(text) = p.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    let is_thought = p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                    let thought_sig = p
                        .get("thoughtSignature")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let pm = thought_sig.map(|sig| {
                        let mut outer = HashMap::new();
                        let mut inner = HashMap::new();
                        inner.insert("thoughtSignature".into(), JsonValue::String(sig));
                        outer.insert("google".into(), inner);
                        outer
                    });
                    if is_thought {
                        content_out.push(v2t::Content::Reasoning {
                            text: text.to_string(),
                            provider_metadata: pm,
                        });
                    } else {
                        content_out.push(v2t::Content::Text {
                            text: text.to_string(),
                            provider_metadata: pm,
                        });
                    }
                    continue;
                }
            }
            // functionCall
            if let Some(fc) = p.get("functionCall").and_then(|v| v.as_object()) {
                let name = fc
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = fc.get("args").cloned().unwrap_or(json!({}));
                let thought_sig = p.get("thoughtSignature").and_then(|v| v.as_str());
                let provider_options = thought_sig.map(|sig| {
                    let mut outer = HashMap::new();
                    let mut inner = HashMap::new();
                    inner.insert(
                        "thoughtSignature".into(),
                        JsonValue::String(sig.to_string()),
                    );
                    outer.insert("google".into(), inner);
                    outer
                });
                let id = uuid::Uuid::new_v4().to_string();
                content_out.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input: args.to_string(),
                    provider_executed: false,
                    provider_metadata: None,
                    dynamic: false,
                    provider_options,
                }));
                continue;
            }
            // inlineData
            if let Some(inline) = p.get("inlineData").and_then(|v| v.as_object()) {
                let media_type = inline
                    .get("mimeType")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let data = inline
                    .get("data")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                content_out.push(v2t::Content::File { media_type, data });
                continue;
            }
        }

        // Add sources from groundingMetadata
        if let Some(cand) = candidate.as_object() {
            if let Some(gm) = cand.get("groundingMetadata") {
                if let Some(chunks) = gm.get("groundingChunks").and_then(|v| v.as_array()) {
                    for ch in chunks {
                        if let Some(web) = ch.get("web").and_then(|v| v.as_object()) {
                            if let Some(url) = web.get("uri").and_then(|v| v.as_str()) {
                                let id = uuid::Uuid::new_v4().to_string();
                                let title = web
                                    .get("title")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                content_out.push(v2t::Content::SourceUrl {
                                    id,
                                    url: url.to_string(),
                                    title,
                                    provider_metadata: None,
                                });
                            }
                        }
                    }
                }
            }
        }

        // finish reason
        let has_tool_calls = content_out
            .iter()
            .any(|c| matches!(c, v2t::Content::ToolCall(_)));
        let finish_reason = match candidate.get("finishReason").and_then(|v| v.as_str()) {
            Some("STOP") => {
                if has_tool_calls {
                    v2t::FinishReason::ToolCalls
                } else {
                    v2t::FinishReason::Stop
                }
            }
            Some("MAX_TOKENS") => v2t::FinishReason::Length,
            Some(
                "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT"
                | "SPII",
            ) => v2t::FinishReason::ContentFilter,
            Some("FINISH_REASON_UNSPECIFIED" | "OTHER") => v2t::FinishReason::Other,
            Some("MALFORMED_FUNCTION_CALL") => v2t::FinishReason::Error,
            _ => v2t::FinishReason::Unknown,
        };

        // usage
        let usage = if let Some(u) = usage_md.as_ref() {
            v2t::Usage {
                input_tokens: u.get("promptTokenCount").and_then(|v| v.as_u64()),
                output_tokens: u.get("candidatesTokenCount").and_then(|v| v.as_u64()),
                total_tokens: u.get("totalTokenCount").and_then(|v| v.as_u64()),
                reasoning_tokens: u.get("thoughtsTokenCount").and_then(|v| v.as_u64()),
                cached_input_tokens: u.get("cachedContentTokenCount").and_then(|v| v.as_u64()),
            }
        } else {
            v2t::Usage::default()
        };

        // provider metadata payload
        let mut provider_metadata: Option<v2t::ProviderMetadata> = None;
        if let Some(obj) = candidate.as_object() {
            let mut inner = HashMap::new();
            inner.insert(
                "groundingMetadata".into(),
                obj.get("groundingMetadata")
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            );
            inner.insert(
                "urlContextMetadata".into(),
                obj.get("urlContextMetadata")
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            );
            inner.insert(
                "safetyRatings".into(),
                obj.get("safetyRatings").cloned().unwrap_or(JsonValue::Null),
            );
            if let Some(u) = usage_md {
                inner.insert("usageMetadata".into(), u);
            }
            let mut outer = HashMap::new();
            outer.insert("google".into(), inner);
            provider_metadata = Some(outer);
        }

        Ok(GenerateResponse {
            content: content_out,
            finish_reason,
            usage,
            provider_metadata,
            request_body: Some(body),
            response_headers: Some(resp_headers.into_iter().collect()),
            response_body: Some(resp_json),
            warnings,
        })
    }

    async fn do_stream(&self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
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
            .filter(|(k, _)| !crate::ai_sdk_core::options::is_internal_sdk_header(k))
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
        let stream = build_google_stream_part_stream(inner, warnings, include_raw, "google");

        Ok(StreamResponse {
            stream,
            request_body: Some(body),
            response_headers: Some(resp_headers.into_iter().collect()),
        })
    }
}
