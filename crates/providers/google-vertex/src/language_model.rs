use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Value as JsonValue};
use std::collections::{HashMap, HashSet};

use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{GenerateResponse, LanguageModel, PartStream, SdkError, StreamResponse};
use crate::ai_sdk_streaming_sse::SseDecoder;
use crate::ai_sdk_types::v2 as v2t;

use crate::provider_google_vertex::error::map_transport_error_to_sdk_error;
use crate::provider_google_vertex::options::{
    parse_google_vertex_provider_options, GoogleVertexProviderOptions,
};
use crate::provider_google_vertex::prepare_tools::{
    convert_json_schema_to_openapi_schema, prepare_tools,
};
use crate::provider_google_vertex::prompt::{convert_to_google_prompt, GooglePrompt};

const TRACE_PREFIX: &str = "[GOOGLE-VERTEX]";

pub struct GoogleVertexConfig<T: HttpTransport = crate::reqwest_transport::ReqwestTransport> {
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

pub struct GoogleVertexLanguageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport>
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
        let mut warnings: Vec<v2t::CallWarning> = Vec::new();

        // provider options
        let google_opts: Option<GoogleVertexProviderOptions> =
            parse_google_vertex_provider_options(&options.provider_options);

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
        if let Some(raw_google_opts) = options
            .provider_options
            .get("google-vertex")
            .or_else(|| options.provider_options.get("google"))
        {
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

impl GoogleVertexLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn provider_name_static() -> &'static str {
        "google.vertex"
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
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
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

        let mut content_out: Vec<v2t::Content> = Vec::new();
        let mut last_code_tool_id: Option<String> = None;
        for p in parts.iter() {
            if let Some(text) = p.get("text").and_then(|t| t.as_str()) {
                let pm = p
                    .get("thoughtSignature")
                    .and_then(|v| v.as_str())
                    .map(|sig| {
                        let mut outer = HashMap::new();
                        let mut inner = HashMap::new();
                        inner.insert(
                            "thoughtSignature".into(),
                            JsonValue::String(sig.to_string()),
                        );
                        outer.insert("google-vertex".into(), inner);
                        outer
                    });
                if p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false) {
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

            if let Some(fc) = p.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let args_obj = fc.get("args").cloned().unwrap_or(JsonValue::Null);
                let args_json = if args_obj.is_null() {
                    "{}".to_string()
                } else {
                    args_obj.to_string()
                };
                let thought_sig = p.get("thoughtSignature").and_then(|v| v.as_str());
                let provider_options = thought_sig.map(|sig| {
                    let mut outer = HashMap::new();
                    let mut inner = HashMap::new();
                    inner.insert(
                        "thoughtSignature".into(),
                        JsonValue::String(sig.to_string()),
                    );
                    outer.insert("google-vertex".into(), inner);
                    outer
                });
                let id = uuid::Uuid::new_v4().to_string();
                last_code_tool_id = Some(id.clone());
                content_out.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input: args_json,
                    provider_executed: false,
                    provider_metadata: None,
                    dynamic: false,
                    provider_options,
                }));
                continue;
            }

            if let Some(fr) = p.get("functionResponse") {
                let name = fr
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let response_val = fr.get("response").cloned().unwrap_or(JsonValue::Null);
                let payload = response_val.get("content").cloned().unwrap_or(response_val);
                content_out.push(v2t::Content::ToolResult {
                    tool_call_id: last_code_tool_id.clone().unwrap_or_else(|| name.clone()),
                    tool_name: name,
                    result: payload,
                    is_error: false,
                    provider_metadata: None,
                });
                continue;
            }

            if let Some(inline) = p.get("inlineData") {
                let mime = inline
                    .get("mimeType")
                    .and_then(|m| m.as_str())
                    .unwrap_or("application/octet-stream");
                let data = inline
                    .get("data")
                    .and_then(|d| d.as_str())
                    .unwrap_or_default()
                    .to_string();
                content_out.push(v2t::Content::File {
                    media_type: mime.to_string(),
                    data,
                });
            }
        }

        let mut usage = v2t::Usage::default();
        if let Some(usage_md) = usage_md {
            if let Some(input) = usage_md.get("promptTokenCount").and_then(|v| v.as_u64()) {
                usage.input_tokens = Some(input);
            }
            if let Some(output) = usage_md
                .get("candidatesTokenCount")
                .and_then(|v| v.as_u64())
            {
                usage.output_tokens = Some(output);
            }
            if let Some(total) = usage_md.get("totalTokenCount").and_then(|v| v.as_u64()) {
                usage.total_tokens = Some(total);
            }
        }

        Ok(GenerateResponse {
            content: content_out,
            finish_reason: v2t::FinishReason::Stop,
            usage,
            provider_metadata: None,
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
        let (mut inner, resp_headers) = <T as HttpTransport>::into_stream(resp);

        let include_raw = options.include_raw_chunks;

        let stream: PartStream = Box::pin(async_stream::try_stream! {
            yield v2t::StreamPart::StreamStart { warnings };
            let mut decoder = SseDecoder::new();
            let mut current_text_id: Option<String> = None;
            let mut current_reasoning_id: Option<String> = None;
            let mut last_code_tool_id: Option<String> = None;
            let mut emitted_source_urls: HashSet<String> = HashSet::new();
            let mut usage: v2t::Usage = v2t::Usage::default();
            let mut finish_reason: v2t::FinishReason = v2t::FinishReason::Unknown;
            let mut provider_metadata: Option<v2t::ProviderMetadata> = None;
            let mut has_tool_calls: bool = false;
            let mut block_counter: u64 = 0;


            macro_rules! handle_sse_event {
                ($ev:expr) => {{
                    let ev = $ev;
                    let data = String::from_utf8_lossy(&ev.data).to_string();
                    if data.trim().is_empty() {
                        continue;
                    }
                    let parsed: serde_json::Value = match serde_json::from_str(&data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if include_raw {
                        yield v2t::StreamPart::Raw { raw_value: parsed.clone() };
                    }

                    if let Some(u) = parsed.get("usageMetadata") {
                        usage = v2t::Usage {
                            input_tokens: u.get("promptTokenCount").and_then(|v| v.as_u64()),
                            output_tokens: u.get("candidatesTokenCount").and_then(|v| v.as_u64()),
                            total_tokens: u.get("totalTokenCount").and_then(|v| v.as_u64()),
                            reasoning_tokens: u.get("thoughtsTokenCount").and_then(|v| v.as_u64()),
                            cached_input_tokens: u.get("cachedContentTokenCount").and_then(|v| v.as_u64()),
                        };
                    }

                    if let Some(cands) = parsed.get("candidates").and_then(|v| v.as_array()) {
                        for cand in cands {
                            if let Some(content) = cand.get("content").and_then(|v| v.as_object()) {
                                let parts = content.get("parts").and_then(|p| p.as_array()).cloned().unwrap_or_default();
                                for p in parts {
                                    if let Some(ec) = p.get("executableCode").and_then(|v| v.as_object()) {
                                        if ec.get("code").and_then(|v| v.as_str()).is_some() {
                                            let id = uuid::Uuid::new_v4().to_string();
                                            last_code_tool_id = Some(id.clone());
                                            yield v2t::StreamPart::ToolInputStart { id: id.clone(), tool_name: "code_execution".into(), provider_executed: true, provider_metadata: None };
                                            let delta = serde_json::to_string(ec).unwrap_or("{}".into());
                                            yield v2t::StreamPart::ToolInputDelta {
                                                id: id.clone(),
                                                delta,
                                                provider_executed: true,
                                                provider_metadata: None,
                                            };
                                            yield v2t::StreamPart::ToolInputEnd {
                                                id: id.clone(),
                                                provider_executed: true,
                                                provider_metadata: None,
                                            };
                                            yield v2t::StreamPart::ToolCall(v2t::ToolCallPart { tool_call_id: id, tool_name: "code_execution".into(), input: serde_json::to_string(ec).unwrap_or("{}".into()), provider_executed: true, provider_metadata: None, dynamic: false, provider_options: None });
                                            has_tool_calls = true;
                                            continue;
                                        }
                                    }
                                    if let Some(res) = p.get("codeExecutionResult").and_then(|v| v.as_object()) {
                                        if let Some(id) = last_code_tool_id.take() {
                                            yield v2t::StreamPart::ToolResult { tool_call_id: id, tool_name: "code_execution".into(), result: json!({"outcome": res.get("outcome"), "output": res.get("output")}), is_error: false, preliminary: false, provider_metadata: None };
                                            continue;
                                        }
                                    }
                                    if let Some(txt) = p.get("text").and_then(|v| v.as_str()) {
                                        if !txt.is_empty() {
                                            let is_thought = p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                                            let thought_sig = p.get("thoughtSignature").and_then(|v| v.as_str());
                                            let pm = thought_sig.map(|sig| {
                                                let mut outer = HashMap::new();
                                                let mut inner = HashMap::new();
                                                inner.insert("thoughtSignature".into(), JsonValue::String(sig.to_string()));
                                                outer.insert("google-vertex".into(), inner);
                                                outer
                                            });
                                            if is_thought {
                                                if let Some(id) = current_text_id.take() { yield v2t::StreamPart::TextEnd { id, provider_metadata: None }; }
                                                if current_reasoning_id.is_none() { let id = format!("r-{}", block_counter); block_counter += 1; current_reasoning_id = Some(id.clone()); yield v2t::StreamPart::ReasoningStart { id, provider_metadata: pm.clone() }; }
                                                let id = current_reasoning_id.clone().unwrap();
                                                yield v2t::StreamPart::ReasoningDelta { id, delta: txt.to_string(), provider_metadata: pm };
                                            } else {
                                                if let Some(id) = current_reasoning_id.take() { yield v2t::StreamPart::ReasoningEnd { id, provider_metadata: None }; }
                                                if current_text_id.is_none() { let id = format!("t-{}", block_counter); block_counter += 1; current_text_id = Some(id.clone()); yield v2t::StreamPart::TextStart { id, provider_metadata: pm.clone() }; }
                                                let id = current_text_id.clone().unwrap();
                                                yield v2t::StreamPart::TextDelta { id, delta: txt.to_string(), provider_metadata: pm };
                                            }
                                            continue;
                                        }
                                    }
                                    if let Some(inline) = p.get("inlineData").and_then(|v| v.as_object()) {
                                        let media_type = inline.get("mimeType").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        let data = inline.get("data").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        yield v2t::StreamPart::File { media_type, data };
                                        continue;
                                    }
                                    if let Some(fc) = p.get("functionCall").and_then(|v| v.as_object()) {
                                        let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                        let args = fc.get("args").cloned().unwrap_or(json!({})).to_string();
                                        let thought_sig = p.get("thoughtSignature").and_then(|v| v.as_str());
                                        let provider_options = thought_sig.map(|sig| {
                                            let mut outer = HashMap::new();
                                            let mut inner = HashMap::new();
                                            inner.insert("thoughtSignature".into(), JsonValue::String(sig.to_string()));
                                            outer.insert("google-vertex".into(), inner);
                                            outer
                                        });
                                        let id = uuid::Uuid::new_v4().to_string();
                                        yield v2t::StreamPart::ToolInputStart { id: id.clone(), tool_name: name.clone(), provider_executed: false, provider_metadata: None };
                                        yield v2t::StreamPart::ToolInputDelta {
                                            id: id.clone(),
                                            delta: args.clone(),
                                            provider_executed: false,
                                            provider_metadata: None,
                                        };
                                        yield v2t::StreamPart::ToolInputEnd {
                                            id: id.clone(),
                                            provider_executed: false,
                                            provider_metadata: None,
                                        };
                                        yield v2t::StreamPart::ToolCall(v2t::ToolCallPart { tool_call_id: id, tool_name: name, input: args, provider_executed: false, provider_metadata: None, dynamic: false, provider_options });
                                        has_tool_calls = true;
                                        continue;
                                    }
                                }
                            }

                            if let Some(gm) = cand.get("groundingMetadata") {
                                if let Some(chunks) = gm.get("groundingChunks").and_then(|v| v.as_array()) {
                                    for ch in chunks {
                                        if let Some(web) = ch.get("web").and_then(|v| v.as_object()) {
                                            if let Some(url) = web.get("uri").and_then(|v| v.as_str()) {
                                                if !emitted_source_urls.contains(url) {
                                                    emitted_source_urls.insert(url.to_string());
                                                    let id = uuid::Uuid::new_v4().to_string();
                                                    let title = web.get("title").and_then(|v| v.as_str()).map(|s| s.to_string());
                                                    yield v2t::StreamPart::SourceUrl { id, url: url.to_string(), title, provider_metadata: None };
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(fr) = cand.get("finishReason").and_then(|v| v.as_str()) {
                                finish_reason = match fr {
                                    "STOP" => if has_tool_calls { v2t::FinishReason::ToolCalls } else { v2t::FinishReason::Stop },
                                    "MAX_TOKENS" => v2t::FinishReason::Length,
                                    "IMAGE_SAFETY" | "RECITATION" | "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => v2t::FinishReason::ContentFilter,
                                    "FINISH_REASON_UNSPECIFIED" | "OTHER" => v2t::FinishReason::Other,
                                    "MALFORMED_FUNCTION_CALL" => v2t::FinishReason::Error,
                                    _ => v2t::FinishReason::Unknown,
                                };
                                let mut inner_map = HashMap::new();
                                inner_map.insert("groundingMetadata".into(), cand.get("groundingMetadata").cloned().unwrap_or(JsonValue::Null));
                                inner_map.insert("urlContextMetadata".into(), cand.get("urlContextMetadata").cloned().unwrap_or(JsonValue::Null));
                                inner_map.insert("safetyRatings".into(), cand.get("safetyRatings").cloned().unwrap_or(JsonValue::Null));
                                let usage_has = usage.input_tokens.is_some() || usage.output_tokens.is_some() || usage.total_tokens.is_some() || usage.reasoning_tokens.is_some() || usage.cached_input_tokens.is_some();
                                if usage_has { inner_map.insert("usageMetadata".into(), json!({
                                    "promptTokenCount": usage.input_tokens,
                                    "candidatesTokenCount": usage.output_tokens,
                                    "totalTokenCount": usage.total_tokens,
                                    "thoughtsTokenCount": usage.reasoning_tokens,
                                    "cachedContentTokenCount": usage.cached_input_tokens,
                                })); }
                                let mut outer = HashMap::new(); outer.insert("google-vertex".into(), inner_map); provider_metadata = Some(outer);
                            }
                        }
                    }
                }};
            }

            while let Some(chunk_res) = inner.next().await {
                match chunk_res {
                    Ok(chunk) => {
                        for ev in decoder.push(&chunk) {
                            handle_sse_event!(ev);
                        }
                    }
                    Err(te) => {
                        let e = map_transport_error_to_sdk_error(te);
                        yield v2t::StreamPart::Error { error: serde_json::json!({"message": e.to_string()}) };
                        break;
                    }
                }
            }

            for ev in decoder.finish() {
                handle_sse_event!(ev);
            }

            if let Some(id) = current_text_id.take() { yield v2t::StreamPart::TextEnd { id, provider_metadata: None }; }
            if let Some(id) = current_reasoning_id.take() { yield v2t::StreamPart::ReasoningEnd { id, provider_metadata: None }; }

            yield v2t::StreamPart::Finish { usage, finish_reason, provider_metadata };
        });

        Ok(StreamResponse {
            stream,
            request_body: Some(body),
            response_headers: Some(resp_headers.into_iter().collect()),
        })
    }
}
