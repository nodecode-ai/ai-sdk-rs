use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::build_call_options;
use crate::ai_sdk_core::transport::HttpTransport;
use crate::ai_sdk_core::{GenerateResponse, LanguageModel, SdkError, StreamResponse};
use crate::ai_sdk_types::v2 as v2t;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::collections::{BTreeMap, HashMap, HashSet};
use tracing::instrument;

use crate::provider_amazon_bedrock::config::BedrockConfig;
use crate::provider_amazon_bedrock::error::map_transport_error;
use crate::provider_amazon_bedrock::messages::{convert_prompt, ConvertedPrompt};
use crate::provider_amazon_bedrock::options::{
    map_to_owned, parse_bedrock_provider_options, BedrockProviderOptions,
};
use crate::provider_amazon_bedrock::signing::{prepare_request, PreparedRequest};

const TRACE_PREFIX: &str = "[BEDROCK]";

pub struct BedrockLanguageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport> {
    pub model_id: String,
    pub cfg: BedrockConfig<T>,
}

impl<T: HttpTransport> BedrockLanguageModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: BedrockConfig<T>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg,
        }
    }

    fn build_model_url(&self, suffix: &str) -> String {
        self.cfg.endpoint_for_model(&self.model_id, suffix)
    }

    fn supported_urls_map(&self) -> HashMap<String, Vec<String>> {
        self.cfg.supported_urls.clone()
    }

    fn base_headers(&self) -> Vec<(String, String)> {
        self.cfg
            .headers
            .iter()
            .filter(|(k, _)| !sdkopt::is_internal_sdk_header(k))
            .cloned()
            .collect()
    }
}

#[async_trait]
impl<T: HttpTransport + Send + Sync> LanguageModel for BedrockLanguageModel<T> {
    fn provider_name(&self) -> &'static str {
        self.cfg.provider_name
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.supported_urls_map()
    }

    #[instrument(name = "bedrock.do_generate", skip_all, fields(model = %self.model_id))]
    async fn do_generate(&self, options: v2t::CallOptions) -> Result<GenerateResponse, SdkError> {
        let options = build_call_options(
            options,
            &self.cfg.provider_scope_name,
            self.cfg.default_options.as_ref(),
        );

        let BuildCommandResult {
            command,
            warnings,
            uses_json_response_tool,
            betas,
            provider_metadata_seed,
        } = build_command(&self.model_id, &options)?;

        let mut headers = merge_headers(self.base_headers(), &options.headers);
        if !betas.is_empty() {
            headers.push((
                "anthropic-beta".to_string(),
                betas.into_iter().collect::<Vec<_>>().join(","),
            ));
        }

        let url = self.build_model_url("/converse");
        let PreparedRequest { body, headers } = prepare_request(
            &self.cfg.auth,
            &url,
            JsonValue::Object(command.clone()),
            &headers,
            &self.cfg.transport_cfg,
        )?;

        tracing::info!("{}: POST {}", TRACE_PREFIX, url);

        let (resp_body, resp_headers) = match self
            .cfg
            .http
            .post_json(&url, &headers, &body, &self.cfg.transport_cfg)
            .await
        {
            Ok(v) => v,
            Err(err) => {
                let mapped = map_transport_error(err);
                tracing::info!("{}: request failed: {}", TRACE_PREFIX, mapped);
                return Err(mapped);
            }
        };

        let response: ConverseResponse = serde_json::from_value(resp_body.clone())?;
        let ConverseResponse {
            metrics: _,
            output,
            stop_reason,
            trace,
            usage: raw_usage,
        } = response;

        let content = map_response_content(&output.message.content, uses_json_response_tool)?;

        let finish_reason = map_finish_reason(stop_reason.as_deref());
        let usage = map_usage(raw_usage.as_ref());

        let provider_metadata = build_provider_metadata(
            provider_metadata_seed,
            trace,
            raw_usage.as_ref(),
            uses_json_response_tool,
        );

        let response_headers = Some(headers_to_map(&resp_headers));

        Ok(GenerateResponse {
            content,
            finish_reason,
            usage,
            provider_metadata,
            request_body: Some(JsonValue::Object(command)),
            response_headers,
            response_body: Some(resp_body),
            warnings,
        })
    }

    async fn do_stream(&self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let _ = options;
        Err(SdkError::Transport(
            crate::ai_sdk_core::error::TransportError::Other(
                "Amazon Bedrock streaming not yet implemented".into(),
            ),
        ))
    }
}

struct BuildCommandResult {
    command: JsonMap<String, JsonValue>,
    warnings: Vec<v2t::CallWarning>,
    uses_json_response_tool: bool,
    betas: HashSet<String>,
    provider_metadata_seed: Option<JsonMap<String, JsonValue>>,
}

fn build_command(
    _model_id: &str,
    options: &v2t::CallOptions,
) -> Result<BuildCommandResult, SdkError> {
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();

    if options.frequency_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "frequencyPenalty".into(),
            details: None,
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "presencePenalty".into(),
            details: None,
        });
    }
    if options.seed.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "seed".into(),
            details: None,
        });
    }

    let bedrock_opts = parse_bedrock_provider_options(&options.provider_options)
        .unwrap_or_else(|| BedrockProviderOptions::default());

    let mut uses_json_response_tool = false;
    let mut json_response_tool: Option<v2t::FunctionTool> = None;
    if let Some(response_format) = &options.response_format {
        match response_format {
            v2t::ResponseFormat::Text => {}
            v2t::ResponseFormat::Json { schema, .. } => {
                if let Some(schema) = schema {
                    uses_json_response_tool = true;
                    json_response_tool = Some(v2t::FunctionTool {
                        name: "json".into(),
                        description: Some("Respond with a JSON object.".into()),
                        input_schema: schema.clone(),
                        provider_options: None,
                        r#type: v2t::FunctionToolType::Function,
                    });
                } else {
                    warnings.push(v2t::CallWarning::UnsupportedSetting {
                        setting: "responseFormat".into(),
                        details: Some(
                            "JSON response format requires a schema; request ignored.".into(),
                        ),
                    });
                }
            }
        }
    }

    let mut tools_for_request: Vec<v2t::FunctionTool> = Vec::new();
    for tool in &options.tools {
        match tool {
            v2t::Tool::Function(f) => tools_for_request.push(f.clone()),
            v2t::Tool::Provider(p) => {
                warnings.push(v2t::CallWarning::UnsupportedTool {
                    tool_name: p.name.clone(),
                    details: Some("provider tools are not supported".into()),
                });
            }
        }
    }
    if uses_json_response_tool {
        if !tools_for_request.is_empty() {
            warnings.push(v2t::CallWarning::Other {
                message: "JSON response format does not support additional tools. Provided tools are ignored.".into(),
            });
            tools_for_request.clear();
        }
        if let Some(tool) = json_response_tool.clone() {
            tools_for_request.push(tool);
        }
    }

    let mut tool_choice = options.tool_choice.clone();
    if uses_json_response_tool {
        tool_choice = Some(v2t::ToolChoice::Tool {
            name: "json".into(),
        });
    }

    let PreparedTools {
        tool_config,
        mut additional_model_fields,
        betas,
        mut tool_warnings,
        has_tools,
    } = prepare_tools(&tools_for_request, &tool_choice);
    warnings.extend(tool_warnings.drain(..));

    let (filtered_prompt, prompt_warnings) = filter_prompt_if_no_tools(&options.prompt, has_tools);
    if let Some(w) = prompt_warnings {
        warnings.push(w);
    }

    let ConvertedPrompt { system, messages } = convert_prompt(&filtered_prompt)?;

    let mut command = JsonMap::new();
    if !system.is_empty() {
        command.insert("system".into(), JsonValue::Array(system));
    }
    command.insert("messages".into(), JsonValue::Array(messages));

    let mut inference = JsonMap::new();
    if let Some(max_tokens) = options.max_output_tokens {
        inference.insert("maxTokens".into(), json!(max_tokens));
    }
    if let Some(temp) = options.temperature {
        inference.insert("temperature".into(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        inference.insert("topP".into(), json!(top_p));
    }
    if let Some(top_k) = options.top_k {
        inference.insert("topK".into(), json!(top_k));
    }
    if let Some(stops) = options.stop_sequences.as_ref() {
        if !stops.is_empty() {
            inference.insert("stopSequences".into(), json!(stops));
        }
    }

    let mut provider_metadata_seed: Option<JsonMap<String, JsonValue>> = None;
    if let Some(reasoning) = bedrock_opts.reasoning_config.as_ref() {
        if matches!(reasoning.r#type.as_deref(), Some("enabled")) {
            let budget = reasoning.budget_tokens.unwrap_or(0) as u64;
            if budget > 0 {
                let entry = additional_model_fields
                    .get_or_insert_with(JsonMap::new)
                    .entry("thinking".to_string())
                    .or_insert_with(|| JsonValue::Object(JsonMap::new()));
                if let JsonValue::Object(map) = entry {
                    map.insert("type".into(), JsonValue::String("enabled".into()));
                    map.insert("budget_tokens".into(), JsonValue::Number(budget.into()));
                }
                if let Some(existing) = inference.get_mut("maxTokens") {
                    if let Some(v) = existing.as_u64() {
                        *existing = JsonValue::Number((v + budget).into());
                    }
                } else {
                    inference.insert(
                        "maxTokens".into(),
                        JsonValue::Number((budget + 4096).into()),
                    );
                }
            }
            if let Some(temp) = inference.remove("temperature") {
                let _ = temp;
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "temperature".into(),
                    details: Some("temperature is not supported when reasoning is enabled".into()),
                });
            }
            if let Some(val) = inference.remove("topP") {
                let _ = val;
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "topP".into(),
                    details: Some("topP is not supported when reasoning is enabled".into()),
                });
            }
            if let Some(val) = inference.remove("topK") {
                let _ = val;
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "topK".into(),
                    details: Some("topK is not supported when reasoning is enabled".into()),
                });
            }
        }
    }

    if !inference.is_empty() {
        command.insert("inferenceConfig".into(), JsonValue::Object(inference));
    }

    if let Some(mut tool_cfg) = tool_config {
        if let JsonValue::Object(obj) = &mut tool_cfg {
            if obj.is_empty() {
                tool_cfg = JsonValue::Null;
            }
        }
        if !tool_cfg.is_null() {
            command.insert("toolConfig".into(), tool_cfg);
        }
    }

    if let Some(extra) = map_to_owned(&bedrock_opts.additional_model_request_fields) {
        additional_model_fields
            .get_or_insert_with(JsonMap::new)
            .extend(extra);
    }

    if let Some(guardrails) = map_to_owned(&bedrock_opts.guardrail_config) {
        command.insert("guardrailConfig".into(), JsonValue::Object(guardrails));
    }
    if let Some(guardrails_stream) = map_to_owned(&bedrock_opts.guardrail_stream_config) {
        command.insert(
            "guardrailStreamConfig".into(),
            JsonValue::Object(guardrails_stream),
        );
    }

    if let Some(extra) = additional_model_fields {
        if !extra.is_empty() {
            command.insert(
                "additionalModelRequestFields".into(),
                JsonValue::Object(extra),
            );
        }
    }

    if uses_json_response_tool {
        provider_metadata_seed = Some(JsonMap::new());
    }

    Ok(BuildCommandResult {
        command,
        warnings,
        uses_json_response_tool,
        betas,
        provider_metadata_seed,
    })
}

fn merge_headers(
    config_headers: Vec<(String, String)>,
    call_headers: &HashMap<String, String>,
) -> Vec<(String, String)> {
    let mut map: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (k, v) in config_headers {
        map.insert(k.to_ascii_lowercase(), (k, v));
    }
    for (k, v) in call_headers {
        if sdkopt::is_internal_sdk_header(k) {
            continue;
        }
        map.insert(k.to_ascii_lowercase(), (k.clone(), v.clone()));
    }
    map.into_values().collect()
}

fn filter_prompt_if_no_tools(
    prompt: &[v2t::PromptMessage],
    has_tools: bool,
) -> (Vec<v2t::PromptMessage>, Option<v2t::CallWarning>) {
    if has_tools {
        return (prompt.to_vec(), None);
    }

    let mut mutated = false;
    let mut filtered: Vec<v2t::PromptMessage> = Vec::with_capacity(prompt.len());
    for message in prompt {
        match message {
            v2t::PromptMessage::System { .. } => filtered.push(message.clone()),
            v2t::PromptMessage::User { .. } => filtered.push(message.clone()),
            v2t::PromptMessage::Tool { .. } => {
                mutated = true;
            }
            v2t::PromptMessage::Assistant {
                content,
                provider_options,
            } => {
                let new_parts: Vec<v2t::AssistantPart> = content
                    .iter()
                    .filter(|part| !matches!(part, v2t::AssistantPart::ToolCall(_)))
                    .cloned()
                    .collect();
                if new_parts.is_empty() {
                    mutated = true;
                    continue;
                }
                filtered.push(v2t::PromptMessage::Assistant {
                    content: new_parts,
                    provider_options: provider_options.clone(),
                });
            }
        }
    }

    let warning = if mutated {
        Some(v2t::CallWarning::UnsupportedSetting {
            setting: "toolContent".into(),
            details: Some(
                "Tool calls and results removed because no tools were provided for Amazon Bedrock."
                    .into(),
            ),
        })
    } else {
        None
    };

    (filtered, warning)
}

struct PreparedTools {
    tool_config: Option<JsonValue>,
    additional_model_fields: Option<JsonMap<String, JsonValue>>,
    betas: HashSet<String>,
    tool_warnings: Vec<v2t::CallWarning>,
    has_tools: bool,
}

fn prepare_tools(
    tools: &[v2t::FunctionTool],
    tool_choice: &Option<v2t::ToolChoice>,
) -> PreparedTools {
    if tools.is_empty() {
        return PreparedTools {
            tool_config: None,
            additional_model_fields: None,
            betas: HashSet::new(),
            tool_warnings: Vec::new(),
            has_tools: false,
        };
    }

    let mut tool_specs: Vec<JsonValue> = Vec::new();
    for tool in tools {
        let schema = if tool.input_schema.is_null() {
            JsonValue::Object(JsonMap::new())
        } else {
            JsonValue::Object(JsonMap::from_iter([(
                "json".into(),
                tool.input_schema.clone(),
            )]))
        };
        tool_specs.push(json!({
            "toolSpec": {
                "name": tool.name,
                "description": tool.description.clone(),
                "inputSchema": schema,
            }
        }));
    }

    let tool_choice_json = match tool_choice {
        None => None,
        Some(v2t::ToolChoice::Auto) => Some(json!({ "auto": {} })),
        Some(v2t::ToolChoice::Required) => Some(json!({ "any": {} })),
        Some(v2t::ToolChoice::None) => None,
        Some(v2t::ToolChoice::Tool { name }) => Some(json!({ "tool": { "name": name } })),
    };

    if matches!(tool_choice, Some(v2t::ToolChoice::None)) {
        return PreparedTools {
            tool_config: None,
            additional_model_fields: None,
            betas: HashSet::new(),
            tool_warnings: Vec::new(),
            has_tools: false,
        };
    }

    let mut obj = JsonMap::new();
    obj.insert("tools".into(), JsonValue::Array(tool_specs));
    if let Some(tc) = tool_choice_json {
        obj.insert("toolChoice".into(), tc);
    }

    PreparedTools {
        tool_config: Some(JsonValue::Object(obj)),
        additional_model_fields: None,
        betas: HashSet::new(),
        tool_warnings: Vec::new(),
        has_tools: true,
    }
}

#[derive(Debug, Deserialize)]
struct ConverseResponse {
    #[allow(dead_code)]
    #[serde(default)]
    metrics: Option<JsonValue>,
    output: ConverseOutput,
    #[serde(default, rename = "stopReason")]
    stop_reason: Option<String>,
    #[serde(default)]
    trace: Option<JsonValue>,
    #[serde(default)]
    usage: Option<ConverseUsage>,
}

#[derive(Debug, Deserialize)]
struct ConverseOutput {
    message: ConverseMessage,
}

#[derive(Debug, Deserialize)]
struct ConverseMessage {
    content: Vec<ConverseContent>,
    #[allow(dead_code)]
    role: String,
}

#[derive(Debug, Deserialize)]
struct ConverseContent {
    #[serde(default)]
    text: Option<String>,
    #[serde(default, rename = "toolUse")]
    tool_use: Option<ConverseToolUse>,
    #[serde(default, rename = "reasoningContent")]
    reasoning_content: Option<ReasoningContent>,
}

#[derive(Debug, Deserialize)]
struct ConverseToolUse {
    #[serde(rename = "toolUseId")]
    tool_use_id: Option<String>,
    name: Option<String>,
    input: Option<JsonValue>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ReasoningContent {
    Text {
        #[serde(rename = "reasoningText")]
        reasoning_text: ReasoningText,
    },
    Redacted {
        #[serde(rename = "redactedReasoning")]
        redacted_reasoning: RedactedReasoning,
    },
}

#[derive(Debug, Deserialize)]
struct ReasoningText {
    text: String,
    #[serde(default)]
    signature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RedactedReasoning {
    data: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ConverseUsage {
    #[serde(rename = "inputTokens")]
    input_tokens: u64,
    #[serde(rename = "outputTokens")]
    output_tokens: u64,
    #[serde(rename = "totalTokens")]
    total_tokens: u64,
    #[serde(rename = "cacheReadInputTokens")]
    cache_read_input_tokens: Option<u64>,
    #[serde(rename = "cacheWriteInputTokens")]
    cache_write_input_tokens: Option<u64>,
}

fn map_response_content(
    parts: &[ConverseContent],
    uses_json_response_tool: bool,
) -> Result<Vec<v2t::Content>, SdkError> {
    let mut out: Vec<v2t::Content> = Vec::new();
    for part in parts {
        if let Some(text) = part.text.as_ref() {
            if uses_json_response_tool {
                // JSON responses come through tool calls; ignore plain text when JSON tool is active
            } else if !text.is_empty() {
                out.push(v2t::Content::Text {
                    text: text.clone(),
                    provider_metadata: None,
                });
            }
        }

        if let Some(reasoning) = part.reasoning_content.as_ref() {
            match reasoning {
                ReasoningContent::Text { reasoning_text } => {
                    let mut provider_metadata = None;
                    if let Some(sig) = reasoning_text.signature.as_ref() {
                        let mut inner = JsonMap::new();
                        inner.insert("signature".into(), JsonValue::String(sig.clone()));
                        let mut outer = HashMap::new();
                        outer.insert(
                            "bedrock".into(),
                            inner.into_iter().collect::<HashMap<_, _>>(),
                        );
                        provider_metadata = Some(outer);
                    }
                    out.push(v2t::Content::Reasoning {
                        text: reasoning_text.text.clone(),
                        provider_metadata,
                    });
                }
                ReasoningContent::Redacted { redacted_reasoning } => {
                    let mut meta_map = JsonMap::new();
                    if let Some(data) = redacted_reasoning.data.as_ref() {
                        meta_map.insert("redactedData".into(), JsonValue::String(data.clone()));
                    }
                    let mut outer = HashMap::new();
                    outer.insert(
                        "bedrock".into(),
                        meta_map.into_iter().collect::<HashMap<_, _>>(),
                    );
                    out.push(v2t::Content::Reasoning {
                        text: String::new(),
                        provider_metadata: Some(outer),
                    });
                }
            }
        }

        if let Some(tool) = part.tool_use.as_ref() {
            let id = tool
                .tool_use_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let name = tool.name.clone().unwrap_or_else(|| format!("tool-{}", id));
            let input = tool.input.clone().unwrap_or(JsonValue::Null);
            if uses_json_response_tool {
                out.push(v2t::Content::Text {
                    text: input.to_string(),
                    provider_metadata: None,
                });
            } else {
                out.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                    tool_call_id: id,
                    tool_name: name,
                    input: input.to_string(),
                    provider_executed: false,
                    provider_metadata: None,
                    dynamic: false,
                    provider_options: None,
                }));
            }
        }
    }
    Ok(out)
}

fn map_finish_reason(reason: Option<&str>) -> v2t::FinishReason {
    match reason {
        Some("stop") | Some("stop_sequence") | Some("end_turn") => v2t::FinishReason::Stop,
        Some("max_tokens") | Some("length") => v2t::FinishReason::Length,
        Some("content_filtered") | Some("content-filter") | Some("guardrail_intervened") => {
            v2t::FinishReason::ContentFilter
        }
        Some("tool_use") | Some("tool-calls") => v2t::FinishReason::ToolCalls,
        Some("error") => v2t::FinishReason::Error,
        None => v2t::FinishReason::Unknown,
        _ => v2t::FinishReason::Unknown,
    }
}

fn map_usage(usage: Option<&ConverseUsage>) -> v2t::Usage {
    usage
        .map(|u| v2t::Usage {
            input_tokens: Some(u.input_tokens as u64),
            output_tokens: Some(u.output_tokens as u64),
            total_tokens: Some(u.total_tokens as u64),
            cached_input_tokens: u.cache_read_input_tokens.map(|n| n as u64),
            reasoning_tokens: None,
        })
        .unwrap_or_default()
}

fn build_provider_metadata(
    seed: Option<JsonMap<String, JsonValue>>,
    trace: Option<JsonValue>,
    usage: Option<&ConverseUsage>,
    uses_json_response_tool: bool,
) -> Option<v2t::ProviderMetadata> {
    if seed.is_none() && trace.is_none() && usage.is_none() && !uses_json_response_tool {
        return None;
    }
    let mut inner = seed.unwrap_or_default();
    if let Some(trace) = trace {
        inner.insert("trace".into(), trace);
    }
    if let Some(usage) = usage {
        if let Some(cache) = usage.cache_write_input_tokens {
            let mut usage_map = JsonMap::new();
            usage_map.insert(
                "cacheWriteInputTokens".into(),
                JsonValue::Number(cache.into()),
            );
            inner.insert("usage".into(), JsonValue::Object(usage_map));
        }
    }
    if uses_json_response_tool {
        inner.insert("isJsonResponseFromTool".into(), JsonValue::Bool(true));
    }
    let mut out = HashMap::new();
    let converted: HashMap<String, JsonValue> = inner.into_iter().collect();
    out.insert("bedrock".into(), converted);
    Some(out)
}

fn headers_to_map(headers: &[(String, String)]) -> v2t::Headers {
    headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}
