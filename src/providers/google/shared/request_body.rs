use serde_json::{json, Map, Value as JsonValue};

use crate::core::SdkError;
use crate::types::v2 as v2t;

use super::options::{
    parse_google_provider_options_for_scopes, GoogleProviderOptions, ThinkingConfig,
};
use super::prepare_tools::{convert_json_schema_to_openapi_schema, prepare_tools, PreparedTools};
use super::prompt::{convert_to_google_prompt_with_scopes, GooglePrompt};

const RESERVED_PROVIDER_OPTION_KEYS: &[&str] = &[
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
];

pub struct GoogleRequestBodyBuildConfig<'a> {
    pub scope_names: &'a [&'a str],
    pub raw_provider_option_keys: &'a [&'a str],
    pub model_id: &'a str,
    pub is_gemma: bool,
    pub trace_prefix: &'a str,
    pub include_thoughts_warning: Option<&'a str>,
}

pub fn build_google_request_body(
    config: GoogleRequestBodyBuildConfig<'_>,
    options: &v2t::CallOptions,
) -> Result<(JsonValue, Vec<v2t::CallWarning>), SdkError> {
    let mut warnings = Vec::new();
    let google_opts =
        parse_google_provider_options_for_scopes(&options.provider_options, config.scope_names);
    maybe_warn_for_include_thoughts(
        &mut warnings,
        google_opts.as_ref(),
        config.include_thoughts_warning,
    );

    let prompt =
        convert_to_google_prompt_with_scopes(&options.prompt, config.is_gemma, config.scope_names)?;
    let PreparedTools {
        tools,
        tool_config,
        tool_warnings,
    } = prepare_tools(&options.tools, &options.tool_choice, config.model_id);
    warnings.extend(tool_warnings);

    let (generation_config, threshold_override, request_body_overrides) = build_generation_config(
        options,
        google_opts.as_ref(),
        config.raw_provider_option_keys,
    );

    let mut body = build_request_body(config.is_gemma, prompt, generation_config);
    apply_google_body_options(&mut body, google_opts.as_ref(), threshold_override);
    attach_prepared_tools(&mut body, tools, tool_config);
    merge_request_body_overrides(&mut body, request_body_overrides, config.trace_prefix);

    Ok((body, warnings))
}

fn maybe_warn_for_include_thoughts(
    warnings: &mut Vec<v2t::CallWarning>,
    google_opts: Option<&GoogleProviderOptions>,
    include_thoughts_warning: Option<&str>,
) {
    let Some(message) = include_thoughts_warning else {
        return;
    };
    if matches!(
        google_opts.and_then(|opts| opts.thinking_config.as_ref()),
        Some(ThinkingConfig {
            include_thoughts: Some(true),
            ..
        })
    ) {
        warnings.push(v2t::CallWarning::Other {
            message: message.to_string(),
        });
    }
}

fn build_generation_config(
    options: &v2t::CallOptions,
    google_opts: Option<&GoogleProviderOptions>,
    raw_provider_option_keys: &[&str],
) -> (JsonValue, Option<JsonValue>, Option<Map<String, JsonValue>>) {
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

    let (response_mime_type, response_schema) = response_format_config(&options.response_format);
    if let Some(response_mime_type) = response_mime_type {
        generation_config["responseMimeType"] = json!(response_mime_type);
    }
    if let Some(response_schema) = response_schema {
        generation_config["responseSchema"] = response_schema;
    }

    if let Some(google_opts) = google_opts {
        if let Some(value) = &google_opts.response_modalities {
            generation_config["responseModalities"] = json!(value);
        }
        if let Some(value) = &google_opts.thinking_config {
            generation_config["thinkingConfig"] =
                serde_json::to_value(value).unwrap_or(JsonValue::Null);
        }
        if let Some(value) = &google_opts.audio_timestamp {
            generation_config["audioTimestamp"] = json!(value);
        }
    }

    let (threshold_override, request_body_overrides) = extract_provider_overrides(
        &options.provider_options,
        raw_provider_option_keys,
        &mut generation_config,
    );

    (
        generation_config,
        threshold_override,
        request_body_overrides,
    )
}

fn response_format_config(
    response_format: &Option<v2t::ResponseFormat>,
) -> (Option<String>, Option<JsonValue>) {
    match response_format {
        Some(v2t::ResponseFormat::Json { schema, .. }) => {
            let response_schema = schema.as_ref().and_then(|schema| {
                let converted = convert_json_schema_to_openapi_schema(schema);
                (!converted.is_null()).then_some(converted)
            });
            (Some("application/json".to_string()), response_schema)
        }
        _ => (None, None),
    }
}

fn extract_provider_overrides(
    provider_options: &v2t::ProviderOptions,
    raw_provider_option_keys: &[&str],
    generation_config: &mut JsonValue,
) -> (Option<JsonValue>, Option<Map<String, JsonValue>>) {
    let Some(raw_google_opts) = raw_provider_option_keys
        .iter()
        .find_map(|key| provider_options.get(*key))
    else {
        return (None, None);
    };

    let mut extras = raw_google_opts.clone();
    if let Some(generation_config_override) = extras
        .remove("generationConfig")
        .or_else(|| extras.remove("generation_config"))
    {
        crate::core::options::deep_merge(generation_config, &generation_config_override);
    }

    for key in RESERVED_PROVIDER_OPTION_KEYS {
        extras.remove(*key);
    }

    let threshold_override = extras.remove("threshold");
    if extras.is_empty() {
        (threshold_override, None)
    } else {
        let mut request_body_overrides = Map::new();
        for (key, value) in extras {
            request_body_overrides.insert(key, value);
        }
        (threshold_override, Some(request_body_overrides))
    }
}

fn build_request_body(
    is_gemma: bool,
    GooglePrompt {
        system_instruction,
        contents,
    }: GooglePrompt,
    generation_config: JsonValue,
) -> JsonValue {
    let mut body = json!({
        "generationConfig": generation_config,
        "contents": contents,
    });
    if !is_gemma {
        body["systemInstruction"] =
            serde_json::to_value(system_instruction).unwrap_or(JsonValue::Null);
    }
    body
}

fn apply_google_body_options(
    body: &mut JsonValue,
    google_opts: Option<&GoogleProviderOptions>,
    threshold_override: Option<JsonValue>,
) {
    if let Some(google_opts) = google_opts {
        if let Some(value) = google_opts.threshold.clone() {
            body["threshold"] = json!(value);
        }
        if let Some(value) = &google_opts.safety_settings {
            body["safetySettings"] = json!(value);
        }
        if let Some(value) = &google_opts.cached_content {
            body["cachedContent"] = json!(value);
        }
        if let Some(value) = &google_opts.labels {
            body["labels"] = json!(value);
        }
    }
    if let Some(threshold_override) = threshold_override {
        body["threshold"] = threshold_override;
    }
}

fn attach_prepared_tools(
    body: &mut JsonValue,
    tools: Option<JsonValue>,
    tool_config: Option<JsonValue>,
) {
    if let Some(tools) = tools {
        body["tools"] = tools;
    }
    if let Some(tool_config) = tool_config {
        body["toolConfig"] = tool_config;
    }
}

fn merge_request_body_overrides(
    body: &mut JsonValue,
    request_body_overrides: Option<Map<String, JsonValue>>,
    trace_prefix: &str,
) {
    let Some(request_body_overrides) = request_body_overrides else {
        return;
    };
    if !request_body_overrides.is_empty() {
        tracing::info!(
            "{}: applying provider request overrides {:?}",
            trace_prefix,
            request_body_overrides.keys().collect::<Vec<_>>()
        );
    }
    crate::core::options::merge_options_with_disallow(
        body,
        &JsonValue::Object(request_body_overrides),
        &["contents", "systemInstruction"],
    );
}
