use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;

use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_core::options::merge_options_with_disallow;
use crate::ai_sdk_core::request_builder::defaults::request_overrides_from_json;
use crate::ai_sdk_core::transport::{HttpTransport, TransportConfig};
use crate::ai_sdk_core::{
    map_events_to_parts, EventMapperConfig, EventMapperHooks, EventMapperState, GenerateResponse,
    LanguageModel, StreamResponse,
};
use crate::ai_sdk_streaming_sse::{PipelineBuilder, ProviderChunk, SseEvent};
use crate::ai_sdk_types::v2 as v2t;
use crate::ai_sdk_types::{Event, TokenUsage};
use base64::Engine;
use futures_core::Stream;
use futures_util::StreamExt;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use serde_json::{json, Map, Value};
use url::Url;
use uuid::Uuid;

use crate::provider_openai::config::OpenAIConfig;
use crate::provider_openai::error::map_transport_error;

type EventStream = Pin<Box<dyn Stream<Item = Result<Event, SdkError>> + Send>>;

pub struct OpenAIResponsesLanguageModel<
    T: HttpTransport = crate::reqwest_transport::ReqwestTransport,
> {
    pub model_id: String,
    pub config: OpenAIConfig,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
}

impl Default for OpenAIResponsesLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    fn default() -> Self {
        let cfg = TransportConfig::default();
        Self {
            model_id: String::new(),
            config: OpenAIConfig {
                provider_name: "openai.responses".into(),
                provider_scope_name: "openai".into(),
                base_url: "https://api.openai.com/v1".into(),
                endpoint_path: "/responses".into(),
                headers: vec![],
                query_params: vec![],
                supported_urls: HashMap::new(),
                file_id_prefixes: Some(vec!["file-".into()]),
                default_options: None,
                request_defaults: None,
            },
            http: crate::reqwest_transport::ReqwestTransport::new(&cfg),
            transport_cfg: cfg,
            limiter: None,
        }
    }
}

impl<T: HttpTransport> OpenAIResponsesLanguageModel<T> {
    pub fn new(
        model_id: impl Into<String>,
        config: OpenAIConfig,
        http: T,
        transport_cfg: TransportConfig,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            config,
            http,
            transport_cfg,
            limiter: None,
        }
    }

    pub fn with_rate_limit_per_sec(mut self, rps: u32) -> Self {
        if let Some(nz) = NonZeroU32::new(rps) {
            let q = Quota::per_second(nz);
            self.limiter = Some(Arc::new(RateLimiter::direct(q)));
        }
        self
    }

    fn endpoint_url(&self) -> String {
        self.config.endpoint_url()
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

    async fn send(
        &self,
        body: serde_json::Value,
    ) -> Result<
        (
            Pin<Box<dyn Stream<Item = Result<bytes::Bytes, SdkError>> + Send>>,
            Vec<(String, String)>,
        ),
        SdkError,
    > {
        let mut url = self.endpoint_url();
        if should_use_codex_oauth_websocket_transport(&self.config.endpoint_path) {
            url = to_websocket_url(&url)?;
        }

        // Merge and lowercase headers, skipping internal SDK headers.
        let mut hdrs: BTreeMap<String, String> = BTreeMap::new();
        hdrs.insert("content-type".into(), "application/json".into());
        hdrs.insert("accept".into(), "application/json".into());
        for (k, v) in &self.config.headers {
            if crate::ai_sdk_core::options::is_internal_sdk_header(k) {
                continue;
            }
            hdrs.insert(k.to_lowercase(), v.clone());
        }
        let headers: Vec<(String, String)> = hdrs
            .into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect();

        if let Some(l) = &self.limiter {
            let _ = l.until_ready().await;
        }

        // network debug prints removed

        match self
            .http
            .post_json_stream(&url, &headers, &body, &self.transport_cfg)
            .await
        {
            Ok(resp) => {
                let (stream, res_headers) = <T as HttpTransport>::into_stream(resp);
                let mapped = stream.map(|chunk_res| {
                    chunk_res.map_err(|te| match te {
                        TransportError::IdleReadTimeout(_) => SdkError::Timeout,
                        TransportError::ConnectTimeout(_) => SdkError::Timeout,
                        other => SdkError::Transport(other),
                    })
                });
                Ok((Box::pin(mapped), res_headers))
            }
            Err(te) => Err(map_transport_error(te)),
        }
    }
}

fn should_use_codex_oauth_websocket_transport(endpoint_path: &str) -> bool {
    endpoint_path
        .trim()
        .trim_end_matches('/')
        .eq_ignore_ascii_case("/backend-api/codex/responses")
}

fn to_websocket_url(url: &str) -> Result<String, SdkError> {
    let mut parsed = Url::parse(url).map_err(|err| SdkError::InvalidArgument {
        message: format!("invalid endpoint url '{url}': {err}"),
    })?;
    let new_scheme = match parsed.scheme() {
        "https" => "wss",
        "http" => "ws",
        "wss" | "ws" => return Ok(parsed.into()),
        scheme => {
            return Err(SdkError::InvalidArgument {
                message: format!("unsupported endpoint scheme '{scheme}' for websocket stream"),
            });
        }
    };
    parsed
        .set_scheme(new_scheme)
        .map_err(|_| SdkError::InvalidArgument {
            message: format!("failed to convert endpoint scheme to '{new_scheme}'"),
        })?;
    Ok(parsed.into())
}

// Convenience constructor for default reqwest transport
impl OpenAIResponsesLanguageModel<crate::reqwest_transport::ReqwestTransport> {
    pub fn create_simple(
        model_id: impl Into<String>,
        base_url: Option<String>,
        api_key: String,
    ) -> Self {
        let mut cfg = TransportConfig::default();
        cfg.idle_read_timeout = std::time::Duration::from_secs(45);
        let http = crate::reqwest_transport::ReqwestTransport::new(&cfg);
        let mut headers: Vec<(String, String)> = vec![
            ("content-type".into(), "application/json".into()),
            ("accept".into(), "application/json".into()),
        ];
        if !api_key.is_empty() {
            headers.push(("authorization".into(), format!("Bearer {}", api_key)));
        }
        let config = OpenAIConfig {
            provider_name: "openai.responses".into(),
            provider_scope_name: "openai".into(),
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".into()),
            endpoint_path: "/responses".into(),
            headers,
            query_params: vec![],
            supported_urls: HashMap::from([
                ("image/*".to_string(), vec![r"^https?://.*$".to_string()]),
                (
                    "application/pdf".to_string(),
                    vec![r"^https?://.*$".to_string()],
                ),
            ]),
            file_id_prefixes: Some(vec!["file-".into()]),
            default_options: None,
            request_defaults: None,
        };
        OpenAIResponsesLanguageModel::new(model_id, config, http, cfg)
    }
}

#[async_trait::async_trait]
impl<T: HttpTransport + Send + Sync + 'static> LanguageModel for OpenAIResponsesLanguageModel<T> {
    fn provider_name(&self) -> &'static str {
        "OpenAI"
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn supported_urls(&self) -> HashMap<String, Vec<String>> {
        self.config.supported_urls.clone()
    }

    async fn do_generate(&self, options: v2t::CallOptions) -> Result<GenerateResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        let tool_name_mapping = build_tool_name_mapping(&options.tools);
        let (body, warnings) = build_request_body(&options, &self.model_id, &self.config)?;
        // Use non-streaming JSON call to Responses API
        let url = self.endpoint_url();
        let mut hdrs: BTreeMap<String, String> = BTreeMap::new();
        hdrs.insert("content-type".into(), "application/json".into());
        hdrs.insert("accept".into(), "application/json".into());
        for (k, v) in &self.config.headers {
            if crate::ai_sdk_core::options::is_internal_sdk_header(k) {
                continue;
            }
            hdrs.insert(k.to_lowercase(), v.clone());
        }
        let headers: Vec<(String, String)> = hdrs
            .into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect();
        let (json, _res_headers) = self
            .http
            .post_json(&url, &headers, &body, &self.transport_cfg)
            .await
            .map_err(map_transport_error)?;

        if let Some(error) = json.get("error").filter(|v| !v.is_null()) {
            let message = error
                .get("message")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| error.to_string());
            return Err(SdkError::Upstream {
                status: 400,
                message,
                source: None,
            });
        }

        let approval_request_id_map = extract_approval_request_id_to_tool_call_id(
            &options.prompt,
            &self.config.provider_scope_name,
        );
        let (content, has_function_calls) =
            extract_response_content(&json, &tool_name_mapping, &approval_request_id_map);

        // Usage best-effort
        let mut usage = v2t::Usage::default();
        let usage_val = json
            .get("usage")
            .or_else(|| json.get("response").and_then(|r| r.get("usage")));
        if let Some(u) = usage_val.and_then(parse_openai_usage) {
            usage.input_tokens = Some(u.input_tokens as u64);
            usage.output_tokens = Some(u.output_tokens as u64);
            usage.total_tokens = Some(u.total_tokens as u64);
            if let Some(v) = u.cache_read_tokens {
                usage.cached_input_tokens = Some(v as u64);
            }
        }
        if let Some(raw_usage) = usage_val {
            apply_openai_usage_details(raw_usage, &mut usage);
        }

        // Finish reason mapping
        let finish_hint = json
            .get("incomplete_details")
            .and_then(|v| v.get("reason"))
            .and_then(|v| v.as_str());
        let finish_reason = map_finish_reason(finish_hint, has_function_calls);

        // Provider metadata: responseId and serviceTier
        let mut provider_metadata: Option<
            std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
        > = None;
        let resp_id = json.get("id").and_then(|v| v.as_str()).or_else(|| {
            json.get("response")
                .and_then(|r| r.get("id"))
                .and_then(|v| v.as_str())
        });
        let tier = json
            .get("service_tier")
            .and_then(|v| v.as_str())
            .or_else(|| {
                json.get("response")
                    .and_then(|r| r.get("service_tier"))
                    .and_then(|v| v.as_str())
            });
        if resp_id.is_some() || tier.is_some() {
            let mut outer = std::collections::HashMap::new();
            let mut inner = std::collections::HashMap::new();
            if let Some(rid) = resp_id {
                inner.insert("responseId".into(), serde_json::json!(rid));
            }
            if let Some(st) = tier {
                inner.insert("serviceTier".into(), serde_json::json!(st));
            }
            outer.insert("openai".into(), inner);
            provider_metadata = Some(outer);
        }

        Ok(GenerateResponse {
            content,
            finish_reason,
            usage,
            provider_metadata,
            request_body: None,
            response_headers: None,
            response_body: Some(json),
            warnings,
        })
    }

    async fn do_stream(&self, options: v2t::CallOptions) -> Result<StreamResponse, SdkError> {
        let options = crate::ai_sdk_core::request_builder::defaults::build_call_options(
            options,
            &self.config.provider_scope_name,
            self.config.default_options.as_ref(),
        );
        let prov = parse_openai_provider_options(
            &options.provider_options,
            &self.config.provider_scope_name,
        );
        let tool_name_mapping = build_tool_name_mapping(&options.tools);
        let (mut body, warnings) = build_request_body(&options, &self.model_id, &self.config)?;
        body["stream"] = Value::Bool(true);
        let store_for_stream = prov.store.unwrap_or(false);
        let logprobs_enabled =
            prov.logprobs_bool.unwrap_or(false) || prov.logprobs_n.unwrap_or(0) > 0;
        let approval_request_id_map = extract_approval_request_id_to_tool_call_id(
            &options.prompt,
            &self.config.provider_scope_name,
        );
        let stream = self
            .stream_with_body(body, options.include_raw_chunks)
            .await?;
        let parts = map_events_to_parts(
            stream,
            build_stream_mapper_config(
                warnings,
                tool_name_mapping,
                approval_request_id_map,
                store_for_stream,
                logprobs_enabled,
            ),
        );
        Ok(StreamResponse {
            stream: parts,
            request_body: None,
            response_headers: None,
        })
    }
}

// ----- Helpers: request building and SSE mapping -----

fn parse_openai_usage(u: &serde_json::Value) -> Option<TokenUsage> {
    let obj = u.as_object()?;
    let input = obj
        .get("input_tokens")
        .or_else(|| obj.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let output = obj
        .get("output_tokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let total = obj
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(input + output);
    let cache_read_tokens = obj
        .get("cache_read_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| parse_openai_cached_input_tokens(u).map(|v| v as usize));
    let cache_write_tokens = obj
        .get("cache_write_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| {
            obj.get("cache_creation")
                .and_then(|cc| cc.as_object())
                .and_then(|co| {
                    let a = co
                        .get("ephemeral_5m_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let b = co
                        .get("ephemeral_1h_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let sum = a + b;
                    if sum > 0 {
                        Some(sum as usize)
                    } else {
                        None
                    }
                })
        });
    Some(TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        cache_read_tokens,
        cache_write_tokens,
    })
}

fn parse_openai_cached_input_tokens(u: &serde_json::Value) -> Option<u64> {
    u.get("input_tokens_details")
        .and_then(|v| v.get("cached_tokens"))
        .or_else(|| {
            u.get("prompt_tokens_details")
                .and_then(|v| v.get("cached_tokens"))
        })
        .and_then(|v| v.as_u64())
}

fn parse_openai_reasoning_tokens(u: &serde_json::Value) -> Option<u64> {
    u.get("output_tokens_details")
        .and_then(|v| v.get("reasoning_tokens"))
        .or_else(|| {
            u.get("completion_tokens_details")
                .and_then(|v| v.get("reasoning_tokens"))
        })
        .and_then(|v| v.as_u64())
}

fn apply_openai_usage_details(u: &serde_json::Value, usage: &mut v2t::Usage) {
    if let Some(cached) = parse_openai_cached_input_tokens(u) {
        usage.cached_input_tokens = Some(cached);
    }
    if let Some(reasoning) = parse_openai_reasoning_tokens(u) {
        usage.reasoning_tokens = Some(reasoning);
    }
}

// Ensure tool schemas always have a top-level "type":"object".
fn normalize_object_schema(schema: &serde_json::Value) -> serde_json::Value {
    match schema {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|v| v.as_str()).is_some() {
                return schema.clone();
            }
            if map.contains_key("properties") || !map.is_empty() {
                let mut out = serde_json::Map::new();
                out.insert("type".into(), serde_json::Value::String("object".into()));
                if let Some(props) = map.get("properties") {
                    out.insert("properties".into(), props.clone());
                } else {
                    out.insert("properties".into(), serde_json::Value::Object(map.clone()));
                }
                if let Some(req) = map.get("required") {
                    out.insert("required".into(), req.clone());
                }
                return serde_json::Value::Object(out);
            }
            json!({"type":"object"})
        }
        _ => json!({"type":"object"}),
    }
}

fn invalid_tool_args(tool: &v2t::ProviderTool, message: impl Into<String>) -> SdkError {
    SdkError::InvalidArgument {
        message: format!(
            "provider tool {} ({}): {}",
            tool.name,
            tool.id,
            message.into()
        ),
    }
}

fn require_args_object(tool: &v2t::ProviderTool) -> Result<&Map<String, Value>, SdkError> {
    tool.args
        .as_object()
        .ok_or_else(|| invalid_tool_args(tool, "args must be an object"))
}

fn require_field<'a>(
    tool: &v2t::ProviderTool,
    args: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a Value, SdkError> {
    args.get(key)
        .ok_or_else(|| invalid_tool_args(tool, format!("args.{key} is required")))
}

fn expect_string(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
    if value.as_str().is_some() {
        Ok(())
    } else {
        Err(invalid_tool_args(tool, format!("{path} must be a string")))
    }
}

fn expect_bool(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
    if value.as_bool().is_some() {
        Ok(())
    } else {
        Err(invalid_tool_args(tool, format!("{path} must be a boolean")))
    }
}

fn expect_number(tool: &v2t::ProviderTool, value: &Value, path: &str) -> Result<(), SdkError> {
    if value.as_f64().is_some() {
        Ok(())
    } else {
        Err(invalid_tool_args(tool, format!("{path} must be a number")))
    }
}

fn expect_int_range(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
    min: i64,
    max: i64,
) -> Result<(), SdkError> {
    let raw = value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
        .or_else(|| {
            value.as_f64().filter(|v| v.fract() == 0.0).and_then(|v| {
                if v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                    Some(v as i64)
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an integer")))?;
    if raw < min || raw > max {
        return Err(invalid_tool_args(
            tool,
            format!("{path} must be between {min} and {max}"),
        ));
    }
    Ok(())
}

fn expect_enum(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
    allowed: &[&str],
) -> Result<(), SdkError> {
    match value.as_str() {
        Some(val) if allowed.contains(&val) => Ok(()),
        Some(_) => Err(invalid_tool_args(
            tool,
            format!("{path} must be one of {}", allowed.join(", ")),
        )),
        None => Err(invalid_tool_args(
            tool,
            format!("{path} must be one of {}", allowed.join(", ")),
        )),
    }
}

fn expect_string_array(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    let arr = value
        .as_array()
        .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an array")))?;
    if arr.iter().all(|item| item.as_str().is_some()) {
        Ok(())
    } else {
        Err(invalid_tool_args(
            tool,
            format!("{path} must be an array of strings"),
        ))
    }
}

fn expect_object<'a>(
    tool: &v2t::ProviderTool,
    value: &'a Value,
    path: &str,
) -> Result<&'a Map<String, Value>, SdkError> {
    value
        .as_object()
        .ok_or_else(|| invalid_tool_args(tool, format!("{path} must be an object")))
}

fn ensure_known_keys(
    tool: &v2t::ProviderTool,
    args: &Map<String, Value>,
    allowed: &[&str],
) -> Result<(), SdkError> {
    let mut unknown = args
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    Err(invalid_tool_args(
        tool,
        format!("args contains unsupported keys: {}", unknown.join(", ")),
    ))
}

fn validate_user_location(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    let obj = expect_object(tool, value, path)?;
    let loc_type = obj
        .get("type")
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.type must be \"approximate\"")))?;
    expect_enum(tool, loc_type, &format!("{path}.type"), &["approximate"])?;
    for key in ["country", "city", "region", "timezone"] {
        if let Some(val) = obj.get(key) {
            expect_string(tool, val, &format!("{path}.{key}"))?;
        }
    }
    Ok(())
}

fn validate_file_search_filter(
    tool: &v2t::ProviderTool,
    value: &Value,
    path: &str,
) -> Result<(), SdkError> {
    let obj = expect_object(tool, value, path)?;
    let filter_type = obj
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| invalid_tool_args(tool, format!("{path}.type is required")))?;
    match filter_type {
        "and" | "or" => {
            let filters = obj
                .get("filters")
                .ok_or_else(|| invalid_tool_args(tool, format!("{path}.filters is required")))?;
            let arr = filters.as_array().ok_or_else(|| {
                invalid_tool_args(tool, format!("{path}.filters must be an array"))
            })?;
            for (idx, entry) in arr.iter().enumerate() {
                validate_file_search_filter(tool, entry, &format!("{path}.filters[{idx}]"))?;
            }
            Ok(())
        }
        "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "in" | "nin" => {
            let key = obj
                .get("key")
                .ok_or_else(|| invalid_tool_args(tool, format!("{path}.key is required")))?;
            expect_string(tool, key, &format!("{path}.key"))?;
            let val = obj
                .get("value")
                .ok_or_else(|| invalid_tool_args(tool, format!("{path}.value is required")))?;
            if val.as_str().is_some() || val.as_bool().is_some() || val.as_f64().is_some() {
                return Ok(());
            }
            if let Some(arr) = val.as_array() {
                if arr.iter().all(|item| item.as_str().is_some()) {
                    return Ok(());
                }
            }
            Err(invalid_tool_args(
                tool,
                format!("{path}.value must be a string, number, boolean, or array of strings"),
            ))
        }
        _ => Err(invalid_tool_args(
            tool,
            format!("{path}.type has an invalid value"),
        )),
    }
}

fn validate_openai_provider_tool_args(
    tool_type: &str,
    tool: &v2t::ProviderTool,
) -> Result<(), SdkError> {
    match tool_type {
        "file_search" => {
            let args = require_args_object(tool)?;
            let ids = require_field(tool, args, "vectorStoreIds")?;
            expect_string_array(tool, ids, "args.vectorStoreIds")?;
            if let Some(max) = args.get("maxNumResults") {
                expect_number(tool, max, "args.maxNumResults")?;
            }
            if let Some(rank) = args.get("ranking") {
                let rank_obj = expect_object(tool, rank, "args.ranking")?;
                if let Some(ranker) = rank_obj.get("ranker") {
                    expect_string(tool, ranker, "args.ranking.ranker")?;
                }
                if let Some(score) = rank_obj.get("scoreThreshold") {
                    expect_number(tool, score, "args.ranking.scoreThreshold")?;
                }
            }
            if let Some(filters) = args.get("filters") {
                validate_file_search_filter(tool, filters, "args.filters")?;
            }
            Ok(())
        }
        "web_search_preview" => {
            let args = require_args_object(tool)?;
            if let Some(size) = args.get("searchContextSize") {
                expect_enum(
                    tool,
                    size,
                    "args.searchContextSize",
                    &["low", "medium", "high"],
                )?;
            }
            if let Some(loc) = args.get("userLocation") {
                validate_user_location(tool, loc, "args.userLocation")?;
            }
            Ok(())
        }
        "web_search" => {
            let args = require_args_object(tool)?;
            if let Some(access) = args.get("externalWebAccess") {
                expect_bool(tool, access, "args.externalWebAccess")?;
            }
            if let Some(filters) = args.get("filters") {
                let obj = expect_object(tool, filters, "args.filters")?;
                if let Some(domains) = obj.get("allowedDomains") {
                    expect_string_array(tool, domains, "args.filters.allowedDomains")?;
                }
            }
            if let Some(size) = args.get("searchContextSize") {
                expect_enum(
                    tool,
                    size,
                    "args.searchContextSize",
                    &["low", "medium", "high"],
                )?;
            }
            if let Some(loc) = args.get("userLocation") {
                validate_user_location(tool, loc, "args.userLocation")?;
            }
            Ok(())
        }
        "code_interpreter" => {
            let args = require_args_object(tool)?;
            if let Some(container) = args.get("container") {
                if let Some(obj) = container.as_object() {
                    if let Some(file_ids) = obj.get("fileIds") {
                        expect_string_array(tool, file_ids, "args.container.fileIds")?;
                    }
                } else if container.as_str().is_none() {
                    return Err(invalid_tool_args(
                        tool,
                        "args.container must be a string or object",
                    ));
                }
            }
            Ok(())
        }
        "image_generation" => {
            let args = require_args_object(tool)?;
            ensure_known_keys(
                tool,
                args,
                &[
                    "background",
                    "inputFidelity",
                    "inputImageMask",
                    "model",
                    "moderation",
                    "outputCompression",
                    "outputFormat",
                    "partialImages",
                    "quality",
                    "size",
                ],
            )?;
            if let Some(background) = args.get("background") {
                expect_enum(
                    tool,
                    background,
                    "args.background",
                    &["auto", "opaque", "transparent"],
                )?;
            }
            if let Some(fidelity) = args.get("inputFidelity") {
                expect_enum(tool, fidelity, "args.inputFidelity", &["low", "high"])?;
            }
            if let Some(mask) = args.get("inputImageMask") {
                let mask_obj = expect_object(tool, mask, "args.inputImageMask")?;
                if let Some(file_id) = mask_obj.get("fileId") {
                    expect_string(tool, file_id, "args.inputImageMask.fileId")?;
                }
                if let Some(image_url) = mask_obj.get("imageUrl") {
                    expect_string(tool, image_url, "args.inputImageMask.imageUrl")?;
                }
            }
            if let Some(model) = args.get("model") {
                expect_string(tool, model, "args.model")?;
            }
            if let Some(moderation) = args.get("moderation") {
                expect_enum(tool, moderation, "args.moderation", &["auto"])?;
            }
            if let Some(output_compression) = args.get("outputCompression") {
                expect_int_range(tool, output_compression, "args.outputCompression", 0, 100)?;
            }
            if let Some(output_format) = args.get("outputFormat") {
                expect_enum(
                    tool,
                    output_format,
                    "args.outputFormat",
                    &["png", "jpeg", "webp"],
                )?;
            }
            if let Some(partial_images) = args.get("partialImages") {
                expect_int_range(tool, partial_images, "args.partialImages", 0, 3)?;
            }
            if let Some(quality) = args.get("quality") {
                expect_enum(
                    tool,
                    quality,
                    "args.quality",
                    &["auto", "low", "medium", "high"],
                )?;
            }
            if let Some(size) = args.get("size") {
                expect_enum(
                    tool,
                    size,
                    "args.size",
                    &["1024x1024", "1024x1536", "1536x1024", "auto"],
                )?;
            }
            Ok(())
        }
        "mcp" => {
            let args = require_args_object(tool)?;
            let server_label = require_field(tool, args, "serverLabel")?;
            expect_string(tool, server_label, "args.serverLabel")?;
            let server_url = args.get("serverUrl");
            let connector_id = args.get("connectorId");
            if let Some(url) = server_url {
                expect_string(tool, url, "args.serverUrl")?;
            }
            if let Some(connector) = connector_id {
                expect_string(tool, connector, "args.connectorId")?;
            }
            if server_url.is_none() && connector_id.is_none() {
                return Err(invalid_tool_args(
                    tool,
                    "args.serverUrl or args.connectorId is required",
                ));
            }
            if let Some(allowed) = args.get("allowedTools") {
                if let Some(arr) = allowed.as_array() {
                    for (idx, entry) in arr.iter().enumerate() {
                        expect_string(tool, entry, &format!("args.allowedTools[{idx}]"))?;
                    }
                } else if let Some(obj) = allowed.as_object() {
                    if let Some(read_only) = obj.get("readOnly") {
                        expect_bool(tool, read_only, "args.allowedTools.readOnly")?;
                    }
                    if let Some(tool_names) = obj.get("toolNames") {
                        expect_string_array(tool, tool_names, "args.allowedTools.toolNames")?;
                    }
                } else {
                    return Err(invalid_tool_args(
                        tool,
                        "args.allowedTools must be an array or object",
                    ));
                }
            }
            if let Some(authorization) = args.get("authorization") {
                expect_string(tool, authorization, "args.authorization")?;
            }
            if let Some(headers) = args.get("headers") {
                let obj = expect_object(tool, headers, "args.headers")?;
                for (key, val) in obj {
                    expect_string(tool, val, &format!("args.headers.{key}"))?;
                }
            }
            if let Some(require_approval) = args.get("requireApproval") {
                if let Some(val) = require_approval.as_str() {
                    if val != "always" && val != "never" {
                        return Err(invalid_tool_args(
                            tool,
                            "args.requireApproval must be \"always\" or \"never\"",
                        ));
                    }
                } else if let Some(obj) = require_approval.as_object() {
                    if let Some(never) = obj.get("never") {
                        let never_obj = expect_object(tool, never, "args.requireApproval.never")?;
                        if let Some(tool_names) = never_obj.get("toolNames") {
                            expect_string_array(
                                tool,
                                tool_names,
                                "args.requireApproval.never.toolNames",
                            )?;
                        }
                    }
                } else {
                    return Err(invalid_tool_args(
                        tool,
                        "args.requireApproval must be a string or object",
                    ));
                }
            }
            if let Some(server_description) = args.get("serverDescription") {
                expect_string(tool, server_description, "args.serverDescription")?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[derive(Clone, Default)]
struct ToolNameMapping {
    custom_to_provider: HashMap<String, String>,
    provider_to_custom: HashMap<String, String>,
    web_search_tool_name: Option<String>,
}

impl ToolNameMapping {
    fn to_provider_tool_name<'a>(&'a self, custom_tool_name: &'a str) -> &'a str {
        self.custom_to_provider
            .get(custom_tool_name)
            .map(|s| s.as_str())
            .unwrap_or(custom_tool_name)
    }

    fn to_custom_tool_name<'a>(&'a self, provider_tool_name: &'a str) -> &'a str {
        self.provider_to_custom
            .get(provider_tool_name)
            .map(|s| s.as_str())
            .unwrap_or(provider_tool_name)
    }
}

fn openai_provider_tool_name(id: &str) -> Option<&'static str> {
    match id {
        "openai.file_search" => Some("file_search"),
        "openai.local_shell" => Some("local_shell"),
        "openai.shell" => Some("shell"),
        "openai.apply_patch" => Some("apply_patch"),
        "openai.web_search_preview" => Some("web_search_preview"),
        "openai.web_search" => Some("web_search"),
        "openai.code_interpreter" => Some("code_interpreter"),
        "openai.image_generation" => Some("image_generation"),
        "openai.mcp" => Some("mcp"),
        _ => None,
    }
}

fn is_openai_builtin_tool(name: &str) -> bool {
    matches!(
        name,
        "file_search"
            | "local_shell"
            | "shell"
            | "apply_patch"
            | "web_search_preview"
            | "web_search"
            | "code_interpreter"
            | "image_generation"
            | "mcp"
    )
}

fn build_tool_name_mapping(tools: &[v2t::Tool]) -> ToolNameMapping {
    let mut mapping = ToolNameMapping::default();
    for tool in tools {
        if let v2t::Tool::Provider(provider_tool) = tool {
            if let Some(provider_name) = openai_provider_tool_name(&provider_tool.id) {
                mapping
                    .custom_to_provider
                    .insert(provider_tool.name.clone(), provider_name.to_string());
                mapping
                    .provider_to_custom
                    .insert(provider_name.to_string(), provider_tool.name.clone());
                if matches!(
                    provider_tool.id.as_str(),
                    "openai.web_search" | "openai.web_search_preview"
                ) && mapping.web_search_tool_name.is_none()
                {
                    mapping.web_search_tool_name = Some(provider_tool.name.clone());
                }
            }
        }
    }
    mapping
}

fn extract_approval_request_id_to_tool_call_id(
    prompt: &[v2t::PromptMessage],
    provider_scope_name: &str,
) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    for message in prompt {
        if let v2t::PromptMessage::Assistant { content, .. } = message {
            for part in content {
                if let v2t::AssistantPart::ToolCall(call) = part {
                    if let Some(opts) = call.provider_options.as_ref() {
                        if let Some(scope) = opts.get(provider_scope_name) {
                            let approval_id =
                                scope.get("approvalRequestId").and_then(|v| v.as_str());
                            if let Some(approval_id) = approval_id {
                                mapping.insert(approval_id.to_string(), call.tool_call_id.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    mapping
}

fn map_web_search_output(action: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = action.as_object()?;
    let action_type = obj.get("type")?.as_str()?;
    match action_type {
        "search" => {
            let mut out = serde_json::Map::new();
            let mut inner = serde_json::Map::new();
            inner.insert("type".into(), serde_json::Value::String("search".into()));
            if let Some(query) = obj.get("query") {
                if !query.is_null() {
                    inner.insert("query".into(), query.clone());
                }
            }
            out.insert("action".into(), serde_json::Value::Object(inner));
            if let Some(sources) = obj.get("sources") {
                if !sources.is_null() {
                    out.insert("sources".into(), sources.clone());
                }
            }
            Some(serde_json::Value::Object(out))
        }
        "open_page" => Some(json!({
            "action": {
                "type": "openPage",
                "url": obj.get("url").cloned().unwrap_or(serde_json::Value::Null),
            }
        })),
        "find_in_page" => Some(json!({
            "action": {
                "type": "findInPage",
                "url": obj.get("url").cloned().unwrap_or(serde_json::Value::Null),
                "pattern": obj.get("pattern").cloned().unwrap_or(serde_json::Value::Null),
            }
        })),
        _ => None,
    }
}

fn build_openai_provider_tool(
    tool: &v2t::ProviderTool,
) -> Result<Option<serde_json::Value>, SdkError> {
    let empty = serde_json::Map::new();
    let tool_type = match openai_provider_tool_name(&tool.id) {
        Some(tool_type) => tool_type,
        None => return Ok(None),
    };
    validate_openai_provider_tool_args(tool_type, tool)?;
    let args = tool.args.as_object().unwrap_or(&empty);
    let val = match tool_type {
        "file_search" => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("file_search"));
            if let Some(ids) = args.get("vectorStoreIds") {
                obj.insert("vector_store_ids".into(), ids.clone());
            }
            if let Some(max) = args.get("maxNumResults") {
                obj.insert("max_num_results".into(), max.clone());
            }
            if let Some(rank) = args.get("ranking").and_then(|v| v.as_object()) {
                let mut opts = serde_json::Map::new();
                if let Some(ranker) = rank.get("ranker") {
                    opts.insert("ranker".into(), ranker.clone());
                }
                if let Some(score) = rank.get("scoreThreshold") {
                    opts.insert("score_threshold".into(), score.clone());
                }
                if !opts.is_empty() {
                    obj.insert("ranking_options".into(), serde_json::Value::Object(opts));
                }
            }
            if let Some(filters) = args.get("filters") {
                obj.insert("filters".into(), filters.clone());
            }
            Some(serde_json::Value::Object(obj))
        }
        "local_shell" => Some(json!({"type":"local_shell"})),
        "shell" => Some(json!({"type":"shell"})),
        "apply_patch" => Some(json!({"type":"apply_patch"})),
        "web_search_preview" => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("web_search_preview"));
            if let Some(size) = args.get("searchContextSize") {
                obj.insert("search_context_size".into(), size.clone());
            }
            if let Some(loc) = args.get("userLocation") {
                obj.insert("user_location".into(), loc.clone());
            }
            Some(serde_json::Value::Object(obj))
        }
        "web_search" => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("web_search"));
            if let Some(filters) = args.get("filters").and_then(|v| v.as_object()) {
                if let Some(allowed_domains) = filters.get("allowedDomains") {
                    obj.insert(
                        "filters".into(),
                        json!({"allowed_domains": allowed_domains}),
                    );
                }
            }
            if let Some(access) = args.get("externalWebAccess") {
                obj.insert("external_web_access".into(), access.clone());
            }
            if let Some(size) = args.get("searchContextSize") {
                obj.insert("search_context_size".into(), size.clone());
            }
            if let Some(loc) = args.get("userLocation") {
                obj.insert("user_location".into(), loc.clone());
            }
            Some(serde_json::Value::Object(obj))
        }
        "code_interpreter" => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("code_interpreter"));
            let container = match args.get("container") {
                None | Some(serde_json::Value::Null) => json!({"type":"auto"}),
                Some(serde_json::Value::String(val)) => json!(val),
                Some(serde_json::Value::Object(map)) => {
                    let mut c = serde_json::Map::new();
                    c.insert("type".into(), json!("auto"));
                    if let Some(file_ids) = map.get("fileIds") {
                        c.insert("file_ids".into(), file_ids.clone());
                    }
                    serde_json::Value::Object(c)
                }
                Some(other) => other.clone(),
            };
            obj.insert("container".into(), container);
            Some(serde_json::Value::Object(obj))
        }
        "image_generation" => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("image_generation"));
            for (src, dst) in [
                ("background", "background"),
                ("inputFidelity", "input_fidelity"),
                ("model", "model"),
                ("moderation", "moderation"),
                ("partialImages", "partial_images"),
                ("quality", "quality"),
                ("outputCompression", "output_compression"),
                ("outputFormat", "output_format"),
                ("size", "size"),
            ] {
                if let Some(val) = args.get(src) {
                    obj.insert(dst.into(), val.clone());
                }
            }
            if let Some(mask) = args.get("inputImageMask").and_then(|v| v.as_object()) {
                let mut mask_obj = serde_json::Map::new();
                if let Some(file_id) = mask.get("fileId") {
                    mask_obj.insert("file_id".into(), file_id.clone());
                }
                if let Some(image_url) = mask.get("imageUrl") {
                    mask_obj.insert("image_url".into(), image_url.clone());
                }
                if !mask_obj.is_empty() {
                    obj.insert(
                        "input_image_mask".into(),
                        serde_json::Value::Object(mask_obj),
                    );
                }
            }
            Some(serde_json::Value::Object(obj))
        }
        "mcp" => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("mcp"));
            if let Some(val) = args.get("serverLabel") {
                obj.insert("server_label".into(), val.clone());
            }
            if let Some(val) = args.get("authorization") {
                obj.insert("authorization".into(), val.clone());
            }
            if let Some(val) = args.get("connectorId") {
                obj.insert("connector_id".into(), val.clone());
            }
            if let Some(val) = args.get("headers") {
                obj.insert("headers".into(), val.clone());
            }
            if let Some(val) = args.get("serverDescription") {
                obj.insert("server_description".into(), val.clone());
            }
            if let Some(val) = args.get("serverUrl") {
                obj.insert("server_url".into(), val.clone());
            }
            if let Some(allowed) = args.get("allowedTools") {
                if let Some(list) = allowed.as_array() {
                    obj.insert(
                        "allowed_tools".into(),
                        serde_json::Value::Array(list.clone()),
                    );
                } else if let Some(filter) = allowed.as_object() {
                    let mut allowed_obj = serde_json::Map::new();
                    if let Some(read_only) = filter.get("readOnly") {
                        allowed_obj.insert("read_only".into(), read_only.clone());
                    }
                    if let Some(tool_names) = filter.get("toolNames") {
                        allowed_obj.insert("tool_names".into(), tool_names.clone());
                    }
                    if !allowed_obj.is_empty() {
                        obj.insert(
                            "allowed_tools".into(),
                            serde_json::Value::Object(allowed_obj),
                        );
                    }
                }
            }
            let require_approval = match args.get("requireApproval") {
                None | Some(serde_json::Value::Null) => None,
                Some(serde_json::Value::String(val)) => {
                    Some(serde_json::Value::String(val.clone()))
                }
                Some(serde_json::Value::Object(map)) => map.get("never").map(|never| {
                    if let Some(filter) = never.as_object() {
                        let mut filter_obj = serde_json::Map::new();
                        if let Some(tool_names) = filter.get("toolNames") {
                            filter_obj.insert("tool_names".into(), tool_names.clone());
                        }
                        json!({"never": filter_obj})
                    } else {
                        json!({"never": {}})
                    }
                }),
                Some(other) => Some(other.clone()),
            };
            obj.insert(
                "require_approval".into(),
                require_approval.unwrap_or_else(|| json!("never")),
            );
            Some(serde_json::Value::Object(obj))
        }
        _ => None,
    };
    Ok(val)
}

fn provider_tool_data_from_output_item(
    item: &serde_json::Map<String, Value>,
) -> Option<serde_json::Value> {
    let item_type = item.get("type")?.as_str()?;
    let item_id = item
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    match item_type {
        "web_search_call" => {
            let tool_call_id = item_id.clone()?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("web_search"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            obj.insert("input".into(), json!({}));
            if let Some(action) = item.get("action").and_then(map_web_search_output) {
                obj.insert("result".into(), action);
            }
            Some(serde_json::Value::Object(obj))
        }
        "file_search_call" => {
            let tool_call_id = item_id.clone()?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("file_search"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            obj.insert("input".into(), json!({}));
            let results_val = item.get("results").and_then(|v| v.as_array()).map(|arr| {
                arr.iter()
                    .filter_map(|entry| entry.as_object())
                    .map(|entry| {
                        let mut mapped = serde_json::Map::new();
                        if let Some(attributes) = entry.get("attributes") {
                            mapped.insert("attributes".into(), attributes.clone());
                        }
                        if let Some(file_id) = entry.get("file_id") {
                            mapped.insert("fileId".into(), file_id.clone());
                        }
                        if let Some(filename) = entry.get("filename") {
                            mapped.insert("filename".into(), filename.clone());
                        }
                        if let Some(score) = entry.get("score") {
                            mapped.insert("score".into(), score.clone());
                        }
                        if let Some(text) = entry.get("text") {
                            mapped.insert("text".into(), text.clone());
                        }
                        serde_json::Value::Object(mapped)
                    })
                    .collect::<Vec<_>>()
            });
            let result = json!({
                "queries": item.get("queries").cloned().unwrap_or(serde_json::Value::Null),
                "results": results_val.map(serde_json::Value::Array).unwrap_or(serde_json::Value::Null),
            });
            obj.insert("result".into(), result);
            Some(serde_json::Value::Object(obj))
        }
        "code_interpreter_call" => {
            let tool_call_id = item_id.clone()?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("code_interpreter"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            let input = json!({
                "code": item.get("code").cloned().unwrap_or(serde_json::Value::Null),
                "containerId": item.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
            });
            obj.insert("input".into(), input);
            let result = json!({
                "outputs": item.get("outputs").cloned().unwrap_or(serde_json::Value::Null),
            });
            obj.insert("result".into(), result);
            Some(serde_json::Value::Object(obj))
        }
        "image_generation_call" => {
            let tool_call_id = item_id.clone()?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("image_generation"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            obj.insert("input".into(), json!({}));
            let result = json!({
                "result": item.get("result").cloned().unwrap_or(serde_json::Value::Null),
            });
            obj.insert("result".into(), result);
            Some(serde_json::Value::Object(obj))
        }
        "computer_call" => {
            let tool_call_id = item_id.clone()?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("computer_use"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            obj.insert("input".into(), json!(""));
            let status = item
                .get("status")
                .cloned()
                .unwrap_or_else(|| json!("completed"));
            obj.insert(
                "result".into(),
                json!({
                    "type": "computer_use_tool_result",
                    "status": status,
                }),
            );
            Some(serde_json::Value::Object(obj))
        }
        "local_shell_call" => {
            let tool_call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| item_id.clone())?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("local_shell"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(false));
            let action = item.get("action").and_then(|v| v.as_object());
            let mut action_obj = serde_json::Map::new();
            if let Some(action) = action {
                if let Some(command) = action.get("command") {
                    action_obj.insert("command".into(), command.clone());
                }
                if let Some(timeout) = action.get("timeout_ms") {
                    action_obj.insert("timeoutMs".into(), timeout.clone());
                }
                if let Some(user) = action.get("user") {
                    action_obj.insert("user".into(), user.clone());
                }
                if let Some(dir) = action.get("working_directory") {
                    action_obj.insert("workingDirectory".into(), dir.clone());
                }
                if let Some(env) = action.get("env") {
                    action_obj.insert("env".into(), env.clone());
                }
            }
            if !action_obj.is_empty() {
                obj.insert("input".into(), json!({ "action": action_obj }));
            } else {
                obj.insert("input".into(), json!({}));
            }
            Some(serde_json::Value::Object(obj))
        }
        "shell_call" => {
            let tool_call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| item_id.clone())?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("shell"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(false));
            let action = item.get("action").and_then(|v| v.as_object());
            let mut action_obj = serde_json::Map::new();
            if let Some(action) = action {
                if let Some(commands) = action.get("commands") {
                    action_obj.insert("commands".into(), commands.clone());
                }
                if let Some(timeout) = action.get("timeout_ms") {
                    action_obj.insert("timeoutMs".into(), timeout.clone());
                }
                if let Some(max_len) = action.get("max_output_length") {
                    action_obj.insert("maxOutputLength".into(), max_len.clone());
                }
            }
            if !action_obj.is_empty() {
                obj.insert("input".into(), json!({ "action": action_obj }));
            } else {
                obj.insert("input".into(), json!({}));
            }
            Some(serde_json::Value::Object(obj))
        }
        "apply_patch_call" => {
            let tool_call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| item_id.clone())?;
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("apply_patch"));
            obj.insert("tool_call_id".into(), json!(tool_call_id.clone()));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(false));
            let input = json!({
                "callId": tool_call_id,
                "operation": item.get("operation").cloned().unwrap_or(serde_json::Value::Null),
            });
            obj.insert("input".into(), input);
            Some(serde_json::Value::Object(obj))
        }
        "mcp_call" => {
            let tool_call_id = item_id.clone()?;
            let name = item.get("name").and_then(|v| v.as_str())?.to_string();
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("mcp"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            obj.insert(
                "input".into(),
                item.get("arguments").cloned().unwrap_or_else(|| json!("")),
            );
            obj.insert("mcp_name".into(), json!(name.clone()));
            if let Some(approval_request_id) =
                item.get("approval_request_id").and_then(|v| v.as_str())
            {
                obj.insert("approval_request_id".into(), json!(approval_request_id));
            }
            if let Some(server_label) = item.get("server_label") {
                obj.insert("server_label".into(), server_label.clone());
            }
            let mut result = serde_json::Map::new();
            result.insert("type".into(), json!("call"));
            if let Some(server_label) = item.get("server_label") {
                result.insert("serverLabel".into(), server_label.clone());
            }
            result.insert("name".into(), json!(name));
            result.insert(
                "arguments".into(),
                item.get("arguments").cloned().unwrap_or_else(|| json!("")),
            );
            if let Some(output) = item.get("output") {
                result.insert("output".into(), output.clone());
            }
            if let Some(error) = item.get("error") {
                result.insert("error".into(), error.clone());
            }
            obj.insert("result".into(), serde_json::Value::Object(result));
            Some(serde_json::Value::Object(obj))
        }
        "mcp_approval_request" => {
            let tool_call_id = item_id.clone()?;
            let name = item.get("name").and_then(|v| v.as_str())?.to_string();
            let mut obj = serde_json::Map::new();
            obj.insert("tool_type".into(), json!("mcp"));
            obj.insert("tool_call_id".into(), json!(tool_call_id));
            if let Some(id) = item_id.as_ref() {
                obj.insert("item_id".into(), json!(id));
            }
            obj.insert("provider_executed".into(), json!(true));
            obj.insert(
                "input".into(),
                item.get("arguments").cloned().unwrap_or_else(|| json!("")),
            );
            obj.insert("mcp_name".into(), json!(name));
            let approval_request_id = item
                .get("approval_request_id")
                .and_then(|v| v.as_str())
                .map(|v| v.to_string())
                .or_else(|| item_id.clone());
            if let Some(approval_request_id) = approval_request_id {
                obj.insert("approval_request_id".into(), json!(approval_request_id));
            }
            obj.insert("approval_request".into(), json!(true));
            if let Some(server_label) = item.get("server_label") {
                obj.insert("server_label".into(), server_label.clone());
            }
            Some(serde_json::Value::Object(obj))
        }
        _ => None,
    }
}

struct ProviderToolParts {
    tool_call_id: String,
    tool_name: String,
    tool_type: String,
    input: String,
    provider_executed: bool,
    dynamic: bool,
    result: Option<serde_json::Value>,
    is_error: bool,
    provider_metadata: Option<v2t::ProviderMetadata>,
    approval_request_id: Option<String>,
    is_approval_request: bool,
}

fn provider_tool_parts_from_data(
    data: &serde_json::Value,
    tool_name_mapping: &ToolNameMapping,
) -> Option<ProviderToolParts> {
    let obj = data.as_object()?;
    let tool_type = obj.get("tool_type").and_then(|v| v.as_str()).unwrap_or("");
    let tool_call_id = obj
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if tool_call_id.is_empty() {
        return None;
    }
    let input_val = obj.get("input").cloned().unwrap_or(serde_json::Value::Null);
    let input = match input_val {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    };
    let dynamic = tool_type == "mcp";
    let tool_name = if tool_type == "mcp" {
        obj.get("mcp_name")
            .and_then(|v| v.as_str())
            .map(|name| format!("mcp.{name}"))
            .unwrap_or_else(|| "mcp".into())
    } else if tool_type == "web_search" {
        tool_name_mapping
            .web_search_tool_name
            .clone()
            .unwrap_or_else(|| {
                tool_name_mapping
                    .to_custom_tool_name("web_search")
                    .to_string()
            })
    } else {
        tool_name_mapping.to_custom_tool_name(tool_type).to_string()
    };
    let provider_executed = obj
        .get("provider_executed")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let result = obj.get("result").cloned().filter(|v| !v.is_null());
    let is_error = obj
        .get("is_error")
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| {
            result
                .as_ref()
                .and_then(|val| val.get("error"))
                .map(|v| !v.is_null())
                .unwrap_or(false)
        });
    let provider_metadata = obj.get("item_id").and_then(|v| v.as_str()).map(|id| {
        let mut inner = HashMap::new();
        inner.insert("itemId".into(), serde_json::json!(id));
        let mut outer = HashMap::new();
        outer.insert("openai".into(), inner);
        outer
    });
    let approval_request_id = obj
        .get("approval_request_id")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let is_approval_request = obj
        .get("approval_request")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(ProviderToolParts {
        tool_call_id,
        tool_name,
        tool_type: tool_type.to_string(),
        input,
        provider_executed,
        dynamic,
        result,
        is_error,
        provider_metadata,
        approval_request_id,
        is_approval_request,
    })
}

fn escape_json_delta(delta: &str) -> String {
    if delta
        .as_bytes()
        .iter()
        .all(|b| *b >= 0x20 && *b != b'"' && *b != b'\\')
    {
        return delta.to_string();
    }
    let encoded = serde_json::to_string(delta).unwrap_or_else(|_| "\"\"".into());
    if encoded.len() >= 2 && encoded.starts_with('"') && encoded.ends_with('"') {
        encoded[1..encoded.len() - 1].to_string()
    } else {
        encoded
    }
}

#[derive(Default)]
pub(crate) struct OpenAIResponsesChunk {
    tool_calls: HashMap<usize, OpenAIToolCallState>,
    pending_deltas: HashMap<usize, Vec<String>>,
}

struct OpenAIToolCallState {
    id: String,
}

impl ProviderChunk for OpenAIResponsesChunk {
    fn try_from_sse(&mut self, event: &SseEvent) -> Result<Option<Vec<Event>>, SdkError> {
        let trimmed = std::str::from_utf8(&event.data).unwrap_or("").trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return Ok(None);
        }
        let json: serde_json::Value = match serde_json::from_slice(&event.data) {
            Ok(v) => v,
            Err(err) => {
                return Ok(Some(vec![Event::Error {
                    message: format!("Invalid JSON chunk: {}", err),
                }]));
            }
        };
        let t = match json.get("type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return Ok(Some(vec![Event::Error {
                    message: "Invalid chunk: missing type".into(),
                }]));
            }
        };
        let mut events = Vec::new();
        match t {
            "response.created" => {
                if let Some(resp) = json.get("response") {
                    let id = resp.get("id").cloned().unwrap_or(serde_json::Value::Null);
                    let model = resp
                        .get("model")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    let created_at = resp
                        .get("created_at")
                        .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)))
                        .map(|v| v * 1000);
                    events.push(Event::Data {
                        key: "openai.response_metadata".into(),
                        value: serde_json::json!({
                            "id": id,
                            "model": model,
                            "created_at": created_at,
                        }),
                    });
                }
            }
            "response.output_text.delta" => {
                let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                let item_id = json.get("item_id").and_then(|v| v.as_str());
                if let Some(item_id) = item_id {
                    if !delta.is_empty() {
                        let logprobs = json
                            .get("logprobs")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        events.push(Event::Data {
                            key: "openai.text_delta".into(),
                            value: serde_json::json!({
                                "item_id": item_id,
                                "delta": delta,
                                "logprobs": logprobs,
                            }),
                        });
                    }
                }
            }
            "response.output_text.annotation.added" => {
                let item_id = json.get("item_id").and_then(|v| v.as_str());
                let annotation = json
                    .get("annotation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if let Some(item_id) = item_id {
                    events.push(Event::Data {
                        key: "openai.text_annotation".into(),
                        value: serde_json::json!({
                            "item_id": item_id,
                            "annotation": annotation,
                        }),
                    });
                }
            }
            "response.output_item.added" => {
                if let Some(item) = json.get("item").and_then(|v| v.as_object()) {
                    if let Some(typ) = item.get("type").and_then(|v| v.as_str()) {
                        match typ {
                            "function_call" => {
                                let name = item
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let call_id = item
                                    .get("call_id")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());
                                let index = json
                                    .get("output_index")
                                    .and_then(|v| v.as_u64())
                                    .map(|n| n as usize);
                                if let (Some(cid), Some(id)) =
                                    (call_id.as_ref(), item.get("id").and_then(|v| v.as_str()))
                                {
                                    events.push(Event::Data {
                                        key: format!("openai.tool_item_id.{}", cid),
                                        value: serde_json::json!({"item_id": id}),
                                    });
                                }
                                if let (Some(idx), Some(cid), Some(tool_name)) =
                                    (index, call_id.clone(), name.clone())
                                {
                                    self.tool_calls
                                        .insert(idx, OpenAIToolCallState { id: cid.clone() });
                                    events.push(Event::ToolCallStart {
                                        id: cid.clone(),
                                        name: tool_name,
                                    });
                                    if let Some(pending) = self.pending_deltas.remove(&idx) {
                                        for delta in pending {
                                            events.push(Event::ToolCallDelta {
                                                id: cid.clone(),
                                                args_json: delta,
                                            });
                                        }
                                    }
                                }
                            }
                            "apply_patch_call" => {
                                let index = json
                                    .get("output_index")
                                    .and_then(|v| v.as_u64())
                                    .map(|n| n as usize);
                                let call_id = item.get("call_id").and_then(|v| v.as_str());
                                let operation = item
                                    .get("operation")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                if let (Some(idx), Some(call_id)) = (index, call_id) {
                                    events.push(Event::Data {
                                        key: "openai.apply_patch_call.added".into(),
                                        value: serde_json::json!({
                                            "output_index": idx,
                                            "call_id": call_id,
                                            "operation": operation,
                                        }),
                                    });
                                }
                            }
                            "message" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.message_added".into(),
                                        value: serde_json::json!({"item_id": id}),
                                    });
                                }
                            }
                            "reasoning" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    let enc = item
                                        .get("encrypted_content")
                                        .cloned()
                                        .unwrap_or(serde_json::Value::Null);
                                    events.push(Event::Data {
                                        key: "openai.reasoning_added".into(),
                                        value: serde_json::json!({
                                            "item_id": id,
                                            "encrypted_content": enc,
                                        }),
                                    });
                                }
                            }
                            "web_search_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.web_search_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            "file_search_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.file_search_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            "image_generation_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.image_generation_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            "code_interpreter_call" => {
                                let output_index = json
                                    .get("output_index")
                                    .and_then(|v| v.as_u64())
                                    .map(|n| n as usize);
                                if let (Some(id), Some(idx)) =
                                    (item.get("id").and_then(|v| v.as_str()), output_index)
                                {
                                    let container_id = item
                                        .get("container_id")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.to_string());
                                    events.push(Event::Data {
                                        key: "openai.code_interpreter_call.added".into(),
                                        value: serde_json::json!({
                                            "output_index": idx,
                                            "tool_call_id": id,
                                            "container_id": container_id,
                                        }),
                                    });
                                }
                            }
                            "computer_call" => {
                                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                    events.push(Event::Data {
                                        key: "openai.computer_call.added".into(),
                                        value: serde_json::json!({"tool_call_id": id}),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "response.output_item.done" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                if let Some(item) = json.get("item").and_then(|v| v.as_object()) {
                    if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                        if item_type == "apply_patch_call" {
                            if let Some(idx) = index {
                                let call_id = item.get("call_id").and_then(|v| v.as_str());
                                let operation = item
                                    .get("operation")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                events.push(Event::Data {
                                    key: "openai.apply_patch_call.done".into(),
                                    value: serde_json::json!({
                                        "output_index": idx,
                                        "call_id": call_id,
                                        "operation": operation,
                                    }),
                                });
                            }
                        }
                        if item_type == "message" {
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                events.push(Event::Data {
                                    key: "openai.message_done".into(),
                                    value: serde_json::json!({"item_id": id}),
                                });
                            }
                        }
                        if item_type == "reasoning" {
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                let enc = item
                                    .get("encrypted_content")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null);
                                events.push(Event::Data {
                                    key: "openai.reasoning_done".into(),
                                    value: serde_json::json!({
                                        "item_id": id,
                                        "encrypted_content": enc,
                                    }),
                                });
                            }
                        }
                        if item_type == "function_call" {
                            events.push(Event::Data {
                                key: "openai.function_call_done".into(),
                                value: serde_json::json!({}),
                            });
                        }
                    }
                    if let Some(tool_data) = provider_tool_data_from_output_item(item) {
                        events.push(Event::Data {
                            key: "openai.provider_tool".into(),
                            value: tool_data,
                        });
                    }
                }
                if let Some(idx) = index {
                    if let Some(state) = self.tool_calls.remove(&idx) {
                        if let Some(pending) = self.pending_deltas.remove(&idx) {
                            for delta in pending {
                                events.push(Event::ToolCallDelta {
                                    id: state.id.clone(),
                                    args_json: delta,
                                });
                            }
                        }
                        events.push(Event::ToolCallEnd { id: state.id });
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let delta = json
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(idx) = index {
                    if let Some(state) = self.tool_calls.get(&idx) {
                        events.push(Event::ToolCallDelta {
                            id: state.id.clone(),
                            args_json: delta,
                        });
                    } else {
                        self.pending_deltas.entry(idx).or_default().push(delta);
                    }
                }
            }
            "response.code_interpreter_call_code.delta" => {
                if let Some(idx) = json.get("output_index").and_then(|v| v.as_u64()) {
                    let delta = json
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !delta.is_empty() {
                        events.push(Event::Data {
                            key: "openai.code_interpreter_call.code_delta".into(),
                            value: serde_json::json!({
                                "output_index": idx as usize,
                                "delta": delta,
                            }),
                        });
                    }
                }
            }
            "response.code_interpreter_call_code.done" => {
                if let Some(idx) = json.get("output_index").and_then(|v| v.as_u64()) {
                    let code = json
                        .get("code")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    events.push(Event::Data {
                        key: "openai.code_interpreter_call.code_done".into(),
                        value: serde_json::json!({
                            "output_index": idx as usize,
                            "code": code,
                        }),
                    });
                }
            }
            "response.image_generation_call.partial_image" => {
                if let (Some(id), Some(b64)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("partial_image_b64").and_then(|v| v.as_str()),
                ) {
                    events.push(Event::Data {
                        key: "openai.image_generation_call.partial".into(),
                        value: serde_json::json!({
                            "tool_call_id": id,
                            "partial_image_b64": b64,
                        }),
                    });
                }
            }
            "response.apply_patch_call_operation_diff.delta" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let delta = json
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(idx) = index {
                    if !delta.is_empty() {
                        events.push(Event::Data {
                            key: "openai.apply_patch_call.diff.delta".into(),
                            value: serde_json::json!({
                                "output_index": idx,
                                "delta": delta,
                            }),
                        });
                    }
                }
            }
            "response.apply_patch_call_operation_diff.done" => {
                let index = json
                    .get("output_index")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
                let diff = json
                    .get("diff")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(idx) = index {
                    events.push(Event::Data {
                        key: "openai.apply_patch_call.diff.done".into(),
                        value: serde_json::json!({
                            "output_index": idx,
                            "diff": diff,
                        }),
                    });
                }
            }
            "response.reasoning_summary_part.added" => {
                if let (Some(id), Some(idx)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("summary_index").and_then(|v| v.as_u64()),
                ) {
                    events.push(Event::Data {
                        key: "openai.reasoning_summary_added".into(),
                        value: serde_json::json!({
                            "item_id": id,
                            "summary_index": idx,
                        }),
                    });
                }
            }
            "response.reasoning_summary_text.delta" => {
                if let (Some(id), Some(idx)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("summary_index").and_then(|v| v.as_u64()),
                ) {
                    let delta = json.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                    if !delta.is_empty() {
                        events.push(Event::Data {
                            key: "openai.reasoning_summary_delta".into(),
                            value: serde_json::json!({
                                "item_id": id,
                                "summary_index": idx,
                                "delta": delta,
                            }),
                        });
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                if let (Some(id), Some(idx)) = (
                    json.get("item_id").and_then(|v| v.as_str()),
                    json.get("summary_index").and_then(|v| v.as_u64()),
                ) {
                    events.push(Event::Data {
                        key: "openai.reasoning_summary_done".into(),
                        value: serde_json::json!({
                            "item_id": id,
                            "summary_index": idx,
                        }),
                    });
                }
            }
            "response.completed" | "response.incomplete" => {
                if let Some(resp) = json.get("response") {
                    if let Some(usage_val) = resp.get("usage") {
                        if let Some(usage) = parse_openai_usage(usage_val) {
                            events.push(Event::Usage { usage });
                        }
                        events.push(Event::Data {
                            key: "usage".into(),
                            value: usage_val.clone(),
                        });
                    }
                    let fin = resp
                        .get("incomplete_details")
                        .and_then(|v| v.get("reason"))
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    events.push(Event::Data {
                        key: "openai.finish".into(),
                        value: serde_json::json!({"incomplete_reason": fin}),
                    });
                    let rid = resp.get("id").cloned().unwrap_or(serde_json::Value::Null);
                    let st = resp
                        .get("service_tier")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    events.push(Event::Data {
                        key: "openai.response".into(),
                        value: serde_json::json!({"id": rid, "service_tier": st}),
                    });
                }
                if !self.tool_calls.is_empty() {
                    for (_idx, state) in self.tool_calls.drain() {
                        events.push(Event::ToolCallEnd { id: state.id });
                    }
                    self.pending_deltas.clear();
                }
                events.push(Event::Done);
            }
            "response.failed" => {
                // TS parity: failed chunks are not treated as "response finished" chunks.
                // They should not set incomplete_reason/service_tier driven finish metadata.
                let failed_payload = json
                    .get("response")
                    .map(|resp| {
                        serde_json::json!({
                            "id": resp.get("id").cloned().unwrap_or(serde_json::Value::Null),
                        })
                    })
                    .unwrap_or_else(|| serde_json::json!({}));
                events.push(Event::Data {
                    key: "openai.failed".into(),
                    value: failed_payload,
                });
                if !self.tool_calls.is_empty() {
                    for (_idx, state) in self.tool_calls.drain() {
                        events.push(Event::ToolCallEnd { id: state.id });
                    }
                    self.pending_deltas.clear();
                }
                events.push(Event::Done);
            }
            "error" => {
                events.push(Event::Data {
                    key: "openai.error".into(),
                    value: json.clone(),
                });
            }
            _ => {}
        }
        if events.is_empty() {
            Ok(None)
        } else {
            Ok(Some(events))
        }
    }
}

#[derive(Debug, Clone)]
struct OpenAIApplyPatchState {
    tool_call_id: String,
    operation_path: Option<String>,
    has_diff: bool,
    end_emitted: bool,
}

#[derive(Debug, Clone)]
struct OpenAICodeInterpreterState {
    tool_call_id: String,
    container_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningSummaryStatus {
    Active,
    CanConclude,
    Concluded,
}

#[derive(Debug, Clone, Default)]
struct OpenAIReasoningState {
    encrypted_content: Option<serde_json::Value>,
    summary_parts: HashMap<u32, ReasoningSummaryStatus>,
}

#[derive(Default)]
struct OpenAIStreamExtras {
    finish_hint: Option<String>,
    response_id: Option<String>,
    service_tier: Option<String>,
    saw_response_failed: bool,
    store: bool,
    logprobs_enabled: bool,
    has_function_calls: bool,
    logprobs: Vec<serde_json::Value>,
    message_annotations: HashMap<String, Vec<serde_json::Value>>,
    active_reasoning: HashMap<String, OpenAIReasoningState>,
    open_text_ids: HashSet<String>,
    open_tool_inputs: HashSet<String>,
    tool_item_ids: HashMap<String, String>, // call_id -> item_id
    approval_request_id_map: HashMap<String, String>,
    apply_patch_calls: HashMap<usize, OpenAIApplyPatchState>,
    code_interpreter_calls: HashMap<usize, OpenAICodeInterpreterState>,
    emitted_tool_calls: HashSet<String>,
    tool_name_mapping: ToolNameMapping,
}

fn openai_item_metadata(
    item_id: &str,
    extras: impl IntoIterator<Item = (String, serde_json::Value)>,
) -> v2t::ProviderMetadata {
    let mut inner = HashMap::new();
    inner.insert("itemId".into(), serde_json::json!(item_id));
    for (key, value) in extras {
        inner.insert(key, value);
    }
    let mut outer = HashMap::new();
    outer.insert("openai".into(), inner);
    outer
}

fn build_stream_mapper_config(
    warnings: Vec<v2t::CallWarning>,
    tool_name_mapping: ToolNameMapping,
    approval_request_id_map: HashMap<String, String>,
    store: bool,
    logprobs_enabled: bool,
) -> EventMapperConfig<OpenAIStreamExtras> {
    let mut hooks: EventMapperHooks<OpenAIStreamExtras> = EventMapperHooks::default();

    hooks.tool_end_metadata = Some(Box::new(
        |state: &mut EventMapperState<OpenAIStreamExtras>, id| {
            state
                .extra
                .tool_item_ids
                .get(id)
                .map(|iid| openai_item_metadata(iid, []))
        },
    ));

    hooks.data = Some(Box::new(
        |state: &mut EventMapperState<OpenAIStreamExtras>, key, value| {
            if key == "usage" {
                if let Some(usage) = parse_openai_usage(value) {
                    state.usage.input_tokens = Some(usage.input_tokens as u64);
                    state.usage.output_tokens = Some(usage.output_tokens as u64);
                    state.usage.total_tokens = Some(usage.total_tokens as u64);
                    state.usage.cached_input_tokens = usage.cache_read_tokens.map(|v| v as u64);
                }
                apply_openai_usage_details(value, &mut state.usage);
                return None;
            } else if key == "openai.response_metadata" {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let model_id = value
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let timestamp_ms = value
                    .get("created_at")
                    .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)));
                if let Some(rid) = id.as_ref() {
                    if state.extra.response_id.is_none() {
                        state.extra.response_id = Some(rid.clone());
                    }
                }
                let meta = v2t::ResponseMetadata {
                    id,
                    timestamp_ms,
                    model_id,
                };
                return Some(vec![v2t::StreamPart::ResponseMetadata { meta }]);
            } else if key == "openai.message_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                state
                    .extra
                    .message_annotations
                    .insert(item_id.to_string(), Vec::new());
                if state.extra.open_text_ids.insert(item_id.to_string()) {
                    let md = openai_item_metadata(item_id, []);
                    return Some(vec![v2t::StreamPart::TextStart {
                        id: item_id.to_string(),
                        provider_metadata: Some(md),
                    }]);
                }
            } else if key == "openai.text_delta" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                let mut out = Vec::new();
                if !state.extra.open_text_ids.contains(item_id) {
                    state.extra.open_text_ids.insert(item_id.to_string());
                    state
                        .extra
                        .message_annotations
                        .entry(item_id.to_string())
                        .or_default();
                    out.push(v2t::StreamPart::TextStart {
                        id: item_id.to_string(),
                        provider_metadata: Some(openai_item_metadata(item_id, [])),
                    });
                }
                if state.extra.logprobs_enabled {
                    if let Some(logprobs) = value.get("logprobs").filter(|v| !v.is_null()) {
                        state.extra.logprobs.push(logprobs.clone());
                    }
                }
                out.push(v2t::StreamPart::TextDelta {
                    id: item_id.to_string(),
                    delta: delta.to_string(),
                    provider_metadata: None,
                });
                return Some(out);
            } else if key == "openai.text_annotation" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let annotation = value.get("annotation")?.clone();
                state
                    .extra
                    .message_annotations
                    .entry(item_id.to_string())
                    .or_default()
                    .push(annotation.clone());
                let annotation_obj = annotation.as_object()?;
                let annotation_type = annotation_obj.get("type")?.as_str()?;
                let make_provider_metadata = |vals: Vec<(&str, serde_json::Value)>| {
                    let mut inner = HashMap::new();
                    for (key, val) in vals {
                        inner.insert(key.into(), val);
                    }
                    let mut outer = HashMap::new();
                    outer.insert("openai".into(), inner);
                    outer
                };
                match annotation_type {
                    "url_citation" => {
                        let url = annotation_obj.get("url")?.as_str()?;
                        let title = annotation_obj
                            .get("title")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: url.to_string(),
                            title,
                            provider_metadata: None,
                        }]);
                    }
                    "file_citation" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let title = annotation_obj
                            .get("quote")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| {
                                annotation_obj
                                    .get("filename")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .or_else(|| Some(file_id.to_string()));
                        let mut metadata_vals = vec![("fileId", json!(file_id))];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title,
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    "container_file_citation" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let container_id = annotation_obj.get("container_id")?.as_str()?;
                        let title = annotation_obj
                            .get("filename")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .or_else(|| Some(file_id.to_string()));
                        let mut metadata_vals = vec![
                            ("fileId", json!(file_id)),
                            ("containerId", json!(container_id)),
                        ];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title,
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    "file_path" => {
                        let file_id = annotation_obj.get("file_id")?.as_str()?;
                        let mut metadata_vals = vec![("fileId", json!(file_id))];
                        if let Some(index) = annotation_obj.get("index").filter(|v| !v.is_null()) {
                            metadata_vals.push(("index", index.clone()));
                        }
                        return Some(vec![v2t::StreamPart::SourceUrl {
                            id: Uuid::new_v4().to_string(),
                            url: file_id.to_string(),
                            title: Some(file_id.to_string()),
                            provider_metadata: Some(make_provider_metadata(metadata_vals)),
                        }]);
                    }
                    _ => {}
                }
            } else if key == "openai.message_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let annotations = state
                    .extra
                    .message_annotations
                    .remove(item_id)
                    .unwrap_or_default();
                state.extra.open_text_ids.remove(item_id);
                let md = if annotations.is_empty() {
                    openai_item_metadata(item_id, [])
                } else {
                    openai_item_metadata(
                        item_id,
                        [("annotations".into(), serde_json::Value::Array(annotations))],
                    )
                };
                return Some(vec![v2t::StreamPart::TextEnd {
                    id: item_id.to_string(),
                    provider_metadata: Some(md),
                }]);
            } else if key == "openai.error" {
                state.extra.finish_hint = Some("error".into());
                return Some(vec![v2t::StreamPart::Error {
                    error: value.clone(),
                }]);
            } else if key == "openai.reasoning_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let enc = value
                    .get("encrypted_content")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let mut state_entry = OpenAIReasoningState {
                    encrypted_content: Some(enc.clone()),
                    summary_parts: HashMap::new(),
                };
                state_entry
                    .summary_parts
                    .insert(0, ReasoningSummaryStatus::Active);
                state
                    .extra
                    .active_reasoning
                    .insert(item_id.to_string(), state_entry);
                return Some(vec![v2t::StreamPart::ReasoningStart {
                    id: format!("{item_id}:0"),
                    provider_metadata: Some(openai_item_metadata(
                        item_id,
                        [("reasoningEncryptedContent".into(), enc)],
                    )),
                }]);
            } else if key == "openai.reasoning_summary_added" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                if summary_index == 0 {
                    return None;
                }
                let reasoning_state = state.extra.active_reasoning.get_mut(item_id)?;
                let mut out = Vec::new();
                for (idx, status) in reasoning_state.summary_parts.iter_mut() {
                    if matches!(status, ReasoningSummaryStatus::CanConclude) {
                        out.push(v2t::StreamPart::ReasoningEnd {
                            id: format!("{item_id}:{idx}"),
                            provider_metadata: Some(openai_item_metadata(item_id, [])),
                        });
                        *status = ReasoningSummaryStatus::Concluded;
                    }
                }
                reasoning_state
                    .summary_parts
                    .insert(summary_index as u32, ReasoningSummaryStatus::Active);
                let enc = reasoning_state
                    .encrypted_content
                    .clone()
                    .unwrap_or(serde_json::Value::Null);
                out.push(v2t::StreamPart::ReasoningStart {
                    id: format!("{item_id}:{summary_index}"),
                    provider_metadata: Some(openai_item_metadata(
                        item_id,
                        [("reasoningEncryptedContent".into(), enc)],
                    )),
                });
                return Some(out);
            } else if key == "openai.reasoning_summary_delta" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                return Some(vec![v2t::StreamPart::ReasoningDelta {
                    id: format!("{item_id}:{summary_index}"),
                    delta: delta.to_string(),
                    provider_metadata: Some(openai_item_metadata(item_id, [])),
                }]);
            } else if key == "openai.reasoning_summary_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let summary_index = value.get("summary_index").and_then(|v| v.as_u64())?;
                if let Some(reasoning_state) = state.extra.active_reasoning.get_mut(item_id) {
                    if state.extra.store {
                        reasoning_state
                            .summary_parts
                            .insert(summary_index as u32, ReasoningSummaryStatus::Concluded);
                        return Some(vec![v2t::StreamPart::ReasoningEnd {
                            id: format!("{item_id}:{summary_index}"),
                            provider_metadata: Some(openai_item_metadata(item_id, [])),
                        }]);
                    }
                    reasoning_state
                        .summary_parts
                        .insert(summary_index as u32, ReasoningSummaryStatus::CanConclude);
                }
            } else if key == "openai.reasoning_done" {
                let item_id = value.get("item_id").and_then(|v| v.as_str())?;
                let enc = value
                    .get("encrypted_content")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                if let Some(reasoning_state) = state.extra.active_reasoning.remove(item_id) {
                    let md =
                        openai_item_metadata(item_id, [("reasoningEncryptedContent".into(), enc)]);
                    let mut out = Vec::new();
                    for (idx, status) in reasoning_state.summary_parts {
                        if matches!(
                            status,
                            ReasoningSummaryStatus::Active | ReasoningSummaryStatus::CanConclude
                        ) {
                            out.push(v2t::StreamPart::ReasoningEnd {
                                id: format!("{item_id}:{idx}"),
                                provider_metadata: Some(md.clone()),
                            });
                        }
                    }
                    if !out.is_empty() {
                        return Some(out);
                    }
                }
            } else if key == "openai.web_search_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .web_search_tool_name
                    .clone()
                    .unwrap_or_else(|| {
                        state
                            .extra
                            .tool_name_mapping
                            .to_custom_tool_name("web_search")
                            .to_string()
                    });
                state.has_tool_calls = true;
                state
                    .extra
                    .emitted_tool_calls
                    .insert(tool_call_id.to_string());
                return Some(vec![
                    v2t::StreamPart::ToolInputStart {
                        id: tool_call_id.to_string(),
                        tool_name: tool_name.clone(),
                        provider_executed: true,
                        provider_metadata: None,
                    },
                    v2t::StreamPart::ToolInputEnd {
                        id: tool_call_id.to_string(),
                        provider_executed: true,
                        provider_metadata: None,
                    },
                    v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                        tool_call_id: tool_call_id.to_string(),
                        tool_name,
                        input: "{}".into(),
                        provider_executed: true,
                        provider_metadata: None,
                        dynamic: false,
                        provider_options: None,
                    }),
                ]);
            } else if key == "openai.file_search_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("file_search")
                    .to_string();
                state.has_tool_calls = true;
                state
                    .extra
                    .emitted_tool_calls
                    .insert(tool_call_id.to_string());
                return Some(vec![v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name,
                    input: "{}".into(),
                    provider_executed: true,
                    provider_metadata: None,
                    dynamic: false,
                    provider_options: None,
                })]);
            } else if key == "openai.image_generation_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("image_generation")
                    .to_string();
                state.has_tool_calls = true;
                state
                    .extra
                    .emitted_tool_calls
                    .insert(tool_call_id.to_string());
                return Some(vec![v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name,
                    input: "{}".into(),
                    provider_executed: true,
                    provider_metadata: None,
                    dynamic: false,
                    provider_options: None,
                })]);
            } else if key == "openai.image_generation_call.partial" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let partial = value.get("partial_image_b64").and_then(|v| v.as_str())?;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("image_generation")
                    .to_string();
                return Some(vec![v2t::StreamPart::ToolResult {
                    tool_call_id: tool_call_id.to_string(),
                    tool_name,
                    result: json!({ "result": partial }),
                    is_error: false,
                    preliminary: true,
                    provider_metadata: None,
                }]);
            } else if key == "openai.code_interpreter_call.added" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                let container_id = value
                    .get("container_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                state.extra.code_interpreter_calls.insert(
                    output_index,
                    OpenAICodeInterpreterState {
                        tool_call_id: tool_call_id.to_string(),
                        container_id: container_id.clone(),
                    },
                );
                state.has_tool_calls = true;
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("code_interpreter")
                    .to_string();
                let cid = container_id.unwrap_or_default();
                return Some(vec![
                    v2t::StreamPart::ToolInputStart {
                        id: tool_call_id.to_string(),
                        tool_name,
                        provider_executed: true,
                        provider_metadata: None,
                    },
                    v2t::StreamPart::ToolInputDelta {
                        id: tool_call_id.to_string(),
                        delta: format!(
                            "{{\"containerId\":\"{}\",\"code\":\"",
                            escape_json_delta(&cid)
                        ),
                        provider_executed: true,
                        provider_metadata: None,
                    },
                ]);
            } else if key == "openai.code_interpreter_call.code_delta" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if delta.is_empty() {
                    return None;
                }
                if let Some(call_state) = state.extra.code_interpreter_calls.get(&output_index) {
                    return Some(vec![v2t::StreamPart::ToolInputDelta {
                        id: call_state.tool_call_id.clone(),
                        delta: escape_json_delta(delta),
                        provider_executed: true,
                        provider_metadata: None,
                    }]);
                }
            } else if key == "openai.code_interpreter_call.code_done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let code = value.get("code").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(call_state) = state.extra.code_interpreter_calls.remove(&output_index) {
                    let mut out = Vec::new();
                    out.push(v2t::StreamPart::ToolInputDelta {
                        id: call_state.tool_call_id.clone(),
                        delta: "\"}".into(),
                        provider_executed: true,
                        provider_metadata: None,
                    });
                    out.push(v2t::StreamPart::ToolInputEnd {
                        id: call_state.tool_call_id.clone(),
                        provider_executed: true,
                        provider_metadata: None,
                    });
                    let input = json!({
                        "code": code,
                        "containerId": call_state.container_id,
                    })
                    .to_string();
                    out.push(v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                        tool_call_id: call_state.tool_call_id.clone(),
                        tool_name: state
                            .extra
                            .tool_name_mapping
                            .to_custom_tool_name("code_interpreter")
                            .to_string(),
                        input,
                        provider_executed: true,
                        provider_metadata: None,
                        dynamic: false,
                        provider_options: None,
                    }));
                    state
                        .extra
                        .emitted_tool_calls
                        .insert(call_state.tool_call_id);
                    return Some(out);
                }
            } else if key == "openai.computer_call.added" {
                let tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str())?;
                state
                    .extra
                    .open_tool_inputs
                    .insert(tool_call_id.to_string());
                state.has_tool_calls = true;
                return Some(vec![v2t::StreamPart::ToolInputStart {
                    id: tool_call_id.to_string(),
                    tool_name: state
                        .extra
                        .tool_name_mapping
                        .to_custom_tool_name("computer_use")
                        .to_string(),
                    provider_executed: true,
                    provider_metadata: None,
                }]);
            } else if key == "openai.apply_patch_call.added" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let call_id = value.get("call_id").and_then(|v| v.as_str())?;
                let operation = value
                    .get("operation")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let operation_type = operation.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if operation_type.is_empty() {
                    return None;
                }
                let operation_path = operation
                    .get("path")
                    .and_then(|v| v.as_str())
                    .map(|v| v.to_string());
                let tool_name = state
                    .extra
                    .tool_name_mapping
                    .to_custom_tool_name("apply_patch")
                    .to_string();
                let mut call_state = OpenAIApplyPatchState {
                    tool_call_id: call_id.to_string(),
                    operation_path,
                    has_diff: false,
                    end_emitted: false,
                };
                let mut out = vec![v2t::StreamPart::ToolInputStart {
                    id: call_id.to_string(),
                    tool_name,
                    provider_executed: false,
                    provider_metadata: None,
                }];
                if operation_type == "delete_file" {
                    let input = json!({
                        "callId": call_id,
                        "operation": operation,
                    })
                    .to_string();
                    out.push(v2t::StreamPart::ToolInputDelta {
                        id: call_id.to_string(),
                        delta: input,
                        provider_executed: false,
                        provider_metadata: None,
                    });
                    out.push(v2t::StreamPart::ToolInputEnd {
                        id: call_id.to_string(),
                        provider_executed: false,
                        provider_metadata: None,
                    });
                    call_state.has_diff = true;
                    call_state.end_emitted = true;
                } else {
                    let path = call_state.operation_path.as_deref().unwrap_or("");
                    let delta = format!(
                        "{{\"callId\":\"{}\",\"operation\":{{\"type\":\"{}\",\"path\":\"{}\",\"diff\":\"",
                        escape_json_delta(call_id),
                        escape_json_delta(operation_type),
                        escape_json_delta(path)
                    );
                    out.push(v2t::StreamPart::ToolInputDelta {
                        id: call_id.to_string(),
                        delta,
                        provider_executed: false,
                        provider_metadata: None,
                    });
                }
                state.has_tool_calls = true;
                state
                    .extra
                    .apply_patch_calls
                    .insert(output_index, call_state);
                return Some(out);
            } else if key == "openai.apply_patch_call.diff.delta" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(call_state) = state.extra.apply_patch_calls.get_mut(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    if !delta.is_empty() {
                        call_state.has_diff = true;
                        return Some(vec![v2t::StreamPart::ToolInputDelta {
                            id: call_state.tool_call_id.clone(),
                            delta: escape_json_delta(delta),
                            provider_executed: false,
                            provider_metadata: None,
                        }]);
                    }
                }
            } else if key == "openai.apply_patch_call.diff.done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                let diff = value.get("diff").and_then(|v| v.as_str()).unwrap_or("");
                if let Some(call_state) = state.extra.apply_patch_calls.get_mut(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    let mut out = Vec::new();
                    if !call_state.has_diff {
                        call_state.has_diff = true;
                        out.push(v2t::StreamPart::ToolInputDelta {
                            id: call_state.tool_call_id.clone(),
                            delta: escape_json_delta(diff),
                            provider_executed: false,
                            provider_metadata: None,
                        });
                    }
                    out.push(v2t::StreamPart::ToolInputDelta {
                        id: call_state.tool_call_id.clone(),
                        delta: "\"}}".into(),
                        provider_executed: false,
                        provider_metadata: None,
                    });
                    out.push(v2t::StreamPart::ToolInputEnd {
                        id: call_state.tool_call_id.clone(),
                        provider_executed: false,
                        provider_metadata: None,
                    });
                    call_state.end_emitted = true;
                    return Some(out);
                }
            } else if key == "openai.apply_patch_call.done" {
                let output_index = value.get("output_index").and_then(|v| v.as_u64())? as usize;
                if let Some(mut call_state) = state.extra.apply_patch_calls.remove(&output_index) {
                    if call_state.end_emitted {
                        return None;
                    }
                    let mut out = Vec::new();
                    if !call_state.has_diff {
                        let diff = value
                            .get("operation")
                            .and_then(|v| v.get("diff"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        call_state.has_diff = true;
                        out.push(v2t::StreamPart::ToolInputDelta {
                            id: call_state.tool_call_id.clone(),
                            delta: escape_json_delta(diff),
                            provider_executed: false,
                            provider_metadata: None,
                        });
                    }
                    out.push(v2t::StreamPart::ToolInputDelta {
                        id: call_state.tool_call_id.clone(),
                        delta: "\"}}".into(),
                        provider_executed: false,
                        provider_metadata: None,
                    });
                    out.push(v2t::StreamPart::ToolInputEnd {
                        id: call_state.tool_call_id.clone(),
                        provider_executed: false,
                        provider_metadata: None,
                    });
                    call_state.end_emitted = true;
                    return Some(out);
                }
            } else if key == "openai.provider_tool" {
                if let Some(mut parts) =
                    provider_tool_parts_from_data(value, &state.extra.tool_name_mapping)
                {
                    state.has_tool_calls = true;
                    let tool_type = parts.tool_type.clone();
                    if parts.is_approval_request {
                        let approval_id = parts
                            .approval_request_id
                            .clone()
                            .unwrap_or_else(|| parts.tool_call_id.clone());
                        let tool_call_id = Uuid::new_v4().to_string();
                        state
                            .extra
                            .approval_request_id_map
                            .insert(approval_id.clone(), tool_call_id.clone());
                        return Some(vec![
                            v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                                tool_call_id: tool_call_id.clone(),
                                tool_name: parts.tool_name,
                                input: parts.input,
                                provider_executed: parts.provider_executed,
                                provider_metadata: None,
                                dynamic: parts.dynamic,
                                provider_options: None,
                            }),
                            v2t::StreamPart::ToolApprovalRequest {
                                approval_id,
                                tool_call_id,
                                provider_metadata: None,
                            },
                        ]);
                    }
                    if let Some(approval_id) = parts.approval_request_id.as_ref() {
                        if let Some(mapped) = state.extra.approval_request_id_map.get(approval_id) {
                            parts.tool_call_id = mapped.clone();
                        }
                    }
                    let tool_call_id = parts.tool_call_id.clone();
                    let tool_call_metadata = match tool_type.as_str() {
                        "apply_patch" | "local_shell" | "shell" => parts.provider_metadata.clone(),
                        _ => None,
                    };
                    let tool_result_metadata = if tool_type == "mcp" {
                        parts.provider_metadata.clone()
                    } else {
                        None
                    };
                    let mut out = Vec::new();
                    if tool_type == "computer_use" {
                        if state.extra.open_tool_inputs.remove(&tool_call_id) {
                            out.push(v2t::StreamPart::ToolInputEnd {
                                id: tool_call_id.clone(),
                                provider_executed: true,
                                provider_metadata: None,
                            });
                        }
                    }
                    let skip_tool_call =
                        matches!(
                            tool_type.as_str(),
                            "web_search" | "file_search" | "image_generation" | "code_interpreter"
                        ) && state.extra.emitted_tool_calls.contains(&tool_call_id);
                    if !skip_tool_call {
                        out.push(v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: parts.tool_name.clone(),
                            input: parts.input,
                            provider_executed: parts.provider_executed,
                            provider_metadata: tool_call_metadata,
                            dynamic: parts.dynamic,
                            provider_options: None,
                        }));
                    }
                    if let Some(result) = parts.result.take() {
                        out.push(v2t::StreamPart::ToolResult {
                            tool_call_id: tool_call_id.clone(),
                            tool_name: parts.tool_name,
                            result,
                            is_error: parts.is_error,
                            preliminary: false,
                            provider_metadata: tool_result_metadata,
                        });
                    }
                    if !out.is_empty() {
                        return Some(out);
                    }
                }
            } else if key.starts_with("openai.tool_item_id.") {
                if let Some(iid) = value.get("item_id").and_then(|v| v.as_str()) {
                    let call_id = key.trim_start_matches("openai.tool_item_id.").to_string();
                    state.extra.tool_item_ids.insert(call_id, iid.to_string());
                }
            } else if key == "openai.function_call_done" {
                state.extra.has_function_calls = true;
            } else if key == "openai.finish" {
                if let Some(r) = value.get("incomplete_reason").and_then(|v| v.as_str()) {
                    state.extra.finish_hint = Some(r.to_string());
                }
            } else if key == "openai.failed" {
                state.extra.saw_response_failed = true;
                if state.extra.response_id.is_none() {
                    if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                        state.extra.response_id = Some(id.to_string());
                    }
                }
            } else if key == "openai.response" {
                if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                    state.extra.response_id = Some(id.to_string());
                }
                if let Some(st) = value.get("service_tier").and_then(|v| v.as_str()) {
                    state.extra.service_tier = Some(st.to_string());
                }
            }
            None
        },
    ));

    hooks.finish = Some(Box::new(|state: &EventMapperState<OpenAIStreamExtras>| {
        let reason = if state.extra.saw_response_failed {
            // TS mapper keeps default "other" for response.failed terminal trajectories.
            v2t::FinishReason::Other
        } else {
            map_finish_reason(
                state.extra.finish_hint.as_deref(),
                state.extra.has_function_calls,
            )
        };
        let mut inner = HashMap::new();
        if let Some(rid) = &state.extra.response_id {
            inner.insert("responseId".into(), serde_json::json!(rid));
        }
        if !state.extra.saw_response_failed {
            if let Some(st) = &state.extra.service_tier {
                inner.insert("serviceTier".into(), serde_json::json!(st));
            }
        }
        if !state.extra.logprobs.is_empty() {
            inner.insert(
                "logprobs".into(),
                serde_json::Value::Array(state.extra.logprobs.clone()),
            );
        }
        let metadata = if inner.is_empty() {
            None
        } else {
            let mut outer = HashMap::new();
            outer.insert("openai".into(), inner);
            Some(outer)
        };
        (reason, metadata)
    }));

    EventMapperConfig {
        warnings,
        treat_tool_names_as_text: HashSet::new(),
        default_text_id: "text-1",
        finish_reason_fallback: v2t::FinishReason::Stop,
        initial_extra: OpenAIStreamExtras {
            tool_name_mapping,
            approval_request_id_map,
            store,
            logprobs_enabled,
            ..Default::default()
        },
        hooks,
    }
}

fn map_finish_reason(hint: Option<&str>, has_function_calls: bool) -> v2t::FinishReason {
    match hint {
        None => {
            if has_function_calls {
                v2t::FinishReason::ToolCalls
            } else {
                v2t::FinishReason::Stop
            }
        }
        Some("max_output_tokens") => v2t::FinishReason::Length,
        Some("content_filter") => v2t::FinishReason::ContentFilter,
        Some(_) => {
            if has_function_calls {
                v2t::FinishReason::ToolCalls
            } else {
                v2t::FinishReason::Other
            }
        }
    }
}

fn extract_response_content(
    json: &serde_json::Value,
    tool_name_mapping: &ToolNameMapping,
    approval_request_id_map: &HashMap<String, String>,
) -> (Vec<v2t::Content>, bool) {
    let mut content: Vec<v2t::Content> = Vec::new();
    let mut has_function_calls = false;
    let mut approval_request_id_map = approval_request_id_map.clone();
    let output = match json.get("output").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return (content, false),
    };

    for item in output {
        let item_obj = match item.as_object() {
            Some(obj) => obj,
            None => continue,
        };
        let item_type = item_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match item_type {
            "message" => {
                if let Some(parts) = item_obj.get("content").and_then(|v| v.as_array()) {
                    let mut text_acc = String::new();
                    for part in parts {
                        if part.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                text_acc.push_str(text);
                            }
                        }
                    }
                    if !text_acc.is_empty() {
                        content.push(v2t::Content::Text {
                            text: text_acc,
                            provider_metadata: None,
                        });
                    }
                }
            }
            "function_call" => {
                if let (Some(call_id), Some(name)) = (
                    item_obj.get("call_id").and_then(|v| v.as_str()),
                    item_obj.get("name").and_then(|v| v.as_str()),
                ) {
                    let args = item_obj
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let provider_metadata = item_obj
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(|id| openai_item_metadata(id, []));
                    content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                        tool_call_id: call_id.to_string(),
                        tool_name: name.to_string(),
                        input: args,
                        provider_executed: false,
                        provider_metadata,
                        dynamic: false,
                        provider_options: None,
                    }));
                    has_function_calls = true;
                }
            }
            _ => {
                if let Some(tool_data) = provider_tool_data_from_output_item(item_obj) {
                    if let Some(mut parts) =
                        provider_tool_parts_from_data(&tool_data, tool_name_mapping)
                    {
                        let tool_type = parts.tool_type.clone();
                        if parts.is_approval_request {
                            let approval_id = parts
                                .approval_request_id
                                .clone()
                                .unwrap_or_else(|| parts.tool_call_id.clone());
                            let tool_call_id = Uuid::new_v4().to_string();
                            approval_request_id_map
                                .insert(approval_id.clone(), tool_call_id.clone());
                            content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                                tool_call_id: tool_call_id.clone(),
                                tool_name: parts.tool_name,
                                input: parts.input,
                                provider_executed: parts.provider_executed,
                                provider_metadata: None,
                                dynamic: parts.dynamic,
                                provider_options: None,
                            }));
                            content.push(v2t::Content::ToolApprovalRequest {
                                approval_id,
                                tool_call_id,
                                provider_metadata: None,
                            });
                            continue;
                        }
                        if let Some(approval_id) = parts.approval_request_id.as_ref() {
                            if let Some(mapped) = approval_request_id_map.get(approval_id) {
                                parts.tool_call_id = mapped.clone();
                            }
                        }
                        let tool_call_metadata = match tool_type.as_str() {
                            "apply_patch" | "local_shell" | "shell" => {
                                parts.provider_metadata.clone()
                            }
                            _ => None,
                        };
                        let tool_result_metadata = if tool_type == "mcp" {
                            parts.provider_metadata.clone()
                        } else {
                            None
                        };
                        content.push(v2t::Content::ToolCall(v2t::ToolCallPart {
                            tool_call_id: parts.tool_call_id.clone(),
                            tool_name: parts.tool_name.clone(),
                            input: parts.input,
                            provider_executed: parts.provider_executed,
                            provider_metadata: tool_call_metadata,
                            dynamic: parts.dynamic,
                            provider_options: None,
                        }));
                        if let Some(result) = parts.result {
                            content.push(v2t::Content::ToolResult {
                                tool_call_id: parts.tool_call_id,
                                tool_name: parts.tool_name,
                                result,
                                is_error: parts.is_error,
                                provider_metadata: tool_result_metadata,
                            });
                        }
                    }
                }
            }
        }
    }

    (content, has_function_calls)
}

// ---------- Request body builder (parity with Vercel getArgs) ----------

const TOP_LOGPROBS_MAX: u32 = 20;

#[derive(Clone, Copy, Debug)]
enum SystemMessageMode {
    Remove,
    System,
    Developer,
}

fn parse_system_message_mode(value: &str) -> Option<SystemMessageMode> {
    match value {
        "remove" => Some(SystemMessageMode::Remove),
        "system" => Some(SystemMessageMode::System),
        "developer" => Some(SystemMessageMode::Developer),
        _ => None,
    }
}

#[derive(Clone)]
struct ResponsesModelConfig {
    is_reasoning_model: bool,
    system_message_mode: SystemMessageMode,
    required_auto_truncation: bool,
    supports_flex_processing: bool,
    supports_priority_processing: bool,
    supports_non_reasoning_parameters: bool,
}

fn get_responses_model_config(model_id: &str) -> ResponsesModelConfig {
    let supports_flex = model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
        || (model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat"));
    let supports_priority = model_id.starts_with("gpt-4")
        || model_id.starts_with("gpt-5-mini")
        || (model_id.starts_with("gpt-5")
            && !model_id.starts_with("gpt-5-nano")
            && !model_id.starts_with("gpt-5-chat"))
        || model_id.starts_with("o3")
        || model_id.starts_with("o4-mini");
    let is_reasoning = model_id.starts_with("o1")
        || model_id.starts_with("o3")
        || model_id.starts_with("o4-mini")
        || model_id.starts_with("codex-mini")
        || model_id.starts_with("computer-use-preview")
        || (model_id.starts_with("gpt-5") && !model_id.starts_with("gpt-5-chat"));
    let system_mode = if is_reasoning {
        SystemMessageMode::Developer
    } else {
        SystemMessageMode::System
    };
    let supports_non_reasoning_parameters =
        model_id.starts_with("gpt-5.1") || model_id.starts_with("gpt-5.2");

    ResponsesModelConfig {
        is_reasoning_model: is_reasoning,
        system_message_mode: system_mode,
        required_auto_truncation: false,
        supports_flex_processing: supports_flex,
        supports_priority_processing: supports_priority,
        supports_non_reasoning_parameters,
    }
}

#[derive(Default, Debug, Clone)]
struct OpenAIProviderOptionsParsed {
    conversation: Option<String>,
    metadata: Option<serde_json::Value>,
    max_tool_calls: Option<u32>,
    parallel_tool_calls: Option<bool>,
    previous_response_id: Option<String>,
    store: Option<bool>,
    user: Option<String>,
    instructions: Option<String>,
    service_tier: Option<String>, // "auto" | "flex" | "priority"
    include: Option<Vec<String>>,
    text_verbosity: Option<String>, // "low" | "medium" | "high"
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<String>,
    safety_identifier: Option<String>,
    system_message_mode: Option<SystemMessageMode>,
    force_reasoning: Option<bool>,
    // JSON response format strict
    strict_json_schema: Option<bool>,
    truncation: Option<String>,
    // Reasoning controls
    reasoning_effort: Option<String>,
    reasoning_summary: Option<String>,
    // Logprobs
    logprobs_bool: Option<bool>,
    logprobs_n: Option<u32>,
}

fn parse_openai_provider_options(
    opts: &v2t::ProviderOptions,
    provider_scope: &str,
) -> OpenAIProviderOptionsParsed {
    let mut parsed = OpenAIProviderOptionsParsed::default();
    let map = match opts.get(provider_scope) {
        Some(map) => map,
        None => return parsed,
    };
    let get_bool = |k: &str| map.get(k).and_then(|v| v.as_bool());
    let get_str = |k: &str| map.get(k).and_then(|v| v.as_str().map(|s| s.to_string()));
    let get_arr = |k: &str| {
        map.get(k).and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect::<Vec<_>>()
        })
    };

    parsed.conversation = get_str("conversation");
    parsed.metadata = map.get("metadata").cloned();
    parsed.max_tool_calls = map
        .get("maxToolCalls")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok());
    parsed.parallel_tool_calls = get_bool("parallelToolCalls");
    parsed.previous_response_id = get_str("previousResponseId");
    parsed.store = get_bool("store");
    parsed.user = get_str("user");
    parsed.instructions = get_str("instructions");
    parsed.service_tier = get_str("serviceTier");
    parsed.include = get_arr("include");
    parsed.text_verbosity = get_str("textVerbosity");
    parsed.prompt_cache_key = get_str("promptCacheKey");
    parsed.prompt_cache_retention = get_str("promptCacheRetention");
    parsed.safety_identifier = get_str("safetyIdentifier");
    parsed.system_message_mode = map
        .get("systemMessageMode")
        .and_then(|v| v.as_str())
        .and_then(parse_system_message_mode);
    parsed.force_reasoning = get_bool("forceReasoning");
    parsed.strict_json_schema = get_bool("strictJsonSchema");
    parsed.truncation = get_str("truncation");
    parsed.reasoning_effort = get_str("reasoningEffort");
    parsed.reasoning_summary = get_str("reasoningSummary");

    // logprobs accepts bool or number
    if let Some(v) = map.get("logprobs") {
        if let Some(b) = v.as_bool() {
            parsed.logprobs_bool = Some(b);
        }
        if let Some(n) = v.as_u64() {
            parsed.logprobs_n = Some(n as u32);
        }
    }
    parsed
}

fn get_provider_option_value<'a>(
    provider_options: &'a Option<v2t::ProviderOptions>,
    provider_scope: &str,
    key: &str,
) -> Option<&'a serde_json::Value> {
    provider_options
        .as_ref()
        .and_then(|opts| opts.get(provider_scope))
        .and_then(|opts| opts.get(key))
}

fn get_provider_option_string(
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope: &str,
    key: &str,
) -> Option<String> {
    get_provider_option_value(provider_options, provider_scope, key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn openai_item_id_from_provider_options(
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope: &str,
) -> Option<String> {
    get_provider_option_string(provider_options, provider_scope, "itemId")
}

fn openai_reasoning_encrypted_content_from_provider_options(
    provider_options: &Option<v2t::ProviderOptions>,
    provider_scope: &str,
) -> Option<String> {
    get_provider_option_string(
        provider_options,
        provider_scope,
        "reasoningEncryptedContent",
    )
}

fn convert_to_openai_messages(
    prompt: &v2t::Prompt,
    system_mode: SystemMessageMode,
    file_id_prefixes: Option<&[String]>,
    provider_scope_name: &str,
    store: bool,
    tool_name_mapping: &ToolNameMapping,
    has_local_shell_tool: bool,
    has_shell_tool: bool,
    has_apply_patch_tool: bool,
) -> (Vec<Value>, Vec<v2t::CallWarning>) {
    let mut messages: Vec<Value> = Vec::new();
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();
    let mut processed_approval_ids = HashSet::new();

    // helper to build data URIs
    fn to_data_uri(media_type: &str, data: &v2t::DataContent) -> Option<String> {
        match data {
            v2t::DataContent::Base64 { base64 } => {
                Some(format!("data:{};base64,{}", media_type, base64))
            }
            v2t::DataContent::Bytes { bytes } => Some(format!(
                "data:{};base64,{}",
                media_type,
                base64::engine::general_purpose::STANDARD.encode(bytes)
            )),
            v2t::DataContent::Url { url } => Some(url.clone()),
        }
    }

    for msg in prompt {
        match msg {
            v2t::PromptMessage::System { content, .. } => match system_mode {
                SystemMessageMode::Remove => warnings.push(v2t::CallWarning::Other {
                    message: "system messages are removed for this model".into(),
                }),
                SystemMessageMode::System => {
                    messages.push(json!({"role":"system","content": content }))
                }
                SystemMessageMode::Developer => {
                    messages.push(json!({"role":"developer","content": content }))
                }
            },
            v2t::PromptMessage::User { content, .. } => {
                let mut parts: Vec<Value> = Vec::new();
                for (idx, part) in content.iter().enumerate() {
                    match part {
                        v2t::UserPart::Text { text, .. } => {
                            parts.push(json!({"type":"input_text","text": text}))
                        }
                        v2t::UserPart::File {
                            filename,
                            data,
                            media_type,
                            ..
                        } => {
                            if media_type.starts_with("image/") {
                                let mt = if media_type == "image/*" {
                                    "image/jpeg"
                                } else {
                                    media_type
                                };
                                if let Some(url) = to_data_uri(mt, data) {
                                    // If it's not an http(s) url and looks like an id with known prefix, use file_id
                                    let is_http =
                                        url.starts_with("http://") || url.starts_with("https://");
                                    if !is_http {
                                        if let Some(prefixes) = file_id_prefixes {
                                            if prefixes.iter().any(|p| url.starts_with(p)) {
                                                parts.push(
                                                    json!({"type":"input_image","file_id": url }),
                                                );
                                                continue;
                                            }
                                        }
                                    }
                                    parts.push(json!({"type":"input_image","image_url": url }));
                                }
                            } else if media_type == "application/pdf" {
                                match data {
                                    v2t::DataContent::Url { url } => {
                                        parts.push(json!({"type":"input_file","file_url": url}))
                                    }
                                    _ => {
                                        // file_id detection
                                        if let v2t::DataContent::Base64 { base64 } = data {
                                            if let Some(prefixes) = file_id_prefixes {
                                                if prefixes.iter().any(|p| base64.starts_with(p)) {
                                                    parts.push(json!({"type":"input_file","file_id": base64}));
                                                    continue;
                                                }
                                            }
                                        }
                                        let fname = filename
                                            .clone()
                                            .unwrap_or_else(|| format!("part-{}.pdf", idx));
                                        if let Some(uri) = to_data_uri("application/pdf", data) {
                                            parts.push(json!({"type":"input_file","filename": fname, "file_data": uri}));
                                        }
                                    }
                                }
                            } else {
                                warnings.push(v2t::CallWarning::Other {
                                    message: format!("unsupported file media type: {}", media_type),
                                });
                            }
                        }
                    }
                }
                if !parts.is_empty() {
                    messages.push(json!({"role":"user","content": parts}));
                }
            }
            v2t::PromptMessage::Assistant { content, .. } => {
                let mut reasoning_message_idx = HashMap::new();
                let mut reasoning_item_refs = HashSet::new();
                for part in content {
                    match part {
                        v2t::AssistantPart::Text {
                            text,
                            provider_options,
                        } => {
                            let item_id = openai_item_id_from_provider_options(
                                provider_options,
                                provider_scope_name,
                            );
                            if store && item_id.is_some() {
                                messages.push(json!({"type":"item_reference","id": item_id}));
                                continue;
                            }
                            let mut message = serde_json::Map::new();
                            message.insert("role".into(), json!("assistant"));
                            message.insert(
                                "content".into(),
                                json!([{"type":"output_text","text": text}]),
                            );
                            if let Some(id) = item_id {
                                message.insert("id".into(), json!(id));
                            }
                            messages.push(Value::Object(message));
                        }
                        v2t::AssistantPart::ToolCall(tc) => {
                            let item_id = openai_item_id_from_provider_options(
                                &tc.provider_options,
                                provider_scope_name,
                            );
                            if tc.provider_executed {
                                if store && item_id.is_some() {
                                    messages.push(json!({"type":"item_reference","id": item_id}));
                                }
                                continue;
                            }
                            if store && item_id.is_some() {
                                messages.push(json!({"type":"item_reference","id": item_id}));
                                continue;
                            }
                            let resolved_tool_name =
                                tool_name_mapping.to_provider_tool_name(&tc.tool_name);
                            let input_json: serde_json::Value =
                                serde_json::from_str(&tc.input).unwrap_or(json!({}));
                            if has_local_shell_tool && resolved_tool_name == "local_shell" {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("local_shell_call"));
                                obj.insert("call_id".into(), json!(tc.tool_call_id));
                                if let Some(id) = item_id {
                                    obj.insert("id".into(), json!(id));
                                }
                                let mut action = serde_json::Map::new();
                                action.insert("type".into(), json!("exec"));
                                if let Some(action_obj) =
                                    input_json.get("action").and_then(|v| v.as_object())
                                {
                                    if let Some(command) = action_obj.get("command") {
                                        action.insert("command".into(), command.clone());
                                    }
                                    if let Some(timeout) = action_obj
                                        .get("timeoutMs")
                                        .or_else(|| action_obj.get("timeout_ms"))
                                    {
                                        action.insert("timeout_ms".into(), timeout.clone());
                                    }
                                    if let Some(user) = action_obj.get("user") {
                                        action.insert("user".into(), user.clone());
                                    }
                                    if let Some(dir) = action_obj
                                        .get("workingDirectory")
                                        .or_else(|| action_obj.get("working_directory"))
                                    {
                                        action.insert("working_directory".into(), dir.clone());
                                    }
                                    if let Some(env) = action_obj.get("env") {
                                        action.insert("env".into(), env.clone());
                                    }
                                }
                                obj.insert("action".into(), Value::Object(action));
                                messages.push(Value::Object(obj));
                                continue;
                            }
                            if has_shell_tool && resolved_tool_name == "shell" {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("shell_call"));
                                obj.insert("call_id".into(), json!(tc.tool_call_id));
                                if let Some(id) = item_id {
                                    obj.insert("id".into(), json!(id));
                                }
                                obj.insert("status".into(), json!("completed"));
                                let mut action = serde_json::Map::new();
                                if let Some(action_obj) =
                                    input_json.get("action").and_then(|v| v.as_object())
                                {
                                    if let Some(commands) = action_obj.get("commands") {
                                        action.insert("commands".into(), commands.clone());
                                    }
                                    if let Some(timeout) = action_obj
                                        .get("timeoutMs")
                                        .or_else(|| action_obj.get("timeout_ms"))
                                    {
                                        action.insert("timeout_ms".into(), timeout.clone());
                                    }
                                    if let Some(max_len) = action_obj
                                        .get("maxOutputLength")
                                        .or_else(|| action_obj.get("max_output_length"))
                                    {
                                        action.insert("max_output_length".into(), max_len.clone());
                                    }
                                }
                                obj.insert("action".into(), Value::Object(action));
                                messages.push(Value::Object(obj));
                                continue;
                            }
                            if has_apply_patch_tool && resolved_tool_name == "apply_patch" {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("apply_patch_call"));
                                obj.insert("call_id".into(), json!(tc.tool_call_id));
                                if let Some(id) = item_id {
                                    obj.insert("id".into(), json!(id));
                                }
                                obj.insert("status".into(), json!("completed"));
                                if let Some(operation) = input_json.get("operation").cloned() {
                                    obj.insert("operation".into(), operation);
                                }
                                messages.push(Value::Object(obj));
                                continue;
                            }
                            let mut obj = serde_json::Map::new();
                            obj.insert("type".into(), json!("function_call"));
                            obj.insert("call_id".into(), json!(tc.tool_call_id));
                            obj.insert(
                                "name".into(),
                                json!(tool_name_mapping.to_provider_tool_name(&tc.tool_name)),
                            );
                            obj.insert("arguments".into(), json!(tc.input));
                            if let Some(id) = item_id {
                                obj.insert("id".into(), json!(id));
                            }
                            messages.push(Value::Object(obj));
                        }
                        v2t::AssistantPart::ToolResult(tr) => {
                            if store {
                                let item_id = openai_item_id_from_provider_options(
                                    &tr.provider_options,
                                    provider_scope_name,
                                )
                                .unwrap_or_else(|| tr.tool_call_id.clone());
                                messages.push(json!({"type":"item_reference","id": item_id}));
                            } else {
                                warnings.push(v2t::CallWarning::Other {
                                    message: format!(
                                        "Results for OpenAI tool {} are not sent to the API when store is false",
                                        tr.tool_name
                                    ),
                                });
                            }
                        }
                        v2t::AssistantPart::Reasoning {
                            text,
                            provider_options,
                        } => {
                            let item_id = openai_item_id_from_provider_options(
                                provider_options,
                                provider_scope_name,
                            );
                            let encrypted =
                                openai_reasoning_encrypted_content_from_provider_options(
                                    provider_options,
                                    provider_scope_name,
                                );
                            let Some(reasoning_id) = item_id else {
                                warnings.push(v2t::CallWarning::Other {
                                    message: format!(
                                        "Non-OpenAI reasoning parts are not supported. Skipping reasoning part: {}",
                                        text
                                    ),
                                });
                                continue;
                            };
                            if store {
                                if reasoning_item_refs.insert(reasoning_id.clone()) {
                                    messages
                                        .push(json!({"type":"item_reference","id": reasoning_id}));
                                }
                                continue;
                            }
                            let summary_entry = if text.is_empty() {
                                None
                            } else {
                                Some(json!({"type":"summary_text","text": text}))
                            };
                            if let Some(idx) = reasoning_message_idx.get(&reasoning_id).copied() {
                                if let Some(entry) = summary_entry {
                                    if let Some(obj) = messages
                                        .get_mut(idx)
                                        .and_then(|v: &mut Value| v.as_object_mut())
                                    {
                                        if let Some(summary) = obj
                                            .get_mut("summary")
                                            .and_then(|v: &mut Value| v.as_array_mut())
                                        {
                                            summary.push(entry);
                                        }
                                        if let Some(enc) = encrypted.clone() {
                                            obj.insert("encrypted_content".into(), json!(enc));
                                        }
                                    }
                                } else {
                                    warnings.push(v2t::CallWarning::Other {
                                        message: format!(
                                            "Cannot append empty reasoning part to existing reasoning sequence. Skipping reasoning part: {}",
                                            reasoning_id
                                        ),
                                    });
                                }
                            } else {
                                let mut obj = serde_json::Map::new();
                                obj.insert("type".into(), json!("reasoning"));
                                obj.insert("id".into(), json!(reasoning_id.clone()));
                                let summary = summary_entry
                                    .map(|entry| Value::Array(vec![entry]))
                                    .unwrap_or_else(|| Value::Array(Vec::new()));
                                obj.insert("summary".into(), summary);
                                if let Some(enc) = encrypted {
                                    obj.insert("encrypted_content".into(), json!(enc));
                                }
                                messages.push(Value::Object(obj));
                                reasoning_message_idx.insert(reasoning_id, messages.len() - 1);
                            }
                        }
                        v2t::AssistantPart::File { .. } => { /* skip */ }
                    }
                }
            }
            v2t::PromptMessage::Tool { content, .. } => {
                for part in content {
                    match part {
                        v2t::ToolMessagePart::ToolApprovalResponse(resp) => {
                            if !processed_approval_ids.insert(resp.approval_id.clone()) {
                                continue;
                            }
                            if store {
                                messages
                                    .push(json!({"type":"item_reference","id": resp.approval_id}));
                            }
                            messages.push(json!({
                                "type":"mcp_approval_response",
                                "approval_request_id": resp.approval_id,
                                "approve": resp.approved,
                            }));
                        }
                        v2t::ToolMessagePart::ToolResult(tr) => {
                            let resolved_tool_name =
                                tool_name_mapping.to_provider_tool_name(&tr.tool_name);
                            if has_local_shell_tool && resolved_tool_name == "local_shell" {
                                if let v2t::ToolResultOutput::Json { value } = &tr.output {
                                    if let Some(obj) = value.as_object() {
                                        if let Some(output) = obj.get("output") {
                                            messages.push(json!({
                                                "type":"local_shell_call_output",
                                                "call_id": tr.tool_call_id,
                                                "output": output.clone(),
                                            }));
                                            continue;
                                        }
                                    }
                                }
                            }
                            if has_shell_tool && resolved_tool_name == "shell" {
                                if let v2t::ToolResultOutput::Json { value } = &tr.output {
                                    if let Some(obj) = value.as_object() {
                                        if let Some(output) =
                                            obj.get("output").and_then(|v| v.as_array())
                                        {
                                            let mapped = output
                                                .iter()
                                                .filter_map(|entry| {
                                                    let entry = entry.as_object()?;
                                                    let stdout = entry.get("stdout")?.clone();
                                                    let stderr = entry.get("stderr")?.clone();
                                                    let outcome_obj = entry.get("outcome")?.as_object()?;
                                                    let outcome_type = outcome_obj.get("type")?.as_str()?;
                                                    let outcome = if outcome_type == "timeout" {
                                                        json!({"type":"timeout"})
                                                    } else if outcome_type == "exit" {
                                                        let exit_code = outcome_obj
                                                            .get("exitCode")
                                                            .or_else(|| outcome_obj.get("exit_code"))?;
                                                        json!({"type":"exit","exit_code": exit_code})
                                                    } else {
                                                        return None;
                                                    };
                                                    Some(json!({
                                                        "stdout": stdout,
                                                        "stderr": stderr,
                                                        "outcome": outcome,
                                                    }))
                                                })
                                                .collect::<Vec<_>>();
                                            messages.push(json!({
                                                "type":"shell_call_output",
                                                "call_id": tr.tool_call_id,
                                                "output": mapped,
                                            }));
                                            continue;
                                        }
                                    }
                                }
                            }
                            if has_apply_patch_tool && resolved_tool_name == "apply_patch" {
                                if let v2t::ToolResultOutput::Json { value } = &tr.output {
                                    if let Some(obj) = value.as_object() {
                                        if let Some(status) = obj.get("status") {
                                            let mut out = serde_json::Map::new();
                                            out.insert(
                                                "type".into(),
                                                json!("apply_patch_call_output"),
                                            );
                                            out.insert("call_id".into(), json!(tr.tool_call_id));
                                            out.insert("status".into(), status.clone());
                                            if let Some(output) = obj.get("output") {
                                                out.insert("output".into(), output.clone());
                                            }
                                            messages.push(Value::Object(out));
                                            continue;
                                        }
                                    }
                                }
                            }
                            let out_val = tool_output_to_value(&tr.output);
                            messages.push(json!({
                                "type":"function_call_output",
                                "call_id": tr.tool_call_id,
                                "output": out_val,
                            }));
                        }
                    }
                }
            }
        }
    }

    (messages, warnings)
}

fn tool_output_to_value(output: &v2t::ToolResultOutput) -> Value {
    match output {
        v2t::ToolResultOutput::Text { value } => Value::String(value.clone()),
        v2t::ToolResultOutput::ErrorText { value } => Value::String(value.clone()),
        v2t::ToolResultOutput::Json { value } => Value::String(value.to_string()),
        v2t::ToolResultOutput::ErrorJson { value } => Value::String(value.to_string()),
        v2t::ToolResultOutput::Content { value } => {
            let mut parts: Vec<Value> = Vec::new();
            for item in value {
                match item {
                    v2t::ToolResultInlineContent::Text { text } => {
                        parts.push(json!({"type":"input_text","text": text}));
                    }
                    v2t::ToolResultInlineContent::Media { data, media_type } => {
                        if media_type.starts_with("image/") {
                            parts.push(json!({
                                "type":"input_image",
                                "image_url": format!("data:{};base64,{}", media_type, data),
                            }));
                        } else {
                            parts.push(json!({
                                "type":"input_file",
                                "filename": "data",
                                "file_data": format!("data:{};base64,{}", media_type, data),
                            }));
                        }
                    }
                }
            }
            Value::Array(parts)
        }
    }
}

fn map_tool_choice(
    choice: &Option<v2t::ToolChoice>,
    tool_name_mapping: &ToolNameMapping,
) -> Option<Value> {
    match choice {
        None => None,
        Some(v2t::ToolChoice::Auto) => Some(Value::String("auto".into())),
        Some(v2t::ToolChoice::None) => Some(Value::String("none".into())),
        Some(v2t::ToolChoice::Required) => Some(Value::String("required".into())),
        Some(v2t::ToolChoice::Tool { name }) => {
            let mapped = tool_name_mapping.to_provider_tool_name(name);
            if is_openai_builtin_tool(mapped) {
                Some(json!({"type": mapped}))
            } else {
                Some(json!({"type":"function","name": mapped}))
            }
        }
    }
}

fn function_tool_strict(tool: &v2t::FunctionTool) -> Option<bool> {
    tool.strict.or_else(|| {
        tool.provider_options.as_ref().and_then(|opts| {
            opts.get("openai")
                .or_else(|| opts.get("openai.responses"))
                .and_then(|scope| scope.get("strict"))
                .and_then(|value| value.as_bool())
        })
    })
}

fn build_request_body(
    options: &v2t::CallOptions,
    model_id: &str,
    cfg: &OpenAIConfig,
) -> Result<(Value, Vec<v2t::CallWarning>), SdkError> {
    let mut warnings: Vec<v2t::CallWarning> = Vec::new();
    // Unsupported settings warnings
    if options.top_k.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "topK".into(),
            details: None,
        });
    }
    if options.seed.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "seed".into(),
            details: None,
        });
    }
    if options.presence_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "presencePenalty".into(),
            details: None,
        });
    }
    if options.frequency_penalty.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "frequencyPenalty".into(),
            details: None,
        });
    }
    if options.stop_sequences.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "stopSequences".into(),
            details: None,
        });
    }

    let prov = parse_openai_provider_options(&options.provider_options, &cfg.provider_scope_name);
    let model_cfg = get_responses_model_config(model_id);
    let base_is_reasoning_model = model_cfg.is_reasoning_model;
    let is_reasoning_model = prov.force_reasoning.unwrap_or(base_is_reasoning_model);
    let system_message_mode = prov.system_message_mode.unwrap_or_else(|| {
        if is_reasoning_model {
            SystemMessageMode::Developer
        } else {
            model_cfg.system_message_mode
        }
    });
    let tool_name_mapping = build_tool_name_mapping(&options.tools);
    let has_local_shell_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.local_shell"));
    let has_shell_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.shell"));
    let has_apply_patch_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.apply_patch"));
    let has_web_search_tool = options.tools.iter().any(|tool| {
        matches!(
            tool,
            v2t::Tool::Provider(t)
                if t.id == "openai.web_search" || t.id == "openai.web_search_preview"
        )
    });
    let has_code_interpreter_tool = options
        .tools
        .iter()
        .any(|tool| matches!(tool, v2t::Tool::Provider(t) if t.id == "openai.code_interpreter"));
    let (messages, mut message_warnings) = convert_to_openai_messages(
        &options.prompt,
        system_message_mode,
        cfg.file_id_prefixes.as_deref(),
        &cfg.provider_scope_name,
        prov.store.unwrap_or(true),
        &tool_name_mapping,
        has_local_shell_tool,
        has_shell_tool,
        has_apply_patch_tool,
    );
    warnings.append(&mut message_warnings);

    if prov.conversation.is_some() && prov.previous_response_id.is_some() {
        warnings.push(v2t::CallWarning::UnsupportedSetting {
            setting: "conversation".into(),
            details: Some("conversation and previousResponseId cannot be used together".into()),
        });
    }
    let store_value = prov.store;
    let top_logprobs = if let Some(n) = prov.logprobs_n {
        Some(n)
    } else if prov.logprobs_bool == Some(true) {
        Some(TOP_LOGPROBS_MAX)
    } else {
        None
    };
    let logprobs_requested = prov.logprobs_bool == Some(true) || prov.logprobs_n.unwrap_or(0) > 0;
    let mut include = prov.include.clone();
    let mut add_include = |key: &str| {
        if let Some(list) = include.as_mut() {
            if !list.iter().any(|s| s == key) {
                list.push(key.to_string());
            }
        } else {
            include = Some(vec![key.to_string()]);
        }
    };
    if logprobs_requested {
        add_include("message.output_text.logprobs");
    }
    if has_web_search_tool {
        add_include("web_search_call.action.sources");
    }
    if has_code_interpreter_tool {
        add_include("code_interpreter_call.outputs");
    }
    if store_value == Some(false) && is_reasoning_model {
        add_include("reasoning.encrypted_content");
    }

    // Response format / text options
    let mut text_obj: Option<Value> = None;
    if let Some(v2t::ResponseFormat::Json {
        schema,
        name,
        description,
    }) = &options.response_format
    {
        let mut format_obj = json!({"type":"json_object"});
        if let Some(s) = schema {
            format_obj = json!({
                "type": "json_schema",
                "strict": prov.strict_json_schema.unwrap_or(true),
                "name": name.clone().unwrap_or_else(|| "response".into()),
                "description": description.clone(),
                "schema": s,
            });
        }
        text_obj = Some(json!({"format": format_obj}));
    }
    if let Some(v) = prov.text_verbosity {
        let base = text_obj.take().unwrap_or_else(|| json!({}));
        let mut obj = base.as_object().cloned().unwrap_or_default();
        obj.insert("verbosity".into(), Value::String(v));
        text_obj = Some(Value::Object(obj));
    }

    // Base args (preserve insertion order so instructions sit next to model)
    let mut body_map = serde_json::Map::new();
    body_map.insert("model".into(), json!(model_id));
    if let Some(i) = prov.instructions.as_deref() {
        body_map.insert("instructions".into(), json!(i));
    }
    body_map.insert("input".into(), json!(messages));
    if let Some(t) = options.temperature {
        body_map.insert("temperature".into(), json!(t));
    }
    if let Some(tp) = options.top_p {
        body_map.insert("top_p".into(), json!(tp));
    }
    if let Some(mx) = options.max_output_tokens {
        body_map.insert("max_output_tokens".into(), json!(mx));
    }
    let mut body = Value::Object(body_map);
    if let Some(t) = text_obj {
        body["text"] = t;
    }

    // Provider options passthrough
    if let Some(m) = prov.metadata {
        body["metadata"] = m;
    }
    if let Some(c) = prov.conversation {
        body["conversation"] = json!(c);
    }
    if let Some(n) = prov.max_tool_calls {
        body["max_tool_calls"] = json!(n);
    }
    if let Some(b) = prov.parallel_tool_calls {
        body["parallel_tool_calls"] = json!(b);
    }
    if let Some(s) = prov.previous_response_id {
        body["previous_response_id"] = json!(s);
    }
    if let Some(b) = store_value {
        body["store"] = json!(b);
    }
    if let Some(u) = prov.user {
        body["user"] = json!(u);
    }
    if let Some(k) = prov.prompt_cache_key {
        body["prompt_cache_key"] = json!(k);
    }
    if let Some(r) = prov.prompt_cache_retention {
        body["prompt_cache_retention"] = json!(r);
    }
    if let Some(s) = prov.safety_identifier {
        body["safety_identifier"] = json!(s);
    }
    if let Some(t) = prov.truncation {
        body["truncation"] = json!(t);
    }

    // Tools
    if !options.tools.is_empty() {
        let mut tools: Vec<Value> = Vec::new();
        for tool in &options.tools {
            match tool {
                v2t::Tool::Function(t) => {
                    let params = normalize_object_schema(&t.input_schema);
                    let mut function_tool = json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": params
                    });
                    if let Some(strict) = function_tool_strict(t) {
                        function_tool["strict"] = json!(strict);
                    }
                    tools.push(function_tool);
                }
                v2t::Tool::Provider(t) => {
                    if let Some(val) = build_openai_provider_tool(t)? {
                        tools.push(val);
                    } else {
                        warnings.push(v2t::CallWarning::UnsupportedTool {
                            tool_name: t.name.clone(),
                            details: Some(format!("unsupported provider tool id {}", t.id)),
                        });
                    }
                }
            }
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
    }

    if let Some(tc) = map_tool_choice(&options.tool_choice, &tool_name_mapping) {
        body["tool_choice"] = tc;
    }

    // Logprobs and include
    if let Some(n) = top_logprobs {
        body["top_logprobs"] = json!(n);
    }
    if let Some(incl) = include {
        body["include"] = json!(incl);
    }

    // Model-specific handling
    if is_reasoning_model {
        let allow_non_reasoning = prov.reasoning_effort.as_deref() == Some("none")
            && model_cfg.supports_non_reasoning_parameters;
        if !allow_non_reasoning {
            if body.get("temperature").is_some() {
                body.as_object_mut().unwrap().remove("temperature");
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "temperature".into(),
                    details: Some("temperature is not supported for reasoning models".into()),
                });
            }
            if body.get("top_p").is_some() {
                body.as_object_mut().unwrap().remove("top_p");
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "topP".into(),
                    details: Some("topP is not supported for reasoning models".into()),
                });
            }
        }
        // reasoning object
        if prov.reasoning_effort.is_some() || prov.reasoning_summary.is_some() {
            let mut r = serde_json::Map::new();
            if let Some(e) = prov.reasoning_effort.as_ref() {
                r.insert("effort".into(), Value::String(e.clone()));
            }
            if let Some(s) = prov.reasoning_summary.as_ref() {
                r.insert("summary".into(), Value::String(s.clone()));
            }
            body["reasoning"] = Value::Object(r);
        }
    } else {
        if prov.reasoning_effort.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "reasoningEffort".into(),
                details: Some("reasoningEffort is not supported for non-reasoning models".into()),
            });
        }
        if prov.reasoning_summary.is_some() {
            warnings.push(v2t::CallWarning::UnsupportedSetting {
                setting: "reasoningSummary".into(),
                details: Some("reasoningSummary is not supported for non-reasoning models".into()),
            });
        }
    }

    if model_cfg.required_auto_truncation {
        body["truncation"] = json!("auto");
    }

    if let Some(defaults) = cfg.request_defaults.as_ref() {
        if let Some(overrides) = request_overrides_from_json(&cfg.provider_scope_name, defaults) {
            tracing::debug!(
                provider_scope = %cfg.provider_scope_name,
                override_keys = ?json_object_keys(&overrides),
                "openai request defaults resolved"
            );
            let disallow = ["model", "input", "stream", "tools", "tool_choice"];
            merge_options_with_disallow(&mut body, &overrides, &disallow);
            tracing::debug!(
                provider_scope = %cfg.provider_scope_name,
                has_reasoning_effort = body.get("reasoning_effort").is_some(),
                has_reasoning = body.get("reasoning").is_some(),
                has_reasoning_effort_nested = body
                    .get("reasoning")
                    .and_then(|v| v.get("effort"))
                    .is_some(),
                "openai request defaults merged"
            );
            // Keep explicit provider options authoritative for effort while preserving
            // defaults for other reasoning fields (e.g. summary).
            if is_reasoning_model {
                if let Some(explicit_effort) = prov.reasoning_effort.as_ref() {
                    if let Some(body_obj) = body.as_object_mut() {
                        let reasoning = body_obj
                            .entry("reasoning".to_string())
                            .or_insert_with(|| Value::Object(serde_json::Map::new()));
                        if !reasoning.is_object() {
                            *reasoning = Value::Object(serde_json::Map::new());
                        }
                        if let Some(reasoning_obj) = reasoning.as_object_mut() {
                            reasoning_obj.insert(
                                "effort".to_string(),
                                Value::String(explicit_effort.clone()),
                            );
                        }
                    }
                }
            }
        } else {
            tracing::debug!(
                provider_scope = %cfg.provider_scope_name,
                "openai request defaults present but no overrides matched"
            );
        }
    }

    // Service tier validation
    if let Some(tier) = prov.service_tier {
        match tier.as_str() {
            "flex" if !model_cfg.supports_flex_processing => {
                warnings.push(v2t::CallWarning::UnsupportedSetting {
                    setting: "serviceTier".into(),
                    details: Some(
                        "flex processing is only available for o3, o4-mini, and gpt-5 models"
                            .into(),
                    ),
                });
            }
            "priority" if !model_cfg.supports_priority_processing => {
                warnings.push(v2t::CallWarning::UnsupportedSetting { setting: "serviceTier".into(), details: Some("priority processing is only available for supported models and requires Enterprise access".into()) });
            }
            _ => {
                body["service_tier"] = json!(tier);
            }
        }
    }

    Ok((body, warnings))
}

fn json_object_keys(value: &Value) -> Vec<String> {
    value
        .as_object()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_default()
}

impl<T: HttpTransport> OpenAIResponsesLanguageModel<T> {
    pub(crate) async fn stream_with_body(
        &self,
        body: Value,
        include_raw: bool,
    ) -> Result<EventStream, SdkError> {
        // Build headers for logging
        let (bytes, _resp_headers) = match self.send(body).await {
            Ok(ok) => ok,
            Err(e) => {
                return Err(e);
            }
        };

        let pipeline = PipelineBuilder::<OpenAIResponsesChunk>::new()
            .with_provider("openai_official")
            .include_raw(include_raw)
            .build(bytes);
        Ok(Box::pin(pipeline))
    }
}
