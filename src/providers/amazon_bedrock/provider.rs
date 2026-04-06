use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::core::request_builder::defaults::provider_defaults_from_json;
use crate::core::{LanguageModel, SdkError};
use crate::provider::{
    build_provider_transport_config, registry::ProviderRegistration, Credentials,
    ReasoningScopeContext,
};
use crate::types::catalog::{ProviderDefinition, SdkType};
use crate::types::v2 as v2t;
use serde_json::Value as JsonValue;
use tracing::info;

use crate::providers::amazon_bedrock::config::{BedrockAuth, BedrockConfig, SigV4Config};
use crate::providers::amazon_bedrock::language_model::BedrockLanguageModel;

const TRACE_PREFIX: &str = "[BEDROCK]";
const DEFAULT_BASE_URL_FMT: &str = "https://bedrock-runtime.{region}.amazonaws.com";
const BEDROCK_TIMEOUT_SECS: u64 = 45;

fn match_bedrock(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::AmazonBedrock)
}

fn default_headers() -> Vec<(String, String)> {
    vec![
        ("content-type".into(), "application/json".into()),
        ("accept".into(), "application/json".into()),
    ]
}

fn resolve_bedrock_api_key(creds: &Credentials) -> Option<String> {
    creds
        .as_api_key()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            creds
                .as_bearer()
                .map(|value| value.trim_start_matches("Bearer ").to_string())
        })
        .or_else(|| std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok())
        .filter(|value| !value.trim().is_empty())
}

fn resolve_bedrock_base_url_and_region(def: &ProviderDefinition) -> (String, String) {
    let mut region = extract_region_hint(def);

    let base_url = if def.base_url.trim().is_empty() {
        let resolved_region = region.clone().unwrap_or_else(|| "us-east-1".into());
        region.get_or_insert(resolved_region.clone());
        DEFAULT_BASE_URL_FMT.replace("{region}", &resolved_region)
    } else {
        def.base_url.clone()
    };

    let region = region
        .or_else(|| infer_region_from_url(&base_url))
        .unwrap_or_else(|| std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into()));

    (base_url, region)
}

fn resolve_bedrock_auth(
    api_key: Option<String>,
    region: &str,
    headers: &mut Vec<(String, String)>,
) -> Result<BedrockAuth, SdkError> {
    if let Some(token) = api_key {
        headers.push(("authorization".into(), format!("Bearer {}", token)));
        return Ok(BedrockAuth::ApiKey { token });
    }

    let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| SdkError::Unauthorized)?;
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| SdkError::Unauthorized)?;
    let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
    Ok(BedrockAuth::SigV4(SigV4Config {
        access_key_id: access_key,
        secret_access_key: secret_key,
        session_token,
        region: region.to_string(),
    }))
}

fn merge_provider_headers(
    def: &ProviderDefinition,
    headers: Vec<(String, String)>,
) -> (Vec<(String, String)>, Option<v2t::ProviderOptions>) {
    let mut header_map: BTreeMap<String, (String, String)> = headers
        .into_iter()
        .map(|(key, value)| (key.to_ascii_lowercase(), (key, value)))
        .collect();

    let mut default_options: Option<v2t::ProviderOptions> = None;
    for (key, value) in &def.headers {
        if crate::core::options::is_internal_sdk_header(key) {
            if default_options.is_none() {
                if let Ok(json) = serde_json::from_str::<JsonValue>(value) {
                    default_options = provider_defaults_from_json(&def.name, &json);
                }
            }
            continue;
        }
        header_map.insert(key.to_ascii_lowercase(), (key.clone(), value.clone()));
    }

    (header_map.into_values().collect(), default_options)
}

fn build_bedrock(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let mut headers = default_headers();
    let api_key = resolve_bedrock_api_key(creds);
    let (base_url, region) = resolve_bedrock_base_url_and_region(def);
    let auth = resolve_bedrock_auth(api_key, &region, &mut headers)?;
    let (headers, default_options) = merge_provider_headers(def, headers);

    let transport_cfg = build_provider_transport_config(
        def,
        Some(std::time::Duration::from_secs(BEDROCK_TIMEOUT_SECS)),
    );

    let http = crate::transport_reqwest::ReqwestTransport::try_new(&transport_cfg)
        .map_err(SdkError::Transport)?;

    let cfg = BedrockConfig {
        provider_name: "amazon-bedrock.converse",
        provider_scope_name: def.name.clone(),
        base_url,
        headers,
        http,
        transport_cfg,
        supported_urls: HashMap::new(),
        default_options,
        auth,
    };

    info!(
        "{}: configured Amazon Bedrock model={} region={} provider_scope={}",
        TRACE_PREFIX, model, region, def.name
    );

    let lm = BedrockLanguageModel::new(model.to_string(), cfg);
    Ok(Arc::new(lm))
}

fn bedrock_reasoning_scope(ctx: &ReasoningScopeContext) -> Option<Vec<String>> {
    let model = ctx.model_id?.to_ascii_lowercase();
    let is_anthropic = model.contains("anthropic")
        || model.contains("claude")
        || model.contains("sonnet")
        || model.contains("haiku")
        || model.contains("opus");
    if !is_anthropic {
        return None;
    }

    let mut aliases = Vec::new();
    aliases.push("anthropic".to_string());
    aliases.push("bedrock".to_string());
    Some(aliases)
}

fn extract_region_hint(def: &ProviderDefinition) -> Option<String> {
    if let Some(region) = def.query_params.get("region") {
        if !region.trim().is_empty() {
            return Some(region.clone());
        }
    }
    for key in ["x-aws-region", "aws-region", "x-region"] {
        if let Some(value) = def.headers.get(key) {
            if !value.trim().is_empty() {
                return Some(value.clone());
            }
        }
    }
    None
}

fn infer_region_from_url(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    let host = parsed.host_str()?;
    if let Some(rest) = host.strip_prefix("bedrock-runtime.") {
        return rest.split('.').next().map(|s| s.to_string());
    }
    let segments: Vec<&str> = host.split('.').collect();
    if segments.len() >= 3 {
        // e.g., runtime.us-east-1.bedrock.amazonaws.com
        return Some(segments[1].to_string());
    }
    None
}

pub(crate) fn provider_registrations() -> &'static [ProviderRegistration] {
    static REGISTRATIONS: &[ProviderRegistration] = &[ProviderRegistration {
        id: "amazon-bedrock",
        sdk_type: SdkType::AmazonBedrock,
        matches: Some(match_bedrock),
        build: build_bedrock,
        reasoning_scope: Some(bedrock_reasoning_scope),
    }];

    REGISTRATIONS
}
