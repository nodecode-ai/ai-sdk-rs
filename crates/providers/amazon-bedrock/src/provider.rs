use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_core::{LanguageModel, SdkError};
use crate::ai_sdk_provider::{
    apply_stream_idle_timeout_ms, registry::ProviderRegistration, Credentials,
    ReasoningScopeContext,
};
use crate::ai_sdk_types::catalog::{ProviderDefinition, SdkType};
use crate::ai_sdk_types::v2 as v2t;
use inventory::submit;
use serde_json::Value as JsonValue;
use tracing::info;

use crate::provider_amazon_bedrock::config::{BedrockAuth, BedrockConfig, SigV4Config};
use crate::provider_amazon_bedrock::language_model::BedrockLanguageModel;

const TRACE_PREFIX: &str = "[BEDROCK]";
const DEFAULT_BASE_URL_FMT: &str = "https://bedrock-runtime.{region}.amazonaws.com";

fn match_bedrock(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::AmazonBedrock)
}

fn build_bedrock(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let mut base_headers: Vec<(String, String)> = Vec::new();
    base_headers.push(("content-type".into(), "application/json".into()));
    base_headers.push(("accept".into(), "application/json".into()));

    let api_key_from_creds = creds
        .as_api_key()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            creds
                .as_bearer()
                .map(|b| b.trim_start_matches("Bearer ").to_string())
        })
        .filter(|s| !s.trim().is_empty());

    let api_key = api_key_from_creds
        .or_else(|| std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok())
        .filter(|s| !s.trim().is_empty());

    let mut region = extract_region_hint(def);

    let base_url = if def.base_url.trim().is_empty() {
        let resolved_region = region.clone().unwrap_or_else(|| "us-east-1".into());
        region.get_or_insert(resolved_region.clone());
        DEFAULT_BASE_URL_FMT.replace("{region}", &resolved_region)
    } else {
        def.base_url.clone()
    };

    if region.is_none() {
        region = infer_region_from_url(&base_url);
    }
    let region = region
        .unwrap_or_else(|| std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into()));

    let auth = if let Some(key) = api_key {
        base_headers.push(("authorization".into(), format!("Bearer {}", key)));
        BedrockAuth::ApiKey { token: key }
    } else {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| SdkError::Unauthorized)?;
        let secret_key =
            std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| SdkError::Unauthorized)?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        BedrockAuth::SigV4(SigV4Config {
            access_key_id: access_key,
            secret_access_key: secret_key,
            session_token,
            region: region.clone(),
        })
    };

    let mut header_map: BTreeMap<String, (String, String)> = BTreeMap::new();
    for (k, v) in base_headers {
        header_map.insert(k.to_ascii_lowercase(), (k, v));
    }

    let mut default_options: Option<v2t::ProviderOptions> = None;
    for (k, v) in def.headers.iter() {
        if crate::ai_sdk_core::options::is_internal_sdk_header(k) {
            if default_options.is_none() {
                if let Ok(json) = serde_json::from_str::<JsonValue>(v) {
                    default_options = provider_defaults_from_json(&def.name, &json);
                }
            }
            continue;
        }
        header_map.insert(k.to_ascii_lowercase(), (k.clone(), v.clone()));
    }

    let headers: Vec<(String, String)> = header_map.into_values().collect();

    let mut transport_cfg = TransportConfig::default();
    transport_cfg.idle_read_timeout = std::time::Duration::from_secs(45);
    apply_stream_idle_timeout_ms(def, &mut transport_cfg);

    let http = crate::reqwest_transport::ReqwestTransport::try_new(&transport_cfg)
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

submit! {
    ProviderRegistration {
        id: "amazon-bedrock",
        sdk_type: SdkType::AmazonBedrock,
        matches: Some(match_bedrock),
        build: build_bedrock,
        reasoning_scope: Some(bedrock_reasoning_scope),
    }
}
