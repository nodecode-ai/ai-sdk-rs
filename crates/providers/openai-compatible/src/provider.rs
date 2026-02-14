use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_core::{EmbeddingModel, ImageModel, LanguageModel, SdkError};
use crate::ai_sdk_provider::{
    apply_stream_idle_timeout_ms, registry::ProviderRegistration, Credentials,
};
use crate::ai_sdk_types::catalog::{ProviderDefinition, SdkType};
use crate::ai_sdk_types::v2 as v2t;

use crate::provider_openai_compatible::chat::language_model::{
    OpenAICompatibleChatConfig, OpenAICompatibleChatLanguageModel,
};
use crate::provider_openai_compatible::completion::language_model::{
    OpenAICompatibleCompletionConfig, OpenAICompatibleCompletionLanguageModel,
};
use crate::provider_openai_compatible::embedding::embedding_model::{
    OpenAICompatibleEmbeddingConfig, OpenAICompatibleEmbeddingModel,
    DEFAULT_MAX_EMBEDDINGS_PER_CALL,
};
use crate::provider_openai_compatible::image::image_model::{
    OpenAICompatibleImageConfig, OpenAICompatibleImageModel,
};

const _TRACE_PREFIX: &str = "[OPENAI-COMP-CMPL]";

fn default_headers_from_creds(
    api_key: Option<String>,
    bearer: Option<String>,
) -> Vec<(String, String)> {
    let mut h = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "application/json".to_string()),
    ];
    if let Some(b) = bearer {
        h.push((
            "authorization".into(),
            if b.to_lowercase().starts_with("bearer ") {
                b
            } else {
                format!("Bearer {}", b)
            },
        ));
    } else if let Some(k) = api_key {
        h.push(("authorization".into(), format!("Bearer {}", k)));
    }
    h
}

fn user_agent_suffix() -> String {
    let pkg_version = env!("CARGO_PKG_VERSION");
    format!("ai-sdk/openai-compatible/{pkg_version}")
}

fn apply_user_agent_suffix(headers: &mut BTreeMap<String, String>) {
    let suffix = user_agent_suffix();
    let existing = headers.remove("user-agent").unwrap_or_default();
    let value = if existing.trim().is_empty() {
        suffix
    } else {
        format!("{existing} {suffix}")
    };
    headers.insert("user-agent".into(), value);
}

fn build_headers(
    def: &ProviderDefinition,
    api_key: Option<String>,
    bearer: Option<String>,
) -> Vec<(String, String)> {
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in default_headers_from_creds(api_key, bearer) {
        headers.insert(k.to_ascii_lowercase(), v);
    }
    for (k, v) in def.headers.iter() {
        let kl = k.to_ascii_lowercase();
        if kl == "content-type" || kl == "accept" || kl == "authorization" || kl == "x-api-key" {
            continue;
        }
        if sdkopt::is_internal_sdk_header(&kl) {
            continue;
        }
        headers.insert(kl, v.clone());
    }
    apply_user_agent_suffix(&mut headers);
    headers.into_iter().collect()
}

struct BaseConfig {
    base_url: String,
    headers: Vec<(String, String)>,
    http: crate::reqwest_transport::ReqwestTransport,
    transport_cfg: TransportConfig,
    query_params: Vec<(String, String)>,
    default_options: Option<v2t::ProviderOptions>,
}

fn build_base_config(
    def: &ProviderDefinition,
    creds: &Credentials,
) -> Result<BaseConfig, SdkError> {
    let api_key = creds.as_api_key();
    let bearer = creds.as_bearer();
    let base_url = def.base_url.trim().to_string();
    if base_url.is_empty() {
        return Err(SdkError::InvalidArgument {
            message: format!(
                "openai-compatible provider '{}' requires base_url",
                def.name
            ),
        });
    }
    let default_options = extract_default_options(def);
    let headers = build_headers(def, api_key, bearer);

    tracing::info!(
        "[PROVOPTS]: openai-compatible headers {:?}",
        def.headers.keys().collect::<Vec<_>>()
    );

    let mut cfg = TransportConfig::default();
    apply_stream_idle_timeout_ms(def, &mut cfg);
    let http =
        crate::reqwest_transport::ReqwestTransport::try_new(&cfg).map_err(SdkError::Transport)?;
    let query_params = def
        .query_params
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    Ok(BaseConfig {
        base_url,
        headers,
        http,
        transport_cfg: cfg,
        query_params,
        default_options,
    })
}

fn match_openai_compatible_chat(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAICompatibleChat)
}
fn match_openai_compatible(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAICompatible)
}
fn match_openai_compatible_completion(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAICompatibleCompletion)
}

fn build_openai_compatible(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let base = build_base_config(def, creds)?;
    let supported_urls = HashMap::from([(
        "text/*".to_string(),
        vec![
            r"^https?://.*/v1/completions$".to_string(),
            r"^https?://.*/v1/chat/completions$".to_string(),
        ],
    )]);

    // Provider settings from internal options header
    let (include_usage_flag, _supports_structured_outputs_flag) = parse_provider_settings(def);
    if let Some(defaults) = &base.default_options {
        tracing::info!(
            "[PROVOPTS]: openai-compatible default scopes {:?}",
            defaults.keys().collect::<Vec<_>>()
        );
    }
    let lm = OpenAICompatibleCompletionLanguageModel::new(
        model.to_string(),
        OpenAICompatibleCompletionConfig {
            provider_scope_name: def.name.clone(),
            base_url: base.base_url,
            headers: base.headers,
            http: base.http,
            transport_cfg: base.transport_cfg,
            include_usage: include_usage_flag,
            supported_urls,
            query_params: base.query_params,
            default_options: base.default_options,
        },
    );

    Ok(Arc::new(lm))
}

pub fn build_openai_compatible_chat(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let base = build_base_config(def, creds)?;
    let supported_urls = HashMap::from([(
        "text/*".to_string(),
        vec![r"^https?://.*/v1/chat/completions$".to_string()],
    )]);
    let (include_usage_flag, supports_structured_outputs_flag) = parse_provider_settings(def);
    tracing::info!(
        "[PROVOPTS]: openai-compatible include_usage={} supports_structured={}",
        include_usage_flag,
        supports_structured_outputs_flag
    );
    if let Some(defaults) = &base.default_options {
        tracing::info!(
            "[PROVOPTS]: openai-compatible default scopes {:?}",
            defaults.keys().collect::<Vec<_>>()
        );
    }

    let lm = OpenAICompatibleChatLanguageModel::new(
        model.to_string(),
        OpenAICompatibleChatConfig {
            provider_scope_name: def.name.clone(),
            base_url: base.base_url,
            headers: base.headers,
            http: base.http,
            transport_cfg: base.transport_cfg,
            include_usage: include_usage_flag,
            supported_urls,
            query_params: base.query_params,
            supports_structured_outputs: supports_structured_outputs_flag,
            default_options: base.default_options,
        },
    );
    Ok(Arc::new(lm))
}

pub fn build_openai_compatible_completion(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    build_openai_compatible(def, model, creds)
}

pub fn build_openai_compatible_embedding(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn EmbeddingModel>, SdkError> {
    let base = build_base_config(def, creds)?;
    let (max_embeddings_per_call, supports_parallel_calls) = parse_embedding_settings(def);

    let cfg = OpenAICompatibleEmbeddingConfig {
        provider_scope_name: def.name.clone(),
        base_url: base.base_url,
        headers: base.headers,
        http: base.http,
        transport_cfg: base.transport_cfg,
        query_params: base.query_params,
        max_embeddings_per_call,
        supports_parallel_calls,
        default_options: base.default_options,
    };

    Ok(Arc::new(OpenAICompatibleEmbeddingModel::new(
        model.to_string(),
        cfg,
    )))
}

pub fn build_openai_compatible_image(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn ImageModel>, SdkError> {
    let base = build_base_config(def, creds)?;
    let cfg = OpenAICompatibleImageConfig {
        provider_scope_name: def.name.clone(),
        base_url: base.base_url,
        headers: base.headers,
        http: base.http,
        transport_cfg: base.transport_cfg,
        query_params: base.query_params,
        default_options: base.default_options,
    };

    Ok(Arc::new(OpenAICompatibleImageModel::new(
        model.to_string(),
        cfg,
    )))
}

inventory::submit! {
    ProviderRegistration {
        id: "openai-compatible",
        sdk_type: SdkType::OpenAICompatible,
        matches: Some(match_openai_compatible),
        build: build_openai_compatible_chat,
        reasoning_scope: None,
    }
}

inventory::submit! {
    ProviderRegistration {
        id: "openai-compatible-chat",
        sdk_type: SdkType::OpenAICompatibleChat,
        matches: Some(match_openai_compatible_chat),
        build: build_openai_compatible_chat,
        reasoning_scope: None,
    }
}

inventory::submit! {
    ProviderRegistration {
        id: "openai-compatible-completion",
        sdk_type: SdkType::OpenAICompatibleCompletion,
        matches: Some(match_openai_compatible_completion),
        build: build_openai_compatible_completion,
        reasoning_scope: None,
    }
}

fn parse_provider_settings(def: &ProviderDefinition) -> (bool, bool) {
    // Defaults (match first-party OpenAI behaviour)
    let mut include_usage = true;
    let mut supports_structured = false;
    // Read internal options header if present
    if let Some(val) = sdkopt::extract_options_from_headers(
        &def.headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>(),
    ) {
        if let Some(section) = val
            .as_object()
            .and_then(|obj| obj.get(&def.name))
            .and_then(|v| v.as_object())
        {
            if let Some(b) = section.get("include_usage").and_then(|v| v.as_bool()) {
                include_usage = b;
            }
            if let Some(b) = section
                .get("supports_structured_outputs")
                .and_then(|v| v.as_bool())
            {
                supports_structured = b;
            }
        }
    }
    (include_usage, supports_structured)
}

fn extract_default_options(def: &ProviderDefinition) -> Option<v2t::ProviderOptions> {
    let header_pairs: Vec<(String, String)> = def
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    sdkopt::extract_options_from_headers(&header_pairs)
        .and_then(|raw| provider_defaults_from_json(&def.name, &raw))
}

fn parse_embedding_settings(def: &ProviderDefinition) -> (Option<usize>, bool) {
    let mut max_embeddings = Some(DEFAULT_MAX_EMBEDDINGS_PER_CALL);
    let mut supports_parallel_calls = true;

    if let Some(val) = sdkopt::extract_options_from_headers(
        &def.headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>(),
    ) {
        if let Some(section) = val
            .as_object()
            .and_then(|obj| obj.get(&def.name))
            .and_then(|v| v.as_object())
        {
            if let Some(n) = section
                .get("max_embeddings_per_call")
                .and_then(|v| v.as_u64())
                .and_then(|v| usize::try_from(v).ok())
            {
                max_embeddings = Some(n);
            }
            if let Some(b) = section
                .get("supports_parallel_calls")
                .and_then(|v| v.as_bool())
            {
                supports_parallel_calls = b;
            }
        }
    }

    (max_embeddings, supports_parallel_calls)
}
