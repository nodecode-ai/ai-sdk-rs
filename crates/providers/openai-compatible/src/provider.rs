use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::ai_sdk_core::options as sdkopt;
use crate::ai_sdk_core::request_builder::defaults::provider_defaults_from_json;
use crate::ai_sdk_core::transport::TransportConfig;
use crate::ai_sdk_core::{EmbeddingModel, ImageModel, LanguageModel, SdkError};
use crate::ai_sdk_provider::{
    build_provider_transport_config, collect_query_params, registry::ProviderRegistration,
    Credentials,
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

fn build_headers_from_pairs(
    header_pairs: &[(String, String)],
    api_key: Option<String>,
    bearer: Option<String>,
) -> Vec<(String, String)> {
    let mut headers: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in default_headers_from_creds(api_key, bearer) {
        headers.insert(k.to_ascii_lowercase(), v);
    }
    for (k, v) in header_pairs.iter() {
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

fn build_base_config_from_parts(
    provider_scope_name: &str,
    base_url: String,
    header_pairs: Vec<(String, String)>,
    api_key: Option<String>,
    bearer: Option<String>,
    transport_cfg: TransportConfig,
    query_params: Vec<(String, String)>,
    default_options: Option<v2t::ProviderOptions>,
) -> Result<BaseConfig, SdkError> {
    let base_url = base_url.trim().to_string();
    if base_url.is_empty() {
        return Err(SdkError::InvalidArgument {
            message: format!(
                "openai-compatible provider '{}' requires base_url",
                provider_scope_name
            ),
        });
    }
    let headers = build_headers_from_pairs(&header_pairs, api_key, bearer);
    let http = crate::reqwest_transport::ReqwestTransport::try_new(&transport_cfg)
        .map_err(SdkError::Transport)?;

    Ok(BaseConfig {
        base_url,
        headers,
        http,
        transport_cfg,
        query_params,
        default_options,
    })
}

#[derive(Clone, Debug)]
struct OpenAICompatibleBuilderBase {
    model_id: String,
    provider_scope_name: String,
    base_url: Option<String>,
    api_key: Option<String>,
    bearer: Option<String>,
    headers: Vec<(String, String)>,
    query_params: Vec<(String, String)>,
    transport_cfg: TransportConfig,
    default_options: Option<v2t::ProviderOptions>,
}

impl OpenAICompatibleBuilderBase {
    fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            provider_scope_name: "openai-compatible".into(),
            base_url: None,
            api_key: None,
            bearer: None,
            headers: Vec::new(),
            query_params: Vec::new(),
            transport_cfg: TransportConfig::default(),
            default_options: None,
        }
    }

    fn with_provider_scope_name(mut self, provider_scope_name: impl Into<String>) -> Self {
        self.provider_scope_name = provider_scope_name.into();
        self
    }

    fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    fn with_bearer(mut self, bearer: impl Into<String>) -> Self {
        self.bearer = Some(bearer.into());
        self
    }

    fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((key.into(), value.into()));
        self
    }

    fn with_headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers
            .extend(headers.into_iter().map(|(key, value)| (key.into(), value.into())));
        self
    }

    fn with_query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query_params.push((key.into(), value.into()));
        self
    }

    fn with_query_params<I, K, V>(mut self, query_params: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.query_params.extend(
            query_params
                .into_iter()
                .map(|(key, value)| (key.into(), value.into())),
        );
        self
    }

    fn with_transport_config(mut self, transport_cfg: TransportConfig) -> Self {
        self.transport_cfg = transport_cfg;
        self
    }

    fn with_default_options(mut self, default_options: v2t::ProviderOptions) -> Self {
        self.default_options = Some(default_options);
        self
    }

    fn build(
        self,
    ) -> Result<(String, String, BaseConfig), SdkError> {
        let provider_scope_name = self.provider_scope_name;
        let model_id = self.model_id;
        let base = build_base_config_from_parts(
            &provider_scope_name,
            self.base_url.unwrap_or_default(),
            self.headers,
            self.api_key,
            self.bearer,
            self.transport_cfg,
            self.query_params,
            self.default_options,
        )?;
        Ok((model_id, provider_scope_name, base))
    }
}

macro_rules! impl_openai_compatible_builder_common {
    ($name:ident) => {
        impl $name {
            pub fn with_provider_scope_name(
                mut self,
                provider_scope_name: impl Into<String>,
            ) -> Self {
                self.base = self.base.with_provider_scope_name(provider_scope_name);
                self
            }

            pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
                self.base = self.base.with_base_url(base_url);
                self
            }

            pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
                self.base = self.base.with_api_key(api_key);
                self
            }

            pub fn with_bearer(mut self, bearer: impl Into<String>) -> Self {
                self.base = self.base.with_bearer(bearer);
                self
            }

            pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
                self.base = self.base.with_header(key, value);
                self
            }

            pub fn with_headers<I, K, V>(mut self, headers: I) -> Self
            where
                I: IntoIterator<Item = (K, V)>,
                K: Into<String>,
                V: Into<String>,
            {
                self.base = self.base.with_headers(headers);
                self
            }

            pub fn with_query_param(
                mut self,
                key: impl Into<String>,
                value: impl Into<String>,
            ) -> Self {
                self.base = self.base.with_query_param(key, value);
                self
            }

            pub fn with_query_params<I, K, V>(mut self, query_params: I) -> Self
            where
                I: IntoIterator<Item = (K, V)>,
                K: Into<String>,
                V: Into<String>,
            {
                self.base = self.base.with_query_params(query_params);
                self
            }

            pub fn with_transport_config(mut self, transport_cfg: TransportConfig) -> Self {
                self.base = self.base.with_transport_config(transport_cfg);
                self
            }

            pub fn with_default_options(
                mut self,
                default_options: v2t::ProviderOptions,
            ) -> Self {
                self.base = self.base.with_default_options(default_options);
                self
            }
        }
    };
}

#[derive(Clone, Debug)]
pub struct OpenAICompatibleChatBuilder {
    base: OpenAICompatibleBuilderBase,
    include_usage: bool,
    supports_structured_outputs: bool,
}

impl OpenAICompatibleChatBuilder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            base: OpenAICompatibleBuilderBase::new(model_id),
            include_usage: true,
            supports_structured_outputs: false,
        }
    }

    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = include_usage;
        self
    }

    pub fn with_structured_outputs(mut self, supports_structured_outputs: bool) -> Self {
        self.supports_structured_outputs = supports_structured_outputs;
        self
    }

    pub fn build(
        self,
    ) -> Result<OpenAICompatibleChatLanguageModel<crate::reqwest_transport::ReqwestTransport>, SdkError>
    {
        let (model_id, provider_scope_name, base) = self.base.build()?;
        Ok(OpenAICompatibleChatLanguageModel::new(
            model_id,
            OpenAICompatibleChatConfig {
                provider_scope_name,
                base_url: base.base_url,
                headers: base.headers,
                http: base.http,
                transport_cfg: base.transport_cfg,
                include_usage: self.include_usage,
                supported_urls: HashMap::from([(
                    "text/*".to_string(),
                    vec![r"^https?://.*/v1/chat/completions$".to_string()],
                )]),
                query_params: base.query_params,
                supports_structured_outputs: self.supports_structured_outputs,
                default_options: base.default_options,
            },
        ))
    }
}

impl_openai_compatible_builder_common!(OpenAICompatibleChatBuilder);

#[derive(Clone, Debug)]
pub struct OpenAICompatibleCompletionBuilder {
    base: OpenAICompatibleBuilderBase,
    include_usage: bool,
}

impl OpenAICompatibleCompletionBuilder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            base: OpenAICompatibleBuilderBase::new(model_id),
            include_usage: true,
        }
    }

    pub fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = include_usage;
        self
    }

    pub fn build(
        self,
    ) -> Result<OpenAICompatibleCompletionLanguageModel<crate::reqwest_transport::ReqwestTransport>, SdkError>
    {
        let (model_id, provider_scope_name, base) = self.base.build()?;
        Ok(OpenAICompatibleCompletionLanguageModel::new(
            model_id,
            OpenAICompatibleCompletionConfig {
                provider_scope_name,
                base_url: base.base_url,
                headers: base.headers,
                http: base.http,
                transport_cfg: base.transport_cfg,
                include_usage: self.include_usage,
                supported_urls: HashMap::from([(
                    "text/*".to_string(),
                    vec![
                        r"^https?://.*/v1/completions$".to_string(),
                        r"^https?://.*/v1/chat/completions$".to_string(),
                    ],
                )]),
                query_params: base.query_params,
                default_options: base.default_options,
            },
        ))
    }
}

impl_openai_compatible_builder_common!(OpenAICompatibleCompletionBuilder);

#[derive(Clone, Debug)]
pub struct OpenAICompatibleEmbeddingBuilder {
    base: OpenAICompatibleBuilderBase,
    max_embeddings_per_call: Option<usize>,
    supports_parallel_calls: bool,
}

impl OpenAICompatibleEmbeddingBuilder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            base: OpenAICompatibleBuilderBase::new(model_id),
            max_embeddings_per_call: Some(DEFAULT_MAX_EMBEDDINGS_PER_CALL),
            supports_parallel_calls: true,
        }
    }

    pub fn with_max_embeddings_per_call(
        mut self,
        max_embeddings_per_call: Option<usize>,
    ) -> Self {
        self.max_embeddings_per_call = max_embeddings_per_call;
        self
    }

    pub fn with_supports_parallel_calls(mut self, supports_parallel_calls: bool) -> Self {
        self.supports_parallel_calls = supports_parallel_calls;
        self
    }

    pub fn build(
        self,
    ) -> Result<OpenAICompatibleEmbeddingModel<crate::reqwest_transport::ReqwestTransport>, SdkError>
    {
        let (model_id, provider_scope_name, base) = self.base.build()?;
        Ok(OpenAICompatibleEmbeddingModel::new(
            model_id,
            OpenAICompatibleEmbeddingConfig {
                provider_scope_name,
                base_url: base.base_url,
                headers: base.headers,
                http: base.http,
                transport_cfg: base.transport_cfg,
                query_params: base.query_params,
                max_embeddings_per_call: self.max_embeddings_per_call,
                supports_parallel_calls: self.supports_parallel_calls,
                default_options: base.default_options,
            },
        ))
    }
}

impl_openai_compatible_builder_common!(OpenAICompatibleEmbeddingBuilder);

#[derive(Clone, Debug)]
pub struct OpenAICompatibleImageBuilder {
    base: OpenAICompatibleBuilderBase,
}

impl OpenAICompatibleImageBuilder {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            base: OpenAICompatibleBuilderBase::new(model_id),
        }
    }

    pub fn build(
        self,
    ) -> Result<OpenAICompatibleImageModel<crate::reqwest_transport::ReqwestTransport>, SdkError>
    {
        let (model_id, provider_scope_name, base) = self.base.build()?;
        Ok(OpenAICompatibleImageModel::new(
            model_id,
            OpenAICompatibleImageConfig {
                provider_scope_name,
                base_url: base.base_url,
                headers: base.headers,
                http: base.http,
                transport_cfg: base.transport_cfg,
                query_params: base.query_params,
                default_options: base.default_options,
            },
        ))
    }
}

impl_openai_compatible_builder_common!(OpenAICompatibleImageBuilder);

fn match_openai_compatible_chat(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAICompatibleChat)
}
fn match_openai_compatible(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAICompatible)
}
fn match_groq(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::Groq)
}
fn match_openai_compatible_completion(def: &ProviderDefinition) -> bool {
    matches!(def.sdk_type, SdkType::OpenAICompatibleCompletion)
}

fn build_openai_compatible(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let (include_usage_flag, _supports_structured_outputs_flag) = parse_provider_settings(def);
    let mut builder = OpenAICompatibleCompletionBuilder::new(model)
        .with_provider_scope_name(def.name.clone())
        .with_base_url(def.base_url.clone())
        .with_headers(
            def.headers
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        )
        .with_query_params(collect_query_params(def))
        .with_transport_config(build_provider_transport_config(def, None))
        .with_include_usage(include_usage_flag);
    if let Some(default_options) = extract_default_options(def) {
        builder = builder.with_default_options(default_options);
    }
    if let Some(bearer) = creds.as_bearer() {
        builder = builder.with_bearer(bearer);
    } else if let Some(api_key) = creds.as_api_key() {
        builder = builder.with_api_key(api_key);
    }
    Ok(Arc::new(builder.build()?))
}

pub fn build_openai_compatible_chat(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn LanguageModel>, SdkError> {
    let (include_usage_flag, supports_structured_outputs_flag) = parse_provider_settings(def);
    tracing::info!(
        "[PROVOPTS]: openai-compatible include_usage={} supports_structured={}",
        include_usage_flag,
        supports_structured_outputs_flag
    );
    let mut builder = OpenAICompatibleChatBuilder::new(model)
        .with_provider_scope_name(def.name.clone())
        .with_base_url(def.base_url.clone())
        .with_headers(
            def.headers
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        )
        .with_query_params(collect_query_params(def))
        .with_transport_config(build_provider_transport_config(def, None))
        .with_include_usage(include_usage_flag)
        .with_structured_outputs(supports_structured_outputs_flag);
    if let Some(default_options) = extract_default_options(def) {
        builder = builder.with_default_options(default_options);
    }
    if let Some(bearer) = creds.as_bearer() {
        builder = builder.with_bearer(bearer);
    } else if let Some(api_key) = creds.as_api_key() {
        builder = builder.with_api_key(api_key);
    }
    Ok(Arc::new(builder.build()?))
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
    let (max_embeddings_per_call, supports_parallel_calls) = parse_embedding_settings(def);
    let mut builder = OpenAICompatibleEmbeddingBuilder::new(model)
        .with_provider_scope_name(def.name.clone())
        .with_base_url(def.base_url.clone())
        .with_headers(
            def.headers
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        )
        .with_query_params(collect_query_params(def))
        .with_transport_config(build_provider_transport_config(def, None))
        .with_max_embeddings_per_call(max_embeddings_per_call)
        .with_supports_parallel_calls(supports_parallel_calls);
    if let Some(default_options) = extract_default_options(def) {
        builder = builder.with_default_options(default_options);
    }
    if let Some(bearer) = creds.as_bearer() {
        builder = builder.with_bearer(bearer);
    } else if let Some(api_key) = creds.as_api_key() {
        builder = builder.with_api_key(api_key);
    }
    Ok(Arc::new(builder.build()?))
}

pub fn build_openai_compatible_image(
    def: &ProviderDefinition,
    model: &str,
    creds: &Credentials,
) -> Result<Arc<dyn ImageModel>, SdkError> {
    let mut builder = OpenAICompatibleImageBuilder::new(model)
        .with_provider_scope_name(def.name.clone())
        .with_base_url(def.base_url.clone())
        .with_headers(
            def.headers
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        )
        .with_query_params(collect_query_params(def))
        .with_transport_config(build_provider_transport_config(def, None));
    if let Some(default_options) = extract_default_options(def) {
        builder = builder.with_default_options(default_options);
    }
    if let Some(bearer) = creds.as_bearer() {
        builder = builder.with_bearer(bearer);
    } else if let Some(api_key) = creds.as_api_key() {
        builder = builder.with_api_key(api_key);
    }
    Ok(Arc::new(builder.build()?))
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
        id: "groq",
        sdk_type: SdkType::Groq,
        matches: Some(match_groq),
        build: build_openai_compatible_chat,
        reasoning_scope: None,
    }
}

macro_rules! register_openai_compatible_alias {
    ($id:literal) => {
        inventory::submit! {
            ProviderRegistration {
                id: $id,
                sdk_type: SdkType::OpenAICompatible,
                matches: Some(match_openai_compatible),
                build: build_openai_compatible_chat,
                reasoning_scope: None,
            }
        }
    };
}

register_openai_compatible_alias!("xai");
register_openai_compatible_alias!("deepseek");
register_openai_compatible_alias!("mistral");
register_openai_compatible_alias!("togetherai");
register_openai_compatible_alias!("fireworks-ai");
register_openai_compatible_alias!("deepinfra");
register_openai_compatible_alias!("openrouter");
register_openai_compatible_alias!("perplexity");

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
