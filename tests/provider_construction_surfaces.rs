use std::collections::HashMap;

use ai_sdk_rs::core::transport::TransportConfig;
use ai_sdk_rs::core::{EmbeddingModel, LanguageModel};
use ai_sdk_rs::provider::{registry, Credentials};
use ai_sdk_rs::providers::anthropic::messages::language_model::{
    AnthropicMessagesConfig, AnthropicMessagesLanguageModel,
};
use ai_sdk_rs::providers::openai::responses::language_model::OpenAIResponsesLanguageModel;
use ai_sdk_rs::providers::openai_compatible::embedding::embedding_model::{
    OpenAICompatibleEmbeddingConfig, OpenAICompatibleEmbeddingModel,
};
use ai_sdk_rs::providers::openai_compatible::provider::build_openai_compatible_embedding;
use ai_sdk_rs::transports::reqwest::ReqwestTransport;
use ai_sdk_rs::types::catalog::{ProviderDefinition, SdkType};

fn openai_definition() -> ProviderDefinition {
    ProviderDefinition {
        name: "openai".into(),
        display_name: "OpenAI".into(),
        sdk_type: SdkType::OpenAI,
        base_url: "https://api.openai.com/v1".into(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/responses".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models: HashMap::new(),
        preserve_model_prefix: true,
    }
}

fn anthropic_definition() -> ProviderDefinition {
    ProviderDefinition {
        name: "anthropic".into(),
        display_name: "Anthropic".into(),
        sdk_type: SdkType::Anthropic,
        base_url: "https://api.anthropic.com/v1".into(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models: HashMap::new(),
        preserve_model_prefix: true,
    }
}

fn openai_compatible_definition(name: &str) -> ProviderDefinition {
    ProviderDefinition {
        name: name.into(),
        display_name: format!("{name} Provider"),
        sdk_type: SdkType::OpenAICompatible,
        base_url: "https://api.compat.example/v1".into(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/chat/completions".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models: HashMap::new(),
        preserve_model_prefix: true,
    }
}

#[test]
fn openai_create_simple_is_the_current_direct_happy_path_surface() {
    let model = OpenAIResponsesLanguageModel::create_simple(
        "gpt-5",
        Some("https://proxy.example/v1".into()),
        "test-key".into(),
    );

    assert_eq!(model.provider_name(), "OpenAI");
    assert_eq!(model.model_id(), "gpt-5");
    assert_eq!(model.config.provider_scope_name, "openai");
    assert_eq!(model.config.base_url, "https://proxy.example/v1");
    assert_eq!(model.config.endpoint_path, "/responses");
}

#[test]
fn openai_registry_build_remains_a_separate_definition_driven_surface() {
    let registration = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("openai"))
        .expect("openai registration");

    let model = (registration.build)(
        &openai_definition(),
        "gpt-4.1-mini",
        &Credentials::ApiKey("test-key".into()),
    )
    .expect("openai registry build");

    assert_eq!(model.provider_name(), "OpenAI");
    assert_eq!(model.model_id(), "gpt-4.1-mini");
}

#[test]
fn anthropic_messages_new_is_the_current_typed_construction_surface() {
    let model = AnthropicMessagesLanguageModel::new(
        "claude-sonnet-4-5-20250929".to_string(),
        AnthropicMessagesConfig {
            provider_name: "anthropic.messages",
            provider_scope_name: "anthropic".into(),
            base_url: "https://api.anthropic.com/v1".into(),
            headers: vec![("x-api-key".into(), "test-key".into())],
            http: ReqwestTransport::default(),
            transport_cfg: TransportConfig::default(),
            supported_urls: HashMap::from([(
                "image/*".to_string(),
                vec![r"^https?://.*$".to_string()],
            )]),
            default_options: None,
        },
    );

    assert_eq!(model.provider_name(), "anthropic.messages");
    assert_eq!(model.model_id(), "claude-sonnet-4-5-20250929");
    assert_eq!(
        model.supported_urls(),
        HashMap::from([("image/*".to_string(), vec![r"^https?://.*$".to_string()],)])
    );
}

#[test]
fn anthropic_registry_build_remains_a_separate_definition_driven_surface() {
    let registration = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("anthropic"))
        .expect("anthropic registration");

    let model = (registration.build)(
        &anthropic_definition(),
        "claude-sonnet-4-5-20250929",
        &Credentials::ApiKey("test-key".into()),
    )
    .expect("anthropic registry build");

    assert_eq!(model.provider_name(), "anthropic.messages");
    assert_eq!(model.model_id(), "claude-sonnet-4-5-20250929");
}

#[test]
fn openai_compatible_embedding_new_is_the_current_direct_typed_surface() {
    let model = OpenAICompatibleEmbeddingModel::new(
        "text-embedding-3-large",
        OpenAICompatibleEmbeddingConfig {
            provider_scope_name: "openai-compatible".into(),
            base_url: "https://api.compat.example/v1".into(),
            headers: vec![("authorization".into(), "Bearer test-key".into())],
            http: ReqwestTransport::default(),
            transport_cfg: TransportConfig::default(),
            query_params: vec![],
            max_embeddings_per_call: Some(128),
            supports_parallel_calls: true,
            default_options: None,
        },
    );

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "text-embedding-3-large");
    assert_eq!(model.max_embeddings_per_call(), Some(128));
    assert!(model.supports_parallel_calls());
}

#[test]
fn openai_compatible_embedding_builder_remains_definition_driven() {
    let model = build_openai_compatible_embedding(
        &openai_compatible_definition("openai-compatible"),
        "text-embedding-3-large",
        &Credentials::ApiKey("test-key".into()),
    )
    .expect("openai-compatible embedding build");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "text-embedding-3-large");
}
