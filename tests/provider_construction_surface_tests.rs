use ai_sdk_rs::core::transport::TransportConfig;
use ai_sdk_rs::core::{EmbeddingModel, LanguageModel};
use ai_sdk_rs::provider::{registry, Credentials};
use ai_sdk_rs::providers::anthropic::AnthropicMessagesLanguageModel;
use ai_sdk_rs::providers::openai::OpenAIResponsesLanguageModel;
use ai_sdk_rs::providers::openai_compatible::OpenAICompatibleEmbeddingModel;
use ai_sdk_rs::types::catalog::{ModelInfo, ProviderDefinition, SdkType};
use std::collections::HashMap;

fn empty_models() -> HashMap<String, ModelInfo> {
    HashMap::new()
}

#[test]
fn openai_builder_builds_the_language_model_used_by_examples() {
    let model = OpenAIResponsesLanguageModel::builder("gpt-4o")
        .with_base_url("https://api.openai.com/v1")
        .with_api_key("test-key")
        .build()
        .expect("build openai model");

    assert_eq!(model.provider_name(), "OpenAI");
    assert_eq!(model.model_id(), "gpt-4o");
}

#[test]
fn anthropic_builder_builds_a_language_model_without_the_registry() {
    let model = AnthropicMessagesLanguageModel::builder("claude-sonnet-4-5-20250929")
        .with_api_key("test-key")
        .build()
        .expect("build anthropic model");

    assert_eq!(model.provider_name(), "anthropic.messages");
    assert_eq!(model.model_id(), "claude-sonnet-4-5-20250929");
}

#[test]
fn openai_compatible_embedding_builder_replaces_manual_config_assembly() {
    let model = OpenAICompatibleEmbeddingModel::builder("qwen3-embedding-0-6b")
        .with_base_url("https://api.compat.example/v1")
        .with_api_key("test-key")
        .with_transport_config(TransportConfig::default())
        .build()
        .expect("build openai-compatible embedding model");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "qwen3-embedding-0-6b");
    assert_eq!(model.max_embeddings_per_call(), Some(2048));
}

#[test]
fn registry_construction_remains_available_for_dynamic_openai_compatible_definitions() {
    let registration = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("openai-compatible"))
        .expect("openai-compatible registration");

    let definition = ProviderDefinition {
        name: "openai-compatible".into(),
        display_name: "Compat".into(),
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
        models: empty_models(),
        preserve_model_prefix: true,
    };

    let model = (registration.build)(
        &definition,
        "gpt-4o-mini",
        &Credentials::ApiKey("test-key".into()),
    )
    .expect("build openai-compatible model");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "gpt-4o-mini");
}
