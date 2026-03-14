use ai_sdk_rs::core::{EmbeddingModel, ImageModel, LanguageModel};
use ai_sdk_rs::providers::anthropic::AnthropicMessagesLanguageModel;
use ai_sdk_rs::providers::openai::OpenAIResponsesLanguageModel;
use ai_sdk_rs::providers::openai_compatible::{
    OpenAICompatibleChatLanguageModel, OpenAICompatibleCompletionLanguageModel,
    OpenAICompatibleEmbeddingModel, OpenAICompatibleImageModel,
};

#[test]
fn openai_root_model_builder_builds_the_canonical_typed_surface() {
    let model = OpenAIResponsesLanguageModel::builder("gpt-5")
        .with_base_url("https://proxy.example/v1")
        .with_api_key("test-key")
        .build()
        .expect("openai builder");

    assert_eq!(model.provider_name(), "OpenAI");
    assert_eq!(model.model_id(), "gpt-5");
    assert_eq!(model.config.provider_scope_name, "openai");
    assert_eq!(model.config.base_url, "https://proxy.example/v1");
}

#[test]
fn anthropic_root_model_builder_builds_the_canonical_typed_surface() {
    let model = AnthropicMessagesLanguageModel::builder("claude-sonnet-4-5-20250929")
        .with_api_key("test-key")
        .build()
        .expect("anthropic builder");

    assert_eq!(model.provider_name(), "anthropic.messages");
    assert_eq!(model.model_id(), "claude-sonnet-4-5-20250929");
}

#[test]
fn openai_compatible_chat_builder_builds_the_canonical_typed_surface() {
    let model = OpenAICompatibleChatLanguageModel::builder("gpt-4o-mini")
        .with_base_url("https://api.compat.example/v1")
        .with_api_key("test-key")
        .with_structured_outputs(true)
        .build()
        .expect("openai-compatible chat builder");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "gpt-4o-mini");
}

#[test]
fn openai_compatible_completion_builder_builds_the_canonical_typed_surface() {
    let model = OpenAICompatibleCompletionLanguageModel::builder("gpt-3.5-turbo-instruct")
        .with_base_url("https://api.compat.example/v1")
        .with_api_key("test-key")
        .build()
        .expect("openai-compatible completion builder");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "gpt-3.5-turbo-instruct");
}

#[test]
fn openai_compatible_embedding_builder_builds_the_canonical_typed_surface() {
    let model = OpenAICompatibleEmbeddingModel::builder("text-embedding-3-large")
        .with_base_url("https://api.compat.example/v1")
        .with_api_key("test-key")
        .build()
        .expect("openai-compatible embedding builder");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "text-embedding-3-large");
    assert!(model.supports_parallel_calls());
}

#[test]
fn openai_compatible_image_builder_builds_the_canonical_typed_surface() {
    let model = OpenAICompatibleImageModel::builder("gpt-image-1")
        .with_base_url("https://api.compat.example/v1")
        .with_api_key("test-key")
        .build()
        .expect("openai-compatible image builder");

    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "gpt-image-1");
    assert_eq!(model.max_images_per_call(), Some(10));
}
