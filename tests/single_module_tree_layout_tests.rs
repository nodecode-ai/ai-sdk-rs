use std::any::TypeId;

use ai_sdk_rs::core::{EmbeddingModel, ImageModel, LanguageModel, SdkError, TransportError};
use ai_sdk_rs::provider::{Credentials, ProviderBootstrapHeaders};
use ai_sdk_rs::providers::{
    anthropic::AnthropicMessagesLanguageModel,
    openai::config::OpenAIConfig,
    openai::responses::language_model::OpenAIResponsesLanguageModel,
    openai_compatible::{
        OpenAICompatibleChatLanguageModel, OpenAICompatibleEmbeddingModel,
        OpenAICompatibleImageModel, StreamMode,
    },
};
use ai_sdk_rs::transports::reqwest::ReqwestTransport;
use ai_sdk_rs::types::{self, ChatRequest, ContentPart, Role};

#[test]
fn public_surface_reexports_compile() {
    fn assert_language_model<T: LanguageModel>() {}
    fn assert_embedding_model<T: EmbeddingModel>() {}
    fn assert_image_model<T: ImageModel>() {}

    assert_language_model::<AnthropicMessagesLanguageModel>();
    assert_language_model::<OpenAIResponsesLanguageModel>();
    assert_language_model::<OpenAICompatibleChatLanguageModel>();
    assert_embedding_model::<OpenAICompatibleEmbeddingModel>();
    assert_image_model::<OpenAICompatibleImageModel>();

    let _ = TypeId::of::<Credentials>();
    let _ = TypeId::of::<ProviderBootstrapHeaders>();
    let _ = TypeId::of::<OpenAIConfig>();
    let _ = TypeId::of::<ReqwestTransport>();
    let _ = TypeId::of::<ai_sdk_rs::streaming_sse::SseDecoder>();
    let _ = TypeId::of::<SdkError>();
    let _ = TypeId::of::<TransportError>();
    let _ = TypeId::of::<ChatRequest>();
    let _ = TypeId::of::<ContentPart>();
    let _ = TypeId::of::<Role>();
    let _ = TypeId::of::<StreamMode>();
    let _ = TypeId::of::<types::ToolArguments>();
}

#[test]
fn lib_root_uses_direct_module_tree_without_pseudo_crate_aliases() {
    let lib_rs = include_str!("../src/lib.rs");

    for removed_mount in [
        r#"#[path = "../crates/core/src/lib.rs"]"#,
        r#"#[path = "../crates/provider/src/lib.rs"]"#,
        r#"#[path = "../crates/streaming-sse/src/lib.rs"]"#,
        r#"#[path = "../crates/transports/reqwest/src/lib.rs"]"#,
        r#"#[path = "../crates/sdk-types/src/lib.rs"]"#,
        r#"#[path = "../crates/providers/openai/src/lib.rs"]"#,
        r#"#[path = "../crates/providers/openai-compatible/src/lib.rs"]"#,
    ] {
        assert!(
            !lib_rs.contains(removed_mount),
            "expected src/lib.rs to stop path-mounting provider trees: {removed_mount}",
        );
    }

    for removed_alias in [
        "pub(crate) use crate::core as ai_sdk_core;",
        "pub(crate) use crate::provider as ai_sdk_provider;",
        "pub(crate) use crate::provider_openai as ai_sdk_providers_openai;",
        "pub(crate) use crate::streaming_sse as ai_sdk_streaming_sse;",
        "pub(crate) use crate::transport_reqwest as reqwest_transport;",
        "pub(crate) use crate::types as ai_sdk_types;",
    ] {
        assert!(
            !lib_rs.contains(removed_alias),
            "expected src/lib.rs to stop exporting pseudo-crate aliases: {removed_alias}",
        );
    }

    assert!(
        lib_rs.contains("pub mod providers;"),
        "expected src/lib.rs to expose the real provider module tree",
    );
}
