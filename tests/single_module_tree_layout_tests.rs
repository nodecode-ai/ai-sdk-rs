use std::any::TypeId;
use std::path::Path;

use ai_sdk_rs::core::{types as v2t, EmbeddingModel, ImageModel, LanguageModel, SdkError, TransportError};
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
use ai_sdk_rs::types::{self, ContentPart};

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
    let _ = TypeId::of::<v2t::CallOptions>();
    let _ = TypeId::of::<v2t::PromptMessage>();
    let _ = TypeId::of::<v2t::StreamPart>();
    let _ = TypeId::of::<ContentPart>();
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

#[test]
fn types_module_no_longer_defines_the_legacy_request_dsl() {
    let types_mod = include_str!("../src/types/mod.rs");

    for removed_surface in [
        "pub enum Role",
        "pub struct ChatMessage",
        "pub struct ToolSpec",
        "pub struct ChatRequest",
        "Role as MessageRole",
    ] {
        assert!(
            !types_mod.contains(removed_surface),
            "expected src/types/mod.rs to stop exporting the legacy request DSL surface: {removed_surface}",
        );
    }

    assert!(
        types_mod.contains("pub mod v2;"),
        "expected src/types/mod.rs to keep the v2 module as the surviving request owner",
    );
}

#[test]
fn v2_call_options_form_the_surviving_request_authority() {
    let options = v2t::CallOptions {
        prompt: vec![
            v2t::PromptMessage::System {
                content: "Keep answers brief.".into(),
                provider_options: None,
            },
            v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "Why is the sky blue?".into(),
                    provider_options: None,
                }],
                provider_options: None,
            },
        ],
        max_output_tokens: Some(256),
        ..Default::default()
    };

    assert_eq!(options.prompt.len(), 2);
    assert_eq!(options.max_output_tokens, Some(256));

    match &options.prompt[0] {
        v2t::PromptMessage::System { content, .. } => {
            assert_eq!(content, "Keep answers brief.");
        }
        other => panic!("expected system prompt, got {other:?}"),
    }

    match &options.prompt[1] {
        v2t::PromptMessage::User { content, .. } => {
            assert_eq!(content.len(), 1);
            match &content[0] {
                v2t::UserPart::Text { text, .. } => {
                    assert_eq!(text, "Why is the sky blue?");
                }
                other => panic!("expected text user part, got {other:?}"),
            }
        }
        other => panic!("expected user prompt, got {other:?}"),
    }
}

#[test]
fn quick_start_points_at_workspace_validated_v2_examples() {
    let readme = include_str!("../README.md");
    let cargo_toml = include_str!("../Cargo.toml");
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    assert!(
        readme.contains("cargo run -p generate-text"),
        "expected README quick start to advertise the generate-text package",
    );
    assert!(
        readme.contains("cargo run -p generate-stream"),
        "expected README quick start to keep the generate-stream package on the surviving v2 path",
    );
    assert!(
        cargo_toml.contains("\"examples/generate-text\""),
        "expected the workspace members list to include generate-text after the v2 migration lands",
    );
    assert!(
        repo_root.join("examples/generate-text/Cargo.toml").exists(),
        "expected the generate-text example package to exist on disk",
    );
    assert!(
        !repo_root.join("examples/anthropic-thinking/Cargo.toml").exists(),
        "expected the orphaned anthropic-thinking package to be removed",
    );
    assert!(
        !repo_root
            .join("examples/anthropic-hello-twice/Cargo.toml")
            .exists(),
        "expected the orphaned anthropic-hello-twice package to be removed",
    );
    assert!(
        !repo_root.join("examples/github-models.rs").exists(),
        "expected the orphaned github-models example to be removed",
    );
}
