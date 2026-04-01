use std::any::TypeId;
use std::path::Path;

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
use ai_sdk_rs::types::{self, ChatMessage, ChatRequest, ContentPart, Role, ToolSpec};
use serde_json::json;

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
    let _ = TypeId::of::<ChatMessage>();
    let _ = TypeId::of::<ChatRequest>();
    let _ = TypeId::of::<ContentPart>();
    let _ = TypeId::of::<Role>();
    let _ = TypeId::of::<StreamMode>();
    let _ = TypeId::of::<ToolSpec>();
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
fn legacy_request_dsl_still_forms_a_public_request_authority() {
    let mut request = ChatRequest::new(
        "openai/gpt-5",
        vec![
            ChatMessage::system("Keep answers brief."),
            ChatMessage::user("Why is the sky blue?"),
        ],
    );
    request.tools.push(ToolSpec {
        name: "lookup".into(),
        description: Some("Fetch extra context".into()),
        json_schema: json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        }),
    });

    assert_eq!(request.model, "openai/gpt-5");
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.messages[0].role, Role::System);
    assert_eq!(request.messages[1].role, Role::User);
    assert_eq!(request.messages[0].text(), "Keep answers brief.");
    assert_eq!(request.messages[1].text(), "Why is the sky blue?");
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "lookup");
}

#[test]
fn quick_start_still_points_at_a_non_workspace_generate_text_example() {
    let readme = include_str!("../README.md");
    let cargo_toml = include_str!("../Cargo.toml");
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    assert!(
        readme.contains("cargo run -p generate-text"),
        "expected README quick start to advertise the generate-text package",
    );
    assert!(
        !cargo_toml.contains("\"examples/generate-text\""),
        "expected the workspace members list to exclude generate-text before the v2 migration lands",
    );
    assert!(
        repo_root.join("examples/generate-text/Cargo.toml").exists(),
        "expected the generate-text example package to exist on disk for this mismatch check",
    );
}
