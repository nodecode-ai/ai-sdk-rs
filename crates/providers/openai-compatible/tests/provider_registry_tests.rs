use crate::ai_sdk_provider::{registry, sdk_type_from_id, Credentials};
use crate::ai_sdk_types::catalog::{ModelInfo, ProviderDefinition, SdkType};
use std::collections::HashMap;

const OPENAI_COMPATIBLE_PROVIDER_ALIASES: &[&str] = &[
    "xai",
    "deepseek",
    "mistral",
    "togetherai",
    "fireworks-ai",
    "deepinfra",
    "openrouter",
    "perplexity",
];

fn openai_compatible_def(name: &str) -> ProviderDefinition {
    let models: HashMap<String, ModelInfo> = HashMap::new();
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
        models,
        preserve_model_prefix: true,
    }
}

#[test]
fn openai_compatible_registry_maps_to_chat_builder() {
    let reg = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("openai-compatible"))
        .expect("openai-compatible registration");
    assert!(matches!(reg.sdk_type, SdkType::OpenAICompatible));

    let models: HashMap<String, ModelInfo> = HashMap::new();
    let def = ProviderDefinition {
        name: "compat".into(),
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
        models,
        preserve_model_prefix: true,
    };

    let model = (reg.build)(&def, "gpt-4o", &Credentials::None).expect("build model");
    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "gpt-4o");
}

#[test]
fn groq_registry_maps_to_openai_compatible_chat_builder() {
    let reg = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("groq"))
        .expect("groq registration");
    assert!(matches!(reg.sdk_type, SdkType::Groq));

    let models: HashMap<String, ModelInfo> = HashMap::new();
    let def = ProviderDefinition {
        name: "groq".into(),
        display_name: "Groq".into(),
        sdk_type: SdkType::Groq,
        base_url: "https://api.groq.com/openai/v1".into(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/chat/completions".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models,
        preserve_model_prefix: true,
    };

    let model =
        (reg.build)(&def, "llama-3.3-70b-versatile", &Credentials::None).expect("build model");
    assert_eq!(model.provider_name(), "openai-compatible");
    assert_eq!(model.model_id(), "llama-3.3-70b-versatile");
}

#[test]
fn openai_compatible_aliases_resolve_to_sdk_type() {
    for alias in OPENAI_COMPATIBLE_PROVIDER_ALIASES {
        assert_eq!(
            sdk_type_from_id(alias),
            Some(SdkType::OpenAICompatible),
            "alias should resolve to sdk_type=openai-compatible: {alias}"
        );
    }
}

#[test]
fn openai_compatible_aliases_build_with_openai_compatible_registration() {
    for alias in OPENAI_COMPATIBLE_PROVIDER_ALIASES {
        let reg = registry::iter()
            .into_iter()
            .find(|entry| entry.id.eq_ignore_ascii_case(alias))
            .unwrap_or_else(|| panic!("missing registration for alias {alias}"));

        assert!(matches!(reg.sdk_type, SdkType::OpenAICompatible));

        let def = openai_compatible_def(alias);
        let model = (reg.build)(&def, "gpt-4o-mini", &Credentials::None)
            .unwrap_or_else(|_| panic!("expected alias {alias} to build openai-compatible model"));
        assert_eq!(model.provider_name(), "openai-compatible");
        assert_eq!(model.model_id(), "gpt-4o-mini");
    }
}
