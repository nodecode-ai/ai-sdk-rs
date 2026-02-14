use crate::ai_sdk_provider::{registry, Credentials};
use crate::ai_sdk_types::catalog::{ModelInfo, ProviderDefinition, SdkType};
use std::collections::HashMap;

#[test]
fn azure_registry_maps_to_responses_builder() {
    let reg = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("azure"))
        .expect("azure registration");
    assert!(matches!(reg.sdk_type, SdkType::Azure));

    let models: HashMap<String, ModelInfo> = HashMap::new();
    let def = ProviderDefinition {
        name: "azure".into(),
        display_name: "Azure".into(),
        sdk_type: SdkType::Azure,
        base_url: "https://example.openai.azure.com/openai".into(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/v1/responses".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models,
        preserve_model_prefix: true,
    };

    let model = (reg.build)(&def, "gpt-4.1-mini", &Credentials::ApiKey("test-key".into()))
        .expect("build model");
    assert_eq!(model.provider_name(), "OpenAI");
    assert_eq!(model.model_id(), "gpt-4.1-mini");
}
