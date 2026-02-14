use crate::ai_sdk_provider::{registry, Credentials};
use crate::ai_sdk_types::catalog::{ModelInfo, ProviderDefinition, SdkType};
use std::collections::HashMap;

#[test]
fn gateway_registry_maps_to_language_model_builder() {
    let reg = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("gateway"))
        .expect("gateway registration");
    assert!(matches!(reg.sdk_type, SdkType::Gateway));

    let models: HashMap<String, ModelInfo> = HashMap::new();
    let def = ProviderDefinition {
        name: "gateway".into(),
        display_name: "Gateway".into(),
        sdk_type: SdkType::Gateway,
        base_url: "https://ai-gateway.vercel.sh/v1/ai".into(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/language-model".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models,
        preserve_model_prefix: true,
    };

    let model = (reg.build)(&def, "openai/gpt-4.1-mini", &Credentials::ApiKey("test-key".into()))
        .expect("build model");
    assert_eq!(model.provider_name(), "gateway");
    assert_eq!(model.model_id(), "openai/gpt-4.1-mini");
}
