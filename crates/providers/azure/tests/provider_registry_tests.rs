use super::provider::resolve_base_prefix;
use crate::provider::{registry, Credentials};
use crate::types::catalog::{ModelInfo, ProviderDefinition, SdkType};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const TEST_ENV_VARS: &[&str] = &[
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_RESOURCE_NAME",
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
];

struct EnvVarGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvVarGuard {
    fn capture(keys: &'static [&'static str]) -> Self {
        let saved = keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in keys {
            std::env::remove_var(key);
        }
        Self { saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn azure_definition() -> ProviderDefinition {
    ProviderDefinition {
        name: "azure".into(),
        display_name: "Azure".into(),
        sdk_type: SdkType::Azure,
        base_url: String::new(),
        env: None,
        npm: None,
        doc: None,
        endpoint_path: "/v1/responses".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models: HashMap::<String, ModelInfo>::new(),
        preserve_model_prefix: true,
    }
}

#[test]
fn azure_registry_maps_to_responses_builder() {
    let reg = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("azure"))
        .expect("azure registration");
    assert!(matches!(reg.sdk_type, SdkType::Azure));

    let mut def = azure_definition();
    def.base_url = "https://example.openai.azure.com/openai".into();

    let model = (reg.build)(
        &def,
        "gpt-4.1-mini",
        &Credentials::ApiKey("test-key".into()),
    )
    .expect("build model");
    assert_eq!(model.provider_name(), "OpenAI");
    assert_eq!(model.model_id(), "gpt-4.1-mini");
}

#[test]
fn azure_uses_resource_name_env_for_base_url_fallback() {
    let _guard = env_lock().lock().expect("env lock");
    let _env = EnvVarGuard::capture(TEST_ENV_VARS);
    std::env::set_var("AZURE_RESOURCE_NAME", "configured-resource");

    let base_prefix = resolve_base_prefix(&azure_definition()).expect("resource name fallback");
    assert_eq!(
        base_prefix,
        "https://configured-resource.openai.azure.com/openai"
    );
}

#[test]
fn azure_does_not_treat_api_key_env_as_resource_name() {
    let _guard = env_lock().lock().expect("env lock");
    let _env = EnvVarGuard::capture(TEST_ENV_VARS);
    std::env::set_var("AZURE_API_KEY", "should-not-be-hostname");

    let mut def = azure_definition();
    def.env = Some("AZURE_API_KEY".into());

    assert!(resolve_base_prefix(&def).is_err());
}
