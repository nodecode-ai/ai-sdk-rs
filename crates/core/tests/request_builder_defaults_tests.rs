use crate::ai_sdk_core::request_builder::defaults::{
    merge_provider_defaults, provider_defaults_from_json, request_overrides_from_json,
};
use crate::ai_sdk_types::v2 as v2t;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn provider_defaults_require_exact_scope() {
    let raw = json!({"openai": {"temperature": 0.1}});
    let defaults = provider_defaults_from_json("openai", &raw).expect("scoped defaults");
    assert_eq!(defaults["openai"]["temperature"], json!(0.1));

    let inline = json!({"temperature": 0.2});
    assert!(provider_defaults_from_json("openai", &inline).is_none());

    let mismatched = json!({"openai-compatible": {"temperature": 0.3}});
    assert!(provider_defaults_from_json("openai", &mismatched).is_none());
}

#[test]
fn request_overrides_require_exact_scope() {
    let raw = json!({"openai": {"top_p": 0.8}});
    let overrides = request_overrides_from_json("openai", &raw).expect("scoped overrides");
    assert_eq!(overrides["top_p"], json!(0.8));

    let inline = json!({"top_p": 0.5});
    assert!(request_overrides_from_json("openai", &inline).is_none());
}

#[test]
fn merge_provider_defaults_preserves_existing_values() {
    let mut target = v2t::ProviderOptions::new();
    target.insert(
        "openai".into(),
        HashMap::from([
            ("temperature".into(), json!(0.9)),
            ("nested".into(), json!({"a": 1})),
        ]),
    );
    let defaults = v2t::ProviderOptions::from([(
        "openai".into(),
        HashMap::from([
            ("temperature".into(), json!(0.1)),
            ("top_p".into(), json!(0.2)),
            ("nested".into(), json!({"a": 2, "b": 3})),
        ]),
    )]);

    merge_provider_defaults(&mut target, &defaults);

    let scope = &target["openai"];
    assert_eq!(scope["temperature"], json!(0.9));
    assert_eq!(scope["top_p"], json!(0.2));
    assert_eq!(scope["nested"], json!({"a": 1, "b": 3}));
}
