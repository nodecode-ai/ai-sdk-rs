use crate::ai_sdk_providers_openai_compatible::completion::options::parse_openai_compatible_completion_provider_options;
use crate::ai_sdk_types::v2 as v2t;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn merges_openai_compatible_and_provider_specific_options() {
    let mut provider_options = v2t::ProviderOptions::new();
    provider_options.insert(
        "openai-compatible".into(),
        HashMap::from([
            ("user".into(), json!("base-user")),
            ("echo".into(), json!(true)),
            ("logitBias".into(), json!({"123": 1.2})),
            ("baseOnly".into(), json!("ignored")),
        ]),
    );
    provider_options.insert(
        "test-provider".into(),
        HashMap::from([
            ("suffix".into(), json!("suffix")),
            ("echo".into(), json!(false)),
            ("someCustom".into(), json!(true)),
        ]),
    );

    let (opts, extras) = parse_openai_compatible_completion_provider_options(
        &provider_options,
        &["openai-compatible", "test-provider"],
    );

    assert_eq!(opts.user, Some("base-user".into()));
    assert_eq!(opts.suffix, Some("suffix".into()));
    assert_eq!(opts.echo, Some(false));
    assert_eq!(
        opts.logit_bias
            .as_ref()
            .and_then(|bias| bias.get("123").copied()),
        Some(1.2)
    );

    let extras = extras.expect("extras");
    assert_eq!(extras.get("someCustom"), Some(&json!(true)));
    assert!(!extras.contains_key("baseOnly"));
}
