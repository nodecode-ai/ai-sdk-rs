use crate::ai_sdk_provider::{
    persisted_reasoning_options, reasoning_scope_aliases, reasoning_stream_options,
};
use ai_sdk_providers_amazon_bedrock as _;
use ai_sdk_providers_anthropic as _;
use crate::ai_sdk_types::catalog::SdkType;
use serde_json::Value as JsonValue;

#[test]
fn anthropic_reasoning_aliases_include_canonical_scope() {
    let aliases = reasoning_scope_aliases(
        "anthropic",
        &SdkType::Anthropic,
        Some("claude-3"),
        Some("https://api.anthropic.com"),
    )
    .expect("anthropic aliases");
    assert!(aliases.iter().any(|a| a.eq_ignore_ascii_case("anthropic")));
    assert!(aliases
        .iter()
        .any(|a| a.eq_ignore_ascii_case("api.anthropic.com")));
}

#[test]
fn bedrock_reasoning_aliases_trigger_for_anthropic_models() {
    let aliases = reasoning_scope_aliases(
        "aws-bedrock",
        &SdkType::AmazonBedrock,
        Some("anthropic.claude-3-5-sonnet-20240620-v1:0"),
        None,
    )
    .expect("bedrock aliases");
    assert!(aliases.iter().any(|a| a.eq_ignore_ascii_case("anthropic")));
    assert!(aliases.iter().any(|a| a.eq_ignore_ascii_case("bedrock")));
    assert!(aliases
        .iter()
        .any(|a| a.eq_ignore_ascii_case("aws-bedrock")));
}

#[test]
fn reasoning_aliases_absent_for_non_anthropic_bedrock_models() {
    let aliases = reasoning_scope_aliases(
        "aws-bedrock",
        &SdkType::AmazonBedrock,
        Some("ai21.jamba-1-large"),
        None,
    );
    assert!(aliases.is_none());
}

#[test]
fn reasoning_stream_options_include_signature() {
    let opts = reasoning_stream_options(
        "anthropic",
        &SdkType::Anthropic,
        Some("claude-3"),
        Some("https://api.anthropic.com"),
        Some("sig"),
        None,
    )
    .expect("stream options");
    let scope = opts.get("anthropic").expect("anthropic scope");
    assert_eq!(
        scope.get("signature"),
        Some(&JsonValue::String("sig".into()))
    );
}

#[test]
fn persisted_reasoning_options_include_text_and_signature() {
    let opts = persisted_reasoning_options(
        "aws-bedrock",
        &SdkType::AmazonBedrock,
        Some("anthropic.claude-3-5-sonnet-20240620-v1:0"),
        None,
        "thinking",
        Some("sig"),
    )
    .expect("persisted options");
    for key in ["anthropic", "bedrock", "aws-bedrock"] {
        let scope = opts.get(key).expect("scope");
        assert_eq!(
            scope.get("persistedReasoningText"),
            Some(&JsonValue::String("thinking".into()))
        );
        assert_eq!(
            scope.get("persistedReasoningSignature"),
            Some(&JsonValue::String("sig".into()))
        );
    }
}
