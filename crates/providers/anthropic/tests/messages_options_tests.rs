use crate::ai_sdk_providers_anthropic::messages::options::{
    parse_anthropic_provider_options, ThinkingOption,
};
use crate::ai_sdk_types::v2::ProviderOptions;
use serde_json::json;
use std::collections::HashMap;

#[test]
fn parses_thinking_from_custom_provider_scope() {
    let mut scope = HashMap::new();
    scope.insert(
        "thinking".to_string(),
        json!({"type": "enabled", "budget_tokens": 16000}),
    );

    let mut opts = ProviderOptions::new();
    opts.insert("newcli".to_string(), scope);

    let parsed = parse_anthropic_provider_options(&opts, "newcli").unwrap();
    match parsed.thinking.unwrap() {
        ThinkingOption::Enabled { budget_tokens } => assert_eq!(budget_tokens, 16000),
        other => panic!("unexpected thinking option: {other:?}"),
    }
}

#[test]
fn does_not_fall_back_to_anthropic_scope() {
    let mut scope = HashMap::new();
    scope.insert(
        "thinking".to_string(),
        json!({"type": "enabled", "budget_tokens": 1234}),
    );

    let mut opts = ProviderOptions::new();
    opts.insert("anthropic".to_string(), scope);

    assert!(parse_anthropic_provider_options(&opts, "newcli").is_none());
}

#[test]
fn parses_thinking_from_anthropic_scope() {
    let mut scope = HashMap::new();
    scope.insert(
        "thinking".to_string(),
        json!({"type": "enabled", "budgetTokens": 4321}),
    );

    let mut opts = ProviderOptions::new();
    opts.insert("anthropic".to_string(), scope);

    let parsed = parse_anthropic_provider_options(&opts, "anthropic").unwrap();
    match parsed.thinking.unwrap() {
        ThinkingOption::Enabled { budget_tokens } => assert_eq!(budget_tokens, 4321),
        other => panic!("unexpected thinking option: {other:?}"),
    }
}
