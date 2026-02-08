use crate::ai_sdk_providers_anthropic::provider::default_headers_from_creds;
use std::collections::HashMap;

const OAUTH_BETA_HEADER_VALUE: &str = "oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14";

fn headers_to_map(headers: Vec<(String, String)>) -> HashMap<String, String> {
    headers
        .into_iter()
        .map(|(k, v)| (k.to_ascii_lowercase(), v))
        .collect()
}

#[test]
fn oauth_headers_include_required_fields() {
    let headers = default_headers_from_creds(None, Some("Bearer test".into()));
    let map = headers_to_map(headers);

    assert_eq!(map.get("authorization"), Some(&"Bearer test".to_string()));
    assert_eq!(
        map.get("anthropic-beta"),
        Some(&OAUTH_BETA_HEADER_VALUE.to_string())
    );
    assert_eq!(map.get("accept"), Some(&"application/json".to_string()));
    assert_eq!(map.get("x-app"), Some(&"cli".to_string()));
    assert_eq!(map.get("x-stainless-lang"), Some(&"rust".to_string()));
}

#[test]
fn api_key_headers_do_not_include_oauth_fields() {
    let headers = default_headers_from_creds(Some("key".into()), None);
    let map = headers_to_map(headers);

    assert!(map.get("authorization").is_none());
    assert!(map.get("anthropic-beta").is_none());
    assert_eq!(map.get("x-api-key"), Some(&"key".to_string()));
}
