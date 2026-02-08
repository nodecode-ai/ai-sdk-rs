use serde_json::Value;

/// Returns true if a header key is reserved for internal SDK use.
pub fn is_internal_sdk_header(key: &str) -> bool {
    key.to_ascii_lowercase().starts_with("x-ai-sdk-")
}

/// Extracts the internal options JSON from a list of extra headers.
/// Expects `X-AI-SDK-Options` with a JSON string value.
pub fn extract_options_from_headers(extra_headers: &[(String, String)]) -> Option<Value> {
    extra_headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("x-ai-sdk-options"))
        .and_then(|(_, v)| serde_json::from_str::<Value>(v).ok())
}

/// Deep-merge `b` into `a`.
pub fn deep_merge(a: &mut Value, b: &Value) {
    match (a, b) {
        (Value::Object(a_map), Value::Object(b_map)) => {
            for (k, v) in b_map {
                if let Some(av) = a_map.get_mut(k) {
                    deep_merge(av, v);
                } else {
                    a_map.insert(k.clone(), v.clone());
                }
            }
        }
        (a_slot, b_val) => {
            *a_slot = b_val.clone();
        }
    }
}

/// Merge provider options into a request body, skipping disallowed structural keys.
pub fn merge_options_with_disallow(body: &mut Value, options: &Value, disallow: &[&str]) {
    if let (Value::Object(bm), Value::Object(om)) = (body, options) {
        for (k, v) in om {
            if disallow.contains(&k.as_str()) {
                continue;
            }
            match bm.get_mut(k) {
                Some(existing) => deep_merge(existing, v),
                None => {
                    bm.insert(k.clone(), v.clone());
                }
            }
        }
    }
}
