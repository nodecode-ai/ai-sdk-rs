use crate::types::TokenUsage;
use serde_json::Value;

/// Parse OpenAI-style usage objects into TokenUsage.
/// Supports both Responses and Chat Completions field names.
pub fn from_openai(u: &Value) -> Option<TokenUsage> {
    let obj = u.as_object()?;
    let input = obj
        .get("input_tokens")
        .or_else(|| obj.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let output = obj
        .get("output_tokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let total = obj
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(input + output);
    let cache_read_tokens = obj
        .get("cache_read_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let cache_write_tokens = obj
        .get("cache_write_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .or_else(|| {
            obj.get("cache_creation")
                .and_then(|cc| cc.as_object())
                .and_then(|co| {
                    let a = co
                        .get("ephemeral_5m_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let b = co
                        .get("ephemeral_1h_input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let sum = a + b;
                    if sum > 0 {
                        Some(sum as usize)
                    } else {
                        None
                    }
                })
        });
    Some(TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        cache_read_tokens,
        cache_write_tokens,
    })
}

/// Normalize Anthropic usage payloads into a TokenUsage-like JSON object.
/// Returns a flat normalized JSON with keys used across the SDK.
pub fn normalize_anthropic(u: &Value) -> Value {
    let get_u64 = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    let input = get_u64("input_tokens");
    let output = get_u64("output_tokens");
    let cache_read = u
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Prefer flat "cache_creation_input_tokens" if present; otherwise sum nested ephemeral counts.
    let cache_write_flat = u
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64());
    let cache_write_nested = u.get("cache_creation").and_then(|cc| {
        let a = cc
            .get("ephemeral_5m_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let b = cc
            .get("ephemeral_1h_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Some(a + b)
    });
    let cache_write = cache_write_flat.or(cache_write_nested).unwrap_or(0);

    // Total may be absent on Anthropic; compute if needed.
    let total = u
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(input + output);

    serde_json::json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": total,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write
    })
}

/// Parse Anthropic-like usage into TokenUsage using normalization.
pub fn from_anthropic(u: &Value) -> TokenUsage {
    let norm = normalize_anthropic(u);
    let input = norm
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let output = norm
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let total = norm
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(input + output);
    let cache_read_tokens = norm
        .get("cache_read_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let cache_write_tokens = norm
        .get("cache_write_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        cache_read_tokens,
        cache_write_tokens,
    }
}
