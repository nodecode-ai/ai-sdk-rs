use serde_json::{json, Map, Value};

/// Builder for OpenAI Responses API provider overrides.
///
/// Produces a JSON object intended for `metadata.provider_overrides` in `ChatRequest`.
/// Only sets fields you opt into; does shallow merging on the provider side.
#[derive(Debug, Default, Clone)]
pub struct OpenAIOverridesBuilder {
    obj: Map<String, Value>,
}

impl OpenAIOverridesBuilder {
    pub fn new() -> Self {
        Self { obj: Map::new() }
    }

    /// Set a free-form instructions string (in addition to message-derived instructions).
    pub fn instructions(mut self, s: impl Into<String>) -> Self {
        self.obj
            .insert("instructions".into(), Value::String(s.into()));
        self
    }

    /// Set reasoning effort for reasoning-capable models (e.g., "low" or "high").
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        let mut r = self
            .obj
            .remove("reasoning")
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        r.insert("effort".into(), Value::String(effort.into()));
        self.obj.insert("reasoning".into(), Value::Object(r));
        self
    }

    /// Request a reasoning summary (e.g., "auto" or "detailed").
    pub fn reasoning_summary(mut self, summary: impl Into<String>) -> Self {
        let mut r = self
            .obj
            .remove("reasoning")
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        r.insert("summary".into(), Value::String(summary.into()));
        self.obj.insert("reasoning".into(), Value::Object(r));
        self
    }

    /// Set service tier (e.g., "standard", "priority", or "flex").
    pub fn service_tier(mut self, tier: impl Into<String>) -> Self {
        self.obj
            .insert("service_tier".into(), Value::String(tier.into()));
        self
    }

    /// Set the `include` array to request additional fields (e.g., "message.output_text.logprobs").
    pub fn include<I, S>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let v: Vec<Value> = items.into_iter().map(|s| Value::String(s.into())).collect();
        self.obj.insert("include".into(), Value::Array(v));
        self
    }

    /// Request top logprobs (1..=20). The Responses API uses `top_logprobs`.
    pub fn top_logprobs(mut self, n: u8) -> Self {
        let capped = n.min(20).max(1);
        self.obj.insert("top_logprobs".into(), json!(capped));
        // Consumers should also include "message.output_text.logprobs" in `include`.
        self
    }

    pub fn previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.obj
            .insert("previous_response_id".into(), Value::String(id.into()));
        self
    }

    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.obj
            .insert("prompt_cache_key".into(), Value::String(key.into()));
        self
    }

    pub fn safety_identifier(mut self, id: impl Into<String>) -> Self {
        self.obj
            .insert("safety_identifier".into(), Value::String(id.into()));
        self
    }

    pub fn store(mut self, b: bool) -> Self {
        self.obj.insert("store".into(), json!(b));
        self
    }
    pub fn user(mut self, u: impl Into<String>) -> Self {
        self.obj.insert("user".into(), Value::String(u.into()));
        self
    }

    /// Configure JSON outputs using a JSON Schema.
    /// Example: `.text_json_schema("response", None, schema, Some(true))`
    pub fn text_json_schema(
        mut self,
        name: impl Into<String>,
        description: Option<String>,
        schema: Value,
        strict: Option<bool>,
    ) -> Self {
        let mut text = self
            .obj
            .remove("text")
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        let mut fmt = Map::new();
        fmt.insert("type".into(), Value::String("json_schema".into()));
        fmt.insert("name".into(), Value::String(name.into()));
        if let Some(d) = description {
            fmt.insert("description".into(), Value::String(d));
        }
        fmt.insert("schema".into(), schema);
        if let Some(s) = strict {
            fmt.insert("strict".into(), json!(s));
        }
        text.insert("format".into(), Value::Object(fmt));
        self.obj.insert("text".into(), Value::Object(text));
        self
    }

    /// Set text verbosity (e.g., "brief" | "balanced" | "verbose").
    pub fn text_verbosity(mut self, v: impl Into<String>) -> Self {
        let mut text = self
            .obj
            .remove("text")
            .and_then(|vv| vv.as_object().cloned())
            .unwrap_or_default();
        text.insert("verbosity".into(), Value::String(v.into()));
        self.obj.insert("text".into(), Value::Object(text));
        self
    }

    /// Build the overrides object to place under `metadata.provider_overrides`.
    pub fn build(self) -> Value {
        Value::Object(self.obj)
    }
}
