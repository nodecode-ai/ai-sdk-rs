use serde_json::{Map, Value, json};

/// Builder for Anthropic Messages API provider overrides.
///
/// Produces the per-provider JSON object to place under
/// `CallOptions::provider_options["anthropic"]`.
/// Only sets fields you opt into; merged shallowly by the provider (structural keys skipped).
#[derive(Debug, Default, Clone)]
pub struct AnthropicOverridesBuilder {
    obj: Map<String, Value>,
}

impl AnthropicOverridesBuilder {
    pub fn new() -> Self { Self { obj: Map::new() } }

    /// Enable Anthropic thinking with a required budget in tokens.
    ///
    /// This inserts `{ thinking: { type: "enabled", budget_tokens: <budget> } }`.
    pub fn thinking_enabled(mut self, budget_tokens: u32) -> Self {
        let mut thinking = Map::new();
        thinking.insert("type".into(), Value::String("enabled".into()));
        thinking.insert("budget_tokens".into(), json!(budget_tokens));
        self.obj.insert("thinking".into(), Value::Object(thinking));
        self
    }

    /// Disable Anthropic thinking. Equivalent to `{ thinking: { type: "disabled" } }`.
    pub fn thinking_disabled(mut self) -> Self {
        let mut thinking = Map::new();
        thinking.insert("type".into(), Value::String("disabled".into()));
        self.obj.insert("thinking".into(), Value::Object(thinking));
        self
    }

    /// Build the overrides object to place under `provider_options["anthropic"]`.
    pub fn build(self) -> Value { Value::Object(self.obj) }
}
