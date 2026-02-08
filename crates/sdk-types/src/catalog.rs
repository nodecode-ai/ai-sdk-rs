//! Model catalog and provider definitions
//!
//! This module contains types for defining AI providers and their available models.

use serde::{de::Deserializer, Deserialize, Serialize};
use std::collections::HashMap;

/// SDK type used by a provider to determine API compatibility
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SdkType {
    /// Official OpenAI SDK (Responses API)
    OpenAI,
    /// Azure-hosted OpenAI API compatibility
    Azure,
    /// OpenAI-compatible chat completions SDK (e.g., compatibility servers, Groq-like)
    #[serde(rename = "openai-compatible")]
    OpenAICompatible,
    /// OpenAI-compatible Chat Completions SDK
    #[serde(rename = "openai-compatible-chat")]
    OpenAICompatibleChat,
    /// OpenAI-compatible Completions SDK
    #[serde(rename = "openai-compatible-completion")]
    OpenAICompatibleCompletion,
    Anthropic,
    Google,
    #[serde(rename = "google-vertex")]
    GoogleVertex,
    Groq,
    Gateway,
    #[serde(rename = "amazon-bedrock")]
    AmazonBedrock,
}

// SdkType parsing is intentionally centralized in the provider registry via
// `crate::ai_sdk_provider::sdk_type_from_id`, which uses registered builders as the
// source of truth. This avoids ad-hoc string parsers living in sdk-types.

/// Information about a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique identifier for the model
    pub id: String,
    /// Human-readable display name
    pub display_name: String,
    /// Provider identifier the model belongs to (if supplied by registry)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Optional human-readable description of the model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Release date string provided by the registry (YYYY-MM-DD)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
    /// Last update timestamp string provided by the registry (YYYY-MM-DD)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
    /// Model knowledge cutoff indicator (YYYY-MM)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knowledge: Option<String>,
    /// Structured capability flags supplied by the registry
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<ModelCapabilities>,
    /// Modalities supported for inputs/outputs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<ModelModalities>,
    /// Limits for context/output tokens
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<ModelLimits>,
    /// Pricing information per token/tooling caches
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// Whether the model weights are openly available
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_weights: Option<bool>,
}

/// Capability flags for a model as described by the registry.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub attachment: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub temperature: bool,
    #[serde(default)]
    pub tool_call: bool,
    #[serde(default)]
    pub computer_use: bool,
    /// Audio input/output capability (from appendix API)
    #[serde(default)]
    pub audio: bool,
    /// Structured JSON output mode capability (from appendix API)
    #[serde(default)]
    pub json_mode: bool,
    /// Vision/image input capability (from appendix API)
    #[serde(default)]
    pub vision: bool,
}

/// Supported input/output modalities for a model.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelModalities {
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_modalities_field"
    )]
    pub input: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_modalities_field"
    )]
    pub output: Vec<String>,
}

/// Token limits communicated by the registry.
/// Supports both old format (context, output) and new appendix format (context_input, context_output).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelLimits {
    /// Context window size (old format)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u64>,
    /// Max output tokens (old format)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<u64>,
    /// Context input window size (new appendix format)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_input: Option<u64>,
    /// Context output max tokens (new appendix format)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_output: Option<u64>,
}

impl ModelLimits {
    /// Get context window size, preferring new format over old
    pub fn get_context(&self) -> Option<u64> {
        self.context_input.or(self.context)
    }

    /// Get max output tokens, preferring new format over old
    pub fn get_output(&self) -> Option<u64> {
        self.context_output.or(self.output)
    }
}

/// Pricing metadata for a model (old format).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelCost {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

/// Single pricing entry from the new appendix API format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PricingEntry {
    /// Currency code (e.g., "USD")
    pub currency: String,
    /// Price type: "tokens_input", "tokens_output", etc.
    pub kind: String,
    /// Price amount
    pub price: f64,
    /// Price source (e.g., "models-dev", "litellm")
    #[serde(default)]
    pub source: Option<String>,
    /// Unit of pricing (e.g., "tok_1m" for per million tokens)
    #[serde(default)]
    pub unit: Option<String>,
}

impl ModelCost {
    /// Create ModelCost from new appendix pricing array format
    pub fn from_pricing_entries(entries: &[PricingEntry]) -> Self {
        let mut cost = ModelCost::default();
        for entry in entries {
            match entry.kind.as_str() {
                "tokens_input" => cost.input = Some(entry.price),
                "tokens_output" => cost.output = Some(entry.price),
                "cache_read" => cost.cache_read = Some(entry.price),
                "cache_write" => cost.cache_write = Some(entry.price),
                _ => {}
            }
        }
        cost
    }
}

fn deserialize_modalities_field<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Single(String),
        Multiple(Vec<String>),
    }

    let value = Option::<StringOrVec>::deserialize(deserializer)?;
    let items = match value {
        Some(StringOrVec::Single(s)) => vec![s],
        Some(StringOrVec::Multiple(list)) => list,
        None => Vec::new(),
    };
    Ok(items)
}

/// Definition of an AI provider and its configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDefinition {
    /// Internal name/identifier for the provider
    pub name: String,
    /// Human-readable display name
    pub display_name: String,
    /// SDK type for API compatibility
    pub sdk_type: SdkType,
    /// Base URL for the provider's API
    pub base_url: String,
    /// Environment variable expected for API authentication
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<String>,
    /// Associated npm package identifier for client SDKs (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub npm: Option<String>,
    /// Documentation URL advertised by the registry
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    /// Optional endpoint path (appended to base_url)
    #[serde(default)]
    pub endpoint_path: String,
    /// Additional headers to include in requests
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Query parameters to include in requests
    #[serde(default)]
    pub query_params: HashMap<String, String>,
    /// Idle timeout (ms) to wait for streaming activity before treating the connection as lost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_idle_timeout_ms: Option<u64>,
    /// Authentication type (e.g., "bearer", "api-key")
    pub auth_type: String,
    /// Available models for this provider
    #[serde(default)]
    pub models: HashMap<String, ModelInfo>,
    /// Whether to preserve model prefixes like "openai/" in the model name
    /// Some providers (like GitHub Models) expect just the model name without prefixes
    #[serde(default = "default_preserve_model_prefix")]
    pub preserve_model_prefix: bool,
}

fn default_preserve_model_prefix() -> bool {
    true
}

/// Catalog of available AI providers
pub struct ProviderCatalog {
    providers: HashMap<String, ProviderDefinition>,
}

impl ProviderCatalog {
    /// Create a new empty catalog
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Create a catalog from a HashMap of providers
    pub fn from_providers(providers: HashMap<String, ProviderDefinition>) -> Self {
        Self { providers }
    }

    /// Find a provider and model by model name
    ///
    /// Supports both prefixed format (e.g., "github/gpt-4o") and direct model names
    pub fn find_provider_for_model(&self, model: &str) -> Option<(&ProviderDefinition, String)> {
        // Check for provider prefix (e.g., "github/gpt-4o")
        if let Some((provider, model_name)) = model.split_once('/') {
            if let Some(def) = self.providers.get(provider) {
                if let Some(model_info) = def.models.get(model_name) {
                    return Some((def, model_info.id.clone()));
                }
            }
        }

        // Check all providers for direct model name
        for (_, def) in &self.providers {
            if let Some(model_info) = def.models.get(model) {
                return Some((def, model_info.id.clone()));
            }
        }

        None
    }

    /// Get a provider by name
    pub fn get_provider(&self, name: &str) -> Option<&ProviderDefinition> {
        self.providers.get(name)
    }

    /// Get all providers
    pub fn providers(&self) -> &HashMap<String, ProviderDefinition> {
        &self.providers
    }

    /// Add a provider to the catalog
    pub fn add_provider(&mut self, provider: ProviderDefinition) {
        self.providers.insert(provider.name.clone(), provider);
    }
}

impl Default for ProviderCatalog {
    fn default() -> Self {
        Self::new()
    }
}
