use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::v2::{headers_is_empty, provider_options_is_empty, ProviderOptions};

/// Single embedding vector.
pub type Embedding = Vec<f32>;

/// Input options for embedding calls.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedOptions {
    #[serde(default)]
    pub values: Vec<String>,
    #[serde(default, skip_serializing_if = "headers_is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(
        default,
        skip_serializing_if = "provider_options_is_empty",
        rename = "providerOptions"
    )]
    pub provider_options: ProviderOptions,
}

impl EmbedOptions {
    pub fn new(values: Vec<String>) -> Self {
        Self {
            values,
            ..Default::default()
        }
    }
}

/// Token usage for an embedding call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmbedUsage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,
}
