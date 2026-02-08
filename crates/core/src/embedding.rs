use crate::ai_sdk_types::embedding as embt;
use crate::ai_sdk_types::v2 as v2t;

use crate::core::SdkError;

/// Response from an embedding model call.
#[derive(Debug, Clone)]
pub struct EmbedResponse {
    pub embeddings: Vec<embt::Embedding>,
    pub usage: Option<embt::EmbedUsage>,
    pub provider_metadata: Option<v2t::ProviderMetadata>,
    pub response_headers: Option<v2t::Headers>,
    pub response_body: Option<serde_json::Value>,
    pub request_body: Option<serde_json::Value>,
}

impl EmbedResponse {
    pub fn empty() -> Self {
        Self {
            embeddings: Vec::new(),
            usage: None,
            provider_metadata: None,
            response_headers: None,
            response_body: None,
            request_body: None,
        }
    }
}

/// Embedding model interface (parity with Vercel EmbeddingModelV3).
#[async_trait::async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Implemented spec version; constant "v3" for all models.
    fn specification_version(&self) -> &'static str {
        "v3"
    }
    /// Provider name for logging/telemetry.
    fn provider_name(&self) -> &'static str;
    /// Provider-specific model identifier.
    fn model_id(&self) -> &str;
    /// Limit of embeddings per call, if enforced by the provider.
    fn max_embeddings_per_call(&self) -> Option<usize> {
        None
    }
    /// Whether multiple embedding calls may be executed in parallel.
    fn supports_parallel_calls(&self) -> bool {
        true
    }

    async fn do_embed(&self, options: embt::EmbedOptions) -> Result<EmbedResponse, SdkError>;
}
