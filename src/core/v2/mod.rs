//! Vercel-compatible LanguageModel interface (formerly V2) and helpers.

use crate::ai_sdk_core::SdkError;
use crate::ai_sdk_types::v2 as v2t;
use futures_core::Stream;
use std::pin::Pin;

/// Stream of structured parts from the model.
pub type PartStream = Pin<Box<dyn Stream<Item = Result<v2t::StreamPart, SdkError>> + Send>>;

/// Generate response payload.
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub content: Vec<v2t::Content>,
    pub finish_reason: v2t::FinishReason,
    pub usage: v2t::Usage,
    pub provider_metadata: Option<v2t::ProviderMetadata>,
    pub request_body: Option<serde_json::Value>,
    pub response_headers: Option<v2t::Headers>,
    pub response_body: Option<serde_json::Value>,
    pub warnings: Vec<v2t::CallWarning>,
}

/// Stream response envelope.
pub struct StreamResponse {
    pub stream: PartStream,
    pub request_body: Option<serde_json::Value>,
    pub response_headers: Option<v2t::Headers>,
}

#[async_trait::async_trait]
pub trait LanguageModelTurnSession: Send {
    fn provider_name(&self) -> &'static str;
    fn model_id(&self) -> &str;

    async fn do_stream(
        &mut self,
        options: v2t::CallOptions,
    ) -> Result<StreamResponse, crate::ai_sdk_core::SdkError>;
}

pub type BoxedLanguageModelTurnSession<'a> = Box<dyn LanguageModelTurnSession + 'a>;

struct StatelessLanguageModelTurnSession<'a, M: LanguageModel + ?Sized> {
    model: &'a M,
}

impl<'a, M: LanguageModel + ?Sized> StatelessLanguageModelTurnSession<'a, M> {
    fn new(model: &'a M) -> Self {
        Self { model }
    }
}

#[async_trait::async_trait]
impl<M: LanguageModel + ?Sized> LanguageModelTurnSession
    for StatelessLanguageModelTurnSession<'_, M>
{
    fn provider_name(&self) -> &'static str {
        self.model.provider_name()
    }

    fn model_id(&self) -> &str {
        self.model.model_id()
    }

    async fn do_stream(
        &mut self,
        options: v2t::CallOptions,
    ) -> Result<StreamResponse, crate::ai_sdk_core::SdkError> {
        self.model.do_stream(options).await
    }
}

/// Language model interface (Vercel AI SDK parity).
#[async_trait::async_trait]
pub trait LanguageModel: Send + Sync {
    /// Implemented spec version; constant "v2" for all models.
    fn specification_version(&self) -> &'static str {
        "v2"
    }
    /// Provider name for logging/telemetry.
    fn provider_name(&self) -> &'static str;
    /// Provider-specific model identifier.
    fn model_id(&self) -> &str;
    /// Supported URL regex patterns by media type, lower-case URLs.
    fn supported_urls(&self) -> std::collections::HashMap<String, Vec<String>> {
        Default::default()
    }

    async fn do_generate(
        &self,
        options: v2t::CallOptions,
    ) -> Result<GenerateResponse, crate::ai_sdk_core::SdkError>;
    async fn do_stream(
        &self,
        options: v2t::CallOptions,
    ) -> Result<StreamResponse, crate::ai_sdk_core::SdkError>;

    fn new_turn_session(&self) -> BoxedLanguageModelTurnSession<'_> {
        Box::new(StatelessLanguageModelTurnSession::new(self))
    }
}

// No adapters or converters: providers implement the v2 surface directly.
