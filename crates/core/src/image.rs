use std::time::SystemTime;

use crate::ai_sdk_types::image as imgt;
use crate::ai_sdk_types::v2 as v2t;

use crate::core::SdkError;

#[derive(Debug, Clone)]
pub struct ImageResponseMeta {
    pub timestamp: SystemTime,
    pub model_id: String,
    pub headers: Option<v2t::Headers>,
}

#[derive(Debug, Clone)]
pub struct ImageResponse {
    pub images: Vec<imgt::ImageData>,
    pub warnings: Vec<imgt::ImageWarning>,
    pub provider_metadata: Option<v2t::ProviderMetadata>,
    pub response: ImageResponseMeta,
    pub usage: Option<imgt::ImageUsage>,
    pub response_body: Option<serde_json::Value>,
    pub request_body: Option<serde_json::Value>,
}

#[async_trait::async_trait]
pub trait ImageModel: Send + Sync {
    /// Implemented spec version; constant "v3" for all models.
    fn specification_version(&self) -> &'static str {
        "v3"
    }
    /// Provider name for logging/telemetry.
    fn provider_name(&self) -> &'static str;
    /// Provider-specific model identifier.
    fn model_id(&self) -> &str;
    /// Limit of images per call, if enforced by the provider.
    fn max_images_per_call(&self) -> Option<usize> {
        None
    }

    async fn do_generate(&self, options: imgt::ImageOptions) -> Result<ImageResponse, SdkError>;
}
