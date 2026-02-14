use crate::ai_sdk_types::v2::ProviderOptions;

pub use crate::provider_google::shared::options::{
    GoogleProviderOptions, SafetySetting, ThinkingConfig,
};

pub type GoogleGenerativeAIModelId = String;

/// Extracts and deserializes the `google` section of provider options.
pub fn parse_google_provider_options(opts: &ProviderOptions) -> Option<GoogleProviderOptions> {
    crate::provider_google::shared::options::parse_google_provider_options_for_scopes(
        opts,
        &["google"],
    )
}
