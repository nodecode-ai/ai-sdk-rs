use crate::ai_sdk_types::v2::ProviderOptions;
use crate::provider_google_vertex::shared::{
    parse_google_provider_options_for_scopes, GoogleProviderOptions,
};

pub type GoogleVertexProviderOptions = GoogleProviderOptions;

pub type GoogleVertexModelId = String;

/// Extracts and deserializes the `google-vertex` section of provider options.
pub fn parse_google_vertex_provider_options(
    opts: &ProviderOptions,
) -> Option<GoogleVertexProviderOptions> {
    parse_google_provider_options_for_scopes(opts, &["google-vertex", "google"])
}
