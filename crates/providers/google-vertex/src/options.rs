use crate::ai_sdk_types::v2::ProviderOptions;

pub type GoogleVertexProviderOptions =
    crate::provider_google::shared::options::GoogleProviderOptions;

pub type GoogleVertexModelId = String;

/// Extracts and deserializes the `google-vertex` section of provider options.
pub fn parse_google_vertex_provider_options(
    opts: &ProviderOptions,
) -> Option<GoogleVertexProviderOptions> {
    crate::provider_google::shared::options::parse_google_provider_options_for_scopes(
        opts,
        &["google-vertex", "google"],
    )
}
