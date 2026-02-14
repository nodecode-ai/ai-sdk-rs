use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_types::v2 as v2t;
use crate::provider_google_vertex::shared::convert_to_google_prompt_with_scopes;
pub use crate::provider_google_vertex::shared::GooglePrompt;

/// Convert V2 prompt into Google Vertex prompt shape.
pub fn convert_to_google_prompt(
    prompt: &v2t::Prompt,
    is_gemma: bool,
) -> Result<GooglePrompt, SdkError> {
    convert_to_google_prompt_with_scopes(prompt, is_gemma, &["google-vertex", "google"])
}
