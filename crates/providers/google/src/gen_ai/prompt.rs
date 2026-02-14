use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_types::v2 as v2t;

pub use crate::provider_google::shared::prompt::{
    GoogleContent, GoogleContentPart, GoogleFileData, GoogleFunctionCall, GoogleFunctionResponse,
    GoogleInlineData, GooglePrompt, GoogleSystemInstruction, GoogleTextPart,
};

/// Convert V2 prompt into Google GenAI prompt shape.
pub fn convert_to_google_prompt(
    prompt: &v2t::Prompt,
    is_gemma: bool,
) -> Result<GooglePrompt, SdkError> {
    crate::provider_google::shared::prompt::convert_to_google_prompt_with_scopes(
        prompt,
        is_gemma,
        &["google"],
    )
}
