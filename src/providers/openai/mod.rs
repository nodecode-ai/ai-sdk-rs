//! OpenAI provider (modular v2 layout)

pub mod config;
pub mod error;
pub mod provider;
pub mod responses;

// Keep overrides module available for typed provider overrides
pub mod overrides;

pub use provider::OpenAIResponsesBuilder;
pub use responses::language_model::OpenAIResponsesLanguageModel;

#[cfg(test)]
#[path = "../../../crates/providers/openai/tests/responses_language_model_tests.rs"]
mod responses_language_model_tests;

#[cfg(test)]
#[path = "../../../crates/providers/openai/tests/stream_fixture_tests.rs"]
mod stream_fixture_tests;
