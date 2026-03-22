//! Google Vertex AI provider implementation (LanguageModel v2) for ai-sdk-rs.

mod language_model;
pub mod provider;

#[cfg(test)]
#[path = "../tests/prepare_tools_tests.rs"]
mod prepare_tools_tests;
