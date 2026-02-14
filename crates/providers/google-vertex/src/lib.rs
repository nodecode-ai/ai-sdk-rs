//! Google Vertex AI provider implementation (LanguageModel v2) for ai-sdk-rs.

mod error;
mod language_model;
pub(crate) mod options;
pub mod prepare_tools;
pub(crate) mod prompt;
pub mod provider;
/// Internal boundary seam for Google Vertex reuse of Google shared implementation.
///
/// Keep cross-provider reuse centralized here so coupling stays explicit and
/// callers in this module tree avoid direct path-indirection to
/// `provider_google::shared::*`.
mod shared;

#[cfg(test)]
#[path = "../tests/prepare_tools_tests.rs"]
mod prepare_tools_tests;
