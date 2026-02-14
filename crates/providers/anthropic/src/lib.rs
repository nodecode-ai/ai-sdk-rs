//! Anthropic provider (Vercel AI SDK parity layout).
//!
//! This crate is being migrated to a modular structure:
//! - api_types.rs: wire types for Anthropic Messages API
//! - error.rs: error parsing/mapping helpers
//! - messages/language_model.rs: LanguageModel v2 implementation
//! - messages/options.rs: provider and file-part options
//! - provider.rs: provider registry + model factory

pub mod api_types;
pub mod error;
pub mod messages {
    pub mod language_model;
    pub mod options;
}
pub mod provider;

// Re-exports for convenience
pub use messages::language_model::AnthropicMessagesLanguageModel;

#[cfg(test)]
#[path = "../tests/messages_options_tests.rs"]
mod messages_options_tests;

#[cfg(test)]
#[path = "../tests/messages_tools_tests.rs"]
mod messages_tools_tests;

#[cfg(test)]
#[path = "../tests/provider_headers_tests.rs"]
mod provider_headers_tests;
