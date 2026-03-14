pub mod config;
pub mod error;
pub mod language_model;
pub mod provider;

pub use language_model::GatewayLanguageModel;

#[cfg(test)]
#[path = "../../../crates/providers/gateway/tests/provider_registry_tests.rs"]
mod provider_registry_tests;
