//! Azure OpenAI provider built on the OpenAI Responses API implementation.

pub mod provider;

#[cfg(test)]
#[path = "../../../crates/providers/azure/tests/provider_registry_tests.rs"]
mod provider_registry_tests;
