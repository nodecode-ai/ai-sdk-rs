//! Azure OpenAI provider built on the OpenAI Responses API implementation.

pub mod provider;

#[cfg(test)]
#[path = "../tests/provider_registry_tests.rs"]
mod provider_registry_tests;
