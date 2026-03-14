//! Google Generative AI (non-Vertex) provider implementation for ai-sdk-rs (LanguageModel v2).

pub mod provider;
pub(crate) mod shared;

pub mod gen_ai {
    pub mod language_model;
}

#[cfg(test)]
#[path = "../../../crates/providers/google/tests/parity_regression_tests.rs"]
mod parity_regression_tests;

#[cfg(test)]
#[path = "../../../crates/providers/google/tests/prepare_tools_tests.rs"]
mod prepare_tools_tests;
