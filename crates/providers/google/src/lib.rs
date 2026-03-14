//! Google Generative AI (non-Vertex) provider implementation for ai-sdk-rs (LanguageModel v2).

pub mod provider;
pub(crate) mod shared;

pub mod gen_ai {
    pub mod language_model;
}

#[cfg(test)]
#[path = "../tests/parity_regression_tests.rs"]
mod parity_regression_tests;

#[cfg(test)]
#[path = "../tests/prepare_tools_tests.rs"]
mod prepare_tools_tests;
