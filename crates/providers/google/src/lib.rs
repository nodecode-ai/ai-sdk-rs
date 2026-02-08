//! Google Generative AI (non-Vertex) provider implementation for ai-sdk-rs (LanguageModel v2).

pub mod error;
pub mod prepare_tools;
pub mod provider;

pub mod gen_ai {
    pub mod language_model;
    pub mod options;
    pub mod prompt;
}
