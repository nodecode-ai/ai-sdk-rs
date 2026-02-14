//! OpenAI-compatible provider (completion-first) aligned to Vercel AI SDK.

pub mod error;
pub mod provider;
mod stream;
pub mod completion {
    pub mod convert;
    pub mod finish_reason;
    pub mod language_model;
    pub mod options;
}
pub mod chat {
    pub mod convert;
    pub mod language_model;
    pub mod options;
    pub mod prepare_tools;
}
pub mod embedding {
    pub mod embedding_model;
    pub mod options;
}
pub mod image {
    pub mod image_model;
    pub mod options;
}

pub use chat::language_model::OpenAICompatibleChatLanguageModel;
pub use completion::language_model::OpenAICompatibleCompletionLanguageModel;
pub use embedding::embedding_model::OpenAICompatibleEmbeddingModel;
pub use image::image_model::OpenAICompatibleImageModel;
pub use provider::build_openai_compatible_embedding;
pub use provider::build_openai_compatible_image;
pub use stream::{build_stream, StreamMode, StreamSettings};

#[cfg(test)]
#[path = "../tests/chat_convert_tests.rs"]
mod chat_convert_tests;

#[cfg(test)]
#[path = "../tests/chat_language_model_tests.rs"]
mod chat_language_model_tests;

#[cfg(test)]
#[path = "../tests/completion_options_tests.rs"]
mod completion_options_tests;

#[cfg(test)]
#[path = "../tests/embedding_model_tests.rs"]
mod embedding_model_tests;

#[cfg(test)]
#[path = "../tests/image_model_tests.rs"]
mod image_model_tests;

#[cfg(test)]
#[path = "../tests/provider_registry_tests.rs"]
mod provider_registry_tests;

#[cfg(test)]
#[path = "../tests/stream_tests.rs"]
mod stream_tests;
