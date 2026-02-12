pub mod capabilities;
pub mod embedding;
pub mod error;
pub mod event_mapper;
pub mod image;
pub mod json;
pub mod options;
pub mod request_builder;
pub mod retry;
pub mod stream_collect;
pub mod transport;
pub mod v2;

pub use crate::core::embedding::{EmbedResponse, EmbeddingModel};
pub use crate::core::error::{SdkError, TransportError};
pub use crate::core::event_mapper::{
    map_events_to_parts, EventMapperConfig, EventMapperHooks, EventMapperState, ProviderMetadata,
};
pub use crate::core::image::{ImageModel, ImageResponse, ImageResponseMeta};

// Re-export v2 (Vercel parity) model trait and typed surfaces at the crate root
pub use crate::core::v2::{GenerateResponse, LanguageModel, PartStream, StreamResponse};
// Convenience re-exports of common types
pub use crate::ai_sdk_types::embedding::{EmbedOptions, EmbedUsage, Embedding};
pub use crate::ai_sdk_types::image::{
    ImageData, ImageFile, ImageOptions, ImageUsage, ImageWarning,
};
pub use crate::ai_sdk_types::v2 as types;
pub use crate::ai_sdk_types::Event;
