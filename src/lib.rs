pub mod core;
pub mod provider;
pub mod providers;
pub mod streaming_sse;
pub mod transport_reqwest;
pub mod types;

pub mod transports {
    pub use crate::transport_reqwest as reqwest;
}

pub(crate) use crate::core as ai_sdk_core;
pub(crate) use crate::provider as ai_sdk_provider;
pub(crate) use crate::providers::amazon_bedrock as provider_amazon_bedrock;
pub(crate) use crate::providers::anthropic as provider_anthropic;
pub(crate) use crate::providers::azure as provider_azure;
pub(crate) use crate::providers::gateway as provider_gateway;
pub(crate) use crate::providers::google as provider_google;
pub(crate) use crate::providers::google_vertex as provider_google_vertex;
pub(crate) use crate::providers::openai as provider_openai;
pub(crate) use crate::providers::openai_compatible as provider_openai_compatible;
pub(crate) use crate::providers::amazon_bedrock as ai_sdk_providers_amazon_bedrock;
pub(crate) use crate::providers::anthropic as ai_sdk_providers_anthropic;
pub(crate) use crate::providers::azure as ai_sdk_providers_azure;
pub(crate) use crate::providers::gateway as ai_sdk_providers_gateway;
pub(crate) use crate::providers::google as ai_sdk_providers_google;
pub(crate) use crate::providers::google_vertex as ai_sdk_providers_google_vertex;
pub(crate) use crate::providers::openai as ai_sdk_providers_openai;
pub(crate) use crate::providers::openai_compatible as ai_sdk_providers_openai_compatible;
pub(crate) use crate::streaming_sse as ai_sdk_streaming_sse;
pub(crate) use crate::transport_reqwest as reqwest_transport;
pub(crate) use crate::types as ai_sdk_types;
