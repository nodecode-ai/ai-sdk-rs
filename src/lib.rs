#[path = "../crates/sdk-types/src/lib.rs"]
pub mod types;
#[path = "../crates/core/src/lib.rs"]
pub mod core;
#[path = "../crates/streaming-sse/src/lib.rs"]
pub mod streaming_sse;
#[path = "../crates/provider/src/lib.rs"]
pub mod provider;
#[path = "../crates/transports/reqwest/src/lib.rs"]
pub mod transport_reqwest;

#[path = "../crates/providers/openai-compatible/src/lib.rs"]
pub mod provider_openai_compatible;
#[path = "../crates/providers/openai/src/lib.rs"]
pub mod provider_openai;
#[path = "../crates/providers/azure/src/lib.rs"]
pub mod provider_azure;
#[path = "../crates/providers/gateway/src/lib.rs"]
pub mod provider_gateway;
#[path = "../crates/providers/anthropic/src/lib.rs"]
pub mod provider_anthropic;
#[path = "../crates/providers/google/src/lib.rs"]
pub mod provider_google;
#[path = "../crates/providers/google-vertex/src/lib.rs"]
pub mod provider_google_vertex;
#[path = "../crates/providers/amazon-bedrock/src/lib.rs"]
pub mod provider_amazon_bedrock;

pub mod transports {
    pub use crate::transport_reqwest as reqwest;
}

pub mod providers {
    pub use crate::provider_amazon_bedrock as amazon_bedrock;
    pub use crate::provider_anthropic as anthropic;
    pub use crate::provider_azure as azure;
    pub use crate::provider_gateway as gateway;
    pub use crate::provider_google as google;
    pub use crate::provider_google_vertex as google_vertex;
    pub use crate::provider_openai as openai;
    pub use crate::provider_openai_compatible as openai_compatible;
}

pub(crate) use crate::core as ai_sdk_core;
pub(crate) use crate::provider as ai_sdk_provider;
#[allow(unused_imports)]
pub(crate) use crate::provider_amazon_bedrock as ai_sdk_providers_amazon_bedrock;
#[allow(unused_imports)]
pub(crate) use crate::provider_anthropic as ai_sdk_providers_anthropic;
#[allow(unused_imports)]
pub(crate) use crate::provider_azure as ai_sdk_providers_azure;
#[allow(unused_imports)]
pub(crate) use crate::provider_gateway as ai_sdk_providers_gateway;
#[allow(unused_imports)]
pub(crate) use crate::provider_google as ai_sdk_providers_google;
#[allow(unused_imports)]
pub(crate) use crate::provider_google_vertex as ai_sdk_providers_google_vertex;
pub(crate) use crate::provider_openai as ai_sdk_providers_openai;
#[allow(unused_imports)]
pub(crate) use crate::provider_openai_compatible as ai_sdk_providers_openai_compatible;
pub(crate) use crate::streaming_sse as ai_sdk_streaming_sse;
pub(crate) use crate::transport_reqwest as reqwest_transport;
pub(crate) use crate::types as ai_sdk_types;
