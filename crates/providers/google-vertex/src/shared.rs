//! Google Vertex internal boundary facade over `provider_google::shared`.
//!
//! This module intentionally centralizes cross-provider reuse so Vertex call
//! sites depend on a local seam rather than scattered path-indirection.

pub(crate) use crate::provider_google::shared::error::map_transport_error_to_sdk_error;
pub(crate) use crate::provider_google::shared::options::{
    parse_google_provider_options_for_scopes, GoogleProviderOptions,
};
pub use crate::provider_google::shared::prepare_tools::{
    convert_json_schema_to_openapi_schema, prepare_tools, PreparedTools,
};
pub(crate) use crate::provider_google::shared::prompt::convert_to_google_prompt_with_scopes;
pub use crate::provider_google::shared::prompt::GooglePrompt;
pub(crate) use crate::provider_google::shared::stream_core::build_google_stream_part_stream;
