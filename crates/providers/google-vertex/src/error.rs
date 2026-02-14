use crate::ai_sdk_core::error::{SdkError, TransportError};

pub fn map_transport_error_to_sdk_error(te: TransportError) -> SdkError {
    crate::provider_google_vertex::shared::map_transport_error_to_sdk_error(te)
}
