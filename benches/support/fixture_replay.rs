use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream;
use serde_json::Value;

use super::ai_sdk_rs::core::error::TransportError;
use super::ai_sdk_rs::core::transport::{HttpTransport, TransportConfig, TransportStream};

#[derive(Clone)]
pub struct FixtureTransport {
    json_response: Arc<Value>,
    chunks: Arc<Vec<Bytes>>,
}

pub struct FixtureStreamResponse {
    chunks: Arc<Vec<Bytes>>,
}

impl FixtureTransport {
    pub fn new(json_response: Value, chunks: Vec<Bytes>) -> Self {
        Self {
            json_response: Arc::new(json_response),
            chunks: Arc::new(chunks),
        }
    }
}

#[async_trait]
impl HttpTransport for FixtureTransport {
    type StreamResponse = FixtureStreamResponse;

    fn into_stream(resp: Self::StreamResponse) -> (TransportStream, Vec<(String, String)>) {
        let chunks = (*resp.chunks).clone();
        (
            Box::pin(stream::iter(
                chunks.into_iter().map(Ok::<Bytes, TransportError>),
            )),
            vec![],
        )
    }

    async fn post_json_stream(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<Self::StreamResponse, TransportError> {
        Ok(FixtureStreamResponse {
            chunks: Arc::clone(&self.chunks),
        })
    }

    async fn post_json(
        &self,
        _url: &str,
        _headers: &[(String, String)],
        _body: &Value,
        _cfg: &TransportConfig,
    ) -> Result<(Value, Vec<(String, String)>), TransportError> {
        Ok(((*self.json_response).clone(), vec![]))
    }
}
