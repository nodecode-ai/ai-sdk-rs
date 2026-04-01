#![allow(dead_code)]
#![allow(unused_imports)]

pub use super::ai_sdk_rs;

mod fixture_replay;
mod openai_responses;

pub use fixture_replay::{FixtureStreamResponse, FixtureTransport};
pub use openai_responses::{
    minimal_text_response, openai_model_with_json_response, openai_model_with_stream_fixture,
    openai_request_scenarios, openai_responses_config, openai_stream_fixtures, simple_call_options,
    stream_call_options, stream_fixture_chunks, stream_fixture_size_bytes, tool_heavy_call_options,
};

pub fn benchmark_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark tokio runtime")
}
