#![allow(dead_code)]
#![allow(unused_imports)]

pub use super::ai_sdk_rs;

mod fixture_replay;
mod openai_responses;
mod provider_matrix;

pub use fixture_replay::{FixtureStreamResponse, FixtureTransport};
pub use openai_responses::{
    minimal_text_response, openai_model_with_json_response, openai_model_with_stream_fixture,
    openai_request_scenarios, openai_responses_config, openai_stream_fixtures, simple_call_options,
    stream_call_options, stream_fixture_chunks, stream_fixture_size_bytes, tool_heavy_call_options,
};
pub use provider_matrix::{
    event_mapping_scenarios, json_parse_scenarios, provider_matrix_families,
    provider_parse_scenarios, run_anthropic_stream, run_azure_generate, run_bedrock_generate,
    run_gateway_generate, run_google_generate, run_google_vertex_generate,
    run_openai_compatible_stream, run_openai_generate, stream_collection_scenarios,
};

pub fn benchmark_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark tokio runtime")
}
