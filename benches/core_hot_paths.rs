use std::collections::HashSet;

use ai_sdk_rs::core::stream_collect::{collect_stream_to_response, StreamCollectorConfig};
use ai_sdk_rs::core::{
    map_events_to_parts, EventMapperConfig, EventMapperHooks, SdkError, StreamResponse,
};
use ai_sdk_rs::types::json::parse_json_loose;
use ai_sdk_rs::types::v2 as v2t;
use ai_sdk_rs::types::Event;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use futures_util::{stream, TryStreamExt};

pub mod ai_sdk_rs {
    pub use ::ai_sdk_rs::core;
    pub use ::ai_sdk_rs::providers;
    pub use ::ai_sdk_rs::types;
}

#[path = "support/mod.rs"]
mod support;

fn bench_parse_json_loose(c: &mut Criterion) {
    let mut group = c.benchmark_group("core/parse_json_loose");
    for scenario in support::json_parse_scenarios() {
        group.throughput(Throughput::Bytes(scenario.payload.len() as u64));
        group.bench_function(scenario.name, |b| {
            let payload = &scenario.payload;
            b.iter(|| black_box(parse_json_loose(black_box(payload.as_str()))));
        });
    }
    group.finish();
}

fn bench_map_events_to_parts(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("core/map_events_to_parts");
    group.sample_size(20);
    for scenario in support::event_mapping_scenarios() {
        group.throughput(Throughput::Elements(scenario.events.len() as u64));
        group.bench_function(scenario.name, |b| {
            let events = &scenario.events;
            b.to_async(&runtime).iter(|| async {
                let events = black_box(events.clone());
                let stream = Box::pin(stream::iter(events.into_iter().map(Ok::<Event, SdkError>)));
                let parts = map_events_to_parts(stream, mapper_config())
                    .try_collect::<Vec<_>>()
                    .await
                    .expect("mapped stream parts");
                black_box(parts);
            });
        });
    }
    group.finish();
}

fn bench_collect_stream_to_response(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("core/collect_stream_to_response");
    group.sample_size(20);
    for scenario in support::stream_collection_scenarios() {
        group.throughput(Throughput::Elements(scenario.parts.len() as u64));
        group.bench_function(scenario.name, |b| {
            let parts = &scenario.parts;
            b.to_async(&runtime).iter(|| async {
                let parts = black_box(parts.clone());
                let response = collect_stream_to_response(
                    StreamResponse {
                        stream: Box::pin(stream::iter(
                            parts.into_iter().map(Ok::<v2t::StreamPart, SdkError>),
                        )),
                        request_body: None,
                        response_headers: None,
                    },
                    StreamCollectorConfig {
                        allow_reasoning: true,
                        reasoning_metadata_scope: Some("openai"),
                        allow_tool_calls: true,
                        allow_tool_results: true,
                        allow_files: false,
                        allow_source_urls: false,
                        fail_on_error: true,
                    },
                )
                .await
                .expect("collected generate response");
                black_box(response);
            });
        });
    }
    group.finish();
}

fn bench_provider_parse_hot_paths(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("core/provider_parse_hot_paths");
    group.sample_size(10);

    for scenario in support::provider_parse_scenarios() {
        group.throughput(Throughput::Bytes(scenario.bytes));
        group.bench_function(scenario.name, |b| {
            let run = scenario.run;
            b.to_async(&runtime).iter(run);
        });
    }

    group.finish();
}

fn mapper_config() -> EventMapperConfig<()> {
    EventMapperConfig {
        warnings: Vec::new(),
        treat_tool_names_as_text: HashSet::new(),
        default_text_id: "text:0",
        finish_reason_fallback: v2t::FinishReason::Stop,
        initial_extra: (),
        hooks: EventMapperHooks::default(),
    }
}

criterion_group!(
    benches,
    bench_parse_json_loose,
    bench_map_events_to_parts,
    bench_collect_stream_to_response,
    bench_provider_parse_hot_paths
);
criterion_main!(benches);
