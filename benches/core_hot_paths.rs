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
use serde_json::json;

pub mod ai_sdk_rs {
    pub use ::ai_sdk_rs::core;
    pub use ::ai_sdk_rs::providers;
    pub use ::ai_sdk_rs::types;
}

#[path = "support/mod.rs"]
mod support;

fn bench_parse_json_loose(c: &mut Criterion) {
    const STRICT_JSON: &str =
        "{\"summary\":\"ok\",\"sources\":[\"streaming\",\"providers\",\"json\"]}";
    const NOISY_JSON: &str = "tool-call-start<<<{\"summary\":\"ok\",\"sources\":[\"streaming\",\"providers\",\"json\"],\"nested\":{\"a\":[1,2,3],\"b\":{\"c\":\"d\"}}}>>>tool-call-end";

    let mut group = c.benchmark_group("core/parse_json_loose");
    group.bench_function("strict_json", |b| {
        b.iter(|| black_box(parse_json_loose(black_box(STRICT_JSON))));
    });
    group.bench_function("noisy_fragment", |b| {
        b.iter(|| black_box(parse_json_loose(black_box(NOISY_JSON))));
    });
    group.finish();
}

fn bench_map_events_to_parts(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let events = synthetic_provider_events();
    let mut group = c.benchmark_group("core/map_events_to_parts");
    group.sample_size(30);
    group.throughput(Throughput::Elements(events.len() as u64));
    group.bench_function("mixed_stream", |b| {
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
    group.finish();
}

fn bench_collect_stream_to_response(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let parts = synthetic_stream_parts();
    let mut group = c.benchmark_group("core/collect_stream_to_response");
    group.sample_size(30);
    group.throughput(Throughput::Elements(parts.len() as u64));
    group.bench_function("tool_rich_parts", |b| {
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

fn synthetic_provider_events() -> Vec<Event> {
    let mut events = Vec::new();
    for _ in 0..32 {
        events.push(Event::TextDelta {
            delta: "chunk ".into(),
        });
    }
    events.push(Event::ReasoningStart {
        id: "reasoning:0".into(),
    });
    for _ in 0..16 {
        events.push(Event::ReasoningDelta {
            delta: "step ".into(),
        });
    }
    events.push(Event::ReasoningEnd);
    events.push(Event::ToolCallStart {
        id: "call_0".into(),
        name: "lookup_docs".into(),
    });
    for delta in ["{\"query\":", "\"streaming pipeline\"", ",\"limit\":", "5}"] {
        events.push(Event::ToolCallDelta {
            id: "call_0".into(),
            args_json: delta.into(),
        });
    }
    events.push(Event::ToolCallEnd {
        id: "call_0".into(),
    });
    events.push(Event::Usage {
        usage: ai_sdk_rs::types::TokenUsage::new(128, 32),
    });
    events.push(Event::Done);
    events
}

fn synthetic_stream_parts() -> Vec<v2t::StreamPart> {
    vec![
        v2t::StreamPart::StreamStart {
            warnings: Vec::new(),
        },
        v2t::StreamPart::TextStart {
            id: "text:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::TextDelta {
            id: "text:0".into(),
            delta: "SDK ".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::TextDelta {
            id: "text:0".into(),
            delta: "overhead".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::TextEnd {
            id: "text:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningStart {
            id: "reasoning:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningDelta {
            id: "reasoning:0".into(),
            delta: "trace the translation path".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningSignature {
            signature: "sig_bench".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ReasoningEnd {
            id: "reasoning:0".into(),
            provider_metadata: None,
        },
        v2t::StreamPart::ToolCall(v2t::ToolCallPart {
            tool_call_id: "call_0".into(),
            tool_name: "lookup_docs".into(),
            input: json!({
                "query": "streaming pipeline"
            })
            .to_string(),
            provider_executed: false,
            provider_metadata: None,
            dynamic: false,
            provider_options: None,
        }),
        v2t::StreamPart::ToolResult {
            tool_call_id: "call_0".into(),
            tool_name: "lookup_docs".into(),
            result: json!({
                "matches": 5
            }),
            is_error: false,
            preliminary: false,
            provider_metadata: None,
        },
        v2t::StreamPart::Finish {
            usage: v2t::Usage {
                input_tokens: Some(128),
                output_tokens: Some(64),
                total_tokens: Some(192),
                reasoning_tokens: Some(16),
                cached_input_tokens: Some(32),
            },
            finish_reason: v2t::FinishReason::Stop,
            provider_metadata: None,
        },
    ]
}

criterion_group!(
    benches,
    bench_parse_json_loose,
    bench_map_events_to_parts,
    bench_collect_stream_to_response
);
criterion_main!(benches);
