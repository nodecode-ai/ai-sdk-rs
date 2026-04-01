use std::time::Duration;

use ai_sdk_rs::core::LanguageModel;
use ai_sdk_rs::streaming_sse::SseDecoder;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures_util::TryStreamExt;

#[path = "support/mod.rs"]
mod support;

fn bench_sse_decoder(c: &mut Criterion) {
    let fixtures = [
        "openai-web-search-tool.1",
        "openai-mcp-tool-approval.1",
        "openai-code-interpreter-tool.1",
    ];
    let mut group = c.benchmark_group("streaming/sse_decoder");
    group.sample_size(30);

    for fixture in fixtures {
        let chunks = support::stream_fixture_chunks(fixture);
        group.throughput(Throughput::Bytes(
            chunks.iter().map(|chunk| chunk.len()).sum::<usize>() as u64,
        ));
        group.bench_function(BenchmarkId::from_parameter(fixture), move |b| {
            let chunks = chunks.clone();
            b.iter(|| {
                let mut decoder = SseDecoder::new();
                let mut event_count = 0usize;
                for chunk in &chunks {
                    event_count += decoder.push(chunk.as_ref()).count();
                }
                event_count += decoder.finish().count();
                black_box(event_count);
            });
        });
    }

    group.finish();
}

fn bench_streaming_end_to_end(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let fixtures = [
        "openai-web-search-tool.1",
        "openai-mcp-tool-approval.1",
        "openai-code-interpreter-tool.1",
    ];
    let mut group = c.benchmark_group("streaming/end_to_end");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    for fixture in fixtures {
        let model = support::openai_model_with_stream_fixture("gpt-5.1-mini", fixture);
        let options = support::stream_call_options(fixture);
        group.throughput(Throughput::Bytes(
            support::stream_fixture_size_bytes(fixture) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(fixture),
            &options,
            |b, options| {
                b.to_async(&runtime).iter(|| async {
                    let response = model
                        .do_stream(black_box(options.clone()))
                        .await
                        .expect("stream benchmark response");
                    let parts = response
                        .stream
                        .try_collect::<Vec<_>>()
                        .await
                        .expect("stream benchmark parts");
                    black_box(parts);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sse_decoder, bench_streaming_end_to_end);
criterion_main!(benches);
