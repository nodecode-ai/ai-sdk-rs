use std::time::Duration;

use ai_sdk_rs::core::LanguageModel;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures_util::TryStreamExt;

pub mod ai_sdk_rs {
    pub use ::ai_sdk_rs::core;
    pub use ::ai_sdk_rs::providers;
    pub use ::ai_sdk_rs::types;
}

#[path = "support/mod.rs"]
mod support;

fn bench_openai_request_translation(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let model =
        support::openai_model_with_json_response("gpt-5.1-mini", support::minimal_text_response());
    let mut group = c.benchmark_group("openai/do_generate");
    group.sample_size(30);

    for (name, options) in support::openai_request_scenarios() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &options, |b, options| {
            b.to_async(&runtime).iter(|| async {
                let response = model
                    .do_generate(black_box(options.clone()))
                    .await
                    .expect("benchmark generate response");
                black_box(response);
            });
        });
    }

    group.finish();
}

fn bench_openai_streaming(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("openai/do_stream");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(8));

    for &fixture in support::openai_stream_fixtures() {
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
                        .expect("benchmark stream response");
                    let parts = response
                        .stream
                        .try_collect::<Vec<_>>()
                        .await
                        .expect("benchmark stream parts");
                    black_box(parts);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_openai_request_translation,
    bench_openai_streaming
);
criterion_main!(benches);
