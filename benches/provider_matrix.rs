use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

pub mod ai_sdk_rs {
    pub use ::ai_sdk_rs::core;
    pub use ::ai_sdk_rs::providers;
    pub use ::ai_sdk_rs::types;
}

#[path = "support/mod.rs"]
mod support;

fn bench_provider_generate_matrix(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("providers/generate_matrix");
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("openai"), |b| {
        b.to_async(&runtime).iter(support::run_openai_generate);
    });
    group.bench_function(BenchmarkId::from_parameter("azure"), |b| {
        b.to_async(&runtime).iter(support::run_azure_generate);
    });
    group.bench_function(BenchmarkId::from_parameter("bedrock"), |b| {
        b.to_async(&runtime).iter(support::run_bedrock_generate);
    });
    group.bench_function(BenchmarkId::from_parameter("gateway"), |b| {
        b.to_async(&runtime).iter(support::run_gateway_generate);
    });
    group.bench_function(BenchmarkId::from_parameter("google"), |b| {
        b.to_async(&runtime).iter(support::run_google_generate);
    });
    group.bench_function(BenchmarkId::from_parameter("google-vertex"), |b| {
        b.to_async(&runtime).iter(support::run_google_vertex_generate);
    });

    group.finish();
}

fn bench_provider_stream_matrix(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("providers/stream_matrix");
    group.sample_size(20);

    group.bench_function(BenchmarkId::from_parameter("anthropic"), |b| {
        b.to_async(&runtime).iter(support::run_anthropic_stream);
    });
    group.bench_function(BenchmarkId::from_parameter("openai-compatible"), |b| {
        b.to_async(&runtime)
            .iter(support::run_openai_compatible_stream);
    });

    group.finish();
}

fn bench_concurrent_replay_matrix(c: &mut Criterion) {
    let runtime = support::benchmark_runtime();
    let mut group = c.benchmark_group("providers/concurrent_replay");
    group.sample_size(10);

    for scenario in support::concurrent_replay_scenarios() {
        group.throughput(Throughput::Bytes(scenario.bytes));
        group.bench_function(BenchmarkId::from_parameter(scenario.name), |b| {
            let run = scenario.run;
            b.to_async(&runtime).iter(run);
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_provider_generate_matrix,
    bench_provider_stream_matrix,
    bench_concurrent_replay_matrix
);
criterion_main!(benches);
