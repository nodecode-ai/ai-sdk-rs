# Benchmarking

`ai-sdk-rs` now has an offline Criterion scaffold focused on SDK overhead instead of network variance.

## What is measured

- Request translation and non-stream `do_generate` overhead on the OpenAI Responses path.
- Streaming decode and normalization using captured SSE fixtures from real provider traffic.
- Core hot paths that are shared across providers: loose JSON parsing, event-to-part mapping, stream collection, and provider-specific parse workloads under large or adversarial inputs.
- Concurrent replay and backpressure-oriented scenarios on the shared provider matrix harness.

## Why offline fixtures

- Network latency would drown out the abstraction cost this crate actually controls.
- Fixture-driven runs are deterministic enough to compare branches locally.
- CI can compile the benchmarks without needing provider credentials.

## Benchmark suites

- `openai_responses`: request translation plus end-to-end `do_stream` on real captured chunks.
- `streaming_pipeline`: SSE decoder and streaming replay benchmarks.
- `core_hot_paths`: provider-agnostic utility and normalization benchmarks.
- `provider_matrix`: one shared-harness request or streaming scenario for OpenAI, Azure, Anthropic, Google, Google Vertex, Bedrock, Gateway, and OpenAI-compatible families.

## Current limits

- The provider matrix now covers all supported provider families, but only the OpenAI path currently uses captured production fixture files under `crates/providers/openai/tests/fixtures/`.
- Anthropic, Gateway, and OpenAI-compatible matrix scenarios still rely on checked-in representative wire-shape fixtures rather than broader captured corpora, and several non-OpenAI families still benchmark minimal synthetic responses instead of captured production transcripts.
- The scale and adversarial suite now covers large payloads, malformed frames, and fragmented chunk boundaries on the shared harness, but it still stops short of live-provider traffic, memory profiling, or concurrent replay baselines.
- CI only proves that the current offline scaffold compiles with `cargo bench --workspace --no-run`; local report inspection is still the only regression workflow.

## Commands

```bash
cargo bench
cargo bench --bench openai_responses
cargo bench --bench streaming_pipeline
cargo bench --bench core_hot_paths
cargo bench --bench provider_matrix
```

## Local baselines

Use the shared scripts when you want a repeatable local save/compare workflow across the full offline suite:

```bash
bash scripts/benchmark-save-baseline.sh main
bash scripts/benchmark-compare-baseline.sh main
```

Pass explicit bench names after the baseline if you only want a subset, for example:

```bash
bash scripts/benchmark-save-baseline.sh main core_hot_paths provider_matrix
bash scripts/benchmark-compare-baseline.sh main core_hot_paths provider_matrix
```

Criterion reports are written under `target/criterion/`.

## CI policy

CI only compiles the benches with `cargo bench --workspace --no-run`.
That keeps the harness from rotting without pretending noisy shared runners are a performance gate.
