# Benchmarking

`ai-sdk-rs` now has an offline Criterion scaffold focused on SDK overhead instead of network variance.

## What is measured

- Request translation and non-stream `do_generate` overhead on the OpenAI Responses path.
- Streaming decode and normalization using captured SSE fixtures from real provider traffic.
- Core hot paths that are shared across providers: loose JSON parsing, event-to-part mapping, and stream collection.

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
- CI only proves that the current offline scaffold compiles with `cargo bench --workspace --no-run`; local report inspection is still the only regression workflow.

## Commands

```bash
cargo bench
cargo bench --bench openai_responses
cargo bench --bench streaming_pipeline
cargo bench --bench core_hot_paths
cargo bench --bench provider_matrix
```

Criterion reports are written under `target/criterion/`.

## CI policy

CI only compiles the benches with `cargo bench --workspace --no-run`.
That keeps the harness from rotting without pretending noisy shared runners are a performance gate.
