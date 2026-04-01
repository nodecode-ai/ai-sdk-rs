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

## Current limits

- Real captured benchmark fixtures currently exist only for the OpenAI Responses path under `crates/providers/openai/tests/fixtures/`.
- No Anthropic, Google, Google Vertex, Amazon Bedrock, Azure, Gateway, or OpenAI-compatible benchmark fixtures are checked in yet.
- CI only proves that the current offline scaffold compiles with `cargo bench --workspace --no-run`; local report inspection is still the only regression workflow.

## Commands

```bash
cargo bench
cargo bench --bench openai_responses
cargo bench --bench streaming_pipeline
cargo bench --bench core_hot_paths
```

Criterion reports are written under `target/criterion/`.

## CI policy

CI only compiles the benches with `cargo bench --workspace --no-run`.
That keeps the harness from rotting without pretending noisy shared runners are a performance gate.
