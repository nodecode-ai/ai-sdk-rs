# Quant Gate Log (`ai-sdk-rs-2zp`)

Generated: 2026-02-14T03:37:00Z (UTC)

## Snapshot Method
- Per-slice code delta snapshot: `git diff --numstat <commit>^ <commit> -- '*.rs'`
- Coupling compare metric: `crate::provider_google::shared::` references inside `crates/providers/google-vertex/src`, excluding `crates/providers/google-vertex/src/shared.rs`
- Workspace LOC snapshot: `tokei src crates examples --output json` (Rust `code`)

## Workspace Snapshot
- Rust LOC (`tokei`): `23341`
- `cargo test --lib`: pass (`55 passed, 0 failed`)
- `cargo check`: pass

## Per-Slice Compare Verdicts

| Slice | Commit | Rust Delta (+/-) | Coupling Metric (before -> after) | Targeted Gate Evidence | Verdict |
| --- | --- | ---: | --- | --- | --- |
| `ai-sdk-rs-4kn` | `69d56f7` | `+0 / -0` | `0 -> 0` | `cargo test --lib provider_openai::responses_language_model_tests::non_stream_response_error_returns_error` (pass) | `no-regression` |
| `ai-sdk-rs-mpj` | `b44c5b1` | `+0 / -0` | `0 -> 0` | `cargo test --lib provider_openai::stream_fixture_tests::stream_error_fixture` (pass) | `no-regression` |
| `ai-sdk-rs-gsu` | `d5f6285` | `+56 / -5` | `0 -> 0` | `cargo test --lib transport_reqwest::tests::` (pass) | `no-regression` |
| `ai-sdk-rs-1ku` | `148be79` | `+261 / -0` | `7 -> 7` | `cargo test --lib provider_google::parity_regression_tests::` (pass) | `no-regression` |
| `ai-sdk-rs-2v2` | `0dca926` | `+32 / -16` | `7 -> 0` | `cargo test --lib provider_google::parity_regression_tests::prepare_tools_provider_path_parity_between_google_and_vertex` (pass) | `improves` |

## Gate Decision
All required slices for `ai-sdk-rs-2zp` meet the compare policy (`improves` or `no-regression`).
