# Rust Idiomaticity Refactor

## Objective

- Migrate Vercel AI SDK style dynamic typing (JSON property bags/deep tuples) and linker-magic (`inventory` crate) in `ai-sdk-rs` to strictly-typed, idiomatic Rust traits and structs.
- Explicit self-check target: Eliminate 100% of the relevant Clippy warnings regarding `type_complexity` and ensure core provider registration doesn't use `inventory`.

## Decision Lineage

- Earliest upstream decision: `ai-sdk-rs` was originally modeled directly after the Vercel AI SDK (node.js).
- Adjacent follow-up decision: To map JS dynamic capabilities, `HashMap<String, serde_json::Value>` and `inventory` macros were introduced to bypass rigid Rust dependency registration rules.
- Current triggering decision: An architectural audit revealed that this "TypeScript-in-Rust" approach fails WASM portability, causes type complexity warnings in `clippy`, and is broadly unidiomatic compared to libraries like `rig`.

## Relation To Prior Lineage

- This is a new plan stemming directly from the comprehensive architecture audit.

## Current Shape

- `src/provider.rs` uses `inventory::collect!` to register providers globally at startup.
- Provider options extensively use `serde_json::Value` rather than rigid configuration structs.
- Deep nested tuples (e.g., `Option<(Vec<u8>, Vec<(String, String)>)>`) are passed around directly in provider implementations, causing 49 Clippy warnings.

## Findings Being Addressed

- Linker-based `inventory` compromises cross-platform compilation targets (e.g. WASM) because of `.init_array`/`ctor` behavior.
- Dynamic property bags (JSON hash maps) circumvent Rust's type safety and cause unpredictable runtime panics.
- Nested tuples are flagged as `type_complexity` by `cargo clippy`.

## Scope

- In scope:
  - Replacing `inventory` provider registry with explicit builder instantiation or a trait-based factory registry initialized explicitly by the user.
  - Defining concrete `struct` types for the deep tuples causing `clippy` warnings (especially in `image_model_tests.rs`).
- Out of scope:
  - Rewriting existing provider endpoint behaviors.
  - Adding new language models to the standard library.

## Constraints

- Keep authority count `= 1` for provider registration logic.
- Keep path count `= 1` for builder initialization execution behavior.
- Keep data flow one-way.
- Preserve explicit parity while simplifying internals.

## Architecture Direction

- Surviving Owner: An explicitly instantiated `ProviderRegistry` or direct, public static builder functions on provider structs logic path.
- Execution Path: `Application Code -> Provider/Registry Configuration Struct -> instantiate LanguageModel`.

## Relevant Files

- `worklog/plans/plan-rust-idiomaticity.md`
- `src/provider.rs`
- `crates/providers/openai-compatible/tests/image_model_tests.rs`
- `Cargo.toml` (for `inventory` removal)

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `command git diff --cached --name-only` must remain inside that slice's declared scope.
- Before marking a slice complete or backfilling its lineage hash, reopen the active slice and rerun `command git rev-parse HEAD` plus `command git diff --cached --name-only`.
- If that refresh shows a different `HEAD` than expected or unrelated staged paths, record lineage contamination before changing the checkbox or hash.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
  `HEAD -> A0 -> A1 -> A2 -> A3 -> A4 -> A5`

## Execution Order

- Define concrete payload structs for tests to lock in safety.
- Explicate the `inventory` trait registry.
- Replace the remaining tuple-heavy test transport helpers.
- Alias the surviving anthropic message pipeline return type.
- Final validation and plan update.
- Commit the explicit self-referential final plan state.

## Atomic Slices

- [x] `A0` Establish concrete structs for complex tuples.
  Lineage commit: `d6c386c189a792f6c6f50415845a5491cd681679`
  Commit subject: `test(providers): introduce concrete structs replacing complex tuples`
  Lineage parent: `HEAD`
  Scope:
  - `crates/providers/openai-compatible/tests/image_model_tests.rs`
  Gate:
  - tests prove the `type_complexity` lint is gone explicitly.
  - no production behavior logic refactors land in this slice.

- [x] `A1` Move the registration behavior onto explicit canonical instantiation.
  Lineage commit: `c3d64eea34fea1631dd132aafac9be66c29cac80`
  Commit subject: `refactor(core): remove inventory dependency for explicit provider registry`
  Lineage parent: `A0`
  Scope:
  - `src/provider.rs`
  - `Cargo.toml`
  - `src/providers/amazon_bedrock/provider.rs`
  - `src/providers/anthropic/provider.rs`
  - `src/providers/azure/provider.rs`
  - `src/providers/gateway/provider.rs`
  - `src/providers/google/provider.rs`
  - `src/providers/google_vertex/provider.rs`
  - `src/providers/openai/provider.rs`
  - `src/providers/openai_compatible/provider.rs`
  Gate:
  - changed callers use one explicit registry initialization.
  - displaced `inventory` trait authority is removed completely in the same slice.

- [x] `A2` Replace the remaining tuple-heavy test transport helpers.
  Lineage commit: `aaf51b68878f305069cb1c8712a25c67a22ca79c`
  Commit subject: `test(providers): factor shared test transport tuple shapes into named types`
  Lineage parent: `A1`
  Scope:
  - `crates/providers/anthropic/tests/messages_tools_tests.rs`
  - `crates/providers/openai-compatible/tests/chat_language_model_tests.rs`
  - `crates/providers/openai-compatible/tests/embedding_model_tests.rs`
  Gate:
  - targeted test transports no longer emit `type_complexity` warnings.
  - no production behavior changes land in this slice.

- [x] `A3` Alias the surviving anthropic message pipeline type complexity.
  Lineage commit: `cbddcd33068d9eb57cb885de1af567cdba9dd174`
  Commit subject: `refactor(anthropic): name message pipeline result types`
  Lineage parent: `A2`
  Scope:
  - `src/providers/anthropic/messages/language_model.rs`
  Gate:
  - the anthropic message pipeline no longer emits `type_complexity`.
  - runtime behavior remains unchanged.

- [x] `A4` Validate the surviving explicit path and refresh checked-in evidence.
  Lineage commit: `d2be4dbd2d9970685b495d482095febf2a593dd7`
  Commit subject: `test(core): validate surviving provider registry path`
  Lineage parent: `A3`
  Scope:
  - focused validation only
  - plan evidence updates only (this file)
  Gate:
  - targeted validation passes `cargo clippy --tests --message-format short -- -W clippy::type_complexity` with zero type complexity warnings.
  - plan reads properly and hashes are filled.

- [x] `A5` Commit the explicit self-referential final plan state.
  Lineage commit: `<self-referential final plan commit>`
  Commit subject: `docs(plan): finalize rust idiomaticity lineage evidence`
  Lineage parent: `A4`
  Scope:
  - `worklog/plans/plan-rust-idiomaticity.md`
  Gate:
  - plan captures the landed `A0` through `A4` hashes directly.
  - the final self-referential state is explicit in the committed plan file.
  - `validate_plan_lineage.py` passes.

## Acceptance Criteria

- One explicitly-checked owner remains for provider registration behavior.
- One explicit runtime path connects config arrays directly to factories without `inventory`.
- Each checked slice maps to exactly one dedicated commit.

## Notes

- This refactor directly satisfies the primary negative constraints discovered in the `ai-sdk-rs` vs `rig` code review.
