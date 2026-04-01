# V2 Request Surface Single-Authority Remediation

## Objective

- Keep typed provider builders plus `types::v2` call/stream structs as the only public request-authoring authority for language-model usage in ai-sdk-rs.
- Restore one explicit onboarding and execution path: docs/examples -> typed provider builder -> `v2::CallOptions` -> `LanguageModel::{do_generate,do_stream}` -> `GenerateResponse` or `StreamResponse`, with the self-check target that request-authority count `= 1` and request-path count `= 1`.

## Decision Lineage

- Earliest upstream decision: `dec_20260401_034234_fb402fc3` chose the broader ai-sdk-rs simplification rule of keeping a small shared core and re-adding heavier capability only at the true owner.
- Adjacent follow-up decisions: `dec_20260401_035329_adec5d7c` localized reqwest features to the actual owner, and `dec_20260401_040221_fc5055c0` continued the same feature-fanout pruning by re-adding websocket and trace features only where they are truly owned.
- Current local adjacent node: `dec_20260401_202534_37c69f86` explicitly rejected a compatibility alias and kept ai-sdk-rs on the hyper-owned surface instead of carrying a second stale contract.
- Current triggering node: the 2026-04-01 next-best-slop-removal audit in this session identified the legacy `ChatRequest` / `ChatMessage` DSL and its stale example workflow as the highest-value remaining duplicate surface.
- `search-decision --related /home/mike/nodecode/ai-sdk-rs/src/types/mod.rs` and `search-decision --related /home/mike/nodecode/ai-sdk-rs/README.md` returned no direct prior decision for this exact seam, so this plan is a fresh follow-up to the adjacent single-owner lineage rather than a reopened prior plan.

## Relation To Prior Lineage

- This plan follows `worklog/plans/plan-provider-construction-single-path-remediation.md`, which established typed provider builders as the documented happy-path owner for model construction.
- This plan also follows `worklog/plans/plan-single-module-tree-remediation.md`, which removed pseudo-crate topology but intentionally preserved public API parity.
- Those plans simplified construction and module ownership, but they left the older request DSL and stale example story alive as a second public path. This plan narrows that remaining public seam.

## Current Shape

- The live language-model trait is already v2-only: providers implement `do_generate` and `do_stream` over `types::v2` call options and stream parts.
- The workspace only carries `embed-openai-compatible`, `generate-stream`, and `openai-reasoning-summary`, but the README still advertises `cargo run -p generate-text`.
- `src/types/mod.rs` still exports `Role`, `ChatMessage`, `ToolSpec`, and `ChatRequest`, which advertise an older request-authoring workflow not used by the canonical v2 examples.
- Out-of-workspace examples such as `generate-text`, `anthropic-thinking`, `anthropic-hello-twice`, and `github-models.rs` still depend on that older request workflow or stale provider exports, so the repo currently carries two incompatible public stories.

## Findings Being Addressed

- The crate already has one working request authority, but the public surface still carries a second request DSL that is not the canonical execution path.
- The README and example inventory drift because the legacy DSL keeps stale examples alive outside the workspace and outside routine validation.
- Public-surface tests still pin the legacy request DSL as part of the exported contract, which blocks the repo from honestly presenting the v2 path as the sole authority.

## Scope

- In scope:
- lock the legacy request seam with focused surface tests and evidence
- rewrite `generate-text` onto the surviving v2 path and add it to workspace validation
- delete the orphaned legacy-request examples that no longer justify active maintenance
- remove the displaced legacy request DSL from the public surface and stop pinning it in surface tests
- refresh README language so the onboarding story names only the surviving v2 request path
- Out of scope:
- removing `Event`, `TokenUsage`, or `streaming_sse`, which still serve internal normalization and benchmark seams
- changing provider runtime behavior beyond request-authoring entrypoints and example callsites
- removing `openai-compatible-completion` or transport alias residue, which are adjacent follow-up seams

## Constraints

- Keep authority count `= 1` for public language-model request authoring.
- Keep path count `= 1` for docs/examples to provider execution.
- Keep data flow one-way from typed provider builder inputs to `v2::CallOptions` and provider runtime execution.
- Preserve working provider behavior while shrinking the public request surface.
- Do not add compatibility shims for the removed legacy request DSL.

## Architecture Direction

- Surviving owner: typed provider builders under `providers::*` plus `types::v2::{CallOptions, PromptMessage, StreamPart}`.
- Surviving path: docs/examples -> typed provider builder -> `v2::CallOptions` assembly -> `LanguageModel::{do_generate,do_stream}` -> `GenerateResponse` or `StreamResponse`.
- Displaced authority to delete: `types::mod.rs` legacy request DSL (`Role`, `ChatMessage`, `ToolSpec`, `ChatRequest`) plus the stale example workflow that depends on it.

## Relevant Files

- `worklog/plans/plan-v2-request-surface-single-authority-remediation.md`
- `Cargo.toml`
- `README.md`
- `src/core/v2/mod.rs`
- `src/types/mod.rs`
- `tests/single_module_tree_layout_tests.rs`
- `examples/generate-text/Cargo.toml`
- `examples/generate-text/src/main.rs`
- `examples/anthropic-thinking/Cargo.toml`
- `examples/anthropic-thinking/src/main.rs`
- `examples/anthropic-hello-twice/Cargo.toml`
- `examples/anthropic-hello-twice/src/main.rs`
- `examples/github-models.rs`

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
- `HEAD -> RQ0 -> RQ1 -> RQ2 -> RQ3`

## Execution Order

- Lock the legacy request seam first.
- Move onboarding examples and docs onto the surviving v2 path.
- Delete the displaced legacy request DSL only after no documented or checked-in caller depends on it.
- Refresh validation and plan evidence last.

## Atomic Slices

- [x] `RQ0` Lock the legacy request seam with focused surface evidence.
  Lineage commit: `6b0516dee7848f6ef6cf879530f3603429d5b2f3`
  Commit subject: `test(api): lock legacy request seam`
  Lineage parent: `HEAD`
  Scope:
  - focused surface tests around `src/types/mod.rs`, README example claims, and workspace example membership
  - directly affected test files only
  - `worklog/plans/plan-v2-request-surface-single-authority-remediation.md`
  Gate:
  - tests make the current legacy request surface and stale example/workspace mismatch explicit
  - no production request-surface removal lands in this slice

- [x] `RQ1` Move onboarding examples and docs onto the v2 request authority.
  Lineage commit: `64e016396f57a2446e157816149c94daac3b397e`
  Commit subject: `refactor(examples): move onboarding to v2 request surface`
  Lineage parent: `RQ0`
  Scope:
  - `Cargo.toml`
  - `README.md`
  - `examples/generate-text/**`
  - `examples/anthropic-thinking/**`
  - `examples/anthropic-hello-twice/**`
  - `examples/github-models.rs`
  - directly affected example validation only
  Gate:
  - `generate-text` is rewritten onto typed provider builders plus `v2::CallOptions` and becomes a workspace-validated example
  - orphaned legacy-request Anthropic and GitHub example paths are deleted rather than left stale
  - README points only at buildable examples that follow the surviving v2 path

- [x] `RQ2` Remove the displaced legacy request DSL from the public surface.
  Lineage commit: `8144a937e4d253137ee006eaba0ff87e4f7044a9`
  Commit subject: `refactor(api): remove legacy request dsl`
  Lineage parent: `RQ1`
  Scope:
  - `src/types/mod.rs`
  - `tests/single_module_tree_layout_tests.rs`
  - directly affected imports, compile proofs, and request-surface tests only
  Gate:
  - `Role`, `ChatMessage`, `ToolSpec`, and `ChatRequest` no longer survive as public request-authoring authorities
  - no checked-in example or surface proof still depends on the removed DSL
  - the surviving public request path is only the v2 builder plus `CallOptions` path

- [x] `RQ3` Validate the surviving v2 request surface and refresh checked-in evidence.
  Lineage commit: `<self; backfill after landing RQ3>`
  Commit subject: `test(api): validate v2 request surface`
  Lineage parent: `RQ2`
  Scope:
  - targeted `cargo check` / `cargo test` validation for the surviving examples and surface tests
  - `README.md` and plan evidence updates only if needed
  - `worklog/plans/plan-v2-request-surface-single-authority-remediation.md`
  Gate:
  - targeted validation passes on the surviving v2 request path
  - checked-in docs and examples match the landed ownership shape
  - the plan records only one surviving request authority and path

## Validation Checklist

- `cargo check -p generate-text`
- `cargo check -p generate-stream`
- `cargo check -p openai-reasoning-summary`
- `cargo check -p embed-openai-compatible`
- `cargo test --test single_module_tree_layout_tests`
- any focused surface test added in `RQ0`

## Validation Evidence

- `2026-04-01`: `cargo check -p generate-text -p generate-stream -p openai-reasoning-summary -p embed-openai-compatible`
- `2026-04-01`: `cargo test --test single_module_tree_layout_tests`

## Acceptance Criteria

- One checked-in public request-authoring owner remains: typed provider builders plus `types::v2` call/stream structs.
- One explicit docs/examples path remains from onboarding callsite to provider execution.
- The README references only workspace-validated examples that build on the surviving path.
- The legacy request DSL no longer survives as a public contract.
- Each checked slice maps to exactly one dedicated commit.

## Notes

- The 2026-04-01 slop-removal audit picked this seam over `openai-compatible-completion` and transport alias cleanup because the legacy request DSL currently causes direct onboarding drift and advertises a second public story that no longer matches the crate's canonical runtime surface.
