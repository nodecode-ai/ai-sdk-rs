# Provider Construction Single-Path Remediation

## Objective

- Keep typed provider constructors and builders as the only documented happy-path owner for `LanguageModel`, `EmbeddingModel`, and `ImageModel` construction; narrow the registry to advanced dynamic-definition use only.
- Restore one explicit construction path: docs and examples -> typed provider builder -> provider config assembly -> concrete model type. Self-check target: authority count `= 1` for happy-path model construction, path count `= 1` from callsite to model instantiation.

## Current Shape

- OpenAI streaming examples use `OpenAIResponsesLanguageModel::create_simple`.
- The README shows an Anthropic happy path built through `registry::iter()` plus a handwritten `ProviderDefinition`.
- OpenAI-compatible embeddings use direct `new(...)` construction, and provider modules also expose registry builders and alias registrations.
- The current surface makes it hard to tell which construction path is the blessed user-facing one versus which paths exist for dynamic provider catalogs or internal composition.

## Findings Being Addressed

- The same capability is exposed through overlapping constructor surfaces.
- The generic registry layer and per-provider builder shells add avoidable happy-path indirection.
- Docs and examples drift because there is no single blessed construction story for ordinary callers.

## Scope

- In scope:
- define one canonical typed construction surface per capability and provider family
- move README and examples onto that canonical construction path
- narrow redundant happy-path helpers and registry-facing entrypoints so the primary path is unambiguous
- add focused compile coverage for the surviving construction surface
- Out of scope:
- deleting the provider registry entirely
- stream normalization refactors
- internal module-tree restructuring covered by `plan-single-module-tree-remediation.md`

## Constraints

- Keep authority count `= 1` for happy-path model construction.
- Keep path count `= 1` from public callsite to concrete provider model instantiation.
- Keep data flow one-way from typed config inputs to provider-specific model assembly.
- Preserve provider parity while simplifying the public construction story.

## Architecture Direction

- Surviving owner: typed provider entrypoints under `providers::*`, with one consistent builder pattern per capability.
- Surviving path: docs and examples -> typed provider constructor or builder -> provider config assembly -> concrete model type -> trait object or typed model use.

## Relevant Files

- `worklog/plans/plan-provider-construction-single-path-remediation.md`
- `README.md`
- `examples/generate-stream/src/main.rs`
- `examples/openai-reasoning-summary/src/main.rs`
- `examples/embed-openai-compatible/src/main.rs`
- `crates/provider/src/lib.rs`
- `crates/providers/openai/src/provider.rs`
- `crates/providers/openai/src/responses/language_model.rs`
- `crates/providers/openai-compatible/src/provider.rs`
- `crates/providers/anthropic/src/provider.rs`
- focused constructor tests under `tests/**` or provider test modules

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `git diff --cached --name-only` must remain inside that slice's declared scope.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
- `HEAD -> PC0 -> PC1 -> PC2 -> PC3`

## Execution Order

- Lock representative construction surfaces first.
- Introduce the canonical typed constructor path before narrowing redundant helpers.
- Move docs and examples onto the surviving path and demote registry use to advanced-only last.
- Re-run compile validation and refresh checked-in evidence after the surface is singular.

## Atomic Slices

- [x] `PC0` Lock representative construction surfaces with focused compile coverage.
  Lineage commit: `d2af584`
  Commit subject: `test(api): lock provider construction happy paths`
  Lineage parent: `HEAD`
  Scope:
  - focused compile tests for representative OpenAI, Anthropic, and OpenAI-compatible construction paths
  - current example coverage and `README.md` evidence only
  - `worklog/plans/plan-provider-construction-single-path-remediation.md`
  Gate:
  - current construction seams are pinned before constructor refactor work lands
  - tests distinguish happy-path construction from advanced registry construction
  - no production constructor refactor lands in this slice

- [x] `PC1` Introduce one canonical typed construction surface per capability.
  Lineage commit: `<fill after landing>`
  Commit subject: `refactor(api): add canonical provider constructors`
  Lineage parent: `PC0`
  Scope:
  - provider constructor APIs under `crates/providers/**`
  - directly affected compile tests only
  Gate:
  - one consistent typed constructor or builder shape exists for the documented happy path
  - representative providers no longer require registry-driven construction in ordinary usage
  - overlapping helper entrypoints are narrowed rather than multiplied

- [ ] `PC2` Move docs and callers to the canonical construction path and demote displaced authorities.
  Lineage commit: `<pending>`
  Commit subject: `refactor(api): narrow redundant provider construction paths`
  Lineage parent: `PC1`
  Scope:
  - `README.md`
  - active examples under `examples/**`
  - registry-facing or helper construction entrypoints directly affected by the new canonical surface
  Gate:
  - docs and examples all use the surviving constructor path
  - registry use is clearly advanced-only rather than the default public story
  - redundant happy-path helpers are removed or explicitly narrowed in the same slice

- [ ] `PC3` Validate the surviving provider construction surface.
  Lineage commit: `<pending>`
  Commit subject: `test(api): validate provider construction surface`
  Lineage parent: `PC2`
  Scope:
  - targeted `cargo check` and `cargo test` coverage for docs/examples and representative constructor tests
  - `worklog/plans/plan-provider-construction-single-path-remediation.md`
  Gate:
  - targeted construction coverage passes on the surviving path
  - checked-in docs and examples match the landed ownership shape
  - no unrelated provider runtime behavior changes land in this slice

## Acceptance Criteria

- One documented happy-path constructor surface remains per capability and provider family.
- One explicit path remains from example or doc callsite to concrete model instantiation.
- Registry-driven construction remains available only where dynamic provider catalogs actually need it.
- Each checked slice maps to exactly one dedicated commit.
