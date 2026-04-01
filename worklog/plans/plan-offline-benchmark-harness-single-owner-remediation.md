# Offline Benchmark Harness Single-Owner Remediation

## Objective

- Keep one shared offline benchmark harness as the only checked-in owner of fixture replay, scenario manifests, measurement presets, and local regression-export plumbing across provider families; individual bench entrypoints should become thin registrations over that owner.
- Restore one explicit performance-evidence path: captured provider fixture or synthetic scale scenario -> shared benchmark harness -> provider or core adapter -> Criterion group -> local report and comparison artifact. Self-check target: authority count `= 1` for offline benchmark-harness behavior, path count `= 1` from scenario definition to recorded benchmark output.

## Decision Lineage

- `$search-decision --related PERFORMANCE_SUITE_AUDIT.md`, `$search-decision --related benches/support/mod.rs`, and `$search-decision --related docs/benchmarking.md` returned no matching ai-sdk-rs decision records, so there is no repo-local benchmark lineage to reopen.
- `dec_20260329_003546_94e17528` established nearby cross-repo precedent that `codex-rs` had no dedicated benchmark harness at all, which makes the current ai-sdk-rs scaffold a fresh seam rather than a missing-owner clone.
- `dec_20260329_004425_449d2180` chose a dedicated perf-bench owner instead of leaving timing evidence mixed into behavior suites; the transferable rule here is to keep benchmark ownership explicit and separate from semantic-contract tests.
- `dec_20260329_010314_9a5a8ffe` kept semantic timeout proofs out of perf-bench ownership, so this ai-sdk-rs plan should stay on harness ownership, fixture realism, and local comparison plumbing rather than reopen correctness or timeout contracts.
- Current trigger: [PERFORMANCE_SUITE_AUDIT.md](/home/mike/nodecode/ai-sdk-rs/PERFORMANCE_SUITE_AUDIT.md) rates the suite `38 / 100` and identifies one OpenAI-heavy scaffold, duplicated fixture replay support, trivial synthetic workloads, no provider-family matrix, and no local regression-comparison path.
- This plan is a fresh ai-sdk-rs benchmark-harness plan with adjacent perf-owner precedent, not a reopened Nodecode perf-budget lineage.

## Relation To Prior Lineage

- No checked-in ai-sdk-rs plan currently owns benchmark-suite maturation, fixture replay deduplication, or local regression comparison as a single seam.
- [plan-stream-part-normalization-single-path-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-stream-part-normalization-single-path-remediation.md) stays adjacent because it changes shared stream behavior, not the benchmark harness that measures it.
- [plan-openai-responses-language-model-thin-shell-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-openai-responses-language-model-thin-shell-remediation.md) also stays adjacent because provider decomposition is not the same authority as fixture replay, scenario inventory, and benchmark measurement plumbing.

## Current Shape

- The current workspace benchmark scaffold lives under `benches/` with three Criterion binaries: `openai_responses`, `streaming_pipeline`, and `core_hot_paths`.
- The shared offline harness owner now lives under `benches/support/**`, with `fixture_replay.rs` owning replay transport and fixture loading while `openai_responses.rs` owns the current OpenAI-only scenario registrations and model/config wiring.
- `benches/openai_responses.rs` and `benches/streaming_pipeline.rs` now register OpenAI scenarios through the shared support owner instead of carrying inline fixture inventories.
- `crates/providers/openai/tests/stream_fixture_tests.rs` now reuses the shared benchmark support owner instead of copying its own replay transport and fixture loader.
- `docs/benchmarking.md` documents an offline Criterion scaffold and compile-only CI, but not provider-family coverage rules, scale/adversarial scenario expectations, or a local regression-comparison workflow.
- Only OpenAI fixture files are present under `crates/providers/openai/tests/fixtures/**`, so the suite has no checked-in benchmark corpus for Anthropic, Google, Google Vertex, Amazon Bedrock, Azure, Gateway, or OpenAI-compatible provider families.

## Findings Being Addressed

- The benchmark suite is OpenAI-heavy and measures only one provider family directly.
- Shared benchmark support mixes generic harness responsibilities with OpenAI-specific model construction.
- Fixture replay support is duplicated between benches and OpenAI fixture tests.
- Core hot-path benches use trivially small synthetic loads rather than scale, adversarial, or concurrent scenario manifests.
- The documented workflow stops at compile-checking and HTML inspection, so local regression comparison has no explicit owner.

## Scope

- In scope:
- choose one checked-in owner for offline benchmark harness behavior, fixture replay, and scenario registration
- move current bench entrypoints onto thin registrations over that owner
- add benchmark-visible fixture and scenario inventory for all supported provider families
- add provider-parse, scale, adversarial, and concurrent replay scenarios through the same owner
- add a local baseline capture and comparison workflow without turning shared-runner CI into a noisy pass or fail gate
- Out of scope:
- live-network benchmarks against real providers
- allocator-specific or hardware-counter profilers such as `dhat`, `jemalloc`, `pprof`, or `cachegrind`
- semantic correctness or timeout-budget assertions that belong in unit or integration tests rather than the benchmark harness
- changes to public provider APIs except where a provider cannot be exercised through the surviving harness owner

## Constraints

- Keep authority count `= 1` for offline benchmark-harness behavior, fixture replay, and scenario registration.
- Keep path count `= 1` from benchmark scenario definition to Criterion measurement output and local comparison artifacts.
- Keep data flow one-way from fixture or synthetic scenario input to shared harness support to benchmark adapter to recorded output.
- Preserve offline and deterministic-enough local execution; do not widen this plan into network or shared-runner timing gates.
- Keep benchmark ownership separate from semantic correctness ownership.

## Architecture Direction

- Surviving owner: one dev-only benchmark harness module tree, starting from `benches/support/mod.rs` or a direct successor, that owns fixture discovery, replay transport, scenario manifests, scale generators, and measurement presets.
- Surviving path: fixture file or synthetic scenario manifest -> shared benchmark harness -> provider or core-path adapter -> Criterion benchmark group -> local report and comparison artifact referenced by docs and CI policy.

## Relevant Files

- `worklog/plans/plan-offline-benchmark-harness-single-owner-remediation.md`
- `Cargo.toml`
- `benches/support/mod.rs`
- `benches/openai_responses.rs`
- `benches/streaming_pipeline.rs`
- `benches/core_hot_paths.rs`
- `docs/benchmarking.md`
- `.github/workflows/release-check.yml`
- `crates/providers/openai/tests/stream_fixture_tests.rs`
- `crates/providers/openai/tests/fixtures/**`
- likely new benchmark fixtures under `crates/providers/*/tests/fixtures/**`
- likely new harness modules under `benches/support/**` or a direct dev-only successor

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
- `HEAD -> BH0 -> BH1 -> BH2 -> BH3 -> BH4 -> BH5`

## Execution Order

- Lock the current harness and replay assumptions first so later restructuring cannot silently change the benchmark contract.
- Extract one shared harness owner before adding more provider families so new coverage lands on the surviving path instead of reinforcing the OpenAI-only scaffold.
- Expand provider-family replay scenarios before scale and adversarial workloads so shared support is proven on real protocol variation first.
- Add local baseline-comparison workflow and docs only after the harness owner and representative scenario matrix exist.

## Atomic Slices

- [x] `BH0` Lock the current benchmark-harness seam with focused replay and inventory proofs.
  Lineage commit: `0bb6269dcadc08986886e7605080952cf2afc5fa`
  Commit subject: `test(bench): lock offline harness seam`
  Lineage parent: `HEAD`
  Scope:
  - focused dev-only tests around fixture replay behavior, benchmark registration inventory, and current compileable bench surface
  - `docs/benchmarking.md`
  - `worklog/plans/plan-offline-benchmark-harness-single-owner-remediation.md`
  Gate:
  - current replay semantics and bench inventory are pinned before harness refactor work lands
  - the suite's current provider-family gaps are explicit in checked-in evidence
  - no benchmark-harness ownership refactor lands in this slice

- [x] `BH1` Extract one shared offline harness owner and remove duplicated replay support.
  Lineage commit: `<self; backfill after landing BH1>`
  Commit subject: `refactor(bench): centralize offline harness support`
  Lineage parent: `BH0`
  Scope:
  - `benches/support/mod.rs`
  - new shared harness support modules under `benches/support/**` or a direct dev-only successor
  - `benches/openai_responses.rs`
  - `benches/streaming_pipeline.rs`
  - `benches/core_hot_paths.rs`
  - `crates/providers/openai/tests/stream_fixture_tests.rs`
  - directly affected support tests only
  Gate:
  - one checked-in owner exists for fixture loading, replay transport, runtime presets, and scenario registration primitives
  - OpenAI fixture tests stop copying replay support that now belongs to the shared harness owner
  - bench entrypoints are thinner and no longer mix generic support with provider-specific wiring

- [ ] `BH2` Add one provider-family scenario matrix on the shared harness.
  Lineage commit: `<pending>`
  Commit subject: `bench(providers): add shared harness provider matrix`
  Lineage parent: `BH1`
  Scope:
  - benchmark scenario manifests and fixtures for supported provider families
  - provider-facing bench entrypoints and directly affected provider fixture tests
  - `Cargo.toml`
  - `docs/benchmarking.md`
  - `worklog/plans/plan-offline-benchmark-harness-single-owner-remediation.md`
  Gate:
  - each supported provider family has at least one request or streaming benchmark path exercised through the shared harness owner
  - provider-specific bench logic is reduced to adapter or scenario declarations rather than custom replay infrastructure
  - the suite no longer depends on OpenAI as the only real-world fixture corpus

- [ ] `BH3` Add provider-parse, scale, and adversarial workloads through the same harness.
  Lineage commit: `<pending>`
  Commit subject: `bench(core): add parse and scale scenarios`
  Lineage parent: `BH2`
  Scope:
  - provider-parse hot-path benchmarks such as `ProviderChunk::try_from_sse` or exact provider-specific equivalents
  - large-payload, malformed-frame, and chunk-boundary stress scenarios
  - core hot-path benchmarks and directly affected harness support
  - `docs/benchmarking.md`
  - `worklog/plans/plan-offline-benchmark-harness-single-owner-remediation.md`
  Gate:
  - provider-specific parsing hot paths are benchmarked through one shared scenario system
  - scale and adversarial scenarios are represented with checked-in fixtures or deterministic generators
  - synthetic benchmark loads are no longer limited to trivially small happy-path samples

- [ ] `BH4` Add concurrent replay scenarios and local regression-comparison workflow.
  Lineage commit: `<pending>`
  Commit subject: `bench(workflow): add concurrent replay baselines`
  Lineage parent: `BH3`
  Scope:
  - concurrent and backpressure-oriented benchmark scenarios
  - local baseline capture and comparison scripts or documented commands
  - `.github/workflows/release-check.yml`
  - `docs/benchmarking.md`
  - `worklog/plans/plan-offline-benchmark-harness-single-owner-remediation.md`
  Gate:
  - the shared harness can exercise multi-stream or backpressure scenarios without bespoke benchmark codepaths
  - developers have one explicit local workflow for capturing and comparing benchmark baselines
  - CI policy stays compile-only unless a deterministic dedicated runner is introduced explicitly

- [ ] `BH5` Validate the surviving offline benchmark harness owner.
  Lineage commit: `<pending>`
  Commit subject: `test(bench): validate single harness owner`
  Lineage parent: `BH4`
  Scope:
  - targeted benchmark compile and local smoke validation commands
  - `docs/benchmarking.md`
  - `worklog/plans/plan-offline-benchmark-harness-single-owner-remediation.md`
  Gate:
  - targeted benchmark suites compile and execute through the shared harness without provider-specific replay duplication
  - one checked-in owner remains for offline benchmark-harness behavior
  - the plan and docs match the landed harness, provider-matrix, and comparison-workflow shape

## Acceptance Criteria

- One checked-in owner remains for offline benchmark-harness behavior, fixture replay, and scenario registration.
- One explicit path remains from benchmark scenario definition to Criterion measurement output and local comparison workflow.
- Provider-specific benchmark files register scenarios against the shared harness instead of reimplementing replay infrastructure.
- The suite covers all supported provider families with at least one representative offline scenario and includes scale or adversarial coverage on the same harness.
- Each checked slice maps to exactly one dedicated commit.

## Notes

- Allocation-specific profiling and hardware-counter tooling remain adjacent future seams after the shared harness and scenario matrix exist.
