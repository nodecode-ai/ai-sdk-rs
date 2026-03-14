# Stream Part Normalization Single-Path Remediation

## Objective

- Keep shared stream normalization core as the only owner of `StreamPart` emission rules across providers; provider modules may decode raw transport chunks, but they must hand off to one shared normalization path.
- Restore one explicit streaming path: provider raw stream -> provider event or chunk adapter -> shared stream normalizer -> `StreamPart` stream. Self-check target: authority count `= 1` for `StreamPart` emission logic, path count `= 1` from provider chunks to normalized stream parts.

## Current Shape

- `src/core/event_mapper.rs` already owns one shared event-to-`StreamPart` mapping path.
- Anthropic stays relatively thin by using that shared mapper with a small provider hook.
- OpenAI Responses keeps a large provider-specific adapter with its own request-body and stream-mapping logic.
- OpenAI-compatible and Gateway each own their own SSE-to-`StreamPart` state machines, and Google keeps another provider-specific stream core.

## Findings Being Addressed

- Stream normalization is duplicated instead of converged.
- Multiple providers independently own text, reasoning, tool, and finish state transitions that should follow one normalized contract.
- The largest provider adapter files are carrying both provider protocol handling and generic `StreamPart` emission responsibilities.

## Scope

- In scope:
- extract or expand one shared normalization core for provider stream state and `StreamPart` emission
- move provider-local SSE and chunk handlers onto that shared owner
- narrow provider-local code down to transport decode and provider-event adaptation
- lock parity with focused cross-provider streaming tests and fixtures
- Out of scope:
- changing provider wire protocols or authentication flows
- broad request-body builder rewrites except where they are required to feed the shared normalizer cleanly
- public `StreamPart` schema changes

## Constraints

- Keep authority count `= 1` for normalized `StreamPart` emission rules.
- Keep path count `= 1` from provider raw stream input to caller-visible `StreamPart` output.
- Keep data flow one-way from provider decode to shared normalization to consumers.
- Preserve provider streaming parity while simplifying internal ownership.

## Architecture Direction

- Surviving owner: shared normalization core in `src/core/event_mapper.rs` or a direct successor extracted from it.
- Surviving path: provider-specific decode -> provider event or metadata adapter -> shared stream state machine -> `StreamPart`.

## Relevant Files

- `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
- `src/core/event_mapper.rs`
- `crates/providers/openai/src/responses/language_model.rs`
- `crates/providers/openai-compatible/src/stream.rs`
- `crates/providers/gateway/src/language_model.rs`
- `crates/providers/google/src/shared/stream_core.rs`
- `crates/providers/anthropic/src/messages/language_model.rs`
- focused stream parity tests under `crates/providers/**/tests/**`

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `git diff --cached --name-only` must remain inside that slice's declared scope.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
- `HEAD -> SN0 -> SN1 -> SN2 -> SN3 -> SN3A -> SN3B -> SN4`

## Execution Order

- Lock representative cross-provider stream parity first.
- Extract the shared normalization owner before moving provider implementations.
- Migrate the simpler duplicated providers first, then thin the largest provider adapters onto the same owner.
- Re-run focused parity coverage and refresh plan evidence after the surviving path is singular.

## Atomic Slices

- [x] `SN0` Lock representative cross-provider stream parity.
  Lineage commit: `e74b41e`
  Commit subject: `test(stream): lock normalized stream parity seams`
  Lineage parent: `HEAD`
  Scope:
  - focused fixtures and parity tests for OpenAI Responses, OpenAI-compatible, Gateway, and one provider already using the shared mapper
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  Gate:
  - current `StreamPart` behavior is pinned before refactor work lands
  - tests cover text, reasoning, tool, raw, and finish transitions on representative providers
  - no production normalization refactor lands in this slice

- [x] `SN1` Extract the shared stream normalization owner.
  Lineage commit: `d86ae2a`
  Commit subject: `refactor(stream): extract shared part normalization core`
  Lineage parent: `SN0`
  Scope:
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  - `src/core/event_mapper.rs`
  - any new shared normalization helper module under `src/core/**`
  - directly affected tests only
  Gate:
  - one shared owner exists for generic stream state and `StreamPart` emission rules
  - provider-specific code no longer needs to define its own generic text or tool state machine primitives
  - no provider behavior changes land beyond what parity tests require

- [x] `SN2` Move the duplicated provider-local stream engines onto the shared owner.
  Lineage commit: `bb6eaf3`
  Commit subject: `refactor(stream): migrate duplicated provider stream engines`
  Lineage parent: `SN1`
  Scope:
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  - `crates/providers/openai-compatible/src/stream.rs`
  - `crates/providers/gateway/src/language_model.rs`
  - `crates/providers/google/src/shared/stream_core.rs`
  - directly affected stream tests only
  Gate:
  - duplicated provider-local SSE-to-`StreamPart` state machines are removed or reduced to provider-event adaptation
  - the shared owner now emits generic text, reasoning, tool, and finish transitions for these providers
  - provider parity remains intact

- [x] `SN3` Thin OpenAI Responses onto the same shared normalization path.
  Lineage commit: `e3fdc8a`
  Commit subject: `refactor(stream): thin openai responses normalization`
  Lineage parent: `SN2`
  Scope:
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  - `crates/providers/openai/src/responses/language_model.rs`
  - directly affected OpenAI stream parity tests only
  Gate:
  - OpenAI Responses no longer owns generic `StreamPart` emission rules that now belong to the shared normalizer
  - provider-local code is reduced to request shaping, transport selection, and provider event adaptation
  - the largest stream adapter file is materially narrowed without changing normalized output

- [x] `SN3A` Route remaining shared lifecycle scaffolding onto the shared owner.
  Lineage commit: `2bf0241`
  Commit subject: `refactor(stream): route remaining shared lifecycle scaffolding`
  Lineage parent: `SN3`
  Scope:
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  - `src/core/event_mapper.rs`
  - `crates/providers/openai-compatible/src/stream.rs`
  - `crates/providers/gateway/src/language_model.rs`
  - `crates/providers/openai/src/responses/language_model.rs`
  - directly affected stream tests only
  Gate:
  - shared text-open and generic tool lifecycle helpers own more of the remaining provider runtime scaffolding
  - provider-local code is further reduced without changing representative stream parity
  - the final validation slice can check the resulting ownership shape honestly

- [x] `SN3B` Centralize the remaining generic part constructors behind the shared owner.
  Lineage commit: `eaaf8f9`
  Commit subject: `refactor(stream): centralize remaining generic part constructors`
  Lineage parent: `SN3A`
  Scope:
  - `src/core/event_mapper.rs`
  - `crates/providers/openai-compatible/src/stream.rs`
  - `crates/providers/gateway/src/language_model.rs`
  - `crates/providers/openai/src/responses/language_model.rs`
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  - directly affected stream tests only
  Gate:
  - provider runtime files no longer directly instantiate generic text, reasoning, tool-input, tool-call, or finish parts
  - shared helper methods in the surviving owner define the remaining generic part-construction rules
  - representative provider parity remains intact

- [x] `SN4` Validate the surviving single-path stream normalizer.
  Lineage commit: `<self; backfill after landing SN4>`
  Commit subject: `test(stream): validate single normalization path`
  Lineage parent: `SN3B`
  Scope:
  - targeted `cargo test` coverage for representative provider streaming suites
  - `worklog/plans/plan-stream-part-normalization-single-path-remediation.md`
  Gate:
  - targeted cross-provider stream parity coverage passes
  - one shared normalization owner remains for `StreamPart` emission logic
  - the plan matches the landed ownership shape

## Acceptance Criteria

- One checked-in normalization owner remains for `StreamPart` emission rules.
- One explicit path remains from provider raw chunks to normalized `StreamPart` output.
- Provider modules keep only protocol-specific adaptation and transport responsibilities.
- Each checked slice maps to exactly one dedicated commit.
