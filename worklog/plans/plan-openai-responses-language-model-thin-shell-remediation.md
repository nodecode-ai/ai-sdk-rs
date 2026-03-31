# OpenAI Responses Language Model Thin-Shell Remediation

## Objective

- Keep `src/providers/openai/responses/**` as the only checked-in owner of OpenAI Responses request adaptation, provider-tool translation, and provider-local stream-hook assembly while narrowing `src/providers/openai/responses/language_model.rs` to transport/session orchestration plus helper delegation.
- Restore one explicit runtime path: `OpenAIResponsesLanguageModel` entrypoints -> request translation helper -> provider-tool helper -> provider-local stream-hook helper -> transport/session shell. Self-check target: authority count `= 1` for the OpenAI Responses adaptation seam, path count `= 1` from model entrypoint to request and stream assembly, and the current hotspot concentration in `language_model.rs` is materially below the 2026-03-31 erosion baseline.

## Decision Lineage

- `dec_20260324_220039_797b7a52` from `$search-decision --related /home/mike/nodecode/ai-sdk-rs/src/providers/openai/responses/language_model.rs` traced a real continuation bug to the OpenAI Responses request-shaping seam, confirming that `language_model.rs` already owns material provider behavior rather than incidental glue.
- `dec_20260324_220240_b01e4dd9` kept the durable fix inside the same provider-session/request-shaping seam instead of pushing policy outward, so this remediation should preserve a provider-local owner and only thin the internal structure.
- Current trigger: the 2026-03-31 `erosion analyze --json` audit for `/home/mike/nodecode/ai-sdk-rs` shows four callables in `src/providers/openai/responses/language_model.rs` still account for `24265.76` `mass.cc`, or `29.0%` of repo total complexity mass.
- This plan is a fresh thin-shell follow-up rather than a reopen of earlier parity work: previous lineages narrowed specific behaviors inside the same file, but the residual request/tool/event-hook monolith remains.

## Relation To Prior Lineage

- This is a follow-up to [plan-stream-part-normalization-single-path-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-stream-part-normalization-single-path-remediation.md), which moved generic `StreamPart` emission rules onto the shared normalizer but intentionally left provider-local OpenAI event-hook assembly in `language_model.rs`.
- This also follows [plan-codex-request-defaults-tool-choice-parallel-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-codex-request-defaults-tool-choice-parallel-remediation.md), which added one serializer-owned defaulting helper in the same file without trying to split the remaining request and tool responsibilities.
- The earlier [plan-provider-construction-single-path-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-provider-construction-single-path-remediation.md) stays adjacent but separate because public constructor surface cleanup is not the same authority as internal OpenAI Responses adaptation decomposition.

## Current Shape

- `src/providers/openai/responses/language_model.rs` is 5,650 lines, or `18.64%` of repo LOC from the current erosion snapshot.
- The same file still mixes websocket/session state, transport fallback policy, provider option parsing, prompt lowering, request-body assembly, provider-tool validation, provider-tool output reconstruction, and provider-local stream-hook configuration.
- The 2026-03-31 erosion JSON attributes the dominant residual mass in this file to `build_stream_mapper_config` (`11523.89` mass, complexity `400`), `convert_to_openai_messages` (`7271.39`, complexity `561`), `build_request_body` (`3052.31`, complexity `162`), and `provider_tool_data_from_output_item` (`2418.17`, complexity `142`).
- `src/providers/openai/responses/mod.rs` currently only exports `language_model`, so there is no checked-in helper seam yet for request translation, provider tools, or stream hooks.

## Findings Being Addressed

- One file still owns multiple non-overlapping OpenAI Responses responsibilities that change for different reasons.
- The earlier stream-normalization lineage removed generic part construction, but the provider-local event-hook forest still lives beside request serialization and tool mapping.
- Request-shaping parity work has accumulated in the same file as provider-tool validation and item/result translation, so each new provider feature or bug fix reopens a large monolith.

## Scope

- In scope:
- introduce a checked-in helper module tree under `src/providers/openai/responses/**` for request translation, provider-tool mapping, and provider-local stream-hook assembly
- narrow `src/providers/openai/responses/language_model.rs` to transport/session orchestration and thin delegation
- lock the current request-body and stream behavior with focused OpenAI Responses tests before moving code
- rerun `erosion analyze --json` and refresh checked-in plan evidence after the thin-shell cut lands
- Out of scope:
- Anthropic Messages request-builder decomposition
- changes to the shared normalizer under `src/core/event_mapper.rs`
- OpenAI wire-protocol, authentication, or transport-policy changes beyond what the decomposition requires
- public provider construction API changes already covered by `plan-provider-construction-single-path-remediation.md`

## Constraints

- Keep authority count `= 1` for the OpenAI Responses adaptation seam.
- Keep path count `= 1` from `OpenAIResponsesLanguageModel` entrypoints to request serialization and provider-local stream-hook assembly.
- Keep data flow one-way from call options and provider options into helper owners, then into request or stream payloads, then into the transport/session shell.
- Preserve current request and stream parity while reducing hotspot concentration.
- Do not widen this plan into Anthropic or other provider families; they require separate lineages.

## Architecture Direction

- Surviving owner: an explicit `src/providers/openai/responses/**` helper module tree, with `language_model.rs` reduced to the transport/session shell and direct delegate.
- Surviving path: `OpenAIResponsesLanguageModel::{do_generate, do_stream, new_turn_session}` -> request translation helper and provider-tool helper -> provider-local stream-hook helper -> existing transport/session orchestration.

## Relevant Files

- `worklog/plans/plan-openai-responses-language-model-thin-shell-remediation.md`
- `src/providers/openai/responses/mod.rs`
- `src/providers/openai/responses/language_model.rs`
- planned helper modules such as `src/providers/openai/responses/request_translation.rs`
- planned helper modules such as `src/providers/openai/responses/provider_tools.rs`
- planned helper modules such as `src/providers/openai/responses/stream_hooks.rs`
- `crates/providers/openai/tests/responses_language_model_tests.rs`
- `crates/providers/openai/tests/stream_fixture_tests.rs`

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
- `HEAD -> OR0 -> OR1 -> OR2 -> OR3 -> OR4`

## Execution Order

- Lock the request-body and stream hotspots first so later decomposition stays behaviorally pinned.
- Extract request translation before provider-tool mapping so request assembly can collapse around stable helper boundaries.
- Extract provider-tool mapping before stream hooks so tool validation and item/result reconstruction stop sharing the transport shell.
- Thin the remaining provider-local stream-hook configuration last, then rerun focused validation and erosion measurement.

## Atomic Slices

- [ ] `OR0` Lock the OpenAI Responses hotspot seam with focused request and stream tests.
  Lineage commit: `<pending>`
  Commit subject: `test(openai): lock responses thin-shell seams`
  Lineage parent: `HEAD`
  Scope:
  - `crates/providers/openai/tests/responses_language_model_tests.rs`
  - `crates/providers/openai/tests/stream_fixture_tests.rs`
  - `worklog/plans/plan-openai-responses-language-model-thin-shell-remediation.md`
  Gate:
  - focused tests pin request-body behavior currently driven by `convert_to_openai_messages` and `build_request_body`
  - focused fixture coverage pins stream output currently driven by provider-local hook assembly and tool-output reconstruction
  - no production refactor lands in this slice

- [ ] `OR1` Extract request translation into one explicit helper owner.
  Lineage commit: `<pending>`
  Commit subject: `refactor(openai): extract responses request translation`
  Lineage parent: `OR0`
  Scope:
  - `src/providers/openai/responses/mod.rs`
  - `src/providers/openai/responses/language_model.rs`
  - new request-translation helper modules under `src/providers/openai/responses/**`
  - directly affected OpenAI request-body tests only
  Gate:
  - prompt lowering, system-message mode handling, request defaults, and body assembly move behind one helper owner
  - `language_model.rs` stops owning the bulk of request-shaping branch logic directly
  - request-body parity remains unchanged

- [ ] `OR2` Extract provider-tool validation and item/result translation into one helper owner.
  Lineage commit: `<pending>`
  Commit subject: `refactor(openai): extract responses provider tool mapping`
  Lineage parent: `OR1`
  Scope:
  - `src/providers/openai/responses/mod.rs`
  - `src/providers/openai/responses/language_model.rs`
  - new provider-tool helper modules under `src/providers/openai/responses/**`
  - directly affected OpenAI request and stream tests only
  Gate:
  - provider-tool schema validation, request tool serialization, and output-item reconstruction live behind one helper owner
  - request-body and stream paths call that helper instead of carrying separate in-file branches
  - no new compatibility wrapper layer is introduced

- [ ] `OR3` Extract provider-local stream-hook assembly and leave `language_model.rs` as a transport shell.
  Lineage commit: `<pending>`
  Commit subject: `refactor(openai): thin responses language model shell`
  Lineage parent: `OR2`
  Scope:
  - `src/providers/openai/responses/mod.rs`
  - `src/providers/openai/responses/language_model.rs`
  - new stream-hook helper modules under `src/providers/openai/responses/**`
  - directly affected OpenAI stream tests only
  Gate:
  - provider-local hook assembly such as `build_stream_mapper_config` no longer dominates `language_model.rs`
  - `language_model.rs` keeps transport selection, websocket/session lifecycle, and thin delegation only
  - shared `StreamPart` normalization ownership stays with the previously landed shared normalizer

- [ ] `OR4` Validate the surviving thin-shell path and refresh erosion evidence.
  Lineage commit: `<pending>`
  Commit subject: `test(openai): validate responses thin shell`
  Lineage parent: `OR3`
  Scope:
  - targeted OpenAI Responses tests
  - fresh `erosion analyze --json` evidence
  - `worklog/plans/plan-openai-responses-language-model-thin-shell-remediation.md`
  Gate:
  - focused OpenAI Responses tests pass on the surviving owner path
  - a fresh erosion snapshot shows lower hotspot concentration than the 2026-03-31 baseline for `src/providers/openai/responses/language_model.rs`
  - the plan matches the landed ownership shape without silently widening into adjacent provider work

## Acceptance Criteria

- One checked-in OpenAI Responses helper tree remains for request adaptation, provider-tool mapping, and provider-local stream-hook assembly.
- `src/providers/openai/responses/language_model.rs` is narrowed to transport/session orchestration plus thin delegation.
- Focused OpenAI Responses tests pin both request-body and stream behavior across the refactor.
- A fresh `erosion analyze --json` run shows the March 31 hotspot concentration in `src/providers/openai/responses/language_model.rs` has moved down materially.
- Each checked slice maps to exactly one dedicated commit.

## Notes

- The next highest erosion seam after this plan is `src/providers/anthropic/messages/language_model.rs`, but it should land under its own plan-lineage file rather than being mixed into this OpenAI-owned chain.
