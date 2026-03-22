# Codex Request Defaults Tool Choice And Parallel Calls Remediation

## Objective

- Keep `src/providers/openai/responses/language_model.rs` as the single owner that materializes Codex request-body defaults for `tool_choice` and `parallel_tool_calls`.
- Restore one explicit runtime path for these fields on the ChatGPT Codex OAuth websocket seam: Codex endpoint detection plus parsed provider options flow into one defaulting helper before JSON request serialization.

## Relation To Prior Lineage

- This is a follow-up to [plan-codex-websocket-session-parity-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-codex-websocket-session-parity-remediation.md), which restored websocket/session-path parity but did not close the remaining request-default gap.
- This plan also narrows one request-body finding adjacent to the broader matrix in [openai-responses-parity-gap-matrix.md](/home/mike/nodecode/ai-sdk-rs/docs/plans/openai-responses-parity-gap-matrix.md).

## Current Shape

- `src/providers/openai/responses/language_model.rs` already knows how to serialize `tool_choice` when `CallOptions.tool_choice` is set and `parallel_tool_calls` when provider options include `parallelToolCalls`.
- The same module already owns Codex endpoint detection through `should_use_codex_oauth_websocket_transport(...)`, so the request-body seam is already centralized there.
- There is no Codex-owned defaulting path for these fields today, so callers must either inject them ad hoc or omit them.
- Live capture from 2026-03-22 showed the delta clearly:
- Codex artifact `/tmp/mitm-codex-nodecode/codex-exec-search-1.flow` sent `tool_choice: "auto"` and `parallel_tool_calls: true`.
- Nodecode artifact `/tmp/mitm-codex-nodecode/nodecode-1.flow` omitted both fields while still hitting the same `/backend-api/codex/responses` websocket endpoint.

## Findings Being Addressed

- Request-body parity is still incomplete even when websocket transport parity is already present.
- The serializer currently has the information needed to decide the Codex path, but not one canonical helper that turns that decision into request defaults.
- Leaving this as caller-owned policy would create multiple authorities across Nodecode, ai-sdk-rs callers, and future Codex integrations.

## Scope

- In scope:
- `src/providers/openai/responses/language_model.rs`
- `crates/providers/openai/tests/responses_language_model_tests.rs`
- `worklog/plans/plan-codex-request-defaults-tool-choice-parallel-remediation.md`
- Out of scope:
- handshake-header parity such as `originator`, `session_id`, `version`, or `x-client-request-id`
- native tool inventory or `web_search` enablement policy
- adjacent request-body defaults such as `service_tier`, `text.verbosity`, or richer `client_metadata`
- non-Codex OpenAI Responses paths and non-OpenAI providers

## Constraints

- Keep authority count `= 1` for Codex request-default materialization.
- Keep path count `= 1` for deciding and emitting `tool_choice` and `parallel_tool_calls`.
- Keep data flow one-way from endpoint path plus parsed options to one defaulting helper to serialized request body.
- Preserve explicit caller overrides: a caller-provided `tool_choice` or `parallelToolCalls` value must still win over Codex defaults.
- Preserve non-Codex behavior exactly.

## Architecture Direction

- Surviving owner: one Codex request-default helper inside `src/providers/openai/responses/language_model.rs`.
- Surviving path: `CallOptions` plus parsed OpenAI provider options plus endpoint-path classification -> Codex request-default helper -> request JSON body assembly.

## Relevant Files

- [plan-codex-request-defaults-tool-choice-parallel-remediation.md](/home/mike/nodecode/ai-sdk-rs/worklog/plans/plan-codex-request-defaults-tool-choice-parallel-remediation.md)
- [language_model.rs](/home/mike/nodecode/ai-sdk-rs/src/providers/openai/responses/language_model.rs)
- [responses_language_model_tests.rs](/home/mike/nodecode/ai-sdk-rs/crates/providers/openai/tests/responses_language_model_tests.rs)
- [openai-responses-parity-gap-matrix.md](/home/mike/nodecode/ai-sdk-rs/docs/plans/openai-responses-parity-gap-matrix.md)

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
- `HEAD -> RD0 -> RD1 -> RD2`

## Execution Order

- Lock the exact request-default seam first.
- Introduce the Codex-owned defaulting helper second.
- Re-run focused parity validation and refresh plan evidence last.

## Atomic Slices

- [x] `RD0` Lock Codex request-default parity with focused tests.
  Lineage commit: `530fa84`
  Commit subject: `test(openai): lock codex request defaults`
  Lineage parent: `HEAD`
  Scope:
  - `crates/providers/openai/tests/responses_language_model_tests.rs`
  - `worklog/plans/plan-codex-request-defaults-tool-choice-parallel-remediation.md`
  Gate:
  - a focused Codex websocket request-body test proves `tool_choice` and `parallel_tool_calls` are currently absent when callers omit them
  - a companion test proves explicit caller-provided values still serialize as requested
  - no production serializer change lands in this slice

- [x] `RD1` Move Codex request defaults onto one serializer-owned helper.
  Lineage commit: `12fcaae`
  Commit subject: `fix(openai): default codex request tool fields`
  Lineage parent: `RD0`
  Scope:
  - `src/providers/openai/responses/language_model.rs`
  - directly affected Codex request-body tests only
  Gate:
  - Codex OAuth websocket requests default `tool_choice` to `auto` when omitted
  - Codex OAuth websocket requests default `parallel_tool_calls` to `true` when omitted
  - explicit caller values still win
  - non-Codex OpenAI Responses paths remain unchanged

- [x] `RD2` Validate the surviving Codex request-default path and refresh evidence.
  Lineage commit: `<self; backfill after landing RD2>`
  Commit subject: `test(openai): validate codex request defaults`
  Lineage parent: `RD1`
  Scope:
  - targeted validation for the new request-body tests
  - plan evidence updates only
  Gate:
  - focused OpenAI Responses tests pass on the surviving owner path
  - plan evidence matches the landed ownership shape
  - no adjacent handshake, tool-inventory, or service-tier changes ride along

## Validation Checklist

- [x] Add a Codex-endpoint request-body test that asserts omitted `tool_choice` and omitted `parallelToolCalls` become `tool_choice: "auto"` and `parallel_tool_calls: true`.
- [x] Add a Codex-endpoint request-body test that proves explicit caller values still override the Codex defaults.
- [x] Re-run the targeted OpenAI Responses Rust test lane covering the new assertions.

## Acceptance Criteria

- One checked-in owner remains for Codex request-body default materialization.
- One explicit runtime path remains for deciding and emitting `tool_choice` and `parallel_tool_calls`.
- Codex OAuth websocket requests default those two fields without caller duplication.
- Non-Codex request behavior remains unchanged.

## Notes

- The live finding this plan addresses is request-body parity, not handshake overhead. The capture difference in headers was tiny relative to the websocket frame size, so that seam should stay separate.
