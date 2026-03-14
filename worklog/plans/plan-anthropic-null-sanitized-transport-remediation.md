# Anthropic Null-Sanitized Transport Remediation

## Objective

- Keep on-wire Anthropic null sanitization owned by transport configuration until `AnthropicMessagesLanguageModel::build_request_body` no longer emits optional object nulls.
- Restore one explicit runtime path for streamed Anthropic calls: build the request body, enable transport null pruning, then send a null-free payload on the wire without conflating that contract with future builder cleanup.

## Relation To Prior Lineage

- This is a review-driven follow-up to the branch change that disabled Anthropic transport null pruning in `build_anthropic`.

## Current Shape

- `AnthropicMessagesLanguageModel::build_request_body` still serializes `cache_control: null` for ordinary text and image parts when no cache policy is supplied.
- `crates/providers/anthropic/src/provider.rs` now disables `TransportConfig::strip_null_fields`, so registry-built Anthropic clients can send those nulls on the wire.
- Existing Anthropic tests already demonstrate that the request body builder does not yet satisfy the null-free-body assumption required to remove the transport prune pass safely.

## Findings Being Addressed

- The provider build path disables null pruning before the Anthropic request builder is null-free.
- Review evidence already shows a null-bearing request body for ordinary streamed requests, so the optimization assumption is false on this branch.

## Scope

- In scope:
- `crates/providers/anthropic/src/provider.rs`
- `crates/providers/anthropic/tests/messages_tools_tests.rs`
- `worklog/plans/plan-anthropic-null-sanitized-transport-remediation.md`
- Out of scope:
- removing every `cache_control: null` emission from `build_request_body`
- broader Anthropic request-shaping refactors unrelated to null sanitization ownership
- changing Anthropic tool or beta-header behavior

## Constraints

- Keep authority count `= 1` for on-wire Anthropic null sanitization.
- Keep path count `= 1` for the builder -> transport prune -> wire send sequence.
- Keep data flow one-way from request building to transport submission.
- Preserve current Anthropic request semantics while restoring the null-free wire contract.

## Architecture Direction

- Surviving owner: `TransportConfig::strip_null_fields` as configured by `build_anthropic` for registry-built Anthropic models.
- Surviving path: `build_request_body` -> `do_stream` appends `stream: true` -> `post_json_stream(..., &self.cfg.transport_cfg)` prunes object nulls -> the transport sends a null-free payload on the wire.

## Relevant Files

- `worklog/plans/plan-anthropic-null-sanitized-transport-remediation.md`
- `crates/providers/anthropic/src/provider.rs`
- `crates/providers/anthropic/src/messages/language_model.rs`
- `crates/providers/anthropic/tests/messages_tools_tests.rs`

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `git diff --cached --name-only` must remain inside that slice's declared scope.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
- `HEAD -> AN0 -> AN1 -> AN2`

## Execution Order

- Lock the Anthropic null-sanitization seam first.
- Restore the transport-owned null-pruning path without broadening builder responsibilities.
- Re-run targeted Anthropic coverage and refresh plan evidence last.

## Atomic Slices

- [x] `AN0` Lock Anthropic builder-versus-wire null behavior.
  Lineage commit: `3ecef89`
  Commit subject: `test(anthropic): lock null sanitization seam`
  Lineage parent: `HEAD`
  Scope:
  - `crates/providers/anthropic/tests/messages_tools_tests.rs`
  - `worklog/plans/plan-anthropic-null-sanitized-transport-remediation.md`
  Gate:
  - focused coverage preserves the current evidence that the builder can still emit optional object nulls
  - focused coverage proves the registry/provider path is expected to keep object nulls off the wire
  - no production changes land in this slice

- [x] `AN1` Restore transport-owned Anthropic null pruning.
  Lineage commit: `<fill after landing>`
  Commit subject: `fix(anthropic): restore transport null pruning`
  Lineage parent: `AN0`
  Scope:
  - `crates/providers/anthropic/src/provider.rs`
  - directly affected Anthropic tests only
  Gate:
  - registry-built Anthropic models no longer disable `strip_null_fields`
  - streamed Anthropic requests remain null-free on the wire even while builder cleanup stays out of scope
  - no unrelated Anthropic request-shaping refactor lands in this slice

- [ ] `AN2` Validate the surviving Anthropic null-sanitized path.
  Lineage commit: `<pending>`
  Commit subject: `test(anthropic): validate null-sanitized transport path`
  Lineage parent: `AN1`
  Scope:
  - targeted `cargo test -p ai-sdk-rs` coverage for Anthropic request-body and transport cases
  - `worklog/plans/plan-anthropic-null-sanitized-transport-remediation.md`
  Gate:
  - targeted Anthropic regression coverage passes
  - the plan records transport as the surviving owner until a later builder-cleanup lineage exists
  - no adjacent provider behavior changes in this slice

## Acceptance Criteria

- Registry-built Anthropic models retain transport null pruning until builder null emission is removed in a separate lineage.
- Tests distinguish builder null emission evidence from the on-wire null-free transport contract.
- Any future removal of Anthropic transport null pruning must follow a separate plan after builder-level null emission is eliminated.
