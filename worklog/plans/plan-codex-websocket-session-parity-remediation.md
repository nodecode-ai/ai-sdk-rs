# Codex Websocket Session Parity Remediation

## Objective

- Keep `OpenAIResponsesTurnSession::ensure_websocket_connection` as the single owner of Codex websocket connection selection inside a turn session.
- Restore one explicit runtime path for first-request Codex calls: request-specific upgrade headers decide whether a preconnected socket may be reused, and response-id chaining follows the cold-request-first websocket flow.

## Relation To Prior Lineage

- This is a review-driven follow-up to the cold-request-first Codex websocket prewarm change in the current branch.

## Current Shape

- `start_codex_websocket_preconnect` can open a headerless websocket before the first request.
- `ensure_websocket_connection` can consume that preconnected socket without checking whether the current `CallOptions.headers` must participate in the websocket upgrade.
- The copied Codex regression tests still assert the older warmup-first response-id ordering and the pre-vendoring websocket URL instrumentation surface.

## Findings Being Addressed

- `codex_websocket_http_fallback_drops_explicit_previous_response_id_after_reset` still checks `stream_urls()` for websocket traffic even though websocket connects are now recorded separately from HTTP fallback URLs.
- `codex_turn_session_updates_previous_response_id_before_stream_drain` and the related session test still expect the follow-up request to chain from `resp-1` instead of the first cold request's consumed response id.
- The first `do_stream` call can drop per-call websocket upgrade headers when it reuses a preconnected socket that was opened without those headers.

## Scope

- In scope:
- `crates/providers/openai/src/responses/language_model.rs`
- `crates/providers/openai/tests/responses_language_model_tests.rs`
- `worklog/plans/plan-codex-websocket-session-parity-remediation.md`
- Out of scope:
- non-Codex OpenAI transports
- new Codex metadata features beyond preserving current per-call websocket upgrade headers
- unrelated turn-session persistence or retry policy changes

## Constraints

- Keep authority count `= 1` for Codex websocket connection selection.
- Keep path count `= 1` for first-request header handling and previous-response-id carry-forward.
- Keep data flow one-way from `CallOptions` to websocket connect/send to session state update.
- Preserve HTTP fallback and websocket retry parity while simplifying test assertions to the transport surface that actually records the behavior.

## Architecture Direction

- Surviving owner: `OpenAIResponsesTurnSession::ensure_websocket_connection`.
- Surviving path: `CallOptions.headers` plus provider transport options -> websocket header assembly -> choose preconnected versus fresh connect -> send request body -> update session response-id state -> assert via websocket connect and request-body captures.

## Relevant Files

- `worklog/plans/plan-codex-websocket-session-parity-remediation.md`
- `crates/providers/openai/src/responses/language_model.rs`
- `crates/providers/openai/tests/responses_language_model_tests.rs`

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `git diff --cached --name-only` must remain inside that slice's declared scope.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
- `HEAD -> CW0 -> CW1 -> CW2`

## Execution Order

- Lock the Codex websocket/session evidence first.
- Move first-request connection selection onto the header-aware owner without widening the seam.
- Re-run targeted Codex regression coverage and refresh plan evidence last.

## Atomic Slices

- [x] `CW0` Lock Codex websocket session expectations.
  Lineage commit: `<fill after landing>`
  Commit subject: `test(openai): lock codex websocket session parity`
  Lineage parent: `HEAD`
  Scope:
  - `crates/providers/openai/tests/responses_language_model_tests.rs`
  - `worklog/plans/plan-codex-websocket-session-parity-remediation.md`
  Gate:
  - stale response-id assertions reflect the cold-request-first websocket sequence
  - fallback coverage checks websocket connects or request bodies instead of `stream_urls()` for websocket usage
  - a focused regression proves a first request with per-call websocket upgrade headers does not consume a headerless preconnect
  - no production changes land in this slice

- [ ] `CW1` Make Codex websocket preconnect selection header-aware.
  Lineage commit: `<pending>`
  Commit subject: `fix(openai): preserve codex websocket upgrade headers`
  Lineage parent: `CW0`
  Scope:
  - `crates/providers/openai/src/responses/language_model.rs`
  - directly affected Codex websocket tests only
  Gate:
  - first Codex requests with per-call headers perform a fresh websocket connect
  - headerless first Codex requests can still reuse a ready preconnect
  - session response-id carry-forward remains aligned with the selected connection path
  - HTTP fallback behavior remains unchanged

- [ ] `CW2` Validate the surviving Codex websocket path.
  Lineage commit: `<pending>`
  Commit subject: `test(openai): validate codex websocket session path`
  Lineage parent: `CW1`
  Scope:
  - targeted `cargo test -p ai-sdk-rs` coverage for Codex websocket session cases
  - `worklog/plans/plan-codex-websocket-session-parity-remediation.md`
  Gate:
  - the copied Codex regression tests pass on the vendored websocket transport surface
  - plan evidence matches the landed owner and runtime path
  - no unrelated provider behavior is modified in this slice

## Acceptance Criteria

- `OpenAIResponsesTurnSession::ensure_websocket_connection` remains the only selector for preconnected versus fresh Codex websocket use.
- Codex tests assert websocket behavior through websocket connect and request-body captures rather than stale HTTP stream URL assumptions.
- First-request websocket upgrade headers and response-id chaining stay consistent across cold requests, reused sockets, and HTTP fallback resets.
