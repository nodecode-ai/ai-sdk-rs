# Hyper HTTP Transport Single-Owner Remediation

## Objective

- Replace the default HTTP transport owner with a `hyper`-backed implementation while keeping websocket behavior and transport trait semantics intact.
- Restore one explicit runtime path for HTTP transport work: provider/model code -> `HttpTransport` trait -> `HyperTransport` -> shared response/error/event helpers, with the self-check target that runtime HTTP authority count `= 1` and path count `= 1`.

## Decision Lineage

- Earliest upstream decision: `dec_20260401_034234_fb402fc3` established the current compile-cost reduction principle of keeping a small shared core and re-adding heavy runtime features only at true owners.
- Adjacent follow-up decisions: `dec_20260401_035329_adec5d7c` slimmed `reqwest` feature fanout, and `dec_20260401_040221_fc5055c0` localized HTTP/websocket feature ownership to the crates that truly need it.
- Current triggering node: the 2026-03-31 ai-sdk-rs dependency-pruning session identified `reqwest` as the largest remaining runtime subtree, confirmed that a custom `hyper` stack is materially smaller, and selected a staged transport-owner migration that keeps websocket protocol behavior unchanged.
- This plan is a fresh ai-sdk-rs follow-up to the broader HTTP feature-fanout lineage. `src/transport_reqwest.rs` and the existing ai-sdk-rs plans had no direct local decision-memory match, so this plan carries the broader transport-minimization rationale into a repo-specific owner cutover.

## Relation To Prior Lineage

- This plan is adjacent to `worklog/plans/plan-openai-responses-language-model-thin-shell-remediation.md` because OpenAI responses defaults currently flow through `crate::reqwest_transport::ReqwestTransport`, but transport ownership remains a separate seam from the language-model thin-shell cleanup.
- This plan also follows the same simplification direction as the Nodecode HTTP feature-fanout decisions without reopening those workspace manifests directly.

## Current Shape

- `src/transport_reqwest.rs` currently owns too many responsibilities at once: reqwest client construction, JSON and multipart HTTP execution, byte downloads, websocket connection/session handling, proxy tunnel logic, SSE framing helpers, response/error mapping, and transport-event emission.
- `src/lib.rs` and multiple provider/model defaults still hard-code `crate::reqwest_transport::ReqwestTransport`, so the runtime HTTP owner is pinned to reqwest even though websocket support is already separate through `tokio-tungstenite`.
- The current dependency audit shows `reqwest` as the largest remaining runtime subtree in this crate, materially larger than the plausible `hyper` HTTP stack.

## Findings Being Addressed

- The HTTP owner and websocket/common helpers are conflated inside one transport file, making transport replacement larger than it needs to be.
- The default alias surface in `src/lib.rs` and provider/model generic defaults ties the entire crate to reqwest instead of an HTTP-owner seam.
- The current runtime graph keeps a large high-level HTTP client subtree even though the crate's actual transport contract is already narrow and explicit in `src/core/transport.rs`.

## Scope

- In scope:
- split reusable HTTP response/error/event helpers out of `src/transport_reqwest.rs` into `src/transport_http_common.rs`
- split websocket/common helpers out of `src/transport_reqwest.rs` into a dedicated websocket helper module while preserving behavior
- add `src/transport_hyper.rs` side-by-side and make it satisfy the existing `HttpTransport` contract for JSON stream, JSON request, multipart request, and byte download paths
- switch default aliases and provider/model default generic parameters from reqwest to hyper only after parity tests pass
- remove reqwest as the default HTTP owner and delete the reqwest dependency last
- Out of scope:
- changing websocket protocol behavior or replacing `tokio-tungstenite`
- changing `HttpTransport`, `JsonStreamWebsocketConnection`, `TransportError`, or `TransportEvent` semantics beyond what is required for transport-owner parity
- dropping SOCKS or proxy behavior without an explicit parity decision and proof
- provider request/response contract changes unrelated to transport ownership

## Constraints

- Keep authority count `= 1` for runtime HTTP transport execution.
- Keep path count `= 1` for `post_json_stream`, `post_json`, `post_multipart`, and `get_bytes`.
- Keep websocket ownership and behavior stable while the HTTP owner changes underneath it.
- Preserve null-pruning, retry-after parsing, idle timeout behavior, multipart semantics, and transport-event emission parity.
- Keep Rust `1.78` compatibility and avoid introducing a larger replacement subtree than the one being removed.

## Architecture Direction

- Surviving owner: `src/transport_hyper.rs::HyperTransport` becomes the canonical runtime owner for HTTP request execution.
- Non-owning support code: `src/transport_http_common.rs` and the extracted websocket/common helper module hold shared framing, response parsing, retry-after, event emission, and proxy helper logic that should not belong to one concrete HTTP client.
- Surviving path: provider/model code -> `HttpTransport` trait -> `HyperTransport` request execution -> shared response/error/event helpers -> returned JSON/bytes/stream, with websocket entrypoints continuing to use the extracted websocket helper path.

## Relevant Files

- `worklog/plans/plan-hyper-http-transport-single-owner-remediation.md`
- `src/core/transport.rs`
- `src/lib.rs`
- `src/transport_reqwest.rs`
- `src/transport_http_common.rs`
- `src/transport_hyper.rs`
- `src/providers/openai/responses/language_model.rs`
- provider and model files that currently default generic transport parameters to `crate::reqwest_transport::ReqwestTransport`

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
- `HEAD -> HT0 -> HT1 -> HT2 -> HT3 -> HT4`

## Execution Order

- Lock the current transport seam with focused parity tests first.
- Extract non-owning HTTP and websocket/common helpers while reqwest still owns runtime HTTP behavior.
- Add the hyper transport side-by-side and prove trait-level parity before changing defaults.
- Switch public/default aliases and provider/model defaults only after the side-by-side proof is green.
- Remove reqwest and refresh validation/evidence last.

## Atomic Slices

- [x] `HT0` Lock the transport parity seam with focused tests.
  Lineage commit: `67df66cb868c7021f5bf94291c60d3973babf374`
  Commit subject: `test(transport): lock http parity seam`
  Lineage parent: `HEAD`
  Scope:
  - focused transport tests for JSON stream, JSON request, multipart, byte download, and websocket/common helper invariants
  - this plan file only
  Gate:
  - tests pin the pre-refactor transport contract explicitly
  - no production transport-owner change lands in this slice

- [x] `HT1` Extract shared HTTP and websocket/common helpers without changing the owner.
  Lineage commit: `b070ad12035430bceb57c9052529c91eb5260ffb`
  Commit subject: `refactor(transport): extract shared helpers`
  Lineage parent: `HT0`
  Scope:
  - `src/transport_reqwest.rs`
  - new helper modules such as `src/transport_http_common.rs` and the websocket/common helper module
  - directly affected tests only
  Gate:
  - reqwest remains the sole HTTP runtime owner after this slice
  - helper extraction does not change transport behavior
  - websocket/common helpers are no longer entangled with reqwest-only HTTP code

- [x] `HT2` Add the hyper transport side-by-side under the existing trait contract.
  Lineage commit: `<self; backfill after landing HT2>`
  Commit subject: `feat(transport): add hyper http owner`
  Lineage parent: `HT1`
  Scope:
  - `src/transport_hyper.rs`
  - shared helper modules reused by the new implementation
  - directly affected transport tests only
  Gate:
  - `HyperTransport` implements `post_json_stream`, `post_json`, `post_multipart`, and `get_bytes`
  - hyper-backed behavior matches the locked trait-level parity seam
  - default aliases still point at reqwest in this slice

- [ ] `HT3` Switch default aliases and generic defaults to hyper after parity proof.
  Lineage commit: `<pending>`
  Commit subject: `refactor(transport): switch default http owner to hyper`
  Lineage parent: `HT2`
  Scope:
  - `src/lib.rs`
  - provider/model files that default to `crate::reqwest_transport::ReqwestTransport`
  - directly affected tests only
  Gate:
  - the public/default HTTP transport owner is now hyper
  - reqwest remains side-by-side only as a temporary fallback owner in this slice
  - all provider/model default generic surfaces compile and test against the new default owner

- [ ] `HT4` Remove reqwest last and refresh final validation evidence.
  Lineage commit: `<pending>`
  Commit subject: `refactor(transport): remove reqwest owner`
  Lineage parent: `HT3`
  Scope:
  - `Cargo.toml`
  - `Cargo.lock`
  - reqwest-specific transport files and residual imports
  - final transport validation updates
  - this plan file
  Gate:
  - reqwest is removed from the runtime graph
  - targeted and full transport validation pass on the surviving hyper owner
  - the plan records hyper as the single surviving HTTP transport owner and reqwest as deleted lineage

## Validation Checklist

- `cargo test --all-features`
- `cargo check --all-features --all-targets`
- focused transport tests added in `HT0`
- `cargo tree -p ai-sdk-rs --depth 1 --charset ascii`
- final slice only: `cargo tree -i reqwest --charset ascii` should show no remaining runtime owner path

## Acceptance Criteria

- One checked-in runtime HTTP owner remains: `HyperTransport`.
- One explicit runtime HTTP path remains under the existing `HttpTransport` trait.
- Websocket behavior remains unchanged while HTTP ownership moves off reqwest.
- Reqwest is removed only after the hyper owner is default and fully validated.

## Notes

- The current repo already separates websocket dependencies from HTTP-client choice, so the transport cutover should optimize for a smaller HTTP owner rather than attempting a simultaneous websocket rewrite.
