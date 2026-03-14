# Single Module Tree Remediation

## Objective

- Keep the library compiled from one real `src/` module tree rooted at `src/lib.rs`; remove `#[path = "../crates/..."]` pseudo-crate mounting as an internal ownership pattern.
- Restore one explicit compilation path: `Cargo.toml` -> normal Rust modules -> public re-exports. Self-check target: authority count `= 1` for library compilation ownership, path count `= 1` for internal module resolution.

## Current Shape

- `src/lib.rs` path-mounts foundational modules and provider modules from `crates/*`.
- Internal aliases such as `ai_sdk_core`, `ai_sdk_provider`, and provider-specific pseudo-crate names keep the old topology alive even though compilation is single-crate.
- The workspace manifest lists only the root package and three example members, so the current layout is neither a real multi-crate workspace nor a normal flat module tree.

## Findings Being Addressed

- Phantom crate topology adds navigation and coupling cost without real Cargo-enforced isolation.
- Pseudo-crate aliases allow broad reach-through imports that would be constrained by either a real workspace or a normal module tree.
- Tooling and example packaging have already drifted because the repository shape advertises crate boundaries that Cargo does not actually enforce.

## Scope

- In scope:
- move library code from `crates/*` into a normal `src/` module tree or equivalent direct submodules
- delete `#[path = ...]` mounting and pseudo-crate alias plumbing from `src/lib.rs`
- preserve the existing public crate API while simplifying internal ownership
- add focused compile and topology validation that keeps the refactor honest
- Out of scope:
- splitting the repository into separately published Cargo packages
- provider runtime behavior changes beyond import and module ownership updates
- public API redesign beyond what is required to keep current exports stable

## Constraints

- Keep authority count `= 1` for internal library compilation ownership.
- Keep path count `= 1` for internal module resolution from `src/lib.rs` to implementation modules.
- Keep data flow one-way from public exports to internal modules without pseudo-crate backchannels.
- Preserve public API parity while simplifying internal structure.

## Architecture Direction

- Surviving owner: the real module tree under `src/`.
- Surviving path: `Cargo.toml` -> `src/lib.rs` -> direct `mod` declarations -> implementation modules -> public re-exports.

## Relevant Files

- `worklog/plans/plan-single-module-tree-remediation.md`
- `src/lib.rs`
- `crates/core/**`
- `crates/provider/**`
- `crates/sdk-types/**`
- `crates/streaming-sse/**`
- `crates/transports/reqwest/**`
- `crates/providers/**`
- `tests/**`

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `git diff --cached --name-only` must remain inside that slice's declared scope.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
- `HEAD -> MT0 -> MT1 -> MT2 -> MT3`

## Execution Order

- Lock the exported module surface and topology evidence first.
- Move foundational modules onto the real `src/` tree before touching provider implementations.
- Delete the remaining pseudo-crate path mounts and aliases only after both layers compile through direct modules.
- Re-run topology and compile validation and refresh plan evidence last.

## Atomic Slices

- [ ] `MT0` Lock public surface and topology evidence.
  Lineage commit: `<pending>`
  Commit subject: `test(layout): lock single module tree seam`
  Lineage parent: `HEAD`
  Scope:
  - focused compile tests for the current public crate surface
  - a topology assertion or test that makes the current `#[path]` seam explicit
  - `worklog/plans/plan-single-module-tree-remediation.md`
  Gate:
  - public imports stay pinned before structural refactor work lands
  - the pseudo-crate topology is described by executable or checked-in evidence
  - no production module moves land in this slice

- [ ] `MT1` Move foundational modules onto the real `src/` tree.
  Lineage commit: `<pending>`
  Commit subject: `refactor(layout): move foundational modules into src`
  Lineage parent: `MT0`
  Scope:
  - `src/lib.rs`
  - foundational modules currently under `crates/core/**`, `crates/provider/**`, `crates/sdk-types/**`, `crates/streaming-sse/**`, and `crates/transports/reqwest/**`
  - directly affected tests only
  Gate:
  - foundational modules are resolved through direct `src/` module paths
  - public exports remain unchanged
  - no provider implementation behavior changes beyond import rewiring land in this slice

- [ ] `MT2` Move provider implementations onto the same module tree and delete pseudo-crate aliases.
  Lineage commit: `<pending>`
  Commit subject: `refactor(layout): remove pseudo crate provider aliases`
  Lineage parent: `MT1`
  Scope:
  - provider modules currently under `crates/providers/**`
  - `src/lib.rs`
  - directly affected provider tests only
  Gate:
  - provider code imports through real modules only
  - pseudo-crate alias plumbing is removed rather than left as compatibility scaffolding
  - the internal module tree has one ownership model end to end

- [ ] `MT3` Validate the surviving single module tree.
  Lineage commit: `<pending>`
  Commit subject: `test(layout): validate single module tree`
  Lineage parent: `MT2`
  Scope:
  - targeted `cargo check` and `cargo test` coverage for the library and active examples
  - topology validation that confirms `#[path = "../crates/..."]` is gone from `src/lib.rs`
  - `worklog/plans/plan-single-module-tree-remediation.md`
  Gate:
  - targeted validation passes through the surviving module tree
  - no pseudo-crate path mounting remains in the library root
  - the plan matches the landed ownership shape

## Acceptance Criteria

- One checked-in module tree under `src/` owns library compilation.
- One explicit internal resolution path remains from `src/lib.rs` to implementation modules.
- Public crate API parity is preserved without pseudo-crate alias plumbing.
- Each checked slice maps to exactly one dedicated commit.
