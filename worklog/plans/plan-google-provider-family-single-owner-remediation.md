# Google Provider Family Single-Owner Remediation

## Objective

- Keep one shared Google provider-family helper module set as the owner of prompt, options, error, tool-prep, and related common logic for both Google and Google Vertex; remove thin forwarding and re-export seams.
- Restore one explicit helper path: provider-specific config -> shared Google family helper modules -> provider language model. Self-check target: authority count `= 1` for shared Google family helper logic, path count `= 1` from both providers to that shared owner.

## Current Shape

- Google Vertex re-exports Google shared modules through a local `shared.rs` seam.
- Google Vertex also keeps thin wrapper modules for error, prompt, options, and tool preparation even when the underlying behavior is shared.
- The split creates two namespaces for one family of helper behavior, which obscures the real owner and makes imports look more provider-specific than they are.

## Findings Being Addressed

- The Google and Google Vertex stacks use forwarding seams that are mostly organizational rather than behavioral.
- Thin wrappers and re-export modules create extra places to touch when shared behavior changes.
- The current structure makes the shared owner harder to discover and reason about than the actual runtime behavior warrants.

## Scope

- In scope:
- choose one checked-in owner for shared Google family helper logic
- move both Google and Google Vertex provider modules to import that owner directly
- delete forwarding wrappers and re-export seams that no longer add behavior
- add focused parity coverage for the shared helper path
- Out of scope:
- non-Google provider families
- public provider construction API changes covered by `plan-provider-construction-single-path-remediation.md`
- cross-provider stream normalization work covered by `plan-stream-part-normalization-single-path-remediation.md`

## Constraints

- Keep authority count `= 1` for shared Google family helper behavior.
- Keep path count `= 1` from Google and Google Vertex language models to shared helper logic.
- Keep data flow one-way from provider-specific config to shared helper modules to request and stream assembly.
- Preserve Google and Google Vertex parity while simplifying internal ownership.

## Architecture Direction

- Surviving owner: one common Google family helper module tree, likely under `crates/providers/google/shared/**` or a direct successor with the same single-owner role.
- Surviving path: Google or Google Vertex provider-specific configuration -> shared helper modules -> provider language model -> transport.

## Relevant Files

- `worklog/plans/plan-google-provider-family-single-owner-remediation.md`
- `crates/providers/google/src/shared/**`
- `crates/providers/google/src/error.rs`
- `crates/providers/google/src/prepare_tools.rs`
- `crates/providers/google/src/gen_ai/options.rs`
- `crates/providers/google/src/gen_ai/prompt.rs`
- `crates/providers/google-vertex/src/shared.rs`
- `crates/providers/google-vertex/src/error.rs`
- `crates/providers/google-vertex/src/prepare_tools.rs`
- `crates/providers/google-vertex/src/options.rs`
- `crates/providers/google-vertex/src/prompt.rs`
- `crates/providers/google-vertex/src/language_model.rs`
- focused Google and Vertex parity tests under `crates/providers/google*/tests/**`

## Commit Enforcement Rules

- Every executable slice below starts as `- [ ]`.
- A slice may be checked only after its work lands in exactly one dedicated git commit.
- Record the landed commit directly under the checked slice as `Lineage commit: <sha>`.
- Before each commit, `git diff --cached --name-only` must remain inside that slice's declared scope.
- If a slice grows beyond one coherent commit, split it into smaller unchecked slices before checking anything off.
- Use `$git-diff-commit` after staging only the files for that slice.

## Lineage Chain

- Planned linear commit lineage:
- `HEAD -> GV0 -> GV1 -> GV2 -> GV3`
- Actual landed git order:
- `1c7a41d (GV0 landed, then a concurrent master rewrite displaced it from ancestry) -> 7f9b581 (concurrent unrelated commit) -> a93a3b3 (concurrent unrelated commit) -> 5e8a0b6 -> e74b41e (concurrent unrelated commit) -> 00385e9 -> a22a457`

## Execution Order

- Lock current Google and Vertex helper parity first.
- Move shared helper ownership into one explicit owner before deleting wrapper seams.
- Delete forwarding modules only after both providers import the surviving owner directly.
- Re-run focused parity validation and refresh plan evidence last.

## Atomic Slices

- [x] `GV0` Lock Google and Vertex shared-helper parity.
  Lineage commit: `1c7a41d`
  Commit subject: `test(google): lock shared helper ownership seams`
  Lineage parent: `HEAD`
  Scope:
  - focused tests or compile coverage for Google and Vertex prompt, options, error, and tool-prep behavior
  - `worklog/plans/plan-google-provider-family-single-owner-remediation.md`
  Gate:
  - current shared-helper behavior is pinned before module ownership changes land
  - tests distinguish real shared behavior from wrapper-only structure
  - no production helper refactor lands in this slice

- [x] `GV1` Move shared helper ownership into one common module tree.
  Lineage commit: `5e8a0b6`
  Commit subject: `refactor(google): centralize shared helper ownership`
  Lineage parent: `GV0`
  Scope:
  - shared Google helper modules
  - Google and Google Vertex language model imports
  - directly affected tests only
  - `worklog/plans/plan-google-provider-family-single-owner-remediation.md`
  Gate:
  - one checked-in helper owner exists for prompt, options, error, and tool preparation logic
  - both providers import the shared owner directly
  - no extra compatibility wrapper layer is introduced

- [x] `GV2` Delete forwarding wrappers and re-export seams.
  Lineage commit: `00385e9`
  Commit subject: `refactor(google): remove forwarding wrapper seams`
  Lineage parent: `GV1`
  Scope:
  - `crates/providers/google-vertex/src/shared.rs`
  - thin wrapper modules under `crates/providers/google-vertex/src/**`
  - any directly affected Google wrapper modules
  - `worklog/plans/plan-google-provider-family-single-owner-remediation.md`
  Gate:
  - forwarding and re-export modules that no longer add behavior are removed
  - there is one obvious owner to modify for shared Google-family behavior
  - imports remain direct and one-way

- [x] `GV3` Validate the surviving Google family helper owner.
  Lineage commit: `a22a457`
  Commit subject: `test(google): validate single helper owner`
  Lineage parent: `GV2`
  Scope:
  - targeted Google and Google Vertex parity tests
  - `worklog/plans/plan-google-provider-family-single-owner-remediation.md`
  Gate:
  - clean detached-worktree `cargo test parity_regression_tests` passes for the Google provider family after wrapper removal
  - one shared helper owner remains for the Google provider family
  - the plan matches the landed ownership shape

## Acceptance Criteria

- One checked-in owner remains for shared Google-family helper behavior.
- One explicit import path remains from both providers to that shared owner.
- Forwarding wrappers and re-export seams are removed once the direct path exists.
- Each checked slice maps to exactly one dedicated commit.
