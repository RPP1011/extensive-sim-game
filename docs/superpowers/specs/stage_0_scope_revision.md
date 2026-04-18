# Stage 0 scope revision — 2026-04-17

Per the plan, Stage 0 wrapped the 13 compute/apply sub-phases and 119 postapply systems as `impl System`. On closer inspection, this was over-scoped:

- **Compute phases** (`compute_high`, `compute_medium`, `compute_low`) dispatch *per entity* inside the tick loop via fidelity classification. Wrapping as phase-level Systems would require restructuring to three separate passes, which hurts cache efficiency. The existing `TickProfile.compute_{high,medium,low}_us` already captures stage-level timing.
- **Apply sub-phases** other than movement (HP, status, economy, etc.) already have distinct `ApplyProfile` timers. Adding System wrappers yields no new profiling data; the wrapper is only valuable where backend dispatch (scalar vs SIMD) will live.
- **Postapply systems** are branchy HashMap-heavy per-settlement logic — SIMD won't help. They need per-system timing, but that's `time_bare!` macro territory (Stage 1 Task 1.3), not trait wrapping.

## What landed

- `System` trait + `Stage` + `Backend` + `SystemCtx` + `SystemRegistry` (Tasks 0.1, 0.2)
- `ApplyMovementSystem` pilot with backend-enum dispatch (Task 0.3) — the template for future SIMD-candidate wrappers

## What's deferred

- Per-phase System wrappers for other apply sub-phases and compute phases: added opportunistically when a SIMD target surfaces for that sub-phase
- `build_registry()` populating compute/apply systems: deferred until a real consumer needs registry iteration
- Postapply system migration: not happening. `time_bare!` provides per-system visibility without the boilerplate

## Rationale

Plan's spec was correct about what the harness needs: per-system timing, backend dispatch, SIMD-targeting. The implementation path to get there is simpler than wrapping 132 Systems. YAGNI: new wrappers land as part of SIMD PRs, not in a pre-emptive refactor.
