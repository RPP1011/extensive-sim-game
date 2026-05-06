# `apply_ability` WGSL dispatcher

> Multi-iteration plan for the GPU side of `#125` (registry-driven
> apply-handler dispatch). The CPU oracle (`engine::ability::apply::apply_program`)
> is the cross-backend reference; this plan brings the GPU backend
> to parity for an initial vocabulary, then expands.

## Goal

Real WGSL emit for `CgStmt::ApplyAbility` so a `.sim` rule can
write `apply_ability self.queued_ability` and have the GPU
backend dispatch the lowered effects (Damage / Heal / Stun /
Slow at first) bit-equivalently to the CPU oracle.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `CgStmt::ApplyAbility` at `crates/dsl_compiler/src/cg/stmt.rs:426` (structural variant — landed `99f953dd`)
  - `PackedAbilityRegistry` at `crates/engine/src/ability/packed.rs` (CPU SoA — Wave 1.9)
  - `PackedAbilityRegistryGpu` at `crates/engine/src/ability/registry_gpu.rs:46` (GPU buffer set — `#135`)
  - `apply_program` at `crates/engine/src/ability/apply.rs:131` (CPU oracle — Wave 2)
  - `DataHandle::SpatialStorage` at `crates/dsl_compiler/src/cg/data_handle.rs:969` (BGL-bound buffer pattern to mirror)
  Search method: `rg` + `Read`.

- **Decision:** extend the existing `DataHandle` enum with one new
  variant per `PackedAbilityRegistry` SoA column rather than smuggling
  a single opaque "AbilityRegistry" handle. The fine-grained variants
  let the BGL composer bind only the columns each kernel actually
  reads (most ApplyAbility kernels only need `effect_kinds` +
  `effect_payload_a/b` + `chances` for the dispatch path; cooldown /
  range / hint live in the cast-decide kernel that runs upstream).

- **Rule-compiler touchpoints:**
  - DSL inputs edited: none in slice α (structural). One `.sim`
    fixture (likely `duel_abilities.sim`) gains `apply_ability` in
    slice γ.
  - Generated outputs re-emitted: `crates/duel_abilities_runtime/src/`
    re-emits its WGSL kernel set in slice γ.

- **Hand-written downstream code:** NONE.
  (Apply paths live in WGSL emitter strings — generated.)

- **Constitution check:**
  - P1 (Compiler-First): PASS — every new arm lives in the WGSL emitter.
  - P2 (Schema-Hash on Layout): PASS — the new `DataHandle` variants
    don't change `SimState` layout. PackedAbilityRegistry's SoA
    column shapes are already in the schema hash (line 60 of
    `crates/engine/src/schema_hash.rs`).
  - P3 (Cross-Backend Parity): PASS — slice γ ships a parity test
    that runs the same `apply_program(ability_id, caster, target)`
    call on both backends and asserts byte-equal `ApplyEvent` traces.
  - P4 (EffectOp Size Budget): N/A.
  - P5 (Determinism via Keyed PCG): PASS — chance gating reuses
    `per_agent_u32(seed, caster, tick, purpose=AbilitySlot)` per
    Wave 1.5#5; the WGSL implementation already exists.
  - P6 (Events Are the Mutation Channel): PASS — the dispatcher
    appends `ApplyEvent::*` to the chronicle ring; HP/timer writes
    happen in the existing physics handlers downstream (ApplyDamage,
    ApplyHeal, etc.).
  - P7 (Replayability Flagged): PASS — every ApplyEvent variant is
    `replayable = true` (set in `engine_data/src/events/mod.rs`).
  - P8 (AIS Required): PASS — this section satisfies it.
  - P9 (Tasks Close With Verified Commit): PASS — each slice closes
    with a non-reverted commit on `main`.
  - P10 (No Runtime Panic): PASS — the dispatcher uses
    `select(...)` for variant branching; OOB-protected by
    `effect_kinds[i] != EFFECT_KIND_EMPTY` early-out.
  - P11 (Reduction Determinism): N/A — the dispatcher's per-effect
    writes are independent (each goes to a distinct chronicle slot
    via atomic-tail append, not a reduction).

- **Runtime gate:** slice γ's parity test
  (`crates/duel_abilities_runtime/tests/apply_ability_parity.rs`)
  runs both backends on the same input and asserts byte-equal
  `ApplyEvent` traces. Slices α and β are pure-types / pure-emit
  dry-runs and have no runtime gate.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).
  [ ] AIS reviewed post-design (after task list stabilises).

## Slice α — DataHandle scaffolding (structural only)

**Scope:** add `DataHandle::AbilityRegistryColumn { column }` and the
`AbilityRegistryColumn` enum (one variant per PackedAbilityRegistry
SoA column). Wire through every exhaustive `match DataHandle` site
(BGL composer, WGSL identifier resolution, type-checker, schedule
synthesizer, debug printer). No emit-side behavior change — the
new variants surface as `unimplemented!()` at WGSL emit time, which
is reachable only via `apply_ability` usage (zero in the corpus
today).

**Tasks:**
1. Add `enum AbilityRegistryColumn` (12 variants, one per SoA column).
2. Add `DataHandle::AbilityRegistryColumn { column }` variant.
3. Wire exhaustive matches (~10 sites estimated; mirror the
   `CgStmt::ApplyAbility` blast-radius pattern — `99f953dd`).
4. Workspace tests stay green; ApplyAbility WGSL emit still
   returns the existing comment placeholder.

**Closes:** `<commit-sha>` (slice α).

## Slice β — BGL composer + WGSL emit shape

**Scope:** map each `AbilityRegistryColumn` to a stable WGSL
identifier (`ability_registry_effect_kinds`, etc.) and a binding
slot. Replace the placeholder `// TODO` in `wgsl_body.rs` with real
WGSL code that:
  - reads `ability_id` from the operand expression,
  - computes `slot = id - 1; base = slot * MAX_EFFECTS_PER_PROGRAM`,
  - loops `for (var i = 0u; i < MAX_EFFECTS_PER_PROGRAM; i++)`,
  - reads `kind = ability_registry_effect_kinds[base + i]`,
  - early-outs on `kind == EFFECT_KIND_EMPTY`,
  - dispatches via `switch (kind)` with arms for each implemented
    EffectOp (slice β covers Damage=0, Heal=1, Stun=3, Slow=4 — the
    duel_abilities staples).

**Tasks:**
1. `data_handle_to_wgsl_id` arm for each column.
2. BGL composer arm that binds each column to its kernel's BGL.
3. WGSL emit string template for the dispatcher.
4. Apply-path WGSL for Damage / Heal / Stun / Slow (chronicle ring
   atomic-append, mirrors existing `Emit` shape).
5. WGSL-emit golden-file test that pins the dispatcher kernel
   string for a fixture that uses `apply_ability` (no runtime).

**Closes:** `<commit-sha>` (slice β).

## Slice γ — duel_abilities wire-up + parity gate

**Scope:** wire one `duel_abilities` verb to use `apply_ability`
end-to-end (replacing its current bare `damage` body). Run the
sim on both `SerialBackend` and `GpuBackend` and assert the
`ApplyEvent` traces are byte-equal across N ticks.

**Tasks:**
1. Edit `assets/sim/duel_abilities.sim` — pick one verb (Strike?)
   and replace its bare `damage 30` with `apply_ability <expr>`.
2. Re-emit `duel_abilities_runtime`.
3. Add `tests/apply_ability_parity.rs` runtime gate.
4. Bump LoL canary baseline if any decl now lowers differently.
5. CPU↔GPU byte-equality assertion (P3).

**Closes:** `<commit-sha>` (slice γ).

## Slice δ — full-vocabulary expansion (#137)

Expand the `switch (kind)` arms to cover every EffectOp variant the
LoL canary uses (currently 30+). Stretches across multiple
iterations; each variant adds one arm + one apply-path test.
