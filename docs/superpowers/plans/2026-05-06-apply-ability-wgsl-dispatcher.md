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

**Closes:** `0d2a20ca` (slice α — DataHandle scaffolding).

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

**Closes:** `04a71d94` (slice β step 2 — real WGSL dispatcher loop+switch),
expanded by `0b09317b` / `a7159689` / `940efc30` (variant coverage to
31/32 — only `CastAbility`=7 deferred to slice δ).

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

**Status (2026-05-06):** dispatcher + CPU pipeline scaffolding shipped
ahead of the actual sim wire-up — the original "1-verb edit + parity
gate" task list compressed into ~10 commits because each step
(EFFECT_KIND_TO_EVENT_KIND_ID pin → 7 chronicle arms wired → smoke
fixture → CPU reference → composition tests → P5 determinism pin →
ApplyEvent variants for TransferGold/ModifyStanding) needed its own
commit to keep the slice green.

**Shipped (in landed order):**
- `1779b0e6` — record EventRing(Append) write on ApplyAbility-bearing ops.
- `440a4073` — pin `EFFECT_KIND_TO_EVENT_KIND_ID` against engine
  `EventKindId` enum (7 chronicle-bearing variants: 0→26..6→32).
- `49345a59` — wire Damage arm to real chronicle_append.
- `31aba175` — wire 6 more chronicle-bearing arms (Heal/Shield/Stun/
  Slow/TransferGold/ModifyStanding).
- `8484ecde` — `assets/sim/apply_ability_smoke.sim` + pipeline smoke test.
- `4c91ae5f` — pin BGL composer wiring of event_ring + event_tail.
- `b15f6845` — `cpu_chronicle_reference` module (CPU ↔ GPU contract).
- `91267f01` — apply_program → CPU reference composition tests.
- `4b299211` — P5 determinism pin via 50-combo chance-gate sweep.
- `f881bed1` — wire `ApplyEvent::TransferGold` + `ModifyStanding`
  variants in engine; close the CPU reference's last `None`
  fall-throughs.
- `2876ecd0` — pipeline integration tests for the new ApplyEvent
  variants.

**Still pending in original scope:**
- Task 1 (sim edit) — behavior-changing; needs deliberate decision
  on which verb to swap and whether existing chronicle handlers
  remain correct in parallel.
- Task 3 (runtime parity gate) — needs a wgpu device-driving test;
  substantial scaffolding ahead of CI gating.
- Task 5 (CPU ↔ GPU byte-equality) — depends on task 3.

The CPU side of the parity equation is now a complete reference; the
GPU side needs a runtime crate to drive the dispatcher kernel before
the byte-equality assertion can run.

**Closes:** sequence above (no single commit closes the slice — the
behavior-changing piece blocks on the sim wire-up decision).

## Slice δ — full-vocabulary expansion (#137)

Expand the `switch (kind)` arms to cover every EffectOp variant the
LoL canary uses (currently 30+). Stretches across multiple
iterations; each variant adds one arm + one apply-path test.

**Status (2026-05-06):** completed. The dispatcher covers 31/32
EffectOp variants — all variants except `CastAbility=7` (recursive
dispatch deferred to a future slice). Shipped across the slice γ
sequence (`0b09317b` / `a7159689` / `940efc30`).

## Slice ε — explicit caster/target operands (#161)

**Status (2026-05-06):** completed across 4 commits.

The original slice γ chronicle wiring used a hardcoded `agent_id`
identifier for the chronicle records' actor + target slots ("slice-γ
self-cast convention"). This worked for PerAgent kernels (where
`agent_id` is the per-thread preamble local) but produced broken
WGSL for PerEvent kernels (no `agent_id` binding) — naga validation
caught this as a real bug.

Slice ε plumbs explicit operands through 5+ layers (AST, parser,
resolver, IR, CG IR, lowering, dispatcher emit, CPU reference) so
the user can write:

    apply_ability <ability>                            // PerAgent: caster=self, target=self
    apply_ability <ability> by <caster>                // PerEvent: explicit caster
    apply_ability <ability> target <target>            // PerAgent + explicit target
    apply_ability <ability> by <caster> target <target>// fully explicit

When operands are omitted, lowering defaults preserve the prior
self-cast behavior (chronicle byte layout unchanged).

**Shipped (in landed order):**
- `92572af8` — part 1: explicit `caster: CgExprId` field on
  `CgStmt::ApplyAbility`; dispatcher reads `caster_slot` instead of
  hardcoded `agent_id`. Naga validator caught a reserved-prefix
  issue (`__caster_slot` → `caster_slot`).
- `c22f105e` — part 2: `LoweringCtx.current_per_agent_rule` flag +
  typed `UnsupportedPhysicsStmt` error for PerEvent ApplyAbility
  without explicit caster (clean failure mode replacing broken
  WGSL emit).
- `7c3ce6e4` — part 3: `apply_ability <a> by <caster>` parser
  syntax. Closes task #161. PerEvent ApplyAbility now lowers
  cleanly with explicit caster (e.g. `by w` from event-pattern
  destructuring).
- `d0bc37fd` — slice ε part 1 (target operand): symmetric
  `target <expr>` syntax. The dispatcher writes distinct caster
  vs target chronicle slots when source supplies them.
- `efef23a1` — CPU reference signature update: now takes
  `target_id: u32` separately, mirroring the GPU dispatcher's
  distinct slot writes. Closes the CPU↔GPU contract.

**Naga validation track record:** caught 3 real layered bugs across
this session (BGL composer column-read recording in `f447d3eb`,
reserved-prefix `__caster_slot`, PerEvent shape gap surfaced as
#161). Each commit's validator gate paid for itself; the gate is
now a load-bearing CI guard.

**Pending slice ε scope (not yet started):**
- Multi-target / AOE: extend the operand surface from a single
  `target` expression to a list / spatial query result. Touches
  the dispatcher loop shape (per-target sub-loop) more than the
  source surface.
- Real GPU device test (#133): runs the kernel through wgpu and
  reads back the chronicle ring for byte-equality vs. the CPU
  reference. Substantial wgpu scaffolding ahead of CI gating.
- duel_abilities verb wire-up (#138): pick a self-cast verb
  (Bleed) and rewrite to `apply_ability` instead of inline
  emit. Behavior-changing — needs deliberate decision on which
  verb + chronicle handler convention.
