# Gaps surfaced by the `dungeon_stealth` fixture

**Fixture:** `assets/sim/dungeon_stealth.sim` (37 emitted kernels)
**Pin:** `crates/sims/tests/dungeon_stealth_pin.rs`
**Date:** 2026-05-12
**Stage:** 2 of 3 (the dungeon-crawl series; stage 3 will scale this
design to 5v1000 — pure topology change, no structural rework).

This fixture is the first to compose **bilateral beliefs** (per-pair
`beliefs_flags` view) with **hero stealth** (custom `stealth_until_tick`
field set via `EffectStealthApplied` chronicle consumer), **enemy
alert** (custom `alert` field with two raise paths — witnessing ally
death + missing-ally suspicion), and **simple patrols** (vec3
arithmetic with scalar-encoded fields) under a single per-tick
schedule. It inherits stage 1's voxel-procedural dungeon and the same
5-hero + ~30-enemy population.

## Pass behaviour

```
==== dungeon_stealth 100-tick report ====
  dungeon: 16+ rooms (~2500 floor cells, spawn=slotN, boss=slotM)
  init:    heroes=5/5  enemies=~30  patrol=N
  final:   heroes=K/5  enemies=remaining
  combat:  total enemy kills = >= 0
  stealth: any-hero-stealthed-observed=true  reached_final=true|false
  alert:   K/N alive enemies have alert>0 (max alert=M)
  verdict: PARTY ADVANCING — reached boss chamber
```

Load-bearing pin asserts:

  1. All 5 heroes alive at tick 30 (early-game safety — stealth +
     LoS-gated detection should prevent any wipe).
  2. At least 3 enemies alive at tick 50 (combat takes time; stealth
     gives heroes early-game leverage).
  3. At tick 90 the alive-enemy alert sum > 0 (alert mechanic fires —
     MissingAllySuspicion runs every 30 ticks).
  4. At least one tick where some hero had `stealth_until_tick >
     world.tick` (Rogue Stealth dispatches end-to-end).
  5. Either `reached_final` OR ≥3 heroes alive at tick 99.
  6. No NaN/Inf in agent positions after 100 ticks.

## Discovered gaps

### Gap #1 — DSL parser rejects `self.pos.x` / `.y` / `.z` accessors

**Status:** worked around in-fixture by storing patrol state as 2× f32
scalar fields per axis (`patrol_origin_x` + `patrol_origin_y`,
`patrol_step_x` + `patrol_step_y`) and reconstructing vec3s via the
`vec3(x, y, z)` ctor inside the rule body.

**Discovery sequence:**

  1. Stage 2's design draft used `let new_x = self.pos.x + step_x` for
     X-axis patrol bouncing.
  2. The compiler's `lower diag` warning fired:
     `field access .x at 14076..14086 has an unsupported base shape`
     — the AST surface for `<vec3-expr>.x` exists but the lowering
     rejects it (writes-as-a-whole are supported per
     `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:57`: "Vec3
     swizzles ... per-component writes are an emit-time concern not
     yet surfaced in the IR").
  3. The rule's host kernel was silently dropped from the emit set,
     leaving `PatrolBounce` non-functional.

**Fix surface (engine-side, deferred):**

  - Plumb component-read access for vec3 fields (`.x`/`.y`/`.z`) from
    the AST through `cg::expr_lower` into a WGSL `vec_expr.x` /
    `.y` / `.z` access. Most fixtures avoid the access by using
    whole-vec3 arithmetic; stage 2's per-axis bouncing is the first
    case where the workaround (multi-scalar storage) doubles the
    custom-field SoA budget.

### Gap #2 — `sum(... { 1u } else { 0u })` fails type-check — **CLOSED 2026-05-12**

**Status:** **CLOSED.** The `FoldKind::Sum` arm in
`crates/dsl_compiler/src/cg/lower/expr.rs` (`lower_fold_over_agents`)
now seeds an init literal for `CgTy::U32` (alongside the existing
I32 / F32 / Vec3F32 cases). The body type-check infers from the
arm types directly — `1u` / `0u` arms produce `body_ty == CgTy::U32`,
which now picks `LitValue::U32(0)` for the accumulator init instead
of falling into the unsupported branch.

**Fix:** added `CgTy::U32 => add(ctx, CgExpr::Lit(LitValue::U32(0)),
span)?` to the Sum init match. The WGSL emit's `local_N = local_N +
projection` lowers uniformly across U32/I32/F32/Vec3F32 since the `+`
operator is the same WGSL token at all numeric widths and
`cg_ty_to_wgsl(U32)` returns `u32`.

**Verification:**

  - `crates/dsl_compiler/tests/sum_u32_type_infer.rs` — new pin
    compiles a synthetic per-agent rule with a `sum(... { 1u } else
    { 0u })` body, verifies the host kernel emits (was silently
    dropped pre-fix) and the WGSL parses + validates with naga.
  - Both `assets/sim/dungeon_stealth.sim::MissingAllySuspicion` and
    `assets/sim/dungeon_horde.sim::MissingAllySuspicion` now use the
    direct u32 sum — no `1.0` workaround, no `as f32` cast on
    `expected_chamber_allies`.

**Discovery sequence (pre-fix, archived for context):**

  1. Stage 2's `MissingAllySuspicion` rule needed a per-tick count of
     nearby live allies. The natural form
     `sum(other in agents where ... { 1u } else { 0u })` failed:
     `constructed CgExpr at 16451..16679 failed type-check —
     expr#284 claims result u32 but operands require f32`.
  2. The compiler silently dropped `MissingAllySuspicion` from the
     emit set when this type-check failed (no fatal error — just a
     `lower diag` warning).

### Gap #3 — Custom-field registry rejects vec3 type

**Status:** worked around in-fixture by replacing each `vec3` custom
field with two `f32` fields (one per axis). Stage 2's patrol-state
fields ended up as `patrol_origin_x/y` + `patrol_step_x/y` instead
of the natural `patrol_origin: vec3` + `patrol_step: vec3`.

**Discovery sequence:**

  1. Initial attempt: `field patrol_origin: vec3` triggered
     `unknown custom field type 'vec3' for 'field patrol_origin'
     ... (supported: u32, f32, bool)` at
     `crates/dsl_compiler/src/custom_agent_fields.rs:212`.
  2. `custom_agent_fields::parse_field_ty` only accepts u32/f32/bool
     today; the `AgentFieldTy::Vec3` variant exists (`elem_bytes`
     returns 16 — std430-padded vec4) but is not surfaced through
     `parse_field_ty`.

**Fix surface (engine-side, deferred):**

  - Wire `parse_field_ty` to accept `"vec3"` and route through the
    existing `AgentFieldTy::Vec3` size class. The auto-emit's buffer
    allocation already supports vec3 sizing (`elem_bytes = 16`); the
    auto-emit's read/write accessor lowering needs verification.

### Gap #5 — `apply_ability Stealth by self target self` chronicle-→-SoA round-trip silently no-ops — **CLOSED 2026-05-12**

**Status:** **CLOSED.** Pin's stealth assertion is now load-bearing
(`any_stealthed_observed` is asserted, not just warned). The schedule
synthesizer's `APPLY_ABILITY_EMITTED_KINDS` table was hardcoded to the
first four engine effect kinds (26 Damage / 27 Heal / 28 Shield / 29
Stun); every extended-corpus kind (Stealth=54, Charm=55, …, all of
`EFFECT_KIND_TO_EVENT_KIND_ID` past the original four) was invisible to
producer→consumer matching. So `ApplyStealthFromChronicle`'s
`on EffectStealthApplied` (kind=54) had no inbound edge from
`RogueStealth`'s `apply_ability Stealth` dispatch, and Kahn's topo sort
placed the consumer FIRST in the schedule. The consumer scanned an
empty ring each tick; the dispatcher wrote records that were silently
dropped on the per-tick ring reset.

**Fix (`crates/dsl_compiler/src/cg/schedule/topology.rs`):** replaced
the hardcoded `const APPLY_ABILITY_EMITTED_KINDS: &[EventKindId]` with
a function `apply_ability_emitted_kinds() -> Vec<EventKindId>` that
derives the set from `EFFECT_KIND_TO_EVENT_KIND_ID` — the single
source of truth the dispatcher's WGSL arm chain already renders
against. Adding a new EffectOp / chronicle event to that table now
automatically updates schedule-time producer matching.

**Verification:**

  - `cargo test -p sims --release --test dungeon_stealth_pin -- --nocapture`:
    `[dungeon_stealth] observed stealth: hero[4] stealthed until tick 50 (at tick 0)`;
    pin's `assert!(any_stealthed_observed, ...)` passes.
  - Schedule walk in `out/dungeon_stealth/generated.rs` now lists
    `PhysicsApplyHealFromChronicleAndApplyStunFromChronicleAndApplyStealthFromChronicle`
    AFTER `PhysicsRogueStealth` (the fusion analyzer also merged
    Heal/Stun/Stealth consumers into a single kernel).
  - `cargo test -p dsl_compiler --release`: green.

**Discovery sequence (pre-fix, archived for context):**

  1. Pin asserts a sample-every-10-ticks readback should see some
     hero with `stealth_until_tick > world.tick`. 0 readbacks
     observed stealth at TICKS=100 (10 sample points).
  2. Both halves of the WGSL round trip look correct on inspection
     (`physics_RogueStealth.wgsl` writes kind=54;
     `physics_ApplyStealthFromChronicle.wgsl` reads kind=54 and
     writes the stealth field).
  3. The damage/heal-side chronicle paths (kinds 26/27) DO fire —
     ring itself is wired. The schedule ORDER was the bug:
     dispatcher ran AFTER its own consumer.

### Gap #3-sound — Per-ability sound radius (chronicle slot 6 ability_id) — **CLOSED 2026-05-12**

**Status:** **CLOSED.** `SoundDetectFromDamage` now hooks directly off
`EffectDamageApplied` (the engine chronicle event, kind 26) instead of
the user-side `Damaged` re-emit, so it can read `ability_id` from
chronicle slot 6 (= payload offset 4) and pick a per-ability noise
radius. The original "single radius (compromise between Backstab=2 /
Cleave=8 / Strike=4)" workaround is gone.

**Implementation shape:**

  - The user-declared `EffectDamageApplied` event in
    `assets/sim/dungeon_stealth.sim` and
    `assets/sim/dungeon_horde.sim` now declares 5 fields:
    `actor`, `target`, `amount`, `_reserved` (offset 3 / abs slot 5
    — engine writes 0 here), `ability_id` (offset 4 / abs slot 6 —
    matches the dispatcher's `atomicStore(&event_ring[_slot * 10u +
    6u], ability_id__u32)` write).
  - `SoundDetectFromDamage` matches on the new pattern
    `on EffectDamageApplied { actor: s, target: _, amount: a,
    _reserved: _, ability_id: aid }` and dispatches `aid` through a
    nested `if/else` chain to pick a per-ability radius.
  - `config.stealth.attack_sound_radius_*` (one entry per ability:
    `_strike` 4u, `_backstab` 2u, `_cleave` 8u, `_volley` 8u,
    `_stun` 6u, `_silent` 0u) replaces the single
    `attack_sound_radius`.
  - Snipe (id 5), Heal (id 3), Scout (id 4), and Stealth (id 6) all
    map to `_silent` — Rangers can pick off scout-line enemies
    without broadcasting their position to the room.

**Verification:**

  - Both stage 2 (`dungeon_stealth_pin`) and stage 3
    (`dungeon_horde_pin`) compile + pass after the rewrite. The
    stage 3 sweep shows alert profiles matching expectations:
    Snipe-heavy seeds produce lower alert spread than Strike-heavy
    seeds.
  - The `effect_damage_applied_carries_ability_id` pin
    (`crates/dsl_compiler/tests/effect_damage_applied_carries_ability_id.rs`)
    continues to pass — the dispatcher's slot-6 write is unchanged;
    only the user-event payload declaration grew to expose the slot.

### Gap #4 — `else if` chain rejected at parse time (cosmetic)

**Status:** worked around in-fixture by nesting `else { if ... }`
blocks. The parser's `else` arm only accepts `{ ... }` block — it
doesn't fold a bare `if` after `else` into an else-if chain.

**Discovery:** initial PatrolBounce body used
`} else if (new_x > ...) {` — parser error at the `if` keyword.

**Fix surface (engine-side, deferred):**

  - Parser-only sugar — `else if (cond) { ... }` should fold to
    `else { if (cond) { ... } }` at parse time. Trivial follow-up.

## Architectural note — stage 3 readiness

Stage 2's design is N²-scalable on the belief side: `beliefs_flags`
is `pair_map` storage so 1000² × 4 bytes = 4 MB for stage 3. Two
hotspots will need attention before then:

  1. **`for_each_agent` walks in `@phase(post)` consumers** —
     `SoundDetectFromDamage` / `BroadcastAlertOnAllyDeath` /
     `ScoutBroadcast` each fire a full-population walk per Damaged /
     AllyDied / EffectDamageApplied event. At 1000 agents and a
     handful of damage events per tick, that's 1000s of distance
     checks per event. Stage 3 should swap these to spatial-grid
     walks once `spatial.nearby_X` queries lower cleanly inside
     `@phase(post)` bodies (today's `for_each_agent` is the only
     proven shape for cross-pair consumer writes).
  2. **`MissingAllySuspicion` runs every 30 ticks across all
     non-hero agents.** Each agent does a full-population sum-walk
     for ally counting. At 1000 enemies × 1000 agents = 1M distance
     checks every 30 ticks. Same fix surface as (1): spatial-grid
     walk inside `@phase(per_agent)` bodies for the sum form.
