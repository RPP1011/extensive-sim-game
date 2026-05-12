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

### Gap #2 — `sum(... { 1u } else { 0u })` fails type-check

**Status:** worked around in-fixture by using `f32` (`{ 1.0 } else
{ 0.0 }`) and casting `expected_chamber_allies` to f32 for the compare.

**Discovery sequence:**

  1. Stage 2's `MissingAllySuspicion` rule needed a per-tick count of
     nearby live allies. The natural form
     `sum(other in agents where ... { 1u } else { 0u })` failed:
     `constructed CgExpr at 16451..16679 failed type-check —
     expr#284 claims result u32 but operands require f32`.
  2. The compiler silently dropped `MissingAllySuspicion` from the
     emit set when this type-check failed (no fatal error — just a
     `lower diag` warning).

**Fix surface (engine-side, deferred):**

  - The `sum(...)` builtin's WGSL emit lowers to `+=` (`atomicAdd`
    on the GPU side); the integer-vs-float dispatch is decided by
    the body-arm types. The check at expr#284 looks to be inferring
    `f32` for the parent expression while the arm is `u32`. Either
    the parent type should be re-inferred from the body (making u32
    sums work end-to-end) or the dispatcher should emit a clearer
    diagnostic + reject at parse-time rather than silently dropping
    the rule downstream.

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

### Gap #5 — `apply_ability Stealth by self target self` chronicle-→-SoA round-trip silently no-ops

**Status:** soft-pinned (warning printed, not asserted). The Rogue's
Stealth verb dispatches every 20 ticks (cd gate verified — verb body
is reached); the dispatcher's `kind == 27u` branch writes a kind=54
chronicle record with the correct slots (verified in
`out/dungeon_stealth/physics_RogueStealth.wgsl`); the consumer rule
`ApplyStealthFromChronicle` reads slot 2 (actor) + slot 3 (duration)
and writes `agent_stealth_until_tick[actor] = tick + duration`
(verified in `physics_ApplyStealthFromChronicle.wgsl`). End-to-end,
the readback observes `stealth_until_tick == 0` for every hero
across every sampled tick in a 100-tick run.

**Discovery sequence:**

  1. Pin asserts a sample-every-10-ticks readback should see some
     hero with `stealth_until_tick > world.tick`. At cd=20 and
     duration=50, the Rogue ought to be stealthed in the first ~50
     ticks of every 20-tick fire window — almost every readback
     sample should land inside a stealth window.
  2. 0 readbacks observed stealth at TICKS=100, sampled every 10
     ticks (10 sample points).
  3. The WGSL kernels for both halves of the round trip look
     correct on inspection. The chronicle ring is shared with the
     ApplyDamageFromChronicle / ApplyHealFromChronicle paths which
     DO fire (combat-side pins pass), so the ring itself is wired.

**Suspected causes (uninvestigated):**

  - `apply_ability X by self target self` for an ability with
    `target: self` might not route through the standard chronicle
    write path. (apply_ability_smoke.sim uses the same surface form
    but its target is `agents.level(self)` which is a numeric id
    rather than the symbolic "self".)
  - The per-agent dispatcher may emit the chronicle record into a
    different ring slot than the @phase(post) consumer reads from
    (event_ring index race).
  - The `agent_stealth_until_tick` buffer write might be racing with
    the per-agent kernel's read (writes during physics_RogueStealth
    aren't visible to the post-phase kernel — but the WGSL clearly
    binds it as `storage, read_write` in both kernels).

**Fix surface (engine-side, deferred to follow-up):**

  - Add a duel_abilities-style integration pin that fires `apply_ability
    Stealth by self target self` in isolation and reads back
    `agent_stealth_until_tick` — would isolate the chronicle vs.
    consumer race without 36 other kernels in the way.

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
