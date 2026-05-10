# Cast State, Interrupts, Threat Zones + AI Consumption (Plan G)

> Goal: extend the existing **multi-tick activity infrastructure** (Lift A `travel_to` + Lift B `cast_recipe`, both already using the universal `BusyUntilTick` SoA column) with three new layers: (1) interrupt mechanics that can cancel any in-progress busy activity uniformly, (2) a `threats` materialised view AI can query spatially, (3) a new `cast { … } effect { … }` grammar block for cast-time abilities. All compiler-emitted from new DSL primitives. Plan V (visualization) reads cast state + threats from sim later; Plan E (emit refactor) refactors the per-verb kernels into generic data-driven dispatchers later still.

## Goal

The engine already has a generic multi-tick activity gate: `BusyUntilTick`. Lift A's `travel_to` writes it (caster moves over multiple ticks toward a destination); Lift B's `cast_recipe` writes it (caster crafts an item over multiple ticks); both consult it as a "I'm busy, can't do anything else" mask. The comment in `crates/dsl_compiler/src/cg/data_handle.rs:240`:

> Multiple busy sources (travel / craft / negotiate) share the same column; a busy agent can't do anything else regardless of which verb made them busy.

What's missing for a viable game-mechanics layer:

- **Interrupt mechanics**: today a busy agent stays busy until `busy_until_tick`, no matter what. A boss can charge a Firebolt for 3 ticks; nothing the heroes do can interrupt it.
- **Threat zones**: AI can't query "is there an in-flight cast threatening me?" — the busy state has no spatial projection.
- **Cast-time ability grammar**: today `cast_time` is implicit in specific verbs (`travel_to (x, y, z) for 5t`, `cast_recipe id`); there's no general `ability X { cast { duration: 3t } effect { … } }` shape for arbitrary cast-time gameplay.
- **AI consumption primitives**: no `where threats.in_zone(self)` for dodging.

This plan adds those four things on top of the existing busy infrastructure. It's a generalization, not a rebuild — recipes and travel automatically benefit from the new interrupt + threat-zone layer once the busy-source consumers are updated to populate the new context fields.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/dsl_compiler/src/cg/data_handle.rs:236-256` — `BusyUntilTick`, `TravelDestX/Y/Z` SoA columns; the universal busy gate.
  - `crates/engine/src/ability/program.rs:697-739` — Lift A `EffectOp::TravelTo` (variant 39) + Lift B `EffectOp::Recipe` (variant 40); consumer rules already set `busy_until_tick`.
  - `docs/spec/ability.md:733-742, 1976-1981, 2202` — full lift inventory; all four lifts (A travel / B production / C consent / D skill+obligation) are runs-today.
  - `crates/dsl_ast/src/ability_parser.rs` — current `.ability` parser; needs new `cast { … }` block.
  - `crates/dsl_ast/src/ast.rs::AbilityDef` — needs the program-step shape upgrade.
  - `crates/dsl_compiler/src/cg/lower/scoring.rs` — current scoring lowering; new spatial primitives plug in here.
  - `crates/engine/src/cascade/views.rs` — current materialised-view registry; the new `threats` view registers here.
  - Per-fixture runtime examples (`crates/duel_25v25_runtime/`, `crates/village_economy_runtime/` for `cast_recipe`, `crates/foraging_real_runtime/` for `travel_to`).

  Search method: `rg`, `grep -nE`, direct `Read`.

- **Decision:** extend existing — generalize the `BusyUntilTick` infrastructure with context fields the interrupt + threat-zone layer needs. New `cast { … } effect { … }` grammar block lowers to a new `CastBegin` EffectOp that uses the same busy machinery. Travel + recipe consumers gain a single field (`busy_with_ability_id`) so they participate in the same interrupt + threat surface. All implementation via the compiler emit pattern (P1).

- **Rule-compiler touchpoints:**
  - DSL inputs edited: every `.ability` file that wants cast-time behaviour gets a `cast { … } effect { … }` program shape; existing `.ability` files (no `cast` block, single effects list) keep current instant-resolve semantics. Existing `travel_to`/`cast_recipe` consumers in `.sim` files don't change. New fixture `assets/sim/firebolt_probe.sim` + `assets/ability_test/firebolt_probe/Firebolt.ability` for the MVP.
  - Generated outputs re-emitted: every per-runtime `OUT_DIR/generated.rs` regens with new busy-context SoA columns + (where used) `threats` view + new scoring primitives. Per-runtime build.rs auto-regenerates.

- **Hand-written downstream code:** NONE. All new engine machinery is compiler-emitted. The new `firebolt_probe_runtime` crate's `lib.rs` follows the existing per-fixture pattern.

- **Constitution check:**
  - **P1 (Compiler-First Engine Extension):** PASS. Cast lifecycle, threat view materialisation, interrupt detection, scoring primitives — all emitted from the new DSL grammar. No `impl Rule` in `crates/engine/src/handlers/`.
  - **P2 (Schema-Hash on Layout):** REQUIRES BUMP. Adds per-agent SoA columns to the busy family: `BusyWithAbilityId: u32`, `BusyStartedAtTick: u32`, `BusyTargetSlot: u32`, `BusyTargetPos: Vec3Padded`. Adds new chronicle event variants: `CastBegan`, `CastResolved`, `CastInterrupted`. `crates/engine/.schema_hash` regenerates per documented procedure.
  - **P3 (Cross-Backend Parity):** PASS. Cast lifecycle is emitted from the same compiler path that already runs cross-backend. Threats view fold + interrupt detection are deterministic operations the existing emit pattern handles. New `tests/parity_cast_lifecycle.rs` exercises a Firebolt cast + a recipe cast on both backends and asserts byte-equal busy SoA + chronicle output.
  - **P4 (`EffectOp` Size Budget):** PASS. New `EffectOp::CastBegin { ability_id: u16, duration_ticks: u16, target_slot: u32, target_pos_q8: [i16; 3], _pad: [u8; 2] }` is 16 bytes. Verified at compile time via existing `const_assert!`.
  - **P5 (Determinism via Keyed PCG):** PASS. No new RNG draws on the deterministic path. Threats view reads busy SoA; cast lifecycle is event-driven from `apply_ability` calls (which already use keyed PCG for target selection).
  - **P6 (Events Are the Mutation Channel):** PASS. Busy SoA mutations come from `CastBegan` / `CastResolved` / `CastInterrupted` events (and existing `EffectTravelToApplied` / `EffectRecipeApplied`). Direct field writes confined to `step::*` and `snapshot::*`.
  - **P7 (Replayability Flagged):** PASS. New events declared `replayable: true` (cast lifecycle is part of the deterministic record; replays must reproduce the same casts in the same ticks). Telegraph display (Plan V) reads from the replayable event stream.
  - **P8 (AIS Required):** PASS — this section satisfies it.
  - **P9 (Tasks Close With Verified Commit):** PASS — every task closes with a SHA on the active branch.
  - **P10 (No Runtime Panic):** PASS. Cast resolution and interrupt detection use saturating ops + `Result` bubbling. New tests added to `tests/proptest_baseline.rs` for cast-state edge cases (cast at tick 0, interrupt at exactly resolve-tick, recipe interrupted mid-craft).
  - **P11 (Reduction Determinism):** PASS WITH SORT. The `threats` view fold over agents-with-active-busy must use `sort_by((busy_with_ability_id, agent_slot))` before reducing into the view. New `tests/parity_threats_view.rs` asserts byte-equal output across backends and runs at the same seed.

- **Runtime gate:** every phase has a runtime test exercising the new code path.
  - **G1**: `dsl_ast::tests::cast_block_parses` — load a `.ability` with `cast { duration: 3t; telegraph: line(...); interrupts: standard }`, assert AST has `AbilityProgramStep::Cast(CastSpec { duration_ticks: 3, ... })`.
  - **G2**: `dsl_compiler::tests::cast_lowering_emits_busy_writes` — lower a cast-time ability, assert emitted WGSL writes `busy_until_tick` + `busy_with_ability_id` + `busy_started_at_tick` + target context. Existing `travel_to` / `cast_recipe` lowering is also updated to write `busy_with_ability_id` alongside their existing `busy_until_tick` writes.
  - **G3**: `firebolt_probe_runtime::tests::cast_state_advances_per_tick` — drive 5 ticks of a Firebolt cast, assert per-tick: `busy_with_ability_id != 0`, `busy_started_at_tick` stable, `busy_until_tick - current_tick` decreases. At resolve tick, assert effect event written + busy state cleared.
  - **G4**: `firebolt_probe_runtime::tests::dodger_avoids_firebolt_when_warned` — caster + dodger, Firebolt with 3t cast, dodger scores threat-aware Move-Perpendicular, dodger leaves line before resolution, no damage taken.
  - **G4 (inverse pin)**: `firebolt_probe_runtime::tests::dodger_eats_damage_when_no_warning` — same setup but `cast { duration: 0t }`, dodger has no time to react, takes damage.
  - **G4 (interrupt pin)**: `firebolt_probe_runtime::tests::firebolt_interrupted_by_damage_during_cast` — caster initiates Firebolt, dodger fires counter-attack mid-cast that lands a `Damaged` event, cast interrupted, no damage at resolve tick, `CastInterrupted` chronicle event present.
  - **G4 (recipe pin)**: `firebolt_probe_runtime::tests::recipe_craft_interrupted_by_damage` — same fixture, second ability is `cast_recipe ironsword` with 5t duration. Caster begins crafting; dodger attacks mid-craft; recipe interrupted, no output spawned. Proves the new interrupt layer applies uniformly across busy sources.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

## Key insight: align with existing busy infrastructure

The existing engine already has a universal multi-tick activity gate (`BusyUntilTick`) used by Lift A (`travel_to`) and Lift B (`cast_recipe`). Plan G's earlier draft proposed parallel `cast_*` SoA columns; that was wrong. The right design **extends the busy family** with the missing context the interrupt + threat layer needs. Travel + recipe automatically gain interrupt mechanics + AI threat-awareness once their consumers populate the new context fields.

## Grammar additions

Decisions locked in 2026-05-09:

### Cast as ability program block (new)

```text
ability Firebolt {
    target: enemy
    range: 30
    cooldown: 5s

    cast {
        duration: 3t
        telegraph: line(self.pos, target.pos, width: 2)
        interrupts: standard
    }

    effect {
        damage 25 in line(self.pos, target.pos, width: 2) [FIRE: 100]
    }
}
```

Composable — multi-stage abilities chain `cast { } effect { } cast { } effect { } …`. A "Bind Soul" with charge-up + bind + drain becomes three cast blocks each with its own duration / telegraph / interrupts and effect blocks between them.

Backwards compatible: an ability with no `cast` block compiles to the existing immediate-resolve path.

### Interrupts uniformly applied to existing busy sources

Existing `travel_to` and `cast_recipe` verbs gain implicit interrupts via the same machinery. The .ability author can specify, e.g.:

```text
ability HuntForage {
    target: position
    cooldown: 0

    # `travel_to` is an existing Lift A verb; the new `interrupts:`
    # syntax around it just declares which interrupts apply to this
    # specific cast-time activity (the verb itself doesn't change).
    cast {
        duration: from-verb              # use the verb's own eta_ticks
        interrupts: standard - { damage } # forager keeps moving even under fire
    }
    effect {
        travel_to (target.x, target.y, target.z)
    }
}
```

Or for recipes:

```text
ability ForgeIronsword {
    target: self
    cooldown: 0

    cast {
        duration: 5t
        interrupts: standard
    }
    effect {
        cast_recipe ironsword
    }
}
```

`duration: from-verb` extracts the duration from the verb in the effect block (for `travel_to` this is the computed eta_ticks; for `cast_recipe` it's the recipe's intrinsic duration). For abilities with constant cast time use `duration: 5t`.

Existing `.ability` files using `travel_to` / `cast_recipe` directly (no enclosing `cast { } effect { }` block) keep working unchanged with default interrupt semantics (= `standard`). Authors who want non-default interrupts opt into the wrapper block.

### Cooldown phase qualifier

```text
cooldown: 5s              # defaults to `@ cast` (spam-cancel can't bypass)
cooldown: 5s @ resolve    # cooldown begins when the cast lands
cooldown: 5s @ interrupt  # only consumed on interrupt (rare)
```

### Interrupts: named set with set ops

```text
set standard = { damage, stun, caster_died, target_died }

interrupts: standard                       # the named default set
interrupts: { damage, stun }               # explicit subset
interrupts: standard + { movement }        # standard plus extras
interrupts: standard - { damage }          # standard minus exclusions
interrupts: none                           # uninterruptible (Bind Soul, etc.)
```

`standard = { damage, stun, caster_died, target_died }` per design 2026-05-09. `target_died` is a no-op for activities without a single target (AOEs, recipes targeting self, travel to a position).

### New scoring primitives

```text
threats.in_zone(self)                  # bool
threats.intensity_at(pos)              # f32 — sum of intensities at pos
threats.nearest(self)                  # ThreatHandle — closest active threat
threats.dir_away_from_nearest(self)    # vec3 — unit vector away from nearest
busy.is_busy(self)                     # bool — generic busy gate
busy.with_ability(self)                # AbilityId — what self is busy with
busy.target_of(self)                   # AgentHandle — what self is busy targeting
busy.ticks_until_done(self)            # u32 — how long until self's busy clears
```

These compose with existing scoring grammar (`where ...`, `score K`, etc.).

## Engine extensions (compiler-emitted)

### Busy SoA family extension

Existing column (do not change):
- `BusyUntilTick: u32` — tick at which the agent's current activity completes. `0` = not busy.

New columns added by this plan:
- `BusyWithAbilityId: u32` — which ability is making me busy. `0` = idle. Populated by ALL busy-source consumers: TravelTo, Recipe, CastBegin.
- `BusyStartedAtTick: u32` — when the activity began. Used for telegraph progress + threat zone duration.
- `BusyTargetSlot: u32` — `u32::MAX` = no single target.
- `BusyTargetPos: Vec3Padded` — target world position, snapshot at busy start.

Existing companion columns (do not change):
- `TravelDestX/Y/Z: f32` — Lift A travel destination. Stays per-source for now.

Schema hash bumps per P2.

### CastBegin EffectOp (new) — variant 46

```rust
EffectOp::CastBegin {
    ability_id: u16,
    duration_ticks: u16,
    target_slot: u32,
    target_pos_q8: [i16; 3],
    _pad: [u8; 2],
} = 46
```

Total: 2 + 2 + 4 + 6 + 2 = 16 bytes (P4 verified). Variant 46 is contiguous with Lift D's CreateObligation=45.

When the new `cast { duration: 3t } effect { damage 25 in line(...) }` block lowers, the dispatcher emits `CastBegin` with the cast's metadata. The CastBegin consumer:
1. Sets `busy_until_tick = current_tick + duration_ticks`
2. Sets `busy_with_ability_id = ability_id`
3. Sets `busy_started_at_tick = current_tick`
4. Sets `busy_target_slot` + `busy_target_pos`
5. Emits `CastBegan` chronicle event (replayable: true)

### Existing busy-source consumer migration

The existing `TravelTo` and `Recipe` consumer rules are extended (same .sim files, recompiled by build.rs) to ALSO write `busy_with_ability_id` + `busy_started_at_tick` + `busy_target_*` alongside their existing `busy_until_tick` writes. This is the only change to existing per-fixture behaviour and it's purely additive — every existing pin still passes (regression check in G5).

### Busy resolution kernel (compiler-emitted)

Per-tick kernel `physics_BusyTick @phase(per_agent)`:
1. For each agent with `busy_until_tick != 0`:
2. Check interrupt conditions (per ability program's `interrupts` set + per-tick events). If interrupted: emit `CastInterrupted`, clear all busy SoA columns. Done.
3. Else check elapsed: if `current_tick >= busy_until_tick`: dispatch the resolution effect (look up by `busy_with_ability_id` in the ability metadata table). For `cast_recipe` resolution this is "produce output X"; for `travel_to` it's "snap pos to dest"; for new cast { } effect { } it's the queued effect block. Emit `CastResolved`, clear busy SoA.
4. Else continue (no-op).

Compiler-emitted from the new ability-program AST shape + the existing TravelTo/Recipe consumers being uniformly recognised as busy sources.

### Threats materialised view

`view threats` registered cross-rule:
- Per-tick fold over agents with `busy_with_ability_id != 0` AND ability metadata indicates a threat-projection (typically: the ability has an `area:` field or `telegraph:` field).
- For each, project the telegraph shape into world space using `busy_target_pos` (or `agent_pos` if self-cast) → produce a `ThreatZone { origin, shape, intensity, expires_at_tick }` row.
- Reduction sorted by `(busy_with_ability_id, agent_slot)` per P11.
- Recipes typically have no threat projection (you're not threatening anyone by forging a sword), so they don't show in the threats view; travel similarly. Cast-time abilities with a `telegraph:` field do.

Spatial query primitives lower to indexed reads against this view.

### Interrupt detection

Compiler emits per-interrupt-source detection rules from each ability's `interrupts` set:
- `damage`: scan `Damaged` events this tick; if target is an agent with active busy that includes `damage` in its interrupts, mark for interruption.
- `stun`: scan `EffectStunApplied` events same way.
- `caster_died`: scan `Defeated` events; if defeated agent had active busy, interrupt.
- `target_died`: scan `Defeated` events; if defeated agent matches `busy_target_slot` of an active busy that includes `target_died`, interrupt.
- `movement`: comparison of pre-tick / post-tick `agent_pos` (only for casts whose interrupts include `movement`).

These all emit as `CastInterrupted` events that the busy-resolution kernel consumes.

## AI consumption pattern

Per-fixture `.sim` files use the new scoring primitives:

```text
# In firebolt_probe.sim
scoring DodgerBehaviour @phase(scoring) {
    # Strong negative score for staying in a threat zone — drives
    # the dodger to move out of it.
    score -200.0 where (self.alive
                       && threats.in_zone(self))

    # Positive score for moving away from nearest threat.
    score 100.0 where (self.alive
                      && threats.in_zone(self))
                action move(threats.dir_away_from_nearest(self))
}
```

Same primitives compose with existing scoring grammar.

## MVP fixture: firebolt_probe

Two-agent fixture demonstrating the full G stack end-to-end:

- **Caster** (slot 0): every 5 ticks, casts Firebolt at Dodger.
- **Dodger** (slot 1): scoring rows above. When in threat zone, moves perpendicular to the line.
- **Smith** (slot 2, second test): forges an Ironsword via `cast_recipe ironsword` with `interrupts: standard` (recipe pin).

Firebolt:
```text
ability Firebolt {
    target: enemy
    range: 30
    cooldown: 5s

    cast {
        duration: 3t
        telegraph: line(self.pos, target.pos, width: 2)
        interrupts: standard
    }

    effect {
        damage 25 in line(self.pos, target.pos, width: 2) [FIRE: 100]
    }
}
```

Behavioural pins (deterministic via P5 PCG seeding):

1. **`dodger_avoids_firebolt_when_warned`** — Caster initiates Firebolt at tick T. Threats view materialises with the line zone. Dodger detects `threats.in_zone(self)`, scores high on perpendicular move, leaves the line by tick T+3. Damage taken = 0. Cast resolves at tick T+3 hitting empty space.
2. **`dodger_eats_damage_when_no_warning`** — Same setup with `cast { duration: 0t }` (instant). Dodger has no opportunity to react. Takes full 25 damage. Confirms dodge behaviour is causally tied to the threat-zone window, not coincidence.
3. **`firebolt_interrupted_by_damage_during_cast`** — Caster initiates Firebolt at tick T. At tick T+1, Dodger's counter-attack lands a `Damaged` event on Caster. Cast interrupted, no damage at tick T+3, `CastInterrupted` chronicle event present, cooldown still consumed (default `@ cast`).
4. **`recipe_craft_interrupted_by_damage`** — Smith begins `cast_recipe ironsword` with 5t duration at tick T. At tick T+2, Dodger attacks Smith. Recipe interrupted: no output spawned, smith's busy state cleared, `CastInterrupted` event present. Proves the new interrupt layer applies uniformly across busy sources (recipes inherit the same machinery).

## Phasing — five PRs

| # | Phase | Deliverable |
|---|---|---|
| **G1** | Grammar + AST + parser | New `cast { duration, telegraph, interrupts }` block, `cooldown @ phase` qualifier, `interrupts: standard | { … } | none` + set ops, `set standard = { … }` declaration. AST extension `AbilityDef::program: Vec<AbilityProgramStep>` (where step is `Cast(CastSpec)` or `Effect(EffectBlock)`). Backwards compat: ability with no `cast` block → single-step `program = [Effect(...)]`. |
| **G2** | Busy-family extension + CastBegin EffectOp + busy-resolution kernel | New SoA columns `BusyWithAbilityId / BusyStartedAtTick / BusyTargetSlot / BusyTargetPos`, schema hash bump, new `EffectOp::CastBegin` (variant 46), compiler-emitted `physics_BusyTick` resolution kernel. Existing `TravelTo` + `Recipe` consumers updated to also write the new context fields (additive — existing fixtures' pins still pass). New chronicle events `CastBegan` / `CastResolved` / `CastInterrupted` (replayable). Interrupt detection per source (damage, stun, caster_died, target_died, movement). |
| **G3** | Threats view + scoring primitives | New `view threats` materialised cross-rule. New scoring grammar primitives (`threats.*`, `busy.*`). Sorted reduction per P11. Cross-backend parity test for the threats fold. |
| **G4** | firebolt_probe runtime + MVP behavioural pins | New `crates/firebolt_probe_runtime/`, `assets/sim/firebolt_probe.sim`, `assets/ability_test/firebolt_probe/{Firebolt,ForgeIronsword}.ability`. Four behavioural pins per the MVP spec. |
| **G5** | Schema hash regen + cross-backend parity tests + per-fixture migration audit | Regenerate `crates/engine/.schema_hash`. Add `tests/parity_cast_lifecycle.rs` and `tests/parity_threats_view.rs`. Audit all existing fixtures using `travel_to` or `cast_recipe`: every pin still passes (the busy-context-field additions are write-only, existing reads are unaffected). |

G1 unblocks G2; G2+G3 unblock G4; G5 lands last as the global verification slice.

Suggested merge order: G1 → G2 → G3 → G4 → G5 (sequential).

## Critical files

- `crates/dsl_ast/src/ability_parser.rs` — parse `cast` block, `cooldown @ phase`, `interrupts:` syntax, `set standard = { … }`.
- `crates/dsl_ast/src/ast.rs` — `AbilityDef::program: Vec<AbilityProgramStep>`, `CastSpec`, `InterruptSet`.
- `crates/dsl_compiler/src/cg/lower/ability.rs` — lower the new program shape; instant abilities (no cast steps) take the existing immediate path.
- `crates/dsl_compiler/src/cg/lower/scoring.rs` — lower new spatial primitives.
- `crates/dsl_compiler/src/cg/lower/views.rs` — register `threats` materialised view.
- `crates/dsl_compiler/src/cg/data_handle.rs` — add `BusyWithAbilityId`, `BusyStartedAtTick`, `BusyTargetSlot`, `BusyTargetPos` SoA enum variants alongside existing `BusyUntilTick`.
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — emit `physics_BusyTick` resolution kernel + interrupt-detection rules; update existing TravelTo/Recipe consumer emit to write the new context fields.
- `crates/engine/src/state/agent.rs` — new SoA columns.
- `crates/engine/.schema_hash` — regen baseline (P2).
- `crates/engine/src/event.rs` — new event variants `CastBegan` / `CastResolved` / `CastInterrupted`.
- `crates/engine/src/ability/program.rs` — add `EffectOp::CastBegin` variant 46.
- `crates/engine/src/cascade/views.rs` — register `threats` view in cross-rule registry.
- `crates/firebolt_probe_runtime/{Cargo.toml, src/lib.rs, src/bin/firebolt_probe_app.rs}` — new MVP fixture.
- `assets/sim/firebolt_probe.sim` — fixture .sim with dodger scoring rows.
- `assets/ability_test/firebolt_probe/{Firebolt,ForgeIronsword}.ability` — MVP abilities.
- `tests/parity_cast_lifecycle.rs`, `tests/parity_threats_view.rs` — cross-backend parity tests.
- `tests/proptest_baseline.rs` — extend with cast-state edge cases.

## Out of scope (explicitly)

- **Plan E (emit refactor for runtime authoring)** — G adds per-ability emitted kernels for the new lifecycle/interrupt rules + extends the existing per-source consumer kernels for TravelTo/Recipe. That's accepted debt; Plan E migrates the per-ability pattern to a generic data-driven dispatcher later.
- **Plan V (visualization)** — viewer reads cast state + threats from sim but doesn't render telegraphs / dodge feedback yet.
- **Multi-target casts** — each cast targets one slot/position; future extension.
- **Partial interrupts** — interrupt either fully clears the cast or doesn't. No "stagger" / "delay by N ticks" mechanics.
- **Cooldown @ resolve / @ interrupt** — syntax parsed and stored in metadata, only `@ cast` implemented in G; future slice.
- **Telegraph display rendering** — `telegraph: line(...)` field is parsed and stored in ability metadata for V to read; no visual production in G.
- **Player-controlled casts** — Firebolt's caster is AI-driven via existing scoring rows.
- **Cast time scaling (haste/buff modifiers)** — fixed per ability for now.
- **Per-busy-source telegraph rules** — recipes / travel get NO threat projection by default (they're not threatening anyone). Telegraph only fires for abilities with explicit `telegraph:` field. Future could add `cast_recipe`-as-construction-zone visualization but not in G.

## Cross-cutting concerns

- **Per-ability vs shared interrupts**: each ability declares its `interrupts:` set in its `cast` block. The `set standard = { … }` declaration is per-engine, declared once at the engine baseline so every fixture inherits.
- **Existing fixture migration**: existing `travel_to` / `cast_recipe` consumer rules are updated by the compiler to write `busy_with_ability_id` etc. alongside existing busy_until_tick writes. This is purely additive — every existing read is unaffected. G5's regression audit confirms.
- **Replay determinism**: cast lifecycle events are `replayable: true`. A snapshot taken mid-cast restores the busy SoA correctly; replay produces identical interrupt timing.
- **Cross-backend parity**: cast lifecycle deterministic, threats view deterministic via P11 sorted reduction. Tests added.
- **Performance**: busy-family extensions add ~20 bytes per agent (4 fields × 4-16 bytes). At 200k-agent stress fixtures, ~4 MB additional VRAM. Threats view is sized to `max_active_threats` (a per-fixture cap; defaults to `agent_cap / 4`). Per-tick busy-resolution kernel cost is O(N_busy), bounded by `max_active_threats`.
- **Backwards compatibility**: every existing `.ability` without a `cast` block compiles to single-step `[Effect(...)]` running the existing immediate-resolve path. No existing pin shifts.

## Verification end-to-end

After all five phases land:

1. `cargo build --release` — clean across the workspace.
2. `cargo test -p dsl_ast --release` — new grammar parses; `cast` block, `interrupts:` set ops, `cooldown @ phase` qualifier all round-trip.
3. `cargo test -p dsl_compiler --release` — new lowering tests pass; cast-time abilities emit busy writes; existing TravelTo/Recipe consumers still emit; instant abilities still emit immediate path.
4. `cargo test -p engine --test schema_hash --release` — new baseline matches.
5. `cargo test -p engine --test parity_cast_lifecycle --release` — byte-equal busy SoA + chronicle across backends, both for Firebolt cast and recipe craft.
6. `cargo test -p engine --test parity_threats_view --release` — byte-equal threats view across backends and runs.
7. `cargo test -p engine --test proptest_baseline --release` — cast-state edge cases never panic.
8. `cargo test -p firebolt_probe_runtime --release` — four MVP behavioural pins pass.
9. `cargo test --workspace --release` — all existing fixtures' behavioural pins still pass (regression check; busy-context-field additions are write-only, existing reads unaffected).
10. Manual: `cargo run -p firebolt_probe_runtime --bin firebolt_probe_app --release` produces deterministic NDJSON showing cast lifecycle events + dodger movement + recipe interrupt.

## Why G before E (and why V last)

This plan accepts emit-pattern debt: cast lifecycle and interrupt-detection rules are emitted per-ability in the existing style. After G ships, **Plan E refactors those emitted kernels into generic data-driven dispatchers** (one busy-resolution kernel parameterised by ability metadata rather than N per-ability lifecycle kernels). G first lets us design the gameplay primitives concretely; E then knows what kernel-class generics need to support.

**Plan V (visualization) sits on top of G**: viewer reads busy SoA + threats view + chronicle to render telegraphs / dodge feedback / cast progress bars / threat-zone overlays. With G in place, V is mostly viewer-side rendering work.

Order locked in 2026-05-09: **G → E → V**.

## Why this generalizes Lift A and Lift B

The realignment from a parallel `cast_*` SoA family to extending `Busy*` is the design lesson of this round. Lift A introduced `BusyUntilTick` as a "universal busy gate" and the comment in `data_handle.rs` explicitly anticipated multiple sources sharing it. Plan G honours that intent: cast-time abilities are just another busy source. Travel and recipes get interrupt mechanics + threat-zone potential for free; new cast-time abilities slot into the same machinery without parallel infrastructure.

This is also the reason Plan G is smaller than the first draft: most of the foundation already exists. Plan G adds the missing context fields, the interrupt layer, the threat view, and the AI consumption primitives — all on top of infrastructure Lift A established.
