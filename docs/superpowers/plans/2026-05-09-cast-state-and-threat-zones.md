# Cast State + Threat Zones + AI Consumption (Plan G)

> Goal: introduce cast-time abilities, threat zones AI can query, and the interrupt machinery that ties them together — as **DSL primitives**, not hand-rolled engine code. The compiler emits the machinery from `cast { … } effect { … }` blocks in `.ability` files; designer authors intent, gets the gameplay loop. This is the gameplay layer that Plan V (visualization) and Plan E (emit refactor for runtime authoring) will sit on top of, in that order.

## Goal

Today every ability resolves in the same tick it fires. There's no charging, no telegraphing, no AI-readable threat zone — `apply_ability` writes effect events directly. That makes whole categories of game-feel and AI behaviour unauthorable:

- Channeled / charged abilities (Firebolt, Bind Soul, ChargedShot)
- Telegraph → react gameplay (the player or AI sees the windup, has a window to dodge)
- Interruptible casts (a hero can interrupt the boss's wind-up by hitting hard before it lands)
- AI dodging (move out of an AOE warning before resolution)
- Multi-stage abilities (windup → impact → recovery, each with its own duration / interrupts)

This plan adds the engine + DSL primitives that make all of these authorable. Per the design conversation 2026-05-09: telegraphs and threat zones are gameplay concepts, not viewer concerns — the viewer (Plan V) will read them, but the sim is where they live.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/engine/src/ability/program.rs::EffectOp` — current EffectOp set; informs cast-program lowering shape.
  - `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::lower_apply_ability` — the call-site that today writes effect events directly; deferred-resolution path branches off here.
  - `crates/dsl_ast/src/ability_parser.rs` — current `.ability` parser; needs new `cast { … }` block grammar + `interrupts:` set syntax.
  - `crates/dsl_ast/src/ast.rs::AbilityDef` — needs `program: Vec<AbilityProgramStep>` (where each step is `Cast(CastSpec)` or `Effect(EffectBlock)`) instead of the current flat `effects` list.
  - `crates/engine/src/state/agent.rs` — agent SoA; needs new cast-state columns.
  - `crates/dsl_compiler/src/cg/lower/scoring.rs` — current scoring lowering; new spatial primitives (`threats.*`) plug in here.
  - `crates/engine/src/cascade/views.rs` — current materialised-view registry; the new `threats` view registers here.
  - Per-fixture runtimes (e.g. `crates/duel_25v25_runtime/`) — examples of the existing `apply_ability` flow; one new fixture (`firebolt_probe_runtime`) covers the MVP.

  Search method: `rg`, `grep -nE`, direct `Read`.

- **Decision:** new — adds engine + DSL primitives for cast lifecycle, threat zones, and AI threat-awareness. All implementation via the compiler emit pattern (P1); no hand-written rule logic in `engine/src/handlers/`.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: every `.ability` file that wants cast-time behaviour gets a `cast { … }` block; existing `.ability` files (no `cast` block) keep the current instant-resolve semantics. New fixture `assets/sim/firebolt_probe.sim` + `assets/ability_test/firebolt_probe/Firebolt.ability` author the MVP.
  - Generated outputs re-emitted: every per-runtime `OUT_DIR/generated.rs` gets the new ability-program shape, cast-state SoA accessors, and (for fixtures using `threats.*`) the threats-view fold + spatial primitives. Per-runtime build.rs auto-regenerates.

- **Hand-written downstream code:** NONE. All new engine machinery is compiler-emitted. The new `firebolt_probe_runtime` crate's `lib.rs` follows the existing per-fixture pattern (compile `.sim`, dispatch the emitted kernels).

- **Constitution check:**
  - **P1 (Compiler-First Engine Extension):** PASS. Cast lifecycle, threat view materialisation, interrupt detection, scoring primitives — all emitted from the new DSL grammar by `dsl_compiler`. No `impl Rule` in `crates/engine/src/handlers/`.
  - **P2 (Schema-Hash on Layout):** REQUIRES BUMP. Adds per-agent SoA columns: `cast_ability_id: u32`, `cast_started_at_tick: u32`, `cast_target_slot: u32`, `cast_target_pos: Vec3Padded`, `cast_program_step: u32`. Adds new chronicle event variants: `CastStarted`, `CastResolved`, `CastInterrupted`. `crates/engine/.schema_hash` regenerates per documented procedure.
  - **P3 (Cross-Backend Parity):** PASS. Cast lifecycle is emitted from the same compiler path that already runs cross-backend. Threats view fold + interrupt detection are deterministic operations the existing emit pattern handles. New `tests/parity_cast_lifecycle.rs` exercises a Firebolt cast on both backends and asserts byte-equal cast-state SoA + chronicle output.
  - **P4 (`EffectOp` Size Budget):** N/A. No EffectOp variant size changes. Cast metadata lives on the registry and SoA, not on EffectOp.
  - **P5 (Determinism via Keyed PCG):** PASS. No new RNG draws on the deterministic path. Threats view reads cast SoA; cast lifecycle is event-driven from `apply_ability` calls (which already use keyed PCG for target selection).
  - **P6 (Events Are the Mutation Channel):** PASS. Cast SoA mutations come from `CastStarted` / `CastResolved` / `CastInterrupted` events. Direct field writes confined to `step::*` and `snapshot::*` per existing convention.
  - **P7 (Replayability Flagged):** PASS. New events declared `replayable: true` (cast lifecycle is part of the deterministic record; replays must reproduce the same casts in the same ticks). Telegraph display (Plan V) reads from the replayable event stream.
  - **P8 (AIS Required):** PASS — this section satisfies it.
  - **P9 (Tasks Close With Verified Commit):** PASS — every task closes with a SHA on the active branch; UDA enforced by project-DAG skill.
  - **P10 (No Runtime Panic):** PASS. Cast resolution and interrupt detection use saturating ops + `Result` bubbling. No `.unwrap()` on the deterministic path. New tests added to `tests/proptest_baseline.rs` exercising cast-state SoA edge cases (cast at tick 0, interrupt at exactly resolve-tick, etc.).
  - **P11 (Reduction Determinism):** PASS WITH SORT. The `threats` view fold over agents-with-active-cast must use `sort_by((ability_id, caster_slot))` before reducing into the view, so the fold is bit-exact across runs and backends. New `tests/parity_threats_view.rs` asserts byte-equal threats-view output across backends and across runs at the same seed.

- **Runtime gate:** every phase has a runtime test that actually exercises the new code path and asserts an observable post-condition. Compile-clean is not runtime-clean.
  - **G1**: `dsl_ast::tests::cast_block_parses` — load a `.ability` with a `cast { duration: 3t; telegraph: line(...); interrupts: standard }` block, assert AST has `AbilityProgramStep::Cast(CastSpec { duration_ticks: 3, ... })`.
  - **G2**: `dsl_compiler::tests::cast_lowering_emits_deferred_resolution` — lower a cast-time ability, assert emitted WGSL contains the deferred-resolution kernel and cast-state writes; instant-resolve abilities (no `cast` block) keep emitting the existing immediate path.
  - **G3**: `firebolt_probe_runtime::tests::cast_state_advances_per_tick` — drive 5 ticks of a Firebolt cast, assert per-tick: `cast_ability_id != 0`, `cast_started_at_tick` stable, `current_tick - cast_started_at_tick` ≤ duration. At resolve tick, assert effect event written + cast state cleared.
  - **G4**: `firebolt_probe_runtime::tests::dodger_avoids_firebolt_when_warned` — caster + dodger, Firebolt with 3t cast, dodger scores threat-aware Move-Perpendicular, dodger leaves line before resolution, no damage taken.
  - **G4 (inverse pin)**: `firebolt_probe_runtime::tests::dodger_eats_damage_when_no_warning` — same setup but `cast { duration: 0t }` (instant), dodger has no time to react, takes damage. Confirms the dodge behaviour is causally tied to the threat-zone window, not coincidence.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).

## Grammar additions

Decisions locked in 2026-05-09:

### Cast as ability program block

Multiple `cast { … } effect { … }` siblings under an ability, applied in order:

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

Multi-stage example (Channelbeam — three stacking damage hits over time):

```text
ability Channelbeam {
    target: enemy
    range: 20

    cast { duration: 1t; interrupts: standard }
    effect { damage 5 [LIGHTNING: 100] }

    cast { duration: 1t; interrupts: standard }
    effect { damage 8 [LIGHTNING: 100] }

    cast { duration: 1t; interrupts: standard }
    effect { damage 15 [LIGHTNING: 100] }
}
```

Backwards compatible: an ability with no `cast` block compiles to the existing immediate-resolve path (current behaviour preserved for every existing fixture).

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

`standard = { damage, stun, caster_died, target_died }` per design 2026-05-09. `target_died` is a no-op for AOE casts (no single target).

### New scoring primitives

```text
threats.in_zone(self)                  # bool — is `self` inside any active threat zone?
threats.intensity_at(pos)              # f32 — sum of intensities of overlapping threat zones at pos
threats.nearest(self)                  # ThreatHandle — the closest active threat zone
threats.dir_away_from_nearest(self)    # vec3 — unit vector pointing away from nearest threat
cast.is_charging(self)                 # bool — is `self` currently charging an ability?
cast.target_of_charging(self)          # AgentHandle — what's `self` charging at?
cast.ability_charging(self)            # AbilityId — which ability `self` is charging
cast.ticks_until_resolve(self)         # u32 — how long until `self`'s current cast resolves
```

These compose with existing scoring grammar (`where ...`, `score K`, etc.).

## Engine extensions (compiler-emitted)

### Cast SoA columns

Per-agent state:

| Column | Type | Meaning |
|---|---|---|
| `cast_ability_id` | `u32` | `0` = idle. Non-zero = currently charging this ability. |
| `cast_started_at_tick` | `u32` | Tick the cast began. |
| `cast_target_slot` | `u32` | `u32::MAX` = no single target (AOE / self-cast). |
| `cast_target_pos` | `Vec3Padded` | Target world position (snapshot at cast start). |
| `cast_program_step` | `u32` | Index of the current `cast`/`effect` step in the program. |

Schema hash bumps per P2.

### Cast lifecycle kernel

Per-tick kernel `physics_CastTick @phase(per_agent)`:
1. For each agent with `cast_ability_id != 0`:
2. Check interrupt conditions (per ability program's `interrupts` set + per-tick events). If interrupted: emit `CastInterrupted`, clear cast SoA.
3. Else check elapsed: if `current_tick - cast_started_at_tick >= step.duration_ticks`: advance `cast_program_step`. If next step is an `effect`, write its EffectOps to chronicle (existing emit pipeline), then advance again. If we're past the last step, emit `CastResolved`, clear cast SoA.
4. Else continue (no-op).

Compiler-emitted from the new ability-program AST shape.

### Threats materialised view

`view threats` registered cross-rule:
- Per-tick fold over agents-with-active-cast (`cast_ability_id != 0`).
- For each, look up the ability's current step in the metadata table; if it's a `cast` step with telegraph data, project the telegraph shape into world space → produce a `ThreatZone { origin, shape, intensity, expires_at_tick }` row.
- Reduction sorted by `(cast_ability_id, caster_slot)` per P11.

Spatial query primitives lower to indexed reads against this view.

### Interrupt detection

Compiler emits per-interrupt-source detection rules from the ability's `interrupts` set:
- `damage`: scan `Damaged` events this tick; if target is an agent with active cast that includes `damage` in its interrupts, mark for interruption.
- `stun`: scan `EffectStunApplied` events same way.
- `caster_died`: scan `Defeated` events; if defeated agent had active cast, interrupt.
- `target_died`: scan `Defeated` events; if defeated agent matches `cast_target_slot` of an active cast that includes `target_died`, interrupt.
- `movement`: comparison of pre-tick / post-tick `agent_pos` (only for casts whose interrupts include `movement`).

These all emit as `CastInterrupted` events that the lifecycle kernel consumes next tick (or same-tick @ phase boundary).

## AI consumption pattern

Per-fixture `.sim` files use the new scoring primitives:

```text
# In firebolt_probe.sim
scoring DodgerBehaviour @phase(scoring) {
    # Strong negative score for staying in a threat zone — drives
    # the dodger to move out of it.
    score -200.0 where (self.alive
                       && threats.in_zone(self))

    # Move action: away from nearest threat.
    score 100.0 where (self.alive
                      && threats.in_zone(self))
                action move(threats.dir_away_from_nearest(self))
}
```

Same primitives compose with existing scoring grammar — designer just drops `threats.in_zone(self)` into a `where` clause and the AI gets threat-aware behaviour.

## MVP fixture: firebolt_probe

Two-agent fixture demonstrating the full G stack end-to-end:

- **Caster** (slot 0): every 5 ticks, casts Firebolt at Dodger.
- **Dodger** (slot 1): scoring rows above. When in threat zone, moves perpendicular to the line.

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
1. **`dodger_avoids_firebolt_when_warned`** — Caster initiates Firebolt at tick T. Threats view materialises with the line zone. Dodger detects `threats.in_zone(self)`, scores high on perpendicular move, leaves the line by tick T+3. Damage taken = 0. Cast resolves at tick T+3 hitting empty space (no targets in line).
2. **`dodger_eats_damage_when_no_warning`** — Same setup with `cast { duration: 0t }` (instant). Dodger has no opportunity to react. Takes full 25 damage.
3. **`firebolt_interrupted_by_damage_during_cast`** — Caster initiates Firebolt at tick T. At tick T+1, the Dodger fires a counter-attack that lands a `Damaged` event on the Caster. Cast interrupted, no damage at tick T+3, `CastInterrupted` chronicle event present.

## Phasing — five PRs

| # | Phase | Deliverable |
|---|---|---|
| **G1** | Grammar + AST + parser | New `cast { duration, telegraph, interrupts }` block, `cooldown @ phase` qualifier, `interrupts: standard | { … } | none + set ops`, `set standard = { … }` declaration. AST extension `AbilityDef::program: Vec<AbilityProgramStep>` (where step is `Cast(CastSpec)` or `Effect(EffectBlock)`). Backwards compat: ability with no `cast` block → single-step `program = [Effect(...)]`. |
| **G2** | Cast SoA + lifecycle kernel | New per-agent SoA columns, schema hash bump, compiler-emitted `physics_CastTick` kernel that advances/resolves/clears casts based on the program. New chronicle events `CastStarted` / `CastResolved` / `CastInterrupted` (replayable). Interrupt detection per-source (damage, stun, caster_died, target_died, movement). |
| **G3** | Threats view + scoring primitives | New `view threats` materialised cross-rule. New scoring grammar primitives (`threats.in_zone(self)`, `threats.intensity_at(pos)`, etc.). Sorted reduction per P11. Cross-backend parity test for the threats fold. |
| **G4** | firebolt_probe runtime + MVP behavioural pins | New `crates/firebolt_probe_runtime/`, `assets/sim/firebolt_probe.sim`, `assets/ability_test/firebolt_probe/Firebolt.ability`. Three behavioural pins per the MVP spec above. |
| **G5** | Schema hash regen + cross-backend parity tests + per-fixture migration audit | Regenerate `crates/engine/.schema_hash`. Add `tests/parity_cast_lifecycle.rs` and `tests/parity_threats_view.rs`. Audit all existing fixtures: any `.ability` file without a `cast` block continues to work; any fixture's behavioural pins still pass (regression check). |

G1 unblocks G2; G2+G3 unblock G4; G5 lands last as the global verification slice.

Suggested merge order: G1 → G2 → G3 → G4 → G5 (sequential, each builds on the previous).

## Critical files

- `crates/dsl_ast/src/ability_parser.rs` — parse `cast` block, `cooldown @ phase`, `interrupts:` syntax, `set standard = { … }`.
- `crates/dsl_ast/src/ast.rs` — `AbilityDef::program: Vec<AbilityProgramStep>`, `CastSpec`, `InterruptSet`.
- `crates/dsl_compiler/src/cg/lower/ability.rs` — lower the new program shape; instant abilities (no cast steps) take the existing immediate path.
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` — emit `physics_CastTick` lifecycle kernel + interrupt-detection rules.
- `crates/dsl_compiler/src/cg/lower/scoring.rs` — lower new spatial primitives.
- `crates/dsl_compiler/src/cg/lower/views.rs` — register `threats` materialised view.
- `crates/engine/src/state/agent.rs` — new SoA columns.
- `crates/engine/.schema_hash` — regen baseline (P2).
- `crates/engine/src/event.rs` — new event variants `CastStarted` / `CastResolved` / `CastInterrupted`.
- `crates/engine/src/cascade/views.rs` — register `threats` view in cross-rule registry.
- `crates/firebolt_probe_runtime/{Cargo.toml, src/lib.rs, src/bin/firebolt_probe_app.rs}` — new MVP fixture.
- `assets/sim/firebolt_probe.sim` — fixture .sim with dodger scoring rows.
- `assets/ability_test/firebolt_probe/Firebolt.ability` — MVP ability with cast block.
- `tests/parity_cast_lifecycle.rs`, `tests/parity_threats_view.rs` — cross-backend parity tests.
- `tests/proptest_baseline.rs` — extend with cast-state edge cases.

## Out of scope (explicitly)

- **Plan E (emit refactor for runtime authoring)** — G adds per-ability emitted kernels for the new lifecycle/interrupt rules. That's the "accepted debt" from the design conversation; Plan E migrates these to a generic data-driven dispatcher later.
- **Plan V (visualization)** — viewer reads `cast_*` SoA + `threats` view + chronicle but doesn't render telegraphs/dodge feedback yet. That's V's job.
- **Multi-target casts** — each cast targets one slot/position. A cast that wants to hit N pre-selected targets needs to be modelled as N parallel casts or as an AOE on a position; explicit list-of-targets at cast start is out of scope.
- **Partial interrupts** — interrupt either fully clears the cast or doesn't. No "stagger" / "stun while casting" / "delay by N ticks" mechanics. (Could be a future `interrupts: partial { delay_by: 2t }` extension.)
- **Cooldown @ resolve / @ interrupt** — the syntax is parsed and stored in metadata, but only `@ cast` is implemented in G. The other two phase qualifiers compile but error at lowering with a "not yet supported" message; future slice.
- **Telegraph display rendering** — the `telegraph: line(...)` field is parsed and stored in ability metadata for V to read, but no visual production happens in G.
- **Player-controlled casts** — Firebolt's caster is AI-driven via existing scoring rows. Player input → cast is a different design (input event → write `CastStarted` from the host).
- **Cast time scaling** — cast_time is fixed per ability. Cast-time-reducing buffs / haste effects are out of scope; future would add a `cast_time_multiplier` SoA field consulted at lifecycle-tick.

## Cross-cutting concerns

- **Per-ability vs shared interrupts**: each ability declares its `interrupts:` set in its `cast` block. The `set standard = { … }` declaration is global (per-fixture or per-engine — design says per-engine, declared once at the engine baseline so every fixture inherits). Designer can override `standard` per-fixture by re-declaring, but there's only one `standard` in scope at any time.
- **Replay determinism**: cast lifecycle events are `replayable: true`. A snapshot taken mid-cast restores cast SoA correctly + replay produces identical interrupt timing.
- **Cross-backend parity**: cast lifecycle deterministic, threats view deterministic via P11 sorted reduction. Tests added.
- **Performance**: cast SoA adds ~24 bytes per agent (5 fields × 4-16 bytes). At 200k-agent stress fixtures, ~5 MB additional VRAM. Threats view is sized to `max_active_casts` (a per-fixture cap; defaults to `agent_cap / 4`). Per-tick lifecycle kernel cost is O(N_casting), bounded by `max_active_casts`.
- **Backwards compatibility**: every existing `.ability` without a `cast` block compiles to a one-step program `[Effect(...)]` that runs the existing immediate-resolve path. No existing fixture's behavioural pin should shift.
- **Migration of existing fixtures**: G ships with NO existing fixture using `cast` blocks. G5's audit confirms every existing pin still passes. Designers opt fixtures into cast-time mechanics in subsequent PRs as gameplay calls for it.

## Verification end-to-end

After all five phases land:

1. `cargo build --release` — clean across the workspace.
2. `cargo test -p dsl_ast --release` — new grammar parses; `cast` block, `interrupts:` set ops, `cooldown @ phase` qualifier all round-trip.
3. `cargo test -p dsl_compiler --release` — new lowering tests pass; cast-time abilities emit deferred-resolution kernel; instant abilities keep emitting immediate path.
4. `cargo test -p engine --test schema_hash --release` — new baseline matches.
5. `cargo test -p engine --test parity_cast_lifecycle --release` — byte-equal cast SoA + chronicle across backends.
6. `cargo test -p engine --test parity_threats_view --release` — byte-equal threats view across backends and across runs.
7. `cargo test -p engine --test proptest_baseline --release` — cast-state edge cases never panic.
8. `cargo test -p firebolt_probe_runtime --release` — three MVP behavioural pins pass.
9. `cargo test --workspace --release` — all existing fixtures' behavioural pins still pass (regression check; no `cast`-less ability behaviour shifts).
10. Manual: `cargo run -p firebolt_probe_runtime --bin firebolt_probe_app --release` produces deterministic NDJSON showing cast lifecycle events + dodger movement.

## Why G before E (and why V last)

This plan accepts emit-pattern debt: cast lifecycle and interrupt-detection rules are emitted per-ability in the existing style. After G ships, **Plan E refactors those emitted kernels into generic data-driven dispatchers** (one cast-lifecycle kernel parameterised by ability metadata rather than N per-ability lifecycle kernels). G first lets us design the gameplay primitives concretely; E then knows what kernel-class generics need to support because we've built G's specific kernels first. Building generics speculatively before concrete uses is the recipe for over-abstraction.

**Plan V (visualization) sits on top of G**: viewer reads cast SoA + threats view + chronicle to render telegraphs / dodge feedback / cast progress bars / threat-zone overlays. With G in place, V is mostly viewer-side rendering work — most of the data exists in sim state.

Order locked in 2026-05-09: **G → E → V**.
