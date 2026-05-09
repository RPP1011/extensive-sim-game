# Wave Defense Stress Fixture — Functionality Stress Test (Foundation Slice)

> Goal: a wave-defense fixture where a settlement around a resource node is overwhelmed by infinitely-ramping monster waves. **Score = resource accumulated before settlement falls.** Every run ends in death; the score is the engine's perf signature.

## Goal

Build `crates/wave_defense_runtime/` — a fixture that exercises code paths the existing stress sims don't:
- **Heterogeneous agent types** in one tick (4 types: node + settler + monster + spawner)
- **Long-range pathing** (Lift A `travel_to` from map edge → settlement)
- **Mixed dispatch** (settlers single-target damage, monsters AOE-Spread, harvesters per-tick increment)
- **Bursty `summon`** (irregular wave spikes, not steady-state)
- **Spatial non-uniformity** (dense center cluster + moving wavefront from edges)
- **Until-death behavioral pin** (real game termination, not just "tick advanced")

This is foundation-only: travel + summon + damage + harvest. Lifts B/C/D verbs (recipes, skills, consent) deferred to follow-up slices when the core stresses are characterized.

## Score model

```
score = resource_yielded at termination
termination = no settlers alive for 10 consecutive ticks
```

Monsters ramp infinitely:
```
wave_size(t) = base_wave_size + (t / wave_period) * wave_growth
```

Settlers can't "win". The fixture is a deterministic benchmark: same seed → same death tick → same score. Any engine change (fusion improvement, sort speedup, AOE cap raise) shows up as a different score.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `crates/spy_network_runtime/` — closest analog of multi-faction multi-verb sim with `creature_type` discriminant.
  - `crates/village_economy_runtime/` — closest analog of long-running fixture with score-like metrics + lift verbs.
  - `crates/stress_cast_density_runtime/` — pattern for AOE Spread + per-tick metric NDJSON.
  - `crates/stress_agent_count_runtime/` — pattern for stand-alone driver bin + catch_unwind.
  - `EffectOp::TravelTo` (Lift A, variant 39) — `travel_to <x> <y> for <duration>` DSL surface; busy_until_tick gate; per-fixture consumer rule writes the destination.
  - `EffectOp::Summon` (variant 24) — multi-spawn allocation just shipped in PR #47 (`apply_summon_event_to_state`).
  - `EffectOp::Damage` (variant 0) + AOE `area: spread { count: N }` (Spread shape, ShapeKind=4).

  Search method: `rg`.

- **Decision:** new `crates/wave_defense_runtime/` crate mirroring `village_economy_runtime`'s shape (long-running sim with bounded score). Compile-from-`.sim` like every other runtime; opts in to AOE Path B (`LowerOpts.aoe_dispatch=true`) like duel_25v25. No new EffectOp variants — uses what's already shipped.

- **Rule-compiler touchpoints:**
  - DSL inputs added: `assets/sim/wave_defense.sim`, `assets/ability_test/wave_defense/{Harvest,Strike,MonsterAdvance,MonsterCleave,SpawnWave}.ability`.
  - Generated outputs emitted: `crates/wave_defense_runtime/build.rs → OUT_DIR/*.{wgsl,rs}` — fresh per-fixture build artifacts.

- **Hand-written downstream code:** NONE. Same compile-from-`.sim` pattern as every other runtime.

- **Constitution check:**
  - P1 (Compiler-First): PASS — every kernel emit through dsl_compiler; no hand-written WGSL.
  - P2 (Schema-Hash): N/A — no SoA / event / mask-predicate changes; engine `.schema_hash` should not move.
  - P3 (Cross-Backend Parity): N/A — fixture is GPU-only; cross-backend determinism in scope for a separate slice.
  - P4 (`EffectOp` Size Budget): N/A — no new variants.
  - P5 (Determinism via Keyed PCG): PASS — monster spawn positions seeded via `per_agent_u32_pcg_with_extra(seed, agent_id, tick, purpose, axis)`. Wave count/period are config constants. Same seed → same death tick → same score.
  - P6 (Events Are the Mutation Channel): PASS — all sim mutation through chronicle events.
  - P7 (Replayability Flagged): PASS — every event `@replayable @gpu_amenable`.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS.
  - P10 (No Runtime Panic): PASS — driver wraps `step()` in catch_unwind so OOM/panic surface as recorded breakpoint.
  - P11 (Reduction Determinism): PASS — uses existing AOE bitonic sort + AgentId tie-break (PR #39 convention).

- **Runtime gate:**
  - `crates/wave_defense_runtime/src/lib.rs::tests::settlement_falls_within_budget` — drive up to `max_ticks=2000` at base wave size; assert `result.died_at_tick < 2000` (settlement does eventually fall) AND `result.died_at_tick > 200` (settlement survives initial warmup). Asserts `result.score > 0` (harvest happened at least once before death).
  - `..::tests::same_seed_same_death_tick` — run twice with same seed, assert byte-identical death tick + score (P5 determinism).

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design.

---

## Sim design

### Agent types (creature_type discriminant)

| Type | Ord | Spawn | Behavior |
|---|---|---|---|
| Node | 1 | 1 at origin (0,0,0) | Doesn't move, doesn't cast. Has `resource_yielded` counter (re-purpose `mana` SoA — spy_network precedent). |
| Settler | 2 | 25 in tight ring around origin (radius ≤ 3) | Per-tick: harvest from node (increment its `mana` by 1) when in range. When in-range monster: single-target damage. |
| Monster | 3 | Spawned at map edges by spawner waves | Travel to nearest settler. When in melee range: AOE-Spread damage. |
| Spawner | 4 | 6 stationary agents at map face midpoints (±64,0,0), (0,±64,0), (0,0,±64) | Per `wave_period` ticks: cast `summon` to spawn N monsters (N grows over time). |

### Verbs

```
verb Harvest(self) =                     // Settler → Node
  action HarvestAction
  when self.alive
       && self.creature_type == settler_type
       && distance(self, target) < harvest_range
       && target.creature_type == node_type
  apply_ability 1 by self target target
  score 1.0

verb Strike(self) =                      // Settler → Monster (single-target damage)
  action StrikeAction
  when self.alive
       && self.creature_type == settler_type
       && target.alive
       && distance(self, target) < strike_range
       && target.creature_type == monster_type
  apply_ability 2 by self target target
  score (max_hp - target.hp)            // prefer wounded monsters

verb MonsterAdvance(self) =              // Monster → Settler (travel)
  action AdvanceAction
  when self.alive
       && self.creature_type == monster_type
       && target.alive
       && target.creature_type == settler_type
       && distance(self, target) > melee_range
       && self.busy_until_tick <= world.tick   // not currently traveling
  apply_ability 3 by self target target          // travel_to(target.pos, eta=10s)
  score 1000.0 - distance(self, target)         // prefer closest settler

verb MonsterCleave(self) =               // Monster → Settler (AOE damage)
  action CleaveAction
  when self.alive
       && self.creature_type == monster_type
       && target.alive
       && distance(self, target) < melee_range
       && target.creature_type == settler_type
  apply_ability 4 by self target target          // damage in spread(2.0, 8)
  score 1.0

verb SpawnWave(self) =                   // Spawner → self (summons N monsters)
  action SpawnAction
  when self.alive
       && self.creature_type == spawner_type
       && (world.tick % wave_period == 0)
  apply_ability 5 by self target self            // summon "monster" count=N
  score 1.0
```

### Per-tick wave size

The summon count grows with tick. Plumbing options:
- **A. Static `count` in the ability** + multiple SpawnWave casts to amplify (each spawner casts N times per wave). Doesn't scale ergonomically.
- **B. Wave size from a config uniform** that the host writes per tick. Cleanest; mirrors how `tick` is already plumbed via the cfg uniform.
- **C. Conditional `apply_ability` selection** based on tick — multiple Spawn abilities (Spawn1, Spawn2, ...). Combinatorial.

Pick **B** — host writes `cfg.wave_size = base + (tick / wave_period) * wave_growth` each tick; the ability reads it via a config field reference. If config-uniform reads aren't supported in `summon count` arg today, implement that as part of this slice (small DSL extension).

### Termination

- Host counts alive settlers each tick (`agents.alive[slot] && agents.creature_type[slot] == settler_type`).
- When count hits 0 for 10 consecutive ticks → terminate.
- Score = `agents.mana(node_slot)` at termination (the resource_yielded counter).
- Death-tick + score recorded in driver NDJSON summary line.

### Map dimensions

- World half-extent: 64 (matches existing fixture convention)
- Settlement radius: 3 around origin
- Spawner positions: 6 face midpoints at ±64
- Monster melee range: 1.5 (one cell)
- Monster cleave radius: 2.0 (`spread(2.0, 8)`)
- Settler strike range: 6.0 (one cell-side, settlers can pick off approaching monsters)
- Harvest range: 3.0 (settlers in tight ring around node)

## Tasks

| # | Task | Files | Description |
|---|---|---|---|
| 1 | Author the .sim + .ability files | `assets/sim/wave_defense.sim`, `assets/ability_test/wave_defense/*.ability` | 5 verbs + 5 abilities + 4 creature_types + termination metric. Mirrors spy_network's structure. |
| 2 | Build `wave_defense_runtime` crate | `crates/wave_defense_runtime/{Cargo.toml, build.rs, src/lib.rs}` | Mirrors village_economy + spy_network shape. Initializes 1 node + 25 settlers + 6 spawners + monster pool. Per-tick step + alive-count tracking + termination check. |
| 3 | Wave-size cfg-uniform plumbing | `crates/wave_defense_runtime/src/lib.rs` + ability parsing if needed | Host writes `wave_size = base + (tick / period) * growth` to the cfg uniform each tick. If `summon count: <cfg-ref>` isn't supported in the .ability surface, extend it (small DSL change). |
| 4 | Behavioral pins | `crates/wave_defense_runtime/src/lib.rs::tests` | `settlement_falls_within_budget` (drives up to max_ticks=2000) + `same_seed_same_death_tick` (P5 determinism). |
| 5 | CLI driver bin | `crates/wave_defense_runtime/src/bin/wave_defense_app.rs` | NDJSON per-tick: `{"tick": N, "alive_settlers": M, "alive_monsters": K, "score": S, "wave_size": W}`. Final summary: `{"summary": true, "died_at_tick": T, "score": S, "max_wave_size": W, "panic": null}`. |
| 6 | Readme + verification | `crates/wave_defense_runtime/README.md` (or doc comment in lib.rs) | How to invoke; what the score means; baseline numbers from a sample run. |

Tasks 1+2 are interleaved (the .sim shape drives the runtime's binding setup). Task 3 may surface a small DSL extension. Tasks 4+5 are independent after 1+2 land; 6 depends on 5.

Total: substantial slice (~1000-1500 LOC across all parts), single PR.

## Out of scope (deferred to follow-up slices)

- **Lift B `cast_recipe`** — settlers crafting weapons. Adds rich combat depth but adds 200+ LOC of inventory plumbing.
- **Lift C `propose` / `announce`** — settlers calling for reinforcements; spawners announcing wave incoming. Adds narrative but doesn't stress new code paths beyond what existing fixtures cover.
- **Lift D `gain_skill`** — defenders leveling up from kills. Most valuable next addition (changes per-tick combat math); ship after foundation is characterized.
- **Defensive structures (walls, towers)** — needs new entity types; defer until wave-defense base is shipping.
- **Resource respawn / harvest depletion** — node infinitely yields today; depletion is per-fixture decision when economy depth needed.
- **Multi-settlement maps** — one settlement at origin only; multi-settlement is a separate fixture.
- **Phase 2 DebugWgslFlags integration** — the in-flight `debug_probe_runtime` slice (#242) will demonstrate wiring; this fixture can opt in later if instrumentation reveals a hotspot worth attributing.

## Why this is the right next stress fixture

The current stress lineup is:
- `stress_agent_count` — 1 verb, 1 type, no behavior. Tests dispatcher overhead.
- `stress_cast_density` — 1 verb, 1 type, max-density AOE. Tests Spread sort + ring saturation.
- `village_economy` — 4 lift verbs, 1 type, real .sim. Tests verb composition.
- `spy_network` — 3 verbs, 3 types, real .sim. Tests multi-faction predicates.

What's missing: a sim that combines **multi-type heterogeneity + spatial dynamics + bursty load + meaningful termination**. Wave defense fills exactly that gap.

It also produces a **single comparable benchmark number** (score) that lets us track engine perf over time without per-tick analysis. Future engine work — codegen improvements, kernel splitting, ring fanout, sort optimizations — all show up as score deltas on this fixture.
