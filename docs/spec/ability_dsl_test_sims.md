# Ability DSL Test Sims — Acceptance Plan

> Companion to `docs/spec/ability_dsl_unified.md`. Defines the test-sim
> sequence that validates each implementation wave end-to-end. Same
> pattern as the 22 prior `*_real` sims that proved the verb cascade
> pipeline.

## Pattern

Each wave gets ONE canonical acceptance sim. The sim's job is to:

1. **Exercise the wave's new surfaces** in a recognizable gameplay shape
2. **Fail loudly today** — runtime / build error or skipped tests pre-wave
3. **Pass cleanly post-wave** with deterministic, replay-stable output
4. **Auto-render via `viz_app <name>`** so we can watch it play out

Test sims live in `crates/<name>_runtime/` + `crates/sim_app/src/<name>_app.rs`,
identical structure to the existing sims. Their `.ability` files live in
`assets/ability_test/<name>/*.ability`.

---

## Wave 1 acceptance: `duel_abilities`

**Validates:** parser + AbilityProgram lowering for EffectOps 0-7
(damage / heal / shield / stun / slow / transfer_gold / modify_standing /
cast).

**Composition:** 2 heroes (mirror of `duel_1v1` but driven by `.ability`
files instead of hand-authored `verb` declarations).

**Hero ability set** (in `assets/ability_test/duel_abilities/`):
```
ability Strike {
    target: enemy
    range: 5.0
    cooldown: 1s
    hint: damage

    damage 15
}

ability ShieldUp {
    target: self
    cooldown: 4s
    hint: defense

    shield 50
}

ability Mend {
    target: self
    cooldown: 3s
    hint: heal

    heal 25
}
```

**Acceptance criteria:**
- `.ability` files parse cleanly via the new `dsl_ast::ability_parser`
- Each ability becomes an `AbilityRegistry` entry with the right
  `EffectOp` variants and Gate fields (cooldown_ticks, range)
- The runtime drives ability dispatch through the existing chronicle
  pipeline (no new dispatch shape needed for variants 0-7)
- Per-tick ASCII viz shows hero HP swings as abilities fire
- Combat resolves to one hero down by tick ~200 (similar curve to
  current duel_1v1)
- `cargo run --bin viz_app -- duel_abilities` renders

**What this catches:** parser bugs, AST→IR lowering bugs, AbilityRegistry
serialization bugs, Gate field mapping bugs, EffectOp dispatch bugs.

**Substitute for the parser-only Wave 1.0?** No. Wave 1.0 lands the
parser; the Wave 1 acceptance sim needs the full lowering chain. So
`duel_abilities` waits for Waves 1.0 + 1.6 + 1.7 (parser + lowering +
registry wiring) before it can pass.

---

## Wave 2 acceptance: `tactical_squad_abilities`

**Validates:** combat depth — control verbs (root/silence/fear/taunt),
movement (dash/blink/knockback/pull), buffs/debuffs (buff/debuff/
damage_modify/lifesteal), advanced (execute/self_damage), tag system
widening, scaling terms, stacking modes, delivery methods.

**Composition:** 5v5 with role distribution (Tank/Healer/DPS), like
`tactical_squad_5v5` but each hero authored as a full `.ability` file
with multi-modifier effect statements.

**Sample ability:**
```
ability Whirlwind {
    target: self_aoe
    cooldown: 8s
    cast: 400ms
    hint: damage

    damage 40 in circle(2.5) [PHYSICAL: 50]
    damage 10 in circle(2.5) when hit_count_above(2) [PHYSICAL: 50]
}

ability HeroicCharge {
    target: enemy
    range: 5.0
    cooldown: 10s
    hint: crowd_control

    dash to_target
    damage 35 [PHYSICAL: 50]
    stun 1500ms [CROWD_CONTROL: 70]
}

ability Fireball {
    target: ground
    range: 8.0
    cooldown: 6s
    cast: 600ms
    hint: damage

    deliver projectile { speed: 12.0, width: 0.5 } {
        on_hit {
            damage 60 in circle(3.0) [FIRE: 80]
        }
    }
}
```

**Acceptance criteria:**
- Multi-modifier effect statements parse (`in circle(2.5) [FIRE: 80] when ... chance ... stacking ...`)
- Projectile delivery resolves at correct tick (speed × distance / 100ms)
- AoE shapes rasterize and select correct targets
- Status effects (root/silence/fear) gate subsequent verb mask predicates
  (rooted agent can't use movement verbs; silenced agent can't cast)
- Lifesteal verb generates per-hit heal events
- Stacking modes resolve correctly (refresh extends duration; stack
  accumulates; extend adds to existing)
- `cargo run --bin viz_app -- tactical_squad_abilities` renders status
  effects as glyph color changes

**What this catches:** modifier-slot parser bugs, shape rasterization
bugs, delivery state-machine bugs, status-effect gate predicate bugs,
stacking-handler bugs.

---

## Wave 3 acceptance: `spy_network`

**Validates:** AI-state writes/reads + BeliefState SoA + Theory of Mind
verbs (scry/reveal/stealth/disguise/decoy/plant_belief/erase_belief) +
standing extension (befriend/enmity/endear/slander) + engagement verbs
(force_engage/break_engagement/taunt) + condition atoms
(`believes(target, subject).<field> <op> <val>`).

**Composition:** 30 agents in 3 factions (5 spies, 5 nobles, 20 commoners)
across one shared world.

**Spy ability set:**
```
ability Disguise {
    target: self
    cooldown: 30s
    cast: 2s
    hint: utility

    disguise as commoner for 60s
}

ability PlantRumor {
    target: enemy
    range: 3.0
    cooldown: 5s
    cast: 1s
    hint: utility

    plant_belief target of subject {
        last_known_creature_type: dragon  // make subject look threatening
    }
}

ability Slander {
    target: enemy
    range: 5.0
    cooldown: 3s
    cast: 800ms
    hint: utility

    slander third_party by 15
}

ability VanishingAct {
    target: self
    cooldown: 60s
    cast: 0ms
    hint: utility

    stealth for 8s
}
```

**Noble ability set:** uses standing checks + engagement
```
ability ChallengeToDuel {
    target: enemy
    range: 8.0
    cooldown: 30s
    when target_standing_below(-50)

    duel_challenge
}

ability Decree {
    target: ally
    range: 12.0
    cooldown: 60s

    befriend 25
}
```

**Acceptance criteria:**
- BeliefState SoA wired and per-tick decay phase runs
- Spies can disguise — observers' beliefs of disguised spies show
  `last_known_creature_type: commoner` instead of true type
- `plant_belief` writes propagate to target's belief map for the named
  subject
- `slander` shifts third-party standing as observed by target
- Spies dying are detected via observer beliefs (confidence drops to 0
  when spy invisible AND not re-observed)
- Combat between nobles is gated on standing threshold — duels fire
  only after slander accumulates
- `cargo run --bin viz_app -- spy_network` renders agents colored by
  believed creature type (i.e. disguised spies look like commoners)

**What this catches:** BeliefState SoA layout bugs, decay phase ordering
bugs, belief-write event flow, condition atom lowering for `believes()`,
ModifyStanding three-party variant, AgentSel/SubjectSel resolution.

---

## Wave 4 acceptance: `siege_defense`

**Validates:** voxel ops + structures + materials + 3D shapes + CSG +
voxel-aware conditions.

**Composition:** 10 defenders inside a structure-placed castle vs 30
attackers using harvest/transform voxel ops.

**Defender ability set:**
```
ability BuildWall {
    target: ground
    range: 5.0
    cooldown: 15s
    cast: 2s
    hint: defense

    place_voxels stone in wall(8, 5, 1) damageable_hp 200 for 600s
}

ability Reinforce {
    target: ground
    range: 8.0
    cooldown: 8s
    cast: 1s
    hint: defense

    transform_voxels wood -> stone in box(3, 3, 3)
}
```

**Attacker ability set:**
```
ability Earthshatter {
    target: ground
    range: 8.0
    cooldown: 20s
    cast: 600ms
    hint: damage

    damage 80 in sphere(3.0) [PHYSICAL: 80]
    harvest_voxels in sphere(3.0) drop_as stone_chunk
}

ability Tunneler {
    target: ground
    range: 4.0
    cooldown: 5s
    cast: 800ms
    hint: utility

    harvest_voxels in cylinder(1.0, 4.0)
}
```

**Acceptance criteria:**
- `material stone { hardness: 0.8, ... }` declared in .sim
- `structure Castle(...) { place stone in box(20, 8, 20) ... }` rasterizes
  to StructureRegistry entry at compile time
- `place_voxels` writes into voxel storage and bumps chunk epoch
- `harvest_voxels` reads + clears voxels, emits drop events
- `transform_voxels` conditional per-cell (only matching voxels change)
- Voxel revert queue pops at lifetime expiry
- Damageable voxels track HP across multiple `damage`-vs-voxel events
- Walls block movement (resolve_movement honors voxel storage)
- `cargo run --bin viz_app -- siege_defense` renders voxel layer as
  background grid overlay

**What this catches:** voxel storage subsystem bugs, mask rasterization
bugs, structure expansion bugs, CSG composition bugs, voxel revert queue
ordering, voxel-aware condition lowering.

---

## Wave 5 acceptance: `village_crafting_real`

**Validates:** economy — recipes, inventory SoA, tools, skills, quality
formulas, EffectOps 17-26.

**Composition:** 30 villagers + 5 resource nodes + commodity flow.

**Recipe set** (in `.sim`):
```
recipe ForgeIron {
    inputs: [iron_ore 2, charcoal 1]
    outputs: [iron_ingot 1]
    duration_ticks: 100
    skill: blacksmithing >= 0.3
    tool: forge
    quality: 0.4 * inputs_quality + 0.5 * skill + 0.1 * tool_quality
}

recipe ForgeSword {
    inputs: [iron_ingot 2, leather 1]
    outputs: [sword 1]
    duration_ticks: 200
    skill: blacksmithing >= 0.5
    tool: forge
    quality: 0.4 * inputs_quality + 0.5 * skill + 0.1 * tool_quality
}

recipe RepairForge {
    inputs: [iron_ingot 1, stone 2]
    target_tool: forge
    reduce_wear: 0.5
    skill: blacksmithing >= 0.4
}
```

**Villager ability set:**
```
ability Mine {
    target: ground
    range: 2.0
    cooldown: 3s
    require_skill mining >= 0.1
    require_tool pickaxe

    consume durability(pickaxe) 0.01
    produce iron_ore quality(0.3 + 0.5 * skill)
}

ability CastForgeIron {
    target: self
    cast: 10s
    require_skill blacksmithing >= 0.3
    require_tool forge

    cast_recipe ForgeIron
}

ability TradeWith {
    target: ally
    range: 3.0
    cooldown: 5s

    transfer_gold -25
    // exchange handled by recipe matching their offer
}
```

**Acceptance criteria:**
- Inventory SoA per villager (commodities + tools)
- Recipes lower to RecipeRegistry entries (compile-time partition)
- `consume`/`produce` clauses match RecipeEntry inputs/outputs
- Quality formulas evaluate per-cast (read input qualities, agent skill,
  tool quality)
- `require_skill` mask predicate gates ability availability
- `require_tool` mask predicate gates against agent's owned-tool inventory
- Tool wear accumulates; broken tools fail recipes
- `RepairForge` is itself a recipe that targets an owned tool
- ApplyTrade chronicle moves both gold AND commodity counts
- `cargo run --bin viz_app -- village_crafting_real` renders villagers
  colored by primary skill, glyph indicating current activity

**What this catches:** RecipeRegistry serialization, inventory variable-
size SoA layout (the central architectural decision from §17), quality
formula evaluation, gate predicate composition, multi-effect chronicle
handlers (Recipe + WearTool simultaneously).

---

## Wave 6 acceptance: `recipe_invention_demo`

**Validates:** runtime ability composition — slot-fill model, three
hard gates (grammar/economic/novelty), bounded registries with LRU GC,
generated abilities as GOAP planner goals.

**Composition:** 10 villagers from `village_crafting_real` plus a
generation-trigger event loop. Run for 50000 ticks. Track:
- Generation events fired
- Compositions that pass each gate
- Adopted compositions (used by other agents via observation/testimony)
- LRU evictions

**Acceptance criteria:**
- Idle-with-resources trigger fires when villager has materials but no
  applicable recipe
- Slot-fill produces grammatically valid AbilityProgram values
- Economic gate rejects money-pumps (output_value < input_value)
- Novelty gate rejects near-duplicates (cosine sim > 0.9)
- Surviving compositions register in the runtime partition of
  RecipeRegistry
- Other villagers learn via apprenticeship (existing K-induction verb
  combined with new domain-knowledge transfer)
- LRU GC evicts unused compositions after threshold
- ML model retrained against new EffectOp vocab + Phase 2 finetune
  produces compositions that adopt at >5% rate

**What this catches:** all the cross-cutting ML + gate + registry
machinery. By far the largest acceptance test — multi-week.

---

## Cross-cutting test sims

These don't fit any single wave but exercise spec-wide invariants.

### `two_phase_split_validator`

Demonstrates the compile-time validator pass rejects ground-truth
queries from decision-time code. Must:

1. Author a `.sim` file that ATTEMPTS to call `engine::voxel_at(...)`
   inside a verb's `score` expression
2. Compiler emits the spec's specified error: `decision-time code may
   not call ground-truth queries`
3. Author a CORRECT version that uses a `view` over emitted events;
   compiler accepts

This is a single fixture in `crates/dsl_compiler/tests/two_phase_split.rs`,
not a runtime sim. Lands with Wave 3 (when belief views become
relevant).

### `kernel_emit_smoke`

Exercises the `engine_gpu_rules/` sibling crate emission path. Lands
with the kernel-emit subsystem (per the 2026-04-26 design). Smallest
fixture: a single ability runtime that builds against `engine_gpu_rules/`
generated code instead of hand-written wgpu wrappers.

### `replay_determinism_abilities`

A sim that runs N ticks, snapshots, runs N more. Compares snapshots
across runs at byte level. Ensures abilities don't introduce
non-determinism. Lands with each wave (parameterized over the wave's
canonical sim).

---

## Phasing summary

| Wave | Acceptance sim | Depends on | Crate |
|---|---|---|---|
| 1 | `duel_abilities` | parser + AbilityProgram lowering + AbilityRegistry wiring | `crates/duel_abilities_runtime` |
| 2 | `tactical_squad_abilities` | Wave 1 + new EffectOps 8-11 + delivery enum + status SoA fields | `crates/tactical_squad_abilities_runtime` |
| 3 | `spy_network` | Wave 1 + BeliefState SoA + EffectOps 12-16 + selectors | `crates/spy_network_runtime` |
| 4 | `siege_defense` | Wave 1 + voxel storage + materials + EffectOps 8-11 (voxel) | `crates/siege_defense_runtime` |
| 5 | `village_crafting_real` | Wave 1 + Wave 3 (beliefs) + inventory SoA + EffectOps 17-26 | `crates/village_crafting_real_runtime` |
| 6 | `recipe_invention_demo` | Waves 1-5 + ML retrain + slot-fill engine | `crates/recipe_invention_demo_runtime` |
| cross | `two_phase_split_validator` | Wave 3 (belief views) | `crates/dsl_compiler/tests/` |
| cross | `kernel_emit_smoke` | engine_gpu_rules/ | `crates/kernel_emit_smoke_runtime` |
| cross | `replay_determinism_abilities` | each wave | reusable test harness |

Each acceptance sim follows the prior real-sim pattern: ~600-1000 LOC
runtime + ~150 LOC app + .ability test corpus + compile-gate test in
`stress_fixtures_compile.rs` + viz hookup via SIMS table in
`viz_app.rs`.

## Today's actionable steps

1. **`assets/ability_test/duel_abilities/*.ability`** — author the 3-4
   ability files now (Strike/ShieldUp/Mend). Pure data, parser doesn't
   exist yet so they're aspirational, but they're the corpus the parser
   will hit first.
2. **Crate skeleton** for `duel_abilities_runtime` deferred until Wave
   1.0 lands (Cargo.toml workspace edit conflicts with the in-flight
   parser agent).
3. **App skeleton** for `duel_abilities_app` similarly deferred.
4. **Compile-gate test stub** — could land now in
   `crates/dsl_compiler/tests/ability_parse_smoke.rs` referencing the
   `assets/ability_test/duel_abilities/Strike.ability` file. Currently
   skipped via `#[ignore]` until parser exists; un-ignore at Wave 1.0
   merge.
