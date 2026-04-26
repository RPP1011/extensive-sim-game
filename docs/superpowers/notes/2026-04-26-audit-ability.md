# Ability Spec Audit (2026-04-26)

> Audit of `docs/spec/ability.md` against `crates/dsl_compiler/`, `crates/tactical_sim/src/effects/dsl/`, `crates/engine/src/ability/`, and assets.

## Architecture clarification (essential context)

There are **two separate ability stacks** in this codebase, and the spec describes a third future state that neither fully implements yet.

| Stack | Parser | Runtime | Scope |
|---|---|---|---|
| **Engine** (`crates/engine`) | None (hand-built `AbilityProgram`) | `CastHandler` + GPU kernel | 100ms deterministic combat; the spec's target |
| **TacticalSim** (`crates/tactical_sim/src/effects/dsl/`) | Winnow-based `.ability` parser (ships today) | `triggers.rs` + `apply_effect.rs` | Campaign / RPG / world-sim hero abilities |
| **Spec future** (`docs/spec/ability.md`) | Not yet | Not yet | Targets engine stack only |

The `.ability` files in `dataset/hero_templates/` and `assets/hero_templates/` are loaded by the **TacticalSim** parser, **not** by the engine or `dsl_compiler`. The `dsl_compiler` crate is the World Sim `.sim` file compiler — it is unrelated to `.ability` parsing.

This creates a systematic spec-vs-implementation mismatch: the spec's §23 Capability Status Matrix was written against the engine stack, but most surface constructs that "work today" (`passive` blocks, `deliver`, shapes, most control verbs, etc.) work only in the TacticalSim layer, which operates on different semantics and does not produce engine `EffectOp`s.

---

## Summary

- §23 Capability matrix items: **48** enumerated rows (grouped)
- ✅ runs-today + implemented (engine stack): **~10**
- 🤔 runs-today per spec but engine-only partial / TacticalSim-only: **~8**
- ⚠️ planned, partially in flight (TacticalSim has impl, engine does not): **~15**
- ❌ planned, no implementation in either stack: **~12**
- 🔒 reserved: **~20** (skipped per scope)

**Critical finding:** `passive` block, `deliver` block, shapes, most control/movement/buff verbs, and conditions are marked `planned` in the spec but are **fully parsed and dispatched** in `tactical_sim`. The spec's §23 status matrix describes only the engine stack; TacticalSim is 1-2 major tracks ahead.

---

## Top gaps (ranked by impact for plan drafting)

### Gap 1 — Engine delivery methods (§10): Instant only, spec marks all others `planned`
**Impact: HIGH.** The engine's `Delivery` enum has exactly one variant: `Instant = 0`. The spec describes `projectile`, `chain`, `zone`, `channel` (and reserves `tether`, `trap`). `.ability` files authored with `deliver projectile {}` or `deliver zone {}` blocks are parsed by TacticalSim and execute there, but the engine stack is blind to them. Every hero `.ability` file uses `deliver` blocks for interesting abilities (Fireball, ArcaneMissiles, Blizzard, etc.). Until the engine adds `Delivery::Projectile`, `Chain`, `Zone`, `Channel` variants + a resolver, the deterministic simulation cannot run multi-hit, travel-time, or persistent zone abilities. This is the highest-leverage single gap.

### Gap 2 — Engine `Area` enum: SingleTarget only (§5.1, §9)
**Impact: HIGH.** Engine's `Area` has exactly one variant: `SingleTarget { range }`. All shape primitives (`circle`, `cone`, `line`, `ring`, and all 3D volumes) are `planned` but unimplemented in engine. TacticalSim has a full `Area` enum with all disc shapes. Nearly every ability with `target: self_aoe` or `target: ground` and an `in <shape>` modifier runs only in TacticalSim. A plan to add `Area::Circle`, `Cone`, `Line`, `Ring` to the engine would unlock the majority of non-single-target combat mechanics.

### Gap 3 — Passive triggers (§6): TacticalSim runs them, engine spec marks all `planned`
**Impact: MEDIUM-HIGH.** TacticalSim has a full `triggers.rs` module that fires 17+ trigger kinds (`on_damage_dealt`, `on_kill`, `on_hp_below`, `on_shield_broken`, `on_stack_reached`, etc.) with a depth guard (max 3) and cooldown enforcement. The spec §23.1 marks `passive` as `planned` and notes "no Trigger AST or handler in engine yet." This is an overclaim in the spec: TacticalSim **does** implement triggers end-to-end. The gap is engine-side only. A plan to add a passive trigger dispatcher to the engine's cascade system (sitting alongside `CastHandler`) would close this.

---

## Per §23 capability findings

### 23.1 Core (§5, §6, §7, §8.1)

| §ref | Construct | Spec marker | Actual status | Evidence |
|---|---|---|---|---|
| §5 | `ability` block | `runs-today` | ✅ Engine: hand-built `AbilityProgram`; TacticalSim: parsed via `parser.rs` | `engine/src/ability/program.rs`, `tactical_sim/src/effects/dsl/parser.rs` |
| §6 | `passive` block | `planned` | ⚠️ TacticalSim: fully parsed + dispatched in `triggers.rs`; engine: not wired | `tactical_sim/src/effects/dsl/lower.rs:247`, `sim/triggers.rs` |
| §5.1 | `target: enemy` / `self` | `runs-today` | ✅ Engine: `Gate.hostile_only`, `Area::SingleTarget`; TacticalSim: `AbilityTargeting` enum | `engine/program.rs:62` |
| §5.1 | `target: ally` / `self_aoe` / `ground` / `direction` | `planned` | ⚠️ TacticalSim: parsed (ally, self_aoe, ground, direction, vector, global, faction, region, market, party, guild, adventurer, location); engine: no `Area` variants | `lower.rs:201-219` |
| §5.1 | `target: vector` / `global` | `reserved` | 🔒 TacticalSim parses them silently | — |
| §5.2 | `range:`, `cooldown:` | `runs-today` | ✅ Engine: `Gate.cooldown_ticks`, `Area::SingleTarget { range }`; TacticalSim: same | Both stacks |
| §5.4 | `cast:` | `planned` | ⚠️ TacticalSim: `cast_time_ms` field stored on `AbilityDef`; engine: no `cast_ticks` on `Gate` | `lower.rs:222` |
| §5.5 | `hint:` (4 values) | `runs-today` | ✅ Engine: `AbilityHint` enum (4 variants); TacticalSim: stored as string | `engine/program.rs:140` |
| §5.5 | `hint: heal` | `planned` | 🤔 Spec says planned because engine enum lacks `heal`; TacticalSim just stores it as string "heal" with no type check | `ir.rs:124-154` — `AbilityHint` has no `Heal` variant |
| §5.6-5.13 | `cost:`, `charges:`, `toggle`, `form:`, `recast`, `morph` | mixed reserved/planned | ⚠️ TacticalSim parses all of them (`lower.rs:225-235`); engine has none | `engine/program.rs` — not present |
| §8.1 | Combat verbs (`damage`, `heal`, `shield`, `stun`, `slow`) | `runs-today` | ✅ Engine: `EffectOp` variants 0-4; TacticalSim: `Effect` enum variants | `engine/program.rs:89-109` |
| §8.2 | Control verbs (`root`, `silence`, `fear`, `taunt`) | `planned` | ⚠️ TacticalSim: parsed and stored (`lower_effects.rs:167-186`); engine: not in `EffectOp` | `lower_effects.rs:167` |
| §8.2 | `charm`+ | `reserved` | ⚠️ TacticalSim: `charm`, `polymorph`, `banish`, `confuse`, `suppress`, `grounded` all parsed (`lower_effects.rs:187-215`); spec says reserved; TacticalSim is ahead of spec | `lower_effects.rs:187` |
| §8.3 | Movement verbs (`dash`, `knockback`, `pull`, `blink`) | `planned` | ⚠️ TacticalSim: all implemented (`lower_effects.rs:217-244`); engine: absent | `lower_effects.rs:217` |
| §8.3 | `swap` | `reserved` | ⚠️ TacticalSim: `Effect::Swap` exists (`lower_effects.rs:246`); engine: absent | — |
| §8.4 | Buffs/debuffs (`buff`, `debuff`, `damage_modify`, `lifesteal`) | `planned` | ⚠️ TacticalSim: all implemented; engine: absent | `lower_effects.rs:247-277` |
| §8.4 | `reflect`, `blind` | `reserved` | ⚠️ TacticalSim: implemented; spec says reserved | `lower_effects.rs:264-277` |
| §8.5 | Advanced (`execute`, `self_damage`) | `planned` | ⚠️ TacticalSim: implemented (`lower_effects.rs:381-388`); engine: absent | — |
| §8.5 | Most other advanced verbs | `reserved` | ⚠️ TacticalSim: implements `summon`, `stealth`, `leash`, `link`, `redirect`, `rewind`, `cooldown_modify`, `apply_stacks`, `dispel`, `immunity`, `death_mark`, `resurrect`, `overheal_shield`, `absorb_to_heal`, `shield_steal`, `status_clone`, `detonate`, `status_transfer`, `on_hit_buff`, `obstacle`, `projectile_block`, `attach`, `evolve_ability`, `command_summons` — all "reserved" per spec | `lower_effects.rs:279-550` |
| §8.7 | `transfer_gold`, `modify_standing` | `runs-today` | ✅ Engine: `EffectOp::TransferGold(5)`, `EffectOp::ModifyStanding(6)`; TacticalSim: separate impl | `engine/program.rs:112-120` |
| §8.9 | `cast` (meta) | `runs-today` | ✅ Engine: `EffectOp::CastAbility(7)` with cascade depth bound; TacticalSim: via `on_hit_cast` | `engine/program.rs:122` |
| §10 | Delivery (`projectile`, `chain`, `zone`, `channel`) | `planned` | ⚠️ TacticalSim: parsed and stored via `parse_delivery.rs`/`lower_delivery.rs`; engine: `Delivery::Instant` only | `engine/program.rs:48-50` |
| §11 | Conditions (physical atoms) | mixed | ⚠️ TacticalSim: `lower.rs` implements hp-based, status-based, tag, campaign conditions; engine: no condition system in `EffectOp` | `lower.rs:13-83` |
| §12 | Stacking modes (`refresh`) | `planned` | ⚠️ TacticalSim: `Stacking::Refresh/Extend/Strongest/Stack` all parsed; engine: no stacking in cascade handlers | `lower_effects.rs:12-19` |
| §13 | Scaling (stat terms) | `planned` | ⚠️ TacticalSim: `ScalingTerm`/`StatRef` with 7 stat variants + stacks; engine: no scaling | `lower_effects.rs:739-779` |
| §14 | Tags (fixed 6 enum) | `runs-today` | ✅ Engine: `AbilityTag` 6-variant enum + packed registry; TacticalSim: string-keyed tags | `engine/program.rs:184` |
| §14 | Tags: open-set mapping | `planned` | ❌ Neither stack has a tag mapping table; TacticalSim stores raw strings (no mapping); engine has no open-set path | — |

### 23.2 Shapes & volumes (§9)

| Construct | Spec marker | Actual status | Evidence |
|---|---|---|---|
| 2D shapes (circle/cone/line/ring/spread) | `planned` | ⚠️ TacticalSim: `lower.rs:101-131` fully parses all 5 disc shapes; engine: `Area::SingleTarget` only | `engine/program.rs:56-58` |
| 3D volumes (box/sphere/column/wall/cylinder/dome/hull) | `planned` | ❌ Neither stack; TacticalSim `lower_area` only handles 2D + `spread` | `lower.rs:101-131` |
| CSG composition (union/diff/intersect) | `planned` | ❌ No implementation in either stack | — |
| Grid snapping & 90° rotation | `planned` | ❌ No implementation | — |
| Shape rasterization + mask registry | `planned` | ❌ No implementation | — |

### 23.3 Templates & structures (§15, §16, §17)

| Construct | Spec marker | Actual status |
|---|---|---|
| `template` block with typed parameters | `planned` | ❌ No template system in either ability stack |
| Compile-time arithmetic | `planned` | ❌ |
| `structure` block with parameters | `planned` | ❌ |
| `include` composition + symmetry | `planned` | ❌ |
| Material declaration in `.sim` | `planned` | ❌ No material catalog wired |
| Material property catalog | `planned` | ❌ |

### 23.4 Voxel ops (§18, §19)

All `planned` or `reserved` per spec. No implementation in either ability stack. The engine voxel subsystem (`crates/engine_voxel`) is referenced but not yet present. `EffectOp` variants 8–11 do not exist. All voxel conditions are `reserved`.

**Status: ❌ across the board.**

### 23.5 AI-state manipulation (§20)

| Subsystem | Spec marker | Actual status | Evidence |
|---|---|---|---|
| Standing basic (`befriend`, `enmity`) | `runs-today` | ✅ Engine: `EffectOp::ModifyStanding { delta: i16 }` exists and fires | `engine/program.rs:120` |
| Standing three-party (`endear`, `slander`, `rally_cry`) | `planned` | ❌ Engine `ModifyStanding` lacks `a_sel`/`b_sel` selectors (spec §22.2 evolution not done) | `engine/program.rs:120` — still `delta: i16` only |
| TOM Phase 1 verbs (J writes) | `planned` | ❌ `EffectOp` variants 12-14 (`ClearBelief`, `PlantBelief`, `RefreshBelief`) don't exist; ToM Phase 1 belief-scoring is separate pipeline | — |
| Communication (M) | `planned` | ❌ No implementation |
| Group/faction (K) | `reserved` | 🔒 |
| Quest valuation (L) | `reserved` | 🔒 |
| Engagement/coordination (N) | `planned` | ❌ `EffectOp::EmitEvent` variant 15 does not exist; TacticalSim has engagement verbs via direct state mutation | — |

### 23.6 IR & events

| Construct | Spec marker | Actual status |
|---|---|---|
| `EffectOp` variants 0–7 (existing) | `runs-today` | ✅ All present in `engine/program.rs` |
| `EffectOp` variants 8–11 (voxel) | `planned` | ❌ Not added |
| `EffectOp` variants 12–14 (TOM writes) | `planned` | ❌ Not added |
| `EffectOp` variant 15 (`EmitEvent`) | `planned` | ❌ Not added |
| `EffectOp` variant 16 (`ModifyGroupStanding`) | `reserved` | 🔒 |
| `PairSelector` / `SubjectSel` / `AgentSel` enums | `planned` | ❌ Selector enums do not exist; `ModifyStanding` still takes bare `delta: i16` |
| Voxel mask registry + schema hash | `planned` | ❌ |
| Structure registry | `planned` | ❌ |
| Belief events | `reserved` | 🔒 |

### 23.7 Budget & determinism

| Constant | Spec value | Actual status |
|---|---|---|
| `MAX_ABILITIES` (per agent) | 8 (ships today) | ✅ `engine/program.rs:39` |
| `MAX_EFFECTS_PER_PROGRAM` | 4 (ships today) | ✅ `engine/program.rs:28` |
| `MAX_TAGS_PER_PROGRAM` | 6 (ships today) | ✅ `engine/program.rs:239` |
| `MAX_CASCADE_ITERATIONS` | 8 (ships today) | ✅ `engine` cascade depth check |
| `MAX_VOXELS_PER_ABILITY` | 16384 (planned) | ❌ |
| `MAX_VOXELS_PER_TICK` | 65536 (planned) | ❌ |
| `MAX_ABILITY_CASTS_PER_TICK` | 1024 (planned) | ❌ |
| Template expansion depth | 16 (planned) | ❌ |

---

## Beyond §23 — sections not in the matrix

### §5.5 `hint: heal` overclaim
Spec §23.1 says `hint:` is `runs-today` for four values, `planned` for `heal`. However the engine's `AbilityHint` enum has exactly four variants: `Damage(0)`, `Defense(1)`, `CrowdControl(2)`, `Utility(3)` — no `Heal`. The TacticalSim stores `ai_hint` as a raw string and doesn't type-check it. Files that write `hint: heal` will parse in TacticalSim but the spec says the engine lowers `heal` to `Utility` with "a documented loss." No such loss-documentation or fallback code exists in the engine. This is a minor **overclaim** (🤔) against the engine stack.

### §8.2 `charm` / §8.4 `reflect` / §8.5 advanced verbs — spec says reserved, TacticalSim implements
The spec marks `charm`, `polymorph`, `banish`, `confuse`, `suppress`, `grounded`, `reflect`, `blind`, and a large swath of `§8.5` as `reserved`. TacticalSim parses and stores all of them (they produce `Effect::*` enum variants). This is not a spec violation — TacticalSim is a different layer — but it does mean the spec's "reserved" designation is misleading for the campaign game; those verbs are fully usable there. A future plan should reconcile which TacticalSim verbs can/should be promoted to the engine.

### §11 Conditions — TacticalSim far ahead of spec claims
The spec marks most physical conditions as `planned` and all AI-state / voxel conditions as `planned`/`reserved`. TacticalSim's `lower.rs` implements 15+ condition atoms (hp-based, status-based, faction, economic). These have no engine-side counterpart but are used in campaign `.ability` files today.

### §20.1 `ModifyStanding` evolution not done
Spec §22.2 states `ModifyStanding` evolves to `{ a_sel: PairSelector, b_sel: PairSelector, delta: i16 }`. The engine's `EffectOp::ModifyStanding` still carries only `{ delta: i16 }`. The `befriend` / `enmity` three-party variants are therefore unimplementable until the schema-hash-bumping evolution lands.

### §4.2 `abilities_file` TOML field
The spec says hero TOML gains `abilities_file = "<name>.ability"`. Looking at actual hero assets (e.g., `mage.ability` in `dataset/hero_templates/`), `.ability` files exist and are consumed by TacticalSim. The engine-side TOML binding path described in §4.2 does not yet exist.

---

## Cross-cutting observations

### Track 1: Engine delivery + area (§10, §9) — zero coverage
Every interesting ability authored in `.ability` files (projectile spells, zone AoEs, chain lightning) runs exclusively in TacticalSim. The engine runs only instant single-target abilities. A sub-plan titled "Engine Delivery Methods v1" covering `Projectile`, `Zone`, and `Area::Circle/Cone` would bring the two stacks to parity for the most common combat patterns.

### Track 2: Passive trigger system (§6) — TacticalSim complete, engine zero
TacticalSim's `triggers.rs` covers 17 trigger kinds including depth-guarded recursion and cooldown enforcement. The engine has no trigger dispatcher. A sub-plan porting the TacticalSim trigger architecture to the engine's cascade system (as a `TriggerHandler` alongside `CastHandler`) would be self-contained.

### Track 3: Control/movement/buff verbs in engine (§8.2–8.4) — all absent
The engine's `EffectOp` has 8 variants; the spec defines 40+. TacticalSim implements most. The bottleneck is not design (TacticalSim provides a reference implementation) but the engine-side cascade handler for each new `EffectOp`. A systematic port plan grouping by category (control group, movement group, buff group) would be the efficient path.

### Track 4: `ModifyStanding` selector evolution (§20.1) — schema bump pending
The selector enum expansion (`PairSelector`) is a well-scoped, schema-hash-bumping change. Three-party standing writes (`endear`, `slander`, `rally_cry`) depend on it. This is a prerequisite for the "AI-state manipulation" track.

### Track 5: Voxel ops and structures (§18, §16) — spec-only, no partial impl
Voxel ops (§18), structure blueprints (§16), material catalog (§17), and voxel-aware conditions (§19) have zero implementation anywhere. The spec is complete, but neither stack has started. This is the longest track and most appropriately its own major plan.

### Observation: TacticalSim as the "running spec"
For practical purposes, `crates/tactical_sim/src/effects/dsl/` is the current working implementation of the ability DSL for campaign play. It is 1-2 tracks ahead of the spec's `runs-today` claims on the engine side, and implements several `reserved` constructs. New ability spec work should cross-check against TacticalSim's existing behavior as a reference, not just the engine's `EffectOp` catalog.
