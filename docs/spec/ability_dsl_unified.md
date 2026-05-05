# Ability DSL — Unified GPU Spec

> **Status:** Design (2026-05-04). Consolidates and supersedes:
> - `docs/spec/ability.md` (canonical ability DSL, CPU-leaning)
> - `docs/spec/economy.md` (recipes/inventory/obligations as ability extensions)
> - `docs/spec/dsl.md` §2.6 (`verb` declaration in .sim)
> - `docs/superpowers/specs/2026-04-22-theory-of-mind-design.md`
> - `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md`
> - `docs/superpowers/specs/2026-04-26-kernel-dispatch-emit-design.md`
> - `docs/superpowers/specs/2026-04-24-ability-dsl-design.md` (brainstorm doc)
>
> **Framing change from prior specs.** The original specs assumed a CPU
> AbilityRegistry + cascade evaluator. This spec assumes everything
> compiles to WGSL kernels through the existing dsl_compiler pipeline.
> Every EffectOp variant, every condition atom, every registry lookup
> has a GPU lowering. CPU paths are reserved for orchestration only
> (frame setup, snapshot/replay, kernel dispatch, viz readback).
>
> **What this spec does NOT cover.** Generative ability composition
> (the runtime ML model from the original spec §14), training pipeline
> details, second-order beliefs, arbitrary-rotation voxel placement.
> These remain in the original docs.

---

## §1 Why this exists

The platform now has 22+ working sims demonstrating the verb cascade
pipeline (mask → scoring → chronicle → ApplyX) compiles cleanly to
WGSL and scales linearly to 10000+ agents at 0.3 ms/tick. The existing
ability DSL spec was authored before this pipeline was proven and
assumes CPU evaluation throughout.

This spec consolidates the ability surface across six source docs and
re-anchors every construct on the verb-cascade pipeline. The result is
a unified design where:

1. The `.sim` DSL gains the missing ability constructs (recipes,
   templates, structures, voxel ops) as new top-level forms.
2. The `.ability` file format is preserved as an authoring sugar,
   parsed and lowered into the same IR as `.sim`.
3. Every EffectOp lowers to either (a) a GPU dispatch fragment within
   the existing apply-physics pipeline, or (b) a CPU orchestration
   primitive when GPU lowering is fundamentally impossible (e.g.
   recipe registry lookup with structured payload — even then, the
   GPU side handles bulk execution).
4. The two-phase epistemic split is enforced at compile time by the
   existing CG validator pass.
5. New per-agent state (BeliefState, Inventory, Skills) follows the
   same SoA pattern the engine already uses, with one architectural
   addition: variable-size per-agent fields (the SoA inventory/belief
   problem).

---

## §2 Architectural pillars

Five constraints govern every design choice in this spec.

### 2.1 Compiler-first (P1)

Every kernel that runs on the GPU is compiler-emitted. Hand-written
kernels in `crates/engine_gpu/` are reserved for the wgpu setup +
megakernel orchestration + buffer pool. The kernel-dispatch-emit
design (referenced spec) routes all ability-derived kernels through
`crates/engine_gpu_rules/` (a new sibling crate).

**Implication:** adding an ability verb means extending the dsl_ast +
CG IR + WGSL emitter. The engine crate doesn't change unless the new
verb needs a new SoA field or a new event variant.

### 2.2 Determinism + cross-backend parity (P3, P5, P11)

All RNG flows through `per_agent_u32(seed, agent_id, tick, purpose)`.
Reductions are sort-then-fold so that float ordering is bit-stable
across CPU and GPU backends. Replay is byte-equal across runs.

**Implication:** any ability whose effect is RNG-dependent (chance
modifier, dice rolls, random target selection) must use the keyed PCG.
Any ability whose effect involves a reduction over multiple agents
(AoE damage, area aggregate condition) must order participants
deterministically.

### 2.3 Events are the mutation channel (P6)

State changes happen through events written to the per-tick event
ring, then consumed by chronicle physics rules. Direct field writes
outside `step::*` and `snapshot::*` are forbidden.

**Implication:** every EffectOp produces zero or more events. The
event variant set is part of the schema hash. EffectOps are NOT
direct mutations — they emit events that the existing chronicle
pipeline consumes.

### 2.4 Two-phase epistemic split

Decision-time code (`mask`, `scoring`, `verb` selection) reads only
from belief-mediated views. Execution-time code (`physics` rules
resolving chosen intents) reads ground truth.

**Implication:** the CG validator gains a new pass that rejects
ground-truth-API calls in any IR node downstream of a `MaskPredicate`
or `ScoringArgmax` op. Belief reads (via `view::*` lookups over
materialized belief views) are the only allowed decision-time data
source.

### 2.5 Schema-hash discipline (P2)

Layout changes (SoA fields, event variants, EffectOp ordinals,
selector enums, registry shapes) bump `crates/engine/.schema_hash`.
Snapshots and replay traces refuse to load against a different hash.

**Implication:** the EffectOp ordinal table (§14) is reserved at
design time so authored abilities don't churn the hash on every
incremental implementation.

---

## §3 Two-DSL surface

The platform retains TWO file formats:

### 3.1 `.sim` files

The world-rules DSL. Declares entities, events, views, physics
cascades, mask predicates, verbs, invariants, probes, metrics,
spatial queries, and (NEW from this spec) materials, recipes,
templates, structures.

**Top-level forms:**
```
entity     event     view      physics     mask       verb
invariant  probe     metric    spatial_query
material   recipe    template  structure                   ← new
```

**Lowering:** parser → dsl_ast → resolver → CG IR → schedule → WGSL
emit + Rust emit. Every fixture's `build.rs` runs this pipeline at
build time and includes the generated artifacts via `include!`.

### 3.2 `.ability` files

The authoring sugar for combat-flavored abilities. Hero TOML files
reference `.ability` files via `abilities_file = "warrior.ability"`.
Ability blocks compile to AbilityProgram entries in the
AbilityRegistry, which is itself a fixed-size SoA the GPU dispatch
pipeline consumes.

**Top-level forms:**
```
ability    passive    template    structure
```

`material` is declared in `.sim` and referenced from `.ability` (per
existing spec §17).

**Lowering:** `.ability` parser → dsl_ast (same AST module as `.sim`)
→ resolver → CG IR (same IR with `AbilityProgram` op) → WGSL emit.
The two parsers feed the same downstream pipeline.

### 3.3 Why two surfaces

A unified single-format approach was considered. Rejected because:

1. `.ability` files are authored by content designers (or a generative
   model), `.sim` files are authored by engine programmers. Different
   audiences want different ergonomics.
2. The training pipeline (ability transformer + entity encoder) takes
   `.ability` token sequences as input. Preserving the format keeps
   the trained weights useful.
3. The `.ability` syntax is more compact for the ability-shaped
   subset (`damage 50 in circle(2.5) [FIRE: 60] when target_hp_below(30%)`).
   Forcing it into `.sim` syntax (`verb`/`physics`/`emit`) loses
   compactness without architectural benefit.

The two formats share the lexer (UTF-8, same token classes), the AST
module, the resolver, and the lowering pipeline. Only the parser
front-end differs.

---

## §4 Top-level form: `ability`

### 4.1 Surface

```
ability ForgeSword {
    target: self
    cast: 400ms
    cooldown: 30s
    hint: economic
    require_skill blacksmithing >= 0.4
    require_tool forge

    consume iron_ingot 2
    consume leather 1
    produce sword quality(0.4 * inputs_quality + 0.5 * skill + 0.1 * tool_quality)

    wear_tool forge 0.05
}
```

### 4.2 Header properties

Set Gate fields and metadata. Order is free; each property at most once.

| Property | Type | GPU mapping |
|---|---|---|
| `target` | targeting mode | sets dispatch shape (PerAgent / PerPair) |
| `range` | f32 metres | mask predicate clause: `distance(self, target) <= range` |
| `cooldown` | duration | mask predicate: `tick >= self.cooldown_next_ready_tick[ability_id]` |
| `cast` | duration | per-agent cast-in-progress state SoA field |
| `hint` | enum | metadata only; influences scoring score expressions |
| `cost` | int | mana/resource gate in mask predicate |
| `zone_tag` | string | combo anchor — interned ID, used by `detonate` |
| `charges` | int | per-agent charge SoA field |
| `recharge` | duration | charge regen rate |
| `toggle` / `toggle_cost` | flag / f32 | sustained ability state (PerAgent) |
| `recast` / `recast_window` | int / dur | multi-stage cast state |
| `unstoppable` | flag | CC-immune-during-cast flag |
| `form` / `swap_form` | string | form-group registry entries |
| `require_skill` | name + threshold | mask predicate against per-agent skill SoA |
| `require_tool` | tool kind | mask predicate against per-agent tool inventory |

### 4.3 Targeting modes

| Mode | Payload | Dispatch shape | Status |
|---|---|---|---|
| `enemy` | AgentId | PerPair (mask narrows to hostile-in-range) | runs-today via verb cascade |
| `ally` | AgentId | PerPair (mask narrows to friendly-in-range) | planned |
| `self` | caster | PerAgent | runs-today |
| `self_aoe` | caster pos + radius | PerAgent + spatial body-form | planned |
| `ground` | Vec3 | PerAgent + spatial body-form | planned |
| `direction` | Vec2 | PerAgent | planned |
| `vector` | Vec3 offset | PerAgent | reserved |
| `global` | none | per-tick singleton | reserved |

### 4.4 Body items

```
body_item = effect_stmt | deliver_block | recast_block | morph_block ;
```

Body holds **either** a `deliver` block **or** one-or-more bare
`effect_stmt`s, not both. `recast` and `morph` combine with either.

### 4.5 GPU lowering

Each `ability` lowers to:

1. **An entry in `AbilityRegistry`** (SoA buffer of `AbilityProgram`
   structs, GPU-resident).
2. **A `verb` IR entry** — header props become the verb's mask
   predicate and scoring expression; effects become the verb's emit
   clauses; the chronicle physics rule applies the effects.
3. **Per-effect chronicle handlers** — one per EffectOp variant the
   ability emits, gated on `action_id == ability_id`.

The existing verb cascade pipeline (Slice A from 2026-05-03) is the
substrate. Abilities are verbs with extra metadata (Gate fields) and
typed payloads (EffectOp emits).

---

## §5 Top-level form: `passive`

### 5.1 Surface

```
passive Riposte {
    trigger: on_damage_taken
    cooldown: 5s
    range: 2.0

    damage 30 [PHYSICAL: 50]
}
```

### 5.2 Trigger catalog (24 kinds)

Triggers fire when the engine emits an event matching the trigger
kind. Compile to PerEvent dispatches in the same way `physics`
chronicle rules do.

**Combat:** `on_damage_dealt`, `on_damage_taken`, `on_kill`, `on_death`,
`on_ability_used`, `on_ally_damaged`, `on_ally_killed`, `on_hp_below`,
`on_hp_above`, `on_shield_broken`, `on_stun_expire`, `on_heal_received`,
`on_status_applied`, `on_status_expired`, `on_dodge`, `on_reflect`,
`on_auto_attack`, `on_stack_reached`, `on_resurrect`.

**Periodic:** `periodic` (fires every duration on PerAgent dispatch).

**Voxel:** `on_voxel_placed`, `on_voxel_harvested`,
`on_voxel_transformed`, `on_voxel_reverted`, `on_structure_placed`.

All triggers support optional modifiers (`range:`, `by:`, type-specific
filters) inside parens.

### 5.3 GPU lowering

Each passive lowers to:

1. **A PerEvent dispatch** keyed on the trigger's event kind.
2. **A mask predicate** filtering by the trigger's modifiers (range,
   by-agent, etc.) — same shape as physics rule `where` clauses.
3. **A chronicle handler** that applies the passive's effects per
   triggering event.

`periodic` is the exception: it lowers to a PerAgent dispatch with a
mask gating on `tick % period_ticks == 0`.

---

## §6 Effect statements

### 6.1 Modifier slots

Ten slots, evaluated left-to-right at parse time, runtime-evaluated
in this order:

1. `verb` + args — mandatory
2. `in <shape_expr>` — scope
3. `[TAG: value, …]` — power ratings
4. `for <duration>` — effect duration
5. `when <cond_expr> [else <eff>]` — conditional gate
6. `chance <p>` — probabilistic gate
7. `stacking <mode>` — stack/refresh/extend
8. `+ N% <stat_ref>` — scaling terms
9. `until_caster_dies` / `damageable_hp <n>` — voxel lifetime
10. `{ … }` — nested effects (verb opts in)

### 6.2 GPU lowering

Each effect_stmt becomes a fragment in the chronicle handler kernel.
Modifiers compile to inline WGSL:

- `when`/`chance` — early-exit branch at top of fragment
- `in <shape>` — scoped fan-out (per-target-in-shape loop unrolling)
- `+ N% stat` — inline arithmetic on the effect's primary scalar
- `for <duration>` — write to per-target status SoA field with expiry tick
- `stacking` — applied at chronicle-handler time, max-wins by default
- `[TAG: value]` — compile-time aggregation into AbilityProgram.tags

### 6.3 Cap

`MAX_EFFECTS_PER_PROGRAM = 4`. Authored abilities exceeding this →
lower-error. Voxel ops count against the same 4.

---

## §7 Effect catalog

~80 verbs across 9 categories. Each entry: verb signature → IR
mapping → GPU lowering note → status.

### 7.1 Combat core (5 verbs, runs-today)

| Verb | Args | EffectOp | GPU lowering |
|---|---|---|---|
| `damage` | `f32` | `Damage { amount }` | inline `agent_hp[t] = max(0, agent_hp[t] - amount)` in chronicle kernel |
| `heal` | `f32` | `Heal { amount }` | inline `agent_hp[t] = min(max_hp[t], agent_hp[t] + amount)` |
| `shield` | `f32` | `Shield { amount }` | inline `agent_shield_hp[t] += amount` |
| `stun` | duration | `Stun { duration_ticks }` | inline `agent_stun_expires_at_tick[t] = max(current, tick + dur)` |
| `slow` | `f32`, dur | `Slow { duration_ticks, factor_q8 }` | inline write to slow-expires + slow-factor SoA pair |

### 7.2 Control (10 verbs)

| Verb | Status | New EffectOp |
|---|---|---|
| `root` | planned | Root |
| `silence` | planned | Silence |
| `fear` | planned | Fear |
| `taunt` | planned | EmitEvent { EngagementCommitted } |
| `charm` | reserved | (Phase 2 TOM dependency) |
| `polymorph` | reserved | Polymorph + shape override |
| `banish` | reserved | Banish (target removed from sim) |
| `confuse` | reserved | intent randomization |
| `suppress` | reserved | ability-use block |
| `grounded` | reserved | MovementMode override |

GPU lowering: each becomes an inline write to a per-agent status SoA
field. `agent_root_expires_at_tick`, `agent_silence_expires_at_tick`,
etc.

### 7.3 Movement (7 verbs, all `planned`)

`dash` (with `to_target`/`to_position` variants) / `blink` /
`knockback` / `pull` / `swap`.

GPU lowering: emit a `Moved` event into the per-tick ring; an
`ApplyMovement` chronicle handler integrates position. Position writes
live in the existing physics path.

### 7.4 Buffs / Debuffs (5 verbs, mostly planned)

| Verb | Status |
|---|---|
| `buff` | planned |
| `debuff` | planned |
| `damage_modify` | planned |
| `lifesteal` | planned |
| `reflect` | reserved |
| `blind` | reserved |

GPU lowering: persistent per-agent multiplier SoA fields with expiry
ticks. `agent_damage_taken_mult`, `agent_lifesteal_frac`, etc.

### 7.5 Advanced / narrative (24 verbs, mostly reserved)

`summon`, `stealth`, `leash`, `link`, `redirect`, `rewind`,
`cooldown_modify`, `apply_stacks`, `execute`, `self_damage`, `dispel`,
`immunity`, `death_mark`, `resurrect`, `overheal_shield`,
`absorb_to_heal`, `shield_steal`, `status_clone`, `detonate`,
`status_transfer`, `on_hit_buff`, `obstacle`, `projectile_block`,
`attach`, `evolve_ability`, `command_summons`.

Status: `execute`, `self_damage`, `lifesteal`, `damage_modify` →
planned. Rest → reserved. GPU lowering varies; most reduce to existing
EffectOp variants (Damage / Heal) plus per-ability state.

### 7.6 Voxel (4 verbs, planned, depend on voxel storage)

| Verb | EffectOp | GPU lowering |
|---|---|---|
| `place_voxels <mat> in <shape>` | PlaceVoxels { mask, material, lifetime, damageable_hp } | per-cell scatter into voxel storage; epoch bump |
| `harvest_voxels in <shape> [drop_mode]` | HarvestVoxels { mask, drop_mode } | per-cell read+clear; emit drop events |
| `transform_voxels <from> -> <to> in <shape>` | TransformVoxels { mask, from, to, lifetime } | per-cell conditional write |
| `place_structure <name>(args)` | PlaceStructure { structure_id, lifetime } | bulk scatter from pre-rasterized structure mask |

These lower to GPU compute kernels operating on voxel chunk buffers.
Modular dependency: the voxel storage subsystem (L0-L6 tiered, per-chunk
write epoch) must land first.

### 7.7 Wealth / economy (10 verbs, all planned)

| Verb | EffectOp ordinal | GPU lowering |
|---|---|---|
| `transfer_gold <i32>` | 5 (TransferGold) | inline atomic add on `agent_gold[caster/target]` |
| `modify_standing <i16>` | 6 (ModifyStanding, evolved) | atomic add on `view_storage[standing_view][caster*N+target]` |
| `consume <commodity> <n>` (recipe body) | 17 (Recipe — bundled) | inline atomic decr on `agent_inventory[caster][commodity]` |
| `produce <commodity> <n> quality(<formula>)` (recipe body) | 17 (Recipe — bundled) | inline atomic incr on `agent_inventory[caster][commodity]`; quality write to per-item SoA |
| `wear_tool <kind> <amount>` | 18 (WearTool) | inline incr on `tool_wear[tool_id]`; check vs durability |
| `transfer_property <pid> <target>` | 19 (TransferProperty) | atomic write `property_owner[pid] = target_agent_id` |
| `pickpocket / demand` | 20 (ForcibleTransfer) | contested: emit event + per-tick contest resolution kernel |
| `create_obligation` | 21 (CreateObligation) | atomic alloc in obligation registry SoA |
| `discharge_obligation / default_obligation` | 22 / 23 | atomic state write on `obligation[oid]` |
| `establish_route <from> <to>` | 24 (EstablishRoute) | append to route registry |
| `join_caravan <caravan_id>` | 25 (JoinCaravan) | write `agent_caravan[caster] = caravan_id` |
| `transfer_obligation <oid> <target>` | 26 (TransferObligation) | atomic write `obligation_holder[oid] = target_agent_id` |

The `consume` and `produce` clauses inside a recipe body are NOT
standalone effect verbs — they're the recipe's input/output declaration.
EffectOp::Recipe carries the recipe_id and the GPU side reads
`recipe_registry[recipe_id]` for the actual i/o.

### 7.8 AI-state writes (~50 verbs across 6 sub-areas)

#### 7.8.1 H — Standing / relationships (7 verbs)

Existing `ModifyStanding` evolves to `{ a_sel: PairSelector, b_sel:
PairSelector, delta: i16 }`. Schema hash bump.

| Verb | Selector pattern | GPU lowering |
|---|---|---|
| `befriend <n>` | (Caster, Target, +n) | atomic add to standing view |
| `enmity <n>` | (Caster, Target, -n) | same |
| `endear <n>` | (Target, Caster, +n) | same |
| `charm <n> for <dur>` | (Caster, Target, -n) + revert queue entry | atomic add + revert SoA write |
| `duel_challenge` | (Caster, Target, MIN) | same |
| `slander <third> by <n>` | (Target, OtherAgent(third), -n) | same |
| `rally_cry <n> in <vol>` | N × (Caster, AgentInVolume, +n) | per-target loop in chronicle kernel |

#### 7.8.2 J — Theory of Mind (Phase 1 — 7 verbs, planned)

New EffectOps:
```
ClearBelief    { observer_sel, subject_sel } = 12
PlantBelief    { observer_sel, subject_sel, fields: BeliefFieldMask + payload } = 13
RefreshBelief  { observer_sel, subject_sel } = 14
```

GPU lowering: each writes to `cold_beliefs[observer]` BoundedMap. The
BoundedMap is a fixed-size per-agent SoA (8 entries × ~40 bytes =
320B/agent) so the GPU side can index into it with a linear scan.

| Verb | Maps to |
|---|---|
| `scry <target>` | RefreshBelief { Caster, Target } — copies ground truth |
| `reveal <target> in <vol>` | bulk RefreshBelief over observers in volume |
| `stealth for <dur>` | bulk PlantBelief setting `confidence: 0` for caster across observers |
| `disguise as <ct> for <dur>` | bulk PlantBelief setting `last_known_creature_type` |
| `decoy at <pos>` | bulk PlantBelief with synthetic `last_known_pos` |
| `erase_belief <target> of <subj>` | ClearBelief { Target, Subject } |
| `plant_belief <target> of <subj> { … }` | PlantBelief with caller-chosen fields |

Phase 2 TOM extensions (`teach_domain` / `forge_testimony` /
`spread_rumor` / etc.) are reserved.

#### 7.8.3 M — Communication (10 verbs, planned)

`silence`, `mute`, `jam_channel`, `translate`, `comprehend_languages`,
`amplify_voice`, `grant_telepathy`, `speak_as`, `impersonate`,
`forge_document`.

GPU lowering: per-agent capability bitset writes + event-attribution
field rewrites. Some (`speak_as`, `impersonate`, `forge_document`)
require new fields on `AgentCommunicated` and `Document` — schema
hash bumps, reserved.

#### 7.8.4 K — Groups / factions (12 verbs, all reserved)

`induct`, `exile`, `defect`, `recruit`, `crown`, `abdicate`, `depose`,
`found_group`, `declare_war`, `form_alliance`, `break_treaty`,
`send_tribute`.

GPU lowering: writes to `cold_memberships` SoA (per-agent SortedVec
of GroupId). Depends on `Group` instance-data path landing first
(referenced as Plan 1 T16 in original spec).

#### 7.8.5 L — Quest valuation (10 verbs, all reserved)

`inflate_bounty`, `deflate_bounty`, `intimidate_takers`,
`embolden_takers`, `urgent_call`, `false_lull`, `endorse_quest`,
`slander_quest`, `broadcast_quest`, `conceal_quest`.

GPU lowering: writes to `quest_beliefs[observer][quest_id]` BoundedMap
field. Mirrors agent-belief shape with `QuestBeliefState` payload
(believed_reward, believed_difficulty, believed_urgency,
trust_in_poster, confidence). Depends on Phase 2 TOM extended to quest
subjects.

#### 7.8.6 N — Engagement / coordination (10 verbs, planned)

| Verb | Event emitted |
|---|---|
| `force_engage <target> with <other>` | `EngagementCommitted { actor: target, engaged_with: other }` |
| `break_engagement <target>` | `EngagementBroken { actor: target }` |
| `taunt <target>` | `EngagementCommitted { actor: target, engaged_with: caster }` |
| `scatter [in <vol>]` | bulk `EngagementBroken` per agent in volume |
| `set_pack_focus <target> for <obs>` | `PackAssist { observer: obs, target }` |
| `rally_call <wounded_kin>` | `RallyCall { observer: caster, wounded_kin }` |
| `rally_cry in <vol> around <wounded>` | bulk `RallyCall` |
| `pack_call <target> in <vol>` | bulk `PackAssist` |
| `incite_fear <toward> for <obs>` | `FearSpread { observer, dead_kin: toward }` |
| `panic_wave in <vol>` | bulk `FearSpread` |

GPU lowering: each lowers to a single `EmitEvent` op (variant 15) with
a `PayloadSel::Preset(<event_kind>)` discriminator. The chronicle
fold rules in `views.sim` already exist — no new fold logic.

### 7.9 Meta (1 verb, runs-today)

| Verb | EffectOp | GPU lowering |
|---|---|---|
| `cast <ability_name>` | 7 (CastAbility { ability, selector }) | recursive dispatch through ability registry; bounded by `MAX_CASCADE_ITERATIONS = 8` |

GPU lowering recurses through the same chronicle handler dispatcher;
the recursion budget is enforced by a per-cast depth counter SoA field.

---

## §8 Shapes (12 primitives + CSG)

### 8.1 Disc family (5)

| Shape | Grammar | Default thickness |
|---|---|---|
| `circle` | `circle(r)` | 1 voxel |
| `cone` | `cone(r, angle_deg)` | 1 voxel |
| `line` | `line(len, width)` | 1 voxel |
| `ring` | `ring(inner, outer)` | 1 voxel |
| `spread` | `spread(r, max_targets)` | 1 voxel |

`circle(3.0) thickness 2` overrides default thickness.

### 8.2 Volume family (7)

| Shape | Grammar |
|---|---|
| `box` | `box(wx, wy, wz)` |
| `sphere` | `sphere(r)` |
| `column` | `column(r, h)` |
| `wall` | `wall(len, h, thick) [facing: deg]` |
| `cylinder` | `cylinder(r, h)` |
| `dome` | `dome(r)` |
| `hull` | `hull(r)` (castle footprint) |

### 8.3 CSG composition

`union`, `diff`, `intersect`. Standard precedence: intersect > diff > union.

```
sphere(5) diff sphere(4)                        # thick shell
(sphere(5) diff sphere(4)) union column(1, 8)   # shell + pillar
```

### 8.4 GPU lowering

CSG is **rasterized at compile time** to a fixed 3D bitmask stored in
`VoxelMaskRegistry`. Schema hash covers the rasterized bitmask, not
the CSG expression.

Per-cast voxel rasterization: lookup `voxel_mask_registry[mask_id]`
(GPU-resident SoA buffer) and iterate cells. Budget cap:
`MAX_VOXELS_PER_ABILITY = 16384`.

For non-voxel ops (AoE damage, AoE buff), the shape rasterizes to a
**target list** at cast resolution time: enumerate agents within the
shape's bbox, filter by per-agent inclusion test (point-in-cylinder,
etc.). The target list feeds the existing PerPair dispatch shape.

### 8.5 Orientation + grid snapping

- Default orientation: caster→target vector projected horizontally
- Explicit `facing: <degrees>` on `wall`, `cylinder`, `box`
- Rotation snaps to {0, 90, 180, 270} degrees
- Shape origins snap to voxel grid (default cell = 1.0m)
- Floats accepted in shape args; rasterization rounds to nearest cell

---

## §9 Delivery methods (6)

`Delivery::Instant` is the only method shipping today. The other 5
are planned:

| Method | Params | Hooks | GPU lowering |
|---|---|---|---|
| `projectile` | speed, width, pierce | on_hit, on_arrival | per-tick projectile-state SoA + collision sweep |
| `chain` | bounces, range, falloff | on_hit | recursive target select; bounded by `bounces` |
| `zone` | duration, tick | on_hit, on_tick, on_expire | persistent zone SoA + per-tick PerAgent dispatch |
| `channel` | duration, tick | on_tick, on_expire | per-tick PerAgent dispatch on caster + interrupt gate |
| `tether` | max_range, tick | on_tick, on_complete | per-tick distance check between caster + target |
| `trap` | duration, trigger_radius, arm_time | on_trigger | persistent trap SoA + spatial body-form trigger check |

All deliveries lower to **persistent SoA buffers** with per-tick
dispatch shapes. Hooks compile to chronicle handlers gated on the
hook's event kind (HitEvent, TickEvent, ExpireEvent).

---

## §10 Conditions (~80 atoms across 5 families)

### 10.1 Cost classes

Every atom carries a cost class. Authors order expensive atoms last
(short-circuit evaluation, left-to-right).

| Class | Examples | GPU cost |
|---|---|---|
| 0 — O(1) scalar | `target_hp_below(X)`, `standing()`, `engaged_with()` | inline branch on agent SoA read |
| 1 — O(K) top-K scan | `pack_focus_of`, `kin_fear`, `rally_boost_of` | K=8 SoA scan |
| 2 — O(M) memory ring | `target_spoke_within()`, `remembers()` | 64+5 entry scan |
| 3 — O(K²) cross-ref | second-order beliefs | reserved |

### 10.2 Atom families

**Physical state:** `target_hp_below(X)`, `caster_hp_above(X)`,
`target_status(stun)`, `target_alive`, `caster_engaged`, `distance_below(X)`,
`facing_target`, etc.

**Voxel-aware (8 single-voxel + 4 volume aggregate):** `floor_material == X`,
`floor_is(<prop>)`, `above_material == X`, `voxel_at(dx, dy, dz) == X`,
`occupied(dx, dy, dz)`, `depth_below_surface < n`,
`any_voxel_in(<shape>, X)`, `voxel_count(<shape>, X) > N`.

**AI-state:** `standing_below(X)`, `target_confidence_below(X)`,
`believes(<target>, <subject>).<field> <op> <val>`, `is_member_of(<group>)`,
`pack_focus_of(observer) == target`, etc.

**Tag-based:** `has_tag(target, FIRE)`, `caster_tagged(<tag>)`.

**Temporal:** `tick_since_last_cast > N`, `time_of_day in [dusk, night]`.

### 10.3 Reference frame

Default reference is determined by ability `target:`:
- `enemy` / `ally` → target's voxel
- `self` / `self_aoe` → caster's voxel
- `ground` → targeted position

Explicit prefixes: `target_floor_material == X`, `caster_floor_material == X`,
`cast_floor_material == X`.

### 10.4 GPU lowering

Each cond_atom compiles to inline WGSL boolean expression. Compound
conditions (`and`/`or`/`not`) compile to short-circuit branches.
Cost-class budgeting is a compiler hint, not a runtime check.

---

## §11 Templates (parameterized abilities)

```
template ElementalBolt(element: Material = fire, radius: float = 3.0) {
    damage 50 in circle($radius) [ELEMENTAL: 60] when material_is($element)
}

ability Fireball : ElementalBolt(fire, 4.0) {
    target: ground, range: 8.0
    cooldown: 6s
}
```

### 11.1 Typed parameters

```
template_param = IDENT [ ":" type_name [ "=" default_val ] ] ;
type_name      = "int" | "float" | "bool" | "Material" | "Structure" ;
```

Unbound non-default params are required at instantiation.

### 11.2 Template arithmetic

`+ - * / %` plus `min`, `max`, `clamp`, `abs`. Used inside structure
bodies for parameter-driven sizing. Compile-time evaluation only.

### 11.3 GPU lowering

Templates expand at compile time. No runtime cost. Recursion bounded
by `template expansion depth = 16`.

---

## §12 Structure blueprints (voxel templates)

```
structure Castle(wall_mat: Material = stone, height: int = 8) {
    bounds: box(20, $height, 20)
    origin: (0, 0, 0)
    rotatable
    symmetry: radial(4)

    place $wall_mat in box(20, 1, 20)                      # floor
    place $wall_mat in (box(20, $height, 20) diff box(18, $height, 18))  # walls
    if $height > 4 { place $wall_mat in box(20, 1, 20) at (0, $height, 0) }  # roof
}
```

### 12.1 Structure body

| Statement | Purpose |
|---|---|
| `place <mat> in <shape>` | stamp material |
| `harvest in <shape>` | remove voxels |
| `transform <from> -> <to> in <shape>` | conditional transform |
| `include <other_structure>(args)` | nested composition |
| `if <cond> { … } [else { … }]` | conditional placement |

### 12.2 GPU lowering

Structures rasterize at compile time to per-parameter-instantiation
3D bitmask + material assignment in `StructureRegistry`. Runtime
`place_structure Castle(stone, 8)` becomes a single bulk scatter op.

Recursion depth: `structure include depth = 16`.

---

## §13 Materials

Declared in `.sim` (NOT `.ability`), referenced everywhere.

```
// In materials.sim
material stone     { hardness: 0.8, density: 2.5, walkable: true }
material wood      { hardness: 0.3, density: 0.6, walkable: true, flammable: true }
material lava      { hardness: 0.0, density: 3.0, walkable: false, damage_per_tick: 5 }
```

### 13.1 GPU lowering

Materials compile to a u8 enum with a parallel SoA buffer of material
properties:

```
material_registry: SoA {
    hardness: [f32; N_MATERIALS]
    density:  [f32; N_MATERIALS]
    walkable: [u32; N_MATERIALS]  // bool packed
    flammable: [u32; N_MATERIALS]
    damage_per_tick: [f32; N_MATERIALS]
}
```

Properties are accessed by index in the GPU code.

---

## §14 EffectOp catalog (target: 27 variants)

Schema-hash-stable ordinal table. Existing variants in `engine` are
0-7; new variants reserved at design time so authored abilities don't
churn the hash incrementally.

### 14.1 Existing (0-7, runs-today)

```rust
Damage         { amount: f32 }                                    = 0
Heal           { amount: f32 }                                    = 1
Shield         { amount: f32 }                                    = 2
Stun           { duration_ticks: u32 }                            = 3
Slow           { duration_ticks: u32, factor_q8: i16 }            = 4
TransferGold   { amount: i32 }                                    = 5
ModifyStanding { a_sel: PairSel, b_sel: PairSel, delta: i16 }     = 6  // EVOLVED
CastAbility    { ability: AbilityId, selector: TargetSel }        = 7
```

### 14.2 Voxel (8-11, planned)

```rust
PlaceVoxels      { mask: VoxelMaskId, material: MaterialId, lifetime: u32, damageable_hp: u16 } = 8
HarvestVoxels    { mask: VoxelMaskId, drop: DropMode }                                          = 9
TransformVoxels  { mask: VoxelMaskId, from: MaterialId, to: MaterialId, lifetime: u32 }         = 10
PlaceStructure   { structure: StructureId, lifetime: u32 }                                      = 11
```

### 14.3 Theory of Mind (12-14, planned)

```rust
ClearBelief    { observer_sel: AgentSel, subject_sel: SubjectSel }                          = 12
PlantBelief    { observer_sel: AgentSel, subject_sel: SubjectSel, fields: BeliefFieldMask + payload } = 13
RefreshBelief  { observer_sel: AgentSel, subject_sel: SubjectSel }                          = 14
```

### 14.4 Generic event emit (15-16, planned)

```rust
EmitEvent           { kind: EventKindId, payload_sel: PayloadSel }       = 15
ModifyGroupStanding { group_sel: GroupSel, other_group: GroupId, delta: i16 } = 16
```

`EmitEvent` is the escape hatch for verbs whose payload is structured
event data already folded by existing `@materialized` views in
`views.sim`.

### 14.5 Economy (17-26, planned)

```rust
Recipe              { recipe: RecipeId, target_tool_sel: ToolSel }    = 17
WearTool            { tool_kind: ToolKindId, amount: f32 }            = 18
TransferProperty    { property_id, target_sel: AgentSel }             = 19
ForcibleTransfer    { subject, target_sel, contest_kind, threshold }  = 20
CreateObligation    { kind, parties, terms }                          = 21
DischargeObligation { obligation_id }                                 = 22
DefaultObligation   { obligation_id }                                 = 23
EstablishRoute      { from, to }                                      = 24
JoinCaravan         { caravan: GroupId }                              = 25
TransferObligation  { obligation_id, target_sel: AgentSel }           = 26
```

### 14.6 Size budget

Every variant ≤ 16 bytes after Rust enum tagging (P4). Verified at
build time via `static_assert::const_assert!(size_of::<EffectOp>() <= 16)`.

`Recipe` is the tightest: needs `RecipeId` (u16) + `ToolSel` enum
(probably u32 with discriminant). Total payload 6-8 bytes + tag.

---

## §15 Selectors

Selectors are inline payloads inside EffectOps. They name a target
role; the GPU side resolves them to an `AgentId` at dispatch time.

```rust
enum PairSelector {
    Caster,
    Target,
    OtherAgent(AgentId),  // compile-time named
}

enum AgentSel {
    Caster,
    Target,
    OtherAgent(AgentId),
}

enum SubjectSel {
    Agent(AgentSel),
    Quest(QuestSel),
}

enum QuestSel {
    ThisQuest,           // quest in ground target
    NamedQuest(QuestId),
}

enum GroupSel {
    CasterGroup,         // first membership of caster
    TargetGroup,
    NamedGroup(GroupId),
}

enum PayloadSel {
    Preset(u8),          // baked payload per verb
    FromScope,           // event fields from cast scope
}

enum ToolSel {
    OwnedNearest,        // caster's tool of matching kind
    OwnedSpecific(ToolId),
    PartySharedNearest,  // group-owned tool
}

enum DropMode {
    Destroy,
    DropAs(ItemKind),
    ReturnToCaster,
}
```

### 15.1 GPU lowering

Selectors compile to inline index expressions in the chronicle kernel:

```wgsl
// EffectOp::Damage { amount: 50 } with target_sel = Target:
let target_id = cast_target_id;  // already passed in cast packet
agent_hp[target_id] = max(0.0, agent_hp[target_id] - 50.0);

// EffectOp::ModifyStanding { a_sel: Target, b_sel: OtherAgent(third) }:
let a = cast_target_id;
let b = third_agent_id_const;
atomicAdd(&standing_view[a * N + b], delta);
```

`OtherAgent(AgentId)` and `NamedQuest(QuestId)` etc. are compile-time
constants — no runtime lookup.

---

## §16 Registries (10 total)

All registries are GPU-resident SoA buffers built at compile time and
loaded at engine init. Schema hash covers their layouts.

| Registry | Indexed by | Stores | Source |
|---|---|---|---|
| `MaterialRegistry` | MaterialId (u8) | properties (hardness, density, …) | .sim materials.sim |
| `AbilityRegistry` | AbilityId (u16) | AbilityProgram (4-effect SmallVec, gate, hint, tags) | .ability files |
| `VoxelMaskRegistry` | VoxelMaskId (u16) | rasterized 3D bitmask + bbox + cell count | compile-time CSG eval |
| `StructureRegistry` | StructureId (u16) | per-instantiation rasterized mask + material assignments | compile-time structure expansion |
| `RecipeRegistry` | RecipeId (u16) | inputs, outputs, duration, skill, tool, quality formula | .sim recipes |
| `ToolKindRegistry` | ToolKindId (u8) | properties (slot, durability_base, repair_recipes) | .sim tools |
| `SkillRegistry` | SkillId (u8) | properties (decay_rate, training_method) | .sim skills |
| `CommodityRegistry` | CommodityId (u16) | properties (base_value, perishable, weight, group) | .sim commodities |
| `RouteRegistry` | RouteId (u16) | from/to/distance/risk_baseline | runtime + .sim seed |
| `ComponentKindRegistry` | ComponentKindId (u8) | properties for component-based assembly | .sim components |

### 16.1 Compile-time vs runtime partition

Each registry partitions: `[1..N_authored]` for compile-time entries
(part of schema hash) and `[N_authored+1..MAX]` for runtime-generated
(recorded in trace, NOT part of hash).

Runtime-generated entries support the §14 runtime ability composition
model (out of scope for this spec, see original §14). Schema-hash
exemption is required because the model can mint new abilities mid-run.

### 16.2 GPU lowering

Each registry is a struct-of-arrays SoA buffer:

```rust
struct AbilityRegistry {
    gate:           Vec<Gate>,           // [N_abilities]
    delivery_idx:   Vec<u8>,             // [N_abilities] enum tag
    delivery_param: Vec<DeliveryParams>, // [N_abilities] union
    area_idx:       Vec<u8>,             // [N_abilities] enum tag
    area_param:     Vec<AreaParams>,     // [N_abilities] union
    effects:        Vec<[EffectOp; 4]>,  // [N_abilities] fixed-size array
    effect_count:   Vec<u8>,             // [N_abilities] actual count <= 4
    hint:           Vec<Option<AbilityHint>>,
    tags:           Vec<[(AbilityTag, f32); 6]>,
}
```

GPU code reads via `ability_registry.effects[ability_id][i]` etc.

---

## §17 Per-agent SoA additions

The biggest architectural challenge: the spec demands per-agent state
that doesn't fit the existing fixed-size scalar SoA pattern.

### 17.1 Existing (fixed-size scalars)

The current AgentFieldId enum has ~30 variants — all fixed-size
scalars or vec3 (Pos, Hp, MaxHp, Alive, Mana, Hunger, Thirst, Level,
CreatureType, etc.).

### 17.2 New fixed-size additions

| Field | Type | Cost |
|---|---|---|
| `cast_in_progress_ability` | Option<AbilityId> | 4 B/agent |
| `cast_resolves_at_tick` | u32 | 4 B/agent |
| `cast_target_id` | AgentId | 4 B/agent |
| `stun_expires_at_tick` | u32 | 4 B/agent (already exists) |
| `slow_expires_at_tick` | u32 | 4 B/agent (already exists) |
| `slow_factor_q8` | i16 | 2 B/agent (already exists) |
| `root_expires_at_tick` | u32 | 4 B/agent NEW |
| `silence_expires_at_tick` | u32 | 4 B/agent NEW |
| `fear_expires_at_tick` | u32 | 4 B/agent NEW |
| `damage_taken_mult` | f32 | 4 B/agent NEW |
| `lifesteal_frac` | f32 | 4 B/agent NEW |
| `cooldown_next_ready_tick[ability_slot]` | [u32; 8] | 32 B/agent (per-ability cooldown) |
| `charges_remaining[ability_slot]` | [u8; 8] | 8 B/agent |
| `gold` | u32 | 4 B/agent |

Total fixed-size additions: ~80 B/agent.

### 17.3 Variable-size per-agent fields (NEW ARCHITECTURAL PRIMITIVE)

The hard part. Spec demands:

| Field | Shape | Purpose |
|---|---|---|
| `inventory` | `SmallVec<(CommodityId, u16), 16>` | per-agent commodity holdings |
| `tools_owned` | `SmallVec<(ToolId, u16), 4>` | per-agent tool ownership |
| `cold_beliefs` | `BoundedMap<AgentId, BeliefState, 8>` | per-agent agent beliefs (320 B/agent) |
| `cold_price_beliefs` | `SortedVec<(CommodityId × SettlementCluster, PriceBelief), 16>` | per-agent price beliefs |
| `cold_merchant_beliefs` | `SortedVec<(AgentId, MerchantBelief), 8>` | per-agent merchant beliefs |
| `cold_route_beliefs` | `SortedVec<(RouteId, RouteBelief), 8>` | per-agent route beliefs |
| `cold_quest_beliefs` | `SortedVec<(QuestId, QuestBeliefState), 8>` | per-agent quest beliefs |
| `cold_memberships` | `SortedVec<Membership, 8>` | per-agent group memberships |
| `cold_skills` | `SortedVec<(SkillId, f32), 16>` | per-agent skill levels |

GPU lowering challenge: GPU SoA wants fixed-size strides per agent.
Variable-size per-agent fields require either:

**Option A: Bounded arrays.** Pick max size per field at compile time
(e.g. `MAX_INVENTORY = 16`). Agents always allocate the max. Wastes
memory but indexable. ~5 KB/agent worst case.

**Option B: Indirect indexing.** Per-agent field stores `(start_offset,
count)` into a global pool. Pool is one giant flat array, written
by an allocator kernel each tick. Saves memory but requires
allocator coordination.

**Option C: Sparse SoA.** Hash table keyed by `(agent_id, item_kind)`.
Lookup cost O(1) average, O(N) worst. Allows arbitrary per-agent
size at the cost of bucket-search latency.

This spec PICKS Option A for the first cut (simplest GPU lowering,
matches existing engine SoA pattern). Memory cost: ~5 KB/agent ×
200K agents = 1 GB (acceptable on modern GPUs). Migration to B or C
deferred until profiling shows it's needed.

### 17.4 Schema hash + snapshot

All fixed-size additions bump the schema hash. Variable-size additions
also bump (new field shape). One coordinated bump per implementation
phase rather than per-field — minimizes parity-baseline churn.

---

## §18 Two-phase epistemic split

### 18.1 The rule

Decision-time code (anything inside a `mask`, `scoring`, or `verb`
selection body) reads only from belief-mediated views. Execution-time
code (`physics` rules resolving chosen intents) reads ground truth.

### 18.2 What's allowed at decision-time

- `view::*` lookups over the agent's own materialized views
- `agent_<field>(self)` and `agent_<field>(target)` for fields the
  agent can know about itself / its target via observation
- `world.tick`, `world.seed`, `config.*`
- Math + comparison + logical operators

### 18.3 What's allowed at execution-time

The above PLUS:

- `engine::voxel_at(world_pos)`
- `engine::walkable(world_pos, mode)`
- `engine::region_at(world_pos)`
- `engine::raycast(origin, dir, max)`
- `engine::resolve_movement(agent, intent)`
- `engine::can_observe(observer, target)`
- `engine::emit_to_observable_agents(template, source_pos, kind, filter)`
- `engine::emit_via_propagation(template, source_pos, propagation_kind, intensity)`
- `engine::apply_voxel_modification(region, op)`

### 18.4 Compile-time enforcement

The CG validator pass walks the lowered IR. Any expression that
resolves to a forbidden API call AND is downstream of a MaskPredicate
or ScoringArgmax op → compile error:

```
error: decision-time code may not call ground-truth queries
       — use a `view` over emitted events to surface this state
       at scoring/mask time
note: occurred in mask predicate of verb 'Hunt' at line 42
```

### 18.5 Belief production

Beliefs are materialized views over emission events. Engine emits
events with epistemically appropriate gating (LOS, range, propagation);
DSL declares folds:

```
@materialized(on_event = [Saw, HeardSound, SmelledScent])
view known_agents(observer: Agent, subject: Agent) -> BeliefState {
    initial: BeliefState::default(),
    on Saw { observer: o, target: s, position: p, tick: t }
        when (o == observer && s == subject) {
        self = BeliefState {
            last_known_pos: p,
            last_known_hp: agents.hp(s),
            last_updated_tick: t,
            confidence: 1.0,
            ...
        }
    }
    on HeardSound { listener: o, ... } when (o == observer) {
        // weaker update — only direction info
        self.last_known_pos = approx_pos_from_direction;
        self.confidence = 0.3;
    }
    decay: per_tick (multiply confidence by 0.98),
    evict: when (confidence < 0.05),
}
```

Standardized `BeliefState` struct payload, pluggable per-domain
(`PriceBelief`, `MerchantBelief`, etc.).

---

## §19 Voxel storage subsystem (referenced)

This spec REFERENCES the voxel storage design (see source spec
2026-04-25). Summary:

### 19.1 Tiered storage (L0-L6)

- L0: per-agent / per-ray result cache (game-owned)
- L1: per-chunk summaries (uniform flag, max_z, material histogram)
- L2: LOD pyramid (mips)
- L3: HOT primary storage — dense 64³ chunks
- L4: WARM primary — palette + RLE OR 8³ brick pool
- L5: COLD primary — SVDAG (deferred)
- L6: DISK or regen-from-worldgen-seed

### 19.2 Per-chunk write epoch

Every chunk slot carries a monotonic u64 write epoch. Voxel writes bump
the epoch. Every derived structure (summaries, mips, indices, render
caches) records the epoch it was built against. Stale epoch triggers
rebuild.

### 19.3 Region registry + indices

Game-owned named volumes ("the city", "the swamp"). DSL-declared
bounded indices over them (e.g. "all agents in this region").

### 19.4 New events

- `MovementBlocked` — Phase 1 schema bump
- `VoxelRegionEntered` / `VoxelRegionLeft` — Phase 2 schema bump
- `Saw` / `HeardSound` / `SmelledScent` / `FeltImpact` /
  `VoxelRegionObservedAtDistance` — Phase 3 schema bump

These feed the belief views described in §18.5.

### 19.5 Status

Voxel storage is its own subsystem with multi-month scope. This spec
assumes it lands as an independent track. Voxel ops in the ability DSL
(§7.6, §8, §12) are blocked on it.

---

## §20 Kernel dispatch architecture

### 20.1 Crate split

```
crates/dsl_compiler/      ← reads .sim/.ability → emits both Rust + WGSL
crates/engine_gpu_rules/  ← committed // GENERATED Rust + WGSL files (NEW)
crates/engine_gpu/        ← hand-written GPU primitives only
                            (wgpu setup, megakernel orchestration, BufferPool)
```

The cut: `engine_gpu/` owns *resources and orchestration*;
`engine_gpu_rules/` owns *kernel-specific code*. Anything per-kernel
(BGL shape, WGSL body, struct layout, dispatch encoding, dependency
graph entry) moves to `engine_gpu_rules/`.

### 20.2 Five lifetime classes

Every emitted buffer falls into one. Mapping is fully inferable from
DSL row type — no annotation:

| Class | Owner | Lifetime | Reset cadence |
|---|---|---|---|
| Transient | BufferPool (engine_gpu) | one dispatch | recycled per tick |
| Resident | ResidentPathContext | batch lifetime | persists across ticks |
| PingPong | CascadeResidentCtx | one cascade iteration | A/B alternates per iter |
| External | engine consumer | passed-in handle | unmanaged by kernel |
| Pooled | shape-keyed pool | reused across compatible kernels | LRU eviction |

### 20.3 Row type → lifetime mapping

| DSL row type | Output buffer | Notable inputs |
|---|---|---|
| `mask` predicate | Transient (per-tick mask buf) | External (agent SoA, sim_cfg) |
| `scoring` (target_bound) | Transient (action buf) | External + Resident (views, cooldowns) |
| `scoring` (per_ability) | Resident (chosen_ability_buf) | External + Resident |
| `view` declaration | Resident (view_storage) | External + PingPong (event ring) |
| `physics` rule | PingPong (next-iter event ring) | External + PingPong (current-iter ring) |
| `apply` phase | Mutates External (agent SoA) + emits PingPong | Transient + Resident |
| `@spatial query` | Pooled | External |
| `ability` (NEW) | Resident (cast-in-progress + cooldown SoA) | External (AbilityRegistry) |
| `passive` (NEW) | PingPong (passive-fired event ring) | External + Resident |
| `recipe` (NEW) | Mutates External (agent inventory + tool wear) | External (RecipeRegistry) |

### 20.4 Compile-time emission

Per `.sim`/`.ability` row that produces a kernel, the compiler emits
**one Rust file + one WGSL file** in `engine_gpu_rules/src/`. Plus
shared infrastructure files.

```
xtask compile-dsl
   ↓
.sim + .ability files
   ↓
dsl_compiler walks IR per row family
   ↓
emit:
   ├─ Rust binding         → engine_gpu_rules/src/<kernel>.rs        (// GENERATED)
   ├─ WGSL                 → engine_gpu_rules/src/<kernel>.wgsl      (// GENERATED)
   ├─ Schedule             → engine_gpu_rules/src/schedule.rs        (// GENERATED)
   ├─ Resident ctx         → engine_gpu_rules/src/resident_context.rs (// GENERATED)
   ├─ PingPong ctx         → engine_gpu_rules/src/pingpong_context.rs (// GENERATED)
   └─ Megakernel           → engine_gpu_rules/src/megakernel.{rs,wgsl} (// GENERATED)
```

Both Rust and WGSL committed. Build is a pure compile step.

---

## §21 Unified status matrix

The previously-split status matrices in `ability.md §23` and
`economy.md §18` consolidated. Status is per-construct; `runs-today`
means the engine + GPU pipeline both work.

### 21.1 Top-level forms

| Construct | Status |
|---|---|
| `entity`, `event`, `view`, `physics`, `mask`, `verb`, `invariant`, `probe`, `metric`, `spatial_query` | runs-today |
| `material` declaration | planned |
| `recipe` declaration | planned |
| `template` declaration | planned |
| `structure` declaration | planned (depends on voxel storage) |
| `ability` block parsing | needs parser (does not exist) |
| `passive` block parsing | needs parser (does not exist) |

### 21.2 Ability header properties

| Property | Status |
|---|---|
| `target: enemy` / `self` | runs-today (via verb cascade) |
| `target: ally` / `self_aoe` / `ground` / `direction` | planned |
| `target: vector` / `global` | reserved |
| `range`, `cooldown`, `hint` | runs-today |
| `cast`, `charges`, `unstoppable`, `recharge` | planned |
| `cost`, `zone_tag`, `toggle`, `form`, `swap_form`, `recast`, `morph` | reserved |
| `require_skill`, `require_tool` | planned (need skill/tool subsystems) |

### 21.3 EffectOp variants

| Ordinals | Status |
|---|---|
| 0-7 (existing combat + meta) | runs-today |
| 8-11 (voxel) | reserved (depend on voxel storage) |
| 12-14 (TOM) | planned (depend on BeliefState SoA) |
| 15-16 (EmitEvent + ModifyGroupStanding) | planned (depend on group instance data) |
| 17-26 (economy) | planned (depend on inventory + skill + tool SoA) |

### 21.4 Effect verbs

| Category | Status |
|---|---|
| Combat core (5) | runs-today |
| Control (10) | planned except charm+ → reserved |
| Movement (7) | planned except swap → reserved |
| Buffs / debuffs (5) | planned except reflect/blind → reserved |
| Advanced (24) | reserved except execute, self_damage, lifesteal, damage_modify (planned) |
| Voxel (4) | reserved (depend on voxel storage) |
| Wealth/economy (10) | planned (depend on EffectOps 17-26) |
| AI-state H (standing, 7) | runs-today (befriend/enmity) + planned (rest) |
| AI-state J Phase 1 (TOM, 7) | planned |
| AI-state J Phase 2 (rumor/testimony) | reserved |
| AI-state M (communication, 10) | planned except impersonate/forge_document → reserved |
| AI-state K (groups, 12) | reserved (depend on Group instance data) |
| AI-state L (quest valuation, 10) | reserved (depend on Phase 2 TOM) |
| AI-state N (engagement, 10) | planned |
| Meta (cast) | runs-today |

### 21.5 Conditions

| Family | Status |
|---|---|
| Physical state | mixed runs-today / planned |
| Voxel-aware | reserved (depend on voxel storage) |
| AI-state | partial planned (depend on belief storage) |
| Tag-based | runs-today |
| Temporal | runs-today |

### 21.6 Cross-cutting infrastructure

| Subsystem | Status |
|---|---|
| BeliefState SoA + decay phase | needs implementation |
| Variable-size per-agent SoA (Inventory, beliefs) | architectural design needed |
| Voxel storage L0-L6 | needs implementation (huge) |
| Region registry + indices | needs implementation |
| Compiler-emitted kernel-wrapper crate (engine_gpu_rules/) | needs implementation |
| Two-phase epistemic split validator pass | needs implementation |
| Bitwise operators in DSL | needs implementation (~100 LOC) |
| Item-SoA field reads in physics bodies | needs implementation (~200 LOC) |
| Hero TOML binding to .ability files | needs implementation |

---

## §22 Budgets and determinism

### 22.1 Per-ability caps

| Constant | Value |
|---|---|
| `MAX_EFFECTS_PER_PROGRAM` | 4 |
| `MAX_TAGS_PER_PROGRAM` | 6 |
| `MAX_CASCADE_ITERATIONS` | 8 |
| `MAX_VOXELS_PER_ABILITY` | 16384 |

Exceeding any → lower-error.

### 22.2 Per-tick caps

| Constant | Value |
|---|---|
| `MAX_VOXELS_PER_TICK` | 65536 |
| `MAX_ABILITY_CASTS_PER_TICK` | 1024 |
| `MAX_PASSIVES_FIRED_PER_TICK` | 4096 |
| `MAX_RECIPES_RESOLVED_PER_TICK` | 1024 |

Overflow casts spill into next tick's queue, ordered by
`(cast_start_tick, cast_id)`.

### 22.3 Per-agent caps

| Constant | Value |
|---|---|
| `MAX_ABILITIES` per agent | 8 |
| `MAX_PASSIVES` per agent | 4 |
| `MAX_INVENTORY_SLOTS` | 16 |
| `MAX_TOOLS_OWNED` | 4 |
| `BELIEFS_PER_AGENT` (cold_beliefs cap) | 8 |
| `MAX_MEMBERSHIPS` | 8 |

### 22.4 Determinism contract

- **Mask rasterization:** bit-identical CPU and GPU (shared formula)
- **Structure compilation:** single-pass, declaration-order
- **Revert queue:** pops in (revert_tick, cast_id) order
- **Bulk event emission:** cast_id-ascending within a tick
- **Overlap rule:** later cast_id wins on conflicting voxel writes
- **World-extent clip:** silent (voxels outside extent dropped at
  rasterization)
- **Cascade depth:** hard bound, per-tick voxel budget is soft spill
- **Atomic reductions:** sort-then-fold for cross-backend bit-equality

### 22.5 Replay invalidation

Bumps schema hash + invalidates prior replays:
- Material catalog additions / edits
- Voxel mask CSG edits
- Structure body / parameter / symmetry edits
- EffectOp ordinal changes
- Selector enum changes
- Registry layout changes
- New event variants

---

## §23 Error model

### 23.1 Parse errors

Reported with file, line, column, span. Categories:

- Lexical: unknown token, bad duration (`5xs`), unterminated string
- Grammar: unexpected token at position, missing brace/paren
- Duplicate header: `duplicate 'cooldown:' in ability 'Fireball'`
- Mixed body: `ability 'X' mixes deliver{} and top-level effects`
- Unknown reserved keyword in identifier position

### 23.2 Lowering errors

Reported with offending source span + IR context:

- Unknown verb: `unknown effect verb 'teleport'` (with suggestion if close match)
- Unknown material: `unknown material 'sandstone' (not in .sim catalog)`
- Unknown structure: `unknown structure 'OrcKeep'`
- Arity mismatch: `damage expects 1 argument, got 2`
- Budget exceeded: `ability 'X' has 5 effects (max 4)`
- Status reserved: `verb 'polymorph' is reserved; engine does not implement it yet`
- Two-phase split violation: `decision-time code may not call ground-truth queries`
- Selector type mismatch: `ToolSel::OwnedSpecific(123) but tool 123 does not exist`
- Recipe inputs/outputs imbalance: `recipe 'ForgeSword' violates value-conservation`

### 23.3 Runtime errors

Engine never panics on cast-time invariants; violations emit
diagnostic events and fail the cast cleanly:

- `CastDepthExceeded` — already ships for `MAX_CASCADE_ITERATIONS`
- `CastBudgetSpilled` — cast spilled into later tick
- `VoxelMaskUnloaded` — registry miss (diagnostic only)
- `RecipeMissingInputs` — agent doesn't have required commodities
- `ToolBroken` — required tool's wear >= durability
- `SkillBelowThreshold` — required skill not met
- `ContestFailed` — pickpocket/demand resolved against caster

### 23.4 Unhandled-construct policy

- Author uses a `reserved` construct → lower-error (build fails)
- Author uses a `planned` construct whose engine side isn't yet wired
  → lower warning + runtime no-op (build succeeds; cast does nothing
  instead of the expected effect)
- Author uses a `runs-today` construct → zero friction

---

## §24 Implementation waves

Six waves with rough estimates and dependencies.

### Wave 1: Parser core + AbilityProgram lowering

**Scope:**
- `.ability` lexer (DURATION, PERCENT, STRING, PARAM_REF, COMMENT)
- `.ability` parser (ability/passive/template/structure top-level forms)
- AST module (shared with `.sim`)
- Resolver (name resolution, registry binding)
- Lowering of header properties to Gate fields
- Lowering of effect_stmts (variants 0-7 only) to verb cascade + chronicle handlers
- AbilityRegistry SoA + GPU buffer layout
- Hero TOML binding (`abilities_file = "warrior.ability"`)
- engine_gpu_rules/ skeleton

**Estimate:** 1500-2500 LOC. 4-8 agent dispatches.

**Unblocks:** parsing existing 30+ .ability files, hero compositions,
basic combat through .ability instead of hand-coded verbs.

**Dependencies:** none.

### Wave 2: Combat depth

**Scope:**
- Control verbs (root/silence/fear/taunt) → new EffectOps
- Movement verbs (dash/blink/knockback/pull) → new EffectOps + per-agent state
- Buff/debuff verbs (buff/debuff/damage_modify/lifesteal) → per-agent multiplier SoA
- Advanced verbs (execute/self_damage)
- Tag system widening (6-enum → open-set with mapping table)
- Scaling terms (`+ N% stat_ref`)
- Conditions: physical state atoms
- Stacking modes (stack/refresh/extend)
- Delivery: projectile, chain, zone, channel (require Delivery enum + per-method dispatch)

**Estimate:** 1500-2500 LOC. 3-5 agent dispatches.

**Unblocks:** rich combat ability authoring (Whirlwind, Fireball,
Earthshatter without voxel ops, ChainLightning).

**Dependencies:** Wave 1.

### Wave 3: AI-state writes + reads

**Scope:**
- BeliefState SoA + per-tick decay phase (per ToM design doc)
- AgentSel + SubjectSel + GroupSel selectors
- Standing extension: ModifyStanding evolves to {a_sel, b_sel, delta}
- TOM Phase 1 verbs (scry/reveal/stealth/disguise/decoy/erase_belief/plant_belief)
- TOM EffectOps (ClearBelief, PlantBelief, RefreshBelief)
- Communication verbs (silence/mute/jam_channel/translate/...)
- Engagement verbs (force_engage/break_engagement/taunt/rally_call/...)
- AI-state condition atoms (standing_below, target_confidence_below, believes())
- Templates with typed parameters

**Estimate:** 2000-3000 LOC. 5-7 agent dispatches.

**Unblocks:** assassin/spy/diplomat ability designs, TOM-based
strategies, faction politics.

**Dependencies:** Wave 1.

### Wave 4: Voxel ops + structures + materials

**Scope:**
- Materials declaration in .sim
- Structure top-level form with parameters/symmetry/includes/conditional placement
- 3D shapes (5 disc family + 7 volume family)
- CSG composition (union/diff/intersect)
- Voxel verbs (place_voxels/harvest_voxels/transform_voxels/place_structure)
- Voxel-aware conditions (floor_material, voxel_at, voxel_count)
- Voxel mask registry, voxel revert queue, damageable voxel set
- Voxel storage subsystem (L0-L6 tiered, region registry, per-chunk epoch)

**Estimate:** 3000-5000 LOC for ability-side surfaces, plus voxel
storage subsystem (separate multi-month track).

**Unblocks:** environmental abilities (PaperArmor, Earthshatter,
IceWall, FortifyKeep), terrain manipulation as gameplay.

**Dependencies:** Wave 1, voxel storage subsystem (independent track).

### Wave 5: Economy

**Scope:**
- Inventory as variable-size per-agent SoA (Option A: bounded arrays)
- Tool entity + ToolKindRegistry + per-tool wear/durability
- Skill catalog + per-agent skill SoA
- Commodity catalog
- Recipe top-level declaration
- RecipeRegistry partition (compile-time + runtime)
- Recipe execution: EffectOp::Recipe + WearTool dispatch
- Property + Obligation entities + registries
- Economy EffectOps 17-26
- Gate predicates: require_skill, require_tool
- Quality formula expression registry
- Counterfeit apparent-vs-actual fields
- New AbilityHint variants (Economic, Financial)
- Per-agent belief extensions (cold_price_beliefs, cold_merchant_beliefs, etc.)

**Estimate:** 2000-3000 LOC plus the inventory SoA architectural work.
6-9 agent dispatches.

**Unblocks:** real economy gameplay (production, trade, contracts,
counterfeiting, brokerage).

**Dependencies:** Wave 1, Wave 3 (belief storage), inventory SoA
architectural decision.

### Wave 6: Generative composition + retraining

**Scope:**
- Slot-fill grammar generation (~30 templates × inputs × outputs × skill × tool × quality)
- Three hard gates (grammar + economic + novelty)
- Bounded registries (4096 ability cap, LRU GC)
- Retrain ability transformer with new vocab + EffectOp set
- New training corpus generation
- Population-level discovery telemetry
- Generated abilities as GOAP planner goals

**Estimate:** mostly outside compiler scope. Training pipeline rebuild
+ ML model retraining. Months of work.

**Unblocks:** the spec's central design intent — agents inventing new
abilities/recipes at runtime, propagating via memory/testimony/
apprenticeship.

**Dependencies:** Waves 1-5.

### 24.1 Suggested first slice

The smallest meaningful deliverable: **Wave 1 + the smallest cut of
Wave 5** (Inventory SoA + RecipeRegistry + EffectOp::Recipe end-to-end).

This produces a working `crafting_recipes_real` sim where:
- Ability blocks parse from `.ability` files
- Recipes declared in `.sim` compile to RecipeRegistry entries
- Agents execute production recipes (consume inputs, produce outputs)
- Combat abilities (Damage/Heal/Shield) work alongside recipes
- Quality formulas evaluated at production time
- Hero TOML files reference both combat + production .ability files

Estimated 4000-6000 LOC. Likely 8-12 agent dispatches over multiple
days. Validates the parser pipeline AND the recipes path simultaneously.

After this lands, Waves 2-4 + remainder of 5 can proceed in parallel
since they share the parser + dispatch infrastructure.

---

## §25 Open questions

Things that need decisions before implementation can start.

### 25.1 Variable-size per-agent SoA

Option A (bounded arrays) chosen as default but not validated. Open
question: at 200K agents × 5 KB/agent = 1 GB GPU memory, are we OK
with that footprint? Profiling needed.

If not, Option B (indirect indexing) requires designing a per-tick
allocator kernel that runs before any inventory-touching dispatch.
Adds complexity; increases dispatch count per tick.

Option C (sparse hash table) is the most flexible but has worst-case
O(N) lookup latency. Probably unacceptable for hot-path SoA.

### 25.2 EffectOp dispatch shape at scale

27 EffectOp variants in a single fused chronicle kernel: is this
GPU-feasible? Each variant compiles to a different code path; a fused
kernel may suffer divergent execution.

Three options:
- **Single fused kernel** with branch-per-variant. Simple, lots of
  divergence at scale.
- **Variant-bucketed kernels** — one kernel per variant. N_variants
  dispatches per tick. Less divergence but more launch overhead.
- **Dynamic dispatch table** — indirect dispatch via a registry of
  per-variant kernel IDs. Most flexible, requires GPU indirect
  dispatch support.

This spec defaults to the single fused kernel for simplicity; revisit
after profiling if divergence becomes a bottleneck.

### 25.3 Recipe execution latency

Recipes have `duration_ticks` (e.g. 400ms = 4 ticks at 10Hz). Real
multi-tick recipes need persistent in-progress state. Options:

- **Per-cast SoA entry**: every active recipe instance occupies a slot
  in `recipe_in_progress[]`. Cap on concurrent recipes per agent.
- **Per-agent slot**: one recipe-in-progress at a time per agent.
  Restrictive but simple.

Spec defaults to per-agent slot (matches `cast_in_progress_ability`
shape). Multi-recipe queueing deferred.

### 25.4 Two-phase split enforcement granularity

The validator pass needs to distinguish "decision-time" from
"execution-time" code at the IR level. Today the IR doesn't tag this.
Options:

- **Whole-rule classification**: every `mask`, `scoring`, `verb`
  selection body is decision-time; everything else is execution-time.
  Simple. May reject legitimate code that uses ground truth at
  cast-resolve time inside a verb.
- **Per-expression classification**: each IR expr node carries an
  epistemic tag, computed at lowering. More precise, more complex
  validator.
- **Annotation-based**: authors declare `@decision` / `@execution`
  on rule blocks. Explicit but burdensome.

Spec defaults to whole-rule classification with `@execution` opt-out
on specific clauses.

### 25.5 Generative composition validation budget

Runtime ability composition (Wave 6) generates new EffectOp programs.
Each must pass grammar + economic + novelty gates. The economic gate
needs expected-value computation — at runtime, against current price
beliefs. How expensive is this? If checking each candidate program
costs 1ms, and the model produces 1000 candidates per agent per
generation event... potentially slow.

Spec defers profiling until Wave 6.

### 25.6 Cross-DSL interop

Should `.sim` files be able to declare `ability` blocks too? Would let
authors keep gameplay-cohesive sims (mob abilities + worldgen rules
in one place). But complicates the parser (two top-level form
namespaces).

Spec defaults to keeping them separate.

### 25.7 Backwards compatibility

Stale `.ability` files in `dataset/hero_templates/*.ability` were
authored for the wiped TacticalSim parser. Some constructs may not
match this spec's reduced surface. Migration path:

- **Lossy import**: parse what we can, warn on unsupported constructs
- **Hand-port**: rewrite each file to match the new spec
- **Delete**: assume training corpus will be regenerated

Spec defaults to hand-port for the small set (~30 files).

---

## §26 Authoring ergonomics

Notes on the syntax that aren't strictly required but improve
authoring experience.

### 26.1 Inheritance / templates

`ability Fireball : ElementalBolt(fire, 4.0) { ... }` — instantiate a
template with overrides. Simpler than copy-paste.

### 26.2 Inline recipe declarations

```
ability ForgeSword {
    ...
    recipe {
        consume iron_ingot 2
        consume leather 1
        produce sword quality(...)
    }
}
```

vs. the spec's preferred pattern of recipes as their own top-level
declaration:

```
recipe ForgeSword { consume ..., produce ... }

ability CastForgeSword {
    ...
    cast_recipe ForgeSword
}
```

Spec defaults to top-level recipes (matches the
"recipes-as-AbilityProgram" framing).

### 26.3 Indentation vs braces

Braces only. Significant whitespace was considered and rejected (more
parser complexity, less editor support, no syntactic benefit).

### 26.4 Comments

Both `//` and `#` accepted. `//` preferred (matches Rust + most code
in the codebase).

---

## §27 Cross-references

Original specs this consolidates:
- `docs/spec/ability.md` — canonical ability DSL
- `docs/spec/economy.md` — recipes / inventory / obligations
- `docs/spec/dsl.md` — .sim DSL (verbs, physics, views)
- `docs/superpowers/specs/2026-04-22-theory-of-mind-design.md` — BeliefState
- `docs/superpowers/specs/2026-04-25-voxel-region-indices-design.md` — voxel storage
- `docs/superpowers/specs/2026-04-26-kernel-dispatch-emit-design.md` — engine_gpu_rules/

External references:
- `docs/constitution.md` — P1 (compiler-first), P2 (schema-hash), P3 (cross-backend), P4 (16B EffectOp), P5 (keyed PCG), P6 (events as mutation), P11 (reduction determinism)
- `crates/engine/src/ability/` — existing AbilityProgram + EffectOp data structures
- `crates/dsl_compiler/src/cg/` — current compiler IR + emit pipeline
- `dataset/hero_templates/*.ability` — orphan training corpus to migrate

---

*End of unified spec.*
