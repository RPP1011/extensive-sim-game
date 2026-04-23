# Ability-evaluation GPU migration — design deep-dive

**Status:** research, feeds a future brainstorming cycle
**Date:** 2026-04-22
**Predecessor:** [GPU-everything scoping](./2026-04-22-gpu-everything-scoping.md)
**Branch:** `world-sim-bench`

## Scope

Item 6 from the scoping report ("ability evaluation beyond DSL-compiled
physics") is flagged as the largest remaining diff on the CPU side. This
document drills into the per-tick decision path that still runs on the
host, catalogues the storage and dispatch primitives the GPU backend
would need, and proposes a phased migration.

Ground truth: commit `98894e70`. Paths are absolute for direct navigation
from later brainstorms.

---

## Section A — Current CPU ability evaluator architecture

The tactical-sim "ability evaluator" is not a single `struct
AbilityEvaluator` — there are three layered decision surfaces that can
each emit an `IntentAction::UseAbility { ability_index, target }`.

### A.1 Layer 1 — squad/combat heuristic scorer

**Entry point:** `evaluate_hero_ability`
(`/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/tactical_sim/src/squad/combat/abilities.rs:11-337`).

Called from `choose_action` at line 357 of the same file, which itself is
invoked from `generate_intents_with_terrain`
(`/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/tactical_sim/src/squad/intents.rs:25`).

Per-ability algorithm (abilities.rs:59-300):

1. For each `AbilitySlot` on the caster:
2. Skip if `cooldown_remaining_ms > 0`, target type mismatched with
   `AbilityTargeting`, resource insufficient, or out of range.
3. Compute a score from `ai_hint` (`"damage"`, `"heal"`, `"crowd_control"`,
   `"defense"`, `"utility"`, `"opener"`) combined with per-effect analysis
   — AoE target count, CC threat-reduction, kill bonus, zone-tag combo
   bonus, conditional-damage deferral.
4. argmax over slots; return `(ability_index, score)`.

This is the *heuristic* ability evaluator. No neural component. The
"urgency > 0.4 interrupt" referenced in `CLAUDE.md` is historical — in
this branch the squad layer always calls `evaluate_hero_ability`
before falling through to `Attack`/`MoveTo` unless `mode ==
FormationMode::Retreat` (abilities.rs:357).

### A.2 Layer 2 — ability transformer (neural)

`ActorCriticWeightsV5`
(`crates/tactical_sim/src/sim/ability_transformer/weights_actor_critic_v5.rs:481`),
opt-in via `transformer-play` / `transformer-rl` CLIs (not default).
d_model=128, 8 heads, 4 layers, frozen Rust inference. Three parts:
ability transformer encoder (`encode_cls`, cached per-fight); per-tick
`FlatEntityEncoderV5` (7 tokens × 34 feat → pooled 128-d); and a
dual-head decision (`dual_head_logits`, line 652) — a 9-way move head
plus a combat-pointer head with per-ability cross-attention over
entity tokens.

### A.3 Layer 3 — pointer action-space translator

**Entry point:** `pointer_action_to_intent`
(`/home/ricky/Projects/game/.worktrees/world-sim-bench/crates/tactical_sim/src/sim/self_play/actions_pointer.rs:25`).

Converts the `(action_type, target_token_idx)` decision from the neural
head into an `IntentAction`. Action-type encoding:

```
0  = attack(target_token)
1  = move(target_token)
2  = hold
3..10 = use ability_0..7 (target_token)
```

`MAX_ABILITIES = 8` (`self_play/mod.rs:8`). `NUM_ACTIONS = 14` (legacy
flat space). The engagement heuristic in
`src/bin/xtask/oracle_cmd/transformer_rl.rs` patches a known failure:
when the model says "hold" with no enemy in attack range, force a move.

### A.4 Integration order

The production path goes through Layer 1 (heuristic). Layer 2 (neural)
is wired behind a CLI flag for research harnesses. Layer 3 (translator)
is used by both the transformer and the oracle/RL training pipelines.

Squad AI → (optional transformer interrupt) → GOAP/control overrides →
`IntentAction` → `crate::sim::step()` which executes the ability through
the CPU `apply_effect`/`cast` path.

### A.5 Concrete counts

* **299 `.ability` files** under `dataset/abilities/**` +
  `dataset/hero_templates/**` (172 LoL imports, 20 hero templates,
  remainder tier kits + 10 campaign).
* **9 `ai_hint` categories** observed: damage, crowd_control, utility,
  heal, defense, buff, leadership, economy, diplomacy.
* **~125 `Effect` variants** in `effects/effect_enum.rs:13` (~25
  combat-relevant, ~100 campaign/meta). GPU physics kernel handles 8
  via `EffectOp` (physics.rs:159).
* **7 combat `AbilityTargeting` variants** (+ 7 campaign-only).
* **MAX_ABILITIES = 256** (registry cap, physics.rs:121).
  **MAX_EFFECTS = 8** per program. **MAX_ABILITIES_PER_UNIT = 8** on
  the self_play side.

---

## Section B — What already runs on GPU

### B.1 Physics kernel coverage

From `crates/engine_gpu/src/physics.rs:17-50`, the rule matrix marks
FULL on GPU: `damage / heal / shield / stun / slow`,
`opportunity_attack / engagement_* / fear_spread_on_death /
pack_focus_on_engagement / rally_on_wound`, and crucially **`cast`**
(physics.rs:43). Chronicle rules are STUB (task 4a);
`transfer_gold / modify_standing / record_memory` are STUB (4b/c).

The `cast` physics rule
(`crates/engine/src/generated/physics/cast.rs:10-129`, DSL at
`assets/sim/physics.sim:222-263`) fires on an `AgentCast` event, reads
the ability registry, walks `EffectOp` items, and emits one event per
effect (Damage/Heal/Shield/Stun/Slow applied; GoldTransfer;
StandingDelta; nested AgentCast for chains), then sets the caster's
cooldown cursor. This is already inside the GPU megakernel.
**The post-cast effect cascade is GPU-resident today.**

### B.2 The gap

`mask Cast(ability: AbilityId)` was skipped from the fused-mask kernel
(mask.rs:23-42) because its DSL predicate takes a non-Agent parameter
and reads views + a cooldown field. Tracing
`crates/engine/src/generated/mask/cast.rs`, the predicate needs:
`agent_alive` (GPU has it), `!is_stunned` (GPU has `stun_expires_at`,
trivial compare), `ability_registry.get` (GPU has packed registry,
physics.rs:356-373), `tick >= cooldown_next_ready` (GPU has a **single
cursor per agent** in `GpuAgentSlot`, not per-(agent, ability) — this
is a correctness gap even on CPU: a caster whose Fireball cooldowns
shares the cursor with Heal), and `engaged_with.is_none()` (GPU has it).

All ingredients except per-ability cooldown are already on GPU. Missing
piece: a scoring-side kernel that *emits* `AgentCast { caster,
ability_id, target }` on the batch path. Today those events only fire
via recursive cascade fan-out (nested `CastAbility`), never
top-level.

### B.3 Pure-physics vs "decide-to-fire" abilities

Every ability's **effects** reduce to `EffectOp` — already GPU-native.
Every ability's **firing decision** is CPU-only in the batch path
today, whether the firing source is:

* squad/combat heuristic (`evaluate_hero_ability`),
* neural ability transformer (`ActorCriticWeightsV5`),
* GOAP planner (`crates/tactical_sim/src/goap/`).

There is exactly one GPU-side `AgentCast` emitter: the `cast` rule's
recursion (nested `CastAbility` → new `AgentCast`). Top-level casts —
the ones the scoring kernel should produce — never fire on the batch
path. That is the correctness gap to close.

### B.4 DSL emitter constraints

Per `docs/technical_overview.md:42`, the GPU-emittability validator
rejects: heap allocation, recursion, dynamic dispatch, parse-time-
unbounded loops. Ability scoring runs afoul of:

* **Unbounded slot loops.** `unit.abilities.iter()` is iterated in the
  CPU evaluator. Bounded statically by `MAX_ABILITIES = 8` (self_play
  side) or `MAX_ABILITIES = 256` (engine side) — both emittable.
* **Dynamic Effect dispatch.** `match &ce.effect { Damage { .. } => ..,
  Stun { .. } => .. }` has ~85 arms; the CPU evaluator touches only
  Damage/Heal/Stun/Slow/Buff/Debuff/Dash/Shield. A WGSL port can
  enumerate as a u32 discriminant switch (the `GpuEffectOp.kind` scheme
  already does this for 8 kinds in physics.rs).
* **String matching on `ai_hint`.** The evaluator reads
  `slot.def.ai_hint` as `&str` at runtime (abilities.rs:220). WGSL
  has no strings. Must be pre-compiled at registry-pack time to a u8
  hint discriminant (9 values).

None of these are fundamental blockers. They are compile-time
flattenings.

---

## Section C — Proposed GPU migration architecture

### C.1 Decision tree

Three realistic migration strategies, each with a different trade-off:

1. **Hybrid A — CPU urgency, GPU execution.**
   CPU keeps the full heuristic/neural evaluator. Once per tick, CPU
   uploads a per-agent `(chosen_ability, target)` buffer; GPU physics
   emits `AgentCast` events from that buffer. Minimal WGSL.
2. **Hybrid B — GPU heuristic, CPU neural.**
   Port `evaluate_hero_ability` to WGSL. Neural evaluator stays on CPU
   (optional research path). This is the default production path move.
3. **Full — GPU heuristic + neural.**
   Port the V5 transformer's entity encoder + cross-attention + pointer
   head to WGSL. Largest diff.

Recommended target: **Hybrid B for v1, Full as a stretch.** The
heuristic is production-critical; the neural path is a research toggle
that's fine on CPU.

### C.2 Hybrid B architecture sketch

New kernel: `pick_ability.wgsl`. One workgroup per agent-block
(`@workgroup_size(64)`). For each live agent:

1. For each `ab_id` in `0..MAX_ABILITIES_PER_UNIT` (8):
   * Load `PackedAbility { hint, range, resource_cost, cooldown_ms,
     effect_summary }` from the agent's ability slot.
   * Gate: cooldown ready, resource sufficient, live, not stunned, not
     engaged.
   * Pick a target via nearest-hostile (for `TargetEnemy`), lowest-HP
     ally (for `TargetAlly`), self (for `SelfCast/SelfAoe`), or a
     position-ptr (for `GroundTarget`). Reuses the kin/spatial results
     the physics kernel already requires.
   * Compute a scalar score using a compile-time score table indexed by
     `hint × effect_summary`.
2. argmax over candidates; write `PackedChoice { ability_id, target, score }`
   to the new `chosen_ability: array<PackedChoice>` buffer.
3. If `score >= threshold` and `ability_id != NO_ABILITY`, emit
   `AgentCast { caster, ability: ability_id, target, depth: 0, tick }`
   into the physics event ring. The existing `cast` rule does the rest.

WGSL sketch: a single `@compute @workgroup_size(64)` entry `pick_ability`
reads `agents[agent_idx]` (gating on alive / stun / engaged), iterates
`0..MAX_ABILITIES_PER_UNIT`, short-circuits on cooldown and target
availability, argmaxes `score_ability(...)`, and — if the winner beats
`cfg.score_threshold` — `atomicAdd`s into the physics event ring to
emit `AgentCast { caster, ability_id, target, depth: 0, tick }`. The
`cast` physics rule drains it. Bindings: 7 total (agent SoA, ability
slots, registry, cooldown buffer, chosen-ability side buffer, cast
event ring + tail, cfg).

### C.3 Urgency computation

The CPU heuristic is a table of multiplicative weights keyed on
`(hint, effect_shape)`. Three subcases:

* `hint == "damage"`: score = `total_damage + kill_bonus + cc_reduction
  + opener_bonus` with a deferral penalty when CC is ready and
  conditional damage is unmet.
* `hint == "crowd_control"`: score = `cc_threat_reduction +
  total_damage` (zero if target already controlled).
* `hint == "defense"`: score scales on `hp_pct` with AoE × ally-count
  bonus.
* `hint == "heal"`: score scales on `hp_pct` of the self/ally target.
* `hint == "utility"`: score = `base + buff/debuff ally multiplier +
  cc_reduction`.

All of these are scalar arithmetic over a fixed-size feature vector.
Port to WGSL is straightforward — no branches that can't be flattened
to `select()`, no loops without a parse-time bound.

**Pre-pack** `hint`, `range`, `resource_cost`, `cooldown_ms`, an
`effect_summary` bitfield (`has_damage | has_heal | has_stun | ...`), and
a precomputed `total_damage` / `cc_duration` / `heal_amount` once at
registry-pack time. Size per entry ≤ 32 B. At `MAX_ABILITIES = 256`,
that's 8 KB — a single uniform array.

### C.4 Target selection

The CPU path walks live enemies for nearest-hostile, lowest-HP-ally,
etc. The GPU physics kernel already has per-agent kin lists
(`GpuKinList`, physics.rs:175) and nearest-hostile precomputes from
`spatial_gpu::rebuild_and_query`. Extend these precomputes with:

* `nearest_hostile_to[agent]: u32` (already present via `kin_nearest`).
* `weakest_ally_to[agent]: u32` (new — ~200 LOC of kernel, O(N·K) scan
  within kin radius).
* `aoe_target_count[agent][ability_slot]: u32` (more expensive —
  defer; use an approximation based on kin density).

For the pointer-attention (Layer 2) port: 128×N matrix multiply per
agent per ability, where N ≤ 7 (entity tokens). ~3.5K MACs per agent
per ability. At `MAX_ABILITIES_PER_UNIT = 8`, 28 KB of compute per
agent — comfortable. Adds one new bind group for the attention key/
value projections (6 `FlatLinear` tensors, ~50 KB total).

### C.5 Effect dispatch

No change. `AgentCast` events from the pick_ability kernel are drained
by the existing physics megakernel via the event ring. The DSL-emitted
`cast` rule (physics.sim:222) already handles `AgentCast` dispatch to
`EffectDamageApplied` and friends. Only new work: the physics kernel's
`cast` rule must read a **per-(agent, ability) cooldown**, not the
current single-cursor `agent_cooldown_next_ready`.

### C.6 Parametric action space

The scoring kernel's `ScoreOutput` (scoring.rs:187-211) is a fixed
16-byte record: `{ chosen_action: u32, chosen_target: u32,
best_score_bits: u32, debug: u32 }`. Adding `chosen_ability: u32`
widens to 20 B; natural padding to 32 B or pair with a second buffer
keyed on `action == Cast`.

Two options:

* **Option A — widen `ScoreOutput`.** One-line schema bump, cascades
  through `apply_actions`, movement kernel, chronicle render.
* **Option B — side buffer.** `chosen_ability: array<PackedChoice>`
  lives alongside `ScoreOutput`; `apply_actions` reads both and knows
  `chosen_action == Cast` means "look up the ability in the side
  buffer."

Option B is cleaner — keeps the existing `ScoreOutput` layout stable
for the 7 non-Cast action types, and the Cast pick_ability kernel is
independent of the scoring kernel. It does cost one extra bind group.

---

## Section D — DSL emitter changes required

### D.1 Scoring emitter (`emit_scoring_wgsl.rs`, 2096 LOC)

Today at line 141, `MASK_SLOT_NONE` is returned for `Cast` / `UseItem` /
`Harvest` / `Converse` / …, and scoring rows for those kinds are
unconditionally skipped by the kernel (emit_scoring_wgsl.rs:158).

To support GPU ability picking we have a clean fork:

* **Keep the scoring row for `Cast` at base 0.0.** The `scoring.sim`
  grammar can't express per-ability scoring — `assets/sim/scoring.sim:248`
  sets `Cast = 0.0`. The GPU scorer already filters it out via
  `MASK_SLOT_NONE`. No emitter change.
* **Emit a new `pick_ability` WGSL module separately.** The DSL compiler
  could grow a new `emit_ability_scoring_wgsl.rs` module (~600 LOC) that
  consumes a new `assets/sim/ability_scoring.sim` file. Alternative: hand-
  write the kernel in `crates/engine_gpu/src/pick_ability.rs` and skip
  the DSL path entirely.

**Recommendation:** skip the DSL for pick_ability v1. The heuristic in
`squad/combat/abilities.rs` is imperative code that doesn't fit the
`scoring.sim` expression grammar (`<lit> ('+' <modifier>)*`). Hand-
writing the WGSL is ~400 LOC, one-off, and keeps the DSL compiler
honest about its current grammar limits. When the grammar grows to
support ability-parametric scoring rows, migrate in a follow-up.

### D.2 Physics emitter (`emit_physics_wgsl.rs`, 2565 LOC)

Already emits the `cast` rule fully (physics.rs:43 confirms `cast` is
FULL). Two changes required:

* **Per-(agent, ability) cooldown storage.** `set_cooldown_next_ready`
  currently writes to `GpuAgentSlot.cooldown_next_ready`. Switch to
  `agent_ability_cooldowns[agent_slot * MAX_ABILITIES_PER_UNIT +
  ability_slot_idx]`. The DSL source `physics.sim:263` reads
  `agents.set_cooldown_next_ready(caster, next_ready)` — change to
  `agents.set_ability_cooldown_next_ready(caster, ab, next_ready)` and
  add the new method to the generated agent mutator surface.
  Compiler impact: a new namespace call + its WGSL emit (~30 LOC).
* **Chronicle-ring emit for cast events.** Covered under item 4a in the
  scoping report.

### D.3 Mask emitter (`emit_mask_wgsl.rs`, 1319 LOC)

`mask Cast(ability)` is currently rejected by the fused-mask validator
(mask.rs:23). To re-admit:

* **Per-ability mask bitmap.** Two approaches:
  1. Flatten: emit `MAX_ABILITIES` distinct masks (`Cast_ab0`, …,
     `Cast_ab255`). Bloats the mask bitmap by 256× (~2 GB at N=100k,
     way over budget).
  2. Runtime loop: the mask kernel writes one bit per `(agent, slot)`
     pair — `cast_mask[agent * MAX_ABILITIES_PER_UNIT + slot] = can_cast`.
     Storage: N × 8 bits = ~800 KB at N=100k. Acceptable.

  Recommendation: option 2, since `MAX_ABILITIES_PER_UNIT = 8`, not 256.
  Emitter change: ~200 LOC to handle the parametric mask head's
  non-Agent `AbilityId` parameter. The current emitter has a compile-time
  assertion that rejects non-Agent parameters (hence the skip).

Alternatively: **bypass the mask emitter entirely** and fold the
"can_cast" check into `pick_ability.wgsl` directly. That is the
pragmatic v1 route and keeps the mask emitter limited to its current
scope.

### D.4 LOC estimates

| Emitter file                         | Current LOC | Expected delta |
|--------------------------------------|-------------|----------------|
| `emit_scoring_wgsl.rs`               | 2096        | +0 (v1), +600 later |
| `emit_physics_wgsl.rs`               | 2565        | +30 (cooldown method) |
| `emit_mask_wgsl.rs`                  | 1319        | +0 (v1), +200 later |
| new `pick_ability.rs` (engine_gpu)   | 0           | ~600 (WGSL + host) |

Main touch: engine_gpu gets a new kernel file. DSL emitters see a
single-method addition. Total ~700 LOC before tests.

---

## Section E — Dependencies + ordering

### E.1 Hard prerequisite: item 4 before item 6

The scoping report claims item 6 depends on item 4 because cast effects
fire `transfer_gold` / `modify_standing` / `record_memory` whose GPU
handlers are currently stubbed. Confirmed:

* `cast.rs:82-97` — `EffectOp::TransferGold` → `Event::EffectGoldTransfer`.
* `cast.rs:90-97` — `EffectOp::ModifyStanding` → `Event::EffectStandingDelta`.
* Neither has a GPU physics handler today (`physics.rs:40-42`, stubbed).

If we enable GPU ability firing before those stubs become real, gold-
transfer and standing-shift abilities become no-ops on the batch path —
a silent correctness regression. The scoping report's "land 4 before 6"
ordering holds.

Chronicle rules (4a) are independent — pure observability side effects.
4b (gold + standing) and 4c (record_memory) are load-bearing for
cast-effect correctness.

### E.2 Agent-slot storage

`GpuAgentSlot` (physics.rs:132-153) is 64 B today. New fields needed:

| Field                                    | Bytes | Notes                             |
|------------------------------------------|-------|-----------------------------------|
| `per_ability_cooldowns[8]`               | 32    | Per-(agent, slot) cooldown tick   |
| `ability_slot_to_registry_id[8]`         | 32    | Which registry entry each slot has|
| `resource` + `max_resource`              | 8     | For mana/rage/energy gates        |
| `ability_hint_bits` (9 bits packed)      | 4     | Quick hint-category filter        |

Total: ~76 B of new per-agent data. Move to a **side buffer**
(`agent_ability_state: array<GpuAgentAbilityState>`) rather than
ballooning `GpuAgentSlot` (which is read by every kernel). At N=100k:
~7.6 MB. Cheap.

---

## Section F — Parametric action-space expansion

Three concrete changes:

### F.1 Scoring output format

Current: `ScoreOutput { chosen_action: u32, chosen_target: u32, … }`
(scoring.rs:187). Expand to include ability slot:

* **Option A — widen ScoreOutput.** Add `chosen_ability: u32`. 16 → 20 B.
  Change count: scoring.rs (+3 LOC), apply_actions.rs (+20 LOC to
  dispatch Cast), movement.rs (no change — Cast doesn't move), every
  consumer of `ScoreOutput` in tests (~80 LOC of test fixtures).
* **Option B — side buffer.** Leave `ScoreOutput` at 16 B, add
  `chosen_ability: array<u32>` in parallel. Change count: scoring.rs
  (+5 LOC to output the extra buffer), apply_actions.rs (+15 LOC),
  no ripple to tests that only inspect `ScoreOutput`.

Recommend **Option B**. The Cast action is separable from other
scoring outputs: only the `apply_actions` dispatch path needs to know
about it, and pick_ability is a separate kernel anyway.

### F.2 apply_actions / movement

apply_actions today skips Cast (apply_actions.rs:19). New branch:

```wgsl
case ACTION_CAST: {
    let choice = chosen_ability[agent_idx];
    // Emit AgentCast into the physics ring if choice.ability_id != NO_ABILITY.
    let slot = atomicAdd(&event_ring_tail, 1u);
    gpu_emit_agent_cast(event_ring, slot, agent_idx, choice.ability_id,
                        choice.target, 0u /* depth */, cfg.tick);
}
```

~20 LOC. Movement kernel unaffected — Cast doesn't move.

### F.3 Chronicle render

Chronicle ring already accepts arbitrary template_id + three u32 args.
No schema change. A cast event would render via template_id=CAST_CAST
(new) with args `(caster, target, ability_id)`. ~10 LOC in the
chronicle renderer, zero engine_gpu change.

### F.4 Test compatibility

Existing `parity_with_cpu` tests in `crates/engine_gpu/tests/` run the
sync path only. Side-buffer approach keeps `ScoreOutput` byte-
compatible — no break. A new `batch_cast_smoke` test would be added
(warlock fixture, N=64, 200-tick batch, assert `EffectDamageApplied`
count > 0).

---

## Section G — Risk matrix

| Risk                                             | Likelihood | Impact | Mitigation                                                                 |
|--------------------------------------------------|------------|--------|----------------------------------------------------------------------------|
| WGSL can't express all 85 Effect variants        | Low        | Low    | Only ~25 are combat-relevant; the physics kernel already dispatches 8. Flattening the remainder to `EffectOp` is a registry-pack concern, not a kernel concern. |
| Per-agent-per-ability cooldown storage size      | Low        | Low    | 8 slots × 4 B × 100k = 3.2 MB. Trivial.                                    |
| Dynamic branching on `ability_id` in WGSL        | Medium     | Medium | Keep the per-ability score function a pure table lookup over pre-packed bitfields. Avoid per-ability code paths — the score is data-driven. |
| Target selection precompute cost                 | Medium     | Medium | Reuse `kin_nearest` / `kin_within` spatial results already computed for physics. Weakest-ally scan is O(N·K) — K~12, cheap. |
| Scoring divergence between heuristic CPU and GPU | High       | Medium | Schema-hash the score-table constants. Add a `batch_vs_sync` statistical smoke test (cast count ±25%). Byte-parity is NOT a goal for the batch path (per scoping report). |
| Neural-evaluator port regression                 | Medium     | High   | Don't port V5 in v1. Leave transformer on CPU, gate via a CLI flag that disables GPU batch path for those research runs. |
| Item 4 not landing first                         | Medium     | High   | Strict sequencing. The brainstorm should not schedule item 6 work until item 4b ships. |

### G.1 Revised cost estimate

Scoping report said "1-2 weeks." Breakdown: 6a (CPU-pick, GPU-emit)
3-4 days; 6b (WGSL heuristic pick) 5-7 days; 6c (neural head port,
optional) 5-7 days; 6d (parametric action space) 2-3 days; per-ability
cooldown storage + DSL tweak 2 days; tests + tuning 3-5 days.

**Senior-staffed sprint: ~1.5 weeks if neural path descoped; 3+ weeks
with the full V5 port.** Scoping's "1-2 weeks" holds only for
heuristic-only + side-buffer action space. The neural port roughly
doubles the cost.

---

## Section H — Recommended decomposition

Four sub-tasks, each independently landable under `step_batch` while
preserving the sync path:

* **6a — CPU-urgency, GPU-emit** (2-3 d). Keep CPU
  `evaluate_hero_ability`; upload `cpu_chosen_ability` per tick;
  `apply_actions` emits `AgentCast` into the physics ring. Restores
  functional correctness on the batch path at the cost of one per-tick
  upload. Done signal: `batch_cast_smoke` passes.
* **6b — GPU heuristic, CPU-free tick** (5-7 d). Port
  `evaluate_hero_ability` to `pick_ability.wgsl`. Add per-(agent,
  ability-slot) cooldown side buffer. Remove 6a's upload. Done signal:
  N=10k profile shows zero CPU→GPU traffic attributable to ability
  evaluation.
* **6c — GPU neural head** (optional, 5-7 d). Port V5 entity encoder +
  cross-attention + decision head to WGSL. `[CLS]` cache stays
  host-side. Research-only.
* **6d — Parametric action space** (2-3 d, parallelisable with 6b).
  Add `chosen_ability` side buffer; `apply_actions` reads it; chronicle
  renders it. Byte-compatible with existing `ScoreOutput`.

**Ordering:** 6a → 6d → 6b. 6c is optional and independent.

---

## Section I — Open questions for brainstorm

1. **Heuristic vs grammar.** Extend `scoring.sim` with an ability-
   parametric row type, or hand-write `pick_ability.wgsl`? Grammar
   route preserves DSL as SOT but needs effect-summary types in the
   IR; hand-written route is pragmatic for v1.

2. **Trained model: GPU port vs CPU inference → GPU argmax.** The
   cleanest V5 path may be running CPU inference every N ticks and
   uploading a per-(agent, ability) logits tensor. Avoids porting 128-d
   attention/FFN to WGSL.

3. **Non-determinism budget.** The batch path already disclaims byte
   parity. Do we want deterministic (argmax by u32-bits + agent_id
   tiebreak) or free-running (float argmax)? Low cost either way; pick
   a policy.

4. **MAX_ABILITIES_PER_UNIT = 8.** Registry carries 256, self_play
   uses 8. No dataset hero exceeds 8 active — 8 is safe for the
   compile-time bound.

5. **Resource (mana/rage/energy) storage.** Not in `GpuAgentSlot`
   today. Adding the gate is 8 B / agent; regen rules are a separate
   scoping question.

6. **Pointer-target float non-associativity.** GPU softmax may pick a
   different target than CPU at exact ties. Document as acceptable
   batch-path non-determinism (same framing as view-fold order).

7. **Engagement heuristic.** The `transformer_rl.rs` "Hold + no enemy
   in range → Move" override must be ported to the 6c WGSL path as a
   single branch.

8. **Campaign ability filtering.** The CPU `_` arm in
   `abilities.rs:270` scores unknown hints at `3.0 + damage + cc` —
   campaign abilities (`ClaimTerritory` etc.) with zero damage/cc would
   still score 3.0, a spurious-cast floor. Audit during 6b.

---

Key file bookmarks are cited inline throughout. The load-bearing ones:
`crates/tactical_sim/src/squad/combat/abilities.rs:11-337` (heuristic),
`crates/tactical_sim/src/sim/ability_transformer/weights_actor_critic_v5.rs:481`
(V5 neural), `crates/engine/src/generated/physics/cast.rs` (GPU-ready
cast rule), `crates/engine_gpu/src/physics.rs:17-50,132-153` (GPU
physics matrix + agent slot), `crates/engine_gpu/src/mask.rs:23-42` and
`crates/engine/src/generated/mask/cast.rs` (the current Cast-mask skip).
