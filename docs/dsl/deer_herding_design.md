# Deer Herding — design memo (task 177)

Goal: deer that are fleeing should bias their flee direction toward
kin (other nearby deer) so the herd clusters for safety, instead of
every deer running in a pure away-from-threat straight line. The
same capability should be available to other prey species without
hardcoding "if creature_type == Deer" in engine primitives.

## 1. Which case applies — A or B?

**Case A** (engine primitive). Confirmed by reading `step.rs`:

```rust
// crates/engine/src/step.rs:380-414  (Flee arm)
ActionKind::Micro { kind: MicroKind::Flee,
                    target: MicroTarget::Agent(threat), } => {
    ...
    let away = (self_pos - threat_pos).normalize_or_zero();
    ...
    let new_pos = self_pos + away * speed;
    state.set_agent_pos(action.agent, new_pos);
    events.push(Event::AgentFled { ... });
}
```

The scorer in `policy/utility.rs::build_action` picks the threat
(`nearest_hostile` within `config.combat.aggro_range`) and wraps it
in `MicroTarget::Agent(threat)`. `step.rs` then hard-codes the flee
direction as `(self_pos - threat_pos).normalize_or_zero()`. There is
no DSL seam in the middle: scoring ranks *whether* to flee, but the
geometry of the flee is baked into the step kernel.

The current DSL surface (views + physics + scoring) cannot influence
the direction chosen by the Flee arm. Views feed scalar boosts into
the scorer's Attack / Flee rows; physics handlers react to emitted
events. Neither pathway reaches the `away = ...` line in step.rs.

## 2. Minimum engine + DSL surface

Narrow-primitive path (option A-narrow from the task spec):

1. **New capability on `entity` blocks**: `herds_when_fleeing: bool`
   (default `false`). Emitted into `Capabilities` alongside
   `can_fly` / `can_climb` / etc. Deer turn it on; Human / Wolf /
   Dragon leave it off. Engine reads it via
   `state.agent_capabilities(agent).herds_when_fleeing` on the Flee
   arm — no species-switch in engine code.

2. **New spatial primitive**
   `spatial::flee_direction_with_kin_bias(state, agent, threat_pos,
   kin_weight, kin_radius) -> Vec3`: returns a unit vector blended
   from `away_from_threat` and `toward_kin_centroid`. Kin centroid
   uses `spatial::nearby_kin(agent, kin_radius)` (already shipped in
   task 167). The blend is weighted: `normalize(away + kin_weight *
   toward_centroid)`.

   Degenerate cases:
   - No live kin in radius → return pure `away` (identical to
     current behaviour).
   - Kin at the agent's exact position → `toward_centroid` is zero;
     falls through to pure `away`.
   - Threat co-located → falls through to the existing "zero flee"
     degenerate no-op in step.rs (nothing emitted).

3. **Modify step.rs Flee arm** to consult the capability:
   ```rust
   let away = if state.agent_capabilities(action.agent)
                      .map(|c| c.herds_when_fleeing).unwrap_or(false) {
       spatial::flee_direction_with_kin_bias(state, action.agent, threat_pos,
           state.config.combat.kin_flee_bias, state.config.combat.kin_flee_radius)
   } else {
       (self_pos - threat_pos).normalize_or_zero()
   };
   ```
   Wolves / humans / dragons take the `false` branch — observationally
   identical to today's behaviour, so the wolves+humans parity baseline
   stays byte-exact.

4. **New config fields** in `config.sim`:
   `combat.kin_flee_bias: f32 = 0.5` (weight on toward-kin vector
   vs. away-from-threat) and `combat.kin_flee_radius: f32 = 12.0`
   (matches fear_spread / pack_focus visual-contact figure).

This is the full surface. No grammar extensions; DSL changes are
additive (new capability bool, new config floats) and run through
existing compiler emitters. The config-hash bumps (two new fields),
which is expected — tuning knobs are a documented schema-change
class in `config.sim`.

## 3. Is this compiler-first clean?

**Yes — with one caveat the task spec flagged.** The principle is
"engine = primitives, DSL = game logic". `flee_direction_with_kin_bias`
is a primitive in the same shape as `nearest_hostile_to` (task 148)
and `nearby_kin` (task 167): a bounded spatial query that returns a
scalar/vector value. The primitive takes `kin_weight` and
`kin_radius` as arguments, not hardcoded numbers — those come from
config, which is DSL-owned. The species filter happens inside
`nearby_kin` (already DSL-clean: "kin" = same `CreatureType`), not
via a species switch in engine code.

The caveat: the Flee arm's decision to *consult* the kin-bias
primitive vs. the pure-away path is gated on a boolean capability,
not a species check. Species-specific behaviour lives in the DSL
(`herds_when_fleeing: true` on the `Deer` entity), not in engine
primitives. This is the same pattern as `can_fly` / `can_climb` —
the engine doesn't `match ct` on them, it reads the capability.

Spec risk: widening `Capabilities` with a movement-mode-shaping bool
crosses a taxonomy line (today `Capabilities` is communication +
movement-mode + social). The alternative is a new `BehavioralTags`
struct, but that doubles the per-agent capability surface for one
bool. I opt for the single new field on `Capabilities` — documented
as "movement-bias when fleeing" alongside `can_climb` (also a
movement capability).

## 4. LOC estimate

| Location | Change | LOC |
|---|---|---|
| `assets/sim/entities.sim` | `herds_when_fleeing: bool` per entity + default | 6 |
| `assets/sim/config.sim` | `kin_flee_bias` + `kin_flee_radius` fields | 2 |
| `crates/dsl_compiler/src/emit_entities.rs` | emit `herds_when_fleeing` into `Capabilities` | ~4 |
| `crates/dsl_compiler/src/emit_config.rs` | emit new combat floats | ~2 |
| (generated output: `engine_rules/src/entities/*.rs`, `config/combat.rs`) | auto | (~30) |
| `crates/engine/src/spatial.rs` | `flee_direction_with_kin_bias` | ~40 |
| `crates/engine/src/step.rs` | capability-gated branch in Flee arm | ~10 |
| `crates/engine/tests/deer_herding.rs` (new) | clustering assertion + regression guards | ~100 |
| **Total new compiler/DSL surface** | | **~14** |
| **Total new engine runtime LOC** | | **~50** |
| **Total test LOC** | | **~100** |

Compiler-first surface (~14 LOC) is well under the 75-LOC gate. The
engine runtime growth is dominated by the new spatial primitive
(mechanical centroid math + a two-vector blend); the step.rs edit
is a ~10-line capability-gated branch.

**Go/no-go threshold** (task spec: ">100 LOC engine changes OR
species-specific logic in engine primitives"):

- Engine LOC: ~50 (`spatial.rs` + `step.rs`). Under 100.
- Species-specific logic in engine primitives: **no**. The primitive
  takes weights/radii as args. The step.rs branch gates on a
  capability bool, not on `CreatureType`. Species-specificity lives
  in the DSL (`herds_when_fleeing: true` on the Deer entity).

**Verdict: GO.**

## 5. Test plan

New test file `crates/engine/tests/deer_herding.rs`:

1. **`deer_cluster_when_fleeing_wolf`** — the core behavioural claim.
   Spawn 3 deer in a line at (5, 0, 0), (10, 0, 0), (15, 0, 0) + one
   wolf at (0, 0, 0). All deer start at hp=40 (below the `hp < 50`
   gate that makes Flee score above MoveToward). Record baseline
   mean pairwise deer distance at t=0; run 20 ticks; record again.
   Assert the final mean pairwise distance is *less* than the initial
   (clustering). Also assert none of the deer moved *toward* the
   wolf (displacement from wolf's initial position > starting
   distance — the invariant the existing `deer_flee_from_wolves`
   guards).

2. **`wolves_dont_herd`** — a regression guard that the capability
   gate actually isolates species. Spawn 3 wolves low-hp (hp=16, the
   `hp_pct < 0.3` gate fires so they Flee) + one dragon. Expected:
   wolves flee in pure away-from-dragon vectors; pairwise distances
   do NOT decrease below a control run. Fundamentally the regression
   the existing wolves-flee-from-dragons behaviour on baseline.

3. **`single_deer_flees_straight`** — degenerate case: one deer, one
   wolf. No kin in radius → pure `away`. Same final position as the
   pre-task behaviour within 1e-5.

4. **`deer_herding_with_threat_in_middle`** — edge case: wolf
   positioned between two deer. Each deer should still flee AWAY
   from the wolf — the kin bias must not override the threat term to
   the point where a deer runs *into* the wolf trying to reach kin
   on the far side. Assert both deer's final distance from the wolf
   is greater than initial.

5. **Regression: `wolves_and_humans_parity`** — the baseline file
   must stay byte-exact. Wolves / humans lack `herds_when_fleeing`,
   so the Flee arm takes the legacy `away` branch — identical code
   path, identical output.

6. **Regression: existing `deer_flee_from_wolves` + `action_flee.rs`
   tests** — `deer_flee_from_wolves` uses 1 deer + 1 wolf, which is
   degenerate case (3) above — no kin, straight flee. `action_flee.rs`
   uses `CreatureType::Human`, which doesn't herd. Both should pass
   unchanged.

Sample pairwise-distance measurement format (to be filled in post-
implementation, per the task's report-back spec):

```
Before (tick=0):  mean pairwise deer distance = 10.00 m
After  (tick=20): mean pairwise deer distance = X.XX m
Clustering delta: Y.YY m  (expected negative)
```

## 6. Risks / snags

- **Kin bias too aggressive** → a deer runs *toward* the wolf to
  reach kin. Mitigation: config `kin_flee_bias = 0.5` caps the
  kin vector at half the away-from-threat magnitude; the blended
  vector is then normalized, so kin can tilt direction but can't
  flip it past 90° away from "away". The unit-test
  `deer_herding_with_threat_in_middle` guards the corner case.

- **Kin centroid is stale** when a kin deer just died. Mitigation:
  `nearby_kin` already filters dead slots (task 167's dead-center
  contract). A deer that just died vanishes from the centroid on
  the next tick.

- **Config hash bumps** (two new `combat.*` floats). Expected;
  document in the commit message. Anyone with a custom
  `assets/config/*.toml` must add the new keys or take defaults.

- **Degenerate "single deer" path**. Empty kin list → centroid is
  zero → blended vector = pure away. Preserves existing single-deer
  behaviour exactly. Pinned by test (3).

- **Determinism**. `nearby_kin` already sorts its result by raw
  AgentId; the centroid sum is order-independent (addition is
  associative in f32 only approximately, but the kin count is
  small and the sort discipline makes the iteration order
  deterministic). No RNG usage.

## 7. Go / no-go

**GO.** Case A with narrow primitive. Engine LOC ~50, compiler/DSL
surface ~14, no species switches in engine code — capability-gated.
Baseline parity preserved (wolves/humans lack the capability).
Determinism preserved (no RNG, deterministic iteration).
