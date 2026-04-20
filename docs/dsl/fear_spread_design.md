# Fear Spread — design memo (task 167)

Goal: when a wolf dies, alive wolves within a radius receive a temporary
`kin_fear` bump that decays over time. `kin_fear > 0.3` biases the Flee
score so the survivors rout. Symmetric for any species — the `kin_fear`
view keys on (observer, dead_kin) pairs, and `observer.creature_type ==
dead_kin.creature_type` is enforced by the *physics* rule that produces
the `FearSpread` events (not the view fold itself — see §3).

## 1. Pipeline target shape

```
AgentDied { agent_id: dead }
  └── physics fear_spread_on_death @phase(event) (new)
        for kin in query.nearby_kin(dead, radius):
          emit FearSpread { observer: kin, dead_kin: dead }

FearSpread { observer, dead_kin, tick }
  └── view kin_fear(observer, dead_kin) @materialized @decay pair_map
        on FearSpread { observer, dead_kin } { self += 1.0 }

scoring Flee:
  + (if view::kin_fear(self, _) > 0.3 : +0.5)
    — wildcard slot `_` sums decayed kin_fear across every recorded
      dead_kin for this observer; `sum_for_first` is already emitted
      for every @decay pair_map view (emit_view.rs:457+).
```

## 2. Primitive inventory

### New stdlib: `query.nearby_kin(center: Agent, radius: f32) -> List<Agent>`

Sibling of the existing `query.nearest_hostile_to` (task 163). Wraps
`SpatialHash::within_radius` with same-species filter + self-exclusion.
Returns `Vec<AgentId>` (lowered as `IrType::List(AgentId)`), sorted on
raw id (inherited from `SpatialHash`'s sort discipline).

Engine-side fn `crates/engine/src/spatial.rs::nearby_kin` — ~25 LOC,
mirrors `nearest_hostile_to`'s tie-break / dead-slot discipline. Note
on ordering: returning a sorted `Vec` matches `abilities.effects`'s
for-loop iterable shape; the physics body emits
`for kin in query.nearby_kin(dead, r) { ... }`.

### No change needed:

- **@decay pair_map view**: existing shape supports everything we need.
  `emit_fold_arm` hardcodes `amount = 1.0` (emit_view.rs:809), which
  matches our `self += 1.0`. A `half_life` variant isn't supported; we
  express the decay as `rate = 0.955, per = tick` which is
  equivalent to a ~15-tick half-life (0.955^15 ≈ 0.50).
- **Scoring `view::<name>(args) > lit : +delta`**: already works; we
  just need to register `kin_fear` in `view_id_for` + `eval_view_call`
  + the `VIEW_ID_*` constant pair (compiler + engine_rules).
- **GPU-emittable validator**: `query.nearby_kin` is a
  `NamespaceCall`, which `validate_physics_iter_source` already accepts
  as a bounded iter source (resolve.rs:2897). The list is bounded by
  the spatial-hash cell-reach cap, same as `abilities.effects`.

## 3. Species scoping: physics filters, view stays species-agnostic

Two options considered:

**A. Filter in physics** (chosen). `query.nearby_kin` does the
   creature-type match; the `FearSpread` emit-site is the only place
   species is consulted. The view fold sees only well-formed events and
   can stay ignorant of species. Matches the current pattern:
   `nearest_hostile_to` does hostility-filter in-stdlib; no view needs
   a "is_hostile" predicate to clean up its input.

**B. Filter in view**. Emit `FearSpread` for every nearby agent
   regardless of species; the view body conditionally folds. Rejected:
   view-body `on ... when <cond>` conditional folds aren't expressible
   in the current grammar (emit_fold_arm has no branch for a filtered
   fold). Adding that would grow the emitter far beyond this task.

A costs one line in `spatial.rs::nearby_kin` (a `ct_other == ct_center`
check) and matches the existing pattern. B would require new
compiler infrastructure.

## 4. LOC estimate

| Location | Change | LOC |
|---|---|---|
| `crates/engine/src/spatial.rs` | `pub fn nearby_kin` | ~28 |
| `crates/dsl_compiler/src/resolve.rs` | stdlib signature arm | 4 |
| `crates/dsl_compiler/src/emit_physics.rs` | namespace-call arm | 7 |
| `crates/dsl_compiler/src/emit_scoring.rs` | `VIEW_ID_KIN_FEAR` const + `view_id_for` arm | 3 |
| `crates/engine_rules/src/scoring/mod.rs` | mirror `VIEW_ID_KIN_FEAR` | 2 |
| `crates/engine/src/policy/utility.rs` | `eval_view_call` arm | ~20 |
| `assets/sim/events.sim` | `event FearSpread` | 1 |
| `assets/sim/views.sim` | `view kin_fear` | ~14 |
| `assets/sim/physics.sim` | `physics fear_spread_on_death` | ~10 |
| `assets/sim/scoring.sim` | `Flee` row modifier | 1 |
| **Total new compiler/parser/emit LOC** | | **~36** |
| **Total new engine runtime LOC** | | **~48** |
| **Total DSL surface changes** | | **~26** |

Total LOC change: ~110, but the gate is *compiler/parser/emit LOC*
(~36). Well under the 75-LOC go/no-go threshold.

## 5. Half-life → rate

`@decay(rate=R, per=tick)` is the only form the resolver accepts
(resolve.rs:2139). For a 15-tick half-life: `rate = 0.5^(1/15) ≈
0.9548`. Literal chosen: `0.955` (close enough at the engine's 100 ms
tick — 15 ticks ≈ 1.5 s, exactly the "brief rout" window we want).

## 6. Test plan

Add `crates/engine/tests/fear_spread_rout.rs` (new file, separate from
the wolves+humans parity test so the baseline stays isolated):

Fixture: 3 wolves at (3,0,0), (4,0,0), (5,0,0) in a tight cluster so
they're all within `nearby_kin` radius (12.0) of each other, plus 2
humans at (0,0,0), (1,0,0). Force-kill wolf A directly via
`state.kill_agent(...)` and manually emit the `AgentDied` event into
the ring before stepping — the fear_spread_on_death physics rule runs
in the event phase of the next tick.

Assertions:
1. After 1 tick post-death, `state.views.kin_fear.sum_for_first(B,
   tick) >= 1.0` and same for C.
2. After 1 tick, the scoring for Flee on B/C is higher than Attack
   (even with an adjacent human in range) — proving the scoring row
   responds to kin_fear through the pipeline.
3. After ~30 ticks (>2 half-lives), kin_fear has decayed below 0.3 and
   the Flee bonus has dropped off.
4. Species symmetry: swap roles — 3 humans, 2 wolves, kill a human,
   assert the other humans pick Flee. Confirms the view / physics
   isn't secretly wolf-only.

## 7. Replayability + parity baseline

`FearSpread` is **@replayable** (no `@non_replayable` annotation). It's
derived from `AgentDied` (replayable) and folds into a state-carrying
view; treating it as side-channel prose would break the view's
reconstructability from the event log.

Consequence: the wolves+humans parity baseline at
`crates/engine/tests/wolves_and_humans_baseline.txt` **will drift**.
The current fixture has wolves killing humans; human death doesn't
trigger wolf fear (different species), but when humans die of wolf
attacks no wolves die, so `AgentDied(wolf)` never fires — so actually
the baseline may not drift *if* no wolf dies in 100 ticks.

Checked: the baseline shows `AgentDied(tick=93,id=2)` (human),
`AgentDied(tick=93,id=3)` (human), etc. — we need to spot-check whether
wolves 4 or 5 die in the 100-tick window. If they don't,
`fear_spread_on_death` never fires and the baseline stays byte-exact.
If they do, new `FearSpread` events appear in the log and we regen
with `WOLVES_AND_HUMANS_REGEN=1` + a commit that explains the delta is
"added FearSpread fan-out from wolf death (task 167)".

Plan: assume baseline drift is possible; if tests fail, regen and
document in the implementation commit.

## 8. Go / no-go

**GO**. Compiler/parser/emit LOC estimate (~36) is well under the
75-LOC threshold. No structural changes needed to fold-body grammar,
view-storage hints, scoring-predicate grammar, or the GPU-emittability
validator — everything fits the existing surfaces.

Potential snag identified upfront: the view's `self += 0.5` literal
is **ignored** by `emit_fold_arm` (it hardcodes `amount = 1.0`). The
design compensates by writing `self += 1.0` and relying on decay +
threshold tuning for the rout window. If a future task wants per-
event fold amounts, that's a separate emitter extension.
