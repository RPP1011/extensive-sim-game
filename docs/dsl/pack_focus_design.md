# Pack Focus — design memo (task 169)

Goal: a wolf whose packmate is already engaged with a human should
prefer attacking that same human over a fresh human of equal hp.
Emergent pack-hunt behavior: the first wolf to engage becomes a focus
beacon; the rest converge. Symmetric across species — a human whose
ally is engaged with a wolf converges on that wolf.

## 1. Pipeline target shape

```
EngagementCommitted { actor, target, tick }
  └── physics pack_focus_on_engagement @phase(event) (new)
        for kin in query.nearby_kin(actor, radius):
          emit PackAssist { observer: kin, target }

PackAssist { observer, target, tick }
  └── view pack_focus(observer, target) @materialized @decay pair_map
        on PackAssist { observer, target } { self += 1.0 }

scoring Attack(target):
  + (if view::pack_focus(self, target) > 0.5 : +delta)
    — specific-slot (not wildcard) because we want "observer's
      pack_focus on THIS target", not "observer's total pack_focus";
      the latter would boost Attack on every candidate.
```

## 2. Candidate evaluation

### Candidate A: direct view on `engaged_with` state

Scoring modifier: `view::is_pack_engaged(self, target) > 0.5 : +delta`
where `is_pack_engaged(observer, target) = 1 iff some same-species-as-
observer is engaged with target`.

This needs a view fold whose body inspects (1) the engaging actor's
species and (2) the observer's species and conditionally folds on
match. The fold-emitter (`emit_fold_arm` in `emit_view.rs`, lines
745–844) is hard-wired: `let amount: T = 1.0;` is emitted verbatim,
with no branch for a `when <cond>` fold filter. The only things the
arm reads from the event are the (actor, target) binding positions to
build `let key = (*actor_field, *target_field)`. There is no
conditional-fold surface, no cross-entity field-read surface, and no
species-comparison surface.

Implementing Candidate A requires adding:
- view-body `on <EvPat> when <BoolExpr> { self += ... }` grammar —
  parser, IR, and emitter changes. The resolver would need to accept
  bindings from the pattern plus `self`-scope access (the view's own
  args, e.g. `observer` / `target`) inside the predicate;
- a way to read `CreatureType` of a bound agent from inside the fold
  body (new stdlib predicate `agents.same_species(a, b)` or
  structural access).

Rough LOC estimate for A: parser ~15, resolve ~40, IR ~10, emit_view
~60, plus the new stdlib primitive ~15 engine-side. Total
compiler/parser/emit LOC: ~125. **Over the 75-LOC go/no-go
threshold**, and introduces a structural grammar extension.

### Candidate B: physics-rule synthesized event (chosen)

Mirror task 167's `fear_spread_on_death`: on every
`EngagementCommitted`, iterate `query.nearby_kin(actor, radius)` and
emit one `PackAssist { observer: kin, target }` per kin. A new
`@materialized` pair_map view `pack_focus(observer, target)` folds
PackAssist into a per-(observer, target) scalar. Scoring reads it
through `view::pack_focus(self, target) > 0.5`.

Every primitive is already present:
- `query.nearby_kin(center, radius) -> List<Agent>` — shipped in
  task 167, species filter + self-exclusion + dead-center tolerance.
- `@materialized pair_map @decay` view with `self += 1.0` fold arm —
  same shape as `kin_fear`.
- Scoring's `view::<name>(self, target) > lit : +delta` — existing
  row shape (used by `my_enemies` on Attack).

No grammar changes. No new stdlib. The new machinery is *only* the
event + view + physics rule + scoring row + VIEW_ID wiring, all of
which already have shapes.

**Species scoping**: same pattern as fear_spread. `query.nearby_kin`
filters by species on the emit side, so every `PackAssist` is
observer-matching-actor-species by construction. The view itself
stays species-agnostic.

**Downside** (acknowledged in task spec): radius is arbitrary; a pack
on the far side of the map doesn't converge on a distant engagement.
Acceptable for the MVP — the intent is *local* pack hunting within
visual contact, matching fear_spread's 12 m "visual contact" figure.

## 3. Decay shape

Pack focus is a transient signal: once the first engagement happens,
kin should converge for a few ticks; after that the normal
threat_level / my_enemies retaliation views carry the grudge. We
want the boost to fade quickly enough that a stale engagement doesn't
keep pulling kin in forever.

Target: ~10-tick half-life (~1 s at 100 ms ticks). Rate R satisfying
R^10 = 0.5 → R = 0.5^(1/10) ≈ 0.933. Literal chosen: **0.933**.

Why faster than kin_fear's 15-tick half-life? Pack focus is a
*coordination nudge*, not a state change. Once packmates converge the
engagement becomes its own anchor (they'll pile via my_enemies /
threat_level from the actual hits landing), so the boost only needs
to survive long enough for movement to close the gap — ~5–10 ticks.
Beyond that, longer decay would cause kin to keep chasing a target
that's already been dogpiled / fled / killed.

## 4. Delta choice

Existing Attack row modifiers for calibration:
- `self fresh (hp_pct >= 80%)` → +0.5
- `target wounded (hp_pct < 50%)` → +0.2
- `target nearly dead (hp_pct < 30%)` → +0.4
- `my_enemies grudge hit` → +0.4
- `threat_level > 20.0` → +0.3

Pack focus should beat a fresh-full-hp alternate target but not
override a `target nearly dead` one-hit kill next door. Stacked with
the base `self fresh +0.5`:

- Fresh wolf, engaged-human-at-full-hp (+pack_focus +0.4) = **0.9**
- Fresh wolf, fresh-human-at-full-hp (no bonuses) = **0.5**

Delta = **+0.4** (matching `my_enemies` in magnitude — pack focus is
effectively "my kin have a grudge with this one"; borrowing the same
constant keeps the scoring surface calibrated and avoids adding
another arbitrary tuning dial).

## 5. Primitive inventory

| Needed | Available? | Where |
|---|---|---|
| `query.nearby_kin(center, radius) -> List<Agent>` | Yes | `crates/engine/src/spatial.rs` (task 167) |
| `@materialized @decay pair_map` view | Yes | same shape as `kin_fear`, `threat_level` |
| `self += 1.0` fold arm | Yes | `emit_fold_arm` hardcodes amount=1.0 |
| `view::<v>(self, target) > lit : +delta` | Yes | Attack row modifier grammar |
| `@replayable` event with (observer: AgentId, target: AgentId) | Yes | `FearSpread` is the template |
| Physics rule triggered on `EngagementCommitted` | New wiring, but same shape as `fear_spread_on_death` |

No grammar extensions. All infrastructure in place.

## 6. LOC estimate

| Location | Change | LOC |
|---|---|---|
| `crates/dsl_compiler/src/emit_scoring.rs` | `VIEW_ID_PACK_FOCUS` const (2 sites) + `view_id_for` arm | ~4 |
| `crates/engine/src/policy/utility.rs` | `eval_view_call` arm | ~15 |
| `assets/sim/events.sim` | `@replayable event PackAssist { observer, target }` | 1 |
| `assets/sim/views.sim` | `view pack_focus` | ~12 |
| `assets/sim/physics.sim` | `physics pack_focus_on_engagement` | ~8 |
| `assets/sim/scoring.sim` | Attack row modifier | 1 |
| `crates/engine/src/cascade/handler.rs` | `EventKindId::PackAssist` variant + `from_event` arm | 2 |
| **Total new compiler/parser/emit LOC** | | **~6** |
| **Total new engine runtime LOC (hand-written)** | | **~17** |
| **Total DSL surface changes** | | **~22** |
| **Total generated code delta** (view + physics + dispatcher) | | ~80 (auto) |

Compiler/parser/emit LOC ~6 is well under the 75-LOC go/no-go
threshold. All generated-code delta is mechanical output from
existing emitters (KinFear shape ≡ PackFocus shape; the
fear_spread_on_death emitter ≡ pack_focus_on_engagement emitter).

## 7. Replayability + parity baseline

`PackAssist` is **@replayable** (no `@non_replayable` annotation).
Derived from `EngagementCommitted` (replayable) and folds into a
state-carrying view (`pack_focus`); it MUST be in the replay log or
the view won't reconstruct.

**Parity baseline impact.** The wolves+humans fixture at
`crates/engine/tests/wolves_and_humans_baseline.txt` begins with:

```
EngagementCommitted(tick=0,actor=1,target=4)
```

…followed later by wolf deaths. Actor=1 is the human (from the
fixture spawn order). At tick 0, when the human (id=1) engages wolf
(id=4), `query.nearby_kin(human=1, 12.0)` will return the human's
*same-species* neighbours within 12 m. The fixture spawns two humans
at (0,0,0) and (1,0,0)-ish, another wolf at (-2,0,0) etc. — humans
are clustered enough to have kin.

At tick 0 the first EngagementCommitted lands, kin are looked up on
the actor, and PackAssist fires for each kin. So **the baseline will
drift** by at least 1–2 PackAssist lines. After tick 4 the wolf
(id=4) dies, EngagementBroken fires — no new commit — but any
mid-fixture engagement transitions would also emit PackAssist rows.

Plan: regen with `WOLVES_AND_HUMANS_REGEN=1` and document in the
implementation commit. The drift is additive (PackAssist lines in
new positions) and the existing event kinds stay byte-exact.

## 8. Test plan

New test file `crates/engine/tests/pack_focus.rs` (separate from
fear_spread_rout to keep the fixtures isolated):

**Fixture**: 2 wolves at (3,0,0), (5,0,0) + 1 human at (0,0,0). The
wolves are within 12 m of each other (distance 2) and of the human
(distances 3, 5).

**Direct-fold tests** (mirror fear_spread shape):
1. `spatial_kin_filter_works_on_engagement_actor` — `nearby_kin(w1)
   = [w2]`. (Sanity on the primitive.)
2. `pack_assist_fold_bumps_view` — emit a PackAssist {observer=w2,
   target=human}, fold, assert `state.views.pack_focus.get(w2, human,
   tick) ≈ 1.0 > 0.5`.
3. `pack_focus_boosts_attack_on_same_target` — baseline
   `score_row_for(Attack, w2, human) = 0.5`; after fold,
   `score_row_for(Attack, w2, human) ≈ 0.9`, delta ≈ +0.4.
4. `pack_focus_does_not_boost_other_targets` — with a second human at
   (-3,0,0), `score_row_for(Attack, w2, other_human)` is unchanged
   (pack_focus is keyed on THIS target, not wildcard).
5. `pack_focus_decays_below_threshold` — after 20 ticks (2
   half-lives ≈ 0.933^20 ≈ 0.25 < 0.5), boost vanishes.
6. `symmetric_for_humans` — swap roles, a human whose ally engages a
   wolf gets the same boost on that wolf.

**End-to-end pipeline test** (mirror pipeline_death_triggers_fear):
7. `pipeline_engagement_triggers_pack_assist` — call
   `generated::physics::pack_focus_on_engagement::pack_focus_on_engagement(
   w1, human, &mut state, &mut events)` directly, assert the
   EventRing gets one PackAssist event with `observer=w2, target=human`.

**Parity convergence test** (the "wolves converge" assertion the task
spec asks for):
8. `wolves_converge_on_engaged_human` — 2 wolves at (3,0,0) / (5,0,0)
   + 1 human at (0,0,0). Prime w1 → human engagement via
   `state.set_agent_engaged_with` + emit EngagementCommitted into the
   ring. Fold views. Assert `score_row_for(Attack, w2, human) >
   Attack(w2, Nothing) > Hold`, i.e. the second wolf's top-scored
   action targets the same human. (Don't run 20 ticks of full sim
   because the engagement dynamics of the fixture are a separate
   test surface.)

## 9. Go / no-go

**GO**. Compiler/parser/emit LOC ~6. No grammar extensions. Mirrors
task 167's fear_spread machinery one-for-one, substituting
`EngagementCommitted` for `AgentDied` as the trigger event and using
the engaging actor as the `nearby_kin` center.

## 10. Risks / snags

- **Baseline drift.** Expected; regen as noted in §7.
- **Delta calibration.** +0.4 chosen to match `my_enemies`. If
  playtesting shows wolves get *too* clumpy (e.g. ignore nearly-dead
  targets adjacent), drop to +0.3 and recalibrate. Single literal
  change in scoring.sim.
- **Radius tuning.** 12 m inherited from fear_spread — roughly
  matches `config.combat.aggro_range = 12`. If engagements happen at
  the aggro-range edge, kin just outside the same radius miss the
  beacon. Acceptable for MVP; a later refinement could read
  `config.combat.aggro_range` directly once the physics-body grammar
  accepts config field references (currently only float literals).
- **Three-agent displacement in engagement_on_move.** The engagement
  physics rule (see `engagement_on_move` in physics.sim) can fire
  two EngagementBroken followed by one EngagementCommitted in the
  same cascade. Pack assist fires on every EngagementCommitted, so
  the observer gets a fresh +1.0 each time their packmate switches
  targets. Intended: each new commitment is a new "the pack is now
  focusing on X" signal. (The alternative — "one bump per unique
  target" — would need view-side dedupe, and the per-pair key means
  re-engagement with the same target already maps to the same slot
  and just refreshes its anchor. So the current shape is self-dedup.)
