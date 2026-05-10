# Plan H — Threat Stresstest + Scoring Effectiveness

> Status: design. Implementation lands after Plan G G3 completes
> (threats view + scoring primitives merged to main).

## Goal

Measure how much the threats infrastructure changes AI behaviour.
Two threat lifetimes, each with different scoring impact:

* **Short-term threats** — active casts (3-10 ticks, populated by
  `EffectCastBeginApplied`, decay at cast resolution / interruption).
  Usage: dodger AI moves out of telegraphed Firebolt zone.
* **Long-term threats** — persistent grudges (10k+ ticks, populated
  by `CompetitorObserved` / `AuctionLost` / `PriceUndercut` events,
  decay slowly). Usage: merchant prioritizes raids on rival caravans
  weeks after the grudge formed.

The fixture proves the threats infrastructure scales across the
short/long axis without bespoke code per use case.

## Architectural Impact Statement

* **P1 (Compiler-First):** PASS. Both threat shapes use the same
  `@per_entity_ring(K)` storage hint with `@dispatch(per_agent_event_scan)`.
  Per-fixture sim rules populate them; no `impl Rule` in
  `crates/engine/src/handlers/`.
* **P2 (Schema-Hash):** No bump. Reuses the threat-cell struct
  layout from G3b. Decay timing is per-cell (`expires_at_tick`
  field) — no new SoA columns.
* **P3 (Cross-Backend Parity):** Tested via the `apply_program`
  sweep + a new fold-body parity arm for the threats view's append
  + decay primitives.
* **P5 (Determinism):** Decay decisions are pure functions of
  `(world.tick, cell.expires_at_tick)` — no RNG.
* **P11 (Reduction Determinism):** Cursor's atomicAdd serializes
  ring-slot allocation per writer; cell writes are non-atomic but
  race-free because each (target, ring slot) gets exactly one
  writer per tick.

## Fixture shape

`assets/sim/threat_stresstest.sim`:

* **256 agents.** 64 casters (mid-cast bosses), 64 brawlers (active
  threat producers), 64 merchants (long-term grudge holders), 64
  observers (the AI being measured).
* **Two threats views:**
  * `cast_threats(observer: Agent)` — `@per_entity_ring(K = 8)`,
    cells: `(zone_kind, center_x_q8, center_y_q8, radius_q8,
    expires_at_tick, source)`. Populated by EffectCastBeginApplied
    on every CASTER agent (busy_with_ability_id != 0). Decay at
    `world.tick >= cell.expires_at_tick` — slot becomes
    "inactive" but stays in ring until overwritten by a new
    threat.
  * `grudges(observer: Agent)` — `@per_entity_ring(K = 16)`,
    cells: `(grudge_kind, target_agent, intensity_q8,
    expires_at_tick)`. Populated by per-fixture `RecordGrudge`
    rule that fires on `BargainBroken` / `OutbidSimulated` /
    `MerchantSlightedSimulated` — 3 scripted event types simulating
    "competitor identified", "lost auction", "underpriced by".
* **Two scoring verbs:**
  * `Dodge(self, target: Vec3)` — score reads
    `cast_threats.intensity_at(target)`. Higher = stronger pressure
    to dodge. Compares with and without the read to measure
    behavioural delta.
  * `RaidCaravan(self, target: Agent)` — score reads
    `grudges.intensity_against(self, target.owner)`. Without
    grudge data, base score is uniform across caravans; with
    grudge data, the target weighting biases toward the agent's
    rival.

## Effectiveness measurements

The fixture runs in two configurations:

1. **Baseline (no threats):** verbs score with the threat reads
   replaced by `0.0` constants. Agents make decisions blind to the
   threat surface.
2. **Threat-aware:** verbs score with the actual threat reads.

For each configuration, drive the fixture for `T = 1024` ticks at
the full 256-agent shape. Record per-tick, per-observer:

* Verb decisions taken (which Dodge / RaidCaravan target).
* Damage taken (for Dodge measurement).
* Caravan-raid hit rate against rival vs random target (for
  RaidCaravan measurement).

Effectiveness metrics:

* **Dodge delta:** mean damage taken per observer = baseline_dmg -
  threat_aware_dmg. Expect threat_aware to take less damage if the
  scoring + dodge mechanic actually translate cast_threats into
  movement.
* **Grudge delta:** ratio of caravan raids landing on the agent's
  declared rival = threat_aware_rival_hits / total_raids -
  baseline_rival_hits / total_raids. Expect threat_aware to raid
  rivals more often than random.
* **Scaling:** wall-clock time per tick at 256 agents. Plan H's
  ring storage scales O(N×K) per view; verify both folds dispatch
  in <2ms per tick at 256 agents.

The numbers go into a `tests/threat_effectiveness.rs` integration
test that runs both configurations and asserts the deltas are at
least `MIN_DODGE_DELTA = 5%` damage reduction and `MIN_GRUDGE_DELTA = 30%`
rival-bias increase. Failure means the scoring primitives don't
actually translate threat data into behaviour change.

## Implementation slices

1. **H1 — Fixture authoring.** `assets/sim/threat_stresstest.sim`
   with the 4 entity types + 2 threats views + 2 scoring verbs +
   the 4 producer rules (DispatchCast, RecordCastBegin, RecordGrudge,
   ResolveBusy).
2. **H2 — Per-runtime crate.** `crates/threat_stresstest_runtime`
   builds the .sim, exposes baseline vs threat-aware mode toggle,
   readback APIs for damage taken + raid targets per agent.
3. **H3 — Effectiveness integration test.** Drives both modes for
   1024 ticks, computes the deltas, asserts the thresholds.
4. **H4 — Performance pin.** Per-tick wall-clock measurement; budget
   pin (e.g. 2ms / tick at 256 agents).

H1+H2 ~3 hours; H3 ~1 hour; H4 ~30 min. Total 4-5 hours.
