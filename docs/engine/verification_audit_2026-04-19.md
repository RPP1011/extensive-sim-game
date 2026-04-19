# Plans 1+2 Test Audit — 2026-04-19

Auditing ~150 tests across `crates/engine/tests/*` for circular
verification. Tests that pass regardless of whether the implementation
is correct.

Scope: 46 test files, ~3174 LoC. Commits `b4e31a30..53f2b6d5`. Suite
currently green on release (`cargo test -p engine --release`).

## Summary

- **HIGH risk**: 5 findings
- **MEDIUM risk**: 7 findings
- **LOW risk**: 4 findings

The most common pattern is **boundary ambiguity around distance
predicates** — test inputs sit comfortably inside (or outside) a range
gate and would continue to pass if the gate were moved 20-40% in
either direction. The engine is saturated with such gates
(`ATTACK_RANGE = 2.0`, `MAX_ANNOUNCE_RADIUS = 80.0`,
`OVERHEAR_RANGE = 30.0`, `AGGRO_RANGE = 50.0`) and none of the
Plans 1+2 tests pin any of them to their exact value.

## HIGH-risk findings

### 1. `tests/announce_audience.rs::announce_area_emits_recordmemory_for_each_agent_within_radius` — radius boundary not pinned

**Current assertion:** `assert_eq!(recipients, 3, "three agents within 10m")` — with three agents placed at distances 5, 6, 7 from `center` and two outliers at 50, 51 (radius = 10).

**What it proves:** the impl separates near agents from far agents and emits one `RecordMemory` each.

**What it DOESN'T prove:** the actual radius that the impl uses. 5/6/7 sit deeply inside any radius in `[7.001, 50)` and the outliers live outside any radius in `(7.001, 50)`. The impl's `op.distance(center) <= radius` could become `< radius - 5.0`, `< radius`, or `<= radius + 20.0` and this test would still pass.

**Specific bug that would pass this test:** change the impl from `op.distance(center) <= radius` to `op.distance(center) < radius - 2.5`. Tests still pass. Agents that sat exactly at radius 10 (an intended primary recipient per spec) would silently stop receiving `RecordMemory`.

**Proposed fix:** place recipients at distances `[0.0, 9.999, 10.0]` and outliers at `[10.001, 20.0]`, then assert exactly 3 recipients. That pins `<=` vs `<` and the exact constant.

**Commit context:** `78fa21d0 feat(engine): Announce — enumerate audience, emit RecordMemory per recipient` (Plan 1 Task 14)

---

### 2. `tests/announce_audience.rs::announce_anyone_uses_max_announce_radius_around_speaker` — 80m not pinned

**Current assertion:** `assert_eq!(recipients, 1, "only the agent within MAX_ANNOUNCE_RADIUS")` — near agent at 50m, far at 200m. Impl constant is 80m.

**What it proves:** there *is* a radius gate, and 200m is outside it while 50m is inside.

**What it DOESN'T prove:** that the radius is 80m. Any value in `(50, 200)` passes. An off-by-2x bug (e.g. halving to 40m) would make the 50m agent stop receiving the announce — silently breaking AnnounceAudience::Anyone for common-distance dialogue — and this test would *flip* to expect 0 recipients, not 1, so the flip won't happen naturally; a reviewer who updated `MAX_ANNOUNCE_RADIUS` to 40 (by accident) could change this test's fixture to restore green.

**Specific bug that would pass this test:** change `MAX_ANNOUNCE_RADIUS` from 80.0 to 60.0 or 150.0; both values classify the (50m, 200m) pair the same way.

**Proposed fix:** add a third agent at 79.9m (should be included) and a fourth at 80.1m (should be excluded). Assert exactly 2 recipients, which pins the constant to 80m ± 0.2m.

**Commit context:** `78fa21d0` (Plan 1 Task 14)

---

### 3. `tests/action_attack_kill.rs::attack_reduces_hp_within_range` + `attack_beyond_range_is_a_no_op` — ATTACK_RANGE not pinned

**Current assertion:** in-range test uses attacker at 1m; out-of-range test uses attacker at 10m. Impl uses `<= 2.0`.

**What it proves:** the engine has *some* range gate that rejects 10m and accepts 1m.

**What it DOESN'T prove:** the range is 2.0m. Any value in `[1.0, 10.0)` passes. An impl using 5.0m range would still pass — and that's an order-of-magnitude error in the combat gating layer. Additionally, the mask-side constant (`ATTACK_RANGE_FOR_MASK` in `mask.rs`) is a *separate* literal `2.0`, also unpinned, so the two could silently drift.

**Specific bug that would pass this test:** bump `ATTACK_RANGE` from 2.0 to 8.0 in `step.rs` while leaving `ATTACK_RANGE_FOR_MASK` at 2.0 in `mask.rs`. The range-splitting mask logic would silently forbid attack at distances in (2.0, 8.0] but the kernel would still fire when a hostile backend forged an attack at those distances — i.e., mask/kernel divergence. The existing `mask_validity.rs` catches mask violations, but not the dual-constant drift itself.

**Proposed fix:** add a boundary case at 1.99m (should hit — assert hp == 90.0, check event damage) and 2.01m (should miss). This pins the kernel constant. Additionally, assert `damage == 10.0` on the `AgentAttacked` event (currently only the type is matched). Finally, a separate test should invoke `mark_attack_allowed_if_target_in_range` at 1.99m and 2.01m to pin the mask constant to the same boundary.

**Commit context:** `38d889c0 feat(engine): Attack action — 10 damage/tick within 2m, death cascade on hp≤0` (Plan 1 Task 12)

---

### 4. `tests/announce_overhear.rs::bystander_within_overhear_range_gets_recordmemory_at_0_6_confidence` — 30m not pinned

**Current assertion:** bystander at 15m gets `confidence ≈ 0.6`; `agent_beyond_overhear_range_gets_nothing` places the distant agent at 100m. Impl `OVERHEAR_RANGE = 30.0`.

**What it proves:** `OVERHEAR_RANGE` is somewhere in `(15, 100)`.

**What it DOESN'T prove:** it's 30m. A change to 20m or 90m still passes. The confidence value 0.6 *is* pinned (good). Coupled with finding #1 this is a second, independent distance-gate that the suite doesn't anchor.

**Specific bug that would pass this test:** halve `OVERHEAR_RANGE` to 15.0 — bystanders at exactly 15m are now on the boundary. Since the test uses `15.0` and the impl uses `<=`, both 30.0 and 15.0 accept it. The bug flies through.

**Proposed fix:** place a bystander at 29.9m (must be included) and a second at 30.1m (must be excluded). Assert 1 at-0.6-confidence `RecordMemory`.

**Commit context:** `9be3ebff` (Plan 1 Task 15 — Announce overhear scan)

---

### 5. `tests/invariant_pool_non_overlap.rs::*` — testing a stub against itself

**Current assertion:** `assert!(inv.check(&state, &events).is_none())` on two scenarios (healthy spawns; kill+respawn).

**What it proves:** the invariant returns `None` under healthy conditions.

**What it DOESN'T prove:** anything about pool non-overlap. Per `invariant/builtins.rs` line 66, the impl is literally `fn check(...) -> Option<Violation> { None }` — it *always* returns None regardless of input. Both tests pass trivially; a broken pool that genuinely had an alive/freelist overlap would also pass. This is the canonical "test passes because the implementation is a stub" pattern.

**Specific bug that would pass this test:** any mutation to `Pool<T>` that makes a killed slot remain alive in `is_alive()`. The stub never observes it.

**Proposed fix:** either (a) implement `PoolNonOverlapInvariant::check` to actually walk the pool's freelist vs its alive set (requires exposing pool internals; aligns with the registered failure_mode of `Panic` in debug); or (b) delete the tests and mark the invariant explicitly as `TODO: real check lands with Plan 3 pool-internals exposure`. Keeping a green test against a stub invariant gives false confidence — the file name `invariant_pool_non_overlap` implies the invariant is live.

**Commit context:** `6c0ac879 feat(engine): built-in invariants — MaskValidity (scratch-aware) + PoolNonOverlap (stub)` (Plan 2 Task 4). The commit message itself flags it as a stub — fine — but the test name does not.

---

## MEDIUM-risk findings

### 6. `tests/action_flee.rs::flee_moves_in_opposite_direction_from_threat` — speed unpinned

`assert!(pos_after.x > pos_before.x)` only proves the *sign* of motion. Any speed ∈ (0, ∞) passes. Impl uses `MOVE_SPEED_MPS = 1.0`; could be 0.001 or 1000.0 and test still green. **Fix:** assert `(pos_after.x - pos_before.x - 1.0).abs() < 1e-5` to pin speed. Same issue and fix for `step_move.rs::agent_moves_toward_nearest_other` (`pos_a.x > 0.0` proves nothing quantitative).

### 7. `tests/action_needs.rs::drinking_restores_thirst` + `resting_restores_rest_timer` — event payload not verified

Both tests assert the *state* delta exactly (0.4 thirst, 0.15 rest) but only `matches!(e, Event::AgentDrank { .. })` without inspecting the `delta` field on the event. The `eating_` test *does* check `delta` — this is an inconsistency. An impl that emits `delta: DRINK_RESTORE` (0.30, raw) instead of `delta: applied` (0.30, clamped) would pass the non-saturation drink test because in the unsaturated path `applied == desired`. **Fix:** mirror the hunger test structure — `find_map` the event, extract `delta`, assert `(delta - 0.30).abs() < 1e-6` for drink, `(delta - 0.15).abs() < 1e-6` for rest, and add a saturation variant for both.

### 8. `tests/cascade_bounded.rs::release_dispatch_truncates_at_max_cascade_iterations` — loose upper bound

`assert!(n <= 9 && n >= 2)`. Actual fixed-point bound is `MAX_CASCADE_ITERATIONS = 8`, so primary dispatch + 8 iterations = exactly 9 fires for the pathological amplifier. An impl that truncated at 3 iterations (n=4) would pass, silently dropping cascades in production scenarios below the advertised depth. **Fix:** assert `n == 9` exactly (tightens the upper bound; preserves intent). Keep the lower bound only if some release-build noise is expected — which I don't think it is for a pure deterministic loop.

### 9. `tests/pipeline_six_phases.rs::six_phase_pipeline_runs_clean` — metric counts but no values

Asserts each of `tick_ms`, `event_count`, `agent_alive`, `mask_true_frac` was emitted 50 times. Never inspects a single value. A broken impl emitting `tick_ms = -1.0` or `event_count = 0` always or `agent_alive = 9999.0` would pass. **Fix:** assert `agent_alive` row value == 8.0 (the spawn count), that `event_count` values are non-negative, and that `mask_true_frac` values are in `[0.0, 1.0]`. At least range-checks stop the "always zero" / "always INFINITY" class of bugs.

### 10. `tests/acceptance.rs::mvp_acceptance` — hash length, not content

`assert_eq!(trace_hash.len(), 32)` — a SHA-256 always produces 32 bytes, so this is a constant-true assertion. Same for `let _: f32 = dmg.value(...)` (type check, not value check). The test *times* 100×1000 and verifies the trajectory file round-trips, which is load-bearing, but nothing confirms the simulation *did anything correct* in those 1000 ticks. **Fix:** add at minimum `assert!(dmg.value(AgentId::new(1).unwrap()) > 0.0)` and `assert!(events.iter().count() > 0)`. Even better: pin the replayable hash to a golden value (as `event_ring.rs::golden_hash_anchors_format` does for a tiny fixture) — this becomes a schema-change canary for the entire MVP pipeline.

### 11. `tests/view_lazy.rs::invalidated_by_agent_moved` — bypasses pipeline

The test manually calls `view.invalidate_on_events(&ring)` after pushing a synthetic `AgentMoved` into a ring that was never routed through `step_full`. Also notable: `LazyView` is not wired into `step_full` at all (I checked `step.rs` — only `MaterializedView` is folded). So this test confirms the trait method, but the integration it implies (the engine automatically invalidating lazy views when events are emitted during a tick) does not exist. **Fix:** either (a) add a passing integration test that spawns two agents, runs `step_full` with UtilityBackend, and asserts the NearestEnemyLazy went stale after a movement event — this will fail today, surfacing the missing wiring; or (b) explicitly comment at the top of the test file that LazyView integration is deferred to Plan N and the current test is trait-shape only.

### 12. `tests/announce_overhear.rs::primary_recipient_not_also_added_as_overhear_bystander` — single-recipient happy path

Exercises one primary agent at 5m with a radius-10 Area. Good that it catches dedup, but does not verify behavior when the primary-cap (32) kicks in — e.g., does an agent classified as "primary-would-have-been-but-for-cap" get the 0.6 overhear memory? The `announce_bounded_by_max_recipients` test covers this count-wise (32 + 32 = 64) but not identity-wise. **Fix:** augment `announce_bounded_by_max_recipients` to assert that the set of primary observer ids ∪ overhear observer ids = all 64 agents, and the two sets are disjoint.

## LOW-risk findings

| Test | Issue | 1-line fix |
|---|---|---|
| `action_attack_kill.rs::attack_dead_target_is_no_op` | Uses `events.len()` (ring size) — vulnerable to ring-eviction if cap shrinks | Compare exact event sequence via `iter().collect()` snapshot |
| `telemetry_file_sink.rs::file_sink_writes_json_lines` | `contains("\"value\":42")` would match `"value":420` too | Parse with `serde_json` and assert field equality |
| `step_basic.rs::step_is_reproducible_for_same_seed` | Hold-only population has empty event rings; `trace(42) == trace(42)` holds even if the impl is fully broken (both are empty hashes) | Comment is honest about this; keep or replace with 2-agent MoveToward (now feasible post-Task 11) |
| `cascade_register_dispatch.rs::multiple_handlers_same_kind_all_fire` | Uses `damage: 0.0` attack — harmless but conceals what an AgentAttacked event means | Use `damage: 10.0` and include a hp-state sanity assertion |

## Non-findings

Checked and concluded safe:

- `tests/event_ring.rs::golden_hash_anchors_format` — pins exact SHA-256; a structural drift in any event variant encoding breaks it. Good canary.
- `tests/event_ring.rs::agent_attacked_hashes_stably_with_float_damage` — distinguishes +0.0 / -0.0 via `to_bits()`; real boundary test.
- `tests/schema_hash.rs::schema_hash_matches_baseline` — baseline pinned externally to `.schema_hash`. Good.
- `tests/spatial_index.rs` — exercise *does* distinguish 2D vs 3D distance via the `column_query_excludes_distant_flyers` case (flyer at z=50 would be 2D-near but 3D-far). Honest boundary test.
- `tests/view_topk.rs::topk_bounded_keeps_highest_scoring_attackers` — pins exact scores (60.0 top, 30.0 bottom) and K=4; well-specified.
- `tests/acceptance_plan2_deterministic.rs` + `tests/acceptance_mixed_actions.rs` — explicitly scoped to determinism; `same_seed_same_hash` + `different_seed_different_hash` is the correct contract for these files.
- `tests/mask_builder.rs::mark_hold_sets_only_hold_bit_per_alive_agent` — checks both positive (Hold set) and negative (all other bits false, all dead-slot rows false). Well-specified.
- `tests/policy_utility.rs::utility_prefers_eat_when_hp_low_and_eat_allowed` — pins exact HP threshold behavior via hp=20 < 30% of 100. Threshold is thus anchored at ≤ 30%; not at some unknown point in a wide range.
- `tests/determinism.rs::different_seeds_diverge_under_load` — 100 agents × 1000 ticks with UtilityBackend; seed-divergence is observable via phase-3 shuffle propagating into position events.
- `tests/cascade_lanes.rs::lanes_run_in_fixed_order_regardless_of_registration` — registration-order stress coupled with observed lane order. Good.
- `tests/invariant_dispatch_modes.rs::violated_panic_mode_panics_immediately` — `#[should_panic(expected = ...)]` anchored to the exact message substring. Good.

## Top 3 HIGH-risk fixes, in priority order

1. **Pin ATTACK_RANGE** via boundary agents at 1.99m / 2.01m and assert damage payload (finding #3). Combat gating is the single highest-stakes magic number in the sim.
2. **Implement or mark-stub `PoolNonOverlapInvariant`** (finding #5). A greenlit invariant test against a constantly-`None` stub is actively deceptive.
3. **Pin MAX_ANNOUNCE_RADIUS / OVERHEAR_RANGE / announce Area radius** at their advertised values (findings #1, #2, #4). Memory propagation distances are the core of the information-spread system.
