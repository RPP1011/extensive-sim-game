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

**Resolution:** fixed in `20f5e414` — recipients now at `[0.0, 9.999, 10.0]` with outliers at `[10.001, 20.0]`, filtering on confidence 0.8 to isolate the primary channel.

---

### 2. `tests/announce_audience.rs::announce_anyone_uses_max_announce_radius_around_speaker` — 80m not pinned

**Current assertion:** `assert_eq!(recipients, 1, "only the agent within MAX_ANNOUNCE_RADIUS")` — near agent at 50m, far at 200m. Impl constant is 80m.

**What it proves:** there *is* a radius gate, and 200m is outside it while 50m is inside.

**What it DOESN'T prove:** that the radius is 80m. Any value in `(50, 200)` passes. An off-by-2x bug (e.g. halving to 40m) would make the 50m agent stop receiving the announce — silently breaking AnnounceAudience::Anyone for common-distance dialogue — and this test would *flip* to expect 0 recipients, not 1, so the flip won't happen naturally; a reviewer who updated `MAX_ANNOUNCE_RADIUS` to 40 (by accident) could change this test's fixture to restore green.

**Specific bug that would pass this test:** change `MAX_ANNOUNCE_RADIUS` from 80.0 to 60.0 or 150.0; both values classify the (50m, 200m) pair the same way.

**Proposed fix:** add a third agent at 79.9m (should be included) and a fourth at 80.1m (should be excluded). Assert exactly 2 recipients, which pins the constant to 80m ± 0.2m.

**Commit context:** `78fa21d0` (Plan 1 Task 14)

**Resolution:** fixed in `20f5e414` — test now spawns agents at `[50, 79.9, 80.1, 200]` and asserts exactly 2 recipients.

---

### 3. `tests/action_attack_kill.rs::attack_reduces_hp_within_range` + `attack_beyond_range_is_a_no_op` — ATTACK_RANGE not pinned

**Current assertion:** in-range test uses attacker at 1m; out-of-range test uses attacker at 10m. Impl uses `<= 2.0`.

**What it proves:** the engine has *some* range gate that rejects 10m and accepts 1m.

**What it DOESN'T prove:** the range is 2.0m. Any value in `[1.0, 10.0)` passes. An impl using 5.0m range would still pass — and that's an order-of-magnitude error in the combat gating layer. Additionally, the mask-side constant (`ATTACK_RANGE_FOR_MASK` in `mask.rs`) is a *separate* literal `2.0`, also unpinned, so the two could silently drift.

**Specific bug that would pass this test:** bump `ATTACK_RANGE` from 2.0 to 8.0 in `step.rs` while leaving `ATTACK_RANGE_FOR_MASK` at 2.0 in `mask.rs`. The range-splitting mask logic would silently forbid attack at distances in (2.0, 8.0] but the kernel would still fire when a hostile backend forged an attack at those distances — i.e., mask/kernel divergence. The existing `mask_validity.rs` catches mask violations, but not the dual-constant drift itself.

**Proposed fix:** add a boundary case at 1.99m (should hit — assert hp == 90.0, check event damage) and 2.01m (should miss). This pins the kernel constant. Additionally, assert `damage == 10.0` on the `AgentAttacked` event (currently only the type is matched). Finally, a separate test should invoke `mark_attack_allowed_if_target_in_range` at 1.99m and 2.01m to pin the mask constant to the same boundary.

**Commit context:** `38d889c0 feat(engine): Attack action — 10 damage/tick within 2m, death cascade on hp≤0` (Plan 1 Task 12)

**Resolution:** fixed in `926c2207` — new `attack_at_1_99m_hits` + `attack_at_2_01m_misses` tests pin the kernel side; new `mask_attack_bit_pins_attack_range_at_2m_boundary` pins the separate mask-side constant. All existing attack tests now also extract and assert `damage == 10.0` on the AgentAttacked event.

---

### 4. `tests/announce_overhear.rs::bystander_within_overhear_range_gets_recordmemory_at_0_6_confidence` — 30m not pinned

**Current assertion:** bystander at 15m gets `confidence ≈ 0.6`; `agent_beyond_overhear_range_gets_nothing` places the distant agent at 100m. Impl `OVERHEAR_RANGE = 30.0`.

**What it proves:** `OVERHEAR_RANGE` is somewhere in `(15, 100)`.

**What it DOESN'T prove:** it's 30m. A change to 20m or 90m still passes. The confidence value 0.6 *is* pinned (good). Coupled with finding #1 this is a second, independent distance-gate that the suite doesn't anchor.

**Specific bug that would pass this test:** halve `OVERHEAR_RANGE` to 15.0 — bystanders at exactly 15m are now on the boundary. Since the test uses `15.0` and the impl uses `<=`, both 30.0 and 15.0 accept it. The bug flies through.

**Proposed fix:** place a bystander at 29.9m (must be included) and a second at 30.1m (must be excluded). Assert 1 at-0.6-confidence `RecordMemory`.

**Commit context:** `9be3ebff` (Plan 1 Task 15 — Announce overhear scan)

**Resolution:** fixed in `84ce7271` — bystanders placed at 29.9m (included) and 30.1m (excluded); assertion now pins exactly 1 recipient at 0.6 confidence.

---

### 5. `tests/invariant_pool_non_overlap.rs::*` — testing a stub against itself

**Current assertion:** `assert!(inv.check(&state, &events).is_none())` on two scenarios (healthy spawns; kill+respawn).

**What it proves:** the invariant returns `None` under healthy conditions.

**What it DOESN'T prove:** anything about pool non-overlap. Per `invariant/builtins.rs` line 66, the impl is literally `fn check(...) -> Option<Violation> { None }` — it *always* returns None regardless of input. Both tests pass trivially; a broken pool that genuinely had an alive/freelist overlap would also pass. This is the canonical "test passes because the implementation is a stub" pattern.

**Specific bug that would pass this test:** any mutation to `Pool<T>` that makes a killed slot remain alive in `is_alive()`. The stub never observes it.

**Proposed fix:** either (a) implement `PoolNonOverlapInvariant::check` to actually walk the pool's freelist vs its alive set (requires exposing pool internals; aligns with the registered failure_mode of `Panic` in debug); or (b) delete the tests and mark the invariant explicitly as `TODO: real check lands with Plan 3 pool-internals exposure`. Keeping a green test against a stub invariant gives false confidence — the file name `invariant_pool_non_overlap` implies the invariant is live.

**Commit context:** `6c0ac879 feat(engine): built-in invariants — MaskValidity (scratch-aware) + PoolNonOverlap (stub)` (Plan 2 Task 4). The commit message itself flags it as a stub — fine — but the test name does not.

**Resolution:** fixed in `bc6fac31` — option (a). `Pool<T>::is_non_overlapping` walks `alive` and `freelist` to verify no slot appears in both and the freelist has no duplicates. `SimState::pool_is_consistent` delegates. `PoolNonOverlapInvariant::check` now returns a real `Violation` on inconsistency. Two new fault-injection tests (via the doc-hidden `pool_mut_for_test` + `force_push_freelist_for_test` hooks) prove the check actually fires on corrupted states — the tests would fail if `check` were replaced with `|| true`.

---

## MEDIUM-risk findings

### 6. `tests/action_flee.rs::flee_moves_in_opposite_direction_from_threat` — speed unpinned

`assert!(pos_after.x > pos_before.x)` only proves the *sign* of motion. Any speed ∈ (0, ∞) passes. Impl uses `MOVE_SPEED_MPS = 1.0`; could be 0.001 or 1000.0 and test still green. **Fix:** assert `(pos_after.x - pos_before.x - 1.0).abs() < 1e-5` to pin speed. Same issue and fix for `step_move.rs::agent_moves_toward_nearest_other` (`pos_a.x > 0.0` proves nothing quantitative).

**Resolution:** fixed in `86638ddd` — flee test now asserts `(dx - 1.0).abs() < 1e-5` and pins y/z drift. Flee fixture moved threat/prey onto same z-plane so the away-vector is pure +x. MoveToward test asserts final position equals exactly (1.0, 0.0, 10.0).

### 7. `tests/action_needs.rs::drinking_restores_thirst` + `resting_restores_rest_timer` — event payload not verified

Both tests assert the *state* delta exactly (0.4 thirst, 0.15 rest) but only `matches!(e, Event::AgentDrank { .. })` without inspecting the `delta` field on the event. The `eating_` test *does* check `delta` — this is an inconsistency. An impl that emits `delta: DRINK_RESTORE` (0.30, raw) instead of `delta: applied` (0.30, clamped) would pass the non-saturation drink test because in the unsaturated path `applied == desired`. **Fix:** mirror the hunger test structure — `find_map` the event, extract `delta`, assert `(delta - 0.30).abs() < 1e-6` for drink, `(delta - 0.15).abs() < 1e-6` for rest, and add a saturation variant for both.

**Resolution:** fixed in `13581ef4` — drink and rest tests now extract `delta` via `find_map` and assert `(delta - 0.30).abs() < 1e-6` / `(delta - 0.15).abs() < 1e-6`. New saturation variants cover drink at thirst=0.85 (clamped delta=0.15) and rest at rest_timer=0.9 (clamped delta=0.10).

### 8. `tests/cascade_bounded.rs::release_dispatch_truncates_at_max_cascade_iterations` — loose upper bound

`assert!(n <= 9 && n >= 2)`. Actual fixed-point bound is `MAX_CASCADE_ITERATIONS = 8`, so primary dispatch + 8 iterations = exactly 9 fires for the pathological amplifier. An impl that truncated at 3 iterations (n=4) would pass, silently dropping cascades in production scenarios below the advertised depth. **Fix:** assert `n == 9` exactly (tightens the upper bound; preserves intent). Keep the lower bound only if some release-build noise is expected — which I don't think it is for a pure deterministic loop.

**Resolution:** fixed in `e8d407cd` — tightened to `assert_eq!(n, 8)`. *Audit note correction:* the expected count is 8, not 9. `run_fixed_point` counts the primary dispatch as iteration 0 (not a separate pre-iteration), so 8 iterations with 1 fire each = 8 total fires. Observed via test output: `left: 8, right: 9` on the initial tightening attempt. Constant is still `MAX_CASCADE_ITERATIONS = 8`; the count simply equals the iteration bound.

### 9. `tests/pipeline_six_phases.rs::six_phase_pipeline_runs_clean` — metric counts but no values

Asserts each of `tick_ms`, `event_count`, `agent_alive`, `mask_true_frac` was emitted 50 times. Never inspects a single value. A broken impl emitting `tick_ms = -1.0` or `event_count = 0` always or `agent_alive = 9999.0` would pass. **Fix:** assert `agent_alive` row value == 8.0 (the spawn count), that `event_count` values are non-negative, and that `mask_true_frac` values are in `[0.0, 1.0]`. At least range-checks stop the "always zero" / "always INFINITY" class of bugs.

**Resolution:** fixed in `5fa05e4b` — all three value-level assertions added. Spawn positions widened from 1m-apart to 200m-apart so agents can't close the 2m ATTACK_RANGE within 50 ticks at MOVE_SPEED_MPS=1.0 (max closure 100m per pair), preserving the `agent_alive == 8.0` invariant.

### 10. `tests/acceptance.rs::mvp_acceptance` — hash length, not content

`assert_eq!(trace_hash.len(), 32)` — a SHA-256 always produces 32 bytes, so this is a constant-true assertion. Same for `let _: f32 = dmg.value(...)` (type check, not value check). The test *times* 100×1000 and verifies the trajectory file round-trips, which is load-bearing, but nothing confirms the simulation *did anything correct* in those 1000 ticks. **Fix:** add at minimum `assert!(dmg.value(AgentId::new(1).unwrap()) > 0.0)` and `assert!(events.iter().count() > 0)`. Even better: pin the replayable hash to a golden value (as `event_ring.rs::golden_hash_anchors_format` does for a tiny fixture) — this becomes a schema-change canary for the entire MVP pipeline.

**Resolution:** fixed in `82f7de1f` — removed the vacuous hash-length check. New assertions: `events.iter().count() > 0` and at least one `Event::AgentMoved` must exist (UtilityBackend + 100 agents guarantees movement). Golden-hash pinning deferred — the existing `event_ring.rs::golden_hash_anchors_format` covers the canary angle for event encoding stability.

### 11. `tests/view_lazy.rs::invalidated_by_agent_moved` — bypasses pipeline

The test manually calls `view.invalidate_on_events(&ring)` after pushing a synthetic `AgentMoved` into a ring that was never routed through `step_full`. Also notable: `LazyView` is not wired into `step_full` at all (I checked `step.rs` — only `MaterializedView` is folded). So this test confirms the trait method, but the integration it implies (the engine automatically invalidating lazy views when events are emitted during a tick) does not exist. **Fix:** either (a) add a passing integration test that spawns two agents, runs `step_full` with UtilityBackend, and asserts the NearestEnemyLazy went stale after a movement event — this will fail today, surfacing the missing wiring; or (b) explicitly comment at the top of the test file that LazyView integration is deferred to Plan N and the current test is trait-shape only.

**Resolution:** fixed in `f75a16fd` — both (a) and (b). Header comment explains the current tests are trait-shape only because LazyView is not yet wired into `step_full`. New `#[ignore]`d test `lazy_view_wired_into_step_full` runs a real `step` with UtilityBackend causing AgentMoved and asserts staleness; it will light up (and be un-ignored) when LazyView integration lands.

### 12. `tests/announce_overhear.rs::primary_recipient_not_also_added_as_overhear_bystander` — single-recipient happy path

Exercises one primary agent at 5m with a radius-10 Area. Good that it catches dedup, but does not verify behavior when the primary-cap (32) kicks in — e.g., does an agent classified as "primary-would-have-been-but-for-cap" get the 0.6 overhear memory? The `announce_bounded_by_max_recipients` test covers this count-wise (32 + 32 = 64) but not identity-wise. **Fix:** augment `announce_bounded_by_max_recipients` to assert that the set of primary observer ids ∪ overhear observer ids = all 64 agents, and the two sets are disjoint.

**Resolution:** fixed in `84ce7271` — test now collects `primary_ids` and `overhear_ids` as `HashSet<u32>`, asserts `intersection.is_empty()` and `union == all 64 non-speaker agents`. Identity-level coverage, not just counts.

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

## Resolution summary (2026-04-19)

All 5 HIGH and all 7 MEDIUM findings fixed across 10 commits on `world-sim-bench`:

| # | Level | Finding | Commit |
|---|---|---|---|
| 1 | HIGH | Announce Area radius boundary | `20f5e414` |
| 2 | HIGH | MAX_ANNOUNCE_RADIUS=80m boundary | `20f5e414` |
| 3 | HIGH | ATTACK_RANGE=2m boundary + damage payload + mask constant | `926c2207` |
| 4 | HIGH | OVERHEAR_RANGE=30m boundary | `84ce7271` |
| 5 | HIGH | PoolNonOverlapInvariant real implementation | `bc6fac31` |
| 6 | MED | Flee + MoveToward magnitude pinning | `86638ddd` |
| 7 | MED | Drink/Rest event delta + saturation | `13581ef4` |
| 8 | MED | Cascade fire count tightened to exact=8 | `e8d407cd` |
| 9 | MED | Pipeline telemetry value assertions | `5fa05e4b` |
| 10 | MED | Acceptance proof-of-work assertions | `82f7de1f` |
| 11 | MED | LazyView integration gap documented + canary | `f75a16fd` |
| 12 | MED | Announce dedup identity assertions | `84ce7271` |

Test count: 150 → **157 (release)**, plus 1 ignored (LazyView canary). Debug build matches. Schema hash unchanged (baseline in `.schema_hash` still current). Clippy clean with `-D warnings`.

Surprises:

- **Finding #8 expected count was 9, actual is 8.** The audit author assumed "primary + 8 iterations = 9" but `run_fixed_point` folds the primary dispatch into iteration 0. `MAX_CASCADE_ITERATIONS = 8` still holds; the fire count simply equals it, not `iter + 1`. This is itself a minor documentation/naming ambiguity worth flagging.
- **Finding #9 required moving spawn positions from 1m-apart to 200m-apart** to preserve the `agent_alive == 8.0` invariant, because UtilityBackend at close range triggers Attack → AgentDied. The original test passed only because the *count* assertion didn't care.
- **HIGH #5 required exposing `Pool<T>` internals and a test-only fault-injection hook.** Used `#[doc(hidden)] pub fn ..._for_test()` rather than a Cargo feature to avoid complicating the dependency graph. The test checks would fail if the invariant body were replaced with `None`, so the circularity is broken.

LOW findings (4) explicitly skipped per scope.
