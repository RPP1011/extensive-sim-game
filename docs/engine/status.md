# Engine Status (as of 2026-04-19)

> The single source of truth for what's built, what to verify, and what to worry about.
> Design lives in `spec.md`. Implementation intent lives in `docs/superpowers/plans/`.
> This doc is the user's jump-off point: "Is section X trustworthy? What would prove it's not?"

After MVP + Plans 1 & 2: 150 green tests across ~20 commits. All subsystems below are
Serial-only; no GPU backend yet. Cross-backend parity lands with Plan 5+.

## Legend

- **Status**: ❌ not started / ⚠️ partial / ✅ implemented (tests green) / 🔍 eyeball-verified / 🎯 externally-verified
- **Visual check**: what someone running `cargo run -p viz` should see that would catch bugs.
  Viz harness itself is not yet written (awaiting Plan 3.0 viz draft); visual checks in
  this doc are the acceptance criteria that harness has to satisfy.

## Plans index

| Plan | Doc | Status |
|---|---|---|
| MVP | `docs/superpowers/plans/2026-04-19-world-sim-engine-mvp.md` | ✅ complete |
| Plan 1 — action space & cascade | `docs/superpowers/plans/2026-04-19-engine-plan-1-action-space.md` | ✅ executed (Tasks 1–18) |
| Plan 2 — pipeline + cross-cutting traits | `docs/superpowers/plans/2026-04-19-engine-plan-2-pipeline-traits.md` | ✅ executed |
| Plan 3 — persistence + obs packer + probes | `docs/superpowers/plans/2026-04-19-engine-plan-3-persistence-obs-probes.md` | ⚠️ draft — awaiting execution |
| Plan 3.0 viz harness | _(to be written)_ | ❌ awaiting draft |
| Plan 4 — debug & trace runtime | _(to be written)_ | ❌ not yet written |
| Plan 5 — `ComputeBackend` trait extraction | _(to be written)_ | ❌ not yet written |
| Plan 6 — `GpuBackend` foundation | _(to be written)_ | ❌ not yet written |
| Plan 7+ — per-kernel GPU porting | _(to be written)_ | ❌ not yet written |

## Subsystem table

Cross-reference: "Tests" column paths are relative to `crates/engine/`. Commits use the
short SHAs from `git log --oneline 64675559..HEAD -- crates/engine/`.

| Spec § | Subsystem | Status | Plan | Code | Tests | Weak-test risk | Visual check |
|---|---|---|---|---|---|---|---|
| §3, §4 | `Pool<T>` + `PoolId<T>` (generic, `NonZeroU32`, freelist) | ✅ | P1 T1 (`b4e31a30`) | `src/pool.rs` | `tests/pool_generic.rs`, `tests/state_agent.rs::kill_frees_slot` | Single kill+spawn only proves one-slot reuse; no test stress-kills ≥ 2 slots and asserts FIFO-vs-LIFO order on the freelist. `kill_then_alloc_reuses_slot` would pass whether freelist is Vec/Stack/HashSet. | 200-tick churn (50 spawns / 50 kills per tick) — total alive count never exceeds cap, IDs recycle to low range (verifiable via agent histogram). |
| §3 | SoA hot state (`pos`, `hp`, `alive`, `movement_mode`, `hunger`, `thirst`, `rest_timer`) | ✅ | P1 T2 (`bbe93150`) | `src/state/mod.rs`, `src/state/agent.rs` | `tests/state_agent.rs`, `tests/state_needs.rs` | `hot_slices_are_independent_vecs` only checks `hp_addr != pos_addr`. Doesn't catch a bug where two fields share a Vec but get sliced at different offsets. | N/A (internal layout). |
| §4 | `AgentId`, `EventId`, `AggregateId`, `QuestId`, `GroupId`, `ResourceRef`, `ItemId` (all `NonZeroU32`) | ✅ | P1 T3 (`31e45d16`) | `src/ids.rs` | `tests/event_id_threading.rs` | `cause_of` sidecar test only checks cause field doesn't affect hash on a 2-event ring. Doesn't verify cause survives ring-buffer overflow eviction of the parent. | N/A (structural). |
| §5 | `EventRing` (`VecDeque<EventEntry>`, ring drop on overflow, per-tick seq) | ✅ | MVP, P1 T3 extends it | `src/event/ring.rs`, `src/event/mod.rs` | `tests/event_ring.rs`, `tests/event_id_threading.rs`, `tests/determinism.rs` | `cause_of` is O(N) linear scan — no test verifies correctness when ring has evicted the cause entry (returns None vs. stale EventId). | Event log should show monotonically increasing tick numbers; within a tick, seq should start at 0 and increment. |
| §6 | Spatial index (2D-column BTreeMap + z-sort + MovementMode sidecar) | ✅ | MVP | `src/spatial.rs` | `tests/spatial_index.rs` | Only 3 test cases — no proptest / fuzz; planar vs. 3D boundary tests use distances 7m and 40m (far from the 10m radius boundary). A bug rounding 10m → 9.5 or 10.5 would still pass. | Fly agents 15m above walker should be hit by 10m 3D-radius query but miss a 5m one. |
| §7 | RNG (`WorldRng` PCG-XSH-RR, `per_agent_u32`, keyed sub-streams) | ✅ | MVP | `src/rng.rs` | `tests/determinism.rs` (indirect) | No direct golden test of `per_agent_u32(42, AgentId(1), 100, b"action")` against a pinned expected u32. Any re-keying of the PCG constants that still produces deterministic output (but different values) would pass the existing suite because no test hashes a known-good RNG byte stream. | N/A. |
| §9 | `MicroKind` — 18 variants (`Hold`..`Remember`) | ✅ | P1 T4 (`01392efc`) | `src/mask.rs` | `tests/micro_kind_full.rs` | Only spot-checks 7 of 18 ordinals (0, 1, 2, 3, 15, 16, 17). A swap between Cast(4)↔UseItem(5) or Harvest(6)↔Eat(7) would pass. | N/A. |
| §9 | MicroKind execution: Hold / MoveToward / Flee / Attack / Eat / Drink / Rest | ✅ | P1 T6–T9 (`84efa271`, `38d889c0`, `c76a1d24`) | `src/step.rs` | `tests/action_flee.rs`, `tests/action_attack_kill.rs`, `tests/action_needs.rs`, `tests/step_move.rs` | **Flee test** (`flee_moves_in_opposite_direction_from_threat`) only asserts `pos_after.x > pos_before.x` — doesn't pin the magnitude. Flee speed 0.001 or 100 would pass. **Move test** same issue (`pos_a.x > 0.0`). **Attack range** uses 1m (well inside 2m) and 10m (well outside) — a bug rounding ATTACK_RANGE 2.0 → 1.5 or 2.5 would still pass. | (1) Attacker 1m from target should drop target HP by exactly 10 per tick. (2) Attacker 3m away should do nothing (outside 2m range). (3) Flee from co-located threat should stay put, not tremble. (4) Eat with hunger=0.9 should saturate to 1.0 within one tick. |
| §9 | 11 event-only micros (Cast, UseItem, Harvest, PlaceTile/Voxel, HarvestVoxel, Converse, ShareStory, Communicate, Ask, Remember) | ✅ | P1 T10 (`c1be86f3`) | `src/step.rs` | `tests/action_emit_only.rs` | Each test only asserts `.any(matches!(...))`. No test asserts *count* = 1, so a duplicate-emit bug (event fired twice) would pass. | N/A (event-only — no visual state change). |
| §10 | `MacroKind` — 5 variants (`PostQuest`, `AcceptQuest`, `Bid`, `Announce`, `NoOp`) + parameter enums | ✅ | P1 T5, T12 (`42b063b1`, `e83c9d0c`) | `src/policy/macro_kind.rs`, `src/policy/query.rs` | `tests/macro_kind.rs`, `tests/macro_emit_only.rs` | `noop_macro_emits_nothing` filters only on Quest/Bid events — a bug emitting `AnnounceEmitted` for NoOp would pass. | N/A. |
| §10 | Announce cascade (audience enumeration, primary + overhear) | ✅ | P1 T14, T15 (`78fa21d0`, `e49208e1`) | `src/step.rs::429-507` | `tests/announce_audience.rs`, `tests/announce_overhear.rs` | Uses 3D distance (`Vec3::distance`) — spec §10 is silent on 3D vs planar. Tests place all agents on z=0 or z=10 uniformly, so a bug using planar distance would pass every current test. | Speaker at (0,0,10); listener at (5,0,50) — 3D distance 40m. Announce Area radius 30 should NOT reach the listener. If it does, impl uses planar. |
| §11 | `CascadeRegistry` + `CascadeHandler` trait + `Lane` ordering (Validation → Effect → Reaction → Audit) | ✅ | P1 T11 (`155a51df`) | `src/cascade/` | `tests/cascade_register_dispatch.rs`, `tests/cascade_lanes.rs` | `within_a_lane_registration_order_preserved` only checks count, not observable order. A reorder bug inside a lane wouldn't be caught. | N/A. |
| §11 | Cascade `run_fixed_point` with `MAX_CASCADE_ITERATIONS=8` | ✅ | P1 T13 (`0bc30cca`) | `src/cascade/dispatch.rs` | `tests/cascade_bounded.rs` | Release-mode truncation test asserts `2 <= n <= 9` — an impl with only 3 iterations before truncation would pass. Nothing pins the exact cap at 8. | N/A. |
| §12 | `MaskBuffer` (18-bit-per-agent head) + `mark_hold_allowed` / `mark_move_allowed_*` | ✅ | MVP | `src/mask.rs` | `tests/mask_builder.rs`, `tests/mask_validity.rs` | Mask builders are tested only against UtilityBackend output. No test forges a case where e.g. `mark_move_allowed_if_others_exist` should be false (single agent present) but mask sets it true anyway. | N/A. |
| §13 | `PolicyBackend` trait + `UtilityBackend` scalar impl | ✅ | MVP | `src/policy/utility.rs`, `src/policy/mod.rs` | `tests/policy_utility.rs`, `tests/mask_validity.rs` | UtilityBackend extension (P1 T17) only scores 4 of 18 micros; the other 14 rely on mask-bit=false to stay unemitted. No invariant-forced test proves those 14 can't leak with a re-weighting. | Run UtilityBackend 1000 ticks, collect action-kind histogram — should be dominated by Hold / MoveToward / Attack (and Eat/Drink/Rest when needs drop). No stray Cast / Harvest / Converse. |
| §14 | 6-phase tick pipeline (`step_full`): mask → policy → shuffle → apply+cascade → views → invariants+telemetry | ✅ | P2 T5 (`0771d16c`) | `src/step.rs::step_full` | `tests/pipeline_six_phases.rs`, `tests/acceptance_plan2_deterministic.rs` | `pipeline_six_phases.rs` only checks that each of 4 built-in metrics fires exactly 50 times across 50 ticks. A bug where phase 5 (view fold) is silently skipped would pass — no view is asserted to have changed after the step. | Agent HP should drop across ticks when adjacent enemies attack (verified end-to-end). Viz: mixed-action scenario overlay showing per-tick action histogram + cumulative damage. |
| §14 | Phase 3 (Fisher-Yates shuffle keyed on `per_agent_u32(seed, AgentId(1), tick<<16, b"shuffle")`) | ✅ | MVP | `src/step.rs::shuffle_actions_in_place` | `tests/determinism.rs` | Determinism tested via hash equality, but there's no test asserting the shuffle *actually* produces different orderings across ticks. A no-op shuffle would still be deterministic. | N/A. |
| §15 | `MaterializedView` trait + `DamageTaken` example | ✅ | MVP | `src/view/materialized.rs`, `src/view/mod.rs` | `tests/view_materialized.rs` | Two-event test; no test of integer-vs-float reduction determinism (commutativity under reorder). | N/A. |
| §15 | `LazyView` trait + `NearestEnemyLazy` + staleness flag | ✅ | P2 T1 (`a1aad00b`) | `src/view/lazy.rs` | `tests/view_lazy.rs` | `compute_populates_and_marks_fresh` uses only 2 agents. A bug where `value()` always returns the same ID regardless of query would pass (test asserts `value(a) == Some(b)` and `value(b) == Some(a)` — symmetric by coincidence of N=2). | N/A. |
| §15 | `TopKView` trait + `MostHostileTopK` (cumulative-damage heap) | ✅ | P2 T2 (`aee396a5`) | `src/view/topk.rs` | `tests/view_topk.rs` | `topk_bounded_keeps_highest_scoring_attackers` asserts top[0]=60, top[3]=30 (K=4, 6 attackers with damage 10/20/30/40/50/60). Boundary between top-4 (30) and out (20) is 10 apart — a bug using `<` vs `<=` on the eviction threshold wouldn't fire here. | N/A. |
| §16 | `AggregatePool<T>` + `Quest` / `Group` Pod-compatible shapes | ✅ | P1 T16 (`c438f249`) | `src/aggregate/` | `tests/aggregate_pool.rs`, `tests/aggregate_types.rs` | `kill_then_alloc_reuses_slot_and_clears_contents` — same single-reuse issue as Pool. Also no test registers a cascade handler that writes to `AggregatePool` and reads it back on the next iteration. | N/A (no visual for aggregates). |
| §17 | Trajectory emission (safetensors; N-tick windowing) | ✅ | MVP | `src/trajectory.rs` | `tests/trajectory_roundtrip.rs` | Python roundtrip test compares only `n_agents` and `n_ticks` — doesn't compare per-tick position/hp data. A silent trajectory corruption that preserves shape would pass. | N/A. |
| §18 | Save / load | ❌ | P3 | — | — | Not started. | N/A. |
| §19 | `Invariant` trait + `Violation` + `FailureMode` | ✅ | P2 T3 (`f8c23715`) | `src/invariant/trait_.rs` | `tests/invariant_trait.rs`, `tests/invariant_dispatch_modes.rs` | — | N/A. |
| §19 | `InvariantRegistry::check_all` + dispatch by failure mode | ✅ | P2 T3 (`21537e11`) | `src/invariant/registry.rs` | `tests/invariant_dispatch_modes.rs` | — | N/A. |
| §19 | Built-in invariants: `MaskValidityInvariant`, `PoolNonOverlapInvariant` | ⚠️ | P2 T3 (`6c0ac879`) | `src/invariant/builtins.rs` | `tests/invariant_mask_validity.rs`, `tests/invariant_pool_non_overlap.rs` | **`PoolNonOverlapInvariant::check` is a stub that always returns `None`** (see `builtins.rs:60-67` — docstring says "Stub for now"). All tests pass trivially. Real pool-overlap check is not wired. Also `MaskValidityInvariant::check` (the trait impl) returns `None`; the real check only happens via `check_with_scratch`, so registering it in `InvariantRegistry::check_all` is a no-op at the invariant-trait level. | N/A. |
| §18 | Probe harness | ❌ | P3 | — | — | Not started. | N/A. |
| §20 | Schema hash (`sha2` over layout fingerprint + `.schema_hash` baseline file) | ✅ | MVP + P1 T4 + P2 T4 | `src/schema_hash.rs`, `.schema_hash` | `tests/schema_hash.rs` | Baseline-comparison test catches any hash drift but doesn't prove the fingerprint string covers every layout-relevant type (e.g., nothing asserts `ResourceRef` or `ItemId` sizes flow into the hash). | N/A. |
| §21 | Observation packer | ❌ | P3 | — | — | Not started. | N/A. |
| §22 | `TelemetrySink` trait + `NullSink` / `VecSink` / `FileSink` + built-in metric name consts | ✅ | P2 T4 (`9be3ebff`, `da1018e9`) | `src/telemetry/` | `tests/telemetry_sink_trait.rs`, `tests/telemetry_vec_sink.rs`, `tests/telemetry_file_sink.rs` | `file_sink_writes_json_lines` checks `lines.len() == 3` and substring `"metric":"foo"`. A JSON serializer bug that swaps two keys or emits malformed UTF-8 still containing the substring would pass. No schema validation. | Run a 1000-tick scenario with FileSink; open the JSONL in `jq` — every row should be valid JSON with `tick`, `metric`, `value` fields. |
| §23 | Debug & trace runtime | ❌ | P4 | — | — | Not started. | — |
| §24 | `ComputeBackend` trait / `SerialBackend` / `GpuBackend` | ❌ | P5, P6 | — | — | Not started. Currently everything is implicitly "Serial" — no trait abstraction. | — |

## Top weak-test risks (prioritized)

1. **`PoolNonOverlapInvariant` is a stub.** `builtins.rs:60-67` has `fn check(...) -> None`. Registering it in `InvariantRegistry` gives false confidence that pool-overlap bugs are guarded. Ditto `MaskValidityInvariant::check` at the trait level (the real check is in the separate `check_with_scratch` path called by the step pipeline). The acceptance test registers `PoolNonOverlapInvariant` but receives no information from it.

2. **Boundary conditions are untested for distances, damage, and restore deltas.** Attack uses 1m (well below 2m cap) and 10m (well above); announce uses 15m (well below 30m overhear) and 100m (well above); topk uses damage 10/20/30/40/50/60 (gaps of 10). A single off-by-one on any boundary (`<=` vs `<`, 2.0 vs 1.99) passes every current test. Add one test per boundary at exactly-boundary + ε.

3. **Movement action tests only check sign, not magnitude.** `action_flee`, `step_move` assert `pos_after.x > pos_before.x` — don't pin `MOVE_SPEED_MPS=1.0`. A bug changing move speed to 0.01 or 100 would pass. Compare to `action_needs` which correctly asserts `(hunger - 0.45).abs() < 1e-6`.

4. **`Announce` uses 3D distance (`Vec3::distance`), spec is silent.** All announce tests place observers on the same z-plane as the speaker, so 3D vs planar is indistinguishable. If the intended semantics are planar (because announcements are "heard in town") then the impl is wrong; if 3D is intended, the spec should say so.

## Open verification questions

Concrete ambiguities that code review can't resolve — audit one at a time.

1. **Does Announce audience use 3D or planar distance?** `step.rs:473` calls `op.distance(center)` which is 3D. Spec §10 doesn't specify. Intended?

2. **Does Attack emit an event when the attacker is dead?** `step.rs:268-290` only guards `if !state.agent_alive(tgt)` (target), not attacker. Can a just-killed attacker's queued Attack action still fire (within the same tick's shuffle order)?

3. **Does Flee with a dead threat emit an event?** `step.rs:248` guards `if !state.agent_alive(threat)` — but does it skip silently (no event) or fall through with `normalize_or_zero`? Current: skips silently. Confirmed reading.

4. **Does the cascade `run_fixed_point` re-dispatch all handlers per iteration, or only handlers triggered by newly-emitted events?** Affects whether a self-amplifying handler quickly hits MAX_CASCADE_ITERATIONS or explodes. Current impl behavior observable via `tests/cascade_bounded.rs` but not asserted.

5. **Does `EventRing::cause_of` return correct results after the parent has been evicted (ring overflow)?** Likely returns `None` silently — should it return a sentinel/error instead? No test covers this.

6. **Does `UtilityBackend` ever emit an action whose mask-bit is false?** `mask_validity_never_flags_a_clean_utility_run` runs 20 ticks and asserts no violations — but it uses peaceful positions (agents on a line), so never exercises the "in-range attack prevented by fleeing" path where mask and score table disagree.

7. **Does `AgentId::new(0)` return `None`?** (NonZeroU32 niche.) Not directly tested; should be — and should be in the invariant set.

8. **Does `hot_pos()` return a slice of length `agent_cap()` (including dead slots)?** `hot_slices_are_independent_vecs` implies yes (asserts `.len() == 5` when capacity is 5, only 3 spawned). Confirmed but worth pinning in spec.

9. **Does `UtilityBackend::evaluate` allocate?** `determinism_no_alloc` is behind `#[cfg(feature = "dhat-heap")]` and runs 16-block budget. Release-mode default runs with `total_blocks <= 16` — may hide a slow leak. Budget should ideally be 0.

10. **Does `shuffle_actions_in_place` actually reorder?** Determinism tests only check hash equality across runs; they can't distinguish a no-op shuffle from a reordering shuffle that happens to match. Need a test that asserts the permutation of a 10-action vec is non-identity for at least one tick.

## What to look at first (verification order)

The items in priority order for the user to externalize (oracle-verify):

1. **Announce 3D-vs-planar** — 1 pair of test cases; resolves one spec ambiguity permanently.
2. **`PoolNonOverlapInvariant` stub** — trivial to reach the "is this wired or not?" answer.
3. **Boundary tests** — 4 new test cases (attack at exactly 2m; announce at exactly 30m overhear; topk eviction at ties) catches a large class of off-by-one bugs.
4. **Move / flee magnitude** — one-line assertion change tightens 3 tests.
5. **Viz harness (Plan 3.0)** — once it exists, the "Visual check" column becomes live verification.

## References

- `spec.md` — runtime contract (§§1–26)
- `README.md` — tree intro
- `docs/superpowers/plans/` — per-plan implementation intent
- `crates/engine/src/` — Rust implementation (Serial only)
- `crates/engine/tests/` — 150-test suite
