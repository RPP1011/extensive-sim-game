# Engine Status (as of 2026-04-24)

> The single source of truth for what's built, what to verify, and what to worry about.
> Design lives in `spec.md`. Implementation intent lives in `docs/superpowers/plans/`.
> This doc is the user's jump-off point: "Is section X trustworthy? What would prove it's not?"

The Serial engine spec (§§1–24) is feature-complete: state, events, mask, policy,
cascade, six-phase tick pipeline, materialized/lazy/topk views, aggregates, save/load
(Plan 3, commit `73c12108`), observation packer (Plan 3), probe harness (Plan 3),
abilities + engagement + 8 EffectOps + recursion (Combat Foundation, commit range
`babb8ec0..a379f3df`). **442 green tests** (release + debug). Schema-hash baseline at
`090f374dcc…`. The 2026-04-19 verification + stub audits are fully resolved (their
detail lives in git history; remaining open items are now line-items here).

GPU backend work is in flight as a parallel track via `engine_gpu` — see the GPU spec
files in `docs/superpowers/specs/` and the active plans in `docs/superpowers/plans/`.

## Legend

- **Status**: ❌ not started / ⚠️ partial / ✅ implemented (tests green) / 🔍 eyeball-verified / 🎯 externally-verified
- **Visual check**: what someone running `cargo run -p viz` should see that would catch bugs.

## Plans index (active only — executed plans live in git history)

| Plan | Doc | Status |
|---|---|---|
| GPU megakernel | `docs/superpowers/plans/gpu_megakernel_plan.md` | ⚠️ in flight (Phase 8 perf-sweep done; perf optimization continuing) |
| GPU cold-state replay umbrella (Subsystem 2) | `docs/superpowers/plans/2026-04-22-gpu-cold-state-replay.md` | ⚠️ Phase 1 done; Phases 2–4 are explicit future work |
| Plan 4 — debug & trace runtime | _(to be written)_ | ❌ not yet written |
| Ability DSL implementation | _(to be planned from `docs/spec/ability.md`)_ | ❌ not yet planned |
| Economic depth implementation | _(to be planned from `docs/spec/economy.md`)_ | ❌ not yet planned |

Deferred subsystems (factions, items, buildings, settlements, regions, personality
utility, interior nav) are indexed in `docs/superpowers/roadmap.md`. Subsystems with
DSL stub primitives partially landed (memberships, memory, relationships, groups,
quests, theory-of-mind) and terrain (MVP `TerrainQuery` trait seam in `crates/engine/src/terrain.rs`)
are partially in flight; full behaviour attachment is pending.

## Subsystem table

Cross-reference: "Tests" column paths are relative to `crates/engine/`. Commits use the
short SHAs from `git log --oneline 64675559..HEAD -- crates/engine/`.

| Spec § | Subsystem | Status | Plan | Code | Tests | Weak-test risk | Visual check |
|---|---|---|---|---|---|---|---|
| §3, §4 | `Pool<T>` + `PoolId<T>` (generic, `NonZeroU32`, freelist) | ✅ 🎯 | P1 T1 (`b4e31a30`), P2.75 proptest (`efe1404d`) + contracts (`74dbd577`) | `src/pool.rs` | `tests/pool_generic.rs`, `tests/state_agent.rs::kill_frees_slot`, `tests/proptest_pool.rs` | Adversarial proptest generates random alloc/kill sequences and asserts `count_alive + freelist_len == next_raw - 1` + no freelist duplicates + `alive ∩ freelist = ∅`. Struct-level `#[invariant]`s enforce the same at every mutation in debug builds. | 200-tick churn (50 spawns / 50 kills per tick) — total alive count never exceeds cap, IDs recycle to low range (verifiable via agent histogram). |
| §3 | SoA agent state — **full `docs/spec/state.md` catalogue** (hot: pos, hp, max_hp, alive, movement_mode, level, move_speed×2, shield_hp/armor/magic_resist/attack_damage/attack_range, mana×2, 3 physiological + 5 psychological needs, 5 personality dims; cold: creature_type, channels, spawn_tick, grid_id, local_pos, move_target, status_effects, memberships, inventory, memory, relationships, class_definitions, creditor_ledger, mentor_lineage) | ✅ 🎯 | P1 T2 (`bbe93150`), state-port plan 2026-04-19 Tasks A–L | `src/state/mod.rs`, `src/state/agent.rs`, `src/state/agent_types.rs` | `tests/state_agent.rs`, `tests/state_needs.rs`, `tests/state_spatial_extras.rs`, `tests/state_combat_extras.rs`, `tests/state_status_effects.rs`, `tests/state_psych_needs.rs`, `tests/state_personality.rs`, `tests/state_capabilities.rs`, `tests/state_memberships.rs`, `tests/state_inventory.rs`, `tests/state_memory.rs`, `tests/state_relationships.rs`, `tests/state_misc_cold.rs` | Each group test asserts: default on spawn (exact cited constant), set/get round-trip, bulk slice length == agent_cap, collection defaults empty. Storage only; subsequent plans wire behaviour. Stub types (`StatusEffect`, `Membership`, `Inventory`, `MemoryEvent`, `Relationship`, `ClassSlot`, `Creditor`, `MentorLink`) in `agent_types.rs` are minimal Pod shells; compiler attaches typed payloads later. Engine has **8 needs total: 3 physiological (hunger/thirst/rest_timer) + 5 psychological (safety/shelter/social/purpose/esteem)** — state.md §Needs Engine note. | N/A (internal layout). |
| §4 | `AgentId`, `GroupId`, `QuestId`, `ItemId`, `AuctionId`, `InviteId`, `SettlementId`, `AbilityId`, `EventId` (all `NonZeroU32` except `EventId` which is `{ tick, seq }`) | ✅ | P1 T3 (`31e45d16`) | `src/ids.rs`, `src/ability/id.rs` | `tests/event_id_threading.rs` | `cause_of` sidecar test only checks cause field doesn't affect hash on a 2-event ring. Doesn't verify cause survives ring-buffer overflow eviction of the parent. `AuctionId` / `InviteId` / `SettlementId` / `ItemId` are reserved for later plans (auction / invite / settlement / item subsystems in roadmap); zero call sites in engine today. | N/A (structural). |
| §5 | `EventRing` (`VecDeque<EventEntry>`, ring drop on overflow, per-tick seq) | ✅ 🎯 | MVP, P1 T3 extends it, P2.75 proptest (`5211d623`) + fuzz target (`47d45856`) | `src/event/ring.rs`, `src/event/mod.rs` | `tests/event_ring.rs`, `tests/event_id_threading.rs`, `tests/determinism.rs`, `tests/proptest_event_hash.rs`, `fuzz/fuzz_targets/event_ring.rs` | Hash-stability proptest generates random event sequences and asserts `sha256(seq) == sha256(seq)` over 100 iterations. cargo-fuzz target runs nightly, hunting panics + hash non-determinism on arbitrary byte inputs. Open gap: `cause_of` after ring-overflow eviction still not asserted. | Event log should show monotonically increasing tick numbers; within a tick, seq should start at 0 and increment. |
| §6 | Spatial index (2D-column BTreeMap + z-sort + MovementMode sidecar) | ✅ 🎯 | MVP, P2.75 proptest (`fdd9fe8d`) | `src/spatial.rs` | `tests/spatial_index.rs`, `tests/proptest_spatial.rs` | Adversarial proptest generates random (spawn, kill, move, query) sequences with positions in `[-20, 20]³` and radii in `[0.01, 30]` — guarantees some boundary cases land at the cell edge (16m). Asserts `within_radius` matches brute-force filter exactly. Catches off-by-one comparators + sidecar-rebuild drift. Covers Walk ↔ Fly mode flips. | Fly agents 15m above walker should be hit by 10m 3D-radius query but miss a 5m one. |
| §7 | RNG (`WorldRng` PCG-XSH-RR, `per_agent_u32`, keyed sub-streams) | ✅ | MVP | `src/rng.rs` | `tests/determinism.rs` (indirect) | No direct golden test of `per_agent_u32(42, AgentId(1), 100, b"action")` against a pinned expected u32. Any re-keying of the PCG constants that still produces deterministic output (but different values) would pass the existing suite because no test hashes a known-good RNG byte stream. | N/A. |
| §9 | `MicroKind` — 18 variants (`Hold`..`Remember`) | ✅ | P1 T4 (`01392efc`) | `src/mask.rs` | `tests/micro_kind_full.rs` | Only spot-checks 7 of 18 ordinals (0, 1, 2, 3, 15, 16, 17). A swap between Cast(4)↔UseItem(5) or Harvest(6)↔Eat(7) would pass. | N/A. |
| §9 | MicroKind execution: Hold / MoveToward / Flee / Attack / Eat / Drink / Rest | ✅ 🎯 | P1 T6–T9 (`84efa271`, `38d889c0`, `c76a1d24`), audit fixes `926c2207`/`86638ddd`/`13581ef4` | `src/step.rs` | `tests/action_flee.rs`, `tests/action_attack_kill.rs`, `tests/action_needs.rs`, `tests/step_move.rs` | All boundary + magnitude pins now landed per audit: ATTACK_RANGE=2m at 1.99m/2.01m (kernel + mask), `damage == 10.0` on AgentAttacked, MOVE_SPEED_MPS=1.0 exact in flee + move, Drink/Rest event `delta` matches constants + saturation variants. | (1) Attacker 1m from target should drop target HP by exactly 10 per tick. (2) Attacker 3m away should do nothing. (3) Flee from co-located threat should stay put, not tremble. (4) Eat with hunger=0.9 should saturate to 1.0 within one tick. |
| §9 | 11 event-only micros (Cast, UseItem, Harvest, PlaceTile/Voxel, HarvestVoxel, Converse, ShareStory, Communicate, Ask, Remember) | ✅ | P1 T10 (`c1be86f3`) | `src/step.rs` | `tests/action_emit_only.rs` | Each test only asserts `.any(matches!(...))`. No test asserts *count* = 1, so a duplicate-emit bug (event fired twice) would pass. | N/A (event-only — no visual state change). |
| §10 | `MacroKind` — 5 variants (`PostQuest`, `AcceptQuest`, `Bid`, `Announce`, `NoOp`) + parameter enums | ✅ | P1 T5, T12 (`42b063b1`, `e83c9d0c`) | `src/policy/macro_kind.rs`, `src/policy/query.rs` | `tests/macro_kind.rs`, `tests/macro_emit_only.rs` | `noop_macro_emits_nothing` filters only on Quest/Bid events — a bug emitting `AnnounceEmitted` for NoOp would pass. | N/A. |
| §10 | Announce cascade (audience enumeration, primary + overhear) | ✅ 🎯 | P1 T14, T15 (`78fa21d0`, `e49208e1`), audit fixes `20f5e414`/`84ce7271` | `src/step.rs::429-507` | `tests/announce_audience.rs`, `tests/announce_overhear.rs` | Distance constants now pinned: Area radius boundary at `[0, 9.999, 10.0]` / `[10.001, 20]`, MAX_ANNOUNCE_RADIUS=80 at `[50, 79.9, 80.1, 200]`, OVERHEAR_RANGE=30 at 29.9/30.1. Primary+overhear dedup verified by identity (disjoint set union covers all 64). **Resolved 2026-04-26:** 3D Euclidean distance confirmed (matches impl); spec/engine.md §4.5 now states this explicitly. Q#1 closed. | Speaker at (0,0,10); listener at (5,0,50) — 3D distance 40m. Announce Area radius 30 should NOT reach the listener. If it does, impl uses planar. |
| §11 | `CascadeRegistry` + `CascadeHandler` trait + `Lane` ordering (Validation → Effect → Reaction → Audit) | ✅ | P1 T11 (`155a51df`) | `src/cascade/` | `tests/cascade_register_dispatch.rs`, `tests/cascade_lanes.rs` | `within_a_lane_registration_order_preserved` only checks count, not observable order. A reorder bug inside a lane wouldn't be caught. | N/A. |
| §11 | Cascade `run_fixed_point` with `MAX_CASCADE_ITERATIONS=8` | ✅ 🎯 | P1 T13 (`0bc30cca`), audit fix `e8d407cd`, P2.75 proptest (`54364d33`) | `src/cascade/dispatch.rs` | `tests/cascade_bounded.rs`, `tests/proptest_cascade_bound.rs` | Release-mode truncation test asserts exact `n == 8`. Proptest generates random handler registries (mix of self-emitters and terminators) + random initial events, asserts handler invocations ≤ `MAX_CASCADE_ITERATIONS × n_initial_events`. Catches infinite-loop regressions. | N/A. |
| §12 | `MaskBuffer` (18-bit-per-agent head) + `mark_hold_allowed` / `mark_move_allowed_*` | ✅ 🎯 | MVP, P2.75 adversarial proptest (`567faaaa`) | `src/mask.rs` | `tests/mask_builder.rs`, `tests/mask_validity.rs`, `tests/proptest_mask_validity.rs` | Adversarial proptest generates random forged-action batches and asserts `MaskValidityInvariant::check_with_scratch` flags every action whose mask bit is false — slot-indexing math verified under agent-id gaps (alloc/kill/alloc patterns produce non-sequential slot↔action correspondence). | N/A. |
| §13 | `PolicyBackend` trait + `UtilityBackend` scalar impl | ✅ | MVP | `src/policy/utility.rs`, `src/policy/mod.rs` | `tests/policy_utility.rs`, `tests/mask_validity.rs` | UtilityBackend extension (P1 T17) only scores 4 of 18 micros; the other 14 rely on mask-bit=false to stay unemitted. No invariant-forced test proves those 14 can't leak with a re-weighting. | Run UtilityBackend 1000 ticks, collect action-kind histogram — should be dominated by Hold / MoveToward / Attack (and Eat/Drink/Rest when needs drop). No stray Cast / Harvest / Converse. |
| §14 | 6-phase tick pipeline (`step_full`): mask → policy → shuffle → apply+cascade → views → invariants+telemetry | ✅ 🎯 | P2 T5 (`0771d16c`), audit fix `5fa05e4b`, P2.75 proptest + contracts (`72495240`, `1210799b`) | `src/step.rs::step_full` | `tests/pipeline_six_phases.rs`, `tests/acceptance_plan2_deterministic.rs`, `tests/proptest_baseline.rs` | Proptest (`step_never_panics_under_random_sizing` in `proptest_baseline.rs`) confirms `step` never panics under random agent-cap + seed configurations. Contracts: `#[requires]` on `step_full` asserts `scratch.mask.n_agents >= state.agent_cap()`; `#[ensures]` on post-state verifies `state.tick == old + 1`. Fault-injection test with undersized scratch panics as expected. | Agent HP should drop across ticks when adjacent enemies attack (verified end-to-end). Viz: mixed-action scenario overlay showing per-tick action histogram + cumulative damage. |
| §14 | Phase 3 (Fisher-Yates shuffle keyed on `per_agent_u32(seed, AgentId(1), tick<<16, b"shuffle")`) | ✅ | MVP | `src/step.rs::shuffle_actions_in_place` | `tests/determinism.rs` | Determinism tested via hash equality, but there's no test asserting the shuffle *actually* produces different orderings across ticks. A no-op shuffle would still be deterministic. | N/A. |
| §15 | `MaterializedView` trait + `DamageTaken` example | ✅ | MVP | `src/view/materialized.rs`, `src/view/mod.rs` | `tests/view_materialized.rs` | Two-event test; no test of integer-vs-float reduction determinism (commutativity under reorder). | N/A. |
| §15 | `LazyView` trait + `NearestEnemyLazy` + staleness flag (trait only — NOT wired into step_full) | ⚠️ | P2 T1 (`a1aad00b`), audit doc `f75a16fd` | `src/view/lazy.rs` | `tests/view_lazy.rs` | Trait-shape tests only. Integration gap surfaced by audit: `step_full` folds `MaterializedView` but not `LazyView`. Header comment + `#[ignore]`d canary `lazy_view_wired_into_step_full` will light up when integration lands. | N/A. |
| §15 | `TopKView` trait + `MostHostileTopK` (cumulative-damage heap) | ✅ | P2 T2 (`aee396a5`) | `src/view/topk.rs` | `tests/view_topk.rs` | `topk_bounded_keeps_highest_scoring_attackers` asserts top[0]=60, top[3]=30 (K=4, 6 attackers with damage 10/20/30/40/50/60). Boundary between top-4 (30) and out (20) is 10 apart — a bug using `<` vs `<=` on the eviction threshold wouldn't fire here. | N/A. |
| §16 | `AggregatePool<T>` + `Quest` / `Group` Pod-compatible shapes | ✅ | P1 T16 (`c438f249`) | `src/aggregate/` | `tests/aggregate_pool.rs` | `kill_then_alloc_reuses_slot_and_clears_contents` (in `aggregate_pool.rs`) — same single-reuse issue as Pool. Also no test registers a cascade handler that writes to `AggregatePool` and reads it back on the next iteration. | N/A (no visual for aggregates). |
| §17 | Trajectory emission (safetensors; N-tick windowing) | ✅ | MVP | `src/trajectory.rs` | `tests/trajectory_roundtrip.rs` | Python roundtrip test compares only `n_agents` and `n_ticks` — doesn't compare per-tick position/hp data. A silent trajectory corruption that preserves shape would pass. | N/A. |
| §18 | Save / load | ✅ | P3 (`73c12108`) | `src/snapshot/{format,migrate}.rs` | `tests/snapshot_{header,roundtrip,schema_mismatch,migration}.rs`, `tests/acceptance_plan3.rs` | Coverage gaps documented inline in `format.rs`: cold_channels, EventRing entries (metadata only), views/registry/terrain/config (rebuilt or caller-supplied). Acceptance test asserts state equality (not event-hash equality) post save+reload. | N/A. |
| §19 | `Invariant` trait + `Violation` + `FailureMode` | ✅ | P2 T3 (`f8c23715`) | `src/invariant/trait_.rs` | `tests/invariant_trait.rs`, `tests/invariant_dispatch_modes.rs` | — | N/A. |
| §19 | `InvariantRegistry::check_all` + dispatch by failure mode | ✅ | P2 T3 (`21537e11`) | `src/invariant/registry.rs` | `tests/invariant_dispatch_modes.rs` | — | N/A. |
| §19 | Built-in invariants: `MaskValidityInvariant`, `PoolNonOverlapInvariant` | ✅ 🎯 | P2 T3 (`6c0ac879`), audit fix `bc6fac31` | `src/invariant/builtins.rs` | `tests/invariant_mask_validity.rs`, `tests/invariant_pool_non_overlap.rs` | **`PoolNonOverlapInvariant::check` is now real** — `Pool<T>::is_non_overlapping` walks `alive` + `freelist` to flag both overlap AND freelist duplicates. Two new fault-injection tests prove the check fires (would fail if body were reverted to `None`). `MaskValidityInvariant::check` (trait impl) still returns `None`; real check is `check_with_scratch`, invoked by `step_full`. Separate documented design, not a regression. | N/A. |
| §18 | Probe harness | ✅ | P3 (`73c12108`) | `src/probe/mod.rs` | `tests/probe_harness.rs`, `tests/probe_determinism.rs` | Probe is a struct (name + seed + spawn fn + ticks + assert fn). Same-seed determinism asserted via `replayable_sha256`. | N/A. |
| §20 | Schema hash (`sha2` over layout fingerprint + `.schema_hash` baseline file) | ✅ | MVP + P1 T4 + P2 T4 | `src/schema_hash.rs`, `.schema_hash` | `tests/schema_hash.rs` | Baseline-comparison test catches any hash drift but doesn't prove the fingerprint string covers every layout-relevant type (e.g., nothing asserts `ResourceRef` or `ItemId` sizes flow into the hash). | N/A. |
| §21 | Observation packer | ✅ | P3 (`73c12108`) | `src/obs/{packer,sources}.rs` | `tests/obs_packer.rs`, `tests/obs_sources_{vitals,position,neighbors}.rs` | `FeatureSource` trait + 3 built-ins (Vitals dim 4, Position dim 7, Neighbor<K> dim 6K). Per-tick alloc in NeighborSource — zero-alloc variant deferred to a SimScratch slot in a later plan. | N/A. |
| §22 | `TelemetrySink` trait + `NullSink` / `VecSink` / `FileSink` + built-in metric name consts | ✅ | P2 T4 (`9be3ebff`, `da1018e9`) | `src/telemetry/` | `tests/telemetry_sink_trait.rs`, `tests/telemetry_vec_sink.rs`, `tests/telemetry_file_sink.rs` | `file_sink_writes_json_lines` checks `lines.len() == 3` and substring `"metric":"foo"`. A JSON serializer bug that swaps two keys or emits malformed UTF-8 still containing the substring would pass. No schema validation. | Run a 1000-tick scenario with FileSink; open the JSONL in `jq` — every row should be valid JSON with `tick`, `metric`, `value` fields. |
| §23 | Debug & trace runtime | ❌ | P4 | — | — | Not started. | — |
| §24 | `ComputeBackend` trait / `SerialBackend` / `GpuBackend` | ❌ | P5, P6 | — | — | Not started. Currently everything is implicitly "Serial" — no trait abstraction. | — |

## Top weak-test risks (prioritized)

1. **`Announce` uses 3D distance (`Vec3::distance`), spec is silent.** All announce tests still place observers on the same z-plane as the speaker, so 3D vs planar is indistinguishable. Distance *constants* are now all pinned (HIGH #1/#2/#4 fixed), so a value change would fail — but a 3D→planar refactor of the same constant would not. If the intended semantics are planar (because announcements are "heard in town") then the impl is wrong; if 3D is intended, the spec should say so.

2. **`LazyView` is not wired into `step_full`.** Trait exists, `NearestEnemyLazy` implementation exists, unit tests exercise the trait surface — but nothing in the tick pipeline calls `invalidate_on_events` on lazy views. Canary test `lazy_view_wired_into_step_full` is `#[ignore]`d; un-ignore when wired.

3. **`MaskValidityInvariant::check` (trait impl) still returns `None`.** The real check is `check_with_scratch`, invoked by `step_full`. Documented-by-design but a registering caller who expected `check` to fire would be surprised.

4. **Phase-5 view-fold skip not asserted.** `pipeline_six_phases` value assertions now catch constant-zero / out-of-range telemetry, but no test proves that a `MaterializedView::fold` was actually invoked during the tick (as distinct from the pre-step compute). Would need a view that mutates visibly on a known-frequency event.

## Open verification questions

Concrete ambiguities that code review can't resolve — audit one at a time.

1. **Does Announce audience use 3D or planar distance?** Announce dispatches through `spatial.within_radius(state, center, radius)` (`step.rs` Announce arm, post-2026-04-23 refactor) which is 3D. Spec §10 doesn't specify. Intended?

2. **Does Attack emit an event when the attacker is dead?** `step.rs:268-290` only guards `if !state.agent_alive(tgt)` (target), not attacker. Can a just-killed attacker's queued Attack action still fire (within the same tick's shuffle order)?

3. **Does Flee with a dead threat emit an event?** `step.rs:248` guards `if !state.agent_alive(threat)` — but does it skip silently (no event) or fall through with `normalize_or_zero`? Current: skips silently. Confirmed reading.

4. **Does the cascade `run_fixed_point` re-dispatch all handlers per iteration, or only handlers triggered by newly-emitted events?** Affects whether a self-amplifying handler quickly hits MAX_CASCADE_ITERATIONS or explodes. Current impl behavior observable via `tests/cascade_bounded.rs` but not asserted.

5. **Does `EventRing::cause_of` return correct results after the parent has been evicted (ring overflow)?** Likely returns `None` silently — should it return a sentinel/error instead? No test covers this.

6. **Does `UtilityBackend` ever emit an action whose mask-bit is false?** `mask_validity_never_flags_a_clean_utility_run` runs 20 ticks and asserts no violations — but it uses peaceful positions (agents on a line), so never exercises the "in-range attack prevented by fleeing" path where mask and score table disagree.

7. **Does `AgentId::new(0)` return `None`?** (NonZeroU32 niche.) Not directly tested; should be — and should be in the invariant set.

8. **Does `hot_pos()` return a slice of length `agent_cap()` (including dead slots)?** `hot_slices_are_independent_vecs` implies yes (asserts `.len() == 5` when capacity is 5, only 3 spawned). Confirmed but worth pinning in spec.

9. **Does `UtilityBackend::evaluate` allocate?** `determinism_no_alloc` is behind `#[cfg(feature = "dhat-heap")]` and runs 16-block budget. Release-mode default runs with `total_blocks <= 16` — may hide a slow leak. Budget should ideally be 0.

10. **Does `shuffle_actions_in_place` actually reorder?** Determinism tests only check hash equality across runs; they can't distinguish a no-op shuffle from a reordering shuffle that happens to match. Need a test that asserts the permutation of a 10-action vec is non-identity for at least one tick.

11. **Co-occupancy is intentional.** ~~Engine has no collision detection.~~ **Resolved 2026-04-26:** Multiple agents may occupy the same `Vec3` simultaneously by design — agents are point particles. Visualization handles overlap via vertical voxel stacking (Plan 3.1). No engine collision pass; if a future scenario needs hard collision (crowd dynamics, structure occlusion), revisit then. Q#11 closed.

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
- `crates/engine/tests/` — 157-test suite (+1 ignored LazyView integration canary)

## Visual-check checklist

Each item is a live acceptance criterion for the Plan 3.0 viz harness.
Run `cargo run -p viz -- <scenario>` and eyeball the result.

| # | Scenario | Expected visual | Catches regression in |
|---|---|---|---|
| V1 | `crates/viz/scenarios/viz_basic.toml` | Ground + 4 blue voxels in an 8 m square + 1 red voxel 20 m NE; the red voxel walks toward the blue cluster over ~2 s. | Move action (§9), nearest-enemy utility scoring. |
| V2 | `viz_basic.toml`, after ~10 s | One or more blue voxels have disappeared; a black voxel persists where each died. | Attack damage + AgentDied emission. |
| V3 | `viz_attack.toml` | Red voxel closes 3 m in ~3 ticks, then a short red line pulses between attacker and target every tick until the human dies. | ATTACK_RANGE = 2.0 m, attack-line overlay ingest. |
| V4 | `viz_attack.toml` post-death | Single black voxel at the former human position; wolf idle (no more targets). | AgentDied cleanup, mask pruning of dead targets. |
| V5 | `viz_announce.toml` + test backend that emits `MacroAction::Announce` (future plan) | White ring expands from speaker over 3 ticks, covering 80 m. Listeners inside 30 m get memories (check events log). | Announce audience enumeration + overhear scan (§10). |
| V6 | Any scenario, paused, pressing `.` | Tick advances by 1 per press; HUD prints `tick={n+1}`. | Pause/step determinism, no accumulator leak. |
| V7 | Any scenario, pressing `]` 4 times | HUD prints `speed=16.00x`; agents move visibly faster; fps ≥ 30. | Tick accumulator math, burst-cap behavior. |
| V8 | Any scenario, pressing `R` | Agents snap back to spawn positions; HUD reports `tick=0`; overlays clear. | Reload path cleans sim + overlays. |
| V9 | `viz_basic.toml`, wait until wolf reaches the human cluster | When humans and wolf converge, you should see a vertical stack of cubes (different colors) where otherwise they'd visually collapse into one cube. | Viz stacking workaround for engine's missing collision detection (Plan 3.1 Task 3). |

Known gaps:
- V5 requires a test backend that emits `Announce`; `UtilityBackend`
  only emits the 7 implemented micros. Becomes live when a future plan
  wires an announce-enabled policy.
- No in-window HUD — HUD is stdout-only. A later plan can layer egui
  or a text-shader overlay; deliberately out of scope for Plan 3.0.
