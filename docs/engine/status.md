# Engine Status (as of 2026-04-24, partially re-audited 2026-09-03 — see banner below)

> The single source of truth for what's built, what to verify, and what to worry about.
> Design lives in `spec.md`. Implementation intent lives in `docs/superpowers/plans/`.
> This doc is the user's jump-off point: "Is section X trustworthy? What would prove it's not?"

> **⚠️ 2026-09-03 staleness audit — read this before trusting the rule-execution rows below.**
> This file's body was written 2026-04-24. The very next day, commits `33ab26bc`/`07922e03`
> (Plan B1' Task 11) deleted the real `step`/`step_full`/mask-fill implementation out of
> `crates/engine` — `src/step.rs` and `src/probe/mod.rs` are now `unimplemented!()`
> compile-only stubs kept solely so old imports still build, and their doc comments point at
> `engine_rules::step` as the "real" driver. Then the 2026-05-02 Phase 7 wolf-sim wipe deleted
> `engine_rules` itself (along with `xtask`, `crates/viz`, `tactical_sim`, `engine_gpu`) without
> restoring a replacement. **Net effect: there is no live tick driver anywhere in `crates/engine`
> today.** Real per-fixture tick execution is compiler-emitted per fixture straight into
> `crates/sims`' `GeneratedRuntime` modules (or a couple of legacy `crates/*_runtime` crates),
> calling engine primitives (`Pool`, `EventRing`, `CascadeRegistry`, `MaskBuffer`, spatial index)
> directly — bypassing `engine::step`/`ComputeBackend` entirely. See `crates/engine/CLAUDE.md`'s
> "Internal architecture" section (and its "reality check" callout) for the authoritative current
> picture. Below, rows whose Code/Tests columns pointed at the now-stub `step.rs`/`probe/mod.rs`
> or at test files deleted in that purge are marked **STALE (2026-09-03)** with a corrected
> description; rows about storage-only primitives (state, pool, event ring, spatial index,
> cascade dispatch, aggregates, save/load, invariants, schema hash, obs packer, telemetry) were
> cross-checked against current `crates/engine/tests/` and still hold — their cited test files
> exist and the code they describe is still live.

The Serial engine spec (§§1–24) was feature-complete as of 2026-04-24: state, events, mask, policy,
cascade, six-phase tick pipeline, materialized/lazy/topk views, aggregates, save/load
(Plan 3, commit `73c12108`), observation packer (Plan 3), probe harness (Plan 3),
abilities + engagement + 8 EffectOps + recursion (Combat Foundation, commit range
`babb8ec0..a379f3df`). **442 green tests (release + debug)** was the count *at that date, before
the very next day's B1' Task 11 purge deleted a large slice of `crates/engine/tests/`* — it is not
a current number and shouldn't be quoted as one. As a rough current substitute (not a
like-for-like comparison — many purged micro-level tests are gone, many DSL/ability-corpus tests
were added since): `cargo test -p engine --lib` currently shows **171 passed, 1 failed** (see the
§7 RNG row below for the failure), and a 2026-09-03 grep count found **282 `#[test]` functions
across 65 files** under `crates/engine/tests/`. Neither figure has been reconciled against a full
green `cargo test -p engine` run — treat both as approximate. Schema-hash baseline: the value
below is what `crates/engine/.schema_hash` actually contains as of 2026-09-03
(`431332367209cbb2…`) — the previously-cited `090f374dcc…` no longer matches the checked-in file
and has been corrected. The 2026-04-19 verification + stub audits are fully resolved (their
detail lives in git history; remaining open items are now line-items here).

GPU backend work is **not** in flight via `engine_gpu` — that crate was deleted in the 2026-05-02
Phase 7 wolf-sim wipe and stayed gone (see the comment above `[workspace]` in the root
`Cargo.toml`). `engine_gpu_rules` (the crate that might sound like its successor) is an empty
placeholder kept alive only so `engine/src/schema_hash.rs` has a `.schema_hash` file to
`include_str!` — it holds no GPU work. Real GPU kernel work today happens per-fixture: the DSL
compiler emits WGSL directly into each fixture's own `OUT_DIR` (`crates/sims/build.rs`,
`crates/tom_probe_runtime/build.rs`, etc.), built on `engine::gpu`'s sim-agnostic platform layer
(`GpuContext`, `Kernel` trait, BGL helpers) — there is no separate "GPU backend track" crate to
point to. See the GPU spec files in `docs/superpowers/specs/` and the active plans in
`docs/superpowers/plans/` for design intent, but don't expect `engine_gpu` to exist.

## Legend

- **Status**: ❌ not started / ⚠️ partial / ✅ implemented (tests green) / 🔍 eyeball-verified / 🎯 externally-verified
- **Visual check**: historical column from the retired Plan 3.0 viz harness (`crates/viz`, deleted
  2026-05-02) — there is no `cargo run -p viz` today. See "Visual-check checklist" near the bottom
  of this file for what (if anything) currently substitutes.

## Plans index (active only — executed plans live in git history)

| Plan | Doc | Status |
|---|---|---|
| GPU megakernel | `docs/superpowers/plans/gpu_megakernel_plan.md` | ⚠️ in flight (Phase 8 perf-sweep done; perf optimization continuing) |
| GPU cold-state replay umbrella (Subsystem 2) | `docs/superpowers/plans/2026-04-22-gpu-cold-state-replay.md` | ⚠️ Phase 1 done; Phases 2–4 are explicit future work |
| Plan 4 — debug & trace runtime | _(to be written)_ | ❌ not yet written |
| Ability DSL implementation | _(execution log in git history; LoL canary saturated 2026-05-06)_ | ✅ lowering 100% (172/172 LoL files) — apply handlers + GPU dispatch (`#125` family) in flight |
| Voxel terrain integration | `docs/superpowers/plans/2026-05-09-voxel-engine-integration.md` | ✅ COMPLETE 2026-05-09 — `engine_voxel` adapter (CPU + GPU mirror), DSL `terrain.X(...)` lowers to WGSL helpers, `wave_defense_runtime` opts in via settler `BuildPalisade` casts; 6 semantic + 1 determinism + 1 fixture-level pin all green |
| Economic depth implementation | _(to be planned from `docs/spec/economy.md`)_ | ❌ not yet planned |

Deferred subsystems (factions, items, buildings, settlements, regions, personality
utility, interior nav) are indexed in `docs/superpowers/roadmap.md`. Subsystems with
DSL stub primitives partially landed (memberships, memory, relationships, groups,
quests, theory-of-mind) are partially in flight; full behaviour attachment is
pending. Terrain is no longer in this category — Phase E of the voxel-engine
integration plan (2026-05-09) shipped real `engine_voxel::VoxelTerrain` + GPU
mirror; the `FlatPlane` default trait impl stays as the no-op fallback for
fixtures that don't opt in.

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
| §7 | RNG (`WorldRng` PCG-XSH-RR, `per_agent_u32`, keyed sub-streams) | ⚠️ **STALE mark (2026-09-03)** | MVP | `src/rng.rs` | `tests/determinism.rs` no longer exists (deleted in the 2026-04-25 purge — was never actually the source of RNG golden coverage anyway). Real current coverage is `src/rng.rs`'s own `#[cfg(test)] mod tests` (in-file unit tests, not `crates/engine/tests/`), which **does** include pinned golden values (`world_rng_golden_value`, `per_agent_golden_value`, `per_agent_pcg_golden_value`, `per_agent_pcg_with_extra_golden_value`) — the old weak-test-risk note below (no golden test exists) is itself now stale. | **A 2026-09-03 `cargo test -p engine --lib` run shows `rng::tests::per_agent_golden_value` FAILING** (`left: 7573652868700658681, right: 15934783303190885974` at `rng.rs:228`) while the sibling `world_rng_golden_value` / `per_agent_pcg_*` golden tests pass. Not investigated further as part of this doc audit — flagging as a live discrepancy between this row's ✅ and reality; someone should check whether `per_agent_u64` changed since the value was pinned or the test itself is wrong before trusting this subsystem. | N/A. |
| §9 | `MicroKind` — 18 variants (`Hold`..`Remember`) | ✅ | P1 T4 (`01392efc`) | `src/mask.rs` | `tests/micro_kind_full.rs` | Only spot-checks 7 of 18 ordinals (0, 1, 2, 3, 15, 16, 17). A swap between Cast(4)↔UseItem(5) or Harvest(6)↔Eat(7) would pass. | N/A. |
| §9 | MicroKind execution: Hold / MoveToward / Flee / Attack / Eat / Drink / Rest | ⚠️ **STALE mark (2026-09-03)** | P1 T6–T9 (`84efa271`, `38d889c0`, `c76a1d24`), audit fixes `926c2207`/`86638ddd`/`13581ef4` — all pre-2026-04-25 | `src/step.rs` is now an `unimplemented!()` compile-only stub (Plan B1' Task 11, 2026-04-25) — calling any of its functions panics. | `tests/action_flee.rs`, `tests/action_attack_kill.rs`, `tests/action_needs.rs`, `tests/step_move.rs` **no longer exist** (deleted in the same purge). No direct successor test suite in `crates/engine/tests/` for this exact behavior. | The micro-kind behaviors this row describes are real in shipped fixtures, but only as DSL-compiler-emitted code per fixture (`crates/sims`' `GeneratedRuntime`s, verified via each fixture's own `*_pin.rs` test) — not through this shared engine-crate kernel or these tests, both of which are gone. If you need current evidence Attack/Move/Flee/Eat work, look at a specific fixture's pin test in `crates/sims/tests/`, not here. | (historical — no current viz path, see banner) |
| §9 | 11 event-only micros (Cast, UseItem, Harvest, PlaceTile/Voxel, HarvestVoxel, Converse, ShareStory, Communicate, Ask, Remember) | ⚠️ **STALE mark (2026-09-03)** | P1 T10 (`c1be86f3`) — pre-2026-04-25 | Same `src/step.rs` stub as the row above. | `tests/action_emit_only.rs` no longer exists. | Same caveat as the row above — real behavior (if any) is per-fixture compiler-emitted code today, not this file/these tests. | N/A (event-only — no visual state change). |
| §10 | `MacroKind` — 5 variants (`PostQuest`, `AcceptQuest`, `Bid`, `Announce`, `NoOp`) + parameter enums | ✅ (Tests column **STALE**, 2026-09-03) | P1 T5, T12 (`42b063b1`, `e83c9d0c`) | `src/policy/macro_kind.rs`, `src/policy/query.rs` | `tests/macro_kind.rs` exists; `tests/macro_emit_only.rs` no longer exists (the `noop_macro_emits_nothing` weak-test-risk note below described that missing file, so it's now moot rather than fixed). | `noop_macro_emits_nothing` filters only on Quest/Bid events — a bug emitting `AnnounceEmitted` for NoOp would pass, *if that test still existed*; it doesn't. | N/A. |
| §10 | Announce cascade (audience enumeration, primary + overhear) | ⚠️ **STALE mark (2026-09-03)** | P1 T14, T15 (`78fa21d0`, `e49208e1`), audit fixes `20f5e414`/`84ce7271` — all pre-2026-04-25 | `src/step.rs::429-507` — that line range is inside the now-`unimplemented!()` stub; the real body was deleted 2026-04-25. | `tests/announce_audience.rs`, `tests/announce_overhear.rs` no longer exist. | The constant/dedup pinning and the 3D-vs-planar resolution described here were real *as of the pre-purge implementation*; whether a current per-fixture Announce implementation (if any) preserves them is unverified — check a specific fixture's DSL/pin tests, not this row. | (historical — no current viz path, see banner) |
| §11 | `CascadeRegistry` + `CascadeHandler` trait + `Lane` ordering (Validation → Effect → Reaction → Audit) | ✅ | P1 T11 (`155a51df`) | `src/cascade/` | `tests/cascade_register_dispatch.rs`, `tests/cascade_lanes.rs` | `within_a_lane_registration_order_preserved` only checks count, not observable order. A reorder bug inside a lane wouldn't be caught. | N/A. |
| §11 | Cascade `run_fixed_point` with `MAX_CASCADE_ITERATIONS=8` | ✅ 🎯 | P1 T13 (`0bc30cca`), audit fix `e8d407cd`, P2.75 proptest (`54364d33`) | `src/cascade/dispatch.rs` | `tests/cascade_bounded.rs`, `tests/proptest_cascade_bound.rs` | Release-mode truncation test asserts exact `n == 8`. Proptest generates random handler registries (mix of self-emitters and terminators) + random initial events, asserts handler invocations ≤ `MAX_CASCADE_ITERATIONS × n_initial_events`. Catches infinite-loop regressions. | N/A. |
| §12 | `MaskBuffer` (18-bit-per-agent head) storage + `set()` | ⚠️ **STALE mark (2026-09-03)** | MVP, P2.75 adversarial proptest (`567faaaa`) — pre-2026-04-25 | `src/mask.rs` still exists and its `MaskBuffer` struct + `set()` are real, but `mark_hold_allowed`/`mark_move_allowed_*` (the rule-aware mask-fill functions this row names) **no longer exist in this file** — the file's own header comment now says "Rule-aware mask-build logic lives in `engine_rules::step`", and `engine_rules` was deleted in the 2026-05-02 Phase 7 wipe with no replacement. `mask.rs` today is storage-primitives-only. | `tests/mask_builder.rs`, `tests/mask_validity.rs`, `tests/proptest_mask_validity.rs` no longer exist; `grep -rl MaskValidityInvariant crates/engine/tests/` finds nothing. | The adversarial-proptest coverage this row describes is gone with no replacement found. | N/A. |
| §13 | `PolicyBackend` trait + `UtilityBackend` scalar impl | ⚠️ **STALE Tests column (2026-09-03)** | MVP | `src/policy/utility.rs`, `src/policy/mod.rs` — `UtilityBackend` still exists as real code (reshaped by Task 138, not a stub) | `tests/policy_utility.rs`, `tests/mask_validity.rs` no longer exist; `grep -rl UtilityBackend crates/engine/tests/` finds nothing — **zero current test coverage** of this struct in `crates/engine/tests/`. | The 4-of-18-micros scoring gap described here was true pre-purge; whether it's still accurate for the current `utility.rs` is unverified — no test exercises it either way today. | Run UtilityBackend 1000 ticks, collect action-kind histogram — should be dominated by Hold / MoveToward / Attack (and Eat/Drink/Rest when needs drop). No stray Cast / Harvest / Converse. (No current harness runs this — see banner.) |
| §14 | 6-phase tick pipeline (`step_full`): mask → policy → shuffle → apply+cascade → views → invariants+telemetry | ⚠️ **STALE mark (2026-09-03)** — spec/contract only | P2 T5 (`0771d16c`), audit fix `5fa05e4b`, P2.75 proptest + contracts (`72495240`, `1210799b`) — all pre-2026-04-25 | `src/step.rs::step_full` **panics with `unimplemented!()` if called** — deleted 2026-04-25, never restored. `crates/engine/CLAUDE.md` states this explicitly: "§14 six-phase tick pipeline (contract only — no current implementation in this crate)." | `tests/pipeline_six_phases.rs`, `tests/acceptance_plan2_deterministic.rs`, `tests/proptest_baseline.rs` no longer exist. | The six phases are a spec contract realized per-fixture by DSL-compiler-emitted schedules (see `crates/sims`), not by a shared `step_full` in this crate. If you need current evidence the pipeline runs end-to-end for a real fixture, check that fixture's own pin test. | (historical — no current viz path, see banner) |
| §14 | Phase 3 (Fisher-Yates shuffle keyed on `per_agent_u32(seed, AgentId(1), tick<<16, b"shuffle")`) | ⚠️ **STALE mark (2026-09-03)** | MVP | `src/step.rs::shuffle_actions_in_place` — `unimplemented!()` stub since 2026-04-25. | `tests/determinism.rs` no longer exists. | Same caveat — shuffle logic (if any) is compiler-emitted per fixture today, unverified against this description. | N/A. |
| §15 | `MaterializedView` trait + `DamageTaken` example | ✅ | MVP | `src/view/materialized.rs`, `src/view/mod.rs` | `tests/view_materialized.rs` | Two-event test; no test of integer-vs-float reduction determinism (commutativity under reorder). | N/A. |
| §15 | `LazyView` trait + `NearestEnemyLazy` + staleness flag (trait only — NOT wired into step_full) | ⚠️ (mark still accurate; Tests column **STALE**, 2026-09-03) | P2 T1 (`a1aad00b`), audit doc `f75a16fd` | `src/view/lazy.rs` still exists with real trait/impl code. | `tests/view_lazy.rs` no longer exists (`grep -rl LazyView crates/engine/tests/` finds nothing outside a `ui/*.stderr` trybuild fixture). | The "not wired into step_full" framing is if anything *more* true now — `step_full` itself is an `unimplemented!()` stub, not just missing the LazyView fold. But the trait-shape test coverage this row claims is gone; unverified today. | N/A. |
| §15 | `TopKView` trait + `MostHostileTopK` (cumulative-damage heap) | ✅ | P2 T2 (`aee396a5`) | `src/view/topk.rs` | `tests/view_topk.rs` | `topk_bounded_keeps_highest_scoring_attackers` asserts top[0]=60, top[3]=30 (K=4, 6 attackers with damage 10/20/30/40/50/60). Boundary between top-4 (30) and out (20) is 10 apart — a bug using `<` vs `<=` on the eviction threshold wouldn't fire here. | N/A. |
| §16 | `AggregatePool<T>` + `Quest` / `Group` Pod-compatible shapes | ✅ | P1 T16 (`c438f249`) | `src/aggregate/` | `tests/aggregate_pool.rs` | `kill_then_alloc_reuses_slot_and_clears_contents` (in `aggregate_pool.rs`) — same single-reuse issue as Pool. Also no test registers a cascade handler that writes to `AggregatePool` and reads it back on the next iteration. | N/A (no visual for aggregates). |
| §17 | Trajectory emission (safetensors; N-tick windowing) | ⚠️ **STALE Tests column (2026-09-03)** | MVP | `src/trajectory.rs` still exists with real `TrajectoryWriter`/`TrajectoryReader` code (not a stub). | `tests/trajectory_roundtrip.rs` no longer exists — **zero current test coverage** in `crates/engine/tests/`. | The shape-only-comparison weakness described here was true of the old (now-deleted) test; there is no test at all today to have that or any other weakness. | N/A. |
| §18 | Save / load | ✅ | P3 (`73c12108`) | `src/snapshot/{format,migrate}.rs` | `tests/snapshot_{header,roundtrip,schema_mismatch,migration}.rs`, `tests/acceptance_plan3.rs` | Coverage gaps documented inline in `format.rs`: cold_channels, EventRing entries (metadata only), views/registry/terrain/config (rebuilt or caller-supplied). Acceptance test asserts state equality (not event-hash equality) post save+reload. | N/A. |
| §19 | `Invariant` trait + `Violation` + `FailureMode` | ✅ | P2 T3 (`f8c23715`) | `src/invariant/trait_.rs` | `tests/invariant_trait.rs`, `tests/invariant_dispatch_modes.rs` | — | N/A. |
| §19 | `InvariantRegistry::check_all` + dispatch by failure mode | ✅ | P2 T3 (`21537e11`) | `src/invariant/registry.rs` | `tests/invariant_dispatch_modes.rs` | — | N/A. |
| §19 | Built-in invariants: `MaskValidityInvariant`, `PoolNonOverlapInvariant` | ⚠️ (downgraded from ✅🎯, 2026-09-03) | P2 T3 (`6c0ac879`), audit fix `bc6fac31` | `src/invariant/builtins.rs` | `tests/invariant_mask_validity.rs` no longer exists; `tests/invariant_pool_non_overlap.rs` still exists. | **`PoolNonOverlapInvariant::check` coverage is still real and green** — `Pool<T>::is_non_overlapping` walks `alive` + `freelist` to flag both overlap AND freelist duplicates, per the surviving test file. `MaskValidityInvariant::check` (trait impl) still returns `None` by design; its real check is `check_with_scratch`, but that was previously "invoked by `step_full`" — and `step_full` is now an `unimplemented!()` stub, so nothing invokes `check_with_scratch` today either. Downgraded because half this row's claim (MaskValidityInvariant actually firing anywhere) no longer holds; PoolNonOverlapInvariant alone would be ✅. | N/A. |
| §18 | Probe harness | ❌ **downgraded 2026-09-03 — was ✅** | P3 (`73c12108`) — pre-2026-04-25 | `src/probe/mod.rs`'s `run_probe` is now literally `unimplemented!()`, per the file's own doc comment: "NOTE: `run_probe` is UNIMPLEMENTED; the tick driver lives in `engine_rules::step::step`" — and `engine_rules` no longer exists. | `tests/probe_harness.rs`, `tests/probe_determinism.rs` no longer exist. | This is an unambiguous downgrade, not a judgment call: the function this row credits with a status of "implemented, tests green" panics unconditionally if called today. `Probe` is still a usable plain struct shape, but there is no driver to run one. | N/A. |
| §20 | Schema hash (`sha2` over layout fingerprint + `.schema_hash` baseline file) | ✅ | MVP + P1 T4 + P2 T4 | `src/schema_hash.rs`, `.schema_hash` | `tests/schema_hash.rs` | Baseline-comparison test catches any hash drift but doesn't prove the fingerprint string covers every layout-relevant type (e.g., nothing asserts `ResourceRef` or `ItemId` sizes flow into the hash). | N/A. |
| §21 | Observation packer | ✅ | P3 (`73c12108`) | `src/obs/{packer,sources}.rs` | `tests/obs_packer.rs`, `tests/obs_sources_{vitals,position,neighbors}.rs` | `FeatureSource` trait + 3 built-ins (Vitals dim 4, Position dim 7, Neighbor<K> dim 6K). Per-tick alloc in NeighborSource — zero-alloc variant deferred to a SimScratch slot in a later plan. | N/A. |
| §22 | `TelemetrySink` trait + `NullSink` / `VecSink` / `FileSink` + built-in metric name consts | ✅ | P2 T4 (`9be3ebff`, `da1018e9`) | `src/telemetry/` | `tests/telemetry_sink_trait.rs`, `tests/telemetry_vec_sink.rs`, `tests/telemetry_file_sink.rs` | `file_sink_writes_json_lines` checks `lines.len() == 3` and substring `"metric":"foo"`. A JSON serializer bug that swaps two keys or emits malformed UTF-8 still containing the substring would pass. No schema validation. | Run a 1000-tick scenario with FileSink; open the JSONL in `jq` — every row should be valid JSON with `tick`, `metric`, `value` fields. |
| §23 | Debug & trace runtime | ❌ | P4 | — | — | Not started. | — |
| §24 | `ComputeBackend` trait / `SerialBackend` / `GpuBackend` | ❌ **description corrected 2026-09-03** | P5, P6 | `src/backend.rs` — the trait shape exists (`fn step`, `reset_mask`, `set_mask_bit`, `commit_mask`, `cascade_dispatch`, `view_fold`, `apply_and_movement`), but `grep -rl "impl.*ComputeBackend" crates/` is empty — **zero concrete implementations anywhere in the workspace** (no `SerialBackend`, no `GpuBackend`). | — | Not started, but not for lack of a trait — the trait compiles and documents an intended shape; nothing implements it. The real per-fixture tick drivers (`crates/sims` generated code, `tom_probe_runtime`, etc.) implement `engine::sim_trait::CompiledSim` instead and never go through `ComputeBackend`. | — |

## Top weak-test risks (prioritized)

> **Stale as of 2026-09-03:** all four items below are about the pre-2026-04-25 `step.rs`/
> `step_full` implementation, which no longer exists (see the banner at the top of this file).
> `tests/pipeline_six_phases.rs` (item 4) and the `lazy_view_wired_into_step_full` canary
> (item 2) are both gone. Left as-is below for historical record rather than deleted — none of
> these questions are answerable against current code, since there's no current `step_full` to
> ask them of.

1. **`Announce` uses 3D distance (`Vec3::distance`), spec is silent.** All announce tests still place observers on the same z-plane as the speaker, so 3D vs planar is indistinguishable. Distance *constants* are now all pinned (HIGH #1/#2/#4 fixed), so a value change would fail — but a 3D→planar refactor of the same constant would not. If the intended semantics are planar (because announcements are "heard in town") then the impl is wrong; if 3D is intended, the spec should say so.

2. **`LazyView` is not wired into `step_full`.** Trait exists, `NearestEnemyLazy` implementation exists, unit tests exercise the trait surface — but nothing in the tick pipeline calls `invalidate_on_events` on lazy views. Canary test `lazy_view_wired_into_step_full` is `#[ignore]`d; un-ignore when wired.

3. **`MaskValidityInvariant::check` (trait impl) still returns `None`.** The real check is `check_with_scratch`, invoked by `step_full`. Documented-by-design but a registering caller who expected `check` to fire would be surprised.

4. **Phase-5 view-fold skip not asserted.** `pipeline_six_phases` value assertions now catch constant-zero / out-of-range telemetry, but no test proves that a `MaterializedView::fold` was actually invoked during the tick (as distinct from the pre-step compute). Would need a view that mutates visibly on a known-frequency event.

## Open verification questions

Concrete ambiguities that code review can't resolve — audit one at a time.

> **Stale as of 2026-09-03:** items 1–4, 6, and 10 below cite specific `step.rs` line numbers /
> functions (`shuffle_actions_in_place`, `UtilityBackend` mask-bit interplay) from the
> pre-2026-04-25 implementation; that file is now an `unimplemented!()` stub, so these line
> references no longer point at live logic. Items 5, 7, 8, 9, 11 are about primitives that are
> still live (`EventRing`, `AgentId`, `hot_pos()`, `UtilityBackend::evaluate`, co-occupancy) and
> remain open questions worth investigating.

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
5. **Viz harness (Plan 3.0)** — **STALE (2026-09-03):** this harness (`crates/viz`) no longer
   exists — it was deleted in the 2026-05-02 Phase 7 wolf-sim wipe and never rebuilt. The "Visual
   check" column throughout this doc and the checklist below are historical; see that section for
   what (if anything) substitutes today.

## References

- `spec.md` — runtime contract (§§1–26)
- `README.md` — tree intro
- `docs/superpowers/plans/` — per-plan implementation intent
- `crates/engine/src/` — Rust implementation. **Not** "Serial only" in the sense the old text
  implied a working Serial tick driver — see the top-of-file banner: there is currently no live
  tick driver of any kind in this crate, Serial or GPU.
- `crates/engine/tests/` — the "157-test suite (+1 ignored LazyView integration canary)" figure
  is stale (it predates the 2026-04-25 purge that deleted a large share of these files). A
  2026-09-03 `grep -rc '#\[test\]' crates/engine/tests/*.rs` count found **282 `#[test]` functions
  across 65 files** — not a like-for-like replacement number (many old files are gone, many new
  DSL/ability-corpus files were added since), but the best current approximation without a full
  `cargo test -p engine` run reconciling it.

## Visual-check checklist

> **STALE — historical record only (2026-09-03 audit).** This checklist described acceptance
> criteria for the Plan 3.0 viz harness (`cargo run -p viz -- <scenario>`, scenario files under
> `crates/viz/scenarios/*.toml`). **None of that exists today**: `crates/viz` and every
> `viz_*.toml` scenario file named below were deleted in the 2026-05-02 Phase 7 wolf-sim wipe,
> and the `step.rs` behavior most rows were built to catch regressions in (Move/Attack/Announce
> execution, `ATTACK_RANGE`, mask pruning) is itself now an `unimplemented!()` stub — see the
> banner at the top of this file. There is no direct current equivalent: today's runnable viewers
> (`cargo run -p sim_app --bin viz_app --features bin-viz_app -- <sim_name>` for a terminal view,
> `cargo run -p engine_play --bin play <fixture>` for a generic windowed player, and
> `crates/viewer_runtime`'s `viewer_app`/`vs_viewer` binaries for two bespoke GPU-voxel pilots)
> drive entirely different compiled fixtures (`dungeon_horde`, `vampire_survivors`, and whatever
> is wired into `viz_app`'s `SIMS` table) with different mechanics than the "4 blue voxels + 1 red
> wolf" scenario these rows describe — they are not a substitute walkthrough for V1–V9 below, just
> the closest thing this workspace currently has to "run a fixture and watch it." Rewritten
> per-row equivalents were not invented here because none of the described scenario/behavior pairs
> currently exist to point at; a future pass would need to author new fixtures and scenarios from
> scratch, not just relabel these.

Each item below is a **historical** acceptance criterion for the retired Plan 3.0 viz harness.

| # | Scenario | Expected visual | Catches regression in |
|---|---|---|---|
| V1 | `crates/viz/scenarios/viz_basic.toml` — **file no longer exists; `crates/viz` itself is deleted** | Ground + 4 blue voxels in an 8 m square + 1 red voxel 20 m NE; the red voxel walks toward the blue cluster over ~2 s. | Move action (§9), nearest-enemy utility scoring. |
| V2 | `viz_basic.toml`, after ~10 s — **file no longer exists** | One or more blue voxels have disappeared; a black voxel persists where each died. | Attack damage + AgentDied emission. |
| V3 | `viz_attack.toml` — **file no longer exists** | Red voxel closes 3 m in ~3 ticks, then a short red line pulses between attacker and target every tick until the human dies. | ATTACK_RANGE = 2.0 m, attack-line overlay ingest. |
| V4 | `viz_attack.toml` post-death — **file no longer exists** | Single black voxel at the former human position; wolf idle (no more targets). | AgentDied cleanup, mask pruning of dead targets. |
| V5 | `viz_announce.toml` + test backend that emits `MacroAction::Announce` (future plan) — **file no longer exists; the "future plan" never landed before the wipe** | White ring expands from speaker over 3 ticks, covering 80 m. Listeners inside 30 m get memories (check events log). | Announce audience enumeration + overhear scan (§10). |
| V6 | Any scenario, paused, pressing `.` — **the `viz` binary this control belonged to no longer exists** | Tick advances by 1 per press; HUD prints `tick={n+1}`. | Pause/step determinism, no accumulator leak. |
| V7 | Any scenario, pressing `]` 4 times — **same, binary gone** | HUD prints `speed=16.00x`; agents move visibly faster; fps ≥ 30. | Tick accumulator math, burst-cap behavior. |
| V8 | Any scenario, pressing `R` — **same, binary gone** | Agents snap back to spawn positions; HUD reports `tick=0`; overlays clear. | Reload path cleans sim + overlays. |
| V9 | `viz_basic.toml`, wait until wolf reaches the human cluster — **file no longer exists** | When humans and wolf converge, you should see a vertical stack of cubes (different colors) where otherwise they'd visually collapse into one cube. | Viz stacking workaround for engine's missing collision detection (Plan 3.1 Task 3). |

Known gaps (historical, pre-dates the removal above):
- V5 requires a test backend that emits `Announce`; `UtilityBackend`
  only emits the 7 implemented micros. Becomes live when a future plan
  wires an announce-enabled policy.
- No in-window HUD — HUD is stdout-only. A later plan can layer egui
  or a text-shader overlay; deliberately out of scope for Plan 3.0.
