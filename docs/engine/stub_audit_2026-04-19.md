# Engine Stub Audit — 2026-04-19

> Hunts for the "PoolNonOverlap returned None" class of bug: places where
> docs/commits claim implementation but code is hollow. Prior related audit:
> `docs/engine/verification_audit_2026-04-19.md` (resolved HIGH + MEDIUM
> boundary-unpinned tests). This audit is broader — discrepancies between
> "claimed done" and "actually does something."
>
> **Snapshot:** audited against SHA `773a6104` (`feat(engine): EffectOp::ModifyStanding — symmetric clamp [-1000, 1000]`). A parallel subagent may be executing Combat Foundation Phase 4 in `crates/engine/src/ability/` + `event/` + `state/`; Phase 4's in-flight changes are **not** reflected here. Re-audit after the Phase 4 merge.
>
> **Scope:** `crates/engine/src/**/*.rs`, `crates/engine/tests/*.rs`, `docs/engine/{spec,status}.md`. State-port stubs documented in `docs/superpowers/plans/2026-04-19-engine-plan-state-port.md` §"Deferred fields" and §"Supporting types added" are treated as documented-stubs and triaged to LOW.

## Summary

- **CRITICAL**: 3 findings (silently broken; users rely on these)
- **HIGH**: 5 findings (stubs that docs don't label; misleading)
- **MEDIUM**: 3 findings (stubs documented elsewhere but not marked in status.md)
- **LOW**: 4 findings (cosmetic / explicit TODO placeholders / already-flagged)

---

## CRITICAL findings

### 1. Spatial index orphaned — `crates/engine/src/spatial.rs:31-57` + `docs/engine/spec.md:187-206`

**Claim** (spec.md §6 + status.md §6 row): "✅ 🎯" for "Spatial index (2D-column BTreeMap + z-sort + MovementMode sidecar)". Spec explicitly states: "Insert/remove: both backends rebuild on `spawn_agent` / `kill_agent` / `set_agent_pos`."

**Reality** (`spatial.rs:31`): `SpatialIndex::build(state)` is a from-scratch snapshot constructor. No code in `state/mod.rs::spawn_agent`, `kill_agent`, `set_agent_pos`, in the cascade dispatcher, in `step_full`, or in `mask.rs` builds, holds, or updates a `SpatialIndex`. A repo-wide `grep SpatialIndex crates/engine/src` returns only the definition site. `mask.rs::mark_flee_allowed_if_threat_exists` + `mark_attack_allowed_if_target_in_range` run O(N²) `state.agents_alive().any(...)` scans; `step.rs::Announce` dispatch iterates all agents linearly.

**Impact**: the published spec promises sub-linear spatial queries and `SpatialIndex` integration; callers who spawn/kill/move agents expect the index to stay current. It doesn't — they'd either query an empty/stale index or (if never built) skip the index entirely. At agent-caps of hundreds/thousands the fallback O(N²) is performance-critical; at any cap the stale-index risk is correctness-critical.

**Test coverage status**: `tests/spatial_index.rs` and `tests/proptest_spatial.rs` exercise `SpatialIndex::build` on a fresh `SimState` snapshot. Neither test runs `step_full` / `spawn_agent` / `set_agent_pos` against a pre-built index to verify freshness. The proptest explicitly rebuilds the index per iteration.

**Proposed fix severity**: large (requires deciding lifetime/ownership of the index — per-tick rebuild vs incremental updates — and threading it through `step_full` or `SimState`).

---

### 2. `evaluate_cast_gate` not consulted by mask build — `crates/engine/src/mask.rs:151-164` vs `ability/gate.rs:1-5` + `tests/mask_can_cast.rs`

**Claim** (`ability/gate.rs:1-2`): "`evaluate_cast_gate` — the cast-time predicate **the mask-building** and the `CastHandler` both consult." Test file named `tests/mask_can_cast.rs` implies the mask consults the gate.

**Reality** (`mask.rs:151-164`): `mark_domain_hook_micros_allowed` unconditionally sets `Cast`, `UseItem`, `Harvest`, `PlaceTile/Voxel`, `HarvestVoxel`, `Converse`, `ShareStory`, `Communicate`, `Ask`, `Remember` to `true` for every alive agent — ignoring every gate predicate. A repo-wide grep confirms `evaluate_cast_gate` is called by tests and by the Task-9 example backend in `cooldown_blocks_recast.rs`, never by any mask builder. The `mask_can_cast.rs` tests exercise the function directly; they do not prove that the mask respects the gate.

**Impact**: any backend that reads `mask.can_cast(agent)` will always see `true`, regardless of cooldown, stun, engagement lock, or target range. A hostile backend that forges a `MicroKind::Cast` bypasses `MaskValidityInvariant` because the mask is permissive. The kernel itself (`apply_actions` `MicroKind::Cast` branch) performs zero gate validation — it unconditionally pushes `AgentCast`. So a forged cast triggers the full cast cascade (damage, heal, shield, etc.) from a stunned / dead / on-cooldown agent. No test exercises this path. The cast-gate test suite (`mask_can_cast.rs`) is **deceptively named**: it tests the helper, not the mask integration.

**Test coverage status**: `tests/mask_can_cast.rs` — 11 tests — all call `evaluate_cast_gate` directly. Zero tests verify that `mask.micro_kind[slot*18 + Cast as usize]` is ever set to `false` by the engine.

**Proposed fix severity**: medium (add a mask-side gate predicate pass for each of the 11 domain-hook micros; at minimum for `Cast`, which is the only one with a defined gate today).

---

### 3. `DamageHandler` docstring claims melee-parity events, code emits a strict subset — `crates/engine/src/ability/damage.rs:9-11`

**Claim** (`damage.rs:9-11` docstring): "Mirrors the direct-`Attack` death semantics in `step.rs`, so **cast-delivered and melee-delivered kills trigger identical replayable events**."

**Reality** (`damage.rs:26-52`): the handler emits `AgentDied` on lethal damage and mutates state; it **never** emits `Event::AgentAttacked`. Melee (`step.rs:339-344`) emits both `AgentAttacked { damage: ATTACK_DAMAGE, .. }` AND `AgentDied` on kill. Cast-damage replay stream is `[EffectDamageApplied, AgentDied]`; melee replay stream is `[AgentAttacked, AgentDied]`. Not identical. `MostHostileTopK::update` (`view/topk.rs:46-50`) only listens to `AgentAttacked`, so cast-dealt damage is **invisible** to the hostility view — a caster who one-shots a target is not registered as an aggressor anywhere.

**Impact**: (a) `replayable_sha256` diverges between a sim that dealt the same damage via melee vs cast — this is a cross-backend parity hazard when GPU eventually ports. (b) Any `MaterializedView` / `TopKView` keyed on `AgentAttacked` (the natural cue for "who hurt me") silently misses all cast damage. (c) The docstring promise itself is a future-bug landmine: a reader who relies on the comment will write a view/handler that listens to `AgentAttacked` and inherits the gap.

**Test coverage status**: `tests/cast_handler_damage.rs::lethal_damage_emits_agent_died_and_kills` asserts `AgentDied` is emitted. No test asserts whether `AgentAttacked` is *also* emitted — the assertion style is `find_map` on `AgentDied`, leaving `AgentAttacked` unconstrained in either direction. `view_topk.rs` tests feed synthetic `AgentAttacked` events directly; no integration test runs `CastHandler + DamageHandler` + a `TopKView` and compares against melee.

**Proposed fix severity**: small (either delete the docstring claim, or emit `AgentAttacked` from `DamageHandler` with `damage = min(overflow, pre-clamp hp)` and add a test that asserts parity of `replayable_sha256` between a melee-only run and a cast-only run at the same net damage).

---

## HIGH findings

### 4. `RecordMemory` emitted but never written to agent memory — `crates/engine/src/step.rs:541-572` + `state/mod.rs:561`

**Claim** (status.md §10 row for "Announce cascade (audience enumeration, primary + overhear)"): "✅ 🎯". Status table row implies the full Announce → memory pipeline is done. `state/mod.rs:102` comment describes `cold_memory` as a per-agent event ring meant to receive these.

**Reality**: `apply_actions`' `MacroAction::Announce` arm emits `Event::RecordMemory { observer, source, fact_payload, confidence, tick }` per recipient (primary at 0.8, overhear at 0.6). No cascade handler is registered for `EventKindId::RecordMemory` — it has an ordinal (22), a hash_event arm (ring.rs:275), and a tick() arm, but no entry in `register_engine_builtins` nor a user-facing writer. `state.push_agent_memory` is called only by `tests/state_memory.rs`. Consequently the per-agent memory store stays empty forever, even though the event log faithfully records the broadcasts.

**Impact**: any consumer who believes "the Announce subsystem is ✅" and reads `state.agent_memory(observer)` expecting to find the broadcast they just heard will find an empty slice. The comment on `Event::RecordMemory` at `event/mod.rs:48` does not flag this as deferred. Plans that wire `MemoryEvent` (economic / GOAP / theory-of-mind, per roadmap) will silently read empty memories and conclude "no one announced anything" for the entire history of the sim.

**Test coverage status**: `tests/announce_audience.rs` + `tests/announce_overhear.rs` assert `RecordMemory` lands in the event ring with correct confidence. **Neither** asserts `state.agent_memory(observer)` contains a corresponding `MemoryEvent`. The state-level side of the claim is unverified.

**Proposed fix severity**: small (register a `RecordMemoryHandler` that pushes a `MemoryEvent` into `cold_memory[observer]`; one new handler + one new test that asserts `state.agent_memory(recipient).len() == 1` after an Announce tick).

---

### 5. `metrics::CASCADE_ITERATIONS` constant exists + schema-hashed but never emitted — `crates/engine/src/telemetry/metrics.rs:6` + `step.rs:192-198`

**Claim** (`metrics.rs:1` "Built-in engine metric names. Change means schema-hash bump." + `schema_hash.rs:61`: `"BuiltinMetrics:tick_ms,event_count,agent_alive,cascade_iterations,mask_true_frac"` — `cascade_iterations` is advertised as a built-in metric alongside the four that are actually emitted).

**Reality** (`step.rs:192-198`): `step_full` emits `TICK_MS`, `EVENT_COUNT`, `AGENT_ALIVE`, `MASK_TRUE_FRAC`. Grep for `CASCADE_ITERATIONS` usage in `src/`: exactly two hits — the declaration itself and a doc-reference. `CascadeRegistry::run_fixed_point` does not call `telemetry.emit*` at all. The constant string `"engine.cascade_iterations"` is unreachable as a metric name.

**Impact**: anyone reading the schema-hash fingerprint or the `metrics.rs` module and wiring a dashboard that filters `engine.cascade_iterations` will get an empty series forever. The schema-hash entry falsely implies the metric is a stable published contract.

**Test coverage status**: `tests/pipeline_six_phases.rs` filters `telemetry` by each of the four emitted metric names; it does NOT assert the absence of CASCADE_ITERATIONS emissions. `tests/telemetry_*` cover the sink trait, not the set of metrics `step_full` emits.

**Proposed fix severity**: small (either emit `CASCADE_ITERATIONS` from `run_fixed_point` — requires threading a `&dyn TelemetrySink` into it, which step_full already holds — or delete the const and rebaseline the schema hash).

---

### 6. `AggregateId` and `ResourceRef` claimed-but-do-not-exist — `docs/engine/status.md:53` (§4 row)

**Claim** (status.md §4 Subsystem row): "`AgentId`, `EventId`, `AggregateId`, `QuestId`, `GroupId`, `ResourceRef`, `ItemId` (all `NonZeroU32`) | ✅".

**Reality**: a repo-wide grep for `AggregateId` or `ResourceRef` in `crates/engine/` returns zero matches. `ids.rs` defines `AgentId`, `GroupId`, `ItemId`, `QuestId`, `AuctionId`, `InviteId`, `SettlementId` via the `id_type!` macro; `EventId` is an inline struct; `AbilityId` is in `ability/id.rs`. The two names in the status row — `AggregateId` and `ResourceRef` — are phantom; nothing in the engine or its tests can construct them. (Aggregate identification *is* provided by `PoolId<T>` against `aggregate/pool.rs`, but that's a phantom-typed wrapper, not a name called `AggregateId`.)

**Impact**: a reader of status.md writing a cascade handler or a probe against the engine API will import these names and find compile errors, or will hallucinate an API surface that doesn't exist. The row is a documentation fabrication.

**Test coverage status**: N/A — they don't exist. `tests/event_id_threading.rs` exercises `EventId` only. No test imports an `AggregateId` or a `ResourceRef`.

**Proposed fix severity**: small (update status.md §4 to list the id types that actually exist — `AgentId, GroupId, QuestId, ItemId, AuctionId, InviteId, SettlementId, AbilityId, EventId` — or delete the two phantoms).

---

### 7. `tests/aggregate_types.rs` cited in status.md but file does not exist — `docs/engine/status.md:71` (§16 row)

**Claim** (status.md §16 row "Tests" column): "`tests/aggregate_pool.rs`, `tests/aggregate_types.rs`".

**Reality**: `tests/aggregate_pool.rs` exists (64 LoC, 3 tests). `tests/aggregate_types.rs` does not (`glob crates/engine/tests/aggregate_types*` returns zero matches). Status.md's "Weak-test risk" cell — "`kill_then_alloc_reuses_slot_and_clears_contents` — same single-reuse issue as Pool" — references a test in a nonexistent file.

**Impact**: readers trying to cross-reference the status table against the test suite will chase a dead link. The row claims broader test coverage than exists.

**Test coverage status**: `aggregate_pool.rs::kill_then_alloc_reuses_slot_and_clears_contents` does exist — status.md quoted the right test name but the wrong file.

**Proposed fix severity**: small (fix the file name in status.md, or delete the reference).

---

### 8. `FailureMode::Rollback { ticks }` defined and advertised but no rollback is performed — `crates/engine/src/invariant/registry.rs:22-34` + `schema_hash.rs:63`

**Claim** (`schema_hash.rs:63`: `"FailureMode:Panic,Log,Rollback"`). Enum variant name "Rollback" implies the engine rewinds `ticks` steps on violation. The field `{ ticks: u32 }` is exposed as part of the variant's public shape.

**Reality** (`registry.rs:26-32`): `InvariantRegistry::check_all` only branches on `FailureMode::Panic` (panics immediately). For all other modes — `Log` AND `Rollback` — it collects the violation into `reports` and returns. There is no rewind-state machinery anywhere in the engine. No SimState snapshot is taken before `step_full`; no rollback is attempted when a `Rollback`-mode invariant fires. `step.rs:183` translates the variant to the string `"rollback"` for telemetry labeling, which is the ONLY place the variant is observably handled, and that handling is a label, not behavior.

**Impact**: a user who registers an invariant with `failure_mode() -> FailureMode::Rollback { ticks: 3 }` expecting the sim to rewind 3 ticks on violation will get the same behavior as `FailureMode::Log` — silent drift with a log record. The telemetry label `"rollback"` actively lies about the underlying action.

**Test coverage status**: `tests/invariant_dispatch_modes.rs` covers Panic (via `#[should_panic]`) and Log (assert violation reports). There is **no** test of Rollback — no test asserts state rewinds, no test even documents what "Rollback" would mean behaviorally.

**Proposed fix severity**: medium (either implement Rollback — requires state snapshot/restore machinery, a meaningful plan — or downgrade the variant to a flagged `// TODO(Plan N): rewind machinery` and/or remove it from the schema hash baseline).

---

## MEDIUM findings

### 9. `channel::channel_range` is a pure orphan — `crates/engine/src/channel.rs:16-30`

**Claim** (`channel.rs`): range-by-channel function exists with distinct values per channel (Speech, PackSignal, Pheromone, Song, Telepathy, Testimony). Tested in `tests/channel_filter.rs::speech_range_is_vocal_strength_scaled` + `telepathy_is_unbounded`. Spec §10 (`Announce`) + the `Capabilities.channels` field imply speech / pack-signal / pheromone should determine the radius of `AnnounceEmitted` events.

**Reality**: `channel_range` is called only by its tests. `step.rs::Announce` uses the constant `MAX_ANNOUNCE_RADIUS = 80.0` for the `Anyone` / `Group` audiences and the constant `OVERHEAR_RANGE = 30.0` for the overhear scan. It never consults `state.agent_channels(speaker)` nor calls `channel_range`. A Wolf speaker (channel `PackSignal`, range 20) and a Human speaker (channel `Speech`, range 30 at vocal=1.0) both reach 80m, contradicting the per-channel physics the function encodes.

**Impact**: per-channel range semantics are advertised by the module (and the test suite validates the function's numeric behavior) but the engine's only announce path ignores them. A reader who audits `channel.rs` concludes "channel range gates audience size"; the engine actually doesn't care.

**Test coverage status**: `channel_filter.rs` covers the function in isolation. No test drives `step_full` with `MacroAction::Announce` from a Human speaker and a Wolf observer and asserts the Wolf does / doesn't receive a `RecordMemory` based on channel match. The test suite validates a pure function; no test validates its integration.

**Proposed fix severity**: medium (decide whether Announce is meant to respect channels — if yes, wire `channel_range` into `step.rs::Announce`; if no, delete `channel_range` or document it as reserved for a later plan).

---

### 10. `hot_attack_damage` + `hot_attack_range` are per-agent but the kernel uses global constants — `crates/engine/src/step.rs:14-16` vs `state/mod.rs:49-50`

**Claim** (state-port plan §"Hot fields added"): `attack_damage: f32 = 10.0`, `attack_range: f32 = 2.0` are per-agent SoA fields. Default values are documented as "matches step.rs constant" / "matches `ATTACK_RANGE`". state.md §Combat commits to per-agent stats, not globals.

**Reality** (`step.rs:14-16`): `ATTACK_DAMAGE = 10.0`, `ATTACK_RANGE = 2.0` are module constants consulted by `apply_actions`' `MicroKind::Attack` branch, by `OpportunityAttackHandler::handle`, and by `mask.rs::ATTACK_RANGE_FOR_MASK`. A caller who uses `state.set_agent_attack_damage(wolf, 25.0)` expecting a Wolf to deal 25 per swing will still see exactly 10 damage per AgentAttacked event. `hot_attack_damage` and `hot_attack_range` are read nowhere outside `state/mod.rs` + `schema_hash.rs`.

**Impact**: the SoA fields are deceptively writeable — setters + bulk slices exposed, test coverage asserting set/get roundtrips (`tests/state_combat_extras.rs`). A downstream compiler pass that emits `set_agent_attack_damage(caster, weapon.damage)` will have zero runtime effect.

**Test coverage status**: `state_combat_extras.rs` covers storage round-trips. No test asserts that setting `agent_attack_damage(agent, 50.0)` causes the next Attack action from that agent to land 50.0 damage. The state-port plan explicitly scopes these as "storage only; subsequent plans wire behaviour" — this is a documented stub.

**Proposed fix severity**: small (the state-port plan already defers wiring; either the next ability/combat plan connects the kernel to the SoA, or docstrings on these accessors explicitly say "storage only until Plan X").

Upgraded to MEDIUM (not LOW) because the defaults are silently consistent with the kernel constants, meaning the stub is invisible to anyone who doesn't write a test that mutates the field and re-runs a tick.

---

### 11. `tests/state_memberships.rs::etc` rely on `clear_agent_*` being invoked on slot reuse, but `kill_agent` does not clear cold collections — `crates/engine/src/state/mod.rs:237-243` vs `state/mod.rs:225-232`

**Claim** (state-port plan §"Design friction" #6): "every `SmallVec`-backed cold field needed an explicit `.clear()` in both `spawn_agent` and `kill_agent`". Plan asserts the clearing happens on kill.

**Reality** (`state/mod.rs:237-243`): `kill_agent(id)` sets `hot_alive[slot] = false` and calls `self.pool.kill_agent(id)`. It does **not** clear `cold_status_effects`, `cold_memberships`, `cold_inventory`, `cold_memory`, `cold_relationships`, `cold_class_definitions`, `cold_creditor_ledger`, or `cold_mentor_lineage`. The clearing actually happens at `spawn_agent` line 225-232 — i.e. on slot **reuse**, not on kill. Between a kill and the next spawn into that slot, the dead slot's collections remain visible through `agent_memberships(id)` / `agent_memory(id)` / etc., which return `Option<&[...]>` based on slot index, not on `alive`.

**Impact**: a read of a dead agent's cold collections during the kill→respawn gap returns the prior tenant's data. The slot-reuse test (e.g. `state_memberships.rs`) only asserts the SECOND spawn's slot is empty, not that a dead-slot query between kill and respawn is empty. A probe / chronicler that inspects `agent_memberships(dead_id)` mid-tick can observe ghost state. Accessor methods on dead slots are not mask-gated; spec does not commit to the semantics either direction.

**Test coverage status**: existing coverage asserts post-respawn clears, matching what the code actually does. Pre-respawn (dead-slot) reads are untested. The plan's claim that `kill_agent` clears is false; the actual design is spawn-side clearing.

**Proposed fix severity**: small (fix the state-port plan's claim OR move the clears to `kill_agent`; former is cheaper and matches current tests).

---

## LOW findings

| # | Subsystem / path:line | Finding | Severity rationale |
|---|---|---|---|
| 12 | `view/lazy.rs` + `tests/view_lazy.rs:108-136` | `LazyView` not wired into `step_full`; integration canary `#[ignore]`d. | Explicitly flagged in status.md as weak-test-risk #2 + header comment on the test; known deferred. |
| 13 | `state/mod.rs:49-69` (21+ per-agent hot stat/personality/psych-need fields) | `hot_level`, `hot_move_speed`, `hot_move_speed_mult`, `hot_mana`, `hot_max_mana`, `hot_armor`, `hot_magic_resist`, `hot_safety`, `hot_shelter`, `hot_social`, `hot_purpose`, `hot_esteem`, `hot_risk_tolerance`, `hot_social_drive`, `hot_ambition`, `hot_altruism`, `hot_curiosity` all read nowhere outside `state/mod.rs` + `schema_hash.rs`. Pure storage stubs. | Explicitly documented in state-port plan §"Hot fields added" + the plan's scope is "no behaviour; subsequent plans wire it in". All are correctly documented deferrals. |
| 14 | `state/mod.rs:108-112` (cold collections) | `cold_class_definitions`, `cold_creditor_ledger`, `cold_mentor_lineage` read nowhere in src. | Same — state-port plan §"Cold fields added" explicit deferral. |
| 15 | `ids.rs:25-29` | `SettlementId`, `AuctionId`, `InviteId`, `ItemId` declared, zero call sites in engine. | No doc actively claims these are consumed today; they're reserved for later plans (auction / settlement / item subsystems in roadmap). |
| 16 | `status.md:66` references `tests/proptest_step_no_panic.rs` | That filename does not exist. The intended test is in `proptest_baseline.rs::step_never_panics_under_random_sizing`. | Cosmetic doc drift; test exists, status.md cites wrong filename. |
| 17 | `trajectory_roundtrip.rs::python_roundtrip_preserves_values` | Test name promises value-preservation; body only compares `n_agents` and `n_ticks`. | Explicitly flagged by status.md §17 row in the Weak-test risk column; known weak. |

---

## Non-findings

Checked and concluded genuinely complete (or correctly documented-as-deferred):

- **`PoolNonOverlapInvariant`** — real check, fault-injection tests prove firing (`invariant/builtins.rs:60-78`, `tests/invariant_pool_non_overlap.rs:38-77`). Resolution from verification audit holds.
- **`MaskValidityInvariant::check` returns `None`** — trait impl is intentionally a no-op; real check is `check_with_scratch`, consumed by `step_full`. Documented at status.md weak-test-risk #3.
- **Event byte-packing round-trip** — all 35 `Event::*` variants present in both `Event::tick()` (event/mod.rs:75-114) and `hash_event` (ring.rs:114-360). No variants missing from either.
- **`EventKindId::from_event` coverage** — all 35 variants mapped (cascade/handler.rs:49-87). One-to-one.
- **`MAX_CASCADE_ITERATIONS = 8` behavior** — tested exactly in `cascade_bounded.rs` and `proptest_cascade_bound.rs`; the "actual count = 8" correction was landed (verification audit finding #8).
- **RNG golden values** — status.md notes "no direct golden test" but `rng.rs::tests::world_rng_golden_value` + `per_agent_golden_value` both pin exact u32/u64 values. Status.md is outdated, code is honest.
- **`AbilityId`, `GroupId`, `QuestId`, `AgentId`, `EventId`** — real id types with niche-optimized `Option<T>`, round-trip tested, used by real call sites.
- **Opportunity-attack cascade** — real handler (`ability/expire.rs:38-58`), tested in `engagement_move_slow.rs`; both melee-speed-reduction and event emission paths exercise it.
- **Cast cooldown + engagement + stun gates** — when `evaluate_cast_gate` is consulted directly, all 6 branches work (tested in `mask_can_cast.rs`). The gap in HIGH #2 is that the engine's MASK does not consult the gate — but the function itself is correct.
- **Stun/Slow expiry** — `ability/expire.rs::decrement_and_expire` + `StunHandler` / `SlowHandler` round-trip tests work (`stun_expiry.rs`, `slow_expiry.rs`, `combat_state_fields.rs`).
- **All 8 `Effect*Applied` handlers** are real, consume their event, mutate state, and are registered in `register_engine_builtins` (`cascade/dispatch.rs:43-57`).

---

## Appendix: methodology + grep commands used

1. `grep -rn "todo!\|unimplemented!\|unreachable!()" crates/engine/src/` — zero hits.
2. `grep -rn "#\[allow(dead_code)\]\|#\[allow(unused" crates/engine/src/` — one hit (`pool.rs:10` on `contracts::` imports that are conditionally unused in release), benign.
3. Read every `impl <Trait> for <Type>` in the engine — any method body under ~5 lines reviewed for stub-return semantics. No surviving `Option<Violation>::None` or `true`/`false`-constant returns beyond the documented `MaskValidityInvariant::check`.
4. Enumerated all 35 `Event::` variants; cross-referenced against `hash_event` + `Event::tick` + `EventKindId::from_event`. All three covered.
5. Enumerated all SoA fields in `SimState`; grepped their accessor names across `crates/engine/src/` excluding `state/mod.rs` + `schema_hash.rs`. Every orphan logged in LOW #13-14 (documented state-port stubs) or HIGH #10 (per-agent combat stats the kernel silently ignores).
6. For every `impl CascadeHandler for` — 10 handlers (OpportunityAttack, Cast, Damage, Heal, Shield, Stun, Slow, TransferGold, ModifyStanding; plus test handlers). All 9 engine handlers do real work. Verified by reading each `handle` body — shortest (`ModifyStanding`, 6 lines) dispatches to `adjust_standing` which actually mutates `SparseStandings`.
7. Verified `TopKView::update` (view/topk.rs:46-50) actually accumulates per-attacker damage — not a stub.
8. Verified `LazyView::compute` in `NearestEnemyLazy::compute` (view/lazy.rs:87-119) is a real O(N²) pairwise scan that populates `per_agent` and clears `stale`.
9. Verified both built-in invariants' real-check paths: `MaskValidityInvariant::check_with_scratch` iterates `scratch.actions` against the mask; `PoolNonOverlapInvariant::check` calls `SimState::pool_is_consistent` which walks alive/freelist.
10. Cross-referenced every test filename in status.md's "Tests" columns against `ls crates/engine/tests/`. Found two phantoms: `aggregate_types.rs` (HIGH #7) and `proptest_step_no_panic.rs` (LOW #16).
11. For each metric name in `telemetry/metrics.rs`, grepped `telemetry.emit*` in src. Found CASCADE_ITERATIONS unused (HIGH #5).
12. Checked `schema_hash.rs` fingerprint string against enum/struct variants in the code — all enum variants present; all claimed fields present. No drift on structural coverage.
