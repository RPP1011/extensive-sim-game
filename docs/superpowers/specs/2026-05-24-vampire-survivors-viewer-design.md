# Vampire Survivors — DSL Waves + Execution + Voxel Viewer (Design)

> Status: design, awaiting review. Next: `writing-plans` → implementation plan with AIS (P8).
> Builds on: `docs/superpowers/specs/2026-05-24-vampire-survivors-design.md` (the foundation `.sim`, landed on `main`). This spec covers roadmap **Slices 2 + 9 + 10** as one phased push: in-engine waves, GPU execution, and a 3D voxel viewer.

## 1. Goal

Make `vampire_survivors` **runnable and watchable in 3D**: enemies spawned by DSL-authored waves (the ability system's `Summon`), the sim executing on the GPU, rendered live in the `viewer_runtime` voxel viewer. Success = a windowed run where the player kites a continuously-spawning swarm, weapons cull it, and the run is watchable.

The work is one spec but **four internally-gated phases (A→D)**; each phase is independently verifiable, and Phase B is validated headlessly before the viewer work depends on it.

## 2. Background: why this is more than wiring

The foundation slice is **compile-gate only** — `assets/sim/vampire_survivors.sim` lowers to WGSL kernels but nothing executes or renders. Investigation of the runtime/viewer stack found:

- **No per-fixture crate** — fixtures live in the `sims` mega-crate; adding `"vampire_survivors"` to `crates/sims/build.rs`'s fixture list auto-generates a `GeneratedRuntime { try_new(seed, agent_count), step() }`.
- **GPU required** — compiled `.sim` runtimes have no CPU/Serial path; `GpuContext::new_blocking()` needs a real GPU or software Vulkan. The dev host has both (`/dev/dri/renderD128`, lavapipe `lvp_icd.json`).
- **The summon last-mile is unbuilt.** The `Summon` EffectOp (variant 24), `EffectSummonApplied` chronicle (kind 62), and `apply_summon_event_to_state` all exist, and `wave_defense.sim` authors tier-ramped waves via the ability system. **But** `apply_summon_event_to_state` mutates the legacy CPU `SimState`, not the GPU runtime's buffers, and the mega-crate `step()` / viewer / `engine_voxel` do **not** drain `EffectSummonApplied` into live GPU agents. So today even `wave_defense` emits summon chronicles that spawn nobody on the GPU. **Building that drainer (Phase B) is the keystone.**
- **The viewer is dungeon-specific.** `viewer_runtime`'s `ViewerApp` hardcodes `sims::dungeon_horde::GeneratedRuntime` + room-based terrain/seeding + a dungeon-specific `VoxelBridge`. Pointing it at an open-arena fixture is ~2–4 days of parallel-path plumbing.

## 3. The crux: summon → live GPU agent allocation

**Decision: Approach 1 — host-readback allocator.**

- **Approach 1 (chosen):** in the step-wrapper, read back `EffectSummonApplied` records from the GPU event ring, compute dead-slot assignments on CPU (mirroring `apply_summon_event_to_state`'s linear-pool alloc + `per_agent_u32` seeded positions), and `write_buffer` the new agents' SoA fields. Smallest, all-Rust, debuggable; reuses the viewer's existing readback machinery; **matches wave_defense's own documented "GPU emits chronicle, host applies — P3 cross-backend boundary."** Reuses the existing allocation *algorithm*, not the `SimState` signature.
- **Approach 2 (deferred):** a GPU-side WGSL kernel atomically claiming dead slots — P3-pure and scalable, but substantial emitter/WGSL work and race-free slot claiming; noted as a future purity/perf slice.
- **Approach 3 (rejected):** DSL-expressed allocation — not viable; the DSL has no slot-allocation primitive (that *is* the gap).

## 4. Phases

### Phase A — DSL waves (pure `.sim`, no Rust)

- **Components:** a `Spawner` role via the existing mana-band scheme — band `[2.5, 3.5]` (static agents at arena edges) — plus tier-ramped `Summon` verbs (`SummonSmall/Medium/Large/Horde`, counts 8/16/32/64) gated by `world.tick` windows, adapted from `wave_defense.sim`'s spawner verbs. Each emits `summon "enemy" N for <lifetime>` → `EffectOp::Summon` → `EffectSummonApplied` chronicle (actor=spawner, template_hash, count, lifetime_ticks).
- **Config:** `wave_period`, tier tick-window thresholds, spawner mana-band bounds, enemy spawn hp/lifetime.
- **Data flow:** spawner agent (spawner-band, alive) → every `wave_period` ticks the tick-windowed verb fires → GPU dispatcher writes `EffectSummonApplied` records to the event ring.
- **Test:** extend `crates/dsl_compiler/tests/vampire_survivors_compile.rs` — assert the Summon dispatcher kernel emits and the `EffectSummonApplied` chronicle write is present in the spawner kernel body. (Exact verb syntax copied from `wave_defense.sim`; reconcile against it if lowering differs.)

### Phase B — Summon allocator (keystone; host module)

- **Component:** one focused host module (`crates/sims/src/summon_drain.rs`, single responsibility: chronicle → live GPU agents). Core fn `drain_summons(runtime, tick, seed)`:
  1. Read back `EffectSummonApplied` (kind 62) records from the event ring for the tick.
  2. **Sort by `seq` (then actor slot)** for deterministic ordering (P3/P11).
  3. For each record: claim `count` dead slots (`alive==0` scan or a maintained free-cursor), and set per slot: `alive=1`, `hp`=spawn hp, `mana`=enemy-band, `creature_type`=enemy, `pos`=spawner pos + `per_agent_u32(seed, new_slot, tick, purpose)`-seeded offset. Cap `count` at remaining pool (mirror `apply_summon_event_to_state`'s exhaustion tolerance).
  4. Batch-`write_buffer` the mutated SoA columns (`agent_alive_buf`, `agent_hp_buf`, `agent_mana_buf`, `agent_pos_buf`, `agent_creature_type_buf`).
- **Interface:** takes a `&mut GeneratedRuntime`-like handle exposing the buffer fields + `gpu.queue`; returns count allocated. Reusable by wave_defense, vampire_survivors, and the viewer.
- **Reuses** the allocation *algorithm* from `apply_summon_event_to_state` (linear pool alloc, deterministic) but targets GPU buffers.
- **Test (headless, validates the keystone on `wave_defense` first):** construct the wave_defense runtime, step N ticks calling `drain_summons` each tick, read back `agent_alive_buf`, assert the alive-enemy count is 0 initially and grows across wave tiers. This proves the keystone independent of vampire_survivors and the viewer.
- **Risk:** correctly parsing the event-ring record layout for kind 62 (offsets); check for an existing readback helper in `crates/engine/src/gpu/event_ring.rs` before hand-rolling.

### Phase C — Execution (headless vampire_survivors)

- Add `"vampire_survivors"` to the fixture `matches!()` list in `crates/sims/build.rs`.
- **Seeding fn** (`crates/sims/src/vampire_survivors_seed.rs` or co-located): player at slot 0 (player-band mana, arena center, spawn hp, `alive=1`); K spawner agents at arena edges (spawner-band, static, `alive=1`); a node if needed; enemy slots `alive=0` (allocator fills them).
- **Headless driver** (an integration test in `crates/sims/tests/` or a small bin): construct runtime → seed → step T ticks, draining summons each tick → read back positions + alive counts. Assert: player position changes over time (kiting), enemy count rises (waves) then partially falls (weapon kills), and the loop completes T ticks without panic (P10).
- **Risk:** confirm the generated runtime exposes the buffer fields the seeder/allocator/readback need (`agent_pos_buf`, `agent_alive_buf`, `agent_mana_buf`, `agent_hp_buf`, `agent_creature_type_buf`). If a needed column isn't an External buffer, adjust the `.sim` so it is.

### Phase D — Voxel viewer

- **Open flat arena:** replace dungeon-room generation with a flat bounded floor plane (a `seed_arena` analogous to `seed_voxel_dungeon`), since vampire_survivors is open-field, not rooms.
- **Seeding:** reuse Phase C's seeding fn to place player + spawners into the viewer's runtime buffers.
- **`VoxelBridge` refresh:** read back agent SoA (pos, mana→role, alive, hp) and paint voxels with a palette — player, enemy, and spawner each a distinct `MAT_*` color; dead agents skipped. **Call `drain_summons` each tick** inside the viewer step so enemies spawn live while watching.
- **Wiring:** a `vampire_survivors` path in `viewer_runtime` parallel to the dungeon_horde path (new `ViewerApp` variant or a parameter selecting fixture + seeding + bridge). Keep the dungeon_horde path untouched.
- **Test:** manual (run the viewer binary, watch the swarm) + a headless smoke that constructs the vampire_survivors viewer variant and steps a few ticks without panicking (skips gracefully if no GPU adapter).

## 5. Constitution check (for the plan's AIS / P8)

- **P1 (compiler-first)** ✅ — the *summon decision* (when/how many) is authored in the DSL (Phase A). The Phase B allocator is sanctioned **runtime lifecycle** ("GPU emits chronicle, host applies", per wave_defense's documented design), not hand-written sim-rule logic in an engine handler path. Seeding (Phase C) and rendering (Phase D) are runtime/viewer code, not rules.
- **P3 (cross-backend parity)** — host-side allocation is a determinism boundary, made deterministic by `seq` ordering. GPU-side allocation (Approach 2) is the P3-pure future; this slice is GPU-execution + host-apply, behavioral cross-backend parity deferred.
- **P5 (keyed PCG)** ✅ — spawn positions via `per_agent_u32`. (Caveat: `per_agent_u64`'s ahash is not toolchain-stable — affects bit-exact replay, not the demo.)
- **P2 (schema-hash)** — N/A: enemies reuse existing Agent SoA columns (`hp/mana/pos/creature_type/alive`); no new engine columns, no hash bump.
- **P10 (no panic on hot path)** — the headless driver (Phase C) asserts T ticks complete without panic.
- **P8** — the implementation plan carries the full AIS.

## 6. Known caveats (noted, not blockers for a visual demo)

- **f32 RMW race** on shared-target HP: multiple weapons (bolt + nova) damaging the same enemy in one tick race on GPU (last-writer-wins) — affects exact HP/replay, not "enemies appear, take damage, die."
- **ahash determinism drift** (`per_agent_u64`): same Cargo.lock + different rustc → different RNG stream. Affects bit-exact replay across toolchains, not a single watchable run.
- **Binding-count limits:** GPU context bumps `max_storage_buffers_per_shader_stage` to 32; vampire_survivors with several custom fields + abilities could approach it. Watch for binding-count emit errors when adding spawner/summon machinery.

## 7. File map

- `assets/sim/vampire_survivors.sim` — Phase A spawner role + Summon verbs (modify).
- `crates/dsl_compiler/tests/vampire_survivors_compile.rs` — Phase A compile-gate assertion (modify).
- `crates/sims/build.rs` — add `"vampire_survivors"` (modify).
- `crates/sims/src/summon_drain.rs` — Phase B host allocator (create).
- `crates/sims/src/vampire_survivors_seed.rs` — Phase C seeding (create).
- `crates/sims/tests/vampire_survivors_exec.rs` — Phase B (wave_defense) + Phase C (vampire_survivors) headless tests (create).
- `crates/viewer_runtime/src/vampire_survivors.rs` (+ arena seeding, bridge) and a viewer binary entry — Phase D (create/modify).

## 8. Out of scope (future slices)

- GPU-side summon allocator (Approach 2, P3 purity).
- The full level-up upgrade *menu* with cross-item evolution (foundation spec Slice 5).
- Physical XP gems + magnet pickup (foundation spec Slice 3).
- Per-enemy archetypes / elites / bosses (foundation spec Slice 6) — this slice uses one enemy template.
- Run timer / meta-progression (foundation spec Slice 8).
