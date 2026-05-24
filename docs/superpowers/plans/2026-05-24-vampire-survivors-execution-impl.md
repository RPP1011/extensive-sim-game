# Vampire Survivors — Execution (Waves + GPU Run) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vampire_survivors` *execute* on the GPU with DSL-authored enemy waves — spawner agents cast `Summon` abilities, a host allocator turns the resulting chronicle records into live enemies, and a headless test proves the player kites a spawning, dying swarm.

**Architecture:** Three gated phases. **A:** add spawner agents + tier-ramped `Summon` verbs/abilities to `assets/sim/vampire_survivors.sim` (wave_defense pattern). **B:** a host "summon allocator" — a pure slot-assignment fn (unit-tested) + a GPU drain that reads `EffectSummonApplied` (kind 62) ring records back and writes new enemies into dead SoA slots. **C:** add the fixture to the `sims` mega-crate, seed initial state, and a headless integration test stepping the runtime with the drain. The **voxel viewer (Phase D)** gets its own concrete plan after C lands (see final task).

**Tech Stack:** World Sim DSL + `.ability` files, `crates/dsl_compiler`, `crates/sims` (mega-crate), `crates/engine` (GpuContext, EventRing), wgpu. References: `assets/sim/wave_defense.sim` + `assets/ability_test/wave_defense/Spawn*.ability` (wave pattern), `crates/sims/tests/navgrid_probe_pin.rs` (headless test template), `crates/viewer_runtime/src/lib.rs:1498-1556` (readback idiom), `crates/engine/src/ability/apply.rs:1629` (`apply_summon_event_to_state` algorithm).

**Spec:** `docs/superpowers/specs/2026-05-24-vampire-survivors-viewer-design.md`.

---

## Architectural Impact Statement (P8)

- **Existing primitives searched:** `wave_defense.sim` + its 4 `Spawn*.ability` files (DSL summon waves via `apply_ability` → `EffectOp::Summon` → `EffectSummonApplied` chronicle kind 62); `apply.rs::apply_summon_event_to_state` (dead-slot alloc algorithm, targets legacy `SimState`); `event_ring.rs` (`EVENT_STRIDE_U32=11`, kind@w0, kind-62 payload: w2=actor, w3=template_hash, w4=count, w5=lifetime); `build_helper.rs` generated `GeneratedRuntime` (`gpu/agent_*_buf/event_ring`, `try_new`/`step`); `navgrid_probe_pin.rs` (headless test idiom). Method: `rg` + targeted reads.
- **Decision:** DSL authors the wave *decision* (Phase A, pure `.sim`+`.ability`). A new host module `crates/sims/src/summon_alloc.rs` performs runtime-lifecycle slot allocation (Phase B) — sanctioned by wave_defense's documented "GPU emits chronicle, host applies" boundary. No new engine SoA columns, no emitter changes, no `EffectOp` variants.
- **Rule-compiler touchpoints:** DSL inputs added: spawner role + `Summon` verbs in `vampire_survivors.sim`; `assets/ability_test/vampire_survivors/Spawn{Small,Medium,Large,Horde}.ability`. Build wiring: `"vampire_survivors"` added to `crates/sims/build.rs` fixture list.
- **Hand-written downstream code:** `crates/sims/src/summon_alloc.rs` (runtime lifecycle, NOT sim-rule logic) + seeding fn + a headless test. No engine handler / generated-rule edits.
- **Constitution check:** P1 ✅ (wave decision in DSL; allocation is runtime lifecycle). P2 N/A (reuses existing `alive/hp/mana/pos` columns; no schema change). P3 — host allocation made deterministic by `seq` ordering; GPU-side alloc deferred. P5 ✅ (`per_agent_u32` seeded spawn offsets). P10 — Phase C test asserts T ticks complete without panic. P8 ✅ (this section).

---

## File Structure

- `assets/sim/vampire_survivors.sim` — **modify**: spawner mana-band config, wave config, 4 `Summon` verbs.
- `assets/ability_test/vampire_survivors/Spawn{Small,Medium,Large,Horde}.ability` — **create** (4 files): `summon "enemy" N`.
- `crates/dsl_compiler/tests/vampire_survivors_compile.rs` — **modify**: assert the Summon dispatcher + chronicle emit.
- `crates/sims/build.rs` — **modify**: add `"vampire_survivors"` to the fixture `matches!()` list.
- `crates/sims/src/summon_alloc.rs` — **create**: `SummonRecord`, pure `plan_allocations(...)`, GPU `drain_summons(...)`.
- `crates/sims/src/lib.rs` — **modify**: `pub mod summon_alloc;` (+ `pub mod vampire_survivors_seed;`).
- `crates/sims/src/vampire_survivors_seed.rs` — **create**: `seed_initial_state(rt)`.
- `crates/sims/tests/vampire_survivors_exec.rs` — **create**: headless integration test.

---

## A note on slot indices (read before Task A3)

`apply_ability N` references the ability **registry slot** for that `.ability`, assigned by the compiler from the fixture's ability corpus. The exact `N` per ability is not knowable a priori — Task A3 **determines them empirically** (build, inspect the generated registry order) rather than guessing. wave_defense used Small=6/Medium=5/Large=4/Horde=3, but vampire_survivors has a different corpus, so its indices will differ.

---

## Phase A — DSL waves

### Task A1: Spawner role + wave config

**Files:** Modify `assets/sim/vampire_survivors.sim`.

- [ ] **Step 1: Add spawner mana band + wave config to the `config vs` block**

Add these fields to the existing `config vs { ... }` block:

```
  // --- Phase A: spawner role + wave ramp ---
  spawner_mana_min: f32 = 2.5,
  spawner_mana_max: f32 = 3.5,
  wave_period:      u32 = 30,
  small_to_medium:  u32 = 1000,
  medium_to_large:  u32 = 2500,
  large_to_horde:   u32 = 4000,
  enemy_spawn_hp:   f32 = 12.0,
  enemy_lifetime:   u32 = 0,
```

- [ ] **Step 2: Build to confirm the config still lowers**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile vampire_survivors_compiles -- --nocapture`
Expected: PASS (config-only change; no behavior yet).

- [ ] **Step 3: Commit**

```bash
git add assets/sim/vampire_survivors.sim
git commit -m "feat(vampire_survivors): spawner mana band + wave-ramp config (Phase A)"
```

### Task A2: The four `Summon` ability files

**Files:** Create `assets/ability_test/vampire_survivors/Spawn{Small,Medium,Large,Horde}.ability`.

- [ ] **Step 1: Create the 4 ability files**

These mirror `assets/ability_test/wave_defense/Spawn*.ability` exactly, with template `"enemy"` and counts 8/16/32/64.

`assets/ability_test/vampire_survivors/SpawnSmall.ability`:
```
ability SpawnSmall {
    target: self
    cooldown: 3s
    hint: utility

    summon "enemy" 8 [UTILITY: 100]
}
```
`SpawnMedium.ability` — identical but `summon "enemy" 16`.
`SpawnLarge.ability` — identical but `summon "enemy" 32`.
`SpawnHorde.ability` — identical but `summon "enemy" 64`.

- [ ] **Step 2: (No standalone test yet — abilities are exercised once the verbs reference them in A3.) Commit**

```bash
git add assets/ability_test/vampire_survivors/
git commit -m "feat(vampire_survivors): 4 Summon ability files (enemy 8/16/32/64) (Phase A)"
```

### Task A3: Spawner `Summon` verbs (determine slot indices, then wire)

**Files:** Modify `assets/sim/vampire_survivors.sim` and `crates/dsl_compiler/tests/vampire_survivors_compile.rs`.

- [ ] **Step 1: Add the 4 tier-ramped spawn verbs (slot literals as placeholders to fix in Step 3)**

Add to `vampire_survivors.sim` (mirrors wave_defense verbs, gated on the spawner mana band + tick windows). Use `apply_ability 0/1/2/3` provisionally:

```
// Spawner self-casts tier-ramped Summon abilities every wave_period ticks.
verb SpawnSmall(self, target: Agent) =
  action SpawnSmallAction
  when (self.alive
        && target == self
        && self.mana >= config.vs.spawner_mana_min
        && self.mana <= config.vs.spawner_mana_max
        && (world.tick < config.vs.small_to_medium)
        && (world.tick % config.vs.wave_period == 0))
  apply_ability 0 by self target self
  score 500.0

verb SpawnMedium(self, target: Agent) =
  action SpawnMediumAction
  when (self.alive
        && target == self
        && self.mana >= config.vs.spawner_mana_min
        && self.mana <= config.vs.spawner_mana_max
        && (world.tick >= config.vs.small_to_medium)
        && (world.tick < config.vs.medium_to_large)
        && (world.tick % config.vs.wave_period == 0))
  apply_ability 1 by self target self
  score 500.0

verb SpawnLarge(self, target: Agent) =
  action SpawnLargeAction
  when (self.alive
        && target == self
        && self.mana >= config.vs.spawner_mana_min
        && self.mana <= config.vs.spawner_mana_max
        && (world.tick >= config.vs.medium_to_large)
        && (world.tick < config.vs.large_to_horde)
        && (world.tick % config.vs.wave_period == 0))
  apply_ability 2 by self target self
  score 500.0

verb SpawnHorde(self, target: Agent) =
  action SpawnHordeAction
  when (self.alive
        && target == self
        && self.mana >= config.vs.spawner_mana_min
        && self.mana <= config.vs.spawner_mana_max
        && (world.tick >= config.vs.large_to_horde)
        && (world.tick % config.vs.wave_period == 0))
  apply_ability 3 by self target self
  score 500.0
```

- [ ] **Step 2: Add the compile-gate test for the Summon dispatcher + chronicle emit**

Append to `crates/dsl_compiler/tests/vampire_survivors_compile.rs`:

```rust
#[test]
fn spawn_verbs_emit_summon_chronicle() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    // The apply_ability dispatcher kernel must emit an EffectSummonApplied
    // chronicle (kind 62) into the event ring. Kind id appears as `62u` in
    // the atomicStore tag write; the dispatcher kernel name contains "Spawn"
    // or "apply_ability"/"dispatch".
    let has_summon_emit = art.wgsl_files.values().any(|body| {
        body.contains("62u") && (body.contains("atomicStore(&event_ring") || body.contains("atomicAdd(&event_tail"))
    });
    assert!(
        has_summon_emit,
        "expected an EffectSummonApplied (kind 62) chronicle emit; kernels: {:?}",
        art.kernel_index,
    );
}
```

- [ ] **Step 3: Determine the real `apply_ability` slot indices and fix the verbs**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture 2>&1 | head -40`

If it PASSES with the provisional `0/1/2/3` literals AND `spawn_verbs_emit_summon_chronicle` passes, the slots happen to align — leave them. If it fails to lower, or to confirm correctness, inspect the registry order: the ability slot is assigned by `dsl_compiler`'s ability-corpus loader (alphabetical by filename within `assets/ability_test/vampire_survivors/` → `SpawnHorde, SpawnLarge, SpawnMedium, SpawnSmall`). Map each verb's `apply_ability N` to the index of the ability it should cast (SpawnSmall verb → the SpawnSmall ability's slot, etc.). Add a one-off debug print in the test if needed:

```rust
// temporary: dump registry order
eprintln!("kernels: {:?}", art.kernel_index);
```
Then set each verb's `apply_ability N` to the correct slot. Re-run until both `vampire_survivors_compiles` and `spawn_verbs_emit_summon_chronicle` pass. (If the corpus loader rejects abilities with no caster stats, mirror exactly what wave_defense's `.ability` headers carry.)

- [ ] **Step 4: Run the full compile-gate suite**

Run: `cargo test -p dsl_compiler --test vampire_survivors_compile -- --nocapture`
Expected: PASS (all prior tests + `spawn_verbs_emit_summon_chronicle`).

- [ ] **Step 5: Commit**

```bash
git add assets/sim/vampire_survivors.sim crates/dsl_compiler/tests/vampire_survivors_compile.rs
git commit -m "feat(vampire_survivors): tier-ramped Summon verbs + chronicle-emit gate (Phase A)"
```

---

## Phase B — Summon allocator

### Task B1: `SummonRecord` + pure `plan_allocations` (unit-tested, no GPU)

**Files:** Create `crates/sims/src/summon_alloc.rs`; modify `crates/sims/src/lib.rs`.

- [ ] **Step 1: Write the failing unit test**

Create `crates/sims/src/summon_alloc.rs` with the test first:

```rust
//! Host-side summon allocator: turns EffectSummonApplied (chronicle kind 62)
//! records into live agents in dead SoA slots. Split into a pure planning
//! fn (unit-tested here, no GPU) and a GPU drain (drain_summons, below).

use glam::Vec3;

/// One decoded EffectSummonApplied record (event ring kind 62).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SummonRecord {
    pub actor_slot: u32,
    pub template_hash: u32,
    pub count: u32,
    pub seq: u32,
}

/// One slot to bring alive at a position. Pure output of planning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotAssignment {
    pub slot: u32,
    pub pos: Vec3,
}

/// Pure allocation planning. Deterministic: records are processed in `seq`
/// order; dead slots (`alive[i] == 0`) are claimed in ascending index order;
/// per-spawn position = spawner pos + a per_agent_u32-seeded offset.
/// Truncates (does not panic) when the dead-slot pool is exhausted.
pub fn plan_allocations(
    alive: &[u32],
    records: &[SummonRecord],
    spawner_pos: impl Fn(u32) -> Vec3,
    seed: u64,
    tick: u64,
) -> Vec<SlotAssignment> {
    let mut sorted: Vec<SummonRecord> = records.to_vec();
    sorted.sort_by_key(|r| (r.seq, r.actor_slot));
    let mut claimed = vec![false; alive.len()];
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for rec in &sorted {
        let base = spawner_pos(rec.actor_slot);
        for _ in 0..rec.count {
            // find next dead, unclaimed slot
            while cursor < alive.len() && (alive[cursor] != 0 || claimed[cursor]) {
                cursor += 1;
            }
            if cursor >= alive.len() {
                return out; // pool exhausted — truncate
            }
            claimed[cursor] = true;
            let new_slot = cursor as u32;
            let off = engine::rng::per_agent_u32(seed, new_slot, tick, b"vs_spawn_pos");
            // map the u32 to a small planar ring offset around the spawner
            let ang = (off & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
            let rad = 1.0 + ((off >> 16) & 0xFF) as f32 / 255.0 * 3.0;
            out.push(SlotAssignment {
                slot: new_slot,
                pos: base + Vec3::new(rad * ang.cos(), rad * ang.sin(), 0.0),
            });
            cursor += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claims_dead_slots_in_order_and_truncates() {
        // slots: 0 alive (player), 1 alive (spawner), 2..6 dead
        let alive = [1u32, 1, 0, 0, 0, 0];
        let recs = [SummonRecord { actor_slot: 1, template_hash: 7, count: 3, seq: 0 }];
        let got = plan_allocations(&alive, &recs, |_| Vec3::ZERO, 0xABCD, 5);
        let slots: Vec<u32> = got.iter().map(|a| a.slot).collect();
        assert_eq!(slots, vec![2, 3, 4], "claims first 3 dead slots in order");

        // exhaustion: ask for 10 but only 4 dead
        let recs2 = [SummonRecord { actor_slot: 1, template_hash: 7, count: 10, seq: 0 }];
        let got2 = plan_allocations(&alive, &recs2, |_| Vec3::ZERO, 0xABCD, 5);
        assert_eq!(got2.len(), 4, "truncates at pool exhaustion, no panic");
    }

    #[test]
    fn deterministic_across_runs() {
        let alive = [0u32; 8];
        let recs = [SummonRecord { actor_slot: 0, template_hash: 1, count: 4, seq: 0 }];
        let a = plan_allocations(&alive, &recs, |_| Vec3::new(10.0, 0.0, 0.0), 42, 1);
        let b = plan_allocations(&alive, &recs, |_| Vec3::new(10.0, 0.0, 0.0), 42, 1);
        assert_eq!(a, b, "same inputs -> same plan (P5)");
    }
}
```

Add to `crates/sims/src/lib.rs`: `pub mod summon_alloc;`

- [ ] **Step 2: Run the test to verify it fails (then passes)**

Run: `cargo test -p sims --lib summon_alloc -- --nocapture`
Expected: compiles and the two tests PASS (the impl is included above). Reconcile two API details if they error:
- **`per_agent_u32` signature is `(world_seed: u64, agent_id: AgentId, tick: u64, purpose: &[u8])`** — `agent_id` is the `engine::AgentId` newtype, not a bare `u32`. Wrap `new_slot`: `engine::rng::per_agent_u32(seed, engine::AgentId(new_slot), tick, b"vs_spawn_pos")` (or the crate's `AgentId::from(new_slot)` if the field is private). Alternatively use the all-`u32` mirror `engine::rng::per_agent_u32_pcg(seed as u32, new_slot, tick as u32, 5)` — either gives a deterministic offset.
- **`Vec3`** — if `glam::Vec3` isn't the type the codebase uses, switch to `engine`'s re-exported `Vec3` (check `wave_defense`'s usage). Add `glam` to `crates/sims/Cargo.toml` only if the codebase genuinely uses glam (check `cargo tree -p sims | grep glam`).

- [ ] **Step 3: Commit**

```bash
git add crates/sims/src/summon_alloc.rs crates/sims/src/lib.rs crates/sims/Cargo.toml
git commit -m "feat(sims): summon_alloc::plan_allocations — pure dead-slot planner + unit tests (Phase B)"
```

### Task B2: GPU `drain_summons` (ring readback → write_buffer)

**Files:** Modify `crates/sims/src/summon_alloc.rs`.

- [ ] **Step 1: Add the GPU drain fn (validated end-to-end in Phase C)**

Append to `summon_alloc.rs`. It is generic over any runtime exposing the needed handles via a small trait so it works for vampire_survivors (and later wave_defense):

```rust
/// Minimal surface a runtime must expose for the drain. The generated
/// GeneratedRuntime has all of these as public fields.
pub struct DrainCtx<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub event_ring: &'a engine::gpu::EventRing,
    pub agent_alive_buf: &'a wgpu::Buffer,
    pub agent_pos_buf: &'a wgpu::Buffer,
    pub agent_count: u32,
    pub seed: u64,
    pub tick: u64,
}

/// Read EffectSummonApplied (kind 62) records from the ring, plan dead-slot
/// allocations, and write `alive=1` + pos for each. Returns count allocated.
/// `spawner_pos` is resolved from the read-back position buffer.
pub fn drain_summons(ctx: DrainCtx) -> usize {
    const KIND_SUMMON: u32 = 62;
    let stride = engine::gpu::EVENT_STRIDE_U32 as usize;

    // 1. read back the ring (tail_value slots)
    let n_slots = ctx.event_ring.tail_value();
    if n_slots == 0 { return 0; }
    let ring_bytes = ((n_slots as u64) * stride as u64 * 4).max(16);
    let ring_words = readback_u32(ctx.device, ctx.queue, ctx.event_ring.ring(), ring_bytes);

    // 2. decode kind-62 records (w2=actor, w3=template_hash, w4=count, w10=seq)
    let mut records = Vec::new();
    for s in 0..n_slots as usize {
        let base = s * stride;
        if ring_words.get(base).copied() == Some(KIND_SUMMON) {
            records.push(SummonRecord {
                actor_slot: ring_words[base + 2],
                template_hash: ring_words[base + 3],
                count: ring_words[base + 4],
                seq: ring_words[base + 10],
            });
        }
    }
    if records.is_empty() { return 0; }

    // 3. read alive + pos buffers
    let alive = readback_u32(ctx.device, ctx.queue, ctx.agent_alive_buf, (ctx.agent_count as u64 * 4).max(16));
    let pos_words = readback_u32(ctx.device, ctx.queue, ctx.agent_pos_buf, (ctx.agent_count as u64 * 16).max(16));
    let read_pos = |slot: u32| -> glam::Vec3 {
        let b = slot as usize * 4; // vec3 stored as 4 f32 (padded)
        glam::Vec3::new(
            f32::from_bits(pos_words[b]),
            f32::from_bits(pos_words[b + 1]),
            f32::from_bits(pos_words[b + 2]),
        )
    };

    // 4. plan + apply
    let plan = plan_allocations(&alive, &records, read_pos, ctx.seed, ctx.tick);
    if plan.is_empty() { return 0; }
    let mut alive_out = alive.clone();
    let mut pos_out = pos_words.clone();
    for a in &plan {
        alive_out[a.slot as usize] = 1;
        let b = a.slot as usize * 4;
        pos_out[b] = a.pos.x.to_bits();
        pos_out[b + 1] = a.pos.y.to_bits();
        pos_out[b + 2] = a.pos.z.to_bits();
    }
    ctx.queue.write_buffer(ctx.agent_alive_buf, 0, bytemuck::cast_slice(&alive_out));
    ctx.queue.write_buffer(ctx.agent_pos_buf, 0, bytemuck::cast_slice(&pos_out));
    plan.len()
}

fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("summon_alloc::readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("summon_alloc::rb") });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, u32>(&view).to_vec()
    };
    staging.unmap();
    out
}
```

> NOTE (enemy field pre-seeding): `drain_summons` sets only `alive` + `pos`. The enemy pool's `mana` (enemy band) and `hp` are pre-seeded once at init (Phase C, Task C2), so a newly-alive slot already has the right role/hp and the existing `ChasePlayer`/weapon gates fire on it. The `pos` vec3 storage stride is assumed 4 words (xyz + pad) — confirm against the agent_pos_buf layout in Task C3; if it is tightly-packed 3-word, adjust `read_pos`/write indices.

- [ ] **Step 2: Confirm it compiles**

Run: `cargo build -p sims 2>&1 | tail -20`
Expected: compiles (drain_summons unused warning is fine — Phase C uses it). Add `bytemuck` to `crates/sims/Cargo.toml` if not present.

- [ ] **Step 3: Commit**

```bash
git add crates/sims/src/summon_alloc.rs crates/sims/Cargo.toml
git commit -m "feat(sims): summon_alloc::drain_summons — ring readback -> alive/pos write (Phase B)"
```

---

## Phase C — Execution (build + seed + headless run)

### Task C1: Add vampire_survivors to the sims mega-crate

**Files:** Modify `crates/sims/build.rs`.

- [ ] **Step 1: Add the fixture to the `matches!()` allowlist**

In `crates/sims/build.rs`, find the `matches!(name, ...)` list (around line 41–96, contains `"wave_defense"`) and add `| "vampire_survivors"`.

- [ ] **Step 2: Build the mega-crate**

Run: `cargo build -p sims 2>&1 | tail -25`
Expected: compiles; `sims::vampire_survivors::GeneratedRuntime` now exists. If the build errors on the `.sim` (e.g., ability-corpus detection, binding-count limit), capture the error — it likely indicates a Phase A `.ability`/verb issue surfacing now that the full runtime is generated; reconcile against wave_defense.

- [ ] **Step 3: Commit**

```bash
git add crates/sims/build.rs
git commit -m "feat(sims): generate vampire_survivors runtime (Phase C)"
```

### Task C2: Initial-state seeding fn

**Files:** Create `crates/sims/src/vampire_survivors_seed.rs`; modify `crates/sims/src/lib.rs`.

- [ ] **Step 1: Write the seeding fn**

Create `crates/sims/src/vampire_survivors_seed.rs`. It writes initial SoA state via `queue.write_buffer` into the generated runtime's buffers. Player at slot 0; K spawners around the arena edge; the rest of the pool pre-seeded as dead enemies (enemy-band mana + enemy hp, `alive=0`) so the allocator only flips alive+pos.

```rust
//! Initial-state seeding for the vampire_survivors runtime.
use sims_runtime_alias::GeneratedRuntime; // see note below

pub const PLAYER_SLOT: u32 = 0;
pub const SPAWNER_COUNT: u32 = 6;
pub const SPAWNER_SLOT_START: u32 = 1;
pub const ENEMY_POOL_START: u32 = SPAWNER_SLOT_START + SPAWNER_COUNT; // 7
pub const SPAWNER_RING_RADIUS: f32 = 40.0;

/// Mana band centers (must fall inside the .sim's [min,max] windows).
const PLAYER_MANA: f32 = 1.0;   // player band [0.5,1.5]
const ENEMY_MANA: f32 = 2.0;    // enemy band  [1.5,2.5]
const SPAWNER_MANA: f32 = 3.0;  // spawner band[2.5,3.5]
const PLAYER_HP: f32 = 100.0;
const SPAWNER_HP: f32 = 1.0e6;
const ENEMY_HP: f32 = 12.0;     // == config.vs.enemy_spawn_hp

pub fn seed_initial_state(rt: &mut crate::vampire_survivors::GeneratedRuntime) {
    let n = rt.agent_count as usize;
    let mut alive = vec![0u32; n];
    let mut mana = vec![ENEMY_MANA; n];
    let mut hp = vec![ENEMY_HP; n];
    let mut pos = vec![0.0f32; n * 4]; // xyz + pad

    // Player (slot 0)
    alive[0] = 1; mana[0] = PLAYER_MANA; hp[0] = PLAYER_HP;
    // pos[0..3] = origin (already zero)

    // Spawners (slots 1..=6) at a ring around the arena edge
    for i in 0..SPAWNER_COUNT {
        let slot = (SPAWNER_SLOT_START + i) as usize;
        let ang = i as f32 / SPAWNER_COUNT as f32 * std::f32::consts::TAU;
        alive[slot] = 1; mana[slot] = SPAWNER_MANA; hp[slot] = SPAWNER_HP;
        pos[slot * 4] = SPAWNER_RING_RADIUS * ang.cos();
        pos[slot * 4 + 1] = SPAWNER_RING_RADIUS * ang.sin();
    }

    // Enemy pool (slots 7..n) left alive=0, mana=ENEMY_MANA, hp=ENEMY_HP.
    let _ = ENEMY_POOL_START;

    rt.gpu.queue.write_buffer(&rt.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    rt.gpu.queue.write_buffer(&rt.agent_mana_buf, 0, bytemuck::cast_slice(&mana));
    rt.gpu.queue.write_buffer(&rt.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
    rt.gpu.queue.write_buffer(&rt.agent_pos_buf, 0, bytemuck::cast_slice(&pos));
}
```

> NOTE: delete the bogus `use sims_runtime_alias::...` line — it is illustrative. Reference the type as `crate::vampire_survivors::GeneratedRuntime` (the mega-crate exposes each fixture as `pub mod <name>`). Confirm the buffer field names (`agent_alive_buf`, `agent_mana_buf`, `agent_hp_buf`, `agent_pos_buf`) exist on the generated struct — Task C3 Step 1 verifies this; if `agent_mana_buf` is absent (mana not emitted as an External buffer), add a trivial read of `self.mana` already exists in the .sim so it should be present, but if not, set mana via the `.sim` `init { }` block instead and drop it here.

Add to `crates/sims/src/lib.rs`: `pub mod vampire_survivors_seed;`

- [ ] **Step 2: Build**

Run: `cargo build -p sims 2>&1 | tail -25`
Expected: compiles. Fix any buffer-field-name mismatch here (this is the concrete confirmation of the field names).

- [ ] **Step 3: Commit**

```bash
git add crates/sims/src/vampire_survivors_seed.rs crates/sims/src/lib.rs
git commit -m "feat(sims): vampire_survivors initial-state seeding (Phase C)"
```

### Task C3: Headless integration test (the payoff gate)

**Files:** Create `crates/sims/tests/vampire_survivors_exec.rs`.

- [ ] **Step 1: Write the integration test**

Mirror `crates/sims/tests/navgrid_probe_pin.rs`'s GPU-skip idiom.

```rust
use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::seed_initial_state;
use sims::summon_alloc::{drain_summons, DrainCtx};

const SEED: u64 = 0x5_F00D_CAFE_0001;
const N: u32 = 512;
const TICKS: u64 = 120; // > wave_period (30) so multiple SpawnSmall waves fire

fn read_alive(rt: &mut GeneratedRuntime) -> Vec<u32> {
    // reuse the same readback idiom as summon_alloc (inline copy for the test)
    let bytes = (N as u64 * 4).max(16);
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::alive_rb"), size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&rt.agent_alive_buf, 0, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    out
}

#[test]
fn vampire_survivors_spawns_and_runs() {
    let mut rt = match GeneratedRuntime::try_new(SEED, N) {
        Some(r) => r,
        None => { eprintln!("[vampire_survivors] skip: no wgpu adapter"); return; }
    };
    seed_initial_state(&mut rt);

    let enemy_count = |alive: &[u32]| alive[7..].iter().filter(|&&a| a == 1).count();
    let alive0 = read_alive(&mut rt);
    assert_eq!(enemy_count(&alive0), 0, "no enemies before any wave");

    for _ in 0..TICKS {
        rt.step();
        drain_summons(DrainCtx {
            device: &rt.gpu.device, queue: &rt.gpu.queue, event_ring: &rt.event_ring,
            agent_alive_buf: &rt.agent_alive_buf, agent_pos_buf: &rt.agent_pos_buf,
            agent_count: rt.agent_count, seed: rt.seed, tick: rt.tick,
        });
    }

    let alive_end = read_alive(&mut rt);
    assert!(
        enemy_count(&alive_end) > 0,
        "expected DSL waves to have spawned live enemies after {TICKS} ticks; got {}",
        enemy_count(&alive_end),
    );
}
```

- [ ] **Step 2: Run it**

Run: `cargo test -p sims --test vampire_survivors_exec -- --nocapture`
Expected: PASS on a GPU/lavapipe host (or a clean skip line if no adapter). **This is the keystone validation** — it proves the DSL Summon → chronicle → host allocator → live GPU enemies path end-to-end.

If enemy_count stays 0: diagnose in order — (a) did `spawn_verbs_emit_summon_chronicle` confirm kind-62 emit? (b) add `eprintln!("recs={}", records.len())` in `drain_summons` to see if records are read; if 0, the ring `tail_value()` may not include this tick's emits before readback (try reading after an extra `step()` or check `note_emits`); (c) verify `agent_pos_buf` stride (3 vs 4 words) and the `agent_alive_buf`/`agent_mana_buf` field names. Borrow-checker note: `drain_summons` borrows several `&rt.*` fields at once — if that conflicts, destructure the needed fields before the call or add a small `rt.drain(&seed)` helper method.

- [ ] **Step 3: Commit**

```bash
git add crates/sims/tests/vampire_survivors_exec.rs
git commit -m "test(vampire_survivors): headless exec — DSL waves spawn live GPU enemies (Phase C)"
```

### Task C4: Player-kiting liveness assertion (optional hardening)

**Files:** Modify `crates/sims/tests/vampire_survivors_exec.rs`.

- [ ] **Step 1: Add a player-movement assertion**

Add a `read_pos` helper (vec3 readback) and, in the test, capture player pos at tick 0 and after TICKS; assert it moved (kiting away from the spawned swarm). Use a small epsilon:

```rust
// after seeding, before the loop: let p0 = read_player_pos(&mut rt);
// after the loop: let p1 = read_player_pos(&mut rt);
// assert!((p1 - p0).length() > 0.01, "player should kite once enemies exist");
```
Implement `read_player_pos` mirroring `read_alive` but reading `agent_pos_buf` slot 0 (first 3 f32). If the player does not move (e.g., no enemies in flee radius early), relax to assert movement OR stable — the load-bearing assertion is C3's spawn check; keep this only if it's reliably true.

- [ ] **Step 2: Run + commit**

Run: `cargo test -p sims --test vampire_survivors_exec -- --nocapture` → PASS.
```bash
git add crates/sims/tests/vampire_survivors_exec.rs
git commit -m "test(vampire_survivors): assert player kites once swarm exists (Phase C)"
```

---

## Phase D — Voxel viewer (separate plan)

### Task D1: Scope the viewer plan against live code

- [ ] **Step 1:** After Phases A–C are green, read the concrete viewer internals — `crates/viewer_runtime/src/lib.rs` (`ViewerApp`, the `VoxelBridge` refresh + readback), `crates/viewer_runtime/src/dungeon.rs` (`seed_voxel_dungeon`, `seed_topology`), and the viewer binary entry — and write a focused implementation plan `docs/superpowers/plans/2026-05-24-vampire-survivors-viewer-plan.md` covering: an open flat-arena terrain seed (replacing dungeon rooms), a vampire_survivors `ViewerApp` path reusing `seed_initial_state` + calling `drain_summons` each tick, and a `VoxelBridge` refresh painting player/enemy/spawner palette colors. This is split out because its exact `VoxelBridge`/`ViewerApp` API surface must be read first to avoid speculative tasks (spec §2, §4: ~2–4 days plumbing).

---

## Final verification (Phases A–C)

- [ ] `cargo test -p dsl_compiler --test vampire_survivors_compile` → all PASS (incl. `spawn_verbs_emit_summon_chronicle`).
- [ ] `cargo test -p sims --lib summon_alloc` → PASS (pure planner).
- [ ] `cargo test -p sims --test vampire_survivors_exec` → PASS or clean GPU-skip; on a GPU host, enemies spawn (count > 0).
- [ ] `cargo build -p sims` → no errors; no new warnings introduced by these files.
- [ ] Ledger/spec: confirm the spec's Phase B "validate on wave_defense" note is reconciled — this plan validates the planner via unit test + the GPU path via vampire_survivors directly (cleaner, since wave_defense lost its seeding). Note this deviation in the Phase C commit message or a spec footnote.
