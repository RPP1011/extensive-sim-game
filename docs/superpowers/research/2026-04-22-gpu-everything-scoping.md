# GPU-everything scoping report

**Status:** research, not a spec
**Date:** 2026-04-22
**Predecessor:** [Engine spec](../../spec/engine.md) §9 (resident cascade)
**Branch:** `world-sim-bench`

## Purpose

`GpuBackend::step_batch(n)` now runs N ticks GPU-resident with one submit + one
poll. The only CPU work that still runs inside each batch tick is:

| # | Item                                          | User verdict    |
|---|-----------------------------------------------|-----------------|
| 1 | `state.tick` wrapping_add (lib.rs:1077)       | migrate         |
| 2 | `state.rng_state` seeding                     | migrate         |
| 3 | Initial agent upload in `ensure_resident_init`| **stay on CPU** |
| 4 | Cascade rule handlers not folded into DSL     | migrate         |
| 5 | Chronicle narrative text (strings)            | **stay on CPU** |
| 6 | Ability evaluation beyond DSL physics         | migrate         |

This report scopes items **1, 2, 4, 6** — the work that still needs doing.

---

## Item 1 — `state.tick` CPU advance

### A. What's there today

- `crates/engine_gpu/src/lib.rs:1077` — `state.tick = state.tick.wrapping_add(1)`
  inside the per-tick `for` loop inside `step_batch`.
- Same file, `lib.rs:1078` — `self.latest_recorded_tick = state.tick` (used by
  `snapshot()` to stamp the observer-visible tick).
- Read sites inside `step_batch`: `scoring_kernel.refresh_tick_cfg_for_resident`
  (lib.rs:968) uploads `state.tick` into the scoring uniform each tick;
  `cascade_resident::run_cascade_resident` packs `state.tick` into
  `PhysicsCfg.tick` (cascade_resident.rs:625).
- Consumers of the tick value on the GPU side: physics WGSL emitter's
  `wgsl_world_tick` alias (physics.rs:585-600), used by rules like
  `Event::AgentDied { tick: state.tick }` event emission.

**Correctness properties:** monotone, u32 wrapping, same value observed
consistently by every kernel dispatched within one logical tick. Chronicle /
event records embed this tick.

### B. Migration sketch

A 4-byte `tick_counter_buf: array<atomic<u32>, 1>` lives on GPU. The existing
"seed indirect" kernel (`SEED_INDIRECT_WGSL`, cascade_resident.rs:219) already
has a 1-thread shape per tick — trivial to extend to also do
`atomicAdd(&tick_counter[0], 1u)` at the top of the tick. Every kernel that
today reads `cfg.tick` reads `tick_counter[0]` instead.

Because `PhysicsCfg.tick` / the scoring cfg uniform are re-uploaded per tick,
the simplest swap is to *replace* the CPU `queue.write_buffer(cfg, ...)` with a
GPU-side `copy_buffer_to_buffer(tick_counter, cfg, dst_offset=0, 4)` injected
into the encoder. One-line change per kernel (scoring, movement, apply_actions,
physics).

Alternatively, add a single `ResidentGlobals` storage buffer with `{tick,
seed_hi, seed_lo}` and have every resident kernel read from it instead of the
per-kernel uniform.

**WGSL sketch (increment):**

```wgsl
@group(0) @binding(0) var<storage, read_write> globals: array<atomic<u32>>;
@compute @workgroup_size(1) fn advance_tick() {
    atomicAdd(&globals[0], 1u);
}
```

**Buffer layout:** 16 B `ResidentGlobals { tick: u32, seed: vec2<u32>, _pad: u32 }`.

**DSL compiler impact:** physics/mask/scoring emitters all spell `cfg.tick` as
the source of `wgsl_world_tick` today. Option A (copy into existing uniform)
requires zero compiler changes. Option B (new shared storage buffer) would
require rewiring every emitted `cfg.tick` reference — engine_gpu-side patch
plus a thin compiler aliasing shim.

Recommendation: **Option A, copy_buffer_to_buffer**. Zero compiler diff,
trivially reviewable.

### C. Dependencies

Standalone. Does not require any other migration. Blocks nothing.

### D. Risk / cost

- **Complexity:** Small (<100 LOC).
- **Blast radius:** lib.rs::step_batch, cascade_resident, and the existing
  `SeedIndirectKernel` (~5 files touched).
- **Tests:** `async_smoke.rs` already asserts `snapshot.tick == N`. Extend
  with an N=1024 variant to catch u32 wrap-around regressions. No new
  infrastructure.
- **Hazards:** `snapshot()` reads `self.latest_recorded_tick`, which stays
  CPU-side. Must keep that in sync — easiest via the existing
  copy_buffer_to_buffer path into the snapshot staging struct (cheap, already
  readback-shaped).

---

## Item 2 — `state.rng_state` CPU seeding

### A. What's there today

- `crates/engine/src/rng.rs` — the only RNG surface. Two exports:
  - `WorldRng::next_u32()` (PCG-XSH-RR, stateful — **currently unused from
    inside the tick loop**; grep confirms zero callers inside `engine::step`
    and `engine_gpu::` crates).
  - `per_agent_u32(world_seed, agent_id, tick, purpose)` — pure hash,
    **the only RNG called inside a tick**.
- `SimState.seed: u64` (state/mod.rs:33) — a constant set at `SimState::new`.
  Never mutated after construction.
- Consumers inside the tick:
  - `crates/engine/src/step.rs:355` — `shuffle_order_into` for the
    deterministic Fisher-Yates in `apply_actions`.
  - The WGSL physics emitter can, in principle, emit `per_agent_u32`
    reductions (grep shows only the host side today).

**Correctness properties:** deterministic for
`(world_seed, agent_id, tick, purpose)`; ahash with fixed seeds; schema-hash
pinned constants.

### B. Migration sketch

The user's framing ("CPU-resident seed, consumed by several kernels")
anticipates a `WorldRng`-style stateful counter. **That's not how engine RNG
works today** — `per_agent_u32` is a pure function of a constant `world_seed`
plus the per-tick tick counter that's already being migrated in item 1.

`world_seed` is immutable after `ensure_resident_init` uploads it. Include
`seed_hi: u32` + `seed_lo: u32` in the `ResidentGlobals` buffer (or in the
existing `PhysicsCfg`) and the CPU↔GPU RNG boundary evaporates — no kernel
has to "consume the seed" beyond reading a u64 constant.

For kernels that need `per_agent_u32`-style hashing on GPU:

**WGSL sketch (aHash port):** aHash is folded Fibonacci + AES-round mixing.
Not practical in WGSL (no AES). Alternatives:
1. Replace with a WGSL-friendly PRF (xxHash64 / wyhash / FxHash). Rebuild the
   deterministic shuffle on GPU using the new PRF. **Acceptable because the
   batch path already has a documented non-determinism disclaimer
   vs. the sync path** (spec §Non-goals). Sync path keeps ahash; batch path
   uses GPU-PRF.
2. Do nothing and let the batch path skip shuffling. Since apply_actions on
   GPU already processes one-thread-per-agent with no ordering semantics
   (atomic event emission), there is **no first-mover-bias to prevent** —
   the shuffle exists because the CPU apply pass is serial.

Recommendation: **option 2** — the shuffle is a CPU-serialisation artifact.
GPU dispatch is parallel by construction; no shuffle needed. Just upload
`world_seed` once in `ensure_resident_init` and never touch it again.

### C. Dependencies

Sequential on item 1 only if the chosen encoding is "extend
`ResidentGlobals`". Otherwise independent.

### D. Risk / cost

- **Complexity:** Small (<50 LOC).
- **Blast radius:** 2 files — the upload path in `ensure_resident_init` and
  whichever kernel needs it (currently none inside step_batch).
- **Tests:** no behavioural test exists for this today; add a fixture that
  runs step_batch at two different seeds and asserts divergent output as a
  sanity check.
- **Hazards:** if a future kernel *does* need a GPU-side `per_agent_u32`
  replacement, the WGSL PRF choice is a schema-hash concern — document the
  batch-path PRF in the non-determinism disclaimer.

---

## Item 4 — Cascade rule handlers + `cold_state_replay`

**This is the big one.** ~80% of the remaining CPU tick work.

### A. What's there today

- `crates/engine_gpu/src/cascade.rs:576-659` — `cold_state_replay(state,
  events, events_slice)`. Iterates the tick's cascade output and replays
  **11 rule handlers** the GPU physics kernel stubs:

  | # | Handler                | Trigger event            | Side effect                              |
  |---|------------------------|--------------------------|------------------------------------------|
  | 1 | `chronicle_attack`     | `AgentAttacked`          | push `ChronicleEntry` (narrative id 2)   |
  | 2 | `chronicle_wound`      | `AgentAttacked` (low hp) | push `ChronicleEntry` (narrative id 8)   |
  | 3 | `chronicle_death`      | `AgentDied`              | push `ChronicleEntry` (narrative id 1)   |
  | 4 | `chronicle_flee`       | `AgentFled`              | push `ChronicleEntry` (narrative id 7)   |
  | 5 | `chronicle_engagement` | `EngagementCommitted`    | push `ChronicleEntry` (narrative id 3)   |
  | 6 | `chronicle_break`      | `EngagementBroken`       | push `ChronicleEntry` (narrative id 4)   |
  | 7 | `chronicle_rout`       | `FearSpread`             | push `ChronicleEntry` (narrative id 6)   |
  | 8 | `chronicle_rally`      | `RallyCall`              | push `ChronicleEntry` (narrative id 5)   |
  | 9 | `transfer_gold`        | `EffectGoldTransfer`     | mutate `cold_inventory[slot].gold: i64`  |
  |10 | `modify_standing`      | `EffectStandingDelta`    | mutate `cold_standing: SparseStandings`  |
  |11 | `record_memory`        | `RecordMemory`           | push into `cold_memory[slot]` smallvec   |

- Generated code: `crates/engine/src/generated/physics/{chronicle_*,
  transfer_gold, modify_standing, record_memory}.rs` — 8 chronicle files
  ×17 LOC + 3 non-chronicle files ×~25 LOC each. Total ~200 LOC of CPU
  handler logic.
- Called from: **nowhere today under `step_batch`** — the resident cascade
  driver (`cascade_resident::run_cascade_resident`) **silently skips these**
  (as the user called out). Sync path still calls `cold_state_replay` via
  the full-tick integration in `GpuBackend::step`.
- Dispatcher machinery that the sync path goes through, but which the batch
  path now bypasses:
  - `CascadeRegistry::dispatch` — `cascade/dispatch.rs:89-104`
  - `CascadeRegistry::run_fixed_point_tel` — `cascade/dispatch.rs:126-167`
  - DSL-generated per-kind dispatchers — `generated/physics/mod.rs:33-286`

**Correctness properties:**
- Chronicle ordering matches CPU path (rule fires in cascade iteration order).
- Gold transfers are wrapping_add on `i64` — overflow allowed, no clamp.
- Standing is symmetric `i16` clamped to `[-1000, 1000]`.
- Memory ring is a per-slot `SmallVec<[MemoryEvent; 64]>` — 64-entry max,
  append discards oldest (check `push_agent_memory` for exact policy).

### B. Migration sketch

**Split into three groups by GPU ergonomics:**

#### Group 4a — chronicle rules (8 handlers)

User says item 5 (chronicle narrative text) can stay on CPU. But the
`chronicle_*` handlers don't format text — they push one `ChronicleEntry
{ template_id, agent, target, tick }` event. Text formatting happens
downstream at drain time by whoever reads the chronicle ring.

**These 8 rules are WGSL-trivial** — each one pushes an `EventRecord` with a
fixed `template_id` constant and three fields plucked from the triggering
event. Dispatch already lives in the physics kernel's event-driven emission
loop. What's stubbed is just the chronicle-ring emit call. The
`chronicle_ring` exists (created in `CascadeResidentCtx::new`,
cascade_resident.rs:449); the physics WGSL emitter already takes a dedicated
chronicle ring binding (physics.rs:399 `build_physics_shader_with_chronicle`).

**Work required:** flip 8 stubs from no-op to
`gpu_emit_chronicle(...)` in the physics WGSL emitter. The DSL compiler
already emits these as `EventKind::ChronicleEntry` pushes in the CPU Rust
output — need an equivalent chronicle-ring emit in the WGSL emitter
(`crates/dsl_compiler/src/emit_physics_wgsl.rs` — not yet verified to accept
chronicle emits; see open question below).

This is a **DSL compiler task**, not pure engine_gpu.

#### Group 4b — `transfer_gold` + `modify_standing`

Both mutate cold storage not yet on GPU:
- `cold_inventory: Vec<Inventory>` — `Inventory` is 24 bytes
  (`i64 gold + [u16; 8] commodities`). Trivially Pod-convertible to
  `array<GpuInventory>` on GPU. Add a new resident buffer sized
  `agent_cap * 24` B.
- `cold_standing: SparseStandings` — BTreeMap keyed by ordered
  `(AgentId, AgentId)`. **Not GPU-friendly.** Two choices:
  1. Dense i16 matrix `agent_cap × agent_cap` — at N=100k that's 20 GB.
     **Rules out dense.**
  2. Fixed-size per-agent neighbour list: `array<StandingEntry, 32>` per agent,
     LRU-eviction on overflow. 32 × 4B = 128 B per agent, ~12 MB at N=100k.
     Covers typical interaction locality; overflow is a silent-drop
     correctness regression vs. the CPU path.
  3. GPU hash table keyed by `(low_id, high_id)`. Classic open-addressing
     via atomic CAS. ~100-200 LOC of WGSL but well-understood pattern.

Recommendation: **option 3** — hash table. Accepts unbounded pairs (the
actual design property of `SparseStandings`), deterministic modulo
GPU-fold-order which is already flagged as non-deterministic in the batch
path.

**WGSL sketch (standing hash):**

```wgsl
// key = (min(a,b) << 32) | max(a,b)
struct StandingSlot { key: u32, key_hi: u32, value: i32, state: u32, };
@group(0) @binding(N) var<storage, read_write> standing: array<StandingSlot>;

fn standing_adjust(a: u32, b: u32, delta: i32) -> i32 {
    let lo = min(a, b);
    let hi = max(a, b);
    var idx = hash2(lo, hi) & (CAP - 1u);
    loop {
        let s = atomicLoad(&standing[idx].state);
        if (s == EMPTY) {
            // CAS insert
            let old = atomicCompareExchangeWeak(
                &standing[idx].state, EMPTY, OCCUPIED).old_value;
            if (old == EMPTY) {
                standing[idx].key = lo; standing[idx].key_hi = hi;
                atomicStore(&standing[idx].value, clamp(delta, -1000, 1000));
                return delta;
            }
        } else if (standing[idx].key == lo && standing[idx].key_hi == hi) {
            let prev = atomicAdd(&standing[idx].value, delta);
            return clamp(prev + delta, -1000, 1000);
        }
        idx = (idx + 1u) & (CAP - 1u);
    }
}
```

`transfer_gold` is simpler — pure per-slot write. Two
`atomicAdd<i64>`-equivalent-via-pair-of-u32 calls on
`inventory[from].gold` / `inventory[to].gold`. WGSL doesn't have
`atomic<i64>`, so emulate via pair-of-atomic-u32 with carry, or accept that
gold transfers happen one-at-a-time in the current cascade dispatch
(atomicAdd on u32 low half + non-atomic write to high half — but requires
correctness argument about concurrent transfers).

Realistic answer: most transfers are bounded to a few ticks apart; emulate
i64 with one atomic<u32> + one non-atomic u32, and acknowledge a tiny
non-deterministic race window that's unlikely to hit in practice. Or keep
gold on CPU because nobody cares about steady-state N-body gold physics.

#### Group 4c — `record_memory`

Per-agent `SmallVec<[MemoryEvent; 64]>`. On GPU, a ring buffer per agent:
`agent_cap × 64 × sizeof(GpuMemoryEvent)`. `MemoryEvent` is 24 B
(source:u32, kind:u8, payload:u64, confidence:u8, tick:u32 + padding) →
~150 MB at N=100k. Feasible but chunky.

Append via atomic write-index per agent (u32 counter, mod 64 when
wrapping). WGSL trivial — same pattern as the existing event ring tail.

Because memory is per-agent bounded, no need for hash-table
fancy-footwork. The per-agent write-index can be stored alongside the
agent SoA or in a separate `agent_cap × u32` buffer.

**DSL compiler impact:** The `emit_physics_wgsl` module needs to grow emit
paths for (a) chronicle entries into the chronicle ring, (b) standing
table adjust, (c) inventory gold add, (d) memory ring push. Four new
"namespace" calls in the WGSL emitter mirroring the Rust ones. **This is
real DSL compiler work**, not just engine_gpu.

### C. Dependencies

- 4a (chronicle stubs) depends only on DSL compiler accepting the emit.
  Independent of other migrations.
- 4b (gold + standing) independent of 4a, 4c.
- 4c (memory) independent of 4a, 4b.
- **None of 4a-c depend on items 1, 2, or 6.**
- The whole of item 4 should land **before** item 6 (see below) because
  ability cast effects can fire `transfer_gold` / `modify_standing` /
  `record_memory` via `EffectOp::TransferGold` etc. Skipping those while
  ability evaluation goes GPU would silently drop side-effects.

### D. Risk / cost

- **Complexity:**
  - 4a: Small (DSL emit + 8 stubs → real chronicle pushes; trust the
    existing chronicle ring plumbing).
  - 4b: Medium (hash table is ~150 LOC WGSL + test infra; gold is annoying
    but small).
  - 4c: Medium (150 MB buffer decision is a memory-budget question).
- **Blast radius:**
  - DSL compiler (`emit_physics_wgsl.rs`) gains 4 new emit paths.
  - engine_gpu gets 3 new storage buffers (inventory, standing hash,
    memory ring) and their bind-group plumbing in the physics kernel.
  - `cold_state_replay` becomes dead code under the batch path; keep it
    for sync-path parity.
- **Tests:**
  - Existing `parity_with_cpu` covers chronicle order (sync path only —
    do not extend to batch path).
  - New: `standing_hash_stress` — N concurrent `modify_standing` events
    against the same pair, assert post-state equals CPU reference
    (±fold-order non-determinism).
  - New: `gold_transfer_parity` — N transfer_gold events, compare sum
    across all agents (should conserve, modulo wrap).
  - New: `memory_ring_overflow` — push 65 memories to one agent, assert
    oldest drops.
- **Hazards:**
  - WGSL has no i64 atomics. Gold is i64. **Open question.**
  - Standing's "silent drop on hash-table overflow" is a correctness
    regression vs. CPU's unbounded BTreeMap. Need to pick a capacity
    policy + surface overflows via a GPU flag that `snapshot()` reads.
  - The 150 MB memory buffer at N=100k may force a config-driven cap
    (e.g. 16 entries not 64).
  - Chronicle emission in WGSL needs a `template_id` constant, which the
    current DSL physics emitter doesn't thread through — trace the
    existing chronicle-ring path in physics.rs and confirm.

---

## Item 6 — Ability evaluation beyond DSL-compiled physics

### A. What's there today

The engine already supports the full ability system in DSL-compiled physics.
`generated/physics/cast.rs` (129 LOC) pattern-matches `EffectOp` and emits
child events; this is already compiled into the GPU physics kernel via the
`emit_physics_wgsl` path (physics.rs:21-22 confirms `cast` is FULL on GPU).

**So what "ability evaluation beyond DSL" actually remains CPU-side?** Two
things:

1. **The `Cast` mask** (`crates/engine/src/generated/mask/cast.rs`, 28 LOC)
   — predicate that decides whether agent `self_id` *can* cast
   `ability`. The GPU fused mask kernel explicitly **skips Cast** (see
   `engine_gpu/src/mask.rs:25-43`) because:
   - Cast's action head takes a non-Agent parameter (`AbilityId`).
   - Predicate reads `views::is_stunned(self)`.
   - Predicate reads `abilities.known(self, ability)` — requires a
     per-agent-per-ability known bitset.
   - Predicate reads `abilities.cooldown_ready(self, ability)` — requires
     per-agent-per-ability cooldown state.

2. **Action scoring for Cast** — the scoring kernel today does not emit
   `MicroKind::Cast` because the Cast mask never flags true on GPU. So
   no Cast actions reach `apply_actions`. Cast effects therefore never
   fire from the action path — they only fire when some other GPU rule
   emits an `AgentCast` event directly (which none of them do today;
   see `generated/physics/` grep).

**Net:** in the GPU batch path, *no abilities ever cast*. Combat is attack-
damage only. This is a silent correctness gap under the batch path.

### B. Migration sketch

**Required storage** (not yet on GPU):
- `abilities.known[agent][ability]` bit — a per-agent `MAX_ABILITIES / 32`
  u32 array, or a packed `array<u32>` keyed by `(agent_slot *
  max_abilities_words + word_idx)`.
- `abilities.cooldown_next_ready[agent][ability]` — a per-agent-per-ability
  u32 tick. At MAX_ABILITIES=64 and agent_cap=100k, that's 25 MB. The GPU
  physics kernel already has a packed `PackedAbilityRegistry`
  (physics.rs:~60) with `known[]` and `cooldown[]` arrays — verify
  whether those are per-ability or per-agent-per-ability (grep indicates
  **per-ability**, i.e. ability template cooldown, not per-caster
  remaining cooldown). Per-caster cooldown *is* stored as
  `GpuAgentSlot.cooldown_next_ready_tick` but that's a single u32 (one
  cooldown slot per agent, ability-agnostic) — insufficient for a
  multi-ability caster.

**WGSL sketch (Cast mask):**

```wgsl
fn mask_cast(self_slot: u32, ab: u32) -> bool {
    if (agents[self_slot].alive == 0u) { return false; }
    if (is_stunned(self_slot)) { return false; }
    if (!abilities_known(self_slot, ab)) { return false; }
    if (cfg.tick < ability_cooldown_ready(self_slot, ab)) { return false; }
    if (agents[self_slot].engaged_with != 0u) { return false; }
    return true;
}
```

Consumed from the scoring kernel — each candidate Cast action is
`(target, ability)` pair; scoring needs a target-selection loop per
ability per agent. Expands the scoring action space from `NUM_MICROS = 5`
(Attack/Move/Flee/Eat/Drink/Rest/Hold) to `7 + MAX_ABILITIES` per agent.
**Significant scoring rework.**

**Buffer layout additions:**
- `abilities_known_bits: array<u32>` (agent_cap × ceil(MAX_ABILITIES / 32))
- `ability_cooldown_next_ready: array<u32>` (agent_cap × MAX_ABILITIES) —
  25 MB at N=100k, MAX_ABILITIES=64. Acceptable.

**Bind group:** one new group (scoring + physics both need cooldown +
known; physics already has ability-template `known[]` for the Cast effect
fanout but does not have per-caster cooldown — needs adding to the
physics kernel's bind group).

**DSL compiler impact:** significant.
- `emit_scoring_wgsl` must learn to emit the Cast action head with its
  `AbilityId` parameter.
- `emit_mask_wgsl` must learn to emit a parametric mask (currently gated
  on "Cast is skipped").
- Either (a) pre-flatten Cast into `MAX_ABILITIES` distinct per-ability
  actions at compile time, or (b) emit a runtime loop in the scoring
  kernel over all known abilities for each caster. Option (a) is cleaner
  but bloats the mask bitmap buffer by MAX_ABILITIES× (from 1 Cast mask
  to 64). Option (b) is uglier WGSL but cheaper memory.

### C. Dependencies

- Depends on **item 4** (cascade registry): casts fire `EffectOp::TransferGold`
  / `ModifyStanding` / `CastAbility`, whose handlers are the 11 stub
  rules. Casting on GPU before those rules land emits events that nothing
  will process.
- Depends on **cooldown storage landing first** (no dependency on items 1,
  2, 4 in a strict sense — cooldown storage could land as part of item 6
  — but it's a prerequisite).
- Independent of items 1 and 2.

### D. Risk / cost

- **Complexity:** Large. Touches 3 DSL emitters + 3 engine_gpu kernels +
  2 new storage buffers + the scoring action-space widening.
- **Blast radius:**
  - `crates/dsl_compiler/src/{emit_mask_wgsl, emit_scoring_wgsl,
    emit_physics_wgsl}.rs` — new emit paths for parametric masks and
    cooldown reads.
  - `engine_gpu/src/{mask, scoring, apply_actions, physics}.rs` — 4 files
    gain new bindings and cast dispatch.
  - `GpuAgentSlot` may need to absorb the per-ability cooldown cursor
    or it lives in a side buffer — that decision fans out across every
    kernel that reads agent slots.
- **Tests:**
  - Existing `parity_with_cpu` runs a cast-using scenario to detect
    byte-exact divergence on the sync path — stays load-bearing.
  - New `batch_cast_smoke`: an N=64 fixture with a warlock-archetype
    agent casting a damage-over-time ability; assert at least one
    `EffectDamageApplied` reaches the physics ring under `step_batch`.
  - New `cooldown_gate`: an agent with a 100-tick-cooldown ability;
    `step_batch(200)` should see ≥ 2 casts via snapshot events.
- **Hazards:**
  - `MAX_ABILITIES` is compile-time in WGSL. If it differs from the
    engine-side `AbilityRegistry::len()` the emit breaks. Today
    `MAX_ABILITIES=64` is defined in physics.rs; registry growth past 64
    is a build-time assert failure.
  - The scoring kernel already allocates score outputs of a fixed size;
    widening NUM_ACTIONS changes the scoring output stride and cascades
    into apply_actions.

---

## Cross-item dependency graph

```
item 3 (startup, CPU) ──┐
                        │
item 1 (tick counter) ──┼──▶ blocks nothing, can land anytime
                        │
item 2 (rng seed)    ───┘

item 4a (chronicle stubs)       ┐
item 4b (gold + standing)       ├─▶ item 6 (ability evaluation)
item 4c (record_memory)         ┘

item 5 (narrative text, CPU) — stays off-path, no effect on ordering
```

No two migrations share a file set large enough to serialise. The only
strict prerequisite is **item 4 before item 6** — abilities cast effects
that fire the 11 stub rules.

## Recommended order

1. **Item 1** (tick counter) — 1 hour. Unblocks deterministic tick stamps
   across all migrated kernels. Lowest risk, smallest diff, highest
   cross-item leverage. **Do first.**

2. **Item 2** (rng seed upload) — 1 hour. Trivial piggyback on item 1's
   `ResidentGlobals` buffer. Do in parallel with item 1 if a second hand
   is available.

3. **Item 4a** (chronicle stubs) — 1-2 days. Pure DSL compiler task:
   make `emit_physics_wgsl` emit `ChronicleEntry` pushes into the
   existing chronicle ring. Already-wired infrastructure (chronicle ring
   exists, physics kernel has the binding). Payoff: chronicle-visible
   batch path narratives.

4. **Item 4c** (record_memory) — 1 day. Self-contained per-agent ring
   buffer. Memory-budget question to resolve (64 vs 16 entries per
   agent). Can parallelise with 4a — different DSL emit path, different
   buffer.

5. **Item 4b** (gold + standing) — 3-5 days. GPU hash table is the
   hardest engineering in the batch. i64 gold is an open question (see
   below). This is the longest single piece of work after item 6.

6. **Item 6** (ability evaluation) — 1-2 weeks. Large cross-cutting diff.
   Scoring action-space widening + 3 DSL emitter changes + 2 new
   storage buffers + a parametric mask emit. Land last because it
   depends on the item 4 side effects firing correctly.

**Parallelisable pairs:**
- (1, 2) — same file, one-commit diff.
- (4a, 4c) — different DSL emit paths.
- (4b) must serialise after 4a/4c because its bind group decisions
  cascade through the physics kernel.
- (6) must serialise after all of item 4.

---

## Open questions

1. **i64 gold on GPU.** WGSL has no `atomic<i64>`. Options: (a) emulate
   via atomic u32 pair with carry — painful and racy; (b) keep gold on
   CPU and replay the gold events in `cold_state_replay` even under the
   batch path — requires a readback of `EffectGoldTransfer` events from
   the physics ring, which is cheap but violates "no per-tick readback";
   (c) restrict gold to i32 and declare a schema-hash bump — simplest,
   probably acceptable since the current balance clamps keep gold
   well inside i32. **Recommend (c), ask user.**

2. **Standing table sizing.** `SparseStandings` is unbounded in practice.
   GPU hash table needs a compile-time capacity. At N=100k agents with
   typical interaction locality, how many distinct pairs hold non-zero
   standing? Need a measurement from an existing long-run.

3. **Memory ring capacity.** 64 entries × 24 B × N=100k = 150 MB. Is 16
   entries acceptable? The CPU path's `SmallVec<[_; 64]>` is a capacity,
   not a frequency — overflow policy is `discard oldest` (verify in
   `push_agent_memory`). Shrinking to 16 is a schema-hash-visible
   behaviour change.

4. **Chronicle narrative text — confirmed CPU-only?** The user said "can
   stay on CPU," but item 4a migrates the *emission* of
   `ChronicleEntry` events, not the *formatting* of narrative strings.
   The split is clean: GPU emits `ChronicleEntry { template_id, agent,
   target, tick }`, CPU drains the chronicle ring at its leisure and
   formats strings. Confirm this is the intended division.

5. **Ability mask parametric emit strategy.** Flatten to MAX_ABILITIES
   bitmaps vs. a runtime loop inside the scoring kernel — both have
   sharp edges. Flatten is cleaner but bloats mask memory; loop is
   cheaper but complicates WGSL emit. Pick during item-6 spec brainstorm.

6. **Does the batch path need a shuffle at all?** Item 2 recommends
   skipping it. Worth explicit confirmation from someone who has
   verified first-mover-bias isn't creeping back in under GPU parallel
   apply — i.e., that `atomicAdd` on the event ring tail really does
   randomise effective order, not just per-workgroup bucketing.

7. **Sync-path parity test under item 6.** Existing `parity_with_cpu`
   runs the sync path only (spec §Testing says "no parity test between
   sync and batch"). Once abilities fire on the batch path, should we
   add a *statistical* smoke test (e.g. cast count within ±25%)? Open.

8. **Retire `cold_state_replay` entirely?** Once item 4 lands under
   `step_batch`, `cold_state_replay` is dead under the batch path but
   still load-bearing on the sync path. Do we delete it (killing the
   sync path's non-chronicle side effects) or keep both paths? The spec
   says the sync path "stays load-bearing for parity tests" so probably
   keep — but document the split explicitly.
