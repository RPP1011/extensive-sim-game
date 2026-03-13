# GPU Compute Sim for Self-Play Pretraining

## Goal

Run 4096+ simplified combat simulations in parallel on GPU via wgpu compute
shaders. Target: 10-100× throughput over CPU for RL episode generation.

## Design Principles

1. **Fixed-size everything** — no dynamic allocation on GPU
2. **Flat data** — structure-of-arrays, no pointers/indirection
3. **Minimal branching** — collapse CC to bitmask checks, not if/else trees
4. **Good enough fidelity** — policies trained here transfer to full sim via fine-tuning
5. **Deterministic** — per-sim RNG stream (PCG hash)

## What We Keep (vs Full Sim)

| Mechanic | Full Sim | GPU Sim |
|----------|----------|---------|
| Units | Vec (unbounded) | Fixed 8 slots (4v4) |
| HP/damage/armor/MR | Yes | Yes |
| Movement | A* pathfinding | Direct move-toward (no grid) |
| Attack + cast time | Yes | Yes (simplified) |
| Abilities | 8 slots, complex delivery | 4 slots, instant delivery only |
| Damage types (phys/magic/true) | Yes | Yes |
| Armor/MR reduction | `r/(100+r)` | Same |
| Cooldowns | 4 base + per-ability | Same |
| Status effects | 33 variants, Vec | 8 types, fixed 4 slots per unit |
| DoT/HoT | Interval-based | Per-tick flat |
| Shield | Yes | Yes |
| Buff/Debuff | String-keyed stat map | Fixed: atk_mult, def_mult, spd_mult |
| CC | stun/slow/root/silence/fear/taunt/etc | stun/slow/root/silence only |
| Projectiles | Flight time + collision | Cut (instant hit) |
| Zones/traps | Persistent ground effects | Cut |
| Channels/tethers | Duration-based | Cut (fold into cast time) |
| Passive triggers | Recursive, 3-deep | Cut |
| Pathfinding/grid | A* on GridNav | Cut (open field, euclidean) |
| Summoning | Dynamic spawn | Cut |
| Event log | Full event stream | Cut (just terminal outcome + HP) |
| RNG | Shared seeded u64 | Per-sim PCG stream |

## What Stays for Policy Transfer

The action space and observation encoding stay identical to the full sim's RL
pipeline. A policy trained on GPU sim sees the same entity tokens and outputs
the same action distribution — it just trains on a simplified world model.

## Data Layout (Structure of Arrays)

All sim state lives in GPU storage buffers. One workgroup = one sim instance.

### Per-Sim Constants (read-only after init)

```
// Buffer: sim_config [N_SIMS]
struct SimConfig {
    max_ticks: u32,
    n_heroes: u32,      // 1-4
    n_enemies: u32,     // 1-4
    _pad: u32,
}
```

### Per-Unit State (read-write)

```
// Buffer: units [N_SIMS × 8]
// SoA layout: each field is a separate buffer for coalesced access

struct UnitState {
    // Vitals (8 floats)
    hp: f32,
    max_hp: f32,
    shield: f32,
    resource: f32,
    max_resource: f32,
    resource_regen: f32,
    alive: f32,           // 1.0 or 0.0
    team: f32,            // 0.0 = hero, 1.0 = enemy

    // Position (4 floats)
    pos_x: f32,
    pos_y: f32,
    move_speed: f32,
    _pos_pad: f32,

    // Combat stats (8 floats)
    attack_damage: f32,
    ability_damage: f32,
    heal_amount: f32,
    armor: f32,
    magic_resist: f32,
    attack_range: f32,
    ability_range: f32,
    heal_range: f32,

    // Cooldowns in ticks (8 u32 → packed as f32)
    attack_cd: f32,
    attack_cd_max: f32,
    ability_cd: f32,
    ability_cd_max: f32,
    heal_cd: f32,
    heal_cd_max: f32,
    control_cd: f32,
    control_cd_max: f32,

    // Cast state (4 floats)
    cast_remaining: f32,  // >0 means currently casting
    cast_type: f32,       // 0=none, 1=attack, 2=ability, 3=heal, 4=control, 5-8=ability_slot
    cast_target: f32,     // target unit index
    _cast_pad: f32,

    // CC state — bitmask + timers (4 floats)
    cc_flags: u32,        // bit0=stun, bit1=slow, bit2=root, bit3=silence
    cc_stun_ticks: f32,
    cc_slow_ticks: f32,
    cc_slow_factor: f32,

    cc_root_ticks: f32,
    cc_silence_ticks: f32,
    _cc_pad0: f32,
    _cc_pad1: f32,

    // Status effect slots [4] (DoT, HoT, buff, debuff)
    // Each: (type, value, ticks_remaining, _pad)
    status_0: vec4<f32>,  // DoT: (dmg_per_tick, 0, ticks_left, 0)
    status_1: vec4<f32>,  // HoT: (heal_per_tick, 0, ticks_left, 0)
    status_2: vec4<f32>,  // Buff: (atk_mult, def_mult, spd_mult, ticks_left)
    status_3: vec4<f32>,  // Debuff: (atk_mult, def_mult, spd_mult, ticks_left)
}
// Total: 60 floats = 240 bytes per unit
// Per sim: 8 × 240 = 1,920 bytes
// 4096 sims: 7.68 MB
```

### Per-Unit Ability Slots (read-only base + read-write cooldown)

```
// Buffer: abilities [N_SIMS × 8 × 4]  (8 units × 4 abilities)
struct AbilitySlot {
    // Base stats (read-only)
    damage: f32,          // negative = heal
    range: f32,
    cooldown_max: f32,
    cast_time: f32,
    damage_type: f32,     // 0=phys, 1=magic, 2=true
    is_aoe: f32,          // 0 or 1 (hits all enemies in range)
    targeting: f32,       // 0=enemy, 1=ally, 2=self
    resource_cost: f32,

    // Effect (one per ability, simplified)
    effect_type: f32,     // 0=none, 1=stun, 2=slow, 3=root, 4=silence, 5=dot, 6=hot, 7=shield, 8=buff, 9=debuff
    effect_value: f32,    // duration in ticks (CC) or value (shield/dot/hot amount)
    effect_aux: f32,      // slow_factor, buff multiplier, etc.
    _pad: f32,

    // Runtime state (read-write)
    cooldown_remaining: f32,
    charges: f32,
    charges_max: f32,
    charge_recharge_remaining: f32,
}
// 16 floats = 64 bytes per ability
// Per sim: 8 × 4 × 64 = 2,048 bytes
// 4096 sims: 8 MB
```

### Per-Sim Scratch / Output

```
// Buffer: sim_state [N_SIMS]
struct SimOutput {
    tick: u32,
    done: u32,            // 0=running, 1=heroes_win, 2=heroes_lose, 3=timeout
    heroes_hp_sum: f32,
    enemies_hp_sum: f32,
    // Per-unit cumulative stats for reward shaping
    damage_done: array<f32, 8>,
    healing_done: array<f32, 8>,
    kills: array<u32, 8>,
    deaths: array<u32, 8>,
}
```

### Action Input (written by CPU/policy each tick)

```
// Buffer: actions [N_SIMS × 8]
struct UnitAction {
    action_type: u32,     // 0=attack_nearest, 1=attack_weakest, 2=move_toward, 3=move_away, 4=hold, 5-8=use_ability[0-3]
    target_idx: u32,      // entity index (for pointer-based targeting)
    _pad0: u32,
    _pad1: u32,
}
```

## Compute Shader Pipeline

Three dispatches per tick, with barriers between them:

### Pass 1: Tick Systems (per-unit, embarrassingly parallel)

```
@workgroup_size(8, 1, 1)  // 8 units per sim, 1 sim per workgroup
fn tick_systems(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sim_id = gid.y;
    let unit_id = gid.x;
    // 1. Decrement all cooldowns
    // 2. Tick DoT/HoT (apply damage/healing from status slots)
    // 3. Tick CC timers, clear expired CC
    // 4. Tick buff/debuff timers
    // 5. Tick ability cooldowns + charge recharge
    // 6. Check death (hp <= 0 → alive = 0)
}
```

### Pass 2: Resolve Actions (per-unit, needs barrier after pass 1)

```
@workgroup_size(8, 1, 1)
fn resolve_actions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sim_id = gid.y;
    let unit_id = gid.x;
    // Skip if dead or stunned
    // Read action from actions buffer
    // Switch on action_type:
    //   attack/ability → range check → start cast or resolve instant
    //   move → compute direction, apply speed (× slow factor), update position
    //   hold → nothing
    //   use_ability[n] → CD check → range check → start cast
    // Write damage/heal/CC to a per-sim "pending effects" scratch buffer
    // (avoids race conditions — effects are gathered, not applied inline)
}
```

### Pass 3: Apply Effects + Check Terminal (per-sim, 1 thread)

```
@workgroup_size(1, 1, 1)
fn apply_and_check(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sim_id = gid.x;
    // 1. Apply all pending damage (with armor/MR reduction)
    // 2. Apply all pending heals
    // 3. Apply all pending CC/status effects
    // 4. Check deaths
    // 5. Update sim_output (hp sums, kills, done flag)
    // 6. Advance RNG state
}
```

### Alternative: Single-Pass with Shared Memory

For small unit counts (8), we can use workgroup shared memory:

```
@workgroup_size(8, 1, 1)
fn sim_tick(@builtin(local_invocation_id) lid: vec3<u32>) {
    let sim_id = workgroup_id.y;
    let unit_id = lid.x;

    // Load unit into shared memory
    var<workgroup> shared_units: array<UnitState, 8>;
    shared_units[unit_id] = load_unit(sim_id, unit_id);
    workgroupBarrier();

    // Phase 1: tick systems (each thread handles its unit)
    tick_cooldowns(&shared_units[unit_id]);
    tick_status(&shared_units[unit_id]);
    workgroupBarrier();

    // Phase 2: resolve action (read other units for targeting)
    let effects = resolve_action(unit_id, &shared_units, actions[sim_id][unit_id]);
    workgroupBarrier();

    // Phase 3: apply effects (all threads cooperate)
    // Thread 0 gathers and applies; others wait
    if unit_id == 0u {
        apply_all_effects(&shared_units, &effects);
        check_terminal(sim_id, &shared_units);
    }
    workgroupBarrier();

    // Write back
    store_unit(sim_id, unit_id, shared_units[unit_id]);
}
```

This is simpler and avoids multiple dispatches. With 240 bytes × 8 = 1,920
bytes shared memory per workgroup, well within limits (16-48 KB typical).

## Tick Loop (CPU Side)

```rust
loop {
    // 1. Read sim states from GPU (only "done" flags, async)
    // 2. For non-done sims: encode entity tokens → policy forward pass → sample actions
    // 3. Write actions to GPU buffer
    // 4. Dispatch compute shader (one tick for all sims)
    // 5. Collect training steps from done sims, reset and re-init them
}
```

The policy network runs on GPU (PyTorch/candle), so the flow is:
- GPU sim state → (readback entity encoding) → GPU policy → (write actions) → GPU sim

Ideally minimize CPU roundtrips by doing entity encoding on GPU too.

## Entity Encoding on GPU

Bonus pass: encode the game state into entity tokens directly on GPU, avoiding
a readback. The entity tokens are simple normalized features (30d per entity).

```
@workgroup_size(8, 1, 1)
fn encode_entities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sim_id = gid.y;
    let unit_id = gid.x;
    let self_unit = shared_units[unit_id];

    // For each other unit, compute entity token:
    for (var i = 0u; i < 8u; i++) {
        let other = shared_units[i];
        let dx = other.pos_x - self_unit.pos_x;
        let dy = other.pos_y - self_unit.pos_y;
        let dist = sqrt(dx*dx + dy*dy);
        // ... normalize and write 30d token
    }
    // Output: entity_tokens[sim_id][unit_id][8][30]
}
```

This feeds directly into the policy network's entity encoder (self-attention
over entity tokens). No CPU roundtrip needed for observation encoding.

## Implementation Plan

### Phase 1: Rust + wgpu Scaffold
- [ ] New crate: `gpu_sim` (or module under `src/gpu/`)
- [ ] Define WGSL shader with UnitState/AbilitySlot/Action structs
- [ ] Buffer allocation + initialization from ScenarioCfg
- [ ] Single-tick dispatch + readback test
- [ ] Verify against CPU sim on simple 1v1 (attack-only, no abilities)

### Phase 2: Core Mechanics
- [ ] Movement (direct, no pathfinding)
- [ ] Attack with cast time + cooldown
- [ ] Damage calculation (phys/magic/true + armor/MR)
- [ ] Death detection + terminal condition
- [ ] Per-sim PCG RNG (damage variance)

### Phase 3: Abilities + Status Effects
- [ ] 4 ability slots per unit with cooldowns/charges
- [ ] CC application (stun/slow/root/silence)
- [ ] DoT/HoT status ticking
- [ ] Shield mechanic
- [ ] Buff/debuff multipliers
- [ ] AoE abilities

### Phase 4: RL Integration
- [ ] Entity token encoding shader
- [ ] Action buffer write from policy network
- [ ] Training step collection (state, action, reward, done)
- [ ] Reward shaping (HP diff + kill bonuses)
- [ ] Episode reset + re-initialization
- [ ] Batch PPO training loop

### Phase 5: Optimization
- [ ] Profile: occupancy, memory throughput, divergence
- [ ] Tune workgroup size (maybe 4 units if mostly 4v4)
- [ ] Async readback with double-buffering
- [ ] Consider wgpu → naga-oil for shader composition

## Throughput Estimate

- 4096 sims × 3000 ticks = 12.3M sim-ticks
- At ~8 units/sim × 1 action/tick = 98M action samples
- GPU dispatch: ~50μs per tick (dominated by launch overhead at this scale)
- Total sim time: ~150ms for all 4096 episodes (3000 ticks)
- **Bottleneck shifts to policy inference** (~5ms per forward pass at batch=4096)
- With 300 policy queries per episode (every 10 ticks): 300 × 5ms = 1.5s
- **Expected throughput: ~65K steps/sec** (vs ~72 steps/sec CPU)
- **~900× speedup on simulation, ~50-100× end-to-end with policy**

## Tech Stack

- **wgpu** — Rust-native GPU abstraction (Vulkan/Metal/DX12)
- **naga** — WGSL shader compilation (comes with wgpu)
- **bytemuck** — zero-copy buffer marshalling
- **candle** or **burn** — Rust ML inference on same GPU (avoid CPU roundtrip)
  - Alternative: keep PyTorch, use CUDA interop for shared buffers

## Open Questions

1. **Candle vs PyTorch for policy?** Candle keeps everything in Rust + same GPU.
   PyTorch is more mature and has the existing training code. Could do: GPU sim
   in wgpu, policy in PyTorch via shared CUDA memory (if Vulkan→CUDA interop
   works), or export wgpu buffers via host staging.

2. **How many ability slots?** 4 keeps it simple. Full sim has 8 — could go to
   6 as compromise. More slots = more shared memory.

3. **Do we need AoE?** Simplest version: all abilities are single-target. AoE
   adds a loop per ability resolution. Worth it for healers (heal all allies).

4. **Fidelity floor**: What's the minimum sim complexity where trained policies
   still transfer to the full sim? Needs empirical testing after Phase 2.
