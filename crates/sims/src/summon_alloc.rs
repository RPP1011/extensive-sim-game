//! Host-side summon allocator: turns EffectSummonApplied (chronicle kind 62)
//! records into live agents in dead SoA slots. Split into a pure planning
//! fn (unit-tested here, no GPU) and a GPU drain (drain_summons, added in B2).

use glam::Vec3;

/// Radius of the ring around the player on which summoned enemies appear.
pub const SPAWN_RING_RADIUS: f32 = 30.0;

/// Enemy types: (hp, move_speed). 0 Grunt (baseline), 1 Swift (fast/fragile),
/// 2 Brute (slow/tanky). Indexed by `SlotAssignment::kind`.
pub const ENEMY_TYPES: [(f32, f32); 3] = [
    (12.0, 0.40), // Grunt
    (6.0, 0.75),  // Swift
    (30.0, 0.22), // Brute
];

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
    /// Enemy type index into ENEMY_TYPES (0 Grunt / 1 Swift / 2 Brute).
    pub kind: u8,
}

/// Pure allocation planning. Deterministic: records are processed in `seq`
/// order; dead slots (`alive[i] == 0`) are claimed in ascending index order;
/// per-spawn position = spawner pos + a keyed-PCG-seeded offset.
/// Truncates (does not panic) when the dead-slot pool is exhausted.
pub fn plan_allocations(
    alive: &[u32],
    records: &[SummonRecord],
    spawner_pos: impl Fn(u32) -> Vec3,
    seed: u64,
    tick: u64,
    pool_start: u32,
) -> Vec<SlotAssignment> {
    let mut sorted: Vec<SummonRecord> = records.to_vec();
    sorted.sort_by_key(|r| (r.seq, r.actor_slot));
    let mut claimed = vec![false; alive.len()];
    let mut out = Vec::new();
    // Only claim slots in the enemy pool — never the reserved sentinel/player/
    // spawner slots below pool_start (even if they read alive==0).
    let mut cursor = pool_start as usize;
    for rec in &sorted {
        let base = spawner_pos(rec.actor_slot);
        for _ in 0..rec.count {
            while cursor < alive.len() && (alive[cursor] != 0 || claimed[cursor]) {
                cursor += 1;
            }
            if cursor >= alive.len() {
                return out; // pool exhausted — truncate
            }
            claimed[cursor] = true;
            let new_slot = cursor as u32;
            let off = seeded_offset(seed, new_slot, tick);
            // Spawn on a RING around the summoner (the player): random angle,
            // radius ~SPAWN_RING_RADIUS. This is the VS "enemies appear around
            // you" pattern (the actor pos is the player; see drain_summons).
            let ang = (off & 0xFFFF) as f32 / 65535.0 * std::f32::consts::TAU;
            let jitter = ((off >> 16) & 0xFF) as f32 / 255.0 * 6.0 - 3.0;
            let rad = SPAWN_RING_RADIUS + jitter;
            // Enemy type from independent high bits → 0 Grunt / 1 Swift / 2 Brute.
            let kind = ((off >> 28) % 3) as u8;
            out.push(SlotAssignment {
                slot: new_slot,
                pos: base + Vec3::new(rad * ang.cos(), rad * ang.sin(), 0.0),
                kind,
            });
            cursor += 1;
        }
    }
    out
}

/// Deterministic per-slot offset hash (P5 keyed PCG).
///
/// Uses `engine::rng::per_agent_u32_pcg` — the GPU-parity integer mixing
/// chain. Purpose id 5 is reserved for "vs_spawn_pos" (host-only; no WGSL
/// mirror required). `per_agent_u32` (the ahash path) is not used here
/// because `AgentId` is NonZeroU32 (rejects slot 0) and this call site is
/// host-only anyway.
fn seeded_offset(seed: u64, new_slot: u32, tick: u64) -> u32 {
    engine::rng::per_agent_u32_pcg(seed as u32, new_slot, tick as u32, 5)
}

/// Minimal surface the GPU drain needs. The generated GeneratedRuntime
/// exposes all of these as public fields.
pub struct DrainCtx<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub event_ring: &'a engine::gpu::EventRing,
    pub agent_alive_buf: &'a wgpu::Buffer,
    pub agent_pos_buf: &'a wgpu::Buffer,
    pub agent_hp_buf: &'a wgpu::Buffer,
    pub agent_move_speed_buf: &'a wgpu::Buffer,
    pub agent_count: u32,
    pub seed: u64,
    pub tick: u64,
    /// First claimable slot — the enemy pool start. Slots below this
    /// (sentinel/player/spawners) are never allocated to summoned enemies.
    pub pool_start: u32,
}

/// Read EffectSummonApplied (kind 62) records from the ring, plan dead-slot
/// allocations, and write `alive=1` + pos for each. Returns count allocated.
pub fn drain_summons(ctx: DrainCtx) -> usize {
    const KIND_SUMMON: u32 = 62;
    let stride = engine::gpu::EVENT_STRIDE_U32 as usize;

    // tail_value() is a host-side estimate that step() resets to 0 at the
    // start of every tick (via reset_tail_estimate). After step() returns
    // the GPU has already run and the real tail is in the GPU buffer, so
    // we always read it back directly rather than trusting the (stale) host
    // estimate.
    let tail_words = readback_u32(ctx.device, ctx.queue, ctx.event_ring.tail(), 4);
    let n_slots = tail_words.first().copied().unwrap_or(0);
    if n_slots == 0 {
        return 0;
    }
    let ring_bytes = ((n_slots as u64) * stride as u64 * 4).max(16);
    let ring_words = readback_u32(ctx.device, ctx.queue, ctx.event_ring.ring(), ring_bytes);

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
    if records.is_empty() {
        return 0;
    }

    let alive = readback_u32(
        ctx.device,
        ctx.queue,
        ctx.agent_alive_buf,
        (ctx.agent_count as u64 * 4).max(16),
    );
    let pos_words = readback_u32(
        ctx.device,
        ctx.queue,
        ctx.agent_pos_buf,
        (ctx.agent_count as u64 * 16).max(16),
    );
    let read_pos = |slot: u32| -> Vec3 {
        let b = slot as usize * 4; // vec3 stored as 4 f32 (xyz + pad)
        Vec3::new(
            f32::from_bits(pos_words[b]),
            f32::from_bits(pos_words[b + 1]),
            f32::from_bits(pos_words[b + 2]),
        )
    };

    let plan = plan_allocations(&alive, &records, read_pos, ctx.seed, ctx.tick, ctx.pool_start);
    if plan.is_empty() {
        return 0;
    }
    // Write ONLY the newly-claimed slots — never rewrite the whole buffer.
    // Clobbering all N agents each tick races with the runtime's own
    // per-tick agent state (it desyncs e.g. the player slot against the
    // pack/unpack cycle); targeted sub-range writes touch only the new
    // enemies and leave every other agent's GPU state untouched.
    for a in &plan {
        let alive_one = [1u32];
        ctx.queue.write_buffer(
            ctx.agent_alive_buf,
            a.slot as u64 * 4,
            bytemuck::cast_slice(&alive_one),
        );
        let pos_one = [a.pos.x.to_bits(), a.pos.y.to_bits(), a.pos.z.to_bits(), 0u32];
        ctx.queue.write_buffer(
            ctx.agent_pos_buf,
            a.slot as u64 * 16,
            bytemuck::cast_slice(&pos_one),
        );
        // Per enemy type: hp + move_speed.
        let (hp, spd) = ENEMY_TYPES[a.kind as usize];
        ctx.queue.write_buffer(ctx.agent_hp_buf, a.slot as u64 * 4, bytemuck::cast_slice(&[hp]));
        ctx.queue
            .write_buffer(ctx.agent_move_speed_buf, a.slot as u64 * 4, bytemuck::cast_slice(&[spd]));
    }
    plan.len()
}

fn readback_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    bytes: u64,
) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("summon_alloc::readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("summon_alloc::rb"),
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claims_dead_slots_in_order_and_truncates() {
        let alive = [1u32, 1, 0, 0, 0, 0];
        let recs = [SummonRecord { actor_slot: 1, template_hash: 7, count: 3, seq: 0 }];
        let got = plan_allocations(&alive, &recs, |_| Vec3::ZERO, 0xABCD, 5, 0);
        let slots: Vec<u32> = got.iter().map(|a| a.slot).collect();
        assert_eq!(slots, vec![2, 3, 4], "claims first 3 dead slots in order");

        let recs2 = [SummonRecord { actor_slot: 1, template_hash: 7, count: 10, seq: 0 }];
        let got2 = plan_allocations(&alive, &recs2, |_| Vec3::ZERO, 0xABCD, 5, 0);
        assert_eq!(got2.len(), 4, "truncates at pool exhaustion, no panic");
    }

    #[test]
    fn seq_ordering_is_respected() {
        let alive = [0u32; 8];
        let recs = [
            SummonRecord { actor_slot: 200, template_hash: 1, count: 2, seq: 10 },
            SummonRecord { actor_slot: 100, template_hash: 2, count: 2, seq: 1 },
        ];
        let pos_of = |actor: u32| Vec3::new(actor as f32, 0.0, 0.0); // actor_slot doubles as x
        let got = plan_allocations(&alive, &recs, pos_of, 7, 1, 0);
        // seq=1 (actor 100) first -> first 2 spawns ring around x=100 (±ring+jitter)
        let tol = SPAWN_RING_RADIUS + 4.0;
        assert!((got[0].pos.x - 100.0).abs() <= tol, "first spawn should come from seq=1 record (x~100±ring); got {}", got[0].pos.x);
        assert!((got[1].pos.x - 100.0).abs() <= tol, "second spawn should come from seq=1 record (x~100±ring); got {}", got[1].pos.x);
        // then seq=10 (actor 200) -> next 2 ring around x=200
        assert!((got[2].pos.x - 200.0).abs() <= tol, "third spawn should come from seq=10 record (x~200±ring); got {}", got[2].pos.x);
        // slots claimed ascending
        let slots: Vec<u32> = got.iter().map(|a| a.slot).collect();
        assert_eq!(slots, vec![0, 1, 2, 3]);
    }

    #[test]
    fn deterministic_across_runs() {
        let alive = [0u32; 8];
        let recs = [SummonRecord { actor_slot: 0, template_hash: 1, count: 4, seq: 0 }];
        let a = plan_allocations(&alive, &recs, |_| Vec3::new(10.0, 0.0, 0.0), 42, 1, 0);
        let b = plan_allocations(&alive, &recs, |_| Vec3::new(10.0, 0.0, 0.0), 42, 1, 0);
        assert_eq!(a, b, "same inputs -> same plan (P5)");
    }
}
