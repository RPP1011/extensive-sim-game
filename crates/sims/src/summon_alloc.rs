//! Host-side summon allocator: turns EffectSummonApplied (chronicle kind 62)
//! records into live agents in dead SoA slots. Split into a pure planning
//! fn (unit-tested here, no GPU) and a GPU drain (drain_summons, added in B2).

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
/// per-spawn position = spawner pos + a keyed-PCG-seeded offset.
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
            while cursor < alive.len() && (alive[cursor] != 0 || claimed[cursor]) {
                cursor += 1;
            }
            if cursor >= alive.len() {
                return out; // pool exhausted — truncate
            }
            claimed[cursor] = true;
            let new_slot = cursor as u32;
            let off = seeded_offset(seed, new_slot, tick);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claims_dead_slots_in_order_and_truncates() {
        let alive = [1u32, 1, 0, 0, 0, 0];
        let recs = [SummonRecord { actor_slot: 1, template_hash: 7, count: 3, seq: 0 }];
        let got = plan_allocations(&alive, &recs, |_| Vec3::ZERO, 0xABCD, 5);
        let slots: Vec<u32> = got.iter().map(|a| a.slot).collect();
        assert_eq!(slots, vec![2, 3, 4], "claims first 3 dead slots in order");

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
