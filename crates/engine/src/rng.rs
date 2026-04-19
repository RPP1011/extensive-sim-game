// crates/engine/src/rng.rs
use crate::ids::AgentId;

/// Simple PCG-XSH-RR 64-bit state / 32-bit output RNG.
pub struct WorldRng { state: u64, inc: u64 }

impl WorldRng {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Self { state: 0, inc: (seed.wrapping_shl(1)) | 1 };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        let _ = rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

/// Derive a deterministic u32 from (world_seed, agent_id, tick, purpose_tag).
/// Purpose tags: b"action", b"sample", b"shuffle", b"conception", etc.
pub fn per_agent_u32(
    world_seed: u64,
    agent_id: AgentId,
    tick: u64,
    purpose: &[u8],
) -> u32 {
    let mut h = ahash::AHasher::default();
    use std::hash::{Hash, Hasher};
    world_seed.hash(&mut h);
    agent_id.raw().hash(&mut h);
    tick.hash(&mut h);
    purpose.hash(&mut h);
    h.finish() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::AgentId;

    #[test]
    fn same_world_seed_gives_same_sequence() {
        let mut a = WorldRng::from_seed(42);
        let mut b = WorldRng::from_seed(42);
        let seq_a: Vec<u32> = (0..100).map(|_| a.next_u32()).collect();
        let seq_b: Vec<u32> = (0..100).map(|_| b.next_u32()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = WorldRng::from_seed(42);
        let mut b = WorldRng::from_seed(43);
        let seq_a: Vec<u32> = (0..100).map(|_| a.next_u32()).collect();
        let seq_b: Vec<u32> = (0..100).map(|_| b.next_u32()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn per_agent_stream_is_deterministic_and_distinct() {
        let world_seed = 42;
        let tick = 100;
        let a1 = per_agent_u32(world_seed, AgentId::new(1).unwrap(), tick, b"action");
        let a2 = per_agent_u32(world_seed, AgentId::new(1).unwrap(), tick, b"action");
        let b1 = per_agent_u32(world_seed, AgentId::new(2).unwrap(), tick, b"action");
        assert_eq!(a1, a2, "same inputs must reproduce");
        assert_ne!(a1, b1, "different agent IDs must diverge");
    }

    #[test]
    fn purpose_tag_separates_streams() {
        let a = per_agent_u32(42, AgentId::new(1).unwrap(), 100, b"action");
        let b = per_agent_u32(42, AgentId::new(1).unwrap(), 100, b"sample");
        assert_ne!(a, b);
    }
}
