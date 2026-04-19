// crates/engine/src/rng.rs
use std::hash::{BuildHasher, Hash, Hasher};

use crate::ids::AgentId;

/// PCG-XSH-RR 64→32 pseudo-random number generator.
/// See O'Neill 2014 — <https://www.pcg-random.org/>.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorldRng {
    state: u64,
    inc: u64,
}

impl WorldRng {
    /// Construct from a seed using a canonical fixed stream id.
    pub fn from_seed(seed: u64) -> Self {
        Self::from_seed_with_stream(seed, 0xCAFE_F00D_D15E_A5E5)
    }

    /// Construct from a seed and an explicit PCG stream id.
    ///
    /// `stream_id` selects which of the 2^63 independent PCG sequences to use.
    /// The canonical stream id `0xCAFEF00DD15EA5E5` is the PCG reference initseq.
    pub fn from_seed_with_stream(seed: u64, stream_id: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream_id.wrapping_shl(1)) | 1,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        let _ = rng.next_u32();
        rng
    }

    /// PCG-XSH-RR 64→32. See O'Neill 2014.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

/// Derive a deterministic u32 from `(world_seed, agent_id, tick, purpose_tag)`.
///
/// Purpose tags: `b"action"`, `b"sample"`, `b"shuffle"`, `b"conception"`, etc.
///
/// Delegates to [`per_agent_u64`] and truncates to 32 bits.
pub fn per_agent_u32(world_seed: u64, agent_id: AgentId, tick: u64, purpose: &[u8]) -> u32 {
    per_agent_u64(world_seed, agent_id, tick, purpose) as u32
}

/// Derive a deterministic u64 from `(world_seed, agent_id, tick, purpose_tag)`.
///
/// # Determinism — schema-hash load-bearing constants
///
/// The four seed constants passed to `ahash::RandomState::with_seeds` are
/// **fixed forever** (or until an explicit engine schema-hash bump). They
/// replace `AHasher::default()`, which pulls a runtime-random seed from the OS
/// and therefore produces different hashes across processes — breaking the
/// engine's determinism contract.
///
/// DO NOT change these constants without bumping the engine schema hash.
pub fn per_agent_u64(world_seed: u64, agent_id: AgentId, tick: u64, purpose: &[u8]) -> u64 {
    // Fixed seeds — DO NOT CHANGE without bumping the engine schema hash.
    // These four constants anchor per-agent stream determinism across processes.
    let state = ahash::RandomState::with_seeds(
        0xA5A5_A5A5_A5A5_A5A5,
        0x5A5A_5A5A_5A5A_5A5A,
        0xDEAD_BEEF_CAFE_F00D,
        0x0123_4567_89AB_CDEF,
    );
    let mut h = state.build_hasher();
    world_seed.hash(&mut h);
    agent_id.raw().hash(&mut h);
    tick.hash(&mut h);
    purpose.hash(&mut h);
    h.finish()
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

    #[test]
    fn world_rng_golden_value() {
        let mut rng = WorldRng::from_seed(42);
        let first = rng.next_u32();
        // Pin the value — any change here is either a bug or an intentional
        // determinism-breaking change that requires a schema-hash bump.
        assert_eq!(first, 0x17BF_3553);
    }

    #[test]
    fn per_agent_golden_value() {
        let v = per_agent_u64(42, AgentId::new(1).unwrap(), 100, b"action");
        // Pin the value — see note above.
        assert_eq!(v, 0xDD23_B8FC_F784_B656);
    }
}
