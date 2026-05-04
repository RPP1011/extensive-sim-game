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

// ---------------------------------------------------------------------------
// GPU-parity per-agent RNG
// ---------------------------------------------------------------------------

/// David Stafford's mix13 splitmix64-style finalizer, narrowed to u32.
///
/// The host-side mirror of the WGSL `rng_mix64` function emitted by
/// the dsl_compiler RNG prelude
/// (`crates/dsl_compiler/src/cg/emit/program.rs::RNG_WGSL_PRELUDE`).
/// Both sides must produce bit-identical output for the same input
/// (P11 — cross-backend bit-equality).
///
/// DO NOT modify the multiplier constants without bumping the engine
/// schema hash AND updating `RNG_WGSL_PRELUDE` in lockstep.
#[inline]
fn rng_mix32(x: u32) -> u32 {
    let mut z = x;
    z = (z ^ (z >> 16)).wrapping_mul(0x7feb_352d);
    z = (z ^ (z >> 15)).wrapping_mul(0x846c_a68b);
    z ^ (z >> 16)
}

/// GPU-parity deterministic per-agent RNG primitive.
///
/// Folds `(world_seed, agent_id, tick, purpose_id)` through the same
/// pure-integer mixing chain the WGSL prelude uses
/// (`crates/dsl_compiler/src/cg/emit/program.rs::RNG_WGSL_PRELUDE`),
/// so a draw made on the GPU at tick T for agent A with purpose P
/// produces the same u32 the host produces for the same inputs (P11).
///
/// # Inputs
///
/// - `world_seed`: low 32 bits of the engine's world seed. The engine
///   seed is 64-bit; the cfg uniform field is u32 so the GPU sees the
///   low half. Any future widening to 64-bit GPU seed mirrors here.
/// - `agent_id`: raw agent slot id (`AgentId::raw() as u32`).
/// - `tick`: world tick as u32 (matches `cfg.tick` — runtimes already
///   cast `state.tick as u32` per the existing cfg shape).
/// - `purpose_id`: stable u32 from `RngPurpose::wgsl_id()`. The
///   canonical mapping is `Action=1, Sample=2, Shuffle=3,
///   Conception=4`. Lockstep with
///   `crates/dsl_compiler/src/cg/data_handle.rs`.
///
/// # When to use which
///
/// - Runtime construction-time RNG (e.g. initial position scattering
///   in `boids_runtime`, `foraging_runtime`) → use [`per_agent_u32`]
///   with a `&[u8]` purpose. Host-only; ergonomic byte-slice keying.
/// - GPU-emitted kernel bodies → the WGSL `per_agent_u32(...)` call;
///   host code that needs to reproduce the same draw uses this.
///
/// See `docs/superpowers/notes/2026-05-04-stochastic_probe.md` (Gap
/// #2) for the discovery doc.
pub fn per_agent_u32_pcg(world_seed: u32, agent_id: u32, tick: u32, purpose_id: u32) -> u32 {
    let mut s = rng_mix32(world_seed ^ 0x9E37_79B9);
    s = rng_mix32(s ^ agent_id);
    s = rng_mix32(s ^ tick);
    s = rng_mix32(s ^ purpose_id);
    s
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

    #[test]
    fn per_agent_pcg_is_deterministic_and_distinct() {
        let a1 = per_agent_u32_pcg(0xDEAD_BEEF, 1, 100, 1);
        let a2 = per_agent_u32_pcg(0xDEAD_BEEF, 1, 100, 1);
        let b = per_agent_u32_pcg(0xDEAD_BEEF, 2, 100, 1);
        assert_eq!(a1, a2, "same inputs must reproduce");
        assert_ne!(a1, b, "different agent ids must diverge");
    }

    #[test]
    fn per_agent_pcg_purpose_separates_streams() {
        let action = per_agent_u32_pcg(0xCAFE_F00D, 1, 100, 1);
        let sample = per_agent_u32_pcg(0xCAFE_F00D, 1, 100, 2);
        let shuffle = per_agent_u32_pcg(0xCAFE_F00D, 1, 100, 3);
        let conception = per_agent_u32_pcg(0xCAFE_F00D, 1, 100, 4);
        // All four purposes must produce distinct streams under the
        // same (seed, agent, tick) tuple.
        assert_ne!(action, sample);
        assert_ne!(action, shuffle);
        assert_ne!(action, conception);
        assert_ne!(sample, shuffle);
    }

    #[test]
    fn per_agent_pcg_distribution_is_roughly_uniform() {
        // Smoke test: 1000 draws under threshold 30%, count fires,
        // expect ≈ 300. The stochastic_probe runtime relies on this
        // distribution to converge to T × p = 300 per slot.
        let mut fires = 0;
        for tick in 0..1000u32 {
            let draw = per_agent_u32_pcg(0x1234_5678, 7, tick, 1);
            if draw % 100 < 30 {
                fires += 1;
            }
        }
        assert!(
            (250..=350).contains(&fires),
            "PCG draw distribution looks skewed: {fires} fires across 1000 \
             ticks at 30% threshold (expected ≈ 300, ±5σ tolerance)",
        );
    }

    #[test]
    fn per_agent_pcg_golden_value() {
        // Pin the value — anchors the WGSL prelude
        // (`RNG_WGSL_PRELUDE` in dsl_compiler) to the exact same
        // mixing chain. P11 cross-backend bit-equality. Any change
        // here is a determinism-breaking change requiring a schema
        // hash bump AND a lockstep update to the WGSL prelude.
        let v = per_agent_u32_pcg(42, 1, 100, 1);
        assert_eq!(v, 0x5CBB_D99C);
    }
}
