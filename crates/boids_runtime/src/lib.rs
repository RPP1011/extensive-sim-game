//! Per-fixture runtime for `assets/sim/boids.sim`.
//!
//! Today: hand-written. Eventually: compiler-emitted from `boids.sim`
//! with the exact same crate name and external API (per the
//! `{DSL_SIM_FILE_NAME}_runtime` naming rule). When emission lands the
//! body of [`BoidsState::step`] will be replaced by dispatch into
//! compiler-emitted CG kernels; everything else (the struct shape, the
//! [`CompiledSim`] impl, the [`make_sim`] factory) stays as-is so the
//! application crate doesn't need to change.
//!
//! ## State shape
//!
//! [`BoidsState`] owns its own per-field storage (`pos`, `vel`) rather
//! than borrowing engine's `state::AgentSoA`. Two reasons:
//!   1. The wolf-sim AgentSoA carries fields (`hp`, `attack_damage`,
//!      `stun_expires_at_tick`) that boids don't need.
//!   2. Per-fixture runtime crates each owning their own state shape
//!      means each fixture's state is independent — no shared global
//!      SoA gets layered with multiple fixtures' fields.
//!
//! Engine's `rng`, `spatial`, and other primitives stay shared.
//!
//! ## Step body
//!
//! Currently a no-op — `step()` only advances the tick counter. The
//! actual velocity-update + position-integration logic from
//! `boids.sim`'s `physics MoveBoid` rule lives in DSL today and will
//! emit through the compiler later. Until then boids are frozen at
//! their initial positions; the loop runs, the tick advances, the
//! application crate's harness exercises the [`CompiledSim`]
//! interface end-to-end.
//!
//! Anything beyond `tick += 1` here would be hand-coded throwaway
//! that the DSL emit will replace; we deliberately don't write it.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use glam::Vec3;

/// Per-fixture state for the boids simulation. Mirrors the shape that
/// `boids.sim`'s `entity Boid : Agent { pos: vec3, vel: vec3 }` will
/// resolve to in the eventually-emitted runtime.
pub struct BoidsState {
    pos: Vec<Vec3>,
    vel: Vec<Vec3>,
    tick: u64,
    seed: u64,
}

impl BoidsState {
    /// Construct an N-boid simulation with deterministic initial
    /// positions + velocities derived from `seed` via engine's keyed
    /// PCG RNG (P5: `per_agent_u32(seed, agent_id, tick=0, purpose)`).
    /// Positions land in a small cube around the origin; velocities
    /// are tiny perturbations so any future integration step has
    /// non-zero motion to apply.
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        // AgentId is a NonZeroU32 newtype — start at 1, not 0.
        for slot in 0..agent_count {
            let agent_id = AgentId::new(slot + 1)
                .expect("slot+1 is non-zero by construction");
            // Three independent RNG draws per axis, normalised to
            // [-spread, +spread]. Distinct `purpose` byte-tags keep
            // each axis's stream independent of the others (P5 keying
            // contract — purpose is &[u8]).
            let spread = 8.0_f32;
            let nudge = 0.05_f32;
            pos.push(Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_pos_z")) * spread,
            ));
            vel.push(Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"boid_init_vel_z")) * nudge,
            ));
        }
        Self { pos, vel, tick: 0, seed }
    }

    /// Read-only view of velocities — kept private to the runtime
    /// crate because the [`CompiledSim`] trait doesn't expose vel
    /// today. When/if the application needs to render velocity arrows,
    /// add a method on the trait and remove this comment.
    #[allow(dead_code)]
    pub(crate) fn velocities(&self) -> &[Vec3] {
        &self.vel
    }

    /// Seed used at construction. Kept for snapshot/replay debugging
    /// even though the trait doesn't expose it; also used by future
    /// step implementations that need keyed RNG draws per tick.
    #[allow(dead_code)]
    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }
}

impl CompiledSim for BoidsState {
    fn step(&mut self) {
        // Minimum-throwaway body. The actual velocity update +
        // position integration from boids.sim's `physics MoveBoid`
        // rule lives in DSL and will emit through the compiler later.
        // No hand-coded physics here — anything written today is
        // exactly what the DSL emit replaces.
        self.tick += 1;
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn agent_count(&self) -> u32 {
        self.pos.len() as u32
    }

    fn positions(&self) -> &[Vec3] {
        &self.pos
    }
}

/// Construct a boxed [`CompiledSim`] for the application crate. The
/// application calls this through its `sim_runtime` package alias —
/// switching to a different fixture's runtime crate is a one-line
/// change in `sim_app/Cargo.toml`'s `package = "..."` rename.
pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(BoidsState::new(seed, agent_count))
}

/// Map a `u32` RNG draw into `[-1.0, 1.0]`. PCG outputs are uniform
/// over `u32::MIN..=u32::MAX`; the centred remap keeps the math
/// branch-free and stable across host/GPU.
fn normalise(raw: u32) -> f32 {
    let half = (u32::MAX / 2) as f32;
    (raw as f32 - half) / half
}
