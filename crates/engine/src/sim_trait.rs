//! `CompiledSim` — the uniform interface every per-fixture runtime crate
//! exposes to the generic application layer.
//!
//! ## Why this lives in `engine`
//!
//! Each `{fixture}_runtime` crate (e.g. `boids_runtime` from
//! `assets/sim/boids.sim`) implements [`CompiledSim`] for its own state
//! type. The application crate (`sim_app`) depends on a runtime through
//! a Cargo package alias — switching from boids to a future `cellular`
//! sim is a one-line edit in `sim_app/Cargo.toml`. The trait is the
//! contract that lets the app stay sim-agnostic; the per-fixture crate
//! holds the data + behavior; engine owns the trait because engine is
//! the shared infrastructure crate every fixture already depends on.
//!
//! ## Surface
//!
//! Intentionally minimal. The first cut covers what the bare-minimum
//! tick loop + future visualization need:
//!
//! - [`CompiledSim::step`] — advance one tick.
//! - [`CompiledSim::tick`] — current tick number.
//! - [`CompiledSim::agent_count`] — population size.
//! - [`CompiledSim::positions`] — slice of per-agent world positions
//!   (visualization + introspection).
//!
//! Methods grow as the application crate needs more. Resist adding
//! anything speculatively — every method here is a contract that every
//! future fixture's runtime must satisfy.
//!
//! ## What's NOT in the trait
//!
//! - **Construction.** Each fixture's constructor signature varies
//!   (boids takes `(seed, n)`; a cellular sim might take a grid size).
//!   Each runtime crate exposes its own `pub fn make_sim(...) -> Box<dyn
//!   CompiledSim>` factory; the application calls the active runtime's
//!   factory by its alias name.
//! - **Per-field reads beyond positions.** When the application needs
//!   to render arrows for velocity, expose `velocities()` here. Until
//!   then, fixture-specific accessors stay private to the fixture
//!   crate.
//! - **Mutation.** Per-tick state changes happen inside `step()`; the
//!   trait does not expose hand-off mutation surfaces.

use glam::Vec3;

/// Uniform interface for every per-fixture compiled simulation.
///
/// Implementors live in `{fixture}_runtime` crates; the application
/// crate calls these methods through a `Box<dyn CompiledSim>` returned
/// from each runtime's `make_sim()` factory. Every method is intended
/// to be cheap (positions returns a borrowed slice; counters are
/// scalar reads); long-running work belongs inside `step()`.
pub trait CompiledSim {
    /// Advance the simulation by one tick. Implementations dispatch
    /// whatever per-tick kernels the compiler emitted (or hand-coded
    /// placeholder logic until those kernels exist) and increment the
    /// internal tick counter.
    fn step(&mut self);

    /// Current tick number — the count of completed `step()` calls.
    /// Newly-constructed sims report `0`.
    fn tick(&self) -> u64;

    /// Number of live agents in the simulation. Constant for fixtures
    /// without spawn/despawn; may grow/shrink for fixtures that emit
    /// `Spawned` / `Despawned` events inside their handlers.
    fn agent_count(&self) -> u32;

    /// Per-agent world position, indexed by agent slot. The slice's
    /// length must equal [`Self::agent_count`].
    ///
    /// Takes `&mut self` because GPU-backed implementations need to
    /// drive a readback (encode pos→staging copy + map+await) before a
    /// host-readable slice exists. CPU implementations that already
    /// keep host-side `Vec<Vec3>` ignore the mutability.
    fn positions(&mut self) -> &[Vec3];
}
