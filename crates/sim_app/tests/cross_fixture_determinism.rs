//! Cross-fixture determinism regression. Originally ran each per-fixture
//! runtime through the `CompiledSim` trait twice from the same seed and
//! asserted bit-identical positions. Every fixture this test covered
//! has since migrated to the `sims` mega-crate (predator_prey,
//! particle_collision, crowd_navigation) — the per-fixture
//! `*_runtime` crates have been deleted.
//!
//! The mega-crate `sims::<fixture>::GeneratedRuntime` shape doesn't
//! yet expose a position-readback surface (`positions()` returns
//! `&[]`), so re-implementing the bit-identical comparison would be
//! vacuous. Re-add fixtures here once the mega-crate exposes the
//! per-fixture position arrays.
