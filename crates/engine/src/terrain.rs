//! Terrain integration seam — minimal trait-object surface for the
//! engine to query height, walkability, and line-of-sight without pulling
//! in any concrete terrain backend.
//!
//! Design history: `docs/superpowers/notes/2026-04-22-terrain-integration-gap.md`.
//! The engine was developed heads-down on the wolves+humans DSL harness
//! and never grew terrain awareness — positions are `Vec3`s in an
//! unbounded R³. This module is the minimum viable seam: a trait the
//! engine can consult and a flat-plane default impl that keeps every
//! existing test deterministic and terrain-agnostic.
//!
//! Recommended pattern (Option B from the gap doc): `SimState` holds
//! `Arc<dyn TerrainQuery + Send + Sync>` defaulted to `FlatPlane`. A
//! future `crates/engine_voxel/` adapter crate can then wrap
//! `/home/ricky/Projects/voxel_engine` and produce a concrete impl
//! without the engine ever depending on wgpu / naga / gpu-allocator.
//!
//! Surface is deliberately tiny — 3 methods. Follow-on features (cover
//! damage modifiers, walkability-filtered MoveToward candidates,
//! destructible terrain) hang off this seam without re-opening the
//! trait.

use glam::Vec3;

pub use crate::state::agent::MovementMode;

/// Engine-facing terrain query trait. Every concrete backend (flat
/// default, hand-authored test hills, future voxel adapter) implements
/// these three methods.
///
/// Method contracts:
///
/// - `height_at(x, y)` — world-space ground height at horizontal
///   `(x, y)`. Returns `0.0` for a flat plane. Units: metres.
/// - `walkable(pos, mode)` — whether the given movement mode can
///   occupy the point. Flat terrain returns `true` unconditionally;
///   a voxel backend might deny `Walk` inside solid rock but allow
///   `Fly` everywhere.
/// - `line_of_sight(from, to)` — whether a straight-line ray between
///   two world points is unobstructed. Flat terrain returns `true`
///   unconditionally. Used by scoring's height-bonus on Attack and
///   by any future cover / ranged-attack gate.
///
/// Callers hold `Arc<dyn TerrainQuery + Send + Sync>`; the trait is
/// `Send + Sync` so parallel tick paths can share one terrain backend
/// without synchronisation overhead.
pub trait TerrainQuery: Send + Sync {
    /// World-space ground height at horizontal `(x, y)`.
    fn height_at(&self, x: f32, y: f32) -> f32;

    /// Whether the given movement mode can occupy `pos`.
    fn walkable(&self, pos: Vec3, mode: MovementMode) -> bool;

    /// Whether the straight-line segment from `from` to `to` is
    /// unobstructed by terrain.
    fn line_of_sight(&self, from: Vec3, to: Vec3) -> bool;
}

/// Flat-plane default. Used by every test that doesn't explicitly
/// inject a terrain backend, so the wolves+humans parity fixture and
/// all legacy unit tests behave exactly as they did pre-seam.
///
/// - Height is `0.0` everywhere.
/// - Every position is walkable in every movement mode.
/// - Every line of sight is clear.
#[derive(Copy, Clone, Debug, Default)]
pub struct FlatPlane;

impl TerrainQuery for FlatPlane {
    #[inline]
    fn height_at(&self, _x: f32, _y: f32) -> f32 {
        0.0
    }

    #[inline]
    fn walkable(&self, _pos: Vec3, _mode: MovementMode) -> bool {
        true
    }

    #[inline]
    fn line_of_sight(&self, _from: Vec3, _to: Vec3) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_plane_height_is_zero() {
        let t = FlatPlane;
        assert_eq!(t.height_at(0.0, 0.0), 0.0);
        assert_eq!(t.height_at(100.0, -50.0), 0.0);
    }

    #[test]
    fn flat_plane_is_always_walkable() {
        let t = FlatPlane;
        assert!(t.walkable(Vec3::new(1.0, 2.0, 3.0), MovementMode::Walk));
        assert!(t.walkable(Vec3::new(-99.0, 0.0, 0.0), MovementMode::Fly));
        assert!(t.walkable(Vec3::new(0.0, 0.0, -10.0), MovementMode::Swim));
    }

    #[test]
    fn flat_plane_has_clear_line_of_sight() {
        let t = FlatPlane;
        assert!(t.line_of_sight(Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)));
        assert!(t.line_of_sight(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0)));
    }
}
