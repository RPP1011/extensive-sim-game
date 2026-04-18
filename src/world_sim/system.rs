//! System trait: uniform interface for all world-sim systems. Enables
//! per-system timing, backend dispatch (scalar vs SIMD), and clean
//! stage-based scheduling.

use super::delta::WorldDelta;
use super::state::WorldState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    ComputeHigh,
    ComputeMedium,
    ComputeLow,
    ComputeGrid,
    ApplyClone,
    ApplyHp,
    ApplyMovement,
    ApplyStatus,
    ApplyEconomy,
    ApplyTransfers,
    ApplyDeaths,
    ApplyGrid,
    ApplyFidelity,
    ApplyPriceReports,
    PostApply,
}

impl Stage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Stage::ComputeHigh => "ComputeHigh",
            Stage::ComputeMedium => "ComputeMedium",
            Stage::ComputeLow => "ComputeLow",
            Stage::ComputeGrid => "ComputeGrid",
            Stage::ApplyClone => "ApplyClone",
            Stage::ApplyHp => "ApplyHp",
            Stage::ApplyMovement => "ApplyMovement",
            Stage::ApplyStatus => "ApplyStatus",
            Stage::ApplyEconomy => "ApplyEconomy",
            Stage::ApplyTransfers => "ApplyTransfers",
            Stage::ApplyDeaths => "ApplyDeaths",
            Stage::ApplyGrid => "ApplyGrid",
            Stage::ApplyFidelity => "ApplyFidelity",
            Stage::ApplyPriceReports => "ApplyPriceReports",
            Stage::PostApply => "PostApply",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Simd,
}

impl Backend {
    /// Runtime default. Returns Scalar until SIMD implementations land;
    /// future work will add CPU-feature detection here.
    pub fn default_for_cpu() -> Self { Backend::Scalar }
}

/// Execution context handed to each system per tick.
pub struct SystemCtx<'a> {
    pub state: &'a WorldState,
    pub deltas: &'a mut Vec<WorldDelta>,
    pub tick: u64,
}

/// Core trait implemented by every actively-dispatched world-sim system.
pub trait System: Send + Sync {
    fn name(&self) -> &'static str;
    fn stage(&self) -> Stage;
    /// Runs one tick's worth of work. Returns entities/units touched
    /// this call (used for ns/entity stats — return 0 if unclear).
    fn run(&self, ctx: &mut SystemCtx) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummySystem;
    impl System for DummySystem {
        fn name(&self) -> &'static str { "dummy" }
        fn stage(&self) -> Stage { Stage::PostApply }
        fn run(&self, _ctx: &mut SystemCtx) -> u32 { 7 }
    }

    #[test]
    fn trait_surface_is_object_safe() {
        let sys: Box<dyn System> = Box::new(DummySystem);
        assert_eq!(sys.name(), "dummy");
        assert_eq!(sys.stage(), Stage::PostApply);
    }

    #[test]
    fn backend_default_is_scalar() {
        assert_eq!(Backend::default_for_cpu(), Backend::Scalar);
    }
}
