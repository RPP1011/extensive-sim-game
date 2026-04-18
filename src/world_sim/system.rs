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

/// Registry of all systems grouped by `Stage`. Preserves insertion order
/// within each stage so dispatch is deterministic.
pub struct SystemRegistry {
    by_stage: std::collections::HashMap<Stage, Vec<Box<dyn System>>>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        Self { by_stage: std::collections::HashMap::new() }
    }

    pub fn register<S: System + 'static>(&mut self, sys: S) {
        self.by_stage.entry(sys.stage()).or_default().push(Box::new(sys));
    }

    pub fn systems_in(&self, stage: Stage) -> &[Box<dyn System>] {
        self.by_stage.get(&stage).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Iterator over all (stage, systems) pairs. Used for profiling reports.
    pub fn all_stages(&self) -> impl Iterator<Item = (Stage, &[Box<dyn System>])> {
        self.by_stage.iter().map(|(k, v)| (*k, v.as_slice()))
    }
}

impl Default for SystemRegistry {
    fn default() -> Self { Self::new() }
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

    struct StageSystem(Stage, &'static str);
    impl System for StageSystem {
        fn name(&self) -> &'static str { self.1 }
        fn stage(&self) -> Stage { self.0 }
        fn run(&self, _ctx: &mut SystemCtx) -> u32 { 0 }
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

    #[test]
    fn registry_groups_by_stage() {
        let mut r = SystemRegistry::new();
        r.register(StageSystem(Stage::PostApply, "a"));
        r.register(StageSystem(Stage::ApplyHp, "b"));
        r.register(StageSystem(Stage::PostApply, "c"));

        let hp: Vec<_> = r.systems_in(Stage::ApplyHp).iter().map(|s| s.name()).collect();
        let post: Vec<_> = r.systems_in(Stage::PostApply).iter().map(|s| s.name()).collect();
        assert_eq!(hp, vec!["b"]);
        assert_eq!(post, vec!["a", "c"]);
    }

    #[test]
    fn registry_empty_stage_returns_empty_slice() {
        let r = SystemRegistry::new();
        assert!(r.systems_in(Stage::ComputeHigh).is_empty());
    }
}
