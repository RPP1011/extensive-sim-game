//! Determinism verification (DET-* error codes).
//!
//! All determinism checks follow the same pattern: clone inputs, run N times,
//! assert equality on outputs. Covers seed generation, action application,
//! oracle output, and spatial features.

use std::fmt::Debug;

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::building_ai::features::{compute_spatial_features, SpatialFeatures};
use crate::world_sim::building_ai::oracle::{strategic_oracle, structural_oracle};
use crate::world_sim::building_ai::scoring::apply_actions;
use crate::world_sim::building_ai::types::{BuildingAction, BuildingObservation};
use crate::world_sim::state::WorldState;

/// Default number of repetitions for determinism checks.
pub const DEFAULT_REPS: usize = 10;

// ---------------------------------------------------------------------------
// Generic determinism assertion helper
// ---------------------------------------------------------------------------

/// Run `f` N times and compare results to the baseline (first run).
/// Returns errors for any run that differs.
fn assert_deterministic<T: PartialEq + Debug>(
    code: &'static str,
    name: &str,
    n: usize,
    f: impl Fn() -> T,
) -> Vec<ValidationError> {
    let baseline = f();
    let mut errors = Vec::new();
    for i in 1..n {
        let result = f();
        if result != baseline {
            errors.push(ValidationError {
                code,
                severity: Severity::Fatal,
                message: format!("{}: run {} differs from run 0", name, i),
                context: ErrorContext::default(),
            });
        }
    }
    errors
}

// ---------------------------------------------------------------------------
// DET-ACT-001: Action application determinism
// ---------------------------------------------------------------------------

/// Verify that `apply_actions` on the same state and actions produces
/// identical results across N runs.
pub fn check_action_determinism(
    state: &WorldState,
    actions: &[BuildingAction],
    n: usize,
) -> Vec<ValidationError> {
    // Run N times, compare entity counts and building positions
    let baseline = {
        let mut s = state.clone();
        apply_actions(&mut s, actions);
        snapshot_state(&s)
    };

    let mut errors = Vec::new();
    for i in 1..n {
        let result = {
            let mut s = state.clone();
            apply_actions(&mut s, actions);
            snapshot_state(&s)
        };
        if result != baseline {
            errors.push(ValidationError {
                code: "DET-ACT-001",
                severity: Severity::Fatal,
                message: format!(
                    "apply_actions: run {} differs from run 0 (entity_count: {} vs {}, building_count: {} vs {})",
                    i, result.entity_count, baseline.entity_count,
                    result.building_positions.len(), baseline.building_positions.len()
                ),
                context: ErrorContext::default(),
            });
        }
    }
    errors
}

// ---------------------------------------------------------------------------
// DET-ACT-002: Strategic oracle determinism
// ---------------------------------------------------------------------------

/// Verify that `strategic_oracle` on the same observation produces
/// identical results across N runs.
pub fn check_strategic_oracle_determinism(
    obs: &BuildingObservation,
    n: usize,
) -> Vec<ValidationError> {
    assert_deterministic(
        "DET-ACT-002",
        "strategic_oracle",
        n,
        || {
            let actions = strategic_oracle(obs);
            snapshot_actions(&actions)
        },
    )
}

// ---------------------------------------------------------------------------
// DET-ACT-003: Structural oracle determinism
// ---------------------------------------------------------------------------

/// Verify that `structural_oracle` on the same observation and strategic
/// actions produces identical results across N runs.
pub fn check_structural_oracle_determinism(
    obs: &BuildingObservation,
    strategic_actions: &[BuildingAction],
    n: usize,
) -> Vec<ValidationError> {
    assert_deterministic(
        "DET-ACT-003",
        "structural_oracle",
        n,
        || {
            let actions = structural_oracle(obs, strategic_actions);
            snapshot_actions(&actions)
        },
    )
}

// ---------------------------------------------------------------------------
// DET-FEAT-001: Feature computation determinism
// ---------------------------------------------------------------------------

/// Verify that `compute_spatial_features` on the same state produces
/// identical results across N runs.
pub fn check_feature_determinism(
    state: &WorldState,
    settlement_id: u32,
    n: usize,
) -> Vec<ValidationError> {
    assert_deterministic(
        "DET-FEAT-001",
        "compute_spatial_features",
        n,
        || {
            let features = compute_spatial_features(state, settlement_id);
            snapshot_features(&features)
        },
    )
}

// ---------------------------------------------------------------------------
// Full determinism suite
// ---------------------------------------------------------------------------

/// Run all determinism checks.
pub fn run_determinism_suite(
    state: &WorldState,
    obs: &BuildingObservation,
    strategic_actions: &[BuildingAction],
    all_actions: &[BuildingAction],
    n: usize,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    errors.extend(check_action_determinism(state, all_actions, n));
    errors.extend(check_strategic_oracle_determinism(obs, n));
    errors.extend(check_structural_oracle_determinism(obs, strategic_actions, n));
    errors.extend(check_feature_determinism(state, obs.settlement_id, n));

    errors
}

// ---------------------------------------------------------------------------
// Snapshot types for comparison (PartialEq + Debug)
// ---------------------------------------------------------------------------

/// Lightweight snapshot of WorldState for comparison.
#[derive(Debug, PartialEq)]
struct StateSnapshot {
    entity_count: usize,
    alive_count: usize,
    building_positions: Vec<(u32, u16, u16)>, // (id, col, row)
    tick: u64,
}

fn snapshot_state(state: &WorldState) -> StateSnapshot {
    let mut building_positions: Vec<(u32, u16, u16)> = state
        .entities
        .iter()
        .filter(|e| e.alive && e.building.is_some())
        .map(|e| {
            let b = e.building.as_ref().unwrap();
            (e.id, b.grid_col, b.grid_row)
        })
        .collect();
    building_positions.sort();

    StateSnapshot {
        entity_count: state.entities.len(),
        alive_count: state.entities.iter().filter(|e| e.alive).count(),
        building_positions,
        tick: state.tick,
    }
}

/// Lightweight snapshot of actions for comparison.
#[derive(Debug, PartialEq)]
struct ActionSnapshot {
    count: usize,
    /// (decision_type_u8, priority_bits) for deterministic comparison
    entries: Vec<(u8, u32)>,
}

fn snapshot_actions(actions: &[BuildingAction]) -> ActionSnapshot {
    ActionSnapshot {
        count: actions.len(),
        entries: actions
            .iter()
            .map(|a| (a.decision_type as u8, a.priority.to_bits()))
            .collect(),
    }
}

/// Lightweight snapshot of features for comparison.
#[derive(Debug, PartialEq)]
struct FeatureSnapshot {
    wall_coverage_bits: u32,
    housing_pressure_bits: u32,
    connected_components: u8,
    evacuation_reachability_bits: u32,
    storage_utilization_bits: u32,
    total_garrison_strength_bits: u32,
}

fn snapshot_features(f: &SpatialFeatures) -> FeatureSnapshot {
    FeatureSnapshot {
        wall_coverage_bits: f.defensive.wall_coverage.to_bits(),
        housing_pressure_bits: f.population.housing_pressure.to_bits(),
        connected_components: f.connectivity.connected_components,
        evacuation_reachability_bits: f.connectivity.evacuation_reachability.to_bits(),
        storage_utilization_bits: f.economic.storage_utilization.to_bits(),
        total_garrison_strength_bits: f.garrison.total_garrison_strength.to_bits(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_helper_passes_on_constant() {
        let errors = assert_deterministic("TEST", "constant", 10, || 42);
        assert!(errors.is_empty());
    }

    #[test]
    fn deterministic_helper_detects_variance() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let counter = AtomicU32::new(0);
        let errors = assert_deterministic("TEST", "varying", 5, || {
            counter.fetch_add(1, Ordering::SeqCst)
        });
        // Runs 1-4 all differ from run 0
        assert_eq!(errors.len(), 4);
    }
}
