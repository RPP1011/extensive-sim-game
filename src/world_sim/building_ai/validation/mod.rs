//! Scenario validation system for building AI training data.
//!
//! Catches every class of data corruption bug BEFORE training. No bad
//! (observation, action) pair should ever reach the model.
//!
//! All validation functions return `Vec<ValidationError>` (empty = pass).
//! No panics, no silent skips. Every check has a unique error code for
//! automated triage.

pub mod world_state;
pub mod action;
pub mod determinism;
pub mod features;
pub mod layout;
pub mod memory;
pub mod probes;

use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single validation failure with a machine-readable code and human-readable message.
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Machine-readable code, e.g. "WS-XREF-001".
    pub code: &'static str,
    /// Severity of this error.
    pub severity: Severity,
    /// Human-readable description.
    pub message: String,
    /// Contextual info for triage.
    pub context: ErrorContext,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {:?}: {}", self.code, self.severity, self.message)
    }
}

/// How severe is this validation failure?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Pair must be rejected. Data is provably corrupt.
    Fatal,
    /// Pair is suspicious. Log and flag for manual review.
    Warning,
    /// Informational. No action needed.
    Info,
}

/// Contextual info attached to a validation error for triage.
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    pub scenario_id: Option<u64>,
    pub entity_id: Option<u32>,
    pub grid_cell: Option<(u16, u16)>,
    pub field: Option<String>,
}

// ---------------------------------------------------------------------------
// Top-level validation entry point
// ---------------------------------------------------------------------------

use super::types::{BuildingAction, BuildingObservation};
use crate::world_sim::state::WorldState;

/// Run all applicable validations on a (state, observation, actions) triple.
///
/// Returns all errors found. An empty vec means the triple is clean.
/// Fatal errors mean the pair must be rejected from the dataset.
pub fn validate_all(
    state: &WorldState,
    obs: &BuildingObservation,
    actions: &[BuildingAction],
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // WorldState consistency
    errors.extend(world_state::validate_world_state(state));

    // Memory buffer integrity
    errors.extend(memory::validate_memory(&obs.memory, state));

    // Spatial feature sanity
    errors.extend(features::validate_features(&obs.spatial, state, obs.settlement_id));

    // Action pre-application checks
    errors.extend(action::validate_action_batch(state, obs, actions));

    errors
}

/// Run pre-check + apply + post-check cycle. Returns errors from all phases.
///
/// `apply_fn` is the action application function (typically `scoring::apply_actions`).
pub fn validate_with_application(
    state: &WorldState,
    obs: &BuildingObservation,
    actions: &[BuildingAction],
    apply_fn: impl FnOnce(&mut WorldState, &[BuildingAction]),
) -> Vec<ValidationError> {
    let mut errors = validate_all(state, obs, actions);

    // Apply actions to a clone and check post-conditions
    let mut post_state = state.clone();
    apply_fn(&mut post_state, actions);
    errors.extend(action::validate_post_action(state, &post_state, actions));

    // Re-validate the post-action state
    let post_errors = world_state::validate_world_state(&post_state);
    for e in post_errors {
        // Tag post-action errors so they can be distinguished
        errors.push(ValidationError {
            code: e.code,
            severity: e.severity,
            message: format!("[post-action] {}", e.message),
            context: e.context,
        });
    }

    errors
}

/// Check whether any errors are fatal.
pub fn has_fatal(errors: &[ValidationError]) -> bool {
    errors.iter().any(|e| e.severity == Severity::Fatal)
}

/// Filter to fatal errors only.
pub fn fatal_errors(errors: &[ValidationError]) -> Vec<&ValidationError> {
    errors.iter().filter(|e| e.severity == Severity::Fatal).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_errors_means_pass() {
        let errors: Vec<ValidationError> = Vec::new();
        assert!(!has_fatal(&errors));
        assert!(fatal_errors(&errors).is_empty());
    }
}
