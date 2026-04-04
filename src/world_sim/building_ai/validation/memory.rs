//! Construction memory buffer integrity checks (MEM-* error codes).
//!
//! Validates ring buffer invariants, event field ranges, and consistency
//! with the current world state tick.

use super::{ErrorContext, Severity, ValidationError};
use crate::world_sim::building_ai::types::{
    bi_tags, ConstructionMemory,
};
use crate::world_sim::state::WorldState;

/// Run all memory buffer integrity checks.
pub fn validate_memory(
    memory: &ConstructionMemory,
    state: &WorldState,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    errors.extend(check_ring_buffer_invariants(memory));
    errors.extend(check_event_fields(memory, state));
    errors.extend(check_pattern_fields(memory));
    errors.extend(check_lesson_fields(memory));
    errors
}

// ---------------------------------------------------------------------------
// MEM-001..003: Ring buffer structural invariants
// ---------------------------------------------------------------------------

fn check_ring_buffer_invariants(memory: &ConstructionMemory) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // MEM-001: items.len() <= capacity
    if memory.short_term.items.len() > memory.short_term.capacity {
        errors.push(ValidationError {
            code: "MEM-001",
            severity: Severity::Fatal,
            message: format!(
                "Short-term buffer overflow: len {} > capacity {}",
                memory.short_term.items.len(),
                memory.short_term.capacity
            ),
            context: ErrorContext {
                field: Some("short_term".into()),
                ..Default::default()
            },
        });
    }
    if memory.medium_term.items.len() > memory.medium_term.capacity {
        errors.push(ValidationError {
            code: "MEM-001",
            severity: Severity::Fatal,
            message: format!(
                "Medium-term buffer overflow: len {} > capacity {}",
                memory.medium_term.items.len(),
                memory.medium_term.capacity
            ),
            context: ErrorContext {
                field: Some("medium_term".into()),
                ..Default::default()
            },
        });
    }
    if memory.long_term.items.len() > memory.long_term.capacity {
        errors.push(ValidationError {
            code: "MEM-001",
            severity: Severity::Fatal,
            message: format!(
                "Long-term buffer overflow: len {} > capacity {}",
                memory.long_term.items.len(),
                memory.long_term.capacity
            ),
            context: ErrorContext {
                field: Some("long_term".into()),
                ..Default::default()
            },
        });
    }

    // MEM-002: head < capacity
    if memory.short_term.capacity > 0 && memory.short_term.head >= memory.short_term.capacity {
        errors.push(ValidationError {
            code: "MEM-002",
            severity: Severity::Fatal,
            message: format!(
                "Short-term head {} >= capacity {}",
                memory.short_term.head, memory.short_term.capacity
            ),
            context: ErrorContext {
                field: Some("short_term.head".into()),
                ..Default::default()
            },
        });
    }
    if memory.medium_term.capacity > 0 && memory.medium_term.head >= memory.medium_term.capacity {
        errors.push(ValidationError {
            code: "MEM-002",
            severity: Severity::Fatal,
            message: format!(
                "Medium-term head {} >= capacity {}",
                memory.medium_term.head, memory.medium_term.capacity
            ),
            context: ErrorContext {
                field: Some("medium_term.head".into()),
                ..Default::default()
            },
        });
    }
    if memory.long_term.capacity > 0 && memory.long_term.head >= memory.long_term.capacity {
        errors.push(ValidationError {
            code: "MEM-002",
            severity: Severity::Fatal,
            message: format!(
                "Long-term head {} >= capacity {}",
                memory.long_term.head, memory.long_term.capacity
            ),
            context: ErrorContext {
                field: Some("long_term.head".into()),
                ..Default::default()
            },
        });
    }

    // MEM-003: expected capacities
    if memory.short_term.capacity != 64 {
        errors.push(ValidationError {
            code: "MEM-003",
            severity: Severity::Warning,
            message: format!(
                "Short-term capacity is {} (expected 64)",
                memory.short_term.capacity
            ),
            context: ErrorContext {
                field: Some("short_term.capacity".into()),
                ..Default::default()
            },
        });
    }
    if memory.medium_term.capacity != 256 {
        errors.push(ValidationError {
            code: "MEM-003",
            severity: Severity::Warning,
            message: format!(
                "Medium-term capacity is {} (expected 256)",
                memory.medium_term.capacity
            ),
            context: ErrorContext {
                field: Some("medium_term.capacity".into()),
                ..Default::default()
            },
        });
    }
    if memory.long_term.capacity != 64 {
        errors.push(ValidationError {
            code: "MEM-003",
            severity: Severity::Warning,
            message: format!(
                "Long-term capacity is {} (expected 64)",
                memory.long_term.capacity
            ),
            context: ErrorContext {
                field: Some("long_term.capacity".into()),
                ..Default::default()
            },
        });
    }

    errors
}

// ---------------------------------------------------------------------------
// MEM-004, MEM-009, MEM-010: Event field checks
// ---------------------------------------------------------------------------

fn check_event_fields(
    memory: &ConstructionMemory,
    state: &WorldState,
) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for (i, event) in memory.short_term.iter().enumerate() {
        // MEM-004: severity in [0.0, 1.0]
        if event.severity < 0.0 || event.severity > 1.0 {
            errors.push(ValidationError {
                code: "MEM-004",
                severity: Severity::Warning,
                message: format!(
                    "short_term[{}].severity = {} (expected [0.0, 1.0])",
                    i, event.severity
                ),
                context: ErrorContext {
                    field: Some(format!("short_term[{}].severity", i)),
                    ..Default::default()
                },
            });
        }

        // MEM-009: tick not in the future
        if event.tick > state.tick {
            errors.push(ValidationError {
                code: "MEM-009",
                severity: Severity::Warning,
                message: format!(
                    "short_term[{}].tick = {} > state.tick = {} (future event)",
                    i, event.tick, state.tick
                ),
                context: ErrorContext {
                    field: Some(format!("short_term[{}].tick", i)),
                    ..Default::default()
                },
            });
        }

        // MEM-010: location within bounds (checked against all grids)
        // We check for unreasonably large coordinates as a heuristic
        if event.location.0 > 512 || event.location.1 > 512 {
            errors.push(ValidationError {
                code: "MEM-010",
                severity: Severity::Warning,
                message: format!(
                    "short_term[{}].location ({}, {}) seems out of bounds",
                    i, event.location.0, event.location.1
                ),
                context: ErrorContext {
                    grid_cell: Some(event.location),
                    field: Some(format!("short_term[{}].location", i)),
                    ..Default::default()
                },
            });
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// MEM-005, MEM-008: Aggregated pattern checks
// ---------------------------------------------------------------------------

fn check_pattern_fields(memory: &ConstructionMemory) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for (i, pattern) in memory.medium_term.iter().enumerate() {
        // MEM-005: importance is finite
        if !pattern.importance.is_finite() {
            errors.push(ValidationError {
                code: "MEM-005",
                severity: Severity::Fatal,
                message: format!(
                    "medium_term[{}].importance = {} (non-finite)",
                    i, pattern.importance
                ),
                context: ErrorContext {
                    field: Some(format!("medium_term[{}].importance", i)),
                    ..Default::default()
                },
            });
        }

        // MEM-008: first_tick <= last_tick
        if pattern.first_tick > pattern.last_tick {
            errors.push(ValidationError {
                code: "MEM-008",
                severity: Severity::Warning,
                message: format!(
                    "medium_term[{}]: first_tick {} > last_tick {} (inverted range)",
                    i, pattern.first_tick, pattern.last_tick
                ),
                context: ErrorContext {
                    field: Some(format!("medium_term[{}]", i)),
                    ..Default::default()
                },
            });
        }

        // MEM-005 extended: mean_severity should be reasonable
        if !pattern.mean_severity.is_finite() {
            errors.push(ValidationError {
                code: "MEM-005",
                severity: Severity::Fatal,
                message: format!(
                    "medium_term[{}].mean_severity = {} (non-finite)",
                    i, pattern.mean_severity
                ),
                context: ErrorContext {
                    field: Some(format!("medium_term[{}].mean_severity", i)),
                    ..Default::default()
                },
            });
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// MEM-006, MEM-007: Structural lesson checks
// ---------------------------------------------------------------------------

fn check_lesson_fields(memory: &ConstructionMemory) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Known lesson tags from bi_tags
    let known_tags: &[u32] = &[
        bi_tags::WALL_TOO_LOW,
        bi_tags::WALL_TOO_THIN,
        bi_tags::WOOD_BURNS,
        bi_tags::FLOOD_LOW_GROUND,
        bi_tags::CHOKEPOINT_EXPLOITABLE,
        bi_tags::GARRISON_COMPENSATES,
        bi_tags::THREAT_PROXIMITY,
        bi_tags::GARRISON_SYNERGY,
        bi_tags::RESOURCE_SCARCITY,
        bi_tags::HOUSING_PRESSURE,
        bi_tags::FIRE_RECOVERY,
        bi_tags::FLOOD_PREVENTION,
        bi_tags::JUMP_COUNTER,
        bi_tags::SIEGE_COUNTER,
        bi_tags::SPECIALIST_ACCESS,
        bi_tags::LEADER_PROTECTION,
        bi_tags::UPGRADE_PATH,
        bi_tags::SEASONAL_PREP,
        bi_tags::TERRAIN_ADAPT,
    ];

    for (i, lesson) in memory.long_term.iter().enumerate() {
        // MEM-006: confidence in [0.0, 1.0]
        if lesson.confidence < 0.0 || lesson.confidence > 1.0 {
            errors.push(ValidationError {
                code: "MEM-006",
                severity: Severity::Warning,
                message: format!(
                    "long_term[{}].confidence = {} (expected [0.0, 1.0])",
                    i, lesson.confidence
                ),
                context: ErrorContext {
                    field: Some(format!("long_term[{}].confidence", i)),
                    ..Default::default()
                },
            });
        }

        // MEM-007: lesson_tag is a known bi_tags constant
        if !known_tags.contains(&lesson.lesson_tag) {
            errors.push(ValidationError {
                code: "MEM-007",
                severity: Severity::Warning,
                message: format!(
                    "long_term[{}].lesson_tag = {} (unknown tag)",
                    i, lesson.lesson_tag
                ),
                context: ErrorContext {
                    field: Some(format!("long_term[{}].lesson_tag", i)),
                    ..Default::default()
                },
            });
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::building_ai::types::ConstructionMemory;

    #[test]
    fn fresh_memory_passes() {
        let memory = ConstructionMemory::new();
        let state = WorldState::new(42);
        let errors = validate_memory(&memory, &state);
        let fatals: Vec<_> = errors.iter().filter(|e| e.severity == Severity::Fatal).collect();
        assert!(fatals.is_empty(), "Unexpected fatals: {:?}", fatals);
    }

    #[test]
    fn wrong_capacity_warned() {
        let mut memory = ConstructionMemory::new();
        memory.short_term.capacity = 32; // wrong
        let state = WorldState::new(42);
        let errors = validate_memory(&memory, &state);
        assert!(
            errors.iter().any(|e| e.code == "MEM-003"),
            "Expected MEM-003 warning"
        );
    }
}
