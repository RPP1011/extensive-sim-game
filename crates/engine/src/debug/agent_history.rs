//! Per-agent state delta tracker — stub (populated in Plan 4 Task 6).

/// Filter controlling which agents and how many ticks to retain.
#[derive(Debug, Default, Clone)]
pub struct Filter {
    pub max_ticks: usize,
}
