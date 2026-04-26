//! Debug & trace runtime — engine primitive infrastructure for observability.
//!
//! Per `spec/runtime.md` §24. Six components, all host-side; GPU backends
//! trigger downloads on demand:
//!
//!   - `trace_mask` — records mask buffer state per tick
//!   - `causal_tree` — event causality presentation over `EventRing`
//!   - `tick_stepper` — per-phase pause-and-inspect harness
//!   - `tick_profile` — phase timing histogram
//!   - `agent_history` — per-agent state delta tracker
//!   - `repro_bundle` — snapshot + causal_tree + N-tick trace bundle
//!
//! Default-disabled. Activated via `DebugConfig` passed to the tick driver.

pub mod agent_history;
pub mod causal_tree;
pub mod repro_bundle;
pub mod tick_profile;
pub mod tick_stepper;
pub mod trace_mask;

/// Per-run debug+trace configuration. Default: all collectors disabled.
#[derive(Debug, Default, Clone)]
pub struct DebugConfig {
    pub trace_mask: bool,
    pub causal_tree: bool,
    pub tick_stepper: Option<tick_stepper::StepperHandle>,
    /// Phase timing histogram. When `Some`, each tick phase records enter/exit
    /// nanosecond samples into the shared `TickProfile`. The `Arc<Mutex<>>`
    /// wrapper allows the caller to inspect samples after `step()` returns
    /// without holding a borrow across the tick boundary.
    pub tick_profile: Option<std::sync::Arc<std::sync::Mutex<tick_profile::TickProfile>>>,
    pub agent_history: Option<agent_history::Filter>,
    pub repro_bundle: Option<std::path::PathBuf>,
}
