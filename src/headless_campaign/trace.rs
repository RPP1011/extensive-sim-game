//! Campaign trace recording and playback.
//!
//! Records periodic `CampaignState` snapshots (keyframes) and all `WorldEvent`s
//! during a headless campaign run. Traces can be loaded into the UI for scrub/
//! play/pause replay and fork-to-play.

use serde::{Deserialize, Serialize};

use super::actions::{CampaignAction, CampaignStepResult, WorldEvent};
use super::state::{CampaignOutcome, CampaignState};
use super::step::step_campaign;

// ---------------------------------------------------------------------------
// Trace data structures
// ---------------------------------------------------------------------------

/// A recorded campaign trace for replay in the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignTrace {
    pub seed: u64,
    pub outcome: Option<CampaignOutcome>,
    /// State snapshots taken every `snapshot_interval` ticks (keyframes).
    pub snapshots: Vec<TraceSnapshot>,
    /// All events across the campaign, in tick order.
    pub events: Vec<TraceEvent>,
    /// Total ticks the campaign ran.
    pub total_ticks: u64,
    /// Actions taken by the policy at each decision point.
    pub actions: Vec<TraceAction>,
}

/// A full state snapshot at a specific tick.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceSnapshot {
    pub tick: u64,
    pub state: CampaignState,
}

/// A world event with its tick timestamp.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceEvent {
    pub tick: u64,
    pub event: WorldEvent,
}

/// An action taken at a specific tick.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceAction {
    pub tick: u64,
    pub action: CampaignAction,
}

// ---------------------------------------------------------------------------
// Trace recorder
// ---------------------------------------------------------------------------

/// Wraps `step_campaign` to record snapshots and events during a run.
pub struct TraceRecorder {
    trace: CampaignTrace,
    snapshot_interval: u64,
}

impl TraceRecorder {
    /// Create a new recorder.
    ///
    /// `snapshot_interval`: how often (in ticks) to take a full state snapshot.
    /// Lower values = larger trace files but faster scrubbing. 100 is a good default.
    pub fn new(seed: u64, snapshot_interval: u64) -> Self {
        Self {
            trace: CampaignTrace {
                seed,
                outcome: None,
                snapshots: Vec::new(),
                events: Vec::new(),
                actions: Vec::new(),
                total_ticks: 0,
            },
            snapshot_interval,
        }
    }

    /// Step the campaign forward and record the results.
    pub fn step(
        &mut self,
        state: &mut CampaignState,
        action: Option<CampaignAction>,
    ) -> CampaignStepResult {
        // Record action if present
        if let Some(ref a) = action {
            self.trace.actions.push(TraceAction {
                tick: state.tick,
                action: a.clone(),
            });
        }

        // Take snapshot at interval (before stepping, so tick=0 gets captured)
        if state.tick % self.snapshot_interval == 0 {
            self.trace.snapshots.push(TraceSnapshot {
                tick: state.tick,
                state: state.clone(),
            });
        }

        // Step
        let result = step_campaign(state, action);

        // Record events
        for event in &result.events {
            self.trace.events.push(TraceEvent {
                tick: state.tick,
                event: event.clone(),
            });
        }

        self.trace.total_ticks = state.tick;

        result
    }

    /// Finalize the trace with the campaign outcome.
    pub fn finish(mut self, outcome: Option<CampaignOutcome>) -> CampaignTrace {
        self.trace.outcome = outcome;
        self.trace
    }

    /// Access the trace being built (for inspection).
    pub fn trace(&self) -> &CampaignTrace {
        &self.trace
    }
}

// ---------------------------------------------------------------------------
// Trace I/O
// ---------------------------------------------------------------------------

impl CampaignTrace {
    /// Save trace to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load trace from a JSON file.
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Get the state at a specific tick by finding the nearest snapshot
    /// and replaying forward.
    pub fn state_at_tick(&self, tick: u64) -> Option<CampaignState> {
        if self.snapshots.is_empty() {
            return None;
        }

        // Binary search for nearest snapshot <= tick
        let idx = match self.snapshots.binary_search_by_key(&tick, |s| s.tick) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else {
                    i - 1
                }
            }
        };

        let snapshot = &self.snapshots[idx];
        let mut state = snapshot.state.clone();

        // Replay forward from snapshot to target tick
        if state.tick < tick {
            // Find actions in the range [snapshot.tick, tick)
            let actions_in_range: Vec<&TraceAction> = self
                .actions
                .iter()
                .filter(|a| a.tick >= snapshot.tick && a.tick < tick)
                .collect();

            while state.tick < tick {
                // Check if there's an action at this tick
                let action = actions_in_range
                    .iter()
                    .find(|a| a.tick == state.tick)
                    .map(|a| a.action.clone());

                step_campaign(&mut state, action);
            }
        }

        Some(state)
    }

    /// Get events in a tick range [start, end).
    pub fn events_in_range(&self, start_tick: u64, end_tick: u64) -> Vec<&TraceEvent> {
        self.events
            .iter()
            .filter(|e| e.tick >= start_tick && e.tick < end_tick)
            .collect()
    }

    /// Format a tick as human-readable game time (MM:SS).
    pub fn format_tick(tick: u64) -> String {
        let total_secs = tick as f64 * 0.1; // 100ms per tick
        let mins = (total_secs / 60.0) as u64;
        let secs = (total_secs % 60.0) as u64;
        format!("{:02}:{:02}", mins, secs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::CampaignState;

    #[test]
    fn test_trace_recorder_captures_snapshots() {
        let seed = 42;
        let mut state = CampaignState::default_test_campaign(seed);
        let mut recorder = TraceRecorder::new(seed, 50);

        for _ in 0..200 {
            recorder.step(&mut state, None);
        }

        let trace = recorder.finish(None);
        // 200 ticks / 50 interval = 4 snapshots (at ticks 0, 50, 100, 150)
        assert!(
            trace.snapshots.len() >= 4,
            "Expected >=4 snapshots, got {}",
            trace.snapshots.len()
        );
        assert_eq!(trace.snapshots[0].tick, 0);
        assert_eq!(trace.snapshots[1].tick, 50);
    }

    #[test]
    fn test_trace_recorder_captures_events() {
        let seed = 42;
        let mut state = CampaignState::default_test_campaign(seed);
        let mut recorder = TraceRecorder::new(seed, 100);

        for _ in 0..2000 {
            recorder.step(&mut state, None);
        }

        let trace = recorder.finish(None);
        // Should have captured some quest generation events
        assert!(
            !trace.events.is_empty(),
            "Expected some events after 2000 ticks"
        );
    }

    #[test]
    fn test_state_at_tick() {
        let seed = 42;
        let mut state = CampaignState::default_test_campaign(seed);
        let mut recorder = TraceRecorder::new(seed, 50);

        for _ in 0..200 {
            recorder.step(&mut state, None);
        }

        let trace = recorder.finish(None);

        // Get state at tick 75 (between snapshots 50 and 100)
        let reconstructed = trace.state_at_tick(75).expect("Should reconstruct");
        assert_eq!(reconstructed.tick, 75);
    }

    #[test]
    fn test_trace_save_load_roundtrip() {
        let seed = 42;
        let mut state = CampaignState::default_test_campaign(seed);
        let mut recorder = TraceRecorder::new(seed, 100);

        for _ in 0..100 {
            recorder.step(&mut state, None);
        }

        let trace = recorder.finish(None);

        // Save
        let dir = std::env::temp_dir().join("chimera_test_traces");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test_trace.json");
        trace.save_to_file(&path).expect("Save should succeed");

        // Load
        let loaded = CampaignTrace::load_from_file(&path).expect("Load should succeed");
        assert_eq!(loaded.seed, trace.seed);
        assert_eq!(loaded.snapshots.len(), trace.snapshots.len());
        assert_eq!(loaded.events.len(), trace.events.len());
        assert_eq!(loaded.total_ticks, trace.total_ticks);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_format_tick() {
        assert_eq!(CampaignTrace::format_tick(0), "00:00");
        assert_eq!(CampaignTrace::format_tick(600), "01:00");
        assert_eq!(CampaignTrace::format_tick(6000), "10:00");
        assert_eq!(CampaignTrace::format_tick(305), "00:30");
    }
}
