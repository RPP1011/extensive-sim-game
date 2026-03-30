//! Trace recording and playback for the world simulation.
//!
//! `WorldSimTraceRecorder` wraps a `WorldSim` and captures periodic snapshots
//! plus all chronicle entries. The resulting `WorldSimTrace` can be saved/loaded
//! and replayed at arbitrary ticks via `state_at_tick()`.

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::runtime::WorldSim;
use super::state::{ChronicleEntry, WorldState};

// ---------------------------------------------------------------------------
// WorldSimTrace — recorded run data
// ---------------------------------------------------------------------------

/// A recorded world simulation run: periodic state snapshots + full chronicle.
#[derive(Clone, Serialize, Deserialize)]
pub struct WorldSimTrace {
    /// RNG seed the simulation was initialized with.
    pub seed: u64,
    /// Periodic snapshots: `(tick, full_state)`.
    pub snapshots: Vec<(u64, WorldState)>,
    /// All chronicle entries produced during the run.
    pub chronicle_log: Vec<ChronicleEntry>,
    /// Total ticks simulated.
    pub total_ticks: u64,
}

impl WorldSimTrace {
    /// Reconstruct the `WorldState` at an arbitrary tick by finding the nearest
    /// snapshot <= `tick` and replaying forward with a temporary `WorldSim`.
    ///
    /// Returns `None` if `tick` exceeds `total_ticks` or there are no snapshots.
    pub fn state_at_tick(&self, tick: u64) -> Option<WorldState> {
        if self.snapshots.is_empty() || tick > self.total_ticks {
            return None;
        }

        // Binary search for the largest snapshot tick <= target tick.
        let idx = match self.snapshots.binary_search_by_key(&tick, |(t, _)| *t) {
            Ok(i) => i,
            Err(0) => return None, // tick is before first snapshot
            Err(i) => i - 1,
        };

        let (snap_tick, snap_state) = &self.snapshots[idx];
        if *snap_tick == tick {
            return Some(snap_state.clone());
        }

        // Replay forward from the snapshot to the target tick.
        let mut sim = WorldSim::new(snap_state.clone());
        let ticks_remaining = tick - snap_tick;
        for _ in 0..ticks_remaining {
            sim.tick();
        }

        Some(sim.state().clone())
    }

    /// Serialize the trace to a JSON file.
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(Path::new(path))?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Deserialize a trace from a JSON file.
    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(Path::new(path))?;
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// ---------------------------------------------------------------------------
// WorldSimTraceRecorder — wraps a WorldSim to capture trace data
// ---------------------------------------------------------------------------

/// Records a `WorldSimTrace` by wrapping a `WorldSim` and capturing snapshots
/// at a fixed interval, plus all new chronicle entries each tick.
pub struct WorldSimTraceRecorder<'a> {
    sim: &'a mut WorldSim,
    snapshot_interval: u64,
    trace: WorldSimTrace,
    /// Number of chronicle entries already captured (to detect new ones).
    last_chronicle_len: usize,
}

impl<'a> WorldSimTraceRecorder<'a> {
    /// Create a new recorder wrapping the given simulation.
    ///
    /// `snapshot_interval` controls how often a full state snapshot is captured
    /// (e.g. every 100 ticks). The initial state (tick 0) is always captured.
    pub fn new(sim: &'a mut WorldSim, snapshot_interval: u64) -> Self {
        let seed = sim.state().rng_state;
        let initial_chronicle_len = sim.state().chronicle.len();

        // Capture the initial snapshot.
        let initial_snapshot = (sim.state().tick, sim.state().clone());

        // Copy any pre-existing chronicle entries.
        let initial_chronicle: Vec<ChronicleEntry> = sim.state().chronicle.clone();

        WorldSimTraceRecorder {
            sim,
            snapshot_interval: snapshot_interval.max(1),
            trace: WorldSimTrace {
                seed,
                snapshots: vec![initial_snapshot],
                chronicle_log: initial_chronicle,
                total_ticks: 0,
            },
            last_chronicle_len: initial_chronicle_len,
        }
    }

    /// Advance the simulation by one tick, capturing snapshots and chronicle
    /// entries as appropriate.
    pub fn tick(&mut self) {
        self.sim.tick();

        let current_tick = self.sim.state().tick;

        // Capture snapshot if on interval boundary.
        if current_tick % self.snapshot_interval == 0 {
            self.trace.snapshots.push((current_tick, self.sim.state().clone()));
        }

        // Capture new chronicle entries added this tick.
        let chronicle = &self.sim.state().chronicle;
        if chronicle.len() > self.last_chronicle_len {
            for entry in &chronicle[self.last_chronicle_len..] {
                self.trace.chronicle_log.push(entry.clone());
            }
            self.last_chronicle_len = chronicle.len();
        }

        self.trace.total_ticks = current_tick;
    }

    /// Consume the recorder and return the completed trace.
    pub fn finish(self) -> WorldSimTrace {
        self.trace
    }

    /// Access the underlying simulation state.
    pub fn state(&self) -> &WorldState {
        self.sim.state()
    }
}
