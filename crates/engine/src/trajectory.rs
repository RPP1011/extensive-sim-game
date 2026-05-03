//! Trajectory recording + playback via `safetensors`.
//!
//! A `TrajectoryWriter` preallocates `[t, n, 3]` position and `[t, n]` HP
//! buffers, then fills one row per recorded tick. `TrajectoryReader` loads the
//! resulting file back and exposes the dimensions for tests and tooling.

use std::path::Path;

use safetensors::tensor::TensorView;
use safetensors::{serialize_to_file, Dtype, SafeTensors};

use crate::state::SimState;

/// A single tick snapshot. Useful for tests / debug harnesses that want to
/// inspect per-tick data without going through the `safetensors` file.
#[derive(Debug, Clone)]
pub struct TickRecord {
    pub tick: u32,
    pub positions: Vec<[f32; 3]>,
    pub hp: Vec<f32>,
}

/// Preallocated buffer for a fixed `(n_ticks, n_agents)` trajectory.
///
/// Slots for agents that are not alive (or where `agents_alive()` yields fewer
/// than `n_agents` entries) are padded with zeros.
pub struct TrajectoryWriter {
    n_agents: usize,
    n_ticks:  usize,
    recorded: usize,
    // `[t, n, 3]` row-major flattened.
    positions: Vec<f32>,
    // `[t, n]` row-major flattened.
    hp:        Vec<f32>,
    // `[t]`
    ticks:     Vec<u32>,
}

impl TrajectoryWriter {
    pub fn new(n_agents: usize, n_ticks: usize) -> Self {
        Self {
            n_agents,
            n_ticks,
            recorded: 0,
            positions: vec![0.0; n_ticks * n_agents * 3],
            hp:        vec![0.0; n_ticks * n_agents],
            ticks:     vec![0; n_ticks],
        }
    }

    pub fn n_agents(&self) -> usize {
        self.n_agents
    }
    pub fn n_ticks(&self) -> usize {
        self.n_ticks
    }
    pub fn recorded(&self) -> usize {
        self.recorded
    }

    /// Capture the current tick into the preallocated buffer. Extra capacity
    /// beyond `n_ticks` is silently ignored to keep callers simple.
    pub fn record_tick(&mut self, state: &SimState) {
        if self.recorded >= self.n_ticks {
            return;
        }
        let t = self.recorded;
        let pos_base = t * self.n_agents * 3;
        let hp_base  = t * self.n_agents;

        // Zero the row (padding for empty slots), then fill live ones.
        for i in 0..self.n_agents {
            self.positions[pos_base + i * 3] = 0.0;
            self.positions[pos_base + i * 3 + 1] = 0.0;
            self.positions[pos_base + i * 3 + 2] = 0.0;
            self.hp[hp_base + i] = 0.0;
        }

        for (i, id) in state.agents_alive().take(self.n_agents).enumerate() {
            if let Some(p) = state.agent_pos(id) {
                self.positions[pos_base + i * 3]     = p.x;
                self.positions[pos_base + i * 3 + 1] = p.y;
                self.positions[pos_base + i * 3 + 2] = p.z;
            }
            if let Some(h) = state.agent_hp(id) {
                self.hp[hp_base + i] = h;
            }
        }

        self.ticks[t] = state.tick;
        self.recorded += 1;
    }

    /// Serialize the recorded trajectory to a `safetensors` file with three
    /// tensors: `positions` `[t, n, 3]` f32, `hp` `[t, n]` f32, `tick` `[t]` u32.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let t = self.n_ticks;
        let n = self.n_agents;

        let pos_bytes = f32_vec_to_bytes(&self.positions);
        let hp_bytes  = f32_vec_to_bytes(&self.hp);
        let tick_bytes = u32_vec_to_bytes(&self.ticks);

        let positions = TensorView::new(Dtype::F32, vec![t, n, 3], &pos_bytes)?;
        let hp        = TensorView::new(Dtype::F32, vec![t, n], &hp_bytes)?;
        let tick      = TensorView::new(Dtype::U32, vec![t], &tick_bytes)?;

        let tensors = vec![
            ("positions".to_string(), positions),
            ("hp".to_string(), hp),
            ("tick".to_string(), tick),
        ];

        serialize_to_file(tensors, &None, path.as_ref())?;
        Ok(())
    }
}

/// Read-only view of a trajectory file. The current implementation eagerly
/// loads the file bytes and parses the header; tensor payloads are held as
/// an owned byte buffer for future accessors.
pub struct TrajectoryReader {
    n_ticks:  usize,
    n_agents: usize,
    _buffer: Vec<u8>,
}

impl TrajectoryReader {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer = std::fs::read(path.as_ref())?;
        let (n_ticks, n_agents) = {
            let st = SafeTensors::deserialize(&buffer)?;
            let positions = st.tensor("positions")?;
            let shape = positions.shape();
            if shape.len() != 3 {
                return Err(format!(
                    "positions tensor must be rank-3 [t, n, 3], got shape {:?}",
                    shape
                )
                .into());
            }
            (shape[0], shape[1])
        };
        Ok(Self { n_ticks, n_agents, _buffer: buffer })
    }

    pub fn n_ticks(&self) -> usize {
        self.n_ticks
    }
    pub fn n_agents(&self) -> usize {
        self.n_agents
    }
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn u32_vec_to_bytes(v: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
