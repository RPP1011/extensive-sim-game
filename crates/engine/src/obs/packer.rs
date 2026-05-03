//! `FeatureSource` trait + `ObsPacker` composer.

use crate::ids::AgentId;
use crate::state::SimState;

/// One feature extractor. Reads agent-scoped slice of `SimState` and writes
/// a fixed-width f32 window into `out`. Implementations MUST write exactly
/// `self.dim()` slots; the packer hands them a matching slice.
pub trait FeatureSource: Send + Sync {
    /// Number of f32 slots this source contributes per agent.
    fn dim(&self) -> usize;
    /// Write `self.dim()` f32s into `out` for the given agent.
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]);
}

/// Composes feature sources into a single row-major packer.
pub struct ObsPacker {
    sources: Vec<Box<dyn FeatureSource>>,
    feature_dim: usize,
}

impl ObsPacker {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            feature_dim: 0,
        }
    }

    pub fn register(&mut self, source: Box<dyn FeatureSource>) {
        self.feature_dim += source.dim();
        self.sources.push(source);
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Pack `[agents.len() × feature_dim]` f32 row-major into `out`.
    ///
    /// Panics if `out.len() < agents.len() * feature_dim`.
    pub fn pack_batch(&self, state: &SimState, agents: &[AgentId], out: &mut [f32]) {
        let need = agents.len() * self.feature_dim;
        assert!(
            out.len() >= need,
            "obs buffer too small: have {}, need {}",
            out.len(),
            need,
        );
        for (row, &agent) in agents.iter().enumerate() {
            let row_start = row * self.feature_dim;
            let mut col = 0;
            for source in &self.sources {
                let d = source.dim();
                source.pack(state, agent, &mut out[row_start + col..row_start + col + d]);
                col += d;
            }
        }
    }
}

impl Default for ObsPacker {
    fn default() -> Self {
        Self::new()
    }
}
