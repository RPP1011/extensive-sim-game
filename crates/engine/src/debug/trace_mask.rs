//! Per-tick mask buffer snapshot collector.
//!
//! Records each tick's `MaskBuffer` state by deep-copying the bit matrix.
//! Memory: O(n_agents × n_kinds × ticks_collected). At 200k agents × 18
//! kinds × 100 ticks = 360 MB; bound the collector to a configurable
//! max_ticks (default 1000) and ring around.

use crate::mask::MaskBuffer;
use std::collections::VecDeque;

pub struct TraceMaskCollector {
    snapshots: VecDeque<MaskSnapshot>,
    max_ticks: usize,
}

#[derive(Clone)]
pub struct MaskSnapshot {
    pub tick: u32,
    /// One bit per (agent, kind). Length = n_agents * n_kinds.
    pub bits: Vec<bool>,
    pub n_agents: u32,
    pub n_kinds: u32,
}

impl TraceMaskCollector {
    pub fn new(max_ticks: usize) -> Self {
        Self { snapshots: VecDeque::with_capacity(max_ticks), max_ticks }
    }

    /// Record one tick. Called from `step` if `DebugConfig::trace_mask` is true.
    pub fn record(&mut self, tick: u32, mask: &MaskBuffer) {
        if self.snapshots.len() == self.max_ticks {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(MaskSnapshot {
            tick,
            bits: mask.bits().to_vec(),
            n_agents: mask.n_agents(),
            n_kinds: mask.n_kinds(),
        });
    }

    pub fn at_tick(&self, tick: u32) -> Option<&MaskSnapshot> {
        self.snapshots.iter().find(|s| s.tick == tick)
    }

    pub fn all(&self) -> impl Iterator<Item = &MaskSnapshot> {
        self.snapshots.iter()
    }
}
