use crate::ids::AgentId;
use engine_data::types::{QuestCategory, Resolution};
use smallvec::SmallVec;

#[derive(Clone, Debug, PartialEq)]
pub struct Quest {
    pub seq:         u32,
    pub poster:      Option<AgentId>,
    pub category:    QuestCategory,
    pub resolution:  Resolution,
    pub acceptors:   SmallVec<[AgentId; 4]>,
    pub posted_tick: u32,
}

impl Quest {
    /// Minimal-shape constructor. Domain code overwrites fields after alloc.
    pub fn stub(seq: u32) -> Self {
        Self {
            seq,
            poster:      None,
            category:    QuestCategory::Physical,
            resolution:  Resolution::HighestBid,
            acceptors:   SmallVec::new(),
            posted_tick: 0,
        }
    }
}
