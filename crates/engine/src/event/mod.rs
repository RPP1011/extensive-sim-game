pub mod ring;
pub use crate::ids::EventId;
pub use ring::EventRing;

use crate::ids::AgentId;
use glam::Vec3;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Event {
    // Replayable subset
    AgentMoved    { agent_id: AgentId, from: Vec3, to: Vec3, tick: u32 },
    AgentAttacked { attacker: AgentId, target: AgentId, damage: f32, tick: u32 },
    AgentDied     { agent_id: AgentId, tick: u32 },
    AgentFled     { agent_id: AgentId, from: Vec3, to: Vec3, tick: u32 },
    AgentAte      { agent_id: AgentId, delta: f32, tick: u32 },
    AgentDrank    { agent_id: AgentId, delta: f32, tick: u32 },
    AgentRested   { agent_id: AgentId, delta: f32, tick: u32 },
    // Non-replayable (chronicle / prose side-channel placeholder)
    ChronicleEntry { tick: u32, template_id: u32 },
}

impl Event {
    pub fn tick(&self) -> u32 {
        match self {
            Event::AgentMoved    { tick, .. } |
            Event::AgentAttacked { tick, .. } |
            Event::AgentDied     { tick, .. } |
            Event::AgentFled     { tick, .. } |
            Event::AgentAte      { tick, .. } |
            Event::AgentDrank    { tick, .. } |
            Event::AgentRested   { tick, .. } |
            Event::ChronicleEntry{ tick, .. } => *tick,
        }
    }
    pub fn is_replayable(&self) -> bool {
        !matches!(self, Event::ChronicleEntry { .. })
    }
}
