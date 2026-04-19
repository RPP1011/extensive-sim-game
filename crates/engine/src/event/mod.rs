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
    // Event-only micros — the engine emits the typed event; actual effects are
    // wired by compiler-registered cascade handlers in later plans.
    AgentCast           { agent_id: AgentId, ability_idx: u8, tick: u32 },
    AgentUsedItem       { agent_id: AgentId, item_slot: u8, tick: u32 },
    AgentHarvested      { agent_id: AgentId, resource: u64, tick: u32 },
    AgentPlacedTile     { agent_id: AgentId, where_pos: Vec3, kind_tag: u32, tick: u32 },
    AgentPlacedVoxel    { agent_id: AgentId, where_pos: Vec3, mat_tag: u32,  tick: u32 },
    AgentHarvestedVoxel { agent_id: AgentId, where_pos: Vec3, tick: u32 },
    AgentConversed      { agent_id: AgentId, partner: AgentId, tick: u32 },
    AgentSharedStory    { agent_id: AgentId, topic: u64, tick: u32 },
    AgentCommunicated   { speaker: AgentId, recipient: AgentId, fact_ref: u64, tick: u32 },
    InformationRequested{ asker: AgentId, target: AgentId, query: u64, tick: u32 },
    AgentRemembered     { agent_id: AgentId, subject: u64, tick: u32 },
    // Non-replayable (chronicle / prose side-channel placeholder)
    ChronicleEntry { tick: u32, template_id: u32 },
}

impl Event {
    pub fn tick(&self) -> u32 {
        match self {
            Event::AgentMoved           { tick, .. } |
            Event::AgentAttacked        { tick, .. } |
            Event::AgentDied            { tick, .. } |
            Event::AgentFled            { tick, .. } |
            Event::AgentAte             { tick, .. } |
            Event::AgentDrank           { tick, .. } |
            Event::AgentRested          { tick, .. } |
            Event::AgentCast            { tick, .. } |
            Event::AgentUsedItem        { tick, .. } |
            Event::AgentHarvested       { tick, .. } |
            Event::AgentPlacedTile      { tick, .. } |
            Event::AgentPlacedVoxel     { tick, .. } |
            Event::AgentHarvestedVoxel  { tick, .. } |
            Event::AgentConversed       { tick, .. } |
            Event::AgentSharedStory     { tick, .. } |
            Event::AgentCommunicated    { tick, .. } |
            Event::InformationRequested { tick, .. } |
            Event::AgentRemembered      { tick, .. } |
            Event::ChronicleEntry       { tick, .. } => *tick,
        }
    }
    pub fn is_replayable(&self) -> bool {
        !matches!(self, Event::ChronicleEntry { .. })
    }
}
