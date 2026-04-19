pub mod agent;
pub mod entity_pool;

use crate::channel::ChannelSet;
use crate::creature::{Capabilities, CreatureType};
use crate::ids::AgentId;
pub use agent::{AgentSpawn, MovementMode};
use entity_pool::{AgentPoolOps, AgentSlotPool};
use glam::Vec3;

pub struct SimState {
    pub tick: u32,
    pub seed: u64,
    pool:     AgentSlotPool,

    // Hot SoA — read/written every tick by observation / mask / step kernels.
    hot_pos:           Vec<Vec3>,
    hot_hp:            Vec<f32>,
    hot_max_hp:        Vec<f32>,
    hot_alive:         Vec<bool>,
    hot_movement_mode: Vec<MovementMode>,

    // Cold SoA — read rarely (spawn, chronicle, debug).
    cold_creature_type: Vec<Option<CreatureType>>,
    cold_channels:      Vec<Option<ChannelSet>>,
    cold_spawn_tick:    Vec<Option<u32>>,
}

impl SimState {
    pub fn new(agent_cap: u32, seed: u64) -> Self {
        let cap = agent_cap as usize;
        Self {
            tick: 0,
            seed,
            pool: AgentSlotPool::new(agent_cap),
            hot_pos:           vec![Vec3::ZERO; cap],
            hot_hp:            vec![0.0; cap],
            hot_max_hp:        vec![0.0; cap],
            hot_alive:         vec![false; cap],
            hot_movement_mode: vec![MovementMode::Walk; cap],
            cold_creature_type: vec![None; cap],
            cold_channels:      (0..cap).map(|_| None).collect(),
            cold_spawn_tick:    vec![None; cap],
        }
    }

    pub fn spawn_agent(&mut self, spec: AgentSpawn) -> Option<AgentId> {
        let id = self.pool.alloc_agent()?;
        let slot = AgentSlotPool::slot_of_agent(id);
        self.hot_pos[slot]           = spec.pos;
        self.hot_hp[slot]            = spec.hp;
        self.hot_max_hp[slot]        = spec.hp.max(1.0);
        self.hot_alive[slot]         = true;
        self.hot_movement_mode[slot] = MovementMode::Walk;
        let caps = Capabilities::for_creature(spec.creature_type);
        self.cold_creature_type[slot] = Some(spec.creature_type);
        self.cold_channels[slot]      = Some(caps.channels);
        self.cold_spawn_tick[slot]    = Some(self.tick);
        Some(id)
    }

    pub fn kill_agent(&mut self, id: AgentId) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(a) = self.hot_alive.get_mut(slot) {
            *a = false;
        }
        self.pool.kill_agent(id);
    }

    // Per-agent field accessors (convenience for non-kernel code).
    pub fn agent_pos(&self, id: AgentId) -> Option<Vec3> {
        self.hot_pos.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_hp(&self, id: AgentId) -> Option<f32> {
        self.hot_hp.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_max_hp(&self, id: AgentId) -> Option<f32> {
        self.hot_max_hp.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_alive(&self, id: AgentId) -> bool {
        self.hot_alive
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
            .unwrap_or(false)
    }
    pub fn agent_movement_mode(&self, id: AgentId) -> Option<MovementMode> {
        self.hot_movement_mode.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_creature_type(&self, id: AgentId) -> Option<CreatureType> {
        self.cold_creature_type
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
            .flatten()
    }
    pub fn agent_channels(&self, id: AgentId) -> Option<&ChannelSet> {
        self.cold_channels.get(AgentSlotPool::slot_of_agent(id))?.as_ref()
    }
    pub fn agent_spawn_tick(&self, id: AgentId) -> Option<u32> {
        self.cold_spawn_tick
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
            .flatten()
    }

    // Per-agent field mutators.
    pub fn set_agent_pos(&mut self, id: AgentId, pos: Vec3) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(p) = self.hot_pos.get_mut(slot) {
            *p = pos;
        }
    }
    pub fn set_agent_hp(&mut self, id: AgentId, hp: f32) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(h) = self.hot_hp.get_mut(slot) {
            *h = hp;
        }
    }
    pub fn set_agent_movement_mode(&mut self, id: AgentId, mode: MovementMode) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(m) = self.hot_movement_mode.get_mut(slot) {
            *m = mode;
        }
    }

    pub fn agent_cap(&self) -> u32 {
        self.pool.alive.len() as u32
    }

    /// Iterator over alive AgentIds. Kernels that need multiple fields look them up by id.
    pub fn agents_alive(&self) -> impl Iterator<Item = AgentId> + '_ {
        self.hot_alive
            .iter()
            .enumerate()
            .filter(|(_, a)| **a)
            .map(|(i, _)| AgentId::new((i + 1) as u32).unwrap())
    }

    // Bulk-slice accessors — kernel-friendly. These are the real payoff of SoA.
    pub fn hot_pos(&self) -> &[Vec3] {
        &self.hot_pos
    }
    pub fn hot_hp(&self) -> &[f32] {
        &self.hot_hp
    }
    pub fn hot_max_hp(&self) -> &[f32] {
        &self.hot_max_hp
    }
    pub fn hot_alive(&self) -> &[bool] {
        &self.hot_alive
    }
    pub fn hot_movement_mode(&self) -> &[MovementMode] {
        &self.hot_movement_mode
    }
}
