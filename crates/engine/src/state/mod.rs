pub mod agent;
pub mod entity_pool;

use crate::channel::ChannelSet;
use crate::creature::{Capabilities, CreatureType};
use crate::ids::AgentId;
pub use agent::{AgentSpawn, MovementMode};
use entity_pool::{AgentPoolOps, AgentSlotPool};
use glam::Vec3;

/// Full SoA agent state — every field `docs/dsl/state.md` commits to, in one
/// struct. Hot fields are `Vec<T>` indexed by slot and read every tick; cold
/// fields are `Vec<Option<T>>` or per-agent collections touched only on
/// spawn / chronicle / debug paths. Behaviour is NOT attached — storage only;
/// subsequent plans wire masks, action_eval, and cascade handlers onto these.
///
/// **Needs split:** state.md commits to a 6-dim Maslow set
/// (`hunger, safety, shelter, social, purpose, esteem`). The engine carries
/// **8 needs: 3 physiological + 5 psychological**. Physiological
/// (`hunger`, `thirst`, `rest_timer`) drive Plan 1 Eat/Drink/Rest actions;
/// psychological (`safety`, `shelter`, `social`, `purpose`, `esteem`) are
/// the Maslow five minus `hunger` (it's already physiological). Both groups
/// are hot SoA `Vec<f32>` and initialise to 1.0 (fully satisfied).
pub struct SimState {
    pub tick: u32,
    pub seed: u64,
    pool:     AgentSlotPool,

    // --- Hot SoA — read/written every tick by observation / mask / step ---
    // Physical (state.md §Physical State + §Combat/Vitality, §Needs)
    hot_pos:            Vec<Vec3>,
    hot_hp:             Vec<f32>,
    hot_max_hp:         Vec<f32>,
    hot_alive:          Vec<bool>,
    hot_movement_mode:  Vec<MovementMode>,
    hot_level:          Vec<u32>,
    hot_move_speed:     Vec<f32>,
    hot_move_speed_mult: Vec<f32>,
    // Combat extras (state.md §Combat/Vitality)
    hot_shield_hp:      Vec<f32>,
    hot_armor:          Vec<f32>,
    hot_magic_resist:   Vec<f32>,
    hot_attack_damage:  Vec<f32>,
    hot_attack_range:   Vec<f32>,
    hot_mana:           Vec<f32>,
    hot_max_mana:       Vec<f32>,
    // Physiological needs (engine MVP, used by Plan 1 Eat/Drink/Rest)
    hot_hunger:         Vec<f32>,
    hot_thirst:         Vec<f32>,
    hot_rest_timer:     Vec<f32>,

    // --- Cold SoA — read rarely (spawn, chronicle, debug, narrative) ---
    cold_creature_type: Vec<Option<CreatureType>>,
    cold_channels:      Vec<Option<ChannelSet>>,
    cold_spawn_tick:    Vec<Option<u32>>,
    // Spatial extras (state.md §Physical State)
    cold_grid_id:       Vec<Option<u32>>,
    cold_local_pos:     Vec<Option<Vec3>>,
    cold_move_target:   Vec<Option<Vec3>>,
}

impl SimState {
    pub fn new(agent_cap: u32, seed: u64) -> Self {
        let cap = agent_cap as usize;
        Self {
            tick: 0,
            seed,
            pool: AgentSlotPool::new(agent_cap),
            hot_pos:             vec![Vec3::ZERO; cap],
            hot_hp:              vec![0.0; cap],
            hot_max_hp:          vec![0.0; cap],
            hot_alive:           vec![false; cap],
            hot_movement_mode:   vec![MovementMode::Walk; cap],
            hot_level:           vec![1; cap],
            hot_move_speed:      vec![1.0; cap],
            hot_move_speed_mult: vec![1.0; cap],
            hot_shield_hp:       vec![0.0; cap],
            hot_armor:           vec![0.0; cap],
            hot_magic_resist:    vec![0.0; cap],
            hot_attack_damage:   vec![10.0; cap],
            hot_attack_range:    vec![2.0; cap],
            hot_mana:            vec![0.0; cap],
            hot_max_mana:        vec![0.0; cap],
            hot_hunger:          vec![1.0; cap],
            hot_thirst:          vec![1.0; cap],
            hot_rest_timer:      vec![1.0; cap],
            cold_creature_type:  vec![None; cap],
            cold_channels:       (0..cap).map(|_| None).collect(),
            cold_spawn_tick:     vec![None; cap],
            cold_grid_id:        vec![None; cap],
            cold_local_pos:      vec![None; cap],
            cold_move_target:    vec![None; cap],
        }
    }

    #[contracts::debug_ensures(
        ret.is_some() -> self.agents_alive().count() == old(self.agents_alive().count()) + 1
    )]
    #[contracts::debug_ensures(
        ret.is_none() -> self.agents_alive().count() == old(self.agents_alive().count())
    )]
    pub fn spawn_agent(&mut self, spec: AgentSpawn) -> Option<AgentId> {
        let id = self.pool.alloc_agent()?;
        let slot = AgentSlotPool::slot_of_agent(id);
        self.hot_pos[slot]             = spec.pos;
        self.hot_hp[slot]              = spec.hp;
        self.hot_max_hp[slot]          = spec.hp.max(1.0);
        self.hot_alive[slot]           = true;
        self.hot_movement_mode[slot]   = MovementMode::Walk;
        self.hot_level[slot]           = 1;
        self.hot_move_speed[slot]      = 1.0;
        self.hot_move_speed_mult[slot] = 1.0;
        self.hot_shield_hp[slot]       = 0.0;
        self.hot_armor[slot]           = 0.0;
        self.hot_magic_resist[slot]    = 0.0;
        self.hot_attack_damage[slot]   = 10.0;
        self.hot_attack_range[slot]    = 2.0;
        self.hot_mana[slot]            = 0.0;
        self.hot_max_mana[slot]        = 0.0;
        self.hot_hunger[slot]          = 1.0;
        self.hot_thirst[slot]          = 1.0;
        self.hot_rest_timer[slot]      = 1.0;
        let caps = Capabilities::for_creature(spec.creature_type);
        self.cold_creature_type[slot]  = Some(spec.creature_type);
        self.cold_channels[slot]       = Some(caps.channels);
        self.cold_spawn_tick[slot]     = Some(self.tick);
        self.cold_grid_id[slot]        = None;
        self.cold_local_pos[slot]      = None;
        self.cold_move_target[slot]    = None;
        Some(id)
    }

    #[contracts::debug_ensures(!self.agent_alive(id))]
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
    pub fn agent_hunger(&self, id: AgentId) -> Option<f32> {
        self.hot_hunger.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_thirst(&self, id: AgentId) -> Option<f32> {
        self.hot_thirst.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_rest_timer(&self, id: AgentId) -> Option<f32> {
        self.hot_rest_timer.get(AgentSlotPool::slot_of_agent(id)).copied()
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

    // Spatial extras (state.md §Physical State).
    pub fn agent_level(&self, id: AgentId) -> Option<u32> {
        self.hot_level.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_move_speed(&self, id: AgentId) -> Option<f32> {
        self.hot_move_speed.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_move_speed_mult(&self, id: AgentId) -> Option<f32> {
        self.hot_move_speed_mult.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_grid_id(&self, id: AgentId) -> Option<u32> {
        self.cold_grid_id.get(AgentSlotPool::slot_of_agent(id)).copied().flatten()
    }
    pub fn agent_local_pos(&self, id: AgentId) -> Option<Vec3> {
        self.cold_local_pos.get(AgentSlotPool::slot_of_agent(id)).copied().flatten()
    }
    pub fn agent_move_target(&self, id: AgentId) -> Option<Vec3> {
        self.cold_move_target.get(AgentSlotPool::slot_of_agent(id)).copied().flatten()
    }

    // Combat extras (Task B).
    pub fn agent_shield_hp(&self, id: AgentId) -> Option<f32> {
        self.hot_shield_hp.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_armor(&self, id: AgentId) -> Option<f32> {
        self.hot_armor.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_magic_resist(&self, id: AgentId) -> Option<f32> {
        self.hot_magic_resist.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_attack_damage(&self, id: AgentId) -> Option<f32> {
        self.hot_attack_damage.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_attack_range(&self, id: AgentId) -> Option<f32> {
        self.hot_attack_range.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_mana(&self, id: AgentId) -> Option<f32> {
        self.hot_mana.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_max_mana(&self, id: AgentId) -> Option<f32> {
        self.hot_max_mana.get(AgentSlotPool::slot_of_agent(id)).copied()
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
    pub fn set_agent_hunger(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_hunger.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_thirst(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_thirst.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_rest_timer(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_rest_timer.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }

    // Spatial-extras setters.
    pub fn set_agent_level(&mut self, id: AgentId, v: u32) {
        if let Some(s) = self.hot_level.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_move_speed(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_move_speed.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_move_speed_mult(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_move_speed_mult.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_grid_id(&mut self, id: AgentId, v: Option<u32>) {
        if let Some(s) = self.cold_grid_id.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_local_pos(&mut self, id: AgentId, v: Option<Vec3>) {
        if let Some(s) = self.cold_local_pos.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_move_target(&mut self, id: AgentId, v: Option<Vec3>) {
        if let Some(s) = self.cold_move_target.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }

    // Combat-extras setters (Task B).
    pub fn set_agent_shield_hp(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_shield_hp.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_armor(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_armor.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_magic_resist(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_magic_resist.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_attack_damage(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_attack_damage.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_attack_range(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_attack_range.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_mana(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_mana.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_max_mana(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_max_mana.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }

    pub fn agent_cap(&self) -> u32 {
        self.pool.alive.len() as u32
    }

    /// Pool self-consistency predicate for `PoolNonOverlapInvariant`.
    /// Returns `true` when no slot is both alive and in the freelist and the
    /// freelist has no duplicates. See `Pool::is_non_overlapping`.
    pub fn pool_is_consistent(&self) -> bool {
        self.pool.is_non_overlapping()
    }

    /// Test-only: expose the underlying pool for fault injection (corrupting
    /// the freelist to prove the invariant check actually runs). Production
    /// code must never call this.
    #[doc(hidden)]
    pub fn pool_mut_for_test(&mut self) -> &mut entity_pool::AgentSlotPool {
        &mut self.pool
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
    pub fn hot_hunger(&self) -> &[f32] {
        &self.hot_hunger
    }
    pub fn hot_thirst(&self) -> &[f32] {
        &self.hot_thirst
    }
    pub fn hot_rest_timer(&self) -> &[f32] {
        &self.hot_rest_timer
    }

    // Spatial-extras bulk slices (Task A).
    pub fn hot_level(&self) -> &[u32] {
        &self.hot_level
    }
    pub fn hot_move_speed(&self) -> &[f32] {
        &self.hot_move_speed
    }
    pub fn hot_move_speed_mult(&self) -> &[f32] {
        &self.hot_move_speed_mult
    }
    pub fn cold_grid_id(&self) -> &[Option<u32>] {
        &self.cold_grid_id
    }
    pub fn cold_local_pos(&self) -> &[Option<Vec3>] {
        &self.cold_local_pos
    }
    pub fn cold_move_target(&self) -> &[Option<Vec3>] {
        &self.cold_move_target
    }

    // Combat-extras bulk slices (Task B).
    pub fn hot_shield_hp(&self) -> &[f32] {
        &self.hot_shield_hp
    }
    pub fn hot_armor(&self) -> &[f32] {
        &self.hot_armor
    }
    pub fn hot_magic_resist(&self) -> &[f32] {
        &self.hot_magic_resist
    }
    pub fn hot_attack_damage(&self) -> &[f32] {
        &self.hot_attack_damage
    }
    pub fn hot_attack_range(&self) -> &[f32] {
        &self.hot_attack_range
    }
    pub fn hot_mana(&self) -> &[f32] {
        &self.hot_mana
    }
    pub fn hot_max_mana(&self) -> &[f32] {
        &self.hot_max_mana
    }
}
