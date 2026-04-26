pub mod agent;
pub mod agent_types;
pub mod entity_pool;

use crate::ability::MAX_ABILITIES;
use engine_data::types::ChannelSet;
use engine_data::entities::{Capabilities, CreatureType};
use crate::ids::AgentId;
use crate::spatial::SpatialHash;
use crate::terrain::{FlatPlane, TerrainQuery};
pub use agent::{AgentSpawn, MovementMode};
use agent_types::{
    ClassSlot, Creditor, Inventory, Membership, Relationship, StatusEffect,
};
use entity_pool::{AgentPoolOps, AgentSlotPool};
use glam::Vec3;
use smallvec::SmallVec;
use std::sync::Arc;

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
    /// Runtime balance tunables — the compiler-emitted aggregate of every
    /// `config` block in `assets/sim/config.sim`. Loaded via
    /// `Config::from_toml(...)` at startup, or defaulted via `Config::default()`
    /// (which bakes in the DSL's `= <default>` clauses). Every balance
    /// constant that used to be a hand-written `pub const` in
    /// `crates/engine/src/step.rs` / `mask.rs` / `ability/expire.rs` is now
    /// a field on this struct; see `docs/game/compiler_progress.md` for the
    /// config-milestone row.
    pub config: engine_data::config::Config,
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
    // Psychological needs (state.md §Needs — Maslow minus hunger)
    hot_safety:         Vec<f32>,
    hot_shelter:        Vec<f32>,
    hot_social:         Vec<f32>,
    hot_purpose:        Vec<f32>,
    hot_esteem:         Vec<f32>,
    // Personality (state.md §Personality; engine uses `altruism` for
    // what state.md calls `compassion` — same helping/empathy trait).
    hot_risk_tolerance: Vec<f32>,
    hot_social_drive:   Vec<f32>,
    hot_ambition:       Vec<f32>,
    hot_altruism:       Vec<f32>,
    hot_curiosity:      Vec<f32>,
    // Combat Foundation Task 1: who this agent is currently locked in melee
    // with. `engaged_with[a] == Some(b)` iff `engaged_with[b] == Some(a)`
    // after `ability::expire::tick_start` runs (bidirectional invariant).
    // `None` means disengaged. Storage here; enforcement in Task 3.
    hot_engaged_with:   Vec<Option<AgentId>>,
    // Combat Foundation Task 2 + Task 143: timed status fields stored as
    // absolute expiry ticks. `stun_expires_at_tick == 0` means not stunned
    // (a real expiry always sits at `state.tick + duration > 0`); same
    // for slow. `slow_factor_q8` is a q8 fixed-point speed multiplier
    // (e.g. 51 ≈ 0.2× speed, 204 ≈ 0.8× speed) read through the
    // `slow_factor` lazy view which zeroes it once the expiry has
    // elapsed. Cooldown is an absolute tick too; mask compares against
    // `state.tick`. Task 143 retired the per-tick `tick_start_timers`
    // decrement — expiry is now a synthetic boundary: `state.tick <
    // expires_at_tick` means active.
    hot_stun_expires_at_tick:     Vec<u32>,
    hot_slow_expires_at_tick:     Vec<u32>,
    hot_slow_factor_q8:           Vec<i16>,
    hot_cooldown_next_ready_tick: Vec<u32>,

    // --- Cold SoA — read rarely (spawn, chronicle, debug, narrative) ---
    cold_creature_type: Vec<Option<CreatureType>>,
    cold_channels:      Vec<Option<ChannelSet>>,
    cold_spawn_tick:    Vec<Option<u32>>,
    // Spatial extras (state.md §Physical State)
    cold_grid_id:       Vec<Option<u32>>,
    cold_local_pos:     Vec<Option<Vec3>>,
    cold_move_target:   Vec<Option<Vec3>>,
    // Status effects (state.md §StatusEffect) — per-agent stack.
    cold_status_effects: Vec<SmallVec<[StatusEffect; 8]>>,
    // Memberships (state.md §Membership) — per-agent group list.
    cold_memberships:    Vec<SmallVec<[Membership; 4]>>,
    // Inventory (state.md §Inventory) — one per agent.
    cold_inventory:      Vec<Inventory>,
    // Memory retired 2026-04-23 by Subsystem 2 Phase 4 — see
    // `state.views.memory` (`@per_entity_ring(K=64)` view). The DSL
    // `agents.record_memory(...)` stdlib call lowers directly to
    // `state.views.memory.push(...)`. GPU driver at
    // `crates/engine_gpu/src/view_storage_per_entity_ring.rs` owns
    // the resident-path mirror; readback rehydrates `state.views.memory`.
    // Relationships (state.md §Relationship) — capped at 8 (state.md caps
    // at 20; the stub uses 8 inline as a smaller default; eviction is a
    // later plan's concern).
    cold_relationships:  Vec<SmallVec<[Relationship; 8]>>,
    // Class definitions (state.md §Skill & Class) — fixed 4 slots per agent.
    cold_class_definitions: Vec<[ClassSlot; 4]>,
    // Creditor ledger (state.md §Economic).
    cold_creditor_ledger:   Vec<SmallVec<[Creditor; 16]>>,
    // Mentor lineage (state.md §Relationships mentor_lineage) — 8-deep chain.
    cold_mentor_lineage:    Vec<[Option<AgentId>; 8]>,
    // Theory-of-Mind Phase 1 (Plan 2026-04-25): per-agent belief map keyed by
    // observed AgentId.  Capacity N=8 (spec §3.1).  `BeliefState` is emitted by
    // the DSL compiler in Task 3; this field is intentionally forward-declared
    // here so the SoA layout is established before the type lands.
    cold_beliefs: Vec<crate::pool::BoundedMap<AgentId, engine_data::belief::BeliefState, 8>>,
    /// Per-(agent, ability-slot) local cooldown cursor. Value = the
    /// tick when this specific ability slot next becomes ready;
    /// `0` means ready now (or never cast). Gated together with
    /// `hot_cooldown_next_ready_tick` (global GCD) by
    /// `SimState::can_cast_ability` (added in a later task).
    ///
    /// Added 2026-04-22 to fix a shared-cursor bug where all
    /// abilities on one agent were gated by the single global cursor.
    /// Read only on cast-gate evaluation, so grouped with the cold
    /// SoA fields.
    pub ability_cooldowns: Vec<[u32; MAX_ABILITIES]>,
    // Per-pair standing retired 2026-04-23 — see `state.views.standing`
    // (Task 3.1 `@materialized` view, K=8 symmetric pair top-k).

    // Spatial index — incremental uniform-grid hash. Mutators
    // (`spawn_agent`, `kill_agent`, `set_agent_pos`,
    // `set_agent_movement_mode`) push O(1) deltas into it as they touch
    // the SoA, so `state.spatial()` is always live without any per-tick
    // rebuild. The previous BTreeMap implementation rebuilt eagerly on
    // every mutation and was the cause of the post-audit hot-path
    // regression at N=500.
    spatial: SpatialHash,

    // `views: ViewRegistry` field deleted along with engine/src/generated/.
    // Callers now thread `&mut ViewRegistry` separately as a `step` parameter.
    // Tests that referenced `state.views.*` are #[ignore]d until Task 11 emits
    // the new view-as-parameter handler signatures into engine_rules.

    /// Ability program registry — append-only table of compiled ability
    /// programs, looked up by `AbilityId` during cast dispatch and mask
    /// evaluation. Moved out of the (retired) per-`CastHandler` `Arc` so
    /// the cascade path reads it directly off `state`; also lets the mask
    /// build read the same registry without threading an `Option<&Arc>`
    /// through `step_full`.
    ///
    /// Defaults to empty. Populate via `state.ability_registry =
    /// builder.build()` after spawning the state.
    pub ability_registry: crate::ability::AbilityRegistry,

    /// Terrain backend the engine consults for height / walkability /
    /// line-of-sight queries. Defaults to `FlatPlane` — height 0
    /// everywhere, every point walkable, every LOS clear — so the
    /// wolves+humans canonical fixture and every legacy test stays
    /// deterministic and terrain-agnostic. Callers that want real
    /// elevation (examples like `hill_fight`, future voxel adapter)
    /// replace this field after construction:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use engine::state::SimState;
    /// # struct MyHill;
    /// # impl engine::terrain::TerrainQuery for MyHill {
    /// #     fn height_at(&self, _x: f32, _y: f32) -> f32 { 0.0 }
    /// #     fn walkable(&self, _pos: glam::Vec3, _m: engine::state::MovementMode) -> bool { true }
    /// #     fn line_of_sight(&self, _from: glam::Vec3, _to: glam::Vec3) -> bool { true }
    /// # }
    /// let mut state = SimState::new(8, 0);
    /// state.terrain = Arc::new(MyHill);
    /// ```
    ///
    /// Held as `Arc<dyn TerrainQuery + Send + Sync>` so parallel tick
    /// paths share one backend without synchronisation. See
    /// `docs/superpowers/notes/2026-04-22-terrain-integration-gap.md`
    /// for the design discussion — this is Option B (trait-object
    /// injection), deliberately chosen so the engine stays headless.
    pub terrain: Arc<dyn TerrainQuery + Send + Sync>,
}

impl SimState {
    /// Convenience constructor that defaults the runtime `Config`. Equivalent
    /// to `SimState::new_with_config(agent_cap, seed, Config::default())`.
    /// Preserves the pre-config constructor signature so every existing test
    /// (there are many) keeps compiling without edits.
    pub fn new(agent_cap: u32, seed: u64) -> Self {
        Self::new_with_config(agent_cap, seed, engine_data::config::Config::default())
    }

    /// Build a `SimState` with a caller-supplied `Config`. This is the
    /// main constructor — tests that want to override a balance constant
    /// (e.g. double damage to assert TOML tuning is live) build a mutated
    /// `Config` and pass it here. The engine never reads balance tunables
    /// from anywhere other than `state.config`.
    pub fn new_with_config(
        agent_cap: u32,
        seed: u64,
        config: engine_data::config::Config,
    ) -> Self {
        let cap = agent_cap as usize;
        // Per-slot attack stats inherit the config defaults so runtime
        // TOML tuning flows through to every freshly-spawned agent.
        // `spawn_agent` also resets these fields using the same config
        // values; custom per-agent stats flow through `set_agent_*` after
        // spawn as before.
        let default_attack_damage = config.combat.attack_damage;
        let default_attack_range = config.combat.attack_range;
        let default_move_speed = config.movement.move_speed_mps;
        Self {
            tick: 0,
            seed,
            config,
            pool: AgentSlotPool::new(agent_cap),
            hot_pos:             vec![Vec3::ZERO; cap],
            hot_hp:              vec![0.0; cap],
            hot_max_hp:          vec![0.0; cap],
            hot_alive:           vec![false; cap],
            hot_movement_mode:   vec![MovementMode::Walk; cap],
            hot_level:           vec![1; cap],
            hot_move_speed:      vec![default_move_speed; cap],
            hot_move_speed_mult: vec![1.0; cap],
            hot_shield_hp:       vec![0.0; cap],
            hot_armor:           vec![0.0; cap],
            hot_magic_resist:    vec![0.0; cap],
            hot_attack_damage:   vec![default_attack_damage; cap],
            hot_attack_range:    vec![default_attack_range; cap],
            hot_mana:            vec![0.0; cap],
            hot_max_mana:        vec![0.0; cap],
            hot_hunger:          vec![1.0; cap],
            hot_thirst:          vec![1.0; cap],
            hot_rest_timer:      vec![1.0; cap],
            hot_safety:          vec![1.0; cap],
            hot_shelter:         vec![1.0; cap],
            hot_social:          vec![1.0; cap],
            hot_purpose:         vec![1.0; cap],
            hot_esteem:          vec![1.0; cap],
            hot_risk_tolerance:  vec![0.5; cap],
            hot_social_drive:    vec![0.5; cap],
            hot_ambition:        vec![0.5; cap],
            hot_altruism:        vec![0.5; cap],
            hot_curiosity:       vec![0.5; cap],
            hot_engaged_with:    vec![None; cap],
            hot_stun_expires_at_tick:     vec![0; cap],
            hot_slow_expires_at_tick:     vec![0; cap],
            hot_slow_factor_q8:           vec![0; cap],
            hot_cooldown_next_ready_tick: vec![0; cap],
            cold_creature_type:  vec![None; cap],
            cold_channels:       (0..cap).map(|_| None).collect(),
            cold_spawn_tick:     vec![None; cap],
            cold_grid_id:        vec![None; cap],
            cold_local_pos:      vec![None; cap],
            cold_move_target:    vec![None; cap],
            cold_status_effects: (0..cap).map(|_| SmallVec::new()).collect(),
            cold_memberships:    (0..cap).map(|_| SmallVec::new()).collect(),
            cold_inventory:      vec![Inventory::default(); cap],
            cold_relationships:  (0..cap).map(|_| SmallVec::new()).collect(),
            cold_class_definitions: vec![[ClassSlot::default(); 4]; cap],
            cold_creditor_ledger:   (0..cap).map(|_| SmallVec::new()).collect(),
            cold_mentor_lineage:    vec![[None; 8]; cap],
            cold_beliefs:           (0..cap).map(|_| crate::pool::BoundedMap::new()).collect(),
            ability_cooldowns:      vec![[0u32; MAX_ABILITIES]; cap],
            // Incremental spatial hash — sized for `cap` agent slots.
            // Mutators push O(1) deltas; no per-mutation rebuild.
            spatial:                SpatialHash::new(agent_cap),
            // Empty ability registry by default. Tests / production code
            // that need specific abilities do `state.ability_registry =
            // builder.build()` after construction.
            ability_registry:       crate::ability::AbilityRegistry::new(),
            // Flat-plane terrain default. Caller replaces via
            // `state.terrain = Arc::new(MyBackend)` to opt into real
            // elevation / LOS. See field-level docs above.
            terrain:                Arc::new(FlatPlane),
        }
    }

    /// Live spatial index. Mutators (`spawn_agent`, `kill_agent`,
    /// `set_agent_pos`, `set_agent_movement_mode`) keep the index in sync
    /// incrementally on every call, so this is always consistent with the
    /// current SoA. Callers that need sub-linear proximity queries should
    /// prefer this over scanning `agents_alive()`.
    pub fn spatial(&self) -> &SpatialHash { &self.spatial }

    #[contracts::debug_ensures(
        ret.is_some() -> self.agents_alive().count() == old(self.agents_alive().count()) + 1
    )]
    #[contracts::debug_ensures(
        ret.is_none() -> self.agents_alive().count() == old(self.agents_alive().count())
    )]
    pub fn spawn_agent(&mut self, spec: AgentSpawn) -> Option<AgentId> {
        let id = self.pool.alloc_agent()?;
        let slot = AgentSlotPool::slot_of_agent(id);
        // Task 150: `max_hp` is an independent cap carried on `AgentSpawn`.
        // Callers that want a wounded spawn pass `hp: 30.0, max_hp: 100.0`
        // — the cap stays at 100 and `hp_pct = 0.3` so target-selection
        // scoring can see the agent is wounded. Previously `max_hp` was
        // written as `spec.hp.max(1.0)` which made every freshly-spawned
        // agent report `hp_pct = 1.0` and silently broke pct-based rows.
        //
        // `max_hp` must be ≥ 1 so downstream `hp / max_hp` never divides
        // by zero. If a caller accidentally passes `max_hp < hp` we clamp
        // hp down to the cap (debug-asserts so tests catch the bug).
        let max_hp = spec.max_hp.max(1.0);
        debug_assert!(
            spec.hp <= spec.max_hp,
            "AgentSpawn: hp ({}) > max_hp ({}) — clamping hp to max_hp. \
             Callers should pass `max_hp >= hp` or leave `max_hp` at the \
             Default (100.0) when spawning a fully-healthy agent.",
            spec.hp, spec.max_hp,
        );
        let hp = spec.hp.min(max_hp);
        self.hot_pos[slot]             = spec.pos;
        self.hot_hp[slot]              = hp;
        self.hot_max_hp[slot]          = max_hp;
        self.hot_alive[slot]           = true;
        self.hot_movement_mode[slot]   = MovementMode::Walk;
        self.hot_level[slot]           = 1;
        self.hot_move_speed[slot]      = self.config.movement.move_speed_mps;
        self.hot_move_speed_mult[slot] = 1.0;
        self.hot_shield_hp[slot]       = 0.0;
        self.hot_armor[slot]           = 0.0;
        self.hot_magic_resist[slot]    = 0.0;
        self.hot_attack_damage[slot]   = self.config.combat.attack_damage;
        self.hot_attack_range[slot]    = self.config.combat.attack_range;
        self.hot_mana[slot]            = 0.0;
        self.hot_max_mana[slot]        = 0.0;
        self.hot_hunger[slot]          = 1.0;
        self.hot_thirst[slot]          = 1.0;
        self.hot_rest_timer[slot]      = 1.0;
        self.hot_safety[slot]          = 1.0;
        self.hot_shelter[slot]         = 1.0;
        self.hot_social[slot]          = 1.0;
        self.hot_purpose[slot]         = 1.0;
        self.hot_esteem[slot]          = 1.0;
        self.hot_risk_tolerance[slot]  = 0.5;
        self.hot_social_drive[slot]    = 0.5;
        self.hot_ambition[slot]        = 0.5;
        self.hot_altruism[slot]        = 0.5;
        self.hot_curiosity[slot]       = 0.5;
        self.hot_engaged_with[slot]    = None;
        self.hot_stun_expires_at_tick[slot]     = 0;
        self.hot_slow_expires_at_tick[slot]     = 0;
        self.hot_slow_factor_q8[slot]           = 0;
        self.hot_cooldown_next_ready_tick[slot] = 0;
        let caps = Capabilities::for_creature(spec.creature_type);
        self.cold_creature_type[slot]  = Some(spec.creature_type);
        self.cold_channels[slot]       = Some(caps.channels);
        self.cold_spawn_tick[slot]     = Some(self.tick);
        self.cold_grid_id[slot]        = None;
        self.cold_local_pos[slot]      = None;
        self.cold_move_target[slot]    = None;
        self.cold_status_effects[slot].clear();
        self.cold_memberships[slot].clear();
        self.cold_inventory[slot] = Inventory::default();
        self.cold_relationships[slot].clear();
        self.cold_class_definitions[slot] = [ClassSlot::default(); 4];
        self.cold_creditor_ledger[slot].clear();
        self.cold_mentor_lineage[slot] = [None; 8];
        // Theory-of-Mind Phase 1: clear belief map on (re)spawn so a recycled
        // slot doesn't carry stale beliefs from the previous occupant.
        self.cold_beliefs[slot]        = crate::pool::BoundedMap::new();
        self.ability_cooldowns[slot]   = [0u32; MAX_ABILITIES];
        // Incremental spatial-hash insert — O(1).
        self.spatial.insert(id, spec.pos, MovementMode::Walk);
        Some(id)
    }

    #[contracts::debug_ensures(!self.agent_alive(id))]
    pub fn kill_agent(&mut self, id: AgentId) {
        let slot = AgentSlotPool::slot_of_agent(id);
        if let Some(a) = self.hot_alive.get_mut(slot) {
            *a = false;
        }
        self.pool.kill_agent(id);
        // Incremental spatial-hash remove — O(1) within the agent's bucket.
        self.spatial.remove(id);
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

    // Status effects (Task C).
    pub fn agent_status_effects(&self, id: AgentId) -> Option<&[StatusEffect]> {
        self.cold_status_effects
            .get(AgentSlotPool::slot_of_agent(id))
            .map(|v| v.as_slice())
    }
    pub fn push_agent_status_effect(&mut self, id: AgentId, fx: StatusEffect) {
        if let Some(v) = self
            .cold_status_effects
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.push(fx);
        }
    }
    pub fn clear_agent_status_effects(&mut self, id: AgentId) {
        if let Some(v) = self
            .cold_status_effects
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.clear();
        }
    }

    // Psychological needs (Task D).
    pub fn agent_safety(&self, id: AgentId) -> Option<f32> {
        self.hot_safety.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_shelter(&self, id: AgentId) -> Option<f32> {
        self.hot_shelter.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_social(&self, id: AgentId) -> Option<f32> {
        self.hot_social.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_purpose(&self, id: AgentId) -> Option<f32> {
        self.hot_purpose.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_esteem(&self, id: AgentId) -> Option<f32> {
        self.hot_esteem.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn set_agent_safety(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_safety.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_shelter(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_shelter.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_social(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_social.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_purpose(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_purpose.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_esteem(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_esteem.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }

    // Personality (Task E).
    pub fn agent_risk_tolerance(&self, id: AgentId) -> Option<f32> {
        self.hot_risk_tolerance.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_social_drive(&self, id: AgentId) -> Option<f32> {
        self.hot_social_drive.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_ambition(&self, id: AgentId) -> Option<f32> {
        self.hot_ambition.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_altruism(&self, id: AgentId) -> Option<f32> {
        self.hot_altruism.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn agent_curiosity(&self, id: AgentId) -> Option<f32> {
        self.hot_curiosity.get(AgentSlotPool::slot_of_agent(id)).copied()
    }
    pub fn set_agent_risk_tolerance(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_risk_tolerance.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_social_drive(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_social_drive.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_ambition(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_ambition.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_altruism(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_altruism.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }
    pub fn set_agent_curiosity(&mut self, id: AgentId, v: f32) {
        if let Some(s) = self.hot_curiosity.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = v;
        }
    }

    // Engagement (Combat Foundation Task 1).
    pub fn agent_engaged_with(&self, id: AgentId) -> Option<AgentId> {
        self.hot_engaged_with
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
            .unwrap_or(None)
    }
    pub fn set_agent_engaged_with(&mut self, id: AgentId, other: Option<AgentId>) {
        if let Some(s) = self
            .hot_engaged_with
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = other;
        }
    }

    // Combat timing (Combat Foundation Task 2 + Task 143). Fields store
    // absolute expiry ticks; an agent is stunned while `state.tick <
    // expires_at_tick`. The `agent_stunned` / `effective_slow_factor_q8`
    // helpers below encapsulate that read and are what mask / scoring
    // code should call.
    pub fn agent_stun_expires_at(&self, id: AgentId) -> Option<u32> {
        self.hot_stun_expires_at_tick
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
    }
    pub fn set_agent_stun_expires_at(&mut self, id: AgentId, v: u32) {
        if let Some(s) = self
            .hot_stun_expires_at_tick
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = v;
        }
    }
    pub fn agent_slow_expires_at(&self, id: AgentId) -> Option<u32> {
        self.hot_slow_expires_at_tick
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
    }
    pub fn set_agent_slow_expires_at(&mut self, id: AgentId, v: u32) {
        if let Some(s) = self
            .hot_slow_expires_at_tick
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = v;
        }
    }
    /// Currently-active stun predicate: returns `true` while
    /// `state.tick < agent_stun_expires_at(id)`. The zero sentinel means
    /// "never stunned" (initial state and post-expiry are both 0 /
    /// `state.tick >= 0`, so the comparison rejects both).
    pub fn agent_stunned(&self, id: AgentId) -> bool {
        let exp = self.agent_stun_expires_at(id).unwrap_or(0);
        self.tick < exp
    }
    /// Effective slow factor: the stored `slow_factor_q8` while the slow
    /// is still active, `0` once the expiry has elapsed. Callers that
    /// just want a speed multiplier should prefer this over the raw
    /// accessor pair so they don't need to re-check the expiry.
    pub fn effective_slow_factor_q8(&self, id: AgentId) -> i16 {
        let exp = self.agent_slow_expires_at(id).unwrap_or(0);
        if self.tick < exp {
            self.agent_slow_factor_q8(id).unwrap_or(0)
        } else {
            0
        }
    }
    pub fn agent_slow_factor_q8(&self, id: AgentId) -> Option<i16> {
        self.hot_slow_factor_q8
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
    }
    pub fn set_agent_slow_factor_q8(&mut self, id: AgentId, v: i16) {
        if let Some(s) = self
            .hot_slow_factor_q8
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = v;
        }
    }
    pub fn agent_cooldown_next_ready(&self, id: AgentId) -> Option<u32> {
        self.hot_cooldown_next_ready_tick
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
    }
    pub fn set_agent_cooldown_next_ready(&mut self, id: AgentId, v: u32) {
        if let Some(s) = self
            .hot_cooldown_next_ready_tick
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = v;
        }
    }

    /// True iff the given ability slot is off cooldown (both global GCD
    /// and per-slot local cooldown have cleared at `now`).
    ///
    /// Returns `false` for unknown agents. Out-of-range slot indices
    /// return `true` (defensive default; callers should bound-check
    /// against the live ability registry separately).
    pub fn can_cast_ability(&self, agent: AgentId, slot: u8, now: u32) -> bool {
        let agent_slot = AgentSlotPool::slot_of_agent(agent);
        let Some(&global_next_ready) = self.hot_cooldown_next_ready_tick.get(agent_slot) else {
            return false;
        };
        if (slot as usize) >= crate::ability::MAX_ABILITIES {
            return true;
        }
        let Some(local_slots) = self.ability_cooldowns.get(agent_slot) else {
            return false;
        };
        let global_ready = global_next_ready <= now;
        let local_ready = local_slots[slot as usize] <= now;
        global_ready && local_ready
    }

    /// Map an `AbilityId` to its per-agent local-cooldown slot index.
    /// MVP contract: the ability registry is shared globally, so the
    /// per-agent slot index is the same as the registry slot. Returns
    /// `None` when the id falls outside the `MAX_ABILITIES` local
    /// cooldown window (the gate still fires; only the local cursor
    /// write/read is skipped).
    ///
    /// Added 2026-04-22 alongside `record_cast_cooldowns` to give the
    /// DSL-emitted `physics cast` rule a single entry point for the
    /// dual-cursor write.
    #[inline]
    pub fn ability_slot_for(&self, _agent: AgentId, ability: crate::ability::AbilityId) -> Option<u8> {
        let slot = ability.slot();
        if slot < crate::ability::MAX_ABILITIES {
            Some(slot as u8)
        } else {
            None
        }
    }

    /// Post-cast cooldown bookkeeping: set **both** the per-agent
    /// global cooldown cursor (GCD, from `config.combat.global_cooldown_ticks`)
    /// and the per-(agent, slot) local cooldown cursor (from
    /// `ability.gate.cooldown_ticks`). Idempotent for unknown agents /
    /// unknown abilities — silently drops rather than panicking, which
    /// matches the `physics cast` rule's "unknown id is a silent drop"
    /// contract.
    ///
    /// Writes use saturating arithmetic; near-`u32::MAX` ticks clamp
    /// rather than wrapping, keeping the cooldown gate monotonic.
    ///
    /// Added 2026-04-22 by the ability-cooldowns subsystem — previously
    /// only the global cursor was written (with the ability's own
    /// cooldown), which was the shared-cursor bug this helper fixes.
    pub fn record_cast_cooldowns(
        &mut self,
        caster: AgentId,
        ability: crate::ability::AbilityId,
        now: u32,
    ) {
        let agent_slot = AgentSlotPool::slot_of_agent(caster);
        // Global GCD — short shared gate across every ability the caster
        // owns. Configurable via `combat.global_cooldown_ticks` (default
        // 5 ticks = 0.5s at 10 Hz).
        let gcd = self.config.combat.global_cooldown_ticks;
        let global_next_ready = now.saturating_add(gcd);
        if let Some(s) = self.hot_cooldown_next_ready_tick.get_mut(agent_slot) {
            *s = global_next_ready;
        }
        // Local cooldown — per-ability refresh. Unknown ability id or
        // out-of-range slot: skip the local write (the global cursor
        // still applies, matching the pre-subsystem behaviour as a
        // defensive fallback).
        let Some(local_cd) = self
            .ability_registry
            .get(ability)
            .map(|p| p.gate.cooldown_ticks)
        else {
            return;
        };
        let Some(slot) = self.ability_slot_for(caster, ability) else {
            return;
        };
        let local_next_ready = now.saturating_add(local_cd);
        if let Some(row) = self.ability_cooldowns.get_mut(agent_slot) {
            row[slot as usize] = local_next_ready;
        }
    }

    // Per-pair standing retired 2026-04-23 — read / mutate via the
    // `@materialized` `standing` view: `state.views.standing.get(a, b)`
    // and `state.views.standing.adjust(a, b, delta, tick)`.

    // Memberships (Task G).
    pub fn agent_memberships(&self, id: AgentId) -> Option<&[Membership]> {
        self.cold_memberships
            .get(AgentSlotPool::slot_of_agent(id))
            .map(|v| v.as_slice())
    }
    pub fn push_agent_membership(&mut self, id: AgentId, m: Membership) {
        if let Some(v) = self
            .cold_memberships
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.push(m);
        }
    }
    pub fn clear_agent_memberships(&mut self, id: AgentId) {
        if let Some(v) = self
            .cold_memberships
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.clear();
        }
    }

    // Inventory (Task H).
    pub fn agent_inventory(&self, id: AgentId) -> Option<Inventory> {
        self.cold_inventory
            .get(AgentSlotPool::slot_of_agent(id))
            .copied()
    }
    pub fn set_agent_inventory(&mut self, id: AgentId, inv: Inventory) {
        if let Some(s) = self.cold_inventory.get_mut(AgentSlotPool::slot_of_agent(id)) {
            *s = inv;
        }
    }

    // Memory (Task I) retired 2026-04-23 by Subsystem 2 Phase 4.
    // Read/mutate via the `@per_entity_ring(K=64)` `memory` view:
    // `state.views.memory.push(observer.raw(), MemoryEntry { .. })`
    // and `state.views.memory.entries(observer)`.

    // Relationships (Task J).
    pub fn agent_relationships(&self, id: AgentId) -> Option<&[Relationship]> {
        self.cold_relationships
            .get(AgentSlotPool::slot_of_agent(id))
            .map(|v| v.as_slice())
    }
    pub fn push_agent_relationship(&mut self, id: AgentId, r: Relationship) {
        if let Some(v) = self
            .cold_relationships
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.push(r);
        }
    }
    pub fn clear_agent_relationships(&mut self, id: AgentId) {
        if let Some(v) = self
            .cold_relationships
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.clear();
        }
    }

    // Class definitions / creditor ledger / mentor lineage (Task K).
    pub fn agent_classes(&self, id: AgentId) -> Option<&[ClassSlot; 4]> {
        self.cold_class_definitions.get(AgentSlotPool::slot_of_agent(id))
    }
    pub fn set_agent_classes(&mut self, id: AgentId, slots: [ClassSlot; 4]) {
        if let Some(s) = self
            .cold_class_definitions
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = slots;
        }
    }
    pub fn agent_creditors(&self, id: AgentId) -> Option<&[Creditor]> {
        self.cold_creditor_ledger
            .get(AgentSlotPool::slot_of_agent(id))
            .map(|v| v.as_slice())
    }
    pub fn push_agent_creditor(&mut self, id: AgentId, c: Creditor) {
        if let Some(v) = self
            .cold_creditor_ledger
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            v.push(c);
        }
    }
    pub fn agent_mentor_lineage(&self, id: AgentId) -> Option<&[Option<AgentId>; 8]> {
        self.cold_mentor_lineage.get(AgentSlotPool::slot_of_agent(id))
    }
    pub fn set_agent_mentor_lineage(&mut self, id: AgentId, lineage: [Option<AgentId>; 8]) {
        if let Some(s) = self
            .cold_mentor_lineage
            .get_mut(AgentSlotPool::slot_of_agent(id))
        {
            *s = lineage;
        }
    }

    // Theory-of-Mind Phase 1 (Plan 2026-04-25): belief-map accessors.
    // `BeliefState` is defined in `engine_data::belief` (Task 3 of the plan).
    pub fn agent_cold_beliefs(
        &self,
        id: AgentId,
    ) -> Option<&crate::pool::BoundedMap<AgentId, engine_data::belief::BeliefState, 8>> {
        self.cold_beliefs.get(AgentSlotPool::slot_of_agent(id))
    }
    pub fn agent_cold_beliefs_mut(
        &mut self,
        id: AgentId,
    ) -> Option<&mut crate::pool::BoundedMap<AgentId, engine_data::belief::BeliefState, 8>> {
        self.cold_beliefs.get_mut(AgentSlotPool::slot_of_agent(id))
    }

    // Per-agent field mutators.
    pub fn set_agent_pos(&mut self, id: AgentId, pos: Vec3) {
        let slot = AgentSlotPool::slot_of_agent(id);
        let mut moved = false;
        if let Some(p) = self.hot_pos.get_mut(slot) {
            *p = pos;
            moved = true;
        }
        if moved {
            // Incremental spatial-hash update. Sub-cell moves early-out
            // inside `update`; cell crossings are O(1) bucket swap.
            let mode = self.hot_movement_mode[slot];
            self.spatial.update(id, pos, mode);
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
        let mut changed = false;
        if let Some(m) = self.hot_movement_mode.get_mut(slot) {
            *m = mode;
            changed = true;
        }
        if changed {
            // Pulls the agent across the walk↔non-walk boundary in the
            // spatial hash when mode crosses it; otherwise a no-op for
            // non-walk → non-walk transitions.
            let pos = self.hot_pos[slot];
            self.spatial.update(id, pos, mode);
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

    // ---- Snapshot restore helpers (#[doc(hidden)]) ----
    //
    // The `snapshot::format::load_snapshot` path needs to write every SoA
    // field + the pool + the spatial index. Rather than exposing each
    // mutable slice publicly, we gate them behind `#[doc(hidden)]`. The
    // snapshot module is the only intended caller; production code should
    // continue to use `spawn_agent` / `set_agent_*` / `kill_agent`.

    #[doc(hidden)]
    pub fn pool_next_raw(&self) -> u32 {
        self.pool.next_raw()
    }
    #[doc(hidden)]
    pub fn pool_freelist_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.pool.freelist_iter()
    }
    #[doc(hidden)]
    pub fn restore_pool_from_parts(
        &mut self,
        next_raw: u32,
        alive: Vec<bool>,
        freelist: Vec<u32>,
    ) {
        self.pool.restore_from_parts(next_raw, alive, freelist);
    }

    #[doc(hidden)] pub fn hot_pos_mut_slice(&mut self) -> &mut [Vec3] { &mut self.hot_pos }
    #[doc(hidden)] pub fn hot_hp_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_hp }
    #[doc(hidden)] pub fn hot_max_hp_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_max_hp }
    #[doc(hidden)] pub fn hot_alive_mut_slice(&mut self) -> &mut [bool] { &mut self.hot_alive }
    #[doc(hidden)] pub fn hot_movement_mode_mut_slice(&mut self) -> &mut [MovementMode] { &mut self.hot_movement_mode }
    #[doc(hidden)] pub fn hot_level_mut_slice(&mut self) -> &mut [u32] { &mut self.hot_level }
    #[doc(hidden)] pub fn hot_move_speed_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_move_speed }
    #[doc(hidden)] pub fn hot_move_speed_mult_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_move_speed_mult }
    #[doc(hidden)] pub fn hot_shield_hp_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_shield_hp }
    #[doc(hidden)] pub fn hot_armor_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_armor }
    #[doc(hidden)] pub fn hot_magic_resist_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_magic_resist }
    #[doc(hidden)] pub fn hot_attack_damage_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_attack_damage }
    #[doc(hidden)] pub fn hot_attack_range_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_attack_range }
    #[doc(hidden)] pub fn hot_mana_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_mana }
    #[doc(hidden)] pub fn hot_max_mana_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_max_mana }
    #[doc(hidden)] pub fn hot_hunger_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_hunger }
    #[doc(hidden)] pub fn hot_thirst_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_thirst }
    #[doc(hidden)] pub fn hot_rest_timer_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_rest_timer }
    #[doc(hidden)] pub fn hot_safety_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_safety }
    #[doc(hidden)] pub fn hot_shelter_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_shelter }
    #[doc(hidden)] pub fn hot_social_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_social }
    #[doc(hidden)] pub fn hot_purpose_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_purpose }
    #[doc(hidden)] pub fn hot_esteem_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_esteem }
    #[doc(hidden)] pub fn hot_risk_tolerance_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_risk_tolerance }
    #[doc(hidden)] pub fn hot_social_drive_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_social_drive }
    #[doc(hidden)] pub fn hot_ambition_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_ambition }
    #[doc(hidden)] pub fn hot_altruism_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_altruism }
    #[doc(hidden)] pub fn hot_curiosity_mut_slice(&mut self) -> &mut [f32] { &mut self.hot_curiosity }
    #[doc(hidden)] pub fn hot_engaged_with_mut_slice(&mut self) -> &mut [Option<AgentId>] { &mut self.hot_engaged_with }
    #[doc(hidden)] pub fn hot_stun_expires_at_tick_mut_slice(&mut self) -> &mut [u32] { &mut self.hot_stun_expires_at_tick }
    #[doc(hidden)] pub fn hot_slow_expires_at_tick_mut_slice(&mut self) -> &mut [u32] { &mut self.hot_slow_expires_at_tick }
    #[doc(hidden)] pub fn hot_slow_factor_q8_mut_slice(&mut self) -> &mut [i16] { &mut self.hot_slow_factor_q8 }
    #[doc(hidden)] pub fn hot_cooldown_next_ready_tick_mut_slice(&mut self) -> &mut [u32] { &mut self.hot_cooldown_next_ready_tick }

    #[doc(hidden)] pub fn cold_creature_type_mut_slice(&mut self) -> &mut [Option<CreatureType>] { &mut self.cold_creature_type }
    #[doc(hidden)] pub fn cold_spawn_tick_mut_slice(&mut self) -> &mut [Option<u32>] { &mut self.cold_spawn_tick }
    #[doc(hidden)] pub fn cold_grid_id_mut_slice(&mut self) -> &mut [Option<u32>] { &mut self.cold_grid_id }
    #[doc(hidden)] pub fn cold_local_pos_mut_slice(&mut self) -> &mut [Option<Vec3>] { &mut self.cold_local_pos }
    #[doc(hidden)] pub fn cold_move_target_mut_slice(&mut self) -> &mut [Option<Vec3>] { &mut self.cold_move_target }
    #[doc(hidden)] pub fn cold_status_effects_mut_slice(&mut self) -> &mut [SmallVec<[StatusEffect; 8]>] { &mut self.cold_status_effects }
    #[doc(hidden)] pub fn cold_memberships_mut_slice(&mut self) -> &mut [SmallVec<[Membership; 4]>] { &mut self.cold_memberships }
    #[doc(hidden)] pub fn cold_relationships_mut_slice(&mut self) -> &mut [SmallVec<[Relationship; 8]>] { &mut self.cold_relationships }
    #[doc(hidden)] pub fn cold_class_definitions_mut_slice(&mut self) -> &mut [[ClassSlot; 4]] { &mut self.cold_class_definitions }
    #[doc(hidden)] pub fn cold_creditor_ledger_mut_slice(&mut self) -> &mut [SmallVec<[Creditor; 16]>] { &mut self.cold_creditor_ledger }
    #[doc(hidden)] pub fn cold_mentor_lineage_mut_slice(&mut self) -> &mut [[Option<AgentId>; 8]] { &mut self.cold_mentor_lineage }

    /// Rebuild the incremental spatial hash from `(hot_pos, hot_alive,
    /// hot_movement_mode)`. Called by `snapshot::load_snapshot` after
    /// restoring hot SoA; the saved snapshot intentionally doesn't
    /// serialise the spatial index (it's fully derivable).
    #[doc(hidden)]
    pub fn rebuild_spatial_from_hot(&mut self) {
        self.spatial = SpatialHash::new(self.pool.cap());
        for slot in 0..self.pool.cap() as usize {
            if self.hot_alive[slot] {
                let id = AgentId::new((slot + 1) as u32).unwrap();
                self.spatial.insert(id, self.hot_pos[slot], self.hot_movement_mode[slot]);
            }
        }
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
    pub fn cold_status_effects(&self) -> &[SmallVec<[StatusEffect; 8]>] {
        &self.cold_status_effects
    }

    // Psychological-needs bulk slices (Task D).
    pub fn hot_safety(&self) -> &[f32] {
        &self.hot_safety
    }
    pub fn hot_shelter(&self) -> &[f32] {
        &self.hot_shelter
    }
    pub fn hot_social(&self) -> &[f32] {
        &self.hot_social
    }
    pub fn hot_purpose(&self) -> &[f32] {
        &self.hot_purpose
    }
    pub fn hot_esteem(&self) -> &[f32] {
        &self.hot_esteem
    }

    // Personality bulk slices (Task E).
    pub fn hot_risk_tolerance(&self) -> &[f32] {
        &self.hot_risk_tolerance
    }
    pub fn hot_social_drive(&self) -> &[f32] {
        &self.hot_social_drive
    }
    pub fn hot_ambition(&self) -> &[f32] {
        &self.hot_ambition
    }
    pub fn hot_altruism(&self) -> &[f32] {
        &self.hot_altruism
    }
    pub fn hot_curiosity(&self) -> &[f32] {
        &self.hot_curiosity
    }

    // Memberships bulk slice (Task G).
    pub fn cold_memberships(&self) -> &[SmallVec<[Membership; 4]>] {
        &self.cold_memberships
    }

    // Inventory bulk slice (Task H).
    pub fn cold_inventory(&self) -> &[Inventory] {
        &self.cold_inventory
    }

    /// Mutable inventory bulk slice. Used by the GPU backend's
    /// `snapshot()` to merge back `gold_buf` readback into the CPU
    /// SimState (Phase 3 Task 3.5 — observers reading post-batch gold
    /// values via `state.cold_inventory`). Sibling to the read-only
    /// `cold_inventory` accessor.
    pub fn cold_inventory_mut(&mut self) -> &mut [Inventory] {
        &mut self.cold_inventory
    }

    // Memory bulk slice retired 2026-04-23 (see the accessor comment
    // block above). Callers iterate through `state.views.memory` now.

    // Relationships bulk slice (Task J).
    pub fn cold_relationships(&self) -> &[SmallVec<[Relationship; 8]>] {
        &self.cold_relationships
    }

    // Task K bulk slices.
    pub fn cold_class_definitions(&self) -> &[[ClassSlot; 4]] {
        &self.cold_class_definitions
    }
    pub fn cold_creditor_ledger(&self) -> &[SmallVec<[Creditor; 16]>] {
        &self.cold_creditor_ledger
    }
    pub fn cold_mentor_lineage(&self) -> &[[Option<AgentId>; 8]] {
        &self.cold_mentor_lineage
    }

    // Engagement bulk slice (Combat Foundation Task 1).
    pub fn hot_engaged_with(&self) -> &[Option<AgentId>] {
        &self.hot_engaged_with
    }

    // Combat-timing bulk slices (Combat Foundation Task 2 + Task 143).
    pub fn hot_stun_expires_at_tick(&self) -> &[u32] {
        &self.hot_stun_expires_at_tick
    }
    pub fn hot_slow_expires_at_tick(&self) -> &[u32] {
        &self.hot_slow_expires_at_tick
    }
    pub fn hot_slow_factor_q8(&self) -> &[i16] {
        &self.hot_slow_factor_q8
    }
    pub fn hot_cooldown_next_ready_tick(&self) -> &[u32] {
        &self.hot_cooldown_next_ready_tick
    }
}
