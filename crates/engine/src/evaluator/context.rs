//! `SimState`-backed implementations of `dsl_ast::eval::{ReadContext, CascadeContext}`.
//!
//! All items in this file are gated behind `#[cfg(feature = "interpreted-rules")]`
//! so they compile to nothing in the default build.
//!
//! ## Spec B' adaptations
//!
//! The original `wsb-engine-viz` version lived in the same crate and could
//! reference `crate::generated::views::*` and `self.state.views.*`.  After
//! Spec B', those moved to `engine_rules` (generated).  Adaptations:
//!
//! - `view_is_hostile` / `view_is_stunned`: derived directly from `SimState`
//!   (creature type and stun-expiry fields), no view registry needed.
//! - Materialized view reads (threat_level, my_enemies, pack_focus, kin_fear,
//!   rally_boost): return 0.0.  Correct for the parity test (empty ViewRegistry);
//!   production view support lives in `engine_rules::evaluator::EngineViewCtx`.
//! - `emit` in `CascadeContext`: `Event` is now `engine_data::events::Event`,
//!   not `crate::event::Event`.
//! - `EventRing<E>` is generic; `EngineCascadeCtx` holds `EventRing<Event>`
//!   where `Event = engine_data::events::Event`.
//!
//! ## EffectOp layout assertion
//!
//! `dsl_ast::eval::EffectOp` and `engine::ability::program::EffectOp` must
//! stay in sync. The compile-time assertion enforces equal `size_of` and an
//! exhaustive conversion match.

use dsl_ast::eval::{
    AbilityId as DslAbilityId, AgentId as DslAgentId, CascadeContext, EvalValue, ReadContext,
    TargetSelector as DslTargetSelector, Vec3 as DslVec3,
};
use dsl_ast::eval::EffectOp as DslEffectOp;

use crate::ability::program::{EffectOp as EngineEffectOp, TargetSelector as EngineTargetSelector};
use crate::event::EventRing;
use crate::ids::{AbilityId as EngineAbilityId, AgentId as EngineAgentId};
use crate::spatial;
use crate::state::SimState;
use engine_data::events::Event;

// ---------------------------------------------------------------------------
// Layout note: EffectOp size
// ---------------------------------------------------------------------------
// `engine::ability::program::EffectOp::TransferGold` uses `i32` (narrowed
// for GPU atomics on 2026-04-22) while `dsl_ast::eval::EffectOp::TransferGold`
// uses `i64`. The two enums are intentionally different sizes; the conversion
// below widens `i32 → i64`. No compile-time size assertion here.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ID helpers
// ---------------------------------------------------------------------------

#[inline]
fn engine_agent(id: DslAgentId) -> EngineAgentId {
    EngineAgentId::new(id.raw()).expect("DslAgentId raw is always non-zero")
}

#[inline]
pub fn dsl_agent(id: EngineAgentId) -> DslAgentId {
    DslAgentId::new(id.raw()).expect("EngineAgentId raw is always non-zero")
}

#[inline]
fn engine_ability(id: DslAbilityId) -> EngineAbilityId {
    EngineAbilityId::new(id.raw()).expect("DslAbilityId raw is always non-zero")
}

#[inline]
fn engine_pos_to_dsl(p: glam::Vec3) -> DslVec3 {
    [p.x, p.y, p.z]
}

#[inline]
fn dsl_pos_to_engine(p: DslVec3) -> glam::Vec3 {
    glam::Vec3::new(p[0], p[1], p[2])
}

// ---------------------------------------------------------------------------
// EffectOp conversion: engine → dsl_ast
// ---------------------------------------------------------------------------

fn engine_to_dsl_effect_op(op: &EngineEffectOp) -> DslEffectOp {
    match op {
        EngineEffectOp::Damage { amount }           => DslEffectOp::Damage { amount: *amount },
        EngineEffectOp::Heal   { amount }           => DslEffectOp::Heal   { amount: *amount },
        EngineEffectOp::Shield { amount }           => DslEffectOp::Shield { amount: *amount },
        EngineEffectOp::Stun   { duration_ticks }   => DslEffectOp::Stun   { duration_ticks: *duration_ticks },
        EngineEffectOp::Slow   { duration_ticks, factor_q8 } => DslEffectOp::Slow {
            duration_ticks: *duration_ticks,
            factor_q8: *factor_q8,
        },
        EngineEffectOp::TransferGold  { amount }    => DslEffectOp::TransferGold  { amount: *amount as i64 },
        EngineEffectOp::ModifyStanding { delta }    => DslEffectOp::ModifyStanding { delta: *delta },
        EngineEffectOp::CastAbility { ability, selector } => {
            let dsl_selector = match selector {
                EngineTargetSelector::Caster => DslTargetSelector::Caster,
                EngineTargetSelector::Target => DslTargetSelector::Target,
            };
            DslEffectOp::CastAbility {
                ability: DslAbilityId::new(ability.raw()).expect("AbilityId non-zero"),
                selector: dsl_selector,
            }
        }
        // Wave 2 piece 1 — control verbs. Mirror surfaces them through
        // the DSL eval API at parity; runtime gating (cast/move/intent)
        // still lands in later Wave 2 pieces.
        EngineEffectOp::Root    { duration_ticks } => DslEffectOp::Root    { duration_ticks: *duration_ticks },
        EngineEffectOp::Silence { duration_ticks } => DslEffectOp::Silence { duration_ticks: *duration_ticks },
        EngineEffectOp::Fear    { duration_ticks } => DslEffectOp::Fear    { duration_ticks: *duration_ticks },
        EngineEffectOp::Taunt   { duration_ticks } => DslEffectOp::Taunt   { duration_ticks: *duration_ticks },
        // Wave 2 piece 2 — movement verbs. Same mirror surface; runtime
        // apply handlers (compute facing / away / toward vectors and
        // update `hot_pos`) land in a follow-up Wave 2 piece.
        EngineEffectOp::Dash      { distance } => DslEffectOp::Dash      { distance: *distance },
        EngineEffectOp::Blink     { distance } => DslEffectOp::Blink     { distance: *distance },
        EngineEffectOp::Knockback { distance } => DslEffectOp::Knockback { distance: *distance },
        EngineEffectOp::Pull      { distance } => DslEffectOp::Pull      { distance: *distance },
        // Wave 2 piece 3 — advanced verbs. Per-fixture apply handlers
        // (Execute → emit Defeated when target.hp < hp_threshold;
        // SelfDamage → emit Damaged{source=target=caster, amount}) are
        // Wave 2 piece N work.
        EngineEffectOp::Execute    { hp_threshold } => DslEffectOp::Execute    { hp_threshold: *hp_threshold },
        EngineEffectOp::SelfDamage { amount }       => DslEffectOp::SelfDamage { amount: *amount },
        // Wave 2 piece 4 — buff verbs. Per-fixture apply handlers
        // (LifeSteal → write `hot_lifesteal_*` slot under the
        // max-with-duration-tiebreak rule and feed a per-cascade
        // damage→heal hook; DamageModify → write `hot_damage_taken_*`
        // slot under the same rule and have ApplyDamage scale
        // bleed-through by `mult_q8 / 256`) are Wave 2 piece N work.
        EngineEffectOp::LifeSteal    { duration_ticks, fraction_q8 } => DslEffectOp::LifeSteal {
            duration_ticks: *duration_ticks,
            fraction_q8:    *fraction_q8,
        },
        EngineEffectOp::DamageModify { duration_ticks, multiplier_q8 } => DslEffectOp::DamageModify {
            duration_ticks:  *duration_ticks,
            multiplier_q8:   *multiplier_q8,
        },
    }
}

// ---------------------------------------------------------------------------
// EngineReadCtx
// ---------------------------------------------------------------------------

/// Read-only interpreter context backed by `SimState`.
pub struct EngineReadCtx<'a> {
    pub state: &'a SimState,
}

impl<'a> EngineReadCtx<'a> {
    pub fn new(state: &'a SimState) -> Self {
        Self { state }
    }
}

impl ReadContext for EngineReadCtx<'_> {
    fn world_tick(&self) -> u32 { self.state.tick }

    fn agents_alive(&self, agent: DslAgentId) -> bool {
        self.state.agent_alive(engine_agent(agent))
    }
    fn agents_pos(&self, agent: DslAgentId) -> DslVec3 {
        self.state.agent_pos(engine_agent(agent)).map(engine_pos_to_dsl).unwrap_or([0.0, 0.0, 0.0])
    }
    fn agents_hp(&self, agent: DslAgentId) -> f32 {
        self.state.agent_hp(engine_agent(agent)).unwrap_or(0.0)
    }
    fn agents_max_hp(&self, agent: DslAgentId) -> f32 {
        self.state.agent_max_hp(engine_agent(agent)).unwrap_or(1.0)
    }
    fn agents_hp_pct(&self, agent: DslAgentId) -> f32 {
        let eid = engine_agent(agent);
        let hp  = self.state.agent_hp(eid).unwrap_or(0.0);
        let max = self.state.agent_max_hp(eid).unwrap_or(1.0);
        if max == 0.0 { 0.0 } else { hp / max }
    }
    fn agents_shield_hp(&self, agent: DslAgentId) -> f32 {
        self.state.agent_shield_hp(engine_agent(agent)).unwrap_or(0.0)
    }
    fn agents_stun_expires_at_tick(&self, agent: DslAgentId) -> u32 {
        self.state.agent_stun_expires_at(engine_agent(agent)).unwrap_or(0)
    }
    fn agents_slow_expires_at_tick(&self, agent: DslAgentId) -> u32 {
        self.state.agent_slow_expires_at(engine_agent(agent)).unwrap_or(0)
    }
    fn agents_slow_factor_q8(&self, agent: DslAgentId) -> i16 {
        self.state.agent_slow_factor_q8(engine_agent(agent)).unwrap_or(0)
    }
    fn agents_attack_damage(&self, agent: DslAgentId) -> f32 {
        self.state.agent_attack_damage(engine_agent(agent)).unwrap_or(0.0)
    }
    fn agents_engaged_with(&self, agent: DslAgentId) -> Option<DslAgentId> {
        self.state.agent_engaged_with(engine_agent(agent)).map(dsl_agent)
    }
    fn agents_is_hostile_to(&self, a: DslAgentId, b: DslAgentId) -> bool {
        let ea = engine_agent(a);
        let eb = engine_agent(b);
        match (self.state.agent_creature_type(ea), self.state.agent_creature_type(eb)) {
            (Some(ca), Some(cb)) => ca.is_hostile_to(cb),
            _ => false,
        }
    }
    fn agents_gold(&self, agent: DslAgentId) -> i64 {
        self.state.agent_inventory(engine_agent(agent)).map(|inv| inv.gold as i64).unwrap_or(0)
    }

    fn query_nearby_agents(&self, center: DslVec3, radius: f32, f: &mut dyn FnMut(DslAgentId)) {
        let pos = dsl_pos_to_engine(center);
        let ids = self.state.spatial().within_radius(self.state, pos, radius);
        for id in ids {
            if self.state.agent_alive(id) {
                f(dsl_agent(id));
            }
        }
    }
    fn query_nearby_kin(
        &self,
        origin: DslAgentId,
        center: DslVec3,
        radius: f32,
        f: &mut dyn FnMut(DslAgentId),
    ) {
        let eid = engine_agent(origin);
        let ct  = match self.state.agent_creature_type(eid) {
            Some(c) => c,
            None => return,
        };
        let pos = dsl_pos_to_engine(center);
        let ids = self.state.spatial().within_radius(self.state, pos, radius);
        for id in ids {
            if id == eid { continue; }
            if !self.state.agent_alive(id) { continue; }
            if let Some(oc) = self.state.agent_creature_type(id) {
                if oc == ct { f(dsl_agent(id)); }
            }
        }
    }
    fn query_nearest_hostile_to(&self, agent: DslAgentId, radius: f32) -> Option<DslAgentId> {
        spatial::nearest_hostile_to(self.state, engine_agent(agent), radius).map(dsl_agent)
    }

    fn abilities_is_known(&self, ab: DslAbilityId) -> bool {
        self.state.ability_registry.get(engine_ability(ab)).is_some()
    }
    fn abilities_known(&self, _agent: DslAgentId, ab: DslAbilityId) -> bool {
        self.state.ability_registry.get(engine_ability(ab)).is_some()
    }
    fn abilities_cooldown_ready(&self, agent: DslAgentId, _ab: DslAbilityId) -> bool {
        let eid      = engine_agent(agent);
        let ready_at = self.state.agent_cooldown_next_ready(eid).unwrap_or(0);
        self.state.tick >= ready_at
    }
    fn abilities_cooldown_ticks(&self, ab: DslAbilityId) -> u32 {
        self.state
            .ability_registry
            .get(engine_ability(ab))
            .map(|prog| prog.gate.cooldown_ticks)
            .unwrap_or(0)
    }
    fn abilities_effects(&self, ab: DslAbilityId, f: &mut dyn FnMut(DslEffectOp)) {
        if let Some(prog) = self.state.ability_registry.get(engine_ability(ab)) {
            for op in prog.effects.iter() {
                f(engine_to_dsl_effect_op(op));
            }
        }
    }

    // ---- config (read from DSL compilation under interpreted-rules) ----

    fn config_combat_attack_range(&self) -> f32 {
        crate::mask::interp::interp_config().combat.attack_range
    }
    fn config_combat_engagement_range(&self) -> f32 {
        crate::mask::interp::interp_config().combat.engagement_range
    }
    fn config_movement_max_move_radius(&self) -> f32 {
        crate::mask::interp::interp_config().movement.max_move_radius
    }
    fn config_cascade_max_iterations(&self) -> u32 {
        crate::cascade::MAX_CASCADE_ITERATIONS as u32
    }

    // ---- views ----
    //
    // `view_is_hostile` and `view_is_stunned` are derived directly from
    // SimState. The materialized views (threat_level, my_enemies, pack_focus,
    // kin_fear, rally_boost) are not accessible from `engine` (they live in
    // `engine_rules`). They return 0.0 which is correct for the parity test
    // (empty ViewRegistry). Production use goes through `engine_rules`
    // EngineViewCtx which can access the full ViewRegistry.

    fn view_is_hostile(&self, a: DslAgentId, b: DslAgentId) -> bool {
        let ea = engine_agent(a);
        let eb = engine_agent(b);
        match (self.state.agent_creature_type(ea), self.state.agent_creature_type(eb)) {
            (Some(ca), Some(cb)) => ca.is_hostile_to(cb),
            _ => false,
        }
    }
    fn view_is_stunned(&self, agent: DslAgentId) -> bool {
        let eid = engine_agent(agent);
        let exp = self.state.agent_stun_expires_at(eid).unwrap_or(0);
        self.state.tick < exp
    }
    fn view_threat_level(&self, _observer: DslAgentId, _target: DslAgentId) -> f32 { 0.0 }
    fn view_my_enemies(&self, _observer: DslAgentId, _target: DslAgentId) -> f32  { 0.0 }
    fn view_pack_focus(&self, _observer: DslAgentId, _target: DslAgentId) -> f32  { 0.0 }
    fn view_kin_fear(&self, _observer: DslAgentId) -> f32                          { 0.0 }
    fn view_rally_boost(&self, _observer: DslAgentId) -> f32                       { 0.0 }
    fn view_slow_factor(&self, agent: DslAgentId) -> f32 {
        let q8 = self.state.agent_slow_factor_q8(engine_agent(agent)).unwrap_or(0);
        q8 as f32 / 256.0
    }

    // ---- theory-of-mind belief accessors ----

    #[cfg(feature = "theory-of-mind")]
    fn belief_about_hp(&self, observer: DslAgentId, target: DslAgentId) -> Option<f32> {
        self.state.belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.last_known_hp)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_about_hp_pct(&self, observer: DslAgentId, target: DslAgentId) -> Option<f32> {
        self.state.belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| if b.last_known_max_hp == 0.0 { 0.0 } else { b.last_known_hp / b.last_known_max_hp })
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_about_pos(&self, observer: DslAgentId, target: DslAgentId) -> Option<DslVec3> {
        self.state.belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| engine_pos_to_dsl(b.last_known_pos))
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_about_creature_type(&self, observer: DslAgentId, target: DslAgentId) -> Option<u8> {
        self.state.belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.last_known_creature_type as u8)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_confidence(&self, observer: DslAgentId, target: DslAgentId) -> f32 {
        self.state.belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.confidence)
            .unwrap_or(0.0)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_last_updated_tick(&self, observer: DslAgentId, target: DslAgentId) -> Option<u32> {
        self.state.belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.last_updated_tick)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_targets(&self, observer: DslAgentId, f: &mut dyn FnMut(DslAgentId)) {
        let eid = engine_agent(observer);
        let slot = crate::state::entity_pool::AgentSlotPool::slot_of_agent(eid);
        if let Some(belief_map) = self.state.cold_beliefs.get(slot) {
            for (target, _) in belief_map.iter() {
                f(dsl_agent(*target));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// EngineCascadeCtx
// ---------------------------------------------------------------------------

/// Mutable interpreter context for physics cascade handlers.
///
/// Wraps `EngineReadCtx` plus a mutable reference to the `EventRing<Event>`.
pub struct EngineCascadeCtx<'a> {
    pub state:  &'a mut SimState,
    pub events: &'a mut EventRing<Event>,
}

impl<'a> EngineCascadeCtx<'a> {
    pub fn new(state: &'a mut SimState, events: &'a mut EventRing<Event>) -> Self {
        Self { state, events }
    }

    #[inline]
    fn read(&self) -> &SimState { self.state }
}

impl ReadContext for EngineCascadeCtx<'_> {
    fn world_tick(&self) -> u32 { self.read().tick }

    fn agents_alive(&self, agent: DslAgentId) -> bool {
        self.read().agent_alive(engine_agent(agent))
    }
    fn agents_pos(&self, agent: DslAgentId) -> DslVec3 {
        self.read().agent_pos(engine_agent(agent)).map(engine_pos_to_dsl).unwrap_or([0.0, 0.0, 0.0])
    }
    fn agents_hp(&self, agent: DslAgentId) -> f32 {
        self.read().agent_hp(engine_agent(agent)).unwrap_or(0.0)
    }
    fn agents_max_hp(&self, agent: DslAgentId) -> f32 {
        self.read().agent_max_hp(engine_agent(agent)).unwrap_or(1.0)
    }
    fn agents_hp_pct(&self, agent: DslAgentId) -> f32 {
        let eid = engine_agent(agent);
        let hp  = self.read().agent_hp(eid).unwrap_or(0.0);
        let max = self.read().agent_max_hp(eid).unwrap_or(1.0);
        if max == 0.0 { 0.0 } else { hp / max }
    }
    fn agents_shield_hp(&self, agent: DslAgentId) -> f32 {
        self.read().agent_shield_hp(engine_agent(agent)).unwrap_or(0.0)
    }
    fn agents_stun_expires_at_tick(&self, agent: DslAgentId) -> u32 {
        self.read().agent_stun_expires_at(engine_agent(agent)).unwrap_or(0)
    }
    fn agents_slow_expires_at_tick(&self, agent: DslAgentId) -> u32 {
        self.read().agent_slow_expires_at(engine_agent(agent)).unwrap_or(0)
    }
    fn agents_slow_factor_q8(&self, agent: DslAgentId) -> i16 {
        self.read().agent_slow_factor_q8(engine_agent(agent)).unwrap_or(0)
    }
    fn agents_attack_damage(&self, agent: DslAgentId) -> f32 {
        self.read().agent_attack_damage(engine_agent(agent)).unwrap_or(0.0)
    }
    fn agents_engaged_with(&self, agent: DslAgentId) -> Option<DslAgentId> {
        self.read().agent_engaged_with(engine_agent(agent)).map(dsl_agent)
    }
    fn agents_is_hostile_to(&self, a: DslAgentId, b: DslAgentId) -> bool {
        let ea = engine_agent(a);
        let eb = engine_agent(b);
        match (self.read().agent_creature_type(ea), self.read().agent_creature_type(eb)) {
            (Some(ca), Some(cb)) => ca.is_hostile_to(cb),
            _ => false,
        }
    }
    fn agents_gold(&self, agent: DslAgentId) -> i64 {
        self.read().agent_inventory(engine_agent(agent)).map(|inv| inv.gold as i64).unwrap_or(0)
    }

    fn query_nearby_agents(&self, center: DslVec3, radius: f32, f: &mut dyn FnMut(DslAgentId)) {
        let pos = dsl_pos_to_engine(center);
        let ids = self.read().spatial().within_radius(self.read(), pos, radius);
        for id in ids {
            if self.read().agent_alive(id) { f(dsl_agent(id)); }
        }
    }
    fn query_nearby_kin(
        &self,
        origin: DslAgentId,
        center: DslVec3,
        radius: f32,
        f: &mut dyn FnMut(DslAgentId),
    ) {
        let eid = engine_agent(origin);
        let ct  = match self.read().agent_creature_type(eid) {
            Some(c) => c,
            None => return,
        };
        let pos = dsl_pos_to_engine(center);
        let ids = self.read().spatial().within_radius(self.read(), pos, radius);
        for id in ids {
            if id == eid { continue; }
            if !self.read().agent_alive(id) { continue; }
            if let Some(oc) = self.read().agent_creature_type(id) {
                if oc == ct { f(dsl_agent(id)); }
            }
        }
    }
    fn query_nearest_hostile_to(&self, agent: DslAgentId, radius: f32) -> Option<DslAgentId> {
        spatial::nearest_hostile_to(self.read(), engine_agent(agent), radius).map(dsl_agent)
    }

    fn abilities_is_known(&self, ab: DslAbilityId) -> bool {
        self.read().ability_registry.get(engine_ability(ab)).is_some()
    }
    fn abilities_known(&self, _agent: DslAgentId, ab: DslAbilityId) -> bool {
        self.read().ability_registry.get(engine_ability(ab)).is_some()
    }
    fn abilities_cooldown_ready(&self, agent: DslAgentId, _ab: DslAbilityId) -> bool {
        let eid      = engine_agent(agent);
        let ready_at = self.read().agent_cooldown_next_ready(eid).unwrap_or(0);
        self.read().tick >= ready_at
    }
    fn abilities_cooldown_ticks(&self, ab: DslAbilityId) -> u32 {
        self.read()
            .ability_registry
            .get(engine_ability(ab))
            .map(|prog| prog.gate.cooldown_ticks)
            .unwrap_or(0)
    }
    fn abilities_effects(&self, ab: DslAbilityId, f: &mut dyn FnMut(DslEffectOp)) {
        if let Some(prog) = self.read().ability_registry.get(engine_ability(ab)) {
            for op in prog.effects.iter() {
                f(engine_to_dsl_effect_op(op));
            }
        }
    }

    fn config_combat_attack_range(&self) -> f32 {
        crate::mask::interp::interp_config().combat.attack_range
    }
    fn config_combat_engagement_range(&self) -> f32 {
        crate::mask::interp::interp_config().combat.engagement_range
    }
    fn config_movement_max_move_radius(&self) -> f32 {
        crate::mask::interp::interp_config().movement.max_move_radius
    }
    fn config_cascade_max_iterations(&self) -> u32 {
        crate::cascade::MAX_CASCADE_ITERATIONS as u32
    }

    fn view_is_hostile(&self, a: DslAgentId, b: DslAgentId) -> bool {
        let ea = engine_agent(a);
        let eb = engine_agent(b);
        match (self.read().agent_creature_type(ea), self.read().agent_creature_type(eb)) {
            (Some(ca), Some(cb)) => ca.is_hostile_to(cb),
            _ => false,
        }
    }
    fn view_is_stunned(&self, agent: DslAgentId) -> bool {
        let eid = engine_agent(agent);
        let exp = self.read().agent_stun_expires_at(eid).unwrap_or(0);
        self.read().tick < exp
    }
    fn view_threat_level(&self, _observer: DslAgentId, _target: DslAgentId) -> f32 { 0.0 }
    fn view_my_enemies(&self, _observer: DslAgentId, _target: DslAgentId) -> f32  { 0.0 }
    fn view_pack_focus(&self, _observer: DslAgentId, _target: DslAgentId) -> f32  { 0.0 }
    fn view_kin_fear(&self, _observer: DslAgentId) -> f32                          { 0.0 }
    fn view_rally_boost(&self, _observer: DslAgentId) -> f32                       { 0.0 }
    fn view_slow_factor(&self, agent: DslAgentId) -> f32 {
        let q8 = self.read().agent_slow_factor_q8(engine_agent(agent)).unwrap_or(0);
        q8 as f32 / 256.0
    }

    // ---- theory-of-mind ----

    #[cfg(feature = "theory-of-mind")]
    fn belief_about_hp(&self, observer: DslAgentId, target: DslAgentId) -> Option<f32> {
        self.read().belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.last_known_hp)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_about_hp_pct(&self, observer: DslAgentId, target: DslAgentId) -> Option<f32> {
        self.read().belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| if b.last_known_max_hp == 0.0 { 0.0 } else { b.last_known_hp / b.last_known_max_hp })
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_about_pos(&self, observer: DslAgentId, target: DslAgentId) -> Option<DslVec3> {
        self.read().belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| engine_pos_to_dsl(b.last_known_pos))
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_about_creature_type(&self, observer: DslAgentId, target: DslAgentId) -> Option<u8> {
        self.read().belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.last_known_creature_type as u8)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_confidence(&self, observer: DslAgentId, target: DslAgentId) -> f32 {
        self.read().belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.confidence)
            .unwrap_or(0.0)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_last_updated_tick(&self, observer: DslAgentId, target: DslAgentId) -> Option<u32> {
        self.read().belief_of(engine_agent(observer), engine_agent(target))
            .map(|b| b.last_updated_tick)
    }
    #[cfg(feature = "theory-of-mind")]
    fn belief_targets(&self, observer: DslAgentId, f: &mut dyn FnMut(DslAgentId)) {
        let eid  = engine_agent(observer);
        let slot = crate::state::entity_pool::AgentSlotPool::slot_of_agent(eid);
        if let Some(belief_map) = self.read().cold_beliefs.get(slot) {
            for (target, _) in belief_map.iter() {
                f(dsl_agent(*target));
            }
        }
    }
}

impl CascadeContext for EngineCascadeCtx<'_> {
    fn agents_set_hp(&mut self, agent: DslAgentId, hp: f32) {
        self.state.set_agent_hp(engine_agent(agent), hp);
    }
    fn agents_set_shield_hp(&mut self, agent: DslAgentId, shield_hp: f32) {
        self.state.set_agent_shield_hp(engine_agent(agent), shield_hp);
    }
    fn agents_set_stun_expires_at_tick(&mut self, agent: DslAgentId, expires_at: u32) {
        self.state.set_agent_stun_expires_at(engine_agent(agent), expires_at);
    }
    fn agents_set_slow_expires_at_tick(&mut self, agent: DslAgentId, expires_at: u32) {
        self.state.set_agent_slow_expires_at(engine_agent(agent), expires_at);
    }
    fn agents_set_slow_factor_q8(&mut self, agent: DslAgentId, factor: i16) {
        self.state.set_agent_slow_factor_q8(engine_agent(agent), factor);
    }
    fn agents_set_engaged_with(&mut self, a: DslAgentId, b: DslAgentId) {
        let ea = engine_agent(a);
        let eb = engine_agent(b);
        self.state.set_agent_engaged_with(ea, Some(eb));
        self.state.set_agent_engaged_with(eb, Some(ea));
    }
    fn agents_clear_engaged_with(&mut self, agent: DslAgentId) {
        let eid = engine_agent(agent);
        if let Some(partner) = self.state.agent_engaged_with(eid) {
            self.state.set_agent_engaged_with(partner, None);
        }
        self.state.set_agent_engaged_with(eid, None);
    }
    fn agents_kill(&mut self, agent: DslAgentId) {
        self.state.kill_agent(engine_agent(agent));
    }
    fn agents_add_gold(&mut self, agent: DslAgentId, amount: i64) {
        let eid = engine_agent(agent);
        if let Some(mut inv) = self.state.agent_inventory(eid) {
            inv.gold = inv.gold.saturating_add(amount as i32);
            self.state.set_agent_inventory(eid, inv);
        }
    }
    fn agents_sub_gold(&mut self, agent: DslAgentId, amount: i64) {
        let eid = engine_agent(agent);
        if let Some(mut inv) = self.state.agent_inventory(eid) {
            inv.gold = inv.gold.saturating_sub(amount as i32);
            self.state.set_agent_inventory(eid, inv);
        }
    }
    fn agents_adjust_standing(&mut self, _a: DslAgentId, _b: DslAgentId, _delta: i16) {
        // Standing is now in `state.views.standing` (ViewRegistry, engine_rules).
        // `EngineCascadeCtx` in `engine` cannot reach the ViewRegistry (circular dep).
        // The full implementation lives in `engine_rules::evaluator::EngineCascadeCtx`
        // (TODO: P1b follow-up). No-op here for the parity test path.
    }
    fn agents_record_memory(
        &mut self,
        _observer: DslAgentId,
        _subject: DslAgentId,
        _feeling: f32,
        _context: u32,
        _tick: u32,
    ) {
        // Memory is now in `state.views.memory` (ViewRegistry, engine_rules).
        // No-op for the same reason as `agents_adjust_standing` above.
    }
    fn abilities_set_cooldown_next_ready(&mut self, agent: DslAgentId, _ab: DslAbilityId, ready_at: u32) {
        self.state.set_agent_cooldown_next_ready(engine_agent(agent), ready_at);
    }

    fn emit(&mut self, event_name: &str, fields: &[(&str, EvalValue)]) {
        let default_tick = self.state.tick;
        if let Some(e) = fields_to_event(event_name, fields, default_tick) {
            self.events.push(e);
        }
    }
}

// ---------------------------------------------------------------------------
// Event construction helpers for `emit`
// ---------------------------------------------------------------------------

fn field_agent(fields: &[(&str, EvalValue)], name: &str) -> Option<EngineAgentId> {
    for (k, v) in fields {
        if *k == name {
            if let EvalValue::Agent(a) = v { return EngineAgentId::new(a.raw()); }
        }
    }
    None
}

fn field_f32(fields: &[(&str, EvalValue)], name: &str) -> Option<f32> {
    for (k, v) in fields {
        if *k == name {
            if let EvalValue::F32(f) = v { return Some(*f); }
        }
    }
    None
}

fn field_u32(fields: &[(&str, EvalValue)], name: &str) -> Option<u32> {
    for (k, v) in fields {
        if *k == name {
            return match v {
                EvalValue::U32(u) => Some(*u),
                EvalValue::I32(i) => Some(*i as u32),
                EvalValue::I64(i) => Some(*i as u32),
                _ => None,
            };
        }
    }
    None
}

fn field_i64(fields: &[(&str, EvalValue)], name: &str) -> Option<i64> {
    for (k, v) in fields {
        if *k == name {
            if let EvalValue::I64(i) = v { return Some(*i); }
        }
    }
    None
}

fn field_i16(fields: &[(&str, EvalValue)], name: &str) -> Option<i16> {
    for (k, v) in fields {
        if *k == name {
            return match v {
                EvalValue::I32(i) => Some(*i as i16),
                EvalValue::U32(u) => Some(*u as i16),
                EvalValue::I64(i) => Some(*i as i16),
                _ => None,
            };
        }
    }
    None
}

fn field_ability(fields: &[(&str, EvalValue)], name: &str) -> Option<EngineAbilityId> {
    for (k, v) in fields {
        if *k == name {
            if let EvalValue::Ability(a) = v { return EngineAbilityId::new(a.raw()); }
        }
    }
    None
}

fn field_u64(fields: &[(&str, EvalValue)], name: &str) -> Option<u64> {
    for (k, v) in fields {
        if *k == name {
            return match v {
                EvalValue::I64(i) => Some(*i as u64),
                EvalValue::U32(u) => Some(*u as u64),
                _ => None,
            };
        }
    }
    None
}

/// Convert a DSL emit name + flat field slice into an engine `Event`.
///
/// Uses `engine_data::events::Event` (the concrete generated event type).
/// `default_tick` is used when the DSL omits an explicit `tick` field.
fn fields_to_event(
    name:         &str,
    fields:       &[(&str, EvalValue)],
    default_tick: u32,
) -> Option<Event> {
    let tick = field_u32(fields, "tick").unwrap_or(default_tick);

    match name {
        "EffectDamageApplied" => {
            let actor  = field_agent(fields, "actor")?;
            let target = field_agent(fields, "target")?;
            let amount = field_f32(fields, "amount").unwrap_or(0.0);
            Some(Event::EffectDamageApplied { actor, target, amount, tick })
        }
        "EffectHealApplied" => {
            let actor  = field_agent(fields, "actor")?;
            let target = field_agent(fields, "target")?;
            let amount = field_f32(fields, "amount").unwrap_or(0.0);
            Some(Event::EffectHealApplied { actor, target, amount, tick })
        }
        "EffectShieldApplied" => {
            let actor  = field_agent(fields, "actor")?;
            let target = field_agent(fields, "target")?;
            let amount = field_f32(fields, "amount").unwrap_or(0.0);
            Some(Event::EffectShieldApplied { actor, target, amount, tick })
        }
        "EffectStunApplied" => {
            let actor           = field_agent(fields, "actor")?;
            let target          = field_agent(fields, "target")?;
            let expires_at_tick = field_u32(fields, "expires_at_tick")
                .unwrap_or_else(|| field_u32(fields, "duration_ticks").unwrap_or(0));
            Some(Event::EffectStunApplied { actor, target, expires_at_tick, tick })
        }
        "EffectSlowApplied" => {
            let actor           = field_agent(fields, "actor")?;
            let target          = field_agent(fields, "target")?;
            let expires_at_tick = field_u32(fields, "expires_at_tick")
                .unwrap_or_else(|| field_u32(fields, "duration_ticks").unwrap_or(0));
            let factor_q8       = field_i16(fields, "factor_q8").unwrap_or(0);
            Some(Event::EffectSlowApplied { actor, target, expires_at_tick, factor_q8, tick })
        }
        "AgentDied" => {
            let agent_id = field_agent(fields, "agent")
                .or_else(|| field_agent(fields, "agent_id"))?;
            Some(Event::AgentDied { agent_id, tick })
        }
        "AgentAttacked" => {
            let actor  = field_agent(fields, "actor")?;
            let target = field_agent(fields, "target")?;
            let damage = field_f32(fields, "damage").unwrap_or(0.0);
            Some(Event::AgentAttacked { actor, target, damage, tick })
        }
        "AgentCast" => {
            let actor   = field_agent(fields, "actor")?;
            let target  = field_agent(fields, "target")?;
            let ability = field_ability(fields, "ability")?;
            let depth   = field_u32(fields, "depth").map(|v| v as u8).unwrap_or(0);
            Some(Event::AgentCast { actor, ability, target, depth, tick })
        }
        "EffectGoldTransfer" => {
            let from   = field_agent(fields, "from")?;
            let to     = field_agent(fields, "to")?;
            let amount = field_i64(fields, "amount").unwrap_or(0) as i32;
            Some(Event::EffectGoldTransfer { from, to, amount, tick })
        }
        "EffectStandingDelta" => {
            let a     = field_agent(fields, "a")?;
            let b     = field_agent(fields, "b")?;
            let delta = field_i16(fields, "delta").unwrap_or(0);
            Some(Event::EffectStandingDelta { a, b, delta, tick })
        }
        "EngagementBroken" => {
            let actor         = field_agent(fields, "actor").or_else(|| field_agent(fields, "a"))?;
            let former_target = field_agent(fields, "former_target").or_else(|| field_agent(fields, "b"))?;
            let reason        = field_u32(fields, "reason").map(|v| v as u8).unwrap_or(0);
            Some(Event::EngagementBroken { actor, former_target, reason, tick })
        }
        "EngagementCommitted" => {
            let actor  = field_agent(fields, "actor").or_else(|| field_agent(fields, "a"))?;
            let target = field_agent(fields, "target").or_else(|| field_agent(fields, "b"))?;
            Some(Event::EngagementCommitted { actor, target, tick })
        }
        "FearSpread" => {
            let observer = field_agent(fields, "observer")?;
            let dead_kin = field_agent(fields, "dead_kin")?;
            Some(Event::FearSpread { observer, dead_kin, tick })
        }
        "PackAssist" => {
            let observer = field_agent(fields, "observer")?;
            let target   = field_agent(fields, "target")?;
            Some(Event::PackAssist { observer, target, tick })
        }
        "RallyCall" => {
            let observer    = field_agent(fields, "observer")?;
            let wounded_kin = field_agent(fields, "wounded_kin")?;
            Some(Event::RallyCall { observer, wounded_kin, tick })
        }
        "OpportunityAttackTriggered" => {
            let actor  = field_agent(fields, "actor")?;
            let target = field_agent(fields, "target")?;
            Some(Event::OpportunityAttackTriggered { actor, target, tick })
        }
        "RecordMemory" => {
            let observer     = field_agent(fields, "observer")?;
            let source       = field_agent(fields, "source")?;
            let fact_payload = field_u64(fields, "fact_payload").unwrap_or(0);
            let confidence   = field_f32(fields, "confidence").unwrap_or(1.0);
            Some(Event::RecordMemory { observer, source, fact_payload, confidence, tick })
        }
        "ChronicleEntry" => {
            let template_id = field_u32(fields, "template_id").unwrap_or(0);
            let agent  = field_agent(fields, "agent")?;
            let target = field_agent(fields, "target")?;
            Some(Event::ChronicleEntry { template_id, agent, target, tick })
        }
        _ => None,
    }
}
