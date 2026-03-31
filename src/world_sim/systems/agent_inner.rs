//! Agent inner state system — updates NPC needs, emotions, memory, personality.
//!
//! This system bridges the gap between world events and NPC inner life.
//! A `WorldView` snapshot is built once before the entity loop, giving each
//! NPC O(1) access to settlement state, faction diplomacy, recent world
//! events, and nearby entity context — without re-scanning during mutation.
//!
//! Cadence: every 10 ticks.

use crate::world_sim::state::*;
use crate::world_sim::commodity;

const INNER_STATE_INTERVAL: u64 = 10;

// ---------------------------------------------------------------------------
// WorldView — pre-computed snapshot of world state for the agent loop
// ---------------------------------------------------------------------------

/// Everything an NPC agent needs to know about the world, pre-computed
/// once before the entity mutation loop. All lookups are O(1) by ID.
struct WorldView {
    tick: u64,

    // Per-settlement (indexed by settlement_id)
    threat: Vec<f32>,
    population: Vec<u32>,
    food: Vec<f32>,
    treasury: Vec<f32>,
    faction_id: Vec<Option<u32>>,
    infrastructure: Vec<f32>,

    // Per-faction (indexed by faction_id)
    faction_stance: Vec<DiplomaticStance>,
    faction_at_war: Vec<bool>,

    // Recent world events (this tick window)
    recent_deaths: Vec<u32>,          // entity IDs that died recently
    recent_battles: Vec<u32>,         // grid IDs with active battles
    recent_conquests: Vec<u32>,       // settlement IDs conquered recently
    season: u8,

    // Per-grid hostility (indexed by grid_id): count of hostile entities
    grid_hostile_count: Vec<u32>,

    // Coworker lookup: building_id → list of NPC entity IDs working there
    coworkers: std::collections::HashMap<u32, Vec<u32>>,

    // Building owner lookup: building entity_id → owner NPC entity_id
    building_owner: std::collections::HashMap<u32, u32>,

    // Resource node positions for perception-based discovery
    resources: Vec<(u32, (f32, f32), ResourceType)>, // (entity_id, pos, type)

    // NPC action types and authority weights for theory of mind (Phase D)
    npc_actions: std::collections::HashMap<u32, (u8, f32)>, // entity_id → (action_type, authority)
}

impl WorldView {
    fn build(state: &WorldState) -> Self {
        let tick = state.tick;
        let max_sid = state.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;
        let max_fid = state.factions.iter().map(|f| f.id).max().unwrap_or(0) as usize + 1;
        let max_gid = state.grids.iter().map(|g| g.id).max().unwrap_or(0) as usize + 1;

        // Settlement data
        let mut threat = vec![0.0f32; max_sid];
        let mut population = vec![0u32; max_sid];
        let mut food = vec![0.0f32; max_sid];
        let mut treasury = vec![0.0f32; max_sid];
        let mut faction_id = vec![None; max_sid];
        let mut infrastructure = vec![0.0f32; max_sid];
        for s in &state.settlements {
            let i = s.id as usize;
            if i < max_sid {
                threat[i] = s.threat_level;
                food[i] = s.stockpile[commodity::FOOD];
                treasury[i] = s.treasury;
                faction_id[i] = s.faction_id;
                infrastructure[i] = s.infrastructure_level;
            }
        }

        // Population count
        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(sid) = entity.npc.as_ref().and_then(|n| n.home_settlement_id) {
                if (sid as usize) < max_sid {
                    population[sid as usize] += 1;
                }
            }
        }

        // Faction data
        let mut faction_stance = vec![DiplomaticStance::Neutral; max_fid];
        let mut faction_at_war = vec![false; max_fid];
        for f in &state.factions {
            let i = f.id as usize;
            if i < max_fid {
                faction_stance[i] = f.diplomatic_stance;
                faction_at_war[i] = f.diplomatic_stance == DiplomaticStance::AtWar;
            }
        }

        // Recent world events (last 50 ticks)
        let _event_window = 50;
        let mut recent_deaths = Vec::new();
        let mut recent_battles = Vec::new();
        let mut recent_conquests = Vec::new();
        let mut season = 0u8;
        for event in &state.world_events {
            match event {
                WorldEvent::EntityDied { entity_id, .. } => {
                    recent_deaths.push(*entity_id);
                }
                WorldEvent::BattleStarted { grid_id, .. } => {
                    recent_battles.push(*grid_id);
                }
                WorldEvent::SettlementConquered { settlement_id, .. } => {
                    recent_conquests.push(*settlement_id);
                }
                WorldEvent::SeasonChanged { new_season } => {
                    season = *new_season;
                }
                _ => {}
            }
        }

        // Grid hostility: count hostile entities per grid
        let mut grid_hostile_count = vec![0u32; max_gid];
        for entity in &state.entities {
            if !entity.alive { continue; }
            if entity.team == WorldTeam::Hostile {
                if let Some(gid) = entity.grid_id {
                    if (gid as usize) < max_gid {
                        grid_hostile_count[gid as usize] += 1;
                    }
                }
            }
        }

        // Coworker lookup
        let mut coworkers: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &entity.npc {
                if let Some(bid) = npc.work_building_id {
                    coworkers.entry(bid).or_default().push(entity.id);
                }
            }
        }

        // Building owner lookup (building entity id → owner NPC id)
        let mut building_owner: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for entity in &state.entities {
            if !entity.alive { continue; }
            if let Some(b) = &entity.building {
                if let Some(oid) = b.owner_id {
                    building_owner.insert(entity.id, oid);
                }
            }
        }

        // Resource node snapshot for perception discovery
        let resources: Vec<(u32, (f32, f32), ResourceType)> = state.entities.iter()
            .filter(|e| e.alive && e.resource.is_some())
            .filter_map(|e| {
                let r = e.resource.as_ref().unwrap();
                if r.remaining > 0.0 {
                    Some((e.id, e.pos, r.resource_type))
                } else {
                    None
                }
            })
            .collect();

        // NPC action + authority snapshot for theory of mind
        let mut npc_actions: std::collections::HashMap<u32, (u8, f32)> = std::collections::HashMap::new();
        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &entity.npc {
                let action_type = npc.action.action_type_id();
                let auth = {
                    let leadership = npc.behavior_value(tags::LEADERSHIP);
                    let discipline = npc.behavior_value(tags::DISCIPLINE);
                    ((leadership + discipline * 0.5) / 20.0).min(3.0)
                };
                npc_actions.insert(entity.id, (action_type, auth));
            }
        }

        Self {
            tick, threat, population, food, treasury, faction_id, infrastructure,
            faction_stance, faction_at_war,
            recent_deaths, recent_battles, recent_conquests, season,
            grid_hostile_count, coworkers, building_owner, resources, npc_actions,
        }
    }

    fn settlement_threat(&self, sid: u32) -> f32 {
        self.threat.get(sid as usize).copied().unwrap_or(0.0)
    }
    fn settlement_pop(&self, sid: u32) -> u32 {
        self.population.get(sid as usize).copied().unwrap_or(0)
    }
    fn settlement_food(&self, sid: u32) -> f32 {
        self.food.get(sid as usize).copied().unwrap_or(0.0)
    }
    fn settlement_treasury(&self, sid: u32) -> f32 {
        self.treasury.get(sid as usize).copied().unwrap_or(0.0)
    }
    fn settlement_faction(&self, sid: u32) -> Option<u32> {
        self.faction_id.get(sid as usize).copied().flatten()
    }
    fn settlement_at_war(&self, sid: u32) -> bool {
        self.settlement_faction(sid)
            .and_then(|fid| self.faction_at_war.get(fid as usize).copied())
            .unwrap_or(false)
    }
    fn grid_has_hostiles(&self, gid: u32) -> bool {
        self.grid_hostile_count.get(gid as usize).copied().unwrap_or(0) > 0
    }
    fn coworker_ids(&self, building_id: u32) -> &[u32] {
        self.coworkers.get(&building_id).map(|v| v.as_slice()).unwrap_or(&[])
    }
    fn building_owner_id(&self, building_entity_id: u32) -> Option<u32> {
        self.building_owner.get(&building_entity_id).copied()
    }
    fn npc_action_and_authority(&self, entity_id: u32) -> (u8, f32) {
        self.npc_actions.get(&entity_id).copied().unwrap_or((0, 0.0))
    }
}

// ---------------------------------------------------------------------------
// Main update loop
// ---------------------------------------------------------------------------

/// Update all NPC inner states. Called post-apply from the runtime
/// (direct mutation, not delta-based).
pub fn update_agent_inner_states(state: &mut WorldState) {
    if state.tick % INNER_STATE_INTERVAL != 0 || state.tick == 0 { return; }

    // Build world snapshot ONCE — all per-entity lookups are O(1) from here.
    let world = WorldView::build(state);

    for entity in &mut state.entities {
        if !entity.alive { continue; }
        if !matches!(entity.kind, EntityKind::Npc | EntityKind::Monster) { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // --- Needs Drift ---
        drift_needs(npc, entity.hp, entity.max_hp, &world, entity.grid_id);

        // --- Emotion Decay ---
        npc.emotions.decay(0.02);

        // --- Morale Recovery ---
        {
            let morale_target = if npc.needs.hunger > 20.0 && npc.needs.shelter > 30.0 { 50.0 } else { 20.0 };
            let diff = morale_target - npc.morale;
            npc.morale += diff * 0.1;
            npc.morale = npc.morale.clamp(10.0, 100.0);
        }

        // --- Emotion Spikes from Needs ---
        spike_emotions_from_needs(&mut npc.emotions, &npc.needs);

        // --- Belief Decay ---
        decay_beliefs(&mut npc.memory, world.tick);

        // --- Personality Drift from Recent Events ---
        drift_personality_from_memory(&mut npc.personality, &npc.memory);

        // --- Settlement context tags ---
        if let Some(sid) = npc.home_settlement_id {
            let threat = world.settlement_threat(sid);
            let pop = world.settlement_pop(sid);
            let food = world.settlement_food(sid);
            let at_war = world.settlement_at_war(sid);

            // Under siege
            if threat > 0.5 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::DEFENSE, 1.0);
                    a.add(tags::RESILIENCE, 0.5);
                    a.add(tags::ENDURANCE, 0.5);
                    a
                });
            }

            // At war — faction is fighting, all NPCs feel the pressure
            if at_war {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::COMBAT, 0.5);
                    a.add(tags::DISCIPLINE, 0.5);
                    a.add(tags::DEFENSE, 0.3);
                    a
                });
                // War stress
                npc.emotions.anxiety = (npc.emotions.anxiety + 0.05).min(1.0);
            }

            // Near-extinction
            if pop < 20 && pop > 0 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::SURVIVAL, 2.0);
                    a.add(tags::RESILIENCE, 2.0);
                    a.add(tags::ENDURANCE, 1.0);
                    a.add(tags::LEADERSHIP, 0.5);
                    a
                });
            }

            // Last stand
            if pop <= 5 && pop > 0 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::SURVIVAL, 5.0);
                    a.add(tags::RESILIENCE, 5.0);
                    a.add(tags::COMBAT, 3.0);
                    a.add(tags::DEFENSE, 3.0);
                    a.add(tags::ENDURANCE, 2.0);
                    a.add(tags::LEADERSHIP, 2.0);
                    a
                });
            }

            // Famine
            if food < 10.0 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::SURVIVAL, 1.5);
                    a.add(tags::ENDURANCE, 1.0);
                    a
                });
                npc.emotions.anxiety = (npc.emotions.anxiety + 0.1).min(1.0);
            }

            // Prosperity boost — wealthy settlements build esteem
            if world.settlement_treasury(sid) > 500.0 {
                npc.needs.esteem = (npc.needs.esteem + 0.5).min(80.0);
            }
        }

        // --- Grid awareness: hostiles nearby ---
        if let Some(gid) = entity.grid_id {
            if world.grid_has_hostiles(gid) {
                npc.emotions.fear = (npc.emotions.fear + 0.1).min(1.0);
                npc.needs.safety = (npc.needs.safety - 3.0).max(0.0);
            }
        }

        // --- World event awareness ---
        // Recent deaths of friends generate grief
        for &dead_id in &world.recent_deaths {
            let knew_them = npc.memory.events.iter().any(|e| e.entity_ids.contains(&dead_id));
            if knew_them {
                let already_grieving = npc.memory.events.iter().rev().take(3).any(|e| {
                    matches!(e.event_type, MemEventType::FriendDied(_))
                        && world.tick.saturating_sub(e.tick) < 200
                });
                if !already_grieving {
                    record_npc_event(npc, MemEventType::FriendDied(dead_id),
                        entity.pos, vec![dead_id], -0.7, world.tick);
                }
            }
        }

        // Recent conquest of home settlement → despair/anger
        if let Some(sid) = npc.home_settlement_id {
            if world.recent_conquests.contains(&sid) {
                npc.emotions.anger = (npc.emotions.anger + 0.3).min(1.0);
                npc.emotions.grief = (npc.emotions.grief + 0.2).min(1.0);
                npc.needs.safety = (npc.needs.safety - 10.0).max(0.0);
            }
        }

        // --- Emotional state tags ---
        if npc.emotions.anger > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMBAT, 1.0);
                a.add(tags::MELEE, 0.5);
                a
            });
        }
        if npc.emotions.fear > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::AWARENESS, 1.0);
                a.add(tags::DEFENSE, 0.5);
                a.add(tags::STEALTH, 0.3);
                a
            });
        }
        if npc.emotions.grief > 0.5 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMPASSION_TAG, 1.0);
                a.add(tags::RESILIENCE, 0.5);
                a
            });
        }
        if npc.emotions.pride > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::LEADERSHIP, 0.5);
                a.add(tags::DISCIPLINE, 0.3);
                a
            });
        }
        if npc.emotions.anxiety > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::AWARENESS, 0.5);
                a.add(tags::SURVIVAL, 0.3);
                a
            });
        }

        // --- Social: real coworker interaction + relationship building ---
        if let Some(bid) = npc.work_building_id {
            if !matches!(npc.work_state, WorkState::Idle) {
                npc.needs.social = (npc.needs.social + 0.3).min(75.0);

                // Co-construction relationship bonus (every 50 ticks).
                if world.tick % 50 == 0 {
                    let coworker_ids: Vec<u32> = world.coworker_ids(bid).iter()
                        .filter(|&&cid| cid != entity.id)
                        .copied().collect();
                    for cid in &coworker_ids {
                        npc.modify_relationship(*cid, 0.03, 0.05, world.tick);
                    }
                }

                // Befriend an actual coworker (not a pseudo-ID)
                if entity_hash(entity.id, world.tick, 0xF81E) % 200 == 0 {
                    let recent_friend = npc.memory.events.iter().rev().take(3).any(|e| {
                        matches!(e.event_type, MemEventType::MadeNewFriend(_))
                            && world.tick.saturating_sub(e.tick) < 1000
                    });
                    if !recent_friend {
                        let coworkers = world.coworker_ids(bid);
                        if let Some(&friend_id) = coworkers.iter()
                            .find(|&&cid| cid != entity.id)
                        {
                            record_npc_event(npc, MemEventType::MadeNewFriend(friend_id),
                                entity.pos, vec![friend_id], 0.5, world.tick);
                        }
                    }
                }
            }
        }

        // --- Combat detection: use grid hostility instead of HP heuristic ---
        if let Some(gid) = entity.grid_id {
            if world.grid_has_hostiles(gid) && entity.hp < entity.max_hp * 0.8 {
                let recent_attack = npc.memory.events.iter().rev().take(3).any(|e| {
                    matches!(e.event_type, MemEventType::WasAttacked)
                        && world.tick.saturating_sub(e.tick) < 200
                });
                if !recent_attack {
                    record_npc_event(npc, MemEventType::WasAttacked, entity.pos, vec![], -0.5, world.tick);
                }
            }
        }

        // --- Victory detection: was attacked + grid now clear ---
        if let Some(gid) = entity.grid_id {
            if !world.grid_has_hostiles(gid) && entity.hp > entity.max_hp * 0.3 {
                let was_attacked = npc.memory.events.iter().rev().take(5).any(|e| {
                    matches!(e.event_type, MemEventType::WasAttacked)
                        && world.tick.saturating_sub(e.tick) < 300
                });
                let already_won = npc.memory.events.iter().rev().take(3).any(|e| {
                    matches!(e.event_type, MemEventType::WonFight)
                        && world.tick.saturating_sub(e.tick) < 300
                });
                if was_attacked && !already_won {
                    record_npc_event(npc, MemEventType::WonFight, entity.pos, vec![], 0.6, world.tick);
                }
            }
        }

        // --- Level-up awareness ---
        if entity.level > 1 && entity.level % 10 == 0 {
            let already_learned = npc.memory.events.iter().rev().take(3).any(|e| {
                matches!(e.event_type, MemEventType::LearnedSkill)
                    && world.tick.saturating_sub(e.tick) < 500
            });
            if !already_learned {
                record_npc_event(npc, MemEventType::LearnedSkill, entity.pos, vec![], 0.4, world.tick);
            }
        }

        // --- Trade detection + relationship boost ---
        if matches!(npc.economic_intent, EconomicIntent::Trade { .. }) {
            let has_goods = entity.inventory.as_ref()
                .map(|inv| inv.commodities.iter().any(|&g| g > 0.1))
                .unwrap_or(false);
            if has_goods {
                let recent_trade = npc.memory.events.iter().rev().take(3).any(|e| {
                    matches!(e.event_type, MemEventType::TradedWith(_))
                        && world.tick.saturating_sub(e.tick) < 500
                });
                if !recent_trade {
                    if let EconomicIntent::Trade { destination_settlement_id } = &npc.economic_intent {
                        let dest_sid = *destination_settlement_id;
                        record_npc_event(npc, MemEventType::TradedWith(dest_sid),
                            entity.pos, vec![], 0.3, world.tick);
                        npc.needs.purpose = (npc.needs.purpose + 5.0).min(80.0);
                        // Trade builds trust with destination settlement NPCs.
                        npc.modify_relationship(dest_sid, 0.05, 0.1, world.tick);
                    }
                }
            }
        }

        // --- Construction purpose + compassion bonus ---
        if let Some(bid) = npc.work_building_id {
            if npc.behavior_value(tags::CONSTRUCTION) > 10.0 {
                npc.needs.purpose = (npc.needs.purpose + 0.2).min(70.0);
            }
            // Compassionate NPCs get purpose boost when helping build others' buildings.
            if npc.personality.compassion > 0.6 {
                let is_others_building = world.building_owner_id(bid)
                    .map(|oid| oid != entity.id)
                    .unwrap_or(false);
                if is_others_building {
                    npc.needs.purpose = (npc.needs.purpose + 0.3).min(80.0);
                }
            }
        }

        // --- Aspiration: recompute need-vector every 500 ticks (Phase B) ---
        if world.tick.saturating_sub(npc.aspiration.vector_formed_at) >= 500 || npc.aspiration.vector_formed_at == 0 {
            compute_aspiration_vector(npc, world.tick);
        }

        // --- Aspiration crystal lifecycle ---
        if let Some(ref crystal) = npc.aspiration.crystal {
            // Expiry: no progress in 1000 ticks → dissolve with frustration.
            if world.tick.saturating_sub(npc.aspiration.crystal_last_advanced) > 1000 {
                npc.emotions.anxiety = (npc.emotions.anxiety + 0.2).min(1.0);
                npc.aspiration.crystal = None;
                npc.aspiration.crystal_progress = 0.0;
            }
            // Hard cap: 5000 ticks max lifetime.
            else if world.tick.saturating_sub(crystal.formed_at) > 5000 {
                npc.aspiration.crystal = None;
                npc.aspiration.crystal_progress = 0.0;
            }
            // Completion check.
            else if npc.aspiration.crystal_progress >= 1.0 {
                npc.needs.esteem = (npc.needs.esteem + 20.0).min(100.0);
                npc.needs.purpose = (npc.needs.purpose + 20.0).min(100.0);
                npc.emotions.pride = (npc.emotions.pride + 0.3).min(1.0);
                npc.aspiration.crystal = None;
                npc.aspiration.crystal_progress = 0.0;
                // Immediate recompute.
                compute_aspiration_vector(npc, world.tick);
            }
        }

        // --- Crystal formation from memory events ---
        if npc.aspiration.crystal.is_none() || npc.aspiration.crystal_progress < 0.1 {
            try_crystallize_from_memory(npc, world.tick);
        }

        // --- Resource perception: discover nearby resource nodes ---
        {
            // Base perception: 30 units. Curious NPCs: 45 units.
            // Passive ability perception bonus (Phase F) stacks multiplicatively.
            let base_radius = if npc.personality.curiosity > 0.6 { 45.0 } else { 30.0 };
            let effective_radius = base_radius * npc.passive_effects.perception_mult;
            let perception_radius_sq = effective_radius * effective_radius;
            for &(rid, rpos, rtype) in &world.resources {
                let dx = entity.pos.0 - rpos.0;
                let dy = entity.pos.1 - rpos.1;
                if dx * dx + dy * dy <= perception_radius_sq {
                    npc.known_resources.insert(rid, ResourceKnowledge {
                        pos: rpos,
                        resource_type: rtype,
                        observed_tick: world.tick,
                    });
                }
            }
        }

        // --- Relationship decay + social need from trusted companions (every 50 ticks) ---
        if world.tick % 50 == 0 {
            // Decay familiarity and trust.
            for rel in npc.relationships.values_mut() {
                rel.familiarity = (rel.familiarity - 0.005).max(0.0);
                rel.trust *= 0.999; // slow drift toward 0
            }
            // Prune negligible relationships.
            npc.relationships.retain(|_, r| r.familiarity >= 0.01 || r.trust.abs() >= 0.05);

            // Proximity to trusted, familiar NPCs satisfies social need.
            // Compatibility (Phase D) amplifies satisfaction from perceived-similar NPCs.
            let pers_copy = npc.personality;
            let mut social_bonus = 0.0f32;
            for rel in npc.relationships.values() {
                if rel.trust > 0.3 && rel.familiarity > 0.3 {
                    let compat = rel.perceived_personality.compatibility(&pers_copy);
                    social_bonus += 0.5 * (0.5 + 0.5 * compat); // 0.25-0.5 per friend
                }
            }
            if social_bonus > 0.0 && npc.home_settlement_id.is_some() {
                npc.needs.social = (npc.needs.social + social_bonus.min(2.0)).min(80.0);
            }

            // Resource knowledge decay: remove stale entries (>2000 ticks old).
            npc.known_resources.retain(|_, k| world.tick.saturating_sub(k.observed_tick) < 2000);

            // Theory of mind: observe coworker actions to build personality models (Phase D).
            if let Some(bid) = npc.work_building_id {
                let coworker_ids: Vec<u32> = world.coworker_ids(bid).iter()
                    .filter(|&&cid| cid != entity.id)
                    .copied().collect();
                for cid in coworker_ids {
                    let (action_type, auth) = world.npc_action_and_authority(cid);
                    if let Some(rel) = npc.relationships.get_mut(&cid) {
                        rel.perceived_personality.observe_action(action_type, auth);
                    }
                }
            }

            // Outcome EMA decay: curious NPCs forget bad outcomes faster (Phase A).
            if npc.personality.curiosity > 0.6 {
                for ema in npc.action_outcomes.values_mut() {
                    ema.value *= 0.995; // drift toward neutral
                }
            }

            // Price belief decay (Phase E): stale beliefs lose confidence.
            for belief in &mut npc.price_beliefs {
                if world.tick.saturating_sub(belief.last_updated) > 200 {
                    belief.decay();
                }
            }
        }
    }

    // --- Emotional Contagion (every 20 ticks) ---
    if state.tick % 20 == 0 {
        spread_emotions(state);
    }

    // --- Witness Events (death/battle reactions) ---
    process_witness_events(state);

    // --- Resource knowledge sharing (every 100 ticks) ---
    if state.tick % 100 == 0 {
        share_resource_knowledge(state);
    }

    // --- Cultural norm update (every 200 ticks, Phase C) ---
    if state.tick % 200 == 0 {
        update_cultural_norms(state);
    }

    // --- Passive effects recompute (every 200 ticks, Phase F) ---
    if state.tick % 200 == 0 {
        for entity in &mut state.entities {
            if !entity.alive { continue; }
            if let Some(npc) = &mut entity.npc {
                npc.passive_effects = PassiveEffects::compute(&npc.behavior_profile);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Needs drift — uses WorldView for full settlement context
// ---------------------------------------------------------------------------

fn drift_needs(npc: &mut NpcData, _hp: f32, _max_hp: f32, world: &WorldView, grid_id: Option<u32>) {
    let at_settlement = npc.home_settlement_id.is_some();
    let pers = npc.personality; // Copy — Personality is Copy

    // --- Hunger (unchanged by personality) ---
    let hunger_drain = if at_settlement { 0.15 } else { 0.3 };
    npc.needs.hunger = (npc.needs.hunger - hunger_drain).max(0.0);

    if npc.needs.hunger < 5.0 {
        let recent_starve = npc.memory.events.iter().rev().take(3).any(|e| {
            matches!(e.event_type, MemEventType::Starved)
                && world.tick.saturating_sub(e.tick) < 500
        });
        if !recent_starve {
            record_npc_event(npc, MemEventType::Starved, (0.0, 0.0), vec![], -0.6, world.tick);
        }
    }

    // --- Shelter (unchanged by personality) ---
    if npc.home_building_id.is_some() {
        npc.needs.shelter = (npc.needs.shelter + 0.5).min(100.0);
    } else if at_settlement {
        npc.needs.shelter = (npc.needs.shelter - 0.1).max(0.0);
    } else {
        npc.needs.shelter = (npc.needs.shelter - 0.3).max(0.0);
    }

    // --- Safety (modulated by risk tolerance) ---
    // High risk tolerance → less bothered by threats (×0.6 decay).
    // Low risk tolerance → paranoid (×1.5 decay).
    let safety_mult = if pers.risk_tolerance > 0.7 {
        0.6
    } else if pers.risk_tolerance < 0.3 {
        1.5
    } else {
        1.0
    };

    let threat = npc.home_settlement_id
        .map(|sid| world.settlement_threat(sid))
        .unwrap_or(0.0);
    let on_hostile_grid = grid_id
        .map(|gid| world.grid_has_hostiles(gid))
        .unwrap_or(false);

    if on_hostile_grid {
        npc.needs.safety = (npc.needs.safety - 5.0 * safety_mult).max(0.0);
    } else if threat > 0.3 {
        npc.needs.safety = (npc.needs.safety - threat * 5.0 * safety_mult).max(0.0);
    } else {
        npc.needs.safety = (npc.needs.safety + 1.0).min(100.0);
    }

    // --- Social (modulated by social drive) ---
    // High social drive → needs others more (×1.5 decay when isolated).
    // Low social drive → loner (×0.5 decay).
    let social_mult = if pers.social_drive > 0.6 {
        1.5
    } else if pers.social_drive < 0.3 {
        0.5
    } else {
        1.0
    };

    if !at_settlement {
        npc.needs.social = (npc.needs.social - 0.2 * social_mult).max(0.0);
    }

    // --- Purpose (base decay, modulated by ambition) ---
    // High ambition → needs meaningful work more (×1.5 decay).
    let purpose_decay = 0.15;
    let purpose_mult = if pers.ambition > 0.6 { 1.5 } else { 1.0 };
    npc.needs.purpose = (npc.needs.purpose - purpose_decay * purpose_mult).max(0.0);

    // Curiosity: stationary NPCs with high curiosity lose purpose faster.
    if pers.curiosity > 0.6 && matches!(npc.work_state, WorkState::Idle) && matches!(npc.action, NpcAction::Idle) {
        npc.needs.purpose = (npc.needs.purpose - 0.1).max(0.0);
    }

    // --- Esteem (base decay, modulated by ambition) ---
    let esteem_decay = 0.1;
    let esteem_mult = if pers.ambition > 0.6 { 1.3 } else { 1.0 };
    npc.needs.esteem = (npc.needs.esteem - esteem_decay * esteem_mult).max(0.0);
}

// ---------------------------------------------------------------------------
// Emotion helpers (unchanged)
// ---------------------------------------------------------------------------

fn spike_emotions_from_needs(emotions: &mut Emotions, needs: &Needs) {
    if needs.hunger < 15.0 {
        emotions.anxiety = (emotions.anxiety + 0.05).min(1.0);
    }
    if needs.safety < 20.0 {
        emotions.fear = (emotions.fear + 0.08).min(1.0);
    }
    if needs.social < 15.0 {
        emotions.grief = (emotions.grief + 0.02).min(1.0);
    }
    if needs.purpose < 15.0 {
        emotions.grief = (emotions.grief + 0.01).min(1.0);
    }
    // Positive spikes
    if needs.hunger > 80.0 && needs.safety > 70.0 {
        emotions.joy = (emotions.joy + 0.02).min(1.0);
    }
}

fn decay_beliefs(memory: &mut Memory, tick: u64) {
    // Beliefs older than 5000 ticks lose relevance.
    const BELIEF_TTL: u64 = 5000;
    memory.beliefs.retain(|b| tick.saturating_sub(b.formed_tick) < BELIEF_TTL);
}

fn drift_personality_from_memory(personality: &mut Personality, memory: &Memory) {
    let attacks = memory.events.iter().rev().take(20)
        .filter(|e| matches!(e.event_type, MemEventType::WasAttacked)).count();
    let wins = memory.events.iter().rev().take(20)
        .filter(|e| matches!(e.event_type, MemEventType::WonFight)).count();
    let deaths = memory.events.iter().rev().take(20)
        .filter(|e| matches!(e.event_type, MemEventType::FriendDied(_))).count();

    if attacks > 3 {
        personality.risk_tolerance = (personality.risk_tolerance - 0.01).max(0.0);
    }
    if wins > 2 {
        personality.risk_tolerance = (personality.risk_tolerance + 0.005).min(1.0);
        personality.ambition = (personality.ambition + 0.005).min(1.0);
    }
    if deaths > 1 {
        personality.compassion = (personality.compassion + 0.01).min(1.0);
    }
}

/// Spread extreme emotions between nearby NPCs, weighted by relationships.
fn spread_emotions(state: &mut WorldState) {
    const CONTAGION_DIST_SQ: f32 = 100.0; // 10-unit radius

    let snapshots: Vec<(u32, (f32, f32), f32, f32, f32, f32)> = state.entities.iter()
        .filter(|e| e.alive && e.npc.is_some())
        .filter(|e| matches!(e.kind, EntityKind::Npc | EntityKind::Monster))
        .filter_map(|e| {
            let em = &e.npc.as_ref().unwrap().emotions;
            if em.fear > 0.7 || em.joy > 0.7 || em.anger > 0.7 || em.grief > 0.5 {
                Some((e.id, e.pos, em.fear, em.joy, em.anger, em.grief))
            } else {
                None
            }
        })
        .take(50)
        .collect();

    let slot = (state.tick / 20) % 4;
    for entity in &mut state.entities {
        if !entity.alive { continue; }
        if !matches!(entity.kind, EntityKind::Npc | EntityKind::Monster) { continue; }
        if entity.id as u64 % 4 != slot { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        let resilience = npc.behavior_value(tags::RESILIENCE);
        let resist = 1.0 / (1.0 + resilience * 0.01);

        let mut fear_absorbed = 0.0f32;
        let mut joy_absorbed = 0.0f32;
        let mut anger_absorbed = 0.0f32;
        let mut grief_absorbed = 0.0f32;
        let mut neighbors = 0u32;

        for &(sid, spos, sfear, sjoy, sanger, sgrief) in &snapshots {
            if sid == entity.id { continue; }
            let dx = spos.0 - entity.pos.0;
            let dy = spos.1 - entity.pos.1;
            if dx * dx + dy * dy > CONTAGION_DIST_SQ { continue; }

            neighbors += 1;
            if neighbors > 5 { break; }

            // Relationship trust amplifies emotional contagion from friends.
            let trust_scale = 1.0 + npc.trust_toward(sid).max(0.0);

            if sfear > 0.7 { fear_absorbed += sfear * 0.30 * trust_scale; }
            if sjoy > 0.7 { joy_absorbed += sjoy * 0.25 * trust_scale; }
            if sanger > 0.7 { anger_absorbed += sanger * 0.20 * trust_scale; }
            if sgrief > 0.5 { grief_absorbed += sgrief * 0.15 * trust_scale; }
        }

        if neighbors == 0 { continue; }

        // Cascade guard: cap total emotion change to ±0.3 per tick.
        let n = neighbors as f32;
        let clamp = |v: f32| v.min(0.3);
        npc.emotions.fear = (npc.emotions.fear + clamp(fear_absorbed / n * resist)).min(1.0);
        npc.emotions.joy = (npc.emotions.joy + clamp(joy_absorbed / n * resist)).min(1.0);
        npc.emotions.anger = (npc.emotions.anger + clamp(anger_absorbed / n * resist)).min(1.0);
        npc.emotions.grief = (npc.emotions.grief + clamp(grief_absorbed / n * resist)).min(1.0);
    }
}

/// Process witness events from recent world events (deaths, battles).
/// NPCs nearby react emotionally, scaled by relationship trust.
fn process_witness_events(state: &mut WorldState) {
    const WITNESS_RANGE_SQ: f32 = 225.0; // 15-unit radius

    // Collect recent deaths this tick.
    let deaths: Vec<(u32, (f32, f32))> = state.world_events.iter()
        .filter_map(|ev| {
            if let WorldEvent::EntityDied { entity_id, .. } = ev {
                // Find position of the dead entity (may already be marked dead).
                state.entities.iter()
                    .find(|e| e.id == *entity_id)
                    .map(|e| (*entity_id, e.pos))
            } else {
                None
            }
        })
        .collect();

    // Collect recent battle victories this tick.
    let victories: Vec<(u32, WorldTeam)> = state.world_events.iter()
        .filter_map(|ev| {
            if let WorldEvent::BattleEnded { grid_id, victor_team } = ev {
                Some((*grid_id, *victor_team))
            } else {
                None
            }
        })
        .collect();

    if deaths.is_empty() && victories.is_empty() { return; }

    for entity in &mut state.entities {
        if !entity.alive { continue; }
        if !matches!(entity.kind, EntityKind::Npc | EntityKind::Monster) { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        let mut grief_delta = 0.0f32;
        let mut fear_delta = 0.0f32;
        let mut joy_delta = 0.0f32;
        let mut pride_delta = 0.0f32;
        let mut anger_delta = 0.0f32;

        // Witness deaths.
        for &(dead_id, dead_pos) in &deaths {
            if dead_id == entity.id { continue; }
            let dx = entity.pos.0 - dead_pos.0;
            let dy = entity.pos.1 - dead_pos.1;
            if dx * dx + dy * dy > WITNESS_RANGE_SQ { continue; }

            let trust = npc.trust_toward(dead_id).max(0.0);
            grief_delta += 0.3 * trust + 0.1; // base grief for any death
            fear_delta += 0.2;
            // Attacked by someone who killed a friend → anger.
            if trust > 0.2 {
                anger_delta += 0.2 * trust;
            }
        }

        // Witness battle victories (monsters killed on our grid → joy/pride).
        for &(grid_id, _victor_team) in &victories {
            if entity.grid_id == Some(grid_id) {
                joy_delta += 0.1;
                pride_delta += 0.05;
            }
        }

        // Apply with cascade guard: cap at ±0.3 per emotion per tick.
        npc.emotions.grief = (npc.emotions.grief + grief_delta.min(0.3)).min(1.0);
        npc.emotions.fear = (npc.emotions.fear + fear_delta.min(0.3)).min(1.0);
        npc.emotions.anger = (npc.emotions.anger + anger_delta.min(0.3)).min(1.0);
        npc.emotions.joy = (npc.emotions.joy + joy_delta.min(0.3)).min(1.0);
        npc.emotions.pride = (npc.emotions.pride + pride_delta.min(0.3)).min(1.0);
    }
}

// ---------------------------------------------------------------------------
// Event recording
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Authority weight (shared by Phase C, D)
// ---------------------------------------------------------------------------

/// Authority weight from leadership-related behavior tags. Scales with tag accumulation.
fn authority_weight(npc: &NpcData) -> f32 {
    let leadership = npc.behavior_value(tags::LEADERSHIP);
    let discipline = npc.behavior_value(tags::DISCIPLINE);
    // Every 20 accumulated leadership ≈ 1.0 authority weight.
    ((leadership + discipline * 0.5) / 20.0).min(3.0)
}

// ---------------------------------------------------------------------------
// Cultural emergence (Phase C)
// ---------------------------------------------------------------------------

/// Update cultural bias via three-channel conformity (frequency/prestige/authority).
/// Called every 200 ticks from the main loop, as a separate pass.
fn update_cultural_norms(state: &mut WorldState) {
    const CULTURE_RADIUS_SQ: f32 = 900.0; // 30 units

    // Phase 1: snapshot NPC positions, action types, esteem, authority.
    struct CultureSnap {
        id: u32,
        pos: (f32, f32),
        action_type: u8,
        esteem: f32,
        authority: f32,
    }
    let snaps: Vec<CultureSnap> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc)
        .filter_map(|e| {
            let npc = e.npc.as_ref()?;
            Some(CultureSnap {
                id: e.id,
                pos: e.pos,
                action_type: npc.action.action_type_id(),
                esteem: npc.needs.esteem,
                authority: authority_weight(npc),
            })
        })
        .collect();

    // Phase 2: for each NPC, sample nearby NPCs and blend cultural bias.
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };
        // Monsters don't participate in culture.
        if !matches!(npc.creature_type, CreatureType::Citizen) { continue; }

        let mut freq = [0.0f32; 12];
        let mut prestige = [0.0f32; 12];
        let mut authority_ch = [0.0f32; 12];
        let mut has_authority = false;
        let mut count = 0u32;

        for snap in &snaps {
            if snap.id == entity.id { continue; }
            let dx = entity.pos.0 - snap.pos.0;
            let dy = entity.pos.1 - snap.pos.1;
            if dx * dx + dy * dy > CULTURE_RADIUS_SQ { continue; }
            if count >= 10 { break; } // cap sample
            count += 1;

            let at = snap.action_type as usize;
            if at >= 12 { continue; }

            freq[at] += 1.0;
            prestige[at] += snap.esteem / 100.0;
            if snap.authority > 0.0 {
                authority_ch[at] += snap.authority;
                has_authority = true;
            }
        }

        if count == 0 {
            // Isolated: slow decay toward neutral.
            for b in &mut npc.cultural_bias { *b *= 0.999; }
            continue;
        }

        // Normalize channels.
        let normalize = |arr: &mut [f32; 12]| {
            let sum: f32 = arr.iter().sum();
            if sum > 0.0 { for v in arr.iter_mut() { *v /= sum; } }
        };
        normalize(&mut freq);
        normalize(&mut prestige);
        if has_authority { normalize(&mut authority_ch); }

        // Blend channels.
        let mut target = [0.0f32; 12];
        if has_authority {
            for i in 0..12 {
                target[i] = 0.4 * freq[i] + 0.2 * prestige[i] + 0.4 * authority_ch[i];
            }
        } else {
            for i in 0..12 {
                target[i] = 0.7 * freq[i] + 0.3 * prestige[i];
            }
        }

        // Conformity rate modulated by personality.
        let mut conformity = (0.05
            + npc.personality.social_drive * 0.05
            - npc.personality.curiosity * 0.03)
            .clamp(0.01, 0.1);
        // Authority figures resist cultural drift.
        if authority_weight(npc) > 0.5 { conformity *= 0.3; }

        // Drift toward local norm.
        for i in 0..12 {
            npc.cultural_bias[i] += conformity * (target[i] - 0.5) * 0.1;
            npc.cultural_bias[i] = npc.cultural_bias[i].clamp(-0.3, 0.3);
        }
    }
}

// ---------------------------------------------------------------------------
// Aspiration helpers (Phase B)
// ---------------------------------------------------------------------------

/// Compute the aspiration need-vector from personality-weighted need gaps.
fn compute_aspiration_vector(npc: &mut NpcData, tick: u64) {
    let pers = npc.personality;
    let needs = &npc.needs;

    // Personality-weighted need importance.
    let weights = [
        1.0,                                                          // hunger (universal)
        if pers.risk_tolerance < 0.3 { 1.5 } else { 1.0 },         // safety
        1.0,                                                          // shelter
        if pers.social_drive > 0.6 { 1.5 } else { 1.0 },           // social
        {                                                              // purpose
            let mut w = 1.0;
            if pers.ambition > 0.6 { w *= 1.5; }
            if pers.compassion > 0.6 { w *= 1.2; }
            if pers.curiosity > 0.6 { w *= 1.2; }
            w
        },
        if pers.ambition > 0.6 { 1.5 } else { 1.0 },               // esteem
    ];

    let need_values = [needs.hunger, needs.safety, needs.shelter,
                       needs.social, needs.purpose, needs.esteem];

    let mut gap = [0.0f32; NUM_NEEDS];
    let mut sum = 0.0f32;
    for i in 0..NUM_NEEDS {
        gap[i] = weights[i] * (100.0 - need_values[i]) / 100.0;
        // Emotional blending: grief/fear boost safety gap.
        if i == 1 && (npc.emotions.grief > 0.3 || npc.emotions.fear > 0.3) {
            gap[i] += 0.2;
        }
        // Frustration from chronic action failure boosts the relevant need.
        // (Simplified: boost purpose when any work-related outcome is negative.)
        if i == 4 { // purpose
            for ema in npc.action_outcomes.values() {
                if ema.value < -0.3 {
                    gap[i] += 0.15;
                    break; // one boost is enough
                }
            }
        }
        sum += gap[i];
    }
    // Normalize.
    if sum > 0.0 {
        for g in &mut gap { *g /= sum; }
    }
    npc.aspiration.need_vector = gap;
    npc.aspiration.vector_formed_at = tick;
}

/// Try to form a crystal from recent memory events.
fn try_crystallize_from_memory(npc: &mut NpcData, tick: u64) {
    // Scan last 5 memory events for crystallization candidates.
    for event in npc.memory.events.iter().rev().take(5) {
        if tick.saturating_sub(event.tick) > 500 { continue; }

        let (need_idx, target) = match &event.event_type {
            MemEventType::BuiltSomething => (4, None), // purpose, no specific target
            MemEventType::LearnedSkill => {
                // Crystal on highest-level class.
                let class_target = npc.classes.iter().max_by_key(|c| c.level)
                    .map(|c| CrystalTarget::Class(c.class_name_hash));
                (5, class_target) // esteem
            }
            MemEventType::TradedWith(partner_id) => {
                (3, Some(CrystalTarget::Entity(*partner_id))) // social
            }
            MemEventType::WasAttacked => {
                (1, None) // safety, no specific attacker target available
            }
            MemEventType::MadeNewFriend(friend_id) => {
                (3, Some(CrystalTarget::Entity(*friend_id))) // social
            }
            _ => continue,
        };

        let Some(target) = target else { continue };

        // Check if this need is a dominant aspiration dimension.
        if npc.aspiration.need_vector[need_idx] < 0.25 { continue; }

        npc.aspiration.crystal = Some(Crystal {
            need_idx,
            target,
            formed_at: tick,
        });
        npc.aspiration.crystal_progress = 0.0;
        npc.aspiration.crystal_last_advanced = tick;
        break; // only one crystal at a time
    }
}

/// Share resource knowledge between NPCs in the same settlement.
/// Each settlement pools known resources from all its NPCs, then each NPC
/// gains entries they didn't already have (with the original observer's tick).
fn share_resource_knowledge(state: &mut WorldState) {
    // Phase 1: collect pooled knowledge per settlement.
    let mut pools: std::collections::HashMap<u32, Vec<(u32, ResourceKnowledge)>> =
        std::collections::HashMap::new();
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        let sid = match npc.home_settlement_id { Some(s) => s, None => continue };
        let pool = pools.entry(sid).or_default();
        for (&rid, k) in &npc.known_resources {
            pool.push((rid, k.clone()));
        }
    }

    // Phase 2: distribute pooled knowledge to settlement NPCs.
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };
        let sid = match npc.home_settlement_id { Some(s) => s, None => continue };
        if let Some(pool) = pools.get(&sid) {
            for (rid, k) in pool {
                npc.known_resources.entry(*rid).or_insert_with(|| k.clone());
            }
        }
    }
}

/// Record a memory event on an NPC.
pub fn record_npc_event(
    npc: &mut NpcData,
    event_type: MemEventType,
    location: (f32, f32),
    entity_ids: Vec<u32>,
    emotional_impact: f32,
    tick: u64,
) {
    npc.memory.record_event(MemoryEvent {
        tick,
        event_type,
        location,
        entity_ids,
        emotional_impact,
    });

    if emotional_impact > 0.3 {
        npc.emotions.joy = (npc.emotions.joy + emotional_impact * 0.5).min(1.0);
        npc.emotions.pride = (npc.emotions.pride + emotional_impact * 0.3).min(1.0);
    }
    if emotional_impact < -0.3 {
        let neg = -emotional_impact;
        npc.emotions.fear = (npc.emotions.fear + neg * 0.4).min(1.0);
        npc.emotions.grief = (npc.emotions.grief + neg * 0.3).min(1.0);
        npc.emotions.anger = (npc.emotions.anger + neg * 0.2).min(1.0);
    }
}
