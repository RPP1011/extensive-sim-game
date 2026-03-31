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

        Self {
            tick, threat, population, food, treasury, faction_id, infrastructure,
            faction_stance, faction_at_war,
            recent_deaths, recent_battles, recent_conquests, season,
            grid_hostile_count, coworkers,
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
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
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

        // --- Social: real coworker interaction ---
        if let Some(bid) = npc.work_building_id {
            if !matches!(npc.work_state, WorkState::Idle) {
                npc.needs.social = (npc.needs.social + 0.3).min(75.0);

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

        // --- Trade detection ---
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
                        record_npc_event(npc, MemEventType::TradedWith(*destination_settlement_id),
                            entity.pos, vec![], 0.3, world.tick);
                        npc.needs.purpose = (npc.needs.purpose + 5.0).min(80.0);
                    }
                }
            }
        }

        // --- Construction purpose ---
        if npc.work_building_id.is_some() {
            if npc.behavior_value(tags::CONSTRUCTION) > 10.0 {
                npc.needs.purpose = (npc.needs.purpose + 0.2).min(70.0);
            }
        }
    }

    // --- Emotional Contagion (every 20 ticks) ---
    if state.tick % 20 == 0 {
        spread_emotions(state);
    }
}

// ---------------------------------------------------------------------------
// Needs drift — uses WorldView for full settlement context
// ---------------------------------------------------------------------------

fn drift_needs(npc: &mut NpcData, _hp: f32, _max_hp: f32, world: &WorldView, grid_id: Option<u32>) {
    let at_settlement = npc.home_settlement_id.is_some();
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

    // Shelter from having a home building. No home = shelter decays.
    if npc.home_building_id.is_some() {
        npc.needs.shelter = (npc.needs.shelter + 0.5).min(100.0);
    } else if at_settlement {
        // At settlement but homeless: slow decay (sleeping rough).
        npc.needs.shelter = (npc.needs.shelter - 0.1).max(0.0);
    } else {
        // In the wilderness: fast decay.
        npc.needs.shelter = (npc.needs.shelter - 0.3).max(0.0);
    }

    // Safety from settlement threat + grid hostility.
    let threat = npc.home_settlement_id
        .map(|sid| world.settlement_threat(sid))
        .unwrap_or(0.0);
    let on_hostile_grid = grid_id
        .map(|gid| world.grid_has_hostiles(gid))
        .unwrap_or(false);

    if on_hostile_grid {
        npc.needs.safety = (npc.needs.safety - 5.0).max(0.0);
    } else if threat > 0.3 {
        npc.needs.safety = (npc.needs.safety - threat * 5.0).max(0.0);
    } else {
        npc.needs.safety = (npc.needs.safety + 1.0).min(100.0);
    }

    // Social decays when alone (not at settlement and no coworkers).
    if !at_settlement {
        npc.needs.social = (npc.needs.social - 0.2).max(0.0);
    }
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

/// Spread extreme emotions between nearby NPCs.
fn spread_emotions(state: &mut WorldState) {
    const CONTAGION_DIST_SQ: f32 = 100.0;

    let snapshots: Vec<(u32, (f32, f32), f32, f32, f32, f32)> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
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
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
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

            if sfear > 0.7 { fear_absorbed += sfear * 0.30; }
            if sjoy > 0.7 { joy_absorbed += sjoy * 0.25; }
            if sanger > 0.7 { anger_absorbed += sanger * 0.20; }
            if sgrief > 0.5 { grief_absorbed += sgrief * 0.15; }
        }

        if neighbors == 0 { continue; }

        let n = neighbors as f32;
        npc.emotions.fear = (npc.emotions.fear + fear_absorbed / n * resist).min(1.0);
        npc.emotions.joy = (npc.emotions.joy + joy_absorbed / n * resist).min(1.0);
        npc.emotions.anger = (npc.emotions.anger + anger_absorbed / n * resist).min(1.0);
        npc.emotions.grief = (npc.emotions.grief + grief_absorbed / n * resist).min(1.0);
    }
}

// ---------------------------------------------------------------------------
// Event recording
// ---------------------------------------------------------------------------

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
