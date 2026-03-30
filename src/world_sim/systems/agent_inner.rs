#![allow(unused)]
//! Agent inner state system — updates NPC needs, emotions, memory, personality.
//!
//! This system bridges the gap between world events and NPC inner life.
//! Needs drift based on world state. Events from combat/trade/social interactions
//! are recorded as memories. Repeated patterns form beliefs. Emotions spike
//! from events and decay over time. Personality shifts from experience.
//!
//! Cadence: every 10 ticks.

use crate::world_sim::state::*;

const INNER_STATE_INTERVAL: u64 = 10;

/// Update all NPC inner states. Called post-apply from the runtime
/// (direct mutation, not delta-based).
pub fn update_agent_inner_states(state: &mut WorldState) {
    if state.tick % INNER_STATE_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Build flat arrays indexed by settlement ID for O(1) lookup.
    let max_sid = state.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;
    let mut threat_by_sid = vec![0.0f32; max_sid];
    let mut pop_by_sid = vec![0u32; max_sid];
    for s in &state.settlements {
        if (s.id as usize) < max_sid {
            threat_by_sid[s.id as usize] = s.threat_level;
        }
    }
    for entity in &state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        if let Some(sid) = entity.npc.as_ref().and_then(|n| n.home_settlement_id) {
            if (sid as usize) < max_sid {
                pop_by_sid[sid as usize] += 1;
            }
        }
    }

    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // --- Needs Drift ---
        drift_needs(npc, entity.hp, entity.max_hp, &threat_by_sid, tick);

        // --- Emotion Decay ---
        npc.emotions.decay(0.02); // ~5% decay per interval

        // --- Morale Recovery ---
        // Morale drifts toward 50 when basic needs met. Hard floor at 10.
        // This prevents permanent morale death spirals from accumulated grief systems.
        {
            let morale_target = if npc.needs.hunger > 20.0 && npc.needs.shelter > 30.0 { 50.0 } else { 20.0 };
            let diff = morale_target - npc.morale;
            npc.morale += diff * 0.1; // 10% drift per interval — strong recovery
            npc.morale = npc.morale.clamp(10.0, 100.0); // hard floor at 10
        }

        // --- Emotion Spikes from Needs ---
        spike_emotions_from_needs(&mut npc.emotions, &npc.needs);

        // --- Belief Decay ---
        decay_beliefs(&mut npc.memory, tick);

        // --- Personality Drift from Recent Events ---
        drift_personality_from_memory(&mut npc.personality, &npc.memory);

        // --- Catastrophe tags from settlement state ---
        if let Some(sid) = npc.home_settlement_id {
            let threat = if (sid as usize) < threat_by_sid.len() { threat_by_sid[sid as usize] } else { 0.0 };
            let pop = if (sid as usize) < pop_by_sid.len() { pop_by_sid[sid as usize] } else { 0 };

            // Under siege: high threat for extended period.
            if threat > 0.5 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::DEFENSE, 1.0);
                    a.add(tags::RESILIENCE, 0.5);
                    a.add(tags::ENDURANCE, 0.5);
                    a
                });
            }

            // Near-extinction: settlement below 20 NPCs (catastrophic loss).
            if pop < 20 && pop > 0 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::SURVIVAL, 2.0);
                    a.add(tags::RESILIENCE, 2.0);
                    a.add(tags::ENDURANCE, 1.0);
                    a.add(tags::LEADERSHIP, 0.5); // forced into responsibility
                    a
                });
            }

            // Last stand: fewer than 5 NPCs alive. Every tick they survive is legendary.
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

            // Famine: settlement food below 10.
            let food = state.settlements.iter()
                .find(|s| s.id == sid)
                .map(|s| s.stockpile[crate::world_sim::commodity::FOOD])
                .unwrap_or(100.0);
            if food < 10.0 {
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::SURVIVAL, 1.5);
                    a.add(tags::ENDURANCE, 1.0);
                    a
                });
            }
        }

        // --- Emotional state tags (extreme emotions shape identity) ---

        // Sustained rage builds combat instincts.
        if npc.emotions.anger > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMBAT, 1.0);
                a.add(tags::MELEE, 0.5);
                a
            });
        }

        // Sustained fear builds awareness and defense.
        if npc.emotions.fear > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::AWARENESS, 1.0);
                a.add(tags::DEFENSE, 0.5);
                a.add(tags::STEALTH, 0.3);
                a
            });
        }

        // Sustained grief builds compassion and resilience.
        if npc.emotions.grief > 0.5 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMPASSION_TAG, 1.0);
                a.add(tags::RESILIENCE, 0.5);
                a
            });
        }

        // Sustained pride builds leadership.
        if npc.emotions.pride > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::LEADERSHIP, 0.5);
                a.add(tags::DISCIPLINE, 0.3);
                a
            });
        }

        // Sustained anxiety builds awareness.
        if npc.emotions.anxiety > 0.7 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::AWARENESS, 0.5);
                a.add(tags::SURVIVAL, 0.3);
                a
            });
        }

        // --- Social: NPCs with work assignments get social need from coworkers ---
        if npc.work_building_id.is_some() && !matches!(npc.work_state, WorkState::Idle) {
            // Working NPCs interact with coworkers.
            npc.needs.social = (npc.needs.social + 0.3).min(75.0);

            // Occasionally form friendship with a coworker.
            if entity_hash(entity.id, tick, 0xF81E) % 200 == 0 { // ~0.5% chance per interval
                let recent_friend = npc.memory.events.iter().rev().take(3).any(|e| {
                    matches!(e.event_type, MemEventType::MadeNewFriend(_))
                        && tick.saturating_sub(e.tick) < 1000
                });
                if !recent_friend {
                    // Use a pseudo-coworker ID based on building.
                    let friend_id = npc.work_building_id.unwrap_or(0).wrapping_mul(31).wrapping_add(entity.id);
                    record_npc_event(npc, MemEventType::MadeNewFriend(friend_id),
                        entity.pos, vec![friend_id], 0.5, tick);
                }
            }
        }

        // --- Detect combat: NPCs with low HP who are on a combat grid ---
        if entity.hp < entity.max_hp * 0.5 && entity.grid_id.is_some() {
            // Check if we already recorded WasAttacked recently.
            let recent_attack = npc.memory.events.iter().rev().take(3).any(|e| {
                matches!(e.event_type, MemEventType::WasAttacked)
                    && tick.saturating_sub(e.tick) < 200
            });
            if !recent_attack {
                record_npc_event(npc, MemEventType::WasAttacked, entity.pos, vec![], -0.5, tick);
            }
        }

        // --- Detect combat victory: NPC with high HP on grid with no hostiles ---
        // (Approximation: if NPC was recently attacked but now has high HP, they survived)
        if entity.hp > entity.max_hp * 0.8 {
            let was_attacked_recently = npc.memory.events.iter().rev().take(5).any(|e| {
                matches!(e.event_type, MemEventType::WasAttacked)
                    && tick.saturating_sub(e.tick) < 300
            });
            let already_won = npc.memory.events.iter().rev().take(3).any(|e| {
                matches!(e.event_type, MemEventType::WonFight)
                    && tick.saturating_sub(e.tick) < 300
            });
            if was_attacked_recently && !already_won {
                record_npc_event(npc, MemEventType::WonFight, entity.pos, vec![], 0.6, tick);
            }
        }

        // --- Social: class level-up generates LearnedSkill ---
        if entity.level > 1 && entity.level % 10 == 0 {
            let already_learned = npc.memory.events.iter().rev().take(3).any(|e| {
                matches!(e.event_type, MemEventType::LearnedSkill)
                    && tick.saturating_sub(e.tick) < 500
            });
            if !already_learned {
                record_npc_event(npc, MemEventType::LearnedSkill, entity.pos, vec![], 0.4, tick);
            }
        }

        // --- Detect trade: NPC on Trade intent with carried goods → trading ---
        if matches!(npc.economic_intent, EconomicIntent::Trade { .. }) {
            let has_goods = entity.inventory.as_ref()
                .map(|inv| inv.commodities.iter().any(|&g| g > 0.1))
                .unwrap_or(false);
            if has_goods {
                let recent_trade = npc.memory.events.iter().rev().take(3).any(|e| {
                    matches!(e.event_type, MemEventType::TradedWith(_))
                        && tick.saturating_sub(e.tick) < 500
                });
                if !recent_trade {
                    // Record trade with the destination settlement (use settlement_id as proxy).
                    if let EconomicIntent::Trade { destination_settlement_id } = &npc.economic_intent {
                        record_npc_event(npc, MemEventType::TradedWith(*destination_settlement_id),
                            entity.pos, vec![], 0.3, tick);
                        // Trading satisfies purpose need.
                        npc.needs.purpose = (npc.needs.purpose + 5.0).min(80.0);
                    }
                }
            }
        }

        // --- Detect building construction: NPCs working at unfinished buildings ---
        if let Some(_wbid) = npc.work_building_id {
            // This is a proxy — actual BuiltSomething events are recorded in buildings.rs
            // when construction completes. Here we just give purpose satisfaction for building work.
            if npc.behavior_value(tags::CONSTRUCTION) > 10.0 {
                npc.needs.purpose = (npc.needs.purpose + 0.2).min(70.0);
            }
        }
    }

    // --- Emotional Contagion (every 20 ticks to limit cost) ---
    if state.tick % 20 == 0 {
        spread_emotions(state);
    }
}

/// Spread extreme emotions between nearby NPCs.
/// Fear spreads fastest (30%), joy second (25%), anger third (20%), grief slowest (15%).
/// NPCs with high resilience resist contagion.
fn spread_emotions(state: &mut WorldState) {
    const CONTAGION_DIST_SQ: f32 = 100.0; // 10 units

    // Phase 1: snapshot emotions + positions of NPCs with extreme emotions only.
    // This avoids the full O(n) snapshot — only NPCs who can spread matter.
    let snapshots: Vec<(u32, (f32, f32), f32, f32, f32, f32)> = state.entities.iter()
        .filter(|e| e.alive && e.kind == EntityKind::Npc && e.npc.is_some())
        .filter_map(|e| {
            let em = &e.npc.as_ref().unwrap().emotions;
            // Only include NPCs with at least one extreme emotion.
            if em.fear > 0.7 || em.joy > 0.7 || em.anger > 0.7 || em.grief > 0.5 {
                Some((e.id, e.pos, em.fear, em.joy, em.anger, em.grief))
            } else {
                None
            }
        })
        .take(50) // cap at 50 emotional sources
        .collect();

    // Phase 2: for each NPC, absorb emotions from nearby NPCs with extreme emotions.
    // Stagger: only process 1/4 of NPCs per call (by entity ID mod 4).
    let slot = (state.tick / 20) % 4;
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        if entity.id as u64 % 4 != slot { continue; } // stagger
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // Resilience reduces contagion susceptibility.
        let resilience = npc.behavior_value(tags::RESILIENCE);
        let resist = 1.0 / (1.0 + resilience * 0.01); // high resilience → low susceptibility

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
            if neighbors > 5 { break; } // cap neighbor scan

            // Only absorb extreme emotions (> 0.7).
            if sfear > 0.7 { fear_absorbed += sfear * 0.30; }   // fear spreads fastest
            if sjoy > 0.7 { joy_absorbed += sjoy * 0.25; }     // joy second
            if sanger > 0.7 { anger_absorbed += sanger * 0.20; } // anger third
            if sgrief > 0.5 { grief_absorbed += sgrief * 0.15; } // grief slowest
        }

        if neighbors == 0 { continue; }

        // Average over neighbors and apply resistance.
        let n = neighbors as f32;
        npc.emotions.fear = (npc.emotions.fear + fear_absorbed / n * resist).min(1.0);
        npc.emotions.joy = (npc.emotions.joy + joy_absorbed / n * resist).min(1.0);
        npc.emotions.anger = (npc.emotions.anger + anger_absorbed / n * resist).min(1.0);
        npc.emotions.grief = (npc.emotions.grief + grief_absorbed / n * resist).min(1.0);
    }
}

/// Record a memory event on an NPC. Called by other systems when
/// significant things happen.
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

    // Spike emotions based on event impact.
    if emotional_impact > 0.3 {
        npc.emotions.joy = (npc.emotions.joy + emotional_impact * 0.5).min(1.0);
        npc.emotions.pride = (npc.emotions.pride + emotional_impact * 0.3).min(1.0);
    }
    if emotional_impact < -0.3 {
        let neg = -emotional_impact;
        npc.emotions.fear = (npc.emotions.fear + neg * 0.4).min(1.0);
        npc.emotions.anxiety = (npc.emotions.anxiety + neg * 0.3).min(1.0);
    }

    // Specific emotion spikes by event type.
    match &npc.memory.events.back().map(|e| &e.event_type) {
        Some(MemEventType::FriendDied(_)) => {
            npc.emotions.grief = (npc.emotions.grief + 0.5).min(1.0);
            npc.emotions.anger = (npc.emotions.anger + 0.3).min(1.0);
            npc.needs.social -= 10.0;
            npc.needs.social = npc.needs.social.max(0.0);
            npc.needs.safety -= 15.0;
            npc.needs.safety = npc.needs.safety.max(0.0);
            npc.personality.risk_tolerance = (npc.personality.risk_tolerance - 0.01).max(0.1);
            npc.personality.compassion = (npc.personality.compassion + 0.01).min(0.9);
            // Witnessing death hardens you — builds resilience, endurance, defense.
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::RESILIENCE, 3.0);   // grief forges resilience
                a.add(tags::ENDURANCE, 2.0);     // surviving loss builds endurance
                a.add(tags::DEFENSE, 1.5);       // instinct to protect others
                a.add(tags::COMPASSION_TAG, 1.0); // empathy from shared suffering
                a
            });
        }
        Some(MemEventType::WonFight) => {
            npc.emotions.pride = (npc.emotions.pride + 0.5).min(1.0);
            npc.needs.esteem += 10.0;
            npc.needs.esteem = npc.needs.esteem.min(100.0);
            npc.personality.risk_tolerance = (npc.personality.risk_tolerance + 0.02).min(1.0);
            // Victory builds combat experience.
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMBAT, 5.0);    // direct combat skill
                a.add(tags::TACTICS, 3.0);   // tactical awareness
                a.add(tags::MELEE, 2.0);     // fighting skill
                a
            });
        }
        Some(MemEventType::WasAttacked) => {
            npc.emotions.anger = (npc.emotions.anger + 0.6).min(1.0);
            npc.emotions.fear = (npc.emotions.fear + 0.4).min(1.0);
            npc.needs.safety -= 30.0;
            npc.needs.safety = npc.needs.safety.max(0.0);
            // Being attacked builds combat awareness and survival instincts.
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMBAT, 3.0);
                a.add(tags::DEFENSE, 4.0);     // defensive instincts spike
                a.add(tags::AWARENESS, 2.0);   // heightened vigilance
                a.add(tags::SURVIVAL, 2.0);
                a
            });
        }
        Some(MemEventType::CompletedQuest) => {
            npc.emotions.joy = (npc.emotions.joy + 0.7).min(1.0);
            npc.emotions.pride = (npc.emotions.pride + 0.5).min(1.0);
            npc.needs.purpose += 20.0;
            npc.needs.purpose = npc.needs.purpose.min(100.0);
            npc.personality.ambition = (npc.personality.ambition + 0.03).min(1.0);
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::COMBAT, 3.0);
                a.add(tags::TACTICS, 2.0);
                a.add(tags::EXPLORATION, 2.0);
                a.add(tags::LEADERSHIP, 1.0);
                a
            });
        }
        Some(MemEventType::Starved) => {
            npc.emotions.anxiety = (npc.emotions.anxiety + 0.8).min(1.0);
            npc.emotions.anger = (npc.emotions.anger + 0.4).min(1.0);
            npc.personality.risk_tolerance = (npc.personality.risk_tolerance + 0.02).min(1.0);
            // Starvation builds survival instincts.
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::SURVIVAL, 4.0);    // learning to endure
                a.add(tags::ENDURANCE, 3.0);
                a.add(tags::RESILIENCE, 2.0);
                a
            });
        }
        Some(MemEventType::MadeNewFriend(_)) => {
            npc.emotions.joy = (npc.emotions.joy + 0.4).min(1.0);
            npc.needs.social += 15.0;
            npc.needs.social = npc.needs.social.min(100.0);
            npc.personality.social_drive = (npc.personality.social_drive + 0.02).min(1.0);
        }
        Some(MemEventType::LearnedSkill) => {
            npc.emotions.pride = (npc.emotions.pride + 0.3).min(1.0);
            npc.needs.esteem += 5.0;
            npc.needs.esteem = npc.needs.esteem.min(100.0);
            npc.personality.curiosity = (npc.personality.curiosity + 0.02).min(1.0);
        }
        Some(MemEventType::BuiltSomething) => {
            npc.emotions.pride = (npc.emotions.pride + 0.4).min(1.0);
            npc.needs.purpose += 10.0;
            npc.needs.purpose = npc.needs.purpose.min(100.0);
            npc.personality.ambition = (npc.personality.ambition + 0.02).min(1.0);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Drift needs based on world state.
fn drift_needs(npc: &mut NpcData, hp: f32, max_hp: f32, threat_by_sid: &[f32], tick: u64) {
    // Hunger decreases steadily (0=starving, 100=full).
    // Only replenished by physically eating (handled in advance_eating).
    let at_settlement = npc.home_settlement_id.is_some();
    let hunger_drain = if at_settlement { 0.15 } else { 0.3 };
    npc.needs.hunger = (npc.needs.hunger - hunger_drain).max(0.0);

    // Record starvation event if critically hungry.
    if npc.needs.hunger < 5.0 {
        let recent_starve = npc.memory.events.iter().rev().take(3).any(|e| {
            matches!(e.event_type, MemEventType::Starved) && tick.saturating_sub(e.tick) < 500
        });
        if !recent_starve {
            record_npc_event(npc, MemEventType::Starved, (0.0, 0.0), vec![], -0.6, tick);
        }
    }

    // Safety based on settlement threat level.
    let threat = npc.home_settlement_id
        .map(|sid| if (sid as usize) < threat_by_sid.len() { threat_by_sid[sid as usize] } else { 0.0 })
        .unwrap_or(0.0);
    if threat > 0.3 {
        npc.needs.safety = (npc.needs.safety - threat * 5.0).max(0.0);
    } else {
        // Safety recovers when threat is low.
        npc.needs.safety = (npc.needs.safety + 1.0).min(100.0);
    }

    // Low HP reduces safety need directly.
    if hp < max_hp * 0.5 {
        npc.needs.safety = (npc.needs.safety - 5.0).max(0.0);
    }

    // Shelter: having a home building satisfies this.
    if npc.home_building_id.is_some() {
        npc.needs.shelter = (npc.needs.shelter + 0.5).min(100.0);
    } else {
        npc.needs.shelter = (npc.needs.shelter - 1.0).max(0.0);
    }

    // Social: decays slowly, refreshed by being at a settlement (proxy for social contact).
    if at_settlement {
        npc.needs.social = (npc.needs.social + 0.1).min(70.0);
    } else {
        npc.needs.social = (npc.needs.social - 0.2).max(0.0);
    }

    // Purpose: having work satisfies this.
    if npc.work_building_id.is_some() || matches!(npc.economic_intent, EconomicIntent::Adventuring { .. }) {
        npc.needs.purpose = (npc.needs.purpose + 0.3).min(100.0);
    } else {
        npc.needs.purpose = (npc.needs.purpose - 0.5).max(0.0);
    }

    // Esteem: decays slowly, refreshed by achievements.
    npc.needs.esteem = (npc.needs.esteem - 0.1).max(0.0);
}

/// Spike emotions based on unmet needs.
fn spike_emotions_from_needs(emotions: &mut Emotions, needs: &Needs) {
    // Multiple unmet needs → anxiety.
    let unmet_count = [needs.hunger, needs.safety, needs.shelter, needs.social, needs.purpose]
        .iter().filter(|&&v| v < 30.0).count();
    if unmet_count >= 2 {
        emotions.anxiety = (emotions.anxiety + 0.1 * unmet_count as f32).min(1.0);
    }

    // Very hungry → anxiety.
    if needs.hunger < 15.0 {
        emotions.anxiety = (emotions.anxiety + 0.2).min(1.0);
    }

    // Very unsafe → fear + anger.
    if needs.safety < 20.0 {
        emotions.fear = (emotions.fear + 0.15).min(1.0);
        emotions.anger = (emotions.anger + 0.1).min(1.0); // threatened → angry
    }

    // Low safety but not desperate → frustration/anger.
    if needs.safety < 40.0 && needs.safety >= 20.0 {
        emotions.anger = (emotions.anger + 0.05).min(1.0);
    }

    // Starving → anger (injustice/desperation).
    if needs.hunger < 10.0 {
        emotions.anger = (emotions.anger + 0.15).min(1.0);
    }

    // No purpose → frustration.
    if needs.purpose < 15.0 {
        emotions.anger = (emotions.anger + 0.05).min(1.0);
    }

    // Low esteem with high purpose → frustrated ambition.
    if needs.esteem < 20.0 && needs.purpose > 50.0 {
        emotions.anger = (emotions.anger + 0.05).min(1.0);
    }

    // All needs satisfied → joy.
    let all_ok = [needs.hunger, needs.safety, needs.shelter, needs.social, needs.purpose, needs.esteem]
        .iter().all(|&v| v > 60.0);
    if all_ok {
        emotions.joy = (emotions.joy + 0.1).min(1.0);
    }

    // Safety + social satisfied → pride (community strength).
    if needs.safety > 70.0 && needs.social > 50.0 && needs.esteem > 40.0 {
        emotions.pride = (emotions.pride + 0.05).min(1.0);
    }
}

/// Decay belief confidence over time. Remove beliefs with confidence < 0.05.
fn decay_beliefs(memory: &mut Memory, tick: u64) {
    for belief in &mut memory.beliefs {
        let age = tick.saturating_sub(belief.formed_tick);
        if age > 100 {
            // Grudges decay much slower — vengeance is patient.
            let rate = match &belief.belief_type {
                BeliefType::Grudge(_) => 0.001,
                _ => 0.005,
            };
            belief.confidence -= rate;
        }
    }
    memory.beliefs.retain(|b| b.confidence > 0.05);
}

/// Personality drifts based on recent memory events.
fn drift_personality_from_memory(personality: &mut Personality, memory: &Memory) {
    // Count recent positive vs negative events (last 10).
    let recent = memory.events.iter().rev().take(10);
    let mut positive = 0;
    let mut negative = 0;
    for event in recent {
        if event.emotional_impact > 0.2 { positive += 1; }
        if event.emotional_impact < -0.2 { negative += 1; }
    }

    // More positive experiences → slightly more risk tolerant, ambitious.
    if positive > negative + 2 {
        personality.risk_tolerance = (personality.risk_tolerance + 0.005).min(1.0);
        personality.ambition = (personality.ambition + 0.005).min(1.0);
    }
    // More negative experiences → more cautious, more compassionate.
    if negative > positive + 2 {
        personality.risk_tolerance = (personality.risk_tolerance - 0.005).max(0.0);
        personality.compassion = (personality.compassion + 0.005).min(1.0);
    }
}
