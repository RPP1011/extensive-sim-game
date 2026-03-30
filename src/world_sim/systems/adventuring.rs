#![allow(unused)]
//! Adventuring parties — NPCs form groups, take quests, and explore.
//!
//! NPCs with high risk_tolerance, combat tags, and purpose/esteem needs form
//! adventuring parties (2-4 members). Parties take available quests, travel
//! to the quest destination, and fight/explore. On completion, they return
//! home with gold/XP/items.
//!
//! Party formation requires: risk_tolerance > 0.4, combat or survival tags > 50,
//! no current work assignment (or Idle economic intent).
//!
//! Cadence: every 100 ticks (party formation), every 10 ticks (party movement).

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const FORMATION_INTERVAL: u64 = 100;
const MOVEMENT_INTERVAL: u64 = 1;
const MIN_PARTY_SIZE: usize = 2;
const MAX_PARTY_SIZE: usize = 4;
const QUEST_ARRIVAL_DIST_SQ: f32 = 900.0; // 30 units

/// Form adventuring parties and manage quest assignments.
/// Called post-apply from runtime.rs.
pub fn advance_adventuring(state: &mut WorldState) {
    let tick = state.tick;

    // --- Party formation (every 100 ticks) ---
    if tick % FORMATION_INTERVAL == 0 && tick > 0 {
        form_parties(state, tick);
    }

    // --- Party quest assignment + movement (every 10 ticks) ---
    if tick % MOVEMENT_INTERVAL == 0 && tick > 0 {
        advance_party_quests(state, tick);
    }
}

/// Form new parties from eligible unpartied NPCs at each settlement.
fn form_parties(state: &mut WorldState, tick: u64) {
    let mut next_party_id = state.entities.iter()
        .filter_map(|e| e.npc.as_ref().and_then(|n| n.party_id))
        .max()
        .unwrap_or(0) + 1;

    for si in 0..state.settlements.len() {
        let sid = state.settlements[si].id;

        // Collect eligible NPCs at this settlement.
        let mut candidates: Vec<usize> = Vec::new();
        for (i, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(sid) { continue; }
            if npc.party_id.is_some() { continue; } // already in a party

            // Must be adventurer-capable: some combat/survival exposure + unmet purpose or esteem.
            let combat = npc.behavior_value(tags::COMBAT) + npc.behavior_value(tags::MELEE)
                + npc.behavior_value(tags::SURVIVAL);
            let not_busy = !matches!(npc.economic_intent,
                EconomicIntent::Trade { .. } | EconomicIntent::Adventuring { .. });
            let wants_adventure = npc.needs.purpose < 60.0 || npc.needs.esteem < 50.0;

            if combat > 10.0 && not_busy && wants_adventure {
                candidates.push(i);
            }
        }

        if candidates.len() < MIN_PARTY_SIZE { continue; }

        // Form parties of 2-4 from candidates (greedy).
        let mut formed = 0usize;
        let mut ci = 0;
        while ci + MIN_PARTY_SIZE <= candidates.len() && formed < 2 { // max 2 parties per settlement per cycle
            let party_size = (candidates.len() - ci).min(MAX_PARTY_SIZE);
            let pid = next_party_id;
            next_party_id += 1;

            for j in 0..party_size {
                let entity_idx = candidates[ci + j];
                if let Some(npc) = state.entities[entity_idx].npc.as_mut() {
                    npc.party_id = Some(pid);
                    // Push Quest goal.
                    npc.goal_stack.push(
                        Goal::new(GoalKind::Quest { quest_id: 0, destination: (0.0, 0.0) },
                            goal_priority::QUEST, tick)
                    );
                    npc.economic_intent = EconomicIntent::Adventuring {
                        quest_id: 0,
                        destination: (0.0, 0.0),
                    };
                    // Boost purpose and esteem from joining a party.
                    npc.needs.purpose = (npc.needs.purpose + 20.0).min(100.0);
                    npc.needs.social = (npc.needs.social + 15.0).min(100.0);
                    // Combat tags from party training.
                    npc.accumulate_tags(&{
                        let mut a = ActionTags::empty();
                        a.add(tags::COMBAT, 2.0);
                        a.add(tags::TACTICS, 1.0);
                        a.add(tags::EXPLORATION, 1.0);
                        a
                    });
                }
            }

            ci += party_size;
            formed += 1;
        }
    }
}

/// Assign quests to parties and move them toward destinations.
fn advance_party_quests(state: &mut WorldState, tick: u64) {
    // Collect active parties: (party_id, [entity_indices], settlement_id).
    let mut parties: Vec<(u32, Vec<usize>, Option<u32>)> = Vec::new();

    for (i, entity) in state.entities.iter().enumerate() {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        let pid = match npc.party_id { Some(id) => id, None => continue };

        if let Some(party) = parties.iter_mut().find(|(id, _, _)| *id == pid) {
            party.1.push(i);
        } else {
            parties.push((pid, vec![i], npc.home_settlement_id));
        }
    }

    for (pid, members, home_sid) in &parties {
        if members.is_empty() { continue; }

        // Get the leader (first member) and their quest destination.
        let leader_idx = members[0];
        let (leader_pos, quest_dest, quest_id) = {
            let entity = &state.entities[leader_idx];
            let npc = entity.npc.as_ref().unwrap();
            let dest = match &npc.economic_intent {
                EconomicIntent::Adventuring { destination, quest_id } => {
                    if *destination == (0.0, 0.0) { None } else { Some((*destination, *quest_id)) }
                }
                _ => None,
            };
            (entity.pos, dest.map(|(d, _)| d), dest.map(|(_, q)| q).unwrap_or(0))
        };

        // If no destination yet, pick one — grudge target or explore wilderness.
        if quest_dest.is_none() {
            let home_pos = home_sid
                .and_then(|sid| state.settlements.iter().find(|s| s.id == sid))
                .map(|s| s.pos)
                .unwrap_or(leader_pos);

            // Check if any party member has a grudge — prioritize revenge.
            let mut grudge_target_pos: Option<(f32, f32)> = None;
            for &mi in members {
                let npc = match &state.entities[mi].npc { Some(n) => n, None => continue };
                for belief in &npc.memory.beliefs {
                    if let BeliefType::Grudge(target_id) = &belief.belief_type {
                        // Find the grudge target's position.
                        if let Some(target) = state.entities.iter().find(|e| e.id == *target_id && e.alive) {
                            grudge_target_pos = Some(target.pos);
                            break;
                        }
                    }
                }
                if grudge_target_pos.is_some() { break; }
            }

            // Check for nearby dungeon sites as targets.
            let dungeon_target: Option<(f32, f32)> = state.regions.iter()
                .flat_map(|r| r.dungeon_sites.iter())
                .filter(|d| !d.is_cleared)
                .min_by_key(|d| {
                    let dx = d.pos.0 - home_pos.0;
                    let dy = d.pos.1 - home_pos.1;
                    (dx * dx + dy * dy) as u32
                })
                .map(|d| d.pos);

            let dest = if let Some(grudge_pos) = grudge_target_pos {
                grudge_pos
            } else if let Some(dungeon_pos) = dungeon_target {
                // Dungeon delving — head to nearest uncleared dungeon.
                dungeon_pos
            } else {
                // Random exploration.
                let h = entity_hash(state.entities[leader_idx].id, tick, 0xAD01);
                let angle = (h % 360) as f32 * std::f32::consts::PI / 180.0;
                let dist = 30.0 + (h % 40) as f32;
                (home_pos.0 + angle.cos() * dist, home_pos.1 + angle.sin() * dist)
            };

            for &mi in members {
                if let Some(npc) = state.entities[mi].npc.as_mut() {
                    npc.economic_intent = EconomicIntent::Adventuring {
                        quest_id: 0,
                        destination: dest,
                    };
                    if let Some(goal) = npc.goal_stack.current_mut() {
                        goal.target_pos = Some(dest);
                    }
                }
            }
            continue;
        }

        let dest = quest_dest.unwrap();

        // Check if party arrived at destination.
        let dx = dest.0 - leader_pos.0;
        let dy = dest.1 - leader_pos.1;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq < QUEST_ARRIVAL_DIST_SQ {
            // Check if we arrived at a dungeon site.
            let dungeon_info: Option<(String, u8, f32)> = state.regions.iter()
                .flat_map(|r| r.dungeon_sites.iter())
                .find(|d| {
                    let ddx = d.pos.0 - dest.0;
                    let ddy = d.pos.1 - dest.1;
                    ddx * ddx + ddy * ddy < 400.0 // within 20 units
                })
                .map(|d| (d.name.clone(), d.max_depth, d.threat_mult));

            // Mark dungeon as explored.
            if let Some((ref dname, _, _)) = dungeon_info {
                for region in &mut state.regions {
                    for dungeon in &mut region.dungeon_sites {
                        let ddx = dungeon.pos.0 - dest.0;
                        let ddy = dungeon.pos.1 - dest.1;
                        if ddx * ddx + ddy * ddy < 400.0 {
                            dungeon.explored_depth = (dungeon.explored_depth + 1).min(dungeon.max_depth);
                            dungeon.last_explored_tick = tick;
                            if dungeon.explored_depth >= dungeon.max_depth {
                                dungeon.is_cleared = true;
                            }
                        }
                    }
                }
            }

            let is_dungeon = dungeon_info.is_some();
            let dungeon_depth = dungeon_info.as_ref().map(|(_, d, _)| *d).unwrap_or(0);
            let dungeon_threat = dungeon_info.as_ref().map(|(_, _, t)| *t).unwrap_or(1.0);
            let dungeon_name = dungeon_info.as_ref().map(|(n, _, _)| n.clone());

            // Adventure complete! Reward and disband.
            for &mi in members {
                let entity = &mut state.entities[mi];
                let npc = entity.npc.as_mut().unwrap();

                // Gold reward — dungeons pay more based on depth.
                let base_gold = 10.0 + entity.level as f32 * 2.0;
                let dungeon_bonus = if is_dungeon { dungeon_depth as f32 * 5.0 } else { 0.0 };
                npc.gold += base_gold + dungeon_bonus;

                // Experience tags — dungeoneering for dungeons.
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::COMBAT, 5.0);
                    a.add(tags::EXPLORATION, 3.0);
                    a.add(tags::SURVIVAL, 2.0);
                    a.add(tags::TACTICS, 2.0);
                    a
                });

                // Dungeon-specific tags.
                if is_dungeon {
                    npc.accumulate_tags(&{
                        let mut a = ActionTags::empty();
                        a.add(tags::DUNGEONEERING, 5.0 + dungeon_depth as f32);
                        a.add(tags::AWARENESS, 2.0);
                        a.add(tags::STEALTH, 1.0);
                        a
                    });
                }

                // Satisfy needs.
                npc.needs.purpose = (npc.needs.purpose + 30.0).min(100.0);
                npc.needs.esteem = (npc.needs.esteem + 20.0).min(100.0);

                // Record memory.
                crate::world_sim::systems::agent_inner::record_npc_event(
                    npc, MemEventType::CompletedQuest,
                    entity.pos, vec![], 0.7, tick,
                );

                // Disband: clear party and return to Produce.
                npc.party_id = None;
                npc.economic_intent = EconomicIntent::Produce;
                npc.goal_stack.remove_kind(&GoalKind::Quest { quest_id: 0, destination: (0.0, 0.0) });
                // Remove any quest goals.
                npc.goal_stack.goals.retain(|g| !matches!(g.kind, GoalKind::Quest { .. }));
            }

            // Chronicle: dungeon exploration.
            if let Some(dname) = &dungeon_name {
                let party_names: Vec<String> = members.iter()
                    .filter_map(|&mi| state.entities.get(mi))
                    .filter_map(|e| e.npc.as_ref())
                    .map(|n| n.name.clone())
                    .take(3)
                    .collect();
                let names_str = party_names.join(", ");
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Achievement,
                    text: format!("A party ({}) explored {} to depth {}.",
                        names_str, dname, dungeon_depth),
                    entity_ids: members.iter().map(|&mi| state.entities[mi].id).collect(),
                });
            }

            // Relic discovery: 20% chance per dungeon exploration.
            if is_dungeon {
                let leader_id = state.entities[members[0]].id;
                let leader_name = state.entities[members[0]].npc.as_ref()
                    .map(|n| n.name.clone()).unwrap_or_default();
                let leader_pos = state.entities[members[0]].pos;
                let leader_sid = state.entities[members[0]].npc.as_ref()
                    .and_then(|n| n.home_settlement_id);

                let relic_roll = entity_hash_f32(leader_id, tick, 0xDE1C);
                if relic_roll < 0.20 {
                    let relic_id = state.next_entity_id();
                    let relic_tag = match entity_hash(leader_id, tick, 0xDE1D) % 6 {
                        0 => (tags::FAITH, "Sacred"),
                        1 => (tags::RESEARCH, "Arcane"),
                        2 => (tags::COMBAT, "War"),
                        3 => (tags::FARMING, "Fertility"),
                        4 => (tags::CRAFTING, "Artisan's"),
                        _ => (tags::LEADERSHIP, "Crown"),
                    };
                    let relic_name = format!("{} Relic of {}", relic_tag.1,
                        dungeon_name.as_deref().unwrap_or("the Deep"));
                    let bonus_value = 5.0 + dungeon_depth as f32 * 2.0;

                    let relic = Entity::new_item(relic_id, leader_pos, ItemData {
                        slot: ItemSlot::Accessory,
                        rarity: ItemRarity::Legendary,
                        quality: 15.0,
                        durability: 100.0,
                        max_durability: 100.0,
                        owner_id: Some(leader_id),
                        settlement_id: leader_sid,
                        name: relic_name.clone(),
                        crafter_id: None,
                        crafted_tick: tick,
                        history: vec![ItemEvent {
                            tick,
                            kind: ItemEventKind::Crafted { crafter_name: "the ancients".into() },
                        }],
                        is_legendary: true,
                        is_relic: true,
                        relic_bonus: Some((relic_tag.0, bonus_value)),
                    });
                    state.entities.push(relic);

                    state.chronicle.push(ChronicleEntry {
                        tick,
                        category: ChronicleCategory::Achievement,
                        text: format!("{} discovered the {} in {}!",
                            leader_name, relic_name,
                            dungeon_name.as_deref().unwrap_or("a dungeon")),
                        entity_ids: vec![leader_id, relic_id],
                    });
                }
            }
        } else {
            // Move party toward destination.
            let dist = dist_sq.sqrt();
            for &mi in members {
                let entity = &mut state.entities[mi];
                let speed = entity.move_speed * crate::world_sim::DT_SEC;
                entity.pos.0 += dx / dist * speed;
                entity.pos.1 += dy / dist * speed;
            }
        }
    }

    // --- Rival party encounters ---
    // When two parties from different factions are within 25 units, they fight.
    if tick % 50 == 0 && parties.len() >= 2 {
        let mut clashes: Vec<(usize, usize)> = Vec::new();

        for i in 0..parties.len() {
            for j in (i + 1)..parties.len() {
                if clashes.len() >= 2 { break; }
                let (pid_a, members_a, sid_a) = &parties[i];
                let (pid_b, members_b, sid_b) = &parties[j];
                if members_a.is_empty() || members_b.is_empty() { continue; }

                // Different factions?
                let faction_a = state.entities[members_a[0]].npc.as_ref()
                    .and_then(|n| n.faction_id);
                let faction_b = state.entities[members_b[0]].npc.as_ref()
                    .and_then(|n| n.faction_id);
                if faction_a == faction_b { continue; } // same faction, no conflict

                // Close enough to fight?
                let pos_a = state.entities[members_a[0]].pos;
                let pos_b = state.entities[members_b[0]].pos;
                let dx = pos_a.0 - pos_b.0;
                let dy = pos_a.1 - pos_b.1;
                if dx * dx + dy * dy < 625.0 { // 25 units
                    clashes.push((i, j));
                }
            }
        }

        for (pi_a, pi_b) in clashes {
            let members_a = &parties[pi_a].1;
            let members_b = &parties[pi_b].1;

            // Calculate total power (HP × attack) for each party.
            let power_a: f32 = members_a.iter()
                .map(|&mi| state.entities[mi].hp * state.entities[mi].attack_damage)
                .sum();
            let power_b: f32 = members_b.iter()
                .map(|&mi| state.entities[mi].hp * state.entities[mi].attack_damage)
                .sum();

            let (winners, losers) = if power_a >= power_b {
                (members_a.clone(), members_b.clone())
            } else {
                (members_b.clone(), members_a.clone())
            };

            // Pre-collect IDs and positions to avoid borrow conflicts.
            let winner_ids: Vec<u32> = winners.iter().map(|&wi| state.entities[wi].id).collect();
            let loser_ids: Vec<u32> = losers.iter().map(|&li| state.entities[li].id).collect();

            // Losers take damage and form grudges against winners.
            for &li in &losers {
                let loser_pos = state.entities[li].pos;
                state.entities[li].hp -= state.entities[li].max_hp * 0.3;
                if let Some(npc) = state.entities[li].npc.as_mut() {
                    npc.emotions.anger = (npc.emotions.anger + 0.6).min(1.0);
                    npc.emotions.fear = (npc.emotions.fear + 0.4).min(1.0);
                    for &wid in &winner_ids {
                        npc.memory.beliefs.push(Belief {
                            belief_type: BeliefType::Grudge(wid),
                            confidence: 1.0,
                            formed_tick: tick,
                        });
                    }
                    crate::world_sim::systems::agent_inner::record_npc_event(
                        npc, MemEventType::WasAttacked,
                        loser_pos, vec![], -0.6, tick,
                    );
                }
            }

            // Winners gain combat tags and loot gold from losers.
            let loser_gold: f32 = losers.iter()
                .filter_map(|&li| state.entities[li].npc.as_ref())
                .map(|n| (n.gold * 0.3).min(10.0))
                .sum();

            for &li in &losers {
                if let Some(npc) = state.entities[li].npc.as_mut() {
                    let loss = (npc.gold * 0.3).min(10.0);
                    npc.gold -= loss;
                }
            }

            let gold_per_winner = loser_gold / winners.len().max(1) as f32;
            for &wi in &winners {
                let winner_pos = state.entities[wi].pos;
                if let Some(npc) = state.entities[wi].npc.as_mut() {
                    npc.gold += gold_per_winner;
                    npc.emotions.pride = (npc.emotions.pride + 0.5).min(1.0);
                    npc.accumulate_tags(&{
                        let mut a = ActionTags::empty();
                        a.add(tags::COMBAT, 3.0);
                        a.add(tags::TACTICS, 2.0);
                        a
                    });
                    crate::world_sim::systems::agent_inner::record_npc_event(
                        npc, MemEventType::WonFight,
                        winner_pos, vec![], 0.6, tick,
                    );
                }
            }

            // Chronicle.
            let winner_names: Vec<String> = winners.iter().take(2)
                .filter_map(|&wi| state.entities[wi].npc.as_ref())
                .map(|n| n.name.clone())
                .collect();
            let loser_names: Vec<String> = losers.iter().take(2)
                .filter_map(|&li| state.entities[li].npc.as_ref())
                .map(|n| n.name.clone())
                .collect();
            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Narrative,
                text: format!("Rival adventuring parties clashed! {} and allies defeated {} and allies.",
                    winner_names.join(", "), loser_names.join(", ")),
                entity_ids: winners.iter().chain(losers.iter()).map(|&mi| state.entities[mi].id).collect(),
            });
        }
    }
}
