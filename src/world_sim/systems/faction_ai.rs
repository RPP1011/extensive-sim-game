#![allow(unused)]
//! Faction AI — every 600 ticks (~60s).
//!
//! Each faction evaluates its situation and picks from multiple possible
//! actions based on scoring. Maps mutations to WorldDelta variants.
//!
//! Ported from `crates/headless_campaign/src/systems/faction_ai.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

/// Cadence: every 600 ticks.
const FACTION_AI_INTERVAL: u64 = 100;

/// Default config values (mirrors FactionAiConfig defaults from campaign).
const ATTACK_POWER_FRACTION: f32 = 0.05;
const TERRITORY_CAPTURE_CONTROL: f32 = 30.0;
const HOSTILE_STRENGTH_GAIN: f32 = 2.0;
const WAR_DECLARATION_THRESHOLD: f32 = 60.0;
const WAR_DECLARATION_PENALTY: f32 = 20.0;
const NEUTRAL_CONTROL_GAIN: f32 = 2.0;
const FRIENDLY_RELATIONSHIP_GAIN: f32 = 1.0;


pub fn compute_faction_ai(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % FACTION_AI_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    for faction in &state.factions {
        let fi = faction.id;
        let strength = faction.military_strength;
        let max_strength = faction.max_military_strength;

        // --- Natural military strength regeneration ---
        // Strength grows toward max, faster for factions with combat NPCs.
        let combat_npcs = state.entities.iter().filter(|e| {
            e.alive && e.kind == EntityKind::Npc
                && e.npc.as_ref().map_or(false, |n| {
                    n.faction_id == Some(fi)
                        && n.behavior_value(tags::COMBAT) > 50.0
                })
        }).count() as f32;

        // Base regen + bonus from combat-trained NPCs.
        let base_regen = (max_strength - strength).max(0.0) * 0.05;
        let regen = base_regen + combat_npcs * 0.2;
        if regen > 0.01 {
            out.push(WorldDelta::UpdateFaction {
                faction_id: fi,
                field: FactionField::MilitaryStrength,
                value: regen,
            });
        }

        // --- Stance-based behavior ---
        match faction.diplomatic_stance {
            DiplomaticStance::Hostile | DiplomaticStance::AtWar => {
                // Hostile/at-war factions build up military faster.
                out.push(WorldDelta::UpdateFaction {
                    faction_id: fi,
                    field: FactionField::MilitaryStrength,
                    value: HOSTILE_STRENGTH_GAIN * 2.0,
                });

                // Hostile factions increase unrest in their regions.
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::UpdateRegion {
                            region_id: region.id,
                            field: RegionField::Unrest,
                            value: 0.5,
                        });
                    }
                }
            }

            DiplomaticStance::Friendly | DiplomaticStance::Coalition => {
                // Friendly/coalition factions stabilize their regions.
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::UpdateRegion {
                            region_id: region.id,
                            field: RegionField::Control,
                            value: NEUTRAL_CONTROL_GAIN,
                        });
                        out.push(WorldDelta::UpdateRegion {
                            region_id: region.id,
                            field: RegionField::Unrest,
                            value: -0.3,
                        });
                    }
                }

                // Relationship improvement.
                out.push(WorldDelta::UpdateFaction {
                    faction_id: fi,
                    field: FactionField::RelationshipToGuild,
                    value: FRIENDLY_RELATIONSHIP_GAIN,
                });
            }

            DiplomaticStance::Neutral => {
                // Neutral factions slowly consolidate control.
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::UpdateRegion {
                            region_id: region.id,
                            field: RegionField::Control,
                            value: NEUTRAL_CONTROL_GAIN * 0.5,
                        });
                    }
                }
            }

            DiplomaticStance::AtWar => {
                // War drains strength and treasury.
                let war_cost = 3.0 + strength * 0.03;
                out.push(WorldDelta::UpdateFaction {
                    faction_id: fi,
                    field: FactionField::MilitaryStrength,
                    value: -war_cost,
                });
                out.push(WorldDelta::UpdateFaction {
                    faction_id: fi,
                    field: FactionField::Treasury,
                    value: -war_cost * 2.0,
                });

                // War increases unrest in owned regions.
                for region in &state.regions {
                    if region.faction_id == Some(fi) {
                        out.push(WorldDelta::UpdateRegion {
                            region_id: region.id,
                            field: RegionField::Unrest,
                            value: 1.0,
                        });
                    }
                }
            }
        }

        // --- Reactive mobilization: factions that lost territory build up faster ---
        let current_territory = state.settlements.iter()
            .filter(|s| s.faction_id == Some(fi))
            .count() as u32;
        if current_territory < faction.territory_size && strength < max_strength {
            // Lost territory → emergency mobilization.
            let urgency = (faction.territory_size - current_territory) as f32 * 3.0;
            out.push(WorldDelta::UpdateFaction {
                faction_id: fi,
                field: FactionField::MilitaryStrength,
                value: urgency,
            });
        }

        // --- Chronicle notable faction events ---
        // Military strength crossing thresholds.
        let roll = entity_hash_f32(fi, state.tick, 77 as u64);
        if strength > 70.0 && roll < 0.1 {
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("{} musters a formidable army (strength {:.0}).", faction.name, strength),
                    entity_ids: vec![],
                },
            });
        }
    }

    // --- Defensive response: any faction that lost territory can fight to reclaim it ---
    for faction in &state.factions {
        // Hostile factions use the conquest system instead.
        if matches!(faction.diplomatic_stance, DiplomaticStance::Hostile | DiplomaticStance::AtWar) {
            continue;
        }
        // Check if this faction has lost territory (current < original).
        let current_territory = state.settlements.iter()
            .filter(|s| s.faction_id == Some(faction.id))
            .count() as u32;
        if current_territory < faction.territory_size && faction.military_strength > 40.0 {
            // Attempt to reclaim a settlement from the aggressor.
            if let Some(target) = state.settlements.iter()
                .filter(|s| s.faction_id.map_or(false, |fid| {
                    state.factions.iter().any(|f| f.id == fid
                        && matches!(f.diplomatic_stance, DiplomaticStance::Hostile | DiplomaticStance::AtWar))
                }))
                .min_by(|a, b| a.population.cmp(&b.population))
            {
                let defense = target.population as f32 / 10.0;
                if faction.military_strength > defense * 2.0 {
                    let target_owner = target.faction_id.and_then(|fid| state.factions.iter().find(|f| f.id == fid));
                    let owner_name = target_owner.map(|f| f.name.as_str()).unwrap_or("unknown");
                    out.push(WorldDelta::UpdateSettlementField {
                        settlement_id: target.id,
                        field: SettlementField::FactionId,
                        value: faction.id as f32,
                    });
                    out.push(WorldDelta::UpdateFaction {
                        faction_id: faction.id,
                        field: FactionField::MilitaryStrength,
                        value: -25.0,
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Battle,
                            text: format!("{} reclaimed {} from {}!",
                                faction.name, target.name, owner_name),
                            entity_ids: vec![],
                        },
                    });
                }
            }
        }
    }

    // --- Faction warfare: hostile/at-war factions attempt to conquer settlements ---
    if state.tick % CONQUEST_INTERVAL == 0 && state.tick >= CONQUEST_INTERVAL {
        compute_faction_conquests(state, out);
    }
}

/// Interval (ticks) between conquest attempts.
const CONQUEST_INTERVAL: u64 = 500;

/// Military strength required before a faction can attempt conquest.
const CONQUEST_MIN_STRENGTH: f32 = 60.0;

/// Strength cost on successful conquest.
const CONQUEST_SUCCESS_COST: f32 = 20.0;

/// Strength cost on failed conquest attempt.
const CONQUEST_FAILURE_COST: f32 = 10.0;

/// Evaluate faction conquest attempts. Hostile/AtWar factions with sufficient
/// military strength pick the weakest enemy settlement and attempt to take it.
fn compute_faction_conquests(state: &WorldState, out: &mut Vec<WorldDelta>) {
    for faction in &state.factions {
        // Only hostile or at-war factions attempt conquest.
        let is_aggressive = matches!(
            faction.diplomatic_stance,
            DiplomaticStance::Hostile | DiplomaticStance::AtWar
        );
        if !is_aggressive || faction.military_strength < CONQUEST_MIN_STRENGTH {
            continue;
        }

        // Rate limit: max one conquest per faction per 2000 ticks.
        // Check recent chronicle for this faction's conquests.
        let recent_conquest = state.chronicle.iter().rev().take(100).any(|e| {
            e.category == ChronicleCategory::Crisis
                && e.text.contains(&faction.name)
                && e.text.contains("conquered")
                && state.tick.saturating_sub(e.tick) < 2000
        });
        if recent_conquest { continue; }

        // Find enemy settlements: settlements owned by a different faction.
        // For AtWar factions, only target factions they are at war with.
        // For Hostile factions, target any other faction's settlements.
        let mut best_target: Option<(u32, f32)> = None; // (settlement_id, weakness_score)
        let mut target_faction_name = String::new();

        for settlement in &state.settlements {
            let owner_id = match settlement.faction_id {
                Some(id) if id != faction.id => id,
                _ => continue, // skip unowned or own settlements
            };

            // AtWar factions only attack their war targets.
            if faction.diplomatic_stance == DiplomaticStance::AtWar
                && !faction.at_war_with.contains(&owner_id)
            {
                continue;
            }

            // Weakness score: lower population + treasury = easier target.
            let weakness = settlement.population as f32 + settlement.treasury;

            if best_target.map_or(true, |(_, best_w)| weakness < best_w) {
                best_target = Some((settlement.id, weakness));
                // Look up target faction name for chronicle text.
                target_faction_name = state.factions.iter()
                    .find(|f| f.id == owner_id)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| format!("Faction {}", owner_id));
            }
        }

        let (target_id, _weakness) = match best_target {
            Some(t) => t,
            None => continue, // no valid target
        };

        let target_settlement = match state.settlement(target_id) {
            Some(s) => s,
            None => continue,
        };

        // Defense = population / 10.
        let defense = target_settlement.population as f32 / 10.0;
        let target_name = target_settlement.name.clone();

        if faction.military_strength > defense * 2.0 {
            // --- Conquest succeeds ---
            // Transfer settlement ownership to the attacker via world event.
            out.push(WorldDelta::RecordEvent {
                event: WorldEvent::SettlementConquered {
                    settlement_id: target_id,
                    new_faction_id: faction.id,
                },
            });

            // Reduce attacker military strength.
            out.push(WorldDelta::UpdateFaction {
                faction_id: faction.id,
                field: FactionField::MilitaryStrength,
                value: -CONQUEST_SUCCESS_COST,
            });

            // Chronicle: Crisis entry for conquest.
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Crisis,
                    text: format!(
                        "{} conquered {} from {}!",
                        faction.name, target_name, target_faction_name,
                    ),
                    entity_ids: vec![],
                },
            });
        } else {
            // --- Conquest fails ---
            // Reduce attacker military strength (failed assault).
            out.push(WorldDelta::UpdateFaction {
                faction_id: faction.id,
                field: FactionField::MilitaryStrength,
                value: -CONQUEST_FAILURE_COST,
            });

            // Chronicle: Battle entry for the failed attempt.
            out.push(WorldDelta::RecordChronicle {
                entry: ChronicleEntry {
                    tick: state.tick,
                    category: ChronicleCategory::Battle,
                    text: format!(
                        "{} launched a failed assault on {} (defended by {}).",
                        faction.name, target_name, target_faction_name,
                    ),
                    entity_ids: vec![],
                },
            });
        }
    }
}
