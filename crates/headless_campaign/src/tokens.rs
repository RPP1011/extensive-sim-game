//! Entity token export for V6 model training.
//!
//! Converts [`CampaignState`] into a flat sequence of typed [`EntityToken`]s
//! that the entity encoder consumes. Each token has a `type_id` discriminating
//! the entity kind and a variable-length `features` vector.

use serde::{Deserialize, Serialize};

use super::action_meta::action_space_summary_token;
use super::state::*;

/// A single entity token for the V6 entity encoder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityToken {
    pub type_id: u8,
    pub features: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Enum → index helpers
// ---------------------------------------------------------------------------

fn quest_type_index(qt: QuestType) -> f32 {
    match qt {
        QuestType::Combat => 0.0,
        QuestType::Exploration => 1.0,
        QuestType::Diplomatic => 2.0,
        QuestType::Escort => 3.0,
        QuestType::Rescue => 4.0,
        QuestType::Gather => 5.0,
    }
}

fn adventurer_status_index(s: AdventurerStatus) -> f32 {
    match s {
        AdventurerStatus::Idle => 0.0,
        AdventurerStatus::Assigned => 1.0,
        AdventurerStatus::Traveling => 2.0,
        AdventurerStatus::OnMission => 3.0,
        AdventurerStatus::Fighting => 4.0,
        AdventurerStatus::Injured => 5.0,
        AdventurerStatus::Dead => 6.0,
    }
}

fn party_status_index(s: PartyStatus) -> f32 {
    match s {
        PartyStatus::Idle => 0.0,
        PartyStatus::Traveling => 1.0,
        PartyStatus::OnMission => 2.0,
        PartyStatus::Fighting => 3.0,
        PartyStatus::Returning => 4.0,
    }
}

fn active_quest_status_index(s: ActiveQuestStatus) -> f32 {
    match s {
        ActiveQuestStatus::Preparing => 0.0,
        ActiveQuestStatus::Dispatched => 1.0,
        ActiveQuestStatus::InProgress => 2.0,
        ActiveQuestStatus::InCombat => 3.0,
        ActiveQuestStatus::Returning => 4.0,
        ActiveQuestStatus::NeedsSupport => 5.0,
    }
}

fn location_type_index(lt: LocationType) -> f32 {
    match lt {
        LocationType::Settlement => 0.0,
        LocationType::Wilderness => 1.0,
        LocationType::Dungeon => 2.0,
        LocationType::Ruin => 3.0,
        LocationType::Outpost => 4.0,
    }
}

fn diplomatic_stance_index(s: DiplomaticStance) -> f32 {
    match s {
        DiplomaticStance::Friendly => 0.0,
        DiplomaticStance::Neutral => 1.0,
        DiplomaticStance::Hostile => 2.0,
        DiplomaticStance::AtWar => 3.0,
        DiplomaticStance::Coalition => 4.0,
    }
}

fn unlock_category_index(c: UnlockCategory) -> f32 {
    match c {
        UnlockCategory::Information => 0.0,
        UnlockCategory::ActiveAbility => 1.0,
        UnlockCategory::PassiveBuff => 2.0,
        UnlockCategory::Economic => 3.0,
    }
}

fn base_type_index(bt: BaseType) -> f32 {
    match bt {
        BaseType::Camp => 0.0,
        BaseType::Fixed => 1.0,
    }
}

/// Map archetype string to a stable index. Unknown archetypes hash to a
/// catch-all bucket so novel archetypes still get a numeric feature.
fn archetype_index(archetype: &str) -> f32 {
    match archetype {
        "knight" => 0.0,
        "ranger" => 1.0,
        "mage" => 2.0,
        "cleric" => 3.0,
        "rogue" => 4.0,
        "paladin" => 5.0,
        "berserker" => 6.0,
        "necromancer" => 7.0,
        "bard" => 8.0,
        "druid" => 9.0,
        "warlock" => 10.0,
        "monk" => 11.0,
        "shaman" => 12.0,
        "artificer" => 13.0,
        "assassin" => 14.0,
        "guardian" => 15.0,
        "sorcerer" => 16.0,
        "warden" => 17.0,
        "archer" => 18.0,
        "healer" => 19.0,
        "warrior" => 20.0,
        "tank" => 21.0,
        "support" => 22.0,
        "caster" => 23.0,
        "fighter" => 24.0,
        "scout" => 25.0,
        "enchanter" => 26.0,
        _ => 99.0, // catch-all for unknown archetypes
    }
}

// ---------------------------------------------------------------------------
// Token construction
// ---------------------------------------------------------------------------

impl CampaignState {
    /// Convert the full campaign state into a flat sequence of entity tokens
    /// for the V6 entity encoder.
    pub fn to_tokens(&self) -> Vec<EntityToken> {
        let mut tokens = Vec::new();

        // Type 0: Guild state (singleton)
        tokens.push(self.guild_token());

        // Type 1: Adventurer (one per adventurer)
        for adv in &self.adventurers {
            tokens.push(Self::adventurer_token(adv));
        }

        // Type 2: Party (one per party)
        for party in &self.parties {
            tokens.push(Self::party_token(party));
        }

        // Type 3: Quest — board requests (status=0 available) + active quests
        for req in &self.request_board {
            tokens.push(self.quest_request_token(req));
        }
        for aq in &self.active_quests {
            tokens.push(self.active_quest_token(aq));
        }

        // Type 4: Battle (one per active battle)
        for battle in &self.active_battles {
            tokens.push(Self::battle_token(battle));
        }

        // Type 5: Location (one per location)
        for loc in &self.overworld.locations {
            tokens.push(Self::location_token(loc));
        }

        // Type 6: Faction (one per faction)
        for faction in &self.factions {
            tokens.push(Self::faction_token(faction));
        }

        // Type 7: Unlock (one per unlock)
        for unlock in &self.unlocks {
            tokens.push(Self::unlock_token(unlock));
        }

        // Type 8: Base (singleton)
        tokens.push(Self::base_token(&self.guild.base));

        // Type 9: Aggregate (singleton)
        tokens.push(self.aggregate_token());

        // Type 10: Action space summary (singleton)
        tokens.push(action_space_summary_token(self));

        tokens
    }

    // --- Type 0: Guild ---
    fn guild_token(&self) -> EntityToken {
        EntityToken {
            type_id: 0,
            features: vec![
                self.guild.gold,
                self.guild.supplies,
                self.guild.reputation,
                self.active_quests.len() as f32,
                self.adventurers.len() as f32,
                self.unlocks.len() as f32,
            ],
        }
    }

    // --- Type 1: Adventurer ---
    fn adventurer_token(adv: &Adventurer) -> EntityToken {
        EntityToken {
            type_id: 1,
            features: vec![
                archetype_index(&adv.archetype),
                adv.level as f32,
                adv.stats.max_hp,
                adv.stats.attack,
                adv.stats.defense,
                adv.stats.speed,
                adv.stats.ability_power,
                adventurer_status_index(adv.status),
                adv.loyalty,
                adv.stress,
                adv.fatigue,
                adv.injury,
                adv.resolve,
                adv.morale,
                if adv.party_id.is_some() { 1.0 } else { 0.0 },
            ],
        }
    }

    // --- Type 2: Party ---
    fn party_token(party: &Party) -> EntityToken {
        EntityToken {
            type_id: 2,
            features: vec![
                party.member_ids.len() as f32,
                party.position.0,
                party.position.1,
                party.speed,
                party_status_index(party.status),
                party.supply_level,
                party.morale,
                if party.quest_id.is_some() { 1.0 } else { 0.0 },
            ],
        }
    }

    // --- Type 3: Quest (board request — available) ---
    fn quest_request_token(&self, req: &QuestRequest) -> EntityToken {
        let deadline_total = req.deadline_ms.saturating_sub(req.arrived_at_ms);
        let elapsed = self.elapsed_ms.saturating_sub(req.arrived_at_ms);
        let elapsed_ratio = if deadline_total > 0 {
            (elapsed as f32 / deadline_total as f32).min(1.0)
        } else {
            0.0
        };

        EntityToken {
            type_id: 3,
            features: vec![
                quest_type_index(req.quest_type),
                req.threat_level,
                req.reward.gold,
                req.reward.reputation,
                req.distance,
                0.0, // status = available
                elapsed_ratio,
            ],
        }
    }

    // --- Type 3: Quest (active) ---
    fn active_quest_token(&self, aq: &ActiveQuest) -> EntityToken {
        let status_idx = match aq.status {
            ActiveQuestStatus::Preparing => 1.0,
            ActiveQuestStatus::Dispatched => 2.0,
            ActiveQuestStatus::InProgress => 3.0,
            ActiveQuestStatus::InCombat => 4.0,
            ActiveQuestStatus::Returning => 5.0,
            ActiveQuestStatus::NeedsSupport => 6.0,
        };

        // Elapsed time ratio: use the quest's deadline relative to acceptance
        let deadline_total = aq
            .request
            .deadline_ms
            .saturating_sub(aq.request.arrived_at_ms);
        let elapsed_ratio = if deadline_total > 0 {
            (aq.elapsed_ms as f32 / deadline_total as f32).min(1.0)
        } else {
            0.0
        };

        EntityToken {
            type_id: 3,
            features: vec![
                quest_type_index(aq.request.quest_type),
                aq.request.threat_level,
                aq.request.reward.gold,
                aq.request.reward.reputation,
                aq.request.distance,
                status_idx,
                elapsed_ratio,
            ],
        }
    }

    // --- Type 4: Battle ---
    fn battle_token(battle: &BattleState) -> EntityToken {
        EntityToken {
            type_id: 4,
            features: vec![
                battle.party_health_ratio,
                battle.enemy_health_ratio,
                battle.elapsed_ticks as f32,
                battle.predicted_outcome,
                if battle.runner_sent { 1.0 } else { 0.0 },
                if battle.mercenary_hired { 1.0 } else { 0.0 },
                if battle.rescue_called { 1.0 } else { 0.0 },
            ],
        }
    }

    // --- Type 5: Location ---
    fn location_token(loc: &Location) -> EntityToken {
        EntityToken {
            type_id: 5,
            features: vec![
                loc.position.0,
                loc.position.1,
                location_type_index(loc.location_type),
                loc.threat_level,
                loc.resource_availability,
                if loc.scouted { 1.0 } else { 0.0 },
                loc.faction_owner.map(|f| f as f32).unwrap_or(-1.0),
            ],
        }
    }

    // --- Type 6: Faction ---
    fn faction_token(faction: &FactionState) -> EntityToken {
        EntityToken {
            type_id: 6,
            features: vec![
                faction.relationship_to_guild,
                faction.military_strength,
                faction.territory_size as f32,
                diplomatic_stance_index(faction.diplomatic_stance),
            ],
        }
    }

    // --- Type 7: Unlock ---
    fn unlock_token(unlock: &UnlockInstance) -> EntityToken {
        let cooldown_ratio = if unlock.properties.cooldown_ms > 0 {
            unlock.cooldown_remaining_ms as f32 / unlock.properties.cooldown_ms as f32
        } else {
            0.0
        };

        EntityToken {
            type_id: 7,
            features: vec![
                unlock_category_index(unlock.category),
                cooldown_ratio,
                unlock.properties.magnitude,
                unlock.properties.duration_ms as f32,
                unlock.properties.resource_cost,
                if unlock.properties.is_passive { 1.0 } else { 0.0 },
                if unlock.active { 1.0 } else { 0.0 },
            ],
        }
    }

    // --- Type 8: Base (singleton) ---
    fn base_token(base: &BaseState) -> EntityToken {
        EntityToken {
            type_id: 8,
            features: vec![
                base_type_index(base.base_type),
                base.position.0,
                base.position.1,
                base.defensive_strength,
                base.upgrade_slots.len() as f32,
            ],
        }
    }

    // --- Type 9: Aggregate (singleton) ---
    fn aggregate_token(&self) -> EntityToken {
        // Normalize tick: assume ~36000 ticks per campaign hour, cap at 1.0
        let tick_norm = (self.tick as f32 / 36000.0).min(1.0);
        let gold_log = (self.guild.gold.max(1.0)).ln();
        let supplies_log = (self.guild.supplies.max(1.0)).ln();

        // Scouting: mean region visibility
        let mean_visibility = if self.overworld.regions.is_empty() {
            0.5
        } else {
            self.overworld.regions.iter().map(|r| r.visibility).sum::<f32>()
                / self.overworld.regions.len() as f32
        };

        // Bond network density
        let alive = self.adventurers.iter()
            .filter(|a| a.status != AdventurerStatus::Dead).count() as f32;
        let max_bonds = (alive * (alive - 1.0) / 2.0).max(1.0);
        let bond_density = self.adventurer_bonds.len() as f32 / max_bonds;
        let mean_bond_strength = if self.adventurer_bonds.is_empty() {
            0.0
        } else {
            self.adventurer_bonds.values().sum::<f32>()
                / self.adventurer_bonds.len() as f32 / 100.0
        };

        // Building tiers (sum normalized to 0-1)
        let b = &self.guild_buildings;
        let building_sum = (b.training_grounds + b.watchtower + b.trade_post
            + b.barracks + b.infirmary + b.war_room) as f32 / 18.0;

        // Rival guild state
        let rival_active = if self.rival_guild.active { 1.0 } else { 0.0 };
        let rival_rep = self.rival_guild.reputation / 100.0;
        let rival_power = self.rival_guild.power_level / 100.0;

        // Season (cyclical)
        let season_idx = match self.overworld.season {
            Season::Spring => 0.0,
            Season::Summer => 0.25,
            Season::Autumn => 0.5,
            Season::Winter => 0.75,
        };

        // Diplomacy summary
        let coalition_count = self.factions.iter()
            .filter(|f| f.coalition_member).count() as f32;
        let war_count = self.factions.iter()
            .filter(|f| f.diplomatic_stance == DiplomaticStance::AtWar).count() as f32;

        // Trade income
        let trade_income_log = (self.guild.total_trade_income.max(1.0)).ln() / 5.0;

        // Territory control
        let guild_faction_id = self.diplomacy.guild_faction_id;
        let guild_regions = self.overworld.regions.iter()
            .filter(|r| r.owner_faction_id == guild_faction_id).count() as f32;
        let territory_ratio = guild_regions / self.overworld.regions.len().max(1) as f32;

        EntityToken {
            type_id: 9,
            features: vec![
                tick_norm,
                self.adventurers.len() as f32,
                self.active_quests.len() as f32,
                self.active_battles.len() as f32,
                gold_log,
                supplies_log,
                self.guild.reputation / 100.0,
                self.overworld.campaign_progress,
                self.overworld.global_threat_level / 100.0,
                self.pending_choices.len() as f32,
                // Extended state dimensions:
                mean_visibility,
                bond_density,
                mean_bond_strength,
                building_sum,
                rival_active,
                rival_rep,
                rival_power,
                season_idx,
                coalition_count,
                war_count,
                trade_income_log,
                territory_ratio,
                // --- Extended system tracker dimensions (pass 3) ---
                // Espionage
                self.system_trackers.spy_count as f32,
                (self.system_trackers.total_intel_gathered.max(1.0)).ln() / 5.0,
                self.system_trackers.mean_spy_cover / 100.0,
                // Mercenaries
                self.system_trackers.mercenaries_hired as f32,
                (self.system_trackers.total_mercenary_strength / 50.0).min(1.0),
                self.system_trackers.mean_mercenary_loyalty / 100.0,
                // Black market
                self.system_trackers.black_market_heat / 100.0,
                (self.system_trackers.black_market_profit.max(1.0)).ln() / 5.0,
                self.system_trackers.active_deals as f32,
                // Prisoners
                self.system_trackers.prisoner_count as f32,
                self.system_trackers.captured_adventurer_count as f32,
                // Loans
                (self.system_trackers.total_debt.max(1.0)).ln() / 5.0,
                self.system_trackers.credit_rating / 100.0,
                // Rumors
                self.system_trackers.active_rumor_count as f32,
                self.system_trackers.investigated_rumor_count as f32,
                // Civil wars
                self.system_trackers.active_civil_war_count as f32,
                if self.system_trackers.guild_civil_war_involvement { 1.0 } else { 0.0 },
                // Diplomacy agreements
                self.system_trackers.trade_agreement_count as f32,
                self.system_trackers.non_aggression_pact_count as f32,
                self.system_trackers.mutual_defense_count as f32,
                (self.system_trackers.active_trade_income.max(1.0)).ln() / 5.0,
                // Rival guild extended
                self.system_trackers.rival_reputation_gap / 100.0,
                self.system_trackers.rival_power_gap / 100.0,
                self.system_trackers.rival_quest_competition_rate,
                // Caravans
                self.system_trackers.active_caravan_routes as f32,
                (self.system_trackers.caravan_trade_income.max(1.0)).ln() / 5.0,
                self.system_trackers.caravans_raided as f32,
                // Retirement
                self.system_trackers.retired_count as f32,
                (self.system_trackers.total_legacy_bonuses / 50.0).min(1.0),
                // Monster ecology
                (self.system_trackers.total_monster_population / 100.0).min(1.0),
                self.system_trackers.highest_monster_aggression / 100.0,
                // Festivals
                self.system_trackers.active_festival_count as f32,
                self.system_trackers.festivals_attended as f32,
                // Backstory
                self.system_trackers.has_backstory_ratio,
                self.system_trackers.rival_faction_connections as f32,
                // Mentorship
                self.system_trackers.active_mentorship_count as f32,
                // Rivalries
                self.system_trackers.active_rivalry_count as f32,
                self.system_trackers.mean_rivalry_intensity / 100.0,
                // War exhaustion
                self.system_trackers.max_war_exhaustion / 100.0,
                // Chronicle
                (self.system_trackers.chronicle_entry_count as f32 / 20.0).min(1.0),
                self.system_trackers.recent_tragedy_count as f32,
                // Deeds
                (self.system_trackers.total_deeds_earned as f32 / 20.0).min(1.0),
                // Population
                (self.system_trackers.total_population.max(1.0)).ln() / 8.0,
                self.system_trackers.mean_population_morale / 100.0,
                (self.system_trackers.total_tax_income.max(1.0)).ln() / 5.0,
                // Site preparation
                (self.system_trackers.total_site_preparation / 10.0).min(1.0),
                // --- Pass 4: expanded system coverage ---
                // Disease/Health (3 features)
                self.diseases.len() as f32 / 5.0_f32.max(1.0),  // active disease count, norm by 5
                (self.system_trackers.infected_adventurer_count as f32
                    / (self.adventurers.len().max(1) as f32)).min(1.0), // infected ratio
                {
                    let n = self.diseases.len();
                    if n == 0 { 0.0 }
                    else {
                        (self.diseases.iter().map(|d| d.severity).sum::<f32>() / n as f32 / 100.0).min(1.0)
                    }
                }, // mean disease severity (0-1)
                // Crafting/Equipment (3 features)
                (self.guild.inventory.len() as f32 / 50.0).min(1.0), // inventory item count norm
                {
                    // Mean equipment durability across all adventurers' equipped items
                    let (total_dur, count) = self.adventurers.iter()
                        .flat_map(|a| a.equipped_items.iter())
                        .fold((0.0_f32, 0u32), |(sum, cnt), item| (sum + item.durability, cnt + 1));
                    if count == 0 { 1.0 } else { (total_dur / count as f32 / 100.0).min(1.0) }
                }, // mean equip durability (0-1)
                (self.system_trackers.active_crafting_projects as f32 / 5.0).min(1.0),
                // Regional Strategy (3 features)
                {
                    let guild_fid = self.diplomacy.guild_faction_id;
                    let controlled = self.overworld.regions.iter()
                        .filter(|r| r.owner_faction_id == guild_fid).count() as f32;
                    controlled / self.overworld.regions.len().max(1) as f32
                }, // controlled region ratio (0-1)
                {
                    let n = self.overworld.regions.len();
                    if n == 0 { 0.0 }
                    else {
                        self.overworld.regions.iter().map(|r| r.threat_level).sum::<f32>()
                            / n as f32 / 100.0
                    }
                }, // mean regional threat (0-1)
                {
                    // Territory contestation: fraction of regions with high unrest (>50)
                    let n = self.overworld.regions.len();
                    if n == 0 { 0.0 }
                    else {
                        self.overworld.regions.iter()
                            .filter(|r| r.unrest > 50.0).count() as f32 / n as f32
                    }
                }, // contestation ratio (0-1)
                // Religion (3 features)
                {
                    // Total devotion across all temples, norm by temple count * 100
                    let n = self.temples.len();
                    if n == 0 { 0.0 }
                    else {
                        (self.temples.iter().map(|t| t.devotion).sum::<f32>()
                            / (n as f32 * 100.0)).min(1.0)
                    }
                }, // mean devotion (0-1)
                (self.temples.len() as f32 / 10.0).min(1.0), // temple count norm
                {
                    let active = self.temples.iter().filter(|t| t.blessing_active).count();
                    let n = self.temples.len();
                    if n == 0 { 0.0 } else { active as f32 / n as f32 }
                }, // active blessings ratio (0-1)
                // Crime/Underworld (3 features)
                (self.system_trackers.total_crimes_committed as f32 / 20.0).min(1.0),
                {
                    // Wanted level: sum of bounty amounts on guild adventurers
                    let total_bounty: f32 = self.wanted_posters.iter()
                        .map(|w| w.bounty_amount).sum();
                    (total_bounty / 500.0).min(1.0)
                }, // wanted level norm (0-1)
                (self.system_trackers.active_heist_count as f32 / 5.0).min(1.0),
                // Knowledge/Research (3 features)
                (self.archives.knowledge_points / 500.0).min(1.0), // archive score norm
                {
                    let active = self.archives.research_topics.iter()
                        .filter(|t| t.progress < 1.0).count();
                    (active as f32 / 5.0).min(1.0)
                }, // active research projects norm
                (self.system_trackers.discoveries_made as f32 / 20.0).min(1.0),
                // Weather/Environment (3 features)
                {
                    // Season severity: winter=1.0, summer=0.0, spring/autumn=0.5
                    match self.overworld.season {
                        Season::Spring => 0.3,
                        Season::Summer => 0.1,
                        Season::Autumn => 0.5,
                        Season::Winter => 1.0,
                    }
                }, // season severity (0-1)
                (self.overworld.active_weather.len() as f32 / 5.0).min(1.0), // active weather events norm
                {
                    // Terrain stability: fraction of regions NOT affected by terrain events
                    let affected: std::collections::HashSet<usize> = self.terrain_events.iter()
                        .flat_map(|e| e.affected_regions.iter().copied())
                        .collect();
                    let n = self.overworld.regions.len();
                    if n == 0 { 1.0 }
                    else { 1.0 - (affected.len() as f32 / n as f32).min(1.0) }
                }, // terrain stability (0-1, higher = more stable)
                // Victory Progress (3 features)
                self.victory_progress.percent.min(1.0), // overall victory progress (0-1)
                (self.victory_progress.quests_completed as f32 / 50.0).min(1.0), // quest completion norm
                (self.victory_progress.total_gold_earned / 5000.0).min(1.0), // gold earning norm
                // Exploration (3 features)
                self.exploration.exploration_percentage.min(1.0), // map exploration (0-1)
                (self.exploration.landmarks_discovered.len() as f32 / 20.0).min(1.0), // landmarks norm
                {
                    let max_depth = self.dungeons.iter()
                        .map(|d| d.depth).max().unwrap_or(0);
                    (max_depth as f32 / 10.0).min(1.0)
                }, // max dungeon depth norm (0-1)
                // Guild Operations (3 features)
                (self.pending_choices.len() as f32 / 5.0).min(1.0), // pending choices norm
                (self.mentor_assignments.len() as f32 / 5.0).min(1.0), // training queue (mentors) norm
                (self.system_trackers.active_liaisons as f32 / 5.0).min(1.0), // active liaisons norm
                // Class System (15 features)
                {
                    let total: f32 = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .map(|a| a.classes.len() as f32).sum();
                    (total / 50.0).min(1.0)
                }, // total classes granted
                {
                    let all_classes: Vec<_> = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .flat_map(|a| a.classes.iter()).collect();
                    if all_classes.is_empty() { 0.0 } else {
                        all_classes.iter().map(|c| c.level as f32).sum::<f32>()
                            / all_classes.len() as f32 / 25.0
                    }
                }, // mean class level / 25
                {
                    self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .flat_map(|a| a.classes.iter())
                        .map(|c| c.level as f32)
                        .fold(0.0f32, f32::max) / 30.0
                }, // max class level / 30
                {
                    let total: f32 = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .flat_map(|a| a.classes.iter())
                        .map(|c| c.skills_granted.len() as f32).sum();
                    (total / 100.0).min(1.0)
                }, // total skills granted
                {
                    let mut names: std::collections::HashSet<&str> = std::collections::HashSet::new();
                    for a in self.adventurers.iter().filter(|a| a.status != AdventurerStatus::Dead) {
                        for c in &a.classes { names.insert(&c.class_name); }
                    }
                    (names.len() as f32 / 20.0).min(1.0)
                }, // unique class count
                {
                    let shame = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter())
                        .filter(|c| c.class_name == "Coward" || c.class_name == "Oathbreaker" || c.class_name == "Deserter")
                        .count() as f32;
                    (shame / 10.0).min(1.0)
                }, // shame class count
                {
                    let crisis = self.unique_class_holders.len() as f32;
                    (crisis / 5.0).min(1.0)
                }, // crisis/unique class count
                (self.consolidation_offers.len() as f32 / 10.0).min(1.0), // consolidation offers
                {
                    let all_classes: Vec<_> = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter()).collect();
                    if all_classes.is_empty() { 1.0 } else {
                        all_classes.iter().map(|c| c.identity_coherence).sum::<f32>()
                            / all_classes.len() as f32
                    }
                }, // mean identity coherence (0-1)
                {
                    let all_classes: Vec<_> = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter()).collect();
                    if all_classes.is_empty() { 0.0 } else {
                        all_classes.iter().map(|c| c.stagnation_ticks as f32).sum::<f32>()
                            / all_classes.len() as f32 / 1000.0
                    }
                }, // mean stagnation / 1000
                {
                    let total_classes: f32 = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter()).count() as f32;
                    let mut names: std::collections::HashSet<&str> = std::collections::HashSet::new();
                    for a in &self.adventurers { for c in &a.classes { names.insert(&c.class_name); } }
                    if total_classes > 0.0 { names.len() as f32 / total_classes } else { 0.0 }
                }, // class diversity ratio
                {
                    let alive: Vec<_> = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead).collect();
                    if alive.is_empty() { 0.0 } else {
                        let sum: f32 = alive.iter().map(|a| {
                            a.behavior_ledger.melee_combat + a.behavior_ledger.ranged_combat
                            + a.behavior_ledger.healing_given + a.behavior_ledger.diplomacy_actions
                            + a.behavior_ledger.trades_completed + a.behavior_ledger.items_crafted
                            + a.behavior_ledger.areas_explored + a.behavior_ledger.units_commanded
                            + a.behavior_ledger.stealth_actions + a.behavior_ledger.research_performed
                            + a.behavior_ledger.damage_absorbed + a.behavior_ledger.allies_supported
                        }).sum();
                        (sum / alive.len() as f32 / 1000.0).min(1.0)
                    }
                }, // mean behavior intensity
                {
                    let alive = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead).count() as f32;
                    let multi = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead && a.classes.len() >= 2)
                        .count() as f32;
                    if alive > 0.0 { multi / alive } else { 0.0 }
                }, // multi-class ratio
                0.0, // evolution count (placeholder)
                // Class progression detail (6 features)
                {
                    // classes_with_ability_dsl: how many skills have actual combat DSL
                    let with_dsl = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter())
                        .flat_map(|c| c.skills_granted.iter())
                        .filter(|s| s.ability_dsl.is_some())
                        .count() as f32;
                    (with_dsl / 50.0).min(1.0)
                },
                {
                    // skill_rarity_distribution: fraction of skills that are Uncommon+
                    let total_skills = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter())
                        .flat_map(|c| c.skills_granted.iter())
                        .count() as f32;
                    let rare_plus = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter())
                        .flat_map(|c| c.skills_granted.iter())
                        .filter(|s| !matches!(s.rarity, SkillRarity::Common))
                        .count() as f32;
                    if total_skills > 0.0 { rare_plus / total_skills } else { 0.0 }
                },
                {
                    // starter_class_count: how many starter classes exist
                    let starters = ["Laborer", "Hunter", "Traveler", "Apprentice", "Farmhand",
                                     "Militia", "Peddler", "Herbalist", "Scribe", "Pickpocket",
                                     "Errand Runner", "Stablehand"];
                    let count = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter())
                        .filter(|c| starters.contains(&c.class_name.as_str()))
                        .count() as f32;
                    (count / 20.0).min(1.0)
                },
                {
                    // evolved_class_count: non-starter, non-shame classes
                    let starters = ["Laborer", "Hunter", "Traveler", "Apprentice", "Farmhand",
                                     "Militia", "Peddler", "Herbalist", "Scribe", "Pickpocket",
                                     "Errand Runner", "Stablehand"];
                    let shame = ["Coward", "Oathbreaker", "Deserter"];
                    let count = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter())
                        .filter(|c| !starters.contains(&c.class_name.as_str()) && !shame.contains(&c.class_name.as_str()))
                        .count() as f32;
                    (count / 20.0).min(1.0)
                },
                {
                    // behavior_specialization: fraction of behavior axes active
                    let alive: Vec<_> = self.adventurers.iter()
                        .filter(|a| a.status != AdventurerStatus::Dead).collect();
                    if alive.is_empty() { 0.0 } else {
                        let axes: Vec<f32> = (0..12).map(|axis| {
                            alive.iter().map(|a| {
                                let l = &a.behavior_ledger;
                                match axis {
                                    0 => l.melee_combat, 1 => l.ranged_combat, 2 => l.healing_given,
                                    3 => l.diplomacy_actions, 4 => l.trades_completed, 5 => l.items_crafted,
                                    6 => l.areas_explored, 7 => l.units_commanded, 8 => l.stealth_actions,
                                    9 => l.research_performed, 10 => l.damage_absorbed, _ => l.allies_supported,
                                }
                            }).sum::<f32>()
                        }).collect();
                        let nonzero = axes.iter().filter(|&&v| v > 0.1).count() as f32;
                        (nonzero / 12.0).min(1.0)
                    }
                },
                {
                    // class_level_distribution: fraction of classes above level 5 (T2+)
                    let all_classes: Vec<_> = self.adventurers.iter()
                        .flat_map(|a| a.classes.iter()).collect();
                    if all_classes.is_empty() { 0.0 } else {
                        let above_5 = all_classes.iter().filter(|c| c.level >= 5).count() as f32;
                        above_5 / all_classes.len() as f32
                    }
                },
            ],
        }
    }
}
