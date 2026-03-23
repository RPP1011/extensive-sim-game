//! Entity token export for V6 model training.
//!
//! Converts [`CampaignState`] into a flat sequence of typed [`EntityToken`]s
//! that the entity encoder consumes. Each token has a `type_id` discriminating
//! the entity kind and a variable-length `features` vector.

use serde::{Deserialize, Serialize};

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
            ],
        }
    }
}
