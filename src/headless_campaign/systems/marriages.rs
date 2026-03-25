//! Diplomatic marriage system — every 500 ticks.
//!
//! Adventurers can marry faction nobles to cement alliances, gain territory,
//! or resolve conflicts. Marriages provide relation bonuses, dowries, and
//! potential heirs, but lock adventurer loyalty to the faction.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    AdventurerStatus, BehaviorLedger, CampaignState, DiplomaticStance, DiseaseStatus, Marriage, MoodState, lcg_f32, lcg_next,
};

/// Maximum number of active marriages the guild can maintain.
pub const MAX_ACTIVE_MARRIAGES: usize = 3;

/// Relation threshold for a faction to propose marriage.
const PROPOSAL_RELATION_THRESHOLD: f32 = 40.0;

/// Minimum adventurer level to be eligible for marriage.
const MIN_MARRIAGE_LEVEL: u32 = 5;

/// Chance per eligible pair per 500-tick check (5%).
const PROPOSAL_CHANCE: f32 = 0.05;

/// Immediate relation bonus when marriage is arranged.
const MARRIAGE_RELATION_BONUS: f32 = 20.0;

/// Ongoing relation bonus per tick_marriages call (family ties).
const ONGOING_RELATION_BONUS: f32 = 2.0;

/// Ticks after marriage before heir chance activates.
const HEIR_GESTATION_TICKS: u64 = 2000;

/// Chance of heir per tick_marriages check after gestation period (10%).
const HEIR_CHANCE: f32 = 0.10;

/// Relation penalty for divorce.
const DIVORCE_RELATION_PENALTY: f32 = 40.0;

/// Reputation penalty for divorce.
const DIVORCE_REPUTATION_PENALTY: f32 = 10.0;

/// Morale penalty when spouse dies (faction attacked).
const SPOUSE_DEATH_MORALE_PENALTY: f32 = 30.0;

/// Relation penalty when spouse dies.
const SPOUSE_DEATH_RELATION_PENALTY: f32 = 20.0;

/// Main tick function. Called every 500 ticks.
///
/// 1. Ongoing relation bonus for active marriages
/// 2. Heir generation chance for marriages older than 2000 ticks
/// 3. Loyalty crisis detection (faction at war with guild)
/// 4. Spouse death detection (faction under attack)
pub fn tick_marriages(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 500 != 0 {
        return;
    }

    // Collect marriage IDs to process (avoid borrow issues).
    let marriage_ids: Vec<u32> = state.marriages.iter().map(|m| m.id).collect();

    for marriage_id in marriage_ids {
        // Re-find the marriage each iteration since state may change.
        let marriage_data = state.marriages.iter().find(|m| m.id == marriage_id).cloned();
        let Some(marriage) = marriage_data else {
            continue;
        };

        // 1. Ongoing relation bonus (family ties).
        if let Some(faction) = state
            .factions
            .iter_mut()
            .find(|f| f.id == marriage.faction_id)
        {
            let old = faction.relationship_to_guild;
            faction.relationship_to_guild =
                (faction.relationship_to_guild + ONGOING_RELATION_BONUS).min(100.0);
            if (faction.relationship_to_guild - old).abs() > 0.01 {
                events.push(WorldEvent::FactionRelationChanged {
                    faction_id: marriage.faction_id,
                    old,
                    new: faction.relationship_to_guild,
                });
            }
        }

        // 2. Heir generation.
        let ticks_married = state.tick.saturating_sub(marriage.married_tick);
        if ticks_married >= HEIR_GESTATION_TICKS && !marriage.produces_heir {
            let roll = lcg_f32(&mut state.rng);
            if roll < HEIR_CHANCE {
                // Mark marriage as having produced an heir.
                if let Some(m) = state.marriages.iter_mut().find(|m| m.id == marriage_id) {
                    m.produces_heir = true;
                }

                // Create a free high-stat recruit.
                let heir = create_heir(state, &marriage);
                let heir_id = heir.id;
                let heir_name = heir.name.clone();
                state.adventurers.push(heir);

                events.push(WorldEvent::HeirBorn {
                    marriage_id,
                    adventurer_id: heir_id,
                    heir_name,
                });
            }
        }

        // 3. Loyalty crisis: faction at war with guild.
        let faction_at_war = state
            .factions
            .iter()
            .find(|f| f.id == marriage.faction_id)
            .map(|f| f.diplomatic_stance == DiplomaticStance::AtWar)
            .unwrap_or(false);

        if faction_at_war {
            // Adventurer may desert.
            let desert_roll = lcg_f32(&mut state.rng);
            if desert_roll < 0.3 {
                // Adventurer deserts.
                if let Some(adv) = state
                    .adventurers
                    .iter_mut()
                    .find(|a| a.id == marriage.adventurer_id)
                {
                    if adv.status != AdventurerStatus::Dead {
                        adv.status = AdventurerStatus::Dead; // Deserted = removed from play
                        events.push(WorldEvent::AdventurerDeserted {
                            adventurer_id: marriage.adventurer_id,
                            reason: format!(
                                "Loyalty crisis: married to {} noble while faction is at war with guild",
                                state.factions.iter().find(|f| f.id == marriage.faction_id)
                                    .map(|f| f.name.as_str()).unwrap_or("unknown")
                            ),
                        });
                    }
                }
                // Remove the marriage.
                state.marriages.retain(|m| m.id != marriage_id);
            } else {
                // Crisis event but adventurer stays (conflicted).
                events.push(WorldEvent::MarriageCrisis {
                    marriage_id,
                    adventurer_id: marriage.adventurer_id,
                    faction_id: marriage.faction_id,
                    reason: "Faction at war with guild — loyalty divided".into(),
                });
                // Morale penalty.
                if let Some(adv) = state
                    .adventurers
                    .iter_mut()
                    .find(|a| a.id == marriage.adventurer_id)
                {
                    adv.morale = (adv.morale - 15.0).max(0.0);
                    adv.stress = (adv.stress + 10.0).min(100.0);
                }
            }
        }

        // 4. Spouse death: faction military strength dropped to 0 (destroyed).
        let faction_destroyed = state
            .factions
            .iter()
            .find(|f| f.id == marriage.faction_id)
            .map(|f| f.military_strength <= 0.0)
            .unwrap_or(true);

        if faction_destroyed {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == marriage.adventurer_id)
            {
                if adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale - SPOUSE_DEATH_MORALE_PENALTY).max(0.0);
                }
            }
            if let Some(faction) = state
                .factions
                .iter_mut()
                .find(|f| f.id == marriage.faction_id)
            {
                let old = faction.relationship_to_guild;
                faction.relationship_to_guild =
                    (faction.relationship_to_guild - SPOUSE_DEATH_RELATION_PENALTY).max(-100.0);
                events.push(WorldEvent::FactionRelationChanged {
                    faction_id: marriage.faction_id,
                    old,
                    new: faction.relationship_to_guild,
                });
            }
            // Remove the marriage (spouse dead).
            state.marriages.retain(|m| m.id != marriage_id);
        }
    }
}

/// Check if an adventurer is eligible for marriage.
pub fn is_eligible(state: &CampaignState, adventurer_id: u32) -> bool {
    let adv = state.adventurers.iter().find(|a| a.id == adventurer_id);
    match adv {
        Some(a) => {
            a.level >= MIN_MARRIAGE_LEVEL
                && a.status != AdventurerStatus::Dead
                && !state
                    .marriages
                    .iter()
                    .any(|m| m.adventurer_id == adventurer_id)
        }
        None => false,
    }
}

/// Check if a faction is eligible to offer marriage.
pub fn faction_eligible(state: &CampaignState, faction_id: usize) -> bool {
    state
        .factions
        .iter()
        .find(|f| f.id == faction_id)
        .map(|f| {
            f.relationship_to_guild > PROPOSAL_RELATION_THRESHOLD
                && f.diplomatic_stance != DiplomaticStance::AtWar
                && f.diplomatic_stance != DiplomaticStance::Hostile
        })
        .unwrap_or(false)
}

/// Apply a marriage between an adventurer and a faction noble.
pub fn arrange_marriage(
    state: &mut CampaignState,
    adventurer_id: u32,
    faction_id: usize,
    events: &mut Vec<WorldEvent>,
) -> Result<(), String> {
    // Validate.
    if state.marriages.len() >= MAX_ACTIVE_MARRIAGES {
        return Err("Maximum marriages reached (3)".into());
    }
    if !is_eligible(state, adventurer_id) {
        return Err("Adventurer not eligible for marriage".into());
    }
    if !faction_eligible(state, faction_id) {
        return Err("Faction not eligible for marriage".into());
    }

    // Generate noble name.
    let noble_name = generate_noble_name(state);

    // Calculate dowry (50-200 based on faction strength).
    let faction_strength = state
        .factions
        .iter()
        .find(|f| f.id == faction_id)
        .map(|f| f.military_strength)
        .unwrap_or(50.0);
    let dowry_base = 50.0 + (faction_strength / 100.0) * 150.0;
    let dowry = dowry_base.clamp(50.0, 200.0);

    // Create marriage.
    let marriage_id = state.next_event_id;
    state.next_event_id += 1;

    let marriage = Marriage {
        id: marriage_id,
        adventurer_id,
        faction_id,
        noble_name: noble_name.clone(),
        married_tick: state.tick,
        relation_bonus: MARRIAGE_RELATION_BONUS,
        dowry_received: dowry,
        produces_heir: false,
    };

    state.marriages.push(marriage);

    // Apply immediate benefits.
    state.guild.gold += dowry;

    // Relation bonus.
    if let Some(faction) = state.factions.iter_mut().find(|f| f.id == faction_id) {
        let old = faction.relationship_to_guild;
        faction.relationship_to_guild =
            (faction.relationship_to_guild + MARRIAGE_RELATION_BONUS).min(100.0);
        events.push(WorldEvent::FactionRelationChanged {
            faction_id,
            old,
            new: faction.relationship_to_guild,
        });
    }

    // Lock adventurer loyalty to faction.
    if let Some(adv) = state
        .adventurers
        .iter_mut()
        .find(|a| a.id == adventurer_id)
    {
        adv.faction_id = Some(faction_id);
        adv.loyalty = 100.0;
    }

    let faction_name = state
        .factions
        .iter()
        .find(|f| f.id == faction_id)
        .map(|f| f.name.clone())
        .unwrap_or_else(|| format!("Faction {}", faction_id));

    events.push(WorldEvent::MarriageArranged {
        marriage_id,
        adventurer_id,
        faction_id,
        noble_name,
        dowry,
    });

    events.push(WorldEvent::GoldChanged {
        amount: dowry,
        reason: format!("Marriage dowry from {}", faction_name),
    });

    Ok(())
}

/// Process a divorce.
pub fn divorce(
    state: &mut CampaignState,
    marriage_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<(), String> {
    let marriage = state
        .marriages
        .iter()
        .find(|m| m.id == marriage_id)
        .cloned();

    let Some(marriage) = marriage else {
        return Err(format!("Marriage {} not found", marriage_id));
    };

    // Remove marriage.
    state.marriages.retain(|m| m.id != marriage_id);

    // Apply penalties.
    if let Some(faction) = state
        .factions
        .iter_mut()
        .find(|f| f.id == marriage.faction_id)
    {
        let old = faction.relationship_to_guild;
        faction.relationship_to_guild =
            (faction.relationship_to_guild - DIVORCE_RELATION_PENALTY).max(-100.0);
        events.push(WorldEvent::FactionRelationChanged {
            faction_id: marriage.faction_id,
            old,
            new: faction.relationship_to_guild,
        });
    }

    state.guild.reputation =
        (state.guild.reputation - DIVORCE_REPUTATION_PENALTY).max(0.0);

    // Unlock adventurer from faction.
    if let Some(adv) = state
        .adventurers
        .iter_mut()
        .find(|a| a.id == marriage.adventurer_id)
    {
        adv.faction_id = None;
    }

    events.push(WorldEvent::Divorced {
        marriage_id,
        adventurer_id: marriage.adventurer_id,
        faction_id: marriage.faction_id,
        relation_penalty: DIVORCE_RELATION_PENALTY,
    });

    Ok(())
}

/// Create a high-stat heir adventurer from a marriage.
fn create_heir(state: &mut CampaignState, marriage: &Marriage) -> super::super::state::Adventurer {
    use super::super::state::{AdventurerStats, Equipment};

    let heir_id = state.next_event_id;
    state.next_event_id += 1;

    let parent_name = state
        .adventurers
        .iter()
        .find(|a| a.id == marriage.adventurer_id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| "Unknown".into());

    let parent_archetype = state
        .adventurers
        .iter()
        .find(|a| a.id == marriage.adventurer_id)
        .map(|a| a.archetype.clone())
        .unwrap_or_else(|| "knight".into());

    let heir_name = format!("{} Jr.", parent_name);

    // High stats — better than a typical level 5 recruit.
    super::super::state::Adventurer {
        id: heir_id,
        name: heir_name,
        archetype: parent_archetype,
        level: 5,
        xp: 0,
        stats: AdventurerStats {
            max_hp: 120.0,
            attack: 18.0,
            defense: 14.0,
            speed: 12.0,
            ability_power: 10.0,
        },
        equipment: Equipment::default(),
        traits: vec!["Noble Heritage".into()],
        status: AdventurerStatus::Idle,
        loyalty: 80.0,
        stress: 0.0,
        fatigue: 0.0,
        injury: 0.0,
        resolve: 70.0,
        morale: 80.0,
        party_id: None,
        guild_relationship: 50.0,
        leadership_role: None,
        is_player_character: false,
        faction_id: Some(marriage.faction_id),
        rallying_to: None,
        tier_status: Default::default(),
        history_tags: Default::default(),

        backstory: None,

        deeds: Vec::new(),

        hobbies: Vec::new(),

        disease_status: DiseaseStatus::Healthy,

        mood_state: MoodState::default(),

        fears: Vec::new(),

        personal_goal: None,

        journal: Vec::new(),

        equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
            behavior_ledger: BehaviorLedger::default(),
            classes: Vec::new(),
            skill_state: Default::default(),
    }
}

/// Generate a deterministic noble name.
fn generate_noble_name(state: &mut CampaignState) -> String {
    const NOBLE_FIRST: &[&str] = &[
        "Elara", "Cedric", "Isolde", "Aldric", "Rowena",
        "Theron", "Lysandra", "Gareth", "Vivienne", "Dorian",
        "Seraphina", "Alaric", "Cordelia", "Magnus", "Evangeline",
    ];
    const NOBLE_HOUSE: &[&str] = &[
        "Ashford", "Blackthorn", "Crestwood", "Draymoor", "Evermont",
        "Fairhaven", "Goldcrest", "Hawkwood", "Ironvale", "Jasperwick",
    ];

    let first_idx = (lcg_next(&mut state.rng) as usize) % NOBLE_FIRST.len();
    let house_idx = (lcg_next(&mut state.rng) as usize) % NOBLE_HOUSE.len();
    format!("{} of House {}", NOBLE_FIRST[first_idx], NOBLE_HOUSE[house_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligible_adventurer_checks() {
        let mut state = CampaignState::default_test_campaign(42);
        // Need to initialize the campaign first.
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;

        // Add a test adventurer at level 5.
        let adv = crate::headless_campaign::state::Adventurer {
            id: 100,
            name: "Test Hero".into(),
            archetype: "knight".into(),
            level: 5,
            xp: 0,
            stats: crate::headless_campaign::state::AdventurerStats {
                max_hp: 100.0,
                attack: 15.0,
                defense: 10.0,
                speed: 10.0,
                ability_power: 5.0,
            },
            equipment: Default::default(),
            traits: vec![],
            status: AdventurerStatus::Idle,
            loyalty: 50.0,
            stress: 0.0,
            fatigue: 0.0,
            injury: 0.0,
            resolve: 50.0,
            morale: 50.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),

            backstory: None,

            deeds: Vec::new(),

            hobbies: Vec::new(),

            disease_status: DiseaseStatus::Healthy,

            mood_state: MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
            behavior_ledger: BehaviorLedger::default(),
            classes: Vec::new(),
            skill_state: Default::default(),
        };
        state.adventurers.push(adv);

        assert!(is_eligible(&state, 100));

        // Level too low.
        state.adventurers.last_mut().unwrap().level = 3;
        assert!(!is_eligible(&state, 100));
    }

    #[test]
    fn max_marriages_enforced() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = crate::headless_campaign::state::CampaignPhase::Playing;

        // Fill to max.
        for i in 0..MAX_ACTIVE_MARRIAGES {
            state.marriages.push(Marriage {
                id: i as u32,
                adventurer_id: i as u32,
                faction_id: 0,
                noble_name: format!("Noble {}", i),
                married_tick: 0,
                relation_bonus: 20.0,
                dowry_received: 100.0,
                produces_heir: false,
            });
        }

        let mut events = vec![];
        let result = arrange_marriage(&mut state, 99, 0, &mut events);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Maximum marriages"));
    }
}
