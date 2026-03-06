use crate::game_core::{self, CharacterCreationState, HubScreen, HubUiState};
use crate::hub_types::CharacterCreationUiState;

#[derive(Clone)]
pub struct FactionSelectionChoice {
    pub index: usize,
    pub id: String,
    pub name: String,
    pub impact: String,
}

#[derive(Clone)]
pub struct BackstorySelectionChoice {
    pub id: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
    pub stat_modifiers: Vec<String>,
    pub recruit_bias_modifiers: Vec<String>,
    pub preferred_recruit_archetypes: Vec<game_core::PersonalityArchetype>,
    pub player_resolve_delta: f32,
    pub player_loyalty_delta: f32,
    pub player_stress_delta: f32,
    pub player_fatigue_delta: f32,
    pub party_speed_delta: f32,
    pub party_supply_delta: f32,
}

pub fn faction_id_from_name(index: usize, name: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in name.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        format!("faction-{}", index)
    } else {
        format!("faction-{}-{}", index, slug)
    }
}

pub fn faction_impact_text(index: usize, name: &str, strength: f32, cohesion: f32) -> String {
    let doctrine = match index % 3 {
        0 => "Merchant doctrine: stronger supply stability and safer trade lanes.",
        1 => "Frontier doctrine: stronger patrol pressure and rapid response.",
        _ => "River doctrine: stronger recruit throughput and mobility support.",
    };
    format!(
        "{doctrine} Starting profile: strength {:.0}, cohesion {:.0}. Recruit pools skew toward {} territories.",
        strength, cohesion, name
    )
}

pub fn build_faction_selection_choices(
    overworld: &game_core::OverworldMap,
) -> Vec<FactionSelectionChoice> {
    overworld
        .factions
        .iter()
        .map(|f| FactionSelectionChoice {
            index: f.id,
            id: faction_id_from_name(f.id, &f.name),
            name: f.name.clone(),
            impact: faction_impact_text(f.id, &f.name, f.strength, f.cohesion),
        })
        .collect()
}

pub fn confirm_faction_selection(
    hub_ui: &mut HubUiState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    diplomacy: &mut game_core::DiplomacyState,
    overworld: &game_core::OverworldMap,
) -> bool {
    let choices = build_faction_selection_choices(overworld);
    let Some(selected_idx) = character_creation.selected_faction_index else {
        creation_ui.status = "Select a faction before continuing to backstory.".to_string();
        return false;
    };
    let Some(choice) = choices.iter().find(|c| c.index == selected_idx) else {
        creation_ui.status =
            "Selected faction is no longer valid. Choose a faction and continue.".to_string();
        character_creation.selected_faction_index = None;
        character_creation.selected_faction_id = None;
        return false;
    };
    character_creation.selected_faction_id = Some(choice.id.clone());
    character_creation.selected_backstory_id = None;
    character_creation.stat_modifiers.clear();
    character_creation.recruit_bias_modifiers.clear();
    character_creation.is_confirmed = false;
    diplomacy.player_faction_id = choice.index.min(overworld.factions.len().saturating_sub(1));
    creation_ui.status = format!(
        "Faction '{}' confirmed. Backstory selection is next.",
        choice.name
    );
    hub_ui.screen = HubScreen::CharacterCreationBackstory;
    true
}

pub fn build_backstory_selection_choices() -> Vec<BackstorySelectionChoice> {
    vec![
        BackstorySelectionChoice {
            id: "scout-pathfinder",
            name: "Scout",
            summary: "Years spent charting contested roads made you fast and hard to ambush.",
            stat_modifiers: vec![
                "Scouting: +2".to_string(),
                "Route awareness: +1".to_string(),
                "Party speed: +0.15".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: higher chance for Tactician and Vanguard recruits."
                    .to_string(),
                "Border-region volunteers appear more often in the recruit pool.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Tactician,
                game_core::PersonalityArchetype::Vanguard,
            ],
            player_resolve_delta: 4.0,
            player_loyalty_delta: 1.0,
            player_stress_delta: -3.0,
            player_fatigue_delta: -2.0,
            party_speed_delta: 0.15,
            party_supply_delta: 3.0,
        },
        BackstorySelectionChoice {
            id: "quartermaster-logistician",
            name: "Quartermaster",
            summary: "You kept columns fed and paid, even through siege winters.",
            stat_modifiers: vec![
                "Logistics: +2".to_string(),
                "Supply reserve: +12".to_string(),
                "Player loyalty: +5".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Guardian and Tactician archetypes.".to_string(),
                "Disciplined support specialists enter the pool first.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Guardian,
                game_core::PersonalityArchetype::Tactician,
            ],
            player_resolve_delta: 2.0,
            player_loyalty_delta: 5.0,
            player_stress_delta: -2.0,
            player_fatigue_delta: -1.0,
            party_speed_delta: 0.05,
            party_supply_delta: 12.0,
        },
        BackstorySelectionChoice {
            id: "raider-veteran",
            name: "Raider",
            summary: "You learned to hit quickly, live lean, and leave before reprisals landed.",
            stat_modifiers: vec![
                "Shock command: +2".to_string(),
                "Resolve: +6".to_string(),
                "Supply reserve: -5".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Vanguard strike specialists.".to_string(),
                "High-risk recruits are promoted ahead of cautious candidates.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Vanguard,
                game_core::PersonalityArchetype::Guardian,
            ],
            player_resolve_delta: 6.0,
            player_loyalty_delta: -2.0,
            player_stress_delta: 4.0,
            player_fatigue_delta: 1.0,
            party_speed_delta: 0.1,
            party_supply_delta: -5.0,
        },
        BackstorySelectionChoice {
            id: "scribe-archivist",
            name: "Scribe",
            summary: "You preserved contested records, turning forgotten ledgers into strategic truth.",
            stat_modifiers: vec![
                "Intel synthesis: +2".to_string(),
                "Diplomatic context: +1".to_string(),
                "Party speed: -0.05".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Tactician analysts.".to_string(),
                "Methodical candidates with low volatility are prioritized.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Tactician,
                game_core::PersonalityArchetype::Guardian,
            ],
            player_resolve_delta: 1.0,
            player_loyalty_delta: 3.0,
            player_stress_delta: -1.0,
            player_fatigue_delta: 0.0,
            party_speed_delta: -0.05,
            party_supply_delta: 5.0,
        },
        BackstorySelectionChoice {
            id: "temple-ward",
            name: "Temple Ward",
            summary: "You stood watch through plague winters, escorting civilians between shrine keeps.",
            stat_modifiers: vec![
                "Defensive command: +2".to_string(),
                "Loyalty anchor: +4".to_string(),
                "Supply reserve: +6".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Guardian protectors.".to_string(),
                "Steadfast recruits arrive more often than opportunists.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Guardian,
                game_core::PersonalityArchetype::Tactician,
            ],
            player_resolve_delta: 3.0,
            player_loyalty_delta: 4.0,
            player_stress_delta: -2.0,
            player_fatigue_delta: -1.0,
            party_speed_delta: 0.0,
            party_supply_delta: 6.0,
        },
        BackstorySelectionChoice {
            id: "forgebound-engineer",
            name: "Forgebound",
            summary: "You rebuilt siege engines from scrap and learned where every wall is weakest.",
            stat_modifiers: vec![
                "Siege insight: +2".to_string(),
                "Shock command: +1".to_string(),
                "Supply reserve: +4".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: balances Vanguard attackers and Tactician planners."
                    .to_string(),
                "Adaptive candidates with mixed profiles are promoted.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Vanguard,
                game_core::PersonalityArchetype::Tactician,
            ],
            player_resolve_delta: 4.0,
            player_loyalty_delta: 0.0,
            player_stress_delta: 1.0,
            player_fatigue_delta: 0.0,
            party_speed_delta: 0.05,
            party_supply_delta: 4.0,
        },
        BackstorySelectionChoice {
            id: "river-smuggler",
            name: "River Smuggler",
            summary: "You ran contraband through flood channels and bribed checkpoints before dawn.",
            stat_modifiers: vec![
                "Evasion routes: +2".to_string(),
                "Party speed: +0.2".to_string(),
                "Loyalty: -2".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Vanguard risk-takers.".to_string(),
                "Unorthodox recruits surface more frequently in the pool.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Vanguard,
                game_core::PersonalityArchetype::Guardian,
            ],
            player_resolve_delta: 3.0,
            player_loyalty_delta: -2.0,
            player_stress_delta: 2.0,
            player_fatigue_delta: 1.0,
            party_speed_delta: 0.2,
            party_supply_delta: -3.0,
        },
        BackstorySelectionChoice {
            id: "oathbreaker-turned-warden",
            name: "Turned Warden",
            summary: "After betraying one banner, you spent years earning trust with hard service.",
            stat_modifiers: vec![
                "Resolve: +5".to_string(),
                "Stress: +2".to_string(),
                "Loyalty repair: +2".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: mixes Guardian steadiness with Vanguard aggression."
                    .to_string(),
                "Second-chance candidates appear earlier in the roster.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Guardian,
                game_core::PersonalityArchetype::Vanguard,
            ],
            player_resolve_delta: 5.0,
            player_loyalty_delta: 2.0,
            player_stress_delta: 2.0,
            player_fatigue_delta: 0.0,
            party_speed_delta: 0.05,
            party_supply_delta: 0.0,
        },
        BackstorySelectionChoice {
            id: "stormwatch-navigator",
            name: "Stormwatch Navigator",
            summary: "You crossed lightning coasts by reading cloudfall and steering around war fleets.",
            stat_modifiers: vec![
                "Route awareness: +2".to_string(),
                "Party speed: +0.12".to_string(),
                "Stress: -1".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Tactician scouts and Vanguard outriders.".to_string(),
                "Mobile expedition specialists are prioritized.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Tactician,
                game_core::PersonalityArchetype::Vanguard,
            ],
            player_resolve_delta: 2.0,
            player_loyalty_delta: 1.0,
            player_stress_delta: -1.0,
            player_fatigue_delta: -1.0,
            party_speed_delta: 0.12,
            party_supply_delta: 2.0,
        },
        BackstorySelectionChoice {
            id: "hearthguard-medic",
            name: "Hearthguard Medic",
            summary: "You held field wards together with poultices, triage discipline, and stubborn calm.",
            stat_modifiers: vec![
                "Crisis endurance: +2".to_string(),
                "Loyalty: +4".to_string(),
                "Stress: -3".to_string(),
            ],
            recruit_bias_modifiers: vec![
                "Recruit generation bias: favors Guardian stability and Tactician planning.".to_string(),
                "Support-oriented recruits appear more often.".to_string(),
            ],
            preferred_recruit_archetypes: vec![
                game_core::PersonalityArchetype::Guardian,
                game_core::PersonalityArchetype::Tactician,
            ],
            player_resolve_delta: 2.0,
            player_loyalty_delta: 4.0,
            player_stress_delta: -3.0,
            player_fatigue_delta: -1.0,
            party_speed_delta: 0.0,
            party_supply_delta: 7.0,
        },
    ]
}

pub fn apply_backstory_effects(
    choice: &BackstorySelectionChoice,
    character_creation: &mut CharacterCreationState,
    roster: &mut game_core::CampaignRoster,
    parties: &mut game_core::CampaignParties,
) {
    character_creation.selected_backstory_id = Some(choice.id.to_string());
    character_creation.stat_modifiers = choice.stat_modifiers.clone();
    character_creation.recruit_bias_modifiers = choice.recruit_bias_modifiers.clone();

    let player_id = roster
        .player_hero_id
        .or_else(|| roster.heroes.first().map(|h| h.id));
    if let Some(player_id) = player_id {
        if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == player_id) {
            hero.resolve = (hero.resolve + choice.player_resolve_delta).clamp(0.0, 100.0);
            hero.loyalty = (hero.loyalty + choice.player_loyalty_delta).clamp(0.0, 100.0);
            hero.stress = (hero.stress + choice.player_stress_delta).clamp(0.0, 100.0);
            hero.fatigue = (hero.fatigue + choice.player_fatigue_delta).clamp(0.0, 100.0);
        }
    }

    if let Some(player_party) = parties.parties.iter_mut().find(|p| p.is_player_controlled) {
        player_party.speed = (player_party.speed + choice.party_speed_delta).clamp(0.4, 2.2);
        player_party.supply = (player_party.supply + choice.party_supply_delta).clamp(0.0, 150.0);
    }

    roster.recruit_pool.sort_by_key(|candidate| {
        let preference = choice
            .preferred_recruit_archetypes
            .iter()
            .position(|a| *a == candidate.archetype)
            .unwrap_or(choice.preferred_recruit_archetypes.len());
        (preference, candidate.id)
    });
}

pub fn confirm_backstory_selection(
    hub_ui: &mut HubUiState,
    character_creation: &mut CharacterCreationState,
    creation_ui: &mut CharacterCreationUiState,
    roster: &mut game_core::CampaignRoster,
    parties: &mut game_core::CampaignParties,
    overworld: &game_core::OverworldMap,
) -> bool {
    if character_creation.selected_faction_id.is_none() {
        creation_ui.status =
            "Confirm a faction before selecting a backstory archetype.".to_string();
        return false;
    }
    let Some(selected_id) = character_creation.selected_backstory_id.as_deref() else {
        creation_ui.status =
            "Select a backstory archetype before entering the overworld.".to_string();
        return false;
    };
    let choices = build_backstory_selection_choices();
    let Some(choice) = choices.iter().find(|entry| entry.id == selected_id) else {
        creation_ui.status =
            "Selected backstory is no longer valid. Choose another archetype and continue."
                .to_string();
        character_creation.selected_backstory_id = None;
        return false;
    };

    apply_backstory_effects(choice, character_creation, roster, parties);
    character_creation.is_confirmed = true;
    parties.notice = format!(
        "Backstory '{}' confirmed. Campaign seed {} retained.",
        choice.name, overworld.map_seed
    );
    creation_ui.status = format!(
        "Backstory '{}' confirmed. Preparing cinematic with seed {}.",
        choice.name, overworld.map_seed
    );
    hub_ui.screen = HubScreen::BackstoryCinematic;
    true
}
