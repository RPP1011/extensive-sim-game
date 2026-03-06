use crate::game_core::{DiplomacyState, OverworldMap};

use super::types::*;

// ---------------------------------------------------------------------------
// LCG helper
// ---------------------------------------------------------------------------

fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ---------------------------------------------------------------------------
// Event generation
// ---------------------------------------------------------------------------

const MERCHANT_ITEMS: [&str; 6] = [
    "a vial of healing salve",
    "a worn but serviceable longsword",
    "a bundle of rope and climbing tools",
    "a crate of preserved rations",
    "a set of scouting maps",
    "a flask of alchemist's fire",
];

const RUMOUR_DESCRIPTIONS: [&str; 6] = [
    "has been secretly negotiating with a rival lord",
    "is rumoured to be gathering forces near the northern pass",
    "suffered a catastrophic internal coup last season",
    "has placed a bounty on deserters from their ranks",
    "is said to be seeking an alliance against the eastern coalition",
    "reportedly lost two commanders in a skirmish at the border fort",
];

pub fn generate_event(
    seed: u64,
    global_turn: u32,
    map: &OverworldMap,
    _diplomacy: &DiplomacyState,
) -> Option<PendingCampaignEvent> {
    let mut lcg = lcg_next(seed ^ global_turn as u64);

    if lcg % 3 != 0 {
        return None;
    }

    lcg = lcg_next(lcg);
    let kind_index = lcg % 8;

    let faction_count = map.factions.len().max(1);
    let region_count = map.regions.len().max(1);

    lcg = lcg_next(lcg);
    let faction_id = (lcg as usize) % faction_count;

    lcg = lcg_next(lcg);
    let region_id = (lcg as usize) % region_count;

    lcg = lcg_next(lcg);
    let small_num = lcg % 16;

    let kind = match kind_index {
        0 => {
            let supply_cost = 15 + (small_num % 16) as u32;
            let item_idx = (small_num as usize) % MERCHANT_ITEMS.len();
            CampaignEventKind::MerchantOffer {
                supply_cost,
                item_description: MERCHANT_ITEMS[item_idx].to_string(),
            }
        }
        1 => {
            let relation_boost = 5 + (small_num % 6) as i32;
            CampaignEventKind::DeserterIntel {
                faction_id,
                relation_boost,
                risk_stress: 8.0,
            }
        }
        2 => CampaignEventKind::PlagueScare {
            region_id,
            stress_penalty: 12.0,
        },
        3 => {
            let reward_supply = 20 + (small_num % 21) as u32;
            CampaignEventKind::RivalPartySpotted { region_id, reward_supply }
        }
        4 => CampaignEventKind::AllyRequest {
            faction_id,
            relation_reward: 8,
            supply_cost: 20,
        },
        5 => {
            let supply_reward = 25 + (small_num % 26) as u32;
            CampaignEventKind::AbandonedCache { supply_reward }
        }
        6 => {
            let turns_of_slow = 3 + (small_num % 4) as u32;
            CampaignEventKind::StormWarning { turns_of_slow }
        }
        _ => {
            let desc_idx = (small_num as usize) % RUMOUR_DESCRIPTIONS.len();
            CampaignEventKind::FactionRumour {
                faction_id,
                description: RUMOUR_DESCRIPTIONS[desc_idx].to_string(),
            }
        }
    };

    let (title, description) = event_text(&kind, map);

    Some(PendingCampaignEvent {
        id: 0,
        turn_generated: global_turn,
        kind,
        title,
        description,
        accepted: None,
    })
}

pub(crate) fn faction_name(map: &OverworldMap, faction_id: usize) -> String {
    map.factions
        .get(faction_id)
        .map(|f| f.name.clone())
        .unwrap_or_else(|| format!("Faction {}", faction_id))
}

pub(crate) fn region_name(map: &OverworldMap, region_id: usize) -> String {
    map.regions
        .get(region_id)
        .map(|r| r.name.clone())
        .unwrap_or_else(|| format!("Region {}", region_id))
}

fn event_text(kind: &CampaignEventKind, map: &OverworldMap) -> (String, String) {
    match kind {
        CampaignEventKind::MerchantOffer { supply_cost, item_description } => (
            "Wandering Merchant".to_string(),
            format!("A travelling merchant offers {}. Cost: {} supply.", item_description, supply_cost),
        ),
        CampaignEventKind::DeserterIntel { faction_id, relation_boost, .. } => (
            "Deserter with Intel".to_string(),
            format!(
                "A deserter from {} offers valuable information in exchange for protection. Relations +{}. Accepting carries some risk to morale.",
                faction_name(map, *faction_id), relation_boost
            ),
        ),
        CampaignEventKind::PlagueScare { region_id, stress_penalty } => (
            "Plague Scare".to_string(),
            format!(
                "Rumours of illness sweep through {}. Heroes stationed there suffer {:.0} stress from the panic.",
                region_name(map, *region_id), stress_penalty
            ),
        ),
        CampaignEventKind::RivalPartySpotted { region_id, reward_supply } => (
            "Rival Party Spotted".to_string(),
            format!(
                "Scouts report a rival party operating near {}. Confronting them may yield {} supply.",
                region_name(map, *region_id), reward_supply
            ),
        ),
        CampaignEventKind::AllyRequest { faction_id, relation_reward, supply_cost } => (
            "Ally Request".to_string(),
            format!(
                "{} requests material support. Accepting costs {} supply but improves relations by {}.",
                faction_name(map, *faction_id), supply_cost, relation_reward
            ),
        ),
        CampaignEventKind::FactionRumour { faction_id, description } => (
            "Faction Rumour".to_string(),
            format!("Word reaches the guild that {} {}.", faction_name(map, *faction_id), description),
        ),
        CampaignEventKind::AbandonedCache { supply_reward } => (
            "Abandoned Cache".to_string(),
            format!("Scouts discover an abandoned supply cache. Claiming it yields {} supply.", supply_reward),
        ),
        CampaignEventKind::StormWarning { turns_of_slow } => (
            "Storm Warning".to_string(),
            format!("A severe storm is forecast. Movement will be hampered for approximately {} turns.", turns_of_slow),
        ),
    }
}
