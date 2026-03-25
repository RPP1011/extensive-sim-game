//! Named NPC reputation system — every 300 ticks (~30s).
//!
//! Each named NPC has an individual reputation score (-100 to 100) that drifts
//! toward neutral over time. Quests completed in an NPC's region boost their
//! reputation. At threshold levels, NPCs unlock role-specific services.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::{
    CampaignState, NamedNpc, NpcRole, QuestResult, Region,
};

/// Tick interval for reputation updates.
const REPUTATION_TICK_INTERVAL: u64 = 10;

/// Reputation drift rate per tick toward 0.
const DRIFT_RATE: f32 = 0.5;

/// Reputation gain when a quest is completed in the NPC's region.
const QUEST_COMPLETION_BONUS: f32 = 5.0;

// ---------------------------------------------------------------------------
// NPC name banks per role
// ---------------------------------------------------------------------------

const MERCHANT_NAMES: &[&str] = &[
    "Halvard the Shrewd", "Elara Coinweaver", "Doran Tallmark",
    "Yuki Silkpurse", "Brannock Fairscale", "Nessa Goldthread",
];

const NOBLE_NAMES: &[&str] = &[
    "Lord Aldric Vane", "Lady Isolde Ashford", "Baron Cedric Whitmore",
    "Duchess Rowena Stormveil", "Count Aldwin Hale", "Lady Margaux DuPont",
];

const INFORMANT_NAMES: &[&str] = &[
    "Whisper", "Giselle Shade", "Ratkin", "Sable the Listener",
    "Corin the Quiet", "Fenna Duskwatch",
];

const HEALER_NAMES: &[&str] = &[
    "Sister Maren", "Healwright Jorin", "Sage Amara",
    "Brother Aldous", "Willowbark", "Apothecary Lena",
];

const BLACKSMITH_NAMES: &[&str] = &[
    "Torgun Ironhand", "Hilda Forgespike", "Korvak Steeltemper",
    "Freya Anvilstrike", "Dunstan Hammervane", "Brynna Emberveil",
];

const SCHOLAR_NAMES: &[&str] = &[
    "Archivist Malen", "Lorekeeper Thessa", "Sage Orwin",
    "Chronicler Fen", "Scribe Aldara", "Lexicant Roth",
];

fn name_bank_for_role(role: NpcRole) -> &'static [&'static str] {
    match role {
        NpcRole::Merchant => MERCHANT_NAMES,
        NpcRole::Noble => NOBLE_NAMES,
        NpcRole::Informant => INFORMANT_NAMES,
        NpcRole::Healer => HEALER_NAMES,
        NpcRole::Blacksmith => BLACKSMITH_NAMES,
        NpcRole::Scholar => SCHOLAR_NAMES,
    }
}

// ---------------------------------------------------------------------------
// NPC generation at campaign init
// ---------------------------------------------------------------------------

/// All six roles in a fixed order for deterministic distribution.
const ALL_ROLES: [NpcRole; 6] = [
    NpcRole::Merchant,
    NpcRole::Noble,
    NpcRole::Informant,
    NpcRole::Healer,
    NpcRole::Blacksmith,
    NpcRole::Scholar,
];

/// Generate 4-6 named NPCs distributed across regions (1-2 per region).
///
/// Uses a deterministic LCG seeded from the campaign seed so results are
/// reproducible across runs.
pub fn generate_initial_npcs(regions: &[Region], seed: u64) -> Vec<NamedNpc> {
    let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(7);
    let mut npcs = Vec::new();

    if regions.is_empty() {
        return npcs;
    }

    // Determine NPC count: clamp to [4, 6], at most 2 per region.
    let max_npcs = (regions.len() * 2).min(6);
    let npc_count = max_npcs.max(4).min(6);

    // Shuffle role order using the RNG for variety.
    let mut roles: Vec<NpcRole> = ALL_ROLES.to_vec();
    // Fisher-Yates shuffle
    for i in (1..roles.len()).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        roles.swap(i, j);
    }

    for i in 0..npc_count {
        let role = roles[i % roles.len()];
        let region_id = i % regions.len();

        // Pick a name deterministically from the role's bank.
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bank = name_bank_for_role(role);
        let name_idx = (rng >> 33) as usize % bank.len();

        npcs.push(NamedNpc {
            id: (i + 1) as u32,
            name: bank[name_idx].to_string(),
            role,
            region_id,
            reputation: 0.0,
            services_unlocked: Vec::new(),
        });
    }

    npcs
}

// ---------------------------------------------------------------------------
// Service thresholds
// ---------------------------------------------------------------------------

/// Returns the (threshold, service_name) pairs for a given NPC role.
fn service_thresholds(role: NpcRole) -> &'static [(f32, &'static str)] {
    match role {
        NpcRole::Merchant => &[
            (30.0, "discount_prices"),   // -10% prices
            (60.0, "rare_item_access"),  // access to rare items
        ],
        NpcRole::Noble => &[
            (40.0, "faction_influence"),  // +5 faction relation bonus
            (70.0, "royal_quest_access"), // exclusive high-tier quests
        ],
        NpcRole::Informant => &[
            (20.0, "free_intel"),           // free intel reports
            (50.0, "early_crisis_warning"), // advance notice of crises
        ],
        NpcRole::Healer => &[
            (30.0, "fast_recovery"),  // faster injury recovery
            (60.0, "resurrection"),   // prevent adventurer death once
        ],
        NpcRole::Blacksmith => &[
            (40.0, "equipment_upgrades"), // upgrade existing equipment
            (70.0, "enchanting"),         // magical enchantments
        ],
        NpcRole::Scholar => &[
            (30.0, "xp_bonus"),              // +10% XP
            (60.0, "history_tag_insights"),   // reveal hidden history tags
        ],
    }
}

// ---------------------------------------------------------------------------
// Tick function
// ---------------------------------------------------------------------------

pub fn tick_npc_reputation(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % REPUTATION_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Count quest completions by region since last tick window.
    // We look at completed_quests whose completion time falls within the last
    // REPUTATION_TICK_INTERVAL ticks.
    let window_start_ms = state
        .elapsed_ms
        .saturating_sub(REPUTATION_TICK_INTERVAL * 100);

    // Build a vec of region IDs where quests were completed (Victory only).
    let mut completed_region_ids: Vec<usize> = Vec::new();
    for cq in &state.completed_quests {
        if cq.result == QuestResult::Victory && cq.completed_at_ms > window_start_ms {
            // Determine region from quest's source_area_id by looking up the
            // original request. CompletedQuest doesn't store region directly,
            // so we use the source_area_id if available, otherwise map
            // the faction owner back to their region.
            // We check active_quests first (may still be there during cleanup),
            // but completed quests no longer appear there. Instead we rely on
            // the reward's relation_faction_id as a proxy for region.
            if let Some(faction_id) = cq.reward_applied.relation_faction_id {
                // Find the first region owned by that faction.
                for region in &state.overworld.regions {
                    if region.owner_faction_id == faction_id {
                        completed_region_ids.push(region.id);
                        break;
                    }
                }
            }
        }
    }

    // Process each named NPC.
    let npc_count = state.named_npcs.len();
    for i in 0..npc_count {
        let old_rep = state.named_npcs[i].reputation;
        let region_id = state.named_npcs[i].region_id;

        // 1. Drift toward 0.
        let mut new_rep = old_rep;
        if new_rep > 0.0 {
            new_rep = (new_rep - DRIFT_RATE).max(0.0);
        } else if new_rep < 0.0 {
            new_rep = (new_rep + DRIFT_RATE).min(0.0);
        }

        // 2. Quest completion bonus in NPC's region.
        let region_completions = completed_region_ids
            .iter()
            .filter(|&&rid| rid == region_id)
            .count();
        new_rep += region_completions as f32 * QUEST_COMPLETION_BONUS;

        // Clamp to [-100, 100].
        new_rep = new_rep.clamp(-100.0, 100.0);

        state.named_npcs[i].reputation = new_rep;

        // Emit event if reputation changed meaningfully (> 0.1 delta).
        if (new_rep - old_rep).abs() > 0.1 {
            events.push(WorldEvent::NpcReputationChanged {
                npc_name: state.named_npcs[i].name.clone(),
                old: old_rep,
                new: new_rep,
            });
        }

        // 3. Check service unlock thresholds.
        let role = state.named_npcs[i].role;
        for &(threshold, service_name) in service_thresholds(role) {
            let service_str = service_name.to_string();
            if new_rep >= threshold
                && !state.named_npcs[i]
                    .services_unlocked
                    .contains(&service_str)
            {
                state.named_npcs[i]
                    .services_unlocked
                    .push(service_str.clone());
                events.push(WorldEvent::NpcServiceUnlocked {
                    npc_name: state.named_npcs[i].name.clone(),
                    service: service_str,
                });
            }
            // Remove service if reputation dropped below threshold.
            if new_rep < threshold {
                state.named_npcs[i]
                    .services_unlocked
                    .retain(|s| s != service_name);
            }
        }
    }
}
