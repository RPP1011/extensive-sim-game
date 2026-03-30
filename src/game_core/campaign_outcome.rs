use bevy::prelude::*;

use super::generation::*;
use super::overworld_types::FlashpointState;
use super::roster_types::*;

// ─────────────────────────────────────────────────────────────────────────────
// Phase-2 campaign consequence helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Applies the outcome of a completed mission to every hero that participated.
///
/// * **Victory** -- small injury/stress relief, loyalty gain, generous XP.
/// * **Defeat**  -- injury/stress penalty, loyalty loss, token XP.
///
/// After updating stats each hero is checked for level-up, desertion, and
/// incapacitation.
pub fn apply_mission_result_to_roster(
    roster: &mut CampaignRoster,
    outcome: crate::mission::sim_bridge::MissionOutcome,
    difficulty: u32,
    heroes_in_mission: &[u32],
) {
    use crate::mission::sim_bridge::MissionOutcome;

    for &hero_id in heroes_in_mission {
        let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == hero_id) else {
            continue;
        };

        match outcome {
            MissionOutcome::Victory => {
                hero.injury  = (hero.injury  - 2.0).max(0.0);
                hero.stress  = (hero.stress  - 5.0).max(0.0);
                hero.loyalty = (hero.loyalty + 3.0).min(100.0);
                hero.xp     += 20 + difficulty * 5;
            }
            MissionOutcome::Defeat => {
                hero.injury  = (hero.injury  + 8.0 + difficulty as f32 * 0.5).min(100.0);
                hero.stress  = (hero.stress  + 12.0).min(100.0);
                hero.loyalty = (hero.loyalty - 4.0).max(0.0);
                hero.xp     += 5;
            }
        }

        check_level_up(hero);

        // Desertion: low loyalty combined with high stress.
        if hero.loyalty < 15.0 && hero.stress > 65.0 {
            hero.deserter = true;
        }

        // Incapacitation: severe injury removes the hero from active duty.
        if hero.injury > 90.0 {
            hero.active = false;
        }
    }
}

/// Checks whether `hero` has accumulated enough XP to level up and applies
/// one seeded stat bonus per level gained.
///
/// Level threshold: `xp_required = level * level * 50`
pub fn check_level_up(hero: &mut HeroCompanion) {
    loop {
        let xp_required = hero.level * hero.level * 50;
        if hero.xp < xp_required {
            break;
        }
        hero.level += 1;
        // Pick one of four bonuses deterministically from hero id + new level.
        let bonus_roll =
            splitmix64((hero.id as u64).wrapping_mul(97).wrapping_add(hero.level as u64)) % 4;
        match bonus_roll {
            0 => {
                // HP bonus -- wired into the sim when the hero unit is
                // constructed from HeroCompanion (EquipmentItem.hp_bonus path).
            }
            1 => {
                hero.loyalty = (hero.loyalty + 5.0).min(100.0);
            }
            2 => {
                hero.resolve += 5.0;
            }
            _ => {
                hero.fatigue = (hero.fatigue - 10.0).max(0.0);
            }
        }
    }
}

/// Generates a deterministic loot drop for a completed mission.
///
/// Returns `None` roughly 40 % of the time based on `mission_seed`.
/// Otherwise returns an [`EquipmentItem`] whose stats scale lightly with
/// `difficulty`.
pub fn generate_loot_drop(mission_seed: u64, difficulty: u32) -> Option<EquipmentItem> {
    // 40 % chance of no drop.
    if rand01(mission_seed, 0xDEAD_BEEF_u64) < 0.40 {
        return None;
    }

    const NAMES: [&str; 12] = [
        "Iron Blade",
        "Scout Boots",
        "Warden Shield",
        "Mystic Amulet",
        "Ranger Cloak",
        "Steel Gauntlets",
        "Battle Axe",
        "Shadow Dagger",
        "Plate Cuirass",
        "Swiftfoot Wraps",
        "Enchanted Talisman",
        "Veteran's Helm",
    ];

    let name_idx =
        (splitmix64(mission_seed ^ 0x1234_5678_u64) % NAMES.len() as u64) as usize;
    let name = NAMES[name_idx].to_string();

    // Rarity: Rare when the rarity roll exceeds 0.8 (roughly 20 % of drops).
    let rarity = if rand01(mission_seed, 0xCAFE_BABE_u64) > 0.8 {
        ItemRarity::Rare
    } else {
        ItemRarity::Standard
    };

    let diff_f = difficulty as f32;
    let attack_bonus  = (rand01(mission_seed, 0x1111_u64) * (2.0 + diff_f * 0.5)) as i32;
    let hp_bonus      = (rand01(mission_seed, 0x2222_u64) * (4.0 + diff_f * 1.0)) as i32;
    let speed_bonus   = rand01(mission_seed, 0x3333_u64) * (0.05 + diff_f * 0.01);
    // cooldown_mult in [0.85, 1.0]: lower means faster attacks.
    let cooldown_mult = 1.0 - rand01(mission_seed, 0x4444_u64) * (0.15 + diff_f * 0.02);

    Some(EquipmentItem {
        name,
        rarity,
        attack_bonus,
        hp_bonus,
        speed_bonus,
        cooldown_mult,
    })
}

// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a completed campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CampaignOutcome {
    Victory,
    Defeat,
}

/// Returns the count of active, non-deserter heroes in the roster.
pub fn active_hero_count(roster: &CampaignRoster) -> usize {
    roster
        .heroes
        .iter()
        .filter(|h| h.active && !h.deserter)
        .count()
}

/// Checks whether the campaign has reached a win or lose condition.
pub fn check_campaign_outcome(
    flashpoint: &FlashpointState,
    roster: &CampaignRoster,
) -> Option<CampaignOutcome> {
    if active_hero_count(roster) == 0 {
        return Some(CampaignOutcome::Defeat);
    }
    if !flashpoint.chains.is_empty() && flashpoint.chains.iter().all(|c| c.completed) {
        return Some(CampaignOutcome::Victory);
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Hub UI state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubScreen {
    StartMenu,
    CharacterCreationFaction,
    CharacterCreationBackstory,
    BackstoryCinematic,
    GuildManagement,
    Overworld,
    OverworldMap,
    RegionView,
    LocalEagleEyeIntro,
    MissionExecution,
    ReplayViewer,
}

impl HubScreen {
    /// Whether this screen shows the shared left side panel.
    pub fn shows_side_panel(&self) -> bool {
        !matches!(
            self,
            HubScreen::CharacterCreationFaction
                | HubScreen::CharacterCreationBackstory
                | HubScreen::BackstoryCinematic
        )
    }
}

#[derive(Resource)]
pub struct HubUiState {
    pub screen: HubScreen,
    pub show_credits: bool,
    pub request_quit: bool,
    pub request_new_campaign: bool,
    pub request_continue_campaign: bool,
}

impl Default for HubUiState {
    fn default() -> Self {
        Self {
            screen: HubScreen::StartMenu,
            show_credits: false,
            request_quit: false,
            request_new_campaign: false,
            request_continue_campaign: false,
        }
    }
}
