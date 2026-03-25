//! Guild reputation tier system.
//!
//! Checks every 500 ticks whether the guild's reputation has crossed a tier
//! boundary, applying bonuses on tier-up and removing them on tier-down.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: every 500 ticks.
const TICK_CADENCE: u64 = 17;

/// Evaluate guild reputation and update the tier if it has changed.
pub fn tick_guild_tiers(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TICK_CADENCE != 0 {
        return;
    }

    let old_tier = state.guild.guild_tier;
    let new_tier = GuildTier::from_reputation(state.guild.reputation);

    if old_tier == new_tier {
        return;
    }

    let tier_up = new_tier > old_tier;

    // Apply tier-specific effects
    if tier_up {
        apply_tier_up_bonuses(state, old_tier, new_tier);
    } else {
        apply_tier_down_penalties(state, old_tier, new_tier);
    }

    let description = if tier_up {
        tier_up_description(old_tier, new_tier)
    } else {
        tier_down_description(old_tier, new_tier)
    };

    state.guild.guild_tier = new_tier;

    events.push(WorldEvent::GuildTierChanged {
        old_tier: old_tier.name().to_string(),
        new_tier: new_tier.name().to_string(),
        description,
    });
}

/// Apply bonuses when the guild advances to a higher tier.
fn apply_tier_up_bonuses(
    state: &mut CampaignState,
    old_tier: GuildTier,
    new_tier: GuildTier,
) {
    // Bronze -> Silver: +1 max party capacity (diplomatic actions are gated in valid_actions)
    if old_tier < GuildTier::Silver && new_tier >= GuildTier::Silver {
        state.guild.active_quest_capacity += 1;
    }

    // Silver -> Gold: better recruit quality is handled by checking tier at recruitment time
    // Quest reward bonus is handled via GuildTier::quest_reward_multiplier()
    // Alliance actions are gated in valid_actions()

    // Gold -> Legendary: combat power bonus is handled via GuildTier::combat_power_multiplier()
    // Royal quests are handled at quest generation time
}

/// Remove bonuses when the guild drops to a lower tier.
fn apply_tier_down_penalties(
    state: &mut CampaignState,
    old_tier: GuildTier,
    new_tier: GuildTier,
) {
    // Losing Silver: remove the +1 party capacity
    if old_tier >= GuildTier::Silver && new_tier < GuildTier::Silver {
        state.guild.active_quest_capacity = state.guild.active_quest_capacity.saturating_sub(1);
    }
}

fn tier_up_description(old_tier: GuildTier, new_tier: GuildTier) -> String {
    let unlock_msg = match new_tier {
        GuildTier::Silver => {
            "Diplomatic trade actions unlocked. Guild can now manage one additional quest."
        }
        GuildTier::Gold => {
            "Alliance proposals unlocked. Quest rewards increased by 10%. Higher-quality recruits available."
        }
        GuildTier::Legendary => {
            "Royal quests available. Combat power increased by 20%. Faction leaders visit the guild."
        }
        GuildTier::Bronze => "Starting tier.", // shouldn't happen on tier-up
    };
    format!(
        "Guild promoted from {} to {}! {}",
        old_tier.name(),
        new_tier.name(),
        unlock_msg
    )
}

fn tier_down_description(old_tier: GuildTier, new_tier: GuildTier) -> String {
    let loss_msg = match new_tier {
        GuildTier::Bronze => "Lost access to trade agreements. Quest capacity reduced.",
        GuildTier::Silver => "Lost access to alliances and quest reward bonus.",
        GuildTier::Gold => "Lost access to royal quests and combat power bonus.",
        GuildTier::Legendary => "No change.", // shouldn't happen on tier-down
    };
    format!(
        "Warning: Guild demoted from {} to {}. {}",
        old_tier.name(),
        new_tier.name(),
        loss_msg
    )
}
