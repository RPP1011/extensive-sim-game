//! Unified campaign effect dispatch.
//!
//! Dispatches `Effect` variants from the unified DSL for campaign-layer execution.
//! Combat effects (Damage, Heal, Stun, etc.) are ignored here -- they're handled by
//! the tactical sim's `apply_effect()`. Campaign effects modify faction, region,
//! guild, and adventurer state.

use tactical_sim::effects::Effect;

use super::state::CampaignState;

/// Target for a campaign effect.
#[allow(dead_code)]
pub enum CampaignTarget {
    Adventurer(u32),
    Party(u32),
    Faction(usize),
    Region(usize),
    Location(usize),
    Guild,
    Market,
    Global,
}

/// Apply a campaign effect. Returns true if the effect was handled.
#[allow(unused_variables)]
pub fn campaign_apply_effect(
    state: &mut CampaignState,
    effect: &Effect,
    caster_id: u32,
    _target: &CampaignTarget,
) -> bool {
    match effect {
        // --- Economy ---
        Effect::CornerMarket {
            commodity,
            duration_ticks,
        } => {
            // Mark commodity as monopolized for duration.
            // Placeholder -- actual campaign state tracking would need a
            // market_monopolies field.
            true
        }
        Effect::ForgeTradeRoute {
            income_per_tick,
            duration_ticks,
        } => {
            // Create a trade route with given income.
            true
        }
        Effect::Appraise => {
            // Reveal true item values for the caster.
            true
        }
        Effect::GoldenTouch { duration_ticks } => {
            // Double loot for duration.
            true
        }
        Effect::TradeEmbargo { duration_ticks } => {
            // Block faction trade.
            true
        }
        Effect::SilverTongue => {
            // Always win negotiations.
            true
        }
        Effect::MarketMaker { duration_ticks } => true,
        Effect::TradeEmpire { income_per_tick } => true,

        // --- Diplomacy ---
        Effect::DemandAudience => true,
        Effect::CeasefireDeclaration { duration_ticks } => true,
        Effect::Destabilize {
            instability,
            duration_ticks,
        } => {
            // Increase faction instability.
            true
        }
        Effect::BrokerAlliance { duration_ticks } => true,
        Effect::SubvertLoyalty => true,
        Effect::TreatyBreaker => true,
        Effect::ShatterAlliance => true,
        Effect::Forgery => true,

        // --- Information ---
        Effect::Reveal { count } => true,
        Effect::PropheticVision { count } => true,
        Effect::BeastLore => true,
        Effect::ReadTheRoom => true,
        Effect::AllSeeingEye => true,
        Effect::Decipher => true,
        Effect::TrapSense => true,
        Effect::SapperEye => true,
        Effect::IntelGathering => true,

        // --- Leadership ---
        Effect::Rally { morale_restore } => {
            // Restore party morale.
            true
        }
        Effect::RallyingCry { morale_restore } => true,
        Effect::Inspire {
            morale_boost,
            duration_ticks,
        } => true,
        Effect::FieldCommand { duration_ticks } => true,
        Effect::CoordinatedStrike => true,
        Effect::WarCry { duration_ticks } => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.taunt_remaining_ticks = *duration_ticks;
            }
            true
        }

        // --- Stealth / Movement ---
        Effect::GhostWalk { duration_ticks } => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.stealth_remaining_ticks = *duration_ticks;
            }
            true
        }
        Effect::ShadowStep { duration_ticks } => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.stealth_remaining_ticks = *duration_ticks;
            }
            true
        }
        Effect::SilentMovement => true,
        Effect::HiddenCamp { duration_ticks } => true,
        Effect::Vanish => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.vanished = true;
            }
            true
        }
        Effect::Distraction { duration_ticks } => true,

        // --- Territory ---
        Effect::ClaimTerritory => true,
        Effect::Fortify { duration_ticks } => true,
        Effect::Sanctuary { duration_ticks } => true,
        Effect::PlagueWard { duration_ticks } => true,
        Effect::SafeHouse { duration_ticks } => true,
        Effect::ClaimByRight => true,

        // --- Supernatural / Body ---
        Effect::BloodOath { stat_bonus } => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.combat_power_bonus = *stat_bonus;
            }
            true
        }
        Effect::Unbreakable => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.unbreakable_active = true;
            }
            true
        }
        Effect::LifeEternal => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.immortal = true;
            }
            true
        }
        Effect::Purify => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.diseases.clear();
            }
            true
        }
        Effect::NameTheNameless => true,
        Effect::ForbiddenKnowledge => true,
        Effect::ImmortalMoment => true,
        Effect::ForgeArtifact => true,
        Effect::RewriteTheRecord => true,

        // --- Passive Skill-State ---
        Effect::FieldTriage {
            heal_rate_multiplier,
        } => true,
        Effect::InspiringPresence { morale_boost } => true,
        Effect::BattleInstinct => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.ambush_immunity = true;
            }
            true
        }
        Effect::QuickStudy => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.xp_multiplier = 2.0;
            }
            true
        }
        Effect::Forage { supply_per_tick } => true,
        Effect::TrackPrey => {
            if let Some(adv) = state
                .adventurers
                .iter_mut()
                .find(|a| a.id == caster_id)
            {
                adv.skill_state.tracking_active = true;
            }
            true
        }
        Effect::FieldRepair => true,
        Effect::StabilizeAlly => true,

        // --- Higher Tier ---
        Effect::Disinformation { duration_ticks } => true,
        Effect::AcceleratedStudy { duration_ticks } => true,
        Effect::TakeTheBlow { duration_ticks } => true,
        Effect::HoldTheLine => true,
        Effect::MasterworkCraft => true,
        Effect::MasterArmorer => true,

        // --- Legendary / Mythic ---
        Effect::LivingLegend => true,
        Effect::RewriteHistory => true,
        Effect::TheLastWord => true,
        Effect::WealthOfNations => true,
        Effect::Omniscience => true,

        // Combat effects -- not handled in campaign dispatch.
        _ => false,
    }
}
