//! Applies tiered skill effects to campaign state (Phase 10).
//!
//! Each SkillEffect variant maps to a concrete mutation of CampaignState.
//! T1-T3 effects are fully implemented. T4+ effects emit events and have
//! simplified mechanics that bend existing systems.
//!
//! DEPRECATED: This module dispatches the legacy `SkillEffect` enum.
//! New code should use the unified `Effect` enum from `tactical_sim::effects`
//! with `campaign_apply_effect()` in `crate::unified_dispatch`.
//! All effects in this file have corresponding variants in the unified Effect enum.

use crate::actions::WorldEvent;
#[allow(deprecated)]
use crate::state::{
    AdventurerStatus, CampaignState, SkillCondition, SkillEffect,
};

/// Check whether a skill condition is currently met.
#[allow(deprecated)]
pub fn condition_met(
    state: &CampaignState,
    caster_id: u32,
    condition: &SkillCondition,
) -> bool {
    match condition {
        SkillCondition::Outnumbered => {
            // Check if the caster's party is outnumbered in any active battle.
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        // Consider outnumbered if party has fewer than 4 members
                        // and there's an active battle involving them.
                        return party.member_ids.len() < 4
                            && state.active_battles.iter().any(|b| b.party_id == pid);
                    }
                }
            }
            false
        }
        SkillCondition::AllyDying => {
            // Any party member below 10% morale (proxy for HP in campaign).
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        return party.member_ids.iter().any(|&mid| {
                            mid != caster_id
                                && state
                                    .adventurers
                                    .iter()
                                    .find(|a| a.id == mid)
                                    .map_or(false, |a| a.morale < 10.0)
                        });
                    }
                }
            }
            false
        }
        SkillCondition::CrisisActive => {
            // Any crisis system active (civil war, sleeping king, etc.)
            !state.overworld.active_crises.is_empty()
        }
        SkillCondition::AtWar => {
            // Any faction at war (check at_war_with list or skill modifier flag).
            state.factions.iter().any(|f| {
                !f.at_war_with.is_empty() || f.skill_modifiers.at_war
            })
        }
        SkillCondition::FactionHostile(faction_id) => {
            state
                .factions
                .get(*faction_id as usize)
                .map_or(false, |f| f.relationship_to_guild < -50.0)
        }
        SkillCondition::Alone => {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                adv.party_id.is_none()
            } else {
                false
            }
        }
        SkillCondition::NearDeath => {
            state
                .adventurers
                .iter()
                .find(|a| a.id == caster_id)
                .map_or(false, |a| a.morale < 20.0)
        }
    }
}

/// Apply a skill effect to the campaign state. Returns a description of what happened.
///
/// TODO: Migrate callers to use `unified_dispatch::campaign_apply_effect()` with
/// `tactical_sim::effects::Effect` instead of the legacy `SkillEffect` enum.
#[allow(deprecated)]
pub fn apply_skill_effect(
    state: &mut CampaignState,
    effect: &SkillEffect,
    caster_id: u32,
    events: &mut Vec<WorldEvent>,
) -> String {
    match effect {
        // =====================================================================
        // T1 — Always-on passives
        // =====================================================================
        SkillEffect::Appraise => {
            // Mark all items in caster's inventory as appraised.
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                for item in &mut adv.equipped_items {
                    item.appraised = true;
                }
            }
            "Items appraised — true values revealed.".into()
        }

        SkillEffect::FieldTriage { heal_rate_multiplier } => {
            // Accelerate wound healing for caster's party.
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        let member_ids: Vec<u32> = party.member_ids.clone();
                        for mid in member_ids {
                            if let Some(m) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                                for wound in &mut m.wounds {
                                    // Advance heal_progress by multiplier factor.
                                    wound.heal_progress +=
                                        wound.severity * (heal_rate_multiplier - 1.0);
                                }
                            }
                        }
                    }
                }
            }
            format!(
                "Field triage applied — wound healing {}x faster.",
                heal_rate_multiplier
            )
        }

        SkillEffect::ReadTheRoom => {
            // Reveal diplomatic stances of all factions.
            for faction in &mut state.factions {
                faction.skill_modifiers.stance_revealed = true;
            }
            "Diplomatic stances revealed for all factions.".into()
        }

        SkillEffect::ShadowStep { duration_ticks } => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.stealth_remaining_ticks = *duration_ticks;
            }
            format!("Shadow step active for {} ticks.", duration_ticks)
        }

        SkillEffect::InspiringPresence { morale_boost } => {
            // Boost morale for entire party.
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        let member_ids: Vec<u32> = party.member_ids.clone();
                        for mid in member_ids {
                            if let Some(m) =
                                state.adventurers.iter_mut().find(|a| a.id == mid)
                            {
                                m.morale = (m.morale + morale_boost).min(100.0);
                            }
                        }
                    }
                }
            }
            format!("Inspiring presence — party morale boosted by {}.", morale_boost)
        }

        SkillEffect::BeastLore => {
            // After one encounter with an enemy type, reveal their weaknesses.
            // Mark all encountered enemy types as analyzed.
            state.beast_lore_active = true;
            "Beast lore active — monster weaknesses revealed after first encounter.".into()
        }

        SkillEffect::BattleInstinct => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.ambush_immunity = true;
            }
            "Battle instinct — ambushes sensed before triggering.".into()
        }

        SkillEffect::QuickStudy => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.xp_multiplier = 2.0;
            }
            "Quick study — learning from events 2x faster.".into()
        }

        SkillEffect::Forage { supply_per_tick } => {
            // Add supplies to the guild passively.
            state.guild.supplies += supply_per_tick;
            format!("Foraging — gained {} supplies.", supply_per_tick)
        }

        SkillEffect::TrackPrey => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.tracking_active = true;
            }
            "Tracking prey — enemies visible through any terrain.".into()
        }

        SkillEffect::FieldRepair => {
            // Repair one durability point on all equipped items.
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                for item in &mut adv.equipped_items {
                    item.durability = (item.durability + 10.0).min(100.0);
                }
            }
            "Field repair — gear repaired without a workshop.".into()
        }

        SkillEffect::SilentMovement => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.stealth_remaining_ticks = adv.skill_state.stealth_remaining_ticks.max(50);
            }
            "Silent movement — moving past patrols undetected.".into()
        }

        SkillEffect::TrapSense => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.trap_detection = true;
            }
            "Trap sense — traps detected before triggering.".into()
        }

        SkillEffect::StabilizeAlly => {
            // Auto-stabilize any dying party members.
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        let member_ids: Vec<u32> = party.member_ids.clone();
                        for mid in member_ids {
                            if let Some(m) =
                                state.adventurers.iter_mut().find(|a| a.id == mid)
                            {
                                if m.morale < 5.0 && m.status != AdventurerStatus::Dead {
                                    m.morale = 10.0;
                                }
                            }
                        }
                    }
                }
            }
            "Stabilize ally — dying party members stabilized.".into()
        }

        SkillEffect::SapperEye => {
            // Reveal fortification weaknesses — just flag it.
            state.sapper_eye_active = true;
            "Sapper eye — fortification weaknesses assessed.".into()
        }

        // =====================================================================
        // T2-T3 — Active powers
        // =====================================================================
        SkillEffect::CornerTheMarket { commodity_index, duration_ticks } => {
            // Set a monopoly on the commodity.
            state.market_monopolies.push((*commodity_index, caster_id, *duration_ticks));
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Corner the Market".into(),
                effect_description: format!(
                    "Monopolized commodity {} for {} ticks.",
                    commodity_index, duration_ticks
                ),
            });
            format!(
                "Cornered the market on commodity {} for {} ticks.",
                commodity_index, duration_ticks
            )
        }

        SkillEffect::ForgeTradeRoute { duration_ticks, income_per_tick } => {
            state.active_trade_routes.push((caster_id, *income_per_tick, *duration_ticks));
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Forge Trade Route".into(),
                effect_description: format!(
                    "New trade route: {} gold/tick for {} ticks.",
                    income_per_tick, duration_ticks
                ),
            });
            format!(
                "Forged trade route — {} gold/tick for {} ticks.",
                income_per_tick, duration_ticks
            )
        }

        SkillEffect::DemandAudience => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Demand Audience".into(),
                effect_description: "Forced diplomatic meeting with any faction.".into(),
            });
            // Improve relation with lowest-relation faction.
            if let Some(f) = state
                .factions
                .iter_mut()
                .min_by(|a, b| a.relationship_to_guild.partial_cmp(&b.relationship_to_guild).unwrap_or(std::cmp::Ordering::Equal))
            {
                f.relationship_to_guild += 15.0;
            }
            "Demand audience — diplomatic meeting forced.".into()
        }

        SkillEffect::GhostWalk { duration_ticks } => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.stealth_remaining_ticks = *duration_ticks;
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Ghost Walk".into(),
                effect_description: format!("Hostile territory traversal for {} ticks.", duration_ticks),
            });
            format!("Ghost walk active for {} ticks.", duration_ticks)
        }

        SkillEffect::NameTheNameless => {
            // Grant guild-wide combat bonus flag.
            state.named_threat_bonus = true;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Name the Nameless".into(),
                effect_description: "Unknown threat identified — guild-wide combat bonus.".into(),
            });
            "Named the nameless — guild-wide combat bonus granted.".into()
        }

        SkillEffect::AcceleratedStudy { duration_ticks } => {
            state.accelerated_study_ticks = *duration_ticks;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Accelerated Study".into(),
                effect_description: format!("Research speed doubled for {} ticks.", duration_ticks),
            });
            format!("Accelerated study — research doubled for {} ticks.", duration_ticks)
        }

        SkillEffect::Disinformation { target_faction, duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Disinformation".into(),
                effect_description: format!(
                    "False intel planted in faction {}'s spy network for {} ticks.",
                    target_faction, duration_ticks
                ),
            });
            // Reduce target faction's espionage effectiveness.
            if let Some(f) = state.factions.get_mut(*target_faction as usize) {
                f.skill_modifiers.espionage_effectiveness = (f.skill_modifiers.espionage_effectiveness - 0.3).max(0.0);
            }
            format!("Disinformation planted in faction {}.", target_faction)
        }

        SkillEffect::SubvertLoyalty { target_faction } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Subvert Loyalty".into(),
                effect_description: format!(
                    "Recruiting from hostile faction {}.",
                    target_faction
                ),
            });
            // Boost guild recruitment quality.
            state.guild.recruitment_bonus += 0.1;
            format!("Subverted loyalty in faction {}.", target_faction)
        }

        SkillEffect::WarCry { duration_ticks } => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.taunt_remaining_ticks = *duration_ticks;
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "War Cry".into(),
                effect_description: format!("All enemies forced to target you for {} ticks.", duration_ticks),
            });
            format!("War cry — taunting for {} ticks.", duration_ticks)
        }

        SkillEffect::Purify => {
            // Cleanse all diseases from the caster.
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.diseases.clear();
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Purify".into(),
                effect_description: "All diseases and poisons cleansed.".into(),
            });
            "Purified — diseases and poisons cleansed.".into()
        }

        SkillEffect::SilverTongue => {
            // Improve current trade deal — boost guild gold slightly.
            state.guild.gold += 50.0;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Silver Tongue".into(),
                effect_description: "Better deal negotiated — +50 gold.".into(),
            });
            "Silver tongue — better deal negotiated.".into()
        }

        SkillEffect::Decipher => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Decipher".into(),
                effect_description: "Any language or inscription can be read.".into(),
            });
            "Decipher — all inscriptions readable.".into()
        }

        SkillEffect::HiddenCamp { duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Hidden Camp".into(),
                effect_description: format!("Concealed camp set up for {} ticks.", duration_ticks),
            });
            format!("Hidden camp established for {} ticks.", duration_ticks)
        }

        SkillEffect::MasterworkCraft => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Masterwork Craft".into(),
                effect_description: "Masterwork item crafted from lesser materials.".into(),
            });
            "Masterwork item crafted.".into()
        }

        SkillEffect::RallyingCry { morale_restore } => {
            // Restore morale to entire party.
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        let member_ids: Vec<u32> = party.member_ids.clone();
                        for mid in member_ids {
                            if let Some(m) =
                                state.adventurers.iter_mut().find(|a| a.id == mid)
                            {
                                m.morale = (m.morale + morale_restore).min(100.0);
                            }
                        }
                    }
                }
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Rallying Cry".into(),
                effect_description: format!("Party morale restored by {}.", morale_restore),
            });
            format!("Rallying cry — morale restored by {}.", morale_restore)
        }

        SkillEffect::Unbreakable => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.unbreakable_active = true;
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Unbreakable".into(),
                effect_description: "Will survive one killing blow with 1 HP.".into(),
            });
            "Unbreakable — will survive one killing blow.".into()
        }

        SkillEffect::Distraction { duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Distraction".into(),
                effect_description: format!("Distraction created for {} ticks.", duration_ticks),
            });
            format!("Distraction created for {} ticks.", duration_ticks)
        }

        SkillEffect::Forgery => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Forgery".into(),
                effect_description: "Documents forged to bypass faction restrictions.".into(),
            });
            "Forgery — faction restrictions bypassed.".into()
        }

        SkillEffect::CoordinatedStrike => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Coordinated Strike".into(),
                effect_description: "Two units attack simultaneously.".into(),
            });
            "Coordinated strike — simultaneous attack.".into()
        }

        SkillEffect::SafeHouse { region_id, duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Safe House".into(),
                effect_description: format!(
                    "Safe house established in region {} for {} ticks.",
                    region_id, duration_ticks
                ),
            });
            format!("Safe house in region {} for {} ticks.", region_id, duration_ticks)
        }

        // =====================================================================
        // T4-T5 — Powerful actives (simplified mechanics, emit events)
        // =====================================================================
        SkillEffect::CeasefireDeclaration => {
            // End all active battles.
            state.active_battles.clear();
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Ceasefire Declaration".into(),
                effect_description: "All active battles halted through diplomatic force.".into(),
            });
            "Ceasefire declared — all battles halted.".into()
        }

        SkillEffect::BloodOath { party_stat_bonus } => {
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        let member_ids: Vec<u32> = party.member_ids.clone();
                        for mid in member_ids {
                            if mid != caster_id {
                                if let Some(m) =
                                    state.adventurers.iter_mut().find(|a| a.id == mid)
                                {
                                    m.skill_state.combat_power_bonus += party_stat_bonus;
                                }
                            }
                        }
                    }
                }
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Blood Oath".into(),
                effect_description: format!(
                    "Life force bonded — party gains +{}% stats.",
                    (party_stat_bonus * 100.0) as i32
                ),
            });
            format!("Blood oath — party gains +{}% stats.", (party_stat_bonus * 100.0) as i32)
        }

        SkillEffect::WhisperOfSuccession { target_faction, instability } => {
            if let Some(f) = state.factions.get_mut(*target_faction as usize) {
                f.skill_modifiers.instability += instability;
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Whisper of Succession".into(),
                effect_description: format!(
                    "Faction {} destabilized by {}.",
                    target_faction, instability
                ),
            });
            format!("Whisper planted — faction {} destabilized.", target_faction)
        }

        SkillEffect::Sanctuary { region_id, duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Sanctuary".into(),
                effect_description: format!(
                    "Sanctuary zone in region {} for {} ticks.",
                    region_id, duration_ticks
                ),
            });
            format!("Sanctuary zone created in region {}.", region_id)
        }

        SkillEffect::PropheticVision { events_revealed } => {
            state.prophetic_visions_remaining += events_revealed;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Prophetic Vision".into(),
                effect_description: format!(
                    "{} upcoming events revealed.",
                    events_revealed
                ),
            });
            format!("Prophetic vision — {} events foreseen.", events_revealed)
        }

        SkillEffect::TreatyBreaker => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Treaty Breaker".into(),
                effect_description: "All factions forced to renegotiate treaties.".into(),
            });
            "Treaty breaker — all treaties renegotiated.".into()
        }

        SkillEffect::TakeTheBlow { target_adventurer, duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Take the Blow".into(),
                effect_description: format!(
                    "Redirecting damage from adventurer {} for {} ticks.",
                    target_adventurer, duration_ticks
                ),
            });
            format!(
                "Take the blow — absorbing damage for adventurer {}.",
                target_adventurer
            )
        }

        SkillEffect::HoldTheLine => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Hold the Line".into(),
                effect_description: "Party becomes immovable while outnumbered.".into(),
            });
            // Boost all party members' defense.
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        let member_ids: Vec<u32> = party.member_ids.clone();
                        for mid in member_ids {
                            if let Some(m) =
                                state.adventurers.iter_mut().find(|a| a.id == mid)
                            {
                                m.skill_state.combat_power_bonus += 0.25;
                            }
                        }
                    }
                }
            }
            "Hold the line — party immovable while outnumbered.".into()
        }

        SkillEffect::BladeStorm => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Blade Storm".into(),
                effect_description: "Attacked all adjacent enemies simultaneously.".into(),
            });
            "Blade storm — all adjacent enemies struck.".into()
        }

        SkillEffect::GoldenTouch => {
            state.golden_touch_active = true;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Golden Touch".into(),
                effect_description: "Quest loot doubled.".into(),
            });
            "Golden touch — quest loot doubled.".into()
        }

        SkillEffect::PlagueWard { region_id, duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Plague Ward".into(),
                effect_description: format!(
                    "Region {} immune to disease for {} ticks.",
                    region_id, duration_ticks
                ),
            });
            format!("Plague ward — region {} immune to disease.", region_id)
        }

        SkillEffect::Resurrection => {
            // Find first dead party member and revive them.
            let mut revived = None;
            if let Some(adv) = state.adventurers.iter().find(|a| a.id == caster_id) {
                if let Some(pid) = adv.party_id {
                    if let Some(party) = state.parties.iter().find(|p| p.id == pid) {
                        for &mid in &party.member_ids {
                            if let Some(m) = state.adventurers.iter().find(|a| a.id == mid) {
                                if m.status == AdventurerStatus::Dead {
                                    revived = Some(mid);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if let Some(mid) = revived {
                if let Some(m) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                    m.status = AdventurerStatus::Idle;
                    m.morale = 30.0;
                }
                events.push(WorldEvent::ClassSkillUsed {
                    adventurer_id: caster_id,
                    skill_name: "Resurrection".into(),
                    effect_description: format!("Adventurer {} revived mid-quest.", mid),
                });
                format!("Resurrection — adventurer {} revived.", mid)
            } else {
                "Resurrection — no fallen allies to revive.".into()
            }
        }

        SkillEffect::PredictiveModel => {
            state.threat_clock_slowdown += 0.2;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Predictive Model".into(),
                effect_description: "Threat clock growth reduced by 20%.".into(),
            });
            "Predictive model — threat clock slowed.".into()
        }

        SkillEffect::ForbiddenKnowledge => {
            state.forbidden_knowledge_active = true;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Forbidden Knowledge".into(),
                effect_description: "Hidden class requirements unlocked.".into(),
            });
            "Forbidden knowledge — hidden class paths revealed.".into()
        }

        SkillEffect::TradeEmpire { income_per_tick } => {
            state.trade_empire_income += income_per_tick;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Trade Empire".into(),
                effect_description: format!(
                    "Passive income: {} gold/tick from all trade routes.",
                    income_per_tick
                ),
            });
            format!("Trade empire — {} gold/tick passive income.", income_per_tick)
        }

        SkillEffect::IntelGathering => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Intel Gathering".into(),
                effect_description: "Enemy tactical plans stolen.".into(),
            });
            "Intel gathered — enemy plans known.".into()
        }

        SkillEffect::MasterArmorer => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Master Armorer".into(),
                effect_description: "All party equipment upgraded by one tier.".into(),
            });
            "Master armorer — equipment upgraded.".into()
        }

        SkillEffect::FieldCommand { duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Field Command".into(),
                effect_description: format!(
                    "Allied NPCs following complex orders for {} ticks.",
                    duration_ticks
                ),
            });
            format!("Field command active for {} ticks.", duration_ticks)
        }

        SkillEffect::Fortify { region_id, duration_ticks } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Fortify".into(),
                effect_description: format!(
                    "Position fortified in region {} for {} ticks.",
                    region_id, duration_ticks
                ),
            });
            format!("Position fortified in region {}.", region_id)
        }

        // =====================================================================
        // T6 — Legendary (emit events, simplified)
        // =====================================================================
        SkillEffect::MarkPrey { enemy_type } => {
            state.marked_prey_types.push(enemy_type.clone());
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Mark Prey".into(),
                effect_description: format!(
                    "Guild-wide combat bonus against {} permanently.",
                    enemy_type
                ),
            });
            format!("Prey marked — guild-wide bonus against {}.", enemy_type)
        }

        SkillEffect::ClaimByRight { region_id } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Claim by Right".into(),
                effect_description: format!("Region {} claimed through reputation.", region_id),
            });
            format!("Region {} claimed by right.", region_id)
        }

        SkillEffect::ShatterAlliance { faction_a, faction_b } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Shatter Alliance".into(),
                effect_description: format!(
                    "Alliance between factions {} and {} dissolved.",
                    faction_a, faction_b
                ),
            });
            format!("Alliance between {} and {} shattered.", faction_a, faction_b)
        }

        SkillEffect::Vanish => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.vanished = true;
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Vanish".into(),
                effect_description: "Erased from all faction records.".into(),
            });
            "Vanished — erased from all faction records.".into()
        }

        SkillEffect::ImmortalMoment => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Immortal Moment".into(),
                effect_description: "Time frozen — ally saved from death.".into(),
            });
            "Immortal moment — time frozen, ally saved.".into()
        }

        SkillEffect::ChampionsChallenge => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Champion's Challenge".into(),
                effect_description: "Enemy leader forced into 1v1 duel.".into(),
            });
            "Champion's challenge — 1v1 duel forced.".into()
        }

        SkillEffect::MarketMaker { commodity_index, duration_ticks } => {
            state.market_monopolies.push((*commodity_index, caster_id, *duration_ticks));
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Market Maker".into(),
                effect_description: format!(
                    "Price of commodity {} set for {} ticks.",
                    commodity_index, duration_ticks
                ),
            });
            format!("Market maker — commodity {} price controlled.", commodity_index)
        }

        SkillEffect::LifeEternal => {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == caster_id) {
                adv.skill_state.immortal = true;
            }
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Life Eternal".into(),
                effect_description: "Immune to aging and disease.".into(),
            });
            "Life eternal — immune to aging and disease.".into()
        }

        SkillEffect::RewriteTheRecord => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Rewrite the Record".into(),
                effect_description: "Chronicle entry altered retroactively.".into(),
            });
            "Record rewritten — chronicle altered.".into()
        }

        SkillEffect::AllSeeingEye { region_id } => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "All-Seeing Eye".into(),
                effect_description: format!("All hidden information in region {} revealed.", region_id),
            });
            format!("All-seeing eye — region {} fully revealed.", region_id)
        }

        SkillEffect::ForgeArtifact => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Forge Artifact".into(),
                effect_description: "Legendary artifact forged.".into(),
            });
            "Legendary artifact forged.".into()
        }

        // =====================================================================
        // T7 — Mythic capstone (guild-wide permanent effects)
        // =====================================================================
        SkillEffect::LivingLegend => {
            state.living_legend_holder = Some(caster_id);
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Living Legend".into(),
                effect_description: "All guild members gain +1 level in primary class.".into(),
            });
            "Living legend — guild-wide +1 level.".into()
        }

        SkillEffect::RewriteHistory => {
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Rewrite History".into(),
                effect_description: "One chronicle entry rewritten as retroactively true.".into(),
            });
            "History rewritten.".into()
        }

        SkillEffect::TheLastWord => {
            // End one ongoing conflict.
            if !state.overworld.active_crises.is_empty() {
                state.overworld.active_crises.remove(0);
                events.push(WorldEvent::ClassSkillUsed {
                    adventurer_id: caster_id,
                    skill_name: "The Last Word".into(),
                    effect_description: "One ongoing crisis permanently ended.".into(),
                });
                "The last word — one crisis ended permanently.".into()
            } else {
                "The last word — no active conflicts to end.".into()
            }
        }

        SkillEffect::WealthOfNations => {
            state.wealth_of_nations_active = true;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Wealth of Nations".into(),
                effect_description: "Guild gold generation permanently doubled.".into(),
            });
            "Wealth of nations — gold generation doubled permanently.".into()
        }

        SkillEffect::Omniscience => {
            state.omniscience_active = true;
            events.push(WorldEvent::ClassSkillUsed {
                adventurer_id: caster_id,
                skill_name: "Omniscience".into(),
                effect_description: "All hidden campaign information revealed.".into(),
            });
            "Omniscience — all hidden information revealed.".into()
        }
    }
}
