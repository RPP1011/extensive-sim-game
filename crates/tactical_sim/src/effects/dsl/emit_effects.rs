//! Effect emission match arms (Effect → DSL string fragments).

use crate::effects::effect_enum::Effect;
use crate::effects::types::*;
use super::emit_helpers::{fmt_duration, fmt_f32, fmt_tick_duration};

pub(super) fn emit_effect(effect: &Effect) -> String {
    match effect {
        Effect::Damage { amount, amount_per_tick, duration_ms, damage_type, .. } => {
            if *amount_per_tick > 0 && *duration_ms > 0 {
                let dt = match damage_type {
                    DamageType::Magic => " magic",
                    DamageType::True => " true",
                    _ => "",
                };
                format!("damage {amount_per_tick}/tick{dt} for {}", fmt_duration(*duration_ms))
            } else {
                let dt = match damage_type {
                    DamageType::Magic => " magic",
                    DamageType::True => " true",
                    _ => "",
                };
                format!("damage {amount}{dt}")
            }
        }
        Effect::Heal { amount, amount_per_tick, duration_ms, .. } => {
            if *amount_per_tick > 0 && *duration_ms > 0 {
                format!("heal {amount_per_tick}/tick for {}", fmt_duration(*duration_ms))
            } else {
                format!("heal {amount}")
            }
        }
        Effect::Shield { amount, duration_ms } => {
            format!("shield {amount} for {}", fmt_duration(*duration_ms))
        }
        Effect::Stun { duration_ms } => format!("stun {}", fmt_duration(*duration_ms)),
        Effect::Slow { factor, duration_ms } => {
            format!("slow {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Root { duration_ms } => format!("root {}", fmt_duration(*duration_ms)),
        Effect::Silence { duration_ms } => format!("silence {}", fmt_duration(*duration_ms)),
        Effect::Fear { duration_ms } => format!("fear {}", fmt_duration(*duration_ms)),
        Effect::Taunt { duration_ms } => format!("taunt {}", fmt_duration(*duration_ms)),
        Effect::Knockback { distance } => format!("knockback {}", fmt_f32(*distance)),
        Effect::Pull { distance } => format!("pull {}", fmt_f32(*distance)),
        Effect::Dash { distance, to_target, is_blink, .. } => {
            let mut s = if *is_blink {
                format!("blink {}", fmt_f32(*distance))
            } else {
                format!("dash {}", fmt_f32(*distance))
            };
            if *to_target { s.push_str(" to_target"); }
            s
        }
        Effect::Buff { stat, factor, duration_ms } => {
            format!("buff {stat} {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Debuff { stat, factor, duration_ms } => {
            format!("debuff {stat} {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Duel { duration_ms } => format!("duel {}", fmt_duration(*duration_ms)),
        Effect::Summon { template, count, clone, directed, .. } => {
            let mut s = format!("summon \"{template}\"");
            if *count > 1 { s.push_str(&format!(" x{count}")); }
            if *clone { s.push_str(" clone"); }
            if *directed { s.push_str(" directed"); }
            // Note: hp_percent is not emitted because the parser does not
            // support the `hp:N%` syntax (lowering always defaults to 100%).
            s
        }
        Effect::CommandSummons { speed } => format!("command_summons speed:{}", fmt_f32(*speed)),
        Effect::Dispel { target_tags } => {
            if target_tags.is_empty() {
                "dispel".to_string()
            } else {
                // Wrap tags as string literals so tokenizer maps them to [STR]
                let tags: Vec<String> = target_tags.iter()
                    .map(|t| format!("\"{t}\""))
                    .collect();
                format!("dispel {}", tags.join(" "))
            }
        }
        Effect::Reflect { percent, duration_ms } => {
            format!("reflect {} for {}", fmt_f32(*percent), fmt_duration(*duration_ms))
        }
        Effect::Lifesteal { percent, duration_ms } => {
            format!("lifesteal {} for {}", fmt_f32(*percent), fmt_duration(*duration_ms))
        }
        Effect::DamageModify { factor, duration_ms } => {
            format!("damage_modify {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::SelfDamage { amount } => format!("self_damage {amount}"),
        Effect::Execute { hp_threshold_percent } => {
            format!("execute {}%", *hp_threshold_percent as i32)
        }
        Effect::Blind { miss_chance, duration_ms } => {
            format!("blind {} for {}", fmt_f32(*miss_chance), fmt_duration(*duration_ms))
        }
        Effect::OnHitBuff { duration_ms, .. } => {
            format!("on_hit_buff for {}", fmt_duration(*duration_ms))
        }
        Effect::Resurrect { hp_percent } => format!("resurrect {}%", *hp_percent as i32),
        Effect::OverhealShield { duration_ms, .. } => {
            format!("overheal_shield for {}", fmt_duration(*duration_ms))
        }
        Effect::AbsorbToHeal { shield_amount, duration_ms, .. } => {
            format!("absorb_to_heal {shield_amount} for {}", fmt_duration(*duration_ms))
        }
        Effect::ShieldSteal { amount } => format!("shield_steal {amount}"),
        Effect::StatusClone { max_count } => format!("status_clone {max_count}"),
        Effect::Immunity { immune_to, duration_ms } => {
            format!("immunity {} for {}", immune_to.join(","), fmt_duration(*duration_ms))
        }
        Effect::Detonate { damage_multiplier } => {
            format!("detonate {}", fmt_f32(*damage_multiplier))
        }
        Effect::StatusTransfer { steal_buffs } => {
            if *steal_buffs { "status_transfer steal".to_string() }
            else { "status_transfer".to_string() }
        }
        Effect::DeathMark { duration_ms, damage_percent } => {
            format!("death_mark {} for {}", fmt_f32(*damage_percent), fmt_duration(*duration_ms))
        }
        Effect::Polymorph { duration_ms } => format!("polymorph {}", fmt_duration(*duration_ms)),
        Effect::Banish { duration_ms } => format!("banish {}", fmt_duration(*duration_ms)),
        Effect::Confuse { duration_ms } => format!("confuse {}", fmt_duration(*duration_ms)),
        Effect::Charm { duration_ms } => format!("charm {}", fmt_duration(*duration_ms)),
        Effect::Stealth { duration_ms, break_on_damage, break_on_ability } => {
            let mut s = format!("stealth for {}", fmt_duration(*duration_ms));
            if *break_on_damage { s.push_str(" break_on_damage"); }
            if *break_on_ability { s.push_str(" break_on_ability"); }
            s
        }
        Effect::Leash { max_range, duration_ms } => {
            format!("leash {} for {}", fmt_f32(*max_range), fmt_duration(*duration_ms))
        }
        Effect::Link { duration_ms, share_percent } => {
            format!("link {}% for {}", *share_percent as i32, fmt_duration(*duration_ms))
        }
        Effect::Redirect { duration_ms, charges } => {
            format!("redirect {charges} for {}", fmt_duration(*duration_ms))
        }
        Effect::Rewind { lookback_ms } => format!("rewind {}", fmt_duration(*lookback_ms)),
        Effect::CooldownModify { amount_ms, ability_name } => {
            if let Some(ref name) = ability_name {
                format!("cooldown_modify {amount_ms}ms \"{name}\"")
            } else {
                format!("cooldown_modify {amount_ms}ms")
            }
        }
        Effect::ApplyStacks { name, count, max_stacks, duration_ms } => {
            let mut s = format!("apply_stacks \"{name}\" {count}/{max_stacks}");
            if *duration_ms > 0 {
                s.push_str(&format!(" for {}", fmt_duration(*duration_ms)));
            }
            s
        }
        Effect::Obstacle { width, height } => {
            format!("obstacle {} {}", fmt_f32(*width), fmt_f32(*height))
        }
        Effect::Suppress { duration_ms } => format!("suppress {}", fmt_duration(*duration_ms)),
        Effect::Grounded { duration_ms } => format!("grounded {}", fmt_duration(*duration_ms)),
        Effect::ProjectileBlock { duration_ms } => {
            format!("projectile_block {}", fmt_duration(*duration_ms))
        }
        Effect::Attach { duration_ms } => format!("attach for {}", fmt_duration(*duration_ms)),
        Effect::EvolveAbility { ability_index } => format!("evolve {ability_index}"),
        Effect::Swap => "swap".to_string(),

        // ===================================================================
        // Meta-Effects
        // ===================================================================
        Effect::RefreshCooldowns => "refresh_cooldowns".to_string(),
        Effect::RefreshCooldown { ability_index } => format!("refresh_cooldown {}", ability_index),
        Effect::Amplify { multiplier, charges } => format!("amplify {} {}", fmt_f32(*multiplier), charges),
        Effect::Echo { charges } => format!("echo {}", charges),
        Effect::ExtendDurations { amount_ms } => format!("extend_durations {}", fmt_duration(*amount_ms)),
        Effect::InstantCast { charges } => format!("instant_cast {}", charges),
        Effect::FreeCast { charges } => format!("free_cast {}", charges),
        Effect::SpellShield { charges } => format!("spell_shield {}", charges),
        Effect::ManaBurn { cost_multiplier, duration_ms } => {
            format!("mana_burn {} for {}", fmt_f32(*cost_multiplier), fmt_duration(*duration_ms))
        }
        Effect::CooldownLock { duration_ms } => format!("cooldown_lock {}", fmt_duration(*duration_ms)),

        // ===================================================================
        // Recursive Effects
        // ===================================================================
        Effect::OnHitCast { ability_name } => format!("on_hit_cast \"{}\"", ability_name),
        Effect::GrantAbility { ability_name, duration_ms } => {
            format!("grant_ability \"{}\" for {}", ability_name, fmt_duration(*duration_ms))
        }
        Effect::CastCopy => "cast_copy last_used".to_string(),
        Effect::EvolveAfter { cast_count } => format!("evolve_after {}", cast_count),

        // ===================================================================
        // Campaign Primitives
        // ===================================================================
        Effect::ModifyStat { entity, property, op: _, amount, duration_ticks } => {
            if *duration_ticks > 0 {
                format!("modify_stat \"{}\" \"{}\" {} for {}t", entity, property, fmt_f32(*amount), duration_ticks)
            } else {
                format!("modify_stat \"{}\" \"{}\" {}", entity, property, fmt_f32(*amount))
            }
        }
        Effect::SetFlag { entity, flag, value, duration_ticks } => {
            let val = if *value { "true" } else { "false" };
            if *duration_ticks > 0 {
                format!("set_flag \"{}\" \"{}\" {} for {}t", entity, flag, val, duration_ticks)
            } else {
                format!("set_flag \"{}\" \"{}\" {}", entity, flag, val)
            }
        }
        Effect::RevealInfo { target_type, scope } => format!("reveal_info \"{}\" \"{}\"", target_type, scope),
        Effect::CreateEntity { entity_type, subtype, duration_ticks } => {
            if *duration_ticks > 0 {
                format!("create_entity \"{}\" \"{}\" for {}t", entity_type, subtype, duration_ticks)
            } else {
                format!("create_entity \"{}\" \"{}\"", entity_type, subtype)
            }
        }
        Effect::DestroyEntity { target_type } => format!("destroy_entity \"{}\"", target_type),
        Effect::TransferValue { from_entity, to_entity, property, amount } => {
            format!("transfer \"{}\" \"{}\" \"{}\" {}", from_entity, to_entity, property, fmt_f32(*amount))
        }

        // ===================================================================
        // Campaign Effects
        // ===================================================================

        // --- Economy ---
        Effect::CornerMarket { commodity, duration_ticks } => {
            if commodity.is_empty() {
                format!("corner_market for {}", fmt_tick_duration(*duration_ticks))
            } else {
                format!("corner_market \"{}\" for {}", commodity, fmt_tick_duration(*duration_ticks))
            }
        }
        Effect::ForgeTradeRoute { income_per_tick, duration_ticks } => {
            format!("forge_trade_route {} for {}", fmt_f32(*income_per_tick), fmt_tick_duration(*duration_ticks))
        }
        Effect::Appraise => "appraise".to_string(),
        Effect::GoldenTouch { duration_ticks } => format!("golden_touch for {}", fmt_tick_duration(*duration_ticks)),
        Effect::TradeEmbargo { duration_ticks } => format!("trade_embargo for {}", fmt_tick_duration(*duration_ticks)),
        Effect::SilverTongue => "silver_tongue".to_string(),

        // --- Diplomacy ---
        Effect::DemandAudience => "demand_audience".to_string(),
        Effect::CeasefireDeclaration { duration_ticks } => format!("ceasefire for {}", fmt_tick_duration(*duration_ticks)),
        Effect::Destabilize { instability, duration_ticks } => {
            format!("destabilize {} for {}", fmt_f32(*instability), fmt_tick_duration(*duration_ticks))
        }
        Effect::BrokerAlliance { duration_ticks } => format!("broker_alliance for {}", fmt_tick_duration(*duration_ticks)),
        Effect::SubvertLoyalty => "subvert_loyalty".to_string(),
        Effect::TreatyBreaker => "treaty_breaker".to_string(),
        Effect::ShatterAlliance => "shatter_alliance".to_string(),

        // --- Information ---
        Effect::Reveal { count } => format!("reveal {}", count),
        Effect::PropheticVision { count } => format!("prophecy {}", count),
        Effect::BeastLore => "beast_lore".to_string(),
        Effect::ReadTheRoom => "read_the_room".to_string(),
        Effect::AllSeeingEye => "all_seeing_eye".to_string(),
        Effect::Decipher => "decipher".to_string(),
        Effect::TrapSense => "trap_sense".to_string(),
        Effect::SapperEye => "sapper_eye".to_string(),

        // --- Leadership ---
        Effect::Rally { morale_restore } => format!("rally {}", fmt_f32(*morale_restore)),
        Effect::RallyingCry { morale_restore } => format!("rallying_cry {}", fmt_f32(*morale_restore)),
        Effect::Inspire { morale_boost, duration_ticks } => {
            format!("inspire {} for {}", fmt_f32(*morale_boost), fmt_tick_duration(*duration_ticks))
        }
        Effect::FieldCommand { duration_ticks } => format!("field_command for {}", fmt_tick_duration(*duration_ticks)),
        Effect::CoordinatedStrike => "coordinated_strike".to_string(),
        Effect::WarCry { duration_ticks } => format!("war_cry for {}", fmt_tick_duration(*duration_ticks)),

        // --- Stealth / Movement ---
        Effect::GhostWalk { duration_ticks } => format!("ghost_walk for {}", fmt_tick_duration(*duration_ticks)),
        Effect::ShadowStep { duration_ticks } => format!("shadow_step for {}", fmt_tick_duration(*duration_ticks)),
        Effect::SilentMovement => "silent_movement".to_string(),
        Effect::HiddenCamp { duration_ticks } => format!("hidden_camp for {}", fmt_tick_duration(*duration_ticks)),
        Effect::Vanish => "vanish".to_string(),
        Effect::Distraction { duration_ticks } => format!("distraction for {}", fmt_tick_duration(*duration_ticks)),

        // --- Territory ---
        Effect::ClaimTerritory => "claim_territory".to_string(),
        Effect::Fortify { duration_ticks } => format!("fortify for {}", fmt_tick_duration(*duration_ticks)),
        Effect::Sanctuary { duration_ticks } => format!("sanctuary for {}", fmt_tick_duration(*duration_ticks)),
        Effect::PlagueWard { duration_ticks } => format!("plague_ward for {}", fmt_tick_duration(*duration_ticks)),
        Effect::SafeHouse { duration_ticks } => format!("safe_house for {}", fmt_tick_duration(*duration_ticks)),

        // --- Supernatural / Body ---
        Effect::BloodOath { stat_bonus } => format!("blood_oath {}", fmt_f32(*stat_bonus)),
        Effect::Unbreakable => "unbreakable".to_string(),
        Effect::LifeEternal => "life_eternal".to_string(),
        Effect::Purify => "purify".to_string(),
        Effect::NameTheNameless => "name_the_nameless".to_string(),
        Effect::ForbiddenKnowledge => "forbidden_knowledge".to_string(),

        // --- Passive Skill-State ---
        Effect::FieldTriage { heal_rate_multiplier } => format!("field_triage {}", fmt_f32(*heal_rate_multiplier)),
        Effect::InspiringPresence { morale_boost } => format!("inspiring_presence {}", fmt_f32(*morale_boost)),
        Effect::BattleInstinct => "battle_instinct".to_string(),
        Effect::QuickStudy => "quick_study".to_string(),
        Effect::Forage { supply_per_tick } => format!("forage {}", fmt_f32(*supply_per_tick)),
        Effect::TrackPrey => "track_prey".to_string(),
        Effect::FieldRepair => "field_repair".to_string(),
        Effect::StabilizeAlly => "stabilize_ally".to_string(),

        // --- Higher Tier ---
        Effect::Disinformation { duration_ticks } => format!("disinformation for {}", fmt_tick_duration(*duration_ticks)),
        Effect::AcceleratedStudy { duration_ticks } => format!("accelerated_study for {}", fmt_tick_duration(*duration_ticks)),
        Effect::TakeTheBlow { duration_ticks } => format!("take_the_blow for {}", fmt_tick_duration(*duration_ticks)),
        Effect::HoldTheLine => "hold_the_line".to_string(),
        Effect::Forgery => "forgery".to_string(),
        Effect::MasterworkCraft => "masterwork_craft".to_string(),
        Effect::IntelGathering => "intel_gathering".to_string(),
        Effect::MasterArmorer => "master_armorer".to_string(),
        Effect::MarketMaker { duration_ticks } => format!("market_maker for {}", fmt_tick_duration(*duration_ticks)),
        Effect::ForgeArtifact => "forge_artifact".to_string(),
        Effect::TradeEmpire { income_per_tick } => format!("trade_empire {}", fmt_f32(*income_per_tick)),

        // --- Legendary / Mythic ---
        Effect::LivingLegend => "living_legend".to_string(),
        Effect::RewriteHistory => "rewrite_history".to_string(),
        Effect::TheLastWord => "the_last_word".to_string(),
        Effect::WealthOfNations => "wealth_of_nations".to_string(),
        Effect::Omniscience => "omniscience".to_string(),
        Effect::ImmortalMoment => "immortal_moment".to_string(),
        Effect::ClaimByRight => "claim_by_right".to_string(),
        Effect::RewriteTheRecord => "rewrite_the_record".to_string(),
    }
}

pub(super) fn emit_scaling(effect: &Effect) -> String {
    let bonus = match effect {
        Effect::Damage { bonus, scaling_stat, scaling_percent, .. } => {
            if let Some(ref stat) = scaling_stat {
                if *scaling_percent > 0.0 {
                    return format!("+ {}% {stat}", *scaling_percent as i32);
                }
            }
            bonus
        }
        Effect::Heal { bonus, scaling_stat, scaling_percent, .. } => {
            if let Some(ref stat) = scaling_stat {
                if *scaling_percent > 0.0 {
                    return format!("+ {}% {stat}", *scaling_percent as i32);
                }
            }
            bonus
        }
        _ => return String::new(),
    };
    if bonus.is_empty() { return String::new(); }
    let mut plus_terms = Vec::new();
    let mut scales_with_terms = Vec::new();
    for t in bonus {
        let campaign_stat = match &t.stat {
            StatRef::KingdomSize => Some("kingdom_size"),
            StatRef::ArmySize => Some("army_size"),
            StatRef::FactionTerritory => Some("faction_territory"),
            StatRef::GuildReputation => Some("guild_reputation"),
            StatRef::AdventurerCount => Some("adventurer_count"),
            StatRef::LoyaltyAverage => Some("loyalty_average"),
            StatRef::PartySize => Some("party_size"),
            StatRef::GuildGold => Some("guild_gold"),
            StatRef::CasterLevel => Some("caster_level"),
            _ => None,
        };
        // Emit campaign stats at 100% as `scales_with STAT` sugar
        if let Some(name) = campaign_stat {
            if (t.percent - 100.0).abs() < 0.01 {
                scales_with_terms.push(format!("scales_with {name}"));
                continue;
            }
        }
        let stat_name = match &t.stat {
            StatRef::TargetMaxHp => "target_max_hp",
            StatRef::TargetCurrentHp => "target_current_hp",
            StatRef::TargetMissingHp => "target_missing_hp",
            StatRef::CasterMaxHp => "caster_max_hp",
            StatRef::CasterCurrentHp => "caster_current_hp",
            StatRef::CasterMissingHp => "caster_missing_hp",
            StatRef::CasterAttackDamage => "caster_attack_damage",
            StatRef::TargetStacks { name } => { plus_terms.push(format!("{}% target_stacks({name})", t.percent as i32)); continue; }
            StatRef::CasterStacks { name } => { plus_terms.push(format!("{}% caster_stacks({name})", t.percent as i32)); continue; }
            StatRef::KingdomSize => "kingdom_size",
            StatRef::ArmySize => "army_size",
            StatRef::FactionTerritory => "faction_territory",
            StatRef::GuildReputation => "guild_reputation",
            StatRef::AdventurerCount => "adventurer_count",
            StatRef::LoyaltyAverage => "loyalty_average",
            StatRef::PartySize => "party_size",
            StatRef::GuildGold => "guild_gold",
            StatRef::CasterLevel => "caster_level",
        };
        plus_terms.push(format!("{}% {stat_name}", t.percent as i32));
    }
    let mut result = String::new();
    if !plus_terms.is_empty() {
        result.push_str(&format!("+ {}", plus_terms.join(" + ")));
    }
    for sw in &scales_with_terms {
        if !result.is_empty() { result.push(' '); }
        result.push_str(sw);
    }
    result
}
