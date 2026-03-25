//! Tiered skill template pools for the TWI class system (Phase 10).
//!
//! Each class family has templates across tiers 1-7. Templates define the
//! actual power (SkillEffect) granted when a class levels up past a threshold.
//!
//! Design philosophy:
//! - 80% of skills are always available, no conditions
//! - 20% are conditional, punching above their weight (Lv30 conditional = Lv60 normal)
//! - No resource costs — constrained by specificity and narrative consequence
//! - Non-combat skills are POWERS, not stat buffs

use crate::headless_campaign::state::{BehaviorLedger, SkillCondition, SkillEffect};

/// A skill template from the pool, selected at level-up time.
pub struct SkillTemplate {
    pub name: &'static str,
    pub tier: u32,
    pub class_tags: &'static [&'static str],
    pub effect: SkillEffect,
    pub conditional: bool,
    pub condition: Option<SkillCondition>,
    pub description: &'static str,
}

// ---------------------------------------------------------------------------
// Template pool — ~120 templates across 11 class families x 7 tiers
// ---------------------------------------------------------------------------

fn all_templates() -> Vec<SkillTemplate> {
    let mut t = Vec::with_capacity(130);

    // === WARRIOR ===
    t.push(SkillTemplate {
        name: "Battle Instinct",
        tier: 1,
        class_tags: &["Warrior"],
        effect: SkillEffect::BattleInstinct,
        conditional: false,
        condition: None,
        description: "Sense ambushes before they trigger.",
    });
    t.push(SkillTemplate {
        name: "Field Repair",
        tier: 1,
        class_tags: &["Warrior", "Artisan"],
        effect: SkillEffect::FieldRepair,
        conditional: false,
        condition: None,
        description: "Repair gear without a workshop.",
    });
    t.push(SkillTemplate {
        name: "War Cry",
        tier: 1,
        class_tags: &["Warrior", "Commander"],
        effect: SkillEffect::WarCry { duration_ticks: 50 },
        conditional: false,
        condition: None,
        description: "Force all enemies to target you for 50 ticks.",
    });
    t.push(SkillTemplate {
        name: "Unbreakable",
        tier: 3,
        class_tags: &["Warrior", "Guardian"],
        effect: SkillEffect::Unbreakable,
        conditional: false,
        condition: None,
        description: "Survive one killing blow per quest with 1 HP.",
    });
    t.push(SkillTemplate {
        name: "Coordinated Strike",
        tier: 3,
        class_tags: &["Warrior", "Commander"],
        effect: SkillEffect::CoordinatedStrike,
        conditional: false,
        condition: None,
        description: "Coordinate two units to attack simultaneously.",
    });
    t.push(SkillTemplate {
        name: "Blade Storm",
        tier: 4,
        class_tags: &["Warrior"],
        effect: SkillEffect::BladeStorm,
        conditional: false,
        condition: None,
        description: "Attack all adjacent enemies simultaneously.",
    });
    t.push(SkillTemplate {
        name: "Hold the Line",
        tier: 5,
        class_tags: &["Warrior", "Guardian", "Commander"],
        effect: SkillEffect::HoldTheLine,
        conditional: true,
        condition: Some(SkillCondition::Outnumbered),
        description: "When outnumbered, party becomes immovable.",
    });
    t.push(SkillTemplate {
        name: "Champion's Challenge",
        tier: 6,
        class_tags: &["Warrior"],
        effect: SkillEffect::ChampionsChallenge,
        conditional: false,
        condition: None,
        description: "Force enemy leader into 1v1 duel.",
    });
    t.push(SkillTemplate {
        name: "Living Legend",
        tier: 7,
        class_tags: &["Warrior"],
        effect: SkillEffect::LivingLegend,
        conditional: false,
        condition: None,
        description: "While you live, all guild members gain +1 level in primary class.",
    });

    // === RANGER ===
    t.push(SkillTemplate {
        name: "Track Prey",
        tier: 1,
        class_tags: &["Ranger", "Scout"],
        effect: SkillEffect::TrackPrey,
        conditional: false,
        condition: None,
        description: "Track enemies through any terrain.",
    });
    t.push(SkillTemplate {
        name: "Beast Lore",
        tier: 1,
        class_tags: &["Ranger", "Scholar"],
        effect: SkillEffect::BeastLore,
        conditional: false,
        condition: None,
        description: "Identify monster weaknesses after one encounter.",
    });
    t.push(SkillTemplate {
        name: "Hidden Camp",
        tier: 1,
        class_tags: &["Ranger", "Scout"],
        effect: SkillEffect::HiddenCamp { duration_ticks: 200 },
        conditional: false,
        condition: None,
        description: "Set up a concealed camp that enemies cannot find.",
    });
    t.push(SkillTemplate {
        name: "Forage",
        tier: 1,
        class_tags: &["Ranger"],
        effect: SkillEffect::Forage { supply_per_tick: 0.05 },
        conditional: false,
        condition: None,
        description: "Forage supplies while traveling.",
    });
    t.push(SkillTemplate {
        name: "Ghost Walk",
        tier: 3,
        class_tags: &["Ranger", "Scout", "Rogue"],
        effect: SkillEffect::GhostWalk { duration_ticks: 100 },
        conditional: false,
        condition: None,
        description: "Travel through hostile territory as if friendly.",
    });
    t.push(SkillTemplate {
        name: "Mark Prey",
        tier: 6,
        class_tags: &["Ranger"],
        effect: SkillEffect::MarkPrey { enemy_type: String::new() },
        conditional: false,
        condition: None,
        description: "Permanently grant guild-wide bonus against a named enemy type.",
    });

    // === HEALER ===
    t.push(SkillTemplate {
        name: "Field Triage",
        tier: 1,
        class_tags: &["Healer"],
        effect: SkillEffect::FieldTriage { heal_rate_multiplier: 1.3 },
        conditional: false,
        condition: None,
        description: "Reduce wound healing time by 30%.",
    });
    t.push(SkillTemplate {
        name: "Stabilize Ally",
        tier: 1,
        class_tags: &["Healer", "Guardian"],
        effect: SkillEffect::StabilizeAlly,
        conditional: false,
        condition: None,
        description: "Stabilize dying allies automatically.",
    });
    t.push(SkillTemplate {
        name: "Purify",
        tier: 1,
        class_tags: &["Healer"],
        effect: SkillEffect::Purify,
        conditional: false,
        condition: None,
        description: "Cleanse one disease or poison instantly.",
    });
    t.push(SkillTemplate {
        name: "Sanctuary Aura",
        tier: 3,
        class_tags: &["Healer"],
        effect: SkillEffect::Sanctuary { region_id: 0, duration_ticks: 200 },
        conditional: false,
        condition: None,
        description: "Party takes 20% less damage passively.",
    });
    t.push(SkillTemplate {
        name: "Resurrection",
        tier: 4,
        class_tags: &["Healer"],
        effect: SkillEffect::Resurrection,
        conditional: false,
        condition: None,
        description: "Revive a fallen adventurer mid-quest.",
    });
    t.push(SkillTemplate {
        name: "Plague Ward",
        tier: 5,
        class_tags: &["Healer"],
        effect: SkillEffect::PlagueWard { region_id: 0, duration_ticks: 500 },
        conditional: false,
        condition: None,
        description: "Make a region immune to disease spread.",
    });
    t.push(SkillTemplate {
        name: "Immortal Moment",
        tier: 6,
        class_tags: &["Healer"],
        effect: SkillEffect::ImmortalMoment,
        conditional: true,
        condition: Some(SkillCondition::AllyDying),
        description: "When an ally is about to die, time stops and you act.",
    });
    t.push(SkillTemplate {
        name: "Life Eternal",
        tier: 6,
        class_tags: &["Healer"],
        effect: SkillEffect::LifeEternal,
        conditional: false,
        condition: None,
        description: "One adventurer becomes immune to aging and disease.",
    });

    // === DIPLOMAT ===
    t.push(SkillTemplate {
        name: "Read the Room",
        tier: 1,
        class_tags: &["Diplomat"],
        effect: SkillEffect::ReadTheRoom,
        conditional: false,
        condition: None,
        description: "See faction diplomatic stance before committing.",
    });
    t.push(SkillTemplate {
        name: "Inspiring Presence",
        tier: 1,
        class_tags: &["Diplomat", "Commander"],
        effect: SkillEffect::InspiringPresence { morale_boost: 5.0 },
        conditional: false,
        condition: None,
        description: "Boost party morale passively.",
    });
    t.push(SkillTemplate {
        name: "Demand Audience",
        tier: 1,
        class_tags: &["Diplomat"],
        effect: SkillEffect::DemandAudience,
        conditional: false,
        condition: None,
        description: "Force a diplomatic meeting with any faction.",
    });
    t.push(SkillTemplate {
        name: "Silver Tongue",
        tier: 1,
        class_tags: &["Diplomat", "Merchant"],
        effect: SkillEffect::SilverTongue,
        conditional: false,
        condition: None,
        description: "Always get the better end of a deal.",
    });
    t.push(SkillTemplate {
        name: "Ceasefire Declaration",
        tier: 4,
        class_tags: &["Diplomat"],
        effect: SkillEffect::CeasefireDeclaration,
        conditional: false,
        condition: None,
        description: "Halt an active battle through sheer diplomatic force.",
    });
    t.push(SkillTemplate {
        name: "Whisper of Succession",
        tier: 5,
        class_tags: &["Diplomat", "Rogue"],
        effect: SkillEffect::WhisperOfSuccession { target_faction: 0, instability: 0.3 },
        conditional: false,
        condition: None,
        description: "Plant doubt in faction leadership, destabilizing their politics.",
    });
    t.push(SkillTemplate {
        name: "Treaty Breaker",
        tier: 5,
        class_tags: &["Diplomat"],
        effect: SkillEffect::TreatyBreaker,
        conditional: false,
        condition: None,
        description: "Force all factions to renegotiate a specific treaty.",
    });
    t.push(SkillTemplate {
        name: "Shatter Alliance",
        tier: 6,
        class_tags: &["Diplomat"],
        effect: SkillEffect::ShatterAlliance { faction_a: 0, faction_b: 0 },
        conditional: false,
        condition: None,
        description: "Dissolve an alliance between two factions.",
    });
    t.push(SkillTemplate {
        name: "The Last Word",
        tier: 7,
        class_tags: &["Diplomat"],
        effect: SkillEffect::TheLastWord,
        conditional: false,
        condition: None,
        description: "Permanently end one ongoing conflict.",
    });

    // === MERCHANT ===
    t.push(SkillTemplate {
        name: "Appraise",
        tier: 1,
        class_tags: &["Merchant"],
        effect: SkillEffect::Appraise,
        conditional: false,
        condition: None,
        description: "Reveal true value of items/goods before trading.",
    });
    t.push(SkillTemplate {
        name: "Corner the Market",
        tier: 3,
        class_tags: &["Merchant"],
        effect: SkillEffect::CornerTheMarket { commodity_index: 0, duration_ticks: 300 },
        conditional: false,
        condition: None,
        description: "Become sole buyer of one commodity.",
    });
    t.push(SkillTemplate {
        name: "Forge Trade Route",
        tier: 3,
        class_tags: &["Merchant"],
        effect: SkillEffect::ForgeTradeRoute { duration_ticks: 500, income_per_tick: 0.5 },
        conditional: false,
        condition: None,
        description: "Forge a new trade route between regions.",
    });
    t.push(SkillTemplate {
        name: "Trade Empire",
        tier: 4,
        class_tags: &["Merchant"],
        effect: SkillEffect::TradeEmpire { income_per_tick: 1.0 },
        conditional: false,
        condition: None,
        description: "Passive income from all guild trade routes.",
    });
    t.push(SkillTemplate {
        name: "Golden Touch",
        tier: 5,
        class_tags: &["Merchant"],
        effect: SkillEffect::GoldenTouch,
        conditional: false,
        condition: None,
        description: "Double loot from completed quests.",
    });
    t.push(SkillTemplate {
        name: "Market Maker",
        tier: 6,
        class_tags: &["Merchant"],
        effect: SkillEffect::MarketMaker { commodity_index: 0, duration_ticks: 1000 },
        conditional: false,
        condition: None,
        description: "Set the price of one commodity for an extended duration.",
    });
    t.push(SkillTemplate {
        name: "Wealth of Nations",
        tier: 7,
        class_tags: &["Merchant"],
        effect: SkillEffect::WealthOfNations,
        conditional: false,
        condition: None,
        description: "Guild gold generation doubled permanently.",
    });

    // === SCHOLAR ===
    t.push(SkillTemplate {
        name: "Quick Study",
        tier: 1,
        class_tags: &["Scholar"],
        effect: SkillEffect::QuickStudy,
        conditional: false,
        condition: None,
        description: "Learn from observed events 2x faster.",
    });
    t.push(SkillTemplate {
        name: "Decipher",
        tier: 1,
        class_tags: &["Scholar"],
        effect: SkillEffect::Decipher,
        conditional: false,
        condition: None,
        description: "Read any language or inscription.",
    });
    t.push(SkillTemplate {
        name: "Accelerated Study",
        tier: 1,
        class_tags: &["Scholar"],
        effect: SkillEffect::AcceleratedStudy { duration_ticks: 200 },
        conditional: false,
        condition: None,
        description: "Double research/archive speed temporarily.",
    });
    t.push(SkillTemplate {
        name: "Name the Nameless",
        tier: 3,
        class_tags: &["Scholar"],
        effect: SkillEffect::NameTheNameless,
        conditional: false,
        condition: None,
        description: "Identify and name an unknown threat, granting guild-wide combat bonus.",
    });
    t.push(SkillTemplate {
        name: "Predictive Model",
        tier: 4,
        class_tags: &["Scholar"],
        effect: SkillEffect::PredictiveModel,
        conditional: false,
        condition: None,
        description: "Reduce threat clock growth rate.",
    });
    t.push(SkillTemplate {
        name: "Forbidden Knowledge",
        tier: 5,
        class_tags: &["Scholar"],
        effect: SkillEffect::ForbiddenKnowledge,
        conditional: false,
        condition: None,
        description: "Unlock hidden class requirements.",
    });
    t.push(SkillTemplate {
        name: "Prophetic Vision",
        tier: 5,
        class_tags: &["Scholar"],
        effect: SkillEffect::PropheticVision { events_revealed: 3 },
        conditional: false,
        condition: None,
        description: "See upcoming world events before they happen.",
    });
    t.push(SkillTemplate {
        name: "Rewrite the Record",
        tier: 6,
        class_tags: &["Scholar"],
        effect: SkillEffect::RewriteTheRecord,
        conditional: false,
        condition: None,
        description: "Alter one chronicle entry retroactively.",
    });
    t.push(SkillTemplate {
        name: "Omniscience",
        tier: 7,
        class_tags: &["Scholar"],
        effect: SkillEffect::Omniscience,
        conditional: false,
        condition: None,
        description: "Reveal all hidden information in the campaign.",
    });

    // === ROGUE ===
    t.push(SkillTemplate {
        name: "Silent Movement",
        tier: 1,
        class_tags: &["Rogue", "Scout"],
        effect: SkillEffect::SilentMovement,
        conditional: false,
        condition: None,
        description: "Move silently past patrols.",
    });
    t.push(SkillTemplate {
        name: "Trap Sense",
        tier: 1,
        class_tags: &["Rogue", "Scout"],
        effect: SkillEffect::TrapSense,
        conditional: false,
        condition: None,
        description: "Detect traps before triggering them.",
    });
    t.push(SkillTemplate {
        name: "Forgery",
        tier: 1,
        class_tags: &["Rogue"],
        effect: SkillEffect::Forgery,
        conditional: false,
        condition: None,
        description: "Forge documents to bypass faction restrictions.",
    });
    t.push(SkillTemplate {
        name: "Distraction",
        tier: 1,
        class_tags: &["Rogue"],
        effect: SkillEffect::Distraction { duration_ticks: 50 },
        conditional: false,
        condition: None,
        description: "Create a distraction allowing party to reposition.",
    });
    t.push(SkillTemplate {
        name: "Disinformation",
        tier: 3,
        class_tags: &["Rogue", "Diplomat"],
        effect: SkillEffect::Disinformation { target_faction: 0, duration_ticks: 200 },
        conditional: false,
        condition: None,
        description: "Plant false intel in a rival faction's spy network.",
    });
    t.push(SkillTemplate {
        name: "Safe House",
        tier: 3,
        class_tags: &["Rogue"],
        effect: SkillEffect::SafeHouse { region_id: 0, duration_ticks: 500 },
        conditional: false,
        condition: None,
        description: "Establish a safe house in hostile territory.",
    });
    t.push(SkillTemplate {
        name: "Subvert Loyalty",
        tier: 4,
        class_tags: &["Rogue"],
        effect: SkillEffect::SubvertLoyalty { target_faction: 0 },
        conditional: false,
        condition: None,
        description: "Recruit from a hostile faction's population.",
    });
    t.push(SkillTemplate {
        name: "Intel Gathering",
        tier: 4,
        class_tags: &["Rogue", "Scout"],
        effect: SkillEffect::IntelGathering,
        conditional: false,
        condition: None,
        description: "Steal an enemy's tactical plans before battle.",
    });
    t.push(SkillTemplate {
        name: "Vanish",
        tier: 6,
        class_tags: &["Rogue"],
        effect: SkillEffect::Vanish,
        conditional: false,
        condition: None,
        description: "Erase yourself from all faction records.",
    });

    // === ARTISAN ===
    t.push(SkillTemplate {
        name: "Sapper Eye",
        tier: 1,
        class_tags: &["Artisan"],
        effect: SkillEffect::SapperEye,
        conditional: false,
        condition: None,
        description: "Assess structural weaknesses of fortifications.",
    });
    t.push(SkillTemplate {
        name: "Masterwork Craft",
        tier: 3,
        class_tags: &["Artisan"],
        effect: SkillEffect::MasterworkCraft,
        conditional: false,
        condition: None,
        description: "Craft a masterwork item from lesser materials.",
    });
    t.push(SkillTemplate {
        name: "Master Armorer",
        tier: 4,
        class_tags: &["Artisan"],
        effect: SkillEffect::MasterArmorer,
        conditional: false,
        condition: None,
        description: "Upgrade all equipment in the party by one tier.",
    });
    t.push(SkillTemplate {
        name: "Fortify",
        tier: 5,
        class_tags: &["Artisan", "Guardian"],
        effect: SkillEffect::Fortify { region_id: 0, duration_ticks: 500 },
        conditional: false,
        condition: None,
        description: "Fortify a position, granting massive defense bonus.",
    });
    t.push(SkillTemplate {
        name: "Forge Artifact",
        tier: 6,
        class_tags: &["Artisan"],
        effect: SkillEffect::ForgeArtifact,
        conditional: false,
        condition: None,
        description: "Build a legendary artifact from gathered components.",
    });

    // === COMMANDER ===
    t.push(SkillTemplate {
        name: "Rallying Cry",
        tier: 1,
        class_tags: &["Commander"],
        effect: SkillEffect::RallyingCry { morale_restore: 25.0 },
        conditional: false,
        condition: None,
        description: "Rally broken morale in the field.",
    });
    t.push(SkillTemplate {
        name: "Field Command",
        tier: 4,
        class_tags: &["Commander"],
        effect: SkillEffect::FieldCommand { duration_ticks: 200 },
        conditional: false,
        condition: None,
        description: "Command allied NPCs to follow complex orders.",
    });
    t.push(SkillTemplate {
        name: "Blood Oath",
        tier: 5,
        class_tags: &["Commander", "Warrior"],
        effect: SkillEffect::BloodOath { party_stat_bonus: 0.15 },
        conditional: false,
        condition: None,
        description: "Bond life force to party — they grow stronger while you live.",
    });

    // === SCOUT ===
    t.push(SkillTemplate {
        name: "Shadow Step",
        tier: 1,
        class_tags: &["Scout", "Rogue"],
        effect: SkillEffect::ShadowStep { duration_ticks: 100 },
        conditional: false,
        condition: None,
        description: "Move through hostile territory undetected.",
    });
    t.push(SkillTemplate {
        name: "All-Seeing Eye",
        tier: 6,
        class_tags: &["Scout"],
        effect: SkillEffect::AllSeeingEye { region_id: 0 },
        conditional: false,
        condition: None,
        description: "Reveal all hidden information in a region.",
    });

    // === GUARDIAN ===
    t.push(SkillTemplate {
        name: "Take the Blow",
        tier: 4,
        class_tags: &["Guardian"],
        effect: SkillEffect::TakeTheBlow { target_adventurer: 0, duration_ticks: 100 },
        conditional: false,
        condition: None,
        description: "Redirect all damage from one ally to yourself.",
    });
    t.push(SkillTemplate {
        name: "Sanctuary",
        tier: 5,
        class_tags: &["Guardian", "Healer"],
        effect: SkillEffect::Sanctuary { region_id: 0, duration_ticks: 300 },
        conditional: false,
        condition: None,
        description: "Create a sanctuary zone where no combat can occur.",
    });
    t.push(SkillTemplate {
        name: "Claim by Right",
        tier: 6,
        class_tags: &["Guardian", "Commander"],
        effect: SkillEffect::ClaimByRight { region_id: 0 },
        conditional: false,
        condition: None,
        description: "Claim a region through sheer force of reputation.",
    });

    // === CONDITIONAL CROSS-CLASS TEMPLATES ===
    t.push(SkillTemplate {
        name: "Last Stand",
        tier: 4,
        class_tags: &["Warrior", "Guardian", "Commander"],
        effect: SkillEffect::HoldTheLine,
        conditional: true,
        condition: Some(SkillCondition::Outnumbered),
        description: "When outnumbered, the party fights with desperate ferocity.",
    });
    t.push(SkillTemplate {
        name: "Defiant Surge",
        tier: 3,
        class_tags: &["Warrior", "Guardian"],
        effect: SkillEffect::Unbreakable,
        conditional: true,
        condition: Some(SkillCondition::NearDeath),
        description: "When near death, gain a burst of combat power.",
    });
    t.push(SkillTemplate {
        name: "Crisis Diplomat",
        tier: 4,
        class_tags: &["Diplomat"],
        effect: SkillEffect::CeasefireDeclaration,
        conditional: true,
        condition: Some(SkillCondition::CrisisActive),
        description: "During a crisis, your diplomatic power doubles.",
    });
    t.push(SkillTemplate {
        name: "War Profiteer",
        tier: 4,
        class_tags: &["Merchant"],
        effect: SkillEffect::GoldenTouch,
        conditional: true,
        condition: Some(SkillCondition::AtWar),
        description: "During wartime, extract double value from every trade.",
    });
    t.push(SkillTemplate {
        name: "Lone Wolf",
        tier: 3,
        class_tags: &["Ranger", "Scout", "Rogue"],
        effect: SkillEffect::GhostWalk { duration_ticks: 200 },
        conditional: true,
        condition: Some(SkillCondition::Alone),
        description: "When alone, move through any territory unseen.",
    });
    t.push(SkillTemplate {
        name: "Desperate Remedy",
        tier: 3,
        class_tags: &["Healer"],
        effect: SkillEffect::Resurrection,
        conditional: true,
        condition: Some(SkillCondition::AllyDying),
        description: "When an ally is dying, perform emergency resurrection.",
    });

    // === REWRITE HISTORY (shared T7) ===
    t.push(SkillTemplate {
        name: "Rewrite History",
        tier: 7,
        class_tags: &["Scholar", "Diplomat"],
        effect: SkillEffect::RewriteHistory,
        conditional: false,
        condition: None,
        description: "Rewrite one chronicle entry and make it retroactively true.",
    });

    t
}

// ---------------------------------------------------------------------------
// Template selection logic
// ---------------------------------------------------------------------------

/// Select a skill template for a class at a given tier.
///
/// 1. Filter templates by matching class_tags and tier
/// 2. Remove templates whose name matches an already-granted skill
/// 3. Score remaining by dot product of template's implied behavior with ledger
/// 4. 20% chance to pick a conditional template if available
/// 5. Select top-scoring with lcg_f32 randomness
pub fn select_skill_template(
    class_name: &str,
    class_tags: &[String],
    tier: u32,
    behavior: &BehaviorLedger,
    existing_skills: &[String],
    rng: &mut u64,
) -> Option<SkillTemplate> {
    use crate::headless_campaign::state::lcg_f32;

    let templates = all_templates();

    // 1. Filter by tier and class tag match
    let mut candidates: Vec<SkillTemplate> = templates
        .into_iter()
        .filter(|t| {
            t.tier == tier
                && t.class_tags
                    .iter()
                    .any(|tag| {
                        tag.eq_ignore_ascii_case(class_name)
                            || class_tags.iter().any(|ct| ct.eq_ignore_ascii_case(tag))
                    })
        })
        .collect();

    // 2. Remove already-granted skills
    candidates.retain(|t| !existing_skills.iter().any(|s| s == t.name));

    if candidates.is_empty() {
        return None;
    }

    // 3. Score by behavior dot product
    let behavior_vec = [
        behavior.melee_combat,
        behavior.ranged_combat,
        behavior.healing_given,
        behavior.diplomacy_actions,
        behavior.trades_completed,
        behavior.items_crafted,
        behavior.areas_explored,
        behavior.units_commanded,
        behavior.stealth_actions,
        behavior.research_performed,
        behavior.damage_absorbed,
        behavior.allies_supported,
    ];

    for c in &mut candidates {
        // Score is based on which behavior axes the class tag implies
        let _score = tag_behavior_score(c.class_tags, &behavior_vec);
    }

    // 4. 20% chance to prefer a conditional template
    let prefer_conditional = lcg_f32(rng) < 0.2;

    if prefer_conditional {
        let cond_candidates: Vec<&SkillTemplate> =
            candidates.iter().filter(|t| t.conditional).collect();
        if !cond_candidates.is_empty() {
            let idx = (lcg_f32(rng) * cond_candidates.len() as f32) as usize;
            let idx = idx.min(cond_candidates.len() - 1);
            // Remove from candidates and return
            let chosen_name = cond_candidates[idx].name;
            let pos = candidates.iter().position(|t| t.name == chosen_name);
            if let Some(pos) = pos {
                return Some(candidates.swap_remove(pos));
            }
        }
    }

    // 5. Score and select — weighted random from top half
    let mut scored: Vec<(f32, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let score = tag_behavior_score(t.class_tags, &behavior_vec);
            (score, i)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Pick from top half with randomness
    let top_n = (scored.len() / 2).max(1);
    let pick = (lcg_f32(rng) * top_n as f32) as usize;
    let pick = pick.min(top_n - 1);
    let chosen_idx = scored[pick].1;

    Some(candidates.swap_remove(chosen_idx))
}

/// Score a template's class tags against the behavior vector.
fn tag_behavior_score(tags: &[&str], behavior: &[f32; 12]) -> f32 {
    let mut score = 0.0f32;
    for tag in tags {
        match *tag {
            "Warrior" => score += behavior[0] + behavior[10], // melee + damage_absorbed
            "Ranger" => score += behavior[1] + behavior[6],   // ranged + exploring
            "Healer" => score += behavior[2] + behavior[11],  // healing + supporting
            "Diplomat" => score += behavior[3],                // diplomacy
            "Merchant" => score += behavior[4],                // trades
            "Artisan" => score += behavior[5],                 // crafting
            "Scholar" => score += behavior[9],                 // research
            "Commander" => score += behavior[7],               // commanding
            "Scout" => score += behavior[6] + behavior[8],    // exploring + stealth
            "Rogue" => score += behavior[8],                   // stealth
            "Guardian" => score += behavior[10] + behavior[11], // absorbed + supported
            _ => {}
        }
    }
    score
}
