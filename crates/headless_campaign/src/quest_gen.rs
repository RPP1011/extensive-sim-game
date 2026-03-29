//! Narrative quest grammar walker — procedurally generates quests by walking
//! a grammar tree conditioned on game state and adventurer history.
//!
//! Grammar: Motivation (why) → Objective (what) → Complication (twist) → Reward (payoff)
//!
//! Follows the same pattern as `ability_gen.rs`: probability profiles biased
//! by context, then weighted categorical sampling at each grammar node.

use std::collections::HashMap;

use super::state::*;

// ---------------------------------------------------------------------------
// RNG helpers (matching ability_gen.rs style but using LCG from state)
// ---------------------------------------------------------------------------

/// Float in [0, 1).
fn rf(rng: &mut u64) -> f32 {
    lcg_f32(rng)
}

/// Weighted categorical sample. Returns index.
fn sample_weighted(rng: &mut u64, weights: &[f32]) -> usize {
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return 0;
    }
    let mut target = rf(rng) * total;
    for (i, &w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

// ---------------------------------------------------------------------------
// Quest profile — probability weights for each grammar decision
// ---------------------------------------------------------------------------

/// Probability distributions for quest generation, conditioned on game state.
pub struct QuestProfile {
    /// Motivation: [faction_request, personal_goal, crisis_response, opportunity, npc_plea, guild_ambition]
    pub motivation: [f32; 6],
    /// Quest type: [Combat, Exploration, Diplomatic, Escort, Rescue, Gather]
    pub quest_type: [f32; 6],
    /// Complication: [rival_faction, time_pressure, moral_dilemma, betrayal_risk, environmental, none]
    pub complication: [f32; 6],
    /// Threat range [min, max]
    pub threat: [f32; 2],
    /// Reward bias: [gold_heavy, rep_heavy, faction_heavy, supply, loot, ability]
    pub reward_bias: [f32; 6],
    /// P(embedded choice event)
    pub choice_chance: f32,
}

// ---------------------------------------------------------------------------
// Motivation / complication / reward enum indices
// ---------------------------------------------------------------------------

const MOT_FACTION: usize = 0;
const MOT_PERSONAL: usize = 1;
const MOT_CRISIS: usize = 2;
const MOT_OPPORTUNITY: usize = 3;
const MOT_NPC: usize = 4;
const MOT_GUILD: usize = 5;

const COMP_RIVAL: usize = 0;
const COMP_TIME: usize = 1;
const COMP_MORAL: usize = 2;
const COMP_BETRAYAL: usize = 3;
const COMP_ENV: usize = 4;
const COMP_NONE: usize = 5;

const QT_COMBAT: usize = 0;
const QT_EXPLORATION: usize = 1;
const QT_DIPLOMATIC: usize = 2;
const QT_ESCORT: usize = 3;
const QT_RESCUE: usize = 4;
const QT_GATHER: usize = 5;

const RW_GOLD: usize = 0;
const RW_REP: usize = 1;
const RW_FACTION: usize = 2;
const RW_SUPPLY: usize = 3;
const RW_LOOT: usize = 4;
const RW_ABILITY: usize = 5;

// ---------------------------------------------------------------------------
// Profile construction from game state
// ---------------------------------------------------------------------------

/// Build a QuestProfile from the current game state.
pub fn profile_for_context(state: &CampaignState, adv_id: u32) -> QuestProfile {
    let mut p = QuestProfile {
        motivation: [1.5, 1.0, 0.5, 1.5, 1.0, 1.0],
        quest_type: [2.0, 1.5, 1.0, 1.0, 0.8, 1.2],
        complication: [1.0, 1.0, 0.8, 0.6, 1.0, 2.0],
        threat: [10.0, 50.0],
        reward_bias: [2.0, 1.5, 1.0, 1.0, 1.0, 0.5],
        choice_chance: 0.15,
    };

    // --- Active crises boost crisis_response ---
    if !state.overworld.active_crises.is_empty() {
        let crisis_count = state.overworld.active_crises.len() as f32;
        p.motivation[MOT_CRISIS] += crisis_count * 2.0;
        p.complication[COMP_TIME] += crisis_count * 0.5;
        p.threat[0] += 5.0 * crisis_count;
        p.threat[1] += 10.0 * crisis_count;

        // Crisis-specific biases
        for crisis in &state.overworld.active_crises {
            match crisis {
                ActiveCrisis::Breach { .. } => {
                    p.quest_type[QT_COMBAT] += 2.0;
                    p.complication[COMP_TIME] += 1.0;
                }
                ActiveCrisis::Corruption { .. } => {
                    p.quest_type[QT_GATHER] += 1.5;
                    p.quest_type[QT_RESCUE] += 1.0;
                    p.complication[COMP_ENV] += 1.5;
                }
                ActiveCrisis::SleepingKing { .. } => {
                    p.quest_type[QT_DIPLOMATIC] += 1.0;
                    p.quest_type[QT_ESCORT] += 1.0;
                    p.complication[COMP_BETRAYAL] += 1.0;
                }
                ActiveCrisis::Unifier { .. } => {
                    p.quest_type[QT_DIPLOMATIC] += 2.0;
                    p.complication[COMP_RIVAL] += 1.5;
                }
                ActiveCrisis::Decline { .. } => {
                    p.quest_type[QT_GATHER] += 1.5;
                    p.quest_type[QT_RESCUE] += 1.0;
                    p.motivation[MOT_GUILD] += 1.0;
                    p.complication[COMP_TIME] += 0.5;
                }
            }
        }
    }

    // --- Faction relations ---
    for faction in &state.factions {
        if faction.relationship_to_guild > 60.0 {
            p.motivation[MOT_FACTION] += 1.5;
            p.reward_bias[RW_FACTION] += 1.0;
        }
        if faction.relationship_to_guild < -20.0 {
            p.complication[COMP_RIVAL] += 1.0;
            p.quest_type[QT_COMBAT] += 0.5;
        }
        if faction.diplomatic_stance == DiplomaticStance::AtWar {
            p.quest_type[QT_COMBAT] += 1.5;
            p.complication[COMP_TIME] += 0.5;
            p.threat[0] += 10.0;
            p.threat[1] += 15.0;
        }
    }

    // --- Gold level ---
    if state.guild.gold < 100.0 {
        p.reward_bias[RW_GOLD] += 3.0;
        p.quest_type[QT_GATHER] += 1.5;
        p.motivation[MOT_GUILD] += 1.0;
    } else if state.guild.gold < 300.0 {
        p.reward_bias[RW_GOLD] += 1.0;
        p.quest_type[QT_GATHER] += 0.5;
    }

    // --- Global threat ---
    let threat = state.overworld.global_threat_level;
    if threat > 60.0 {
        p.quest_type[QT_COMBAT] += 2.0;
        p.quest_type[QT_RESCUE] += 1.0;
        p.threat[0] += threat * 0.3;
        p.threat[1] += threat * 0.5;
        p.complication[COMP_NONE] -= 0.5_f32.min(p.complication[COMP_NONE]);
    } else if threat > 30.0 {
        p.quest_type[QT_COMBAT] += 0.5;
        p.threat[0] += threat * 0.1;
        p.threat[1] += threat * 0.2;
    }

    // --- Campaign progress ---
    let progress = state.overworld.campaign_progress;
    if progress > 0.7 {
        p.threat[0] += 15.0;
        p.threat[1] += 25.0;
        p.complication[COMP_NONE] -= 0.5_f32.min(p.complication[COMP_NONE]);
        p.choice_chance += 0.1;
        p.reward_bias[RW_ABILITY] += 1.0;
    }

    // --- Reputation ---
    if state.guild.reputation > 70.0 {
        p.motivation[MOT_FACTION] += 1.0;
        p.motivation[MOT_NPC] += 0.5;
        p.quest_type[QT_DIPLOMATIC] += 0.5;
    }

    // --- Adventurer-specific context ---
    if adv_id > 0 {
        if let Some(adv) = state.adventurers.iter().find(|a| a.id == adv_id) {
            // Low-level adventurers get easier quests
            if adv.level <= 2 {
                p.threat[0] = (p.threat[0] - 10.0).max(5.0);
                p.threat[1] = (p.threat[1] - 15.0).max(20.0);
                p.complication[COMP_NONE] += 1.5;
            }
            // High stress → personal/rescue quests
            if adv.stress > 60.0 {
                p.motivation[MOT_PERSONAL] += 1.0;
                p.quest_type[QT_RESCUE] += 0.5;
            }
            // Apply history tag biases
            apply_quest_history_biases(&mut p, &adv.history_tags);
        }
    }

    p
}

// ---------------------------------------------------------------------------
// History tag biasing
// ---------------------------------------------------------------------------

/// Bias quest profile based on adventurer history tags.
pub fn apply_quest_history_biases(p: &mut QuestProfile, history: &HashMap<String, u32>) {
    let get = |key: &str| -> f32 {
        history.get(key).copied().unwrap_or(0) as f32
    };

    // solo → personal motivation, high threat, loot rewards
    let solo = get("solo");
    if solo > 0.0 {
        let s = (solo / 5.0).min(2.0);
        p.motivation[MOT_PERSONAL] += s * 1.5;
        p.threat[0] += s * 5.0;
        p.threat[1] += s * 8.0;
        p.reward_bias[RW_LOOT] += s * 1.5;
        p.complication[COMP_BETRAYAL] += s * 0.5;
    }

    // diplomatic → diplomatic objectives, faction rewards
    let diplo = get("diplomatic");
    if diplo > 0.0 {
        let s = (diplo / 5.0).min(2.0);
        p.quest_type[QT_DIPLOMATIC] += s * 2.0;
        p.reward_bias[RW_FACTION] += s * 1.5;
        p.motivation[MOT_FACTION] += s;
        p.complication[COMP_MORAL] += s * 0.8;
        p.choice_chance += s * 0.05;
    }

    // exploration → opportunity, exploration, scout/artifact quests
    let explore = get("exploration");
    if explore > 0.0 {
        let s = (explore / 5.0).min(2.0);
        p.motivation[MOT_OPPORTUNITY] += s * 1.5;
        p.quest_type[QT_EXPLORATION] += s * 2.0;
        p.reward_bias[RW_LOOT] += s;
        p.complication[COMP_ENV] += s * 0.5;
    }

    // crisis tags → crisis motivation
    let crisis_blight = get("crisis_blight_prevention");
    if crisis_blight > 0.0 {
        let s = (crisis_blight / 3.0).min(2.0);
        p.motivation[MOT_CRISIS] += s * 2.0;
        p.quest_type[QT_GATHER] += s;
        p.quest_type[QT_RESCUE] += s * 0.5;
        p.complication[COMP_ENV] += s;
    }

    let crisis_breach = get("crisis_breach_defense");
    if crisis_breach > 0.0 {
        let s = (crisis_breach / 3.0).min(2.0);
        p.motivation[MOT_CRISIS] += s * 2.0;
        p.quest_type[QT_COMBAT] += s * 1.5;
        p.complication[COMP_TIME] += s;
    }

    let crisis_king = get("crisis_sleeping_king");
    if crisis_king > 0.0 {
        let s = (crisis_king / 3.0).min(2.0);
        p.motivation[MOT_CRISIS] += s * 1.5;
        p.quest_type[QT_ESCORT] += s;
        p.quest_type[QT_DIPLOMATIC] += s * 0.5;
    }

    // near_death → rescue objectives
    let near_death = get("near_death");
    if near_death > 0.0 {
        let s = (near_death / 3.0).min(2.0);
        p.quest_type[QT_RESCUE] += s * 2.0;
        p.motivation[MOT_NPC] += s;
        p.complication[COMP_BETRAYAL] += s * 0.5;
        p.reward_bias[RW_REP] += s;
    }

    // party_combat → escort/combat, faction motivation
    let party = get("party_combat");
    if party > 0.0 {
        let s = (party / 8.0).min(1.5);
        p.quest_type[QT_COMBAT] += s;
        p.quest_type[QT_ESCORT] += s;
        p.motivation[MOT_FACTION] += s * 0.5;
    }

    // region_defense → faction motivation, loot
    let region_def = get("region_defense");
    if region_def > 0.0 {
        let s = (region_def / 8.0).min(1.5);
        p.motivation[MOT_FACTION] += s;
        p.quest_type[QT_COMBAT] += s * 0.5;
        p.reward_bias[RW_FACTION] += s;
    }

    // rescue → more rescue
    let rescue = get("rescue");
    if rescue > 0.0 {
        let s = (rescue / 5.0).min(2.0);
        p.quest_type[QT_RESCUE] += s * 1.5;
        p.motivation[MOT_NPC] += s;
        p.reward_bias[RW_REP] += s;
    }

    // gather → gather + supply rewards
    let gather = get("gather");
    if gather > 0.0 {
        let s = (gather / 8.0).min(1.5);
        p.quest_type[QT_GATHER] += s;
        p.reward_bias[RW_SUPPLY] += s * 1.5;
        p.motivation[MOT_GUILD] += s * 0.5;
    }

    // high_threat → combat + loot
    let high_threat = get("high_threat");
    if high_threat > 0.0 {
        let s = (high_threat / 8.0).min(1.5);
        p.quest_type[QT_COMBAT] += s;
        p.threat[0] += s * 5.0;
        p.reward_bias[RW_LOOT] += s;
    }
}

// ---------------------------------------------------------------------------
// Narrative templates
// ---------------------------------------------------------------------------

/// Motivation label for template matching.
const MOTIVATION_NAMES: [&str; 6] = [
    "faction_request",
    "personal_goal",
    "crisis_response",
    "opportunity",
    "npc_plea",
    "guild_ambition",
];

/// Complication label for template matching.
const COMPLICATION_NAMES: [&str; 6] = [
    "rival_faction",
    "time_pressure",
    "moral_dilemma",
    "betrayal_risk",
    "environmental",
    "none",
];

/// Template: (motivation, quest_type_idx, complication, template_string)
/// Variables: {faction}, {location}, {region}, {adventurer}, {threat_desc},
///            {crisis}, {hostile_faction}, {deadline_desc}
const TEMPLATES: &[(&str, usize, &str, &str)] = &[
    // --- faction_request ---
    ("faction_request", QT_COMBAT, "time_pressure",
     "{faction} urgently requests aid — {threat_desc} threaten {location}. The guild must respond quickly or risk losing an ally."),
    ("faction_request", QT_COMBAT, "rival_faction",
     "{faction} calls for defenders at {location}, but {hostile_faction} agents have been spotted nearby, complicating the approach."),
    ("faction_request", QT_COMBAT, "none",
     "{faction} has posted a bounty: {threat_desc} have been raiding near {location}. Clear them out."),
    ("faction_request", QT_COMBAT, "environmental",
     "{faction} needs a warband at {location}, but treacherous terrain and foul weather make the journey perilous."),
    ("faction_request", QT_DIPLOMATIC, "moral_dilemma",
     "{faction} asks the guild to broker a deal at {location} — but the terms may not sit well with everyone involved."),
    ("faction_request", QT_DIPLOMATIC, "none",
     "{faction} seeks an envoy to {location} to negotiate trade agreements. A diplomatic touch is needed."),
    ("faction_request", QT_DIPLOMATIC, "betrayal_risk",
     "{faction} wants the guild to represent them at talks in {location}, but whispers suggest a trap."),
    ("faction_request", QT_GATHER, "none",
     "{faction} requests a supply run to {location}. The goods are needed to shore up defenses in {region}."),
    ("faction_request", QT_GATHER, "time_pressure",
     "{faction} desperately needs supplies from {location} before the trade routes close for the season."),
    ("faction_request", QT_RESCUE, "time_pressure",
     "{faction} reports missing scouts near {location}. They must be found before the {threat_desc} close in."),
    ("faction_request", QT_RESCUE, "none",
     "{faction} asks for help retrieving a stranded patrol from {location}."),
    ("faction_request", QT_ESCORT, "rival_faction",
     "{faction} needs an escort for a caravan heading through {location}, where {hostile_faction} raiders operate."),
    ("faction_request", QT_ESCORT, "none",
     "{faction} has a dignitary traveling to {location} and needs reliable guards."),
    ("faction_request", QT_EXPLORATION, "none",
     "{faction} has heard rumors of ancient ruins near {location} and wants the guild to investigate."),

    // --- personal_goal ---
    ("personal_goal", QT_EXPLORATION, "none",
     "{adventurer} follows old maps to {location}, drawn by stories of what lies buried there."),
    ("personal_goal", QT_EXPLORATION, "moral_dilemma",
     "{adventurer} has a lead on a relic at {location}, but recovering it means disturbing sacred ground."),
    ("personal_goal", QT_EXPLORATION, "environmental",
     "{adventurer} is determined to reach {location} despite the harsh terrain — something personal calls them there."),
    ("personal_goal", QT_COMBAT, "none",
     "{adventurer} has unfinished business at {location}. {threat_desc} stand between them and closure."),
    ("personal_goal", QT_COMBAT, "betrayal_risk",
     "{adventurer} returns to {location} to settle an old score, but a former ally may have switched sides."),
    ("personal_goal", QT_DIPLOMATIC, "none",
     "{adventurer} seeks to mend a broken alliance at {location}, hoping words can do what swords cannot."),
    ("personal_goal", QT_RESCUE, "none",
     "{adventurer} has learned of a friend trapped near {location} and won't rest until they're safe."),
    ("personal_goal", QT_GATHER, "environmental",
     "{adventurer} knows of rare materials at {location}, but the land itself guards them jealously."),

    // --- crisis_response ---
    ("crisis_response", QT_COMBAT, "time_pressure",
     "The {crisis} demands immediate action! {threat_desc} mass near {location} — delay means devastation."),
    ("crisis_response", QT_COMBAT, "rival_faction",
     "The {crisis} has drawn {hostile_faction} opportunists to {location}. Fight through them to reach the crisis front."),
    ("crisis_response", QT_COMBAT, "none",
     "The {crisis} has spawned {threat_desc} near {location}. The guild must deal with them before they spread."),
    ("crisis_response", QT_RESCUE, "environmental",
     "The {crisis} has trapped survivors in {region}. Treacherous conditions make the rescue perilous."),
    ("crisis_response", QT_RESCUE, "rival_faction",
     "The {crisis} has trapped survivors in {region}. {hostile_faction} controls the only route — fight through or negotiate passage?"),
    ("crisis_response", QT_GATHER, "time_pressure",
     "The {crisis} requires rare reagents found near {location}. Every tick counts."),
    ("crisis_response", QT_GATHER, "none",
     "Containing the {crisis} requires supplies from {location}. A straightforward but vital mission."),
    ("crisis_response", QT_DIPLOMATIC, "moral_dilemma",
     "The {crisis} has fractured alliances. A delegation to {location} could rally support — but at what cost?"),
    ("crisis_response", QT_ESCORT, "time_pressure",
     "Critical supplies for the {crisis} front must be escorted through {location} before it's too late."),
    ("crisis_response", QT_EXPLORATION, "environmental",
     "The {crisis} may have an ancient solution buried at {location}. The path is treacherous."),

    // --- opportunity ---
    ("opportunity", QT_EXPLORATION, "none",
     "Scouts report unexplored territory near {location} in {region}. Adventure and profit await the bold."),
    ("opportunity", QT_EXPLORATION, "environmental",
     "A newly accessible passage at {location} promises discoveries, but the terrain is unforgiving."),
    ("opportunity", QT_EXPLORATION, "rival_faction",
     "Reports suggest treasure at {location}, but {hostile_faction} scouts were seen heading the same way."),
    ("opportunity", QT_COMBAT, "none",
     "A {threat_desc} lair has been spotted near {location}. Its hoard could fund the guild for months."),
    ("opportunity", QT_COMBAT, "betrayal_risk",
     "An informant offers the location of a {threat_desc} stronghold near {location}. Too good to be true?"),
    ("opportunity", QT_GATHER, "none",
     "Rich deposits have been found near {location}. A well-organized expedition could reap great rewards."),
    ("opportunity", QT_GATHER, "rival_faction",
     "Valuable resources at {location} — but {hostile_faction} has staked a claim too."),
    ("opportunity", QT_DIPLOMATIC, "none",
     "An opportunity to forge new trade connections at {location} has presented itself."),

    // --- npc_plea ---
    ("npc_plea", QT_RESCUE, "time_pressure",
     "A desperate traveler begs for help — their family is trapped near {location} with {threat_desc} closing in."),
    ("npc_plea", QT_RESCUE, "none",
     "A village elder at {location} pleads for the guild to find their missing people."),
    ("npc_plea", QT_RESCUE, "moral_dilemma",
     "A plea from {location}: save the village from {threat_desc}, but the villagers harbor a dangerous secret."),
    ("npc_plea", QT_COMBAT, "none",
     "A merchant reports {threat_desc} blocking the roads near {location} and offers payment for their removal."),
    ("npc_plea", QT_COMBAT, "betrayal_risk",
     "A stranger offers a reward for clearing {threat_desc} from {location}, but their motives seem unclear."),
    ("npc_plea", QT_ESCORT, "environmental",
     "A refugee column needs safe passage through {location}. The terrain is as dangerous as any enemy."),
    ("npc_plea", QT_ESCORT, "none",
     "A traveling scholar needs protection on the road to {location}. The pay is modest but the knowledge shared may be invaluable."),
    ("npc_plea", QT_GATHER, "none",
     "A healer in {region} needs rare herbs from {location} to treat a spreading illness."),
    ("npc_plea", QT_DIPLOMATIC, "moral_dilemma",
     "Two families at {location} are feuding. An outsider's perspective might help — or make things worse."),

    // --- guild_ambition ---
    ("guild_ambition", QT_EXPLORATION, "none",
     "The guild seeks to expand its reach. Charting {location} in {region} would open new opportunities."),
    ("guild_ambition", QT_EXPLORATION, "environmental",
     "Establishing a waypoint at {location} would be valuable, but the area is inhospitable."),
    ("guild_ambition", QT_COMBAT, "none",
     "The guild's reputation grows. Proving the guild can handle {threat_desc} at {location} would cement it."),
    ("guild_ambition", QT_COMBAT, "time_pressure",
     "A rival guild is eyeing {location}. Clearing the {threat_desc} there first will secure the guild's claim."),
    ("guild_ambition", QT_GATHER, "none",
     "Building reserves is prudent. A supply expedition to {location} will strengthen the guild's position."),
    ("guild_ambition", QT_GATHER, "rival_faction",
     "The guild needs resources from {location}, but {hostile_faction} trade embargoes complicate matters."),
    ("guild_ambition", QT_DIPLOMATIC, "none",
     "Forging an alliance with the people of {location} would advance the guild's strategic goals."),
    ("guild_ambition", QT_DIPLOMATIC, "betrayal_risk",
     "A diplomatic overture to {location} could yield great rewards — if the other side negotiates in good faith."),
    ("guild_ambition", QT_ESCORT, "none",
     "The guild has arranged a trade deal. Escorting goods to {location} will formalize the relationship."),
    ("guild_ambition", QT_RESCUE, "none",
     "Rescuing stranded travelers near {location} would boost the guild's reputation across {region}."),
];

/// Threat descriptions scaled by threat level.
fn threat_description(threat: f32, rng: &mut u64) -> &'static str {
    if threat < 20.0 {
        let opts = ["bandits", "wild beasts", "desperate scavengers", "feral creatures", "highway robbers"];
        opts[(lcg_next(rng) as usize) % opts.len()]
    } else if threat < 40.0 {
        let opts = ["a war party", "undead patrols", "orc raiders", "a mercenary band", "roaming marauders"];
        opts[(lcg_next(rng) as usize) % opts.len()]
    } else if threat < 65.0 {
        let opts = ["an ancient beast", "a dark cult", "a warlord's host", "corrupted guardians", "a siege force"];
        opts[(lcg_next(rng) as usize) % opts.len()]
    } else {
        let opts = ["a dragon", "a demon lord's vanguard", "an elder horror", "a titan", "an army of the damned"];
        opts[(lcg_next(rng) as usize) % opts.len()]
    }
}

/// Crisis name from the active crisis variant.
fn crisis_name(crisis: &ActiveCrisis) -> &'static str {
    match crisis {
        ActiveCrisis::SleepingKing { .. } => "Sleeping King crisis",
        ActiveCrisis::Breach { .. } => "Breach",
        ActiveCrisis::Corruption { .. } => "Corruption",
        ActiveCrisis::Unifier { .. } => "Unifier crisis",
        ActiveCrisis::Decline { .. } => "Great Decline",
    }
}

/// Pick a deadline description for time_pressure complications.
fn deadline_description(rng: &mut u64) -> &'static str {
    let opts = [
        "before the next moon",
        "within days",
        "before reinforcements arrive",
        "before the routes are cut off",
        "before the situation deteriorates further",
    ];
    opts[(lcg_next(rng) as usize) % opts.len()]
}

/// Fill template variables from game state context.
fn fill_template(
    template: &str,
    faction_name: &str,
    location_name: &str,
    region_name: &str,
    adventurer_name: &str,
    threat_desc: &str,
    crisis_desc: &str,
    hostile_faction_name: &str,
    _deadline_desc: &str,
) -> String {
    template
        .replace("{faction}", faction_name)
        .replace("{location}", location_name)
        .replace("{region}", region_name)
        .replace("{adventurer}", adventurer_name)
        .replace("{threat_desc}", threat_desc)
        .replace("{crisis}", crisis_desc)
        .replace("{hostile_faction}", hostile_faction_name)
        .replace("{deadline_desc}", _deadline_desc)
}

/// Generate a narrative description by finding a matching template.
fn generate_description(
    motivation_idx: usize,
    quest_type_idx: usize,
    complication_idx: usize,
    state: &CampaignState,
    adv_id: u32,
    threat: f32,
    source_faction_id: Option<usize>,
    location_name: &str,
    rng: &mut u64,
) -> String {
    let mot_name = MOTIVATION_NAMES[motivation_idx];
    let comp_name = COMPLICATION_NAMES[complication_idx];

    // Gather context strings
    let faction_name = source_faction_id
        .and_then(|id| state.factions.iter().find(|f| f.id == id))
        .map(|f| f.name.as_str())
        .unwrap_or("A distant ally");

    let region_name = if state.overworld.regions.is_empty() {
        "the frontier"
    } else {
        let idx = (lcg_next(rng) as usize) % state.overworld.regions.len();
        state.overworld.regions[idx].name.as_str()
    };

    let adventurer_name = if adv_id > 0 {
        state
            .adventurers
            .iter()
            .find(|a| a.id == adv_id)
            .map(|a| a.name.as_str())
            .unwrap_or("An adventurer")
    } else {
        "The guild"
    };

    let threat_desc = threat_description(threat, rng);

    let crisis_desc = if state.overworld.active_crises.is_empty() {
        "ongoing crisis"
    } else {
        let idx = (lcg_next(rng) as usize) % state.overworld.active_crises.len();
        crisis_name(&state.overworld.active_crises[idx])
    };

    // Pick a hostile faction (one with negative relation or random)
    let hostile_faction_name = state
        .factions
        .iter()
        .find(|f| f.relationship_to_guild < 0.0 && Some(f.id) != source_faction_id)
        .or_else(|| {
            state
                .factions
                .iter()
                .find(|f| Some(f.id) != source_faction_id)
        })
        .map(|f| f.name.as_str())
        .unwrap_or("a hostile faction");

    let dl_desc = deadline_description(rng);

    // Find matching templates: exact (mot, qt, comp) first, then (mot, qt, any), then (any, qt, comp), then fallback
    let mut candidates: Vec<usize> = Vec::new();

    // Exact match
    for (i, &(m, qt, c, _)) in TEMPLATES.iter().enumerate() {
        if m == mot_name && qt == quest_type_idx && c == comp_name {
            candidates.push(i);
        }
    }

    // Relaxed: match motivation + quest type, any complication
    if candidates.is_empty() {
        for (i, &(m, qt, _, _)) in TEMPLATES.iter().enumerate() {
            if m == mot_name && qt == quest_type_idx {
                candidates.push(i);
            }
        }
    }

    // Relaxed: match quest type + complication, any motivation
    if candidates.is_empty() {
        for (i, &(_, qt, c, _)) in TEMPLATES.iter().enumerate() {
            if qt == quest_type_idx && c == comp_name {
                candidates.push(i);
            }
        }
    }

    // Fallback: match quest type only
    if candidates.is_empty() {
        for (i, &(_, qt, _, _)) in TEMPLATES.iter().enumerate() {
            if qt == quest_type_idx {
                candidates.push(i);
            }
        }
    }

    // Last resort: any template
    if candidates.is_empty() {
        candidates.push(0);
    }

    let template_idx = candidates[(lcg_next(rng) as usize) % candidates.len()];
    let template = TEMPLATES[template_idx].3;

    fill_template(
        template,
        faction_name,
        location_name,
        region_name,
        adventurer_name,
        threat_desc,
        crisis_desc,
        hostile_faction_name,
        dl_desc,
    )
}

// ---------------------------------------------------------------------------
// Choice event generation
// ---------------------------------------------------------------------------

/// Generate an optional embedded choice event for a quest.
fn generate_choice_event(
    quest_id: u32,
    _motivation_idx: usize,
    complication_idx: usize,
    quest_type_idx: usize,
    threat: f32,
    source_faction_id: Option<usize>,
    state: &CampaignState,
    rng: &mut u64,
) -> ChoiceEvent {
    let event_id = lcg_next(rng);

    // Generate choices based on complication + motivation
    let (prompt, options) = match complication_idx {
        COMP_MORAL => {
            let prompt = "A moral crossroads: the mission's objective conflicts with local customs. How do you proceed?".to_string();
            let options = vec![
                ChoiceOption {
                    label: "Respect local customs".to_string(),
                    description: "Honor the traditions, even if it means less reward.".to_string(),
                    effects: vec![
                        ChoiceEffect::Reputation(5.0),
                        ChoiceEffect::Gold(-20.0),
                    ],
                },
                ChoiceOption {
                    label: "Press forward regardless".to_string(),
                    description: "Complete the mission as planned. Efficiency over sentiment.".to_string(),
                    effects: vec![
                        ChoiceEffect::Gold(15.0),
                        ChoiceEffect::Reputation(-3.0),
                    ],
                },
                ChoiceOption {
                    label: "Find a compromise".to_string(),
                    description: "Take extra time to satisfy both the mission and local concerns.".to_string(),
                    effects: vec![
                        ChoiceEffect::Reputation(2.0),
                    ],
                },
            ];
            (prompt, options)
        }
        COMP_BETRAYAL => {
            let prompt = "Your contact's loyalties are suspect. Intelligence suggests a double-cross.".to_string();
            let options = vec![
                ChoiceOption {
                    label: "Proceed with caution".to_string(),
                    description: "Continue but prepare for betrayal. Slower but safer.".to_string(),
                    effects: vec![
                        ChoiceEffect::Reputation(2.0),
                    ],
                },
                ChoiceOption {
                    label: "Confront them directly".to_string(),
                    description: "Force the issue now. Risk a fight but resolve the uncertainty.".to_string(),
                    effects: vec![
                        ChoiceEffect::ModifyQuestThreat {
                            quest_id,
                            multiplier: 1.3,
                        },
                        ChoiceEffect::Gold(10.0),
                    ],
                },
                ChoiceOption {
                    label: "Cut them out entirely".to_string(),
                    description: "Go it alone. Harder mission but no risk of betrayal.".to_string(),
                    effects: vec![
                        ChoiceEffect::ModifyQuestThreat {
                            quest_id,
                            multiplier: 1.5,
                        },
                        ChoiceEffect::Reputation(3.0),
                    ],
                },
            ];
            (prompt, options)
        }
        COMP_RIVAL => {
            let hostile = source_faction_id
                .and_then(|sid| {
                    state
                        .factions
                        .iter()
                        .find(|f| f.relationship_to_guild < 0.0 && f.id != sid)
                })
                .or_else(|| state.factions.iter().find(|f| f.relationship_to_guild < 0.0));

            let hostile_name = hostile.map(|f| f.name.as_str()).unwrap_or("A rival faction");
            let hostile_id = hostile.map(|f| f.id).unwrap_or(0);

            let prompt = format!("{} forces have arrived at the objective. The situation is tense.", hostile_name);
            let options = vec![
                ChoiceOption {
                    label: "Negotiate".to_string(),
                    description: "Try to reach an agreement. May improve relations.".to_string(),
                    effects: vec![
                        ChoiceEffect::FactionRelation {
                            faction_id: hostile_id,
                            delta: 10.0,
                        },
                        ChoiceEffect::Gold(-10.0),
                    ],
                },
                ChoiceOption {
                    label: "Fight for control".to_string(),
                    description: "Drive them off. Risky but decisive.".to_string(),
                    effects: vec![
                        ChoiceEffect::FactionRelation {
                            faction_id: hostile_id,
                            delta: -15.0,
                        },
                        ChoiceEffect::ModifyQuestThreat {
                            quest_id,
                            multiplier: 1.4,
                        },
                        ChoiceEffect::Reputation(3.0),
                    ],
                },
                ChoiceOption {
                    label: "Avoid them".to_string(),
                    description: "Find another approach. Slower but avoids conflict.".to_string(),
                    effects: vec![
                        ChoiceEffect::Reputation(-1.0),
                    ],
                },
            ];
            (prompt, options)
        }
        COMP_TIME => {
            let prompt = "Time is running out. An opportunity presents itself but demands sacrifice.".to_string();
            let options = vec![
                ChoiceOption {
                    label: "Rush the objective".to_string(),
                    description: "Push hard to meet the deadline. Exhausting but effective.".to_string(),
                    effects: vec![
                        ChoiceEffect::Reputation(3.0),
                        ChoiceEffect::Supplies(-15.0),
                    ],
                },
                ChoiceOption {
                    label: "Accept the delay".to_string(),
                    description: "Take the steady approach and accept some consequences.".to_string(),
                    effects: vec![
                        ChoiceEffect::Reputation(-2.0),
                    ],
                },
            ];
            (prompt, options)
        }
        _ => {
            // Generic choice based on quest type
            match quest_type_idx {
                QT_COMBAT => {
                    let prompt = "Your party encounters an unexpected obstacle on the way to the fight.".to_string();
                    let options = vec![
                        ChoiceOption {
                            label: "Charge through".to_string(),
                            description: "Brute force through the obstacle.".to_string(),
                            effects: vec![
                                ChoiceEffect::ModifyQuestThreat {
                                    quest_id,
                                    multiplier: 0.9,
                                },
                                ChoiceEffect::Supplies(-10.0),
                            ],
                        },
                        ChoiceOption {
                            label: "Find a way around".to_string(),
                            description: "Take a longer path to avoid unnecessary risk.".to_string(),
                            effects: vec![
                                ChoiceEffect::Supplies(-5.0),
                            ],
                        },
                    ];
                    (prompt, options)
                }
                QT_EXPLORATION => {
                    let prompt = "The expedition discovers a hidden passage. It could lead to treasure — or danger.".to_string();
                    let options = vec![
                        ChoiceOption {
                            label: "Explore the passage".to_string(),
                            description: "Risk the unknown for potential rewards.".to_string(),
                            effects: vec![
                                ChoiceEffect::Gold(25.0),
                                ChoiceEffect::ModifyQuestThreat {
                                    quest_id,
                                    multiplier: 1.2,
                                },
                            ],
                        },
                        ChoiceOption {
                            label: "Mark it for later".to_string(),
                            description: "Note the location and continue the current objective.".to_string(),
                            effects: vec![
                                ChoiceEffect::Reputation(1.0),
                            ],
                        },
                    ];
                    (prompt, options)
                }
                _ => {
                    let prompt = "An unexpected situation demands a decision.".to_string();
                    let options = vec![
                        ChoiceOption {
                            label: "Take the bold path".to_string(),
                            description: "Higher risk, higher reward.".to_string(),
                            effects: vec![
                                ChoiceEffect::Gold(threat * 0.3),
                                ChoiceEffect::Reputation(2.0),
                            ],
                        },
                        ChoiceOption {
                            label: "Play it safe".to_string(),
                            description: "Minimize risk and stay on course.".to_string(),
                            effects: vec![
                                ChoiceEffect::Reputation(1.0),
                            ],
                        },
                    ];
                    (prompt, options)
                }
            }
        }
    };

    ChoiceEvent {
        id: event_id,
        source: ChoiceSource::QuestBranch { quest_id },
        prompt,
        options,
        default_option: 0,
        deadline_ms: Some(state.elapsed_ms + 50_000),
        created_at_ms: state.elapsed_ms,
    }
}

// ---------------------------------------------------------------------------
// Main generation entry point
// ---------------------------------------------------------------------------

/// Generate a narrative quest by walking the grammar tree.
///
/// Returns (QuestRequest, Option<ChoiceEvent>, narrative_description).
/// If `adv_id` is 0, generates a guild-wide quest.
pub fn generate_quest(
    state: &CampaignState,
    adv_id: u32,
    rng: &mut u64,
) -> (QuestRequest, Option<ChoiceEvent>, String) {
    let p = profile_for_context(state, adv_id);

    // 1. Sample motivation
    let motivation_idx = sample_weighted(rng, &p.motivation);

    // 2. Sample quest type
    let quest_type_idx = sample_weighted(rng, &p.quest_type);
    let quest_type = match quest_type_idx {
        0 => QuestType::Combat,
        1 => QuestType::Exploration,
        2 => QuestType::Diplomatic,
        3 => QuestType::Escort,
        4 => QuestType::Rescue,
        _ => QuestType::Gather,
    };

    // 3. Sample complication
    let complication_idx = sample_weighted(rng, &p.complication);

    // 4. Calculate threat from profile range + game state
    let threat_level = {
        let base = p.threat[0] + rf(rng) * (p.threat[1] - p.threat[0]);
        let cfg = &state.config.quest_generation;
        let progress_scaling =
            1.0 + state.overworld.campaign_progress * cfg.progress_threat_scaling;
        let (threat_min, threat_max) = if cfg.min_threat <= cfg.max_threat {
            (cfg.min_threat, cfg.max_threat)
        } else {
            (cfg.max_threat, cfg.min_threat)
        };
        (base * progress_scaling).clamp(threat_min, threat_max)
    };

    // 5. Pick a location
    let (target_pos, source_area, distance, location_name) = if state.overworld.locations.is_empty()
    {
        let x = rf(rng) * 100.0 - 50.0;
        let y = rf(rng) * 100.0 - 50.0;
        let dx = x - state.guild.base.position.0;
        let dy = y - state.guild.base.position.1;
        (
            (x, y),
            None,
            (dx * dx + dy * dy).sqrt(),
            "the wilds".to_string(),
        )
    } else {
        let idx = (lcg_next(rng) as usize) % state.overworld.locations.len();
        let loc = &state.overworld.locations[idx];
        let dx = loc.position.0 - state.guild.base.position.0;
        let dy = loc.position.1 - state.guild.base.position.1;
        (
            loc.position,
            Some(loc.id),
            (dx * dx + dy * dy).sqrt(),
            loc.name.clone(),
        )
    };

    // 6. Pick source faction (biased by motivation)
    let source_faction = if state.factions.is_empty() {
        None
    } else if motivation_idx == MOT_FACTION {
        // Prefer friendly factions
        let friendly: Vec<&FactionState> = state
            .factions
            .iter()
            .filter(|f| f.relationship_to_guild > 0.0)
            .collect();
        if friendly.is_empty() {
            let idx = (lcg_next(rng) as usize) % state.factions.len();
            Some(state.factions[idx].id)
        } else {
            let idx = (lcg_next(rng) as usize) % friendly.len();
            Some(friendly[idx].id)
        }
    } else {
        let idx = (lcg_next(rng) as usize) % state.factions.len();
        Some(state.factions[idx].id)
    };

    // 7. Calculate rewards based on reward_bias + threat
    let cfg = &state.config.quest_generation;
    let reward_idx = sample_weighted(rng, &p.reward_bias);
    let base_gold = threat_level * cfg.gold_per_threat + rf(rng) * cfg.gold_variance;
    let base_rep = (threat_level * cfg.rep_per_threat).min(cfg.max_rep_reward);

    let reward = QuestReward {
        gold: match reward_idx {
            RW_GOLD => base_gold * 1.5,
            RW_SUPPLY => base_gold * 0.5,
            _ => base_gold,
        },
        reputation: match reward_idx {
            RW_REP => base_rep * 1.5,
            RW_GOLD => base_rep * 0.7,
            _ => base_rep,
        },
        relation_faction_id: source_faction,
        relation_change: match reward_idx {
            RW_FACTION => 10.0,
            _ => {
                if source_faction.is_some() {
                    5.0
                } else {
                    0.0
                }
            }
        },
        supply_reward: match reward_idx {
            RW_SUPPLY => cfg.gather_supply_reward * 1.5,
            _ => {
                if matches!(quest_type, QuestType::Gather) {
                    cfg.gather_supply_reward
                } else {
                    0.0
                }
            }
        },
        potential_loot: matches!(reward_idx, RW_LOOT)
            || matches!(quest_type, QuestType::Combat | QuestType::Exploration),
    };

    // 8. Generate narrative description
    let description = generate_description(
        motivation_idx,
        quest_type_idx,
        complication_idx,
        state,
        adv_id,
        threat_level,
        source_faction,
        &location_name,
        rng,
    );

    // 9. Optionally generate a ChoiceEvent
    let choice = if rf(rng) < p.choice_chance {
        Some(generate_choice_event(
            0, // quest_id will be assigned by the caller
            motivation_idx,
            complication_idx,
            quest_type_idx,
            threat_level,
            source_faction,
            state,
            rng,
        ))
    } else {
        None
    };

    let request = QuestRequest {
        id: 0, // assigned by caller
        source_faction_id: source_faction,
        source_area_id: source_area,
        quest_type,
        threat_level,
        reward,
        distance,
        target_position: target_pos,
        deadline_ms: state.elapsed_ms + cfg.quest_deadline_ms,
        description,
        arrived_at_ms: state.elapsed_ms,
    };

    (request, choice, location_name)
}
