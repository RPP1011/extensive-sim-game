//! Adventurer nicknames and earned titles — every 500 ticks.
//!
//! Adventurers earn descriptive nicknames from their deeds that affect how
//! NPCs and factions react to them. Positive nicknames grant faction relation
//! bonuses and recruitment attraction, while infamous nicknames add
//! intimidation at the cost of faction standing.
//!
//! Max 3 nicknames per adventurer (keeps the most significant).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    lcg_next, AdventurerStatus, CampaignState, Nickname, NicknameSource,
};

/// Maximum nicknames an adventurer can hold at once.
const MAX_NICKNAMES: usize = 3;

/// Combat kill threshold for a combat nickname.
const COMBAT_KILL_THRESHOLD: u32 = 10;

/// Diplomatic success threshold.
const DIPLOMATIC_THRESHOLD: u32 = 5;

/// Exploration feat threshold.
const EXPLORATION_THRESHOLD: u32 = 8;

/// Near-death survival threshold.
const NEAR_DEATH_THRESHOLD: u32 = 3;

/// Solo kill threshold.
const SOLO_KILL_THRESHOLD: u32 = 50;

/// Gold earned threshold (tracked via history tags).
const TRADE_GOLD_THRESHOLD: u32 = 500;

/// Fear conquered threshold.
const FEAR_CONQUERED_THRESHOLD: u32 = 1;

// ---------------------------------------------------------------------------
// Nickname templates
// ---------------------------------------------------------------------------

const COMBAT_NICKNAMES: &[&str] = &["the Blade", "Ironheart", "Bloodied"];
const DIPLOMATIC_NICKNAMES: &[&str] = &["the Peacemaker", "Silver Tongue", "the Ambassador"];
const EXPLORATION_NICKNAMES: &[&str] = &["Wanderer", "Pathfinder", "the Cartographer"];
const SURVIVAL_NICKNAMES: &[&str] = &["the Undying", "Phoenix", "Lucky"];
const SOLO_NICKNAMES: &[&str] = &["Lone Wolf", "the Solitary"];
const TRADE_NICKNAMES: &[&str] = &["Goldhand", "the Merchant Prince"];
const FEAR_NICKNAMES: &[&str] = &["Fearless", "the Brave"];

/// Pick a deterministic nickname from a template list using RNG state.
fn pick_nickname(templates: &[&str], rng: &mut u64) -> String {
    let idx = lcg_next(rng) as usize % templates.len();
    templates[idx].to_string()
}

/// Check if an adventurer already has a nickname from the given source.
fn has_nickname_from(nicknames: &[Nickname], source: &NicknameSource) -> bool {
    nicknames.iter().any(|n| &n.source == source)
}

/// Evaluate all adventurers and award earned nicknames. Runs every 500 ticks.
pub fn tick_nicknames(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 17 != 0 || state.tick == 0 {
        return;
    }

    // Collect grants first (borrow checker: can't mutate adventurers while iterating).
    let mut grants: Vec<(usize, Nickname)> = Vec::new();

    let mut rng = state.rng;

    for (idx, adv) in state.adventurers.iter().enumerate() {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }

        let tags = &adv.history_tags;
        let nicks = &adv.nicknames;

        // --- Combat kills (combat + high_threat) ---
        let kill_count = tags.get("combat").copied().unwrap_or(0)
            + tags.get("high_threat").copied().unwrap_or(0);
        if kill_count >= COMBAT_KILL_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::CombatDeed)
        {
            let title = pick_nickname(COMBAT_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::CombatDeed,
                    reputation_modifier: 0.05,
                },
            ));
        }

        // --- Diplomatic successes ---
        let diplo_count = tags.get("diplomatic").copied().unwrap_or(0);
        if diplo_count >= DIPLOMATIC_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::DiplomaticAchievement)
        {
            let title = pick_nickname(DIPLOMATIC_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::DiplomaticAchievement,
                    reputation_modifier: 0.08,
                },
            ));
        }

        // --- Exploration feats ---
        let explore_count = tags.get("exploration").copied().unwrap_or(0);
        if explore_count >= EXPLORATION_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::ExplorationFeat)
        {
            let title = pick_nickname(EXPLORATION_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::ExplorationFeat,
                    reputation_modifier: 0.04,
                },
            ));
        }

        // --- Near-death survival ---
        let near_death = tags.get("near_death").copied().unwrap_or(0);
        if near_death >= NEAR_DEATH_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::Sacrifice)
        {
            let title = pick_nickname(SURVIVAL_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::Sacrifice,
                    reputation_modifier: 0.06,
                },
            ));
        }

        // --- Defeated a nemesis ---
        // Check if any defeated nemesis was slain by this adventurer.
        let defeated_nemesis = state.nemeses.iter().find(|n| {
            n.defeated
                && tags.get("nemesis_slayer").copied().unwrap_or(0) > 0
        });
        if let Some(nem) = defeated_nemesis {
            if !has_nickname_from(nicks, &NicknameSource::CombatDeed)
                || !nicks.iter().any(|n| n.title.contains("slayer") || n.title.contains("bane"))
            {
                let title = format!("{}slayer", nem.name.split_whitespace().next().unwrap_or("Nemesis"));
                grants.push((
                    idx,
                    Nickname {
                        title,
                        earned_tick: state.tick,
                        source: NicknameSource::CombatDeed,
                        reputation_modifier: 0.10,
                    },
                ));
            }
        }

        // --- Solo kills ---
        let solo_count = tags.get("solo").copied().unwrap_or(0);
        if solo_count >= SOLO_KILL_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::MysteriousEvent)
        {
            let title = pick_nickname(SOLO_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::MysteriousEvent,
                    reputation_modifier: 0.03,
                },
            ));
        }

        // --- Trade / gold earned ---
        let gold_earned = tags.get("gold_earned").copied().unwrap_or(0);
        if gold_earned >= TRADE_GOLD_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::EconomicFeat)
        {
            let title = pick_nickname(TRADE_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::EconomicFeat,
                    reputation_modifier: 0.05,
                },
            ));
        }

        // --- Fear conquered ---
        let fear_conquered = tags.get("fear_conquered").copied().unwrap_or(0);
        if fear_conquered >= FEAR_CONQUERED_THRESHOLD
            && !has_nickname_from(nicks, &NicknameSource::InfamousAct)
        {
            let title = pick_nickname(FEAR_NICKNAMES, &mut rng);
            grants.push((
                idx,
                Nickname {
                    title,
                    earned_tick: state.tick,
                    source: NicknameSource::InfamousAct,
                    reputation_modifier: 0.07,
                },
            ));
        }
    }

    // Write RNG back
    state.rng = rng;

    // Apply grants
    let tick = state.tick;
    for (idx, nickname) in grants {
        let adv = &mut state.adventurers[idx];
        let adv_id = adv.id;

        // Enforce max nickname limit — keep the most significant ones.
        adv.nicknames.push(nickname.clone());
        if adv.nicknames.len() > MAX_NICKNAMES {
            // Sort by reputation_modifier descending, keep top MAX_NICKNAMES.
            adv.nicknames.sort_by(|a, b| {
                b.reputation_modifier
                    .partial_cmp(&a.reputation_modifier)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            adv.nicknames.truncate(MAX_NICKNAMES);
        }

        // Apply nickname effects to faction relations.
        let modifier = nickname.reputation_modifier;
        if modifier > 0.0 {
            // Positive nicknames boost guild reputation slightly.
            state.guild.reputation += modifier * 2.0;
        } else {
            // Infamous nicknames: lose faction standing but gain intimidation.
            state.guild.reputation += modifier; // negative
        }

        events.push(WorldEvent::NicknameEarned {
            adventurer_id: adv_id,
            title: nickname.title.clone(),
            source: format!("{:?}", nickname.source),
        });

        // Also record in chronicle if it exists.
        if state.chronicle.len() < 100 {
            state.chronicle.push(crate::state::ChronicleEntry {
                tick,
                entry_type: crate::state::ChronicleType::HeroicDeed,
                text: format!(
                    "{} earned the nickname \"{}\"",
                    state.adventurers[idx].name, nickname.title
                ),
                participants: vec![adv_id],
                location_id: None,
                faction_id: None,
                significance: 4.0 + modifier * 20.0,
            });
        }
    }
}
