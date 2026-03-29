//! Campaign systems ported to the world sim delta architecture.
//!
//! Each system reads the WorldState snapshot and pushes WorldDeltas.
//! Systems are registered here and called from the runtime compute phase.

#[allow(unused_imports)]
use super::delta::WorldDelta;
#[allow(unused_imports)]
use super::state::WorldState;

pub mod addiction;
pub mod adventurer_condition;
pub mod adventurer_recovery;
pub mod alliance_blocs;
pub mod auction;
pub mod awakening;
pub mod bankruptcy_cascade;
pub mod battles;
pub mod black_market;
pub mod bloodlines;
pub mod bonds;
pub mod bounties;
pub mod buildings;
pub mod caravans;
pub mod charter;
pub mod choices;
pub mod chronicle;
pub mod civil_war;
pub mod class_progression;
pub mod commodity_futures;
pub mod companions;
pub mod contract_negotiation;
pub mod contracts;
pub mod cooldowns;
pub mod corruption;
pub mod council;
pub mod counter_espionage;
pub mod coup_engine;
pub mod crafting;
pub mod crisis;
pub mod currency_debasement;
pub mod dead_zones;
pub mod defection_cascade;
pub mod demonic_pacts;
pub mod difficulty_scaling;
pub mod diplomacy;
pub mod disease;
pub mod divine_favor;
pub mod dungeons;
pub mod economic_competition;
pub mod economy;
pub mod equipment_durability;
pub mod escalation_protocol;
pub mod espionage;
pub mod exploration;
pub mod faction_ai;
pub mod faction_tech;
pub mod fears;
pub mod festivals;
pub mod folk_hero;
pub mod food;
pub mod geography;
pub mod grudges;
pub mod guild_identity;
pub mod guild_rooms;
pub mod guild_tiers;
pub mod heist_planning;
pub mod hobbies;
pub mod infrastructure;
pub mod insurance;
pub mod interception;
pub mod intrigue;
pub mod journals;
pub mod last_stand;
pub mod legacy_weapons;
pub mod legendary_deeds;
pub mod loans;
pub mod leadership;
pub mod loot;
pub mod marriages;
pub mod memorials;
pub mod mentorship;
pub mod mercenaries;
pub mod messengers;
pub mod migration;
pub mod monster_ecology;
pub mod moods;
pub mod nicknames;
pub mod npc_relationships;
pub mod npc_reputation;
pub mod oaths;
pub mod party_chemistry;
pub mod personal_goals;
pub mod plague_vectors;
pub mod population;
pub mod price_controls;
pub mod prisoners;
pub mod progression;
pub mod propaganda;
pub mod quest_expiry;
pub mod quest_generation;
pub mod quest_lifecycle;
pub mod random_events;
pub mod recruitment;
pub mod religion;
pub mod reputation_decay;
pub mod reputation_stories;
pub mod retirement;
pub mod rival_guild;
pub mod rivalries;
pub mod romance;
pub mod rumors;
pub mod scouting;
pub mod seasonal_quests;
pub mod seasons;
pub mod secrets;
pub mod signal_towers;
pub mod skill_challenges;
pub mod smuggling;
pub mod supply;
pub mod supply_lines;
pub mod terrain_events;
pub mod threat;
pub mod timed_events;
pub mod trade_goods;
pub mod travel;
pub mod traveling_merchants;
pub mod treasure_hunts;
pub mod trophies;
pub mod vassalage;
pub mod verify;
pub mod victory_conditions;
pub mod visions;
pub mod wanted;
pub mod war_exhaustion;
pub mod weather;
pub mod wound_persistence;

macro_rules! run_system {
    ($name:expr, $func:expr, $state:expr, $out:expr) => {{
        #[cfg(feature = "profile-systems")]
        {
            let _t = std::time::Instant::now();
            $func($state, $out);
            let _elapsed = _t.elapsed().as_micros();
            if _elapsed > 100 {
                eprintln!("  SLOW: {} {}µs", $name, _elapsed);
            }
        }
        #[cfg(not(feature = "profile-systems"))]
        {
            $func($state, $out);
        }
    }};
}

/// Run all systems. For single-threaded use.
pub fn compute_all_systems(state: &WorldState, out: &mut Vec<WorldDelta>) {
    compute_settlement_systems(state, out);
    compute_global_systems(state, out);
}

/// Settlement-scoped systems for ALL settlements (single-threaded fallback).
pub fn compute_settlement_systems(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // ===================================================================
    // Phase 1: Settlement-local systems
    // Each gets called once per settlement with the entity slice for that settlement.
    // The systems still take (state, out) and do their own cadence check,
    // but they only iterate entities[range] instead of all entities.
    // ===================================================================

    // For now, settlement-scoped systems still use their global signature.
    // The key optimization: they internally skip entities not at "their" settlement.
    // With the group index sort, entities at the same settlement are contiguous,
    // so the branch predictor and cache prefetcher handle this efficiently —
    // the first non-matching entity exits the inner loop immediately.
    //
    // TODO: refactor hot systems to take (state, settlement_id, entity_range, out)
    // for explicit slice-based iteration. This eliminates even the branch cost.

    // Settlement-scoped entity systems (iterate entities, filter by settlement)
    run_system!("economy", economy::compute_economy, state, out);
    run_system!("food", food::compute_food, state, out);
    run_system!("population", population::compute_population, state, out);
    run_system!("mentorship", mentorship::compute_mentorship, state, out);
    run_system!("adventurer_condition", adventurer_condition::compute_adventurer_condition, state, out);
    run_system!("adventurer_recovery", adventurer_recovery::compute_adventurer_recovery, state, out);
    run_system!("progression", progression::compute_progression, state, out);
    run_system!("class_progression", class_progression::compute_class_progression, state, out);
    run_system!("recruitment", recruitment::compute_recruitment, state, out);
    run_system!("retirement", retirement::compute_retirement, state, out);
    run_system!("hobbies", hobbies::compute_hobbies, state, out);
    run_system!("fears", fears::compute_fears, state, out);
    run_system!("personal_goals", personal_goals::compute_personal_goals, state, out);
    run_system!("journals", journals::compute_journals, state, out);
    run_system!("wound_persistence", wound_persistence::compute_wound_persistence, state, out);
    run_system!("addiction", addiction::compute_addiction, state, out);
    run_system!("equipment_durability", equipment_durability::compute_equipment_durability, state, out);
    run_system!("moods", moods::compute_moods, state, out);
    run_system!("bonds", bonds::compute_bonds, state, out);
    run_system!("npc_relationships", npc_relationships::compute_npc_relationships, state, out);
    run_system!("npc_reputation", npc_reputation::compute_npc_reputation, state, out);
    run_system!("romance", romance::compute_romance, state, out);
    run_system!("rivalries", rivalries::compute_rivalries, state, out);
    run_system!("companions", companions::compute_companions, state, out);
    run_system!("party_chemistry", party_chemistry::compute_party_chemistry, state, out);
    run_system!("nicknames", nicknames::compute_nicknames, state, out);
    run_system!("legendary_deeds", legendary_deeds::compute_legendary_deeds, state, out);
    run_system!("folk_hero", folk_hero::compute_folk_hero, state, out);
    run_system!("memorials", memorials::compute_memorials, state, out);
    run_system!("trophies", trophies::compute_trophies, state, out);
    run_system!("awakening", awakening::compute_awakening, state, out);
    run_system!("visions", visions::compute_visions, state, out);
    run_system!("bloodlines", bloodlines::compute_bloodlines, state, out);
    run_system!("divine_favor", divine_favor::compute_divine_favor, state, out);
    run_system!("religion", religion::compute_religion, state, out);
    run_system!("demonic_pacts", demonic_pacts::compute_demonic_pacts, state, out);
    run_system!("legacy_weapons", legacy_weapons::compute_legacy_weapons, state, out);
    run_system!("cooldowns", cooldowns::compute_cooldowns, state, out);

    // Grid-scoped systems (iterate grids, then entities on those grids)
    run_system!("battles", battles::compute_battles, state, out);
    run_system!("loot", loot::compute_loot, state, out);
    run_system!("last_stand", last_stand::compute_last_stand, state, out);
    run_system!("interception", interception::compute_interception, state, out);
    run_system!("skill_challenges", skill_challenges::compute_skill_challenges, state, out);
    run_system!("dungeons", dungeons::compute_dungeons, state, out);
    run_system!("escalation_protocol", escalation_protocol::compute_escalation_protocol, state, out);

    // Settlement-level systems (iterate settlements, not entities)
    run_system!("trade_goods", trade_goods::compute_trade_goods, state, out);
    run_system!("infrastructure", infrastructure::compute_infrastructure, state, out);
    run_system!("crafting", crafting::compute_crafting, state, out);
    run_system!("buildings", buildings::compute_buildings, state, out);
    run_system!("guild_rooms", guild_rooms::compute_guild_rooms, state, out);
    run_system!("guild_tiers", guild_tiers::compute_guild_tiers, state, out);
    run_system!("festivals", festivals::compute_festivals, state, out);

}

/// Global systems — NOT parallelizable per-settlement.
/// These iterate factions, quests, regions, or cross-settlement entity pairs.
pub fn compute_global_systems(state: &WorldState, out: &mut Vec<WorldDelta>) {

    // Travel / overworld (iterate unaffiliated entities — monsters, travelers)
    run_system!("travel", travel::compute_travel, state, out);
    run_system!("supply", supply::compute_supply, state, out);
    run_system!("caravans", caravans::compute_caravans, state, out);
    run_system!("traveling_merchants", traveling_merchants::compute_traveling_merchants, state, out);
    run_system!("scouting", scouting::compute_scouting, state, out);
    run_system!("messengers", messengers::compute_messengers, state, out);

    // World-level (iterate regions/settlements, not entities)
    run_system!("seasons", seasons::compute_seasons, state, out);
    run_system!("weather", weather::compute_weather, state, out);
    run_system!("monster_ecology", monster_ecology::compute_monster_ecology, state, out);
    run_system!("threat", threat::compute_threat, state, out);
    run_system!("migration", migration::compute_migration, state, out);
    run_system!("dead_zones", dead_zones::compute_dead_zones, state, out);
    run_system!("terrain_events", terrain_events::compute_terrain_events, state, out);
    run_system!("geography", geography::compute_geography, state, out);
    run_system!("supply_lines", supply_lines::compute_supply_lines, state, out);
    run_system!("signal_towers", signal_towers::compute_signal_towers, state, out);
    run_system!("timed_events", timed_events::compute_timed_events, state, out);
    run_system!("random_events", random_events::compute_random_events, state, out);
    run_system!("crisis", crisis::compute_crisis, state, out);
    run_system!("difficulty_scaling", difficulty_scaling::compute_difficulty_scaling, state, out);

    // Quest systems (iterate quest state, not entities)
    run_system!("quest_generation", quest_generation::compute_quest_generation, state, out);
    run_system!("quest_lifecycle", quest_lifecycle::compute_quest_lifecycle, state, out);
    run_system!("quest_expiry", quest_expiry::compute_quest_expiry, state, out);
    run_system!("seasonal_quests", seasonal_quests::compute_seasonal_quests, state, out);
    run_system!("bounties", bounties::compute_bounties, state, out);
    run_system!("treasure_hunts", treasure_hunts::compute_treasure_hunts, state, out);
    run_system!("exploration", exploration::compute_exploration, state, out);
    run_system!("heist_planning", heist_planning::compute_heist_planning, state, out);

    // Faction systems (iterate factions, not entities)
    run_system!("faction_ai", faction_ai::compute_faction_ai, state, out);
    run_system!("diplomacy", diplomacy::compute_diplomacy, state, out);
    run_system!("espionage", espionage::compute_espionage, state, out);
    run_system!("counter_espionage", counter_espionage::compute_counter_espionage, state, out);
    run_system!("war_exhaustion", war_exhaustion::compute_war_exhaustion, state, out);
    run_system!("civil_war", civil_war::compute_civil_war, state, out);
    run_system!("council", council::compute_council, state, out);
    run_system!("coup_engine", coup_engine::compute_coup_engine, state, out);
    run_system!("defection_cascade", defection_cascade::compute_defection_cascade, state, out);
    run_system!("propaganda", propaganda::compute_propaganda, state, out);
    run_system!("alliance_blocs", alliance_blocs::compute_alliance_blocs, state, out);
    run_system!("vassalage", vassalage::compute_vassalage, state, out);
    run_system!("faction_tech", faction_tech::compute_faction_tech, state, out);
    run_system!("mercenaries", mercenaries::compute_mercenaries, state, out);

    // Social global (entity pairs across settlements)
    run_system!("reputation_decay", reputation_decay::compute_reputation_decay, state, out);
    run_system!("reputation_stories", reputation_stories::compute_reputation_stories, state, out);
    run_system!("rumors", rumors::compute_rumors, state, out);
    run_system!("chronicle", chronicle::compute_chronicle, state, out);
    run_system!("marriages", marriages::compute_marriages, state, out);
    run_system!("grudges", grudges::compute_grudges, state, out);
    run_system!("oaths", oaths::compute_oaths, state, out);
    run_system!("intrigue", intrigue::compute_intrigue, state, out);
    run_system!("secrets", secrets::compute_secrets, state, out);
    run_system!("prisoners", prisoners::compute_prisoners, state, out);
    run_system!("wanted", wanted::compute_wanted, state, out);
    run_system!("leadership", leadership::compute_leadership, state, out);
    run_system!("guild_identity", guild_identity::compute_guild_identity, state, out);

    // Economic global
    run_system!("loans", loans::compute_loans, state, out);
    run_system!("insurance", insurance::compute_insurance, state, out);
    run_system!("auction", auction::compute_auction, state, out);
    run_system!("black_market", black_market::compute_black_market, state, out);
    run_system!("commodity_futures", commodity_futures::compute_commodity_futures, state, out);
    run_system!("price_controls", price_controls::compute_price_controls, state, out);
    run_system!("currency_debasement", currency_debasement::compute_currency_debasement, state, out);
    run_system!("smuggling", smuggling::compute_smuggling, state, out);
    run_system!("economic_competition", economic_competition::compute_economic_competition, state, out);
    run_system!("bankruptcy_cascade", bankruptcy_cascade::compute_bankruptcy_cascade, state, out);
    run_system!("contracts", contracts::compute_contracts, state, out);
    run_system!("contract_negotiation", contract_negotiation::compute_contract_negotiation, state, out);
    run_system!("corruption", corruption::compute_corruption, state, out);

    // Meta
    run_system!("charter", charter::compute_charter, state, out);
    run_system!("choices", choices::compute_choices, state, out);
    run_system!("rival_guild", rival_guild::compute_rival_guild, state, out);
    run_system!("victory_conditions", victory_conditions::compute_victory_conditions, state, out);
}
