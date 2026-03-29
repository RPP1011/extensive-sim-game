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

/// Run all registered world sim systems, pushing deltas into `out`.
pub fn compute_all_systems(state: &WorldState, out: &mut Vec<WorldDelta>) {
    // Core tick systems
    cooldowns::compute_cooldowns(state, out);
    interception::compute_interception(state, out);
    battles::compute_battles(state, out);
    loot::compute_loot(state, out);
    last_stand::compute_last_stand(state, out);
    travel::compute_travel(state, out);
    supply::compute_supply(state, out);
    scouting::compute_scouting(state, out);
    messengers::compute_messengers(state, out);

    // Economy
    economy::compute_economy(state, out);
    food::compute_food(state, out);
    trade_goods::compute_trade_goods(state, out);
    infrastructure::compute_infrastructure(state, out);
    population::compute_population(state, out);
    caravans::compute_caravans(state, out);
    crafting::compute_crafting(state, out);
    loans::compute_loans(state, out);
    insurance::compute_insurance(state, out);
    auction::compute_auction(state, out);
    black_market::compute_black_market(state, out);
    commodity_futures::compute_commodity_futures(state, out);
    price_controls::compute_price_controls(state, out);
    currency_debasement::compute_currency_debasement(state, out);
    smuggling::compute_smuggling(state, out);
    economic_competition::compute_economic_competition(state, out);
    bankruptcy_cascade::compute_bankruptcy_cascade(state, out);
    contracts::compute_contracts(state, out);
    contract_negotiation::compute_contract_negotiation(state, out);

    // Quests
    quest_generation::compute_quest_generation(state, out);
    quest_lifecycle::compute_quest_lifecycle(state, out);
    quest_expiry::compute_quest_expiry(state, out);
    seasonal_quests::compute_seasonal_quests(state, out);
    bounties::compute_bounties(state, out);
    skill_challenges::compute_skill_challenges(state, out);
    treasure_hunts::compute_treasure_hunts(state, out);
    exploration::compute_exploration(state, out);
    dungeons::compute_dungeons(state, out);
    heist_planning::compute_heist_planning(state, out);
    difficulty_scaling::compute_difficulty_scaling(state, out);

    // Adventurer development
    adventurer_condition::compute_adventurer_condition(state, out);
    adventurer_recovery::compute_adventurer_recovery(state, out);
    progression::compute_progression(state, out);
    recruitment::compute_recruitment(state, out);
    retirement::compute_retirement(state, out);
    mentorship::compute_mentorship(state, out);

    // World
    seasons::compute_seasons(state, out);
    weather::compute_weather(state, out);
    monster_ecology::compute_monster_ecology(state, out);
    threat::compute_threat(state, out);
    migration::compute_migration(state, out);
    dead_zones::compute_dead_zones(state, out);
    terrain_events::compute_terrain_events(state, out);
    geography::compute_geography(state, out);
    supply_lines::compute_supply_lines(state, out);
    signal_towers::compute_signal_towers(state, out);
    escalation_protocol::compute_escalation_protocol(state, out);
    timed_events::compute_timed_events(state, out);
    random_events::compute_random_events(state, out);
    crisis::compute_crisis(state, out);
    traveling_merchants::compute_traveling_merchants(state, out);

    // Factions
    faction_ai::compute_faction_ai(state, out);
    diplomacy::compute_diplomacy(state, out);
    espionage::compute_espionage(state, out);
    counter_espionage::compute_counter_espionage(state, out);
    war_exhaustion::compute_war_exhaustion(state, out);
    civil_war::compute_civil_war(state, out);
    council::compute_council(state, out);
    coup_engine::compute_coup_engine(state, out);
    defection_cascade::compute_defection_cascade(state, out);
    propaganda::compute_propaganda(state, out);
    alliance_blocs::compute_alliance_blocs(state, out);
    vassalage::compute_vassalage(state, out);
    faction_tech::compute_faction_tech(state, out);
    mercenaries::compute_mercenaries(state, out);

    // Social / narrative
    bonds::compute_bonds(state, out);
    moods::compute_moods(state, out);
    npc_relationships::compute_npc_relationships(state, out);
    npc_reputation::compute_npc_reputation(state, out);
    reputation_decay::compute_reputation_decay(state, out);
    reputation_stories::compute_reputation_stories(state, out);
    rumors::compute_rumors(state, out);
    chronicle::compute_chronicle(state, out);
    romance::compute_romance(state, out);
    marriages::compute_marriages(state, out);
    rivalries::compute_rivalries(state, out);
    companions::compute_companions(state, out);
    grudges::compute_grudges(state, out);
    oaths::compute_oaths(state, out);
    intrigue::compute_intrigue(state, out);
    secrets::compute_secrets(state, out);
    prisoners::compute_prisoners(state, out);
    wanted::compute_wanted(state, out);
    nicknames::compute_nicknames(state, out);
    folk_hero::compute_folk_hero(state, out);
    legendary_deeds::compute_legendary_deeds(state, out);
    personal_goals::compute_personal_goals(state, out);
    fears::compute_fears(state, out);
    hobbies::compute_hobbies(state, out);
    journals::compute_journals(state, out);
    memorials::compute_memorials(state, out);
    trophies::compute_trophies(state, out);
    leadership::compute_leadership(state, out);
    guild_identity::compute_guild_identity(state, out);
    guild_rooms::compute_guild_rooms(state, out);
    guild_tiers::compute_guild_tiers(state, out);
    party_chemistry::compute_party_chemistry(state, out);

    // Health / spiritual
    disease::compute_disease(state, out);
    plague_vectors::compute_plague_vectors(state, out);
    wound_persistence::compute_wound_persistence(state, out);
    addiction::compute_addiction(state, out);
    equipment_durability::compute_equipment_durability(state, out);
    divine_favor::compute_divine_favor(state, out);
    religion::compute_religion(state, out);
    demonic_pacts::compute_demonic_pacts(state, out);
    awakening::compute_awakening(state, out);
    visions::compute_visions(state, out);
    bloodlines::compute_bloodlines(state, out);
    legacy_weapons::compute_legacy_weapons(state, out);

    // Meta
    charter::compute_charter(state, out);
    buildings::compute_buildings(state, out);
    choices::compute_choices(state, out);
    corruption::compute_corruption(state, out);
    festivals::compute_festivals(state, out);
    rival_guild::compute_rival_guild(state, out);
    victory_conditions::compute_victory_conditions(state, out);
}
