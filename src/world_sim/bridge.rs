//! Bridge between CampaignState (headless_campaign) and WorldState (world_sim).
//!
//! Converts campaign entities into the unified delta-based world simulation
//! and writes results back.

use crate::headless_campaign::state::{
    self as campaign, Adventurer, CampaignState, Location,
};

use super::state::*;
use super::fidelity::Fidelity;
use super::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// CampaignState → WorldState
// ---------------------------------------------------------------------------

/// Build a WorldState snapshot from the current CampaignState.
///
/// This is the entry point for running the world sim on campaign data.
/// Each adventurer becomes an Entity, each location with residents becomes
/// a settlement, and regions map to RegionState.
pub fn campaign_to_world(campaign: &CampaignState) -> WorldState {
    let mut world = WorldState::new(campaign.tick);

    // Convert locations to settlements + grids.
    for loc in &campaign.overworld.locations {
        let settlement = location_to_settlement(loc);
        let grid = LocalGrid {
            id: loc.id as u32,
            fidelity: if loc.threat_level > 50.0 { Fidelity::Low } else { Fidelity::Medium },
            center: loc.position,
            radius: 20.0,
            entity_ids: loc.resident_ids.clone(),
        };
        world.settlements.push(settlement);
        world.grids.push(grid);
    }

    // Convert regions.
    for region in &campaign.overworld.regions {
        world.regions.push(RegionState {
            id: region.id as u32,
            name: region.name.clone(),
            terrain: Terrain::Plains, // default for bridged campaign data
            monster_density: region.threat_level / 10.0,
            faction_id: Some(region.owner_faction_id as u32),
            threat_level: region.threat_level,
            has_river: false,
            has_lake: false,
            is_coastal: false,
            river_connections: Vec::new(),
            dungeon_sites: Vec::new(),
            sub_biome: SubBiome::Standard,
            neighbors: Vec::new(),
            is_chokepoint: false,
            elevation: 1,
            is_floating: false,
            unrest: 0.0,
            control: 1.0,
        });
    }

    // Convert adventurers to entities.
    for adv in &campaign.adventurers {
        world.entities.push(adventurer_to_entity(adv));
    }

    // Compute global economy totals.
    for s in &world.settlements {
        for i in 0..NUM_COMMODITIES {
            world.economy.total_commodities[i] += s.stockpile[i];
        }
    }
    for e in &world.entities {
        if let Some(npc) = &e.npc {
            world.economy.total_gold_supply += npc.gold;
        }
    }

    world
}

fn adventurer_to_entity(adv: &Adventurer) -> Entity {
    let home_settlement = adv.home_location_id.map(|id| id as u32);

    // Determine grid membership from home location.
    let grid_id = home_settlement;
    let pos = (0.0, 0.0); // Position comes from the location or party.

    let economic_intent = match &adv.economic_intent {
        campaign::EconomicIntent::Idle => EconomicIntent::Idle,
        campaign::EconomicIntent::Working => EconomicIntent::Produce,
        campaign::EconomicIntent::Relocating { target_location_id } => {
            EconomicIntent::Travel { destination: (*target_location_id as f32, 0.0) }
        }
        campaign::EconomicIntent::SeekingParty { .. } => EconomicIntent::Idle,
        campaign::EconomicIntent::Adventuring => EconomicIntent::Idle,
        campaign::EconomicIntent::Traveling => EconomicIntent::Idle,
    };

    let price_knowledge = adv.price_knowledge.iter().map(|pr| {
        PriceReport {
            settlement_id: pr.location_id as u32,
            prices: pr.prices,
            tick_observed: pr.reported_tick as u64,
        }
    }).collect();

    Entity {
        id: adv.id,
        kind: EntityKind::Npc,
        team: WorldTeam::Friendly,
        pos,
        grid_id,
        local_pos: None,
        alive: !matches!(adv.status, campaign::AdventurerStatus::Dead),
        hp: adv.stats.max_hp * (1.0 - adv.injury / 100.0),
        max_hp: adv.stats.max_hp,
        shield_hp: 0.0,
        armor: adv.stats.defense,
        magic_resist: 0.0,
        attack_damage: adv.stats.attack,
        attack_range: 1.5,
        move_speed: adv.stats.speed,
        level: adv.level,
        status_effects: Vec::new(),
        npc: Some(NpcData {
            adventurer_id: adv.id,
            name: String::new(),
            gold: adv.gold,
            home_settlement_id: home_settlement,
            home_building_id: None,
            work_building_id: None,
            work_state: WorkState::Idle,
            economic_intent,
            action: NpcAction::Idle,
            oaths: Vec::new(),
            spouse_id: None,
            children: Vec::new(),
            parents: Vec::new(),
            born_tick: 0,
            mentor_lineage: Vec::new(),
            inside_building_id: None,
            current_room: None,
            goal_stack: GoalStack::default(),
            cached_path: Vec::new(),
            path_index: 0,
            price_knowledge,
            carried_goods: adv.carried_goods,
            class_tags: Vec::new(),
            behavior_production: Vec::new(),
            morale: adv.morale,
            stress: adv.stress,
            fatigue: adv.fatigue,
            loyalty: adv.loyalty,
            injury: adv.injury,
            xp: adv.xp,
            archetype: adv.archetype.clone(),
            party_id: adv.party_id,
            faction_id: adv.faction_id.map(|f| f as u32),
            mood: 0,
            fears: Vec::new(),
            deeds: adv.deeds.iter().enumerate().map(|(i, _)| i as u8).collect(),
            resolve: adv.resolve,
            guild_relationship: adv.guild_relationship,
            needs: Needs::default(),
            memory: Memory::default(),
            personality: Personality::default(),
            emotions: Emotions::default(),
            equipment: Equipment::default(),
            equipped_items: EquippedItems::default(),
            behavior_profile: Vec::new(),
            classes: Vec::new(),
        }),
        building: None,
        item: None,
        inventory: Some(Inventory::with_capacity(50.0)),
    }
}

fn location_to_settlement(loc: &Location) -> SettlementState {
    let stockpile = [
        loc.stockpile.food,
        loc.stockpile.iron,
        loc.stockpile.wood,
        loc.stockpile.herbs,
        loc.stockpile.hide,
        loc.stockpile.crystal,
        loc.stockpile.equipment,
        loc.stockpile.medicine,
    ];

    SettlementState {
        id: loc.id as u32,
        name: loc.name.clone(),
        pos: loc.position,
        grid_id: Some(loc.id as u32),
        stockpile,
        prices: loc.local_prices,
        treasury: loc.treasury,
        population: loc.resident_ids.len() as u32,
        specialty: SettlementSpecialty::default(),
        faction_id: None,
        threat_level: 0.0,
        infrastructure_level: 0.0,
        context_tags: Vec::new(),
        city_grid_idx: None,
        treasury_building_id: None,
    }
}

// ---------------------------------------------------------------------------
// WorldState → CampaignState (write-back)
// ---------------------------------------------------------------------------

/// Write world simulation results back to the campaign state.
///
/// Updates adventurer gold, carried goods, price knowledge, and
/// settlement stockpiles/prices/treasury from the world state.
pub fn world_to_campaign(world: &WorldState, campaign: &mut CampaignState) {
    // Write back adventurer economic state.
    for entity in &world.entities {
        if let Some(npc) = &entity.npc {
            if let Some(adv) = campaign.adventurers.iter_mut().find(|a| a.id == npc.adventurer_id) {
                adv.gold = npc.gold;
                adv.carried_goods = npc.carried_goods;

                // Update price knowledge.
                adv.price_knowledge = npc.price_knowledge.iter().map(|pr| {
                    campaign::PriceReport {
                        location_id: pr.settlement_id as usize,
                        prices: pr.prices,
                        stockpiles: [0.0; 8], // Not tracked in world sim.
                        reported_tick: pr.tick_observed as u32,
                    }
                }).collect();
            }
        }
    }

    // Write back settlement state.
    for settlement in &world.settlements {
        if let Some(loc) = campaign.overworld.locations.iter_mut()
            .find(|l| l.id == settlement.id as usize)
        {
            loc.stockpile.food = settlement.stockpile[0];
            loc.stockpile.iron = settlement.stockpile[1];
            loc.stockpile.wood = settlement.stockpile[2];
            loc.stockpile.herbs = settlement.stockpile[3];
            loc.stockpile.hide = settlement.stockpile[4];
            loc.stockpile.crystal = settlement.stockpile[5];
            loc.stockpile.equipment = settlement.stockpile[6];
            loc.stockpile.medicine = settlement.stockpile[7];
            loc.local_prices = settlement.prices;
            loc.treasury = settlement.treasury;
        }
    }
}
