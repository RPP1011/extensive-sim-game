use super::generation::*;
use super::overworld_types::OverworldMap;
use super::roster_types::*;

pub fn generate_recruit(seed: u64, id: u32) -> RecruitCandidate {
    let map = OverworldMap::default();
    generate_recruit_for_overworld(seed, id, &map)
}

pub fn generate_recruit_for_overworld(
    seed: u64,
    id: u32,
    overworld: &OverworldMap,
) -> RecruitCandidate {
    let codename = recruit_codename(id, seed);
    let archetype = archetype_for(seed, id as u64);
    let (origin_faction_id, origin_region_id) = pick_origin(overworld, seed, id);
    let (faction_name, region_name, unrest, control) = {
        let region = overworld
            .regions
            .iter()
            .find(|r| r.id == origin_region_id)
            .or_else(|| overworld.regions.first());
        let fallback_region = "Unknown March";
        let fallback_faction = "Unaligned House";
        match region {
            Some(r) => {
                let f = overworld
                    .factions
                    .get(r.owner_faction_id)
                    .map(|x| x.name.as_str())
                    .unwrap_or(fallback_faction);
                (f.to_string(), r.name.clone(), r.unrest, r.control)
            }
            None => (
                fallback_faction.to_string(),
                fallback_region.to_string(),
                50.0,
                50.0,
            ),
        }
    };
    RecruitCandidate {
        id,
        codename: codename.clone(),
        origin_faction_id,
        origin_region_id,
        backstory: backstory_for_recruit(
            &codename,
            archetype,
            &faction_name,
            &region_name,
            unrest,
            control,
        ),
        archetype,
        resolve: 52.0 + rand01(seed, id as u64 + 11) * 34.0,
        loyalty_bias: 44.0 + rand01(seed, id as u64 + 31) * 30.0,
        risk_tolerance: 28.0 + rand01(seed, id as u64 + 53) * 55.0,
    }
}

impl Default for CampaignRoster {
    fn default() -> Self {
        let mut roster = CampaignRoster {
            heroes: vec![
                HeroCompanion {
                    id: 1,
                    name: "Warden Lyra".to_string(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Guardian,
                    loyalty: 72.0,
                    stress: 12.0,
                    fatigue: 10.0,
                    injury: 4.0,
                    resolve: 78.0,
                    active: true,
                    deserter: false,
                    xp: 0,
                    level: 1,
                    equipment: HeroEquipment::default(),
                    traits: Vec::new(),
                },
                HeroCompanion {
                    id: 2,
                    name: "Kade Ember".to_string(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Vanguard,
                    loyalty: 64.0,
                    stress: 20.0,
                    fatigue: 18.0,
                    injury: 8.0,
                    resolve: 66.0,
                    active: true,
                    deserter: false,
                    xp: 0,
                    level: 1,
                    equipment: HeroEquipment::default(),
                    traits: Vec::new(),
                },
                HeroCompanion {
                    id: 3,
                    name: "Iris Quill".to_string(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Tactician,
                    loyalty: 68.0,
                    stress: 16.0,
                    fatigue: 14.0,
                    injury: 2.0,
                    resolve: 74.0,
                    active: true,
                    deserter: false,
                    xp: 0,
                    level: 1,
                    equipment: HeroEquipment::default(),
                    traits: Vec::new(),
                },
            ],
            recruit_pool: Vec::new(),
            player_hero_id: Some(1),
            next_id: 4,
            generation_counter: 0,
        };
        refill_recruit_pool(&mut roster);
        let map = OverworldMap::default();
        for hero in &mut roster.heroes {
            let generated =
                generate_recruit_for_overworld(map.map_seed ^ 0xBACC_5700, hero.id, &map);
            hero.origin_faction_id = generated.origin_faction_id;
            hero.origin_region_id = generated.origin_region_id;
            hero.backstory = generated.backstory;
        }
        roster
    }
}

pub fn refill_recruit_pool(roster: &mut CampaignRoster) {
    while roster.recruit_pool.len() < 3 {
        let id = roster.next_id;
        let seed = 0xC0DE_0000_0000_0042_u64 ^ roster.generation_counter;
        roster.recruit_pool.push(generate_recruit(seed, id));
        roster.next_id += 1;
    }
}

pub fn sign_top_recruit(roster: &mut CampaignRoster) -> Option<HeroCompanion> {
    if roster.recruit_pool.is_empty() {
        refill_recruit_pool(roster);
    }
    let recruit = roster.recruit_pool.remove(0);
    let hero = HeroCompanion {
        id: recruit.id,
        name: recruit.codename,
        origin_faction_id: recruit.origin_faction_id,
        origin_region_id: recruit.origin_region_id,
        backstory: recruit.backstory,
        archetype: recruit.archetype,
        loyalty: recruit.loyalty_bias,
        stress: (6.0 + recruit.risk_tolerance * 0.12).clamp(0.0, 100.0),
        fatigue: (5.0 + recruit.risk_tolerance * 0.08).clamp(0.0, 100.0),
        injury: 0.0,
        resolve: recruit.resolve,
        active: true,
        deserter: false,
        xp: 0,
        level: 1,
        equipment: HeroEquipment::default(),
        traits: Vec::new(),
    };
    roster.heroes.push(hero.clone());
    if roster.player_hero_id.is_none() {
        roster.player_hero_id = Some(hero.id);
    }
    roster.generation_counter = roster.generation_counter.wrapping_add(1);
    refill_recruit_pool(roster);
    Some(hero)
}
