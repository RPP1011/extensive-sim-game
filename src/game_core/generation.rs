use super::overworld_types::*;
use super::roster_types::PersonalityArchetype;

pub(crate) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(crate) fn rand01(seed: u64, salt: u64) -> f32 {
    let v = splitmix64(seed ^ salt);
    (v as f64 / u64::MAX as f64) as f32
}

pub(crate) fn archetype_for(seed: u64, salt: u64) -> PersonalityArchetype {
    match (splitmix64(seed ^ salt) % 3) as u8 {
        0 => PersonalityArchetype::Vanguard,
        1 => PersonalityArchetype::Guardian,
        _ => PersonalityArchetype::Tactician,
    }
}

pub(crate) fn recruit_codename(id: u32, seed: u64) -> String {
    let adj = ["Ash", "Stone", "Silver", "Rook", "Ember", "Gale"];
    let noun = ["Fox", "Lance", "Warden", "Bell", "Sable", "Moth"];
    let a = (splitmix64(seed ^ id as u64) % adj.len() as u64) as usize;
    let n = (splitmix64(seed ^ (id as u64).wrapping_mul(17)) % noun.len() as u64) as usize;
    format!("{} {}", adj[a], noun[n])
}

pub(crate) fn pick_origin(overworld: &OverworldMap, seed: u64, id: u32) -> (usize, usize) {
    if overworld.regions.is_empty() {
        return (0, 0);
    }
    let region_idx =
        (splitmix64(seed ^ (id as u64).wrapping_mul(73)) % overworld.regions.len() as u64) as usize;
    let region = &overworld.regions[region_idx];
    (region.owner_faction_id, region.id)
}

pub(crate) fn backstory_for_recruit(
    codename: &str,
    archetype: PersonalityArchetype,
    faction_name: &str,
    region_name: &str,
    unrest: f32,
    control: f32,
) -> String {
    let archetype_hook = match archetype {
        PersonalityArchetype::Vanguard => "learned to lead from the front",
        PersonalityArchetype::Guardian => "became a shield for civilians",
        PersonalityArchetype::Tactician => "survived by reading every battlefield angle",
    };
    let pressure_hook = if unrest >= 60.0 {
        "after repeated border collapses"
    } else if control >= 70.0 {
        "under disciplined garrison rule"
    } else {
        "through constant jurisdiction disputes"
    };
    format!(
        "{codename} was raised in {region_name} under {faction_name}, {pressure_hook}, and {archetype_hook}."
    )
}

pub(crate) fn vassal_name(faction: &str, id: u32) -> String {
    let titles = [
        "Warden",
        "Reeve",
        "Marshal",
        "Castellan",
        "Captain",
        "Steward",
    ];
    let tags = ["Ash", "Stone", "Gale", "Sable", "Iron", "Dawn"];
    let t = (splitmix64(id as u64) % titles.len() as u64) as usize;
    let g = (splitmix64((id as u64).wrapping_mul(13)) % tags.len() as u64) as usize;
    format!("{} {} of {}", titles[t], tags[g], faction)
}

pub(crate) fn specialty_for(id: u32) -> VassalSpecialty {
    match splitmix64((id as u64).wrapping_mul(29)) % 4 {
        0 => VassalSpecialty::Siege,
        1 => VassalSpecialty::Patrol,
        2 => VassalSpecialty::Escort,
        _ => VassalSpecialty::Logistics,
    }
}

pub(crate) fn target_vassal_count(strength: f32) -> usize {
    ((strength / 18.0).round() as i32).clamp(2, 14) as usize
}

pub(crate) fn hex_distance(a: (i32, i32), b: (i32, i32)) -> i32 {
    let dq = a.0 - b.0;
    let dr = a.1 - b.1;
    (dq.abs() + dr.abs() + (dq + dr).abs()) / 2
}

pub(crate) fn overworld_hex_coords() -> Vec<(i32, i32)> {
    let mut coords = Vec::new();
    let radius = 2;
    for q in -radius..=radius {
        let r_min = (-radius).max(-q - radius);
        let r_max = radius.min(-q + radius);
        for r in r_min..=r_max {
            coords.push((q, r));
        }
    }
    coords.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    coords
}

pub fn overworld_region_plot_positions(overworld: &OverworldMap) -> Vec<(f32, f32)> {
    let coords = overworld_hex_coords();
    if overworld.regions.len() == coords.len() {
        coords
            .iter()
            .map(|(q, r)| {
                // Axial hex -> 2D pointy-top projection.
                let x = *q as f32 + (*r as f32) * 0.5;
                let y = (*r as f32) * 0.866_025_4;
                (x, y)
            })
            .collect()
    } else {
        // Fallback for custom maps with different region counts.
        let n = overworld.regions.len().max(1) as f32;
        overworld
            .regions
            .iter()
            .map(|r| {
                let theta = (r.id as f32 / n) * std::f32::consts::TAU;
                (theta.cos(), theta.sin())
            })
            .collect()
    }
}

pub(crate) fn build_seeded_overworld(seed: u64) -> OverworldMap {
    let faction_names = ["Guild Compact", "Iron Marches", "River Concord"];
    let coords = overworld_hex_coords();

    let name_left = [
        "Ash", "Hollow", "Rift", "Ember", "Dawn", "Stone", "Mist", "Iron",
    ];
    let name_right = [
        "Bastion", "Reach", "March", "Fen", "Spire", "Crossing", "Watch", "Roads",
    ];
    let neighbor_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

    let mut regions = Vec::with_capacity(coords.len());
    for (id, coord) in coords.iter().enumerate() {
        let mut neighbors = Vec::new();
        for (dq, dr) in neighbor_dirs {
            let target = (coord.0 + dq, coord.1 + dr);
            if let Some(idx) = coords.iter().position(|c| *c == target) {
                neighbors.push(idx);
            }
        }
        neighbors.sort_unstable();
        let left =
            (splitmix64(seed ^ (id as u64).wrapping_mul(17)) % name_left.len() as u64) as usize;
        let right =
            (splitmix64(seed ^ (id as u64).wrapping_mul(31)) % name_right.len() as u64) as usize;
        regions.push(OverworldRegion {
            id,
            name: format!("{} {} {}", name_left[left], name_right[right], id + 1),
            neighbors,
            owner_faction_id: 0,
            mission_slot: None,
            unrest: 0.0,
            control: 0.0,
            intel_level: 0.0,
        });
    }

    let capital_coords = [(-2, 0), (2, 0), (0, 2)];
    let capital_ids = capital_coords
        .iter()
        .map(|cc| coords.iter().position(|c| c == cc).unwrap_or(0))
        .collect::<Vec<_>>();

    for (id, coord) in coords.iter().enumerate() {
        let mut best_faction = 0usize;
        let mut best_score = f32::MAX;
        for (faction_id, cap_coord) in capital_coords.iter().enumerate() {
            let dist = hex_distance(*coord, *cap_coord) as f32;
            let jitter = (rand01(seed, 10_000 + (id as u64) * 73 + faction_id as u64) - 0.5) * 1.1;
            let score = dist + jitter;
            if score < best_score {
                best_score = score;
                best_faction = faction_id;
            }
        }
        regions[id].owner_faction_id = best_faction;
    }
    for (faction_id, cap_id) in capital_ids.iter().enumerate() {
        if *cap_id < regions.len() {
            regions[*cap_id].owner_faction_id = faction_id;
        }
    }

    let owners = regions
        .iter()
        .map(|r| r.owner_faction_id)
        .collect::<Vec<_>>();
    for region in &mut regions {
        let border_pressure = region
            .neighbors
            .iter()
            .filter(|n| {
                owners.get(**n).copied().unwrap_or(region.owner_faction_id)
                    != region.owner_faction_id
            })
            .count() as f32;
        let unrest =
            (14.0 + border_pressure * 11.0 + rand01(seed, 20_000 + region.id as u64) * 20.0)
                .clamp(0.0, 100.0);
        region.unrest = unrest;
        region.control = (100.0 - unrest).clamp(0.0, 100.0);
        let intel = (35.0
            + rand01(seed, 25_000 + region.id as u64) * 50.0
            + if region.owner_faction_id == 0 {
                8.0
            } else {
                0.0
            })
        .clamp(0.0, 100.0);
        region.intel_level = intel;
    }

    for (slot, faction_id) in (0..faction_names.len()).enumerate() {
        if let Some((idx, _)) = regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.owner_faction_id == faction_id)
            .max_by(|(_, a), (_, b)| a.unrest.total_cmp(&b.unrest).then(a.id.cmp(&b.id)))
        {
            regions[idx].mission_slot = Some(slot);
        }
    }

    let mut factions = Vec::with_capacity(faction_names.len());
    for (id, name) in faction_names.iter().enumerate() {
        let owned = regions
            .iter()
            .filter(|r| r.owner_faction_id == id)
            .collect::<Vec<_>>();
        let owned_count = owned.len().max(1) as f32;
        let avg_unrest = owned.iter().map(|r| r.unrest).sum::<f32>() / owned_count;
        let avg_control = owned.iter().map(|r| r.control).sum::<f32>() / owned_count;
        let strength = (70.0
            + owned_count * 8.0
            + avg_control * 0.35
            + rand01(seed, 30_000 + id as u64) * 12.0)
            .clamp(40.0, 180.0);
        let cohesion = (48.0 + avg_control * 0.22 - avg_unrest * 0.08
            + rand01(seed, 40_000 + id as u64) * 10.0)
            .clamp(20.0, 95.0);
        factions.push(FactionState {
            id,
            name: (*name).to_string(),
            strength,
            cohesion,
            war_goal_faction_id: None,
            war_focus: 0.0,
            vassals: Vec::new(),
        });
    }

    let current_region = capital_ids[0].min(regions.len().saturating_sub(1));
    let mut map = OverworldMap {
        regions,
        factions,
        current_region,
        selected_region: current_region,
        travel_cooldown_turns: 0,
        travel_cooldown_max: 2,
        travel_cost: 12.0,
        next_vassal_id: 1,
        map_seed: seed,
    };
    for idx in 0..map.factions.len() {
        let owned = map
            .regions
            .iter()
            .filter(|r| r.owner_faction_id == idx)
            .map(|r| r.id)
            .collect::<Vec<_>>();
        rebalance_faction_vassals(&mut map.factions[idx], &owned, &mut map.next_vassal_id);
    }
    map
}

pub(crate) fn build_vassal(faction: &FactionState, id: u32) -> FactionVassal {
    FactionVassal {
        id,
        name: vassal_name(&faction.name, id),
        martial: 38.0 + rand01(id as u64, 7) * 48.0,
        loyalty: 42.0 + rand01(id as u64, 17) * 46.0,
        specialty: specialty_for(id),
        post: VassalPost::Roaming,
        home_region_id: faction.id,
    }
}

pub(crate) fn rebalance_faction_vassals(
    faction: &mut FactionState,
    owned_regions: &[usize],
    next_vassal_id: &mut u32,
) {
    let target = target_vassal_count(faction.strength);
    while faction.vassals.len() < target {
        let id = *next_vassal_id;
        *next_vassal_id = next_vassal_id.saturating_add(1);
        faction.vassals.push(build_vassal(faction, id));
    }
    if faction.vassals.len() > target {
        faction
            .vassals
            .sort_by(|a, b| a.loyalty.total_cmp(&b.loyalty).then(a.id.cmp(&b.id)));
        faction.vassals.truncate(target);
    }

    if faction.vassals.is_empty() {
        return;
    }
    let region_cycle = if owned_regions.is_empty() {
        vec![faction.id]
    } else {
        owned_regions.to_vec()
    };
    let desired_managers = usize::min(
        region_cycle.len(),
        usize::max(1, (faction.vassals.len() as f32 * 0.35).round() as usize),
    );

    faction.vassals.sort_by_key(|v| v.id);
    for (idx, vassal) in faction.vassals.iter_mut().enumerate() {
        let home = region_cycle[idx % region_cycle.len()];
        vassal.home_region_id = home;
        vassal.post = if idx < desired_managers {
            VassalPost::ZoneManager
        } else {
            VassalPost::Roaming
        };
    }
}
