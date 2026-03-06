use std::collections::HashMap;

use serde_json::Value;

use super::types::*;
use super::partition::*;

pub fn load_overworld(data: &Value) -> Result<OverworldData, String> {
    let overworld = data
        .get("overworld_map")
        .ok_or_else(|| "Save file missing overworld_map".to_string())?;

    let map_seed = overworld.get("map_seed").and_then(Value::as_u64).unwrap_or(0);

    let regions_raw = overworld
        .get("regions").and_then(Value::as_array)
        .ok_or_else(|| "overworld_map missing regions".to_string())?;
    let factions_raw = overworld
        .get("factions").and_then(Value::as_array)
        .ok_or_else(|| "overworld_map missing factions".to_string())?;

    let mut regions = Vec::with_capacity(regions_raw.len());
    for (i, r) in regions_raw.iter().enumerate() {
        let id = r.get("id").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(i);
        let name = r.get("name").and_then(Value::as_str).map(|s| s.to_string()).unwrap_or_else(|| format!("Region {id}"));
        let owner_faction_id = r.get("owner_faction_id").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(0);
        let unrest = r.get("unrest").and_then(Value::as_f64).unwrap_or(50.0);
        let control = r.get("control").and_then(Value::as_f64).unwrap_or(50.0);
        let intel_level = r.get("intel_level").and_then(Value::as_f64).unwrap_or(0.0);
        let neighbors = r.get("neighbors").and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_u64).map(|x| x as usize).collect::<Vec<_>>())
            .unwrap_or_default();

        regions.push(RegionInput { id, name, owner_faction_id, unrest, control, intel_level, neighbors });
    }

    let mut factions = Vec::with_capacity(factions_raw.len());
    for (i, f) in factions_raw.iter().enumerate() {
        let id = f.get("id").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(i);
        let name = f.get("name").and_then(Value::as_str).map(|s| s.to_string()).unwrap_or_else(|| format!("Faction {id}"));
        let strength = f.get("strength").and_then(Value::as_f64).unwrap_or(0.0);
        let cohesion = f.get("cohesion").and_then(Value::as_f64).unwrap_or(0.0);
        factions.push(FactionInput { id, name, strength, cohesion });
    }

    Ok(OverworldData { map_seed, regions, factions })
}

fn round_n(value: f64, n: usize) -> f64 {
    let scale = 10_f64.powi(n as i32);
    (value * scale).round() / scale
}

pub fn build_spec(
    overworld: &OverworldData,
    grid_w: usize,
    grid_h: usize,
    strength_scale: f64,
    organic_jitter: f64,
) -> VoronoiSpec {
    let points = normalize_points(&build_site_positions(overworld));
    let weights = compute_region_weights(overworld, strength_scale);
    let grid = weighted_partition(&points, &weights, grid_w, grid_h);

    let mut rid_counts: HashMap<usize, usize> = HashMap::new();
    for row in &grid {
        for idx in row {
            *rid_counts.entry(*idx).or_insert(0) += 1;
        }
    }

    let total = (grid_w * grid_h).max(1) as f64;
    let mut complexity_values = Vec::with_capacity(overworld.regions.len());

    let mut region_meta = Vec::with_capacity(overworld.regions.len());
    for (idx, r) in overworld.regions.iter().enumerate() {
        let area_pct = rid_counts.get(&idx).copied().unwrap_or(0) as f64 * 100.0 / total;
        let complexity = boundary_complexity(&grid, idx);
        complexity_values.push(complexity);

        let seed = overworld.map_seed ^ ((r.id as u64).wrapping_mul(0x9E37));
        let organic = (rand01(seed) - 0.5) * 2.0 * organic_jitter;

        region_meta.push(RegionMeta {
            id: r.id,
            name: r.name.clone(),
            owner_faction_id: r.owner_faction_id,
            site: Site { x: round_n(points[idx].0, 5), y: round_n(points[idx].1, 5) },
            weight: round_n(weights[idx], 6),
            area_pct: round_n(area_pct, 3),
            organic_jitter: round_n(organic, 4),
            boundary_complexity: round_n(complexity, 4),
        });
    }

    let faction_share = faction_area_ratio(&grid, overworld);
    let roads = roads_from_neighbors(&overworld.regions);
    let settlements = settlement_candidates(&overworld.regions, 2);

    let factions = overworld.factions.iter().map(|f| FactionMeta {
        id: f.id,
        name: f.name.clone(),
        strength: f.strength,
        cohesion: f.cohesion,
        territory_pct: round_n(faction_share.get(&f.id).copied().unwrap_or(0.0) * 100.0, 2),
    }).collect::<Vec<_>>();

    let avg_complexity = if complexity_values.is_empty() { 0.0 } else {
        complexity_values.iter().sum::<f64>() / complexity_values.len() as f64
    };
    let terrain_notes = format!("organic edge roughness {:.2}, mean boundary complexity {:.2}", organic_jitter, avg_complexity);

    let owner_by_region: HashMap<usize, usize> = overworld.regions.iter().map(|r| (r.id, r.owner_faction_id)).collect();
    let border_count = roads.iter().filter(|road| owner_by_region.get(&road.a) != owner_by_region.get(&road.b)).count();
    let frontier_notes = format!("{} inter-faction front edges across {} total edges", border_count, roads.len());

    VoronoiSpec {
        map_seed: overworld.map_seed,
        grid: GridSize { w: grid_w, h: grid_h },
        strength_scale,
        organic_jitter,
        regions: region_meta,
        factions,
        road_count: roads.len(),
        roads,
        settlements,
        terrain_notes,
        frontier_notes,
    }
}

pub fn build_prompt(spec: &VoronoiSpec) -> String {
    let faction_lines = spec.factions.iter().map(|f| {
        format!("- {}: strength {:.0}, cohesion {:.0}, territory share ~{:.1}%", f.name, f.strength, f.cohesion, f.territory_pct)
    }).collect::<Vec<_>>();

    let settlement_lines = spec.settlements.iter().take(8)
        .map(|s| format!("- {} (Faction {})", s.name, s.faction_id))
        .collect::<Vec<_>>();

    [
        vec!["Top-down fantasy campaign map, Mount & Blade style strategic overworld.".to_string()],
        vec![String::new()],
        vec!["Render goals:".to_string()],
        vec![
            "- Organic territorial regions with irregular Voronoi-like borders.".to_string(),
            "- Territory size should visibly scale with faction power.".to_string(),
            "- Soft terrain shapes and painterly, hand-drawn contour treatment.".to_string(),
            "- Distinct frontier lines where hostile factions meet.".to_string(),
            "- Roads connecting neighboring settlements and passes.".to_string(),
            "- Show one player party route projection between two regions.".to_string(),
        ],
        vec![String::new()],
        vec!["World facts (from simulation):".to_string()],
        faction_lines,
        vec![String::new()],
        vec![format!("- Roads in graph: {}", spec.road_count)],
        vec![format!("- Terrain blend guidance: {}", spec.terrain_notes)],
        vec![format!("- Frontier guidance: {}", spec.frontier_notes)],
        vec![String::new()],
        vec!["Settlement anchors:".to_string()],
        settlement_lines,
        vec![String::new()],
        vec!["Style:".to_string()],
        vec![
            "- Stylized 2D parchment + painted relief hybrid.".to_string(),
            "- High readability, low clutter, no text labels embedded in image.".to_string(),
            "- Faction colors should be distinct but slightly desaturated.".to_string(),
            "- Avoid UI panels; output should be pure map art.".to_string(),
        ],
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("\n")
}
