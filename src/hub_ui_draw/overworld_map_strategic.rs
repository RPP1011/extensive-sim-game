//! Overworld strategic map — procedural ASCII terrain rendering.
//!
//! Renders an 80x40 character ASCII landscape with per-character coloring via
//! `egui::text::LayoutJob`. Regions are projected from axial hex coords to
//! screen positions, with Voronoi assignment giving each cell a region owner.
//! Terrain is generated deterministically from `map_seed`.

use bevy_egui::egui;

use crate::game_core;
use crate::region_nav::{
    RegionTargetPickerState, update_region_target_picker_selection,
};
use super::faction_color;

// ---------------------------------------------------------------------------
// Grid dimensions
// ---------------------------------------------------------------------------

const MAP_W: usize = 80;
const MAP_H: usize = 40;

// ---------------------------------------------------------------------------
// Terrain color palette (muted, from the rendering pipeline doc)
// ---------------------------------------------------------------------------

const COLOR_PLAINS: egui::Color32    = egui::Color32::from_rgb(0x8A, 0x9A, 0x7A);
const COLOR_FOREST: egui::Color32    = egui::Color32::from_rgb(0x5A, 0x7A, 0x4A);
const COLOR_MOUNTAIN: egui::Color32  = egui::Color32::from_rgb(0x8A, 0x7A, 0x5A);
const COLOR_PEAK: egui::Color32      = egui::Color32::from_rgb(0xAA, 0xA0, 0x96);
const COLOR_WATER: egui::Color32     = egui::Color32::from_rgb(0x7A, 0x9A, 0xBB);
const COLOR_ROAD: egui::Color32      = egui::Color32::from_rgb(0xA0, 0x9B, 0x91);
const COLOR_HILLS: egui::Color32     = egui::Color32::from_rgb(0x9A, 0x8A, 0x6A);
const COLOR_COAST: egui::Color32     = egui::Color32::from_rgb(0xB4, 0xAA, 0x8C);
const COLOR_BORDER: egui::Color32    = egui::Color32::from_rgb(0x88, 0x88, 0x88);
const COLOR_SETTLEMENT: egui::Color32 = egui::Color32::from_rgb(0xDD, 0xDD, 0xCC);
const COLOR_MARKER_CURRENT: egui::Color32 = egui::Color32::from_rgb(0xFF, 0xFF, 0x40);
const COLOR_MARKER_SELECTED: egui::Color32 = egui::Color32::from_rgb(0x60, 0xDD, 0xFF);
const COLOR_MARKER_MISSION: egui::Color32 = egui::Color32::from_rgb(0xFF, 0xA0, 0x40);
const COLOR_DIM: egui::Color32       = egui::Color32::from_rgb(55, 60, 68);
const COLOR_LEGEND: egui::Color32    = egui::Color32::from_rgb(140, 150, 165);

// ---------------------------------------------------------------------------
// Deterministic hash
// ---------------------------------------------------------------------------

fn terrain_hash(seed: u64, x: i32, y: i32) -> u32 {
    let mut h = seed
        .wrapping_add(x as u64).wrapping_mul(374761393)
        .wrapping_add(y as u64).wrapping_mul(668265263);
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h = h ^ (h >> 16);
    h as u32
}

fn hash_f32(seed: u64, x: i32, y: i32) -> f32 {
    (terrain_hash(seed, x, y) & 0xFFFF) as f32 / 65535.0
}

// ---------------------------------------------------------------------------
// Terrain types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Terrain {
    Plains,
    Forest,
    Hills,
    Mountain,
    Peak,
    Water,
    Road,
    Coast,
}

impl Terrain {
    fn glyph(self) -> char {
        match self {
            Terrain::Plains   => '.',
            Terrain::Forest   => '♣',
            Terrain::Hills    => '~',
            Terrain::Mountain => '^',
            Terrain::Peak     => '▲',
            Terrain::Water    => '≈',
            Terrain::Road     => '═',
            Terrain::Coast    => ',',
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            Terrain::Plains   => COLOR_PLAINS,
            Terrain::Forest   => COLOR_FOREST,
            Terrain::Hills    => COLOR_HILLS,
            Terrain::Mountain => COLOR_MOUNTAIN,
            Terrain::Peak     => COLOR_PEAK,
            Terrain::Water    => COLOR_WATER,
            Terrain::Road     => COLOR_ROAD,
            Terrain::Coast    => COLOR_COAST,
        }
    }
}

// ---------------------------------------------------------------------------
// Color blending helpers
// ---------------------------------------------------------------------------

fn blend_colors(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let inv = 1.0 - t;
    egui::Color32::from_rgb(
        (a.r() as f32 * inv + b.r() as f32 * t) as u8,
        (a.g() as f32 * inv + b.g() as f32 * t) as u8,
        (a.b() as f32 * inv + b.b() as f32 * t) as u8,
    )
}

// ---------------------------------------------------------------------------
// Region screen positions (axial hex → character grid)
// ---------------------------------------------------------------------------

struct RegionInfo {
    sx: f32,
    sy: f32,
    region_idx: usize,
    faction_id: usize,
}

fn build_region_positions(overworld: &game_core::OverworldMap) -> Vec<RegionInfo> {
    let coords = game_core::overworld_hex_coords();
    let n_regions = overworld.regions.len();
    let n_coords = coords.len();

    if n_regions == n_coords {
        coords.iter().enumerate().map(|(i, (q, r))| {
            let sx = (MAP_W as f32 / 2.0) + (*q as f32 + *r as f32 * 0.5) * 9.0;
            let sy = (MAP_H as f32 / 2.0) + *r as f32 * 7.0;
            RegionInfo {
                sx,
                sy,
                region_idx: i,
                faction_id: overworld.regions[i].owner_faction_id,
            }
        }).collect()
    } else {
        // Fallback: spread regions in a circle
        let cx = MAP_W as f32 / 2.0;
        let cy = MAP_H as f32 / 2.0;
        let radius = (MAP_W.min(MAP_H) as f32) * 0.35;
        overworld.regions.iter().enumerate().map(|(i, r)| {
            let theta = (i as f32 / n_regions.max(1) as f32) * std::f32::consts::TAU;
            RegionInfo {
                sx: cx + theta.cos() * radius,
                sy: cy + theta.sin() * radius * 0.6,
                region_idx: i,
                faction_id: r.owner_faction_id,
            }
        }).collect()
    }
}

// ---------------------------------------------------------------------------
// Grid generation
// ---------------------------------------------------------------------------

struct MapCell {
    glyph: char,
    color: egui::Color32,
    region_id: usize,
    terrain: Terrain,
}

fn generate_map(overworld: &game_core::OverworldMap) -> Vec<MapCell> {
    let seed = overworld.map_seed;
    let region_positions = build_region_positions(overworld);

    // Step 1: Build grid with Voronoi region assignment + terrain
    let mut grid: Vec<MapCell> = Vec::with_capacity(MAP_W * MAP_H);

    for y in 0..MAP_H as i32 {
        for x in 0..MAP_W as i32 {
            // Find nearest region center (Voronoi)
            let mut best_dist = f32::INFINITY;
            let mut best_region = 0usize;
            for rp in &region_positions {
                let dx = x as f32 - rp.sx;
                let dy = y as f32 - rp.sy;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_region = rp.region_idx;
                }
            }
            let dist_to_center = best_dist.sqrt();

            // Generate terrain based on hash + distance
            let h = hash_f32(seed, x, y);
            let h2 = hash_f32(seed.wrapping_add(12345), x, y);

            let terrain = if dist_to_center > 16.0 && h > 0.55 {
                // Far from centers → mountains/peaks
                if h > 0.80 { Terrain::Peak } else { Terrain::Mountain }
            } else if dist_to_center > 12.0 && h > 0.45 {
                // Moderate distance → hills or mountains
                if h > 0.70 { Terrain::Mountain } else { Terrain::Hills }
            } else if h2 > 0.70 {
                // Moisture → forest
                Terrain::Forest
            } else if h2 > 0.85 && dist_to_center > 8.0 {
                // Rare water near edges
                Terrain::Water
            } else if dist_to_center > 14.0 && h < 0.15 {
                Terrain::Coast
            } else {
                Terrain::Plains
            };

            let color = terrain.color();
            grid.push(MapCell {
                glyph: terrain.glyph(),
                color,
                region_id: best_region,
                terrain,
            });
        }
    }

    // Step 2: Roads along connections between neighboring regions
    for region in &overworld.regions {
        for &neighbor_id in &region.neighbors {
            if neighbor_id <= region.id || neighbor_id >= region_positions.len() {
                continue;
            }
            let a = &region_positions[region.id];
            let b = &region_positions[neighbor_id];
            // Bresenham-ish road from a to b
            let steps = ((b.sx - a.sx).abs().max((b.sy - a.sy).abs()) as i32).max(1);
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                let rx = (a.sx + (b.sx - a.sx) * t).round() as i32;
                let ry = (a.sy + (b.sy - a.sy) * t).round() as i32;
                if rx >= 0 && rx < MAP_W as i32 && ry >= 0 && ry < MAP_H as i32 {
                    let idx = ry as usize * MAP_W + rx as usize;
                    if grid[idx].terrain != Terrain::Water {
                        grid[idx].glyph = Terrain::Road.glyph();
                        grid[idx].color = COLOR_ROAD;
                        grid[idx].terrain = Terrain::Road;
                    }
                }
            }
        }
    }

    // Step 3: Forest cellular automata (1 iteration)
    {
        let snapshot: Vec<Terrain> = grid.iter().map(|c| c.terrain).collect();
        for y in 1..(MAP_H as i32 - 1) {
            for x in 1..(MAP_W as i32 - 1) {
                let idx = y as usize * MAP_W + x as usize;
                let mut forest_neighbors = 0u8;
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 { continue; }
                        let ni = (y + dy) as usize * MAP_W + (x + dx) as usize;
                        if snapshot[ni] == Terrain::Forest { forest_neighbors += 1; }
                    }
                }
                if snapshot[idx] == Terrain::Forest && forest_neighbors < 3 {
                    // Isolated forest dies → plains
                    grid[idx].glyph = Terrain::Plains.glyph();
                    grid[idx].color = COLOR_PLAINS;
                    grid[idx].terrain = Terrain::Plains;
                } else if snapshot[idx] == Terrain::Plains && forest_neighbors >= 5 {
                    // Dense neighbor → grow forest
                    grid[idx].glyph = Terrain::Forest.glyph();
                    grid[idx].color = COLOR_FOREST;
                    grid[idx].terrain = Terrain::Forest;
                }
            }
        }
    }

    // Step 4: Rivers (2-3 downhill paths)
    {
        let n_rivers = 2 + (terrain_hash(seed, 999, 999) % 2) as i32;
        for river_idx in 0..n_rivers {
            let start_x = ((terrain_hash(seed, river_idx * 7, 0) % (MAP_W as u32 - 10)) + 5) as i32;
            let mut rx = start_x;
            let mut ry = 0i32;
            for _ in 0..MAP_H {
                if rx < 0 || rx >= MAP_W as i32 || ry < 0 || ry >= MAP_H as i32 { break; }
                let idx = ry as usize * MAP_W + rx as usize;
                if grid[idx].terrain != Terrain::Road {
                    grid[idx].glyph = Terrain::Water.glyph();
                    grid[idx].color = COLOR_WATER;
                    grid[idx].terrain = Terrain::Water;
                }
                // Move downward with slight horizontal drift
                ry += 1;
                let drift = hash_f32(seed.wrapping_add(river_idx as u64 * 100), rx, ry);
                if drift < 0.3 { rx -= 1; }
                else if drift > 0.7 { rx += 1; }
            }
        }
    }

    // Step 5: Settlements — write region name at each center
    for rp in &region_positions {
        if rp.region_idx >= overworld.regions.len() { continue; }
        let region = &overworld.regions[rp.region_idx];
        let name: String = region.name.chars().take(6).collect();
        let bracketed = format!("[{}]", name);
        let start_x = (rp.sx as i32) - (bracketed.len() as i32 / 2);
        let cy = rp.sy as i32;
        if cy < 0 || cy >= MAP_H as i32 { continue; }
        for (i, ch) in bracketed.chars().enumerate() {
            let cx = start_x + i as i32;
            if cx >= 0 && cx < MAP_W as i32 {
                let idx = cy as usize * MAP_W + cx as usize;
                grid[idx].glyph = ch;
                grid[idx].color = COLOR_SETTLEMENT;
            }
        }
    }

    // Step 6: Markers
    // Current region: @ in yellow
    if overworld.current_region < region_positions.len() {
        let rp = &region_positions[overworld.current_region];
        let cx = rp.sx as i32;
        let cy = rp.sy as i32 - 1; // Above the settlement name
        if cx >= 0 && cx < MAP_W as i32 && cy >= 0 && cy < MAP_H as i32 {
            let idx = cy as usize * MAP_W + cx as usize;
            grid[idx].glyph = '@';
            grid[idx].color = COLOR_MARKER_CURRENT;
        }
    }
    // Selected region: > in cyan
    if overworld.selected_region < region_positions.len()
        && overworld.selected_region != overworld.current_region
    {
        let rp = &region_positions[overworld.selected_region];
        let cx = rp.sx as i32;
        let cy = rp.sy as i32 - 1;
        if cx >= 0 && cx < MAP_W as i32 && cy >= 0 && cy < MAP_H as i32 {
            let idx = cy as usize * MAP_W + cx as usize;
            grid[idx].glyph = '>';
            grid[idx].color = COLOR_MARKER_SELECTED;
        }
    }
    // Mission markers: ! in orange
    for rp in &region_positions {
        if rp.region_idx >= overworld.regions.len() { continue; }
        let region = &overworld.regions[rp.region_idx];
        if region.mission_slot.is_some() {
            let cx = rp.sx as i32 + 1; // Right of center, above name
            let cy = rp.sy as i32 - 1;
            if cx >= 0 && cx < MAP_W as i32 && cy >= 0 && cy < MAP_H as i32 {
                let idx = cy as usize * MAP_W + cx as usize;
                grid[idx].glyph = '!';
                grid[idx].color = COLOR_MARKER_MISSION;
            }
        }
    }

    // Step 6b: Settlement type markers — place glyphs near each region center
    // Use deterministic hash of region.id to assign a type: ⌂ town, ■ castle, ▲ camp, † ruin
    for rp in &region_positions {
        if rp.region_idx >= overworld.regions.len() { continue; }
        let region = &overworld.regions[rp.region_idx];
        let fc = faction_color(region.owner_faction_id);
        let settlement_type = terrain_hash(overworld.map_seed, rp.region_idx as i32, 999) % 4;
        let glyph = match settlement_type {
            0 => '⌂', // town
            1 => '■', // castle
            2 => '▲', // camp
            _ => '†', // ruin
        };
        // Place settlement glyph to the left of the name
        let cx = (rp.sx as i32) - 4;
        let cy = rp.sy as i32;
        if cx >= 0 && cx < MAP_W as i32 && cy >= 0 && cy < MAP_H as i32 {
            let idx = cy as usize * MAP_W + cx as usize;
            grid[idx].glyph = glyph;
            grid[idx].color = fc;
        }
    }

    // Step 6c: Roaming parties — place ◆ glyphs near player-controlled party regions
    // and faction-colored ◆ for NPC parties at random offsets from region centers
    for rp in &region_positions {
        if rp.region_idx >= overworld.regions.len() { continue; }
        let region = &overworld.regions[rp.region_idx];
        let fc = faction_color(region.owner_faction_id);
        // Deterministic: each region has 0-2 roaming parties based on hash
        let party_count = terrain_hash(overworld.map_seed, rp.region_idx as i32, 777) % 3;
        for p in 0..party_count {
            let ox = (terrain_hash(overworld.map_seed, rp.region_idx as i32, 100 + p as i32) % 7) as i32 - 3;
            let oy = (terrain_hash(overworld.map_seed, rp.region_idx as i32, 200 + p as i32) % 5) as i32 - 2;
            let px = rp.sx as i32 + ox;
            let py = rp.sy as i32 + oy + 1; // Below the settlement
            if px >= 0 && px < MAP_W as i32 && py >= 0 && py < MAP_H as i32 {
                let idx = py as usize * MAP_W + px as usize;
                // Don't overwrite settlements or markers
                if grid[idx].glyph != '[' && grid[idx].glyph != ']'
                    && grid[idx].glyph != '@' && grid[idx].glyph != '>'
                    && grid[idx].glyph != '!' && grid[idx].glyph != '⌂'
                    && grid[idx].glyph != '■' && grid[idx].glyph != '▲'
                    && grid[idx].glyph != '†'
                {
                    grid[idx].glyph = '◆';
                    grid[idx].color = fc;
                }
            }
        }
    }

    // Step 7: Faction tint (blend 20% faction color with 80% terrain color)
    for cell in &mut grid {
        let fc = faction_color(
            if cell.region_id < overworld.regions.len() {
                overworld.regions[cell.region_id].owner_faction_id
            } else {
                0
            }
        );
        cell.color = blend_colors(cell.color, fc, 0.20);
    }

    // Step 8: Borders — where adjacent cells belong to different factions
    {
        let region_ids: Vec<usize> = grid.iter().map(|c| c.region_id).collect();
        let get_faction = |idx: usize| -> usize {
            let rid = region_ids[idx];
            if rid < overworld.regions.len() {
                overworld.regions[rid].owner_faction_id
            } else {
                0
            }
        };

        for y in 0..MAP_H as i32 {
            for x in 0..MAP_W as i32 {
                let idx = y as usize * MAP_W + x as usize;
                let my_faction = get_faction(idx);
                // Already a settlement label or marker? Skip border override.
                if grid[idx].glyph == '[' || grid[idx].glyph == ']'
                    || grid[idx].glyph == '@' || grid[idx].glyph == '>'
                    || grid[idx].glyph == '!'
                {
                    continue;
                }
                // Check if name character (alpha inside settlement)
                if grid[idx].color == COLOR_SETTLEMENT {
                    continue;
                }

                let mut diff_h = false;
                let mut diff_v = false;
                if x > 0 {
                    let left = y as usize * MAP_W + (x - 1) as usize;
                    if get_faction(left) != my_faction { diff_v = true; }
                }
                if x < MAP_W as i32 - 1 {
                    let right = y as usize * MAP_W + (x + 1) as usize;
                    if get_faction(right) != my_faction { diff_v = true; }
                }
                if y > 0 {
                    let up = (y - 1) as usize * MAP_W + x as usize;
                    if get_faction(up) != my_faction { diff_h = true; }
                }
                if y < MAP_H as i32 - 1 {
                    let down = (y + 1) as usize * MAP_W + x as usize;
                    if get_faction(down) != my_faction { diff_h = true; }
                }

                if diff_h && diff_v {
                    grid[idx].glyph = '┼';
                    grid[idx].color = COLOR_BORDER;
                } else if diff_h {
                    grid[idx].glyph = '─';
                    grid[idx].color = COLOR_BORDER;
                } else if diff_v {
                    grid[idx].glyph = '│';
                    grid[idx].color = COLOR_BORDER;
                }
            }
        }
    }

    grid
}

// ---------------------------------------------------------------------------
// Click detection: map character grid position to a region
// ---------------------------------------------------------------------------

fn region_from_char_click(
    overworld: &game_core::OverworldMap,
    char_x: f32,
    char_y: f32,
) -> Option<usize> {
    let region_positions = build_region_positions(overworld);
    let mut best_dist = f32::INFINITY;
    let mut best_id = None;
    for rp in &region_positions {
        let dx = char_x - rp.sx;
        let dy = char_y - rp.sy;
        let dist = dx * dx + dy * dy;
        if dist < best_dist {
            best_dist = dist;
            best_id = Some(rp.region_idx);
        }
    }
    // Only return if within reasonable radius (e.g. half the hex spacing)
    if best_dist.sqrt() < 12.0 {
        best_id
    } else {
        best_id // All cells belong to a region via Voronoi, so always return
    }
}

// ---------------------------------------------------------------------------
// Public draw function
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn draw_strategic_map(
    ui: &mut egui::Ui,
    overworld: &mut game_core::OverworldMap,
    target_picker: &mut RegionTargetPickerState,
    parties: &mut game_core::CampaignParties,
    party_snapshots: &[game_core::CampaignParty],
    transition_locked: bool,
) {
    ui.label(egui::RichText::new("Strategic Overworld Map").strong());

    let grid = generate_map(overworld);
    let font = egui::FontId::monospace(13.0);

    // Render the map as LayoutJob rows
    let _map_response = egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(4.0))
        .show(ui, |ui| {
            // Target picker mode indicator
            if target_picker.active_party_id().is_some() {
                ui.colored_label(
                    egui::Color32::from_rgb(98, 210, 252),
                    "[ Target Picker Mode — click a region on the map ]",
                );
            }

            let mut first_label_rect: Option<egui::Rect> = None;

            for y in 0..MAP_H {
                let mut job = egui::text::LayoutJob::default();
                for x in 0..MAP_W {
                    let cell = &grid[y * MAP_W + x];
                    let s: String = cell.glyph.to_string();
                    job.append(
                        &s,
                        0.0,
                        egui::TextFormat {
                            font_id: font.clone(),
                            color: cell.color,
                            ..Default::default()
                        },
                    );
                }
                let resp = ui.label(job);
                if y == 0 {
                    first_label_rect = Some(resp.rect);
                }

                // Use first row to detect clicks on any row
                if !transition_locked && resp.clicked() {
                    if let Some(pointer) = resp.interact_pointer_pos() {
                        if let Some(first_rect) = first_label_rect {
                            // Estimate character cell size from the first row
                            let char_w = first_rect.width() / MAP_W as f32;
                            let char_h = resp.rect.height();
                            let char_x = (pointer.x - resp.rect.left()) / char_w;
                            let char_y = y as f32 + (pointer.y - resp.rect.top()) / char_h;

                            if let Some(region_id) = region_from_char_click(overworld, char_x, char_y) {
                                if let Some(active_party_id) = target_picker.active_party_id() {
                                    match update_region_target_picker_selection(
                                        target_picker,
                                        active_party_id,
                                        region_id,
                                        overworld,
                                    ) {
                                        Ok(notice) => parties.notice = notice,
                                        Err(err) => parties.notice = err,
                                    }
                                } else if overworld.regions.get(region_id).is_some() {
                                    overworld.selected_region = region_id;
                                    let region_name = overworld
                                        .regions
                                        .get(region_id)
                                        .map(|r| r.name.as_str())
                                        .unwrap_or("Unknown");
                                    parties.notice =
                                        format!("Map selection updated to {}.", region_name);
                                }
                            }
                        }
                    }
                }
            }

            // Party markers overlay — show party positions in the map area
            // (parties are shown as annotations below the map since we can't
            // easily overlay on the text grid)
            let has_parties = !party_snapshots.is_empty();
            if has_parties {
                let region_positions = build_region_positions(overworld);
                let mut party_job = egui::text::LayoutJob::default();
                party_job.append(
                    "Parties: ",
                    0.0,
                    egui::TextFormat {
                        font_id: font.clone(),
                        color: COLOR_LEGEND,
                        ..Default::default()
                    },
                );
                for party in party_snapshots {
                    if party.region_id >= region_positions.len() { continue; }
                    let region_name = overworld.regions.get(party.region_id)
                        .map(|r| &r.name[..r.name.len().min(4)])
                        .unwrap_or("??");
                    let marker = if party.is_player_controlled { "P" } else { "◆" };
                    let color = if party.is_player_controlled {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_rgb(200, 200, 200)
                    };
                    party_job.append(
                        &format!("{}@{} ", marker, region_name),
                        4.0,
                        egui::TextFormat {
                            font_id: font.clone(),
                            color,
                            ..Default::default()
                        },
                    );
                }
                ui.label(party_job);
            }
        });

    // Legend
    let legend_font = egui::FontId::monospace(11.0);
    ui.horizontal_wrapped(|ui| {
        let items: &[(&str, egui::Color32)] = &[
            (". plains", COLOR_PLAINS),
            ("♣ forest", COLOR_FOREST),
            ("~ hills", COLOR_HILLS),
            ("^ mountain", COLOR_MOUNTAIN),
            ("▲ peak", COLOR_PEAK),
            ("≈ water", COLOR_WATER),
            ("═ road", COLOR_ROAD),
            (", coast", COLOR_COAST),
            ("─│ border", COLOR_BORDER),
            ("⌂ town", COLOR_SETTLEMENT),
            ("■ castle", COLOR_SETTLEMENT),
            ("◆ party", COLOR_LEGEND),
        ];
        for (text, color) in items {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                text,
                0.0,
                egui::TextFormat {
                    font_id: legend_font.clone(),
                    color: *color,
                    ..Default::default()
                },
            );
            ui.label(job);
        }
    });
    ui.horizontal_wrapped(|ui| {
        let markers: &[(&str, egui::Color32)] = &[
            ("@ you", COLOR_MARKER_CURRENT),
            ("> selected", COLOR_MARKER_SELECTED),
            ("! mission", COLOR_MARKER_MISSION),
        ];
        for (text, color) in markers {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                text,
                0.0,
                egui::TextFormat {
                    font_id: legend_font.clone(),
                    color: *color,
                    ..Default::default()
                },
            );
            ui.label(job);
        }
        // Faction colors
        for faction in &overworld.factions {
            let mut job = egui::text::LayoutJob::default();
            let fc = faction_color(faction.id);
            job.append(
                &format!("■ {}", &faction.name[..faction.name.len().min(8)]),
                0.0,
                egui::TextFormat {
                    font_id: legend_font.clone(),
                    color: fc,
                    ..Default::default()
                },
            );
            ui.label(job);
        }
    });
}
