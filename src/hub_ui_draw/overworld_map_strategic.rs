//! Overworld strategic map — hex-tile rendering and click handling.

use bevy_egui::egui;
use std::collections::VecDeque;

use crate::game_core;
use crate::region_nav::{
    RegionTargetPickerState, update_region_target_picker_selection,
    region_from_map_click,
};
use super::faction_color;

#[allow(clippy::too_many_arguments)]
pub(super) fn draw_strategic_map(
    ui: &mut egui::Ui,
    overworld: &mut game_core::OverworldMap,
    target_picker: &mut RegionTargetPickerState,
    parties: &mut game_core::CampaignParties,
    party_snapshots: &[game_core::CampaignParty],
    transition_locked: bool,
) {
    ui.label(egui::RichText::new("Strategic Overworld Map").strong());
    egui::Frame::none()
        .fill(egui::Color32::from_rgb(13, 16, 22))
        .inner_margin(egui::Margin::same(8.0))
        .show(ui, |ui| {
            let desired = egui::vec2(ui.available_width().max(420.0), 360.0);
            let (response, painter) = ui.allocate_painter(desired, egui::Sense::click());
            let rect = response.rect;
            painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(24, 28, 30));
            let world_points = game_core::overworld_region_plot_positions(overworld);
            let mut points = vec![rect.center(); overworld.regions.len()];
            let margin = 24.0;
            if !world_points.is_empty() {
                let mut min_x = f32::INFINITY;
                let mut max_x = f32::NEG_INFINITY;
                let mut min_y = f32::INFINITY;
                let mut max_y = f32::NEG_INFINITY;
                for (x, y) in &world_points {
                    min_x = min_x.min(*x);
                    max_x = max_x.max(*x);
                    min_y = min_y.min(*y);
                    max_y = max_y.max(*y);
                }
                let world_w = (max_x - min_x).max(0.001);
                let world_h = (max_y - min_y).max(0.001);
                let draw_w = (rect.width() - margin * 2.0).max(1.0);
                let draw_h = (rect.height() - margin * 2.0).max(1.0);
                let scale = (draw_w / world_w).min(draw_h / world_h);
                let ox = rect.left() + (rect.width() - world_w * scale) * 0.5;
                let oy = rect.top() + (rect.height() - world_h * scale) * 0.5;

                for region in &overworld.regions {
                    if region.id >= world_points.len() || region.id >= points.len() {
                        continue;
                    }
                    let (wx, wy) = world_points[region.id];
                    let sx = ox + (wx - min_x) * scale;
                    let sy = oy + (wy - min_y) * scale;
                    points[region.id] = egui::pos2(sx, sy);
                }
            }

            let rand01 = |key: u64| -> f32 {
                let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f32) / ((1u64 << 53) as f32)
            };
            if target_picker.active_party_id().is_some() {
                painter.rect_stroke(
                    rect.shrink(2.0),
                    6.0,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(98, 210, 252)),
                );
            }
            // Terrain circles
            for region in &overworld.regions {
                let pos = points[region.id];
                let t = rand01(overworld.map_seed ^ (region.id as u64).wrapping_mul(997));
                let terrain = if t < 0.33 {
                    egui::Color32::from_rgba_premultiplied(52, 72, 54, 65)
                } else if t < 0.66 {
                    egui::Color32::from_rgba_premultiplied(78, 82, 54, 55)
                } else {
                    egui::Color32::from_rgba_premultiplied(64, 60, 52, 55)
                };
                painter.circle_filled(pos, 28.0, terrain);
            }
            // Edge distances for hex sizing
            let mut min_edge = f32::INFINITY;
            for region in &overworld.regions {
                let a = points[region.id];
                for n in &region.neighbors {
                    if *n > region.id && *n < points.len() {
                        min_edge = min_edge.min(a.distance(points[*n]));
                    }
                }
            }
            let hex_radius = if min_edge.is_finite() {
                (min_edge * 0.46).clamp(10.0, 24.0)
            } else {
                16.0
            };

            // Click handling
            if !transition_locked && response.clicked() {
                if let Some(pointer) = response.interact_pointer_pos() {
                    if let Some(region_id) =
                        region_from_map_click(&points, pointer, hex_radius + 10.0)
                    {
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

            let hex_points = |center: egui::Pos2, radius: f32| -> Vec<egui::Pos2> {
                (0..6)
                    .map(|i| {
                        let a = (i as f32 / 6.0) * std::f32::consts::TAU
                            + std::f32::consts::FRAC_PI_6;
                        egui::pos2(center.x + radius * a.cos(), center.y + radius * a.sin())
                    })
                    .collect()
            };

            // Hex fills
            for region in &overworld.regions {
                let pos = points[region.id];
                let owner_color = faction_color(region.owner_faction_id);
                let fill = egui::Color32::from_rgba_premultiplied(
                    owner_color.r(),
                    owner_color.g(),
                    owner_color.b(),
                    78,
                );
                painter.add(egui::Shape::convex_polygon(
                    hex_points(pos, hex_radius),
                    fill,
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_premultiplied(150, 165, 188, 90),
                    ),
                ));
            }

            // Connections + front-line borders
            for region in &overworld.regions {
                let a = points[region.id];
                for n in &region.neighbors {
                    if *n > region.id && *n < points.len() {
                        let b = points[*n];
                        painter.line_segment(
                            [a, b],
                            egui::Stroke::new(
                                2.0,
                                egui::Color32::from_rgba_premultiplied(140, 150, 160, 120),
                            ),
                        );
                        if region.owner_faction_id != overworld.regions[*n].owner_faction_id {
                            painter.line_segment(
                                [a, b],
                                egui::Stroke::new(
                                    4.0,
                                    egui::Color32::from_rgba_premultiplied(245, 230, 175, 110),
                                ),
                            );
                        }
                    }
                }
            }

            // Mission slot icons
            for region in &overworld.regions {
                if region.mission_slot.is_none() {
                    continue;
                }
                let pos = points[region.id];
                let owner_color = faction_color(region.owner_faction_id);
                let s = 5.0;
                let icon = vec![
                    egui::pos2(pos.x, pos.y - s - 3.0),
                    egui::pos2(pos.x + s, pos.y - 3.0),
                    egui::pos2(pos.x, pos.y + s - 3.0),
                    egui::pos2(pos.x - s, pos.y - 3.0),
                ];
                painter.add(egui::Shape::convex_polygon(
                    icon,
                    owner_color,
                    egui::Stroke::new(1.0, egui::Color32::BLACK),
                ));
            }

            // BFS path from current to selected
            let current =
                overworld.current_region.min(overworld.regions.len().saturating_sub(1));
            let selected =
                overworld.selected_region.min(overworld.regions.len().saturating_sub(1));
            if !overworld.regions.is_empty()
                && current < points.len()
                && selected < points.len()
            {
                let mut came_from = vec![usize::MAX; overworld.regions.len()];
                let mut queue = VecDeque::new();
                queue.push_back(current);
                came_from[current] = current;
                while let Some(node) = queue.pop_front() {
                    if node == selected {
                        break;
                    }
                    for next in &overworld.regions[node].neighbors {
                        if *next < came_from.len() && came_from[*next] == usize::MAX {
                            came_from[*next] = node;
                            queue.push_back(*next);
                        }
                    }
                }
                if came_from[selected] != usize::MAX {
                    let mut chain = vec![selected];
                    let mut walk = selected;
                    while walk != current {
                        walk = came_from[walk];
                        chain.push(walk);
                    }
                    chain.reverse();
                    for pair in chain.windows(2) {
                        let a = points[pair[0]];
                        let b = points[pair[1]];
                        painter.line_segment(
                            [a, b],
                            egui::Stroke::new(
                                3.0,
                                egui::Color32::from_rgba_premultiplied(255, 235, 110, 210),
                            ),
                        );
                    }
                }
            }

            // Region nodes + markers
            for region in &overworld.regions {
                let pos = points[region.id];
                let owner_color = faction_color(region.owner_faction_id);
                let mut node_radius = 7.0;
                if region.id == overworld.selected_region {
                    node_radius = 10.0;
                } else if region.id == overworld.current_region {
                    node_radius = 9.0;
                }
                painter.circle_filled(pos, node_radius, owner_color);
                if region.id == overworld.current_region {
                    painter.circle_stroke(
                        pos,
                        node_radius + 4.0,
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                }
                if region.id == overworld.selected_region {
                    painter.circle_stroke(
                        pos,
                        node_radius + 2.0,
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(255, 245, 145)),
                    );
                }
                if region.id == overworld.current_region {
                    let tip = egui::pos2(pos.x, pos.y - node_radius - 13.0);
                    let left = egui::pos2(pos.x - 7.0, pos.y - node_radius - 2.0);
                    let right = egui::pos2(pos.x + 7.0, pos.y - node_radius - 2.0);
                    painter.add(egui::Shape::convex_polygon(
                        vec![tip, left, right],
                        egui::Color32::WHITE,
                        egui::Stroke::new(1.0, egui::Color32::BLACK),
                    ));
                }
                if target_picker.selected_region_id() == Some(region.id) {
                    painter.circle_stroke(
                        pos,
                        node_radius + 7.0,
                        egui::Stroke::new(2.5, egui::Color32::from_rgb(110, 255, 170)),
                    );
                }
            }

            // Party markers
            for party in party_snapshots {
                if party.region_id >= points.len() {
                    continue;
                }
                let pos = points[party.region_id];
                if party.is_player_controlled {
                    painter.circle_stroke(
                        pos,
                        14.0,
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                    painter.circle_filled(pos, 2.6, egui::Color32::WHITE);
                } else {
                    painter.circle_filled(
                        egui::pos2(pos.x + 8.0, pos.y + 8.0),
                        3.0,
                        egui::Color32::from_rgb(200, 200, 200),
                    );
                }
            }
        });

    // Map legend
    ui.horizontal_wrapped(|ui| {
        ui.small("Legend:");
        ui.small("faction tint = territorial control");
        ui.small("thick pale borders = front lines");
        ui.small("gold path = projected travel route");
        ui.small("white marker = player party");
        if target_picker.active_party_id().is_some() {
            ui.small("cyan frame = target picker mode");
        }
    });
}
