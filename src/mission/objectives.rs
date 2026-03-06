use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::ai::core::Team;
use crate::game_core;
use crate::mission::room_gen::RoomLayout;
use crate::mission::sim_bridge::{MissionOutcome, MissionSimState};

// ---------------------------------------------------------------------------
// LCG — mirrors the one in room_gen so we need no new crates
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        let s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut lcg = Self(s);
        for _ in 0..8 {
            lcg.next_u64();
        }
        lcg
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Return a value in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ObjectiveKind {
    /// Kill all enemies in the room (default).
    Eliminate,
    /// Keep a squad member within the zone for N accumulated ticks.
    Hold {
        zone_center: (f32, f32),
        zone_radius: f32,
        ticks_required: u32,
        ticks_held: u32,
    },
    /// Move a designated slow NPC (sim unit id) to the exit zone.
    Extract {
        npc_unit_id: u32,
        exit_zone: (f32, f32),
        exit_radius: f32,
    },
    /// Interact with N nodes — hero must be within 1.0 units; auto-activates.
    Sabotage {
        nodes: Vec<(f32, f32)>,
        nodes_activated: Vec<bool>,
    },
}

#[derive(Debug, Clone)]
pub struct RoomObjective {
    pub kind: ObjectiveKind,
    pub description: String,
    pub completed: bool,
    pub failed: bool,
}

#[derive(Resource, Default)]
pub struct MissionObjectiveState {
    pub current_objective: Option<RoomObjective>,
    pub completed_objectives: Vec<RoomObjective>,
}

// ---------------------------------------------------------------------------
// Objective generation
// ---------------------------------------------------------------------------

/// Generate a `RoomObjective` for the given room type and layout.
///
/// Uses a seeded LCG so the result is deterministic for a given (seed, room_type).
pub fn generate_objective(
    room_type: &game_core::RoomType,
    layout: &RoomLayout,
    seed: u64,
) -> RoomObjective {
    let mut rng = Lcg::new(seed);
    let roll = rng.next_f32(); // [0, 1)

    let cx = layout.width * 0.5;
    let cz = layout.depth * 0.5;

    let kind = match room_type {
        // Entry: always Eliminate
        game_core::RoomType::Entry => ObjectiveKind::Eliminate,

        // Recovery: always Eliminate (relief room)
        game_core::RoomType::Recovery => ObjectiveKind::Eliminate,

        // Climax: always Eliminate (boss room)
        game_core::RoomType::Climax => ObjectiveKind::Eliminate,

        // Pressure: 60% Eliminate, 40% Hold
        game_core::RoomType::Pressure => {
            if roll < 0.60 {
                ObjectiveKind::Eliminate
            } else {
                ObjectiveKind::Hold {
                    zone_center: (cx, cz),
                    zone_radius: 3.0,
                    ticks_required: 200,
                    ticks_held: 0,
                }
            }
        }

        // Pivot: 50% Eliminate, 30% Hold, 20% Sabotage (2 nodes)
        game_core::RoomType::Pivot => {
            if roll < 0.50 {
                ObjectiveKind::Eliminate
            } else if roll < 0.80 {
                ObjectiveKind::Hold {
                    zone_center: (cx, cz),
                    zone_radius: 3.0,
                    ticks_required: 200,
                    ticks_held: 0,
                }
            } else {
                ObjectiveKind::Sabotage {
                    nodes: sabotage_nodes(cx, cz, 2),
                    nodes_activated: vec![false; 2],
                }
            }
        }

        // Setpiece: 40% Eliminate, 30% Extract, 30% Sabotage (3 nodes)
        game_core::RoomType::Setpiece => {
            if roll < 0.40 {
                ObjectiveKind::Eliminate
            } else if roll < 0.70 {
                // Exit zone is on the player-spawn side: low-Z edge of the room.
                let exit_z = layout.depth * 0.1;
                ObjectiveKind::Extract {
                    npc_unit_id: 9999,
                    exit_zone: (cx, exit_z),
                    exit_radius: 3.0,
                }
            } else {
                ObjectiveKind::Sabotage {
                    nodes: sabotage_nodes(cx, cz, 3),
                    nodes_activated: vec![false; 3],
                }
            }
        }
    };

    let description = describe_kind(&kind);
    RoomObjective { kind, description, completed: false, failed: false }
}

/// Place `n` sabotage nodes evenly around the room centre.
fn sabotage_nodes(cx: f32, cz: f32, n: usize) -> Vec<(f32, f32)> {
    // Space nodes in a horizontal line through the centre.
    let spacing = 4.0_f32;
    let total_span = spacing * (n as f32 - 1.0);
    let start_x = cx - total_span * 0.5;
    (0..n).map(|i| (start_x + i as f32 * spacing, cz)).collect()
}

fn describe_kind(kind: &ObjectiveKind) -> String {
    match kind {
        ObjectiveKind::Eliminate => "Defeat all hostiles.".to_string(),
        ObjectiveKind::Hold { .. } => {
            "Hold the centre for 20 seconds while under attack.".to_string()
        }
        ObjectiveKind::Extract { .. } => "Escort the survivor to the exit.".to_string(),
        ObjectiveKind::Sabotage { nodes, .. } => {
            format!("Disable {} wards before the ritual completes.", nodes.len())
        }
    }
}

// ---------------------------------------------------------------------------
// Reset helper
// ---------------------------------------------------------------------------

/// Replace the current objective with a freshly generated one for a new room.
pub fn reset_objective(state: &mut MissionObjectiveState, obj: RoomObjective) {
    if let Some(prev) = state.current_objective.take() {
        state.completed_objectives.push(prev);
    }
    state.current_objective = Some(obj);
}

// ---------------------------------------------------------------------------
// check_objective_system
// ---------------------------------------------------------------------------

/// Evaluates the current objective each frame and sets `MissionOutcome` when done.
pub fn check_objective_system(
    mut obj_state: ResMut<MissionObjectiveState>,
    mut sim_state: ResMut<MissionSimState>,
) {
    // Only evaluate while the mission is still in progress.
    if sim_state.outcome.is_some() {
        return;
    }

    let Some(ref mut obj) = obj_state.current_objective else {
        return;
    };

    if obj.completed || obj.failed {
        return;
    }

    match &mut obj.kind {
        // ----------------------------------------------------------------
        // Eliminate — complete when all enemies are dead
        // ----------------------------------------------------------------
        ObjectiveKind::Eliminate => {
            let all_enemies_dead = sim_state
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy)
                .all(|u| u.hp <= 0);

            if all_enemies_dead {
                obj.completed = true;
            }
        }

        // ----------------------------------------------------------------
        // Hold — increment ticks while any hero is in the zone
        // ----------------------------------------------------------------
        ObjectiveKind::Hold {
            zone_center,
            zone_radius,
            ticks_required,
            ticks_held,
        } => {
            let (zx, zy) = *zone_center;
            let r = *zone_radius;

            let any_hero_in_zone = sim_state.sim.units.iter().any(|u| {
                u.team == Team::Hero
                    && u.hp > 0
                    && dist2d(u.position.x, u.position.y, zx, zy) <= r
            });

            if any_hero_in_zone {
                *ticks_held += 1;
            }

            if *ticks_held >= *ticks_required {
                obj.completed = true;
            }
        }

        // ----------------------------------------------------------------
        // Extract — check NPC position / death
        // ----------------------------------------------------------------
        ObjectiveKind::Extract {
            npc_unit_id,
            exit_zone,
            exit_radius,
        } => {
            let npc_id = *npc_unit_id;
            let (ex, ey) = *exit_zone;
            let er = *exit_radius;

            if let Some(npc) = sim_state.sim.units.iter().find(|u| u.id == npc_id) {
                if npc.hp <= 0 {
                    // NPC died — objective failed.
                    obj.failed = true;
                } else if dist2d(npc.position.x, npc.position.y, ex, ey) <= er {
                    obj.completed = true;
                }
            }
            // If the NPC unit doesn't exist yet, do nothing — it may be spawned later.
        }

        // ----------------------------------------------------------------
        // Sabotage — auto-activate nodes when a hero walks within 1.0 unit
        // ----------------------------------------------------------------
        ObjectiveKind::Sabotage { nodes, nodes_activated } => {
            for (i, &(nx, ny)) in nodes.iter().enumerate() {
                if nodes_activated[i] {
                    continue;
                }
                let hero_nearby = sim_state.sim.units.iter().any(|u| {
                    u.team == Team::Hero
                        && u.hp > 0
                        && dist2d(u.position.x, u.position.y, nx, ny) <= 1.0
                });
                if hero_nearby {
                    nodes_activated[i] = true;
                }
            }

            if nodes_activated.iter().all(|&a| a) {
                obj.completed = true;
            }
        }
    }

    // When objective just completed, record the mission outcome.
    if obj.completed && sim_state.outcome.is_none() {
        sim_state.outcome = Some(MissionOutcome::Victory);
    }
}

// ---------------------------------------------------------------------------
// draw_objective_hud_system
// ---------------------------------------------------------------------------

/// Renders a top-centre egui panel with the current objective and progress.
pub fn draw_objective_hud_system(
    mut contexts: EguiContexts,
    obj_state: Res<MissionObjectiveState>,
    sim_state: Option<Res<MissionSimState>>,
) {
    let ctx = contexts.ctx_mut();

    let Some(ref obj) = obj_state.current_objective else {
        return;
    };

    egui::TopBottomPanel::top("objective_hud").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading(&obj.description);
        });
        ui.separator();

        match &obj.kind {
            ObjectiveKind::Eliminate => {
                let enemies_remaining = sim_state
                    .as_ref()
                    .map(|s| {
                        s.sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count()
                    })
                    .unwrap_or(0);
                ui.label(format!("Enemies remaining: {}", enemies_remaining));
            }

            ObjectiveKind::Hold { ticks_held, ticks_required, .. } => {
                let progress = *ticks_held as f32 / (*ticks_required as f32).max(1.0);
                let progress_bar = egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                    .text(format!("{} / {} ticks", ticks_held, ticks_required));
                ui.add(progress_bar);
            }

            ObjectiveKind::Extract { npc_unit_id, .. } => {
                let npc_id = *npc_unit_id;
                let status = sim_state
                    .as_ref()
                    .and_then(|s| s.sim.units.iter().find(|u| u.id == npc_id))
                    .map(|npc| {
                        // Consider the survivor endangered if any enemy is within 5 units.
                        let any_enemy_close = sim_state.as_ref().map_or(false, |s| {
                            s.sim.units.iter().any(|u| {
                                u.team == Team::Enemy
                                    && u.hp > 0
                                    && dist2d(
                                        u.position.x,
                                        u.position.y,
                                        npc.position.x,
                                        npc.position.y,
                                    ) <= 5.0
                            })
                        });
                        if any_enemy_close { "endangered" } else { "safe" }
                    })
                    .unwrap_or("unknown");
                ui.label(format!("Survivor: [{}]", status));
            }

            ObjectiveKind::Sabotage { nodes, nodes_activated } => {
                let activated = nodes_activated.iter().filter(|&&a| a).count();
                let total = nodes.len();
                ui.label(format!("Wards disabled: {} / {}", activated, total));
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn dist2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mission::room_gen::generate_room;

    #[test]
    fn entry_room_always_eliminate() {
        let layout = generate_room(42, game_core::RoomType::Entry);
        let obj = generate_objective(&game_core::RoomType::Entry, &layout, 42);
        assert!(matches!(obj.kind, ObjectiveKind::Eliminate));
    }

    #[test]
    fn recovery_room_always_eliminate() {
        let layout = generate_room(7, game_core::RoomType::Recovery);
        let obj = generate_objective(&game_core::RoomType::Recovery, &layout, 7);
        assert!(matches!(obj.kind, ObjectiveKind::Eliminate));
    }

    #[test]
    fn climax_room_always_eliminate() {
        let layout = generate_room(99, game_core::RoomType::Climax);
        let obj = generate_objective(&game_core::RoomType::Climax, &layout, 99);
        assert!(matches!(obj.kind, ObjectiveKind::Eliminate));
    }

    #[test]
    fn sabotage_nodes_count_pivot() {
        // Across a broad seed range at least one Pivot room should pick Sabotage.
        // Seed 3 produces roll > 0.80 for Pivot with this LCG.
        let found_sabotage = (0u64..200).any(|seed| {
            let layout = generate_room(seed, game_core::RoomType::Pivot);
            let obj = generate_objective(&game_core::RoomType::Pivot, &layout, seed);
            matches!(&obj.kind, ObjectiveKind::Sabotage { nodes, .. } if nodes.len() == 2)
        });
        assert!(found_sabotage, "Expected at least one Pivot room with 2-node Sabotage");
    }

    #[test]
    fn sabotage_nodes_count_setpiece() {
        let found_sabotage = (0u64..200).any(|seed| {
            let layout = generate_room(seed, game_core::RoomType::Setpiece);
            let obj = generate_objective(&game_core::RoomType::Setpiece, &layout, seed);
            matches!(&obj.kind, ObjectiveKind::Sabotage { nodes, .. } if nodes.len() == 3)
        });
        assert!(found_sabotage, "Expected at least one Setpiece room with 3-node Sabotage");
    }

    #[test]
    fn descriptions_are_non_empty() {
        for rt in [
            game_core::RoomType::Entry,
            game_core::RoomType::Pressure,
            game_core::RoomType::Pivot,
            game_core::RoomType::Setpiece,
            game_core::RoomType::Recovery,
            game_core::RoomType::Climax,
        ] {
            let layout = generate_room(1, rt);
            let obj = generate_objective(&rt, &layout, 1);
            assert!(!obj.description.is_empty());
        }
    }

    #[test]
    fn reset_objective_moves_old_to_completed() {
        let mut state = MissionObjectiveState::default();
        let layout = generate_room(1, game_core::RoomType::Entry);
        let obj1 = generate_objective(&game_core::RoomType::Entry, &layout, 1);
        let obj2 = generate_objective(&game_core::RoomType::Entry, &layout, 2);
        reset_objective(&mut state, obj1);
        reset_objective(&mut state, obj2);
        assert_eq!(state.completed_objectives.len(), 1);
        assert!(state.current_objective.is_some());
    }
}
