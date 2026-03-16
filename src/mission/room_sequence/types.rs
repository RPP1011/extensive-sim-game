use bevy::prelude::*;

use crate::ai::core::SimVec2;
use crate::game_core::RoomType;

// ---------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------

/// Defines the ordered sequence of rooms for the current mission and tracks
/// which room is currently active.
#[derive(Resource)]
pub struct MissionRoomSequence {
    /// The room types to visit in order (e.g. `[Entry, Pressure, Climax]`).
    pub rooms: Vec<RoomType>,
    /// Optional per-room text prompts for ML generation.
    pub prompts: Vec<Option<String>>,
    /// Index of the room that is currently loaded and active.
    pub current_index: usize,
    /// Base seed; individual rooms use `seed + current_index` for variance.
    pub seed: u64,
    /// World-space origin of the room that is currently loaded.
    pub current_room_origin: Vec3,
}

impl MissionRoomSequence {
    /// Build a new sequence appropriate for the given difficulty level.
    pub fn new(difficulty: u32, seed: u64) -> Self {
        let rooms = match difficulty {
            0..=2 => vec![RoomType::Entry, RoomType::Climax],
            3..=4 => vec![RoomType::Entry, RoomType::Pressure, RoomType::Climax],
            _ => vec![
                RoomType::Entry,
                RoomType::Pressure,
                RoomType::Pivot,
                RoomType::Climax,
            ],
        };
        let prompts = vec![None; rooms.len()];
        Self {
            rooms,
            prompts,
            current_index: 0,
            seed,
            current_room_origin: Vec3::ZERO,
        }
    }

    /// The `RoomType` of the room that is currently active, or `None` if the
    /// sequence is exhausted.
    pub fn current_room_type(&self) -> Option<&RoomType> {
        self.rooms.get(self.current_index)
    }

    /// Returns `true` when there is no room after the current one.
    pub fn is_last_room(&self) -> bool {
        self.current_index + 1 >= self.rooms.len()
    }
}

// ---------------------------------------------------------------------------
// Door component
// ---------------------------------------------------------------------------

/// Marks the entity used as the "advance to next room" trigger.
#[derive(Component)]
pub struct RoomDoor;

// ---------------------------------------------------------------------------
// Helper: spawn boss visual
// ---------------------------------------------------------------------------

/// Spawns a visually distinct entity for the climax-room boss.
pub(crate) fn spawn_boss_visual(
    sim_unit_id: u32,
    position: SimVec2,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let world_pos = Vec3::new(position.x, 0.0, position.y);

    let body_material = materials.add(StandardMaterial {
        base_color: Color::rgb(1.0, 0.84, 0.0),
        emissive: Color::rgb(0.6, 0.4, 0.0),
        metallic: 0.3,
        perceptual_roughness: 0.5,
        ..default()
    });
    let body_mesh = meshes.add(Capsule3d {
        radius: 0.3,
        half_length: 0.5,
    });

    let bar_bg_mesh = meshes.add(Cuboid::new(0.8, 0.08, 0.08));
    let bar_bg_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.2, 0.2, 0.2),
        ..default()
    });

    let bar_fg_mesh = meshes.add(Cuboid::new(0.8, 0.08, 0.08));
    let bar_fg_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.1, 0.9, 0.1),
        ..default()
    });

    use crate::mission::unit_vis::{HpBarBg, HpBarFg, UnitVisual};
    use crate::ai::core::Team;

    commands
        .spawn((
            SpatialBundle {
                transform: Transform {
                    translation: world_pos,
                    scale: Vec3::splat(1.5),
                    ..default()
                },
                ..default()
            },
            UnitVisual { sim_unit_id, team: Team::Enemy },
            Name::new("Boss"),
        ))
        .with_children(|parent| {
            parent.spawn(PbrBundle {
                mesh: body_mesh,
                material: body_material,
                transform: Transform::from_xyz(0.0, 0.8, 0.0),
                ..default()
            });
            parent.spawn((
                PbrBundle {
                    mesh: bar_bg_mesh,
                    material: bar_bg_material,
                    transform: Transform::from_xyz(0.0, 2.0, 0.0),
                    ..default()
                },
                HpBarBg,
            ));
            parent.spawn((
                PbrBundle {
                    mesh: bar_fg_mesh,
                    material: bar_fg_material,
                    transform: Transform::from_xyz(0.0, 2.0, 0.01),
                    ..default()
                },
                HpBarFg,
            ));
        });
}
