use bevy::prelude::*;
use crate::ai::core::{Team, SimVec2};

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marks a Bevy entity as the visual representation of a sim unit.
#[derive(Component)]
pub struct UnitVisual {
    pub sim_unit_id: u32,
    pub team: Team,
}

/// Marks the selection ring child entity.
#[derive(Component)]
pub struct SelectionRing;

/// Marks the HP bar foreground child entity.
#[derive(Component)]
pub struct HpBarFg;

/// Marks the HP bar background child entity.
#[derive(Component)]
pub struct HpBarBg;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Tracks which unit IDs are currently selected.
#[derive(Resource, Default)]
pub struct UnitSelection {
    pub selected_ids: Vec<u32>,
}

/// Maps sim_unit_id -> (current_hp, max_hp).
#[derive(Resource, Default)]
pub struct UnitHealthData {
    pub hp: std::collections::HashMap<u32, (i32, i32)>,
}

/// Maps sim_unit_id -> world position (x, z).
#[derive(Resource, Default)]
pub struct UnitPositionData {
    pub positions: std::collections::HashMap<u32, (f32, f32)>,
}

// ---------------------------------------------------------------------------
// Spawn function
// ---------------------------------------------------------------------------

/// Spawns a 3-D visual representation for one sim unit and returns the root entity.
///
/// Children:
///  1. Body capsule  – team-coloured PBR mesh
///  2. Selection ring – torus, initially hidden
///  3. HP bar background – dark grey cuboid
///  4. HP bar foreground – green cuboid (scale.x driven by health)
pub fn spawn_unit_visual(
    sim_unit_id: u32,
    team: Team,
    position: SimVec2,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) -> Entity {
    // World position: sim X -> world X, sim Y -> world Z
    let world_pos = Vec3::new(position.x, 0.0, position.y);

    // ---- Body material ----
    let body_color = match team {
        Team::Hero  => Color::rgb(0.13, 0.33, 1.0),
        Team::Enemy => Color::rgb(1.0,  0.13, 0.13),
    };
    let body_material = materials.add(StandardMaterial {
        base_color: body_color,
        metallic:   0.0,
        perceptual_roughness: 0.7,
        ..default()
    });

    // ---- Body mesh ----
    let body_mesh = meshes.add(Capsule3d {
        radius:      0.3,
        half_length: 0.5,
    });

    // ---- Selection ring ----
    let ring_mesh = meshes.add(Torus {
        minor_radius: 0.05,
        major_radius: 0.5,
    });
    let ring_material = materials.add(StandardMaterial {
        base_color: Color::rgb(1.0, 0.9, 0.2),
        ..default()
    });

    // ---- HP bar background ----
    let bar_bg_mesh = meshes.add(Cuboid::new(0.8, 0.08, 0.08));
    let bar_bg_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.2, 0.2, 0.2),
        ..default()
    });

    // ---- HP bar foreground ----
    let bar_fg_mesh = meshes.add(Cuboid::new(0.8, 0.08, 0.08));
    let bar_fg_material = materials.add(StandardMaterial {
        base_color: Color::rgb(0.1, 0.9, 0.1),
        ..default()
    });

    // ---- Spawn root + children ----
    commands
        .spawn((
            SpatialBundle {
                transform: Transform::from_translation(world_pos),
                ..default()
            },
            UnitVisual { sim_unit_id, team },
        ))
        .with_children(|parent| {
            // 1. Body
            parent.spawn(PbrBundle {
                mesh:      body_mesh,
                material:  body_material,
                transform: Transform::from_xyz(0.0, 0.8, 0.0),
                ..default()
            });

            // 2. Selection ring (hidden by default)
            parent.spawn((
                PbrBundle {
                    mesh:       ring_mesh,
                    material:   ring_material,
                    transform:  Transform::from_xyz(0.0, 0.05, 0.0),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                SelectionRing,
            ));

            // 3. HP bar background
            parent.spawn((
                PbrBundle {
                    mesh:      bar_bg_mesh,
                    material:  bar_bg_material,
                    transform: Transform::from_xyz(0.0, 2.0, 0.0),
                    ..default()
                },
                HpBarBg,
            ));

            // 4. HP bar foreground (slightly in front on Z)
            parent.spawn((
                PbrBundle {
                    mesh:      bar_fg_mesh,
                    material:  bar_fg_material,
                    transform: Transform::from_xyz(0.0, 2.0, 0.01),
                    ..default()
                },
                HpBarFg,
            ));
        })
        .id()
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Shows or hides selection rings based on `UnitSelection`.
pub fn update_unit_selection_rings(
    selection:  Res<UnitSelection>,
    unit_query: Query<(&UnitVisual, &Children)>,
    mut ring_query: Query<&mut Visibility, With<SelectionRing>>,
) {
    for (unit, children) in unit_query.iter() {
        let should_show = selection.selected_ids.contains(&unit.sim_unit_id);
        for &child in children.iter() {
            if let Ok(mut vis) = ring_query.get_mut(child) {
                *vis = if should_show {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
}

/// Scales the HP bar foreground according to current health and shifts it so
/// the bar shrinks from the right.
pub fn update_hp_bars(
    health:     Res<UnitHealthData>,
    unit_query: Query<(&UnitVisual, &Children)>,
    mut fg_query: Query<&mut Transform, With<HpBarFg>>,
) {
    for (unit, children) in unit_query.iter() {
        let hp_fraction = if let Some(&(current, max)) = health.hp.get(&unit.sim_unit_id) {
            if max > 0 {
                (current as f32 / max as f32).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            1.0 // unknown -> assume full health
        };

        for &child in children.iter() {
            if let Ok(mut transform) = fg_query.get_mut(child) {
                transform.scale.x = hp_fraction;
                // Shift left so the bar shrinks from the right edge
                transform.translation.x = (hp_fraction - 1.0) * 0.4;
            }
        }
    }
}

/// Moves unit root entities to match simulated positions.
pub fn update_unit_positions(
    sim_positions: Res<UnitPositionData>,
    mut unit_query: Query<(&UnitVisual, &mut Transform)>,
) {
    for (unit, mut transform) in unit_query.iter_mut() {
        if let Some(&(x, z)) = sim_positions.positions.get(&unit.sim_unit_id) {
            transform.translation.x = x;
            transform.translation.z = z;
        }
    }
}
