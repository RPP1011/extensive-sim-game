use bevy::prelude::*;

use crate::ai::core::{self, IntentAction, SimEvent, SimState, SimVec2, Team, UnitIntent, UnitStore, FIXED_TICK_MS};
use crate::ai::pathing::{cover_factor, GridNav};
use crate::audio::{AudioEvent, AudioEventQueue, SfxKind};
use crate::mission::unit_vis::{UnitHealthData, UnitPositionData};
use crate::mission::vfx::{VfxEvent, VfxEventQueue};

use super::types::*;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Advances the sim by 100 ms fixed steps, syncs position/health data, and
/// writes all produced `SimEvent`s into `SimEventBuffer` for downstream
/// consumers.
pub fn advance_sim_system(
    time: Res<Time>,
    mut sim_state: ResMut<MissionSimState>,
    mut pos_data: ResMut<UnitPositionData>,
    mut hp_data: ResMut<UnitHealthData>,
    mut event_buf: ResMut<SimEventBuffer>,
    mut mission_log: Option<ResMut<MissionEventLog>>,
) {
    event_buf.events.clear();

    if sim_state.outcome.is_some() {
        return;
    }

    let delta_ms = time.delta().as_millis() as u32;
    sim_state.tick_remainder_ms = sim_state.tick_remainder_ms.saturating_add(delta_ms);

    while sim_state.tick_remainder_ms >= FIXED_TICK_MS {
        sim_state.tick_remainder_ms -= FIXED_TICK_MS;

        let hero_intents: Vec<UnitIntent> = {
            let stored = sim_state.hero_intents.clone();
            sim_state
                .sim
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && u.hp > 0)
                .map(|u| {
                    stored
                        .iter()
                        .find(|i| i.unit_id == u.id)
                        .copied()
                        .unwrap_or(UnitIntent {
                            unit_id: u.id,
                            action: IntentAction::Hold,
                        })
                })
                .collect()
        };

        let sim_snapshot = sim_state.sim.clone();
        let enemy_intents: Vec<UnitIntent> =
            sim_state.enemy_ai.generate_intents(&sim_snapshot, FIXED_TICK_MS);

        let mut all_intents = hero_intents;
        all_intents.extend(enemy_intents);

        // Inject GridNav into SimState so movement respects obstacles
        sim_state.sim.grid_nav = sim_state.grid_nav.clone();
        let current_sim = std::mem::replace(
            &mut sim_state.sim,
            SimState { tick: 0, rng_state: 0, units: UnitStore::new(Vec::new()), projectiles: Vec::new(), passive_trigger_depth: 0, zones: Vec::new(), tethers: Vec::new(), grid_nav: None },
        );
        let (mut new_sim, events) = core::step(current_sim, &all_intents, FIXED_TICK_MS);
        // Extract GridNav back (zones may have modified blocked cells)
        sim_state.grid_nav = new_sim.grid_nav.take();

        // Update per-unit terrain modifiers (cover & elevation) from GridNav
        if let Some(ref nav) = sim_state.grid_nav {
            update_unit_terrain_modifiers(&mut new_sim, nav);
        }

        sim_state.sim = new_sim;

        if let Some(ref mut log) = mission_log {
            log.all_events.extend(events.iter().cloned());
        }
        event_buf.events.extend(events);

        let all_enemies_dead = sim_state
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy)
            .all(|u| u.hp <= 0);
        let all_heroes_dead = sim_state
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero)
            .all(|u| u.hp <= 0);

        if all_enemies_dead {
            sim_state.outcome = Some(MissionOutcome::Victory);
            break;
        }
        if all_heroes_dead {
            sim_state.outcome = Some(MissionOutcome::Defeat);
            break;
        }
    }

    for unit in &sim_state.sim.units {
        pos_data.positions.insert(unit.id, (unit.position.x, unit.position.y));
        hp_data.hp.insert(unit.id, (unit.hp, unit.max_hp));
    }
}

/// Translates `SimEventBuffer` events into VFX events.
pub fn apply_vfx_from_sim_events_system(
    sim_state: Res<MissionSimState>,
    event_buf: Res<SimEventBuffer>,
    mut vfx_queue: Option<ResMut<VfxEventQueue>>,
) {
    let Some(ref mut vfx) = vfx_queue.as_deref_mut() else {
        return;
    };
    for sim_event in &event_buf.events {
        match sim_event {
            SimEvent::DamageApplied { target_id, amount, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *target_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Damage {
                        world_pos,
                        amount: *amount,
                        is_crit: false,
                    });
                }
            }
            SimEvent::HealApplied { target_id, amount, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *target_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Heal {
                        world_pos,
                        amount: *amount,
                    });
                }
            }
            SimEvent::UnitDied { unit_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *unit_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Death { world_pos });
                }
            }
            SimEvent::ControlApplied { target_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *target_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Control { world_pos });
                }
            }
            SimEvent::UnitControlled { unit_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *unit_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Control { world_pos });
                }
            }
            SimEvent::ChannelStarted { unit_id, .. } => {
                vfx.pending.push(VfxEvent::ChannelStart { unit_id: *unit_id });
            }
            SimEvent::ChannelCompleted { unit_id, .. }
            | SimEvent::ChannelInterrupted { unit_id, .. } => {
                vfx.pending.push(VfxEvent::ChannelEnd { unit_id: *unit_id });
            }
            SimEvent::ChainBounce { source_id, target_id, .. } => {
                let from = sim_state.sim.units.iter().find(|u| u.id == *source_id);
                let to = sim_state.sim.units.iter().find(|u| u.id == *target_id);
                if let (Some(f), Some(t)) = (from, to) {
                    vfx.pending.push(VfxEvent::ChainFlash {
                        from_pos: Vec3::new(f.position.x, 0.8, f.position.y),
                        to_pos: Vec3::new(t.position.x, 0.8, t.position.y),
                        color: Color::rgb(0.9, 0.9, 0.3),
                    });
                }
            }
            SimEvent::DashPerformed { unit_id: _, from_x100, from_y100, to_x100, to_y100, .. } => {
                let from = Vec3::new(*from_x100 as f32 / 100.0, 0.3, *from_y100 as f32 / 100.0);
                let to = Vec3::new(*to_x100 as f32 / 100.0, 0.3, *to_y100 as f32 / 100.0);
                vfx.pending.push(VfxEvent::Trail {
                    from,
                    to,
                    color: Color::rgb(1.0, 1.0, 1.0),
                });
            }
            SimEvent::KnockbackApplied { target_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *target_id) {
                    let pos = Vec3::new(unit.position.x, 0.3, unit.position.y);
                    vfx.pending.push(VfxEvent::Trail {
                        from: pos,
                        to: pos + Vec3::new(0.0, 0.5, 0.0),
                        color: Color::rgb(1.0, 0.5, 0.0),
                    });
                }
            }
            SimEvent::PullApplied { target_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *target_id) {
                    let pos = Vec3::new(unit.position.x, 0.3, unit.position.y);
                    vfx.pending.push(VfxEvent::Trail {
                        from: pos + Vec3::new(0.0, 0.5, 0.0),
                        to: pos,
                        color: Color::rgb(0.3, 0.9, 1.0),
                    });
                }
            }
            SimEvent::SwapPerformed { unit_a, unit_b, .. } => {
                let a = sim_state.sim.units.iter().find(|u| u.id == *unit_a);
                let b = sim_state.sim.units.iter().find(|u| u.id == *unit_b);
                if let (Some(ua), Some(ub)) = (a, b) {
                    let pa = Vec3::new(ua.position.x, 0.3, ua.position.y);
                    let pb = Vec3::new(ub.position.x, 0.3, ub.position.y);
                    let color = Color::rgb(0.3, 0.9, 1.0);
                    vfx.pending.push(VfxEvent::Trail { from: pa, to: pb, color });
                    vfx.pending.push(VfxEvent::Trail { from: pb, to: pa, color });
                }
            }
            SimEvent::ShieldApplied { unit_id, amount, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *unit_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::ShieldFlash { world_pos, amount: *amount });
                }
            }
            SimEvent::AttackMissed { target_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *target_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Miss { world_pos });
                }
            }
            SimEvent::EffectResisted { unit_id, .. } => {
                if let Some(unit) = sim_state.sim.units.iter().find(|u| u.id == *unit_id) {
                    let world_pos = Vec3::new(unit.position.x, 0.0, unit.position.y);
                    vfx.pending.push(VfxEvent::Resist { world_pos });
                }
            }
            SimEvent::ZoneTick { zone_id, .. } => {
                vfx.pending.push(VfxEvent::ZonePulse { zone_id: *zone_id });
            }
            _ => {}
        }
    }
}

/// Translates `SimEventBuffer` events into audio SFX events.
pub fn apply_audio_sfx_from_sim_events_system(
    event_buf: Res<SimEventBuffer>,
    mut audio_queue: Option<ResMut<AudioEventQueue>>,
) {
    let Some(ref mut audio) = audio_queue.as_deref_mut() else {
        return;
    };
    for sim_event in &event_buf.events {
        match sim_event {
            SimEvent::DamageApplied { .. } => {
                audio.pending.push(AudioEvent::PlaySfx(SfxKind::Hit));
            }
            SimEvent::UnitDied { .. } => {
                audio.pending.push(AudioEvent::PlaySfx(SfxKind::Death));
            }
            SimEvent::AbilityCastStarted { .. } => {
                audio.pending.push(AudioEvent::PlaySfx(SfxKind::Ability));
            }
            SimEvent::HealCastStarted { .. } => {
                audio.pending.push(AudioEvent::PlaySfx(SfxKind::Ability));
            }
            _ => {}
        }
    }
}

/// Translates player mouse input into pending move orders and unit selection.
pub fn player_ground_click_system(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    windows: Query<&Window>,
    mut order_state: ResMut<PlayerOrderState>,
    unit_query: Query<(&PlayerUnitMarker, &Transform)>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = camera_query.get_single() else {
        return;
    };

    if mouse.just_pressed(MouseButton::Right) {
        order_state.selected_unit_ids.clear();
        return;
    }

    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    let Some(ray) = camera.viewport_to_world(camera_transform, cursor_pos) else {
        return;
    };

    let dir_y = ray.direction.y;
    if dir_y.abs() < f32::EPSILON {
        return;
    }
    let t = -ray.origin.y / dir_y;
    if t <= 0.0 {
        return;
    }
    let hit = ray.origin + *ray.direction * t;

    let mut clicked_unit: Option<u32> = None;
    let mut best_dist = 0.8_f32;
    for (marker, transform) in unit_query.iter() {
        let pos = transform.translation;
        let dx = hit.x - pos.x;
        let dz = hit.z - pos.z;
        let dist = (dx * dx + dz * dz).sqrt();
        if dist < best_dist {
            best_dist = dist;
            clicked_unit = Some(marker.sim_unit_id);
        }
    }

    if let Some(unit_id) = clicked_unit {
        let shift_held = keyboard.pressed(KeyCode::ShiftLeft)
            || keyboard.pressed(KeyCode::ShiftRight);
        if shift_held {
            if !order_state.selected_unit_ids.contains(&unit_id) {
                order_state.selected_unit_ids.push(unit_id);
            }
        } else {
            order_state.selected_unit_ids = vec![unit_id];
        }
    } else {
        order_state.pending_move = Some(SimVec2 { x: hit.x, y: hit.z });
    }
}

/// Updates each living unit's cover_bonus and elevation based on their
/// current position in the GridNav. Cover is computed against the nearest enemy.
fn update_unit_terrain_modifiers(sim: &mut SimState, nav: &GridNav) {
    let unit_count = sim.units.len();
    for i in 0..unit_count {
        if sim.units[i].hp <= 0 {
            sim.units[i].cover_bonus = 0.0;
            sim.units[i].elevation = 0.0;
            continue;
        }

        sim.units[i].elevation = nav.elevation_at_pos(sim.units[i].position);

        let pos = sim.units[i].position;
        let team = sim.units[i].team;
        let mut nearest_enemy_pos: Option<SimVec2> = None;
        let mut nearest_dist = f32::INFINITY;
        for j in 0..unit_count {
            if sim.units[j].hp <= 0 || sim.units[j].team == team {
                continue;
            }
            let d = crate::ai::core::distance(pos, sim.units[j].position);
            if d < nearest_dist {
                nearest_dist = d;
                nearest_enemy_pos = Some(sim.units[j].position);
            }
        }

        sim.units[i].cover_bonus = if let Some(enemy_pos) = nearest_enemy_pos {
            cover_factor(nav, pos, enemy_pos)
        } else {
            0.0
        };
    }
}

/// Applies pending player move orders to the selected units in the sim.
pub fn apply_player_orders_system(
    mut order_state: ResMut<PlayerOrderState>,
    mut sim_state: ResMut<MissionSimState>,
) {
    let Some(target) = order_state.pending_move.take() else {
        return;
    };

    for &unit_id in &order_state.selected_unit_ids {
        sim_state.hero_intents.retain(|i| i.unit_id != unit_id);
        sim_state.hero_intents.push(UnitIntent {
            unit_id,
            action: IntentAction::MoveTo { position: target },
        });
    }
}
