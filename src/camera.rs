use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;

use crate::ui::settings::SettingsMenuState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const CAMERA_SETTINGS_PATH: &str = "generated/settings/camera_settings.json";

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Copy)]
pub struct SceneViewBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_z: f32,
    pub max_z: f32,
}

impl Default for SceneViewBounds {
    fn default() -> Self {
        Self {
            min_x: -6.0,
            max_x: 6.0,
            min_z: -4.0,
            max_z: 4.0,
        }
    }
}

#[derive(Component)]
pub struct OrbitCameraController {
    pub focus: Vec3,
    pub radius: f32,
    pub min_radius: f32,
    pub max_radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraFocusTrigger {
    TakeCommand,
    FocusSelectedParty,
}

impl CameraFocusTrigger {
    pub fn label(self) -> &'static str {
        match self {
            Self::TakeCommand => "Take Command",
            Self::FocusSelectedParty => "Focus Selected Party",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CameraFocusTransition {
    pub target_party_id: u32,
    pub target_region_id: usize,
    pub trigger: CameraFocusTrigger,
    pub start_focus: Vec3,
    pub target_focus: Vec3,
    pub elapsed_seconds: f32,
    pub duration_seconds: f32,
}

impl CameraFocusTransition {
    pub fn progress(&self) -> f32 {
        if self.duration_seconds <= f32::EPSILON {
            1.0
        } else {
            (self.elapsed_seconds / self.duration_seconds).clamp(0.0, 1.0)
        }
    }

    pub fn interpolated_focus(&self) -> Vec3 {
        self.start_focus.lerp(self.target_focus, self.progress())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraFocusTransitionQueueResult {
    Started,
    Retargeted,
}

#[derive(Debug, Clone, Copy)]
pub struct CameraFocusTransitionStep {
    pub focus: Vec3,
    // Read from main.rs (e.g. in tests and in queue_party_camera_focus_transition).
    #[allow(dead_code)]
    pub completed: bool,
}

#[derive(Resource, Debug, Clone)]
pub struct CameraFocusTransitionState {
    pub active: Option<CameraFocusTransition>,
    pub duration_seconds: f32,
}

impl Default for CameraFocusTransitionState {
    fn default() -> Self {
        Self {
            active: None,
            duration_seconds: 0.55,
        }
    }
}

impl CameraFocusTransitionState {
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn begin(
        &mut self,
        start_focus: Vec3,
        target_focus: Vec3,
        target_party_id: u32,
        target_region_id: usize,
        trigger: CameraFocusTrigger,
    ) -> CameraFocusTransitionQueueResult {
        let was_active = self.active.is_some();
        self.active = Some(CameraFocusTransition {
            target_party_id,
            target_region_id,
            trigger,
            start_focus,
            target_focus,
            elapsed_seconds: 0.0,
            duration_seconds: self.duration_seconds.max(0.01),
        });
        if was_active {
            CameraFocusTransitionQueueResult::Retargeted
        } else {
            CameraFocusTransitionQueueResult::Started
        }
    }

    pub fn step(&mut self, dt_seconds: f32) -> Option<CameraFocusTransitionStep> {
        let transition = self.active.as_mut()?;
        transition.elapsed_seconds += dt_seconds.max(0.0);
        let focus = transition.interpolated_focus();
        let completed = transition.progress() >= 1.0;
        if completed {
            self.active = None;
        }
        Some(CameraFocusTransitionStep { focus, completed })
    }
}

#[derive(Resource, Serialize, Deserialize, Clone, Copy)]
pub struct CameraSettings {
    pub orbit_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub invert_orbit_y: bool,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            orbit_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
            invert_orbit_y: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

pub fn setup_camera(
    mut commands: Commands,
    bounds: Res<SceneViewBounds>,
    runtime_mode: Res<crate::RuntimeModeState>,
) {
    let center_x = (bounds.min_x + bounds.max_x) * 0.5;
    let center_z = (bounds.min_z + bounds.max_z) * 0.5;
    let span_x = (bounds.max_x - bounds.min_x).max(8.0);
    let span_z = (bounds.max_z - bounds.min_z).max(8.0);
    let span = span_x.max(span_z);
    let focus = Vec3::new(center_x, 0.0, center_z);
    let start = Vec3::new(center_x, span * 0.95, center_z + span * 1.45);
    let offset = start - focus;
    let radius = offset.length().max(0.001);
    let yaw = offset.x.atan2(offset.z);
    let pitch = (offset.y / radius).asin();

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(start).looking_at(focus, Vec3::Y),
            ..default()
        },
        OrbitCameraController {
            focus,
            radius,
            min_radius: 3.0,
            max_radius: span * 6.0,
            yaw,
            pitch,
        },
    ));

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 15000.0,
            // Hub UI uses a mostly static background; disabling shadows avoids visible shimmer.
            shadows_enabled: !runtime_mode.hub_mode,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.1, -0.8, 0.0)),
        ..default()
    });
}

pub fn orbit_camera_controller_system(
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut mouse_wheel_events: EventReader<MouseWheel>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    time: Res<Time>,
    bounds: Res<SceneViewBounds>,
    camera_settings: Res<CameraSettings>,
    settings_menu: Res<SettingsMenuState>,
    mut camera_focus_transition: ResMut<CameraFocusTransitionState>,
    mut query: Query<(&mut OrbitCameraController, &mut Transform)>,
) {
    const ORBIT_YAW_SENSITIVITY: f32 = 0.0019;
    const ORBIT_PITCH_SENSITIVITY: f32 = 0.0015;
    const ZOOM_SENSITIVITY: f32 = 0.08;
    const MIN_PITCH_RADIANS: f32 = 0.22;
    const MAX_PITCH_RADIANS: f32 = 1.18;
    const MIN_CAMERA_HEIGHT: f32 = 1.1;

    let mut mouse_delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        mouse_delta += event.delta;
    }

    let mut scroll_delta = 0.0_f32;
    for event in mouse_wheel_events.read() {
        scroll_delta += event.y;
    }

    let Some(keyboard) = keyboard else {
        return;
    };
    if settings_menu.is_open {
        return;
    }
    let transition_active = camera_focus_transition.is_active();
    let transition_step = camera_focus_transition.step(time.delta_seconds());

    for (mut controller, mut transform) in &mut query {
        let mut changed = false;
        let orbit_sens = camera_settings.orbit_sensitivity.clamp(0.2, 2.5);
        let zoom_sens = camera_settings.zoom_sensitivity.clamp(0.2, 2.5);
        let y_invert = if camera_settings.invert_orbit_y {
            1.0
        } else {
            -1.0
        };

        if let Some(step) = transition_step {
            controller.focus = step.focus;
            changed = true;
        }

        if !transition_active {
            if scroll_delta.abs() > f32::EPSILON {
                controller.radius -=
                    scroll_delta * ZOOM_SENSITIVITY * zoom_sens * controller.radius;
                changed = true;
            }

            if mouse_buttons.pressed(MouseButton::Right) && mouse_delta.length_squared() > 0.0 {
                controller.yaw -= mouse_delta.x * ORBIT_YAW_SENSITIVITY * orbit_sens;
                controller.pitch += y_invert * mouse_delta.y * ORBIT_PITCH_SENSITIVITY * orbit_sens;
                changed = true;
            }

            if mouse_buttons.pressed(MouseButton::Middle) && mouse_delta.length_squared() > 0.0 {
                let right = transform.rotation * Vec3::X;
                let forward = transform.rotation * Vec3::NEG_Z;
                let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
                let pan_scale = 0.006 * controller.radius;
                controller.focus += (-mouse_delta.x * pan_scale) * right;
                controller.focus += (mouse_delta.y * pan_scale) * forward_flat;
                changed = true;
            }

            let mut keyboard_pan = Vec3::ZERO;
            let right = transform.rotation * Vec3::X;
            let forward = transform.rotation * Vec3::NEG_Z;
            let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
            if keyboard.pressed(KeyCode::KeyA) {
                keyboard_pan -= right;
            }
            if keyboard.pressed(KeyCode::KeyD) {
                keyboard_pan += right;
            }
            if keyboard.pressed(KeyCode::KeyW) {
                keyboard_pan += forward_flat;
            }
            if keyboard.pressed(KeyCode::KeyS) {
                keyboard_pan -= forward_flat;
            }
            if keyboard_pan.length_squared() > 0.0 {
                let radius = controller.radius;
                controller.focus += keyboard_pan.normalize() * (0.045 * radius);
                changed = true;
            }

            if keyboard.just_pressed(KeyCode::KeyF) {
                controller.focus = Vec3::new(
                    (bounds.min_x + bounds.max_x) * 0.5,
                    0.0,
                    (bounds.min_z + bounds.max_z) * 0.5,
                );
                controller.radius =
                    ((bounds.max_x - bounds.min_x).max(bounds.max_z - bounds.min_z) * 1.4).max(6.0);
                controller.pitch = 0.62;
                changed = true;
            }
        }

        controller.pitch = controller.pitch.clamp(MIN_PITCH_RADIANS, MAX_PITCH_RADIANS);
        controller.radius = controller.radius.clamp(
            controller.min_radius,
            controller.max_radius.max(controller.min_radius + 1.0),
        );
        if controller.yaw > std::f32::consts::PI || controller.yaw < -std::f32::consts::PI {
            controller.yaw = controller.yaw.rem_euclid(std::f32::consts::TAU);
            if controller.yaw > std::f32::consts::PI {
                controller.yaw -= std::f32::consts::TAU;
            }
        }
        controller.focus.x = controller
            .focus
            .x
            .clamp(bounds.min_x - 30.0, bounds.max_x + 30.0);
        controller.focus.z = controller
            .focus
            .z
            .clamp(bounds.min_z - 30.0, bounds.max_z + 30.0);

        if !changed {
            continue;
        }

        let cos_pitch = controller.pitch.cos();
        let offset = Vec3::new(
            controller.radius * cos_pitch * controller.yaw.sin(),
            controller.radius * controller.pitch.sin(),
            controller.radius * cos_pitch * controller.yaw.cos(),
        );
        transform.translation = controller.focus + offset;
        if transform.translation.y < MIN_CAMERA_HEIGHT {
            transform.translation.y = MIN_CAMERA_HEIGHT;
        }
        transform.look_at(controller.focus, Vec3::Y);
    }
}

pub fn persist_camera_settings_system(camera_settings: Res<CameraSettings>) {
    if !camera_settings.is_changed() && !camera_settings.is_added() {
        return;
    }
    let serialized = match serde_json::to_string_pretty(&*camera_settings) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("Failed to serialize camera settings: {}", err);
            return;
        }
    };
    if let Some(parent) = std::path::Path::new(CAMERA_SETTINGS_PATH).parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("Failed to create settings directory: {}", err);
            return;
        }
    }
    if let Err(err) = fs::write(CAMERA_SETTINGS_PATH, serialized) {
        eprintln!("Failed to persist camera settings: {}", err);
    }
}

pub fn load_camera_settings() -> CameraSettings {
    let text = match fs::read_to_string(CAMERA_SETTINGS_PATH) {
        Ok(value) => value,
        Err(_) => return CameraSettings::default(),
    };
    let loaded: CameraSettings = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(err) => {
            eprintln!(
                "Invalid camera settings at '{}': {}. Using defaults.",
                CAMERA_SETTINGS_PATH, err
            );
            return CameraSettings::default();
        }
    };
    CameraSettings {
        orbit_sensitivity: loaded.orbit_sensitivity.clamp(0.2, 2.5),
        zoom_sensitivity: loaded.zoom_sensitivity.clamp(0.2, 2.5),
        invert_orbit_y: loaded.invert_orbit_y,
    }
}

/// Map an overworld region to its 3-D world-space camera focus point.
pub fn camera_focus_for_overworld_region(
    overworld: &crate::game_core::OverworldMap,
    bounds: &SceneViewBounds,
    region_id: usize,
) -> Option<Vec3> {
    let points = crate::game_core::overworld_region_plot_positions(overworld);
    let (wx, wy) = *points.get(region_id)?;

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in &points {
        min_x = min_x.min(*x);
        max_x = max_x.max(*x);
        min_y = min_y.min(*y);
        max_y = max_y.max(*y);
    }
    let span_x = (max_x - min_x).max(0.001);
    let span_y = (max_y - min_y).max(0.001);
    let normalized_x = (wx - min_x) / span_x;
    let normalized_y = (wy - min_y) / span_y;

    let world_span_x = (bounds.max_x - bounds.min_x).max(0.001);
    let world_span_z = (bounds.max_z - bounds.min_z).max(0.001);
    let margin_ratio = 0.12;
    let mapped_x = bounds.min_x
        + world_span_x * margin_ratio
        + normalized_x * world_span_x * (1.0 - margin_ratio * 2.0);
    let mapped_z = bounds.min_z
        + world_span_z * margin_ratio
        + normalized_y * world_span_z * (1.0 - margin_ratio * 2.0);
    Some(Vec3::new(mapped_x, 0.0, mapped_z))
}
