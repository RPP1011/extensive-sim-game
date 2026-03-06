use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;
use std::fs;
use std::path::Path;

use crate::camera::CameraSettings;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const MANUAL_SCREENSHOT_DIR: &str = "generated/screenshots";

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct SettingsMenuState {
    pub is_open: bool,
}

#[derive(Component)]
pub struct SettingsMenuRoot;

#[derive(Component)]
pub struct OrbitSensitivitySliderTrack;

#[derive(Component)]
pub struct OrbitSensitivitySliderFill;

#[derive(Component)]
pub struct OrbitSensitivityLabel;

#[derive(Component)]
pub struct ZoomSensitivitySliderTrack;

#[derive(Component)]
pub struct ZoomSensitivitySliderFill;

#[derive(Component)]
pub struct ZoomSensitivityLabel;

#[derive(Component)]
pub struct InvertOrbitYButton;

#[derive(Component)]
pub struct InvertOrbitYLabel;

#[derive(Component)]
pub struct ResetSettingsButton;

#[derive(Component)]
pub struct TakeScreenshotButton;

#[derive(Resource, Default)]
pub struct ManualScreenshotState {
    pub pending_frames: u8,
    pub captures_written: u32,
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

pub fn setup_settings_menu(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/DejaVuSans.ttf");
    commands
        .spawn((
            SettingsMenuRoot,
            NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    right: Val::Px(20.0),
                    top: Val::Px(20.0),
                    width: Val::Px(320.0),
                    height: Val::Auto,
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Stretch,
                    padding: UiRect::all(Val::Px(12.0)),
                    row_gap: Val::Px(8.0),
                    display: Display::None,
                    ..default()
                },
                background_color: Color::rgba(0.05, 0.06, 0.08, 0.90).into(),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn(TextBundle::from_sections([TextSection::new(
                "Settings (Esc)",
                TextStyle {
                    font: font.clone(),
                    font_size: 18.0,
                    color: Color::rgb(0.93, 0.93, 0.97),
                },
            )]));

            parent.spawn((
                OrbitSensitivityLabel,
                TextBundle::from_sections([TextSection::new(
                    "Orbit Sensitivity: 1.00x",
                    TextStyle {
                        font: font.clone(),
                        font_size: 15.0,
                        color: Color::rgb(0.88, 0.88, 0.92),
                    },
                )]),
            ));
            parent
                .spawn((
                    OrbitSensitivitySliderTrack,
                    RelativeCursorPosition::default(),
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(16.0),
                            position_type: PositionType::Relative,
                            ..default()
                        },
                        background_color: Color::rgb(0.20, 0.22, 0.28).into(),
                        ..default()
                    },
                ))
                .with_children(|slider| {
                    slider.spawn((
                        OrbitSensitivitySliderFill,
                        NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(0.0),
                                top: Val::Px(0.0),
                                bottom: Val::Px(0.0),
                                width: Val::Percent(35.0),
                                ..default()
                            },
                            background_color: Color::rgb(0.30, 0.66, 0.88).into(),
                            ..default()
                        },
                    ));
                });

            parent.spawn((
                ZoomSensitivityLabel,
                TextBundle::from_sections([TextSection::new(
                    "Zoom Sensitivity: 1.00x",
                    TextStyle {
                        font: font.clone(),
                        font_size: 15.0,
                        color: Color::rgb(0.88, 0.88, 0.92),
                    },
                )]),
            ));
            parent
                .spawn((
                    ZoomSensitivitySliderTrack,
                    RelativeCursorPosition::default(),
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(16.0),
                            position_type: PositionType::Relative,
                            ..default()
                        },
                        background_color: Color::rgb(0.20, 0.22, 0.28).into(),
                        ..default()
                    },
                ))
                .with_children(|slider| {
                    slider.spawn((
                        ZoomSensitivitySliderFill,
                        NodeBundle {
                            style: Style {
                                position_type: PositionType::Absolute,
                                left: Val::Px(0.0),
                                top: Val::Px(0.0),
                                bottom: Val::Px(0.0),
                                width: Val::Percent(35.0),
                                ..default()
                            },
                            background_color: Color::rgb(0.26, 0.58, 0.80).into(),
                            ..default()
                        },
                    ));
                });

            parent
                .spawn((
                    InvertOrbitYButton,
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(30.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.18, 0.20, 0.25).into(),
                        ..default()
                    },
                ))
                .with_children(|button| {
                    button.spawn((
                        InvertOrbitYLabel,
                        TextBundle::from_sections([TextSection::new(
                            "Invert Orbit Y: Off",
                            TextStyle {
                                font: font.clone(),
                                font_size: 15.0,
                                color: Color::rgb(0.92, 0.92, 0.96),
                            },
                        )]),
                    ));
                });

            parent
                .spawn((
                    ResetSettingsButton,
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(30.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.28, 0.20, 0.18).into(),
                        ..default()
                    },
                ))
                .with_children(|button| {
                    button.spawn(TextBundle::from_sections([TextSection::new(
                        "Reset Camera Defaults",
                        TextStyle {
                            font: font.clone(),
                            font_size: 15.0,
                            color: Color::rgb(0.95, 0.92, 0.90),
                        },
                    )]));
                });

            parent
                .spawn((
                    TakeScreenshotButton,
                    ButtonBundle {
                        style: Style {
                            width: Val::Percent(100.0),
                            height: Val::Px(30.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.15, 0.28, 0.24).into(),
                        ..default()
                    },
                ))
                .with_children(|button| {
                    button.spawn(TextBundle::from_sections([TextSection::new(
                        "Take Screenshot",
                        TextStyle {
                            font: font.clone(),
                            font_size: 15.0,
                            color: Color::rgb(0.90, 0.95, 0.93),
                        },
                    )]));
                });
        });
}

pub fn settings_menu_toggle_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut state: ResMut<SettingsMenuState>,
    mut menu_query: Query<&mut Style, With<SettingsMenuRoot>>,
) {
    let Some(keyboard) = keyboard else {
        return;
    };
    if !keyboard.just_pressed(KeyCode::Escape) {
        return;
    }
    state.is_open = !state.is_open;
    let display = if state.is_open {
        Display::Flex
    } else {
        Display::None
    };
    for mut style in &mut menu_query {
        style.display = display;
    }
}

pub fn settings_menu_slider_input_system(
    menu_state: Res<SettingsMenuState>,
    mut camera_settings: ResMut<CameraSettings>,
    orbit_track_query: Query<
        (&Interaction, &RelativeCursorPosition),
        With<OrbitSensitivitySliderTrack>,
    >,
    zoom_track_query: Query<
        (&Interaction, &RelativeCursorPosition),
        With<ZoomSensitivitySliderTrack>,
    >,
) {
    if !menu_state.is_open {
        return;
    }

    for (interaction, cursor) in &orbit_track_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(pos) = cursor.normalized else {
            continue;
        };
        let t = pos.x.clamp(0.0, 1.0);
        camera_settings.orbit_sensitivity = 0.2 + (2.5 - 0.2) * t;
    }

    for (interaction, cursor) in &zoom_track_query {
        if *interaction != Interaction::Pressed {
            continue;
        }
        let Some(pos) = cursor.normalized else {
            continue;
        };
        let t = pos.x.clamp(0.0, 1.0);
        camera_settings.zoom_sensitivity = 0.2 + (2.5 - 0.2) * t;
    }
}

pub fn settings_menu_toggle_input_system(
    mut menu_state: ResMut<SettingsMenuState>,
    mut camera_settings: ResMut<CameraSettings>,
    mut manual_screenshot: ResMut<ManualScreenshotState>,
    mut menu_query: Query<&mut Style, With<SettingsMenuRoot>>,
    mut invert_button_query: Query<&Interaction, (With<InvertOrbitYButton>, Changed<Interaction>)>,
    mut reset_button_query: Query<&Interaction, (With<ResetSettingsButton>, Changed<Interaction>)>,
    mut screenshot_button_query: Query<
        &Interaction,
        (With<TakeScreenshotButton>, Changed<Interaction>),
    >,
) {
    if !menu_state.is_open {
        return;
    }

    for interaction in &mut invert_button_query {
        if *interaction == Interaction::Pressed {
            camera_settings.invert_orbit_y = !camera_settings.invert_orbit_y;
        }
    }

    for interaction in &mut reset_button_query {
        if *interaction == Interaction::Pressed {
            *camera_settings = CameraSettings::default();
        }
    }

    for interaction in &mut screenshot_button_query {
        if *interaction == Interaction::Pressed {
            // Hide menu first, then capture on a later frame so UI state has updated.
            menu_state.is_open = false;
            for mut style in &mut menu_query {
                style.display = Display::None;
            }
            manual_screenshot.pending_frames = 2;
        }
    }
}

pub fn screenshot_hotkey_input_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut menu_state: ResMut<SettingsMenuState>,
    mut manual_screenshot: ResMut<ManualScreenshotState>,
    mut menu_query: Query<&mut Style, With<SettingsMenuRoot>>,
) {
    let Some(keyboard) = keyboard else {
        return;
    };
    if !keyboard.just_pressed(KeyCode::F12) && !keyboard.just_pressed(KeyCode::Semicolon) {
        return;
    }
    menu_state.is_open = false;
    for mut style in &mut menu_query {
        style.display = Display::None;
    }
    manual_screenshot.pending_frames = 2;
}

pub fn manual_screenshot_capture_system(world: &mut World) {
    {
        let mut state = world.resource_mut::<ManualScreenshotState>();
        if state.pending_frames == 0 {
            return;
        }
        state.pending_frames -= 1;
        if state.pending_frames > 0 {
            return;
        }
    }

    let window = match world
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .get_single(world)
    {
        Ok(entity) => entity,
        Err(_) => return,
    };

    if let Err(err) = fs::create_dir_all(MANUAL_SCREENSHOT_DIR) {
        eprintln!(
            "Failed to create screenshot directory '{}': {}",
            MANUAL_SCREENSHOT_DIR, err
        );
        return;
    }

    let (capture_index, timestamp) = {
        let state = world.resource::<ManualScreenshotState>();
        (state.captures_written, crate::campaign_ops::unix_now_seconds())
    };
    let image_path = Path::new(MANUAL_SCREENSHOT_DIR)
        .join(format!("manual_{}_{}.png", timestamp, capture_index))
        .to_string_lossy()
        .to_string();
    let wrote = {
        let mut screenshot_manager =
            world.resource_mut::<bevy::render::view::screenshot::ScreenshotManager>();
        screenshot_manager
            .save_screenshot_to_disk(window, &image_path)
            .map(|_| ())
    };
    let mut state = world.resource_mut::<ManualScreenshotState>();
    if let Err(err) = wrote {
        eprintln!("Failed to save screenshot '{}': {}", image_path, err);
    } else {
        state.captures_written += 1;
        println!("Screenshot saved: {}", image_path);
    }
}

pub fn update_settings_menu_visual_system(
    camera_settings: Res<CameraSettings>,
    mut style_sets: ParamSet<(
        Query<&mut Style, With<OrbitSensitivitySliderFill>>,
        Query<&mut Style, With<ZoomSensitivitySliderFill>>,
    )>,
    mut text_sets: ParamSet<(
        Query<&mut Text, With<OrbitSensitivityLabel>>,
        Query<&mut Text, With<ZoomSensitivityLabel>>,
        Query<&mut Text, With<InvertOrbitYLabel>>,
    )>,
) {
    if !camera_settings.is_changed() && !camera_settings.is_added() {
        return;
    }

    let orbit_t = ((camera_settings.orbit_sensitivity - 0.2) / (2.5 - 0.2)).clamp(0.0, 1.0);
    let zoom_t = ((camera_settings.zoom_sensitivity - 0.2) / (2.5 - 0.2)).clamp(0.0, 1.0);

    for mut style in &mut style_sets.p0() {
        style.width = Val::Percent(orbit_t * 100.0);
    }
    for mut style in &mut style_sets.p1() {
        style.width = Val::Percent(zoom_t * 100.0);
    }
    for mut text in &mut text_sets.p0() {
        text.sections[0].value = format!(
            "Orbit Sensitivity: {:.2}x",
            camera_settings.orbit_sensitivity
        );
    }
    for mut text in &mut text_sets.p1() {
        text.sections[0].value =
            format!("Zoom Sensitivity: {:.2}x", camera_settings.zoom_sensitivity);
    }
    for mut text in &mut text_sets.p2() {
        text.sections[0].value = format!(
            "Invert Orbit Y: {}",
            if camera_settings.invert_orbit_y {
                "On"
            } else {
                "Off"
            }
        );
    }
}
