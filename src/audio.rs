// ---------------------------------------------------------------------------
// Audio system foundation
// ---------------------------------------------------------------------------
//
// Architecture:
//   Bevy 0.13.2 built-in audio (bevy::audio) is used exclusively — no extra
//   crates.  No actual audio files exist yet, so every load is expected to
//   produce a Handle that resolves to nothing; `process_audio_events_system`
//   guards every playback site with `if let Some(handle)` and silently skips
//   missing files.  Dropping real `.ogg` files into `assets/audio/` is
//   sufficient to enable sound at runtime with no code changes.
//
// Asset paths (relative to the `assets/` directory):
//   audio/hub_music.ogg
//   audio/combat_music_base.ogg
//   audio/sfx_hit.ogg
//   audio/sfx_death.ogg
//   audio/sfx_ability.ogg
//   audio/sfx_ui_click.ogg

use bevy::audio::{AudioBundle, PlaybackSettings, Volume};
use bevy::prelude::*;

use crate::mission::sim_bridge::MissionSimState;
use crate::ai::core::Team;

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Master volume settings.  Changes take effect for newly-spawned audio
/// entities only; already-playing sinks are not retroactively adjusted.
#[derive(Resource)]
pub struct AudioSettings {
    pub master_volume: f32,
    pub music_volume: f32,
    pub sfx_volume: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master_volume: 0.8,
            music_volume: 0.6,
            sfx_volume: 1.0,
        }
    }
}

/// Cached asset handles for all audio files.
/// Fields are `Option` because the files may not exist on disk yet.
#[derive(Resource, Default)]
pub struct AudioHandles {
    pub hub_music: Option<Handle<AudioSource>>,
    pub combat_music_base: Option<Handle<AudioSource>>,
    pub sfx_hit: Option<Handle<AudioSource>>,
    pub sfx_death: Option<Handle<AudioSource>>,
    pub sfx_ability: Option<Handle<AudioSource>>,
    pub sfx_ui_click: Option<Handle<AudioSource>>,
}

/// Per-frame event queue for audio requests.
/// Callers push events here; `process_audio_events_system` drains them.
#[derive(Resource, Default)]
pub struct AudioEventQueue {
    pub pending: Vec<AudioEvent>,
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum AudioEvent {
    PlaySfx(SfxKind),
    StartMusic(MusicKind),
    StopMusic,
}

#[derive(Debug, Clone, Copy)]
pub enum SfxKind {
    Hit,
    Death,
    Ability,
    UiClick,
}

#[derive(Debug, Clone, Copy)]
pub enum MusicKind {
    Hub,
    Combat,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker placed on entities spawned for looping music so they can be found
/// and despawned by `StopMusic`.
#[derive(Component)]
pub struct MusicMarker;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Startup system — loads audio asset handles and creates `assets/audio/`.
///
/// Bevy's `AssetServer::load` is non-blocking: it returns a `Handle`
/// immediately even if the file does not exist.  We store all handles
/// regardless, then gate playback on the handle actually being present via the
/// `Option<Handle<AudioSource>>` fields in `AudioHandles`.
pub fn load_audio_assets_system(
    asset_server: Res<AssetServer>,
    mut handles: ResMut<AudioHandles>,
) {
    // Ensure the directory exists so artists can drop files in without
    // needing to create it manually.  Failure is silently ignored.
    let _ = std::fs::create_dir_all("dataset/audio");

    handles.hub_music = Some(asset_server.load("audio/hub_music.ogg"));
    handles.combat_music_base = Some(asset_server.load("audio/combat_music_base.ogg"));
    handles.sfx_hit = Some(asset_server.load("audio/sfx_hit.ogg"));
    handles.sfx_death = Some(asset_server.load("audio/sfx_death.ogg"));
    handles.sfx_ability = Some(asset_server.load("audio/sfx_ability.ogg"));
    handles.sfx_ui_click = Some(asset_server.load("audio/sfx_ui_click.ogg"));
}

/// Per-frame system — drains `AudioEventQueue` and spawns audio entities.
///
/// Missing handles (file not on disk) are silently skipped — no panics.
pub fn process_audio_events_system(
    mut commands: Commands,
    mut queue: ResMut<AudioEventQueue>,
    handles: Res<AudioHandles>,
    settings: Res<AudioSettings>,
    music_query: Query<Entity, With<MusicMarker>>,
) {
    let events: Vec<AudioEvent> = queue.pending.drain(..).collect();

    let effective_sfx_volume = (settings.master_volume * settings.sfx_volume).clamp(0.0, 1.0);
    let effective_music_volume = (settings.master_volume * settings.music_volume).clamp(0.0, 1.0);

    for event in events {
        match event {
            AudioEvent::PlaySfx(kind) => {
                let handle_opt: Option<Handle<AudioSource>> = match kind {
                    SfxKind::Hit => handles.sfx_hit.clone(),
                    SfxKind::Death => handles.sfx_death.clone(),
                    SfxKind::Ability => handles.sfx_ability.clone(),
                    SfxKind::UiClick => handles.sfx_ui_click.clone(),
                };
                if let Some(handle) = handle_opt {
                    commands.spawn(AudioBundle {
                        source: handle,
                        settings: PlaybackSettings::ONCE
                            .with_volume(Volume::new(effective_sfx_volume)),
                    });
                }
                // No handle → silently skip (file not on disk yet).
            }

            AudioEvent::StartMusic(kind) => {
                let handle_opt: Option<Handle<AudioSource>> = match kind {
                    MusicKind::Hub => handles.hub_music.clone(),
                    MusicKind::Combat => handles.combat_music_base.clone(),
                };
                if let Some(handle) = handle_opt {
                    // Stop any currently playing music first to avoid overlap.
                    for entity in music_query.iter() {
                        commands.entity(entity).despawn_recursive();
                    }
                    commands.spawn((
                        AudioBundle {
                            source: handle,
                            settings: PlaybackSettings::LOOP
                                .with_volume(Volume::new(effective_music_volume)),
                        },
                        MusicMarker,
                    ));
                }
            }

            AudioEvent::StopMusic => {
                for entity in music_query.iter() {
                    commands.entity(entity).despawn_recursive();
                }
            }
        }
    }
}

/// Per-frame system — watches the live enemy count and manages combat music.
///
/// Only runs while a mission is active (i.e. `MissionSimState` exists as a
/// resource).  Pushes `StartMusic(Combat)` when enemies exceed the threshold
/// and `StopMusic` when they are all defeated.
pub fn combat_music_intensity_system(
    sim_state: Option<Res<MissionSimState>>,
    mut queue: ResMut<AudioEventQueue>,
    music_query: Query<Entity, With<MusicMarker>>,
) {
    let Some(state) = sim_state else {
        return;
    };

    let alive_enemies = state
        .sim
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .count();

    let music_playing = !music_query.is_empty();

    if alive_enemies > 3 && !music_playing {
        queue.pending.push(AudioEvent::StartMusic(MusicKind::Combat));
    } else if alive_enemies == 0 && music_playing {
        queue.pending.push(AudioEvent::StopMusic);
    }
}
