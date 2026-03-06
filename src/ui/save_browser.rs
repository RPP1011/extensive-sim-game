use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const CAMPAIGN_SAVE_PATH: &str = "generated/saves/campaign_save.json";
pub const CAMPAIGN_SAVE_SLOT_2_PATH: &str = "generated/saves/campaign_slot_2.json";
pub const CAMPAIGN_SAVE_SLOT_3_PATH: &str = "generated/saves/campaign_slot_3.json";
pub const CAMPAIGN_AUTOSAVE_PATH: &str = "generated/saves/campaign_autosave.json";
pub const CAMPAIGN_SAVE_INDEX_PATH: &str = "generated/saves/campaign_index.json";
pub const SAVE_VERSION_V1: u32 = 1;
pub const SAVE_VERSION_V2: u32 = 2;
pub const CURRENT_SAVE_VERSION: u32 = 3;

// ---------------------------------------------------------------------------
// Helper used by CampaignSaveData serialization (stays pub so main.rs can use it)
// ---------------------------------------------------------------------------

pub fn default_save_version() -> u32 {
    SAVE_VERSION_V1
}

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

#[derive(Resource)]
pub struct CampaignSaveNotice {
    pub message: String,
}

impl Default for CampaignSaveNotice {
    fn default() -> Self {
        Self {
            message: "Save: none (F5/F9 slot1, Shift slot2, Ctrl slot3)".to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SaveSlotMetadata {
    pub slot: String,
    pub path: String,
    pub save_version: u32,
    pub compatible: bool,
    pub global_turn: u32,
    pub map_seed: u64,
    pub saved_unix_seconds: u64,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct CampaignSaveIndex {
    pub slots: Vec<SaveSlotMetadata>,
    pub autosave: Option<SaveSlotMetadata>,
}

#[derive(Resource, Clone, Default)]
pub struct CampaignSaveIndexState {
    pub index: CampaignSaveIndex,
}

#[derive(Resource, Clone, Default)]
pub struct CampaignSavePanelState {
    pub open: bool,
    pub selected: usize,
    pub pending_load_path: Option<String>,
    pub pending_label: Option<String>,
    pub preview: String,
}

#[derive(Resource, Clone)]
pub struct CampaignAutosaveState {
    pub enabled: bool,
    pub interval_turns: u32,
    pub last_autosave_turn: u32,
}

impl Default for CampaignAutosaveState {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_turns: 10,
            last_autosave_turn: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions
// ---------------------------------------------------------------------------

pub fn save_slot_label(slot: u8) -> String {
    format!("slot{}", slot)
}

pub fn campaign_slot_path(slot: u8) -> &'static str {
    match slot {
        2 => CAMPAIGN_SAVE_SLOT_2_PATH,
        3 => CAMPAIGN_SAVE_SLOT_3_PATH,
        _ => CAMPAIGN_SAVE_PATH,
    }
}

pub fn load_campaign_save_index() -> CampaignSaveIndex {
    let text = match fs::read_to_string(CAMPAIGN_SAVE_INDEX_PATH) {
        Ok(value) => value,
        Err(_) => return CampaignSaveIndex::default(),
    };
    serde_json::from_str::<CampaignSaveIndex>(&text).unwrap_or_default()
}

pub fn persist_campaign_save_index(index: &CampaignSaveIndex) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(index).map_err(|e| e.to_string())?;
    if let Some(parent) = std::path::Path::new(CAMPAIGN_SAVE_INDEX_PATH).parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(CAMPAIGN_SAVE_INDEX_PATH, serialized).map_err(|e| e.to_string())
}

pub fn upsert_slot_metadata(
    index: &mut CampaignSaveIndex,
    slot: u8,
    metadata: SaveSlotMetadata,
) {
    let key = save_slot_label(slot);
    if let Some(existing) = index.slots.iter_mut().find(|m| m.slot == key) {
        *existing = metadata;
    } else {
        index.slots.push(metadata);
    }
    index.slots.sort_by(|a, b| a.slot.cmp(&b.slot));
}

pub fn load_campaign_save_index_state() -> CampaignSaveIndexState {
    CampaignSaveIndexState {
        index: load_campaign_save_index(),
    }
}

#[derive(Clone)]
pub struct ContinueCandidate {
    pub label: String,
    pub path: String,
    pub saved_unix_seconds: u64,
}

pub fn continue_campaign_candidates(index: &CampaignSaveIndex) -> Vec<ContinueCandidate> {
    let mut candidates = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();

    let mut from_index = Vec::new();
    from_index.extend(index.slots.iter().cloned());
    if let Some(autosave) = index.autosave.clone() {
        from_index.push(autosave);
    }
    for meta in from_index {
        if !meta.compatible || !seen_paths.insert(meta.path.clone()) {
            continue;
        }
        candidates.push(ContinueCandidate {
            label: meta.slot,
            path: meta.path,
            saved_unix_seconds: meta.saved_unix_seconds,
        });
    }

    let known_paths = [
        ("autosave".to_string(), CAMPAIGN_AUTOSAVE_PATH.to_string()),
        ("slot1".to_string(), CAMPAIGN_SAVE_PATH.to_string()),
        ("slot2".to_string(), CAMPAIGN_SAVE_SLOT_2_PATH.to_string()),
        ("slot3".to_string(), CAMPAIGN_SAVE_SLOT_3_PATH.to_string()),
    ];
    for (label, path) in known_paths {
        if !std::path::Path::new(&path).exists() || !seen_paths.insert(path.clone()) {
            continue;
        }
        let saved_unix_seconds = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        candidates.push(ContinueCandidate {
            label,
            path,
            saved_unix_seconds,
        });
    }

    candidates.sort_by(|a, b| {
        b.saved_unix_seconds
            .cmp(&a.saved_unix_seconds)
            .then_with(|| a.label.cmp(&b.label))
    });
    candidates
}

/// Build a human-readable preview string for a save slot.
/// Calls `crate::campaign_ops::format_slot_badge` for the badge prefix.
pub fn build_save_preview(meta: &SaveSlotMetadata) -> String {
    format!(
        "{} {} | turn={} version={} seed={} timestamp={}",
        crate::campaign_ops::format_slot_badge(Some(meta)),
        meta.slot,
        meta.global_turn,
        meta.save_version,
        meta.map_seed,
        meta.saved_unix_seconds
    )
}

pub fn panel_selected_entry(
    state: &CampaignSavePanelState,
    index: &CampaignSaveIndex,
) -> (String, String, Option<SaveSlotMetadata>) {
    let slot1 = index.slots.iter().find(|m| m.slot == "slot1").cloned();
    let slot2 = index.slots.iter().find(|m| m.slot == "slot2").cloned();
    let slot3 = index.slots.iter().find(|m| m.slot == "slot3").cloned();
    let autosave = index.autosave.clone();
    let entries = [
        ("slot1".to_string(), CAMPAIGN_SAVE_PATH.to_string(), slot1),
        (
            "slot2".to_string(),
            CAMPAIGN_SAVE_SLOT_2_PATH.to_string(),
            slot2,
        ),
        (
            "slot3".to_string(),
            CAMPAIGN_SAVE_SLOT_3_PATH.to_string(),
            slot3,
        ),
        (
            "autosave".to_string(),
            CAMPAIGN_AUTOSAVE_PATH.to_string(),
            autosave,
        ),
    ];
    let idx = state.selected.min(entries.len().saturating_sub(1));
    entries[idx].clone()
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

pub fn campaign_save_load_input_system(world: &mut World) {
    let (save_pressed, load_pressed, shift_pressed, ctrl_pressed) = {
        let Some(keyboard) = world.get_resource::<ButtonInput<KeyCode>>() else {
            return;
        };
        (
            keyboard.just_pressed(KeyCode::F5),
            keyboard.just_pressed(KeyCode::F9),
            keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight),
            keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight),
        )
    };
    if !save_pressed && !load_pressed {
        return;
    }
    let slot = if ctrl_pressed {
        3
    } else if shift_pressed {
        2
    } else {
        1
    };
    if save_pressed {
        let message = match crate::campaign_ops::save_campaign_to_slot(world, slot) {
            Ok(msg) => msg,
            Err(err) => format!("Save failed: {}", err),
        };
        world.resource_mut::<CampaignSaveNotice>().message = message;
    }

    if load_pressed {
        let message = match crate::campaign_ops::load_campaign_from_path_into_world(
            world,
            &format!("slot {}", slot),
            campaign_slot_path(slot),
        ) {
            Ok(msg) => msg,
            Err(err) => format!("Load failed: {}", err),
        };
        world.resource_mut::<CampaignSaveNotice>().message = message;
    }
}

pub fn campaign_save_panel_input_system(world: &mut World) {
    let (toggle_panel, up, down, save_key, request_load, confirm, cancel) = {
        let Some(keyboard) = world.get_resource::<ButtonInput<KeyCode>>() else {
            return;
        };
        (
            keyboard.just_pressed(KeyCode::F6),
            keyboard.just_pressed(KeyCode::ArrowUp),
            keyboard.just_pressed(KeyCode::ArrowDown),
            keyboard.just_pressed(KeyCode::KeyS),
            keyboard.just_pressed(KeyCode::KeyG),
            keyboard.just_pressed(KeyCode::Enter),
            keyboard.just_pressed(KeyCode::Escape),
        )
    };

    if !toggle_panel && !up && !down && !save_key && !request_load && !confirm && !cancel {
        return;
    }

    if toggle_panel {
        let mut panel = world.resource_mut::<CampaignSavePanelState>();
        panel.open = !panel.open;
        if !panel.open {
            panel.pending_load_path = None;
            panel.pending_label = None;
            panel.preview.clear();
        }
        world.resource_mut::<CampaignSaveNotice>().message = if panel.open {
            "Save panel opened (Up/Down select, S save, G load preview, Enter confirm, Esc cancel)"
                .to_string()
        } else {
            "Save panel closed.".to_string()
        };
        return;
    }

    if !world.resource::<CampaignSavePanelState>().open {
        return;
    }

    {
        let mut panel = world.resource_mut::<CampaignSavePanelState>();
        if up {
            panel.selected = panel.selected.saturating_sub(1);
        }
        if down {
            panel.selected = (panel.selected + 1).min(3);
        }
        if cancel {
            panel.pending_load_path = None;
            panel.pending_label = None;
            panel.preview.clear();
            world.resource_mut::<CampaignSaveNotice>().message =
                "Load confirmation canceled.".to_string();
            return;
        }
    }

    if save_key {
        let selected = world.resource::<CampaignSavePanelState>().selected;
        let message = match selected {
            0 => crate::campaign_ops::save_campaign_to_slot(world, 1),
            1 => crate::campaign_ops::save_campaign_to_slot(world, 2),
            _ => crate::campaign_ops::save_campaign_to_slot(world, 3),
        }
        .unwrap_or_else(|e| format!("Save failed: {}", e));
        world.resource_mut::<CampaignSaveNotice>().message = message;
        return;
    }

    if request_load {
        let (label, path, meta) = {
            let panel = world.resource::<CampaignSavePanelState>();
            let index = &world.resource::<CampaignSaveIndexState>().index;
            panel_selected_entry(&panel, index)
        };
        if let Some(m) = meta {
            if !m.compatible {
                world.resource_mut::<CampaignSaveNotice>().message = format!(
                    "Cannot load {}: save version {} is newer than supported {}.",
                    label, m.save_version, CURRENT_SAVE_VERSION
                );
                return;
            }
            let mut panel = world.resource_mut::<CampaignSavePanelState>();
            panel.pending_load_path = Some(path.clone());
            panel.pending_label = Some(label.clone());
            panel.preview = build_save_preview(&m);
            world.resource_mut::<CampaignSaveNotice>().message =
                format!("Previewing {}. Press Enter to confirm load.", label);
        } else {
            world.resource_mut::<CampaignSaveNotice>().message =
                format!("No save data found for {}.", label);
        }
    }

    if confirm {
        let (label, path) = {
            let panel = world.resource::<CampaignSavePanelState>();
            (
                panel
                    .pending_label
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                panel.pending_load_path.clone(),
            )
        };
        let Some(path) = path else {
            world.resource_mut::<CampaignSaveNotice>().message =
                "No load preview active. Press L first.".to_string();
            return;
        };
        let message = match crate::campaign_ops::load_campaign_from_path_into_world(world, &label, &path) {
            Ok(msg) => msg,
            Err(err) => format!("Load failed: {}", err),
        };
        {
            let mut panel = world.resource_mut::<CampaignSavePanelState>();
            panel.pending_load_path = None;
            panel.pending_label = None;
            panel.preview.clear();
        }
        world.resource_mut::<CampaignSaveNotice>().message = message;
    }
}

pub fn campaign_autosave_system(world: &mut World) {
    let (enabled, interval, last_turn) = {
        let state = world.resource::<CampaignAutosaveState>();
        (
            state.enabled,
            state.interval_turns,
            state.last_autosave_turn,
        )
    };
    if !enabled || interval == 0 {
        return;
    }
    let turn = world.resource::<crate::game_core::RunState>().global_turn;
    if turn == 0 || turn.saturating_sub(last_turn) < interval {
        return;
    }
    let data = crate::campaign_ops::snapshot_campaign_from_world(world);
    let message = match crate::campaign_ops::save_campaign_data(CAMPAIGN_AUTOSAVE_PATH, &data) {
        Ok(_) => {
            world
                .resource_mut::<CampaignAutosaveState>()
                .last_autosave_turn = turn;
            let metadata = crate::campaign_ops::campaign_save_metadata(
                "autosave".to_string(),
                CAMPAIGN_AUTOSAVE_PATH,
                &data,
            );
            {
                let mut index_state = world.resource_mut::<CampaignSaveIndexState>();
                index_state.index.autosave = Some(metadata);
                let _ = persist_campaign_save_index(&index_state.index);
            }
            format!("Autosaved t{} -> {}", turn, CAMPAIGN_AUTOSAVE_PATH)
        }
        Err(err) => format!("Autosave failed: {}", err),
    };
    world.resource_mut::<CampaignSaveNotice>().message = message;
}
