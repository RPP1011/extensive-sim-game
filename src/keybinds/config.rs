use std::collections::HashMap;
use std::path::Path;
use bevy::prelude::*;
use super::actions::GameAction;

/// Maps each game action to one or more key codes.
#[derive(Resource, Clone)]
pub struct KeybindConfig {
    pub bindings: HashMap<GameAction, Vec<KeyCode>>,
}

impl Default for KeybindConfig {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        // Navigation
        bindings.insert(GameAction::MenuUp, vec![KeyCode::ArrowUp, KeyCode::KeyW]);
        bindings.insert(GameAction::MenuDown, vec![KeyCode::ArrowDown, KeyCode::KeyS]);
        bindings.insert(GameAction::MenuLeft, vec![KeyCode::ArrowLeft, KeyCode::KeyA]);
        bindings.insert(GameAction::MenuRight, vec![KeyCode::ArrowRight, KeyCode::KeyD]);
        bindings.insert(GameAction::MenuSelect, vec![KeyCode::Enter, KeyCode::Space]);
        bindings.insert(GameAction::MenuBack, vec![KeyCode::Escape, KeyCode::Backspace]);
        // UI
        bindings.insert(GameAction::ToggleQuestLog, vec![KeyCode::KeyJ]);
        bindings.insert(GameAction::ToggleSettings, vec![KeyCode::Escape]);
        bindings.insert(GameAction::ToggleTutorial, vec![KeyCode::F1]);
        // Save/Load
        bindings.insert(GameAction::QuickSave, vec![KeyCode::F5]);
        bindings.insert(GameAction::QuickLoad, vec![KeyCode::F9]);
        bindings.insert(GameAction::SaveBrowser, vec![KeyCode::F6]);
        bindings.insert(GameAction::QuickSaveSlot2, vec![]); // Shift+F5 handled via modifier
        bindings.insert(GameAction::QuickLoadSlot2, vec![]); // Shift+F9
        bindings.insert(GameAction::QuickSaveSlot3, vec![]); // Ctrl+F5
        bindings.insert(GameAction::QuickLoadSlot3, vec![]); // Ctrl+F9
        // Combat
        bindings.insert(GameAction::PauseResume, vec![KeyCode::Space]);
        bindings.insert(GameAction::SpeedUp, vec![KeyCode::BracketRight]);
        bindings.insert(GameAction::SlowDown, vec![KeyCode::BracketLeft]);
        bindings.insert(GameAction::NextUnit, vec![KeyCode::Tab]);
        bindings.insert(GameAction::PrevUnit, vec![]); // Shift+Tab
        bindings.insert(GameAction::ViewBattle, vec![KeyCode::KeyB]);
        // Abilities
        bindings.insert(GameAction::Ability1, vec![KeyCode::Digit1]);
        bindings.insert(GameAction::Ability2, vec![KeyCode::Digit2]);
        bindings.insert(GameAction::Ability3, vec![KeyCode::Digit3]);
        bindings.insert(GameAction::Ability4, vec![KeyCode::Digit4]);
        bindings.insert(GameAction::Ability5, vec![KeyCode::Digit5]);
        bindings.insert(GameAction::Ability6, vec![KeyCode::Digit6]);
        bindings.insert(GameAction::Ability7, vec![KeyCode::Digit7]);
        bindings.insert(GameAction::Ability8, vec![KeyCode::Digit8]);
        bindings.insert(GameAction::Ability9, vec![KeyCode::Digit9]);
        // Camera
        bindings.insert(GameAction::CameraUp, vec![KeyCode::ArrowUp]);
        bindings.insert(GameAction::CameraDown, vec![KeyCode::ArrowDown]);
        bindings.insert(GameAction::CameraLeft, vec![KeyCode::ArrowLeft]);
        bindings.insert(GameAction::CameraRight, vec![KeyCode::ArrowRight]);
        bindings.insert(GameAction::CameraZoomIn, vec![KeyCode::Equal]);
        bindings.insert(GameAction::CameraZoomOut, vec![KeyCode::Minus]);
        // Screenshot
        bindings.insert(GameAction::Screenshot, vec![KeyCode::F12, KeyCode::Semicolon]);
        // Replay
        bindings.insert(GameAction::ReplayPlayPause, vec![KeyCode::Space]);
        bindings.insert(GameAction::ReplayNextFrame, vec![KeyCode::ArrowRight]);
        bindings.insert(GameAction::ReplayPrevFrame, vec![KeyCode::ArrowLeft]);
        bindings.insert(GameAction::ReplayExit, vec![KeyCode::Escape]);
        // General
        bindings.insert(GameAction::Confirm, vec![KeyCode::Enter]);
        bindings.insert(GameAction::Cancel, vec![KeyCode::Escape]);

        Self { bindings }
    }
}

impl KeybindConfig {
    /// Load keybind config from TOML file, falling back to defaults.
    pub fn load_or_default(path: Option<&Path>) -> Self {
        let Some(path) = path else {
            return Self::default();
        };
        match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    fn from_toml(text: &str) -> Option<Self> {
        let table: toml::Table = text.parse().ok()?;
        let mut config = Self::default();

        // Parse sections and override defaults
        for (section, values) in &table {
            let Some(section_table) = values.as_table() else { continue };
            for (action_name, keys) in section_table {
                if let (Some(action), Some(key_list)) = (
                    Self::parse_action_name(section, action_name),
                    keys.as_array(),
                ) {
                    let keycodes: Vec<KeyCode> = key_list
                        .iter()
                        .filter_map(|v| v.as_str())
                        .filter_map(Self::parse_keycode)
                        .collect();
                    if !keycodes.is_empty() {
                        config.bindings.insert(action, keycodes);
                    }
                }
            }
        }
        Some(config)
    }

    fn parse_action_name(section: &str, name: &str) -> Option<GameAction> {
        match (section, name) {
            ("navigation", "menu_up") => Some(GameAction::MenuUp),
            ("navigation", "menu_down") => Some(GameAction::MenuDown),
            ("navigation", "menu_left") => Some(GameAction::MenuLeft),
            ("navigation", "menu_right") => Some(GameAction::MenuRight),
            ("navigation", "menu_select") => Some(GameAction::MenuSelect),
            ("navigation", "menu_back") => Some(GameAction::MenuBack),
            ("ui", "quest_log") => Some(GameAction::ToggleQuestLog),
            ("ui", "settings") => Some(GameAction::ToggleSettings),
            ("ui", "tutorial") => Some(GameAction::ToggleTutorial),
            ("save", "quick_save") => Some(GameAction::QuickSave),
            ("save", "quick_load") => Some(GameAction::QuickLoad),
            ("save", "save_browser") => Some(GameAction::SaveBrowser),
            ("combat", "pause") => Some(GameAction::PauseResume),
            ("combat", "speed_up") => Some(GameAction::SpeedUp),
            ("combat", "slow_down") => Some(GameAction::SlowDown),
            ("combat", "next_unit") => Some(GameAction::NextUnit),
            ("combat", "prev_unit") => Some(GameAction::PrevUnit),
            ("screenshot", "capture") => Some(GameAction::Screenshot),
            ("replay", "play_pause") => Some(GameAction::ReplayPlayPause),
            ("replay", "next_frame") => Some(GameAction::ReplayNextFrame),
            ("replay", "prev_frame") => Some(GameAction::ReplayPrevFrame),
            ("replay", "exit") => Some(GameAction::ReplayExit),
            _ => None,
        }
    }

    fn parse_keycode(s: &str) -> Option<KeyCode> {
        match s {
            "ArrowUp" => Some(KeyCode::ArrowUp),
            "ArrowDown" => Some(KeyCode::ArrowDown),
            "ArrowLeft" => Some(KeyCode::ArrowLeft),
            "ArrowRight" => Some(KeyCode::ArrowRight),
            "Enter" => Some(KeyCode::Enter),
            "Space" => Some(KeyCode::Space),
            "Escape" => Some(KeyCode::Escape),
            "Backspace" => Some(KeyCode::Backspace),
            "Tab" => Some(KeyCode::Tab),
            "KeyA" => Some(KeyCode::KeyA),
            "KeyB" => Some(KeyCode::KeyB),
            "KeyD" => Some(KeyCode::KeyD),
            "KeyJ" => Some(KeyCode::KeyJ),
            "KeyS" => Some(KeyCode::KeyS),
            "KeyW" => Some(KeyCode::KeyW),
            "F1" => Some(KeyCode::F1),
            "F5" => Some(KeyCode::F5),
            "F6" => Some(KeyCode::F6),
            "F9" => Some(KeyCode::F9),
            "F12" => Some(KeyCode::F12),
            "Semicolon" => Some(KeyCode::Semicolon),
            "Digit1" => Some(KeyCode::Digit1),
            "Digit2" => Some(KeyCode::Digit2),
            "Digit3" => Some(KeyCode::Digit3),
            "Digit4" => Some(KeyCode::Digit4),
            "Digit5" => Some(KeyCode::Digit5),
            "Digit6" => Some(KeyCode::Digit6),
            "Digit7" => Some(KeyCode::Digit7),
            "Digit8" => Some(KeyCode::Digit8),
            "Digit9" => Some(KeyCode::Digit9),
            "BracketLeft" => Some(KeyCode::BracketLeft),
            "BracketRight" => Some(KeyCode::BracketRight),
            "Equal" => Some(KeyCode::Equal),
            "Minus" => Some(KeyCode::Minus),
            _ => None,
        }
    }

    /// Check for duplicate key bindings across different actions.
    pub fn find_conflicts(&self) -> Vec<(KeyCode, Vec<GameAction>)> {
        let mut key_to_actions: HashMap<KeyCode, Vec<GameAction>> = HashMap::new();
        for (action, keys) in &self.bindings {
            for key in keys {
                key_to_actions.entry(*key).or_default().push(*action);
            }
        }
        key_to_actions
            .into_iter()
            .filter(|(_, actions)| actions.len() > 1)
            .collect()
    }
}
