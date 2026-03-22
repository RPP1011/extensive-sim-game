use std::collections::HashSet;
use bevy::prelude::*;
use super::actions::{GameAction, InputContext};
use super::config::KeybindConfig;

/// Resource holding the set of game actions triggered this frame.
#[derive(Resource, Default)]
pub struct ActionEvents {
    pressed: HashSet<GameAction>,
}

impl ActionEvents {
    /// Returns true if the given action was triggered this frame.
    pub fn just_pressed(&self, action: GameAction) -> bool {
        self.pressed.contains(&action)
    }

    /// Returns a boolean mask over all GameAction variants indicating
    /// which actions are currently valid in the given context.
    pub fn action_mask(&self, context: InputContext) -> Vec<bool> {
        GameAction::all_variants()
            .iter()
            .map(|a| a.valid_in_context(context))
            .collect()
    }

    pub fn clear(&mut self) {
        self.pressed.clear();
    }
}

/// Bevy system that runs early in Update to populate ActionEvents
/// from keyboard input using the KeybindConfig mapping.
pub fn keybind_dispatch_system(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    config: Res<KeybindConfig>,
    mut actions: ResMut<ActionEvents>,
) {
    actions.clear();

    let Some(keyboard) = keyboard else {
        return;
    };

    for (action, keys) in &config.bindings {
        if keys.iter().any(|k| keyboard.just_pressed(*k)) {
            actions.pressed.insert(*action);
        }
    }
}
