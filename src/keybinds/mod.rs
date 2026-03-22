//! Centralized keybind system with context-aware input abstraction.
//!
//! Replaces scattered `keyboard.just_pressed(KeyCode::X)` checks with a
//! `GameAction` enum and TOML-configurable bindings.

mod actions;
mod config;
mod dispatcher;

pub use actions::{GameAction, InputContext};
pub use config::KeybindConfig;
pub use dispatcher::{ActionEvents, keybind_dispatch_system};
