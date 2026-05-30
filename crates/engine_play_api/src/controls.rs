//! Controls descriptor — maps keys to `@runtime` input-field writes. Emitted by
//! the compiler and consumed by the generic player's input mapper.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum BindMode {
    /// Apply `value` every frame the key is held (summed across held keys, then
    /// the player normalizes paired axes like move_x/move_y).
    Hold,
    /// Fire `value` once on key-down (the player tracks edges).
    Press,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KeyBinding {
    /// Lowercased key name (e.g. "w", "space").
    pub key: String,
    /// Target `@runtime` field, "<block>.<field>" (e.g. "ctl.move_y").
    pub field: String,
    pub value: f32,
    pub mode: BindMode,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ControlsDescriptor {
    pub bindings: Vec<KeyBinding>,
}

impl ControlsDescriptor {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
