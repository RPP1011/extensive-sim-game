//! Render descriptor — how a `.sim`'s agents map to visuals. Emitted by the
//! compiler (matching this serde shape) and consumed by the generic player.
use serde::{Deserialize, Serialize};

/// A `[lo, hi]` range over a named agent column (e.g. "mana"). Used both to
/// pick an agent's visual and to select VFX/camera targets.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FieldRange {
    pub field: String,
    pub lo: f32,
    pub hi: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CameraSpec {
    /// Follow the (first) agent whose column falls in the range (e.g. the player).
    Follow(FieldRange),
    /// Fixed observer framing the arena.
    Observer,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AgentVisual {
    pub when: FieldRange,
    pub color: [u8; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VfxKind {
    /// A ring at `radius` around the followed agent on each period tick.
    Ring,
    /// A beam from the followed agent to the nearest agent matching `target`.
    BeamToNearest { target: FieldRange },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VfxSpec {
    /// The rule whose firing this VFX visualizes (informational; timing uses `period`).
    pub on_rule: String,
    /// Draw the VFX when `tick % period == 0`.
    pub period: u32,
    pub kind: VfxKind,
    pub radius: f32,
    pub color: [u8; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RenderDescriptor {
    pub arena_radius: f32,
    pub camera: CameraSpec,
    pub agents: Vec<AgentVisual>,
    pub vfx: Vec<VfxSpec>,
}

impl RenderDescriptor {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
