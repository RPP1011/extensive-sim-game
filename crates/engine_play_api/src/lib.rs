//! `engine_play_api` — the frozen contract between a compiled `.sim` runtime
//! and the generic player (`engine_play`). A leaf crate (no heavy deps) so it
//! can be depended on by both `sims` (which generates the trait impl) and
//! `engine_play` (which consumes it) without a dependency cycle.
//!
//! The UI descriptor reuses `engine_ui::UiModel` (parsed there), so this crate
//! defines only the render + controls descriptors and the runtime trait.

pub mod controls;
pub mod render;

pub use controls::{BindMode, ControlsDescriptor, KeyBinding};
pub use render::{AgentVisual, CameraSpec, FieldRange, RenderDescriptor, VfxKind, VfxSpec};

/// Uniform interface a compiled `.sim` runtime exposes so one binary can play
/// any game. Sits beside `engine::backend::Backend` (the CPU/GPU seam) as the
/// authoring/player seam.
pub trait PlayableRuntime {
    fn tick(&self) -> u64;
    fn step(&mut self);
    /// Write an `@runtime` config field by name (dispatches to the generated
    /// `set_config_<block>_<field>` setter; u32 fields cast from the f32 arg).
    fn set_input(&mut self, field: &str, value: f32);
    /// Snapshot the live agents (the columns `render` keys on).
    fn agent_snapshot(&mut self) -> Vec<AgentView>;
    /// Materialized-view value at an agent slot (e.g. "xp"); 0.0 if unknown.
    fn view_value(&mut self, view: &str, slot: u32) -> f32;
    /// S13 — the TEXT half of `view_value`: a named STRING the HUD may print
    /// (`None` = this runtime has no text under that name, and the numeric
    /// channel answers instead). DEFAULTED, so every existing implementor —
    /// including every generated `.sim` runtime — is unaffected; a host that
    /// carries prose (`webband_play`'s petition, the selected colonist's
    /// name) overrides it.
    fn view_text(&mut self, _view: &str) -> Option<String> {
        None
    }
    fn render_descriptor(&self) -> &'static str;
    fn controls_descriptor(&self) -> &'static str;
    fn ui_descriptor(&self) -> &'static str;
}

/// A live agent as the renderer sees it — position plus the columns a
/// `RenderDescriptor` may key on. Missing columns default to 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AgentView {
    pub pos: [f32; 3],
    pub alive: bool,
    pub hp: f32,
    pub mana: f32,
    pub move_speed: f32,
    pub creature_type: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_roundtrip() {
        let r = RenderDescriptor {
            arena_radius: 42.0,
            camera: CameraSpec::Follow(FieldRange { field: "mana".into(), lo: 0.5, hi: 1.5 }),
            agents: vec![AgentVisual {
                when: FieldRange { field: "mana".into(), lo: 0.5, hi: 1.5 },
                color: [0, 220, 220],
            }],
            vfx: vec![VfxSpec {
                on_rule: "NovaFire".into(),
                period: 40,
                kind: VfxKind::Ring,
                radius: 6.0,
                color: [255, 255, 120],
            }],
        };
        let j = serde_json::to_string(&r).unwrap();
        assert_eq!(RenderDescriptor::from_json(&j).unwrap(), r);

        let c = ControlsDescriptor {
            bindings: vec![KeyBinding {
                key: "w".into(),
                field: "ctl.move_y".into(),
                value: 1.0,
                mode: BindMode::Hold,
            }],
        };
        let cj = serde_json::to_string(&c).unwrap();
        assert_eq!(ControlsDescriptor::from_json(&cj).unwrap(), c);
    }

    #[test]
    fn sample_fixtures_parse() {
        let r = include_str!("../fixtures/vs_render.json");
        let c = include_str!("../fixtures/vs_controls.json");
        assert_eq!(RenderDescriptor::from_json(r).unwrap().agents.len(), 2);
        assert_eq!(ControlsDescriptor::from_json(c).unwrap().bindings.len(), 4);
    }
}
