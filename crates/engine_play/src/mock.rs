//! A headless `PlayableRuntime` for tests. Returns `engine_play_api`'s sample
//! descriptor fixtures (so the whole crate lands in parallel with the
//! compiler) plus a couple of fake agents the bridge/loop tests key on.

use engine_play_api::{AgentView, PlayableRuntime};

/// The render descriptor served by `MockRuntime` (the Plan-0 sample fixture:
/// cyan player in mana band [0.5,1.5], orange enemy in [1.5,2.5], a nova Ring
/// every 40 ticks, a bolt BeamToNearest every 12 ticks).
pub const MOCK_RENDER: &str = include_str!("../../engine_play_api/fixtures/vs_render.json");
/// The controls descriptor served by `MockRuntime` (WASD → ctl.move_x/move_y).
pub const MOCK_CONTROLS: &str = include_str!("../../engine_play_api/fixtures/vs_controls.json");

/// A hand-written UI descriptor JSON owned by this crate. `engine_ui::UiModel`
/// gains a `from_json` in a parallel track; until then the player builds its
/// `UiModel` directly in Rust (see [`crate::player::mock_ui_model`]). This
/// string is returned by `ui_descriptor()` purely to satisfy the trait — the
/// player does not parse it yet.
pub const MOCK_UI: &str = r#"{ "hud": [], "screens": [] }"#;

/// A minimal in-memory `PlayableRuntime`. Holds a fixed agent roster (a player
/// at the origin in mana band ~1.0 and an enemy in band ~2.0), advances `tick`
/// on `step`, and records every `set_input` write so tests can assert the
/// controls→input path fired.
pub struct MockRuntime {
    tick: u64,
    agents: Vec<AgentView>,
    /// Every `(field, value)` written via `set_input`, in order.
    pub last_input: Vec<(String, f32)>,
    /// Materialized-view values keyed `(view, slot)`. Defaults to 0.0.
    pub views: std::collections::HashMap<(String, u32), f32>,
    /// S13: named TEXTS this runtime answers `view_text` with.
    pub texts: std::collections::HashMap<String, String>,
}

impl MockRuntime {
    /// A player at the origin (mana 1.0, hp 100) and an enemy nearby (mana 2.0,
    /// hp 30) so render-band selection + nearest-target VFX have something to
    /// key on.
    pub fn new() -> Self {
        Self {
            tick: 0,
            agents: vec![
                AgentView {
                    pos: [0.0, 0.0, 0.0],
                    alive: true,
                    hp: 100.0,
                    mana: 1.0,
                    move_speed: 0.4,
                    creature_type: 0,
                },
                AgentView {
                    pos: [8.0, 0.0, 0.0],
                    alive: true,
                    hp: 30.0,
                    mana: 2.0,
                    move_speed: 0.5,
                    creature_type: 1,
                },
            ],
            last_input: Vec::new(),
            views: std::collections::HashMap::new(),
            texts: std::collections::HashMap::new(),
        }
    }

    /// The last value written for `field`, if any (test helper).
    pub fn last_input_for(&self, field: &str) -> Option<f32> {
        self.last_input
            .iter()
            .rev()
            .find(|(f, _)| f == field)
            .map(|(_, v)| *v)
    }
}

impl Default for MockRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayableRuntime for MockRuntime {
    fn tick(&self) -> u64 {
        self.tick
    }
    fn step(&mut self) {
        self.tick += 1;
    }
    fn set_input(&mut self, field: &str, value: f32) {
        self.last_input.push((field.to_string(), value));
    }
    fn agent_snapshot(&mut self) -> Vec<AgentView> {
        self.agents.clone()
    }
    fn view_value(&mut self, view: &str, slot: u32) -> f32 {
        self.views
            .get(&(view.to_string(), slot))
            .copied()
            .unwrap_or(0.0)
    }
    fn view_text(&mut self, view: &str) -> Option<String> {
        self.texts.get(view).cloned()
    }
    fn render_descriptor(&self) -> &'static str {
        MOCK_RENDER
    }
    fn controls_descriptor(&self) -> &'static str {
        MOCK_CONTROLS
    }
    fn ui_descriptor(&self) -> &'static str {
        MOCK_UI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_play_api::{ControlsDescriptor, RenderDescriptor};

    #[test]
    fn mock_serves_parseable_descriptors() {
        let mut rt = MockRuntime::new();
        assert_eq!(rt.tick(), 0);
        let r = RenderDescriptor::from_json(rt.render_descriptor()).unwrap();
        assert_eq!(r.agents.len(), 2);
        let c = ControlsDescriptor::from_json(rt.controls_descriptor()).unwrap();
        assert_eq!(c.bindings.len(), 4);
        // Two live agents in the snapshot.
        assert_eq!(rt.agent_snapshot().len(), 2);
        rt.step();
        assert_eq!(rt.tick(), 1);
    }

    #[test]
    fn mock_records_inputs() {
        let mut rt = MockRuntime::new();
        rt.set_input("ctl.move_x", 1.0);
        assert_eq!(rt.last_input_for("ctl.move_x"), Some(1.0));
        assert_eq!(rt.last_input_for("ctl.move_y"), None);
    }
}
