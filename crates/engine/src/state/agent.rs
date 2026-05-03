use engine_data::entities::CreatureType;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MovementMode {
    #[default]
    Walk,
    Climb,
    Fly,
    Swim,
    Fall,
}

#[derive(Clone, Debug)]
pub struct AgentSpawn {
    pub creature_type: CreatureType,
    pub pos:           glam::Vec3,
    pub hp:            f32,
    /// Maximum HP — the cap used by `hp_pct = hp / max_hp` scoring and
    /// by any healing that restores up to full. Independent of `hp` so
    /// a "wounded" spawn can start at `hp=10, max_hp=100` and report a
    /// low `hp_pct` for target-selection. Task 150 split this out of
    /// `spec.hp.max(1.0)`, which made freshly-spawned agents always
    /// report `hp_pct = 1.0` and broke pct-based scoring.
    pub max_hp:        f32,
}

impl Default for AgentSpawn {
    fn default() -> Self {
        // Match default hp so the default fixture reports `hp_pct = 1.0`
        // rather than the degenerate `hp / 0` produced by a zeroed cap.
        Self {
            creature_type: CreatureType::default(),
            pos:           glam::Vec3::ZERO,
            hp:            100.0,
            max_hp:        100.0,
        }
    }
}
