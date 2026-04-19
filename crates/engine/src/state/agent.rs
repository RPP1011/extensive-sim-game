use crate::creature::CreatureType;

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

#[derive(Clone, Debug, Default)]
pub struct AgentSpawn {
    pub creature_type: CreatureType,
    pub pos:           glam::Vec3,
    pub hp:            f32,
}
