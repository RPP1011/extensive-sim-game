use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Event queue resource
// ---------------------------------------------------------------------------

#[derive(Resource, Default)]
pub struct VfxEventQueue {
    pub pending: Vec<VfxEvent>,
}

pub enum VfxEvent {
    Damage { world_pos: Vec3, amount: i32, is_crit: bool },
    Heal   { world_pos: Vec3, amount: i32 },
    Death  { world_pos: Vec3 },
    Control { world_pos: Vec3 },
    ChannelStart { unit_id: u32 },
    ChannelEnd { unit_id: u32 },
    ChainFlash { from_pos: Vec3, to_pos: Vec3, color: Color },
    Trail { from: Vec3, to: Vec3, color: Color },
    ShieldFlash { world_pos: Vec3, amount: i32 },
    Miss { world_pos: Vec3 },
    Resist { world_pos: Vec3 },
    ZonePulse { zone_id: u32 },
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct FloatingText {
    pub lifetime: f32,
    pub velocity: Vec3,
}

#[derive(Component)]
pub struct HitFlash {
    pub remaining: f32,
}

#[derive(Component)]
pub struct DeathFade {
    pub remaining: f32,
}

/// Sync-from-state: tracks a sim projectile.
#[derive(Component)]
pub struct ProjectileVisual {
    pub source_id: u32,
    pub target_id: u32,
}

/// Sync-from-state: tracks a sim zone.
#[derive(Component)]
pub struct ZoneVisual {
    pub zone_id: u32,
}

/// Brief emissive pulse on zone tick.
#[derive(Component)]
pub struct ZonePulseEffect {
    pub remaining: f32,
}

/// Sync-from-state: tracks a sim tether beam.
#[derive(Component)]
pub struct TetherVisual {
    pub source_id: u32,
    pub target_id: u32,
}

/// Channel ring at unit feet.
#[derive(Component)]
pub struct ChannelRing {
    pub unit_id: u32,
    pub elapsed: f32,
}

/// Sync-from-state: shield indicator ring.
#[derive(Component)]
pub struct ShieldIndicator {
    pub unit_id: u32,
}

/// Sync-from-state: CC/status indicator above unit.
#[derive(Component)]
pub struct StatusIndicator {
    pub unit_id: u32,
    pub kind: StatusIndicatorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusIndicatorKind {
    Stun,
    Root,
    Silence,
    Slow,
    Fear,
}

/// Sync-from-state: buff/debuff ring at feet.
#[derive(Component)]
pub struct BuffDebuffRing {
    pub unit_id: u32,
    pub is_buff: bool,
}

/// Periodic particle emitter for DoT/HoT.
#[derive(Component)]
pub struct DotHotParticleTimer {
    pub unit_id: u32,
    pub has_dot: bool,
    pub has_hot: bool,
    pub cooldown: f32,
}

// Total fade duration in seconds.
pub const DEATH_FADE_TOTAL: f32 = 0.4;
