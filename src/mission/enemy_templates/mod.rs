mod templates;
mod waves;

// Re-export everything for backward compatibility
pub use waves::{
    default_enemy_wave,
    BOSS_UNIT_ID,
    generate_boss,
    is_climax_room,
};
