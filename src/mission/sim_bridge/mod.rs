mod types;
mod builders;
mod systems;

// Re-export types (used by bin crates)
#[allow(unused_imports)]
pub use types::{
    MissionOutcome,
    MissionEventLog,
    SimEventBuffer,
    MissionSimState,
    EnemyAiState,
    PlayerOrderState,
    PlayerUnitMarker,
    scale_enemy_stats,
    threat_level,
    threat_level_roman,
};

// Re-export builders (used by bin crates)
#[allow(unused_imports)]
pub use builders::{
    build_default_sim,
    build_sim_with_hero_templates,
    build_sim_with_templates,
};

// Re-export systems
pub use systems::{
    advance_sim_system,
    apply_vfx_from_sim_events_system,
    apply_audio_sfx_from_sim_events_system,
    player_ground_click_system,
    apply_player_orders_system,
};
