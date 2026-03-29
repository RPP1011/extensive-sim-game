mod types;
mod spawn;
mod sync;
mod indicators;

// Re-export all public types
pub use types::*;

// Re-export spawn/update systems
#[allow(unused_imports)]
pub use spawn::{
    spawn_vfx_system,
    update_floating_text_system,
    update_hit_flash_system,
    update_death_fade_system,
};

// Re-export sync systems
#[allow(unused_imports)]
pub use sync::{
    sync_projectile_visuals_system,
    sync_zone_visuals_system,
    update_zone_pulse_system,
    sync_tether_visuals_system,
    update_channel_ring_system,
};

// Re-export indicator systems
#[allow(unused_imports)]
pub use indicators::{
    sync_shield_indicators_system,
    sync_status_indicators_system,
    sync_buff_debuff_rings_system,
    emit_dot_hot_particles_system,
};
