//! Unified movement system — the ONLY system that moves entities.
//!
//! All other systems set entity.move_target to request movement.
//! This system reads move_target, moves the entity toward it at
//! entity.move_speed * entity.move_speed_mult, and clears it on arrival.

use crate::world_sim::state::*;

const ARRIVAL_DIST: f32 = 1.5;

pub fn advance_movement(state: &mut WorldState) {
    let dt = crate::world_sim::DT_SEC;
    for entity in &mut state.entities {
        if !entity.alive { continue; }
        let target = match entity.move_target {
            Some(t) => t,
            None => continue,
        };

        // CC blocks movement — stun/root/freeze prevent moving
        let cc_blocked = entity.status_effects.iter().any(|s| matches!(s.kind,
            StatusEffectKind::Stun | StatusEffectKind::Root
        ));
        if cc_blocked { continue; }

        let dx = target.0 - entity.pos.0;
        let dy = target.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < ARRIVAL_DIST {
            entity.move_target = None;
            continue;
        }
        let speed = entity.move_speed * entity.move_speed_mult * dt;
        entity.pos.0 += dx / dist * speed;
        entity.pos.1 += dy / dist * speed;
    }
}
