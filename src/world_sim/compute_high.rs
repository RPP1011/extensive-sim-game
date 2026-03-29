//! High-fidelity entity compute: full tactical combat as deltas.
//!
//! Each entity reads the frozen snapshot and produces Damage/Heal/Move/Status
//! deltas. Nothing mutates during compute.

use super::delta::WorldDelta;
use super::state::{Entity, EntityKind, WorldState, WorldTeam, StatusEffectKind};

/// Compute deltas for an entity at High fidelity (combat).
pub fn compute_entity_deltas(entity: &Entity, state: &WorldState) -> Vec<WorldDelta> {
    let mut out = Vec::new();
    compute_entity_deltas_into(entity, state, &mut out);
    out
}

/// Push deltas into `out` without allocating.
pub fn compute_entity_deltas_into(entity: &Entity, state: &WorldState, out: &mut Vec<WorldDelta>) {
    match entity.kind {
        EntityKind::Npc | EntityKind::Monster => compute_combatant_into(entity, state, out),
        EntityKind::Building | EntityKind::Projectile => {}
    }
}

fn compute_combatant_into(entity: &Entity, state: &WorldState, out: &mut Vec<WorldDelta>) {
    if is_stunned(entity) { return; }

    let target = find_nearest_hostile(entity, state);

    if let Some(target) = target {
        let dx = target.pos.0 - entity.pos.0;
        let dy = target.pos.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= entity.attack_range {
            out.push(WorldDelta::Damage {
                target_id: target.id,
                amount: entity.attack_damage,
                source_id: entity.id,
            });
        } else if dist > 0.001 {
            let speed = entity.move_speed * crate::world_sim::DT_SEC;
            let fx = dx / dist * speed;
            let fy = dy / dist * speed;
            out.push(WorldDelta::Move {
                entity_id: entity.id,
                force: (fx, fy),
            });
        }
    }

    tick_status_effects_into(entity, out);
}

fn is_stunned(entity: &Entity) -> bool {
    entity.status_effects.iter().any(|s| {
        matches!(s.kind, StatusEffectKind::Stun | StatusEffectKind::Root)
    })
}

fn find_nearest_hostile<'a>(entity: &Entity, state: &'a WorldState) -> Option<&'a Entity> {
    let hostile_team = match entity.team {
        WorldTeam::Friendly => WorldTeam::Hostile,
        WorldTeam::Hostile => WorldTeam::Friendly,
        WorldTeam::Neutral => return None,
    };

    state.entities.iter()
        .filter(|e| e.alive && e.team == hostile_team && e.id != entity.id)
        .min_by(|a, b| {
            let da = dist_sq(entity, a);
            let db = dist_sq(entity, b);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn dist_sq(a: &Entity, b: &Entity) -> f32 {
    let dx = a.pos.0 - b.pos.0;
    let dy = a.pos.1 - b.pos.1;
    dx * dx + dy * dy
}

fn tick_status_effects_into(entity: &Entity, out: &mut Vec<WorldDelta>) {
    for status in &entity.status_effects {
        match &status.kind {
            StatusEffectKind::Dot { damage_per_tick, tick_interval_ms, tick_elapsed_ms } => {
                if *tick_elapsed_ms >= *tick_interval_ms {
                    out.push(WorldDelta::Damage {
                        target_id: entity.id,
                        amount: *damage_per_tick,
                        source_id: status.source_id,
                    });
                }
            }
            StatusEffectKind::Hot { heal_per_tick, tick_interval_ms, tick_elapsed_ms } => {
                if *tick_elapsed_ms >= *tick_interval_ms {
                    out.push(WorldDelta::Heal {
                        target_id: entity.id,
                        amount: *heal_per_tick,
                        source_id: status.source_id,
                    });
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::state::*;
    use crate::world_sim::delta::merge_deltas;
    use crate::world_sim::apply::apply_deltas;

    fn combat_state() -> WorldState {
        let mut s = WorldState::new(42);
        let mut npc = Entity::new_npc(1, (0.0, 0.0));
        npc.attack_damage = 20.0;
        npc.attack_range = 2.0;
        npc.grid_id = Some(100);

        let mut monster = Entity::new_monster(2, (1.0, 0.0), 1);
        monster.hp = 50.0;
        monster.max_hp = 50.0;
        monster.attack_damage = 10.0;
        monster.attack_range = 2.0;
        monster.grid_id = Some(100);

        s.entities.push(npc);
        s.entities.push(monster);
        s.grids.push(LocalGrid {
            id: 100,
            fidelity: super::super::fidelity::Fidelity::High,
            center: (0.0, 0.0),
            radius: 20.0,
            entity_ids: vec![1, 2],
        });
        s
    }

    #[test]
    fn in_range_attack() {
        let state = combat_state();
        let deltas = compute_entity_deltas(&state.entities[0], &state);
        // NPC should attack monster (in range).
        assert!(deltas.iter().any(|d| matches!(d,
            WorldDelta::Damage { target_id: 2, amount, .. } if (*amount - 20.0).abs() < 1e-6
        )));
    }

    #[test]
    fn out_of_range_moves() {
        let mut state = combat_state();
        state.entities[1].pos = (10.0, 0.0); // far away
        let deltas = compute_entity_deltas(&state.entities[0], &state);
        // NPC should move, not attack.
        assert!(deltas.iter().any(|d| matches!(d, WorldDelta::Move { entity_id: 1, .. })));
        assert!(!deltas.iter().any(|d| matches!(d, WorldDelta::Damage { .. })));
    }

    #[test]
    fn stunned_does_nothing() {
        let mut state = combat_state();
        state.entities[0].status_effects.push(StatusEffect {
            kind: StatusEffectKind::Stun,
            source_id: 2,
            remaining_ms: 1000,
        });
        let deltas = compute_entity_deltas(&state.entities[0], &state);
        assert!(deltas.is_empty());
    }

    #[test]
    fn mutual_combat() {
        let state = combat_state();
        // Both entities produce deltas independently from the same snapshot.
        let npc_deltas = compute_entity_deltas(&state.entities[0], &state);
        let monster_deltas = compute_entity_deltas(&state.entities[1], &state);
        // Both should attack each other.
        assert!(npc_deltas.iter().any(|d| matches!(d, WorldDelta::Damage { target_id: 2, .. })));
        assert!(monster_deltas.iter().any(|d| matches!(d, WorldDelta::Damage { target_id: 1, .. })));

        // Merge and apply.
        let all: Vec<WorldDelta> = npc_deltas.into_iter().chain(monster_deltas).collect();
        let merged = merge_deltas(all);
        let next = apply_deltas(&state, &merged);

        // Both took damage simultaneously.
        let npc = next.entity(1).unwrap();
        let monster = next.entity(2).unwrap();
        assert!(npc.hp < 100.0);
        assert!(monster.hp < 50.0);
    }
}
