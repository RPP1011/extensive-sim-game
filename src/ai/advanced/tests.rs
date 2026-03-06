use super::*;
use crate::ai::core::{Team, FIXED_TICK_MS};

#[test]
fn phase7_is_deterministic() {
    let a = run_spatial_sample(37, 320, FIXED_TICK_MS);
    let b = run_spatial_sample(37, 320, FIXED_TICK_MS);
    assert_eq!(a.event_log_hash, b.event_log_hash);
    assert_eq!(a.final_state_hash, b.final_state_hash);
}

#[test]
fn phase8_is_deterministic() {
    let a = run_tactical_sample(37, 320, FIXED_TICK_MS);
    let b = run_tactical_sample(37, 320, FIXED_TICK_MS);
    assert_eq!(a.event_log_hash, b.event_log_hash);
    assert_eq!(a.final_state_hash, b.final_state_hash);
}

#[test]
fn phase9_improves_resolution_over_phase7() {
    let p7 = run_spatial_sample(31, 320, FIXED_TICK_MS);
    let p9 = run_coordination_sample(31, 320, FIXED_TICK_MS);
    let p7_enemy_alive = p7
        .final_state
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .count();
    let p9_enemy_alive = p9
        .final_state
        .units
        .iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0)
        .count();
    assert!(p9_enemy_alive <= p7_enemy_alive);
    assert_eq!(p9.metrics.invariant_violations, 0);
}

#[test]
fn horde_chokepoint_pathing_is_deterministic() {
    let a = run_horde_chokepoint_sample(101, 420, FIXED_TICK_MS);
    let b = run_horde_chokepoint_sample(101, 420, FIXED_TICK_MS);
    assert_eq!(a.event_log_hash, b.event_log_hash);
    assert_eq!(a.final_state_hash, b.final_state_hash);
    assert_eq!(a.metrics.invariant_violations, 0);
}

#[test]
fn horde_chokepoint_hero_favored_is_hero_win() {
    let result = run_horde_chokepoint_hero_favored_sample(202, 420, FIXED_TICK_MS);
    assert_eq!(result.metrics.winner, Some(Team::Hero));
    assert_eq!(result.metrics.invariant_violations, 0);
}
