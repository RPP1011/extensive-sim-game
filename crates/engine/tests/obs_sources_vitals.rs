use engine_data::entities::CreatureType;
use engine::obs::{FeatureSource, VitalsSource};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn vitals_dim_is_four() {
    assert_eq!(VitalsSource.dim(), 4);
}

#[test]
fn vitals_pack_reads_hp_frac_hunger_thirst_rest() {
    let mut state = SimState::new(4, 42);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 50.0,
            max_hp: 100.0,
        })
        .unwrap();
    // hp_frac = hp/max_hp = 50/100 = 0.5 at spawn.
    state.set_agent_hunger(a, 0.3);
    state.set_agent_thirst(a, 0.7);
    state.set_agent_rest_timer(a, 0.9);

    let mut out = [0.0f32; 4];
    VitalsSource.pack(&state, a, &mut out);
    assert!((out[0] - 0.5).abs() < 1e-6, "hp_frac = {}", out[0]);
    assert!((out[1] - 0.3).abs() < 1e-6);
    assert!((out[2] - 0.7).abs() < 1e-6);
    assert!((out[3] - 0.9).abs() < 1e-6);
}

#[test]
fn vitals_pack_divides_hp_by_max_hp() {
    let mut state = SimState::new(2, 0);
    let a = state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::ZERO,
            hp: 100.0,
            max_hp: 100.0,
        })
        .unwrap();
    // Damage to 25 — hp_frac should be 0.25.
    state.set_agent_hp(a, 25.0);

    let mut out = [0.0f32; 4];
    VitalsSource.pack(&state, a, &mut out);
    assert!((out[0] - 0.25).abs() < 1e-6, "hp_frac = {}", out[0]);
}
