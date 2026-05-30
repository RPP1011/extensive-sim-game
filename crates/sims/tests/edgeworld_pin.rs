//! edgeworld Phase 0 behavioral pin. Phase 0 = hunger + food +
//! forage/eat/starve/regrow + seek-food movement. Shared constants and
//! GPU readback helpers live in the sibling `edgeworld_common` module so
//! the (future Task 6) render test can reuse them.

use sims::edgeworld::GeneratedRuntime;

mod edgeworld_common;
use edgeworld_common::*;

const SEED: u64 = 0xED6E_0001;
const N_TOTAL: u32 = 4;

#[test]
fn edgeworld_runtime_constructs() {
    let state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter on host.");
            return;
        }
    };
    // Constructing + dropping the runtime is the Task 1 assertion:
    // the fixture compiled and the GPU pipeline built.
    drop(state);
}

#[test]
fn edgeworld_hunger_rises() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter.");
            return;
        }
    };
    let n = N_TOTAL as usize;
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&vec![CT_SURVIVOR; n]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&vec![0.0f32; n]));
    for _ in 0..10 {
        state.step();
    }
    let hunger = read_hunger(&mut state, n);
    println!("[edgeworld] hunger after 10 ticks: {hunger:?}");
    assert!(
        hunger[0] > 0.4 && hunger[0] < 0.6,
        "hunger should be ~0.5 after 10 ticks at 0.05/tick, got {}",
        hunger[0]
    );
}

#[test]
fn edgeworld_starvation_kills() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter.");
            return;
        }
    };
    let n = N_TOTAL as usize;
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&vec![CT_SURVIVOR; n]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&vec![0.0f32; n]));
    state.gpu.queue.write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&vec![1.0f32; n])); // for fallback path
    // No food → everyone starves. 20 ticks to threshold; run 30.
    for _ in 0..30 {
        state.step();
    }
    let alive = read_alive(&mut state, n);
    let n_alive: u32 = alive.iter().sum();
    println!("[edgeworld] survivors alive after 30 starving ticks: {n_alive}");
    assert_eq!(n_alive, 0, "all survivors should have starved with no food");
}

#[test]
fn edgeworld_eating_feeds_and_depletes() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skip: no adapter.");
            return;
        }
    };
    // slot 0 = FoodNode (type 0), slot 1 = Survivor (type 1), co-located within eat_radius.
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[CT_FOOD, CT_SURVIVOR]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[0.5,0.0,0.0,0.0]]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.6f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[5.0f32, 0.0f32]));
    for _ in 0..3 {
        state.step();
    }
    let hunger = read_hunger(&mut state, 2);
    let mana = read_mana(&mut state, 2);
    println!("[edgeworld] survivor hunger={} node quantity={}", hunger[1], mana[0]);
    assert!(hunger[1] < 0.6, "survivor should have eaten and lowered hunger, got {}", hunger[1]);
    assert!(mana[0] < 5.0, "food node should have been depleted by eating, got {}", mana[0]);
}

#[test]
fn edgeworld_seekfood_moves_toward_food() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skip: no adapter.");
            return;
        }
    };
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[CT_FOOD, CT_SURVIVOR]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[8.0,0.0,0.0,0.0]])); // food at origin, survivor 8 away
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.5f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[5.0f32, 0.0f32]));
    let start = read_positions(&mut state, 2)[1][0];
    for _ in 0..10 { state.step(); }
    let end = read_positions(&mut state, 2)[1][0];
    println!("[edgeworld] survivor x: {start} -> {end}");
    assert!(end < start - 1.0, "hungry survivor should move toward food (x decreasing), {start}->{end}");
}
