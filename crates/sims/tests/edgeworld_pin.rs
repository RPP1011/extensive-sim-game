//! edgeworld Phase 0 behavioral pin. Phase 0 = hunger + food +
//! forage/eat/starve/regrow + seek-food movement. Shared constants and
//! GPU readback helpers live in the sibling `edgeworld_common` module so
//! the (future Task 6) render test can reuse them.

use sims::edgeworld::GeneratedRuntime;

mod edgeworld_common;
use edgeworld_common::*;

const SEED: u64 = 0xED6E_0001;
const N_TOTAL: u32 = 4;

// Boom/bust scenario sizing. Overshoot: seed more survivors than the
// (scarce, slow-regrowing) food can sustain → strip → cull → remnant.
const N_SURV: usize = 28;
const N_FOODN: usize = 3;
const N_SCEN: u32 = (N_SURV + N_FOODN) as u32;

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

#[test]
fn edgeworld_boom_then_bust_then_remnant() {
    let mut state = match GeneratedRuntime::try_new(0xED6E_0001, N_SCEN) {
        Some(s) => s, None => { eprintln!("[edgeworld] skip: no adapter."); return; }
    };
    seed_world(&mut state, N_SURV, N_FOODN, 0, 8.0);
    let mut min_alive = u32::MAX;
    let mut max_alive = 0u32;
    let mut samples = Vec::new();
    for tick in 0..600 {
        if tick % 20 == 0 {
            let alive = read_alive(&mut state, N_SCEN as usize);
            let types = read_creature_types(&mut state, N_SCEN as usize);
            let a: u32 = (0..N_SCEN as usize).filter(|&i| alive[i]==1 && types[i]==CT_SURVIVOR).count() as u32;
            min_alive = min_alive.min(a); max_alive = max_alive.max(a); samples.push(a);
        }
        state.step();
    }
    let final_alive = *samples.last().unwrap();
    println!("[edgeworld] max={max_alive} min={min_alive} final={final_alive} trace={samples:?}");
    assert!(max_alive >= 6, "expected a sustained early population (boom/hold), got max {max_alive}");
    assert!(min_alive < max_alive, "expected a crash (min < max), got flat {min_alive}");
    assert!(final_alive >= 1, "expected a surviving remnant, got extinction");
}

// Wolf-presence smoke: with the Wolf entity added (creature_type 2),
// a mixed world of survivors + food + wolves seeds and steps cleanly and
// the three creature_types are distinguishable. With Task 2 (WolfHunt)
// live, wolves now pursue and kill survivors that fall inside their
// 6-unit perception ring, so the survivor count may drop — the
// assertions only require all three types to coexist and the predators +
// larder to persist (no starvation in this short window, all hunger
// seeded to 0).
#[test]
fn edgeworld_wolves_present() {
    const N_SURV: usize = 6;
    const N_FOOD: usize = 2;
    const N_WOLF: usize = 2;
    const N: u32 = (N_SURV + N_FOOD + N_WOLF) as u32; // 10

    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skip: no adapter.");
            return;
        }
    };
    seed_world(&mut state, N_SURV, N_FOOD, N_WOLF, 8.0);
    // seed_world applies a graded hunger ramp (up to 2.8) to survivors to
    // drive the boom/bust crash. This smoke test only checks that the
    // mixed world seeds + steps cleanly and the three types coexist, so
    // override all agent hunger to a benign 0.0 — nothing dies from
    // starvation in this short window. (Wolves may now kill survivors via
    // Task 2 WolfHunt, so the survivor count is allowed to drop.)
    state.gpu.queue.write_buffer(
        &state.agent_hunger_buf,
        0,
        bytemuck::cast_slice(&vec![0.0f32; N as usize]),
    );

    for _ in 0..5 {
        state.step();
    }

    let alive = read_alive(&mut state, N as usize);
    let types = read_creature_types(&mut state, N as usize);

    let count = |ct: u32| {
        (0..N as usize)
            .filter(|&i| alive[i] == 1 && types[i] == ct)
            .count()
    };
    let survivors = count(CT_SURVIVOR);
    let wolves = count(CT_WOLF);
    let food = count(CT_FOOD);
    println!("[edgeworld] survivors={survivors} wolves={wolves} food={food} types={types:?}");

    // Wolves are slow to starve and never get eaten; the larder never
    // starves. Survivors may be culled by the wolves, but at least one
    // should still stand after only 5 ticks.
    assert!(survivors >= 1, "some survivors should still be alive after 5 ticks, got {survivors}");
    assert_eq!(wolves, N_WOLF, "all wolves should be alive after 5 ticks");
    assert_eq!(food, N_FOOD, "all food nodes should be present after 5 ticks");

    // The three creature_types must be distinguishable.
    assert!(types.contains(&CT_FOOD) && types.contains(&CT_SURVIVOR) && types.contains(&CT_WOLF),
        "all three creature_types should be present, got {types:?}");
}

// Task 2 kill pin: a wolf adjacent to a survivor (within kill_range = 1.2)
// kills it and feeds (resets its own hunger to 0). Both seeded manually at
// LOW hunger so the dead survivor cannot be confused with a starvation
// death — Starvation only fires at hunger >= hunger_max (1.0), and these
// are seeded at 0.0. Slot layout: slot0 = Wolf (type 2), slot1 = Survivor
// (type 1), no food.
#[test]
fn edgeworld_wolf_kills_close_survivor() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skip: no adapter.");
            return;
        }
    };
    // slot0 = Wolf at origin, slot1 = Survivor 0.5 away (inside kill_range 1.2).
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[CT_WOLF, CT_SURVIVOR]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[0.5,0.0,0.0,0.0]]));
    // Seed both at zero hunger so the only death channel is a wolf kill.
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.0f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.0f32]));
    for _ in 0..3 {
        state.step();
    }
    let alive = read_alive(&mut state, 2);
    let hunger = read_hunger(&mut state, 2);
    println!("[edgeworld] kill: wolf_alive={} surv_alive={} wolf_hunger={}", alive[0], alive[1], hunger[0]);
    assert_eq!(alive[1], 0, "survivor within kill_range should be dead, got alive={}", alive[1]);
    assert_eq!(alive[0], 1, "wolf should still be alive after the kill, got alive={}", alive[0]);
    // The wolf fed on the kill → its own hunger reset to 0 (then WolfHunger
    // adds ~0.03/tick afterward; after the kill tick it stays near 0).
    assert!(hunger[0] < 0.1, "wolf should have fed (hunger near 0), got {}", hunger[0]);
}

// Task 2 pursuit pin: a wolf beyond kill_range but inside the 6-unit
// perception ring steers toward a survivor. Survivor seeded at zero hunger
// so it neither flees (Flee lands in Task 3) nor seeks food (SeekFood's
// hunger > 0.2 gate stays shut), keeping it stationary; the wolf should
// close the gap. Slot layout: slot0 = Wolf at origin, slot1 = Survivor at
// x = 4.0.
#[test]
fn edgeworld_wolf_pursues_distant_survivor() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skip: no adapter.");
            return;
        }
    };
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[CT_WOLF, CT_SURVIVOR]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[4.0,0.0,0.0,0.0]]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.0f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.0f32]));
    let start = read_positions(&mut state, 2);
    let wolf_x0 = start[0][0];
    let surv_x0 = start[1][0];
    for _ in 0..8 { state.step(); }
    let end = read_positions(&mut state, 2);
    let wolf_x1 = end[0][0];
    let surv_x1 = end[1][0];
    let d0 = (surv_x0 - wolf_x0).abs();
    let d1 = (surv_x1 - wolf_x1).abs();
    println!("[edgeworld] pursuit: wolf_x {wolf_x0}->{wolf_x1} surv_x {surv_x0}->{surv_x1} dist {d0}->{d1}");
    assert!(wolf_x1 > wolf_x0, "wolf should move toward survivor (x increasing), {wolf_x0}->{wolf_x1}");
    assert!(d1 < d0, "wolf-survivor distance should shrink, {d0}->{d1}");
}
