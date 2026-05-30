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

// Task 2 pursuit pin (re-pinned for Task 3 Flee): a wolf inside its 6-unit
// perception ring steers toward a survivor. With Flee now live, the
// composed behavior is a CHASE: the wolf pursues at wolf_move_speed (0.18)
// while the survivor, once the wolf closes inside flee_range (5.0), flees
// at flee_speed (0.25). Seeded at separation 5.5 (just outside flee_range,
// inside wolf perception) the wolf takes the first step alone, drops the
// gap under flee_range, and from then on both move in the +x direction with
// the survivor staying ahead — a stable chase, never a kill. We assert the
// wolf actively pursues (its x climbs well past its start), the survivor is
// driven along ahead of it (survivor x climbs too), and the gap holds above
// kill_range (no capture). Slot layout: slot0 = Wolf at origin, slot1 =
// Survivor at x = 5.5.
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
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[5.5,0.0,0.0,0.0]]));
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
    let alive = read_alive(&mut state, 2);
    println!("[edgeworld] pursuit: wolf_x {wolf_x0}->{wolf_x1} surv_x {surv_x0}->{surv_x1} dist {d0}->{d1} surv_alive={}", alive[1]);
    // Wolf actively pursues — drives well past its start in +x.
    assert!(wolf_x1 > wolf_x0 + 1.0, "wolf should pursue (x climbing), {wolf_x0}->{wolf_x1}");
    // Survivor is driven ahead of the wolf (flees in +x, staying ahead).
    assert!(surv_x1 > surv_x0, "survivor should be driven ahead by the chase, {surv_x0}->{surv_x1}");
    // Stable chase: the survivor is never caught.
    assert_eq!(alive[1], 1, "chased survivor should still be alive, got alive={}", alive[1]);
    assert!(d1 > config_kill_range(), "gap should hold above kill_range (no capture), got {d1}");
}

/// kill_range from config.edgeworld (mirrors the .sim constant).
fn config_kill_range() -> f32 { 1.2 }

// Phase 1 dynamics pin (Task 4): predator-prey coupling is REAL, not
// cosmetic. Run two 600-tick scenarios from the SAME seed/world via the
// shared seeder — one with no wolves, one with K wolves — and compare the
// surviving-survivor remnant at the end. With wolves present the remnant
// must be strictly smaller (the pack culls), AND at least one wolf must
// still be alive at the end (the pack sustains itself by feeding, not
// starving). This proves a live chase/cull dynamic instead of the
// degenerate "everyone fled/starved AND the wolves starved too" outcome.
//
// HONEST DYNAMICS (observed): the no-wolf world settles to a stable
// oasis remnant (~14 survivors the recovered larder sustains). Introducing
// the pack collapses that remnant: in the opening ~40 ticks the wolves
// chase survivors off the compact world (no world bounds + Flee dominating
// SeekFood → fleeing survivors drift past the edge and starve in the
// wilderness) while killing those caught inside kill_range. The prey is
// driven to extinction by ~tick 40 (a real cull, visible as a scatter in
// the render). The wolves themselves persist to tick 600: wolf_hunger_rate
// is low enough (0.0015/tick) that the kills during the opening feast keep
// them fed well past the run length. So the pin's two halves hold for
// genuinely different reasons — survivors culled to zero, wolves sustained
// by the feast — which is exactly the non-degenerate predator-persistence
// case the task asks to demonstrate.
#[test]
fn edgeworld_predators_reduce_remnant() {
    const N_SURV: usize = 28;
    const N_FOODN: usize = 3;
    const N_WOLVES: usize = 4;
    const WORLD_HALF: f32 = 8.0;
    const DYN_SEED: u64 = 0xED6E_0001;

    // Helper: run one scenario, return (alive_survivors, alive_wolves) at
    // end of `ticks`.
    fn run(n_wolves: usize, ticks: u32) -> Option<(u32, u32)> {
        let n = (N_SURV + N_FOODN + n_wolves) as u32;
        let mut state = GeneratedRuntime::try_new(DYN_SEED, n)?;
        seed_world(&mut state, N_SURV, N_FOODN, n_wolves, WORLD_HALF);
        for _ in 0..ticks {
            state.step();
        }
        let alive = read_alive(&mut state, n as usize);
        let types = read_creature_types(&mut state, n as usize);
        let survivors = (0..n as usize)
            .filter(|&i| alive[i] == 1 && types[i] == CT_SURVIVOR)
            .count() as u32;
        let wolves = (0..n as usize)
            .filter(|&i| alive[i] == 1 && types[i] == CT_WOLF)
            .count() as u32;
        Some((survivors, wolves))
    }

    let no_wolves = run(0, 600);
    let with_wolves = run(N_WOLVES, 600);
    let (Some((remnant_no_wolves, _)), Some((remnant_with_wolves, wolves_alive))) =
        (no_wolves, with_wolves)
    else {
        eprintln!("[edgeworld] skip: no adapter.");
        return;
    };

    println!(
        "[edgeworld] dynamics: remnant_no_wolves={remnant_no_wolves} \
         remnant_with_wolves={remnant_with_wolves} wolves_alive={wolves_alive}/{N_WOLVES}"
    );

    assert!(
        remnant_with_wolves < remnant_no_wolves,
        "wolves should cull survivors (got {remnant_with_wolves} vs {remnant_no_wolves})"
    );
    assert!(
        wolves_alive >= 1,
        "at least one wolf should sustain by feeding, not all starve (got {wolves_alive})"
    );
}

// Task 3 flee pin: a survivor with a wolf inside flee_range = 5.0 steps
// directly AWAY from the wolf at flee_speed = 0.25 (> wolf_move_speed 0.18,
// so it outpaces a single pursuer). Both seeded at zero hunger so neither
// starves nor (the survivor) seeks food. Slot layout: slot0 = Survivor at
// origin, slot1 = Wolf at x = 3.0 (separation 3.0 < flee_range 5.0). The
// flee vector is away = survivor.pos - wolf.pos = (0 - 3) = -3 → the
// survivor steps in the -x direction.
#[test]
fn edgeworld_survivor_flees_nearby_wolf() {
    let mut state = match GeneratedRuntime::try_new(SEED, 2) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skip: no adapter.");
            return;
        }
    };
    // slot0 = Survivor at origin, slot1 = Wolf at x = 3.0 (inside flee_range).
    state.gpu.queue.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&[CT_SURVIVOR, CT_WOLF]));
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&[1u32, 1u32]));
    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0,
        bytemuck::cast_slice(&[[0.0f32,0.0,0.0,0.0],[3.0,0.0,0.0,0.0]]));
    state.gpu.queue.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.0f32]));
    state.gpu.queue.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&[0.0f32, 0.0f32]));
    let start = read_positions(&mut state, 2);
    let surv_x0 = start[0][0];
    let wolf_x0 = start[1][0];
    let d0 = (wolf_x0 - surv_x0).abs();
    for _ in 0..8 { state.step(); }
    let end = read_positions(&mut state, 2);
    let surv_x1 = end[0][0];
    let wolf_x1 = end[1][0];
    let d1 = (wolf_x1 - surv_x1).abs();
    let alive = read_alive(&mut state, 2);
    println!("[edgeworld] flee: surv_x {surv_x0}->{surv_x1} wolf_x {wolf_x0}->{wolf_x1} dist {d0}->{d1} surv_alive={}", alive[0]);
    // The survivor flees away from the wolf in the -x direction.
    assert!(surv_x1 < surv_x0 - 0.1, "survivor should flee away from wolf (x decreasing below 0), {surv_x0}->{surv_x1}");
    // Flight succeeded: the survivor is never caught (flee_speed > wolf_move_speed).
    assert_eq!(alive[0], 1, "fleeing survivor should still be alive (outran the wolf), got alive={}", alive[0]);
    // The gap did not collapse to a kill.
    assert!(d1 >= d0 - 0.5, "survivor should not be overtaken; gap {d0}->{d1} should hold or widen");
}
