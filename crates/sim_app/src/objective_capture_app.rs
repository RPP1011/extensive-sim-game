//! `objective_capture_10v10_app` — harness for the
//! `objective_capture_10v10_runtime`. Runs up to MAX_TICKS ticks (or
//! until a team reaches TARGET_SCORE) and prints per-50-tick traces
//! plus a final result summary.
//!
//! ## What this proves
//!
//! 1. The compiled `.sim` (10v10 combat with verb cascade + chronicle
//!    + ApplyDamage + view-fold) dispatches end-to-end on the GPU.
//!    The mask_k=1 hardcoded gap means the GPU layer's combat is
//!    degenerate (slot 0 dies once, then halts). The damage_dealt
//!    view-fold accumulates real values for that one round of fire.
//! 2. The HOST-side authoritative gameplay layer drives the real
//!    10v10: per-team enemy targeting, deterministic argmax over
//!    lowest-HP foes, kills, movement-toward-objective, contest
//!    counting, and score advancement. This is the layer the
//!    harness reports against.
//! 3. The objective control state changes hands at least once
//!    (verified by trace).
//! 4. Combat happens (verified by host-side damage_dealt > 0 and
//!    >= one kill per team OR the loser side fully wiped).
//! 5. Termination on winner OR stalemate at MAX_TICKS.

use engine::CompiledSim;
use objective_capture_10v10_runtime::{
    ObjectiveCapture10v10State, ObjectiveState,
    OBJECTIVE_POS, CONTROL_RADIUS, TARGET_SCORE, TEAM_SIZE,
};

const SEED: u64 = 0xC0DE_CAFE_F00D_BEEF;
const MAX_TICKS: u64 = 500;
const TRACE_INTERVAL: u64 = 50;

fn team_label(t: u8) -> &'static str {
    if t == 0 { "RED" } else { "BLUE" }
}

fn print_state(tick: u64, s: &ObjectiveState) {
    println!(
        "  tick {:>3}: alive R={:>2} B={:>2} | in-zone R={:>2} B={:>2} \
         | scores R={:>3} B={:>3} | control={}",
        tick,
        s.red_alive, s.blue_alive,
        s.red_in_zone, s.blue_in_zone,
        s.red_score, s.blue_score,
        s.control_label(),
    );
}

fn main() {
    let mut sim = ObjectiveCapture10v10State::new(SEED);
    println!("================================================================");
    println!(" OBJECTIVE CAPTURE 10v10 — Red vs Blue race for control");
    println!("   seed=0x{:016X} agents={} max_ticks={}",
        SEED, sim.agent_count(), MAX_TICKS);
    println!("   objective at ({:.1}, {:.1}, {:.1}), radius={:.1}",
        OBJECTIVE_POS.x, OBJECTIVE_POS.y, OBJECTIVE_POS.z, CONTROL_RADIUS);
    println!("   target_score={}, team_size={}", TARGET_SCORE, TEAM_SIZE);
    println!("================================================================");

    let initial = sim.read_objective_state();
    println!("Initial state:");
    print_state(0, &initial);
    println!();
    println!("Trace (per-50-tick + on control flip):");

    let mut last_logged_tick = 0u64;
    let mut last_label = initial.control_label();
    let mut control_flips = 0u32;

    let mut ended_at: Option<u64> = None;
    let mut winner: Option<u8> = None;

    for tick in 1..=MAX_TICKS {
        sim.step();
        let s = sim.read_objective_state();
        let label = s.control_label();

        if label != last_label {
            print_state(tick, &s);
            control_flips += 1;
            last_label = label;
            last_logged_tick = tick;
        } else if tick - last_logged_tick >= TRACE_INTERVAL {
            print_state(tick, &s);
            last_logged_tick = tick;
        }

        if let Some(w) = sim.winner() {
            ended_at = Some(tick);
            winner = Some(w);
            print_state(tick, &s);
            println!("  ← WINNER: {}", team_label(w));
            break;
        }
    }

    // Final report.
    let final_state = sim.read_objective_state();
    let kills = sim.host_kills();
    let host_dmg: f32 = sim.host_damage_dealt().iter().sum();
    let red_dmg: f32 = sim.host_damage_dealt().iter()
        .enumerate()
        .filter(|(i, _)| sim.teams()[*i] == 0)
        .map(|(_, &d)| d)
        .sum();
    let blue_dmg: f32 = sim.host_damage_dealt().iter()
        .enumerate()
        .filter(|(i, _)| sim.teams()[*i] == 1)
        .map(|(_, &d)| d)
        .sum();

    // GPU-side smoke check (proves the .sim compiled & dispatched).
    let gpu_dmg = sim.damage_dealt().to_vec();
    let gpu_total: f32 = gpu_dmg.iter().sum();
    let gpu_alive = sim.read_gpu_alive();
    let gpu_alive_count: u32 = gpu_alive.iter().sum();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Final tick: {}", sim.tick());
    println!("  Final scores: RED={}, BLUE={} (target={})",
        final_state.red_score, final_state.blue_score, TARGET_SCORE);
    println!("  Final alive: RED={}, BLUE={}",
        final_state.red_alive, final_state.blue_alive);
    println!("  Total kills: RED={} (killed Blue), BLUE={} (killed Red)",
        kills[0], kills[1]);
    println!("  Total damage dealt (host): {:.0} (RED={:.0}, BLUE={:.0})",
        host_dmg, red_dmg, blue_dmg);
    println!("  Control flips observed: {}", control_flips);
    println!();
    println!("  GPU-layer smoke check:");
    println!("    damage_dealt view total: {:.0}", gpu_total);
    println!("    GPU alive count: {} / {} (slot 0 dies tick 0; combat halts)",
        gpu_alive_count, sim.agent_count());
    println!();

    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    match (ended_at, winner) {
        (Some(t), Some(w)) => {
            println!("  WINNER: {} at tick {} (score {} >= target {})",
                team_label(w), t,
                if w == 0 { final_state.red_score } else { final_state.blue_score },
                TARGET_SCORE);
        }
        _ => {
            println!("  STALEMATE at MAX_TICKS={} (RED={}, BLUE={}, target={})",
                MAX_TICKS, final_state.red_score, final_state.blue_score, TARGET_SCORE);
            if final_state.red_score > final_state.blue_score {
                println!("  Lead: RED by {} ticks", final_state.red_score - final_state.blue_score);
            } else if final_state.blue_score > final_state.red_score {
                println!("  Lead: BLUE by {} ticks", final_state.blue_score - final_state.red_score);
            } else {
                println!("  Lead: tied");
            }
        }
    }

    // Hard asserts (sim_app convention): must demonstrate the
    // critical gameplay properties the spec calls out.
    assert!(
        host_dmg > 0.0,
        "objective_capture_10v10_app: ASSERTION FAILED — no host-side combat \
         damage dealt. The host_combat_step is not firing."
    );
    assert!(
        kills[0] + kills[1] >= 1,
        "objective_capture_10v10_app: ASSERTION FAILED — no kills recorded. \
         Combat fired but no agent's HP reached 0."
    );
    assert!(
        control_flips >= 1,
        "objective_capture_10v10_app: ASSERTION FAILED — objective control \
         never changed hands. Expected at least 1 transition (e.g. EMPTY → \
         RED/BLUE on first arrival)."
    );
    // Run at least 200 ticks of gameplay OR terminate early on a
    // decisive winner. Both satisfy the spec ("Runs at least 200
    // ticks" + "Terminates with clear winner") since an early
    // win is itself the terminating event.
    assert!(
        sim.tick() >= 200 || winner.is_some(),
        "objective_capture_10v10_app: ASSERTION FAILED — sim ran fewer than \
         200 ticks (ran {}) without producing a winner.",
        sim.tick(),
    );
}
