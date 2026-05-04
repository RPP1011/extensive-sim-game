//! Scripted battle harness — drives `scripted_battle_runtime` for the
//! full 800-tick narrative arc:
//!
//!   - Phase 1 (ticks 0-199):   Peaceful — villagers gather food
//!                              (passive +1 hp/tick self-heal). No
//!                              combat verbs fire.
//!   - Phase 2 (ticks 200-499): Assault — host wakes every enemy slot
//!                              at tick 200; villagers and enemies
//!                              trade blows on alternating ticks.
//!   - Phase 3 (ticks 500-799): Aftermath — host mass-clears every
//!                              enemy slot at tick 500; surviving
//!                              villagers passively heal.
//!
//! Reports per-50-tick traces (current phase, alive villagers, alive
//! enemies, total damage dealt, total healing done) and verb-fire
//! counts per phase (so the phase-gating is observable as a step
//! function in the per-verb activity).

use engine::CompiledSim;
use scripted_battle_runtime::{ScriptedBattleState, PHASE1_END, PHASE2_END, PHASE3_END};

const SEED: u64 = 0xCAFE_DEAD_BEEF_F00D;
const VILLAGER_COUNT: u32 = 20;
const ENEMY_COUNT: u32 = 50;
const MAX_TICKS: u64 = PHASE3_END; // 800
const TRACE_INTERVAL_TICKS: u64 = 50;

fn main() {
    let mut sim = ScriptedBattleState::new(SEED, VILLAGER_COUNT, ENEMY_COUNT);
    println!("================================================================");
    println!(" SCRIPTED BATTLE — narrative arc (peaceful → assault → aftermath)");
    println!("   seed=0x{:016X} villagers={} enemy_slots={}",
        SEED, VILLAGER_COUNT, ENEMY_COUNT);
    println!("   phases: P1=[0..{}) P2=[{}..{}) P3=[{}..{})",
        PHASE1_END, PHASE1_END, PHASE2_END, PHASE2_END, PHASE3_END);
    println!("================================================================");

    // Initial baseline.
    log_trace(&sim, &mut Vec::new());

    // Per-phase verb-fire counters. We track these by deltas to the
    // per-source view aggregates — `damage_dealt` and `healing_done`
    // are monotonically non-decreasing f32 accumulators (clamped at
    // [0, 1e6] in the .sim) so per-phase deltas equal "damage/heal
    // emitted during that phase".
    let mut phase_damage = [0.0f32; 4]; // [_, P1, P2, P3]
    let mut phase_heal = [0.0f32; 4];
    let mut prev_total_damage = 0.0f32;
    let mut prev_total_heal = 0.0f32;
    let mut phase_changes: Vec<(u64, u8)> = vec![(0, 1)];
    let mut last_phase = 1u8;

    let mut history: Vec<(u64, u8, u32, u32, f32, f32)> = Vec::new();

    for _tick in 1..=MAX_TICKS {
        // The step that's about to run uses `world.tick = sim.tick()`
        // for its mask predicates. Attribute the damage / healing
        // deltas it emits to THAT tick's phase, not to the tick the
        // step lands on (which is one higher and may be in the next
        // phase).
        let active_tick = sim.tick();
        let active_phase = phase_for(active_tick);

        sim.step();
        let cur_tick = sim.tick();
        let cur_phase = sim.phase();

        if cur_phase != last_phase {
            phase_changes.push((cur_tick, cur_phase));
            last_phase = cur_phase;
        }

        // Per-tick: snap totals, attribute deltas to ACTIVE phase
        // (the phase the step just ran in, not the one we land in).
        let total_damage: f32 = sim.damage_dealt().iter().sum();
        let total_heal: f32 = sim.healing_done().iter().sum();
        let d_damage = (total_damage - prev_total_damage).max(0.0);
        let d_heal = (total_heal - prev_total_heal).max(0.0);
        phase_damage[active_phase as usize] += d_damage;
        phase_heal[active_phase as usize] += d_heal;
        prev_total_damage = total_damage;
        prev_total_heal = total_heal;

        if cur_tick % TRACE_INTERVAL_TICKS == 0 || cur_tick == PHASE1_END
            || cur_tick == PHASE2_END
        {
            log_trace(&sim, &mut history);
        }
    }

    // -- Final report --
    let final_alive = sim.read_alive();
    let final_hp = sim.read_hp();
    let villagers_alive = (0..VILLAGER_COUNT as usize)
        .filter(|i| final_alive[*i] == 1)
        .count();
    let villagers_lost = (VILLAGER_COUNT as usize) - villagers_alive;
    let enemies_alive = (VILLAGER_COUNT as usize..final_alive.len())
        .filter(|i| final_alive[*i] == 1)
        .count();
    let enemies_repelled = ENEMY_COUNT as usize - enemies_alive;

    let total_damage: f32 = sim.damage_dealt().iter().sum();
    let total_heal: f32 = sim.healing_done().iter().sum();

    println!();
    println!("================================================================");
    println!(" PHASE TRANSITIONS");
    println!("================================================================");
    for (tick, ph) in &phase_changes {
        let label = match ph {
            1 => "Peaceful",
            2 => "Assault",
            3 => "Aftermath",
            _ => "?",
        };
        println!("  tick {:>4} → phase {} ({})", tick, ph, label);
    }

    println!();
    println!("================================================================");
    println!(" PER-PHASE VERB ACTIVITY (deltas to view aggregates)");
    println!("================================================================");
    println!("  Phase 1 (Peaceful):   damage emitted = {:.1}, healing emitted = {:.1}",
        phase_damage[1], phase_heal[1]);
    println!("  Phase 2 (Assault):    damage emitted = {:.1}, healing emitted = {:.1}",
        phase_damage[2], phase_heal[2]);
    println!("  Phase 3 (Aftermath):  damage emitted = {:.1}, healing emitted = {:.1}",
        phase_damage[3], phase_heal[3]);

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Villagers alive:   {} / {} (lost {})",
        villagers_alive, VILLAGER_COUNT, villagers_lost);
    println!("  Enemies alive:     {} / {} (repelled {})",
        enemies_alive, ENEMY_COUNT, enemies_repelled);
    println!("  Total damage:      {:.1}", total_damage);
    println!("  Total healing:     {:.1}", total_heal);

    // Per-villager surviving HP summary.
    let mut sum_hp = 0.0f32;
    let mut min_hp = f32::INFINITY;
    let mut max_hp: f32 = 0.0;
    let mut surv = 0;
    for i in 0..VILLAGER_COUNT as usize {
        if final_alive[i] == 1 {
            sum_hp += final_hp[i];
            min_hp = min_hp.min(final_hp[i]);
            max_hp = max_hp.max(final_hp[i]);
            surv += 1;
        }
    }
    if surv > 0 {
        println!(
            "  Villager HP range: min={:.1}, max={:.1}, avg={:.1}",
            min_hp, max_hp, sum_hp / (surv as f32),
        );
    }

    println!();
    println!("================================================================");
    println!(" NARRATIVE OUTCOME");
    println!("================================================================");
    if villagers_alive == 0 {
        println!("  TRAGEDY — every villager fell during the assault. The settlement is silent.");
    } else if enemies_repelled == ENEMY_COUNT as usize {
        println!("  VICTORY — {} of {} villagers survived the assault. All {} enemies repelled.",
            villagers_alive, VILLAGER_COUNT, enemies_repelled);
    } else {
        println!("  PYRRHIC — {} of {} villagers survived. {} of {} enemies repelled.",
            villagers_alive, VILLAGER_COUNT, enemies_repelled, ENEMY_COUNT);
    }

    // -- Hard asserts (sim_app convention): the phase transitions
    // must be observable as activity step-functions on the view
    // aggregates. Otherwise the GPU isn't actually phase-gating the
    // verb cascade and the run is just a smoke test.
    assert!(
        phase_damage[1] < 1.0,
        "scripted_battle_app: ASSERTION FAILED — phase 1 emitted {:.1} damage; \
         the peaceful phase should not run combat verbs.",
        phase_damage[1],
    );
    assert!(
        phase_damage[2] > 1.0,
        "scripted_battle_app: ASSERTION FAILED — phase 2 emitted {:.1} damage; \
         the assault phase should run combat verbs.",
        phase_damage[2],
    );
    assert!(
        phase_damage[3] < 1.0,
        "scripted_battle_app: ASSERTION FAILED — phase 3 emitted {:.1} damage; \
         the aftermath phase should not run combat verbs.",
        phase_damage[3],
    );
    assert!(
        phase_heal[1] > 1.0,
        "scripted_battle_app: ASSERTION FAILED — phase 1 emitted {:.1} healing; \
         peaceful villagers should be passively healing.",
        phase_heal[1],
    );
    assert!(
        phase_heal[3] > 1.0 || villagers_alive == 0,
        "scripted_battle_app: ASSERTION FAILED — phase 3 emitted {:.1} healing \
         while {} villagers survived; aftermath should let survivors recover.",
        phase_heal[3], villagers_alive,
    );
}

/// Map a `world.tick` value to its narrative phase (matches the
/// `when` predicates in `assets/sim/scripted_battle.sim`).
fn phase_for(tick: u64) -> u8 {
    if tick < PHASE1_END { 1 }
    else if tick < PHASE2_END { 2 }
    else { 3 }
}

/// Print one trace line for the current sim tick. Also pushes the
/// snapshot onto `history` so callers can post-process if desired.
fn log_trace(
    sim: &ScriptedBattleState,
    history: &mut Vec<(u64, u8, u32, u32, f32, f32)>,
) {
    let alive = sim.read_alive();
    let villager_count = sim.villager_count() as usize;
    let villagers_alive = (0..villager_count)
        .filter(|i| alive[*i] == 1)
        .count() as u32;
    let enemies_alive = (villager_count..alive.len())
        .filter(|i| alive[*i] == 1)
        .count() as u32;

    // Note: `damage_dealt()` and `healing_done()` mutate (cache-flip
    // readback). They're called in the run loop too — log_trace is
    // called only from main where we have an immutable borrow, so
    // we read view aggregates via the buffered HP signal here.
    let hp = sim.read_hp();
    let total_villager_hp: f32 = (0..villager_count)
        .filter(|i| alive[*i] == 1)
        .map(|i| hp[i])
        .sum();
    let total_enemy_hp: f32 = (villager_count..hp.len())
        .filter(|i| alive[*i] == 1)
        .map(|i| hp[i])
        .sum();
    let phase = sim.phase();
    let label = match phase {
        1 => "P1 Peaceful",
        2 => "P2 Assault ",
        3 => "P3 Aftermath",
        _ => "??",
    };

    println!(
        "  tick {:>4} [{}]: villagers_alive={:>3}/{:<3} enemies_alive={:>3}/{:<3} \
         hp_total v={:>6.1} e={:>6.1}",
        sim.tick(), label,
        villagers_alive, sim.villager_count(),
        enemies_alive, sim.enemy_count(),
        total_villager_hp, total_enemy_hp,
    );
    history.push((
        sim.tick(), phase, villagers_alive, enemies_alive,
        total_villager_hp, total_enemy_hp,
    ));
}
