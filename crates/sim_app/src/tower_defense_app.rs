//! tower_defense harness — drives `tower_defense_runtime` for up to
//! MAX_TICKS ticks. Spawns 10 escalating waves of enemies; defenders
//! pick the closest enemy each tick and shoot. The base loses HP each
//! time an enemy crosses the melee threshold. Win = all waves cleared
//! with base HP > 0; lose = base HP <= 0.
//!
//! Per-50-tick trace: [tick, wave, enemies_alive, defender_total_dmg,
//! base_hp, base_damage_taken_total].

use engine::CompiledSim;
use tower_defense_runtime::{
    TowerDefenseState, BASE_SLOT, DEFENDER_SLOT_START, DEFENDER_COUNT,
    ENEMY_SLOT_START, ENEMY_SLOT_COUNT, WAVE_COUNT,
};

const SEED: u64 = 0xDEADBEEF_CAFE_F00D;
const MAX_TICKS: u64 = 1500;
const TRACE_EVERY: u64 = 50;

fn main() {
    let mut sim = TowerDefenseState::new(SEED);
    println!("================================================================");
    println!(" TOWER DEFENSE — 10 defenders + 1 base vs {} waves", WAVE_COUNT);
    println!("   seed=0x{:016X}  agents={}  max_ticks={}",
        SEED, sim.agent_count(), MAX_TICKS);
    println!("   slots: base=[{}], defenders=[{}..={}], enemies=[{}..={}]",
        BASE_SLOT,
        DEFENDER_SLOT_START, DEFENDER_SLOT_START + DEFENDER_COUNT - 1,
        ENEMY_SLOT_START, ENEMY_SLOT_START + ENEMY_SLOT_COUNT - 1);
    println!("================================================================");
    println!("{:>5} {:>5} {:>9} {:>10} {:>9} {:>10}",
        "tick", "wave", "alive_enm", "def_dmg", "base_hp", "base_taken");
    println!("{}", "-".repeat(64));

    let mut last_outcome = "running";
    let mut ended_at: Option<u64> = None;

    // Tick 0 trace + initial wave spawn (tick 0 fires the first wave).
    let spawned = sim.maybe_spawn_wave();
    let dd: f32 = sim.defender_damage_dealt().iter().sum();
    let bd_total = sim.base_damage_taken_total();
    println!(
        "{:>5} {:>5} {:>9} {:>10.1} {:>9.1} {:>10.1}  (wave 1 spawn: {} enemies)",
        0, sim.next_wave_idx, sim.alive_enemy_count(), dd,
        sim.base_hp, bd_total, spawned,
    );

    for tick in 1..=MAX_TICKS {
        sim.step();
        sim.sync_base_hp();

        // Try to fire the next wave (no-op if not due).
        let _ = sim.maybe_spawn_wave();

        if tick % TRACE_EVERY == 0 {
            let dd: f32 = sim.defender_damage_dealt().iter().sum();
            let bd_total = sim.base_damage_taken_total();
            println!(
                "{:>5} {:>5} {:>9} {:>10.1} {:>9.1} {:>10.1}",
                tick, sim.next_wave_idx,
                sim.alive_enemy_count(), dd,
                sim.base_hp, bd_total,
            );
        }

        // Loss condition: base HP dropped to zero or below.
        if sim.base_hp <= 0.0 {
            ended_at = Some(tick);
            last_outcome = "DEFENDERS LOSE — base destroyed";
            break;
        }

        // Win condition: all waves spawned AND no enemies remain alive.
        if sim.next_wave_idx >= WAVE_COUNT && sim.alive_enemy_count() == 0 {
            ended_at = Some(tick);
            last_outcome = "DEFENDERS WIN — all waves repelled";
            break;
        }
    }

    let final_dd: f32 = sim.defender_damage_dealt().iter().sum();
    let final_bd = sim.base_damage_taken_total();
    let alive_def = sim.alive_defender_count();
    let alive_enm = sim.alive_enemy_count();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Outcome:               {}", last_outcome);
    if let Some(t) = ended_at {
        println!("  Ended at tick:         {}", t);
    } else {
        println!("  Ran to MAX_TICKS={} without resolution", MAX_TICKS);
    }
    println!("  Waves spawned:         {} / {}", sim.next_wave_idx, WAVE_COUNT);
    println!("  Total enemies spawned: {}", sim.total_enemies_spawned);
    // Approximate kills via cumulative defender damage / enemy HP. The
    // per-enemy spawn HP is 20.0, so dd / 20 ≈ enemies-killed (assumes
    // overkill per enemy is small).
    let est_kills = (final_dd / 20.0_f32).floor() as u32;
    println!("  Defender total damage: {:.1}  (≈ {} kills @ 20 HP each)",
        final_dd, est_kills);
    println!("  Base damage taken:     {:.1}  (base_hp = {:.1})",
        final_bd, sim.base_hp);
    println!("  Defenders alive:       {} / {}", alive_def, DEFENDER_COUNT);
    println!("  Enemies alive at end:  {}", alive_enm);

    // Per-defender breakdown.
    let dd_slice = sim.defender_damage_dealt().to_vec();
    println!();
    println!("  Per-defender damage:");
    for i in 0..DEFENDER_COUNT {
        let slot = (DEFENDER_SLOT_START + i) as usize;
        if slot < dd_slice.len() && dd_slice[slot] > 0.0 {
            println!("    defender[{:>2}]: {:>7.1}", slot, dd_slice[slot]);
        }
    }

    // Hard asserts: at least one defender shot something, at least one
    // wave fully spawned. Either failure indicates the verb cascade or
    // wave-spawn pipe broke.
    assert!(
        final_dd > 0.0,
        "tower_defense_app: ASSERTION FAILED — no defender damage was \
         dealt over {} ticks. The Shoot verb cascade did not reach the \
         ApplyDamage chronicle.",
        MAX_TICKS,
    );
    assert!(
        sim.next_wave_idx >= 1,
        "tower_defense_app: ASSERTION FAILED — no wave fired in {} ticks.",
        MAX_TICKS,
    );
}
