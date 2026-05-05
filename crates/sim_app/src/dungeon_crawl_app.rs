//! Dungeon-crawl harness — drives `dungeon_crawl_runtime` for up to
//! `MAX_TICKS` ticks. Five heroes traverse three sequential rooms,
//! with CPU-side encounter pacing spawning the next room's enemies
//! whenever the active room is cleared.
//!
//! Per-100-tick trace: current room, hero count + total HP, enemy
//! count + total HP. Room transitions and outcome (victory / defeat)
//! are also logged at the moment they happen.

use dungeon_crawl_runtime::{
    DungeonCrawlState, DungeonOutcome, ANCHOR_SLOT, HERO_COUNT, HERO_FIRST_SLOT, HERO_HP, ROOMS,
    TOTAL_SLOTS,
};
use engine::CompiledSim;

const SEED: u64 = 0x_D006E0_C2A_F11Eu64;
const MAX_TICKS: u64 = 3000;
const TRACE_EVERY: u64 = 100;

fn enemy_count_and_hp(alive: &[u32], hp: &[f32]) -> (u32, f32) {
    let mut count = 0u32;
    let mut total = 0.0f32;
    let enemy_first = HERO_FIRST_SLOT + HERO_COUNT;
    for slot in enemy_first..TOTAL_SLOTS {
        let s = slot as usize;
        if alive.get(s).copied() == Some(1) {
            count += 1;
            total += hp.get(s).copied().unwrap_or(0.0);
        }
    }
    (count, total)
}

fn hero_count_and_hp(alive: &[u32], hp: &[f32]) -> (u32, f32) {
    let mut count = 0u32;
    let mut total = 0.0f32;
    for h in 0..HERO_COUNT {
        let slot = HERO_FIRST_SLOT + h;
        let s = slot as usize;
        if alive.get(s).copied() == Some(1) {
            count += 1;
            total += hp.get(s).copied().unwrap_or(0.0);
        }
    }
    (count, total)
}

fn room_label(idx: Option<usize>) -> String {
    match idx {
        Some(i) => format!("Room {} ({})", i + 1, ROOMS[i].label),
        None => "<no room>".to_string(),
    }
}

fn trace(tick: u64, sim: &DungeonCrawlState) {
    let alive = sim.read_alive();
    let hp = sim.read_hp();
    let (h_n, h_hp) = hero_count_and_hp(&alive, &hp);
    let (e_n, e_hp) = enemy_count_and_hp(&alive, &hp);
    println!(
        "Tick {:>4}: {:<26} | heroes alive={}/{} hp_sum={:7.1} | enemies alive={:>2} hp_sum={:7.1}",
        tick, room_label(sim.current_room()), h_n, HERO_COUNT, h_hp, e_n, e_hp,
    );
}

fn main() {
    let mut sim = DungeonCrawlState::new(SEED);

    println!("================================================================");
    println!(" DUNGEON CRAWL — 5 heroes vs 3 rooms");
    println!("   seed=0x{:016X} max_ticks={}", SEED, MAX_TICKS);
    println!(
        "   total_slots={}, anchor=slot{}, hero_slots={}..{}, hero_hp={:.0}",
        TOTAL_SLOTS,
        ANCHOR_SLOT,
        HERO_FIRST_SLOT,
        HERO_FIRST_SLOT + HERO_COUNT,
        HERO_HP,
    );
    for (i, room) in ROOMS.iter().enumerate() {
        println!(
            "   {} = slots {}..{} ({}× HP={:.0})",
            room_label(Some(i)),
            room.start_slot,
            room.start_slot + room.count,
            room.count,
            room.hp,
        );
    }
    println!("================================================================");

    trace(0, &sim);
    let mut last_room_logged: Option<usize> = sim.current_room();

    let mut final_tick = 0u64;
    for tick in 1..=MAX_TICKS {
        sim.step();
        sim.advance_encounter();
        final_tick = tick;

        // Log at every 100-tick boundary.
        if tick % TRACE_EVERY == 0 {
            trace(tick, &sim);
        }

        // Log room transitions immediately.
        if sim.current_room() != last_room_logged {
            println!(
                ">>> Tick {}: room transition — now {}",
                tick,
                room_label(sim.current_room()),
            );
            trace(tick, &sim);
            last_room_logged = sim.current_room();
        }

        match sim.outcome() {
            DungeonOutcome::Victory => {
                println!(">>> Tick {}: VICTORY — all 3 rooms cleared!", tick);
                break;
            }
            DungeonOutcome::Defeat => {
                println!(">>> Tick {}: DEFEAT — all heroes are dead.", tick);
                break;
            }
            DungeonOutcome::InProgress => {}
        }
    }

    let alive = sim.read_alive();
    let hp = sim.read_hp();
    let (h_n, h_hp) = hero_count_and_hp(&alive, &hp);
    let (e_n, e_hp) = enemy_count_and_hp(&alive, &hp);
    let damage = sim.damage_dealt().to_vec();
    let healing = sim.healing_done().to_vec();
    let total_damage: f32 = damage.iter().sum();
    let total_healing: f32 = healing.iter().sum();

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Final tick:       {}", final_tick);
    println!("  Outcome:          {:?}", sim.outcome());
    println!("  Rooms cleared:    {} / {}", sim.rooms_cleared(), ROOMS.len());
    println!("  Heroes survived:  {}/{} (HP sum {:.1})", h_n, HERO_COUNT, h_hp);
    println!("  Enemies alive:    {} (HP sum {:.1})", e_n, e_hp);
    println!("  Total damage:     {:.1}", total_damage);
    println!("  Total healing:    {:.1}", total_healing);
    println!();
    println!("  Per-hero damage_dealt:");
    for h in 0..HERO_COUNT {
        let slot = HERO_FIRST_SLOT + h;
        let s = slot as usize;
        let st = if alive.get(s).copied() == Some(1) { "alive" } else { "dead " };
        println!(
            "    Hero {} (slot {}) [{}] hp={:6.2} dmg_dealt={:7.1} heal_done={:6.1}",
            h, slot, st, hp[s], damage.get(s).copied().unwrap_or(0.0),
            healing.get(s).copied().unwrap_or(0.0),
        );
    }

    assert!(
        matches!(sim.outcome(), DungeonOutcome::Victory | DungeonOutcome::Defeat),
        "dungeon_crawl_app: ASSERTION FAILED — sim ran to MAX_TICKS={} without \
         victory or defeat. rooms_cleared={}, heroes_alive={}, enemies_alive={}.",
        MAX_TICKS, sim.rooms_cleared(), h_n, e_n,
    );
    assert!(
        sim.rooms_cleared() >= 1,
        "dungeon_crawl_app: ASSERTION FAILED — no rooms cleared. The sim \
         did not progress past room 1.",
    );
    assert!(
        total_damage > 0.0,
        "dungeon_crawl_app: ASSERTION FAILED — no damage dealt all run. \
         The verb cascade did not reach the ApplyDamage kernel.",
    );
}
