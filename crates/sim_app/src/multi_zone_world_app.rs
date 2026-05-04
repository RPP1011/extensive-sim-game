//! Multi-zone world harness — drives `multi_zone_world_runtime` for
//! 1000 ticks. Reports per-100-tick zone-population trace, total
//! resources, and final stats.
//!
//! ## Predicted observable
//!
//! 30 Adventurers + 5 Monsters share an Agent SoA. Adventurers all
//! start in Forest (level=0) with hp=0 (gold), mana=0 (wood). Each
//! tick:
//!
//!   - Forest verb (gates `self.level == 0`): every 5 ticks emits
//!     WoodGathered → mana += 1
//!   - Town verb (`self.level == 1` + mana>=10): every 5 ticks emits
//!     TradeMade → mana -= 10, hp += 5
//!   - Dungeon verb (`self.level == 2` + target.level == 2 alive):
//!     every 3 ticks emits Damaged → target hp -= 12
//!
//! Migration (CPU-side, per-tick):
//!   - Forest + mana>=20 → Town
//!   - Town + mana<10 + hp>=100 → Dungeon
//!   - Dungeon + hp<50 → Forest
//!   - Monster killed → ALL dungeon adventurers retreat with -50 gold
//!
//! Steady state expectation: most adventurers cycle Forest → Town →
//! Dungeon → Forest. By tick 1000 the per-zone counts vary based on
//! tick-phase and migration races; we just demand at least one
//! adventurer touches each zone over the run.

use multi_zone_world_runtime::{
    MultiZoneWorldState, ADVENTURER_COUNT, MONSTER_COUNT, TOTAL_AGENTS,
    ZONE_DUNGEON, ZONE_FOREST, ZONE_TOWN,
};
use engine::CompiledSim;

const SEED: u64 = 0xA8FE_BEAD_15B0_55ED;
const TICKS: u64 = 1000;

fn main() {
    let mut sim = MultiZoneWorldState::new(SEED, TOTAL_AGENTS);
    println!("================================================================");
    println!(" MULTI-ZONE WORLD — Forest / Town / Dungeon (sim 19)");
    println!("   seed=0x{:016X} agents={} ({} Adventurers + {} Monsters)",
             SEED, TOTAL_AGENTS, ADVENTURER_COUNT, MONSTER_COUNT);
    println!("   ticks={}", TICKS);
    println!("================================================================");

    // Initial state.
    let init_zones = sim.zone_counts();
    println!(
        "Tick    0: zones=[forest={:>2} town={:>2} dungeon={:>2}]  \
         wood=0  gold=0  kills=0",
        init_zones[ZONE_FOREST as usize],
        init_zones[ZONE_TOWN as usize],
        init_zones[ZONE_DUNGEON as usize],
    );

    // Track which zones any adventurer ever entered.
    let mut zones_visited = [false; 3];
    zones_visited[ZONE_FOREST as usize] = true;

    for tick in 1..=TICKS {
        sim.step();

        // Update visited tracker from level_mirror.
        let z = sim.zone_counts();
        for (i, &c) in z.iter().enumerate() {
            if c > 0 {
                zones_visited[i] = true;
            }
        }

        if tick % 100 == 0 {
            let wood: f32 = sim.read_mana().iter().take(ADVENTURER_COUNT as usize).sum();
            let gold: f32 = sim.read_hp().iter().take(ADVENTURER_COUNT as usize).sum();
            let wood_total: f32 = sim.wood_gathered_total().iter().sum();
            let gold_total: f32 = sim.gold_earned_total().iter().sum();
            let dmg_total: f32 = sim.damage_dealt().iter().sum();
            println!(
                "Tick {:>4}: zones=[forest={:>2} town={:>2} dungeon={:>2}]  \
                 wood_inv={:>5.0}  gold_inv={:>5.0}  \
                 wood_total={:>6.0}  gold_total={:>5.0}  \
                 dmg={:>6.0}  kills={}  \
                 mig[F→T={} T→D={} D→F={}]",
                tick,
                z[ZONE_FOREST as usize],
                z[ZONE_TOWN as usize],
                z[ZONE_DUNGEON as usize],
                wood, gold,
                wood_total, gold_total, dmg_total,
                sim.kill_count,
                sim.forest_to_town_migrations,
                sim.town_to_dungeon_migrations,
                sim.dungeon_to_forest_migrations,
            );
        }
    }

    let final_zones = sim.zone_counts();
    let wood_inv: f32 = sim.read_mana().iter().take(ADVENTURER_COUNT as usize).sum();
    let gold_inv: f32 = sim.read_hp().iter().take(ADVENTURER_COUNT as usize).sum();
    let wood_total: f32 = sim.wood_gathered_total().iter().sum();
    let gold_total: f32 = sim.gold_earned_total().iter().sum();
    let dmg_total: f32 = sim.damage_dealt().iter().sum();

    println!();
    println!("================================================================");
    println!(" FINAL STATE (tick {})", sim.tick());
    println!("================================================================");
    println!("  Zone distribution: forest={} town={} dungeon={}",
             final_zones[ZONE_FOREST as usize],
             final_zones[ZONE_TOWN as usize],
             final_zones[ZONE_DUNGEON as usize]);
    println!("  Wood currently in inventory: {:.0}", wood_inv);
    println!("  Gold currently in inventory: {:.0}", gold_inv);
    println!("  Total wood ever gathered:    {:.0}", wood_total);
    println!("  Total gold ever earned:      {:.0}", gold_total);
    println!("  Total dungeon damage dealt:  {:.0}", dmg_total);
    println!("  Monsters killed:             {}", sim.kill_count);
    println!("  Monsters respawned:          {}", sim.monster_respawns);
    println!("  Migrations Forest → Town:    {}", sim.forest_to_town_migrations);
    println!("  Migrations Town → Dungeon:   {}", sim.town_to_dungeon_migrations);
    println!("  Migrations Dungeon → Forest: {}", sim.dungeon_to_forest_migrations);
    println!("  Zones any adventurer visited: forest={} town={} dungeon={}",
             zones_visited[0], zones_visited[1], zones_visited[2]);

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    let all_zones = zones_visited.iter().all(|&v| v);
    let economy_loop = wood_total > 0.0 && gold_total > 0.0;
    let dungeon_combat = dmg_total > 0.0;
    if all_zones && economy_loop && dungeon_combat {
        println!(
            "  (a) FULL FIRE — multi-zone world ran end-to-end. Adventurers \
             migrated through all three zones, the wood→gold→dungeon economy \
             closed (wood gathered={:.0}, gold earned={:.0}, dungeon damage \
             dealt={:.0}), and {} monsters were killed across the run. \
             First sim with per-agent location-based context (zone migration).",
             wood_total, gold_total, dmg_total, sim.kill_count,
        );
    } else if all_zones && economy_loop {
        println!(
            "  (a-partial) ZONES + ECONOMY OK, NO COMBAT — adventurers reached \
             all three zones and the wood→gold loop closed (wood={:.0}, \
             gold={:.0}), but no dungeon damage was dealt ({} kills). The \
             AttackMonster verb's pair-field gate (target.level==2) may not \
             be evaluating for monsters.",
             wood_total, gold_total, sim.kill_count,
        );
    } else if economy_loop {
        println!(
            "  (b) ECONOMY ONLY — wood→gold loop closed (wood={:.0}, gold={:.0}) \
             but adventurers didn't reach the Dungeon. Migration arithmetic \
             may be too restrictive for the given tick budget.",
             wood_total, gold_total,
        );
    } else if wood_total > 0.0 {
        println!(
            "  (b) GATHERING ONLY — wood gathering fires (total={:.0}) but no \
             trades happened ({:.0} gold earned). Either no adventurer crossed \
             the wood>=20 migration threshold, or the Town verb's mana>=10 \
             gate isn't firing.",
             wood_total, gold_total,
        );
    } else {
        println!(
            "  (b) NO ACTIVITY — neither wood gathered nor trades fired. The \
             zone-gated verb cascade isn't reaching the chronicle. Most likely \
             gap: `self.level == N` predicate doesn't lower correctly through \
             the mask emitter's PerPair scan.",
        );
    }

    // Hard assertions (sim_app convention).
    assert!(
        wood_total > 0.0,
        "multi_zone_world_app: ASSERTION FAILED — no wood gathered. The \
         GatherWood verb (gated on self.level == 0) did not fire. See OUTCOME above.",
    );
    assert!(
        gold_total > 0.0,
        "multi_zone_world_app: ASSERTION FAILED — no gold earned via trade. \
         Either the Forest→Town migration isn't happening (need wood>=20), \
         or the TradeWoodForGold verb (gated on self.level == 1) isn't firing.",
    );
    assert!(
        all_zones,
        "multi_zone_world_app: ASSERTION FAILED — at least one adventurer \
         must visit each of [Forest, Town, Dungeon]. zones_visited={:?}",
        zones_visited,
    );
}
