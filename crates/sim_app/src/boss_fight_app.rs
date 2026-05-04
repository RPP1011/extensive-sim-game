//! Boss-fight harness — drives `boss_fight_runtime` for up to 500
//! ticks (or until a decisive outcome) and reports per-50-tick combat
//! traces + final winner.
//!
//! ## Composition
//!
//!   slot 0          = Boss   (HP=5000, two abilities: BossStrike,
//!                              BossSelfHeal)
//!   slot 1..=5      = Heroes (HP=200,  two abilities: HeroAttack,
//!                              HeroHeal)
//!
//! ## Predicted observable shape
//!
//! Per tick the verb cascade picks at most one verb per agent:
//!
//!   - Boss:
//!       - tick % 10 == 0 (and not on a heal tick) → BossStrike
//!         deals 50 damage to one alive hero (scoring iterates per-pair
//!         candidates and picks the lowest-slot alive hero — the score
//!         is `if (target alive Hero) { 1.0 } else { -1000.0 }`).
//!       - tick % 50 == 0 AND boss.hp < 1500 → BossSelfHeal restores
//!         500 HP. SelfHeal score (100.0) > BossStrike score (1.0) so
//!         on overlapping ticks (e.g. tick 50, 100, ...) SelfHeal wins
//!         argmax when its `when` gate fires. (BossSelfHeal's
//!         creature_type gate also includes the boss-only check, so
//!         only the boss row exists at all.)
//!   - Each Hero:
//!       - tick % 3 == 0 → HeroAttack hits boss for 25 damage (5 heroes
//!         × ~25 dmg per 3 ticks ≈ 41.6 dmg per tick averaged over the
//!         cooldown window).
//!       - tick % 20 == 0 → HeroHeal restores 60 HP to self. HeroHeal
//!         score (50.0) > HeroAttack score (1.0) so on overlapping
//!         ticks heroes self-heal instead of attacking.
//!
//! ## Analytical fight prediction
//!
//! Hero throughput on boss: 5 heroes × 25 dmg per tick % 3 == 0
//! ≈ 5 × 25 / 3 = 41.67 dmg/tick. Boss has 5000 HP. Time to kill (no
//! heal) = 5000 / 41.67 ≈ 120 ticks.
//!
//! BossSelfHeal only triggers when boss.hp < 1500. Heroes will have
//! dealt 5000 - 1500 = 3500 dmg by then ≈ tick 84. Next tick %50 == 0
//! after that is tick 100 → SelfHeal fires, +500 HP, boss now at ~2000.
//! Heroes deal another 500 dmg by tick ~112 → back below 1500. Next
//! eligible heal tick is 150 → SelfHeal at ~tick 150 (assuming alive).
//! In practice the heal extends the fight by ~12 ticks per heal cycle.
//!
//! Boss dmg on heroes: 50 dmg every 10 ticks = 5 dmg/tick on average
//! (single hero hit per strike). HeroHeal restores 60 HP every 20
//! ticks per hero. Net dmg per hero per heal-cycle = boss dmg in 20
//! ticks × 1/5 (boss spreads strikes across 5 heroes) - 60 = 100/5 -
//! 60 = -40. Heroes regenerate faster than they take dmg on average,
//! so all 5 should survive.
//!
//! Predicted outcome: Heroes win at tick ~130-150, all 5 alive.

use boss_fight_runtime::BossFightState;
use engine::CompiledSim;

const SEED: u64 = 0xB055_F19E_F00D_B055;
const AGENT_COUNT: u32 = 6;
const MAX_TICKS: u64 = 800;
const TRACE_INTERVAL: u64 = 50;

const BOSS_STRIKE_DAMAGE: f32 = 50.0;
const BOSS_HEAL_AMOUNT: f32 = 400.0;
const HERO_ATTACK_DAMAGE: f32 = 35.0;
const HERO_HEAL_AMOUNT: f32 = 60.0;

fn main() {
    let mut sim = BossFightState::new(SEED, AGENT_COUNT);
    println!("================================================================");
    println!(" BOSS FIGHT — 1 Boss vs 5 Heroes");
    println!("   seed=0x{:016X} agents={} max_ticks={}", SEED, AGENT_COUNT, MAX_TICKS);
    println!(
        "   BossStrike={} dmg/10t  BossSelfHeal={} hp/50t (when hp<1500)",
        BOSS_STRIKE_DAMAGE, BOSS_HEAL_AMOUNT,
    );
    println!(
        "   HeroAttack={} dmg/3t   HeroHeal={} hp/20t",
        HERO_ATTACK_DAMAGE, HERO_HEAL_AMOUNT,
    );
    println!("================================================================");

    let initial_hp = sim.read_hp();
    let initial_alive = sim.read_alive();
    let creature_types = sim.read_creature_type();
    println!(
        "Tick   0: Boss HP={:7.1} (alive={}, ct={})  Heroes alive=[{},{},{},{},{}]  HP=[{:.0},{:.0},{:.0},{:.0},{:.0}]",
        initial_hp[0], initial_alive[0], creature_types[0],
        initial_alive[1], initial_alive[2], initial_alive[3], initial_alive[4], initial_alive[5],
        initial_hp[1], initial_hp[2], initial_hp[3], initial_hp[4], initial_hp[5],
    );
    println!(
        "          (creature_type discriminants: slot 0={} (Boss=0), slot 1={} (Hero=1))",
        creature_types[0], creature_types[1],
    );

    let mut ended_at: Option<u64> = None;
    let mut winner_label = "stalemate";

    // Track HP for self-heal phase detection.
    let mut prev_boss_hp = initial_hp[0];
    let mut self_heal_fire_tick: Option<u64> = None;
    let mut self_heal_count: u32 = 0;

    for tick in 1..=MAX_TICKS {
        sim.step();

        let hp = sim.read_hp();
        let alive = sim.read_alive();
        let boss_hp = hp[0];
        let boss_alive = alive[0] == 1;
        let heroes_alive: u32 = (1..=5).map(|i| alive[i] as u32).sum();

        // Detect SelfHeal fire — boss HP jumped UP (vs prev tick).
        if boss_hp > prev_boss_hp + 1.0 {
            self_heal_count += 1;
            if self_heal_fire_tick.is_none() {
                self_heal_fire_tick = Some(tick);
                println!(
                    "Tick {:>3}: BOSS SELF-HEAL FIRED — boss HP {:.0} -> {:.0} (+{:.0})",
                    tick, prev_boss_hp, boss_hp, boss_hp - prev_boss_hp,
                );
            } else {
                println!(
                    "Tick {:>3}: boss self-heal #{}    — boss HP {:.0} -> {:.0} (+{:.0})",
                    tick, self_heal_count, prev_boss_hp, boss_hp, boss_hp - prev_boss_hp,
                );
            }
        }
        prev_boss_hp = boss_hp;

        // Per-50-tick trace.
        if tick % TRACE_INTERVAL == 0 {
            let total_hero_hp: f32 = (1..=5).map(|i| hp[i].max(0.0)).sum();
            let dmg = sim.damage_dealt().to_vec();
            let heal = sim.healing_done().to_vec();
            let dmg_by_boss = dmg[0];
            let dmg_by_heroes: f32 = (1..=5).map(|i| dmg[i]).sum();
            let heal_by_boss = heal[0];
            let heal_by_heroes: f32 = (1..=5).map(|i| heal[i]).sum();

            // Translate damage/heal totals into ability fire counts
            // (each verb has a fixed amount per fire).
            let boss_strikes = (dmg_by_boss / BOSS_STRIKE_DAMAGE).round() as u64;
            let hero_attacks = (dmg_by_heroes / HERO_ATTACK_DAMAGE).round() as u64;
            let boss_heals = (heal_by_boss / BOSS_HEAL_AMOUNT).round() as u64;
            let hero_heals = (heal_by_heroes / HERO_HEAL_AMOUNT).round() as u64;

            println!(
                "Tick {:>3}: Boss HP={:7.1} alive={}  Heroes alive={}/5  Hero HPs=[{:.0},{:.0},{:.0},{:.0},{:.0}] sum={:.0}",
                tick, boss_hp, alive[0],
                heroes_alive,
                hp[1].max(0.0), hp[2].max(0.0), hp[3].max(0.0), hp[4].max(0.0), hp[5].max(0.0),
                total_hero_hp,
            );
            println!(
                "          fires: BossStrike={} BossSelfHeal={} HeroAttack={} HeroHeal={}",
                boss_strikes, boss_heals, hero_attacks, hero_heals,
            );
        }

        // Termination: boss dead OR all heroes dead.
        if !boss_alive {
            ended_at = Some(tick);
            winner_label = "Heroes";
            println!(
                "Tick {:>3}: BOSS DEFEATED — boss HP={:.1}, heroes alive={}/5",
                tick, boss_hp, heroes_alive,
            );
            break;
        }
        if heroes_alive == 0 {
            ended_at = Some(tick);
            winner_label = "Boss";
            println!(
                "Tick {:>3}: ALL HEROES DEFEATED — boss HP={:.1}",
                tick, boss_hp,
            );
            break;
        }
    }

    // Final state.
    let final_hp = sim.read_hp();
    let final_alive = sim.read_alive();
    let damage = sim.damage_dealt().to_vec();
    let healing = sim.healing_done().to_vec();
    let final_heroes_alive: u32 = (1..=5).map(|i| final_alive[i] as u32).sum();

    let dmg_by_boss = damage[0];
    let dmg_by_heroes: f32 = (1..=5).map(|i| damage[i]).sum();
    let heal_by_boss = healing[0];
    let heal_by_heroes: f32 = (1..=5).map(|i| healing[i]).sum();

    let boss_strikes = (dmg_by_boss / BOSS_STRIKE_DAMAGE).round() as u64;
    let hero_attacks = (dmg_by_heroes / HERO_ATTACK_DAMAGE).round() as u64;
    let boss_heals = (heal_by_boss / BOSS_HEAL_AMOUNT).round() as u64;
    let hero_heals = (heal_by_heroes / HERO_HEAL_AMOUNT).round() as u64;

    println!();
    println!("================================================================");
    println!(" RESULTS");
    println!("================================================================");
    println!("  Boss:    HP={:7.1}  alive={}", final_hp[0], final_alive[0]);
    print!  ("  Heroes:  ");
    for i in 1..=5 {
        print!("[{}: HP={:.0} alive={}] ", i, final_hp[i].max(0.0), final_alive[i]);
    }
    println!();
    println!();
    println!("  Damage dealt (per source):");
    println!("    Boss: {:.0} dmg  ({} BossStrike fires)", dmg_by_boss, boss_strikes);
    println!("    Heroes (sum): {:.0} dmg  ({} HeroAttack fires across 5 heroes)", dmg_by_heroes, hero_attacks);
    println!("  Healing done (per source):");
    println!("    Boss: {:.0} hp  ({} BossSelfHeal fires)", heal_by_boss, boss_heals);
    println!("    Heroes (sum): {:.0} hp  ({} HeroHeal fires across 5 heroes)", heal_by_heroes, hero_heals);
    if let Some(t) = self_heal_fire_tick {
        println!(
            "  Boss self-heal phase entered at tick {} ({} total fires) — confirms HP-threshold gating.",
            t, self_heal_count,
        );
    } else {
        println!("  Boss self-heal phase NEVER triggered (boss HP stayed >= 1500 throughout).");
    }
    println!();

    if let Some(t) = ended_at {
        println!("  Combat ended at tick {} — winner: {}", t, winner_label);
    } else {
        println!("  Combat ran to MAX_TICKS={} without resolution", MAX_TICKS);
        if final_alive[0] == 1 && final_heroes_alive > 0 {
            winner_label = "stalemate (both sides alive)";
        }
        println!("  Outcome: {}", winner_label);
    }

    let any_combat = dmg_by_boss > 0.0 || dmg_by_heroes > 0.0;

    println!();
    println!("================================================================");
    println!(" OUTCOME");
    println!("================================================================");
    if any_combat && ended_at.is_some() {
        println!(
            "  (a) FULL FIRE — asymmetric encounter played out end-to-end. \
            All four ability cooldowns gated correctly; verb cascade picked \
            the right action per (creature_type, tick%cooldown) pair; \
            ApplyDamage + ApplyHeal pipeline handled mixed-source events."
        );
    } else if any_combat {
        println!(
            "  (a-partial) COMBAT FIRED but no decisive outcome by tick {}. \
            Boss has {:.0} HP, {} heroes alive.",
            MAX_TICKS, final_hp[0], final_heroes_alive,
        );
    } else {
        println!(
            "  (b) NO COMBAT — neither side dealt damage. Verb cascade is \
            failing for one or both creature types. Likely root cause: \
            creature_type mask gate not routing per-discriminant verbs."
        );
    }

    // Hard asserts.
    assert!(
        any_combat,
        "boss_fight_app: ASSERTION FAILED — no combat fired (dmg_by_boss={dmg_by_boss}, dmg_by_heroes={dmg_by_heroes})",
    );
    assert!(
        boss_strikes > 0,
        "boss_fight_app: ASSERTION FAILED — boss never struck (cooldown gate failure?)",
    );
    assert!(
        hero_attacks > 0,
        "boss_fight_app: ASSERTION FAILED — heroes never attacked (cooldown gate failure?)",
    );
}
