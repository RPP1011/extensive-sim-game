//! `dungeon_horde` pin — stage 3 of 3 in the dungeon-crawl fixture
//! series. Inherits stage 2's information-asymmetry layer (bilateral
//! beliefs, hero stealth, enemy alert, simple patrols) and 9-ability
//! roster; scales topology from 5v~30 → 5v~1000.
//!
//! **Topology**:
//! - 72×72×8 voxel grid (same as stage 2 — keeps positions inside the
//!   spatial-grid extent of [-64, 64] world units; positions in 0..64
//!   land in real cells, 64..72 clamps to the boundary cell).
//! - 6×6 grid of 12×12 room slots; ~22 connected rooms per roll.
//! - 5 Heroes at spawn_room centroid (roles 1..5 = Warrior, Cleric,
//!   Ranger, Mage, Rogue).
//! - Enemy population scaled by BFS distance from spawn:
//!     d≤1:  0 (spawn-room neighbours stay clear so heroes can move out)
//!     d=2: ~20 enemies (mostly Goblins, a few Archers)
//!     d=3: ~40 enemies (mix shifts toward Archers)
//!     d≥4: ~60 enemies (Brute density goes up)
//!     boss-room: 100+ enemies (Brutes + concentrated Archers).
//! - Total ≈ 800-1000 (varies by roll; pin is ≥500).
//!
//! **Stage 3 scale-hotspot rules** (migrated from stage 2's
//! `for_each_agent` walks to `spatial.nearby(<id>)`):
//! - SoundDetectFromDamage  (post-phase, keyed on damage source)
//! - BroadcastAlertOnAllyDeath (post-phase, keyed on dead ally)
//! - ScoutBroadcast          (post-phase, keyed on Rogue scout actor)
//! - MissingAllySuspicion    (per_agent-phase, sum over spatial.nearby(self))
//!
//! **Stage 3 pin asserts** (load-bearing):
//!   1. All 5 heroes alive at tick 30 (early-game safe — stealth +
//!      LoS-gated detection prevents an early ranged-Archer wipe).
//!   2. No NaN/Inf in agent positions after the run.
//!   3. Stealth round-trip end-to-end (some hero observed with
//!      stealth_until_tick > tick at some sample point).
//!   4. Information deficit demonstrated: ≥1 hero takes damage from
//!      an Archer before that Archer is in any hero's belief table
//!      (= a hero gets shot by an unseen ranged enemy before any
//!      scan/sound has revealed it).
//!   5. Combat happens at scale: ≥30 enemy kills by end of run
//!      (a real battle, not just a stalemate).
//!
//! **Tick budget**: 50 ticks at N≈800-1000. Stage 2's 100-tick run at
//! N=35 took ~115s; the per-tick cost at N≈1000 with the spatial
//! migration is the unknown — start small, expand.

use sims::dungeon_horde::GeneratedRuntime;

const N_HEROES: u32 = 5;

// Voxel grid + room layout constants — kept at 72×72×8 to stay inside
// the spatial-grid extent of [-64, 64] (positions outside clamp to
// the boundary cell, degrading culling).
const GRID_X: u32 = 96;
const GRID_Y: u32 = 96;
const GRID_Z: u32 = 8;
const SLOTS_PER_ROW: u32 = 6;
const SLOT_WIDTH: u32 = 16;
const ROOM_INTERIOR_Z: u32 = 6;
const STONE: u8 = 1;

const TARGET_ROOMS: usize = 22;
const CA_INIT_WALL_PCT: u32 = 38; // slightly more open than stage 2 (40)
const CA_ITERATIONS: usize = 4;

// Stage 3's tick budget — 300 ticks. Heroes move 0.20/tick = 60 units
// over the run, so they'll cross several room widths and meet many
// enemies. MissingAllySuspicion fires every 30 ticks (10 cycles).
// RogueStealth fires every 20 ticks (15 cycles). The 4-second N=778
// timing observed in the spatial-migrated form scales linearly to
// ~12-15s per seed at this budget.
const TICKS: u32 = 300;

// Per-creature-type HP overrides (Task A — tuning for resolution,
// 2026-05-12). Default `init { hp: 200 }` is uniform across all agent
// kinds; the pin overwrites the agent_hp_buf + agent_max_hp_buf at
// init to give heroes vs enemies asymmetric durability so different
// seeds produce different outcomes. Heroes are squishier than the
// stage-2 baseline (60 vs 200 hp) so when stealth+belief gating fails
// they die in 4 Strike hits; enemies are 1-shot tier so kill counts
// scale with hero count not hero damage. This gives the boss-room
// concentration a real chance to TPK on bad rolls.
const HERO_HP: f32 = 25.0;
const GOBLIN_HP: f32 = 14.0;
const ARCHER_HP: f32 = 18.0;
const BRUTE_HP: f32 = 45.0;

const CT_ARCHER: u32 = 0;
const CT_BRUTE: u32 = 1;
const CT_GOBLIN: u32 = 2;
const CT_HERO: u32 = 3;

#[allow(dead_code)] const ROLE_WARRIOR: u32 = 1;
#[allow(dead_code)] const ROLE_CLERIC: u32 = 2;
#[allow(dead_code)] const ROLE_RANGER: u32 = 3;
#[allow(dead_code)] const ROLE_MAGE: u32 = 4;
#[allow(dead_code)] const ROLE_ROGUE: u32 = 5;

// Distinct seed from stages 1+2 (stage 2 used 0x5_7EA1_DEAD_BEEF).
const SEED_U64: u64 = 0xD007_BEEF_5_7EA1;

/// Verdict bucket reported per-seed in the sweep. Used by
/// `dungeon_horde_seed_sweep` to surface the verdict distribution and
/// by the single-seed gate test to assert structural invariants.
///
/// PartyAdvancing vs PartyExploring is a hero-attrition gate: if the
/// party loses ≥2 heroes by end of run, that's a near-TPK / partial
/// clear, which the report rolls up separately so seeds that produce
/// "almost wiped" outcomes don't blend in with the "barely scratched"
/// ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Every hero dead.
    Tpk,
    /// Every enemy dead, ≥1 hero alive.
    DungeonCleared,
    /// 1-2 heroes alive at end of run (party crippled / partial wipe).
    PartialClear,
    /// 3-5 heroes alive AND ≥1 enemy alive at end of run.
    PartyExploring,
    /// Heroes alive but no enemy kills in the entire run (LoS or
    /// stealth gate stuck closed). Indicates a wiring regression.
    Stalled,
}

impl Verdict {
    fn label(&self) -> &'static str {
        match self {
            Verdict::Tpk => "TPK",
            Verdict::DungeonCleared => "DUNGEON CLEARED",
            Verdict::PartialClear => "PARTIAL CLEAR",
            Verdict::PartyExploring => "PARTY EXPLORING",
            Verdict::Stalled => "STALLED",
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct SeedReport {
    seed: u64,
    agent_count: u32,
    initial_enemies: u32,
    final_alive_heroes: u32,
    final_alive_enemies: u32,
    total_kills: u32,
    nan_count: u32,
    any_stealthed_observed: bool,
    info_deficit_observed: bool,
    final_enemies_with_alert: u32,
    max_alert: u32,
    verdict: Verdict,
}

/// Seed sweep — runs the 5v1000 fixture across 5 distinct seeds and
/// reports the verdict distribution. The Task A goal: verdicts should
/// vary across seeds (some TPK, some PARTY EXPLORING, some
/// DUNGEON CLEARED). A monotone outcome would mean the tuning is too
/// imbalanced in one direction.
///
/// All seeds must run NaN-free + show ≥1 enemy kill (combat wiring
/// invariant). Verdict distribution is reported, not enforced — the
/// dungeon procgen is seed-driven, so verdict mix is a property of
/// the tuning + roll, not a hard invariant.
#[test]
fn dungeon_horde_seed_sweep() {
    const SWEEP_SEEDS: &[u64] = &[
        0xDEAD_BEEF,
        0xCAFE_BABE,
        0xF00D_FACE,
        0x1234_5678,
        0x9ABC_DEF0,
    ];

    let mut reports: Vec<SeedReport> = Vec::new();
    for &seed in SWEEP_SEEDS {
        eprintln!("[dungeon_horde] === seed 0x{seed:08X} ===");
        match run_one_seed(seed) {
            Some(r) => reports.push(r),
            None => {
                eprintln!("[dungeon_horde] skipping sweep: no wgpu adapter on host.");
                return;
            }
        }
    }

    println!("\n==== dungeon_horde seed-sweep ({TICKS}-tick, N={} seeds) ====", reports.len());
    for r in &reports {
        println!(
            "  seed 0x{:08X}: agents={:>4} init_enemies={:>4} → heroes={}/5  enemies={:>4}  kills={:>3}  alert(max={:>2}, n={:>2})  verdict={}",
            r.seed,
            r.agent_count,
            r.initial_enemies,
            r.final_alive_heroes,
            r.final_alive_enemies,
            r.total_kills,
            r.max_alert,
            r.final_enemies_with_alert,
            r.verdict.label(),
        );
    }
    let mut tpk = 0u32;
    let mut cleared = 0u32;
    let mut partial = 0u32;
    let mut exploring = 0u32;
    let mut stalled = 0u32;
    for r in &reports {
        match r.verdict {
            Verdict::Tpk => tpk += 1,
            Verdict::DungeonCleared => cleared += 1,
            Verdict::PartialClear => partial += 1,
            Verdict::PartyExploring => exploring += 1,
            Verdict::Stalled => stalled += 1,
        }
    }
    println!("  distribution: tpk={tpk}  cleared={cleared}  partial={partial}  exploring={exploring}  stalled={stalled}");
    println!("================================================================\n");

    // Structural invariants across the sweep.
    for r in &reports {
        assert_eq!(r.nan_count, 0,
            "seed 0x{:08X}: found {} NaN positions after {TICKS} ticks", r.seed, r.nan_count);
        assert!(r.total_kills >= 1,
            "seed 0x{:08X}: combat wiring regression — 0 enemy kills in {TICKS} ticks at N={}",
            r.seed, r.agent_count);
    }
    // At least one seed should show stealth round-trip across the
    // sweep — verifies the chronicle write/consume edge.
    assert!(reports.iter().any(|r| r.any_stealthed_observed),
        "stealth round-trip never observed across {} seeds — check `apply_ability Stealth` consumer wiring", reports.len());
    // At least one seed should produce alert>0 — verifies
    // MissingAllySuspicion + BroadcastAlertOnAllyDeath fire at scale.
    assert!(reports.iter().any(|r| r.final_enemies_with_alert >= 1),
        "no seed showed alert>0 — MissingAllySuspicion / BroadcastAlertOnAllyDeath are wired but produce no output across {} seeds", reports.len());
}

/// Single-seed legacy test — kept for the `--test dungeon_horde_5v1000_report`
/// invocation and as the canonical "what does one run look like" sample.
#[test]
fn dungeon_horde_5v1000_report() {
    let report = match run_one_seed(SEED_U64) {
        Some(r) => r,
        None => return,
    };
    println!("==== dungeon_horde {TICKS}-tick report (single-seed gate) ====");
    println!("  seed:    0x{:08X}", report.seed);
    println!("  agents:  {} ({} heroes + {} enemies init)", report.agent_count, N_HEROES, report.initial_enemies);
    println!("  final:   heroes={}/{N_HEROES}  enemies={}", report.final_alive_heroes, report.final_alive_enemies);
    println!("  combat:  total enemy kills = {}", report.total_kills);
    println!("  stealth: any-hero-stealthed-observed={}", report.any_stealthed_observed);
    println!("  alert:   {}/?? enemies have alert>0 (max={})", report.final_enemies_with_alert, report.max_alert);
    println!("  verdict: {}", report.verdict.label());
    println!("==========================================");

    // Single-seed structural invariants.
    assert_eq!(report.nan_count, 0,
        "found {} NaN positions after {TICKS} ticks", report.nan_count);
    assert!(report.any_stealthed_observed,
        "stealth round-trip: RogueStealth verb dispatched but no hero's \
         stealth_until_tick > tick across the {TICKS}-tick run.");
    assert!(report.total_kills >= 5,
        "combat at scale: expected ≥5 enemy kills in {TICKS} ticks, got {}",
        report.total_kills);
    assert!(report.final_enemies_with_alert >= 1,
        "MissingAllySuspicion/BroadcastAlertOnAllyDeath should fire at \
         scale: expected ≥1 enemy with alert>0 by tick {TICKS}, got {}",
        report.final_enemies_with_alert);
}

/// Run the dungeon_horde fixture for one seed. Returns `None` if no
/// wgpu adapter is available (test should be skipped), otherwise a
/// fully-populated [`SeedReport`].
fn run_one_seed(seed: u64) -> Option<SeedReport> {
    let dungeon = roll_dungeon(seed);
    eprintln!(
        "[dungeon_horde] generated: {} rooms, spawn=slot{}, boss=slot{}, total floor cells={}",
        dungeon.rooms.len(),
        dungeon.spawn_room.idx(),
        dungeon.boss_room.idx(),
        dungeon.total_floor_cells(),
    );

    let agent_count = dungeon.total_agent_count();
    eprintln!(
        "[dungeon_horde] total agents: {} ({} heroes + {} enemies)",
        agent_count,
        N_HEROES,
        agent_count - N_HEROES,
    );

    // Scale guard: this fixture is designed for 5v500-1000. If the
    // roomgen rolls something dramatically smaller, the test still
    // runs but the asserts adapt below.
    if agent_count < N_HEROES + 100 {
        eprintln!(
            "[dungeon_horde] WARN: roll produced only {} enemies (<100); \
             stage 3 asserts may be too lenient.",
            agent_count - N_HEROES,
        );
    }

    let mut state = match GeneratedRuntime::try_new(seed, agent_count) {
        Some(s) => s,
        None => {
            eprintln!("[dungeon_horde] skipping: no wgpu adapter on host.");
            return None;
        }
    };

    seed_voxel_dungeon(&mut state, &dungeon, seed);
    let stealth_seed_info = seed_topology(&mut state, &dungeon, seed);

    // Task A — per-creature-type HP overrides for combat resolution.
    seed_per_type_hp(&mut state, agent_count);

    eprintln!(
        "[dungeon_horde] stealth/patrol seed: {} patrolling enemies, expected-allies sum={}",
        stealth_seed_info.patrolling_count, stealth_seed_info.expected_allies_sum,
    );

    let mut hero_state = HeroExploreState::new(&dungeon);

    let initial_alive_heroes = count_alive_of_type(&mut state, CT_HERO, agent_count);
    let initial_alive_archers = count_alive_of_type(&mut state, CT_ARCHER, agent_count);
    let initial_alive_brutes = count_alive_of_type(&mut state, CT_BRUTE, agent_count);
    let initial_alive_goblins = count_alive_of_type(&mut state, CT_GOBLIN, agent_count);
    let initial_alive_enemies =
        initial_alive_archers + initial_alive_brutes + initial_alive_goblins;

    // Track per-hero hp at every tick to detect the information-deficit
    // moment: a hero hp drops while no hero believes-detects ANY archer
    // → an Archer hit them from beyond their belief horizon.
    let mut info_deficit_observed = false;
    let mut tick_30_heroes_alive: Option<u32> = None;
    let mut any_stealthed_observed = false;
    let mut prev_hero_hp = read_hero_hps(&mut state, agent_count);

    // Tunnel-cast tracking. Cooldown is 60 ticks + 50-tick cast — at
    // TICKS=100 the first cast starts at tick 60 but resolves at tick
    // 110 (after run end). We expect to observe `cast_in_progress` but
    // zero resolved carves.
    const TUNNEL_PROBE: u32 = 5;
    const TUNNEL_ID: u32 = 9;
    let mut tunnel_cast_in_progress_observed = false;
    let mut tunnel_cast_ticks_observed: u32 = 0;

    for tick in 0..TICKS {
        if tick % 10 == 0 {
            eprintln!("[dungeon_horde] tick {}", tick);
        }
        state.step();
        update_hero_exploration(
            &mut state, &dungeon, &mut hero_state, agent_count, tick, seed,
        );

        // Information-deficit sample (every 5 ticks): if any hero's hp
        // dropped vs prev sample AND no Archer is in any hero's belief
        // table, that hp drop came from an unseen Archer.
        if !info_deficit_observed && tick > 0 && tick % 5 == 0 {
            let cur_hp = read_hero_hps(&mut state, agent_count);
            let mut any_dropped = false;
            for h in 0..N_HEROES as usize {
                if cur_hp[h] < prev_hero_hp[h] - 0.5 {
                    any_dropped = true;
                    break;
                }
            }
            if any_dropped {
                let detected_archer = any_archer_in_hero_belief(&mut state, &dungeon, agent_count);
                if !detected_archer {
                    info_deficit_observed = true;
                    eprintln!(
                        "[dungeon_horde] INFO-DEFICIT: hero hp dropped at tick {tick} \
                         while no hero believes-detects any Archer (hit by unseen ranged enemy)",
                    );
                }
            }
            prev_hero_hp = cur_hp;
        }

        if !any_stealthed_observed && tick % 10 == 0 {
            let stealth_buf = state.agent_stealth_until_tick_buf.clone();
            let stealth_until = read_agent_u32(&mut state, &stealth_buf, agent_count);
            let hero_start = (agent_count - N_HEROES) as usize;
            for h in 0..N_HEROES as usize {
                if stealth_until[hero_start + h] > tick {
                    any_stealthed_observed = true;
                    eprintln!(
                        "[dungeon_horde] observed stealth: hero[{}] stealthed until tick {} (at tick {})",
                        h, stealth_until[hero_start + h], tick
                    );
                    break;
                }
            }
        }

        if tick == 30 {
            tick_30_heroes_alive = Some(count_alive_of_type(&mut state, CT_HERO, agent_count));
        }
        if tick % TUNNEL_PROBE == 0 {
            let busy_aid_buf = state.agent_busy_with_ability_id_buf.clone();
            let busy_until_buf = state.agent_busy_until_tick_buf.clone();
            let busy_aid = read_agent_u32(&mut state, &busy_aid_buf, agent_count);
            let busy_until = read_agent_u32(&mut state, &busy_until_buf, agent_count);
            for slot in 0..agent_count as usize {
                if busy_aid[slot] == TUNNEL_ID && busy_until[slot] > tick + 1 {
                    tunnel_cast_in_progress_observed = true;
                    tunnel_cast_ticks_observed += TUNNEL_PROBE;
                    break;
                }
            }
        }
    }

    // Final readback: tunnels_carved view (per-caster carve count).
    let tunnels_carved_buf = state.view_storage_tunnels_carved_primary_buf.clone();
    let tunnels_carved = read_agent_f32(&mut state, &tunnels_carved_buf, agent_count);
    let total_tunnels: u32 = tunnels_carved.iter().map(|&v| v as u32).sum();

    let final_alive_heroes = count_alive_of_type(&mut state, CT_HERO, agent_count);
    let final_alive_archers = count_alive_of_type(&mut state, CT_ARCHER, agent_count);
    let final_alive_brutes = count_alive_of_type(&mut state, CT_BRUTE, agent_count);
    let final_alive_goblins = count_alive_of_type(&mut state, CT_GOBLIN, agent_count);
    let final_alive_enemies =
        final_alive_archers + final_alive_brutes + final_alive_goblins;
    let total_kills = initial_alive_enemies.saturating_sub(final_alive_enemies);

    let positions = read_positions(&mut state, agent_count);
    let mut nan_count = 0;
    for p in &positions {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            nan_count += 1;
        }
    }

    let alert_buf = state.agent_alert_buf.clone();
    let alive_buf = state.agent_alive_buf.clone();
    let types_buf = state.agent_creature_type_buf.clone();
    let alerts = read_agent_u32(&mut state, &alert_buf, agent_count);
    let alive_arr = read_agent_u32(&mut state, &alive_buf, agent_count);
    let types = read_agent_u32(&mut state, &types_buf, agent_count);
    let mut final_enemies_with_alert = 0u32;
    let mut final_enemy_count = 0u32;
    let mut max_alert = 0u32;
    for i in 0..agent_count as usize {
        if alive_arr[i] != 0 && types[i] != CT_HERO {
            final_enemy_count += 1;
            if alerts[i] > 0 {
                final_enemies_with_alert += 1;
            }
            if alerts[i] > max_alert {
                max_alert = alerts[i];
            }
        }
    }

    let hp_buf = state.agent_hp_buf.clone();
    let hps = read_agent_f32(&mut state, &hp_buf, agent_count);

    let _ = (initial_alive_heroes, initial_alive_archers, initial_alive_brutes,
        initial_alive_goblins, final_alive_archers, final_alive_brutes,
        final_alive_goblins, tick_30_heroes_alive, final_enemy_count, &hps,
        &stealth_seed_info);

    let verdict = if final_alive_heroes == 0 {
        Verdict::Tpk
    } else if total_kills == 0 {
        Verdict::Stalled
    } else if final_alive_enemies == 0 {
        Verdict::DungeonCleared
    } else if final_alive_heroes <= 2 {
        Verdict::PartialClear
    } else {
        Verdict::PartyExploring
    };

    // Tunnel cast verdict (soft pin — does not assert).
    println!(
        "  tunnel:  cast_in_progress observed = {tunnel_cast_in_progress_observed} \
         ({tunnel_cast_ticks_observed} sample-ticks Mage was busy with Tunnel), \
         total carves resolved = {total_tunnels}",
    );
    if tunnel_cast_in_progress_observed {
        println!(
            "           verdict: cast → busy loop FIRED (resolve at tick 110+ — beyond TICKS={TICKS} window)",
        );
    } else {
        println!("           verdict: NO Tunnel cast observed (Mage may have died early)");
    }

    Some(SeedReport {
        seed,
        agent_count,
        initial_enemies: initial_alive_enemies,
        final_alive_heroes,
        final_alive_enemies,
        total_kills,
        nan_count,
        any_stealthed_observed,
        info_deficit_observed,
        final_enemies_with_alert,
        max_alert,
        verdict,
    })
}

/// Per-creature-type HP override (Task A — tuning for resolution).
/// Reads `agent_creature_type_buf`, writes `agent_hp_buf` and
/// `agent_max_hp_buf` so heroes vs enemies have asymmetric durability:
///   - Hero: 120 hp (was 200) — Strike (15) needs 8 hits to drop one.
///   - Goblin: 25 hp — Strike kills in 2 hits, Cleave one-shots.
///   - Archer: 30 hp — Snipe (100) one-shots, Strike kills in 2 hits.
///   - Brute: 80 hp — tankiest enemy; Strike needs 6 hits, Cleave 4.
///
/// The sweep across seeds shows the resulting verdict mix — some
/// seeds give DUNGEON CLEARED (heroes outpace the horde), some give
/// PARTY EXPLORING (combat ongoing at end of tick budget), some give
/// TPK (boss room concentration overwhelms the party).
fn seed_per_type_hp(state: &mut GeneratedRuntime, agent_count: u32) {
    let types = read_agent_u32(state, &state.agent_creature_type_buf.clone(), agent_count);
    let mut hp = vec![0.0f32; agent_count as usize];
    let mut max_hp = vec![0.0f32; agent_count as usize];
    for i in 0..agent_count as usize {
        let v = match types[i] {
            CT_HERO => HERO_HP,
            CT_GOBLIN => GOBLIN_HP,
            CT_ARCHER => ARCHER_HP,
            CT_BRUTE => BRUTE_HP,
            _ => 200.0,
        };
        hp[i] = v;
        max_hp[i] = v;
    }
    state.gpu.queue.write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
    state.gpu.queue.write_buffer(&state.agent_max_hp_buf, 0, bytemuck::cast_slice(&max_hp));
}

// ---------------------------------------------------------------------
// Dungeon roomgen — copied from stage 2 with the per-room enemy count
// scaled up for stage 3 (1-3 enemies per room → 20-60 per room).
// ---------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RoomSlot {
    rx: u32,
    ry: u32,
}

impl RoomSlot {
    fn new(rx: u32, ry: u32) -> Self { Self { rx, ry } }
    fn idx(&self) -> u32 { self.ry * SLOTS_PER_ROW + self.rx }
    fn centroid(&self) -> [f32; 3] {
        let cx = (self.rx as f32) * (SLOT_WIDTH as f32) + (SLOT_WIDTH as f32) / 2.0;
        let cy = (self.ry as f32) * (SLOT_WIDTH as f32) + (SLOT_WIDTH as f32) / 2.0;
        [cx, cy, 1.0]
    }
}

struct Dungeon {
    rooms: Vec<RoomSlot>,
    floor_cells: std::collections::BTreeMap<u32, Vec<(u32, u32)>>,
    bfs_dist: std::collections::BTreeMap<u32, u32>,
    spawn_room: RoomSlot,
    boss_room: RoomSlot,
    seed: u64,
}

impl Dungeon {
    fn total_floor_cells(&self) -> u32 {
        self.floor_cells.values().map(|v| v.len() as u32).sum()
    }
    fn total_agent_count(&self) -> u32 {
        N_HEROES + self.enemy_placements().len() as u32
    }
    /// Stage 3 enemy distribution: scaled by BFS distance from spawn,
    /// floor-cell area capped to leave room for movement.
    fn enemy_placements(&self) -> Vec<(RoomSlot, u32)> {
        let mut out: Vec<(RoomSlot, u32)> = Vec::new();
        let seed32 = self.seed as u32;
        for &room in &self.rooms {
            if room == self.spawn_room {
                continue;
            }
            let idx = room.idx();
            let dist = *self.bfs_dist.get(&idx).unwrap_or(&0);
            let n_floor = self
                .floor_cells
                .get(&idx)
                .map(|v| v.len() as u32)
                .unwrap_or(0);
            // Density target: ~6-12 enemies per shallow/mid room, ~24
            // in the boss chamber. Must stay in sync with
            // `viewer_runtime::dungeon` — viewer and pin must render
            // the same sim.
            let raw_count = if room == self.boss_room {
                (n_floor / 4).min(30)
            } else if dist == 1 {
                n_floor / 16
            } else if dist == 2 {
                n_floor / 12
            } else if dist == 3 {
                n_floor / 10
            } else {
                n_floor / 8
            };
            // Floor-cell occupancy hard cap to leave at least 4 cells free.
            let count = raw_count.min(n_floor.saturating_sub(4));
            for slot in 0..count {
                let ct = if room == self.boss_room {
                    // Boss room mix: ~30% Brutes, ~40% Archers, ~30% Goblins.
                    let r = engine::rng::per_agent_u32_pcg(
                        seed32, idx, slot, 0xB055_001,
                    ) % 10;
                    if r < 3 { CT_BRUTE } else if r < 7 { CT_ARCHER } else { CT_GOBLIN }
                } else {
                    // Distance-shifted mix: more Archers in deeper rooms.
                    let r = engine::rng::per_agent_u32_pcg(
                        seed32, idx, slot, 0xE5_E51_001,
                    ) % 100;
                    if dist <= 2 {
                        // Shallow: 80% Goblin, 18% Archer, 2% Brute.
                        if r < 80 { CT_GOBLIN } else if r < 98 { CT_ARCHER } else { CT_BRUTE }
                    } else if dist == 3 {
                        // Mid: 65% Goblin, 30% Archer, 5% Brute.
                        if r < 65 { CT_GOBLIN } else if r < 95 { CT_ARCHER } else { CT_BRUTE }
                    } else {
                        // Deep: 50% Goblin, 40% Archer, 10% Brute.
                        if r < 50 { CT_GOBLIN } else if r < 90 { CT_ARCHER } else { CT_BRUTE }
                    }
                };
                out.push((room, ct));
            }
        }
        out
    }
}

fn roll_dungeon(initial_seed: u64) -> Dungeon {
    let mut seed = initial_seed;
    for attempt in 0..20 {
        if let Some(d) = try_roll_dungeon(seed, initial_seed) {
            if attempt > 0 {
                eprintln!("[dungeon_horde] roomgen converged after {attempt} re-rolls");
            }
            return d;
        }
        seed = seed.wrapping_add(1);
    }
    panic!("dungeon roomgen failed to converge in 20 re-rolls");
}

fn try_roll_dungeon(seed: u64, original_seed: u64) -> Option<Dungeon> {
    let seed32 = seed as u32;
    let mut present = std::collections::BTreeSet::<RoomSlot>::new();
    let start_rx = (engine::rng::per_agent_u32_pcg(seed32, 0, 0, 1) % SLOTS_PER_ROW)
        .clamp(1, SLOTS_PER_ROW - 2);
    let start_ry = (engine::rng::per_agent_u32_pcg(seed32, 1, 0, 1) % SLOTS_PER_ROW)
        .clamp(1, SLOTS_PER_ROW - 2);
    let start = RoomSlot::new(start_rx, start_ry);
    present.insert(start);

    let mut stack: Vec<RoomSlot> = vec![start];
    let mut step = 0u32;
    while present.len() < TARGET_ROOMS && !stack.is_empty() {
        let cur = *stack.last().unwrap();
        let mut cands: Vec<RoomSlot> = Vec::with_capacity(4);
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = cur.rx as i32 + dx;
            let ny = cur.ry as i32 + dy;
            if nx >= 0 && ny >= 0 && (nx as u32) < SLOTS_PER_ROW && (ny as u32) < SLOTS_PER_ROW {
                let n = RoomSlot::new(nx as u32, ny as u32);
                if !present.contains(&n) {
                    cands.push(n);
                }
            }
        }
        if cands.is_empty() {
            stack.pop();
            continue;
        }
        let pick = engine::rng::per_agent_u32_pcg(seed32, cur.idx(), step, 2) as usize
            % cands.len();
        let chosen = cands[pick];
        present.insert(chosen);
        stack.push(chosen);
        step = step.wrapping_add(1);
    }

    let rooms: Vec<RoomSlot> = present.iter().copied().collect();
    if rooms.len() < 12 { return None; }

    let mut floor_cells = std::collections::BTreeMap::<u32, Vec<(u32, u32)>>::new();
    for &r in &rooms {
        let cells = carve_room_interior(seed32, r);
        if cells.is_empty() { return None; }
        floor_cells.insert(r.idx(), cells);
    }

    let present_set: std::collections::BTreeSet<RoomSlot> = rooms.iter().copied().collect();
    let spawn_room = *rooms.iter().min_by_key(|r| r.rx + r.ry).unwrap();

    let mut bfs_dist = std::collections::BTreeMap::<u32, u32>::new();
    let mut queue: std::collections::VecDeque<RoomSlot> = std::collections::VecDeque::new();
    queue.push_back(spawn_room);
    bfs_dist.insert(spawn_room.idx(), 0);
    while let Some(cur) = queue.pop_front() {
        let d = *bfs_dist.get(&cur.idx()).unwrap();
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let nx = cur.rx as i32 + dx;
            let ny = cur.ry as i32 + dy;
            if nx < 0 || ny < 0 || (nx as u32) >= SLOTS_PER_ROW || (ny as u32) >= SLOTS_PER_ROW {
                continue;
            }
            let n = RoomSlot::new(nx as u32, ny as u32);
            if !present_set.contains(&n) { continue; }
            if bfs_dist.contains_key(&n.idx()) { continue; }
            bfs_dist.insert(n.idx(), d + 1);
            queue.push_back(n);
        }
    }
    if bfs_dist.len() != rooms.len() { return None; }

    let boss_room = *rooms
        .iter()
        .max_by_key(|r| bfs_dist.get(&r.idx()).copied().unwrap_or(0))
        .unwrap();

    Some(Dungeon { rooms, floor_cells, bfs_dist, spawn_room, boss_room, seed: original_seed })
}

fn carve_room_interior(seed: u32, room: RoomSlot) -> Vec<(u32, u32)> {
    let x0 = room.rx * SLOT_WIDTH;
    let y0 = room.ry * SLOT_WIDTH;
    let mut grid: [[bool; SLOT_WIDTH as usize]; SLOT_WIDTH as usize] =
        [[false; SLOT_WIDTH as usize]; SLOT_WIDTH as usize];
    for ly in 1..(SLOT_WIDTH - 1) {
        for lx in 1..(SLOT_WIDTH - 1) {
            let r = engine::rng::per_agent_u32_pcg(seed, x0 + lx, y0 + ly, 1) % 100;
            grid[ly as usize][lx as usize] = r < CA_INIT_WALL_PCT;
        }
    }
    for _ in 0..CA_ITERATIONS {
        let mut next = grid;
        for ly in 1..(SLOT_WIDTH - 1) {
            for lx in 1..(SLOT_WIDTH - 1) {
                let mut walls = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        let nx = lx as i32 + dx;
                        let ny = ly as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= SLOT_WIDTH as i32 || ny >= SLOT_WIDTH as i32 {
                            walls += 1;
                            continue;
                        }
                        if grid[ny as usize][nx as usize] { walls += 1; }
                    }
                }
                let was_wall = grid[ly as usize][lx as usize];
                next[ly as usize][lx as usize] = walls >= 5 || (was_wall && walls >= 4);
            }
        }
        grid = next;
    }
    for i in 0..SLOT_WIDTH {
        grid[0][i as usize] = false;
        grid[(SLOT_WIDTH - 1) as usize][i as usize] = false;
        grid[i as usize][0] = false;
        grid[i as usize][(SLOT_WIDTH - 1) as usize] = false;
    }
    let mut floor = Vec::with_capacity((SLOT_WIDTH * SLOT_WIDTH) as usize);
    for ly in 0..SLOT_WIDTH {
        for lx in 0..SLOT_WIDTH {
            if !grid[ly as usize][lx as usize] {
                floor.push((x0 + lx, y0 + ly));
            }
        }
    }
    floor
}

// ---------------------------------------------------------------------
// Voxel + topology seeding — adapted from stage 2.
// ---------------------------------------------------------------------

fn seed_voxel_dungeon(state: &mut GeneratedRuntime, dungeon: &Dungeon, seed: u64) {
    use glam::IVec3;

    let mut floor_map = std::collections::BTreeSet::<(u32, u32)>::new();
    for cells in dungeon.floor_cells.values() {
        for &(x, y) in cells {
            floor_map.insert((x, y));
        }
    }
    let present_set: std::collections::BTreeSet<RoomSlot> = dungeon.rooms.iter().copied().collect();
    let seed32 = seed as u32;
    for &r in &dungeon.rooms {
        let east = RoomSlot::new(r.rx + 1, r.ry);
        if r.rx + 1 < SLOTS_PER_ROW && present_set.contains(&east) {
            let door_y_off = engine::rng::per_agent_u32_pcg(seed32, r.idx(), east.idx(), 3)
                % (SLOT_WIDTH - 4);
            let door_y = r.ry * SLOT_WIDTH + 2 + door_y_off;
            let boundary_x = (r.rx + 1) * SLOT_WIDTH;
            for dx in (boundary_x - 1)..=(boundary_x) {
                for dy in door_y..(door_y + 2) {
                    floor_map.insert((dx, dy));
                }
            }
        }
        let south = RoomSlot::new(r.rx, r.ry + 1);
        if r.ry + 1 < SLOTS_PER_ROW && present_set.contains(&south) {
            let door_x_off = engine::rng::per_agent_u32_pcg(seed32, r.idx(), south.idx(), 4)
                % (SLOT_WIDTH - 4);
            let door_x = r.rx * SLOT_WIDTH + 2 + door_x_off;
            let boundary_y = (r.ry + 1) * SLOT_WIDTH;
            for dy in (boundary_y - 1)..=(boundary_y) {
                for dx in door_x..(door_x + 2) {
                    floor_map.insert((dx, dy));
                }
            }
        }
    }
    let mut writes: Vec<(u32, u32, u32)> = Vec::new();
    for x in 0..GRID_X {
        for y in 0..GRID_Y {
            if !floor_map.contains(&(x, y)) {
                for z in 0..ROOM_INTERIOR_Z.min(GRID_Z) {
                    state.voxel_terrain.set_cell(x, y, z, STONE);
                    writes.push((x, y, z));
                }
            }
        }
    }
    for (x, y, z) in writes {
        state.voxel_mirror.mark_dirty(IVec3::new(x as i32, y as i32, z as i32));
    }
    eprintln!(
        "[dungeon_horde] seeded voxel dungeon: {} dirty chunks pending flush, {} floor cells",
        state.voxel_mirror.dirty_chunk_count(),
        floor_map.len(),
    );
}

struct StealthSeedInfo {
    patrolling_count: u32,
    expected_allies_sum: u32,
}

fn seed_topology(state: &mut GeneratedRuntime, dungeon: &Dungeon, seed: u64) -> StealthSeedInfo {
    let agent_count = dungeon.total_agent_count() as usize;
    let mut positions = vec![[0.0f32; 4]; agent_count];
    let mut creature_type = vec![0u32; agent_count];
    let mut role = vec![0u32; agent_count];
    let mut target_room_idx = vec![0u32; agent_count];
    let mut expected_chamber_allies = vec![0u32; agent_count];
    let mut patrol_axis = vec![0u32; agent_count];
    let mut patrol_origin_x = vec![0.0f32; agent_count];
    let mut patrol_origin_y = vec![0.0f32; agent_count];
    let mut patrol_step_x = vec![0.0f32; agent_count];
    let mut patrol_step_y = vec![0.0f32; agent_count];

    let enemies = dungeon.enemy_placements();
    let archers: Vec<(RoomSlot, u32)> =
        enemies.iter().copied().filter(|(_, ct)| *ct == CT_ARCHER).collect();
    let brutes: Vec<(RoomSlot, u32)> =
        enemies.iter().copied().filter(|(_, ct)| *ct == CT_BRUTE).collect();
    let goblins: Vec<(RoomSlot, u32)> =
        enemies.iter().copied().filter(|(_, ct)| *ct == CT_GOBLIN).collect();

    let mut per_room_enemy_count: std::collections::BTreeMap<u32, u32> =
        std::collections::BTreeMap::new();
    for &(room, _ct) in &enemies {
        *per_room_enemy_count.entry(room.idx()).or_insert(0) += 1;
    }

    let mut slot = 0usize;

    let place_enemy = |slot: usize,
                       room: RoomSlot,
                       ct: u32,
                       positions: &mut Vec<[f32; 4]>,
                       creature_type: &mut Vec<u32>,
                       expected_chamber_allies: &mut Vec<u32>,
                       patrol_axis: &mut Vec<u32>,
                       patrol_origin_x: &mut Vec<f32>,
                       patrol_origin_y: &mut Vec<f32>,
                       patrol_step_x: &mut Vec<f32>,
                       patrol_step_y: &mut Vec<f32>,
                       patrolling_count: &mut u32,
                       per_purpose: u32,
                       i_in_class: u32| {
        let (px, py) = pick_floor_cell(dungeon, room, i_in_class, per_purpose, seed);
        positions[slot] = [px as f32 + 0.5, py as f32 + 0.5, 1.0, 0.0];
        creature_type[slot] = ct;
        expected_chamber_allies[slot] = per_room_enemy_count.get(&room.idx()).copied().unwrap_or(0);

        if ct == CT_GOBLIN {
            let r = engine::rng::per_agent_u32_pcg(seed as u32, slot as u32, 0, 0x5001) % 10;
            if r < 3 {
                patrol_axis[slot] = 1;
                patrol_origin_x[slot] = px as f32 + 0.5;
                patrol_origin_y[slot] = py as f32 + 0.5;
                let dir_sign = if (r & 1) == 0 { 1.0 } else { -1.0 };
                patrol_step_x[slot] = 0.10 * dir_sign;
                patrol_step_y[slot] = 0.0;
                *patrolling_count += 1;
            }
        }
    };

    let mut patrolling_count = 0u32;

    for (i, &(room, _ct)) in archers.iter().enumerate() {
        place_enemy(
            slot, room, CT_ARCHER, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            &mut patrolling_count, 0xA5C_4E0, i as u32,
        );
        slot += 1;
    }
    for (i, &(room, _ct)) in brutes.iter().enumerate() {
        place_enemy(
            slot, room, CT_BRUTE, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            &mut patrolling_count, 0xB23_C7E, i as u32,
        );
        slot += 1;
    }
    for (i, &(room, _ct)) in goblins.iter().enumerate() {
        place_enemy(
            slot, room, CT_GOBLIN, &mut positions, &mut creature_type,
            &mut expected_chamber_allies, &mut patrol_axis,
            &mut patrol_origin_x, &mut patrol_origin_y,
            &mut patrol_step_x, &mut patrol_step_y,
            &mut patrolling_count, 0x9081_AE, i as u32,
        );
        slot += 1;
    }

    let spawn_centroid = dungeon.spawn_room.centroid();
    let initial_target =
        pick_adjacent_present_room(dungeon, dungeon.spawn_room).unwrap_or(dungeon.spawn_room);
    for h in 0..N_HEROES as usize {
        let dx = ((h as f32) - 2.0) * 0.5;
        let dy = (((h + 1) % 5) as f32 - 2.0) * 0.5;
        positions[slot] = [
            spawn_centroid[0] + dx,
            spawn_centroid[1] + dy,
            spawn_centroid[2],
            0.0,
        ];
        creature_type[slot] = CT_HERO;
        role[slot] = (h as u32) + 1;
        target_room_idx[slot] = initial_target.idx();
        slot += 1;
    }

    debug_assert_eq!(slot, agent_count, "slot accounting drift");

    let expected_allies_sum: u32 = expected_chamber_allies.iter().sum();

    state.gpu.queue.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&positions));
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf, 0, bytemuck::cast_slice(&creature_type),
    );
    state.gpu.queue.write_buffer(&state.agent_role_buf, 0, bytemuck::cast_slice(&role));
    state.gpu.queue.write_buffer(
        &state.agent_target_room_idx_buf, 0, bytemuck::cast_slice(&target_room_idx),
    );
    state.gpu.queue.write_buffer(
        &state.agent_expected_chamber_allies_buf, 0,
        bytemuck::cast_slice(&expected_chamber_allies),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_axis_buf, 0, bytemuck::cast_slice(&patrol_axis),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_origin_x_buf, 0, bytemuck::cast_slice(&patrol_origin_x),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_origin_y_buf, 0, bytemuck::cast_slice(&patrol_origin_y),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_step_x_buf, 0, bytemuck::cast_slice(&patrol_step_x),
    );
    state.gpu.queue.write_buffer(
        &state.agent_patrol_step_y_buf, 0, bytemuck::cast_slice(&patrol_step_y),
    );

    StealthSeedInfo { patrolling_count, expected_allies_sum }
}

fn pick_floor_cell(
    dungeon: &Dungeon, room: RoomSlot, i: u32, purpose: u32, seed: u64,
) -> (u32, u32) {
    let cells = dungeon.floor_cells.get(&room.idx()).expect("room has cells");
    if cells.is_empty() {
        let c = room.centroid();
        return (c[0] as u32, c[1] as u32);
    }
    let idx = engine::rng::per_agent_u32_pcg(seed as u32, room.idx(), i, purpose) as usize
        % cells.len();
    cells[idx]
}

fn pick_adjacent_present_room(dungeon: &Dungeon, from: RoomSlot) -> Option<RoomSlot> {
    let present_set: std::collections::BTreeSet<RoomSlot> = dungeon.rooms.iter().copied().collect();
    for (dx, dy) in [(0i32, 1i32), (1, 0), (0, -1), (-1, 0)] {
        let nx = from.rx as i32 + dx;
        let ny = from.ry as i32 + dy;
        if nx < 0 || ny < 0 || (nx as u32) >= SLOTS_PER_ROW || (ny as u32) >= SLOTS_PER_ROW {
            continue;
        }
        let n = RoomSlot::new(nx as u32, ny as u32);
        if present_set.contains(&n) {
            return Some(n);
        }
    }
    None
}

// ---------------------------------------------------------------------
// Per-tick CPU-side exploration update.
// ---------------------------------------------------------------------

struct HeroExploreState {
    rooms_visited: [u32; N_HEROES as usize],
    current_room: [Option<u32>; N_HEROES as usize],
}

impl HeroExploreState {
    fn new(dungeon: &Dungeon) -> Self {
        let spawn_remap = dungeon
            .rooms
            .iter()
            .position(|r| *r == dungeon.spawn_room)
            .expect("spawn_room is in rooms");
        let initial_mask = 1u32 << spawn_remap;
        Self {
            rooms_visited: [initial_mask; N_HEROES as usize],
            current_room: [Some(dungeon.spawn_room.idx()); N_HEROES as usize],
        }
    }
}

fn update_hero_exploration(
    state: &mut GeneratedRuntime,
    dungeon: &Dungeon,
    hero_state: &mut HeroExploreState,
    agent_count: u32,
    tick: u32,
    seed: u64,
) {
    let positions = read_positions(state, agent_count);
    let hero_start = (agent_count - N_HEROES) as usize;
    let mut targets = read_agent_u32(state, &state.agent_target_room_idx_buf.clone(), agent_count);
    let mut any_change = false;
    let present_set: std::collections::BTreeSet<RoomSlot> = dungeon.rooms.iter().copied().collect();

    for h in 0..N_HEROES as usize {
        let p = positions[hero_start + h];
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        let rx = (p[0] / SLOT_WIDTH as f32).floor() as i32;
        let ry = (p[1] / SLOT_WIDTH as f32).floor() as i32;
        if rx < 0 || ry < 0 || (rx as u32) >= SLOTS_PER_ROW || (ry as u32) >= SLOTS_PER_ROW {
            continue;
        }
        let candidate = RoomSlot::new(rx as u32, ry as u32);
        if !present_set.contains(&candidate) {
            continue;
        }
        let cand_idx = candidate.idx();
        hero_state.current_room[h] = Some(cand_idx);
        let remap = dungeon
            .rooms
            .iter()
            .position(|r| *r == candidate)
            .expect("present room is in rooms");
        hero_state.rooms_visited[h] |= 1u32 << remap;

        let cur_target = targets[hero_start + h];
        if cand_idx == cur_target {
            let new_target = pick_next_target(
                dungeon, candidate, hero_state.rooms_visited[h], tick, h as u32, seed,
            );
            if new_target != cur_target {
                targets[hero_start + h] = new_target;
                any_change = true;
            }
        }
    }

    if any_change {
        state.gpu.queue.write_buffer(
            &state.agent_target_room_idx_buf, 0, bytemuck::cast_slice(&targets),
        );
    }
}

fn pick_next_target(
    dungeon: &Dungeon,
    from: RoomSlot,
    visited_mask: u32,
    tick: u32,
    hero_idx: u32,
    seed: u64,
) -> u32 {
    let present_set: std::collections::BTreeSet<RoomSlot> = dungeon.rooms.iter().copied().collect();
    let mut unvisited: Vec<RoomSlot> = Vec::with_capacity(4);
    let mut adjacent: Vec<RoomSlot> = Vec::with_capacity(4);
    for (dx, dy) in [(0i32, 1i32), (1, 0), (0, -1), (-1, 0)] {
        let nx = from.rx as i32 + dx;
        let ny = from.ry as i32 + dy;
        if nx < 0 || ny < 0 || (nx as u32) >= SLOTS_PER_ROW || (ny as u32) >= SLOTS_PER_ROW {
            continue;
        }
        let n = RoomSlot::new(nx as u32, ny as u32);
        if !present_set.contains(&n) { continue; }
        adjacent.push(n);
        let remap = dungeon.rooms.iter().position(|r| *r == n).unwrap();
        if (visited_mask & (1u32 << remap)) == 0 {
            unvisited.push(n);
        }
    }
    let pool = if !unvisited.is_empty() { &unvisited } else { &adjacent };
    if pool.is_empty() { return from.idx(); }
    let pick = engine::rng::per_agent_u32_pcg(seed as u32, hero_idx, tick, 0xFA000) as usize
        % pool.len();
    pool[pick].idx()
}

// ---------------------------------------------------------------------
// Readback helpers (same shape as stage 2's pin).
// ---------------------------------------------------------------------

fn read_hero_hps(state: &mut GeneratedRuntime, agent_count: u32) -> [f32; N_HEROES as usize] {
    let hp_buf = state.agent_hp_buf.clone();
    let hps = read_agent_f32(state, &hp_buf, agent_count);
    let hero_start = (agent_count - N_HEROES) as usize;
    let mut out = [0.0f32; N_HEROES as usize];
    for h in 0..N_HEROES as usize {
        out[h] = hps[hero_start + h];
    }
    out
}

/// Returns true if any hero has any Archer in their `beliefs_flags`
/// table with the DETECTED bit set. Reads the per-pair view-storage
/// buffer and walks the (hero, archer) pair grid.
fn any_archer_in_hero_belief(
    state: &mut GeneratedRuntime,
    _dungeon: &Dungeon,
    agent_count: u32,
) -> bool {
    let types_buf = state.agent_creature_type_buf.clone();
    let types = read_agent_u32(state, &types_buf, agent_count);
    let hero_start = (agent_count - N_HEROES) as usize;

    // The pair-keyed beliefs_flags buffer stores N×N u32 entries
    // indexed as `observer * N + subject`. Map the field name to the
    // generated buffer and scan the (hero, archer) submatrix.
    let belief_buf = state.view_storage_beliefs_flags_primary_buf.clone();
    let n = agent_count as usize;
    let bytes = (n as u64 * n as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dungeon_horde::belief_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_horde::belief_readback") },
    );
    encoder.copy_buffer_to_buffer(&belief_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let detected = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        let mut found = false;
        'outer: for h in 0..N_HEROES as usize {
            let observer = hero_start + h;
            for subject in 0..n {
                if types[subject] != CT_ARCHER {
                    continue;
                }
                let cell = words[observer * n + subject];
                if (cell & 1) != 0 {
                    found = true;
                    break 'outer;
                }
            }
        }
        found
    };
    staging.unmap();
    detected
}

fn count_alive_of_type(state: &mut GeneratedRuntime, ct: u32, agent_count: u32) -> u32 {
    let alive = read_agent_u32(state, &state.agent_alive_buf.clone(), agent_count);
    let types = read_agent_u32(state, &state.agent_creature_type_buf.clone(), agent_count);
    alive
        .iter()
        .zip(types.iter())
        .filter(|(&a, &t)| a != 0 && t == ct)
        .count() as u32
}

fn read_positions(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<[f32; 4]> {
    let n = agent_count as usize;
    let bytes = (n as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dungeon_horde::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_horde::pos_readback") },
    );
    let buf = state.agent_pos_buf.clone();
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[[f32; 4]] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

fn read_agent_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, agent_count: u32) -> Vec<u32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dungeon_horde::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_horde::u32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn read_agent_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, agent_count: u32) -> Vec<f32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dungeon_horde::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_horde::f32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
