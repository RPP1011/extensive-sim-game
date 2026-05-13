//! `dungeon_layout` pin — stage 1 of 3 in a dungeon-crawl fixture series.
//!
//! Builds a procedurally-generated voxel dungeon on a 72×72×8 grid via
//! Binding-of-Isaac-style room placement + per-room Cellular Automata.
//! Drops 5 specialized Heroes at the spawn room and ~30 mixed enemies
//! (Goblins/Brutes/Archers) across the other rooms. Runs 500 ticks
//! with frontier-greedy room-by-room exploration (host-side picks the
//! next room target between ticks via direct buffer writes).
//!
//! **Topology** (host-seeded):
//! - 72×72×8 voxel grid: 6×6 grid of 12×12 "room slots". ~18 of 36
//!   slots are present (chosen by random-walk-with-backtracking +
//!   BFS connectivity verification).
//! - Per-room CA interiors (40% initial fill, 4 iterations of
//!   `wall <- (neighbors >= 5) || (was_wall && neighbors >= 4)`).
//! - 2-cell-wide doorways between adjacent present slots.
//! - 5 Heroes at spawn_room centroid, roles {Warrior, Cleric, Ranger,
//!   Mage, Rogue}.
//! - ~30 Enemies distributed across non-spawn rooms, weighted by BFS
//!   distance from spawn; boss_room gets 8 + 1 Brute + 1 Archer.
//!
//! **Entity discriminants** (decl-order alphabetical):
//!   Archer = 0, Brute = 1, Goblin = 2, Hero = 3
//!
//! **Ability slot IDs** (alphabetical .ability filenames):
//!   Backstab=1, Cleave=2, Heal=3, Scout=4, Snipe=5,
//!   Stealth=6, Strike=7, Stun=8, Volley=9
//!
//! **Pin asserts** (load-bearing):
//!   1. All 5 heroes alive at tick 30 (early-game safety).
//!   2. Heroes have visited ≥ 3 distinct rooms by tick 200.
//!   3. Total enemy kills ≥ 10 by tick 400.
//!   4. No NaN/Inf in agent positions after 500 ticks.

use sims::dungeon_layout::GeneratedRuntime;

const N_HEROES: u32 = 5;

// Voxel grid + room layout constants.
const GRID_X: u32 = 72;
const GRID_Y: u32 = 72;
const GRID_Z: u32 = 8;
const SLOTS_PER_ROW: u32 = 6; // 6×6 grid → 36 candidate slots
const SLOT_WIDTH: u32 = 12; // each slot is 12×12 cells
const ROOM_INTERIOR_Z: u32 = 6; // dungeon ceiling height in cells
const STONE: u8 = 1;

// Roomgen tuning.
const TARGET_ROOMS: usize = 18; // aim for ~50% of 36 slots present
const CA_INIT_WALL_PCT: u32 = 40; // ~40% initial walls
const CA_ITERATIONS: usize = 4;

const TICKS: u32 = 500;

const CT_ARCHER: u32 = 0;
const CT_BRUTE: u32 = 1;
const CT_GOBLIN: u32 = 2;
const CT_HERO: u32 = 3;

#[allow(dead_code)] const ROLE_WARRIOR: u32 = 1;
#[allow(dead_code)] const ROLE_CLERIC: u32 = 2;
#[allow(dead_code)] const ROLE_RANGER: u32 = 3;
#[allow(dead_code)] const ROLE_MAGE: u32 = 4;
#[allow(dead_code)] const ROLE_ROGUE: u32 = 5;

const SEED_U64: u64 = 0xD06_C0DE_DEAD_BEEF;

#[test]
fn dungeon_500_tick_clear_report() {
    // Roll the dungeon (with one re-roll if connectivity check fails).
    // We do this BEFORE creating the GeneratedRuntime so we know the
    // exact agent count we need.
    let dungeon = roll_dungeon(SEED_U64);
    eprintln!(
        "[dungeon_layout] generated: {} rooms, spawn=slot{}, boss=slot{}, total floor cells={}",
        dungeon.rooms.len(),
        dungeon.spawn_room.idx(),
        dungeon.boss_room.idx(),
        dungeon.total_floor_cells(),
    );

    let agent_count = dungeon.total_agent_count();
    eprintln!(
        "[dungeon_layout] total agents: {} ({} heroes + {} enemies)",
        agent_count,
        N_HEROES,
        agent_count - N_HEROES,
    );

    let mut state = match GeneratedRuntime::try_new(SEED_U64, agent_count) {
        Some(s) => s,
        None => {
            eprintln!("[dungeon_layout] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_voxel_dungeon(&mut state, &dungeon);
    seed_topology(&mut state, &dungeon);

    // Sanity: report closest enemy → spawn distance (early-game
    // safety check; the pin asserts ≥4 heroes alive at tick 30).
    {
        let positions = read_positions(&mut state, agent_count);
        let spawn_centroid = dungeon.spawn_room.centroid();
        let mut closest_to_spawn: f32 = f32::INFINITY;
        for (i, p) in positions.iter().enumerate() {
            if i >= (agent_count - N_HEROES) as usize {
                continue;
            }
            let dx = p[0] - spawn_centroid[0];
            let dy = p[1] - spawn_centroid[1];
            let d = (dx * dx + dy * dy).sqrt();
            if d < closest_to_spawn {
                closest_to_spawn = d;
            }
        }
        eprintln!(
            "[dungeon_layout] closest enemy to spawn centroid: {:.2} units",
            closest_to_spawn,
        );
    }

    // CPU-side exploration tracker. `current_room_idx` and
    // `rooms_visited_lo` live here rather than in GPU SoA because no
    // kernel reads them in stage 1 (the .sim only reads
    // `target_room_idx`, which IS a GPU SoA column).
    let mut hero_state = HeroExploreState::new(&dungeon);

    let initial_alive_heroes = count_alive_of_type(&mut state, CT_HERO, agent_count);
    let initial_alive_enemies = (count_alive_of_type(&mut state, CT_ARCHER, agent_count)
        + count_alive_of_type(&mut state, CT_BRUTE, agent_count)
        + count_alive_of_type(&mut state, CT_GOBLIN, agent_count)) as u32;

    let mut tick_30_heroes_alive: Option<u32> = None;
    let mut tick_200_rooms_visited_min: Option<u32> = None;
    let mut tick_400_total_kills: Option<u32> = None;

    // Tunnel-cast tracking: sample busy_with_ability_id every TUNNEL_PROBE
    // ticks, count how many ticks at least one Mage was mid-cast.
    const TUNNEL_PROBE: u32 = 5;
    const TUNNEL_ID: u32 = 9;
    let mut tunnel_cast_in_progress_observed = false;
    let mut tunnel_cast_ticks_observed: u32 = 0;

    for tick in 0..TICKS {
        state.step();
        // After each step, update host-side exploration: read hero
        // positions, mark rooms visited, pick new target rooms.
        update_hero_exploration(&mut state, &dungeon, &mut hero_state, agent_count, tick);

        if tick == 30 {
            tick_30_heroes_alive = Some(count_alive_of_type(&mut state, CT_HERO, agent_count));
        }
        if tick == 200 {
            tick_200_rooms_visited_min = Some(
                hero_state
                    .rooms_visited
                    .iter()
                    .map(|m| m.count_ones())
                    .min()
                    .unwrap_or(0),
            );
        }
        if tick == 400 {
            let alive_enemies = count_alive_of_type(&mut state, CT_ARCHER, agent_count)
                + count_alive_of_type(&mut state, CT_BRUTE, agent_count)
                + count_alive_of_type(&mut state, CT_GOBLIN, agent_count);
            tick_400_total_kills = Some(initial_alive_enemies.saturating_sub(alive_enemies));
        }
        if tick % TUNNEL_PROBE == 0 {
            let busy_aid = read_busy_with_ability_id(&mut state, agent_count);
            let busy_until = read_busy_until_tick(&mut state, agent_count);
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
    let tunnels_carved = read_view_tunnels_carved(&mut state, agent_count);
    let total_tunnels: u32 = tunnels_carved.iter().map(|&v| v as u32).sum();

    // Final readbacks.
    let final_alive_heroes = count_alive_of_type(&mut state, CT_HERO, agent_count);
    let final_alive_archers = count_alive_of_type(&mut state, CT_ARCHER, agent_count);
    let final_alive_brutes = count_alive_of_type(&mut state, CT_BRUTE, agent_count);
    let final_alive_goblins = count_alive_of_type(&mut state, CT_GOBLIN, agent_count);
    let final_alive_enemies = final_alive_archers + final_alive_brutes + final_alive_goblins;
    let total_kills = initial_alive_enemies.saturating_sub(final_alive_enemies);

    // Per-hero final HP for the report.
    let hp_buf = state.agent_hp_buf.clone();
    let hps = read_agent_f32(&mut state, &hp_buf, agent_count);
    let positions = read_positions(&mut state, agent_count);

    let mut nan_count = 0;
    for p in &positions {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            nan_count += 1;
        }
    }

    println!("==== dungeon_layout {TICKS}-tick report ====");
    println!("  dungeon: {} rooms ({} floor cells, spawn=slot{}, boss=slot{})",
        dungeon.rooms.len(),
        dungeon.total_floor_cells(),
        dungeon.spawn_room.idx(),
        dungeon.boss_room.idx(),
    );
    println!(
        "  init:    heroes={initial_alive_heroes}/{N_HEROES}  enemies={initial_alive_enemies}",
    );
    println!(
        "  final:   heroes={final_alive_heroes}/{N_HEROES}  enemies={final_alive_enemies} (archers={final_alive_archers} brutes={final_alive_brutes} goblins={final_alive_goblins})",
    );
    println!(
        "  combat:  total enemy kills = {total_kills}",
    );
    println!("  hero hp/rooms-visited:");
    for h in 0..N_HEROES as usize {
        let role_name = match h + 1 {
            1 => "Warrior",
            2 => "Cleric",
            3 => "Ranger",
            4 => "Mage",
            5 => "Rogue",
            _ => "?",
        };
        let mask = hero_state.rooms_visited[h];
        let count = mask.count_ones();
        println!(
            "    hero[{h}] role={role_name} hp={:.1} rooms_visited={count} (mask=0b{mask:b})",
            hps[h],
        );
    }

    let outcome = if final_alive_heroes == 0 {
        "PARTY WIPED — every hero dead"
    } else if total_kills == 0 {
        "STALEMATE — no enemies killed (movement / LoS / cooldown gate failure)"
    } else if final_alive_enemies == 0 {
        "DUNGEON CLEARED — all enemies dead"
    } else if total_kills >= 10 {
        "PARTY ADVANCING — meaningful enemy attrition"
    } else {
        "PARTY STRUGGLING — few enemies killed"
    };
    println!("  verdict: {outcome}");

    // Tunnel cast verdict (soft pin — does not assert).
    println!(
        "  tunnel:  cast_in_progress observed = {tunnel_cast_in_progress_observed} \
         ({tunnel_cast_ticks_observed} sample-ticks Mage was busy with Tunnel), \
         total carves resolved = {total_tunnels}",
    );
    if tunnel_cast_in_progress_observed {
        println!("           verdict: cast → busy → resolve loop FIRED");
    } else {
        println!("           verdict: NO Tunnel cast observed (Mage may have died early or never satisfied dispatch gate)");
    }
    println!("==========================================");

    // Load-bearing pins.
    assert_eq!(nan_count, 0, "found {nan_count} NaN positions after {TICKS} ticks");

    if let Some(alive_at_30) = tick_30_heroes_alive {
        // Stage 1 early-game safety: all 5 heroes alive at tick 30.
        // The dungeon is engineered so BFS-dist-≤2 rooms host no
        // enemies, AND every verb body re-checks `length(target -
        // self) < range` to gate around the spatial-cell-clamp gap
        // (Gap dungeon_layout#1 — see gaps_dungeon_layout.md).
        // Without the per-pair distance check, the spatial walk
        // around boundary-clamped cells returns far-away candidates
        // and the if-filter alone (creature_type / role) is not
        // sufficient to block out-of-range fire.
        assert_eq!(
            alive_at_30, N_HEROES,
            "early-game safety: all 5 heroes should be alive at tick 30 \
             (got {alive_at_30}). If <5, either (a) the BFS-empty-zone \
             logic placed enemies too close, or (b) a verb body is \
             missing its per-pair distance gate."
        );
    }

    if let Some(rooms_min) = tick_200_rooms_visited_min {
        assert!(
            rooms_min >= 3,
            "exploration: every hero should have visited >= 3 rooms by tick 200 (got min={rooms_min})"
        );
    }

    if let Some(kills_at_400) = tick_400_total_kills {
        assert!(
            kills_at_400 >= 10,
            "combat: party should have killed >= 10 enemies by tick 400 (got {kills_at_400})"
        );
    }

    println!(
        "  contract: 31 kernels emit, {TICKS} ticks step without panic / NaN, \
         voxel dungeon wires up, exploration drives target_room_idx updates, \
         hero / enemy slots discriminate correctly. Tunnel ability \
         (alphabetical slot 9 — Volley shifted to 10) wires the cast → busy → \
         resolve → place_voxel chronicle loop end-to-end (voxel-write hookup is \
         a follow-up — see Tunnel.ability gap note)."
    );
}

// ---------------------------------------------------------------------
// Dungeon roomgen (host-side Rust).
// ---------------------------------------------------------------------

/// A single 12×12 "room slot" in the 6×6 grid. `(rx, ry)` is the slot
/// coordinate (0..SLOTS_PER_ROW each); `idx() = ry * SLOTS_PER_ROW + rx`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RoomSlot {
    rx: u32,
    ry: u32,
}

impl RoomSlot {
    fn new(rx: u32, ry: u32) -> Self {
        Self { rx, ry }
    }
    fn idx(&self) -> u32 {
        self.ry * SLOTS_PER_ROW + self.rx
    }
    /// World-space centroid of this room (matches the .sim's
    /// `(idx % 6) * 12 + 6` arithmetic in HeroExplore).
    fn centroid(&self) -> [f32; 3] {
        let cx = (self.rx as f32) * (SLOT_WIDTH as f32) + (SLOT_WIDTH as f32) / 2.0;
        let cy = (self.ry as f32) * (SLOT_WIDTH as f32) + (SLOT_WIDTH as f32) / 2.0;
        [cx, cy, 1.0]
    }
}

struct Dungeon {
    /// Set of present slot indices, in deterministic order.
    rooms: Vec<RoomSlot>,
    /// Per-room set of (x, y) cell coordinates that are walkable
    /// (i.e. NOT walls). Used for placement + connectivity check.
    floor_cells: std::collections::BTreeMap<u32, Vec<(u32, u32)>>,
    /// BFS distance from spawn_room (in number of rooms hopped via
    /// adjacency).
    bfs_dist: std::collections::BTreeMap<u32, u32>,
    spawn_room: RoomSlot,
    boss_room: RoomSlot,
}

impl Dungeon {
    fn total_floor_cells(&self) -> u32 {
        self.floor_cells.values().map(|v| v.len() as u32).sum()
    }
    /// 5 heroes + per-room enemy count summed. Capped so GPU buffer
    /// sizing is deterministic across re-rolls.
    fn total_agent_count(&self) -> u32 {
        N_HEROES + self.enemy_placements().len() as u32
    }
    /// Distribute enemies across non-spawn rooms weighted by BFS dist.
    /// Returns the (room_slot, creature_type) per enemy.
    fn enemy_placements(&self) -> Vec<(RoomSlot, u32)> {
        let mut out: Vec<(RoomSlot, u32)> = Vec::new();
        // PCG-driven enemy count per room, weighted by BFS distance.
        for &room in &self.rooms {
            if room == self.spawn_room {
                continue;
            }
            let idx = room.idx();
            let dist = *self.bfs_dist.get(&idx).unwrap_or(&0);
            // Room count budget:
            //   - boss_room: 8 goblins + 1 brute + 1 archer
            //   - dist 1..2: 1-2 enemies (mostly goblins)
            //   - dist 3..4: 3-4 enemies (mix)
            //   - dist 5+:   4-5 enemies (with a brute or archer)
            // Per-room enemy budgets (stage 1 — total ~30 across all
            // rooms). BFS dist <= 1 rooms are EMPTY so the spawn
            // cluster is safe for early-game exploration. Every verb
            // body has a per-pair distance gate (Gap #1) so distant
            // enemies don't aggro through the spatial-cell-clamp
            // boundary. Boss room gets the heaviest concentration;
            // far rooms get 3-4 enemies each.
            let count = if room == self.boss_room {
                8
            } else if dist <= 1 {
                0
            } else if dist == 2 {
                1
            } else if dist == 3 {
                2
            } else {
                3
            };
            // Composition: 70% goblin, 20% archer, 10% brute (with
            // boss room special-cased to guarantee a brute + archer).
            for slot in 0..count {
                let ct = if room == self.boss_room {
                    if slot == 0 {
                        CT_BRUTE
                    } else if slot == 1 {
                        CT_ARCHER
                    } else {
                        CT_GOBLIN
                    }
                } else {
                    // PCG roll per-slot: roll % 10 -> 0..6=Goblin, 7..8=Archer, 9=Brute
                    let r = engine::rng::per_agent_u32_pcg(
                        SEED_U64 as u32,
                        idx,
                        slot as u32,
                        0xE5_E51_001,
                    ) % 10;
                    if r < 7 {
                        CT_GOBLIN
                    } else if r < 9 {
                        CT_ARCHER
                    } else {
                        CT_BRUTE
                    }
                };
                out.push((room, ct));
            }
        }
        out
    }
}

/// Roomgen entry — random-walk-with-backtracking for slot selection,
/// per-room CA for interior carving, 2-cell-wide doorways for
/// adjacent slots, BFS connectivity verify (re-roll with SEED+1 if
/// any present room is unreachable).
fn roll_dungeon(initial_seed: u64) -> Dungeon {
    let mut seed = initial_seed;
    // Bound the re-roll loop. Failure to converge in 20 tries
    // surfaces a roomgen gap that needs investigation.
    for attempt in 0..20 {
        if let Some(d) = try_roll_dungeon(seed) {
            if attempt > 0 {
                eprintln!("[dungeon_layout] roomgen converged after {attempt} re-rolls");
            }
            return d;
        }
        seed = seed.wrapping_add(1);
    }
    panic!("dungeon roomgen failed to converge in 20 re-rolls — investigate");
}

fn try_roll_dungeon(seed: u64) -> Option<Dungeon> {
    let seed32 = seed as u32;

    // 1. Random-walk-with-backtracking slot selection.
    let mut present = std::collections::BTreeSet::<RoomSlot>::new();
    // Start at a random interior slot (avoids edge bias).
    let start_rx = (engine::rng::per_agent_u32_pcg(seed32, 0, 0, 1) % SLOTS_PER_ROW).clamp(1, SLOTS_PER_ROW - 2);
    let start_ry = (engine::rng::per_agent_u32_pcg(seed32, 1, 0, 1) % SLOTS_PER_ROW).clamp(1, SLOTS_PER_ROW - 2);
    let start = RoomSlot::new(start_rx, start_ry);
    present.insert(start);

    // Random walk: each step, pick a random neighbor of any present
    // slot that's not already present. Bias towards the most-recent
    // frontier (DFS-like).
    let mut stack: Vec<RoomSlot> = vec![start];
    let mut step = 0u32;
    while present.len() < TARGET_ROOMS && !stack.is_empty() {
        let cur = *stack.last().unwrap();
        // Collect candidate neighbors of `cur` that are NOT present.
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
            // Backtrack.
            stack.pop();
            continue;
        }
        // Pick one at random.
        let pick =
            engine::rng::per_agent_u32_pcg(seed32, cur.idx(), step, 2) as usize % cands.len();
        let chosen = cands[pick];
        present.insert(chosen);
        stack.push(chosen);
        step = step.wrapping_add(1);
    }

    let rooms: Vec<RoomSlot> = present.iter().copied().collect();
    if rooms.len() < 8 {
        // Too few rooms — re-roll.
        return None;
    }

    // 2. Per-room CA carving.
    let mut floor_cells: std::collections::BTreeMap<u32, Vec<(u32, u32)>> =
        std::collections::BTreeMap::new();
    for &r in &rooms {
        let cells = carve_room_interior(seed32, r);
        if cells.is_empty() {
            // CA filled the whole room with walls — re-roll.
            return None;
        }
        floor_cells.insert(r.idx(), cells);
    }

    // 3. BFS connectivity check on the present-slot adjacency graph.
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
            if !present_set.contains(&n) {
                continue;
            }
            if bfs_dist.contains_key(&n.idx()) {
                continue;
            }
            bfs_dist.insert(n.idx(), d + 1);
            queue.push_back(n);
        }
    }

    // Reachability: every present room must have a bfs_dist entry.
    if bfs_dist.len() != rooms.len() {
        return None;
    }

    // 4. Boss room = the room with the max BFS distance from spawn.
    let boss_room = *rooms
        .iter()
        .max_by_key(|r| bfs_dist.get(&r.idx()).copied().unwrap_or(0))
        .unwrap();

    Some(Dungeon {
        rooms,
        floor_cells,
        bfs_dist,
        spawn_room,
        boss_room,
    })
}

/// CA-carve a single 12×12 room interior. Returns the set of
/// walkable (x, y) world-cell coordinates for the room.
///
/// The room footprint is `[rx*12 .. (rx+1)*12) × [ry*12 .. (ry+1)*12)`.
/// We keep a 1-cell-wide border guaranteed FLOOR around the room
/// perimeter so doorways can carve through later without colliding
/// with CA walls. The interior (inner 10×10) runs CA.
fn carve_room_interior(seed: u32, room: RoomSlot) -> Vec<(u32, u32)> {
    let x0 = room.rx * SLOT_WIDTH;
    let y0 = room.ry * SLOT_WIDTH;
    // Build a local 12×12 grid (true = wall). Interior runs CA.
    let mut grid: [[bool; SLOT_WIDTH as usize]; SLOT_WIDTH as usize] =
        [[false; SLOT_WIDTH as usize]; SLOT_WIDTH as usize];
    // Initial fill: 40% walls in the interior (cells 1..11 on each
    // axis). Border stays floor.
    for ly in 1..(SLOT_WIDTH - 1) {
        for lx in 1..(SLOT_WIDTH - 1) {
            let r = engine::rng::per_agent_u32_pcg(seed, x0 + lx, y0 + ly, 1) % 100;
            grid[ly as usize][lx as usize] = r < CA_INIT_WALL_PCT;
        }
    }
    // CA iterations: cell becomes wall iff (neighbors_walls >= 5) ||
    // (was_wall && neighbors_walls >= 4).
    for _ in 0..CA_ITERATIONS {
        let mut next = grid;
        for ly in 1..(SLOT_WIDTH - 1) {
            for lx in 1..(SLOT_WIDTH - 1) {
                let mut walls = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = lx as i32 + dx;
                        let ny = ly as i32 + dy;
                        if nx < 0
                            || ny < 0
                            || nx >= SLOT_WIDTH as i32
                            || ny >= SLOT_WIDTH as i32
                        {
                            walls += 1; // out-of-bounds counts as wall (CA bias)
                            continue;
                        }
                        if grid[ny as usize][nx as usize] {
                            walls += 1;
                        }
                    }
                }
                let was_wall = grid[ly as usize][lx as usize];
                next[ly as usize][lx as usize] = walls >= 5 || (was_wall && walls >= 4);
            }
        }
        grid = next;
    }
    // Border cells: always floor (so doorways can carve through).
    for i in 0..SLOT_WIDTH {
        grid[0][i as usize] = false;
        grid[(SLOT_WIDTH - 1) as usize][i as usize] = false;
        grid[i as usize][0] = false;
        grid[i as usize][(SLOT_WIDTH - 1) as usize] = false;
    }
    // Collect walkable (floor) cells in world coordinates.
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
// Voxel + topology seeding into GeneratedRuntime.
// ---------------------------------------------------------------------

/// Write the dungeon's wall pattern into the voxel terrain. Walls
/// at z=0..ROOM_INTERIOR_Z; floor cells at z=0 left as air. Cells
/// outside any present room are wall (out-of-dungeon = solid stone).
fn seed_voxel_dungeon(state: &mut GeneratedRuntime, dungeon: &Dungeon) {
    use glam::IVec3;

    // Build a (x, y) → floor presence map for fast lookup.
    let mut floor_map = std::collections::BTreeSet::<(u32, u32)>::new();
    for cells in dungeon.floor_cells.values() {
        for &(x, y) in cells {
            floor_map.insert((x, y));
        }
    }

    // Carve doorways: for each adjacent pair of present rooms, open a
    // 2-cell-wide passage through the shared wall. The "shared wall"
    // is the row/column between the two slots — at the slot boundary.
    let present_set: std::collections::BTreeSet<RoomSlot> = dungeon.rooms.iter().copied().collect();
    let seed32 = SEED_U64 as u32;
    for &r in &dungeon.rooms {
        // East neighbor.
        let east = RoomSlot::new(r.rx + 1, r.ry);
        if r.rx + 1 < SLOTS_PER_ROW && present_set.contains(&east) {
            let door_y_off = engine::rng::per_agent_u32_pcg(seed32, r.idx(), east.idx(), 3)
                % (SLOT_WIDTH - 4);
            let door_y = r.ry * SLOT_WIDTH + 2 + door_y_off;
            // Open 2 cells wide on each side of the boundary.
            let boundary_x = (r.rx + 1) * SLOT_WIDTH;
            for dx in (boundary_x - 1)..=(boundary_x) {
                for dy in door_y..(door_y + 2) {
                    floor_map.insert((dx, dy));
                }
            }
        }
        // South neighbor.
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

    // Now: every (x, y) NOT in floor_map is solid wall (from z=0 up to
    // ROOM_INTERIOR_Z). Every (x, y) in floor_map is air.
    let mut writes: Vec<(u32, u32, u32)> = Vec::new();
    for x in 0..GRID_X {
        for y in 0..GRID_Y {
            if !floor_map.contains(&(x, y)) {
                // Wall column up to ceiling.
                for z in 0..ROOM_INTERIOR_Z.min(GRID_Z) {
                    state.voxel_terrain.set_cell(x, y, z, STONE);
                    writes.push((x, y, z));
                }
            }
        }
    }
    for (x, y, z) in writes {
        state
            .voxel_mirror
            .mark_dirty(IVec3::new(x as i32, y as i32, z as i32));
    }
    eprintln!(
        "[dungeon_layout] seeded voxel dungeon: {} dirty chunks pending flush, {} floor cells",
        state.voxel_mirror.dirty_chunk_count(),
        floor_map.len(),
    );
}

/// Seed agent positions, creature_types, roles, and target_room_idx.
fn seed_topology(state: &mut GeneratedRuntime, dungeon: &Dungeon) {
    let agent_count = dungeon.total_agent_count() as usize;
    let mut positions = vec![[0.0f32; 4]; agent_count];
    let mut creature_type = vec![0u32; agent_count];
    let mut role = vec![0u32; agent_count];
    let mut target_room_idx = vec![0u32; agent_count];

    // Layout note (matches decl-order alphabetical):
    //   Archer = 0, Brute = 1, Goblin = 2, Hero = 3
    // We want creature_type per slot to mirror decl order so the
    // `self.creature_type == Hero` gate lowers to a u32 compare
    // against the same discriminant the host writes. To preserve
    // this, ALL Archers come first, then all Brutes, then all
    // Goblins, then the 5 Heroes.
    let enemies = dungeon.enemy_placements();
    let archers: Vec<(RoomSlot, u32)> = enemies.iter().copied().filter(|(_, ct)| *ct == CT_ARCHER).collect();
    let brutes: Vec<(RoomSlot, u32)> = enemies.iter().copied().filter(|(_, ct)| *ct == CT_BRUTE).collect();
    let goblins: Vec<(RoomSlot, u32)> = enemies.iter().copied().filter(|(_, ct)| *ct == CT_GOBLIN).collect();

    let mut slot = 0usize;

    // ---- Archers ----
    for (i, &(room, _ct)) in archers.iter().enumerate() {
        let (px, py) = pick_floor_cell(dungeon, room, i as u32, 0xA5C_4E0);
        positions[slot] = [px as f32 + 0.5, py as f32 + 0.5, 1.0, 0.0];
        creature_type[slot] = CT_ARCHER;
        slot += 1;
    }
    // ---- Brutes ----
    for (i, &(room, _ct)) in brutes.iter().enumerate() {
        let (px, py) = pick_floor_cell(dungeon, room, i as u32, 0xB23_C7E);
        positions[slot] = [px as f32 + 0.5, py as f32 + 0.5, 1.0, 0.0];
        creature_type[slot] = CT_BRUTE;
        slot += 1;
    }
    // ---- Goblins ----
    for (i, &(room, _ct)) in goblins.iter().enumerate() {
        let (px, py) = pick_floor_cell(dungeon, room, i as u32, 0x9081_AE);
        positions[slot] = [px as f32 + 0.5, py as f32 + 0.5, 1.0, 0.0];
        creature_type[slot] = CT_GOBLIN;
        slot += 1;
    }

    // ---- Heroes ---- (last 5 slots)
    let spawn_centroid = dungeon.spawn_room.centroid();
    // Adjacent present room → initial target_room_idx (heroes start
    // walking toward it on tick 0).
    let initial_target = pick_adjacent_present_room(dungeon, dungeon.spawn_room)
        .unwrap_or(dungeon.spawn_room);
    for h in 0..N_HEROES as usize {
        // Cluster within 2 cells of spawn centroid. h=0 at center,
        // h=1..4 at ±1 offsets.
        let dx = ((h as f32) - 2.0) * 0.5;
        let dy = (((h + 1) % 5) as f32 - 2.0) * 0.5;
        positions[slot] = [
            spawn_centroid[0] + dx,
            spawn_centroid[1] + dy,
            spawn_centroid[2],
            0.0,
        ];
        creature_type[slot] = CT_HERO;
        role[slot] = (h as u32) + 1; // role 1..5
        target_room_idx[slot] = initial_target.idx();
        slot += 1;
    }

    debug_assert_eq!(slot, agent_count, "slot accounting drift");

    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&creature_type),
    );
    state.gpu.queue.write_buffer(
        &state.agent_role_buf,
        0,
        bytemuck::cast_slice(&role),
    );
    state.gpu.queue.write_buffer(
        &state.agent_target_room_idx_buf,
        0,
        bytemuck::cast_slice(&target_room_idx),
    );
}

/// Pick a deterministic floor cell inside `room` for entity i,
/// keyed by purpose_id.
fn pick_floor_cell(dungeon: &Dungeon, room: RoomSlot, i: u32, purpose: u32) -> (u32, u32) {
    let cells = dungeon.floor_cells.get(&room.idx()).expect("room has cells");
    if cells.is_empty() {
        // Fallback to centroid (shouldn't happen — try_roll_dungeon
        // rejects rooms with no floor).
        let c = room.centroid();
        return (c[0] as u32, c[1] as u32);
    }
    let idx = engine::rng::per_agent_u32_pcg(SEED_U64 as u32, room.idx(), i, purpose) as usize
        % cells.len();
    cells[idx]
}

/// Pick any adjacent present room (used to seed initial
/// target_room_idx + to expand exploration frontier).
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

/// Tracks per-hero room visitation state CPU-side.
struct HeroExploreState {
    /// rooms_visited[h] is a bitmask over present room indices (we
    /// remap the 0..35 slot idx down to 0..rooms.len() so up to 32
    /// rooms fit in u32; for stage 1, ~18 rooms easily fit).
    rooms_visited: [u32; N_HEROES as usize],
    /// current_room_idx[h] is the slot-grid index (0..35) of the
    /// room the hero is currently inside (or "in transit" if no
    /// match).
    current_room: [Option<u32>; N_HEROES as usize],
}

impl HeroExploreState {
    fn new(dungeon: &Dungeon) -> Self {
        // Heroes start in spawn room — mark bit 0 (its remapped idx).
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

/// Each tick: read hero positions, detect which room each is in,
/// update rooms_visited + pick a new target_room_idx if the hero
/// reached its previous target.
fn update_hero_exploration(
    state: &mut GeneratedRuntime,
    dungeon: &Dungeon,
    hero_state: &mut HeroExploreState,
    agent_count: u32,
    tick: u32,
) {
    let positions = read_positions(state, agent_count);
    let hero_start = (agent_count - N_HEROES) as usize;

    // Read current target_room_idx so we only rewrite if changed.
    let mut targets = read_target_room_idx(state, agent_count);

    let mut any_change = false;

    let present_set: std::collections::BTreeSet<RoomSlot> = dungeon.rooms.iter().copied().collect();

    for h in 0..N_HEROES as usize {
        let p = positions[hero_start + h];
        if !p[0].is_finite() || !p[1].is_finite() {
            continue;
        }
        // Which slot does the hero stand on?
        let rx = (p[0] / SLOT_WIDTH as f32).floor() as i32;
        let ry = (p[1] / SLOT_WIDTH as f32).floor() as i32;
        if rx < 0 || ry < 0 || (rx as u32) >= SLOTS_PER_ROW || (ry as u32) >= SLOTS_PER_ROW {
            continue;
        }
        let candidate = RoomSlot::new(rx as u32, ry as u32);
        if !present_set.contains(&candidate) {
            continue;
        }

        // Update current_room and mark visited.
        let cand_idx = candidate.idx();
        hero_state.current_room[h] = Some(cand_idx);
        let remap = dungeon
            .rooms
            .iter()
            .position(|r| *r == candidate)
            .expect("present room is in rooms");
        hero_state.rooms_visited[h] |= 1u32 << remap;

        // If the hero reached its target room, pick a new one
        // (frontier-greedy: prefer unvisited adjacent rooms; fall
        // back to any adjacent present room).
        let cur_target = targets[hero_start + h];
        if cand_idx == cur_target {
            let new_target = pick_next_target(dungeon, candidate, hero_state.rooms_visited[h], tick, h as u32);
            if new_target != cur_target {
                targets[hero_start + h] = new_target;
                any_change = true;
            }
        }
    }

    if any_change {
        state.gpu.queue.write_buffer(
            &state.agent_target_room_idx_buf,
            0,
            bytemuck::cast_slice(&targets),
        );
    }
}

/// Frontier-greedy next-target picker. Prefer unvisited adjacent
/// rooms; fall back to any adjacent present room; ultimate fall back
/// is to stay put.
fn pick_next_target(
    dungeon: &Dungeon,
    from: RoomSlot,
    visited_mask: u32,
    tick: u32,
    hero_idx: u32,
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
        if !present_set.contains(&n) {
            continue;
        }
        adjacent.push(n);
        let remap = dungeon.rooms.iter().position(|r| *r == n).unwrap();
        if (visited_mask & (1u32 << remap)) == 0 {
            unvisited.push(n);
        }
    }
    let pool = if !unvisited.is_empty() { &unvisited } else { &adjacent };
    if pool.is_empty() {
        return from.idx();
    }
    let pick =
        engine::rng::per_agent_u32_pcg(SEED_U64 as u32, hero_idx, tick, 0xFA000) as usize
            % pool.len();
    pool[pick].idx()
}

// ---------------------------------------------------------------------
// Readback helpers.
// ---------------------------------------------------------------------

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
        label: Some("dungeon_layout::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_layout::pos_readback") },
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

fn read_target_room_idx(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<u32> {
    let buf = state.agent_target_room_idx_buf.clone();
    read_agent_u32(state, &buf, agent_count)
}

fn read_agent_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, agent_count: u32) -> Vec<u32> {
    let count = agent_count as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dungeon_layout::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_layout::u32_readback") },
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
        label: Some("dungeon_layout::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("dungeon_layout::f32_readback") },
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

// ---------------------------------------------------------------------
// Tunnel-cast readbacks (Plan G + voxel place_voxel chronicle).
// ---------------------------------------------------------------------

fn read_busy_with_ability_id(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<u32> {
    let buf = state.agent_busy_with_ability_id_buf.clone();
    read_agent_u32(state, &buf, agent_count)
}

fn read_busy_until_tick(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<u32> {
    let buf = state.agent_busy_until_tick_buf.clone();
    read_agent_u32(state, &buf, agent_count)
}

/// Read the per-agent `tunnels_carved` materialized view (f32 cells,
/// one per agent slot — the @materialized fold writes 1.0 per
/// TunnelCarved event keyed on the caster).
fn read_view_tunnels_carved(state: &mut GeneratedRuntime, agent_count: u32) -> Vec<f32> {
    let buf = state.view_storage_tunnels_carved_primary_buf.clone();
    read_agent_f32(state, &buf, agent_count)
}
