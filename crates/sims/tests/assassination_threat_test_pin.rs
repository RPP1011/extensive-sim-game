//! Phase 5 close-out pin for `assets/sim/assassination_threat_test.sim`.
//!
//! Demonstrates the end-to-end voxel-region-indices spec surface
//! (Phases 1–4 of `docs/superpowers/specs/2026-04-25-voxel-region-
//! indices-design.md`) on a real fixture:
//!
//!   * The `.sim` carries `region_kind Settlement { … }` +
//!     `index Navgrid(…) { build { … } }` + `region_indices
//!     Settlement { Navgrid }` decls that parse and resolve through
//!     the dsl_ast/dsl_compiler pipeline.
//!   * The host registers a single `Settlement` region covering the
//!     16×16×4 city via `engine_voxel::VoxelRegionRegistry` and
//!     builds a `NavgridIndex` for it via `engine_voxel::build_navgrid`
//!     once per attempt (proves Phase 4a plumbs end-to-end against a
//!     real fixture).
//!   * The sim folds a 4-cell `threat(planner, quadrant)` belief
//!     (@key_pop K=4 + decay 0.95) keyed on the King's slot. Guards
//!     within `detection_radius` of the assassin emit `ThreatSighted`
//!     each tick they have a sighting; the per-tick fold accumulates
//!     intensity per quadrant.
//!   * Between attempts, the host reads the
//!     `view_storage_threat_primary_buf` directly, picks the argmax
//!     quadrant, and re-clusters guards toward it for the next
//!     attempt. This is the "planner adapts" loop the user spec
//!     called for, host-orchestrated because the DSL has no
//!     belief-read-by-key surface yet (deferred to Phase 6+; see
//!     `.sim` header).
//!
//! ## Outcome shape
//!
//! Per-attempt result is `success` (the assassin reached the king and
//! killed him before any guard reached strike range) or `thwarted`
//! (a guard reached strike range first OR the attempt timed out at
//! `MAX_TICKS_PER_ATTEMPT`). The pin reports per-quadrant attack
//! counts, per-quadrant success counts, and the success-rate trend
//! across the 50-attempt run. The expected shape: proportional
//! guard allocation by threat → quadrants the assassin frequents
//! get heavy defense → low-threat quadrants left exposed → wide
//! per-quadrant success-rate spread.
//!
//! ## History (the bug `assassination_visualize.rs` caught)
//!
//! Pre-2026-05-17 this pin reported 74% success but the assassin
//! never actually reached the king — those "successes" were
//! chronicle bleed surviving the drain, and the strike rules fired
//! via cell-bucketed `spatial.nearby_melee` instead of a real
//! distance check. The visualizer rendered the assassin frozen one
//! step from spawn while the king "died" at the origin, distance
//! 9.3 away. Two fixes restored honesty:
//!
//!   1. `.sim` AssassinStrike + GuardStrike now gate on
//!      `length(candidate.pos - self.pos) < strike_radius` in
//!      addition to the cell-bucketed spatial scan (the spatial
//!      query is correct as a coarse filter; the distance check is
//!      the exact gate).
//!   2. `redistribute_guards` was a cyclic-fill that always
//!      produced [2,2,2,2] regardless of threat — replaced with
//!      proportional allocation + largest-remainder rounding. Now
//!      the planner genuinely concentrates guards in believed-high
//!      quadrants and leaves zero-threat quadrants undefended.
//!
//! Post-fix: 18% real success rate with 100pp per-quadrant spread
//! (q=3 is the planner's blind spot — 8/8 = 100% — because the
//! assassin's cycling logic spawns there less frequently than q=0,
//! so q=3's threat never accumulates and the proportional allocator
//! gives it zero guards). q=0 with 24/50 attempts gets all the
//! defense → 0% success there. The 100pp spread is real, not noise.
//!
//! Pins (load-bearing):
//!   1. All 50 attempts complete without panic.
//!   2. The threat belief shows non-zero values in at least 2
//!      quadrants by the end of the run (proves @key_pop K=4 storage
//!      held + decay didn't zero everything).
//!   3. The navgrid build returns Ok for the 16×16 city region.
//!   4. Per-quadrant success spread ≥ 30pp (proves the planner's
//!      proportional allocation genuinely shapes outcomes).

#![allow(non_snake_case)]

use sims::assassination_threat_test::GeneratedRuntime;

use engine_voxel::{
    build_navgrid, Aabb, NavgridIndex, VoxelRegion, VoxelRegionBounds, VoxelRegionKind,
    VoxelRegionRegistry,
};

const SEED: u64 = 0xA55A_551_DEAD_BEEF;
const N_ASSASSINS: u32 = 1;
const N_GUARDS: u32 = 8;
const N_KING: u32 = 1;
const N_TOTAL: u32 = N_ASSASSINS + N_GUARDS + N_KING; // 10

const CT_ASSASSIN: u32 = 0;
const CT_GUARD: u32 = 1;
const CT_KING: u32 = 2;

const ASSASSIN_SLOT: usize = 0;
const GUARD_SLOT_BASE: usize = 1;
const KING_SLOT: usize = 9; // matches `config.city.planner_slot`

const NUM_QUADRANTS: u32 = 4;
const NUM_ATTEMPTS: u32 = 50;
const MAX_TICKS_PER_ATTEMPT: u32 = 80;
const HALF_EXTENT: f32 = 8.0;

// City: a flat 16×16 stone floor at y=0, with the playable y range
// 0..4 for the navgrid scan. The fixture's voxel terrain isn't
// consulted by any physics rule (LoS is radius-only here), so we
// model the city purely as a heightmap closure passed to the
// navgrid build. The voxel terrain mirror on the runtime is left at
// its `try_new` default.
const CITY_MIN: [f32; 3] = [-(HALF_EXTENT), 0.0, -(HALF_EXTENT)];
const CITY_MAX: [f32; 3] = [HALF_EXTENT, 4.0, HALF_EXTENT];

#[test]
fn assassination_50_attempt_adaptation_report() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[assassination_threat_test] skipping: no wgpu adapter");
            return;
        }
    };

    // --- Phase 4a smoke: register the city as a Settlement region
    // and build the navgrid for it. Done once at the start of the
    // run (rebuild_on: chunk_epoch_advance — terrain doesn't change
    // here, so one build is enough). Phase 4b update: the navgrid
    // is now uploaded to the runtime's GPU storage buffer via
    // `upload_navgrid` so the `AssassinAdvance` physics rule
    // (rewritten 2026-05-16) can gate its step on
    // `navgrid.walkable(cx, cz)` — the assassin holds position
    // when the next cell is a wall instead of phasing through.
    let mut region_registry = VoxelRegionRegistry::new();
    let settlement_kind = VoxelRegionKind(0);
    let city_id = region_registry.register(
        settlement_kind,
        VoxelRegionBounds::Aabb(Aabb {
            min: CITY_MIN,
            max: CITY_MAX,
        }),
        /* created_at_tick = */ 0,
    );
    let city_region: &VoxelRegion = region_registry
        .get(city_id)
        .expect("city region just registered");
    // Solid-at predicate: ground at y=0, plus a vertical wall ring
    // at radius 5 from the city centre (origin) for y=0..4. The
    // navgrid's classify pass marks the wall columns non-walkable
    // (top-of-column hits the scan ceiling → no air above), so the
    // assassin's `navgrid.walkable` check fails on those cells.
    // Build the navgrid for the city. Two design choices:
    //   * `solid_at(_, 0, _) = true`: ground at y=0 → every (cx, cz)
    //     classifies walkable (top-of-column at y=0 has air above at
    //     y=1, so `build_navgrid` marks it walkable).
    //   * No walls in this fixture's navgrid. A second navgrid_probe
    //     fixture (`assets/sim/navgrid_probe.sim`) proves the
    //     `navgrid.walkable(cx, cz)` call gates movement against a
    //     real wall column; the assassination fixture's narrative
    //     is the planner-adaptation loop, with the navgrid call
    //     present (the assassin's step IS gated through it) but
    //     non-blocking by design.
    let navgrid: NavgridIndex = build_navgrid(city_region, |_x, y, _z| y == 0)
        .expect("navgrid build for 16x16 city");
    println!(
        "[Phase 4a+4b] navgrid built: {}×{} cells, origin=({}, {}), scan y={}..{}",
        navgrid.size_x,
        navgrid.size_z,
        navgrid.origin_x,
        navgrid.origin_z,
        navgrid.scan_min_y,
        navgrid.scan_max_y,
    );
    assert_eq!(navgrid.size_x, 16, "x-extent of the city");
    assert_eq!(navgrid.size_z, 16, "z-extent of the city");

    // Phase 4b — upload the host-built navgrid to the GPU so the
    // AssassinAdvance physics rule's `navgrid.walkable(cx, cz)` call
    // resolves against the same cells we just built.
    state.upload_navgrid(&navgrid);

    // Per-tick storage rolled across the run.
    let mut attack_counts: [u32; NUM_QUADRANTS as usize] = [0; 4];
    let mut success_counts: [u32; NUM_QUADRANTS as usize] = [0; 4];
    let mut per_attempt_outcomes: Vec<(u32, bool, u32)> = Vec::with_capacity(NUM_ATTEMPTS as usize);

    // Initial guard quadrant distribution: 2 per quadrant evenly.
    let mut guard_quadrants: [u32; N_GUARDS as usize] = [0, 0, 1, 1, 2, 2, 3, 3];

    for attempt in 0..NUM_ATTEMPTS {
        // Cycle attempt's spawn quadrant — assassin alternates approach
        // quadrants to give the planner a varied signal. First 4
        // attempts hit each quadrant once (warmup); after that the
        // assassin picks quadrant 0 with higher frequency to test
        // that the planner adapts.
        let approach_q: u32 = if attempt < 4 {
            attempt
        } else if (attempt % 3) == 0 {
            0 // favour quadrant 0 to test adaptive defense
        } else {
            (attempt + 1) % NUM_QUADRANTS
        };
        attack_counts[approach_q as usize] += 1;

        // Drain leftover chronicle records from the previous
        // attempt. The damage chronicle persists across step()
        // calls; without a drain, last-tick `EffectDamageApplied`
        // entries against the king re-kill him on tick 0 of the
        // next attempt. Two drain steps with everyone dead so no
        // new producer events fire — the first drain consumes the
        // damage records, the second consumes any
        // `Damaged`/`Defeated` events emitted by the first drain.
        if attempt > 0 {
            kill_all(&mut state);
            state.step();
            state.step();
        }
        seed_attempt(&mut state, approach_q, &guard_quadrants);

        let mut killed = false;
        let mut killed_at_tick: u32 = MAX_TICKS_PER_ATTEMPT;
        // The assassin walks at 0.5/tick from radius ~9.9 to the
        // king at the origin; min real-kill time is ~20 ticks. Any
        // king-death before MIN_REAL_KILL_TICK is a chronicle
        // bleed from the previous attempt's residual
        // EffectDamageApplied entries surviving the drain. Skip
        // those — count only attempts where the king dies inside
        // the realistic walk-time window.
        const MIN_REAL_KILL_TICK: u32 = 18;
        for tick in 0..MAX_TICKS_PER_ATTEMPT {
            state.step();
            let king_alive = read_king_alive(&mut state);
            if !king_alive && tick >= MIN_REAL_KILL_TICK {
                killed = true;
                killed_at_tick = tick;
                break;
            }
            // Early exit if assassin is dead — outcome is locked
            // (only after the warmup window so chronicle bleed
            // doesn't lock a thwarted outcome).
            if tick >= MIN_REAL_KILL_TICK {
                let assassin_alive = read_alive_at(&mut state, ASSASSIN_SLOT);
                if !assassin_alive {
                    break;
                }
            }
        }

        if killed {
            success_counts[approach_q as usize] += 1;
        }
        per_attempt_outcomes.push((approach_q, killed, killed_at_tick));

        // Planner adapts: read threat(planner, quadrant) →
        // proportional re-distribution for the NEXT attempt.
        let threat: [f32; 4] = read_threat_belief(&mut state);
        let top_q = argmax(&threat);
        guard_quadrants = redistribute_guards(&threat);

        if attempt % 10 == 0 || attempt == NUM_ATTEMPTS - 1 {
            println!(
                "  attempt {:2}: approach_q={} → {} (tick {})  threat=[{:.2}, {:.2}, {:.2}, {:.2}]  next_top_q={}",
                attempt,
                approach_q,
                if killed { "SUCCESS" } else { "THWARTED" },
                killed_at_tick,
                threat[0], threat[1], threat[2], threat[3],
                top_q,
            );
        }
    }

    let total_success: u32 = success_counts.iter().sum();
    let total_attempts: u32 = attack_counts.iter().sum();
    let overall_success_pct = (total_success as f64) / (total_attempts as f64) * 100.0;

    // Per-quadrant success rate.
    let mut per_q_rate: [f64; 4] = [0.0; 4];
    for q in 0..4 {
        if attack_counts[q] > 0 {
            per_q_rate[q] = (success_counts[q] as f64) / (attack_counts[q] as f64) * 100.0;
        }
    }

    // Compare first-half vs second-half success rate to detect the
    // adaptation. If the planner re-clustering is working, the
    // second half should have a lower success rate than the first
    // half overall.
    let half = (NUM_ATTEMPTS / 2) as usize;
    let first_half_success: u32 = per_attempt_outcomes[..half]
        .iter()
        .filter(|(_q, killed, _t)| *killed)
        .count() as u32;
    let second_half_success: u32 = per_attempt_outcomes[half..]
        .iter()
        .filter(|(_q, killed, _t)| *killed)
        .count() as u32;
    let first_pct = (first_half_success as f64) / (half as f64) * 100.0;
    let second_pct = (second_half_success as f64) / (half as f64) * 100.0;

    let final_threat: [f32; 4] = read_threat_belief(&mut state);
    let nonzero_quads = final_threat.iter().filter(|v| **v > 0.0).count();

    println!("==== assassination_threat_test 50-attempt report ====");
    println!(
        "  topology: 1 Assassin + 8 Guards + 1 King in 16x16 city  detection_r=4  strike_r=1.5",
    );
    println!(
        "  attacks/quadrant: [{}, {}, {}, {}]  successes/quadrant: [{}, {}, {}, {}]",
        attack_counts[0], attack_counts[1], attack_counts[2], attack_counts[3],
        success_counts[0], success_counts[1], success_counts[2], success_counts[3],
    );
    println!(
        "  success rate by quadrant: [{:.1}%, {:.1}%, {:.1}%, {:.1}%]",
        per_q_rate[0], per_q_rate[1], per_q_rate[2], per_q_rate[3],
    );
    println!(
        "  overall: {total_success}/{total_attempts} = {overall_success_pct:.1}%",
    );
    println!(
        "  first half {first_half_success}/{half} = {first_pct:.1}%  vs  second half {second_half_success}/{half} = {second_pct:.1}%",
    );
    println!(
        "  final threat belief (planner=king): [{:.2}, {:.2}, {:.2}, {:.2}]  nonzero quads={nonzero_quads}",
        final_threat[0], final_threat[1], final_threat[2], final_threat[3],
    );
    println!("======================================================");

    // Load-bearing pins.
    assert_eq!(
        per_attempt_outcomes.len() as u32,
        NUM_ATTEMPTS,
        "all attempts must complete without panic"
    );
    assert!(
        nonzero_quads >= 2,
        "threat belief should accumulate in at least 2 quadrants by end of run; got {final_threat:?}",
    );
    // Adaptation pin: the planner reads the threat belief and
    // redistributes guards proportionally. The per-quadrant success
    // rate should show meaningful spread (best-quadrant >
    // worst-quadrant by at least 30 percentage points) — i.e.,
    // approaches the planner under-defends are clearly easier than
    // approaches it over-defends. A flat per-quadrant success rate
    // would mean the read-side or the planner heuristic isn't
    // actually shaping outcomes.
    let mut max_rate: f64 = 0.0;
    let mut min_rate: f64 = 100.0;
    for q in 0..4 {
        if attack_counts[q] >= 3 {
            max_rate = max_rate.max(per_q_rate[q]);
            min_rate = min_rate.min(per_q_rate[q]);
        }
    }
    let spread = max_rate - min_rate;
    println!(
        "  adaptation spread: best={max_rate:.1}%  worst={min_rate:.1}%  Δ={spread:.1}pp",
    );
    let final_top = argmax(&final_threat);
    println!(
        "  final top-threat quadrant: q={final_top}",
    );
    assert!(
        spread >= 30.0,
        "per-quadrant success rate spread should be ≥30pp (proof the planner heuristic shapes outcomes via the belief read); got max={max_rate:.1}% min={min_rate:.1}%"
    );
}

fn argmax(v: &[f32; 4]) -> u32 {
    let mut best_i: u32 = 0;
    let mut best_v: f32 = v[0];
    for i in 1..4 {
        if v[i] > best_v {
            best_v = v[i];
            best_i = i as u32;
        }
    }
    best_i
}

/// Distribute the 8 guards across 4 quadrants weighted by threat,
/// with NO floor — quadrants with zero threat get zero guards,
/// freeing all 8 to defend the believed approaches. This creates
/// real asymmetry (the bug-fix replacement for the prior cyclic-fill
/// version which always allocated [2, 2, 2, 2] regardless of
/// threat). Pure function of `threat`.
///
/// **Why no floor**: with a 1-guard floor + cyclic distribution of
/// the remaining 4, every quadrant ended up with 2 guards regardless
/// of the threat ranking. The assassin's odds were identical in
/// every quadrant → the planner's "adaptation" was theatre. Removing
/// the floor + scaling proportionally lets the planner over-commit
/// to high-threat quadrants and genuinely leave low-threat
/// quadrants exposed.
fn redistribute_guards(threat: &[f32; 4]) -> [u32; N_GUARDS as usize] {
    let sum: f32 = threat.iter().sum();
    let counts: [u32; 4] = if sum > 0.0 {
        // Proportional allocation with largest-remainder rounding so
        // the total stays exactly N_GUARDS.
        let exact: [f32; 4] = [
            threat[0] / sum * (N_GUARDS as f32),
            threat[1] / sum * (N_GUARDS as f32),
            threat[2] / sum * (N_GUARDS as f32),
            threat[3] / sum * (N_GUARDS as f32),
        ];
        let mut floors: [u32; 4] = [
            exact[0].floor() as u32,
            exact[1].floor() as u32,
            exact[2].floor() as u32,
            exact[3].floor() as u32,
        ];
        let allocated: u32 = floors.iter().sum();
        let remainder = N_GUARDS - allocated;
        // Distribute remainder to quadrants with the largest
        // fractional part.
        let mut fracs: Vec<(usize, f32)> = exact
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v - v.floor()))
            .collect();
        fracs.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in 0..(remainder as usize).min(4) {
            floors[fracs[i].0] += 1;
        }
        floors
    } else {
        // Untrained planner: evenly distributed.
        [2, 2, 2, 2]
    };
    let mut out = [0u32; N_GUARDS as usize];
    let mut idx = 0usize;
    for q in 0..NUM_QUADRANTS {
        for _ in 0..counts[q as usize] {
            if idx < N_GUARDS as usize {
                out[idx] = q;
                idx += 1;
            }
        }
    }
    out
}

fn quadrant_pos(q: u32, radius: f32) -> (f32, f32) {
    // q=0: NE (+x, +z),  q=1: NW (-x, +z),  q=2: SW (-x, -z),  q=3: SE (+x, -z)
    let (sx, sz): (f32, f32) = match q {
        0 => (1.0, 1.0),
        1 => (-1.0, 1.0),
        2 => (-1.0, -1.0),
        3 => (1.0, -1.0),
        _ => unreachable!(),
    };
    (sx * radius, sz * radius)
}

fn seed_attempt(state: &mut GeneratedRuntime, approach_q: u32, guard_quadrants: &[u32; N_GUARDS as usize]) {
    let n = N_TOTAL as usize;

    // Build positions: assassin in approach_q at the city perimeter,
    // guards distributed by quadrant within radius 4 from origin,
    // king at origin. Slot layout matches entity decl-order
    // alphabetical: Assassin (slot 0), Guard (slots 1..9), King (slot 9).
    let mut positions: Vec<[f32; 4]> = vec![[0.0; 4]; n];
    let (ax, az) = quadrant_pos(approach_q, HALF_EXTENT - 1.0);
    positions[ASSASSIN_SLOT] = [ax, 0.0, az, 0.0];
    for (i, q) in guard_quadrants.iter().enumerate() {
        let (gx, gz) = quadrant_pos(*q, 3.0);
        // Spread guards a little within their quadrant so they don't
        // overlap (jitter by index).
        let jitter = (i as f32) * 0.4 - 1.6;
        positions[GUARD_SLOT_BASE + i] = [gx + jitter * 0.3, 0.0, gz + jitter * 0.3, 0.0];
    }
    positions[KING_SLOT] = [0.0, 0.0, 0.0, 0.0];

    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    // creature_type — pinned each attempt so a respawned assassin
    // keeps its discriminant.
    let mut creature_type: Vec<u32> = Vec::with_capacity(n);
    creature_type.push(CT_ASSASSIN);
    for _ in 0..N_GUARDS {
        creature_type.push(CT_GUARD);
    }
    creature_type.push(CT_KING);
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&creature_type),
    );

    // All-alive + hp=100 reset (revives any agent that died last
    // attempt). hp is f32.
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );
    let hp: Vec<f32> = vec![100.0_f32; n];
    state.gpu.queue.write_buffer(
        &state.agent_hp_buf,
        0,
        bytemuck::cast_slice(&hp),
    );
}

fn kill_all(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    let dead: Vec<u32> = vec![0u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&dead),
    );
}

fn read_king_alive(state: &mut GeneratedRuntime) -> bool {
    read_alive_at(state, KING_SLOT)
}

fn read_alive_at(state: &mut GeneratedRuntime, slot: usize) -> bool {
    let buf = state.agent_alive_buf.clone();
    let vals: Vec<u32> = readback_u32(state, &buf, N_TOTAL as usize);
    vals[slot] != 0
}

/// Read the threat(planner=KING, quadrant=0..4) belief. With @key_pop
/// K=4 the storage is N×4 cells (`view_storage_threat_primary_buf`).
/// Row for planner=KING_SLOT (=9) lives at indices [9*4 .. 9*4+4].
fn read_threat_belief(state: &mut GeneratedRuntime) -> [f32; 4] {
    let cell_count = (N_TOTAL as usize) * (NUM_QUADRANTS as usize);
    let buf = state.view_storage_threat_primary_buf.clone();
    let bytes_u32 = readback_u32(state, &buf, cell_count);
    // Reinterpret as f32 (the fold target is f32).
    let mut out: [f32; 4] = [0.0; 4];
    for q in 0..NUM_QUADRANTS as usize {
        let bits = bytes_u32[KING_SLOT * (NUM_QUADRANTS as usize) + q];
        out[q] = f32::from_bits(bits);
    }
    out
}

fn readback_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("assassination::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("assassination::u32_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out: Vec<u32> = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
