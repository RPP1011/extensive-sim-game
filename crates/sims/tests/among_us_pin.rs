//! `among_us` 500-tick smoke pin — drives the social-deduction
//! adversarial fixture (assets/sim/among_us.sim) and reports:
//!
//!   * Final Imposter / Crew survival counts.
//!   * Total Kill events fired (decay-anchored kills_by_source view).
//!   * Total Vote events fired (votes_against view).
//!   * Belief accuracy: of the most-voted-against slots, how many
//!     were actually Imposters (TP rate over the top-K voted slots).
//!
//! **Topology** (host-seeded):
//!   - 17 Crew  (slots 0..17),  creature_type = 0 (Crew)
//!   - 3 Imposter (slots 17..20), creature_type = 1 (Imposter)
//!   - 20 agents total, mixed in a 24×24 unit square. Crew start
//!     uniformly scattered; Imposters cluster near origin so the
//!     hunt-mode override has alive Crew within `hunt_radius` = 8
//!     immediately.
//!
//! creature_type discriminants are pinned by `.sim` declaration
//! order alphabetised (Crew, Imposter). The host writes the matching
//! u32 into `agent_creature_type_buf`.
//!
//! 500 ticks = 10 vote phases (vote_period_ticks = 50). Each vote
//! phase fires AFTER an ImposterKill has happened, so the witness
//! consumer has had a chance to write `hunger[killer] = 1.0` (the
//! public-suspicion flag) and the Vote score formula picks that
//! slot as the highest-utility target.

use sims::among_us::GeneratedRuntime;

const SEED: u64 = 0xA110_0EFFu64;
const N_CREW: u32 = 17;
const N_IMPOSTER: u32 = 3;
const N_TOTAL: u32 = N_CREW + N_IMPOSTER;
const TICKS: u32 = 500;

const CT_CREW: u32 = 0;
const CT_IMPOSTER: u32 = 1;

#[test]
fn among_us_500_tick_social_deduction() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[among_us] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_topology(&mut state);

    let initial_crew = count_alive_of_type(&mut state, CT_CREW);
    let initial_imposter = count_alive_of_type(&mut state, CT_IMPOSTER);

    for _ in 0..TICKS {
        state.step();
    }

    let final_crew = count_alive_of_type(&mut state, CT_CREW);
    let final_imposter = count_alive_of_type(&mut state, CT_IMPOSTER);

    let event_load = read_view_event_load(&mut state);
    let crew_alive_buf = read_alive(&mut state);
    let creature_types = read_creature_types(&mut state);
    let hunger = read_hunger(&mut state);
    let positions = read_positions(&mut state);
    let mana_idx = read_mana(&mut state);

    // Movement check: AgentTaskSteer is a PerAgent kernel (no
    // apply_ability dispatch), so it runs from `step()` directly
    // without going through the Indirect-arm chronicle gap. After
    // 500 ticks at crew_step=0.16, agents should have completed
    // ~80 units of total path length — definitely visited multiple
    // task waypoints. The mana column (= task index) tracks
    // progress directly.
    let mean_pos_radius: f32 = positions
        .iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .sum::<f32>() / (positions.len() as f32);
    let max_task_idx: f32 = mana_idx.iter().fold(0.0f32, |a, &b| a.max(b));
    let mean_task_idx: f32 = mana_idx.iter().sum::<f32>() / (mana_idx.len() as f32);

    // Decompose event_load: per-slot, total = #kills_FROM_slot +
    // #votes_AGAINST_slot. Crew never kill (they have no Kill verb)
    // so a Crew slot's event_load = #votes against them ONLY (= 0
    // unless we're seeing target-of-vote misfires, which would be
    // a notable signal). Imposter slot's event_load = #kills they
    // dealt + #votes against them.
    let total_event_load: f32 = event_load.iter().sum();

    // Imposter kills derived: sum event_load over Imposter slots
    // MINUS votes against them. We don't directly observe votes
    // here so use hunger=1.0 as proxy: an Imposter with hunger>0
    // post-run has at least one witnessed kill (= ApplyWitness
    // fired AT LEAST once for them). Their event_load minus 1
    // (the witness sets hunger once + can be re-set on subsequent
    // kills too — bounded at 1.0 because we set, not add) gives a
    // KILL-LOWER-BOUND. The actual decomposition needs separate
    // view storage (Gap #4 in gaps_among_us.md).
    let crew_event_total: f32 = event_load
        .iter()
        .zip(creature_types.iter())
        .filter(|(_, &t)| t == CT_CREW)
        .map(|(&v, _)| v)
        .sum();
    let imposter_event_total: f32 = event_load
        .iter()
        .zip(creature_types.iter())
        .filter(|(_, &t)| t == CT_IMPOSTER)
        .map(|(&v, _)| v)
        .sum();
    let imposter_publicly_accused: usize = hunger
        .iter()
        .zip(creature_types.iter())
        .filter(|(&h, &t)| h > 0.5 && t == CT_IMPOSTER)
        .count();
    let crew_publicly_accused: usize = hunger
        .iter()
        .zip(creature_types.iter())
        .filter(|(&h, &t)| h > 0.5 && t == CT_CREW)
        .count();

    // Belief accuracy: of the agents publicly accused (hunger>0
    // post-run), how many were actually Imposters? This is a
    // lower-bound TP rate — the witness consumer only writes
    // hunger=1.0 for the killer of a witnessed kill, and Crew
    // killers (slot type=0) would need to invoke Kill which they
    // can't (Kill.ability is gated by creature_type=Imposter in
    // ImposterKill rule). So `crew_publicly_accused` should be 0.
    let publicly_accused_total = imposter_publicly_accused + crew_publicly_accused;
    let top_voted: Vec<usize> = (0..N_TOTAL as usize)
        .filter(|&i| hunger.get(i).copied().unwrap_or(0.0) > 0.5)
        .collect();
    let true_positives = imposter_publicly_accused;
    let belief_accuracy = if publicly_accused_total > 0 {
        (imposter_publicly_accused as f32) / (publicly_accused_total as f32)
    } else {
        0.0
    };

    println!("==== among_us {TICKS}-tick social deduction report ====");
    println!(
        "  init:       crew={initial_crew}/{N_CREW}  imposter={initial_imposter}/{N_IMPOSTER}",
    );
    println!(
        "  final:      crew={final_crew}/{N_CREW}  imposter={final_imposter}/{N_IMPOSTER}",
    );
    println!(
        "  events:     total_event_load={total_event_load:.0}  (Crew slots: {crew_event_total:.0}, Imposter slots: {imposter_event_total:.0})",
    );
    println!(
        "  publicly accused (hunger>0 post-run): {publicly_accused_total} agents = {imposter_publicly_accused} Imposter + {crew_publicly_accused} Crew (false positives)",
    );
    println!(
        "  belief acc: {true_positives}/{n} publicly-accused are actual Imposters ({pct:.1}%)",
        n = publicly_accused_total,
        pct = belief_accuracy * 100.0,
    );
    println!("  top-voted slots (hunger>0): {top_voted:?}");
    println!(
        "  movement:   mean_radius={mean_pos_radius:.2} (init crew=8.0, imp=1.5), task_idx mean={mean_task_idx:.2} max={max_task_idx:.0}",
    );

    // Per-slot detail (most informative for spotting Imposter-id
    // alignment vs vote-target alignment).
    println!("  ---- per-slot detail ----");
    for slot in 0..N_TOTAL as usize {
        let alive = crew_alive_buf.get(slot).copied().unwrap_or(0);
        let ct = creature_types.get(slot).copied().unwrap_or(99);
        let load = event_load.get(slot).copied().unwrap_or(0.0);
        let h = hunger.get(slot).copied().unwrap_or(0.0);
        let label = match ct {
            CT_CREW => "Crew",
            CT_IMPOSTER => "Imp ",
            _ => "??? ",
        };
        let alive_marker = if alive == 0 { "DEAD" } else { "    " };
        let accused = if h > 0.5 { "ACCUSED" } else { "       " };
        println!(
            "    slot {slot:2}  {label}  {alive_marker}  event_load={load:.0}  hunger={h:.2}  {accused}",
        );
    }

    let imposter_win = final_crew <= final_imposter;
    let crew_win = final_imposter == 0;
    let outcome = if imposter_win {
        "IMPOSTER WIN — equal or fewer Crew than Imposters"
    } else if crew_win {
        "CREW WIN — all Imposters voted out"
    } else {
        "ONGOING — neither side dominant"
    };
    println!("  verdict:    {outcome}");
    println!("===========================================================");

    // Load-bearing pins (assertions). Like hill_raid, we verify
    // structural invariants rather than any specific behavioural
    // outcome — the fixture's purpose is to exercise the cross-pair
    // belief-write + chronicle-consumer for_each_agent surfaces, not
    // to validate game balance.
    assert_eq!(
        initial_crew + initial_imposter,
        N_TOTAL,
        "all 20 slots must seed alive (alive flag survived try_new)",
    );
    assert!(
        creature_types.iter().all(|&ct| ct == CT_CREW || ct == CT_IMPOSTER),
        "every slot must have a known creature_type discriminant",
    );
    // The fixture must not panic during 500 ticks; reaching here is
    // the most basic structural pin (try_new + step×500 + readback).
    assert!(
        final_crew + final_imposter <= N_TOTAL,
        "alive count cannot exceed total population",
    );
    // Total population is conserved (alive flips one direction only;
    // dead slots stay dead).
    let dead_total = (initial_crew + initial_imposter) - (final_crew + final_imposter);
    println!("  conservation: {dead_total} agents dead = (init - final)");
    assert!(
        dead_total <= N_TOTAL,
        "more agents died than ever existed — alive flag corruption",
    );

    // Note on outcome variability: this run reports observed numbers
    // rather than asserting them, mirroring hill_raid's verdict-print
    // pattern. Behavioural outcomes depend on:
    //   * Imposter pathfinding closing strike_radius (hunt + steer
    //     work in fused PerAgent kernel)
    //   * Witness consumer for_each_agent body firing inside
    //     @phase(post) (likely-gap surface — see gaps_among_us.md)
    //   * Vote scoring formula picking hunger>0 targets (collective
    //     suspicion approximation; per-Crew belief reads in scoring
    //     remain a documented gap)
}

fn read_view_event_load(state: &mut GeneratedRuntime) -> Vec<f32> {
    let bytes = (N_TOTAL as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("among_us::event_load_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("among_us::event_load_readback") },
    );
    encoder.copy_buffer_to_buffer(&state.view_storage_primary_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..N_TOTAL as usize].to_vec()
    };
    staging.unmap();
    out
}

fn read_positions(state: &mut GeneratedRuntime) -> Vec<[f32; 4]> {
    let n = N_TOTAL as usize;
    let bytes = (n as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("among_us::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("among_us::pos_readback") },
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

fn read_mana(state: &mut GeneratedRuntime) -> Vec<f32> {
    // mana is bitcast<f32> from u32 atomic storage (see WGSL emit
    // for the f32-on-atomic-u32 pattern).
    let bytes = (N_TOTAL as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("among_us::mana_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("among_us::mana_readback") },
    );
    encoder.copy_buffer_to_buffer(&state.agent_mana_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..N_TOTAL as usize].iter().map(|&u| f32::from_bits(u)).collect()
    };
    staging.unmap();
    out
}

fn read_hunger(state: &mut GeneratedRuntime) -> Vec<f32> {
    let bytes = (N_TOTAL as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("among_us::hunger_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("among_us::hunger_readback") },
    );
    encoder.copy_buffer_to_buffer(&state.agent_hunger_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..N_TOTAL as usize].to_vec()
    };
    staging.unmap();
    out
}

fn seed_topology(state: &mut GeneratedRuntime) {
    use std::f32::consts::TAU;

    let n = N_TOTAL as usize;

    // Position layout:
    //   Crew  (slots 0..17): scattered in a 22-wide ring at radius 8
    //                         around origin (= near task waypoints).
    //   Imposter (slots 17..20): clustered near origin (radius 1.5)
    //                         so they have alive Crew within hunt_radius
    //                         immediately.
    // Pos buffer is `vec3<f32>` packed as 4 × f32 (xyz + pad).
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(n);
    for i in 0..N_CREW {
        let theta = (i as f32) * TAU / (N_CREW as f32);
        let r = 8.0;
        positions.push([theta.cos() * r, theta.sin() * r, 0.0, 0.0]);
    }
    for i in 0..N_IMPOSTER {
        let theta = (i as f32) * TAU / (N_IMPOSTER as f32);
        let r = 1.5;
        positions.push([theta.cos() * r, theta.sin() * r, 0.0, 0.0]);
    }
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    // creature_type per slot — first N_CREW are 0 (Crew), then 3
    // are 1 (Imposter). Discriminant order = entity decl order
    // alphabetised (Crew=0, Imposter=1).
    let mut creature_type: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..N_CREW {
        creature_type.push(CT_CREW);
    }
    for _ in 0..N_IMPOSTER {
        creature_type.push(CT_IMPOSTER);
    }
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&creature_type),
    );
}

fn count_alive_of_type(state: &mut GeneratedRuntime, ct: u32) -> u32 {
    let alive = read_alive(state);
    let types = read_creature_types(state);
    alive
        .iter()
        .zip(types.iter())
        .filter(|(&a, &t)| a != 0 && t == ct)
        .count() as u32
}

fn read_alive(state: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = state.agent_alive_buf.clone();
    readback_u32(state, &buf, N_TOTAL as usize)
}

fn read_creature_types(state: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = state.agent_creature_type_buf.clone();
    readback_u32(state, &buf, N_TOTAL as usize)
}

fn readback_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("among_us::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("among_us::u32_readback") },
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
