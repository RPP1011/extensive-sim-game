//! `detective_investigation` smoke pin — drives the information-asymmetric
//! belief-accumulation fixture for 1000 ticks at N=18 (3 Detectives + 15
//! Suspects, 3 of which are pre-marked guilty by the seed) and reports:
//!   * True positives (guilty accused) / false positives (innocent
//!     accused) per detective.
//!   * Per-detective accuracy = correct / total.
//!   * Final pair-keyed `evidence(d, s)` view summary stats — total
//!     across all (detective, suspect) cells, max single-cell value,
//!     diagonal-vs-off-diagonal split. Stress-tests view storage
//!     stability across 1000 ticks of monotonic atomic writes with no
//!     decay; if the buffer is under-sized or the atomicAdd path leaks,
//!     this pin surfaces the failure mode (panic on readback, NaN, or
//!     wildly out-of-range values).
//!   * Total Witnessed events fired (sanity check ObserveAndAccrue).
//!   * Total Accused events fired (sanity check the verb cascade).
//!
//! **Topology** (host-seeded):
//! - 3 Detectives at slots 0..3, positions on a small triangle near origin
//!   (within spatial.nearby_targets radius of every Suspect).
//! - 15 Suspects at slots 3..18, distributed on a circle at radius 8
//!   around the centre. Slots 5, 9, 13 are GUILTY (host writes
//!   `cooldown_next_ready_tick = 99` for those slots; all others get
//!   the default 0). The .sim's ObserveAndAccrue rule reads the
//!   column to bias the per-(detective, suspect) RNG threshold.
//!
//! creature_type discriminants are pinned by `.sim` declaration order
//! (`EntityRef.0` in `crates/dsl_ast/src/resolve.rs`). The
//! `detective_investigation.sim` entities are declared alphabetically —
//! Detective, Suspect — so:
//!   Detective = 0, Suspect = 1
//!
//! **What's load-bearing vs informational.** This is an adversarial
//! fixture by design: the goal is to surface compiler gaps. The pin's
//! load-bearing assertions cover what the auto-emit pipeline IS
//! expected to support today (kernel emit, step() runs, view storage
//! sized + readable). Behavioural numbers — accuracy %, true-positive
//! rate, etc. — are reported but only soft-asserted (no panic on
//! mismatch) so a successful 1000-tick drain that surfaces a NEW gap
//! still produces a useful report.

use sims::detective_investigation::GeneratedRuntime;

const SEED: u64 = 0xDE7EC71_5;
const N_DETECTIVES: u32 = 3;
const N_SUSPECTS: u32 = 15;
const N_TOTAL: u32 = N_DETECTIVES + N_SUSPECTS;
const TICKS: u32 = 1000;

const CT_DETECTIVE: u32 = 0;
const CT_SUSPECT: u32 = 1;

// Suspect slots 5, 9, 13 are GUILTY (= absolute slot ids, not relative
// to N_DETECTIVES). The host writes `cooldown_next_ready_tick = 99`
// for those slots; the .sim rule reads the column to bias the RNG
// threshold.
const GUILTY_SLOTS: [u32; 3] = [5, 9, 13];
const GUILTY_SENTINEL: u32 = 99;

#[test]
fn detective_investigation_1000_tick_drain() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[detective_investigation] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_topology(&mut state);

    let initial_detectives_alive = count_alive_of_type(&mut state, CT_DETECTIVE);
    let initial_suspects_alive = count_alive_of_type(&mut state, CT_SUSPECT);

    // Sanity check: positions wrote successfully.
    let positions = read_positions(&mut state);
    println!("  pos[0] (detective slot 0): {p0:?}", p0 = positions[0]);
    println!("  pos[3] (suspect slot 3): {p3:?}", p3 = positions[3]);

    // Run for a small handful of ticks first and read the event tail
    // and view storage to see whether anything's flowing.
    for warmup_tick in 0..5 {
        state.step();
        let tail = read_event_tail(&mut state);
        let evidence_snapshot = read_evidence_view(&mut state);
        let nonzero_cells: u32 = evidence_snapshot.iter().filter(|&&c| c > 0).count() as u32;
        let max_cell = evidence_snapshot.iter().copied().max().unwrap_or(0);
        println!(
            "  warmup tick {warmup_tick}: event_tail={tail}  evidence cells nonzero={nonzero_cells} max={max_cell}",
        );
    }

    for _ in 5..TICKS {
        state.step();
    }

    let final_detectives = count_alive_of_type(&mut state, CT_DETECTIVE);
    let final_suspects = count_alive_of_type(&mut state, CT_SUSPECT);

    let evidence = read_evidence_view(&mut state);
    let accusation_count = read_view_per_agent(&mut state, "view_storage_accusation_count_primary_buf")
        .unwrap_or_else(|| read_first_view_storage_per_agent(&mut state));
    let total_witnessed = read_total_witnessed(&mut state).unwrap_or_default();

    // -- Per-detective accuracy. --
    // We don't have a per-(d, s) accusation log on-GPU today (the
    // chronicle writes don't separate Accuse from Investigate /
    // Observe — see gap doc). Approximate per-detective accuracy
    // using the FINAL evidence(d, s) cells as a proxy for "who would
    // d accuse if forced to": for each detective, take the max-cell
    // suspect; mark "correct" if that suspect is guilty.
    let mut correct_top1 = 0u32;
    let mut total_predictions = 0u32;
    let mut detective_reports: Vec<DetectiveReport> = Vec::new();
    for d in 0..N_DETECTIVES {
        let mut best_cell = 0u32;
        let mut best_suspect: Option<u32> = None;
        for s in N_DETECTIVES..N_TOTAL {
            let cell = evidence_cell(&evidence, d, s);
            if cell > best_cell {
                best_cell = cell;
                best_suspect = Some(s);
            }
        }
        let pred = best_suspect.unwrap_or(0);
        let pred_guilty = GUILTY_SLOTS.contains(&pred);
        if best_suspect.is_some() {
            total_predictions += 1;
            if pred_guilty {
                correct_top1 += 1;
            }
        }
        // Tally per-detective evidence summary.
        let mut guilty_cell_sum = 0u64;
        let mut innocent_cell_sum = 0u64;
        for s in N_DETECTIVES..N_TOTAL {
            let cell = evidence_cell(&evidence, d, s);
            if GUILTY_SLOTS.contains(&s) {
                guilty_cell_sum += cell as u64;
            } else {
                innocent_cell_sum += cell as u64;
            }
        }
        detective_reports.push(DetectiveReport {
            slot: d,
            top1_pred: pred,
            top1_pred_correct: pred_guilty,
            top1_cell_value: best_cell,
            guilty_cell_sum,
            innocent_cell_sum,
            accusations_logged: accusation_count
                .get(d as usize)
                .copied()
                .unwrap_or(0.0),
            witnessed_total: total_witnessed.get(d as usize).copied().unwrap_or(0.0),
        });
    }

    // -- Aggregate evidence-view stress stats. --
    let mut max_cell = 0u32;
    let mut total_cells = 0u64;
    let mut nonzero_cells = 0u64;
    for d in 0..N_DETECTIVES {
        for s in 0..N_TOTAL {
            let cell = evidence_cell(&evidence, d, s);
            total_cells += cell as u64;
            if cell > 0 {
                nonzero_cells += 1;
            }
            if cell > max_cell {
                max_cell = cell;
            }
        }
    }
    // Sanity: any cell value > 2 * TICKS would imply duplicate writes
    // or memory corruption (each tick can contribute at most 1 to a
    // (d, s) cell because the rule emits a single Witnessed per pair
    // per tick).
    let cell_overflow_suspect = max_cell > 2 * TICKS;

    let total_accusations: f32 = accusation_count.iter().sum();
    let total_witnessed_sum: f32 = total_witnessed.iter().sum();

    println!("==== detective_investigation 1000-tick report ====");
    println!(
        "  init:    detectives={initial_detectives_alive}/{N_DETECTIVES}  suspects={initial_suspects_alive}/{N_SUSPECTS}",
    );
    println!(
        "  final:   detectives={final_detectives}/{N_DETECTIVES}  suspects={final_suspects}/{N_SUSPECTS}",
    );
    println!("  guilty slots (host-seeded): {GUILTY_SLOTS:?}");
    println!();
    println!("  -- per-detective top-1 belief vs guilty truth --");
    for r in &detective_reports {
        let verdict = if r.top1_pred_correct { "correct" } else { "WRONG" };
        println!(
            "    detective {d}: top1 suspect = slot {pred:>2} (cell={cell:>4})  {verdict}  acc-logged={accs:.0}  witnessed={wit:.0}  guilty-cell-sum={gcs}  innocent-cell-sum={ics}",
            d = r.slot,
            pred = r.top1_pred,
            cell = r.top1_cell_value,
            verdict = verdict,
            accs = r.accusations_logged,
            wit = r.witnessed_total,
            gcs = r.guilty_cell_sum,
            ics = r.innocent_cell_sum,
        );
    }
    let accuracy_pct = if total_predictions > 0 {
        100.0 * (correct_top1 as f64) / (total_predictions as f64)
    } else {
        0.0
    };
    println!(
        "  top-1 belief accuracy: {correct_top1}/{total_predictions} ({accuracy_pct:.1}%)",
    );
    println!();
    println!("  -- evidence-view storage stress (1000-tick atomicAdd, no decay) --");
    println!("    max single (d, s) cell value: {max_cell}  (overflow-suspect: {cell_overflow_suspect})");
    println!("    nonzero cells: {nonzero_cells}/{total} (sparsity = {density:.1}%)",
        total = (N_DETECTIVES as u64) * (N_TOTAL as u64),
        density = 100.0 * (nonzero_cells as f64) / ((N_DETECTIVES as u64 * N_TOTAL as u64) as f64),
    );
    println!("    total accumulated evidence (sum across all cells): {total_cells}");
    println!("    total Witnessed events fired (sum of total_witnessed): {total_witnessed_sum:.0}");
    println!("    total Accused events fired (sum of accusation_count): {total_accusations:.0}");
    println!("=======================================================");

    // ---- Load-bearing pins (structural / contract) ----
    // 1. Initial seeding survived try_new + the first step.
    assert_eq!(initial_detectives_alive, N_DETECTIVES, "detective seed didn't survive init");
    assert_eq!(initial_suspects_alive, N_SUSPECTS, "suspect seed didn't survive init");

    // 2. No agent died (Accuse's nominal `damage 0.1` is enough to
    //    kill a 0-hp suspect over many hits; the seed below stamps
    //    hp=10000 to keep them alive across the run).
    assert!(
        final_detectives == initial_detectives_alive,
        "no detective should die — verify the apply_ability damage isn't accumulating beyond seeded hp"
    );

    // 3. Pair-keyed view storage didn't catastrophically overflow.
    //    The maximum legitimate per-cell value over 1000 ticks at the
    //    guilty rate (30%) is ~300; over the innocent rate (5%) is
    //    ~50. A cell exceeding 2× TICKS = 2000 would imply duplicate
    //    writes or memory corruption.
    assert!(
        !cell_overflow_suspect,
        "evidence cell overflow suspected: max cell = {max_cell} > 2 × TICKS = {ceiling}; likely view-storage sizing or atomic-write issue",
        ceiling = 2 * TICKS,
    );

    // ---- Soft pins (behavioural / informational) ----
    // These document expectations but don't fail — the goal of the
    // adversarial fixture is to surface gaps, not to enforce semantics.
    if total_witnessed_sum == 0.0 {
        println!(
            "  NOTE: zero Witnessed events fired — ObserveAndAccrue rule didn't propagate.\n        Possible causes: spatial.nearby_targets returns no candidates,\n        rng.action() lowering broken in PerAgent dispatch, or\n        the fused-kernel order swallowed the emit.",
        );
    }
    // Load-bearing: the chronicle re-emit physics block (`on
    // EffectDamageApplied → emit Accused`) MUST consume the verb
    // cascade's apply_ability outputs. Pre-2026-05-12 the schedule
    // ordered the chronicle consumer BEFORE the producer (kind-aware
    // edge fix) and total_accusations stayed at 0 across 1000 ticks.
    // Post-fix the load-bearing assertion confirms the in-step GPU
    // producer/consumer chain wires up correctly.
    assert!(
        total_accusations > 0.0,
        "zero Accused events after {TICKS} ticks — chronicle re-emit physics rule \
         (`physics ApplyDamageFromChronicle` consuming EffectDamageApplied) didn't \
         fire. Pre-2026-05-12 kind-aware schedule fix this was the symptom of the \
         in-step GPU producer/consumer ordering gap (consumers ran before producers, \
         `cfg.event_count = 0` snap missed the producer's atomicAdd). Now the \
         assertion is load-bearing: any regression in the producer-before-consumer \
         schedule order will trip it.",
    );
    if accuracy_pct > 0.0 && accuracy_pct < 33.3 {
        println!(
            "  NOTE: top-1 belief accuracy ({accuracy_pct:.1}%) is below the\n        baseline of 3/15 = 20% (random pick). Investigate whether the\n        guilty-bit channel is being read correctly by ObserveAndAccrue.",
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct DetectiveReport {
    slot: u32,
    top1_pred: u32,
    top1_pred_correct: bool,
    top1_cell_value: u32,
    guilty_cell_sum: u64,
    innocent_cell_sum: u64,
    accusations_logged: f32,
    witnessed_total: f32,
}

/// Linear index into the evidence pair-keyed view.
/// The fold body writes `view_storage_primary[detective * agent_cap + suspect]`
/// — agent_cap = N_TOTAL.
fn evidence_cell(evidence: &[u32], detective: u32, suspect: u32) -> u32 {
    let idx = (detective as usize) * (N_TOTAL as usize) + (suspect as usize);
    evidence.get(idx).copied().unwrap_or(0)
}

fn seed_topology(state: &mut GeneratedRuntime) {
    use std::f32::consts::TAU;

    let n = N_TOTAL as usize;

    // Position layout: detectives in a small triangle near origin,
    // suspects on a circle at radius 8 (well within the 50-unit
    // observe_radius so spatial.nearby_targets returns every suspect
    // for every detective).
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(n);
    for i in 0..N_DETECTIVES {
        let theta = (i as f32) * TAU / (N_DETECTIVES as f32);
        let r = 1.5;
        positions.push([theta.cos() * r, theta.sin() * r, 0.0, 0.0]);
    }
    for i in 0..N_SUSPECTS {
        let theta = (i as f32) * TAU / (N_SUSPECTS as f32);
        let r = 8.0;
        positions.push([theta.cos() * r, theta.sin() * r, 0.0, 0.0]);
    }
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    // creature_type per slot — Detective for the first 3, Suspect
    // for the next 15. Discriminants are alphabetical decl order.
    let mut creature_type: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..N_DETECTIVES {
        creature_type.push(CT_DETECTIVE);
    }
    for _ in 0..N_SUSPECTS {
        creature_type.push(CT_SUSPECT);
    }
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&creature_type),
    );

    // alive=1 for every slot. The .sim has no `init { alive: 1 }`
    // block (we deliberately leave it blank so the host stamps the
    // pattern explicitly here). A blank init defaults agent_alive to
    // 0; without this stamp the rule's `self.alive` filter rejects
    // every agent and the run is silent.
    let alive: Vec<u32> = vec![1u32; n];
    state.gpu.queue.write_buffer(
        &state.agent_alive_buf,
        0,
        bytemuck::cast_slice(&alive),
    );

    // hp = 10000 for every slot — Accuse's nominal `damage 0.1` per
    // hit accumulates over many ticks; keep hp well above the
    // 1000-tick worst-case drain (~100 hits × 0.1 = 10 hp) so no
    // suspect dies and the per-tick accusation surface stays open.
    let hp: Vec<f32> = vec![10000.0; n];
    state.gpu.queue.write_buffer(
        &state.agent_hp_buf,
        0,
        bytemuck::cast_slice(&hp),
    );

    // Stamp the guilty sentinel into cooldown_next_ready_tick for
    // the 3 guilty suspect slots. The .sim's ObserveAndAccrue rule
    // reads this column to branch the per-(detective, suspect) RNG
    // threshold — guilty=30%, innocent=5%.
    let mut cooldown: Vec<u32> = vec![0u32; n];
    for &slot in &GUILTY_SLOTS {
        cooldown[slot as usize] = GUILTY_SENTINEL;
    }
    state.gpu.queue.write_buffer(
        &state.agent_cooldown_next_ready_tick_buf,
        0,
        bytemuck::cast_slice(&cooldown),
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

fn read_event_tail(state: &mut GeneratedRuntime) -> u32 {
    let buf = state.event_ring.tail().clone();
    let bytes = 16u64;
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("detective_investigation::event_tail_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("detective_investigation::event_tail_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, 4);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let val = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[0]
    };
    staging.unmap();
    val
}

fn read_positions(state: &mut GeneratedRuntime) -> Vec<[f32; 4]> {
    let n = N_TOTAL as usize;
    let bytes = (n as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("detective_investigation::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("detective_investigation::pos_readback") },
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

fn read_alive(state: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = state.agent_alive_buf.clone();
    readback_u32(state, &buf, N_TOTAL as usize)
}

fn read_creature_types(state: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = state.agent_creature_type_buf.clone();
    readback_u32(state, &buf, N_TOTAL as usize)
}

/// Read the pair-keyed `evidence(detective, suspect)` view. The
/// fold body writes `view_storage_primary[detective * agent_cap +
/// suspect]` so the buffer is sized N² u32 cells (N = agent_count).
/// We read the full N² slice; the report extracts (d in 0..3, s in
/// 0..N) cells via `evidence_cell()`.
fn read_evidence_view(state: &mut GeneratedRuntime) -> Vec<u32> {
    let n2 = (N_TOTAL as usize) * (N_TOTAL as usize);
    // Per-view storage post the aliasing-gap fix — each view
    // (accusation_count, evidence, total_witnessed) now has its own
    // backing buffer. Read the pair-keyed `evidence` slab directly.
    let buf = state.view_storage_evidence_primary_buf.clone();
    readback_u32(state, &buf, n2)
}

/// Per-view stub kept for callers that pass a runtime-string name.
/// With per-view buffers landed, callers should prefer typed direct
/// field accesses (`state.view_storage_<view>_primary_buf`); this
/// helper stays as a no-op shim for back-compat.
#[allow(dead_code)]
fn read_view_buffer_by_name(
    _state: &mut GeneratedRuntime,
    _name: &str,
    _n: usize,
) -> Option<Vec<u32>> {
    None
}

/// Per-view stub kept for callers that pass a runtime-string name.
fn read_view_per_agent(_state: &mut GeneratedRuntime, _field_name: &str) -> Option<Vec<f32>> {
    None
}

/// Read the per-agent `accusation_count` view as f32 — one of the
/// two per-agent views in this fixture. Per-view storage post the
/// aliasing-gap fix means this now reads accusation_count cleanly,
/// not garbage from another view's writes.
fn read_first_view_storage_per_agent(state: &mut GeneratedRuntime) -> Vec<f32> {
    let n = N_TOTAL as usize;
    let bytes = (n as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("detective_investigation::per_agent_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("detective_investigation::per_agent_readback"),
        },
    );
    encoder.copy_buffer_to_buffer(&state.view_storage_accusation_count_primary_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

fn read_total_witnessed(_state: &mut GeneratedRuntime) -> Option<Vec<f32>> {
    // Same field-reflection limitation — total_witnessed shares the
    // primary buffer slot with the other views in this fixture, so
    // there's no separate `view_storage_total_witnessed_primary_buf`
    // field to read.
    None
}

fn readback_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("detective_investigation::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("detective_investigation::u32_readback"),
        },
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
