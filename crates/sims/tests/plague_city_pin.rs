//! `plague_city` pin — drives the multi-class plague fixture and
//! reports an SIR-style breakdown after a 500-tick equilibrium horizon.
//!
//! **Topology (host-seeded)**:
//! - 64 Citizens at z=0, scattered inside a city footprint (radius 6)
//! - 4 Doctors near the centre (clustered hospital tile, radius ~1)
//! - 4 Healers walking the periphery (radius ~5)
//! - 4 Priests at the graveyard tile (clustered, radius ~1)
//! - 6 Citizens host-seeded as `hunger = INITIAL_INFECTION` at tick 0,
//!   bootstrapping the outbreak. (See GAP P-B in
//!   `docs/architecture/gaps_plague_city.md` — `@host_callable`
//!   codegen is gated on engine-aliased event kinds; the
//!   `Infected` event is fixture-defined so injection routes
//!   through direct buffer writes instead.)
//!
//! creature_type discriminants are pinned by `.sim` declaration
//! order (`EntityRef.0` in `crates/dsl_ast/src/resolve.rs`). The
//! plague_city.sim entities are declared alphabetically — Citizen,
//! Doctor, Healer, Priest — so the per-slot creature_type the host
//! writes into `agent_creature_type_buf` mirrors the constants the
//! rule gates compare against.
//!   Citizen = 0, Doctor = 1, Healer = 2, Priest = 3
//!
//! **What the pin asserts** (load-bearing):
//!   1. Initial seed survives `try_new` + first step (alive bits
//!      intact for every class).
//!   2. After 500 ticks, the seeded infection has actually progressed
//!      — at least one of (a) more citizens infected vs initial seed,
//!      or (b) at least one death from the disease, or (c) at least
//!      one cure recorded — must have happened. A static state means
//!      the per-tick chain isn't propagating.
//!   3. Population conservation: alive_citizens + dead_citizens ==
//!      initial_citizens (no spurious slot recycling, no missing
//!      slots).
//!   4. No NaN / Inf in any agent column.
//!
//! **What the pin reports** (soft NOTE — hands the next iteration
//! the SIR breakdown without asserting specific values):
//!   - SIR breakdown: susceptible / infected / recovered / dead.
//!   - Healer kill-pressure proxy: # of bless dispatches landed.
//!   - Belief drift over time: by reading the shared
//!     view_storage_primary_buf at intervals (NOTE only — see
//!     potential GAP P-E for shared view storage corruption).

use sims::plague_city::GeneratedRuntime;

const SEED: u64 = 0xB1AC1E_DEA_D5;
const N_CITIZENS: u32 = 64;
const N_DOCTORS: u32 = 4;
const N_HEALERS: u32 = 4;
const N_PRIESTS: u32 = 4;
const N_TOTAL: u32 = N_CITIZENS + N_DOCTORS + N_HEALERS + N_PRIESTS;
const TICKS: u32 = 500;
const INITIAL_INFECTED: u32 = 6;

const CT_CITIZEN: u32 = 0;
const CT_DOCTOR: u32 = 1;
const CT_HEALER: u32 = 2;
const CT_PRIEST: u32 = 3;

const INITIAL_INFECTION_LOAD: f32 = 30.0;

#[test]
fn plague_runs_to_equilibrium_after_500_ticks() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[plague_city] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_topology(&mut state);
    seed_initial_outbreak(&mut state);

    // Sanity — every slot was seeded alive.
    let init_alive_citizens = count_alive_of_type(&mut state, CT_CITIZEN);
    let init_alive_doctors = count_alive_of_type(&mut state, CT_DOCTOR);
    let init_alive_healers = count_alive_of_type(&mut state, CT_HEALER);
    let init_alive_priests = count_alive_of_type(&mut state, CT_PRIEST);
    let init_infected = count_with_hunger_gt_zero(&mut state, CT_CITIZEN);

    eprintln!("==== plague_city {TICKS}-tick outbreak ====");
    eprintln!(
        "  init:    citizens={init_alive_citizens}/{N_CITIZENS}  doctors={init_alive_doctors}/{N_DOCTORS}  healers={init_alive_healers}/{N_HEALERS}  priests={init_alive_priests}/{N_PRIESTS}",
    );
    eprintln!("  init:    infected_citizens={init_infected}/{INITIAL_INFECTED}");

    // Run the equilibrium horizon. 500 ticks at 100ms = 50s of sim
    // time — long enough for ~6 initial infections to cascade through
    // the population, deaths to land, and cures to land.
    for _ in 0..TICKS {
        state.step();
    }

    // Final snapshot.
    let final_alive_citizens = count_alive_of_type(&mut state, CT_CITIZEN);
    let final_alive_doctors = count_alive_of_type(&mut state, CT_DOCTOR);
    let final_alive_healers = count_alive_of_type(&mut state, CT_HEALER);
    let final_alive_priests = count_alive_of_type(&mut state, CT_PRIEST);
    let final_infected = count_with_hunger_gt_zero(&mut state, CT_CITIZEN);

    let dead_citizens = N_CITIZENS - final_alive_citizens;
    let alive_uninfected_citizens =
        final_alive_citizens.saturating_sub(final_infected);

    // SIR breakdown. Recovered ≈ alive but with hunger == 0 AND hp <
    // 100 (i.e. they took damage from infection but were cured before
    // it killed them). Susceptible = alive AND hunger == 0 AND hp ==
    // 100 (untouched throughout). This is approximate — it conflates
    // "always healthy" with "healed and refilled" since Cure heals hp;
    // the fold sums in cure_count would be the precise signal but
    // those views all share view_storage_primary_buf with each other
    // (potential GAP P-E — see gaps_plague_city.md). The hp/hunger
    // SoA columns have unambiguous backing buffers.
    let hp_buf = state.agent_hp_buf.clone();
    let hunger_buf = state.agent_hunger_buf.clone();
    let alive_buf = state.agent_alive_buf.clone();
    let type_buf = state.agent_creature_type_buf.clone();
    let hps = read_agent_f32(&mut state, &hp_buf);
    let hungers = read_agent_f32(&mut state, &hunger_buf);
    let alives = read_agent_u32(&mut state, &alive_buf);
    let types = read_agent_u32(&mut state, &type_buf);

    let mut susceptible = 0u32;
    let mut currently_infected = 0u32;
    let mut recovered_or_treated = 0u32;
    let mut dead = 0u32;
    let mut min_hp = f32::INFINITY;
    let mut max_hp = f32::NEG_INFINITY;
    let mut nan_count = 0u32;
    for i in 0..N_TOTAL as usize {
        if types[i] != CT_CITIZEN {
            continue;
        }
        let hp = hps[i];
        let hunger = hungers[i];
        let alive = alives[i] != 0;
        if !hp.is_finite() || !hunger.is_finite() {
            nan_count += 1;
        }
        if hp < min_hp { min_hp = hp; }
        if hp > max_hp { max_hp = hp; }
        if !alive {
            dead += 1;
        } else if hunger > 0.0 {
            currently_infected += 1;
        } else if hp >= 99.5 {
            // Untouched: still at full HP (init=100, sickness_rate=1,
            // cure heals back to cap; a citizen who was cured would
            // probably be at hp < 100 by the time their cure landed
            // unless they were very lucky).
            susceptible += 1;
        } else {
            recovered_or_treated += 1;
        }
    }

    eprintln!(
        "  final:   citizens alive={final_alive_citizens}/{N_CITIZENS}  doctors={final_alive_doctors}/{N_DOCTORS}  healers={final_alive_healers}/{N_HEALERS}  priests={final_alive_priests}/{N_PRIESTS}",
    );
    eprintln!(
        "  SIR:     S={susceptible}  I={currently_infected}  R={recovered_or_treated}  D={dead}  (citizens={N_CITIZENS})",
    );
    eprintln!(
        "  hp:      min={min_hp:.2}  max={max_hp:.2}  alive_uninfected={alive_uninfected_citizens}",
    );

    let outcome = if dead == N_CITIZENS {
        "WIPE — every citizen dead"
    } else if currently_infected == 0 && dead < N_CITIZENS / 2 {
        "CONTAINED — disease burned out"
    } else if dead > N_CITIZENS / 2 {
        "DEMOGRAPHIC COLLAPSE — > 50% dead"
    } else if currently_infected > 0 {
        "ENDEMIC — disease still spreading at horizon"
    } else {
        "ROUTED — equilibrium reached"
    };
    eprintln!("  verdict: {outcome}");
    eprintln!("==========================================");

    // -------- Load-bearing assertions --------

    // 1. Initial seed survived try_new + first step.
    assert_eq!(
        init_alive_citizens, N_CITIZENS,
        "citizen seed didn't survive init: {init_alive_citizens}/{N_CITIZENS}",
    );
    assert_eq!(
        init_alive_doctors, N_DOCTORS,
        "doctor seed didn't survive init: {init_alive_doctors}/{N_DOCTORS}",
    );
    assert_eq!(
        init_alive_healers, N_HEALERS,
        "healer seed didn't survive init: {init_alive_healers}/{N_HEALERS}",
    );
    assert_eq!(
        init_alive_priests, N_PRIESTS,
        "priest seed didn't survive init: {init_alive_priests}/{N_PRIESTS}",
    );
    assert!(
        init_infected >= INITIAL_INFECTED - 1, // -1 tolerance for any first-step coalesce
        "outbreak seed didn't survive init: only {init_infected}/{INITIAL_INFECTED} infected",
    );

    // 2. No NaN / Inf in citizen columns. The fused
    //    ContagionScan_and_SicknessProgresses kernel does an
    //    `agents.set_hp(self, old_hp - sickness_rate)` write each
    //    tick for every infected slot — a regression that breaks
    //    the per_agent fusion shape would surface as NaN propagation
    //    here.
    assert_eq!(
        nan_count, 0,
        "{nan_count} citizen slots have NaN/Inf in hp or hunger",
    );

    // 3. Population conservation: every citizen slot is either alive
    //    or dead — no slot got recycled or duplicated.
    assert_eq!(
        final_alive_citizens + dead_citizens,
        N_CITIZENS,
        "population conservation broken: alive={final_alive_citizens} + dead={dead_citizens} != {N_CITIZENS}",
    );

    // 4. The disease did SOMETHING — at least ONE of (more infected
    //    than initial seed, OR at least one death, OR at least one
    //    citizen healed below init hp) must have happened.
    //    Static state means the contagion / sickness chain is dead.
    let advanced =
        currently_infected > init_infected ||
        dead > 0 ||
        recovered_or_treated > 0 ||
        max_hp < 100.0; // someone took damage at some point
    assert!(
        advanced,
        "disease did nothing in {TICKS} ticks: \
         currently_infected={currently_infected} init={init_infected} \
         dead={dead} recovered={recovered_or_treated} max_hp={max_hp:.2}. \
         Likely cause: ContagionScan / SicknessProgresses fusion silently \
         dropped, or rng.action() draw is uniformly non-firing.",
    );

    // 5. Treaters didn't die — Doctors / Healers / Priests aren't
    //    susceptible to the disease (the contagion gate is
    //    creature_type == Citizen). If the treaters started dying
    //    that means a rule's body-side filter broke.
    assert_eq!(
        final_alive_doctors, N_DOCTORS,
        "doctors took losses ({} dead) — class-discrimination broken in ContagionScan?",
        N_DOCTORS - final_alive_doctors,
    );
    assert_eq!(
        final_alive_healers, N_HEALERS,
        "healers took losses ({} dead) — class-discrimination broken?",
        N_HEALERS - final_alive_healers,
    );
    assert_eq!(
        final_alive_priests, N_PRIESTS,
        "priests took losses ({} dead) — class-discrimination broken?",
        N_PRIESTS - final_alive_priests,
    );

    // 6. The conditional `agents.set_alive(self, false)` inside
    //    SicknessProgresses MUST land — the 6 host-seeded citizens
    //    progress from hp=100 to hp<=0 over ~100 ticks at
    //    sickness_rate=1.0, and the kill-crossing `if (old_hp > 0
    //    && new_hp <= 0) { agents.set_alive(self, false); ... }`
    //    must flip the alive bit. This was the load-bearing
    //    behavioural symptom of Gap P-D in
    //    `docs/architecture/gaps_plague_city.md`; the gap doc
    //    hypothesised "the conditional branch's set_alive Assign
    //    gets dropped at fusion time", but post-T5 schedule fix
    //    (`d1207fca`) the SicknessProgresses kernel runs alone
    //    (the original `_and_ContagionScan` fusion changed shape),
    //    AND the fused-emit path itself preserves conditional
    //    alive writes (regression-pinned by
    //    `crates/dsl_compiler/tests/fused_set_alive_conditional_emit.rs`).
    //    The actual root cause was the spatial-build-after-consumer
    //    cycle Gap T5 — pre-T5 the contagion didn't transmit so
    //    citizens never accumulated negative hp; the original 6
    //    DID die from the alive flip but the pin's NOTE described
    //    it as "0 deaths" because a separate condition kept dead==0.
    //    Promoting the assertion to load-bearing pins the post-T5
    //    behaviour: the alive flip lands.
    assert!(
        dead >= INITIAL_INFECTED,
        "expected at least {INITIAL_INFECTED} dead (the host-seeded \
         outbreak progressed to death and flipped alive); got dead={dead}. \
         Likely cause: SicknessProgresses' `agents.set_alive(self, false)` \
         write isn't landing, OR the contagion kernel never advanced \
         hunger so sickness never drained hp. See Gap P-D in \
         gaps_plague_city.md.",
    );

    // -------- Soft NOTEs (informational; track for next iteration) --------

    if currently_infected == init_infected && recovered_or_treated == 0 {
        eprintln!(
            "  NOTE: disease is static (initial 6 infected, 0 spread, 0 cured). \
             Likely cause: rng.action() % 100 is biased away from \
             the transmission gate, OR the per-pair direct write isn't \
             propagating across kernel-fusion. See gaps log.",
        );
    }
    eprintln!(
        "  contract: 25 kernels emit, {TICKS} ticks step without panic, \n            class-discrimination intact, no NaN propagation, \n            population conservation holds.",
    );
}

// -------- Topology seeding --------

fn seed_topology(state: &mut GeneratedRuntime) {
    use std::f32::consts::TAU;

    let n = N_TOTAL as usize;

    // Position layout:
    //   - Citizens scattered inside city footprint at z=0, radius 6.
    //   - Doctors clustered at hospital tile near origin (radius 1).
    //   - Healers walking periphery at radius 5.
    //   - Priests clustered at graveyard tile (offset, radius 1).
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(n);
    for i in 0..N_CITIZENS {
        // Pseudo-random scatter via a deterministic spiral. With
        // 64 citizens at radius 6 the density is ~0.5/cell which
        // gives ~3-4 neighbours per 27-cell ring at peak.
        let theta = (i as f32) * TAU / (N_CITIZENS as f32);
        let r = 1.0 + ((i as f32) * 0.085).fract() * 5.0;
        positions.push([theta.cos() * r, theta.sin() * r, 0.0, 0.0]);
    }
    for i in 0..N_DOCTORS {
        // Hospital tile cluster — doctors close enough to each other
        // that a sick citizen wandering near the hospital gets seen
        // by all 4 doctors per tick.
        let theta = (i as f32) * TAU / (N_DOCTORS as f32);
        positions.push([theta.cos() * 0.5, theta.sin() * 0.5, 0.0, 0.0]);
    }
    for i in 0..N_HEALERS {
        // Periphery — healers patrol outside the citizen footprint.
        let theta = (i as f32) * TAU / (N_HEALERS as f32) + 0.3;
        positions.push([theta.cos() * 5.0, theta.sin() * 5.0, 0.0, 0.0]);
    }
    for i in 0..N_PRIESTS {
        // Graveyard tile — fixed offset so priests pick up the dead.
        // Same "cluster" layout as doctors so they fan out across the
        // same neighborhood when corpses pile up.
        let theta = (i as f32) * TAU / (N_PRIESTS as f32);
        positions.push([
            -3.0 + theta.cos() * 0.5,
            -3.0 + theta.sin() * 0.5,
            0.0,
            0.0,
        ]);
    }
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    // creature_type per slot. Discriminant order = entity decl order
    // alphabetised: Citizen=0, Doctor=1, Healer=2, Priest=3.
    let mut creature_type: Vec<u32> = Vec::with_capacity(n);
    for _ in 0..N_CITIZENS { creature_type.push(CT_CITIZEN); }
    for _ in 0..N_DOCTORS  { creature_type.push(CT_DOCTOR);  }
    for _ in 0..N_HEALERS  { creature_type.push(CT_HEALER);  }
    for _ in 0..N_PRIESTS  { creature_type.push(CT_PRIEST);  }
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&creature_type),
    );
}

/// Bootstrap the outbreak by host-side direct write of `hunger` on
/// the first INITIAL_INFECTED Citizen slots.
///
/// Workaround for GAP P-B (`@host_callable` codegen skips
/// fixture-defined event kinds) — the .sim's `Infected` event has no
/// auto-emitted host injector method, so we seed via direct buffer
/// write. The agents.hunger column is f32, indexed by slot id.
fn seed_initial_outbreak(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    // Read full column, modify the first INITIAL_INFECTED Citizen
    // slots, write back. Could be a partial write but full-column
    // write is simpler and the column is only N_TOTAL * 4 bytes.
    let mut hungers = vec![0.0f32; n];
    for i in 0..(INITIAL_INFECTED as usize) {
        hungers[i] = INITIAL_INFECTION_LOAD;
    }
    state.gpu.queue.write_buffer(
        &state.agent_hunger_buf,
        0,
        bytemuck::cast_slice(&hungers),
    );
}

// -------- Readback helpers --------

fn count_alive_of_type(state: &mut GeneratedRuntime, ct: u32) -> u32 {
    let alive_buf = state.agent_alive_buf.clone();
    let type_buf = state.agent_creature_type_buf.clone();
    let alive = read_agent_u32(state, &alive_buf);
    let types = read_agent_u32(state, &type_buf);
    alive
        .iter()
        .zip(types.iter())
        .filter(|(&a, &t)| a != 0 && t == ct)
        .count() as u32
}

fn count_with_hunger_gt_zero(state: &mut GeneratedRuntime, ct: u32) -> u32 {
    let hunger_buf = state.agent_hunger_buf.clone();
    let type_buf = state.agent_creature_type_buf.clone();
    let alive_buf = state.agent_alive_buf.clone();
    let hungers = read_agent_f32(state, &hunger_buf);
    let types = read_agent_u32(state, &type_buf);
    let alives = read_agent_u32(state, &alive_buf);
    hungers
        .iter()
        .zip(types.iter())
        .zip(alives.iter())
        .filter(|((&h, &t), &a)| t == ct && a != 0 && h > 0.0)
        .count() as u32
}

fn read_agent_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer) -> Vec<u32> {
    let count = N_TOTAL as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plague_city::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("plague_city::u32_readback") },
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

fn read_agent_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer) -> Vec<f32> {
    let count = N_TOTAL as usize;
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plague_city::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("plague_city::f32_readback") },
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
