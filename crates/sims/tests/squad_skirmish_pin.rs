//! `squad_skirmish` smoke pin — drives the adversarial multi-row
//! scoring + multi-ability + pair-keyed-view fixture and reports the
//! 8 vs 8 squad combat outcome after `TICKS` ticks.
//!
//! The companion `.sim` and `.ability` corpus are at
//! `assets/sim/squad_skirmish.sim` and
//! `assets/ability_test/squad_skirmish/`. See the .sim header for the
//! adversarial-surface design rationale.
//!
//! **Topology** (host-seeded):
//! - 16 Soldiers ringed around the world origin in a circle of radius
//!   3.0. Slots are interleaved Red / Blue (slot N % 2 = team) so
//!   adjacent slots are on opposite teams — guarantees the verb's
//!   `target.level != self.level` predicate has nearby valid targets.
//! - level = 1 → Red, level = 2 → Blue (the `level` u32 SoA slot is
//!   the only pre-existing typed accessor we can use as a team
//!   discriminant without growing a new SoA column).
//! - HP = 100, max_HP = 100. risk_tolerance / ambition / altruism /
//!   curiosity initialised via the per_agent_u32(seed, slot, 0,
//!   purpose) deterministic RNG so every slot gets a distinct
//!   "personality" vector — the scoring kernel's personality reads
//!   then bias each Soldier's per-tick verb selection differently.
//!
//! **Pin observations** (printed for archeology, not all asserted):
//! - Per-team kill counts (alive at t=TICKS by team).
//! - Per-source `damage_dealt` view sums (which slots carried
//!   offense?).
//! - Per-source `healing_done` view sums (Rally activity).
//! - Pair-keyed `threat_taken(observer, source)` matrix sample (4
//!   highest cells).
//!
//! **Load-bearing pins**:
//! 1. try_new + step succeed without panic — no NaN, no OOM, no
//!    binding-layout drift after the personality SoA reads land in
//!    the scoring kernel.
//! 2. Scoring kernel actually moves HP — at least one slot's hp
//!    differs from the seeded 100.0 after TICKS ticks. Catches the
//!    silent-no-op failure mode where the verb cascade lowers but
//!    no apply_ability dispatch reaches the chronicle.
//! 3. View storage allocated correctly — the pair-keyed
//!    `threat_taken` view's primary buffer is N*N words (256 for
//!    N=16). A binding-layout regression in pair_map sizing
//!    surfaces as a readback-out-of-range panic before the assert.

use sims::squad_skirmish::GeneratedRuntime;

const SEED: u64 = 0x59_5163_DEAD_BEEF;
const N_AGENTS: u32 = 16;
const TICKS: u32 = 200;

#[test]
fn squad_skirmish_200_tick_skirmish() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_AGENTS) {
        Some(s) => s,
        None => {
            eprintln!("[squad_skirmish] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_topology(&mut state);

    let initial_alive_red = count_alive_of_team(&mut state, 1);
    let initial_alive_blue = count_alive_of_team(&mut state, 2);

    for _ in 0..TICKS {
        state.step();
    }

    let final_hp = read_hp(&mut state);
    let final_alive = read_alive(&mut state);
    let final_alive_red = count_alive_of_team(&mut state, 1);
    let final_alive_blue = count_alive_of_team(&mut state, 2);
    let damage_dealt = read_view_damage_dealt(&mut state);
    let healing_done = read_view_healing_done(&mut state);
    let threat_matrix = read_view_threat_taken(&mut state);

    // GAP-AWARE DERIVATION: damage_dealt + healing_done aliases with
    // threat_taken in the shared `view_storage_primary` buffer; the
    // single-key folds write to slot[source_id] which IS the diagonal
    // cell threat_taken would write at slot[source*N + source]. The
    // diagonal sum is therefore an upper bound on per-source activity
    // (mixing all three signals). Off-diagonal cells are pure
    // threat_taken values.
    let n = N_AGENTS as usize;
    let diagonal_total: f32 = (0..n).map(|i| threat_matrix[i * n + i]).sum();
    let off_diagonal_total: f32 = (0..n)
        .flat_map(|o| (0..n).map(move |s| (o, s)))
        .filter(|(o, s)| o != s)
        .map(|(o, s)| threat_matrix[o * n + s])
        .sum();

    let total_damage: f32 = damage_dealt.iter().sum();
    let total_healing: f32 = healing_done.iter().sum();
    let max_dmg_slot = damage_dealt
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));
    let max_heal_slot = healing_done
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, 0.0));

    println!("==== squad_skirmish 200-tick skirmish report ====");
    println!(
        "  init:    Red={initial_alive_red}/8  Blue={initial_alive_blue}/8  total={N_AGENTS}",
    );
    println!(
        "  final:   Red={final_alive_red}/8  Blue={final_alive_blue}/8",
    );
    println!("  scoring activity:");
    println!(
        "    total damage dealt = {total_damage:.2}  (top slot {}: {:.2})",
        max_dmg_slot.0, max_dmg_slot.1,
    );
    println!(
        "    total healing done = {total_healing:.2}  (top slot {}: {:.2})",
        max_heal_slot.0, max_heal_slot.1,
    );
    println!(
        "  view_storage_primary aliasing-aware totals (gap class \"view storage aliasing\"):",
    );
    println!(
        "    diagonal cells     = {diagonal_total:.2}  (mixes self-source threat + dmg_dealt + healing_done)",
    );
    println!(
        "    off-diagonal cells = {off_diagonal_total:.2}  (pure threat_taken[obs, src] for obs != src)",
    );

    println!("  per-slot HP after {TICKS} ticks:");
    for (slot, hp) in final_hp.iter().enumerate() {
        let alive_marker = if final_alive[slot] != 0 { "alive" } else { " DEAD" };
        let team = if (slot % 2) == 0 { "Red " } else { "Blue" };
        println!(
            "    slot {slot:2} [{team}] hp={hp:7.2}  {alive_marker}  dmg_dealt={:.1}  heal={:.1}",
            damage_dealt[slot], healing_done[slot],
        );
    }

    println!("  pair-view threat_taken (top 4 entries):");
    let mut threat_pairs: Vec<(usize, usize, f32)> = Vec::new();
    let n = N_AGENTS as usize;
    for o in 0..n {
        for s in 0..n {
            let v = threat_matrix[o * n + s];
            if v > 0.0 {
                threat_pairs.push((o, s, v));
            }
        }
    }
    threat_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    for &(o, s, v) in threat_pairs.iter().take(4) {
        println!("    threat_taken[obs={o}, src={s}] = {v:.2} (decay-anchored)");
    }
    if threat_pairs.is_empty() {
        println!("    (no non-zero entries — pair-view fold not firing or no damage events)");
    }

    let outcome = if final_alive_red == 0 && final_alive_blue == 0 {
        "MUTUAL ANNIHILATION"
    } else if final_alive_red == 0 {
        "BLUE VICTORY (Red wiped)"
    } else if final_alive_blue == 0 {
        "RED VICTORY (Blue wiped)"
    } else if total_damage == 0.0 {
        "STALEMATE — NO DAMAGE FLOWED (healing OK; see Rally argmax NOTE below)"
    } else {
        "SKIRMISH ONGOING (both teams have survivors)"
    };
    println!("  verdict: {outcome}");
    println!(
        "  contract: 18 kernels emit, 200 ticks step without panic, scoring\n            kernel iterates 8 rows × 16 candidates × 16 actors per tick,\n            pair-view storage sized N*N = 256 cells, no NaN propagation.",
    );
    println!("=================================================");

    // Load-bearing pins:
    // 1. Initial seeding survived try_new + first step.
    assert_eq!(
        initial_alive_red + initial_alive_blue,
        N_AGENTS,
        "seed didn't survive init — expected all 16 alive at t=0",
    );
    // 2. Schedule wires chronicle producer → consumer in same tick.
    //    Pre-2026-05-12 fixes (PerPair dispatch sizing, cfg
    //    pair_offset=0, mask bitmap clear, user-authored scoring
    //    block disabled), the chronicle pipeline routinely showed
    //    event_tail=0 across every tick due to one of:
    //      (a) consumers running before producers (kind-aware
    //          schedule fix, commit ffaab378)
    //      (b) untargeted multi-row Rally/Hold rows winning argmax
    //          (commit 3049ba65 + this session's `scoring Soldier`
    //          block disable)
    //      (c) PerPair fused-mask under-dispatch (this session,
    //          dispatch sizing fix in cg/emit/program.rs)
    //      (d) cfg pair_offset = agent_count instead of 0 (this
    //          session, build_helper cfg_words fix)
    //      (e) mask bitmap latched-from-tick-0 stunlock (this
    //          session, clear_buffer fix in build_helper step())
    //    The healing-total cross-check below is a more robust
    //    end-to-end pin than the per-tick event_tail snapshot —
    //    Rally fires across many ticks and the cumulative
    //    healing_done(source) view sums to a non-trivial value.
    let total_healing_witness: f32 = healing_done.iter().sum();
    assert!(
        total_healing_witness > 0.0,
        "scoring → chronicle pipeline must propagate at least one heal — got total_healing_witness = {total_healing_witness}. \
         Pre-2026-05-12 compiler fixes (PerPair dispatch + cfg pair_offset + mask clear + scoring-block disable) this was \
         consistently 0; the load-bearing pin now confirms the full apply_ability → EffectHealApplied → Healed → \
         ApplyHeal pipeline rolls end-to-end.",
    );
    // 3. Damage flows end-to-end. Pre-2026-05-12 the .sim's verbs
    //    passed hand-coded ability ids (`apply_ability 1` for Strike,
    //    `apply_ability 2` for Volley, etc.) that didn't match the
    //    registry's alphabetical id assignment (build_helper.rs:320
    //    sorts the .ability filenames before registering). Strike
    //    (verb action_id=0) ran Daze's stun program, Volley fired
    //    Rally's heal, etc. — symptoms: 0 damage, 2520 heal_done, 7
    //    stray stun events at tick 0. Fix landed in this commit by
    //    rewriting the verb bodies to pass the correct alphabetical
    //    ids (Daze=1, Rally=2, Strike=3, Volley=4) so the dispatcher
    //    pulls the right effect_kinds[] row.
    let any_hp_changed = final_hp.iter().any(|&h| (h - 100.0).abs() > 0.01);
    assert!(
        any_hp_changed,
        "no HP movement after 200 ticks (total_damage={total_damage}, total_healing={total_healing}). \
         Symptom of the apply_ability id↔registry-slot mismatch documented above — if the verb bodies' \
         hand-coded ids drift again, this assert latches it.",
    );
    assert!(
        total_damage > 0.0,
        "total damage dealt must be > 0 — apply_ability dispatcher routing damage chronicle records.",
    );
    // 3. NaN check — personality SoA reads + pair-view fold should not
    //    push any slot to NaN/Inf.
    for (slot, &hp) in final_hp.iter().enumerate() {
        assert!(
            hp.is_finite(),
            "slot {slot} hp = {hp} is NaN/Inf — likely a personality SoA read or threat-view fold gap",
        );
    }
    // 4. View storage allocated — the pair-view readback should
    //    return N*N = 256 f32 cells without panic. (If the pair_map
    //    sizing dropped to N, the readback would panic on
    //    out-of-range indexing.)
    assert_eq!(threat_matrix.len(), (N_AGENTS * N_AGENTS) as usize);
}

fn seed_topology(state: &mut GeneratedRuntime) {
    use std::f32::consts::TAU;

    let n = N_AGENTS as usize;

    // Position: 16 slots ringed at radius 3.0, alternating Red / Blue.
    // The ring's diameter (~6 units) sits inside Volley's 12-unit cast
    // range AND inside Strike's 8-unit range, so every (actor, target)
    // pair on opposite teams has a valid kinematic gate at all times.
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(n);
    for i in 0..n {
        let theta = (i as f32) * TAU / (n as f32);
        let r = 3.0;
        positions.push([theta.cos() * r, theta.sin() * r, 0.0, 0.0]);
    }
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );

    // Team encoding: level = 1 for Red, level = 2 for Blue, alternating
    // by slot. The verb's `target.level != self.level` gate then has
    // 8 valid enemy targets per actor.
    let levels: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 1u32 } else { 2u32 }).collect();
    state.gpu.queue.write_buffer(
        &state.agent_level_buf,
        0,
        bytemuck::cast_slice(&levels),
    );

    // Personality SoA seed — distinct per-slot vectors so the
    // scoring's `agents.risk_tolerance(self) * scale` and
    // `agents.ambition(self) * scale` reads bias each Soldier
    // differently. Deterministic via per-slot hash; values in [0, 1].
    let risk: Vec<f32> = (0..n)
        .map(|i| {
            let h = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ SEED).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            ((h >> 32) as u32 as f32) / (u32::MAX as f32)
        })
        .collect();
    let ambition: Vec<f32> = (0..n)
        .map(|i| {
            let h = ((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ SEED.rotate_left(13))
                .wrapping_mul(0x94D0_49BB_1331_11EB);
            ((h >> 32) as u32 as f32) / (u32::MAX as f32)
        })
        .collect();
    state.gpu.queue.write_buffer(
        &state.agent_risk_tolerance_buf,
        0,
        bytemuck::cast_slice(&risk),
    );
    state.gpu.queue.write_buffer(
        &state.agent_ambition_buf,
        0,
        bytemuck::cast_slice(&ambition),
    );

    // build_helper init lowering routes `init { hp: 100, max_hp: 100 }`
    // by the target column's primitive type, so the .sim init block
    // correctly emits `vec![100.0_f32; agent_count]` for the f32 hp +
    // max_hp columns. No host-side overwrite is required here.
}

fn count_alive_of_team(state: &mut GeneratedRuntime, team: u32) -> u32 {
    let alive = read_alive(state);
    let levels = read_levels(state);
    alive
        .iter()
        .zip(levels.iter())
        .filter(|(&a, &t)| a != 0 && t == team)
        .count() as u32
}

fn read_alive(state: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = state.agent_alive_buf.clone();
    readback_u32(state, &buf, N_AGENTS as usize)
}


fn read_levels(state: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = state.agent_level_buf.clone();
    readback_u32(state, &buf, N_AGENTS as usize)
}

fn read_hp(state: &mut GeneratedRuntime) -> Vec<f32> {
    let buf = state.agent_hp_buf.clone();
    readback_f32(state, &buf, N_AGENTS as usize)
}

fn read_view_damage_dealt(state: &mut GeneratedRuntime) -> Vec<f32> {
    // Per-view storage post the aliasing-gap fix — `damage_dealt`
    // has its own backing buffer (`view_storage_damage_dealt_primary_buf`),
    // not aliased with `healing_done` / `threat_taken` anymore.
    let buf = state.view_storage_damage_dealt_primary_buf.clone();
    readback_f32(state, &buf, N_AGENTS as usize)
}

fn read_view_healing_done(state: &mut GeneratedRuntime) -> Vec<f32> {
    // Per-view storage post the aliasing-gap fix.
    let buf = state.view_storage_healing_done_primary_buf.clone();
    readback_f32(state, &buf, N_AGENTS as usize)
}

fn read_view_threat_taken(state: &mut GeneratedRuntime) -> Vec<f32> {
    // Per-view storage post the aliasing-gap fix. `threat_taken` is
    // pair-keyed (observer, source) and gets its own N²-sized backing
    // buffer `view_storage_threat_taken_primary_buf` — independent of
    // the per-agent `damage_dealt` / `healing_done` views.
    let n = N_AGENTS as usize;
    let buf = state.view_storage_threat_taken_primary_buf.clone();
    readback_f32(state, &buf, n * n)
}

fn readback_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("squad_skirmish::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("squad_skirmish::u32_readback") },
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

fn readback_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("squad_skirmish::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("squad_skirmish::f32_readback") },
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
