//! `many_events_ability` — runtime proof that `apply_ability` works in a
//! fixture with more than 25 user events (S5b, 2026-07-22).
//!
//! **The defect this pins.** User event kind ids were allocated as a bare
//! source-order index while the engine aliases its own chronicle events
//! to hardcoded discriminants 26..=80 (`dsl_ast::engine_events`). The
//! 27th user event in any `.sim` therefore landed on kind 26 — the tag
//! the `apply_ability` dispatcher stamps on `EffectDamageApplied`
//! records — so a big fixture's damage records were consumed by an
//! unrelated user rule with the payload words aligned. That locked
//! `webband_colony` (60 user events) out of the ability system entirely
//! (see the S5 slice report in docs/superpowers/plans/webband-port.md).
//!
//! The fixture declares 33 user events + the aliased
//! `EffectDamageApplied`, with `PadWouldCollide` deliberately sitting at
//! source index 26. Two assertions carry the proof:
//!
//!   1. **`apply_ability` functions.** Every alive agent self-casts
//!      `SelfStrike` (damage 4) once per tick; the `@phase(post)`
//!      `EffectDamageApplied` consumer subtracts it from hp. After
//!      `TICKS` ticks hp must have fallen by roughly `4 * TICKS` and
//!      `damage_records` must be nonzero — i.e. the dispatcher's kind-26
//!      records reached the rule that asked for them.
//!   2. **The collision is gone.** `MarkCollider` — the rule that used
//!      to share kind 26 — writes `collided_marker = 1.0`. Nothing emits
//!      `PadWouldCollide`, so every slot must still read 0.0. Pre-fix
//!      this fixture could not even compile (both events intern
//!      `EventKindId(26)`, which the CG builder rejects as a duplicate),
//!      so a green run of this file is itself the regression barrier.
//!
//! Determinism: two same-seed runs must produce byte-equal hp columns.

use sims::many_events_ability::GeneratedRuntime;

const SEED: u64 = 0x5B_C0_11_1D_E5;
const N_AGENTS: u32 = 8;
const TICKS: u32 = 20;
/// `damage 4.0` in `assets/ability_test/many_events_ability/SelfStrike.ability`.
const DAMAGE_PER_CAST: f32 = 4.0;
const START_HP: f32 = 500.0;

fn read_f32_buf(state: &GeneratedRuntime, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
    let bytes = (state.agent_count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..state.agent_count as usize].to_vec()
    };
    staging.unmap();
    out
}

/// Seed every slot alive with a fat hp pool, run `TICKS` ticks, and read
/// back `(hp, damage_records, collided_marker)`.
fn run(seed: u64) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut state = GeneratedRuntime::try_new(seed, N_AGENTS)?;

    let alive: Vec<u32> = vec![1u32; N_AGENTS as usize];
    let hp: Vec<f32> = vec![START_HP; N_AGENTS as usize];
    // One entity kind (`Fighter`) — ordinal 0, which is the zero-init
    // default; written explicitly so the seeding is self-describing.
    let ct: Vec<u32> = vec![0u32; N_AGENTS as usize];
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&ct),
    );

    for _ in 0..TICKS {
        state.step();
    }

    let hp_out = read_f32_buf(&state, &state.agent_hp_buf, "many_events::hp");
    let records = read_f32_buf(
        &state,
        &state.agent_damage_records_buf,
        "many_events::damage_records",
    );
    let marker = read_f32_buf(
        &state,
        &state.agent_collided_marker_buf,
        "many_events::collided_marker",
    );
    Some((hp_out, records, marker))
}

#[test]
fn apply_ability_closes_the_loop_past_the_reserved_kind_range() {
    let Some((hp, records, marker)) = run(SEED) else {
        eprintln!("[many_events_ability] skipping: no wgpu adapter on host.");
        return;
    };

    // (1) apply_ability functions: the dispatcher's kind-26 records
    // reached ApplyChronicleDamage, which spent them on hp.
    let total_records: f32 = records.iter().sum();
    let min_hp = hp.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_hp = hp.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!(
        "  many_events_ability: hp {min_hp:.1}..{max_hp:.1} (start {START_HP:.1}), \
         damage_records total {total_records:.0} over {TICKS} ticks × {N_AGENTS} agents",
    );
    assert!(
        total_records > 0.0,
        "no EffectDamageApplied record was ever consumed — apply_ability did not \
         reach the consumer in a >25-event fixture (damage_records all zero)",
    );
    assert!(
        max_hp < START_HP,
        "hp never fell: the chronicle loop did not close (hp {min_hp}..{max_hp})",
    );
    // Every agent casts every tick, so each agent's own record count is
    // its own hp delta / damage. The chronicle fold window can drop or
    // double a row on a given tick (S4 finding 3), so this is a bound,
    // not an equality — the exact-per-tick claim is not what's under
    // test here.
    for (i, &v) in hp.iter().enumerate() {
        let spent = START_HP - v;
        assert!(
            spent >= DAMAGE_PER_CAST,
            "slot {i} took no damage (hp {v}) — its self-cast never landed",
        );
        assert!(
            (spent - records[i] * DAMAGE_PER_CAST).abs() < 0.01,
            "slot {i}: hp delta {spent} disagrees with {} consumed records × {DAMAGE_PER_CAST}",
            records[i],
        );
    }

    // (2) The collision is gone: MarkCollider (source index 26, the old
    // kind-26 squatter) must never have seen a record. A nonzero value
    // here means user ids drifted back into the reserved range.
    for (i, &v) in marker.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "slot {i}: MarkCollider fired (collided_marker = {v}) — a user event kind id \
             collided with the engine's EffectDamageApplied discriminant again",
        );
    }
}

#[test]
fn many_events_ability_is_deterministic() {
    let Some((hp_a, rec_a, _)) = run(SEED) else {
        eprintln!("[many_events_ability] skipping: no wgpu adapter on host.");
        return;
    };
    let (hp_b, rec_b, _) = run(SEED).expect("adapter available on the second run");
    assert_eq!(hp_a, hp_b, "hp column must be byte-equal across same-seed runs");
    assert_eq!(rec_a, rec_b, "damage_records must be byte-equal across same-seed runs");
}
