//! Host-side scaffolding for the ported `tom_probe_runtime` test pins
//! (now hosted in `crates/sims/tests/tom_probe_*.rs`).
//!
//! The auto-emitted `effect_*_applied` injectors on
//! `sims::tom_probe::GeneratedRuntime` only write a chronicle record into
//! the unified event ring — they do NOT dispatch the matching consumer
//! kernel. The compiler-emitted `step()` walks SCHEDULE but only matches
//! `DispatchOp::Kernel(_)`; the belief-consumer entries are
//! `DispatchOp::Indirect{...}` and silently fall through (`_ => {}`).
//!
//! Wiring the Indirect arm in `synthesize_generated_runtime_struct`
//! is blocked on a four-gap coordinated change documented at the
//! catch-all match arm (`crates/dsl_compiler/src/build_helper.rs`).
//! Until that lands, these wrappers hand-roll the inject + consumer
//! dispatch in one method — exactly the pattern the bespoke
//! `tom_probe_runtime/src/lib.rs` uses for its `observe`/`scry`/etc.
//! verb methods.
//!
//! To keep the ported pins self-sufficient TODAY without changing the
//! generated-runtime surface, this module wraps each typed injector with
//! a synchronous companion that:
//!   1. Calls `rt.effect_*_applied(...)` to enqueue the chronicle record.
//!   2. Writes the per-consumer cfg uniform with `event_count = 1` (or
//!      `agent_count` for the reveal multi-target fan-out).
//!   3. Encodes a manual dispatch of the matching consumer kernel.
//!
//! It also re-implements the seed/readback helpers the original
//! `TomProbeState` exposed (the BeliefState SoA columns, agent_pos,
//! agent_creature_type, the disguise SoA columns), and a `decay_step`
//! shim that builds + dispatches the bespoke `belief_decay_wgsl::
//! decay_kernel_wgsl()` kernel out-of-band from the schedule.
//!
//! NOTE: Belief-state byte columns (confidence / type / suspicion) are
//! stored in the auto-emitted runtime as one byte per cell packed 4 to
//! a u32 word (see the `agents_set_beliefs_confidence` helper in each
//! generated WGSL — it does `cell / 4u; byte_in_word = cell % 4u`). The
//! seed/readback helpers below do the byte-level pack/unpack against the
//! u32-storage layout.

#![allow(dead_code)]

use sims::tom_probe::GeneratedRuntime;

pub const EVENT_STRIDE_U32: usize = 11;

/// Construct a `GeneratedRuntime` if the host has a wgpu adapter; return
/// `None` otherwise so the test logs a skip message and exits 0. Mirrors
/// the `try_new` helper from each per-fixture-crate test pin.
pub fn try_new(seed: u64, agent_count: u32) -> Option<GeneratedRuntime> {
    GeneratedRuntime::try_new(seed, agent_count)
}

/// Number of cells in the BeliefState SoA grid (= `agent_count^2`).
pub fn cell_count(rt: &GeneratedRuntime) -> usize {
    (rt.agent_count as usize) * (rt.agent_count as usize)
}

// ------------------------------------------------------------------
// Seeders — write host-supplied bytes into the GPU buffers.
// ------------------------------------------------------------------

pub fn seed_beliefs_flags(rt: &mut GeneratedRuntime, values: &[u32]) {
    assert_eq!(values.len(), cell_count(rt));
    rt.gpu.queue.write_buffer(
        &rt.beliefs_flags_buf,
        0,
        bytemuck::cast_slice(values),
    );
}

pub fn seed_beliefs_pos(rt: &mut GeneratedRuntime, values: &[[f32; 4]]) {
    assert_eq!(values.len(), cell_count(rt));
    rt.gpu.queue.write_buffer(
        &rt.beliefs_pos_buf,
        0,
        bytemuck::cast_slice(values),
    );
}

/// Seed `beliefs_type` (one byte per cell, packed 4-per-u32). The buffer
/// is sized as `cell_count * 4` bytes (one word per cell at the host
/// layer for simplicity), so we widen each input byte into the low byte
/// of its own u32 word.
///
/// CAVEAT: the WGSL stores 4 cells per u32 (`cell / 4u`), so we have to
/// build the packed words explicitly. See file-level NOTE.
pub fn seed_beliefs_type(rt: &mut GeneratedRuntime, values: &[u8]) {
    seed_packed_byte_column(rt, &rt.beliefs_type_buf, values);
}

pub fn seed_beliefs_confidence(rt: &mut GeneratedRuntime, values: &[u8]) {
    seed_packed_byte_column(rt, &rt.beliefs_confidence_buf, values);
}

pub fn seed_beliefs_suspicion(rt: &mut GeneratedRuntime, values: &[u8]) {
    seed_packed_byte_column(rt, &rt.beliefs_suspicion_buf, values);
}

pub fn seed_beliefs_tick(rt: &mut GeneratedRuntime, values: &[u32]) {
    assert_eq!(values.len(), cell_count(rt));
    rt.gpu.queue.write_buffer(
        &rt.beliefs_tick_buf,
        0,
        bytemuck::cast_slice(values),
    );
}

pub fn seed_agent_pos(rt: &mut GeneratedRuntime, values: &[[f32; 4]]) {
    assert_eq!(values.len(), rt.agent_count as usize);
    rt.gpu.queue.write_buffer(
        &rt.agent_pos_buf,
        0,
        bytemuck::cast_slice(values),
    );
}

/// Seed agent SoA `creature_type` column. The buffer is `array<u32>`
/// (one u32 per agent); each input byte zero-extends into the low byte.
pub fn seed_agent_creature_type(rt: &mut GeneratedRuntime, values: &[u8]) {
    assert_eq!(values.len(), rt.agent_count as usize);
    let widened: Vec<u32> = values.iter().map(|&b| b as u32).collect();
    rt.gpu.queue.write_buffer(
        &rt.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&widened),
    );
}

// Packed-byte SoA seeder. The cell-column buffer holds one u32 word per
// 4 cells (matching the WGSL `cell / 4u; byte_in_word = cell % 4u`
// indexing). This packs `values` into the right word/byte slots and
// writes the whole prefix at once.
fn seed_packed_byte_column(
    rt: &GeneratedRuntime,
    buf: &wgpu::Buffer,
    values: &[u8],
) {
    let cells = cell_count(rt);
    assert_eq!(values.len(), cells);
    let word_count = cells.div_ceil(4);
    let mut words: Vec<u32> = vec![0u32; word_count];
    for (i, &b) in values.iter().enumerate() {
        let w = i / 4;
        let s = (i % 4) * 8;
        words[w] |= (b as u32) << s;
    }
    rt.gpu
        .queue
        .write_buffer(buf, 0, bytemuck::cast_slice(&words));
}

// ------------------------------------------------------------------
// Readbacks — copy buffer → staging → host slice.
// ------------------------------------------------------------------

pub fn read_beliefs_flags(rt: &GeneratedRuntime) -> Vec<u32> {
    let cells = cell_count(rt);
    read_u32_buffer(&rt.gpu, &rt.beliefs_flags_buf, cells)
}

/// Read the materialized-view storage that the .sim's
/// `view beliefs_flags(...)` fold writes into. This is a SEPARATE
/// physical buffer from `beliefs_flags_buf` in the auto-emitted runtime
/// (the original `tom_probe_runtime` shared a single buffer across both
/// the view and the consumer-side `agents.set_beliefs_flags(...)`
/// writes; the mega-crate split them when the view-fold codegen landed).
pub fn read_view_storage_primary(rt: &GeneratedRuntime) -> Vec<u32> {
    let cells = cell_count(rt);
    // Per-view storage post the aliasing-gap fix — `beliefs_flags`
    // is the materialized view in tom_probe; backing buffer is
    // `view_storage_beliefs_flags_primary_buf`.
    read_u32_buffer(&rt.gpu, &rt.view_storage_beliefs_flags_primary_buf, cells)
}

pub fn read_beliefs_pos(rt: &GeneratedRuntime) -> Vec<[f32; 4]> {
    let cells = cell_count(rt);
    let bytes = (cells * 16) as u64;
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tom_probe_helpers::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        rt.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::pos_readback"),
        });
    encoder.copy_buffer_to_buffer(
        &rt.beliefs_pos_buf,
        0,
        &staging,
        0,
        bytes,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out: Vec<[f32; 4]> = {
        let view = slice.get_mapped_range();
        let raw: &[[f32; 4]] = bytemuck::cast_slice(&view);
        raw[..cells].to_vec()
    };
    staging.unmap();
    out
}

pub fn read_beliefs_type(rt: &GeneratedRuntime) -> Vec<u8> {
    read_packed_byte_column(rt, &rt.beliefs_type_buf)
}

pub fn read_beliefs_confidence(rt: &GeneratedRuntime) -> Vec<u8> {
    read_packed_byte_column(rt, &rt.beliefs_confidence_buf)
}

pub fn read_beliefs_suspicion(rt: &GeneratedRuntime) -> Vec<u8> {
    read_packed_byte_column(rt, &rt.beliefs_suspicion_buf)
}

pub fn read_beliefs_tick(rt: &GeneratedRuntime) -> Vec<u32> {
    let cells = cell_count(rt);
    read_u32_buffer(&rt.gpu, &rt.beliefs_tick_buf, cells)
}

pub fn read_agent_disguise_expires_at_tick(rt: &GeneratedRuntime) -> Vec<u32> {
    let n = rt.agent_count as usize;
    read_u32_buffer(&rt.gpu, &rt.agent_disguise_expires_at_tick_buf, n)
}

pub fn read_agent_disguise_fake_type(rt: &GeneratedRuntime) -> Vec<u32> {
    let n = rt.agent_count as usize;
    read_u32_buffer(&rt.gpu, &rt.agent_disguise_fake_type_buf, n)
}

fn read_u32_buffer(
    gpu: &engine::GpuContext,
    buf: &wgpu::Buffer,
    word_count: usize,
) -> Vec<u32> {
    let bytes = (word_count.max(4) * 4) as u64;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tom_probe_helpers::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::u32_readback"),
        });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let raw: &[u32] = bytemuck::cast_slice(&view);
        raw[..word_count].to_vec()
    };
    staging.unmap();
    out
}

fn read_packed_byte_column(
    rt: &GeneratedRuntime,
    buf: &wgpu::Buffer,
) -> Vec<u8> {
    let cells = cell_count(rt);
    let word_count = cells.div_ceil(4);
    let words = read_u32_buffer(&rt.gpu, buf, word_count);
    let mut out = Vec::with_capacity(cells);
    for i in 0..cells {
        let w = words[i / 4];
        let b = ((w >> ((i % 4) * 8)) & 0xFF) as u8;
        out.push(b);
    }
    out
}

// ------------------------------------------------------------------
// Synchronous verb wrappers — inject + dispatch consumer kernel.
// ------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConsumerCfg {
    event_count: u32,
    tick: u32,
    seed: u32,
    agent_cap: u32,
}

fn write_consumer_cfg(
    rt: &GeneratedRuntime,
    cfg_buf: &wgpu::Buffer,
    event_count: u32,
) {
    let cfg = ConsumerCfg {
        event_count,
        tick: rt.tick as u32,
        seed: rt.seed as u32,
        agent_cap: rt.agent_count,
    };
    rt.gpu
        .queue
        .write_buffer(cfg_buf, 0, bytemuck::bytes_of(&cfg));
}

/// Synchronous `observe`: enqueue an EffectObserveApplied chronicle
/// record (kind=64) and immediately dispatch the
/// `physics_ApplyObserveBeliefUpdate` consumer kernel. After the call,
/// the 4 BeliefState columns the consumer writes (pos, type, tick,
/// confidence) reflect the observation at `[observer * N + target]`.
pub fn dispatch_observe(rt: &mut GeneratedRuntime, observer: u32, target: u32) {
    use sims::tom_probe::dispatch;
    use sims::tom_probe::physics_ApplyObserveBeliefUpdate as obs;

    rt.effect_observe_applied(observer, target, 0);
    write_consumer_cfg(rt, &rt.cfg_physics_ApplyObserveBeliefUpdate_buf, 1);

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::dispatch_observe"),
        },
    );
    let bindings = obs::PhysicsApplyObserveBeliefUpdateBindings {
        event_ring: rt.event_ring.ring(),
        event_tail: rt.event_ring.tail(),
        agent_pos: &rt.agent_pos_buf,
        agent_disguise_expires_at_tick: &rt.agent_disguise_expires_at_tick_buf,
        agent_disguise_fake_type: &rt.agent_disguise_fake_type_buf,
        agent_creature_type: &rt.agent_creature_type_buf,
        beliefs_pos: &rt.beliefs_pos_buf,
        beliefs_type: &rt.beliefs_type_buf,
        beliefs_tick: &rt.beliefs_tick_buf,
        beliefs_confidence: &rt.beliefs_confidence_buf,
        cfg: &rt.cfg_physics_ApplyObserveBeliefUpdate_buf,
    };
    dispatch::dispatch_physics_applyobservebeliefupdate(
        &mut rt.cache,
        &bindings,
        &rt.gpu.device,
        &mut encoder,
        1,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
}

/// Synchronous `scry`: enqueue an EffectScryApplied chronicle record
/// (kind=65) and dispatch the `physics_ApplyScryBeliefUpdate` consumer.
/// Copies all 6 BeliefState columns from `[target_observer * N + subject]`
/// to `[observer * N + subject]`.
pub fn dispatch_scry(
    rt: &mut GeneratedRuntime,
    observer: u32,
    target_observer: u32,
    subject: u32,
) {
    use sims::tom_probe::dispatch;
    use sims::tom_probe::physics_ApplyScryBeliefUpdate as scry;

    // Per `cpu_chronicle_reference::Scry`: slot 2 = caster (= observer),
    // slot 3 = target (= subject), slot 4 = target_observer, slot 5 = subject_idx.
    rt.effect_scry_applied(observer, subject, target_observer, subject);
    write_consumer_cfg(rt, &rt.cfg_physics_ApplyScryBeliefUpdate_buf, 1);

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::dispatch_scry"),
        },
    );
    let bindings = scry::PhysicsApplyScryBeliefUpdateBindings {
        event_ring: rt.event_ring.ring(),
        event_tail: rt.event_ring.tail(),
        beliefs_pos: &rt.beliefs_pos_buf,
        beliefs_type: &rt.beliefs_type_buf,
        beliefs_tick: &rt.beliefs_tick_buf,
        beliefs_confidence: &rt.beliefs_confidence_buf,
        beliefs_suspicion: &rt.beliefs_suspicion_buf,
        beliefs_flags: &rt.beliefs_flags_buf,
        cfg: &rt.cfg_physics_ApplyScryBeliefUpdate_buf,
    };
    dispatch::dispatch_physics_applyscrybeliefupdate(
        &mut rt.cache,
        &bindings,
        &rt.gpu.device,
        &mut encoder,
        1,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
}

/// Synchronous `reveal`: enqueues N EffectRevealApplied chronicle
/// records (one per observer) and dispatches the
/// `physics_ApplyRevealBeliefUpdate` consumer with `event_count = N`.
/// Mirrors the original tom_probe_runtime hand-rolled fan-out.
pub fn dispatch_reveal(rt: &mut GeneratedRuntime, caster: u32, subject: u32) {
    use sims::tom_probe::dispatch;
    use sims::tom_probe::physics_ApplyRevealBeliefUpdate as rev;

    let n = rt.agent_count;

    // First call resets the tail estimate. Subsequent appends bump
    // through slots 1..N. Per `cpu_chronicle_reference::Reveal`:
    //   slot 2 = caster, slot 3 = target (= observer cell), slot 4 = subject_idx.
    rt.effect_reveal_applied(caster, 0, subject);
    for observer in 1..n {
        // Append (no tail reset) for slots 1..N. We avoid re-using
        // effect_reveal_applied because it would reset the tail again;
        // build the record + append via the engine helper directly.
        let mut record = [0u32; EVENT_STRIDE_U32];
        record[0] = 66;
        record[1] = rt.tick as u32;
        record[2] = caster;
        record[3] = observer;
        record[4] = subject;
        rt.event_ring
            .append_chronicle_record(&rt.gpu.queue, &record);
    }

    write_consumer_cfg(rt, &rt.cfg_physics_ApplyRevealBeliefUpdate_buf, n);

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::dispatch_reveal"),
        },
    );
    let bindings = rev::PhysicsApplyRevealBeliefUpdateBindings {
        event_ring: rt.event_ring.ring(),
        event_tail: rt.event_ring.tail(),
        beliefs_pos: &rt.beliefs_pos_buf,
        beliefs_type: &rt.beliefs_type_buf,
        beliefs_tick: &rt.beliefs_tick_buf,
        beliefs_confidence: &rt.beliefs_confidence_buf,
        beliefs_suspicion: &rt.beliefs_suspicion_buf,
        beliefs_flags: &rt.beliefs_flags_buf,
        cfg: &rt.cfg_physics_ApplyRevealBeliefUpdate_buf,
    };
    dispatch::dispatch_physics_applyrevealbeliefupdate(
        &mut rt.cache,
        &bindings,
        &rt.gpu.device,
        &mut encoder,
        n,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
}

/// Synchronous `decoy`: enqueues an EffectDecoyApplied chronicle record
/// (kind=68) and dispatches the `physics_ApplyDecoyBeliefUpdate` consumer.
pub fn dispatch_decoy(
    rt: &mut GeneratedRuntime,
    caster: u32,
    target: u32,
    subject_idx: u32,
    fake_pos: u32,
) {
    use sims::tom_probe::dispatch;
    use sims::tom_probe::physics_ApplyDecoyBeliefUpdate as decoy;

    rt.effect_decoy_applied(caster, target, subject_idx, fake_pos);
    write_consumer_cfg(rt, &rt.cfg_physics_ApplyDecoyBeliefUpdate_buf, 1);

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::dispatch_decoy"),
        },
    );
    let bindings = decoy::PhysicsApplyDecoyBeliefUpdateBindings {
        event_ring: rt.event_ring.ring(),
        event_tail: rt.event_ring.tail(),
        beliefs_pos: &rt.beliefs_pos_buf,
        beliefs_type: &rt.beliefs_type_buf,
        beliefs_tick: &rt.beliefs_tick_buf,
        beliefs_confidence: &rt.beliefs_confidence_buf,
        cfg: &rt.cfg_physics_ApplyDecoyBeliefUpdate_buf,
    };
    dispatch::dispatch_physics_applydecoybeliefupdate(
        &mut rt.cache,
        &bindings,
        &rt.gpu.device,
        &mut encoder,
        1,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
}

/// Synchronous `erase_belief`: enqueues an EffectEraseBeliefApplied
/// chronicle record (kind=69) and dispatches the fused
/// `physics_ApplyEraseBeliefUpdate_and_ApplyDisguise` consumer (the
/// compiler fuses the two `@phase(post)` PerEvent rules; the kind tag
/// gates which arm fires).
pub fn dispatch_erase_belief(
    rt: &mut GeneratedRuntime,
    caster: u32,
    target: u32,
    subject_idx: u32,
    fields: u8,
) {
    use sims::tom_probe::dispatch;
    use sims::tom_probe::physics_ApplyEraseBeliefUpdate_and_ApplyDisguise as fused;

    rt.effect_erase_belief_applied(caster, target, subject_idx, fields as u32);
    write_consumer_cfg(
        rt,
        &rt.cfg_physics_ApplyEraseBeliefUpdate_and_ApplyDisguise_buf,
        1,
    );

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::dispatch_erase_belief"),
        },
    );
    let bindings = fused::PhysicsApplyEraseBeliefUpdateAndApplyDisguiseBindings {
        event_ring: rt.event_ring.ring(),
        event_tail: rt.event_ring.tail(),
        agent_disguise_expires_at_tick: &rt.agent_disguise_expires_at_tick_buf,
        agent_disguise_fake_type: &rt.agent_disguise_fake_type_buf,
        beliefs_pos: &rt.beliefs_pos_buf,
        beliefs_type: &rt.beliefs_type_buf,
        beliefs_tick: &rt.beliefs_tick_buf,
        beliefs_confidence: &rt.beliefs_confidence_buf,
        beliefs_suspicion: &rt.beliefs_suspicion_buf,
        beliefs_flags: &rt.beliefs_flags_buf,
        cfg: &rt.cfg_physics_ApplyEraseBeliefUpdate_and_ApplyDisguise_buf,
    };
    dispatch::dispatch_physics_applyerasebeliefupdate_and_applydisguise(
        &mut rt.cache,
        &bindings,
        &rt.gpu.device,
        &mut encoder,
        1,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
}

/// Synchronous `disguise`: packs `(duration_ticks << 8) | fake_type`
/// into a single u32 payload (matching the GPU dispatcher's pack), then
/// enqueues an EffectDisguiseApplied chronicle record (kind=67) and
/// dispatches the fused
/// `physics_ApplyEraseBeliefUpdate_and_ApplyDisguise` consumer.
pub fn dispatch_disguise(
    rt: &mut GeneratedRuntime,
    caster: u32,
    fake_type: u8,
    duration_ticks: u32,
) {
    use sims::tom_probe::dispatch;
    use sims::tom_probe::physics_ApplyEraseBeliefUpdate_and_ApplyDisguise as fused;

    let payload = (duration_ticks << 8) | (fake_type as u32);
    rt.effect_disguise_applied(caster, caster, payload);
    write_consumer_cfg(
        rt,
        &rt.cfg_physics_ApplyEraseBeliefUpdate_and_ApplyDisguise_buf,
        1,
    );

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::dispatch_disguise"),
        },
    );
    let bindings = fused::PhysicsApplyEraseBeliefUpdateAndApplyDisguiseBindings {
        event_ring: rt.event_ring.ring(),
        event_tail: rt.event_ring.tail(),
        agent_disguise_expires_at_tick: &rt.agent_disguise_expires_at_tick_buf,
        agent_disguise_fake_type: &rt.agent_disguise_fake_type_buf,
        beliefs_pos: &rt.beliefs_pos_buf,
        beliefs_type: &rt.beliefs_type_buf,
        beliefs_tick: &rt.beliefs_tick_buf,
        beliefs_confidence: &rt.beliefs_confidence_buf,
        beliefs_suspicion: &rt.beliefs_suspicion_buf,
        beliefs_flags: &rt.beliefs_flags_buf,
        cfg: &rt.cfg_physics_ApplyEraseBeliefUpdate_and_ApplyDisguise_buf,
    };
    dispatch::dispatch_physics_applyerasebeliefupdate_and_applydisguise(
        &mut rt.cache,
        &bindings,
        &rt.gpu.device,
        &mut encoder,
        1,
    );
    rt.gpu.queue.submit(Some(encoder.finish()));
}

// ------------------------------------------------------------------
// Decay step shim — bespoke kernel dispatched out-of-band.
// ------------------------------------------------------------------

/// Per-tick decay phase. Compiles + dispatches the bespoke decay kernel
/// emitted by `dsl_compiler::belief_decay_wgsl::decay_kernel_wgsl()`.
/// Mirrors the original `tom_probe_runtime::TomProbeState::decay_step`
/// (which carried this kernel inline).
///
/// SUBSUMPTION NOTE: a parallel agent is folding decay into the
/// compiler-emitted `@decay` pipeline. Once that lands, this shim can
/// retire — pin tests will instead rely on `step()` running the decay
/// kernel automatically. Until then, tests opt into the manual
/// `decay_step(&mut rt)` call after each `rt.step()`.
pub fn decay_step(rt: &mut GeneratedRuntime) {
    let pipeline = decay_pipeline(rt);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct DecayCfg {
        observer: u32,
        target_or_subject: u32,
        subject: u32,
        agent_count: u32,
        tick: u32,
        cell_count: u32,
        _pad0: u32,
        _pad1: u32,
    }

    let cfg = DecayCfg {
        observer: 0,
        target_or_subject: 0,
        subject: 0,
        agent_count: rt.agent_count,
        tick: rt.tick as u32,
        cell_count: cell_count(rt) as u32,
        _pad0: 0,
        _pad1: 0,
    };
    rt.gpu
        .queue
        .write_buffer(&pipeline.cfg_buf, 0, bytemuck::bytes_of(&cfg));

    let bg = rt.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tom_probe_helpers::decay::bg"),
        layout: &pipeline.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: pipeline.cfg_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rt.beliefs_confidence_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rt.beliefs_tick_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = rt.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("tom_probe_helpers::decay::encoder"),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tom_probe_helpers::decay::pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let cell = cell_count(rt) as u32;
        let word_count = cell.div_ceil(4);
        let workgroups = word_count.div_ceil(64).max(1);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    rt.gpu.queue.submit(Some(encoder.finish()));
}

struct DecayPipeline {
    cfg_buf: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

// Process-wide cache of the decay kernel pipeline keyed on the
// `wgpu::Buffer`'s `global_id` for the decay cfg buffer that the runtime
// owns... wait, we don't have that. The simpler path: cache against the
// device pointer reachable via wgpu's internal handle. Today's tests
// each spin up their own GeneratedRuntime (= own device) but only call
// `decay_step` in a single tight loop, so just rebuild per-call: the
// compile cost is one-time inside that loop because wgpu caches shader
// modules, and pipeline creation amortises against the 100+ dispatches.
//
// The Box::leak avoids a borrow-lifetime tangle with `rt` (we want the
// returned pipeline to outlive the immutable borrow of `rt` so the
// caller can subsequently mutably borrow `rt`'s buffers).
fn decay_pipeline(rt: &GeneratedRuntime) -> &'static DecayPipeline {
    let cfg_buf = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tom_probe_helpers::decay::cfg"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bgl = rt.gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tom_probe_helpers::decay::bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let wgsl_src = dsl_compiler::belief_decay_wgsl::decay_kernel_wgsl();
    let module = rt.gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tom_probe_helpers::decay::wgsl"),
        source: wgpu::ShaderSource::Wgsl(wgsl_src.into()),
    });
    let pl = rt.gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tom_probe_helpers::decay::pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = rt.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("tom_probe_helpers::decay::pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("cs_decay"),
        compilation_options: Default::default(),
        cache: None,
    });

    let dp: &'static DecayPipeline =
        Box::leak(Box::new(DecayPipeline { cfg_buf, pipeline, bgl }));
    dp
}
