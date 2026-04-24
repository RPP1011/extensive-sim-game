//! Task #79 SP-6 — modify_standing fires on the resident physics kernel.
//!
//! Seeds `EffectStandingDelta` events directly into the resident physics
//! kernel's input ring (same harness shape as
//! `cold_state_gold_transfer.rs::transfer_gold_fires_on_resident_kernel`),
//! dispatches `run_batch_resident`, reads back the GPU standing
//! storage, and asserts:
//!
//!   1. Find-or-reserve: a seeded delta lands at the canonical owner
//!      row with the expected value + symmetry
//!      (`standing(a,b) == standing(b,a)`).
//!   2. Update-in-place: subsequent events on the same pair accumulate.
//!   3. Clamp: over-saturating with large positive deltas pins at +1000.
//!   4. Canonicalisation: (b=5, a=2) lands on owner_id=2's row with
//!      other=5 — not on owner_id=5's row.
//!
//! Intentionally bypasses the `GpuBackend::step_batch` end-to-end flow
//! because the Phase 2 chronicle tests uncovered a bug in step_batch's
//! apply_event_ring → events_in wiring (Task #68). When that bug is
//! fixed, a full step_batch-driven standing test can be added
//! alongside re-enabling the chronicle tests.
//!
//! What this test DOES cover:
//!   * SP-4: resident physics bind group 18 / 19 point at the standing
//!     storage buffers; WGSL `state_adjust_standing` find-or-reserve-
//!     or-evict fold lands correctly.
//!   * End-to-end arithmetic: positive delta accumulation + clamp.
//!   * Symmetry: the canonical (min, max) rule is enforced.
//!
//! What this test does NOT cover (tracked as deferred):
//!   * The `step_batch` flow (cast → cascade → modify_standing emit).
//!   * Negative-delta sign extension through `event_ring::pack_event` —
//!     the event ring packs i16 delta as `(x as u16) as u32` which
//!     zero-extends rather than sign-extends. The kernel emits events
//!     with `bitcast<u32>(i32)` which IS correct on the GPU-emit
//!     side; pack_event has a latent sign-ext bug that would cause
//!     negative deltas seeded from the CPU-side test harness to
//!     become large positive values on the GPU. Tracked as a separate
//!     follow-up.
//!   * Eviction policy (weakest-|value| replace) — exercised in unit
//!     tests under `view_storage_symmetric_pair`; a future test could
//!     saturate the 8-slot row and verify eviction end-to-end.
//!   * `SimState.views.standing` snapshot merge (covered by
//!     `snapshot_merges_standing_into_state` from SP-5).

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::event_ring::{
    EventKindTag, EventRecord, GpuChronicleRing, GpuEventRing, PAYLOAD_WORDS,
};
use engine_gpu::gpu_util::indirect::IndirectArgsBuffer;
use engine_gpu::physics::{
    pack_agent_slots, GpuEffectOp, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PhysicsKernel,
    MAX_ABILITIES, MAX_EFFECTS,
};
use engine_gpu::view_storage_symmetric_pair::{
    StandingEdgeGpu, ViewStorageSymmetricPair, STANDING_K,
};
use glam::Vec3;
use wgpu::util::DeviceExt;

fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("adapter");
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cold_state_standing::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");
        (device, queue)
    })
}

fn load_physics() -> dsl_compiler::ir::Compilation {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.push("assets/sim");
    let mut merged = Program { decls: Vec::new() };
    for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
        let src = std::fs::read_to_string(root.join(f)).expect("read sim");
        merged
            .decls
            .extend(dsl_compiler::parse(&src).expect("parse").decls);
    }
    dsl_compiler::compile_ast(merged).expect("resolve")
}

fn build_fixture() -> SimState {
    // Kernel-level microfixture per SP-6 scope: `agent_cap = 8` is
    // enough to exercise canonical (owner, other) slot routing
    // against the BGL without paying the N=20k setup cost. Two
    // humans (IDs 1 & 2), two wolves (IDs 3 & 4); rest are unspawned.
    let mut state = SimState::new(8, 0xDEAD_BEEF);
    for i in 0..2 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32 * 5.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");
    }
    for i in 0..2 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(50.0 + i as f32 * 5.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("wolf spawn");
    }
    state
}

fn upload_storage<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Pack an `EffectStandingDelta`-shaped event using a raw i32 delta
/// (avoids the i16 sign-extension quirk in `event_ring::pack_event`).
/// Layout must match the physics emitter's `bitcast<i32>(payload[2])`
/// so the GPU sees the same `delta: i32` the CPU-side modify_standing
/// rule would see.
fn pack_standing_delta(a: AgentId, b: AgentId, delta: i32, tick: u32) -> EventRecord {
    let mut payload = [0u32; PAYLOAD_WORDS];
    payload[0] = a.raw();
    payload[1] = b.raw();
    payload[2] = delta as u32; // bit-preserving cast; GPU bitcasts back to i32
    EventRecord::new(EventKindTag::EffectStandingDelta, tick, payload)
}

/// Readback the GPU standing storage into a Vec<StandingEdgeGpu>.
fn readback_records(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    records: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<StandingEdgeGpu> {
    let bytes = (agent_cap as usize)
        * (STANDING_K as usize)
        * std::mem::size_of::<StandingEdgeGpu>();
    engine_gpu::gpu_util::readback::readback_typed::<StandingEdgeGpu>(
        device, queue, records, bytes,
    )
    .expect("records readback")
}

fn readback_counts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    counts: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<u32> {
    let bytes = (agent_cap as usize) * 4;
    engine_gpu::gpu_util::readback::readback_typed::<u32>(device, queue, counts, bytes)
        .expect("counts readback")
}

/// Dispatch the resident physics kernel with a single event against
/// the given standing storage. Returns after the queue submit +
/// poll(Wait) completes.
fn dispatch_one_event(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &mut PhysicsKernel,
    state: &SimState,
    ev: EventRecord,
    standing_records: &wgpu::Buffer,
    standing_counts: &wgpu::Buffer,
) {
    let agent_cap = state.agent_cap();
    let agent_slots = pack_agent_slots(state);
    let abilities = PackedAbilityRegistry::empty();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];

    let agents_buf = upload_storage(device, "standing::agents", &agent_slots);
    let ab_known = upload_storage(device, "standing::ab_known", &abilities.known);
    let ab_cd = upload_storage(device, "standing::ab_cd", &abilities.cooldown);
    let ab_ecount = upload_storage(device, "standing::ab_ecount", &abilities.effects_count);
    let ab_effects_buf: &[GpuEffectOp] = &abilities.effects;
    let ab_effects_buf = upload_storage(device, "standing::ab_effects", ab_effects_buf);
    let kin_buf = upload_storage(device, "standing::kin", &kin_lists);
    let nearest_buf = upload_storage(device, "standing::nearest", &nearest_hostile);

    let events_in = vec![ev];
    let events_in_buf = upload_storage(device, "standing::events_in", &events_in);

    let event_ring = GpuEventRing::new(device, 4096);
    event_ring.reset(queue);
    let chronicle_ring = GpuChronicleRing::new(device, 4096);

    let indirect = IndirectArgsBuffer::new(device, 4);
    {
        let seed = [engine_gpu::gpu_util::indirect::IndirectArgs { x: 1, y: 1, z: 1 }];
        queue.write_buffer(indirect.buffer(), 0, bytemuck::cast_slice(&seed));
    }
    let num_events_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("standing::num_events_buf"),
        contents: bytemuck::cast_slice(&[1u32, 0u32, 0u32, 0u32]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let cfg = PhysicsCfg {
        num_events: 1,
        agent_cap,
        max_abilities: MAX_ABILITIES as u32,
        max_effects: MAX_EFFECTS as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };

    let sim_cfg_snapshot = engine_gpu::sim_cfg::SimCfg::from_state(state);
    let sim_cfg_buf = engine_gpu::sim_cfg::create_sim_cfg_buffer(device);
    engine_gpu::sim_cfg::upload_sim_cfg(queue, &sim_cfg_buf, &sim_cfg_snapshot);

    // Gold buf unused by this test but required by the BGL (slot 17).
    let gold_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("standing::gold_buf"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Memory view storage: unused by this test but required by the
    // BGL (slots 20 / 21).
    let memory_records_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("standing::memory_records"),
        size: (agent_cap as u64 * 64 * 24).max(24),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let memory_cursors_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("standing::memory_cursors"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Alive bitmap: pack all-alive (agent_cap bits set).
    let alive_bitmap_buf =
        engine_gpu::alive_bitmap::create_alive_bitmap_buffer(device, agent_cap);
    {
        let words = engine_gpu::alive_bitmap::alive_bitmap_words(agent_cap) as usize;
        let mut packed = vec![0u32; words.max(1)];
        for slot in 0..agent_cap as usize {
            packed[slot >> 5] |= 1u32 << (slot & 31);
        }
        queue.write_buffer(&alive_bitmap_buf, 0, bytemuck::cast_slice(&packed));
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("standing::encoder"),
    });
    kernel
        .run_batch_resident(
            device,
            queue,
            &mut encoder,
            &agents_buf,
            &ab_known,
            &ab_cd,
            &ab_ecount,
            &ab_effects_buf,
            &kin_buf,
            &nearest_buf,
            &events_in_buf,
            &event_ring,
            &chronicle_ring,
            &indirect,
            &num_events_buf,
            &sim_cfg_buf,
            &gold_buf,
            standing_records,
            standing_counts,
            &memory_records_buf,
            &memory_cursors_buf,
            &alive_bitmap_buf,
            None, // per_rule_counter_buf (research mode only)
            0, // read_slot
            1, // write_slot
            cfg,
        )
        .expect("run_batch_resident");
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);
}

#[test]
fn modify_standing_fires_on_resident_kernel() {
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let state = build_fixture();
    let agent_cap = state.agent_cap();

    // SP-4's real `state_adjust_standing` body writes into these.
    let storage = ViewStorageSymmetricPair::new(&device, agent_cap, STANDING_K);
    // Buffers are zero-init by wgpu; explicit empty upload guarantees
    // the counts buffer also reads as zeros on the device.
    let cpu_empty = engine::generated::views::standing::Standing::new();
    storage.upload_from_cpu(&queue, &cpu_empty);

    // --- 1. Find-or-reserve: seed a single EffectStandingDelta. ---
    const DELTA_1: i32 = 42;
    let a = AgentId::new(1).unwrap();
    let b = AgentId::new(3).unwrap();
    let ev = pack_standing_delta(a, b, DELTA_1, 0);
    dispatch_one_event(
        &device,
        &queue,
        &mut kernel,
        &state,
        ev,
        &storage.records_buf,
        &storage.counts_buf,
    );

    // Read back raw GPU state. Canonical owner for (1, 3) is owner_slot
    // = 0 (owner_id=1); other = 3.
    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let counts = readback_counts(&device, &queue, &storage.counts_buf, agent_cap);

    assert_eq!(
        counts[0], 1,
        "owner_slot 0 count should be 1 after first delta",
    );
    let row0_base = 0usize;
    assert_eq!(records[row0_base].other, 3, "owner 1 row slot 0: other=3");
    assert_eq!(
        records[row0_base].value,
        DELTA_1,
        "initial delta landed",
    );

    // Symmetry through CPU rebuild.
    let mut cpu_rebuild = engine::generated::views::standing::Standing::new();
    storage
        .readback_into_cpu(&device, &queue, &mut cpu_rebuild)
        .expect("readback_into_cpu");
    assert_eq!(
        cpu_rebuild.get(a, b),
        DELTA_1,
        "get(a,b) via CPU rebuild",
    );
    assert_eq!(
        cpu_rebuild.get(b, a),
        DELTA_1,
        "symmetry: get(b,a) == get(a,b)",
    );

    // --- 2. Update-in-place: same pair, another positive delta. ---
    const DELTA_2: i32 = 100;
    let ev2 = pack_standing_delta(a, b, DELTA_2, 1);
    dispatch_one_event(
        &device,
        &queue,
        &mut kernel,
        &state,
        ev2,
        &storage.records_buf,
        &storage.counts_buf,
    );
    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let counts = readback_counts(&device, &queue, &storage.counts_buf, agent_cap);
    assert_eq!(
        counts[0], 1,
        "count stays 1 after update-in-place",
    );
    assert_eq!(
        records[row0_base].value,
        DELTA_1 + DELTA_2,
        "update-in-place: values accumulate (42 + 100 = 142)",
    );

    // --- 3. Clamp to +1000: over-saturate with 9 positive bursts. ---
    // Current value is 142; bump by 500 nine times → would be 4642
    // uncapped. Must be clamped to 1000.
    for t in 0..9u32 {
        let ev = pack_standing_delta(a, b, 500, 2 + t);
        dispatch_one_event(
            &device,
            &queue,
            &mut kernel,
            &state,
            ev,
            &storage.records_buf,
            &storage.counts_buf,
        );
    }
    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    assert_eq!(
        records[row0_base].value, 1000,
        "value clamped to +1000 after over-saturation",
    );

    // --- 4. New pair: reserve a fresh slot on the same owner row. ---
    let c = AgentId::new(4).unwrap(); // pair (1, 4) → owner 1 row slot 1.
    let ev = pack_standing_delta(a, c, 75, 100);
    dispatch_one_event(
        &device,
        &queue,
        &mut kernel,
        &state,
        ev,
        &storage.records_buf,
        &storage.counts_buf,
    );
    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let counts = readback_counts(&device, &queue, &storage.counts_buf, agent_cap);
    assert_eq!(counts[0], 2, "count advances to 2 after new-pair reserve");
    assert_eq!(records[row0_base + 1].other, 4, "new slot: other=4");
    assert_eq!(records[row0_base + 1].value, 75, "new slot: value=75");
    // Original (1, 3) unchanged after the new-pair dispatch.
    assert_eq!(records[row0_base].other, 3);
    assert_eq!(records[row0_base].value, 1000);
}

#[test]
fn modify_standing_canonicalises_pair_order() {
    // Seeding (a=5, b=2) with delta 30 must land on the canonical
    // owner row (owner_slot=1, i.e. owner_id=2), with other=5 — NOT
    // on owner_slot=4 with other=2.
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let state = build_fixture();
    let agent_cap = state.agent_cap();
    let storage = ViewStorageSymmetricPair::new(&device, agent_cap, STANDING_K);
    let cpu_empty = engine::generated::views::standing::Standing::new();
    storage.upload_from_cpu(&queue, &cpu_empty);

    let a = AgentId::new(5).unwrap();
    let b = AgentId::new(2).unwrap();
    let ev = pack_standing_delta(a, b, 30, 0);
    dispatch_one_event(
        &device,
        &queue,
        &mut kernel,
        &state,
        ev,
        &storage.records_buf,
        &storage.counts_buf,
    );

    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let counts = readback_counts(&device, &queue, &storage.counts_buf, agent_cap);

    // Owner slot 1 (owner_id = 2) should have the record. Owner
    // slot 4 (owner_id = 5) must not.
    assert_eq!(counts[1], 1, "owner_slot 1 (id=2) got the record");
    assert_eq!(counts[4], 0, "owner_slot 4 (id=5) stays empty");
    let row1_base = STANDING_K as usize;
    assert_eq!(records[row1_base].other, 5, "other = 5 (max)");
    assert_eq!(records[row1_base].value, 30);
}
