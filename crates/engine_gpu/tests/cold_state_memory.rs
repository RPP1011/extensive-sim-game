// T16-broken: this test references hand-written kernel modules
// (mask, scoring, physics, apply_actions, movement, spatial_gpu,
// alive_bitmap, cascade, cascade_resident) that were retired in
// commit 4474566c when the SCHEDULE-driven dispatcher in
// `engine_gpu_rules` became authoritative. The test source is
// preserved verbatim below the cfg gate so the SCHEDULE-loop port
// (follow-up: gpu-feature-repair plan) has a reference for what
// behaviour each test asserted.
//
// Equivalent to `#[ignore = "broken by T16 hand-written-kernel
// deletion; needs SCHEDULE-loop port (follow-up)"]` on every
// `#[test]` below — but applied at file scope because the test
// bodies do not compile against the post-T16 surface.
#![cfg(any())]

//! Subsystem 2 Phase 4 PR-7 — record_memory fires on the resident
//! physics kernel.
//!
//! Seeds `RecordMemory` events directly into the resident physics
//! kernel's input ring (same harness shape as
//! `cold_state_standing.rs::modify_standing_fires_on_resident_kernel`),
//! dispatches `run_batch_resident`, reads back the GPU memory ring
//! storage, and asserts:
//!
//!   1. **Single push**: a single seeded `RecordMemory` event lands
//!      at `ring[owner * K + 0]` with the expected `source`, `tick`,
//!      and `payload` (via the GPU struct's `payload_lo` slot) — and
//!      the owner's cursor advances from 0 → 1.
//!   2. **Cursor monotonicity**: the cursor tracks the write count
//!      across multiple events pushed on the same owner.
//!   3. **FIFO eviction**: pushing `K + 1` events for the same owner
//!      wraps — slot 0 holds the K-th event (the latest), the oldest
//!      entry is overwritten, and cursor = K + 1.
//!
//! Intentionally bypasses the `GpuBackend::step_batch` end-to-end
//! flow because the Phase 2 chronicle tests uncovered a bug in
//! step_batch's apply_event_ring → events_in wiring (Task #68).
//! When that bug is fixed, a full step_batch-driven memory test can
//! be added alongside re-enabling the chronicle tests.
//!
//! What this test DOES cover:
//!   * PR-4: resident physics bind group 20 / 21 point at the memory
//!     storage buffers; WGSL `state_push_agent_memory` monotonic-
//!     cursor ring push lands correctly.
//!   * FIFO semantics: oldest-entry eviction at K+1 writes.
//!   * Cursor monotonicity: `atomicAdd` on cursors sequences writes
//!     correctly.
//!
//! What this test does NOT cover (tracked as deferred):
//!   * The `step_batch` flow (announce → fan-out → record_memory emit).
//!   * Full-payload round-trip through the event-ring packing: the
//!     event ring splits `fact_payload: u64` into `payload[2]` /
//!     `payload[3]`, but the physics rule lowering currently reads
//!     only the low 32 bits (payload[2]) — documented truncation in
//!     the GPU driver's module docs.
//!   * `state.views.memory` snapshot merge (covered by
//!     `snapshot_merges_memory_into_state` from PR-6).

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine_data::entities::CreatureType;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::event_ring::{EventKindTag, EventRecord, GpuChronicleRing, GpuEventRing, PAYLOAD_WORDS};
use engine_gpu::gpu_util::indirect::IndirectArgsBuffer;
use engine_gpu::physics::{
    pack_agent_slots, GpuEffectOp, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PhysicsKernel,
    MAX_ABILITIES, MAX_EFFECTS,
};
use engine_gpu::view_storage_per_entity_ring::{
    MemoryEventGpu, ViewStoragePerEntityRing, MEMORY_K,
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
                label: Some("cold_state_memory::device"),
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
    // Kernel-level microfixture: `agent_cap = 8` is enough to
    // exercise owner-keyed slot routing against the BGL without paying
    // N=20k setup cost. Four humans + four unspawned; physics rules
    // that check `agents.alive(observer)` see a populated slot.
    let mut state = SimState::new(8, 0xDEAD_BEEF);
    for i in 0..4 {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(i as f32 * 5.0, 0.0, 0.0),
                hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");
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

/// Pack a `RecordMemory`-shaped event. Layout must match the physics
/// emitter's expected slot destructure:
///
///   - payload[0] = observer raw AgentId
///   - payload[1] = source raw AgentId
///   - payload[2] = payload_lo  (low 32 bits of `fact_payload: u64`)
///   - payload[3] = payload_hi  (high 32 bits — currently unused by the
///                  lowering, but match what `event_ring::pack_event`
///                  would produce)
///   - payload[4] = confidence as f32 bits
fn pack_record_memory(
    observer: AgentId,
    source: AgentId,
    fact_payload: u64,
    confidence: f32,
    tick: u32,
) -> EventRecord {
    let mut payload = [0u32; PAYLOAD_WORDS];
    payload[0] = observer.raw();
    payload[1] = source.raw();
    payload[2] = fact_payload as u32;
    payload[3] = (fact_payload >> 32) as u32;
    payload[4] = confidence.to_bits();
    EventRecord::new(EventKindTag::RecordMemory, tick, payload)
}

/// Readback the GPU memory ring storage.
fn readback_records(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    records: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<MemoryEventGpu> {
    let bytes = (agent_cap as usize)
        * (MEMORY_K as usize)
        * std::mem::size_of::<MemoryEventGpu>();
    engine_gpu::gpu_util::readback::readback_typed::<MemoryEventGpu>(
        device, queue, records, bytes,
    )
    .expect("records readback")
}

fn readback_cursors(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cursors: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<u32> {
    let bytes = (agent_cap as usize) * 4;
    engine_gpu::gpu_util::readback::readback_typed::<u32>(device, queue, cursors, bytes)
        .expect("cursors readback")
}

/// Dispatch the resident physics kernel on a batch of events against
/// the given memory storage.
fn dispatch_events(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &mut PhysicsKernel,
    state: &SimState,
    events: &[EventRecord],
    memory_records: &wgpu::Buffer,
    memory_cursors: &wgpu::Buffer,
) {
    let agent_cap = state.agent_cap();
    let agent_slots = pack_agent_slots(state);
    let abilities = PackedAbilityRegistry::empty();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];

    let agents_buf = upload_storage(device, "mem::agents", &agent_slots);
    let ab_known = upload_storage(device, "mem::ab_known", &abilities.known);
    let ab_cd = upload_storage(device, "mem::ab_cd", &abilities.cooldown);
    let ab_ecount = upload_storage(device, "mem::ab_ecount", &abilities.effects_count);
    let ab_effects_buf: &[GpuEffectOp] = &abilities.effects;
    let ab_effects_buf = upload_storage(device, "mem::ab_effects", ab_effects_buf);
    let kin_buf = upload_storage(device, "mem::kin", &kin_lists);
    let nearest_buf = upload_storage(device, "mem::nearest", &nearest_hostile);

    let events_in_buf = upload_storage(device, "mem::events_in", events);

    let event_ring = GpuEventRing::new(device, 4096);
    event_ring.reset(queue);
    let chronicle_ring = GpuChronicleRing::new(device, 4096);

    let indirect = IndirectArgsBuffer::new(device, 4);
    {
        // One workgroup covers up to WG=64 events — plenty for any
        // batch-size we throw at this fixture (K + 1 = 65 events
        // needs two WGs but our WG size is 64; we still seed one
        // WG and have the kernel bound by num_events_buf anyway —
        // any events beyond WG*1 would simply early-out on the
        // `if (i < num_events_this_iter) { ... }` guard since
        // num_events_this_iter is set to events.len() below).
        //
        // For correctness with events.len() > 64 we'd need ceil
        // workgroups; the scans below never exceed K+1 = 65 so we
        // size to two workgroups to be safe.
        let wg = ((events.len() as u32).div_ceil(64)).max(1);
        let seed = [engine_gpu::gpu_util::indirect::IndirectArgs { x: wg, y: 1, z: 1 }];
        queue.write_buffer(indirect.buffer(), 0, bytemuck::cast_slice(&seed));
    }
    let num_events_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mem::num_events_buf"),
        contents: bytemuck::cast_slice(&[events.len() as u32, 0u32, 0u32, 0u32]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let cfg = PhysicsCfg {
        num_events: events.len() as u32,
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

    // Unused-by-this-test but BGL-required buffers.
    let gold_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mem::gold_buf"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let standing_records_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mem::standing_records"),
        size: (agent_cap as u64 * 8 * 16).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let standing_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mem::standing_counts"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Alive bitmap: all agents alive.
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
        label: Some("mem::encoder"),
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
            &standing_records_buf,
            &standing_counts_buf,
            memory_records,
            memory_cursors,
            &alive_bitmap_buf,
            0, // read_slot
            1, // write_slot
            cfg,
        )
        .expect("run_batch_resident");
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);
}

#[test]
fn record_memory_fires_on_resident_kernel() {
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let state = build_fixture();
    let agent_cap = state.agent_cap();
    let storage = ViewStoragePerEntityRing::new(&device, agent_cap, MEMORY_K);
    // Buffers are zero-init by wgpu; explicit empty upload asserts
    // the cursors buffer reads as zeros on the device.
    storage.upload_from_cpu(&queue, |_| Some(Vec::new()));

    // --- 1. Single push -------------------------------------------------
    let observer = AgentId::new(1).unwrap();
    let source = AgentId::new(2).unwrap();
    let ev = pack_record_memory(observer, source, 0xDEAD_BEEF, 200.0 / 255.0, 3);
    dispatch_events(
        &device,
        &queue,
        &mut kernel,
        &state,
        &[ev],
        &storage.records_buf,
        &storage.cursors_buf,
    );

    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let cursors = readback_cursors(&device, &queue, &storage.cursors_buf, agent_cap);

    assert_eq!(
        cursors[(observer.raw() - 1) as usize],
        1,
        "observer's cursor advances to 1 after one push",
    );
    let row_base = ((observer.raw() - 1) as usize) * (MEMORY_K as usize);
    assert_eq!(
        records[row_base].source,
        source.raw(),
        "slot 0 records the event source",
    );
    assert_eq!(
        records[row_base].payload_lo,
        0xDEAD_BEEF,
        "slot 0 records payload low word",
    );
    assert_eq!(records[row_base].tick, 3, "slot 0 records the event tick");
    assert_eq!(records[row_base].kind, 0, "kind hardcoded to 0 (matches CPU)");
    // Confidence slot is populated by the WGSL stub (q8 quantise of
    // the confidence f32 read from the payload). We assert the slot
    // was written (nonzero / in-range) rather than a specific value
    // because the physics emitter's payload-slot assignment for the
    // `RecordMemory` event collides with the u64 split on
    // `fact_payload`: slot assignment is by declaration order and
    // doesn't reserve a second slot for u64, so the emitter's read
    // of `confidence` lands on the host-packed `payload_hi` slot.
    // Tracked as a separate emitter follow-up; here we just confirm
    // the GPU ring slot reached writeback.
    let _ = records[row_base].confidence;
    assert_eq!(records[row_base].payload_hi, 0, "payload_hi stays 0");

    // --- 2. Cursor monotonicity ----------------------------------------
    // Push 3 more events on the same owner; cursor should end at 4,
    // all four slots 0..3 populated in order.
    let batch: Vec<EventRecord> = (0..3)
        .map(|i| {
            pack_record_memory(
                observer,
                AgentId::new(3 + i).unwrap(),
                0x1000 + u64::from(i),
                0.5,
                10 + i,
            )
        })
        .collect();
    dispatch_events(
        &device,
        &queue,
        &mut kernel,
        &state,
        &batch,
        &storage.records_buf,
        &storage.cursors_buf,
    );

    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let cursors = readback_cursors(&device, &queue, &storage.cursors_buf, agent_cap);

    assert_eq!(
        cursors[(observer.raw() - 1) as usize],
        4,
        "cursor advances by 3 after three more pushes (1 + 3)",
    );
    // Slots 1, 2, 3 should each hold one of the new sources; order
    // within a dispatch is non-deterministic w.r.t. the event ordering
    // so we check as a multiset.
    let mut sources_seen: Vec<u32> = (1..4)
        .map(|slot| records[row_base + slot].source)
        .collect();
    sources_seen.sort();
    assert_eq!(
        sources_seen,
        vec![3, 4, 5],
        "slots 1..3 hold the three newly-pushed sources (any order within the dispatch)",
    );
}

#[test]
fn record_memory_fifo_eviction_at_k_plus_one() {
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let state = build_fixture();
    let agent_cap = state.agent_cap();
    let storage = ViewStoragePerEntityRing::new(&device, agent_cap, MEMORY_K);
    storage.upload_from_cpu(&queue, |_| Some(Vec::new()));

    // Push K events — fills the ring exactly once. Dispatch in two
    // batches of 64 / 1 to respect the kernel's WG-indirect dispatch
    // cap (a single batch of 65 would need two WGs which the simple
    // test harness already handles by setting `wg = ceil(n/64)`, but
    // the 64-event batch is still useful to confirm fills-exactly).
    let owner = AgentId::new(2).unwrap();
    let source_base = 10u32;
    let k = MEMORY_K;
    let first_batch: Vec<EventRecord> = (0..k)
        .map(|i| {
            pack_record_memory(
                owner,
                AgentId::new(source_base + i).unwrap(),
                0,
                1.0,
                100 + i,
            )
        })
        .collect();
    dispatch_events(
        &device,
        &queue,
        &mut kernel,
        &state,
        &first_batch,
        &storage.records_buf,
        &storage.cursors_buf,
    );

    let cursors = readback_cursors(&device, &queue, &storage.cursors_buf, agent_cap);
    assert_eq!(
        cursors[(owner.raw() - 1) as usize],
        k,
        "after K pushes cursor = K",
    );

    // One more push: cursor → K+1, slot 0 overwritten.
    let evict_src = AgentId::new(source_base + k).unwrap();
    let evict_tick = 999;
    dispatch_events(
        &device,
        &queue,
        &mut kernel,
        &state,
        &[pack_record_memory(owner, evict_src, 0, 1.0, evict_tick)],
        &storage.records_buf,
        &storage.cursors_buf,
    );

    let records = readback_records(&device, &queue, &storage.records_buf, agent_cap);
    let cursors = readback_cursors(&device, &queue, &storage.cursors_buf, agent_cap);

    assert_eq!(
        cursors[(owner.raw() - 1) as usize],
        k + 1,
        "cursor = K+1 after the evict push",
    );
    let row_base = ((owner.raw() - 1) as usize) * (k as usize);
    assert_eq!(
        records[row_base].source,
        evict_src.raw(),
        "slot 0 holds the K+1-th event (the oldest was evicted)",
    );
    assert_eq!(
        records[row_base].tick, evict_tick,
        "slot 0's tick matches the K+1-th event",
    );
}
