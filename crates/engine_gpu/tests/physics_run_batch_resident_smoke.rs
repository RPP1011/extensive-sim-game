//! Task B7 — smoke test for `PhysicsKernel::run_batch_resident`.
//!
//! This test does NOT exercise end-to-end cascade parity — correctness
//! of the resident path is the C1/C2 integration tests' concern.
//! What we verify here:
//!
//!   * The resident pipeline builds + compiles (naga unit test covers
//!     shader parse; this test covers full-GPU pipeline creation + an
//!     actual indirect dispatch).
//!   * The caller can seed slot 0 of the indirect args buffer, call
//!     `run_batch_resident(read_slot=0, write_slot=1)`, submit, and
//!     read back slot 1 with the expected workgroup-count + event-count
//!     for the next iteration.
//!   * A follow-on indirect dispatch that reads from the
//!     kernel-written slot is a no-op when no events were emitted
//!     (the cascade's natural fixed point).
//!
//! Intentionally skips any SimState / parity assertions.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::event_ring::{pack_event, EventRecord, GpuChronicleRing, GpuEventRing};
use engine_gpu::gpu_util::indirect::IndirectArgsBuffer;
use engine_gpu::physics::{
    pack_agent_slots, GpuEffectOp, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PhysicsKernel,
    MAX_ABILITIES, MAX_EFFECTS, PHYSICS_WORKGROUP_SIZE,
};
use engine::event::Event;
use engine::ids::AgentId;
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
                label: Some("physics_run_batch_resident_smoke::device"),
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
        merged.decls.extend(dsl_compiler::parse(&src).expect("parse").decls);
    }
    dsl_compiler::compile_ast(merged).expect("resolve")
}

fn build_fixture() -> SimState {
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

#[test]
fn run_batch_resident_zero_input_writes_noop_slot() {
    // With zero input events the indirect dispatch (slot 0) runs with
    // 0 workgroups — so no physics threads fire, nothing is emitted,
    // and thread 0 never executes the epilogue that writes slot 1.
    // Slot 1 should stay at its init value (0, 1, 1) — the
    // IndirectArgsBuffer::new default.
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel =
        PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let state = build_fixture();
    let agent_cap = state.agent_cap();
    let agent_slots = pack_agent_slots(&state);
    let abilities = PackedAbilityRegistry::empty();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];

    // Caller-owned buffers.
    let agents_buf = upload_storage(&device, "smoke::agents", &agent_slots);
    let ab_known = upload_storage(&device, "smoke::ab_known", &abilities.known);
    let ab_cd = upload_storage(&device, "smoke::ab_cd", &abilities.cooldown);
    let ab_ecount = upload_storage(&device, "smoke::ab_ecount", &abilities.effects_count);
    let ab_effects: &[GpuEffectOp] = &abilities.effects;
    let ab_effects_buf = upload_storage(&device, "smoke::ab_effects", ab_effects);
    let kin_buf = upload_storage(&device, "smoke::kin", &kin_lists);
    let nearest_buf = upload_storage(&device, "smoke::nearest", &nearest_hostile);

    // events_in: empty. Size it to hold at least one record for the
    // binding (storage buffers can't be zero-size with wgpu).
    let placeholder = vec![EventRecord::new(
        engine_gpu::event_ring::EventKindTag::AgentAttacked,
        0,
        [0; engine_gpu::event_ring::PAYLOAD_WORDS],
    )];
    let events_in_buf = upload_storage(&device, "smoke::events_in", &placeholder);

    // Event ring + chronicle ring (both caller-owned for the resident path).
    let event_ring = GpuEventRing::new(&device, 4096);
    event_ring.reset(&queue);
    let chronicle_ring = GpuChronicleRing::new(&device, 4096);

    // Indirect args buffer with 4 slots. Slot 0 is seeded to 0 workgroups
    // (matches IndirectArgsBuffer::new default (0, 1, 1)).
    let indirect = IndirectArgsBuffer::new(&device, 4);
    let num_events_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("smoke::num_events_buf"),
        contents: bytemuck::cast_slice(&[0u32, 0u32, 0u32, 0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    let cfg = PhysicsCfg {
        num_events: 0,
        agent_cap,
        max_abilities: MAX_ABILITIES as u32,
        max_effects: MAX_EFFECTS as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };

    // Task 2.8 — world-scalars via shared SimCfg buffer.
    let sim_cfg_snapshot = engine_gpu::sim_cfg::SimCfg::from_state(&state);
    let sim_cfg_buf = engine_gpu::sim_cfg::create_sim_cfg_buffer(&device);
    engine_gpu::sim_cfg::upload_sim_cfg(&queue, &sim_cfg_buf, &sim_cfg_snapshot);

    // Phase 3 Task 3.4 — gold ledger side buffer. Sized to `agent_cap`
    // i32s (one atomic<i32> per slot). Contents unused by this test
    // (no transfer_gold events) — just satisfies the binding.
    let gold_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke::gold_buf"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke::encoder"),
    });
    kernel
        .run_batch_resident(
            &device,
            &queue,
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
            0, // read_slot
            1, // write_slot
            cfg,
        )
        .expect("run_batch_resident");
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Slot 1 is the fresh-init (0, 1, 1) — the kernel never fired a
    // single thread so the epilogue didn't run.
    let args = indirect.read(&device, &queue);
    assert_eq!(args[0].x, 0, "slot 0 seeded as 0 workgroups (start state)");
    assert_eq!(args[1].x, 0, "slot 1 init (0, 1, 1) preserved — no dispatch ran");
    assert_eq!(args[1].y, 1);
    assert_eq!(args[1].z, 1);
}

#[test]
fn run_batch_resident_nonzero_input_publishes_next_slot() {
    // Seed slot 0 with one event worth of work and verify the kernel
    // runs + the epilogue writes slot 1. We use an `AgentAttacked`
    // event on our humans vs wolves fixture — the sync path's
    // `physics_batch_parity_agent_attacked` test confirms this fires
    // one replayable event (none with our 100-hp fixture, actually —
    // but we still expect slot 1 to be written by the `i == 0u` thread).
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel =
        PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let state = build_fixture();
    let agent_cap = state.agent_cap();
    let agent_slots = pack_agent_slots(&state);
    let abilities = PackedAbilityRegistry::empty();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];

    let agents_buf = upload_storage(&device, "smoke::agents", &agent_slots);
    let ab_known = upload_storage(&device, "smoke::ab_known", &abilities.known);
    let ab_cd = upload_storage(&device, "smoke::ab_cd", &abilities.cooldown);
    let ab_ecount = upload_storage(&device, "smoke::ab_ecount", &abilities.effects_count);
    let ab_effects_buf: &[GpuEffectOp] = &abilities.effects;
    let ab_effects_buf = upload_storage(&device, "smoke::ab_effects", ab_effects_buf);
    let kin_buf = upload_storage(&device, "smoke::kin", &kin_lists);
    let nearest_buf = upload_storage(&device, "smoke::nearest", &nearest_hostile);

    // 1 input event.
    let events_in = vec![pack_event(&Event::AgentAttacked {
        actor: AgentId::new(1).unwrap(),
        target: AgentId::new(3).unwrap(),
        tick: 0,
        damage: 5.0,
    })
    .expect("pack")];
    let events_in_buf = upload_storage(&device, "smoke::events_in", &events_in);

    let event_ring = GpuEventRing::new(&device, 4096);
    event_ring.reset(&queue);
    let chronicle_ring = GpuChronicleRing::new(&device, 4096);

    let indirect = IndirectArgsBuffer::new(&device, 4);
    // Seed slot 0 — 1 event → 1 workgroup (1 <= WG=64).
    {
        let seed = [engine_gpu::gpu_util::indirect::IndirectArgs { x: 1, y: 1, z: 1 }];
        queue.write_buffer(indirect.buffer(), 0, bytemuck::cast_slice(&seed));
    }
    let num_events_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("smoke::num_events_buf"),
        contents: bytemuck::cast_slice(&[1u32, 0u32, 0u32, 0u32]), // slot 0 = 1 event
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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

    // Task 2.8 — world-scalars via shared SimCfg buffer.
    let sim_cfg_snapshot = engine_gpu::sim_cfg::SimCfg::from_state(&state);
    let sim_cfg_buf = engine_gpu::sim_cfg::create_sim_cfg_buffer(&device);
    engine_gpu::sim_cfg::upload_sim_cfg(&queue, &sim_cfg_buf, &sim_cfg_snapshot);

    // Phase 3 Task 3.4 — gold ledger side buffer.
    let gold_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke::gold_buf"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke::encoder"),
    });
    kernel
        .run_batch_resident(
            &device,
            &queue,
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
            0, // read_slot
            1, // write_slot
            cfg,
        )
        .expect("run_batch_resident");
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // The kernel ran one workgroup. Thread 0 executed the epilogue
    // and wrote slot 1. Expected slot 1 = (wg, 1, 1) where wg =
    // ceil(emitted / PHYSICS_WORKGROUP_SIZE), clamped to
    // ceil(agent_cap / WG) = 1 for agent_cap=8.
    let args = indirect.read(&device, &queue);
    assert_eq!(args[0].x, 1, "slot 0 seeded as 1 workgroup");
    assert_eq!(args[1].y, 1, "slot 1 y=1: got args[1]={:?}", args[1]);
    assert_eq!(args[1].z, 1, "slot 1 z=1");
    let cap_wg = agent_cap.div_ceil(PHYSICS_WORKGROUP_SIZE);
    assert!(
        args[1].x <= cap_wg,
        "slot 1 wg {} exceeds cap_wg {}",
        args[1].x,
        cap_wg,
    );

    // Readback num_events slot 1 and sanity check it matches the
    // event_ring's tail after dispatch.
    let ne_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke::ne_rb"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("smoke::enc_rb"),
    });
    enc2.copy_buffer_to_buffer(&num_events_buf, 0, &ne_readback, 0, 16);
    queue.submit(Some(enc2.finish()));
    let slice = ne_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().expect("rx").expect("map");
    let data = slice.get_mapped_range();
    let ne: &[u32] = bytemuck::cast_slice(&data);
    // slot 0 was the input count (1), slot 1 was written by the
    // kernel to the number of records it appended this dispatch.
    assert_eq!(ne[0], 1);
    // slot 1 can be 0 (no replayable emissions for this fixture) or
    // more — we just assert it's consistent with args[1].x.
    let wg = args[1].x;
    let expected_wg = ne[1].div_ceil(PHYSICS_WORKGROUP_SIZE).min(cap_wg);
    assert_eq!(
        wg, expected_wg,
        "slot 1 wg {} inconsistent with emitted count {} (expected {})",
        wg, ne[1], expected_wg,
    );
    drop(data);
    ne_readback.unmap();
}
