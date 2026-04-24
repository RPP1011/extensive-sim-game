//! Contract test: `PhysicsKernel::run_batch_resident` emits a
//! chronicle record when dispatched on a single AgentAttacked event.
//!
//! Mirrors `physics_run_batch_resident_smoke::run_batch_resident_nonzero_input_publishes_next_slot`
//! but asserts on the caller-supplied `chronicle_ring`'s tail rather
//! than on `indirect_args` / `num_events_buf`. Passing this test but
//! failing `chronicle_batch_path` would confirm that the kernel itself
//! is fine and the bug is in `step_batch`'s wiring. Kept post-fix as a
//! bisect anchor for future regressions.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::event_ring::{pack_event, GpuChronicleRing, GpuEventRing};
use engine_gpu::gpu_util::indirect::IndirectArgsBuffer;
use engine_gpu::physics::{
    pack_agent_slots, GpuEffectOp, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PhysicsKernel,
    MAX_ABILITIES, MAX_EFFECTS,
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
                label: Some("chronicle_isolated_smoke::device"),
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
fn run_batch_resident_emits_chronicle_on_attacked() {
    let (device, queue) = gpu_device_queue();
    let comp = load_physics();
    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("kernel init");

    let mut state = SimState::new(8, 0xDEAD_BEEF);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(5.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("wolf");

    let agent_cap = state.agent_cap();
    let agent_slots = pack_agent_slots(&state);
    let abilities = PackedAbilityRegistry::empty();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];

    let agents_buf = upload_storage(&device, "probe::agents", &agent_slots);
    let ab_known = upload_storage(&device, "probe::ab_known", &abilities.known);
    let ab_cd = upload_storage(&device, "probe::ab_cd", &abilities.cooldown);
    let ab_ecount = upload_storage(&device, "probe::ab_ecount", &abilities.effects_count);
    let ab_effects_buf: &[GpuEffectOp] = &abilities.effects;
    let ab_effects_buf = upload_storage(&device, "probe::ab_effects", ab_effects_buf);
    let kin_buf = upload_storage(&device, "probe::kin", &kin_lists);
    let nearest_buf = upload_storage(&device, "probe::nearest", &nearest_hostile);

    let events_in = vec![pack_event(&Event::AgentAttacked {
        actor: AgentId::new(1).unwrap(),
        target: AgentId::new(2).unwrap(),
        tick: 1,
        damage: 5.0,
    })
    .expect("pack")];
    let events_in_buf = upload_storage(&device, "probe::events_in", &events_in);

    let event_ring = GpuEventRing::new(&device, 4096);
    event_ring.reset(&queue);
    let chronicle_ring = GpuChronicleRing::new(&device, 4096);

    let indirect = IndirectArgsBuffer::new(&device, 4);
    {
        let seed = [engine_gpu::gpu_util::indirect::IndirectArgs { x: 1, y: 1, z: 1 }];
        queue.write_buffer(indirect.buffer(), 0, bytemuck::cast_slice(&seed));
    }
    let num_events_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe::num_events_buf"),
        contents: bytemuck::cast_slice(&[1u32, 0u32, 0u32, 0u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    let cfg = PhysicsCfg {
        num_events: 1,
        agent_cap,
        max_abilities: MAX_ABILITIES as u32,
        max_effects: MAX_EFFECTS as u32,
        _pad0: 0, _pad1: 0, _pad2: 0, _pad3: 0,
    };

    let sim_cfg_snapshot = engine_gpu::sim_cfg::SimCfg::from_state(&state);
    let sim_cfg_buf = engine_gpu::sim_cfg::create_sim_cfg_buffer(&device);
    engine_gpu::sim_cfg::upload_sim_cfg(&queue, &sim_cfg_buf, &sim_cfg_snapshot);

    let gold_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe::gold_buf"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Task #79 SP-4 — standing view storage (records + counts).
    // Test-local buffers sized to agent_cap × K=8 × 12 bytes + counts.
    let standing_records_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe::standing_records"),
        size: (agent_cap as u64 * 8 * 12).max(12),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let standing_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe::standing_counts"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Subsystem 2 Phase 4 PR-4 — memory view storage (records + cursors).
    let memory_records_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe::memory_records"),
        size: (agent_cap as u64 * 64 * 24).max(24),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let memory_cursors_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe::memory_cursors"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Alive bitmap: all alive.
    let alive_bitmap_buf =
        engine_gpu::alive_bitmap::create_alive_bitmap_buffer(&device, agent_cap);
    {
        let words = engine_gpu::alive_bitmap::alive_bitmap_words(agent_cap) as usize;
        let mut packed = vec![0u32; words.max(1)];
        for slot in 0..agent_cap as usize {
            packed[slot >> 5] |= 1u32 << (slot & 31);
        }
        queue.write_buffer(&alive_bitmap_buf, 0, bytemuck::cast_slice(&packed));
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe::encoder"),
    });
    kernel.run_batch_resident(
        &device, &queue, &mut encoder,
        &agents_buf, &ab_known, &ab_cd, &ab_ecount, &ab_effects_buf,
        &kin_buf, &nearest_buf,
        &events_in_buf, &event_ring, &chronicle_ring,
        &indirect, &num_events_buf, &sim_cfg_buf, &gold_buf,
        &standing_records_buf, &standing_counts_buf,
        &memory_records_buf, &memory_cursors_buf,
        &alive_bitmap_buf,
        None, // per_rule_counter_buf (research mode only)
        0, 1, cfg,
    ).expect("run_batch_resident");
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // Read back chronicle_ring tail.
    let tail_v: Vec<u32> = engine_gpu::gpu_util::readback::readback_typed::<u32>(
        &device, &queue, chronicle_ring.tail_buffer(), 4,
    )
    .expect("readback chron tail");
    let chron_tail = tail_v.first().copied().unwrap_or(0);
    eprintln!("isolated chronicle_tail after 1 AgentAttacked event = {chron_tail}");
    assert!(
        chron_tail > 0,
        "ISOLATED smoke: expected chronicle_tail > 0 but got {chron_tail}; \
         physics_chronicle_attack should have emitted ChronicleEntry for the AgentAttacked event"
    );
}
