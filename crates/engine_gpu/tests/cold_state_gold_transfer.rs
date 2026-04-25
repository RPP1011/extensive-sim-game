//! Task 3.6 — transfer_gold fires on the resident physics kernel.
//!
//! Seeds an `EffectGoldTransfer` event directly into the resident
//! physics kernel's input ring (same harness shape as
//! `physics_run_batch_resident_smoke::run_batch_resident_nonzero_input_publishes_next_slot`),
//! dispatches `run_batch_resident`, reads back `gold_buf`, and
//! asserts the atomic `state_add_agent_gold` stub correctly applied
//! the delta.
//!
//! NOTE: this test intentionally bypasses the `GpuBackend::step_batch`
//! end-to-end flow because the Phase 2 chronicle tests uncovered a
//! bug in `step_batch`'s `apply_event_ring → events_in` wiring
//! (tracked as "step_batch chronicle-emit bug"). When that bug is
//! fixed, a full `step_batch`-driven gold-transfer test can be added
//! alongside re-enabling the chronicle tests.
//!
//! What this test DOES cover:
//!   * Task 3.4: resident physics bind group 17 correctly points at
//!     `gold_buf`; WGSL `state_add_agent_gold` atomics land in the
//!     right slots.
//!   * End-to-end i32 arithmetic: `sub_gold` via negative add,
//!     `add_gold` via positive add.
//!
//! What this test does NOT cover (tracked as deferred):
//!   * The `step_batch` flow (cast → cascade → `transfer_gold` emit).
//!   * Multi-tick accumulation with resets.
//!   * `SimState.cold_inventory` snapshot merge (covered by
//!     `snapshot_merges_gold_into_state` from Task 3.5).

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use dsl_compiler::ast::Program;
use dsl_compiler::emit_physics_wgsl::EmitContext;
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_gpu::event_ring::{pack_event, GpuChronicleRing, GpuEventRing};
use engine_gpu::gpu_util::indirect::IndirectArgsBuffer;
use engine_gpu::physics::{
    pack_agent_slots, GpuEffectOp, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PhysicsKernel,
    MAX_ABILITIES, MAX_EFFECTS,
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
                label: Some("cold_state_gold_transfer::device"),
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
    // Two humans → AgentId(1) and AgentId(2). We'll transfer gold
    // from 1 → 2. Two wolves added to match the smoke test shape
    // (and to keep agent_cap = 8 with at least one alive slot per
    // team for any incidental rule logic).
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

/// Map `gold_buf` back to a Vec<i32> of length `agent_cap`. Mirrors
/// the pattern used by `num_events_buf` readback in the smoke test.
fn readback_gold_i32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gold_buf: &wgpu::Buffer,
    agent_cap: u32,
) -> Vec<i32> {
    let bytes = (agent_cap as u64 * 4).max(4);
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gold_transfer::gold_rb"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gold_transfer::gold_rb_enc"),
    });
    enc.copy_buffer_to_buffer(gold_buf, 0, &rb, 0, bytes);
    queue.submit(Some(enc.finish()));

    let slice = rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv().expect("rx").expect("map");
    let data = slice.get_mapped_range();
    let out: Vec<i32> = bytemuck::cast_slice::<u8, i32>(&data).to_vec();
    drop(data);
    rb.unmap();
    out
}

#[test]
fn transfer_gold_fires_on_resident_kernel() {
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

    let agents_buf = upload_storage(&device, "gold::agents", &agent_slots);
    let ab_known = upload_storage(&device, "gold::ab_known", &abilities.known);
    let ab_cd = upload_storage(&device, "gold::ab_cd", &abilities.cooldown);
    let ab_ecount = upload_storage(&device, "gold::ab_ecount", &abilities.effects_count);
    let ab_effects: &[GpuEffectOp] = &abilities.effects;
    let ab_effects_buf = upload_storage(&device, "gold::ab_effects", ab_effects);
    let kin_buf = upload_storage(&device, "gold::kin", &kin_lists);
    let nearest_buf = upload_storage(&device, "gold::nearest", &nearest_hostile);

    // Seed: one EffectGoldTransfer event from agent 1 → agent 2,
    // amount 25. slot_of(1) = 0, slot_of(2) = 1.
    const AMOUNT: i32 = 25;
    let events_in = vec![pack_event(&Event::EffectGoldTransfer {
        from: AgentId::new(1).unwrap(),
        to: AgentId::new(2).unwrap(),
        amount: AMOUNT,
        tick: 0,
    })
    .expect("pack")];
    let events_in_buf = upload_storage(&device, "gold::events_in", &events_in);

    let event_ring = GpuEventRing::new(&device, 4096);
    event_ring.reset(&queue);
    let chronicle_ring = GpuChronicleRing::new(&device, 4096);

    // Seed slot 0 with 1 workgroup's worth of work.
    let indirect = IndirectArgsBuffer::new(&device, 4);
    {
        let seed = [engine_gpu::gpu_util::indirect::IndirectArgs { x: 1, y: 1, z: 1 }];
        queue.write_buffer(indirect.buffer(), 0, bytemuck::cast_slice(&seed));
    }
    let num_events_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gold::num_events_buf"),
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

    let sim_cfg_snapshot = engine_gpu::sim_cfg::SimCfg::from_state(&state);
    let sim_cfg_buf = engine_gpu::sim_cfg::create_sim_cfg_buffer(&device);
    engine_gpu::sim_cfg::upload_sim_cfg(&queue, &sim_cfg_buf, &sim_cfg_snapshot);

    // Phase 3 Task 3.4 — gold ledger side buffer. Seed the initial
    // per-slot gold values: slot 0 (agent 1) = 100, slot 1 (agent 2) = 0,
    // rest = 0. Must include COPY_SRC so we can read it back after
    // dispatch.
    let gold_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gold::gold_buf"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut initial_gold = vec![0i32; agent_cap as usize];
    initial_gold[0] = 100;
    initial_gold[1] = 0;
    queue.write_buffer(&gold_buf, 0, bytemuck::cast_slice(&initial_gold));

    // Task #79 SP-4 — standing view storage. Unused by this test
    // (only transfer_gold fires) but the resident BGL requires it.
    let standing_records_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gold::standing_records"),
        size: (agent_cap as u64 * 8 * 16).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let standing_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gold::standing_counts"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Subsystem 2 Phase 4 PR-4 — memory view storage. Unused by this
    // test but the resident BGL requires it.
    let memory_records_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gold::memory_records"),
        size: (agent_cap as u64 * 64 * 24).max(24),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let memory_cursors_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gold::memory_cursors"),
        size: (agent_cap as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Alive bitmap: all agents alive.
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

    // Baseline readback (sanity: initial values reached the device
    // before dispatch).
    let pre = readback_gold_i32(&device, &queue, &gold_buf, agent_cap);
    assert_eq!(pre[0], 100, "pre-dispatch: slot 0 seeded to 100");
    assert_eq!(pre[1], 0, "pre-dispatch: slot 1 seeded to 0");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gold::encoder"),
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
            &standing_records_buf,
            &standing_counts_buf,
            &memory_records_buf,
            &memory_cursors_buf,
            &alive_bitmap_buf,
            0, // read_slot
            1, // write_slot
            cfg,
        )
        .expect("run_batch_resident");
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::PollType::Wait);

    // The transfer_gold rule fires on the EffectGoldTransfer input:
    //   agents.sub_gold(from=1, 25)  → atomicAdd(gold_buf[0], -25)
    //   agents.add_gold(to=2,   25)  → atomicAdd(gold_buf[1], +25)
    let post = readback_gold_i32(&device, &queue, &gold_buf, agent_cap);
    assert_eq!(
        post[0], 75,
        "slot 0 (from=agent 1): 100 - 25 = 75, got {}",
        post[0],
    );
    assert_eq!(
        post[1], 25,
        "slot 1 (to=agent 2):   0 + 25 = 25, got {}",
        post[1],
    );
    for (i, g) in post.iter().enumerate().skip(2) {
        assert_eq!(*g, 0, "slot {i} should be unchanged (0), got {g}");
    }
}
