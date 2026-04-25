//! Phase 6b — GPU event ring parity harness.
//!
//! Gated on `--features gpu`. Run with
//! `cargo test -p engine_gpu --features gpu --test event_ring_parity`.
//!
//! ## What's tested
//!
//! 1. **Basic emit + drain.** A tiny kernel emits 100 records of four
//!    different event kinds; the host drains and asserts every event
//!    is present in deterministic order. Also exercises the three
//!    core per-kind WGSL helpers (`gpu_emit_agent_attacked`,
//!    `gpu_emit_agent_died`, `gpu_emit_effect_damage`,
//!    `gpu_emit_engagement_committed`).
//! 2. **Atomic contention.** 64 workgroups × 64 threads = 4096
//!    concurrent emitters all `atomicAdd`ing into the tail. The drain
//!    must see all 4096 events with no duplicates and no drops. This
//!    is the core determinism-under-contention contract.
//! 3. **Overflow.** Emit more records than capacity. Drain must
//!    truncate cleanly, set the `overflowed` flag, and keep the CPU
//!    ring internally consistent (no panic, events count <= capacity).

#![cfg(feature = "gpu")]

use engine::event::EventRing;
use engine_data::events::Event;
use engine_gpu::event_ring::{
    chronicle_wgsl_prefix, pack_event, wgsl_prefix, ChronicleRecord, DrainOutcome, EventKindTag,
    EventRecord, GpuChronicleRing, GpuEventRing, CHRONICLE_RING_WGSL, EVENT_RING_WGSL,
    PAYLOAD_WORDS,
};
use std::collections::HashMap;

fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue, String) {
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
        let label = format!("{:?}", adapter.get_info().backend);
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("event_ring_parity::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter_limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");
        (device, queue, label)
    })
}

// ---------------------------------------------------------------------------
// Test kernel source — uses `EVENT_RING_WGSL` + emits one record per
// `global_invocation_id.x` in `[0, NUM)`. NUM is a uniform so tests
// can reuse the same pipeline across sizes.
// ---------------------------------------------------------------------------

fn build_test_shader(capacity: u32) -> String {
    let prefix = wgsl_prefix(capacity);
    let bindings = r#"
struct Cfg {
    num_emits: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> event_ring: array<EventRecord>;
@group(0) @binding(1) var<storage, read_write> event_ring_tail: atomic<u32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(64)
fn cs_emit_mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.num_emits) { return; }
    // Mix event kinds so the drain has something to sort on.
    let kind_sel = i % 4u;
    let actor: u32 = i + 1u;              // NonZeroU32 — always >= 1
    let target_id: u32 = (i % 8u) + 1u;   // mostly collides w/ small ids
    let tick: u32 = cfg.tick + (i % 3u);  // three distinct ticks
    if (kind_sel == 0u) {
        // AgentAttacked
        let dmg: f32 = f32(i) * 0.5;
        _ = gpu_emit_agent_attacked(actor, target_id, dmg, tick);
    } else if (kind_sel == 1u) {
        // AgentDied
        _ = gpu_emit_agent_died(actor, tick);
    } else if (kind_sel == 2u) {
        // EffectDamageApplied
        let amt: f32 = f32(i) * 0.25;
        _ = gpu_emit_effect_damage(actor, target_id, amt, tick);
    } else {
        // EngagementCommitted
        _ = gpu_emit_engagement_committed(actor, target_id, tick);
    }
}

// Contention variant — every thread fires on the same tick so the
// drain's (tick, kind, payload[0]) sort key has the payload[0] arm
// actually doing work. Uses a single kind to stress the atomic
// counter rather than spreading writes across decoders.
@compute @workgroup_size(64)
fn cs_emit_contention(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.num_emits) { return; }
    let actor: u32 = i + 1u;
    _ = gpu_emit_agent_attacked(actor, 1u, f32(i), cfg.tick);
}
"#;
    format!("{prefix}{EVENT_RING_WGSL}{bindings}")
}

#[allow(dead_code)] // `capacity` stored for diagnostics; not read in current tests.
struct TestHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend_label: String,
    ring: GpuEventRing,
    pipeline_mix: wgpu::ComputePipeline,
    pipeline_contention: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    cfg_buffer: wgpu::Buffer,
    capacity: u32,
}

impl TestHarness {
    fn new(capacity: u32) -> Self {
        let (device, queue, backend_label) = gpu_device_queue();
        let ring = GpuEventRing::new(&device, capacity);

        let src = build_test_shader(capacity);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("event_ring_parity::shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("event_ring_parity shader compile: {err}");
        }

        let bgl_entries = {
            let mut v = ring.bind_group_layout_entries(0);
            v.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            v
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("event_ring_parity::bgl"),
            entries: &bgl_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("event_ring_parity::pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline_mix = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("event_ring_parity::cs_emit_mix"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_emit_mix"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_contention = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("event_ring_parity::cs_emit_contention"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_emit_contention"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cfg_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("event_ring_parity::cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = {
            let mut entries = ring.bind_group_entries(0);
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: cfg_buffer.as_entire_binding(),
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("event_ring_parity::bg"),
                layout: &bgl,
                entries: &entries,
            })
        };

        Self {
            device,
            queue,
            backend_label,
            ring,
            pipeline_mix,
            pipeline_contention,
            bind_group,
            cfg_buffer,
            capacity,
        }
    }

    fn write_cfg(&self, num_emits: u32, tick: u32) {
        let words: [u32; 4] = [num_emits, tick, 0, 0];
        self.queue
            .write_buffer(&self.cfg_buffer, 0, bytemuck::cast_slice(&words));
    }

    fn run_mix(&self, num_emits: u32, tick: u32) {
        self.ring.reset(&self.queue);
        self.write_cfg(num_emits, tick);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("event_ring_parity::enc_mix"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("event_ring_parity::cpass_mix"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_pipeline(&self.pipeline_mix);
            let groups = num_emits.div_ceil(64).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn run_contention(&self, num_emits: u32, tick: u32) {
        self.ring.reset(&self.queue);
        self.write_cfg(num_emits, tick);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("event_ring_parity::enc_contention"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("event_ring_parity::cpass_contention"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_pipeline(&self.pipeline_contention);
            let groups = num_emits.div_ceil(64).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

// ---------------------------------------------------------------------------
// Test 1 — basic mix of 100 events across 4 kinds
// ---------------------------------------------------------------------------

#[test]
fn event_ring_basic_emit_and_drain() {
    const NUM: u32 = 100;
    let harness = TestHarness::new(1024);
    eprintln!("[basic] backend={}", harness.backend_label);

    harness.run_mix(NUM, 10);

    let mut cpu_ring = EventRing::with_cap(NUM as usize * 4);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut cpu_ring)
        .expect("drain");

    assert_eq!(outcome.tail_raw, NUM, "tail should equal number of emits");
    assert_eq!(outcome.drained, NUM, "every emit should deserialize");
    assert!(!outcome.overflowed, "no overflow at NUM={NUM} << capacity");
    assert_eq!(cpu_ring.len(), NUM as usize);

    // Count each kind in the ring — expected split is i%4.
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for e in cpu_ring.iter() {
        let k = match e {
            Event::AgentAttacked { .. } => EventKindTag::AgentAttacked.raw(),
            Event::AgentDied { .. } => EventKindTag::AgentDied.raw(),
            Event::EffectDamageApplied { .. } => EventKindTag::EffectDamageApplied.raw(),
            Event::EngagementCommitted { .. } => EventKindTag::EngagementCommitted.raw(),
            _ => panic!("unexpected event kind in drained ring: {e:?}"),
        };
        *counts.entry(k).or_default() += 1;
    }
    // With NUM=100 and `kind_sel = i % 4`, each bucket gets 25.
    assert_eq!(counts.get(&EventKindTag::AgentAttacked.raw()).copied(), Some(25));
    assert_eq!(counts.get(&EventKindTag::AgentDied.raw()).copied(), Some(25));
    assert_eq!(counts.get(&EventKindTag::EffectDamageApplied.raw()).copied(), Some(25));
    assert_eq!(
        counts.get(&EventKindTag::EngagementCommitted.raw()).copied(),
        Some(25),
    );

    // Determinism: ring is ordered by (tick, kind, payload[0]=actor).
    // Assert the sort invariant actually holds.
    let mut prev: Option<(u32, u32, u32)> = None;
    for e in cpu_ring.iter() {
        let r = pack_event(e).expect("pack roundtrip");
        let key = (r.tick, r.kind, r.payload[0]);
        if let Some(p) = prev {
            assert!(p <= key, "ring not sorted: prev={p:?} next={key:?}");
        }
        prev = Some(key);
    }
}

// ---------------------------------------------------------------------------
// Test 2 — contention
// ---------------------------------------------------------------------------

#[test]
fn event_ring_contention_under_4096_concurrent_emits() {
    const NUM: u32 = 64 * 64; // 4096
    let harness = TestHarness::new(8192);
    eprintln!("[contention] backend={} num={NUM}", harness.backend_label);

    harness.run_contention(NUM, 42);

    let mut cpu_ring = EventRing::with_cap(NUM as usize * 2);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut cpu_ring)
        .expect("drain");

    assert_eq!(outcome.tail_raw, NUM, "all {NUM} threads incremented tail");
    assert_eq!(outcome.drained, NUM, "no deserialization drops");
    assert!(!outcome.overflowed);

    // Each emit uses actor = global_id + 1, so actor ids should be
    // exactly {1, 2, ..., NUM} with no duplicates and no gaps.
    let mut actors: Vec<u32> = cpu_ring
        .iter()
        .map(|e| match e {
            Event::AgentAttacked { actor, .. } => actor.raw(),
            _ => panic!("wrong kind in contention drain: {e:?}"),
        })
        .collect();
    actors.sort_unstable();
    for (i, a) in actors.iter().enumerate() {
        assert_eq!(*a, (i as u32) + 1, "hole or duplicate at index {i}: got {a}");
    }
}

// ---------------------------------------------------------------------------
// Test 3 — overflow
// ---------------------------------------------------------------------------

#[test]
fn event_ring_overflow_is_detected_and_truncated() {
    const CAP: u32 = 128;
    const NUM: u32 = 512;
    let harness = TestHarness::new(CAP);
    eprintln!(
        "[overflow] backend={} cap={CAP} emits={NUM}",
        harness.backend_label
    );

    harness.run_contention(NUM, 7);

    let mut cpu_ring = EventRing::with_cap(NUM as usize);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut cpu_ring)
        .expect("drain");

    // Tail reflects every attempted emit — atomics fire even for
    // slots past the cap; only the guarded store is skipped.
    assert_eq!(outcome.tail_raw, NUM);
    // We only drain what fits in the buffer.
    assert_eq!(outcome.drained, CAP, "drain truncates to capacity");
    assert!(outcome.overflowed, "overflow flag should fire");
    assert_eq!(cpu_ring.len(), CAP as usize);

    // Every drained event should still deserialize cleanly — the
    // overflow is in the slots we *didn't* read, not the ones we did.
    for e in cpu_ring.iter() {
        match e {
            Event::AgentAttacked { .. } => {}
            other => panic!("unexpected kind in overflow drain: {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4 — cpu-seeded round-trip (parity harness for the rehydration
// path without needing the kernel to be correct).
// ---------------------------------------------------------------------------

#[test]
fn event_ring_cpu_seeded_roundtrip_sorts_deterministically() {
    let harness = TestHarness::new(32);
    eprintln!("[cpu_seed] backend={}", harness.backend_label);

    // Seed the ring with three records whose raw emit order is
    // *reversed* w.r.t. the desired sort key. After drain the CPU
    // ring must carry them in (tick, kind, payload[0]) ascending
    // order.
    let r_newer = pack_event(&Event::AgentAttacked {
        actor: engine::ids::AgentId::new(3).unwrap(),
        target: engine::ids::AgentId::new(1).unwrap(),
        damage: 1.0,
        tick: 10,
    })
    .unwrap();
    let r_older = pack_event(&Event::AgentAttacked {
        actor: engine::ids::AgentId::new(1).unwrap(),
        target: engine::ids::AgentId::new(2).unwrap(),
        damage: 2.0,
        tick: 5,
    })
    .unwrap();
    let r_mid = pack_event(&Event::AgentAttacked {
        actor: engine::ids::AgentId::new(2).unwrap(),
        target: engine::ids::AgentId::new(3).unwrap(),
        damage: 3.0,
        tick: 5,
    })
    .unwrap();

    // Write in the "wrong" order: newer first, older last.
    harness
        .ring
        .seed_for_test(&harness.queue, &[r_newer, r_older, r_mid]);

    let mut cpu_ring = EventRing::with_cap(8);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut cpu_ring)
        .expect("drain");
    assert_eq!(outcome.drained, 3);

    let events: Vec<Event> = cpu_ring.iter().copied().collect();
    // Expected order: tick 5 before tick 10, and within tick 5 the
    // actor-id-1 record before the actor-id-2 record.
    match &events[..] {
        [Event::AgentAttacked { actor: a1, tick: 5, .. },
         Event::AgentAttacked { actor: a2, tick: 5, .. },
         Event::AgentAttacked { actor: a3, tick: 10, .. }] => {
            assert_eq!(a1.raw(), 1);
            assert_eq!(a2.raw(), 2);
            assert_eq!(a3.raw(), 3);
        }
        _ => panic!("drain did not sort deterministically: {events:?}"),
    }
}

// ---------------------------------------------------------------------------
// Test 5 — constants + buffer size documentation.
// ---------------------------------------------------------------------------

#[test]
fn event_ring_buffer_size_matches_documented() {
    // At the default 65_536 capacity the GPU buffer is 2.5 MiB.
    let harness = TestHarness::new(65_536);
    let expected = 65_536u64 * (2 + PAYLOAD_WORDS as u64) * 4;
    assert_eq!(harness.ring.buffer_bytes(), expected);
    assert_eq!(harness.ring.capacity(), 65_536);
    // Record size = 40 B with PAYLOAD_WORDS = 8.
    assert_eq!(std::mem::size_of::<EventRecord>(), 40);
}

// ---------------------------------------------------------------------------
// Test 6 — empty drain returns cleanly.
// ---------------------------------------------------------------------------

#[test]
fn event_ring_drain_of_empty_ring() {
    let harness = TestHarness::new(32);
    harness.ring.reset(&harness.queue);

    let mut cpu_ring = EventRing::with_cap(8);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut cpu_ring)
        .expect("drain");

    assert_eq!(outcome.tail_raw, 0);
    assert_eq!(outcome.drained, 0);
    assert!(!outcome.overflowed);
    assert_eq!(cpu_ring.len(), 0);

    // Silence the unused-variant warning on DrainOutcome's Debug print.
    let _debug = format!("{outcome:?}");
    let _: DrainOutcome = outcome;
}

// ---------------------------------------------------------------------------
// Task 203 — chronicle ring tests
// ---------------------------------------------------------------------------
//
// Validates the dedicated chronicle ring:
//   * Basic emit from a test kernel → drain → `Event::ChronicleEntry`
//     records land in the CPU ring with the right template ids.
//   * CPU-seeded round trip — the drain reads back records even when
//     nobody dispatched a kernel (validates the readback path in
//     isolation from the kernel-emit path).
//   * Overflow wraps — the ring reports wrapping when `tail_raw > cap`
//     and still returns the resident records without panicking.

fn build_chronicle_test_shader(capacity: u32) -> String {
    let prefix = chronicle_wgsl_prefix(capacity);
    let bindings = r#"
struct Cfg {
    num_emits: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> chronicle_ring: array<ChronicleRecord>;
@group(0) @binding(1) var<storage, read_write> chronicle_ring_tail: atomic<u32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@compute @workgroup_size(64)
fn cs_chronicle_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.num_emits) { return; }
    // Emit one chronicle per thread. Cycle template ids 1..=8 so a
    // drain sees all eight narrative kinds.
    let tmpl: u32 = (i % 8u) + 1u;
    let agent: u32 = i + 1u;              // NonZeroU32
    let target_id: u32 = (i % 4u) + 1u;   // collides with small ids
    _ = gpu_emit_chronicle_event(tmpl, agent, target_id, cfg.tick);
}
"#;
    format!("{prefix}{CHRONICLE_RING_WGSL}{bindings}")
}

struct ChronicleHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    #[allow(dead_code)]
    backend_label: String,
    ring: GpuChronicleRing,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    cfg_buffer: wgpu::Buffer,
}

impl ChronicleHarness {
    fn new(capacity: u32) -> Self {
        let (device, queue, backend_label) = gpu_device_queue();
        let ring = GpuChronicleRing::new(&device, capacity);

        let src = build_chronicle_test_shader(capacity);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chronicle_ring_parity::shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("chronicle_ring_parity shader compile: {err}");
        }

        let bgl_entries = {
            let mut v = ring.bind_group_layout_entries(0);
            v.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            v
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chronicle_ring_parity::bgl"),
            entries: &bgl_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chronicle_ring_parity::pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chronicle_ring_parity::pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_chronicle_emit"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cfg_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chronicle_ring_parity::cfg"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = {
            let mut entries = ring.bind_group_entries(0);
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: cfg_buffer.as_entire_binding(),
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chronicle_ring_parity::bg"),
                layout: &bgl,
                entries: &entries,
            })
        };

        Self {
            device,
            queue,
            backend_label,
            ring,
            pipeline,
            bind_group,
            cfg_buffer,
        }
    }

    fn run(&self, num_emits: u32, tick: u32) {
        self.ring.reset(&self.queue);
        let words: [u32; 4] = [num_emits, tick, 0, 0];
        self.queue
            .write_buffer(&self.cfg_buffer, 0, bytemuck::cast_slice(&words));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chronicle_ring_parity::enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chronicle_ring_parity::cpass"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_pipeline(&self.pipeline);
            let groups = num_emits.div_ceil(64).max(1);
            cpass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

#[test]
fn chronicle_ring_basic_emit_and_drain() {
    const NUM: u32 = 800;
    let harness = ChronicleHarness::new(1024);

    harness.run(NUM, 42);

    let mut events = EventRing::with_cap(NUM as usize * 2);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut events)
        .expect("drain");

    assert_eq!(outcome.tail_raw, NUM);
    assert_eq!(outcome.drained, NUM);
    assert!(!outcome.wrapped, "well under capacity, should not wrap");
    assert_eq!(events.len(), NUM as usize);

    // Every drained event is a ChronicleEntry — the chronicle ring
    // produces nothing else.
    let mut tmpl_counts: HashMap<u32, u32> = HashMap::new();
    for ev in events.iter() {
        match ev {
            Event::ChronicleEntry { template_id, tick, .. } => {
                assert_eq!(*tick, 42);
                *tmpl_counts.entry(*template_id).or_default() += 1;
            }
            other => panic!("chronicle drain produced non-chronicle event: {other:?}"),
        }
    }
    // 800 / 8 templates = 100 each.
    for t in 1u32..=8 {
        assert_eq!(
            tmpl_counts.get(&t).copied().unwrap_or(0),
            100,
            "template {t} miscount in chronicle drain",
        );
    }
}

#[test]
fn chronicle_ring_wrap_on_overflow() {
    const CAP: u32 = 128;
    const NUM: u32 = 512;
    let harness = ChronicleHarness::new(CAP);

    harness.run(NUM, 7);

    let mut events = EventRing::with_cap(NUM as usize * 2);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut events)
        .expect("drain");

    // Tail reflects every attempted emit — atomics fire for every
    // slot including post-capacity ones (just overwrite a wrapped
    // slot instead of dropping).
    assert_eq!(outcome.tail_raw, NUM);
    // Drain truncates to capacity — the resident window is `[0, CAP)`.
    assert_eq!(outcome.drained, CAP, "should drain exactly capacity records");
    assert!(outcome.wrapped, "wrap flag should fire when tail > cap");
    assert_eq!(events.len(), CAP as usize);

    // Every drained event must still deserialise cleanly — the
    // wrap-on-overflow path overwrites older slots but never
    // produces malformed records.
    for ev in events.iter() {
        match ev {
            Event::ChronicleEntry { template_id, tick, .. } => {
                assert!(*template_id >= 1 && *template_id <= 8);
                assert_eq!(*tick, 7);
            }
            other => panic!("unexpected event kind in wrap drain: {other:?}"),
        }
    }
}

#[test]
fn chronicle_ring_cpu_seeded_roundtrip() {
    let harness = ChronicleHarness::new(16);

    // Seed three records directly. The drain path should handle CPU-
    // seeded content identically to kernel-emitted content.
    let seeded = vec![
        ChronicleRecord {
            template_id: 1,
            agent: 10,
            target: 10,
            tick: 5,
        },
        ChronicleRecord {
            template_id: 2,
            agent: 7,
            target: 3,
            tick: 5,
        },
        ChronicleRecord {
            template_id: 4,
            agent: 11,
            target: 12,
            tick: 10,
        },
    ];
    harness.ring.seed_for_test(&harness.queue, &seeded);

    let mut events = EventRing::with_cap(8);
    let outcome = harness
        .ring
        .drain(&harness.device, &harness.queue, &mut events)
        .expect("drain");
    assert_eq!(outcome.drained, 3);
    assert!(!outcome.wrapped);

    let drained: Vec<_> = events.iter().cloned().collect();
    assert_eq!(drained.len(), 3);
    match &drained[..] {
        [
            Event::ChronicleEntry { template_id: 1, agent: a0, target: t0, tick: 5 },
            Event::ChronicleEntry { template_id: 2, agent: a1, target: t1, tick: 5 },
            Event::ChronicleEntry { template_id: 4, agent: a2, target: t2, tick: 10 },
        ] => {
            assert_eq!(a0.raw(), 10);
            assert_eq!(t0.raw(), 10);
            assert_eq!(a1.raw(), 7);
            assert_eq!(t1.raw(), 3);
            assert_eq!(a2.raw(), 11);
            assert_eq!(t2.raw(), 12);
        }
        _ => panic!("unexpected drain order: {drained:?}"),
    }
}

// ---------------------------------------------------------------------------
// Task 203 — chronicle routing parity (GPU physics path)
// ---------------------------------------------------------------------------
//
// End-to-end: build a physics kernel against assets/sim, seed an
// AgentAttacked event, run one cascade iteration, and assert:
//   1. The main event ring carries ZERO chronicle records (no kind=24).
//   2. The chronicle ring carries the expected `ChronicleEntry` records.
//
// This pins the routing contract — a regression that re-routes chronicle
// to the main ring would fail (1); a regression that drops chronicle
// entirely would fail (2).

#[test]
fn chronicle_events_route_to_chronicle_ring_not_main_ring() {
    use std::path::PathBuf;

    use dsl_compiler::ast::Program;
    use dsl_compiler::emit_physics_wgsl::EmitContext;
    use engine_data::entities::CreatureType;
    use engine::ids::AgentId;
    use engine::state::{AgentSpawn, SimState};
    use engine_gpu::event_ring::EventKindTag;
    use engine_gpu::physics::{
        pack_agent_slots, GpuKinList, PackedAbilityRegistry, PhysicsCfg, PhysicsKernel,
    };
    use glam::Vec3;

    let (device, queue, backend_label) = gpu_device_queue();
    eprintln!("[chronicle_routing] backend={backend_label}");

    // Load assets/sim to get the full physics rule set including the
    // 8 chronicle rules. Same loader as physics_parity.
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root.push("assets/sim");
    let mut merged = Program { decls: Vec::new() };
    for f in &["config.sim", "enums.sim", "events.sim", "physics.sim"] {
        let src = std::fs::read_to_string(root.join(f))
            .unwrap_or_else(|e| panic!("read {}: {e}", f));
        merged.decls.extend(dsl_compiler::parse(&src).unwrap().decls);
    }
    let comp = dsl_compiler::compile_ast(merged).expect("resolve");

    // Two-agent fixture with enough HP to keep chronicle_wound gated off
    // (hp_pct >= 0.5 after a 5-damage attack on 100hp targets). This
    // keeps the set of firing chronicles predictable for the assertion.
    let mut state = SimState::new(4, 0xD00D_BEEF);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("h1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("w1 spawn");

    let ctx = EmitContext {
        events: &comp.events,
        event_tags: &comp.event_tags,
    };
    let mut kernel = PhysicsKernel::new(&device, &comp.physics, &ctx, 4096).expect("physics");

    let agent_cap = state.agent_cap();
    let kin_lists = vec![GpuKinList::default(); agent_cap as usize];
    let nearest_hostile = vec![u32::MAX; agent_cap as usize];
    let cfg = PhysicsCfg {
        num_events: 0,
        agent_cap,
        max_abilities: engine_gpu::physics::MAX_ABILITIES as u32,
        max_effects: engine_gpu::physics::MAX_EFFECTS as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    // Task 2.8 — world-scalars flow via the shared SimCfg buffer.
    let sim_cfg = engine_gpu::sim_cfg::SimCfg::from_state(&state);
    let abilities = PackedAbilityRegistry::empty();
    let seeded = vec![Event::AgentAttacked {
        actor: AgentId::new(1).unwrap(),
        target: AgentId::new(2).unwrap(),
        damage: 5.0,
        tick: 0,
    }];
    let events_in: Vec<_> = seeded.iter().filter_map(pack_event).collect();

    let out = kernel
        .run_batch(
            &device,
            &queue,
            &pack_agent_slots(&state),
            &abilities,
            &kin_lists,
            &nearest_hostile,
            &events_in,
            cfg,
            &sim_cfg,
        )
        .expect("run_batch");
    assert!(!out.drain.overflowed, "main event ring should not overflow");

    // --- Invariant 1: no chronicle records in the main event ring ---
    let chronicle_kind_raw = 24u32; // EVENT_KIND_CHRONICLE_ENTRY
    for rec in &out.events_out {
        assert_ne!(
            rec.kind, chronicle_kind_raw,
            "chronicle record leaked into main event ring: {rec:?}"
        );
        // Sanity — every main-ring record must deserialise to a known
        // replayable event. `EventKindTag::from_u32` returns None for
        // the reserved chronicle slot; if the emitter regressed and
        // started writing kind=24 back into the main ring, unpack
        // would return None here.
        assert!(
            EventKindTag::from_u32(rec.kind).is_some(),
            "main ring has unknown kind {}: {:?}",
            rec.kind,
            rec
        );
    }

    // --- Invariant 2: chronicle ring has the expected records ---
    let mut chronicle_events = EventRing::with_cap(64);
    let chronicle_outcome = kernel
        .chronicle_ring()
        .drain(&device, &queue, &mut chronicle_events)
        .expect("chronicle drain");
    assert!(!chronicle_outcome.wrapped, "chronicle ring should not wrap");

    // `AgentAttacked { actor: 1, target: 2 }` fires only
    // `chronicle_attack` on the GPU side (chronicle_wound is gated on
    // hp_pct < 0.5; both agents are at full HP so the inner if is
    // short-circuited). Expected: one chronicle record with
    // template_id=2.
    let chronicles: Vec<_> = chronicle_events.iter().cloned().collect();
    assert_eq!(
        chronicles.len(),
        1,
        "expected exactly one chronicle entry for a single AgentAttacked; got {chronicles:?}"
    );
    match &chronicles[0] {
        Event::ChronicleEntry { template_id: 2, agent, target, tick: 0 } => {
            assert_eq!(agent.raw(), 1);
            assert_eq!(target.raw(), 2);
        }
        other => panic!("unexpected chronicle payload: {other:?}"),
    }
}
