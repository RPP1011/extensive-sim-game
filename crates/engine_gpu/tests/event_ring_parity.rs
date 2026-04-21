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

use engine::event::{Event, EventRing};
use engine_gpu::event_ring::{
    pack_event, wgsl_prefix, DrainOutcome, EventKindTag, EventRecord, GpuEventRing,
    EVENT_RING_WGSL, PAYLOAD_WORDS,
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
