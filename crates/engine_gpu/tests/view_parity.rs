//! Phase 4 parity harness — per-view CPU/GPU fold parity.
//!
//! For each materialized view shape (slot_map, pair_map, pair_map
//! @decay), generate a deterministic event stream, fold it on both
//! sides, and assert byte-equal storage + byte-equal decayed reads.
//!
//! This test doesn't drive a full `GpuBackend::step` — the view
//! storage isn't yet wired into the tick loop (that's the follow-up
//! integration task). Instead it uses the GPU view_storage API
//! directly: `ViewStorage::new` for setup, `fold_pair_events` /
//! `fold_slot_events` for dispatch, and the readback family for the
//! assert.
//!
//! CPU reference comes from the compiler-emitted Rust views
//! (`engine::generated::views::*`) — the same types the engine uses in
//! production. Feeding identical events through both sides ensures
//! the WGSL fold matches the Rust fold bit-for-bit (modulo the
//! HashMap-sparse vs dense-array layout difference, which the test
//! reconciles pair-by-pair).
//!
//! Run with `cargo test -p engine_gpu --features gpu`.

#![cfg(feature = "gpu")]

use engine::event::Event;
use engine::generated::views::{EngagedWith, KinFear, MyEnemies, PackFocus, RallyBoost, ThreatLevel};
use engine::ids::AgentId;
use engine_gpu::view_storage::{
    build_all_specs, emit::ViewShape, DecayCellCpu, FoldInputPair, FoldInputSlot, TopkReadback,
    ViewStorage,
};
use glam::Vec3;

// TEST_AGENT_CAP >= K(8) — topk views occupy at most K slots per
// observer. With K=8 and <= 6 distinct attackers in the test fixtures
// below the topk path never evicts, so CPU↔GPU parity holds per-slot
// (modulo sort order; helpers sort before comparison).
const TEST_AGENT_CAP: u32 = 16;
const TICKS_TO_SIM: u32 = 12;

/// Lazy, shared wgpu device — pollster-blocks the async setup. Keeping
/// this as a function (not a thread-local) because the test runner
/// spawns threads per test; each test builds and tears down its own
/// ViewStorage.
fn make_device() -> (wgpu::Device, wgpu::Queue, String) {
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
            .expect("request_adapter");
        let label = format!("{:?}", adapter.get_info().backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("view_parity::device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request_device");
        (device, queue, label)
    })
}

fn agent(slot: u32) -> AgentId {
    AgentId::new(slot + 1).expect("slot -> agent id")
}

// ---------------------------------------------------------------------------
// Shape-coverage test 1: slot_map — engaged_with.
//
// Fold a sequence of EngagementCommitted / EngagementBroken events on
// both sides; assert the dense GPU `array<u32>` slot map matches the
// CPU `HashMap`. Both sides encode AgentId+1 (with 0 = empty).
// ---------------------------------------------------------------------------

#[test]
fn slot_map_engaged_with_parity() {
    let (device, queue, backend_label) = make_device();
    eprintln!("view_parity: wgpu backend = {backend_label}");

    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = EngagedWith::default();

    // Scripted sequence: commit, commit (different), break first, commit again.
    //   tick 1: 0 ↔ 1   (slot)
    //   tick 2: 2 ↔ 3
    //   tick 3: break (0, 1)
    //   tick 4: 4 ↔ 5
    //   tick 5: commit 0 ↔ 6 (reuse 0)
    let events: Vec<(u32, Event)> = vec![
        (
            1,
            Event::EngagementCommitted {
                actor: agent(0),
                target: agent(1),
                tick: 1,
            },
        ),
        (
            2,
            Event::EngagementCommitted {
                actor: agent(2),
                target: agent(3),
                tick: 2,
            },
        ),
        (
            3,
            Event::EngagementBroken {
                actor: agent(0),
                former_target: agent(1),
                reason: 0,
                tick: 3,
            },
        ),
        (
            4,
            Event::EngagementCommitted {
                actor: agent(4),
                target: agent(5),
                tick: 4,
            },
        ),
        (
            5,
            Event::EngagementCommitted {
                actor: agent(0),
                target: agent(6),
                tick: 5,
            },
        ),
    ];

    // CPU fold.
    for (tick, ev) in &events {
        cpu.fold_event(ev, *tick);
    }

    // GPU fold — translate each engagement event to FoldInputSlot.
    let gpu_events: Vec<FoldInputSlot> = events
        .iter()
        .map(|(_tick, ev)| match ev {
            Event::EngagementCommitted { actor, target, .. } => FoldInputSlot {
                first: actor.raw() - 1,
                second: target.raw() - 1,
                kind: 0,
                _pad: 0,
            },
            Event::EngagementBroken {
                actor,
                former_target,
                ..
            } => FoldInputSlot {
                first: actor.raw() - 1,
                second: former_target.raw() - 1,
                kind: 1,
                _pad: 0,
            },
            _ => unreachable!(),
        })
        .collect();
    // Dispatch one event at a time so the fold is strictly ordered.
    // Atomic CAS + in-batch parallelism is non-deterministic when
    // multiple events target the same slot; the parity guarantee only
    // holds for a sequential dispatch stream (see module docstring
    // on determinism policy).
    for ev in &gpu_events {
        gpu.fold_slot_events(&device, &queue, "engaged_with", std::slice::from_ref(ev))
            .expect("fold slot");
    }

    let gpu_slots = gpu
        .readback_slot_map(&device, &queue, "engaged_with")
        .expect("readback");

    // Assert every slot matches.
    for slot in 0..TEST_AGENT_CAP {
        let id = agent(slot);
        let cpu_partner = cpu.get(id);
        let cpu_encoded = cpu_partner.map(|p| p.raw()).unwrap_or(0);
        let gpu_encoded = gpu_slots[slot as usize];
        assert_eq!(
            gpu_encoded, cpu_encoded,
            "slot_map mismatch at slot {slot}: cpu={cpu_encoded} gpu={gpu_encoded}"
        );
    }
}

// ---------------------------------------------------------------------------
// Shape-coverage test 2: per_entity_topk(K=8) non-decay — my_enemies.
//
// Task 196 replaced the dense pair_map storage with topk(K=8). Each
// observer now has up to 8 slots of (id, value, anchor). The fold
// behavior for "self += 1.0 clamp [0,1]" is: first hit adds a slot at
// 1.0; repeats are no-ops (already at clamp max); new attackers fill
// empty slots or evict smallest (ties don't evict). The GPU kernel
// serialises events (1/dispatch) so CAS ordering matches CPU.
// ---------------------------------------------------------------------------

#[test]
fn topk_my_enemies_parity() {
    let (device, queue, _label) = make_device();

    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = MyEnemies::default();

    // Several attacks including repeats (clamp to 1.0) and distinct pairs.
    let pairs: &[(u32, u32, u32)] = &[
        // (tick, attacker, observer)
        (1, 0, 1),
        (2, 0, 1), // repeat — value stays at 1.0
        (3, 2, 3),
        (4, 4, 5),
        (5, 0, 3),
        (6, 2, 1),
    ];

    // CPU.
    for (tick, attacker, observer) in pairs {
        let ev = Event::AgentAttacked {
            actor: agent(*attacker),
            target: agent(*observer),
            damage: 1.0,
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
    }

    // GPU — FoldInputPair is (first=observer, second=attacker).
    let gpu_events: Vec<FoldInputPair> = pairs
        .iter()
        .map(|(tick, attacker, observer)| FoldInputPair {
            first: *observer,
            second: *attacker,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, "my_enemies", std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_rb = gpu
        .readback_topk(&device, &queue, "my_enemies")
        .expect("readback topk");
    assert_eq!(gpu_rb.k, 8, "expected K=8 for my_enemies");

    // For each (obs, att) the CPU get() and GPU readback should agree.
    // Walk every observer; per-slot sort-by-id so fill order doesn't
    // matter, then check that the set of populated (id, value) pairs
    // matches the CPU row.
    let n = TEST_AGENT_CAP;
    for obs in 0..n {
        let (ids, vals, _anchors) = gpu_rb.row(obs);
        let mut gpu_slots: Vec<(u32, f32)> = ids
            .iter()
            .zip(vals.iter())
            .filter(|(id, _)| **id != 0)
            .map(|(id, v)| (*id, *v))
            .collect();
        gpu_slots.sort_by_key(|(id, _)| *id);

        // Build the CPU side by scanning all 0..n potential attackers
        // and picking the non-zero ones.
        let mut cpu_slots: Vec<(u32, f32)> = (0..n)
            .map(|att| (agent(att).raw(), cpu.get(agent(obs), agent(att))))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        cpu_slots.sort_by_key(|(id, _)| *id);

        assert_eq!(
            cpu_slots.len(),
            gpu_slots.len(),
            "my_enemies obs={obs} populated count mismatch: cpu={cpu_slots:?} gpu={gpu_slots:?}"
        );
        for ((cpu_id, cpu_v), (gpu_id, gpu_v)) in cpu_slots.iter().zip(gpu_slots.iter()) {
            assert_eq!(
                cpu_id, gpu_id,
                "my_enemies obs={obs} id mismatch: cpu={cpu_slots:?} gpu={gpu_slots:?}"
            );
            assert!(
                (cpu_v - gpu_v).abs() < 1e-6,
                "my_enemies obs={obs} id={cpu_id} value mismatch: cpu={cpu_v} gpu={gpu_v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shape-coverage test 3: per_entity_topk(K=8) @decay — threat_level.
//
// Task 196 — same event flow as the old pair_map test, now asserted
// against the topk readback. `get(obs, att, tick)` should agree
// byte-for-byte between CPU and GPU after decay is applied.
// ---------------------------------------------------------------------------

#[test]
fn topk_decay_threat_level_parity() {
    let (device, queue, _label) = make_device();

    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);
    let mut cpu = ThreatLevel::default();

    // Spread across ticks so decay is non-trivial. View params are
    // (a, b) — handler pattern binds `actor -> b, target -> a`, so
    // FoldInputPair.first = target, .second = actor.
    let events: &[(u32, u32, u32)] = &[
        // (tick, actor (= attacker), target)
        (1, 0, 1),
        (2, 0, 1),
        (4, 0, 1), // same cell, 3 hits spread out
        (3, 2, 3),
        (6, 2, 3),
        (5, 4, 5),
    ];

    for (tick, actor, target) in events {
        let ev = Event::AgentAttacked {
            actor: agent(*actor),
            target: agent(*target),
            damage: 1.0,
            tick: *tick,
        };
        cpu.fold_event(&ev, *tick);
    }

    let gpu_events: Vec<FoldInputPair> = events
        .iter()
        .map(|(tick, actor, target)| FoldInputPair {
            first: *target,
            second: *actor,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, "threat_level", std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_rb = gpu
        .readback_topk(&device, &queue, "threat_level")
        .expect("readback topk");
    assert_eq!(gpu_rb.k, 8, "expected K=8 for threat_level");

    // For each (a, b), compare get(a, b, tick) across query ticks.
    let n = TEST_AGENT_CAP;
    let query_ticks = [6u32, 10, 50];
    for a in 0..n {
        for b in 0..n {
            for &qt in &query_ticks {
                let cpu_val = cpu.get(agent(a), agent(b), qt);
                let gpu_val = topk_lookup_decay(
                    &gpu_rb,
                    a,
                    agent(b).raw(),
                    ThreatLevel::RATE,
                    qt,
                    0.0,
                    1000.0,
                );
                assert!(
                    (cpu_val - gpu_val).abs() < 1e-5,
                    "threat_level get mismatch at ({a},{b}) tick {qt}: cpu={cpu_val} gpu={gpu_val}"
                );
            }
        }
    }
}

/// Read the (decayed, clamped) value for attacker `attacker_raw` from
/// observer `obs_slot`'s topk row. Mirrors the generated Rust
/// `ThreatLevel::get` on a flat readback buffer.
fn topk_lookup_decay(
    rb: &TopkReadback,
    obs_slot: u32,
    attacker_raw: u32,
    rate: f32,
    tick: u32,
    lo: f32,
    hi: f32,
) -> f32 {
    let (ids, vals, anchors) = rb.row(obs_slot);
    for i in 0..ids.len() {
        if ids[i] == attacker_raw {
            let dt = tick.saturating_sub(anchors[i]);
            let decayed = vals[i] * rate.powi(dt as i32);
            return decayed.clamp(lo, hi);
        }
    }
    0.0
}

/// Apply the decay formula exactly as the CPU `get()` does — used to
/// map a readback `DecayCellCpu` to its observable value at a query
/// tick. Mirrors `threat_level::ThreatLevel::get` and the emitter's
/// WGSL formula: `base * rate^(tick - anchor)` clamped.
fn decay_get(cell: &DecayCellCpu, rate: f32, tick: u32, lo: f32, hi: f32) -> f32 {
    if cell.value == 0.0 {
        return 0.0;
    }
    let dt = tick.saturating_sub(cell.anchor);
    let decayed = cell.value * rate.powi(dt as i32);
    decayed.clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// Extended shape-coverage — the other two topk @decay views plus the
// single remaining dense pair_map @decay (kin_fear). Repeats the same
// flow as threat_level's test with its own rate / handler shape.
// ---------------------------------------------------------------------------

#[test]
fn pair_map_decay_kin_fear_parity() {
    // kin_fear is NOT migrated to topk (task 196) — FearSpread events
    // are per-death and bounded-population. Stays on dense pair_map.
    run_dense_decay_parity_scenario(
        "kin_fear",
        KinFear::RATE,
        0.0,
        10.0,
        |tick, first, second| Event::FearSpread {
            observer: agent(first),
            dead_kin: agent(second),
            tick,
        },
        |cpu, ev, tick| {
            let mut m = cpu;
            m.fold_event(&ev, tick);
            m
        },
        KinFear::default(),
        |m, a, b, t| m.get(agent(a), agent(b), t),
    );
}

#[test]
fn topk_decay_pack_focus_parity() {
    run_topk_decay_parity_scenario(
        "pack_focus",
        PackFocus::RATE,
        0.0,
        10.0,
        |tick, first, second| Event::PackAssist {
            observer: agent(first),
            target: agent(second),
            tick,
        },
        |cpu, ev, tick| {
            let mut m = cpu;
            m.fold_event(&ev, tick);
            m
        },
        PackFocus::default(),
        |m, a, b, t| m.get(agent(a), agent(b), t),
    );
}

#[test]
fn topk_decay_rally_boost_parity() {
    run_topk_decay_parity_scenario(
        "rally_boost",
        RallyBoost::RATE,
        0.0,
        10.0,
        |tick, first, second| Event::RallyCall {
            observer: agent(first),
            wounded_kin: agent(second),
            tick,
        },
        |cpu, ev, tick| {
            let mut m = cpu;
            m.fold_event(&ev, tick);
            m
        },
        RallyBoost::default(),
        |m, a, b, t| m.get(agent(a), agent(b), t),
    );
}

/// Topk @decay parity scaffold. Folds the same events on both sides
/// and compares `get(a, b, tick)` across a few query ticks. The CPU
/// type already applies the decay math; for GPU we reconstruct via
/// `topk_lookup_decay` from the readback buffer.
fn run_topk_decay_parity_scenario<M>(
    view_name: &str,
    rate: f32,
    clamp_lo: f32,
    clamp_hi: f32,
    make_event: impl Fn(u32, u32, u32) -> Event,
    mut fold: impl FnMut(M, Event, u32) -> M,
    initial: M,
    get: impl Fn(&M, u32, u32, u32) -> f32,
) {
    let (device, queue, _label) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);

    let events: &[(u32, u32, u32)] = &[
        (1, 0, 1),
        (2, 0, 1),
        (3, 2, 3),
        (5, 0, 1), // 3rd hit, later tick → different decay
        (4, 4, 5),
        (6, 2, 3),
    ];

    let mut cpu = initial;
    for (tick, first, second) in events {
        let ev = make_event(*tick, *first, *second);
        cpu = fold(cpu, ev, *tick);
    }

    let gpu_events: Vec<FoldInputPair> = events
        .iter()
        .map(|(tick, first, second)| FoldInputPair {
            first: *first,
            second: *second,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, view_name, std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_rb = gpu
        .readback_topk(&device, &queue, view_name)
        .expect("readback topk");

    let n = TEST_AGENT_CAP;
    let query_ticks = [6u32, 10, 30];
    for a in 0..n {
        for b in 0..n {
            for &qt in &query_ticks {
                let cpu_val = get(&cpu, a, b, qt);
                let gpu_val = topk_lookup_decay(
                    &gpu_rb,
                    a,
                    agent(b).raw(),
                    rate,
                    qt,
                    clamp_lo,
                    clamp_hi,
                );
                assert!(
                    (cpu_val - gpu_val).abs() < 1e-5,
                    "{view_name} get mismatch at ({a},{b}) tick {qt}: cpu={cpu_val} gpu={gpu_val}"
                );
            }
        }
    }
    let _ = Vec3::ZERO;
}

/// Dense pair_map @decay parity scaffold — kin_fear still lives on
/// this path (task 196 kept it dense).
fn run_dense_decay_parity_scenario<M>(
    view_name: &str,
    rate: f32,
    clamp_lo: f32,
    clamp_hi: f32,
    make_event: impl Fn(u32, u32, u32) -> Event,
    mut fold: impl FnMut(M, Event, u32) -> M,
    initial: M,
    get: impl Fn(&M, u32, u32, u32) -> f32,
) {
    let (device, queue, _label) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);

    let events: &[(u32, u32, u32)] = &[
        (1, 0, 1),
        (2, 0, 1),
        (3, 2, 3),
        (5, 0, 1),
        (4, 4, 5),
        (6, 2, 3),
    ];

    let mut cpu = initial;
    for (tick, first, second) in events {
        let ev = make_event(*tick, *first, *second);
        cpu = fold(cpu, ev, *tick);
    }

    let gpu_events: Vec<FoldInputPair> = events
        .iter()
        .map(|(tick, first, second)| FoldInputPair {
            first: *first,
            second: *second,
            tick: *tick,
            _pad: 0,
        })
        .collect();
    for ev in &gpu_events {
        gpu.fold_pair_events(&device, &queue, view_name, std::slice::from_ref(ev))
            .expect("fold pair");
    }

    let gpu_cells = gpu
        .readback_pair_decay(&device, &queue, view_name)
        .expect("readback decay");

    let n = TEST_AGENT_CAP;
    let query_ticks = [6u32, 10, 30];
    for a in 0..n {
        for b in 0..n {
            let cpu_v6 = get(&cpu, a, b, 6);
            let cpu_populated = !(cpu_v6 == 0.0 && get(&cpu, a, b, 0) == 0.0);
            let gpu_cell = gpu_cells[(a * n + b) as usize];
            if !cpu_populated {
                assert_eq!(gpu_cell.value, 0.0, "{view_name} stale cell ({a},{b}) value");
                assert_eq!(gpu_cell.anchor, 0, "{view_name} stale cell ({a},{b}) anchor");
                continue;
            }
            for &qt in &query_ticks {
                let cpu_val = get(&cpu, a, b, qt);
                let gpu_val = decay_get(&gpu_cell, rate, qt, clamp_lo, clamp_hi);
                assert!(
                    (cpu_val - gpu_val).abs() < 1e-5,
                    "{view_name} get mismatch at ({a},{b}) tick {qt}: cpu={cpu_val} gpu={gpu_val}"
                );
            }
        }
    }
    let _ = Vec3::ZERO;
}

// ---------------------------------------------------------------------------
// Storage-layout smoke check — make sure every materialized view is
// visible through `build_all_specs()`, which the ViewStorage
// constructor iterates. A new materialized view added to views.sim
// without being mirrored into engine_gpu's IR shim (see
// `build_materialized_view_irs`) would fail this early.
// ---------------------------------------------------------------------------

#[test]
fn ships_every_materialized_view_shape() {
    let specs = build_all_specs();
    let names: Vec<_> = specs.iter().map(|s| s.view_name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "engaged_with",
            "my_enemies",
            "threat_level",
            "kin_fear",
            "pack_focus",
            "rally_boost",
        ],
        "materialized view list drift — update engine_gpu::view_storage::build_materialized_view_irs"
    );
    // Shape coverage: one slot_map, two no-decay or decay-free, four @decay.
    let slot_count = specs
        .iter()
        .filter(|s| matches!(s.shape, ViewShape::SlotMap { .. }))
        .count();
    let pair_scalar_count = specs
        .iter()
        .filter(|s| matches!(s.shape, ViewShape::PairMapScalar))
        .count();
    let pair_decay_count = specs
        .iter()
        .filter(|s| matches!(s.shape, ViewShape::PairMapDecay { .. }))
        .count();
    assert_eq!(slot_count, 1, "expect 1 slot_map (engaged_with)");
    assert_eq!(
        pair_scalar_count, 1,
        "expect 1 pair_map non-decay (my_enemies)"
    );
    assert_eq!(
        pair_decay_count, 4,
        "expect 4 pair_map @decay (threat_level, kin_fear, pack_focus, rally_boost)"
    );
    let _ = TICKS_TO_SIM;
}

// ---------------------------------------------------------------------------
// Byte-equal readback for the topk storage — compare raw (id, value,
// anchor) tuples after a single fold. With task 196's topk layout the
// buffer is N·K entries (16·8 = 128 slots) rather than N² = 256.
// ---------------------------------------------------------------------------

#[test]
fn topk_byte_equal_readback() {
    let (device, queue, _label) = make_device();
    let mut gpu = ViewStorage::new(&device, TEST_AGENT_CAP).expect("ViewStorage::new");
    gpu.reset(&queue);

    // Empty read after reset — every cell must be 0x00000000.
    let rb = gpu
        .readback_topk(&device, &queue, "my_enemies")
        .expect("readback after reset");
    assert_eq!(rb.ids.len(), (TEST_AGENT_CAP * rb.k as u32) as usize);
    assert!(rb.ids.iter().all(|&id| id == 0), "ids must be zero after reset");
    assert!(
        rb.values.iter().all(|&v| v.to_bits() == 0),
        "values must be zero bits after reset"
    );
    assert!(rb.anchors.iter().all(|&a| a == 0), "anchors zero after reset");

    // Single fold — observer slot 2, attacker slot 3. The topk fold
    // places the pair in row[0] (first empty slot): id=4 (raw), v=1.0,
    // anchor=1.
    gpu.fold_pair_events(
        &device,
        &queue,
        "my_enemies",
        &[FoldInputPair {
            first: 2,
            second: 3,
            tick: 1,
            _pad: 0,
        }],
    )
    .expect("fold");
    let rb = gpu
        .readback_topk(&device, &queue, "my_enemies")
        .expect("readback after fold");
    let (ids, values, anchors) = rb.row(2);
    let attacker_raw = agent(3).raw();
    assert_eq!(ids[0], attacker_raw, "expected attacker raw id in slot 0");
    assert_eq!(
        values[0].to_bits(),
        1.0_f32.to_bits(),
        "expected 1.0 value at slot 0; got {:?}",
        values[0]
    );
    assert_eq!(anchors[0], 1, "expected anchor = 1");
    // Other slots in this row stay zero.
    for i in 1..ids.len() {
        assert_eq!(ids[i], 0, "stray id in slot {i}");
        assert_eq!(values[i].to_bits(), 0, "stray value in slot {i}");
        assert_eq!(anchors[i], 0, "stray anchor in slot {i}");
    }
    // Other observers' rows must stay clear.
    for obs in 0..TEST_AGENT_CAP {
        if obs == 2 {
            continue;
        }
        let (ids, vals, ans) = rb.row(obs);
        assert!(ids.iter().all(|&id| id == 0), "row {obs} has stray id");
        assert!(vals.iter().all(|&v| v.to_bits() == 0), "row {obs} has stray value");
        assert!(ans.iter().all(|&a| a == 0), "row {obs} has stray anchor");
    }
}
